use crate::{
    engine_client::EngineClient,
    queue::TaskAssigner,
    ScheduleContext,
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[allow(dead_code)] // referenced by start_vllm_colocation_event_loop's return; fields not currently consumed
pub(crate) struct ColocationController {
    pub all_schedule_contexts: Vec<Arc<Mutex<ScheduleContext>>>,
    pub batching_queue: TaskAssigner,
}

impl ColocationController {}

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)] // BackendFault carries error context for future surfacing
pub(crate) enum ExtExcept {
    #[error("frontend aborted: client dropped the response channel")]
    FrontendAbort,

    #[error("backend error: {0}")]
    BackendFault(String),
}

pub(crate) fn start_vllm_colocation_event_loop(
    queue: TaskAssigner,
    all_engine_clients: Vec<Box<dyn EngineClient>>,
    all_schedule_contexts: Vec<Arc<Mutex<ScheduleContext>>>,
    shared_tokenizer: Option<Arc<tokenizers::Tokenizer>>,
) -> Arc<ColocationController> {
    for (replica_index, engine_client) in all_engine_clients.into_iter().enumerate() {
        tokio::spawn(task_assignment::work_event_loop(
            replica_index,
            engine_client,
            queue.clone(),
            all_schedule_contexts[replica_index].clone(),
            shared_tokenizer.clone(),
        ));
    }

    let controller = ColocationController { batching_queue: queue, all_schedule_contexts };
    Arc::new(controller)
}

mod task_assignment {
    use nohash_hasher::{BuildNoHashHasher, IntMap};
    use pb::generate::v2 as proto;
    use tokio::sync::mpsc::{self, channel, error::SendError, Receiver};
    use tokio::sync::Mutex;
    use tokio::task::yield_now;
    use tokio::time::Duration;

    use core::panic;
    use std::sync::Arc;

    use super::except_management::{ExtContext, ExtState};
    use super::ScheduleContext;
    use crate::{
        engine_client::{EngineClient, EngineStepReceiver, RequestStepOutput},
        infer::{InferError, InferStreamResponse},
        kvcache::BlockHash,
        queue::{Entry, QueuePro},
        ExtExcept, LMetricDec, Token,
    };

    /// Tracks the lifecycle phase of each request within the event loop.
    ///
    /// # State Transition Diagram
    ///
    /// ```text
    ///   (request arrives via WQ channel)
    ///           |
    ///           v
    ///       Waiting
    ///           |
    ///           |--[SSE state="PREFILL"]--> Prefilling
    ///           |
    ///           v
    ///       Prefilling
    ///           |
    ///           |--[SSE state="DECODE"]--> Decoding
    ///           |--[is_finished=true]--> Finished
    ///           |
    ///           v
    ///       Decoding
    ///           |
    ///           |--[is_finished=true]--> Finished
    ///           |
    ///           v
    ///       Finished (removed from tracking)
    /// ```
    ///
    /// Any phase may also transition to `Excepted` if the request enters
    /// exception handling (frontend abort or backend fault).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum RequestPhase {
        /// Request has been submitted but no SSE status received yet.
        Waiting,
        /// First SSE event reported state="PREFILL".
        Prefilling,
        /// SSE event reported state="DECODE".
        Decoding,
        /// Request moved to exception handling (abort or fault).
        Excepted,
    }

    impl RequestPhase {
        /// Validate and apply a phase transition based on SSE-reported state.
        /// Returns the new phase. Panics on invalid transitions in debug builds.
        fn transition(self, sse_state: &str, is_finished: bool) -> RequestPhase {
            match (self, sse_state) {
                // Waiting -> Prefilling (first SSE report)
                (RequestPhase::Waiting, "PREFILL") => {
                    if is_finished { RequestPhase::Prefilling } else { RequestPhase::Prefilling }
                }
                // Prefilling -> Prefilling (continued prefill, e.g. chunked prefill)
                (RequestPhase::Prefilling, "PREFILL") => RequestPhase::Prefilling,
                // Waiting -> Decoding (prefill was so fast it was not observed)
                (RequestPhase::Waiting, "DECODE") => {
                    tracing::debug!(
                        "Request skipped Prefilling phase (prefill not observed in SSE)"
                    );
                    RequestPhase::Decoding
                }
                // Prefilling -> Decoding (normal transition)
                (RequestPhase::Prefilling, "DECODE") => RequestPhase::Decoding,
                // Decoding -> Decoding (continued decode)
                (RequestPhase::Decoding, "DECODE") => RequestPhase::Decoding,
                // Invalid: Decoding -> Prefilling (should never happen)
                (RequestPhase::Decoding, "PREFILL") => {
                    debug_assert!(
                        false,
                        "Invalid phase transition: Decoding -> Prefilling"
                    );
                    // In release builds, tolerate gracefully
                    RequestPhase::Prefilling
                }
                // Excepted requests should not receive phase transitions
                (RequestPhase::Excepted, _) => {
                    debug_assert!(
                        false,
                        "Phase transition on Excepted request (state={})",
                        sse_state
                    );
                    RequestPhase::Excepted
                }
                (_, unknown) => {
                    panic!("Unknown SSE state in phase transition: {}", unknown);
                }
            }
        }
    }

    /// Control the workload level of delegated instance.
    ///
    /// The `EngineClient` is split: `take_step_receiver()` yields a receiver
    /// for the completion loop, while this function keeps the client for
    /// sending requests. No shared mutex needed — eliminates the deadlock
    /// where `completion_event_loop` held the client lock in `recv_step()`
    /// while `work_event_loop` waited to acquire it for `add_request()`.
    pub async fn work_event_loop<Q: QueuePro>(
        replica_index: usize,
        mut engine_client: Box<dyn EngineClient>,
        queue: Q,
        schedule_context: Arc<Mutex<ScheduleContext>>,
        shared_tokenizer: Option<Arc<tokenizers::Tokenizer>>,
    ) {
        // Channel: work queue -> completion queue (only carries Entry now)
        let (wq_tx, cq_wqe_rx) = channel::<Entry>(64);
        let wqe_mtx = Arc::new(Mutex::new(()));

        // Extract error receiver before splitting
        let cq_error_rx = engine_client.get_error_rx();

        // Split: step receiver goes to completion loop, client stays here
        let step_receiver = engine_client.take_step_receiver();

        // Init completion queue poller — owns the step receiver exclusively
        tokio::spawn(completion_event_loop(
            replica_index,
            step_receiver,
            wqe_mtx.clone(),
            cq_wqe_rx,
            cq_error_rx,
            schedule_context,
            shared_tokenizer,
        ));

        // Work loop: owns engine_client exclusively, no lock needed
        loop {
            match wq_tx.reserve().await {
                Ok(permit) => {
                    if let Some((id, entry)) = queue.next_request(replica_index).await {
                        let _g = wqe_mtx.lock().await;
                        if let Err(e) = engine_client.add_request(id, &entry.request).await {
                            tracing::error!(
                                request_id = id,
                                engine = replica_index,
                                error = %e,
                                "ADD_REQUEST_FAILED"
                            );
                        }
                        tracing::info!(
                            target: "lifecycle",
                            request_id = id,
                            queue_time_us = entry.batch_time.unwrap().duration_since(entry.queue_time).as_micros() as u64,
                            input_length = entry.request.input_length,
                            max_new_tokens = entry.request.stopping_parameters.max_new_tokens,
                            engine = replica_index,
                            "REQUEST_ADMIT"
                        );
                        let _ = permit.send(entry);
                    } else {
                        yield_now().await;
                    }
                }
                Err(SendError(_)) => {
                    unreachable!("engine#{replica_index} WQ channel closed by receiver at CQ");
                }
            }
        }
    }

    /// First phase in abortion 2PC.
    /// Moves aborted requests from `entries` into `except_context` and marks
    /// their lifecycle phase as `Excepted`.
    fn on_frontend_abort(
        entries: &mut IntMap<u64, Entry>,
        cancel_req_ids: &mut Vec<u64>,
        skip_entries: &mut (Vec<u64>, Vec<Entry>),
        except_context: &mut ExtContext,
        request_phases: &mut IntMap<u64, RequestPhase>,
    ) {
        while let Some(id) = cancel_req_ids.pop() {
            tracing::warn!(target: "lifecycle", request_id = id, "REQUEST_CANCELLING");
            skip_entries.0.push(id);
            let entry = entries
                .remove(&id)
                .expect(format!("Request_{} not found in entries. This is a bug.", id).as_str());
            // Mark phase as excepted before moving to exception context
            if let Some(phase) = request_phases.get_mut(&id) {
                *phase = RequestPhase::Excepted;
            }
            except_context.put(ExtState::Abort(id, entry));
        }
    }

    /// Handle any request error at backend.
    ///
    /// First phase in fault 2PC. Moves faulted requests from `entries` into
    /// `except_context` and marks their lifecycle phase as `Excepted`.
    async fn on_backend_fault(
        entries: &mut IntMap<u64, Entry>,
        error_rx: &mut mpsc::UnboundedReceiver<u64>,
        skip_entries: &(Vec<u64>, Vec<Entry>),
        except_context: &mut ExtContext,
        request_phases: &mut IntMap<u64, RequestPhase>,
    ) {
        while let Ok(id) = error_rx.try_recv() {
            if skip_entries.0.contains(&id) {
                continue;
            }
            tracing::info!(
                target: "lifecycle",
                request_id = id,
                "BACKEND_FAULT"
            );
            // NOTE: tolerant the error request occurs in the same step output, but no more
            let entry = entries
                .remove(&id)
                .expect(format!("Request_{} not found in entries. This is a bug.", id).as_str());
            // NOTE: skip possible `SendError`, backend resource has been freed
            let _ = entry
                .response_tx
                .send(Err(InferError::GenerationError("Engine refused to serve!".to_string())));
            // Mark phase as excepted before moving to exception context
            if let Some(phase) = request_phases.get_mut(&id) {
                *phase = RequestPhase::Excepted;
            }
            except_context.put(ExtState::Fault(id, entry));
        }
    }

    /// # Precondition:
    ///   + `entries.lock()`
    async fn on_prefill(
        replica_index: usize,
        entry: &mut Entry,
        request_id: u64,
        hit_token_cnt: u64,
        new_token_ids: &Vec<u32>,
        shared_tokenizer: &Option<Arc<tokenizers::Tokenizer>>,
    ) -> Result<(), ExtExcept> {
        tracing::info!(
            target: "lifecycle",
            engine = replica_index,
            request_id = request_id,
            hit_token_cnt = hit_token_cnt,
            "PREFILL_DONE"
        );
        entry
            .response_tx
            .send(Ok(InferStreamResponse::Prefill(proto::Tokens::default())))
            .map_err(|_| ExtExcept::FrontendAbort)?;
        for &t in new_token_ids {
            entry
                .response_tx
                .send(Ok(InferStreamResponse::Intermediate {
                    token: Token {
                        id: t,
                        text: shared_tokenizer.as_ref()
                            .and_then(|tok| tok.decode(&[t], false).ok())
                            .unwrap_or_default(),
                        logprob: 0.0,
                        special: false,
                    },
                    top_tokens: Vec::default(),
                }))
                .map_err(|_| ExtExcept::FrontendAbort)?
        }

        Ok(())
    }

    /// # Precondition:
    ///   + `ctx` is clean, no lock is held
    async fn on_decode(
        replica_index: usize,
        entry: &mut Entry,
        request_id: u64,
        new_token_ids: &Vec<u32>,
        shared_tokenizer: &Option<Arc<tokenizers::Tokenizer>>,
    ) -> Result<(), ExtExcept> {
        tracing::trace!(engine = replica_index, request_id = request_id, "DECODE_STEP");

        for &t in new_token_ids {
            entry
                .response_tx
                .send(Ok(InferStreamResponse::Intermediate {
                    token: Token {
                        id: t,
                        text: shared_tokenizer.as_ref()
                            .and_then(|tok| tok.decode(&[t], false).ok())
                            .unwrap_or_default(),
                        logprob: 0.0,
                        special: false,
                    },
                    top_tokens: Vec::default(),
                }))
                .map_err(|_| ExtExcept::FrontendAbort)?;
        }

        Ok(())
    }

    /// # Precondition:
    ///   + `ctx` is clean, no lock is held
    ///
    /// With the trait-based EngineClient, completion is detected via
    /// `recv_step()` reporting `is_finished=true`. No HTTP response
    /// JoinHandle is needed.
    fn on_finish_request(
        replica_index: usize,
        entry: Entry,
        request_id: u64,
    ) {
        let id = request_id;
        tracing::info!(
            target: "lifecycle",
            engine = replica_index,
            request_id = id,
            generated_tokens = entry.generated_token_cnt,
            "REQUEST_FINISH"
        );
        let _skip = entry.response_tx.send(Ok(InferStreamResponse::End {
            token: Token::default(),
            top_tokens: Vec::default(),
            generated_text: proto::GeneratedText::default(),
            start: entry.batch_time.unwrap(),
            queued: entry.queue_time,
            max_time_between_tokens: entry.max_time_between_tokens,
        }));
        // NOTE: skip possible `SendError`, backend resource has been freed
    }

    /// Poll backend engine's step outputs and enable callbacks.
    ///
    /// This is the core event loop that processes `EngineStepOutput` from
    /// any backend (HTTP+SSE or ZMQ) via the `EngineClient` trait.
    async fn completion_event_loop(
        replica_index: usize,
        mut step_receiver: Box<dyn EngineStepReceiver>,
        wqe_mtx: Arc<Mutex<()>>,
        mut cq_wqe_rx: Receiver<Entry>,
        mut cq_error_rx: mpsc::UnboundedReceiver<u64>,
        schedule_context: Arc<Mutex<ScheduleContext>>,
        shared_tokenizer: Option<Arc<tokenizers::Tokenizer>>,
    ) {
        let mut entries =
            IntMap::with_capacity_and_hasher(256, BuildNoHashHasher::<u64>::default());
        // Request lifecycle phase tracking (debug invariant enforcement)
        let mut request_phases: IntMap<u64, RequestPhase> =
            IntMap::with_capacity_and_hasher(256, BuildNoHashHasher::<u64>::default());

        // Handle frontend abortions and backend errors
        let mut cancel_req_ids = Vec::new();
        let mut except_context = ExtContext::new();
        let mut temp_leaving_entries = (Vec::with_capacity(8), Vec::with_capacity(8));
        let mut skipped_fault_entries = (Vec::with_capacity(8), Vec::with_capacity(8));

        // Main event loop: receive step outputs — no shared lock, owns receiver exclusively
        loop {
            let step_result = step_receiver.recv_step().await;

            let m = match step_result {
                Ok(step) => step,
                Err(e) => {
                    tracing::error!(engine = replica_index, error = %e, "RECV_STEP_ERROR");
                    // For stream-ended errors, break out of the loop
                    if matches!(e, crate::engine_client::EngineClientError::StreamEnded) {
                        break;
                    }
                    continue;
                }
            };

            // Adds newly posted requests
            {
                let _g = wqe_mtx.lock().await;
                while let Ok(entry) = cq_wqe_rx.try_recv() {
                    let rid = entry.request.request_id;
                    debug_assert!(
                        !entries.contains_key(&rid),
                        "Request_{rid} already in entries on insertion"
                    );
                    debug_assert!(
                        !except_context.entries.contains_key(&rid),
                        "Request_{rid} in except_context when being inserted into entries"
                    );
                    entries.insert(rid, entry);
                    request_phases.insert(rid, RequestPhase::Waiting);
                }
            }
            // postcond: all requests in step output are visible to CQ

            tracing::trace!(engine = replica_index, step = ?m, "STEP_RECEIVED");

            let current_epoch = schedule_context.lock().await.block_hash.epoch();

            if !m.preempted_ids.is_empty() {
                m.preempted_ids.iter().for_each(|&id| {
                    tracing::warn!(target: "lifecycle", request_id = id, "REQUEST_PREEMPTED");
                });
            }

            // fast path: update metrics
            let tbt = Duration::from_millis(m.latency);
            let mut metric_delta = LMetricDec::new(&tbt);
            // NOTE: `prefill_tokens` doesn't count hit tokens, while
            //       `all_tokens` does count hit tokens
            metric_delta.prefill_tokens_dec = m.prefill_tokens as isize;
            for request_status in &m.outputs {
                let RequestStepOutput {
                    request_id,
                    new_token_ids: ref new_tokens,
                    ref state,
                    is_finished,
                    hit_token_cnt,
                } = *request_status;
                match state.as_str() {
                    "PREFILL" => {
                        // Only count waiting_reqs_dec if we actually know this request
                        let is_known = entries.contains_key(&request_id)
                            || except_context.entries.contains_key(&request_id);
                        if is_known {
                            metric_delta.waiting_reqs_dec += 1;
                        }
                        if is_finished {
                            if let Some(entry) = entries.get_mut(&request_id) {
                                let input_length = entry.request.input_length as isize;
                                metric_delta.bs_dec += 1;
                                metric_delta.all_tokens_inc -= input_length as isize;
                            } else if let Some(entry) =
                                except_context
                                    .put(ExtState::Exit(request_id))
                            {
                                let input_length = entry.request.input_length as isize;
                                metric_delta.bs_dec += 1;
                                metric_delta.all_tokens_inc -= input_length as isize;
                                // NOTE:
                                temp_leaving_entries.0.push(request_id);
                                temp_leaving_entries.1.push(entry);
                            } else if !is_known {
                                tracing::warn!(
                                    target: "lifecycle",
                                    request_id = request_id,
                                    "STALE_PREFILL_FINISHED"
                                );
                            }
                        } else {
                            metric_delta.all_tokens_inc += new_tokens.len() as isize;
                            let entry_opt = entries.get_mut(&request_id)
                                .or_else(|| except_context.entries.get_mut(&request_id));
                            if let Some(entry) = entry_opt {
                                let inc_hit_nblks = entry
                                .block_hash_state
                                .set_real_token_hits_get_diff(hit_token_cnt);
                            let bs = entry.block_hash_state.get_block_size();
                            tracing::info!(
                                target: "correction",
                                request_id = request_id,
                                engine = replica_index,
                                predicted = entry.block_hash_state.pred_hit_tokens(),
                                actual = hit_token_cnt,
                                diff = inc_hit_nblks * bs as isize,
                                decision_epoch = entry.block_hash_state.decision_epoch(),
                                current_epoch = current_epoch,
                                "CORRECTION"
                            );
                            metric_delta.prefill_tokens_dec += inc_hit_nblks
                                * entry.block_hash_state.get_block_size() as isize;
                            entry.append_state(new_tokens, &tbt);
                            } else {
                                tracing::warn!(
                                    target: "lifecycle",
                                    request_id = request_id,
                                    "STALE_PREFILL_UPDATE"
                                );
                                // Undo the metric delta we already applied above
                                metric_delta.all_tokens_inc -= new_tokens.len() as isize;
                            }
                        }
                    }
                    "DECODE" => {
                        if is_finished {
                            if let Some(entry) = entries.get_mut(&request_id) {
                                let request = &entry.request;
                                metric_delta.bs_dec += 1;
                                // NOTE: `generated_token_cnt` has not been appended, so just make decrement
                                metric_delta.all_tokens_inc -= request.input_length
                                    as isize
                                    + entry.generated_token_cnt as isize;
                            } else if let Some(entry) =
                                // `unwrap` inside, `entry` must be either in `entries` or `except_context`
                                except_context
                                    .put(ExtState::Exit(request_id))
                            {
                                let request = &entry.request;
                                metric_delta.bs_dec += 1;
                                // NOTE: `generated_token_cnt` has not been appended, so just make decrement
                                metric_delta.all_tokens_inc -= request.input_length
                                    as isize
                                    + entry.generated_token_cnt as isize;
                                // NOTE:
                                temp_leaving_entries.0.push(request_id);
                                temp_leaving_entries.1.push(entry);
                            } else {
                                tracing::warn!(
                                    target: "lifecycle",
                                    request_id = request_id,
                                    "STALE_DECODE_FINISHED"
                                );
                            }
                        } else {
                            if let Some(entry) = entries.get_mut(&request_id) {
                                entry.append_state(new_tokens, &tbt);
                                // postcond: `Some(entry.tpot)`
                                metric_delta.all_tokens_inc += new_tokens.len() as isize;
                                metric_delta.tpot +=
                                    entry.time_of_per_token.unwrap().as_secs_f32();
                            } else if let Some(entry) = except_context.entries.get_mut(&request_id) {
                                entry.append_state(new_tokens, &tbt);
                                metric_delta.all_tokens_inc += new_tokens.len() as isize;
                                metric_delta.tpot +=
                                    entry.time_of_per_token.unwrap().as_secs_f32();
                            } else {
                                // Stale SSE event for unknown request (e.g., from a
                                // previous router session). Skip gracefully.
                                tracing::warn!(
                                    target: "lifecycle",
                                    request_id = request_id,
                                    "STALE_DECODE_UPDATE"
                                );
                            }
                        }
                    }
                    _ => {
                        eprintln!("Request_{request_id} invalid state: {state}!");
                        panic!("engine#{replica_index}::step erroneous output {:?}", m);
                    }
                }
            }
            let mut sctx = schedule_context.lock().await;

            // Update PrefixBlockHash
            let epoch_before = sctx.block_hash.epoch();
            if !m.evicted_block_ids.is_empty() {
                tracing::info!(
                    target: "cache_tracking",
                    engine = replica_index,
                    evicted_blocks = m.evicted_block_ids.len(),
                    "EVICTION"
                );
            }
            sctx.block_hash.remove(m.evicted_block_ids.clone());
            for (rid, block_indices) in &m.cur_used_block_ids {
                if block_indices.is_empty() {
                    continue;
                }
                // Total backend bids
                let entry = entries.get_mut(rid).unwrap_or_else(|| {
                    except_context.entries.get_mut(rid).unwrap_or_else(|| {
                        let i = temp_leaving_entries
                            .0
                            .iter()
                            .position(|&eid| *rid == eid)
                            .unwrap();
                        temp_leaving_entries.1.get_mut(i).unwrap()
                    })
                });
                tracing::debug!(
                    target: "cache_tracking",
                    request_id = *rid,
                    block_indices = ?block_indices,
                    "BACKEND_BIDS_UPDATE"
                );
                entry.block_hash_state.set_bids(block_indices.clone());
            }
            let mut total_inserted: usize = 0;
            for (rid, block_indices) in &m.new_block_hashes_ids {
                if block_indices.is_empty() {
                    continue;
                }
                let entry = entries.get_mut(rid).unwrap_or_else(|| {
                    except_context.entries.get_mut(rid).unwrap_or_else(|| {
                        let i = temp_leaving_entries
                            .0
                            .iter()
                            .position(|&eid| *rid == eid)
                            .unwrap();
                        temp_leaving_entries.1.get_mut(i).unwrap()
                    })
                });
                match entry.block_hash_state.get_onto_hashes(block_indices) {
                    Ok(onto_hashes) => {
                        tracing::debug!(
                            target: "cache_tracking",
                            request_id = *rid,
                            onto_hashes = ?onto_hashes,
                            block_indices = ?block_indices,
                            "BLOCK_INSERT"
                        );
                        let n = sctx.block_hash.insert(onto_hashes, block_indices.clone());
                        total_inserted += n;
                    }
                    Err(backend_bids) => {
                        let err_msg = format!("Entry_{rid} inconsistent bid: frontend marking occupied {:?} | backend newly committed {:?}", backend_bids, block_indices);
                        // tracing::error!(err_msg);
                        panic!("{err_msg}");
                    }
                }
            }
            let epoch_after = sctx.block_hash.epoch();
            tracing::info!(
                target: "cache_tracking",
                engine = replica_index,
                step_id = m.step_id,
                evicted = m.evicted_block_ids.len(),
                inserted = total_inserted,
                epoch_before = epoch_before,
                epoch_after = epoch_after,
                "SSE_EVENT"
            );
            // Publishes updated instance-level metric state
            sctx.lmetric -= metric_delta;
            drop(sctx);

            // Aborted requests ACK-ed by backend
            let term_requests = m.aborted_requests;

            // slow path: pass generations
            for request_status in m.outputs {
                let RequestStepOutput {
                    request_id,
                    new_token_ids,
                    state,
                    is_finished,
                    hit_token_cnt,
                } = request_status;

                // Update request lifecycle phase tracking
                if let Some(phase) = request_phases.get_mut(&request_id) {
                    let new_phase = phase.transition(state.as_str(), is_finished);
                    *phase = new_phase;
                }

                match state.as_str() {
                    "PREFILL" if entries.get(&request_id).is_some() => {
                        let entry = entries.get_mut(&request_id).unwrap();
                        if let Err(ExtExcept::FrontendAbort) = on_prefill(
                            replica_index,
                            entry,
                            request_id,
                            hit_token_cnt,
                            &new_token_ids,
                            &shared_tokenizer,
                        )
                        .await
                        {
                            cancel_req_ids.push(request_id);
                        }
                        if is_finished {
                            let entry = entries.remove(&request_id).unwrap();
                            request_phases.remove(&request_id);
                            on_finish_request(
                                replica_index,
                                entry,
                                request_id,
                            );
                            // NOTE: revokes FrontendAbort, since both channels are terminated
                            if request_id
                                == cancel_req_ids.last().copied().unwrap_or(!request_id)
                            {
                                cancel_req_ids.pop();
                            }
                        }
                    }
                    "DECODE" if entries.get(&request_id).is_some() => {
                        let entry = entries.get_mut(&request_id).unwrap();
                        if let Err(ExtExcept::FrontendAbort) =
                            on_decode(replica_index, entry, request_id, &new_token_ids, &shared_tokenizer)
                                .await
                        {
                            cancel_req_ids.push(request_id);
                        }
                        if is_finished {
                            let entry = entries.remove(&request_id).unwrap();
                            request_phases.remove(&request_id);
                            on_finish_request(
                                replica_index,
                                entry,
                                request_id,
                            );
                            // NOTE: revokes FrontendAbort, since both channels are terminated
                            if request_id
                                == cancel_req_ids.last().copied().unwrap_or(!request_id)
                            {
                                cancel_req_ids.pop();
                            }
                        }
                    }
                    "PREFILL" | "DECODE" => {
                        debug_assert!(
                            except_context.entries.contains_key(&request_id)
                                || temp_leaving_entries.0.contains(&request_id),
                            "Request_{request_id} is missing!"
                        );
                    }
                    _ => {
                        eprintln!("Request_{request_id} invalid state: {state}!");
                        unreachable!();
                    }
                }
            }

            // Handle request exceptions.
            //
            // Newly canceled requests move to 1st commit phase
            on_frontend_abort(
                &mut entries,
                &mut cancel_req_ids,
                &mut skipped_fault_entries,
                &mut except_context,
                &mut request_phases,
            );
            on_backend_fault(
                &mut entries,
                &mut cq_error_rx,
                &skipped_fault_entries,
                &mut except_context,
                &mut request_phases,
            )
            .await;
            // Requests to be marked as terminated with backend abort ACK-ed
            let mut bs_dec = 0;
            let mut all_tokens_inc = 0;
            term_requests
                .into_iter()
                .map(|term_id| {
                    request_phases.remove(&term_id);
                    except_context.put(ExtState::Term(term_id))
                })
                .filter_map(|x| x)
                .for_each(|entry| {
                    bs_dec += 1;
                    all_tokens_inc -=
                        entry.request.input_length as isize + entry.generated_token_cnt as isize;
                });
            except_context.filter_drop().into_iter().for_each(|(id, entry)| {
                request_phases.remove(&id);
                bs_dec += 1;
                all_tokens_inc -=
                    entry.request.input_length as isize + entry.generated_token_cnt as isize;
            });
            if bs_dec != 0 {
                let mut sctx = schedule_context.lock().await;
                sctx.lmetric.bs = sctx.lmetric.bs.saturating_sub(bs_dec);
                sctx.lmetric.all_tokens =
                    sctx.lmetric.all_tokens.saturating_add_signed(all_tokens_inc);
            }
            // Renew states for next cycle
            temp_leaving_entries.0.clear();
            temp_leaving_entries.1.clear();
            skipped_fault_entries.0.clear();
            skipped_fault_entries.1.clear();

            // Invariant checks at event loop boundary
            #[cfg(debug_assertions)]
            {
                // No request should exist in both `entries` and `except_context.entries`
                for &id in entries.keys() {
                    debug_assert!(
                        !except_context.entries.contains_key(&id),
                        "Request_{id} found in both entries and except_context.entries"
                    );
                }
                // Every entry in `entries` should have a corresponding phase
                for &id in entries.keys() {
                    debug_assert!(
                        request_phases.contains_key(&id),
                        "Request_{id} in entries but missing from request_phases"
                    );
                }
                // No phase should be non-Excepted for requests in except_context
                for &id in except_context.entries.keys() {
                    if let Some(&phase) = request_phases.get(&id) {
                        debug_assert!(
                            phase == RequestPhase::Excepted,
                            "Request_{id} in except_context but phase is {:?}, expected Excepted",
                            phase
                        );
                    }
                }
            }
        }
    }
}

mod except_management {
    use nohash_hasher::{BuildNoHashHasher, IntMap};

    use crate::Entry;

    /// Events that drive the exception state machine.
    ///
    /// These are the *inputs* to the state machine, not the states themselves.
    /// Each variant triggers a well-defined transition in `ExtStInner`.
    #[allow(dead_code)] // Live(_) reserved for SSE-keep-alive transitions
    pub(super) enum ExtState {
        /// Frontend aborted the request; entry moves into exception tracking.
        Abort(u64, Entry),
        /// Backend reported an error for the request; entry moves into exception tracking.
        Fault(u64, Entry),
        /// Backend ACK-ed the abort (termination signal received).
        Term(u64),
        /// SSE stream reported the request as finished (is_finished=true).
        Exit(u64),
        /// SSE stream still references this request (keeps it alive one more cycle).
        Live(u64),
    }

    /// Exception state machine for a single request.
    ///
    /// # State Transition Diagram
    ///
    /// ```text
    ///   Abort(entry)          Fault(entry)
    ///       |                      |
    ///       v                      v
    ///   WaitingTerm           FaultedLive
    ///       |                      |
    ///       |--[Live]--> WaitingTermLive    |--[end-of-cycle/no Live]--> DropReady --> (dropped)
    ///       |                      |
    ///       |--[Exit]--> WaitingTermExited  |--[Live]--> FaultedLive (stay)
    ///       |                      |
    ///       |--[Term]--> (removed,         |--[Exit]--> FaultedExited
    ///       |             entry returned)   |
    ///       |                              |--[Term]--> (removed, impossible for Fault)
    ///       v
    ///   WaitingTermLive
    ///       |--[Exit]--> WaitingTermExited
    ///       |--[Term]--> (removed, entry returned)
    ///       |--[end-of-cycle]--> WaitingTerm (clear live flag)
    ///
    ///   WaitingTermExited
    ///       |--[Term]--> (removed, no entry -- already taken on Exit)
    ///
    ///   FaultedExited
    ///       |--[end-of-cycle]--> DropReady --> (dropped)
    ///       (entry already taken on Exit)
    /// ```
    ///
    /// The `live` flag is transient: set by `Live`, cleared at end of each SSE cycle
    /// by `clear_live_flag()`. It prevents premature dropping of faulted requests
    /// that are still referenced in the current SSE event.
    #[derive(Debug, PartialEq, Clone, Copy)]
    enum ExtStInner {
        /// Frontend abort initiated; waiting for backend Term ACK.
        /// Entry is held in `entries` map.
        WaitingTerm { live: bool },

        /// Backend fault; entry is held but may be dropped if not referenced
        /// in the next SSE cycle.
        /// `live=true` means the request was referenced in the current SSE event.
        Faulted { live: bool },

        /// SSE stream finished this request (Exit received).
        /// Entry has been removed from `entries` (returned to caller on Exit).
        /// For Abort path: still waiting for backend Term ACK.
        /// For Fault path: will be dropped at end of cycle.
        Exited { waiting_term: bool },
    }

    impl ExtStInner {
        /// Apply the `Live` event: mark the request as referenced in current SSE cycle.
        fn on_live(&mut self) {
            match self {
                ExtStInner::WaitingTerm { live, .. } => *live = true,
                ExtStInner::Faulted { live, .. } => *live = true,
                ExtStInner::Exited { .. } => {
                    // Already exited; Live after Exit is benign (SSE may still
                    // contain references to a finished request in the same event).
                }
            }
        }

        /// Apply the `Exit` event: SSE stream says the request is finished.
        /// Returns `true` if this is the first Exit (entry should be removed and returned).
        /// Returns `false` if already exited (double finish -- log error).
        fn on_exit(&mut self) -> bool {
            match *self {
                ExtStInner::WaitingTerm { .. } => {
                    *self = ExtStInner::Exited { waiting_term: true };
                    true
                }
                ExtStInner::Faulted { .. } => {
                    *self = ExtStInner::Exited { waiting_term: false };
                    true
                }
                ExtStInner::Exited { .. } => {
                    // Double finish -- caller should log error
                    false
                }
            }
        }

        /// Apply the `Term` event: backend ACK-ed the abort.
        /// Returns `true` if the entry is still in `entries` (not yet exited)
        /// and should be removed and returned.
        /// Returns `false` if the entry was already taken on Exit.
        ///
        /// After Term, the state entry should be removed from the states vec.
        fn on_term(&self) -> bool {
            match *self {
                ExtStInner::WaitingTerm { .. } => {
                    // Not yet exited; entry still in `entries`
                    true
                }
                ExtStInner::Exited { waiting_term: true } => {
                    // Already exited; entry was taken on Exit
                    false
                }
                ExtStInner::Faulted { .. } => {
                    debug_assert!(false, "Term received for Faulted request (should not happen)");
                    true
                }
                ExtStInner::Exited { waiting_term: false } => {
                    debug_assert!(
                        false,
                        "Term received for Faulted+Exited request (should not happen)"
                    );
                    false
                }
            }
        }

        /// Clear the `live` flag at the end of an SSE cycle.
        fn clear_live_flag(&mut self) {
            match self {
                ExtStInner::WaitingTerm { live, .. } => *live = false,
                ExtStInner::Faulted { live, .. } => *live = false,
                ExtStInner::Exited { .. } => {}
            }
        }

        /// Returns true if this entry should be dropped (faulted, not live, not exited).
        fn is_drop_ready(&self) -> bool {
            matches!(self, ExtStInner::Faulted { live: false })
        }
    }

    pub(super) struct ExtContext {
        pub entries: IntMap<u64, Entry>,
        states: Vec<(u64, ExtStInner)>,
    }

    impl ExtContext {
        pub fn new() -> Self {
            Self {
                entries: IntMap::with_capacity_and_hasher(32, BuildNoHashHasher::<u64>::default()),
                states: Vec::with_capacity(32),
            }
        }

        pub fn put(&mut self, st: ExtState) -> Option<Entry> {
            match st {
                ExtState::Abort(id, entry) => {
                    debug_assert!(
                        !self.entries.contains_key(&id),
                        "Request_{id} already in exception context on Abort"
                    );
                    self.entries.insert(id, entry);
                    self.states.push((id, ExtStInner::WaitingTerm { live: false }));
                    None
                }
                ExtState::Fault(id, entry) => {
                    debug_assert!(
                        !self.entries.contains_key(&id),
                        "Request_{id} already in exception context on Fault"
                    );
                    self.entries.insert(id, entry);
                    // Pessimistically mark as live for the current cycle;
                    // if not refreshed by Live next cycle, becomes drop-ready.
                    self.states.push((id, ExtStInner::Faulted { live: true }));
                    None
                }
                ExtState::Live(id) => {
                    if let Some((_, st)) = self.states.iter_mut().find(|(eid, _)| *eid == id) {
                        st.on_live();
                    }
                    None
                }
                ExtState::Exit(id) => {
                    let i = self.states.iter().position(|(eid, _)| *eid == id).expect(
                        format!("Request_{} not captured in exception context!", id).as_str(),
                    );
                    let (_, st) = self.states.get_mut(i).unwrap();
                    if st.on_exit() {
                        tracing::info!(target: "lifecycle", request_id = id, "EXCEPTION_EXIT");
                        self.entries.remove(&id)
                    } else {
                        tracing::error!(request_id = id, "DOUBLE_FINISH");
                        None
                    }
                }
                ExtState::Term(id) => {
                    let i = self.states.iter().position(|(eid, _)| *eid == id).expect(
                        format!("Request_{} not captured in exception context!", id).as_str(),
                    );
                    let (_, st) = self.states.get(i).unwrap();
                    let has_entry = st.on_term();
                    if has_entry {
                        tracing::info!(target: "lifecycle", request_id = id, "EXCEPTION_TERM");
                    } else {
                        tracing::debug!(target: "lifecycle", request_id = id, "EXCEPTION_TERM_AFTER_EXIT");
                    }
                    self.states.remove(i);
                    if has_entry { self.entries.remove(&id) } else { None }
                }
            }
        }

        /// End-of-cycle cleanup: drop faulted requests that are no longer referenced
        /// by SSE, and clear the live flag for all remaining entries.
        pub fn filter_drop(&mut self) -> Vec<(u64, Entry)> {
            let mut rem = Vec::new();
            let mut dropped = Vec::new();

            for &(id, st) in self.states.iter() {
                if st.is_drop_ready() {
                    tracing::info!(target: "lifecycle", request_id = id, "EXCEPTION_DROP");
                    dropped.push((
                        id,
                        self.entries.remove(&id).expect(
                            format!("Request_{} not captured in exception context!", id).as_str(),
                        ),
                    ));
                } else {
                    let mut next_st = st;
                    next_st.clear_live_flag();
                    rem.push((id, next_st));
                }
            }

            self.states = rem;
            dropped
        }
    }
}
