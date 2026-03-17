use crate::{
    queue::TaskAssigner,
    vllmlet::{VllmClient, VllmClientError},
    LMetric, ScheduleContext, THROTTLE_THLD, TPOT_THRESHOLD, TPS_THRESHOLD,
};
use reqwest::Response;
use std::sync::Arc;
use tokio::{sync::Mutex, task::JoinHandle};

pub(crate) struct ColocationController {
    pub all_schedule_contexts: Vec<Arc<Mutex<ScheduleContext>>>,
    pub batching_queue: TaskAssigner,
}

impl ColocationController {}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ExtExcept {
    #[error("frontend aborted: client dropped the response channel")]
    FrontendAbort,

    #[error("backend error: {0}")]
    BackendFault(#[from] VllmClientError),
}

pub(crate) fn start_vllm_colocation_event_loop(
    queue: TaskAssigner,
    all_vllm_clients: Vec<VllmClient>,
    all_schedule_contexts: Vec<Arc<Mutex<ScheduleContext>>>,
) -> Arc<ColocationController> {
    for (replica_index, vllm_client) in all_vllm_clients.into_iter().enumerate() {
        tokio::spawn(task_assignment::work_event_loop(
            replica_index,
            vllm_client,
            queue.clone(),
            all_schedule_contexts[replica_index].clone(),
        ));
    }

    let controller = ColocationController { batching_queue: queue, all_schedule_contexts };
    Arc::new(controller)
}

type VllmClientResp = JoinHandle<Result<Response, VllmClientError>>;

#[deprecated]
fn throttle_for_decoding(idx: usize, lmetric: &LMetric, throttled: &mut bool) {
    let nreq = lmetric.bs;
    let tps = lmetric.tps;
    let tpot_mili = (lmetric.tpot * 1000.) as usize;
    if nreq > THROTTLE_THLD && (tps < TPS_THRESHOLD || tpot_mili > TPOT_THRESHOLD) {
        if !*throttled {
            *throttled = true;
            tracing::warn!(
                "Vllm#{} throttled! [#req={}, tps={}, tpot={}ms]",
                idx,
                nreq,
                tps,
                tpot_mili
            );
        }
    } else {
        *throttled = false;
    }
}

mod task_assignment {
    use eventsource_client as es;
    use futures::StreamExt;
    use nohash_hasher::{BuildNoHashHasher, IntMap};
    use pb::generate::v2 as proto;
    use reqwest::Response;
    use tokio::sync::mpsc::{self, channel, error::SendError, Receiver};
    use tokio::sync::Mutex;
    use tokio::task::{yield_now, JoinHandle};
    use tokio::time::Duration;

    use core::panic;
    use std::sync::Arc;

    use super::except_management::{ExtContext, ExtState};
    use super::{ScheduleContext, VllmClientResp};
    use crate::{
        infer::{InferError, InferStreamResponse},
        kvcache::BlockHash,
        queue::{Entry, QueuePro},
        vllmlet::{VllmClient, VllmClientError, VllmMetric, VllmRequestStatus},
        ExtExcept, LMetricDec, Token,
    };

    /// Control the workload level of delegated instance
    pub async fn work_event_loop<Q: QueuePro>(
        replica_index: usize,
        mut vllm_client: VllmClient,
        queue: Q,
        schedule_context: Arc<Mutex<ScheduleContext>>,
    ) {
        // Channel: work queue -> completion queue
        let (wq_tx, cq_wqe_rx) = channel::<(Entry, VllmClientResp)>(64);
        let wqe_mtx = Arc::new(Mutex::new(()));

        // Init completion queue poller
        let cq_error_rx = vllm_client.get_error_rx();
        let vllm = vllm_client.clone();
        tokio::spawn(completion_event_loop(
            replica_index,
            vllm,
            wqe_mtx.clone(),
            cq_wqe_rx,
            cq_error_rx,
            schedule_context,
        ));

        // flag for logging
        loop {
            match wq_tx.reserve().await {
                Ok(permit) => {
                    if let Some((id, entry)) = queue.next_request(replica_index).await {
                        // NOTE: before-or-after atomicity to avoid RAW-like problem when using channel,
                        //       i.e., http request is posted, but WQE has not put into channel
                        let _g = wqe_mtx.lock().await;
                        let resp_from_vllm_client =
                            vllm_client.add_request(id, &entry.request).await;
                        tracing::info!(
                            "Request_{id} queued {}us, with input length {} output length {}, added to vLLM#{replica_index}",
                            entry.batch_time.unwrap().duration_since(entry.queue_time).as_micros(),
                            entry.request.input_length,
                            entry.request.stopping_parameters.max_new_tokens,
                        );
                        // The receiver won't actively drop channel
                        let _ = permit.send((entry, resp_from_vllm_client));
                    } else {
                        yield_now().await;
                    }
                }
                Err(SendError(_)) => {
                    unreachable!("vLLM#{replica_index} WQ channel closed by receiver at CQ");
                }
            }
        }
    }

    /// First phase in abortion 2PC
    fn on_frontend_abort(
        entries: &mut IntMap<u64, Entry>,
        vllm_resps: &mut IntMap<u64, JoinHandle<Result<Response, VllmClientError>>>,
        cancel_req_ids: &mut Vec<u64>,
        skip_entries: &mut (Vec<u64>, Vec<Entry>),
        except_context: &mut ExtContext,
    ) {
        while let Some(id) = cancel_req_ids.pop() {
            tracing::warn!("Request_{id} is cancelling...");
            vllm_resps
                .remove(&id)
                .expect(format!("JoinHandle of Request_{} has been moved!", id).as_str())
                .abort();
            tracing::debug!("Request_{id} has aborted handle.");
            skip_entries.0.push(id);
            let entry = entries
                .remove(&id)
                .expect(format!("Request_{} not found in entries. This is a bug.", id).as_str());
            except_context.put(ExtState::Abort(id, entry));
        }
    }

    /// Handle any request error at backend
    ///
    /// First phase in fault 2PC
    async fn on_backend_fault(
        entries: &mut IntMap<u64, Entry>,
        vllm_resps: &mut IntMap<u64, JoinHandle<Result<Response, VllmClientError>>>,
        error_rx: &mut mpsc::UnboundedReceiver<u64>,
        skip_entries: &(Vec<u64>, Vec<Entry>),
        except_context: &mut ExtContext,
    ) {
        while let Ok(id) = error_rx.try_recv() {
            if skip_entries.0.contains(&id) {
                continue;
            }
            // `response.await.unwrap()` := the spawned task nethier panics nor is not cancelled
            // `response.await.map()` := only handles `Ok` value
            let _ = vllm_resps.remove(&id).unwrap().await.map(|resp| {
                tracing::info!(
                    "Request_{id} has error {} at backend, notifying frontend...",
                    resp.unwrap_err()
                );
            });
            // NOTE: tolerent the error request occurs in the same SSE event, but no more
            let entry = entries
                .remove(&id)
                .expect(format!("Request_{} not found in entries. This is a bug.", id).as_str());
            // NOTE: skip possible `SendError`, backend resource has been freed
            let _ = entry
                .response_tx
                .send(Err(InferError::GenerationError("Vllm refused to serve!".to_string())));
            //
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
    ) -> Result<(), ExtExcept> {
        tracing::info!(
            "Vllm#{replica_index}::Request_{request_id} prefill done with {} actual hit tokens!",
            hit_token_cnt
        );
        entry
            .response_tx
            .send(Ok(InferStreamResponse::Prefill(proto::Tokens::default())))
            .map_err(|_| ExtExcept::FrontendAbort)?;
        for &t in new_token_ids {
            entry
                .response_tx
                .send(Ok(InferStreamResponse::Intermediate {
                    token: Token { id: t, text: String::default(), logprob: 0.0, special: false },
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
    ) -> Result<(), ExtExcept> {
        tracing::trace!("Vllm#{replica_index}::Request_{request_id} decoding!");

        for &t in new_token_ids {
            entry
                .response_tx
                .send(Ok(InferStreamResponse::Intermediate {
                    token: Token { id: t, text: String::default(), logprob: 0.0, special: false },
                    top_tokens: Vec::default(),
                }))
                .map_err(|_| ExtExcept::FrontendAbort)?;
        }

        Ok(())
    }

    /// # Precondition:
    ///   + `ctx` is clean, no lock is held
    async fn on_finish_request(
        replica_index: usize,
        entry: Entry,
        response: JoinHandle<Result<Response, VllmClientError>>,
        request_id: u64,
    ) -> Option<Entry> {
        let id = request_id;
        tracing::info!(
            "Vllm#{replica_index}::Request_{id} is finished generating {} tokens",
            entry.generated_token_cnt
        );
        match response.await {
            // Spawned POST task has a return value
            Ok(resp) => match resp {
                // vLLM's response is OK
                Ok(resp) => {
                    let _generation = resp.bytes().await.unwrap();
                    let _skip = entry.response_tx.send(Ok(InferStreamResponse::End {
                        token: Token::default(),
                        top_tokens: Vec::default(),
                        generated_text: proto::GeneratedText::default(),
                        start: entry.batch_time.unwrap(),
                        queued: entry.queue_time,
                        max_time_between_tokens: entry.max_time_between_tokens,
                    }));
                    // NOTE: skip possible `SendError`, backend resource has been freed
                    None
                }
                // vLLM's response is Error
                Err(_e) => {
                    // NOTE: skip this error here, since inner `vllm_client` has already
                    // put `request_id` into cancel list
                    Some(entry)
                }
            },
            // Spawned POST task is cancelled
            Err(e) if e.is_cancelled() => {
                tracing::info!(
                    "Vllm#{replica_index}::Request_{id} HTTP request has been cancelled."
                );
                None
            }
            // Spawned POST task panics
            Err(e) => {
                tracing::error!("Vllm#{}::Request_{} join error: {}.", replica_index, id, e);
                let _ = entry.response_tx.send(Err(InferError::GenerationError(e.to_string())));
                None
            }
        }
    }

    /// Poll backend instance's event stream, and enable callbacks
    async fn completion_event_loop(
        replica_index: usize,
        vllm: VllmClient,
        wqe_mtx: Arc<Mutex<()>>,
        mut cq_wqe_rx: Receiver<(Entry, VllmClientResp)>,
        mut cq_error_rx: mpsc::UnboundedReceiver<u64>,
        schedule_context: Arc<Mutex<ScheduleContext>>,
    ) {
        let mut entries =
            IntMap::with_capacity_and_hasher(256, BuildNoHashHasher::<u64>::default());
        let mut vllm_resps =
            IntMap::with_capacity_and_hasher(256, BuildNoHashHasher::<u64>::default());

        // Handle frontend abortions and backend errors
        let mut cancel_req_ids = Vec::new();
        let mut except_context = ExtContext::new();
        let mut temp_leaving_entries = (Vec::with_capacity(8), Vec::with_capacity(8));
        let mut skipped_fault_entries = (Vec::with_capacity(8), Vec::with_capacity(8));

        // Init sse stream
        let sse_client = vllm
            .init_sse_client()
            .await
            .expect(format!("Vllm#{} failed to initialize metric sse!", replica_index).as_str());
        let mut sse_stream = sse_client.stream();

        match sse_stream.next().await.unwrap() {
            Ok(cnnt) => {
                if let es::SSE::Connected(connection_details) = cnnt {
                    let response = connection_details.response();
                    assert_eq!(
                        response.status(),
                        200,
                        "Vllm#{} metric sse erroneous status {}",
                        replica_index,
                        response.status()
                    );
                }
            }
            Err(e) => {
                tracing::error!("Vllm#{} metric sse error: {}", replica_index, e);
                panic!("Vllm#{} failed to connect metric sse!", replica_index);
            }
        }

        // Error message slot for debugging
        let mut error_event: Option<VllmMetric> = None;

        // Main event loop
        'raise_err: while let Some(sse) = sse_stream.next().await {
            // Adds newly posted requests
            {
                let _g = wqe_mtx.lock().await;
                while let Ok((entry, vllm_respd)) = cq_wqe_rx.try_recv() {
                    let rid = entry.request.request_id;
                    entries.insert(rid, entry);
                    vllm_resps.insert(rid, vllm_respd);
                }
            }
            // postcond: all requests in sse are visible to CQ

            let mut term_requests = Vec::new();
            match sse {
                Ok(es::SSE::Event(e)) => {
                    let es::Event { event_type: _, data, id: _, retry: _ } = e;
                    let m: VllmMetric = serde_json::from_str(&data).expect(
                        format!("vLLM#{} es::Event::data = {:?}", replica_index, data).as_str(),
                    );
                    tracing::trace!("vLLM#{}::Event::data received {:?}", replica_index, m);

                    if !m.preempted_ids.is_empty() {
                        m.preempted_ids.iter().for_each(|&id| {
                            tracing::warn!("Request_{id} is preempted at backend!");
                        });
                    }

                    // fast path: update metrics
                    let tbt = Duration::from_millis(m.latency);
                    let mut metric_delta = LMetricDec::new(&tbt);
                    // NOTE: `prefill_tokens` dosen't count hit tokens, while
                    //       `all_tokens` does count hit tokens
                    metric_delta.prefill_tokens_dec = m.prefill_tokens as isize;
                    for request_status in &m.outputs {
                        let VllmRequestStatus {
                            request_id,
                            new_token_ids: new_tokens,
                            state,
                            is_finished,
                            hit_token_cnt,
                        } = request_status;
                        match state.as_str() {
                            "PREFILL" => {
                                metric_delta.waiting_reqs_dec += 1;
                                if *is_finished {
                                    if let Some(entry) = entries.get_mut(request_id) {
                                        let input_length = entry.request.input_length as isize;
                                        metric_delta.bs_dec += 1;
                                        metric_delta.all_tokens_inc -= input_length as isize;
                                    } else if let Some(entry) =
                                        // `unwrap` inside, `entry` must be either in `entries` or `except_context`
                                        except_context
                                            .put(ExtState::Exit(*request_id))
                                    {
                                        let input_length = entry.request.input_length as isize;
                                        metric_delta.bs_dec += 1;
                                        metric_delta.all_tokens_inc -= input_length as isize;
                                        // NOTE:
                                        temp_leaving_entries.0.push(*request_id);
                                        temp_leaving_entries.1.push(entry);
                                    }
                                } else {
                                    metric_delta.all_tokens_inc += new_tokens.len() as isize;
                                    let entry = entries.get_mut(request_id).unwrap_or_else(|| {
                                        except_context.put(ExtState::Live(*request_id));
                                        except_context.entries.get_mut(request_id).unwrap()
                                    });
                                    let inc_hit_nblks = entry
                                        .block_hash_state
                                        .set_real_token_hits_get_diff(*hit_token_cnt);
                                    tracing::info!(
                                        "vLLM#{replica_index}::Request_{} correct {} hit tokens",
                                        *request_id,
                                        inc_hit_nblks
                                            * entry.block_hash_state.get_block_size() as isize
                                    );
                                    metric_delta.prefill_tokens_dec += inc_hit_nblks
                                        * entry.block_hash_state.get_block_size() as isize;
                                    entry.append_state(new_tokens, &tbt);
                                }
                            }
                            "DECODE" => {
                                if *is_finished {
                                    if let Some(entry) = entries.get_mut(request_id) {
                                        let request = &entry.request;
                                        metric_delta.bs_dec += 1;
                                        // NOTE: `generated_token_cnt` has not been appeneded, so just make decrement
                                        metric_delta.all_tokens_inc -= request.input_length
                                            as isize
                                            + entry.generated_token_cnt as isize;
                                    } else if let Some(entry) =
                                        // `unwrap` inside, `entry` must be either in `entries` or `except_context`
                                        except_context
                                            .put(ExtState::Exit(*request_id))
                                    {
                                        let request = &entry.request;
                                        metric_delta.bs_dec += 1;
                                        // NOTE: `generated_token_cnt` has not been appeneded, so just make decrement
                                        metric_delta.all_tokens_inc -= request.input_length
                                            as isize
                                            + entry.generated_token_cnt as isize;
                                        // NOTE:
                                        temp_leaving_entries.0.push(*request_id);
                                        temp_leaving_entries.1.push(entry);
                                    }
                                } else {
                                    let entry = entries.get_mut(request_id).unwrap_or_else(|| {
                                        except_context.put(ExtState::Live(*request_id));
                                        except_context.entries.get_mut(request_id).unwrap()
                                    });
                                    entry.append_state(new_tokens, &tbt);
                                    // postcond: `Some(entry.tpot)`
                                    metric_delta.all_tokens_inc += new_tokens.len() as isize;
                                    metric_delta.tpot +=
                                        entry.time_of_per_token.unwrap().as_secs_f32();
                                }
                            }
                            _ => {
                                eprintln!("Request_{request_id} invalid state: {state}!");
                                error_event = Some(m);
                                break 'raise_err;
                            }
                        }
                    }
                    let mut sctx = schedule_context.lock().await;

                    // Update PrefixBlockHash
                    if !m.evicted_block_ids.is_empty() {
                        tracing::debug!("Backend removes bids {:?}", m.evicted_block_ids);
                    }
                    sctx.block_hash.remove(m.evicted_block_ids);
                    for (rid, block_indices) in m.cur_used_block_ids {
                        if block_indices.is_empty() {
                            continue;
                        }
                        // Total backend bids
                        let entry = entries.get_mut(&rid).unwrap_or_else(|| {
                            except_context.entries.get_mut(&rid).unwrap_or_else(|| {
                                let i = temp_leaving_entries
                                    .0
                                    .iter()
                                    .position(|&eid| rid == eid)
                                    .unwrap();
                                temp_leaving_entries.1.get_mut(i).unwrap()
                            })
                        });
                        tracing::debug!("Entry_{rid} update backend bids {:?}", block_indices);
                        entry.block_hash_state.set_bids(block_indices);
                    }
                    for (rid, block_indices) in m.new_block_hashes_ids {
                        if block_indices.is_empty() {
                            continue;
                        }
                        let entry = entries.get_mut(&rid).unwrap_or_else(|| {
                            except_context.entries.get_mut(&rid).unwrap_or_else(|| {
                                let i = temp_leaving_entries
                                    .0
                                    .iter()
                                    .position(|&eid| rid == eid)
                                    .unwrap();
                                temp_leaving_entries.1.get_mut(i).unwrap()
                            })
                        });
                        match entry.block_hash_state.get_onto_hashes(&block_indices) {
                            Ok(onto_hashes) => {
                                tracing::debug!(
                                    "Entry_{rid} insert ({:?}) |-> [{:?}]",
                                    onto_hashes,
                                    block_indices
                                );
                                sctx.block_hash.insert(onto_hashes, block_indices);
                            }
                            Err(backend_bids) => {
                                let err_msg = format!("Entry_{rid} inconsistent bid: frontend marking occupied {:?} | backend newly committed {:?}", backend_bids, block_indices);
                                // tracing::error!(err_msg);
                                panic!("{err_msg}");
                            }
                        }
                    }
                    // Publishes updated instance-level metric state
                    sctx.lmetric -= metric_delta;
                    drop(sctx);

                    // Aborted requests ACK-ed by backend
                    term_requests = m.aborted_requests;

                    // slow path: pass generations
                    for request_status in m.outputs {
                        let VllmRequestStatus {
                            request_id,
                            new_token_ids,
                            state,
                            is_finished,
                            hit_token_cnt,
                        } = request_status;
                        match state.as_str() {
                            "PREFILL" if entries.get(&request_id).is_some() => {
                                let entry = entries.get_mut(&request_id).unwrap();
                                if let Err(ExtExcept::FrontendAbort) = on_prefill(
                                    replica_index,
                                    entry,
                                    request_id,
                                    hit_token_cnt,
                                    &new_token_ids,
                                )
                                .await
                                {
                                    cancel_req_ids.push(request_id);
                                }
                                if is_finished {
                                    let entry = entries.remove(&request_id).unwrap();
                                    let response = vllm_resps.remove(&request_id).unwrap();
                                    if let Some(entry) = on_finish_request(
                                        replica_index,
                                        entry,
                                        response,
                                        request_id,
                                    )
                                    .await
                                    {
                                        skipped_fault_entries.0.push(request_id);
                                        skipped_fault_entries.1.push(entry);
                                    }
                                    // NOTE: revokes FrontendAbort, since SSE & POST channels are both terminated
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
                                    on_decode(replica_index, entry, request_id, &new_token_ids)
                                        .await
                                {
                                    cancel_req_ids.push(request_id);
                                }
                                if is_finished {
                                    let entry = entries.remove(&request_id).unwrap();
                                    let response: VllmClientResp =
                                        vllm_resps.remove(&request_id).unwrap();
                                    if let Some(entry) = on_finish_request(
                                        replica_index,
                                        entry,
                                        response,
                                        request_id,
                                    )
                                    .await
                                    {
                                        skipped_fault_entries.0.push(request_id);
                                        skipped_fault_entries.1.push(entry);
                                    }
                                    // NOTE: revokes FrontendAbort, since SSE & POST channels are both terminated
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
                }
                Ok(es::SSE::Comment(c)) => {
                    tracing::error!("Vllm#{} metric sse send comment {}", replica_index, c);
                }
                Ok(es::SSE::Connected(_)) => {
                    unreachable!();
                }
                Err(e) => {
                    tracing::error!("Vllm#{} metric sse error: {}", replica_index, e);
                }
            }

            // Handle request exceptions.
            //
            // Newly canceled requests move to 1st commit phase
            on_frontend_abort(
                &mut entries,
                &mut vllm_resps,
                &mut cancel_req_ids,
                &mut skipped_fault_entries,
                &mut except_context,
            );
            on_backend_fault(
                &mut entries,
                &mut vllm_resps,
                &mut cq_error_rx,
                &skipped_fault_entries,
                &mut except_context,
            )
            .await;
            // Requests to be marked as terminated with backend abort ACK-ed
            let mut bs_dec = 0;
            let mut all_tokens_inc = 0;
            term_requests
                .into_iter()
                .map(|term_id| except_context.put(ExtState::Term(term_id)))
                .filter_map(|x| x)
                .for_each(|entry| {
                    bs_dec += 1;
                    all_tokens_inc -=
                        entry.request.input_length as isize + entry.generated_token_cnt as isize;
                });
            except_context.filter_drop().into_iter().for_each(|(_, entry)| {
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
        }

        // Display message with erroneous event
        if let Some(event) = error_event {
            eprintln!("Vllm#{replica_index}::SSE erroneous event {:?}", event);
            panic!();
        }
    }
}

mod except_management {
    use nohash_hasher::{BuildNoHashHasher, IntMap};

    use crate::Entry;

    use std::ops::Not;

    pub(super) enum ExtState {
        Abort(u64, Entry),
        Fault(u64, Entry),
        Term(u64),
        Exit(u64),
        Live(u64),
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct ExtStInner(u64);

    impl ExtStInner {
        /// Init state of backend error,
        /// and is dropped if not occurs in current SSE event,
        /// i.e., `WAIT_CLOSE` unset.
        const DROP_NEXT: ExtStInner = ExtStInner(0);
        /// Init state of frontend abort,
        /// where a termination signal is expected.
        const WAIT_TERM: ExtStInner = ExtStInner(1 << 0);
        /// Occurs in current SSE event,
        /// and unset this mask at the end of SSE stream.
        const WAIT_CLOSE: ExtStInner = ExtStInner(1 << 1);
        // SSE stream finishes, ownership has been moved
        const EXITED: ExtStInner = ExtStInner(1 << 2);
    }

    impl Not for ExtStInner {
        type Output = ExtStInner;

        fn not(self) -> Self::Output {
            ExtStInner(!self.0)
        }
    }

    macro_rules! impl_fmt_span_bit_op {
        ($trait:ident, $func:ident, $op:tt) => {
            impl std::ops::$trait for ExtStInner {
                type Output = ExtStInner;

                fn $func(self, rhs: Self) -> Self::Output {
                    ExtStInner(self.0 $op rhs.0)
                }
            }
        };
    }

    macro_rules! impl_fmt_span_bit_assign_op {
        ($trait:ident, $func:ident, $op:tt) => {
            impl std::ops::$trait for ExtStInner {
                fn $func(&mut self, rhs: Self) {
                    *self = ExtStInner(self.0 $op rhs.0)
                }
            }
        };
    }

    impl_fmt_span_bit_op!(BitAnd, bitand, &);
    impl_fmt_span_bit_op!(BitOr, bitor, |);
    impl_fmt_span_bit_op!(BitXor, bitxor, ^);

    impl_fmt_span_bit_assign_op!(BitAndAssign, bitand_assign, &);
    impl_fmt_span_bit_assign_op!(BitOrAssign, bitor_assign, |);
    impl_fmt_span_bit_assign_op!(BitXorAssign, bitxor_assign, ^);

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
                    self.entries.insert(id, entry);
                    self.states.push((id, ExtStInner::WAIT_TERM));
                    None
                }
                ExtState::Fault(id, entry) => {
                    self.entries.insert(id, entry);
                    self.states.push((id, ExtStInner::WAIT_CLOSE)); // pessimistically wait 1 cycle
                    None
                }
                ExtState::Live(id) => {
                    if let Some((_, st)) = self.states.iter_mut().find(|(eid, _)| *eid == id) {
                        *st |= ExtStInner::WAIT_CLOSE;
                    }
                    None
                }
                ExtState::Exit(id) => {
                    let i = self.states.iter().position(|(eid, _)| *eid == id).expect(
                        format!("Request_{} not captured in exception context!", id).as_str(),
                    );
                    let (_, st) = self.states.get_mut(i).unwrap();
                    if *st & ExtStInner::EXITED == ExtStInner::EXITED {
                        tracing::error!("Request_{id} double finish!");
                        None
                    } else {
                        tracing::info!("Request_{id} exits exception context.");
                        *st |= ExtStInner::EXITED;
                        self.entries.remove(&id)
                    }
                }
                ExtState::Term(id) => {
                    let i = self.states.iter().position(|(eid, _)| *eid == id).expect(
                        format!("Request_{} not captured in exception context!", id).as_str(),
                    );
                    let (_, st) = self.states.get_mut(i).unwrap();
                    if *st & ExtStInner::EXITED == ExtStInner::DROP_NEXT {
                        tracing::info!("Request_{id} terminates in exception context.");
                        // not yet Exit => Term -> safely drop
                        self.states.remove(i);
                        self.entries.remove(&id)
                    } else {
                        tracing::debug!("Request_{id} terminates after exited.");
                        // Exit => Term
                        self.states.remove(i);
                        None
                    }
                }
            }
        }

        pub fn filter_drop(&mut self) -> Vec<(u64, Entry)> {
            let mut rem = Vec::new();
            let mut drop = Vec::new();
            // clear WAIT_CLOSE bitmask
            let unset_mask = !ExtStInner::WAIT_CLOSE;

            self.states.iter().for_each(|&(id, st)| {
                if st == ExtStInner::DROP_NEXT {
                    tracing::info!("Request_{id} is dropped from exception context.");
                    drop.push((
                        id,
                        self.entries.remove(&id).expect(
                            format!("Request_{} not captured in exception context!", id).as_str(),
                        ),
                    ));
                } else {
                    // {WAIT_TERM | EXITED, WAIT_TERM, DROP_NEXT} | WAIT_CLOSE
                    rem.push((id, st & unset_mask));
                }
            });

            self.states = rem;
            drop
        }
    }
}
