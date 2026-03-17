// Debug round-robin scheduling policy.
//
// This is a special-purpose queue with its own custom `queue_task` that
// bypasses the standard `QueueRunner` machinery.  It always commits
// immediately and does not use the generic scheduling step.

use super::{Entry, NextBatch, NextRequest, QueueCommandPro, QueuePro};
use crate::kvcache::BlockHash;
use crate::{LMetricInc, ScheduleContext};

use std::collections::VecDeque;
use std::sync::Arc;

use nohash_hasher::{BuildNoHashHasher, IntMap};
use pb::generate::v2::*;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::{info_span, instrument, Span};

/// Debug round-robin queue with a fully custom background task.
#[derive(Clone)]
pub(crate) struct DbgRRQueue {
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

impl DbgRRQueue {
    pub fn new(
        num_replicas: usize,
        all_schedule_context: Vec<Arc<Mutex<ScheduleContext>>>,
    ) -> Self {
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        tokio::spawn(DbgRRQueue::queue_task(
            num_replicas,
            queue_receiver,
            all_schedule_context,
        ));

        Self { queue_sender }
    }

    async fn queue_task(
        num_replicas: usize,
        mut receiver: mpsc::UnboundedReceiver<QueueCommandPro>,
        all_schedule_context: Vec<Arc<Mutex<ScheduleContext>>>,
    ) {
        assert_eq!(num_replicas, all_schedule_context.len());
        // Internal data structures
        let mut next_batch_id = 0;
        let mut next_replica_id: usize = 0;
        let mut all_commit_req_buffers: Vec<VecDeque<(u64, Entry)>> =
            (0..num_replicas).map(|_| VecDeque::with_capacity(64)).collect();

        while let Some(cmd) = receiver.recv().await {
            match cmd {
                QueueCommandPro::Append(entry, _span) => {
                    metrics::increment_gauge!("blitz_queue_size", 1.0);
                    // HACK: always eligible -> bypass eligibility checking
                    let replica_idx = next_replica_id;
                    let request = &entry.request;

                    // postcond: assignment is committed
                    next_replica_id = (next_replica_id + 1) % num_replicas;
                    // Calculate and apply metric increments
                    {
                        let ScheduleContext { lmetric, block_hash } =
                            &mut *all_schedule_context[replica_idx].lock().await;
                        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
                        entry.block_hash_state.set_pred_block_hits(hit_nblks);
                        let new_ntkns = request.input_tokens.len()
                            - /*inconsistent*/ hit_nblks
                                * entry.block_hash_state.get_block_size();
                        tracing::debug!(
                            "vLLM#{replica_idx}::Request_{} hits {hit_nblks} kvcache blocks",
                            request.request_id
                        );
                        let metric_inc = LMetricInc {
                            bs_inc: 1,
                            waiting_reqs_inc: 1,
                            prefill_tokens_inc: new_ntkns,
                            all_tokens_inc: request.input_tokens.len(),
                        };
                        (*lmetric) += metric_inc;
                    }

                    let committed_req_buf =
                        all_commit_req_buffers.get_mut(replica_idx).unwrap();
                    committed_req_buf.push_back((request.request_id, *entry));
                }
                QueueCommandPro::NextBatch(replica_idx, response_sender) => {
                    let entries = all_commit_req_buffers
                        .get_mut(replica_idx)
                        .unwrap()
                        .drain(..)
                        .collect::<Vec<_>>();

                    if entries.is_empty() {
                        response_sender.send(None).unwrap();
                        continue;
                    }

                    // Create span for this batch
                    let next_batch_span =
                        info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
                    next_batch_span.follows_from(&Span::current());

                    let mut batch_requests = Vec::with_capacity(entries.len());
                    let mut batch_entries = IntMap::with_capacity_and_hasher(
                        entries.len(),
                        BuildNoHashHasher::default(),
                    );

                    for (id, mut entry) in entries {
                        let entry_batch_span = info_span!(parent: &entry.span, "infer");
                        next_batch_span.follows_from(&entry_batch_span);
                        entry_batch_span.follows_from(&next_batch_span);
                        entry.temp_span = Some(entry_batch_span);

                        batch_requests.push(Request {
                            id,
                            prefill_logprobs: entry.request.decoder_input_details,
                            inputs: entry.request.inputs.clone(),
                            truncate: Some(entry.request.truncate),
                            parameters: Some(entry.request.parameters.clone()),
                            stopping_parameters: entry.request.stopping_parameters.clone(),
                            top_n_tokens: entry.request.top_n_tokens,
                            input_tokens: entry.request.input_tokens.clone(),
                        });
                        entry.batch_time = Some(Instant::now());
                        batch_entries.insert(id, entry);
                    }

                    let size = batch_requests.len() as u32;
                    next_batch_span.record("batch_size", size);

                    let batch = Batch {
                        id: next_batch_id,
                        requests: batch_requests,
                        size,
                        max_tokens: 0, // neglect it!
                    };
                    next_batch_id += 1;

                    metrics::histogram!("blitz_batch_next_size", batch.size as f64);

                    response_sender
                        .send(Some((batch_entries, batch, next_batch_span)))
                        .unwrap();
                }
                QueueCommandPro::NextRequest(replica_idx, response_sender) => {
                    // HACK: always eligible -> always commits -> bypass progressiveness check
                    if let Some((id, mut entry)) = all_commit_req_buffers
                        .get_mut(replica_idx)
                        .unwrap()
                        .pop_front()
                    {
                        entry.batch_time = Some(Instant::now());
                        response_sender.send(Some((id, entry))).unwrap();
                    } else {
                        response_sender.send(None).unwrap();
                    }
                }
                QueueCommandPro::WaitingPrefillTokens(_sender) => {
                    unimplemented!();
                }
                QueueCommandPro::WaitingRequests(_sender) => {
                    unimplemented!();
                }
            }
        }
    }
}

impl QueuePro for DbgRRQueue {
    #[instrument(skip_all)]
    fn append(&self, entry: Entry) {
        self.queue_sender
            .send(QueueCommandPro::Append(Box::new(entry), Span::current()))
            .unwrap();
    }

    #[instrument(skip_all)]
    async fn next_batch(&self, replica_id: usize) -> Option<NextBatch> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.queue_sender
            .send(QueueCommandPro::NextBatch(replica_id, tx))
            .unwrap();
        rx.await.unwrap()
    }

    #[instrument(skip_all)]
    async fn next_request(&self, replica_id: usize) -> Option<NextRequest> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.queue_sender
            .send(QueueCommandPro::NextRequest(replica_id, tx))
            .unwrap();
        rx.await.unwrap()
    }

    #[instrument(skip_all)]
    async fn waiting_prefill_tokens(&self) -> usize {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.queue_sender
            .send(QueueCommandPro::WaitingPrefillTokens(tx))
            .unwrap();
        rx.await.unwrap()
    }

    #[instrument(skip_all)]
    async fn waiting_requests(&self) -> usize {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.queue_sender
            .send(QueueCommandPro::WaitingRequests(tx))
            .unwrap();
        rx.await.unwrap()
    }
}
