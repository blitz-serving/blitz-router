//! `PolicyRunner<P: Policy>` — the queue runner that drives DSL-emitted
//! policies.
//!
//! Mirrors the legacy `QueueRunner<P: QueuePlusPlus>` queue management
//! (mpsc command channel, per-replica commit buffers, batch dispatch)
//! but calls `P::schedule(entry, all_sctx, &mut gctx)` from
//! `policies::policy_trait::Policy` instead of the old
//! `eligible_with_kvblock_hit + sampler + apply_schedule_decision`
//! triplet. The `apply_default_after` step is now part of the codegen
//! emitted by the `policy!` macro, so the runner does not call it
//! itself.
//!
//! The `gctx: P::GlobalContext` lives across calls inside the queue
//! task; policy `after:` clauses mutate it directly.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::{instrument, Span};

use crate::ScheduleContext;

use super::policy_trait::Policy;
use super::{Entry, NextRequest, QueuePro};

enum PolicyCommand {
    Append(Box<Entry>, Span),
    NextRequest(usize, oneshot::Sender<Option<NextRequest>>),
}

pub(crate) struct PolicyRunner<P: Policy> {
    queue_sender: mpsc::UnboundedSender<PolicyCommand>,
    _marker: PhantomData<P>,
}

impl<P: Policy> Clone for PolicyRunner<P> {
    fn clone(&self) -> Self {
        Self {
            queue_sender: self.queue_sender.clone(),
            _marker: PhantomData,
        }
    }
}

impl<P> PolicyRunner<P>
where
    P: Policy + Send + Sync + 'static,
{
    pub fn new(
        num_replicas: usize,
        all_schedule_context: Vec<Arc<Mutex<ScheduleContext>>>,
    ) -> Self {
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        tokio::spawn(Self::queue_task(
            num_replicas,
            queue_receiver,
            all_schedule_context,
        ));

        Self {
            queue_sender,
            _marker: PhantomData,
        }
    }

    async fn queue_task(
        num_replicas: usize,
        mut receiver: mpsc::UnboundedReceiver<PolicyCommand>,
        all_schedule_context: Vec<Arc<Mutex<ScheduleContext>>>,
    ) {
        let mut gctx = <P::GlobalContext as Default>::default();
        let mut uncommit_buffer: VecDeque<Entry> = VecDeque::with_capacity(128);
        let mut all_commit_req_buffers: Vec<VecDeque<(u64, Entry)>> =
            (0..num_replicas).map(|_| VecDeque::with_capacity(64)).collect();

        while let Some(cmd) = receiver.recv().await {
            match cmd {
                PolicyCommand::Append(entry, _span) => {
                    metrics::increment_gauge!("blitz_queue_size", 1.0);
                    uncommit_buffer.push_back(*entry);
                    let entry = uncommit_buffer.front().unwrap();
                    if let Some(replica_idx) =
                        P::schedule(entry, &all_schedule_context, &mut gctx).await
                    {
                        let entry = uncommit_buffer.pop_front().unwrap();
                        let request_id = entry.request.request_id;
                        #[cfg(feature = "simulator")]
                        super::super::simulator::on_admit(replica_idx, &entry);
                        all_commit_req_buffers[replica_idx]
                            .push_back((request_id, entry));
                    }
                }
                PolicyCommand::NextRequest(replica_idx, response_sender) => {
                    'ineligible: while all_commit_req_buffers[replica_idx].is_empty() {
                        if let Some(entry) = uncommit_buffer.front() {
                            if let Some(tmp_replica_idx) =
                                P::schedule(entry, &all_schedule_context, &mut gctx).await
                            {
                                let entry = uncommit_buffer.pop_front().unwrap();
                                let request_id = entry.request.request_id;
                                #[cfg(feature = "simulator")]
                                super::super::simulator::on_admit(tmp_replica_idx, &entry);
                                all_commit_req_buffers[tmp_replica_idx]
                                    .push_back((request_id, entry));
                            } else {
                                tracing::warn!(
                                    "Replica#{replica_idx} is idle, but scheduler does not assign task to it"
                                );
                                break 'ineligible;
                            }
                        } else {
                            break 'ineligible;
                        }
                    }

                    if let Some((id, mut entry)) =
                        all_commit_req_buffers[replica_idx].pop_front()
                    {
                        entry.batch_time = Some(Instant::now());
                        response_sender.send(Some((id, entry))).unwrap();
                    } else {
                        response_sender.send(None).unwrap();
                    }
                }
            }
        }
    }
}

impl<P> QueuePro for PolicyRunner<P>
where
    P: Policy + Send + Sync + 'static,
{
    #[instrument(skip_all)]
    fn append(&self, entry: Entry) {
        self.queue_sender
            .send(PolicyCommand::Append(Box::new(entry), Span::current()))
            .unwrap();
    }

    #[instrument(skip_all)]
    async fn next_request(&self, replica_id: usize) -> Option<NextRequest> {
        let (tx, rx) = oneshot::channel();
        self.queue_sender
            .send(PolicyCommand::NextRequest(replica_id, tx))
            .unwrap();
        rx.await.unwrap()
    }
}
