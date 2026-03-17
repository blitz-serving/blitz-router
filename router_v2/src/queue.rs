// Request queue module.
//
// This module re-exports the scheduling policy infrastructure from the
// `policies` module and retains the legacy TGI `Queue` for the
// `blitzllm-backend` code path.

// Re-export the policy-based scheduling types used by the rest of the crate.
pub(crate) use crate::policies::{
    Entry, QueuePro, TaskAssigner,
};

// ---------------------------------------------------------------------------
// TGI Queue -- legacy queue used by the blitzllm-backend code path
// ---------------------------------------------------------------------------

use crate::infer::{InferError, InferStreamResponse};
use crate::kvcache::BlockHashState;
use crate::validation::ValidGenerateRequest;

use std::cmp::min;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use nohash_hasher::{BuildNoHashHasher, IntMap};
use pb::generate::v2::*;
use tokio::sync::Mutex;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::{info_span, instrument, Span};

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

/// TGI's Request Queue, kept for the blitzllm-backend code path
#[allow(unused)]
mod text_generation_inference {
    use super::*;
    #[derive(Debug, Clone)]
    pub(crate) struct Queue {
        /// Channel to communicate with the background queue task
        queue_sender: mpsc::UnboundedSender<QueueCommand>,
    }

    impl Queue {
        pub(crate) fn new(
            requires_padding: bool,
            block_size: u32,
            window_size: Option<u32>,
            speculate: u32,
        ) -> Self {
            // Create channel
            let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

            // Launch background queue task
            tokio::spawn(queue_task(
                requires_padding,
                block_size,
                window_size,
                speculate,
                queue_receiver,
            ));

            Self { queue_sender }
        }

        /// Append an entry to the queue
        #[instrument(skip_all)]
        pub(crate) fn append(&self, entry: Entry) {
            // Send append command to the background task managing the state
            // Unwrap is safe here
            self.queue_sender
                .send(QueueCommand::Append(Box::new(entry), Span::current()))
                .unwrap();
        }

        /// Get the number of waiting prefill tokens
        #[instrument(skip_all)]
        pub(crate) async fn waiting_prefill_tokens(&self) -> u32 {
            let (tx, rx) = oneshot::channel();
            self.queue_sender.send(QueueCommand::WaitingPrefillTokens(tx)).unwrap();
            rx.await.unwrap()
        }

        // Get the next batch
        #[instrument(skip_all)]
        pub(crate) async fn next_batch(
            &self,
            min_size: Option<usize>,
            prefill_token_budget: u32,
            token_budget: u32,
            block_budget: Option<u32>,
        ) -> Option<NextBatch> {
            // Create response channel
            let (response_sender, response_receiver) = oneshot::channel();
            // Send next batch command to the background task managing the state
            // Unwrap is safe here
            self.queue_sender
                .send(QueueCommand::NextBatch {
                    min_size,
                    prefill_token_budget,
                    token_budget,
                    block_budget,
                    response_sender,
                    span: Span::current(),
                })
                .unwrap();
            // Await on response channel
            // Unwrap is safe here
            response_receiver.await.unwrap()
        }
    }

    // Background task responsible of the queue state
    async fn queue_task(
        requires_padding: bool,
        block_size: u32,
        window_size: Option<u32>,
        speculate: u32,
        mut receiver: mpsc::UnboundedReceiver<QueueCommand>,
    ) {
        let mut state = State::new(requires_padding, block_size, window_size, speculate);

        while let Some(cmd) = receiver.recv().await {
            match cmd {
                QueueCommand::Append(entry, span) => {
                    span.in_scope(|| state.append(*entry));
                    metrics::increment_gauge!("blitz_queue_size", 1.0);
                }
                QueueCommand::NextBatch {
                    min_size,
                    prefill_token_budget,
                    token_budget,
                    block_budget,
                    response_sender,
                    span,
                } => span.in_scope(|| {
                    let next_batch = state.next_batch(
                        min_size,
                        prefill_token_budget,
                        token_budget,
                        block_budget,
                    );
                    response_sender.send(next_batch).unwrap();
                    metrics::gauge!("blitz_queue_size", state.entries.len() as f64);
                }),
                QueueCommand::WaitingPrefillTokens(sender) => {
                    sender.send(state.waiting_prefill_tokens()).unwrap();
                }
            }
        }
    }

    /// Queue State
    #[derive(Debug)]
    struct State {
        /// Queue entries organized in a Vec
        entries: VecDeque<(u64, Entry)>,

        /// Id of the next batch
        next_batch_id: u64,

        /// Whether the model is using padding
        requires_padding: bool,

        /// Paged Attention block size
        block_size: u32,

        /// Sliding window
        window_size: Option<u32>,

        /// Speculation amount
        speculate: u32,

        /// Waiting prefill tokens in the queue
        waiting_prefill_tokens: u32,
    }

    impl State {
        fn new(
            requires_padding: bool,
            block_size: u32,
            window_size: Option<u32>,
            speculate: u32,
        ) -> Self {
            Self {
                entries: VecDeque::with_capacity(256),

                next_batch_id: 0,
                requires_padding,
                block_size,
                window_size,
                speculate,
                waiting_prefill_tokens: 0,
            }
        }

        /// Append an entry to the queue
        fn append(&mut self, mut entry: Entry) {
            // Create a span that will live as long as the entry is in the queue waiting to be batched
            let queue_span = info_span!(parent: &entry.span, "queued");
            entry.temp_span = Some(queue_span);

            // Update waiting prefill tokens
            self.waiting_prefill_tokens += entry.request.input_length;

            // Push entry in the queue
            self.entries.push_back((entry.request.request_id, entry));
        }

        // Get the next batch
        fn next_batch(
            &mut self,
            min_size: Option<usize>,
            prefill_token_budget: u32,
            token_budget: u32,
            block_budget: Option<u32>,
        ) -> Option<NextBatch> {
            if self.entries.is_empty() {
                return None;
            }

            // Check if we have enough entries
            if let Some(min_size) = min_size {
                if self.entries.len() < min_size {
                    return None;
                }
            }

            // Create span for this batch to add context to inference calls
            let next_batch_span =
                info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
            next_batch_span.follows_from(&Span::current());

            let mut batch_requests = Vec::with_capacity(self.entries.len());
            let mut batch_entries = IntMap::with_capacity_and_hasher(
                self.entries.len(),
                BuildNoHashHasher::default(),
            );

            let mut prefill_tokens: u32 = 0;
            let mut max_tokens = 0;

            let num_all_enties = self.entries.len() as u32;
            let mut consumed_entires: u32 = 0;

            // Pop entries starting from the front of the queue
            while let Some((id, mut entry)) = self.entries.pop_front() {
                consumed_entires += 1;
                let entry_prefill_tokens;
                // Filter entries where the response receiver was dropped (== entries where the request
                // was dropped by the client)
                if entry.response_tx.is_closed() {
                    metrics::increment_counter!("blitz_request_failure", "err" => "dropped");
                    continue;
                }

                let max_request_tokens;
                if self.requires_padding {
                    unimplemented!("Padding is unsupported now!");
                } else {
                    // pad to block size
                    entry_prefill_tokens = entry.request.input_length;
                    prefill_tokens += entry_prefill_tokens;

                    let max_new_tokens = match self.window_size {
                        None => entry.request.stopping_parameters.max_new_tokens,
                        Some(window_size) => min(
                            window_size.saturating_sub(entry.request.input_length),
                            entry.request.stopping_parameters.max_new_tokens,
                        ),
                    };
                    // pad to block size
                    max_request_tokens =
                        (entry.request.input_length + max_new_tokens + self.block_size - 1)
                            / self.block_size
                            * self.block_size;
                }

                if let Some(block_budget) = block_budget {
                    if max_request_tokens + max_tokens > block_budget * self.block_size {
                        self.entries.push_front((id, entry));
                        break;
                    }
                }

                if prefill_tokens > prefill_token_budget
                    || (max_request_tokens + self.speculate) > token_budget
                {
                    // Entry is over budget
                    // Add it back to the front
                    self.entries.push_front((id, entry));
                    break;
                }

                assert!(
                    self.waiting_prefill_tokens >= entry_prefill_tokens,
                    "{} {}",
                    self.waiting_prefill_tokens,
                    entry_prefill_tokens
                );
                self.waiting_prefill_tokens -= entry_prefill_tokens;
                max_tokens += max_request_tokens;

                // Create a new span to link the batch back to this entry
                let entry_batch_span = info_span!(parent: &entry.span, "infer");
                // Add relationships
                next_batch_span.follows_from(&entry_batch_span);
                entry_batch_span.follows_from(&next_batch_span);
                // Update entry
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
                // Set batch_time
                entry.batch_time = Some(Instant::now());
                // Insert in batch_entries IntMap
                batch_entries.insert(id, entry);
            }

            // Empty batch
            if batch_requests.is_empty() {
                return None;
            }

            // Check if our batch is big enough
            if let Some(min_size) = min_size {
                // Batch is too small
                if batch_requests.len() < min_size {
                    // Add back entries to the queue in the correct order
                    for r in batch_requests.into_iter().rev() {
                        let id = r.id;
                        let entry = batch_entries.remove(&id).unwrap();
                        self.entries.push_front((id, entry));
                    }

                    return None;
                }
            }

            // Final batch size
            let size = batch_requests.len() as u32;
            next_batch_span.record("batch_size", size);

            let batch =
                Batch { id: self.next_batch_id, requests: batch_requests, size, max_tokens };
            // Increment batch id
            self.next_batch_id += 1;

            metrics::histogram!("blitz_batch_next_size", batch.size as f64);

            Some((batch_entries, batch, next_batch_span))
        }

        fn waiting_prefill_tokens(&self) -> u32 {
            self.waiting_prefill_tokens
        }
    }

    #[derive(Debug)]
    enum QueueCommand {
        Append(Box<Entry>, Span),
        NextBatch {
            min_size: Option<usize>,
            prefill_token_budget: u32,
            token_budget: u32,
            block_budget: Option<u32>,
            response_sender: oneshot::Sender<Option<NextBatch>>,
            span: Span,
        },
        WaitingPrefillTokens(oneshot::Sender<u32>),
    }
}

pub(crate) use text_generation_inference::Queue;
