use crate::infer::{InferError, InferStreamResponse};
use crate::kvcache::{BlockHash, BlockHashState};
use crate::queue::queue_plus_plus::{AssignScore, NumHitKvBlock, QueuePlusPlus};
use crate::validation::ValidGenerateRequest;
use crate::{select_best_replica, step, step_w_sampler, weigh_replica};
use crate::{LMetricInc, ScheduleContext};

use rand::{thread_rng, Rng};
use std::cmp::min;
use std::collections::VecDeque;
use std::ops::AddAssign;
use std::sync::Arc;
use std::time::Duration;

use nohash_hasher::{BuildNoHashHasher, IntMap};
use pb::generate::v2::*;
use tokio::sync::Mutex;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::{info_span, instrument, Span};

/// Queue entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: ValidGenerateRequest,
    /// BlockHash
    pub block_hash_state: BlockHashState,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: mpsc::UnboundedSender<Result<InferStreamResponse, InferError>>,
    /// Span that will live as long as entry
    pub span: Span,
    /// Temporary span used as a guard when logging inference, wait times...
    pub temp_span: Option<Span>,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
    /// Number of current generated tokens
    pub generated_token_cnt: usize,
    /// Instant when generated previous token, set after prefilling
    pub prev_token_time: Option<Instant>,
    /// Averaged TPOT, set after first decoding
    pub time_of_per_token: Option<Duration>,
    /// Max TBT
    pub max_time_between_tokens: Duration,
}

impl Entry {
    pub fn append_state(&mut self, new_tokens: &Vec<u32>, tbt: &Duration) {
        self.append_tbt_to_tpot(tbt);
        self.block_hash_state.append_tokens(new_tokens);
    }

    fn append_tbt_to_tpot(&mut self, tbt: &Duration) {
        if let Some(tpot) = self.time_of_per_token {
            let s = self.generated_token_cnt;
            let old_tpot = tpot.as_secs_f32();
            let new_tpot = (old_tpot * (s as f32) + tbt.as_secs_f32()) / {
                self.generated_token_cnt += 1;
                self.generated_token_cnt as f32
            };
            self.time_of_per_token.replace(Duration::from_secs_f32(new_tpot));
        } else {
            self.generated_token_cnt += 1;
            self.time_of_per_token = Some(tbt.clone());
        }
    }
}

trait NaiiveLattice {
    fn meet(&self, other: &Self) -> Self;
    fn join(&self, other: &Self) -> Self;

    const TOP: Self;
    const BOTTOM: Self;
}

impl NaiiveLattice for () {
    fn join(&self, _: &Self) -> Self {
        ()
    }
    fn meet(&self, _: &Self) -> Self {
        ()
    }

    const TOP: Self = ();
    const BOTTOM: Self = ();
}

type NextRequest = (u64, Entry);
type NextBatch = (IntMap<u64, Entry>, Batch, Span);
type ReplicaIndex = usize;

#[derive(Default, Clone)]
struct EmptyContext {}

impl AddAssign<(ReplicaIndex, usize)> for EmptyContext {
    fn add_assign(&mut self, _: (ReplicaIndex, usize)) {}
}

mod queue_plus_plus {
    use std::cmp::Ordering;

    use super::*;

    pub(super) type NumHitKvBlock = usize;

    #[derive(Debug, Clone)]
    pub(super) enum AssignScore<M, W> {
        Greatest(M),
        Least(M),
        Weighted(W),
    }

    impl<M: Ord, W> PartialEq for AssignScore<M, W> {
        fn eq(&self, other: &Self) -> bool {
            use AssignScore::*;
            match (self, other) {
                (Greatest(a), Greatest(b)) => a.eq(b),
                (Least(a), Least(b)) => b.eq(a),
                _ => false, // sample weight is incomparable
            }
        }
    }

    impl<M: Ord, W> PartialOrd for AssignScore<M, W> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            use AssignScore::*;
            match (self, other) {
                (Greatest(a), Greatest(b)) => Some(a.cmp(b)),
                (Least(a), Least(b)) => Some(b.cmp(a)),
                _ => None, // sample weight is incomparable
            }
        }
    }

    impl<M: Copy + Into<usize>, W> AssignScore<M, W> {
        pub fn as_usize(&self) -> Option<usize> {
            match self {
                AssignScore::Greatest(m) => Some((*m).into()),
                AssignScore::Least(m) => Some((*m).into()),
                _ => None,
            }
        }
    }

    impl<M: Copy + Into<f32>, W: Copy + Into<f32>> AssignScore<M, W> {
        pub fn as_f32(&self) -> f32 {
            match self {
                AssignScore::Greatest(m) => (*m).into(),
                AssignScore::Least(m) => (*m).into(),
                AssignScore::Weighted(w) => (*w).into(),
            }
        }
    }

    pub(super) trait QueuePlusPlus {
        type QueueContext: Default + Clone + AddAssign<(ReplicaIndex, usize)>;
        type Measure;
        type Weight;
        fn eligible_with_kvblock_hit(
            replica_id: usize,
            entry: &Entry,
            qctx: &Self::QueueContext,
            sctx: &ScheduleContext,
        ) -> Option<(AssignScore<Self::Measure, Self::Weight>, Option<NumHitKvBlock>)>;
    }

    #[macro_export]
    macro_rules! select_best_replica {
        ($ty:ty; $entry:expr, $qctx:expr, $all_sctx:expr) => {{
            use futures::StreamExt as FuturesStreamExt;
            use tokio_stream::StreamExt as TokioStreamExt;

            let stream = TokioStreamExt::map(
                tokio_stream::iter($all_sctx.iter().cloned().enumerate()),
                |(replica_id, sched_ctx)| {
                    let qctx = $qctx.clone();
                    async move {
                        let sctx = sched_ctx.lock().await;
                        <$ty as QueuePlusPlus>::eligible_with_kvblock_hit(
                            replica_id, &$entry, &qctx, &sctx,
                        )
                        .map(|(score, hit_nblks)| (replica_id, score, hit_nblks))
                    }
                },
            );

            TokioStreamExt::fold(
                TokioStreamExt::filter_map(FuturesStreamExt::buffer_unordered(stream, 8), |x| x),
                None,
                |best_replica: Option<(
                    usize,
                    AssignScore<<$ty as QueuePlusPlus>::Measure, <$ty as QueuePlusPlus>::Weight>,
                    Option<NumHitKvBlock>,
                )>,
                 next_replica| {
                    match best_replica {
                        None => Some(next_replica),
                        Some((_best_id, ref best_score, ref _best_hit))
                            if {
                                let (_next_id, next_score, _next_hit) = &next_replica;
                                next_score > best_score
                            } =>
                        {
                            Some(next_replica)
                        }
                        Some(replica) => Some(replica),
                    }
                },
            )
        }};
    }

    #[macro_export]
    macro_rules! weigh_replica {
        ($ty:ty; $entry:expr, $qctx:expr, $all_sctx:expr) => {{
            use futures::StreamExt as FuturesStreamExt;
            use tokio_stream::StreamExt as TokioStreamExt;

            let stream = TokioStreamExt::map(
                tokio_stream::iter($all_sctx.iter().cloned().enumerate()),
                |(replica_id, sched_ctx)| {
                    let qctx = $qctx.clone();
                    async move {
                        let sctx = sched_ctx.lock().await;
                        <$ty as QueuePlusPlus>::eligible_with_kvblock_hit(
                            replica_id, &$entry, &qctx, &sctx,
                        )
                        .map(|(score, hit_nblks)| (replica_id, score, hit_nblks))
                    }
                },
            );

            // Calculates max and min for weight transformation,
            // while preserving `(replica_id, score, hit_nblks)` stream
            TokioStreamExt::fold(
                TokioStreamExt::filter_map(FuturesStreamExt::buffer_unordered(stream, 8), |x| x),
                (
                    // All scores for future use
                    Vec::<_>::with_capacity($all_sctx.len()),
                    // Lower bound of each component in score
                    <<$ty as QueuePlusPlus>::Weight as NaiiveLattice>::TOP,
                    // Upper bound of each component in score
                    <<$ty as QueuePlusPlus>::Weight as NaiiveLattice>::BOTTOM,
                ),
                |(mut all_scores, mut inf_w, mut sup_w), (replica_id, score, hit_nblks)| {
                    if let AssignScore::Weighted(w) = score {
                        inf_w = inf_w.meet(&w);
                        sup_w = sup_w.join(&w);
                        all_scores.push((replica_id, w, hit_nblks));
                    }

                    (all_scores, inf_w, sup_w)
                },
            )
        }};
    }

    /// Try to make single effective schedule decision
    #[macro_export]
    macro_rules! step {
        ($ty:ty; $entry:expr, $qctx:expr, $all_sctx:expr) => {{
            // Deductive program reasoning at compile time:
            // `$entry` must be an &Entry; `all_sctx` must be a Vec<Arc<Mutex<ScheduleContext>>>
            if let Some((replica_idx, score, hit_nblks)) =
                select_best_replica!($ty; $entry, $qctx, $all_sctx).await
            {
                let request = &$entry.request;
                let ScheduleContext { lmetric, block_hash } =
                    &mut *$all_sctx[replica_idx].lock().await;

                let hit_nblks: usize = if hit_nblks.is_none() {
                    block_hash.get($entry.block_hash_state.get_hashes())
                } else {
                    hit_nblks.unwrap()
                };
                $entry.block_hash_state.set_pred_block_hits(hit_nblks);
                let new_ntkns = request.input_tokens.len()
                    - /*inconsistent=*/hit_nblks * $entry.block_hash_state.get_block_size();

                tracing::debug!(
                    "vLLM#{replica_idx}::Request_{} with {hit_nblks} presumed hit blocks, adding {new_ntkns} new tokens.",
                    request.request_id
                );

                let metric_inc = LMetricInc {
                    bs_inc: 1,
                    waiting_reqs_inc: 1,
                    prefill_tokens_inc: new_ntkns,
                    all_tokens_inc: request.input_tokens.len(),
                };
                (*lmetric) += metric_inc;

                Some(replica_idx)
            } else {
                tracing::info!("Cluster overloaded!");
                None
            }
        }};
    }

    #[macro_export]
    macro_rules! step_w_sampler {
        ($ty:ty, $sampler:expr; $entry:expr, $qctx:expr, $all_sctx:expr) => {{
            // Deductive program reasoning at compile time:
            // `$entry` must be an &Entry; `all_sctx` must be a Vec<Arc<Mutex<ScheduleContext>>>
            let (all_scores, lower_bound, upper_bound) =
                weigh_replica!($ty; $entry, $qctx, $all_sctx).await;

            if all_scores.is_empty() {
                tracing::info!("Cluster overloaded!");
                None
            } else {
                let (replica_idx, hit_nblks) =
                    $sampler(all_scores, lower_bound, upper_bound);

                let request = &$entry.request;
                let ScheduleContext { lmetric, block_hash } =
                    &mut *$all_sctx[replica_idx].lock().await;

                let hit_nblks: usize = if hit_nblks.is_none() {
                    block_hash.get($entry.block_hash_state.get_hashes())
                } else {
                    hit_nblks.unwrap()
                };
                $entry.block_hash_state.set_pred_block_hits(hit_nblks);
                let new_ntkns = request.input_tokens.len()
                    - /*inconsistent=*/hit_nblks * $entry.block_hash_state.get_block_size();

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

                Some(replica_idx)
            }
        }};
    }
}

pub trait QueuePro {
    fn append(&self, entry: Entry);
    async fn next_batch(&self, replica_id: usize) -> Option<NextBatch>;
    async fn next_request(&self, replica_id: usize) -> Option<NextRequest>;
    async fn waiting_requests(&self) -> usize;
    async fn waiting_prefill_tokens(&self) -> usize;
}

enum QueueCommandPro {
    Append(Box<Entry>, Span),
    NextBatch(usize, oneshot::Sender<Option<NextBatch>>),
    NextRequest(usize, oneshot::Sender<Option<NextRequest>>),
    WaitingRequests(oneshot::Sender<usize>),
    WaitingPrefillTokens(oneshot::Sender<usize>),
}

macro_rules! impl_queue_pro_trait {
    ($t:ty) => {
        impl QueuePro for $t
        where
            $t: queue_plus_plus::QueuePlusPlus,
        {
            #[instrument(skip_all)]
            fn append(&self, entry: Entry) {
                // Send append command to the background task managing the state
                // Unwrap is safe here
                self.queue_sender
                    .send(QueueCommandPro::Append(Box::new(entry), Span::current()))
                    .unwrap();
            }

            #[instrument(skip_all)]
            async fn next_batch(&self, replica_id: usize) -> Option<NextBatch> {
                let (tx, rx) = oneshot::channel();
                self.queue_sender.send(QueueCommandPro::NextBatch(replica_id, tx)).unwrap();
                rx.await.unwrap()
            }

            #[instrument(skip_all)]
            async fn next_request(&self, replica_id: usize) -> Option<NextRequest> {
                let (tx, rx) = oneshot::channel();
                self.queue_sender.send(QueueCommandPro::NextRequest(replica_id, tx)).unwrap();
                rx.await.unwrap()
            }

            #[instrument(skip_all)]
            async fn waiting_prefill_tokens(&self) -> usize {
                let (tx, rx) = oneshot::channel();
                self.queue_sender.send(QueueCommandPro::WaitingPrefillTokens(tx)).unwrap();
                rx.await.unwrap()
            }

            #[instrument(skip_all)]
            async fn waiting_requests(&self) -> usize {
                let (tx, rx) = oneshot::channel();
                self.queue_sender.send(QueueCommandPro::WaitingRequests(tx)).unwrap();
                rx.await.unwrap()
            }
        }
    };
}

macro_rules! impl_queue_with_task {
    ($t:ty) => {
        impl $t
        where
            $t: QueuePro + queue_plus_plus::QueuePlusPlus,
        {
            #[allow(unused)]
            pub fn new(
                num_replicas: usize,
                all_schedule_context: Vec<Arc<Mutex<ScheduleContext>>>,
            ) -> Self {
                let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

                tokio::spawn(<$t>::queue_task(
                    num_replicas,
                    queue_receiver,
                    all_schedule_context,
                ));

                Self { queue_sender }
            }

            #[allow(unused)]
            async fn queue_task(
                num_replicas: usize,
                mut receiver: mpsc::UnboundedReceiver<QueueCommandPro>,
                all_schedule_context: Vec<Arc<Mutex<ScheduleContext>>>,
            ) {
                // create inner data structures
                let mut queue_context = <<$t as queue_plus_plus::QueuePlusPlus>::QueueContext as Default>::default();
                let mut next_batch_id = 0;
                let mut uncommit_buffer = VecDeque::with_capacity(128);
                let mut all_commit_req_buffers: Vec<VecDeque<(u64, Entry)>> =
                    (0..num_replicas).map(|_| VecDeque::with_capacity(64)).collect();

                while let Some(cmd) = receiver.recv().await {
                    match cmd {
                        QueueCommandPro::Append(entry, span) => {
                            metrics::increment_gauge!("blitz_queue_size", 1.0);
                            uncommit_buffer.push_back(*entry);
                            // Make schedule system step
                            // precond: ¬ empty(uncommit_buffer)
                            let entry = uncommit_buffer.front().unwrap();
                            if let Some(replica_idx) =
                                step!($t; entry, queue_context.clone(), all_schedule_context)
                            {
                                let entry = uncommit_buffer.pop_front().unwrap();
                                all_commit_req_buffers[replica_idx]
                                    .push_back((entry.request.request_id, entry));
                                queue_context += (replica_idx, num_replicas);
                            }
                        }
                        QueueCommandPro::NextBatch(replica_idx, response_sender) => {
                            'Ineligible: while all_commit_req_buffers[replica_idx].is_empty() {
                                if let Some(entry) = uncommit_buffer.front() {
                                    if let Some(tmp_replica_idx) =
                                        step!($t; entry, queue_context, all_schedule_context)
                                    {
                                        let entry = uncommit_buffer.pop_front().unwrap();
                                        all_commit_req_buffers[tmp_replica_idx]
                                            .push_back((entry.request.request_id, entry));
                                    } else {
                                        // postcond: all replicas are ineligible
                                        // TODO: use refinement to check
                                        tracing::warn!("Replica#{replica_idx} is idle, but scheduler does not assign task to it");
                                        break 'Ineligible;
                                    }
                                } else {
                                    // postcond: there is no incoming requests
                                    break 'Ineligible;
                                }
                            }

                            let entries = all_commit_req_buffers[replica_idx].drain(..).collect::<Vec<_>>();

                            if entries.is_empty() {
                                response_sender.send(None).unwrap();
                                continue;
                            }

                            // Create span for this batch to add context to inference calls
                            let next_batch_span =
                                info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
                            next_batch_span.follows_from(&Span::current());

                            // Construct response
                            let mut batch_requests = Vec::with_capacity(entries.len() / num_replicas);
                            let mut batch_entries = IntMap::with_capacity_and_hasher(
                                entries.len() / num_replicas,
                                BuildNoHashHasher::default(),
                            );

                            for (id, mut entry) in entries {
                                // Filter entries where the response receiver was dropped (== entries where the request
                                // was dropped by the client)
                                if entry.response_tx.is_closed() {
                                    metrics::increment_counter!("blitz_request_failure", "err" => "dropped");
                                    continue;
                                }

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

                            // Finalize batch
                            let size = batch_requests.len() as u32;
                            next_batch_span.record("batch_size", size);

                            let batch = Batch {
                                id: next_batch_id,
                                requests: batch_requests,
                                size,
                                max_tokens: 0, // neglect it!
                            };
                            // Update queue state
                            next_batch_id += 1;

                            metrics::histogram!("blitz_batch_next_size", batch.size as f64);

                            response_sender.send(Some((batch_entries, batch, next_batch_span))).unwrap();
                        }
                        QueueCommandPro::NextRequest(replica_idx, response_sender) => {
                            'Ineligible: while all_commit_req_buffers[replica_idx].is_empty() {
                                if let Some(entry) = uncommit_buffer.front() {
                                    if let Some(tmp_replica_idx) =
                                        step!($t; entry, queue_context, all_schedule_context)
                                    {
                                        let entry = uncommit_buffer.pop_front().unwrap();
                                        all_commit_req_buffers[tmp_replica_idx]
                                            .push_back((entry.request.request_id, entry));
                                    } else {
                                        // postcond: all replicas are ineligible
                                        // TODO: use refinement to check
                                        tracing::warn!("Replica#{replica_idx} is idle, but scheduler does not assign task to it");
                                        break 'Ineligible;
                                    }
                                } else {
                                    // postcond: there is no incoming requests
                                    break 'Ineligible;
                                }
                            }

                            if let Some((id, mut entry)) = all_commit_req_buffers[replica_idx].pop_front() {
                                // Set batch_time
                                entry.batch_time = Some(Instant::now());

                                response_sender
                                    .send(Some((id, entry)))
                                    .unwrap();
                            } else {
                                response_sender.send(None).unwrap();
                            }
                        }
                        QueueCommandPro::WaitingPrefillTokens(sender) => {
                            unimplemented!("undecided API")
                        }
                        QueueCommandPro::WaitingRequests(sender) => {
                            unimplemented!("undecided API")
                        }
                    }
                }
            }
        }
    };
}

macro_rules! impl_queue_with_task_and_sampler {
    ($t:ty, $sampler:expr) => {
        impl $t
        where
            $t: QueuePro + queue_plus_plus::QueuePlusPlus,
        {
            #[allow(unused)]
            pub fn new(
                num_replicas: usize,
                all_schedule_context: Vec<Arc<Mutex<ScheduleContext>>>,
            ) -> Self {
                let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

                tokio::spawn(<$t>::queue_task(
                    num_replicas,
                    queue_receiver,
                    all_schedule_context,
                ));

                Self { queue_sender }
            }

            #[allow(unused)]
            async fn queue_task(
                num_replicas: usize,
                mut receiver: mpsc::UnboundedReceiver<QueueCommandPro>,
                all_schedule_context: Vec<Arc<Mutex<ScheduleContext>>>,
            ) {
                // create inner data structures
                let mut queue_context = <<$t as queue_plus_plus::QueuePlusPlus>::QueueContext as Default>::default();
                let mut next_batch_id = 0;
                let mut uncommit_buffer = VecDeque::with_capacity(128);
                let mut all_commit_req_buffers: Vec<VecDeque<(u64, Entry)>> =
                    (0..num_replicas).map(|_| VecDeque::with_capacity(64)).collect();

                while let Some(cmd) = receiver.recv().await {
                    match cmd {
                        QueueCommandPro::Append(entry, span) => {
                            metrics::increment_gauge!("blitz_queue_size", 1.0);
                            uncommit_buffer.push_back(*entry);
                            // Make schedule system step
                            // precond: ¬ empty(uncommit_buffer)
                            let entry = uncommit_buffer.front().unwrap();
                            if let Some(replica_idx) =
                                step_w_sampler!($t, $sampler; entry, queue_context.clone(), all_schedule_context)
                            {
                                let entry = uncommit_buffer.pop_front().unwrap();
                                all_commit_req_buffers[replica_idx]
                                    .push_back((entry.request.request_id, entry));
                                queue_context += (replica_idx, num_replicas);
                            }
                        }
                        QueueCommandPro::NextBatch(replica_idx, response_sender) => {
                            'Ineligible: while all_commit_req_buffers[replica_idx].is_empty() {
                                if let Some(entry) = uncommit_buffer.front() {
                                    if let Some(tmp_replica_idx) =
                                        step_w_sampler!($t, $sampler; entry, queue_context, all_schedule_context)
                                    {
                                        let entry = uncommit_buffer.pop_front().unwrap();
                                        all_commit_req_buffers[tmp_replica_idx]
                                            .push_back((entry.request.request_id, entry));
                                    } else {
                                        // postcond: all replicas are ineligible
                                        // TODO: use refinement to check
                                        tracing::warn!("Replica#{replica_idx} is idle, but scheduler does not assign task to it");
                                        break 'Ineligible;
                                    }
                                } else {
                                    // postcond: there is no incoming requests
                                    break 'Ineligible;
                                }
                            }

                            let entries = all_commit_req_buffers[replica_idx].drain(..).collect::<Vec<_>>();

                            if entries.is_empty() {
                                response_sender.send(None).unwrap();
                                continue;
                            }

                            // Create span for this batch to add context to inference calls
                            let next_batch_span =
                                info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
                            next_batch_span.follows_from(&Span::current());

                            // Construct response
                            let mut batch_requests = Vec::with_capacity(entries.len() / num_replicas);
                            let mut batch_entries = IntMap::with_capacity_and_hasher(
                                entries.len() / num_replicas,
                                BuildNoHashHasher::default(),
                            );

                            for (id, mut entry) in entries {
                                // Filter entries where the response receiver was dropped (== entries where the request
                                // was dropped by the client)
                                if entry.response_tx.is_closed() {
                                    metrics::increment_counter!("blitz_request_failure", "err" => "dropped");
                                    continue;
                                }

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

                            // Finalize batch
                            let size = batch_requests.len() as u32;
                            next_batch_span.record("batch_size", size);

                            let batch = Batch {
                                id: next_batch_id,
                                requests: batch_requests,
                                size,
                                max_tokens: 0, // neglect it!
                            };
                            // Update queue state
                            next_batch_id += 1;

                            metrics::histogram!("blitz_batch_next_size", batch.size as f64);

                            response_sender.send(Some((batch_entries, batch, next_batch_span))).unwrap();
                        }
                        QueueCommandPro::NextRequest(replica_idx, response_sender) => {
                            'Ineligible: while all_commit_req_buffers[replica_idx].is_empty() {
                                if let Some(entry) = uncommit_buffer.front() {
                                    if let Some(tmp_replica_idx) =
                                        step!($t; entry, queue_context, all_schedule_context)
                                    {
                                        let entry = uncommit_buffer.pop_front().unwrap();
                                        all_commit_req_buffers[tmp_replica_idx]
                                            .push_back((entry.request.request_id, entry));
                                    } else {
                                        // postcond: all replicas are ineligible
                                        // TODO: use refinement to check
                                        tracing::warn!("Replica#{replica_idx} is idle, but scheduler does not assign task to it");
                                        break 'Ineligible;
                                    }
                                } else {
                                    // postcond: there is no incoming requests
                                    break 'Ineligible;
                                }
                            }

                            if let Some((id, mut entry)) = all_commit_req_buffers[replica_idx].pop_front() {
                                // Set batch_time
                                entry.batch_time = Some(Instant::now());

                                response_sender
                                    .send(Some((id, entry)))
                                    .unwrap();
                            } else {
                                response_sender.send(None).unwrap();
                            }
                        }
                        QueueCommandPro::WaitingPrefillTokens(sender) => {
                            unimplemented!("undecided API")
                        }
                        QueueCommandPro::WaitingRequests(sender) => {
                            unimplemented!("undecided API")
                        }
                    }
                }
            }
        }
    };
}

#[derive(Default, Clone)]
struct RRContext {
    next_replica_id: usize,
}

impl AddAssign<(ReplicaIndex, usize)> for RRContext {
    fn add_assign(&mut self, rhs: (ReplicaIndex, usize)) {
        let (_, num_replicas) = rhs;
        self.next_replica_id = (self.next_replica_id + 1) % num_replicas;
    }
}

/// --- Round Robin Queue for Debugging --- ///
#[derive(Clone)]
pub(crate) struct DbgRRQueue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

impl QueuePlusPlus for DbgRRQueue {
    type QueueContext = RRContext;
    type Measure = ();
    type Weight = ();

    fn eligible_with_kvblock_hit(
        replica_id: usize,
        _entry: &Entry,
        qctx: &RRContext,
        _sctx: &ScheduleContext,
    ) -> Option<(AssignScore<(), ()>, Option<NumHitKvBlock>)> {
        if qctx.next_replica_id == replica_id {
            Some((AssignScore::Least(()), None))
        } else {
            None
        }
    }
}

impl_queue_pro_trait!(DbgRRQueue);

impl DbgRRQueue {
    pub fn new(
        num_replicas: usize,
        all_schedule_context: Vec<Arc<Mutex<ScheduleContext>>>,
    ) -> Self {
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        tokio::spawn(DbgRRQueue::queue_task(num_replicas, queue_receiver, all_schedule_context));

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
        let mut queue_ctx = RRContext { next_replica_id: 0 };
        let mut all_commit_req_buffers: Vec<VecDeque<(u64, Entry)>> =
            (0..num_replicas).map(|_| VecDeque::with_capacity(64)).collect();

        while let Some(cmd) = receiver.recv().await {
            match cmd {
                QueueCommandPro::Append(entry, span) => {
                    metrics::increment_gauge!("blitz_queue_size", 1.0);
                    // HACK: always eligible -> bypass eligibility checking
                    let replica_idx = queue_ctx.next_replica_id;
                    let request = &entry.request;

                    // postcond: assignment is commited
                    // brief: update thread local state
                    queue_ctx.next_replica_id = (queue_ctx.next_replica_id + 1) % num_replicas;
                    // brief: calculate and apply metric increments here
                    {
                        let ScheduleContext { lmetric, block_hash } =
                            &mut *all_schedule_context[replica_idx].lock().await;
                        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
                        entry.block_hash_state.set_pred_block_hits(hit_nblks);
                        let new_ntkns = request.input_tokens.len()
                            - /*inconsistent*/hit_nblks * entry.block_hash_state.get_block_size();
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

                    let committed_req_buf = all_commit_req_buffers.get_mut(replica_idx).unwrap();
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

                    // Construct response

                    // Create span for this batch to add context to inference calls
                    let next_batch_span =
                        info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
                    next_batch_span.follows_from(&Span::current());

                    let mut batch_requests = Vec::with_capacity(entries.len());
                    let mut batch_entries = IntMap::with_capacity_and_hasher(
                        entries.len(),
                        BuildNoHashHasher::default(),
                    );

                    for (id, mut entry) in entries {
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

                    // Final batch size
                    let size = batch_requests.len() as u32;
                    next_batch_span.record("batch_size", size);

                    // deprecated: unused request, initially prepared for TGI backend
                    let batch = Batch {
                        id: next_batch_id,
                        requests: batch_requests,
                        size,
                        max_tokens: 0, // neglect it!
                    };
                    // Increment batch id
                    next_batch_id += 1;

                    metrics::histogram!("blitz_batch_next_size", batch.size as f64);

                    response_sender.send(Some((batch_entries, batch, next_batch_span))).unwrap();
                }
                QueueCommandPro::NextRequest(replica_idx, response_sender) => {
                    // HACK: always eligible -> always successfully commits -> bypass progressiveness check
                    if let Some((id, mut entry)) =
                        all_commit_req_buffers.get_mut(replica_idx).unwrap().pop_front()
                    {
                        // Set batch_time
                        entry.batch_time = Some(Instant::now());

                        response_sender.send(Some((id, entry))).unwrap();
                    } else {
                        response_sender.send(None).unwrap();
                    }
                }
                QueueCommandPro::WaitingPrefillTokens(sender) => {
                    unimplemented!();
                }
                QueueCommandPro::WaitingRequests(sender) => {
                    unimplemented!();
                }
            }
        }
    }
}
/// ------ ······· DbgRRQueue ······· ------ ///

/// ----- Round Robin Queue as Demo ----- ///
#[derive(Clone)]
pub(crate) struct RRQueue {
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

/// Instruction for users:
///
/// step#1: specify queue task local schedule context, use `EmptyContext` for doing nothing;
/// step#2: specify metric for selection,
///         `Measure` is the metric Type for comparing, `Greatest` for select max, `Least` vice versa;
///         `Weight` is the metric Type for sampling;
/// step#3: define the eligibility rule: if ineligible, return None; else return metric
///
/// finally, use 2 impl macros for automatic code generation
impl QueuePlusPlus for RRQueue {
    type QueueContext = RRContext;
    type Measure = ();
    type Weight = ();

    fn eligible_with_kvblock_hit(
        replica_id: usize,
        _entry: &Entry,
        qctx: &RRContext,
        _sctx: &ScheduleContext,
    ) -> Option<(AssignScore<(), ()>, Option<NumHitKvBlock>)> {
        if qctx.next_replica_id == replica_id {
            Some((AssignScore::Least(()), None))
        } else {
            None
        }
    }
}

impl_queue_pro_trait!(RRQueue);
impl_queue_with_task!(RRQueue);
/// ------ ······· RRQueue ······· ------ ///

/// ------ Join Bounded Most Hit Queue ------ ///
use crate::WAITINGT_PREFILL_TOKEN_BOUND;

#[derive(Clone)]
pub(super) struct JBoundMostHitQ2 {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

impl QueuePlusPlus for JBoundMostHitQ2 {
    type QueueContext = EmptyContext;
    type Measure = usize;
    type Weight = ();

    /// # Returns: hit tokens!
    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<Self::Measure, ()>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash } = sctx;

        if lmetric.prefill_tokens.max(0) as usize >= WAITINGT_PREFILL_TOKEN_BOUND {
            None
        } else {
            // postcond: current waiting prefill tokens are within the bound
            //            -> this schedule shall succeed
            // Calculates hit prefill tokens
            let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());

            Some((AssignScore::Greatest(hit_nblks), Some(hit_nblks)))
        }
    }
}

impl_queue_pro_trait!(JBoundMostHitQ2);
impl_queue_with_task!(JBoundMostHitQ2);
/// ------ ······· JBMHQueue ······· ------ ///

/// ---- Least Prefill Tokens Queue ----- ///
#[derive(Clone)]
pub(super) struct JLeastWaitTokenQ {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

impl QueuePlusPlus for JLeastWaitTokenQ {
    type QueueContext = EmptyContext;
    type Measure = usize;
    type Weight = ();

    /// # Returns: hit tokens!
    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<Self::Measure, ()>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash } = sctx;
        let request = &entry.request;

        let num_waiting_tokens = lmetric.prefill_tokens;
        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
        let num_new_tokens =
            request.input_tokens.len() - hit_nblks * entry.block_hash_state.get_block_size();

        Some((
            AssignScore::Least(num_waiting_tokens.max(0) as usize + num_new_tokens),
            Some(hit_nblks),
        ))
    }
}

impl_queue_pro_trait!(JLeastWaitTokenQ);
impl_queue_with_task!(JLeastWaitTokenQ);

/// ------ Join Shortest Queue Tuple ------ ///
#[derive(Clone)]
pub(super) struct JShortestQTuple {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

impl QueuePlusPlus for JShortestQTuple {
    type QueueContext = EmptyContext;
    type Measure = (usize, usize);
    type Weight = ();

    /// # Returns: hit tokens!
    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<Self::Measure, ()>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash } = sctx;

        let num_waiting_requests = lmetric.waiting_reqs;
        let num_running_requests = lmetric.bs - lmetric.waiting_reqs;
        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
        Some((AssignScore::Least((num_waiting_requests, num_running_requests)), Some(hit_nblks)))
    }
}

impl_queue_pro_trait!(JShortestQTuple);
impl_queue_with_task!(JShortestQTuple);

/// --------------------------------- ///

/// ------ Join Shortest Queue Weight ------ ///
#[derive(Clone)]
pub(super) struct JShortestQWeight {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

impl QueuePlusPlus for JShortestQWeight {
    type QueueContext = EmptyContext;
    type Measure = usize;
    type Weight = ();

    /// # Returns: hit tokens!
    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<Self::Measure, ()>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash } = sctx;

        let num_waiting_requests = lmetric.waiting_reqs;
        let num_running_requests = lmetric.bs - lmetric.waiting_reqs;
        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
        Some((AssignScore::Least(num_waiting_requests * 4 + num_running_requests), Some(hit_nblks)))
    }
}

impl_queue_pro_trait!(JShortestQWeight);
impl_queue_with_task!(JShortestQWeight);

/// --------------------------------- ///

/// -------- Bailian's Impl --------- ///
use crate::{BAILIAN_ALPHA, BAILIAN_BETA, BAILIAN_GAMMA};

impl NaiiveLattice for (f32, f32, f32) {
    fn meet(&self, other: &Self) -> Self {
        (self.0.min(other.0), self.1.min(other.1), self.2.min(other.2))
    }

    fn join(&self, other: &Self) -> Self {
        (self.0.max(other.0), self.1.max(other.1), self.2.max(other.2))
    }

    const TOP: Self = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    const BOTTOM: Self = (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
}

#[derive(Clone)]
pub(super) struct BailianImplQ {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

fn bailian_sampler(
    all_scores: Vec<(usize, (f32, f32, f32), Option<usize>)>,
    lower_bound: (f32, f32, f32),
    upper_bound: (f32, f32, f32),
) -> (usize, Option<usize>) {
    let eps = 1e-6f32;
    let dx0 = (upper_bound.0 - lower_bound.0).abs().max(eps);
    let dx1 = (upper_bound.1 - lower_bound.1).abs().max(eps);
    let dx2 = (upper_bound.2 - lower_bound.2).abs().max(eps);

    let mut total_weight = 0.0f32;
    let mut norm_scores = Vec::with_capacity(all_scores.len());

    tracing::debug!("All scores: {:?}", all_scores);

    for (replica_id, (hit_ratio, nreqs, ntkns), hit_nblks) in all_scores.into_iter() {
        // 归一化到 [0,1]
        let n0 = ((hit_ratio - lower_bound.0) / dx0).clamp(0.0, 1.0);
        let n1 = ((upper_bound.1 - nreqs) / dx1).clamp(0.0, 1.0);
        let n2 = ((upper_bound.2 - ntkns) / dx2).clamp(0.0, 1.0);

        // 线性组合 (可以调权重，比如 0.5,0.3,0.2)
        let p = n0 * BAILIAN_ALPHA + n1 * BAILIAN_BETA + n2 * BAILIAN_GAMMA;
        let p = if p.is_finite() && p > 0.0 { p } else { 0.0 };

        total_weight += p;
        norm_scores.push((replica_id, p, hit_nblks));
    }

    let mut rng = thread_rng();
    let mut r = rng.gen::<f32>() * total_weight;

    tracing::debug!("Normed scores: {:?}; r={r}", norm_scores);

    let mut ret = (0, None);
    for (replica_id, p, hit_nblks) in norm_scores {
        if r <= p {
            return (replica_id, hit_nblks);
        }
        r -= p;
        ret = (replica_id, hit_nblks);
    }

    ret
}

impl QueuePlusPlus for BailianImplQ {
    type QueueContext = EmptyContext;
    type Measure = ();
    type Weight = (f32, f32, f32);

    /// # Returns: hit tokens!
    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<(), Self::Weight>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash } = sctx;
        let request = &entry.request;

        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
        let hit_ratio = (hit_nblks * entry.block_hash_state.get_block_size()) as f32
            / request.input_tokens.len() as f32;
        let num_requests = lmetric.bs as f32;
        let num_tokens = lmetric.all_tokens as f32;

        Some((AssignScore::Weighted((hit_ratio, num_requests, num_tokens)), Some(hit_nblks)))
    }
}

impl_queue_pro_trait!(BailianImplQ);
impl_queue_with_task_and_sampler!(BailianImplQ, bailian_sampler);
/// --------------------------------- ///

/// -------- Random Weighted -------- ///

#[derive(Clone)]
pub(super) struct RandomQ {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

impl QueuePlusPlus for RandomQ {
    type QueueContext = EmptyContext;
    type Measure = ();
    type Weight = (); // `()` is also an lattice, hahaha!

    /// # Returns: hit tokens!
    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        _entry: &Entry,
        _qctx: &Self::QueueContext,
        _sctx: &ScheduleContext,
    ) -> Option<(AssignScore<(), Self::Weight>, Option<NumHitKvBlock>)> {
        Some((AssignScore::Weighted(()), None))
    }
}

fn random_sampler(
    all_scores: Vec<(usize, (), Option<usize>)>, // (replica_id, Weight, kvcache_hit)
    _lower_bound: (),
    _upper_bound: (),
) -> (usize, Option<usize>) {
    let mut rng = thread_rng();
    let x = rng.gen_range(0..all_scores.len());
    let (id, _, hit) = all_scores.get(x).copied().unwrap();
    (id, hit)
}

impl_queue_pro_trait!(RandomQ);
impl_queue_with_task_and_sampler!(RandomQ, random_sampler);
/// --------------------------------- ///

/// TGI' Request Queue, just keep for Memo
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
            self.queue_sender.send(QueueCommand::Append(Box::new(entry), Span::current())).unwrap();
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
            let mut batch_entries =
                IntMap::with_capacity_and_hasher(self.entries.len(), BuildNoHashHasher::default());

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

#[cfg(feature = "bailian-impl-q")]
pub(crate) use BailianImplQ as TaskAssigner;
#[cfg(feature = "bounded-most-hit-q")]
pub(crate) use JBoundMostHitQ2 as TaskAssigner;
#[cfg(feature = "least-wait-token-q")]
pub(crate) use JLeastWaitTokenQ as TaskAssigner;
#[cfg(feature = "join-shortest-q-tuple")]
pub(crate) use JShortestQTuple as TaskAssigner;
#[cfg(feature = "join-shortest-q-weight")]
pub(crate) use JShortestQWeight as TaskAssigner;
#[cfg(feature = "round-robin-q")]
pub(crate) use RRQueue as TaskAssigner;
#[cfg(feature = "random-q")]
pub(crate) use RandomQ as TaskAssigner;
