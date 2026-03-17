use crate::infer::{InferError, InferStreamResponse};
use crate::kvcache::{BlockHash, BlockHashState};
use crate::queue::queue_plus_plus::{AssignScore, NumHitKvBlock, QueuePlusPlus};
use crate::simulator::batch::Request as SimulatorRequest;
use crate::simulator::metrics::SystemMetrics;
use crate::validation::ValidGenerateRequest;
use crate::{select_best_replica, step, step_w_sampler, weigh_replica};
use crate::{LMetricInc, ScheduleContext, SystemMetric};
use rand::{thread_rng, Rng};
use std::cmp::min;
use std::collections::VecDeque;
use std::ops::AddAssign;
use std::sync::Arc;
use std::time::Duration;
use std::time::SystemTime;
#[cfg(feature = "slo-serve-impl-q")]
use std::usize::MAX;

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
    pub fn append_state(&mut self, new_tokens: &Vec<u32>, tbt: &Duration) -> Option<u64> {
        self.append_tbt_to_tpot(tbt);
        self.block_hash_state.append_tokens(new_tokens)
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
                #[cfg(feature = "simulator-cap")]
                let ScheduleContext { lmetric, block_hash, simulator } =
                    &mut *$all_sctx[replica_idx].lock().await;
                #[cfg(not(feature = "simulator-cap"))]
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

                tracing::info!("Assigning Request_{} to Replica#{}", request.request_id, replica_idx);
                #[cfg(feature = "simulator-cap")]{
                    simulator.add_request(SimulatorRequest {
                        request_id: request.request_id,
                        prompt_len: request.input_length,
                        generation_len: Some(request.stopping_parameters.max_new_tokens),
                        processed_tokens: 0,
                        arrival_time: Some(SystemTime::now()),
                        num_token_per_output: 1,
                        hashes: Some($entry.block_hash_state.block_hashes.clone()),

                        max_generation_len: 16384,
                        hit_token_cnt: 0,
                        ttft: None,
                    }).await;
                }

        //         let result: SystemMetrics = sctx.simulator.query_sim(SimulatorRequest {
        //     request_id: entry.request.request_id,
        //     prompt_len: entry.request.input_length,
        //     generation_len: Some(entry.request.stopping_parameters.max_new_tokens),
        //     processed_tokens: 0,
        //     arrival_time: None,
        //     num_token_per_output: 1,
        //     hashes: Some(entry.block_hash_state.block_hashes.clone()),

        //     max_generation_len: 16384,
        //     hit_token_cnt: 0,
        //     ttft: None,
        // });

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
                #[cfg(not(feature = "simulator-cap"))]
                let ScheduleContext { lmetric, block_hash } =
                    &mut *$all_sctx[replica_idx].lock().await;
                #[cfg(feature = "simulator-cap")]
                let ScheduleContext { lmetric, block_hash, simulator } =
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
                tracing::info!("Assigning Request_{} to Replica#{}", request.request_id, replica_idx);

                let metric_inc = LMetricInc {
                    bs_inc: 1,
                    waiting_reqs_inc: 1,
                    prefill_tokens_inc: new_ntkns,
                    all_tokens_inc: request.input_tokens.len(),
                };
                (*lmetric) += metric_inc;
                #[cfg(feature = "simulator-cap")]{
                    simulator.add_request(SimulatorRequest {
                        request_id: request.request_id,
                        prompt_len: request.input_length,
                        generation_len: Some(request.stopping_parameters.max_new_tokens),
                        processed_tokens: 0,
                        arrival_time: Some(SystemTime::now()),
                        num_token_per_output: 1,
                        hashes: Some($entry.block_hash_state.block_hashes.clone()),

                        max_generation_len: 16384,
                        hit_token_cnt: 0,
                        ttft: None,
                    }).await;
                }
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
                                // all_schedule_context[replica_idx].lock().await.simulator.add_request(SimulatorRequest {
                                //     request_id: entry.request.request_id,
                                //     prompt_len: entry.request.input_length,
                                //     generation_len: Some(entry.request.stopping_parameters.max_new_tokens),
                                //     processed_tokens: 0,
                                //     arrival_time: None,
                                //     num_token_per_output: 1,
                                //     hashes: Some(entry.block_hash_state.block_hashes.clone()),

                                // });
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
                        #[cfg(feature = "simulator-cap")]
                        let ScheduleContext { lmetric, block_hash, simulator } =
                            &mut *all_schedule_context[replica_idx].lock().await;
                        #[cfg(not(feature = "simulator-cap"))]
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
#[cfg(feature = "round-robin-q")]
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
#[cfg(feature = "round-robin-q")]
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

#[cfg(feature = "round-robin-q")]
impl_queue_pro_trait!(RRQueue);
#[cfg(feature = "round-robin-q")]
impl_queue_with_task!(RRQueue);
/// ------ ······· RRQueue ······· ------ ///

/// ------ Join Bounded Most Hit Queue ------ ///
#[cfg(feature = "bounded-most-hit-q")]
use crate::WAITINGT_PREFILL_TOKEN_BOUND;

#[cfg(feature = "bounded-most-hit-q")]
#[derive(Clone)]
pub(super) struct JBoundMostHitQ2 {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

#[cfg(feature = "bounded-most-hit-q")]
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

#[cfg(feature = "bounded-most-hit-q")]
impl_queue_pro_trait!(JBoundMostHitQ2);
#[cfg(feature = "bounded-most-hit-q")]
impl_queue_with_task!(JBoundMostHitQ2);
/// ------ ······· JBMHQueue ······· ------ ///

/// ---- Least Prefill Tokens Queue ----- ///
#[cfg(feature = "least-wait-token-q")]
#[derive(Clone)]
pub(super) struct JLeastWaitTokenQ {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

#[cfg(feature = "least-wait-token-q")]
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

#[cfg(feature = "least-wait-token-q")]
impl_queue_pro_trait!(JLeastWaitTokenQ);
#[cfg(feature = "least-wait-token-q")]
impl_queue_with_task!(JLeastWaitTokenQ);

/// ------ Join Shortest Queue Tuple ------ ///
#[cfg(feature = "join-shortest-q-tuple")]
#[derive(Clone)]
pub(super) struct JShortestQTuple {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

#[cfg(feature = "join-shortest-q-tuple")]
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

#[cfg(feature = "join-shortest-q-tuple")]
impl_queue_pro_trait!(JShortestQTuple);
#[cfg(feature = "join-shortest-q-tuple")]
impl_queue_with_task!(JShortestQTuple);

/// ------ Join Shortest Queue ttft ------ ///
#[derive(Clone)]
#[cfg(feature = "join-shortest-q-ttft")]
pub(super) struct JSQTTFTQueue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

#[cfg(feature = "join-shortest-q-ttft")]
impl QueuePlusPlus for JSQTTFTQueue {
    type QueueContext = EmptyContext;
    type Measure = u32;
    type Weight = ();

    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<Self::Measure, ()>, Option<NumHitKvBlock>)> {
        // use futures::executor::block_on;

        let ScheduleContext { lmetric, block_hash, simulator } = sctx;
        let result: SystemMetrics = sctx.simulator.query_sim(SimulatorRequest {
            request_id: entry.request.request_id,
            prompt_len: entry.request.input_length,
            generation_len: Some(entry.request.stopping_parameters.max_new_tokens),
            processed_tokens: 0,
            arrival_time: Some(SystemTime::now()),
            num_token_per_output: 1,
            hashes: Some(entry.block_hash_state.block_hashes.clone()),

            max_generation_len: 16384,
            hit_token_cnt: 0,
            ttft: None,
        });
        let mut ttft = *result.ttft.last().unwrap();
        ttft = if ttft.is_finite() && ttft > 0.0 { ttft } else { f32::INFINITY };

        tracing::info!(
            "Request_{} estimated ttft: {:.2} ms on Vllm#{}",
            entry.request.request_id,
            ttft,
            _replica_id
        );
        Some((AssignScore::Least(ttft as u32), None))
    }
}

#[cfg(feature = "join-shortest-q-ttft")]
impl_queue_pro_trait!(JSQTTFTQueue);
#[cfg(feature = "join-shortest-q-ttft")]
impl_queue_with_task!(JSQTTFTQueue);
/// --------------------------------- ///

/// ------ Join Shortest Queue Weight ------ ///
#[cfg(feature = "join-shortest-q-weight")]
#[derive(Clone)]
pub(super) struct JShortestQWeight {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

#[cfg(feature = "join-shortest-q-weight")]
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

#[cfg(feature = "join-shortest-q-weight")]
impl_queue_pro_trait!(JShortestQWeight);
#[cfg(feature = "join-shortest-q-weight")]
impl_queue_with_task!(JShortestQWeight);

/// --------------------------------- ///

/// -------- SLO Serve's Impl --------- ///
#[cfg(feature = "slo-serve-impl-q")]
impl NaiiveLattice for (usize) {
    fn meet(&self, other: &Self) -> Self {
        (*self.min(other))
    }

    fn join(&self, other: &Self) -> Self {
        (*self.max(other))
    }

    const TOP: Self = MAX;
    const BOTTOM: Self = 0;
}

#[cfg(feature = "slo-serve-impl-q")]
pub fn slo_sampler(
    // (replica_id, (ttft_headroom, tpot_headroom), hit_nblks)
    all_scores: Vec<(usize, (usize), Option<usize>)>,
    lower_bound: (usize),
    upper_bound: (usize),
    // (replica_id, hitnblks)
) -> (usize, Option<usize>) {
    let mut rng = rand::thread_rng();
    let mut without_violation =
        all_scores.iter().filter(|(_, score, _)| *score == 0).collect::<Vec<_>>();

    if !without_violation.is_empty() {
        // all replicas are within SLOs, uniformly sample one
        let choice_idx = rng.gen_range(0..without_violation.len());
        let (replica_id, _, hit_nblks) = without_violation[choice_idx];
        (*replica_id, *hit_nblks)
    } else {
        // all replicas violate SLOs, weighted sample one
        all_scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|a| return (a.0, a.2))
            .unwrap()
    }
}

#[cfg(feature = "slo-serve-impl-q")]
#[derive(Clone)]
pub(super) struct SloServeImplQ {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

#[cfg(feature = "slo-serve-impl-q")]
impl QueuePlusPlus for SloServeImplQ {
    type QueueContext = EmptyContext;
    type Measure = ();
    type Weight = (usize);

    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<(), Self::Weight>, Option<NumHitKvBlock>)> {
        use crate::TPOT_SLO;

        let ScheduleContext { lmetric, block_hash, simulator } = sctx;

        let systemMetric = simulator.query_sim(SimulatorRequest {
            request_id: entry.request.request_id,
            prompt_len: entry.request.input_length,
            generation_len: Some(entry.request.stopping_parameters.max_new_tokens),
            processed_tokens: 0,
            arrival_time: Some(SystemTime::now()),
            num_token_per_output: 1,
            hashes: Some(entry.block_hash_state.block_hashes.clone()),

            max_generation_len: 16384,
            hit_token_cnt: 0,
            ttft: None,
        });
        let mut ttft = *systemMetric.ttft.last().unwrap();
        ttft = if ttft.is_finite() && ttft > 0.0 { ttft } else { f32::INFINITY };

        tracing::info!(
            "Request_{} estimated ttft: {:.2} ms on Vllm#{}",
            entry.request.request_id,
            ttft,
            _replica_id
        );

        let tpot_violation = lmetric
            .req_tpot
            .iter()
            .filter(|(_, value)| (**value) * 1_000.0 > TPOT_SLO as f32)
            .count();
        let ttft_violation = systemMetric.ttft.iter().filter(|&&v| v > TTFT_SLO as f32).count();
        let violation_weight = ttft_violation + tpot_violation;
        tracing::debug!(
            "Request_{} on Vllm#{} ttft_violation: {}, tpot_violation: {}, total violation weight: {}",
            entry.request.request_id,
            _replica_id,
            ttft_violation,
            tpot_violation,
            violation_weight
        );
        Some((AssignScore::Weighted((violation_weight)), None))
    }
}

#[cfg(feature = "slo-serve-impl-q")]
impl_queue_pro_trait!(SloServeImplQ);
#[cfg(feature = "slo-serve-impl-q")]
impl_queue_with_task_and_sampler!(SloServeImplQ, slo_sampler);

/// -------- LLM-d Impl --------- ///
use crate::{LLMD_ALPHA, LLMD_GAMMA, TPOT_SLO, TTFT_SLO};

#[cfg(feature = "llmd-impl-q")]
impl NaiiveLattice for (f32, f32) {
    fn meet(&self, other: &Self) -> Self {
        (self.0.min(other.0), self.1.min(other.1))
    }

    fn join(&self, other: &Self) -> Self {
        (self.0.max(other.0), self.1.max(other.1))
    }

    const TOP: Self = (f32::INFINITY, f32::INFINITY);
    const BOTTOM: Self = (f32::NEG_INFINITY, f32::NEG_INFINITY);
}

/// 归一化到 [lower, upper] 区间
fn normalize(value: f32, min_v: f32, max_v: f32, lower: f32, upper: f32) -> f32 {
    if !value.is_finite() || !min_v.is_finite() || !max_v.is_finite() {
        return (lower + upper) * 0.5;
    }

    if (max_v - min_v).abs() < f32::EPSILON {
        // 所有值都一样，统一给中间值
        return (lower + upper) * 0.5;
    }

    let mut t = (value - min_v) / (max_v - min_v);
    if t < 0.0 {
        t = 0.0;
    } else if t > 1.0 {
        t = 1.0;
    }
    lower + t * (upper - lower)
}

/// 带权随机，从 weights 中选一个索引
fn weighted_choice(weights: &[f32], rng: &mut impl Rng) -> usize {
    let total: f32 = weights.iter().copied().sum();
    if total <= 0.0 {
        // 退化：全是 0 权重，退回到均匀随机
        return rng.gen_range(0..weights.len());
    }

    let mut r = rng.gen::<f32>() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= *w;
        if r <= 0.0 {
            return i;
        }
    }
    // 理论上不会到这儿，防御性写法
    weights.len() - 1
}

#[cfg(feature = "llmd-impl-q")]
pub fn llmd_sampler(
    // (replica_id, (ttft_headroom, tpot_headroom), hit_nblks)
    all_scores: Vec<(usize, (f32, f32), Option<usize>)>,
    lower_bound: (f32, f32),
    upper_bound: (f32, f32),
    // (replica_id, hitnblks)
) -> (usize, Option<usize>) {
    assert!(!all_scores.is_empty(), "llmd_sampler called with empty all_scores");

    let mut rng = thread_rng();

    // 1. 按 headroom 正负分桶：
    //    ttft >= 0 且 tpot >= 0 认为是 positive bucket，其余为 negative bucket
    let mut positive = Vec::new();
    let mut negative = Vec::new();

    for (id, (ttft_hr, tpot_hr), hit) in all_scores.into_iter() {
        if ttft_hr >= 0.0 && tpot_hr >= 0.0 {
            positive.push((id, (ttft_hr, tpot_hr), hit));
        } else {
            negative.push((id, (ttft_hr, tpot_hr), hit));
        }
    }

    // 2. 先做 cross-bucket 选择：99% 走 positive，1% 走 negative
    let choose_positive = if !positive.is_empty() && !negative.is_empty() {
        rng.gen::<f32>() < 0.99
    } else if !positive.is_empty() {
        true
    } else {
        // 没有 positive，只能走 negative（上层可在外面再接 fallback 逻辑）
        false
    };

    if choose_positive {
        // 3. Positive bucket：按 headroom 加权随机（spread 策略）
        // 3.1 找出 TTFT/TPOT headroom 的 min/max，用于归一化

        let (min_ttft, max_ttft, min_tpot, max_tpot) = positive.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY),
            |(min_ttft, max_ttft, min_tpot, max_tpot), (_, (ttft_hr, tpot_hr), _)| {
                (
                    min_ttft.min(*ttft_hr),
                    max_ttft.max(*ttft_hr),
                    min_tpot.min(*tpot_hr),
                    max_tpot.max(*tpot_hr),
                )
            },
        );

        // 3.2 计算 blendedScore，并作为权重（headroom 越大权重越高 → spread）
        let mut weights = Vec::with_capacity(positive.len());
        for (_, (ttft_hr, tpot_hr), _) in positive.iter() {
            let norm_ttft = normalize(*ttft_hr, min_ttft, max_ttft, 0.0, 1.0);
            let norm_tpot = normalize(*tpot_hr, min_tpot, max_tpot, 0.0, 1.0);

            let blended = LLMD_ALPHA * norm_ttft + (1.0 - LLMD_ALPHA) * norm_tpot;

            // spread：headroom 越大，权重越大
            let weight = blended.max(0.0);
            weights.push(weight);
        }
        tracing::debug!("Positive bucket blended scores: {:?}", weights);

        let idx = weighted_choice(&weights, &mut rng);
        let (id, _, hit) = positive[idx];
        (id, hit)
    } else {
        // 4.2 分别找出 TTFT/TPOT deficit 的 min/max（只在有非零 deficit 的子集上计算）
        let mut min_ttft_def = f32::INFINITY;
        let mut max_ttft_def = f32::NEG_INFINITY;
        let mut min_tpot_def = f32::INFINITY;
        let mut max_tpot_def = f32::NEG_INFINITY;
        let mut ttft_defs = Vec::new();
        let mut tpot_defs = Vec::new();
        for (_, (ttft_hr, tpot_hr), _) in negative.iter() {
            let ttft_def = (-*ttft_hr).max(0.0); // headroom -> deficit
            let tpot_def = (-*tpot_hr).max(0.0);

            ttft_defs.push(ttft_def);
            tpot_defs.push(tpot_def);

            if ttft_def > 0.0 {
                min_ttft_def = min_ttft_def.min(ttft_def);
                max_ttft_def = max_ttft_def.max(ttft_def);
            }
            if tpot_def > 0.0 {
                min_tpot_def = min_tpot_def.min(tpot_def);
                max_tpot_def = max_tpot_def.max(tpot_def);
            }
        }

        // 4.3 计算 blendedBadness，然后把「越小越好」转成「越大越好」的权重
        let mut weights = Vec::with_capacity(negative.len());
        for i in 0..negative.len() {
            let ttft_def = ttft_defs[i];
            let tpot_def = tpot_defs[i];

            let norm_ttft_def = if ttft_def > 0.0 {
                normalize(ttft_def, min_ttft_def, max_ttft_def, 0.0, 1.0)
            } else {
                0.0
            };

            let norm_tpot_def = if tpot_def > 0.0 {
                normalize(tpot_def, min_tpot_def, max_tpot_def, 0.0, 1.0)
            } else {
                0.0
            };

            let blended_bad = LLMD_GAMMA * norm_ttft_def + (1.0 - LLMD_GAMMA) * norm_tpot_def;

            // badness 越小，权重越大：
            // 先找所有 blended_bad 的最大值，再做 (max_bad - bad)
            weights.push(blended_bad);
        }
        tracing::debug!("Negative bucket blended badness: {:?}", weights);

        let max_bad = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max).max(0.0); // 至少 0

        let weights: Vec<f32> =
            weights.into_iter().map(|b| (max_bad - b).max(0.0) + f32::EPSILON).collect();

        let idx = weighted_choice(&weights, &mut rng);
        let (id, _, hit) = negative[idx];
        (id, hit)
    }
}

#[cfg(feature = "llmd-impl-q")]
#[derive(Clone)]
pub(super) struct LLMDImplQ {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

#[cfg(feature = "llmd-impl-q")]
impl QueuePlusPlus for LLMDImplQ {
    type QueueContext = EmptyContext;
    type Measure = ();
    type Weight = (f32, f32);

    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<(), Self::Weight>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash, simulator } = sctx;

        let systemMetric = simulator.query_sim(SimulatorRequest {
            request_id: entry.request.request_id,
            prompt_len: entry.request.input_length,
            generation_len: Some(entry.request.stopping_parameters.max_new_tokens),
            processed_tokens: 0,
            arrival_time: Some(SystemTime::now()),
            num_token_per_output: 1,
            hashes: Some(entry.block_hash_state.block_hashes.clone()),

            max_generation_len: 16384,
            hit_token_cnt: 0,
            ttft: None,
        });

        let ttft_headroom = TTFT_SLO - systemMetric.ttft.last().unwrap();
        let tpot_headroom = TPOT_SLO - lmetric.tpot * 1_000.0;
        let mut ttft = *systemMetric.ttft.last().unwrap();
        ttft = if ttft.is_finite() && ttft > 0.0 { ttft } else { f32::INFINITY };

        tracing::info!(
            "Request_{} estimated ttft: {:.2} ms on Vllm#{}",
            entry.request.request_id,
            ttft,
            _replica_id
        );
        Some((AssignScore::Weighted((ttft_headroom, tpot_headroom)), None))
    }
}

#[cfg(feature = "llmd-impl-q")]
impl_queue_pro_trait!(LLMDImplQ);
#[cfg(feature = "llmd-impl-q")]
impl_queue_with_task_and_sampler!(LLMDImplQ, llmd_sampler);

/// -------- Poly-Serve Impl --------- ///

#[cfg(feature = "poly-serve-impl-q")]
impl NaiiveLattice for (f32, bool, usize) {
    fn meet(&self, other: &Self) -> Self {
        *self
    }

    fn join(&self, other: &Self) -> Self {
        *self
    }

    const TOP: Self = (f32::INFINITY, false, 0);
    const BOTTOM: Self = (f32::NEG_INFINITY, false, 0);
}

#[cfg(feature = "poly-serve-impl-q")]
fn poly_serve_sampler(
    // (replica_id, (ttft_headroom, tpot_headroom), hit_nblks)
    all_scores: Vec<(usize, (f32, bool, usize), Option<usize>)>,
    lower_bound: (f32, bool, usize),
    upper_bound: (f32, bool, usize),
    // (replica_id, hitnblks)
) -> (usize, Option<usize>) {
    // get satisfied replicas: bs > threshold && no tpot violation
    tracing::debug!("All replicas scores: {:?}", all_scores);
    let satisfied_replica = all_scores
        .iter()
        .filter(|(_, (tpot, tpot_violation, bs), _)| {
            !tpot_violation && *bs > POLYSERVE_BS_THRESHOLD
        })
        .collect::<Vec<_>>();

    tracing::debug!("Satisfied replicas: {:?}", satisfied_replica);
    match satisfied_replica.is_empty() {
        // get from satisfied replicas, max tpot but not violation
        false => {
            // sample from satisfied replicas
            return satisfied_replica
                .iter()
                .max_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(replica_id, _, hit_nblks)| (*replica_id, *hit_nblks))
                .unwrap();
        }
        _ => {}
    }

    let mut rng = thread_rng();
    // sample from replicas that bs <= threshold
    let unutilized_replica = all_scores
        .iter()
        .filter(|(_, (_, _, bs), _)| *bs <= POLYSERVE_BS_THRESHOLD)
        .collect::<Vec<_>>();

    tracing::debug!("Unutilized replicas: {:?}", unutilized_replica);
    match unutilized_replica.is_empty() {
        // get from unutilized replicas, max bs
        false => {
            let idx = rng.gen_range(0..unutilized_replica.len());
            let (replica_id, _, hit_nblks) = &unutilized_replica[idx];
            return (*replica_id, *hit_nblks);
        }
        _ => {}
    }

    // sample from all replicas if no satisfied replicas
    let idx = rng.gen_range(0..all_scores.len());
    let (replica_id, _, hit_nblks) = &all_scores[idx];
    (*replica_id, *hit_nblks)
}

#[cfg(feature = "poly-serve-impl-q")]
#[derive(Clone)]
pub(super) struct PolyServeImplQ {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

use crate::{POLYSERVE_BS_THRESHOLD, TPOT_THRESHOLD};

#[cfg(feature = "poly-serve-impl-q")]
impl QueuePlusPlus for PolyServeImplQ {
    type QueueContext = EmptyContext;
    type Measure = ();
    type Weight = (f32, bool, usize);

    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<(), Self::Weight>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash, simulator } = sctx;
        let systemMetric = simulator.query_sim(SimulatorRequest {
            request_id: entry.request.request_id,
            prompt_len: entry.request.input_length,
            generation_len: Some(entry.request.stopping_parameters.max_new_tokens),
            processed_tokens: 0,
            arrival_time: Some(SystemTime::now()),
            num_token_per_output: 1,
            hashes: Some(entry.block_hash_state.block_hashes.clone()),

            max_generation_len: 16384,
            hit_token_cnt: 0,
            ttft: None,
        });

        let tpot_violation = lmetric.tpot * 1_000.0 > TPOT_SLO;
        let ttft_violation = systemMetric.ttft.iter().any(|&v| v > TTFT_SLO as f32);
        let slo_violation = tpot_violation || ttft_violation;

        let mut ttft = *systemMetric.ttft.last().unwrap();
        ttft = if ttft.is_finite() && ttft > 0.0 { ttft } else { f32::INFINITY };

        tracing::info!(
            "Request_{} estimated ttft: {:.2} ms on Vllm#{}",
            entry.request.request_id,
            ttft,
            _replica_id
        );
        Some((AssignScore::Weighted((lmetric.tpot * 1_000.0, slo_violation, lmetric.bs)), None))
    }
}

#[cfg(feature = "poly-serve-impl-q")]
impl_queue_pro_trait!(PolyServeImplQ);
#[cfg(feature = "poly-serve-impl-q")]
impl_queue_with_task_and_sampler!(PolyServeImplQ, poly_serve_sampler);

/// -------- Bailian's Impl --------- ///
#[cfg(feature = "bailian-impl-q")]
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

#[cfg(feature = "bailian-impl-q")]
#[derive(Clone)]
pub(super) struct BailianImplQ {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

#[cfg(feature = "bailian-impl-q")]
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

#[cfg(feature = "bailian-impl-q")]
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

#[cfg(feature = "bailian-impl-q")]
impl_queue_pro_trait!(BailianImplQ);
#[cfg(feature = "bailian-impl-q")]
impl_queue_with_task_and_sampler!(BailianImplQ, bailian_sampler);
/// --------------------------------- ///

/// -------- Random Weighted -------- ///

#[cfg(feature = "random-q")]
#[derive(Clone)]
pub(super) struct RandomQ {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
}

#[cfg(feature = "random-q")]
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

#[cfg(feature = "random-q")]
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

#[cfg(feature = "random-q")]
impl_queue_pro_trait!(RandomQ);
#[cfg(feature = "random-q")]
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

#[cfg(feature = "join-shortest-q-ttft")]
pub(crate) use JSQTTFTQueue as TaskAssigner;

#[cfg(feature = "slo-serve-impl-q")]
pub(crate) use SloServeImplQ as TaskAssigner;

#[cfg(feature = "llmd-impl-q")]
pub(crate) use LLMDImplQ as TaskAssigner;

#[cfg(feature = "poly-serve-impl-q")]
pub(crate) use PolyServeImplQ as TaskAssigner;
