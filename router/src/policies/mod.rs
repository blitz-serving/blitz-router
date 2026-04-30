// Scheduling policies for replica selection.
//
// This module replaces the former macro-based codegen in `queue.rs` with a
// trait-based design.  Every policy implements `QueuePlusPlus` (scoring) and
// optionally provides a `Sampler`.  The generic `QueueRunner<P>` struct
// supplies the background task, `QueuePro` implementation, and constructor
// for free -- no macros required.

pub(crate) mod aibrix;
pub(crate) mod bailian;
pub(crate) mod bounded_most_hit;
pub(crate) mod dynamo;
pub(crate) mod least_wait_token;
pub(crate) mod lmetric;
pub(crate) mod preble;
pub(crate) mod random;
pub(crate) mod round_robin;
pub(crate) mod shortest_q_weight;

// Re-export the concrete types so the rest of the crate can refer to them by
// their original short names.
pub(crate) use aibrix::AibrixQ;
pub(crate) use bailian::BailianImplQ;
pub(crate) use bounded_most_hit::JBoundMostHitQ2;
pub(crate) use dynamo::{DynamoQ, DynamoDecodeQ};
pub(crate) use least_wait_token::JLeastWaitTokenQ;
pub(crate) use lmetric::LmetricQ;
pub(crate) use preble::PrebleQ;
pub(crate) use random::RandomQ;
pub(crate) use round_robin::RRQueue;
pub(crate) use shortest_q_weight::JShortestQWeight;

use crate::infer::{InferError, InferStreamResponse};
use crate::kvcache::{BlockHash, BlockHashState};
use crate::validation::ValidGenerateRequest;
use crate::{LMetricInc, ScheduleContext};

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

// ---------------------------------------------------------------------------
// Common type aliases
// ---------------------------------------------------------------------------

pub(crate) type NextRequest = (u64, Entry);
pub(crate) type NextBatch = (IntMap<u64, Entry>, Batch, Span);
pub(crate) type ReplicaIndex = usize;

// ---------------------------------------------------------------------------
// Entry -- a single queued request
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// NaiiveLattice -- min/max bounds for weighted scoring
// ---------------------------------------------------------------------------

pub(crate) trait NaiiveLattice {
    fn meet(&self, other: &Self) -> Self;
    fn join(&self, other: &Self) -> Self;

    const TOP: Self;
    const BOTTOM: Self;
}

impl NaiiveLattice for () {
    fn join(&self, _: &Self) -> Self {}
    fn meet(&self, _: &Self) -> Self {}

    const TOP: Self = ();
    const BOTTOM: Self = ();
}

// ---------------------------------------------------------------------------
// EmptyContext -- used by policies that need no extra queue-local state
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
pub(crate) struct EmptyContext {}

impl AddAssign<(ReplicaIndex, usize)> for EmptyContext {
    fn add_assign(&mut self, _: (ReplicaIndex, usize)) {}
}

// ---------------------------------------------------------------------------
// RRContext -- round-robin queue-local state
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
pub(crate) struct RRContext {
    pub next_replica_id: usize,
}

impl AddAssign<(ReplicaIndex, usize)> for RRContext {
    fn add_assign(&mut self, rhs: (ReplicaIndex, usize)) {
        let (_, num_replicas) = rhs;
        self.next_replica_id = (self.next_replica_id + 1) % num_replicas;
    }
}

// ---------------------------------------------------------------------------
// AssignScore -- the scoring enum
// ---------------------------------------------------------------------------

pub(crate) type NumHitKvBlock = usize;

#[derive(Debug, Clone)]
pub(crate) enum AssignScore<M, W> {
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
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
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

// ---------------------------------------------------------------------------
// QueuePlusPlus -- the core policy trait
// ---------------------------------------------------------------------------

/// Core trait that every scheduling policy must implement.
///
/// * `QueueContext` - per-queue state (e.g. round-robin counter).
/// * `Measure`     - deterministic metric for `Greatest` / `Least` selection.
/// * `Weight`      - multi-dimensional metric for weighted sampling.
pub(crate) trait QueuePlusPlus {
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

// ---------------------------------------------------------------------------
// Sampler -- optional weighted-sampling callback
// ---------------------------------------------------------------------------

/// A sampler takes the collected `(replica_id, weight, hit_nblks)` triples
/// together with global lower/upper bounds and returns the chosen replica.
pub(crate) type SamplerFn<W> =
    fn(Vec<(usize, W, Option<usize>)>, W, W) -> (usize, Option<usize>);

/// Marker trait: policies that use `select_best_replica` (deterministic).
/// Implement this for policies that use `step!`-style scheduling.
pub(crate) trait DeterministicPolicy: QueuePlusPlus {}

/// Marker trait: policies that use `weigh_replica` + a sampler.
/// Implement this for policies that use `step_w_sampler!`-style scheduling.
pub(crate) trait StochasticPolicy: QueuePlusPlus
where
    Self::Weight: NaiiveLattice,
{
    fn sampler() -> SamplerFn<Self::Weight>;
}

// ---------------------------------------------------------------------------
// select_best_replica -- async replica evaluation (deterministic)
// ---------------------------------------------------------------------------

async fn select_best_replica<P>(
    entry: &Entry,
    qctx: &P::QueueContext,
    all_sctx: &[Arc<Mutex<ScheduleContext>>],
) -> Option<(
    usize,
    AssignScore<P::Measure, P::Weight>,
    Option<NumHitKvBlock>,
)>
where
    P: QueuePlusPlus,
    P::Measure: Ord,
{
    use futures::StreamExt as FuturesStreamExt;
    use tokio_stream::StreamExt as TokioStreamExt;

    let stream = TokioStreamExt::map(
        tokio_stream::iter(all_sctx.iter().cloned().enumerate()),
        |(replica_id, sched_ctx)| {
            let qctx = qctx.clone();
            async move {
                let sctx = sched_ctx.lock().await;
                P::eligible_with_kvblock_hit(replica_id, entry, &qctx, &sctx)
                    .map(|(score, hit_nblks)| (replica_id, score, hit_nblks))
            }
        },
    );

    TokioStreamExt::fold(
        TokioStreamExt::filter_map(FuturesStreamExt::buffer_unordered(stream, 8), |x| x),
        None,
        |best_replica: Option<(
            usize,
            AssignScore<P::Measure, P::Weight>,
            Option<NumHitKvBlock>,
        )>,
         next_replica| match best_replica {
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
        },
    )
    .await
}

// ---------------------------------------------------------------------------
// weigh_replica -- async replica evaluation (stochastic)
// ---------------------------------------------------------------------------

async fn weigh_replica<P>(
    entry: &Entry,
    qctx: &P::QueueContext,
    all_sctx: &[Arc<Mutex<ScheduleContext>>],
) -> (Vec<(usize, P::Weight, Option<usize>)>, P::Weight, P::Weight)
where
    P: QueuePlusPlus,
    P::Weight: NaiiveLattice,
{
    use futures::StreamExt as FuturesStreamExt;
    use tokio_stream::StreamExt as TokioStreamExt;

    let stream = TokioStreamExt::map(
        tokio_stream::iter(all_sctx.iter().cloned().enumerate()),
        |(replica_id, sched_ctx)| {
            let qctx = qctx.clone();
            async move {
                let sctx = sched_ctx.lock().await;
                P::eligible_with_kvblock_hit(replica_id, entry, &qctx, &sctx)
                    .map(|(score, hit_nblks)| (replica_id, score, hit_nblks))
            }
        },
    );

    TokioStreamExt::fold(
        TokioStreamExt::filter_map(FuturesStreamExt::buffer_unordered(stream, 8), |x| x),
        (
            Vec::<_>::with_capacity(all_sctx.len()),
            <P::Weight as NaiiveLattice>::TOP,
            <P::Weight as NaiiveLattice>::BOTTOM,
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
    .await
}

// ---------------------------------------------------------------------------
// step / step_w_sampler -- unified scheduling decision
// ---------------------------------------------------------------------------

/// Deterministic scheduling step: pick the best replica by score ordering.
async fn step_deterministic<P>(
    entry: &Entry,
    qctx: &P::QueueContext,
    all_sctx: &[Arc<Mutex<ScheduleContext>>],
) -> Option<usize>
where
    P: DeterministicPolicy,
    P::Measure: Ord,
{
    if let Some((replica_idx, _score, hit_nblks)) =
        select_best_replica::<P>(entry, qctx, all_sctx).await
    {
        apply_schedule_decision::<P>(entry, replica_idx, hit_nblks, all_sctx).await;
        Some(replica_idx)
    } else {
        tracing::info!("Cluster overloaded!");
        None
    }
}

/// Stochastic scheduling step: collect weighted scores, sample a replica.
async fn step_stochastic<P>(
    entry: &Entry,
    qctx: &P::QueueContext,
    all_sctx: &[Arc<Mutex<ScheduleContext>>],
) -> Option<usize>
where
    P: StochasticPolicy,
    P::Weight: NaiiveLattice,
{
    let (all_scores, lower_bound, upper_bound) =
        weigh_replica::<P>(entry, qctx, all_sctx).await;

    if all_scores.is_empty() {
        tracing::info!(target: "scheduling", "CLUSTER_OVERLOADED");
        None
    } else {
        let sampler = P::sampler();
        let (replica_idx, hit_nblks) = sampler(all_scores, lower_bound, upper_bound);
        apply_schedule_decision::<P>(entry, replica_idx, hit_nblks, all_sctx).await;
        Some(replica_idx)
    }
}

/// Common post-selection logic: compute cache hits, update metrics.
async fn apply_schedule_decision<P: QueuePlusPlus>(
    entry: &Entry,
    replica_idx: usize,
    _hit_nblks: Option<usize>,
    all_sctx: &[Arc<Mutex<ScheduleContext>>],
) {
    let request = &entry.request;
    let ScheduleContext { lmetric, block_hash } =
        &mut *all_sctx[replica_idx].lock().await;

    // Re-evaluate the prediction under THIS lock acquisition so the recorded
    // hit_nblks and decision_epoch are consistent. The hit_nblks computed in
    // select_best_replica/weigh_replica was under a separate lock acquisition;
    // an SSE handler may have advanced block_hash.epoch() in the window
    // between scoring and decision recording, leaving the score stale relative
    // to the current epoch. Recording the stale prediction together with the
    // newer epoch would be a TOCTOU hazard for any policy that reads
    // block_hash, not specific to any particular scheduling algorithm. (This
    // re-eval was an early hypothesis for the staleness violations tracked in
    // issue #10; the actual root cause turned out to be alias-bid drops in
    // RadixTreeBlockHash, fixed in kvcache.rs. This rescore is kept as a
    // strict observability enhancement — sub-microsecond cost, no regression.)
    let hit_nblks: usize = block_hash.get(entry.block_hash_state.get_hashes());
    entry.block_hash_state.set_pred_block_hits(hit_nblks);
    entry.block_hash_state.set_decision_epoch(block_hash.epoch());
    let new_ntkns = request.input_tokens.len()
        - /*inconsistent=*/ hit_nblks * entry.block_hash_state.get_block_size();

    tracing::info!(
        target: "scheduling",
        request_id = request.request_id,
        engine = replica_idx,
        predicted_hits = hit_nblks,
        radix_epoch = block_hash.epoch(),
        new_tokens = new_ntkns,
        "DECISION"
    );

    let metric_inc = LMetricInc {
        bs_inc: 1,
        waiting_reqs_inc: 1,
        prefill_tokens_inc: new_ntkns,
        all_tokens_inc: request.input_tokens.len(),
    };
    (*lmetric) += metric_inc;
}

// ---------------------------------------------------------------------------
// ScheduleStep -- unifies deterministic and stochastic scheduling behind one
// async method so QueueRunner can be generic over both.
// ---------------------------------------------------------------------------

/// Trait that bridges `DeterministicPolicy` and `StochasticPolicy` into a
/// single `schedule_step` async function used by `QueueRunner`.
pub(crate) trait ScheduleStep: QueuePlusPlus {
    fn schedule_step(
        entry: &Entry,
        qctx: &Self::QueueContext,
        all_sctx: &[Arc<Mutex<ScheduleContext>>],
    ) -> impl std::future::Future<Output = Option<usize>> + Send;
}

/// Blanket impl for deterministic policies.
impl<P> ScheduleStep for P
where
    P: DeterministicPolicy + Send + Sync,
    P::QueueContext: Send + Sync,
    P::Measure: Ord + Send + Sync,
    P::Weight: Send + Sync,
{
    fn schedule_step(
        entry: &Entry,
        qctx: &P::QueueContext,
        all_sctx: &[Arc<Mutex<ScheduleContext>>],
    ) -> impl std::future::Future<Output = Option<usize>> + Send {
        step_deterministic::<P>(entry, qctx, all_sctx)
    }
}

// NOTE: StochasticPolicy blanket impl cannot overlap with the above.
// We handle it explicitly for each stochastic policy via a manual impl
// of ScheduleStep in their respective modules.

// ---------------------------------------------------------------------------
// QueueCommandPro -- commands sent to the background task
// ---------------------------------------------------------------------------

enum QueueCommandPro {
    Append(Box<Entry>, Span),
    NextBatch(usize, oneshot::Sender<Option<NextBatch>>),
    NextRequest(usize, oneshot::Sender<Option<NextRequest>>),
    WaitingRequests(oneshot::Sender<usize>),
    WaitingPrefillTokens(oneshot::Sender<usize>),
}

// ---------------------------------------------------------------------------
// QueuePro -- public interface for a scheduling queue
// ---------------------------------------------------------------------------

pub(crate) trait QueuePro {
    fn append(&self, entry: Entry);
    async fn next_batch(&self, replica_id: usize) -> Option<NextBatch>;
    async fn next_request(&self, replica_id: usize) -> Option<NextRequest>;
    async fn waiting_requests(&self) -> usize;
    async fn waiting_prefill_tokens(&self) -> usize;
}

// ---------------------------------------------------------------------------
// QueueRunner<P> -- generic scheduling queue driven by policy P
// ---------------------------------------------------------------------------

/// Generic scheduling queue that wraps any policy `P` implementing
/// `QueuePlusPlus + ScheduleStep`.
///
/// This struct replaces both `impl_queue_with_task!` and
/// `impl_queue_with_task_and_sampler!` with a single generic implementation.
pub(crate) struct QueueRunner<P: QueuePlusPlus> {
    queue_sender: mpsc::UnboundedSender<QueueCommandPro>,
    _marker: std::marker::PhantomData<P>,
}

// Manual Clone impl: P does not need to be Clone because we only store PhantomData<P>.
impl<P: QueuePlusPlus> Clone for QueueRunner<P> {
    fn clone(&self) -> Self {
        Self {
            queue_sender: self.queue_sender.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<P> QueueRunner<P>
where
    P: QueuePlusPlus + ScheduleStep + Send + Sync + 'static,
    P::QueueContext: Send + Sync,
    P::Measure: Send + Sync,
    P::Weight: Send + Sync,
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
            _marker: std::marker::PhantomData,
        }
    }

    async fn queue_task(
        num_replicas: usize,
        mut receiver: mpsc::UnboundedReceiver<QueueCommandPro>,
        all_schedule_context: Vec<Arc<Mutex<ScheduleContext>>>,
    ) {
        // Create inner data structures
        let mut queue_context = <P::QueueContext as Default>::default();
        let mut next_batch_id = 0;
        let mut uncommit_buffer = VecDeque::with_capacity(128);
        let mut all_commit_req_buffers: Vec<VecDeque<(u64, Entry)>> =
            (0..num_replicas).map(|_| VecDeque::with_capacity(64)).collect();

        while let Some(cmd) = receiver.recv().await {
            match cmd {
                QueueCommandPro::Append(entry, _span) => {
                    metrics::increment_gauge!("blitz_queue_size", 1.0);
                    uncommit_buffer.push_back(*entry);
                    // Make schedule system step
                    // precond: not empty(uncommit_buffer)
                    let entry = uncommit_buffer.front().unwrap();
                    if let Some(replica_idx) = P::schedule_step(
                        entry,
                        &queue_context.clone(),
                        &all_schedule_context,
                    )
                    .await
                    {
                        let entry = uncommit_buffer.pop_front().unwrap();
                        all_commit_req_buffers[replica_idx]
                            .push_back((entry.request.request_id, entry));
                        queue_context += (replica_idx, num_replicas);
                    }
                }
                QueueCommandPro::NextBatch(replica_idx, response_sender) => {
                    'ineligible: while all_commit_req_buffers[replica_idx].is_empty() {
                        if let Some(entry) = uncommit_buffer.front() {
                            if let Some(tmp_replica_idx) = P::schedule_step(
                                entry,
                                &queue_context,
                                &all_schedule_context,
                            )
                            .await
                            {
                                let entry = uncommit_buffer.pop_front().unwrap();
                                all_commit_req_buffers[tmp_replica_idx]
                                    .push_back((entry.request.request_id, entry));
                            } else {
                                // postcond: all replicas are ineligible
                                tracing::warn!(
                                    "Replica#{replica_idx} is idle, but scheduler does not assign task to it"
                                );
                                break 'ineligible;
                            }
                        } else {
                            // postcond: there is no incoming requests
                            break 'ineligible;
                        }
                    }

                    let entries = all_commit_req_buffers[replica_idx]
                        .drain(..)
                        .collect::<Vec<_>>();

                    if entries.is_empty() {
                        response_sender.send(None).unwrap();
                        continue;
                    }

                    // Create span for this batch to add context to inference calls
                    let next_batch_span =
                        info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
                    next_batch_span.follows_from(&Span::current());

                    // Construct response
                    let mut batch_requests =
                        Vec::with_capacity(entries.len() / num_replicas);
                    let mut batch_entries = IntMap::with_capacity_and_hasher(
                        entries.len() / num_replicas,
                        BuildNoHashHasher::default(),
                    );

                    for (id, mut entry) in entries {
                        // Filter entries where the response receiver was dropped
                        if entry.response_tx.is_closed() {
                            metrics::increment_counter!(
                                "blitz_request_failure",
                                "err" => "dropped"
                            );
                            continue;
                        }

                        // Create a new span to link the batch back to this entry
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

                    // Finalize batch
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
                    'ineligible: while all_commit_req_buffers[replica_idx].is_empty() {
                        if let Some(entry) = uncommit_buffer.front() {
                            if let Some(tmp_replica_idx) = P::schedule_step(
                                entry,
                                &queue_context,
                                &all_schedule_context,
                            )
                            .await
                            {
                                let entry = uncommit_buffer.pop_front().unwrap();
                                all_commit_req_buffers[tmp_replica_idx]
                                    .push_back((entry.request.request_id, entry));
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
                QueueCommandPro::WaitingPrefillTokens(_sender) => {
                    unimplemented!("undecided API")
                }
                QueueCommandPro::WaitingRequests(_sender) => {
                    unimplemented!("undecided API")
                }
            }
        }
    }
}

/// Blanket `QueuePro` implementation for `QueueRunner<P>`.
impl<P> QueuePro for QueueRunner<P>
where
    P: QueuePlusPlus,
{
    #[instrument(skip_all)]
    fn append(&self, entry: Entry) {
        self.queue_sender
            .send(QueueCommandPro::Append(Box::new(entry), Span::current()))
            .unwrap();
    }

    #[instrument(skip_all)]
    async fn next_batch(&self, replica_id: usize) -> Option<NextBatch> {
        let (tx, rx) = oneshot::channel();
        self.queue_sender
            .send(QueueCommandPro::NextBatch(replica_id, tx))
            .unwrap();
        rx.await.unwrap()
    }

    #[instrument(skip_all)]
    async fn next_request(&self, replica_id: usize) -> Option<NextRequest> {
        let (tx, rx) = oneshot::channel();
        self.queue_sender
            .send(QueueCommandPro::NextRequest(replica_id, tx))
            .unwrap();
        rx.await.unwrap()
    }

    #[instrument(skip_all)]
    async fn waiting_prefill_tokens(&self) -> usize {
        let (tx, rx) = oneshot::channel();
        self.queue_sender
            .send(QueueCommandPro::WaitingPrefillTokens(tx))
            .unwrap();
        rx.await.unwrap()
    }

    #[instrument(skip_all)]
    async fn waiting_requests(&self) -> usize {
        let (tx, rx) = oneshot::channel();
        self.queue_sender
            .send(QueueCommandPro::WaitingRequests(tx))
            .unwrap();
        rx.await.unwrap()
    }
}

// ---------------------------------------------------------------------------
// Feature-gated TaskAssigner alias
// ---------------------------------------------------------------------------

#[cfg(feature = "bailian-impl-q")]
pub(crate) type TaskAssigner = QueueRunner<BailianImplQ>;
#[cfg(feature = "bounded-most-hit-q")]
pub(crate) type TaskAssigner = QueueRunner<JBoundMostHitQ2>;
#[cfg(feature = "least-wait-token-q")]
pub(crate) type TaskAssigner = QueueRunner<JLeastWaitTokenQ>;
#[cfg(feature = "aibrix-q")]
pub(crate) type TaskAssigner = QueueRunner<AibrixQ>;
#[cfg(feature = "dynamo-q")]
pub(crate) type TaskAssigner = QueueRunner<DynamoQ>;
#[cfg(feature = "dynamo-decoupled-q")]
pub(crate) type TaskAssigner = QueueRunner<DynamoDecodeQ>;
#[cfg(feature = "lmetric-q")]
pub(crate) type TaskAssigner = QueueRunner<LmetricQ>;
#[cfg(feature = "preble-q")]
pub(crate) type TaskAssigner = QueueRunner<PrebleQ>;
#[cfg(feature = "join-shortest-q-weight")]
pub(crate) type TaskAssigner = QueueRunner<JShortestQWeight>;
#[cfg(feature = "round-robin-q")]
pub(crate) type TaskAssigner = QueueRunner<RRQueue>;
#[cfg(feature = "random-q")]
pub(crate) type TaskAssigner = QueueRunner<RandomQ>;
// Legacy aliases
#[cfg(all(feature = "join-shortest-q", not(any(
    feature = "bailian-impl-q",
    feature = "bounded-most-hit-q",
    feature = "least-wait-token-q",
    feature = "aibrix-q",
    feature = "dynamo-q",
    feature = "join-shortest-q-weight",
    feature = "round-robin-q",
    feature = "random-q",
))))]
pub(crate) type TaskAssigner = QueueRunner<AibrixQ>;
