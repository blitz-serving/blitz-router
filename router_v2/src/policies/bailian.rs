// Bailian scheduling policy.
//
// Uses a three-component weighted score (cache hit ratio, request count,
// token count) with normalization and stochastic sampling.

use super::{
    AssignScore, EmptyContext, Entry, NaiiveLattice, NumHitKvBlock, QueuePlusPlus,
    SamplerFn, ScheduleStep, StochasticPolicy, step_stochastic,
};
use crate::kvcache::BlockHash;
use crate::{ScheduleContext, BAILIAN_ALPHA, BAILIAN_BETA, BAILIAN_GAMMA};

use rand::{thread_rng, Rng};
use std::sync::Arc;
use tokio::sync::Mutex;

// Lattice instance for the three-component weight vector.
impl NaiiveLattice for (f32, f32, f32) {
    fn meet(&self, other: &Self) -> Self {
        (
            self.0.min(other.0),
            self.1.min(other.1),
            self.2.min(other.2),
        )
    }

    fn join(&self, other: &Self) -> Self {
        (
            self.0.max(other.0),
            self.1.max(other.1),
            self.2.max(other.2),
        )
    }

    const TOP: Self = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    const BOTTOM: Self = (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
}

/// Bailian's scheduling queue: normalizes a 3-component score across all
/// replicas and samples proportionally.
pub(crate) struct BailianImplQ;

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
        // Normalize to [0,1]
        let n0 = ((hit_ratio - lower_bound.0) / dx0).clamp(0.0, 1.0);
        let n1 = ((upper_bound.1 - nreqs) / dx1).clamp(0.0, 1.0);
        let n2 = ((upper_bound.2 - ntkns) / dx2).clamp(0.0, 1.0);

        // Linear combination with configurable weights
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

        Some((
            AssignScore::Weighted((hit_ratio, num_requests, num_tokens)),
            Some(hit_nblks),
        ))
    }
}

impl StochasticPolicy for BailianImplQ {
    fn sampler() -> SamplerFn<Self::Weight> {
        bailian_sampler
    }
}

impl ScheduleStep for BailianImplQ {
    fn schedule_step(
        entry: &Entry,
        qctx: &Self::QueueContext,
        all_sctx: &[Arc<Mutex<ScheduleContext>>],
    ) -> impl std::future::Future<Output = Option<usize>> + Send {
        step_stochastic::<Self>(entry, qctx, all_sctx)
    }
}
