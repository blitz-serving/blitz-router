// AIBrix prefix-cache scheduling policy.
//
// Replicates AIBrix's prefix-cache routing algorithm (prefix_cache.go):
// 1. Check load imbalance across replicas (max - min > threshold)
// 2. If imbalanced, restrict candidates to min-loaded replicas
// 3. Sort candidates by (prefix match % DESC, request count ASC)
// 4. Select first replica within mean + stddev_factor * stddev threshold
// 5. Fallback to last in sorted list

use super::super::{
    AssignScore, EmptyContext, Entry, NaiiveLattice, NumHitKvBlock, QueuePlusPlus, SamplerFn,
    ScheduleStep, StochasticPolicy, step_stochastic,
};
use crate::kvcache::BlockHash;
use crate::ScheduleContext;

use std::sync::Arc;
use tokio::sync::Mutex;

/// Load imbalance threshold (AIBrix default: 8).
/// When max_requests - min_requests > this value, only min-loaded replicas
/// are considered as candidates.
const AIBRIX_IMBALANCE_ABS_COUNT: usize = 8;

/// Standard deviation factor for load filtering (AIBrix default: 1).
/// A replica is eligible only if request_count <= mean + factor * stddev.
const AIBRIX_STDDEV_FACTOR: f32 = 1.0;

// Lattice instance for the two-component weight vector (match_pct, request_count).
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

/// AIBrix prefix-cache scheduling: sorts replicas by prefix match percentage
/// and request count, filtered by stddev-based load threshold.
pub(crate) struct AibrixQ;

/// AIBrix prefix-cache sampler.
///
/// Implements the full selection algorithm from AIBrix's prefix_cache.go:
///
/// 1. Load imbalance detection: if max_req - min_req > IMBALANCE_ABS_COUNT,
///    restrict candidates to replicas with minimum request count.
///
/// 2. Compute mean and sample stddev of request counts across candidates.
///
/// 3. Sort candidates by (match% DESC, request_count ASC).
///
/// 4. Select the first candidate where request_count <= mean + factor * stddev.
///
/// 5. Fallback: last candidate in the sorted list (per AIBrix behavior).
fn aibrix_prefix_cache_sampler(
    all_scores: Vec<(usize, (f32, f32), Option<(usize, u64)>)>,
    _lower_bound: (f32, f32),
    _upper_bound: (f32, f32),
) -> (usize, Option<(usize, u64)>) {
    debug_assert!(!all_scores.is_empty());

    let all_req_counts: Vec<f32> = all_scores.iter().map(|s| s.1 .1).collect();
    let min_req = all_req_counts.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_req = all_req_counts
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Step 1: Load imbalance check.
    let imbalanced = (max_req - min_req) > AIBRIX_IMBALANCE_ABS_COUNT as f32;
    let candidates: Vec<(usize, (f32, f32), Option<(usize, u64)>)> = if imbalanced {
        all_scores
            .into_iter()
            .filter(|s| (s.1 .1 - min_req).abs() < 0.5)
            .collect()
    } else {
        all_scores
    };

    // Step 2: Compute mean and stddev from candidate request counts.
    let cand_req_counts: Vec<f32> = candidates.iter().map(|s| s.1 .1).collect();
    let n = cand_req_counts.len() as f32;
    let mean = cand_req_counts.iter().sum::<f32>() / n;
    let variance = if n > 1.0 {
        cand_req_counts
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / (n - 1.0) // sample variance, matching AIBrix's standardDeviation()
    } else {
        0.0
    };
    let stddev = variance.sqrt();
    let threshold = mean + AIBRIX_STDDEV_FACTOR * stddev;

    // Step 3: Sort by (match% DESC, request_count ASC).
    let mut sorted = candidates;
    sorted.sort_by(|a, b| {
        b.1 .0
            .partial_cmp(&a.1 .0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.1 .1
                    .partial_cmp(&b.1 .1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Step 4: Select first replica within the threshold.
    for &(replica_id, (_, req_count), cached) in sorted.iter() {
        if req_count <= threshold {
            return (replica_id, cached);
        }
    }

    // Step 5: Fallback -- last in sorted list (per AIBrix's prefix_cache.go).
    let last = sorted.last().unwrap();
    (last.0, last.2)
}

impl QueuePlusPlus for AibrixQ {
    type QueueContext = EmptyContext;
    type Measure = ();
    type Weight = (f32, f32);

    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<(), Self::Weight>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash } = sctx;
        let request = &entry.request;

        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
        let match_pct = (hit_nblks * entry.block_hash_state.get_block_size()) as f32
            / request.input_tokens.len() as f32;
        let request_count = lmetric.bs as f32;

        Some((
            AssignScore::Weighted((match_pct, request_count)),
            Some(hit_nblks),
        ))
    }
}

impl StochasticPolicy for AibrixQ {
    fn sampler() -> SamplerFn<Self::Weight> {
        aibrix_prefix_cache_sampler
    }
}

impl ScheduleStep for AibrixQ {
    fn schedule_step(
        entry: &Entry,
        qctx: &Self::QueueContext,
        all_sctx: &[Arc<Mutex<ScheduleContext>>],
    ) -> impl std::future::Future<Output = Option<usize>> + Send {
        step_stochastic::<Self>(entry, qctx, all_sctx)
    }
}
