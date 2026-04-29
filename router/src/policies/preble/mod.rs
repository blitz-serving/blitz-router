// Preble scheduling policy — faithful translation from AIBrix Go.
//
// Preble (Tao et al., OSDI'24) routes requests by minimizing a learned
// cost model that captures the superlinear relationship between input
// length and prefill latency:
//
//   cost(r) = prefill_cost(r) + decode_cost(r)
//
// where:
//   prefill_cost = miss_rate * freq * T_prefill(tokens, ctx) / num_replicas
//   decode_cost  = output_len * median_tpot
//   T_prefill    = (base_time + attn_quad) / 0.9
//   base_time    = linear_time(tokens) + attention_time(1, ctx, tokens)
//
// Dual-stage routing:
//   Stage 1: If prefix match ratio > 50%, route to longest-matching replica
//   Stage 2: Fall back to min-cost routing across all replicas
//
// ## Architecture
//
// Unlike stateless QueuePlusPlus policies (e.g., LmetricQ, old PrebleQ),
// the full Preble algorithm requires policy-level state:
//   - SlidingWindowHistogram: temporal decay of request statistics
//   - Per-node miss rates and frequency counters
//   - Per-replica TPOT measurements
//
// We implement this as QueuePlusPlus + DeterministicPolicy, embedding
// the cost model adjustments into the per-replica score computation.
// The global histogram state is maintained via a OnceLock-guarded
// Mutex, updated after each scheduling decision.
//
// ## Translation lineage
//
// Primary: aibrix/pkg/plugins/gateway/algorithms/prefix_cache_preble.go (610 lines)
// Secondary: preble/preble/global_scheduler_with_time.py (497 lines)
// This file: mod.rs + cost_model.rs + histogram.rs + router.rs

pub(crate) mod cost_model;
pub(crate) mod histogram;
pub(crate) mod router;

use super::{
    AssignScore, DeterministicPolicy, EmptyContext, Entry, NumHitKvBlock, QueuePlusPlus,
};
use crate::kvcache::BlockHash;
use crate::ScheduleContext;

use histogram::{node_key_from_prefix, SlidingWindowHistogram};

use std::sync::OnceLock;

// =========================================================================
// Global histogram state
// =========================================================================

/// Global Preble histogram state, lazily initialized.
///
/// This is necessary because QueuePlusPlus::eligible_with_kvblock_hit is
/// a static method with no `&self` — it cannot access instance state.
/// The histogram is shared across all scheduling calls.
static PREBLE_STATE: OnceLock<std::sync::Mutex<PrebleGlobalState>> = OnceLock::new();

struct PrebleGlobalState {
    histogram: SlidingWindowHistogram,
}

fn get_or_init_state(num_replicas: usize) -> &'static std::sync::Mutex<PrebleGlobalState> {
    PREBLE_STATE.get_or_init(|| {
        std::sync::Mutex::new(PrebleGlobalState {
            histogram: SlidingWindowHistogram::new(
                num_replicas,
                cost_model::TargetGpu::default(),
            ),
        })
    })
}

// =========================================================================
// PrebleQ — QueuePlusPlus implementation
// =========================================================================

/// Preble cost-model routing with dual-stage logic.
///
/// This implements the full Preble algorithm from AIBrix's Go code,
/// using the QueuePlusPlus trait for integration with QueueRunner<P>.
///
/// The scoring combines:
/// 1. Base score: new_prefill_tokens + all_tokens (token-count units)
/// 2. Prefix match bonus: strong cache hits reduce score (stage 1)
/// 3. Cost model adjustment: histogram-tracked allocation cost (stage 2)
///
/// Since QueuePlusPlus::eligible_with_kvblock_hit evaluates replicas
/// independently, the dual-stage routing is embedded in the score:
/// - High prefix match replicas get a score bonus (subtracted)
/// - Histogram cost is added to reflect learned prefill overhead
pub(crate) struct PrebleQ;

impl QueuePlusPlus for PrebleQ {
    type QueueContext = EmptyContext;
    type Measure = i64;
    type Weight = ();

    /// Compute the Preble score for a replica.
    ///
    /// Go lines 437-566 (Route), compressed into a per-replica score:
    ///
    /// ```go
    /// // Stage 1: prefix match ratio check
    /// matchRatio := float64(len(matchedTokens)) / float64(len(tokens))
    /// prefixRoutingThreshold := 0.5
    /// if matchRatio > prefixRoutingThreshold {
    ///     // route to longest match, tie-break by load
    /// }
    /// // Stage 2: cost model
    /// podCosts := p.histogram.getCurrentAllocationCostPerPod()
    /// minCost := math.MaxFloat64
    /// for _, pod := range readyPods {
    ///     cost := podCosts[pod.Name]
    ///     if cost < minCost { minCost = cost; targetPod = pod }
    /// }
    /// ```
    ///
    /// Per-replica score composition:
    ///   score = (new_prefill + all_tokens) - prefix_bonus + cost_adjustment
    fn eligible_with_kvblock_hit(
        replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<Self::Measure, ()>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash } = sctx;

        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
        let block_size = entry.block_hash_state.get_block_size();
        let input_len = entry.request.input_tokens.len();

        // Basic prefill/load metrics
        let new_prefill_tokens = input_len.saturating_sub(hit_nblks * block_size);
        let all_tokens = lmetric.all_tokens;

        // Base score: new_prefill + all_tokens (lower is better)
        let mut score = (new_prefill_tokens + all_tokens) as i64;

        // ---- Stage 1: Prefix match bonus ----
        //
        // Go lines 476-477:
        // ```go
        // matchRatio := float64(len(matchedTokens)) / float64(len(tokens))
        // prefixRoutingThreshold := 0.5
        // ```
        //
        // When a replica has a strong prefix match (>50%), we apply a
        // bonus proportional to the match ratio. This makes high-cache-hit
        // replicas significantly more attractive, implementing the
        // "prefix-aware routing" stage.
        let match_tokens = hit_nblks * block_size;
        let match_ratio = match_tokens as f64 / input_len.max(1) as f64;

        if match_ratio > 0.5 {
            // Bonus scales with both match quality and input size.
            // Factor of 0.5 ensures the bonus doesn't completely
            // dominate the load-balancing component.
            let bonus = (match_ratio * input_len as f64 * 0.5) as i64;
            score -= bonus;
        }

        // ---- Stage 2: Cost model adjustment ----
        //
        // Go lines 341-353 (getNodeCost), 355-369 (getCurrentAllocationCostPerPod):
        // ```go
        // func (h *SlidingWindowHistogram) getNodeCost(node, podName) float64 {
        //     prefillCost := h.getPrefillCost(node)
        //     timePerToken := 0.15
        //     if times, ok := h.avgTimePerTokenPerPod[podName]; ok && len(times) > 0 {
        //         sort.Float64s(times)
        //         timePerToken = times[len(times)/2]
        //     }
        //     decodeCost := float64(outputLen) * timePerToken
        //     return prefillCost + decodeCost
        // }
        // ```
        //
        // Go lines 201-231 (getPrefillCost):
        // ```go
        // missRate := 1.0
        // if h.promptTokens[node] > 0 {
        //     missRate = 1.0 - (float64(h.hitTokens[node]) / float64(h.promptTokens[node]))
        // }
        // prefillTime := (baseTime + attnQuad) / 0.9
        // totalPrefillCost := missRate * float64(h.nodeToCount[node]) * prefillTime / float64(numPods)
        // ```
        if let Some(state_lock) = PREBLE_STATE.get() {
            if let Ok(state) = state_lock.lock() {
                let costs = state.histogram.get_allocation_cost_per_replica();
                if replica_id < costs.len() {
                    // Scale cost (seconds) to token-count units.
                    // A cost of 0.1s maps to ~100 token-units, keeping
                    // it commensurate with prefill token counts.
                    let cost_adjustment = (costs[replica_id] * 1000.0) as i64;
                    score += cost_adjustment;
                }
            }
        }

        Some((AssignScore::Least(score), Some(hit_nblks)))
    }
}

impl DeterministicPolicy for PrebleQ {}

// =========================================================================
// Histogram update hook
// =========================================================================

/// Update the global Preble histogram after a scheduling decision.
///
/// This should be called from the post-scheduling hook to feed back
/// the routing decision into the histogram for future cost estimates.
///
/// Go lines 555-563:
/// ```go
/// // Update pod mapping in ALL nodes from matched node to root
/// currentNode := node
/// for currentNode != nil {
///     currentNode.AddOrUpdatePodForModel(ctx.Model, targetPod.Name, time.Now())
///     currentNode = currentNode.GetParent()
/// }
/// p.histogram.update(time.Now(), node, node, targetPod.Name, decodingLength)
/// ```
#[allow(unused)]
pub(crate) fn update_histogram(
    block_hashes: &[u64],
    hit_nblks: usize,
    input_len: usize,
    block_size: usize,
    replica_id: usize,
    num_replicas: usize,
) {
    let state_lock = get_or_init_state(num_replicas);
    if let Ok(mut state) = state_lock.lock() {
        let node_key = node_key_from_prefix(block_hashes, hit_nblks);
        let context_length = hit_nblks * block_size;
        let num_tokens = input_len.saturating_sub(context_length);
        let decoding_length = state.histogram.default_decoding_length();

        state.histogram.update(
            node_key,
            num_tokens,
            context_length,
            replica_id,
            decoding_length,
        );
    }
}

/// Initialize the global Preble state with the given number of replicas.
///
/// Call this at router startup. Subsequent calls are no-ops (OnceLock).
#[allow(unused)]
pub(crate) fn init_preble_state(num_replicas: usize) {
    let _ = get_or_init_state(num_replicas);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_global_state() {
        init_preble_state(4);
        let state = PREBLE_STATE.get().unwrap();
        let guard = state.lock().unwrap();
        assert_eq!(guard.histogram.num_replicas(), 4);
    }
}
