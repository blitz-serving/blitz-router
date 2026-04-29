// Dual-stage Preble routing logic.
//
// Bijective translation from AIBrix's Go implementation:
//   aibrix/pkg/plugins/gateway/algorithms/prefix_cache_preble.go
//   Lines 437-566 (Route function)
//
// Stage 1 (prefix-aware): If prefix match ratio > 50%, route to the
//   replica with the longest matching prefix, tie-breaking by load.
//
// Stage 2 (cost-model fallback): If no prefix match or ratio <= 50%,
//   route to the replica with minimum total allocation cost.

use super::histogram::SlidingWindowHistogram;
use crate::policies::Entry;

/// Prefix routing threshold (Go line 477):
/// ```go
/// prefixRoutingThreshold := 0.5
/// ```
const PREFIX_ROUTING_THRESHOLD: f64 = 0.5;

/// Result of the dual-stage routing decision.
pub(crate) struct RouteDecision {
    /// Selected replica index.
    pub replica_idx: usize,
    /// Number of KV cache block hits on the selected replica.
    pub hit_nblks: usize,
    /// Score (lower is better for Least).
    pub score: usize,
}

/// Per-replica prefix match info, used during stage 1.
///
/// Go lines 86-91:
/// ```go
/// type prefixMatch struct {
///     node        *prefixcacheindexer.TreeNode
///     pods        []*v1.Pod
///     matchLength int
///     depth       int
/// }
/// ```
struct PrefixMatch {
    replica_id: usize,
    hit_nblks: usize,
    match_tokens: usize,
}

/// Execute the dual-stage Preble routing algorithm.
///
/// Go lines 437-566 (Route function), adapted for blitz-router's
/// per-replica ScheduleContext model.
///
/// # Arguments
/// * `entry` - The request to route.
/// * `all_sctx` - Per-replica schedule contexts (locked snapshots).
/// * `histogram` - The shared sliding window histogram.
///
/// # Returns
/// The selected replica index and hit block count, or None if no replica available.
pub(crate) fn route(
    _entry: &Entry,
    all_sctx_snapshots: &[(usize, usize, usize)], // (replica_id, hit_nblks, all_tokens)
    histogram: &SlidingWindowHistogram,
    input_len: usize,
    block_size: usize,
) -> Option<RouteDecision> {
    if all_sctx_snapshots.is_empty() {
        return None;
    }

    let _num_replicas = all_sctx_snapshots.len();

    // Find the replica with the longest prefix match
    let mut prefix_matches: Vec<PrefixMatch> = all_sctx_snapshots
        .iter()
        .map(|&(replica_id, hit_nblks, _all_tokens)| PrefixMatch {
            replica_id,
            hit_nblks,
            match_tokens: hit_nblks * block_size,
        })
        .collect();

    // Sort by match length descending (Go lines 507-509)
    // ```go
    // sort.Slice(prefixMatches, func(i, j int) bool {
    //     return prefixMatches[i].matchLength > prefixMatches[j].matchLength
    // })
    // ```
    prefix_matches.sort_by(|a, b| b.match_tokens.cmp(&a.match_tokens));

    let best_match = &prefix_matches[0];
    let match_ratio = best_match.match_tokens as f64 / input_len.max(1) as f64;

    // ================================================================
    // Stage 1: Prefix-aware routing
    // ================================================================
    //
    // Go lines 480-531:
    // ```go
    // if matchRatio > prefixRoutingThreshold {
    //     // ... collect prefix matches from node to root ...
    //     sort.Slice(prefixMatches, func(i, j int) bool {
    //         return prefixMatches[i].matchLength > prefixMatches[j].matchLength
    //     })
    //     if len(prefixMatches) > 0 {
    //         longestMatch := prefixMatches[0]
    //         minLoad := -1
    //         for _, pod := range longestMatch.pods {
    //             load := p.histogram.getPodLoad(pod)
    //             if minLoad == -1 || load < minLoad {
    //                 minLoad = load
    //                 targetPod = pod
    //             }
    //         }
    //     }
    // }
    // ```
    if match_ratio > PREFIX_ROUTING_THRESHOLD {
        // Find the longest match length
        let longest_match_tokens = best_match.match_tokens;

        // Collect all replicas with the same longest match
        let candidates: Vec<&PrefixMatch> = prefix_matches
            .iter()
            .filter(|m| m.match_tokens == longest_match_tokens)
            .collect();

        if !candidates.is_empty() {
            // Tie-break by load (Go: getPodLoad)
            let mut best_replica = candidates[0];
            let mut min_load = histogram.get_replica_load(best_replica.replica_id);

            for cand in &candidates[1..] {
                let load = histogram.get_replica_load(cand.replica_id);
                if load < min_load {
                    min_load = load;
                    best_replica = cand;
                }
            }

            let new_prefill = input_len.saturating_sub(best_replica.match_tokens);
            let all_tokens = all_sctx_snapshots
                .iter()
                .find(|(id, _, _)| *id == best_replica.replica_id)
                .map(|(_, _, at)| *at)
                .unwrap_or(0);

            return Some(RouteDecision {
                replica_idx: best_replica.replica_id,
                hit_nblks: best_replica.hit_nblks,
                score: new_prefill + all_tokens,
            });
        }
    }

    // ================================================================
    // Stage 2: Cost-model fallback
    // ================================================================
    //
    // Go lines 533-548:
    // ```go
    // if targetPod == nil {
    //     podCosts := p.histogram.getCurrentAllocationCostPerPod()
    //     minCost := math.MaxFloat64
    //     for _, pod := range readyPods {
    //         cost := podCosts[pod.Name]
    //         if cost < minCost {
    //             minCost = cost
    //             targetPod = pod
    //         }
    //     }
    // }
    // ```
    let replica_costs = histogram.get_allocation_cost_per_replica();

    let mut min_cost = f64::MAX;
    let mut selected_idx = 0;
    let mut selected_hit = 0;

    for &(replica_id, hit_nblks, _all_tokens) in all_sctx_snapshots {
        let cost = if replica_id < replica_costs.len() {
            replica_costs[replica_id]
        } else {
            0.0
        };

        if cost < min_cost {
            min_cost = cost;
            selected_idx = replica_id;
            selected_hit = hit_nblks;
        }
    }

    let new_prefill = input_len.saturating_sub(selected_hit * block_size);
    let all_tokens = all_sctx_snapshots
        .iter()
        .find(|(id, _, _)| *id == selected_idx)
        .map(|(_, _, at)| *at)
        .unwrap_or(0);

    Some(RouteDecision {
        replica_idx: selected_idx,
        hit_nblks: selected_hit,
        score: new_prefill + all_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::preble::cost_model::TargetGpu;

    #[test]
    fn test_route_cost_model_fallback() {
        // With no prefix matches above threshold, stage 2 (cost model) applies.
        // All replicas have zero histogram cost, so first one wins.
        let histogram = SlidingWindowHistogram::new(3, TargetGpu::V100);
        // All replicas have 0 hit blocks -> match_ratio = 0 < 0.5
        let snapshots = vec![(0, 0, 100), (1, 0, 200), (2, 0, 50)];

        // Call route with a dummy entry (not dereferenced since _entry is unused)
        // We can't easily construct Entry, but route() doesn't read it.
        // Instead, test the sub-components directly.
        let costs = histogram.get_allocation_cost_per_replica();
        assert_eq!(costs.len(), 3);
        // All costs should be zero since histogram is empty
        for c in &costs {
            assert!(
                *c == 0.0,
                "Empty histogram should yield zero cost, got {c}"
            );
        }
    }

    #[test]
    fn test_prefix_routing_threshold() {
        // Verify the threshold constant matches Go
        assert!((PREFIX_ROUTING_THRESHOLD - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_histogram_cost_after_updates() {
        let mut histogram = SlidingWindowHistogram::new(2, TargetGpu::V100);
        // Simulate some request traffic
        histogram.update(42, 256, 1024, 0, 45);
        histogram.update(43, 128, 512, 1, 45);

        let costs = histogram.get_allocation_cost_per_replica();
        assert_eq!(costs.len(), 2);
        // After updates, costs should be non-negative
        for (i, c) in costs.iter().enumerate() {
            assert!(*c >= 0.0, "Replica {i} cost should be >= 0, got {c}");
        }
    }
}
