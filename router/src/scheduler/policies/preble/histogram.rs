// Sliding window histogram for Preble cost model state tracking.
//
// Bijective translation from AIBrix's Go implementation:
//   aibrix/pkg/plugins/gateway/algorithms/prefix_cache_preble.go
//   Lines 53-82 (SlidingWindowHistogram struct)
//   Lines 262-318 (removeOldEntries / removeEvictedNodes)
//   Lines 341-353 (getNodeCost)
//   Lines 355-369 (getCurrentAllocationCostPerPod)
//   Lines 569-582 (getPodLoad)
//   Lines 585-610 (update)
//
// In blitz-router, "pod" maps to "replica" (replica_id: usize).
// "TreeNode" — Go's prefix-cache trie node, identified by `*TreeNode`
// pointer — maps to a path in our `Trie<u64, OwnedNodeStats>` keyed by
// the matched-prefix block hashes.
//
// Per-replica observations of TPOT and load are NOT stored in this
// struct. They live on `LMetric` in `ScheduleContext`, snapshotted into
// `Observation.tpot` / `Observation.bs` etc. by the policy `schedule()`
// path. See `dsl_runtime::capture_observations`.

use std::time::{Duration, Instant};

use radixtree::{ReplicaSet, Trie};
use smallvec::SmallVec;

use super::cost_model::{self, TargetGpu};

/// A single histogram entry, recording a request arrival.
///
/// Go lines 71-75:
/// ```go
/// type histogramEntry struct {
///     timestamp time.Time
///     node      *prefixcacheindexer.TreeNode
///     leafNode  *prefixcacheindexer.TreeNode
/// }
/// ```
///
/// In our port the "node identity" is the matched-prefix path itself
/// (stored as `Box<[u64]>`); we need to remember it on the entry so
/// `remove_old_entries` can navigate back to the right tree node when
/// the entry expires. We also track the `replica_id` because Go's
/// per-pod owner-set decay (`podAllocations`) requires per-(node, pod)
/// counts: ownership ends only when the LAST recent entry for that
/// (node, pod) pair leaves the window.
#[derive(Debug, Clone)]
struct HistogramEntry {
    timestamp: Instant,
    /// Path identifying the prefix node this entry was recorded under.
    prefix: Box<[u64]>,
    /// Number of new (uncached) tokens this request contributed.
    leaf_num_tokens: usize,
    /// Total context length up to and including this prefix node.
    leaf_context_length: usize,
    /// Replica that served this entry. Drives per-(node, replica)
    /// owner-set expiry — see `remove_old_entries`.
    replica_id: usize,
}

/// Per-node accumulated statistics within the sliding window. Lives on
/// the trie node payload.
#[derive(Debug, Clone, Default)]
pub(crate) struct NodeStats {
    /// Total context length tokens accumulated for this node.
    /// Go: `histogram map[*TreeNode]int`
    histogram: usize,
    /// Number of requests routed through this node.
    /// Go: `nodeToCount map[*TreeNode]int`
    node_to_count: usize,
    /// Tokens that were cache hits (context_length - num_tokens per request).
    /// Go: `hitTokens map[*TreeNode]int`
    hit_tokens: usize,
    /// Total prompt tokens (context_length per request).
    /// Go: `promptTokens map[*TreeNode]int`
    prompt_tokens: usize,
    /// Output (decoding) length, last-write-wins per node.
    /// Go: `decodingSize map[*TreeNode]int`
    decoding_size: usize,
    /// Total decode lengths accumulated.
    /// Go: `perNodeTotalDecodeLengths map[*TreeNode]int`
    total_decode_lengths: usize,
    /// Per-(node, replica) entry count within the sliding window. Drives
    /// owner-set expiry — when the count for `replica_id` drops to zero,
    /// the replica is removed from the trie node's owners. SmallVec
    /// inline storage = 4 replicas covers the common "few replicas
    /// touch any one node" case; spills to heap otherwise.
    replica_entry_counts: SmallVec<[(usize, usize); 4]>,
}

impl NodeStats {
    fn replica_count_mut(&mut self, replica_id: usize) -> &mut usize {
        if let Some(idx) = self
            .replica_entry_counts
            .iter()
            .position(|(r, _)| *r == replica_id)
        {
            &mut self.replica_entry_counts[idx].1
        } else {
            self.replica_entry_counts.push((replica_id, 0));
            let last = self.replica_entry_counts.len() - 1;
            &mut self.replica_entry_counts[last].1
        }
    }

    /// Decrement the count for `replica_id`, removing the entry if it
    /// reaches zero. Returns `true` iff the count reached zero (caller
    /// should drop `replica_id` from `owners`).
    fn decrement_replica_count(&mut self, replica_id: usize) -> bool {
        if let Some(idx) = self
            .replica_entry_counts
            .iter()
            .position(|(r, _)| *r == replica_id)
        {
            let c = &mut self.replica_entry_counts[idx].1;
            *c = c.saturating_sub(1);
            if *c == 0 {
                self.replica_entry_counts.swap_remove(idx);
                return true;
            }
        }
        false
    }
}

/// Trie payload: per-node owner set + accumulated stats. Both live on
/// the same node — owner-set membership and stats decay together.
#[derive(Default, Debug, Clone)]
pub(crate) struct OwnedNodeStats {
    pub owners: ReplicaSet,
    pub stats: NodeStats,
}

/// Sliding window histogram tracking request-to-node assignments.
///
/// Go lines 53-69:
/// ```go
/// type SlidingWindowHistogram struct {
///     mu                         sync.RWMutex
///     windowDuration             time.Duration
///     histogram                  map[*TreeNode]int
///     nodeToCount                map[*TreeNode]int
///     hitTokens                  map[*TreeNode]int
///     promptTokens               map[*TreeNode]int
///     decodingSize               map[*TreeNode]int
///     timestamps                 []histogramEntry
///     numPods                    int
///     podAllocations             map[*TreeNode]map[int]bool
///     currentDecodeLengthsPerPod map[string]int
///     avgTimePerTokenPerPod      map[string][]float64
///     perNodeTotalDecodeLengths  map[*TreeNode]int
/// }
/// ```
///
/// In our port:
///   * Per-node fields (`histogram`, `nodeToCount`, …) collapse into
///     `NodeStats` and live on the trie node payload.
///   * `podAllocations[*TreeNode]` becomes `OwnedNodeStats.owners:
///     ReplicaSet` on the same payload — fixing the structural bug
///     where the original port distributed cost evenly because it had
///     no per-node owner info.
///   * `avgTimePerTokenPerPod` is dropped: per-replica TPOT is
///     `LMetric.tpot`, already maintained by the SSE loop and
///     snapshotted into `Observation.tpot`. The caller passes it into
///     `cost_for_replica`.
///   * `currentDecodeLengthsPerPod` is dropped: not consumed by any
///     read path in the original port, and the equivalent live signal
///     lives on `LMetric` if a future caller needs it.
pub(crate) struct SlidingWindowHistogram {
    /// Window duration for temporal decay.
    /// Go: `slidingWindowPeriod = 3 * time.Minute`
    window_duration: Duration,

    /// Prefix trie keyed by block hash, payload = (owners, stats).
    /// Replaces Go's per-node maps + `podAllocations`.
    tree: Trie<u64, OwnedNodeStats>,

    /// Ordered list of entries for sliding window eviction.
    timestamps: Vec<HistogramEntry>,

    /// Number of replicas — sizes the `ReplicaSet` bitsets created on
    /// node insert.
    num_replicas: usize,

    /// Target GPU for cost model.
    target_gpu: TargetGpu,

    /// Default decoding length when unknown.
    /// Go: `decodingLength = utils.LoadEnvInt(PREBLE_DECODING_LENGTH, 45)`
    default_decoding_length: usize,
}

/// Default sliding window period (Go: 3 minutes).
const DEFAULT_WINDOW_DURATION: Duration = Duration::from_secs(3 * 60);

/// Default expected output length (Go: 45 tokens).
const DEFAULT_DECODING_LENGTH: usize = 45;

/// Default TPOT when no measurements available (Go: 0.15 seconds = 150 ms).
///
/// Used when the caller passes a non-finite or zero `replica_tpot_ms`
/// into `cost_for_replica`.
const DEFAULT_TIME_PER_TOKEN_MS: f64 = 150.0;

impl SlidingWindowHistogram {
    /// Create a new histogram.
    pub fn new(num_replicas: usize, target_gpu: TargetGpu) -> Self {
        Self {
            window_duration: DEFAULT_WINDOW_DURATION,
            tree: Trie::new(),
            timestamps: Vec::new(),
            num_replicas,
            target_gpu,
            default_decoding_length: DEFAULT_DECODING_LENGTH,
        }
    }

    /// Update histogram with a new request assignment.
    ///
    /// Go lines 585-610:
    /// ```go
    /// func (h *SlidingWindowHistogram) update(timestamp time.Time, node, leafNode *TreeNode, podName string, decodingLength int) {
    ///     h.timestamps = append(h.timestamps, histogramEntry{timestamp, node, leafNode})
    ///     h.histogram[node] += leafNode.ContextLength()
    ///     h.nodeToCount[node]++
    ///     h.decodingSize[node] = decodingLength
    ///     h.hitTokens[node] += leafNode.ContextLength() - leafNode.NumTokens()
    ///     h.promptTokens[node] += leafNode.ContextLength()
    ///     h.currentDecodeLengthsPerPod[podName] += decodingLength
    ///     h.perNodeTotalDecodeLengths[node] += decodingLength
    ///     // ... podAllocations[node][pod] = true ...
    /// }
    /// ```
    ///
    /// `prefix` is the matched-prefix block-hash path that identifies
    /// the prefix node. If `prefix.is_empty()`, the call is a no-op
    /// (no node to attach the entry to — Go uses the root sentinel for
    /// no-prefix-match requests, but we elide it).
    pub fn update(
        &mut self,
        prefix: &[u64],
        num_tokens: usize,
        context_length: usize,
        replica_id: usize,
        decoding_length: usize,
    ) {
        if prefix.is_empty() {
            return;
        }
        let now = Instant::now();
        let num_replicas = self.num_replicas;

        // Materialise the path in the trie and update the payload.
        let payload = self.tree.ensure_path(prefix);
        if payload.owners.is_empty() && payload.stats.node_to_count == 0 {
            // First time we touch this node — initialise the bitset
            // with the current replica count. (ReplicaSet::new lazily
            // sizes the bitvec.)
            payload.owners = ReplicaSet::new(num_replicas);
        }
        payload.owners.insert(replica_id);

        let stats = &mut payload.stats;
        stats.histogram += context_length;
        stats.node_to_count += 1;
        stats.decoding_size = decoding_length;
        stats.hit_tokens += context_length.saturating_sub(num_tokens);
        stats.prompt_tokens += context_length;
        stats.total_decode_lengths += decoding_length;
        *stats.replica_count_mut(replica_id) += 1;

        self.timestamps.push(HistogramEntry {
            timestamp: now,
            prefix: prefix.into(),
            leaf_num_tokens: num_tokens,
            leaf_context_length: context_length,
            replica_id,
        });

        // Evict old entries
        self.remove_old_entries(now);
    }

    /// Remove entries outside the sliding window.
    ///
    /// Go lines 292-318:
    /// ```go
    /// func (h *SlidingWindowHistogram) removeOldEntries(currentTime time.Time) {
    ///     windowStart := currentTime.Add(-h.windowDuration)
    ///     newTimestamps := make([]histogramEntry, 0)
    ///     for _, entry := range h.timestamps {
    ///         if entry.timestamp.After(windowStart) {
    ///             newTimestamps = append(newTimestamps, entry)
    ///         } else {
    ///             // decrement node stats and pod-allocation counts;
    ///             // delete the node entirely if histogram[node] <= 0.
    ///         }
    ///     }
    ///     h.timestamps = newTimestamps
    /// }
    /// ```
    fn remove_old_entries(&mut self, current_time: Instant) {
        let window_start = current_time - self.window_duration;
        let tree = &mut self.tree;

        let mut new_timestamps = Vec::with_capacity(self.timestamps.len());
        for entry in self.timestamps.drain(..) {
            if entry.timestamp > window_start {
                new_timestamps.push(entry);
                continue;
            }
            // Decrement node stats and per-(node, replica) count.
            let mut should_drop = false;
            if let Some((_, payload)) = tree.longest_match_mut(&entry.prefix) {
                let stats = &mut payload.stats;
                stats.histogram = stats.histogram.saturating_sub(entry.leaf_context_length);
                stats.node_to_count = stats.node_to_count.saturating_sub(1);
                stats.hit_tokens = stats.hit_tokens.saturating_sub(
                    entry.leaf_context_length.saturating_sub(entry.leaf_num_tokens),
                );
                stats.prompt_tokens = stats.prompt_tokens.saturating_sub(entry.leaf_context_length);
                if stats.decrement_replica_count(entry.replica_id) {
                    payload.owners.remove(entry.replica_id);
                }
                should_drop = stats.node_to_count == 0;
            }
            if should_drop {
                tree.remove_path_with(&entry.prefix, |p| {
                    p.stats.node_to_count == 0 && p.owners.is_empty()
                });
            }
        }
        self.timestamps = new_timestamps;
    }

    /// Compute the prefill cost for a node's stats.
    ///
    /// Go lines 201-231 (inside `getPrefillCost`):
    /// ```go
    /// missRate := 1.0
    /// if h.promptTokens[node] > 0 {
    ///     missRate = 1.0 - (float64(h.hitTokens[node]) / float64(h.promptTokens[node]))
    /// }
    /// // ... prefillTime := (baseTime + attnQuad) / 0.9 ...
    /// numPods := node.GetModelToPodCount()
    /// totalPrefillCost := missRate * float64(h.nodeToCount[node]) * prefillTime / float64(numPods)
    /// ```
    fn prefill_cost_for(&self, stats: &NodeStats, num_owners: usize) -> f64 {
        if stats.node_to_count == 0 {
            return 0.0;
        }
        let miss_rate = if stats.prompt_tokens > 0 {
            1.0 - (stats.hit_tokens as f64 / stats.prompt_tokens as f64)
        } else {
            1.0
        };
        // Approximate the node's per-request work: average context
        // length and uncached-token count derived from the windowed
        // accumulators.
        let avg_context = stats.histogram / stats.node_to_count.max(1);
        let avg_hit = stats.hit_tokens / stats.node_to_count.max(1);
        let avg_num_tokens = avg_context.saturating_sub(avg_hit);
        let prefill_t = cost_model::prefill_time(self.target_gpu, avg_num_tokens, avg_context);
        let n_owners = num_owners.max(1) as f64;
        miss_rate * (stats.node_to_count as f64) * prefill_t / n_owners
    }

    /// Compute total cost (prefill + decode) for a node on a specific
    /// replica, given the replica's live TPOT.
    ///
    /// Go lines 341-353:
    /// ```go
    /// func (h *SlidingWindowHistogram) getNodeCost(node *TreeNode, podName string) float64 {
    ///     prefillCost := h.getPrefillCost(node)
    ///     timePerToken := 0.15  // seconds, default
    ///     if times, ok := h.avgTimePerTokenPerPod[podName]; ok && len(times) > 0 {
    ///         sort.Float64s(times)
    ///         timePerToken = times[len(times)/2]  // median
    ///     }
    ///     outputLen := h.decodingSize[node]
    ///     decodeCost := float64(outputLen) * timePerToken
    ///     return prefillCost + decodeCost
    /// }
    /// ```
    ///
    /// `replica_tpot_ms` comes from `LMetric.tpot` via
    /// `Observation.tpot`. NaN / non-positive values fall back to
    /// `DEFAULT_TIME_PER_TOKEN_MS`.
    fn node_cost_for(&self, stats: &NodeStats, num_owners: usize, replica_tpot_ms: f64) -> f64 {
        let prefill_cost = self.prefill_cost_for(stats, num_owners);
        let tpt_ms = if replica_tpot_ms.is_finite() && replica_tpot_ms > 0.0 {
            replica_tpot_ms
        } else {
            DEFAULT_TIME_PER_TOKEN_MS
        };
        // Match Go's units: prefill_cost is in seconds (cost_model
        // returns seconds), tpot is in ms there too (0.15 default), and
        // outputLen * timePerToken is also in seconds. Our LMetric.tpot
        // is in ms — divide by 1000 to align.
        let output_len = if stats.decoding_size > 0 {
            stats.decoding_size
        } else {
            self.default_decoding_length
        };
        let decode_cost = output_len as f64 * (tpt_ms / 1000.0);
        prefill_cost + decode_cost
    }

    /// Total allocation cost for a single replica, summing
    /// `node_cost_for` across all nodes that the replica currently
    /// owns.
    ///
    /// Bijective to Go's `getCurrentAllocationCostPerPod` walk that
    /// only considers `node.GetModelToPods()` — i.e. only nodes the
    /// pod actually serves contribute.
    pub fn cost_for_replica(&self, replica_id: usize, replica_tpot_ms: f64) -> f64 {
        let mut total = 0.0_f64;
        self.tree.for_each_path(|_path, payload| {
            if !payload.owners.contains(replica_id) {
                return;
            }
            let n_owners = payload.owners.len();
            total += self.node_cost_for(&payload.stats, n_owners, replica_tpot_ms);
        });
        total
    }

    /// Number of in-window requests routed through nodes owned by
    /// `replica_id`. Used for Stage 1 longest-match tie-break.
    ///
    /// Go lines 569-582:
    /// ```go
    /// func (h *SlidingWindowHistogram) getPodLoad(pod *v1.Pod) int {
    ///     load := 0
    ///     for node, count := range h.nodeToCount {
    ///         for _, podMap := range node.GetModelToPods() {
    ///             if _, exists := podMap[pod.Name]; exists {
    ///                 load += count; break
    ///             }
    ///         }
    ///     }
    ///     return load
    /// }
    /// ```
    pub fn load_for_replica(&self, replica_id: usize) -> usize {
        let mut total = 0usize;
        self.tree.for_each_path(|_path, payload| {
            if payload.owners.contains(replica_id) {
                total += payload.stats.node_to_count;
            }
        });
        total
    }

    /// Number of distinct trie nodes currently in the histogram.
    pub fn num_nodes(&self) -> usize {
        self.tree.len()
    }

    /// Number of replicas configured at construction.
    pub fn num_replicas(&self) -> usize {
        self.num_replicas
    }

    /// Default per-request decode length used when a node has no
    /// recorded `decoding_size` yet.
    pub fn default_decoding_length(&self) -> usize {
        self.default_decoding_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_records_owner_and_stats() {
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[10, 20], 100, 200, 0, 45);
        // The path [10, 20] should now exist; the deepest node is the
        // one that holds owner=0 and the stats.
        let payload = h.tree.exact(&[10, 20]).unwrap();
        assert_eq!(payload.stats.node_to_count, 1);
        assert_eq!(payload.stats.histogram, 200);
        assert_eq!(payload.stats.hit_tokens, 100); // 200 - 100
        assert_eq!(payload.stats.prompt_tokens, 200);
        assert!(payload.owners.contains(0));
        assert!(!payload.owners.contains(1));
    }

    #[test]
    fn cost_differs_per_replica_when_owners_differ() {
        // Replicas 0 and 2 each touch overlapping prefixes; replica 1
        // touches none. cost_for_replica(1) should be 0; cost_for_replica(0)
        // and (2) should be > 0 and roughly comparable (each owns its own
        // path plus the shared prefix).
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[10, 20, 30], 256, 1024, 0, 45);
        h.update(&[10, 20, 40], 256, 1024, 2, 45);
        let c0 = h.cost_for_replica(0, 50.0);
        let c1 = h.cost_for_replica(1, 50.0);
        let c2 = h.cost_for_replica(2, 50.0);
        assert!(c0 > 0.0, "replica 0 owns nodes; cost_for_replica(0) > 0");
        assert!(c2 > 0.0, "replica 2 owns nodes; cost_for_replica(2) > 0");
        assert_eq!(c1, 0.0, "replica 1 owns no nodes; cost_for_replica(1) == 0");
    }

    #[test]
    fn load_per_replica_counts_owned_nodes() {
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[10, 20, 30], 256, 1024, 0, 45);
        h.update(&[10, 20, 30], 256, 1024, 0, 45);
        h.update(&[10, 20, 40], 256, 1024, 1, 45);
        // Replica 0 served 2 requests; both pass through 3 nodes ([10],
        // [10,20], [10,20,30]) — load = 2 * 3 = 6 (each node's
        // node_to_count is summed for the replica's owned nodes).
        // Replica 1 served 1 request; passes through 3 nodes too.
        // BUT [10] and [10,20] are co-owned by 0 and 1, so each
        // contributes to BOTH load counts.
        let load0 = h.load_for_replica(0);
        let load1 = h.load_for_replica(1);
        assert!(load0 > 0);
        assert!(load1 > 0);
        assert!(load0 > load1, "replica 0 served more requests: {load0} vs {load1}");
    }

    #[test]
    fn empty_load_for_unknown_replica() {
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[10, 20], 256, 1024, 0, 45);
        assert_eq!(h.load_for_replica(2), 0);
        assert_eq!(h.cost_for_replica(2, 50.0), 0.0);
    }

    #[test]
    fn per_replica_eviction_drops_owner_when_count_zero() {
        // Manually drive the eviction path without waiting for the wall
        // clock by calling the inner `remove_old_entries` directly with
        // a synthesised "now".
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        // Use the constructor but manually set the window to 0 so any
        // future timestamp evicts everything.
        h.window_duration = Duration::from_secs(0);
        h.update(&[10, 20], 256, 1024, 0, 45);
        // After the update, the (synthesised "later") instant should
        // already be past window_start = (now - 0). Force eviction.
        let later = Instant::now() + Duration::from_millis(1);
        h.remove_old_entries(later);
        // Path [10, 20] should be gone entirely (owner-set empty +
        // node_to_count == 0 → remove_path_with fires).
        assert!(h.tree.exact(&[10, 20]).is_none());
        assert_eq!(h.load_for_replica(0), 0);
        assert_eq!(h.cost_for_replica(0, 50.0), 0.0);
    }

    #[test]
    fn per_replica_eviction_partial_keeps_other_owner() {
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        // Both replica 0 and replica 1 touch [10, 20] before the
        // window opens. Then we evict one of them by manually crafting
        // an old timestamp — a behaviour test that the per-(node,
        // replica) decrement removes ONLY replica 0 from owners,
        // keeping replica 1.
        h.update(&[10, 20], 256, 1024, 0, 45);
        h.update(&[10, 20], 256, 1024, 1, 45);
        // Now make the first entry old by hand-rewriting its
        // timestamp; the second stays fresh. Use a borrow scope so
        // we can mutate `timestamps` without conflicting with the
        // immutable read for the assertion afterwards.
        {
            let old = h.timestamps[0].timestamp - Duration::from_secs(10000);
            h.timestamps[0].timestamp = old;
        }
        // Sliding window is the default 3 minutes, so the rewritten
        // entry will be considered expired.
        let now = Instant::now();
        h.remove_old_entries(now);
        // Replica 0 lost its only entry on [10, 20] → no longer owner.
        // Replica 1 still owns the node.
        let payload = h.tree.exact(&[10, 20]).unwrap();
        assert!(!payload.owners.contains(0));
        assert!(payload.owners.contains(1));
        assert_eq!(payload.stats.node_to_count, 1);
    }

    #[test]
    fn empty_prefix_update_is_noop() {
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[], 256, 1024, 0, 45);
        assert_eq!(h.num_nodes(), 0);
    }
}
