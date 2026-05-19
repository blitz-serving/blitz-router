// Sliding window histogram for Preble cost model state tracking.
//
// Bijective translation from AIBrix's Go implementation:
//   aibrix/pkg/plugins/gateway/algorithms/prefix_cache_preble.go
//   Lines 53-82  (SlidingWindowHistogram struct)
//   Lines 262-318 (removeEvictedNodes / removeOldEntries)
//   Lines 320-329 (evictionLoop — driven externally by the policy)
//   Lines 341-353 (getNodeCost)
//   Lines 355-369 (getCurrentAllocationCostPerPod)
//   Lines 437-565 (Route — ancestor walk for owner-set update at lines 555-560)
//   Lines 569-582 (getPodLoad)
//   Lines 585-610 (update)
//
// In blitz-router, "pod" maps to "replica" (replica_id: usize). "TreeNode"
// — Go's prefix-cache trie node, identified by `*TreeNode` pointer — maps
// to a path in our `Trie<u64, OwnedNodeStats>` keyed by the request's
// FULL block-hash sequence.
//
// Ownership semantics (1:1 with Go's `TreeNode.modelToPods`):
//   * Replicas are added to ALL ancestors of the matched leaf when a
//     request is dispatched (Go: `Route` lines 555-560).
//   * Ownership is STICKY across the sliding window — `removeOldEntries`
//     decrements per-node aggregate stats but does NOT touch ownership.
//     Ownership is cleared only by LRU subtree eviction
//     (`evict_stale_subtrees` here, mirroring Go's
//     `LPRadixCache.Evict` + `removeEvictedNodes`).
//   * Per-replica observations (TPOT, live load) are NOT stored here.
//     They live on `LMetric` in `ScheduleContext`, snapshotted into
//     `Observation` by `dsl_runtime::capture_observations`. The Go
//     `avgTimePerTokenPerPod` map is dead code in production
//     (never written), so the cost model uses the constant 0.15 s/tok.

use std::time::{Duration, Instant};

use radixtree::{ReplicaSet, Trie};

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
/// In our port the "node identity" is the FULL block-hash path of the
/// request (stored as `Box<[u64]>`); we need to remember it on the
/// entry so `remove_old_entries` can find the right tree node when the
/// entry expires. We do NOT track `replica_id` for owner-set decay —
/// owners are sticky, decayed only by LRU eviction.
#[derive(Debug, Clone)]
struct HistogramEntry {
    timestamp: Instant,
    /// Path identifying the prefix node this entry was recorded under
    /// (the full block-hash sequence, 1:1 with Go's `node` returned
    /// from `cache.AddPrefix(tokens, ...)`).
    prefix: Box<[u64]>,
    /// Number of new (uncached) tokens this request contributed.
    leaf_num_tokens: usize,
    /// Total context length up to and including this prefix node.
    leaf_context_length: usize,
}

/// Per-node accumulated statistics within the sliding window.
///
/// Go's `histogram[*TreeNode]int`, `nodeToCount[*TreeNode]int`, etc.
/// collapse into one struct living on the trie node payload.
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
}

/// Trie payload: per-node owner set + accumulated stats + last-access
/// time for LRU eviction. The owner set is sticky across the sliding
/// window — `last_access` drives the only ownership-clearing path
/// (LRU eviction, mirroring Go's `LPRadixCache.Evict`).
#[derive(Debug, Clone)]
pub(crate) struct OwnedNodeStats {
    pub owners: ReplicaSet,
    pub stats: NodeStats,
    /// Most recent time this node (or any descendant traversal) touched
    /// this node. `None` only between `TrieNode::default()` and the
    /// first `ensure_path_with` callback; in production every visited
    /// node is stamped before being read. Used by
    /// `evict_stale_subtrees`.
    pub last_access: Option<Instant>,
}

impl Default for OwnedNodeStats {
    fn default() -> Self {
        Self {
            owners: ReplicaSet::default(),
            stats: NodeStats::default(),
            last_access: None,
        }
    }
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
///   * Per-node aggregate fields collapse into `NodeStats` on the trie
///     node payload.
///   * `podAllocations[*TreeNode]` is dead code in Go (never written
///     in production — only tests assign). Replaced by
///     `OwnedNodeStats.owners`, populated via the Route-flow ancestor
///     walk at Go lines 555-560.
///   * `avgTimePerTokenPerPod` is dead code in Go (never written in
///     production — only tests assign). The cost model uses the
///     constant 0.15 s/tok at line 345.
///   * `currentDecodeLengthsPerPod` is dropped (never read by any
///     scheduling-side path).
pub(crate) struct SlidingWindowHistogram {
    /// Sliding window duration for stat decay.
    /// Go: `slidingWindowPeriod = 3 * time.Minute`
    window_duration: Duration,

    /// LRU max-age — a node whose `last_access` is older than this is
    /// dropped along with all descendants. Go:
    /// `evictionDuration = 5 * time.Minute` (`tree.go:29`).
    lru_max_age: Duration,

    /// Background eviction tick period. Go: `evictionLoopInterval =
    /// 1 * time.Second` (`prefix_cache_preble.go:50`). We don't run a
    /// dedicated ticker; `evict_if_due` throttles to this interval and
    /// is called lazily from the per-request `update` path (1:1 with
    /// Go's eventual-consistency between Route and evictionLoop).
    tick_interval: Duration,

    /// Last time `evict_if_due` actually fired (both decay passes).
    /// `None` until the first eviction; first call always fires.
    last_tick: Option<Instant>,

    /// Prefix trie keyed by block hash, payload = (owners, stats,
    /// last_access). Replaces Go's per-node maps + `LPRadixCache` +
    /// `podAllocations`.
    tree: Trie<u64, OwnedNodeStats>,

    /// Ordered list of entries for sliding-window stat decay.
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

/// Default LRU eviction age (Go: 5 minutes; `tree.go:29`).
const DEFAULT_LRU_MAX_AGE: Duration = Duration::from_secs(5 * 60);

/// Default eviction tick period (Go: 1 second;
/// `prefix_cache_preble.go:50`).
const DEFAULT_TICK_INTERVAL: Duration = Duration::from_secs(1);

/// Default expected output length (Go: 45 tokens).
const DEFAULT_DECODING_LENGTH: usize = 45;

/// Default per-token decode time when the caller passes a non-finite
/// or zero `replica_tpot_ms` (in milliseconds — matches `LMetric.tpot`
/// units). Go uses a hardcoded 0.15 s (`prefix_cache_preble.go:345`);
/// keeping the per-replica TPOT parameter and falling back to this
/// default is a deviation from Go (D6) — deferred for user discussion.
const DEFAULT_TIME_PER_TOKEN_MS: f64 = 150.0;

impl SlidingWindowHistogram {
    /// Create a new histogram.
    pub fn new(num_replicas: usize, target_gpu: TargetGpu) -> Self {
        Self {
            window_duration: DEFAULT_WINDOW_DURATION,
            lru_max_age: DEFAULT_LRU_MAX_AGE,
            tick_interval: DEFAULT_TICK_INTERVAL,
            last_tick: None,
            tree: Trie::new(),
            timestamps: Vec::new(),
            num_replicas,
            target_gpu,
            default_decoding_length: DEFAULT_DECODING_LENGTH,
        }
    }

    /// Update histogram with a new request assignment.
    ///
    /// Go's combined Route-flow effect (lines 555-562):
    /// ```go
    /// // Add chosen pod to every ancestor of the matched leaf.
    /// currentNode := node
    /// for currentNode != nil {
    ///     currentNode.AddOrUpdatePodForModel(ctx.Model, targetPod.Name, time.Now())
    ///     currentNode = currentNode.GetParent()
    /// }
    /// // Then update the LEAF node's aggregate stats.
    /// p.histogram.update(time.Now(), node, node, targetPod.Name, decodingLength)
    /// ```
    ///
    /// `prefix` is the FULL block-hash path of the request (output of
    /// `entry.block_hash_state.get_hashes()`). `context_length` is the
    /// FULL input token count (= `prefix.len() * block_size` for
    /// requests aligned to the block boundary, plus any tail).
    /// `num_tokens` is the count of NEW (uncached) tokens this request
    /// contributed at the leaf — Go: `leafNode.NumTokens()`.
    pub fn update(
        &mut self,
        prefix: &[u64],
        num_tokens: usize,
        context_length: usize,
        replica_id: usize,
        decoding_length: usize,
    ) {
        if prefix.is_empty() {
            // Go's `node` is the root sentinel for no-prefix-match
            // requests. We elide it (it has no per-node cost
            // contribution — root never appears in `histogram` map).
            return;
        }
        let now = Instant::now();
        let num_replicas = self.num_replicas;

        // Walk leaf-to-root semantically, but the trie API walks
        // top-down. Same effect: every ancestor + leaf gets the chosen
        // replica added to its owner set. Go: `prefix_cache_preble.go:555-560`.
        let payload = self.tree.ensure_path_with(prefix, |_depth, p| {
            if p.owners.is_empty() && p.stats.node_to_count == 0 {
                // Lazy-init the bitset on first touch.
                p.owners = ReplicaSet::new(num_replicas);
            }
            p.owners.insert(replica_id);
            p.last_access = Some(now);
        });

        // Leaf-only stat update — Go: `histogram.update` lines 595-609,
        // which only writes to maps keyed by `node` (the leaf).
        let stats = &mut payload.stats;
        stats.histogram += context_length;
        stats.node_to_count += 1;
        stats.decoding_size = decoding_length;
        stats.hit_tokens += context_length.saturating_sub(num_tokens);
        stats.prompt_tokens += context_length;
        stats.total_decode_lengths += decoding_length;

        self.timestamps.push(HistogramEntry {
            timestamp: now,
            prefix: prefix.into(),
            leaf_num_tokens: num_tokens,
            leaf_context_length: context_length,
        });
    }

    /// Remove sliding-window-expired entries: decrement leaf stats only.
    /// Owners are sticky and untouched by this path — they decay only
    /// via LRU eviction (`evict_stale_subtrees`).
    ///
    /// Go lines 292-318: the "delete pod allocations" branch in Go's
    /// `removeOldEntries` is a no-op in production (`podAllocations`
    /// inner maps are never populated), so we drop that branch entirely.
    pub fn remove_old_entries(&mut self, current_time: Instant) {
        let window_start = current_time - self.window_duration;
        let tree = &mut self.tree;

        let mut new_timestamps = Vec::with_capacity(self.timestamps.len());
        for entry in self.timestamps.drain(..) {
            if entry.timestamp > window_start {
                new_timestamps.push(entry);
                continue;
            }
            // Entry expired — decrement leaf-node stats only. Go:
            // `histogram.removeOldEntries` lines 303-314 (without the
            // `delete podAllocations` branch).
            if let Some((_, payload)) = tree.longest_match_mut(&entry.prefix) {
                let stats = &mut payload.stats;
                stats.histogram = stats.histogram.saturating_sub(entry.leaf_context_length);
                stats.node_to_count = stats.node_to_count.saturating_sub(1);
                stats.hit_tokens = stats.hit_tokens.saturating_sub(
                    entry.leaf_context_length.saturating_sub(entry.leaf_num_tokens),
                );
                stats.prompt_tokens = stats.prompt_tokens.saturating_sub(entry.leaf_context_length);
            }
        }
        self.timestamps = new_timestamps;
    }

    /// LRU eviction — drop nodes (and all descendants) whose
    /// `last_access` is older than `lru_max_age`. Mirrors Go's
    /// `LPRadixCache.Evict` (`tree.go:499`) + `removeEvictedNodes`
    /// (`prefix_cache_preble.go:262-290`): owners and stats go away
    /// together when the node is evicted.
    ///
    /// Returns the number of nodes evicted. Bumps the trie's epoch
    /// once if any evictions happened.
    pub fn evict_stale_subtrees(&mut self, current_time: Instant) -> usize {
        let max_age = self.lru_max_age;
        let removed = self.tree.evict_subtrees_where(|p| {
            p.last_access
                .map(|t| current_time.saturating_duration_since(t) > max_age)
                .unwrap_or(true)
        });
        if removed > 0 {
            // Drop timestamps that point at evicted nodes — Go:
            // `removeEvictedNodes` filters `h.timestamps`. We don't
            // have an explicit list of evicted prefixes, so we filter
            // by re-checking tree presence.
            let tree = &self.tree;
            self.timestamps.retain(|e| tree.exact(&e.prefix).is_some());
        }
        removed
    }

    /// Run both decay passes (`remove_old_entries` then
    /// `evict_stale_subtrees`) iff at least `tick_interval` has
    /// elapsed since the last tick. Called lazily from
    /// `update_histogram_into` to mirror Go's 1Hz `evictionLoop`
    /// without spawning a separate task.
    ///
    /// Returns `true` if eviction ran this call.
    pub fn evict_if_due(&mut self, current_time: Instant) -> bool {
        let due = match self.last_tick {
            None => true,
            Some(prev) => current_time.saturating_duration_since(prev) >= self.tick_interval,
        };
        if !due {
            return false;
        }
        self.remove_old_entries(current_time);
        self.evict_stale_subtrees(current_time);
        self.last_tick = Some(current_time);
        true
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

    /// Compute total cost (prefill + decode) for a node, using a
    /// per-replica TPOT signal for the decode term.
    ///
    /// Go lines 341-353:
    /// ```go
    /// func (h *SlidingWindowHistogram) getNodeCost(node *TreeNode, podName string) float64 {
    ///     prefillCost := h.getPrefillCost(node)
    ///     timePerToken := 0.15
    ///     if times, ok := h.avgTimePerTokenPerPod[podName]; ok && len(times) > 0 {
    ///         sort.Float64s(times)
    ///         timePerToken = times[len(times)/2]
    ///     }
    ///     outputLen := h.decodingSize[node]
    ///     decodeCost := float64(outputLen) * timePerToken
    ///     return prefillCost + decodeCost
    /// }
    /// ```
    ///
    /// `replica_tpot_ms` comes from `LMetric.tpot` (already snapshot
    /// per-replica into `Observation.tpot`); NaN / non-positive falls
    /// back to `DEFAULT_TIME_PER_TOKEN_MS`. Note this is a deviation
    /// from Go's hardcoded 0.15 s — deferred for user discussion (D6).
    fn node_cost_for(&self, stats: &NodeStats, num_owners: usize, replica_tpot_ms: f64) -> f64 {
        let prefill_cost = self.prefill_cost_for(stats, num_owners);
        let tpt_ms = if replica_tpot_ms.is_finite() && replica_tpot_ms > 0.0 {
            replica_tpot_ms
        } else {
            DEFAULT_TIME_PER_TOKEN_MS
        };
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
    /// iterates `h.histogram` and adds `getNodeCost` for each
    /// (node, pod) where the pod owns the node —
    /// `prefix_cache_preble.go:355-369`.
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

    /// Length (in blocks) of the longest prefix match of `prefix`
    /// against the global Preble tree, regardless of owners. Bijective
    /// with Go's `len(matchedTokens)` from `cache.AddPrefix(tokens, ...)`
    /// at `prefix_cache_preble.go:459`. Returns `0` if no node along
    /// `prefix` exists in the tree.
    ///
    /// Used by Stage 1's threshold check: enter Stage 1 iff
    /// `match_blocks * block_size / input_len > 0.5`.
    pub fn global_match_blocks(&self, prefix: &[u64]) -> usize {
        match self.tree.longest_match(prefix) {
            Some((d, _)) => d,
            None => 0,
        }
    }

    /// Length (in blocks) of the longest prefix of `prefix` whose
    /// matched tree node has `replica_id` as an owner. Walks down
    /// `prefix` and returns the deepest depth at which `replica_id` is
    /// in the node's owner set.
    ///
    /// Bijective with Go's Stage 1 ancestor walk at
    /// `prefix_cache_preble.go:484-505` (followed by sort by
    /// `matchLength` descending and pick of `prefixMatches[0]`):
    /// `prefixMatches[0].matchLength` for the candidate replica is
    /// exactly this depth × `block_size`.
    ///
    /// Returns `0` if no ancestor of `prefix` is owned by `replica_id`.
    pub fn owned_match_blocks(&self, prefix: &[u64], replica_id: usize) -> usize {
        match self.tree.longest_match_with(prefix, |p| p.owners.contains(replica_id)) {
            Some((d, _)) => d,
            None => 0,
        }
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
    fn update_records_owner_on_every_ancestor() {
        // D2: owner must be added to leaf AND all ancestors. Go:
        // `prefix_cache_preble.go:555-560`.
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[10, 20, 30], 100, 200, 0, 45);

        // Ancestor [10] should also have owner 0.
        let p1 = h.tree.exact(&[10]).unwrap();
        assert!(p1.owners.contains(0), "owner 0 must be on depth-1 ancestor [10]");
        // Ancestor [10, 20].
        let p2 = h.tree.exact(&[10, 20]).unwrap();
        assert!(p2.owners.contains(0), "owner 0 must be on depth-2 ancestor [10, 20]");
        // Leaf [10, 20, 30].
        let p3 = h.tree.exact(&[10, 20, 30]).unwrap();
        assert!(p3.owners.contains(0), "owner 0 must be on the leaf");

        // Stats only on leaf — Go's `histogram.update` only writes to
        // maps keyed by the leaf node.
        assert_eq!(p3.stats.node_to_count, 1);
        assert_eq!(p3.stats.histogram, 200);
        assert_eq!(p3.stats.hit_tokens, 100); // 200 - 100
        assert_eq!(p3.stats.prompt_tokens, 200);
        assert_eq!(p1.stats.node_to_count, 0, "ancestors carry no aggregate stats");
        assert_eq!(p2.stats.node_to_count, 0);
    }

    #[test]
    fn ownership_is_sticky_across_window() {
        // Sliding-window expiry MUST NOT clear owners. Only LRU
        // eviction does. Go: `removeOldEntries` only deletes histogram
        // map entries, never `TreeNode.modelToPods`.
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[10, 20], 100, 200, 0, 45);
        // Force the entry stale (window = 0 makes everything stale).
        h.window_duration = Duration::from_secs(0);
        let later = Instant::now() + Duration::from_secs(1);
        h.remove_old_entries(later);
        // Stats should be decayed; owners should remain.
        let p = h.tree.exact(&[10, 20]).unwrap();
        assert!(p.owners.contains(0), "owners persist across window expiry");
        assert_eq!(p.stats.node_to_count, 0, "stats are decayed");
    }

    #[test]
    fn lru_evicts_subtree_clearing_owners_and_stats() {
        // LRU eviction removes the node + descendants entirely — owners
        // and stats vanish together. Go: `LPRadixCache.Evict` +
        // `removeEvictedNodes`.
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[10, 20], 100, 200, 0, 45);
        h.update(&[10, 30], 100, 200, 1, 45);
        // Push max-age to 0 so any node is "stale" (its last_access is
        // strictly less than `now + 1ms`).
        h.lru_max_age = Duration::from_secs(0);
        let later = Instant::now() + Duration::from_secs(1);
        let removed = h.evict_stale_subtrees(later);
        // [10], [10,20], [10,30] all evicted — 3 nodes.
        assert_eq!(removed, 3);
        assert_eq!(h.num_nodes(), 0);
        assert_eq!(h.cost_for_replica(0, 50.0), 0.0);
        assert_eq!(h.load_for_replica(0), 0);
    }

    #[test]
    fn cost_differs_per_replica_when_owners_differ() {
        // Replicas 0 and 2 each touch overlapping prefixes; replica 1
        // touches none. cost_for_replica(1) should be 0.
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
        // Replica 0 owns [10], [10,20], [10,20,30] — leaf has count 2,
        // ancestors have count 0 (stats only at leaf). Replica 1 owns
        // the same ancestors plus [10,20,40] (count 1). Both also
        // co-own [10] and [10,20] from cross-replica updates.
        // Net: load_for_replica(0) = leaf count = 2.
        //      load_for_replica(1) = leaf count = 1.
        assert_eq!(h.load_for_replica(0), 2);
        assert_eq!(h.load_for_replica(1), 1);
    }

    #[test]
    fn global_match_blocks_returns_deepest_existing_path() {
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[10, 20, 30], 256, 1024, 0, 45);
        // Full match.
        assert_eq!(h.global_match_blocks(&[10, 20, 30]), 3);
        // Partial — [10, 20] exists, [99] doesn't.
        assert_eq!(h.global_match_blocks(&[10, 20, 99]), 2);
        // Path that diverges at the start.
        assert_eq!(h.global_match_blocks(&[99]), 0);
    }

    #[test]
    fn owned_match_blocks_returns_deepest_owned_ancestor() {
        // D3: Stage 1 winner is the replica with the deepest owned
        // ancestor. Go: `prefix_cache_preble.go:484-509`.
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        // Replica 0 owns [10], [10,20], [10,20,30] — full path.
        h.update(&[10, 20, 30], 256, 1024, 0, 45);
        // Replica 1 owns only [10] (via a request with diverging path).
        h.update(&[10, 99], 256, 1024, 1, 45);
        // For request [10, 20, 30, ...]: replica 0's deepest owned
        // ancestor is [10,20,30] (depth 3); replica 1's is [10] (depth 1).
        assert_eq!(h.owned_match_blocks(&[10, 20, 30, 40], 0), 3);
        assert_eq!(h.owned_match_blocks(&[10, 20, 30, 40], 1), 1);
        // Replica 2 owns nothing along this path.
        assert_eq!(h.owned_match_blocks(&[10, 20, 30, 40], 2), 0);
    }

    #[test]
    fn empty_prefix_update_is_noop() {
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[], 256, 1024, 0, 45);
        assert_eq!(h.num_nodes(), 0);
    }

    #[test]
    fn old_entries_only_decay_stats_not_owners() {
        // Two replicas touch the same node within the window; expire
        // the entries; verify owners survive but stats decay.
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[10, 20], 256, 1024, 0, 45);
        h.update(&[10, 20], 256, 1024, 1, 45);
        {
            let entry = &mut h.timestamps[0];
            entry.timestamp = entry.timestamp - Duration::from_secs(10_000);
        }
        let now = Instant::now();
        h.remove_old_entries(now);
        let p = h.tree.exact(&[10, 20]).unwrap();
        // Both replicas remain owners (sticky).
        assert!(p.owners.contains(0));
        assert!(p.owners.contains(1));
        // Stats decremented by exactly one entry.
        assert_eq!(p.stats.node_to_count, 1);
    }

    #[test]
    fn evict_if_due_throttles_to_tick_interval() {
        // First call fires (no prior tick). Second call within
        // tick_interval is a no-op. Call after tick_interval fires
        // again. Mirrors Go's 1Hz `evictionLoop` cadence
        // (`prefix_cache_preble.go:50`).
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        let t0 = Instant::now();
        assert!(h.evict_if_due(t0), "first call must fire");
        // Within tick_interval — should be a no-op.
        let t1 = t0 + Duration::from_millis(100);
        assert!(!h.evict_if_due(t1), "within tick_interval must skip");
        // Past tick_interval — fires again.
        let t2 = t0 + h.tick_interval + Duration::from_millis(1);
        assert!(h.evict_if_due(t2), "after tick_interval must fire");
    }

    #[test]
    fn evict_if_due_drains_stale_entries() {
        // Insert one entry, age it past the window, then trigger
        // evict_if_due — the stat decay path should fire.
        let mut h = SlidingWindowHistogram::new(4, TargetGpu::V100);
        h.update(&[10, 20], 256, 1024, 0, 45);
        // Backdate the entry past the window.
        {
            let e = &mut h.timestamps[0];
            e.timestamp = e.timestamp - Duration::from_secs(10_000);
        }
        let now = Instant::now() + Duration::from_secs(2); // > tick_interval
        assert!(h.evict_if_due(now));
        // Stats decayed (the entry was past window_duration), but
        // owners stick (LRU age 5 min not reached for a fresh-touch
        // node).
        let p = h.tree.exact(&[10, 20]).unwrap();
        assert_eq!(p.stats.node_to_count, 0);
        assert!(p.owners.contains(0));
    }
}
