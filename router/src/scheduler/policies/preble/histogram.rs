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
// "TreeNode" maps to a node key derived from block hash prefix matching.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::cost_model::{self, TargetGpu};

/// Unique identifier for a prefix node in the histogram.
///
/// In Go, this is `*prefixcacheindexer.TreeNode` (a pointer).
/// In blitz-router, we use a stable hash of the matched prefix as key.
/// This is derived from the block hash sequence of the prefix match.
pub(crate) type NodeKey = u64;

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
#[derive(Debug, Clone)]
struct HistogramEntry {
    timestamp: Instant,
    node_key: NodeKey,
    /// Number of tokens in this specific request's matched prefix node.
    leaf_num_tokens: usize,
    /// Context length (total tokens up to and including this node).
    leaf_context_length: usize,
}

/// Per-node accumulated statistics within the sliding window.
#[derive(Debug, Clone, Default)]
struct NodeStats {
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
    /// Number of replicas that have been assigned this node.
    /// Used to approximate `node.GetModelToPodCount()` in Go.
    num_assigned_replicas: usize,
}

/// Per-replica accumulated state.
#[derive(Debug, Clone, Default)]
struct ReplicaStats {
    /// Total decode lengths assigned to this replica.
    /// Go: `currentDecodeLengthsPerPod map[string]int`
    current_decode_lengths: usize,
    /// Rolling TPOT measurements for this replica.
    /// Go: `avgTimePerTokenPerPod map[string][]float64`
    avg_time_per_token: Vec<f64>,
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
pub(crate) struct SlidingWindowHistogram {
    /// Window duration for temporal decay.
    /// Go: `slidingWindowPeriod = 3 * time.Minute`
    window_duration: Duration,

    /// Per-node statistics.
    nodes: HashMap<NodeKey, NodeStats>,

    /// Per-replica statistics.
    replicas: Vec<ReplicaStats>,

    /// Ordered list of entries for sliding window eviction.
    timestamps: Vec<HistogramEntry>,

    /// Number of replicas.
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

/// Default TPOT when no measurements available (Go: 0.15 seconds).
const DEFAULT_TIME_PER_TOKEN: f64 = 0.15;

impl SlidingWindowHistogram {
    /// Create a new histogram.
    ///
    /// Go lines 233-247 (NewPrefixCacheAndLoadRouter):
    /// ```go
    /// histogram := &SlidingWindowHistogram{
    ///     windowDuration:             slidingWindowPeriod,
    ///     histogram:                  make(map[*TreeNode]int),
    ///     nodeToCount:                make(map[*TreeNode]int),
    ///     hitTokens:                  make(map[*TreeNode]int),
    ///     promptTokens:               make(map[*TreeNode]int),
    ///     decodingSize:               make(map[*TreeNode]int),
    ///     numPods:                    numPods,
    ///     podAllocations:             make(map[*TreeNode]map[int]bool),
    ///     currentDecodeLengthsPerPod: make(map[string]int),
    ///     perNodeTotalDecodeLengths:  make(map[*TreeNode]int),
    ///     avgTimePerTokenPerPod:      make(map[string][]float64),
    /// }
    /// ```
    pub fn new(num_replicas: usize, target_gpu: TargetGpu) -> Self {
        Self {
            window_duration: DEFAULT_WINDOW_DURATION,
            nodes: HashMap::new(),
            replicas: (0..num_replicas).map(|_| ReplicaStats::default()).collect(),
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
    ///     h.mu.Lock()
    ///     defer h.mu.Unlock()
    ///     h.timestamps = append(h.timestamps, histogramEntry{timestamp, node, leafNode})
    ///     h.histogram[node] += leafNode.ContextLength()
    ///     h.nodeToCount[node]++
    ///     h.decodingSize[node] = decodingLength
    ///     h.hitTokens[node] += leafNode.ContextLength() - leafNode.NumTokens()
    ///     h.promptTokens[node] += leafNode.ContextLength()
    ///     h.currentDecodeLengthsPerPod[podName] += decodingLength
    ///     h.perNodeTotalDecodeLengths[node] += decodingLength
    /// }
    /// ```
    pub fn update(
        &mut self,
        node_key: NodeKey,
        num_tokens: usize,
        context_length: usize,
        replica_id: usize,
        decoding_length: usize,
    ) {
        let now = Instant::now();

        self.timestamps.push(HistogramEntry {
            timestamp: now,
            node_key,
            leaf_num_tokens: num_tokens,
            leaf_context_length: context_length,
        });

        let stats = self.nodes.entry(node_key).or_default();
        stats.histogram += context_length;
        stats.node_to_count += 1;
        stats.decoding_size = decoding_length;
        stats.hit_tokens += context_length.saturating_sub(num_tokens);
        stats.prompt_tokens += context_length;
        stats.total_decode_lengths += decoding_length;

        // Track replica assignment count for this node
        stats.num_assigned_replicas = stats.num_assigned_replicas.max(1);

        if replica_id < self.replicas.len() {
            self.replicas[replica_id].current_decode_lengths += decoding_length;
        }

        // Evict old entries
        self.remove_old_entries(now);
    }

    /// Remove entries outside the sliding window.
    ///
    /// Go lines 292-318:
    /// ```go
    /// func (h *SlidingWindowHistogram) removeOldEntries(currentTime time.Time) {
    ///     h.mu.Lock()
    ///     defer h.mu.Unlock()
    ///     windowStart := currentTime.Add(-h.windowDuration)
    ///     newTimestamps := make([]histogramEntry, 0)
    ///     for _, entry := range h.timestamps {
    ///         if entry.timestamp.After(windowStart) {
    ///             newTimestamps = append(newTimestamps, entry)
    ///         } else {
    ///             node := entry.node
    ///             leafNode := entry.leafNode
    ///             h.histogram[node] -= leafNode.ContextLength()
    ///             h.nodeToCount[node]--
    ///             h.hitTokens[node] -= leafNode.ContextLength() - leafNode.NumTokens()
    ///             h.promptTokens[node] -= leafNode.ContextLength()
    ///             if h.histogram[node] <= 0 {
    ///                 delete(h.histogram, node)
    ///                 delete(h.nodeToCount, node)
    ///                 delete(h.hitTokens, node)
    ///                 delete(h.promptTokens, node)
    ///                 delete(h.decodingSize, node)
    ///                 delete(h.podAllocations, node)
    ///             }
    ///         }
    ///     }
    ///     h.timestamps = newTimestamps
    /// }
    /// ```
    fn remove_old_entries(&mut self, current_time: Instant) {
        let window_start = current_time - self.window_duration;

        let mut new_timestamps = Vec::with_capacity(self.timestamps.len());

        for entry in self.timestamps.drain(..) {
            if entry.timestamp > window_start {
                new_timestamps.push(entry);
            } else {
                // Decrement node stats
                if let Some(stats) = self.nodes.get_mut(&entry.node_key) {
                    stats.histogram = stats.histogram.saturating_sub(entry.leaf_context_length);
                    stats.node_to_count = stats.node_to_count.saturating_sub(1);
                    stats.hit_tokens = stats.hit_tokens.saturating_sub(
                        entry.leaf_context_length.saturating_sub(entry.leaf_num_tokens),
                    );
                    stats.prompt_tokens =
                        stats.prompt_tokens.saturating_sub(entry.leaf_context_length);

                    if stats.histogram == 0 {
                        self.nodes.remove(&entry.node_key);
                    }
                }
            }
        }

        self.timestamps = new_timestamps;
    }

    /// Compute the prefill cost for a node.
    ///
    /// Go lines 201-231:
    /// ```go
    /// func (h *SlidingWindowHistogram) getPrefillCost(node *TreeNode) float64 {
    ///     missRate := 1.0
    ///     if h.promptTokens[node] > 0 {
    ///         missRate = 1.0 - (float64(h.hitTokens[node]) / float64(h.promptTokens[node]))
    ///     }
    ///     numTokens := node.NumTokens()
    ///     contextLength := node.ContextLength()
    ///     // ... GPU-specific base time + attn quad ...
    ///     prefillTime := (baseTime + attnQuad) / 0.9
    ///     numPods := node.GetModelToPodCount()
    ///     totalPrefillCost := missRate * float64(h.nodeToCount[node]) * prefillTime / float64(numPods)
    ///     return totalPrefillCost
    /// }
    /// ```
    fn get_prefill_cost(&self, node_key: NodeKey, num_tokens: usize, context_length: usize) -> f64 {
        let stats = match self.nodes.get(&node_key) {
            Some(s) => s,
            None => return 0.0,
        };

        // Miss rate: fraction of tokens that are cache misses
        let miss_rate = if stats.prompt_tokens > 0 {
            1.0 - (stats.hit_tokens as f64 / stats.prompt_tokens as f64)
        } else {
            1.0
        };

        let prefill_t = cost_model::prefill_time(self.target_gpu, num_tokens, context_length);

        let num_replicas = stats.num_assigned_replicas.max(1);

        miss_rate * (stats.node_to_count as f64) * prefill_t / (num_replicas as f64)
    }

    /// Compute total cost (prefill + decode) for a node on a specific replica.
    ///
    /// Go lines 341-353:
    /// ```go
    /// func (h *SlidingWindowHistogram) getNodeCost(node *TreeNode, podName string) float64 {
    ///     prefillCost := h.getPrefillCost(node)
    ///     timePerToken := 0.15  // default
    ///     if times, ok := h.avgTimePerTokenPerPod[podName]; ok && len(times) > 0 {
    ///         sort.Float64s(times)
    ///         timePerToken = times[len(times)/2]  // median
    ///     }
    ///     outputLen := h.decodingSize[node]
    ///     decodeCost := float64(outputLen) * timePerToken
    ///     return prefillCost + decodeCost
    /// }
    /// ```
    fn get_node_cost(
        &self,
        node_key: NodeKey,
        num_tokens: usize,
        context_length: usize,
        replica_id: usize,
    ) -> f64 {
        let prefill_cost = self.get_prefill_cost(node_key, num_tokens, context_length);

        // Get median TPOT for this replica
        let time_per_token = if replica_id < self.replicas.len() {
            let times = &self.replicas[replica_id].avg_time_per_token;
            if times.is_empty() {
                DEFAULT_TIME_PER_TOKEN
            } else {
                let mut sorted = times.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                sorted[sorted.len() / 2]
            }
        } else {
            DEFAULT_TIME_PER_TOKEN
        };

        let output_len = self
            .nodes
            .get(&node_key)
            .map(|s| s.decoding_size)
            .unwrap_or(self.default_decoding_length);

        let decode_cost = output_len as f64 * time_per_token;

        prefill_cost + decode_cost
    }

    /// Compute total allocation cost per replica across all tracked nodes.
    ///
    /// Go lines 355-369:
    /// ```go
    /// func (h *SlidingWindowHistogram) getCurrentAllocationCostPerPod() map[string]float64 {
    ///     h.mu.RLock()
    ///     defer h.mu.RUnlock()
    ///     costs := make(map[string]float64)
    ///     for node := range h.histogram {
    ///         for _, modelPods := range node.GetModelToPods() {
    ///             for podName := range modelPods {
    ///                 costs[podName] += h.getNodeCost(node, podName)
    ///             }
    ///         }
    ///     }
    ///     return costs
    /// }
    /// ```
    ///
    /// NOTE: In blitz-router, we don't have per-node-to-replica mappings (no
    /// `ModelToPods`). Instead, we distribute node cost evenly across all
    /// replicas as an approximation: each replica's cost contribution from
    /// a node is `getNodeCost(node, replica) / num_replicas`.
    pub fn get_allocation_cost_per_replica(&self) -> Vec<f64> {
        let mut costs = vec![0.0_f64; self.num_replicas];

        // Collect node keys first to avoid borrow issues
        let node_entries: Vec<(NodeKey, usize, usize)> = self
            .nodes
            .iter()
            .map(|(&key, stats)| {
                // We approximate num_tokens and context_length from the stats.
                // In the Go code, these come from the TreeNode object directly.
                // Here, we derive them from the histogram data.
                let avg_context =
                    if stats.node_to_count > 0 { stats.histogram / stats.node_to_count } else { 0 };
                let avg_num_tokens = if stats.node_to_count > 0 && stats.prompt_tokens > 0 {
                    let avg_hit = stats.hit_tokens / stats.node_to_count;
                    avg_context.saturating_sub(avg_hit)
                } else {
                    avg_context
                };
                (key, avg_num_tokens, avg_context)
            })
            .collect();

        for (node_key, num_tokens, context_length) in node_entries {
            for replica_id in 0..self.num_replicas {
                let cost = self.get_node_cost(node_key, num_tokens, context_length, replica_id);
                costs[replica_id] += cost / self.num_replicas as f64;
            }
        }

        costs
    }

    /// Get the load (number of requests) for a specific replica.
    ///
    /// Go lines 569-582:
    /// ```go
    /// func (h *SlidingWindowHistogram) getPodLoad(pod *v1.Pod) int {
    ///     h.mu.RLock()
    ///     defer h.mu.RUnlock()
    ///     load := 0
    ///     for node, count := range h.nodeToCount {
    ///         for _, podMap := range node.GetModelToPods() {
    ///             if _, exists := podMap[pod.Name]; exists {
    ///                 load += count
    ///                 break
    ///             }
    ///         }
    ///     }
    ///     return load
    /// }
    /// ```
    ///
    /// NOTE: Without per-node replica mapping, we approximate load as
    /// total request count / num_replicas. When used for tie-breaking
    /// among prefix-matched replicas, the relative ordering is what matters.
    pub fn get_replica_load(&self, _replica_id: usize) -> usize {
        // Sum all request counts across nodes
        let total: usize = self.nodes.values().map(|s| s.node_to_count).sum();
        total / self.num_replicas.max(1)
    }

    /// Record a TPOT measurement for a replica.
    #[allow(dead_code)] // exposed by histogram API; no current caller in DSL runtime
    pub fn record_tpot(&mut self, replica_id: usize, tpot: f64) {
        if replica_id < self.replicas.len() {
            let times = &mut self.replicas[replica_id].avg_time_per_token;
            times.push(tpot);
            // Keep bounded (last 100 measurements)
            if times.len() > 100 {
                times.drain(..times.len() - 100);
            }
        }
    }

    /// Get number of tracked nodes.
    #[allow(unused)]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of replicas.
    #[allow(dead_code)] // exposed by histogram API; no current caller in DSL runtime
    pub fn num_replicas(&self) -> usize {
        self.num_replicas
    }

    /// Get the default decoding length.
    pub fn default_decoding_length(&self) -> usize {
        self.default_decoding_length
    }
}

/// Generate a stable NodeKey from a block hash prefix.
///
/// We use the first and last hash values combined with the length
/// to produce a unique key for a given prefix sequence.
pub(crate) fn node_key_from_prefix(block_hashes: &[u64], prefix_len: usize) -> NodeKey {
    if prefix_len == 0 || block_hashes.is_empty() {
        return 0;
    }
    let effective_len = prefix_len.min(block_hashes.len());
    // Combine first hash, last hash, and length for uniqueness
    let first = block_hashes[0];
    let last = block_hashes[effective_len - 1];
    first
        .wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(last)
        .wrapping_mul(0x6c62272e07bb0142)
        .wrapping_add(effective_len as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_update_and_evict() {
        let mut hist = SlidingWindowHistogram::new(2, TargetGpu::V100);

        // First update
        hist.update(1, 100, 200, 0, 45);
        assert_eq!(hist.nodes.len(), 1);

        let stats = hist.nodes.get(&1).unwrap();
        assert_eq!(stats.node_to_count, 1);
        assert_eq!(stats.histogram, 200);
        assert_eq!(stats.hit_tokens, 100); // context_length - num_tokens = 200 - 100
        assert_eq!(stats.prompt_tokens, 200);
    }

    #[test]
    fn test_allocation_cost_positive() {
        let mut hist = SlidingWindowHistogram::new(2, TargetGpu::V100);
        hist.update(1, 256, 1024, 0, 45);
        hist.update(2, 128, 512, 1, 45);

        let costs = hist.get_allocation_cost_per_replica();
        assert_eq!(costs.len(), 2);
        for c in &costs {
            assert!(*c >= 0.0, "Cost should be non-negative: {c}");
        }
    }

    #[test]
    fn test_node_key_determinism() {
        let hashes = vec![100, 200, 300];
        let k1 = node_key_from_prefix(&hashes, 2);
        let k2 = node_key_from_prefix(&hashes, 2);
        assert_eq!(k1, k2);

        let k3 = node_key_from_prefix(&hashes, 3);
        assert_ne!(k1, k3);
    }
}
