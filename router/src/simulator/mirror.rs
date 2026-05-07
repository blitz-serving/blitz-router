// Incremental mirror — small, per-replica overlay on top of `ScheduleContext.
// block_hash` (the live engine prefix-cache state). Tracks ONLY the blocks
// belonging to in-flight requests (admitted by the router but not yet
// SSE-confirmed as cached/evicted by the engine).
//
// Lifecycle:
//   admission   → `insert_request(req_id, hashes, indices)`  (speculative)
//   SSE step    → blocks in `m.new_block_hashes_ids` are now confirmed in
//                 SCtx.block_hash; the mirror entry can be considered
//                 redundant. We simply leave it; the next eviction or finish
//                 prunes it. A periodic compaction is overkill for the small
//                 cardinalities expected here.
//   SSE finish  → `remove_request(req_id)` clears the mirror entry.
//   SSE evict   → `remove_blocks(block_ids)` removes by index.
//
// During rollout the outer DES queries `prefix_match(hashes)` to decide how
// many of the candidate's blocks are already cached — combining the live
// SCtx tree (via the caller) with the mirror.
//
// Reuses `PrefixBlockHash` from `kvcache.rs` (radix-tree or hash-table impl
// chosen by Cargo feature). Per-request bookkeeping is a separate `HashMap`.

use std::collections::HashMap;

use crate::engine_client::EngineStepOutput;
use crate::kvcache::{BlockHash, PrefixBlockHash};

pub struct IncrementalMirror {
    tree: PrefixBlockHash,
    /// request_id -> (block_hashes inserted speculatively, block_indices used)
    by_request: HashMap<u64, (Vec<u64>, Vec<u64>)>,
}

impl IncrementalMirror {
    pub fn new(num_blocks: usize) -> Self {
        Self { tree: PrefixBlockHash::new(num_blocks), by_request: HashMap::new() }
    }

    /// Speculatively insert a request's prefix blocks. Idempotent per
    /// `request_id`: a duplicate call replaces the previous entry.
    pub fn insert_request(&mut self, request_id: u64, hashes: &[u64], indices: Vec<u64>) {
        if hashes.is_empty() {
            return;
        }
        if let Some((_, prev_indices)) = self.by_request.remove(&request_id) {
            self.tree.remove(prev_indices);
        }
        self.tree.insert(hashes, indices.clone());
        self.by_request.insert(request_id, (hashes.to_vec(), indices));
    }

    /// Drop a request from the mirror, removing its blocks. Used on
    /// SSE-reported finish or abort. No-op for unknown ids.
    pub fn remove_request(&mut self, request_id: u64) {
        if let Some((_, indices)) = self.by_request.remove(&request_id) {
            self.tree.remove(indices);
        }
    }

    /// Remove specific block indices (e.g. on SSE-reported eviction). Cleans
    /// the per-request bookkeeping entries that referenced any of them.
    pub fn remove_blocks(&mut self, block_indices: &[u64]) {
        if block_indices.is_empty() {
            return;
        }
        let evicted: std::collections::HashSet<u64> = block_indices.iter().copied().collect();
        self.by_request.retain(|_req_id, (_hashes, indices)| {
            !indices.iter().any(|i| evicted.contains(i))
        });
        self.tree.remove(block_indices.to_vec());
    }

    /// Absorb one engine step's worth of state changes from the SSE
    /// payload. This is the single canonical entry-point for the SSE
    /// path (called from `PCtx::on_sse`); piecewise calls to the lower
    /// `remove_*` helpers remain available for unit-test setup but are
    /// not called by the hot path.
    ///
    /// Order of operations matters:
    ///   1. Evictions first — the engine has already freed these block
    ///      indices, so any per-request entry that referenced them is
    ///      now stale and must be pruned before checking finishes.
    ///   2. Finishes next — request-level removal cleans up indices
    ///      that survived eviction (a finished request typically frees
    ///      its blocks via the engine's standard release path, not via
    ///      `evicted_block_ids`).
    ///   3. Aborts last — same shape as finishes but driven by the
    ///      router rather than the engine. No-op for ids the mirror
    ///      doesn't know.
    /// Preempted requests are intentionally NOT removed: the engine
    /// keeps the prefix tree entries (the request will be rescheduled
    /// from the same prompt), so the mirror should also retain its
    /// view of them.
    pub fn apply_sse(&mut self, m: &EngineStepOutput) {
        if !m.evicted_block_ids.is_empty() {
            self.remove_blocks(&m.evicted_block_ids);
        }
        for o in &m.outputs {
            if o.is_finished {
                self.remove_request(o.request_id);
            }
        }
        for &rid in &m.aborted_requests {
            self.remove_request(rid);
        }
    }

    /// Returns the matched prefix length over the mirror only. The outer DES
    /// composes this with the live SCtx tree's match (taking the max, or
    /// summing, depending on whether they overlap — typically the mirror is a
    /// strict superset of yet-to-be-SSE-confirmed blocks, so a max is
    /// appropriate).
    pub fn prefix_match(&self, hashes: &[u64]) -> usize {
        self.tree.get(hashes)
    }

    /// Number of distinct in-flight requests currently mirrored.
    pub fn in_flight_count(&self) -> usize {
        self.by_request.len()
    }

    /// Mirror's underlying epoch counter; useful for cache-key invalidation.
    pub fn epoch(&self) -> u64 {
        self.tree.epoch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_match() {
        let mut m = IncrementalMirror::new(1024);
        m.insert_request(1, &[10, 20, 30], vec![0, 1, 2]);
        assert_eq!(m.prefix_match(&[10, 20, 30]), 3);
        assert_eq!(m.prefix_match(&[10, 20]), 2);
        assert_eq!(m.prefix_match(&[40]), 0);
        assert_eq!(m.in_flight_count(), 1);
    }

    #[test]
    fn remove_request_clears() {
        let mut m = IncrementalMirror::new(1024);
        m.insert_request(1, &[10, 20], vec![0, 1]);
        m.insert_request(2, &[10, 30], vec![2, 3]);
        assert_eq!(m.in_flight_count(), 2);
        m.remove_request(1);
        assert_eq!(m.in_flight_count(), 1);
        // The shared prefix (block hash 10) is still cached because request 2
        // also referenced it via index 2.
        assert_eq!(m.prefix_match(&[10]), 1);
    }

    #[test]
    fn remove_blocks_drops_request_bookkeeping() {
        let mut m = IncrementalMirror::new(1024);
        m.insert_request(1, &[10, 20, 30], vec![0, 1, 2]);
        // Evict only the leaf (index 2). The radix tree cascades parent→child
        // eviction, so removing a non-leaf would also drop descendants and
        // trip a debug assertion. In practice the engine reports the full
        // eviction set in one batch, so cascade semantics work out.
        m.remove_blocks(&[2]);
        // request 1's bookkeeping is gone (we removed one of its blocks).
        assert_eq!(m.in_flight_count(), 0);
    }

    #[test]
    fn duplicate_insert_replaces() {
        let mut m = IncrementalMirror::new(1024);
        m.insert_request(1, &[10, 20], vec![0, 1]);
        m.insert_request(1, &[40, 50], vec![2, 3]);
        assert_eq!(m.in_flight_count(), 1);
        assert_eq!(m.prefix_match(&[40, 50]), 2);
        assert_eq!(m.prefix_match(&[10]), 0);
    }

    use crate::engine_client::{EngineStepOutput, RequestStepOutput};
    use nohash_hasher::{BuildNoHashHasher, IntMap};

    fn step(
        outputs: Vec<RequestStepOutput>,
        evicted: Vec<u64>,
        aborted: Vec<u64>,
    ) -> EngineStepOutput {
        EngineStepOutput {
            prefill_tokens: 0,
            prefill_token_budget: 1024,
            latency: 1,
            outputs,
            new_block_hashes: Vec::new(),
            evicted_block_hashes: Vec::new(),
            evicted_block_ids: evicted,
            cur_used_block_ids: IntMap::with_hasher(BuildNoHashHasher::default()),
            new_block_hashes_ids: IntMap::with_hasher(BuildNoHashHasher::default()),
            op_exec_log: None,
            preempted_ids: Vec::new(),
            aborted_requests: aborted,
            step_id: 0,
        }
    }

    fn output(rid: u64, finished: bool, state: &str) -> RequestStepOutput {
        RequestStepOutput {
            request_id: rid,
            new_token_ids: vec![],
            state: state.to_string(),
            is_finished: finished,
            hit_token_cnt: 0,
            prev_computed_tokens: 0,
        }
    }

    #[test]
    fn apply_sse_drops_finished_request() {
        let mut m = IncrementalMirror::new(1024);
        m.insert_request(1, &[10, 20], vec![0, 1]);
        m.insert_request(2, &[30, 40], vec![2, 3]);
        let s = step(vec![output(1, true, "DECODE"), output(2, false, "DECODE")], vec![], vec![]);
        m.apply_sse(&s);
        assert_eq!(m.in_flight_count(), 1);
        assert_eq!(m.prefix_match(&[10]), 0);
        assert_eq!(m.prefix_match(&[30]), 1);
    }

    #[test]
    fn apply_sse_drops_aborted_request() {
        let mut m = IncrementalMirror::new(1024);
        m.insert_request(1, &[10, 20], vec![0, 1]);
        let s = step(vec![], vec![], vec![1]);
        m.apply_sse(&s);
        assert_eq!(m.in_flight_count(), 0);
    }

    #[test]
    fn apply_sse_evicts_blocks_before_finishing() {
        let mut m = IncrementalMirror::new(1024);
        m.insert_request(1, &[10, 20, 30], vec![0, 1, 2]);
        // The leaf block (index 2) is evicted in the same step as request
        // 1 finishes. Eviction-then-finish ordering inside apply_sse
        // ensures the request bookkeeping is gone after both events.
        let s = step(vec![output(1, true, "DECODE")], vec![2], vec![]);
        m.apply_sse(&s);
        assert_eq!(m.in_flight_count(), 0);
    }

    #[test]
    fn apply_sse_keeps_preempted_request() {
        let mut m = IncrementalMirror::new(1024);
        m.insert_request(1, &[10, 20], vec![0, 1]);
        let mut s = step(vec![], vec![], vec![]);
        s.preempted_ids = vec![1];
        m.apply_sse(&s);
        // Preempted requests retain their mirror entry (will be re-prefilled).
        assert_eq!(m.in_flight_count(), 1);
    }

    #[test]
    fn apply_sse_unknown_finish_is_noop() {
        let mut m = IncrementalMirror::new(1024);
        m.insert_request(1, &[10, 20], vec![0, 1]);
        let s = step(vec![output(99, true, "DECODE")], vec![], vec![]);
        m.apply_sse(&s);
        assert_eq!(m.in_flight_count(), 1);
    }
}
