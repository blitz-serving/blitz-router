// L1 incremental mirror — per-replica overlay of the in-flight prefix
// cache. Sits next to the engine's own cache (mirrored router-side as
// `ScheduleContext.block_hash`) and contributes the **speculative** part:
// hashes the router has dispatched but the engine may not have surfaced
// yet via SSE. The composite cache-hit estimate the simulator uses is
// `max(SCtx.block_hash.get(h), Mirror.prefix_match(h))` — locked as A2.
//
// Lifecycle (Group A + B locked):
//
//   admit time          → `insert_request(rid, hashes)`
//                         (hashes from `entry.block_hash_state`).
//   SSE per-step        → `apply_sse(m, sched)`:
//                            * for each finished output  → remove_request(rid)
//                            * for each aborted_request  → remove_request(rid)
//                            * for each preempted_id     → remove_request(rid)
//                            * for each evicted_block_hash:
//                                 - if any owner still in_flight → flag (rare,
//                                   per Group B Point 1 should not happen with
//                                   vLLM); v1 swallows, v1.5 will redo
//                                 - else → drop subtree (orphan cleanup)
//                         The evict path is **self-arbitrating** thanks to
//                         V=ReqId (Group B Point 2): the mirror itself can
//                         decide whether to drop or preserve without an
//                         out-of-band signal.
//
// What the mirror does NOT do (deliberately):
//
//   * Track block indices. The engine's index assignment happens
//     post-admit; mirror exists precisely so the router doesn't need
//     to wait for that. Eviction is processed via `evicted_block_hashes`
//     not `evicted_block_ids` (both are present on the SSE payload).
//   * Maintain its own preempt/abort state machine — that lives in
//     `SchedSnapshot`. Mirror only reacts.
//   * Try to stay perfectly in sync under exotic eviction patterns —
//     A2's redo path absorbs residual drift.

use std::collections::HashMap;

use crate::engine::EngineStepOutput;

use radixtree::RadixTreeReqIdHash;
use super::sched::SchedSnapshot;

pub struct IncrementalMirror {
    tree: RadixTreeReqIdHash,
    /// rid → cached copy of the hash sequence inserted at admit time.
    /// Needed because `remove_by_hashes` works against the original
    /// sequence, and we don't want to rederive it from `BlockHashState`
    /// at finish time.
    by_request: HashMap<u64, Vec<u64>>,
}

impl IncrementalMirror {
    pub fn new() -> Self {
        Self { tree: RadixTreeReqIdHash::new(), by_request: HashMap::new() }
    }

    /// Register that `rid` claims this prefix sequence. Idempotent
    /// per rid: a duplicate call replaces the previous sequence
    /// (guards against re-admission edge cases).
    pub fn insert_request(&mut self, rid: u64, hashes: &[u64]) {
        if hashes.is_empty() {
            return;
        }
        if let Some(prev) = self.by_request.remove(&rid) {
            self.tree.remove_by_hashes(&prev, rid);
        }
        self.tree.insert_hashes(hashes, rid);
        self.by_request.insert(rid, hashes.to_vec());
    }

    /// Drop `rid`'s claim entirely. Used by `apply_sse` for finish /
    /// abort / preempt branches. No-op for unknown ids.
    pub fn remove_request(&mut self, rid: u64) {
        if let Some(prev) = self.by_request.remove(&rid) {
            self.tree.remove_by_hashes(&prev, rid);
        }
    }

    /// SSE absorption — single canonical entry from `PCtx::on_sse`.
    /// Order:
    ///   1. evict (so subsequent finish/abort sees a clean tree)
    ///   2. finish (per-output is_finished)
    ///   3. abort (top-level aborted_requests)
    ///   4. preempt (top-level preempted_ids — full-request drop;
    ///      sched will push the request back to its waiting queue
    ///      in parallel)
    pub fn apply_sse(&mut self, m: &EngineStepOutput, sched: &SchedSnapshot) {
        for evicted_hash in &m.evicted_block_hashes {
            // BackendBlockHash is `[u64; N]` (sha256) or `u64`
            // (default-hash). The simulator hash key is the same
            // 64-bit hash that BlockHashState uses; under the
            // default-hash-algo feature this is the first u64 of the
            // BackendBlockHash array.
            let key = backend_hash_to_key(evicted_hash);
            let still_in_flight = self
                .tree
                .owners_of(key)
                .any(|rid| sched.is_in_flight(rid));
            if !still_in_flight {
                self.tree.evict_orphan_hash(key);
            }
            // else: rare per Group B Point 1; v1 leaves the entry as-is
            // (the cross-check / redo path absorbs any resulting drift).
        }
        for o in &m.outputs {
            if o.is_finished {
                self.remove_request(o.request_id);
            }
        }
        for &rid in &m.aborted_requests {
            self.remove_request(rid);
        }
        for &rid in &m.preempted_ids {
            self.remove_request(rid);
        }
    }

    /// Longest matched prefix length over the mirror only. The DES
    /// composes this with `SCtx.block_hash.get(hashes)` via A2's
    /// `max` rule.
    pub fn prefix_match(&self, hashes: &[u64]) -> usize {
        self.tree.get(hashes)
    }

    /// Number of distinct in-flight rids currently mirrored.
    pub fn in_flight_count(&self) -> usize {
        self.by_request.len()
    }

    /// Mirror epoch — bumped on every tree mutation. Useful as a
    /// cache-invalidation key for downstream memoisation.
    pub fn epoch(&self) -> u64 {
        self.tree.epoch()
    }
}

impl Default for IncrementalMirror {
    fn default() -> Self {
        Self::new()
    }
}

/// Reduce a backend block hash (sha256 = `[u64;4]`, default = `[u64;1]`)
/// down to the 64-bit key the mirror's trie operates on. `BlockHashState`
/// uses the same first-u64 reduction internally so the keys agree
/// across the mirror, the SCtx tree, and per-request `block_hashes`.
fn backend_hash_to_key(h: &crate::scheduler::kvcache::BackendBlockHash) -> u64 {
    h[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineStepOutput, RequestStepOutput};
    use nohash_hasher::{BuildNoHashHasher, IntMap};

    fn step(
        outputs: Vec<RequestStepOutput>,
        evicted_hashes: Vec<crate::scheduler::kvcache::BackendBlockHash>,
        aborted: Vec<u64>,
        preempted: Vec<u64>,
    ) -> EngineStepOutput {
        EngineStepOutput {
            prefill_tokens: 0,
            prefill_token_budget: 1024,
            latency: 1,
            outputs,
            new_block_hashes: Vec::new(),
            evicted_block_hashes: evicted_hashes,
            evicted_block_ids: Vec::new(), // unused in v1
            cur_used_block_ids: IntMap::with_hasher(BuildNoHashHasher::default()),
            new_block_hashes_ids: IntMap::with_hasher(BuildNoHashHasher::default()),
            op_exec_log: None,
            preempted_ids: preempted,
            aborted_requests: aborted,
            step_id: 0,
        }
    }

    fn out(rid: u64, finished: bool, state: &str) -> RequestStepOutput {
        RequestStepOutput {
            request_id: rid,
            new_token_ids: vec![],
            state: state.to_string(),
            is_finished: finished,
            hit_token_cnt: 0,
            prev_computed_tokens: 0,
        }
    }

    #[cfg(not(feature = "sha256-hash-algo"))]
    fn bh(h: u64) -> crate::scheduler::kvcache::BackendBlockHash {
        [h]
    }
    #[cfg(feature = "sha256-hash-algo")]
    fn bh(h: u64) -> crate::scheduler::kvcache::BackendBlockHash {
        [h, 0, 0, 0]
    }

    fn sched_with(rids: &[u64]) -> SchedSnapshot {
        let mut s = SchedSnapshot::new();
        for &rid in rids {
            s.admit(super::super::sched::ReqProgress::new(rid, 100));
        }
        s
    }

    #[test]
    fn insert_and_prefix_match() {
        let mut m = IncrementalMirror::new();
        m.insert_request(1, &[10, 20, 30]);
        assert_eq!(m.prefix_match(&[10, 20, 30]), 3);
        assert_eq!(m.prefix_match(&[10, 20]), 2);
        assert_eq!(m.prefix_match(&[40]), 0);
        assert_eq!(m.in_flight_count(), 1);
    }

    #[test]
    fn finish_removes_request() {
        let mut m = IncrementalMirror::new();
        m.insert_request(1, &[10, 20]);
        m.insert_request(2, &[10, 30]);
        let s = sched_with(&[2]); // 1 already finished, sched dropped it
        let step = step(vec![out(1, true, "DECODE")], vec![], vec![], vec![]);
        m.apply_sse(&step, &s);
        assert_eq!(m.in_flight_count(), 1);
        // 1's branch (10→20) is gone, but [10] itself is still cached by
        // rid 2's prefix → longest match for [10,20] is the leading [10].
        assert_eq!(m.prefix_match(&[10, 20]), 1);
        assert_eq!(m.prefix_match(&[10, 30]), 2); // 2 still there
    }

    #[test]
    fn abort_removes_request() {
        let mut m = IncrementalMirror::new();
        m.insert_request(1, &[10, 20]);
        let s = sched_with(&[]); // sched dropped 1 in parallel
        let step = step(vec![], vec![], vec![1], vec![]);
        m.apply_sse(&step, &s);
        assert_eq!(m.in_flight_count(), 0);
    }

    #[test]
    fn preempt_removes_request() {
        let mut m = IncrementalMirror::new();
        m.insert_request(1, &[10, 20]);
        // sched will push 1 back to waiting (with reset progress) in
        // parallel; from mirror's POV the request needs re-admission
        // so its hashes are dropped.
        let s = sched_with(&[1]);
        let step = step(vec![], vec![], vec![], vec![1]);
        m.apply_sse(&step, &s);
        assert_eq!(m.in_flight_count(), 0);
    }

    #[test]
    fn evict_orphan_hash_drops_when_owner_not_in_flight() {
        let mut m = IncrementalMirror::new();
        m.insert_request(1, &[10, 20]);
        // sched no longer has rid 1 (e.g. an old finish we never saw);
        // mirror still does. Engine evicts hash 10. Self-arbitration
        // should drop it.
        let s = sched_with(&[]);
        let step = step(vec![], vec![bh(10)], vec![], vec![]);
        m.apply_sse(&step, &s);
        assert_eq!(m.prefix_match(&[10]), 0);
    }

    #[test]
    fn evict_keeps_in_flight_owner_intact() {
        let mut m = IncrementalMirror::new();
        m.insert_request(1, &[10, 20]);
        let s = sched_with(&[1]); // 1 still in flight
        // Rare per Group B Point 1, but if the SSE claims to evict
        // hash 10 of an in-flight request, v1 swallows.
        let step = step(vec![], vec![bh(10)], vec![], vec![]);
        m.apply_sse(&step, &s);
        // Mirror untouched.
        assert_eq!(m.prefix_match(&[10, 20]), 2);
    }

    #[test]
    fn evict_unknown_hash_is_noop() {
        let mut m = IncrementalMirror::new();
        m.insert_request(1, &[10, 20]);
        let s = sched_with(&[1]);
        let step = step(vec![], vec![bh(99)], vec![], vec![]);
        m.apply_sse(&step, &s);
        assert_eq!(m.prefix_match(&[10, 20]), 2);
    }

    #[test]
    fn duplicate_insert_replaces() {
        let mut m = IncrementalMirror::new();
        m.insert_request(1, &[10, 20]);
        m.insert_request(1, &[40, 50]);
        assert_eq!(m.prefix_match(&[10]), 0);
        assert_eq!(m.prefix_match(&[40, 50]), 2);
        assert_eq!(m.in_flight_count(), 1);
    }
}
