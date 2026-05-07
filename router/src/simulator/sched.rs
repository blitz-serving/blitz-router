// L3 sub-state: per-request progress snapshot maintained by `PCtx`.
//
// Tracks the engine-internal lifecycle of every request the router has
// dispatched to this replica (admit time → finish/abort time). Two
// queues mirror the engine's own `waiting` / `running` separation:
//
//   * `waiting`  — admitted but not yet observed in any SSE step. The
//                  engine has the request in its scheduling queue but
//                  no chunk has been processed yet.
//   * `running`  — observed in at least one SSE step. Per-request
//                  `processed_tokens` advances as PREFILL chunks
//                  complete and as DECODE steps emit tokens.
//
// `query_sim` clones this whole snapshot, pushes the candidate to the
// cloned `waiting`, then rolls forward via the schedule loop. The
// snapshot is intentionally lightweight (no KV-cache manager, no
// block accounting); per-request `hashes` are NOT carried here — they
// live exclusively in `IncrementalMirror` which is the canonical
// prefix-cache view (see Group B(g) decision).
//
// Lock discipline (set by `PCtx`): `sched → mirror → ephemeral`.

use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct ReqProgress {
    pub request_id: u64,
    pub input_length: u32,
    /// `0..input_length` while in PREFILL; `input_length..` after the
    /// first DECODE step (each subsequent DECODE step += 1).
    pub processed_tokens: u32,
}

impl ReqProgress {
    pub fn new(request_id: u64, input_length: u32) -> Self {
        Self { request_id, input_length, processed_tokens: 0 }
    }

    /// True iff this request still owes prefill work (needs more
    /// chunks before it can DECODE).
    pub fn is_prefilling(&self) -> bool {
        self.processed_tokens < self.input_length
    }

    /// Tokens still to prefill. Saturates at zero.
    pub fn prefill_remaining(&self) -> u32 {
        self.input_length.saturating_sub(self.processed_tokens)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SchedSnapshot {
    pub waiting: VecDeque<ReqProgress>,
    pub running: HashMap<u64, ReqProgress>,
}

impl SchedSnapshot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit time: enqueue at the back of `waiting` (FCFS — engine
    /// processes earlier admissions first, locked in Group A F4).
    pub fn admit(&mut self, req: ReqProgress) {
        self.waiting.push_back(req);
    }

    /// Move a `waiting` entry to `running`. Returns the moved entry
    /// for callers that need to continue mutating it (e.g. to record
    /// the chunk just processed).
    pub fn promote_to_running(&mut self, request_id: u64) -> Option<&mut ReqProgress> {
        let pos = self.waiting.iter().position(|r| r.request_id == request_id)?;
        let req = self.waiting.remove(pos)?;
        self.running.insert(request_id, req);
        self.running.get_mut(&request_id)
    }

    /// Engine preempt path: yank from `running`, reset progress, push
    /// to the **front** of `waiting` (engine reschedules preempted
    /// requests with priority — locked in F2).
    pub fn preempt(&mut self, request_id: u64) {
        if let Some(mut req) = self.running.remove(&request_id) {
            req.processed_tokens = 0;
            self.waiting.push_front(req);
        }
    }

    /// Finish or abort: drop from both queues. Idempotent.
    pub fn drop_request(&mut self, request_id: u64) {
        self.running.remove(&request_id);
        self.waiting.retain(|r| r.request_id != request_id);
    }

    /// Iterate every in-flight request (waiting ∪ running). Used by
    /// `apply_sse`'s evict path to check whether an evicted hash's
    /// owner is still in flight (Group B Point 2 self-arbitration).
    pub fn iter_in_flight(&self) -> impl Iterator<Item = &ReqProgress> {
        self.waiting.iter().chain(self.running.values())
    }

    /// O(1) check used by the evict-path arbitration. Both queues are
    /// small (typically tens), so even the linear scan over `waiting`
    /// is microseconds.
    pub fn is_in_flight(&self, request_id: u64) -> bool {
        self.running.contains_key(&request_id)
            || self.waiting.iter().any(|r| r.request_id == request_id)
    }

    /// Total in-flight count (both queues). Observability only.
    pub fn in_flight_count(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    /// SSE absorption — keep the snapshot in sync with the engine's
    /// authoritative state after one forward step. Should be called
    /// from `PCtx::on_sse` *before* the L3 cross-check (so the
    /// cross-check sees the post-step state) and *after* the L2
    /// calibrate (which only reads the batch features).
    ///
    /// Order of operations:
    ///   1. Per-output state transitions:
    ///      - `PREFILL` — promote from waiting if first-seen; bump
    ///        `processed_tokens` by this step's chunk for the request.
    ///      - `DECODE`/`RUNNING` — `running[rid].processed_tokens += 1`.
    ///      - `is_finished` — drop the request entirely.
    ///   2. `preempted_ids` — yank from running, reset progress, push
    ///      to the front of waiting (matches engine's "preempted goes
    ///      first on next round" behaviour).
    ///   3. `aborted_requests` — top-level abort signal; drop.
    ///
    /// The chunk size for a PREFILL request in this step is approximated
    /// by even split of `m.prefill_tokens` across PREFILL outputs — same
    /// shape as `batch_from_step` in `simulator/mod.rs`. For typical
    /// chunked-prefill configs at most one PREFILL is active per step,
    /// so this is exact in practice.
    pub fn sync(&mut self, m: &crate::engine_client::EngineStepOutput) {
        // Pre-compute per-PREFILL chunk size (even split with remainder
        // assigned to the leading entries — matches batch_from_step).
        let prefill_count = m.outputs.iter().filter(|o| o.state == "PREFILL").count();
        let (per_chunk, remainder) = if prefill_count > 0 {
            (m.prefill_tokens / prefill_count, m.prefill_tokens % prefill_count)
        } else {
            (0, 0)
        };
        let mut prefill_seen = 0usize;

        for o in &m.outputs {
            let rid = o.request_id;
            match o.state.as_str() {
                "PREFILL" => {
                    let chunk = if prefill_seen < remainder { per_chunk + 1 } else { per_chunk };
                    prefill_seen += 1;
                    if !self.running.contains_key(&rid) {
                        self.promote_to_running(rid);
                    }
                    if let Some(req) = self.running.get_mut(&rid) {
                        // prev_computed_tokens is the KV size BEFORE this
                        // step ran, so the post-step processed count is
                        // exactly that plus this step's chunk.
                        req.processed_tokens = (o.prev_computed_tokens + chunk as u32)
                            .min(req.input_length.max(1));
                    }
                }
                "DECODE" | "RUNNING" => {
                    if !self.running.contains_key(&rid) {
                        // Late observation — request finished prefill in a
                        // step we missed. Promote and assume prefill done.
                        self.promote_to_running(rid);
                        if let Some(req) = self.running.get_mut(&rid) {
                            req.processed_tokens = req.input_length;
                        }
                    }
                    if let Some(req) = self.running.get_mut(&rid) {
                        req.processed_tokens = req.processed_tokens.saturating_add(1);
                    }
                }
                _ => {}
            }
            if o.is_finished {
                self.drop_request(rid);
            }
        }
        for &rid in &m.preempted_ids {
            self.preempt(rid);
        }
        for &rid in &m.aborted_requests {
            self.drop_request(rid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rp(rid: u64, input: u32) -> ReqProgress {
        ReqProgress::new(rid, input)
    }

    #[test]
    fn admit_then_promote_then_drop() {
        let mut s = SchedSnapshot::new();
        s.admit(rp(1, 100));
        s.admit(rp(2, 200));
        assert_eq!(s.in_flight_count(), 2);
        assert!(s.is_in_flight(1));

        s.promote_to_running(1);
        assert!(s.running.contains_key(&1));
        assert!(s.waiting.iter().all(|r| r.request_id != 1));
        assert_eq!(s.in_flight_count(), 2);

        s.drop_request(1);
        assert!(!s.is_in_flight(1));
        assert_eq!(s.in_flight_count(), 1);
    }

    #[test]
    fn preempt_resets_progress_and_pushes_front() {
        let mut s = SchedSnapshot::new();
        s.admit(rp(1, 100));
        s.admit(rp(2, 200));
        // Promote 1 and advance its prefill.
        s.promote_to_running(1).unwrap().processed_tokens = 64;
        // Promote 2 to running too.
        s.promote_to_running(2).unwrap().processed_tokens = 128;

        s.preempt(1);
        // 1 is back in waiting, at the front (priority over 2's eventual re-admit).
        assert_eq!(s.waiting.front().unwrap().request_id, 1);
        assert_eq!(s.waiting.front().unwrap().processed_tokens, 0);
        assert!(!s.running.contains_key(&1));
        assert!(s.running.contains_key(&2));
    }

    #[test]
    fn iter_in_flight_covers_both_queues() {
        let mut s = SchedSnapshot::new();
        s.admit(rp(1, 100));
        s.admit(rp(2, 200));
        s.promote_to_running(1);
        let rids: Vec<u64> = s.iter_in_flight().map(|r| r.request_id).collect();
        assert!(rids.contains(&1));
        assert!(rids.contains(&2));
        assert_eq!(rids.len(), 2);
    }

    #[test]
    fn req_progress_helpers() {
        let mut r = rp(1, 100);
        assert!(r.is_prefilling());
        assert_eq!(r.prefill_remaining(), 100);
        r.processed_tokens = 80;
        assert!(r.is_prefilling());
        assert_eq!(r.prefill_remaining(), 20);
        r.processed_tokens = 100;
        assert!(!r.is_prefilling());
        assert_eq!(r.prefill_remaining(), 0);
        r.processed_tokens = 105; // 5 decode tokens emitted
        assert!(!r.is_prefilling());
        assert_eq!(r.prefill_remaining(), 0); // saturating
    }

    #[test]
    fn drop_idempotent_on_unknown() {
        let mut s = SchedSnapshot::new();
        s.drop_request(99); // no panic
        s.admit(rp(1, 100));
        s.drop_request(1);
        s.drop_request(1); // second drop also no panic
        assert_eq!(s.in_flight_count(), 0);
    }

    use crate::engine_client::{EngineStepOutput, RequestStepOutput};
    use nohash_hasher::{BuildNoHashHasher, IntMap};

    fn step(
        prefill_tokens: usize,
        outputs: Vec<RequestStepOutput>,
        preempted: Vec<u64>,
        aborted: Vec<u64>,
    ) -> EngineStepOutput {
        EngineStepOutput {
            prefill_tokens,
            prefill_token_budget: 1024,
            latency: 1,
            outputs,
            new_block_hashes: Vec::new(),
            evicted_block_hashes: Vec::new(),
            evicted_block_ids: Vec::new(),
            cur_used_block_ids: IntMap::with_hasher(BuildNoHashHasher::default()),
            new_block_hashes_ids: IntMap::with_hasher(BuildNoHashHasher::default()),
            op_exec_log: None,
            preempted_ids: preempted,
            aborted_requests: aborted,
            step_id: 0,
        }
    }

    fn out(rid: u64, state: &str, prev_computed: u32, finished: bool) -> RequestStepOutput {
        RequestStepOutput {
            request_id: rid,
            new_token_ids: vec![],
            state: state.to_string(),
            is_finished: finished,
            hit_token_cnt: 0,
            prev_computed_tokens: prev_computed,
        }
    }

    #[test]
    fn sync_promotes_first_prefill_seen() {
        let mut s = SchedSnapshot::new();
        s.admit(rp(1, 100));
        // Single PREFILL request, full chunk (100 tokens this step).
        let m = step(100, vec![out(1, "PREFILL", 0, false)], vec![], vec![]);
        s.sync(&m);
        assert!(s.running.contains_key(&1));
        assert!(s.waiting.iter().all(|r| r.request_id != 1));
        // Capped at input_length, since we sent the full prompt.
        assert_eq!(s.running[&1].processed_tokens, 100);
    }

    #[test]
    fn sync_chunked_prefill_advances_processed() {
        let mut s = SchedSnapshot::new();
        s.admit(rp(1, 200));
        // First chunk: 100 of 200 tokens.
        s.sync(&step(100, vec![out(1, "PREFILL", 0, false)], vec![], vec![]));
        assert_eq!(s.running[&1].processed_tokens, 100);
        // Next chunk: another 100 (cumulative 200, prefill done).
        s.sync(&step(100, vec![out(1, "PREFILL", 100, false)], vec![], vec![]));
        assert_eq!(s.running[&1].processed_tokens, 200);
    }

    #[test]
    fn sync_decode_increments_processed() {
        let mut s = SchedSnapshot::new();
        s.admit(rp(1, 100));
        s.promote_to_running(1).unwrap().processed_tokens = 100;
        s.sync(&step(0, vec![out(1, "DECODE", 100, false)], vec![], vec![]));
        assert_eq!(s.running[&1].processed_tokens, 101);
        s.sync(&step(0, vec![out(1, "DECODE", 100, false)], vec![], vec![]));
        assert_eq!(s.running[&1].processed_tokens, 102);
    }

    #[test]
    fn sync_finished_drops_request() {
        let mut s = SchedSnapshot::new();
        s.admit(rp(1, 100));
        s.promote_to_running(1).unwrap().processed_tokens = 100;
        s.sync(&step(0, vec![out(1, "DECODE", 100, true)], vec![], vec![]));
        assert!(!s.is_in_flight(1));
    }

    #[test]
    fn sync_preempt_pushes_front_with_reset() {
        let mut s = SchedSnapshot::new();
        s.admit(rp(1, 100));
        s.admit(rp(2, 200));
        s.promote_to_running(1).unwrap().processed_tokens = 50;
        s.promote_to_running(2).unwrap().processed_tokens = 100;
        s.sync(&step(0, vec![], vec![1], vec![]));
        // 1 is back in waiting at front, progress reset.
        assert_eq!(s.waiting.front().unwrap().request_id, 1);
        assert_eq!(s.waiting.front().unwrap().processed_tokens, 0);
        assert!(!s.running.contains_key(&1));
        assert!(s.running.contains_key(&2));
    }

    #[test]
    fn sync_abort_drops() {
        let mut s = SchedSnapshot::new();
        s.admit(rp(1, 100));
        s.sync(&step(0, vec![], vec![], vec![1]));
        assert!(!s.is_in_flight(1));
    }

    #[test]
    fn sync_chunked_split_two_prefills_distributes_chunk() {
        let mut s = SchedSnapshot::new();
        s.admit(rp(1, 200));
        s.admit(rp(2, 300));
        // Two PREFILL outputs, total 200 prefill tokens this step.
        // Even split: 100 each.
        s.sync(&step(
            200,
            vec![out(1, "PREFILL", 0, false), out(2, "PREFILL", 0, false)],
            vec![],
            vec![],
        ));
        assert_eq!(s.running[&1].processed_tokens, 100);
        assert_eq!(s.running[&2].processed_tokens, 100);
    }
}
