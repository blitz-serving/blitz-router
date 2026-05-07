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
}
