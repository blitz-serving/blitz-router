// PCtx — per-replica predictor context owned by the colocation controller.
//
// Three-layer state model (Phase 3 lock):
//
//   * L1 — `IncrementalMirror`. Per-replica overlay of in-flight prefix
//     hashes (V=ReqId trie). Updated by `on_admit` (insert) and
//     `on_sse` (apply_sse: finish/abort/preempt + self-arbitrating
//     evict). See `mirror.rs` and `req_id_tree.rs`.
//   * L2 — `LinregCorrected` regressor (offline-trained inner model
//     wrapped by an online linear correction). Updated by `on_sse`'s
//     piggyback `observe_step` call. No "drop" — calibration is
//     monotonically incremental over the process lifetime.
//   * L3 — `EphemeralRollout`. A single-slot Option<…> holding the
//     most recent `query()`'s `RolloutBuffer` plus the candidate id
//     and SSE step id at which it was anchored. Drop / rebuild
//     semantics live entirely on this field.
//
// Plus an auxiliary state used by the L1/L2 paths and consumed by
// `query`'s schedule loop:
//
//   * **`SchedSnapshot`** — per-request progress (waiting / running)
//     for every request the router has dispatched to this replica.
//     Mirrors the engine's own scheduling lifecycle without trying
//     to be a KV-cache manager (mirror handles cache state). Cloned
//     and rolled forward inside `query_sim` (Phase 3 algorithm).
//
// Lock discipline (NEVER take in a different order):
//
//   `sched → mirror → ephemeral`
//
// Every mutating method documents which locks it touches; tests in
// the same module rely on this order being respected.
//
// Public API — three triggers, locked in Phase 2 / refined in Phase 3:
//
//   * `query(candidate_id) -> RolloutGist` — speculative rollout for
//     a candidate. Phase-3 work fills in the schedule loop; the
//     current Phase-2 placeholder still lives here pending T8.
//   * `on_admit(entry: &Entry)` — admission. Inserts into both
//     `sched.waiting` and `mirror`; runs the L3 promote-or-drop
//     branch. Signature widened in Phase 3 (T5) so the simulator can
//     pull `input_length` and `block_hash_state.get_hashes()` from
//     the entry.
//   * `on_sse(batch, m)` — SSE event from the engine: drives L2
//     calibration, syncs `sched`, runs `mirror.apply_sse`, enforces
//     the L3 invariant. Phase-3 (T6/T7) adds the SchedSnapshot sync
//     and the F3 cross-check.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::engine_client::EngineStepOutput;

use super::batch::BatchForPredictor;
use super::mirror::IncrementalMirror;
use super::predictor::TrainedPredictor;
use super::rollout::{RolloutBuffer, RolloutGist};
use super::sched::SchedSnapshot;

/// L3 — single-slot ephemeral rollout. Identified by the candidate id
/// at the time of `query`, anchored to the SSE step id observed when
/// the rollout was constructed.
#[derive(Debug)]
struct EphemeralRollout {
    /// Candidate id this rollout was generated for. `None` after a
    /// successful `on_admit` promotion (the buffer is then a baseline
    /// describing the actual scheduled trajectory).
    candidate_id: Option<u64>,
    buffer: RolloutBuffer,
    sse_anchor_step_id: u64,
}

pub struct PCtx {
    /// L2.
    regressor: Mutex<Box<dyn TrainedPredictor>>,
    /// L1.
    mirror: Mutex<IncrementalMirror>,
    /// L3.
    ephemeral: Mutex<Option<EphemeralRollout>>,
    /// Per-request progress. Cloned by `query` for rollout simulation.
    /// Distinct from L1 mirror in that it carries lifecycle progress
    /// (PREFILL / DECODE token counts) but **not** prefix hashes.
    sched: Mutex<SchedSnapshot>,
    /// Monotonic engine step id observed by the most recent `on_sse`
    /// call. Used as the anchor for newly created `EphemeralRollout`s
    /// and as the freshness check in `on_sse`'s invariant maintenance.
    last_sse_step_id: AtomicU64,
}

impl PCtx {
    pub fn new(regressor: Box<dyn TrainedPredictor>) -> Self {
        Self {
            regressor: Mutex::new(regressor),
            mirror: Mutex::new(IncrementalMirror::new()),
            ephemeral: Mutex::new(None),
            sched: Mutex::new(SchedSnapshot::new()),
            last_sse_step_id: AtomicU64::new(0),
        }
    }

    /// L2 predict + calibrate in one critical section. Returns
    /// `(predicted_ms, actual_ms)` for metric emission.
    pub fn observe_step(&self, batch: &BatchForPredictor, actual_ms: f32) -> (f32, f32) {
        let mut g = self.regressor.lock().expect("PCtx regressor poisoned");
        let predicted = g.predict(batch);
        g.calibrate(batch, actual_ms);
        (predicted, actual_ms)
    }

    /// L2 predict-only (no calibration).
    pub fn predict(&self, batch: &BatchForPredictor) -> f32 {
        self.regressor.lock().expect("PCtx regressor poisoned").predict(batch)
    }

    /// L1 INSERT (low-level, public for unit tests). Production call
    /// site is `on_admit`.
    pub fn insert_in_flight(&self, request_id: u64, hashes: &[u64]) {
        self.mirror
            .lock()
            .expect("PCtx mirror poisoned")
            .insert_request(request_id, hashes);
    }

    /// L1 REMOVE (low-level). Used by `on_sse` via `apply_sse`; kept
    /// public for direct use by policy code that aborts a request
    /// before it reaches the SSE path.
    pub fn remove_in_flight(&self, request_id: u64) {
        self.mirror.lock().expect("PCtx mirror poisoned").remove_request(request_id);
    }

    /// In-flight request count (mirror's view). Observability only.
    pub fn in_flight_count(&self) -> usize {
        self.mirror.lock().expect("PCtx mirror poisoned").in_flight_count()
    }

    /// In-flight request count (sched's view — both queues).
    /// Observability only; in steady state should equal `in_flight_count`.
    pub fn sched_in_flight_count(&self) -> usize {
        self.sched.lock().expect("PCtx sched poisoned").in_flight_count()
    }

    // ---------- L3 trigger surface ----------

    /// Trigger A — speculative rollout for `candidate_id`. Caches
    /// the resulting buffer in L3 anchored to the latest SSE step id
    /// and returns the projected gist for the policy scheduler.
    ///
    /// Phase-3 status: the schedule loop itself (T8) is not yet
    /// implemented. This method returns the default gist (all-None)
    /// and stores an empty buffer at the L3 slot so that the
    /// lifecycle (`on_admit` promotion, `on_sse` invariant) can be
    /// exercised end-to-end ahead of T8.
    pub fn query(&self, candidate_id: u64) -> RolloutGist {
        let anchor = self.last_sse_step_id.load(Ordering::Acquire);
        let buffer = RolloutBuffer { candidate_id: Some(candidate_id), ..Default::default() };
        let gist = buffer.gist();
        *self.ephemeral.lock().expect("PCtx ephemeral poisoned") = Some(EphemeralRollout {
            candidate_id: Some(candidate_id),
            buffer,
            sse_anchor_step_id: anchor,
        });
        gist
    }

    /// Trigger B — admission. Called by `simulator::on_admit` after
    /// `policy_runner` commits the request. Takes the three fields
    /// needed for the three layers; intentionally does NOT depend on
    /// `Entry` so unit tests can exercise it without constructing a
    /// full pipeline value.
    ///
    /// L1: insert the request's prefix hashes into the mirror.
    /// Sched: enqueue a fresh `ReqProgress` at the back of `waiting`.
    /// L3: promote-or-drop branch (matching candidate → baseline,
    ///     mismatch → drop).
    pub fn on_admit(&self, request_id: u64, input_length: u32, hashes: &[u64]) {
        // Lock order: sched → mirror → ephemeral.
        {
            let mut sched = self.sched.lock().expect("PCtx sched poisoned");
            sched.admit(super::sched::ReqProgress::new(request_id, input_length));
        }
        {
            let mut mirror = self.mirror.lock().expect("PCtx mirror poisoned");
            mirror.insert_request(request_id, hashes);
        }
        {
            let mut slot = self.ephemeral.lock().expect("PCtx ephemeral poisoned");
            match slot.as_mut() {
                Some(e) if e.candidate_id == Some(request_id) => {
                    e.candidate_id = None;
                    e.buffer.candidate_id = None;
                }
                _ => {
                    *slot = None;
                }
            }
        }
    }

    /// Trigger C — SSE event from the engine for this replica.
    ///
    /// 1. L2: piggyback predict + calibrate (returned for metrics).
    /// 2. L1: absorb the step's evictions / finishes / aborts /
    ///    preempts via `IncrementalMirror::apply_sse`.
    /// 3. L3 invariant: when the engine still has prefill work after
    ///    this step, the ephemeral buffer must be non-empty; if it
    ///    isn't, log (Phase-3 T7 will add the F3 cross-check that
    ///    upgrades anchor-based stale detection to
    ///    composition-based).
    ///
    /// Phase-3 will additionally call `SchedSnapshot::sync(m)` here
    /// (T6) so the schedule loop in `query` always has fresh state.
    pub fn on_sse(&self, batch: &BatchForPredictor, m: &EngineStepOutput) -> (f32, f32) {
        // L2 first.
        let (predicted, actual) = self.observe_step(batch, m.latency as f32);

        // L1 absorption — needs sched lock for the evict path's
        // self-arbitration (`is_in_flight`). Lock order: sched → mirror.
        let sched_guard = self.sched.lock().expect("PCtx sched poisoned");
        self.mirror.lock().expect("PCtx mirror poisoned").apply_sse(m, &sched_guard);
        drop(sched_guard);

        // Bump anchor before L3 maintenance.
        self.last_sse_step_id.store(m.step_id, Ordering::Release);

        // L3 invariant maintenance.
        let engine_has_prefill = step_has_prefill(m);
        let mut slot = self.ephemeral.lock().expect("PCtx ephemeral poisoned");
        match (slot.as_mut(), engine_has_prefill) {
            (Some(e), true) => {
                if e.sse_anchor_step_id < m.step_id.saturating_sub(1) {
                    *slot = None;
                }
            }
            (None, true) => {
                tracing::trace!(
                    target: "simulator",
                    step = m.step_id,
                    "L3 invariant: engine has prefill but ephemeral is empty (rebuild deferred to T8)"
                );
            }
            (Some(_), false) => {
                *slot = None;
            }
            (None, false) => {}
        }

        (predicted, actual)
    }

    /// Test-only inspection: `Some(candidate_id)` if L3 currently
    /// holds a candidate-bound rollout, `None` if empty or baseline.
    #[cfg(test)]
    fn ephemeral_candidate(&self) -> Option<u64> {
        self.ephemeral
            .lock()
            .expect("PCtx ephemeral poisoned")
            .as_ref()
            .and_then(|e| e.candidate_id)
    }

    #[cfg(test)]
    fn ephemeral_present(&self) -> bool {
        self.ephemeral.lock().expect("PCtx ephemeral poisoned").is_some()
    }
}

/// Heuristic for "engine has prefill work in progress". A step is
/// counted as having prefill work if it processed any prefill tokens
/// (`prefill_tokens > 0`) or if any output is in the PREFILL state.
fn step_has_prefill(m: &EngineStepOutput) -> bool {
    if m.prefill_tokens > 0 {
        return true;
    }
    m.outputs.iter().any(|o| o.state == "PREFILL")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::engine_client::RequestStepOutput;
    use crate::simulator::config::SimulatorConfig;
    use crate::simulator::predictor::{LinregCorrected, Predictor};
    use nohash_hasher::{BuildNoHashHasher, IntMap};

    struct ConstPredictor(f32);
    impl Predictor for ConstPredictor {
        fn predict(&self, _b: &BatchForPredictor) -> f32 {
            self.0
        }
    }

    fn pctx() -> PCtx {
        let inner = Arc::new(ConstPredictor(1.0));
        let trained = Box::new(LinregCorrected::new(inner, &SimulatorConfig::default()));
        PCtx::new(trained)
    }

    fn step(
        step_id: u64,
        prefill_tokens: usize,
        outputs: Vec<RequestStepOutput>,
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
            preempted_ids: Vec::new(),
            aborted_requests: Vec::new(),
            step_id,
        }
    }

    fn out(rid: u64, state: &str, finished: bool) -> RequestStepOutput {
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
    fn observe_step_calibrates() {
        let mut cfg = SimulatorConfig::default();
        cfg.linreg_warmup = 0;
        cfg.learning_rate = 0.05;
        cfg.linreg_outlier_threshold_ms = 100.0;

        let inner = Arc::new(ConstPredictor(2.0));
        let trained = Box::new(LinregCorrected::new(inner, &cfg));
        let pctx = PCtx::new(trained);

        let batch = BatchForPredictor::default();
        let mut last_pred = 0.0f32;
        for _ in 0..1500 {
            let (pred, actual) = pctx.observe_step(&batch, 4.0);
            assert!((actual - 4.0).abs() < 1e-6);
            last_pred = pred;
        }
        assert!(
            (last_pred - 4.0).abs() < 0.05,
            "linreg did not converge inside PCtx: last_pred={}",
            last_pred
        );
    }

    #[test]
    fn mirror_lifecycle() {
        let pctx = pctx();
        pctx.insert_in_flight(1, &[10, 20]);
        pctx.insert_in_flight(2, &[10, 30]);
        assert_eq!(pctx.in_flight_count(), 2);
        pctx.remove_in_flight(1);
        assert_eq!(pctx.in_flight_count(), 1);
    }

    #[test]
    fn query_records_candidate_bound_ephemeral() {
        let pctx = pctx();
        let _ = pctx.query(7);
        assert_eq!(pctx.ephemeral_candidate(), Some(7));
    }

    #[test]
    fn on_admit_promotes_matching_candidate_to_baseline() {
        let pctx = pctx();
        let _ = pctx.query(7);
        pctx.on_admit(7, 100, &[10, 20]);
        assert!(pctx.ephemeral_present());
        assert_eq!(pctx.ephemeral_candidate(), None);
        // L1 + sched side-effects.
        assert_eq!(pctx.in_flight_count(), 1);
        assert_eq!(pctx.sched_in_flight_count(), 1);
    }

    #[test]
    fn on_admit_drops_mismatched_candidate() {
        let pctx = pctx();
        let _ = pctx.query(7);
        pctx.on_admit(99, 100, &[10, 20]);
        assert!(!pctx.ephemeral_present());
        // L1 + sched side-effects still happen for the admitted req.
        assert_eq!(pctx.in_flight_count(), 1);
        assert_eq!(pctx.sched_in_flight_count(), 1);
    }

    #[test]
    fn on_admit_with_empty_ephemeral_is_noop_for_l3() {
        let pctx = pctx();
        pctx.on_admit(1, 100, &[10, 20]);
        // L3 stays empty; L1 + sched populated.
        assert!(!pctx.ephemeral_present());
        assert_eq!(pctx.in_flight_count(), 1);
        assert_eq!(pctx.sched_in_flight_count(), 1);
    }

    #[test]
    fn on_sse_decode_only_step_drops_ephemeral() {
        let pctx = pctx();
        let _ = pctx.query(7);
        let s = step(1, 0, vec![out(7, "DECODE", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert!(!pctx.ephemeral_present());
    }

    #[test]
    fn on_sse_keeps_fresh_ephemeral_when_engine_has_prefill() {
        let pctx = pctx();
        let _ = pctx.query(7);
        let s = step(1, 32, vec![out(7, "PREFILL", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert!(pctx.ephemeral_present());
    }

    #[test]
    fn on_sse_drops_stale_ephemeral_when_engine_has_prefill() {
        let pctx = pctx();
        let _ = pctx.query(7);
        let s = step(5, 32, vec![out(7, "PREFILL", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert!(!pctx.ephemeral_present());
    }

    #[test]
    fn on_sse_drives_l1_apply_sse() {
        let pctx = pctx();
        pctx.insert_in_flight(1, &[10, 20]);
        pctx.insert_in_flight(2, &[10, 30]);
        let s = step(1, 0, vec![out(1, "DECODE", true), out(2, "DECODE", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert_eq!(pctx.in_flight_count(), 1);
    }

    #[test]
    fn on_sse_advances_anchor() {
        let pctx = pctx();
        let s = step(42, 0, vec![]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert_eq!(pctx.last_sse_step_id.load(Ordering::Acquire), 42);
    }
}
