// PCtx — per-replica predictor context owned by the colocation controller.
//
// Three-layer state model:
//
//   * L1 — `IncrementalMirror`. Per-replica overlay of the in-flight
//     prefix-cache state. Updated by `on_admit` (speculative ADD) and
//     `on_sse` (eviction / finish absorption via `apply_sse`).
//   * L2 — `LinregCorrected` regressor (offline-trained inner model
//     wrapped by an online linear correction). Updated by `on_sse`'s
//     piggyback `observe_step` call. Has no notion of admission and is
//     unaffected by `on_admit`. There is no L2 "drop" — calibration is
//     monotonically incremental over the process lifetime.
//   * L3 — `EphemeralRollout`. A single-slot Option<…> holding the
//     most recent `query()`'s `RolloutBuffer` plus the candidate id and
//     SSE step id at which it was anchored. Drop / rebuild semantics
//     live entirely on this field; see field doc on `ephemeral`.
//
// Public API (the three triggers):
//
//   * `query(candidate_id) -> RolloutGist` — speculative rollout for a
//     candidate request. Stores the rollout buffer in L3 keyed by the
//     candidate id and the current SSE step id; returns the gist for
//     the caller (the policy scheduler).
//   * `on_admit(request_id)` — invoked by `policy_runner` immediately
//     after the chosen <name>-q policy commits a request to this
//     replica's commit buffer. Promotes a matching L3 buffer from
//     "candidate-bound" to "baseline" (`candidate_id := None`), or
//     marks L3 stale if the admitted request did not match.
//   * `on_sse(step)` — SSE consumer hook (piggyback). Drives L2
//     calibration, absorbs the step into L1 via `apply_sse`, and
//     enforces the L3 invariant: when the engine still has prefill
//     work in progress (`prefill_token_budget < token_budget`) the
//     ephemeral buffer must be non-empty; if it isn't, schedule a
//     rebuild.
//
// Phase-2 status: L1 + L2 paths are fully wired. L3 lifecycle bits
// (storage, anchor, promote-or-rebuild markers) are in place but the
// rollout algorithm itself (`query_sim`) and the rebuild routine are
// stubbed and tracked by Phase-3.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::engine_client::EngineStepOutput;

use super::batch::BatchForPredictor;
use super::mirror::IncrementalMirror;
use super::predictor::TrainedPredictor;
use super::rollout::{RolloutBuffer, RolloutGist};

/// L3 — single-slot ephemeral rollout. Identified by the candidate id
/// at the time of `query`, anchored to the SSE step id observed when
/// the rollout was constructed (used by `on_sse` to detect stale L3
/// state across an SSE event boundary).
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
    /// L2 — online-corrected predictor (linreg wrapper around an inner
    /// ML model). `Mutex` because `TrainedPredictor::calibrate` takes
    /// `&mut self`. The critical section is microseconds; std `Mutex`
    /// is fine.
    regressor: Mutex<Box<dyn TrainedPredictor>>,
    /// L1 — per-replica incremental KV-cache mirror. Updated by
    /// admission / eviction. Read by the outer DES rollout.
    mirror: Mutex<IncrementalMirror>,
    /// L3 — at most one rollout buffer at a time. `None` is a legal
    /// state when (a) no `query` has run yet, or (b) the engine has
    /// transitioned into a decode-only steady-state (no prefill work
    /// pending), in which case the buffer's only legal content would
    /// be empty anyway.
    ephemeral: Mutex<Option<EphemeralRollout>>,
    /// Monotonic engine step id observed by the most recent `on_sse`
    /// call. Used as the anchor for newly created `EphemeralRollout`s
    /// and as the freshness check in `on_sse`'s invariant maintenance.
    /// Atomic because `query()` reads it without locking the regressor
    /// or mirror Mutexes.
    last_sse_step_id: AtomicU64,
}

impl PCtx {
    pub fn new(regressor: Box<dyn TrainedPredictor>, num_blocks: usize) -> Self {
        Self {
            regressor: Mutex::new(regressor),
            mirror: Mutex::new(IncrementalMirror::new(num_blocks)),
            ephemeral: Mutex::new(None),
            last_sse_step_id: AtomicU64::new(0),
        }
    }

    /// L2 predict + calibrate in one critical section. Returns the
    /// `(predicted_ms, actual_ms)` pair so the caller can emit
    /// per-step Prometheus histograms. This method is the only L2
    /// mutation point.
    pub fn observe_step(&self, batch: &BatchForPredictor, actual_ms: f32) -> (f32, f32) {
        let mut g = self.regressor.lock().expect("PCtx regressor poisoned");
        let predicted = g.predict(batch);
        g.calibrate(batch, actual_ms);
        (predicted, actual_ms)
    }

    /// L2 predict-only (no calibration). Available for forward-looking
    /// scoring; not used by the SSE hot path.
    pub fn predict(&self, batch: &BatchForPredictor) -> f32 {
        self.regressor.lock().expect("PCtx regressor poisoned").predict(batch)
    }

    /// L1 ADD (low-level). Used both by `on_admit` (speculative
    /// admission ADD with placeholder indices once the index story is
    /// finalised) and by unit tests. Idempotent per request_id.
    pub fn insert_in_flight(&self, request_id: u64, hashes: &[u64], indices: Vec<u64>) {
        self.mirror
            .lock()
            .expect("PCtx mirror poisoned")
            .insert_request(request_id, hashes, indices);
    }

    /// L1 REMOVE (low-level). Used by `on_sse` via `apply_sse`; kept
    /// public for direct use by policy code that aborts a request
    /// before it reaches the SSE path.
    pub fn remove_in_flight(&self, request_id: u64) {
        self.mirror.lock().expect("PCtx mirror poisoned").remove_request(request_id);
    }

    /// L1 EVICT (low-level). Same scope as `remove_in_flight`.
    pub fn evict_blocks(&self, block_indices: &[u64]) {
        self.mirror.lock().expect("PCtx mirror poisoned").remove_blocks(block_indices);
    }

    /// In-flight request count, observability only.
    pub fn in_flight_count(&self) -> usize {
        self.mirror.lock().expect("PCtx mirror poisoned").in_flight_count()
    }

    // ---------- L3 trigger surface ----------

    /// Trigger A — speculative rollout for `candidate_id`. Caches the
    /// resulting buffer in L3 anchored to the latest SSE step id and
    /// returns the projected gist for the policy scheduler.
    ///
    /// Phase-2 placeholder: the rollout algorithm itself (`query_sim`)
    /// is not yet implemented. This method returns the default gist
    /// (all-None) and stores an empty buffer at the L3 slot so that
    /// the lifecycle (`on_admit` promotion, `on_sse` invariant) can
    /// be exercised end-to-end ahead of the Phase-3 algorithm work.
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

    /// Trigger B — admission. Called by `policy_runner` once the
    /// scheduler commits `request_id` to this replica's commit buffer.
    ///
    /// L3 effect: if the cached ephemeral was generated by a `query`
    /// for the same candidate, promote it (`candidate_id := None`) so
    /// it becomes the baseline trajectory observed from now on. If
    /// the cached ephemeral targeted a different candidate (or there
    /// is no cached ephemeral), drop L3; the next `on_sse` invariant
    /// check will trigger a baseline rebuild.
    ///
    /// L1 effect: the speculative ADD is intentionally a no-op in
    /// Phase 2. At admission time the engine has not yet allocated
    /// block indices for the new request, so insertion into the
    /// mirror's index-keyed tree is deferred to the first SSE event
    /// that surfaces the request via `cur_used_block_ids`. Phase 3
    /// will move this earlier once the placeholder-index design is
    /// settled.
    pub fn on_admit(&self, request_id: u64) {
        let mut slot = self.ephemeral.lock().expect("PCtx ephemeral poisoned");
        match slot.as_mut() {
            Some(e) if e.candidate_id == Some(request_id) => {
                // Promote: rollout was for this exact request, so its
                // contents now describe the actual baseline trajectory.
                e.candidate_id = None;
                e.buffer.candidate_id = None;
            }
            _ => {
                // Stale or absent. Drop and let `on_sse` rebuild.
                *slot = None;
            }
        }
    }

    /// Trigger C — SSE event from the engine for this replica.
    ///
    /// 1. L2: piggyback predict + calibrate (returned for metrics).
    /// 2. L1: absorb the step's evictions / finishes / aborts via
    ///    `IncrementalMirror::apply_sse`.
    /// 3. L3 invariant: if the engine has prefill work in progress
    ///    after this step (heuristic: `prefill_tokens > 0` OR there
    ///    is at least one PREFILL output) AND the ephemeral slot is
    ///    empty, mark a rebuild request so the next `query()` (or a
    ///    Phase-3 background rebuilder) regenerates a baseline. The
    ///    actual rebuild routine is Phase-3.
    ///
    /// Returns the L2 `(predicted, actual)` pair so the caller can
    /// emit Prometheus histograms; mirrors the previous
    /// `observe_step` return shape.
    pub fn on_sse(&self, batch: &BatchForPredictor, m: &EngineStepOutput) -> (f32, f32) {
        // L2 first — keeps the regressor critical section short.
        let (predicted, actual) = self.observe_step(batch, m.latency as f32);

        // L1 absorption.
        self.mirror.lock().expect("PCtx mirror poisoned").apply_sse(m);

        // Bump the SSE anchor *before* the invariant check so any new
        // ephemeral built downstream is anchored to this step.
        self.last_sse_step_id.store(m.step_id, Ordering::Release);

        // L3 invariant maintenance.
        let engine_has_prefill = step_has_prefill(m);
        let mut slot = self.ephemeral.lock().expect("PCtx ephemeral poisoned");
        match (slot.as_mut(), engine_has_prefill) {
            (Some(e), true) => {
                // Stale-buffer detection: if the buffer is anchored to
                // an older SSE step the predicted slots no longer line
                // up with reality. Phase-3 will validate-or-rebuild
                // in-place; for now we conservatively drop, prompting
                // the next `query`/rebuild to start fresh.
                if e.sse_anchor_step_id < m.step_id.saturating_sub(1) {
                    *slot = None;
                }
            }
            (None, true) => {
                // Invariant violated: engine has prefill work but L3
                // is empty. Phase-3 will trigger a baseline rebuild
                // here. For now leave None and trace.
                tracing::trace!(
                    target: "simulator",
                    step = m.step_id,
                    "L3 invariant: engine has prefill but ephemeral is empty (rebuild deferred to Phase 3)"
                );
            }
            (Some(_), false) => {
                // Engine is decode-only — empty buffer is legal here,
                // and a non-empty buffer should be discarded since its
                // remaining-prefill predictions are vacuously stale.
                *slot = None;
            }
            (None, false) => {
                // Both legal: nothing to do.
            }
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
/// The `prefill_token_budget < total_token_budget` test would be more
/// precise but the totals aren't carried on the SSE payload, so the
/// per-step view is what we have.
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

    /// A predictor that returns `value`, useful for closed-form testing of
    /// the linreg correction loop without any CSV/file dependency.
    struct ConstPredictor(f32);
    impl Predictor for ConstPredictor {
        fn predict(&self, _b: &BatchForPredictor) -> f32 {
            self.0
        }
    }

    fn pctx() -> PCtx {
        let inner = Arc::new(ConstPredictor(1.0));
        let trained = Box::new(LinregCorrected::new(inner, &SimulatorConfig::default()));
        PCtx::new(trained, 1024)
    }

    fn step(
        step_id: u64,
        prefill_tokens: usize,
        outputs: Vec<RequestStepOutput>,
        evicted: Vec<u64>,
    ) -> EngineStepOutput {
        EngineStepOutput {
            prefill_tokens,
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
        let pctx = PCtx::new(trained, 1024);

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
        pctx.insert_in_flight(1, &[10, 20], vec![0, 1]);
        pctx.insert_in_flight(2, &[10, 30], vec![2, 3]);
        assert_eq!(pctx.in_flight_count(), 2);
        pctx.remove_in_flight(1);
        assert_eq!(pctx.in_flight_count(), 1);
        pctx.evict_blocks(&[3]);
        assert_eq!(pctx.in_flight_count(), 0);
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
        pctx.on_admit(7);
        assert!(pctx.ephemeral_present());
        assert_eq!(pctx.ephemeral_candidate(), None); // promoted = baseline
    }

    #[test]
    fn on_admit_drops_mismatched_candidate() {
        let pctx = pctx();
        let _ = pctx.query(7);
        pctx.on_admit(99); // different request admitted
        assert!(!pctx.ephemeral_present());
    }

    #[test]
    fn on_admit_with_empty_ephemeral_is_noop() {
        let pctx = pctx();
        pctx.on_admit(1);
        assert!(!pctx.ephemeral_present());
    }

    #[test]
    fn on_sse_decode_only_step_drops_ephemeral() {
        let pctx = pctx();
        let _ = pctx.query(7);
        // decode-only step: prefill_tokens=0, no PREFILL outputs.
        let s = step(1, 0, vec![out(7, "DECODE", false)], vec![]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        // Decode-only path drops the ephemeral entirely.
        assert!(!pctx.ephemeral_present());
    }

    #[test]
    fn on_sse_keeps_fresh_ephemeral_when_engine_has_prefill() {
        let pctx = pctx();
        let _ = pctx.query(7); // anchored at step_id=0
        // Fresh: anchor (0) >= step_id - 1 (= 0).
        let s = step(1, 32, vec![out(7, "PREFILL", false)], vec![]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert!(pctx.ephemeral_present());
    }

    #[test]
    fn on_sse_drops_stale_ephemeral_when_engine_has_prefill() {
        let pctx = pctx();
        let _ = pctx.query(7); // anchored at step_id=0
        // Stale: anchor (0) < step_id - 1 (= 4).
        let s = step(5, 32, vec![out(7, "PREFILL", false)], vec![]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert!(!pctx.ephemeral_present());
    }

    #[test]
    fn on_sse_drives_l1_apply_sse() {
        let pctx = pctx();
        pctx.insert_in_flight(1, &[10, 20], vec![0, 1]);
        pctx.insert_in_flight(2, &[10, 30], vec![2, 3]);
        let s = step(1, 0, vec![out(1, "DECODE", true), out(2, "DECODE", false)], vec![]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        // Request 1 finished → mirror should drop it.
        assert_eq!(pctx.in_flight_count(), 1);
    }

    #[test]
    fn on_sse_advances_anchor() {
        let pctx = pctx();
        let s = step(42, 0, vec![], vec![]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert_eq!(pctx.last_sse_step_id.load(Ordering::Acquire), 42);
    }
}
