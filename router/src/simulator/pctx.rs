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
use super::rollout::{RolloutBuffer, RolloutGist, RolloutSlot};
use super::sched::{ReqProgress, SchedSnapshot};

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
    /// Engine block size in tokens (typically 16). Used to convert
    /// between cache-hit block counts and token counts in `query`'s
    /// schedule loop.
    block_size: u32,
    /// Engine per-step token budget (typically 1024). The `query`
    /// schedule loop uses this as the chunked-prefill budget per slot.
    token_budget: u32,
}

impl PCtx {
    pub fn new(regressor: Box<dyn TrainedPredictor>, block_size: u32, token_budget: u32) -> Self {
        Self {
            regressor: Mutex::new(regressor),
            mirror: Mutex::new(IncrementalMirror::new()),
            ephemeral: Mutex::new(None),
            sched: Mutex::new(SchedSnapshot::new()),
            last_sse_step_id: AtomicU64::new(0),
            block_size: block_size.max(1),
            token_budget: token_budget.max(1),
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

    /// Trigger A — speculative rollout for the candidate request.
    ///
    /// Inputs:
    ///   * `candidate_id`     — request id to score.
    ///   * `input_length`     — candidate's prompt length in tokens.
    ///   * `candidate_hashes` — candidate's prefix block hashes (from
    ///     `entry.block_hash_state.get_hashes()`).
    ///   * `sctx_prefix_hits` — number of candidate's leading blocks
    ///     the engine's SSE-confirmed cache (`SCtx.block_hash`)
    ///     reports as cached. The composite hit-count used by the
    ///     simulator is `max(sctx_prefix_hits, mirror.prefix_match)`
    ///     per A2's max-merge rule.
    ///
    /// Algorithm:
    ///   1. Compute composite cache-hit blocks; subtract from the
    ///      candidate's prompt length to get effective prefill work.
    ///   2. Clone `sched`, push candidate to `waiting.back()` (FCFS
    ///      per F4).
    ///   3. Schedule loop (per slot):
    ///      a. All running decode requests contribute 1 token each.
    ///      b. Remaining `token_budget` goes to chunked prefill —
    ///         continuing prefill in `running` first, then pulling
    ///         new requests from `waiting` until the budget is full
    ///         or the queue is empty.
    ///      c. Build `BatchForPredictor`; call `regressor.predict`.
    ///      d. Push `RolloutSlot { batch, predicted_lat_ms,
    ///         prefill_rids, decode_rids }`.
    ///      e. Track candidate lifecycle: `prefill_begin_step` =
    ///         first slot it appears in PREFILL; `prefill_end_step` =
    ///         slot where its `processed_tokens == input_length`;
    ///         `in_decode_step` = first slot after it enters DECODE.
    ///      f. Stop after the in-decode step (per A3's lock).
    ///   4. Cache the buffer in L3, return its `gist()`.
    ///
    /// Hard cap of 256 slots prevents runaway loops in pathological
    /// states; reaching it returns whatever was simulated so far.
    pub fn query(
        &self,
        candidate_id: u64,
        input_length: u32,
        candidate_hashes: &[u64],
        sctx_prefix_hits: usize,
    ) -> RolloutGist {
        let anchor = self.last_sse_step_id.load(Ordering::Acquire);

        // Composite cache hit blocks → tokens.
        let mirror_hits = self
            .mirror
            .lock()
            .expect("PCtx mirror poisoned")
            .prefix_match(candidate_hashes);
        let composite_hits = sctx_prefix_hits.max(mirror_hits);
        let cached_tokens = (composite_hits as u32).saturating_mul(self.block_size);
        let initial_processed = cached_tokens.min(input_length);

        // Snapshot sched and add candidate.
        let mut sched_local = self.sched.lock().expect("PCtx sched poisoned").clone();
        let mut candidate = ReqProgress::new(candidate_id, input_length);
        candidate.processed_tokens = initial_processed;
        sched_local.admit(candidate);

        let buffer = self.run_schedule_loop(candidate_id, sched_local);
        let gist = buffer.gist();

        *self.ephemeral.lock().expect("PCtx ephemeral poisoned") = Some(EphemeralRollout {
            candidate_id: Some(candidate_id),
            buffer,
            sse_anchor_step_id: anchor,
        });

        gist
    }

    /// Inner schedule loop. Pure function over the cloned snapshot —
    /// does not touch any of `self`'s locks beyond the regressor's
    /// (acquired per-slot for predict).
    ///
    /// Stop conditions (any one):
    ///   * candidate has completed prefill AND one in-decode slot
    ///     has been simulated (the canonical A3 stop)
    ///   * 256-slot hard cap (defensive, should never trigger in
    ///     normal traffic)
    ///   * sched goes empty before candidate is reached (also
    ///     defensive — implies state inconsistency)
    fn run_schedule_loop(
        &self,
        candidate_id: u64,
        mut sched: SchedSnapshot,
    ) -> RolloutBuffer {
        const MAX_SLOTS: usize = 256;

        let mut buffer = RolloutBuffer {
            candidate_id: Some(candidate_id),
            ..Default::default()
        };

        let mut prefill_begin_step: Option<usize> = None;
        let mut prefill_end_step: Option<usize> = None;
        let mut in_decode_step: Option<usize> = None;

        for slot_idx in 0..MAX_SLOTS {
            let mut budget = self.token_budget as i64;
            let mut prefill_rids: smallvec::SmallVec<[u64; 2]> = Default::default();
            let mut decode_rids: smallvec::SmallVec<[u64; 4]> = Default::default();
            let mut num_prefill_tokens: Vec<usize> = Vec::new();
            let mut num_prefill_computed: Vec<usize> = Vec::new();
            let mut num_decode_computed: Vec<usize> = Vec::new();

            // (a) all running decode requests: 1 token each.
            // (b) running's continuing chunked prefill takes whatever
            //     budget remains. We iterate running.values() but
            //     handle decode-vs-prefill differently. Collect first
            //     so we can sort / determinise.
            let mut running_decoders: Vec<u64> = Vec::new();
            let mut running_prefillers: Vec<u64> = Vec::new();
            for r in sched.running.values() {
                if r.is_prefilling() {
                    running_prefillers.push(r.request_id);
                } else {
                    running_decoders.push(r.request_id);
                }
            }
            running_decoders.sort_unstable();
            running_prefillers.sort_unstable();

            for rid in &running_decoders {
                if let Some(req) = sched.running.get(rid) {
                    decode_rids.push(*rid);
                    num_decode_computed.push(req.processed_tokens as usize);
                    budget -= 1;
                }
            }

            // (c) running prefillers continue first.
            for rid in &running_prefillers {
                if budget <= 0 {
                    break;
                }
                if let Some(req) = sched.running.get_mut(rid) {
                    let chunk = (req.prefill_remaining() as i64).min(budget).max(0) as u32;
                    if chunk == 0 {
                        continue;
                    }
                    prefill_rids.push(*rid);
                    num_prefill_tokens.push(chunk as usize);
                    num_prefill_computed.push(req.processed_tokens as usize);
                    req.processed_tokens += chunk;
                    budget -= chunk as i64;
                }
            }

            // (d) pull new requests from waiting until budget exhausted
            //     or queue empty.
            while budget > 0 {
                let Some(mut next) = sched.waiting.pop_front() else {
                    break;
                };
                let chunk = (next.prefill_remaining() as i64).min(budget).max(0) as u32;
                if chunk == 0 {
                    // Already-cached request (initial_processed >= input_length).
                    // Move it to running anyway so subsequent slots can
                    // count its decodes.
                    sched.running.insert(next.request_id, next);
                    continue;
                }
                prefill_rids.push(next.request_id);
                num_prefill_tokens.push(chunk as usize);
                num_prefill_computed.push(next.processed_tokens as usize);
                next.processed_tokens += chunk;
                budget -= chunk as i64;
                sched.running.insert(next.request_id, next);
            }

            // Stop early if no work this slot (sched truly empty after
            // candidate finished). Defensive — shouldn't happen given
            // the schedule loop pushes candidate at least once.
            if prefill_rids.is_empty() && decode_rids.is_empty() {
                break;
            }

            // Build BatchForPredictor for the regressor.
            let num_tokens: usize =
                num_prefill_tokens.iter().sum::<usize>() + decode_rids.len();
            let bs_safe = self.block_size as usize;
            let num_tokens_rounded = ((num_tokens + bs_safe - 1) / bs_safe) * bs_safe;
            let batch = BatchForPredictor {
                num_tokens,
                num_tokens_rounded,
                num_prefill_tokens,
                num_prefill_computed_tokens: num_prefill_computed,
                num_decode_computed_tokens: num_decode_computed,
                size: prefill_rids.len() + decode_rids.len(),
            };

            let predicted_lat_ms = self
                .regressor
                .lock()
                .expect("PCtx regressor poisoned")
                .predict(&batch);

            // Track candidate lifecycle.
            let candidate_in_prefill = prefill_rids.iter().any(|r| *r == candidate_id);
            let candidate_in_decode = decode_rids.iter().any(|r| *r == candidate_id);
            if candidate_in_prefill && prefill_begin_step.is_none() {
                prefill_begin_step = Some(slot_idx);
            }
            if candidate_in_prefill {
                if let Some(req) = sched.running.get(&candidate_id) {
                    if !req.is_prefilling() {
                        prefill_end_step = Some(slot_idx);
                    }
                }
            }
            if candidate_in_decode && in_decode_step.is_none() {
                in_decode_step = Some(slot_idx);
            }

            buffer.slots.push(RolloutSlot {
                batch,
                predicted_lat_ms,
                prefill_rids,
                decode_rids,
            });

            // Stop after we've simulated candidate's first decode step.
            if in_decode_step.is_some() {
                break;
            }
        }

        buffer.prefill_begin_step = prefill_begin_step;
        buffer.prefill_end_step = prefill_end_step;
        buffer.in_decode_step = in_decode_step;
        buffer
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
    /// 2. Sched: `SchedSnapshot::sync(m)` to keep per-request progress
    ///    (waiting / running, processed_tokens) consistent with the
    ///    engine's view. Required so the next `query()` rolls forward
    ///    from a fresh state.
    /// 3. L1: absorb the step's evictions / finishes / aborts /
    ///    preempts via `IncrementalMirror::apply_sse`.
    /// 4. L3: F3 cross-check (composition-based) when ephemeral has
    ///    a populated `slots[0]`; falls back to stale-anchor check
    ///    when the slot has empty composition (typical for Phase-3
    ///    placeholder buffer before T8 lands).
    ///
    /// Returns the L2 `(predicted, actual)` pair for Prometheus.
    pub fn on_sse(&self, batch: &BatchForPredictor, m: &EngineStepOutput) -> (f32, f32) {
        // L2 first.
        let (predicted, actual) = self.observe_step(batch, m.latency as f32);

        // Sched + L1 absorption: lock order sched → mirror.
        // Sched.sync first so mirror.apply_sse sees the post-step
        // sched state when it self-arbitrates the evict path.
        let mut sched_guard = self.sched.lock().expect("PCtx sched poisoned");
        sched_guard.sync(m);
        self.mirror.lock().expect("PCtx mirror poisoned").apply_sse(m, &sched_guard);
        drop(sched_guard);

        // Bump anchor before L3 maintenance.
        self.last_sse_step_id.store(m.step_id, Ordering::Release);

        // L3 invariant + F3 cross-check.
        let engine_has_prefill = step_has_prefill(m);
        let mut slot = self.ephemeral.lock().expect("PCtx ephemeral poisoned");
        match (slot.as_mut(), engine_has_prefill) {
            (Some(e), true) => {
                let drift = if let Some(s0) = e.buffer.slots.first() {
                    if s0.composition_known() {
                        // F3: compare predicted slot[0] composition vs
                        // engine's actual step. Set-equality on
                        // (prefill_rids, decode_rids).
                        !rid_sets_match(s0, m)
                    } else {
                        // Composition not recorded (T8 hasn't filled
                        // it yet) — fall back to stale-anchor check.
                        e.sse_anchor_step_id < m.step_id.saturating_sub(1)
                    }
                } else {
                    // Empty slots: anchor-based stale check is the
                    // only signal available.
                    e.sse_anchor_step_id < m.step_id.saturating_sub(1)
                };
                if drift {
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

/// F3 cross-check: do the rid sets predicted in `slot` match the rid
/// sets the engine reports in `m.outputs`? Set-equality on both
/// PREFILL and DECODE buckets. Caller is expected to have verified
/// `slot.composition_known()` first.
fn rid_sets_match(
    slot: &super::rollout::RolloutSlot,
    m: &EngineStepOutput,
) -> bool {
    use std::collections::HashSet;
    let pred_pref: HashSet<u64> = slot.prefill_rids.iter().copied().collect();
    let pred_dec: HashSet<u64> = slot.decode_rids.iter().copied().collect();
    let mut act_pref: HashSet<u64> = HashSet::new();
    let mut act_dec: HashSet<u64> = HashSet::new();
    for o in &m.outputs {
        match o.state.as_str() {
            "PREFILL" => {
                act_pref.insert(o.request_id);
            }
            "DECODE" | "RUNNING" => {
                act_dec.insert(o.request_id);
            }
            _ => {}
        }
    }
    pred_pref == act_pref && pred_dec == act_dec
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
        PCtx::new(trained, 16, 1024)
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
        let pctx = PCtx::new(trained, cfg.block_size as u32, cfg.token_budget);

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
        let _ = pctx.query(7, 100, &[], 0);
        assert_eq!(pctx.ephemeral_candidate(), Some(7));
    }

    #[test]
    fn on_admit_promotes_matching_candidate_to_baseline() {
        let pctx = pctx();
        let _ = pctx.query(7, 100, &[], 0);
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
        let _ = pctx.query(7, 100, &[], 0);
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
        let _ = pctx.query(7, 100, &[], 0);
        let s = step(1, 0, vec![out(7, "DECODE", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert!(!pctx.ephemeral_present());
    }

    #[test]
    fn on_sse_keeps_fresh_ephemeral_when_engine_has_prefill() {
        let pctx = pctx();
        let _ = pctx.query(7, 100, &[], 0);
        // Match query's predicted slot[0] (candidate alone in PREFILL).
        let s = step(1, 32, vec![out(7, "PREFILL", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert!(pctx.ephemeral_present());
    }

    #[test]
    fn on_sse_drops_stale_ephemeral_via_anchor_fallback() {
        let pctx = pctx();
        // Manually install an ephemeral with empty slots so the F3
        // path falls through to the anchor-based stale check.
        {
            let _ = pctx.query(7, 100, &[], 0); // populate then clear
            let mut slot = pctx.ephemeral.lock().unwrap();
            if let Some(e) = slot.as_mut() {
                e.buffer.slots.clear();
                e.sse_anchor_step_id = 0;
            }
        }
        let s = step(5, 32, vec![out(7, "PREFILL", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        // Anchor (0) is older than step_id-1 (= 4) → stale → drop.
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

    #[test]
    fn on_sse_syncs_sched_promote_and_advance() {
        let pctx = pctx();
        // Admit + check it goes to waiting via sched.
        pctx.on_admit(7, 200, &[10, 20]);
        assert_eq!(pctx.sched_in_flight_count(), 1);
        // Send a PREFILL step processing 100 of the 200 prompt tokens.
        let mut o = out(7, "PREFILL", false);
        o.prev_computed_tokens = 0;
        let s = step(1, 100, vec![o]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        // Sched promoted 7 to running; processed_tokens advanced to 100.
        let sched = pctx.sched.lock().unwrap();
        assert!(sched.running.contains_key(&7));
        assert_eq!(sched.running[&7].processed_tokens, 100);
    }

    #[test]
    fn end_to_end_admit_step_query_lifecycle() {
        // Walks the full Phase-3 surface in one shot: admit a few
        // requests, drive the engine via several synthetic SSE steps,
        // then query a fresh candidate and verify the resulting gist
        // is non-trivial and the L3/sched/mirror state is consistent.
        let pctx = pctx();

        // Admit r1 (input=200) and r2 (input=300) with prefix hashes.
        pctx.on_admit(1, 200, &[10, 20, 30]);
        pctx.on_admit(2, 300, &[40, 50, 60]);
        assert_eq!(pctx.in_flight_count(), 2);
        assert_eq!(pctx.sched_in_flight_count(), 2);

        // Step 1: r1 starts PREFILL (200 tokens chunked at 200/step).
        let s1 = step(
            1,
            200,
            vec![{
                let mut o = out(1, "PREFILL", false);
                o.prev_computed_tokens = 0;
                o
            }],
        );
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s1);
        // After step 1, r1's processed_tokens = 200 (prefill done).
        assert_eq!(pctx.sched.lock().unwrap().running[&1].processed_tokens, 200);

        // Step 2: r2 starts PREFILL (300 tokens chunked at 300/step).
        let s2 = step(
            2,
            300,
            vec![{
                let mut o = out(2, "PREFILL", false);
                o.prev_computed_tokens = 0;
                o
            }],
        );
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s2);
        assert_eq!(pctx.sched.lock().unwrap().running[&2].processed_tokens, 300);

        // Step 3: both in DECODE.
        let s3 = step(3, 0, vec![out(1, "DECODE", false), out(2, "DECODE", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s3);
        let sched = pctx.sched.lock().unwrap();
        assert_eq!(sched.running[&1].processed_tokens, 201);
        assert_eq!(sched.running[&2].processed_tokens, 301);
        drop(sched);

        // Now query a fresh candidate (id=99, input=500 tokens).
        let gist = pctx.query(99, 500, &[], 0);
        assert!(gist.ttft_ms.is_some(), "ttft_ms must be populated");
        assert!(gist.chunked_prefill_steps.is_some(), "prefill steps populated");
        assert!(gist.in_decode_tbt_ms.is_some(), "in-decode TBT populated");

        // 500 tokens, budget=1024, 2 ongoing decoders eat 2 tokens →
        // candidate's chunk fits in one slot.
        assert_eq!(gist.chunked_prefill_steps, Some(1));

        // Inspect L3 buffer composition for slot 0 — candidate
        // PREFILL alongside r1+r2 DECODE.
        let slot = pctx.ephemeral.lock().unwrap();
        let buf = &slot.as_ref().unwrap().buffer;
        assert!(buf.slots.len() >= 2);
        assert_eq!(&buf.slots[0].prefill_rids[..], &[99]);
        let mut s0_dec: Vec<u64> = buf.slots[0].decode_rids.iter().copied().collect();
        s0_dec.sort();
        assert_eq!(s0_dec, vec![1, 2]);
        // Slot 1: candidate joins decode set.
        let mut s1_dec: Vec<u64> = buf.slots[1].decode_rids.iter().copied().collect();
        s1_dec.sort();
        assert_eq!(s1_dec, vec![1, 2, 99]);
        // Anchor must be the latest SSE step we sent.
        assert_eq!(slot.as_ref().unwrap().sse_anchor_step_id, 3);
    }

    #[test]
    fn query_sim_empty_sched_candidate_alone() {
        let pctx = pctx();
        // Candidate input = 800 tokens, token_budget = 1024 → fits in
        // a single PREFILL slot. Then 1 in_decode slot. Total 2 slots.
        let gist = pctx.query(42, 800, &[], 0);
        assert!(gist.ttft_ms.is_some(), "ttft should be set");
        assert_eq!(gist.chunked_prefill_steps, Some(1));
        assert!(gist.in_decode_tbt_ms.is_some());

        let slot = pctx.ephemeral.lock().unwrap();
        let buf = &slot.as_ref().unwrap().buffer;
        assert_eq!(buf.slots.len(), 2);
        assert_eq!(buf.prefill_begin_step, Some(0));
        assert_eq!(buf.prefill_end_step, Some(0));
        assert_eq!(buf.in_decode_step, Some(1));
        assert_eq!(&buf.slots[0].prefill_rids[..], &[42]);
        assert!(buf.slots[0].decode_rids.is_empty());
        assert!(buf.slots[1].prefill_rids.is_empty());
        assert_eq!(&buf.slots[1].decode_rids[..], &[42]);
    }

    #[test]
    fn query_sim_chunked_prefill_spans_multiple_slots() {
        let pctx = pctx();
        // Candidate input = 3000 tokens, budget = 1024 → 3 PREFILL
        // slots, then in_decode.
        let gist = pctx.query(42, 3000, &[], 0);
        assert_eq!(gist.chunked_prefill_steps, Some(3));
        let slot = pctx.ephemeral.lock().unwrap();
        let buf = &slot.as_ref().unwrap().buffer;
        assert_eq!(buf.slots.len(), 4);
        assert_eq!(buf.prefill_begin_step, Some(0));
        assert_eq!(buf.prefill_end_step, Some(2));
        assert_eq!(buf.in_decode_step, Some(3));
    }

    #[test]
    fn query_sim_with_existing_decoders_in_running() {
        let pctx = pctx();
        pctx.on_admit(1, 50, &[]);
        pctx.on_admit(2, 50, &[]);
        {
            let mut sched = pctx.sched.lock().unwrap();
            sched.promote_to_running(1).unwrap().processed_tokens = 50;
            sched.promote_to_running(2).unwrap().processed_tokens = 50;
        }
        let gist = pctx.query(42, 800, &[], 0);
        assert_eq!(gist.chunked_prefill_steps, Some(1));

        let slot = pctx.ephemeral.lock().unwrap();
        let buf = &slot.as_ref().unwrap().buffer;
        assert_eq!(&buf.slots[0].prefill_rids[..], &[42]);
        let mut s0_dec: Vec<u64> = buf.slots[0].decode_rids.iter().copied().collect();
        s0_dec.sort();
        assert_eq!(s0_dec, vec![1, 2]);
        let mut s1_dec: Vec<u64> = buf.slots[1].decode_rids.iter().copied().collect();
        s1_dec.sort();
        assert_eq!(s1_dec, vec![1, 2, 42]);
    }

    #[test]
    fn query_sim_cache_hit_shrinks_effective_input() {
        let pctx = pctx();
        // 1600-token prompt; SCtx says 50 leading blocks cached
        // (= 50 * 16 = 800 tokens). Effective prefill = 800 → 1 slot.
        let gist = pctx.query(42, 1600, &[], 50);
        assert_eq!(gist.chunked_prefill_steps, Some(1));
    }

    #[test]
    fn f3_drops_ephemeral_when_predicted_composition_mismatches() {
        use smallvec::smallvec;
        let pctx = pctx();
        let _ = pctx.query(7, 100, &[], 0);
        // Manually inject a slot[0] predicting decode={1,2}, prefill={}.
        {
            let mut slot = pctx.ephemeral.lock().unwrap();
            if let Some(e) = slot.as_mut() {
                e.buffer.slots.push(super::super::rollout::RolloutSlot {
                    batch: BatchForPredictor::default(),
                    predicted_lat_ms: 1.0,
                    prefill_rids: smallvec![],
                    decode_rids: smallvec![1, 2],
                });
            }
        }
        // Engine actually steps with decode={1,3} — mismatch.
        let s = step(1, 32, vec![out(1, "DECODE", false), out(3, "DECODE", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        assert!(!pctx.ephemeral_present());
    }

    #[test]
    fn f3_keeps_ephemeral_when_predicted_composition_matches() {
        use smallvec::smallvec;
        let pctx = pctx();
        let _ = pctx.query(7, 100, &[], 0);
        // Replace slot[0] with the composition we want to test against.
        {
            let mut slot = pctx.ephemeral.lock().unwrap();
            if let Some(e) = slot.as_mut() {
                e.buffer.slots.clear();
                e.buffer.slots.push(super::super::rollout::RolloutSlot {
                    batch: BatchForPredictor::default(),
                    predicted_lat_ms: 1.0,
                    prefill_rids: smallvec![7],
                    decode_rids: smallvec![1, 2],
                });
                e.sse_anchor_step_id = 1; // fresh anchor — fallback won't fire
            }
        }
        let s = step(
            1,
            32,
            vec![
                out(7, "PREFILL", false),
                out(1, "DECODE", false),
                out(2, "DECODE", false),
            ],
        );
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);
        // Prediction matched — ephemeral kept.
        assert!(pctx.ephemeral_present());
    }
}
