// PCtx — per-replica predictor context owned by the colocation controller.
//
// Three-layer state model:
//
//   * L1 — `IncrementalMirror` + `SchedSnapshot`. The world model.
//     Mirror = per-replica overlay of in-flight prefix hashes (V=ReqId
//     trie); sched = per-request progress (waiting / running). Both
//     updated by `on_admit` (insert / push) and `on_sse` (apply_sse +
//     sync). See `mirror.rs` and `sched.rs`. Two data structures
//     because they hold different shapes of in-flight state; one
//     conceptual layer.
//   * L2 — `RegressionalPredictor<P, C>` (offline-trained inner
//     `Predictor` wrapped by an online `Corrector`). Updated by
//     `on_sse`'s piggyback `observe_step` call. No "drop" —
//     calibration is monotonically incremental over the process
//     lifetime.
//   * L3 — `EphemeralRollout`. A single-slot Option<…> holding the
//     most recent rollout: `RolloutBuffer` + `candidate_id` + SSE
//     anchor + `tail_sched` (state at the buffer's tail, used by the
//     next `query`'s `extend_with_candidate`) + optional `savepoint`
//     (split index + saved baseline tail, used to recover the
//     baseline portion of a stale candidate-bound buffer when the
//     candidate is admitted elsewhere).
//
// Lock discipline (NEVER take in a different order):
//
//   `sched → mirror → ephemeral`
//
// Every mutating method documents which locks it touches; tests in
// the same module rely on this order being respected.
//
// Public API — three triggers:
//
//   * `query(candidate_id, ...) -> RolloutGist` — speculative rollout
//     for a candidate. Three-way recovery: clean baseline (cid=None) →
//     extend; stale candidate-bound with savepoint → truncate to
//     baseline portion + extend; otherwise → rebuild from scratch.
//     Touches only L3.ephemeral on the live PCtx (sched cloned for
//     the per-call simulation; mirror borrowed read-only).
//   * `on_admit(rid, input_length, hashes)` — admission. Pushes to
//     `sched.waiting` and `mirror`; runs the L3 promote-or-drop
//     branch (clears savepoint on promote since the whole buffer is
//     now baseline).
//   * `on_sse(batch, m)` — SSE event from the engine: drives L2
//     calibration, syncs `sched`, runs `mirror.apply_sse`, advances
//     the anchor, runs L3 4-branch maintenance (Branch 1 F3-PASS pops
//     slot[0] and decrements savepoint index; F3-FAIL or stale-anchor
//     drops the entire ephemeral).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::engine::EngineStepOutput;

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
    /// Sched state at the END of the entire buffer (post-extension if
    /// extended). Used by `extend_with_candidate` to continue the
    /// simulation from the buffer's tail when the next `query` fires.
    tail_sched: SchedSnapshot,
    /// Savepoint set by `extend_with_candidate` BEFORE extending. Tuple
    /// is `(baseline_split_idx, baseline_tail_sched)`:
    ///   - `baseline_split_idx`: `slots[..idx]` are baseline; `slots[idx..]`
    ///     are the candidate's tail.
    ///   - `baseline_tail_sched`: sched state at the END of the baseline
    ///     portion (i.e. before the candidate's first slot was appended).
    /// Invariant: `Some(_)` iff `cid == Some(_)` AND a recoverable
    /// baseline portion exists.
    /// Cleared by:
    ///   - `on_admit` promote (entire buffer becomes baseline; no
    ///     separate savepoint needed).
    ///   - `on_sse` Branch 1 pop-on-keep when the saved index hits 0
    ///     (baseline portion fully consumed by SSE pops).
    /// Always `None` when `cid == None` (a clean baseline has no
    /// savepoint — the whole buffer IS the baseline).
    savepoint: Option<(usize, SchedSnapshot)>,
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
    /// Average generated length used to project per-request average TPOT.
    avg_output_len: u32,
}

impl PCtx {
    pub fn new(
        regressor: Box<dyn TrainedPredictor>,
        block_size: u32,
        token_budget: u32,
        avg_output_len: u32,
    ) -> Self {
        Self {
            regressor: Mutex::new(regressor),
            mirror: Mutex::new(IncrementalMirror::new()),
            ephemeral: Mutex::new(None),
            sched: Mutex::new(SchedSnapshot::new()),
            last_sse_step_id: AtomicU64::new(0),
            block_size: block_size.max(1),
            token_budget: token_budget.max(1),
            avg_output_len: avg_output_len.max(1),
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
    /// Three-way recovery on the cached `EphemeralRollout`:
    ///   1. **Clean baseline** (`cid == None`): reuse the whole buffer
    ///      and `tail_sched`; extend with the new candidate.
    ///   2. **Stale candidate-bound with savepoint**: truncate to the
    ///      saved baseline portion and recover `baseline_tail`; extend
    ///      with the new candidate.
    ///   3. **Otherwise**: scratch rebuild from `pctx.sched`.
    ///
    /// Algorithm:
    ///   1. Compute composite cache-hit blocks; subtract from the
    ///      candidate's prompt length to get effective prefill work.
    ///   2. Acquire the baseline (per the recovery above) or a fresh
    ///      `SchedSnapshot` clone.
    ///   3. Schedule loop (per slot):
    ///      a. All running decode requests contribute 1 token each.
    ///      b. Remaining `token_budget` goes to chunked prefill —
    ///         continuing prefill in `running` first, then pulling
    ///         new requests from `waiting` until the budget is full
    ///         or the queue is empty.
    ///      c. Build `BatchForPredictor`; call `regressor.predict`.
    ///      d. Push `RolloutSlot { batch, predicted_lat_ms,
    ///         prefill_rids, decode_rids }`.
    ///      e. Track candidate lifecycle markers on the buffer.
    ///      f. Stop after the in-decode step (per A3's lock).
    ///   4. Cache the buffer in L3 alongside `tail_sched` and the
    ///      savepoint, then return its `gist()`.
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
        // Composite cache hit blocks → max-merge per A2.
        let mirror_hits = self
            .mirror
            .lock()
            .expect("PCtx mirror poisoned")
            .prefix_match(candidate_hashes);
        let composite_hits = sctx_prefix_hits.max(mirror_hits);

        // Three-way recovery on the cached ephemeral.
        let baseline = self
            .ephemeral
            .lock()
            .expect("PCtx ephemeral poisoned")
            .take()
            .and_then(|e| match e.candidate_id {
                None => {
                    // Clean baseline — reuse buffer + tail directly.
                    Some((e.buffer, e.sse_anchor_step_id, e.tail_sched))
                }
                Some(_) => {
                    // Stale candidate-bound. Try savepoint recovery.
                    match e.savepoint {
                        Some((idx, saved_tail)) if idx > 0 => {
                            let mut buf = e.buffer;
                            buf.slots.truncate(idx);
                            buf.candidate_id = None;
                            buf.prefill_begin_step = None;
                            buf.prefill_end_step = None;
                            buf.in_decode_step = None;
                            buf.max_avg_tpot_ms = None;
                            Some((buf, e.sse_anchor_step_id, saved_tail))
                        }
                        _ => None, // savepoint cleared or never set → scratch rebuild
                    }
                }
            });

        let anchor = self.last_sse_step_id.load(Ordering::Acquire);

        let (buffer, tail_sched, savepoint) = match baseline {
            Some((b, _baseline_anchor, baseline_tail)) => {
                // We discard `_baseline_anchor` and use the current
                // `anchor` because the baseline buffer has already been
                // popped in lock-step with the engine via on_sse.
                self.extend_with_candidate(
                    b,
                    baseline_tail,
                    candidate_id,
                    input_length,
                    composite_hits,
                )
            }
            None => {
                let (buf, tail) = self.run_schedule_loop_from_scratch(
                    candidate_id,
                    input_length,
                    composite_hits,
                );
                (buf, tail, None)
            }
        };

        let gist = buffer.gist();

        *self.ephemeral.lock().expect("PCtx ephemeral poisoned") = Some(EphemeralRollout {
            candidate_id: Some(candidate_id),
            buffer,
            sse_anchor_step_id: anchor,
            tail_sched,
            savepoint,
        });

        gist
    }

    /// Build a fresh rollout from scratch starting from a clone of the
    /// current `pctx.sched`. Returns the buffer plus the `SchedSnapshot`
    /// at loop exit (used as `tail_sched` for the new ephemeral).
    fn run_schedule_loop_from_scratch(
        &self,
        candidate_id: u64,
        input_length: u32,
        composite_hits: usize,
    ) -> (RolloutBuffer, SchedSnapshot) {
        // Snapshot sched and add candidate.
        let mut sched_local = self.sched.lock().expect("PCtx sched poisoned").clone();
        let cached_tokens = (composite_hits as u32).saturating_mul(self.block_size);
        let initial_processed = cached_tokens.min(input_length);
        let mut candidate = ReqProgress::new(candidate_id, input_length);
        candidate.processed_tokens = initial_processed;
        sched_local.admit(candidate);

        let mut buffer = RolloutBuffer {
            candidate_id: Some(candidate_id),
            ..Default::default()
        };

        const MAX_SLOTS: usize = 256;
        while buffer.slots.len() < MAX_SLOTS {
            let slot_idx = buffer.slots.len();
            let Some(slot) =
                self.simulate_one_slot(&mut sched_local, slot_idx, candidate_id, &mut buffer)
            else {
                break;
            };
            buffer.slots.push_back(slot);
            if buffer.in_decode_step.is_some() {
                break;
            }
        }

        (buffer, sched_local)
    }

    /// Extend an existing baseline buffer with a new candidate. Sets a
    /// savepoint capturing the baseline split index and tail sched
    /// before running the per-slot loop, so a future `query` arriving
    /// with a different candidate can recover the baseline portion.
    ///
    /// Returns `(extended_buffer, post_extension_sched, savepoint)`.
    /// The 256-slot hard cap applies to the TOTAL buffer length
    /// (baseline + extension).
    fn extend_with_candidate(
        &self,
        mut buffer: RolloutBuffer,
        baseline_tail_sched: SchedSnapshot,
        candidate_id: u64,
        input_length: u32,
        composite_hits: usize,
    ) -> (RolloutBuffer, SchedSnapshot, Option<(usize, SchedSnapshot)>) {
        // SAVEPOINT: snapshot before extending. Empty baseline is the
        // edge case where there is nothing to recover (e.g. a 1-slot
        // baseline that was fully consumed by SSE pops).
        let savepoint = if !buffer.slots.is_empty() {
            Some((buffer.slots.len(), baseline_tail_sched.clone()))
        } else {
            None
        };

        // Continue from baseline_tail_sched with the new candidate.
        let mut local_sched = baseline_tail_sched;
        let cached_tokens = (composite_hits as u32).saturating_mul(self.block_size);
        let initial_processed = cached_tokens.min(input_length);
        let mut candidate = ReqProgress::new(candidate_id, input_length);
        candidate.processed_tokens = initial_processed;
        local_sched.admit(candidate);

        // Set candidate-bound markers (overwrite any baseline markers —
        // the buffer's previous candidate's lifecycle is no longer
        // relevant; the new candidate's lifecycle is what matters).
        buffer.candidate_id = Some(candidate_id);
        buffer.prefill_begin_step = None;
        buffer.prefill_end_step = None;
        buffer.in_decode_step = None;
        buffer.max_avg_tpot_ms = None;

        const MAX_SLOTS: usize = 256;
        while buffer.slots.len() < MAX_SLOTS {
            let slot_idx = buffer.slots.len();
            let Some(slot) =
                self.simulate_one_slot(&mut local_sched, slot_idx, candidate_id, &mut buffer)
            else {
                break;
            };
            buffer.slots.push_back(slot);
            if buffer.in_decode_step.is_some() {
                break;
            }
        }

        (buffer, local_sched, savepoint)
    }

    /// Simulate one engine forward step: pulls work from `sched`,
    /// builds a `BatchForPredictor`, asks the regressor for a latency
    /// estimate, and updates `buffer`'s lifecycle markers. Returns
    /// `None` when the slot would be empty (defensive early-stop;
    /// implies state inconsistency).
    fn simulate_one_slot(
        &self,
        sched: &mut SchedSnapshot,
        slot_idx: usize,
        candidate_id: u64,
        buffer: &mut RolloutBuffer,
    ) -> Option<RolloutSlot> {
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
            if let Some(req) = sched.running.get_mut(rid) {
                decode_rids.push(*rid);
                num_decode_computed.push(req.processed_tokens as usize);
                req.processed_tokens = req.processed_tokens.saturating_add(1);
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

        // Defensive early-stop: shouldn't happen because the loop's
        // caller pushed the candidate to waiting at least once.
        if prefill_rids.is_empty() && decode_rids.is_empty() {
            return None;
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

        // Track candidate lifecycle on the shared buffer.
        let candidate_in_prefill = prefill_rids.iter().any(|r| *r == candidate_id);
        let candidate_in_decode = decode_rids.iter().any(|r| *r == candidate_id);
        if candidate_in_prefill && buffer.prefill_begin_step.is_none() {
            buffer.prefill_begin_step = Some(slot_idx);
        }
        if candidate_in_prefill {
            if let Some(req) = sched.running.get(&candidate_id) {
                if !req.is_prefilling() {
                    buffer.prefill_end_step = Some(slot_idx);
                }
            }
        }
        if candidate_in_decode && buffer.in_decode_step.is_none() {
            buffer.in_decode_step = Some(slot_idx);
            let rollout_elapsed_ms = buffer
                .slots
                .iter()
                .map(|s| s.predicted_lat_ms)
                .sum::<f32>()
                + predicted_lat_ms;
            buffer.max_avg_tpot_ms =
                self.max_avg_tpot_ms(sched, predicted_lat_ms, rollout_elapsed_ms);
        }

        Some(RolloutSlot {
            batch,
            predicted_lat_ms,
            prefill_rids,
            decode_rids,
        })
    }

    fn max_avg_tpot_ms(
        &self,
        sched: &SchedSnapshot,
        first_tbt_time_ms: f32,
        rollout_elapsed_ms: f32,
    ) -> Option<f32> {
        let now = Instant::now();
        let avg_output_len = self.avg_output_len as f32;
        let mut max_avg_tpot_ms: Option<f32> = None;

        for req in sched.iter_in_flight() {
            let generated_len = req.generated_len().min(self.avg_output_len);
            let remaining_len = self.avg_output_len.saturating_sub(generated_len) as f32;
            let age_ms =
                now.duration_since(req.admit_time).as_secs_f32() * 1000.0 + rollout_elapsed_ms;
            let total_lifespan_ms = remaining_len * first_tbt_time_ms + age_ms;
            let avg_tpot_ms = total_lifespan_ms / avg_output_len;
            if max_avg_tpot_ms.map_or(true, |prev| avg_tpot_ms > prev) {
                max_avg_tpot_ms = Some(avg_tpot_ms);
            }
        }

        max_avg_tpot_ms
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
                    // Whole buffer is now baseline; no separate
                    // savepoint needed (and any prior savepoint is now
                    // logically merged into the baseline).
                    e.savepoint = None;
                    // tail_sched untouched — it's the post-extension
                    // tail, which is now the post-promote-baseline tail.
                }
                Some(e) if e.candidate_id.is_none() => {
                    // Baseline doesn't account for this admission — must
                    // rebuild on next query.
                    let _ = e;
                    *slot = None;
                }
                Some(e) => {
                    debug_assert!(
                        false,
                        "on_admit({}): protocol violation — cached ephemeral candidate_id={:?}",
                        request_id,
                        e.candidate_id
                    );
                    *slot = None; // safety net in release
                }
                None => {}
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
                let drift = match e.buffer.slots.front() {
                    Some(s0) if s0.composition_known() => !rid_sets_match(s0, m),
                    Some(_) => {
                        debug_assert!(
                            false,
                            "F3: slot[0] composition not recorded — invariant violated post-T8"
                        );
                        true // safety net in release: treat as drift, drop
                    }
                    None => {
                        debug_assert!(
                            false,
                            "L3 invariant: ephemeral exists but has no slots"
                        );
                        true
                    }
                };
                if drift {
                    *slot = None;
                } else {
                    // F3 PASS — consume validated head, advance anchor.
                    e.buffer.slots.pop_front();
                    e.buffer.prefill_begin_step =
                        e.buffer.prefill_begin_step.map(|i| i.saturating_sub(1));
                    e.buffer.prefill_end_step =
                        e.buffer.prefill_end_step.map(|i| i.saturating_sub(1));
                    e.buffer.in_decode_step =
                        e.buffer.in_decode_step.map(|i| i.saturating_sub(1));
                    e.sse_anchor_step_id = m.step_id;
                    // Decrement savepoint's baseline-split index in
                    // lock-step with the buffer pop. When it hits 0,
                    // the baseline portion has been fully consumed by
                    // SSE pops — drop the savepoint so the next query
                    // falls through to scratch rebuild instead of
                    // reusing the now-empty baseline + stale tail.
                    let drop_savepoint = if let Some((idx, _)) = e.savepoint.as_mut() {
                        *idx = idx.saturating_sub(1);
                        *idx == 0
                    } else {
                        false
                    };
                    if drop_savepoint {
                        e.savepoint = None;
                    }
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
    use crate::engine::RequestStepOutput;
    use super::super::config::SimulatorConfig;
    use super::super::predictor::{LinregCorrector, Predictor, RegressionalPredictor};
    use nohash_hasher::{BuildNoHashHasher, IntMap};

    struct ConstPredictor(f32);
    impl Predictor for ConstPredictor {
        fn predict(&self, _b: &BatchForPredictor) -> f32 {
            self.0
        }
    }

    fn pctx() -> PCtx {
        let inner = Arc::new(ConstPredictor(1.0));
        let trained = Box::new(RegressionalPredictor::new(
            inner,
            LinregCorrector::new(&SimulatorConfig::default()),
        ));
        PCtx::new(trained, 16, 1024, 1024)
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
        let trained = Box::new(RegressionalPredictor::new(inner, LinregCorrector::new(&cfg)));
        let pctx = PCtx::new(
            trained,
            cfg.block_size as u32,
            cfg.token_budget,
            cfg.avg_output_len,
        );

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
    #[cfg_attr(debug_assertions, should_panic(expected = "protocol violation"))]
    fn on_admit_drops_mismatched_candidate() {
        let pctx = pctx();
        let _ = pctx.query(7, 100, &[], 0);
        // Cached ephemeral has candidate_id = Some(7); admitting rid=8
        // is a protocol violation (PolicyRunner committed an admission
        // for a rid that was never `query`'d on this replica).
        // - debug builds: `debug_assert!` fires → test panics ✓
        // - release builds: safety net drops the ephemeral; verify
        //   the post-state below.
        pctx.on_admit(8, 100, &[10, 20]);
        assert!(!pctx.ephemeral_present(), "release safety net must drop ephemeral");
        // L1 + sched side-effects still happen for the admitted req.
        assert_eq!(pctx.in_flight_count(), 1);
        assert_eq!(pctx.sched_in_flight_count(), 1);
    }

    #[test]
    fn on_admit_baseline_drops_ephemeral() {
        let pctx = pctx();
        // First, install a baseline ephemeral by query+on_admit(matching).
        let _ = pctx.query(7, 100, &[], 0);
        pctx.on_admit(7, 100, &[10, 20]);
        // Sanity: ephemeral is now Some{cid=None} (baseline).
        assert!(pctx.ephemeral_present());
        assert_eq!(pctx.ephemeral_candidate(), None);

        // Admitting a different rid must drop the baseline (since
        // baseline doesn't account for the new admission and would be
        // stale on the next query).
        pctx.on_admit(99, 50, &[30, 40]);
        assert!(!pctx.ephemeral_present(), "baseline must be dropped on unrelated admit");
        // L1 + sched side-effects: both rid=7 and rid=99 are admitted.
        assert_eq!(pctx.in_flight_count(), 2);
        assert_eq!(pctx.sched_in_flight_count(), 2);
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

    // removed: `on_sse_drops_stale_ephemeral_via_anchor_fallback` exercised
    // the empty-slots anchor-fallback path inside Branch 1; that path is
    // gone (post-T8 invariant: slot[0] composition is always recorded, so
    // the only legal Branch-1 outcomes are F3 PASS or F3 mismatch). Empty
    // `slots` and missing composition now hit a `debug_assert!` and are
    // treated as drift in release builds — see Branch 1 in `on_sse`.

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
                e.buffer.slots.push_back(super::super::rollout::RolloutSlot {
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
        // Replace slots with two predicted slots so we can observe the
        // F3 PASS pop on slot[0] while still having a head left.
        {
            let mut slot = pctx.ephemeral.lock().unwrap();
            if let Some(e) = slot.as_mut() {
                e.buffer.slots.clear();
                e.buffer.slots.push_back(super::super::rollout::RolloutSlot {
                    batch: BatchForPredictor::default(),
                    predicted_lat_ms: 1.0,
                    prefill_rids: smallvec![7],
                    decode_rids: smallvec![1, 2],
                });
                e.buffer.slots.push_back(super::super::rollout::RolloutSlot {
                    batch: BatchForPredictor::default(),
                    predicted_lat_ms: 2.0,
                    prefill_rids: smallvec![],
                    decode_rids: smallvec![1, 2, 7],
                });
                e.buffer.prefill_begin_step = Some(0);
                e.buffer.prefill_end_step = Some(0);
                e.buffer.in_decode_step = Some(1);
                e.sse_anchor_step_id = 0;
            }
        }
        let s = step(
            5,
            32,
            vec![
                out(7, "PREFILL", false),
                out(1, "DECODE", false),
                out(2, "DECODE", false),
            ],
        );
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);

        // F3 PASS — ephemeral kept, head popped, indices decremented,
        // anchor advanced to m.step_id.
        let slot = pctx.ephemeral.lock().unwrap();
        let e = slot.as_ref().expect("ephemeral kept on F3 PASS");
        assert_eq!(e.buffer.slots.len(), 1, "head slot should be popped");
        assert_eq!(e.buffer.slots[0].predicted_lat_ms, 2.0);
        assert_eq!(e.buffer.prefill_begin_step, Some(0)); // saturating_sub clamps at 0
        assert_eq!(e.buffer.prefill_end_step, Some(0));   // saturating_sub clamps at 0
        assert_eq!(e.buffer.in_decode_step, Some(0));     // 1 - 1
        assert_eq!(e.sse_anchor_step_id, 5);
    }

    #[test]
    fn on_sse_pop_chain_survives_multi_step_keep() {
        use smallvec::smallvec;
        let pctx = pctx();
        let _ = pctx.query(7, 200, &[], 0);

        // Manually install a 3-slot buffer with matching predicted
        // composition for steps k+1, k+2, k+3.
        {
            let mut slot = pctx.ephemeral.lock().unwrap();
            let e = slot.as_mut().unwrap();
            e.buffer.slots.clear();
            for _ in 0..3 {
                e.buffer.slots.push_back(super::super::rollout::RolloutSlot {
                    batch: BatchForPredictor::default(),
                    predicted_lat_ms: 1.0,
                    prefill_rids: smallvec![7],
                    decode_rids: smallvec![],
                });
            }
            e.buffer.prefill_begin_step = Some(0);
            e.buffer.prefill_end_step = Some(2);
            e.buffer.in_decode_step = None;
            e.sse_anchor_step_id = 10; // k = 10
        }

        // Fire 3 sequential on_sse calls with matching composition.
        for step_id in 11..=13 {
            let s = step(step_id, 32, vec![out(7, "PREFILL", false)]);
            let _ = pctx.on_sse(&BatchForPredictor::default(), &s);

            let slot = pctx.ephemeral.lock().unwrap();
            let e = slot.as_ref().expect("ephemeral kept across pops");
            // After each on_sse, exactly one slot has been popped from
            // the head; anchor advanced to the current step_id.
            let expected_remaining = (13 - step_id) as usize;
            assert_eq!(
                e.buffer.slots.len(),
                expected_remaining,
                "after step {} we expect {} slots left",
                step_id,
                expected_remaining
            );
            assert_eq!(e.sse_anchor_step_id, step_id);
        }

        // After three pops the buffer is empty. The next on_sse with
        // engine-has-prefill would hit the `None` debug_assert in
        // Branch 1; we don't drive it here so release builds stay
        // quiet.
        let slot = pctx.ephemeral.lock().unwrap();
        assert_eq!(slot.as_ref().unwrap().buffer.slots.len(), 0);
    }

    // ---------- Step 5: incremental query via extend_with_candidate ----------

    /// `query(B)` after a baseline (`query(A) + on_admit(A)`) reuses
    /// the baseline slots and appends `B`'s tail.
    #[test]
    fn query_reuses_baseline_when_present() {
        let pctx = pctx();
        // 1. query(10, 800): from-scratch. Buffer = 2 slots.
        let _ = pctx.query(10, 800, &[], 0);
        // 2. on_admit(10, 800): promote to baseline. Savepoint=None.
        pctx.on_admit(10, 800, &[]);
        // 3. query(20, 500): clean-baseline recovery + extend.
        let _ = pctx.query(20, 500, &[], 0);

        let slot = pctx.ephemeral.lock().unwrap();
        let e = slot.as_ref().unwrap();
        assert_eq!(e.candidate_id, Some(20));
        assert_eq!(e.buffer.slots.len(), 4, "baseline (2) + candidate tail (2)");

        // First 2 slots are from the baseline (predict rid=10 only).
        assert_eq!(&e.buffer.slots[0].prefill_rids[..], &[10]);
        assert!(e.buffer.slots[0].decode_rids.is_empty());
        assert!(e.buffer.slots[1].prefill_rids.is_empty());
        assert_eq!(&e.buffer.slots[1].decode_rids[..], &[10]);

        // Last 2 slots are the candidate's tail.
        assert_eq!(&e.buffer.slots[2].prefill_rids[..], &[20]);
        assert_eq!(&e.buffer.slots[2].decode_rids[..], &[10]);
        assert!(e.buffer.slots[3].prefill_rids.is_empty());
        let mut s3_dec: Vec<u64> = e.buffer.slots[3].decode_rids.iter().copied().collect();
        s3_dec.sort();
        assert_eq!(s3_dec, vec![10, 20]);

        // Savepoint is set with split_idx = baseline size (2).
        let (split_idx, _) = e.savepoint.as_ref().expect("savepoint set after extend");
        assert_eq!(*split_idx, 2);
    }

    /// Fresh PCtx → `query(A)` falls through to the scratch rebuild
    /// path because no baseline exists. No savepoint is set.
    #[test]
    fn query_falls_back_to_scratch_when_no_baseline() {
        let pctx = pctx();
        // No prior query/on_admit — ephemeral starts None.
        let _ = pctx.query(7, 100, &[], 0);
        let slot = pctx.ephemeral.lock().unwrap();
        let e = slot.as_ref().unwrap();
        assert_eq!(e.candidate_id, Some(7));
        assert!(!e.buffer.slots.is_empty(), "scratch rebuild produces a buffer");
        assert!(e.savepoint.is_none(), "scratch rebuild has no savepoint");
    }

    /// `query(B)` recovers the baseline portion of a stale
    /// candidate-bound ephemeral via the savepoint when the previous
    /// candidate was queried but never admitted on this replica.
    #[test]
    fn query_recovers_baseline_via_savepoint() {
        let pctx = pctx();
        // Setup baseline first: query(10) + on_admit(10).
        let _ = pctx.query(10, 800, &[], 0);
        pctx.on_admit(10, 800, &[]);

        // First query (A=20) — extends from baseline. Savepoint set.
        let _ = pctx.query(20, 500, &[], 0);
        {
            let slot = pctx.ephemeral.lock().unwrap();
            let e = slot.as_ref().unwrap();
            assert_eq!(e.candidate_id, Some(20));
            assert!(e.savepoint.is_some(), "extend sets savepoint");
        }

        // Second query (B=30) — RECOVERY via savepoint (NOT on_admit'd A).
        let _ = pctx.query(30, 500, &[], 0);

        let slot = pctx.ephemeral.lock().unwrap();
        let e = slot.as_ref().unwrap();
        assert_eq!(e.candidate_id, Some(30));
        // 2 baseline slots + 2 B-tail slots = 4 total.
        assert_eq!(e.buffer.slots.len(), 4);

        // First 2 slots are recovered baseline (predict rid=10 only —
        // rid=20 was discarded by truncation, NOT carried over).
        assert_eq!(&e.buffer.slots[0].prefill_rids[..], &[10]);
        assert!(e.buffer.slots[0].decode_rids.is_empty());
        assert!(e.buffer.slots[1].prefill_rids.is_empty());
        assert_eq!(&e.buffer.slots[1].decode_rids[..], &[10]);

        // Tail is B's: B prefills, then in-decode with rid=10.
        assert_eq!(&e.buffer.slots[2].prefill_rids[..], &[30]);
        assert_eq!(&e.buffer.slots[2].decode_rids[..], &[10]);
        let mut s3_dec: Vec<u64> = e.buffer.slots[3].decode_rids.iter().copied().collect();
        s3_dec.sort();
        assert_eq!(s3_dec, vec![10, 30]);

        // New savepoint is set with idx = recovered baseline size.
        let (split_idx, _) = e.savepoint.as_ref().expect("new savepoint set");
        assert_eq!(*split_idx, 2);
    }

    /// `query(B)` after a partial F3-PASS pop reuses the
    /// (baseline_size − 1) remaining baseline slots before appending
    /// `B`'s tail.
    #[test]
    fn query_recovery_after_partial_pop() {
        let pctx = pctx();
        // Setup baseline: query(10) + on_admit(10). Baseline has 2 slots.
        let _ = pctx.query(10, 800, &[], 0);
        pctx.on_admit(10, 800, &[]);

        // First query (A=20) — extends, savepoint=(2, baseline_tail).
        let _ = pctx.query(20, 500, &[], 0);

        // Fire ONE F3-passing on_sse: matches baseline_slot0 [10]/[].
        let s = step(1, 800, vec![out(10, "PREFILL", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s);

        // After 1 pop: 4 - 1 = 3 slots remain. savepoint idx = 2 - 1 = 1.
        {
            let slot = pctx.ephemeral.lock().unwrap();
            let e = slot.as_ref().unwrap();
            assert_eq!(e.buffer.slots.len(), 3);
            let (idx, _) = e.savepoint.as_ref().unwrap();
            assert_eq!(*idx, 1);
        }

        // Second query (B=30) — savepoint recovery with idx=1.
        let _ = pctx.query(30, 500, &[], 0);

        let slot = pctx.ephemeral.lock().unwrap();
        let e = slot.as_ref().unwrap();
        assert_eq!(e.candidate_id, Some(30));
        // Recovered (baseline_size - 1) = 1 baseline slot + B's tail.
        // B's tail from baseline_tail={10@801}: PREFILL(30,500) + DECODE(both).
        assert_eq!(e.buffer.slots.len(), 3);

        // First slot is the recovered baseline_slot1 (after 1 pop):
        // baseline_slot1 was rid=10 in DECODE — so it's []/[10].
        assert!(e.buffer.slots[0].prefill_rids.is_empty());
        assert_eq!(&e.buffer.slots[0].decode_rids[..], &[10]);

        // Tail's first slot: rid=30 PREFILL, rid=10 DECODE.
        assert_eq!(&e.buffer.slots[1].prefill_rids[..], &[30]);
        assert_eq!(&e.buffer.slots[1].decode_rids[..], &[10]);
    }

    /// After 2 F3-passing on_sse calls consume a 2-slot baseline,
    /// the savepoint is dropped (baseline portion fully consumed).
    #[test]
    fn savepoint_cleared_when_baseline_fully_consumed() {
        let pctx = pctx();
        // Setup baseline (2 slots): query(10) + on_admit(10).
        let _ = pctx.query(10, 800, &[], 0);
        pctx.on_admit(10, 800, &[]);

        // query(A=20) — extends with savepoint=(2, baseline_tail).
        let _ = pctx.query(20, 500, &[], 0);

        // First F3-passing on_sse: matches baseline_slot0 [10]/[].
        let s1 = step(1, 800, vec![out(10, "PREFILL", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s1);

        // Confirm savepoint idx is 1 (still recoverable).
        {
            let slot = pctx.ephemeral.lock().unwrap();
            let (idx, _) = slot
                .as_ref()
                .unwrap()
                .savepoint
                .as_ref()
                .expect("savepoint still present after 1 pop");
            assert_eq!(*idx, 1);
        }

        // Second F3-passing on_sse: matches baseline_slot1 []/[10].
        // Use prefill_tokens=1 to satisfy `engine_has_prefill` while
        // keeping the actual PREFILL composition empty (matching the
        // slot's `prefill_rids = []`). Semantically contrived but
        // exercises the savepoint clear-on-zero logic.
        let s2 = step(2, 1, vec![out(10, "DECODE", false)]);
        let _ = pctx.on_sse(&BatchForPredictor::default(), &s2);

        // Savepoint should be cleared (baseline portion fully consumed).
        let slot = pctx.ephemeral.lock().unwrap();
        let e = slot.as_ref().expect("ephemeral kept across pops");
        assert!(
            e.savepoint.is_none(),
            "savepoint must be cleared when baseline portion is fully consumed"
        );
    }

    /// `on_admit` promote arm clears any existing savepoint — the
    /// whole buffer becomes the new baseline; no separate savepoint
    /// is needed.
    #[test]
    fn promote_clears_savepoint() {
        let pctx = pctx();
        // Setup an initial baseline so the next query(A) sets a savepoint.
        let _ = pctx.query(10, 800, &[], 0);
        pctx.on_admit(10, 800, &[]);

        // query(A=20) — extends, savepoint set.
        let _ = pctx.query(20, 500, &[], 0);
        {
            let slot = pctx.ephemeral.lock().unwrap();
            assert!(
                slot.as_ref().unwrap().savepoint.is_some(),
                "savepoint expected after extend"
            );
        }

        // on_admit(A=20) — promote. Savepoint must clear even though
        // buffer is non-empty.
        pctx.on_admit(20, 500, &[]);

        let slot = pctx.ephemeral.lock().unwrap();
        let e = slot.as_ref().unwrap();
        assert_eq!(e.candidate_id, None, "promote sets cid to None");
        assert!(
            !e.buffer.slots.is_empty(),
            "buffer is non-empty (whole buffer is baseline)"
        );
        assert!(e.savepoint.is_none(), "promote must clear savepoint");
    }

    /// `extend_with_candidate` (exercised via the public `query` path)
    /// appends the new candidate's tail to the baseline buffer's
    /// existing slots without disturbing them.
    #[test]
    fn extend_with_candidate_appends_to_baseline_buffer_slots() {
        let pctx = pctx();

        // Build baseline and capture its slot composition.
        let _ = pctx.query(10, 800, &[], 0);
        pctx.on_admit(10, 800, &[]);
        let baseline_compositions: Vec<(Vec<u64>, Vec<u64>)> = {
            let slot = pctx.ephemeral.lock().unwrap();
            let buf = &slot.as_ref().unwrap().buffer;
            buf.slots
                .iter()
                .map(|s| {
                    (
                        s.prefill_rids.iter().copied().collect(),
                        s.decode_rids.iter().copied().collect(),
                    )
                })
                .collect()
        };
        assert_eq!(baseline_compositions.len(), 2);

        // Extend with a new candidate via `query`.
        let _ = pctx.query(20, 500, &[], 0);
        let slot = pctx.ephemeral.lock().unwrap();
        let buf = &slot.as_ref().unwrap().buffer;
        assert!(
            buf.slots.len() > baseline_compositions.len(),
            "buffer must be extended beyond baseline"
        );
        // Verify the leading slots match the captured baseline verbatim.
        for (i, (pf, dc)) in baseline_compositions.iter().enumerate() {
            let s = &buf.slots[i];
            let s_pf: Vec<u64> = s.prefill_rids.iter().copied().collect();
            let s_dc: Vec<u64> = s.decode_rids.iter().copied().collect();
            assert_eq!(&s_pf, pf, "baseline slot {} prefill_rids preserved", i);
            assert_eq!(&s_dc, dc, "baseline slot {} decode_rids preserved", i);
        }
        // The first appended slot must include the new candidate.
        let tail_first = &buf.slots[baseline_compositions.len()];
        assert!(
            tail_first.prefill_rids.iter().any(|&r| r == 20),
            "candidate's tail must include rid=20 in PREFILL"
        );
    }
}
