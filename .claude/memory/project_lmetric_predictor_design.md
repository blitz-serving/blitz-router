---
name: lmetric latency-predictor design (stable)
description: Stable architecture of blitz-router's predictor subsystem (the latency simulator). Points into docs/predictor/ for detail; flags non-code-derivable gotchas.
type: project
---

The predictor subsystem (a.k.a. the latency simulator, feature-gated by
`simulator`) sits in MIDDLE as a **service-sidecar to `PolicyRunner`**.
Its design has converged; this entry is a pointer to the docs plus the
non-obvious facts a future agent should keep in mind.

## Source of truth

Code: `router/src/scheduler/simulator/`. Design docs:

- `docs/predictor/README.md` — index + one-screen summary.
- `docs/predictor/three-layer-architecture.md` — structural (L1 / L2 / L3, savepoint mechanics, naming rationale).
- `docs/predictor/behavior.md` — behavioral (the three APIs, exact state transitions, mermaid diagrams, worked example).
- `docs/predictor/least-ttft-q-test-results.md` — first end-to-end run on Qwen2.5-7B × 8.
- `docs/architecture/latency-simulator.md` — system-level role (subsection).

The verbose design used to live as a single file under `docs/architecture/`;
it was split: the system-level "what is this" stays in `architecture/`,
the deep design lives in `docs/predictor/`. Don't add design content
to the architecture file; deep edits go into `docs/predictor/`.

## Stable architecture (one paragraph)

Per replica, a `PCtx` privately holds three layers of state:
**L1 = world model** (`mirror: IncrementalMirror` + `sched: SchedSnapshot`),
**L2 = cost oracle** (`regressor: Box<dyn TrainedPredictor>`, in production
`RegressionalPredictor<VidurRfPredictor, LinregCorrector>`), and
**L3 = DES rollout** (`ephemeral: Option<EphemeralRollout>` carrying
`buffer: RolloutBuffer` whose `slots: VecDeque<RolloutSlot>` satisfy
the invariant *`slots.front()` is the next engine step relative to
`sse_anchor_step_id`*, plus `tail_sched` and `savepoint`).
Three APIs: `on_sse` (silent wiring from BACK; pop-on-keep on F3 PASS),
`on_admit` (silent wiring from `PolicyRunner`; promote-or-drop with
savepoint clear on promote), `query` (active call from a
prediction-based policy; three-way recovery — clean baseline / savepoint
truncate-and-restore / scratch rebuild). Output of `query` is
`RolloutGist { ttft_ms, chunked_prefill_steps, in_decode_tbt_ms }`.

## L2 trait split — naming pinned

```rust
trait Predictor    { fn predict(&self, batch) -> f32; }                          // offline
trait Corrector    { fn correct(&self, raw); fn calibrate(&mut, raw, actual); }  // online
struct RegressionalPredictor<P: Predictor + ?Sized, C: Corrector> { inner: Arc<P>, corrector: C }
struct LinregCorrector { … }   // affine w0·raw+w1 with SGD
struct NullCorrector;          // passthrough
```

Don't propose alternative names: `Calibrated`, `Regressor`,
`LinregRegressor` were all rejected as either vague past-participle or
"regression regressor" sleep-talk.

## Things the agent needs to know that aren't in the docs

### 1. Hand-written `Policy` bookkeeping

**Why:** A hand-written `impl Policy` (i.e. one that doesn't go through
the `policy!` proc macro) MUST, after picking a replica, do the
following on the chosen replica's `ScheduleContext`:

```rust
let mut g = all_sctx[chosen].lock().await;
let current_epoch = g.block_hash.epoch();
let hit_nblks = g.block_hash.get(hashes);
entry.block_hash_state.set_pred_block_hits(hit_nblks);   // clears NONE_SENTINEL
entry.block_hash_state.set_decision_epoch(current_epoch);
let new_ntkns = entry.request.input_tokens.len()
    .saturating_sub(hit_nblks * entry.block_hash_state.get_block_size());
g.lmetric += LMetricInc { bs_inc: 1, waiting_reqs_inc: 1,
                          prefill_tokens_inc: new_ntkns,
                          all_tokens_inc: entry.request.input_tokens.len() };
```

DSL policies get this for free via
`policies::dsl_runtime::apply_default_after`. Skipping it makes the
engine completion path's `BlockHashState::set_real_token_hits_get_diff`
fire its `pred_hit_nblks & NONE_SENTINEL == 0` assertion → router
panics on the first request that completes prefill.

**How to apply:** copy the pattern from `router/src/scheduler/policies/least_ttft.rs`
(the canonical example of a hand-written predictor-driven policy).

### 2. NullCorrector behaviour without a CLI flag

**Why:** there's no `--simulator-corrector` flag. To get NullCorrector
behaviour from the production wiring, pass `--simulator-learning-rate 0.0`.
`LinregCorrector` initialises at `(w0=1.0, w1=0.0)` so the first
`correct(raw)` returns `raw`; with `learning_rate=0.0`, `calibrate` is
a no-op (`weight += 0 * error * raw`), so weights stay at `(1.0, 0.0)`
forever and `corrected = raw` always.

**How to apply:** any test that wants L2's `(predicted, actual)`
metrics without the SGD loop interfering — e.g. the
`least-ttft-q-test-results.md` run.

### 3. Llama-2-7B Vidur grids vs Qwen2.5-7B traffic

**Why:** the Llama-2-7B stopgap grids miss most queried `(kv_cache,
flops)` cells under Qwen2.5 traffic shape. The first 20-min run on
Qwen2.5-7B logged ~2.17M `missing prediction for op=…` warnings, with
the L2 inner predictor falling back to per-op constants. With
NullCorrector active, no calibration absorbed the residual. The
policy still ran end-to-end (per-replica TTFT differences came from L1
state, not L2 quality), but L2 was effectively a constant for grid
misses.

**How to apply:** when interpreting predictor metrics from a run that
used the Llama-2 stopgap, downweight L2-quality conclusions. For
sweeps that need real predictions, run the full Vidur profiling
pipeline against Qwen2.5-7B first.

### 4. Cross-sidecar ordering in `colocation.rs`

**Why:** historically `simulator::on_sse` fired BEFORE the SCtx
prefix-cache update, opening a TOCTOU window where a concurrent
`query` could see (mirror post-step, SCtx pre-step). Fixed
2026-05-09: `simulator::on_sse` now fires AFTER `drop(sctx)` and the
prefix-cache writes. Any future change to the colocation event loop
must preserve this ordering (or the bug returns).

**How to apply:** if reading `colocation.rs` during a refactor, look
for `crate::scheduler::simulator::on_sse(replica_index, &m)` and check
that it sits BELOW the SCtx prefix-cache `insert` calls.

## Load-bearing code facts (file:line)

- `router/src/scheduler/simulator/pctx.rs` — `PCtx` struct + the
  three triggers (`query`, `on_admit`, `on_sse`). Lock order is
  `sched → mirror → ephemeral`.
- `router/src/scheduler/simulator/predictor.rs` — the L2 traits +
  concrete correctors.
- `router/src/scheduler/simulator/mirror.rs` — `IncrementalMirror`
  (L1 prefix-hash overlay).
- `router/src/scheduler/simulator/sched.rs` — `SchedSnapshot` +
  `ReqProgress` (L1 per-request progress).
- `router/src/scheduler/simulator/rollout.rs` — `RolloutBuffer` (slots
  is `VecDeque<RolloutSlot>`), `RolloutGist`.
- `router/src/scheduler/policies/least_ttft.rs` — canonical
  hand-written predictor-driven policy.
- `router/src/scheduler/policies/dsl_runtime.rs::apply_default_after` —
  the post-decision bookkeeping that DSL policies get for free
  (reference for hand-written policies).
- `router/src/engine/colocation.rs` — `simulator::on_sse` call site
  (post SCtx prefix-cache write; see §4 above).

## Piggyback metrics (Prometheus histograms)

- `simulator_predicted_ms`, `simulator_actual_ms` — per-step.
- `simulator_signed_error_ms`, `simulator_abs_error_ms`,
  `simulator_relative_error` — derived per-step.

These are emitted from `simulator::on_sse`'s return value; available
without any policy consuming `query`.

## What's intentionally NOT in this memory

The earlier version of this memory tracked phase-by-phase
implementation history (Phase 1/2/3, Group A/B/C decisions, the v1/v1.5
split, the L2 "refactor proposal", the TOCTOU as accepted-and-deferred,
and a specific 2026-05-02 piggyback verification numbers). All of that
either landed and is now visible in code/docs, or is superseded by the
stable design captured above. If you need provenance, `git log` on
`router/src/scheduler/simulator/` and the design docs is the
authoritative record.
