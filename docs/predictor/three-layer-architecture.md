# Simulator three-layer architecture (`PCtx` L1 / L2 / L3)

> Companion to [`../architecture/latency-simulator.md`](../architecture/latency-simulator.md).
> That file describes the simulator as a **service-sidecar to PolicyRunner**;
> this file zooms in on the **per-replica `PCtx`** that lives inside the
> simulator and explains the three layers (L1 / L2 / L3) of state it owns,
> *why each layer is shaped the way it is*, and how the layers reconcile
> with the rest of the system.

> Disambiguation. The system as a whole has a three-layer architecture
> (FRONT / MIDDLE / BACK) — see [`../architecture/README.md`](../architecture/README.md) §2.
> The simulator's three-layer architecture described here is **a different
> three-layer split**, internal to one MIDDLE subsystem (the simulator
> service-sidecar). Do not conflate the two.

## 0. The simulator at a glance — three APIs, three layers

The `simulator` module exposes exactly three module-level functions; none
of them takes any of `PCtx`'s internal types.

| Function    | Direction        | Caller                                 | Frequency |
|-------------|------------------|----------------------------------------|-----------|
| `on_sse`    | silent wiring ←  | BACK `completion_event_loop`           | one per engine forward step |
| `on_admit`  | silent wiring ←  | MIDDLE `policies::PolicyRunner::queue_task` | one per admission |
| `query`     | active call →    | a prediction-based policy              | one per scoring decision |

Each replica gets a `PCtx`. `PCtx` privately holds three layers of state:

```
                                ┌──────────────────────────────┐
                                │  PCtx (one per replica)       │
   on_admit ───silent wiring──→ │                              │
   on_sse   ───silent wiring──→ │   L1  world model            │
                                │   L2  cost oracle            │
   query    ───active call ───→ │   L3  DES rollout            │
                                │                              │
                                │   aux: last_sse_step_id, …   │
                                └──────────────────────────────┘
                                            ↓
                                       RolloutGist
```

- **L1 — world model.** What the simulator believes about the engine
  *right now* (which requests are in flight, what prefix blocks each
  has, where each request sits in its prefill/decode lifecycle).
- **L2 — cost oracle.** Given a hypothetical batch composition, return
  the predicted per-step latency in milliseconds.
- **L3 — DES rollout.** Roll the world model forward step-by-step,
  asking the cost oracle at each step, until a stop condition; project
  the resulting trajectory into `RolloutGist`.

L1 and L2 are *capabilities*; L3 is the *consumer* that uses both to
answer "if I admit candidate X, what does the next chunk of life look
like?" The single shape that escapes `PCtx` is `RolloutGist`.

## 1. L1 — world model (`mirror + sched`)

L1 is **one concept implemented as two data structures** because the
data they hold has different shapes. Both are written by the same two
events (`on_admit` and `on_sse`) and consumed by the same reader (L3
at `query` time).

| sub-struct | type | tracks                                    | written by                  |
|------------|------|--------------------------------------------|------------------------------|
| `mirror`   | `IncrementalMirror { tree: RadixTreeReqIdHash, by_request: HashMap<rid, Vec<u64>> }` | per-block prefix hashes of in-flight requests | `on_admit` (insert), `on_sse → apply_sse` (evict / finish / abort / preempt) |
| `sched`    | `SchedSnapshot { waiting: VecDeque<ReqProgress>, running: HashMap<rid, ReqProgress> }` | per-request lifecycle (input_length, processed_tokens) | `on_admit` (push back of `waiting`), `on_sse → sync` (PREFILL chunk advance / DECODE +1 / finish / abort / preempt) |

A radix trie is the wrong shape for scalar progress; a flat per-request
snapshot is the wrong shape for prefix dedup. So L1 is split, with
`pctx.rs:80-82`'s comment making the rationale explicit:

> *"Distinct from L1 mirror in that it carries lifecycle progress but
> **not** prefix hashes."*

### 1.1 `SchedSnapshot` vs `ScheduleContext` — same world, different shapes

`SchedSnapshot` and the non-`block_hash` part of `ScheduleContext`
(i.e. `LMetric`) describe **the same engine state** but at different
granularities for different consumers:

| | `SchedSnapshot` (PCtx, L1) | `LMetric` (SCtx) |
|---|---|---|
| Granularity | per-request: `running: HashMap<rid, ReqProgress>` | aggregate scalar: `bs`, `waiting_reqs`, `prefill_tokens`, `tbt`, `tpot` |
| Per-request progress | yes (`processed_tokens` per rid) | no |
| Latency EMAs (TBT, TPOT) | no | yes |
| Consumer | L3 DES (needs to replay each request slot-by-slot) | scheduling policies (need scalar load metrics for scoring) |
| Writer | `simulator::on_sse → SchedSnapshot::sync` | `back::completion_event_loop → LMetric -= delta` |

So `SchedSnapshot ≈ ScheduleContext \ block_hash` is **a useful
intuition but not literally true**: the *scope* matches (both are
"engine-state-mirror minus prefix hashes"), but the *shape* differs
(per-request granular vs. aggregate scalar). They are derived from the
same SSE stream, kept in different forms because their respective
consumers want different views.

The `block_hash` part of `ScheduleContext` mirrors `IncrementalMirror`
in a similar way: same data-structure infrastructure (radix trie),
different concern (`block_hash` tracks *what the engine actually has*
post-SSE; `IncrementalMirror` tracks *what the simulator believes
about in-flight requests* including the speculative window).

### 1.2 Hit composition rule (A2) and the cross-sidecar window

When L3 needs the prefix-cache hit for a candidate, it composes both
mirrors via the **A2 max-merge rule**:

```
hit(hashes) = max(SCtx.block_hash.get(hashes),
                  PCtx.mirror.prefix_match(hashes))
```

The max-merge tolerates the two views drifting (SCtx is authoritative
post-step; PCtx mirror is speculative pre-step). False positives on
speculative entries trigger redo, not divergence (locked as A2).

**Race that used to exist — now fixed.** The completion event loop
*used to* fire `simulator::on_sse` *before* it acquired the SCtx lock
for the step's prefix-cache update. In the window between
`simulator::on_sse` and the SCtx update, the mirror was post-step but
SCtx was pre-step. A `query()` from another task during this window
could underestimate hits in the narrow case where:

1. A request `X` finishes in step `s` (so `mirror.apply_sse` drops `X`'s hashes).
2. The same step `s` emits new `new_block_hashes_ids` for `X` (e.g. `X`
   completes prefill and stops at the first decode token).
3. A sibling `Y` querying with `X`'s prefix arrives during the window.

Then `mirror.get(Y_prefix) = 0` (X dropped) and `SCtx.get(Y_prefix) = 0`
(blocks not yet inserted) → `max(0, 0) = 0` underestimates.

**Fix landed (2026-05-09).** `simulator::on_sse(replica_index, &m)` was
relocated in `colocation.rs` from immediately after the step receive
(pre-SCtx-update) to after `drop(sctx)` (post-SCtx prefix-cache
update). The window is closed: by the time `simulator::on_sse` fires,
SCtx already reflects `m.new_block_hashes_ids` and the max-merge sees a
consistent pair of views. No additional locking required; on_sse
remains independent of the SCtx critical section.

## 2. L2 — cost oracle (`offline P + online C`)

L2 is **two concerns split across two traits**, each with its own
generic parameter:

| concern | role | trait |
|---|---|---|
| **(a) offline `P`** | feature → ms scalar via grid lookup; trained offline (Vidur RandomForest, llm-d XGBoost, …) | `Predictor` |
| **(b) online `C`** | calibrate the offline output against observed actuals; absorb hardware/model/driver bias the offline grids do not see | `Corrector` |

Today's implementation (as of 2026-05-09):

```rust
// As-built (predictor.rs):
pub trait Predictor: Send + Sync {
    fn predict(&self, batch: &BatchForPredictor) -> f32;
}
pub trait TrainedPredictor: Send + Sync {
    fn predict(&self, batch: &BatchForPredictor) -> f32;
    fn calibrate(&mut self, batch: &BatchForPredictor, actual: f32);
}
pub trait Corrector: Send + Sync {
    fn correct(&self, raw: f32) -> f32;                        // applied at predict time
    fn calibrate(&mut self, raw: f32, actual: f32);            // applied at observe time
}
pub struct RegressionalPredictor<P: Predictor + ?Sized, C: Corrector> {
    inner: Arc<P>,
    corrector: C,
}
impl<P, C> TrainedPredictor for RegressionalPredictor<P, C> where … { … }

// Concrete correctors:
pub struct LinregCorrector { weight: (f32, f32), learning_rate: f32, … }  // affine SGD
pub struct NullCorrector;   // correct(raw) = raw; calibrate(_, _) = no-op
```

PCtx field stays `regressor: Mutex<Box<dyn TrainedPredictor>>` — the
trait-object surface is unchanged; the generics live one level
deeper (the concrete `RegressionalPredictor<VidurRfPredictor, LinregCorrector>`
that production wires up).

### 2.1 Naming rationale (pinned)

- `Corrector` (not `Regressor`) — the role is *adjusting one prediction
  toward an observation*, not "performing a regression". Calling the
  trait `Regressor` and its SGD impl `LinregRegressor` reads as
  "regression regressor", which is sleep-talk.
- `RegressionalPredictor<P, C>` (not `Calibrated`) — adjective form
  ("a predictor that has a regression correction on top of it") avoids
  the vague past-participle `Calibrated` ("calibrated *what*?") while
  still surfacing the "regression" terminology as a property rather
  than a noun-noun compound.
- `LinregCorrector` (not `LinregRegressor`) — `Linreg` describes
  *how* it corrects (affine `w0·raw + w1` with SGD); `Corrector`
  describes the *role*. No echo.
- `NullCorrector` (not `NoRegressor`) — passthrough; matches the
  `Linreg`/`Null` pair convention used elsewhere for null-object
  variants.

### 2.2 Why a `NullCorrector`

- **Bypass switch.** Run with raw offline predictions for diagnosis
  (is the linreg correction overfitting? is the offline grid finally
  good enough that correction adds noise?).
- **A/B comparison.** Toggle `LinregCorrector` ↔ `NullCorrector` to
  measure the actual contribution of online correction to prediction
  quality. The piggyback metrics (`simulator_predicted_ms`,
  `simulator_actual_ms`) make this directly observable.
- **Future-proofs other strategies.** Kalman filter, EMA, llm-d-style
  sidecar retraining — all plug in as new `impl Corrector` without
  touching the offline model wrapper.

## 3. L3 — DES rollout (essence vs. wrapper)

The essence of L3, in one sentence:

> *Take L1 as the world model and L2 as the cost oracle, walk the
> world forward one slot at a time calling the oracle per slot, and
> project the resulting trajectory into `RolloutGist`.*

A slot is one engine forward step. Per-slot composition: each in-flight
DECODE request takes one token; remaining `token_budget` goes to chunked
prefill (continuing prefillers first, then waiting front in FCFS order;
locked as A4). Stop condition: candidate completes prefill plus one
in-decode slot (locked as A3, with a 256-slot defensive ceiling). The
final shape:

```rust
pub struct RolloutGist {
    pub ttft_ms: Option<f32>,
    pub chunked_prefill_steps: Option<usize>,
    pub in_decode_tbt_ms: Option<f32>,
}
```

That is the entirety of L3's essence: a viable trajectory view → a
gist.

### 3.1 `EphemeralRollout` is a design choice, not essence

The on-disk shape of L3 is:

```rust
struct EphemeralRollout {
    candidate_id:        Option<u64>,                  // None ⇔ baseline (post-promotion)
    buffer:              RolloutBuffer,
    sse_anchor_step_id:  u64,
    tail_sched:          SchedSnapshot,                // state at the END of buffer (used by next query's extend)
    savepoint:           Option<(usize, SchedSnapshot)>, // (split_idx, baseline_tail) — see below
}
ephemeral: Mutex<Option<EphemeralRollout>>,
```

The minimum representation that delivers the same essence would be:

```rust
ephemeral_buffer:    Mutex<Option<RolloutBuffer>>,
ephemeral_candidate: AtomicI64,   // -1 = baseline / None
ephemeral_anchor:    AtomicU64,
```

The chosen `EphemeralRollout` wrapper is a **bundling design choice**,
chosen for three reasons:

1. **Atomic dropping.** `*ephemeral.lock() = None` discards
   `(buffer, candidate_id, anchor, tail_sched, savepoint)` together. With
   separate fields you have to reason about partial-drop races (buffer
   cleared but anchor still pointing at it). One wrapper → one drop,
   invariant for free.
2. **Reads are joint.** Every L3 consumer (`query`'s recovery probe,
   `on_sse`'s 4-branch maintenance) reads multiple fields together
   (`Some(rollout) if rollout.candidate_id == … && rollout.sse_anchor_step_id == self.last_sse_step_id.load()`,
   plus `rollout.savepoint` for the recovery path). Co-locating them
   under one lock matches the access pattern.
3. **Carries the savepoint cleanly.** `tail_sched` and `savepoint`
   exist precisely because `query` reuses cached state across calls
   (the `commit_prediction`/`uncommit_prediction`-style incremental
   path). Without the wrapper, the savepoint's fields would be peer to
   PCtx's other fields — easy to forget when adding a new operation.

The cost is one extra struct definition and one indirection. The
benefit is that the L3 invariants (buffer + candidate_id + anchor
mutually consistent; savepoint set iff cid is Some and recoverable
baseline exists) are **type-enforced** rather than convention-enforced.

#### `tail_sched` vs `savepoint` — distinct purposes

- **`tail_sched: SchedSnapshot`** is the simulated state at the END of
  the entire `buffer` (post-extension if the buffer is candidate-bound).
  It's used by the *next* `query` to continue the simulation forward
  when reusing this buffer as a baseline (after `on_admit` promotes
  it). Always present (it's not an `Option`).
- **`savepoint: Option<(usize, SchedSnapshot)>`** is set by
  `extend_with_candidate` BEFORE extending. The tuple is
  `(baseline_split_idx, baseline_tail_sched)`:
  - `baseline_split_idx`: `slots[..idx]` are the baseline portion;
    `slots[idx..]` are the candidate's tail that was just appended.
  - `baseline_tail_sched`: sched state at the END of the baseline
    portion (i.e. the input to the candidate-tail simulation).
  - `Some(_)` iff `cid == Some(_)` AND a recoverable baseline portion
    exists. Cleared by `on_admit` promote (the whole buffer becomes
    baseline; no separate savepoint needed) and by `on_sse` Branch 1
    pop-on-keep when `split_idx` hits 0 (baseline portion fully
    consumed).
  - Always `None` when `cid == None` (a clean baseline has no
    savepoint — the whole buffer IS the baseline).

The savepoint enables the next `query` to recover from a stale
candidate-bound buffer (when the candidate was queried on this replica
but admitted elsewhere): truncate to `slots[..split_idx]`, restore
`baseline_tail_sched`, then extend with the new candidate. Without it,
`query` would discard the entire stale buffer and rebuild from scratch
— which still works correctly but loses the baseline portion's reuse.

### 3.2 Drop semantics

L3 maintenance happens inside `on_sse` after L2 calibration and L1
absorption. The 4-branch table:

| Engine state on SSE | L3 candidate state | Action |
|---|---|---|
| prefill in progress | `Some(_)`, F3 PASS         | **pop_front** slot[0]; advance anchor; decrement savepoint idx (clear if 0) |
| prefill in progress | `Some(_)`, F3 FAIL (drift) | drop entire ephemeral |
| prefill in progress | `None`        | trace-warn (invariant violation; rebuild on next `query`) |
| decode-only         | `Some(_)`     | drop (predictions vacuously stale)  |
| decode-only         | `None`        | no-op (legal empty)                  |

Plus one promote-or-drop branch on `on_admit(rid)`:
- `ephemeral.candidate_id == Some(rid)` → **promote**: `candidate_id := None`,
  `savepoint := None` (whole buffer becomes baseline; no separate savepoint
  needed).
- `ephemeral.candidate_id == None` (baseline) → **drop**: this admission
  invalidates the cached baseline projection.
- `ephemeral.candidate_id == Some(other≠rid)` → **debug_assert** (protocol
  violation; safety-net drop in release).
- `ephemeral == None` → no-op.

**Invariant:** if engine has prefill work in progress (`prefill_tokens > 0`
or any output has `state == "PREFILL"`), L3 must be non-empty. Empty
buffer is legal iff engine is decode-only. Violation logs at
`target=simulator`.

L3 reuse is implemented via the `tail_sched` + `savepoint` mechanics
in `EphemeralRollout`: `query`'s 3-way recovery (clean baseline → extend
in place; stale candidate-bound with savepoint → truncate to baseline
portion + extend; otherwise → rebuild from scratch). F3 cross-check on
`on_sse` Branch 1 is the per-step invalidation trigger.
$$
## 4. Aux state — what is *not* a layer, and why

```rust
pub struct PCtx {
    regressor:        Mutex<Box<dyn TrainedPredictor>>,   // L2
    mirror:           Mutex<IncrementalMirror>,            // L1.mirror
    ephemeral:        Mutex<Option<EphemeralRollout>>,    // L3
    sched:            Mutex<SchedSnapshot>,               // L1.sched
    last_sse_step_id: AtomicU64,                          // ← aux
    block_size:       usize,                              // ← aux
}
```

Two aux fields, neither a layer:

- **`last_sse_step_id: AtomicU64`** — the **freshness clock** that
  connects all three layers. Lives outside any layer because:
  - L3 may be empty (`None`); the anchor must remain readable so the
    *next* `query` can stamp the new buffer.
  - It is the *coordination primitive* between the SSE writer and L3's
    stale-check; not state of any layer, but the clock that lets L3
    *talk about* its own freshness.
  - `Atomic` (not `Mutex`) because writers advance it monotonically
    inside `on_sse`'s already-serialized critical section, and readers
    just want a recent value with no locking.
- **`block_size: usize`** — pure config constant. Cached on `PCtx`
  (and duplicated on `SimulatorRuntime`) so `query` doesn't have to
  reach back through the runtime for unit conversion (block-count ↔
  token-count). Could equally be a `const` if it weren't engine-
  configurable.

If you prefer a tidier mental model, fold `last_sse_step_id` and
`block_size` into "the runtime constants and clocks PCtx needs" rather
than a distinct concept: they are infrastructure, not layered state.

## 5. UML chain (single-screen reference)

```
SIMULATOR: OnceLock<SimulatorRuntime>
   │
   └── SimulatorRuntime { pctxs: Vec<Arc<PCtx>>, block_size }
        │
        └── PCtx (one per replica)
             │
             ├── L1: world model
             │     ├── mirror: Mutex<IncrementalMirror>
             │     │             ├── tree: RadixTreeReqIdHash
             │     │             └── by_request: HashMap<rid, Vec<u64>>
             │     └── sched:  Mutex<SchedSnapshot>
             │                   ├── waiting: VecDeque<ReqProgress>
             │                   └── running: HashMap<rid, ReqProgress>
             │
             ├── L2: cost oracle
             │     └── regressor: Mutex<Box<dyn TrainedPredictor>>
             │           └── RegressionalPredictor<P, C>
             │                 ├── inner:     Arc<P: Predictor>     ← (a) offline
             │                 └── corrector: C: Corrector           ← (b) online
             │                     ├── LinregCorrector (production: w0·raw + w1, SGD)
             │                     └── NullCorrector  (passthrough: correct = raw, calibrate = no-op)
             │
             ├── L3: DES rollout
             │     └── ephemeral: Mutex<Option<EphemeralRollout>>
             │                      ├── candidate_id:        Option<u64>
             │                      ├── buffer:              RolloutBuffer (slots: VecDeque<RolloutSlot>)
             │                      ├── sse_anchor_step_id:  u64
             │                      ├── tail_sched:          SchedSnapshot
             │                      └── savepoint:           Option<(usize, SchedSnapshot)>
             │
             └── aux
                  ├── last_sse_step_id: AtomicU64
                  └── block_size: usize
```

Lock order, sole writers, and lifetimes:

| field | written by | read by | drop trigger |
|---|---|---|---|
| `mirror`  | `on_admit` (insert), `on_sse → apply_sse` | L3 at `query` | per-entry, by `apply_sse` |
| `sched`   | `on_admit` (push), `on_sse → sync`        | L3 at `query` | per-request, by `sync` |
| `regressor` | `on_sse → calibrate` (SGD update)        | L3 at `query` (per slot) | never |
| `ephemeral` | `query` (build), `on_admit` (promote/drop), `on_sse` (4-branch maintenance) | next `query` | aggressively per design (see §3.2) |
| `last_sse_step_id` | `on_sse` (monotonic bump) | L3 at `query` (anchor stamp), `on_sse` (stale check) | never |

Lock order across the layer mutexes (set by PCtx): **`sched → mirror → ephemeral`**.
The regressor mutex is independent (L2 has no cross-dependency on L1
or L3 state) and is taken by both `on_sse` (calibrate) and `query` (per-
slot predict).

## 6. Glossary of "what each call actually does"

```
on_admit(rid, input_length, hashes):
  sched.lock()      → admit(ReqProgress::new(rid, input_length))
  mirror.lock()     → insert_request(rid, hashes)
  ephemeral.lock()  → 4-arm match:
                       cid == Some(rid) → promote (cid := None, savepoint := None)
                       cid == None       → drop (baseline doesn't account for this admission)
                       cid == Some(other)→ debug_assert (protocol violation), drop in release
                       None              → no-op
  // last_sse_step_id untouched (no engine event)

on_sse(batch, m):
  // L2 first: capture predicted-vs-actual against the batch as the
  // engine saw it.
  regressor.lock()  → predict(batch); calibrate(batch, m.latency); return (predicted, actual)
  // Then L1: absorb engine state transitions.
  sched.lock()      → sync(m)
  mirror.lock()     → apply_sse(m, &sched)
  // Then aux + L3 maintenance.
  last_sse_step_id.store(m.step_id)
  ephemeral.lock()  → 4-branch maintenance:
                       (Some, prefill, F3 PASS) → pop_front; advance anchor; decrement savepoint idx (clear if 0)
                       (Some, prefill, F3 FAIL) → drop ephemeral
                       (None, prefill)           → trace-warn (invariant violation; rebuild on next query)
                       (Some, decode-only)       → drop (vacuously stale)
                       (None, decode-only)       → no-op
  // Returns (predicted, actual) for Prometheus.

query(candidate_id, input_length, candidate_hashes, sctx_prefix_hits):
  // 3-way recovery from the existing ephemeral:
  //   cid == None → use buffer + tail_sched directly (clean baseline)
  //   cid == Some(_), savepoint=Some(idx,saved_tail), idx>0
  //                  → truncate buffer.slots to slots[..idx], use saved_tail
  //   otherwise     → fall through to scratch rebuild
  // Then either:
  //   extend_with_candidate(buffer, baseline_tail, candidate_id, …)
  //     → snapshot savepoint; clone baseline_tail; admit candidate;
  //       continue DES loop until candidate's first DECODE
  //   or run_schedule_loop_from_scratch(candidate_id, …)
  //     → clone live sched; admit candidate; run DES loop
  // Stamp the new EphemeralRollout with last_sse_step_id.load() and
  // tail_sched / savepoint from the chosen path.
  // Project: RolloutGist via buffer.gist().
```

## 7. Cross-references

- [`../architecture/latency-simulator.md`](../architecture/latency-simulator.md) —
  the simulator at the system level (when it activates, public API,
  comparison with the data-sidecar).
- [`../architecture/README.md`](../architecture/README.md) §2.3 — the sidecar
  pattern (data-sidecar vs. service-sidecar).
- [`.claude/memory/project_lmetric_predictor_design.md`](../.claude/memory/project_lmetric_predictor_design.md) —
  full design rationale: locked spec, phase status, group A/B/C decisions,
  feature schema, calibration loop, E2E piggyback verification numbers.
- `router/src/scheduler/simulator/` — the implementation.
  - `mod.rs` — `SIMULATOR: OnceLock<SimulatorRuntime>` and the public API surface.
  - `pctx.rs` — `PCtx` struct (the L1/L2/L3 owner) and trigger implementations.
  - `mirror.rs` — `IncrementalMirror` (L1.mirror).
  - `sched.rs` — `SchedSnapshot` and `ReqProgress` (L1.sched).
  - `predictor.rs` — `Predictor` + `Corrector` traits; `RegressionalPredictor<P, C>`; concrete correctors `LinregCorrector` and `NullCorrector` (L2).
  - `rollout.rs` — `RolloutBuffer` (slots: `VecDeque<RolloutSlot>`), `RolloutSlot`, `RolloutGist` (L3 data shapes).
- `router/src/engine/colocation.rs` — the SSE-side completion loop;
  `simulator::on_sse` is called after the SCtx prefix-cache update
  completes (post-fix; see §1.2).
