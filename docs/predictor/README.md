# Predictor — design docs

This directory holds the deep-dive design for `blitz-router`'s **latency
simulator** (a.k.a. the predictor subsystem). The simulator's
**system-level role** — that it's a service-sidecar to `PolicyRunner` —
lives in [`../architecture/latency-simulator.md`](../architecture/latency-simulator.md);
read that first if you only need to know where the simulator fits.

This doc set zooms in on the predictor's internals and on how the
external APIs mutate its state.

## When to read what

| You want to understand … | Read |
|---|---|
| What the simulator is (system-level role, when it activates, public API) | [`../architecture/latency-simulator.md`](../architecture/latency-simulator.md) |
| The structural design — per-replica `PCtx`, the L1 / L2 / L3 split, why each layer is shaped the way it is | [`three-layer-architecture.md`](three-layer-architecture.md) |
| The behavioral description — how `on_admit` / `on_sse` / `query` interleave, exact state transitions per API, the savepoint mechanics, the 4-branch maintenance | [`behavior.md`](behavior.md) |
| The first end-to-end run of a prediction-based policy (`least-ttft-q`) — dispatch behavior, simulator caveats with off-model grids, follow-up list | [`least-ttft-q-test-results.md`](least-ttft-q-test-results.md) |

## One-screen summary

Per-replica state, owned by the simulator subsystem and accessed only
through three module-level functions:

```
PCtx (one per replica)
 ├── L1: world model
 │     ├── mirror: IncrementalMirror (RadixTreeReqIdHash + by_request)
 │     └── sched:  SchedSnapshot     (waiting + running per-request progress)
 ├── L2: cost oracle
 │     └── regressor: Mutex<Box<dyn TrainedPredictor>>
 │           └── RegressionalPredictor<P, C>
 │                 ├── inner:     Arc<P: Predictor>      ← offline grid
 │                 └── corrector: C: Corrector
 │                     ├── LinregCorrector  (production: w0·raw + w1, SGD)
 │                     └── NullCorrector    (passthrough)
 ├── L3: DES rollout
 │     └── ephemeral: Mutex<Option<EphemeralRollout>>
 │                      ├── candidate_id, buffer (slots: VecDeque)
 │                      ├── sse_anchor_step_id, tail_sched
 │                      └── savepoint: Option<(usize, SchedSnapshot)>
 └── aux: last_sse_step_id (AtomicU64), block_size (usize)
```

Three APIs:

| Function   | Direction        | Caller                                          |
|------------|------------------|-------------------------------------------------|
| `on_sse`   | silent wiring ←  | BACK `completion_event_loop` (per engine step)  |
| `on_admit` | silent wiring ←  | MIDDLE `policies::PolicyRunner::queue_task`     |
| `query`    | active call →    | a prediction-based policy (e.g. `least-ttft-q`) |

For everything else — invariants, drop semantics, the savepoint
mechanics, the cross-sidecar TOCTOU history — see the two design files
linked above.
