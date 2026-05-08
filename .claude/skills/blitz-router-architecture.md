---
name: blitz-router-architecture
description: Routes architecture questions about blitz-router to the specific docs/architecture/ file the agent needs. Invoke when the agent has to understand the system design — layers (FRONT/MIDDLE/BACK), the sidecar pattern, request lifecycle, simulator, concurrency model, Cargo features, or any other structural aspect — instead of reading the entire architecture doc set.
---

# blitz-router architecture skill

When you need to understand `blitz-router`'s architecture, **do NOT
read every doc in `docs/architecture/`**. Use the routing table below
to read only the files relevant to the current question.

The doc set lives at `docs/architecture/` (relative to the repo root).
Each file is self-contained (it links to `README.md` for foundational
concepts when needed) so reading one in isolation works.

## Step 1 — read this first if you've never seen the codebase

`docs/architecture/README.md` covers:
- The system boundary (what blitz-router is and isn't).
- The **three-layer architecture** (FRONT / MIDDLE / BACK) and the
  `decide-vs-execute` split between MIDDLE and BACK.
- The **sidecar pattern** (data-sidecar like `ScheduleContext` vs
  service-sidecar like `simulator`) — load-bearing concept; later
  files reference it without re-explaining.
- The **inter-layer contracts** (what types cross which boundary).
- The **diagram conventions** (containment via subgraphs, wiring via
  arrow style, layer-stacking discipline) — useful if you ever need
  to add or modify a mermaid block.

Skip this only if the question is narrowly scoped and you already
know the layer split.

## Step 2 — routing table for specific questions

| You're asked about… | Read |
|---|---|
| Overall design; what FRONT/MIDDLE/BACK mean; sidecar pattern | `docs/architecture/README.md` |
| HTTP API surface, axum routes, validation, chat templates, model_config | `docs/architecture/front-gateway.md` |
| Scheduling policies, `Policy` trait, `PolicyRunner`, `ScheduleContext` data-sidecar, why the sidecar pattern | `docs/architecture/middle-scheduler.md` |
| `EngineClient` trait, `vLLM`/ZMQ adapters, `colocation` event loops (work + completion), where the sidecar gets written | `docs/architecture/back-engine-driver.md` |
| End-to-end request lifecycle (`POST /v1/chat/completions` → reply); SSE consumption (engine `/v1/metrics` → `ScheduleContext` write) | `docs/architecture/request-lifecycle.md` |
| Tasks per replica, channels, locks, `Mutex` vs `SpinLock`, sole-writer invariants | `docs/architecture/concurrency-model.md` |
| Latency simulator, the `query()` service-sidecar API, `on_admit`/`on_sse` silent wiring, `PCtx` 3-layer predictor | `docs/architecture/latency-simulator.md` |
| Cargo workspace members (`router`, `radixtree`, `policy-dsl`, `rust-proto`, `request-sim`); feature flags as wiring switches | `docs/architecture/workspace-and-features.md` |

## Step 3 — when to read multiple docs

Some questions span layers. Common combinations:

- **"How does adding a new policy work?"** → `middle-scheduler.md`
  (Policy trait + DSL) plus `back-engine-driver.md` (where the
  policy's chosen replica is actually dispatched to).
- **"Why does feature X exist?"** → `workspace-and-features.md`
  (the feature itself) plus the relevant layer file (what it
  rewires).
- **"How does the simulator integrate?"** → `latency-simulator.md`
  (the simulator itself) plus `middle-scheduler.md` (how policies
  call `query()`) plus `request-lifecycle.md` (where `on_sse` fires).
- **"What's the contract between MIDDLE and BACK?"** → `README.md`
  §2.5 — that table is the single source of truth.

## Step 4 — when NOT to use this skill

- If the question is about **how to implement** a specific change
  (not about understanding the existing design) — use code search
  (`Grep`/`Read` on `router/src/`) instead. The architecture docs
  are conceptual; for implementation details, read the code.
- If the question is about a **different repo** (`sglang`, `yaullm`,
  `blitz-scale`, `request-sim`) — this skill is blitz-router-only.
- If you only need a configuration knob (not the design) — try
  `CLAUDE.md` §"Configuration" first, which is the operational
  reference (the architecture docs are for understanding, not
  cookbooking).

## Step 5 — companion docs (related reading)

- `docs/refactor-plan.md` — planned migration of `router/src/`'s
  flat layout into `gateway/`, `scheduler/`, `engine/` directories.
  Read this if the question is about the **target** layout, not the
  current one.
- `docs/dsl/policies.md` and `docs/dsl/implementation.md` — the
  scheduling policy DSL spec form and its lowering into Rust.
- `radixtree/README.md` — the verified-then-lowered Patricia-trie
  crate used by both sidecars.
- `formal/tlaplus/` — TLA+ spec of the colocation work-loop ↔
  completion-loop handoff (catches order-dependent races, not
  policy logic).
- `.claude/memory/project_lmetric_predictor_design.md` — design
  rationale for the simulator's three-layer predictor (`PCtx`),
  the calibration loop, and the future `simulator-q` policy.
