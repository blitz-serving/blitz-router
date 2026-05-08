# Refactor Plan: Layer-Reflecting Module Layout

This is a **plan** — not yet executed. It brings `router/src/`'s on-disk
layout into alignment with the three-layer architecture documented in
[`architecture.md`](architecture.md) §2. Pure code reorganization;
zero algorithmic change; identical binary output expected.

## 1. Why

Today every module under `router/src/` is a flat sibling of every
other module. There is no syntactic separation between the gateway,
the scheduler, and the engine driver — a new contributor reading the
directory listing has no way to tell that, e.g., `vllmlet.rs` and
`server.rs` belong to opposite ends of the request path. 17 flat
files, 7,581 lines of code, three logically distinct concerns sitting
in one namespace.

We want the directory tree to be the architecture diagram. Then the
boundaries enforced by `mod` visibility (`pub(crate)` vs
`pub(super)`) actually mean something.

## 2. Target layout

```
router/src/
├── main.rs                # CLI entry + startup wiring (mostly unchanged)
├── lib.rs                 # SLIMMED: layer mod decls + cross-cutting types
├── error.rs               # cross-cutting; stays at root
│
├── gateway/               # FRONT
│   ├── mod.rs             # re-exports
│   ├── server.rs          # was router/src/server.rs
│   ├── validation.rs      # was router/src/validation.rs
│   ├── chat_template.rs   # was router/src/chat_template.rs
│   ├── model_config.rs    # was router/src/model_config.rs
│   ├── health.rs          # was router/src/health.rs
│   └── api_types.rs       # extracted from lib.rs:
│                          #   GenerateRequest, GenerateParameters,
│                          #   CompatGenerateRequest, GenerateResponse,
│                          #   StreamResponse, StreamDetails, Details,
│                          #   ChatMessage, ChatCompletionRequest,
│                          #   ChatCompletionResponse, ChatCompletionChunk,
│                          #   ChatCompletionChoice, ChatCompletionStop,
│                          #   PrefillToken, Token, FinishReason,
│                          #   BestOfSequence, ErrorResponse
│
├── scheduler/             # MIDDLE
│   ├── mod.rs             # re-exports
│   ├── infer.rs           # was router/src/infer.rs
│   ├── queue.rs           # was router/src/queue.rs (8-LOC shim — keep as-is)
│   ├── colocation.rs      # was router/src/colocation.rs
│   ├── kvcache.rs         # was router/src/kvcache.rs
│   ├── state.rs           # was router/src/metrics.rs — RENAMED to avoid
│   │                      #   collision with the `metrics` crate from
│   │                      #   crates.io (the file holds LMetric,
│   │                      #   ScheduleContext, policy tunable constants)
│   ├── statistic.rs       # was router/src/statistic.rs
│   ├── policies/          # was router/src/policies/ (moved as a unit)
│   │   ├── mod.rs
│   │   ├── policy_trait.rs
│   │   ├── policy_runner.rs
│   │   ├── dsl_runtime.rs
│   │   ├── simple.rs
│   │   ├── vllm.rs
│   │   ├── lmetric.rs
│   │   ├── bailian.rs
│   │   ├── aibrix.rs
│   │   ├── dynamo.rs
│   │   ├── preble/
│   │   └── llm_d/
│   └── simulator/         # was router/src/simulator/ (moved as a unit)
│       └── …
│
└── engine/                # BACK
    ├── mod.rs             # re-exports + EngineClient trait re-export
    ├── client.rs          # was router/src/engine_client.rs
    ├── vllm_http.rs       # was router/src/vllmlet.rs (RENAMED for
    │                      #   precision; the file is the HTTP+SSE
    │                      #   adapter, not "a vllm-let")
    └── zmq.rs             # was router/src/zmq_engine.rs
```

## 3. Renames to flag explicitly

| From                       | To                          | Why |
|----------------------------|-----------------------------|-----|
| `metrics.rs`               | `scheduler/state.rs`        | Naming collides with the `metrics` crate (`metrics::increment_counter!` is used throughout `server.rs`). The file is actually the per-replica scheduling state (`LMetric` + `ScheduleContext` + policy tunables) — `state.rs` describes its content. |
| `vllmlet.rs`               | `engine/vllm_http.rs`       | "vllmlet" is not a noun. The file is the HTTP+SSE adapter for engines that speak the yaullm wire format. `vllm_http.rs` is searchable. |
| `engine_client.rs`         | `engine/client.rs`          | Inside the `engine/` namespace `client` is unambiguous. |
| `zmq_engine.rs`            | `engine/zmq.rs`             | Same reasoning. |

Module renames are surfaced through `pub use` in `engine/mod.rs` so
type names (`EngineClient`, `VllmClient`, `ZmqEngineClient`) do **not**
change.

## 4. Migration in topological order

Each step compiles green on its own. Run `cargo check -p router
--features lmetric-q` after every step.

### Step 0 — preconditions

- Branch off `lmetric/camera-ready`. Suggested branch:
  `refactor/layer-modules`.
- Confirm `cargo test -p router --features lmetric-q --lib kvcache`
  passes 15/16 (the `bug_insert_count_duplicate_hashes` failure is
  pre-existing inherited drift; it must keep its current status, not
  get worse).

### Step 1 — extract `gateway/api_types.rs`

The cleanest thing first: split `lib.rs` (460 LOC) into
"layer-mod-decls + crate-public glue" (stays in `lib.rs`) and "API
DTOs" (move to `gateway/api_types.rs`).

1. `mkdir router/src/gateway`
2. `touch router/src/gateway/mod.rs`
3. Move from `lib.rs` to `gateway/api_types.rs`: `GenerateParameters`,
   `GenerateRequest`, `CompatGenerateRequest`, `GenerateResponse`,
   `StreamResponse`, `StreamDetails`, `Details`, `BestOfSequence`,
   `ChatMessage`, `ChatCompletionRequest`, `ChatCompletionStop`,
   `ChatCompletionResponse`, `ChatCompletionChunk`,
   `ChatCompletionChoice`, `ChatCompletionDelta`,
   `ChatCompletionUsage`, `PrefillToken`, `Token`, `FinishReason`,
   `ErrorResponse`, `default_parameters`, `default_max_new_tokens`.
4. `lib.rs` adds `mod gateway;` and `pub(crate) use gateway::api_types::*;`
   — preserves every existing call site.
5. `cargo check`.

### Step 2 — relocate gateway leaves

Move the obviously-front modules. Each is independent of the others;
do them as one commit.

```
router/src/server.rs        → router/src/gateway/server.rs
router/src/validation.rs    → router/src/gateway/validation.rs
router/src/chat_template.rs → router/src/gateway/chat_template.rs
router/src/model_config.rs  → router/src/gateway/model_config.rs
router/src/health.rs        → router/src/gateway/health.rs
```

Update `gateway/mod.rs`:

```rust
pub(crate) mod api_types;
pub(crate) mod server;
pub(crate) mod validation;
pub(crate) mod chat_template;
pub(crate) mod model_config;
pub(crate) mod health;

pub(crate) use api_types::*;
pub(crate) use chat_template::{ChatRenderer, load_chat_template};
// (re-export only what was previously crate-public)
```

Update `lib.rs`: drop the five old `mod` declarations. Replace
`pub use chat_template::{ChatRenderer, load_chat_template};` with
`pub use gateway::{ChatRenderer, load_chat_template};` (or just rely
on the wildcard).

`cargo check`. Fix any `use crate::server::…` / `use crate::validation::…`
hits — should be a small handful (mostly inside `main.rs`).

### Step 3 — extract `engine/`

Mirror Step 2 for the back layer.

```
router/src/engine_client.rs → router/src/engine/client.rs
router/src/vllmlet.rs       → router/src/engine/vllm_http.rs
router/src/zmq_engine.rs    → router/src/engine/zmq.rs
```

`engine/mod.rs`:

```rust
pub(crate) mod client;
pub(crate) mod vllm_http;
#[cfg(feature = "zmq-backend")]
pub(crate) mod zmq;

pub(crate) use client::{
    EngineClient, EngineStepReceiver, EngineStepOutput,
    EngineClientError, RequestStepOutput,
};
pub(crate) use vllm_http::{VllmClient, VllmClientError, VllmMetric};
#[cfg(feature = "zmq-backend")]
pub(crate) use zmq::ZmqEngineClient;
```

Update callers: `use crate::engine_client::…` → `use crate::engine::…`.
Update `lib.rs`: drop the three old `mod` declarations, add `mod engine;`.

`cargo check`. The change is purely path; trait/struct names are
unchanged.

### Step 4 — extract `scheduler/` (the big one)

Move the middle-layer modules as a unit. Do this in one commit because
they cross-reference each other heavily.

```
router/src/infer.rs       → router/src/scheduler/infer.rs
router/src/queue.rs       → router/src/scheduler/queue.rs
router/src/colocation.rs  → router/src/scheduler/colocation.rs
router/src/kvcache.rs     → router/src/scheduler/kvcache.rs
router/src/metrics.rs     → router/src/scheduler/state.rs   # rename
router/src/statistic.rs   → router/src/scheduler/statistic.rs
router/src/policies/      → router/src/scheduler/policies/  # move dir
router/src/simulator/     → router/src/scheduler/simulator/ # move dir
```

`scheduler/mod.rs`:

```rust
pub(crate) mod infer;
pub(crate) mod queue;
pub(crate) mod colocation;
pub(crate) mod kvcache;
pub(crate) mod state;
pub(crate) mod statistic;
pub(crate) mod policies;
#[cfg(feature = "simulator")]
pub mod simulator;        // pub: simulator::query / simulator::on_admit
                          //      are called by main.rs

pub(crate) use infer::Infer;
pub(crate) use queue::TaskAssigner;
pub(crate) use colocation::{ColocationController, ExtExcept};
pub(crate) use kvcache::{
    BlockHash, BlockHashState,
    PrefixBlockHash,                    // feature-gated re-export
    BackendBlockHash,
    DEFAULT_BLOCK_HASH,
};
pub(crate) use state::{
    LMetric, LMetricDec, LMetricInc, ScheduleContext,
    BAILIAN_ALPHA, BAILIAN_BETA, BAILIAN_GAMMA,
    LOAD_AWARE_QUEUE_T, WAITINGT_PREFILL_TOKEN_BOUND,
    MOST_HIT_LOAD_W_HIT, MOST_HIT_LOAD_W_LOAD,
    MOST_HIT_LOAD_ACTIVE_W_HIT, MOST_HIT_LOAD_ACTIVE_W_LOAD,
    MOST_HIT_LOAD_ACTIVE_W_KV,
    PREFILL_TKN_FREQ_EMA_GAMMA, TBT_EMA_GAMMA,
};
```

Update `lib.rs`:
- Drop the eight old `mod` declarations (`infer`, `queue`, `colocation`,
  `kvcache`, `metrics`, `statistic`, `policies`, `simulator`).
- Add `mod scheduler;` and `pub use scheduler::*;` (matches the
  current `pub use metrics::*;` blanket).

Internal use-path edits (the bulk of the diff):
- `use crate::infer::…`        → `use crate::scheduler::infer::…`
  (or just `use crate::scheduler::Infer;`)
- `use crate::kvcache::…`      → `use crate::scheduler::kvcache::…`
- `use crate::metrics::…`      → `use crate::scheduler::state::…`
- `use crate::policies::…`     → `use crate::scheduler::policies::…`
- `use crate::simulator::…`    → `use crate::scheduler::simulator::…`
- `use crate::ScheduleContext` (via lib.rs blanket) → keep, unchanged
- `crate::ColocationController` → `crate::scheduler::ColocationController`

Inside `scheduler/policies/` use-paths:
- `use crate::kvcache::BlockHashState` → `use crate::scheduler::kvcache::BlockHashState`
  OR (cleaner) `use super::kvcache::BlockHashState`.
- `use crate::ScheduleContext` (the re-export) keeps working.

Inside `scheduler/simulator/`:
- `use crate::policies::Entry` → `use super::policies::Entry`.
- `use crate::engine_client::EngineStepOutput` →
  `use crate::engine::EngineStepOutput`.
- `use radixtree::RadixTreeReqIdHash` — unchanged (cross-crate).

`cargo check`. There will be on the order of 30-50 use-path fixups
across `policies/`, `simulator/`, `colocation.rs`, `infer.rs`, and
`server.rs`. Mostly mechanical.

### Step 5 — slim `lib.rs`

After Steps 1-4, `lib.rs` should hold only:

```rust
// ~80 LOC instead of 460
mod gateway;
mod scheduler;
mod engine;
mod error;

#[cfg(not(any(feature = "vllm-backend", feature = "zmq-backend")))]
compile_error!("You must enable either `vllm-backend` or `zmq-backend`!");

// Re-exports for the binary entry point and external consumers
pub use gateway::{api_types::*, ChatRenderer, load_chat_template};
pub use scheduler::*;          // catch-all: ScheduleContext, Infer, …
pub use engine::*;
pub use error::*;

// Cross-cutting types that don't belong to a layer:
//   HubModelInfo, Info, TokenizerRender
//   (these are pure DTOs / wrappers used at startup)
…
```

Verify by line count: `wc -l router/src/lib.rs` should drop from 460
to under 100.

### Step 6 — doc-code consistency

Per `CLAUDE.md`'s "Doc-Code Consistency at Commit Time" rule, the same
PR updates:

- `CLAUDE.md` — project-tree section: replace the flat `router/src/` list
  with a layered tree.
- `CLAUDE.md` — `Helper Scripts` section was already removed; nothing to
  add here.
- `docs/architecture.md` — drop the "directory does not yet reflect
  layers" admonition box at the top, drop the parenthetical "(currently
  flat)" mentions.
- `docs/architecture.md` §4-§6 — current path columns become target
  paths; the "(in the refactor: …)" parenthetical inside the `Policy`
  trait code block becomes the actual path.
- `docs/refactor-plan.md` — this file gets a new top-level note
  "Executed in commit `…`" rather than being deleted (keeps the
  rationale + migration narrative as repo history).
- `.claude/skills/add-policy.md` and `.claude/skills/build-router.md` —
  any hard-coded file paths (`router/src/policies/…`,
  `router/src/Cargo.toml` is unchanged) get the `scheduler/` prefix.
- `.claude/skills/verify-policy.md` — same.
- `radixtree/README.md` — references to `router/src/kvcache.rs` and
  `router/src/simulator/` get the `scheduler/` prefix.

## 5. What does NOT change

These deserve explicit non-goals so reviewers don't ask:

- **No algorithmic change.** Every byte of executable code is moved
  unchanged. `cargo test -p router --features lmetric-q --lib` should
  produce an identical pass/fail set (modulo the pre-existing inherited
  failure).
- **No public API change.** External consumers of the `router` crate
  (e.g., the bin entry point, integration tests) see the same symbols
  via `lib.rs` re-exports.
- **No Cargo manifest change** beyond what's needed for the rename
  (`metrics.rs` → `state.rs` requires no manifest edit since it's not
  an explicit `path = …` entry; same for the others).
- **No feature flag change.** Every existing `<name>-q`,
  `radixtree-blockhash`, `simulator`, `zmq-backend`, etc. continues to
  exist with the same effect.
- **`policies/` and `simulator/` internal layout** is preserved
  verbatim — only the parent path changes.
- **`radixtree`, `policy-dsl`, `rust-proto`, `request-sim` workspace
  members** are not touched.
- **TLA+ spec (`formal/tlaplus/`)** is not touched. (It models
  colocation-loop dynamics, not module structure.)

## 6. Verification

A clean refactor is verifiable. The PR description should include:

1. **Identical compile output**:
   ```bash
   cargo build --release -p router --features lmetric-q
   sha256sum target/release/router  # before and after the merge
   ```
   These should match for the same Cargo.lock and the same compiler
   version. (If they don't, something semantic changed by accident.)

2. **Tests unchanged**:
   ```bash
   cargo test -p router --features lmetric-q --lib 2>&1 | tail -20
   # 29/30 pass (same as today)
   ```

3. **No new warnings**:
   ```bash
   cargo check -p router --features lmetric-q 2>&1 | grep -c warning
   # equal to today's count (6 pre-existing simulator-CLI vars)
   ```

4. **Re-export equivalence**:
   ```bash
   # Symbols visible at crate boundary should be unchanged
   cargo doc -p router --features lmetric-q --no-deps
   # Compare the index page before/after — only paths in source
   # locations should differ; symbol list should be identical.
   ```

5. **Architecture diagrams render**:
   - `docs/architecture.md` — render and check all 7 mermaid blocks.

## 7. Rollback strategy

If review surfaces an issue with the layout (e.g., a contributor
strongly disagrees with where one file lives), single-commit revert
is safe because:
- The refactor is one PR, one squash-merge candidate.
- No rebased work depends on the new paths until merge.
- Doc updates in the same PR rollback together.

After merge, partial rollback (move just one module back) is also low
risk because the changes are mechanical.

## 8. Estimated scope

| Activity                              | Estimate |
|---------------------------------------|----------|
| File moves                            | ~17 files |
| New mod.rs files                      | 3 (`gateway/`, `scheduler/`, `engine/`) |
| Use-path fixups                       | ~30-50 sites |
| Doc-consistency edits                 | 6 surfaces (CLAUDE.md, architecture.md, refactor-plan.md, 3 skills) |
| Total expected PR diff                | ~+200 / -100 LOC of glue + ~7,500 LOC of moves (mostly path-only renames in git's view) |

The git rename detector should pick up most files at >90% similarity,
so reviewers see file-rename annotations rather than 7,500-line diffs.

## 9. Suggested commit shape

One PR, two commits:

1. `refactor(modules): adopt three-layer directory layout`
   — all file moves, mod.rs creations, use-path fixups, lib.rs slim-down.
2. `docs(architecture): align doc surfaces with new layout`
   — CLAUDE.md project tree, architecture.md path columns,
   refactor-plan.md header note, skill path updates.

(Or one squash-merge of both commits if that matches the team's PR
norms.)
