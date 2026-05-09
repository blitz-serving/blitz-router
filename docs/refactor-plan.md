# Refactor Plan: Layer-Reflecting Module Layout

> **Status: EXECUTED in commit `75090e0`** (refactor) and the
> companion doc-consistency commit. Kept in repo as the migration
> narrative. The current `router/src/` layout already matches the
> "Target layout" in §2 below; new contributors should read this for
> the *rationale* of the layered split, then read
> [`architecture/README.md`](architecture/README.md) for the
> architecture itself.

This is the original migration plan. It brings `router/src/`'s on-disk
layout into alignment with the three-layer architecture documented in
[`architecture/README.md`](architecture/README.md). Pure code
reorganization; zero algorithmic change; identical binary output
expected.

## 1. Why

Today every module under `router/src/` is a flat sibling of every
other module. There is no syntactic separation between the gateway,
the scheduler, and the engine driver — a new contributor reading the
directory listing has no way to tell that, e.g., `vllmlet.rs` and
`server.rs` belong to opposite ends of the request path. **17 flat
files**, 7,581 lines of code, three logically distinct concerns
sitting in one namespace.

We want the directory tree to be the architecture diagram. Then the
boundaries enforced by `mod` visibility (`pub(crate)` vs
`pub(super)`) actually mean something.

## 2. Target layout

The architecture's **decide-vs-execute** split between MIDDLE and
BACK (see [`architecture/README.md`](architecture/README.md) §2.2)
puts `colocation.rs` — both the work and completion event loops —
firmly in BACK, not MIDDLE. The target layout reflects that.

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
│                          #   GenerateResponse, StreamResponse,
│                          #   StreamDetails, Details, BestOfSequence,
│                          #   ChatMessage, ChatCompletionRequest,
│                          #   ChatCompletionResponse, ChatCompletionChunk,
│                          #   ChatCompletionChoice, ChatCompletionStop,
│                          #   ChatCompletionDelta, ChatCompletionUsage,
│                          #   PrefillToken, Token, FinishReason,
│                          #   ErrorResponse, default_parameters,
│                          #   default_max_new_tokens
│
├── scheduler/             # MIDDLE — decide
│   ├── mod.rs             # re-exports
│   ├── infer.rs           # was router/src/infer.rs
│   ├── queue.rs           # was router/src/queue.rs (8-LOC shim — keep as-is)
│   ├── kvcache.rs         # was router/src/kvcache.rs
│   ├── state.rs           # was router/src/metrics.rs — RENAMED to avoid
│   │                      #   collision with the `metrics = "0.21.1"` crate
│   │                      #   from crates.io (which `server.rs` uses for
│   │                      #   `metrics::increment_counter!`). The file
│   │                      #   actually holds LMetric, ScheduleContext,
│   │                      #   policy tunable constants.
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
└── engine/                # BACK — execute + observe
    ├── mod.rs             # re-exports + EngineClient trait re-export
    ├── client.rs          # was router/src/engine_client.rs
    ├── colocation.rs      # was router/src/colocation.rs — moved here
    │                      #   per architecture: colocation owns the work +
    │                      #   completion event loops, calls EngineClient,
    │                      #   writes the ScheduleContext sidecar. It
    │                      #   *executes* scheduler decisions; it does not
    │                      #   *make* them.
    ├── vllm_http.rs       # was router/src/vllmlet.rs (RENAMED for
    │                      #   precision; the file is the HTTP+SSE
    │                      #   adapter, not "a vllm-let")
    └── zmq.rs             # was router/src/zmq_engine.rs
```

## 3. Renames to flag explicitly

| From                       | To                          | Why |
|----------------------------|-----------------------------|-----|
| `metrics.rs`               | `scheduler/state.rs`        | Naming collides with the `metrics = "0.21.1"` crate from crates.io (`metrics::increment_counter!` is used throughout `server.rs`). The file is actually the per-replica scheduling state (`LMetric` + `ScheduleContext` + policy tunables) — `state.rs` describes its content. |
| `vllmlet.rs`               | `engine/vllm_http.rs`       | "vllmlet" is not a noun. The file is the HTTP+SSE adapter for engines that speak the yaullm wire format. `vllm_http.rs` is searchable. |
| `engine_client.rs`         | `engine/client.rs`          | Inside the `engine/` namespace `client` is unambiguous. |
| `zmq_engine.rs`            | `engine/zmq.rs`             | Same reasoning. |

Module renames are surfaced through `pub use` in `engine/mod.rs` so
type names (`EngineClient`, `VllmClient`, `ZmqEngineClient`,
`ColocationController`) do **not** change.

## 4. Migration in topological order

Each step compiles green on its own. Run `cargo check -p router
--features lmetric-q` after every step.

### Step 0 — preconditions

- Branch off `lmetric/camera-ready`. Suggested branch:
  `refactor/layer-modules`.
- Confirm baseline tests pass:
  ```bash
  cargo test -p router --features lmetric-q --lib kvcache 2>&1 | tail -5
  ```
  Expect **16/16 passing**, including `insert_count_duplicate_hashes_agrees`
  (which asserts that both `RadixTreeBlockHash` and the hashtable impl
  count duplicate-hash-with-distinct-bid as fresh insertion → both
  return 2 for `[6,6]`).

### Step 1 — extract `gateway/api_types.rs`

The cleanest thing first: split `lib.rs` (367 LOC today) into
"layer-mod-decls + crate-public glue" (stays in `lib.rs`) and "API
DTOs" (move to `gateway/api_types.rs`).

1. `mkdir router/src/gateway`
2. `touch router/src/gateway/mod.rs`
3. Move from `lib.rs` to `gateway/api_types.rs`: `GenerateParameters`,
   `GenerateRequest`, `GenerateResponse`, `StreamResponse`,
   `StreamDetails`, `Details`, `BestOfSequence`, `ChatMessage`,
   `ChatCompletionRequest`, `ChatCompletionStop`,
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

### Step 3 — extract `engine/` (now includes `colocation`)

Mirror Step 2 for the back layer. **Note**: `colocation.rs` moves
here, NOT to `scheduler/`. Per
[`architecture/README.md`](architecture/README.md) §2.2 and
[`architecture/back-engine-driver.md`](architecture/back-engine-driver.md),
the colocation event loops execute scheduler decisions and write the
data-sidecar — those are BACK responsibilities.

```
router/src/engine_client.rs → router/src/engine/client.rs
router/src/vllmlet.rs       → router/src/engine/vllm_http.rs
router/src/zmq_engine.rs    → router/src/engine/zmq.rs
router/src/colocation.rs    → router/src/engine/colocation.rs
```

`engine/mod.rs`:

```rust
pub(crate) mod client;
pub(crate) mod colocation;
pub(crate) mod vllm_http;
#[cfg(feature = "zmq-backend")]
pub(crate) mod zmq;

pub(crate) use client::{
    EngineClient, EngineStepReceiver, EngineStepOutput,
    EngineClientError, RequestStepOutput,
};
pub(crate) use colocation::{ColocationController, ExtExcept};
pub(crate) use vllm_http::{VllmClient, VllmClientError, VllmMetric};
#[cfg(feature = "zmq-backend")]
pub(crate) use zmq::ZmqEngineClient;
```

Update callers:
- `use crate::engine_client::…` → `use crate::engine::…`
- `use crate::colocation::…`    → `use crate::engine::colocation::…`
  (or `use crate::engine::ColocationController`)
- `use crate::vllmlet::…`       → `use crate::engine::vllm_http::…`
- `use crate::zmq_engine::…`    → `use crate::engine::zmq::…`

Update `lib.rs`: drop the four old `mod` declarations
(`engine_client`, `vllmlet`, `zmq_engine`, `colocation`); add
`mod engine;` and `pub use engine::*;` (matches the previous
`pub use colocation::*` blanket so `crate::ColocationController`
keeps resolving from outside `engine/`).

At this point `colocation.rs` still imports things from
`crate::policies`, `crate::metrics`, `crate::simulator`,
`crate::kvcache` — those are still at root, so this step keeps
compiling. Step 4 will retarget those.

`cargo check`. The change is purely path; trait/struct names are
unchanged.

### Step 4 — extract `scheduler/` (the big one)

Move the middle-layer modules as a unit. Do this in one commit because
they cross-reference each other heavily, and `engine/colocation.rs`
(from Step 3) needs its imports retargeted to the new scheduler paths.

```
router/src/infer.rs       → router/src/scheduler/infer.rs
router/src/queue.rs       → router/src/scheduler/queue.rs
router/src/kvcache.rs     → router/src/scheduler/kvcache.rs
router/src/metrics.rs     → router/src/scheduler/state.rs        # rename
router/src/statistic.rs   → router/src/scheduler/statistic.rs
router/src/policies/      → router/src/scheduler/policies/       # move dir
router/src/simulator/     → router/src/scheduler/simulator/      # move dir
```

`scheduler/mod.rs`:

```rust
pub(crate) mod infer;
pub(crate) mod queue;
pub(crate) mod kvcache;
pub(crate) mod state;
pub(crate) mod statistic;
pub(crate) mod policies;
#[cfg(feature = "simulator")]
pub mod simulator;        // pub: simulator::query is the future
                          // service-sidecar API; simulator::on_admit
                          // and simulator::on_sse are pub(crate) hooks
                          // called by policies/policy_runner.rs and
                          // engine/colocation.rs respectively.

pub(crate) use infer::Infer;
pub(crate) use queue::TaskAssigner;
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

Note: `colocation` is **not** in this list — it lives in `engine/`
from Step 3.

Update `lib.rs`:
- Drop the seven old `mod` declarations (`infer`, `queue`,
  `kvcache`, `metrics`, `statistic`, `policies`, `simulator`).
- Add `mod scheduler;` and `pub use scheduler::*;` (matches the
  current `pub use metrics::*;` blanket).

Internal use-path edits (the bulk of the diff):

- `use crate::infer::…`      → `use crate::scheduler::infer::…`
  (or just `use crate::scheduler::Infer;`)
- `use crate::kvcache::…`    → `use crate::scheduler::kvcache::…`
- `use crate::metrics::…`    → `use crate::scheduler::state::…`
- `use crate::policies::…`   → `use crate::scheduler::policies::…`
- `use crate::simulator::…`  → `use crate::scheduler::simulator::…`
- `use crate::ScheduleContext` (via lib.rs blanket) → keep, unchanged

Inside `engine/colocation.rs` (the cross-layer caller):
- `use crate::policies::Entry` → `use crate::scheduler::policies::Entry`
- `use crate::metrics::ScheduleContext` → `use crate::scheduler::state::ScheduleContext`
- `use crate::kvcache::BlockHashState` → `use crate::scheduler::kvcache::BlockHashState`
- `simulator::on_sse(...)` call site → `crate::scheduler::simulator::on_sse(...)`

Inside `scheduler/policies/` use-paths:
- `use crate::kvcache::BlockHashState` → `use super::kvcache::BlockHashState`
  (cleaner than `crate::scheduler::kvcache::…`)
- `use crate::ScheduleContext` (the re-export) keeps working.
- `simulator::on_admit(...)` call site → `super::simulator::on_admit(...)`

Inside `scheduler/simulator/`:
- `use crate::policies::Entry` → `use super::policies::Entry`
- `use crate::engine_client::EngineStepOutput` → `use crate::engine::EngineStepOutput`
- `use radixtree::RadixTreeReqIdHash` — unchanged (cross-crate).

`cargo check`. There will be on the order of 30-50 use-path fixups
across `policies/`, `simulator/`, `engine/colocation.rs`, `infer.rs`,
and `gateway/server.rs`. Mostly mechanical.

### Step 5 — slim `lib.rs`

After Steps 1-4, `lib.rs` should hold only:

```rust
// ~80 LOC instead of 367
mod gateway;
mod scheduler;
mod engine;
mod error;

#[cfg(not(any(feature = "vllm-backend", feature = "zmq-backend")))]
compile_error!("You must enable either `vllm-backend` or `zmq-backend`!");

// Re-exports for the binary entry point and external consumers
pub use gateway::{api_types::*, ChatRenderer, load_chat_template};
pub use scheduler::*;          // catch-all: ScheduleContext, Infer, …
pub use engine::*;             // catch-all: EngineClient, ColocationController, …
pub use error::*;

// Cross-cutting types that don't belong to a layer:
//   HubModelInfo, Info, TokenizerRender
//   (these are pure DTOs / wrappers used at startup)
…
```

Verify by line count: `wc -l router/src/lib.rs` should drop from 367
(today) to under 100.

### Step 6 — doc-code consistency

Per `CLAUDE.md`'s "Doc-Code Consistency at Commit Time" rule, the same
PR updates:

- `CLAUDE.md` — project-tree section: replace the flat `router/src/`
  list with a layered tree (`gateway/`, `scheduler/`, `engine/` rows).
  (The `architecture/` reference in the tree comment is already
  correct after the architecture-doc split — no change needed there.)
- `docs/architecture/README.md` — drop the
  "directory does not yet reflect layers" admonition box near the top
  (`> The three-layer view is the conceptual architecture. The on-disk
  layout under router/src/ does not yet reflect it…`).
- `docs/architecture/front-gateway.md`, `middle-scheduler.md`,
  `back-engine-driver.md` — current path columns in their module
  tables become target paths (`router/src/gateway/server.rs`,
  `router/src/scheduler/policies/`, `router/src/engine/colocation.rs`,
  etc.).
- `docs/architecture/middle-scheduler.md` — drop the parenthetical
  about `metrics.rs` colliding with the crates.io `metrics` crate
  (the rename happened in Step 4); replace with a one-line note
  `state.rs (formerly metrics.rs)`.
- `docs/architecture/middle-scheduler.md` §"The Policy abstraction" —
  the `Currently at: router/src/policies/policy_trait.rs` comment
  becomes `router/src/scheduler/policies/policy_trait.rs`.
- `docs/refactor-plan.md` — this file gets a new top-level note
  "Executed in commit `…`" rather than being deleted (keeps the
  rationale + migration narrative as repo history).
- `.claude/skills/add-policy.md`, `.claude/skills/build-router.md`,
  `.claude/skills/verify-policy.md` — any hard-coded file paths
  (`router/src/policies/…`; `router/src/Cargo.toml` is unchanged)
  get the `scheduler/` prefix.
- `.claude/skills/blitz-router-architecture.md` — only its
  *external-companion* paths might shift (e.g., the `Step 5`
  references to `radixtree/README.md`); the per-topic
  `docs/architecture/*.md` paths are unaffected by the module-layout
  refactor.
- `radixtree/README.md` — references to `router/src/kvcache.rs` and
  `router/src/simulator/` get the `scheduler/` prefix.
- `router/src/gateway/server.rs` — no edit needed; the
  `tgi_deprecated` handler already references
  `docs/architecture/README.md` after the architecture-doc split.

## 5. What does NOT change

These deserve explicit non-goals so reviewers don't ask:

- **No algorithmic change.** Every byte of executable code is moved
  unchanged. `cargo test -p router --features lmetric-q --lib` should
  produce an identical pass/fail set (16/16 today; 16/16 after).
- **No public API change.** External consumers of the `router` crate
  (e.g., the bin entry point, integration tests) see the same symbols
  via `lib.rs` re-exports.
- **No Cargo manifest change** beyond what's needed for the rename
  (`metrics.rs` → `state.rs` requires no manifest edit since it's not
  an explicit `path = …` entry; same for the others).
- **No feature flag change.** Every existing `<name>-q` (one of 18),
  `radixtree-blockhash`, `simulator`, `zmq-backend`, etc. continues
  to exist with the same effect.
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
   # 16/16 kvcache pass (same as today); overall pass/fail unchanged.
   ```

3. **No new warnings**:
   ```bash
   cargo check -p router --features lmetric-q 2>&1 | grep -c warning
   # equal to today's count (pre-existing simulator-CLI vars).
   ```

4. **Re-export equivalence**:
   ```bash
   # Symbols visible at crate boundary should be unchanged
   cargo doc -p router --features lmetric-q --no-deps
   # Compare the index page before/after — only paths in source
   # locations should differ; symbol list should be identical.
   ```

5. **Architecture diagrams render**:
   - All `*.md` files in `docs/architecture/` — render and check the
     mermaid blocks (8 mermaid blocks across the directory).

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
| Doc-consistency edits                 | 8 surfaces (CLAUDE.md, 4 files in docs/architecture/, refactor-plan.md, 4 skills, radixtree/README.md, server.rs docstring) |
| Total expected PR diff                | ~+200 / -100 LOC of glue + ~7,500 LOC of moves (mostly path-only renames in git's view) |

The git rename detector should pick up most files at >90% similarity,
so reviewers see file-rename annotations rather than 7,500-line diffs.

## 9. Suggested commit shape

One PR, two commits:

1. `refactor(modules): adopt three-layer directory layout`
   — all file moves, mod.rs creations, use-path fixups, lib.rs slim-down.
2. `docs(architecture): align doc surfaces with new layout`
   — CLAUDE.md project tree, docs/architecture/* path columns,
   refactor-plan.md header note, skill path updates,
   radixtree/README.md updates, server.rs docstring update.

(Or one squash-merge of both commits if that matches the team's PR
norms.)
