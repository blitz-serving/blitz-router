# BlitzScale Router - Distributed LLM Inference Router

## Overview
BlitzScale Router (blitz-router) is the **routing component** of the lmetric distributed LLM inference system. Written in Rust, it routes client requests to backend **yaullm** engines (a patched vLLM), manages KV cache state via RadixTree prefix matching, and dynamically scales replicas.

## Communication Protocol — IMPORTANT

**lmetric uses HTTP + SSE, NOT gRPC.**

Entry point: `vllmlet.rs` → `VllmClient` (HTTP) with `/v1/metrics` SSE consumption.

### proto/ and rust-proto/ — internal data structures
The protobuf-generated types (`Tokens`, `GeneratedText`, `Batch`, `Request`, `CachedBatch`, etc.) are used as **internal data structures** throughout the router (queue, infer, validation, colocation) regardless of backend. They are NOT used as a wire protocol — the actual transport is HTTP/SSE via `VllmClient` in `vllmlet.rs`.

### lmetric Data Flow
```
Client (HTTP) → Router (Axum server.rs)
    → [Validation] → [Queue + BlockHashState] → [Replica Selection]
    → VllmClient (vllmlet.rs) → HTTP → yaullm engine
    ← HTTP streaming response ← yaullm
    ← SSE metrics push (/v1/metrics) ← yaullm  [async, separate connection]
```

### SSE Metrics Consumption
The router connects to each yaullm engine's `/v1/metrics` SSE endpoint and receives per-step metrics:
```json
{
    "outputs": [{"request_id": "req-123", "new_token_ids": [456], "state": "RUNNING", "num_cached_tokens": 128}],
    "latency": 45,
    "prefill_token_budget": 2048,
    "evicted_block_ids": [10, 11, 12]
}
```
This drives cache-aware routing (via `evicted_block_ids`) and scheduling decisions (via latency, request states).

## Comparison with AI-Dynamo & AIBrix

| Aspect | BlitzScale (lmetric) | AI-Dynamo | AIBrix |
|--------|-----------|-----------|--------|
| Router language | Rust | Python/C++ | Python/Go |
| Engine | yaullm (patched vLLM) | Custom | Custom |
| Router↔Engine protocol | HTTP + SSE | gRPC / custom | gRPC / custom |
| KV Cache Routing | RadixTree prefix matching | Simplified | Cache-aware |
| Scheduling Policies | 18 compile-time switchable | Fixed | Fixed |
| Metrics | Real-time SSE push per engine step | Basic | Limited telemetry |
| Config Polymorphism | Cargo feature flags (zero-overhead) | Runtime config | Runtime config |

## Project Structure

```
blitz-router/
├── router/src/              # Rust router (~14,000 LOC)
│   ├── main.rs              # CLI args & entry point
│   ├── lib.rs               # Crate root
│   ├── server.rs            # HTTP server (Axum): /generate, /info, /health, /metrics (~1,140 LOC)
│   ├── infer.rs             # Inference orchestration (~530 LOC)
│   ├── queue.rs             # Request queue scaffolding (~380 LOC; policy logic now in policies/)
│   ├── kvcache.rs           # BlockHashState + HashTableBlockHash (~1,165 LOC)
│   │                        #   (radix impl moved to the `radixtree/` crate)
│   ├── colocation.rs        # Co-location controller (~1,120 LOC)
│   ├── engine_client.rs     # Engine client trait & dispatch (~510 LOC)
│   ├── vllmlet.rs           # yaullm/vLLM HTTP+SSE backend
│   ├── zmq_engine.rs        # ZMQ engine variant
│   ├── metrics.rs           # SystemMetric counters & replica states
│   ├── validation.rs        # Request validation
│   ├── chat_template.rs     # Chat template handling
│   ├── model_config.rs      # Model config auto-discovery
│   ├── statistic.rs         # Statistics collection
│   ├── health.rs            # Health checks
│   ├── error.rs             # Error types
│   ├── policies/            # Scheduling policies — DSL-driven,
│   │   │                    # one module per upstream baseline system
│   │   ├── mod.rs               # ~200 LOC: Entry, QueuePro, TaskAssigner aliases
│   │   ├── policy_trait.rs      # `Policy` trait (5 lines, lowering target)
│   │   ├── policy_runner.rs     # `PolicyRunner<P: Policy>` queue runner
│   │   ├── dsl_runtime.rs       # combinators, reducers, named pure fns,
│   │   │                        # `apply_default_after`, Observation schema
│   │   ├── simple.rs            # random-q, round-robin-q, least-wait-token-q,
│   │   │                        # bounded-most-hit-q (no upstream-system origin)
│   │   ├── vllm.rs              # join-shortest-weight-q (vLLM 4·waiting + bs)
│   │   ├── lmetric.rs           # lmetric-q
│   │   ├── bailian.rs           # bailian-impl-q
│   │   ├── aibrix.rs            # aibrix-q
│   │   ├── dynamo.rs            # dynamo-q + dynamo-po-q (Decode-/Prefill-node logits)
│   │   ├── preble/              # preble-q + cost_model/histogram/router utils
│   │   └── llm_d/               # llm-d single-scorer ablations + multi-scorer combos
│   │       ├── mod.rs
│   │       ├── most_hit.rs / most_hit_load.rs / most_hit_load_active.rs
│   │       └── least_active.rs / least_bs.rs / least_token_load.rs / least_waiting.rs
│   └── simulator/           # Latency simulator (feature-gated `simulator`)
│       ├── mod.rs               # piggyback observation entry points (on_sse, on_admit, query)
│       ├── pctx.rs              # process-wide PredictorContext (OnceLock); 3-layer state machine
│       ├── batch.rs             # BatchForPredictor (inner-regressor input)
│       ├── predictor.rs         # Predictor + TrainedPredictor traits
│       ├── rollout.rs           # RolloutBuffer + RolloutSlot + RolloutGist
│       ├── mirror.rs            # PCtx L1 incremental mirror (uses radixtree::RadixTreeReqIdHash)
│       ├── sched.rs              # PCtx SchedSnapshot (per-request progress: waiting/running)
│       ├── vidur_rf.rs          # VidurRfPredictor (port of everparadise LlamaPredictor)
│       └── config.rs            # SimulatorConfig (CSV path, model_hash, granularities)
├── policy-dsl/              # ~150 LOC proc-macro: parser + lint + lowering
│   └── src/{lib,ast,parse,check,lower}.rs
├── radixtree/               # Patricia trie crate consumed by router/kvcache + simulator
│   ├── src/
│   │   ├── lib.rs                  # public surface (BlockHash, RadixTreeBlockHash, RadixTreeReqIdHash)
│   │   ├── core.rs                 # generic K,V Patricia primitives (Node, Children, CommonPrefixInner)
│   │   ├── block_hash.rs           # production L3 specialization (V=Bids, SpinLock, epoch)
│   │   ├── req_id_hash.rs          # simulator-mirror specialization (V=ReqId, Box-based)
│   │   └── verified.rs             # Verus-verified L0 spec (gated by `verify` feature)
│   └── benches/
│       ├── workloads.rs            # Criterion harness (KV-cache-shaped workloads)
│       └── lowering_levels.rs      # L0..L3 trait-based lowering ladder for regression tracking
├── docs/dsl/                # DSL spec, split for progressive disclosure
│   ├── schema.md                # surface syntax + field/reducer schema
│   ├── policies.md              # canonical DSL listings for every policy
│   └── implementation.md        # `policy!` macro: rewrite table + lint allowlist
├── proto/generate.proto     # Protobuf type definitions (used as internal data structures)
├── rust-proto/              # Protobuf codegen (internal types only)
├── request-sim/             # Request simulator (git submodule, main branch)
├── formal/tlaplus/          # TLA+ spec — colocation/CompletionLoop entry lifecycle (NOT a policy spec)
└── docs/                    # reproduce.md
```

## Key Components

### Scheduling Policies (DSL-driven, compile-time via Cargo features)

Each policy is one Cargo feature flag plus a `policy! { ... }` invocation in `router/src/policies/` that the `policy-dsl/` proc macro lowers into an `impl Policy for X { fn schedule(...) }` block. The macro enforces an allowlist lint (`docs/dsl/implementation.md` §2.2) so the impl always corresponds 1:1 to a spec-form DSL listing (`docs/dsl/policies.md` §2) via the rewrite table (`docs/dsl/implementation.md` §2.1). `PolicyRunner<P: Policy>` is the dispatch shim. Feature names are the source of truth — anything not in this list is stale.

Policies are organized under `router/src/policies/` by their upstream baseline system, plus `simple.rs` for trivial policies that have no upstream-system origin.

**`simple.rs`** — trivial, no upstream-system origin
- `random-q` — `Select rand by 1`.
- `round-robin-q` — LRU-style with `RRGCtx { next_replica_id }` global state.
- `least-wait-token-q` — `Select min by prefill_tokens(req, sctx)`.
- `bounded-most-hit-q` — `Filter (queued_tokens < BOUND) (max hit) (min prefill_tokens)`, with the attention-black-hole fallback fix.

**`vllm.rs`** — vLLM baseline
- `join-shortest-weight-q` — `Select min by 4·sctx.waiting + sctx.bs`. The "weight" qualifier names the formula's defining feature (waiting carries weight 4 vs weight 1 on bs); also serves as the catch-all default when no policy feature is selected.

**`bailian.rs`** — Bailian baseline
- `bailian-impl-q` — Per-component normalize then weighted-sample `(α, β, γ)` over `(hit_pct, 1-bs/M_bs, 1-tok/M_tok)`.

**`aibrix.rs`** — AIBrix baseline
- `aibrix-q` — Nested `Filter` (load-imbalance gate × stddev threshold) with tuple-keyed `(-hit_pct, bs)` selection.

**`dynamo.rs`** — AI-Dynamo baselines (PD-disaggregated formulas, ported as ablations)
- `dynamo-q` — Dynamo's **Decode-node** formula: `Select min by w·(new_tokens/block_size) + (new_blocks + decode_blocks)`.
- `dynamo-po-q` — Dynamo's **Prefill-node** formula ("po" = prefill-only node, NOT "uses new_tokens only"): `Select min by w·(prefill_tokens/block_size) + floor(prefill_blocks)`.

**`lmetric.rs`** — our system
- `lmetric-q` — `Select min by prefill_tokens · (bs+1)`.

**`preble/`** — Preble baseline
- `preble-q` — Dual-stage Preble: `Filter (match_pct > 0.5) (max match) (min preble_cost)`; SlidingWindowHistogram updated via `after_extra`.

**`llm_d/`** — llm-d baselines (single-scorer ablations + multi-scorer combos)
- `most-hit-q` — `Select max by hit_blocks(req, sctx)`. Native name for our port of llm-d's production baseline (`sim-epp-kvcache-config.yaml`: precise-prefix-cache-scorer w=10 + max-score-picker). Single-scorer + constant-weight + min-max-norm reduces to `argmax hit_blocks` (`most_hit.rs`).
- `least-waiting-q` — `Select min by sctx.waiting`. llm-d's `load-aware-scorer` and `queue-depth-scorer` as single-scorer ablations both collapse to this argmin (`least_waiting.rs`).
- `least-bs-q` — `Select min by sctx.bs`. Closest single-scorer port of llm-d's `running-requests-scorer`; honest about composite signal (`sctx.bs = running + queued` per §4.2, not pure RunningRequestsSize) (`least_bs.rs`).
- `least-active-q` — `Select min by sctx.all_tokens`. llm-d's `kv-cache-utilization-scorer` single-scorer ablation, cap-free (`1 − all_tokens/CAP` argmax = `all_tokens` argmin for any fixed CAP) (`least_active.rs`).
- `least-token-load-q` — `Select min by queued_tokens(sctx) + sctx.all_tokens`. llm-d's `token-load-scorer` single-scorer ablation (`least_token_load.rs`).
- `most-hit-load-q` — llm-d's two-scorer combo (precise-prefix-cache w=10 + load-aware w=1): per-component min-max-norm + weighted sum, argmax via `select_max_by`. Tunables in `metrics.rs::MOST_HIT_LOAD_W_*` (`most_hit_load.rs`).
- `most-hit-load-active-q` — three-scorer combo (above + kv-cache-utilization w=1, cap-free via `1 − norm(all_tokens)`). Tunables `MOST_HIT_LOAD_ACTIVE_W_*` (`most_hit_load_active.rs`).

llm-d policies that cannot be expressed in the DSL (e.g. session-aware) are NOT ported; reference: `workspace/llm-d-scheduler/`.

### KV Cache Tracking
- **RadixTree** (`radixtree` crate, specialization `RadixTreeBlockHash`): Patricia trie mapping token sequences to block hashes. The `radixtree` crate also holds the Verus-verified L0 spec (`verified.rs`, `verify` feature) and the L0..L3 lowering ladder under `benches/lowering_levels.rs`.
- **Hash Algorithms**: `default-hash-algo` (single u64) or `sha256-hash-algo` (4x u64 SHA256)
- **Implementations**: `radixtree-blockhash` (tree) or `hashtable-blockhash` (hash table)
- Each request carries `BlockHashState` for cache-aware routing
- **Eviction tracking**: Router receives `evicted_block_ids` from yaullm SSE and updates its RadixTree accordingly

### Replica State Machine (20+ states)
```
Inactive → LoadingPrefill → Prefill → NewPrefill (Zigzag)
                                       ↓
                                    OldPrefill → ShuttingPrefill
Prefill → MutatingToDecode → Decode → ShuttingDecode → Inactive
Broadcasting: NvlCasting, RdmaCasting, TanzCasting, RdmaSending, RdmaLoading
```

### Conditional Compilation (policy selection)
The scheduling policy is selected at compile time via Cargo features. Each policy file under `policies/` is gated by `#[cfg(feature = "<name>-q")]`. Exactly one policy feature should be enabled per build.

### Latency Simulator (feature-gated `simulator`, ORTHOGONAL to `<name>-q`)

`router/src/simulator/` is a per-replica latency-prediction subsystem. Two layers:
- **Inner regressor** (`predictor.rs`, `vidur_rf.rs`): offline-trained ML model (port of Vidur RandomForest from `tmp/blitz-infer-pack-sim/`) with online linear-regression correction (`LinregCorrected`). `Predictor::predict(&BatchForPredictor) -> f32` (ms).
- **Outer discrete-event simulator** (`rollout.rs`): rolls forward engine steps from the current `ScheduleContext`, calls the inner regressor per step, fills a `RolloutBuffer`. Stop condition: `waiting==∅ && chunked_prefill_in_progress==∅`. (RolloutBuffer types defined; full DES `query_sim` driver is a follow-up.)

**Piggyback mode (default and only mode today):** when built with `--features simulator,<name>-q` AND launched with `--enable-simulator`, the simulator observes the active `<name>-q` policy via two hooks: `simulator::on_sse(replica_index, &EngineStepOutput)` from `colocation.rs::completion_event_loop` (per SSE event) and `simulator::on_admit(replica_index, request_id)` from `policy_runner.rs::queue_task` (per admission, both Append and NextRequest paths). It emits Prometheus histograms (`simulator_predicted_ms`, `simulator_actual_ms`, `simulator_signed_error_ms`, `simulator_abs_error_ms`, `simulator_relative_error`). **It does not influence routing.** The `Policy` trait is untouched; PCtx is owned by a process-wide `OnceLock` initialised from `main.rs` at startup.

PCtx exposes three triggers (the public API): `query(candidate_id, input_length, hashes, sctx_prefix_hits) → RolloutGist` (Trigger A — speculative rollout used by the future `simulator-q`), `on_admit(&Entry)` (Trigger B — populates L1 mirror + sched waiting + L3 promote-or-drop), `on_sse(batch, &EngineStepOutput) → (predicted_ms, actual_ms)` (Trigger C — drives L2 calibration, sched.sync, mirror.apply_sse, F3 cross-check). Three-layer state model: L1 incremental mirror (V=ReqId trie, self-arbitrating evict), L2 online-corrected regressor, L3 single-slot ephemeral rollout. Plus auxiliary `SchedSnapshot` (per-request progress, cloned by `query`'s schedule loop). Lock order: `sched → mirror → ephemeral`. Load-bearing invariant: when the engine has prefill work in progress, the L3 buffer must be non-empty.

`query`'s schedule loop is a discrete-event simulator that clones `SchedSnapshot`, pushes the candidate to `waiting.back()` (FCFS), then per-slot: assigns the running decoders 1 token each, fills remaining `token_budget` with chunked prefill (continuing prefillers first, then pulling from waiting), calls the inner regressor for per-step latency, tracks the candidate's `prefill_begin_step` / `prefill_end_step` / `in_decode_step`, and stops one slot after the candidate enters DECODE. The composite cache-hit estimate is `max(SCtx.block_hash.get(hashes), mirror.prefix_match(hashes))` per A2's max-merge rule. See `.claude/memory/project_lmetric_predictor_design.md` §"Locked spec" + §"Phase 3" for the full state machine and Group A/B/C decisions.

**CLI flags** (all behind `simulator` feature; no-op otherwise):
- `--enable-simulator` — turn the subsystem on at runtime.
- `--simulator-cache-dir <PATH>` — directory containing `{op}_{hash}_predictions.csv` files. Default `/nvme/zkx/Modified_vidur/cache`.
- `--simulator-model-hash <HASH>` — Vidur model hash. Qwen2.5: `9f4b3b9a`, Llama3: `d29f0375`.
- `--simulator-num-layers <N>` — transformer block count (default 28).
- `--simulator-moe` — use MoE op set (`moe_linear` instead of dense MLP grids).
- `--simulator-learning-rate <LR>` — online linreg SGD step size (default 1e-4).

A future `simulator-q` policy that consumes `RolloutBuffer` for dispatch decisions will be added once piggyback validates the simulator's accuracy.

See `.claude/memory/project_lmetric_predictor_design.md` for the full design rationale and decision log.

## Build

```bash
# Default build (lmetric scoring policy)
cargo build -p router --features lmetric-q

# Pick any other policy by swapping the feature
cargo build -p router --features bounded-most-hit-q
cargo build -p router --features aibrix-q
```

`vllm-backend` is the only backend and is enabled implicitly by other features that depend on it; you do not normally need to pass it explicitly.

**Cargo workspace members**: `router`, `request-sim`, `rust-proto`, `policy-dsl`, `radixtree`

## Configuration

### CLI Arguments (router)
```
--max-concurrent-requests     (default: 128)
--max-batch-prefill-tokens    (default: 4096)
--max-waiting-tokens          (default: 20)
--max-total-tokens            (default: 2048)
--waiting-served-ratio        (default: 1.2)
--deployment                  ("disaggregation" | "colocation")
--deployment-config-path      (TOML path)
--tokens-prefilled-per-sec    (default: 13000)
--tokens-transferred-per-sec  (default: 20000)
--port                        (default: 3000)
--otlp-endpoint               (OpenTelemetry)
```

### TOML Config Structure
```toml
[runtime.raw]
config = [
    { app = "vllm1", background = true },
    { app = "blitz_scale", background = true },
    { app = "azure_client", background = false }
]
[app.vllm1]
executable = ["bash", "-c"]
extra_args = ["vllm serve /model --port 50180"]
envs = { CUDA_VISIBLE_DEVICES = "0" }
```

## Related Projects
- **[yaullm](https://github.com/blitz-serving/yaullm)** (branch `lmetric/step-reporter-v2`) — Patched vLLM engine; provides HTTP inference API + SSE metrics push at `/v1/metrics`

## License
Apache-2.0. Code derived from Hugging Face Text Generation Inference (TGI).

## Memory Policy

Project-level memory (operational lessons, deployment pitfalls, design decisions) MUST be stored in `.claude/memory/` within this repo, NOT in user-level `~/.claude/projects/` directories. This ensures all agents and sessions working on this project share the same knowledge.

## Doc-Code Consistency at Commit Time

Before every commit that changes code, scan **all related doc surfaces** and fold any required doc updates into the SAME commit. Doc surfaces drift coherently — the dynamo formula swap and the `queued_pre` type drift incidents (see [issue #11](https://github.com/blitz-serving/blitz-router/issues/11)) both involved a prior session writing the same error into code AND every sibling doc surface. The scan must therefore cover every surface a future reader might cite to verify the change, not just the files `git diff` shows touched.

Per-change-type checklist (apply when relevant):

- **Policy added / renamed / deleted** → `router/Cargo.toml` features, `router/src/policies/mod.rs` (module decl + re-export + TaskAssigner alias + catch-all exclusion list), `docs/dsl/policies.md` §2 listing + module-org table in §1, `CLAUDE.md` scheduling-policies list, file header doc, `.claude/memory/*.md` if a memory file references it.
- **Algorithm change inside a policy body** → file header doc, `docs/dsl/policies.md` §2 listing, `CLAUDE.md` one-liner, related `.claude/memory/*.md` if any.
- **New / renamed / removed named-fn or reducer** → `docs/dsl/schema.md` §5 + `docs/dsl/implementation.md` §2.1 rewrite table + §2.2 allowlist doc, `policy-dsl/src/check.rs` `ALLOWED_FNS`.
- **`Observation` field rename or removal from DSL surface** → `docs/dsl/schema.md` §4.2 (or remove the row if no longer DSL-canonical) + §5 Body / Reads columns referring to it + `docs/dsl/policies.md` §2 listings using it, file headers using it, `policy-dsl/src/check.rs` if relevant.
- **New cargo feature** → `router/Cargo.toml`, `mod.rs` catch-all exclusion list, `CLAUDE.md`, `docs/dsl/policies.md` §2 if applicable.
- **Tunable constant added** → `router/src/metrics.rs` (per `docs/dsl/schema.md` §10.4 convention), file header pointer, `CLAUDE.md` if user-facing.
- **Public API / CLI flag change** → `README.md`, `CLAUDE.md` Configuration section, related `.claude/memory/*.md`.

**Skills (load on-demand)**: invoke `/add-policy` when adding / renaming / deleting a policy — it walks the first checklist entry mechanically. Invoke `/verify-policy` when reviewing a `policy!` body for impl ↔ spec drift against its `docs/dsl/policies.md` §2 listing. Both skills reference this checklist as source of truth, so any change to the checklist must also be reflected in `.claude/skills/{add,verify}-policy.md`.

Two failure modes to handle differently:

1. **Push failure (your commit creates inconsistency)** — fold the doc fix INTO the same commit. Do not defer. A separate "doc cleanup" commit invites further drift.
2. **Pull failure (drift was already there before your commit, inherited from a prior session)** — fix it as a separate post-mortem commit AND open a GitHub issue documenting the swap-and-codify pattern (template: issue #11). Do NOT silently fix inherited drift inside an unrelated feature commit; that pattern is what created the original drift in the first place.

This rule is the **push-side** prevention. The **pull-side** complement (independent verification against upstream sources at audit time, e.g. `selector.rs:150` for Dynamo) lives in issue #11's reviewer guard. Both are needed; this rule is the stronger lever because it activates at every commit (no reliance on future-reader discipline) and lives in always-loaded context.
