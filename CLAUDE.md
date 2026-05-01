# BlitzScale Router - Distributed LLM Inference Router

## Overview
BlitzScale Router (blitz-router) is the **routing component** of the lmetric distributed LLM inference system. Written in Rust, it routes client requests to backend **yaullm** engines (a patched vLLM), manages KV cache state via RadixTree prefix matching, and dynamically scales replicas.

This repo was extracted from `blitz-infer-pack`, retaining only the Rust router. The C++ inference engine (BlitzTransformer) is not part of the lmetric system.

## Communication Protocol — IMPORTANT

**lmetric uses HTTP + SSE, NOT gRPC.**

The legacy `blitzllm-backend` mode (with C++ BlitzTransformer engine over gRPC) has been removed. Only `vllm-backend` remains; it is the default and effectively non-optional. Entry point: `vllmlet.rs` → `VllmClient` (HTTP) plus `/v1/metrics` SSE consumption.

### Why proto/ and rust-proto/ still exist
The protobuf-generated types (`Tokens`, `GeneratedText`, `Batch`, `Request`, `CachedBatch`, etc.) are used as **internal data structures** throughout the router (queue, infer, validation, colocation) regardless of backend. They are NOT used as a wire protocol — the actual transport is HTTP/SSE via `VllmClient` in `vllmlet.rs`. `rust-grpc` is similarly vestigial.

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
| Scheduling Policies | 11 compile-time switchable | Fixed | Fixed |
| Metrics | Real-time SSE push per engine step | Basic | Limited telemetry |
| Config Polymorphism | Cargo feature flags (zero-overhead) | Runtime config | Runtime config |

## Project Structure

> **Note**: As of commit 2327a08, the crate `router_v2/` was renamed to `router/`. Anything that still says `router_v2` in scripts or older notes is stale.

```
blitz-router/
├── router/src/              # Rust router (~14,000 LOC)
│   ├── main.rs              # CLI args & entry point
│   ├── lib.rs               # Crate root
│   ├── server.rs            # HTTP server (Axum): /generate, /info, /health, /metrics (~1,140 LOC)
│   ├── infer.rs             # Inference orchestration (~530 LOC)
│   ├── queue.rs             # Request queue scaffolding (~380 LOC; policy logic now in policies/)
│   ├── kvcache.rs           # KV cache tracking with BlockHashState (~2,020 LOC)
│   ├── radixtrie.rs         # Patricia trie for prefix matching (~1,470 LOC)
│   ├── verified_radix.rs    # Cross-checked RadixTree implementation
│   ├── colocation.rs        # Co-location controller (formerly replica/colocation.rs) (~1,120 LOC)
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
│   └── policies/            # Scheduling policies — DSL-driven
│       ├── mod.rs               # ~160 LOC: Entry, QueuePro, TaskAssigner aliases
│       ├── policy_trait.rs      # `Policy` trait (5 lines, lowering target)
│       ├── policy_runner.rs     # `PolicyRunner<P: Policy>` queue runner
│       ├── dsl_runtime.rs       # combinators, reducers, named pure fns,
│       │                        # `apply_default_after`, Observation schema
│       ├── simple.rs            # random-q, round-robin-q, join-shortest-q,
│       │                        # join-shortest-q-weight, least-wait-token-q,
│       │                        # bounded-most-hit-q (each is 1–10 line `policy!`)
│       ├── lmetric.rs           # lmetric-q
│       ├── bailian.rs           # bailian-impl-q
│       ├── aibrix.rs            # aibrix-q
│       ├── dynamo.rs            # dynamo-q + dynamo-po-q (T1 + T2 ablation)
│       └── preble/              # preble-q + cost_model/histogram/router utils
├── policy-dsl/              # ~150 LOC proc-macro: parser + lint + lowering
│   └── src/{lib,ast,parse,check,lower}.rs
├── docs/dsl-schema.md       # DSL spec (§1–§13)
├── proto/generate.proto     # Protobuf type definitions (used as internal data structures)
├── rust-proto/              # Protobuf codegen (internal types only)
├── rust-grpc/               # gRPC metadata injection (vestigial)
├── request-sim/             # Request simulator (git submodule, main branch)
├── tokenizer/               # Tokenizer library
├── formal/tlaplus/          # TLA+ spec — colocation/CompletionLoop entry lifecycle (NOT a policy spec)
├── config/                  # 39+ TOML configs (lmetric*, metrics_*, dense_*, eval_*)
├── exps/                    # Experiment harness & generated configs
├── scripts/                 # e2e tests, batch utils, debug tools
└── docs/                    # reproduce.md
```

The legacy `replica/` subdirectory and the `cybernetics/` planner code described in earlier revisions of this file have been removed; their relevant pieces are flattened into the top-level `router/src/` files (`colocation.rs`, `metrics.rs`, `engine_client.rs`).

## Key Components

### Scheduling Policies (DSL-driven, compile-time via Cargo features)

Each policy is one Cargo feature flag plus a `policy! { ... }` invocation in `router/src/policies/` that the `policy-dsl/` proc macro lowers into an `impl Policy for X { fn schedule(...) }` block. The macro enforces an allowlist lint (`docs/dsl-schema.md` §13.2) so the impl always corresponds 1:1 to a paper-form DSL listing (§8) via the rewrite table (§13.1). `PolicyRunner<P: Policy>` is the dispatch shim. Feature names are the source of truth — anything not in this list is stale.

- `random-q` — `Select rand by 1` (`simple.rs`).
- `round-robin-q` — LRU-style with `RRGCtx { next_replica_id }` global state (`simple.rs`).
- `join-shortest-q` — vLLM `4·waiting + bs` (`simple.rs`).
- `join-shortest-q-weight` — same formula, separate cargo flag (`simple.rs`).
- `least-wait-token-q` — `Select min by prefill_tokens(req, sctx)` (`simple.rs`).
- `bounded-most-hit-q` — `Filter (queued_tokens < BOUND) (max hit) (min prefill_tokens)`, with the attention-black-hole fallback fix (`simple.rs`).
- `most-hit-q` — `Select max by hit_blocks(req, sctx)`. Native name for our port of llm-d's production baseline (`sim-epp-kvcache-config.yaml`: precise-prefix-cache-scorer w=10 + max-score-picker). Single-scorer + constant-weight + min-max-norm reduces to `argmax hit_blocks` (`most_hit.rs`).
- `least-waiting-q` — `Select min by sctx.waiting`. llm-d's `load-aware-scorer` and `queue-depth-scorer` as single-scorer ablations both collapse to this argmin (`least_waiting.rs`).
- `least-bs-q` — `Select min by sctx.bs`. Closest single-scorer port of llm-d's `running-requests-scorer`; honest about composite signal (`sctx.bs = running + queued` per §4.2, not pure RunningRequestsSize) (`least_bs.rs`).
- `least-active-q` — `Select min by sctx.all_tokens`. llm-d's `kv-cache-utilization-scorer` single-scorer ablation, cap-free (`1 − all_tokens/CAP` argmax = `all_tokens` argmin for any fixed CAP) (`least_active.rs`).
- `least-token-load-q` — `Select min by queued_tokens(sctx) + sctx.all_tokens`. llm-d's `token-load-scorer` single-scorer ablation (`least_token_load.rs`).
- `most-hit-load-q` — llm-d's two-scorer combo (precise-prefix-cache w=10 + load-aware w=1): per-component min-max-norm + weighted sum, argmax via `select_max_by`. Tunables in `metrics.rs::MOST_HIT_LOAD_W_*` (`most_hit_load.rs`).
- `most-hit-load-active-q` — three-scorer combo (above + kv-cache-utilization w=1, cap-free via `1 − norm(all_tokens)`). Tunables `MOST_HIT_LOAD_ACTIVE_W_*` (`most_hit_load_active.rs`).
- `bailian-impl-q` — Per-component normalize then weighted-sample `(α, β, γ)` over `(hit_pct, 1-bs/M_bs, 1-tok/M_tok)` (`bailian.rs`).
- `aibrix-q` — Nested `Filter` (load-imbalance gate × stddev threshold) with tuple-keyed `(-hit_pct, bs)` selection (`aibrix.rs`).
- `dynamo-q` — Dynamo's **Decode-node** formula: `Select min by w·(new_tokens/block_size) + (new_blocks + decode_blocks)` (`dynamo.rs`).
- `dynamo-po-q` — Dynamo's **Prefill-node** formula ("po" = prefill-only node, NOT "uses new_tokens only"): `Select min by w·(prefill_tokens/block_size) + floor(prefill_blocks)`. (Renamed from legacy `dynamo-decoupled-q`.)
- `lmetric-q` — `Select min by prefill_tokens · (bs+1)` (`lmetric.rs`).
- `preble-q` — Dual-stage Preble: `Filter (match_pct > 0.5) (max match) (min preble_cost)`; SlidingWindowHistogram updated via `after_extra` (`preble/`).

Earlier revisions referred to `join-shortest-q-tuple` and `join-shortest-q-weight` — those are not Cargo features and have been superseded by `aibrix-q` and `shortest_q_weight.rs` respectively.

### KV Cache Tracking
- **RadixTree** (`radixtrie.rs`): Patricia trie mapping token sequences to block hashes
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

## Build

```bash
# Default build (lmetric scoring policy)
cargo build -p router --features lmetric-q

# Pick any other policy by swapping the feature
cargo build -p router --features bounded-most-hit-q
cargo build -p router --features aibrix-q
```

`vllm-backend` is the only backend and is enabled implicitly by other features that depend on it; you do not normally need to pass it explicitly. The legacy `blitzllm-backend`, `impl_blitz`, `impl_fast_pro`, `impl_live_pro` features no longer exist.

**Cargo workspace members**: `tokenizer`, `router`, `request-sim`, `rust-grpc`, `rust-proto`

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

## Helper Scripts
- `align_timestamp.py` — Synchronize log timestamps
- `cache_miss_rate.py` — KV cache hit/miss rate analysis
- `process_client_log.py` — Client trace parsing
- `extract_scale_events.py` — Extract scaling decisions from logs
- `plot_cache_miss.py` — Cache miss visualization
- `calculate_hyper_params.py` — Parameter tuning
- `scripts/e2e/eval_azure.sh` — E2E evaluation orchestrator

## Related Projects
- **[yaullm](https://github.com/blitz-serving/yaullm)** (branch `lmetric/step-reporter-v2`) — Patched vLLM engine; provides HTTP inference API + SSE metrics push at `/v1/metrics`
- **[blitz-infer-pack](https://github.com/blitz-serving/blitz-infer-pack)** — Original monorepo (router + C++ BlitzTransformer engine, gRPC-based)

## License
Apache-2.0. Code derived from Hugging Face Text Generation Inference (TGI).

## Memory Policy

Project-level memory (operational lessons, deployment pitfalls, design decisions) MUST be stored in `.claude/memory/` within this repo, NOT in user-level `~/.claude/projects/` directories. This ensures all agents and sessions working on this project share the same knowledge.

## Doc-Code Consistency at Commit Time

Before every commit that changes code, scan **all related doc surfaces** and fold any required doc updates into the SAME commit. Doc surfaces drift coherently — the dynamo formula swap and the `queued_pre` type drift incidents (see [issue #11](https://github.com/blitz-serving/blitz-router/issues/11)) both involved a prior session writing the same error into code AND every sibling doc surface. The scan must therefore cover every surface a future reader might cite to verify the change, not just the files `git diff` shows touched.

Per-change-type checklist (apply when relevant):

- **Policy added / renamed / deleted** → `router/Cargo.toml` features, `router/src/policies/mod.rs` (module decl + re-export + TaskAssigner alias + catch-all exclusion list), `docs/dsl-schema.md` §8 listing + count, `CLAUDE.md` scheduling-policies list, file header doc, `.claude/memory/*.md` if a memory file references it.
- **Algorithm change inside a policy body** → file header doc, `docs/dsl-schema.md` §8 listing, `CLAUDE.md` one-liner, related `.claude/memory/*.md` if any.
- **New / renamed / removed named-fn or reducer** → `docs/dsl-schema.md` §5 + §13.1 rewrite table + §13.2 allowlist doc, `policy-dsl/src/check.rs` `ALLOWED_FNS`.
- **`Observation` field rename or removal from DSL surface** → `docs/dsl-schema.md` §4.2 (or remove the row if no longer DSL-canonical) + §5 Body / Reads columns referring to it + §8 listings using it, file headers using it, `policy-dsl/src/check.rs` if relevant.
- **New cargo feature** → `router/Cargo.toml`, `mod.rs` catch-all exclusion list, `CLAUDE.md`, `docs/dsl-schema.md` §8 if applicable.
- **Tunable constant added** → `router/src/metrics.rs` (per dsl-schema.md §12.4 convention), file header pointer, `CLAUDE.md` if user-facing.
- **Public API / CLI flag change** → `README.md`, `CLAUDE.md` Configuration section, related `.claude/memory/*.md`.

Two failure modes to handle differently:

1. **Push failure (your commit creates inconsistency)** — fold the doc fix INTO the same commit. Do not defer. A separate "doc cleanup" commit invites further drift.
2. **Pull failure (drift was already there before your commit, inherited from a prior session)** — fix it as a separate post-mortem commit AND open a GitHub issue documenting the swap-and-codify pattern (template: issue #11). Do NOT silently fix inherited drift inside an unrelated feature commit; that pattern is what created the original drift in the first place.

This rule is the **push-side** prevention. The **pull-side** complement (independent verification against upstream sources at audit time, e.g. `selector.rs:150` for Dynamo) lives in issue #11's reviewer guard. Both are needed; this rule is the stronger lever because it activates at every commit (no reliance on future-reader discipline) and lives in always-loaded context.
