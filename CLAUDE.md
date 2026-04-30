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
│   └── policies/            # Scheduling policies (one file per algorithm)
│       ├── mod.rs               # QueuePlusPlus trait, Deterministic/StochasticPolicy (~825 LOC)
│       ├── random.rs            # random-q
│       ├── round_robin.rs       # round-robin-q (stateful — RR counter)
│       ├── least_wait_token.rs  # least-wait-token-q
│       ├── bounded_most_hit.rs  # bounded-most-hit-q (cache-aware)
│       ├── shortest_q_weight.rs # join-shortest-q + variants
│       ├── lmetric.rs           # lmetric-q (multiplicative scoring)
│       ├── bailian.rs           # bailian-impl-q (linear combination + sample)
│       └── aibrix/prefix_cache.rs   # aibrix-q port
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

### Scheduling Policies (compile-time via Cargo features)

Each policy is one Cargo feature flag and exactly one file under `router/src/policies/`. Feature names are the source of truth — anything not in this list is stale.

- `random-q` — Uniform random replica selection (`policies/random.rs`).
- `round-robin-q` — Cyclic distribution; **stateful** (RR counter in `RRContext`) (`policies/round_robin.rs`).
- `join-shortest-q` — Shortest queue depth (uses `shortest_q_weight.rs`).
- `least-wait-token-q` — Minimize total waiting tokens, prospective (`policies/least_wait_token.rs`).
- `bounded-most-hit-q` — Max KV cache hits subject to prefill-token budget filter (`policies/bounded_most_hit.rs`).
- `bailian-impl-q` — Per-component normalize then sample with weights `(α, β, γ)` over `(hit_ratio, req_count_inv, token_count_inv)` (`policies/bailian.rs`).
- `aibrix-q` — Port of AIBrix prefix-cache routing: load-imbalance gate then sort-and-stddev-sample (`policies/aibrix/prefix_cache.rs`).
- `dynamo-q` — Dynamo logit `w·potential_prefill_blocks + decode_blocks`.
- `dynamo-decoupled-q` — Dynamo variant decoupling per-request prefill from engine state; pairs with `dynamo-q` for ablation.
- `lmetric-q` — Multiplicative score `(queued_prefill + new_prefill) × (bs+1)`; the lmetric paper's contribution (`policies/lmetric.rs`).
- `preble-q` — Preble cost-model port; under QueuePlusPlus simplifies to `new_prefill + all_tokens`.

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
