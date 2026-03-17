# BlitzScale Router - Distributed LLM Inference Router

## Overview
BlitzScale Router (blitz-router) is the **routing component** of the lmetric distributed LLM inference system. Written in Rust, it routes client requests to backend **yaullm** engines (a patched vLLM), manages KV cache state via RadixTree prefix matching, and dynamically scales replicas.

This repo was extracted from `blitz-infer-pack`, retaining only the Rust router. The C++ inference engine (BlitzTransformer) is not part of the lmetric system.

## Communication Protocol — IMPORTANT

**lmetric uses HTTP + SSE, NOT gRPC.**

The router supports two backend modes selected at compile time via Cargo features:

| | `vllm-backend` (lmetric) | `blitzllm-backend` (BlitzScale legacy) |
|---|---|---|
| **Engine** | yaullm (patched vLLM) | BlitzTransformer (C++) |
| **Inference transport** | HTTP (OpenAI-compatible API) | gRPC (`TextGenerationService`) |
| **Metrics transport** | SSE push (`/v1/metrics`) | gRPC response fields |
| **Entry point** | `vllmlet.rs` → `VllmClient` | `stub.rs` → `Stub` (gRPC) |
| **Used by lmetric?** | **YES** | No |

### Why proto/ and rust-proto/ still exist
The protobuf-generated types (`Tokens`, `GeneratedText`, `Batch`, `Request`, `CachedBatch`, etc.) are used as **internal data structures** throughout the router (queue, infer, validation, replica) regardless of backend mode. They are NOT used as a wire protocol in lmetric — the actual transport is HTTP/SSE via `VllmClient` in `vllmlet.rs`.

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
| Scheduling Policies | 8+ compile-time switchable | Fixed | Fixed |
| Metrics | Real-time SSE push per engine step | Basic | Limited telemetry |
| Config Polymorphism | Cargo feature flags (zero-overhead) | Runtime config | Runtime config |

## Project Structure

```
blitz-router/
├── router_v2/src/           # Rust router (~8,400 LOC)
│   ├── main.rs              # CLI args & entry point
│   ├── server.rs            # HTTP server (Axum): /generate, /info, /health, /metrics
│   ├── infer.rs             # Inference orchestration (730 LOC)
│   ├── queue.rs             # Request queue & scheduling policies (1,653 LOC)
│   ├── kvcache.rs           # KV cache tracking with BlockHashState (1,373 LOC)
│   ├── radixtrie.rs         # Patricia trie for prefix matching (1,072 LOC)
│   ├── vllmlet.rs           # ** yaullm/vLLM HTTP+SSE backend (lmetric path) **
│   ├── stub.rs              # gRPC stubs (blitzllm-backend only, NOT lmetric)
│   ├── validation.rs        # Request validation
│   └── replica/             # Replica state machine & controllers (~4,500 LOC)
│       ├── config.rs         # DisaggregationConfig
│       ├── metrics.rs        # SystemMetric (AtomicUsize counters), 20+ replica states
│       ├── disaggregation.rs # P-D disaggregation controller
│       ├── colocation.rs     # Co-location controller (vllm-backend)
│       ├── steersman.rs      # Replica lifecycle management
│       └── cybernetics/      # Dynamic scaling planner & execution
│           ├── planner.rs    # ScalePlan generation
│           ├── exec_blitz.rs # Scaling execution (114K)
│           └── exec_serverless.rs
├── proto/generate.proto     # Protobuf type definitions (used as internal data structures)
├── rust-proto/              # Protobuf codegen (internal types, NOT wire protocol in lmetric)
├── rust-grpc/               # gRPC metadata injection (blitzllm-backend only)
├── request-sim/             # Request simulator (git submodule, main branch)
├── config/                  # 39 TOML configs (lmetric*, metrics_*, dense_*, eval_*)
├── scripts/                 # e2e tests, batch utils, debug tools
└── tokenizer/               # Tokenizer library
```

## Key Components

### Scheduling Policies (compile-time via features)
- `random-q`: Random replica selection
- `round-robin-q`: Cyclic distribution
- `join-shortest-q`: Shortest queue depth
- `least-wait-token-q`: Minimize total waiting tokens
- `bounded-most-hit-q`: Prefer replicas with more KV cache hits
- `join-shortest-q-tuple`: Multi-objective optimization
- `join-shortest-q-weight`: Weighted queue depth
- `bailian-impl-q`: Proprietary algorithm

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

### Conditional Compilation (backend selection)
The backend is selected at compile time. Key `#[cfg]` guards:
- `main.rs`: `VllmClient::new()` (vllm-backend) vs `Stub::connect()` (blitzllm-backend)
- `infer.rs`: `Infer` struct definition differs per backend
- `server.rs`: Backend-specific imports and handler parameters

## Build

```bash
# lmetric build (vllm-backend with a scheduling policy)
cargo build -p router_v2 --features vllm-backend,join-shortest-q

# Full feature build (legacy, for blitzllm + scaling)
cargo build -p router_v2 --features impl_blitz,impl_fast_pro,impl_live_pro
```

**Cargo workspace members**: `tokenizer`, `router_v2`, `request-sim`, `rust-grpc`, `rust-proto`

## Configuration

### CLI Arguments (router_v2)
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
