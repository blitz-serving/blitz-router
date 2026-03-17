# BlitzScale Router - Distributed LLM Inference Router

## Overview
BlitzScale Router (blitz-router) is a research-grade distributed LLM inference router and scheduler, written in **Rust**. It routes client requests to backend engines (vLLM or BlitzLLM), manages KV cache state, and dynamically scales replicas. Extracted from `blitz-infer-pack` — the C++ inference engine (BlitzTransformer) lives separately and communicates via gRPC.

## Comparison with AI-Dynamo & AIBrix

| Aspect | BlitzScale | AI-Dynamo | AIBrix |
|--------|-----------|-----------|--------|
| Language | Rust | Python/C++ | Python/Go |
| P/D Disaggregation | Layer-wise split with Zigzag scaling | Co-located | Separated |
| KV Cache Routing | RadixTree prefix matching at router | Simplified | Cache-aware |
| Scaling | Zigzag (incremental layer loading) | Ring-based | Static disaggregation |
| Scheduling Policies | 8+ compile-time switchable | Fixed | Fixed |
| Broadcast | NVLink / RDMA / Tanz (tree+ring hybrid) | N/A | Specialized |
| Metrics | Comprehensive real-time (lmetric) | Basic | Limited telemetry |
| Config Polymorphism | 165 Cargo feature flags (zero-overhead) | Runtime config | Runtime config |

### Key Differentiators
1. **Zigzag Scaling**: Incrementally loads model layers on new replicas while old ones continue serving — no full model broadcast needed.
2. **Prefix Cache Integration**: Router's RadixTree tracks per-replica block hashes for O(log L) prefix matching, enabling cache-aware routing decisions.
3. **Compile-Time Polymorphism**: 165 feature flags allow different system configurations (scheduling, hashing, scaling) without runtime overhead.
4. **Request Migration**: Mid-stream request migration between replicas with batch state transfer.

## Architecture

```
Client Request → [Validation] → [Queue + BlockHashState] → [Replica Selection]
    → [Prefill Phase] → [KV Cache Tracking] → [Decode Phase] → [Response Streaming]
    → [Migration if needed]
```

### Deployment Modes (compile-time)
- **Colocation** (`vllm-backend`): Full model per replica, direct token generation
- **Disaggregation** (`blitzllm-backend`): Separate prefill/decode replicas, parameter transfer via NCCL/RDMA

## Project Structure

```
blitz-router/
├── router_v2/src/           # Rust router (~8,400 LOC)
│   ├── main.rs              # CLI args & entry point
│   ├── server.rs            # HTTP/gRPC server (Axum), /generate, /info, /health, /metrics
│   ├── infer.rs             # Inference orchestration (730 LOC)
│   ├── queue.rs             # Request queue & scheduling policies (1,653 LOC)
│   ├── kvcache.rs           # KV cache tracking with BlockHashState (1,373 LOC)
│   ├── radixtrie.rs         # Patricia trie for prefix matching (1,072 LOC)
│   ├── stub.rs              # gRPC client stubs to backends
│   ├── validation.rs        # Request validation
│   ├── vllmlet.rs           # vLLM backend integration
│   └── replica/             # Replica state machine & controllers (~4,500 LOC)
│       ├── config.rs         # DisaggregationConfig
│       ├── metrics.rs        # SystemMetric (AtomicUsize counters), 20+ replica states
│       ├── disaggregation.rs # P-D disaggregation controller
│       ├── colocation.rs     # Co-location controller
│       ├── steersman.rs      # Replica lifecycle management
│       └── cybernetics/      # Dynamic scaling planner & execution
│           ├── planner.rs    # ScalePlan generation (prefill/decode thresholds)
│           ├── exec_blitz.rs # Zigzag, multicast, NVLink/RDMA execution (114K!)
│           └── exec_serverless.rs
├── proto/generate.proto     # gRPC definitions (TextGenerationService)
├── rust-grpc/               # gRPC metadata injection
├── rust-proto/              # Protobuf generated Rust code
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

### Replica State Machine (20+ states)
```
Inactive → LoadingPrefill → Prefill → NewPrefill (Zigzag)
                                       ↓
                                    OldPrefill → ShuttingPrefill
Prefill → MutatingToDecode → Decode → ShuttingDecode → Inactive
Broadcasting: NvlCasting, RdmaCasting, TanzCasting, RdmaSending, RdmaLoading
```

### Scaling Mechanisms
- **Zigzag** (`impl_live`/`impl_live_pro`): Incremental layer-wise model loading
- **Multicast** (`impl_fast`/`impl_fast_pro`): One-to-many parameter broadcast
- **NVLink** (`impl_nvl`): Intra-node GPU-to-GPU broadcast
- **RDMA** (`impl_rdma`): Inter-node InfiniBand broadcast
- **Tanz** (`impl_tanz`): Hybrid tree+ring topology broadcast

### gRPC Protocol (proto/generate.proto)
Key RPCs on `TextGenerationService`:
- `Prefill`, `Decode`, `PrefillV2`, `DecodeV2`, `ZagPrefill` — inference
- `SendParams`, `RecvParams`, `LoadParams` — parameter transfer
- `Migrate`, `Immigrate`, `MigratePartial`, `ImmigratePartial` — request migration
- `NvlBroadcast`, `RdmaBroadcast`, `TanzBroadcast` — broadcast
- `Health`, `Info`, `ServiceDiscovery`, `ClearCache`, `FilterBatch`, `Warmup`

## Build

```bash
# Build the full workspace
cargo build --release

# Build router with specific features
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
- **[yaullm](https://github.com/blitz-serving/yaullm)** — Patched vLLM engine with step-level SSE metrics
- **[blitz-infer-pack](https://github.com/blitz-serving/blitz-infer-pack)** — Full system (router + C++ BlitzTransformer engine)

## License
Apache-2.0. Code derived from Hugging Face Text Generation Inference (TGI).
