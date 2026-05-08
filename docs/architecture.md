# Architecture

A 10,000-foot view of `blitz-router`'s components and how they wire together.
This is a companion to `CLAUDE.md` (which is the operational reference) — if
you have never read this codebase before, start here.

> Renderers: every `mermaid` block below renders on GitHub, GitLab, mdBook,
> and Obsidian. Plain `cmark` will show them as fenced code; that is
> acceptable but you lose the diagrams.

> The three-layer view in §2 is the **conceptual** architecture. The
> on-disk layout under `router/src/` does not yet reflect it (everything
> is currently flat). Bringing the file tree into alignment with these
> layers is tracked as a separate refactor in
> [`refactor-plan.md`](refactor-plan.md).

## 1. System boundary

`blitz-router` is **only the routing layer**. It exchanges traffic with two
classes of external counterparts — neither lives in this repo, neither is
part of the system this document describes:

- **Inbound**: HTTP clients (OpenAI-style chat-completions or `/generate`).
- **Outbound**: inference engines, addressed over HTTP for `/generate` and
  subscribed over Server-Sent Events on `/v1/metrics`.

```mermaid
graph LR
    Cli["External HTTP clients"]
    subgraph Router_System["blitz-router (the system)"]
        Server["server.rs<br/>(Axum HTTP)"]
        Infer["infer.rs<br/>+ queue + policy"]
        Coloc["colocation.rs<br/>(per-replica event loops)"]
        Server --> Infer --> Coloc
    end
    Eng["External inference engines<br/>(N replicas, HTTP + SSE)"]

    Cli -- HTTP --> Server
    Coloc -- HTTP /generate --> Eng
    Eng -. SSE /v1/metrics .-> Coloc
```

- **HTTP request path** (solid lines): a client request enters at
  `server.rs`, is admitted by the scheduler, and is dispatched to one
  external engine; the streamed response flows back the same way.
- **SSE metrics path** (dotted line): every engine pushes one event per
  forward step on its `/v1/metrics` endpoint. `colocation.rs` consumes
  all engines' streams concurrently. **No gRPC anywhere.**

For concreteness: in lmetric the inbound clients are typically
`request-sim` (a Rust load generator) and the outbound engines are
`yaullm` (a patched vLLM). End-to-end paper experiments are orchestrated
by [MetricsTestRunner](https://github.com/blitz-serving/MetricsTestRunner).
None of these are part of `blitz-router` and none appear elsewhere in
this document's diagrams — they are listed here only so you know where
the live traffic actually originates and terminates.

## 2. Three-layer architecture

The system decomposes into three layers, each with a single
responsibility and a thin contract to the next:

```mermaid
flowchart TB
    classDef ext fill:#f5f5f5,stroke:#999,stroke-dasharray: 4 4,color:#666
    classDef front fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef middle fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    classDef back fill:#fff3e0,stroke:#e65100,color:#bf360c

    InCli["External HTTP clients"]:::ext
    OutEng["External inference engines<br/>(N replicas, HTTP + SSE)"]:::ext

    subgraph FRONT["FRONT — Gateway"]
        direction LR
        f_server["server<br/>(axum routes)"]:::front
        f_validation["validation<br/>(tokenize + check)"]:::front
        f_chat["chat_template"]:::front
        f_model["model_config"]:::front
        f_health["health"]:::front
    end

    subgraph MIDDLE["MIDDLE — Scheduler"]
        direction LR
        m_infer["infer<br/>(orchestrator)"]:::middle
        m_queue["queue +<br/>PolicyRunner&lt;P&gt;"]:::middle
        m_policies["policies<br/>(DSL-driven)"]:::middle
        m_coloc["colocation<br/>(per-replica loops)"]:::middle
        m_state["ScheduleContext<br/>+ LMetric + KVCache"]:::middle
        m_sim["simulator<br/>(piggyback, optional)"]:::middle
    end

    subgraph BACK["BACK — Engine driver"]
        direction LR
        b_trait["EngineClient +<br/>EngineStepReceiver<br/>(traits)"]:::back
        b_vllm["VllmClient<br/>(HTTP + SSE)"]:::back
        b_zmq["ZmqEngineClient<br/>(ZMQ, alt)"]:::back
    end

    InCli -- HTTP --> FRONT
    FRONT -- "ValidGenerateRequest +<br/>response_tx (mpsc)" --> MIDDLE
    MIDDLE -- "EngineClient::add_request" --> BACK
    BACK -- "HTTP /generate" --> OutEng
    OutEng -. "SSE /v1/metrics<br/>→ EngineStepOutput" .-> BACK
    BACK -. "EngineStepReceiver::recv_step" .-> MIDDLE
```

**Layer responsibilities**:

| Layer | What it owns | What it knows nothing about |
|---|---|---|
| **Front (Gateway)** | HTTP framing, OpenAI-compatible API surface, tokenizer rendering, request validation | which replica will run the request, how the engine speaks |
| **Middle (Scheduler)** | Admission, replica selection (policies), KV-cache state per replica, the dispatch loop, the SSE-consumption loop | HTTP wire format, the engine's transport (sees `EngineClient` trait only) |
| **Back (Engine driver)** | Engine transports (HTTP+SSE today, ZMQ alternative), translation between unified types (`EngineStepOutput`) and wire-specific payloads | scheduling policy, request validation, tokenization |

**Contract between layers** (the only types that cross):

| Boundary | Surface |
|---|---|
| Front → Middle | `ValidGenerateRequest` + a `mpsc::UnboundedSender<Result<InferStreamResponse, _>>` per request, packaged into a queued `Entry` |
| Middle → Back | `engine::EngineClient::add_request(req) -> Future<()>` (one direction) and `engine::EngineStepReceiver::recv_step() -> Future<EngineStepOutput>` (the other direction) |

These two surfaces are the only places the layers touch. Everything
else is internal to its layer.

## 3. Workspace layout

`blitz-router` is a Cargo workspace with five members:

| Member         | Role                                                          | Loaded by         |
|----------------|---------------------------------------------------------------|-------------------|
| `router/`      | The router itself: all three layers above                     | top-level binary  |
| `radixtree/`   | Patricia-trie crate (`BlockHash` trait + production impl + Verus L0 spec + L0..L3 lowering bench ladder) | `router/` middle layer |
| `policy-dsl/`  | Proc macro `policy! { … }` that lowers a DSL spec into `impl Policy for X` | `router/` middle layer |
| `rust-proto/`  | Protobuf type definitions used as **internal Rust types** (NOT a wire protocol) | `router/` |
| `request-sim/` | Git submodule — Rust load generator. Lives in this workspace only because the Cargo workspace lets developers `cargo run -p request-sim` from one checkout | standalone binary |

```mermaid
graph TD
    radixtree --> router
    policy_dsl["policy-dsl"] --> router
    rust_proto["rust-proto"] --> router
    radixtree -. used directly by .-> sim["router middle layer:<br/>simulator/"]
    router --- sim
```

## 4. Front layer — Gateway

**Responsibility**: be the HTTP face of the system. Accept OpenAI-style
and TGI-style requests, validate them, render chat templates, hand off
a `ValidGenerateRequest` plus a response channel to the scheduler.

| Module (current path) | Role                                                                          | LOC  |
|-----------------------|-------------------------------------------------------------------------------|------|
| `server.rs`           | Axum router. Endpoints: `/generate`, `/generate_stream`, `/v1/chat/completions`, `/info`, `/health`, `/metrics`, `/invocations` | 1143 |
| `validation.rs`       | `Validation` — fan-out of CPU-bound tokenization to a thread pool via `spawn_blocking`; round-robin task; produces `ValidGenerateRequest` | 467  |
| `chat_template.rs`    | Chat template rendering (Jinja-style or PyO3-backed when feature `python-chat-template` is on) | 340  |
| `model_config.rs`     | Auto-discovery of `config.json` / `tokenizer_config.json` at startup          | 81   |
| `health.rs`           | Health endpoint logic                                                         | 11   |

The gateway exposes **one** outbound surface: it calls
`Infer::generate(ValidGenerateRequest) -> impl Stream<InferStreamResponse>`.
That single call is the entire contract to the scheduler.

## 5. Middle layer — Scheduler

**Responsibility**: decide which replica runs each request, drive the
work, and maintain per-replica state (KV-cache view + load metrics).
This is by far the biggest layer.

| Module (current path) | Role                                                                          | LOC  |
|-----------------------|-------------------------------------------------------------------------------|------|
| `infer.rs`            | `Infer` struct — central orchestrator. Owns validation, queue, ColocationController, the concurrency `Semaphore`. Public method `generate(req)` is the gateway-facing handle. | 530  |
| `queue.rs`            | 8-line shim — re-exports `Entry`, `QueuePro`, `TaskAssigner` from `policies/`. Queue logic itself lives in `policies/policy_runner.rs`. | 8    |
| `colocation.rs`       | `ColocationController` + the two per-replica async loops: `work_event_loop` (admit + dispatch) and `completion_event_loop` (consume engine SSE) | 1105 |
| `kvcache.rs`          | `BlockHashState` (per-request prefix hash builder), `mod hashtable_block_hash` (alternative `BlockHash` impl selected by the `hashtable-blockhash` feature), the feature-gated `PrefixBlockHash` re-export | 1130 |
| `metrics.rs`          | `LMetric` per-replica scheduling counters, `ScheduleContext` (replica state = `LMetric` + `PrefixBlockHash`), tunable constants for policies (`BAILIAN_*`, `MOST_HIT_LOAD_*`, …). **Note**: this `metrics.rs` is unrelated to the Prometheus crate also called `metrics` — see refactor-plan.md for the rename | 216  |
| `statistic.rs`        | Statistics task spawned by `Infer` to dump `ScheduleContext` periodically     | 47   |
| `policies/`           | DSL-driven scheduling policies (18 of them, one per upstream baseline). One Cargo feature flag selects the active policy at compile time | dir  |
| `simulator/`          | Latency simulator (feature `simulator`). Piggyback observer over the active policy. Uses `radixtree::RadixTreeReqIdHash` for its L1 mirror | dir  |

Scheduler-internal dependency arrows:

```mermaid
graph TD
    infer["infer (Infer)"] --> validation_in["⇡ from gateway"]
    infer --> queue["queue<br/>= TaskAssigner = PolicyRunner&lt;P&gt;"]
    infer --> coloc["colocation<br/>(ColocationController)"]
    queue --> policies["policies/<br/>(P : Policy)"]
    coloc --> ec["⇣ to engine layer<br/>(EngineClient trait)"]
    coloc --> sctx["metrics.rs<br/>(ScheduleContext)"]
    policies --> sctx
    policies --> kv["kvcache<br/>(BlockHashState, PrefixBlockHash)"]
    kv --> rt["radixtree::<br/>RadixTreeBlockHash"]
    sim["simulator/<br/>(feature 'simulator')"] -. piggyback .-> coloc
    sim -. piggyback .-> policies
    sim --> rt2["radixtree::<br/>RadixTreeReqIdHash"]
```

Two scheduler-internal invariants worth knowing:
- **`completion_event_loop` is the only writer to a replica's
  `ScheduleContext.block_hash`.** Every reader takes the same `Mutex`.
- **Policies are picked at compile time.** Exactly one `<name>-q`
  Cargo feature is enabled per build → exactly one
  `TaskAssigner = PolicyRunner<XQ>` alias is monomorphised into the
  binary. There is no runtime policy switching.

### 5.1. The `Policy` abstraction

Every policy implements one trait:

```rust
// Currently at: router/src/policies/policy_trait.rs
// (in the refactor: router/src/scheduler/policies/policy_trait.rs)
pub trait Policy {
    type GlobalContext: Default + Send + Sync + 'static;

    fn schedule<'a>(
        entry: &'a Entry,
        all_sctx: &'a [Arc<Mutex<ScheduleContext>>],
        gctx: &'a mut Self::GlobalContext,
    ) -> impl Future<Output = Option<usize>> + Send + 'a;
}
```

You don't write this `impl` by hand. You write a DSL spec and the
`policy! { … }` proc macro from `policy-dsl/` lowers it. Example
(`router/src/policies/lmetric.rs`):

```text
policy lmetric-q (gctx: ()):
    Select min by sctx.prefill_tokens(req) · (sctx.bs + 1)
    after: default
```

The spec form is documented in `docs/dsl/policies.md` §2. The lowering
table (DSL → Rust) is in `docs/dsl/implementation.md` §2.1.

`PolicyRunner<P: Policy>` (`policy_runner.rs`) is the generic shim that:
1. Owns the per-replica commit buffers (lossless admission).
2. Spawns one `queue_task` per `append()` call.
3. Calls `P::schedule(...)` and pushes the entry into the chosen
   replica's buffer.

## 6. Back layer — Engine driver

**Responsibility**: speak the engine's wire protocol. Translate the
scheduler's abstract `add_request` / `recv_step` calls into HTTP+SSE
(or ZMQ) traffic against the external engines.

| Module (current path)  | Role                                                                          | LOC  |
|------------------------|-------------------------------------------------------------------------------|------|
| `engine_client.rs`     | The `EngineClient` + `EngineStepReceiver` traits + the unified `EngineStepOutput` type. The single point of dispatch from middle to back | 518  |
| `vllmlet.rs`           | `VllmClient` — reqwest-based HTTP client to a single engine; `/v1/metrics` SSE consumer that yields `VllmMetric` | 309  |
| `zmq_engine.rs`        | Alternate ZMQ transport (feature-gated `zmq-backend`)                          | 595  |

Two adapters, one trait:

```mermaid
graph TD
    coloc["⇡ from scheduler<br/>(colocation.rs)"] --> trait["EngineClient<br/>+ EngineStepReceiver<br/>(engine_client.rs)"]
    trait --> v["VllmClient<br/>(vllmlet.rs)"]
    trait --> z["ZmqEngineClient<br/>(zmq_engine.rs)"]
    v -- HTTP /generate --> ext["⇣ external engine"]
    ext -. SSE .-> v
    z -- ZMQ pub/sub --> ext
```

The trait surface is small enough to quote in full:

```rust
// engine_client.rs
pub trait EngineClient: Send {
    async fn add_request(&mut self, req: ValidGenerateRequest) -> Result<(), EngineClientError>;
    async fn abort_request(&mut self, id: u64) -> Result<(), EngineClientError>;
    fn get_error_rx(&mut self) -> mpsc::UnboundedReceiver<u64>;
    fn take_step_receiver(&mut self) -> Box<dyn EngineStepReceiver>;
}

pub trait EngineStepReceiver: Send {
    async fn recv_step(&mut self) -> Result<EngineStepOutput, EngineClientError>;
}
```

Adding a new engine transport = `impl`-ing this trait. The scheduler
needs zero changes.

## 7. Request lifecycle (across layers)

End-to-end happy path, `POST /v1/chat/completions` to streamed
response. The colours match §2's layer palette.

```mermaid
sequenceDiagram
    autonumber
    participant Cl as External client
    participant Sv as FRONT: server.rs
    participant Vl as FRONT: Validation
    participant If as MIDDLE: Infer
    participant Pr as MIDDLE: PolicyRunner&lt;P&gt;
    participant Po as MIDDLE: Policy::schedule
    participant Wq as MIDDLE: work_event_loop<br/>(replica i)
    participant Ec as BACK: EngineClient<br/>(VllmClient)
    participant En as External engine i

    Cl->>Sv: POST /v1/chat/completions
    Sv->>Vl: validate(request)
    Vl->>Vl: tokenize on threadpool
    Vl-->>Sv: ValidGenerateRequest
    Note right of Sv: Front→Middle handoff
    Sv->>If: Infer::generate(req)
    If->>Pr: queue.append(Entry)
    Pr->>Po: schedule(entry, all_sctx, gctx)
    Po-->>Pr: Some(replica_i)
    Pr->>Wq: commit_buffer[i].push_back(entry)
    Note right of Wq: Middle→Back handoff
    Wq->>Ec: add_request(req)
    Ec->>En: HTTP POST /generate
    loop streaming
        En-->>Ec: token chunk (HTTP body)
        Ec-->>If: InferStreamResponse
        If-->>Sv: tokio_stream::Stream
        Sv-->>Cl: SSE chunk
    end
```

## 8. Engine SSE consumption (the second half)

In parallel with the request path, every replica has a **second** async
task in the middle layer pulling SSE events from the back layer:

```mermaid
sequenceDiagram
    autonumber
    participant En as External engine i
    participant Vc as BACK: VllmClient<br/>(SSE consumer)
    participant Cl as MIDDLE: completion_event_loop<br/>(replica i)
    participant Sx as MIDDLE: ScheduleContext[i]<br/>(Mutex)

    loop forever
        En-->>Vc: SSE event /v1/metrics<br/>(VllmMetric)
        Vc-->>Cl: EngineStepOutput
        Cl->>Sx: lock + apply (insert/evict block hashes,<br/>update lmetric.tbt, …)
        Sx-->>Cl: ok
        Cl-->>Cl: per-request lifecycle bookkeeping
    end
```

Why this matters:
- The router's view of "what's in each replica's KV cache" — the
  `radixtree::RadixTreeBlockHash` inside `ScheduleContext.block_hash` —
  is updated **only** here. Scheduling policies read this state when
  picking a replica.
- The completion loop is the SOLE writer to `ScheduleContext.block_hash`.
  The work loop and the policy hot path are readers (under the same
  `Mutex`).
- The completion loop also drives the simulator's piggyback hook
  (`simulator::on_sse`) when feature `simulator` is enabled.

## 9. Cargo features as wiring switches

Several features rewire components at compile time. The complete list:

| Feature                     | Switches                                      | Layer |
|-----------------------------|-----------------------------------------------|-------|
| `radixtree-blockhash` *(default)* | `PrefixBlockHash = radixtree::RadixTreeBlockHash` | middle |
| `hashtable-blockhash`       | `PrefixBlockHash = HashTableBlockHash`        | middle |
| `default-hash-algo` *(default)* | `BackendBlockHash = [u64; 1]`                | middle |
| `sha256-hash-algo`          | `BackendBlockHash = [u64; 4]`                 | middle |
| `vllm-backend` *(default)*  | enables `VllmClient` (HTTP+SSE)               | back   |
| `zmq-backend`               | enables `ZmqEngineClient`                     | back   |
| `<name>-q` (one of 18)      | selects `TaskAssigner = PolicyRunner<XQ>`     | middle |
| `simulator`                 | builds the `simulator/` subsystem; piggyback observer activates only if `--enable-simulator` is also passed at runtime | middle |
| `python-chat-template`      | enables PyO3-based chat template renderer     | front  |
| `ngrok` *(default)*         | exposes the server through ngrok              | front  |

Rule of thumb: anything in the table swaps a concrete impl behind a
trait or type alias. There are no runtime-config branches for these.

## 10. Concurrency model

### Tasks per replica (n = number of engines)

```mermaid
graph LR
    subgraph startup["main.rs at startup"]
        S0["spawn validation round-robin"]
        S1["spawn statistic"]
        S2["spawn (per replica i)"]
    end
    S2 --> WL["work_event_loop[i]<br/>(admit / dispatch)"]
    S2 --> CL["completion_event_loop[i]<br/>(SSE consumer)"]
    WL <-. mpsc(cq_wqe) .-> CL
    CL --> SCX["ScheduleContext[i]<br/>(Mutex&lt;…&gt;)"]
    WL --> SCX
    POL["PolicyRunner queue_task<br/>(spawned per append)"] --> SCX
```

### Channels and locks

| Object | Type | Producers | Consumers |
|---|---|---|---|
| `Entry`'s `response_tx` | `mpsc::UnboundedSender<Result<InferStreamResponse, _>>` | per-replica work loops, completion loop | `Infer::generate` (per request) |
| `cq_wqe` (per replica) | `mpsc::Receiver<Entry>` | `PolicyRunner::queue_task` | `completion_event_loop` |
| `cq_error` (per replica) | `mpsc::UnboundedReceiver<u64>` | `EngineClient::get_error_rx()` | `completion_event_loop` |
| `all_commit_req_buffers[i]` | `VecDeque<(u64, Entry)>` (NO mutex — owned exclusively by `PolicyRunner::queue_task`) | filled by the `Append` branch of `queue_task` | drained by the `NextRequest` branch of `queue_task` (called from `work_event_loop[i]`) |
| `ScheduleContext[i]` | `Arc<Mutex<…>>` | completion loop (writes), work loop + policy (read/update) | as above |
| `RadixTreeBlockHash::mtx` | internal `SpinLock` | inside `radixtree::block_hash` only | inside the same |
| `Semaphore` | `Arc<tokio::sync::Semaphore>` | every accepted request decrements | every finished request increments |

Two invariants worth knowing:
- **`completion_event_loop` is the only writer to a replica's
  `ScheduleContext.block_hash`.**
- **One `Entry` lives in exactly one place at a time.** It is produced
  by `Validation`, owned by `PolicyRunner`'s commit buffer, then by
  the work loop's `entries` map, then dropped after the engine reports
  finish. Lifetime tracking is via `request_id` in debug asserts.

## 11. Optional: the latency simulator

When built with `--features simulator,<name>-q` AND launched with
`--enable-simulator`, the simulator (a middle-layer subsystem) observes
— but does not influence — routing:

```mermaid
graph LR
    SSE["completion_event_loop<br/>(middle)"] -- on_sse(replica, &EngineStepOutput) --> S
    Pol["PolicyRunner::queue_task<br/>(middle)"] -- on_admit(replica, request_id) --> S
    S["simulator::SIMULATOR<br/>(OnceLock&lt;Vec&lt;Arc&lt;PCtx&gt;&gt;&gt;)"] -- predict, calibrate --> Hist["Prometheus histograms<br/>simulator_predicted_ms,<br/>simulator_actual_ms,<br/>simulator_signed_error_ms,…"]
    S --> RIH["radixtree::<br/>RadixTreeReqIdHash<br/>(per-replica L1 mirror)"]
```

The mirror tracks per-request prefix membership (V = `ReqId`), distinct
from the engine's KV cache (V = `Bids`) tracked inside
`ScheduleContext.block_hash`. Both are radix tries, but the use cases —
multi-bid concurrent engine cache vs. simulator's in-flight reasoning
— have non-overlapping requirements, so they live as different
specializations inside the `radixtree` crate.

Design rationale for the three-layer `PCtx`, the calibration
loop, and the future `simulator-q` policy is in
`.claude/memory/project_lmetric_predictor_design.md`.

## 12. Where to go next

| You want to … | Read |
|---|---|
| Bring the directory layout into alignment with the three-layer model above | [`refactor-plan.md`](refactor-plan.md) |
| Add a new scheduling policy | `docs/dsl/policies.md` + invoke the `/add-policy` skill |
| Verify a policy spec matches its impl | invoke `/verify-policy` |
| Build / run end-to-end experiments | the [MetricsTestRunner](https://github.com/blitz-serving/MetricsTestRunner) repo |
| Understand the radix tree's verified-then-lowered chain | `radixtree/README.md` |
| Look up a specific configuration knob | `CLAUDE.md` §"Configuration" |
| Trace what a `policy! { … }` body lowers into | `docs/dsl/implementation.md` §2.1 |

The TLA+ specification for the colocation/CompletionLoop entry lifecycle
lives in `formal/tlaplus/`. It is not policy logic — it models the
work-loop ↔ completion-loop handoff to catch order-dependent races.
