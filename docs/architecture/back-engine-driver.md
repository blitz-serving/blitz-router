# BACK layer — Engine driver

> Back to [`README.md`](README.md). See also: [`request-lifecycle.md`](request-lifecycle.md) for the request flow that hits this layer, [`middle-scheduler.md`](middle-scheduler.md) for the upstream MIDDLE that hands work to BACK.

**Responsibility**: **execute** the scheduler's decisions. Pull the
already-routed `Entry` from `PolicyRunner`'s per-replica buffer, drive
it through the engine via `EngineClient::add_request`, consume the
engine's SSE step stream, own the in-flight request lifecycle, and
**write the observed engine state back to the `ScheduleContext`
sidecar** in MIDDLE.

BACK has no policy logic. It does not pick replicas — by the time
an `Entry` reaches a `work_event_loop[i]`, MIDDLE has already
committed it to replica `i`. BACK's only freedom is the engine
transport (HTTP+SSE today, ZMQ alternative).

## Modules

| Module (current path)  | Role                                                                          | LOC  |
|------------------------|-------------------------------------------------------------------------------|------|
| `colocation.rs`        | `ColocationController` + the two per-replica async loops: `work_event_loop` (drains the commit buffer via `next_request(i)`, calls `EngineClient::add_request`, owns the in-flight `entries` map) and `completion_event_loop` (consumes engine SSE → `EngineStepOutput`, writes to the `ScheduleContext` sidecar, drives request lifecycle bookkeeping) | 1105 |
| `engine_client.rs`     | The `EngineClient` + `EngineStepReceiver` traits + the unified `EngineStepOutput` type. The single point of dispatch from `colocation` to a transport adapter | 518  |
| `vllmlet.rs`           | `VllmClient` — reqwest-based HTTP client to a single engine; `/v1/metrics` SSE consumer that yields `VllmMetric` | 309  |
| `zmq_engine.rs`        | Alternate ZMQ transport (feature-gated `zmq-backend`)                          | 595  |

## Back-internal containment + wiring

```mermaid
flowchart TB
    classDef back fill:#fff3e0,stroke:#e65100,color:#bf360c
    classDef sidecar fill:#fff9c4,stroke:#f57c00,color:#bf360c
    classDef ext fill:#f5f5f5,stroke:#999,stroke-dasharray:4 4,color:#666
    classDef seam fill:#fafafa,stroke:#bbb,stroke-dasharray:4 4,color:#666

    fromMid["⇡ from MIDDLE<br/>BACK pulls via<br/>PolicyRunner::next_request(i)"]:::seam
    toMid["⇣ to MIDDLE (sidecar write)<br/>insert/evict block hashes,<br/>lmetric.tbt"]:::seam
    extEng["⇣ External engine"]:::ext

    subgraph coloc["colocation::ColocationController — N replica pairs"]
        direction TB
        cc_work["work_event_loop[i]<br/>(N tasks: drain buffer,<br/>call add_request,<br/>own in-flight entries map)"]:::back
        cc_done["completion_event_loop[i]<br/>(N tasks: consume SSE,<br/>WRITE to sidecar,<br/>drive request lifecycle)"]:::back
        cc_work <-. mpsc(cq_wqe) .-> cc_done
    end

    subgraph trait_layer["the transport-adapter trait surface"]
        b_trait["engine_client::EngineClient<br/>+ engine_client::EngineStepReceiver<br/>(EngineStepOutput unified type)"]:::back
    end

    subgraph adapters["impls (one selected by feature flag)"]
        direction LR
        b_vllm["vllmlet::VllmClient<br/>(reqwest HTTP +<br/>eventsource-client SSE)"]:::back
        b_zmq["zmq_engine::ZmqEngineClient<br/>(ZMQ, feature 'zmq-backend')"]:::back
    end

    %% Hot-path: work loop drains commit buffer and dispatches
    fromMid ==> cc_work
    cc_work == "EngineClient::add_request" ==> b_trait

    b_trait -- impl --> b_vllm
    b_trait -- impl --> b_zmq

    b_vllm == "POST /v1/completions" ==> extEng
    b_zmq == "ZMQ pub/sub" ==> extEng

    %% SSE return path: adapter → completion loop → sidecar
    extEng -. "SSE /v1/metrics" .-> b_vllm
    extEng -. "ZMQ messages" .-> b_zmq
    b_vllm -. "recv_step → EngineStepOutput" .-> cc_done
    b_zmq -. "recv_step → EngineStepOutput" .-> cc_done

    %% Sidecar write — the BACK→MIDDLE telemetry back-flow
    cc_done == "WRITE" ==> toMid
```

## The engine-adapter trait surface

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

Adding a new engine transport = `impl`-ing this trait. Neither MIDDLE
nor `colocation` change.
