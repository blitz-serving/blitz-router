# Architecture

This document has been split into the [`architecture/`](architecture/) directory
for selective reading. Start with [`architecture/README.md`](architecture/README.md);
it covers the system boundary, the three-layer architecture, the sidecar
pattern, and the inter-layer contracts, plus an index pointing at the
per-topic files.

| Topic | File |
|---|---|
| Overview, system boundary, three-layer architecture, sidecar pattern, contracts | [`architecture/README.md`](architecture/README.md) |
| Workspace members + Cargo features | [`architecture/workspace-and-features.md`](architecture/workspace-and-features.md) |
| FRONT (Gateway) | [`architecture/front-gateway.md`](architecture/front-gateway.md) |
| MIDDLE (Scheduler) | [`architecture/middle-scheduler.md`](architecture/middle-scheduler.md) |
| BACK (Engine driver) | [`architecture/back-engine-driver.md`](architecture/back-engine-driver.md) |
| Request lifecycle + SSE consumption | [`architecture/request-lifecycle.md`](architecture/request-lifecycle.md) |
| Concurrency model | [`architecture/concurrency-model.md`](architecture/concurrency-model.md) |
| Latency simulator | [`architecture/latency-simulator.md`](architecture/latency-simulator.md) |
