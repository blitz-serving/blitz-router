# FRONT layer — Gateway

> Back to [`README.md`](README.md). See also: [`request-lifecycle.md`](request-lifecycle.md) for how a request flows through here into MIDDLE.

**Responsibility**: be the HTTP face of the system. Accept OpenAI-style
chat-completions requests, validate them, render chat templates, hand off
a `ValidGenerateRequest` plus a response channel to the scheduler.

## Modules

| Module (current path) | Role                                                                          | LOC  |
|-----------------------|-------------------------------------------------------------------------------|------|
| `server.rs`           | Axum router. Canonical endpoint: `POST /v1/chat/completions` (OpenAI-compatible). Legacy TGI URLs (`/`, `/generate`, `/generate_stream`, `/invocations`) are tombstoned to a single `tgi_deprecated` handler that returns `HTTP 410 Gone`. Plus `/info`, `/health`, `/metrics` | ~755 |
| `validation.rs`       | `Validation` — fan-out of CPU-bound tokenization to a thread pool via `spawn_blocking`; round-robin task; produces `ValidGenerateRequest` | 467  |
| `chat_template.rs`    | Chat template rendering (Jinja-style or PyO3-backed when feature `python-chat-template` is on) | 340  |
| `model_config.rs`     | Auto-discovery of `config.json` / `tokenizer_config.json` at startup          | 81   |
| `health.rs`           | Health endpoint logic                                                         | 11   |

## Outbound surface

The gateway exposes **one** outbound surface to the rest of the system:

```rust
Infer::generate(ValidGenerateRequest) -> impl Stream<InferStreamResponse>
```

That single call is the entire contract to the scheduler. See
[`README.md`](README.md) §2.5 for the full list of inter-layer
contracts.
