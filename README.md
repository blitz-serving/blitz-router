# BlitzScale Router

Distributed LLM inference router written in Rust. Routes client requests to backend **yaullm** engines (a patched vLLM) over HTTP+SSE, manages KV cache state via RadixTree prefix matching, and dynamically scales replicas.

## Build

```bash
# Default build (lmetric scheduling policy)
cargo build -p router --features lmetric-q

# Pick any other policy by swapping the feature
cargo build -p router --features bounded-most-hit-q
cargo build -p router --features aibrix-q
```

See `router/Cargo.toml` for the full list of policy feature flags.

## Run

Modify the scripts in `scripts/e2e/` with your own PATH, logging directory and `address:port`. The e2e scripts launch backend, router, and client sequentially.

## Docs

Full docs: <https://blitz-serving.github.io/blitzscale-doc/>

## Project Structure

```
blitz-router/
├── router/             # Rust router (~14,000 LOC)
├── rust-proto/         # Protobuf generated Rust code (internal types)
├── request-sim/        # Request simulator (git submodule)
├── proto/              # Protobuf type definitions (internal data structures)
├── scripts/            # e2e tests, batch utils
└── docs/               # Documentation
```

## Related Projects

- **[yaullm](https://github.com/blitz-serving/yaullm)** — Patched vLLM engine with step-level SSE metrics

## Acknowledgements

These projects incorporate code from **Text Generation Inference (TGI)**
<https://github.com/huggingface/text-generation-inference>
Copyright 2022-present Hugging Face Inc.
Licensed under the **Apache License, Version 2.0**.
The full license text is provided in `LICENSES/Apache-2.0.txt`.
Modification details in `NOTICE` and individual source files.
