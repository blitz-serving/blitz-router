# BlitzScale Router

Distributed LLM inference router written in Rust. Routes client requests to backend engines (vLLM or BlitzLLM), manages KV cache state via RadixTree prefix matching, and dynamically scales replicas.

## Build

```bash
# Build the full workspace
cargo build --release

# Build router with specific scheduling policy
cargo build -p router_v2 --features <policy>
```

Visit `router_v2/Cargo.toml` for available feature flags and scheduling policies.

The system configuration is `impl_blitz,impl_fast_pro,impl_live_pro`; the implemented baseline system is `impl_sllm,cache_replace`.

## Run

Modify the scripts in `scripts/e2e/` with your own PATH, logging directory and `address:port`. The e2e scripts launch backend, router, and client sequentially.

## Docs

Full docs: <https://blitz-serving.github.io/blitzscale-doc/>

## Project Structure

```
blitz-router/
├── router_v2/          # Rust router (~8,400 LOC)
├── rust-proto/         # Protobuf generated Rust code
├── request-sim/        # Request simulator (git submodule)
├── proto/              # gRPC proto definitions
├── scripts/            # e2e tests, batch utils, debug tools
├── docs/               # Documentation
└── *.py                # Analysis & visualization scripts
```

## Related Projects

- **[yaullm](https://github.com/blitz-serving/yaullm)** — Patched vLLM engine with step-level SSE metrics
- **[blitz-infer-pack](https://github.com/blitz-serving/blitz-infer-pack)** — Full system (router + C++ engine)

## Acknowledgements

These projects incorporate code from **Text Generation Inference (TGI)**
<https://github.com/huggingface/text-generation-inference>
Copyright 2022-present Hugging Face Inc.
Licensed under the **Apache License, Version 2.0**.
The full license text is provided in `LICENSES/Apache-2.0.txt`.
Modification details in `NOTICE` and individual source files.
