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

End-to-end tests (router + yaullm + request-sim orchestration) live in the [MetricsTestRunner](https://github.com/blitz-serving/MetricsTestRunner) repo, not here.

## OSDI'26 Artifact Evaluation

Paper: [Simple is Better: Multiplication May Be All You Need for LLM
Request Scheduling](https://arxiv.org/pdf/2603.15202) (arXiv:2603.15202).

The AE workflow is documented in [`ae/README.md`](ae/README.md). It explains
how to prepare the required repositories, rerun the MetricsTestRunner
experiments that produce `client.jsonl` and router logs, and regenerate the
paper evaluation figures from `${AE_ROOT}/xmetric-plots/figs`. Use
`AE_ROOT` for the artifact root directory, `blitz-router` branch
`osdi26-ae-workflow`, `yaullm` branch
`lmetric/step-reporter-v2`, `MetricsTestRunner` branch `ae`, and
`xmetric-plots` branch `ae`.

To set up the required repositories:

```bash
cd "${AE_ROOT}/blitz-router"
bash ae/setup_repos.sh
```

To print the full experiment command list without launching the cluster jobs:

```bash
bash ae/run_all.sh print-experiments
```

To regenerate Figure 21 and Figure 22 from the archived data in
`${AE_ROOT}/xmetric-plots/data`:

```bash
bash ae/run_all.sh figures
```

## Project Structure

```
blitz-router/
├── router/             # Rust router (~14,000 LOC)
├── radixtree/          # Patricia trie crate (production + verified L0 + lowering bench ladder)
├── policy-dsl/         # Scheduling-policy DSL proc-macro
├── request-sim/        # Request simulator (git submodule)
└── docs/               # Documentation
```

## Related Projects

- **[yaullm](https://github.com/blitz-serving/yaullm)** — Patched vLLM engine with step-level SSE metrics
- **[request-sim](https://github.com/blitz-serving/request-sim)** — Request-load generator used by the AE orchestration
- **[MetricsTestRunner](https://github.com/blitz-serving/MetricsTestRunner)** — Cluster-level evaluation harness for router, yaullm, and request-sim experiments

## Acknowledgements

These projects incorporate code from **Text Generation Inference (TGI)**
<https://github.com/huggingface/text-generation-inference>
Copyright 2022-present Hugging Face Inc.
Licensed under the **Apache License, Version 2.0**.
The full license text is provided in `LICENSES/Apache-2.0.txt`.
Modification details in `NOTICE` and individual source files.
