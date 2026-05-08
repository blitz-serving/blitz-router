## Reproducing experiments

End-to-end experiment orchestration (router + yaullm + request-sim) lives in [MetricsTestRunner](https://github.com/blitz-serving/MetricsTestRunner). Per-cluster TOML configs are under `MetricsTestRunner/config/<cluster>/`; sweep definitions are under `MetricsTestRunner/sweeps/`.

This repo only ships the router itself.
