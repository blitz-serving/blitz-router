# `least-ttft-q` policy — first end-to-end run

> Result of the first production run of the new `least-ttft-q` scheduling
> policy under `simulator-aware` routing. Captures the policy's dispatch
> behaviour and observed serving metrics on a typical workload, plus
> the caveats from running with off-model Vidur grids.

## What was tested

| Knob                            | Value                                              |
|---------------------------------|----------------------------------------------------|
| Policy                          | `least-ttft-q` (simulator-driven; min `RolloutGist.ttft_ms`) |
| Cluster                         | `a800-8gpu` (1×Docker container, 8×A800-SXM4-80GB) |
| Backend                         | yaullm (Qwen2.5-7B-Instruct), 8 instances, 1 per GPU |
| Backend mode                    | **graph mode** (no `--enforce-eager`), `--gpu-memory-utilization 0.4` |
| Router build                    | `cargo build --features simulator,least-ttft-q --release` |
| Simulator predictor             | Vidur RF over **Llama-2-7B grids** (param-count proxy for Qwen2.5-7B) |
| Corrector                       | `NullCorrector` behaviour (`--simulator-learning-rate 0.0`; weights stay at `(1.0, 0.0)` so `corrected = raw`) |
| Workload                        | `qwen_traceA_blksz_16.jsonl`, `--mode trace-replay`, `--api openai`, `--stream`, **scale factor 2.5** |
| Trace duration                  | 1200 s (20 min) |
| Driver                          | `metro launch --policy least-ttft-q` (semi-agent tmux mode) |

## What `least-ttft-q` does

Hand-written `impl Policy` (the `policy!` proc macro can't express a
per-replica predictor call). For each candidate request:

1. Iterate replicas; for each, snapshot `SCtx.block_hash.get(hashes)`.
2. Call `simulator::query(replica, candidate_id, input_length, hashes, sctx_hits)` — returns `Option<RolloutGist>`.
3. Pick the replica whose `gist.ttft_ms` is minimum.
4. Apply the post-decision bookkeeping that DSL policies get for free
   via `apply_default_after`: `set_pred_block_hits`, `set_decision_epoch`,
   `lmetric += LMetricInc{...}`. **Skipping this triggers a hard panic
   in `BlockHashState::set_real_token_hits_get_diff` (the `pred_hit_nblks
   != NONE_SENTINEL` assertion) on the first request that completes
   prefill** — caught and fixed in this run.

Fallback: if all replicas return `None` (cold L2 cache or
simulator-inactive misconfiguration), pick replica 0 with a `WARN`. In
this run the fallback fired **0 times**.

## Headline numbers

| Metric                        | Value         |
|-------------------------------|---------------|
| Requests issued               | 15,544        |
| Requests succeeded            | 15,544 (100%) |
| Run duration                  | 1,215.7 s     |
| Achieved RPS                  | 12.79         |
| Target RPS (after SF=2.5)     | 14.95         |
| Aggregate token throughput    | 5,299 tok/s   |
| Output tokens generated       | 6,441,793     |
| TTFT mean / p90 / p95 / p99   | 143.6 / 391.1 / 562.9 / 949.8 ms |
| TPOT mean / p90 / p95 / p99   | 19.5 / 24.5 / 27.1 / 34.9 ms |
| E2E mean / p90 / p95 / p99    | 7,758 / 14,917 / 18,460 / 28,932 ms |

Achieved RPS sitting at ~85% of the SF=2.5 target indicates the cluster
saturated; queue tail latency (E2E p99 ≈ 29 s) is the visible symptom.
Lower SF (e.g. SF=2.0) would let the head latencies dominate and make
the policy comparison cleaner — flagged for the follow-up sweep.

## Per-replica dispatch distribution

`grep REQUEST_ADMIT … | grep -oE engine=N | sort | uniq -c` from
`router_v2.log`:

| Engine | Admissions | % of total |
|--------|------------|------------|
| 0      | 1,976      | 12.78 %    |
| 1      | 1,886      | 12.20 %    |
| 2      | 1,949      | 12.61 %    |
| 3      | 1,945      | 12.58 %    |
| 4      | 1,827      | 11.82 %    |
| 5      | 2,086      | 13.49 %    |
| 6      | 2,021      | 13.07 %    |
| 7      | 1,775      | 11.48 %    |

Min/max ratio = 1.18×, σ ≈ 100 admissions over the 20-min window. The
spread reflects the policy chasing per-replica L1-state differences as
they evolve — not a uniform round-robin or a fully balanced shortest-
queue. Worth comparing against `join-shortest-weight-q`'s spread on the
same trace as a baseline (deferred to the sweep).

## Simulator behaviour observations

- **Llama-2 grids miss most queried `(kv_cache, flops)` cells.** The
  router log contains 2,172,848 `missing prediction for op=…` warnings
  over the 20-min run — i.e. the L2 inner predictor was hitting fallback
  constants for the majority of slot-level queries. With `NullCorrector`
  active (`learning_rate=0.0`), no online correction absorbed the
  systematic error.
- **Despite this, the policy still produced a meaningful per-replica
  ordering**, because the simulator's L1 state (in-flight requests,
  `SchedSnapshot` progress, `RadixTreeReqIdHash` membership) differs
  per replica. The L2 fallback returns the *same* per-op constant
  regardless of replica, so the gist's TTFT differences came from L1
  composition, not L2 quality. This is enough to drive load-aware
  dispatch but does NOT exercise the simulator's prefix-cache-hit
  scoring path the way a real Qwen2.5 grid would.
- **Zero panics, zero policy fallbacks.** The fix to call
  `set_pred_block_hits` / `lmetric += LMetricInc` inside the policy
  was load-bearing — the run that didn't have it crashed the router
  on request #1's completion event.

## What's good for follow-up

1. **Real Qwen2.5-7B Vidur grids.** Run the full Vidur profiling
   pipeline (`python -m vidur.profiling.attention.main --models
   Qwen/Qwen2.5-7B-Instruct ...`) to replace the Llama-2 stopgap. With
   a calibrated grid, the L2 cost oracle returns model-aware
   predictions and the policy's TTFT minimum becomes truly informative
   rather than a tie-breaker on L1 state.
2. **Toggle `LinregCorrector` on (`--simulator-learning-rate 1e-9`).**
   Even with a stopgap grid, the linreg correction absorbs the
   multiplicative bias after warmup. The current run intentionally
   used `NullCorrector` per the test spec.
3. **Sweep against baselines.** `join-shortest-weight-q`,
   `lmetric-q`, `bailian-impl-q` on the same trace at SF ∈ {2.0, 2.5,
   3.0} — quantify whether `least-ttft-q` actually improves TTFT (or
   any other SLO metric) over the load-only baselines.
4. **Lower SF for cleaner comparison.** The current SF=2.5 saturated
   the cluster; head/tail latency separation is muddied by queue
   buildup. SF=1.5–2.0 would isolate the policy's contribution.

## Files

| Path | Content |
|------|---------|
| `router/src/scheduler/policies/least_ttft.rs` | The new policy (hand-written `impl Policy`). |
| `router/Cargo.toml` (`[features]`) | `least-ttft-q = ["simulator"]` (the policy implies the simulator subsystem). |
| `MetricsTestRunner/config/ali-a800/launch_vllm_dp8_qwen25_least_ttft.toml` | Backend TOML: 8 vllm instances, Qwen2.5-7B, mem-frac 0.4, graph mode. |
| `MetricsTestRunner/config/ali-a800/vllm_router_least_ttft.toml` | Router TOML: simulator flags (`--simulator-cache-dir`, `--simulator-model-hash llama2_7b`, `--simulator-num-layers 28`, `--simulator-learning-rate 0.0`). |
| `MetricsTestRunner/clusters/a800-8gpu.toml` | Cluster profile gained `qwen25-7bi` model entry. |
| `tmp/cache/*_llama2_7b_predictions.csv` | 14 Vidur grids built via `tmp/build_stopgap_grids.py`. |
| `/workspace/tmp/router-bins/router-least-ttft-q` | Pre-built router binary (130 MB) — metro launcher picks this up to skip cargo build. |
| `/workspace/exps/metro/20260510094501_least-ttft-q/` | Run output: `router_v2.log`, `vllm{1..8}.log`, `client.jsonl`, `client.jsonl.summary.json`, `metro_status_done`. |

## Reproduce

```bash
# inside zdy-yaullm-metro-8gpu container
cd /workspace/MetricsTestRunner
python3 -m metro.cli launch \
  --cluster a800-8gpu \
  --model qwen25-7bi \
  --backend config/ali-a800/launch_vllm_dp8_qwen25_least_ttft.toml \
  --router  config/ali-a800/vllm_router_least_ttft.toml \
  --client  config/ali-a800/client_bailian_container.toml \
  --policy  least-ttft-q \
  --time    1200 \
  --set     scale-factor=2.5 \
  --no-remote \
  --session least-ttft-q
# poll status:
python3 -m metro.cli status --output-dir /workspace/exps/metro/<timestamp>_least-ttft-q
# collect:
python3 -m metro.cli collect --output-dir /workspace/exps/metro/<timestamp>_least-ttft-q --json
```
