#!/usr/bin/env python3
import json, sys, os, glob, statistics

base = "/workspace/tmp/lmetric/e2e-camera-ready"
policies = ["vllm", "bailian", "aibrix", "dynamo-t1", "dynamo-t2", "lmetric", "preble"]
scale_factors = ["2.0", "3.0", "4.0", "5.0"]

def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    d = k - f
    return s[f] + d * (s[c] - s[f])

def mean(data):
    return sum(data) / len(data) if data else 0

print("=" * 100)
print("  LATENCY METRICS — Camera-Ready Experiments")
print("  Trace: traceA (ChatBot), Model: Qwen3-30B-A3B, 7 GPUs")
print("=" * 100)

for sf in scale_factors:
    print(f"\n--- Scale Factor: {sf} ---")
    print(f"{'Policy':<12} {'N':>6} {'AvgTTFT':>10} {'P50TTFT':>10} {'P99TTFT':>10} {'AvgTPOT':>10} {'P50TPOT':>10} {'P99TPOT':>10}")
    print("-" * 82)

    for policy in policies:
        # Find best result file for this policy+sf
        pattern = os.path.join(base, f"*/{policy}/sf{sf}_*/results.jsonl")
        files = glob.glob(pattern)
        if not files:
            continue

        best_file = max(files, key=lambda f: os.path.getsize(f))

        ttfts, tpots = [], []
        with open(best_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("status") != "200":
                        continue
                    ftt = d.get("first_token_time")
                    atbt = d.get("avg_time_between_tokens")
                    if ftt is not None:
                        ttfts.append(float(ftt))
                    if atbt is not None and atbt != "nil":
                        tpots.append(float(atbt))
                except:
                    pass

        n = len(ttfts)
        if n < 50:
            continue

        t = ttfts
        tp = tpots if tpots else [0]

        print(f"{policy:<12} {n:>6} {mean(t):>10.1f} {percentile(t,50):>10.1f} {percentile(t,99):>10.1f} {mean(tp):>10.1f} {percentile(tp,50):>10.1f} {percentile(tp,99):>10.1f}")

print("\nNote: Some policies have lower N due to SSE state contamination bug (colocation.rs:521).")
print("Policies with N significantly lower than peers at the same SF should be re-run with clean yaullm state.")
