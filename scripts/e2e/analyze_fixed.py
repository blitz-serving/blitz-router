#!/usr/bin/env python3
import json, os

base = "/workspace/tmp/lmetric/e2e-camera-ready/20260426-171403"
policies = ["vllm", "bailian", "aibrix", "dynamo-t1", "dynamo-t2", "lmetric", "preble"]

def pct(data, p):
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])

header = "%-12s %6s %10s %10s %10s %10s %10s %10s %8s %8s" % (
    "Policy", "N", "AvgTTFT", "P50TTFT", "P99TTFT", "AvgTPOT", "P50TPOT", "P99TPOT", "AvgISL", "AvgOSL")

print("=== SF=3.0 FIXED (all N=1349, no SSE contamination) ===")
print(header)
print("-" * len(header))

for pol in policies:
    f = os.path.join(base, pol, "sf3.0_qwen_traceA_blksz_16", "results.jsonl")
    if not os.path.exists(f):
        continue
    ttfts, tpots, isls, osls = [], [], [], []
    with open(f) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("status") != "200":
                continue
            ftt = d.get("first_token_time")
            atbt = d.get("avg_time_between_tokens")
            isl = d.get("input_length")
            osl = d.get("output_length")
            if ftt is not None:
                ttfts.append(float(ftt))
            if atbt is not None and atbt != "nil":
                tpots.append(float(atbt))
            if isl is not None:
                isls.append(int(isl))
            if osl is not None:
                osls.append(int(osl))
    n = len(ttfts)
    if n < 50:
        continue
    avg_isl = sum(isls) / len(isls) if isls else 0
    avg_osl = sum(osls) / len(osls) if osls else 0
    print("%-12s %6d %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %8.0f %8.0f" % (
        pol, n,
        sum(ttfts)/n, pct(ttfts, 50), pct(ttfts, 99),
        sum(tpots)/len(tpots), pct(tpots, 50), pct(tpots, 99),
        avg_isl, avg_osl))

# Also print ISL distribution
print("\n=== Input/Output Length Distribution (vllm run) ===")
f = os.path.join(base, "vllm", "sf3.0_qwen_traceA_blksz_16", "results.jsonl")
isls, osls = [], []
with open(f) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get("status") != "200":
            continue
        isls.append(int(d.get("input_length", 0)))
        osls.append(int(d.get("output_length", 0)))
print("ISL: min=%d, p50=%d, p90=%d, p99=%d, max=%d" % (
    min(isls), pct(isls, 50), pct(isls, 90), pct(isls, 99), max(isls)))
print("OSL: min=%d, p50=%d, p90=%d, p99=%d, max=%d" % (
    min(osls), pct(osls, 50), pct(osls, 90), pct(osls, 99), max(osls)))
