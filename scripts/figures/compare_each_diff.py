import os
import re
import argparse
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# ========= 参数 =========
parser = argparse.ArgumentParser()
parser.add_argument("--dir1", type=str, required=True, help="第一目录 (如 simulator)")
parser.add_argument("--dir2", type=str, required=True, help="第二目录 (如 vllm)")
parser.add_argument("--out", type=str, required=True, help="输出图像路径")
args = parser.parse_args()

# ========= 正则 =========
vm_pattern = re.compile(
    r"(?P<timestamp>[\d\-:T\.]+Z).*?Vllm#(?P<instance_id>\d+)::Event::data received VllmMetric { .*?latency: (?P<latency>\d+), outputs: \[(?P<outputs>.*?)\] }"
)
output_pattern = re.compile(
    r"request_id: (?P<request_id>\d+), .*?state: \"(?P<state>PREFILL|DECODE)\""
)

# ========= 函数：解析一个目录，返回每个 event =========
def parse_dir(prefix):
    raw_log = os.path.join(prefix, "router_v2.log")
    processed_file = os.path.join(prefix, "processed.log")

    # 提取 VllmMetric 行
    with open(processed_file, "w") as outfile:
        subprocess.run(["grep", "VllmMetric", raw_log], stdout=outfile)

    instance_events = defaultdict(list)
    req0_prefill_candidates = []

    with open(processed_file, "r") as f:
        for line in f:
            m = vm_pattern.search(line)
            if not m:
                continue
            ts = datetime.fromisoformat(m.group("timestamp").replace("Z", "+00:00"))
            latency = int(m.group("latency"))
            inst = f"Vllm#{m.group('instance_id')}"
            outputs = m.group("outputs")

            matched = False
            for out in output_pattern.finditer(outputs):
                state = out.group("state")
                req_id = int(out.group("request_id"))
                instance_events[inst].append({
                    "time": ts,
                    "latency": latency,
                    "request_id": req_id
                })
                matched = True
                if state == "PREFILL" and req_id == 0:
                    req0_prefill_candidates.append((ts, latency))
            if not matched:
                instance_events[inst].append({
                    "time": ts,
                    "latency": latency,
                    "request_id": -1
                })

    # anchor 时间
    s0_abs = None
    if req0_prefill_candidates:
        t0, lat0 = sorted(req0_prefill_candidates, key=lambda x: x[0])[0]
        s0_abs = t0 - timedelta(milliseconds=int(lat0))
    else:
        any_events = [r["time"] for lst in instance_events.values() for r in lst]
        if any_events:
            s0_abs = min(any_events)

    # 合并所有实例到一个 DataFrame
    dfs = []
    for inst, events in instance_events.items():
        df = pd.DataFrame(events)
        if df.empty:
            continue
        df["instance"] = inst
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    total_df = pd.concat(dfs, ignore_index=True)
    if s0_abs is not None:
        total_df["aligned_sec"] = (total_df["time"] - s0_abs).dt.total_seconds()
    else:
        total_df["aligned_sec"] = (total_df["time"] - total_df["time"].min()).dt.total_seconds()

    return total_df

# ========= 主流程 =========
df1 = parse_dir(args.dir1)
df2 = parse_dir(args.dir2)

# ========= 绘图 =========
fig, axes = plt.subplots(2, 1, figsize=(14,8), sharex=True)

# --- 上图：Latency 每个 event 点
ax1 = axes[0]
ax1.scatter(df1["aligned_sec"], df1["latency"], label="Dir1 latency", color="blue", s=10, alpha=0.6)
ax1.scatter(df2["aligned_sec"], df2["latency"], label="Dir2 latency", color="orange", s=10, alpha=0.6)
ax1.set_ylabel("Latency (ms)")
ax1.legend()
ax1.grid(True)

# --- 下图：Count 分布 (直方图)
ax2 = axes[1]
ax2.hist(df1["aligned_sec"], bins=100, alpha=0.5, label="Dir1 count", color="blue")
ax2.hist(df2["aligned_sec"], bins=100, alpha=0.5, label="Dir2 count", color="orange")
ax2.set_ylabel("Decode Count")
ax2.set_xlabel("Aligned Time (s)")
ax2.legend()
ax2.grid(True)

plt.suptitle("Comparison of Two Runs (Each Event as Data Point)")
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(args.out)
print(f"Saved comparison figure: {args.out}")
