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
    r"(?P<timestamp>[\d\-:T\.]+Z).*?Vllm#(?P<instance_id>\d+)::Event::data received VllmMetric { .*?latency: (?P<latency>\d+), outputs: \[(?P<outputs>.*?)\][^}]*}"
)

output_pattern = re.compile(
    r"request_id: (?P<request_id>\d+), .*?state: \"(?P<state>PREFILL|DECODE)\""
)

# ========= 函数：解析一个目录，返回 total_df =========
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
                    "second": pd.to_datetime(ts).floor("s"),
                    "latency": latency,
                    "request_id": req_id,
                    "state": state
                })
                matched = True
                if state == "PREFILL" and req_id == 0:
                    req0_prefill_candidates.append((ts, latency))
            # 即使没有 outputs（outputs=[]），也要保留事件
            if not matched:
                instance_events[inst].append({
                    "time": ts,
                    "second": pd.to_datetime(ts).floor("s"),
                    "latency": latency,
                    "request_id": -1,
                    "state": "UNKNOWN"
                })

    # anchor 时间 (s0_abs)
    s0_abs = None
    if req0_prefill_candidates:
        t0, lat0 = sorted(req0_prefill_candidates, key=lambda x: x[0])[0]
        s0_abs = t0 - timedelta(milliseconds=int(lat0))
    else:
        any_events = [r["time"] for lst in instance_events.values() for r in lst]
        if any_events:
            s0_abs = min(any_events)

    # 聚合 per instance -> total
    per_inst_proc = {}
    for inst, events in instance_events.items():
        df = pd.DataFrame(events)
        if df.empty:
            continue

        g = df.groupby("second")
        # 现在所有 event 都统计，不再仅限 DECODE
        latency_avg = g["latency"].mean()
        count = df.drop_duplicates(subset=["second", "request_id"]).groupby("second").size()

        merged = pd.concat([
            latency_avg.rename("event_latency_avg"),
            count.rename("event_count")
        ], axis=1).sort_index()

        per_inst_proc[inst] = merged

    # 合并所有实例
    def to_wide(colname):
        cols = {}
        for inst in per_inst_proc:
            df = per_inst_proc[inst]
            if colname in df:
                cols[inst] = df[colname].rename(inst)
        if not cols:
            return pd.DataFrame()
        return pd.concat(cols.values(), axis=1).sort_index()

    latency_wide = to_wide("event_latency_avg")
    count_wide = to_wide("event_count")

    def safe_sum(df):
        return df.sum(axis=1, min_count=1) if not df.empty else pd.Series(dtype=float)

    def safe_mean(df):
        return df.mean(axis=1) if not df.empty else pd.Series(dtype=float)

    idx_union = None
    for df in [latency_wide, count_wide]:
        if not df.empty:
            idx_union = df.index if idx_union is None else idx_union.union(df.index)
    if idx_union is None:
        return pd.DataFrame()

    total_df = pd.DataFrame(index=idx_union)
    total_df["avg_event_latency"] = safe_mean(latency_wide).reindex(idx_union)
    total_df["total_event_count"] = safe_sum(count_wide).reindex(idx_union)

    # 用 s0_abs 对齐到 t=0
    if s0_abs is not None:
        total_df = total_df.reset_index()
        total_df["aligned_sec"] = (total_df["second"] - s0_abs).dt.total_seconds()
        total_df = total_df.set_index("aligned_sec")
    else:
        total_df = total_df.reset_index().rename(columns={"second": "aligned_sec"}).set_index("aligned_sec")

    return total_df

# ========= 主流程 =========
df1 = parse_dir(args.dir1)
df2 = parse_dir(args.dir2)

# ========= 绘图 =========
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# --- 上图：Latency 对比 (实线)
ax1 = axes[0]
if not df1.empty and "avg_event_latency" in df1:
    ax1.plot(df1.index, df1["avg_event_latency"], label="Dir1 avg latency", color="blue")
if not df2.empty and "avg_event_latency" in df2:
    ax1.plot(df2.index, df2["avg_event_latency"], label="Dir2 avg latency", color="orange")
ax1.set_ylabel("Avg Latency (ms)")
ax1.legend()
ax1.grid(True)

# --- 下图：Count 对比 (实线)
ax2 = axes[1]
if not df1.empty and "total_event_count" in df1:
    ax2.plot(df1.index, df1["total_event_count"], label="Dir1 count", color="blue")
if not df2.empty and "total_event_count" in df2:
    ax2.plot(df2.index, df2["total_event_count"], label="Dir2 count", color="orange")
ax2.set_ylabel("Event Count (req/s)")
ax2.set_xlabel("Aligned Time (s)")
ax2.legend()
ax2.grid(True)

plt.suptitle("Comparison of Two Runs (All Events as Batch Execution Time)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(args.out)
print(f"Saved comparison figure: {args.out}")
