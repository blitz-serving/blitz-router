import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import subprocess

# ========= 基本路径设置 =========
PREFIX = "/nvme/lmetric/logs/dp8_2code_2conv/lwl/"
RAW_LOG_FILE = f"{PREFIX}router_v2.log"
processed_file = f"{PREFIX}processed.log"
alert_file = f"{PREFIX}alert.log"

# 预处理：从 router_v2.log 中抓取含 VllmMetric 的行到 processed.log
os.makedirs(PREFIX, exist_ok=True)
with open(processed_file, 'w') as outfile:
    subprocess.run(['grep', 'VllmMetric', RAW_LOG_FILE], stdout=outfile)

with open(alert_file, 'w') as outfile:
    # 新格式下，我们保留监控循环的所有行，便于统一解析
    subprocess.run(['grep', 'replica_state_moniter_loop', RAW_LOG_FILE], stdout=outfile)

GRAPH_PATH = os.path.join(os.path.dirname(processed_file), "instance_fig/")
os.makedirs(GRAPH_PATH, exist_ok=True)

# ========= 日志正则（processed.log: VllmMetric）=========
vm_pattern = re.compile(
    r"(?P<timestamp>[\d\-:T\.]+Z).*?Vllm#(?P<instance_id>\d+)::Event::data received VllmMetric { prefill_tokens: (?P<prefill_tokens>\d+),.*?latency: (?P<latency>\d+), outputs: \[(?P<outputs>.*?)\] }"
)
output_pattern = re.compile(
    r"request_id: (?P<request_id>\d+), new_token_ids: \[.*?\], state: \"(?P<state>PREFILL|DECODE)\", is_finished: (true|false)"
)

# ========= 日志正则（alert.log: 新格式）=========
# 每实例
alert_inst_pattern = re.compile(
    r'(?P<timestamp>[\d\-:T\.]+Z).*?Vllm#(?P<inst>\d+)::replica_state_moniter_loop prefill_tokens=(?P<pt>\d+), request_num=(?P<rn>\d+)'
)
# 全局
alert_global_pattern = re.compile(
    r'(?P<timestamp>[\d\-:T\.]+Z).*?replica_state_moniter_loop global_prefill_tokens_per_sec=(?P<gptps>\d+), global_request_num=(?P<grn>\d+)'
)
# 队列/批数组
alert_array_pattern = re.compile(
    r'(?P<timestamp>[\d\-:T\.]+Z).*?replica_state_moniter_loop all_num_request_in_queue=\[(?P<queues>[^\]]*)\], all_instance_batch_size=\[(?P<batches>[^\]]*)\]'
)

# ========= 解析 processed.log =========
instance_events = defaultdict(list)  # inst -> list of records

with open(processed_file, "r") as f:
    for line in f:
        m = vm_pattern.search(line)
        if not m:
            continue
        ts = datetime.fromisoformat(m.group("timestamp").replace("Z", "+00:00"))
        inst = f"Vllm#{m.group('instance_id')}"
        latency = int(m.group("latency"))
        outputs = m.group("outputs")

        for out in output_pattern.finditer(outputs):
            state = out.group("state")
            req_id = int(out.group("request_id"))
            instance_events[inst].append({
                "time": ts,
                "second": pd.to_datetime(ts).floor("s"),
                "state": state,
                "latency": latency,
                "request_id": req_id
            })

# 将 processed.log 聚合为每秒每实例 metrics
per_inst_frames = {}  # inst -> df(index=second, cols=decode_latency_avg, prefill_ratio, decode_count)
for inst, events in instance_events.items():
    df = pd.DataFrame(events)
    if df.empty:
        continue
    g = df.groupby("second")

    # 平均 DECODE latency
    def decode_mean(s):
        # s 是同一 second 下的 latency 序列，这里需要按原 df 的 state 过滤
        idx = s.index
        return df.loc[idx][df.loc[idx, "state"] == "DECODE"]["latency"].mean()

    decode_latency = g["latency"].apply(decode_mean)

    # prefill_ratio
    prefill_lat_sum = df[df["state"] == "PREFILL"].groupby("second")["latency"].sum()
    prefill_ratio = (prefill_lat_sum / 1000.0).clip(upper=1.0)

    # decode_count（按秒去重 request_id）
    decode_df = df[df["state"] == "DECODE"]
    decode_count = (
        decode_df.drop_duplicates(subset=["second", "request_id"])
                .groupby("second").size()
    )

    merged = pd.concat([
        decode_latency.rename("decode_latency_avg"),
        prefill_ratio.rename("prefill_ratio"),
        decode_count.rename("decode_count")
    ], axis=1).sort_index()

    per_inst_frames[inst] = merged

# ========= 解析 alert.log（新格式）=========
inst_rows = []     # 每实例行
global_rows = []   # 全局行
array_rows = []    # 数组行
instance_count = None

if os.path.exists(alert_file):
    with open(alert_file, "r") as af:
        for line in af:
            if (m := alert_inst_pattern.search(line)):
                ts = datetime.fromisoformat(m.group("timestamp").replace("Z", "+00:00"))
                second = pd.to_datetime(ts).floor("s")
                inst = int(m.group("inst"))
                pt = int(m.group("pt"))
                rn = int(m.group("rn"))
                inst_rows.append({"second": second, "inst": inst, "prefill_tokens": pt, "request_num": rn})
                continue

            if (m := alert_global_pattern.search(line)):
                ts = datetime.fromisoformat(m.group("timestamp").replace("Z", "+00:00"))
                second = pd.to_datetime(ts).floor("s")
                global_rows.append({
                    "second": second,
                    "global_prefill_tokens_per_sec": int(m.group("gptps")),
                    "global_request_num": int(m.group("grn")),
                })
                continue

            if (m := alert_array_pattern.search(line)):
                ts = datetime.fromisoformat(m.group("timestamp").replace("Z", "+00:00"))
                second = pd.to_datetime(ts).floor("s")
                q = [int(x.strip()) for x in m.group("queues").split(",") if x.strip() != ""]
                b = [int(x.strip()) for x in m.group("batches").split(",") if x.strip() != ""]
                if instance_count is None:
                    instance_count = len(q)
                else:
                    if len(q) != instance_count or len(b) != instance_count:
                        raise ValueError(
                            f"alert.log 数组长度不一致：期望 {instance_count}，但遇到 queues={len(q)}, batches={len(b)} @ {second}"
                        )
                row = {"second": second}
                for i in range(len(q)):
                    row[f"queue_{i}"] = q[i]
                    row[f"batch_{i}"] = b[i]
                array_rows.append(row)

# 整理 alert 数据
alert_inst_df = (
    pd.DataFrame(inst_rows).set_index("second").sort_index()
    if inst_rows else pd.DataFrame(columns=["inst", "prefill_tokens", "request_num"])
)

alert_global_df = (
    pd.DataFrame(global_rows).drop_duplicates(subset=["second"]).set_index("second").sort_index()
    if global_rows else pd.DataFrame(columns=["global_prefill_tokens_per_sec", "global_request_num"])
)

queues_batches_wide = (
    pd.DataFrame(array_rows).set_index("second").sort_index()
    if array_rows else pd.DataFrame()
)

# ========= 将 processed 与 alert 按实例对齐，做“每秒每实例”的宽表 =========
# 找出所有实例名（以 processed 为主，也兼容 alert 里的实例编号）
inst_names = sorted(per_inst_frames.keys(), key=lambda x: int(x.split("#")[1])) if per_inst_frames else []
if not inst_names and not alert_inst_df.empty:
    # 如果 processed 为空但 alert 有数据，按 alert 实例构造名
    max_inst = int(alert_inst_df["inst"].max())
    inst_names = [f"Vllm#{i}" for i in range(max_inst + 1)]

# 各指标汇总为列=实例的宽表
def collect_metric(metric_name, series_getter):
    cols = {}
    for inst in inst_names:
        s = series_getter(inst)
        if s is not None and not s.empty:
            cols[inst] = s.rename(inst)
    if not cols:
        return pd.DataFrame()
    df = pd.concat(cols.values(), axis=1).sort_index()
    return df

# processed 三项
decode_latency_wide = collect_metric(
    "decode_latency_avg",
    lambda inst: per_inst_frames.get(inst, pd.DataFrame()).get("decode_latency_avg")
)
decode_count_wide = collect_metric(
    "decode_count",
    lambda inst: per_inst_frames.get(inst, pd.DataFrame()).get("decode_count")
)
prefill_ratio_wide = collect_metric(
    "prefill_ratio",
    lambda inst: per_inst_frames.get(inst, pd.DataFrame()).get("prefill_ratio")
)

# alert 每实例两项（通过 inst 编号映射）
def alert_per_inst_series(inst, col):
    if alert_inst_df.empty:
        return None
    try:
        idx = int(inst.split("#")[1])  # Vllm#N
    except Exception:
        return None
    df = alert_inst_df[alert_inst_df["inst"] == idx]
    if df.empty or col not in df:
        return None
    return df[col]

prefill_tokens_wide = collect_metric("prefill_tokens", lambda inst: alert_per_inst_series(inst, "prefill_tokens"))
request_num_wide    = collect_metric("request_num",    lambda inst: alert_per_inst_series(inst, "request_num"))

# queues / batches 每实例
def qb_series(inst, kind):
    if queues_batches_wide.empty:
        return None
    try:
        idx = int(inst.split("#")[1])
    except Exception:
        return None
    col = f"{kind}_{idx}"
    if col not in queues_batches_wide.columns:
        return None
    return queues_batches_wide[col]

queue_wide = collect_metric("queue", lambda inst: qb_series(inst, "queue"))
batch_wide = collect_metric("batch", lambda inst: qb_series(inst, "batch"))

# ========= 画图：每秒每实例 =========
# 我们做 6 个子图，如果某个宽表为空就跳过该子图内容
per_instance_fig = os.path.join(GRAPH_PATH, "per_instance_metrics.png")

series_list = [
    ("Avg Decode Latency (ms)", decode_latency_wide),
    ("Decode Count (req/s)",    decode_count_wide),
    ("Prefill Ratio (0-1)",     prefill_ratio_wide),
    ("Prefill Tokens",          prefill_tokens_wide),
    ("Request Num",             request_num_wide),
    ("Queue / Batch",           None),  # 这个子图里我们同时画 queue 和 batch
]

# 计算需要的子图数量（非空的 + Queue/Batch 至少占一个）
n_plots = sum(1 for name, df in series_list if (df is None or not df.empty))
fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.8 * n_plots), sharex=True)
if n_plots == 1:
    axes = [axes]

ax_idx = 0
for name, wide in series_list:
    ax = axes[ax_idx]
    if name == "Queue / Batch":
        drew = False
        if not queue_wide.empty:
            for col in queue_wide.columns:
                ax.plot(queue_wide.index, queue_wide[col], linestyle="-", label=f"{col.replace('_', ' ').title()}")
            drew = True
        if not batch_wide.empty:
            for col in batch_wide.columns:
                ax.plot(batch_wide.index, batch_wide[col], linestyle="--", label=f"{col.replace('_', ' ').title()}")
            drew = True
        if drew:
            ax.set_ylabel(name)
            ax.legend(ncol=3, fontsize=8)
            ax_idx += 1
        continue

    if wide is not None and not wide.empty:
        for col in wide.columns:
            ax.plot(wide.index, wide[col], label=col)
        ax.set_ylabel(name)
        ax.legend(ncol=4, fontsize=8)
        ax_idx += 1

# 统一 X 轴时间格式
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.grid(True)

axes[-1].set_xlabel("Time (seconds)")
fig.autofmt_xdate()
fig.suptitle("Per-Instance Metrics (per second)")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(per_instance_fig)
print(f"Saved: {per_instance_fig}")

# ========= 统计总量并画图：每秒 total =========
# 聚合：总和/平均
def safe_sum(df):
    return df.sum(axis=1, min_count=1) if not df.empty else pd.Series(dtype=float)

def safe_mean(df):
    return df.mean(axis=1) if not df.empty else pd.Series(dtype=float)

total_df = pd.DataFrame(index=pd.Index([], name="second"))

# 统一时间索引：取所有时间点的并集
all_indices = []
for df in [decode_latency_wide, decode_count_wide, prefill_ratio_wide,
           prefill_tokens_wide, request_num_wide, queue_wide, batch_wide,
           alert_global_df]:
    if df is None:
        continue
    if isinstance(df, pd.DataFrame):
        if not df.empty:
            all_indices.append(df.index)
    elif isinstance(df, pd.Series):
        if not df.empty:
            all_indices.append(df.index)

if all_indices:
    union_index = all_indices[0]
    for idx in all_indices[1:]:
        union_index = union_index.union(idx)
    total_df = pd.DataFrame(index=union_index.sort_values())

# 聚合计算
total_df["avg_decode_latency"]  = safe_mean(decode_latency_wide).reindex(total_df.index)
total_df["total_decode_count"]  = safe_sum(decode_count_wide).reindex(total_df.index)
total_df["avg_prefill_ratio"]   = safe_mean(prefill_ratio_wide).reindex(total_df.index)
total_df["sum_prefill_tokens"]  = safe_sum(prefill_tokens_wide).reindex(total_df.index)
total_df["sum_request_num"]     = safe_sum(request_num_wide).reindex(total_df.index)
total_df["sum_queue"]           = safe_sum(queue_wide).reindex(total_df.index)
total_df["sum_batch"]           = safe_sum(batch_wide).reindex(total_df.index)

# 合并全局
if not alert_global_df.empty:
    total_df = total_df.join(alert_global_df[["global_prefill_tokens_per_sec", "global_request_num"]], how="left")

# 画 total 图
total_fig = os.path.join(GRAPH_PATH, "total_metrics.png")
fig2, axes2 = plt.subplots(6, 1, figsize=(14, 2.8 * 6), sharex=True)

# 1) avg_decode_latency
axes2[0].plot(total_df.index, total_df["avg_decode_latency"])
axes2[0].set_ylabel("Avg Decode Latency (ms)")
axes2[0].grid(True)

# 2) total_decode_count
axes2[1].plot(total_df.index, total_df["total_decode_count"])
axes2[1].set_ylabel("Total Decode Count (req/s)")
axes2[1].grid(True)

# 3) avg_prefill_ratio
axes2[2].plot(total_df.index, total_df["avg_prefill_ratio"])
axes2[2].set_ylabel("Avg Prefill Ratio (0-1)")
axes2[2].grid(True)

# 4) sum_prefill_tokens
axes2[3].plot(total_df.index, total_df["sum_prefill_tokens"])
axes2[3].set_ylabel("Sum Prefill Tokens")
axes2[3].grid(True)

# 5) sum_request_num
axes2[4].plot(total_df.index, total_df["sum_request_num"])
axes2[4].set_ylabel("Sum Request Num")
axes2[4].grid(True)

# 6) sum_queue / sum_batch / global*
axes2[5].plot(total_df.index, total_df["sum_queue"], label="Sum Queue")
axes2[5].plot(total_df.index, total_df["sum_batch"], label="Sum Batch", linestyle="--")
if "global_prefill_tokens_per_sec" in total_df:
    axes2[5].plot(total_df.index, total_df["global_prefill_tokens_per_sec"], label="Global Prefill Tokens/s")
if "global_request_num" in total_df:
    axes2[5].plot(total_df.index, total_df["global_request_num"], label="Global Request Num", linestyle=":")
axes2[5].set_ylabel("Totals / Global")
axes2[5].legend(ncol=3, fontsize=9)
axes2[5].grid(True)

for ax in axes2:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

axes2[-1].set_xlabel("Time (seconds)")
fig2.autofmt_xdate()
fig2.suptitle("Total Metrics (per second)")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(total_fig)
print(f"Saved: {total_fig}")

print("All done.")
