import re
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()


# ========= 基本路径设置 =========
PREFIX = args.output_dir
RAW_LOG_FILE = f"{PREFIX}/router_v2.log"
processed_file = f"{PREFIX}/processed.log"
alert_file = f"{PREFIX}/alert.log"
CLIENT_FILE = f"{PREFIX}/client.jsonl"
TMP_LOG_FILE = f"{PREFIX}/router_v2.tmp"

def keep_from_first(pattern: str, infile: str, outfile: str):
    with open(outfile, "wb") as out:
        subprocess.run(
            ["awk", "-v", f"p={pattern}", r'f||$0~p{f=1} f', infile],
            check=True,
            stdout=out
        )

def keep_to_last(pattern2: str, infile: str, outfile: str):
    with open(outfile, "wb") as out:
        subprocess.run(
            [
                "awk",
                "-v", f"p={pattern2}",
                r"$0~p{last=NR}{buf[NR]=$0} END{for(i=1;i<=last;i++) print buf[i]}",
                infile,
            ],
            check=True,
            stdout=out,
        )



keep_from_first("added request", RAW_LOG_FILE, TMP_LOG_FILE)
keep_to_last("VllmMetric", TMP_LOG_FILE, RAW_LOG_FILE)
# subprocess.run(['rm', TMP_LOG_FILE])
# exit(0)

# ========= 预处理 =========
os.makedirs(PREFIX, exist_ok=True)
with open(processed_file, 'w') as outfile:
    subprocess.run(['grep', 'VllmMetric', RAW_LOG_FILE], stdout=outfile)

with open(alert_file, 'w') as outfile:
    subprocess.run(['grep', 'replica_state_moniter_loop', RAW_LOG_FILE], stdout=outfile)

GRAPH_PATH = os.path.join(os.path.dirname(processed_file), "instance_fig/")
os.makedirs(GRAPH_PATH, exist_ok=True)

# ========= 颜色与样式 =========
salmon_cmap = mcolors.LinearSegmentedColormap.from_list("prefill_cmap", ["white", "salmon"])
COLOR_PREFILL_TOKENS = "#1f77b4"  # 蓝
COLOR_REQUEST_PERSEC  = "#ff7f0e"  # 橙
COLOR_BATCH_INST      = "#9467bd"  # 紫
COLOR_DECODE_LAT      = "#2ca02c"  # 绿
COLOR_DECODE_COUNT    = "#d62728"  # 红
COLOR_TTFT            = "#17becf"  # 青（Total 第三图：左轴）
COLOR_TPOT            = "#8c564b"  # 褐（Total 第三图：右轴）

LW_MAIN   = 2.0     # 主要曲线粗细
LW_THIN   = 1.2     # 全局与 TTFT/TPOT 的细线

def set_time_formatter(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.grid(True)

# ========= 正则（processed.log: VllmMetric）=========
vm_pattern = re.compile(
    r"(?P<timestamp>[\d\-:T\.]+Z).*?Vllm#(?P<instance_id>\d+)::Event::data received VllmMetric { prefill_tokens: (?P<prefill_tokens>\d+),.*?latency: (?P<latency>\d+), outputs: \[(?P<outputs>.*?)\], log_info: .*? }"
)
output_pattern = re.compile(
    r"request_id: (?P<request_id>\d+), new_token_ids: \[.*?\], state: \"(?P<state>PREFILL|DECODE)\", is_finished: (true|false)"
)

# ========= 正则（alert.log: 新格式）=========
alert_inst_pattern = re.compile(
    r'(?P<timestamp>[\d\-:T\.]+Z).*?Vllm#(?P<inst>\d+)::replica_state_moniter_loop prefill_tokens=(?P<pt>\d+), request_num=(?P<rn>\d+)'
)
alert_global_pattern = re.compile(
    r'(?P<timestamp>[\d\-:T\.]+Z).*?replica_state_moniter_loop global_prefill_tokens_per_sec=(?P<gptps>\d+), global_request_num=(?P<grn>\d+)'
)
alert_array_pattern = re.compile(
    r'(?P<timestamp>[\d\-:T\.]+Z).*?replica_state_moniter_loop all_num_request_in_queue=\[(?P<queues>[^\]]*)\], all_instance_batch_size=\[(?P<batches>[^\]]*)\]'
)

# ========= 解析 processed.log（细粒度事件 -> 每秒每实例）=========
instance_events = defaultdict(list)
# 记录 request_id==0 的 PREFILL 候选，用于对齐 s_time=0 的绝对时间
req0_prefill_candidates = []  # [(timestamp, latency_ms)]

with open(processed_file, "r") as f:
    for line in f:
        m = vm_pattern.search(line)
        if not m:
            continue
        ts = datetime.fromisoformat(m.group("timestamp").replace("Z", "+00:00"))  # aware UTC
        latency = int(m.group("latency"))
        inst = f"Vllm#{m.group('instance_id')}"
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
            if state == "PREFILL" and req_id == 0:
                req0_prefill_candidates.append((ts, latency))

# processed 每秒聚合
per_inst_proc = {}
all_prefill_candidates_any = []  # 作为兜底
for inst, events in instance_events.items():
    df = pd.DataFrame(events)
    if df.empty:
        continue
    # 兜底：任意 PREFILL 候选
    any_prefill = df[df["state"] == "PREFILL"][["time", "latency"]]
    for _, r in any_prefill.iterrows():
        all_prefill_candidates_any.append((r["time"], int(r["latency"])))

    g = df.groupby("second")

    def decode_mean(s):
        idx = s.index
        return df.loc[idx][df.loc[idx, "state"] == "DECODE"]["latency"].mean()

    decode_latency = g["latency"].apply(decode_mean)

    prefill_lat_sum = df[df["state"] == "PREFILL"].groupby("second")["latency"].sum()
    prefill_ratio = (prefill_lat_sum / 1000.0).clip(upper=1.0)

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

    per_inst_proc[inst] = merged

# ========= 解析 alert.log（新格式）=========
inst_rows = []
global_rows = []
array_rows = []   # 仅使用 batch

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
                b = [x.strip() for x in m.group("batches").split(",") if x.strip() != ""]
                row = {"second": second}
                for i, bi in enumerate(b):
                    row[f"batch_{i}"] = int(bi)
                array_rows.append(row)

alert_inst_df = (
    pd.DataFrame(inst_rows).set_index("second").sort_index()
    if inst_rows else pd.DataFrame(columns=["inst", "prefill_tokens", "request_num"])
)

alert_global_df = (
    pd.DataFrame(global_rows).drop_duplicates(subset=["second"]).set_index("second").sort_index()
    if global_rows else pd.DataFrame(columns=["global_prefill_tokens_per_sec", "global_request_num"])
)

batches_wide = (
    pd.DataFrame(array_rows).set_index("second").sort_index()
    if array_rows else pd.DataFrame()
)

# ========= 计算 s_time=0 的绝对时间（对齐 client.jsonl 与 processed.log）=========
s0_abs = None
if req0_prefill_candidates:
    # 取时间最早的 request_id=0 的 PREFILL 事件
    t0, lat0 = sorted(req0_prefill_candidates, key=lambda x: x[0])[0]
    s0_abs = t0 - timedelta(milliseconds=int(lat0))
elif all_prefill_candidates_any:
    # 兜底：任意 PREFILL 事件最早的一条
    t0, lat0 = sorted(all_prefill_candidates_any, key=lambda x: x[0])[0]
    s0_abs = t0 - timedelta(milliseconds=int(lat0))
else:
    # 最弱兜底：任意事件最早时间
    any_events = [r["time"] for lst in instance_events.values() for r in lst]
    if any_events:
        s0_abs = min(any_events)
# 若仍为空，后续 TTFT/TPOT 将为空处理

# ========= 解析“请求级 JSON 指标”（TTFT/TPOT，按绝对日志时间聚合）=========
# 以 统计时间 = s_time(秒) + first_token_time(毫秒)/1000，映射到 绝对时间 = s0_abs + 统计时间
ttft_tpot_rows = []
if s0_abs is not None and os.path.exists(CLIENT_FILE):
    with open(CLIENT_FILE, "r") as f:
        for line in f:
            if '"avg_time_between_tokens"' not in line:
                continue
            # 抓取 JSON
            try:
                start = line.index("{")
                end = line.rindex("}") + 1
                j = json.loads(line[start:end])
            except Exception:
                continue

            s_time = j.get("s_time", j.get("stime"))
            ttft_ms = j.get("first_token_time")
            tpot_ms = j.get("avg_time_between_tokens")
            if s_time is None or ttft_ms is None or tpot_ms is None:
                continue

            try:
                s_time_ms = float(s_time)     # 秒
                ttft_ms = float(ttft_ms)   # 毫秒
                tpot_ms = float(tpot_ms)   # 毫秒
            except Exception:
                continue

            abs_ts = s0_abs + timedelta(milliseconds=s_time_ms + ttft_ms)
            bucket = pd.to_datetime(abs_ts).floor("s")

            ttft_tpot_rows.append({
                "second": bucket,
                "TTFT_ms": ttft_ms,
                "TPOT_ms": tpot_ms
            })

# 每秒聚合：把“统计时间”落在这一秒的请求的 TTFT/TPOT 求均值
if ttft_tpot_rows:
    ttft_df = pd.DataFrame(ttft_tpot_rows).set_index("second").sort_index()
    ttft_agg = ttft_df.groupby(ttft_df.index).agg(
        avg_TTFT_ms=("TTFT_ms", "mean"),
        avg_TPOT_ms=("TPOT_ms", "mean"),
        count=("TTFT_ms", "size"),
    ).sort_index()
else:
    ttft_agg = pd.DataFrame(columns=["avg_TTFT_ms", "avg_TPOT_ms", "count"])

# ========= 实例名 =========
inst_names = sorted(per_inst_proc.keys(), key=lambda x: int(x.split("#")[1])) if per_inst_proc else []
if not inst_names and not alert_inst_df.empty:
    max_inst = int(alert_inst_df["inst"].max())
    inst_names = [f"Vllm#{i}" for i in range(max_inst + 1)]

def alert_per_inst_series(inst, col):
    if alert_inst_df.empty: return None
    try:
        idx = int(inst.split("#")[1])
    except Exception:
        return None
    df = alert_inst_df[alert_inst_df["inst"] == idx]
    if df.empty or col not in df: return None
    return df[col]

def batch_series(inst):
    if batches_wide.empty: return None
    try:
        idx = int(inst.split("#")[1])
    except Exception:
        return None
    col = f"batch_{idx}"
    return batches_wide[col] if col in batches_wide.columns else None

# ========= 绘制：每实例一张图（2 子图）=========
for inst in inst_names:
    proc_df = per_inst_proc.get(inst, pd.DataFrame())
    s_decode_lat = proc_df.get("decode_latency_avg")
    s_prefill_ratio = proc_df.get("prefill_ratio")
    s_decode_cnt = proc_df.get("decode_count")

    s_prefill_tokens = alert_per_inst_series(inst, "prefill_tokens")  # 左：Prefill Tokens per sec
    s_request_num    = alert_per_inst_series(inst, "request_num")     # 右：Request per sec
    s_batch          = batch_series(inst)                              # 外右：#Request in Instance

    idx_list = [s.index for s in [s_decode_lat, s_prefill_ratio, s_decode_cnt,
                                  s_prefill_tokens, s_request_num, s_batch] if s is not None and not s.empty]
    if not idx_list:
        continue
    time_index = idx_list[0]
    for idx in idx_list[1:]:
        time_index = time_index.union(idx)
    time_index = time_index.sort_values()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ----- 上半：三文鱼背景 + decode_latency(left) + decode_count(right)
    if s_prefill_ratio is not None and not s_prefill_ratio.empty:
        for t in s_prefill_ratio.index:
            ratio = s_prefill_ratio.loc[t]
            if pd.notna(ratio):
                ax_top.axvspan(t, t + pd.Timedelta(seconds=1), color=salmon_cmap(ratio), alpha=1.0)

    left_line = right_line = None
    if s_decode_lat is not None and not s_decode_lat.empty:
        left_line, = ax_top.plot(s_decode_lat.index, s_decode_lat.values, linewidth=LW_MAIN,
                                 label="Avg Decode Latency (left)", color=COLOR_DECODE_LAT)
        ax_top.set_ylabel("Decode Latency (ms)")

    ax_top_r = ax_top.twinx()
    if s_decode_cnt is not None and not s_decode_cnt.empty:
        right_line, = ax_top_r.plot(s_decode_cnt.index, s_decode_cnt.values, linewidth=LW_MAIN,
                                    label="Total Decode Count (right)", color=COLOR_DECODE_COUNT)
        ax_top_r.set_ylabel("Decode Count (req/s)")

    handles = [mpatches.Patch(color="salmon", label="Prefill Ratio (bg)")]
    if left_line is not None:
        handles.append(Line2D([0], [0], color=COLOR_DECODE_LAT, lw=LW_MAIN, label="Avg Decode Latency (left)"))
    if right_line is not None:
        handles.append(Line2D([0], [0], color=COLOR_DECODE_COUNT, lw=LW_MAIN, label="Decode Count (right)"))
    ax_top.legend(handles=handles, loc="upper left", fontsize=9, ncol=2)
    set_time_formatter(ax_top)

    # ----- 下半：三线三轴（显式配色；实例图保持 batch 为虚线可选，如需也改为实线可去掉 linestyle）
    ax_l = ax_bot
    ax_r = ax_bot.twinx()
    ax_rr = ax_bot.twinx()
    ax_rr.spines.right.set_position(("axes", 1.12))

    if s_prefill_tokens is not None and not s_prefill_tokens.empty:
        ax_l.plot(s_prefill_tokens.index, s_prefill_tokens.values, linewidth=LW_MAIN,
                  label="Prefill Tokens per sec", color=COLOR_PREFILL_TOKENS)
        ax_l.set_ylabel("Prefill Tokens per sec")
    if s_request_num is not None and not s_request_num.empty:
        ax_r.plot(s_request_num.index, s_request_num.values, linewidth=LW_MAIN,
                  label="Request per sec", color=COLOR_REQUEST_PERSEC)
        ax_r.set_ylabel("Request per sec")
    if s_batch is not None and not s_batch.empty:
        ax_rr.plot(s_batch.index, s_batch.values, linewidth=LW_MAIN,
                   label="#Request in Instance", color=COLOR_BATCH_INST)
        ax_rr.set_ylabel("#Request in Instance")

    bot_handles = []
    bot_handles.append(Line2D([0], [0], color=COLOR_PREFILL_TOKENS, lw=LW_MAIN, label="Prefill Tokens per sec"))
    bot_handles.append(Line2D([0], [0], color=COLOR_REQUEST_PERSEC,  lw=LW_MAIN, label="Request per sec"))
    bot_handles.append(Line2D([0], [0], color=COLOR_BATCH_INST,     lw=LW_MAIN, label="#Request in Instance"))
    ax_l.legend(handles=bot_handles, loc="upper left", fontsize=9, ncol=2)
    set_time_formatter(ax_bot)

    ax_top.set_xlim(time_index.min(), time_index.max())
    ax_bot.set_xlim(time_index.min(), time_index.max())
    ax_bot.set_xlabel("Time (seconds)")
    fig.suptitle(f"{inst} — Prefill/Decode (top) & Alert Metrics (bottom)")
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(GRAPH_PATH, f"instance_panel_{inst}.png")
    plt.savefig(out_path)
    print(f"Saved per-instance panel: {out_path}")
    plt.close(fig)

# ========= Total 汇总 =========
def to_wide_from_per_inst(colname):
    cols = {}
    for inst in inst_names:
        df = per_inst_proc.get(inst, pd.DataFrame())
        s = df.get(colname) if not df.empty else None
        if s is not None and not s.empty:
            cols[inst] = s.rename(inst)
    if not cols: return pd.DataFrame()
    return pd.concat(cols.values(), axis=1).sort_index()

decode_latency_wide = to_wide_from_per_inst("decode_latency_avg")
decode_count_wide   = to_wide_from_per_inst("decode_count")
prefill_ratio_wide  = to_wide_from_per_inst("prefill_ratio")

def safe_sum(df):  return df.sum(axis=1, min_count=1) if not df.empty else pd.Series(dtype=float)
def safe_mean(df): return df.mean(axis=1)            if not df.empty else pd.Series(dtype=float)

# total 指标
all_idx = []
for df in [decode_latency_wide, decode_count_wide, prefill_ratio_wide, alert_global_df, ttft_agg]:
    if isinstance(df, pd.DataFrame) and not df.empty:
        all_idx.append(df.index)
    elif isinstance(df, pd.Series) and not df.empty:
        all_idx.append(df.index)
if all_idx:
    idx_union = all_idx[0]
    for idx in all_idx[1:]:
        idx_union = idx_union.union(idx)
    idx_union = idx_union.sort_values()
else:
    idx_union = pd.Index([], name="second")

total_df = pd.DataFrame(index=idx_union)
total_df["avg_decode_latency"] = safe_mean(decode_latency_wide).reindex(idx_union)
total_df["total_decode_count"] = safe_sum(decode_count_wide).reindex(idx_union)
total_df["avg_prefill_ratio"]  = safe_mean(prefill_ratio_wide).reindex(idx_union)

# sum batch 供 total 中部“第三轴”
sum_batch = None
if not batches_wide.empty:
    sum_batch = batches_wide.filter(regex=r"^batch_\d+$").sum(axis=1, min_count=1).reindex(idx_union)

# ========= Total 图（3 子图：上合并面板 / 中 global 三轴（全部实线细线） / 下 TTFT&TPOT（绝对时间，实线细线））=========
fig2, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
ax_top2, ax_mid2, ax_ttft = axes  # 上 / 中 / 下

# --- 顶部：三文鱼背景( avg_prefill_ratio ) + avg_decode_latency(左) + total_decode_count(右)
if "avg_prefill_ratio" in total_df and total_df["avg_prefill_ratio"].notna().any():
    for t, ratio in total_df["avg_prefill_ratio"].dropna().items():
        ax_top2.axvspan(t, t + pd.Timedelta(seconds=1), color=salmon_cmap(ratio), alpha=1.0)

if "avg_decode_latency" in total_df:
    ax_top2.plot(total_df.index, total_df["avg_decode_latency"], linewidth=LW_MAIN,
                 label="Avg Decode Latency (left)", color=COLOR_DECODE_LAT)
    ax_top2.set_ylabel("Decode Latency (ms)")
ax_top2_r = ax_top2.twinx()
if "total_decode_count" in total_df:
    ax_top2_r.plot(total_df.index, total_df["total_decode_count"], linewidth=LW_MAIN,
                   label="Total Decode Count (right)", color=COLOR_DECODE_COUNT)
    ax_top2_r.set_ylabel("Decode Count (req/s)")

handles2 = [mpatches.Patch(color="salmon", label="Avg Prefill Ratio (bg)"),
            Line2D([0], [0], color=COLOR_DECODE_LAT, lw=LW_MAIN, label="Avg Decode Latency (left)"),
            Line2D([0], [0], color=COLOR_DECODE_COUNT, lw=LW_MAIN, label="Total Decode Count (right)")]
ax_top2.legend(handles=handles2, loc="upper left", fontsize=9, ncol=2)
set_time_formatter(ax_top2)

# --- 中部：三轴（全部实线 + 低线宽）：Global Prefill Tokens/s、Global Request Num、Sum Batch（若有）
ax_g_left  = ax_mid2                     # Global Prefill Tokens/s
ax_g_right = ax_mid2.twinx()             # Global Request Num
ax_g_outer = ax_mid2.twinx()             # Sum Batch
ax_g_outer.spines.right.set_position(("axes", 1.12))

g_handles = []
if not alert_global_df.empty and "global_prefill_tokens_per_sec" in alert_global_df:
    ax_g_left.plot(alert_global_df.index, alert_global_df["global_prefill_tokens_per_sec"],
                   label="Global Prefill Tokens/s", linewidth=LW_THIN, color=COLOR_PREFILL_TOKENS)
    ax_g_left.set_ylabel("Global Prefill Tokens/s")
    g_handles.append(Line2D([0], [0], color=COLOR_PREFILL_TOKENS, lw=LW_THIN, label="Global Prefill Tokens/s"))

if not alert_global_df.empty and "global_request_num" in alert_global_df:
    ax_g_right.plot(alert_global_df.index, alert_global_df["global_request_num"],
                    label="Global Request Num", linewidth=LW_THIN, color=COLOR_REQUEST_PERSEC)
    ax_g_right.set_ylabel("Global Request Num")
    g_handles.append(Line2D([0], [0], color=COLOR_REQUEST_PERSEC, lw=LW_THIN, label="Global Request Num"))

if sum_batch is not None and not sum_batch.empty:
    ax_g_outer.plot(sum_batch.index, sum_batch.values,
                    label="Sum Batch", linewidth=LW_THIN, color=COLOR_BATCH_INST)
    ax_g_outer.set_ylabel("Sum Batch")
    g_handles.append(Line2D([0], [0], color=COLOR_BATCH_INST, lw=LW_THIN, label="Sum Batch"))

if g_handles:
    ax_g_left.legend(handles=g_handles, loc="upper left", fontsize=9, ncol=3)
set_time_formatter(ax_mid2)

# --- 底部：TTFT/TPOT（使用绝对日志时间；两条实线细线；每秒桶聚合均值）
if not ttft_agg.empty:
    ax_ttft_l = ax_ttft
    ax_ttft_r = ax_ttft.twinx()

    ax_ttft_l.plot(ttft_agg.index, ttft_agg["avg_TTFT_ms"], color=COLOR_TTFT, linewidth=LW_THIN, label="Avg TTFT (ms)")
    ax_ttft_r.plot(ttft_agg.index, ttft_agg["avg_TPOT_ms"], color=COLOR_TPOT, linewidth=LW_THIN, label="Avg TPOT (ms)")

    ax_ttft_l.set_ylabel("Avg TTFT (ms)")
    ax_ttft_r.set_ylabel("Avg TPOT (ms)")
    ax_ttft_l.legend(handles=[
        Line2D([0],[0], color=COLOR_TTFT, lw=LW_THIN, label="Avg TTFT (ms)"),
        Line2D([0],[0], color=COLOR_TPOT, lw=LW_THIN, label="Avg TPOT (ms)")
    ], loc="upper left", fontsize=9)
    set_time_formatter(ax_ttft_l)
    ax_ttft.set_xlabel("Time (seconds)")
else:
    ax_ttft.text(0.5, 0.5, "No TTFT/TPOT data (or s_time=0 anchor not found)", ha="center", va="center", transform=ax_ttft.transAxes)
    ax_ttft.axis("off")

fig2.suptitle("Total Metrics — Combined Panels (TTFT/TPOT aligned to log time)")
fig2.autofmt_xdate()
plt.tight_layout(rect=[0, 0, 1, 0.97])
out_global = os.path.join(GRAPH_PATH, "global_metrics.png")
plt.savefig(out_global)
print(f"Saved total figure: {out_global}")
print("All done.")
