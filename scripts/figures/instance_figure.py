import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
from collections import defaultdict
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import subprocess

# 日志路径
#LOG_FILE = "/data/lmetric/pack/2024_conv_2min_lwl_2/processed.log"
PREFIX = "/mnt/debugger/hjb/lmmetric-logs/dp8_24_1conv/rr/"
LOG_FILE = f"{PREFIX}router_v2.log"
output_file = f"{PREFIX}processed.log"

with open(output_file, 'w') as outfile:
    result = subprocess.run(['grep', 'VllmMetric', LOG_FILE], stdout=outfile)

LOG_FILE = output_file

GRAPH_PATH = os.path.join(os.path.dirname(LOG_FILE), "instance_fig/")
os.makedirs(GRAPH_PATH, exist_ok=True)

# 设置颜色渐变 colormap：白色 → 三文鱼色
cmap = mcolors.LinearSegmentedColormap.from_list("prefill_cmap", ["white", "salmon"])

# 设置绘图风格
sns.set(style="whitegrid")

# 日志正则提取模式
log_pattern = re.compile(
    r"(?P<timestamp>[\d\-\:TZ\.]+).*?Vllm#(?P<instance_id>\d+)::Event::data received VllmMetric { prefill_tokens: (?P<prefill_tokens>\d+),.*?latency: (?P<latency>\d+), outputs: \[(?P<outputs>.*?)\] }"
)
output_pattern = re.compile(
    r"request_id: (?P<request_id>\d+), new_token_ids: \[.*?\], state: \"(?P<state>PREFILL|DECODE)\", is_finished: (true|false)"
)

# 解析日志
instance_data = defaultdict(list)

with open(LOG_FILE, "r") as f:
    for line in f:
        match = log_pattern.search(line)
        if not match:
            continue

        timestamp = datetime.fromisoformat(match.group("timestamp").replace("Z", "+00:00"))
        instance_id = f"Vllm#{match.group('instance_id')}"
        latency = int(match.group("latency"))
        outputs = match.group("outputs")

        for output in output_pattern.finditer(outputs):
            state = output.group("state")
            request_id = int(output.group("request_id"))

            instance_data[instance_id].append({
                "time": timestamp,
                "state": state,
                "latency": latency,
                "request_id": request_id,
            })

# 分析每个 Vllm 实例
for instance_id, events in instance_data.items():
    df = pd.DataFrame(events)

    df["time"] = pd.to_datetime(df["time"])
    df["second"] = df["time"].dt.floor("s")

    # 1. 平均 decode latency
    agg = df.groupby("second").agg(
        decode_latency_avg=("latency", lambda x: x[df.loc[x.index, "state"] == "DECODE"].mean())
    )

    # 2. prefill latency ratio (0~1)
    prefill_ratios = df[df["state"] == "PREFILL"].groupby("second")["latency"].sum() / 1000.0
    prefill_ratios = prefill_ratios.clip(upper=1.0)
    agg["prefill_ratio"] = prefill_ratios
    agg["prefill_ratio"] = agg["prefill_ratio"].fillna(0.0)

    # 3. decode count (按秒去重 request_id)
    decode_df = df[df["state"] == "DECODE"].copy()
    decode_count_series = decode_df.drop_duplicates(subset=["second", "request_id"]).groupby("second").size()
    agg["decode_count"] = decode_count_series
    agg["decode_count"] = agg["decode_count"].fillna(0).astype(int)

    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # 背景色：根据 prefill_ratio 颜色渐变
    for time_point, ratio in zip(agg.index, agg["prefill_ratio"]):
        color = cmap(ratio)
        ax1.axvspan(time_point, time_point + pd.Timedelta(seconds=1), color=color, alpha=1.0)

    # 折线图：decode latency（左轴）
    ax1.set_ylim(0, 100)
    ax1.plot(
        agg.index, agg["decode_latency_avg"],
        label="Avg Decode Latency (ms)",
        color="green",
        linewidth=2,
    )
    ax1.set_ylabel("Decode Latency (ms)")

    # 折线图：decode count（右轴）
    ax2.set_ylim(0, 50)
    ax2.plot(
        agg.index, agg["decode_count"],
        label="Decode Count (req/s)",
        color="orange",
        linewidth=2,
    )
    ax2.set_ylabel("Decode Count (req/s)")

    # 时间轴格式
    ax1.set_xlim(agg.index.min(), agg.index.max())
    ax1.set_xlabel("Time (seconds)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    # 图例
    bg_patch = mpatches.Patch(color="salmon", label="Prefill Latency Ratio (bg)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend([bg_patch] + lines1 + lines2,
               ["Prefill Latency Ratio (bg)"] + labels1 + labels2,
               loc="upper left", fontsize=10)

    ax1.set_title(f"System Behavior - {instance_id}")

    plt.tight_layout()
    plt.savefig(f"{GRAPH_PATH}instance_figure_{instance_id}.png")
    #plt.show()
    print(f"{instance_id=} finished!")
