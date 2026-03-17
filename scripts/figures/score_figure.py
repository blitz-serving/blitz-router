import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict
import seaborn as sns

sns.set(style="whitegrid")

LOG_FILE = "/data/lmetric/pack/new_lwl_4_2/score.log"  # 修改为你的日志路径

# 匹配日志内容
log_pattern = re.compile(
    r"(?P<timestamp>[\d\-T:\.]+)Z.*?Vllm#(?P<instance_id>\d+): token_num_mili_rcd_mtx = \[(?P<matrix>.*?)\]"
)
matrix_item_pattern = re.compile(r"\((\-?\d+),\s*([\d\.Ee+-]+)\)")

# 数据结构：每个位置 0~7 的时间序列
position_data = defaultdict(list)  # key: position index 0~7, value: list of (timestamp, value)

with open(LOG_FILE, "r") as f:
    for line in f:
        match = log_pattern.search(line)
        if not match:
            continue

        timestamp = datetime.fromisoformat(match.group("timestamp"))
        matrix_str = match.group("matrix")

        matrix_matches = matrix_item_pattern.findall(matrix_str)
        if len(matrix_matches) != 8:
            continue  # 确保是 8 个点

        for idx, (token_str, _) in enumerate(matrix_matches):
            token_num = int(token_str)
            position_data[idx].append((timestamp, token_num))

# 创建图像
fig, ax = plt.subplots(figsize=(14, 6))

# 颜色和线型自动分配
for idx in range(8):
    data = position_data[idx]
    if not data:
        continue
    timestamps, values = zip(*data)
    ax.plot(timestamps, values, label=f"Instance {idx}", linewidth=1)

# 时间格式
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()

ax.set_ylim(0, 1000)
all_times = [t for data in position_data.values() for (t, _) in data]
if all_times:
    start_time = min(all_times) + pd.Timedelta(seconds=45)
    ax.set_xlim(start_time, start_time + pd.Timedelta(seconds=10))
ax.set_xlabel("Time")
ax.set_ylabel("prefill latency(predicted) (ms)")
ax.set_title("prefill latency(predicted) per Instance Over Time")
ax.legend(loc="upper right", title="Matrix Index")

plt.tight_layout()
plt.savefig("tmp/score_4_2_lwl.png")

