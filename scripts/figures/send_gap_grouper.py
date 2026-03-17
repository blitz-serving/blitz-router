import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse

"""
传入两个目录，分别包含 VLLM 客户端和 Sim 客户端的输出 JSONL 文件。
仅对文件名中包含 "code" 的 .jsonl 文件进行分析。
每个 JSONL 文件包含 "send_gap" 字段（单位：毫秒）。
脚本会聚合两个目录中的这些文件，对比 send_gap 分布，
并在同一个柱状图中展示两个分布。
"""

parser = argparse.ArgumentParser()
parser.add_argument("--client1", type=str, required=True, help="第一目录 (VLLM)")
parser.add_argument("--client2", type=str, required=True, help="第二目录 (Sim)")
parser.add_argument("--out", type=str, required=True, help="输出图像前缀路径")
args = parser.parse_args()

dir_a = args.client1  # VLLM
dir_b = args.client2  # Sim
output_prefix = args.out
field = "send_gap"


# ======== 从目录加载所有包含 "code" 的 JSONL 的 send_gap ========
def load_field_from_dir(directory, field_name):
    """返回目录中所有 send_gap 值的列表，仅包含文件名含 'code' 的文件"""
    all_values = []
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    jsonl_files = [f for f in jsonl_files if "code" in os.path.basename(f)]

    if not jsonl_files:
        print(f"⚠️ Warning: No .jsonl files with 'code' found in {directory}")
    else:
        print(f"Scanning {len(jsonl_files)} 'code' jsonl files in {directory}")

    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if field_name in obj:
                        all_values.append(float(obj[field_name]))
                except Exception as e:
                    print(f"Error parsing line in {file_path}: {e}")
    return all_values


# ======== 加载数据 ========
send_gaps_a = load_field_from_dir(dir_a, field)
send_gaps_b = load_field_from_dir(dir_b, field)

print(f"Loaded {len(send_gaps_a)} send_gap values from {dir_a}")
print(f"Loaded {len(send_gaps_b)} send_gap values from {dir_b}")

if not send_gaps_a and not send_gaps_b:
    print("No send_gap data found in either directory.")
    exit(0)

# ======== 定义桶（bins） ========
bins = list(np.arange(0, 101, 10)) + [np.inf]
labels = [
    f"{bins[i]}-{bins[i+1]}ms" if bins[i+1] != np.inf else f">{bins[i]}ms"
    for i in range(len(bins) - 1)
]

# ======== 计算两个目录的直方图（归一化为比例） ========
hist_a, _ = np.histogram(send_gaps_a, bins=bins)
hist_b, _ = np.histogram(send_gaps_b, bins=bins)

total_a = hist_a.sum() if hist_a.sum() > 0 else 1
total_b = hist_b.sum() if hist_b.sum() > 0 else 1
hist_a = hist_a / total_a
hist_b = hist_b / total_b

# ======== 绘制对比柱状图 ========
x = np.arange(len(labels))
bar_width = 0.4

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width/2, hist_a * 100, width=bar_width, label="oldclient", color="#1f77b4", edgecolor="black")
plt.bar(x + bar_width/2, hist_b * 100, width=bar_width, label="newclient", color="#ff7f0e", edgecolor="black")

plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Percentage of Requests (%)")
plt.xlabel("Send Gap (ms)")
plt.title("Distribution Comparison of Send Gap (VLLM vs Sim) [code files only]")
plt.legend()
plt.tight_layout()

output_plot = f"{output_prefix}_send_gap_code_comparison.png"
plt.savefig(output_plot)
plt.close()
print(f"✅ Send gap comparison plot saved to {output_plot}")

# ======== 打印统计信息 ========
def print_stats(name, data):
    if not data:
        print(f"{name}: No data.")
        return
    arr = np.array(data)
    print(f"{name} statistics:")
    print(f"  Mean send_gap: {np.mean(arr):.2f} ms")
    print(f"  Median send_gap: {np.median(arr):.2f} ms")
    print(f"  90th percentile: {np.percentile(arr, 90):.2f} ms")
    print(f"  Max send_gap: {np.max(arr):.2f} ms")
    print()

print_stats("VLLM", send_gaps_a)
print_stats("Sim", send_gaps_b)
