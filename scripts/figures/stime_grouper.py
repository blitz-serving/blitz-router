import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
"""
传入两个目录，分别包含 vLLM 客户端和 Sim 客户端的输出 JSONL 文件。
每个 JSONL 文件包含多行 JSON 对象，每个对象包含一个 "s_time" 字段，表示请求的时间戳（单位：毫秒）。
脚本会比较两个目录中同名文件的 "s_time" 列表，计算绝对误差，并绘制误差分布图。
"""
parser = argparse.ArgumentParser()
parser.add_argument("--client1", type=str, required=True, help="第一目录 (VLLM)")
parser.add_argument("--client2", type=str, required=True, help="第二目录 (Sim)")
parser.add_argument("--out", type=str, required=True, help="输出图像前缀路径")
args = parser.parse_args()

dir_a = args.client1  # VLLM
dir_b = args.client2  # Sim
output_prefix = args.out
field = "s_time"

# ======== 加载目录中所有 .jsonl 文件的 s_time 列表 ========
def load_stime_from_dir(directory):
    """返回 {filename: [s_time1, s_time2, ...]}"""
    data = {}
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in {directory}")
    for file_path in jsonl_files:
        fname = os.path.basename(file_path)
        s_times = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if field in obj:
                        s_times.append(float(obj[field]))
                except Exception as e:
                    print(f"Error parsing line in {file_path}: {e}")
        if s_times:
            data[fname] = sorted(s_times)  # ✅ 排序 s_time
    return data

# ======== 加载两个目录的数据 ========
data_a = load_stime_from_dir(dir_a)
data_b = load_stime_from_dir(dir_b)

# ======== 仅比较两个目录中同名的 jsonl 文件 ========
common_files = sorted(list(set(data_a.keys()) & set(data_b.keys())))
print(f"Found {len(common_files)} common files to compare.")

if not common_files:
    print("No common JSONL files found. Exiting.")
    exit(0)

# ======== 聚合所有文件的绝对误差 ========
abs_errors = []

for fname in common_files:
    list_a = data_a[fname]
    list_b = data_b[fname]
    n = min(len(list_a), len(list_b))
    if n == 0:
        print(f"Skipping {fname} (no matching records).")
        continue

    # 排序后逐行比较（已排序）
    for i in range(n):
        va, vb = list_a[i], list_b[i]
        abs_errors.append(abs(vb - va))

print(f"Aggregated {len(abs_errors)} sorted s_time comparisons across {len(common_files)} files.")

# ======== 绘制整体误差分布 ========
if abs_errors:
    bins = list(np.arange(0, 501, 10)) + [np.inf]
    labels = [f"{bins[i]}-{bins[i+1]}ms" if bins[i+1] != np.inf else f">{bins[i]}ms"
              for i in range(len(bins)-1)]

    hist, _ = np.histogram(abs_errors, bins=bins)
    hist = hist / len(abs_errors)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(hist)), hist, width=0.8, align="center")
    plt.xticks(range(len(hist)), labels, rotation=45, ha="right")
    plt.ylabel("Proportion of Requests")
    plt.xlabel("Absolute Error (ms)")
    plt.title(f"Aggregated Absolute Error Distribution for sorted s_time (Sim vs VLLM)")
    plt.tight_layout()

    output_plot = f"{output_prefix}_s_time_abs_error_sorted.png"
    plt.savefig(output_plot)
    plt.close()
    print(f"✅ Aggregated sorted absolute error plot saved to {output_plot}")

    # 打印统计信息
    arr = np.array(abs_errors)
    print(f"Mean abs error: {np.mean(arr):.2f} ms")
    print(f"Median abs error: {np.median(arr):.2f} ms")
    print(f"90th percentile: {np.percentile(arr, 90):.2f} ms")
    print(f"Max abs error: {np.max(arr):.2f} ms")
else:
    print("No absolute error data for s_time.")
