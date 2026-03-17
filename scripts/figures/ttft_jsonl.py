import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--client1", type=str, required=True, help="第一目录 (VLLM)")
parser.add_argument("--client2", type=str, required=True, help="第二目录 (Sim)")
parser.add_argument("--out", type=str, required=True, help="输出图像前缀路径")
parser.add_argument("--label1", type=str, default="VLLM", help="第一目录标签")
parser.add_argument("--label2", type=str, default="Sim", help="第二目录标签")
args = parser.parse_args()

father = args.out
dir_a = args.client1  # VLLM
dir_b = args.client2  # Sim
label1 = args.label1
label2 = args.label2
output_prefix = father

# 关心的字段
fields = ["first_token_time", "avg_time_between_tokens", "s_time"]

# =====================
# 修改点 1: 不再用 client_id 区分，而是按文件聚合所有数据
# =====================
def load_jsonl_by_file(directory, fields):
    data = defaultdict(list)  # data[fname] = [ {field1: val1, field2: val2, ...}, ... ]
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in {directory}")
    for file_path in jsonl_files:
        fname = os.path.basename(file_path)
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    record = {}
                    for field in fields:
                        if field in obj:
                            record[field] = float(obj[field])
                    if record:
                        data[fname].append(record)
                except Exception as e:
                    print(f"Error parsing line in {file_path}: {e}")
                    continue
    return data

# 加载数据
data_a = load_jsonl_by_file(dir_a, fields)  # VLLM
data_b = load_jsonl_by_file(dir_b, fields)  # Sim

# =====================
# 绘制 CDF 图
# =====================
linestyles = {label1: "-", label2: "--"}  # 实线 vs 虚线

fields_cdf = ["first_token_time", "avg_time_between_tokens"]
for field in fields_cdf:
    plt.figure(figsize=(8, 6))
    for dataset, label in [(data_a, label1), (data_b, label2)]:
        arr = []
        for fname in dataset:
            for record in dataset[fname]:
                if field in record:
                    arr.append(record[field])
        arr = np.array(arr)
        if len(arr) == 0:
            print(f"No data for {field} in {label}")
            continue
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)

        mean_val = np.mean(arr)
        legend_label = f"{field} ({label}) - mean: {mean_val:.2f} ms"

        plt.plot(sorted_arr, cdf,
                 linestyle=linestyles[label],
                 label=legend_label)

    plt.xlabel("Value (ms)")
    plt.ylabel("CDF")
    plt.title(f"CDF Comparison ({field}) ({label1}) vs ({label2})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_plot = f"{output_prefix}_{field}.png"
    plt.savefig(output_plot)
    plt.close()
    print(f"Plot saved to {output_plot}")

# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import glob
# import argparse
# from collections import defaultdict

# parser = argparse.ArgumentParser()
# parser.add_argument("--client1", type=str, required=True, help="第一目录 (VLLM)")
# parser.add_argument("--client2", type=str, required=True, help="第二目录 (Sim)")
# parser.add_argument("--out", type=str, required=True, help="输出图像前缀路径")
# args = parser.parse_args()

# father = args.out
# dir_a = args.client1  # VLLM
# dir_b = args.client2  # Sim
# output_prefix = father

# # 关心的字段
# fields = ["first_token_time", "avg_time_between_tokens", "s_time"]

# # 按 文件名 + client_id 加载数据
# def load_jsonl_by_file_and_clientid(directory, fields):
#     data = defaultdict(lambda: defaultdict(dict))  
#     jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
#     if not jsonl_files:
#         print(f"Warning: No .jsonl files found in {directory}")
#     for file_path in jsonl_files:
#         fname = os.path.basename(file_path)
#         with open(file_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     obj = json.loads(line)
#                     cid = obj.get("client_id")  # 用 client_id
#                     if cid is None:
#                         continue
#                     for field in fields:
#                         val = float(obj.get(field, 0))
#                         data[fname][cid][field] = val
#                 except Exception as e:
#                     print(f"Error parsing line in {file_path}: {e}")
#                     continue
#     return data

# # 加载数据
# data_a = load_jsonl_by_file_and_clientid(dir_a, fields)  # VLLM
# data_b = load_jsonl_by_file_and_clientid(dir_b, fields)  # Sim

# # =====================
# # 绘制 CDF 图（逻辑保持不变）
# # =====================
# linestyles = {"VLLM": "-", "Sim": "--"}  # 实线 vs 虚线

# fields = ["first_token_time", "avg_time_between_tokens"]
# for field in fields:
#     plt.figure(figsize=(8, 6))
#     for dataset, label in [(data_a, "VLLM"), (data_b, "Sim")]:
#         arr = []
#         for fname in dataset:
#             for cid in dataset[fname]:
#                 if field in dataset[fname][cid]:
#                     arr.append(dataset[fname][cid][field])
#         arr = np.array(arr)
#         if len(arr) == 0:
#             print(f"No data for {field} in {label}")
#             continue
#         sorted_arr = np.sort(arr)
#         cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)

#         mean_val = np.mean(arr)
#         legend_label = f"{field} ({label}) - mean: {mean_val:.2f} ms"

#         plt.plot(sorted_arr, cdf,
#                  linestyle=linestyles[label],
#                  label=legend_label)

#     plt.xlabel("Value (ms)")
#     plt.ylabel("CDF")
#     plt.title(f"CDF Comparison ({field}) VLLM vs Sim")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     output_plot = f"{output_prefix}_{field}.png"
#     plt.savefig(output_plot)
#     plt.close()
#     print(f"Plot saved to {output_plot}")

# # =====================
# # 计算 Error Rate
# # =====================
# error_rates = {field: [] for field in fields}
# error_rids_over_100 = {field: [] for field in fields}

# # 遍历同名文件
# common_files = set(data_a.keys()) & set(data_b.keys())
# print(f"Found {len(common_files)} common files")

# for fname in common_files:
#     cids = set(data_a[fname].keys()) & set(data_b[fname].keys())
#     for cid in cids:
#         for field in fields:
#             va = data_a[fname][cid].get(field, None)  # VLLM
#             vb = data_b[fname][cid].get(field, None)  # Sim
#             if va is None or vb is None or va == 0:
#                 continue
#             err = abs(vb - va) / va * 100  # 以 VLLM 为基准
#             error_rates[field].append(err)
#             if err > 100:
#                 error_rids_over_100[field].append((fname, cid))

# # =====================
# # 绘制 Error Rate 分布柱状图
# # =====================
# for field in fields:
#     errs = error_rates[field]
#     if not errs:
#         print(f"No error rate data for {field}")
#         continue

#     bins = np.arange(0, 105, 5).tolist() + [np.inf]
#     labels = [f"{bins[i]}-{bins[i+1]}%" if bins[i+1] != np.inf else f">{bins[i]}%" 
#               for i in range(len(bins)-1)]
#     hist, _ = np.histogram(errs, bins=bins)
#     hist = hist / len(errs)

#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(hist)), hist, width=0.8, align="center")
#     plt.xticks(range(len(hist)), labels, rotation=45, ha="right")
#     plt.ylabel("Proportion of Requests")
#     plt.xlabel("Error Rate Bucket")
#     plt.title(f"Error Rate Distribution for {field} (Sim vs VLLM)")
#     plt.tight_layout()

#     output_plot = f"{output_prefix}_{field}_error_rate.png"
#     plt.savefig(output_plot)
#     plt.close()
#     print(f"Error rate plot saved to {output_plot}")

#     if error_rids_over_100[field]:
#         print(f"[Warning] {field} has {len(error_rids_over_100[field])} requests with error rate > 100%")
#         for fname, cid in error_rids_over_100[field]:
#             print(f"  - file: {fname}, client_id: {cid}")


# for field in fields:
#     signed_error_rates = []

#     for fname in common_files:
#         cids = set(data_a[fname].keys()) & set(data_b[fname].keys())
#         for cid in cids:
#             va = data_a[fname][cid].get(field, None)  # VLLM
#             vb = data_b[fname][cid].get(field, None)  # Sim
#             if va is None or vb is None or va == 0:
#                 continue
#             # 不取绝对值
#             err_signed = (vb - va) / va * 100
#             signed_error_rates.append(err_signed)

#     if signed_error_rates:
#         # 定义区间：-100% 到 100% 每 5% 一个 bucket，额外一个 "<-100%" 和 ">100%"
#         bins = [-np.inf] + list(np.arange(-100, 105, 5)) + [np.inf]
#         labels = []
#         for i in range(len(bins) - 1):
#             if bins[i] == -np.inf:
#                 labels.append(f"<{bins[i+1]}%")
#             elif bins[i+1] == np.inf:
#                 labels.append(f">{bins[i]}%")
#             else:
#                 labels.append(f"{bins[i]}-{bins[i+1]}%")

#         hist, _ = np.histogram(signed_error_rates, bins=bins)
#         hist = hist / len(signed_error_rates)

#         plt.figure(figsize=(12, 6))
#         plt.bar(range(len(hist)), hist, width=0.8, align="center")
#         plt.xticks(range(len(hist)), labels, rotation=45, ha="right")
#         plt.ylabel("Proportion of Requests")
#         plt.xlabel("Signed Error Rate Bucket")
#         plt.title(f"Signed Error Rate Distribution for {field} (Sim vs VLLM)")
#         plt.tight_layout()

#         output_plot = f"{output_prefix}_{field}_signed_error_rate.png"
#         plt.savefig(output_plot)
#         plt.close()
#         print(f"Signed error rate plot saved to {output_plot}")
#     else:
#         print(f"No signed error rate data for {field}")


# fields = ["s_time"]
# for field in fields:
#     abs_errors = []

#     for fname in common_files:
#         cids = set(data_a[fname].keys()) & set(data_b[fname].keys())
#         for cid in cids:
#             va = data_a[fname][cid].get(field, None)  # VLLM
#             vb = data_b[fname][cid].get(field, None)  # Sim
#             if va is None or vb is None:
#                 continue
#             err_abs = abs(vb - va)
#             abs_errors.append(err_abs)

#     if abs_errors:
#         # 自动选择区间：0~2000ms 每100ms一个bucket，多余的归入 ">2000ms"
#         bins = list(np.arange(0, 501, 10)) + [np.inf]
#         labels = [f"{bins[i]}-{bins[i+1]}ms" if bins[i+1] != np.inf else f">{bins[i]}ms"
#                   for i in range(len(bins)-1)]

#         hist, _ = np.histogram(abs_errors, bins=bins)
#         hist = hist / len(abs_errors)

#         plt.figure(figsize=(12, 6))
#         plt.bar(range(len(hist)), hist, width=0.8, align="center")
#         plt.xticks(range(len(hist)), labels, rotation=45, ha="right")
#         plt.ylabel("Proportion of Requests")
#         plt.xlabel("Absolute Error (ms)")
#         plt.title(f"Absolute Error Distribution for {field} (Sim vs VLLM)")
#         plt.tight_layout()

#         output_plot = f"{output_prefix}_{field}_abs_error.png"
#         plt.savefig(output_plot)
#         plt.close()
#         print(f"Absolute error plot saved to {output_plot}")
#     else:
#         print(f"No absolute error data for {field}")