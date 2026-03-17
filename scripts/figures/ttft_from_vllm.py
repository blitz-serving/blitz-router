import re
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

### =====================================================
### 从 vLLM 多日志文件提取 TTFT 和 TBT，并绘制 CDF 对比
### =====================================================
"""
这是一个用于从 vLLM 多日志文件中提取 TTFT 和 TBT 的 Python 工具脚本。
它可以处理多个日志文件，并将提取的结果绘制成 CDF 图。
输入是两个运行前缀目录，分别包含多个 vllm*.log 文件。
输出是两张 CDF 对比图，分别展示 TTFT 和 TBT 的分布情况。
用法示例:
    python scripts/figures/ttft_from_vllm.py --log_a <runA_prefix> --log_b <runB_prefix> --out <output_prefix>
例如:
    python scripts/figures/ttft_from_vllm.py --log_a /nvme/logs/run1 --log_b /nvme/logs/run2 --out /nvme/figures/ttft_compare
"""


# -----------------------------
# 工具函数：解析 ISO 时间 → 秒（float）
# -----------------------------
def parse_ts(ts_str: str) -> float:
    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
    return dt.timestamp()


# -----------------------------
# 解析单个日志文件
# -----------------------------
def parse_single_vllm_log(log_path: str, ttft, tbt):
    req_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z: request (\d+), tokens: (\d+), max_tokens: (\d+)"
    )
    sse_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z: SSE Event pushed: data: (.+)"
    )

    req_info = {}      # request_id -> {"ts": float, "max_tokens": int}
    prefill_done = {}  # request_id -> prefill结束时间

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                # 解析 request 到达时间
                m1 = req_pattern.search(line)
                if m1:
                    ts_str, req_id, tokens, max_tokens = m1.groups()
                    req_info[int(req_id)] = {
                        "ts": parse_ts(ts_str),
                        "max_tokens": int(max_tokens),
                    }
                    continue

                # 解析 SSE 事件
                m2 = sse_pattern.search(line)
                if m2:
                    ts_str, json_data = m2.groups()
                    ts = parse_ts(ts_str)
                    try:
                        data = json.loads(json_data)
                    except json.JSONDecodeError:
                        continue

                    if "outputs" not in data:
                        continue

                    for out in data["outputs"]:
                        req_id = out.get("request_id")
                        state = out.get("state")
                        finished = out.get("is_finished")

                        if req_id not in req_info:
                            continue

                        # TTFT：prefill - request_start
                        if state == "PREFILL":
                            start_ts = req_info[req_id]["ts"]
                            ttft_val = ts - start_ts
                            ttft.append(ttft_val)
                            prefill_done[req_id] = ts

                        # TBT：最后一次decode结束 - prefill结束
                        if finished:
                            if req_id in prefill_done:
                                pre_ts = prefill_done[req_id]
                                total_decode_time = ts - pre_ts
                                max_tokens = req_info[req_id]["max_tokens"]
                                if max_tokens > 1:
                                    tbt_val = total_decode_time / (max_tokens - 1)
                                    tbt.append(tbt_val)
    except FileNotFoundError:
        print(f"⚠️ 文件未找到: {log_path}")
    except Exception as e:
        print(f"⚠️ 解析 {log_path} 出错: {e}")


# -----------------------------
# 遍历一个运行前缀目录（包含多个日志）
# -----------------------------
def parse_vllm_run(prefix_dir: str):
    pattern = os.path.join(prefix_dir, "vllm*.log")
    log_files = sorted(glob.glob(pattern))
    if not log_files:
        print(f"⚠️ 未找到匹配文件: {pattern}")
        return [], []

    ttft_all, tbt_all = [], []
    for log_path in log_files:
        print(f"  解析日志: {os.path.basename(log_path)}")
        parse_single_vllm_log(log_path, ttft_all, tbt_all)

    print(f"✅ 共解析 {len(ttft_all)} 条 TTFT, {len(tbt_all)} 条 TBT\n")
    return ttft_all, tbt_all


# -----------------------------
# 绘制并保存 CDF 图
# -----------------------------
def plot_cdf_and_save(data_a, data_b, title, xlabel, label_a, label_b, output_prefix, filename):
    def cdf_data(data):
        data = np.sort(np.array(data))
        p = np.arange(len(data)) / float(len(data))
        return data, p

    if not data_a or not data_b:
        print(f"⚠️ 数据不足，跳过 {title}")
        return

    x1, y1 = cdf_data(data_a)
    x2, y2 = cdf_data(data_b)

    mean_a = np.mean(data_a)
    mean_b = np.mean(data_b)

    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, label=f"{label_a} (mean={mean_a*1000:.1f} ms)", linewidth=2)
    plt.plot(x2, y2, label=f"{label_b} (mean={mean_b*1000:.1f} ms)", linewidth=2, linestyle="--")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    output_path = f"{output_prefix}_{filename}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ 图像已保存: {output_path}")
    print(f"   平均值 - {label_a}: {mean_a*1000:.2f} ms, {label_b}: {mean_b*1000:.2f} ms\n")


# -----------------------------
# 命令行入口
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare TTFT/TBT CDF between two vLLM runs.")
    parser.add_argument("--log_a", type=str, required=True, help="Path prefix for run A (contains vllm*.log)")
    parser.add_argument("--log_b", type=str, required=True, help="Path prefix for run B (contains vllm*.log)")
    parser.add_argument("--out", type=str, required=True, help="Output file prefix (no extension)")
    args = parser.parse_args()

    print(f"=== 解析运行 A: {args.log_a} ===")
    ttft_a, tbt_a = parse_vllm_run(args.log_a)

    print(f"=== 解析运行 B: {args.log_b} ===")
    ttft_b, tbt_b = parse_vllm_run(args.log_b)

    # 绘制并保存对比图
    plot_cdf_and_save(
        ttft_a, ttft_b,
        "TTFT CDF Comparison",
        "TTFT (seconds)",
        "Run A", "Run B",
        args.out, "ttft_cdf"
    )

    plot_cdf_and_save(
        tbt_a, tbt_b,
        "TBT CDF Comparison",
        "TBT (seconds/token)",
        "Run A", "Run B",
        args.out, "tbt_cdf"
    )
