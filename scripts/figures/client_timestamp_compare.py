#!/usr/bin/env python3
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os

"""
这是一个用于分析和比较两组请求到达时间分布的 Python 工具脚本。
它可以从 vLLM 日志文件或 trace CSV 文件中提取请求的时间戳信息，然后：
    1. 计算相对时间；
    2. 绘制时间分布图；
    3. 分析两组数据之间的时间差异。

input:  {sys.argv[0]} <log1> <log2> [scale_factor] [segment_seconds]
"""

LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z\s+INFO.*add request\s+(?P<req_id>\d+)"
)

# ========== 解析函数 ==========

def parse_log(path):
    """解析 vLLM 日志格式，返回 request_id -> timestamp"""
    result = {}
    with open(path, "r") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                ts = datetime.strptime(m.group("timestamp"), "%Y-%m-%dT%H:%M:%S.%f")
                req_id = int(m.group("req_id"))
                result[req_id] = ts
    return result

def parse_trace_csv(path, limit=None, scale_factor=1.0):
    """
    解析 trace CSV 格式:
    TIMESTAMP,ContextTokens,GeneratedTokens
    scale 时间戳：scaled = (ts - base) / scale_factor
    """
    df = pd.read_csv(path)
    if "TIMESTAMP" not in df.columns:
        raise ValueError("CSV 文件中缺少 TIMESTAMP 列")

    # 截取前 limit 条
    if limit is not None:
        df = df.head(limit)

    # 转换时间戳
    df["ts"] = pd.to_datetime(df["TIMESTAMP"], format="%Y-%m-%d %H:%M:%S.%f")

    base_time = df["ts"].iloc[0]
    df["time_ms"] = (df["ts"] - base_time).dt.total_seconds() * 1000 / scale_factor

    # 构造与日志格式相同的数据结构
    result = {i: df["ts"].iloc[i] for i in range(len(df))}
    rel_times = [(i, df["time_ms"].iloc[i]) for i in range(len(df))]
    return pd.DataFrame(rel_times, columns=["req_id", "time_ms"])

def to_relative_times(req_times):
    """将日志解析结果转换为相对时间（ms）"""
    sorted_items = sorted(req_times.items())
    base_time = sorted_items[0][1]
    rel_times = [(rid, (ts - base_time).total_seconds() * 1000) for rid, ts in sorted_items]
    return pd.DataFrame(rel_times, columns=["req_id", "time_ms"])

# ========== 绘图函数 ==========

def plot_timeline_segmented(df1, df2, label1, label2, total_ms=240_000, interval_ms=30, segment_sec=20):
    """分段绘制请求时间分布"""
    bins = np.arange(0, total_ms + interval_ms, interval_ms)
    counts1, _ = np.histogram(df1["time_ms"], bins=bins)
    counts2, _ = np.histogram(df2["time_ms"], bins=bins)
    time_s = bins[:-1] / 1000

    segment_ms = segment_sec * 1000
    num_segments = int(np.ceil(total_ms / segment_ms))

    for i in range(num_segments):
        start_ms = i * segment_ms
        end_ms = min((i + 1) * segment_ms, total_ms)
        mask = (time_s * 1000 >= start_ms) & (time_s * 1000 < end_ms)

        plt.figure(figsize=(12, 5))
        plt.plot(time_s[mask], counts1[mask], label=label1, alpha=0.7)
        plt.plot(time_s[mask], counts2[mask], label=label2, alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Requests per 10ms")
        plt.title(f"Request Arrival Timeline ({start_ms/1000:.1f}s - {end_ms/1000:.1f}s)")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_diff_distribution(df1, df2):
    """计算相同 request id 的时间差分布（5ms 粒度，最大 100ms）"""
    merged = pd.merge(df1, df2, on="req_id", suffixes=("_1", "_2"))
    merged["diff_ms"] = merged["time_ms_2"] - merged["time_ms_1"]

    bins = np.arange(-100, 105, 5)
    counts, edges = np.histogram(merged["diff_ms"], bins=bins)
    percents = counts / counts.sum() * 100

    plt.figure(figsize=(10, 5))
    plt.bar((edges[:-1] + edges[1:]) / 2, percents, width=5)
    plt.xlabel("Arrival Time Difference (ms)")
    plt.ylabel("Percentage (%)")
    plt.title("Request Arrival Time Difference Distribution")
    plt.tight_layout()
    plt.show()

# ========== 主函数 ==========

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <log1> <log2> [scale_factor] [segment_seconds]")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    scale_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    segment_sec = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    # 判断文件类型
    ext1 = os.path.splitext(file1)[1].lower()
    ext2 = os.path.splitext(file2)[1].lower()

    print(f"Parsing {file1} and {file2} ...")

    # 先确定日志文件的请求数量
    if ext1 == ".csv" and ext2 != ".csv":
        log_times = parse_log(file2)
        df2 = to_relative_times(log_times)
        req_limit = len(df2)
        df1 = parse_trace_csv(file1, limit=req_limit, scale_factor=scale_factor)
    elif ext2 == ".csv" and ext1 != ".csv":
        log_times = parse_log(file1)
        df1 = to_relative_times(log_times)
        req_limit = len(df1)
        df2 = parse_trace_csv(file2, limit=req_limit, scale_factor=scale_factor)
    elif ext1 == ".csv" and ext2 == ".csv":
        df1 = parse_trace_csv(file1, scale_factor=scale_factor)
        df2 = parse_trace_csv(file2, scale_factor=scale_factor)
    else:
        df1 = to_relative_times(parse_log(file1))
        df2 = to_relative_times(parse_log(file2))

    print(f"Requests parsed: {len(df1)} vs {len(df2)}")

    print(f"Plotting segmented timeline (segment={segment_sec}s)...")
    plot_timeline_segmented(df1, df2, file1, file2, segment_sec=segment_sec)

    print("Plotting difference distribution...")
    plot_diff_distribution(df1, df2)

if __name__ == "__main__":
    main()
