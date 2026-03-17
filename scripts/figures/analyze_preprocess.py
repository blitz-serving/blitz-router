import re
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

### --- 工具函数 ---
def parse_ts(ts_str: str) -> float:
    """解析 ISO8601 时间字符串为 float 秒"""
    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
    return dt.timestamp()


### --- 分析1: 请求排队时间 ---
def parse_vllm_queue_gap(log_path: str):
    """
    解析 vLLM 日志，计算:
    - request_id
    - tokens
    - timestamp_gap = core.ts - api.ts
    """
    req_recv_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z: Received chat completion request: (\d+)"
    )
    core_exec_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z: request (\d+), tokens: (\d+), max_tokens: (\d+)"
    )

    recv_time = {}   # request_id -> 接收时间
    data_points = [] # (request_id, tokens, timestamp_gap)

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m1 = req_recv_pattern.search(line)
            if m1:
                ts_str, req_id = m1.groups()
                recv_time[int(req_id)] = parse_ts(ts_str)
                continue

            m2 = core_exec_pattern.search(line)
            if m2:
                ts_str, req_id, tokens, max_tokens = m2.groups()
                req_id = int(req_id)
                if req_id in recv_time:
                    start_ts = recv_time[req_id]
                    core_ts = parse_ts(ts_str)
                    gap = core_ts - start_ts
                    data_points.append((req_id, int(tokens), gap))

    return data_points


def plot_gap_vs_tokens(data_points, title="Request Queue Time vs Prompt Length"):
    """绘制 tokens vs timestamp_gap 散点图"""
    if not data_points:
        print("No queue-gap data to plot.")
        return

    _, tokens, gaps = zip(*data_points)
    plt.figure(figsize=(7, 5))
    plt.scatter(tokens, gaps, alpha=0.7, s=50, color="steelblue", edgecolors="k")
    plt.title(title)
    plt.xlabel("Prompt Tokens")
    plt.ylabel("Queue Wait Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


### --- 分析2: Batch算子耗时与model_forward差异 ---
def parse_operator_latencies(log_path: str):
    """
    从 [core.py:302] 日志中解析算子耗时与 model_forward 时间
    返回:
      events = [(sum_ops, model_forward, diff)]
    """
    pattern = re.compile(
        r"Batch op latencies \(ms\): .*?norm=(\d+\.\d+), qkv_proj=(\d+\.\d+), rotary_emb=(\d+\.\d+), "
        r"attention=(\d+\.\d+), o_proj=(\d+\.\d+), mlp_gate_up_proj=(\d+\.\d+), "
        r"mlp_activation=(\d+\.\d+), mlp_down_proj=(\d+\.\d+), model_forward=(\d+\.\d+)"
    )

    events = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                vals = list(map(float, m.groups()))
                sum_ops = sum(vals[:-1])  # 所有算子之和
                model_forward = vals[-1]
                diff = model_forward - sum_ops
                events.append((sum_ops, model_forward, diff))
    return events


def plot_operator_diff(events):
    """绘制 model_forward - sum(operators) vs model_forward"""
    if not events:
        print("No operator latency events found.")
        return

    sum_ops, model_forward, diff = zip(*events)

    # 图1：差值 vs model_forward 散点图
    plt.figure(figsize=(7, 5))
    plt.scatter(model_forward, diff, color="tomato", alpha=0.7, edgecolors="k", s=40)
    plt.title("Model Forward - Σ(Operators) vs Model Forward Time")
    plt.xlabel("Model Forward Time (ms)")
    plt.ylabel("Δ = Model Forward - Σ(Operators) (ms)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    avg_diff = sum(diff) / len(diff)
    print(f"\n平均差值: {avg_diff:.3f} ms  (model_forward - Σ算子)")


### --- 主函数 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze vLLM logs: queue wait + operator latency")
    parser.add_argument("log_path", help="Path to vLLM log file")
    args = parser.parse_args()

    print(f"Parsing log: {args.log_path}\n")

    # 1️⃣ 请求排队时间分析
    queue_data = parse_vllm_queue_gap(args.log_path)
    print(f"Parsed {len(queue_data)} matched request entries:")
    for req_id, tokens, gap in queue_data[:10]:
        print(f"  Request {req_id} | tokens={tokens} | queue_wait={gap:.3f}s")
    plot_gap_vs_tokens(queue_data)

    # 2️⃣ Batch算子耗时差异分析
    operator_events = parse_operator_latencies(args.log_path)
    print(f"\nParsed {len(operator_events)} operator latency entries.")
    plot_operator_diff(operator_events)
