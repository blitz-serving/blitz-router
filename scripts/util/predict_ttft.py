#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析 vLLM TTFT 预测精度：
- 读取 trace1（router 日志文本）与 trace2（每行一条 JSON）
- 以 request id 对齐，计算 |pred - actual| / actual
- 输出总体统计、按 vllm 分组统计，并导出 CSV 明细

用法示例：
    python analyze_ttft.py --trace1 path/to/trace1.log --trace2 path/to/trace2.log \
        --out ttft_compare.csv
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict, Counter
from statistics import mean, median

TRACE1_RE = re.compile(
    r"""
    Vllm\#(?P<vllm>\d+)          # Vllm#<id>
    ::Request\#(?P<req>\d+)      # ::Request#<id>
    \s+predict\ TTFT:\s+(?P<pred>\d+)   # predict TTFT: <num>
    """,
    re.VERBOSE,
)

def parse_trace1(path):
    """
    解析 trace1 文本日志：
    返回：
        pred_by_req: {req_id(int): pred_ttft(int)}
        vllm_by_req: {req_id(int): vllm_id(int)}
        dup_counts:  统计相同 req_id 出现多次的计数（通常取最后一次）
    说明：
        - 若同一个 request# 多次出现，默认采用“最后一次”的预测值（覆盖之前的）。
        - 会忽略没有匹配到模式的行（例如 SLO violation 的 WARN 行本身没预测值）。
    """
    pred_by_req = {}
    vllm_by_req = {}
    dup_counts = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = TRACE1_RE.search(line)
            if not m:
                continue
            req_id = int(m.group("req"))
            vllm_id = int(m.group("vllm"))
            pred = int(m.group("pred"))
            if req_id in pred_by_req:
                dup_counts[req_id] += 1
            pred_by_req[req_id] = pred
            vllm_by_req[req_id] = vllm_id
    return pred_by_req, vllm_by_req, dup_counts

def parse_trace2(path):
    """
    解析 trace2（每行一个 JSON 对象）。
    取字段：
        - request_id: 字符串或数字 -> 转为 int
        - first_token_time: 实际 TTFT（毫秒）
    若同一 request_id 出现多次，默认采用“最后一次”（覆盖）。
    返回：
        actual_by_req: {req_id(int): actual_ttft(int)}
    """
    actual_by_req = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                # 若某些行不是纯 JSON，尝试截取到第一个 '}' 再解析（可选增强）
                try:
                    cut = s[: s.index("}") + 1]
                    obj = json.loads(cut)
                except Exception:
                    continue
            if "request_id" not in obj or "first_token_time" not in obj:
                continue
            try:
                req_id = int(obj["request_id"])
                actual = int(obj["first_token_time"])
            except (ValueError, TypeError):
                continue
            actual_by_req[req_id] = actual
    return actual_by_req

def pct(x):
    return f"{100.0 * x:.2f}%"

def percentile(values, p):
    """
    简单百分位数（p in [0,100]），使用“最近秩”法的线性插值。
    """
    if not values:
        return float("nan")
    if p <= 0:
        return sorted(values)[0]
    if p >= 100:
        return sorted(values)[-1]
    arr = sorted(values)
    k = (len(arr) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    d0 = arr[f] * (c - k)
    d1 = arr[c] * (k - f)
    return d0 + d1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace1", required=True, help="trace1 文本日志路径（包含 'predict TTFT' 行）")
    ap.add_argument("--trace2", required=True, help="trace2 JSON 行日志路径（包含 first_token_time）")
    ap.add_argument("--out", default="ttft_compare.csv", help="导出明细 CSV 文件名")
    ap.add_argument("--ignore_zero_actual", action="store_true",
                    help="若实际 TTFT 为 0 则跳过（避免除零）")
    args = ap.parse_args()

    pred_by_req, vllm_by_req, dup_counts = parse_trace1(args.trace1)
    actual_by_req = parse_trace2(args.trace2)

    # 对齐 request id
    common_ids = sorted(set(pred_by_req) & set(actual_by_req))
    missing_in_t2 = sorted(set(pred_by_req) - set(actual_by_req))
    missing_in_t1 = sorted(set(actual_by_req) - set(pred_by_req))

    # 计算误差
    rows = []
    ratios = []
    bias_list = []  # pred - actual
    per_vllm_ratios = defaultdict(list)

    skipped_zero = 0
    for rid in common_ids:
        pred = pred_by_req[rid]
        actual = actual_by_req[rid]
        if args.ignore_zero_actual and actual == 0:
            skipped_zero += 1
            continue
        if actual == 0:
            # 避免除零，记录为 NaN
            ratio = float("nan")
        else:
            ratio = abs(pred - actual) / actual

        vllm_id = vllm_by_req.get(rid, None)
        rows.append({
            "request_id": rid,
            "vllm_id": vllm_id,
            "pred_ttft_ms": pred,
            "actual_ttft_ms": actual,
            "abs_error_ms": abs(pred - actual),
            "rel_error": ratio,
        })
        if not math.isnan(ratio):
            ratios.append(ratio)
            if vllm_id is not None:
                per_vllm_ratios[vllm_id].append(ratio)
        bias_list.append(pred - actual)

    # 导出 CSV 明细
    fieldnames = ["request_id", "vllm_id", "pred_ttft_ms", "actual_ttft_ms", "abs_error_ms", "rel_error"]
    with open(args.out, "w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # 打印总体统计
    print("=" * 80)
    print(f"总请求（trace1）：{len(pred_by_req)}")
    print(f"总请求（trace2）：{len(actual_by_req)}")
    print(f"成功对齐的请求数：{len(rows)}")
    if args.ignore_zero_actual:
        print(f"（忽略 actual==0 的条目：{skipped_zero}）")
    print(f"trace1 中重复 request 的数量：{sum(dup_counts.values())}（按最后一次覆盖）")
    print(f"仅在 trace1 中出现的 request 数：{len(missing_in_t2)}")
    print(f"仅在 trace2 中出现的 request 数：{len(missing_in_t1)}")

    if ratios:
        print("-" * 80)
        print("相对误差 |pred-actual|/actual 统计：")
        print(f"  均值:  {pct(mean(ratios))}")
        print(f"  中位:  {pct(median(ratios))}")
        print(f"  p90 :  {pct(percentile(ratios, 90))}")
        print(f"  p95 :  {pct(percentile(ratios, 95))}")
        print(f"  最大:  {pct(max(ratios))}")
    else:
        print("没有可计算的相对误差（可能全部 actual==0 或未对齐）。")

    if bias_list:
        # 预测偏差（正数表示整体高估，负数表示整体低估）
        print("-" * 80)
        print("预测偏差（pred - actual，单位：ms）：")
        print(f"  均值:  {mean(bias_list):.2f} ms")
        print(f"  中位:  {median(bias_list):.2f} ms")
        print(f"  p90 :  {percentile(bias_list, 90):.2f} ms")
        print(f"  p95 :  {percentile(bias_list, 95):.2f} ms")
        print(f"  最小:  {min(bias_list):.2f} ms")
        print(f"  最大:  {max(bias_list):.2f} ms")

    # 按 vllm 维度统计
    if per_vllm_ratios:
        print("-" * 80)
        print("按 vllm_id 的相对误差统计：")
        for vllm_id in sorted(per_vllm_ratios):
            vals = per_vllm_ratios[vllm_id]
            print(
                f"  vllm#{vllm_id}: n={len(vals)}, "
                f"mean={pct(mean(vals))}, median={pct(median(vals))}, "
                f"p90={pct(percentile(vals, 90))}"
            )

    print("=" * 80)
    print(f"明细已导出：{args.out}")

if __name__ == "__main__":
    main()
