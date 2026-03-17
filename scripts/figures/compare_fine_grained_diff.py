# 对比所有算子执行时长(细粒度, 按时间轴独立展示 + attention三类对比)
# usage:
#   python compare_fine_grained_diff.py \
#     --log1=/path/to/log1_processed.log \
#     --log2=/path/to/log2_processed.log \
#     --out=/path/to/compare_prefix
#
# 输出：
#   /path/to/compare_prefix_attention_decode_only.png
#   /path/to/compare_prefix_attention_prefill_only.png
#   /path/to/compare_prefix_attention_mixed.png
#   /path/to/compare_prefix_prefill_only_log1.csv
#   /path/to/compare_prefix_prefill_only_log2.csv
#   以及各同名算子的对比图：/path/to/compare_prefix_<op>.png

import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import os

# ================= 参数 =================
parser = argparse.ArgumentParser()
parser.add_argument("--log1", type=str, required=True, help="log1 文件路径（旧格式或细分格式均可）")
parser.add_argument("--log2", type=str, required=True, help="log2 文件路径（含 attn_* 字段更佳）")
parser.add_argument("--out", type=str, required=True, help="输出图像路径前缀")
args = parser.parse_args()

# ================= 正则 =================
log_pattern = re.compile(
    r'(?P<timestamp>[\d\-:T\.]+Z).*?prefill_tokens: (?P<prefill_tokens>\d+), '
    r'prefill_token_budget: (?P<prefill_token_budget>\d+), latency: (?P<latency>\d+), '
    r'outputs: \[(?P<outputs>.*?)\], log_info: "(?P<log_info>.*?)"'
)

# log1: "norm=1.313, qkv_proj=8.232, ..."
log1_kv_pattern = re.compile(r'([A-Za-z0-9_]+)=([\d\.]+)')
# log2: 兼容 "qkv_proj 5.236ms" 和 "qkv_proj: 5.236ms"
log2_kv_pattern = re.compile(r'([^,]+?)=\s*([\d\.]+)ms')

# outputs 中解析 state
state_pattern = re.compile(r'state:\s*"(\w+)"')

# ================= 工具函数 =================
def parse_log(filepath, log_type):
    events = []
    with open(filepath, "r") as f:
        for line in f:
            m = log_pattern.search(line)
            if not m:
                continue
            ts = datetime.fromisoformat(m.group("timestamp").replace("Z", "+00:00"))
            latency = float(m.group("latency"))
            prefill_tokens = int(m.group("prefill_tokens"))
            outputs = m.group("outputs")
            log_info = m.group("log_info")

            if log_type == "log1":
                kvs = {k.strip(): float(v) for k, v in log1_kv_pattern.findall(log_info)}
            else:
                kvs = {k.strip(): float(v) for k, v in log2_kv_pattern.findall(log_info)}

            states = state_pattern.findall(outputs)

            row = {
                "time": ts,
                "latency": latency,
                "prefill_tokens": prefill_tokens,
                "states": ",".join(states),
            }
            row.update(kvs)
            events.append(row)

    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    t0 = df["time"].min()
    df["aligned_time"] = (df["time"] - t0).dt.total_seconds()
    df = df.set_index("aligned_time").sort_index()
    return df


def align_ops(df1, df2):
    # log2 的派生列
    if not df2.empty:
        df2["attention"] = (
            df2.get("attn_kv_cache_save", 0)
            + df2.get("attn_decode", 0)
            + df2.get("attn_prefill", 0)
        )
        df2["norm"] = (
            df2.get("add", 0)
            + df2.get("mlp_norm", 0)
            + df2.get("attn_norm", 0)
        )

    # log1 的细分兜底
    if not df1.empty:
        if "attn_prefill" not in df1.columns and "attention" in df1.columns:
            df1["attn_prefill"] = pd.NA
        if "attn_decode" not in df1.columns and "attention" in df1.columns:
            df1["attn_decode"] = pd.NA
        if "attn_prefill" in df1.columns and "attn_decode" in df1.columns:
            df1["attn_prefill_plus_decode"] = df1[
                ["attn_prefill", "attn_decode"]
            ].sum(axis=1, min_count=1)
        elif "attention" in df1.columns:
            df1["attn_prefill_plus_decode"] = df1["attention"]
        else:
            df1["attn_prefill_plus_decode"] = pd.NA

    return df1, df2


def _row_has(states_str, token):
    return token in states_str.split(",") if isinstance(states_str, str) and states_str else False


def classify_conditions(df):
    """分类：
       - decode_only: prefill_tokens == 0 且 有 DECODE 且 无 PREFILL
       - prefill_only: prefill_tokens > 0 且 不含 DECODE
       - mixed: 其他情况
    """
    if df.empty:
        df["condition"] = []
        return df
    decode_only_mask = (df["prefill_tokens"] == 0) & df["states"].apply(
        lambda s: _row_has(s, "DECODE") and not _row_has(s, "PREFILL")
    )
    prefill_only_mask = (df["prefill_tokens"] > 0) & df["states"].apply(
        lambda s: not _row_has(s, "DECODE")
    )
    cond = pd.Series(["mixed"] * len(df), index=df.index, dtype="object")
    cond.loc[decode_only_mask] = "decode_only"
    cond.loc[prefill_only_mask] = "prefill_only"
    df = df.copy()
    df["condition"] = cond
    return df


def _plot_lines(xy_list, title, outpath):
    plt.figure(figsize=(10, 6))
    have_any = False
    for x, y, label, color in xy_list:
        if y is not None and len(y) > 0:
            plt.plot(x, y, label=label, color=color)
            have_any = True
    if not have_any:
        plt.text(
            0.5,
            0.5,
            "No matching samples",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    plt.xlabel("Aligned Time (s)")
    plt.ylabel("Execution Time (ms)")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath)
    plt.close()
    print(f"Saved figure: {outpath}")


def plot_attention_three_cases(df1, df2, out_prefix):
    df1 = classify_conditions(df1)
    df2 = classify_conditions(df2)

    def pick_series(df, prefer_cols, fallback_cols=None):
        for c in prefer_cols:
            if c in df.columns:
                s = df[c].dropna()
                if len(s) > 0:
                    return s
        if fallback_cols:
            for c in fallback_cols:
                if c in df.columns:
                    s = df[c].dropna()
                    if len(s) > 0:
                        return s
        return None

    # decode_only
    df1_dec = df1[df1["condition"] == "decode_only"]
    df2_dec = df2[df2["condition"] == "decode_only"]
    log1_s_decode = pick_series(df1_dec, ["attn_decode"], ["attention"])
    xy_list = []
    if log1_s_decode is not None:
        xy_list.append(
            (log1_s_decode.index, log1_s_decode.values, "log1 attn_decode (or attention)", "blue")
        )
    if "attention" in df2_dec.columns and len(df2_dec) > 0:
        xy_list.append(
            (df2_dec.index, df2_dec["attention"].values, "log2 attention (decode_only)", "orange")
        )
    _plot_lines(xy_list, "Attention Comparison (decode_only)", f"{out_prefix}_attention_decode_only.png")

    # prefill_only
    df1_pre = df1[df1["condition"] == "prefill_only"]
    df2_pre = df2[df2["condition"] == "prefill_only"]
    log1_s_prefill = pick_series(df1_pre, ["attn_prefill"], ["attention"])
    xy_list = []
    if log1_s_prefill is not None:
        xy_list.append(
            (log1_s_prefill.index, log1_s_prefill.values, "log1 attn_prefill (or attention)", "blue")
        )
    if "attention" in df2_pre.columns and len(df2_pre) > 0:
        xy_list.append(
            (df2_pre.index, df2_pre["attention"].values, "log2 attention (prefill_only)", "orange")
        )
    _plot_lines(xy_list, "Attention Comparison (prefill_only)", f"{out_prefix}_attention_prefill_only.png")

    # mixed
    df1_mix = df1[df1["condition"] == "mixed"]
    df2_mix = df2[df2["condition"] == "mixed"]
    log1_s_mix = pick_series(df1_mix, ["attn_prefill_plus_decode"], ["attention"])
    xy_list = []
    if log1_s_mix is not None:
        xy_list.append(
            (log1_s_mix.index, log1_s_mix.values, "log1 attn_prefill+attn_decode (or attention)", "blue")
        )
    if "attention" in df2_mix.columns and len(df2_mix) > 0:
        xy_list.append(
            (df2_mix.index, df2_mix["attention"].values, "log2 attention (mixed)", "orange")
        )
    _plot_lines(xy_list, "Attention Comparison (mixed)", f"{out_prefix}_attention_mixed.png")


def plot_same_name_ops(df1, df2, out_prefix):
    common_ops = (
        set(df1.columns) & set(df2.columns)
    ) - {"time", "latency", "prefill_tokens", "states", "condition"}
    if not common_ops:
        print("No common ops to compare (besides attention).")
        return
    for op in sorted(common_ops):
        s1 = df1[op].dropna() if op in df1.columns else None
        s2 = df2[op].dropna() if op in df2.columns else None
        xy_list = []
        if s1 is not None and len(s1) > 0:
            xy_list.append((s1.index, s1.values, f"log1 {op}", "blue"))
        if s2 is not None and len(s2) > 0:
            xy_list.append((s2.index, s2.values, f"log2 {op}", "orange"))
        _plot_lines(xy_list, f"Operator Comparison: {op}", f"{out_prefix}_{op}.png")


def dump_prefill_only_csv(df1, df2, out_prefix):
    """将两个日志中 'prefill_only' 的样本分别写入 CSV，便于做特征观察。"""
    df1c = classify_conditions(df1)
    df2c = classify_conditions(df2)

    df1_pre = df1c[df1c["condition"] == "prefill_only"].copy()
    df2_pre = df2c[df2c["condition"] == "prefill_only"].copy()

    # 确保需要的派生列存在
    if "attn_prefill_plus_decode" not in df1_pre.columns and "attention" in df1_pre.columns:
        df1_pre["attn_prefill_plus_decode"] = df1_pre["attention"]

    # index 是 aligned_time，把它写回列
    if not df1_pre.empty:
        df1_pre_out = df1_pre.reset_index().rename(columns={"aligned_time": "aligned_time"})
        p1 = f"{out_prefix}_prefill_only_log1.csv"
        os.makedirs(os.path.dirname(p1) or ".", exist_ok=True)
        df1_pre_out.to_csv(p1, index=False)
        print(f"Saved prefill_only dataset (log1): {p1}  rows={len(df1_pre_out)}")
    else:
        print("log1 has no prefill_only samples.")

    if not df2_pre.empty:
        df2_pre_out = df2_pre.reset_index().rename(columns={"aligned_time": "aligned_time"})
        p2 = f"{out_prefix}_prefill_only_log2.csv"
        os.makedirs(os.path.dirname(p2) or ".", exist_ok=True)
        df2_pre_out.to_csv(p2, index=False)
        print(f"Saved prefill_only dataset (log2): {p2}  rows={len(df2_pre_out)}")
    else:
        print("log2 has no prefill_only samples.")


# ================= 主流程 =================
df1 = parse_log(args.log1, "log1")
df2 = parse_log(args.log2, "log2")

df1, df2 = align_ops(df1, df2)

# 先导出 prefill_only 的原始样本，便于你做数据诊断
dump_prefill_only_csv(df1, df2, args.out)

# 再画 attention 三类对比
plot_attention_three_cases(df1, df2, args.out)

# 以及同名算子对比
plot_same_name_ops(df1, df2, args.out)
