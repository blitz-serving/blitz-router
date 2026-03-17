import json
import sys
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 加载数据 & 预处理
# =========================


def load_requests(path):
    reqs = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)

            s = float(obj["s_time"])
            e = float(obj["e_time"])
            out_len = float(obj["output_length"])
            try:
                status = int(obj.get("status", 0))
            except ValueError:
                status = -1

            if status == 200:
                ft = float(obj["first_token_time"])
                avg_bt = float(obj["avg_time_between_tokens"])
            else:
                ft = None
                avg_bt = None

            # tuple fields:
            # (s_time, e_time, output_length, first_token_time, avg_bt, status)
            reqs.append((s, e, out_len, ft, avg_bt, status))

    return reqs


# =========================
# L(t) 时间序列
# =========================


def build_L_timeseries(requests, step=50.0):
    """
    L(t): 系统中请求数（包括成功 + 失败），因为都占用系统资源
    返回:
      times: [t0, t1, ...]
      Ls:    [L(t0), L(t1), ...]
    """
    events = []
    for s, e, _, _, _, _ in filter(lambda x: x[-1] == 200, requests):
        events.append((s, +1))
        events.append((e, -1))
    events.sort()

    L = 0
    start = min(s for s, _, _, _, _, _ in requests)
    end = max(e for _, e, _, _, _, _ in requests)
    sample_times = np.arange(start, end, step)

    Ls = []
    ev_i = 0
    n = len(events)

    for t in sample_times:
        while ev_i < n and events[ev_i][0] <= t:
            L += events[ev_i][1]
            ev_i += 1
        Ls.append(L)

    return sample_times, np.array(Ls)


# =========================
# λ_real(t) — 真实离开率，只看成功请求
# =========================


def build_departure_rate_timeseries(requests, times, window=500.0):
    """
    λ_real(t): 真实离开率, 滑动窗口估计。
    只统计 status == 200 的请求:
      departures = e_time of successful requests

    对每个 t:
      λ_real(t) = (# departures with t-window < e <= t) / window
    """
    departures = [e for _, e, _, _, _, status in requests if status == 200]
    departures.sort()

    lam = []
    n = len(departures)
    left_idx = 0
    right_idx = 0

    for t in times:
        left = t - window
        # 移动 left_idx，使得 departures[left_idx] > left
        while left_idx < n and departures[left_idx] <= left:
            left_idx += 1
        # 移动 right_idx，使得 departures[right_idx] <= t
        while right_idx < n and departures[right_idx] <= t:
            right_idx += 1

        count = max(0, right_idx - left_idx)
        lam.append(count / window if window > 0 else 0.0)

    return np.array(lam)


# =========================
# SLO 违例比例（滑动窗口）
# =========================


def build_slo_violation_timeseries(requests, times, window=5000.0):
    reqs_sorted = sorted(requests, key=lambda x: x[0])
    slo_ratio = []
    n = len(reqs_sorted)

    start_idx = 0
    end_idx = 0

    for t in times:
        left = t - window

        # slide window
        while start_idx < n and reqs_sorted[start_idx][0] < left:
            start_idx += 1
        while end_idx < n and reqs_sorted[end_idx][0] <= t:
            end_idx += 1

        window_reqs = reqs_sorted[start_idx:end_idx]

        if not window_reqs:
            slo_ratio.append(0.0)
            continue

        total = len(window_reqs)
        bad = 0

        for _, _, _, ft, avg_bt, status in window_reqs:

            if status != 200:
                # ==== NEW RULE: 失败请求永远 SLO 违例 ====
                bad += 1
            else:
                # 成功请求使用正常 SLO 判断
                if ft > 5000 or avg_bt > 50:
                    bad += 1

        slo_ratio.append(bad / total)

    return np.array(slo_ratio)


# =========================
# Little's Law: L(t) vs λ_real(t)*W_est
# =========================


def plot_L_vs_lambdaW(times, Ls, lam_real, W_est, save_path):
    """
    画图:
      - L(t)
      - λ_real(t)*W_est
      - 背景填充 λ_real(t)*W_est > L(t) 的区域
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    LW = lam_real * W_est

    # L(t)
    ax.plot(times, Ls, label="L(t) — #requests in system", linewidth=2)

    # λ_real(t)*W_est
    ax.plot(times, LW, label="λ_real(t) × W_est", linewidth=2, linestyle="--")

    # 背景高亮：λW > L 的区域
    above = LW < Ls
    start_idx = None
    for i in range(len(times)):
        if above[i] and start_idx is None:
            start_idx = i
        elif not above[i] and start_idx is not None:
            ax.axvspan(
                times[start_idx], times[i], facecolor="red", alpha=0.15, edgecolor="none"
            )
            start_idx = None

    if start_idx is not None:
        ax.axvspan(
            times[start_idx], times[-1], facecolor="red", alpha=0.15, edgecolor="none"
        )

    ax.set_xlabel("Time (ms, normalized)")
    ax.set_ylabel("Value")
    ax.grid(True)
    plt.title("L(t) vs λ_real(t) × W_est  (shaded where λW_est > L)")

    ax.legend()
    plt.savefig(f"{save_path}.png", dpi=300)


# =========================
# SLO 违例比例图
# =========================


def plot_slo(times, slo_ratio, save_path="slo_violation_ratio.png"):
    plt.figure(figsize=(12, 4))
    plt.plot(times, slo_ratio, linewidth=2, label="SLO violation ratio")

    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel("Time (ms, normalized)")
    plt.ylabel("Violation ratio")
    plt.title("SLO violation ratio over time (sliding window)")

    plt.legend()
    plt.savefig(f"{save_path}_slo.png", dpi=300)


# =========================
# 输出长度统计（含失败请求）
# =========================


def compute_output_length_stats(requests):
    """
    输出:
      - 所有请求的平均 output_length
      - 成功请求 / 失败请求 的平均 output_length
      - 失败率
    """
    out_all = [out for _, _, out, _, _, _ in requests]
    out_ok = [out for _, _, out, _, _, status in requests if status == 200]
    out_fail = [out for _, _, out, _, _, status in requests if status != 200]

    mean_all = np.mean(out_all) if out_all else 0.0
    mean_ok = np.mean(out_ok) if out_ok else 0.0
    mean_fail = np.mean(out_fail) if out_fail else 0.0

    total = len(requests)
    fail_cnt = len(out_fail)
    fail_rate = fail_cnt / total if total > 0 else 0.0

    return {
        "mean_all": mean_all,
        "mean_ok": mean_ok,
        "mean_fail": mean_fail,
        "fail_rate": fail_rate,
        "total": total,
        "fail_cnt": fail_cnt,
    }


# =========================
# main
# =========================


def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_departure_rates.py data.jsonl")
        return

    path = sys.argv[1]
    save_fig = sys.argv[2]
    requests = load_requests(path)

    # 1) 输出长度统计（包含失败请求）
    stats = compute_output_length_stats(requests)
    print("=== Output length stats ===")
    print(f"Total requests        : {stats['total']}")
    print(f"Failed requests       : {stats['fail_cnt']}")
    print(f"Failure rate          : {stats['fail_rate']*100:.2f}%")
    print(f"Mean output length(all): {stats['mean_all']:.2f}")
    print(f"Mean output length(ok) : {stats['mean_ok']:.2f}")
    print(f"Mean output length(fail): {stats['mean_fail']:.2f}")
    print()

    # 2) L(t)
    times, Ls = build_L_timeseries(requests, step=50.0)

    # 3) 用所有请求（含失败）估计 W_est
    avg_time_between_tokens = 50.0
    first_token_time_const = 5000.0
    mean_out_all = stats["mean_all"]
    W_est = first_token_time_const + avg_time_between_tokens * mean_out_all
    print(f"Estimated W_est (ms): {W_est:.2f}")
    print()

    # 4) λ_real(t) 只用成功请求
    lam_real = build_departure_rate_timeseries(requests, times, window=25 * 1e3)

    # 5) 画 Little's Law 对比图
    plot_L_vs_lambdaW(times, Ls, lam_real, W_est, save_path=save_fig)

    # 6) SLO 违例时间序列 & 图
    slo_ratio = build_slo_violation_timeseries(requests, times, window=25 * 1e3)
    plot_slo(times, slo_ratio, save_path=save_fig)


if __name__ == "__main__":
    main()
