import json
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_requests(path):
    reqs = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            s = float(obj["s_time"])
            e = float(obj["e_time"])
            out_len = float(obj["output_length"])

            # ==== NEW ====
            ttft = float(obj["first_token_time"])
            tpot = float(obj["avg_time_between_tokens"])

            reqs.append((s, e, out_len, ttft, tpot))
    return reqs


def build_L_timeseries(requests, step=50.0):
    events = []
    for s, e, _, _, _ in requests:
        events.append((s, +1))
        events.append((e, -1))
    events.sort()

    L = 0
    start = min(s for s, _, _, _, _ in requests)
    end = max(e for _, e, _, _, _ in requests)
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


def build_departure_rate_timeseries(requests, times, window=200.0):
    """
    真实离开率 λ_real(t) = (#departures in [t-window, t]) / window
    """
    departures = [e for _, e, _, _, _ in requests]
    departures.sort()

    lam = []
    d_i = 0
    n = len(departures)

    for t in times:
        # count departures in window
        cnt = 0
        # move pointer
        while d_i < n and departures[d_i] < t - window:
            d_i += 1
        j = d_i
        while j < n and departures[j] <= t:
            cnt += 1
            j += 1

        lam.append(cnt / window)

    return np.array(lam)


# ==== NEW ====
def build_slo_violation_timeseries(requests, times, window=1000.0):
    """
    返回: SLO_violation_ratio(t)
    计算窗口内:
        first_token_time > 5 或 avg_time_between_tokens > 50 的请求占比
    """
    # 将请求按开始时间排序
    reqs_sorted = sorted(requests, key=lambda x: x[0])  # sort by s_time

    slo_ratio = []
    n = len(reqs_sorted)
    i = 0

    for t in times:
        # 窗口范围：仅统计 [t-window, t]
        start_t = t - window
        count_total = 0
        count_bad = 0

        # 移动 i 至窗口开始
        while i < n and reqs_sorted[i][0] < start_t:
            i += 1

        j = i
        while j < n and reqs_sorted[j][0] <= t:
            _, _, _, ft, avgbt = reqs_sorted[j]
            count_total += 1
            if ft > 5000 or avgbt > 50:
                count_bad += 1
            j += 1

        if count_total == 0:
            slo_ratio.append(0.0)
        else:
            slo_ratio.append(count_bad / count_total)

    return np.array(slo_ratio)


# def plot(times, Ls, lam_real, lam_est, save_fig):
#     fig, ax1 = plt.subplots(figsize=(12, 6))

#     # Plot L(t)
#     ax1.plot(times, Ls, label="L(t) — requests in system", linewidth=2)
#     ax1.set_xlabel("Time (ms)")
#     ax1.set_ylabel("L(t)")
#     ax1.grid(True)

#     # Second y-axis for λ
#     ax2 = ax1.twinx()
#     ax2.plot(times, lam_real, color="orange", label="λ_real(t) — actual departures", linewidth=2)

#     # Make λ_est visible by auto-scaling
#     scale = np.max(lam_real) / max(np.max(lam_est), 1e-12)
#     lam_est_scaled = lam_est * scale
#     ax2.plot(times, lam_est_scaled, color="green",
#              label=f"λ_est(t) × {scale:.1f}  (scaled)",
#              linewidth=2, linestyle="--")

#     ax2.set_ylabel("Departure rate λ(t)")

#     plt.title("Real departure rate vs estimated departure rate")

#     # Combined legend
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

#     plt.savefig(f"{save_fig}.png")


def plot(times, Ls, lam_real, W_est, save_fig):
    fig, ax = plt.subplots(figsize=(12, 6))

    # λ_real(t) * W_est
    LW = lam_real * W_est

    # Plot L(t)
    ax.plot(times, Ls, label="L(t) — real #requests", linewidth=2)

    # Plot λ_real * W_est
    ax.plot(times, LW, label=f"λ_real(t) × W_est", linewidth=2, linestyle="--")

    # Shading where λ_real * W_est > L(t)
    above = LW < Ls

    # Identify contiguous segments
    start_idx = None
    for i in range(len(times)):
        if above[i] and start_idx is None:
            start_idx = i
        elif not above[i] and start_idx is not None:
            ax.axvspan(
                times[start_idx],
                times[i],
                facecolor="red",
                alpha=0.15,
                edgecolor="none",
            )
            start_idx = None

    # Tail segment
    if start_idx is not None:
        ax.axvspan(
            times[start_idx], times[-1], facecolor="red", alpha=0.15, edgecolor="none"
        )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Value")
    ax.grid(True)
    plt.title("L(t) vs λ_real(t) × W_est   — shaded where λW_est > L(t)")

    ax.legend()

    # Save figure
    plt.savefig(f"{save_fig}.png", dpi=300)


# ==== NEW ====
def plot_slo(times, slo_ratio, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(times, slo_ratio, linewidth=2, label="SLO violation ratio")

    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel("Time (ms)")
    plt.ylabel("Violation %")
    plt.title("SLO violation ratio over time")

    plt.legend()
    plt.savefig(f"{save_path}_slo.png", dpi=300)


def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_departure_rates.py data.jsonl")
        return

    path = sys.argv[1]
    save_fig = sys.argv[2]
    reqs = load_requests(path)

    # L(t)
    times, Ls = build_L_timeseries(reqs, step=50.0)

    # W_est = constant
    _, _, outlens, _, _ = zip(*reqs)
    avg_time_between_tokens = 50.0
    first_token_time = 5000.0
    avg_out_len = np.mean(outlens)
    print(f"Average output length = {avg_out_len}")  # 424
    W_est = first_token_time + avg_time_between_tokens * avg_out_len

    # λ_real(t)
    lam_real = build_departure_rate_timeseries(reqs, times, window=25 * 1e3)

    # Draw both
    plot(times, Ls, lam_real, W_est, save_fig)

    # Compute SLO violation ratio
    slo_ratio = build_slo_violation_timeseries(reqs, times, window=25 * 1e3)

    # Plot SLO graph
    plot_slo(times, slo_ratio, save_path=save_fig)


if __name__ == "__main__":
    main()
