# -*- coding: utf-8 -*-
"""Monte Carlo CPA simulations for the manuscript.

The simulation uses the same Hamming-weight Gaussian leakage model as the
theory section.  It tracks empirical success probability and mean key rank as
the number of traces increases.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


S_PRESENT = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
             0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
S_GIFT = [0x1, 0xA, 0x4, 0xC, 0x6, 0xF, 0x3, 0x9,
          0x2, 0xD, 0xB, 0x7, 0x5, 0x0, 0x8, 0xE]
S_SKINNY = [0xC, 0x6, 0x9, 0x0, 0x1, 0xA, 0x2, 0xB,
            0x3, 0x8, 0x5, 0xD, 0x4, 0xE, 0x7, 0xF]

SBOXES = [("PRESENT", S_PRESENT), ("GIFT", S_GIFT), ("SKINNY", S_SKINNY)]
TRACE_POINTS = np.array([1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128])
N_TRIALS = 3000
SIGMA = 2.0
TRUE_KEY = 0
SEED = 20260622


def hw(x):
    return int(x).bit_count()


def min_pairwise_gap(sbox):
    gaps = []
    for delta in range(1, 16):
        gap = np.mean([(hw(sbox[a]) - hw(sbox[a ^ delta])) ** 2
                       for a in range(16)])
        gaps.append(gap)
    return float(min(gaps))


def simulate_sbox(name, sbox, rng):
    sbox = np.asarray(sbox, dtype=np.int64)
    hw_by_input = np.asarray([hw(sbox[x]) for x in range(16)], dtype=np.float64)
    max_traces = int(TRACE_POINTS[-1])

    plains = rng.integers(0, 16, size=(N_TRIALS, max_traces), endpoint=False)
    noise = rng.normal(0.0, SIGMA, size=(N_TRIALS, max_traces))
    true_mean = hw_by_input[plains ^ TRUE_KEY]
    leakage = true_mean + noise

    loglik = np.zeros((N_TRIALS, 16), dtype=np.float64)
    rows = []

    for t in range(max_traces):
        a_t = plains[:, t]
        w_t = leakage[:, t]
        predicted = np.empty((N_TRIALS, 16), dtype=np.float64)
        for key in range(16):
            predicted[:, key] = hw_by_input[a_t ^ key]
        loglik -= ((w_t[:, None] - predicted) ** 2) / (2.0 * SIGMA ** 2)

        trace_count = t + 1
        if trace_count in TRACE_POINTS:
            true_score = loglik[:, TRUE_KEY]
            ranks = 1 + np.sum(loglik > true_score[:, None], axis=1)
            success = np.mean(ranks == 1)
            rows.append({
                "sbox": name,
                "traces": trace_count,
                "success_probability": float(success),
                "mean_rank": float(np.mean(ranks)),
                "median_rank": float(np.median(ranks)),
                "min_pairwise_gap": min_pairwise_gap(sbox),
                "sigma": SIGMA,
                "trials": N_TRIALS,
                "seed": SEED,
            })
    return rows


def write_csv(rows, path):
    fields = [
        "sbox", "traces", "success_probability", "mean_rank", "median_rank",
        "min_pairwise_gap", "sigma", "trials", "seed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    styles = {
        "PRESENT": ("o-", "#1f77b4"),
        "GIFT": ("s-", "#d62728"),
        "SKINNY": ("^-", "#2ca02c"),
    }
    for name, _ in SBOXES:
        xs = np.array([r["traces"] for r in rows if r["sbox"] == name])
        success = np.array([r["success_probability"] for r in rows if r["sbox"] == name])
        ranks = np.array([r["mean_rank"] for r in rows if r["sbox"] == name])
        style, color = styles[name]
        axes[0].plot(xs, success, style, color=color, lw=1.8, ms=5, label=name)
        axes[1].plot(xs, ranks, style, color=color, lw=1.8, ms=5, label=name)

    axes[0].set_xlabel("Number of traces")
    axes[0].set_ylabel("Empirical success probability")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, ls="--", alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Number of traces")
    axes[1].set_ylabel("Mean true-key rank")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.9, 16.5)
    axes[1].grid(True, which="both", ls="--", alpha=0.3)

    fig.suptitle(
        f"Simulated CPA under Hamming-weight Gaussian leakage "
        f"(sigma={SIGMA:g}, {N_TRIALS} trials, seed={SEED})",
        fontsize=10,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    figures = os.path.join(root, "figures")
    os.makedirs(figures, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = []
    for name, sbox in SBOXES:
        rows.extend(simulate_sbox(name, sbox, rng))

    csv_path = os.path.join(root, "cpa_simulation_results.csv")
    fig_path = os.path.join(figures, "fig6_cpa_simulation.pdf")
    write_csv(rows, csv_path)
    plot_results(rows, fig_path)

    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
