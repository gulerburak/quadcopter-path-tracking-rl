#!/usr/bin/env python3
"""Rebuild tracking tables and a multi-seed figure from `shape_eval.csv`.

Prints the tracking and ablation tables on stdout and writes
`shape_deviation_bars.png`.

Seed is the unit of replication. PID is deterministic, so no std is shown.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results_matrix", "shape_eval.csv")
FIGDIR = os.environ.get("FIGDIR", os.path.join(ROOT, "report", "figures"))

SHAPES = ["hsquare", "triangle", "circle", "vsquare", "cube"]
SHAPE_LABEL = {
    "hsquare": "Horizontal square", "triangle": "Triangle", "circle": "Circle",
    "vsquare": "Vertical square", "cube": "Cube edges",
}
METRIC = "dev_rms_m"

COLORS = {"baseline": "#2a78d6", "randomized": "#eb6834",
          "capstone": "#1baf7a", "pid": "#eda100"}


def load():
    """rows[config][shape] -> list of per-seed metric values (cm)."""
    rows = defaultdict(lambda: defaultdict(list))
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows[r["config"]][r["shape"]].append(float(r[METRIC]) * 100.0)
    return rows


def cell_stats(vals):
    a = np.asarray(vals, float)
    return a.mean(), (a.std(ddof=0) if a.size > 1 else 0.0), a.size


def per_seed_shape_means(rows, cfg):
    """Each seed's mean over the five shapes -> array of length n_seeds."""
    n = len(next(iter(rows[cfg].values())))
    per_seed = np.zeros((n, len(SHAPES)))
    for j, sh in enumerate(SHAPES):
        per_seed[:, j] = rows[cfg][sh]
    return per_seed.mean(axis=1)


def fmt(mean, std, n):
    return f"{mean:.1f}" if n <= 1 else f"{mean:.1f}\\,$\\pm$\\,{std:.1f}"


def table_track(rows):
    cols = [("RL baseline", "baseline"), ("RL +init-rand", "randomized"),
            ("RL capstone", "capstone"), ("PID", "pid")]
    lines = [
        r"\begin{tabular}{lcccc}", r"\toprule",
        "Shape & " + " & ".join(c[0] for c in cols) + r" \\", r"\midrule",
    ]
    for sh in SHAPES:
        cells = [fmt(*cell_stats(rows[c[1]][sh])) for c in cols]
        lines.append(f"{SHAPE_LABEL[sh]:<18}& " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    avg = []
    for _, cfg in cols:
        m = per_seed_shape_means(rows, cfg)
        avg.append(r"\textbf{" + fmt(m.mean(), m.std(ddof=0), m.size) + "}")
    lines += [r"\textbf{Average}   & " + " & ".join(avg) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_ablation(rows):
    arc = [("Baseline", "baseline", "--", "--"),
           ("+ terminated fix", "control", "--", r"\checkmark"),
           ("+ initial-state random.", "randomized", r"\checkmark", r"\checkmark")]
    base = per_seed_shape_means(rows, "baseline").mean()
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Configuration & init-rand & crash-terminated & Avg.\ RMS \\",
             r"\midrule"]
    for name, cfg, ir, ct in arc:
        m = per_seed_shape_means(rows, cfg)
        cell = fmt(m.mean(), m.std(ddof=0), m.size)
        delta = "" if cfg == "baseline" else f" \\quad(${(m.mean()/base-1)*100:+.0f}\\%$)"
        lines.append(f"{name:<24}& {ir} & {ct} & {cell}{delta} " + r"\\")
    pid = per_seed_shape_means(rows, "pid")
    lines += [r"\midrule",
              r"PID baseline             & \multicolumn{2}{c}{---} & "
              + f"{pid.mean():.1f} " + r"\\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def figure(rows):
    cfgs = ["baseline", "randomized", "capstone", "pid"]
    labels = {"baseline": "RL baseline", "randomized": "RL +init-rand",
              "capstone": "RL capstone", "pid": "PID"}
    x = np.arange(len(SHAPES))
    w = 0.20
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, cfg in enumerate(cfgs):
        means = [cell_stats(rows[cfg][sh])[0] for sh in SHAPES]
        stds = [cell_stats(rows[cfg][sh])[1] for sh in SHAPES]
        off = (i - (len(cfgs) - 1) / 2) * w
        ax.bar(x + off, means, w, yerr=stds, capsize=2.5, label=labels[cfg],
               color=COLORS[cfg], edgecolor="white", linewidth=0.6,
               error_kw=dict(lw=0.8, ecolor="0.35"))
    ax.set_xticks(x)
    ax.set_xticklabels([SHAPE_LABEL[s] for s in SHAPES], fontsize=8)
    ax.set_ylabel("RMS cross-track deviation [cm]", fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color="0.85", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, ncol=4, loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, 1.13))
    ax.text(0.995, 0.97, "error bars: std over 3 seeds (PID deterministic)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="0.4")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "shape_deviation_bars.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    rows = load()
    print("=" * 70, "\ntab:track\n" + "=" * 70)
    print(table_track(rows))
    print("\n" + "=" * 70, "\ntab:ablation\n" + "=" * 70)
    print(table_ablation(rows))
    out = figure(rows)
    print("\n[figure] ->", out)


if __name__ == "__main__":
    main()
