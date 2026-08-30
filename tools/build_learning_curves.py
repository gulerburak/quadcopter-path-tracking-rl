#!/usr/bin/env python3
"""Learning curves from EvalCallback `evaluations.npz` files.

Writes `learning_curves.png`. One column per curriculum stage
(eval return is not comparable across stages) and two rows (fault vs no-fault eval).
"""
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "results_matrix", "models")
OUT = os.path.join(os.environ.get("FIGDIR", os.path.join(ROOT, "report", "figures")), "learning_curves.png")

# Colours match build_report_tables.py; slot 4 (PID yellow) is unused here.
STYLE = {
    "baseline":    ("#2a78d6", "-",  "Baseline"),
    "control":     ("#e87ba4", "--", "+ crash terminates"),
    "randomized":  ("#eb6834", ":",  "+ init-state random."),
    "capstone":    ("#1baf7a", "-",  r"Capstone ($\pm$15%)"),
    "authority25": ("#4a3aa7", "--", r"Wide ($\pm$25%)"),
}
ROWS = [
    (["baseline", "control", "randomized"], "eval return\n" + r"($\pm$5%, no fault in eval)"),
    (["capstone", "authority25"], "eval return\n(fault-trained, fault in eval)"),
]
STAGES = [
    (1, "Stage 1: 2 m, 1 segment, 6 s"),
    (2, "Stage 2: 4 m, 2 segments, 10 s"),
    (3, "Stage 3: 6 m, 4 segments, 16 s"),
]


def load():
    """runs[config][stage] -> list of (timesteps, mean_reward) arrays, one per seed."""
    runs = defaultdict(lambda: defaultdict(list))
    pat = re.compile(r"models/([a-z0-9]+)_s(\d)/ppo_[^/]+/stage(\d)/")
    for f in sorted(glob.glob(os.path.join(MATRIX, "*", "ppo_*", "stage*", "evaluations.npz"))):
        m = pat.search(f)
        if not m:
            continue
        cfg, _seed, stage = m.group(1), int(m.group(2)), int(m.group(3))
        d = np.load(f)
        # results is (n_evals, n_episodes); average the 10 eval episodes per point.
        runs[cfg][stage].append((d["timesteps"], d["results"].mean(axis=1)))
    return runs


SMOOTH = 5  # rolling-mean window, in eval points (1 point = 10k steps)


def band(seed_curves, n=200):
    """Interpolate seeds onto a common x grid, then mean +- std.

    Stage boundaries land on slightly different global step counts per seed, so
    raw x vectors cannot be averaged elementwise. Each seed is smoothed with a
    centred rolling mean of SMOOTH eval points first.
    """
    lo = max(c[0][0] for c in seed_curves)
    hi = min(c[0][-1] for c in seed_curves)
    grid = np.linspace(lo, hi, n)
    ys = []
    for ts, raw in seed_curves:
        r = raw
        if SMOOTH > 1:
            k = np.ones(SMOOTH) / SMOOTH
            r = np.convolve(raw, k, mode="same")
            # np.convolve tapers the ends against zero; rebuild from partial means.
            half = SMOOTH // 2
            for i in list(range(half)) + list(range(len(r) - half, len(r))):
                r[i] = raw[max(0, i - half):i + half + 1].mean()
        ys.append(np.interp(grid, ts, r))
    ys = np.vstack(ys)
    return grid, ys.mean(axis=0), ys.std(axis=0)


def main():
    runs = load()
    n_runs = sum(len(v[3]) for v in runs.values())
    print(f"[INFO] loaded {n_runs} runs x {len(STAGES)} stages")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for r, (cfgs, row_note) in enumerate(ROWS):
        for c, (stage, title) in enumerate(STAGES):
            ax = axes[r][c]
            for cfg in cfgs:
                curves = runs[cfg][stage]
                if not curves:
                    continue
                color, ls, label = STYLE[cfg]
                x, mu, sd = band(curves)
                xm = x / 1e6
                ax.fill_between(xm, mu - sd, mu + sd, color=color, alpha=0.16, lw=0)
                ax.plot(xm, mu, color=color, ls=ls, lw=1.8, label=label, zorder=3)
            ax.grid(axis="y", color="0.88", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(title, fontsize=8)
            if r == 1:
                ax.set_xlabel("training steps (M)", fontsize=8)
            if c == 0:
                ax.set_ylabel(row_note, fontsize=7.5)
        axes[r][0].legend(fontsize=6.5, frameon=False, loc="upper left")

    fig.tight_layout(pad=0.6, w_pad=1.0, h_pad=1.0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"[INFO] wrote {OUT}")

    print("\nfinal stage-3 eval return (mean +- std over seeds):")
    for cfg in STYLE:
        curves = runs[cfg][3]
        if not curves:
            continue
        finals = np.array([r[-10:].mean() for _ts, r in curves])
        print(f"  {cfg:12} {finals.mean():7.0f} +- {finals.std():5.0f}   (n={len(finals)})")


if __name__ == "__main__":
    main()
