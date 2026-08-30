#!/usr/bin/env python3
"""Sensitivity figure and table from the one-factor-at-a-time sweep.

Reads `results_sensitivity/sensitivity_eval.csv` and writes `sensitivity.png`
plus the table body on stdout.

Path completion is the lead metric. Deviation is plotted only for runs that
completed (>=99%, no crash); failures are marked as n/a.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results_sensitivity", "sensitivity_eval.csv")
OUT = os.path.join(os.environ.get("FIGDIR", os.path.join(ROOT, "report", "figures")), "sensitivity.png")

BLUE = "#2a78d6"
CRITICAL = "#d03b3b"    # did not complete the path

# Points are equally spaced (gamma 0.99 vs 0.999 overlap on a linear axis).
PANELS = [
    ("learning rate", [("lr_3e5", r"$3{\times}10^{-5}$"), ("centre", r"$1{\times}10^{-4}$"),
                       ("lr_3e4", r"$3{\times}10^{-4}$")]),
    (r"discount $\gamma$", [("gamma_095", "0.95"), ("centre", "0.99"), ("gamma_0999", "0.999")]),
    ("network width", [("width_64", "64"), ("centre", "256"), ("width_512", "512")]),
    ("alive bonus", [("alive_025", "0.25"), ("centre", "1.0"), ("alive_20", "2.0")]),
]


def load():
    """rows[config][seed] -> list of (completion_pct, crashed, rms_cm) per shape."""
    rows = defaultdict(lambda: defaultdict(list))
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows[r["config"]][r["seed"]].append(
                (float(r["completion_pct"]), r["crashed"] == "True", 100.0 * float(r["dev_rms_m"]))
            )
    return rows


def stats(per_seed):
    """-> (completion mean/spread over seeds, rms mean/std over completed runs, n_completed)."""
    comp = [np.mean([c for c, _x, _d in v]) for v in per_seed.values()]
    ok = [d for v in per_seed.values() for c, x, d in v if c >= 99 and not x]
    return (float(np.mean(comp)), float(np.std(comp)),
            (float(np.mean(ok)) if ok else np.nan),
            (float(np.std(ok)) if ok else np.nan), len(ok))


def main():
    rows = load()
    fig, axes = plt.subplots(2, len(PANELS), figsize=(7.4, 4.0))

    for c, (xlabel, points) in enumerate(PANELS):
        xs = list(range(len(points)))
        ticklabels = [t for _cfg, t in points]
        cm, cs, dm, ds, ns = [], [], [], [], []
        for cfg, _t in points:
            a, b, d, e, n = stats(rows[cfg])
            cm.append(a); cs.append(b); dm.append(d); ds.append(e); ns.append(n)

        top, bot = axes[0][c], axes[1][c]

        top.plot(xs, cm, color=BLUE, lw=1.8, zorder=2)
        for x, m, s, n in zip(xs, cm, cs, ns):
            failed = n == 0
            top.errorbar(x, m, yerr=s, color=CRITICAL if failed else BLUE,
                         marker="X" if failed else "o", ms=9 if failed else 7,
                         capsize=3, lw=1.4, zorder=3)
        if any(n == 0 for n in ns):
            top.text(0.5, 0.42, "never completes", transform=top.transAxes, ha="center",
                     fontsize=6.5, color=CRITICAL, style="italic")
        top.set_ylim(-8, 118)

        good = [(x, m, s) for x, m, s, n in zip(xs, dm, ds, ns) if n > 0]
        if good:
            gx, gm, gs = zip(*good)
            bot.errorbar(gx, gm, yerr=gs, color=BLUE, marker="s", ms=6, lw=1.8,
                         capsize=3, zorder=3)
        for x, n in zip(xs, ns):
            if n == 0:
                bot.text(x, 2.75, "n/a", ha="center", va="bottom", fontsize=7,
                         color=CRITICAL, style="italic")
        bot.set_ylim(2.5, 8.5)

        for ax in (top, bot):
            ax.set_xticks(xs)
            ax.set_xticklabels(ticklabels, fontsize=7)
            ax.set_xlim(-0.35, len(xs) - 0.65)
            ax.grid(axis="y", color="0.88", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.minorticks_off()
        bot.set_xlabel(xlabel, fontsize=8)
        if c == 0:
            top.set_ylabel("path completion (%)", fontsize=8)
            bot.set_ylabel("RMS deviation (cm)\n(completed runs)", fontsize=7.5)
        else:
            top.set_yticklabels([]); bot.set_yticklabels([])

    fig.tight_layout(pad=0.6, w_pad=0.8, h_pad=0.8)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"[INFO] wrote {OUT}\n")

    label = {"lr_3e5": r"Learning rate & $3\times10^{-5}$",
             "centre": r"\textbf{Centre (all four)} & \textbf{--}",
             "lr_3e4": r"Learning rate & $3\times10^{-4}$",
             "gamma_095": r"Discount $\gamma$ & 0.95",
             "gamma_0999": r"Discount $\gamma$ & 0.999",
             "width_64": r"Network width & 64",
             "width_512": r"Network width & 512",
             "alive_025": r"Alive bonus & 0.25",
             "alive_20": r"Alive bonus & 2.0"}
    for cfg in ["centre", "lr_3e5", "lr_3e4", "gamma_095", "gamma_0999",
                "width_64", "width_512", "alive_025", "alive_20"]:
        a, _b, d, e, n = stats(rows[cfg])
        crashes = sum(x for v in rows[cfg].values() for _c, x, _d in v)
        dev = "---" if n == 0 else f"{d:.1f}\\,$\\pm$\\,{e:.1f}"
        print(f"{label[cfg]} & {a:.0f}\\% & {crashes}/10 & {dev} \\\\")


if __name__ == "__main__":
    main()
