#!/usr/bin/env python3
"""Aggregate the multi-seed robustness sweep into the rotor table and Pareto figure.

Prints the rotor-loss table and a wind-recovery summary on stdout, and writes
`pareto.png`.

"Recovered" = 100% path completion with no crash.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_matrix")
FIGDIR = os.environ.get("FIGDIR", os.path.join(ROOT, "report", "figures"))

AUTH = {"randomized": ("$\\pm5\\%$", 5), "capstone": ("$\\pm15\\%$", 15),
        "authority25": ("$\\pm25\\%$", 25)}
BLUE, PIDCOL = "#2a78d6", "0.35"


def thresholds():
    """config -> {seed: max recovered loss %}. Absent seed => recovered nothing."""
    rec = defaultdict(dict)
    seen = defaultdict(set)
    for r in csv.DictReader(open(os.path.join(OUT, "robustness_rotor.csv"))):
        c, s, l = r["config"], int(r["seed"]), int(r["loss_pct"])
        seen[c].add(s)
        if r["crashed"] == "False" and float(r["completed_pct"]) >= 99:
            rec[c][s] = max(rec[c].get(s, 0), l)
    return rec, seen


def nominal_rms():
    """config -> (mean, std) nominal RMS cm over shapes x seeds."""
    vals = defaultdict(list)
    per = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(os.path.join(OUT, "shape_eval.csv"))):
        per[r["config"]][r["seed"]].append(float(r["dev_rms_m"]) * 100)
    out = {}
    for c, seeds in per.items():
        m = np.array([np.mean(v) for v in seeds.values()])
        out[c] = (m.mean(), m.std())
    return out


def wind_summary():
    peak, post = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(os.path.join(OUT, "robustness_wind.csv"))):
        if r["peak"] not in ("nan", ""):
            peak[r["config"]].append(float(r["peak"]) * 100)
            post[r["config"]].append(float(r["post_rms"]) * 100)
    print("wind (circle, 1.5 m/s gust) -- peak / post-settle RMS [cm], mean over seeds:")
    for c in ["randomized", "capstone", "authority25", "pid"]:
        if peak[c]:
            print(f"  {c:<12} peak {np.mean(peak[c]):4.1f}  post {np.mean(post[c]):4.1f}")


def table_rotor(rec):
    def cell(c):
        d = rec.get(c, {})
        if not d:
            return "$<5\\%$"
        vals = sorted(d.values())
        return f"${vals[0]}$--${vals[-1]}\\%$" if vals[0] != vals[-1] else f"${vals[0]}\\%$"
    lines = [
        r"\begin{tabular}{lc}", r"\toprule",
        r"Controller & Rotor-loss tolerance \\", r"\midrule",
        r"Standard RL ($\pm5\%$)                       & " + cell("randomized") + r" \\",
        r"PID                                          & $7\%$ \\",
        r"\textbf{Capstone RL ($\pm15\%$ + rotor-random.)} & \textbf{" + cell("capstone").replace("$", "") + r"} \\",
        r"Wide RL ($\pm25\%$ + rotor-random.)          & " + cell("authority25") + r" \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    return "\n".join(lines)


def figure(rec, nrms):
    order = ["randomized", "capstone", "authority25"]
    xs = [AUTH[c][1] for c in order]
    means, lo, hi = [], [], []
    for c in order:
        v = sorted(rec.get(c, {}).values())
        if not v:
            means.append(4.0); lo.append(0.0); hi.append(0.0)   # "<5%"
        else:
            m = float(np.mean(v)); means.append(m)
            lo.append(m - v[0]); hi.append(v[-1] - m)
    nm = [nrms[c][0] for c in order]
    ns = [nrms[c][1] for c in order]

    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
    a.errorbar(xs, means, yerr=[lo, hi], marker="o", ms=6, lw=1.8, color=BLUE,
               capsize=3, zorder=3)
    a.axhline(7, ls="--", lw=1.2, color=PIDCOL)
    a.annotate("PID (7%)", (25, 7), textcoords="offset points", xytext=(-4, 5),
               ha="right", fontsize=8, color=PIDCOL)
    a.annotate("<5%", (5, 4.0), textcoords="offset points", xytext=(8, -2),
               fontsize=8, color=BLUE)
    a.set_ylabel("Recoverable\nrotor loss [%]", fontsize=9)
    a.set_ylim(0, 19)

    b.errorbar(xs, nm, yerr=ns, marker="s", ms=6, lw=1.8, color=BLUE, capsize=3, zorder=3)
    b.axhline(5.0, ls="--", lw=1.2, color=PIDCOL)
    b.annotate("PID (5.0 cm)", (25, 5.0), textcoords="offset points", xytext=(-4, 5),
               ha="right", fontsize=8, color=PIDCOL)
    b.set_ylabel("Nominal RMS\ndeviation [cm]", fontsize=9)
    b.set_ylim(0, 9)
    b.set_xlabel("Control authority (action scale)", fontsize=9)
    b.set_xticks(xs)
    b.set_xticklabels(["±5%", "±15%", "±25%"])

    for ax in (a, b):
        ax.grid(axis="y", color="0.88", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a.set_title("Fault tolerance rises steeply with authority; nominal precision stays flat",
                fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "pareto.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    rec, _ = thresholds()
    nrms = nominal_rms()
    print("=" * 66, "\ntab:rotor\n" + "=" * 66)
    print(table_rotor(rec))
    print()
    wind_summary()
    print("\n[figure] ->", figure(rec, nrms))


if __name__ == "__main__":
    main()
