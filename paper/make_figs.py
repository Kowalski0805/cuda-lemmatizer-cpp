#!/usr/bin/env python3
"""Chapter 4 figures. Run from the repository root:  python3 paper/make_figs.py

Fig 4.1 (S4.3, P1)  precision saturation: the two opposing trends
Fig 4.2 (S4.6, P4)  per-token cost against corpus size
Fig 4.3 (S4.7)      single-pass economics and the operating window

Data: results/ncu_compact.csv is summarised inline (seven rows, already in
Table 4.2); the two scale figures read the sweep CSVs directly so the plots
cannot drift from the tables.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RES = "bench/alphasort/results"
OUT = "paper/figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  {OUT}/{name}.png")


def load(tag):
    """{variant: {tokens: row}} from a sweep CSV."""
    d = {}
    for r in csv.DictReader(open(f"{RES}/{tag}.csv")):
        d.setdefault(r["variant"], {})[int(r["tokens"])] = r
    return d


# --------------------------------------------------------------- figure 4.1
# Ordering precision on the x-axis, measured in Cyrillic letters that
# participate in the key (2 bytes per letter in UTF-8).  Counters from
# results/ncu_compact.csv, fiction 18.3 M tokens; times from the same run.
PREC = [
    #  label,          letters, branch unif, L1 hit, DRAM GB, lookup ms
    ("A0\nnone",             0,   72.78, 69.0, 0.40, 10.07),
    ("A4\n1 letter",         1,   78.61, 62.0, 1.35,  5.86),
    ("A3\n2 letters",        2,   82.57, 56.2, 1.47,  8.02),
    ("A2\n4 letters",        4,   91.78, 38.3, 1.48,  9.91),
    ("A2b\nexact",           8,   98.45, 25.3, 1.47, 10.07),
]
x = list(range(len(PREC)))
lab = [p[0] for p in PREC]

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.4, 4.4), sharex=True,
                              gridspec_kw={"height_ratios": [1.25, 1]})

ax.plot(x, [p[2] for p in PREC], "o-", color="#1b4965",
        label="warp uniformity (path sharing)")
ax.plot(x, [p[3] for p in PREC], "s--", color="#bc4b51",
        label="L1 sector hit rate (locality)")
ax.set_ylabel("percent")
ax.set_ylim(15, 108)
ax.legend(loc="lower left", frameon=False)
ax.annotate("sorting buys uniformity\nand pays in locality",
            xy=(2.55, 72), fontsize=8, color="#555555", ha="left", va="center")

ax2.bar(x, [p[5] for p in PREC], width=0.55, color="#8d99ae",
        edgecolor="#3d405b", linewidth=0.6)
ax2.axhline(PREC[0][5], color="#3d405b", linestyle=":", linewidth=1.0)
ax2.set_ylabel("traversal time, ms")
ax2.set_xlabel("ordering precision (leading Cyrillic letters in the key)")
ax2.set_xticks(x)
ax2.set_xticklabels(lab)
for xi, p in zip(x, PREC):
    ax2.text(xi, p[5] + 0.3, f"{p[5]:.2f}", ha="center", fontsize=7.5,
             color="#3d405b")
ax2.set_ylim(0, 12.2)
save(fig, "fig4_1_saturation")

# --------------------------------------------------------------- figure 4.2
SERIES = [
    ("0 baseline",         "A0 baseline",       "#3d405b", "o", "-"),
    ("2 gpu-sort-8B",      "A2 sort, 4 letters", "#bc4b51", "s", "-"),
    ("3 gpu-prefix",       "A3 prefix, 2 letters", "#e09f3e", "^", "-"),
    ("4 gpu-partition",    "A4 partition, 1 letter", "#7f9c6c", "v", "-"),
    ("6 sort+compact",     "A6 sort + compact", "#1b4965", "D", "--"),
    ("6b sort-2B+compact", "A6b coarse + compact", "#5fa8d3", "P", "--"),
]
d = load("scale_alpha")
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for key, label, colour, marker, ls in SERIES:
    pts = sorted(d[key].items())
    ax.plot([n for n, _ in pts], [float(r["ns_per_token"]) for _, r in pts],
            marker=marker, ls=ls, color=colour, label=label, markersize=4)
ax.set_xscale("log")
ax.set_xlabel("corpus size, tokens")
ax.set_ylabel("traversal cost, ns / token")
ax.set_ylim(0, 0.65)
ax.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f"{v/1e6:g} M" if v >= 1e6 else f"{v/1e3:g} K"))
ax.set_xticks([1e6, 2e6, 5e6, 1e7, 2e7, 3.5e7, 5e7])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3,
          frameon=False, columnspacing=1.4, handletextpad=0.5)
ax.annotate("A2 crosses the baseline near 27 M tokens:\nexact ordering is a net loss thereafter",
            xy=(2.7e7, 0.545), xytext=(1.15e6, 0.615), fontsize=7.5,
            color="#bc4b51", ha="left",
            arrowprops=dict(arrowstyle="->", color="#bc4b51", lw=0.8,
                            connectionstyle="arc3,rad=-0.15"))
save(fig, "fig4_2_scale")

# --------------------------------------------------------------- figure 4.3
d = load("scale_fine2")
base = {n: float(r["lookup_ms"]) for n, r in d["0 baseline"].items()}
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for key, label, colour, marker, ls in SERIES[1:]:
    pts = sorted(d[key].items())
    ax.plot([n for n, _ in pts],
            [base[n] / (float(r["prep_ms"]) + float(r["lookup_ms"]))
             for n, r in pts],
            marker=marker, ls=ls, color=colour, label=label, markersize=4)
ax.axhline(1.0, color="black", linewidth=0.9)
ax.axvspan(4.4e5, 8.2e6, color="#e09f3e", alpha=0.12, lw=0)
ax.axvline(8.2e6, color="#8a6d1f", ls=":", lw=1.0)
ax.text(1.55e6, 0.34, "operating window: A3 pays on a single pass,\n"
        "peak 1.94\u00d7 at 1.5 M, break-even 8.2 M",
        ha="center", fontsize=7.5, color="#8a6d1f")
ax.set_xlim(4.4e5, 1.15e7)
ax.set_xscale("log")
ax.set_xlabel("batch size, tokens")
ax.set_ylabel("single-pass speedup  (preparation charged)")
ax.set_ylim(0.25, 2.15)
ax.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f"{v/1e6:g} M" if v >= 1e6 else f"{v/1e3:g} K"))
ax.set_xticks([5e5, 1e6, 2e6, 4e6, 6e6, 1e7])
ax.legend(loc="upper right", frameon=False)
save(fig, "fig4_3_oneshot")
