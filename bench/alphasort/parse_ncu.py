#!/usr/bin/env python3
"""Tabulate ncu counters across scale: one block per metric, variants x tokens.

Usage:  python3 bench/alphasort/parse_ncu.py
Reads   bench/alphasort/results/ncu_scale_<n>.csv  (one per token count)

The question these tables answer: as the corpus grows, do the ordered variants
decay because of L2 (hit rate falling) or because of coalescing (sectors per
request rising)? That distinction is the term Chapter 3's cost model is missing.
"""
import csv
import glob
import os
import re

RES = os.path.join(os.path.dirname(__file__), "results")
# Launch order = perm order in bench.cu with --nocpu 1.
NAMES = ["0 baseline", "2 gpu-sort-8B", "2b gpu-full", "3 gpu-prefix",
         "4 gpu-partition", "6 sort+compact", "6b coarse+compact"]
METRICS = [
    ("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
     "sectors / request  (coalescing; lower is better)", "{:>10.2f}"),
    ("smsp__sass_average_branch_targets_threads_uniform.pct",
     "branch uniformity %  (path sharing; higher is better)", "{:>10.2f}"),
    ("l1tex__t_sector_hit_rate.pct", "L1 hit %", "{:>10.2f}"),
    ("lts__t_sector_hit_rate.pct", "L2 hit %", "{:>10.2f}"),
    ("dram__bytes_read.sum", "DRAM read, GB  (absolute traffic)", "{:>10.2f}"),
]

runs = {}
for path in glob.glob(os.path.join(RES, "ncu_scale_*.csv")):
    n = int(re.search(r"ncu_scale_(\d+)\.csv$", path).group(1))
    per = {}
    for r in csv.DictReader(open(path)):
        per.setdefault(r["Metric Name"], []).append(
            float(r["Metric Value"].replace(",", "")))
    if per:
        runs[n] = per

if not runs:
    raise SystemExit(f"no ncu_scale_*.csv in {RES}")
ns = sorted(runs)

for key, title, fmt in METRICS:
    print(f"\n{title}")
    print(f"{'variant':<20}" + "".join(f"{n // 1000000:>9}M" for n in ns))
    print("-" * (20 + 10 * len(ns)))
    for i, name in enumerate(NAMES):
        row = f"{name:<20}"
        for n in ns:
            vals = runs[n].get(key, [])
            if i >= len(vals):
                row += f"{'-':>10}"
                continue
            v = vals[i] / 1e9 if key.startswith("dram") else vals[i]
            row += fmt.format(v)
        print(row)
print()
