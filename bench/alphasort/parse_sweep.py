#!/usr/bin/env python3
"""Summarise a scale sweep: one row per token count, speedup per variant.

Usage:  python3 bench/alphasort/parse_sweep.py [key]      (default: alpha)
Reads   bench/alphasort/results/scale_<key>_<n>.txt
Writes  bench/alphasort/results/scale_<key>.csv and prints a table.
"""
import csv
import glob
import os
import re
import sys

KEY = sys.argv[1] if len(sys.argv) > 1 else "alpha"
RES = os.path.join(os.path.dirname(__file__), "results")

# "0 baseline   0.000  10.074  10.061  1.099  1.00x  1818.9"
ROW = re.compile(
    r"^(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)\s*$"
)
LOADED = re.compile(r"^loaded (\d+) words \((\d+) unique, type/token ([\d.]+)\)")
HIT = re.compile(r"^hit rate: (\d+) / (\d+) tokens recognised \(([\d.]+)%\)")

runs = []
for path in sorted(glob.glob(os.path.join(RES, f"scale_{KEY}_*.txt")),
                   key=lambda p: int(re.search(r"_(\d+)\.txt$", p).group(1))):
    rec = {"file": os.path.basename(path), "variants": {}}
    for line in open(path, encoding="utf-8", errors="replace"):
        if m := LOADED.match(line):
            rec["n"], rec["uniq"], rec["tt"] = int(m[1]), int(m[2]), float(m[3])
        elif m := HIT.match(line):
            rec["hit"] = float(m[3])
        elif m := ROW.match(line.rstrip()):
            name = m[1].strip()
            if name == "variant":
                continue
            rec["variants"][name] = {
                "prep": float(m[2]), "lookup": float(m[3]),
                "best": float(m[4]), "spread": float(m[5]),
                "speedup": float(m[6]), "mwps": float(m[7]),
            }
    if "n" in rec and rec["variants"]:
        runs.append(rec)

if not runs:
    sys.exit(f"no parsable results in {RES}/scale_{KEY}_*.txt")

names = list(runs[-1]["variants"])
# Corpus bytes are ~11.9 B/token for these Ukrainian corpora; L2 is 64 MB.
L2_MB = 64.0

print(f"\nscale sweep, key={KEY}   (L2 = {L2_MB:.0f} MB)\n")
hdr = f"{'tokens':>10} {'uniq':>9} {'hit%':>6} {'corpus MB':>10}  " + \
      "".join(f"{n.split()[0]:>9}" for n in names)
print(hdr)
print("-" * len(hdr))
for r in runs:
    mb = r["n"] * 11.9 / 1e6
    line = f"{r['n']:>10} {r['uniq']:>9} {r.get('hit', 0):>6.2f} {mb:>10.0f}  "
    line += "".join(f"{r['variants'][n]['speedup']:>8.2f}x" for n in names)
    print(line)
print("\n(cells are lookup speedup vs baseline at that token count)\n")

# Absolute lookup times, to see whether the baseline itself degrades with size.
print(f"{'tokens':>10}  " + "".join(f"{n.split()[0]:>9}" for n in names) +
      "   (lookup ms)")
for r in runs:
    print(f"{r['n']:>10}  " +
          "".join(f"{r['variants'][n]['lookup']:>9.2f}" for n in names))

# Per-token cost isolates the cache effect from the problem size.
print(f"\n{'tokens':>10}  " + "".join(f"{n.split()[0]:>9}" for n in names) +
      "   (ns/token)")
for r in runs:
    print(f"{r['n']:>10}  " +
          "".join(f"{r['variants'][n]['lookup'] * 1e6 / r['n']:>9.2f}"
                  for n in names))

# One-shot economics: does the ordering pay with no reuse at all?
print(f"\none-shot speedup = base_lookup / (prep + lookup);  >1 wins with no "
      f"reuse\n")
print(f"{'tokens':>10}  " + "".join(f"{n.split()[0]:>9}" for n in names[1:]))
oneshot = {n: [] for n in names[1:]}
for r in runs:
    b = r["variants"][names[0]]["lookup"]
    line = f"{r['n']:>10}  "
    for n in names[1:]:
        v = r["variants"][n]
        s = b / (v["prep"] + v["lookup"])
        oneshot[n].append((r["n"], s))
        line += f"{s:>8.2f}x"
    print(line)

# Where does each variant cross 1.00? Linear interpolation in log(tokens).
import math
print("\none-shot break-even corpus size (interpolated):")
for n, pts in oneshot.items():
    cross = None
    for (n0, s0), (n1, s1) in zip(pts, pts[1:]):
        if (s0 - 1.0) * (s1 - 1.0) < 0:  # sign change
            t = (1.0 - s0) / (s1 - s0)
            cross = math.exp(math.log(n0) + t * (math.log(n1) - math.log(n0)))
    if cross:
        print(f"  {n:<22} crosses 1.00x at ~{cross/1e6:.2f} M tokens")
    elif all(s > 1.0 for _, s in pts):
        print(f"  {n:<22} wins one-shot across the whole range")
    else:
        print(f"  {n:<22} never wins one-shot in this range")

out = os.path.join(RES, f"scale_{KEY}.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tokens", "unique", "type_token", "hit_pct", "corpus_mb",
                "variant", "prep_ms", "lookup_ms", "best_ms", "speedup",
                "mwords_per_s", "ns_per_token"])
    for r in runs:
        for n, v in r["variants"].items():
            w.writerow([r["n"], r["uniq"], r["tt"], r.get("hit", ""),
                        round(r["n"] * 11.9 / 1e6, 1), n, v["prep"],
                        v["lookup"], v["best"], v["speedup"], v["mwps"],
                        round(v["lookup"] * 1e6 / r["n"], 3)])
print(f"\nwrote {out}")
