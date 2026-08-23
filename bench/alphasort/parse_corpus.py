#!/usr/bin/env python3
"""Corpus sweep: token count fixed, corpus varied. The generality check.

Reads  bench/alphasort/results/corpus_<name>_<n>.txt
Usage  python3 bench/alphasort/parse_corpus.py
"""
import glob
import os
import re
from collections import defaultdict

RES = os.path.join(os.path.dirname(__file__), "results")
ROW = re.compile(r"^(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)\s*$")
LOADED = re.compile(r"^loaded (\d+) words \((\d+) unique, type/token ([\d.]+)\)")
HIT = re.compile(r"^hit rate: \d+ / \d+ tokens recognised \(([\d.]+)%\)")

runs = {}
for path in glob.glob(os.path.join(RES, "corpus_*.txt")):
    m = re.search(r"corpus_([a-z]+)_(\d+)\.txt$", path)
    if not m:
        continue
    rec = {"variants": {}}
    for line in open(path, encoding="utf-8", errors="replace"):
        if g := LOADED.match(line):
            rec.update(n=int(g[1]), uniq=int(g[2]), tt=float(g[3]))
        elif g := HIT.match(line):
            rec["hit"] = float(g[1])
        elif g := ROW.match(line.rstrip()):
            if g[1].strip() != "variant":
                rec["variants"][g[1].strip()] = {
                    "lookup": float(g[3]), "speedup": float(g[6]),
                    "nspt": float(g[3]) * 1e6 / int(rec["n"]),
                }
    if rec["variants"]:
        runs[(m[1], int(m[2]))] = rec

by_n = defaultdict(list)
for (name, n), rec in runs.items():
    by_n[n].append((name, rec))

names = list(next(iter(runs.values()))["variants"])
for n in sorted(by_n):
    print(f"\n=== {n:,} tokens ===\n")
    hdr = f"{'corpus':<10}{'uniq':>9}{'type/tok':>10}{'hit%':>7}   " + \
          "".join(f"{v.split()[0]:>9}" for v in names)
    print(hdr)
    print("-" * len(hdr))
    for name, rec in sorted(by_n[n]):
        line = f"{name:<10}{rec['uniq']:>9}{rec['tt']:>10.4f}{rec.get('hit',0):>7.2f}   "
        line += "".join(f"{rec['variants'][v]['speedup']:>8.2f}x" for v in names)
        print(line)
    print()
    print(f"{'  ns/token':<10}{'':>26}   " + "".join(f"{v.split()[0]:>9}" for v in names))
    for name, rec in sorted(by_n[n]):
        line = f"{name:<10}{'':>26}   "
        line += "".join(f"{rec['variants'][v]['nspt']:>9.3f}" for v in names)
        print(line)
print()
