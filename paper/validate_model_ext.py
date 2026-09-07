#!/usr/bin/env python3
"""Eq. (3) tested against 49 observations, with the factors measured directly.

The 21-point version of this test (paper/validate_model_21pt.py) left two
substitutions in place that it could not avoid, and both are removed here:

    nu   was inferred from the DRAM-traffic identity
         B/token = 32 nu (1-h1)(1-h2).  Now counted:
         l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / n.
    e    was the branch-target uniformity percentage, a proxy.  Now the
         quantity itself: smsp__thread_inst_executed_per_inst_executed.ratio
         / 32, the mean fraction of a warp's lanes retiring each instruction.

Seven scales x seven variants, validated by leave-one-scale-out: each fold
fits on six scales and predicts the seventh, so every reported error is on a
corpus size the fit never saw.  Model families are scored on identical folds.

Run from the repository root:  python3 paper/validate_model_ext.py
"""
import csv
import glob
import os
import re

import numpy as np

RES = "bench/alphasort/results"
ORDER = ["0 baseline", "2 gpu-sort-8B", "2b gpu-full", "3 gpu-prefix",
         "4 gpu-partition", "6 sort+compact", "6b sort-2B+compact"]
MET = {
    "h1": "l1tex__t_sector_hit_rate.pct",
    "h2": "lts__t_sector_hit_rate.pct",
    "B": "dram__bytes_read.sum",
    "unif": "smsp__sass_average_branch_targets_threads_uniform.pct",
    "sreq": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "req": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "sec": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "tie": "smsp__thread_inst_executed_per_inst_executed.ratio",
    "occ": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "stall": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
}

C, SCALES = {}, []
for path in sorted([q for q in glob.glob(f"{RES}/ncu_ext_*.csv")
                    if re.search(r"_(\d+)\.csv$", q)],
                   key=lambda p: int(re.search(r"_(\d+)\.csv$", p).group(1))):
    n = int(re.search(r"_(\d+)\.csv$", path).group(1))
    per = {}
    for r in csv.DictReader(open(path)):
        per.setdefault(r["Metric Name"], []).append(
            float(r["Metric Value"].replace(",", "")))
    if len(per.get(MET["h2"], [])) < len(ORDER):
        print(f"  skipping {os.path.basename(path)}: incomplete")
        continue
    C[n] = {}
    for i, v in enumerate(ORDER):
        c = {k: per[m][i] for k, m in MET.items()}
        C[n][v] = {
            "nu": c["sec"] / n,                 # transactions per token, counted
            "e": c["tie"] / 32.0,               # warp execution efficiency, Eq. (7)
            "e_proxy": c["unif"] / 100.0,       # the substitute used before
            "nu_proxy": c["sreq"],              # ditto
            "h1": c["h1"] / 100.0, "h2": c["h2"] / 100.0,
            "occ": c["occ"] / 100.0, "stall": c["stall"],
            "B": c["B"] / n, "rq": c["req"] / n,
        }
    SCALES.append(n)

TAU = {}
for tag in ("alpha",):
    for r in csv.DictReader(open(f"{RES}/scale_{tag}.csv")):
        n = int(r["tokens"])
        nom = min(SCALES, key=lambda s: abs(s - n)) if SCALES else n
        if SCALES and abs(nom - n) / nom < 0.02:
            TAU[(r["variant"], nom)] = float(r["ns_per_token"])
PTS = [(v, n) for n in SCALES for v in ORDER if (v, n) in TAU]
print(f"\n{len(PTS)} observations, {len(SCALES)} scales "
      f"({', '.join(f'{n//10**6}M' for n in SCALES)}), {len(ORDER)} variants")

BAR = "-" * 84

# ------------------------------------------------------------------ part 1
print("\n\nPART 1 — separability, with the factors measured rather than proxied")
print("Coefficient of variation across the full scale range, per variant.\n")
print(f"{'variant':<22}{'CV(nu)':>9}{'CV(e)':>9}{'CV(h1)':>9}{'CV(h2)':>9}"
      f"{'CV(occ)':>9}   {'nu':>6} {'e':>5}")
print(BAR)
agg = {k: [] for k in ("nu", "e", "h1", "h2", "occ")}
for v in ORDER:
    row = f"{v:<22}"
    for f in ("nu", "e", "h1", "h2", "occ"):
        xs = np.array([C[n][v][f] for n in SCALES])
        cv = xs.std(ddof=0) / xs.mean()
        agg[f].append(cv)
        row += f"{100 * cv:>8.2f}%"
    m = C[SCALES[0]][v]
    print(row + f"   {m['nu']:>6.1f} {m['e']:>5.3f}")
print(BAR)
print(f"{'mean':<22}" + "".join(f"{100*np.mean(agg[f]):>8.2f}%"
                                for f in ("nu", "e", "h1", "h2", "occ")))

# ------------------------------------------------------------------ models
def feats(v, n, kw):
    c = C[n][v]
    h1, h2 = (C[kw["freeze"]][v]["h1"], C[kw["freeze"]][v]["h2"]) \
        if kw.get("freeze") else (c["h1"], c["h2"])
    e = 1.0 if kw.get("no_e") else (c["e_proxy"] if kw.get("e_proxy") else c["e"])
    nu = c["nu_proxy"] if kw.get("nu_proxy") else c["nu"]
    k = nu / e
    mem = [k * h1, k * (1 - h1) * h2, k * (1 - h1) * (1 - h2)]
    fam = kw.get("fam", "B")
    if fam == "B":
        return mem
    if fam == "C":                       # additive issue term
        return [1.0 / e] + mem
    if fam == "D":                       # issue term + concurrency divisor
        return [1.0 / e] + [x / c["occ"] for x in mem]
    raise ValueError(fam)


def solve(train, **kw):
    X = np.array([feats(v, n, kw) for v, n in train])
    y = np.array([TAU[(v, n)] for v, n in train])
    w = 1.0 / y                          # fit relative error, not absolute
    return np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]


def loso(**kw):
    """Leave-one-scale-out. Returns (per-point errors, per-fold argmin hits)."""
    errs, hits, folds = [], 0, 0
    for held in SCALES:
        train = [(v, n) for v, n in PTS if n != held]
        test = [(v, n) for v, n in PTS if n == held]
        if not test:
            continue
        b = solve(train, **kw)
        X = np.array([feats(v, n, kw) for v, n in test])
        y = np.array([TAU[(v, n)] for v, n in test])
        p = X @ b
        errs.append(100 * np.abs(p - y) / y)
        hits += int(test[int(np.argmin(y))][0] == test[int(np.argmin(p))][0])
        folds += 1
    return np.concatenate(errs), hits, folds


print("\n\nPART 2 — leave-one-scale-out over model families")
print("Each error is on a corpus size its fit never saw.\n")
print(f"{'model':<52}{'MAPE':>8}{'worst':>8}{'argmin':>9}")
print(BAR)
FAMS = [
    ("B  tau = nu L_eff / e                    Eq. (3)", {"fam": "B"}),
    ("C  tau = (I + nu L_eff) / e              + issue term", {"fam": "C"}),
    ("D  tau = (I + nu L_eff / occupancy) / e  + concurrency", {"fam": "D"}),
]
best = None
for name, kw in FAMS:
    e, h, f = loso(**kw)
    print(f"{name:<52}{e.mean():>7.1f}%{e.max():>7.1f}%{f'{h}/{f}':>9}")
    if best is None or e.mean() < best[1]:
        best = (kw, e.mean(), name)
print(BAR)
print(f"argmin = folds in which the model picked the algorithm that actually "
      f"won.\nAnything below {len(SCALES)}/{len(SCALES)} means the model cannot be "
      f"used to choose one.")

print("\n\nPART 3 — ablations on the winning family")
print(f"({best[2].split()[0]}), same folds throughout.\n")
kw = dict(best[0])
print(f"{'variant of the model':<52}{'MAPE':>8}{'worst':>8}{'argmin':>9}")
print(BAR)
mu_err = []
for held in SCALES:
    tr = [TAU[(v, n)] for v, n in PTS if n != held]
    mu = float(np.mean(tr))
    mu_err.append([100 * abs(mu - TAU[(v, n)]) / TAU[(v, n)]
                   for v, n in PTS if n == held])
rows = [("as fitted", kw)]
rows += [("without the divergence factor e", {**kw, "no_e": True}),
         ("e as branch uniformity (the old proxy)", {**kw, "e_proxy": True}),
         ("nu as sectors per request (the old proxy)", {**kw, "nu_proxy": True}),
         ("both old proxies", {**kw, "e_proxy": True, "nu_proxy": True})]
for name, k in rows:
    e, h, f = loso(**k)
    print(f"{name:<52}{e.mean():>7.1f}%{e.max():>7.1f}%{f'{h}/{f}':>9}")
flat = np.concatenate([np.array(x) for x in mu_err])
print(f"{'null model (mean tau of the training folds)':<52}{flat.mean():>7.1f}%"
      f"{flat.max():>7.1f}%{'-':>9}")

print("\n\nPART 4 — per-point detail for the winning family, all held out")
b_full = {}
for held in SCALES:
    train = [(v, n) for v, n in PTS if n != held]
    bb = solve(train, **kw)
    for v, n in [(v, n) for v, n in PTS if n == held]:
        b_full[(v, n)] = (TAU[(v, n)], float(np.array(feats(v, n, kw)) @ bb))
print(f"\n{'variant':<22}" + "".join(f"{n//10**6:>6}M" for n in SCALES))
print(BAR)
for v in ORDER:
    print(f"{v:<22}" + "".join(
        f"{100*abs(b_full[(v,n)][1]-b_full[(v,n)][0])/b_full[(v,n)][0]:>6.0f}%"
        if (v, n) in b_full else f"{'-':>7}" for n in SCALES))
print(BAR)
print("relative error, per cent, of the held-out prediction\n")

# ------------------------------------------------------------------ part 5
print("\nPART 5 — the same model with one calibration constant per algorithm")
print("""PART 4 shows the baseline mispredicted by 41-44 % at EVERY scale: a
constant offset, not a drift.  That is the signature of a model that captures
how cost varies with corpus size but not how it compares between algorithms.
Eq. (3) is therefore re-tested in the form it would actually be deployed in —

    tau(pi, n) = k(pi) * nu(pi) L_eff(h(pi, n)) / e(pi)

with k(pi) measured ONCE per algorithm, at one corpus size, and the model
asked only to transfer that measurement to sizes it has not seen.  This is the
claim of Section 3.4b: measure the precision-dependent terms at any convenient
size, and the scale dependence follows.\n""")

kwB = {"fam": "B"}


def model_raw(v, n, kw=kwB):
    c = C[n][v]
    h1, h2 = c["h1"], c["h2"]
    return c["nu"] / c["e"] * (h1 * L[0] + (1 - h1) * h2 * L[1]
                               + (1 - h1) * (1 - h2) * L[2])


L = solve(PTS, **kwB)          # latency constants from the pooled fit
print(f"latency constants, pooled fit:  L1={L[0]:.4f}  L2={L[1]:.4f}  "
      f"L_DRAM={L[2]:.4f} ns/transaction   (L1<L2<L_DRAM: "
      f"{'yes' if L[0] < L[1] < L[2] else 'NO'})\n")

print(f"{'calibrated at':>14}{'held-out MAPE':>15}{'worst':>8}{'argmin':>9}")
print(BAR)
detail = {}
for n0 in SCALES:
    k = {v: TAU[(v, n0)] / model_raw(v, n0) for v in ORDER}
    errs, hits, folds = [], 0, 0
    for n in SCALES:
        if n == n0:
            continue
        y = np.array([TAU[(v, n)] for v in ORDER])
        p = np.array([k[v] * model_raw(v, n) for v in ORDER])
        errs.append(100 * np.abs(p - y) / y)
        hits += int(ORDER[int(np.argmin(y))] == ORDER[int(np.argmin(p))])
        folds += 1
        detail[(n0, n)] = (y, p)
    e = np.concatenate(errs)
    print(f"{n0//10**6:>12} M{e.mean():>14.1f}%{e.max():>7.1f}%{f'{hits}/{folds}':>9}")
print(BAR)

n0 = SCALES[0]
k = {v: TAU[(v, n0)] / model_raw(v, n0) for v in ORDER}
print(f"\nPer-point relative error, calibrated at {n0//10**6} M only:\n")
print(f"{'variant':<22}" + "".join(f"{n//10**6:>6}M" for n in SCALES) + f"{'k':>8}")
print(BAR)
for v in ORDER:
    row = f"{v:<22}"
    for n in SCALES:
        p = k[v] * model_raw(v, n)
        row += f"{100*abs(p-TAU[(v,n)])/TAU[(v,n)]:>6.0f}%" if n != n0 else f"{'cal':>7}"
    print(row + f"{k[v]:>8.2f}")
print(BAR)
print("\nThe crossover test: A6 wins at small corpora, A6b at large. Does the\n"
      "calibrated model reproduce the switch?\n")
print(f"{'tokens':>8}{'measured best':>26}{'predicted best':>26}")
print(BAR)
for n in SCALES:
    y = np.array([TAU[(v, n)] for v in ORDER])
    p = np.array([k[v] * model_raw(v, n) for v in ORDER])
    bm, bp = ORDER[int(np.argmin(y))], ORDER[int(np.argmin(p))]
    tag = "  (calibration point)" if n == n0 else ("" if bm == bp else "   <-- MISS")
    print(f"{n//10**6:>6} M{bm:>26}{bp:>26}{tag}")
print()

# ------------------------------------------------------------------ part 6
print("\nPART 6 — what the residual was hiding: latency exposure")
print("""PART 5 leaves the ordered variants decaying faster than nu L_eff / e can
express, and PART 3 shows the model IMPROVING when its factors are replaced by
cruder proxies -- the signature of a wrong functional form rather than of noisy
inputs.  The extended metric list contains the missing quantity:

    sigma(pi, n) = smsp__average_warps_issue_stalled_long_scoreboard
                        _per_issue_active.ratio

warp-cycles stalled on an outstanding memory operation per cycle in which the
scheduler could issue: the exposure of memory latency that Eq. (3) tries to
reach indirectly through a hit-rate-weighted average, and misses because an
average latency says nothing about how much of it is hidden.\n""")

print(f"{'variant':<22}" + "".join(f"{n//10**6:>7}M" for n in SCALES)
      + f"{'CV':>8}   quantity")
print(BAR)
for lab, key in (("nu", "nu"), ("sigma", "stall")):
    for v in ORDER:
        xs = np.array([C[n][v][key] for n in SCALES])
        print(f"{v:<22}" + "".join(f"{x:>8.1f}" for x in xs)
              + f"{100*xs.std()/xs.mean():>7.1f}%   {lab}")
    print(BAR)

print("""nu is flat in scale for every variant (CV 1.1-4.3 %): a pure function of
ordering precision, exactly as Proposition 2 requires.  sigma carries the whole
scale dependence, and its per-variant sensitivity reproduces the paper's
ranking without reference to any timing: the baseline 1.7 % and A6b 7.8 % are
the two strategies measured scale-invariant in Section 3.4, while A2b 43.3 %,
A2 38.7 % and A6 35.0 % are precisely the ones that decay.\n""")


def cols(v, n, fam):
    c = C[n][v]
    nu, e, h1, h2 = c["nu"], c["e"], c["h1"], c["h2"]
    s, rq = c["stall"] / 100, c["rq"]
    return {
        "E3":  [nu * h1 / e, nu * (1 - h1) * h2 / e, nu * (1 - h1) * (1 - h2) / e],
        "L1p": [nu * s],
        "L2p": [nu * s, 1.0],
        "L2e": [nu * s / e, 1.0],
        "L3p": [nu * s, 1.0, nu * (1 - h1) * (1 - h2)],
        "L3r": [rq * s, rq, 1.0],
    }[fam]


def loso2(fam):
    errs, hits = [], 0
    for held in SCALES:
        tr = [(v, n) for v, n in PTS if n != held]
        te = [(v, n) for v, n in PTS if n == held]
        X = np.array([cols(v, n, fam) for v, n in tr])
        y = np.array([TAU[k] for k in tr])
        w = 1 / y
        b = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        yt = np.array([TAU[k] for k in te])
        p = np.array([cols(v, n, fam) for v, n in te]) @ b
        errs.append(100 * np.abs(p - yt) / yt)
        hits += int(te[int(np.argmin(yt))][0] == te[int(np.argmin(p))][0])
    return np.concatenate(errs), hits, b


print(f"{'model':<44}{'p':>3}{'MAPE':>8}{'worst':>8}{'argmin':>9}")
print(BAR)
for name, fam in [
    ("Eq. (3), three-level L_eff", "E3"),
    ("alpha nu sigma", "L1p"),
    ("alpha nu sigma + beta", "L2p"),
    ("alpha nu sigma / e + beta   (divergence kept)", "L2e"),
    ("alpha nu sigma + beta + DRAM term", "L3p"),
    ("on requests rather than sectors", "L3r"),
]:
    e, h, b = loso2(fam)
    print(f"{name:<44}{len(b):>3}{e.mean():>7.1f}%{e.max():>7.1f}%{f'{h}/7':>9}")
print(BAR)

print("""
CONCLUSION OF THE VALIDATION

Equation (3) is not the right functional form.  Replacing its hit-rate-weighted
latency average with a direct measurement of latency exposure cuts held-out
error from 34.5 % to about 13 % with FEWER parameters, and the replacement

    tau(pi, n)  ~  alpha * nu(pi) * sigma(pi, n) + beta                  (3')

subsumes the divergence factor -- dividing by e makes the fit worse (23.9 %
against 13.1 %), so warp efficiency is not an independent term but a
contributor to how much latency a warp exposes.  The cache hit rates likewise
stop earning their place once sigma is present.

Proposition 2 survives in a stronger form than it was stated.  The separation
is not merely that two factors depend on precision and one on scale: nu is
flat to 2 % across a fiftyfold range for every algorithm, so the ENTIRE scale
dependence of the problem lives in one scalar per algorithm.

What must be conceded, and stated in the chapter: sigma is obtained from the
kernel it describes, so (3') explains and interpolates but does not predict
before the fact.  A predictive form needs sigma modelled from precision and
working-set size -- the chain is scale -> L2 residency -> latency exposure ->
time, and this study measures the endpoints of that chain but not its middle
link.  The A6/A6b crossover stays at the edge of resolution: the two differ by
under 20 % beyond 35 M tokens, and whether a given parameterization calls it
correctly is not stable across model families.  It should not be claimed.
""")
