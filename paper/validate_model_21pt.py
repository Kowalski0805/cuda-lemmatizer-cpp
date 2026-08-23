#!/usr/bin/env python3
"""Quantitative validation of the three-factor model, Eq. (3) of Chapter 3.

    tau(pi, n)  ~  nu(pi) * L_eff(h(pi, n)) / e(pi)          (3)
    L_eff       =  h * L_cache + (1 - h) * L_DRAM            (6)

Proposition 2 asserts separability: nu and e depend on the ordering precision
alone, h on the working-set size alone.  Both halves are testable against data
already collected, and the test here is a prediction rather than a fit: the
latency constants are estimated at ONE corpus size and used unchanged at the
others, so two thirds of the observations are held out.

MEASURING THE FACTORS.  Nothing below is tuned; each is a counter or an
identity between counters.

  nu(pi)   transactions per token.  Not directly counted, but recoverable:
           every sector that misses both caches becomes DRAM traffic, so
                 B/token = 32 * nu * (1 - h1) * (1 - h2)
           and nu = B / (32 (1-h1)(1-h2)).  An earlier version of this script
           used sectors *per request* instead, which is not the same quantity
           -- it omits how many requests a token issues, and that differs by
           a factor of four between the compacted and permuted variants.  With
           that substitution the model does not survive its own test; the
           result in PART 5 is what it looked like.
  e(pi)    smsp__sass_average_branch_targets_threads_uniform.pct / 100.
           A proxy for Eq. (7): the hardware reports branch-target uniformity,
           not the ratio of mean to maximum trip count.  They coincide only
           when divergence is dominated by the traversal loop, which is the
           case here but is an assumption, not a measurement.
  h1, h2   l1tex__ and lts__t_sector_hit_rate.pct / 100.

Eq. (6) is written for one cache level.  The device has two, and the token
stream hits in L1 while the trie stream misses to L2, so the two cannot be
lumped without losing exactly the distinction the chapter draws.  L_eff is
therefore evaluated at three levels,

    L_eff = h1 L1 + (1-h1) h2 L2 + (1-h1)(1-h2) L_DRAM      (6')

which is Eq. (6) with the cache term resolved into its two components.  Three
free parameters against seven observations at the fit scale.

Run from the repository root:  python3 paper/validate_model.py
"""
import csv

import numpy as np

RES = "bench/alphasort/results"
ORDER = ["0 baseline", "2 gpu-sort-8B", "2b gpu-full", "3 gpu-prefix",
         "4 gpu-partition", "6 sort+compact", "6b sort-2B+compact"]
MET = {
    "sreq": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "e": "smsp__sass_average_branch_targets_threads_uniform.pct",
    "h1": "l1tex__t_sector_hit_rate.pct",
    "h2": "lts__t_sector_hit_rate.pct",
    "B": "dram__bytes_read.sum",
}
SCALES = [1_000_000, 20_000_000, 50_000_000]


def load(n):
    per = {}
    for r in csv.DictReader(open(f"{RES}/ncu_scale_{n}.csv")):
        per.setdefault(r["Metric Name"], []).append(
            float(r["Metric Value"].replace(",", "")))
    out = {}
    for i, v in enumerate(ORDER):
        c = {k: per[m][i] for k, m in MET.items()}
        c["e"] /= 100.0
        c["h1"] /= 100.0
        c["h2"] /= 100.0
        c["B"] /= n                                    # bytes per token
        c["nu"] = c["B"] / (32 * (1 - c["h1"]) * (1 - c["h2"]))
        out[v] = c
    return out


C = {n: load(n) for n in SCALES}
TAU = {}
for r in csv.DictReader(open(f"{RES}/scale_alpha.csv")):
    n = int(r["tokens"])
    nom = min(SCALES, key=lambda s: abs(s - n))
    if abs(nom - n) / nom < 0.02:
        TAU[(r["variant"], nom)] = float(r["ns_per_token"])

BAR = "-" * 78

# ------------------------------------------------------------------ part 1
print("\nPART 1 — separability (Proposition 2)")
print("Coefficient of variation across the fiftyfold scale range, per variant.\n")
print(f"{'variant':<22}{'CV(nu)':>9}{'CV(e)':>9}{'CV(h1)':>9}{'CV(h2)':>9}"
      f"   {'h2: 1M -> 50M':>15}")
print(BAR)
agg = {k: [] for k in ("nu", "e", "h1", "h2")}
for v in ORDER:
    row = f"{v:<22}"
    for f in ("nu", "e", "h1", "h2"):
        xs = np.array([C[n][v][f] for n in SCALES])
        cv = xs.std(ddof=0) / xs.mean()
        agg[f].append(cv)
        row += f"{100 * cv:>8.2f}%"
    row += f"   {C[SCALES[0]][v]['h2']:>6.3f} ->{C[SCALES[-1]][v]['h2']:>6.3f}"
    print(row)
print(BAR)
print(f"{'mean':<22}" + "".join(f"{100 * np.mean(agg[f]):>8.2f}%"
                                for f in ("nu", "e", "h1", "h2")))

# ------------------------------------------------------------------ model
def design(n, freeze=None, no_e=False):
    X, y, keep = [], [], []
    for v in ORDER:
        if (v, n) not in TAU:
            continue
        c = C[n][v]
        h1, h2 = (C[freeze][v]["h1"], C[freeze][v]["h2"]) if freeze else (c["h1"], c["h2"])
        e = 1.0 if no_e else c["e"]
        nu = c["nu"]
        X.append([nu * h1 / e, nu * (1 - h1) * h2 / e, nu * (1 - h1) * (1 - h2) / e])
        y.append(TAU[(v, n)])
        keep.append(v)
    return np.array(X), np.array(y), keep


def fit(n, **kw):
    """Least squares on RELATIVE error.  tau spans 0.09 to 0.59 ns/token, so
    unweighted OLS would fit the slow variants and ignore the fast ones --
    exactly backwards, since the fast ones are the paper's recommendation."""
    X, y, _ = design(n, **kw)
    w = 1.0 / y
    return np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]


def score(beta, n, **kw):
    X, y, keep = design(n, **kw)
    p = X @ beta
    return 100 * np.abs(p - y) / y, y, p, keep


def heldout(n_fit, **kw):
    b = fit(n_fit, **kw)
    return np.concatenate([score(b, n, **kw)[0] for n in SCALES if n != n_fit])


# ------------------------------------------------------------------ part 2
print("\n\nPART 2 — prediction from a single operating point")
print("Every fit scale tried, so the result cannot depend on a lucky choice.\n")
print(f"{'fitted at':>10}{'L1':>9}{'L2':>9}{'L_DRAM':>9}   "
      f"{'held-out MAPE':>14}{'worst':>8}")
print(BAR)
for nf in SCALES:
    b = fit(nf)
    h = heldout(nf)
    print(f"{nf//10**6:>8} M{b[0]:>9.4f}{b[1]:>9.4f}{b[2]:>9.4f}   "
          f"{h.mean():>13.1f}%{h.max():>7.1f}%")
print(BAR)
print("L1 < L2 < L_DRAM is required for the fit to be physically meaningful:\n"
      "a hit that costs more than a miss means the model is absorbing\n"
      "something other than latency into its coefficients.")

FIT_AT = 20_000_000
beta = fit(FIT_AT)
print(f"\n\nPer-point detail, fitted at {FIT_AT//10**6} M tokens.\n")
print(f"{'variant':<22}" + "".join(f"{f'--- {n//10**6} M ---':^24}" for n in SCALES))
print(f"{'':<22}" + "".join(f"{'meas':>8}{'pred':>8}{'err':>8}" for _ in SCALES))
print("-" * (22 + 24 * len(SCALES)))
cell = {}
for n in SCALES:
    err, y, p, keep = score(beta, n)
    for v, a, b_, c_ in zip(keep, y, p, err):
        cell[(v, n)] = (a, b_, c_)
for v in ORDER:
    print(f"{v:<22}" + "".join(
        f"{cell[(v,n)][0]:>8.3f}{cell[(v,n)][1]:>8.3f}{cell[(v,n)][2]:>7.1f}%"
        for n in SCALES))
print("-" * (22 + 24 * len(SCALES)))
print(f"{'MAPE':<22}" + "".join(
    f"{'':>16}{score(beta,n)[0].mean():>7.1f}%" for n in SCALES)
    + "   (the middle column is the fit)")

# ------------------------------------------------------------------ part 3
print("\n\nPART 3 — ablations, scored on the same held-out points")
print(f"All fitted at {FIT_AT//10**6} M tokens.\n")
mu = np.mean([TAU[(v, FIT_AT)] for v in ORDER])
null = np.concatenate([[100 * abs(mu - TAU[(v, n)]) / TAU[(v, n)] for v in ORDER]
                       for n in SCALES if n != FIT_AT])
def issue_term_model(n_fit):
    """Eq. (3) plus a constant instruction cost per token, tau = (I + nu L)/e.
    Section 2.1's clock experiment argues for it: the baseline scales with core
    frequency at 0.98, so its time is issue, not memory, and no purely memory
    model can cover both regimes.  With seven observations per scale and four
    free parameters the extension is not identifiable, which is itself the
    finding -- the experiment needed to test it is more counter runs, not a
    better fit."""
    def cols(n):
        X, y = [], []
        for v in ORDER:
            c = C[n][v]
            k = c["nu"] / c["e"]
            X.append([1.0 / c["e"], k * c["h1"], k * (1 - c["h1"]) * c["h2"],
                      k * (1 - c["h1"]) * (1 - c["h2"])])
            y.append(TAU[(v, n)])
        return np.array(X), np.array(y)
    X, y = cols(n_fit)
    w = 1.0 / y
    b = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    out = []
    for n in SCALES:
        if n == n_fit:
            continue
        Xn, yn = cols(n)
        out.append(100 * np.abs(Xn @ b - yn) / yn)
    return np.concatenate(out).mean()


rows = [
    ("full model, Eq. (3) with (6')", heldout(FIT_AT).mean()),
    ("plus a constant issue term, (I + nu L)/e", issue_term_model(FIT_AT)),
    ("without the divergence factor e", heldout(FIT_AT, no_e=True).mean()),
    ("with the hit rates frozen at the fit scale", heldout(FIT_AT, freeze=FIT_AT).mean()),
    ("null model (mean tau of the fit scale)", null.mean()),
]
print(f"{'model':<45}{'held-out MAPE':>15}")
print("-" * 60)
for name, m in rows:
    print(f"{name:<45}{m:>14.1f}%")

# ------------------------------------------------------------------ part 4
print("\n\nPART 4 — does the model select the right algorithm?")
print("The deployable question is the ranking, and above all whether the\n"
      "A6 -> A6b crossover between 20 M and 50 M tokens is predicted.\n")
print(f"{'scale':>8}  {'measured best':<24}{'predicted best':<24}{'rank rho':>9}")
print("-" * 68)
for n in SCALES:
    err, y, p, keep = score(beta, n)
    bm, bp = keep[int(np.argmin(y))], keep[int(np.argmin(p))]
    rho = np.corrcoef(np.argsort(np.argsort(y)), np.argsort(np.argsort(p)))[0, 1]
    print(f"{n//10**6:>6} M  {bm:<24}{bp:<24}{rho:>9.2f}"
          + ("" if bm == bp else "   <-- MISS"))

# ------------------------------------------------------------------ part 5
print("\n\nPART 5 — the operationalization matters, and this is the evidence")
print("Same model, same data, nu read as sectors per REQUEST instead of\n"
      "transactions per TOKEN — the substitution the first attempt made.\n")
for n in SCALES:
    for v in ORDER:
        C[n][v]["nu_true"], C[n][v]["nu"] = C[n][v]["nu"], C[n][v]["sreq"]
print(f"{'fitted at':>10}{'L1':>9}{'L2':>9}{'L_DRAM':>9}   {'held-out MAPE':>14}")
print(BAR)
for nf in SCALES:
    b, h = fit(nf), heldout(nf)
    print(f"{nf//10**6:>8} M{b[0]:>9.4f}{b[1]:>9.4f}{b[2]:>9.4f}   {h.mean():>13.1f}%")
print(BAR)
print("Negative latencies and errors at or above the null model. A counter\n"
      "that is merely correlated with the modelled quantity is not a\n"
      "measurement of it.")

print("""

VERDICT

Proposition 2 is confirmed as a measurement.  Across a fiftyfold change in
corpus size nu varies by 3.7 % and e by 0.4 %, while the L2 hit rate moves by
4.8 % and -- the part that matters -- moves in OPPOSITE directions for the
permuted and the compacted variants.  Two factors carry the precision
dependence and one carries the scale dependence, which is exactly what the
proposition asserts and is what allows the optimum to shift with corpus size
while the per-variant terms are measured only once.

Equation (3) as a quantitative predictor is NOT confirmed.  Fitted at one
corpus size it predicts held-out per-token times to roughly a third in
relative error.  That is far better than the null model and every factor earns
its place under ablation, so the decomposition is real; but it does not
resolve the A6 -> A6b crossover between 20 M and 50 M tokens, which is the one
prediction with an operational consequence.  The model should be reported as a
diagnostic decomposition with measured separability, not as a calculator, and
Chapter 3 must say so.

Two reasons the residual is structural rather than noise, both stated in the
chapter's own data:

  1. No memory-level-parallelism term.  Little's law gives tau = nu L / MLP,
     and MLP is variant-dependent -- uniform warps issue more independent
     requests -- but nothing here measures it.
  2. Two regimes, one functional form.  Section 2.1 measures the baseline at
     0.98 core-clock sensitivity (issue-bound) and every ordered variant at
     0.42 or below (memory-bound).  A single multiplicative memory expression
     cannot span both, and the natural repair -- an additive issue term -- is
     unidentifiable from seven observations per scale (PART 3).

The experiment that would settle it is cheap and is not a better fit: profile
the same seven variants at the four scales already swept but not profiled
(2 M, 5 M, 10 M, 35 M).  That gives 49 observations against four parameters,
enough to test the issue-term model honestly and to fit MLP if a counter for
it is added to the metric list.
""")
