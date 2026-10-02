"""Section 5: the rows of Table s5_hard:tab:hard (harder problems) and of Table s5_hard:tab:fd
(PINN against finite differences), generated from the stored results.  No training, no timing.

Inputs (relative to the root of the repository)
  research_benchmark/results/benchmark_runs.csv     benchmark rows (25 runs per problem)
  data/s5_hard_wave2d_long.json                     Section 5.1 (longer runs on the 2-D wave problem)
  data/s5_hard_laplace3d_smooth.json                Section 5.2 (compatible data, and the control)
  data/s5_hard_wave1d_long.json                     Section 5.3 (41 000 iterations on the two-mode wave problem)
  data/s5_hard_cost.json                            written by code/s5_hard_cost.py (run it first)
  verify_research_benchmark/v_fd.out                re-derivation of the finite-difference errors by the verification
                                                    (printed in the output of this script; quoted in the text of Section 5.4)
Outputs
  data/s5_hard_tables.tex                           both row blocks
  sections/s5_hard.tex                              the blocks between "%% BEGIN GENERATED s5_hard:tab:hard" /
                                                    "%% BEGIN GENERATED s5_hard:tab:fd" and the matching END
                                                    lines are replaced (if the file exists)
Usage:  python code/s5_hard_tables.py            write
        python code/s5_hard_tables.py --check    exit with an error if the section differs from the stored results
"""
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from inject import inject

ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)  # repository root
DATA = os.path.join(ART, "data")
STRATS = ["grid", "random", "resample", "sobol", "rad"]
SMAC = {"grid-c": r"\strat{grid-c}", "random": r"\strat{random}", "resample": r"\strat{resample}",
        "Sobol": r"\strat{Sobol}", "RAD": r"\strat{RAD}"}
PMAC = {"heat1d": r"\PH", "laplace2d": r"\PLtwo", "wave1d": r"\PWone", "wave2d": r"\PWtwo", "laplace3d": r"\PLthree"}


def sci(x, nd=2):
    m, e = f"{x:.{nd}e}".split("e")
    return f"\\sci{{{m}}}{{{int(e)}}}"


def pair(lo, hi, sep):
    """Two numbers in one cell: plain decimals if both are at least 0.05 (to rounding), else both scientific."""
    if lo >= 0.0495:
        return f"{lo:.3f}", f"{hi:.3f}"
    return sci(lo), sci(hi)


def dash(lo, hi):
    a, b = pair(lo, hi, "--")
    return f"${a}$--${b}$"


def brack(lo, hi):
    a, b = pair(lo, hi, ",")
    return f"$[{a},\\,{b}]$"


def pm(mean, sd):
    """(a +- b) x 10^e with the exponent of the mean; plain decimals for means of at least 0.1."""
    if mean >= 0.1:
        return f"${mean:.3f}\\pm {sd:.3f}$"
    e = int(np.floor(np.log10(mean)))
    return f"$({mean / 10 ** e:.2f}\\pm {sd / 10 ** e:.2f})\\times 10^{{{e}}}$"


rows = list(csv.DictReader(open(os.path.join(ROOT, "research_benchmark", "results", "benchmark_runs.csv"))))


def bench(p, arm, key):
    v = {s: np.array([float(r[key]) for r in rows if r["problem"] == p and r["sampler"] == s and r["arm"] == arm]) for s in STRATS}
    means = [v[s].mean() for s in STRATS]
    allv = np.concatenate(list(v.values()))
    return min(means), max(means), allv.min(), allv.max(), len(allv)


def bench_rows(p, label):
    a = bench(p, "adam", "rel_l2"); b = bench(p, "adam_lbfgs", "rel_l2"); c = bench(p, "adam_lbfgs", "linf")
    assert a[4] == 25
    r1 = f"{label}, benchmark & {dash(a[0], a[1])} & {dash(b[0], b[1])} & {dash(c[0], c[1])} \\\\"
    r2 = f"\\quad 25 runs & {brack(a[2], a[3])} & {brack(b[2], b[3])} & {brack(c[2], c[3])} \\\\[0.3em]"
    return [r1, r2]


def seeds_rows(label, n, st):
    """st: dict with (mean, s.d., min, max) of the Adam rel-L2, Adam -> L-BFGS rel-L2 and Adam -> L-BFGS L-infinity errors."""
    r1 = f"{label} & {pm(*st['a'][:2])} & {pm(*st['b'][:2])} & {pm(*st['c'][:2])} \\\\"
    r2 = (f"\\quad {n} seeds & {brack(st['a'][2], st['a'][3])} & {brack(st['b'][2], st['b'][3])} & "
          f"{brack(st['c'][2], st['c'][3])} \\\\[0.3em]")
    return [r1, r2]


def ms(v):
    v = np.asarray(v, float)
    return (float(v.mean()), float(v.std(ddof=1)), float(v.min()), float(v.max()))


# ------------------------------------------------------------------ Table hard
hard = []
hard += bench_rows("wave2d", r"\PWtwo")
w = json.load(open(os.path.join(DATA, "s5_hard_wave2d_long.json")))["summary"]
ps = w["per_seed"]
hard += seeds_rows(r"\PWtwo, long run", len(ps),
                   {"a": ms([x["adam_rel_l2"] for x in ps]), "b": ms([x["adam_lbfgs_rel_l2"] for x in ps]),
                    "c": ms([x["adam_lbfgs_linf"] for x in ps])})
hard += bench_rows("laplace3d", r"\PLthree")
l3 = json.load(open(os.path.join(DATA, "s5_hard_laplace3d_smooth.json")))


def n2(o):
    return {k: (o[f"{q}_mean"], o[f"{q}_std"], o[f"{q}_min"], o[f"{q}_max"])
            for k, q in (("a", "adam_rel_l2"), ("b", "adam_lbfgs_rel_l2"), ("c", "adam_lbfgs_linf"))}


hard += seeds_rows(r"\PLthree, control", len(l3["L3"]["rows"]), n2(l3["L3"]))
hard += seeds_rows(r"\PLthreeS", len(l3["L3s"]["rows"]), n2(l3["L3s"]))
hard += bench_rows("wave1d", r"\PWone")
w1 = json.load(open(os.path.join(DATA, "s5_hard_wave1d_long.json")))["summary"]
ps1 = w1["per_seed"]
hard += seeds_rows(r"\PWone, long run", len(ps1),
                   {"a": ms([x["adam_rel_l2"] for x in ps1]), "b": ms([x["adam_lbfgs_rel_l2"] for x in ps1]),
                    "c": ms([x["adam_lbfgs_linf"] for x in ps1])})
hard[-1] = hard[-1].replace("\\\\[0.3em]", "\\\\")
long_iters = sorted(x["lbfgs_iters"] for x in ps)
long_iters_w1 = sorted(x["lbfgs_iters"] for x in ps1)

# ------------------------------------------------------------------ Table fd
cost = json.load(open(os.path.join(DATA, "s5_hard_cost.json")))["problems"]
vfd = open(os.path.join(ROOT, "verify_research_benchmark", "v_fd.out")).read()


def rederived(p, n):
    """Relative L2 error of the n-cell grid as re-derived by the verification (None if absent)."""
    head = {"heat1d": "heat1d", "wave2d": "wave2d", "laplace2d": "laplace2d", "laplace3d": "laplace3d"}.get(p)
    if head is None:
        return None
    block = vfd[vfd.index("\n" + head) if not vfd.startswith(head) else 0:]
    block = block.split("\n", 2)[2] if block.startswith("\n") else block.split("\n", 1)[1]
    for line in block.splitlines():
        if not line.startswith("  "):
            break
        m = re.match(rf"\s+n={n} .*?rel-L2 ([0-9.e+-]+)", line)
        if m and "face values = 0" not in line:
            return m.group(1)
    return None


def rng(lo, hi, f):
    return f(lo) if abs(hi - lo) < 1e-9 * hi else f"{f(lo)}--{f(hi)}"


def ratio(lo, hi):
    if abs(hi - lo) < 1e-9 * hi:
        return f"$\\sci{{{f'{hi:.1e}'.split('e')[0]}}}{{{int(f'{hi:.1e}'.split('e')[1])}}}$"
    e = int(np.floor(np.log10(hi)))
    return f"$({lo / 10 ** e:.1f}$--${hi / 10 ** e:.1f})\\times 10^{{{e}}}$"


fd, notes = [], {}
for p in ["heat1d", "laplace2d", "wave1d", "wave2d", "laplace3d"]:
    c = cost[p]
    f = c["fd_cheapest_at_least_as_accurate"]
    B = c["time_B_verifier_s"]["adam_lbfgs"]
    R = c["time_ratio_pinn_over_fd"]
    best = c["adam_lbfgs"]["best_strategy_geomean"]
    bs = f"${best:.3f}$" if best >= 0.1 else f"${sci(best)}$"
    rd = rederived(p, f["n"])
    if rd is None:
        rd_tex = "---$^{b}$"
    else:
        m, e = rd.split("e")
        rd_tex = f"$\\sci{{{m}}}{{{int(e)}}}$" + ("\\,$^{a}$" if p == "heat1d" else "")
    Cm = c["time_C_stacked_gpu_per_network_s"]["adam_lbfgs"]
    fd.append(f"{PMAC[p]} & {bs} ({SMAC[c['adam_lbfgs']['best_strategy']]}) & {f['n']} & ${sci(f['rel_l2'])}$ & ${sci(f['linf'])}$ & "
              f"${sci(f['time_s'], 1)}$ & {c['time_A_reference_driver_s']['adam_lbfgs']:.0f} & "
              f"{rng(B['min'], B['max'], lambda t: f'{t:.0f}')} & "
              f"{rng(Cm['min'], Cm['max'], lambda t: f'{t:.0f}')} & "
              f"{ratio(R['B_verifier_min'], R['B_verifier_max'])} & {ratio(R['C_stacked_min'], R['C_stacked_max'])} \\\\")
    notes[p] = {"n_runs_B": B["n_runs"], "rederived_rel_l2": rd}

tex = {"s5_hard:tab:hard": "\n".join(hard), "s5_hard:tab:fd": "\n".join(fd)}
info = {"wave2d_long_lbfgs_iters_min_max": [long_iters[0], long_iters[-1]],
        "wave1d_long_lbfgs_iters_min_max": [long_iters_w1[0], long_iters_w1[-1]], "n_runs_B": notes}
if "--check" in sys.argv:
    bad = [k for k, v in tex.items() if inject("s5_hard.tex", k, v, check=True) is False]
    if bad:
        sys.exit("sections/s5_hard.tex differs from the stored results in: " + ", ".join(bad))
    print("sections/s5_hard.tex: Tables s5_hard:tab:hard and s5_hard:tab:fd agree with the stored results")
else:
    with open(os.path.join(DATA, "s5_hard_tables.tex"), "w") as fh:
        for k, v in tex.items():
            fh.write(f"%% ---- {k} (generated by code/s5_hard_tables.py) ----\n{v}\n\n")
    for k, v in tex.items():
        inject("s5_hard.tex", k, v)
    print("\n\n".join(f"% {k}\n{v}" for k, v in tex.items()))
    print(json.dumps(info))
