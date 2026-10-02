"""Section 4 tables and every number quoted in the prose, recomputed from the raw result files.

No training.  Inputs (relative to the root of the repository):
  research_benchmark/results/benchmark_runs.csv      350 rows, one per (problem, strategy, arm, seed)
  research_benchmark/results/budget_sweep.json       heat problem, N_r = 64 / 256 / 1024, 5 seeds
  research_benchmark/results/runs/*_s*-*.json        training curves (Adam last-iterate fluctuation)
  verify_research_benchmark/v_runs.json, v_runs_extra.json, v_runs_budget.json, v_adam_decay.json
                                                     separately written single-model re-implementation (V-bench)
Outputs:
  data/s4_lowdim_numbers.json                all derived numbers
  data/s4_lowdim_tables.tex                  LaTeX rows of Tables s4_lowdim:tab:main and :tab:strategy
                                                     (pasted into sections/s4_lowdim.tex)
"""
import csv
import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
ROOT = ARTICLE if os.path.isdir(os.path.join(ARTICLE, "research_benchmark")) else os.path.dirname(ARTICLE)  # repository root
B = os.path.join(ROOT, "research_benchmark", "results")
V = os.path.join(ROOT, "verify_research_benchmark")

PROBLEMS = ["heat1d", "laplace2d", "wave1d", "wave2d", "laplace3d"]
PMAC = {"heat1d": r"\PH", "laplace2d": r"\PLtwo", "wave1d": r"\PWone", "wave2d": r"\PWtwo", "laplace3d": r"\PLthree"}
STRATS = ["grid", "random", "resample", "sobol", "rad"]
ARMS = ["adam", "adam_lbfgs"]
AMAC = {"adam": r"\Adam", "adam_lbfgs": r"\AdamLBFGS"}

rows = list(csv.DictReader(open(os.path.join(B, "benchmark_runs.csv"))))
for r in rows:
    r["seed"] = int(r["seed"]); r["rel_l2"] = float(r["rel_l2"]); r["linf"] = float(r["linf"])


def sel(p, s, a, key="rel_l2"):
    rr = sorted((r for r in rows if r["problem"] == p and r["sampler"] == s and r["arm"] == a), key=lambda r: r["seed"])
    return np.array([r[key] for r in rr]), [r["seed"] for r in rr]


def geomean(v):
    return float(10 ** np.log10(np.asarray(v, float)).mean())


def paired(num, den):
    """Geometric-mean ratio num/den, number of pairs with num < den, exact two-sided Wilcoxon p."""
    d = np.log10(np.asarray(num, float)) - np.log10(np.asarray(den, float))
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p = None
    return {"ratio": float(10 ** d.mean()), "n_better": int((d < 0).sum()), "n": int(len(d)), "p": p,
            "per_seed_ratio": [float(x) for x in 10 ** d]}


N = {"n_rows": len(rows)}
tex = []

# ------------------------------------------------------------------ Table main (T1, rel-L2) ---
N["main"] = {}
n_r = {}
for f in sorted(glob.glob(os.path.join(B, "runs", "*_s*-*.json"))):
    d = json.load(open(f))
    n_r[d["runs"][0]["problem"]] = d["runs"][0]["n_r"]
tex.append("% ---- Table s4_lowdim:tab:main (mean +- s.d. of the relative L2 error, in units of 10^e) ----")
for p in PROBLEMS:
    for a in ARMS:
        stats_ = {}
        for s in STRATS:
            v, seeds = sel(p, s, a)
            stats_[s] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "geo_mean": geomean(v),
                         "min": float(v.min()), "max": float(v.max()), "n": len(v)}
        N["main"][f"{p}:{a}"] = stats_
        e = int(np.floor(np.log10(min(x["mean"] for x in stats_.values()))))
        cells = " & ".join(f"${stats_[s]['mean'] / 10 ** e:.2f}\\pm{stats_[s]['std'] / 10 ** e:.2f}$" for s in STRATS)
        head = f"{PMAC[p]} & {n_r[p]} & {stats_['grid']['n']}" if a == "adam" else " & & "
        end = "[0.3em]" if (a == "adam_lbfgs" and p != PROBLEMS[-1]) else ""
        tex.append(f"{head} & {AMAC[a]} & $10^{{{e}}}$ & {cells} \\\\{end}")
MAIN_ROWS = [t for t in tex if not t.startswith("%")]
N["n_r"] = n_r

# ------------------------------------------------------------------ optimiser (T2) -----------
N["optimiser"] = {}
tot_better = tot = 0
for p in PROBLEMS:
    va = np.concatenate([sel(p, s, "adam")[0] for s in STRATS])
    vb = np.concatenate([sel(p, s, "adam_lbfgs")[0] for s in STRATS])
    seeds = sel(p, "grid", "adam")[1]
    # per-seed aggregation: geometric mean over the five strategies of one seed (they share the
    # initial network), then an exact Wilcoxon signed-rank test over seeds
    ga = np.array([geomean([sel(p, s, "adam")[0][i] for s in STRATS]) for i in range(len(seeds))])
    gb = np.array([geomean([sel(p, s, "adam_lbfgs")[0][i] for s in STRATS]) for i in range(len(seeds))])
    ps = paired(gb, ga)
    N["optimiser"][p] = {"geo_adam": geomean(va), "geo_adam_lbfgs": geomean(vb), "ratio": geomean(vb) / geomean(va),
                         "pairs_better": int((vb < va).sum()), "pairs": int(len(va)),
                         "per_seed_better": ps["n_better"], "n_seeds": ps["n"], "per_seed_wilcoxon_p": ps["p"],
                         "adam_min": float(va.min()), "adam_max": float(va.max()),
                         "adam_lbfgs_min": float(vb.min()), "adam_lbfgs_max": float(vb.max()),
                         "closest_pair_ratio_lbfgs_over_adam": float((vb / va).max())}
    tot_better += int((vb < va).sum()); tot += len(va)
N["optimiser"]["all"] = {"pairs_better": tot_better, "pairs": tot}

# Adam last-iterate fluctuation between iterations 1500 and 3000 (benchmark training curves;
# these are errors on the 4096-point subset of the held-out grid used for the curves)
fl = {}
for f in sorted(glob.glob(os.path.join(B, "runs", "*_s*-*.json"))):
    for r in json.load(open(f))["runs"]:
        c = [x["test_rel_l2"] for x in r["adam"]["curve"] if x["it"] >= 1500]
        fl.setdefault(r["problem"], []).append(max(c) / min(c))
N["adam_fluctuation_max_over_min_it1500_3000"] = {
    p: {"median": float(np.median(v)), "max": float(np.max(v)), "n_ge_10": int((np.array(v) >= 10).sum()), "n": len(v)}
    for p, v in fl.items()}

# L-BFGS end point against the last, the median and the best logged Adam iterate of the same run
# (iterations 1500 to 3000; all on the 4096-point subset of the held-out grid on which the
# curves are logged).  The "best logged iterate" is an oracle checkpoint: it uses the exact
# solution to choose the iterate and is not available in practice.
bi = {}
for f in sorted(glob.glob(os.path.join(B, "runs", "*_s*-*.json"))):
    for r in json.load(open(f))["runs"]:
        c = np.array([x["test_rel_l2"] for x in r["adam"]["curve"] if x["it"] >= 1500])
        lb = r["adam_lbfgs"]["curve"][-1]["test_rel_l2"]
        bi.setdefault(r["problem"], []).append((lb / c[-1], lb / np.median(c), lb / c.min(), lb < c.min()))
N["lbfgs_vs_logged_adam_iterates_it1500_3000"] = {}
for p, v in bi.items():
    v = np.array(v, float)
    N["lbfgs_vs_logged_adam_iterates_it1500_3000"][p] = {
        "n": int(len(v)), "geo_ratio_to_last": geomean(v[:, 0]), "geo_ratio_to_median": geomean(v[:, 1]),
        "geo_ratio_to_best": geomean(v[:, 2]), "ratio_to_best_min": float(v[:, 2].min()),
        "ratio_to_best_max": float(v[:, 2].max()), "n_lbfgs_below_best_adam": int(v[:, 3].sum())}

# ------------------------------------------------------------------ strategy (T3) ------------
N["strategy"] = {}
tex.append("% ---- Table s4_lowdim:tab:strategy (ratio to random; seeds better / n; Wilcoxon p) ----")


def cell(q, with_p=True):
    s = f"{q['ratio']:.2f} ({q['n_better']}/{q['n']}"
    if with_p and q["p"] is not None:
        s += "; 0.0625" if abs(q["p"] - 0.0625) < 1e-12 else (f"; {q['p']:.3f}" if q["p"] >= 0.001 else f"; {q['p']:.1e}")
    return s + ")"


for a, plist in [("adam_lbfgs", PROBLEMS), ("adam", ["heat1d", "laplace2d"])]:
    for p in plist:
        base, seeds = sel(p, "random", a)
        out = {}
        for s in ["grid", "resample", "sobol", "rad"]:
            v, sd = sel(p, s, a)
            assert sd == seeds
            out[s] = paired(v, base)
        out["kruskal_p"] = float(stats.kruskal(*[np.log10(sel(p, s, a)[0]) for s in STRATS]).pvalue)
        N["strategy"][f"{p}:{a}"] = out
        tex.append(f"{PMAC[p]} & {AMAC[a]} & " + " & ".join(cell(out[s]) for s in ["grid", "resample", "sobol", "rad"]) + r" \\"
                   + ("[0.2em]" if (a, p) == ("adam_lbfgs", PROBLEMS[-1]) else ""))
# Sobol' relative to the cell-centred grid (for completeness)
N["sobol_over_grid_adam_lbfgs"] = {p: paired(sel(p, "sobol", "adam_lbfgs")[0], sel(p, "grid", "adam_lbfgs")[0]) for p in PROBLEMS}

# ------------------------------------------------------------------ separately written re-implementation (V-bench)
vr = {}
for f in ["v_runs.json", "v_runs_extra.json", "v_runs_budget.json"]:
    vr.update(json.load(open(os.path.join(V, f))))


def vget(p, s, seeds, arm="adam_lbfgs", n=256, mode="fixed"):
    return np.array([vr[f"{p}|{s}|{k}|{n}|{mode}"][arm]["rel_l2"] for k in seeds])


R = {}
s6 = range(6)
R["heat_grid_c_over_random_6seeds"] = paired(vget("heat1d", "grid", s6), vget("heat1d", "random", s6))
R["heat_grid_n_over_random_6seeds"] = paired(vget("heat1d", "nodegrid", s6), vget("heat1d", "random", s6))
R["heat_grid_n_over_grid_c_6seeds"] = paired(vget("heat1d", "nodegrid", s6), vget("heat1d", "grid", s6))
R["heat_means_6seeds"] = {s: {"mean": float(vget("heat1d", s, s6).mean()), "std": float(vget("heat1d", s, s6).std(ddof=1)),
                              "values": vget("heat1d", s, s6).tolist()} for s in ["grid", "random", "nodegrid"]}
s3 = range(3)
R["heat_sobol_over_random_3seeds"] = paired(vget("heat1d", "sobol", s3), vget("heat1d", "random", s3))
R["lap2d_grid_c_over_random_3seeds"] = paired(vget("laplace2d", "grid", s3), vget("laplace2d", "random", s3))
R["lap2d_sobol_over_random_3seeds"] = paired(vget("laplace2d", "sobol", s3), vget("laplace2d", "random", s3))
R["lap2d_adam_resample_over_random_3seeds"] = paired(vget("laplace2d", "random", s3, arm="adam", mode="resample"),
                                                     vget("laplace2d", "random", s3, arm="adam"))
R["lap2d_lbfgs_resample_over_random_3seeds"] = paired(vget("laplace2d", "random", s3, mode="resample"),
                                                      vget("laplace2d", "random", s3))
R["heat_1024"] = {k: vr[k]["adam_lbfgs"]["rel_l2"] for k in vr if k.startswith("heat1d") and "|1024|" in k}
# optimiser arm comparison in the re-implementation
better = [(k, vr[k]["adam"]["rel_l2"], vr[k]["adam_lbfgs"]["rel_l2"]) for k in vr]
R["lbfgs_better"] = {"n_better": sum(b < a for _, a, b in better), "n": len(better),
                     "exceptions": [{"run": k, "adam": a, "adam_lbfgs": b} for k, a, b in better if not b < a]}
# agreement with the benchmark on identical configurations
ratios = []
for k in vr:
    p, s, seed, n, mode = k.split("|")
    s_b = "resample" if mode == "resample" else s
    if s == "nodegrid" or mode != "fixed" or int(n) != N["n_r"][p]:
        continue
    v, sd = sel(p, s_b, "adam_lbfgs")
    if int(seed) in sd:
        ratios.append(vr[k]["adam_lbfgs"]["rel_l2"] / v[sd.index(int(seed))])
R["agreement_identical_configs"] = {"n": len(ratios), "geo_ratio": geomean(ratios), "min": float(min(ratios)), "max": float(max(ratios))}
# Adam end point versus Adam at the branch point (noisy last iterate)
R["adam_it1500_vs_it3000_examples"] = {k: {"it1500": vr[k]["adam_at_branch"]["rel_l2"], "it3000": vr[k]["adam"]["rel_l2"]}
                                       for k in ["heat1d|random|1|256|fixed", "laplace2d|grid|0|256|fixed"]}
dec = json.load(open(os.path.join(V, "v_adam_decay.json")))
R["adam_exponential_decay"] = {k: v["rel_l2"] for k, v in dec.items()}
R["adam_constant_lr_same_runs"] = {k: vr[k + "|256|fixed"]["adam"]["rel_l2"] for k in dec}
R["adam_decay_over_constant_same_runs"] = {k: R["adam_exponential_decay"][k] / R["adam_constant_lr_same_runs"][k] for k in dec}
N["replication"] = R
i0 = tex.index(next(t for t in tex if "tab:strategy" in t)) + 1
STRAT_ROWS = [r"\multicolumn{6}{@{}l}{\emph{Benchmark (10 seeds for \PH, \PLtwo; 5 seeds for \PWone, \PWtwo, \PLthree)}}\\"] + tex[i0:]
tex.append("% ---- rows of the verification code V-bench (6 or 3 seeds; a dash means not run) ----")
V_ROWS = [
    r"\PH & \AdamLBFGS & " + cell(R["heat_grid_c_over_random_6seeds"]) + " & -- & " + cell(R["heat_sobol_over_random_3seeds"], False) + r" & -- \\",
    r"\PH, \strat{grid-n} & \AdamLBFGS & " + cell(R["heat_grid_n_over_random_6seeds"]) + r" & -- & -- & -- \\",
    r"\PLtwo & \AdamLBFGS & " + cell(R["lap2d_grid_c_over_random_3seeds"], False) + " & " + cell(R["lap2d_lbfgs_resample_over_random_3seeds"], False)
    + " & " + cell(R["lap2d_sobol_over_random_3seeds"], False) + r" & -- \\",
    r"\PLtwo & \Adam & -- & " + cell(R["lap2d_adam_resample_over_random_3seeds"], False) + r" & -- & -- \\"]
tex += V_ROWS
STRAT_ROWS += [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{Verification code \textsf{V-bench} (6 or 3 seeds)}}\\"] + V_ROWS

# ------------------------------------------------------------------ budget sweep (T6) --------
bs = [r for r in json.load(open(os.path.join(B, "budget_sweep.json"))) if r["problem"] == "heat1d"]
N["budget"] = {}
tex.append("% ---- Table s4_lowdim:tab:budget (heat, Adam->L-BFGS, seeds 0-4): mean +- s.d.; ratio to random ----")


def sci(x, nd=2):
    m, e = f"{x:.{nd}e}".split("e")
    return f"\\sci{{{m}}}{{{int(e)}}}"
for n in sorted({r["n_r"] for r in bs}):
    def bv(s):
        rr = sorted((r for r in bs if r["n_r"] == n and r["sampler"] == s), key=lambda r: r["seed"])
        return np.array([r["adam_lbfgs"]["rel_l2"] for r in rr])
    e = {s: {"mean": float(bv(s).mean()), "std": float(bv(s).std(ddof=1)), "geo_mean": geomean(bv(s)), "n": len(bv(s)),
             "min": float(bv(s).min()), "max": float(bv(s).max())} for s in ["grid", "random", "sobol"]}
    e["grid_over_random"] = paired(bv("grid"), bv("random"))
    e["sobol_over_random"] = paired(bv("sobol"), bv("random"))
    N["budget"][str(n)] = e
    def pm(mu, sd):
        ex = int(np.floor(np.log10(mu)))
        return f"$({mu / 10 ** ex:.2f}\\pm{sd / 10 ** ex:.2f})\\times10^{{{ex}}}$"
    tex.append(f"{n} & " + " & ".join(pm(e[s]['mean'], e[s]['std']) for s in ["grid", "random", "sobol"])
               + f" & {cell(e['grid_over_random'])} & {cell(e['sobol_over_random'])} \\\\")

import sys
sys.path.insert(0, HERE)
from inject import inject
inject("s4_lowdim.tex", "s4_lowdim:tab:main", "\n".join(MAIN_ROWS))
inject("s4_lowdim.tex", "s4_lowdim:tab:strategy", "\n".join(STRAT_ROWS))
inject("s4_lowdim.tex", "s4_lowdim:tab:budget", "\n".join(tex[tex.index(next(t for t in tex if "tab:budget" in t)) + 1:]))
json.dump(N, open(os.path.join(ARTICLE, "data", "s4_lowdim_numbers.json"), "w"), indent=1)
open(os.path.join(ARTICLE, "data", "s4_lowdim_tables.tex"), "w").write("\n".join(tex) + "\n")
print("\n".join(tex))
print()
for p in PROBLEMS:
    print(p, {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in N["optimiser"][p].items()})
print("all", N["optimiser"]["all"])
print("adam fluctuation", N["adam_fluctuation_max_over_min_it1500_3000"])
for k, v in R.items():
    print(k, v)
print("sobol/grid", {p: cell(v) for p, v in N["sobol_over_grid_adam_lbfgs"].items()})
print("kruskal", {k: f"{v['kruskal_p']:.3g}" for k, v in N["strategy"].items()})
print("heat adam_lbfgs sobol per-seed ratio", [f"{x:.2f}" for x in N["strategy"]["heat1d:adam_lbfgs"]["sobol"]["per_seed_ratio"]])
