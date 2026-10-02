# -- added for the public repository: absolute paths of the development machine are replaced by
# -- paths relative to the repository root, through the helper _repo_path defined here.
import os as _os


def _repo_path(rel=""):
    d = _os.path.dirname(_os.path.abspath(__file__))
    while not (_os.path.isdir(_os.path.join(d, "research_benchmark")) and _os.path.isdir(_os.path.join(d, "research_highdim"))):
        p = _os.path.dirname(d)
        if p == d:
            raise RuntimeError("repository root not found")
        d = p
    return _os.path.join(d, rel) if rel else d
# -- end of the added lines

import json, numpy as np, glob
from scipy import stats
R = {}
for f in ("v_runs.json", "v_runs_extra.json", "v_runs_budget.json"):
    R.update(json.load(open(f)))
# study's values for the same (problem, sampler, seed)
T = {}
for f in glob.glob(_repo_path("research_benchmark/results/runs/*_s*-*.json")):
    for r in json.load(open(f))["runs"]:
        T[(r["problem"], r["sampler"], r["seed"])] = (r["adam"]["rel_l2"], r["adam_lbfgs"]["rel_l2"])
def g(p, k, s, n=256, mode="fixed", arm="adam_lbfgs"): return R[f"{p}|{k}|{s}|{n}|{mode}"][arm]["rel_l2"]
def geo(v): return float(np.exp(np.mean(np.log(v))))
print("re-check vs study, same (problem, strategy, seed): Adam | Adam->L-BFGS")
for key, r in R.items():
    p, k, s, n, mode = key.split("|"); s = int(s)
    tk = (p, "resample" if mode == "resample" else k, s)
    t = T.get(tk) if n in ("256", "512", "1024") and not (p == "heat1d" and n == "1024") else None
    print(f"  {key:38s} recheck {r['adam']['rel_l2']:.3e} | {r['adam_lbfgs']['rel_l2']:.3e}   study " + (f"{t[0]:.3e} | {t[1]:.3e}" if t else "n/a") + f"   LBFGS<Adam: {r['adam_lbfgs']['rel_l2'] < r['adam']['rel_l2']}  t_arm={r['adam_lbfgs']['time']:.0f}s")
nb = sum(r["adam_lbfgs"]["rel_l2"] < r["adam"]["rel_l2"] for r in R.values()); print("L-BFGS arm better than Adam arm in", nb, "of", len(R), "of the re-check runs")
# paired log-ratio between the L-BFGS results of the re-check and the study's for identical configs
d = [np.log(r["adam_lbfgs"]["rel_l2"] / T[(k.split("|")[0], k.split("|")[1], int(k.split("|")[2]))][1]) for k, r in R.items()
     if k.split("|")[4] == "fixed" and k.split("|")[1] != "nodegrid" and (k.split("|")[0], k.split("|")[1], int(k.split("|")[2])) in T and not (k.startswith("heat1d") and "|1024|" in k)]
print("re-check/study Adam->L-BFGS error, identical configs: n=%d geo ratio %.2f, range %.2f-%.2f" % (len(d), np.exp(np.mean(d)), np.exp(min(d)), np.exp(max(d))))
print("heat1d N=256, Adam->L-BFGS")
S = range(6)
G = np.array([g("heat1d", "grid", s) for s in S]); Rn = np.array([g("heat1d", "random", s) for s in S]); N = np.array([g("heat1d", "nodegrid", s) for s in S])
for nm, a, b in (("cell-grid/random", G, Rn), ("node-grid/random", N, Rn), ("node-grid/cell-grid", N, G)):
    d = np.log(a / b); print(f"  {nm:20s} geo ratio {np.exp(d.mean()):.2f} per-seed {np.round(np.exp(d), 2).tolist()} better {int((d<0).sum())}/6 wilcoxon p={stats.wilcoxon(d).pvalue:.3f}")
print("  means: cell-grid %.2e +- %.2e, random %.2e +- %.2e, node-grid %.2e +- %.2e" % (G.mean(), G.std(ddof=1), Rn.mean(), Rn.std(ddof=1), N.mean(), N.std(ddof=1)))
So = np.array([g("heat1d", "sobol", s) for s in range(3)]); print("  sobol/random seeds0-2", np.round(So / Rn[:3], 2), "geo", round(geo(So / Rn[:3]), 2))
print("laplace2d N=256")
G = np.array([g("laplace2d", "grid", s) for s in range(3)]); Rn = np.array([g("laplace2d", "random", s) for s in range(3)]); So = np.array([g("laplace2d", "sobol", s) for s in range(3)])
print("  grid", G, "random", Rn, "sobol", So)
print("  grid/random", np.round(G / Rn, 2), "geo", round(geo(G / Rn), 2), "| sobol/random", np.round(So / Rn, 2), "geo", round(geo(So / Rn), 2))
A = np.array([g("laplace2d", "random", s, arm="adam") for s in range(3)]); Ar = np.array([g("laplace2d", "random", s, mode="resample", arm="adam") for s in range(3)])
print("  Adam arm: resample/fixed-random", np.round(Ar / A, 2), "geo", round(geo(Ar / A), 2))
print("heat N=1024:", {k: round(v["adam_lbfgs"]["rel_l2"], 5) for k, v in R.items() if "|1024|" in k and k.startswith("heat")})
