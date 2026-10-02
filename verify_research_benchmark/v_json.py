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

import json, numpy as np
from scipy import stats
R = _repo_path("research_benchmark/results")
print("== budget sweep")
rows = json.load(open(R + "/budget_sweep.json"))
for n in sorted(set(r["n_r"] for r in rows)):
    for s in ("grid", "random", "sobol"):
        v = np.array([r["adam_lbfgs"]["rel_l2"] for r in rows if r["n_r"] == n and r["sampler"] == s])
        seeds = [r["seed"] for r in rows if r["n_r"] == n and r["sampler"] == s]
        print(f" N={n} {s:7s} n={len(v)} seeds={seeds} mean {v.mean():.3e} +- {v.std(ddof=1):.3e}  vals {np.array2string(v, precision=4)}")
    # paired tests at this N
    def g(s): return np.array([r["adam_lbfgs"]["rel_l2"] for r in sorted([r for r in rows if r["n_r"] == n and r["sampler"] == s], key=lambda r: r["seed"])])
    for s in ("grid", "sobol"):
        d = np.log(g(s)) - np.log(g("random"))
        print(f"   {s}/random geo ratio {np.exp(d.mean()):.2f} better {int((d<0).sum())}/5 p={stats.wilcoxon(d).pvalue:.3f}")
print("== classical")
cl = json.load(open(R + "/classical.json"))
print(type(cl), (list(cl.keys()) if isinstance(cl, dict) else cl[0]))
print("== timing")
print(json.dumps(json.load(open(R + "/timing_single.json")), indent=0)[:1700])
print("== validate_batched")
vb = json.load(open(R + "/validate_batched.json"))
print(type(vb), (list(vb.keys()) if isinstance(vb, dict) else (len(vb), vb[0])))
