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

"""Separately written re-aggregation of the study's raw run JSONs (no use of the study's analyze.py)."""
import json, glob, os, itertools, math
import numpy as np
from scipy import stats
R = _repo_path("research_benchmark/results")
rows = {}
for f in sorted(glob.glob(R + "/runs/*_s*-*.json")):
    d = json.load(open(f))
    for r in d["runs"]:
        for arm in ("adam", "adam_lbfgs"):
            key = (r["problem"], r["sampler"], arm, r["seed"])
            assert key not in rows, key
            rows[key] = (r[arm]["rel_l2"], r[arm]["linf"], r[arm]["curve"])
    print(os.path.basename(f), d["config"], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in d["timing"].items()})
print("total rows", len(rows))
# compare against CSV
import csv
csvrows = list(csv.DictReader(open(R + "/benchmark_runs.csv")))
print("csv rows", len(csvrows))
mx = 0
for c in csvrows:
    k = (c["problem"], c["sampler"], c["arm"], int(c["seed"]))
    mx = max(mx, abs(float(c["rel_l2"]) - rows[k][0]), abs(float(c["linf"]) - rows[k][1]))
print("max abs diff csv vs raw json:", mx)
P = ["heat1d", "wave1d", "wave2d", "laplace2d", "laplace3d"]; S = ["grid", "random", "resample", "sobol", "rad"]
def get(p, s, a, j=0):
    seeds = sorted(k[3] for k in rows if k[:3] == (p, s, a))
    return np.array([rows[(p, s, a, sd)][j] for sd in seeds]), seeds
tot_better = 0; tot = 0
for p in P:
    print("==", p)
    for a in ("adam", "adam_lbfgs"):
        for s in S:
            v, seeds = get(p, s, a); l, _ = get(p, s, a, 1)
            print(f"  {a:10s} {s:9s} n={len(v)} relL2 {v.mean():.3e} +- {v.std(ddof=1):.3e} geo {np.exp(np.log(v).mean()):.3e} min {v.min():.3e} max {v.max():.3e} | Linf {l.mean():.3e} +- {l.std(ddof=1):.3e}")
    va = np.concatenate([get(p, s, "adam")[0] for s in S]); vb = np.concatenate([get(p, s, "adam_lbfgs")[0] for s in S])
    d = np.log(vb) - np.log(va)
    tot_better += int((d < 0).sum()); tot += len(d)
    print(f"  LBFGS vs Adam: geo {np.exp(np.log(va).mean()):.3e} -> {np.exp(np.log(vb).mean()):.3e} ratio {np.exp(d.mean()):.3f} better {int((d<0).sum())}/{len(d)} wilcoxon p={stats.wilcoxon(d).pvalue:.2e}; range adam {va.min():.3e}-{va.max():.3e}, lbfgs {vb.min():.3e}-{vb.max():.3e}")
    # per-seed aggregated (avoid pseudo-replication): geo-mean over strategies per seed
    seeds = get(p, "grid", "adam")[1]
    ds = np.array([np.mean([math.log(rows[(p, s, "adam_lbfgs", sd)][0]) - math.log(rows[(p, s, "adam", sd)][0]) for s in S]) for sd in seeds])
    print(f"    per-seed aggregated: {int((ds<0).sum())}/{len(ds)} seeds, wilcoxon p={stats.wilcoxon(ds).pvalue:.4f}")
    for a in ("adam", "adam_lbfgs"):
        base, seeds = get(p, "random", a)
        for s in S:
            if s == "random": continue
            v, sd = get(p, s, a); assert sd == seeds
            d = np.log(v) - np.log(base)
            print(f"  vs random {a:10s} {s:9s} ratio {np.exp(d.mean()):.2f} better {int((d<0).sum())}/{len(d)} p={stats.wilcoxon(d).pvalue:.4f}  per-seed ratios {np.round(np.exp(d),2).tolist()}")
        # sobol vs grid directly
        g, _ = get(p, "grid", a); so, _ = get(p, "sobol", a); d = np.log(so) - np.log(g)
        print(f"  sobol/grid {a}: ratio {np.exp(d.mean()):.2f} better {int((d<0).sum())}/{len(d)} p={stats.wilcoxon(d).pvalue:.4f}")
print("LBFGS better overall:", tot_better, "/", tot)
# wave2d L-BFGS progress
for p in ("wave2d",):
    for it in (2500, 3000):
        vals = []
        for k, v in rows.items():
            if k[0] == p and k[2] == "adam_lbfgs":
                c = [x for x in v[2] if x["it"] == it]
                if c: vals.append(c[-1]["test_rel_l2"])
        print(p, "lbfgs it", it, "median test rel_l2", np.median(vals), "n", len(vals))
