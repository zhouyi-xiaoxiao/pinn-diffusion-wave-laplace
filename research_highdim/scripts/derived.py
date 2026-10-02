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

"""Derived statistics of the study (pure stdlib; reads results/runs*.jsonl). Output: results/derived.json + stdout."""
import json, glob, math, statistics as st
ROOT = _repo_path("research_highdim")
R = []
for f in sorted(glob.glob(f"{ROOT}/results/runs*.jsonl")):
    R += [json.loads(l) for l in open(f) if l.strip()]
main = [r for r in R if r["phase"] == "main" and r["iters"] == 4000]
gm = lambda v: math.exp(st.mean(map(math.log, v)))
out = {"n_runs_total": len(R), "n_main": len(main), "n_sweep": sum(r["phase"] == "sweep" for r in R),
       "n_long": sum(r["phase"] == "long" for r in R), "n_equaltime": sum(r["phase"] == "equaltime" for r in R)}
G = {}
for r in main:
    G.setdefault((r["problem"], r["method"], r["d"]), []).append(r)
DS = sorted({r["d"] for r in main})


def slope(xs, ys):
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
    mx, my = st.mean(lx), st.mean(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


print("--- log-log slope of geo-mean test error vs d (error ~ d^p)")
out["slopes"] = {}
for p in ["laplace", "poisson"]:
    for m in ["pinn", "ritz"]:
        for metric in ["rel_l2", "rel_l2_centered", "rel_h1semi"]:
            g = [gm([r[metric] for r in G[(p, m, d)]]) for d in DS]
            s10 = slope(DS[:4], g[:4]); s20 = slope(DS, g)
            out["slopes"][f"{p}/{m}/{metric}"] = dict(d2_10=s10, d2_20=s20, growth_2_to_10=g[3] / g[0], growth_2_to_20=g[4] / g[0])
            print(f"{p:8s} {m:4s} {metric:16s} p(2..10)={s10:5.2f}  p(2..20)={s20:5.2f}  err(10)/err(2)={g[3]/g[0]:6.2f}  err(20)/err(2)={g[4]/g[0]:7.2f}")
print("--- PINN vs Ritz at equal iterations: ratio of mean rel-L2 (Ritz/PINN), and do the 3-seed ranges overlap?")
out["ritz_over_pinn"] = {}
for p in ["laplace", "poisson"]:
    for d in DS:
        a = [r["rel_l2"] for r in G[(p, "pinn", d)]]; b = [r["rel_l2"] for r in G[(p, "ritz", d)]]
        overlap = not (max(a) < min(b) or max(b) < min(a))
        out["ritz_over_pinn"][f"{p}/d{d}"] = dict(ratio=st.mean(b) / st.mean(a), ranges_overlap=overlap)
        print(f"{p:8s} d={d:2d}: Ritz/PINN = {st.mean(b)/st.mean(a):5.2f}   ranges overlap: {overlap}")
print("--- CPU ms/iter in the main runs (mean over 3 seeds x 2 problems; load-contaminated, see benchmark)")
out["cpu_ms_main"] = {}
for m in ["ritz", "pinn"]:
    for d in DS:
        v = [r["cpu_ms_per_iter"] for p in ["laplace", "poisson"] for r in G[(p, m, d)]]
        out["cpu_ms_main"][f"{m}/d{d}"] = dict(mean=st.mean(v), min=min(v), max=max(v))
        print(f"{m:4s} d={d:2d}: {st.mean(v):5.2f}  [{min(v):.2f}, {max(v):.2f}]")
print("--- plateau: validation rel-L2 at selected iterations (mean over seeds)")
out["plateau"] = {}
for p in ["laplace", "poisson"]:
    for m in ["pinn", "ritz"]:
        for d in [10, 20]:
            row = {}
            for it in [250, 500, 1000, 1500, 2000, 3000, 4000]:
                row[it] = st.mean(next(c["val_rel_l2"] for c in r["curve"] if c["it"] == it) for r in G[(p, m, d)])
            out["plateau"][f"{p}/{m}/d{d}"] = row
            print(f"{p:8s} {m:4s} d={d}: " + "  ".join(f"{it}:{v:.3f}" for it, v in row.items()))
print("--- boundary vs interior: rel-L2 on the boundary / rel-L2 in the interior (mean of seeds)")
for p in ["laplace", "poisson"]:
    for m in ["pinn", "ritz"]:
        print(f"{p:8s} {m:4s}: " + "  ".join(f"d={d}:{st.mean(r['bd_rel_l2'] for r in G[(p,m,d)])/st.mean(r['rel_l2'] for r in G[(p,m,d)]):.2f}" for d in DS))
json.dump(out, open(f"{ROOT}/results/derived.json", "w"), indent=1)
