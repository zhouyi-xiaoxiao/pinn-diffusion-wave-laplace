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

"""Own aggregation of the study's raw run records (results/runs*.jsonl) -> check every table number."""
import json, glob, math, statistics as st, collections, numpy as np
R=[]
for f in sorted(glob.glob(_repo_path("research_highdim/results/runs*.jsonl"))):
    for l in open(f):
        if l.strip():
            r=json.loads(l); r["_file"]=f.split("/")[-1]; R.append(r)
print("records",len(R), collections.Counter(r["phase"] for r in R))
keys=collections.Counter((r["phase"],r["problem"],r["method"],r["d"],r["seed"],r["w"],r["iters"]) for r in R)
print("duplicate keys:",[k for k,v in keys.items() if v>1])
print("lap_impl by phase:",collections.Counter((r["phase"],r["method"],r.get("lap_impl","MISSING")) for r in R))
main=[r for r in R if r["phase"]=="main"]
G=collections.defaultdict(list)
for r in main: G[(r["problem"],r["method"],r["d"])].append(r)
out={}
print("\nMAIN (4000 its): mean ± sd [min,max] rel_l2 | centred | H1 | bd/int | w | seeds | cpu ms/it")
for p in ["laplace","poisson"]:
    for d in [2,3,5,10,20]:
        row={}
        for m in ["pinn","ritz"]:
            rs=sorted(G[(p,m,d)],key=lambda r:r["seed"])
            v=[r["rel_l2"] for r in rs]
            assert all(r["iters"]==4000 for r in rs)
            row[m]=v
            print(f"{p:8s} d={d:2d} {m}: {st.mean(v):.3e} ± {st.stdev(v):.2e} [{min(v):.3e},{max(v):.3e}] | cen {st.mean([r['rel_l2_centered'] for r in rs]):.3e} | H1 {st.mean([r['rel_h1semi'] for r in rs]):.3e} | bd/int {st.mean([r['bd_rel_l2'] for r in rs])/st.mean(v):.2f} | w={set(r['w'] for r in rs)} seeds={[r['seed'] for r in rs]} | {st.mean([r['cpu_ms_per_iter'] for r in rs]):.2f} | val-test maxreldiff {max(abs(r['val_rel_l2']/r['rel_l2']-1) for r in rs):.3f} | nan? {[r['final_loss'] for r in rs if not math.isfinite(r['final_loss'])]}")
        ratio=st.mean(row["ritz"])/st.mean(row["pinn"])
        overlap=not (max(row["pinn"])<min(row["ritz"]) or max(row["ritz"])<min(row["pinn"]))
        allbelow=max(row["pinn"])<min(row["ritz"])
        out[(p,d)]=dict(pinn=row["pinn"],ritz=row["ritz"])
        print(f"      -> Ritz/PINN = {ratio:.2f}; ranges overlap {overlap}; every PINN seed < every Ritz seed: {allbelow}")
# slopes
print("\nlog-log slopes of geo-mean rel_l2")
for p in ["laplace","poisson"]:
    for m in ["pinn","ritz"]:
        for key in ["rel_l2","rel_l2_centered"]:
            for ds in ([2,3,5,10],[2,3,5,10,20]):
                y=[np.mean(np.log([r[key] for r in G[(p,m,d)]])) for d in ds]
                s=np.polyfit(np.log(ds),y,1)[0]
                print(f"  {p} {m} {key} d={ds[0]}..{ds[-1]}: p={s:.2f}",end=";")
            print()
# extensions
print("\nEXTENSIONS")
for ph in ["long","equaltime"]:
    E=collections.defaultdict(list)
    for r in R:
        if r["phase"]==ph: E[(r["problem"],r["method"],r["d"],r["iters"],r["w"])].append(r)
    for k,rs in E.items():
        rs=sorted(rs,key=lambda r:r["seed"]); v=[r["rel_l2"] for r in rs]
        print(ph,k,[f"{x:.3e}" for x in v],f"mean {st.mean(v):.3e} sd {st.stdev(v):.2e} cpu_s mean {st.mean([r['cpu_time_s'] for r in rs]):.1f} cpu ms/it {st.mean([r['cpu_ms_per_iter'] for r in rs]):.2f}")
# equal-time rule inputs
for p in ["laplace","poisson"]:
    tp=[r["cpu_time_s"] for r in G[(p,"pinn",10)]]; mr=[r["cpu_ms_per_iter"] for r in G[(p,"ritz",10)]]
    print(f"rule {p}: PINN cpu {st.mean(tp):.1f}s {['%.1f'%t for t in tp]}; Ritz ms/it {st.mean(mr):.2f} {['%.2f'%t for t in mr]} -> iters {st.mean(tp)/(st.mean(mr)/1e3):.0f}")
# sweep
print("\nSWEEP")
for r in R:
    if r["phase"]=="sweep": print(r["problem"],r["method"],r["d"],r["seed"],r["w"],f"val {r['val_rel_l2']:.4e} test {r['rel_l2']:.4e}", r.get("lap_impl","(no lap_impl)"), r["_file"])
# compute totals
print("\nwall total s",sum(r["wall_s"] for r in R),"cpu total s",sum(r.get("cpu_time_s",0) for r in R),"max wall",max(r["wall_s"] for r in R), "n no-cpu", sum(1 for r in R if "cpu_time_s" not in r))
# plateau curves
print("\nplateau mean val")
for (p,m,d) in [("poisson","ritz",10),("poisson","pinn",10),("poisson","pinn",20),("poisson","ritz",20),("laplace","pinn",20),("laplace","ritz",20)]:
    rs=G[(p,m,d)]
    its=[250,500,1000,1500,2000,3000,4000]
    print(p,m,d,[f"{it}:{st.mean([dict((c['it'],c['val_rel_l2']) for c in r['curve'])[it] for r in rs]):.3f}" for it in its])
# cost fits
import csv
C=list(csv.DictReader(open(_repo_path("research_highdim/results/cost_vs_d.csv"))))
for m in ["ritz","pinn_forward","pinn_nested"]:
    ds=np.array([float(c["d"]) for c in C if c["method"]==m]); y=np.array([float(c["cpu_ms_median"]) for c in C if c["method"]==m])
    a,b=np.polyfit(ds,y,1); print(f"cost fit {m}: {b:.2f} + {a:.3f} d ; cost(100)/cost(2) = {y[-1]/y[0]:.1f}")
