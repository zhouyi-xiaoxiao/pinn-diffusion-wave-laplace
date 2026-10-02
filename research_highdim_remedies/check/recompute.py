"""Check: recompute every quoted number from the raw run records (no reuse of analyze.py)."""
import json, glob, statistics as st, os, collections
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
runs = []
for f in sorted(glob.glob(R + "/runs_*.jsonl")):
    for l in open(f):
        if l.strip():
            r = json.loads(l); r["_file"] = os.path.basename(f); runs.append(r)
cnt = collections.Counter((r["_file"], r["phase"]) for r in runs)
print("counts", dict(cnt))
fin = sorted((r.get("finished_at", ""), r["phase"]) for r in runs if r["phase"] not in ("timing", "sweep"))
print("first counted finished_at", fin[0], "last", fin[-1])
fin2 = sorted(r.get("finished_at", "") for r in runs if r["phase"] in ("timing", "sweep"))
print("timing/sweep finished range", fin2[0] if fin2 else None, fin2[-1] if fin2 else None)
# duplicates
keys = collections.Counter((r["phase"], r["problem"], r["method"], r["d"], r["seed"], r["arm"], r["iters"], r["schedule"]) for r in runs)
print("duplicate keys", [k for k, v in keys.items() if v > 1])
P = {"laplace": "P1", "poisson": "P2", "ridge": "P3", "cospair": "P4"}
def cell(phase, prob, meth, arm, d=20, iters=4000, sched=True):
    return {r["seed"]: r for r in runs if r["phase"] == phase and r["problem"] == prob and r["method"] == meth
            and r["arm"] == arm and r["d"] == d and (iters is None or r["iters"] == iters) and r["schedule"] == sched}
out = {}
for d, ph in ((20, "main20"), (10, "d10")):
    for prob in P:
        for meth in ("ritz", "pinn"):
            base = cell(ph, prob, meth, "plain", d)
            if not base: continue
            for arm, ph2, it in (("presolve", ph, 4000), ("lift3c", ph, 4000), ("lift3u", ph, 4000), ("rep3", ph, 4000),
                                 ("lift3c", "eqcpu", None), ("presolve", "eqcpu", None), ("lift1", "posthoc_lift1", 4000)):
                c = cell(ph2, prob, meth, arm, d, it)
                if not c: continue
                s = sorted(set(c) & set(base))
                rho = [base[k]["rel_l2"] / c[k]["rel_l2"] for k in s]
                out[f"{P[prob]}/d{d}/{meth}/{arm}/{ph2}"] = dict(seeds=s, iters=sorted({c[k]['iters'] for k in s}),
                    med=round(st.median(rho), 4), mn=round(min(rho), 4), mx=round(max(rho), 4), wins=sum(x > 1 for x in rho),
                    mean_err=st.mean(c[k]["rel_l2"] for k in s), mean_plain=st.mean(base[k]["rel_l2"] for k in s))
for k, v in out.items():
    print(k, v)
# plain means
for prob in P:
    for meth in ("ritz", "pinn"):
        b = cell("main20", prob, meth, "plain")
        print("plain", P[prob], meth, [round(b[s]["rel_l2"], 4) for s in sorted(b)], "mean", round(st.mean(r["rel_l2"] for r in b.values()), 4))
# presolve shift cpu
sc = [r["cpu_shift_s"] for r in runs if r["arm"] == "presolve" and r["phase"] in ("main20", "eqcpu", "d10", "stall")]
print("cpu_shift range", min(sc), max(sc))
# P4 presolve identical?
for meth in ("ritz", "pinn"):
    b = cell("main20", "cospair", meth, "plain"); c = cell("main20", "cospair", meth, "presolve")
    print("P4 identical", meth, [b[s]["rel_l2"] == c[s]["rel_l2"] for s in sorted(b)], [b[s]["rel_l2"] - c[s]["rel_l2"] for s in sorted(b)])
# stall
for arm in ("plain", "presolve"):
    c = cell("stall", "poisson", "ritz", arm, 20, 12000, False)
    print("stall", arm, [(s, round(c[s]["gamma_end"], 4), round(c[s]["rel_l2"], 4), c[s]["T_half"]) for s in sorted(c)])
# ridge ritz seed 10
print("ridge ritz s10 plain/presolve", cell("main20","ridge","ritz","plain")[10]["rel_l2"], cell("main20","ridge","ritz","presolve")[10]["rel_l2"])
# loads and walls
print("load range", min(r.get("load1", 0) for r in runs if "load1" in r), max(r.get("load1", 0) for r in runs if "load1" in r))
print("wall max", max(r.get("wall_s", 0) for r in runs))
# T_half
for arm in ("plain","presolve","lift3c","lift3u","rep3"):
    for prob in ("laplace","poisson"):
        c = cell("main20", prob, "ritz", arm)
        print("Thalf", P[prob], arm, [c[s]["T_half"] for s in sorted(c)])
