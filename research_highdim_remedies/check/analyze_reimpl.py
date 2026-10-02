import json, statistics as st
R = [json.loads(l) for l in open("runs_reimpl.jsonl")]
def cell(block, p, m, d, arm, iters=None):
    return {r["seed"]: r for r in R if r["block"] == block and r["p"] == p and r["method"] == m and r["d"] == d and r["arm"] == arm and (iters is None or r["iters"] == iters)}
out = {}
for p in ("P1", "P3", "P5"):
    for m in ("ritz", "pinn"):
        for d in (5, 10, 20):
            base = cell("main", p, m, d, "plain", 4000)
            if not base: continue
            out[f"{p}/{m}/d{d}/plain"] = dict(err=[round(base[s]["rel_l2"], 4) for s in sorted(base)], cpu=[round(base[s]["cpu_total"], 1) for s in sorted(base)])
            for arm, blk in (("presolve", "main"), ("lift3c", "main"), ("lift3u", "main"), ("rep3", "main"), ("lift1", "main"), ("lift3c", "eqcpu")):
                c = cell(blk, p, m, d, arm)
                s = sorted(set(c) & set(base))
                if not s: continue
                rho = [base[k]["rel_l2"] / c[k]["rel_l2"] for k in s]
                out[f"{p}/{m}/d{d}/{arm}/{blk}"] = dict(iters=sorted({c[k]["iters"] for k in s}), rho=[round(x, 2) for x in rho], median=round(st.median(rho), 2),
                                                       wins=f"{sum(x > 1 for x in rho)}/{len(rho)}", err=[round(c[k]["rel_l2"], 4) for k in s],
                                                       cpu=[round(c[k]["cpu_total"], 1) for k in s])
            l1 = cell("main", p, m, d, "lift1")
            l3 = cell("main", p, m, d, "lift3c", 4000)
            s = sorted(set(l1) & set(l3))
            if s:
                rho = [l1[k]["rel_l2"] / l3[k]["rel_l2"] for k in s]
                out[f"{p}/{m}/d{d}/lift3c_vs_lift1"] = dict(rho=[round(x, 2) for x in rho], median=round(st.median(rho), 2))
for k, v in out.items(): print(k, v)
json.dump(out, open("reimpl_summary.json", "w"), indent=1)
