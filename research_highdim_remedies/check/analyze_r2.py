"""Summarise runs_reimpl2.jsonl by the criteria of the plan of the second round (notes/VERIFICATION.md, section 4.3) -> reimpl2_summary.json / .txt"""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, "runs_reimpl2.jsonl"))]
def cell(block, p, m, d, arm, iters=None):
    return {r["seed"]: r for r in R if r["block"] == block and r["p"] == p and r["method"] == m and r["d"] == d and r["arm"] == arm and (iters is None or r["iters"] == iters)}
def verdict(rho):
    med = st.median(rho)
    if all(x > 1 for x in rho) and med >= 1.5: return "helps"
    if med <= 1/1.1: return "hurts"
    return "neutral"
out, lines = {}, []
for p, m, d in (("P1", "pinn", 20), ("P6", "ritz", 20), ("P6", "ritz", 10), ("P7", "ritz", 20), ("P6", "pinn", 20)):
    pl = cell("main", p, m, d, "plain", 4000)
    if not pl: continue
    key = f"{p}/{m}/d{d}"
    out[key + "/plain"] = dict(err=[round(pl[s]["rel_l2"], 4) for s in sorted(pl)], cpu=[round(pl[s]["cpu_total"], 1) for s in sorted(pl)])
    for arm, blk in (("presolve", "main"), ("lift3c", "main"), ("lift1", "main"), ("lift3c", "eqcpu")):
        a = cell(blk, p, m, d, arm, 4000 if blk == "main" else None)
        s = sorted(set(a) & set(pl))
        if not s: continue
        rho = [pl[k]["rel_l2"] / a[k]["rel_l2"] for k in s]
        out[f"{key}/{arm}/{blk}"] = dict(iters=sorted({a[k]["iters"] for k in s}), rho=[round(x, 2) for x in rho], median=round(st.median(rho), 2),
                                        wins=f"{sum(x > 1 for x in rho)}/{len(rho)}", verdict=verdict(rho), err=[round(a[k]["rel_l2"], 4) for k in s],
                                        cpu=[round(a[k]["cpu_total"], 1) for k in s])
    l1, l3 = cell("main", p, m, d, "lift1", 4000), cell("main", p, m, d, "lift3c", 4000)
    s = sorted(set(l1) & set(l3))
    if s:
        rho = [l1[k]["rel_l2"] / l3[k]["rel_l2"] for k in s]
        out[f"{key}/lift3c_vs_lift1"] = dict(rho=[round(x, 2) for x in rho], median=round(st.median(rho), 2), verdict=verdict(rho))
for k, v in out.items(): lines.append(f"{k} {v}")
open(os.path.join(HERE, "reimpl2_summary.txt"), "w").write("\n".join(lines) + "\n")
json.dump(out, open(os.path.join(HERE, "reimpl2_summary.json"), "w"), indent=1)
print("\n".join(lines))
