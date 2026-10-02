"""Second round of the re-check: recompute the numbers quoted in the article (incl. P5 addendum, lift1, CPU ratios, row count) from the raw runs_*.jsonl."""
import json, os, statistics as st, glob, csv
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
L = lambda f: [json.loads(l) for l in open(os.path.join(RES, f))]
P = {"laplace": "P1", "cospoisson": "P2", "ridge": "P3", "prodcos": "P4", "altridge": "P5"}
rows = []
for f in ("runs_main20.jsonl", "runs_eqcpu.jsonl", "runs_posthoc_lift1.jsonl", "runs_p5main20.jsonl", "runs_p5eqcpu.jsonl", "runs_p5dim.jsonl", "runs_d10.jsonl", "runs_stall.jsonl"):
    rows += L(f)
names = sorted({r["problem"] for r in rows}); print("problem names", names)

def get(problem, method, arm, d=20, phases=None, iters=None):
    out = {}
    for r in rows:
        if r["problem"] == problem and r["method"] == method and r["arm"] == arm and r["d"] == d and r.get("schedule", True):
            if phases and r["phase"] not in phases: continue
            if iters is not None and r["iters"] != iters: continue
            out[r["seed"]] = r
    return out

def ratio(num, den):
    s = sorted(set(num) & set(den)); rho = [num[k]["rel_l2"] / den[k]["rel_l2"] for k in s]
    return dict(median=round(st.median(rho), 3), wins=sum(x > 1 for x in rho), n=len(rho), min=round(min(rho), 3), max=round(max(rho), 3)) if rho else None

out = {}
main = ("main20", "posthoc_lift1", "p5main20", "d10", "p5dim")
for pr in names:
    for m in ("ritz", "pinn"):
        for d in (20, 10, 5):
            pl = get(pr, m, "plain", d, main, 4000)
            if not pl: continue
            out[f"{P.get(pr, pr)}/{m}/d{d}/plain_mean"] = round(st.mean(r["rel_l2"] for r in pl.values()), 4)
            for arm in ("presolve", "lift3c", "lift3u", "rep3", "lift1"):
                a = get(pr, m, arm, d, main, 4000)
                if a: out[f"{P.get(pr, pr)}/{m}/d{d}/{arm}"] = ratio(pl, a)
                if arm in ("lift3c", "presolve") and d == 20:
                    e = {k: r for k, r in get(pr, m, arm, d, ("eqcpu", "p5eqcpu")).items() if r["iters"] != 4000}
                    if e: out[f"{P.get(pr, pr)}/{m}/d20/{arm}_eqcpu"] = dict(ratio(pl, e), iters=sorted({r['iters'] for r in e.values()}))
                    if arm == "presolve" and a:
                        out[f"{P.get(pr, pr)}/{m}/d20/presolve_cpu_ratio"] = round(st.median(r["cpu_total_s"] for r in a.values()) / st.median(r["cpu_total_s"] for r in pl.values()), 3)
            l1 = get(pr, m, "lift1", 20, main, 4000); l3 = get(pr, m, "lift3c", 20, main, 4000)
            if l1 and l3 and d == 20: out[f"{P.get(pr, pr)}/{m}/d20/lift3c_vs_lift1"] = ratio(l1, l3)
print(json.dumps(out, indent=0))
with open(os.path.join(RES, "per_seed_all.csv")) as fh: n = sum(1 for _ in csv.DictReader(fh))
print("per_seed_all rows", n)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute2.json"), "w"), indent=1)
