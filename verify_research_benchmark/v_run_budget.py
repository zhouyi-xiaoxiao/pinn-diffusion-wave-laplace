"""Spot check of the budget-sweep claim: heat1d, N_r = 1024, grid vs random, seeds 0,1 (Adam->L-BFGS)."""
import json, os, time
from vpinn import problem, train
OUT = "v_runs_budget.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
P = problem("heat1d")
for seed in (0, 1):
    for kind in ("grid", "random"):
        key = f"heat1d|{kind}|{seed}|1024|fixed"
        if key in res: continue
        t0 = time.time(); r = train(P, kind, seed, 1024); r["wall"] = time.time() - t0; res[key] = r
        json.dump(res, open(OUT, "w"), indent=1)
        print(key, "adam %.3e | adam_lbfgs %.3e (iters %d) wall %.0f" % (r["adam"]["rel_l2"], r["adam_lbfgs"]["rel_l2"], r["adam_lbfgs"]["lbfgs_iters"], r["wall"]), flush=True)
print("BUDGET DONE")
