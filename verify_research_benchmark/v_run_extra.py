"""Extra seeds for the heat grid-vs-random claim and the node-centred-grid control (seeds 3,4,5)."""
import json, os, time
from vpinn import problem, train
OUT = "v_runs_extra.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
P = problem("heat1d")
for seed in (3, 4, 5):
    for kind in ("grid", "random", "nodegrid"):
        key = f"heat1d|{kind}|{seed}|256|fixed"
        if key in res: continue
        t0 = time.time(); r = train(P, kind, seed, 256); r["wall"] = time.time() - t0; res[key] = r
        json.dump(res, open(OUT, "w"), indent=1)
        print(key, "adam %.3e | adam_lbfgs %.3e (iters %d)" % (r["adam"]["rel_l2"], r["adam_lbfgs"]["rel_l2"], r["adam_lbfgs"]["lbfgs_iters"]), flush=True)
print("EXTRA DONE")
