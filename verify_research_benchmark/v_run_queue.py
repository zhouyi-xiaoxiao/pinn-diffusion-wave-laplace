"""Re-runs of the verification (own implementation vpinn.py); checkpointed to v_runs.json after every run."""
import json, os, sys, time
import numpy as np
from vpinn import problem, train
OUT = "v_runs.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
queue = []
for seed in (0, 1, 2):
    for kind in ("grid", "random", "nodegrid"):
        queue.append(("heat1d", kind, seed, 256, False))
for seed in (0, 1, 2):
    for kind in ("grid", "random"):
        queue.append(("laplace2d", kind, seed, 256, False))
queue += [("wave2d", "random", 0, 512, False), ("laplace3d", "sobol", 0, 512, False), ("wave1d", "random", 0, 1024, False)]
for seed in (0, 1, 2):
    queue.append(("laplace2d", "random", seed, 256, True))     # per-step resampling (Adam arm is the one of interest)
queue += [("heat1d", "sobol", s, 256, False) for s in (0, 1, 2)]
queue += [("laplace2d", "sobol", s, 256, False) for s in (0, 1, 2)]
probs = {}
for name, kind, seed, n, rs in queue:
    key = f"{name}|{kind}|{seed}|{n}|{'resample' if rs else 'fixed'}"
    if key in res: continue
    if name not in probs: probs[name] = problem(name)
    t0 = time.time()
    r = train(probs[name], kind, seed, n, resample=rs)
    r["wall"] = time.time() - t0
    res[key] = r
    json.dump(res, open(OUT, "w"), indent=1)
    print(key, "adam %.3e | adam@1500 %.3e | adam_lbfgs %.3e (Linf %.2e, iters %d, nfev %d) wall %.0fs" % (
        r["adam"]["rel_l2"], r["adam_at_branch"]["rel_l2"], r["adam_lbfgs"]["rel_l2"], r["adam_lbfgs"]["linf"], r["adam_lbfgs"]["lbfgs_iters"], r["adam_lbfgs"]["nfev"], r["wall"]), flush=True)
print("QUEUE DONE", flush=True)
