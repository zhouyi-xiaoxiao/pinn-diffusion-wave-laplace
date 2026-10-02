"""Validation of the batched driver against the single-model reference driver.

 V1 (CPU, exactness of batched Adam): same seeds / same collocation sets -> the Adam loss
    trajectories of run_batched and run_single must agree to float32 round-off over the first
    few hundred steps (they diverge slowly afterwards because float32 round-off is chaotic).
 V2 (L-BFGS implementation): final held-out errors of our batched Armijo L-BFGS vs
    torch.optim.LBFGS(strong_wolfe) started from the same Adam snapshot, heat1d + laplace2d.
 V3 (speed): batched wall-clock on MPS for S = 25.
Output: results/validate_batched.json + log on stdout.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(4)
from pinnbench.problems import make_problem
from pinnbench.core import run_single, run_batched

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = {}
samplers = ["grid", "random", "resample", "sobol", "rad"]

def single(name, sampler, seed, n_r, n_adam, branch):
    P = make_problem(name, device="cpu")
    eX = P.eval_points(); eu = P.exact(eX); tX = P.test_subset(); tu = P.exact(tX)
    return run_single(P, sampler, seed, n_r, n_adam, branch, log_every=50, test_X=tX, test_u=tu, eval_X=eX, eval_u=eu)

# ---------------- V1 + V2 ----------------
for name, n_adam in [("heat1d", 600), ("laplace2d", 600)]:
    specs = [(s, seed) for s in samplers for seed in (0, 1)]
    t0 = time.time()
    outb, timing = run_batched(name, specs, 1024, n_adam, n_adam // 2, device="cpu", log_every=50)
    print(f"[{name}] batched CPU S={len(specs)} wall {time.time()-t0:.1f}s", flush=True)
    rows = []
    for ob in outb:
        t0 = time.time()
        os_ = single(name, ob["sampler"], ob["seed"], 1024, n_adam, n_adam // 2)
        cb = {c["it"]: c["loss"] for c in ob["adam"]["curve"]}
        cs = {c["it"]: c["loss"] for c in os_["adam"]["curve"]}
        dev = {it: abs(cb[it] - cs[it]) / abs(cs[it]) for it in (1, 50, 100, 300) if it in cb and it in cs}
        row = {"problem": name, "sampler": ob["sampler"], "seed": ob["seed"],
               "adam_loss_reldiff_it1": dev.get(1), "adam_loss_reldiff_it50": dev.get(50),
               "adam_loss_reldiff_it100": dev.get(100), "adam_loss_reldiff_it300": dev.get(300),
               "adam_rel_l2_batched": ob["adam"]["rel_l2"], "adam_rel_l2_single": os_["adam"]["rel_l2"],
               "lbfgs_rel_l2_batched": ob["adam_lbfgs"]["rel_l2"], "lbfgs_rel_l2_single": os_["adam_lbfgs"]["rel_l2"],
               "lbfgs_loss_batched": ob["adam_lbfgs"]["final_loss"], "lbfgs_loss_single": os_["adam_lbfgs"]["final_loss"],
               "single_time_adam": os_["adam"]["time"], "single_time_adam_lbfgs": os_["adam_lbfgs"]["time"],
               "single_lbfgs_fevals": os_["adam_lbfgs"]["lbfgs_fevals"], "single_lbfgs_iters": os_["adam_lbfgs"]["lbfgs_iters"],
               "single_lbfgs_stop": os_["adam_lbfgs"]["stop"], "batched_lbfgs_fevals": ob["adam_lbfgs"]["lbfgs_batched_fevals"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    res[name] = rows
    json.dump(res, open(os.path.join(ROOT, "results", "validate_batched.json"), "w"), indent=1)

# ---------------- V3 speed on MPS ----------------
if torch.backends.mps.is_available():
    for name, n_r in [("heat1d", 1024), ("laplace3d", 4096)]:
        specs = [(s, seed) for s in samplers for seed in range(5)]
        t0 = time.time()
        outb, timing = run_batched(name, specs, n_r, 200, 100, device="mps", log_every=100)
        print(f"[{name}] MPS S=25 N_r={n_r}: 200 its wall {time.time()-t0:.1f}s timing={timing}", flush=True)
        res[f"mps_speed_{name}"] = timing
    json.dump(res, open(os.path.join(ROOT, "results", "validate_batched.json"), "w"), indent=1)
print("DONE", flush=True)
