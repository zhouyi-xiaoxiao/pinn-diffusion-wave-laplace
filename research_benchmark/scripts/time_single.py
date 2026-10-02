"""Per-run cost of the single-model reference driver (CPU, 4 threads, float32).

The multi-seed benchmark trains stacked models on the GPU, so per-run wall-clock is measured
here separately: 400 Adam steps and 200 L-BFGS iterations (torch.optim.LBFGS, strong Wolfe) per
problem, converted to ms/step. The cost of a full benchmark run is then
  adam:        n_adam * ms_adam
  adam_lbfgs:  branch_at * ms_adam + (n_adam - branch_at) * ms_lbfgs_iter
Output: results/timing_single.json
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(4)
from pinnbench.problems import make_problem
from pinnbench.core import run_single, MLP
from run_benchmark import CONFIG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {"note": "CPU, torch.set_num_threads(4), measured on a shared machine under heavy external load "
               "(load average 50-100 on 12 cores) -> upper bounds; relative costs are meaningful.",
       "loadavg_start": os.getloadavg()}
for name, cfg in CONFIG.items():
    P = make_problem(name, device="cpu")
    eX = P.eval_points()[:2000]; eu = P.exact(eX)
    r = run_single(P, "sobol", 0, cfg["n_r"], 400, 200, log_every=200, test_X=eX, test_u=eu, eval_X=eX, eval_u=eu)
    ms_adam = 1e3 * r["adam"]["time"] / 400
    t_branch = [c["time"] for c in r["adam"]["curve"] if c["it"] == 200][0]
    it_l = max(r["adam_lbfgs"]["lbfgs_iters"], 1)
    ms_lbfgs = 1e3 * (r["adam_lbfgs"]["time"] - t_branch) / it_l
    n_par = sum(p.numel() for p in MLP(P.d, P.hidden).parameters())
    est_adam = cfg["n_adam"] * ms_adam / 1e3
    est_lb = (cfg["branch_at"] * ms_adam + (cfg["n_adam"] - cfg["branch_at"]) * ms_lbfgs) / 1e3
    out[name] = {"ms_per_adam_step": ms_adam, "ms_per_lbfgs_iter": ms_lbfgs, "lbfgs_fevals_per_iter": r["adam_lbfgs"]["lbfgs_fevals"] / it_l,
                 "n_params": n_par, "n_r": cfg["n_r"], "est_run_s_adam": est_adam, "est_run_s_adam_lbfgs": est_lb}
    print(name, json.dumps(out[name]), flush=True)
out["loadavg_end"] = os.getloadavg()
json.dump(out, open(os.path.join(ROOT, "results", "timing_single.json"), "w"), indent=1)
