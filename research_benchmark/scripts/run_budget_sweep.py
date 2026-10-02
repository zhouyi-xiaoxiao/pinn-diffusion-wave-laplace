"""Collocation-budget sweep (extension): rel-L2 vs number of interior points N_r for the three
fixed strategies (grid, i.i.d. random, Sobol), 5 seeds, same protocol as the main benchmark.
The N_r = main-benchmark value is taken from results/runs/ (not recomputed).

Usage: run_budget_sweep.py <problem> <N_r,N_r,...> [device]
Output: results/budget_sweep.json (appended / merged)
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(4)
from pinnbench.core import run_batched
from run_benchmark import CONFIG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "budget_sweep.json")
name = sys.argv[1]
sizes = [int(v) for v in sys.argv[2].split(",")]
device = sys.argv[3] if len(sys.argv) > 3 else "mps"
rows = json.load(open(OUT)) if os.path.exists(OUT) else []
cfg = CONFIG[name]
SAMPLERS = ["grid", "random", "sobol"]

def strip(r):
    return {"problem": r["problem"], "sampler": r["sampler"], "seed": r["seed"], "n_r": r["n_r"],
            "adam": {k: r["adam"][k] for k in ("rel_l2", "linf")},
            "adam_lbfgs": {k: r["adam_lbfgs"][k] for k in ("rel_l2", "linf")}}

# main-benchmark budget, reused
if not any(r["problem"] == name and r["n_r"] == cfg["n_r"] for r in rows):
    for f in glob.glob(os.path.join(ROOT, "results", "runs", f"{name}_s*-*.json")):
        rows += [strip(r) for r in json.load(open(f))["runs"] if r["sampler"] in SAMPLERS and r["seed"] < 5]
for n in sizes:
    if any(r["problem"] == name and r["n_r"] == n for r in rows):
        continue
    specs = [(s, seed) for s in SAMPLERS for seed in range(5)]
    out, timing = run_batched(name, specs, n, cfg["n_adam"], cfg["branch_at"], device=device, log_every=100, verbose=True)
    rows += [strip(r) for r in out]
    json.dump(rows, open(OUT, "w"), indent=1)
    for s in SAMPLERS:
        e = np.array([r["adam_lbfgs"]["rel_l2"] for r in out if r["sampler"] == s])
        print(f"{name} N_r={n} {s}: adam_lbfgs rel_l2 mean {e.mean():.3e} std {e.std(ddof=1):.3e}", flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
print("DONE", flush=True)
