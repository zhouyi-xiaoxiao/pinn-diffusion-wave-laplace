"""Part B -- main benchmark: 5 problems x 5 collocation strategies x {Adam, Adam->L-BFGS} x seeds.

Usage: run_benchmark.py <problem[,problem...]> <seed_start> <seed_stop> [device]
Each (problem, seed block) is one batched run of 5*(seed_stop-seed_start) stacked models and
is checkpointed to results/runs/<problem>_s<start>-<stop>.json (skipped if it exists).
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(4)
from pinnbench.core import run_batched

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLERS = ["grid", "random", "resample", "sobol", "rad"]
# Matched interior budgets (perfect powers so that grid, Sobol and random use exactly the same N):
#   16^2 = 2^8 for the smooth 2-D problems, 32^2 = 2^10 for the two-mode wave (8 points per
#   shortest period), 8^3 = 2^9 for the 3-D domains. These are deliberately in the regime where
#   the collocation set matters; results/budget_sweep.json shows the dependence on N.
# 3000 iterations per arm; the L-BFGS arm branches from the Adam trajectory at iteration 1500.
CONFIG = {
    "heat1d":    dict(n_r=256,  n_adam=3000, branch_at=1500),
    "laplace2d": dict(n_r=256,  n_adam=3000, branch_at=1500),
    "wave1d":    dict(n_r=1024, n_adam=3000, branch_at=1500),
    "wave2d":    dict(n_r=512,  n_adam=3000, branch_at=1500),
    "laplace3d": dict(n_r=512,  n_adam=3000, branch_at=1500),
}

if __name__ == "__main__":
    problems = sys.argv[1].split(",")
    s0, s1 = int(sys.argv[2]), int(sys.argv[3])
    device = sys.argv[4] if len(sys.argv) > 4 else ("mps" if torch.backends.mps.is_available() else "cpu")
    for name in problems:
        out_path = os.path.join(ROOT, "results", "runs", f"{name}_s{s0}-{s1}.json")
        if os.path.exists(out_path):
            print("skip (exists)", out_path, flush=True)
            continue
        cfg = CONFIG[name]
        specs = [(s, seed) for s in SAMPLERS for seed in range(s0, s1)]
        t0 = time.time()
        print(f"=== {name}: {len(specs)} stacked models, {cfg}, device={device}", flush=True)
        out, timing, preds = run_batched(name, specs, cfg["n_r"], cfg["n_adam"], cfg["branch_at"], device=device,
                                         log_every=100, verbose=True, return_preds=True)
        timing["wall_total_s"] = time.time() - t0
        # keep the held-out predictions of the first seed of each sampler (for error maps)
        keep = [i for i, (s, seed) in enumerate(specs) if seed == s0]
        np.savez_compressed(os.path.join(ROOT, "results", "runs", f"{name}_pred_seed{s0}.npz"),
                            samplers=np.array([specs[i][0] for i in keep]),
                            adam=preds["adam"][keep], adam_lbfgs=preds["adam_lbfgs"][keep])
        json.dump({"config": cfg, "timing": timing, "runs": out}, open(out_path, "w"))
        print(f"=== {name} done in {timing['wall_total_s']:.1f}s  timing={timing}", flush=True)
        for arm in ("adam", "adam_lbfgs"):
            for s in SAMPLERS:
                e = np.array([o[arm]["rel_l2"] for o in out if o["sampler"] == s])
                print(f"   {arm:10s} {s:9s} rel_l2 mean {e.mean():.3e} std {e.std(ddof=1):.3e} median {np.median(e):.3e}", flush=True)
    print("DONE", flush=True)
