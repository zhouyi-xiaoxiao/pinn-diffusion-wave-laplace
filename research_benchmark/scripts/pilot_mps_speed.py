import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(4)
from pinnbench.core import run_batched
samplers = ["grid", "random", "resample", "sobol", "rad"]
for name, n_r, S in [("heat1d", 1024, 25), ("heat1d", 1024, 50), ("laplace3d", 4096, 25), ("wave1d", 1024, 50)]:
    specs = [(s, seed) for s in samplers for seed in range(S // 5)]
    t0 = time.time()
    out, timing = run_batched(name, specs, n_r, 300, 200, device="mps", log_every=100)
    print(f"[{name}] MPS S={S} N_r={n_r}: 300 adam + 100 lbfgs wall {time.time()-t0:.1f}s timing={timing}", flush=True)
