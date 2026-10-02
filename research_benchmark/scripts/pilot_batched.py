"""Pilot: batched-vs-single equivalence of the Adam phase (CPU) and batched speed on MPS."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(4)
from pinnbench.problems import make_problem
from pinnbench.core import run_single, run_batched

name = sys.argv[1]; dev = sys.argv[2]; S = int(sys.argv[3]); n_adam = int(sys.argv[4]); n_r = int(sys.argv[5])
samplers = ["grid", "random", "resample", "sobol", "rad"]
specs = [(s, seed) for s in samplers for seed in range(S // 5)]
t0 = time.time()
out, timing = run_batched(name, specs, n_r, n_adam, n_adam // 2, device=dev, log_every=100, verbose=True)
print("batched wall", time.time() - t0, timing)
for o in out:
    print(o["sampler"], o["seed"], "adam rel_l2 %.3e linf %.3e | adam+lbfgs rel_l2 %.3e linf %.3e" % (o["adam"]["rel_l2"], o["adam"]["linf"], o["adam_lbfgs"]["rel_l2"], o["adam_lbfgs"]["linf"]))
