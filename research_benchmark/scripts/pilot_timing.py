"""Pilot: per-step cost of each problem (CPU, 4 threads) to size the benchmark budgets."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(4)
from pinnbench.problems import make_problem
from pinnbench.core import MLP, total_loss
from pinnbench.samplers import make_sampler

for name, n_r in [("heat1d", 1024), ("wave1d", 1024), ("laplace2d", 1024), ("wave2d", 4096), ("laplace3d", 4096)]:
    for dev in (["cpu", "mps"] if name in ("wave2d", "laplace3d") and len(sys.argv) > 1 else ["cpu"]):
        P = make_problem(name, device=dev)
        torch.manual_seed(0)
        net = MLP(P.d, P.hidden).to(dev)
        X = make_sampler("sobol", P, n_r, 0).initial()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for w in range(5):
            opt.zero_grad(); total_loss(P, net, X)[0].backward(); opt.step()
        t0 = time.perf_counter(); n = 60
        for _ in range(n):
            opt.zero_grad(); tot = total_loss(P, net, X)[0]; tot.backward(); opt.step()
        if dev == "mps": torch.mps.synchronize()
        dt = (time.perf_counter() - t0) / n
        print(f"{name:10s} {dev} N_r={n_r} hidden={P.hidden}: {dt*1e3:.2f} ms/step", flush=True)
