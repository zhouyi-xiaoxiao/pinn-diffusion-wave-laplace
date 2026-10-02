# -- added for the public repository: absolute paths of the development machine are replaced by
# -- paths relative to the repository root, through the helper _repo_path defined here.
import os as _os


def _repo_path(rel=""):
    d = _os.path.dirname(_os.path.abspath(__file__))
    while not (_os.path.isdir(_os.path.join(d, "research_benchmark")) and _os.path.isdir(_os.path.join(d, "research_highdim"))):
        p = _os.path.dirname(d)
        if p == d:
            raise RuntimeError("repository root not found")
        d = p
    return _os.path.join(d, rel) if rel else d
# -- end of the added lines

"""Cost per training iteration vs dimension.
Same network/batch as the study (MLP d-64-64-64-1 tanh, 1024 interior + 1024 boundary points, one Adam step).
Three variants: Deep Ritz (1 reverse pass for ∇u), PINN with forward-Laplacian (used in the study),
PINN with nested reverse-mode Laplacian (d extra reverse passes).
The machine is shared and heavily loaded, so: single thread, the three variants are INTERLEAVED in every
round (so load drifts hit all of them alike), and the primary metric is process CPU time; wall time is also stored.
For each d: 5 warm-up steps per variant, then 7 rounds of 10 timed steps per variant; median/min/max over rounds.
The PINN/Ritz cost ratio is computed PER ROUND (paired) and summarised by median/min/max, which is robust to load.
Output: results/cost_vs_d.json / .csv   (results/cost_vs_d_run1.* = first, unpaired run kept for the record)
"""
import sys, os, time, json, statistics, torch
sys.path.insert(0, os.path.dirname(__file__))
from hd_core import MLP, loss_pinn, loss_pinn_nested, loss_ritz, sample_interior, sample_boundary

ROOT = _repo_path("research_highdim")
torch.set_num_threads(1)
DS = [2, 3, 5, 10, 20, 50, 100]
VARIANTS = [("ritz", loss_ritz), ("pinn_forward", loss_pinn), ("pinn_nested", loss_pinn_nested)]
rows = []
for d in DS:
    steps = {}
    for name, lossf in VARIANTS:
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(0)
        net = MLP(d); opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        def step(net=net, opt=opt, gen=gen, lossf=lossf):
            xi = sample_interior(1024, d, gen); xb = sample_boundary(1024, d, gen)
            opt.zero_grad(set_to_none=True)
            loss, _, _ = lossf(net, "poisson", xi, xb, 10.0)
            loss.backward(); opt.step()
        steps[name] = (step, sum(p.numel() for p in net.parameters()))
        for _ in range(5):
            step()
    cpu = {n: [] for n, _ in VARIANTS}; wall = {n: [] for n, _ in VARIANTS}
    for _ in range(7):
        for name, _ in VARIANTS:
            c0 = time.process_time(); t0 = time.perf_counter()
            for _ in range(10):
                steps[name][0]()
            cpu[name].append(1e3 * (time.process_time() - c0) / 10); wall[name].append(1e3 * (time.perf_counter() - t0) / 10)
    ratio_f = [a / b for a, b in zip(cpu["pinn_forward"], cpu["ritz"])]   # paired within a round
    ratio_n = [a / b for a, b in zip(cpu["pinn_nested"], cpu["ritz"])]
    for name, _ in VARIANTS:
        r = dict(d=d, method=name, cpu_ms_median=statistics.median(cpu[name]), cpu_ms_min=min(cpu[name]),
                 cpu_ms_max=max(cpu[name]), wall_ms_median=statistics.median(wall[name]), n_params=steps[name][1],
                 cpu_ms_rounds=cpu[name], wall_ms_rounds=wall[name])
        if name != "ritz":
            rr = ratio_f if name == "pinn_forward" else ratio_n
            r.update(ratio_to_ritz_median=statistics.median(rr), ratio_to_ritz_min=min(rr), ratio_to_ritz_max=max(rr))
        rows.append(r); print({k: v for k, v in r.items() if not k.endswith("rounds")}, flush=True)
json.dump(rows, open(f"{ROOT}/results/cost_vs_d.json", "w"), indent=1)
with open(f"{ROOT}/results/cost_vs_d.csv", "w") as f:
    f.write("d,method,cpu_ms_median,cpu_ms_min,cpu_ms_max,wall_ms_median,n_params,ratio_to_ritz_median,ratio_to_ritz_min,ratio_to_ritz_max\n")
    for r in rows:
        f.write(f"{r['d']},{r['method']},{r['cpu_ms_median']:.3f},{r['cpu_ms_min']:.3f},{r['cpu_ms_max']:.3f},{r['wall_ms_median']:.3f},{r['n_params']},"
                f"{r.get('ratio_to_ritz_median', 1.0):.3f},{r.get('ratio_to_ritz_min', 1.0):.3f},{r.get('ratio_to_ritz_max', 1.0):.3f}\n")
