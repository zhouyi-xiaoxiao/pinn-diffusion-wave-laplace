"""Section 7.2 of the article: a PINN on a boundary-value problem with too few conditions.

The Laplace equation in (0, pi)^3 with the Dirichlet datum u(pi, y, z) = sin y cos z on the face x = pi
and no condition on the other five faces (it has infinitely many solutions; Section 7.2 of the
article). Five seeded networks (MLP 2 x 100, tanh, PyTorch default initialisation) are trained as one
stack on the closed 21^3 grid of (0, pi)^3 (9261 points; the datum on its 441 points with x = pi), with
the loss mean(Lap u)^2 + mean(u - datum)^2, AdamW (learning rate 1e-3, default weight decay 0.01),
2000 full-batch iterations, single precision. Each network is compared with the solution of problem
Lap3 (the same datum, u = 0 on the other five faces) on its 40^3 cell-centred evaluation grid, and the
mean of |u| is recorded on 20 x 20 points of each of the five faces without a condition.

    PINN_DEV=cpu python research_benchmark/scripts/run_oneface.py      # default device: mps if available

Output: results/oneface_laplace3d.json (key "oneface_laplace3d"). The stored file holds the run made on
the mps device; in floating point the result of a network depends on the device and on the stack.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(4)
from pinnbench.problems import make_problem, gradient, second, PI
from pinnbench.core import MLP, BatchedMLP, predict, error_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "oneface_laplace3d.json")
DEV = os.environ.get("PINN_DEV", "mps" if torch.backends.mps.is_available() else "cpu")
SEEDS = [0, 1, 2, 3, 4]

S = len(SEEDS)
Pc = make_problem("laplace3d")                      # the fully specified problem Lap3, for comparison
eX = Pc.eval_points(); eu = Pc.exact(eX)
ge = 21
a = np.linspace(0, PI, ge)
grid = np.stack([g.ravel() for g in np.meshgrid(a, a, a, indexing="ij")], 1)
mask = np.isclose(grid[:, 0], PI)
Xb = grid[mask]
gb = np.sin(Xb[:, 1]) * np.cos(Xb[:, 2])
mlps = []
for seed in SEEDS:
    torch.manual_seed(seed); mlps.append(MLP(3, [100, 100]))
net = BatchedMLP(mlps).to(DEV)
X = torch.as_tensor(np.repeat(grid[None].astype(np.float32), S, 0), device=DEV).requires_grad_(True)
Xb_t = torch.as_tensor(Xb.astype(np.float32), device=DEV)
gb_t = torch.as_tensor(gb.astype(np.float32), device=DEV)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
t0 = time.perf_counter()
for ep in range(2000):
    opt.zero_grad(set_to_none=True)
    u = net(X)
    g = gradient(u, X)
    lap = second(g, X, 0) + second(g, X, 1) + second(g, X, 2)
    lb = ((net(Xb_t).squeeze(-1) - gb_t) ** 2).mean(-1)
    ll = (lap ** 2).mean(-1)
    (lb + ll).sum().backward()
    opt.step()
wall = time.perf_counter() - t0
pred = predict(net, eX, DEV)
# mean |u| on the five faces without a condition (zero for the solution of Lap3)
m = 20; s = (np.arange(m) + 0.5) * PI / m
F = np.stack([g_.ravel() for g_ in np.meshgrid(s, s, indexing="ij")], 1)
faces = {}
for axis, val, nm in [(0, 0.0, "x0"), (1, 0.0, "y0"), (1, PI, "ypi"), (2, 0.0, "z0"), (2, PI, "zpi")]:
    pts = np.zeros((F.shape[0], 3)); others = [k for k in range(3) if k != axis]
    pts[:, axis] = val; pts[:, others[0]] = F[:, 0]; pts[:, others[1]] = F[:, 1]
    faces[nm] = np.abs(predict(net, pts, DEV)).mean(1)
rows = []
for i, seed in enumerate(SEEDS):
    rows.append({"seed": seed, **error_metrics(pred[i], eu), "final_bc_loss": float(lb[i].detach()),
                 "final_pde_loss": float(ll[i].detach()),
                 **{f"mean_abs_u_face_{k}": float(v[i]) for k, v in faces.items()}})
    print("oneface", rows[-1], flush=True)
# seed-to-seed disagreement of the prediction in the interior
spread = float(np.mean(np.std(pred, axis=0)) / np.sqrt(np.mean(eu ** 2)))
res = {"oneface_laplace3d": {"rows": rows, "interior_seed_std_over_rms_exact": spread, "batched_wall_s": wall,
                             "device": DEV}}
json.dump(res, open(OUT, "w"), indent=1)
print("seed spread (mean pointwise std / rms exact):", spread, flush=True)
print("DONE", flush=True)
