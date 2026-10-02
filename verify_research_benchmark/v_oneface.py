"""Re-run of the demonstration of Section 7.2 (Laplace equation in the cube with data on one face only) with the
network and derivative code of V-bench (vpinn.py: own initialisation, Laplacian by forward propagation of
derivatives through the layers), seeds 0, 1, 2.

Setting as in research_benchmark/scripts/run_oneface.py: 2 x 100 tanh network, residual on the closed 21^3 grid
of [0, pi]^3, datum sin(y) cos(z) on its 441 points with x = pi only, loss mean(lap u)^2 + mean(u - datum)^2,
AdamW with learning rate 1e-3 and its default weight decay, 2000 iterations, one network at a time on the CPU.
Evaluation against the solution of Lap3 on its 40^3 cell-centred grid.

    python v_oneface.py        # trains the seeds missing from v_oneface.json (about 1.5 min each), prints all
"""
import json, os, time, math, numpy as np, torch
from vpinn import make_mlp, jets, grid, lap3d_exact, PI

OUT = "v_oneface.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))

a = (np.arange(40) + 0.5) * PI / 40; eX = grid(a, a, a); eU = lap3d_exact(eX)
ge = int(np.cbrt(10000)); lin = torch.linspace(0, math.pi, ge)
G = torch.cartesian_prod(lin, lin, lin); mask = G[:, 0] == math.pi
gb = torch.sin(G[mask, 1]) * torch.cos(G[mask, 2])
s = (np.arange(20) + 0.5) * PI / 20; F = grid(s, s); face0 = np.zeros((400, 3)); face0[:, 1:] = F
for seed in (0, 1, 2):
    key = f"oneface|{seed}"
    if key not in res:
        net = make_mlp(3, [100, 100], seed)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        t0 = time.time()
        for ep in range(2000):
            opt.zero_grad(set_to_none=True)
            u, g, h = jets(net, G)
            lp = (h.sum(1) ** 2).mean(); lb = ((u[mask] - gb) ** 2).mean()
            (lp + lb).backward(); opt.step()
        with torch.no_grad():
            p = net(T(eX))[:, 0].double().numpy(); f0 = float(net(T(face0)).abs().mean())
        res[key] = {"rel_l2": float(np.linalg.norm(p - eU) / np.linalg.norm(eU)), "linf": float(np.abs(p - eU).max()),
                    "bc_loss": float(lb), "pde_loss": float(lp), "mean_abs_u_face_x0": f0, "wall": time.time() - t0}
        json.dump(res, open(OUT, "w"), indent=1)
    print(key, res[key], flush=True)
