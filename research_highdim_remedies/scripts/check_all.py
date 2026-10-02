"""Numerical checks that must pass before any counted run (PREREG_C1.md, section 2). No training.
Writes results/checks.json. float64 throughout unless stated.

 C1  sources: f = -Lap u* by nested reverse-mode autograd, P1..P4, d in {2, 5, 10, 20}
 C2  forward Laplacian of every arm (plain, lift3c, lift3u, rep3, with and without affine shift)
     against nested reverse-mode autograd
 C3  boundary Gram matrix M: closed form against Gauss-Legendre product quadrature and Monte Carlo
 C4  affine minimisers: exact (quadrature) against brute force on two independent 10^6-point Monte-Carlo samples
     (Deep Ritz: L-BFGS on the sampled loss written as in hd_core.loss_ritz; PINN: torch.linalg.lstsq on the
     boundary sample), and the structural facts p = 0 on P4, c_1..c_d = 0 on P3
 C5  floors: exact one-dimensional formulas (src/floors.py) against least squares on 2 x 10^5 Monte-Carlo
     points and on the held-out test set of hd_core.fixed_sets (the set on which errors are reported)
"""
import sys, os, json, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import torch
import c1core as cc
import presolve as ps
import floors as fl
from c1core import hc

torch.set_num_threads(1)
F64 = torch.float64
OUT = {}
PROBS = ["laplace", "poisson", "ridge", "cospair"]


def nested_lap(fn, x):
    x = x.clone().requires_grad_(True)
    u = fn(x)
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        lap = lap + torch.autograd.grad(g[:, i].sum(), x, create_graph=True)[0][:, i:i + 1]
    return u.detach(), lap.detach()


# C1 ------------------------------------------------------------------------------------------
c1 = {}
for p in PROBS:
    for d in (2, 5, 10, 20):
        x = torch.rand(256, d, dtype=F64, generator=torch.Generator().manual_seed(d))
        u, lap = nested_lap(lambda z: cc.exact(p, z), x)
        f = cc.source(p, x)
        c1[f"{p}/d{d}"] = dict(max_abs_diff=(f + lap).abs().max().item(), max_abs_f=f.abs().max().item())
OUT["C1_sources"] = c1
print("C1", max(v["max_abs_diff"] for v in c1.values()))

# C2 ------------------------------------------------------------------------------------------
c2 = {}
for arm in ("plain", "lift3c", "lift3u", "rep3"):
    for d in (5, 20):
        for shifted in (False, True):
            torch.manual_seed(3)
            shift = torch.randn(d + 1) if shifted else None
            m = cc.Model(d, arm, shift=shift).double()
            x = torch.rand(128, d, dtype=F64, generator=torch.Generator().manual_seed(7))
            u_ref, lap_ref = nested_lap(m, x)
            u, lap = m.u_lap(x)
            c2[f"{arm}/d{d}/shift{int(shifted)}"] = dict(
                max_abs_diff_lap=(lap - lap_ref).abs().max().item(), max_abs_lap=lap_ref.abs().max().item(),
                max_abs_diff_u=(u - u_ref).abs().max().item())
OUT["C2_forward_laplacian"] = c2
print("C2", max(v["max_abs_diff_lap"] for v in c2.values()))

# C3 ------------------------------------------------------------------------------------------
c3 = {}
for d in (2, 5, 10, 20):
    M = ps.boundary_gram(d)
    # quadrature: g = 1 gives int phi; second moments via a problem with h = 1 are not available, so do
    # face-by-face products directly
    Mq = np.zeros((d + 1, d + 1))
    I1, I2 = np.sum(ps._GW * ps._GX), np.sum(ps._GW * ps._GX ** 2)   # 1/2 and 1/3 by quadrature
    for k in range(d):
        for s in (0.0, 1.0):
            m1 = np.full(d, I1); m1[k] = s           # int over face of x_i
            Mq[0, 0] += 1.0
            Mq[0, 1:] += m1; Mq[1:, 0] += m1
            for i in range(d):
                for j in range(d):
                    if i == j:
                        Mq[1 + i, 1 + j] += (s * s if i == k else I2)
                    else:
                        Mq[1 + i, 1 + j] += m1[i] * m1[j]
    gb = torch.Generator().manual_seed(500 + d)
    xb = hc.sample_boundary(2_000_000, d, gb).double()
    phi = torch.cat([torch.ones(len(xb), 1, dtype=F64), xb], 1)
    Mmc = (2 * d) * (phi.t() @ phi / len(xb)).numpy()
    c3[f"d{d}"] = dict(max_abs_diff_quadrature=float(np.abs(M - Mq).max()),
                       max_rel_diff_mc=float(np.abs(M - Mmc).max() / np.abs(M).max()))
OUT["C3_boundary_gram"] = c3
print("C3", c3)

# C4 ------------------------------------------------------------------------------------------
BETA = {"laplace": 100.0, "poisson": 1.0}   # study's weights; P3/P4 weights are checked for every candidate
c4 = {}
for p in PROBS:
    for d in (5, 20):
        for method, ws in (("ritz", (1.0, 10.0, 100.0, 1000.0)), ("pinn", (1000.0,))):
            for w in ws:
                c_ex = ps.affine_minimiser(p, d, method, w)
                sols = []
                for rep in range(2):
                    g = torch.Generator().manual_seed(90_000 + 100 * d + rep)
                    n = 1_000_000
                    xi = torch.rand(n, d, generator=g, dtype=F64)
                    xb = hc.sample_boundary(n, d, g).double()
                    if method == "pinn":
                        A = torch.cat([torch.ones(n, 1, dtype=F64), xb], 1)
                        sol = torch.linalg.lstsq(A, cc.exact(p, xb)).solution[:, 0]
                    else:
                        fi = cc.source(p, xi); gbv = cc.exact(p, xb)
                        c = torch.zeros(d + 1, dtype=F64, requires_grad=True)
                        opt = torch.optim.LBFGS([c], lr=1, max_iter=400, tolerance_grad=1e-13,
                                                tolerance_change=1e-16, history_size=50,
                                                line_search_fn="strong_wolfe")

                        def closure():
                            opt.zero_grad()
                            u_i = c[0] + xi @ c[1:]
                            u_b = c[0] + xb @ c[1:]
                            # sampled Ritz loss, as hd_core.loss_ritz: grad p = c[1:]
                            L = (0.5 * (c[1:] ** 2).sum() - (fi[:, 0] * u_i).mean()) \
                                + w * ((u_b - gbv[:, 0]) ** 2).mean() * (2 * d)
                            L.backward()
                            return L
                        opt.step(closure)
                        sol = c.detach()
                    sols.append(sol.numpy())
                mc_spread = float(np.abs(sols[0] - sols[1]).max())
                err = float(max(np.abs(s - c_ex).max() for s in sols))
                c4[f"{p}/d{d}/{method}/w{w:g}"] = dict(max_abs_diff_exact_vs_bruteforce=err,
                                                       max_abs_diff_between_two_mc_samples=mc_spread,
                                                       max_abs_coef=float(np.abs(c_ex).max()),
                                                       c0=float(c_ex[0]), c_mean=float(c_ex[1:].mean()),
                                                       c_absmax_linear=float(np.abs(c_ex[1:]).max()))
                print("C4", p, d, method, w, f"diff {err:.2e} mc {mc_spread:.2e} |c| {np.abs(c_ex).max():.3e}", flush=True)
OUT["C4_affine_minimiser"] = c4

# C5 ------------------------------------------------------------------------------------------
c5 = {}
for p in PROBS:
    for d in (2, 5, 10, 20):
        ex = fl.exact_floors(p, d)
        row = {"exact_" + k: float(v) for k, v in ex.items()}
        for tag, x in (("mc2e5", torch.rand(200_000, d, dtype=F64, generator=torch.Generator().manual_seed(4242 + d))),
                       ("test", hc.fixed_sets(d)["test"].double())):
            ue = cc.exact(p, x)
            for name, F in (("affine", x), ("lift3c", cc.feature_jet(x, "lift3c")[0]),
                            ("lift3u", cc.feature_jet(x, "lift3u")[0])):
                A = torch.cat([torch.ones(len(x), 1, dtype=F64), F], 1)
                r = ue - A @ torch.linalg.lstsq(A, ue).solution
                row[f"{tag}_{name}"] = (r.pow(2).sum() / ue.pow(2).sum()).sqrt().item()
            row[f"{tag}_const"] = ((ue - ue.mean()).pow(2).sum() / ue.pow(2).sum()).sqrt().item()
        c5[f"{p}/d{d}"] = row
        print("C5", p, d, {k: f"{v:.4e}" for k, v in row.items() if not k.startswith("exact_E")}, flush=True)
OUT["C5_floors"] = c5

json.dump(OUT, open(os.path.join(ROOT, "results", "checks.json"), "w"), indent=1)
print("written results/checks.json")
