"""Checks for problem P5 (PREREG_C1_P5.md), run before any P5 training run. No training.
Writes results/checks_p5.json. float64 throughout.

 C1  source: f = -Lap u* by nested reverse-mode autograd, d in {2, 5, 10, 20}
 C4b affine minimisers (src/presolve.py) against tensor-product Gauss-Legendre brute force, d = 2, 3
     (same construction as scripts/check_presolve_quad.py)
 C4  affine minimisers against brute force on two independent 10^6-point Monte-Carlo samples, d = 20
     (PINN: least squares on the boundary sample; Deep Ritz: the quadratic of the sampled loss solved exactly)
 C5  floors: exact one-dimensional formulas (src/floors.py) against least squares on the held-out test set
 C6  the edits that added P5 leave P1-P4 unchanged: exact(), source(), affine minimisers and floors of P1-P4
     equal the values stored in results/checks.json before the edit
"""
import sys, os, json, itertools
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
P = "altridge"
OUT = {}


def nested_lap(fn, x):
    x = x.clone().requires_grad_(True)
    u = fn(x)
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        lap = lap + torch.autograd.grad(g[:, i].sum(), x, create_graph=True)[0][:, i:i + 1]
    return u.detach(), lap.detach()


c1 = {}
for d in (2, 5, 10, 20):
    x = torch.rand(256, d, dtype=F64, generator=torch.Generator().manual_seed(d))
    u, lap = nested_lap(lambda z: cc.exact(P, z), x)
    f = cc.source(P, x)
    # independent formula for u*
    e = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
    s = ((2 * x.numpy() - 1) * e).sum(1) / np.sqrt(d)
    c1[f"d{d}"] = dict(max_abs_diff=(f + lap).abs().max().item(), max_abs_f=f.abs().max().item(),
                       max_abs_diff_u_vs_numpy=float(np.abs(u[:, 0].numpy() - np.cos(np.pi * s + 1)).max()))
OUT["C1_source"] = c1
print("C1", c1, flush=True)

gx, gw = np.polynomial.legendre.leggauss(40)
gx = 0.5 * (gx + 1); gw = 0.5 * gw
c4b = {}
for d in (2, 3):
    pts = np.array(list(itertools.product(gx, repeat=d)))
    wts = np.prod(np.array(list(itertools.product(gw, repeat=d))), 1)
    sub = np.array(list(itertools.product(gx, repeat=d - 1)))
    subw = np.prod(np.array(list(itertools.product(gw, repeat=d - 1))), 1)
    fpts, fw = [], []
    for k in range(d):
        for sv in (0.0, 1.0):
            fpts.append(np.insert(sub, k, sv, axis=1)); fw.append(subw)
    xi = torch.tensor(pts, dtype=F64); wi = torch.tensor(wts, dtype=F64)
    xb = torch.tensor(np.concatenate(fpts), dtype=F64); wb = torch.tensor(np.concatenate(fw), dtype=F64)
    Phi_i = torch.cat([torch.ones(len(xi), 1, dtype=F64), xi], 1)
    Phi_b = torch.cat([torch.ones(len(xb), 1, dtype=F64), xb], 1)
    f = cc.source(P, xi)[:, 0]; g = cc.exact(P, xb)[:, 0]
    sq = wb.sqrt()
    c_pinn = torch.linalg.lstsq(Phi_b * sq[:, None], (g * sq)[:, None]).solution[:, 0].numpy()
    rec = {"pinn": float(np.abs(c_pinn - ps.affine_minimiser(P, d, "pinn", 1000.0)).max())}
    for beta in (1.0, 10.0, 100.0, 1000.0):
        E = torch.eye(d + 1, dtype=F64); E[0, 0] = 0
        H = E + 2 * beta * (Phi_b * wb[:, None]).t() @ Phi_b
        b = (Phi_i * (wi * f)[:, None]).sum(0) + 2 * beta * (Phi_b * (wb * g)[:, None]).sum(0)
        rec[f"ritz_w{beta:g}"] = float(np.abs(torch.linalg.solve(H, b).numpy() - ps.affine_minimiser(P, d, "ritz", beta)).max())
    c4b[f"d{d}"] = rec
OUT["C4b_presolve_quadrature"] = c4b
print("C4b", c4b, flush=True)

c4 = {}
d = 20
for method, ws in (("ritz", (1.0, 10.0, 100.0, 1000.0)), ("pinn", (1000.0,))):
    for w in ws:
        c_ex = ps.affine_minimiser(P, d, method, w)
        sols = []
        for rep in range(2):
            gen = torch.Generator().manual_seed(95_000 + 100 * d + rep)
            n = 1_000_000
            xi = torch.rand(n, d, generator=gen, dtype=F64)
            xb = hc.sample_boundary(n, d, gen).double()
            Ab = torch.cat([torch.ones(n, 1, dtype=F64), xb], 1)
            gb = cc.exact(P, xb)
            if method == "pinn":
                sol = torch.linalg.lstsq(Ab, gb).solution[:, 0]
            else:
                # sampled loss 1/2|c'|^2 - mean(f p) + w 2d mean_b (p - g)^2 is quadratic in c: solve exactly
                Ai = torch.cat([torch.ones(n, 1, dtype=F64), xi], 1)
                E = torch.eye(d + 1, dtype=F64); E[0, 0] = 0
                H = E + 2 * w * (2 * d) * Ab.t() @ Ab / n
                rhs = (Ai * cc.source(P, xi)).mean(0) + 2 * w * (2 * d) * (Ab * gb).mean(0)
                sol = torch.linalg.solve(H, rhs)
            sols.append(sol.numpy())
        c4[f"d20/{method}/w{w:g}"] = dict(
            max_abs_diff_exact_vs_bruteforce=float(max(np.abs(s_ - c_ex).max() for s_ in sols)),
            max_abs_diff_between_two_mc_samples=float(np.abs(sols[0] - sols[1]).max()),
            max_abs_coef=float(np.abs(c_ex).max()), c0=float(c_ex[0]),
            c_odd_mean=float(c_ex[1::2].mean()), c_even_mean=float(c_ex[2::2].mean()))
        print("C4", method, w, c4[f"d20/{method}/w{w:g}"], flush=True)
OUT["C4_presolve_montecarlo"] = c4

c5 = {}
for d in (5, 10, 20):
    ex = fl.exact_floors(P, d)
    row = {"exact_" + k: float(v) for k, v in ex.items()}
    x = hc.fixed_sets(d)["test"].double()
    ue = cc.exact(P, x)
    for name, F in (("affine", x), ("lift3c", cc.feature_jet(x, "lift3c")[0])):
        A = torch.cat([torch.ones(len(x), 1, dtype=F64), F], 1)
        r = ue - A @ torch.linalg.lstsq(A, ue).solution
        row[f"test_{name}"] = (r.pow(2).sum() / ue.pow(2).sum()).sqrt().item()
    row["test_const"] = ((ue - ue.mean()).pow(2).sum() / ue.pow(2).sum()).sqrt().item()
    c5[f"d{d}"] = row
    print("C5", d, row, flush=True)
OUT["C5_floors"] = c5

# C6: P1-P4 unchanged by the edit (values stored in results/checks.json before it)
old = json.load(open(os.path.join(ROOT, "results", "checks.json")))
c6 = {}
for p in ("laplace", "poisson", "ridge", "cospair"):
    for d in (5, 20):
        ex = fl.exact_floors(p, d)
        o = old["C5_floors"][f"{p}/d{d}"]
        c6[f"floors/{p}/d{d}"] = max(abs(float(ex[k]) - o["exact_" + k]) for k in ("affine", "lift3", "const"))
        for method, w in (("ritz", 100.0), ("pinn", 1000.0)):
            c_ex = ps.affine_minimiser(p, d, method, w)
            o4 = old["C4_affine_minimiser"][f"{p}/d{d}/{method}/w{w:g}"]
            c6[f"presolve/{p}/d{d}/{method}"] = max(abs(float(c_ex[0]) - o4["c0"]),
                                                     abs(float(c_ex[1:].mean()) - o4["c_mean"]))
    x = torch.rand(256, 20, dtype=F64, generator=torch.Generator().manual_seed(20))
    u, lap = nested_lap(lambda z: cc.exact(p, z), x)
    c6[f"source/{p}/d20"] = (cc.source(p, x) + lap).abs().max().item()
OUT["C6_P1_P4_unchanged"] = c6
OUT["C6_max"] = max(c6.values())
print("C6 max", OUT["C6_max"], flush=True)
json.dump(OUT, open(os.path.join(ROOT, "results", "checks_p5.json"), "w"), indent=1)
print("written results/checks_p5.json")
