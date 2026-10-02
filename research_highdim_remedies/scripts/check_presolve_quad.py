"""C4b: deterministic brute-force check of the affine minimisers for d = 2, 3 (results/check_presolve_quad.json).
The population losses are discretised by tensor-product Gauss-Legendre quadrature (interior: 40^d nodes;
each face: 40^(d-1) nodes), written as an explicit weighted least-squares / quadratic problem in the
coefficients and solved by torch.linalg.lstsq (PINN) or by the stationarity system assembled from the
quadrature (Deep Ritz). Independent of src/presolve.py except for the problem definitions in c1core."""
import sys, os, json, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import torch
import c1core as cc
import presolve as ps

F64 = torch.float64
gx, gw = np.polynomial.legendre.leggauss(40)
gx = 0.5 * (gx + 1); gw = 0.5 * gw
out = {}
for d in (2, 3):
    pts = np.array(list(itertools.product(gx, repeat=d)))
    wts = np.prod(np.array(list(itertools.product(gw, repeat=d))), 1)
    fpts, fw = [], []
    sub = np.array(list(itertools.product(gx, repeat=d - 1)))
    subw = np.prod(np.array(list(itertools.product(gw, repeat=d - 1))), 1)
    for k in range(d):
        for s in (0.0, 1.0):
            P = np.insert(sub, k, s, axis=1)
            fpts.append(P); fw.append(subw)
    fpts = np.concatenate(fpts); fw = np.concatenate(fw)
    xi = torch.tensor(pts, dtype=F64); wi = torch.tensor(wts, dtype=F64)
    xb = torch.tensor(fpts, dtype=F64); wb = torch.tensor(fw, dtype=F64)
    Phi_i = torch.cat([torch.ones(len(xi), 1, dtype=F64), xi], 1)
    Phi_b = torch.cat([torch.ones(len(xb), 1, dtype=F64), xb], 1)
    for p in ("laplace", "poisson", "ridge", "cospair"):
        f = cc.source(p, xi)[:, 0]; g = cc.exact(p, xb)[:, 0]
        # PINN: minimise sum_b wb (Phi_b c - g)^2  -> weighted least squares
        sq = wb.sqrt()
        c_pinn = torch.linalg.lstsq(Phi_b * sq[:, None], (g * sq)[:, None]).solution[:, 0].numpy()
        rec = {"pinn": float(np.abs(c_pinn - ps.affine_minimiser(p, d, "pinn", 1000.0)).max())}
        for beta in (1.0, 100.0):
            E = torch.eye(d + 1, dtype=F64); E[0, 0] = 0
            H = E + 2 * beta * (Phi_b * wb[:, None]).t() @ Phi_b
            b = (Phi_i * (wi * f)[:, None]).sum(0) + 2 * beta * (Phi_b * (wb * g)[:, None]).sum(0)
            c_r = torch.linalg.solve(H, b).numpy()
            rec[f"ritz_w{beta:g}"] = float(np.abs(c_r - ps.affine_minimiser(p, d, "ritz", beta)).max())
        out[f"{p}/d{d}"] = rec
        print(p, d, rec)
json.dump(out, open(os.path.join(ROOT, "results", "check_presolve_quad.json"), "w"), indent=1)
