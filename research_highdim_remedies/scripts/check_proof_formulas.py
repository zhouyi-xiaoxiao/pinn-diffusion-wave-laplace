"""Numerical check of every formula in the working document of the proofs that is not already covered by check_all.py
(results/check_proof_formulas.json). Light: no training, < 1 min.
  L1  closed-form boundary Gram matrix M: smallest eigenvalue > 0 (positive definite), d = 1..50
  P2a P4: r_f = r_g = 0, hence p = 0 (both methods)
  P2b P3: c_1 = ... = c_d = 0 (both methods, several weights)
  P2c P1, PINN: p = sum_k (x_{2k-1} + x_{2k})/2 - m/4 = the L2(Omega)-best affine fit of u*
  P3a P1 floor closed form sqrt((m/144) / (m/9 + m(m-1)/16)) against src/floors.py
  P3b P4 floors = 1; P3 affine floor = constant floor (src/floors.py)
  P3c E[u^2] closed forms: P1 m/9 + m(m-1)/16; P2 1/2; P3 (1 + phi(4/sqrt d)^d)/2 with phi(a) = sin(a)/a; P4 m/4
"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import presolve as ps
import floors as fl

out = {}
out["L1_min_eig_M"] = {d: float(np.linalg.eigvalsh(ps.boundary_gram(d)).min()) for d in range(1, 51)}
out["L1_min_eig_M_closed_form_check"] = {}
for d in (1, 2, 5, 20, 50):
    # eigenvalues derived in the working document of the proofs: (1 + 2(d-1)/3 - d/2) = (d+2)/6 with multiplicity d-1, and the two
    # eigenvalues of [[2d, d sqrt(d)], [d sqrt(d), 1 + 2(d-1)/3 + (d-1) d/2]] on span{e0, (0,1,...,1)/sqrt d}
    a = 1 + 2 * (d - 1) / 3; b = d / 2
    B = np.array([[2 * d, d * math.sqrt(d)], [d * math.sqrt(d), a + (d - 1) * b]])
    pred = sorted([*np.linalg.eigvalsh(B), *([a - b] * (d - 1))])
    out["L1_min_eig_M_closed_form_check"][d] = float(np.abs(np.array(pred) - np.linalg.eigvalsh(ps.boundary_gram(d))).max())
rec = {}
for d in (2, 5, 10, 20):
    rf, rg = ps.moments("cospair", d)
    rec[f"P4/d{d}/max|r_f|,|r_g|"] = float(max(np.abs(rf).max(), np.abs(rg).max()))
    for method, w in (("ritz", 1.0), ("ritz", 100.0), ("pinn", 1000.0)):
        rec[f"P4/d{d}/{method}/w{w:g}/max|c|"] = float(np.abs(ps.affine_minimiser("cospair", d, method, w)).max())
        rec[f"P3/d{d}/{method}/w{w:g}/max|c_lin|"] = float(np.abs(ps.affine_minimiser("ridge", d, method, w)[1:]).max())
    m = d // 2
    c = ps.affine_minimiser("laplace", d, "pinn", 1000.0)
    target = np.r_[-m / 4, [0.5] * (2 * m), [0.0] * (d - 2 * m)]
    rec[f"P1/d{d}/pinn/max|c - target|"] = float(np.abs(c - target).max())
    # L2(Omega)-best affine fit of P1: 12 E[(x_i - 1/2) u] slopes and E u - sum slopes/2 intercept
    ex = fl.exact_floors("laplace", d)
    rec[f"P1/d{d}/floor_closed_form_minus_quadrature"] = float(
        math.sqrt((m / 144) / (m / 9 + m * (m - 1) / 16)) - ex["affine"])
    rec[f"P1/d{d}/lift_floor_minus_affine_floor"] = float(ex["lift3"] - ex["affine"])
    rec[f"P1/d{d}/E_u2_closed_minus_quad"] = float(m / 9 + m * (m - 1) / 16 - ex["E_u2"])
    e2 = fl.exact_floors("poisson", d)
    rec[f"P2/d{d}/E_u2_closed_minus_quad"] = float(0.5 - e2["E_u2"])
    e3 = fl.exact_floors("ridge", d)
    a = 4 / math.sqrt(d)
    rec[f"P3/d{d}/E_u2_closed_minus_quad"] = float((1 + (math.sin(a) / a) ** d) / 2 - e3["E_u2"])
    a2 = 2 / math.sqrt(d)
    rec[f"P3/d{d}/E_u_closed_minus_quad"] = float((math.sin(a2) / a2) ** d - e3["E_u"])
    rec[f"P3/d{d}/affine_floor_minus_const_floor"] = float(e3["affine"] - e3["const"])
    e4 = fl.exact_floors("cospair", d)
    rec[f"P4/d{d}/E_u2_closed_minus_quad"] = float(m / 4 - e4["E_u2"])
    rec[f"P4/d{d}/1-affine_floor"] = float(1 - e4["affine"]); rec[f"P4/d{d}/1-lift_floor"] = float(1 - e4["lift3"])
out["P"] = rec
worst = max(abs(v) for k, v in rec.items() if not k.endswith("max|r_f|,|r_g|"))
out["max_abs_deviation_all_identities"] = worst
json.dump(out, open(os.path.join(ROOT, "results", "check_proof_formulas.json"), "w"), indent=1)
print("min eig M, d=1..50:", min(out["L1_min_eig_M"].values()), "closed-form eig check:", out["L1_min_eig_M_closed_form_check"])
print("worst identity deviation:", worst)
for k, v in rec.items():
    if abs(v) > 1e-10: print("LARGE", k, v)
