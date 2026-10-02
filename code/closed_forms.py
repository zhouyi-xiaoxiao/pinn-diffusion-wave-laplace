"""Closed-form quantities quoted in the article (no training, runs in < 1 s).

Writes ../data/closed_forms.json.  Quantities:
  * the relative L2((0,1) x Omega) distance between the decaying mode v = exp(-t) S, with
    S = sin(pi x) sin(pi y), and the solution cos(sqrt(2) pi t) S of problem Wave2 (Proposition
    s7_pitfalls:prop:notwave; the spatial factor S cancels, so this is a 1-D integral);
  * trivial-predictor floors for the d-dimensional problems P1 and P2 (Proposition "floors"),
    cross-checked by Monte Carlo;
  * the factor exp(pi^2) by which an absolute error is amplified into a relative error on the
    t = 1 slice of the heat problem;
  * ||d_n u*||^2 on the boundary of the unit cube for P1 (= 4 m / 3), used in the Robin-bias bound.
Usage:  python closed_forms.py
"""
import json, os
import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "closed_forms.json")
w = np.sqrt(2.0) * np.pi                      # angular frequency of the (1,1) mode on (-1,1)^2


def rel(f, g):
    num = quad(lambda t: (f(t) - g(t)) ** 2, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13)[0]
    den = quad(lambda t: g(t) ** 2, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13)[0]
    return float(np.sqrt(num / den))


target = lambda t: np.exp(-t)                               # the decaying mode / S
exact_v0 = lambda t: np.cos(w * t)                          # u(0)=S, u_t(0)=0

res = {
    "wave2d": {
        "omega": float(w),
        "rel_L2_target_vs_exact_ut0_zero": rel(target, exact_v0),
        "exact_factor_at_t_1_2_3": [float(np.cos(w * t)) for t in (1, 2, 3)],
        "exp_minus_t_at_t_1_2_3": [float(np.exp(-t)) for t in (1, 2, 3)],
    },
    "heat1d": {"exp_minus_pi2": float(np.exp(-np.pi ** 2)), "exp_plus_pi2": float(np.exp(np.pi ** 2))},
}

# ---- floors (closed form + Monte Carlo cross-check) -------------------------------------
rng = np.random.default_rng(0)
floors = {}
for d in (2, 3, 5, 10, 20):
    m = d // 2
    x = rng.random((400_000, d))
    A = np.hstack([np.ones((x.shape[0], 1)), x])
    u1 = sum(x[:, 2 * k] * x[:, 2 * k + 1] for k in range(m))
    u2 = np.cos(np.pi * x).sum(1) / np.sqrt(d)
    mc = {}
    for name, u in (("P1", u1), ("P2", u2)):
        c = np.sqrt(np.mean((u - u.mean()) ** 2) / np.mean(u ** 2))
        coef = np.linalg.lstsq(A, u, rcond=None)[0]
        a = np.sqrt(np.mean((u - A @ coef) ** 2) / np.mean(u ** 2))
        mc[name] = {"const_mc": float(c), "affine_mc": float(a)}
    floors[str(d)] = {
        "m": m,
        "P1_const": float(np.sqrt(7.0 / (7.0 + 9.0 * m))),
        "P1_affine": float(np.sqrt(1.0 / (7.0 + 9.0 * m))),
        "P2_const": 1.0,
        "P2_affine": float(np.sqrt(1.0 - 96.0 / np.pi ** 4)),
        "P1_normal_derivative_sq_norm": 4.0 * m / 3.0,
        "monte_carlo": mc,
    }
res["floors"] = floors

json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
