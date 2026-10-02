"""Section 6.7 and Remark sA_proofs:rem:torsion: the bound of kappa_d^2 through the torsion function of the cube.

For the torsion function phi of Omega = (0,1)^d (-Lap phi = 1 in Omega, phi = 0 on the boundary) and a harmonic h,
Green's formula gives  int h^2 = -2 int phi |grad h|^2 - int_{boundary} h^2 d_n phi <= m_d ||h||^2_{boundary},
with m_d = max over the boundary of |d_n phi|.  On the cube phi(x) = int_0^inf prod_i u(t, x_i) dt, where u solves the
one-dimensional heat equation on (0,1) with u(0, .) = 1 and zero boundary values,
    u(t, x) = sum_{k odd} 4/(pi k) sin(pi k x) exp(-pi^2 k^2 t),
and, since 0 <= u(t, y) <= u(t, 1/2), the maximum of |d_n phi| is attained at the centres of the faces:
    m_d = int_0^inf u_x(t, 0) u(t, 1/2)^(d-1) dt,   u_x(t, 0) = 4 sum_{k odd} exp(-pi^2 k^2 t).
m_1 = 1/2 exactly.  For small t the theta-function form u_x(t, 0) = (pi t)^(-1/2) [1 + 4 sum_n e^(-n^2/t)
- 2 sum_n e^(-n^2/(4t))] is used.  The integral is evaluated in s = log t by the trapezoidal rule on two grids
(as a check of the quadrature) and also with scipy.integrate.quad.

    python code/s6_highdim_torsion.py          # writes data/s6_highdim_torsion.json (a few seconds, one core)

Output: data/s6_highdim_torsion.json with m_d, U_d, sqrt(m_d / U_d) and sqrt((1/2) / m_d) for the dimensions of
Table s6_highdim:tab:kappa, and the agreement of the quadratures.
"""
import json
import math
import os

import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
DS = [1, 2, 3, 5, 10, 20, 50, 100]
T0 = 1e-6     # below T0: u(t, 1/2) = 1 and u_x(t, 0) = (pi t)^(-1/2) to within exp(-1/(16 T0)) (no contribution)
T1 = 0.02     # switch from the theta-function form (t < T1) to the sine series (t >= T1) for u_x(t, 0)


def U(d):
    if d == 1:
        return 0.5
    a = math.sqrt(d - 1)
    return math.tanh(math.pi * a / 2) / (math.pi * a)


def ux0(t):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t)
    small = t < T1
    ts = t[small]
    n = np.arange(1, 8)[:, None]
    out[small] = (1 + 4 * np.exp(-n ** 2 / ts).sum(0) - 2 * np.exp(-n ** 2 / (4 * ts)).sum(0)) / np.sqrt(np.pi * ts)
    tl = t[~small]
    k = np.arange(1, 400, 2)[:, None]
    out[~small] = 4 * np.exp(-np.pi ** 2 * k ** 2 * tl).sum(0)
    return out


def u_mid(t):
    """u(t, 1/2) by the alternating sine series; for t >= T0 the terms up to exp(-60) are kept."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    kmax = int(math.sqrt(60 / (math.pi ** 2 * T0))) + 3
    j = np.arange(0, kmax // 2 + 2)[:, None]
    k = 2 * j + 1
    out = [(4 / np.pi) * ((-1.0) ** j / k * np.exp(-np.pi ** 2 * k ** 2 * c)).sum(0)
           for c in np.array_split(t, max(1, t.size // 40))]
    return np.concatenate(out)


def m_trap(d, n):
    s = np.linspace(math.log(T0), math.log(5.0), n)
    t = np.exp(s)
    f = ux0(t) * u_mid(t) ** (d - 1) * t
    return 2 * math.sqrt(T0 / math.pi) + float(np.trapezoid(f, s))


def m_quad(d):
    g = lambda s: float(ux0(math.exp(s))[0] * u_mid(math.exp(s))[0] ** (d - 1) * math.exp(s))
    val, err = quad(g, math.log(T0), math.log(5.0), limit=400, epsabs=1e-13, epsrel=1e-11)
    return 2 * math.sqrt(T0 / math.pi) + val, err


def main():
    # consistency of the two forms of u_x(t, 0) at the switch point, and of u(t, 1/2) with the image sum
    t = np.array([T1 * 0.999, T1 * 1.001])
    k = np.arange(1, 4001, 2)[:, None]
    series = 4 * np.exp(-np.pi ** 2 * k ** 2 * t).sum(0)
    theta = (1 + 4 * np.exp(-1 / t) - 2 * np.exp(-1 / (4 * t))) / np.sqrt(np.pi * t)
    switch_rel = float(np.max(np.abs(series / theta - 1)))
    from scipy.special import erf
    tt = np.array([1e-4, 1e-3, 1e-2, 0.05])
    img = sum(0.5 * (erf((0.5 - 2 * n) / (2 * np.sqrt(tt))) - erf((0.5 - 2 * n - 1) / (2 * np.sqrt(tt))))
            - 0.5 * (erf((0.5 - 2 * n + 1) / (2 * np.sqrt(tt))) - erf((0.5 - 2 * n) / (2 * np.sqrt(tt))))
            for n in range(-6, 7))
    umid_rel = float(np.max(np.abs(u_mid(tt) / img - 1)))
    rows = []
    for d in DS:
        a, b = m_trap(d, 20001), m_trap(d, 40001)
        q, qerr = m_quad(d)
        rows.append({"d": d, "m_d": b, "m_d_trap_20001": a, "m_d_quad": q, "quad_err_estimate": qerr,
                     "U_d": U(d), "sqrt_m_over_U": math.sqrt(b / U(d)), "sqrt_half_over_m": math.sqrt(0.5 / b),
                     "sqrt_half_over_U": math.sqrt(0.5 / U(d))})
        print(f"d={d:4d}  m_d={b:.6f}  (trapezoid 20001: {a:.6f}, quad: {q:.6f})  U_d={U(d):.6f}  "
              f"sqrt(m_d/U_d)={math.sqrt(b / U(d)):.3f}  sqrt((1/2)/m_d)={math.sqrt(0.5 / b):.3f}")
    out = {"source": "code/s6_highdim_torsion.py",
           "definition": "m_d = max over the boundary of |grad phi|, phi the torsion function of (0,1)^d; "
                         "kappa_d^2 <= m_d (Remark sA_proofs:rem:torsion)",
           "check_m1_equals_half": abs(rows[0]["m_d"] - 0.5),
           "check_ux0_forms_rel_diff_at_switch": switch_rel,
           "check_u_mid_against_image_sum_rel": umid_rel,
           "max_rel_diff_between_quadratures": max(max(abs(r["m_d_trap_20001"] / r["m_d"] - 1), abs(r["m_d_quad"] / r["m_d"] - 1))
                                                   for r in rows),
           "rows": rows}
    json.dump(out, open(os.path.join(ART, "data", "s6_highdim_torsion.json"), "w", encoding="utf8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k.startswith("check") or k.startswith("max")}, indent=1))
    print("written data/s6_highdim_torsion.json")


if __name__ == "__main__":
    main()
