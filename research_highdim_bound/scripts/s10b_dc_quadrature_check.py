"""Quadrature check for scripts/s10_dc_large_d.py: the t-integral of Proposition 3 with other integration limits
(t from 1e-15 or 1e-11 to 80 or 400) and a tighter tolerance, at d in {1000, 10000}, beta = 1e9, 2e5 roots.
Also splits the integral at its peak so that the adaptive rule cannot miss it. Output: results/dc_quadrature_check.json"""
import math, os, sys
import numpy as np
from scipy import integrate
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2common import dump
from s3_exact_bias import one_d, ustar_norm

def dc(d, beta, nmax, tlo, thi, eps, split):
    o = one_d(2 * beta, nmax)
    m = d // 2
    mu_e, mu_o = o["mu_e"], o["mu_o"]
    Me2, Xe2, Do2, Xo2, DX = o["M_e"] ** 2, o["X_e"] ** 2, o["D_o"] ** 2, o["X_o"] ** 2, o["D_o"] * o["X_o"]
    def f(y):
        t = math.exp(y); ee = np.exp(-t * mu_e); eo = np.exp(-t * mu_o)
        g = float(Me2 @ ee); sDD = float(Do2 @ eo); sXX = float(Xe2 @ ee + Xo2 @ eo); sDX = float(DX @ eo)
        return t * t * 2 * (sDD * sXX + sDX ** 2) * g ** (d - 2)
    ys = np.linspace(math.log(tlo), math.log(thi), 4001)
    fv = np.array([f(y) for y in ys])
    pts = [math.log(tlo)] + ([float(ys[np.argmax(fv)])] if split else []) + [math.log(thi)]
    tot, err = 0.0, 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        v, e = integrate.quad(f, a, b, limit=2000, epsrel=eps, epsabs=0); tot += v; err += e
    trap = float(np.trapezoid(fv, ys))
    rel = math.sqrt(m * tot) / ustar_norm(d)
    return dict(d=d, beta=beta, nmax=nmax, tlo=tlo, thi=thi, epsrel=eps, split=split, d_beta_rel=d * beta * rel,
                d_beta_rel_trapezoid_4001=d * beta * math.sqrt(m * trap) / ustar_norm(d),
                peak_t=math.exp(float(ys[np.argmax(fv)])), quad_err_rel=err / tot)

rows = []
for d in (1000, 10000):
    for tlo, thi, eps, split in ((1e-11, 80.0, 1e-10, False), (1e-15, 80.0, 1e-12, True), (1e-15, 400.0, 1e-12, True)):
        r = dc(d, 1e9, 200_000, tlo, thi, eps, split); rows.append(r); print(r, flush=True)
dump("dc_quadrature_check.json", rows)
