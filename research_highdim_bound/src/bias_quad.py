"""Quadrature of Proposition 3 (the working document of the proofs) with a wide t-range (later choice, notes/VERIFICATION.md section 4).

scripts/s3_exact_bias.py integrates y = log t over [log 1e-11, log 80]. For large beta the integrand in y decays only
like t^{1/2} as t -> 0 (Q(t) ~ t^{-3/2}/alpha^2 for 1/alpha^2 << t), so the part below t = 1e-11 is not negligible when
beta is large and d is large. This version integrates over [log TLO, log THI] (default 1e-18 .. 400), splits the
interval at the peak of the integrand (found on a 2001-point grid) and uses a tighter tolerance."""
import math
import numpy as np
from scipy import integrate


def bias_sq_wide(o, d, tlo=1e-18, thi=400.0, epsrel=1e-12):
    """o = one_d(2 beta, nmax) from scripts/s3_exact_bias.py. Returns (||u_beta - u*||^2, abs. quad. error, peak t)."""
    m = d // 2
    mu_e, mu_o = o["mu_e"], o["mu_o"]
    Me2, Xe2, Do2, Xo2, DX = o["M_e"] ** 2, o["X_e"] ** 2, o["D_o"] ** 2, o["X_o"] ** 2, o["D_o"] * o["X_o"]

    def f(y):
        t = math.exp(y)
        ee = np.exp(-t * mu_e); eo = np.exp(-t * mu_o)
        g = float(Me2 @ ee)
        sDD = float(Do2 @ eo); sXX = float(Xe2 @ ee + Xo2 @ eo); sDX = float(DX @ eo)
        return t * t * 2 * (sDD * sXX + sDX ** 2) * g ** (d - 2)

    a, b = math.log(tlo), math.log(thi)
    ys = np.linspace(a, b, 2001)
    yp = float(ys[int(np.argmax([f(y) for y in ys]))])
    tot = err = 0.0
    for lo, hi in ((a, yp), (yp, b)):
        v, e = integrate.quad(f, lo, hi, limit=2000, epsrel=epsrel, epsabs=0)
        tot += v; err += e
    return m * tot, m * err, math.exp(yp)
