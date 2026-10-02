"""Exact L2(Omega) floors (the working document of the proofs, Proposition 3): relative error of the best affine function of x and of
the best affine function of the lift3 features, from one-dimensional integrals only.

For u in L2((0,1)^d) with the uniform (product) measure, the functions 1, x_i - 1/2 (i = 1..d) are orthogonal,
and so are 1 and P_k(t_i), t_i = 2 x_i - 1 (k = 1, 2, 3; i = 1..d), with ||x_i - 1/2||^2 = 1/12 and
||P_k(t_i)||^2 = 1/(2k + 1). Hence
  floor_aff^2  = 1 - [ (E u)^2 + sum_i 12 (E[(x_i - 1/2) u])^2 ] / E[u^2]
  floor_lift^2 = 1 - [ (E u)^2 + sum_i sum_k (2k + 1) (E[P_k(t_i) u])^2 ] / E[u^2].
"""
import numpy as np
from presolve import terms, _GX, _GW

LEG = [lambda x: 2 * x - 1, lambda x: 1.5 * (2 * x - 1) ** 2 - 0.5, lambda x: 2.5 * (2 * x - 1) ** 3 - 1.5 * (2 * x - 1)]


def _int(fn):
    return np.sum(_GW * fn(_GX))


def exact_floors(problem, d):
    T, _ = terms(problem, d)
    real = problem not in ("ridge", "altridge")
    # per term: 1-D integrals of h_j, h_j * w for test functions w, and products for E[u^2]
    def E_lin(w_i, i):
        """E[w(x_i) u] for a 1-D function w of coordinate i."""
        s = 0j
        for coef, hs in T:
            p = coef + 0j
            for j in range(d):
                p *= _int(lambda x, h=hs[j], j=j: h(x) * (w_i(x) if j == i else 1.0))
            s += p
        return s.real
    Eu = E_lin(lambda x: np.ones_like(x), 0)
    # E[u^2]
    Eu2 = 0.0
    for ca, ha in T:
        for cb, hb in T:
            pz = ca * cb + 0j       # E[Z_a Z_b]
            pc = ca * np.conj(cb) + 0j  # E[Z_a conj(Z_b)]
            for j in range(d):
                pz *= _int(lambda x, f=ha[j], g=hb[j]: f(x) * g(x))
                pc *= _int(lambda x, f=ha[j], g=hb[j]: f(x) * np.conj(g(x)))
            Eu2 += (pz.real if real else 0.5 * (pz + pc).real)
    lin = sum(12.0 * E_lin(lambda x: x - 0.5, i) ** 2 for i in range(d))
    lif = sum((2 * k + 3) * E_lin(LEG[k], i) ** 2 for i in range(d) for k in range(3))
    return dict(E_u=Eu, E_u2=Eu2, const=np.sqrt(max(0.0, 1 - Eu ** 2 / Eu2)),
                affine=np.sqrt(max(0.0, 1 - (Eu ** 2 + lin) / Eu2)),
                lift3=np.sqrt(max(0.0, 1 - (Eu ** 2 + lif) / Eu2)))
