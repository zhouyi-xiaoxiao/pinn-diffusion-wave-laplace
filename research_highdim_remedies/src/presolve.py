"""Exact minimiser of the method's own loss over affine functions p(x) = c0 + c.x on (0,1)^d.

Population losses (the training losses are Monte-Carlo estimates of these; |Omega| = 1, |dOmega| = 2d):
  Deep Ritz : R(p) = 1/2 int_Omega |grad p|^2 - int_Omega f p + beta int_dOmega (p - g)^2
  PINN      : P(p) = int_Omega (Lap p + f)^2 + lambda/(2d) int_dOmega (p - g)^2
With phi = (1, x_1, ..., x_d), M = int_dOmega phi phi^T, r_f = int_Omega f phi, r_g = int_dOmega g phi and
E = diag(0, 1, ..., 1), the minimisers solve (the working document of the proofs, Proposition 1)
  Deep Ritz : (E + 2 beta M) c = r_f + 2 beta r_g
  PINN      : M c = r_g                      (independent of lambda and of f, since Lap p = 0)
M has the closed form of the working document of the proofs, Lemma 1. r_f and r_g are computed exactly up to quadrature round-off:
every u* and f used here is a finite sum of products of one-dimensional functions (P3 and P5 as the real
part of one complex product), so each integral over the cube or over a face is a product of 1-D integrals,
done by 64-point Gauss-Legendre quadrature in float64/complex128.
"""
import math
import numpy as np

_GX, _GW = np.polynomial.legendre.leggauss(64)
_GX = 0.5 * (_GX + 1.0)
_GW = 0.5 * _GW


def _I(h, k):
    """int_0^1 x^k h(x) dx, k in {0, 1}."""
    return np.sum(_GW * _GX ** k * h(_GX))


def terms(problem, d):
    """u* = Re sum_T coef_T prod_j h_{T,j}(x_j); f = kappa u*. Returns (list of (coef, [h_1..h_d])), kappa."""
    one = lambda x: np.ones_like(x, dtype=float)
    T = []
    if problem == "laplace":
        for k in range(d // 2):
            hs = [one] * d
            hs = list(hs); hs[2 * k] = lambda x: x; hs[2 * k + 1] = lambda x: x
            T.append((1.0, hs))
        kappa = 0.0
    elif problem == "poisson":
        for i in range(d):
            hs = [one] * d
            hs = list(hs); hs[i] = lambda x: np.cos(math.pi * x)
            T.append((1.0 / math.sqrt(d), hs))
        kappa = math.pi ** 2
    elif problem == "ridge":
        a = 2.0 / math.sqrt(d)
        T.append((1.0, [lambda x, a=a: np.exp(1j * a * (2.0 * x - 1.0))] * d))
        kappa = 16.0
    elif problem == "cospair":
        for k in range(d // 2):
            hs = [one] * d
            hs = list(hs); hs[2 * k] = lambda x: np.cos(math.pi * x); hs[2 * k + 1] = lambda x: np.cos(math.pi * x)
            T.append((1.0, hs))
        kappa = 2.0 * math.pi ** 2
    elif problem == "altridge":   # P5 (PREREG_C1_P5.md): Re[e^{i} prod_j exp(i pi e_j (2x_j - 1)/sqrt(d))]
        a = math.pi / math.sqrt(d)
        hs = [(lambda x, a=a * (1.0 if j % 2 == 0 else -1.0): np.exp(1j * a * (2.0 * x - 1.0))) for j in range(d)]
        T.append((complex(math.cos(1.0), math.sin(1.0)), hs))
        kappa = 4.0 * math.pi ** 2
    else:
        raise ValueError(problem)
    return T, kappa


def boundary_gram(d):
    """M = int_dOmega phi phi^T in closed form (the working document of the proofs, Lemma 1)."""
    M = np.empty((d + 1, d + 1))
    M[0, 0] = 2.0 * d
    M[0, 1:] = M[1:, 0] = float(d)
    M[1:, 1:] = d / 2.0
    np.fill_diagonal(M[1:, 1:], 1.0 + 2.0 * (d - 1) / 3.0)
    return M


def moments(problem, d):
    """r_f = int_Omega f phi and r_g = int_dOmega g phi (g = u* on the boundary)."""
    T, kappa = terms(problem, d)
    r_u = np.zeros(d + 1, dtype=complex)   # int_Omega u* phi
    r_g = np.zeros(d + 1, dtype=complex)
    for coef, hs in T:
        I0 = np.array([_I(h, 0) for h in hs], dtype=complex)
        I1 = np.array([_I(h, 1) for h in hs], dtype=complex)
        H0 = np.array([h(np.array([0.0]))[0] for h in hs], dtype=complex)
        H1 = np.array([h(np.array([1.0]))[0] for h in hs], dtype=complex)

        def prod_except(skip):
            p = 1.0 + 0j
            for j in range(d):
                if j not in skip:
                    p *= I0[j]
            return p

        # interior
        r_u[0] += coef * prod_except(())
        for i in range(d):
            r_u[1 + i] += coef * I1[i] * prod_except((i,))
        # faces x_k = s, s in {0, 1}
        for k in range(d):
            for s, Hs in ((0.0, H0[k]), (1.0, H1[k])):
                base = coef * Hs
                r_g[0] += base * prod_except((k,))
                for i in range(d):
                    if i == k:
                        r_g[1 + i] += base * s * prod_except((k,))
                    else:
                        r_g[1 + i] += base * I1[i] * prod_except((k, i))
    return kappa * r_u.real, r_g.real


def affine_minimiser(problem, d, method, w):
    """Coefficients (c0, c_1..c_d) of the exact affine minimiser of the method's population loss."""
    M = boundary_gram(d)
    r_f, r_g = moments(problem, d)
    if method == "ritz":
        E = np.eye(d + 1); E[0, 0] = 0.0
        return np.linalg.solve(E + 2.0 * w * M, r_f + 2.0 * w * r_g)
    if method == "pinn":
        return np.linalg.solve(M, r_g)
    raise ValueError(method)
