"""Round-2 check of Theorem 2 numbers and edge cases (own code).
- d = 1: C_1^2 by the exact generalized eigenproblem on harmonic (linear) functions.
- L_d (closed-form tail), U_d, 1/(2d) crossover incl. d = 9 (not stated in the theorem).
- Payne factor sqrt((1/2)/U_d), article/Payne factor.
- d = 2 lower bound for C_2^2 from an extension-side Rayleigh-Ritz on face-wise harmonic modes
  (Gauss-Legendre quadrature), as an independent value to compare with the Lanczos 0.28746.
"""
import json, math, sys
import numpy as np
from scipy import linalg, special

def U(d):
    if d == 1: return 0.5
    a = math.sqrt(d - 1); return math.tanh(math.pi * a / 2) / (math.pi * a)

def L(d, J=2_000_001):
    j = np.arange(1, J + 1, 2, dtype=float)
    s = np.sum(j * j / (j * j + d - 1) ** 2)
    # tail ~ sum_{j odd > J} 1/j^2 (leading) = psi1((J+2)/2)/4
    s += special.polygamma(1, (J + 2) / 2) / 4
    return 4 / math.pi ** 2 * s

out = {}
# d = 1: harmonic h = a + b x, ||h||^2 = a^2 + ab + b^2/3, boundary: h(0)^2 + h(1)^2
A = np.array([[1, .5], [.5, 1 / 3]]); B = np.array([[2, 1], [1, 1]])
ev = linalg.eigh(A, B, eigvals_only=True)
out["d1_C2"] = float(ev.max()); print("d=1 C^2 =", ev.max())
rows = []
for d in list(range(1, 13)) + [20, 50, 100, 1000]:
    Ld, Ud = L(d), U(d)
    rows.append(dict(d=d, L=float(Ld), U=Ud, half_over_d=1 / (2 * d), L_beats_elem=bool(Ld > 1 / (2 * d)), L_val=float(Ld),
                     payne_factor=math.sqrt(0.5 / Ud), article_over_payne=2 * math.sqrt(1 + 1 / (d * math.pi ** 2)) / math.sqrt(0.5),
                     sqrt2dU=math.sqrt(2 * d * Ud), sqrtd_U=math.sqrt(d) * Ud, sqrtd_L=float(math.sqrt(d) * Ld)))
    print(rows[-1])
out["rows"] = rows

# d = 2 extension-side Rayleigh-Ritz: harmonic modes with sine data on one face each
def rr_d2(K, nq=400):
    xq, wq = np.polynomial.legendre.leggauss(nq); xq = (xq + 1) / 2; wq = wq / 2
    X, Y = np.meshgrid(xq, xq, indexing="ij"); W = np.outer(wq, wq)
    funcs = []
    for k in range(1, K + 1):
        s = np.sin(math.pi * k * Y)
        # face x=0 data sin(pi k y): sinh(pi k (1-x))/sinh(pi k) * sin(pi k y)  (stable form)
        fx0 = np.exp(-math.pi * k * X) * (1 - np.exp(-2 * math.pi * k * (1 - X))) / (1 - np.exp(-2 * math.pi * k)) * s
        fx1 = np.exp(-math.pi * k * (1 - X)) * (1 - np.exp(-2 * math.pi * k * X)) / (1 - np.exp(-2 * math.pi * k)) * s
        sx = np.sin(math.pi * k * X)
        fy0 = np.exp(-math.pi * k * Y) * (1 - np.exp(-2 * math.pi * k * (1 - Y))) / (1 - np.exp(-2 * math.pi * k)) * sx
        fy1 = np.exp(-math.pi * k * (1 - Y)) * (1 - np.exp(-2 * math.pi * k * Y)) / (1 - np.exp(-2 * math.pi * k)) * sx
        funcs += [fx0, fx1, fy0, fy1]
    F = np.array([f.ravel() for f in funcs]); Wv = W.ravel()
    Gin = (F * Wv) @ F.T
    Gbd = np.eye(len(funcs)) * 0.5  # each mode has data sin(pi k s) on exactly one face, ||.||^2 = 1/2
    ev = linalg.eigh(Gin, Gbd, eigvals_only=True)
    return float(ev.max())
rr = {}
for K in (10, 40):
    rr[K] = rr_d2(K); print("d=2 extension RR K =", K, rr[K], "ratio to U_2", rr[K] / U(2), flush=True)
out["d2_extension_RR"] = rr
json.dump(out, open(sys.argv[1], "w"), indent=1)
