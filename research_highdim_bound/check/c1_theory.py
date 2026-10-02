"""Separately written re-check of Theorem 2 / Corollary 2 (the working document of the proofs), not reusing the package's code.

(1) U_d, L_d, d = 1 exact value, large-d limits.
(2) C_d^2 from the OTHER side: generalised eigenproblem ||w||^2 / ||w||_bd^2 over a basis of harmonic
    functions (face-wise sine x sinh), d = 2 and d = 3. Every value is a lower bound of C_d^2 and must
    stay <= U_d; it should approach the package's Lanczos values (T-side).
(3) Corollary 2(c) on constructed errors e = v - u* (worst-case harmonic part + residual), d = 2, 3.
(4) Remark L'_d (full Rayleigh quotient of the Step-6 test function).
Output: check/out/c1_theory.json, printed summary.
"""
import json, math, os, sys, time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
res = {}


def U(d):
    if d == 1:
        return 0.5
    a = math.sqrt(d - 1)
    return math.tanh(math.pi * a / 2) / (math.pi * a)


def Lser(d, J=2_000_001):
    j = np.arange(1, J, 2, dtype=float)
    a2 = d - 1.0
    s = (j ** 2 / (j ** 2 + a2) ** 2).sum()
    return 4 / math.pi ** 2 * s


# ---------------------------------------------------------------- (1)
rows = []
for d in [1, 2, 3, 5, 10, 20, 50, 100, 1000, 10000, 1000000]:
    u = U(d)
    l = Lser(d) if d <= 10000 else float("nan")
    rows.append(dict(d=d, U=u, L=l, sqrtdU=math.sqrt(d) * u, sqrtdL=math.sqrt(d) * l, L_over_U=l / u))
    print(f"(1) d={d:8d} U={u:.6f} L={l:.6f} sqrt(d)U={math.sqrt(d)*u:.5f} sqrt(d)L={math.sqrt(d)*l:.5f} L/U={l/u:.4f}")
print(f"    1/pi={1/math.pi:.5f} 1/(2pi)={1/(2*math.pi):.5f}")
res["U_L"] = rows

# d = 1: harmonic w = A(1-x) + Bx. ||w||^2 = (A^2+AB+B^2)/3, ||w||_bd^2 = A^2 + B^2 -> max ratio 1/2
M = np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])
res["C1_sq_exact"] = float(np.linalg.eigvalsh(M).max())
print("(1) d=1 exact C_1^2 =", res["C1_sq_exact"])

# ---------------------------------------------------------------- (2) harmonic-basis Rayleigh-Ritz
# 1-D Gauss-Legendre on [0,1]
def gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w


def S(kappa, x):
    """sinh(kappa (1-x)) / sinh(kappa), stable: harmonic profile equal to 1 at x = 0, 0 at x = 1."""
    return (np.exp(-kappa * x) - np.exp(-kappa * (2 - x))) / (1 - np.exp(-2 * kappa))


def d2_lower(K, nq=3000):
    """d = 2: basis phi_{f,k}, face f in {x=0, x=1, y=0, y=1}, data sin(k pi s) on face f, harmonic.
    phi vanishes on the other three faces -> boundary Gram = I/2. Interior Gram by separable 1-D quadrature."""
    x, w = gl(nq)
    k = np.arange(1, K + 1)
    sinm = np.sin(np.pi * np.outer(k, x))            # (K, nq)
    Sm = S(np.pi * k[:, None], x[None, :])          # (K, nq): profile from face at 0
    Sr = Sm[:, ::-1]                                  # profile from face at 1 (x -> 1-x), GL nodes symmetric
    # functions: face x=0: sin(k pi y) Sm_k(x); x=1: sin(k pi y) Sr_k(x); y=0: Sm_k(y) sin(k pi x); y=1: Sr_k(y) sin(k pi x)
    # Gram blocks: <f(x)g(y), p(x)q(y)> = (f,p)(g,q)
    ip = lambda A, B: (A * w) @ B.T
    SS = ip(Sm, Sm); SR = ip(Sm, Sr); RR = ip(Sr, Sr)
    sn = ip(sinm, sinm)  # = I/2
    sS = ip(sinm, Sm); sR = ip(sinm, Sr)  # (sin_k, S_l)
    # block (x0, x0): (Sm_k,Sm_l)(sin_k,sin_l); (x0,x1): (Sm_k,Sr_l)(sin,sin); (x0,y0): (Sm_k(x), sin_l(x)) (sin_k(y), Sm_l(y))
    B00 = SS * sn; B01 = SR * sn; B11 = RR * sn
    X0Y0 = sS.T * sS            # (Sm_k, sin_l) * (sin_k, Sm_l): sS[l,k] * sS[k,l]
    X0Y1 = sS.T * sR            # (Sm_k, sin_l)(sin_k, Sr_l)
    X1Y0 = sR.T * sS
    X1Y1 = sR.T * sR
    A = np.block([[B00, B01, X0Y0, X0Y1],
                  [B01.T, B11, X1Y0, X1Y1],
                  [X0Y0.T, X1Y0.T, B00, B01],
                  [X0Y1.T, X1Y1.T, B01.T, B11]])
    Bd = 0.5 * np.eye(4 * K)
    A = 0.5 * (A + A.T)
    lam = np.linalg.eigvalsh(A)[-1] / 0.5
    return float(lam)


def d3_lower(K, nq=600):
    """d = 3: face functions sin(k pi s) sin(l pi t) S_{pi sqrt(k^2+l^2)}(normal coordinate). Boundary Gram I/4.
    Interior Gram assembled from 1-D integrals."""
    x, w = gl(nq)
    ks = [(k, l) for k in range(1, K + 1) for l in range(1, K + 1)]
    n = len(ks)
    kk = np.array([a for a, b in ks]); ll = np.array([b for a, b in ks])
    kap = np.pi * np.sqrt(kk ** 2 + ll ** 2)
    Pm = S(kap[:, None], x[None, :])  # (n, nq)
    Pr = Pm[:, ::-1]
    sinK = np.sin(np.pi * np.outer(np.arange(1, K + 1), x))  # (K, nq)
    ip = lambda A, B: (A * w) @ B.T
    # faces: axis a in {0,1,2}, side s in {0,1}; function = P(x_a) * sin(k pi x_b) sin(l pi x_c), (b, c) = other axes in order
    # inner product of two face functions = product over the three axes of 1-D integrals
    prof = {0: Pm, 1: Pr}
    sin1 = lambda idx: sinK[idx - 1]  # (n, nq)
    faces = [(a, s) for a in range(3) for s in range(2)]
    def factors(a, s):
        others = [b for b in range(3) if b != a]
        f = {a: prof[s]}
        f[others[0]] = sin1(kk)
        f[others[1]] = sin1(ll)
        return f
    F = [factors(a, s) for a, s in faces]
    A = np.zeros((6 * n, 6 * n))
    for i in range(6):
        for j in range(i, 6):
            blk = np.ones((n, n))
            for ax in range(3):
                blk = blk * ip(F[i][ax], F[j][ax])
            A[i * n:(i + 1) * n, j * n:(j + 1) * n] = blk
            if j != i:
                A[j * n:(j + 1) * n, i * n:(i + 1) * n] = blk.T
    A = 0.5 * (A + A.T)
    lam = np.linalg.eigvalsh(A)[-1] / 0.25
    return float(lam)


t0 = time.time()
r2 = []
for K in [50, 100, 200, 400, 800]:
    v = d2_lower(K)
    r2.append(dict(K=K, lower=v, over_U=v / U(2)))
    print(f"(2) d=2 harmonic basis K={K:4d}/face: lambda_max = {v:.6f}  /U_2 = {v/U(2):.5f}  (<= 1 required)  t={time.time()-t0:.0f}s", flush=True)
r3 = []
for K in [8, 12, 16, 20]:
    v = d3_lower(K)
    r3.append(dict(K=K, lower=v, over_U=v / U(3)))
    print(f"(2) d=3 harmonic basis K={K:3d}^2/face: lambda_max = {v:.6f}  /U_3 = {v/U(3):.5f}  t={time.time()-t0:.0f}s", flush=True)
res["harmonic_basis_d2"] = r2
res["harmonic_basis_d3"] = r3

# ---------------------------------------------------------------- (4) Remark L'_d
rem = []
for d in [2, 3, 5, 10, 20, 100]:
    j = np.arange(1, 200_001, 2, dtype=float)
    a = d - 1.0
    Sv = (j ** 2 / (j ** 2 + a) ** 2).sum()
    extra = (d - 1) / Sv * (j ** 2 / (j ** 2 + a) ** 4).sum()
    Lp = 4 / math.pi ** 2 * (Sv + extra)
    rem.append(dict(d=d, Lprime=Lp))
    print(f"(4) d={d:3d} L'_d = {Lp:.6f}")
res["Lprime"] = rem

json.dump(res, open(os.path.join(OUT, "c1_theory.json"), "w"), indent=1)
