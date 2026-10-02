"""Separately written re-check of Proposition 3 (exact penalty bias of P1), Theorem 1 and the conjecture
on d c(d). Written from the statements in the working document of the proofs, not from scripts/s3_exact_bias.py.

(A) 1-D Robin eigenpairs by vectorised bisection; ||w||^2 = m int_0^inf t Q(t) g(t)^(d-2) dt evaluated
    with the substitution t = exp(s) and the trapezoidal rule in s (a different quadrature from the package).
(B) ||v1|| (beta -> infinity limit) from the Dirichlet sine eigenfunctions with the same heat-kernel trick:
    ||v1||^2 = (m/4) int t Q_D(t) g_D(t)^(d-2) dt, D'_n = phi_n'(0) + phi_n'(1).
(C) d = 2, 3: Chebyshev collocation solve of Lap w = 0, dn w + 2 beta w = -dn u* (fully independent of (A)).
(D) Theorem 1: | ||w|| - ||v1||/beta | <= K_d N beta^(-3/2), also at small beta (0.01, 0.1) not in the package.
Output: check/out/c2_bias.json
"""
import json, math, os, sys, time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
NROOT = int(os.environ.get("NROOT", "100000"))


def robin_1d(alpha, n=NROOT):
    """Return mu, D, X, M (arrays) for the even and odd Robin eigenfunctions, theta = omega/2."""
    k = np.arange(n, dtype=float)
    # even: theta tan theta = alpha/2 on (k pi, k pi + pi/2); f = theta sin - (alpha/2) cos
    lo = k * np.pi + 1e-15; hi = k * np.pi + np.pi / 2 - 1e-15
    f = lambda t: t * np.sin(t) - alpha / 2 * np.cos(t)
    flo = f(lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi); fm = f(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo); flo = np.where(same, fm, flo); hi = np.where(same, hi, mid)
    te = 0.5 * (lo + hi)
    # odd: theta cot theta = -alpha/2 on (k pi + pi/2, (k+1) pi); f = theta cos + (alpha/2) sin
    lo = k * np.pi + np.pi / 2 + 1e-15; hi = (k + 1) * np.pi - 1e-15
    g = lambda t: t * np.cos(t) + alpha / 2 * np.sin(t)
    glo = g(lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi); gm = g(mid)
        same = np.sign(gm) == np.sign(glo)
        lo = np.where(same, mid, lo); glo = np.where(same, gm, glo); hi = np.where(same, hi, mid)
    to = 0.5 * (lo + hi)
    we = 2 * te; wo = 2 * to
    # even: phi = c cos(w (x - 1/2)), int cos^2 = 1/2 + sin(w)/(2w)
    ce = 1 / np.sqrt(0.5 + np.sin(we) / (2 * we))
    Me = ce * 2 * np.sin(we / 2) / we
    Xe = 0.5 * Me
    De = np.zeros_like(Me)
    # odd: phi = c sin(w (x - 1/2)), int sin^2 = 1/2 - sin(w)/(2w)
    co = 1 / np.sqrt(0.5 - np.sin(wo) / (2 * wo))
    Mo = np.zeros_like(co)
    Do = co * 2 * np.sin(wo / 2)
    Xo = co * (-np.cos(wo / 2) / wo + 2 * np.sin(wo / 2) / wo ** 2)
    mu = np.concatenate([we ** 2, wo ** 2])
    return mu, np.concatenate([De, Do]), np.concatenate([Xe, Xo]), np.concatenate([Me, Mo]), te, to


def dirichlet_1d(n=NROOT):
    j = np.arange(1, n + 1, dtype=float)
    mu = (j * np.pi) ** 2
    Dp = np.sqrt(2) * j * np.pi * (1 + (-1) ** j)            # phi'(0) + phi'(1)
    M = np.sqrt(2) * (1 - (-1) ** j) / (j * np.pi)
    # X = int x sqrt2 sin(j pi x) dx = sqrt2 * (-(-1)^j / (j pi))
    X = np.sqrt(2) * (-(-1) ** j) / (j * np.pi)
    return mu, Dp, X, M


def heat_integral(mu, D, X, M, d, s_lo=-40.0, s_hi=None, ns=4000):
    m = d // 2
    mu0 = mu.min()
    if s_hi is None:
        s_hi = math.log(60.0 / (d * mu0))  # integrand ~ exp(-t d mu0) beyond
    s = np.linspace(s_lo, s_hi, ns)
    t = np.exp(s)
    vals = np.empty(ns)
    for i0 in range(0, ns, 100):
        tt = t[i0:i0 + 100]
        E = np.exp(-np.outer(tt, mu))
        SD2 = E @ D ** 2; SX2 = E @ X ** 2; SDX = E @ (D * X); g = E @ M ** 2
        Q = 2 * (SD2 * SX2 + SDX ** 2)
        vals[i0:i0 + 100] = tt * tt * Q * g ** (d - 2)  # dt = t ds
    h = s[1] - s[0]
    integral = h * (vals.sum() - 0.5 * (vals[0] + vals[-1]))
    return m * integral, float(vals[0]), float(vals[-1])


def unorm_P1(d):
    m = d // 2
    return math.sqrt(m / 9 + m * (m - 1) / 16)


def N_P1(d):
    m = d // 2
    return math.sqrt(4 * m / 3)


def K(d):
    return (math.sqrt(d) + 2 / (math.pi * math.sqrt(d))) / (4 * math.sqrt(2))


# ---------------------------------------------------------------- (C) Chebyshev collocation, d = 2, 3
def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    Dm = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    Dm = Dm - np.diag(Dm.sum(1))
    return Dm, x


def clenshaw_curtis(N):
    # weights on [-1,1] for Chebyshev points cos(pi k/N)
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1 / (N ** 2 - 1)
        for k in range(1, N // 2):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k ** 2 - 1)
        v -= np.cos(N * theta[1:-1]) / (N ** 2 - 1)
    else:
        w[0] = w[N] = 1 / N ** 2
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k ** 2 - 1)
    w[1:-1] = 2 * v / N
    return w


def colloc_bias(d, beta, N):
    Dm, xc = cheb(N)
    x = 0.5 * (xc + 1)            # map to [0,1]; d/dx = 2 d/dxc
    D1 = 2 * Dm; D2 = D1 @ D1
    I = np.eye(N + 1)
    n1 = N + 1
    grids = np.meshgrid(*([x] * d), indexing="ij")
    pts = np.stack([g.ravel() for g in grids], 1)
    def kron_axis(Mat, ax):
        out = np.array([[1.0]])
        for b in range(d):
            out = np.kron(out, Mat if b == ax else I)
        return out
    Lap = sum(kron_axis(D2, a) for a in range(d))
    Dax = [kron_axis(D1, a) for a in range(d)]
    A = Lap.copy(); rhs = np.zeros(n1 ** d)
    # u* = x1 x2 (d = 2) or x1 x2 (d = 3, coordinate 3 free): grad u* = (x2, x1, 0)
    gu = np.zeros_like(pts); gu[:, 0] = pts[:, 1]; gu[:, 1] = pts[:, 0]
    alpha = 2 * beta
    tol = 1e-14
    for i, p in enumerate(pts):
        onb = [(a, -1.0) for a in range(d) if abs(p[a]) < tol] + [(a, 1.0) for a in range(d) if abs(p[a] - 1) < tol]
        if not onb:
            continue
        # at edges/corners: use the average outward normal direction of the adjacent faces (unnormalised sum)
        row = alpha * np.eye(1, n1 ** d, i).ravel() * len(onb)
        r = 0.0
        for a, sgn in onb:
            row = row + sgn * Dax[a][i]
            r += -sgn * gu[i, a]
        A[i] = row; rhs[i] = r
    w = np.linalg.solve(A, rhs)
    wq = clenshaw_curtis(N) * 0.5
    W = wq
    for _ in range(d - 1):
        W = np.kron(W, wq)
    return math.sqrt(float((W * w ** 2).sum()))


if __name__ == "__main__":
    res = {}
    betas = [0.01, 0.1, 1, 3, 10, 100, 1000, 1e4, 1e5]
    ds = [2, 3, 5, 10, 20, 50, 100]
    # (B) ||v1||
    t0 = time.time()
    muD, DpD, XD, MD = dirichlet_1d()
    v1 = {}
    for d in ds + [1000, 10000]:
        val, a, b = heat_integral(muD, DpD, XD, MD, d)
        v1[d] = math.sqrt(val / 4)
        print(f"(B) d={d:6d} ||v1|| = {v1[d]:.6f}  c = {v1[d]/unorm_P1(d):.6f}  d c = {d*v1[d]/unorm_P1(d):.5f}  (end vals {a:.1e},{b:.1e})", flush=True)
    res["v1"] = {str(k): dict(v1_norm=v, c=v / unorm_P1(k), d_c=k * v / unorm_P1(k)) for k, v in v1.items()}
    # (A) exact bias
    rows = []
    for beta in betas:
        mu, D, X, M, te, to = robin_1d(2 * beta)
        for d in ds:
            val, a, b = heat_integral(mu, D, X, M, d)
            wn = math.sqrt(val)
            lhs = abs(wn - v1[d] / beta); rhs = K(d) * N_P1(d) * beta ** -1.5
            rows.append(dict(d=d, beta=beta, abs_bias=wn, rel_bias=wn / unorm_P1(d), thm1_lhs=lhs, thm1_rhs=rhs,
                             thm1_ok=bool(lhs <= rhs), bound_b=math.sqrt((math.tanh(math.pi*math.sqrt(d-1)/2)/(math.pi*math.sqrt(d-1)))) * N_P1(d) / (2 * beta),
                             ratio_exact_over_first=wn / (v1[d] / beta)))
            print(f"(A) d={d:3d} beta={beta:8g} rel bias {wn/unorm_P1(d):.6e}  thm1 {lhs:.3e} <= {rhs:.3e} {lhs<=rhs}  exact/first {wn/(v1[d]/beta):.4f}  t={time.time()-t0:.0f}s", flush=True)
    res["bias"] = rows
    # (C) collocation
    coll = []
    for d, Ns in [(2, [16, 24, 32, 48]), (3, [8, 12, 14])]:
        for beta in [1, 10, 100]:
            for N in Ns:
                wn = colloc_bias(d, beta, N)
                coll.append(dict(d=d, beta=beta, N=N, rel_bias=wn / unorm_P1(d)))
                print(f"(C) collocation d={d} beta={beta} N={N}: rel bias {wn/unorm_P1(d):.6e}  t={time.time()-t0:.0f}s", flush=True)
    res["collocation"] = coll
    json.dump(res, open(os.path.join(OUT, "c2_bias.json"), "w"), indent=1)
