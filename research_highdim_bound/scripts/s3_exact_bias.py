# -- added for the public repository: absolute paths of the development machine are replaced by
# -- paths relative to the repository root, through the helper _repo_path defined here.
import os as _os


def _repo_path(rel=""):
    d = _os.path.dirname(_os.path.abspath(__file__))
    while not (_os.path.isdir(_os.path.join(d, "research_benchmark")) and _os.path.isdir(_os.path.join(d, "research_highdim"))):
        p = _os.path.dirname(d)
        if p == d:
            raise RuntimeError("repository root not found")
        d = p
    return _os.path.join(d, rel) if rel else d
# -- end of the added lines

"""Proposition 3 (exact penalty bias for P1 on the cube) and Theorem 1 (first-order term), numerically.

Formula (the working document of the proofs, Proposition 3), alpha = 2 beta, m = floor(d/2):
    ||u_beta - u*||^2 = m int_0^inf t Q(t) g(t)^{d-2} dt,
    g(t) = sum_n M_n^2 e^{-t mu_n},  Q(t) = 2[(sum D_n^2 e^{-t mu_n})(sum X_n^2 e^{-t mu_n}) + (sum D_n X_n e^{-t mu_n})^2],
with the L2-normalised 1-D Robin eigenpairs (mu_n, phi_n) and D_n = phi_n(1) - phi_n(0), X_n = int x phi_n,
M_n = int phi_n.  Separately written checks:
  (i)   d = 1 closed form ||w|| = 1/(sqrt(12) (1 + beta))  (u* = x; the formula with d = 1 is not of the
        P1 form, so the 1-D eigen-expansion sum_n (h . phi_n)^2/mu_n^2 is checked instead);
  (ii)  d = 2 and d = 3: direct (truncated) eigenfunction sums sum_n F_n^2/Lambda_n^2, without the t-integral;
  (iii) d = 2: second-order finite differences of the article (beta ||w|| = 0.16673, 0.18279, 0.18476);
  (iv)  beta -> infinity: beta ||w|| -> ||v1||, ||v1|| from the Dirichlet sine series (exact_v1 below);
  (v)   the article's series values (data/s6_highdim_robin_bias.json, Monte Carlo over coordinates for d >= 5)
        and the polynomial Galerkin values computed as a separate check (data/s6_highdim_robin_bias.txt, inputs/robin_bias_sparse.out);
  (vi)  Theorem 1: | ||w|| - ||v1||/beta | <= K_d N beta^{-3/2}, K_d = (sqrt d + 2/(pi sqrt d))/(4 sqrt 2).
Truncation study: number of roots per type NMAX in {50 000, 200 000}; quadrature tolerance 1e-7 and 1e-9.
Outputs: results/exact_bias.jsonl (one record per (d, beta, setting)), results/exact_bias_summary.json,
results/exact_v1.json, results/exact_bias_checks.json."""
import json, math, os, sys, time
import numpy as np
from scipy import integrate, interpolate
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from t2common import U, RES, dump, append_jsonl, load_jsonl

DS = [2, 3, 5, 10, 20, 50, 100]
BETAS = [1, 3, 10, 100, 1000, 10000, 100000]
JSONL = os.path.join(RES, "exact_bias.jsonl")
ART = _repo_path("data/s6_highdim_robin_bias.json")


# ------------------------------------------------------------------ 1-D Robin eigen-data
def roots(alpha, n):
    k = np.arange(n, dtype=float)
    lo, hi = k * math.pi + 1e-15, k * math.pi + math.pi / 2 - 1e-15   # theta tan theta = alpha/2
    for _ in range(90):
        mid = 0.5 * (lo + hi); f = mid * np.tan(mid) - alpha / 2
        lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
    th_e = 0.5 * (lo + hi)
    lo, hi = k * math.pi + math.pi / 2 + 1e-15, (k + 1) * math.pi - 1e-15  # theta cot theta = -alpha/2
    for _ in range(90):
        mid = 0.5 * (lo + hi); f = mid / np.tan(mid) + alpha / 2
        lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
    th_o = 0.5 * (lo + hi)
    return 2 * th_e, 2 * th_o


def one_d(alpha, n):
    om_e, om_o = roots(alpha, n)
    ne = np.sqrt(0.5 + np.sin(om_e) / (2 * om_e))
    no = np.sqrt(0.5 - np.sin(om_o) / (2 * om_o))
    M_e = 2 * np.sin(om_e / 2) / om_e / ne
    X_e = M_e / 2
    D_o = 2 * np.sin(om_o / 2) / no
    X_o = 2 * (np.sin(om_o / 2) / om_o ** 2 - np.cos(om_o / 2) / (2 * om_o)) / no
    return dict(om_e=om_e, om_o=om_o, ne=ne, no=no, mu_e=om_e ** 2, mu_o=om_o ** 2,
                M_e=M_e, X_e=X_e, D_o=D_o, X_o=X_o)


def bias_sq(d, beta, nmax=200_000, epsrel=1e-9):
    o = one_d(2 * beta, nmax)
    m = d // 2
    mu_e, mu_o = o["mu_e"], o["mu_o"]
    Me2, Xe2, Do2, Xo2, DX = o["M_e"] ** 2, o["X_e"] ** 2, o["D_o"] ** 2, o["X_o"] ** 2, o["D_o"] * o["X_o"]

    def integrand(y):
        t = math.exp(y)
        ee = np.exp(-t * mu_e); eo = np.exp(-t * mu_o)
        g = float(Me2 @ ee)
        sDD = float(Do2 @ eo); sXX = float(Xe2 @ ee + Xo2 @ eo); sDX = float(DX @ eo)
        return t * t * 2 * (sDD * sXX + sDX ** 2) * g ** (d - 2)
    v, err = integrate.quad(integrand, math.log(1e-11), math.log(80.0), limit=800, epsrel=epsrel, epsabs=0)
    return m * v, m * err


def ustar_norm(d):
    m = d // 2
    return math.sqrt(m / 9 + m * (m - 1) / 16)


def N_norm(d):
    return math.sqrt(4 * (d // 2) / 3)


# ------------------------------------------------------------------ separately written check (i): d = 1
def check_d1():
    rows = []
    for beta in [1, 10, 100]:
        o = one_d(2 * beta, 200_000)
        # u* = x, h = -1 at x=0, +1 at x=1: int_bd h phi = phi(1) - phi(0) = D (odd type only)
        w2 = float((o["D_o"] ** 2 / o["mu_o"] ** 2).sum())
        rows.append(dict(beta=beta, series=math.sqrt(w2), closed=1 / (math.sqrt(12) * (1 + beta))))
    return rows


# ------------------------------------------------------------------ separately written check (ii): direct sums
def direct_sum(d, beta, n):
    o = one_d(2 * beta, n)
    # coordinate types: even (D = 0, M != 0) and odd (M = 0, D != 0); X both
    mu = np.concatenate([o["mu_e"], o["mu_o"]])
    D = np.concatenate([np.zeros(n), o["D_o"]])
    X = np.concatenate([o["X_e"], o["X_o"]])
    M = np.concatenate([o["M_e"], np.zeros(n)])
    if d == 2:
        F = D[:, None] * X[None, :] + X[:, None] * D[None, :]
        Lam = mu[:, None] + mu[None, :]
        return float((F ** 2 / Lam ** 2).sum())
    if d == 3:  # m = 1, pair (1,2), third coordinate contributes M
        tot = 0.0
        F12 = D[:, None] * X[None, :] + X[:, None] * D[None, :]
        Lam12 = mu[:, None] + mu[None, :]
        for j in range(n):  # only even type has M != 0
            tot += float((F12 ** 2 * M[j] ** 2 / (Lam12 + mu[j]) ** 2).sum())
        return tot
    raise ValueError


# ------------------------------------------------------------------ check (iv): Dirichlet sine series for ||v1||
def G(tau):
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    out = np.empty_like(tau)
    small = tau < 1e-3
    out[small] = 1.0 - 4.0 * np.sqrt(tau[small] / math.pi)
    big = ~small
    if big.any():
        k = np.arange(1, 4001, 2, dtype=float)[:, None]
        out[big] = (8.0 / (math.pi ** 2 * k ** 2) * np.exp(-math.pi ** 2 * k ** 2 * tau[big][None, :])).sum(0)
    return out


def check_G():
    k = np.arange(1, 400001, 2, dtype=float)
    return [dict(tau=t, series=float((8.0 / (math.pi ** 2 * k ** 2) * np.exp(-math.pi ** 2 * k ** 2 * t)).sum()),
                 approx=1.0 - 4.0 * math.sqrt(t / math.pi)) for t in [1e-3, 3e-4, 1e-4]]


def W_func(dm2):
    if dm2 == 0:
        return lambda s: 1.0 / np.asarray(s, float) ** 2
    ss = np.exp(np.linspace(math.log(1.0), math.log(1e7), 500))
    vals = []
    for s in ss:
        f = lambda y: math.pi ** 4 * math.exp(2 * y) * math.exp(-math.pi ** 2 * math.exp(y) * s) * float(G(math.exp(y))[0]) ** dm2
        yc = -math.log(math.pi ** 2 * s)
        v, _ = integrate.quad(f, yc - 25, yc + 8, limit=400, epsabs=0, epsrel=1e-10)
        vals.append(v)
    spl = interpolate.CubicSpline(np.log(ss), np.log(vals))
    return lambda s: np.exp(spl(np.log(np.asarray(s, float))))


def S_trunc(Wf, K):
    k1 = np.arange(1, K + 1, dtype=float)[:, None]
    k2 = np.arange(1, K + 1, dtype=float)[None, :]
    A = np.where((k1 % 2 == 0), 4 * k1 * (-1.0) ** (k2 + 1) / k2, 0.0)
    B = np.where((k2 % 2 == 0), 4 * k2 * (-1.0) ** (k1 + 1) / k1, 0.0)
    return float(((A + B) ** 2 * Wf(k1 ** 2 + k2 ** 2)).sum() / (4 * math.pi ** 4))


def exact_v1(d, Ks=(250, 500, 1000, 2000)):
    m = d // 2
    Wf = W_func(d - 2)
    vals = [m * S_trunc(Wf, K) for K in Ks]
    ext = 2 * vals[-1] - vals[-2]          # tail c/K + O(1/K^2)
    return dict(d=d, v1_norm=math.sqrt(ext), partial=[math.sqrt(v) for v in vals], Ks=list(Ks),
                c=math.sqrt(ext) / ustar_norm(d))


def main():
    t0 = time.time()
    checks = dict(d1=check_d1(), G=check_G())
    o = one_d(20.0, 200_000)
    checks["parseval_M"] = float((o["M_e"] ** 2).sum())
    checks["parseval_X"] = float((o["X_e"] ** 2).sum() + (o["X_o"] ** 2).sum())
    checks["parseval_note"] = "sum M_n^2 -> 1 = ||1||^2, sum X_n^2 -> 1/3 = ||x||^2 (truncated at 2e5 roots per type)"
    # Robin condition residuals of the eigenfunctions
    om_e, om_o = o["om_e"][:1000], o["om_o"][:1000]
    checks["robin_residual_even"] = float(np.max(np.abs(om_e * np.tan(om_e / 2) - 20.0)))
    checks["robin_residual_odd"] = float(np.max(np.abs(om_o / np.tan(om_o / 2) + 20.0)))
    print(checks, flush=True)
    dump("exact_bias_checks.json", checks)

    # main table, two truncation settings
    done = {(r["d"], r["beta"], r["nmax"], r["epsrel"]) for r in load_jsonl(JSONL)}
    for nmax, eps in [(200_000, 1e-9), (50_000, 1e-7)]:
        for d in DS:
            for beta in BETAS:
                if (d, beta, nmax, eps) in done:
                    continue
                if nmax == 50_000 and beta in (10000, 100000):
                    continue
                c0 = time.process_time()
                w2, err = bias_sq(d, beta, nmax, eps)
                rec = dict(d=d, beta=beta, nmax=nmax, epsrel=eps, bias_abs=math.sqrt(w2), quad_err_sq=err,
                           ustar_norm=ustar_norm(d), rel_bias=math.sqrt(w2) / ustar_norm(d),
                           beta_times_rel=beta * math.sqrt(w2) / ustar_norm(d), cpu_s=time.process_time() - c0)
                append_jsonl(JSONL, rec)
                print(rec, flush=True)

    # direct sums d = 2, 3
    ds = []
    for d, n in [(2, 1000), (2, 3000), (3, 100), (3, 200)]:
        for beta in [1, 10, 100]:
            ds.append(dict(d=d, n_per_type=n, beta=beta, direct_abs=math.sqrt(direct_sum(d, beta, n))))
            print(ds[-1], flush=True)
    checks["direct_sums"] = ds
    dump("exact_bias_checks.json", checks)

    # sine series for ||v1||
    v1 = []
    for d in DS:
        r = exact_v1(d); v1.append(r); print(r, flush=True)
        dump("exact_v1.json", dict(check_G=checks["G"], rows=v1))
    print("wall", time.time() - t0)


if __name__ == "__main__":
    main()
