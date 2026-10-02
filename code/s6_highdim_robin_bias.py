"""Penalty bias of the Deep Ritz energy for problem LapD in dimension d = 2, 3, 5, 10, 20 (Section 6.6,
Numerical observation of Supplementary Section S7).  No training.

The minimiser u_beta of the penalised Ritz energy solves a Robin problem (Proposition "Penalised Ritz
energy").  With w = u_beta - u*, u* = sum_k x_{2k-1} x_{2k}, gamma = 2 beta:

        Lap w = 0 in (0,1)^d,      d_n w + gamma w = - d_n u*  on the boundary               (weak form A.werr)

Two separate methods compute ||w||_{L2} / ||u*||_{L2}.

(1) SERIES (the values quoted in the article).  Let phi_n be the L2-normalised eigenfunctions of
    -phi'' = mu^2 phi on (0,1) with -phi'(0) + gamma phi(0) = 0, phi'(1) + gamma phi(1) = 0:
        symmetric      cos(mu (x - 1/2)),  mu tan(mu/2) = gamma,
        antisymmetric  sin(nu (x - 1/2)),  nu cot(nu/2) = -gamma.
    Write x = sum_n xi_n phi_n and 1 = sum_n eta_n phi_n (eta_n = 0 for antisymmetric modes).  For a coordinate
    i with partner j (u* contains x_i x_j) the datum is -x_j on {x_i = 1} and +x_j on {x_i = 0}, and
        w_i = - sum  xi_{n_j} prod_{l != i,j} eta_{n_l} * phi_{n_j}(x_j) prod_l phi_{n_l}(x_l) * psi_M(x_i),
        psi_M(x) = sinh(m (x - 1/2)) / (m cosh(m/2) + gamma sinh(m/2)),   m = sqrt(M),
        M = mu_{n_j}^2 + S,   S = sum_{l != i,j} mu_{n_l}^2,
    is harmonic, satisfies the homogeneous Robin condition on the faces of the other coordinates and the
    inhomogeneous one on the two faces of x_i; w is the sum of the w_i over the 2 floor(d/2) paired
    coordinates.  Orthogonality gives, with m_pairs = floor(d/2),
        ||w||^2 = 2 m_pairs * E_S[ H_diag(S) + H_cross(S) ],
        H_diag(S)  = sum_a xi_a^2 G(mu_a^2 + S),             G(M) = int_0^1 psi_M^2 dx,
        H_cross(S) = sum_{a,b antisym.} xi_a xi_b T(mu_a^2 + S; b) T(mu_b^2 + S; a),   T(M; b) = <psi_M, phi_b>,
    where S is the sum of d - 2 independent draws of mu_n^2 with probabilities eta_n^2 (they sum to 1); the
    products of w_i and w_i' vanish when i and i' belong to different pairs.  For d = 2, S = 0; for d = 3 the
    expectation is summed exactly; for d >= 5 it is estimated by Monte Carlo (fixed seed; standard error stored).
    The sums over a and b are truncated at K modes; the values for K/2 are stored as a convergence check.

(2) GALERKIN (cross-check).  Tensor products of shifted Legendre polynomials of total degree <= p in the weak
    form.  The datum is discontinuous across the edges of the cube, so this converges slowly; the values for
    several p are stored.  The same computation was repeated with separate code within the project.

The d = 2 values are also compared with the finite-difference values of verify_research_highdim/results/
v_robin_fd.json.  Finally the bias is set against the errors of the trained networks (Tables of Section 6).

Output: data/s6_highdim_robin_bias.json, data/s6_highdim_robin_bias.txt, data/s6_highdim_robin_bias_table.tex
Usage:  python code/s6_highdim_robin_bias.py            (about two minutes on one core)
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)
DATA = os.path.join(ART, "data")
DIMS = [2, 3, 5, 10, 20]
BETAS = [1.0, 10.0, 100.0, 1000.0]
K_MODES = 1000           # antisymmetric / symmetric modes kept in the sums over a and b
K_S = 200000             # symmetric modes kept in the distribution of S (the rest has mass < 1e-5 in total)
N_MC = 4_000_000
SEED = 20261001


# ------------------------------------------------------------------------------------ 1-D Robin modes
def modes(gamma, K):
    """First K symmetric and first K antisymmetric eigen-frequencies, with the coefficients of the
    normalised eigenfunctions:  eta (of the function 1), xi (of the function x)."""
    eps = 1e-13
    k = np.arange(K)
    mu = np.array([brentq(lambda m: m * math.sin(m / 2) - gamma * math.cos(m / 2),
                          2 * kk * math.pi + eps, (2 * kk + 1) * math.pi - eps, xtol=1e-14, rtol=1e-15) for kk in k])
    nu = np.array([brentq(lambda m: m * math.cos(m / 2) + gamma * math.sin(m / 2),
                          (2 * kk + 1) * math.pi + eps, (2 * kk + 2) * math.pi - eps, xtol=1e-14, rtol=1e-15) for kk in k])
    ns = np.sqrt(0.5 + np.sin(mu) / (2 * mu))            # norm of cos(mu (x - 1/2))
    na = np.sqrt(0.5 - np.sin(nu) / (2 * nu))            # norm of sin(nu (x - 1/2))
    eta = 2 * np.sin(mu / 2) / mu / ns                   # <1, phi_sym>
    xi_s = 0.5 * eta                                     # <x, phi_sym>
    xi_a = (-np.cos(nu / 2) / nu + 2 * np.sin(nu / 2) / nu ** 2) / na    # <x, phi_antisym>
    return mu, nu, eta, xi_s, xi_a, na


def G(M, gamma):
    m = np.sqrt(M)
    th = np.tanh(m / 2)
    return (th / m - 0.5 * (1 - th ** 2)) / (m + gamma * th) ** 2


def H_of_S(S, gamma, md, K):
    """H_diag(S) and H_cross(S) with K modes of each parity."""
    mu, nu, eta, xi_s, xi_a, na = (v[:K] for v in md)
    hd = float(np.sum(xi_s ** 2 * G(mu ** 2 + S, gamma)) + np.sum(xi_a ** 2 * G(nu ** 2 + S, gamma)))
    m = np.sqrt(nu ** 2 + S)[:, None]                    # psi index a (rows), antisymmetric mode b (columns)
    th = np.tanh(m / 2)
    nb = nu[None, :]
    T = 2 * (m * np.sin(nb / 2) - nb * th * np.cos(nb / 2)) / ((m ** 2 + nb ** 2) * (m + gamma * th) * na[None, :])
    hc = float(xi_a @ (T * T.T) @ xi_a)
    return hd, hc


class Series:
    """Everything that depends on beta only: the 1-D modes, the distribution of one transverse draw and the
    table of H = H_diag + H_cross on a grid of S (interpolated by a cubic spline in log S)."""

    def __init__(self, beta, K=K_MODES):
        self.beta, self.gamma, self.K = beta, 2 * beta, K
        g = self.gamma
        self.md = modes(g, K)
        kk = np.arange(K_S)
        lo, hi = 2 * kk * math.pi, (2 * kk + 1) * math.pi
        sgn = np.where(kk % 2 == 0, 1.0, -1.0)           # f(2k pi) = -gamma (-1)^k: make f negative at lo
        for _ in range(70):                              # vectorised bisection for mu tan(mu/2) = gamma
            mid = 0.5 * (lo + hi)
            neg = sgn * (mid * np.sin(mid / 2) - g * np.cos(mid / 2)) < 0
            lo = np.where(neg, mid, lo)
            hi = np.where(neg, hi, mid)
        self.muS = 0.5 * (lo + hi)
        self.pS = (2 * np.sin(self.muS / 2) / self.muS) ** 2 / (0.5 + np.sin(self.muS) / (2 * self.muS))
        self.cdf = np.cumsum(self.pS)
        s0 = self.muS[0] ** 2
        self.grid = np.concatenate([np.geomspace(s0, 1e7, 440), np.geomspace(1.5e7, 1e14, 60)])
        tab = np.array([sum(H_of_S(s, g, self.md, K)) for s in self.grid])
        tab_half = np.array([sum(H_of_S(s, g, self.md, K // 2)) for s in self.grid])
        self.spl = CubicSpline(np.log(self.grid), tab)
        self.spl_half = CubicSpline(np.log(self.grid), tab_half)
        mids = np.exp(0.5 * (np.log(self.grid[:-1]) + np.log(self.grid[1:])))[::9]
        exact = np.array([sum(H_of_S(s, g, self.md, K)) for s in mids])
        self.interp_err = float(np.max(np.abs(self.spl(np.log(mids)) - exact) / np.abs(exact)))

    def bias(self, d, rng):
        mpairs = d // 2
        unorm = math.sqrt((9 * mpairs ** 2 + 7 * mpairs) / 144.0)
        md = self.md
        out = {"K": self.K, "sum_eta_sq": float(np.sum(md[2] ** 2)),
               "sum_xi_sq": float(np.sum(md[3] ** 2) + np.sum(md[4] ** 2)),
               "mass_kept_per_coordinate": float(self.pS.sum()), "interpolation_max_rel_error": self.interp_err}

        def finish(e, e_half, se=0.0):
            out["rel_l2"] = math.sqrt(2 * mpairs * e) / unorm
            out["rel_l2_K_half"] = math.sqrt(2 * mpairs * e_half) / unorm
            out["mc_rel_standard_error"] = 0.5 * se / e
            return out

        if d == 2:                                       # no transverse coordinate: S = 0
            return finish(sum(H_of_S(0.0, self.gamma, md, self.K)), sum(H_of_S(0.0, self.gamma, md, self.K // 2)))
        if d == 3:                                       # one transverse coordinate: exact sum
            S = np.minimum(self.muS ** 2, self.grid[-1])
            return finish(float(self.pS @ self.spl(np.log(S))), float(self.pS @ self.spl_half(np.log(S))))
        if d == 4 and rng is None:                       # two transverse coordinates: double sum (check of the sampler)
            kq = 3000
            S = np.minimum(self.muS[:kq, None] ** 2 + self.muS[None, :kq] ** 2, self.grid[-1])
            w2 = self.pS[:kq, None] * self.pS[None, :kq]
            out["mass_kept_double_sum"] = float(w2.sum())
            return finish(float((w2 * self.spl(np.log(S))).sum()), float((w2 * self.spl_half(np.log(S))).sum()))
        tot, tot2, tot_half, n = 0.0, 0.0, 0.0, 0
        chunk = 250_000
        for _ in range(N_MC // chunk):
            idx = np.searchsorted(self.cdf, rng.random((chunk, d - 2)))
            tail = (idx >= K_S).any(1)                   # truncated tail of the distribution: S = infinity, H = 0
            S = (self.muS[np.minimum(idx, K_S - 1)] ** 2).sum(1)
            ls = np.log(np.minimum(S, self.grid[-1]))
            h = np.where(tail, 0.0, self.spl(ls))
            hh = np.where(tail, 0.0, self.spl_half(ls))
            tot += h.sum(); tot2 += (h ** 2).sum(); tot_half += hh.sum(); n += h.size
        e, e_half = tot / n, tot_half / n
        out["n_mc"] = n
        return finish(e, e_half, math.sqrt(max(tot2 / n - e ** 2, 0.0) / n))


# ------------------------------------------------------------------------------------ Galerkin cross-check
def galerkin_bias(d, p, betas):
    """Tensor shifted-Legendre polynomials of total degree <= p.  Returns (number of unknowns, {beta: rel})."""
    j = np.arange(p + 1)
    mass = 1.0 / (2 * j + 1)                              # int_0^1 L_i L_j = delta_ij / (2j+1)
    kmin = np.minimum.outer(j, j)
    D = np.where((j[:, None] + j[None, :]) % 2 == 0, 2.0 * kmin * (kmin + 1), 0.0)   # int_0^1 L_i' L_j'
    v0 = (-1.0) ** j                                       # L_j(0);  L_j(1) = 1
    idx = []

    def rec(prefix, left):
        if len(prefix) == d:
            idx.append(tuple(prefix)); return
        for a in range(left + 1):
            rec(prefix + [a], left - a)
    rec([], p)
    pos = {a: i for i, a in enumerate(idx)}
    n = len(idx)
    A = np.array(idx)
    mprod = np.prod(mass[A], axis=1)
    rk, ck, vk, rb, cb, vb = [], [], [], [], [], []
    for i, a in enumerate(idx):
        tot = sum(a)
        for l in range(d):
            base = mprod[i] / mass[a[l]]
            for c in range(0, p - (tot - a[l]) + 1):
                jj = pos[a[:l] + (c,) + a[l + 1:]]
                if D[a[l], c] != 0.0:
                    rk.append(i); ck.append(jj); vk.append(D[a[l], c] * base)
                rb.append(i); cb.append(jj); vb.append((v0[a[l]] * v0[c] + 1.0) * base)
    Kmat = sp.csr_matrix((vk, (rk, ck)), shape=(n, n))
    Bmat = sp.csr_matrix((vb, (rb, cb)), shape=(n, n))
    mpairs = d // 2
    xint = {0: 0.5, 1: 1.0 / 6.0}                         # int_0^1 x L_c(x) dx
    rhs = np.zeros(n)
    for gi, g in enumerate(idx):
        nz = [l for l in range(d) if g[l] != 0]
        s = 0.0
        for k in range(mpairs):
            for (i, jc) in ((2 * k, 2 * k + 1), (2 * k + 1, 2 * k)):
                if any(l not in (i, jc) for l in nz) or g[jc] not in xint:
                    continue
                s += (1.0 - v0[g[i]]) * xint[g[jc]]       # (L(1) - L(0)) * int x_j L
        rhs[gi] = -s
    unorm = math.sqrt((9 * mpairs ** 2 + 7 * mpairs) / 144.0)
    res = {}
    for beta in betas:
        c = spla.spsolve((Kmat + 2 * beta * Bmat).tocsc(), rhs)
        res[beta] = math.sqrt(float(np.sum(c * c * mprod))) / unorm
    return n, res


# ------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out = {"note": "Relative penalty bias ||u_beta - u*|| / ||u*|| of the penalised Ritz energy for LapD. "
                   "'series': separation of variables in 1-D Robin eigenfunctions (see the module docstring); "
                   "'galerkin': tensor Legendre polynomials of total degree <= p; 'fd_d2': second-order finite "
                   "differences of verify_research_highdim/results/v_robin_fd.json.",
           "K_modes": K_MODES, "K_S": K_S, "n_mc": N_MC, "seed": SEED, "series": {}, "galerkin": {}}
    lines = []
    out["check_d4"] = {}
    for beta in BETAS:
        ser = Series(beta)
        # check of the Monte-Carlo sampler at d = 4, where the expectation is also a double sum
        out["check_d4"][f"{beta:g}"] = {"double_sum": ser.bias(4, None)["rel_l2"], "monte_carlo": ser.bias(4, rng)["rel_l2"]}
        for d in DIMS:
            r = ser.bias(d, rng)
            out["series"][f"{d}/{beta:g}"] = r
            lines.append(f"series   d={d:2d} beta={beta:6g}: rel = {r['rel_l2']:.5e}   (K/2: {r['rel_l2_K_half']:.5e}; "
                         f"MC rel. s.e. {r['mc_rel_standard_error']:.1e}; beta*rel = {beta * r['rel_l2']:.4f})")
            print(lines[-1], flush=True)
    gal_cases = {2: (8, 12, 16), 3: (8, 12, 16), 4: (10, 14), 5: (6, 8, 10, 12), 10: (4, 6), 20: (2, 4)}
    for d, ps in gal_cases.items():
        for p in ps:
            n, res = galerkin_bias(d, p, BETAS)
            out["galerkin"][f"{d}/p{p}"] = {"n_unknowns": n, **{f"{b:g}": v for b, v in res.items()}}
            lines.append(f"galerkin d={d:2d} p={p:2d} n={n:6d}: " + "  ".join(f"beta={b:g}: {v:.4e}" for b, v in res.items()))
            print(lines[-1], flush=True)
    fd = json.load(open(os.path.join(ROOT, "verify_research_highdim", "results", "v_robin_fd.json")))
    out["fd_d2"] = {b: fd[b]["rel_l2_N200"] for b in ("1", "10", "100", "1000")}
    out["series_over_fd_d2"] = {b: out["series"][f"2/{b}"]["rel_l2"] / out["fd_d2"][b] for b in out["fd_d2"]}
    # agreement of the two methods where the Galerkin values have converged
    best = {2: 16, 3: 16, 5: 12, 10: 6, 20: 4}
    out["series_over_galerkin_largest_p"] = {
        f"{d}/{b:g}": out["series"][f"{d}/{b:g}"]["rel_l2"] / out["galerkin"][f"{d}/p{best[d]}"][f"{b:g}"]
        for d in DIMS for b in BETAS}
    for b in BETAS:
        c = out["check_d4"][f"{b:g}"]
        c["galerkin_p14"] = out["galerkin"]["4/p14"][f"{b:g}"]
        lines.append(f"check d=4 beta={b:g}: double sum {c['double_sum']:.5e}  Monte Carlo {c['monte_carlo']:.5e}  "
                     f"Galerkin p=14 {c['galerkin_p14']:.5e}")
    lines.append("series / finite differences, d = 2: " + ", ".join(f"beta={b}: {v:.5f}" for b, v in out["series_over_fd_d2"].items()))
    lines.append("series / Galerkin (largest p): " + ", ".join(f"{k}: {v:.3f}" for k, v in out["series_over_galerkin_largest_p"].items()))

    # ---------------------------------------------------------------- against the trained networks
    def bias(d, beta=100.0):
        return out["series"][f"{d}/{beta:g}"]["rel_l2"]

    summ = {(r["problem"], r["method"], int(r["d"])): float(r["rel_l2_mean"])
            for r in csv.DictReader(open(os.path.join(ROOT, "research_highdim", "results", "summary.csv")))}
    num = json.load(open(os.path.join(DATA, "s6_highdim_numbers.json")))
    ext = num["extensions"]
    cmp4000, cmp16000 = {}, {}
    for d in DIMS:
        pinn, ritz = summ[("laplace", "pinn", d)], summ[("laplace", "ritz", d)]
        cmp4000[str(d)] = {"bias": bias(d), "pinn": pinn, "ritz": ritz, "gap": ritz - pinn,
                           "bias_over_ritz": bias(d) / ritz, "bias_over_gap": bias(d) / (ritz - pinn),
                           "bias_over_pinn": bias(d) / pinn}
    for d in (10, 20):
        pinn = ext[f"long/laplace/pinn/{d}/16000"]["rel_l2"]["mean"]
        ritz = ext[f"long/laplace/ritz/{d}/16000"]["rel_l2"]["mean"]
        cmp16000[str(d)] = {"bias": bias(d), "pinn": pinn, "ritz": ritz, "gap": ritz - pinn,
                            "bias_over_ritz": bias(d) / ritz, "bias_over_gap": bias(d) / (ritz - pinn),
                            "ritz_distance_to_u_beta_bounds": [ritz - bias(d), ritz + bias(d)]}
    eq = ext["equaltime/laplace/ritz/10/10000"]["rel_l2"]["mean"]
    cmp_eq = {"bias": bias(10), "ritz_equal_compute": eq, "bias_over_ritz": bias(10) / eq}
    sweep = {float(r["w"]): float(r["val_rel_l2"]) for r in
             csv.DictReader(open(os.path.join(ROOT, "research_highdim", "results", "sweep.csv")))
             if r["problem"] == "laplace" and r["method"] == "ritz" and int(r["d"]) == 5}
    cmp_sweep = {f"{b:g}": {"bias": bias(5, b), "trained_val_error": sweep[b], "trained_over_bias": sweep[b] / bias(5, b)}
                 for b in BETAS}
    out["against_trained_networks"] = {"beta_100_4000_iterations": cmp4000, "beta_100_16000_iterations": cmp16000,
                                       "beta_100_equal_compute_d10": cmp_eq, "weight_sweep_d5": cmp_sweep}
    out["d_times_bias_beta100"] = {str(d): d * bias(d) for d in DIMS}
    for d in DIMS:
        c = cmp4000[str(d)]
        lines.append(f"beta=100, 4000 its, d={d:2d}: bias {c['bias']:.3e}  PINN {c['pinn']:.3e}  Ritz {c['ritz']:.3e}  "
                     f"bias/Ritz {c['bias_over_ritz']:.2f}  bias/gap {c['bias_over_gap']:.2f}")
    for d in ("10", "20"):
        c = cmp16000[d]
        lines.append(f"beta=100, 16000 its, d={d}: bias {c['bias']:.3e}  PINN {c['pinn']:.3e}  Ritz {c['ritz']:.3e}  "
                     f"gap {c['gap']:.3e}  bias/Ritz {c['bias_over_ritz']:.2f}  bias/gap {c['bias_over_gap']:.2f}  "
                     f"Ritz distance to u_beta in [{c['ritz_distance_to_u_beta_bounds'][0]:.2e}, {c['ritz_distance_to_u_beta_bounds'][1]:.2e}]")
    lines.append(f"beta=100, equal compute, d=10: bias/Ritz {cmp_eq['bias_over_ritz']:.2f}")
    for b, c in cmp_sweep.items():
        lines.append(f"d=5 sweep, beta={b}: bias {c['bias']:.3e}  trained (validation) {c['trained_val_error']:.3e}  ratio {c['trained_over_bias']:.2f}")
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(DATA, "s6_highdim_robin_bias.json"), "w"), indent=1)
    open(os.path.join(DATA, "s6_highdim_robin_bias.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-14:]))

    # ---------------------------------------------------------------- LaTeX rows (units of 1e-2)
    def u2(x, nd=3):
        return f"{100 * x:.{nd}f}"
    rows = []
    for d in DIMS:
        c4 = cmp4000[str(d)]
        c16 = cmp16000.get(str(d))
        rows.append(" & ".join([
            str(d), u2(c4["bias"]), u2(c4["pinn"]), u2(c4["ritz"]), f"{100 * c4['bias_over_ritz']:.0f}",
            u2(c16["pinn"]) if c16 else "--", u2(c16["ritz"]) if c16 else "--",
            f"{100 * c16['bias_over_ritz']:.0f}" if c16 else "--"]) + r" \\")
    open(os.path.join(DATA, "s6_highdim_robin_bias_table.tex"), "w").write("\n".join(rows) + "\n")
    print("wrote data/s6_highdim_robin_bias.{json,txt}, data/s6_highdim_robin_bias_table.tex in", out["seconds"], "s")


if __name__ == "__main__":
    main()
