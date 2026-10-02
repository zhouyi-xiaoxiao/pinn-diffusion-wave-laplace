"""Symbolic and numerical checks of every computational step used in Supplementary Section S7 (Proofs).

No training.  Runs in well under a minute on one core.  Writes
    ../data/sA_proofs_checks.json   (all numbers)
    ../data/sA_proofs_checks.txt    (human-readable log)

The proofs in sections/S7_proofs.tex do not depend on this script; it is a cross-check that
the displayed formulae are the ones that the algebra gives.

Blocks
  1  Prop. exact (a): PDE, initial and boundary residuals of the seven closed-form solutions.
  2  Prop. exact (b): sine coefficients b_n of cos z, sum b_n^2 = 1, harmonic terms,
     the bound rho_n(x) <= exp(-k_n (pi - x)), the mean-square boundary defect, the energy
     formula on {x < pi - delta} and its logarithmic divergence; the two Parseval formulae
     (boundary defect, slice energy) against direct quadrature of the truncated series.
  2u Prop. unique: the energy identities for the heat and wave equations and the
     integration by parts behind c'' = (j^2 + n^2) c, on explicit test functions.
  3  Prop. floors: exact moments (variances 7/144 and 1/144, covariance -2/pi^2).
  4  Prop. robin: the two face inequalities and the trace bound, the coercivity constant,
     the Parseval identities of the duality argument, the bound on the normal derivative of
     psi_M and the duality inequality on harmonic functions, the normal derivatives of P1
     and P2, and every inequality of (b)-(c) on a finite-difference solution of the Robin
     problem for P1 at d = 2.
  5  Prop. nonunique and Prop. notwave: direct computation and the value 1.380.

Usage:  python sA_proofs_checks.py
"""
import json
import os

import mpmath as mp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import sympy as sy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "..", "data", "sA_proofs_checks.json")
OUT_TXT = os.path.join(HERE, "..", "data", "sA_proofs_checks.txt")
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)  # repository root

res, log = {}, []


def say(msg):
    print(msg)
    log.append(msg)


def zero(expr):
    """True if a sympy expression simplifies to 0."""
    return sy.simplify(expr) == 0


# =========================================================================================
# 1. Prop. exact (a)
# =========================================================================================
t, x, y, z = sy.symbols("t x y z", real=True)
pi = sy.pi
blk = {}

u = sy.exp(-pi**2 * t) * sy.sin(pi * x)                                       # H
blk["H"] = all([zero(sy.diff(u, t) - sy.diff(u, x, 2)), zero(u.subs(t, 0) - sy.sin(pi * x)),
                zero(u.subs(x, 0)), zero(u.subs(x, 1))])

u = sy.sin(pi * x) * sy.cos(2 * pi * t) + sy.Rational(1, 2) * sy.sin(4 * pi * x) * sy.cos(8 * pi * t)   # W1
blk["W1"] = all([zero(sy.diff(u, t, 2) - 4 * sy.diff(u, x, 2)),
                 zero(u.subs(t, 0) - sy.sin(pi * x) - sy.sin(4 * pi * x) / 2),
                 zero(sy.diff(u, t).subs(t, 0)), zero(u.subs(x, 0)), zero(u.subs(x, 1))])

S = sy.sin(pi * x) * sy.sin(pi * y)
om = sy.sqrt(2) * pi
u = S * sy.cos(om * t)                                                        # W2
blk["W2"] = all([zero(sy.diff(u, t, 2) - sy.diff(u, x, 2) - sy.diff(u, y, 2)), zero(u.subs(t, 0) - S),
                 zero(sy.diff(u, t).subs(t, 0))] + [zero(u.subs(v, s)) for v in (x, y) for s in (-1, 1)])

u = x / (2 * pi) + sy.cos(2 * y) * sy.sinh(2 * x) / (2 * sy.sinh(2 * pi))     # L2
blk["L2"] = all([zero(sy.diff(u, x, 2) + sy.diff(u, y, 2)), zero(u.subs(x, 0)),
                 zero(u.subs(x, pi) - (1 + sy.cos(2 * y)) / 2),
                 zero(sy.diff(u, y).subs(y, 0)), zero(sy.diff(u, y).subs(y, pi))])

u = sy.sin(y) * sy.sin(z) * sy.sinh(sy.sqrt(2) * x) / sy.sinh(sy.sqrt(2) * pi)  # L3s
blk["L3s"] = all([zero(sum(sy.diff(u, v, 2) for v in (x, y, z))), zero(u.subs(x, pi) - sy.sin(y) * sy.sin(z)),
                  zero(u.subs(x, 0))] + [zero(u.subs(v, s)) for v in (y, z) for s in (0, pi)])

for d in (2, 3, 5, 6):
    X = sy.symbols(f"x1:{d + 1}", real=True)
    m = d // 2
    u1 = sum(X[2 * k] * X[2 * k + 1] for k in range(m))                       # P1
    u2 = sum(sy.cos(pi * xi) for xi in X) / sy.sqrt(d)                        # P2
    blk[f"P1_d{d}"] = zero(sum(sy.diff(u1, xi, 2) for xi in X))
    blk[f"P2_d{d}"] = zero(-sum(sy.diff(u2, xi, 2) for xi in X) - pi**2 * u2)
res["exact_a_all_residuals_zero"] = blk
say(f"[1] exact (a): all residuals zero: {blk}")
assert all(blk.values())

# =========================================================================================
# 2. Prop. exact (b)
# =========================================================================================
blk = {}
n = sy.symbols("n", integer=True, positive=True)
bn_exact = {}
for nn in range(1, 9):
    val = sy.Rational(2) / pi * sy.integrate(sy.cos(z) * sy.sin(nn * z), (z, 0, pi))
    bn_exact[nn] = sy.simplify(val)
    target = sy.Rational(4 * nn) / (pi * (nn**2 - 1)) if nn % 2 == 0 else 0
    assert zero(val - target), nn
blk["b_n_formula_checked_for_n_1_to_8"] = True
blk["sum_bn_squared"] = float(mp.nsum(lambda j: (4 * (2 * j) / (mp.pi * ((2 * j) ** 2 - 1))) ** 2, [1, mp.inf]))
# each term harmonic
kn = sy.sqrt(1 + n**2)
term = sy.sin(y) * sy.sin(n * z) * sy.sinh(kn * x) / sy.sinh(kn * pi)
blk["term_harmonic"] = zero(sum(sy.diff(term, v, 2) for v in (x, y, z)))
# rho_n(x) <= exp(-k_n (pi - x)) and cosh(k x)/sinh(k pi) <= 3 exp(-k (pi - x)) on a grid
xs = np.linspace(0.0, np.pi, 2001)
worst_s, worst_c = -np.inf, -np.inf
for nn in range(2, 41, 2):
    k = np.sqrt(1.0 + nn * nn)
    # written with exponentials to avoid overflow
    rho = np.exp(-k * (np.pi - xs)) * (1 - np.exp(-2 * k * xs)) / (1 - np.exp(-2 * k * np.pi))
    chs = np.exp(-k * (np.pi - xs)) * (1 + np.exp(-2 * k * xs)) / (1 - np.exp(-2 * k * np.pi))
    worst_s = max(worst_s, float(np.max(rho - np.exp(-k * (np.pi - xs)))))
    worst_c = max(worst_c, float(np.max(chs - 3 * np.exp(-k * (np.pi - xs)))))
blk["max_rho_minus_bound"] = worst_s           # must be <= 0
blk["max_coshratio_minus_3bound"] = worst_c    # must be <= 0
assert worst_s <= 1e-15 and worst_c <= 0
# |b_n| <= 8/(3 pi) and k_n <= 2n
blk["max_abs_bn"] = max(4 * nn / (np.pi * (nn * nn - 1)) for nn in range(2, 2001, 2))
blk["bound_8_over_3pi"] = 8 / (3 * np.pi)


def b(nn):
    return 4 * nn / (mp.pi * (nn * nn - 1))


def rho_mp(nn, xx):
    k = mp.sqrt(1 + nn * nn)
    return mp.exp(-k * (mp.pi - xx)) * (1 - mp.exp(-2 * k * xx)) / (1 - mp.exp(-2 * k * mp.pi))


# mean-square boundary defect  (pi^2/4) sum b_n^2 (1 - rho_n(x))^2  -> 0 as x -> pi
defect = {}
for j in (1, 2, 3, 4):
    xx = mp.pi - mp.mpf(10) ** (-j)
    s = mp.nsum(lambda q: b(2 * q) ** 2 * (1 - rho_mp(2 * q, xx)) ** 2, [1, mp.inf])
    defect[f"pi-1e-{j}"] = float(mp.pi**2 / 4 * s)
blk["mean_square_defect"] = defect
# 1-D identity behind the energy formula:
#   int_0^{pi-delta} k^2 (cosh^2 + sinh^2)(k x) / sinh^2(k pi) dx = k sinh(2k(pi-delta)) / (2 sinh^2(k pi))
errs = []
for nn in (2, 4, 10):
    for delta in (0.5, 0.1):
        k = mp.sqrt(1 + nn * nn)
        lhs = mp.quad(lambda s: k**2 * mp.cosh(2 * k * s) / mp.sinh(k * mp.pi) ** 2, [0, mp.pi - delta])
        rhs = k * mp.sinh(2 * k * (mp.pi - delta)) / (2 * mp.sinh(k * mp.pi) ** 2)
        errs.append(float(abs(lhs - rhs) / rhs))
blk["energy_x_integral_max_rel_err"] = max(errs)
# divergence: partial sums of (pi^2/4) b_n^2 k_n coth(k_n pi) against (pi^2/4) * 16/(pi^2 n) = 4/n
part, lower = {}, {}
for N in (10**2, 10**4, 10**6):
    ns = np.arange(2, N + 1, 2, dtype=float)
    ks = np.sqrt(1 + ns * ns)
    bb = 4 * ns / (np.pi * (ns * ns - 1))
    part[str(N)] = float(np.pi**2 / 4 * np.sum(bb**2 * ks / np.tanh(ks * np.pi)))
    lower[str(N)] = float(np.sum(4.0 / ns))
    assert np.all(bb**2 * ks >= 16 / (np.pi**2 * ns))
blk["energy_partial_sums"] = part
blk["harmonic_lower_bound_partial_sums"] = lower

# The two Parseval formulae against direct Gauss-Legendre quadrature of the truncated series
#   u_T(x,y,z) = sin y sum_{n even <= NT} b_n sin(nz) rho_n(x).
# The y-integrals are exact (int sin^2 = int cos^2 = pi/2); the z-integrals are done numerically.
NT = 200
zq, zw = np.polynomial.legendre.leggauss(1200)
zq, zw = 0.5 * np.pi * (zq + 1), 0.5 * np.pi * zw
n_even = np.arange(2, NT + 1, 2, dtype=float)
k_even = np.sqrt(1 + n_even**2)
b_even = 4 * n_even / (np.pi * (n_even**2 - 1))
sin_nz, cos_nz = np.sin(np.outer(n_even, zq)), np.cos(np.outer(n_even, zq))
direct = {}
for xx in (2.0, 2.8, np.pi - 0.05):
    e1 = np.exp(-k_even * (np.pi - xx))
    den_ = 1 - np.exp(-2 * k_even * np.pi)
    rho_ = e1 * (1 - np.exp(-2 * k_even * xx)) / den_            # sinh(k x)/sinh(k pi)
    chs_ = e1 * (1 + np.exp(-2 * k_even * xx)) / den_            # cosh(k x)/sinh(k pi)
    # (i) boundary defect  int int (u_T - sin y cos z)^2 dy dz
    series_z = (b_even * rho_) @ sin_nz
    quad_defect = np.pi / 2 * np.sum(zw * (series_z - np.cos(zq)) ** 2)
    formula_defect = np.pi**2 / 4 * (np.sum(b_even**2 * (1 - rho_) ** 2) + (1 - np.sum(b_even**2)))
    # (ii) slice energy  int int |grad u_T|^2 dy dz
    ux = (b_even * k_even * chs_) @ sin_nz                       # coefficient of sin y
    uy = (b_even * rho_) @ sin_nz                                # coefficient of cos y
    uz = (b_even * n_even * rho_) @ cos_nz                       # coefficient of sin y
    quad_energy = np.pi / 2 * np.sum(zw * (ux**2 + uy**2 + uz**2))
    cosh2_over_sinh2 = chs_**2 + rho_**2                         # cosh(2 k x)/sinh^2(k pi)
    formula_energy = np.pi**2 / 4 * np.sum(b_even**2 * k_even**2 * cosh2_over_sinh2)
    direct[f"x={xx:.4f}"] = {"defect_quadrature": float(quad_defect), "defect_formula": float(formula_defect),
                             "slice_energy_quadrature": float(quad_energy), "slice_energy_formula": float(formula_energy)}
    assert abs(quad_defect - formula_defect) <= 1e-9 * max(1.0, formula_defect)
    assert abs(quad_energy - formula_energy) <= 1e-9 * formula_energy
blk["parseval_vs_direct_quadrature_truncation_200"] = direct
res["exact_b"] = blk
say(f"[2] exact (b): {json.dumps(blk)}")

# =========================================================================================
# 2u. Prop. unique: the identities used by the energy arguments, on explicit test functions
# =========================================================================================
blk = {}
jj, nn_ = sy.symbols("j n", integer=True, positive=True)
# heat: w vanishes at x = 0, 1 but solves nothing;  E' = 2 int w (w_t - w_xx) - 2 int w_x^2
w = x * (1 - x) * (sy.exp(t) + x * sy.sin(t))
E = sy.integrate(w**2, (x, 0, 1))
rhs = sy.integrate(2 * w * (sy.diff(w, t) - sy.diff(w, x, 2)) - 2 * sy.diff(w, x) ** 2, (x, 0, 1))
blk["heat_energy_identity"] = zero(sy.diff(E, t) - rhs)
# wave (c = 1) on (-1,1)^2: w vanishes on the boundary;  E' = int w_t (w_tt - Lap w)
w = (1 - x**2) * (1 - y**2) * (sy.cos(t) + x * y * t**2)
dens = (sy.diff(w, t) ** 2 + sy.diff(w, x) ** 2 + sy.diff(w, y) ** 2) / 2
E = sy.integrate(dens, (x, -1, 1), (y, -1, 1))
rhs = sy.integrate(sy.diff(w, t) * (sy.diff(w, t, 2) - sy.diff(w, x, 2) - sy.diff(w, y, 2)), (x, -1, 1), (y, -1, 1))
blk["wave_energy_identity"] = zero(sy.diff(E, t) - rhs)
# Laplace with mixed data (as L2): w = 0 on x = 0, pi and w_y = 0 on y = 0, pi;
#   int |grad w|^2 = - int w Lap w   (all boundary terms vanish)
w = x * (pi - x) * (1 + sy.cos(y) * x + sy.cos(2 * y))
lhs = sy.integrate(sy.diff(w, x) ** 2 + sy.diff(w, y) ** 2, (x, 0, pi), (y, 0, pi))
rhs = -sy.integrate(w * (sy.diff(w, x, 2) + sy.diff(w, y, 2)), (x, 0, pi), (y, 0, pi))
blk["mixed_green_identity"] = zero(lhs - rhs)
# integration by parts behind c'' = (j^2 + n^2) c: for g with g(0) = g(pi) = 0,
#   int_0^pi g'' sin(jy) dy = -j^2 int_0^pi g sin(jy) dy
ok = True
for g in (y * (pi - y), y**2 * (pi - y) * sy.exp(y), sy.sin(y) * sy.cosh(y)):
    for jv in (1, 2, 5):
        ok = ok and zero(sy.integrate(sy.diff(g, y, 2) * sy.sin(jv * y), (y, 0, pi))
                         + jv**2 * sy.integrate(g * sy.sin(jv * y), (y, 0, pi)))
blk["integration_by_parts_identity"] = bool(ok)
# the coefficient ODE on the exact solution of L3s:  c(x) = (pi^2/4) sinh(sqrt2 x)/sinh(sqrt2 pi), j = n = 1
us = sy.sin(y) * sy.sin(z) * sy.sinh(sy.sqrt(2) * x) / sy.sinh(sy.sqrt(2) * pi)
cx = sy.integrate(us * sy.sin(y) * sy.sin(z), (y, 0, pi), (z, 0, pi))
blk["coefficient_ode_on_L3s"] = zero(sy.diff(cx, x, 2) - 2 * cx) and zero(cx.subs(x, 0))
assert all(blk.values())
res["unique"] = blk
say(f"[2u] unique: {json.dumps(blk)}")

# =========================================================================================
# 3. Prop. floors
# =========================================================================================
a, c = sy.symbols("a c", real=True)
blk = {}
E = lambda f, *v: sy.integrate(f, *[(s, 0, 1) for s in v])
mean_p = E(a * c, a, c)
var_p = E((a * c) ** 2, a, c) - mean_p**2
cov_p = E(a * c * a, a, c) - mean_p * sy.Rational(1, 2)
var_x = E(a**2, a) - sy.Rational(1, 4)
resid_p = var_p - 2 * cov_p**2 / var_x
blk["P1_mean_pair"], blk["P1_var_pair"] = str(mean_p), str(var_p)
blk["P1_cov_pair_coord"], blk["var_coord"], blk["P1_affine_residual_pair"] = str(cov_p), str(var_x), str(resid_p)
assert var_p == sy.Rational(7, 144) and resid_p == sy.Rational(1, 144)
# orthogonality of (x1-1/2)(x2-1/2) to affine functions and the decomposition of x1 x2
q = (a - sy.Rational(1, 2)) * (c - sy.Rational(1, 2))
assert zero(a * c - (q + a / 2 + c / 2 - sy.Rational(1, 4)))
assert all(E(q * f, a, c) == 0 for f in (1, a, c)) and E(q**2, a, c) == sy.Rational(1, 144)
mean_c = E(sy.cos(pi * a), a)
var_c = E(sy.cos(pi * a) ** 2, a)
cov_c = E(a * sy.cos(pi * a), a)
blk["P2_mean_cos"], blk["P2_var_cos"], blk["P2_cov_cos_coord"] = str(mean_c), str(var_c), str(cov_c)
assert mean_c == 0 and var_c == sy.Rational(1, 2) and zero(cov_c + 2 / pi**2)
blk["P2_affine_floor"] = float(sy.sqrt(1 - 96 / pi**4))
cf = json.load(open(os.path.join(HERE, "..", "data", "closed_forms.json")))["floors"]
cmp_ = {}
for d in ("2", "3", "5", "10", "20"):
    m = int(d) // 2
    cmp_[d] = {
        "P1_const": float(np.sqrt(7 / (7 + 9 * m))), "P1_const_mc": cf[d]["monte_carlo"]["P1"]["const_mc"],
        "P1_affine": float(np.sqrt(1 / (7 + 9 * m))), "P1_affine_mc": cf[d]["monte_carlo"]["P1"]["affine_mc"],
        "P2_affine_mc": cf[d]["monte_carlo"]["P2"]["affine_mc"],
    }
blk["closed_form_vs_monte_carlo"] = cmp_
res["floors"] = blk
say(f"[3] floors: {json.dumps(blk)}")

# =========================================================================================
# 4. Prop. robin
# =========================================================================================
blk = {}
rng = np.random.default_rng(20261001)
# Gauss-Legendre rule on (0,1)
gx, gw = np.polynomial.legendre.leggauss(40)
gx, gw = 0.5 * (gx + 1), 0.5 * gw
X1, X2 = np.meshgrid(gx, gx, indexing="ij")
W2 = np.outer(gw, gw)

# (4a) the two face inequalities of the lemma on random smooth functions, d = 2, face {x2 = 0}
worst_i, worst_ii = -np.inf, -np.inf
for _ in range(200):
    A = rng.normal(size=(4, 4))
    ph = rng.uniform(0, 2 * np.pi, size=(4, 4, 2))
    f = lambda p, q: sum(A[i, j] * np.cos(i * np.pi * p + ph[i, j, 0]) * np.cos(j * np.pi * q + ph[i, j, 1])
                         for i in range(4) for j in range(4))
    f2 = lambda p, q: sum(-A[i, j] * j * np.pi * np.cos(i * np.pi * p + ph[i, j, 0]) * np.sin(j * np.pi * q + ph[i, j, 1])
                          for i in range(4) for j in range(4))
    vol = np.sum(W2 * f(X1, X2) ** 2)
    der = np.sum(W2 * f2(X1, X2) ** 2)
    face = np.sum(gw * f(gx, 0 * gx) ** 2)
    worst_i = max(worst_i, vol - 2 * face - 2 * der)
    worst_ii = max(worst_ii, face - 2 * vol - 2 * der)
blk["lemma_ineq_i_max_violation"] = float(worst_i)     # must be <= 0
blk["lemma_ineq_ii_max_violation"] = float(worst_ii)   # must be <= 0
assert worst_i <= 0 and worst_ii <= 0

# (4a') the same two inequalities on functions chosen to stress them (near-equality cases):
#   v = 1 (ratio 1/2 in both), v = boundary layer exp(-x2/eps), v = x2^p
stress = {}
for name, f, f2 in (
    ("const", lambda q: 1 + 0 * q, lambda q: 0 * q),
    ("layer_eps_0.05", lambda q: np.exp(-q / 0.05), lambda q: -np.exp(-q / 0.05) / 0.05),
    ("layer_eps_1", lambda q: np.exp(-q), lambda q: -np.exp(-q)),
    ("x2", lambda q: q, lambda q: 1 + 0 * q),
    ("1_minus_x2_cubed", lambda q: (1 - q) ** 3, lambda q: -3 * (1 - q) ** 2),
):
    vol, der, face = np.sum(gw * f(gx) ** 2), np.sum(gw * f2(gx) ** 2), float(f(np.array([0.0]))[0] ** 2)
    stress[name] = {"i_lhs_over_rhs": float(vol / (2 * face + 2 * der)), "ii_lhs_over_rhs": float(face / (2 * vol + 2 * der))}
    assert vol <= 2 * face + 2 * der + 1e-14 and face <= 2 * vol + 2 * der + 1e-14
blk["lemma_stress_cases"] = stress

# (4a'') coercivity constant of part (a):  2 A + 3 G <= K (G + 2 beta A) with K = 3 / min(1, 2 beta)
okc = True
for beta in (1e-3, 0.1, 0.5, 1.0, 10.0, 1e3):
    K = 3 / min(1.0, 2 * beta)
    okc = okc and (K >= 3 - 1e-15) and (2 * beta * K >= 2 - 1e-15)
blk["coercivity_constant_ok"] = bool(okc)
assert okc

# (4b) Parseval identities for psi = sum w_k e_k / (pi^2 |k|^2), d = 2 and d = 3
for d in (2, 3):
    K = 4
    ks = np.array(np.meshgrid(*[np.arange(1, K + 1)] * d, indexing="ij")).reshape(d, -1).T
    wk = rng.normal(size=len(ks))
    k2 = (ks**2).sum(1)
    w_sq = np.sum(wk**2)
    hess_sq = sum(np.sum(wk**2 * ks[:, i] ** 2 * ks[:, j] ** 2 / k2**2) for i in range(d) for j in range(d))
    grad_sq = np.sum(wk**2 / (np.pi**2 * k2))
    blk[f"parseval_d{d}_hess_over_w"] = float(hess_sq / w_sq)                       # = 1
    blk[f"parseval_d{d}_grad_bound_ratio"] = float(grad_sq / (w_sq / (d * np.pi**2)))  # <= 1
    assert abs(hess_sq / w_sq - 1) < 1e-12 and grad_sq <= w_sq / (d * np.pi**2) + 1e-15
# direct quadrature check of ||D^2 psi|| = ||w|| in d = 2 (not through the coefficient formula)
K = 4
wk = rng.normal(size=(K, K))
kk = np.arange(1, K + 1)
s1, c1 = np.sin(np.pi * np.outer(kk, gx)), np.cos(np.pi * np.outer(kk, gx))     # (K, n)
den = np.pi**2 * (kk[:, None] ** 2 + kk[None, :] ** 2)
wv = 2 * np.einsum("ij,ia,jb->ab", wk, s1, s1)
pxx = 2 * np.einsum("ij,ia,jb->ab", -wk * (np.pi * kk[:, None]) ** 2 / den, s1, s1)
pyy = 2 * np.einsum("ij,ia,jb->ab", -wk * (np.pi * kk[None, :]) ** 2 / den, s1, s1)
pxy = 2 * np.einsum("ij,ia,jb->ab", wk * np.pi**2 * kk[:, None] * kk[None, :] / den, c1, c1)
blk["quadrature_hess_sq_over_w_sq"] = float(np.sum(W2 * (pxx**2 + pyy**2 + 2 * pxy**2)) / np.sum(W2 * wv**2))
blk["quadrature_minus_lap_psi_equals_w_maxerr"] = float(np.max(np.abs(-(pxx + pyy) - wv)))


# (4b') the duality step in d = 2, by quadrature.  For w in L2 with sine coefficients w_k
#   (k1, k2 <= M):  psi_M = sum w_k e_k / (pi^2 |k|^2),  -Lap psi_M = w_M,  and
#     ||d_n psi_M||^2_{L2(dOmega)} <= 4 C_2^2 ||w_M||^2 .
#   If moreover w is harmonic, then  ||w_M||^2 = - int_{dOmega} w d_n psi_M  and hence
#     ||w_M|| <= 2 C_2 ||w||_{L2(dOmega)} .
def duality_d2(wfun, M):
    """wfun(x1, x2) on arrays.  Returns ||w_M||^2, ||d_n psi_M||^2_bdry, -int_bdry w d_n psi_M, ||w||^2, ||w||^2_bdry."""
    km = np.arange(1, M + 1)
    sn = np.sqrt(2) * np.sin(np.pi * np.outer(km, gx))                 # orthonormal sines at the nodes
    wv_ = wfun(X1, X2)
    coef = np.einsum("ab,ia,jb->ij", W2 * wv_, sn, sn)                 # w_k
    pc = coef / (np.pi**2 * (km[:, None] ** 2 + km[None, :] ** 2))     # coefficients of psi_M
    d0 = np.sqrt(2) * np.pi * km                                       # d/ds [sqrt2 sin(pi k s)] at s = 0
    d1 = d0 * np.cos(np.pi * km)                                       # ... at s = 1
    # outward normal derivative on the four edges, as functions of the tangential node gx
    dn = {"x1=0": -np.einsum("ij,i,jb->b", pc, d0, sn), "x1=1": np.einsum("ij,i,jb->b", pc, d1, sn),
          "x2=0": -np.einsum("ij,j,ia->a", pc, d0, sn), "x2=1": np.einsum("ij,j,ia->a", pc, d1, sn)}
    wb = {"x1=0": wfun(0 * gx, gx), "x1=1": wfun(1 + 0 * gx, gx), "x2=0": wfun(gx, 0 * gx), "x2=1": wfun(gx, 1 + 0 * gx)}
    dn_sq = sum(np.sum(gw * v**2) for v in dn.values())
    pairing = -sum(np.sum(gw * wb[key] * dn[key]) for key in dn)
    wb_sq = sum(np.sum(gw * v**2) for v in wb.values())
    return float(np.sum(coef**2)), float(dn_sq), float(pairing), float(np.sum(W2 * wv_**2)), float(wb_sq)


C2sq = 1 + 1 / (2 * np.pi**2)
dual = {}
worst_dn = 0.0
for trial in range(50):                                               # arbitrary (non-harmonic) w
    cf_ = rng.normal(size=(6, 6))
    wf = lambda p, q: sum(cf_[i, j] * np.cos(i * p + 0.3 * j) * np.sin(1.7 * j * q + i) for i in range(6) for j in range(6))
    wM2, dn2, _, _, _ = duality_d2(wf, 12)
    worst_dn = max(worst_dn, dn2 / (4 * C2sq * wM2))
blk["dn_psi_bound_max_ratio_random_w"] = float(worst_dn)              # must be <= 1
assert worst_dn <= 1.0
for name, wf in (("1", lambda p, q: 1 + 0 * p), ("x1*x2", lambda p, q: p * q), ("x1^2-x2^2", lambda p, q: p**2 - q**2),
                 ("exp(x1)cos(x2)", lambda p, q: np.exp(p) * np.cos(q)),
                 ("cosh(3 x1)sin(3 x2)", lambda p, q: np.cosh(3 * p) * np.sin(3 * q))):
    wM2, dn2, pairing, w2, wb2 = duality_d2(wf, 16)
    dual[name] = {"wM_sq": wM2, "minus_boundary_pairing": pairing, "dn_psi_sq_over_4C2sq_wM_sq": dn2 / (4 * C2sq * wM2),
                  "w_L2_over_boundary_L2": float(np.sqrt(w2 / wb2)), "bound_2C2": float(2 * np.sqrt(C2sq))}
    assert abs(wM2 - pairing) <= 1e-8 * wM2                           # Green's formula for harmonic w
    assert dn2 <= 4 * C2sq * wM2 and np.sqrt(w2 / wb2) <= 2 * np.sqrt(C2sq)
blk["duality_on_harmonic_functions_d2_M16"] = dual

# (4b'') the trace bound  ||v||^2_{dOmega} <= 4 d ||v||^2 + 4 ||grad v||^2  (d = 2) on random smooth functions
worst_tr = 0.0
for _ in range(200):
    A = rng.normal(size=(4, 4))
    ph = rng.uniform(0, 2 * np.pi, size=(4, 4, 2))
    f = lambda p, q: sum(A[i, j] * np.cos(i * np.pi * p + ph[i, j, 0]) * np.cos(j * np.pi * q + ph[i, j, 1])
                         for i in range(4) for j in range(4))
    f1 = lambda p, q: sum(-A[i, j] * i * np.pi * np.sin(i * np.pi * p + ph[i, j, 0]) * np.cos(j * np.pi * q + ph[i, j, 1])
                          for i in range(4) for j in range(4))
    f2 = lambda p, q: sum(-A[i, j] * j * np.pi * np.cos(i * np.pi * p + ph[i, j, 0]) * np.sin(j * np.pi * q + ph[i, j, 1])
                          for i in range(4) for j in range(4))
    vol = np.sum(W2 * f(X1, X2) ** 2)
    grd = np.sum(W2 * (f1(X1, X2) ** 2 + f2(X1, X2) ** 2))
    bnd = sum(np.sum(gw * v**2) for v in (f(0 * gx, gx), f(1 + 0 * gx, gx), f(gx, 0 * gx), f(gx, 1 + 0 * gx)))
    worst_tr = max(worst_tr, bnd / (8 * vol + 4 * grd))
blk["trace_bound_max_ratio"] = float(worst_tr)                        # must be <= 1
assert worst_tr <= 1.0

# (4c) normal derivatives of P1 and P2
nd = {}
for d in (2, 3, 5, 6):
    Xs = sy.symbols(f"x1:{d + 1}", real=True)
    m = d // 2
    u1 = sum(Xs[2 * k] * Xs[2 * k + 1] for k in range(m))
    u2 = sum(sy.cos(pi * xi) for xi in Xs) / sy.sqrt(d)
    tot1, tot2 = 0, 0
    for i in range(d):
        others = [(Xs[j], 0, 1) for j in range(d) if j != i]
        for side, sign in ((0, -1), (1, 1)):
            g1 = (sign * sy.diff(u1, Xs[i])).subs(Xs[i], side)
            g2 = (sign * sy.diff(u2, Xs[i])).subs(Xs[i], side)
            tot1 += sy.integrate(g1**2, *others) if others else g1**2
            tot2 += sy.integrate(g2**2, *others) if others else g2**2
    nd[str(d)] = {"P1_norm_sq": str(sy.nsimplify(tot1)), "expected_4m_over_3": str(sy.Rational(4 * m, 3)),
                  "P2_norm_sq": str(sy.simplify(tot2))}
    assert sy.simplify(tot1 - sy.Rational(4 * m, 3)) == 0 and sy.simplify(tot2) == 0
blk["normal_derivatives"] = nd


# (4d) finite-difference Robin solution for P1, d = 2:  -Lap u = 0, d_n u + 2 beta (u - xy) = 0
def robin_fd(beta, N):
    h, n1 = 1.0 / N, N + 1
    idx = lambda i, j: i * n1 + j
    rows, cols, vals = [], [], []
    rhs = np.zeros(n1 * n1)
    g = np.linspace(0, 1, n1)
    for i in range(n1):
        for j in range(n1):
            k, diag = idx(i, j), -4.0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii <= N and 0 <= jj <= N:
                    rows.append(k); cols.append(idx(ii, jj)); vals.append(1.0)
                else:       # ghost value = mirror value - 4 h beta (u_k - g_k)
                    rows.append(k); cols.append(idx(i - di, j - dj)); vals.append(1.0)
                    diag -= 4 * h * beta
                    rhs[k] -= 4 * h * beta * g[i] * g[j]
            rows.append(k); cols.append(k); vals.append(diag)
    Amat = sp.csr_matrix((vals, (rows, cols)), shape=(n1 * n1, n1 * n1))
    return g, spl.spsolve(Amat, rhs).reshape(n1, n1)


def trap(v, h):
    return h * (np.sum(v) - 0.5 * (v[0] + v[-1]))


N = 200
h = 1.0 / N
Nn = np.sqrt(4.0 / 3.0)                        # ||d_n u*||_{L2(dOmega)} for P1, d = 2
Cd = np.sqrt(1 + 1 / (2 * np.pi**2))
ustar_norm = 1.0 / 3.0                         # sqrt((9 m^2 + 7 m)/144) with m = 1
ref = json.load(open(os.path.join(ROOT, "verify_research_highdim", "results", "v_robin_fd.json")))
fd = {}
for beta in (1, 10, 100, 1000):
    g, ub = robin_fd(beta, N)
    Xg, Yg = np.meshgrid(g, g, indexing="ij")
    w = ub - Xg * Yg
    wt = np.ones(N + 1); wt[0] = wt[-1] = 0.5
    l2 = np.sqrt(h * h * np.sum(np.outer(wt, wt) * w**2))
    bnd_sq = trap(w[0, :] ** 2, h) + trap(w[-1, :] ** 2, h) + trap(w[:, 0] ** 2, h) + trap(w[:, -1] ** 2, h)
    gx_ = np.diff(w, axis=0) / h               # on vertical cell edges
    gy_ = np.diff(w, axis=1) / h
    grad_sq = h * h * (np.sum(wt[None, :] * gx_**2) + np.sum(wt[:, None] * gy_**2))
    # -int_{dOmega} d_n u* w :  d_n u* = -y (x=0), +y (x=1), -x (y=0), +x (y=1)
    rhs_id = -(trap(-g * w[0, :], h) + trap(g * w[-1, :], h) + trap(-g * w[:, 0], h) + trap(g * w[:, -1], h))
    lhs_id = grad_sq + 2 * beta * bnd_sq
    a_ = np.sqrt(bnd_sq)
    fd[str(beta)] = {
        "rel_L2": float(l2 / ustar_norm), "rel_L2_verifier_N200": ref[str(beta)]["rel_l2_N200"],
        "L2": float(l2), "bound_c_Cd_N_over_beta": float(Cd * Nn / beta),
        "bnd": float(a_), "bound_b_N_over_2beta": float(Nn / (2 * beta)),
        "grad_sq": float(grad_sq), "bound_b_N2_over_8beta": float(Nn**2 / (8 * beta)),
        "duality_L2_over_bnd": float(l2 / a_), "bound_2Cd": float(2 * Cd),
        "energy_identity_lhs": float(lhs_id), "energy_identity_rhs": float(rhs_id),
        "beta_times_L2": float(beta * l2),
    }
    r = fd[str(beta)]
    assert r["L2"] <= r["bound_c_Cd_N_over_beta"] and r["bnd"] <= r["bound_b_N_over_2beta"]
    assert r["grad_sq"] <= r["bound_b_N2_over_8beta"] and r["duality_L2_over_bnd"] <= r["bound_2Cd"]
    # the verification code normalised by the trapezoidal value of ||u*||; we use the exact value 1/3
    assert abs(r["rel_L2"] / r["rel_L2_verifier_N200"] - 1) < 1e-4
    assert abs(lhs_id - rhs_id) <= 0.02 * abs(rhs_id)
blk["C_2"] = float(Cd)
blk["norm_dn_ustar_P1_d2"] = float(Nn)
blk["fd_robin_P1_d2_N200"] = fd
res["robin"] = blk
say(f"[4] robin: {json.dumps(blk)}")

# =========================================================================================
# 5. Prop. nonunique and Prop. notwave
# =========================================================================================
blk = {}
lap3 = lambda f: sum(sy.diff(f, v, 2) for v in (x, y, z))
u0 = sy.cosh(sy.sqrt(2) * (x - pi)) * sy.sin(y) * sy.cos(z)
uN = sy.sin(y) * sy.cos(z) * sy.sinh(sy.sqrt(2) * x) / sy.sinh(sy.sqrt(2) * pi)
Acst = sy.symbols("A", real=True)
blk["u0_harmonic_and_datum"] = zero(lap3(u0)) and zero(u0.subs(x, pi) - sy.sin(y) * sy.cos(z))
hp = [sy.re(sy.expand((y + sy.I * z) ** k)) for k in range(0, 7)]
blk["x_minus_pi_times_harmonic_polys"] = all(zero(lap3((x - pi) * hk)) and zero(((x - pi) * hk).subs(x, pi)) for hk in hp)
blk["value_at_face_centre"] = str(sy.simplify((u0 + Acst * (x - pi)).subs({x: 0, y: pi / 2, z: pi / 2})))
blk["uN_checks"] = all([zero(lap3(uN)), zero(uN.subs(x, pi) - sy.sin(y) * sy.cos(z)), zero(uN.subs(x, 0)),
                        zero(uN.subs(y, 0)), zero(uN.subs(y, pi)),
                        zero(sy.diff(uN, z).subs(z, 0)), zero(sy.diff(uN, z).subs(z, pi))])
blk["u0_on_face_x0"] = str(sy.simplify(u0.subs(x, 0)))
assert blk["u0_harmonic_and_datum"] and blk["x_minus_pi_times_harmonic_polys"] and blk["uN_checks"]

box = lambda f: sy.diff(f, t, 2) - sy.diff(f, x, 2) - sy.diff(f, y, 2)
v = sy.exp(-t) * S
blk["v_residual"] = str(sy.factor(sy.simplify(box(v))))
blk["v_residual_ok"] = zero(box(v) - (1 + 2 * pi**2) * sy.exp(-t) * S)
assert blk["v_residual_ok"]
omf = mp.sqrt(2) * mp.pi
num_cf = (1 - mp.e**-2) / 2 - 2 * (1 + mp.e**-1 * (omf * mp.sin(omf) - mp.cos(omf))) / (1 + omf**2) \
    + mp.mpf(1) / 2 + mp.sin(2 * omf) / (4 * omf)
den_cf = mp.mpf(1) / 2 + mp.sin(2 * omf) / (4 * omf)
num_q = mp.quad(lambda s: (mp.exp(-s) - mp.cos(omf * s)) ** 2, [0, 1])
den_q = mp.quad(lambda s: mp.cos(omf * s) ** 2, [0, 1])
blk["ratio_closed_form"] = float(mp.sqrt(num_cf / den_cf))
blk["ratio_quadrature"] = float(mp.sqrt(num_q / den_q))
blk["ratio_closed_forms_json"] = json.load(open(os.path.join(HERE, "..", "data", "closed_forms.json")))["wave2d"]["rel_L2_target_vs_exact_ut0_zero"]
blk["numerator"], blk["denominator"] = float(num_cf), float(den_cf)
assert abs(blk["ratio_closed_form"] - blk["ratio_quadrature"]) < 1e-12
assert abs(blk["ratio_closed_form"] - blk["ratio_closed_forms_json"]) < 1e-10
res["nonunique_notwave"] = blk
say(f"[5] nonunique / notwave: {json.dumps(blk)}")

json.dump(res, open(OUT_JSON, "w"), indent=1)
open(OUT_TXT, "w").write("\n".join(log) + "\nALL CHECKS PASSED\n")
print("ALL CHECKS PASSED ->", os.path.normpath(OUT_JSON))
