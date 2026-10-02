"""Numerical check of every step of the proof of Theorem 5 of the article (Supplementary Section S7.5;
Theorem 2 of the study's own numbering, see README.md).

Step labels (A1 to A7, used in the src comments of Supplementary Section S7.5):
  (A1) face formula: on the face x_i = 0, dn psi_k = -(sqrt2 k_i/(pi |k|^2)) F_{k_{-i}} and on x_i = 1,
       dn psi_k = (-1)^{k_i} (sqrt2 k_i/(pi |k|^2)) F_{k_{-i}}, psi_k = e_k/(pi^2 |k|^2); checked against
       centred finite differences of psi_k on both faces (d = 2, 3) and for a random finite zeta (d = 2)
       the coefficient formula for ||T zeta||^2 against quadrature on the four sides.
  (A2) the parity identity (sum b)^2 + (sum (-1)^k b)^2 = 2 E^2 + 2 O^2 (random vectors).
  (A3) Cauchy-Schwarz step (random coefficients: E_i^2 + O_i^2 <= sum a^2 w * odd-sum).
  (A4) partial fractions: sum_{j odd} 1/(j^2+a^2) = pi tanh(pi a/2)/(4a); monotone decrease in a;
       even sum <= odd sum.
  (A5) table of L_d, U_d (d = 1..100), U_1 = L_1 = 1/2, sqrt(d) U_d -> 1/pi, sqrt(d) L_d -> 1/(2 pi).
  (A6) Lanczos: largest eigenvalue of T*T on sine modes with max k_i <= K (d = 2, 3; d = 4 small K),
       increasing in K, between L_d and U_d.
  (A7) lower-bound modes: the Rayleigh quotient of zeta_J (a_k ~ j/(j^2+d-1) on k = (j,1,...,1),
       j odd <= J) by the closed formula and by the Lanczos operator, >= (4/pi^2) S_J -> L_d.
Outputs: results/theorem2_checks.json, results/theorem2_checks.txt."""
import json, math, os, sys, time
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from t2common import U, L, L_tail_bound, odd_sum_closed, RES, dump

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------- (A1) face formula
def psi_k(k, x):
    """psi_k = e_k / (pi^2 |k|^2), e_k = 2^{d/2} prod sin(pi k_i x_i); x: (n, d)."""
    k = np.asarray(k, float)
    return 2 ** (len(k) / 2) * np.prod(np.sin(math.pi * k * x), axis=1) / (math.pi ** 2 * (k ** 2).sum())


def face_formula_check():
    out = []
    rng = np.random.default_rng(0)
    h = 1e-5
    for k in [(3, 2), (1, 1), (4, 5), (2, 1, 3), (1, 2, 2)]:
        d = len(k); kk = sum(t * t for t in k)
        for i in range(d):
            for side in (0, 1):
                y = rng.random((2000, d))
                xp = y.copy(); xm = y.copy()
                xp[:, i] = side + h; xm[:, i] = side - h
                d_i = (psi_k(k, xp) - psi_k(k, xm)) / (2 * h)  # psi is smooth across the face (odd extension)
                normal = -1.0 if side == 0 else 1.0
                dn_fd = normal * d_i
                others = [j for j in range(d) if j != i]
                F = 2 ** ((d - 1) / 2) * np.prod(np.sin(math.pi * np.array(k, float)[others] * y[:, others]), axis=1)
                coef = math.sqrt(2) * k[i] / (math.pi * kk)
                dn_formula = (-coef if side == 0 else (-1) ** k[i] * coef) * F
                err = float(np.max(np.abs(dn_fd - dn_formula)) / np.max(np.abs(dn_formula)))
                out.append(dict(k=k, face=f"x{i+1}={side}", max_rel_err=err))
    worst = max(r["max_rel_err"] for r in out)
    say(f"(A1) face formula vs centred finite differences, {len(out)} faces: worst max-rel-err {worst:.2e}")
    return dict(rows=out, worst=worst)


def random_zeta_check(K=6, nq=400):
    """d = 2: zeta = sum_{k<=K} a_k e_k (random a); ||T zeta||^2 by the coefficient formula and by
    Gauss-Legendre quadrature of (dn psi)^2 on the four sides, dn psi from the exact derivative."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((K, K))
    k1, k2 = np.meshgrid(np.arange(1, K + 1), np.arange(1, K + 1), indexing="ij")
    k2s = k1 ** 2 + k2 ** 2
    # formula
    tot = 0.0
    for kk_i, kk_o in ((k1, k2), (k2, k1)):
        v = a * kk_i / k2s
        axis = 0 if kk_i is k1 else 1
        for par in (0, 1):
            mask = (kk_i % 2 == par)
            s = (v * mask).sum(axis=axis)
            tot += (s ** 2).sum()
    formula = 4 / math.pi ** 2 * tot
    # quadrature
    t, wq = np.polynomial.legendre.leggauss(nq)
    t = 0.5 * (t + 1); wq = 0.5 * wq
    def d1psi(x1, x2):
        return sum(a[p, q] * 2 * math.pi * (p + 1) * np.cos(math.pi * (p + 1) * x1) * np.sin(math.pi * (q + 1) * x2)
                   / (math.pi ** 2 * ((p + 1) ** 2 + (q + 1) ** 2)) for p in range(K) for q in range(K))
    def d2psi(x1, x2):
        return sum(a[p, q] * 2 * math.pi * (q + 1) * np.sin(math.pi * (p + 1) * x1) * np.cos(math.pi * (q + 1) * x2)
                   / (math.pi ** 2 * ((p + 1) ** 2 + (q + 1) ** 2)) for p in range(K) for q in range(K))
    quad = 0.0
    quad += (wq * d1psi(0.0, t) ** 2).sum() + (wq * d1psi(1.0, t) ** 2).sum()
    quad += (wq * d2psi(t, 0.0) ** 2).sum() + (wq * d2psi(t, 1.0) ** 2).sum()
    zeta2 = float((a ** 2).sum())
    say(f"(A1') random zeta, d = 2, K = {K}: ||T zeta||^2 formula {formula:.12e}, quadrature {quad:.12e}; "
        f"ratio to ||zeta||^2 {formula/zeta2:.6f} <= U_2 = {U(2):.6f}")
    return dict(formula=formula, quadrature=float(quad), rel_diff=abs(formula - quad) / quad,
                ratio=formula / zeta2, U2=U(2))


# ----------------------------------------------------------------- (A2), (A3)
def parity_and_cs_checks():
    rng = np.random.default_rng(2)
    worst_par = 0.0
    for _ in range(200):
        n = rng.integers(1, 30)
        b = rng.standard_normal(n); kidx = np.arange(1, n + 1)
        lhs = b.sum() ** 2 + ((-1.0) ** kidx * b).sum() ** 2
        E = b[kidx % 2 == 0].sum(); O = b[kidx % 2 == 1].sum()
        worst_par = max(worst_par, abs(lhs - 2 * E * E - 2 * O * O) / max(lhs, 1e-300))
    # Cauchy-Schwarz on one fibre: fixed k_{-i} with m = |k_{-i}|^2, k_i = 1..J
    worst_cs = 0.0
    for _ in range(200):
        m = float(rng.integers(1, 50)); J = int(rng.integers(1, 200))
        ki = np.arange(1, J + 1, dtype=float); a = rng.standard_normal(J)
        k2 = ki ** 2 + m
        for par in (0, 1):
            sel = (ki % 2 == par)
            lhs = ((a * ki / k2)[sel].sum()) ** 2
            rhs = (a[sel] ** 2 * ki[sel] ** 2 / k2[sel]).sum() * odd_sum_closed(math.sqrt(m))
            worst_cs = max(worst_cs, lhs / rhs if rhs > 0 else 0.0)
    say(f"(A2) parity identity worst rel. error {worst_par:.1e}; (A3) Cauchy-Schwarz step: max lhs/rhs {worst_cs:.4f} (<= 1 required)")
    return dict(parity_worst_rel_err=worst_par, cs_max_ratio=worst_cs)


# ----------------------------------------------------------------- (A4)
def partial_fraction_checks():
    j = np.arange(1, 8_000_001, 2, dtype=float)
    rows = []
    for a in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, math.sqrt(19), 10.0, 99 ** 0.5, 30.0]:
        s = float((1 / (j ** 2 + a * a)).sum())
        tail = 1 / (2 * (j[-1] + 2 - 2))  # sum_{j odd > jmax} 1/j^2 <= 1/(2 jmax)
        rows.append(dict(a=a, series=s, closed=odd_sum_closed(a), diff=s - odd_sum_closed(a), tail_bound=tail))
    a_grid = np.linspace(1e-6, 50, 200001)
    f = np.tanh(math.pi * a_grid / 2) / a_grid
    mono = bool(np.all(np.diff(f) < 0))
    je = np.arange(2, 400001, 2, dtype=float); jo = je - 1
    even_le_odd = all(float((1 / (je ** 2 + m)).sum()) <= float((1 / (jo ** 2 + m)).sum()) for m in [1, 2, 5, 19, 99])
    worst = max(abs(r["diff"]) / r["closed"] for r in rows)
    say(f"(A4) partial fractions: worst |series - closed|/closed = {worst:.2e} (series truncated at j = 8e6, tail <= 6.3e-8); "
        f"tanh(pi a/2)/a strictly decreasing on (0, 50]: {mono}; even sum <= odd sum: {even_le_odd}")
    return dict(rows=rows, worst_rel=worst, tanh_over_a_decreasing=mono, even_le_odd=even_le_odd)


# ----------------------------------------------------------------- (A5)
def table_LU():
    rows = []
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 1000]:
        u, l = U(d), L(d)
        rows.append(dict(d=d, L=l, U=u, U_over_L=u / l, sqrt_d_U=math.sqrt(d) * u, sqrt_d_L=math.sqrt(d) * l,
                         sqrt_U=math.sqrt(u), sqrt_2dU=math.sqrt(2 * d * u),
                         article_constant_sq=4 * (1 + 1 / (d * math.pi ** 2))))
        say(f"(A5) d={d:4d}  L_d={l:.6f}  U_d={u:.6f}  U/L={u/l:.4f}  sqrt(d)U={math.sqrt(d)*u:.5f}  "
            f"sqrt(d)L={math.sqrt(d)*l:.5f}  sqrt(2dU)={math.sqrt(2*d*u):.4f}  (2C)^2 article={4*(1+1/(d*math.pi**2)):.4f}")
    say(f"(A5) 1/pi = {1/math.pi:.5f}, 1/(2 pi) = {1/(2*math.pi):.5f}; L_d series tail <= {L_tail_bound():.1e}")
    return rows


# ----------------------------------------------------------------- (A6) Lanczos
def op_TT(d, K):
    shape = (K,) * d
    grids = np.meshgrid(*[np.arange(1, K + 1, dtype=float)] * d, indexing="ij")
    k2 = sum(g ** 2 for g in grids)
    vs = [grids[i] / k2 for i in range(d)]
    masks = [[(grids[i] % 2 == par) for par in (0, 1)] for i in range(d)]
    n = K ** d

    def mv(a):
        a = a.reshape(shape)
        out = np.zeros(shape)
        for i in range(d):
            for par in (0, 1):
                s = (a * vs[i] * masks[i][par]).sum(axis=i, keepdims=True)
                out += s * vs[i] * masks[i][par]
        return (4 / math.pi ** 2) * out.reshape(-1)
    return LinearOperator((n, n), matvec=mv, dtype=float), grids


def lanczos():
    rows = []
    for d, Ks in [(2, [50, 100, 200, 400, 800]), (3, [20, 40, 60, 80]), (4, [10, 16, 24])]:
        for K in Ks:
            t0 = time.process_time()
            op, _ = op_TT(d, K)
            lam = float(eigsh(op, k=1, which="LA", tol=1e-10, return_eigenvectors=False)[0])
            rows.append(dict(d=d, K=K, lam_max=lam, L=L(d), U=U(d), lam_over_U=lam / U(d), cpu_s=time.process_time() - t0))
            say(f"(A6) d={d} K={K:4d}: largest eigenvalue {lam:.6f}  (L_d {L(d):.6f}, U_d {U(d):.6f}, ratio to U_d {lam/U(d):.4f})")
    return rows


# ----------------------------------------------------------------- (A7) lower-bound modes
def lower_modes():
    rows = []
    for d in [2, 3, 5, 10, 20, 100]:
        for J in [1, 11, 101, 1001, 100001]:
            j = np.arange(1, J + 1, 2, dtype=float)
            a = j / (j ** 2 + d - 1)
            S = float((a ** 2).sum())                       # ||zeta_J||^2 = S_J
            face1 = 4 / math.pi ** 2 * S                    # (4/pi^2) O_1^2 / ||zeta||^2 = (4/pi^2) S_J
            others = 4 / math.pi ** 2 * (d - 1) * float((a ** 2 / (j ** 2 + d - 1) ** 2).sum()) / S
            rq = face1 + others
            rows.append(dict(d=d, J=J, S_J_bound=face1, rayleigh=rq, L=L(d), U=U(d)))
        say(f"(A7) d={d}: (4/pi^2) S_J for J=1..1e5: " + ", ".join(f"{r['S_J_bound']:.6f}" for r in rows[-5:])
            + f"; full Rayleigh quotient at J=1e5: {rows[-1]['rayleigh']:.6f}; L_d={L(d):.6f}; U_d={U(d):.6f}")
    # cross-check the closed Rayleigh quotient with the Lanczos operator (d = 2, 3)
    cross = []
    for d, K, J in [(2, 60, 59), (3, 30, 29)]:
        op, grids = op_TT(d, K)
        z = np.zeros((K,) * d)
        j = np.arange(1, J + 1, 2)
        idx = tuple([j - 1] + [np.zeros_like(j)] * (d - 1))
        z[idx] = j / (j ** 2 + d - 1.0)
        zz = z.reshape(-1)
        rq_op = float(zz @ op.matvec(zz) / (zz @ zz))
        jf = j.astype(float); a = jf / (jf ** 2 + d - 1); S = float((a ** 2).sum())
        rq_closed = 4 / math.pi ** 2 * (S + (d - 1) * float((a ** 2 / (jf ** 2 + d - 1) ** 2).sum()) / S)
        cross.append(dict(d=d, J=J, rq_operator=rq_op, rq_closed=rq_closed))
        say(f"(A7') d={d}, J={J}: Rayleigh quotient by operator {rq_op:.12f}, closed formula {rq_closed:.12f}")
    return dict(rows=rows, cross=cross)


def main():
    t0 = time.time()
    out = {}
    out["A1_face"] = face_formula_check()
    out["A1_random_zeta_d2"] = random_zeta_check()
    out["A2_A3"] = parity_and_cs_checks()
    out["A4_partial_fractions"] = partial_fraction_checks()
    out["A5_table"] = table_LU()
    dump("theorem2_checks.json", out)
    out["A7_lower_modes"] = lower_modes()
    dump("theorem2_checks.json", out)
    out["A6_lanczos"] = lanczos()
    out["wall_s"] = time.time() - t0
    dump("theorem2_checks.json", out)
    with open(os.path.join(RES, "theorem2_checks.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
