"""Numerical checks of Corollary 2 (a), (c) and of the maximum-principle bound (Remark 4 of the working document of the proofs)
on explicit functions, with norms computed from exact 1-D integrals (no Monte Carlo).

(a) harmonic w = exp(kappa x_1) prod_{i>=2} cos(om_i x_i + c_i), kappa^2 = sum om_i^2, and
    w = cosh(kappa (x_1 - 1/2)) prod cos(om_i (x_i - 1/2)): ||w||^2 / ||w||_bd^2 <= U_d.
(c) v = u* + p with separable p = eps prod_i phi(x_i): ||p|| <= ||Lap p||/(d pi^2) + sqrt(U_d) ||p||_bd,
    using A = int phi^2, B = int phi''^2, C = int phi phi'':
      ||p||^2 = A^d, ||Lap p||^2 = d B A^{d-1} + d(d-1) C^2 A^{d-2}, ||p||_bd^2 = d (phi(0)^2 + phi(1)^2) A^{d-1}
    (each checked against Monte Carlo for d = 3).
(MP) comparison function q(x) = (d/4 - |x - c|^2)/(2d): Lap q = -1, 0 <= q <= 1/8 on the closed cube;
    max|e| <= max_bd|e| + max|Lap e|/8 on test functions (d = 1, 2, 3, 5), equality for e = x(1-x)/2, d = 1.
Output: results/corollary_checks.json, results/corollary_checks.txt."""
import json, math, os, sys
import numpy as np
from scipy import integrate
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from t2common import U, RES, dump

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

GX, GW = np.polynomial.legendre.leggauss(200)
GX = 0.5 * (GX + 1); GW = 0.5 * GW


def q1(f):
    return float((GW * f(GX)).sum())


def harmonic_checks():
    rng = np.random.default_rng(3)
    rows = []
    for d in [2, 3, 5, 10, 20, 50]:
        worst = 0.0
        for trial in range(200):
            om = rng.uniform(0, 3 * math.pi, d - 1) * rng.integers(0, 2, d - 1)
            c = rng.uniform(0, 2 * math.pi, d - 1)
            kappa = math.sqrt((om ** 2).sum())
            if trial % 2 == 0:
                f1 = lambda t: np.exp(kappa * t)
            else:
                f1 = lambda t: np.cosh(kappa * (t - 0.5))
            fs = [f1] + [(lambda t, o=o, cc=cc: np.cos(o * t + cc)) for o, cc in zip(om, c)]
            A = [q1(lambda t, f=f: f(t) ** 2) for f in fs]
            E = [float(f(0.0) ** 2 + f(1.0) ** 2) for f in fs]
            vol = float(np.prod(A))
            bd = sum(E[i] * np.prod([A[j] for j in range(d) if j != i]) for i in range(d))
            worst = max(worst, vol / bd)
        rows.append(dict(d=d, max_ratio=worst, U=U(d), ok=worst <= U(d)))
        say(f"(a) d={d}: max ||w||^2/||w||_bd^2 over 200 harmonic test functions {worst:.5f} <= U_d {U(d):.5f}: {worst <= U(d)}")
    return rows


PHIS = {
    "cos(2t+0.3)": (lambda t: np.cos(2 * t + 0.3), lambda t: -4 * np.cos(2 * t + 0.3)),
    "1+t^2": (lambda t: 1 + t ** 2, lambda t: 2 + 0 * t),
    "exp(t)": (lambda t: np.exp(t), lambda t: np.exp(t)),
    "sin(pi t)+0.1": (lambda t: np.sin(math.pi * t) + 0.1, lambda t: -math.pi ** 2 * np.sin(math.pi * t)),
    "sin(5 pi t)+0.02": (lambda t: np.sin(5 * math.pi * t) + 0.02, lambda t: -25 * math.pi ** 2 * np.sin(5 * math.pi * t)),
    "t(1-t)": (lambda t: t * (1 - t), lambda t: -2 + 0 * t),
}


def sep_norms(phi, phi2, d):
    A = q1(lambda t: phi(t) ** 2); B = q1(lambda t: phi2(t) ** 2); C = q1(lambda t: phi(t) * phi2(t))
    e2 = A ** d
    lap2 = d * B * A ** (d - 1) + d * (d - 1) * C ** 2 * A ** (d - 2)
    bd2 = d * float(phi(0.0) ** 2 + phi(1.0) ** 2) * A ** (d - 1)
    return math.sqrt(e2), math.sqrt(max(lap2, 0)), math.sqrt(bd2)


def corollary_c_checks():
    rows = []
    for name, (phi, phi2) in PHIS.items():
        for d in [1, 2, 3, 5, 10, 20, 50, 100]:
            e, lap, bd = sep_norms(phi, phi2, d)
            bnd = lap / (d * math.pi ** 2) + math.sqrt(U(d)) * bd
            rows.append(dict(phi=name, d=d, err=e, B_int=lap / (d * math.pi ** 2), B_bd=math.sqrt(U(d)) * bd,
                             bound=bnd, eta=bnd / e, ok=bnd >= e))
        say(f"(c) phi={name:18s} eta for d=1..100: " + ", ".join(f"{r['eta']:.3f}" for r in rows[-8:]))
    # Monte-Carlo cross-check of the three separable formulas at d = 3
    rng = np.random.default_rng(4)
    mc = []
    for name, (phi, phi2) in list(PHIS.items())[:3]:
        d = 3; n = 2_000_000
        x = rng.random((n, d))
        P = np.prod(phi(x), axis=1)
        lap = sum(phi2(x[:, i]) * np.prod(phi(np.delete(x, i, axis=1)), axis=1) for i in range(d))
        xb = rng.random((n, d)); face = rng.integers(0, d, n); side = rng.integers(0, 2, n)
        xb[np.arange(n), face] = side
        Pb = np.prod(phi(xb), axis=1)
        e, l, b = sep_norms(phi, phi2, d)
        mc.append(dict(phi=name, err=(e, math.sqrt((P ** 2).mean())), lap=(l, math.sqrt((lap ** 2).mean())),
                       bd=(b, math.sqrt(2 * d * (Pb ** 2).mean()))))
        say(f"(c') d=3 {name}: formula vs MC  err {e:.5f}/{mc[-1]['err'][1]:.5f}  lap {l:.5f}/{mc[-1]['lap'][1]:.5f}  bd {b:.5f}/{mc[-1]['bd'][1]:.5f}")
    allok = all(r["ok"] for r in rows)
    say(f"(c) bound >= error in all {len(rows)} cases: {allok}; eta range {min(r['eta'] for r in rows):.3f}-{max(r['eta'] for r in rows):.3f}")
    return dict(rows=rows, mc=mc, all_ok=allok)


def maxprinciple_checks():
    rng = np.random.default_rng(5)
    out = {}
    # comparison function
    for d in [1, 2, 3, 5, 10, 20]:
        x = rng.random((200_000, d))
        c = 0.5
        q = (d / 4 - ((x - c) ** 2).sum(1)) / (2 * d)
        lapq = -2 * d / (2 * d)  # Lap |x-c|^2 = 2d
        out[f"q_d{d}"] = dict(min_sampled=float(q.min()), max_sampled=float(q.max()), max_exact=1 / 8, lap=lapq,
                              corner_value=float((d / 4 - d * 0.25) / (2 * d)))
    say("(MP) q = (d/4 - |x-c|^2)/(2d): Lap q = -1, q(corner) = 0, q(centre) = 1/8; sampled min/max: "
        + ", ".join(f"d={k[3:]}: {v['min_sampled']:.3f}/{v['max_sampled']:.4f}" for k, v in out.items()))
    # inequality on test errors e = prod phi (exact max over the cube via 1-D grids for |prod| = prod|.|)
    t = np.linspace(0, 1, 20001)
    rows = []
    for name, (phi, phi2) in PHIS.items():
        for d in [1, 2, 3, 5]:
            m1 = np.abs(phi(t)).max(); mb1 = max(abs(float(phi(0.0))), abs(float(phi(1.0))))
            max_e = m1 ** d
            max_bd = mb1 * m1 ** (d - 1)
            # max |Lap e|: dense samples (an under-estimate, which makes the check stricter)
            if d == 1:
                max_lap = np.abs(phi2(t)).max()
            else:
                x = np.vstack([rng.random((400_000, d)), np.array(np.meshgrid(*[np.linspace(0, 1, int(round(400_000 ** (1 / d))))] * d)).reshape(d, -1).T])
                lap = sum(phi2(x[:, i]) * np.prod(phi(np.delete(x, i, axis=1)), axis=1) for i in range(d))
                max_lap = np.abs(lap).max()
            rhs = max_bd + max_lap / 8
            rows.append(dict(phi=name, d=d, max_err=max_e, max_bd=max_bd, max_lap_sampled=float(max_lap), rhs=rhs, ok=max_e <= rhs + 1e-12))
    eq = dict(e="x(1-x)/2, d=1", max_err=1 / 8, rhs=0 + 1 / 8)
    allok = all(r["ok"] for r in rows)
    say(f"(MP) max|e| <= max_bd|e| + max|Lap e|/8 in all {len(rows)} cases: {allok}; equality case {eq}")
    return dict(comparison=out, rows=rows, equality=eq, all_ok=allok)


def main():
    res = dict(harmonic=harmonic_checks(), corollary_c=corollary_c_checks(), maxprinciple=maxprinciple_checks())
    dump("corollary_checks.json", res)
    with open(os.path.join(RES, "corollary_checks.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
