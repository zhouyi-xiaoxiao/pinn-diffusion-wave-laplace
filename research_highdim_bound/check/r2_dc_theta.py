"""Round-2 check: first-order bias coefficient c(d) = ||v1||/||u*|| for P1 (u* = sum_p x_{2p-1}x_{2p})
computed from the DIRICHLET sine expansion of the harmonic extension (not from Robin eigenfunctions),
with the one-dimensional sums in closed form (Poisson summation; exponentially small remainders for
t < T0) and direct sums for t >= T0.  No truncation in the number of modes.

<v1, e_k> = (1/2) int_bd h T e_k  (definition of v1),  T e_k from the face formula of Theorem 2, Step 1.
||v1||^2 = m/(2 pi^2) int_0^inf t [2 P R + (16/pi^2) E^2] G^(d-2) dt  with
  P = sum_{k even} 4k^2 e^{-tk^2},  R = sum_{k>=1} 2/(pi^2 k^2) e^{-tk^2},
  E = sum_{k even} e^{-tk^2},       G = sum_{k odd} 8/(pi^2 k^2) e^{-tk^2}.
Leading-order Laplace asymptotics give ||v1||^2 ~ m/(12 d), ||u*||^2 ~ m^2/16, so d c(d) -> sqrt(8/3).
"""
import json, math, sys
import numpy as np
from scipy import integrate

T0 = 0.05
SQPI = math.sqrt(math.pi)

def sums_direct(t, K=400):
    k = np.arange(1, K + 1, dtype=float)
    ek = np.exp(-t * k * k)
    even = (k % 2 == 0); odd = ~even
    P = np.sum(4 * k[even] ** 2 * ek[even])
    R = np.sum(2 / (math.pi ** 2 * k * k) * ek)
    E = np.sum(ek[even])
    G = np.sum(8 / (math.pi ** 2 * k[odd] ** 2) * ek[odd])
    return P, R, E, G

def integrand(t, d):
    m = d // 2
    if t < T0:
        st = math.sqrt(t)
        P = SQPI / (2 * t * st)
        R = (2 / math.pi ** 2) * (math.pi ** 2 / 6 - SQPI * st + t / 2)
        E = 0.25 * SQPI / st - 0.5
        logG = math.log1p(-4 * st / math.pi ** 1.5)
    else:
        P, R, E, G = sums_direct(t)
        logG = math.log(G)
    core = 2 * P * R + (16 / math.pi ** 2) * E * E
    return m / (2 * math.pi ** 2) * t * core * math.exp((d - 2) * logG)

def v1sq(d):
    # integrate in s = log t
    f = lambda s: integrand(math.exp(s), d) * math.exp(s)
    # scale of the peak: t* ~ (pi^1.5/(4 d))^2
    tpk = (math.pi ** 1.5 / (4 * max(d - 2, 1))) ** 2
    pts = sorted(set([math.log(tpk) + k for k in (-12, -6, -3, 0, 3, 6)] + [math.log(T0)]))
    lo, hi = math.log(tpk) - 40, math.log(400.0)
    pts = [p for p in pts if lo < p < hi]
    edges = [lo] + pts + [hi]
    tot, err = 0.0, 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        v, e = integrate.quad(f, a, b, epsabs=0, epsrel=1e-12, limit=400)
        tot += v; err += e
    return tot, err

def usq(d):
    m = d // 2
    return m / 9 + m * (m - 1) / 16

def sanity_direct_sum_d2(K=3000):
    """Direct Dirichlet sum for d = 2 (no t-integral): <v1,e_k> for k = (k1,k2)."""
    k1 = np.arange(1, K + 1, dtype=float)[:, None]; k2 = np.arange(1, K + 1, dtype=float)[None, :]
    A = lambda k: math.sqrt(2) * (-1) ** (k + 1) / (math.pi * k)
    ev = lambda k: (k % 2 == 0).astype(float)
    k2sq = k1 ** 2 + k2 ** 2
    coef = 0.5 * math.sqrt(2) / (math.pi * k2sq) * (2 * k1 * ev(k1) * A(k2) + 2 * k2 * ev(k2) * A(k1))
    return float(np.sum(coef ** 2))

if __name__ == "__main__":
    out = {"method": "Dirichlet sine expansion, closed-form 1-D sums for t<0.05, scipy quad in log t",
           "limit_sqrt_8_over_3": math.sqrt(8 / 3), "rows": []}
    # d = 2 sanity: direct double sum (truncated) vs integral
    v2, _ = v1sq(2)
    ds = [sanity_direct_sum_d2(K) for K in (500, 1000, 2000)]
    out["d2_check"] = {"integral": v2, "direct_sum_K500_1000_2000": ds}
    print("d=2 integral", v2, "direct sums", ds, flush=True)
    for d in [2, 3, 4, 5, 10, 20, 50, 100, 300, 1000, 3000, 10000, 100000, 1000000]:
        v, e = v1sq(d)
        c = math.sqrt(v / usq(d))
        row = dict(d=d, v1_sq=v, quad_err=e, c=c, d_times_c=d * c)
        out["rows"].append(row)
        print(row, flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
