"""Computed as a separate check: penalty bias of the penalised Ritz energy for LapD, Galerkin with tensor Legendre
polynomials of total degree <= p that depend on at most r coordinates (r = d gives the full space).
Checks the convergence in p and r.  Same weak form as robin_bias_d.py."""
import itertools, math, sys, time
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from numpy.polynomial import legendre as L

def one_d(p):
    xg, wg = L.leggauss(p + 2)
    x = 0.5 * (xg + 1); w = 0.5 * wg
    dV = np.zeros((p + 1, len(x)))
    for j in range(1, p + 1):
        c = np.zeros(j + 1); c[j] = 1
        dV[j] = 2 * L.legval(2 * x - 1, L.legder(c))
    M = 1.0 / (2 * np.arange(p + 1) + 1)
    D = (dV * w) @ dV.T
    return M, D, (-1.0) ** np.arange(p + 1), np.ones(p + 1)

def basis(d, p, r):
    out = [tuple([0] * d)]
    for k in range(1, min(r, d) + 1):
        for coords in itertools.combinations(range(d), k):
            # positive degrees a_1..a_k with sum <= p
            def rec(prefix, left, i):
                if i == k:
                    a = [0] * d
                    for c, v in zip(coords, prefix): a[c] = v
                    out.append(tuple(a)); return
                for v in range(1, left - (k - i - 1) + 1):
                    rec(prefix + [v], left - v, i + 1)
            rec([], p, 0)
    return out

def solve(d, p, r, betas):
    M, D, v0, v1 = one_d(p)
    idx = basis(d, p, r)
    pos = {a: i for i, a in enumerate(idx)}
    n = len(idx)
    A = np.array(idx)
    Mprod = np.prod(M[A], axis=1)
    rk, ck, vk, rb, cb, vb = [], [], [], [], [], []
    for i, a in enumerate(idx):
        tot = sum(a)
        for l in range(d):
            base = Mprod[i] / M[a[l]]
            for c in range(0, p - (tot - a[l]) + 1):
                j = pos.get(a[:l] + (c,) + a[l + 1:])
                if j is None: continue
                if D[a[l], c] != 0.0:
                    rk.append(i); ck.append(j); vk.append(D[a[l], c] * base)
                rb.append(i); cb.append(j); vb.append((v0[a[l]] * v0[c] + 1.0) * base)
    K = sp.csr_matrix((vk, (rk, ck)), shape=(n, n)); B = sp.csr_matrix((vb, (rb, cb)), shape=(n, n))
    m = d // 2
    xint = {0: 0.5, 1: 1.0 / 6.0}
    rhs = np.zeros(n)
    for gi, g in enumerate(idx):
        nz = [l for l in range(d) if g[l] != 0]
        s = 0.0
        for k in range(m):
            for (i, j) in ((2 * k, 2 * k + 1), (2 * k + 1, 2 * k)):
                if any(l not in (i, j) for l in nz) or g[j] not in xint: continue
                s += (v1[g[i]] - v0[g[i]]) * xint[g[j]]
        rhs[gi] = -s
    unorm = math.sqrt((9 * m * m + 7 * m) / 144.0)
    out = {}
    for beta in betas:
        c = spla.spsolve((K + 2 * beta * B).tocsc(), rhs)
        out[beta] = math.sqrt(np.sum(c * c * Mprod)) / unorm
    return n, out

if __name__ == "__main__":
    cases = [(2, 12, 2), (5, 10, 5), (5, 10, 2), (5, 10, 3), (5, 10, 4), (5, 14, 2), (5, 14, 3),
             (10, 6, 10), (10, 6, 2), (10, 6, 3), (10, 10, 2), (10, 10, 3), (10, 14, 2), (10, 12, 3),
             (20, 4, 20), (20, 4, 2), (20, 10, 2), (20, 14, 2), (20, 8, 3)]
    for d, p, r in cases:
        t0 = time.time()
        n, res = solve(d, p, r, (1, 10, 100, 1000))
        print(f"d={d:2d} p={p:2d} r={r:2d} n={n:6d} " + "  ".join(f"b={b}: {v:.4e}" for b, v in res.items()),
              f"[{time.time()-t0:.0f}s]", flush=True)
