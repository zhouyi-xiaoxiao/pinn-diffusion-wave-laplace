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

"""2-D finite-difference solution of the penalised-Ritz Euler-Lagrange problem
   -Lap u = 0 in (0,1)^2,  d_n u + 2*beta*(u - g) = 0 on the boundary,  g = x*y
to test the study's claim that the penalised Deep Ritz minimiser carries an O(1/beta) Robin bias on P1."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, json
def solve(beta, N):
    h = 1.0 / N; n = N + 1
    idx = lambda i, j: i * n + j
    rows, cols, vals = [], [], []; b = np.zeros(n * n)
    xs = np.linspace(0, 1, n)
    for i in range(n):
        for j in range(n):
            k = idx(i, j); diag = -4.0; rhs = 0.0
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii <= N and 0 <= jj <= N:
                    rows.append(k); cols.append(idx(ii, jj)); vals.append(1.0)
                else:
                    # ghost = mirror - 4 h beta (u_k - g_k)
                    rows.append(k); cols.append(idx(i - di, j - dj)); vals.append(1.0)
                    diag += -4 * h * beta; rhs += -4 * h * beta * xs[i] * xs[j]
            rows.append(k); cols.append(k); vals.append(diag); b[k] = rhs
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))
    u = spl.spsolve(A, b).reshape(n, n)
    return xs, u
def rel(u, xs):
    X, Y = np.meshgrid(xs, xs, indexing="ij"); ue = X * Y
    w = np.ones_like(xs); w[0] = w[-1] = 0.5; Wt = np.outer(w, w)
    return np.sqrt((Wt * (u - ue) ** 2).sum() / (Wt * ue ** 2).sum()), np.abs(u - ue).max()
out = {}
for beta in [1, 10, 100, 1000]:
    r = {}
    for N in [100, 200]:
        xs, u = solve(beta, N); r[N] = rel(u, xs)
    out[beta] = dict(rel_l2_N100=float(r[100][0]), rel_l2_N200=float(r[200][0]), max_abs_N200=float(r[200][1]))
    print(f"beta={beta}: Robin-minimiser rel L2 distance from u*=xy: N=100 {r[100][0]:.4e}, N=200 {r[200][0]:.4e}; max abs {r[200][1]:.3e}; beta*rel = {beta*r[200][0]:.3f}")
    np.save(_repo_path(f"verify_research_highdim/results/robin_u_beta{beta}.npy"), u)
json.dump(out, open(_repo_path("verify_research_highdim/results/v_robin_fd.json"), "w"), indent=1)
