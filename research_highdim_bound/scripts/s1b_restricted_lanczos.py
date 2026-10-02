"""Lower bounds for C_d^2 = ||T||^2 in higher dimension by Rayleigh-Ritz on subspaces (numerical).

For any subspace S of L2(Omega), max_{zeta in S} ||T zeta||^2/||zeta||^2 <= C_d^2. We use
  S(d, r, K, parity) = span{ e_k : every k_i in the allowed set, at most r coordinates of k differ from 1,
                             k_i <= K },
with parity 'odd' (all k_i odd) or 'all'. T*T maps the coefficients by
  (T*T a)_k = (4/pi^2) sum_i v_i(k) sum_{k': k'_{-i} = k_{-i}, k'_i = k_i mod 2} v_i(k') a_{k'},  v_i(k) = k_i/|k|^2
(the working document of the proofs, eq. (A.3)); restricted to S this is P T*T P, whose largest eigenvalue is a lower bound.
It also checks on the full truncated space (d = 2, 3) that T*T preserves the parity pattern of k and
that the largest eigenvalue sits in the all-odd block.
Outputs: results/restricted_lanczos.json, results/restricted_lanczos.txt."""
import itertools, json, math, os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from t2common import U, L, RES, dump

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def index_set(d, r, K, parity):
    vals = list(range(3, K + 1, 2)) if parity == "odd" else list(range(2, K + 1))
    blocks = [np.ones((1, d), dtype=np.int16)]
    for q in range(1, r + 1):
        combos = np.array(list(itertools.combinations(range(d), q)), dtype=np.int64)
        prods = np.array(list(itertools.product(vals, repeat=q)), dtype=np.int16)
        n = len(combos) * len(prods)
        if n == 0:
            continue
        A = np.ones((n, d), dtype=np.int16)
        ci = np.repeat(np.arange(len(combos)), len(prods))
        pi = np.tile(np.arange(len(prods)), len(combos))
        for t in range(q):
            A[np.arange(n), combos[ci, t]] = prods[pi, t]
        blocks.append(A)
    return np.vstack(blocks)


def build_B(k):
    n, d = k.shape
    kf = k.astype(float)
    k2 = (kf ** 2).sum(1)
    rows, cols, data = [], [], []
    off = 0
    for i in range(d):
        key = k.copy()
        key[:, i] = np.where(k[:, i] % 2 == 1, -1, -2)  # fibre = (k_{-i}, parity of k_i)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        rows.append(inv + off); cols.append(np.arange(n)); data.append(kf[:, i] / k2)
        off += inv.max() + 1
    B = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(off, n))
    return B


def top_eig(k):
    B = build_B(k)
    Bt = B.T.tocsr()
    n = k.shape[0]
    op = LinearOperator((n, n), matvec=lambda a: (4 / math.pi ** 2) * (Bt @ (B @ a)), dtype=float)
    if n == 1:
        return float((4 / math.pi ** 2) * (B.toarray() ** 2).sum())
    return float(eigsh(op, k=1, which="LA", tol=1e-9, return_eigenvectors=False)[0])


def parity_block_check():
    out = []
    for d, K in [(2, 120), (3, 30)]:
        full = top_eig(index_set(d, d, K, "all"))
        odd = top_eig(index_set(d, d, K, "odd"))
        out.append(dict(d=d, K=K, full=full, all_odd_block=odd))
        say(f"parity check d={d} K={K}: top eigenvalue full {full:.6f}, all-odd block {odd:.6f}")
    return out


def main():
    out = dict(parity=parity_block_check(), rows=[])
    dump("restricted_lanczos.json", out)
    cfgs = [(2, 2, 799), (3, 3, 79), (3, 2, 199), (4, 2, 199), (4, 3, 59), (4, 4, 29), (5, 2, 199), (5, 3, 59),
            (5, 4, 29), (5, 5, 19), (10, 1, 3999), (10, 2, 199), (10, 3, 39), (20, 1, 3999), (20, 2, 99),
            (20, 3, 23), (50, 1, 3999), (50, 2, 41), (100, 1, 3999), (100, 2, 21)]
    for d, r, K in cfgs:
        t0 = time.process_time()
        k = index_set(d, r, K, "odd")
        if k.shape[0] > 2_500_000:
            say(f"skip d={d} r={r} K={K}: n={k.shape[0]}"); continue
        lam = top_eig(k)
        row = dict(d=d, r=r, K=K, n=int(k.shape[0]), lam=lam, L=L(d), U=U(d), lam_over_U=lam / U(d),
                   lam_over_L=lam / L(d), cpu_s=time.process_time() - t0)
        out["rows"].append(row)
        say(f"d={d:3d} r={r} K={K:4d} n={k.shape[0]:8d}: lower bound {lam:.6f}  (L_d {L(d):.6f}, U_d {U(d):.6f}; "
            f"ratio to U_d {lam/U(d):.4f}, to L_d {lam/L(d):.4f}; cpu {row['cpu_s']:.0f}s)")
        dump("restricted_lanczos.json", out)
        with open(os.path.join(RES, "restricted_lanczos.txt"), "w") as f:
            f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
