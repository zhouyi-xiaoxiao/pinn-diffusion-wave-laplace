"""Harmonic-basis Rayleigh-Ritz lower bounds for C_d^2 in d = 2..6 (separately written; extension side).
Basis: for each of the 2d faces, P_kappa(x_a) * prod_{b != a} sin(k_b pi x_b), with all k_b ODD and <= Kmax,
kappa = pi |k|, P harmonic profile (1 on the face, 0 on the opposite face). Each basis function vanishes on
all other faces, so the boundary Gram matrix is I / 2^(d-1). Interior Gram: products of 1-D Gauss-Legendre
integrals. Every value is a lower bound for C_d^2 (subspace), hence must be <= U_d.
Usage: c1b_harmonic_basis.py d Kmax [Kmax ...]
"""
import itertools, json, math, os, sys, time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def U(d):
    a = math.sqrt(d - 1)
    return math.tanh(math.pi * a / 2) / (math.pi * a)


def gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w


def P(kappa, x):
    return (np.exp(-kappa * x) - np.exp(-kappa * (2 - x))) / (1 - np.exp(-2 * kappa))


def lower(d, Kmax, nq=800):
    x, w = gl(nq)
    odd = list(range(1, Kmax + 1, 2))
    idx = np.array(list(itertools.product(odd, repeat=d - 1)), dtype=float)  # (n, d-1)
    n = len(idx)
    kap = np.pi * np.sqrt((idx ** 2).sum(1))
    Pm = P(kap[:, None], x[None, :]); Pr = Pm[:, ::-1]
    faces = [(a, s) for a in range(d) for s in range(2)]
    F = []
    for a, s in faces:
        f = {a: (Pm if s == 0 else Pr)}
        others = [b for b in range(d) if b != a]
        for col, b in enumerate(others):
            f[b] = np.sin(np.pi * np.outer(idx[:, col], x))
        F.append(f)
    ip = lambda A, B: (A * w) @ B.T
    nf = len(faces)
    A = np.zeros((nf * n, nf * n))
    for i in range(nf):
        for j in range(i, nf):
            blk = np.ones((n, n))
            for ax in range(d):
                blk *= ip(F[i][ax], F[j][ax])
            A[i * n:(i + 1) * n, j * n:(j + 1) * n] = blk
            if j != i:
                A[j * n:(j + 1) * n, i * n:(i + 1) * n] = blk.T
    A = 0.5 * (A + A.T)
    if nf * n > 7000:
        from scipy.sparse.linalg import eigsh
        top = eigsh(A, k=1, which='LA', tol=1e-12)[0][0]
    else:
        top = np.linalg.eigvalsh(A)[-1]
    return float(top * 2 ** (d - 1)), nf * n


if __name__ == "__main__":
    d = int(sys.argv[1])
    path = os.path.join(OUT, "c1b_harmonic_basis.jsonl")
    for K in map(int, sys.argv[2:]):
        t0 = time.time()
        v, size = lower(d, K)
        rec = dict(d=d, Kmax_odd=K, basis_size=size, lower=v, U=U(d), over_U=v / U(d), seconds=time.time() - t0)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
