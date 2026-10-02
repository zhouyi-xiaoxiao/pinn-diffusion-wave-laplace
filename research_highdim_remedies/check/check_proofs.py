"""Numerical check of the working document of the proofs (Lemma 1, Props 1-4; numbering mapped in README.md), written separately from src/.
Edge cases: d = 1, 2, 3, large d; small and large penalty weights."""
import math, itertools, json, sys
import numpy as np
from scipy import integrate
out = {}

# ---------- Lemma 1: brute-force M by exact face sums (moments of uniform on [0,1]: E x = 1/2, E x^2 = 1/3)
def M_brute(d):
    M = np.zeros((d + 1, d + 1))
    for k in range(d):
        for s in (0.0, 1.0):
            # moments on face x_k = s
            m1 = np.full(d, 0.5); m1[k] = s
            m2 = np.full(d, 1/3); m2[k] = s * s
            F = np.zeros((d + 1, d + 1)); F[0, 0] = 1; F[0, 1:] = F[1:, 0] = m1
            F[1:, 1:] = np.outer(m1, m1); np.fill_diagonal(F[1:, 1:], m2)
            M += F
    return M
def M_closed(d):
    M = np.full((d + 1, d + 1), d / 2.0); M[0, 0] = 2 * d; M[0, 1:] = M[1:, 0] = d
    np.fill_diagonal(M[1:, 1:], 1 + 2 * (d - 1) / 3); M[0, 0] = 2 * d; return M
L1 = {}
for d in (1, 2, 3, 5, 20, 100, 400):
    Mb, Mc = M_brute(d), M_closed(d)
    ev = np.sort(np.linalg.eigvalsh(Mc))
    a = 1 + 2 * (d - 1) / 3
    B = np.array([[2 * d, d * math.sqrt(d)], [d * math.sqrt(d), a + (d - 1) * d / 2]])
    pred = np.sort(np.r_[[(d + 2) / 6] * (d - 1), np.linalg.eigvalsh(B)])
    L1[d] = dict(closed_vs_brute=float(np.abs(Mb - Mc).max()), eig_rel=float(np.abs(ev - pred).max() / ev.max()),
                 detB_rel=abs(np.linalg.det(B) - d * (d + 2) / 3) / (d * (d + 2) / 3), lam_min=float(ev[0]),
                 cond=float(ev[-1] / ev[0]))
out["L1"] = L1
# MC check of M at d = 4 (sampler written here)
rng = np.random.default_rng(5)
d = 4; n = 2_000_000
X = rng.random((n, d)); f = rng.integers(0, d, n); X[np.arange(n), f] = rng.integers(0, 2, n)
Phi = np.c_[np.ones(n), X]; Mmc = 2 * d * Phi.T @ Phi / n
out["L1_mc_d4_maxabs"] = float(np.abs(Mmc - M_closed(d)).max())

# ---------- Prop 1 and 2: affine minimisers, brute force for d = 1, 2, 3 via tensor quadrature of the population loss
gx, gw = np.polynomial.legendre.leggauss(40); gx = 0.5 * (gx + 1); gw = 0.5 * gw
def problem(name, d):
    m = d // 2
    if name == "P1":
        u = lambda X: sum(X[..., 2*k] * X[..., 2*k+1] for k in range(m)) + 0 * X[..., 0]; kap = 0.0
    elif name == "P2":
        u = lambda X: np.cos(np.pi * X).sum(-1) / math.sqrt(d); kap = math.pi ** 2
    elif name == "P3":
        u = lambda X: np.cos(2 * (2 * X - 1).sum(-1) / math.sqrt(d)); kap = 16.0
    elif name == "P4":
        u = lambda X: sum(np.cos(np.pi*X[..., 2*k]) * np.cos(np.pi*X[..., 2*k+1]) for k in range(m)) + 0 * X[..., 0]; kap = 2 * math.pi ** 2
    return u, kap
def grids(d):
    G = np.stack(np.meshgrid(*[gx] * d, indexing="ij"), -1).reshape(-1, d)
    W = np.prod(np.stack(np.meshgrid(*[gw] * d, indexing="ij"), -1).reshape(-1, d), 1)
    faces = []
    if d == 1:
        for s in (0., 1.): faces.append((np.array([[s]]), np.array([1.0])))
    else:
        Gf = np.stack(np.meshgrid(*[gx] * (d - 1), indexing="ij"), -1).reshape(-1, d - 1)
        Wf = np.prod(np.stack(np.meshgrid(*[gw] * (d - 1), indexing="ij"), -1).reshape(-1, d - 1), 1)
        for k in range(d):
            for s in (0., 1.):
                X = np.insert(Gf, k, s, axis=1); faces.append((X, Wf))
    return G, W, faces
def brute(name, d, method, w):
    u, kap = problem(name, d); G, W, faces = grids(d)
    Phi = np.c_[np.ones(len(G)), G]; f = kap * u(G)
    rf = Phi.T @ (W * f)
    M = np.zeros((d + 1, d + 1)); rg = np.zeros(d + 1)
    for X, Wf in faces:
        P = np.c_[np.ones(len(X)), X]; M += P.T @ (Wf[:, None] * P); rg += P.T @ (Wf * u(X))
    E = np.eye(d + 1); E[0, 0] = 0
    # check M against closed form independently
    assert np.abs(M - M_closed(d)).max() < 1e-12, (d, np.abs(M - M_closed(d)).max())
    if method == "pinn":
        # minimise the population PINN loss by least squares over the boundary quadrature (Lap p = 0)
        A = np.vstack([np.sqrt(Wf)[:, None] * np.c_[np.ones(len(X)), X] for X, Wf in faces])
        b = np.concatenate([np.sqrt(Wf) * u(X) for X, Wf in faces])
        return np.linalg.lstsq(A, b, rcond=None)[0], rf, rg
    # Deep Ritz: minimise 1/2 c^T E c - c.rf + w (c^T M c - 2 c.rg) by a generic optimiser (not the normal equations)
    from scipy.optimize import minimize
    Hs = E + 2 * w * M
    obj = lambda c: 0.5 * c @ E @ c - c @ rf + w * (c @ M @ c - 2 * c @ rg)
    jac = lambda c: E @ c - rf + 2 * w * (M @ c - rg)
    r = minimize(obj, np.zeros(d + 1), jac=jac, method="BFGS", options=dict(gtol=1e-13, maxiter=10000))
    return r.x, rf, rg
sys.path.insert(0, "../src")
import presolve as ps
P12 = []
for name in ("P1", "P2", "P3", "P4"):
    for d in (1, 2, 3):
        for method, ws in (("ritz", (1e-3, 1.0, 100.0, 1e4)), ("pinn", (1000.0,))):
            for w in ws:
                cb, rf, rg = brute(name, d, method, w)
                cs = ps.affine_minimiser({"P1": "laplace", "P2": "poisson", "P3": "ridge", "P4": "cospair"}[name], d, method, w)
                rf2, rg2 = ps.moments({"P1": "laplace", "P2": "poisson", "P3": "ridge", "P4": "cospair"}[name], d)
                P12.append(dict(p=name, d=d, method=method, w=w, max_abs_c=float(np.abs(cb - cs).max()),
                                rel_c=float(np.abs(cb - cs).max() / max(1e-300, np.abs(cb).max())),
                                moments=float(max(np.abs(rf - rf2).max(), np.abs(rg - rg2).max())),
                                c_brute=[float(x) for x in cb]))
out["P1P2_brute_d123"] = P12
out["P1P2_worst_rel"] = max(r["rel_c"] for r in P12 if max(abs(x) for x in r["c_brute"]) > 1e-10)
out["P1P2_worst_abs_when_c_zero"] = max(r["max_abs_c"] for r in P12 if max(abs(x) for x in r["c_brute"]) <= 1e-10) if any(max(abs(x) for x in r["c_brute"]) <= 1e-10 for r in P12) else None
# Large d structural facts with presolve's own code vs the closed forms of this check (P1 PINN and P3 constancy, P4 zero)
S = {}
for d in (2, 5, 20, 50, 100):
    m = d // 2
    c = ps.affine_minimiser("laplace", d, "pinn", 1000.0)
    cexp = np.zeros(d + 1); cexp[0] = -m / 4; cexp[1:2*m+1] = 0.5
    S[f"P1_pinn_d{d}"] = float(np.abs(c - cexp).max())
    for w in (1e-4, 1.0, 100.0, 1e6):
        c3 = ps.affine_minimiser("ridge", d, "ritz", w); S[f"P3_ritz_d{d}_w{w:g}_lin"] = float(np.abs(c3[1:]).max())
        c4 = ps.affine_minimiser("cospair", d, "ritz", w); S[f"P4_ritz_d{d}_w{w:g}"] = float(np.abs(c4).max())
    S[f"P3_pinn_d{d}_lin"] = float(np.abs(ps.affine_minimiser("ridge", d, "pinn", 1.0)[1:]).max())
    # Robin bias as beta -> 0 / infinity: Deep Ritz -> PINN projection as beta -> inf
    for name in ("laplace", "ridge", "poisson"):
        S[f"{name}_d{d}_ritz_beta1e8_minus_pinn"] = float(np.abs(ps.affine_minimiser(name, d, "ritz", 1e8) - ps.affine_minimiser(name, d, "pinn", 1.)).max())
out["structural_large_d"] = S
# P2 Deep Ritz: as beta->0 c0 should blow up iff int f != 0 (P2 int f = 0 => stays finite)
out["P2_ritz_small_beta"] = {f"{w:g}": float(np.abs(ps.affine_minimiser("poisson", 20, "ritz", w)).max()) for w in (1e-6, 1e-3, 1, 1e3)}
out["P3_ritz_small_beta_c0"] = {f"{w:g}": float(ps.affine_minimiser("ridge", 20, "ritz", w)[0]) for w in (1e-6, 1e-3, 1, 100, 1e3)}

# ---------- Prop 3: floors, independently (1-D quad with scipy, separately written ANOVA for P3/P2 lift floor)
def leg(k, t): return [np.ones_like(t), t, 1.5*t*t-0.5, 2.5*t**3-1.5*t][k]
F = {}
for d in (1, 2, 5, 10, 20, 100):
    m = d // 2
    if m >= 1:
        F[f"P1_d{d}"] = math.sqrt((m/144) / (m/9 + m*(m-1)/16))
    # P2: per coordinate cos(pi x), mean 0, E cos^2 = 1/2
    c1 = integrate.quad(lambda x: math.cos(math.pi*x)*(x-0.5), 0, 1)[0]
    F[f"P2_aff_d{d}"] = math.sqrt(1 - 12 * c1**2 / 0.5)
    lk = [integrate.quad(lambda t, k=k: -math.sin(math.pi*t/2)*leg(k, t)/2, -1, 1)[0] for k in (1, 2, 3)]  # E[P_k(t) cos(pi x)], x=(t+1)/2
    F[f"P2_lift_d{d}"] = math.sqrt(1 - sum((2*k+1)*lk[k-1]**2 for k in (1, 2, 3)) / 0.5)
    # P3
    a = 2 / math.sqrt(d); phi = lambda z: math.sin(z)/z
    Eu = phi(a)**d; Eu2 = 0.5*(1 + phi(2*a)**d)
    F[f"P3_const_d{d}"] = math.sqrt(1 - Eu**2/Eu2)
    # lift floor: E[u|t_i] = cos(a t_i) phi(a)^(d-1) (sine part vanishes by symmetry)
    pk = [integrate.quad(lambda t, k=k: math.cos(a*t)*leg(k, t)/2, -1, 1)[0] * phi(a)**(d-1) for k in (1, 2, 3)]
    F[f"P3_lift_d{d}"] = math.sqrt(1 - (Eu**2 + d*sum((2*k+1)*pk[k-1]**2 for k in (1, 2, 3)))/Eu2)
    # additive (S) floor of P3, for reference: Var of E[u|t_i]
    v1 = integrate.quad(lambda t: (math.cos(a*t)*phi(a)**(d-1))**2/2, -1, 1)[0] - Eu**2
    F[f"P3_additive_d{d}"] = math.sqrt(max(0.0, 1 - (Eu**2 + d*v1)/Eu2))
out["floors"] = F
# MC cross-check of floors at d = 20 by least squares on 300k fresh points (sampler of this check)
rng = np.random.default_rng(11); n = 300_000; d = 20
X = rng.random((n, d)); T = 2*X - 1
A_aff = np.c_[np.ones(n), X]; A_lift = np.c_[np.ones(n), T, 1.5*T*T-0.5, 2.5*T**3-1.5*T]
for name, u in (("P1", (X[:, 0:20:2]*X[:, 1:20:2]).sum(1)), ("P2", np.cos(np.pi*X).sum(1)/math.sqrt(d)),
                ("P3", np.cos(2*T.sum(1)/math.sqrt(d))), ("P4", (np.cos(np.pi*X[:, 0:20:2])*np.cos(np.pi*X[:, 1:20:2])).sum(1))):
    for tag, A in (("aff", A_aff), ("lift", A_lift)):
        r = u - A @ np.linalg.lstsq(A, u, rcond=None)[0]
        F[f"MC_{name}_{tag}_d20"] = float(np.linalg.norm(r) / np.linalg.norm(u))
json.dump(out, open("check_proofs.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "P1P2_brute_d123"}, indent=1, default=float))
