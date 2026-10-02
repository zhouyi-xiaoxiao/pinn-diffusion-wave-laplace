"""Second round of the re-check (plan: notes/VERIFICATION.md, section 4.3). Extends the separately written reimpl.py with two new problems P6, P7; seeds 40-42.
Usage: python reimpl2.py <block>. Appends to check/runs_reimpl2.jsonl; finished runs are skipped."""
import sys, os, json, math, statistics
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reimpl as R

R.OUT = os.path.join(HERE, "runs_reimpl2.jsonl")
_u0, _f0, _pt0 = R.u_star, R.f_src, R.product_terms


def u_star(p, x):
    d = x.shape[1]; m = d // 2
    if p == "P6":
        return (torch.sin(math.pi * x[:, 0:2*m:2]) * x[:, 1:2*m:2]).sum(1, keepdim=True)
    if p == "P7":
        return (torch.exp(x[:, 0:2*m:2]) * torch.cos(x[:, 1:2*m:2])).sum(1, keepdim=True)
    return _u0(p, x)


def f_src(p, x):
    if p == "P6": return math.pi ** 2 * u_star(p, x)
    if p == "P7": return 0.0 * u_star(p, x)
    return _f0(p, x)


def product_terms(p, d):
    if p in ("P6", "P7"):
        a, b = ((lambda x: np.sin(math.pi * x) + 0j), (lambda x: x + 0j)) if p == "P6" else \
               ((lambda x: np.exp(x) + 0j), (lambda x: np.cos(x) + 0j))
        T = []
        for k in range(d // 2):
            hs = [(lambda x: 1.0 + 0j)] * d
            hs = list(hs); hs[2*k] = a; hs[2*k+1] = b
            T.append((1.0, hs))
        return T, (math.pi ** 2 if p == "P6" else 0.0)
    return _pt0(p, d)


R.u_star, R.f_src, R.product_terms = u_star, f_src, product_terms
SEEDS = (40, 41, 42)
run, runs = R.run, R.runs


def beta(p):
    S = runs(block="sweep", p=p)
    if len(S) < 4: return None
    return min(S, key=lambda r: r["val_rel_l2"])["w"]


def neq(p, method, d=20):
    pl = [r["cpu_train"] for r in runs(block="main", p=p, method=method, d=d, arm="plain", iters=4000)]
    lf = [r["cpu_train"] for r in runs(block="main", p=p, method=method, d=d, arm="lift3c", iters=4000)]
    return int(50 * round(4000 * statistics.median(pl) / statistics.median(lf) / 50))


if __name__ == "__main__":
    b = sys.argv[1]
    if b == "selftest2":
        out = {}
        for p in ("P6", "P7"):
            for d in (2, 5, 20):
                x = torch.rand(128, d, dtype=torch.float64, requires_grad=True)
                u = u_star(p, x)
                gr = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
                lap = sum(torch.autograd.grad(gr[:, i].sum(), x, retain_graph=True)[0][:, i] for i in range(d))
                out[f"{p}_d{d}_src"] = float((f_src(p, x).squeeze() + lap).abs().max())
            d = 20
            # PINN minimiser = L2(boundary) projection: closed vs Monte-Carlo least squares on two samples
            c = R.affine_min(p, d, "pinn", 1000.0); cms = []
            for sd in (1, 2):
                xb = R.boundary(400_000, d, torch.Generator().manual_seed(sd)).double()
                A = torch.cat([torch.ones(len(xb), 1, dtype=torch.float64), xb], 1)
                cms.append(torch.linalg.lstsq(A, u_star(p, xb)).solution.squeeze(1).numpy())
            out[f"{p}_pinn_closed_vs_mc"] = float(np.abs(c - cms[0]).max()); out[f"{p}_pinn_mc_vs_mc"] = float(np.abs(cms[0] - cms[1]).max())
            # Deep Ritz: MC estimates of r_f, r_g, M then the same normal equations, beta = 10
            for w in (1.0, 100.0):
                c = R.affine_min(p, d, "ritz", w); cs = []
                for sd in (3, 4):
                    g = torch.Generator().manual_seed(sd)
                    xi = R.interior(400_000, d, g).double(); xb = R.boundary(400_000, d, g).double()
                    Pi = torch.cat([torch.ones(len(xi), 1, dtype=torch.float64), xi], 1)
                    Pb = torch.cat([torch.ones(len(xb), 1, dtype=torch.float64), xb], 1)
                    rf = (Pi * f_src(p, xi)).mean(0).numpy(); rg = 2*d*(Pb * u_star(p, xb)).mean(0).numpy()
                    M = 2*d*(Pb.T @ Pb / len(xb)).numpy(); E = np.eye(d + 1); E[0, 0] = 0
                    cs.append(np.linalg.solve(E + 2*w*M, rf + 2*w*rg))
                out[f"{p}_ritz_b{w:g}_closed_vs_mc"] = float(np.abs(c - cs[0]).max()); out[f"{p}_ritz_b{w:g}_mc_vs_mc"] = float(np.abs(cs[0] - cs[1]).max())
            # floors at d = 20 by least squares on 300k points
            X = R.interior(300_000, d, torch.Generator().manual_seed(5)).double(); T = 2*X - 1
            u = u_star(p, X).squeeze(); one = torch.ones(len(X), 1, dtype=torch.float64)
            for tag, A in (("const", one), ("affine", torch.cat([one, X], 1)),
                           ("lift3", torch.cat([one, T, 0.5*(3*T**2-1), 0.5*(5*T**3-3*T)], 1))):
                r = u - A @ torch.linalg.lstsq(A, u.unsqueeze(1)).solution.squeeze(1)
                out[f"{p}_floor_{tag}"] = float(r.norm() / u.norm())
        print(json.dumps(out, indent=1))
        json.dump(out, open(os.path.join(HERE, "selftest2.json"), "w"), indent=1)
    elif b == "sweeps":
        for p in ("P6", "P7"):
            for w in (1.0, 10.0, 100.0, 1000.0): run("sweep", p, "ritz", 5, 100, "plain", w)
            print("beta", p, beta(p))
    elif b == "B2":
        w = beta("P6"); assert w
        for seed in SEEDS:
            for arm in ("plain", "presolve", "lift3c", "lift1"): run("main", "P6", "ritz", 20, seed, arm, w)
        n = neq("P6", "ritz"); print("N_eq P6 ritz", n)
        for seed in SEEDS: run("eqcpu", "P6", "ritz", 20, seed, "lift3c", w, n)
        for seed in SEEDS:
            for arm in ("plain", "lift3c"): run("main", "P6", "ritz", 10, seed, arm, w)
    elif b == "B3":
        w = beta("P7"); assert w
        for seed in SEEDS:
            for arm in ("plain", "presolve", "lift3c", "lift1"): run("main", "P7", "ritz", 20, seed, arm, w)
    elif b == "B1":
        for seed in SEEDS:
            for arm in ("plain", "presolve", "lift3c"): run("main", "P1", "pinn", 20, seed, arm, 1000.0)
    elif b == "B1eq":
        n = neq("P1", "pinn"); print("N_eq P1 pinn", n)
        for seed in SEEDS: run("eqcpu", "P1", "pinn", 20, seed, "lift3c", 1000.0, n)
    elif b == "B4":
        for seed in SEEDS:
            for arm in ("plain", "presolve", "lift3c", "lift1"): run("main", "P6", "pinn", 20, seed, arm, 1000.0)
