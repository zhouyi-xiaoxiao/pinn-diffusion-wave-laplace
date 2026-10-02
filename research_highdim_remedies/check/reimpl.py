"""Separately written re-implementation (first round of the re-check; plan: notes/VERIFICATION.md, section 4.3). Usage: python reimpl.py <block>.
Appends one JSON line per run to check/runs_reimpl.jsonl; finished runs are skipped."""
import sys, os, json, math, time, statistics
import numpy as np
from scipy import integrate
import torch
import torch.nn as nn
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_reimpl.jsonl")
F32 = torch.float32

# ------------------------------------------------------------ problems (u*, f = -Lap u*, product-form terms for moments)
def eps(d):
    return torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(d)], dtype=F32)   # e_i = +1 for odd i (1-based)

def u_star(p, x):
    d = x.shape[1]
    if p == "P1":
        m = d // 2
        return (x[:, 0:2*m:2] * x[:, 1:2*m:2]).sum(1, keepdim=True)
    if p == "P3":
        return torch.cos(2.0 * (2*x - 1).sum(1, keepdim=True) / math.sqrt(d))
    if p == "P5":
        s = ((2*x - 1) * eps(d)).sum(1, keepdim=True) / math.sqrt(d)
        return torch.cos(math.pi * s + 1.0)
    raise ValueError(p)

def f_src(p, x):
    return {"P1": 0.0, "P3": 16.0, "P5": 4 * math.pi ** 2}[p] * u_star(p, x)

def product_terms(p, d):
    """u* = Re sum coef * prod_j h_j(x_j)"""
    if p == "P1":
        T = []
        for k in range(d // 2):
            hs = [lambda x: 1.0 + 0j] * d
            hs = list(hs); hs[2*k] = lambda x: x + 0j; hs[2*k+1] = lambda x: x + 0j
            T.append((1.0, hs))
        return T, 0.0
    if p == "P3":
        a = 2.0 / math.sqrt(d)
        return [(1.0, [lambda x, a=a: np.exp(1j*a*(2*x - 1))] * d)], 16.0
    if p == "P5":
        hs = [(lambda x, e=(1.0 if i % 2 == 0 else -1.0): np.exp(1j*math.pi*e*(2*x - 1)/math.sqrt(d))) for i in range(d)]
        return [(np.exp(1j), hs)], 4 * math.pi ** 2
    raise ValueError(p)

def cquad(fun, k):
    re = integrate.quad(lambda x: (x**k * fun(x)).real, 0, 1, epsabs=1e-14, epsrel=1e-13)[0]
    im = integrate.quad(lambda x: (x**k * fun(x)).imag, 0, 1, epsabs=1e-14, epsrel=1e-13)[0]
    return re + 1j*im

def gram(d):
    M = np.zeros((d + 1, d + 1))
    for k in range(d):
        for s in (0.0, 1.0):
            m1 = np.full(d, 0.5); m1[k] = s
            m2 = np.full(d, 1/3); m2[k] = s*s
            F = np.zeros((d + 1, d + 1)); F[0, 0] = 1; F[0, 1:] = F[1:, 0] = m1
            F[1:, 1:] = np.outer(m1, m1); np.fill_diagonal(F[1:, 1:], m2); M += F
    return M

def affine_min(p, d, method, w):
    T, kap = product_terms(p, d)
    rf = np.zeros(d + 1, complex); rg = np.zeros(d + 1, complex)
    for coef, hs in T:
        I0 = [cquad(h, 0) for h in hs]; I1 = [cquad(h, 1) for h in hs]
        def pr(skip):
            v = 1 + 0j
            for j in range(d):
                if j not in skip: v *= I0[j]
            return v
        rf[0] += coef * kap * pr(set())
        for i in range(d): rf[1+i] += coef * kap * I1[i] * pr({i})
        for k in range(d):
            for s in (0.0, 1.0):
                hk = complex(hs[k](s)); rg[0] += coef * hk * pr({k})
                for i in range(d):
                    rg[1+i] += coef * hk * (s * pr({k}) if i == k else I1[i] * pr({k, i}))
    rf, rg = rf.real, rg.real
    M = gram(d)
    if method == "pinn":
        return np.linalg.solve(M, rg)
    E = np.eye(d + 1); E[0, 0] = 0
    return np.linalg.solve(E + 2*w*M, rf + 2*w*rg)

# ------------------------------------------------------------ sampling
def interior(n, d, g): return torch.rand(n, d, generator=g, dtype=F32)
def boundary(n, d, g):
    x = torch.rand(n, d, generator=g, dtype=F32)
    k = torch.randint(0, 2*d, (n,), generator=g)          # one of 2d faces
    x[torch.arange(n), k % d] = (k // d).to(F32)
    return x

# ------------------------------------------------------------ model
def legendre_feats(x, kind):
    t = 2*x - 1
    if kind == "lift3c": return torch.cat([t, 0.5*(3*t**2 - 1), 0.5*(5*t**3 - 3*t)], 1)
    if kind == "lift3u": return torch.cat([x, 0.5*(3*t**2 - 1), 0.5*(5*t**3 - 3*t)], 1)
    if kind == "rep3": return torch.cat([t, t, t], 1)
    if kind == "lift1": return t
    return x

class Net(nn.Module):
    def __init__(self, d, arm, shift=None):
        super().__init__()
        self.arm = arm
        din = 3*d if arm in ("lift3c", "lift3u", "rep3") else d
        self.body = nn.Sequential(nn.Linear(din, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                  nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.shift = None if shift is None else torch.tensor(shift, dtype=F32)
    def forward(self, x):
        u = self.body(legendre_feats(x, self.arm))
        if self.shift is not None:
            u = u + self.shift[0] + x @ self.shift[1:].unsqueeze(1)
        return u

def loss_fn(net, p, method, w, xi, xb):
    d = xi.shape[1]
    xi = xi.clone().requires_grad_(True)
    u = net(xi)
    gu = torch.autograd.grad(u.sum(), xi, create_graph=True)[0]
    if method == "ritz":
        inner = (0.5 * (gu**2).sum(1, keepdim=True) - f_src(p, xi.detach()) * u).mean()
        return inner + w * 2*d * ((net(xb) - u_star(p, xb))**2).mean()
    lap = torch.zeros_like(u)
    for i in range(d):
        lap = lap + torch.autograd.grad(gu[:, i].sum(), xi, create_graph=True)[0][:, i:i+1]
    return ((lap + f_src(p, xi.detach()))**2).mean() + w * ((net(xb) - u_star(p, xb))**2).mean()

def rel_err(net, p, x):
    with torch.no_grad():
        u = net(x).double(); ue = u_star(p, x).double()
    return float(((u - ue).norm() / ue.norm()))

def train(p, method, d, seed, arm, w, iters=4000):
    shift, cpu_shift = None, 0.0
    if arm == "presolve":
        c0 = time.process_time(); shift = affine_min(p, d, method, w); cpu_shift = time.process_time() - c0
    torch.manual_seed(seed)
    net = Net(d, arm, shift)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, [iters // 2, 3*iters // 4], 0.1)
    g = torch.Generator().manual_seed(5000 + seed)
    cpu = 0.0
    for it in range(iters):
        c0 = time.process_time()
        xi = interior(1024, d, g); xb = boundary(1024, d, g)
        opt.zero_grad(); L = loss_fn(net, p, method, w, xi, xb); L.backward(); opt.step(); sch.step()
        cpu += time.process_time() - c0
    test = interior(50_000, d, torch.Generator().manual_seed(777000 + d))
    val = interior(5_000, d, torch.Generator().manual_seed(888000 + d))
    return dict(rel_l2=rel_err(net, p, test), val_rel_l2=rel_err(net, p, val), cpu_train=cpu, cpu_shift=cpu_shift,
                cpu_total=cpu + cpu_shift, final_loss=float(L), shift=None if shift is None else [float(v) for v in shift])

def done_keys():
    if not os.path.exists(OUT): return set()
    return {(r["block"], r["p"], r["method"], r["d"], r["seed"], r["arm"], r["w"], r["iters"]) for r in map(json.loads, open(OUT))}

def run(block, p, method, d, seed, arm, w, iters=4000):
    k = (block, p, method, d, seed, arm, w, iters)
    if k in done_keys(): return
    t0 = time.time(); r = train(p, method, d, seed, arm, w, iters)
    r.update(block=block, p=p, method=method, d=d, seed=seed, arm=arm, w=w, iters=iters, wall=time.time() - t0,
             load1=os.getloadavg()[0], finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(OUT, "a") as fh: fh.write(json.dumps(r) + "\n")
    print(block, p, method, d, seed, arm, w, iters, f"rel={r['rel_l2']:.4e} cpu={r['cpu_total']:.1f} wall={r['wall']:.0f} load={r['load1']:.0f}", flush=True)

def runs(**kw):
    if not os.path.exists(OUT): return []
    return [r for r in map(json.loads, open(OUT)) if all(r[a] == b for a, b in kw.items())]

SEEDS = (30, 31, 32)
def beta_P5():
    R = runs(block="sweepP5")
    if len(R) < 4: return None
    return min(R, key=lambda r: r["val_rel_l2"])["w"]

def neq(p, method, d=20):
    pl = [r["cpu_train"] for r in runs(block="main", p=p, method=method, d=d, arm="plain", iters=4000)]
    lf = [r["cpu_train"] for r in runs(block="main", p=p, method=method, d=d, arm="lift3c", iters=4000)]
    return int(50 * round(4000 * statistics.median(pl) / statistics.median(lf) / 50))

if __name__ == "__main__":
    b = sys.argv[1]
    if b == "sweepP5":
        for w in (1.0, 10.0, 100.0, 1000.0): run("sweepP5", "P5", "ritz", 5, 100, "plain", w)
        print("beta_P5 =", beta_P5())
    elif b == "R1":
        for seed in SEEDS:
            for arm in ("plain", "presolve", "lift3c", "lift3u"): run("main", "P1", "ritz", 20, seed, arm, 100.0)
            for arm in ("plain", "presolve", "lift3c"): run("main", "P3", "ritz", 20, seed, arm, 100.0)
    elif b == "R2":
        w = beta_P5(); assert w
        for seed in SEEDS:
            for arm in ("plain", "presolve", "lift3c", "rep3", "lift1"): run("main", "P5", "ritz", 20, seed, arm, w)
    elif b == "R3":
        for p, w in (("P3", 100.0), ("P5", beta_P5())):
            n = neq(p, "ritz"); print("N_eq", p, n)
            for seed in SEEDS: run("eqcpu", p, "ritz", 20, seed, "lift3c", w, n)
    elif b == "R4":
        w = beta_P5()
        for d in (10, 5):
            for seed in SEEDS:
                for arm in ("plain", "lift3c"): run("main", "P5", "ritz", d, seed, arm, w)
    elif b == "R5":
        for p in sys.argv[2:] or ["P5"]:
            for seed in SEEDS:
                for arm in ("plain", "lift3c"): run("main", p, "pinn", 20, seed, arm, 1000.0)
    elif b == "R5eq":
        for p in sys.argv[2:] or ["P5"]:
            n = neq(p, "pinn"); print("N_eq", p, n)
            for seed in SEEDS: run("eqcpu", p, "pinn", 20, seed, "lift3c", 1000.0, n)
    elif b == "R6":   # added after the plan of the first round (notes/VERIFICATION.md, section 4.3): centred-input arm lift1 as a normalised baseline
        for seed in SEEDS:
            run("main", "P3", "ritz", 20, seed, "lift1", 100.0)
        for p in sys.argv[2:] or ["P5"]:
            for seed in SEEDS:
                run("main", p, "pinn", 20, seed, "lift1", 1000.0)
    elif b == "R7":   # exploratory sensitivity (notes/VERIFICATION.md, section 4.3): P5 Deep Ritz with beta = 100 instead of the selected 1000
        for seed in SEEDS:
            for arm in ("plain", "lift3c", "lift1"): run("sens", "P5", "ritz", 20, seed, arm, 100.0)
    elif b == "selftest":
        # the affine minimiser of this check vs the study's presolve on P1/P3 (agreement of two separately written codes)
        sys.path.insert(0, os.path.join(HERE, "..", "src")); import presolve as ps
        for p, name in (("P1", "laplace"), ("P3", "ridge")):
            for m, w in (("ritz", 100.0), ("pinn", 1000.0)):
                a = affine_min(p, 20, m, w); bb = ps.affine_minimiser(name, 20, m, w)
                print(p, m, float(np.abs(a - bb).max()))
        # P5 affine minimiser vs Monte-Carlo least squares on the boundary (PINN) and its floors
        d = 20; c = affine_min("P5", d, "pinn", 1.0)
        g = torch.Generator().manual_seed(1); xb = boundary(400_000, d, g).double()
        A = torch.cat([torch.ones(len(xb), 1, dtype=torch.float64), xb], 1)
        cm = torch.linalg.lstsq(A, u_star("P5", xb.float()).double()).solution.squeeze(1)
        print("P5 pinn closed vs MC", float(np.abs(c - cm.numpy()).max()), "max|c|", float(np.abs(c).max()))
        # Laplacian / source check for P5 in float64 by autograd
        x = torch.rand(64, d, dtype=torch.float64, requires_grad=True)
        s = ((2*x - 1) * eps(d).double()).sum(1) / math.sqrt(d); u = torch.cos(math.pi*s + 1.0)
        gr = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        lap = sum(torch.autograd.grad(gr[:, i].sum(), x, retain_graph=True)[0][:, i] for i in range(d))
        print("P5 |f + Lap u|", float((4*math.pi**2*u + lap).abs().max()))
        # floors of P5 at d = 20 by least squares on 300k points
        X = interior(300_000, d, torch.Generator().manual_seed(2)).double(); T = 2*X - 1
        u = u_star("P5", X.float()).double().squeeze()
        for tag, A in (("const", torch.ones(len(X), 1, dtype=torch.float64)), ("aff", torch.cat([torch.ones(len(X), 1, dtype=torch.float64), X], 1)),
                       ("lift3", torch.cat([torch.ones(len(X), 1, dtype=torch.float64), T, 0.5*(3*T**2-1), 0.5*(5*T**3-3*T)], 1))):
            r = u - A @ torch.linalg.lstsq(A, u.unsqueeze(1)).solution.squeeze(1)
            print("P5 floor", tag, float(r.norm() / u.norm()))
