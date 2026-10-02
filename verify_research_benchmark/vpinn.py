"""Separately written PINN implementation of the verification (no import from the study's pinnbench).

Differences from the study's code on purpose:
  * derivatives by forward propagation of (value, gradient, diagonal Hessian) through the tanh MLP
    (analytic jets) instead of nested torch.autograd.grad;
  * single-model training on CPU, float32, torch.optim.LBFGS(strong_wolfe) for the L-BFGS arm
    (the study's multi-seed numbers use a custom vectorised Armijo L-BFGS on MPS).
Same protocol as the study: loss = mean(r^2) + unit-weight MSE per constraint group; Adam lr 1e-3;
arm A = n_adam Adam steps; arm B = first `branch` Adam steps then L-BFGS for n_adam-branch iterations.
"""
import math, time, copy, json
import numpy as np, torch, torch.nn as nn
from scipy.stats import qmc
torch.set_num_threads(4)
PI = math.pi


def make_mlp(d, hidden, seed):
    torch.manual_seed(seed)
    layers, prev = [], d
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]; prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def jets(net, X, need_hess=True):
    """Return u (N,), du (N,d), d2u (N,d) [diagonal second derivatives] by forward jet propagation."""
    lin = [m for m in net if isinstance(m, nn.Linear)]
    N, d = X.shape
    z = X
    dz = torch.eye(d, dtype=X.dtype).unsqueeze(0).expand(N, d, d)       # (N, d(dir), features)
    d2z = torch.zeros(N, d, d, dtype=X.dtype)
    for k, L in enumerate(lin):
        a = z @ L.weight.t() + L.bias                 # (N, H)
        da = dz @ L.weight.t()                         # (N, d, H)
        d2a = d2z @ L.weight.t() if need_hess else None
        if k < len(lin) - 1:
            h = torch.tanh(a); s = 1 - h * h           # (N,H)
            z = h
            dz = s.unsqueeze(1) * da
            if need_hess:
                d2z = s.unsqueeze(1) * d2a - 2 * (h * s).unsqueeze(1) * da * da
        else:
            z, dz, d2z = a, da, d2a
    return z[:, 0], dz[:, :, 0], (d2z[:, :, 0] if need_hess else None)


def grid(*axes):
    m = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in m], 1)


def lap3d_exact(X, nmax=4000):
    X = np.asarray(X, float); out = np.zeros(len(X))
    n = np.arange(2, nmax + 1, 2.0); b = 4 * n / (PI * (n * n - 1)); k = np.sqrt(1 + n * n)
    for s in range(0, len(X), 8000):
        x = X[s:s + 8000, 0:1]; z = X[s:s + 8000, 2:3]
        # sinh(kx)/sinh(k pi) in overflow-safe form
        r = np.exp(k * (x - PI)) * (1 - np.exp(-2 * k * x)) / (1 - np.exp(-2 * k * PI))
        out[s:s + 8000] = np.sin(X[s:s + 8000, 1]) * (b * np.sin(n * z) * r).sum(1)
    return out


class Prob:
    pass


def problem(name):
    P = Prob(); P.name = name
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
    if name == "heat1d":
        P.d, P.lo, P.hi, P.hidden = 2, np.array([0., 0.]), np.array([1., 1.]), [128, 128]
        s = np.linspace(0, 1, 100)
        P.sets = [(T(np.stack([0 * s, s], 1)), "u", T(np.sin(PI * s))), (T(np.stack([s, 0 * s], 1)), "u", T(0 * s)), (T(np.stack([s, 0 * s + 1], 1)), "u", T(0 * s))]
        P.res = lambda u, g, h: g[:, 0] - h[:, 1]
        P.exact = lambda X: np.exp(-PI ** 2 * X[:, 0]) * np.sin(PI * X[:, 1])
        a = np.linspace(0, 1, 201); P.evalX = grid(a, a)
    elif name == "wave1d":
        P.d, P.lo, P.hi, P.hidden = 2, np.array([0., 0.]), np.array([1., 1.]), [64, 64, 64]
        s = np.linspace(0, 1, 100); ic = np.stack([0 * s, s], 1)
        P.sets = [(T(ic), "u", T(np.sin(PI * s) + 0.5 * np.sin(4 * PI * s))), (T(ic), 0, T(0 * s)), (T(np.stack([s, 0 * s], 1)), "u", T(0 * s)), (T(np.stack([s, 0 * s + 1], 1)), "u", T(0 * s))]
        P.res = lambda u, g, h: h[:, 0] - 4.0 * h[:, 1]
        P.exact = lambda X: np.sin(PI * X[:, 1]) * np.cos(2 * PI * X[:, 0]) + 0.5 * np.sin(4 * PI * X[:, 1]) * np.cos(8 * PI * X[:, 0])
        a = np.linspace(0, 1, 201); P.evalX = grid(a, a)
    elif name == "wave2d":
        P.d, P.lo, P.hi, P.hidden = 3, np.array([0., -1., -1.]), np.array([1., 1., 1.]), [50, 50]
        s = np.linspace(-1, 1, 21); ic = grid(np.array([0.0]), s, s)
        P.sets = [(T(ic), "u", T(np.sin(PI * ic[:, 1]) * np.sin(PI * ic[:, 2]))), (T(ic), 0, T(np.zeros(len(ic))))]
        F = grid(np.linspace(0, 1, 11), s)
        for ax in (1, 2):
            for val in (-1.0, 1.0):
                pts = np.zeros((len(F), 3)); pts[:, 0] = F[:, 0]; pts[:, ax] = val; pts[:, 3 - ax] = F[:, 1]
                P.sets.append((T(pts), "u", T(np.zeros(len(F)))))
        P.res = lambda u, g, h: h[:, 0] - h[:, 1] - h[:, 2]
        P.exact = lambda X: np.sin(PI * X[:, 1]) * np.sin(PI * X[:, 2]) * np.cos(math.sqrt(2) * PI * X[:, 0])
        P.evalX = grid(np.linspace(0, 1, 21), np.linspace(-1, 1, 81), np.linspace(-1, 1, 81))
    elif name == "laplace2d":
        P.d, P.lo, P.hi, P.hidden = 2, np.array([0., 0.]), np.array([PI, PI]), [128, 128, 128]
        s = np.linspace(0, PI, 100); z = 0 * s; p = z + PI
        P.sets = [(T(np.stack([z, s], 1)), "u", T(z)), (T(np.stack([p, s], 1)), "u", T(0.5 * (1 + np.cos(2 * s)))),
                  (T(np.stack([s, z], 1)), 1, T(z)), (T(np.stack([s, p], 1)), 1, T(z))]
        P.res = lambda u, g, h: h[:, 0] + h[:, 1]
        P.exact = lambda X: X[:, 0] / (2 * PI) + np.cos(2 * X[:, 1]) * np.sinh(2 * X[:, 0]) / (2 * np.sinh(2 * PI))
        a = np.linspace(0, PI, 201); P.evalX = grid(a, a)
    elif name == "laplace3d":
        P.d, P.lo, P.hi, P.hidden = 3, np.zeros(3), np.full(3, PI), [100, 100]
        m = 12; s = (np.arange(m) + 0.5) * PI / m; F = grid(s, s); P.sets = []
        for ax in range(3):
            for val in (0.0, PI):
                pts = np.zeros((len(F), 3)); oth = [a for a in range(3) if a != ax]
                pts[:, ax] = val; pts[:, oth[0]] = F[:, 0]; pts[:, oth[1]] = F[:, 1]
                tgt = np.sin(pts[:, 1]) * np.cos(pts[:, 2]) if (ax == 0 and val == PI) else np.zeros(len(F))
                P.sets.append((T(pts), "u", T(tgt)))
        P.res = lambda u, g, h: h[:, 0] + h[:, 1] + h[:, 2]
        P.exact = lap3d_exact
        a = (np.arange(40) + 0.5) * PI / 40; P.evalX = grid(a, a, a)
    P.evalU = P.exact(P.evalX)
    return P


def points(P, kind, n, seed):
    d = P.d
    if kind == "grid":            # the study's cell-centred grid
        m = round(n ** (1 / d)); assert m ** d == n
        U = grid(*[(np.arange(m) + 0.5) / m] * d)
    elif kind == "nodegrid":      # the re-check's control: closed (node-centred) grid incl. the boundary
        m = round(n ** (1 / d)); assert m ** d == n
        U = grid(*[np.linspace(0, 1, m)] * d)
    elif kind == "random":        # same construction as the study so that seeds are comparable
        U = np.random.default_rng([seed, 7919]).random((n, d))
    elif kind == "sobol":
        U = qmc.Sobol(d=d, scramble=True, seed=np.random.default_rng([seed, 7919])).random_base2(int(np.ceil(np.log2(n))))[:n]
    return torch.as_tensor((P.lo + (P.hi - P.lo) * U).astype(np.float32))


def loss_fn(P, net, Xr):
    u, g, h = jets(net, Xr)
    tot = (P.res(u, g, h) ** 2).mean()
    for Xc, kind, tgt in P.sets:
        uc, gc, _ = jets(net, Xc, need_hess=False)
        val = uc if kind == "u" else gc[:, kind]
        tot = tot + ((val - tgt) ** 2).mean()
    return tot


def evaluate(P, net):
    with torch.no_grad():
        out = []
        X = torch.as_tensor(P.evalX.astype(np.float32))
        for s in range(0, len(X), 20000):
            out.append(net(X[s:s + 20000])[:, 0].double().numpy())
    e = np.concatenate(out) - P.evalU
    return float(np.linalg.norm(e) / np.linalg.norm(P.evalU)), float(np.abs(e).max())


def train(P, kind, seed, n_r, n_adam=3000, branch=1500, resample=False, verbose=False):
    net = make_mlp(P.d, P.hidden, seed)
    Xr = points(P, kind, n_r, seed)
    rng = np.random.default_rng([seed, 4242])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0 = time.perf_counter(); snap = None
    for it in range(1, n_adam + 1):
        if resample:
            Xr = torch.as_tensor((P.lo + (P.hi - P.lo) * rng.random((n_r, P.d))).astype(np.float32))
        opt.zero_grad(set_to_none=True)
        L = loss_fn(P, net, Xr); L.backward(); opt.step()
        if it == branch:
            snap = copy.deepcopy(net.state_dict()); t_branch = time.perf_counter() - t0
            mid = evaluate(P, net)
    t_adam = time.perf_counter() - t0
    ra = evaluate(P, net)
    out = {"adam": {"rel_l2": ra[0], "linf": ra[1], "loss": float(L), "time": t_adam}, "adam_at_branch": {"rel_l2": mid[0], "linf": mid[1]}}
    net.load_state_dict(snap)
    n_l = n_adam - branch
    lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=n_l, max_eval=int(1.5 * n_l), history_size=50,
                           tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    nfev = [0]
    def closure():
        lb.zero_grad(set_to_none=True); L_ = loss_fn(P, net, Xr); L_.backward(); nfev[0] += 1; return L_
    t0 = time.perf_counter(); lb.step(closure); t_l = time.perf_counter() - t0
    rb = evaluate(P, net)
    out["adam_lbfgs"] = {"rel_l2": rb[0], "linf": rb[1], "loss": float(loss_fn(P, net, Xr)), "time": t_branch + t_l,
                         "lbfgs_iters": lb.state[lb._params[0]]["n_iter"], "nfev": nfev[0]}
    return out


if __name__ == "__main__":
    # unit test of the jet derivatives against autograd, and a speed probe
    torch.manual_seed(0)
    net = make_mlp(3, [50, 50], 0)
    X = torch.rand(7, 3, requires_grad=True)
    u, g, h = jets(net, X)
    ga = torch.autograd.grad(net(X).sum(), X, create_graph=True)[0]
    ha = torch.stack([torch.autograd.grad(ga[:, i].sum(), X, retain_graph=True)[0][:, i] for i in range(3)], 1)
    print("jet vs autograd: max|du diff| %.2e, max|d2u diff| %.2e" % ((g - ga).abs().max(), (h - ha).abs().max()))
    for name, n in [("heat1d", 256), ("laplace2d", 256), ("wave2d", 512), ("laplace3d", 512), ("wave1d", 1024)]:
        P = problem(name); net = make_mlp(P.d, P.hidden, 0); Xr = points(P, "random", n, 0)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        t0 = time.perf_counter()
        for _ in range(30):
            opt.zero_grad(); L = loss_fn(P, net, Xr); L.backward(); opt.step()
        print(name, "ms/step", 1000 * (time.perf_counter() - t0) / 30, "loss", float(L), flush=True)
