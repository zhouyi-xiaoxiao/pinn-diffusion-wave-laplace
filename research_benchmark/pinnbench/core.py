"""Networks, loss, metrics and the training drivers.

Protocol for one model = (problem, sampler, seed)
-------------------------------------------------
  * tanh MLP with the layer sizes of the article, PyTorch default nn.Linear
    initialisation, raw (un-normalised) inputs, float32.
  * loss = mean(r^2) over the interior collocation set
           + sum over constraint groups (each edge / face / initial condition) of the group MSE
    (unit weights).
  * arm "adam":        Adam(lr=1e-3, constant), n_adam steps.
  * arm "adam_lbfgs":  the SAME first `branch_at` Adam steps (shared trajectory), then L-BFGS
                       (memory 50) for n_adam - branch_at iterations on a fixed collocation set,
                       i.e. a matched iteration count.

Two drivers implement this protocol:
  run_single   one model at a time; L-BFGS = torch.optim.LBFGS(strong_wolfe). Reference
               implementation, also used for per-run wall-clock measurements.
  run_batched  S independent models stacked along a leading axis and trained simultaneously
               (one Adam over the stacked parameters is *exactly* S independent Adams because
               the summed loss has block-separable gradients and Adam is element-wise; L-BFGS
               is our own per-model-vectorised implementation with Armijo backtracking and
               cautious curvature updates). Used for the multi-seed benchmark because it is
               ~S times cheaper on a GPU; validated against run_single in
               scripts/validate_batched.py.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from .problems import make_problem
from .samplers import make_sampler

ADAM_LR = 1e-3
LBFGS_MEMORY = 50


# ----------------------------------------------------------------------------------------
# networks
# ----------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BatchedMLP(nn.Module):
    """S independent MLPs evaluated together: x (S, N, d) -> (S, N, 1).

    Built from a list of ordinary MLPs so that model s has exactly the initial weights of
    `MLP` seeded with its own seed."""

    def __init__(self, mlps):
        super().__init__()
        self.S = len(mlps)
        linears = [[m for m in mlp.net if isinstance(m, nn.Linear)] for mlp in mlps]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for layer in zip(*linears):
            self.W.append(nn.Parameter(torch.stack([l.weight.detach().t() for l in layer])))        # (S, in, out)
            self.b.append(nn.Parameter(torch.stack([l.bias.detach().unsqueeze(0) for l in layer])))  # (S, 1, out)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.S, -1, -1)
        n = len(self.W)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b, x, W)
            if i < n - 1:
                x = torch.tanh(x)
        return x


# ----------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------
def predict(net, X, device="cpu", chunk=16384):
    """Network prediction (float64 numpy). Returns (M,) for MLP and (S, M) for BatchedMLP."""
    out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], chunk):
            xb = torch.as_tensor(X[s:s + chunk].astype(np.float32), device=device)
            out.append(net(xb).squeeze(-1).cpu().double().numpy())
    return np.concatenate(out, axis=-1)


def error_metrics(u_pred, u_true):
    e = u_pred - u_true
    return {"rel_l2": float(np.linalg.norm(e) / np.linalg.norm(u_true)),
            "linf": float(np.abs(e).max()),
            "rel_linf": float(np.abs(e).max() / np.abs(u_true).max())}


def total_loss(problem, net, X_r):
    """Returns (total, pde, dict of constraint terms); scalars for MLP, (S,) for BatchedMLP."""
    lres, comps = problem.losses(net, X_r)
    return lres + sum(comps.values()), lres, comps


# ----------------------------------------------------------------------------------------
# single-model reference driver
# ----------------------------------------------------------------------------------------
def run_single(problem, sampler_name, seed, n_r, n_adam, branch_at, log_every=100,
               lbfgs_chunk=50, test_X=None, test_u=None, eval_X=None, eval_u=None,
               sampler_kw=None, verbose=False):
    """Both arms for one configuration with torch.optim.LBFGS; JSON-serialisable result."""
    dev = problem.device
    torch.manual_seed(seed)
    net = MLP(problem.d, problem.hidden).to(dev)
    sampler = make_sampler(sampler_name, problem, n_r, seed, **(sampler_kw or {}))
    X_r = sampler.initial()
    opt = torch.optim.Adam(net.parameters(), lr=ADAM_LR)

    def log_point(net_, it, t_train, tot, lres, comps, curve):
        u = predict(net_, test_X, dev)
        curve.append({"it": it, "time": t_train, "loss": float(tot), "loss_pde": float(lres),
                      "loss_bc": float(sum(float(v.detach()) for v in comps.values())),
                      "test_rel_l2": float(np.linalg.norm(u - test_u) / np.linalg.norm(test_u))})

    curve_adam, t_train, snapshot = [], 0.0, None
    for it in range(1, n_adam + 1):
        t0 = time.perf_counter()
        if sampler.resample_each_step:
            X_r = sampler.draw()
        elif sampler.adaptive and it % sampler.period == 0 and it < n_adam:
            X_r = sampler.update(net)
        opt.zero_grad(set_to_none=True)
        tot, lres, comps = total_loss(problem, net, X_r)
        tot.backward()
        opt.step()
        t_train += time.perf_counter() - t0
        if it % log_every == 0 or it == 1:
            log_point(net, it, t_train, tot.item(), lres.item(), comps, curve_adam)
            if verbose and it % (10 * log_every) == 0:
                print(f"  adam it={it} loss={tot.item():.3e} rel_l2={curve_adam[-1]['test_rel_l2']:.3e} t={t_train:.1f}s", flush=True)
        if it == branch_at:
            snapshot = (copy.deepcopy(net.state_dict()), X_r.detach().clone(), t_train, list(curve_adam))
    res_adam = {"curve": curve_adam, "time": t_train, **error_metrics(predict(net, eval_X, dev), eval_u),
                "final_loss": curve_adam[-1]["loss"]}

    # ---------------- L-BFGS branch (torch.optim.LBFGS, strong Wolfe) -----------------
    net_b = MLP(problem.d, problem.hidden).to(dev)
    net_b.load_state_dict(snapshot[0])
    t_b, curve_b = snapshot[2], list(snapshot[3])
    t0 = time.perf_counter()
    X_fix = sampler.for_lbfgs(net_b, snapshot[1]).detach().clone().requires_grad_(True)
    t_b += time.perf_counter() - t0
    lb = torch.optim.LBFGS(net_b.parameters(), lr=1.0, max_iter=lbfgs_chunk, max_eval=int(lbfgs_chunk * 1.25),
                           history_size=LBFGS_MEMORY, tolerance_grad=1e-9, tolerance_change=1e-12,
                           line_search_fn="strong_wolfe")
    n_lbfgs = n_adam - branch_at
    done, nfev, stop_reason = 0, 0, "budget"

    def closure():
        nonlocal nfev
        lb.zero_grad(set_to_none=True)
        tot_, _, _ = total_loss(problem, net_b, X_fix)
        tot_.backward()
        nfev += 1
        return tot_

    while done < n_lbfgs:
        prev = copy.deepcopy(net_b.state_dict())
        lb.param_groups[0]["max_iter"] = min(lbfgs_chunk, n_lbfgs - done)
        lb.param_groups[0]["max_eval"] = int(1.25 * min(lbfgs_chunk, n_lbfgs - done))
        t0 = time.perf_counter()
        n_before = lb.state[lb._params[0]].get("n_iter", 0)
        fev_before = nfev
        lb.step(closure)
        t_b += time.perf_counter() - t0
        n_after = lb.state[lb._params[0]].get("n_iter", 0)
        tot, lres, comps = total_loss(problem, net_b, X_fix)
        if not np.isfinite(tot.item()):
            net_b.load_state_dict(prev)
            stop_reason = "nonfinite_reverted"
            break
        done += max(n_after - n_before, 0)
        log_point(net_b, branch_at + done, t_b, tot.item(), lres.item(), comps, curve_b)
        # L-BFGS returns early either because max_eval was hit (normal) or because a
        # tolerance was met (converged). Only the latter terminates the arm.
        if (n_after - n_before < lb.param_groups[0]["max_iter"]
                and nfev - fev_before < lb.param_groups[0]["max_eval"]):
            stop_reason = "converged_tolerance"
            break
    res_b = {"curve": curve_b, "time": t_b, **error_metrics(predict(net_b, eval_X, dev), eval_u),
             "final_loss": curve_b[-1]["loss"], "lbfgs_iters": done, "lbfgs_fevals": nfev, "stop": stop_reason}
    return {"adam": res_adam, "adam_lbfgs": res_b}


# ----------------------------------------------------------------------------------------
# batched driver
# ----------------------------------------------------------------------------------------
def _flat(tensors, S):
    return torch.cat([t.reshape(S, -1) for t in tensors], 1)


def _set_flat(params, x):
    o = 0
    with torch.no_grad():
        for p in params:
            n = p[0].numel()
            p.copy_(x[:, o:o + n].reshape(p.shape))
            o += n


def batched_lbfgs(params, S, loss_fn, n_iter, memory=LBFGS_MEMORY, c1=1e-4, max_ls=20,
                  callback=None, callback_every=50):
    """L-BFGS for S independent problems at once.

    params   : list of tensors with leading dimension S (model s owns slice [s]).
    loss_fn  : () -> (S,) tensor of per-model losses at the current parameters.
    Two-loop recursion with per-model histories; backtracking Armijo line search
    (step 1, halved on failure, first step min(1, 1/||g||_1)); a (s, y) pair is stored for model
    s only if y.s > 1e-10 ||y|| ||s|| (cautious update); a model whose line search fails keeps
    its parameters and has its history reset.
    Returns the total number of batched loss/gradient evaluations.
    """
    def fg(x):
        _set_flat(params, x)
        for p in params:
            p.grad = None
        f = loss_fn()
        f.sum().backward()
        g = _flat([p.grad for p in params], S)
        f = f.detach()
        bad = ~torch.isfinite(f) | ~torch.isfinite(g).all(1)
        return f, g, bad

    x = _flat([p.detach() for p in params], S).clone()
    f, g, _ = fg(x)
    nfev = 1
    hist = []                                   # list of (s, y, rho), each (S, P) / (S,)
    gamma = torch.ones(S, device=x.device)
    first = torch.ones(S, dtype=torch.bool, device=x.device)
    for it in range(1, n_iter + 1):
        # ---- two-loop recursion (per model) ----
        q = g.clone()
        alphas = []
        for s_, y_, rho in reversed(hist):
            a = rho * (s_ * q).sum(1)
            q = q - a[:, None] * y_
            alphas.append(a)
        r = gamma[:, None] * q
        for (s_, y_, rho), a in zip(hist, reversed(alphas)):
            b = rho * (y_ * r).sum(1)
            r = r + s_ * (a - b)[:, None]
        d = -r
        gd = (g * d).sum(1)
        notdesc = gd >= 0                       # safeguard: fall back to steepest descent
        if notdesc.any():
            d = torch.where(notdesc[:, None], -g, d)
            gd = (g * d).sum(1)
        t = torch.where(first, torch.clamp(1.0 / g.abs().sum(1).clamp_min(1e-30), max=1.0), torch.ones_like(gd))
        # ---- backtracking Armijo line search (per model) ----
        ok = torch.zeros(S, dtype=torch.bool, device=x.device)
        for _ in range(max_ls):
            x_new = x + t[:, None] * d
            f_new, g_new, bad = fg(x_new)
            nfev += 1
            ok = (~bad) & (f_new <= f + c1 * t * gd)
            if ok.all():
                break
            t = torch.where(ok, t, 0.5 * t)
        if not ok.all():                        # failed models: stay put, reset their memory
            t = torch.where(ok, t, torch.zeros_like(t))
            x_new = x + t[:, None] * d
            f_new, g_new, bad = fg(x_new)
            nfev += 1
            fail = ~ok
            for s_, y_, rho in hist:
                s_[fail] = 0.0
                y_[fail] = 0.0
                rho[fail] = 0.0
            gamma = torch.where(fail, torch.ones_like(gamma), gamma)
            first = fail                        # restart those models with a scaled first step
        else:
            first = torch.zeros_like(first)
        s_vec, y_vec = x_new - x, g_new - g
        ys = (y_vec * s_vec).sum(1)
        good = ys > 1e-10 * y_vec.norm(dim=1) * s_vec.norm(dim=1)
        good = good & ok
        rho = torch.where(good, 1.0 / ys.clamp_min(1e-30), torch.zeros_like(ys))
        s_vec = torch.where(good[:, None], s_vec, torch.zeros_like(s_vec))
        y_vec = torch.where(good[:, None], y_vec, torch.zeros_like(y_vec))
        hist.append((s_vec, y_vec, rho))
        if len(hist) > memory:
            hist.pop(0)
        gamma = torch.where(good, ys / (y_vec * y_vec).sum(1).clamp_min(1e-30), gamma)
        x, f, g = x_new, f_new, g_new
        if callback is not None and (it % callback_every == 0 or it == n_iter):
            _set_flat(params, x)
            callback(it, f)
    _set_flat(params, x)
    return nfev


def run_batched(problem_name, specs, n_r, n_adam, branch_at, device="cpu", log_every=100,
                rad_kw=None, verbose=False, return_preds=False):
    """Train len(specs) independent models (spec = (sampler_name, seed)) simultaneously.

    Returns a list of per-model result dicts with the same layout as run_single (without
    per-model wall-clock, which is meaningless for a batched run) plus batch-level timing."""
    S = len(specs)
    P = make_problem(problem_name, device=device, batch=S)
    P_cpu = make_problem(problem_name, device="cpu")          # geometry + exact solution only
    eval_X = P_cpu.eval_points()
    eval_u = P_cpu.exact(eval_X)
    test_X = P_cpu.test_subset()
    test_u = P_cpu.exact(test_X)
    test_norm = np.linalg.norm(test_u)

    mlps, samplers = [], []
    for name, seed in specs:
        torch.manual_seed(seed)
        mlps.append(MLP(P.d, P.hidden))
        kw = (rad_kw or {}) if name == "rad" else {}
        samplers.append(make_sampler(name, P_cpu, n_r, seed, **kw))
    net = BatchedMLP(mlps).to(device)
    params = list(net.parameters())
    X_np = np.stack([s.initial().detach().numpy() for s in samplers]).astype(np.float32)   # (S, N, d)
    resample_rows = [i for i, s in enumerate(samplers) if s.resample_each_step]
    rad_rows = [i for i, s in enumerate(samplers) if s.adaptive]
    rad_period = samplers[rad_rows[0]].period if rad_rows else None

    def to_dev(a):
        return torch.as_tensor(a, device=device).requires_grad_(True)

    def rad_update(net_, X_np_):
        """RAD resampling for the adaptive rows (residuals evaluated with the stacked net)."""
        pools = [samplers[i].propose() for i in rad_rows]
        npool = pools[0].shape[0]
        full = np.zeros((S, npool, P.d), dtype=np.float32)
        for i, pl in zip(rad_rows, pools):
            full[i] = pl
        res = []
        for s0 in range(0, npool, 8192):
            Xc = to_dev(full[:, s0:s0 + 8192])
            res.append(P.residual(net_, Xc).detach().abs().cpu().double().numpy())
        res = np.concatenate(res, 1)
        for i, pl in zip(rad_rows, pools):
            X_np_[i] = samplers[i].select(pl, res[i]).astype(np.float32)
        return X_np_

    curves = [[] for _ in range(S)]

    def log_all(curve_list, it, tot, lres, comps):
        rel = np.linalg.norm(predict(net, test_X, device) - test_u, axis=1) / test_norm
        bc = sum(v.detach() for v in comps.values()).cpu().numpy()
        tot, lres = tot.detach().cpu().numpy(), lres.detach().cpu().numpy()
        for i in range(S):
            curve_list[i].append({"it": it, "loss": float(tot[i]), "loss_pde": float(lres[i]),
                                  "loss_bc": float(bc[i]), "test_rel_l2": float(rel[i])})

    opt = torch.optim.Adam(params, lr=ADAM_LR)
    X = to_dev(X_np)
    t_adam, snapshot = 0.0, None
    for it in range(1, n_adam + 1):
        t0 = time.perf_counter()
        changed = False
        if resample_rows:
            for i in resample_rows:
                X_np[i] = samplers[i].draw().detach().numpy()
            changed = True
        if rad_rows and it % rad_period == 0 and it < n_adam:
            X_np = rad_update(net, X_np)
            changed = True
        if changed:
            X = to_dev(X_np)
        opt.zero_grad(set_to_none=True)
        tot, lres, comps = total_loss(P, net, X)
        tot.sum().backward()
        opt.step()
        if it % log_every == 0 or it == 1:
            tot.sum().item()                      # synchronise the device before reading the clock
            t_adam += time.perf_counter() - t0
            log_all(curves, it, tot, lres, comps)
            if verbose and it % (5 * log_every) == 0:
                rel = np.array([c[-1]["test_rel_l2"] for c in curves])
                print(f"  [{problem_name}] adam it={it} median rel_l2={np.median(rel):.3e} t={t_adam:.1f}s", flush=True)
        else:
            t_adam += time.perf_counter() - t0
        if it == branch_at:
            snapshot = ([p.detach().clone() for p in params], X_np.copy(), [list(c) for c in curves], t_adam)

    pred = predict(net, eval_X, device)
    preds = {"adam": pred.astype(np.float32)}
    res_adam = [{"curve": curves[i], **error_metrics(pred[i], eval_u), "final_loss": curves[i][-1]["loss"]}
                for i in range(S)]

    # ---------------- L-BFGS branch from the shared snapshot -----------------
    with torch.no_grad():
        for p, q in zip(params, snapshot[0]):
            p.copy_(q)
    X_np_b = snapshot[1]
    curves_b = snapshot[2]
    t0 = time.perf_counter()
    for i in resample_rows:                        # fixed fresh i.i.d. draw
        X_np_b[i] = samplers[i].draw().detach().numpy()
    if rad_rows:                                   # one last RAD resampling at the branch point
        X_np_b = rad_update(net, X_np_b)
    X_fix = to_dev(X_np_b)
    n_lbfgs = n_adam - branch_at

    def loss_fn():
        return total_loss(P, net, X_fix)[0]

    def cb(it, f):
        tot, lres, comps = total_loss(P, net, X_fix)
        log_all(curves_b, branch_at + it, tot, lres, comps)
        if verbose and it % 250 == 0:
            rel = np.array([c[-1]["test_rel_l2"] for c in curves_b])
            print(f"  [{problem_name}] lbfgs it={it} median rel_l2={np.median(rel):.3e} t={time.perf_counter() - t0:.1f}s", flush=True)

    nfev = batched_lbfgs(params, S, loss_fn, n_lbfgs, callback=cb, callback_every=50)
    t_lbfgs = time.perf_counter() - t0               # includes logging callbacks (small)
    pred = predict(net, eval_X, device)
    preds["adam_lbfgs"] = pred.astype(np.float32)
    res_b = [{"curve": curves_b[i], **error_metrics(pred[i], eval_u), "final_loss": curves_b[i][-1]["loss"],
              "lbfgs_iters": n_lbfgs, "lbfgs_batched_fevals": nfev} for i in range(S)]
    out = []
    for i, (name, seed) in enumerate(specs):
        out.append({"problem": problem_name, "sampler": name, "seed": seed, "n_r": n_r, "n_adam": n_adam,
                    "branch_at": branch_at, "adam": res_adam[i], "adam_lbfgs": res_b[i]})
    timing = {"batch_size": S, "t_adam_total": t_adam, "t_adam_to_branch": snapshot[3], "t_lbfgs_total": t_lbfgs,
              "lbfgs_batched_fevals": nfev, "device": device}
    if return_preds:
        return out, timing, preds
    return out, timing
