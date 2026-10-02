"""Core code for the high-dimensional PINN-vs-Deep-Ritz study on the unit cube (0,1)^d.

Problems (both with closed-form exact solutions):
  P1 "laplace"  : -Δu = 0,  u = g on ∂Ω,  u*(x) = Σ_{k=1}^{⌊d/2⌋} x_{2k-1} x_{2k}
                  (d = 10 is the example of E & Yu 2018, Deep Ritz §3.2)
  P2 "poisson"  : -Δu = f,  u = g on ∂Ω,  u*(x) = d^{-1/2} Σ_{i=1}^{d} cos(π x_i),  f = π² u*
                  (zero-mean, ||u*||_{L2(Ω)}² = 1/2 for every d, so relative errors are not
                   flattered by concentration of measure)

Methods (same network, same optimiser, same fresh-Monte-Carlo sampling, same boundary penalty form):
  PINN : L = mean_Ω (Δu + f)²              + λ · mean_∂Ω (u - g)²
  Ritz : L = mean_Ω (½|∇u|² - f u) · |Ω|  + β · mean_∂Ω (u - g)² · |∂Ω|,   |Ω| = 1, |∂Ω| = 2d
"""
import math, time, json, os
import torch
import torch.nn as nn

torch.set_num_threads(1)  # the plan asked for <=4; 1 thread measured faster under the shared-machine load (see notes/VERIFICATION.md, section 3.4)
DTYPE = torch.float32


# ----------------------------------------------------------------------------- problems
def exact(problem, x):
    d = x.shape[1]
    if problem == "laplace":
        m = d // 2
        return (x[:, 0:2 * m:2] * x[:, 1:2 * m:2]).sum(1, keepdim=True)
    if problem == "poisson":
        return torch.cos(math.pi * x).sum(1, keepdim=True) / math.sqrt(d)
    raise ValueError(problem)


def exact_grad(problem, x):
    d = x.shape[1]
    g = torch.zeros_like(x)
    if problem == "laplace":
        m = d // 2
        g[:, 0:2 * m:2] = x[:, 1:2 * m:2]
        g[:, 1:2 * m:2] = x[:, 0:2 * m:2]
        return g
    if problem == "poisson":
        return -math.pi * torch.sin(math.pi * x) / math.sqrt(d)
    raise ValueError(problem)


def source(problem, x):
    """f = -Δu*"""
    if problem == "laplace":
        return torch.zeros(x.shape[0], 1, dtype=x.dtype)
    if problem == "poisson":
        return math.pi ** 2 * exact(problem, x)
    raise ValueError(problem)


# ----------------------------------------------------------------------------- sampling
def sample_interior(n, d, gen):
    return torch.rand(n, d, generator=gen, dtype=DTYPE)


def sample_boundary(n, d, gen):
    """Uniform on ∂[0,1]^d: pick one of the 2d faces uniformly (all faces have unit area),
    then set that coordinate to 0 or 1."""
    x = torch.rand(n, d, generator=gen, dtype=DTYPE)
    face = torch.randint(0, d, (n,), generator=gen)
    side = torch.randint(0, 2, (n,), generator=gen).to(DTYPE)
    x[torch.arange(n), face] = side
    return x


# ----------------------------------------------------------------------------- network
class MLP(nn.Module):
    def __init__(self, d, width=64, depth=3):
        super().__init__()
        layers, k = [], d
        for _ in range(depth):
            layers += [nn.Linear(k, width), nn.Tanh()]
            k = width
        layers += [nn.Linear(k, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def laplacian(u, x):
    """Exact Laplacian by d reverse-mode passes (cost grows linearly with d)."""
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        lap = lap + torch.autograd.grad(g[:, i].sum(), x, create_graph=True)[0][:, i:i + 1]
    return lap, g


def forward_laplacian(net, x):
    """Exact (u, ∇u, Δu) of a Linear/Tanh MLP in ONE forward sweep ("forward Laplacian" / second-order
    Taylor-mode propagation): carry h, J = ∂h/∂x (n×d×width) and Δh through every layer.
      linear : h←hWᵀ+b, J←JWᵀ, L←LWᵀ
      tanh   : t=tanh(h); L←(1-t²)·L − 2t(1-t²)·Σ_i J_i² ; J←(1-t²)·J ; h←t
    Same value as `laplacian` up to round-off (checked in scripts/check_core.py); cost still ∝ d, but as
    batched matmuls instead of d separate reverse passes, and no higher-order graph is needed."""
    n, d = x.shape
    h = x
    J = torch.eye(d, dtype=x.dtype).expand(n, d, d)
    L = torch.zeros(n, d, dtype=x.dtype)
    for layer in net.net:
        if isinstance(layer, nn.Linear):
            Wt = layer.weight.t()
            h = h @ Wt + layer.bias
            J = J @ Wt
            L = L @ Wt
        elif isinstance(layer, nn.Tanh):
            t = torch.tanh(h)
            s1 = 1.0 - t * t
            L = s1 * L - 2.0 * t * s1 * (J * J).sum(1)
            J = s1.unsqueeze(1) * J
            h = t
        else:
            raise TypeError(layer)
    return h, J[:, :, 0], L


# ----------------------------------------------------------------------------- losses
def loss_pinn_nested(net, problem, xi, xb, w):
    """PINN loss with the Laplacian from nested reverse-mode autograd."""
    xi = xi.requires_grad_(True)
    u = net(xi)
    lap, _ = laplacian(u, xi)
    res = lap + source(problem, xi.detach())
    l_int = (res ** 2).mean()
    l_bd = ((net(xb) - exact(problem, xb)) ** 2).mean()
    return l_int + w * l_bd, l_int.detach(), l_bd.detach()


def loss_pinn(net, problem, xi, xb, w):
    """PINN loss with the forward-Laplacian (identical loss value, cheaper)."""
    u, _, lap = forward_laplacian(net, xi)
    res = lap + source(problem, xi)
    l_int = (res ** 2).mean()
    l_bd = ((net(xb) - exact(problem, xb)) ** 2).mean()
    return l_int + w * l_bd, l_int.detach(), l_bd.detach()


def loss_ritz(net, problem, xi, xb, w):
    d = xi.shape[1]
    xi = xi.requires_grad_(True)
    u = net(xi)
    g = torch.autograd.grad(u.sum(), xi, create_graph=True)[0]
    energy = (0.5 * (g ** 2).sum(1, keepdim=True) - source(problem, xi.detach()) * u).mean()  # |Ω| = 1
    l_bd = ((net(xb) - exact(problem, xb)) ** 2).mean() * (2 * d)  # |∂Ω| = 2d
    return energy + w * l_bd, energy.detach(), l_bd.detach()


# ----------------------------------------------------------------------------- metrics
def evaluate(net, problem, x, xb=None, grad=False):
    """Relative L2 error on MC points x (plus optional extras)."""
    out = {}
    if grad:
        x = x.clone().requires_grad_(True)
        u = net(x)
        gu = torch.autograd.grad(u.sum(), x)[0]
        ge = exact_grad(problem, x.detach())
        out["rel_h1semi"] = (((gu - ge) ** 2).sum() / (ge ** 2).sum()).sqrt().item()
        u = u.detach(); x = x.detach()
    else:
        with torch.no_grad():
            u = net(x)
    ue = exact(problem, x)
    e2 = ((u - ue) ** 2).sum()
    out["rel_l2"] = (e2 / (ue ** 2).sum()).sqrt().item()
    # error relative to the fluctuation of u* (not flattered when u* concentrates around its mean)
    out["rel_l2_centered"] = (e2 / ((ue - ue.mean()) ** 2).sum()).sqrt().item()
    out["max_abs"] = (u - ue).abs().max().item()
    if xb is not None:
        with torch.no_grad():
            ub = net(xb); ubx = exact(problem, xb)
        out["bd_rel_l2"] = (((ub - ubx) ** 2).sum() / (ubx ** 2).sum()).sqrt().item()
    return out


def fixed_sets(d, n_test=50_000, n_val=5_000, n_bd=20_000):
    """Held-out sets use their own generators (seed depends only on d), disjoint from training streams."""
    gt = torch.Generator().manual_seed(10_000 + d)
    gv = torch.Generator().manual_seed(20_000 + d)
    return dict(test=sample_interior(n_test, d, gt), test_bd=sample_boundary(n_bd, d, gt),
                val=sample_interior(n_val, d, gv))


# ----------------------------------------------------------------------------- training
def train(problem, method, d, seed, w, iters=4000, n_int=1024, n_bd=1024, lr=1e-3,
          width=64, depth=3, log_every=250, sets=None, verbose=False, lap_impl="forward"):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1_000 + seed)
    net = MLP(d, width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[iters // 2, (3 * iters) // 4], gamma=0.1)
    lossf = (loss_pinn if lap_impl == "forward" else loss_pinn_nested) if method == "pinn" else loss_ritz
    sets = sets or fixed_sets(d)
    curve, t_train, c_train = [], 0.0, 0.0
    for it in range(iters + 1):
        if it % log_every == 0 or it == iters:
            ev = evaluate(net, problem, sets["val"])
            curve.append(dict(it=it, t=t_train, val_rel_l2=ev["rel_l2"]))
            if verbose:
                print(f"{method} d={d} seed={seed} it={it} val={ev['rel_l2']:.3e} t={t_train:.1f}s", flush=True)
        if it == iters:
            break
        t0 = time.perf_counter(); c0 = time.process_time()
        xi = sample_interior(n_int, d, gen); xb = sample_boundary(n_bd, d, gen)
        opt.zero_grad(set_to_none=True)
        loss, li, lb = lossf(net, problem, xi, xb, w)
        loss.backward()
        opt.step(); sched.step()
        t_train += time.perf_counter() - t0; c_train += time.process_time() - c0
        if not torch.isfinite(loss):
            break
    final = evaluate(net, problem, sets["test"], sets["test_bd"], grad=True)
    final.update(val_rel_l2=curve[-1]["val_rel_l2"], train_time_s=t_train,
                 ms_per_iter=1e3 * t_train / max(iters, 1), cpu_time_s=c_train,
                 cpu_ms_per_iter=1e3 * c_train / max(iters, 1),
                 lap_impl=(lap_impl if method == "pinn" else "n/a"), final_loss=loss.item(),
                 final_loss_int=li.item(), final_loss_bd=lb.item(),
                 n_params=sum(p.numel() for p in net.parameters()))
    return net, final, curve


def baselines(problem, d, sets=None):
    """Error of trivial predictors on the test set: u ≡ 0 gives 1; best constant gives the floor
    a method must beat to have learned anything."""
    sets = sets or fixed_sets(d)
    ue = exact(problem, sets["test"])
    c = ue.mean()
    return dict(const_rel_l2=(((ue - c) ** 2).sum() / (ue ** 2).sum()).sqrt().item(),
                u_rms=ue.pow(2).mean().sqrt().item(), u_mean=c.item(), u_std=ue.std().item())


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush(); os.fsync(f.fileno())


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]
