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

"""Core code for the pre-registered test of two cheap remedies for the affine plateau (PREREG_C1.md).

Everything that the existing study fixes is imported unchanged from
pinn/research_highdim/scripts/hd_core.py: network (MLP d-64-64-64-1, tanh), samplers, held-out sets,
the forward Laplacian of a plain MLP, optimiser and step schedule. The plain arm of P1/P2 calls exactly
the same operations in the same order as hd_core.train, so that it reproduces the main study bit for bit
(checked by scripts/run_c1.py, phase "repro").

Problems on the unit cube (0,1)^d, -Lap u = f, u = g on the boundary:
  P1 "laplace" : u* = sum_{k<=d/2} x_{2k-1} x_{2k},                f = 0          (hd_core)
  P2 "poisson" : u* = d^{-1/2} sum_i cos(pi x_i),                   f = pi^2 u*    (hd_core)
  P3 "ridge"   : u* = cos(2 s), s = d^{-1/2} sum_i (2 x_i - 1),     f = 16 u*
  P4 "cospair" : u* = sum_{k<=d/2} cos(pi x_{2k-1}) cos(pi x_{2k}),  f = 2 pi^2 u*
  P5 "altridge": u* = cos(pi s + 1), s = d^{-1/2} sum_i e_i (2 x_i - 1), e_i = +1 (i odd), -1 (i even),
                 f = 4 pi^2 u*   (added after the check, PREREG_C1_P5.md; not part of PREREG_C1.md)
Arms:
  plain    : u = N(x)
  presolve : u = p(x) + N(x), p the exact minimiser of the method's own loss over affine functions
             (src/presolve.py), fixed; only N is trained
  lift3c   : u = N(P1(t), P2(t), P3(t)), t = 2x - 1 coordinate-wise (Legendre polynomials; 3d inputs)
  lift3u   : u = N(x, P2(t), P3(t))  (linear feature not centred; Deep Ritz only)
  rep3     : u = N(t, t, t)           (fan-in control; Deep Ritz only)
"""
import math, os, sys, time, json
import torch
import torch.nn as nn

HD = _repo_path("research_highdim/scripts")
sys.path.insert(0, HD)
import hd_core as hc  # noqa: E402

torch.set_num_threads(1)
DT = hc.DTYPE
PROBLEMS = {"P1": "laplace", "P2": "poisson", "P3": "ridge", "P4": "cospair", "P5": "altridge"}


def alt_signs(d, dtype=None):
    """e_i = +1 for odd i, -1 for even i (1-based), i.e. +1, -1, +1, ... in 0-based column order."""
    e = torch.ones(d, dtype=dtype if dtype is not None else DT)
    e[1::2] = -1.0
    return e


# ----------------------------------------------------------------------------- problems
def exact(problem, x):
    if problem in ("laplace", "poisson"):
        return hc.exact(problem, x)
    d = x.shape[1]
    if problem == "ridge":
        s = (2.0 * x - 1.0).sum(1, keepdim=True) / math.sqrt(d)
        return torch.cos(2.0 * s)
    if problem == "cospair":
        m = d // 2
        c = torch.cos(math.pi * x[:, :2 * m])
        return (c[:, 0:2 * m:2] * c[:, 1:2 * m:2]).sum(1, keepdim=True)
    if problem == "altridge":
        s = ((2.0 * x - 1.0) * alt_signs(d, x.dtype)).sum(1, keepdim=True) / math.sqrt(d)
        return torch.cos(math.pi * s + 1.0)
    raise ValueError(problem)


def source(problem, x):
    """f = -Lap u*"""
    if problem in ("laplace", "poisson"):
        return hc.source(problem, x)
    if problem == "ridge":
        return 16.0 * exact(problem, x)
    if problem == "cospair":
        return 2.0 * math.pi ** 2 * exact(problem, x)
    if problem == "altridge":
        return 4.0 * math.pi ** 2 * exact(problem, x)
    raise ValueError(problem)


# ----------------------------------------------------------------------------- input maps
def feature_jet(x, kind):
    """features phi (n x 3d, block layout [a=0 | a=1 | a=2], block a holds coordinate i at column a*d+i),
    their derivative phi' and second derivative phi'' with respect to the coordinate they belong to."""
    t = 2.0 * x - 1.0
    one = torch.ones_like(t)
    zero = torch.zeros_like(t)
    if kind == "lift3c":
        f = [t, 1.5 * t * t - 0.5, 2.5 * t ** 3 - 1.5 * t]
        f1 = [2.0 * one, 6.0 * t, 15.0 * t * t - 3.0]
        f2 = [zero, 12.0 * one, 60.0 * t]
    elif kind == "lift3u":
        f = [x, 1.5 * t * t - 0.5, 2.5 * t ** 3 - 1.5 * t]
        f1 = [one, 6.0 * t, 15.0 * t * t - 3.0]
        f2 = [zero, 12.0 * one, 60.0 * t]
    elif kind == "lift1":   # post-hoc exploratory arm (later choice, notes/VERIFICATION.md section 4): centred input only
        f, f1, f2 = [t], [2.0 * one], [zero]
    elif kind == "rep3":
        f = [t, t, t]
        f1 = [2.0 * one] * 3
        f2 = [zero] * 3
    else:
        raise ValueError(kind)
    return torch.cat(f, 1), torch.cat(f1, 1), torch.cat(f2, 1)


class Model(nn.Module):
    """u(x) = c0 + c.x + N(phi(x)).  For the plain arm phi = id and c = 0; then forward() is N(x)
    with no extra operation, so the plain arm is the hd_core network."""

    def __init__(self, d, arm, shift=None, width=64, depth=3):
        super().__init__()
        self.d, self.arm = d, arm
        self.lift = arm if arm in ("lift3c", "lift3u", "rep3", "lift1") else None
        self.nb = 1 if arm in ("plain", "presolve", "lift1") else 3   # number of feature blocks of width d
        self.mlp = hc.MLP(self.nb * d, width, depth)
        self.net = self.mlp.net
        self.has_shift = shift is not None
        c = torch.zeros(d + 1, dtype=DT) if shift is None else torch.as_tensor(shift).to(DT)
        self.register_buffer("c", c)

    def forward(self, x):
        z = feature_jet(x, self.lift)[0] if self.lift else x
        u = self.mlp(z)
        if self.has_shift:
            u = u + (self.c[0] + x @ self.c[1:].unsqueeze(1))
        return u

    def u_lap(self, x):
        """(u, Lap u) by one forward sweep of second-order Taylor mode (no higher-order graph)."""
        if not self.lift:
            u, _, lap = hc.forward_laplacian(self.mlp, x)
        else:
            n, d = x.shape
            h, g1, g2 = feature_jet(x, self.lift)
            J = torch.zeros(n, d, self.nb * d, dtype=x.dtype)
            idx = torch.arange(d)
            for a in range(self.nb):
                J[:, idx, a * d + idx] = g1[:, a * d:(a + 1) * d]
            L = g2
            for layer in self.mlp.net:
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
            u, lap = h, L
        if self.has_shift:  # the shift is affine: it adds to u and nothing to Lap u
            u = u + (self.c[0] + x @ self.c[1:].unsqueeze(1))
        return u, lap


# ----------------------------------------------------------------------------- losses
def loss_ritz(model, problem, xi, xb, w):
    """Same expression and operation order as hd_core.loss_ritz."""
    d = xi.shape[1]
    xi = xi.requires_grad_(True)
    u = model(xi)
    g = torch.autograd.grad(u.sum(), xi, create_graph=True)[0]
    energy = (0.5 * (g ** 2).sum(1, keepdim=True) - source(problem, xi.detach()) * u).mean()
    l_bd = ((model(xb) - exact(problem, xb)) ** 2).mean() * (2 * d)
    return energy + w * l_bd, energy.detach(), l_bd.detach()


def loss_pinn(model, problem, xi, xb, w):
    """Same expression and operation order as hd_core.loss_pinn."""
    u, lap = model.u_lap(xi)
    res = lap + source(problem, xi)
    l_int = (res ** 2).mean()
    l_bd = ((model(xb) - exact(problem, xb)) ** 2).mean()
    return l_int + w * l_bd, l_int.detach(), l_bd.detach()


# ----------------------------------------------------------------------------- metrics
class Decomp:
    """u* = a + q on a fixed point set, a = least-squares affine fit (in x), q = remainder.
    gamma = share of q captured by the model's non-affine part (as in ideas_dynamics/dyn_core.py)."""

    def __init__(self, problem, x):
        n, d = x.shape
        A = torch.cat([torch.ones(n, 1, dtype=torch.float64), x.double()], 1)
        self.Q, _ = torch.linalg.qr(A)
        self.ue = exact(problem, x).double()
        self.q = self.ue - self.P(self.ue)
        self.qq = (self.q ** 2).sum()
        self.nu = self.ue.norm()

    def P(self, v):
        return self.Q @ (self.Q.t() @ v)

    def measure(self, u):
        u = u.double()
        e = u - self.ue
        r = u - self.P(u)
        return dict(rel=(e.norm() / self.nu).item(), e_aff=(self.P(e).norm() / self.nu).item(),
                    gamma=((r * self.q).sum() / self.qq).item())


def evaluate(model, problem, x, xb):
    """Relative L2 (same formula as hd_core.evaluate), centred relative L2, max error, boundary error."""
    with torch.no_grad():
        u = model(x)
        ue = exact(problem, x)
        e2 = ((u - ue) ** 2).sum()
        ub = model(xb)
        ubx = exact(problem, xb)
    return dict(rel_l2=(e2 / (ue ** 2).sum()).sqrt().item(),
                rel_l2_centered=(e2 / ((ue - ue.mean()) ** 2).sum()).sqrt().item(),
                max_abs=(u - ue).abs().max().item(),
                bd_rel_l2=(((ub - ubx) ** 2).sum() / (ubx ** 2).sum()).sqrt().item())


# ----------------------------------------------------------------------------- training
def train(problem, method, d, seed, w, arm="plain", iters=4000, schedule=True, lr=1e-3,
          n_int=1024, n_bd=1024, log_every=50, shift=None, shift_cpu=0.0):
    """Mirrors hd_core.train: torch.manual_seed(seed) -> network init; sample stream from
    Generator(1000 + seed); Adam(lr); MultiStepLR at iters//2 and 3 iters//4 (factor 0.1) if schedule.
    `shift` (presolve arm) is computed by the caller before this call and uses no random numbers."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1_000 + seed)
    model = Model(d, arm, shift=shift)
    opt = torch.optim.Adam(model.mlp.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[iters // 2, (3 * iters) // 4], gamma=0.1)
             if schedule else None)
    lossf = loss_pinn if method == "pinn" else loss_ritz
    sets = hc.fixed_sets(d)
    dec = Decomp(problem, sets["val"])
    curve, cpu, wall = [], 0.0, 0.0
    loss = li = lb = torch.tensor(float("nan"))
    for it in range(iters + 1):
        if it % log_every == 0 or it == iters:
            with torch.no_grad():
                m = dec.measure(model(sets["val"]))
            m.update(it=it, cpu=cpu)
            curve.append(m)
        if it == iters:
            break
        t0 = time.perf_counter(); c0 = time.process_time()
        xi = hc.sample_interior(n_int, d, gen); xb = hc.sample_boundary(n_bd, d, gen)
        opt.zero_grad(set_to_none=True)
        loss, li, lb = lossf(model, problem, xi, xb, w)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        wall += time.perf_counter() - t0; cpu += time.process_time() - c0
        if not torch.isfinite(loss):
            break
    rec = evaluate(model, problem, sets["test"], sets["test_bd"])
    rec.update(val_rel_l2=curve[-1]["rel"], gamma_end=curve[-1]["gamma"], its_done=it,
               cpu_train_s=cpu, cpu_shift_s=shift_cpu, cpu_total_s=cpu + shift_cpu, wall_train_s=wall,
               cpu_ms_per_iter=1e3 * cpu / max(it, 1), final_loss=loss.item(), final_loss_int=li.item(),
               final_loss_bd=lb.item(), n_params=sum(p.numel() for p in model.mlp.parameters()),
               T_half=next((c["it"] for c in curve if c["gamma"] >= 0.5), None),
               T10=next((c["it"] for c in curve if c["gamma"] >= 0.1), None))
    return model, rec, curve


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush(); os.fsync(f.fileno())


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]
