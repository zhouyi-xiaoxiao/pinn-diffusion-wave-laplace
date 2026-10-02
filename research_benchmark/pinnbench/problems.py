"""PDE problem definitions.

Every problem exposes
  name, d (input dimension), lo/hi (box bounds, numpy float64), hidden (network widths)
  losses(net, X_r)   -> (pde_mse, {constraint name: mse}) from ONE forward pass over the
                        concatenation [X_r ; all constraint points]
  residual(net, X)   -> PDE residual at arbitrary points (used by residual-adaptive sampling)
  exact(X)           -> exact solution (float64 numpy) at points X (N, d)
  eval_points()      -> held-out dense evaluation grid (float64 numpy, (M, d))

Shapes: point tensors are (N, d) for a single model and (S, N, d) for S stacked independent
models; all means are taken over the point axis only, so losses are scalars or (S,) vectors.
Coordinate convention: for time-dependent problems column 0 is t.

Loss composition: mean(r^2) over the interior collocation set plus, with unit
weights, one mean-squared term per edge / face / initial condition.

Problem statements (exact solutions are checked in scripts/verify_exact.py):

heat1d     u_t = u_xx, (t,x) in [0,1]^2, u(0,x)=sin(pi x), u(t,0)=u(t,1)=0.
           exact u = exp(-pi^2 t) sin(pi x).
wave1d     u_tt = c^2 u_xx, c = 2, (t,x) in [0,1]^2, u(0,x)=sin(pi x)+0.5 sin(4 pi x),
           u_t(0,x)=0, u(t,0)=u(t,1)=0;
           exact u = sin(pi x)cos(2 pi t) + 0.5 sin(4 pi x)cos(8 pi t).
           (the standard hard case of Wang, Yu & Perdikaris, JCP 2022)
wave2d     u_tt = u_xx+u_yy, t in [0,1], (x,y) in [-1,1]^2, u(0)=sin(pi x)sin(pi y), u_t(0)=0,
           u=0 on the boundary; exact u = sin(pi x) sin(pi y) cos(sqrt(2) pi t).
laplace2d  u_xx+u_yy=0 on [0,pi]^2, u(0,y)=0, u(pi,y)=(1+cos 2y)/2, u_y(x,0)=u_y(x,pi)=0;
           exact u = x/(2 pi) + cos(2y) sinh(2x) / (2 sinh(2 pi)).
laplace3d  u_xx+u_yy+u_zz=0 on [0,pi]^3, u(pi,y,z)=sin y cos z, u=0 on the other 5 faces;
           exact u = sin y * sum_{n even} b_n sin(n z) sinh(k_n x)/sinh(k_n pi),
           b_n = 4n / (pi (n^2-1)), k_n = sqrt(1+n^2). The data are discontinuous along the
           edges {x=pi, z=0} and {x=pi, z=pi}.
"""
from __future__ import annotations

import math

import numpy as np
import torch

PI = math.pi


# ----------------------------------------------------------------------------------------
# derivative helpers (work for (N, d) and stacked (S, N, d) inputs alike)
# ----------------------------------------------------------------------------------------
def gradient(u: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """du/dX for a network output u (..., N, 1) evaluated at X (..., N, d); keeps the graph."""
    return torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]


def second(g: torch.Tensor, X: torch.Tensor, i: int) -> torch.Tensor:
    """d^2u/dX_i^2 given the full first gradient g = du/dX."""
    return torch.autograd.grad(g[..., i].sum(), X, create_graph=True)[0][..., i]


def _grid(*axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


# ----------------------------------------------------------------------------------------
class Problem:
    """Base class. Subclasses define `blocks()` (constraint point sets), `terms()` (loss terms on
    those sets), `pde(g, X)` (residual from the first gradient), `exact` and `eval_points`."""
    name = "base"
    d = 2
    lo = np.zeros(2)
    hi = np.ones(2)
    hidden = [128, 128]

    def __init__(self, device="cpu", batch=None):
        self.device, self.batch = device, batch
        blocks = self.blocks()                               # list of (N_b, d) arrays
        offs = np.cumsum([0] + [b.shape[0] for b in blocks])
        self._ranges = [(int(offs[k]), int(offs[k + 1])) for k in range(len(blocks))]
        Xc = torch.as_tensor(np.concatenate(blocks).astype(np.float32), device=device)
        self.Xc = Xc if batch is None else Xc.unsqueeze(0).repeat(batch, 1, 1)
        # terms: (name, block index, kind, target); kind = "u" (value) or an int k (du/dX_k)
        self._terms = [(nm, b, kind, torch.as_tensor(np.asarray(tgt, dtype=np.float32), device=device))
                       for nm, b, kind, tgt in self.terms(blocks)]

    # ---- to be provided by subclasses ----
    def blocks(self):
        raise NotImplementedError

    def terms(self, blocks):
        raise NotImplementedError

    def pde(self, g, X):
        raise NotImplementedError

    def exact(self, X):
        raise NotImplementedError

    def eval_points(self):
        raise NotImplementedError

    # ---- generic machinery ----
    def residual(self, net, X):
        """PDE residual at X (X must require grad)."""
        return self.pde(gradient(net(X), X), X)

    def losses(self, net, X_r):
        """(pde_mse, {name: constraint_mse}) from a single forward pass.

        One pass over [X_r ; constraint points] keeps a single tensor shape per layer, which
        matters on Apple MPS (many distinct shapes trigger graph recompilation) and is also
        cheaper in the dispatch-bound regime."""
        n_r = X_r.shape[-2]
        X = torch.cat([X_r.detach(), self.Xc], dim=-2).requires_grad_(True)
        u = net(X)
        g = gradient(u, X)
        pde_mse = (self.pde(g, X)[..., :n_r] ** 2).mean(-1)
        comps = {}
        for nm, b, kind, tgt in self._terms:
            a, e = n_r + self._ranges[b][0], n_r + self._ranges[b][1]
            val = u[..., a:e, 0] if kind == "u" else g[..., a:e, kind]
            comps[nm] = ((val - tgt) ** 2).mean(-1)
        return pde_mse, comps

    def test_subset(self, n=4096, seed=12345):
        """Fixed random subset of the held-out grid, used only for training curves."""
        P = self.eval_points()
        idx = np.random.default_rng(seed).choice(P.shape[0], size=min(n, P.shape[0]), replace=False)
        return P[idx]


# ----------------------------------------------------------------------------------------
class Heat1D(Problem):
    name = "heat1d"
    d = 2
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    hidden = [128, 128]

    def blocks(self):
        s = np.linspace(0, 1, 100)                           # 100 IC + 2 x 100 BC points
        return [np.stack([0 * s, s], 1), np.stack([s, 0 * s], 1), np.stack([s, 0 * s + 1], 1)]

    def terms(self, B):
        return [("ic", 0, "u", np.sin(PI * B[0][:, 1])), ("bc_x0", 1, "u", np.zeros(100)), ("bc_x1", 2, "u", np.zeros(100))]

    def pde(self, g, X):
        return g[..., 0] - second(g, X, 1)

    def exact(self, X):
        return np.exp(-PI ** 2 * X[:, 0]) * np.sin(PI * X[:, 1])

    def eval_points(self):
        a = np.linspace(0, 1, 201)
        return _grid(a, a)


# ----------------------------------------------------------------------------------------
class Wave1D(Problem):
    name = "wave1d"
    d = 2
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    hidden = [64, 64, 64]  # 3 x 64 tanh
    c = 2.0

    def blocks(self):
        s = np.linspace(0, 1, 100)
        return [np.stack([0 * s, s], 1), np.stack([s, 0 * s], 1), np.stack([s, 0 * s + 1], 1)]

    def terms(self, B):
        x = B[0][:, 1]
        return [("ic", 0, "u", np.sin(PI * x) + 0.5 * np.sin(4 * PI * x)), ("ic_ut", 0, 0, np.zeros(100)),
                ("bc_x0", 1, "u", np.zeros(100)), ("bc_x1", 2, "u", np.zeros(100))]

    def pde(self, g, X):
        return second(g, X, 0) - self.c ** 2 * second(g, X, 1)

    def exact(self, X):
        t, x, c = X[:, 0], X[:, 1], self.c
        return np.sin(PI * x) * np.cos(c * PI * t) + 0.5 * np.sin(4 * PI * x) * np.cos(4 * c * PI * t)

    def eval_points(self):
        a = np.linspace(0, 1, 201)
        return _grid(a, a)


# ----------------------------------------------------------------------------------------
class Wave2D(Problem):
    name = "wave2d"
    d = 3
    lo = np.array([0.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    hidden = [50, 50]
    omega = math.sqrt(2.0) * PI

    def blocks(self):
        s = np.linspace(-1, 1, 21)
        out = [_grid(np.array([0.0]), s, s)]                 # 21 x 21 initial-condition grid
        F = _grid(np.linspace(0, 1, 11), s)                  # 11 (t) x 21 points per side face
        for axis in (1, 2):
            for val in (-1.0, 1.0):
                pts = np.zeros((F.shape[0], 3))
                pts[:, 0], pts[:, axis], pts[:, 3 - axis] = F[:, 0], val, F[:, 1]
                out.append(pts)
        return out

    def terms(self, B):
        ic = B[0]
        t = [("ic", 0, "u", np.sin(PI * ic[:, 1]) * np.sin(PI * ic[:, 2])), ("ic_ut", 0, 0, np.zeros(ic.shape[0]))]
        return t + [(f"bc{k}", k, "u", np.zeros(B[k].shape[0])) for k in range(1, 5)]

    def pde(self, g, X):
        return second(g, X, 0) - second(g, X, 1) - second(g, X, 2)

    def exact(self, X):
        return np.sin(PI * X[:, 1]) * np.sin(PI * X[:, 2]) * np.cos(self.omega * X[:, 0])

    def eval_points(self):
        return _grid(np.linspace(0, 1, 21), np.linspace(-1, 1, 81), np.linspace(-1, 1, 81))


# ----------------------------------------------------------------------------------------
class Laplace2D(Problem):
    name = "laplace2d"
    d = 2
    lo = np.array([0.0, 0.0])
    hi = np.array([PI, PI])
    hidden = [128, 128, 128]

    def blocks(self):
        s = np.linspace(0, PI, 100)                          # 100 points per edge
        z, p = 0 * s, 0 * s + PI
        return [np.stack([z, s], 1), np.stack([p, s], 1), np.stack([s, z], 1), np.stack([s, p], 1)]

    def terms(self, B):
        y = B[1][:, 1]
        return [("d_x0", 0, "u", np.zeros(100)), ("d_xpi", 1, "u", 0.5 * (1 + np.cos(2 * y))),
                ("n_y0", 2, 1, np.zeros(100)), ("n_ypi", 3, 1, np.zeros(100))]

    def pde(self, g, X):
        return second(g, X, 0) + second(g, X, 1)

    def exact(self, X):
        x, y = X[:, 0], X[:, 1]
        return x / (2 * PI) + np.cos(2 * y) * np.sinh(2 * x) / (2 * np.sinh(2 * PI))

    def eval_points(self):
        a = np.linspace(0, PI, 201)
        return _grid(a, a)


# ----------------------------------------------------------------------------------------
def laplace3d_series(X, n_max=4000, chunk=20000):
    """Exact solution of Laplace3D by the (double-precision) sine series in z.

    Uses sinh(k x)/sinh(k pi) = exp(k (x - pi)) (1 - exp(-2 k x)) / (1 - exp(-2 k pi)).
    Terms with n > n_max are dropped; scripts/verify_exact.py checks n_max = 4000 vs 8000."""
    X = np.asarray(X, dtype=np.float64)
    n = np.arange(2, n_max + 1, 2, dtype=np.float64)
    b = 4 * n / (PI * (n ** 2 - 1))
    k = np.sqrt(1 + n ** 2)
    out = np.empty(X.shape[0])
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk, 0:1]
        z = X[s:s + chunk, 2:3]
        ratio = np.exp(k * (x - PI)) * (-np.expm1(-2 * k * x)) / (-np.expm1(-2 * k * PI))
        out[s:s + chunk] = np.sin(X[s:s + chunk, 1]) * (b * np.sin(n * z) * ratio).sum(1)
    return out


class Laplace3D(Problem):
    name = "laplace3d"
    d = 3
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([PI, PI, PI])
    hidden = [100, 100]
    face_names = ["x0", "xpi", "y0", "ypi", "z0", "zpi"]

    def blocks(self):
        m = 12
        s = (np.arange(m) + 0.5) * PI / m                    # open 12 x 12 face grids (avoid the singular edges)
        F = _grid(s, s)
        out = []
        for axis in range(3):
            for val in (0.0, PI):
                pts = np.zeros((F.shape[0], 3))
                others = [a for a in range(3) if a != axis]
                pts[:, axis], pts[:, others[0]], pts[:, others[1]] = val, F[:, 0], F[:, 1]
                out.append(pts)
        return out

    def terms(self, B):
        t = []
        for k, nm in enumerate(self.face_names):
            tgt = np.sin(B[k][:, 1]) * np.cos(B[k][:, 2]) if nm == "xpi" else np.zeros(B[k].shape[0])
            t.append((f"bc_{nm}", k, "u", tgt))
        return t

    def pde(self, g, X):
        return second(g, X, 0) + second(g, X, 1) + second(g, X, 2)

    def exact(self, X):
        return laplace3d_series(X)

    def eval_points(self):
        m = 40
        a = (np.arange(m) + 0.5) * PI / m                    # cell-centred interior grid
        return _grid(a, a, a)


PROBLEMS = {c.name: c for c in (Heat1D, Wave1D, Wave2D, Laplace2D, Laplace3D)}


def make_problem(name, device="cpu", batch=None):
    return PROBLEMS[name](device=device, batch=batch)
