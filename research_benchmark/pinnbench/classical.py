"""Classical finite-difference baselines (second order) for the five benchmark problems.

Every solver returns (axes, U) with U on the tensor grid spanned by `axes` (same coordinate
order as the PINN problems, t first). `evaluate` interpolates U to arbitrary points so the
error is measured on exactly the same held-out grid as the PINNs.

heat1d     Crank-Nicolson in time, centred differences in space, dt = dx.
wave1d     explicit leapfrog (centred in t and x), Courant number 0.5 (NOT 1, where the 1D
           scheme is accidentally exact), second-order start-up step.
wave2d     explicit leapfrog, 5-point Laplacian, dt/dx = 0.5 (stability limit 1/sqrt 2).
laplace2d  5-point Laplacian, Dirichlet in x, Neumann in y via second-order ghost points,
           sparse direct solve (SuperLU).
laplace3d  7-point Laplacian with Dirichlet data, solved directly by a 2D discrete sine
           transform in (y, z) and the closed-form solution of the remaining three-term
           recurrence in x (fast Poisson solver, O(N^3 log N)).
"""
from __future__ import annotations

import numpy as np
import scipy.fft
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_banded

PI = np.pi


def heat1d_cn(nx):
    nt = nx
    x = np.linspace(0, 1, nx + 1)
    t = np.linspace(0, 1, nt + 1)
    dx, dt = 1 / nx, 1 / nt
    r = dt / dx ** 2
    m = nx - 1
    ab = np.zeros((3, m))
    ab[0, 1:] = -r / 2
    ab[1, :] = 1 + r
    ab[2, :-1] = -r / 2
    U = np.zeros((nt + 1, nx + 1))
    U[0] = np.sin(PI * x)
    for n in range(nt):
        u = U[n, 1:-1]
        rhs = (1 - r) * u
        rhs[1:] += r / 2 * u[:-1]
        rhs[:-1] += r / 2 * u[1:]
        U[n + 1, 1:-1] = solve_banded((1, 1), ab, rhs)
    return (t, x), U


def wave1d_leapfrog(nx, c=2.0, courant=0.5):
    dx = 1 / nx
    nt = int(round(c / (courant * dx)))
    dt = 1 / nt
    s2 = (c * dt / dx) ** 2
    x = np.linspace(0, 1, nx + 1)
    t = np.linspace(0, 1, nt + 1)
    U = np.zeros((nt + 1, nx + 1))
    U[0] = np.sin(PI * x) + 0.5 * np.sin(4 * PI * x)
    U[0, 0] = U[0, -1] = 0
    lap = lambda u: u[2:] - 2 * u[1:-1] + u[:-2]
    U[1, 1:-1] = U[0, 1:-1] + 0.5 * s2 * lap(U[0])           # u_t(0) = 0
    for n in range(1, nt):
        U[n + 1, 1:-1] = 2 * U[n, 1:-1] - U[n - 1, 1:-1] + s2 * lap(U[n])
    return (t, x), U


def wave2d_leapfrog(n, courant=0.5, t_out=np.linspace(0, 1, 21)):
    dx = 2 / n
    nt = int(round(1 / (courant * dx)))
    dt = 1 / nt
    s2 = (dt / dx) ** 2
    x = np.linspace(-1, 1, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u0 = np.sin(PI * X) * np.sin(PI * Y)
    u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0

    def lap(u):
        return u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]

    out_idx = {int(round(tt * nt)): k for k, tt in enumerate(t_out)}
    assert np.allclose(np.array(sorted(out_idx)) * dt, t_out), "output times must be on the time grid"
    U = np.zeros((len(t_out), n + 1, n + 1))
    U[out_idx[0]] = u0
    um, u = u0, u0.copy()
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + 0.5 * s2 * lap(u0)        # u_t(0) = 0
    if 1 in out_idx:
        U[out_idx[1]] = u
    for k in range(1, nt):
        un = np.zeros_like(u)
        un[1:-1, 1:-1] = 2 * u[1:-1, 1:-1] - um[1:-1, 1:-1] + s2 * lap(u)
        um, u = u, un
        if k + 1 in out_idx:
            U[out_idx[k + 1]] = u
    return (np.asarray(t_out), x, x), U


def laplace2d_fd(n):
    h = PI / n
    x = np.linspace(0, PI, n + 1)
    y = np.linspace(0, PI, n + 1)
    nxi, ny = n - 1, n + 1                                    # unknowns: x interior, y all (Neumann)
    Tx = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nxi, nxi), format="lil")
    Ty = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(ny, ny), format="lil")
    Ty[0, 1] = 2                                              # ghost point: u_{-1} = u_{1}
    Ty[ny - 1, ny - 2] = 2
    A = (sp.kron(Tx.tocsr(), sp.identity(ny)) + sp.kron(sp.identity(nxi), Ty.tocsr())).tocsc() / h ** 2
    b = np.zeros((nxi, ny))
    b[-1, :] -= 0.5 * (1 + np.cos(2 * y)) / h ** 2            # Dirichlet data at x = pi
    u = spla.spsolve(A, b.ravel()).reshape(nxi, ny)
    U = np.zeros((n + 1, n + 1))
    U[1:-1, :] = u
    U[-1, :] = 0.5 * (1 + np.cos(2 * y))
    return (x, y), U


def laplace3d_fd(n):
    h = PI / n
    x = np.linspace(0, PI, n + 1)
    j = np.arange(1, n)
    g = np.sin(x[1:-1])[:, None] * np.cos(x[1:-1])[None, :]   # data on x = pi at interior (y, z) nodes
    ghat = scipy.fft.dstn(g, type=1)
    lam = 4 * np.sin(j * PI / (2 * n)) ** 2                    # h^2 * eigenvalues of -d2/dy2 (Dirichlet)
    theta = np.arccosh(1 + 0.5 * (lam[:, None] + lam[None, :]))
    i = np.arange(n + 1)[:, None, None]
    ratio = np.exp(theta * (i - n)) * (-np.expm1(-2 * theta * i)) / (-np.expm1(-2 * theta * n))
    Uint = scipy.fft.idstn(ratio * ghat[None], type=1, axes=(1, 2))
    U = np.zeros((n + 1, n + 1, n + 1))
    U[:, 1:-1, 1:-1] = Uint
    return (x, x, x), U


def evaluate(axes, U, pts):
    """Interpolate the grid solution to `pts`.

    If every evaluation point is a grid node the nodal values are returned (multilinear
    interpolation is exact there); otherwise cubic tensor-product splines are used, whose
    O(h^4) error is below the O(h^2) discretisation error of the schemes."""
    coincide = True
    for d, a in enumerate(axes):
        q = (pts[:, d] - a[0]) / (a[1] - a[0])
        coincide &= bool(np.abs(q - np.round(q)).max() < 1e-6)
    if coincide:
        return RegularGridInterpolator(axes, U, method="linear")(pts)
    assert U.size <= 400_000, "non-coincident evaluation on a large grid: choose n so that eval points are nodes"
    return RegularGridInterpolator(axes, U, method="cubic")(pts)


SOLVERS = {
    "heat1d": (heat1d_cn, [25, 50, 100, 200, 400, 800]),
    "wave1d": (wave1d_leapfrog, [25, 50, 100, 200, 400, 800]),
    "wave2d": (wave2d_leapfrog, [20, 40, 80, 160, 320]),
    "laplace2d": (laplace2d_fd, [25, 50, 100, 200, 400]),
    "laplace3d": (laplace3d_fd, [20, 40, 80, 160]),
}
