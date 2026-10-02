"""Separately written checks of the study's finite-difference baseline numbers.
heat / wave2d / laplace2d: single-eigenmode data -> the FD solution is known in closed form from the
discrete amplification factor / dispersion relation (no solver needed). laplace3d: own sparse CG solve."""
import math, time, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
PI = math.pi
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
print("heat1d Crank-Nicolson (dt=dx=1/n): closed-form discrete solution, error on the node grid")
for n in (25, 50, 100, 200):
    dx = dt = 1 / n; lam = 4 * math.sin(PI * dx / 2) ** 2 / dx ** 2
    G = (1 - dt * lam / 2) / (1 + dt * lam / 2)
    t = np.arange(n + 1) * dt; x = np.arange(n + 1) * dx
    U = (G ** np.arange(n + 1))[:, None] * np.sin(PI * x)[None]; E = np.exp(-PI ** 2 * t)[:, None] * np.sin(PI * x)[None]
    print("  n=%d rel-L2 %.3e Linf %.3e" % (n, rel(U, E), np.abs(U - E).max()))
print("wave2d leapfrog (dt/dx=0.5): closed-form discrete dispersion, error on t=k/20 output grid")
for n in (20, 40, 80):
    dx = 2 / n; nt = round(1 / (0.5 * dx)); dt = 1 / nt; s2 = (dt / dx) ** 2
    lam = 8 * math.sin(PI * dx / 2) ** 2
    wh = math.acos(1 - s2 * lam / 2) / dt
    t = np.linspace(0, 1, 21)
    print("  n=%d omega_h %.5f (exact %.5f) rel-L2 %.3e Linf %.3e" % (n, wh, math.sqrt(2) * PI, rel(np.cos(wh * t), np.cos(math.sqrt(2) * PI * t)), np.abs(np.cos(wh * t) - np.cos(math.sqrt(2) * PI * t)).max()))
print("laplace2d 5-point: closed-form discrete solution x/2pi + cos2y sinh(theta i)/(2 sinh(theta n)), cosh(theta)=1+2 sin^2 h")
for n in (25, 50, 100, 200):
    h = PI / n; th = math.acosh(1 + 2 * math.sin(h) ** 2)
    i = np.arange(n + 1); x = i * h; y = i * h
    U = x[:, None] / (2 * PI) + np.cos(2 * y)[None] * (np.sinh(th * i) / (2 * np.sinh(th * n)))[:, None]
    E = x[:, None] / (2 * PI) + np.cos(2 * y)[None] * (np.sinh(2 * x) / (2 * np.sinh(2 * PI)))[:, None]
    # verify U satisfies the 5-point scheme
    r = np.abs(U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2] - 4 * U[1:-1, 1:-1]).max()
    print("  n=%d rel-L2 %.3e Linf %.3e (max 5-pt residual %.1e)" % (n, rel(U, E), np.abs(U - E).max(), r))
print("laplace3d 7-point: own sparse CG solve, evaluated by cubic interpolation on the 40^3 cell-centred grid")
def exact3(X, nmax=4000):
    out = np.zeros(len(X)); nn = np.arange(2, nmax + 1, 2.0); b = 4 * nn / (PI * (nn * nn - 1)); k = np.sqrt(1 + nn * nn)
    for s in range(0, len(X), 8000):
        x = X[s:s + 8000, 0:1]; z = X[s:s + 8000, 2:3]
        out[s:s + 8000] = np.sin(X[s:s + 8000, 1]) * (b * np.sin(nn * z) * np.exp(k * (x - PI)) * (1 - np.exp(-2 * k * x)) / (1 - np.exp(-2 * k * PI))).sum(1)
    return out
a = (np.arange(40) + 0.5) * PI / 40; eX = np.stack([g.ravel() for g in np.meshgrid(a, a, a, indexing="ij")], 1); eU = exact3(eX)
for n in (20, 40):
    t0 = time.perf_counter()
    m = n - 1; h = PI / n; ax = np.linspace(0, PI, n + 1)
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m)); I = sp.identity(m)
    A = (sp.kron(sp.kron(T, I), I) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(I, I), T)).tocsr()
    b = np.zeros((m, m, m)); b[-1] = np.sin(ax[1:-1])[:, None] * np.cos(ax[1:-1])[None, :]
    u, info = spla.cg(A, b.ravel(), rtol=1e-12, maxiter=5000)
    U = np.zeros((n + 1,) * 3); U[1:-1, 1:-1, 1:-1] = u.reshape(m, m, m)
    # boundary face x=pi carries the data (as in the study's array the face values are left 0 except interior? use data)
    U[-1, 1:-1, 1:-1] = np.sin(ax[1:-1])[:, None] * np.cos(ax[1:-1])[None, :]
    wall = time.perf_counter() - t0
    for face_data in (True, False):
        V = U.copy()
        if not face_data: V[-1] = 0
        p = RegularGridInterpolator((ax, ax, ax), V, method="cubic")(eX)
        print("  n=%d (x=pi face values %s) rel-L2 %.3e Linf %.3e  [cg info %d, %.2fs]" % (n, "= data" if face_data else "= 0 (as in study's array)", rel(p, eU), np.abs(p - eU).max(), info, wall))
