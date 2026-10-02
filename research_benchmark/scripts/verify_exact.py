"""Verify the exact solutions used as ground truth (float64, finite-difference residuals,
boundary/initial data, and series truncation for laplace3d). Output: checks/verify_exact.out"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pinnbench.problems import make_problem, laplace3d_series, PI

rng = np.random.default_rng(0)
h = 1e-4

def fd2(f, X, i):
    E = np.zeros_like(X); E[:, i] = h
    return (f(X + E) - 2 * f(X) + f(X - E)) / h**2

def fd1(f, X, i):
    E = np.zeros_like(X); E[:, i] = h
    return (f(X + E) - f(X - E)) / (2 * h)

for name in ["heat1d", "wave1d", "wave2d", "laplace2d", "laplace3d"]:
    P = make_problem(name)
    lo, hi = P.lo + 0.05 * (P.hi - P.lo), P.hi - 0.05 * (P.hi - P.lo)
    X = lo + (hi - lo) * rng.random((2000, P.d))
    f = P.exact
    if name == "heat1d":
        r = fd1(f, X, 0) - fd2(f, X, 1)
    elif name == "wave1d":
        r = fd2(f, X, 0) - P.c**2 * fd2(f, X, 1)
    elif name == "wave2d":
        r = fd2(f, X, 0) - fd2(f, X, 1) - fd2(f, X, 2)
    else:
        r = sum(fd2(f, X, i) for i in range(P.d))
    scale = np.abs(f(X)).max()
    print(f"{name}: max|FD residual of exact| = {np.abs(r).max():.2e} (|u|max {scale:.2e}; FD truncation ~h^2*u'''' and roundoff ~1e-16/h^2=1e-8)")

# boundary/initial data checks
P = make_problem("heat1d"); s = np.linspace(0, 1, 11)
print("heat1d IC err", np.abs(P.exact(np.stack([0*s, s], 1)) - np.sin(PI*s)).max(), "BC", np.abs(P.exact(np.stack([s, 0*s], 1))).max(), np.abs(P.exact(np.stack([s, 0*s+1], 1))).max())
P = make_problem("wave1d")
print("wave1d IC err", np.abs(P.exact(np.stack([0*s, s], 1)) - (np.sin(PI*s)+0.5*np.sin(4*PI*s))).max(), "u_t(0)", np.abs(fd1(P.exact, np.stack([0*s, s], 1), 0)).max(), "BC", np.abs(P.exact(np.stack([s, 0*s+1], 1))).max())
P = make_problem("wave2d"); s2 = np.linspace(-1, 1, 11)
G = np.stack(np.meshgrid(s2, s2, indexing='ij'), -1).reshape(-1, 2)
X0 = np.concatenate([np.zeros((len(G), 1)), G], 1)
print("wave2d IC err", np.abs(P.exact(X0) - np.sin(PI*G[:,0])*np.sin(PI*G[:,1])).max(), "u_t(0)", np.abs(fd1(P.exact, X0, 0)).max())
P = make_problem("laplace2d"); y = np.linspace(0, PI, 11)
print("laplace2d u(0,y)", np.abs(P.exact(np.stack([0*y, y], 1))).max(), "u(pi,y)-g", np.abs(P.exact(np.stack([0*y+PI, y], 1)) - 0.5*(1+np.cos(2*y))).max(),
      "u_y(x,0)", np.abs(fd1(P.exact, np.stack([y, 0*y], 1), 1)).max(), "u_y(x,pi)", np.abs(fd1(P.exact, np.stack([y, 0*y+PI], 1), 1)).max())

# laplace3d series truncation on the eval grid and data on faces away from singular edges
P = make_problem("laplace3d")
E = P.eval_points()
a = laplace3d_series(E, n_max=4000); b = laplace3d_series(E, n_max=8000)
print(f"laplace3d eval grid ({E.shape[0]} pts): max|series(4000)-series(8000)| = {np.abs(a-b).max():.2e}, ||u||_inf = {np.abs(b).max():.4f}")
yy, zz = np.meshgrid(np.linspace(0.3, PI-0.3, 15), np.linspace(0.3, PI-0.3, 15), indexing='ij')
F = np.stack([np.full(yy.size, PI - 1e-9), yy.ravel(), zz.ravel()], 1)
print("laplace3d face x=pi (z in [0.3, pi-0.3]): max|u - sin y cos z| =", np.abs(laplace3d_series(F, n_max=20000) - np.sin(F[:,1])*np.cos(F[:,2])).max())
F0 = F.copy(); F0[:, 0] = 1e-12
print("laplace3d face x=0: max|u| =", np.abs(laplace3d_series(F0)).max())
