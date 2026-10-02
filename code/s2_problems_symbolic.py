"""Symbolic check (SymPy) of the eight test problems of Section 2 (s2_problems).

For every problem of Table "problems" the script differentiates the stated exact solution
symbolically and prints the PDE residual and the mismatch of every initial / boundary
condition.  Every printed expression must be 0.  No training, no floating-point data;
runs in a few seconds.

It complements, and does not replace, the separately written checks
  research_benchmark/checks/verify_exact.out        (finite differences: H, W1, W2, L2, L3)
  research_highdim/results/check_core.txt           (automatic differentiation: P1, P2)
by covering the cases those files do not treat symbolically (W1, L3 term by term, L3s, P2,
and P1/P2 for d = 2, 3, 5, 10, 20).

Usage:  python s2_problems_symbolic.py      (writes ../data/s2_problems_symbolic.txt)
"""
import os
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "s2_problems_symbolic.txt")

lines = []
all_zero = True


def report(problem, what, expr):
    """Simplify expr, record it, and remember whether it is identically zero."""
    global all_zero
    val = sp.simplify(expr)
    ok = (val == 0)
    all_zero = all_zero and ok
    lines.append(f"{problem:4s} {what:46s} {val}")


t, x, y, z = sp.symbols("t x y z", real=True)
pi = sp.pi

# ---- H: heat --------------------------------------------------------------------------
u = sp.exp(-pi**2 * t) * sp.sin(pi * x)
report("H", "u_t - u_xx", sp.diff(u, t) - sp.diff(u, x, 2))
report("H", "u(0,x) - sin(pi x)", u.subs(t, 0) - sp.sin(pi * x))
report("H", "u(t,0)", u.subs(x, 0))
report("H", "u(t,1)", u.subs(x, 1))

# ---- W1: two-mode wave, c = 2 ---------------------------------------------------------
u = sp.sin(pi * x) * sp.cos(2 * pi * t) + sp.Rational(1, 2) * sp.sin(4 * pi * x) * sp.cos(8 * pi * t)
report("W1", "u_tt - 4 u_xx", sp.diff(u, t, 2) - 4 * sp.diff(u, x, 2))
report("W1", "u(0,x) - sin(pi x) - sin(4 pi x)/2",
       u.subs(t, 0) - sp.sin(pi * x) - sp.Rational(1, 2) * sp.sin(4 * pi * x))
report("W1", "u_t(0,x)", sp.diff(u, t).subs(t, 0))
report("W1", "u(t,0)", u.subs(x, 0))
report("W1", "u(t,1)", u.subs(x, 1))

# ---- W2: two-dimensional wave ---------------------------------------------------------
u = sp.sin(pi * x) * sp.sin(pi * y) * sp.cos(sp.sqrt(2) * pi * t)
report("W2", "u_tt - u_xx - u_yy", sp.diff(u, t, 2) - sp.diff(u, x, 2) - sp.diff(u, y, 2))
report("W2", "u(0,x,y) - sin(pi x) sin(pi y)", u.subs(t, 0) - sp.sin(pi * x) * sp.sin(pi * y))
report("W2", "u_t(0,x,y)", sp.diff(u, t).subs(t, 0))
for var in (x, y):
    for val in (-1, 1):
        report("W2", f"u on {{{var} = {val}}}", u.subs(var, val))

# ---- L2: two-dimensional Laplace, mixed data -------------------------------------------
u = x / (2 * pi) + sp.cos(2 * y) * sp.sinh(2 * x) / (2 * sp.sinh(2 * pi))
report("L2", "u_xx + u_yy", sp.diff(u, x, 2) + sp.diff(u, y, 2))
report("L2", "u(0,y)", u.subs(x, 0))
report("L2", "u(pi,y) - (1 + cos 2y)/2", u.subs(x, pi) - (1 + sp.cos(2 * y)) / 2)
report("L2", "u_y(x,0)", sp.diff(u, y).subs(y, 0))
report("L2", "u_y(x,pi)", sp.diff(u, y).subs(y, pi))
# corner compatibility of the data: d/dy of the Dirichlet data vanishes at y = 0, pi
g = (1 + sp.cos(2 * y)) / 2
report("L2", "g'(0), datum on x = pi", sp.diff(g, y).subs(y, 0))
report("L2", "g'(pi), datum on x = pi", sp.diff(g, y).subs(y, pi))

# ---- L3: generic term of the series and its coefficients -------------------------------
n = sp.symbols("n", integer=True, positive=True)
m = sp.symbols("m", integer=True, positive=True)
k = sp.sqrt(1 + n**2)
term = sp.sin(y) * sp.sin(n * z) * sp.sinh(k * x) / sp.sinh(k * pi)
report("L3", "Laplacian of the n-th term", sp.diff(term, x, 2) + sp.diff(term, y, 2) + sp.diff(term, z, 2))
report("L3", "n-th term on {x = 0}", term.subs(x, 0))
report("L3", "n-th term on {y = 0}", term.subs(y, 0))
report("L3", "n-th term on {y = pi}", term.subs(y, pi))
report("L3", "n-th term on {z = 0}", term.subs(z, 0))
report("L3", "n-th term on {z = pi}", term.subs(z, pi))
report("L3", "n-th term on {x = pi} - sin y sin nz", term.subs(x, pi) - sp.sin(y) * sp.sin(n * z))
# sine coefficients of cos z on (0, pi): (2/pi) int_0^pi cos z sin(nz) dz
b_even = (2 / pi) * sp.integrate(sp.cos(z) * sp.sin(2 * m * z), (z, 0, pi))
report("L3", "b_n - 4n/(pi(n^2-1)), n = 2m", b_even - 4 * (2 * m) / (pi * ((2 * m)**2 - 1)))
b_odd = (2 / pi) * sp.integrate(sp.cos(z) * sp.sin((2 * m + 1) * z), (z, 0, pi))
report("L3", "b_n, n = 2m+1 >= 3", b_odd)
report("L3", "b_1", (2 / pi) * sp.integrate(sp.cos(z) * sp.sin(z), (z, 0, pi)))

# ---- L3s: compatible datum sin y sin z -------------------------------------------------
u = sp.sin(y) * sp.sin(z) * sp.sinh(sp.sqrt(2) * x) / sp.sinh(sp.sqrt(2) * pi)
report("L3s", "u_xx + u_yy + u_zz", sp.diff(u, x, 2) + sp.diff(u, y, 2) + sp.diff(u, z, 2))
report("L3s", "u(pi,y,z) - sin y sin z", u.subs(x, pi) - sp.sin(y) * sp.sin(z))
for var, val in ((x, 0), (y, 0), (y, pi), (z, 0), (z, pi)):
    report("L3s", f"u on {{{var} = {val}}}", u.subs(var, val))

# ---- P1, P2 on the unit cube, d = 2, 3, 5, 10, 20 --------------------------------------
for d in (2, 3, 5, 10, 20):
    X = sp.symbols(f"x1:{d + 1}", real=True)
    u1 = sum(X[2 * j] * X[2 * j + 1] for j in range(d // 2))
    report("P1", f"-Laplacian u*, d = {d}", -sum(sp.diff(u1, v, 2) for v in X))
    u2 = sum(sp.cos(pi * v) for v in X) / sp.sqrt(d)
    report("P2", f"-Laplacian u* - pi^2 u*, d = {d}", -sum(sp.diff(u2, v, 2) for v in X) - pi**2 * u2)
    # normal derivative of P2 on every face (x_i = 0 and x_i = 1)
    nd = [sp.diff(u2, v).subs(v, s) for v in X for s in (0, 1)]
    report("P2", f"sum of |d_n u*| over the 2d faces, d = {d}", sum(sp.Abs(e) for e in nd))

lines.append("")
lines.append(f"ALL ZERO: {all_zero}")
text = "\n".join(lines) + "\n"
with open(OUT, "w") as fh:
    fh.write(text)
print(text)
