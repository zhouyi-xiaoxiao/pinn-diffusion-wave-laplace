"""Section 5: the seven-point finite-difference scheme on the compatible-data problem Lap3s, next to Lap3.

Question (Section 5.4): the observed order of the scheme on Lap3 between n = 80 and n = 160, two grids on which
the 40^3 evaluation points are nodes, is 1.89.  Is that due to the discontinuous datum?  The same solver is run
with the datum sin y sin z of Lap3s, which is compatible with the zero data on the adjacent faces.

Solver: the discrete-sine-transform solver of research_benchmark/pinnbench/classical.py (laplace3d_fd), copied
here with the datum on the face x = pi as an argument; for the datum of Lap3 the copy is checked to return the
array of the original, bit for bit.  Evaluation as in the benchmark (pinnbench.classical.evaluate): cubic
interpolation for n = 20, 40, nodal values for n = 80, 160.  No training, no timing.

Output: data/s5_hard_fd_lap3s.json
Usage:  python code/s5_hard_fd_lap3s.py
"""
import json
import os
import sys

import numpy as np
import scipy.fft

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)
sys.path.insert(0, os.path.join(ROOT, "research_benchmark"))
from pinnbench.classical import evaluate, laplace3d_fd          # noqa: E402
from pinnbench.problems import make_problem                     # noqa: E402

PI = np.pi


def fd(n, datum_z):
    x = np.linspace(0, PI, n + 1)
    j = np.arange(1, n)
    g = np.sin(x[1:-1])[:, None] * datum_z(x[1:-1])[None, :]    # datum on x = pi at the interior (y, z) nodes
    ghat = scipy.fft.dstn(g, type=1)
    lam = 4 * np.sin(j * PI / (2 * n)) ** 2
    theta = np.arccosh(1 + 0.5 * (lam[:, None] + lam[None, :]))
    i = np.arange(n + 1)[:, None, None]
    ratio = np.exp(theta * (i - n)) * (-np.expm1(-2 * theta * i)) / (-np.expm1(-2 * theta * n))
    U = np.zeros((n + 1, n + 1, n + 1))
    U[:, 1:-1, 1:-1] = scipy.fft.idstn(ratio * ghat[None], type=1, axes=(1, 2))
    return (x, x, x), U


P = make_problem("laplace3d")
X = P.eval_points()                                              # 40^3 cell centres
exact = {"Lap3": P.exact(X),
         "Lap3s": np.sin(X[:, 1]) * np.sin(X[:, 2]) * np.sinh(np.sqrt(2) * X[:, 0]) / np.sinh(np.sqrt(2) * PI)}
datum = {"Lap3": np.cos, "Lap3s": np.sin}
out = {"note": "Seven-point scheme, relative L2 and L-infinity error on the 40^3 cell centres; cubic interpolation "
               "for n = 20, 40, nodal values for n = 80, 160.", "problems": {}}
for name in ("Lap3", "Lap3s"):
    rows, prev = [], None
    for n in (20, 40, 80, 160):
        axes, U = fd(n, datum[name])
        if name == "Lap3":
            assert np.array_equal(U, laplace3d_fd(n)[1]), "copy of the solver differs from pinnbench.classical.laplace3d_fd"
        u = evaluate(axes, U, X)
        e = float(np.linalg.norm(u - exact[name]) / np.linalg.norm(exact[name]))
        rows.append({"n": n, "rel_l2": e, "linf": float(np.abs(u - exact[name]).max()),
                     "observed_order": None if prev is None else float(np.log2(prev / e))})
        prev = e
        print(name, rows[-1], flush=True)
    out["problems"][name] = rows
out["ratio_Lap3_over_Lap3s_rel_l2"] = {str(a["n"]): a["rel_l2"] / b["rel_l2"]
                                       for a, b in zip(out["problems"]["Lap3"], out["problems"]["Lap3s"])}
print("ratio Lap3 / Lap3s:", out["ratio_Lap3_over_Lap3s_rel_l2"])
json.dump(out, open(os.path.join(ART, "data", "s5_hard_fd_lap3s.json"), "w"), indent=1)
