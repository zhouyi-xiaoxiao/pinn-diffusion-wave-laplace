"""Sizes of the held-out evaluation sets of the five low-dimensional benchmark problems and
their overlap with the (strategy-independent) boundary / initial training points.

No training.  For each problem of research_benchmark/pinnbench/problems.py the script counts
  * the evaluation points (Problem.eval_points()),
  * the boundary / initial-condition training points (Problem.blocks()),
  * the evaluation points that coincide with one of those training points (distance < 1e-9),
  * the evaluation points that lie on the boundary of the box (including t = 0 and t = 1),
  * the smallest distance from an evaluation point to a face of the box,
  * the interior collocation points of the cell-centred grid strategy (`grid-c`,
    pinnbench.samplers.GridSampler, at the benchmark budget N_r) that coincide with an
    evaluation point.
Random and Sobol' collocation points are not counted: they are continuous random variables
and coincide with a grid node with probability zero.
Output: ../data/s2_problems_evalsets.json

Usage:  python s2_problems_evalsets.py
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)  # repository root
sys.path.insert(0, os.path.join(ROOT, "research_benchmark"))
from pinnbench.problems import make_problem  # noqa: E402

# interior budgets of the benchmark (research_benchmark/scripts/run_benchmark.py)
N_R = {"heat1d": 256, "wave1d": 1024, "wave2d": 512, "laplace2d": 256, "laplace3d": 512}


def grid_c(P, n):
    """Cell-centred tensor grid with n points, as pinnbench.samplers.GridSampler.initial()."""
    m = round(n ** (1.0 / P.d))
    assert m ** P.d == n
    a = (np.arange(m) + 0.5) / m
    mesh = np.meshgrid(*([a] * P.d), indexing="ij")
    return P.lo + (P.hi - P.lo) * np.stack([g.ravel() for g in mesh], 1)


out = {}
for name in ("heat1d", "wave1d", "wave2d", "laplace2d", "laplace3d"):
    P = make_problem(name)
    E = P.eval_points()
    G = grid_c(P, N_R[name])
    dist_g, _ = cKDTree(E).query(G)
    C = np.concatenate(P.blocks())
    dist, _ = cKDTree(C).query(E)
    on_bd = np.isclose(E, P.lo).any(1) | np.isclose(E, P.hi).any(1)
    gap = float(np.minimum(E - P.lo, P.hi - E).min())
    out[name] = {
        "n_eval": int(E.shape[0]),
        "n_constraint_train": int(C.shape[0]),
        "n_eval_coinciding_with_constraint_train": int((dist < 1e-9).sum()),
        "fraction_coinciding": float((dist < 1e-9).mean()),
        "n_eval_on_box_boundary": int(on_bd.sum()),
        "min_distance_eval_to_box_face": gap,
        "min_distance_over_pi": gap / np.pi,
        "n_r": N_R[name],
        "n_grid_c_collocation_coinciding_with_eval": int((dist_g < 1e-9).sum()),
        "fraction_of_eval_that_are_grid_c_collocation": float((dist_g < 1e-9).sum() / E.shape[0]),
    }

path = os.path.join(HERE, "..", "data", "s2_problems_evalsets.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
