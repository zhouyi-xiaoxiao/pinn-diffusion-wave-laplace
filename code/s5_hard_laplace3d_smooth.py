"""Compatible-data control on Lap3 (Section 5.2): is the 3-5 % error on the 3-D Laplace problem L3 due to the
discontinuity of its boundary data, or to the third dimension?

Problem L3s (compatible data): Lap u = 0 on (0,pi)^3, u(pi,y,z) = sin y sin z, u = 0 on the
other five faces; exact u* = sin y sin z sinh(sqrt(2) x) / sinh(sqrt(2) pi).
Problem L3 (benchmark, discontinuous data): as above with datum sin y cos z (series solution).

Protocol -- identical for both problems
----------------------------------------------------------------
  network     2 x 100 tanh (10 601 parameters), PyTorch default initialisation, float32
  points      N_r = 512 scrambled-Sobol interior points (fixed); 12 x 12 open grid per face
  optimiser   arm "adam": 3000 Adam steps (lr 1e-3); arm "adam_lbfgs": the same first 1500 Adam
              steps, then 1500 L-BFGS iterations (memory 50)
  driver      pinnbench.core.run_batched, UNMODIFIED (the stacked multi-seed driver that produced
              the benchmark tables), here on the CPU with the five seeds stacked
  seeds       0, 1, 2, 3, 4 (a seed fixes the initial weights and the Sobol scrambling, so the
              two problems are compared with the same initial network and the same points)
  evaluation  40^3 cell-centred interior grid (64 000 held-out points)

L3s is defined here as a subclass of pinnbench.problems.Laplace3D that changes only the datum
on the face x = pi and the exact solution, and is registered under the name "laplace3d_smooth".
L3 is re-run with the same driver/device as a control, so that the L3-versus-L3s comparison is
not confounded by the device (the benchmark's L3 runs were on the Apple GPU backend).

Also stored: where the L3 error sits (share of the squared error carried by the evaluation
points closest to the two singular edges {x = pi, z = 0} and {x = pi, z = pi}).
Wall-clock times are NOT stored; no timing statement of the article uses this experiment.

Usage:  python s5_hard_laplace3d_smooth.py        # -> data/s5_hard_laplace3d_smooth.json
"""
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)  # repository root
sys.path.insert(0, os.path.join(ROOT, "research_benchmark"))
from pinnbench import core, problems  # noqa: E402

PI = math.pi
OUT = os.path.join(ART, "data", "s5_hard_laplace3d_smooth.json")
SEEDS = [0, 1, 2, 3, 4]
N_R, N_ADAM, BRANCH_AT, SAMPLER = 512, 3000, 1500, "sobol"


class Laplace3DSmooth(problems.Laplace3D):
    """L3s: datum sin y sin z on x = pi (vanishes on the edges of that face: compatible)."""
    name = "laplace3d_smooth"

    def terms(self, B):
        t = []
        for k, nm in enumerate(self.face_names):
            tgt = np.sin(B[k][:, 1]) * np.sin(B[k][:, 2]) if nm == "xpi" else np.zeros(B[k].shape[0])
            t.append((f"bc_{nm}", k, "u", tgt))
        return t

    def exact(self, X):
        X = np.asarray(X, dtype=np.float64)
        r2 = math.sqrt(2.0)
        # sinh(r2 x)/sinh(r2 pi) written with decaying exponentials
        ratio = np.exp(r2 * (X[:, 0] - PI)) * (-np.expm1(-2 * r2 * X[:, 0])) / (-np.expm1(-2 * r2 * PI))
        return np.sin(X[:, 1]) * np.sin(X[:, 2]) * ratio


problems.PROBLEMS["laplace3d_smooth"] = Laplace3DSmooth


def check_exact():
    """Finite-difference Laplacian of the L3s exact solution and its boundary values."""
    P = Laplace3DSmooth()
    rng = np.random.default_rng(0)
    X = rng.uniform(0.2, PI - 0.2, size=(2000, 3))
    h = 1e-3
    lap = np.zeros(len(X))
    for i in range(3):
        e = np.zeros(3); e[i] = h
        lap += (P.exact(X + e) - 2 * P.exact(X) + P.exact(X - e)) / h ** 2
    blocks = P.blocks()
    bc = 0.0
    for k, nm in enumerate(P.face_names):
        tgt = np.sin(blocks[k][:, 1]) * np.sin(blocks[k][:, 2]) if nm == "xpi" else 0.0
        bc = max(bc, float(np.abs(P.exact(blocks[k]) - tgt).max()))
    return {"max_abs_fd_laplacian_h1e-3": float(np.abs(lap).max()), "max_abs_boundary_mismatch": bc}


def run(problem_name):
    specs = [(SAMPLER, s) for s in SEEDS]
    res, _timing, preds = core.run_batched(problem_name, specs, N_R, N_ADAM, BRANCH_AT, device="cpu",
                                           log_every=100, verbose=True, return_preds=True)
    P = problems.make_problem(problem_name)
    X = P.eval_points()
    u = P.exact(X)
    # distance of each evaluation point to the nearer of the two edges {x = pi, z = 0}, {x = pi, z = pi}
    dist = np.sqrt((PI - X[:, 0]) ** 2 + np.minimum(X[:, 2], PI - X[:, 2]) ** 2)
    near = dist <= np.quantile(dist, 0.05)
    rows = []
    for i, r in enumerate(res):
        row = {"seed": r["seed"]}
        for arm in ("adam", "adam_lbfgs"):
            e = preds[arm][i].astype(np.float64) - u
            row[arm] = {"rel_l2": r[arm]["rel_l2"], "linf": r[arm]["linf"], "final_loss": r[arm]["final_loss"],
                        "share_sq_error_nearest_5pct_to_singular_edges": float((e[near] ** 2).sum() / (e ** 2).sum()),
                        "rel_l2_excluding_those_points": float(np.linalg.norm(e[~near]) / np.linalg.norm(u[~near])),
                        "linf_excluding_those_points": float(np.abs(e[~near]).max()),
                        "curve": [{k: v for k, v in c.items()} for c in r[arm]["curve"]]}
        rows.append(row)
    out = {"problem": problem_name, "rows": rows, "norm_exact_l2_rms": float(np.sqrt((u ** 2).mean())),
           "max_abs_exact": float(np.abs(u).max())}
    for arm in ("adam", "adam_lbfgs"):
        for key in ("rel_l2", "linf"):
            v = np.array([row[arm][key] for row in rows])
            out[f"{arm}_{key}_mean"], out[f"{arm}_{key}_std"] = float(v.mean()), float(v.std(ddof=1))
            out[f"{arm}_{key}_min"], out[f"{arm}_{key}_max"] = float(v.min()), float(v.max())
            out[f"{arm}_{key}_geomean"] = float(np.exp(np.log(v).mean()))
    return out


if __name__ == "__main__":
    torch.set_num_threads(2)
    out = {"note": "Compatible-data control on Lap3 (Section 5.2). Same network, points, optimiser and driver for L3s (compatible data) and L3 "
                   "(discontinuous data, control re-run). pinnbench.core.run_batched on CPU, float32, 5 seeds "
                   "stacked. Errors on the 40^3 cell-centred grid. No timings stored.",
           "config": {"n_r": N_R, "n_adam": N_ADAM, "branch_at": BRANCH_AT, "sampler": SAMPLER, "seeds": SEEDS,
                      "device": "cpu", "torch_threads": 2, "torch_version": torch.__version__},
           "exact_check_L3s": check_exact()}
    print(out["exact_check_L3s"], flush=True)
    out["L3s"] = run("laplace3d_smooth")
    out["L3"] = run("laplace3d")
    a, b = out["L3s"], out["L3"]
    ratio = np.array([r1["adam_lbfgs"]["rel_l2"] / r0["adam_lbfgs"]["rel_l2"] for r1, r0 in zip(b["rows"], a["rows"])])
    out["paired_ratio_L3_over_L3s_adam_lbfgs_rel_l2"] = ratio.tolist()
    out["paired_ratio_geomean"] = float(np.exp(np.log(ratio).mean()))
    json.dump(out, open(OUT, "w"), indent=1)
    for nm, o in (("L3s", a), ("L3", b)):
        print(nm, "Adam rel_l2 %.3e +- %.3e | Adam->L-BFGS rel_l2 %.3e +- %.3e [%.3e, %.3e], linf %.3e +- %.3e" % (
            o["adam_rel_l2_mean"], o["adam_rel_l2_std"], o["adam_lbfgs_rel_l2_mean"], o["adam_lbfgs_rel_l2_std"],
            o["adam_lbfgs_rel_l2_min"], o["adam_lbfgs_rel_l2_max"], o["adam_lbfgs_linf_mean"], o["adam_lbfgs_linf_std"]))
    print("paired ratio L3/L3s:", ratio, "geo-mean", out["paired_ratio_geomean"])
