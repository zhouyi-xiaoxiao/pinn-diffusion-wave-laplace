"""Section 5, problem W1: relative L2 error of the predictor that keeps only the slow mode.

u*(t, x) = sin(pi x) cos(2 pi t) + 0.5 sin(4 pi x) cos(8 pi t)   on (t, x) in [0,1]^2.
The two modes are L2-orthogonal on [0,1]^2 with squared norms 1/4 and 1/16, so the predictor
v = sin(pi x) cos(2 pi t) has relative L2 error sqrt((1/16) / (5/16)) = 1/sqrt(5) = 0.4472...
This script evaluates the same quantity on the 201 x 201 held-out grid used for W1 and lists
the range of the 50 benchmark runs next to it. No training.

Output: data/s5_hard_wave1d_floor.json
"""
import csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)  # repository root
sys.path.insert(0, os.path.join(ROOT, "research_benchmark"))
from pinnbench.problems import make_problem  # noqa: E402

P = make_problem("wave1d")
X = P.eval_points()
u = P.exact(X)
v = np.sin(np.pi * X[:, 1]) * np.cos(2 * np.pi * X[:, 0])
runs = [r for r in csv.DictReader(open(os.path.join(ROOT, "research_benchmark", "results", "benchmark_runs.csv")))
        if r["problem"] == "wave1d"]
e = {arm: [float(r["rel_l2"]) for r in runs if r["arm"] == arm] for arm in ("adam", "adam_lbfgs")}
out = {"slow_mode_only_rel_l2_continuous": 1 / math.sqrt(5),
       "slow_mode_only_rel_l2_on_eval_grid": float(np.linalg.norm(v - u) / np.linalg.norm(u)),
       "slow_mode_only_linf_on_eval_grid": float(np.abs(v - u).max()),
       "benchmark_runs": {arm: {"n": len(x), "min": min(x), "max": max(x)} for arm, x in e.items()},
       "note": "This script does not project the trained networks onto the two modes; this is a reference level only. The projections are in data/s5_hard_wave1d_long.json."}
json.dump(out, open(os.path.join(ART, "data", "s5_hard_wave1d_floor.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
