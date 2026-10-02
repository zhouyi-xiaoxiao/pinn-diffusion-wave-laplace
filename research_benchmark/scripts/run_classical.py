"""Finite-difference baselines: convergence study on the same held-out grids as the PINNs.
Output: results/classical.json, results/classical.csv"""
import sys, os, time, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pinnbench.problems import make_problem
from pinnbench.classical import SOLVERS, evaluate
from pinnbench.core import error_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for name, (solver, sizes) in SOLVERS.items():
    P = make_problem(name)
    eX = P.eval_points()
    eu = P.exact(eX)
    prev = None
    for n in sizes:
        best = np.inf
        for rep in range(3):
            t0 = time.perf_counter()
            axes, U = solver(n)
            best = min(best, time.perf_counter() - t0)
        m = error_metrics(evaluate(axes, U, eX), eu)
        order = None if prev is None else float(np.log2(prev / m["rel_l2"]))
        prev = m["rel_l2"]
        row = {"problem": name, "n": n, "grid_values": int(U.size), "time_s_best_of_3": best, **m, "observed_order_rel_l2": order}
        rows.append(row)
        print(json.dumps(row), flush=True)
json.dump(rows, open(os.path.join(ROOT, "results", "classical.json"), "w"), indent=1)
with open(os.path.join(ROOT, "results", "classical.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
