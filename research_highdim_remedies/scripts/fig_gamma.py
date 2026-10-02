"""Descriptive figure (no decision uses it): gamma(t), the share of the non-affine part of u* captured on the
validation set, logged every 50 iterations; d = 20, 4000 iterations, median over seeds 10-14 with min-max band.
figures/gamma_curves.pdf/.png. Source: results/runs_main20.jsonl."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "runs_main20.jsonl")) if l.strip()]
COL = {"plain": "0.25", "presolve": "#e07b00", "lift3c": "#1f6fb4", "lift3u": "#7b3fa0", "rep3": "#4a9a4a"}
PID = {"laplace": "P1", "poisson": "P2", "ridge": "P3", "cospair": "P4"}
plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True, sharey=True)
for i, method in enumerate(("ritz", "pinn")):
    for j, problem in enumerate(PID):
        ax = axes[i, j]
        for arm in COL:
            cs = [r["curve"] for r in R if r["problem"] == problem and r["method"] == method and r["arm"] == arm]
            if not cs:
                continue
            it = np.array([c["it"] for c in cs[0]])
            g = np.array([[c["gamma"] for c in cv] for cv in cs])
            ax.plot(it, np.median(g, 0), color=COL[arm], lw=1.3, label=f"{arm} ({len(cs)})")
            ax.fill_between(it, g.min(0), g.max(0), color=COL[arm], alpha=0.15, lw=0)
        ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
        ax.set_title(f"{'Deep Ritz' if method == 'ritz' else 'PINN'}, {PID[problem]}")
        ax.set_ylim(-0.1, 1.1)
        if i == 1: ax.set_xlabel("iteration")
        if j == 0: ax.set_ylabel("gamma (non-affine share captured)")
        ax.legend(fontsize=6.5, loc="lower right", frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "figures", f"gamma_curves.{ext}"), dpi=200)
print("written figures/gamma_curves.pdf/.png")
