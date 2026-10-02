"""Figure s3_methods:fig:points -- the fixed interior point sets of the heat problem H.

Plot-only (no training).  Four panels on the space-time square (t, x) in [0,1]^2 with the
benchmark budget N_r = 256:

  grid-c   cell-centred 16 x 16 tensor grid          (pinnbench.samplers.GridSampler)
  grid-n   node-centred closed 16 x 16 tensor grid   (control of the verification code V-bench:
           verify_research_benchmark/vpinn.py, points(kind="nodegrid") -> linspace(0, 1, 16)
           per axis; re-implemented here in one line because pinnbench has no such sampler)
  random   256 i.i.d. uniform points, seed 0         (pinnbench.samplers.RandomSampler)
  Sobol    first 256 scrambled Sobol' points, seed 0 (pinnbench.samplers.SobolSampler)

The constraint points (100 on the initial line, 100 on each lateral boundary) are those of
pinnbench.problems.Heat1D and are identical in every panel.  `resample` redraws the `random`
set before every Adam step and `RAD` starts from the `Sobol` set and redraws it from the
network residual; neither is a fixed set, so neither has a panel.

Outputs
  data/s3_methods_points.json     coordinates of every plotted point + summary numbers
  figures/s3_methods_points.pdf   vector figure, designed at 6.3 in width

Run (from the repository root):  python code/s3_methods_points.py [preview.png]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parents[1]             # folder holding code/, data/, figures/
ROOT = ART if (ART / "research_benchmark").is_dir() else ART.parent   # repository root
sys.path.insert(0, str(ROOT / "research_benchmark"))

import matplotlib                                      # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from pinnbench.problems import make_problem            # noqa: E402
from pinnbench.samplers import make_sampler            # noqa: E402

N_R, SEED = 256, 0
M = round(N_R ** 0.5)                                  # 16 points per axis

P = make_problem("heat1d", device="cpu")


def from_sampler(name):
    return make_sampler(name, P, N_R, SEED).initial().detach().cpu().numpy().astype(np.float64)


def node_grid():
    a = np.linspace(0.0, 1.0, M)
    mesh = np.meshgrid(a, a, indexing="ij")
    U = np.stack([g.ravel() for g in mesh], 1)
    return P.lo + (P.hi - P.lo) * U


sets = {
    "grid-c": from_sampler("grid"),
    "grid-n": node_grid(),
    "random": from_sampler("random"),
    "Sobol": from_sampler("sobol"),
}
constraint = np.concatenate(P.blocks())                # 100 IC + 100 + 100 BC points, columns (t, x)

# ---- summary numbers quoted in the text / caption -------------------------------------
summary = {}
for name, X in sets.items():
    assert X.shape == (N_R, 2), (name, X.shape)
    t = X[:, 0]
    summary[name] = {
        "n_points": int(X.shape[0]),
        "min_t": float(t.min()),
        "n_points_with_t_below_1_over_32": int((t < 1.0 / 32 - 1e-9).sum()),
        "n_points_on_boundary_or_initial_line": int(
            ((np.abs(X[:, 0]) < 1e-9) | (np.abs(X[:, 1]) < 1e-9) | (np.abs(X[:, 1] - 1) < 1e-9)
             | (np.abs(X[:, 0] - 1) < 1e-9)).sum()),
    }

out = {
    "problem": "heat1d (H): columns are (t, x) in [0,1]^2",
    "n_r": N_R, "seed": SEED, "points_per_axis_grid": M,
    "sources": {
        "grid-c": "research_benchmark/pinnbench/samplers.py GridSampler",
        "grid-n": "verify_research_benchmark/vpinn.py points(kind='nodegrid') (re-implemented)",
        "random": "research_benchmark/pinnbench/samplers.py RandomSampler, seed 0",
        "Sobol": "research_benchmark/pinnbench/samplers.py SobolSampler, seed 0",
        "constraint": "research_benchmark/pinnbench/problems.py Heat1D.blocks()",
    },
    "summary": summary,
    "n_constraint_points": int(constraint.shape[0]),
    "points": {k: np.round(v, 7).tolist() for k, v in sets.items()},
    "constraint_points": np.round(constraint, 7).tolist(),
}
data_path = ART / "data" / "s3_methods_points.json"
data_path.parent.mkdir(parents=True, exist_ok=True)
data_path.write_text(json.dumps(out, indent=1))

# ---- figure ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import apply_style, INK, INK2 as MUTED, BLUE  # noqa: E402  (style shared by all figures)
BAND, FRAME = "#f2c9b8", "#c5ccd3"
apply_style(**{"axes.grid": False, "axes.linewidth": 0.5})
fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.0), sharex=True, sharey=True)
labels = {"grid-c": "grid-c (cell-centred)", "grid-n": "grid-n (node-centred)",
          "random": "random", "Sobol": "Sobol'"}
for ax, (name, X) in zip(axes, sets.items()):
    # strip 0 < t < 1/32 between the initial line and the first column of the cell-centred grid
    band = ax.axvspan(0.0, 1.0 / 32, color=BAND, lw=0, zorder=0)
    # outline of the space-time domain (the side t = 1 carries no condition)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=FRAME, lw=0.5, zorder=1, clip_on=False)
    c = ax.scatter(constraint[:, 0], constraint[:, 1], s=1.6, marker="o", color=MUTED,
                   linewidths=0, zorder=2, clip_on=False)
    r = ax.scatter(X[:, 0], X[:, 1], s=4.5, marker="o", color=BLUE, linewidths=0, zorder=3,
                   clip_on=False)
    ax.set_title(labels[name], pad=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.tick_params(length=2, width=0.5, pad=3)
    ax.set_xlabel("$t$", labelpad=1)
axes[0].set_ylabel("$x$", labelpad=2)
fig.legend([r, c, band],
           [r"interior points ($N_r=256$)",
            "initial and boundary points (300, same in every panel)",
            r"strip $0<t<1/32$"],
           loc="lower center", ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.6,
           bbox_to_anchor=(0.52, -0.01), markerscale=1.8, fontsize=7.5)
fig.subplots_adjust(left=0.065, right=0.985, top=0.89, bottom=0.31, wspace=0.16)
fig_path = ART / "figures" / "s3_methods_points.pdf"
fig.savefig(fig_path)
if len(sys.argv) > 1:                                  # optional raster preview: python ... <file.png>
    fig.savefig(sys.argv[1], dpi=220)
print("wrote", data_path)
print("wrote", fig_path)
print(json.dumps(summary, indent=1))
