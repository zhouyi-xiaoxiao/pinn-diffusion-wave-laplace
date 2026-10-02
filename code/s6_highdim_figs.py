"""Figures of Section 6 (dimension scaling), regenerated as vector PDF from stored results.
Plot-only: no training, no timing.  Adapted from research_highdim/scripts/analyze.py.

Inputs (relative to the root of the repository)
    research_highdim/results/runs*.jsonl              per-run records with validation curves
    research_highdim/results/cost_vs_d.json           primary cost benchmark
    verify_research_highdim/results/v_bench.json      cost benchmark of the verification code V-dim
    data/closed_forms.json                    closed-form trivial-predictor floors
    data/s6_highdim_long_d20.jsonl            16 000 iterations at d = 20, Section 6.3 (optional; used if present)
Outputs
    figures/s6_highdim_error.pdf
    figures/s6_highdim_cost.pdf
    figures/s6_highdim_curves.pdf
"""
import glob
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

ART = Path(__file__).resolve().parents[1]             # folder holding code/, data/, figures/
ROOT = ART if (ART / "research_benchmark").is_dir() else ART.parent   # repository root
HD = ROOT / "research_highdim" / "results"
FIG = ART / "figures"
FIG.mkdir(exist_ok=True)


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


R = []
for f in sorted(glob.glob(str(HD / "runs*.jsonl"))):
    R += load_jsonl(f)
N5_PATH = ART / "data" / "s6_highdim_long_d20.jsonl"
LONG20 = load_jsonl(N5_PATH) if N5_PATH.exists() else []
FLOORS = json.load(open(ART / "data" / "closed_forms.json"))["floors"]
COST = json.load(open(HD / "cost_vs_d.json"))
VB = json.load(open(ROOT / "verify_research_highdim" / "results" / "v_bench.json"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import apply_style, panel_label, BLUE, ORANGE, DARK, GREY, PNAME  # noqa: E402  (style shared by all figures)
C = {"ritz": BLUE, "pinn": ORANGE}
MK = {"ritz": "o", "pinn": "s"}
LAB = {"ritz": "Deep Ritz", "pinn": "PINN"}
FS = 8
apply_style(**{"lines.linewidth": 1.3})
# figures are saved at exactly their figsize (6.3 in wide = text width); margins are set by hand


def floor(problem, d, kind="affine"):
    return FLOORS[str(d)][("P1_" if problem == "laplace" else "P2_") + kind]


def xdims(ax, dd):
    ax.set_xscale("log")
    ax.set_xticks(dd)
    ax.set_xticklabels([str(d) for d in dd])
    ax.xaxis.set_minor_locator(NullLocator())


def logy(ax, plain=False):
    """log y-axis with minor ticks at 2 and 5; plain=True labels all ticks as plain numbers."""
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(LogLocator(subs=[2.0, 5.0]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if plain:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


main = [r for r in R if r["phase"] == "main" and r["iters"] == 4000]
groups = {}
for r in main:
    groups.setdefault((r["problem"], r["method"], r["d"]), []).append(r)
DS = sorted({r["d"] for r in main})
assert DS == [2, 3, 5, 10, 20] and all(len(v) == 3 for v in groups.values())
long10 = [r for r in R if r["phase"] == "long"]            # P1, d = 10, 16 000 iterations
longruns = {}
for r in long10 + LONG20:
    longruns.setdefault((r["problem"], r["method"], r["d"]), []).append(r)

# ------------------------------------------------------------------ Fig. error vs d
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.7), sharex=True)
for j, p in enumerate(["laplace", "poisson"]):
    for i, (metric, ylab) in enumerate([("rel_l2", r"relative $L^2$ error $\varepsilon_2$"),
                                        ("rel_h1semi", r"relative $H^1$-seminorm error $\varepsilon_{H^1}$")]):
        ax = axes[i, j]
        for m in ["ritz", "pinn"]:
            off = 0.955 if m == "ritz" else 1.045
            xs, ys = [], []
            for d in DS:
                v = [r[metric] for r in groups[(p, m, d)]]
                ax.scatter([d * off] * len(v), v, s=9, color=C[m], alpha=0.5, lw=0, zorder=2)
                xs.append(d * off)
                ys.append(statistics.mean(v))
            ax.plot(xs, ys, marker=MK[m], ms=4.5, color=C[m], zorder=3)
            for d in DS:                                       # 16 000-iteration runs
                rs = longruns.get((p, m, d), [])
                if rs:
                    v = [r[metric] for r in rs]
                    ax.scatter([d * off] * len(v), v, s=9, facecolors="none", edgecolors=C[m], lw=0.6, zorder=2)
                    ax.plot([d * off], [statistics.mean(v)], marker=MK[m], ms=5.5, mfc="white", mec=C[m],
                            mew=1.1, ls="none", zorder=4)
        if metric == "rel_l2":
            ax.plot(DS, [floor(p, d) for d in DS], ls="--", color=GREY, lw=1.0, zorder=1)
        xdims(ax, DS)
        logy(ax, plain=(metric == "rel_h1semi"))      # H1 row: label 0.02, 0.05, 0.1, ... (short range)
        if i == 0:
            ax.set_title(f"{PNAME[p]}", fontsize=FS)
        if i == 1:
            ax.set_xlabel("dimension $d$")
        if j == 0:
            ax.set_ylabel(ylab)
handles = [Line2D([], [], color=C["pinn"], marker="s", ms=4.5, label="PINN, 4000 iterations (mean)"),
           Line2D([], [], color=C["ritz"], marker="o", ms=4.5, label="Deep Ritz, 4000 iterations (mean)"),
           Line2D([], [], color=GREY, marker="o", ms=3, alpha=0.6, ls="none", label="individual seeds"),
           Line2D([], [], color=GREY, marker="s", ms=5.5, mfc="white", mew=1.1, ls="none",
                  label="16 000 iterations (mean)"),
           Line2D([], [], color=GREY, ls="--", lw=1.0, label=r"best affine fit of $u^*$ (floor)")]
fig.subplots_adjust(left=0.095, right=0.99, top=0.955, bottom=0.185, wspace=0.19, hspace=0.07)
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
           columnspacing=1.5, handlelength=1.8, fontsize=7.5)
fig.savefig(FIG / "s6_highdim_error.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Fig. cost
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
ax = axes[0]
VAR = {"ritz": (BLUE, "o", "ritz"), "pinn_forward": (ORANGE, "s", "fwd"), "pinn_nested": (DARK, "^", "nested")}
for m, (col, mk, vkey) in VAR.items():
    cs = sorted([c for c in COST if c["method"] == m], key=lambda c: c["d"])
    ax.errorbar([c["d"] for c in cs], [c["cpu_ms_median"] for c in cs],
                yerr=[[c["cpu_ms_median"] - c["cpu_ms_min"] for c in cs],
                      [c["cpu_ms_max"] - c["cpu_ms_median"] for c in cs]],
                marker=mk, ms=4, color=col, ls="-", lw=1.1, capsize=1.5, elinewidth=0.7, zorder=3)
    dv = sorted(int(k) for k in VB)
    ax.plot(dv, [VB[str(d)]["cpu_ms"][vkey] for d in dv], marker=mk, ms=4, mfc="white", mew=0.9, color=col,
            ls=":", lw=1.0, zorder=2)
dd = sorted({c["d"] for c in COST})
xdims(ax, dd)
logy(ax, plain=True)
ax.set_xlabel("dimension $d$")
ax.set_ylabel("CPU time per iteration (ms)")
leg_a = [Line2D([], [], color=DARK, marker="^", ms=4, label="PINN, nested"),
         Line2D([], [], color=ORANGE, marker="s", ms=4, label="PINN, forward"),
         Line2D([], [], color=BLUE, marker="o", ms=4, label="Deep Ritz"),
         Line2D([], [], color=GREY, ls="-", lw=1.1, label="primary"),
         Line2D([], [], color=GREY, ls=":", marker="o", mfc="white", ms=3.5, lw=1.0, label="second measurement")]
panel_label(ax, "(a) cost per iteration")

RATIO = next(c for c in COST if c["d"] == 10 and c["method"] == "pinn_forward")["ratio_to_ritz_median"]
for j, p in enumerate(["laplace", "poisson"]):
    ax = axes[1 + j]
    series = [("main", "ritz", "-", 1.0), ("equaltime", "ritz", "--", 1.0), ("main", "pinn", "-", RATIO)]
    for ph, m, ls, scale in series:
        rs = [r for r in R if r["phase"] == ph and r["problem"] == p and r["method"] == m and r["d"] == 10
              and (ph != "main" or r["iters"] == 4000)]
        assert len(rs) == 3
        for r in rs:
            ax.plot([c["it"] * scale / 1e3 for c in r["curve"]], [c["val_rel_l2"] for c in r["curve"]],
                    ls=ls, color=C[m], alpha=0.9, lw=0.9)
    logy(ax, plain=True)
    ax.set_xlim(0, 10.4)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    if j == 0:
        ax.set_ylabel(r"validation relative $L^2$ error")
    panel_label(ax, f"({'bc'[j]}) {PNAME[p]}, $d=10$")
fig.subplots_adjust(left=0.075, right=0.985, top=0.925, bottom=0.36, wspace=0.42)
pa = axes[0].get_position()
fig.legend(handles=leg_a, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(pa.x0 - 0.045, 0.215), ncol=2,
           handlelength=2.0, columnspacing=1.0, labelspacing=0.3, borderaxespad=0.0)
xmid = 0.5 * (axes[1].get_position().x0 + axes[2].get_position().x1)
fig.text(xmid, 0.245, r"compute, in units of $10^3$ Deep Ritz iterations", ha="center", va="center", fontsize=FS)
fig.legend(handles=[Line2D([], [], color=ORANGE, ls="-", lw=0.9, label="PINN, 4000 iterations"),
                    Line2D([], [], color=BLUE, ls="-", lw=0.9, label="Deep Ritz, 4000 iterations"),
                    Line2D([], [], color=BLUE, ls="--", lw=0.9, label="Deep Ritz, equal compute")],
           frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(xmid, 0.2), ncol=2,
           handlelength=2.0, columnspacing=1.2, labelspacing=0.3)
fig.savefig(FIG / "s6_highdim_cost.pdf")
plt.close(fig)
print(f"cost figure: PINN iterations scaled by the primary paired cost ratio at d = 10: {RATIO:.3f}")

# ------------------------------------------------------------------ Fig. curves
fig, axes = plt.subplots(3, 5, figsize=(6.3, 5.6))
fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.07, wspace=0.22, hspace=0.62)


def curves(ax, rs_by_method, p, d, iters):
    for m in ["ritz", "pinn"]:
        for r in rs_by_method.get(m, []):
            ax.plot([c["it"] / 1e3 for c in r["curve"]], [c["val_rel_l2"] for c in r["curve"]], color=C[m],
                    lw=0.8, alpha=0.9)
    ax.axhline(floor(p, d), ls="--", color=GREY, lw=0.8, zorder=1)
    ax.grid(axis="x", visible=False)
    for frac in (0.5, 0.75):                                  # learning-rate drops
        ax.axvline(frac * iters / 1e3, ls=":", color="#3d3d3a", lw=0.7, zorder=1)
    logy(ax)
    ax.set_xlim(0, iters / 1e3)
    ax.set_xticks([0, iters // 2000, iters // 1000])
    ax.set_title(f"{PNAME[p]}, $d={d}$", fontsize=FS, pad=3)


for i, p in enumerate(["laplace", "poisson"]):
    for j, d in enumerate(DS):
        ax = axes[i, j]
        curves(ax, {m: groups[(p, m, d)] for m in ["ritz", "pinn"]}, p, d, 4000)
        ax.set_ylim((2e-3, 1.5) if p == "laplace" else (8e-4, 1.5))
        if j > 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"validation error")
cells = [("laplace", 10), ("laplace", 20), ("poisson", 20)]
k = 0
for p, d in cells:
    rs = {m: longruns.get((p, m, d), []) for m in ["ritz", "pinn"]}
    if not (rs["ritz"] or rs["pinn"]):
        continue
    ax = axes[2, k]
    curves(ax, rs, p, d, 16000)
    ax.set_ylim(2e-3, 1.5)
    ax.set_xlabel(r"$10^3$ iterations")
    if k > 0:
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel(r"validation error")
    k += 1
for kk in range(k, 5):
    axes[2, kk].axis("off")
for i in range(2):
    for j in range(5):
        axes[i, j].set_xlabel(r"$10^3$ iterations", labelpad=1)
axes[2, 3].legend(handles=[Line2D([], [], color=ORANGE, lw=0.9, label="PINN (3 seeds)"),
                           Line2D([], [], color=BLUE, lw=0.9, label="Deep Ritz (3 seeds)"),
                           Line2D([], [], color=GREY, ls="--", lw=0.8, label=r"best affine fit of $u^*$"),
                           Line2D([], [], color="#3d3d3a", ls=":", lw=0.7, label="learning-rate drops")],
                  frameon=False, loc="center left", fontsize=7.5, bbox_to_anchor=(0.0, 0.5),
                  title="bottom row: 16 000 iterations", title_fontsize=7.5, alignment="left")
fig.savefig(FIG / "s6_highdim_curves.pdf")
plt.close(fig)
print("LONG20 records used:", len(LONG20))
