# -- added for the public repository: absolute paths of the development machine are replaced by
# -- paths relative to the repository root, through the helper _repo_path defined here.
import os as _os


def _repo_path(rel=""):
    d = _os.path.dirname(_os.path.abspath(__file__))
    while not (_os.path.isdir(_os.path.join(d, "research_benchmark")) and _os.path.isdir(_os.path.join(d, "research_highdim"))):
        p = _os.path.dirname(d)
        if p == d:
            raise RuntimeError("repository root not found")
        d = p
    return _os.path.join(d, rel) if rel else d
# -- end of the added lines

"""Aggregate results/runs.jsonl -> CSV/JSON tables + figures.
Outputs:
  results/runs.csv            one row per run (no curves)
  results/summary.csv/.json   per (problem, method, d): mean, sample std, min, max over seeds
  results/sweep.csv           penalty-weight selection runs
  figs/fig_error_vs_d.png     rel L2 and rel H1-seminorm vs d, both problems
  figs/fig_cost_vs_d.png      ms/iteration vs d (benchmark) + error-vs-wallclock at d=10
  figs/fig_curves_d10.png     validation rel L2 vs iteration, d=10
  figs/fig_sweep.png          validation error vs penalty weight (d=5)
"""
import sys, os, json, math, statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator


def logy(ax):
    """log y-axis with labelled 2 and 5 minor ticks (so narrow ranges still have readable ticks)."""
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(LogLocator(subs=[2.0, 5.0]))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:.0e}".replace("e-0", "e-")))
    ax.tick_params(axis="y", which="minor", labelsize=7)


def xdims(ax, dd):
    ax.set_xscale("log"); ax.set_xticks(dd); ax.set_xticklabels([str(d) for d in dd])
    ax.xaxis.set_minor_locator(NullLocator())
sys.path.insert(0, os.path.dirname(__file__))


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]

ROOT = _repo_path("research_highdim")
import glob
R = []
for _f in sorted(glob.glob(f"{ROOT}/results/runs*.jsonl")):
    R += load_jsonl(_f)
os.makedirs(f"{ROOT}/figs", exist_ok=True)
C = {"ritz": "#2a78d6", "pinn": "#eb6834", "base": "#8a8985"}
MK = {"ritz": "o", "pinn": "s"}
LAB = {"ritz": "Deep Ritz", "pinn": "PINN"}
PLAB = {"laplace": r"P1: $-\Delta u=0$, $u^*=\sum_k x_{2k-1}x_{2k}$",
        "poisson": r"P2: $-\Delta u=\pi^2u^*$, $u^*=d^{-1/2}\sum_i\cos\pi x_i$"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

SCAL = ["rel_l2", "rel_l2_centered", "rel_h1semi", "max_abs", "bd_rel_l2", "val_rel_l2", "train_time_s",
        "ms_per_iter", "cpu_time_s", "cpu_ms_per_iter", "lap_impl", "final_loss", "final_loss_int", "final_loss_bd", "n_params", "base_const_rel_l2"]
COLS = ["phase", "problem", "method", "d", "seed", "w", "iters"] + SCAL
with open(f"{ROOT}/results/runs.csv", "w") as f:
    f.write(",".join(COLS) + "\n")
    for r in R:
        f.write(",".join(str(r.get(c, "")) for c in COLS) + "\n")

main = [r for r in R if r["phase"] == "main" and r["iters"] == 4000]
groups = {}
for r in main:
    groups.setdefault((r["problem"], r["method"], r["d"]), []).append(r)
summary = []
for (p, m, d), rs in sorted(groups.items()):
    row = dict(problem=p, method=m, d=d, w=rs[0]["w"], n_seeds=len(rs), seeds=[r["seed"] for r in rs])
    v = [r["rel_l2"] for r in rs]
    row["rel_l2_geomean"] = math.exp(statistics.mean(map(math.log, v)))
    for k in ["rel_l2", "rel_l2_centered", "rel_h1semi", "max_abs", "bd_rel_l2", "cpu_time_s", "cpu_ms_per_iter",
              "base_const_rel_l2", "n_params"]:
        v = [r[k] for r in rs]
        row[k + "_mean"] = statistics.mean(v)
        row[k + "_std"] = statistics.stdev(v) if len(v) > 1 else float("nan")
        row[k + "_min"] = min(v); row[k + "_max"] = max(v)
    summary.append(row)
json.dump(summary, open(f"{ROOT}/results/summary.json", "w"), indent=1)
keys = [k for k in summary[0] if k != "seeds"] if summary else []
with open(f"{ROOT}/results/summary.csv", "w") as f:
    f.write(",".join(keys) + "\n")
    for row in summary:
        f.write(",".join((f"{row[k]:.4e}" if isinstance(row[k], float) else str(row[k])) for k in keys) + "\n")

sweep = [r for r in R if r["phase"] == "sweep"]
with open(f"{ROOT}/results/sweep.csv", "w") as f:
    f.write("problem,method,d,seed,w,val_rel_l2,test_rel_l2,bd_rel_l2\n")
    for r in sweep:
        f.write(f"{r['problem']},{r['method']},{r['d']},{r['seed']},{r['w']},{r['val_rel_l2']:.4e},{r['rel_l2']:.4e},{r['bd_rel_l2']:.4e}\n")


# trivial-predictor floors on the test set: best constant and best affine (least squares) fit of u*
floors = {}
_fp = f"{ROOT}/results/floors.json"
_need = {(p, d) for p in ["laplace", "poisson"] for d in {r["d"] for r in main}}
if os.path.exists(_fp):
    floors = {(f["problem"], f["d"]): f for f in json.load(open(_fp))}
if not _need <= set(floors):          # (re)compute only if missing: needs torch + the fixed test sets
    import torch
    from hd_core import exact, fixed_sets
    for p, d in sorted(_need):
        x = fixed_sets(d)["test"].double(); u = exact(p, x)
        A = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)
        coef = torch.linalg.lstsq(A, u).solution
        nrm = (u ** 2).sum().sqrt()
        floors[(p, d)] = dict(problem=p, d=d, const_rel_l2=((u - u.mean()) ** 2).sum().sqrt().div(nrm).item(),
                              affine_rel_l2=((A @ coef - u) ** 2).sum().sqrt().div(nrm).item(),
                              u_rms=(u ** 2).mean().sqrt().item())
json.dump(list(floors.values()), open(f"{ROOT}/results/floors.json", "w"), indent=1)
for row in summary:
    row["affine_rel_l2"] = floors[(row["problem"], row["d"])]["affine_rel_l2"]
    row["const_rel_l2"] = floors[(row["problem"], row["d"])]["const_rel_l2"]
json.dump(summary, open(f"{ROOT}/results/summary.json", "w"), indent=1)
keys = [k for k in summary[0] if k != "seeds"] if summary else []
with open(f"{ROOT}/results/summary.csv", "w") as f:
    f.write(",".join(keys) + "\n")
    for row in summary:
        f.write(",".join((f"{row[k]:.4e}" if isinstance(row[k], float) else str(row[k])) for k in keys) + "\n")


def S(p, m, d):
    return next((s for s in summary if s["problem"] == p and s["method"] == m and s["d"] == d), None)


# ---------------------------------------------------------------- fig 1: error vs d
DS = sorted({r["d"] for r in main})
if main:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True)
    for j, p in enumerate(["laplace", "poisson"]):
        for i, (metric, ylab) in enumerate([("rel_l2", r"relative $L^2$ error (test)"),
                                            ("rel_h1semi", r"relative $H^1$-seminorm error (test)")]):
            ax = axes[i, j]
            for m in ["ritz", "pinn"]:
                xs, ys = [], []
                for d in DS:
                    rs = groups.get((p, m, d), [])
                    if not rs:
                        continue
                    v = [r[metric] for r in rs]
                    off = 0.96 if m == "ritz" else 1.04
                    ax.scatter([d * off] * len(v), v, s=14, color=C[m], alpha=0.45, lw=0, zorder=2)
                    xs.append(d * off); ys.append(math.exp(statistics.mean(map(math.log, v))))
                ax.plot(xs, ys, marker=MK[m], ms=6, color=C[m], label=f"{LAB[m]} (geo-mean of seeds)", zorder=3)
            if metric == "rel_l2":
                ax.plot(DS, [floors[(p, d)]["affine_rel_l2"] for d in DS], ls="--", color=C["base"], lw=1.2,
                        label="best affine fit of $u^*$ (floor)")
            xdims(ax, DS); logy(ax)
            if i == 0:
                ax.set_title(PLAB[p], fontsize=9)
            if i == 1:
                ax.set_xlabel("dimension $d$")
            if j == 0:
                ax.set_ylabel(ylab)
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[1, 0].legend(h, l, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Error vs dimension at a fixed budget (4000 Adam its, 1024+1024 fresh MC points/it); small dots = individual seeds",
                 fontsize=8.5)
    fig.savefig(f"{ROOT}/figs/fig_error_vs_d.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 2: cost
cost_path = f"{ROOT}/results/cost_vs_d.json"
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
ax = axes[0]
CV = {"ritz": ("Deep Ritz (1 reverse pass)", C["ritz"], "o", "-"),
      "pinn_forward": ("PINN, forward-Laplacian (used here)", C["pinn"], "s", "-"),
      "pinn_nested": ("PINN, nested reverse-mode", C["pinn"], "^", "--")}
if os.path.exists(cost_path):
    cost = json.load(open(cost_path))
    for m, (lab, col, mk, ls) in CV.items():
        cs = [c for c in cost if c["method"] == m]
        ax.errorbar([c["d"] for c in cs], [c["cpu_ms_median"] for c in cs],
                    yerr=[[c["cpu_ms_median"] - c["cpu_ms_min"] for c in cs],
                          [c["cpu_ms_max"] - c["cpu_ms_median"] for c in cs]],
                    marker=mk, ms=5, color=col, ls=ls, label=lab, capsize=2,
                    markerfacecolor="white" if m == "pinn_nested" else col)
    dd = sorted({c["d"] for c in cost}); xdims(ax, dd); logy(ax)
    ax.set_xlabel("dimension $d$"); ax.set_ylabel("CPU ms per training iteration (1 thread)")
    ax.set_title("Cost per iteration (median, min-max of 7×10 its)", fontsize=9); ax.legend(frameon=False, fontsize=7)
for j, p in enumerate(["laplace", "poisson"]):
    ax = axes[1 + j]
    series = [("main", "ritz", "-", "Deep Ritz, 4000 its"), ("equaltime", "ritz", "--", "Deep Ritz, equal CPU"),
              ("main", "pinn", "-", "PINN, 4000 its")]
    for ph, m, ls, lab in series:
        rs = [r for r in R if r["phase"] == ph and r["problem"] == p and r["method"] == m and r["d"] == 10
              and (ph != "main" or r["iters"] == 4000)]
        for k, r in enumerate(rs):
            cpi = r["cpu_ms_per_iter"] / 1e3
            ax.plot([c["it"] * cpi for c in r["curve"]], [c["val_rel_l2"] for c in r["curve"]], ls=ls, color=C[m],
                    alpha=0.85, lw=1.1, label=lab if k == 0 else None)
    logy(ax)
    ax.set_xlabel("training CPU time (s)")
    if j == 0:
        ax.set_ylabel(r"validation rel. $L^2$ error")
    ax.set_title(f"{'P1' if p == 'laplace' else 'P2'}, d = 10: error vs compute (3 seeds)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{ROOT}/figs/fig_cost_vs_d.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: curves at each d
if main:
    fig, axes = plt.subplots(2, len(DS), figsize=(2.3 * len(DS), 4.6), sharey="row")
    for i, p in enumerate(["laplace", "poisson"]):
        for j, d in enumerate(DS):
            ax = axes[i, j]
            for m in ["ritz", "pinn"]:
                for k, r in enumerate(groups.get((p, m, d), [])):
                    ax.plot([c["it"] for c in r["curve"]], [c["val_rel_l2"] for c in r["curve"]], color=C[m],
                            lw=1.0, alpha=0.85, label=LAB[m] if k == 0 else None)
            logy(ax); ax.set_title(f"{'P1' if p == 'laplace' else 'P2'}, d = {d}", fontsize=9)
            if i == 1:
                ax.set_xlabel("iteration")
            if j == 0:
                ax.set_ylabel(r"val. rel. $L^2$ error")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{ROOT}/figs/fig_curves.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 4: sweep
if sweep:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for j, p in enumerate(["laplace", "poisson"]):
        ax = axes[j]
        for m in ["ritz", "pinn"]:
            rs = sorted([r for r in sweep if r["problem"] == p and r["method"] == m], key=lambda r: r["w"])
            if rs:
                ax.plot([r["w"] for r in rs], [r["val_rel_l2"] for r in rs], marker=MK[m], color=C[m], label=LAB[m])
        ax.set_xscale("log"); logy(ax); ax.set_xlabel(r"boundary penalty weight ($\lambda$ PINN, $\beta$ Ritz)")
        ax.set_title(f"{'P1' if p == 'laplace' else 'P2'}, d = 5, selection seed 100", fontsize=9)
        if j == 0:
            ax.set_ylabel(r"validation rel. $L^2$ error")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{ROOT}/figs/fig_sweep.png"); plt.close(fig)

# ---------------------------------------------------------------- console table
print(f"{'prob':8s} {'meth':5s} {'d':>3s} {'n':>2s} {'relL2 mean':>11s} {'std':>9s} {'min':>9s} {'max':>9s} "
      f"{'centered':>9s} {'H1semi':>9s} {'bdL2':>9s} {'cpums':>6s} {'const':>6s} {'affine':>6s}")
for s in summary:
    print(f"{s['problem']:8s} {s['method']:5s} {s['d']:3d} {s['n_seeds']:2d} {s['rel_l2_mean']:11.3e} {s['rel_l2_std']:9.2e} "
          f"{s['rel_l2_min']:9.2e} {s['rel_l2_max']:9.2e} {s['rel_l2_centered_mean']:9.2e} {s['rel_h1semi_mean']:9.2e} "
          f"{s['bd_rel_l2_mean']:9.2e} {s['cpu_ms_per_iter_mean']:6.1f} {s['const_rel_l2']:6.3f} {s['affine_rel_l2']:6.3f}")

# ---------------------------------------------------------------- extensions: long budget, equal compute
ext = []
for ph in ["long", "equaltime"]:
    g2 = {}
    for r in R:
        if r["phase"] == ph:
            g2.setdefault((r["problem"], r["method"], r["d"], r["iters"]), []).append(r)
    for (p, m, d, it), rs in sorted(g2.items()):
        v = [r["rel_l2"] for r in rs]; h = [r["rel_h1semi"] for r in rs]; c = [r["cpu_time_s"] for r in rs]
        ext.append(dict(phase=ph, problem=p, method=m, d=d, iters=it, n_seeds=len(rs), rel_l2=v,
                        rel_l2_mean=statistics.mean(v), rel_l2_std=statistics.stdev(v) if len(v) > 1 else float("nan"),
                        rel_h1semi_mean=statistics.mean(h), cpu_time_s_mean=statistics.mean(c)))
json.dump(ext, open(f"{ROOT}/results/extensions.json", "w"), indent=1)
for e in ext:
    print(f"[{e['phase']}] {e['problem']} {e['method']} d={e['d']} iters={e['iters']} n={e['n_seeds']}: rel_l2 mean={e['rel_l2_mean']:.3e} "
          f"std={e['rel_l2_std']:.2e} seeds={['%.3e' % x for x in e['rel_l2']]} H1={e['rel_h1semi_mean']:.3e} cpu={e['cpu_time_s_mean']:.0f}s")
