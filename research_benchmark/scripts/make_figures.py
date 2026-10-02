"""Publication figures (vector PDF + PNG) from the saved results. No training here.

fig1_sampler_errors      rel-L2 per collocation strategy and optimiser arm, all seeds shown
fig2_training_curves     held-out rel-L2 vs iteration: Adam vs Adam->L-BFGS (median + IQR over all models)
fig3_sampler_curves      held-out rel-L2 vs iteration per collocation strategy (Adam->L-BFGS arm)
fig4_accuracy_cost       accuracy vs wall-clock: PINN arms vs finite-difference convergence curves
fig5_budget_sweep        rel-L2 vs number of collocation points (if results/budget_sweep.json exists)
fig6_error_maps          pointwise |error| fields, seed 0 (heat1d, laplace2d)

Colour: categorical palette (blue, orange, green, amber, pink)
(blue, orange, aqua, yellow, magenta in fixed order; marker shape is the secondary encoding).
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
PROBLEMS = ["heat1d", "wave1d", "wave2d", "laplace2d", "laplace3d"]
TITLES = {"heat1d": "Heat 1D", "wave1d": "Wave 1D (c = 2, two modes)", "wave2d": "Wave 2D",
          "laplace2d": "Laplace 2D", "laplace3d": "Laplace 3D"}
SAMPLERS = ["grid", "random", "resample", "sobol", "rad"]
SLABEL = {"grid": "grid", "random": "random\n(fixed)", "resample": "random\n(resampled)", "sobol": "Sobol", "rad": "RAD"}
SLABEL1 = {k: v.replace("\n", " ") for k, v in SLABEL.items()}
COL = {"grid": "#2a78d6", "random": "#eb6834", "resample": "#1baf7a", "sobol": "#eda100", "rad": "#e87ba4"}
MARK = {"grid": "o", "random": "s", "resample": "^", "sobol": "D", "rad": "v"}
ARM_COL = {"adam": "#2a78d6", "adam_lbfgs": "#eb6834", "fd": "#52514e"}
ARM_LAB = {"adam": "Adam", "adam_lbfgs": "Adam → L-BFGS"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "pdf.fonttype": 42,
    "lines.linewidth": 1.6, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def load_runs():
    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*_s*-*.json"))):
        runs += json.load(open(f))["runs"]
    return runs


def geo(v):
    lg = np.log10(np.asarray(v))
    return 10 ** lg.mean(), 10 ** (lg.mean() - lg.std(ddof=1)), 10 ** (lg.mean() + lg.std(ddof=1))


def fig1(runs):
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.9), sharey=False)
    for ax, p in zip(axes, PROBLEMS):
        for k, s in enumerate(SAMPLERS):
            for j, arm in enumerate(["adam", "adam_lbfgs"]):
                v = np.array([r[arm]["rel_l2"] for r in runs if r["problem"] == p and r["sampler"] == s])
                if len(v) == 0:
                    continue
                x0 = k + (-0.17 if j == 0 else 0.17)
                jit = np.linspace(-0.09, 0.09, len(v))
                ax.scatter(x0 + jit, v, s=9, color=ARM_COL[arm], alpha=0.45, linewidths=0, zorder=2)
                g, lo, hi = geo(v)
                ax.errorbar([x0], [g], yerr=[[g - lo], [hi - g]], fmt="o" if j == 0 else "s", ms=4.5, color=ARM_COL[arm],
                            mec="white", mew=0.8, elinewidth=1.0, capsize=0, zorder=3,
                            label=ARM_LAB[arm] if k == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(range(5))
        ax.set_xticklabels([SLABEL1[s] for s in SAMPLERS], fontsize=6.5, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title(TITLES[p])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("relative $L^2$ error (held-out grid)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), handletextpad=0.3)
    fig.suptitle("Collocation strategy × optimiser: every seed (small dots) and geometric mean ± 1 geometric s.d.", fontsize=9, y=1.09)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig1_sampler_errors")


def curve_stats(runs, p, arm, samplers):
    cs = [r[arm]["curve"] for r in runs if r["problem"] == p and r["sampler"] in samplers]
    if not cs:
        return None
    n = min(len(c) for c in cs)
    it = np.array([c["it"] for c in cs[0][:n]])
    E = np.array([[c["test_rel_l2"] for c in cc[:n]] for cc in cs])
    return it, np.median(E, 0), np.percentile(E, 25, 0), np.percentile(E, 75, 0)


def fig2(runs):
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.7))
    for ax, p in zip(axes, PROBLEMS):
        branch = [r["branch_at"] for r in runs if r["problem"] == p]
        for arm in ["adam", "adam_lbfgs"]:
            st = curve_stats(runs, p, arm, SAMPLERS)
            if st is None:
                continue
            it, med, q1, q3 = st
            ax.fill_between(it, q1, q3, color=ARM_COL[arm], alpha=0.18, linewidth=0)
            ax.plot(it, med, color=ARM_COL[arm], label=ARM_LAB[arm])
        if branch:
            ax.axvline(branch[0], color=INK2, lw=0.7, ls=(0, (3, 3)))
        ax.set_yscale("log"); ax.set_title(TITLES[p]); ax.set_xlabel("iteration")
    axes[0].set_ylabel("relative $L^2$ error (held-out)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Training curves: median and inter-quartile band over all strategies × seeds; dashed line = switch to L-BFGS", fontsize=9, y=1.09)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig2_training_curves")


def fig3(runs):
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.7))
    for ax, p in zip(axes, PROBLEMS):
        for s in SAMPLERS:
            st = curve_stats(runs, p, "adam_lbfgs", [s])
            if st is None:
                continue
            it, med, q1, q3 = st
            ax.plot(it, med, color=COL[s], lw=1.4, label=SLABEL1[s], marker=MARK[s], markevery=(len(it) // 6 or 1), ms=3.5,
                    mec="white", mew=0.5)
        ax.set_yscale("log"); ax.set_title(TITLES[p]); ax.set_xlabel("iteration")
    axes[0].set_ylabel("median relative $L^2$ error (held-out)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Adam → L-BFGS arm: median held-out error per collocation strategy (matched point budgets)", fontsize=9, y=1.09)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig3_sampler_curves")


def fig4(runs):
    cl = json.load(open(os.path.join(ROOT, "results", "classical.json")))
    tm = json.load(open(os.path.join(ROOT, "results", "timing_single.json")))
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.9))
    for ax, p in zip(axes, PROBLEMS):
        c = [r for r in cl if r["problem"] == p]
        t = np.array([r["time_s_best_of_3"] for r in c]); e = np.array([r["rel_l2"] for r in c])
        ax.plot(t, e, color=ARM_COL["fd"], marker="o", ms=3.5, mec="white", mew=0.5, label="finite differences")
        ax.annotate(f"n={c[0]['n']}", (t[0], e[0]), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=INK2)
        ax.annotate(f"n={c[-1]['n']}", (t[-1], e[-1]), textcoords="offset points", xytext=(5, -2), fontsize=6.5, color=INK2)
        for arm, key, mk in [("adam", "est_run_s_adam", "o"), ("adam_lbfgs", "est_run_s_adam_lbfgs", "s")]:
            v = np.array([r[arm]["rel_l2"] for r in runs if r["problem"] == p])
            if len(v) == 0 or p not in tm:
                continue
            g, lo, hi = geo(v)
            ax.errorbar([tm[p][key]], [g], yerr=[[g - v.min()], [v.max() - g]], fmt=mk, ms=5, color=ARM_COL[arm], mec="white",
                        mew=0.8, elinewidth=1.0, label="PINN, " + ARM_LAB[arm])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(TITLES[p]); ax.set_xlabel("wall-clock per solve (s)")
    axes[0].set_ylabel("relative $L^2$ error (held-out grid)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Accuracy vs cost: PINN (geometric mean, bar = min–max over all strategies × seeds) against 2nd-order finite differences", fontsize=9, y=1.09)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig4_accuracy_cost")



def fig5():
    path = os.path.join(ROOT, "results", "budget_sweep.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    probs = sorted({r["problem"] for r in d})
    fig, axes = plt.subplots(1, len(probs), figsize=(3.4 * len(probs), 2.9), squeeze=False)
    for ax, p in zip(axes[0], probs):
        for s in ["grid", "random", "sobol"]:
            ns = sorted({r["n_r"] for r in d if r["problem"] == p and r["sampler"] == s})
            if not ns:
                continue
            G = [geo([r["adam_lbfgs"]["rel_l2"] for r in d if r["problem"] == p and r["sampler"] == s and r["n_r"] == n]) for n in ns]
            g = np.array([x[0] for x in G]); lo = np.array([x[1] for x in G]); hi = np.array([x[2] for x in G])
            ax.fill_between(ns, lo, hi, color=COL[s], alpha=0.15, linewidth=0)
            ax.plot(ns, g, color=COL[s], marker=MARK[s], ms=4.5, mec="white", mew=0.6, label=SLABEL1[s])
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(TITLES[p]); ax.set_xlabel("interior collocation points $N_r$")
    axes[0][0].set_ylabel("relative $L^2$ error (held-out)")
    axes[0][0].legend(loc="upper right")
    fig.suptitle("Collocation budget sweep, Adam → L-BFGS (geometric mean ± 1 geometric s.d., 5 seeds)", fontsize=9, y=1.04)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig5_budget_sweep")


def fig6():
    from pinnbench.problems import make_problem
    todo = [p for p in ["heat1d", "laplace2d"] if glob.glob(os.path.join(ROOT, "results", "runs", f"{p}_pred_seed*.npz"))]
    if not todo:
        return
    show = ["grid", "random", "sobol", "rad"]
    fig, axes = plt.subplots(len(todo), len(show), figsize=(2.6 * len(show), 2.6 * len(todo)), squeeze=False,
                             gridspec_kw={"hspace": 0.6, "wspace": 0.25})
    for i, p in enumerate(todo):
        z = np.load(sorted(glob.glob(os.path.join(ROOT, "results", "runs", f"{p}_pred_seed*.npz")))[0])
        P = make_problem(p)
        X = P.eval_points(); u = P.exact(X)
        n = int(round(np.sqrt(X.shape[0])))
        names = list(z["samplers"])
        errs = {s: np.abs(z["adam_lbfgs"][names.index(s)] - u).reshape(n, n) for s in show}
        vmax = max(e.max() for e in errs.values())
        for j, s in enumerate(show):
            ax = axes[i][j]
            im = ax.pcolormesh(X[:, 0].reshape(n, n), X[:, 1].reshape(n, n), errs[s], cmap="Blues", vmin=0, vmax=vmax, shading="auto", rasterized=True)
            ax.set_title(f"{TITLES[p]} — {SLABEL1[s]}\nmax {errs[s].max():.1e}", fontsize=7.5)
            ax.set_xlabel("t" if p == "heat1d" else "x")
            if j == 0:
                ax.set_ylabel("x" if p == "heat1d" else "y")
            ax.grid(False)
        cb = fig.colorbar(im, ax=axes[i].tolist(), fraction=0.025, pad=0.02)
        cb.set_label("|u_PINN − u_exact|")
    fig.suptitle("Pointwise absolute error, seed 0, Adam → L-BFGS (shared colour scale per row)", fontsize=9, y=0.99)
    save(fig, "fig6_error_maps")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    runs = load_runs()
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    if runs:
        if "1" in which: fig1(runs)
        if "2" in which: fig2(runs)
        if "3" in which: fig3(runs)
        if "4" in which and os.path.exists(os.path.join(ROOT, "results", "timing_single.json")): fig4(runs)
    if "5" in which: fig5()
    if "6" in which: fig6()
