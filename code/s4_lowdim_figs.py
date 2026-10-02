"""Section 4 figures, regenerated at print size from the stored benchmark results.

Plot-only (no training); adapted from research_benchmark/scripts/make_figures.py.

  figures/s4_lowdim_samplers.pdf   relative L2 error per collocation strategy and optimiser arm,
                                   every seed and geometric mean +- 1 geometric s.d.
                                   <- research_benchmark/results/runs/*_s*-*.json
  figures/s4_lowdim_curves.pdf     held-out relative L2 error vs iteration, Adam vs Adam -> L-BFGS,
                                   median, inter-quartile band and min-max band over strategies x seeds
                                   <- research_benchmark/results/runs/*_s*-*.json
  figures/s4_lowdim_budget.pdf     budget sweep on the heat problem (N_r = 64, 256, 1024)
                                   <- research_benchmark/results/budget_sweep.json
  figures/s4_lowdim_errmaps.pdf    pointwise absolute error, seed 0, Adam -> L-BFGS (H and L2)
                                   <- research_benchmark/results/runs/{heat1d,laplace2d}_pred_seed0.npz
Style: code/figstyle.py (shared by all figures).

Usage:  python s4_lowdim_figs.py [samplers] [curves] [budget] [errmaps]
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s4_lowdim_style import (apply_style, panel_label as _plabel, ARM_COL, ARM_LAB, ARM_MARK, BENCH, FIGDIR, INK, INK2, PLONG, PNAME,
                             PROBLEMS, SLAB, STRATS, STRAT_COL, STRAT_MARK)

sys.path.insert(0, BENCH)
apply_style()
W = 6.3                                      # text width of the article (inches)
# decimal y-ticks for the panels whose errors span less than about one decade
NARROW = {"wave2d": [0.03, 0.05, 0.1, 0.2, 0.3], "laplace3d": [0.03, 0.05, 0.1, 0.2, 0.5, 1.0],
          "wave1d": [0.42, 0.45, 0.48, 0.51]}
NARROW_CURVES = {"wave2d": [0.05, 0.1, 0.2, 0.5, 1.0], "laplace3d": [0.05, 0.1, 0.2, 0.5, 1.0],
                 "wave1d": [0.4, 0.5, 0.7, 1.0]}


def load_runs():
    runs = []
    for f in sorted(glob.glob(os.path.join(BENCH, "results", "runs", "*_s*-*.json"))):
        runs += json.load(open(f))["runs"]
    return runs


def geo(v):
    lg = np.log10(np.asarray(v, float))
    return 10 ** lg.mean(), 10 ** (lg.mean() - lg.std(ddof=1)), 10 ** (lg.mean() + lg.std(ddof=1))


def save(fig, name):
    path = os.path.join(FIGDIR, name + ".pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)   # tight box: no panel title is cut at the edge
    plt.close(fig)
    print("wrote", path)


def panel_label(ax, k, p, runs):
    rr = [r for r in runs if r["problem"] == p]
    n_seeds = len({r["seed"] for r in rr})
    _plabel(ax, f"({'abcde'[k]}) {PNAME[p]}, $N_r={rr[0]['n_r']}$, {n_seeds} seeds")


def plain_log_ticks(ax, ticks):
    """Decimal tick labels on a log axis that spans less than about one decade."""
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))


# ----------------------------------------------------------------------------------------
def fig_samplers(runs):
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.75))
    flat = axes.ravel()
    for k, p in enumerate(PROBLEMS):
        ax = flat[k]
        for j, s in enumerate(STRATS):
            for a, arm in enumerate(["adam", "adam_lbfgs"]):
                v = np.array([r[arm]["rel_l2"] for r in sorted((r for r in runs if r["problem"] == p and r["sampler"] == s),
                                                                key=lambda r: r["seed"])])
                x0 = j + (-0.2 if a == 0 else 0.2)
                ax.scatter(x0 + np.linspace(-0.11, 0.11, len(v)), v, s=7, color=ARM_COL[arm], alpha=0.45, linewidths=0, zorder=2)
                g, lo, hi = geo(v)
                ax.errorbar([x0], [g], yerr=[[g - lo], [hi - g]], fmt=ARM_MARK[arm], ms=4.2, color=ARM_COL[arm],
                            mec="white", mew=0.7, elinewidth=1.0, capsize=0, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(range(len(STRATS)))
        ax.set_xticklabels([SLAB[s] for s in STRATS], rotation=32, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(STRATS) - 0.4)
        ax.grid(axis="x", visible=False)
        if p in NARROW:
            plain_log_ticks(ax, NARROW[p])
        panel_label(ax, k, p, runs)
        if k % 3 == 0:
            ax.set_ylabel(r"relative $L^{2}$ error $\varepsilon_{2}$")
    lg = flat[5]
    lg.axis("off")
    handles = [Line2D([], [], marker=ARM_MARK[a], ms=5, color=ARM_COL[a], mec="white", mew=0.7, lw=1.0, label=ARM_LAB[a])
               for a in ["adam", "adam_lbfgs"]]
    handles.append(Line2D([], [], marker="o", ms=2.8, color=INK2, alpha=0.5, lw=0, label="one seed"))
    handles.append(Line2D([], [], marker="o", ms=4.5, color=INK2, mec="white", mew=0.7, lw=1.0,
                          label="geometric mean\n\u00b1 1 geometric s.d."))
    lg.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.55), handlelength=1.6, labelspacing=0.9)
    fig.tight_layout(w_pad=0.7, h_pad=1.0)
    save(fig, "s4_lowdim_samplers")


# ----------------------------------------------------------------------------------------
def curve_matrix(runs, p, arm):
    cs = [r[arm]["curve"] for r in runs if r["problem"] == p]
    its = [c["it"] for c in cs[0]]
    assert all([c["it"] for c in cc] == its for cc in cs), "curves are not logged at the same iterations"
    return np.array(its), np.array([[c["test_rel_l2"] for c in cc] for cc in cs])


def fig_curves(runs):
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.1))
    flat = axes.ravel()
    for k, p in enumerate(PROBLEMS):
        ax = flat[k]
        branch = {r["branch_at"] for r in runs if r["problem"] == p}
        assert len(branch) == 1
        for arm in ["adam", "adam_lbfgs"]:
            it, E = curve_matrix(runs, p, arm)
            if arm == "adam_lbfgs":                       # identical to the Adam arm up to the switch
                keep = it >= min(branch)
                it, E = it[keep], E[:, keep]
            ax.fill_between(it, E.min(0), E.max(0), color=ARM_COL[arm], alpha=0.10, linewidth=0, zorder=1)
            ax.fill_between(it, np.percentile(E, 25, 0), np.percentile(E, 75, 0), color=ARM_COL[arm], alpha=0.28, linewidth=0, zorder=2)
            ax.plot(it, np.median(E, 0), color=ARM_COL[arm], zorder=3)
        ax.axvline(min(branch), color=INK2, lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax.set_yscale("log")
        ax.set_xlim(0, 3000)
        ax.set_xticks([0, 1000, 2000, 3000])
        if p in NARROW_CURVES:
            plain_log_ticks(ax, NARROW_CURVES[p])
        panel_label(ax, k, p, runs)
        if k >= 2:
            ax.set_xlabel("iteration")
        if k % 3 == 0:
            ax.set_ylabel(r"relative $L^{2}$ error $\varepsilon_{2}$")
    lg = flat[5]
    lg.axis("off")
    handles = [Line2D([], [], color=ARM_COL[a], lw=1.6, label=ARM_LAB[a] + ", median") for a in ["adam", "adam_lbfgs"]]
    handles += [Patch(facecolor=INK2, alpha=0.35, linewidth=0, label="inter-quartile range"),
                Patch(facecolor=INK2, alpha=0.13, linewidth=0, label="minimum to maximum"),
                Line2D([], [], color=INK2, lw=0.7, ls=(0, (3, 3)), label="switch to L-BFGS")]
    lg.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.55), handlelength=1.8, labelspacing=0.9)
    fig.tight_layout(w_pad=0.7, h_pad=1.0)
    save(fig, "s4_lowdim_curves")


# ----------------------------------------------------------------------------------------
def fig_budget():
    d = json.load(open(os.path.join(BENCH, "results", "budget_sweep.json")))
    d = [r for r in d if r["problem"] == "heat1d"]
    ns = sorted({r["n_r"] for r in d})
    seeds = sorted({r["seed"] for r in d})
    off = {"grid": 0.86, "random": 1.0, "sobol": 1.16}       # multiplicative x-offsets (log axis)

    def val(s, n):
        rr = sorted((r for r in d if r["sampler"] == s and r["n_r"] == n), key=lambda r: r["seed"])
        assert [r["seed"] for r in rr] == seeds
        return np.array([r["adam_lbfgs"]["rel_l2"] for r in rr])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.75))
    for s in ["grid", "random", "sobol"]:
        g = []
        for n in ns:
            v = val(s, n)
            a1.scatter(n * off[s] * np.ones(len(v)), v, s=9, color=STRAT_COL[s], alpha=0.5, linewidths=0, zorder=2)
            g.append(geo(v)[0])
        a1.plot(np.array(ns) * off[s], g, color=STRAT_COL[s], marker=STRAT_MARK[s], ms=4.8, mec="white", mew=0.7, zorder=3,
                label=SLAB[s])
    a1.set_xscale("log", base=2); a1.set_yscale("log")
    a1.set_xticks(ns); a1.set_xticklabels([str(n) for n in ns]); a1.minorticks_off()
    a1.tick_params(axis="y", which="minor", left=True)
    a1.set_xlim(44, 1500)
    a1.set_xlabel(r"interior collocation points $N_r$")
    a1.set_ylabel(r"relative $L^{2}$ error $\varepsilon_{2}$")
    _plabel(a1, "(a) error, every seed and geometric mean")
    a1.legend(loc="upper right", handlelength=1.8)

    for s in ["grid", "sobol"]:
        g = []
        for n in ns:
            q = val(s, n) / val("random", n)
            a2.scatter(n * off[s] * np.ones(len(q)), q, s=9, color=STRAT_COL[s], alpha=0.5, linewidths=0, zorder=2)
            g.append(10 ** np.log10(q).mean())
        a2.plot(np.array(ns) * off[s], g, color=STRAT_COL[s], marker=STRAT_MARK[s], ms=4.8, mec="white", mew=0.7, zorder=3,
                label=SLAB[s] + " / random")
    a2.axhline(1.0, color=INK2, lw=0.7, ls=(0, (3, 3)), zorder=1)
    a2.set_xscale("log", base=2); a2.set_yscale("log")
    a2.set_xticks(ns); a2.set_xticklabels([str(n) for n in ns]); a2.minorticks_off()
    a2.set_xlim(44, 1500)
    a2.set_xlabel(r"interior collocation points $N_r$")
    a2.set_ylabel(r"paired ratio of $\varepsilon_{2}$ to random")
    _plabel(a2, "(b) ratio to random, same seed")
    a2.legend(loc="lower right", handlelength=1.8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "s4_lowdim_budget")


# ----------------------------------------------------------------------------------------
def fig_errmaps():
    from pinnbench.problems import make_problem

    show = ["grid", "random", "sobol", "rad"]
    rows = [("heat1d", "$t$", "$x$"), ("laplace2d", "$x$", "$y$")]
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.75), gridspec_kw={"hspace": 0.62, "wspace": 0.16,
                                                                     "left": 0.075, "right": 0.885, "top": 0.92, "bottom": 0.115})
    info = {}
    for i, (p, xl, yl) in enumerate(rows):
        z = np.load(os.path.join(BENCH, "results", "runs", f"{p}_pred_seed0.npz"))
        P = make_problem(p)
        X = P.eval_points(); u = P.exact(X)
        n = int(round(np.sqrt(X.shape[0])))
        names = [str(s) for s in z["samplers"]]
        errs = {s: np.abs(z["adam_lbfgs"][names.index(s)].astype(np.float64) - u).reshape(n, n) for s in show}
        vmax = max(e.max() for e in errs.values())
        ex10 = int(np.floor(np.log10(vmax)))                 # colour bar in units of 10^ex10
        for j, s in enumerate(show):
            ax = axes[i][j]
            im = ax.pcolormesh(X[:, 0].reshape(n, n), X[:, 1].reshape(n, n), errs[s] / 10.0 ** ex10, cmap="Blues", vmin=0,
                               vmax=vmax / 10.0 ** ex10, shading="auto", rasterized=True)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.4)
            e = errs[s]
            imax = np.unravel_index(e.argmax(), e.shape)
            info[f"{p}:{s}"] = {"max_abs_error": float(e.max()), "argmax_coord0": float(X[:, 0].reshape(n, n)[imax]),
                                "argmax_coord1": float(X[:, 1].reshape(n, n)[imax])}
            m, ex = f"{e.max():.1e}".split("e")
            ax.set_title(f"{PNAME[p]}, {SLAB[s]}\nmax {m}×10$^{{{int(ex)}}}$", fontsize=7, pad=3, linespacing=1.25)
            ax.set_xlabel(xl, labelpad=1)
            ax.set_aspect("equal")
            ax.grid(False)
            if p == "heat1d":
                ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
                ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
                if s == "grid":                              # first column of the cell-centred 16 x 16 grid
                    ax.axvline(1 / 32, color="#eb6834", lw=0.7, ls=(0, (2, 2)))
            else:
                ax.set_xticks([0, np.pi / 2, np.pi]); ax.set_xticklabels(["0", "π/2", "π"])
                ax.set_yticks([0, np.pi / 2, np.pi]); ax.set_yticklabels(["0", "π/2", "π"])
            if j == 0:
                ax.set_ylabel(yl, labelpad=2)
            else:
                ax.set_yticklabels([])
        bb = axes[i][-1].get_position()
        cax = fig.add_axes([bb.x1 + 0.014, bb.y0, 0.012, bb.height])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(rf"$|u_\theta - u^{{*}}|\;/\;10^{{{ex10}}}$", fontsize=7.5)
        cb.outline.set_linewidth(0.4)
        # where the grid error sits (heat): share of the squared error in 0 <= t <= 0.05
        if p == "heat1d":
            t = X[:, 0].reshape(n, n)[:, 0]
            for s in show:
                e2 = errs[s] ** 2
                info[f"{p}:{s}"]["share_sq_error_t_le_0.05"] = float(e2[t <= 0.05 + 1e-12].sum() / e2.sum())
                info[f"{p}:{s}"]["max_abs_error_t_gt_0.05"] = float(errs[s][t > 0.05 + 1e-12].max())
    save(fig, "s4_lowdim_errmaps")
    out = os.path.join(os.path.dirname(FIGDIR), "data", "s4_lowdim_errmaps_info.json")
    json.dump(info, open(out, "w"), indent=1)
    print("wrote", out)
    for k, v in info.items():
        print("  ", k, v)


if __name__ == "__main__":
    which = sys.argv[1:] or ["samplers", "curves", "budget", "errmaps"]
    runs = load_runs() if {"samplers", "curves"} & set(which) else None
    if "samplers" in which: fig_samplers(runs)
    if "curves" in which: fig_curves(runs)
    if "budget" in which: fig_budget()
    if "errmaps" in which: fig_errmaps()
