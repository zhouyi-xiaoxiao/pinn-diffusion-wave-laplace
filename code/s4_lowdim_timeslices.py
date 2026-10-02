"""Error on time slices (Section 4.5): time-resolved relative error of the heat problem H.

Question.  The space-time relative L2 error of the heat PINN is about 1e-3 after
Adam -> L-BFGS.  How large is the relative error on a single time slice, in particular on
late slices where the exact solution exp(-pi^2 t) sin(pi x) has decayed by up to
exp(-pi^2) = 5.17e-5?

Protocol (identical to the benchmark of research_benchmark/, problem `heat1d`):
  network 2 x 128 tanh, PyTorch default initialisation, float32, CPU;
  N_r = 256 interior points, strategies `random` (fixed i.i.d. uniform) and `sobol`
  (scrambled Sobol', fixed); 100 initial + 2 x 100 boundary points; unit loss weights;
  arm "adam":       Adam, lr 1e-3 constant, 3000 iterations;
  arm "adam_lbfgs": the same first 1500 Adam iterations, then 1500 L-BFGS iterations
                    (torch.optim.LBFGS, strong Wolfe, memory 50) on the fixed point set;
  seeds 0..9 (a seed fixes the initial weights and the point set).
Training is done by the unmodified single-model reference driver
`pinnbench.core.run_single`; the only adaptation is that the two networks it creates
(Adam arm, L-BFGS arm) are captured through a recording wrapper around `pinnbench.core.MLP`
so that they can be evaluated after training.

Evaluation.  Held-out grid 201 (t) x 201 (x) on [0,1]^2, i.e. the benchmark's evaluation
grid.  For every time level t_i of that grid
    slice_rel[i] = || u_theta(t_i, .) - u*(t_i, .) ||_2 / || u*(t_i, .) ||_2   (201 x-points)
    slice_abs[i] = sqrt( mean_x (u_theta - u*)^2 )
The trained float32 weights are evaluated in float64 so that evaluation round-off does not
contaminate the late slices (at t = 1 the solution is 5e-5).  The space-time relative L2 and
L-infinity errors on the same grid are stored as well (float64 evaluation, and the float32
value returned by run_single for cross-checking).

Usage
  python s4_lowdim_timeslices.py run  <strategy> <seed_lo> <seed_hi>   # one file per run
  python s4_lowdim_timeslices.py merge                                 # -> data/s4_lowdim_timeslices.json
  python s4_lowdim_timeslices.py plot                                  # -> figures/s4_lowdim_timeslices.pdf
Raw per-run output: data/s4_lowdim_timeslices_runs/<strategy>_s<seed>.json
No timing statement is derived from these runs (the machine is shared).
"""
import copy
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
ROOT = ARTICLE if os.path.isdir(os.path.join(ARTICLE, "research_benchmark")) else os.path.dirname(ARTICLE)  # repository root
sys.path.insert(0, os.path.join(ROOT, "research_benchmark"))

DATA = os.path.join(ARTICLE, "data")
RUNS = os.path.join(DATA, "s4_lowdim_timeslices_runs")
OUT_JSON = os.path.join(DATA, "s4_lowdim_timeslices.json")
OUT_FIG = os.path.join(ARTICLE, "figures", "s4_lowdim_timeslices.pdf")

PROBLEM, N_R, N_ADAM, BRANCH_AT = "heat1d", 256, 3000, 1500
STRATEGIES = ["random", "sobol"]
SEEDS = list(range(10))
T_REPORT = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
ARMS = ["adam", "adam_lbfgs"]


# ----------------------------------------------------------------------------------------
# training (reference driver) + evaluation
# ----------------------------------------------------------------------------------------
def run_single_with_nets(problem, strategy, seed, test_X, test_u, eval_X, eval_u):
    """pinnbench.core.run_single, unmodified, plus the two trained networks."""
    import pinnbench.core as core

    created = []
    original = core.MLP

    def recording_mlp(*a, **k):
        net = original(*a, **k)
        created.append(net)
        return net

    core.MLP = recording_mlp
    try:
        res = core.run_single(problem, strategy, seed, N_R, N_ADAM, BRANCH_AT, log_every=100,
                              test_X=test_X, test_u=test_u, eval_X=eval_X, eval_u=eval_u)
    finally:
        core.MLP = original
    assert len(created) == 2, "run_single is expected to build exactly two networks"
    return res, {"adam": created[0], "adam_lbfgs": created[1]}


def evaluate(net, P):
    """Slice-wise and space-time errors on the 201 x 201 held-out grid, float64 evaluation."""
    import torch

    X = P.eval_points()                                   # (201*201, 2), t-major ordering
    u = P.exact(X)
    net64 = copy.deepcopy(net).double()
    with torch.no_grad():
        up = net64(torch.as_tensor(X)).squeeze(-1).numpy()
    n = 201
    E = (up - u).reshape(n, n)                            # rows: t, columns: x
    U = u.reshape(n, n)
    slice_abs = np.sqrt((E ** 2).mean(1))
    slice_rel = np.linalg.norm(E, axis=1) / np.linalg.norm(U, axis=1)
    return {"rel_l2_spacetime": float(np.linalg.norm(E) / np.linalg.norm(U)),
            "linf_spacetime": float(np.abs(E).max()),
            "slice_rel_l2": slice_rel.tolist(), "slice_abs_l2": slice_abs.tolist(),
            "slice_linf": np.abs(E).max(1).tolist()}


def cmd_run(strategy, seed_lo, seed_hi):
    import torch
    from pinnbench.problems import make_problem

    torch.set_num_threads(2)
    os.makedirs(RUNS, exist_ok=True)
    P = make_problem(PROBLEM, device="cpu")
    eval_X = P.eval_points()
    eval_u = P.exact(eval_X)
    test_X = P.test_subset()
    test_u = P.exact(test_X)
    for seed in range(seed_lo, seed_hi):
        path = os.path.join(RUNS, f"{strategy}_s{seed}.json")
        if os.path.exists(path):
            print("skip (exists)", path, flush=True)
            continue
        res, nets = run_single_with_nets(P, strategy, seed, test_X, test_u, eval_X, eval_u)
        rec = {"problem": PROBLEM, "strategy": strategy, "seed": seed, "n_r": N_R, "n_adam": N_ADAM,
               "branch_at": BRANCH_AT, "driver": "pinnbench.core.run_single (torch.optim.LBFGS, strong Wolfe)",
               "device": "cpu", "train_dtype": "float32", "eval_dtype": "float64",
               "torch_version": torch.__version__}
        for arm in ARMS:
            ev = evaluate(nets[arm], P)
            ev["rel_l2_float32_driver"] = res[arm]["rel_l2"]
            ev["linf_float32_driver"] = res[arm]["linf"]
            ev["final_loss"] = res[arm]["final_loss"]
            if arm == "adam_lbfgs":
                ev["lbfgs_iters"] = res[arm]["lbfgs_iters"]
                ev["lbfgs_stop"] = res[arm]["stop"]
            rec[arm] = ev
        json.dump(rec, open(path, "w"))
        print(f"{strategy} seed {seed}: space-time rel-L2 adam {rec['adam']['rel_l2_spacetime']:.3e}, "
              f"adam_lbfgs {rec['adam_lbfgs']['rel_l2_spacetime']:.3e}; slice t=1 adam_lbfgs "
              f"{rec['adam_lbfgs']['slice_rel_l2'][-1]:.3e} ({rec['adam_lbfgs']['lbfgs_stop']})", flush=True)


# ----------------------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------------------
def summarise(v):
    v = np.asarray(v, float)
    lg = np.log10(v)
    return {"n": int(len(v)), "geo_mean": float(10 ** lg.mean()), "median": float(np.median(v)),
            "mean": float(v.mean()), "std": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}


def cmd_merge():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RUNS, "*.json")))]
    t = np.linspace(0, 1, 201)
    idx = [int(round(tt * 200)) for tt in T_REPORT]
    summary, missing = {}, []
    for s in STRATEGIES:
        have = sorted(r["seed"] for r in runs if r["strategy"] == s)
        missing += [(s, k) for k in SEEDS if k not in have]
        for arm in ARMS:
            rr = sorted([r for r in runs if r["strategy"] == s], key=lambda r: r["seed"])
            if not rr:
                continue
            e = {"seeds": [r["seed"] for r in rr],
                 "rel_l2_spacetime": summarise([r[arm]["rel_l2_spacetime"] for r in rr]),
                 "linf_spacetime": summarise([r[arm]["linf_spacetime"] for r in rr]),
                 "slices": {}}
            for tt, i in zip(T_REPORT, idx):
                e["slices"][f"{tt:g}"] = {
                    "rel_l2": summarise([r[arm]["slice_rel_l2"][i] for r in rr]),
                    "abs_l2": summarise([r[arm]["slice_abs_l2"][i] for r in rr]),
                    "rel_l2_per_seed": [r[arm]["slice_rel_l2"][i] for r in rr]}
            # ratio of the slice error at t = 1 to the space-time error, per seed
            e["ratio_slice_t1_over_spacetime"] = summarise(
                [r[arm]["slice_rel_l2"][-1] / r[arm]["rel_l2_spacetime"] for r in rr])
            # first time at which the geometric-mean slice error exceeds 1 %, 10 %
            S = np.array([r[arm]["slice_rel_l2"] for r in rr])
            g = 10 ** np.log10(S).mean(0)
            e["geo_mean_slice_rel_l2_all_t"] = g.tolist()
            for thr in (0.01, 0.1):
                above = np.nonzero(g > thr)[0]
                e[f"first_t_geo_mean_above_{thr:g}"] = float(t[above[0]]) if len(above) else None
            e["n_seeds_slice_t1_above_0.1"] = int((S[:, -1] > 0.1).sum())
            e["n_seeds_slice_t1_above_1"] = int((S[:, -1] > 1.0).sum())
            # where in time the squared space-time error sits: share of the time levels t <= 0.05 and t <= 0.25
            A2 = np.array([r[arm]["slice_abs_l2"] for r in rr]) ** 2
            for tmax in (0.05, 0.25):
                sh = A2[:, t <= tmax + 1e-12].sum(1) / A2.sum(1)
                e[f"share_sq_error_t_le_{tmax:g}"] = {"median": float(np.median(sh)), "min": float(sh.min()),
                                                      "max": float(sh.max()), "per_seed": sh.tolist()}
            summary[f"{s}:{arm}"] = e
    # the same shares for the exact solution: the norm in the denominator of the space-time error
    u2 = np.exp(-2 * np.pi ** 2 * t)                      # ||u*(t,.)||^2 on the grid, up to a constant factor
    norm_share = {f"t_le_{tmax:g}": float(u2[t <= tmax + 1e-12].sum() / u2.sum()) for tmax in (0.05, 0.25)}
    cells = [summary[f"{s}:{arm}"] for s in STRATEGIES for arm in ARMS]
    err_share = {f"t_le_{tmax:g}": {"median_range_over_cells": [min(c[f"share_sq_error_t_le_{tmax:g}"]["median"] for c in cells),
                                                                max(c[f"share_sq_error_t_le_{tmax:g}"]["median"] for c in cells)],
                                    "range_over_runs": [min(c[f"share_sq_error_t_le_{tmax:g}"]["min"] for c in cells),
                                                        max(c[f"share_sq_error_t_le_{tmax:g}"]["max"] for c in cells)]}
                 for tmax in (0.05, 0.25)}
    out = {"experiment": "Section 4.5: time-resolved relative L2 error, heat problem H",
           "protocol": {"problem": PROBLEM, "n_r": N_R, "n_adam": N_ADAM, "branch_at": BRANCH_AT,
                        "strategies": STRATEGIES, "seeds": SEEDS, "t_report": T_REPORT,
                        "eval_grid": "201 (t) x 201 (x), uniform, closed", "driver": "pinnbench.core.run_single"},
           "t_grid": t.tolist(), "missing_runs": missing, "summary": summary,
           "share_of_exact_solution_norm_sq": norm_share, "share_of_sq_error": err_share, "runs": runs}
    # LaTeX rows for the slice-error table of Section 4 (geometric means over seeds, per cent)
    def pc(x):
        x = 100 * x
        if x < 1000:
            return f"{x:#.3g}".rstrip(".")
        v = f"{x:.0f}"
        return v if len(v) < 5 else v[:-3] + "\\," + v[-3:]      # thin space from five digits on
    lab = {"random": r"\strat{random}", "sobol": r"\strat{Sobol}"}
    arm_lab = {"adam": r"\Adam", "adam_lbfgs": r"\AdamLBFGS"}
    rows_tex = []
    for s_ in STRATEGIES:
        for arm in ARMS:
            e = summary[f"{s_}:{arm}"]
            cells = [pc(e["rel_l2_spacetime"]["geo_mean"])] + [pc(e["slices"][f"{tt:g}"]["rel_l2"]["geo_mean"]) for tt in T_REPORT]
            r1 = e["slices"]["1"]["rel_l2"]
            rows_tex.append(f"{lab[s_]} & {arm_lab[arm]} & " + " & ".join(cells) + f" & {pc(r1['min'])}--{pc(r1['max'])} \\\\")
    open(os.path.join(DATA, "s4_lowdim_timeslices_table.tex"), "w").write("\n".join(rows_tex) + "\n")
    sys.path.insert(0, HERE)
    from inject import inject
    inject("s4_lowdim.tex", "s4_lowdim:tab:timeslice", "\n".join(rows_tex))
    print("\n".join(rows_tex))
    json.dump(out, open(OUT_JSON, "w"))
    print("wrote", OUT_JSON, "| runs:", len(runs), "| missing:", missing)
    print("share of ||u*||^2 on the grid:", norm_share, "| share of the squared error:", err_share)
    for k, e in summary.items():
        print(f"\n{k}  (n = {e['rel_l2_spacetime']['n']})  space-time rel-L2 geo {e['rel_l2_spacetime']['geo_mean']:.3e} "
              f"[{e['rel_l2_spacetime']['min']:.3e}, {e['rel_l2_spacetime']['max']:.3e}]")
        for tt in T_REPORT:
            sl = e["slices"][f"{tt:g}"]["rel_l2"]
            ab = e["slices"][f"{tt:g}"]["abs_l2"]
            print(f"   t = {tt:<4g}  rel geo {sl['geo_mean']:.3e}  median {sl['median']:.3e}  [{sl['min']:.3e}, {sl['max']:.3e}]"
                  f"   abs geo {ab['geo_mean']:.3e}")
        r = e["ratio_slice_t1_over_spacetime"]
        print(f"   slice(t=1)/space-time: geo {r['geo_mean']:.1f} [{r['min']:.1f}, {r['max']:.1f}];"
              f" seeds with slice(t=1) > 0.1: {e['n_seeds_slice_t1_above_0.1']}, > 1: {e['n_seeds_slice_t1_above_1']};"
              f" geo-mean slice error first > 1 % at t = {e['first_t_geo_mean_above_0.01']}, > 10 % at t = {e['first_t_geo_mean_above_0.1']}")


# ----------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------
def cmd_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, HERE)
    from s4_lowdim_style import apply_style, panel_label, ARM_COL, ARM_LAB, INK2

    apply_style()
    d = json.load(open(OUT_JSON))
    t = np.array(d["t_grid"])
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.05), sharey=True)
    titles = {"random": "(a) random", "sobol": "(b) Sobol'"}
    for ax, s in zip(axes, STRATEGIES):
        rr = [r for r in d["runs"] if r["strategy"] == s]
        for arm in ARMS:
            S = np.array([r[arm]["slice_rel_l2"] for r in rr])
            for row in S:
                ax.plot(t, row, color=ARM_COL[arm], lw=0.5, alpha=0.4, zorder=2)
            g = 10 ** np.log10(S).mean(0)
            ax.plot(t, g, color=ARM_COL[arm], lw=1.8, zorder=4)
            st = d["summary"][f"{s}:{arm}"]["rel_l2_spacetime"]["geo_mean"]
            ax.axhline(st, color=ARM_COL[arm], lw=1.1, ls=(0, (1, 1.6)), zorder=3)
        # constant absolute error: slope of exp(pi^2 t), anchored at the L-BFGS curve at t = 0
        S = np.array([r["adam_lbfgs"]["slice_rel_l2"] for r in rr])
        g0 = 10 ** np.log10(S[:, 0]).mean()
        ax.plot(t, g0 * np.exp(np.pi ** 2 * t), color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=3)
        ax.axhline(1.0, color=INK2, lw=0.6, zorder=1)
        ax.text(0.015, 1.25, "100 %", ha="left", va="bottom", fontsize=7, color=INK2)
        ax.set_yscale("log")
        ax.set_xlabel("$t$")
        ax.set_xlim(0, 1)
        ax.set_ylim(1e-4, 3e3)
        panel_label(ax, titles[s])
    axes[0].set_ylabel(r"relative $L^{2}$ error on the slice, $\varepsilon_{2}^{\mathrm{slice}}(t)$")
    handles = [Line2D([], [], color=ARM_COL["adam"], lw=1.8, label="Adam"),
               Line2D([], [], color=ARM_COL["adam_lbfgs"], lw=1.8, label="Adam \u2192 L-BFGS"),
               Line2D([], [], color=INK2, lw=1.1, ls=(0, (1, 1.6)), label="space\u2013time error of the same runs"),
               Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3)), label=r"$\propto \mathrm{e}^{\pi^{2}t}$ (constant absolute error)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.0)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print("wrote", OUT_FIG)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "merge"
    if cmd == "run":
        cmd_run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "merge":
        cmd_merge()
    elif cmd == "plot":
        cmd_plot()
    else:
        raise SystemExit(__doc__)
