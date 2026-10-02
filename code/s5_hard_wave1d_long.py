"""Longer runs on Wave1 (Section 5.3): is the failure of the unit-weight PINN on the two-mode wave problem Wave1 a
property of the loss, or of the small budget of the benchmark?

Rathore et al. (ICML 2024, Table 1) train an unweighted PINN on the same equation with a faster second mode
(sin(pi x) + 0.5 sin(5 pi x)) for 41 000 iterations and report a relative L2 error of 0.349 with Adam and 0.055
with Adam followed by L-BFGS.  This script repeats the benchmark configuration of Wave1 with that budget.

Protocol
--------
  problem     pinnbench "wave1d": u_tt = 4 u_xx on (0,1), t in (0,1], u(0,x) = sin(pi x) + 0.5 sin(4 pi x),
              u_t(0,x) = 0, u(t,0) = u(t,1) = 0; exact u* = sin(pi x) cos(2 pi t) + 0.5 sin(4 pi x) cos(8 pi t)
  network     3 x 64 tanh (8577 parameters), PyTorch default initialisation, float32, CPU
  points      N_r = 1024 scrambled-Sobol interior points (fixed), 100 initial and 2 x 100 boundary points:
              exactly the benchmark's sets; unit loss weights
  optimiser   reference single-model driver pinnbench.core.run_single, UNMODIFIED:
              arm "adam_lbfgs": 1500 Adam steps (lr 1e-3), then torch.optim.LBFGS (strong Wolfe, memory 50)
                                until it stops on its own tolerance or TOTAL iterations are reached;
              arm "adam":       TOTAL Adam steps (the same first 1500 steps), as a comparator
  seeds       0, 1, 2 (one process per seed)
  output      held-out relative L2 error against iteration (4096-point subset of the 201 x 201 evaluation
              grid, as in the benchmark curves); final relative L2 and L-infinity error on the full grid;
              and the projection of the trained network onto the two spatial modes,
                  a_k(t) = <u(t,.), sin(k pi x)> / <sin(k pi x), sin(k pi x)>,  k = 1, 4,
              at the 201 time levels of the grid, against cos(2 pi t) and 0.5 cos(8 pi t).
The same projection is applied to the stored predictions of the benchmark (seed 0, five strategies, both arms):
  research_benchmark/results/runs/wave1d_pred_seed0.npz.
No wall-clock time is stored: other jobs ran on the machine.

Usage
-----
  python s5_hard_wave1d_long.py run <seed> [total_iterations]   # -> data/s5_hard_wave1d_long_parts/seed<k>.json
  python s5_hard_wave1d_long.py merge                           # -> data/s5_hard_wave1d_long.json
  python s5_hard_wave1d_long.py plot                            # -> figures/s5_hard_wave1d.pdf
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)  # repository root
sys.path.insert(0, os.path.join(ROOT, "research_benchmark"))
PARTS = os.path.join(ART, "data", "s5_hard_wave1d_long_parts")
OUT = os.path.join(ART, "data", "s5_hard_wave1d_long.json")
FIG = os.path.join(ART, "figures", "s5_hard_wave1d.pdf")

N_R, BRANCH_AT, TOTAL_DEFAULT, SAMPLER = 1024, 1500, 41000, "sobol"
SEEDS = [0, 1, 2]
NT = NX = 201


def mode_projection(u):
    """u: (201*201,) values on the evaluation grid, t the first (slow) index.  Returns the amplitudes a_1(t),
    a_4(t) (trapezoidal rule in x, exact for these modes up to rounding), the relative L2 error of each
    amplitude over t, and the share of the squared error that lies in each mode and in the remainder."""
    t = np.linspace(0.0, 1.0, NT)
    x = np.linspace(0.0, 1.0, NX)
    U = np.asarray(u, dtype=np.float64).reshape(NT, NX)
    w = np.full(NX, 1.0 / (NX - 1)); w[0] = w[-1] = 0.5 / (NX - 1)
    s1, s4 = np.sin(np.pi * x), np.sin(4 * np.pi * x)
    a1 = (U * s1) @ w / ((s1 * s1) @ w)
    a4 = (U * s4) @ w / ((s4 * s4) @ w)
    e1, e4 = np.cos(2 * np.pi * t), 0.5 * np.cos(8 * np.pi * t)
    ex = np.outer(e1, s1) + np.outer(e4, s4)
    err = U - ex
    p1 = np.outer(a1 - e1, s1)
    p4 = np.outer(a4 - e4, s4)
    rest = err - p1 - p4
    tot = float((err ** 2).sum())
    return {"a1": a1.tolist(), "a4": a4.tolist(),
            "rel_l2_a1": float(np.linalg.norm(a1 - e1) / np.linalg.norm(e1)),
            "rel_l2_a4": float(np.linalg.norm(a4 - e4) / np.linalg.norm(e4)),
            "rms_a4_over_rms_exact": float(np.linalg.norm(a4) / np.linalg.norm(e4)),
            "share_sq_error_mode1": float((p1 ** 2).sum() / tot),
            "share_sq_error_mode4": float((p4 ** 2).sum() / tot),
            "share_sq_error_rest": float((rest ** 2).sum() / tot)}


def run(seed, total):
    import torch
    torch.set_num_threads(2)
    from pinnbench import core
    from pinnbench.problems import make_problem

    created = []

    class RecordingMLP(core.MLP):                      # same network; only remembers the instance
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    core.MLP = RecordingMLP

    P = make_problem("wave1d", device="cpu")
    eval_X = P.eval_points()
    eval_u = P.exact(eval_X)
    test_X = P.test_subset()
    test_u = P.exact(test_X)
    res = core.run_single(P, SAMPLER, seed, N_R, total, BRANCH_AT, log_every=100, lbfgs_chunk=50,
                          test_X=test_X, test_u=test_u, eval_X=eval_X, eval_u=eval_u, verbose=True)
    net_adam, net_lbfgs = created[0], created[1]

    def strip(arm):
        a = dict(arm)
        a.pop("time", None)
        a["curve"] = [{k: v for k, v in c.items() if k != "time"} for c in arm["curve"]]
        return a

    out = {"problem": "wave1d", "sampler": SAMPLER, "seed": seed, "n_r": N_R, "branch_at": BRANCH_AT,
           "total_iterations": total, "driver": "pinnbench.core.run_single (torch.optim.LBFGS, strong Wolfe)",
           "device": "cpu", "dtype": "float32", "torch_threads": 2, "torch_version": torch.__version__,
           "adam": {**strip(res["adam"]), "modes": mode_projection(core.predict(net_adam, eval_X, "cpu"))},
           "adam_lbfgs": {**strip(res["adam_lbfgs"]), "modes": mode_projection(core.predict(net_lbfgs, eval_X, "cpu"))}}
    os.makedirs(PARTS, exist_ok=True)
    json.dump(out, open(os.path.join(PARTS, f"seed{seed}.json"), "w"), indent=1)
    b = out["adam_lbfgs"]
    print(f"seed {seed}: Adam({total}) rel_l2 {out['adam']['rel_l2']:.4e} | Adam->L-BFGS rel_l2 {b['rel_l2']:.4e} "
          f"linf {b['linf']:.4e} lbfgs_iters {b['lbfgs_iters']} stop {b['stop']}", flush=True)


def _at(curve, it):
    v = [c["test_rel_l2"] for c in curve if c["it"] == it]
    return v[0] if v else None


def _first_below(curve, level):
    for c in curve:
        if c["test_rel_l2"] < level:
            return c["it"]
    return None


def merge():
    runs = [json.load(open(os.path.join(PARTS, f"seed{s}.json"))) for s in SEEDS]
    marks = (1500, 3000, 5000, 10000, 20000, 30000, 41000)
    summ = {"note": "Longer runs on Wave1 (Section 5.3). 'test' errors are on the 4096-point subset of the evaluation grid used for "
                    "training curves; rel_l2 / linf are on the full 201 x 201 grid. No timings stored.",
            "seeds": SEEDS, "total_iterations": [r["total_iterations"] for r in runs], "per_seed": []}
    for r in runs:
        cb, ca = r["adam_lbfgs"]["curve"], r["adam"]["curve"]
        mb, ma = r["adam_lbfgs"]["modes"], r["adam"]["modes"]
        row = {"seed": r["seed"], "lbfgs_last_iteration": cb[-1]["it"], "lbfgs_iters": r["adam_lbfgs"]["lbfgs_iters"],
               "lbfgs_fevals": r["adam_lbfgs"]["lbfgs_fevals"], "stop": r["adam_lbfgs"]["stop"],
               "adam_lbfgs_rel_l2": r["adam_lbfgs"]["rel_l2"], "adam_lbfgs_linf": r["adam_lbfgs"]["linf"],
               "adam_lbfgs_final_loss": r["adam_lbfgs"]["final_loss"],
               "adam_rel_l2": r["adam"]["rel_l2"], "adam_linf": r["adam"]["linf"], "adam_final_loss": r["adam"]["final_loss"],
               "adam_lbfgs_test_rel_l2_at": {str(it): _at(cb, it) for it in marks},
               "adam_test_rel_l2_at": {str(it): _at(ca, it) for it in marks},
               "adam_lbfgs_first_iteration_below": {str(lv): _first_below(cb, lv) for lv in (0.3, 0.1, 0.05, 0.01)},
               "adam_first_iteration_below": {str(lv): _first_below(ca, lv) for lv in (0.3, 0.1, 0.05, 0.01)},
               "adam_lbfgs_modes": {k: v for k, v in mb.items() if k not in ("a1", "a4")},
               "adam_modes": {k: v for k, v in ma.items() if k not in ("a1", "a4")}}
        summ["per_seed"].append(row)
    for key in ("adam_lbfgs_rel_l2", "adam_lbfgs_linf", "adam_rel_l2", "adam_linf"):
        v = np.array([p[key] for p in summ["per_seed"]])
        summ[key + "_mean"], summ[key + "_std"] = float(v.mean()), float(v.std(ddof=1))
        summ[key + "_geomean"] = float(np.exp(np.log(v).mean()))
        summ[key + "_min"], summ[key + "_max"] = float(v.min()), float(v.max())

    # the same projection for the stored benchmark predictions (seed 0; five strategies; both arms)
    z = np.load(os.path.join(ROOT, "research_benchmark", "results", "runs", "wave1d_pred_seed0.npz"))
    from pinnbench.problems import make_problem
    P = make_problem("wave1d")
    eu = P.exact(P.eval_points())
    bench = {}
    for arm in ("adam", "adam_lbfgs"):
        for name, u in zip(z["samplers"], z[arm]):
            m = mode_projection(u)
            bench[f"{arm}|{name}"] = {"rel_l2": float(np.linalg.norm(u - eu) / np.linalg.norm(eu)),
                                      **{k: v for k, v in m.items() if k not in ("a1", "a4")}}
    for arm in ("adam", "adam_lbfgs"):
        rows = [v for k, v in bench.items() if k.startswith(arm + "|")]
        for q in ("rel_l2_a1", "rel_l2_a4", "rms_a4_over_rms_exact", "share_sq_error_mode1", "share_sq_error_mode4",
                  "share_sq_error_rest"):
            vals = [x[q] for x in rows]
            summ[f"benchmark_seed0_{arm}_{q}_range"] = [float(min(vals)), float(max(vals))]
    summ["benchmark_seed0"] = bench
    json.dump({"summary": summ, "runs": runs}, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(OUT))
    runs = d["runs"]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from figstyle import apply_style, panel_label, BLUE, ORANGE, INK, INK2   # style shared by all figures
    apply_style(**{"legend.fontsize": 7})
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.3, 1.0]})
    styles = ["-", "--", ":"]
    for r, ls in zip(runs, styles):
        ca, cb = r["adam"]["curve"], r["adam_lbfgs"]["curve"]
        a1.plot([c["it"] for c in ca], [c["test_rel_l2"] for c in ca], color=BLUE, ls=ls, lw=1.0)
        cbl = [c for c in cb if c["it"] >= r["branch_at"]]
        a1.plot([c["it"] for c in cbl], [c["test_rel_l2"] for c in cbl], color=ORANGE, ls=ls, lw=1.3)
        if r["adam_lbfgs"]["stop"] != "budget":
            a1.plot([cbl[-1]["it"]], [cbl[-1]["test_rel_l2"]], marker="|", ms=7, mew=1.2, color=ORANGE)
    a1.axvline(3000, color=INK2, lw=0.6, ls=(0, (1, 2)))
    a1.axhline(1 / math.sqrt(5), color=INK2, lw=0.6, ls=(0, (4, 2)))
    a1.set_yscale("log")
    a1.set_ylim(0.3, 1.25)
    a1.yaxis.set_major_locator(FixedLocator([0.3, 0.4, 0.5, 0.7, 1.0]))
    a1.yaxis.set_minor_locator(NullLocator())
    a1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    a1.set_xlim(0, max(r["total_iterations"] for r in runs) * 1.01)
    a1.text(40500, 1 / math.sqrt(5) * 1.015, r"slow mode only, $1/\sqrt{5}$", fontsize=7, color=INK2, va="bottom", ha="right")
    a1.text(3000 + 500, 1.17, "benchmark budget", fontsize=7, color=INK2, va="top", ha="left")
    a1.xaxis.set_major_locator(FixedLocator([0, 10000, 20000, 30000, 40000]))
    a1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
    a1.set_xlabel(r"$10^3$ iterations")
    a1.set_ylabel(r"relative $L^2$ error $\varepsilon_2$ (held-out)")
    h1 = [Line2D([], [], color=BLUE, lw=1.0, label="Adam only"),
          Line2D([], [], color=ORANGE, lw=1.3, label=r"Adam$\to$L-BFGS"),
          Line2D([], [], color=INK2, lw=1.0, ls="-", label="seed 0"),
          Line2D([], [], color=INK2, lw=1.0, ls="--", label="seed 1"),
          Line2D([], [], color=INK2, lw=1.0, ls=":", label="seed 2")]
    a1.legend(handles=h1, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 0.9), handlelength=2.0, columnspacing=1.0,
              borderaxespad=0.3, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    panel_label(a1, "(a)")

    t = np.linspace(0, 1, NT)
    a2.plot(t, 0.5 * np.cos(8 * np.pi * t), color=INK, lw=1.0)
    for r, ls in zip(runs, styles):
        a2.plot(t, r["adam"]["modes"]["a4"], color=BLUE, lw=0.9, ls=ls)
        a2.plot(t, r["adam_lbfgs"]["modes"]["a4"], color=ORANGE, lw=1.2, ls=ls)
    a2.axhline(0, color=INK2, lw=0.6)
    a2.set_xlabel("$t$")
    a2.set_ylabel(r"amplitude $a_4(t)$ of $\sin 4\pi x$")
    h2 = [Line2D([], [], color=INK, lw=1.0, label=r"exact, $\frac{1}{2}\cos 8\pi t$")]
    a2.set_ylim(-0.6, 0.78)
    a2.legend(handles=h2, loc="upper right", handlelength=1.6, borderaxespad=0.2,
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    panel_label(a2, "(b)")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG, bbox_inches="tight")
    print("wrote", FIG)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if cmd == "run":
        run(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else TOTAL_DEFAULT)
    elif cmd == "merge":
        merge()
    else:
        plot()
