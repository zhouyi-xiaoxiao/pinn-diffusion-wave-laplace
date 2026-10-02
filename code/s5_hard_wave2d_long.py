"""Longer runs on Wave2 (Section 5.1): does the 2-D wave PINN (problem Wave2) converge with a
longer L-BFGS phase?

Protocol
---------------------------------
  problem     pinnbench "wave2d": u_tt = Lap u on (-1,1)^2, t in (0,1], u(0) = sin(pi x) sin(pi y),
              u_t(0) = 0, u = 0 on the boundary; exact u* = sin(pi x) sin(pi y) cos(sqrt(2) pi t)
  network     2 x 50 tanh (2801 parameters), PyTorch default initialisation, float32, CPU
  points      N_r = 512 scrambled-Sobol interior points (fixed), 21 x 21 initial points,
              11 x 21 points per side face -- exactly the benchmark's sets
  optimiser   reference single-model driver pinnbench.core.run_single, UNMODIFIED:
              arm "adam_lbfgs": 1500 Adam steps (lr 1e-3) then torch.optim.LBFGS (strong Wolfe,
                                memory 50) up to TOTAL iterations in total;
              arm "adam":       TOTAL Adam steps (the same first 1500 steps), as a comparator
  seeds       0, 1, 2 (one process per seed)
  output      held-out relative L2 error against iteration (4096-point random subset of the
              21 x 81 x 81 evaluation grid, as in the benchmark curves), final relative L2 and
              L-infinity error on the full evaluation grid, the centre value u(t, 0.5, 0.5) and
              the mode amplitude <u(t), S>/<S, S>, S = sin(pi x) sin(pi y), at t = 0, 0.05, ..., 1.

The trained networks are captured by registering every MLP that run_single constructs (the
first is the Adam network, the second the L-BFGS network); run_single itself is not changed.
Wall-clock times are deliberately NOT stored: other jobs ran on the machine at the same time
and no timing statement of the article is derived from this experiment.

Usage
-----
  python s5_hard_wave2d_long.py run <seed> [total_iterations]   # -> data/s5_hard_wave2d_long_parts/seed<k>.json
  python s5_hard_wave2d_long.py merge                           # -> data/s5_hard_wave2d_long.json
  python s5_hard_wave2d_long.py plot                            # -> figures/s5_hard_wave2d.pdf
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
PARTS = os.path.join(ART, "data", "s5_hard_wave2d_long_parts")
OUT = os.path.join(ART, "data", "s5_hard_wave2d_long.json")
FIG = os.path.join(ART, "figures", "s5_hard_wave2d.pdf")

N_R, BRANCH_AT, TOTAL_DEFAULT, SAMPLER = 512, 1500, 10000, "sobol"
SEEDS = [0, 1, 2]
OMEGA = math.sqrt(2.0) * math.pi


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

    P = make_problem("wave2d", device="cpu")
    eval_X = P.eval_points()
    eval_u = P.exact(eval_X)
    test_X = P.test_subset()
    test_u = P.exact(test_X)
    res = core.run_single(P, SAMPLER, seed, N_R, total, BRANCH_AT, log_every=100, lbfgs_chunk=50,
                          test_X=test_X, test_u=test_u, eval_X=eval_X, eval_u=eval_u, verbose=True)
    net_adam, net_lbfgs = created[0], created[1]

    t = np.linspace(0.0, 1.0, 21)
    centre = np.stack([t, 0.5 + 0 * t, 0.5 + 0 * t], 1)
    xs = np.linspace(-1, 1, 81)
    S = np.outer(np.sin(np.pi * xs), np.sin(np.pi * xs)).ravel()

    def diagnostics(net):
        u = core.predict(net, eval_X, "cpu").reshape(21, -1)          # (t, 81*81)
        ue = eval_u.reshape(21, -1)
        amp = (u @ S) / (S @ S)
        slice_err = np.linalg.norm(u - ue, axis=1) / np.linalg.norm(S)   # relative to ||S||, not to ||u*(t)||
        return {"centre_value": core.predict(net, centre, "cpu").tolist(),
                "mode_amplitude": amp.tolist(),
                "slice_l2_error_over_norm_S": slice_err.tolist()}

    def strip(arm):
        a = dict(arm)
        a.pop("time", None)
        a["curve"] = [{k: v for k, v in c.items() if k != "time"} for c in arm["curve"]]
        return a

    out = {"problem": "wave2d", "sampler": SAMPLER, "seed": seed, "n_r": N_R, "branch_at": BRANCH_AT,
           "total_iterations": total, "driver": "pinnbench.core.run_single (torch.optim.LBFGS, strong Wolfe)",
           "device": "cpu", "dtype": "float32", "torch_threads": 2, "torch_version": torch.__version__,
           "t": t.tolist(), "exact_centre_value": np.cos(OMEGA * t).tolist(),
           "adam": {**strip(res["adam"]), **diagnostics(net_adam)},
           "adam_lbfgs": {**strip(res["adam_lbfgs"]), **diagnostics(net_lbfgs)}}
    os.makedirs(PARTS, exist_ok=True)
    json.dump(out, open(os.path.join(PARTS, f"seed{seed}.json"), "w"), indent=1)
    b = out["adam_lbfgs"]
    print(f"seed {seed}: Adam({total}) rel_l2 {out['adam']['rel_l2']:.4e} | Adam->L-BFGS rel_l2 {b['rel_l2']:.4e} "
          f"linf {b['linf']:.4e} lbfgs_iters {b['lbfgs_iters']} stop {b['stop']}", flush=True)


def _at(curve, it):
    v = [c["test_rel_l2"] for c in curve if c["it"] == it]
    return v[0] if v else None


def merge():
    runs = [json.load(open(os.path.join(PARTS, f"seed{s}.json"))) for s in SEEDS]
    summ = {"note": "Longer runs on Wave2 (Section 5.1). 'test' errors are on the 4096-point subset of the evaluation grid used for "
                    "training curves; rel_l2 / linf are on the full 21 x 81 x 81 grid. No timings stored.",
            "seeds": SEEDS, "total_iterations": [r["total_iterations"] for r in runs], "per_seed": []}
    for r in runs:
        cb, ca = r["adam_lbfgs"]["curve"], r["adam"]["curve"]
        last_it = cb[-1]["it"]
        row = {"seed": r["seed"], "lbfgs_last_iteration": last_it, "lbfgs_iters": r["adam_lbfgs"]["lbfgs_iters"],
               "lbfgs_fevals": r["adam_lbfgs"]["lbfgs_fevals"], "stop": r["adam_lbfgs"]["stop"],
               "adam_lbfgs_rel_l2": r["adam_lbfgs"]["rel_l2"], "adam_lbfgs_linf": r["adam_lbfgs"]["linf"],
               "adam_lbfgs_final_loss": r["adam_lbfgs"]["final_loss"],
               "adam_rel_l2": r["adam"]["rel_l2"], "adam_linf": r["adam"]["linf"], "adam_final_loss": r["adam"]["final_loss"],
               "adam_lbfgs_test_rel_l2_at": {str(it): _at(cb, it) for it in (1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)},
               "adam_lbfgs_loss_at": {str(it): ([c["loss"] for c in cb if c["it"] == it] or [None])[0]
                                      for it in (1500, 3000, 5000, 8000, 10000)},
               "adam_test_rel_l2_at": {str(it): _at(ca, it) for it in (1500, 3000, 5000, 8000, 10000)},
               # flattening diagnostics: error ratio over the last 1000 logged L-BFGS iterations ending at
               # iteration 9500 (the last logged iteration common to all seeds), and over 2000 -> 3000
               "adam_lbfgs_test_ratio_9500_over_8500": _at(cb, 9500) / _at(cb, 8500),
               "adam_lbfgs_test_ratio_3000_over_2000": _at(cb, 3000) / _at(cb, 2000),
               "adam_lbfgs_test_rel_l2_last": cb[-1]["test_rel_l2"],
               "adam_test_ratio_10000_over_9000": _at(ca, 10000) / _at(ca, 9000),
               "max_abs_centre_error": float(np.abs(np.array(r["adam_lbfgs"]["centre_value"]) - np.array(r["exact_centre_value"])).max()),
               "centre_value_t0.35_t0.40": [r["adam_lbfgs"]["centre_value"][7], r["adam_lbfgs"]["centre_value"][8]],
               "centre_value_t1": r["adam_lbfgs"]["centre_value"][-1],
               "mode_amplitude_t1": r["adam_lbfgs"]["mode_amplitude"][-1]}
        summ["per_seed"].append(row)
    for key in ("adam_lbfgs_rel_l2", "adam_lbfgs_linf", "adam_rel_l2", "adam_linf"):
        v = np.array([p[key] for p in summ["per_seed"]])
        summ[key + "_mean"], summ[key + "_std"] = float(v.mean()), float(v.std(ddof=1))
        summ[key + "_geomean"] = float(np.exp(np.log(v).mean()))
    summ["exact_centre_value_t1"] = math.cos(OMEGA)
    json.dump({"summary": summ, "runs": runs}, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(OUT))
    runs = d["runs"]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from figstyle import apply_style, panel_label, BLUE, ORANGE, INK, INK2, GRID   # style shared by all figures
    apply_style(**{"legend.fontsize": 7})
    from matplotlib.lines import Line2D
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.3, 1.0]})
    styles = ["-", "--", ":"]
    for r, ls in zip(runs, styles):
        ca, cb = r["adam"]["curve"], r["adam_lbfgs"]["curve"]
        a1.plot([c["it"] for c in ca], [c["test_rel_l2"] for c in ca], color=BLUE, ls=ls, lw=1.0)
        cbl = [c for c in cb if c["it"] >= r["branch_at"]]
        a1.plot([c["it"] for c in cbl], [c["test_rel_l2"] for c in cbl], color=ORANGE, ls=ls, lw=1.3)
        a1.plot([cbl[-1]["it"]], [cbl[-1]["test_rel_l2"]], marker="|", ms=7, mew=1.2, color=ORANGE)   # L-BFGS stopped here
    a1.axvline(runs[0]["branch_at"], color=INK2, lw=0.6, ls=(0, (1, 2)))
    a1.axvline(3000, color=INK2, lw=0.6, ls=(0, (1, 2)))
    a1.set_yscale("log")
    a1.set_ylim(3e-3, 6.0)
    a1.set_xlim(0, max(r["total_iterations"] for r in runs) + 150)
    a1.set_xlabel("iteration")
    a1.set_ylabel(r"relative $L^2$ error $\varepsilon_2$ (held-out)")
    a1.text(runs[0]["branch_at"] - 110, 3.6e-3, "switch", fontsize=7, color=INK2, va="bottom", ha="right")
    a1.text(3000 + 110, 3.6e-3, "benchmark budget", fontsize=7, color=INK2, va="bottom", ha="left")
    h1 = [Line2D([], [], color=BLUE, lw=1.0, label="Adam only"),
          Line2D([], [], color=ORANGE, lw=1.3, label=r"Adam$\to$L-BFGS"),
          Line2D([], [], color=INK2, lw=1.0, ls="-", label="seed 0"),
          Line2D([], [], color=INK2, lw=1.0, ls="--", label="seed 1"),
          Line2D([], [], color=INK2, lw=1.0, ls=":", label="seed 2")]
    a1.legend(handles=h1, ncol=3, loc="upper right", handlelength=2.0, columnspacing=1.0, borderaxespad=0.2,
              frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    panel_label(a1, "(a)")

    tt = np.linspace(0, 1, 201)
    a2.plot(tt, np.cos(OMEGA * tt), color=INK, lw=1.0)
    for k, r in enumerate(runs):
        a2.plot(r["t"], r["adam"]["centre_value"], color=BLUE, lw=0.9, ls=styles[k])
    for r, mk in zip(runs, ["o", "s", "^"]):
        a2.plot(r["t"], r["adam_lbfgs"]["centre_value"], ls="none", marker=mk, ms=3.4, mfc=ORANGE, mec="white", mew=0.4)
    a2.axhline(0, color=INK2, lw=0.6)
    a2.set_xlabel("$t$")
    a2.set_ylabel(r"$u_\theta(t,\,0.5,\,0.5)$")
    h2 = [Line2D([], [], color=INK, lw=1.0, label=r"exact, $\cos(\sqrt{2}\,\pi t)$"),
          Line2D([], [], color=ORANGE, lw=0, marker="o", ms=3.4, mec="white", mew=0.4, label=r"Adam$\to$L-BFGS"),
          Line2D([], [], color=BLUE, lw=0.9, label="Adam only")]
    a2.legend(handles=h2, loc="upper right", handlelength=1.6, borderaxespad=0.2)
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
