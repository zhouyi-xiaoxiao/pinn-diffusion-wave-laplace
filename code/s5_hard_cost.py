"""Section 5: accuracy against wall-clock cost, PINN versus second-order finite differences.
Plot/table regeneration only -- NO training and NO new timing is done here.

Inputs (all stored results)
  research_benchmark/results/classical.json        FD convergence study: n, time (best of 3), rel-L2, L-inf
  research_benchmark/results/benchmark_runs.csv    PINN errors, 5 strategies x 5-10 seeds x 2 arms
  research_benchmark/results/timing_single.json    PINN time, measurement A: single-model reference driver
                                                   (CPU, 4 threads) while the machine was at load average
                                                   95-167; 400 Adam steps + 200 L-BFGS iterations timed and
                                                   scaled to 3000 iterations
  verify_research_benchmark/v_runs*.json           PINN time, measurement B: the single-network verification
                                                   code V-bench (CPU), full runs timed at load average about
                                                   20-45 on the same machine
  research_benchmark/results/runs/*_s*-*.json      PINN time, measurement C: the stacked GPU driver that produced
                                                   the benchmark errors; training time of a stack divided by the
                                                   number of networks in it (25 or 50), i.e. the amortised time
                                                   per network (load not recorded)

Outputs
  data/s5_hard_cost.json     every number of Table s5_hard:tab:fd and of the cost paragraph
  figures/s5_hard_cost.pdf   Figure s5_hard:fig:cost
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)  # repository root
RB = os.path.join(ROOT, "research_benchmark", "results")
VB = os.path.join(ROOT, "verify_research_benchmark")

import sys
sys.path.insert(0, HERE)
from figstyle import apply_style, panel_label, PROBLEMS, PNAME as LABEL, BLUE, ORANGE, INK, INK2, GRID  # noqa: E402
N_R = {"heat1d": 256, "laplace2d": 256, "wave2d": 512, "laplace3d": 512, "wave1d": 1024}
STRAT = {"grid": "grid-c", "random": "random", "resample": "resample", "sobol": "Sobol", "rad": "RAD"}


def geo(v):
    return float(np.exp(np.mean(np.log(v))))


def load():
    cl = json.load(open(os.path.join(RB, "classical.json")))
    tm = json.load(open(os.path.join(RB, "timing_single.json")))
    runs = list(csv.DictReader(open(os.path.join(RB, "benchmark_runs.csv"))))
    for r in runs:
        r["rel_l2"] = float(r["rel_l2"])
    V = {}
    for f in ("v_runs.json", "v_runs_extra.json", "v_runs_budget.json"):
        p = os.path.join(VB, f)
        if os.path.exists(p):
            V.update(json.load(open(p)))
    # measurement C: timing blocks of the benchmark stacks (one stack per problem; two for laplace2d)
    C = {}
    for f in sorted(glob.glob(os.path.join(RB, "runs", "*_s*-*.json"))):
        d = json.load(open(f))
        t = d["timing"]
        C.setdefault(d["runs"][0]["problem"], []).append(
            {"file": os.path.basename(f), "stack_size": t["batch_size"], "device": t["device"],
             "adam": t["t_adam_total"] / t["batch_size"],
             "adam_lbfgs": (t["t_adam_to_branch"] + t["t_lbfgs_total"]) / t["batch_size"]})
    return cl, tm, runs, V, C


def build(cl, tm, runs, V, C):
    out = {"note": "Regenerated from stored results by code/s5_hard_cost.py; no new timing.",
           "loadavg_reference_driver": {"start": tm["loadavg_start"], "end": tm["loadavg_end"]}, "problems": {}}
    for p in PROBLEMS:
        fd = [r for r in cl if r["problem"] == p]
        d = {}
        for arm in ("adam", "adam_lbfgs"):
            v = np.array([r["rel_l2"] for r in runs if r["problem"] == p and r["arm"] == arm])
            d[arm] = {"n_runs": int(len(v)), "geomean_all": geo(v), "min": float(v.min()), "max": float(v.max())}
            per = {s: geo([r["rel_l2"] for r in runs if r["problem"] == p and r["arm"] == arm and r["sampler"] == s])
                   for s in STRAT}
            best = min(per, key=per.get)
            d[arm]["best_strategy"], d[arm]["best_strategy_geomean"] = STRAT[best], per[best]
        target = d["adam_lbfgs"]["best_strategy_geomean"]
        ok = [r for r in fd if r["rel_l2"] <= target]
        c = min(ok, key=lambda r: r["time_s_best_of_3"])
        d["fd_cheapest_at_least_as_accurate"] = {"n": c["n"], "rel_l2": c["rel_l2"], "linf": c["linf"],
                                                 "time_s": c["time_s_best_of_3"], "grid_values": c["grid_values"]}
        d["fd_curve"] = [{"n": r["n"], "time_s": r["time_s_best_of_3"], "rel_l2": r["rel_l2"], "linf": r["linf"],
                          "observed_order": r["observed_order_rel_l2"]} for r in fd]
        # measurement A: reference driver under heavy load
        d["time_A_reference_driver_s"] = {"adam": tm[p]["est_run_s_adam"], "adam_lbfgs": tm[p]["est_run_s_adam_lbfgs"]}
        # measurement B: implementation of the verification code V-bench, runs at the benchmark's N_r only
        keys = [k for k in V if k.split("|")[0] == p and int(k.split("|")[3]) == N_R[p]]
        tb = {arm: sorted(V[k][arm]["time"] for k in keys) for arm in ("adam", "adam_lbfgs")}
        d["time_B_verifier_s"] = {arm: {"n_runs": len(t), "min": float(min(t)), "median": float(np.median(t)),
                                        "max": float(max(t))} for arm, t in tb.items()}
        # measurement C: amortised time per network in the GPU stacks that produced the benchmark errors
        d["time_C_stacked_gpu_per_network_s"] = {
            arm: {"min": min(x[arm] for x in C[p]), "max": max(x[arm] for x in C[p])} for arm in ("adam", "adam_lbfgs")}
        d["time_C_stacked_gpu_per_network_s"]["stacks"] = C[p]
        tfd = c["time_s_best_of_3"]
        d["time_ratio_pinn_over_fd"] = {
            "A_reference_driver": d["time_A_reference_driver_s"]["adam_lbfgs"] / tfd,
            "B_verifier_min": d["time_B_verifier_s"]["adam_lbfgs"]["min"] / tfd,
            "B_verifier_median": d["time_B_verifier_s"]["adam_lbfgs"]["median"] / tfd,
            "B_verifier_max": d["time_B_verifier_s"]["adam_lbfgs"]["max"] / tfd,
            "C_stacked_min": d["time_C_stacked_gpu_per_network_s"]["adam_lbfgs"]["min"] / tfd,
            "C_stacked_max": d["time_C_stacked_gpu_per_network_s"]["adam_lbfgs"]["max"] / tfd}
        d["fd_error_over_pinn_error"] = c["rel_l2"] / target
        d["time_A_over_B_median"] = d["time_A_reference_driver_s"]["adam_lbfgs"] / d["time_B_verifier_s"]["adam_lbfgs"]["median"]
        out["problems"][p] = d
    P = out["problems"]
    allB = [P[p]["time_B_verifier_s"]["adam_lbfgs"][k] for p in PROBLEMS for k in ("min", "max")]
    allA = [P[p]["time_A_reference_driver_s"]["adam_lbfgs"] for p in PROBLEMS]
    fdt = [P[p]["fd_cheapest_at_least_as_accurate"]["time_s"] for p in PROBLEMS]
    rB = [P[p]["time_ratio_pinn_over_fd"][k] for p in PROBLEMS for k in ("B_verifier_min", "B_verifier_max")]
    rA = [P[p]["time_ratio_pinn_over_fd"]["A_reference_driver"] for p in PROBLEMS]
    rC = [P[p]["time_ratio_pinn_over_fd"][k] for p in PROBLEMS for k in ("C_stacked_min", "C_stacked_max")]
    allC = [P[p]["time_C_stacked_gpu_per_network_s"]["adam_lbfgs"][k] for p in PROBLEMS for k in ("min", "max")]
    acc = [P[p]["fd_error_over_pinn_error"] for p in PROBLEMS]
    orders = {p: [r["observed_order"] for r in P[p]["fd_curve"] if r["observed_order"] is not None] for p in PROBLEMS}
    smooth = [o for p in PROBLEMS if p != "laplace3d" for o in orders[p]]
    out["summary"] = {"time_B_range_s": [min(allB), max(allB)], "time_A_range_s": [min(allA), max(allA)],
                      "fd_time_range_s": [min(fdt), max(fdt)], "ratio_B_range": [min(rB), max(rB)],
                      "ratio_A_range": [min(rA), max(rA)],
                      "time_C_range_s": [min(allC), max(allC)], "ratio_C_range": [min(rC), max(rC)],
                      "ratio_smallest_of_all_measurements": min(rA + rB + rC),
                      "ratio_largest_of_all_measurements": max(rA + rB + rC),
                      "fd_error_over_pinn_error_range": [min(acc), max(acc)],
                      "A_over_B_median_range": [min(P[p]["time_A_over_B_median"] for p in PROBLEMS),
                                                max(P[p]["time_A_over_B_median"] for p in PROBLEMS)],
                      "fd_observed_orders": orders,
                      "fd_order_range_smooth_problems": [min(smooth), max(smooth)], "fd_n_refinements_smooth": len(smooth)}
    return out


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    FD = INK2
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.1), sharex=True)
    axes = axes.ravel()
    for ax, p in zip(axes, PROBLEMS):
        d = out["problems"][p]
        t = [r["time_s"] for r in d["fd_curve"]]
        e = [r["rel_l2"] for r in d["fd_curve"]]
        ax.plot(t, e, color=FD, marker="o", ms=3, mec="white", mew=0.4, lw=1.1)
        ax.annotate(f"$n={d['fd_curve'][0]['n']}$", (t[0], e[0]), textcoords="offset points", xytext=(4, 2), fontsize=7, color=INK2)
        ax.annotate(f"$n={d['fd_curve'][-1]['n']}$", (t[-1], e[-1]), textcoords="offset points", xytext=(5, -3), fontsize=7, color=INK2)
        for arm, col, mk in (("adam", BLUE, "o"), ("adam_lbfgs", ORANGE, "s")):
            g, lo, hi = d[arm]["geomean_all"], d[arm]["min"], d[arm]["max"]
            tA = d["time_A_reference_driver_s"][arm]
            tB = d["time_B_verifier_s"][arm]
            tC = d["time_C_stacked_gpu_per_network_s"][arm]
            ax.plot([min(tB["min"], tC["min"]), tA], [g, g], color=col, lw=1.2, solid_capstyle="butt", zorder=3)   # span of the measurements
            ax.plot([0.5 * (tC["min"] + tC["max"])], [g], marker="D", ms=3.6, mfc=col, mec="white", mew=0.6, zorder=4)
            ax.plot([tB["median"], tB["median"]], [lo, hi], color=col, lw=0.9, zorder=3)          # min-max over runs
            ax.plot([tB["median"]], [g], marker=mk, ms=4.5, mfc=col, mec="white", mew=0.6, zorder=4)
            ax.plot([tA], [g], marker=mk, ms=4.5, mfc="white", mec=col, mew=1.0, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        panel_label(ax, f"({'abcde'[PROBLEMS.index(p)]}) {LABEL[p]}")
    for ax in axes[3:5]:
        ax.set_xlabel("wall-clock time of one solve (s)")
    axes[2].set_xlabel("wall-clock time of one solve (s)")
    axes[2].tick_params(labelbottom=True)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(r"relative $L^2$ error $\varepsilon_2$")
    axes[0].set_xlim(1e-4, 1.5e3)
    lg = axes[5]
    lg.axis("off")
    handles = [Line2D([], [], color=FD, marker="o", ms=3, mec="white", mew=0.4, lw=1.1, label="finite differences,\n$n$ doubled along the curve"),
               Line2D([], [], color=BLUE, marker="o", ms=4.5, mec="white", mew=0.6, lw=0, label="PINN, Adam"),
               Line2D([], [], color=ORANGE, marker="s", ms=4.5, mec="white", mew=0.6, lw=0, label=r"PINN, Adam$\to$L-BFGS"),
               Line2D([], [], color=INK2, marker="D", ms=3.6, mfc=INK2, mec="white", mew=0.6, lw=0,
                      label="diamond: per network in a GPU\nstack of 25 or 50 (measurement C)"),
               Line2D([], [], color=INK2, marker="s", ms=4.5, mfc=INK2, mec="white", mew=0.6, lw=0,
                      label="filled: single network, load 20–45\n(V-bench, measurement B)"),
               Line2D([], [], color=INK2, marker="s", ms=4.5, mfc="white", mec=INK2, mew=1.0, lw=0,
                      label="open: single network, load 95–167\n(reference driver, measurement A)")]
    lg.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.16, 0.5), handlelength=1.6, labelspacing=0.7, fontsize=7)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    path = os.path.join(ART, "figures", "s5_hard_cost.pdf")
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    cl, tm, runs, V, C = load()
    out = build(cl, tm, runs, V, C)
    json.dump(out, open(os.path.join(ART, "data", "s5_hard_cost.json"), "w"), indent=1)
    for p in PROBLEMS:
        d = out["problems"][p]
        c = d["fd_cheapest_at_least_as_accurate"]
        print(f"{LABEL[p]:5s} best PINN {d['adam_lbfgs']['best_strategy_geomean']:.3e} ({d['adam_lbfgs']['best_strategy']}) | "
              f"FD n={c['n']} {c['rel_l2']:.3e} linf {c['linf']:.3e} t={c['time_s']:.2e}s | "
              f"PINN t A {d['time_A_reference_driver_s']['adam_lbfgs']:.0f}s, B {d['time_B_verifier_s']['adam_lbfgs']} | "
              f"C {d['time_C_stacked_gpu_per_network_s']['adam_lbfgs']['min']:.1f}-{d['time_C_stacked_gpu_per_network_s']['adam_lbfgs']['max']:.1f}s | "
              f"ratio {d['time_ratio_pinn_over_fd']} | A/B {d['time_A_over_B_median']:.1f} | FD err / PINN err {d['fd_error_over_pinn_error']:.2f}")
    print(json.dumps(out["summary"], indent=1))
    plot(out)
