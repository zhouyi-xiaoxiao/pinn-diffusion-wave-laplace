"""Supplementary Section S8: hyper-parameter table, full error table, paired-test table, finite-difference
convergence table, cost table, and every number quoted in the prose of sections/S8_details.tex.

NO training and NO new timing.  Everything is read from stored result files or from the code and
configuration of the experiments (imported, or parsed and checked by assertions, so that a change
of the code makes this script fail instead of silently printing a stale table).

Inputs (relative to the root of the repository)
  research_benchmark/pinnbench/{problems,core,samplers}.py, scripts/run_benchmark.py   (imported)
  research_benchmark/scripts/{validate_batched,time_single,run_budget_sweep,run_oneface}.py (parsed)
  research_benchmark/results/{benchmark_runs.csv,benchmark_summary.csv,stats.json,classical.csv,
      timing_single.json,validate_batched.json,budget_sweep.json,oneface_laplace3d.json,runs/*.json,
      runs_superseded/*.json}
  research_benchmark/checks/verify_losses.out
  research_highdim/scripts/{hd_core.py (imported), bench_cost.py (parsed)}
  research_highdim/results/{runs*.jsonl,runs.csv}
  verify_research_benchmark/{v_runs.json,v_runs_extra.json,v_runs_budget.json,v_summary.out}
  verify_research_highdim/results/v_determinism.txt
  data/checks/study_records.json (benchmark: validation_predates_single_pass_loss)
  code/{s4_lowdim_timeslices,s5_hard_wave2d_long,s5_hard_laplace3d_smooth,
      s6_highdim_long_d20,s7_pitfalls,s4_lowdim_gridcontrol}.py                         (parsed)
Outputs
  data/sC_details_numbers.json   every derived number
  data/sC_details_tables.tex     the five LaTeX tables
  sections/S8_details.tex        the blocks between "%% BEGIN GENERATED <name>" and
                                         "%% END GENERATED <name>" are replaced (if the file exists)

Usage:  python code/sC_details_tables.py
"""
import ast
import csv
import glob
import inspect
import json
import os
import platform
import re
import sys
from importlib.metadata import version

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)  # repository root
RB = os.path.join(ROOT, "research_benchmark")
RH = os.path.join(ROOT, "research_highdim")
VB = os.path.join(ROOT, "verify_research_benchmark")
VH = os.path.join(ROOT, "verify_research_highdim")
sys.path.insert(0, RB)
sys.path.insert(0, os.path.join(RB, "scripts"))
sys.path.insert(0, os.path.join(RH, "scripts"))

PROBLEMS = ["heat1d", "laplace2d", "wave1d", "wave2d", "laplace3d"]
PMAC = {"heat1d": r"\PH", "laplace2d": r"\PLtwo", "wave1d": r"\PWone", "wave2d": r"\PWtwo", "laplace3d": r"\PLthree"}
STRATS = ["grid", "random", "resample", "sobol", "rad"]
SMAC = {"grid": r"\strat{grid-c}", "random": r"\strat{random}", "resample": r"\strat{resample}",
        "sobol": r"\strat{Sobol}", "rad": r"\strat{RAD}", "nodegrid": r"\strat{grid-n}"}
ARMS = ["adam", "adam_lbfgs"]
AMAC = {"adam": r"\Adam", "adam_lbfgs": r"\AdamLBFGS"}
FD_SCHEME = {"heat1d": "Crank--Nicolson, $\\Delta t=\\Delta x=1/n$",
             "wave1d": "leapfrog, Courant number 0.5, $\\Delta x=1/n$",
             "wave2d": "leapfrog, Courant number 0.5, $\\Delta x=2/n$",
             "laplace2d": "five-point Laplacian, $\\Delta x=\\pi/n$",
             "laplace3d": "seven-point Laplacian, $\\Delta x=\\pi/n$"}

NUM = {}          # every number used in the prose
TEX = {}          # name -> LaTeX block


# ------------------------------------------------------------------------------- helpers
def rd(*p):
    return open(os.path.join(ROOT, *p)).read()


def rd_art(*p):
    return open(os.path.join(ART, *p)).read()


def rj(*p):
    return json.load(open(os.path.join(ROOT, *p)))


def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, float)))))


def e3(x, digits=2):
    """2.98e-2 typeset in math mode (three significant digits by default)."""
    m, e = f"{x:.{digits}e}".split("e")
    return f"${m}\\mathrm{{e}}{{{'-' if int(e) < 0 else ''}}}{abs(int(e))}$"


def sci(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    return f"$\\sci{{{m}}}{{{int(e)}}}$"


def thousands(n):
    """Thin-space grouping for five digits and more (17\\,025), none for four (3000), as in the article."""
    return f"{n:,}".replace(",", r"\,") if n >= 10000 else str(n)


def fmt_p(p):
    if p is None:
        return "n/a"
    if abs(p - 0.0625) < 1e-12:
        return "0.0625"
    return f"{p:.3f}"


def seed_range(seeds):
    s = sorted(set(int(x) for x in seeds))
    if len(s) > 2 and s == list(range(s[0], s[-1] + 1)):
        return f"{s[0]}--{s[-1]}"
    return ", ".join(str(x) for x in s)


def consts(path):
    """Module-level constants of a script, without executing it."""
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            val = eval(compile(ast.Expression(node.value), path, "eval"), {"__builtins__": {}},
                       {"list": list, "range": range})
        except Exception:
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                out[tgt.id] = val
            elif isinstance(tgt, ast.Tuple) and isinstance(val, tuple) and len(val) == len(tgt.elts):
                for t, v in zip(tgt.elts, val):
                    if isinstance(t, ast.Name):
                        out[t.id] = v
    return out


def need(src, *snippets):
    for s in snippets:
        assert s in src, f"expected code fragment not found: {s!r}"


# =========================================================================================
# 1. configuration read from the code
# =========================================================================================
import torch  # noqa: E402  (after the path set-up; only needed to instantiate the problem classes)
from pinnbench import core, samplers  # noqa: E402
from pinnbench.problems import make_problem  # noqa: E402
import run_benchmark  # noqa: E402  (CONFIG, SAMPLERS; has a __main__ guard)
import hd_core  # noqa: E402

CONFIG = run_benchmark.CONFIG
assert run_benchmark.SAMPLERS == STRATS
tm = rj("research_benchmark", "results", "timing_single.json")

LOW = {}
for name in PROBLEMS:
    P = make_problem(name)
    sizes = [P.d] + list(P.hidden) + [1]
    n_par = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
    assert n_par == tm[name]["n_params"], (name, n_par)
    assert len(set(P.hidden)) == 1
    E = P.eval_points()
    shape = [len(np.unique(np.round(E[:, k], 12))) for k in range(P.d)]
    assert int(np.prod(shape)) == E.shape[0]
    blocks = [e - a for a, e in P._ranges]
    LOW[name] = {"d_in": P.d, "hidden": list(P.hidden), "n_params": n_par, "n_r": CONFIG[name]["n_r"],
                 "n_adam": CONFIG[name]["n_adam"], "branch_at": CONFIG[name]["branch_at"],
                 "constraint_blocks": blocks, "n_constraint_points": int(sum(blocks)),
                 "n_loss_terms": len(P._terms), "eval_shape": shape, "eval_points": int(E.shape[0])}
    assert CONFIG[name]["n_r"] == tm[name]["n_r"]
NUM["lowdim_config"] = LOW
NUM["adam_lr"] = core.ADAM_LR
NUM["lbfgs_memory"] = core.LBFGS_MEMORY
sig = inspect.signature(core.batched_lbfgs).parameters
NUM["stacked_lbfgs"] = {"armijo_c1": sig["c1"].default, "max_backtracks": sig["max_ls"].default,
                        "memory": sig["memory"].default}
src_core = rd("research_benchmark", "pinnbench", "core.py")
need(src_core, 'line_search_fn="strong_wolfe"', "tolerance_grad=1e-9", "tolerance_change=1e-12",
     "history_size=LBFGS_MEMORY", "good = ys > 1e-10 * y_vec.norm(dim=1) * s_vec.norm(dim=1)",
     "t = torch.where(ok, t, 0.5 * t)", "torch.clamp(1.0 / g.abs().sum(1).clamp_min(1e-30), max=1.0)")
NUM["torch_lbfgs"] = {"line_search": "strong_wolfe", "tolerance_grad": 1e-9, "tolerance_change": 1e-12}
sig = inspect.signature(samplers.RADSampler.__init__).parameters
NUM["rad"] = {k: sig[k].default for k in ("period", "pool_factor", "k", "c")}
src_s = rd("research_benchmark", "pinnbench", "samplers.py")
need(src_s, "np.random.default_rng([seed, 7919])")

# seeds, stack sizes and devices actually used (from the stored runs)
bruns = list(csv.DictReader(open(os.path.join(RB, "results", "benchmark_runs.csv"))))
for r in bruns:
    r["seed"] = int(r["seed"]); r["rel_l2"] = float(r["rel_l2"]); r["linf"] = float(r["linf"])
assert len(bruns) == 350
RAW = {}
for f in sorted(glob.glob(os.path.join(RB, "results", "runs", "*_s*-*.json"))):
    RAW[os.path.basename(f)] = json.load(open(f))
STACK = {p: [] for p in PROBLEMS}
for fn, d in RAW.items():
    p = d["runs"][0]["problem"]
    assert d["config"] == CONFIG[p], fn
    STACK[p].append({"file": fn, "batch_size": d["timing"]["batch_size"], "device": d["timing"]["device"],
                     "wall_total_s": d["timing"]["wall_total_s"],
                     # measurement C: training time of the Adam -> L-BFGS arm of the stack per network
                     "C_adam_lbfgs_per_network_s": (d["timing"]["t_adam_to_branch"] + d["timing"]["t_lbfgs_total"]) / d["timing"]["batch_size"],
                     "lbfgs_fevals_per_iter": d["timing"]["lbfgs_batched_fevals"] / (d["config"]["n_adam"] - d["config"]["branch_at"])})
for p in PROBLEMS:
    LOW[p]["seeds"] = sorted({r["seed"] for r in bruns if r["problem"] == p})
    LOW[p]["stacks"] = STACK[p]
    assert all(s["device"] == "mps" for s in STACK[p])
NUM["benchmark_stack_wall_s"] = {"min": min(s["wall_total_s"] for p in PROBLEMS for s in STACK[p]),
                                 "max": max(s["wall_total_s"] for p in PROBLEMS for s in STACK[p]),
                                 "sum": sum(s["wall_total_s"] for p in PROBLEMS for s in STACK[p]),
                                 "n_stacks": sum(len(STACK[p]) for p in PROBLEMS)}
NUM["n_networks"] = len(bruns) // 2

# budget sweep
bs = rj("research_benchmark", "results", "budget_sweep.json")
src_bs = rd("research_benchmark", "scripts", "run_budget_sweep.py")
need(src_bs, 'SAMPLERS = ["grid", "random", "sobol"]', "for seed in range(5)")
src_chain = rd("research_benchmark", "scripts", "chain_after_main.sh")
need(src_chain, "run_budget_sweep.py heat1d 64,1024 mps")
SWEEP = {"problems": sorted({r["problem"] for r in bs}), "n_r": sorted({r["n_r"] for r in bs}),
         "strategies": sorted({r["sampler"] for r in bs}), "seeds": sorted({r["seed"] for r in bs}),
         "new_n_r": [64, 1024], "stack": 15, "device": "mps"}
assert SWEEP["problems"] == ["heat1d"] and SWEEP["n_r"] == [64, 256, 1024] and SWEEP["seeds"] == [0, 1, 2, 3, 4]
NUM["budget_sweep_config"] = SWEEP

# validation of the stacked driver, single-run timing
src_v = rd("research_benchmark", "scripts", "validate_batched.py")
need(src_v, '[("heat1d", 600), ("laplace2d", 600)]', "for seed in (0, 1)",
     'run_batched(name, specs, 1024, n_adam, n_adam // 2, device="cpu"',
     "single(name, ob[\"sampler\"], ob[\"seed\"], 1024, n_adam, n_adam // 2)", "torch.set_num_threads(4)")
src_t = rd("research_benchmark", "scripts", "time_single.py")
need(src_t, 'run_single(P, "sobol", 0, cfg["n_r"], 400, 200', "torch.set_num_threads(4)")

# the demonstration of Section 7.2 (data on one face only)
src_of = rd("research_benchmark", "scripts", "run_oneface.py")
need(src_of, "MLP(3, [100, 100])", "ge = 21", "torch.optim.AdamW(net.parameters(), lr=1e-3)", "for ep in range(2000)",
     "SEEDS = [0, 1, 2, 3, 4]")
of = rj("research_benchmark", "results", "oneface_laplace3d.json")["oneface_laplace3d"]
NUM["oneface"] = {"seeds": [r["seed"] for r in of["rows"]], "device": of["device"], "grid_points": 21 ** 3,
                  "constrained": 21 ** 2,
                  "adamw_weight_decay_default": inspect.signature(torch.optim.AdamW).parameters["weight_decay"].default}

# the further experiments of Sections 4.3, 4.5, 5.1-5.3, 6.3 and 7.1 (constants of the scripts)
c_n1 = consts(os.path.join(HERE, "s5_hard_wave2d_long.py"))
c_n8 = consts(os.path.join(HERE, "s5_hard_wave1d_long.py"))
need(rd_art("code", "s5_hard_wave1d_long.py"), "core.run_single(P, SAMPLER, seed, N_R, total, BRANCH_AT", "torch.set_num_threads(2)")
c_n2 = consts(os.path.join(HERE, "s5_hard_laplace3d_smooth.py"))
c_n5 = consts(os.path.join(HERE, "s6_highdim_long_d20.py"))
c_n6 = consts(os.path.join(HERE, "s4_lowdim_timeslices.py"))
need(rd_art("code", "s5_hard_wave2d_long.py"), "core.run_single(P, SAMPLER, seed, N_R, total, BRANCH_AT", "torch.set_num_threads(2)")
need(rd_art("code", "s5_hard_laplace3d_smooth.py"), 'core.run_batched(problem_name, specs, N_R, N_ADAM, BRANCH_AT, device="cpu"')
need(rd_art("code", "s4_lowdim_timeslices.py"), "core.run_single(problem, strategy, seed, N_R, N_ADAM, BRANCH_AT", 'make_problem(PROBLEM, device="cpu")')
c_n3 = consts(os.path.join(HERE, "s7_pitfalls.py"))
c_n7 = consts(os.path.join(HERE, "s4_lowdim_gridcontrol.py"))
need(rd_art("code", "s7_pitfalls.py"), "torch.set_num_threads(1)", "torch.optim.Adam(net.parameters(), lr=LR)",
     "self.fc1 = torch.nn.Linear(3, 50)", "torch.rand(N_TRAIN, 3)")
need(rd_art("code", "s4_lowdim_gridcontrol.py"), "run_batched(PROBLEM, specs, n_r, N_ADAM, BRANCH_AT, device=device",
     '"A": (["random", "grid", "gridn", "gridt0"], M * M)', '"B": (["gridplus0", "gridplusm"], M * M + M)')
NUM["further_experiments"] = {
    "pitfalls_autograd": {k: c_n3[k] for k in ("SEED", "N_TRAIN", "PASSES", "BATCH", "LR")},
    "gridcontrol": {k: c_n7[k] for k in ("PROBLEM", "N_ADAM", "BRANCH_AT", "M", "SEEDS")},
    "wave2d_long": {k: c_n1[k] for k in ("N_R", "BRANCH_AT", "TOTAL_DEFAULT", "SAMPLER", "SEEDS")},
    "wave1d_long": {k: c_n8[k] for k in ("N_R", "BRANCH_AT", "TOTAL_DEFAULT", "SAMPLER", "SEEDS")},
    "laplace3d_smooth": {k: c_n2[k] for k in ("N_R", "N_ADAM", "BRANCH_AT", "SAMPLER", "SEEDS")},
    "highdim_long_d20": {k: c_n5[k] for k in ("D", "ITERS", "SEEDS", "PROBLEMS", "METHODS")},
    "timeslices": {k: c_n6[k] for k in ("PROBLEM", "N_R", "N_ADAM", "BRANCH_AT", "STRATEGIES", "SEEDS")}}
p_n1 = os.path.join(ART, "data", "s5_hard_wave2d_long.json")
if os.path.exists(p_n1):          # the budget actually used, if the experiment has been run
    NUM["further_experiments"]["wave2d_long"]["total_iterations_used"] = json.load(open(p_n1))["summary"]["total_iterations"]

# the separately written re-implementation of the low-dimensional benchmark (V-bench)
V = {}
for f in ("v_runs.json", "v_runs_extra.json", "v_runs_budget.json"):
    V.update(rj("verify_research_benchmark", f))
vkeys = [k.split("|") for k in V]
VER = {"n_runs": len(V), "seeds": {}, "lbfgs_better": int(sum(v["adam_lbfgs"]["rel_l2"] < v["adam"]["rel_l2"] for v in V.values())),
       "n_lbfgs_stopped_early": int(sum(v["adam_lbfgs"]["lbfgs_iters"] < 1500 for v in V.values())),
       "min_lbfgs_iters": int(min(v["adam_lbfgs"]["lbfgs_iters"] for v in V.values())),
       "sum_wall_s": float(sum(v["wall"] for v in V.values()))}
for p in PROBLEMS:
    VER["seeds"][p] = sorted({int(k[2]) for k in vkeys if k[0] == p})
VER["kinds_heat"] = sorted({k[1] for k in vkeys if k[0] == "heat1d"})
VER["n_r_heat"] = sorted({int(k[3]) for k in vkeys if k[0] == "heat1d"})
src_vp = rd("verify_research_benchmark", "vpinn.py")
need(src_vp, "torch.set_num_threads(4)", "strong_wolfe", "max_iter=n_l, max_eval=int(1.5 * n_l)", "tolerance_grad=1e-9, tolerance_change=1e-12")
VER["n_early_stops_below_eval_cap"] = int(sum(v["adam_lbfgs"]["lbfgs_iters"] < 1500 and v["adam_lbfgs"]["nfev"] < 2250 for v in V.values()))
# the runs that enter measurement B of the cost table: those at the N_r of the benchmark
B_KEYS = [k for k in V if int(k.split("|")[3]) == CONFIG[k.split("|")[0]]["n_r"]]
B_EARLY = [k for k in B_KEYS if V[k]["adam_lbfgs"]["lbfgs_iters"] < 1500]
VER["B"] = {"n_runs": len(B_KEYS), "n_stopped_early": len(B_EARLY),
            "stopped_early_by_problem": {p: sum(k.split("|")[0] == p for k in B_EARLY) for p in PROBLEMS},
            "min_lbfgs_iters": int(min(V[k]["adam_lbfgs"]["lbfgs_iters"] for k in B_KEYS))}
assert all(V[k]["adam_lbfgs"]["nfev"] < 2250 for k in B_EARLY)          # stopped on a tolerance, not on the evaluation cap
NUM["verifier_lowdim"] = VER

# the d-dimensional study
sig = inspect.signature(hd_core.train).parameters
HD = {k: sig[k].default for k in ("iters", "n_int", "n_bd", "lr", "width", "depth", "lap_impl")}
sig = inspect.signature(hd_core.fixed_sets).parameters
HD.update({k: sig[k].default for k in ("n_test", "n_val")}); HD["n_test_bd"] = sig["n_bd"].default
src_hd = rd("research_highdim", "scripts", "hd_core.py")
need(src_hd, "torch.set_num_threads(1)", "DTYPE = torch.float32", "milestones=[iters // 2, (3 * iters) // 4], gamma=0.1",
     "torch.Generator().manual_seed(1_000 + seed)", "manual_seed(10_000 + d)", "manual_seed(20_000 + d)")
for d_ in (2, 20):
    n_ = sum(p.numel() for p in hd_core.MLP(d_).parameters())
    assert n_ == 64 * d_ + 8449
hrecs = []
for f in sorted(glob.glob(os.path.join(RH, "results", "runs*.jsonl"))):
    hrecs += [json.loads(l) for l in open(f) if l.strip()]
HD["n_runs"] = len(hrecs)
HD["phases"] = {}
for ph in sorted({r["phase"] for r in hrecs}):
    rr = [r for r in hrecs if r["phase"] == ph]
    HD["phases"][ph] = {"n": len(rr), "d": sorted({r["d"] for r in rr}), "seeds": sorted({r["seed"] for r in rr}),
                        "iters": sorted({r["iters"] for r in rr}),
                        "weights": {f"{p}/{m}": sorted({r["w"] for r in rr if r["problem"] == p and r["method"] == m})
                                    for p in ("laplace", "poisson") for m in ("pinn", "ritz")},
                        "iters_by_problem": {p: sorted({r["iters"] for r in rr if r["problem"] == p}) for p in ("laplace", "poisson")}}
HD["sweep_pinn_runs_nested"] = int(sum(1 for r in hrecs if r["phase"] == "sweep" and r["method"] == "pinn" and r.get("lap_impl") != "forward"))
HD["sum_wall_s"] = float(sum(r["wall_s"] for r in hrecs))
HD["sum_cpu_s"] = float(sum(r["cpu_time_s"] for r in hrecs if r.get("cpu_time_s") is not None))
HD["n_without_cpu_timer"] = int(sum(1 for r in hrecs if r.get("cpu_time_s") is None))
HD["max_wall_s"] = float(max(r["wall_s"] for r in hrecs))
mw = HD["phases"]["main"]["weights"]
assert mw == {"laplace/pinn": [1000.0], "laplace/ritz": [100.0], "poisson/pinn": [1000.0], "poisson/ritz": [1.0]}, mw
assert HD["phases"]["main"]["d"] == [2, 3, 5, 10, 20] and HD["phases"]["main"]["seeds"] == [0, 1, 2]
assert HD["phases"]["sweep"]["d"] == [5] and HD["phases"]["sweep"]["seeds"] == [100]
src_bc = rd("research_highdim", "scripts", "bench_cost.py")
need(src_bc, "DS = [2, 3, 5, 10, 20, 50, 100]", "for _ in range(5):", "for _ in range(7):", "for _ in range(10):",
     "time.process_time()", "torch.set_num_threads(1)")
NUM["highdim_config"] = HD
det = [l for l in rd("verify_research_highdim", "results", "v_determinism.txt").splitlines() if l.strip()]
NUM["bit_for_bit"] = {"n_cells": len(det), "n_identical": sum("reldiff 0.00e+00" in l for l in det)}

# software, read from the environment that ran the experiments
NUM["software"] = {"python": platform.python_version(), **{p: version(p) for p in ("torch", "numpy", "scipy", "matplotlib")},
                   "machine": platform.machine(), "os": platform.platform(), "cpu_count": os.cpu_count()}


# =========================================================================================
# 2. validation of the stacked driver
# =========================================================================================
vb = rj("research_benchmark", "results", "validate_batched.json")
VAL = {}
for p in ("heat1d", "laplace2d"):
    rows = vb[p]
    d = {"n": len(rows), "seeds": sorted({r["seed"] for r in rows}), "strategies": sorted({r["sampler"] for r in rows})}
    for it in (1, 50, 100, 300):
        a = np.array([r[f"adam_loss_reldiff_it{it}"] for r in rows])
        d[f"adam_reldiff_it{it}_max"] = float(a.max())
        d[f"adam_reldiff_it{it}_median"] = float(np.median(a))
    for arm, key in (("lbfgs", "lbfgs_rel_l2"), ("adam", "adam_rel_l2")):
        q = np.array([r[key + "_batched"] / r[key + "_single"] for r in rows])
        d[arm + "_ratio_geo"], d[arm + "_ratio_min"], d[arm + "_ratio_max"] = geo(q), float(q.min()), float(q.max())
    d["fevals_stack"] = sorted({r["batched_lbfgs_fevals"] for r in rows})
    d["fevals_single"] = [min(r["single_lbfgs_fevals"] for r in rows), max(r["single_lbfgs_fevals"] for r in rows)]
    d["single_lbfgs_iters"] = sorted({r["single_lbfgs_iters"] for r in rows})
    VAL[p] = d
NUM["validation"] = VAL
vl = rd("research_benchmark", "checks", "verify_losses.out")
NUM["loss_assembly"] = {"single_vs_termwise_max": [float(x) for x in re.findall(r"term-by-term, max relative difference over \d+ terms = ([0-9.e+-]+)", vl)],
                        "terms": [int(x) for x in re.findall(r"over (\d+) terms", vl)],
                        "stacked_vs_single_max": [float(x) for x in re.findall(r"stacked \(S=3\) vs single total loss, max relative difference = ([0-9.e+-]+)", vl)]}
assert len(NUM["loss_assembly"]["single_vs_termwise_max"]) == 5 and max(NUM["loss_assembly"]["single_vs_termwise_max"]) == 0.0
assert len(NUM["loss_assembly"]["stacked_vs_single_max"]) == 5 and max(NUM["loss_assembly"]["stacked_vs_single_max"]) == 0.0
# caveat recorded in the working records of the benchmark (data/checks/study_records.json): the Adam / L-BFGS validation of the stacked driver was run with the earlier
# loss assembly (separate forward passes); the single-pass assembly was checked afterwards by verify_losses.py
CAVEAT = "The validation was done before the loss was re-assembled into a single forward pass"
_rec = json.loads(rd_art("data", "checks", "study_records.json"))["benchmark"]["validation_predates_single_pass_loss"]
assert _rec["value"] is True and _rec["statement"] == CAVEAT
NUM["validation_predates_single_pass_loss"] = True

# separately written single-model implementation against the stacked driver at production settings
T = {(r["problem"], r["sampler"], r["seed"], r["arm"]): r["rel_l2"] for r in bruns}
q = []
for k, v in V.items():
    p, kind, s, n, mode = k.split("|")
    if mode != "fixed" or kind == "nodegrid" or int(n) != CONFIG[p]["n_r"]:
        continue
    q.append(v["adam_lbfgs"]["rel_l2"] / T[(p, kind, int(s), "adam_lbfgs")])
NUM["independent_vs_stacked"] = {"n": len(q), "geo": geo(q), "min": float(min(q)), "max": float(max(q))}
m = re.search(r"identical configs: n=(\d+) geo ratio ([0-9.]+), range ([0-9.]+)-([0-9.]+)", rd("verify_research_benchmark", "v_summary.out"))
assert (int(m.group(1)), m.group(2), m.group(3), m.group(4)) == (len(q), f"{geo(q):.2f}", f"{min(q):.2f}", f"{max(q):.2f}"), m.groups()
m = re.search(r"better than Adam arm in (\d+) of (\d+)", rd("verify_research_benchmark", "v_summary.out"))
assert (int(m.group(1)), int(m.group(2))) == (VER["lbfgs_better"], VER["n_runs"])

# the superseded first heat block: same seeds, mathematically identical loss assembled differently
old = json.load(open(glob.glob(os.path.join(RB, "results", "runs_superseded", "heat1d_s0-5*.json"))[0]))
new = {(r["sampler"], r["seed"]): r for r in RAW["heat1d_s0-10.json"]["runs"]}
SUP = {"n": len(old["runs"]), "device": old["timing"]["device"], "batch_size": old["timing"]["batch_size"]}
for arm in ARMS:
    qq = np.array([o[arm]["rel_l2"] / new[(o["sampler"], o["seed"])][arm]["rel_l2"] for o in old["runs"]])
    SUP[arm] = {"ratio_geo": geo(qq), "ratio_min": float(qq.min()), "ratio_max": float(qq.max()),
                "n_identical": int((qq == 1.0).sum())}
l1 = [abs(o["adam"]["curve"][0]["loss"] - new[(o["sampler"], o["seed"])]["adam"]["curve"][0]["loss"]) for o in old["runs"]]
assert all(o["adam"]["curve"][0]["it"] == 1 for o in old["runs"])
SUP["n_identical_loss_at_iteration_1"] = int(sum(x == 0.0 for x in l1))
NUM["superseded_heat_block"] = SUP


# =========================================================================================
# 3. tables
# =========================================================================================
# ---- full error table (T1 with L-infinity and geometric means) --------------------------
summ = {(r["problem"], r["arm"], r["sampler"]): r for r in csv.DictReader(open(os.path.join(RB, "results", "benchmark_summary.csv")))}
L = [r"\begin{tabular}{@{}l rrr rr @{\hspace{10pt}} rrr rr@{}}", r"\toprule",
     r" & \multicolumn{5}{c}{\Adam} & \multicolumn{5}{c}{\AdamLBFGS} \\", r"\cmidrule(lr){2-6}\cmidrule(l){7-11}",
     r" & \multicolumn{3}{c}{$\errL$} & \multicolumn{2}{c}{$\errI$} & \multicolumn{3}{c}{$\errL$} & \multicolumn{2}{c}{$\errI$} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(l){10-11}",
     r"Strategy & mean & s.d. & geo. & mean & s.d. & mean & s.d. & geo. & mean & s.d. \\"]
for p in PROBLEMS:
    n_seeds = int(summ[(p, "adam", "grid")]["rel_l2_n"])
    assert n_seeds == len(LOW[p]["seeds"])
    L.append(r"\midrule")
    L.append(rf"\multicolumn{{11}}{{@{{}}l}}{{{PMAC[p]}: $\Nr={LOW[p]['n_r']}$, {n_seeds} seeds}} \\")
    for s in STRATS:
        cells = []
        for arm in ARMS:
            r = summ[(p, arm, s)]
            v = np.array([x["rel_l2"] for x in bruns if x["problem"] == p and x["arm"] == arm and x["sampler"] == s])
            w = np.array([x["linf"] for x in bruns if x["problem"] == p and x["arm"] == arm and x["sampler"] == s])
            # the summary file must agree with the raw per-run file
            assert abs(v.mean() / float(r["rel_l2_mean"]) - 1) < 1e-9 and abs(v.std(ddof=1) / float(r["rel_l2_std"]) - 1) < 1e-9
            assert abs(geo(v) / float(r["rel_l2_geo_mean"]) - 1) < 1e-9 and abs(w.mean() / float(r["linf_mean"]) - 1) < 1e-9
            assert abs(w.std(ddof=1) / float(r["linf_std"]) - 1) < 1e-9
            cells += [e3(float(r[k])) for k in ("rel_l2_mean", "rel_l2_std", "rel_l2_geo_mean", "linf_mean", "linf_std")]
        L.append(f"{SMAC[s]} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
TEX["sC_details:tab:full"] = "\n".join(L)
NUM["single_run_ranges"] = {p: {arm: [min(x["rel_l2"] for x in bruns if x["problem"] == p and x["arm"] == arm),
                                      max(x["rel_l2"] for x in bruns if x["problem"] == p and x["arm"] == arm)] for arm in ARMS}
                            for p in PROBLEMS}

# ---- paired tests (T3 complete) and the optimiser comparison per seed ---------------------
st = rj("research_benchmark", "results", "stats.json")


def series(p, s, arm):
    rr = sorted((x for x in bruns if x["problem"] == p and x["sampler"] == s and x["arm"] == arm), key=lambda x: x["seed"])
    return np.array([x["rel_l2"] for x in rr])


L = [r"\begin{tabular}{@{}l llll@{}}", r"\toprule",
     r"Problem & \strat{grid-c} & \strat{resample} & \strat{Sobol} & \strat{RAD} \\"]
PAIR = {}
for arm in ("adam_lbfgs", "adam"):
    L.append(r"\midrule")
    L.append(rf"\multicolumn{{5}}{{@{{}}l}}{{Arm {AMAC[arm]}}} \\")
    for p in PROBLEMS:
        cells = []
        base = series(p, "random", arm)
        for s in ("grid", "resample", "sobol", "rad"):
            o = st[p]["sampler_vs_random"][f"{arm}:{s}"]
            dlog = np.log(series(p, s, arm)) - np.log(base)
            pv = float(stats.wilcoxon(dlog).pvalue)
            # stats.json must agree with a recomputation from the raw per-run file
            assert abs(float(np.exp(dlog.mean())) / o["geo_mean_ratio_to_random"] - 1) < 1e-9
            assert int((dlog < 0).sum()) == o["n_seeds_better"] and abs(pv - o["wilcoxon_p_two_sided"]) < 1e-12
            assert o["n"] == len(base) == len(LOW[p]["seeds"])
            PAIR[f"{p}:{arm}:{s}"] = {"ratio": o["geo_mean_ratio_to_random"], "better": o["n_seeds_better"], "n": o["n"], "p": pv}
            cells.append(f"{o['geo_mean_ratio_to_random']:.2f} ({o['n_seeds_better']}/{o['n']}; {fmt_p(pv)})")
        L.append(f"{PMAC[p]} & " + " & ".join(cells) + " \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
TEX["sC_details:tab:paired"] = "\n".join(L)
NUM["paired"] = PAIR

L = [r"\begin{tabular}{@{}lr rrr cc l@{}}", r"\toprule",
     r" & & \multicolumn{2}{c}{geometric mean of $\errL$} & & \multicolumn{2}{c}{\AdamLBFGS\ better in} & Wilcoxon $p$ \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
     r"Problem & Seeds & \Adam & \AdamLBFGS & Ratio & runs & seeds & (per seed) \\", r"\midrule"]
OPT = {}
for p in PROBLEMS:
    o = st[p]["adam_vs_adam_lbfgs"]
    seeds = LOW[p]["seeds"]
    A = np.array([[series(p, s, arm) for s in STRATS] for arm in ARMS])      # axes: arm, strategy, seed
    per_seed = np.log(A).mean(1)                       # geometric mean over the five strategies of a seed
    dlog = per_seed[1] - per_seed[0]
    pv = float(stats.wilcoxon(dlog).pvalue)
    n_runs_better = int((A[1] < A[0]).sum())
    assert n_runs_better == o["n_pairs_lbfgs_better"] and A[0].size == o["n_pairs"]
    assert abs(geo(A[0]) / o["adam_geo_mean_rel_l2"] - 1) < 1e-9 and abs(geo(A[1]) / o["adam_lbfgs_geo_mean_rel_l2"] - 1) < 1e-9
    OPT[p] = {"geo_adam": geo(A[0]), "geo_adam_lbfgs": geo(A[1]), "ratio": o["geo_mean_ratio_lbfgs_over_adam"],
              "runs_better": n_runs_better, "n_runs": int(A[0].size), "seeds_better": int((dlog < 0).sum()),
              "n_seeds": len(seeds), "p_per_seed": pv}
    L.append(f"{PMAC[p]} & {len(seeds)} & {e3(geo(A[0]))} & {e3(geo(A[1]))} & {o['geo_mean_ratio_lbfgs_over_adam']:.3f} & "
             f"{n_runs_better}/{A[0].size} & {int((dlog < 0).sum())}/{len(seeds)} & {fmt_p(pv)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
TEX["sC_details:tab:optim"] = "\n".join(L)
NUM["optimiser"] = OPT
NUM["optimiser_total_runs_better"] = [sum(o["runs_better"] for o in OPT.values()), sum(o["n_runs"] for o in OPT.values())]

# ---- finite-difference convergence (T4) ----------------------------------------------------
cl = list(csv.DictReader(open(os.path.join(RB, "results", "classical.csv"))))
L = [r"\begin{tabular}{@{}rrrrrr@{\hspace{18pt}}rrrrrr@{}}", r"\toprule",
     r"$n$ & values & $\errL$ & $\errI$ & order & time (s) & $n$ & values & $\errL$ & $\errI$ & order & time (s) \\"]


def fd_rows(p):
    out = []
    for r in (x for x in cl if x["problem"] == p):
        od = "" if r["observed_order_rel_l2"] == "" else f"{float(r['observed_order_rel_l2']):.2f}"
        out.append([str(int(r["n"])), thousands(int(r["grid_values"])), e3(float(r["rel_l2"])), e3(float(r["linf"])), od,
                    e3(float(r["time_s_best_of_3"]))])
    return out


def fd_block(pl, pr):
    """Two problems side by side."""
    a, b = fd_rows(pl), (fd_rows(pr) if pr else [])
    hl = rf"\multicolumn{{6}}{{@{{}}l}}{{{PMAC[pl]}: {FD_SCHEME[pl]}}}"
    hr = rf"\multicolumn{{6}}{{@{{}}l}}{{{PMAC[pr]}: {FD_SCHEME[pr]}}}" if pr else r"\multicolumn{6}{@{}l}{}"
    out = [r"\midrule", hl + " & " + hr + r" \\"]
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else [""] * 6
        y = b[i] if i < len(b) else [""] * 6
        out.append(" & ".join(x + y) + r" \\")
    return out


L += fd_block("heat1d", "laplace2d") + fd_block("wave1d", "wave2d") + fd_block("laplace3d", None)
L += [r"\bottomrule", r"\end{tabular}"]
TEX["sC_details:tab:fdconv"] = "\n".join(L)
orders = {p: [float(r["observed_order_rel_l2"]) for r in cl if r["problem"] == p and r["observed_order_rel_l2"] != ""] for p in PROBLEMS}
smooth = [o for p in PROBLEMS if p != "laplace3d" for o in orders[p]]
NUM["fd"] = {"orders": orders, "order_min_without_L3": min(smooth), "order_max_without_L3": max(smooth),
             "n_rows": len(cl), "time_min": min(float(r["time_s_best_of_3"]) for r in cl),
             "time_max": max(float(r["time_s_best_of_3"]) for r in cl)}

# ---- cost (T5 plus the lighter-load measurement) -----------------------------------------
# the PINN error, the strategy and the grid of each row are those of Table s5_hard:tab:fd; they are kept in
# data/sC_details_numbers.json (cost) and left out of this table so that it is printed at a readable size
L = [r"\begin{tabular}{@{}l rrrr r r r rrr@{}}", r"\toprule",
     r" & \multicolumn{4}{c}{measurement A} & \multicolumn{1}{c}{B} & \multicolumn{1}{c}{C} & FD & \multicolumn{3}{c}{time ratio} \\",
     r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}\cmidrule(lr){8-8}\cmidrule(l){9-11}",
     r"Problem & \multicolumn{1}{c}{ms/it} & \multicolumn{1}{c}{ms/it} & ev./it & run (s) & run (s) & run (s) & time (s) & \multicolumn{1}{c}{A} & \multicolumn{1}{c}{B} & \multicolumn{1}{c}{C} \\",
     r" & \multicolumn{1}{c}{\Adam} & \multicolumn{1}{c}{L-BFGS} & & & & & & & & \\", r"\midrule"]
COST = {}
for p in PROBLEMS:
    t = tm[p]
    per = {s: geo(series(p, s, "adam_lbfgs")) for s in STRATS}
    best_s = min(per, key=per.get)
    best = per[best_s]
    cand = [r for r in cl if r["problem"] == p and float(r["rel_l2"]) <= best]
    c = min(cand, key=lambda r: float(r["time_s_best_of_3"]))
    tfd = float(c["time_s_best_of_3"])
    keys = [k for k in V if k.split("|")[0] == p and int(k.split("|")[3]) == CONFIG[p]["n_r"]]
    tb = sorted(V[k]["adam_lbfgs"]["time"] for k in keys)
    ra = t["est_run_s_adam_lbfgs"] / tfd
    rb = [tb[0] / tfd, tb[-1] / tfd]
    tc = sorted(s_["C_adam_lbfgs_per_network_s"] for s_ in STACK[p])
    rc = [tc[0] / tfd, tc[-1] / tfd]
    COST[p] = {"n_params": t["n_params"], "ms_adam": t["ms_per_adam_step"], "ms_lbfgs": t["ms_per_lbfgs_iter"],
               "fevals_per_iter": t["lbfgs_fevals_per_iter"], "run_A_s": t["est_run_s_adam_lbfgs"],
               "run_B_s": {"n": len(tb), "min": tb[0], "median": float(np.median(tb)), "max": tb[-1]},
               "best_strategy": best_s, "best_geo": best, "fd_n": int(c["n"]), "fd_rel_l2": float(c["rel_l2"]), "fd_time_s": tfd,
               "ratio_A": ra, "ratio_B": rb, "A_over_B_median": t["est_run_s_adam_lbfgs"] / float(np.median(tb)),
               "run_C_s": {"n_stacks": len(tc), "min": tc[0], "max": tc[-1]}, "ratio_C": rc}
    runb = f"{tb[0]:.0f}--{tb[-1]:.0f} [{len(tb)}]" if len(tb) > 1 else f"{tb[0]:.0f} [1]"
    ratb = (f"{e3(rb[0], 1)}--{e3(rb[1], 1)}" if len(tb) > 1 else e3(rb[0], 1))
    runc = f"{tc[0]:.1f}--{tc[-1]:.1f}" if len(tc) > 1 and f"{tc[0]:.1f}" != f"{tc[-1]:.1f}" else f"{tc[0]:.1f}"
    ratc = f"{e3(rc[0], 1)}--{e3(rc[1], 1)}" if e3(rc[0], 1) != e3(rc[1], 1) else e3(rc[0], 1)
    ev = f"{t['lbfgs_fevals_per_iter']:.3f}".rstrip("0")
    L.append(f"{PMAC[p]} & {t['ms_per_adam_step']:.0f} & {t['ms_per_lbfgs_iter']:.0f} & {ev} & "
             f"{t['est_run_s_adam_lbfgs']:.0f} & {runb} & {runc} & "
             f"{e3(tfd, 1)} & {e3(ra, 1)} & {ratb} & {ratc} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
TEX["sC_details:tab:cost"] = "\n".join(L)
NUM["cost"] = COST
NUM["cost_summary"] = {"loadavg_A_start": tm["loadavg_start"], "loadavg_A_end": tm["loadavg_end"],
                       "ratio_A_range": [min(c["ratio_A"] for c in COST.values()), max(c["ratio_A"] for c in COST.values())],
                       "ratio_B_range": [min(c["ratio_B"][0] for c in COST.values()), max(c["ratio_B"][1] for c in COST.values())],
                       "ratio_C_range": [min(c["ratio_C"][0] for c in COST.values()), max(c["ratio_C"][1] for c in COST.values())],
                       "run_C_range_s": [min(c["run_C_s"]["min"] for c in COST.values()), max(c["run_C_s"]["max"] for c in COST.values())],
                       "run_A_range_s": [min(c["run_A_s"] for c in COST.values()), max(c["run_A_s"] for c in COST.values())],
                       "run_B_range_s": [min(c["run_B_s"]["min"] for c in COST.values()), max(c["run_B_s"]["max"] for c in COST.values())],
                       "A_over_B_median_range": [min(c["A_over_B_median"] for c in COST.values()), max(c["A_over_B_median"] for c in COST.values())],
                       "fevals_per_iter_range": [min(c["fevals_per_iter"] for c in COST.values()), max(c["fevals_per_iter"] for c in COST.values())],
                       "stack_fevals_per_iter_range": [min(s["lbfgs_fevals_per_iter"] for p in PROBLEMS for s in STACK[p]),
                                                       max(s["lbfgs_fevals_per_iter"] for p in PROBLEMS for s in STACK[p])]}


# ---- hyper-parameter table -----------------------------------------------------------------
def net(p):
    c = LOW[p]
    return f"${len(c['hidden'])}\\times{c['hidden'][0]}$ ({thousands(c['n_params'])})"


def evalset(p):
    c = LOW[p]
    sh = c["eval_shape"]
    if len(set(sh)) == 1 and len(sh) == 3:
        return f"${sh[0]}^3$"
    return "$" + "\\times".join(str(x) for x in sh) + "$"


def stack(p):
    ss = LOW[p]["stacks"]
    b = sorted({s["batch_size"] for s in ss})
    assert len(b) == 1
    return (f"stack of {b[0]}" if len(ss) == 1 else f"{len(ss)} stacks of {b[0]}") + ", GPU"


c0 = LOW["heat1d"]
assert all(LOW[p]["n_adam"] == c0["n_adam"] and LOW[p]["branch_at"] == c0["branch_at"] for p in PROBLEMS)
n_ad, n_br = c0["n_adam"], c0["branch_at"]
arms_txt = f"\\Adam: {n_ad}; \\AdamLBFGS: {n_br}\\,+\\,{n_ad - n_br}"
n1, n2, n5, n6 = (NUM["further_experiments"][k] for k in ("wave2d_long", "laplace3d_smooth", "highdim_long_d20", "timeslices"))
n3, n7 = NUM["further_experiments"]["pitfalls_autograd"], NUM["further_experiments"]["gridcontrol"]
assert (n7["PROBLEM"], n7["N_ADAM"], n7["BRANCH_AT"], n7["M"] ** 2) == ("heat1d", n_ad, n_br, c0["n_r"])
p_n7 = os.path.join(ART, "data", "s4_lowdim_gridcontrol.json")
n7_stacks = json.load(open(p_n7))["stacks"] if os.path.exists(p_n7) else None
if n7_stacks:
    NUM["further_experiments"]["gridcontrol"]["stacks"] = n7_stacks
    assert {v["device"] for v in n7_stacks.values()} == {"mps"}
n3_steps = n3["PASSES"] * -(-n3["N_TRAIN"] // n3["BATCH"])
NUM["further_experiments"]["pitfalls_autograd"]["iterations"] = n3_steps
assert n2["N_R"] == LOW["laplace3d"]["n_r"] and (n2["N_ADAM"], n2["BRANCH_AT"]) == (n_ad, n_br) and n2["SAMPLER"] == "sobol"
assert n1["N_R"] == LOW["wave2d"]["n_r"] and n1["BRANCH_AT"] == n_br and n1["SAMPLER"] == "sobol"
assert (n6["PROBLEM"], n6["N_R"], n6["N_ADAM"], n6["BRANCH_AT"]) == ("heat1d", c0["n_r"], n_ad, n_br)
assert n5["D"] == 20 and sorted(n5["PROBLEMS"]) == ["laplace", "poisson"] and sorted(n5["METHODS"]) == ["pinn", "ritz"]
ph = HD["phases"]
val = VAL["heat1d"]
assert VAL["laplace2d"]["seeds"] == val["seeds"] and val["single_lbfgs_iters"] == [300]
ver_seeds = "; ".join(f"{PMAC[p]}: {seed_range(VER['seeds'][p])}" for p in ("heat1d", "laplace2d")) + "; else " + \
    seed_range(VER["seeds"]["wave1d"])
assert VER["seeds"]["wave1d"] == VER["seeds"]["wave2d"] == VER["seeds"]["laplace3d"]
assert VER["kinds_heat"] == ["grid", "nodegrid", "random", "sobol"] and VER["n_r_heat"] == [256, 1024]
COLS = (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.9cm}>{\raggedright\arraybackslash}p{3.3cm}"
        r">{\raggedright\arraybackslash}p{3.5cm}>{\raggedright\arraybackslash}p{1.25cm}"
        r">{\raggedright\arraybackslash}p{2.05cm}>{\raggedright\arraybackslash}p{1.9cm}@{}}")
HEAD = (r"Experiment & Network (parameters); $\Nr$ and strategy; constraint points & Optimiser; iterations & Seeds & "
        r"Driver, device & Evaluation points \\")
WIDE = r"\multicolumn{6}{@{}p{15.8cm}@{}}"
H = [COLS, r"\toprule", HEAD, r"\midrule"]
H.append(WIDE + r"{\emph{Low-dimensional problems} (package \code{pinnbench}; Sections~\ref{s4_lowdim:sec} and~\ref{s5_hard:sec}). "
         rf"Loss~\eqref{{s3_methods:eq:pinn}} with $\lam=1$; \Adam\ with constant learning rate $10^{{{int(round(np.log10(core.ADAM_LR)))}}}$; "
         rf"\LBFGS\ with memory {core.LBFGS_MEMORY}; single precision. ``As benchmark'': as in the benchmark row of the same problem.}} \\[3pt]")
for i, p in enumerate(PROBLEMS):
    c = LOW[p]
    H.append(f"Benchmark, {PMAC[p]} & {net(p)}; {c['n_r']}{', five strategies' if i == 0 else ''}; {thousands(c['n_constraint_points'])} & "
             f"{arms_txt if i == 0 else 'as for ' + PMAC['heat1d']} & {seed_range(c['seeds'])} & {stack(p)} & {evalset(p)} \\\\")
H.append(f"Budget sweep, \\PH & as benchmark; {SWEEP['new_n_r'][0]} and {thousands(SWEEP['new_n_r'][1])}; \\strat{{grid-c}}, \\strat{{random}}, \\strat{{Sobol}} & "
         f"as benchmark & {seed_range(SWEEP['seeds'])} & stack of {SWEEP['stack']}, GPU & as benchmark \\\\")
H.append(f"Verification code \\textsf{{V-bench}} (Section~\\ref{{sC_details:sec:verification}}) & as benchmark; for \\PH\\ also \\strat{{grid-n}} and $\\Nr={thousands(VER['n_r_heat'][1])}$ & "
         f"as benchmark, PyTorch \\LBFGS & {ver_seeds} & single network, CPU, 4 threads & as benchmark \\\\")
if n7_stacks:
    sizes = sorted(v["batch_size"] for v in n7_stacks.values())
    H.append(f"Grid control (Section~\\ref{{s4_lowdim:sec:strategy}}), \\PH & as benchmark; {n7['M'] ** 2}: \\strat{{random}}, \\strat{{grid-c}}, \\strat{{grid-n}}, "
             f"\\strat{{grid-c}} shifted in $t$; {n7['M'] ** 2 + n7['M']}: \\strat{{grid-c}} with one further column & "
             f"as benchmark & {seed_range(n7['SEEDS'])} & stacks of {sizes[1]} and {sizes[0]}, GPU & as benchmark \\\\")
H.append(f"Time slices, \\PH & as benchmark; {n6['N_R']}; " + ", ".join(SMAC[s] for s in n6["STRATEGIES"]) + " & "
         f"as benchmark, PyTorch \\LBFGS & {seed_range(n6['SEEDS'])} & single network, CPU & as benchmark$^{{a}}$ \\\\")
H.append(f"Longer runs, \\PWtwo & as benchmark; {n1['N_R']}; {SMAC[n1['SAMPLER']]} & "
         f"\\Adam: {thousands(n1['TOTAL_DEFAULT'])}; \\Adam\\ {n1['BRANCH_AT']}, then PyTorch \\LBFGS\\ up to {thousands(n1['TOTAL_DEFAULT'])} in total & "
         f"{seed_range(n1['SEEDS'])} & single network, CPU & as benchmark \\\\")
n8 = NUM["further_experiments"]["wave1d_long"]
assert n8["N_R"] == LOW["wave1d"]["n_r"] and n8["BRANCH_AT"] == n_br and n8["SAMPLER"] == "sobol"
H.append(f"Longer runs, \\PWone & as benchmark; {n8['N_R']}; {SMAC[n8['SAMPLER']]} & "
         f"\\Adam: {thousands(n8['TOTAL_DEFAULT'])}; \\Adam\\ {n8['BRANCH_AT']}, then PyTorch \\LBFGS\\ up to {thousands(n8['TOTAL_DEFAULT'])} in total & "
         f"{seed_range(n8['SEEDS'])} & single network, CPU & as benchmark \\\\")
H.append(f"Compatible data, \\PLthreeS\\ and \\PLthree & as \\PLthree; {n2['N_R']}; {SMAC[n2['SAMPLER']]} & as benchmark & {seed_range(n2['SEEDS'])} & "
         f"stack of {len(n2['SEEDS'])}, CPU & as \\PLthree \\\\")
H.append(f"Driver validation (Section~\\ref{{sC_details:sec:validation}}), \\PH, \\PLtwo & as benchmark; 1024; five strategies & "
         f"\\Adam: 600; \\AdamLBFGS: 300\\,+\\,300 & {seed_range(val['seeds'])} & stack of {val['n']} and single network, CPU & as benchmark \\\\")
H.append(r"Timing A (Table~\ref{sC_details:tab:cost}), five problems & as benchmark; \strat{Sobol} & 400 \Adam\ and 200 PyTorch \LBFGS\ iterations timed, "
         rf"scaled to {n_br}\,+\,{n_ad - n_br} & 0 & single network, CPU, 4 threads & --- \\")
H.append(r"Finite differences, five problems & $n$ cells per direction (Table~\ref{sC_details:tab:fdconv}) & direct solve or time stepping, double precision; "
         r"time: best of 3 & --- & NumPy, SciPy, CPU & as benchmark \\")
H += [r"\bottomrule", r"\end{tabular}"]
TEX["sC_details:tab:hyper"] = "\n".join(H)

H = [COLS.replace(r"p{3.5cm}", r"p{3.6cm}").replace(r"p{2.05cm}", r"p{1.6cm}").replace(r"p{1.9cm}", r"p{2.25cm}"), r"\toprule",
     r"Experiment & Problems, dimensions, weights; or network, points & Optimiser; iterations & Seeds & Runs; driver, device & Evaluation points \\", r"\midrule"]
H.append(WIDE + r"{\emph{Problems \Pone\ and \Ptwo\ in $d$ dimensions} (module \code{hd_core}; Section~\ref{s6_highdim:sec}). "
         rf"Network $3\times{HD['width']}$ with $64d+8449$ parameters; $\Nr=\Nb={HD['n_int']}$, redrawn at every iteration; Adam, learning rate "
         rf"$10^{{{int(round(np.log10(HD['lr'])))}}}$, multiplied by 0.1 after 50 and after 75 per cent of the run; one network per run, single precision, "
         r"CPU, one thread.} \\[3pt]")
H.append(f"Choice of weights & $d={ph['sweep']['d'][0]}$; $\\lam,\\bet\\in\\{{1,10,100,1000\\}}$ & {thousands(ph['sweep']['iters'][0])} & {ph['sweep']['seeds'][0]} & "
         f"{ph['sweep']['n']} runs$^{{b}}$ & {thousands(HD['n_val'])} validation points \\\\")
H.append("Main study & $d=" + ",".join(str(x) for x in ph["main"]["d"]) + "$; $\\lam=1000$; $\\bet=100$ (\\Pone), $1$ (\\Ptwo) & "
         f"{thousands(ph['main']['iters'][0])} & {seed_range(ph['main']['seeds'])} & {ph['main']['n']} runs & {thousands(HD['n_test'])} test points, seed $10\\,000+d$ \\\\")
H.append(f"Longer schedule & \\Pone, $d={ph['long']['d'][0]}$, both methods & {thousands(ph['long']['iters'][0])} & {seed_range(ph['long']['seeds'])} & {ph['long']['n']} runs & as main study \\\\")
H.append(f"Equal CPU time & Deep Ritz, $d={ph['equaltime']['d'][0]}$ & {thousands(ph['equaltime']['iters_by_problem']['laplace'][0])} (\\Pone), "
         f"{thousands(ph['equaltime']['iters_by_problem']['poisson'][0])} (\\Ptwo) & {seed_range(ph['equaltime']['seeds'])} & {ph['equaltime']['n']} runs & as main study \\\\")
H.append(f"Longer schedule, $d=20$ & \\Pone\\ and \\Ptwo, $d={n5['D']}$, both methods & {thousands(n5['ITERS'])} & {seed_range(n5['SEEDS'])} & "
         f"{len(n5['SEEDS']) * len(n5['PROBLEMS']) * len(n5['METHODS'])} runs & as main study \\\\")
H.append(r"Cost per iteration & $d=2,3,5,10,20,50,100$; Deep Ritz, PINN with forward and with nested Laplacian & 5 warm-up iterations, then 7 rounds of "
         r"10 timed iterations for each variant & 0 & CPU time of the process & --- \\")
H.append(r"\midrule")
H.append(WIDE + r"{\emph{Demonstrations of Section~\ref{s7_pitfalls:sec}}; single precision.} \\[3pt]")
H.append(f"Zero residual (Section~\\ref{{s7_pitfalls:sec:autograd}}) & 3--50--50--1; {thousands(n3['N_TRAIN'])} uniform points, batches of "
         f"{n3['BATCH']}; faulty residual, data term $\\ee^{{{{-t}}}}S$, 400 boundary points at $t=0$ & Adam, learning rate $10^{{{int(round(np.log10(n3['LR'])))}}}$; {n3['PASSES']} passes, {thousands(n3_steps)} "
         f"iterations (also 1500 passes) & {n3['SEED']} & single network, CPU, 1 thread & $21\\times81\\times81$ of \\PWtwo \\\\")
H.append(f"Data on one face (Section~\\ref{{s7_pitfalls:sec:oneface}}) & {net('laplace3d')}; closed $21^3$ grid ({thousands(21 ** 3)} points); datum on the {21 ** 2} points with $x=\\pi$ only & "
         f"AdamW, learning rate $10^{{-3}}$, weight decay {NUM['oneface']['adamw_weight_decay_default']:g} (the default); {thousands(2000)} & "
         f"{seed_range(NUM['oneface']['seeds'])} & stack of {len(NUM['oneface']['seeds'])}, GPU & "
         f"{evalset('laplace3d')}, against $\\uex$ of \\PLthree \\\\")
H += [r"\bottomrule", r"\end{tabular}"]
TEX["sC_details:tab:hyper2"] = "\n".join(H)

# =========================================================================================
# 4. write
# =========================================================================================
with open(os.path.join(ART, "data", "sC_details_numbers.json"), "w") as f:
    json.dump(NUM, f, indent=1, default=float)
with open(os.path.join(ART, "data", "sC_details_tables.tex"), "w") as f:
    for k, v in TEX.items():
        f.write(f"%% ---- {k} (generated by code/sC_details_tables.py) ----\n{v}\n\n")
sec = os.path.join(ART, "sections", "S8_details.tex")
if os.path.exists(sec):
    s = open(sec).read()
    for k, v in TEX.items():
        a, b = f"%% BEGIN GENERATED {k}", f"%% END GENERATED {k}"
        assert a in s and b in s, f"marker for {k} missing in sections/S8_details.tex"
        i = s.index(a) + len(a)
        i = s.index("\n", i) + 1
        j = s.index(b)
        s = s[:i] + v + "\n" + s[j:]
    open(sec, "w").write(s)
    print("tables injected into", sec)
print(json.dumps({k: NUM[k] for k in ("validation", "independent_vs_stacked", "superseded_heat_block", "cost_summary",
                                      "bit_for_bit", "software", "benchmark_stack_wall_s", "loss_assembly", "optimiser",
                                      "verifier_lowdim", "further_experiments", "fd")}, indent=1, default=float))
print({k: v for k, v in HD.items() if k != "phases"})
print(json.dumps(HD["phases"], indent=1))
