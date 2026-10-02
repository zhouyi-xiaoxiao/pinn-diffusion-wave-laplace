#!/usr/bin/env python3
"""Check the headline numbers of the article (abstract, Section 8.1, README) against the stored results.
(The checks of Sections 6.7 and 6.8 read the outputs of code/s6_highdim_bound.py and the decision files of
research_highdim_remedies/.)

    python code/check_headline_numbers.py        (from the repository root; no training, a few seconds)

Each check recomputes one headline statement from the raw per-run files (or, where a number is itself
the output of an analysis script, from that script's output) and compares it with the value printed in
the article.  Output: standard output, last line "ALL HEADLINE CHECKS PASSED"; optionally --output PATH.
The default invocation is read-only. Use --output PATH to save the report.
Reference values represent the manuscript at release; this is a data-consistency check, not a parser of LaTeX prose.
"""
import argparse
import csv
import json
import math
import os
import sys

LABEL = {"primary": "primary", "verifier": "verification"}  # names of the two cost measurements

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)  # repository root
RB = os.path.join(ROOT, "research_benchmark", "results")
RH = os.path.join(ROOT, "research_highdim", "results")
DATA = os.path.join(ART, "data")

lines = []
ok_all = True


def check(name, cond, detail):
    global ok_all
    ok_all &= bool(cond)
    lines.append(f"[{'ok' if cond else 'FAIL'}] {name}: {detail}")


def geo(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def rd(x, n):
    """Round to n significant digits."""
    return float(f"{x:.{n - 1}e}")


# ---------------------------------------------------------------- low-dimensional benchmark
rows = list(csv.DictReader(open(os.path.join(RB, "benchmark_runs.csv"))))
err = {(r["problem"], r["sampler"], r["arm"], int(r["seed"])): float(r["rel_l2"]) for r in rows}
pairs = sorted({k[:2] + (k[3],) for k in err})
check("350 final errors, 175 networks", len(rows) == 350 and len(pairs) == 175, f"{len(rows)} rows, {len(pairs)} networks")
better = sum(err[p, s, "adam_lbfgs", k] < err[p, s, "adam", k] for p, s, k in pairs)
check("L-BFGS arm better in all 175 pairs", better == 175, f"{better} of {len(pairs)}")
for prob, a_txt, b_txt in (("heat1d", 2.1e-2, 1.6e-3), ("laplace2d", 7.8e-3, 3.4e-4)):
    ga = geo([v for k, v in err.items() if k[0] == prob and k[2] == "adam"])
    gb = geo([v for k, v in err.items() if k[0] == prob and k[2] == "adam_lbfgs"])
    check(f"geometric means, {prob}", rd(ga, 2) == a_txt and rd(gb, 2) == b_txt, f"{ga:.3e} -> {gb:.3e} (text {a_txt:g} -> {b_txt:g})")
w1 = [v for k, v in err.items() if k[0] == "wave1d"]
check("Wave1: all 50 runs between 41 and 51 per cent", len(w1) == 50 and 0.41 <= min(w1) and max(w1) < 0.515,
      f"{len(w1)} runs, {min(w1):.3f} to {max(w1):.3f}")
means = []
for prob in ("wave2d", "laplace3d"):
    for s in sorted({k[1] for k in err if k[0] == prob}):
        v = [x for k, x in err.items() if k[0] == prob and k[1] == s and k[2] == "adam_lbfgs"]
        means.append(sum(v) / len(v))
check("Wave2, Lap3: strategy means between 3.6 and 5.0 per cent", rd(min(means), 2) == 0.036 and rd(max(means), 2) == 0.050,
      f"{min(means):.4f} to {max(means):.4f}")
# cell-centred grid against random on the heat problem (paired by seed)
ratios = [err["heat1d", "grid", "adam_lbfgs", k] / err["heat1d", "random", "adam_lbfgs", k] for k in range(10)]
check("Heat: cell-centred grid about four times worse than random, 10 of 10 seeds",
      abs(geo(ratios) - 4.21) < 0.005 and all(r > 1 for r in ratios), f"ratio {geo(ratios):.2f}, worse in {sum(r > 1 for r in ratios)}/10")
# Recompute both verification headlines from per-run records, not a cached prose summary.
v_runs = {}
for filename in ("v_runs.json", "v_runs_extra.json", "v_runs_budget.json"):
    with open(os.path.join(ROOT, "verify_research_benchmark", filename), encoding="utf8") as handle:
        for key, value in json.load(handle).items():
            if key in v_runs and v_runs[key] != value:
                raise ValueError(f"conflicting verification record: {key}")
            v_runs[key] = value
v_better = sum(r["adam_lbfgs"]["rel_l2"] < r["adam"]["rel_l2"] for r in v_runs.values())
check("verification code V-bench: 39 of 40", len(v_runs) == 40 and v_better == 39,
      f"{v_better} of {len(v_runs)} recomputed from raw verification records")
node_ratios = [v_runs[f"heat1d|nodegrid|{seed}|256|fixed"]["adam_lbfgs"]["rel_l2"] /
               v_runs[f"heat1d|random|{seed}|256|fixed"]["adam_lbfgs"]["rel_l2"] for seed in range(6)]
node_ratio = geo(node_ratios)
check("verification code V-bench: node-centred grid 1.26 times random (6 seeds)",
      round(node_ratio, 2) == 1.26, f"raw six-seed geometric ratio {node_ratio:.8f}")
# L-BFGS arm against the best logged Adam iterate ("about one order of magnitude")
bi = json.load(open(os.path.join(DATA, "s4_lowdim_numbers.json")))["lbfgs_vs_logged_adam_iterates_it1500_3000"]
check("Heat, Lap2: L-BFGS arm 8 to 13 times below the best logged Adam iterate, in 49/50 and 50/50 runs",
      rd(1 / bi["heat1d"]["geo_ratio_to_best"], 1) == 8 and rd(1 / bi["laplace2d"]["geo_ratio_to_best"], 2) == 13
      and bi["heat1d"]["n_lbfgs_below_best_adam"] == 49 and bi["laplace2d"]["n_lbfgs_below_best_adam"] == 50,
      f"ratios {bi['heat1d']['geo_ratio_to_best']:.3f} and {bi['laplace2d']['geo_ratio_to_best']:.3f}; "
      f"{bi['heat1d']['n_lbfgs_below_best_adam']}/50 and {bi['laplace2d']['n_lbfgs_below_best_adam']}/50")
check("Heat, Lap2: L-BFGS arm 14 to 23 times below the last Adam iterate",
      rd(1 / bi["heat1d"]["geo_ratio_to_last"], 2) == 14 and rd(1 / bi["laplace2d"]["geo_ratio_to_last"], 2) == 23,
      f"ratios {bi['heat1d']['geo_ratio_to_last']:.4f} and {bi['laplace2d']['geo_ratio_to_last']:.4f} "
      f"(factors {1 / bi['heat1d']['geo_ratio_to_last']:.1f} and {1 / bi['laplace2d']['geo_ratio_to_last']:.1f}; the text quotes 0.074 and 0.044)")
# control experiment: the penalty of the cell-centred grid is caused by the empty strip next to the initial line
gc = json.load(open(os.path.join(DATA, "s4_lowdim_gridcontrol.json")))
vr, pl = gc["vs_random"]["adam_lbfgs"], gc["column_in_strip_vs_placebo"]["adam_lbfgs"]
check("Heat, control experiment: 16 points in the strip remove the penalty, 16 points at t = 1/2 do not",
      0.8 < vr["gridplus0"]["ratio"] < 1.2 and vr["gridplusm"]["ratio"] > 4 and vr["gridplusm"]["n_better"] == 0
      and pl["n_better"] == 10 and abs(pl["p"] - 2 / 1024) < 1e-9,
      f"column at 1/64: {vr['gridplus0']['ratio']:.2f} x random; column at 1/2: {vr['gridplusm']['ratio']:.2f} x random "
      f"(worse in {10 - vr['gridplusm']['n_better']}/10); column in strip against placebo {pl['ratio']:.2f}, {pl['n_better']}/10, p = {pl['p']:.4f}")
check("Heat, control experiment: cell-centred grid 4.31, node-centred grid 1.98 times random",
      abs(vr["grid"]["ratio"] - 4.31) < 0.005 and abs(vr["gridn"]["ratio"] - 1.98) < 0.005 and abs(vr["gridn"]["p"] - 0.049) < 0.0005,
      f"grid-c {vr['grid']['ratio']:.2f} ({vr['grid']['n_better']}/10 better), grid-n {vr['gridn']['ratio']:.2f} (p = {vr['gridn']['p']:.3f})")
bs = json.load(open(os.path.join(DATA, "s4_lowdim_numbers.json")))["budget"]["1024"]
check("Heat, 1024 points: strategies agree", abs(bs["grid_over_random"]["ratio"] - 1.11) < 0.005 and bs["grid_over_random"]["p"] > 0.4
      and abs(bs["sobol_over_random"]["ratio"] - 1.02) < 0.005,
      f"grid/random {bs['grid_over_random']['ratio']:.2f} (p={bs['grid_over_random']['p']:.2f}), Sobol/random {bs['sobol_over_random']['ratio']:.2f}")

# ---------------------------------------------------------------- time slices of the heat problem
ts = json.load(open(os.path.join(DATA, "s4_lowdim_timeslices.json")))["summary"]
n_above = sum(c["n_seeds_slice_t1_above_1"] for c in ts.values())
g = [ts[k]["rel_l2_spacetime"]["geo_mean"] for k in ("random:adam_lbfgs", "sobol:adam_lbfgs")]
check("Heat: slice t = 1 above 100 per cent in all 40 runs (20 networks, both arms)", n_above == 40, f"{n_above} of 40")
check("Heat: space-time error after L-BFGS about 0.1 per cent", all(0.9e-3 < x < 1.25e-3 for x in g), f"{g[0]:.2e}, {g[1]:.2e}")

# ---------------------------------------------------------------- harder problems
w2 = [s["adam_lbfgs_rel_l2"] for s in json.load(open(os.path.join(DATA, "s5_hard_wave2d_long.json")))["summary"]["per_seed"]]
check("Wave2, longer L-BFGS phase: 0.6 to 0.9 per cent, three seeds", len(w2) == 3 and rd(min(w2), 1) == 0.006 and rd(max(w2), 1) == 0.009,
      ", ".join(f"{x:.2e}" for x in w2))
l3 = json.load(open(os.path.join(DATA, "s5_hard_laplace3d_smooth.json")))
l3s = [r["adam_lbfgs"]["rel_l2"] for r in l3["L3s"]["rows"]]
check("Lap3s: 0.3 per cent", rd(sum(l3s) / len(l3s), 1) == 0.003, f"mean {sum(l3s) / len(l3s):.2e} over {len(l3s)} seeds")
cost = json.load(open(os.path.join(DATA, "s5_hard_cost.json")))["summary"]
lo, hi = (math.log10(x) for x in cost["ratio_B_range"])
check("finite differences three to five orders of magnitude faster than a single network on the CPU (measurement B)",
      3 <= lo < 4 and 5 <= hi < 5.5, f"time ratio {cost['ratio_B_range'][0]:.2e} to {cost['ratio_B_range'][1]:.2e}")
check("finite differences more than 400 times faster in every timing (measurements A, B, C); smallest ratio 4.8e2",
      cost["ratio_smallest_of_all_measurements"] > 400 and rd(cost["ratio_smallest_of_all_measurements"], 2) == 480
      and min(cost["ratio_C_range"]) == cost["ratio_smallest_of_all_measurements"],
      f"smallest ratio {cost['ratio_smallest_of_all_measurements']:.1f} (C: {cost['ratio_C_range'][0]:.3g} to {cost['ratio_C_range'][1]:.3g})")
fe = cost["fd_error_over_pinn_error_range"]
check("finite differences at equal or better accuracy", fe[1] <= 1, f"FD error / PINN error {fe[0]:.2f} to {fe[1]:.2f}")
w1l = json.load(open(os.path.join(DATA, "s5_hard_wave1d_long.json")))["summary"]["per_seed"]
v = [r["adam_rel_l2"] for r in w1l] + [r["adam_lbfgs_rel_l2"] for r in w1l]
check("Wave1, longer runs: 0.34 to 0.38 in both arms, three seeds", len(w1l) == 3 and rd(min(v), 2) == 0.34 and rd(max(v), 2) == 0.38,
      ", ".join(f"{x:.3f}" for x in v))

# ---------------------------------------------------------------- dimension study
summ = {(r["problem"], r["method"], int(r["d"])): float(r["rel_l2_mean"]) for r in csv.DictReader(open(os.path.join(RH, "summary.csv")))}
rat = [summ[p, "ritz", d] / summ[p, "pinn", d] for p in ("laplace", "poisson") for d in (2, 3, 5, 10)]
check("d <= 10: PINN 1.5 to 4.3 times more accurate", rd(min(rat), 2) == 1.5 and rd(max(rat), 2) == 4.3, f"{min(rat):.2f} to {max(rat):.2f}")
floor = math.sqrt(1 - 96 / math.pi ** 4)
check("d = 20, 4000 iterations: both above the affine level on PoiD",
      summ["poisson", "pinn", 20] > floor and summ["poisson", "ritz", 20] > floor,
      f"PINN {summ['poisson', 'pinn', 20]:.4f}, Deep Ritz {summ['poisson', 'ritz', 20]:.4f}, floor {floor:.4f}")
long20 = {}
for line in open(os.path.join(DATA, "s6_highdim_long_d20.jsonl")):
    r = json.loads(line)
    long20.setdefault((r["problem"], r["method"]), []).append(r["rel_l2"])
check("d = 20, 16 000 iterations: 12 runs", sum(len(v) for v in long20.values()) == 12, str({k: len(v) for k, v in long20.items()}))
check("d = 20, 16 000 iterations: all PoiD runs below the affine level", max(long20["poisson", "pinn"] + long20["poisson", "ritz"]) < floor,
      f"largest {max(long20['poisson', 'pinn'] + long20['poisson', 'ritz']):.4f}")
check("d = 20, 16 000 iterations: PINN more accurate in every seed",
      all(max(long20[p, "pinn"]) < min(long20[p, "ritz"]) for p in ("laplace", "poisson")),
      "; ".join(f"{p}: PINN max {max(long20[p, 'pinn']):.2e}, Deep Ritz min {min(long20[p, 'ritz']):.2e}" for p in ("laplace", "poisson")))
ext = json.load(open(os.path.join(RH, "extensions.json")))
eq = {e["problem"]: e for e in ext if e["phase"] == "equaltime"}
check("equal compute at d = 10: Deep Ritz more accurate on both problems",
      all(max(eq[p]["rel_l2"]) < summ[p, "pinn", 10] for p in ("laplace", "poisson")),
      "; ".join(f"{p}: Deep Ritz {eq[p]['rel_l2_mean']:.2e}, PINN {summ[p, 'pinn', 10]:.2e}" for p in ("laplace", "poisson")))
num = json.load(open(os.path.join(DATA, "s6_highdim_numbers.json")))["cost_growth_d2_to_d100"]
check("cost from d = 2 to 100: Deep Ritz below 2x, PINN above 12x (both measurements)",
      all(num[m]["ritz"] < 2 and num[m]["fwd"] > 12 for m in ("primary", "verifier")),
      "; ".join(f"{LABEL[m]}: Deep Ritz x{num[m]['ritz']:.2f}, PINN (forward) x{num[m]['fwd']:.1f}" for m in ("primary", "verifier")))

# ---------------------------------------------------------------- stability constant, error bound, remedies (Sections 6.7, 6.8)
bd = json.load(open(os.path.join(DATA, "s6_highdim_bound_numbers.json")))
nb = bd["n_networks_bound_valid"]
check("bound of Section 6.7, Corollary (c), valid on all 133 networks evaluated (88 saved, 24 + 21 retrained)",
      nb["total"] == 133 and nb["all_valid"], f"{nb['saved_networks']} + {nb['recheck_round1']} + {nb['recheck_round2']}, all valid: {nb['all_valid']}")
e = bd["eta_d10_d20_all_networks"]
check("efficiency 1.9 to 3.4 at d = 10 and 20, up to 25 at d = 2", rd(e[0], 2) == 1.9 and rd(e[1], 2) == 3.4
      and rd(bd["eta_by_d_all_networks"]["2"]["max"], 2) == 25, f"{e[0]:.3f} to {e[1]:.3f}; d = 2 max {bd['eta_by_d_all_networks']['2']['max']:.2f}")
check("constants of Table s6_highdim:tab:kappa agree with their formulas; sqrt(d) U_d -> 1/pi",
      all(k["formulas_match_table"] for k in bd["kappa_constants"]) and abs(bd["sqrt_d_U_d_at_1e6"] - 1 / math.pi) < 1e-5,
      f"sqrt(d) U_d at d = 1e6: {bd['sqrt_d_U_d_at_1e6']:.6f}")
r2c = bd["recheck_r2"]["cells"]
check("equal compute at d = 20 (re-check, LapD, three seeds): Deep Ritz more accurate than the PINN in every pair",
      r2c["P1_ritz_eqcpu_d20"]["n"] == 3 and r2c["P1_pinn_d20"]["n"] == 3
      and r2c["P1_ritz_eqcpu_d20"]["rel_err"][1] < r2c["P1_pinn_d20"]["rel_err"][0],
      f"Deep Ritz at the PINN's CPU time {r2c['P1_ritz_eqcpu_d20']['rel_err'][0]:.4f} to {r2c['P1_ritz_eqcpu_d20']['rel_err'][1]:.4f}, "
      f"PINN {r2c['P1_pinn_d20']['rel_err'][0]:.4f} to {r2c['P1_pinn_d20']['rel_err'][1]:.4f}")
kap = {k["d"]: k for k in bd["kappa_constants"]}
check("sqrt(2 d U_d) grows (factor of the root-mean-square boundary misfit): 1.08 at d = 2 to 2.53 at d = 100",
      all(math.sqrt(2 * a * kap[a]["U_d"]) < math.sqrt(2 * b * kap[b]["U_d"]) for a, b in zip(sorted(kap)[1:], sorted(kap)[2:])),
      ", ".join(f"d={d}: {math.sqrt(2 * d * kap[d]['U_d']):.3f}" for d in sorted(kap)))
RR = os.path.join(ROOT, "research_highdim_remedies", "results")
v4 = json.load(open(os.path.join(RR, "decision.json")))
v5 = json.load(open(os.path.join(RR, "decision_p5.json")))["extended_verdict_P1_P5"]
check("remedies: each helps on some problems and hurts on others; neither general on the five problems",
      all("HELPS" in v5[a]["verdicts"].values() and "HURTS" in v5[a]["verdicts"].values() and not v5[a]["general_on_P1_P5"]
          for a in ("presolve", "lift3c")) and v4["general_lift3c"] == "GENERAL REMEDY",
      f"presolve {v5['presolve']['verdicts']}; lift3c {v5['lift3c']['verdicts']}; frozen verdict on P1-P4: lift3c {v4['general_lift3c']}")

RX = json.load(open(os.path.join(RR, "summary_exploratory.json")))
RS = json.load(open(os.path.join(RR, "summary.json")))
fl = json.load(open(os.path.join(RR, "checks.json")))["C5_floors"]["poisson/d20"]["test_affine"]
check("plateau on PoiD at d = 20: plain above the affine floor; centred input (lift1, exploratory) below it in every seed, both methods",
      all(RS[f"P2/d20/{m}/plain/4000"]["rel_l2"]["min"] > fl and RX[f"P2/d20/{m}/lift1/4000"]["rel_l2"]["max"] < fl
          for m in ("ritz", "pinn")),
      "; ".join(f"{m}: plain min {RS[f'P2/d20/{m}/plain/4000']['rel_l2']['min']:.4f}, lift1 max "
                f"{RX[f'P2/d20/{m}/lift1/4000']['rel_l2']['max']:.4f}" for m in ("ritz", "pinn")) + f"; floor {fl:.4f}")
check("PoiD at d = 20: lift3c below the affine floor in every seed, both methods; rep3 (Deep Ritz) as well",
      all(RS[f"P2/d20/{m}/lift3c/4000"]["rel_l2"]["max"] < fl for m in ("ritz", "pinn"))
      and RS["P2/d20/ritz/rep3/4000"]["rel_l2"]["max"] < fl,
      "; ".join(f"{m}: lift3c max {RS[f'P2/d20/{m}/lift3c/4000']['rel_l2']['max']:.4f}" for m in ("ritz", "pinn"))
      + f"; ritz rep3 max {RS['P2/d20/ritz/rep3/4000']['rel_l2']['max']:.4f}; floor {fl:.4f}")
R5 = json.load(open(os.path.join(RR, "summary_p5.json")))
check("AltRidgeD (added after the pre-registered test): lift3c less accurate than plain for both methods (median ratio < 1)",
      all(R5[f"P5/d20/{m}/lift3c/4000"]["paired"]["median"] < 1 for m in ("ritz", "pinn")),
      ", ".join(f"{m}: median {R5[f'P5/d20/{m}/lift3c/4000']['paired']['median']:.2f}" for m in ("ritz", "pinn")))
h2 = json.load(open(os.path.join(ROOT, "research_highdim_bound", "results", "decisions.json")))["H2"]
check("pre-registered efficiency target (eta <= 3 for d >= 5) fails on 21 of 57 networks",
      h2["decision"] == "FAILS" and h2["n_tested"] == 57 and len(h2["failures"]) == 21,
      f"{h2['decision']}: {len(h2['failures'])} of {h2['n_tested']}")
check("presolve does not help on PoiD (the plateau problem)",
      max(RS[f"P2/d20/{m}/presolve/4000"]["paired"]["median"] for m in ("ritz", "pinn")) < 1.5,
      ", ".join(f"{m}: median {RS[f'P2/d20/{m}/presolve/4000']['paired']['median']:.2f}" for m in ("ritz", "pinn")))

# ---------------------------------------------------------------- the two pitfalls of Section 7
wb = json.load(open(os.path.join(DATA, "s7_pitfalls_autograd.json")))
tr = wb["training"]
check("residual made zero by differentiation with respect to a slice: PDE term identically zero",
      tr["max_pde_term_over_all_steps"] == 0.0 and tr["none_returns"] == tr["autograd_calls"],
      f"max over {tr['steps']} iterations = {tr['max_pde_term_over_all_steps']}; {tr['none_returns']} of {tr['autograd_calls']} calls None")
ut = wb["operator_unit_test_float64"]
check("faulty operator returns 0 on a solution and on a non-solution; the correct one separates them",
      ut["exact_solution_cos"]["max_abs_faulty"] == 0.0 and ut["non_solution_exp"]["max_abs_faulty"] == 0.0
      and ut["exact_solution_cos"]["max_abs_correct"] < 1e-14 and ut["non_solution_exp"]["max_abs_correct"] > 19,
      json.dumps(ut))
tr = json.load(open(os.path.join(DATA, "s7_pitfalls_autograd_1500passes.json")))["training"]
check("the same with 1500 passes: PDE term identically zero", tr["max_pde_term_over_all_steps"] == 0.0 and tr["steps"] == 235500,
      f"max over {tr['steps']} iterations = {tr['max_pde_term_over_all_steps']}")
of = json.load(open(os.path.join(ROOT, "research_benchmark", "results", "oneface_laplace3d.json")))["oneface_laplace3d"]["rows"]
lo = [min(r["final_bc_loss"], r["final_pde_loss"]) for r in of]
hi = [max(r["final_bc_loss"], r["final_pde_loss"]) for r in of]
dist = [r["rel_l2"] for r in of]
check("data on one face: five seeds, both loss terms between 2.7e-4 and 2.2e-3, distances 1.6 to 13 from Lap3",
      len(of) == 5 and rd(min(lo), 2) == 2.7e-4 and rd(max(hi), 2) == 2.2e-3 and rd(min(dist), 2) == 1.6 and rd(max(dist), 2) == 13,
      f"loss terms {min(lo):.2e} to {max(hi):.2e}; distances " + ", ".join(f"{x:.2f}" for x in dist))

lines.append("ALL HEADLINE CHECKS PASSED" if ok_all else "SOME HEADLINE CHECKS FAILED")
out = "\n".join(lines) + "\n"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", help="optional report path; the default invocation writes no files")
args = parser.parse_args()
if args.output:
    with open(args.output, "w", encoding="utf8") as handle:
        handle.write(out)
sys.stdout.write(out)
sys.exit(0 if ok_all else 1)
