"""Supplementary Section S8: check that every number quoted in the prose of sections/S8_details.tex is the one in
data/sC_details_numbers.json (written by sC_details_tables.py from the stored results).

No training, no timing.  Usage:  python code/sC_details_check.py
Exits with an error if a quoted number is not found in the text in the expected form.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
N = json.load(open(os.path.join(ART, "data", "sC_details_numbers.json")))
tex = open(os.path.join(ART, "sections", "S8_details.tex")).read()
# prose only: drop the tables and the comment lines, normalise white space
prose = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", tex, flags=re.S)
prose = re.sub(r"(?m)^\s*%.*$", " ", prose)
prose = " ".join(prose.split())


def sci(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    return f"\\sci{{{m}}}{{{int(e)}}}"


checks = []


def need(label, text):
    ok = " ".join(text.split()) in prose
    checks.append((ok, label, text))


v = N["validation"]
h, l = v["heat1d"], v["laplace2d"]
need("networks", f"The {N['n_networks']} networks of the benchmark were trained in six stacks")
assert N["benchmark_stack_wall_s"]["n_stacks"] == 6
assert h["n"] + l["n"] == 20 and h["seeds"] == [0, 1]
need("configs", "seeds 0 and 1: 20 configurations")
assert max(h["adam_reldiff_it1_max"], l["adam_reldiff_it1_max"]) == 0.0
need("it50", f"by at most ${sci(max(h['adam_reldiff_it50_max'], l['adam_reldiff_it50_max']))}$ at iteration 50")
need("it100", f"at most ${sci(max(h['adam_reldiff_it100_max'], l['adam_reldiff_it100_max']))}$ at iteration 100")
need("adam ratio", f"lies between {min(h['adam_ratio_min'], l['adam_ratio_min']):.2f} and {max(h['adam_ratio_max'], l['adam_ratio_max']):.2f}")
need("lbfgs H", f"mean {h['lbfgs_ratio_geo']:.2f} (range {h['lbfgs_ratio_min']:.2f} to {h['lbfgs_ratio_max']:.2f}) on \\PH")
need("lbfgs L2", f"{l['lbfgs_ratio_geo']:.2f} ({l['lbfgs_ratio_min']:.2f} to {l['lbfgs_ratio_max']:.2f}) on \\PLtwo")
pct = sorted(round(100 * (x["lbfgs_ratio_geo"] - 1)) for x in (h, l))
need("lbfgs deficit", f"version is {pct[0]} to {pct[1]} per cent less accurate in geometric mean")
assert max(h["adam_ratio_max"], l["adam_ratio_max"]) > max(h["lbfgs_ratio_max"], l["lbfgs_ratio_max"])
assert min(h["adam_ratio_min"], l["adam_ratio_min"]) < min(h["lbfgs_ratio_min"], l["lbfgs_ratio_min"])
assert N["validation_predates_single_pass_loss"] is True
need("caveat", "were run before the loss was re-assembled into one forward pass")
i = N["independent_vs_stacked"]
need("27 configs", f"shares {i['n']} configurations")
need("0.98", f"geometric mean {i['geo']:.2f} (range {i['min']:.2f} to {i['max']:.2f})")
assert max(N["loss_assembly"]["single_vs_termwise_max"] + N["loss_assembly"]["stacked_vs_single_max"]) == 0.0
need("loss", "(largest relative difference 0)")

f = N["fd"]
need("fd order", f"the observed order is {f['order_min_without_L3']:.2f} to {f['order_max_without_L3']:.2f} for \\PH")
need("fd order L3", " and ".join([", ".join(f"{x:.2f}" for x in f["orders"]["laplace3d"][:-1]), f"{f['orders']['laplace3d'][-1]:.2f}"]) + " for \\PLthree")
c = N["cost_summary"]
need("load", f"rose from {c['loadavg_A_start'][0]:.0f} to {c['loadavg_A_end'][0]:.0f}")
need("A over B", f"A is {c['A_over_B_median_range'][0]:.1f} to {c['A_over_B_median_range'][1]:.1f} times the median of B")
need("ratio A", f"is ${sci(c['ratio_A_range'][0])}$ to ${sci(c['ratio_A_range'][1])}$ with A")
need("ratio B", f"${sci(c['ratio_B_range'][0])}$ to ${sci(c['ratio_B_range'][1])}$ with B")
need("ratio C", f"and ${sci(c['ratio_C_range'][0])}$ to ${sci(c['ratio_C_range'][1])}$ with C")
need("run C", f"{c['run_C_range_s'][0]:.1f} to {c['run_C_range_s'][1]:.1f}~s per network")
w = N["verifier_lowdim"]
assert w["n_early_stops_below_eval_cap"] == w["n_lbfgs_stopped_early"]
B = w["B"]
assert sorted(p for p, n in B["stopped_early_by_problem"].items() if n) == ["heat1d", "laplace2d"]
assert sum(c["run_B_s"]["n"] for c in N["cost"].values()) == B["n_runs"]
need("B runs", f"the {B['n_runs']} runs of \\textsf{{V-bench}}")
need("early stops", f"{B['n_stopped_early']} of them (on \\PH\\ and \\PLtwo) stopped on the \\LBFGS\\ tolerance before iteration 1500, the earliest after {B['min_lbfgs_iters']}")
need("39/40", f"won {w['lbfgs_better']} of the {w['n_runs']} pairs")
need("40 configs", f"trained {w['n_runs']} configurations with \\textsf{{V-bench}}")

b = N["bit_for_bit"]
assert b["n_cells"] == b["n_identical"] == 7
need("seven", "repeated seven of them")
s = N["superseded_heat_block"]
assert s["n"] == s["n_identical_loss_at_iteration_1"] == 25 and s["adam"]["n_identical"] == s["adam_lbfgs"]["n_identical"] == 0
need("25", "identical in all 25 networks and the final error in none")
a, al = s["adam"], s["adam_lbfgs"]
need("superseded lbfgs", f"geometric mean {al['ratio_geo']:.2f} (range {al['ratio_min']:.2f} to {al['ratio_max']:.2f}) after \\AdamLBFGS")
need("superseded adam", f"{a['ratio_geo']:.2f} ({a['ratio_min']:.2f} to {a['ratio_max']:.1f}) after \\Adam")
sw = N["software"]
need("software", f"Python {sw['python']}, PyTorch {sw['torch']}, NumPy {sw['numpy']} and SciPy~{sw['scipy']}")
need("cores", f"{sw['cpu_count']}-core")
k = N["benchmark_stack_wall_s"]
need("stack wall", f"took {k['min']:.0f} to {k['max']:.0f} s each ({k['sum']:.0f} s together)")
hd = N["highdim_config"]
need("88 runs", f"The {hd['n_runs']} runs of the dimension study sum to {hd['sum_wall_s']:.0f} s")
need("350", "re-aggregated the 350 rows")
# footnote b of Table sC_details:tab:hyper (13 of 16 runs on four threads; five PINN runs with the nested Laplacian)
assert hd["n_without_cpu_timer"] == 13 and hd["sweep_pinn_runs_nested"] == 5 and hd["phases"]["sweep"]["n"] == 16
assert N["stacked_lbfgs"] == {"armijo_c1": 1e-4, "max_backtracks": 20, "memory": 50}
need("armijo", "Armijo backtracking with constant $10^{-4}$")
need("20 trials", "(at most 20 trials)")

bad = [(lab, t) for ok, lab, t in checks if not ok]
for ok, lab, t in checks:
    print("ok  " if ok else "FAIL", lab, "|", t)
if bad:
    raise SystemExit(f"{len(bad)} quoted number(s) not found in the prose")
print(f"all {len(checks)} checks passed")
