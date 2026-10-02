"""Every number quoted in Section 6, recomputed from the stored per-run records.
No training, no timing.  Writes
    data/s6_highdim_numbers.json    all derived numbers (machine-readable)
    data/s6_highdim_tables.tex      LaTeX table bodies pasted into sections/s6_highdim.tex
Inputs: research_highdim/results/{runs*.jsonl,cost_vs_d.json}, verify_research_highdim/results/
{v_bench.json,v_determinism.txt}, data/{closed_forms.json,s6_highdim_long_d20.jsonl,
s6_highdim_diag_d20.json}.
"""
import glob
import json
import math
import re
import statistics as st
from pathlib import Path

ART = Path(__file__).resolve().parents[1]             # folder holding code/, data/, figures/
ROOT = ART if (ART / "research_benchmark").is_dir() else ART.parent   # repository root
HD = ROOT / "research_highdim" / "results"
VR = ROOT / "verify_research_highdim" / "results"
DATA = ART / "data"


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


R = []
for f in sorted(glob.glob(str(HD / "runs*.jsonl"))):
    R += load_jsonl(f)
LONG20 = load_jsonl(DATA / "s6_highdim_long_d20.jsonl") if (DATA / "s6_highdim_long_d20.jsonl").exists() else []
FL = json.load(open(DATA / "closed_forms.json"))["floors"]
DS = [2, 3, 5, 10, 20]
P = ["laplace", "poisson"]
M = ["pinn", "ritz"]
NUM = {}
TEX = []


def sel(recs, **kw):
    out = [r for r in recs if all(r[k] == v for k, v in kw.items())]
    return sorted(out, key=lambda r: r["seed"])


def stats(v):
    return dict(values=v, mean=st.mean(v), sd=st.stdev(v) if len(v) > 1 else None, min=min(v), max=max(v),
                geomean=math.exp(st.mean(map(math.log, v))), n=len(v))


def slope(xs, ys):
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = st.mean(lx), st.mean(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


# ----------------------------------------------------------------- main grid (4000 iterations)
main = {}
for p in P:
    for m in M:
        for d in DS:
            rs = sel(R, phase="main", problem=p, method=m, d=d, iters=4000)
            assert len(rs) == 3, (p, m, d, len(rs))
            main[(p, m, d)] = {k: stats([r[k] for r in rs]) for k in
                               ["rel_l2", "rel_l2_centered", "rel_h1semi", "bd_rel_l2"]}
            main[(p, m, d)]["w"] = rs[0]["w"]
NUM["main"] = {f"{p}/{m}/{d}": v for (p, m, d), v in main.items()}
NUM["ratio_ritz_over_pinn"] = {}
NUM["ranges_disjoint"] = {}
for p in P:
    for d in DS:
        a, b = main[(p, "pinn", d)]["rel_l2"], main[(p, "ritz", d)]["rel_l2"]
        NUM["ratio_ritz_over_pinn"][f"{p}/{d}"] = b["mean"] / a["mean"]
        NUM["ranges_disjoint"][f"{p}/{d}"] = (a["max"] < b["min"]) or (b["max"] < a["min"])
NUM["slopes_geomean"] = {}
for p in P:
    for m in M:
        for k in ["rel_l2", "rel_l2_centered"]:
            g = [main[(p, m, d)][k]["geomean"] for d in DS]
            NUM["slopes_geomean"][f"{p}/{m}/{k}"] = dict(d2_10=slope(DS[:4], g[:4]), d2_20=slope(DS, g))


def u2(x, nd=3):          # value in units of 1e-2
    return f"{100 * x:.{nd}f}"


def floor(p, d, kind):
    return FL[str(d)][("P1_" if p == "laplace" else "P2_") + kind]


for p in P:
    TEX.append(f"% ---- table {('p1' if p == 'laplace' else 'p2')}: all entries in units of 1e-2")
    for d in DS:
        a, b = main[(p, "pinn", d)], main[(p, "ritz", d)]
        ratio = NUM["ratio_ritz_over_pinn"][f"{p}/{d}"]
        cells = [str(d),
                 f"${u2(a['rel_l2']['mean'])}\\pm{u2(a['rel_l2']['sd'])}$",
                 f"${u2(b['rel_l2']['mean'])}\\pm{u2(b['rel_l2']['sd'])}$",
                 f"{ratio:.2f}",
                 u2(a["rel_h1semi"]["mean"], 2), u2(b["rel_h1semi"]["mean"], 2)]
        if p == "laplace":
            cells += [u2(a["rel_l2_centered"]["mean"], 2), u2(b["rel_l2_centered"]["mean"], 2)]
        cells += [u2(floor(p, d, "affine"), 1)]
        if p == "laplace":
            cells += [u2(floor(p, d, "const"), 1)]
        TEX.append(" & ".join(cells) + r" \\")
        rng = ["", f"{{\\footnotesize $[{u2(a['rel_l2']['min'])},\\,{u2(a['rel_l2']['max'])}]$}}",
               f"{{\\footnotesize $[{u2(b['rel_l2']['min'])},\\,{u2(b['rel_l2']['max'])}]$}}"]
        rng += [""] * (len(cells) - 3)
        TEX.append(" & ".join(rng) + r" \\" + (r"[2pt]" if d != 20 else ""))

# ----------------------------------------------------------------- plateau values (validation curves)
NUM["plateau_val_mean"] = {}
for p in P:
    for m in M:
        for d in [10, 20]:
            rs = sel(R, phase="main", problem=p, method=m, d=d, iters=4000)
            NUM["plateau_val_mean"][f"{p}/{m}/{d}"] = {
                str(it): st.mean(next(c["val_rel_l2"] for c in r["curve"] if c["it"] == it) for r in rs)
                for it in [250, 500, 1000, 1500, 2000, 3000, 4000]}

# ----------------------------------------------------------------- budget: long and equal-compute runs
ext = {}
for r in R:
    if r["phase"] in ("long", "equaltime"):
        ext.setdefault((r["phase"], r["problem"], r["method"], r["d"], r["iters"]), []).append(r)
for r in LONG20:
    ext.setdefault(("long", r["problem"], r["method"], r["d"], r["iters"]), []).append(r)
NUM["extensions"] = {}
for key, rs in sorted(ext.items()):
    rs = sorted(rs, key=lambda r: r["seed"])
    NUM["extensions"]["/".join(map(str, key))] = dict(
        seeds=[r["seed"] for r in rs],
        **{k: stats([r[k] for r in rs]) for k in ["rel_l2", "rel_l2_centered", "rel_h1semi", "bd_rel_l2"]})
# iteration rule of the equal-compute runs (run_study.py): mean PINN CPU time / mean Ritz CPU ms per iteration
NUM["equal_compute_rule"] = {}
for p in P:
    tp = [r["cpu_time_s"] for r in sel(R, phase="main", problem=p, method="pinn", d=10, iters=4000)]
    mr = [r["cpu_ms_per_iter"] for r in sel(R, phase="main", problem=p, method="ritz", d=10, iters=4000)]
    raw = st.mean(tp) / (st.mean(mr) / 1e3)
    NUM["equal_compute_rule"][p] = dict(raw_iters=raw, rounded=int(round(raw / 1000.0)) * 1000, ratio=raw / 4000)
NUM["n5_val_curve_mean"] = {}
for p in P:
    for m in M:
        rs = sel(LONG20, problem=p, method=m)
        if rs:
            NUM["n5_val_curve_mean"][f"{p}/{m}"] = {
                str(it): st.mean(next(c["val_rel_l2"] for c in r["curve"] if c["it"] == it) for r in rs)
                for it in [250, 1000, 2000, 4000, 8000, 12000, 16000]}


def pm(key, metric="rel_l2"):
    e = NUM["extensions"].get(key)
    if e is None:
        return "--"
    return f"${u2(e[metric]['mean'])}\\pm{u2(e[metric]['sd'])}$"


def pm_main(p, m, d):
    s = main[(p, m, d)]["rel_l2"]
    return f"${u2(s['mean'])}\\pm{u2(s['sd'])}$"


TEX.append("% ---- table budget: relative L2 error in units of 1e-2, mean +- s.d. over seeds 0-2")
eq_iters = {p: NUM["equal_compute_rule"][p]["rounded"] for p in P}
for p, d in [("laplace", 10), ("poisson", 10), ("laplace", 20), ("poisson", 20)]:
    name = "\\Pone" if p == "laplace" else "\\Ptwo"
    w = {"laplace": 100.0, "poisson": 1.0}[p]
    eq = pm(f"equaltime/{p}/ritz/{d}/{eq_iters[p]}") if d == 10 else "--"
    eq_n = (f"{eq_iters[p]:,}".replace(",", "\\,") if eq_iters[p] >= 10000 else f"{eq_iters[p]}") if d == 10 else ""
    TEX.append(" & ".join([name, str(d), pm_main(p, "pinn", d), pm_main(p, "ritz", d),
                           pm(f"long/{p}/pinn/{d}/16000"), pm(f"long/{p}/ritz/{d}/16000"),
                           eq + (f" ({eq_n})" if eq_n else ""), u2(floor(p, d, "affine"), 1)]) + r" \\")

# ----------------------------------------------------------------- cost per iteration
def cost_row(d, row):
    """d | Ritz ms | forward: ms, ratio, [min, max] | nested: ms, ratio, [min, max] | nested/forward"""
    return (f"{d} & {row['ritz']:.2f} & {row['fwd']:.2f} & {row['ratio_fwd'][1]:.2f} & "
            f"[{row['ratio_fwd'][0]:.2f}, {row['ratio_fwd'][2]:.2f}] & {row['nested']:.2f} & "
            f"{row['ratio_nested'][1]:.2f} & [{row['ratio_nested'][0]:.2f}, {row['ratio_nested'][2]:.2f}] & "
            f"{row['nested_over_fwd_medians']:.2f} \\\\")


COST = json.load(open(HD / "cost_vs_d.json"))
VB = json.load(open(VR / "v_bench.json"))
cost = {}
for c in COST:
    cost.setdefault(c["d"], {})[c["method"]] = c
NUM["cost_primary"] = {}
NUM["cost_verifier"] = {}
TEX.append("% ---- table cost (a): primary benchmark, CPU ms per iteration (median of 7 rounds), paired ratio median [min, max]")
for d in sorted(cost):
    c = cost[d]
    # nested/forward: paired per round
    nf = [a / b for a, b in zip(c["pinn_nested"]["cpu_ms_rounds"], c["pinn_forward"]["cpu_ms_rounds"])]
    row = dict(ritz=c["ritz"]["cpu_ms_median"], fwd=c["pinn_forward"]["cpu_ms_median"],
               nested=c["pinn_nested"]["cpu_ms_median"],
               ratio_fwd=[c["pinn_forward"][f"ratio_to_ritz_{k}"] for k in ("min", "median", "max")],
               ratio_nested=[c["pinn_nested"][f"ratio_to_ritz_{k}"] for k in ("min", "median", "max")],
               nested_over_fwd_medians=c["pinn_nested"]["cpu_ms_median"] / c["pinn_forward"]["cpu_ms_median"],
               nested_over_fwd_paired=[min(nf), st.median(nf), max(nf)])
    NUM["cost_primary"][str(d)] = row
    TEX.append(cost_row(d, row))
TEX.append("% ---- table cost (b): benchmark of the verification code V-dim")
for d in sorted(int(k) for k in VB):
    v = VB[str(d)]
    row = dict(ritz=v["cpu_ms"]["ritz"], fwd=v["cpu_ms"]["fwd"], nested=v["cpu_ms"]["nested"],
               ratio_fwd=v["ratio_fwd"], ratio_nested=v["ratio_nested"],
               nested_over_fwd_medians=v["cpu_ms"]["nested"] / v["cpu_ms"]["fwd"])
    NUM["cost_verifier"][str(d)] = row
    TEX.append(cost_row(d, row))
NUM["cost_growth_d2_to_d100"] = {
    "primary": {k: NUM["cost_primary"]["100"][k] / NUM["cost_primary"]["2"][k] for k in ("ritz", "fwd", "nested")},
    "verifier": {k: NUM["cost_verifier"]["100"][k] / NUM["cost_verifier"]["2"][k] for k in ("ritz", "fwd", "nested")}}
NUM["nested_over_fwd_d_ge_10"] = {
    "primary": [NUM["cost_primary"][str(d)]["nested_over_fwd_medians"] for d in (10, 20, 50, 100)],
    "verifier": [NUM["cost_verifier"][str(d)]["nested_over_fwd_medians"] for d in (10, 20, 50, 100)]}
# load-dependence of process CPU time: bit-identical reruns by the verification code V-dim
rer = []
for line in open(VR / "v_determinism.txt"):
    mm = re.search(r"cpu ([\d.]+)s \(stored ([\d.]+)s\)", line)
    if mm:
        rer.append(float(mm.group(1)) / float(mm.group(2)))
NUM["cpu_rerun_over_stored"] = dict(values=rer, min=min(rer), max=max(rer), n=len(rer))

# ----------------------------------------------------------------- penalty-weight sweep (d = 5)
NUM["sweep_val"] = {}
for p in P:
    for m in M:
        rs = sorted([r for r in R if r["phase"] == "sweep" and r["problem"] == p and r["method"] == m],
                    key=lambda r: r["w"])
        NUM["sweep_val"][f"{p}/{m}"] = {f"{r['w']:g}": r["val_rel_l2"] for r in rs}

# ----------------------------------------------------------------- diagnostics at d = 20
dp = DATA / "s6_highdim_diag_d20.json"
if dp.exists():
    diag = json.load(open(dp))
    NUM["diag_d20"] = {}
    for p in P:
        for m in M:
            for it in (4000, 16000):
                rs = [r for r in diag if r["problem"] == p and r["method"] == m and r["iters"] == it]
                if rs:
                    NUM["diag_d20"][f"{p}/{m}/{it}"] = {
                        k: [min(r[k] for r in rs), max(r[k] for r in rs)]
                        for k in ["rel_l2", "nonaffine_ratio", "nonaffine_corr", "affine_err", "lap_rms", "bd_rel_l2"]
                        + (["resid_ratio"] if p == "poisson" else [])}

# ----------------------------------------------------------------- derived ratios quoted in the text
def emean(key):
    return NUM["extensions"][key]["rel_l2"]["mean"]


NUM["budget_ratios"] = {}
for p, d in [("laplace", 10), ("laplace", 20), ("poisson", 20)]:
    kp, kr = f"long/{p}/pinn/{d}/16000", f"long/{p}/ritz/{d}/16000"
    if kp in NUM["extensions"] and kr in NUM["extensions"]:
        a, b = NUM["extensions"][kp]["rel_l2"], NUM["extensions"][kr]["rel_l2"]
        NUM["budget_ratios"][f"{p}/{d}"] = dict(
            ritz_over_pinn_16000=b["mean"] / a["mean"],
            ranges_disjoint_16000=(a["max"] < b["min"]) or (b["max"] < a["min"]),
            pinn_4000_over_16000=main[(p, "pinn", d)]["rel_l2"]["mean"] / a["mean"],
            ritz_4000_over_16000=main[(p, "ritz", d)]["rel_l2"]["mean"] / b["mean"])
# growth of the error from d = 10 to d = 20 on P1, at 4000 and at 16 000 iterations (ratio of means)
NUM["growth_P1_d10_to_d20"] = {
    m: dict(it4000=main[("laplace", m, 20)]["rel_l2"]["mean"] / main[("laplace", m, 10)]["rel_l2"]["mean"],
            it16000=emean(f"long/laplace/{m}/20/16000") / emean(f"long/laplace/{m}/10/16000"))
    for m in M if f"long/laplace/{m}/20/16000" in NUM["extensions"]}
NUM["equal_compute_gain"] = {
    p: dict(ritz_equal_over_pinn_4000=emean(f"equaltime/{p}/ritz/10/{eq_iters[p]}") / main[(p, "pinn", 10)]["rel_l2"]["mean"],
            ranges_disjoint=NUM["extensions"][f"equaltime/{p}/ritz/10/{eq_iters[p]}"]["rel_l2"]["max"]
            < main[(p, "pinn", 10)]["rel_l2"]["min"])
    for p in P}
# bound of the remark in Section 6: ||e_0|| / ||u*|| <= (||Lap u + f|| / ||f||) / d   (P2)
if "diag_d20" in NUM:
    NUM["e0_bound_P2_d20"] = {k: NUM["diag_d20"][k]["resid_ratio"][1] / 20 for k in NUM["diag_d20"] if k.startswith("poisson")}

# ----------------------------------------------------------------- table: weight sweep at d = 5
def u2s(x):
    """Value in units of 1e-2 with three significant digits."""
    return f"{100 * x:#.3g}".rstrip(".")


SWEEP_TEX = []
for p in P:
    for m in M:
        v = NUM["sweep_val"][f"{p}/{m}"]
        assert list(v) == ["1", "10", "100", "1000"]
        name = ("\\Pone" if p == "laplace" else "\\Ptwo") + ", " + ("PINN ($\\lam$)" if m == "pinn" else "Deep Ritz ($\\bet$)")
        SWEEP_TEX.append(name + " & " + " & ".join(u2s(v[w]) for w in v) + r" \\")
TEX.append("% ---- table sweep: validation relative L2 error in units of 1e-2 at d = 5, seed 100, weights 1, 10, 100, 1000")
TEX += SWEEP_TEX

# ----------------------------------------------------------------- table: runs of the verification code V-dim (4000 iterations)
import ast
VD = {}
for line in open(VR / "v_summary.txt"):
    if line.startswith("(") and "seeds [" in line:
        key = ast.literal_eval(line[:line.index(")") + 1])
        seeds = ast.literal_eval(line[line.index("seeds ") + 6:line.index("]") + 1])
        vals = [float(x) for x in line[line.index("]") + 1:line.index("|")].split()]
        assert len(seeds) == len(vals)
        VD[key] = dict(zip(seeds, vals))
NUM["vdim_main"] = {"/".join(str(x) for x in k[1:4]): v for k, v in VD.items() if k[0] in ("main", "main_fwd")}
NUM["vdim_main_fwd_P1_d20"] = VD.get(("main_fwd", "laplace", "pinn", 20, 1000.0, 4000))
import statistics as _st
v20 = list(VD[("main", "laplace", "ritz", 20, 100.0, 4000)].values())
NUM["vdim_P1_d20_ritz_sd_5_seeds"] = _st.stdev(v20)


def seedlist(d):
    return ", ".join(u2s(x) for _, x in sorted(d.items()))


def seedrange(d):
    k = sorted(d)
    return f"{k[0]}" if len(k) == 1 else f"{k[0]}--{k[-1]}"


VDIM_TEX = []
for p, wp, wr in (("laplace", 1000.0, 100.0), ("poisson", 1000.0, 1.0)):
    for d in (2, 10, 20):
        a = VD[("main", p, "pinn", d, wp, 4000)]
        b = VD[("main", p, "ritz", d, wr, 4000)]
        name = "\\Pone" if p == "laplace" else "\\Ptwo"
        VDIM_TEX.append(f"{name} & {d} & {seedlist(a)} & {seedrange(a)} & {seedlist(b)} & {seedrange(b)} \\\\")
TEX.append("% ---- table vdim: relative L2 error in units of 1e-2, individual seeds of the verification code V-dim, 4000 iterations")
TEX += VDIM_TEX

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject import inject
inject("s6_highdim.tex", "s6_highdim:tab:sweep", "\n".join(SWEEP_TEX))
inject("s6_highdim.tex", "s6_highdim:tab:vdim", "\n".join(VDIM_TEX))

json.dump(NUM, open(DATA / "s6_highdim_numbers.json", "w"), indent=1)
open(DATA / "s6_highdim_tables.tex", "w").write("\n".join(TEX) + "\n")
print("\n".join(TEX))
print(json.dumps({k: NUM[k] for k in ["ratio_ritz_over_pinn", "ranges_disjoint", "slopes_geomean", "equal_compute_rule",
                                      "cost_growth_d2_to_d100", "nested_over_fwd_d_ge_10", "cpu_rerun_over_stored",
                                      "sweep_val", "n5_val_curve_mean", "budget_ratios", "growth_P1_d10_to_d20", "equal_compute_gain"]
                  + (["e0_bound_P2_d20"] if "e0_bound_P2_d20" in NUM else [])}, indent=1))
for k, v in NUM["extensions"].items():
    print(k, v["seeds"], ["%.4e" % x for x in v["rel_l2"]["values"]], "mean %.4e sd %.2e" % (v["rel_l2"]["mean"], v["rel_l2"]["sd"]),
          "centred %.4e" % v["rel_l2_centered"]["mean"], "H1 %.4e" % v["rel_h1semi"]["mean"])
print(json.dumps(NUM["plateau_val_mean"], indent=0))
if "diag_d20" in NUM:
    print(json.dumps(NUM["diag_d20"], indent=0))
