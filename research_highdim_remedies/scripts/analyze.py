"""Analysis exactly as frozen in PREREG_C1.md (sections 6, 7). Reads results/runs_*.jsonl (counted phases only);
writes results/per_seed.csv, results/summary.csv, results/summary.json, results/decision.json,
results/tables.tex. Re-runnable at any time; incomplete cells are marked incomplete."""
import sys, os, json, glob, csv, statistics, math
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
PID = {"laplace": "P1", "poisson": "P2", "ridge": "P3", "cospair": "P4"}
PROBS = ["laplace", "poisson", "ridge", "cospair"]
COUNTED = ("main20", "eqcpu", "stall", "d10")
SEEDS = [10, 11, 12, 13, 14]

runs = []
for f in sorted(glob.glob(os.path.join(RES, "runs_*.jsonl"))):
    runs += [json.loads(l) for l in open(f) if l.strip()]
runs = [r for r in runs if r["phase"] in COUNTED]

# ------------------------------------------------------------------ per-seed table
cols = ["phase", "problem_id", "problem", "method", "arm", "d", "seed", "w", "iters", "schedule", "rel_l2",
        "rel_l2_centered", "bd_rel_l2", "max_abs", "cpu_train_s", "cpu_shift_s", "cpu_total_s", "T10", "T_half",
        "gamma_end", "load1"]
with open(os.path.join(RES, "per_seed.csv"), "w", newline="") as fh:
    wr = csv.writer(fh, lineterminator="\n"); wr.writerow(cols)
    for r in sorted(runs, key=lambda r: (r["phase"], r["d"], r["problem"], r["method"], r["arm"], r["iters"], r["seed"])):
        wr.writerow([PID[r["problem"]] if c == "problem_id" else r.get(c) for c in cols])


def get(phase, problem, method, arm, d, iters=None, schedule=True):
    out = {}
    for r in runs:
        if (r["phase"] == phase and r["problem"] == problem and r["method"] == method and r["arm"] == arm
                and r["d"] == d and r["schedule"] == schedule and (iters is None or r["iters"] == iters)):
            out[r["seed"]] = r
    return out


def stats(v):
    return dict(n=len(v), mean=statistics.mean(v), sd=statistics.stdev(v) if len(v) > 1 else float("nan"),
                min=min(v), max=max(v)) if v else dict(n=0)


def paired(base, other, metric="rel_l2"):
    seeds = sorted(set(base) & set(other))
    rho = [base[s][metric] / other[s][metric] for s in seeds]
    if not rho:
        return dict(n=0)
    return dict(n=len(rho), seeds=seeds, rho=rho, median=statistics.median(rho), min=min(rho), max=max(rho),
                wins=sum(x > 1 for x in rho), losses=sum(x < 1 for x in rho))


# ------------------------------------------------------------------ summary cells
summary, rows = {}, []
for d, phase, seeds_needed in ((20, "main20", 5), (10, "d10", 3)):
    for problem in PROBS:
        for method in ("ritz", "pinn"):
            base = get(phase, problem, method, "plain", d, 4000)
            if not base:
                continue
            for arm in ("plain", "presolve", "lift3c", "lift3u", "rep3"):
                for budget, ph2 in (("4000", phase), ("eqcpu", "eqcpu")):
                    if budget == "eqcpu" and (d != 20 or arm not in ("lift3c", "presolve")):
                        continue
                    cell = get(ph2, problem, method, arm, d, 4000 if budget == "4000" else None)
                    if not cell:
                        continue
                    k = f"{PID[problem]}/d{d}/{method}/{arm}/{budget}"
                    e = [cell[s]["rel_l2"] for s in sorted(cell)]
                    ec = [cell[s]["rel_l2_centered"] for s in sorted(cell)]
                    cpu = [cell[s]["cpu_total_s"] for s in sorted(cell)]
                    rec = dict(problem=PID[problem], d=d, method=method, arm=arm, budget=budget,
                               iters=sorted({cell[s]["iters"] for s in cell}), complete=len(cell) >= seeds_needed,
                               rel_l2=stats(e), rel_l2_centered=stats(ec), cpu_total_s=stats(cpu),
                               T_half=[cell[s]["T_half"] for s in sorted(cell)],
                               gamma_end=[cell[s]["gamma_end"] for s in sorted(cell)],
                               per_seed={s: cell[s]["rel_l2"] for s in sorted(cell)})
                    if arm != "plain":
                        rec["paired"] = paired(base, cell)
                        rec["paired_centered"] = paired(base, cell, "rel_l2_centered")
                    summary[k] = rec
                    p = rec.get("paired", {})
                    rows.append([PID[problem], d, method, arm, budget, "/".join(map(str, rec["iters"])), rec["rel_l2"]["n"],
                                 rec["rel_l2"]["mean"], rec["rel_l2"]["sd"], rec["rel_l2"]["min"], rec["rel_l2"]["max"],
                                 rec["rel_l2_centered"]["mean"], rec["cpu_total_s"]["mean"],
                                 p.get("median", ""), p.get("min", ""), p.get("max", ""),
                                 f"{p['wins']}/{p['n']}" if p.get("n") else ""])
with open(os.path.join(RES, "summary.csv"), "w", newline="") as fh:
    wr = csv.writer(fh, lineterminator="\n")
    wr.writerow(["problem", "d", "method", "arm", "budget", "iters", "n_seeds", "rel_l2_mean", "rel_l2_sd", "rel_l2_min",
                 "rel_l2_max", "rel_l2_centered_mean", "cpu_total_s_mean", "rho_median", "rho_min", "rho_max", "wins"])
    wr.writerows(rows)

# ------------------------------------------------------------------ decision rule (PREREG_C1.md section 7)
eqplan = json.load(open(os.path.join(RES, "eqcpu_plan.json"))) if os.path.exists(os.path.join(RES, "eqcpu_plan.json")) else {}


def cond(p):
    return bool(p.get("n") == 5 and p["wins"] >= 4 and p["median"] >= 1.5)


decision = {"rule": "PREREG_C1.md section 7", "per_problem": {}, "complete": True}
for arm in ("presolve", "lift3c"):
    for problem in PROBS:
        k = f"{arm}/{PID[problem]}"
        det, helps, hurts, complete = {}, True, False, True
        for method in ("ritz", "pinn"):
            p4 = summary.get(f"{PID[problem]}/d20/{method}/{arm}/4000", {}).get("paired", {})
            if arm == "lift3c":
                peq = summary.get(f"{PID[problem]}/d20/{method}/{arm}/eqcpu", {}).get("paired", {})
                eq_source = "eqcpu runs"
            else:
                plan = eqplan.get(f"{problem}/{method}/presolve", {})
                if plan.get("iters") == 0:
                    peq, eq_source = p4, f"4000-it runs (CPU ratio {plan.get('ratio', float('nan')):.3f} <= 1.05)"
                else:
                    peq = summary.get(f"{PID[problem]}/d20/{method}/{arm}/eqcpu", {}).get("paired", {})
                    eq_source = f"eqcpu runs ({plan})"
            complete &= p4.get("n") == 5 and peq.get("n") == 5
            c4, ceq = cond(p4), cond(peq)
            helps &= c4 and ceq
            hm4 = bool(p4.get("n")) and p4["median"] <= 1 / 1.1
            separate_eq = arm == "lift3c" or bool(eqplan.get(f"{problem}/{method}/presolve", {}).get("iters"))
            hmeq = separate_eq and bool(peq.get("n")) and peq["median"] <= 1 / 1.1
            hm = hm4 or hmeq
            hurts |= bool(hm)
            det[method] = dict(at4000={k2: p4.get(k2) for k2 in ("median", "min", "max", "wins", "n")}, cond4000=c4,
                               eqcpu={k2: peq.get(k2) for k2 in ("median", "min", "max", "wins", "n")}, condeq=ceq,
                               eq_source=eq_source, hurts_here=bool(hm))
        verdict = "HELPS" if helps else ("HURTS" if hurts else "NEUTRAL/MIXED")
        if helps and hurts:
            verdict = "HELPS (and a hurt condition also met; see detail)"
        decision["per_problem"][k] = dict(verdict=verdict if complete else verdict + " (INCOMPLETE)", detail=det,
                                          feature_matched=(problem == "poisson"), complete=complete)
        decision["complete"] &= complete

for arm in ("presolve", "lift3c"):
    V = {p: decision["per_problem"][f"{arm}/{p}"]["verdict"] for p in ("P1", "P3", "P4")}
    general = V["P1"].startswith("HELPS") and (V["P3"].startswith("HELPS") or V["P4"].startswith("HELPS")) and \
        not any(v.startswith("HURTS") for v in V.values())
    decision[f"general_{arm}"] = "GENERAL REMEDY" if general else "PROBLEM-SPECIFIC (every cell shown)"

# centring clause
pc = summary.get("P1/d20/ritz/lift3c/4000", {}).get("paired", {})
pu = summary.get("P1/d20/ritz/lift3u/4000", {}).get("paired", {})
if pc.get("n") == 5 and pu.get("n") == 5:
    conflict = (pc["wins"] >= 4 and pu["losses"] >= 4) or (pc["losses"] >= 4 and pu["wins"] >= 4)
    decision["centring_clause"] = dict(lift3c=dict(wins=pc["wins"], losses=pc["losses"], median=pc["median"]),
                                       lift3u=dict(wins=pu["wins"], losses=pu["losses"], median=pu["median"]),
                                       triggered=conflict,
                                       consequence=("P1 lift effect attributed to input centring/scale; no statement "
                                                    "about Legendre features on P1") if conflict else "not triggered")
pr = summary.get("P1/d20/ritz/rep3/4000", {}).get("paired", {})
if pr.get("n"):
    decision["rep3_P1_descriptive"] = {k2: pr.get(k2) for k2 in ("median", "min", "max", "wins", "n")}

# failure flags
anyhelp34 = any(decision["per_problem"][f"{a}/{p}"]["verdict"].startswith("HELPS") for a in ("presolve", "lift3c")
                for p in ("P3", "P4"))
pre_p1 = decision["per_problem"]["presolve/P1"]["detail"]
decision["failure_flags"] = dict(
    neither_helps_on_P3_or_P4=not anyhelp34,
    presolve_P1_gain_disappears_at_equal_cpu=bool(all(pre_p1[m]["cond4000"] for m in ("ritz", "pinn"))
                                                  and not all(pre_p1[m]["condeq"] for m in ("ritz", "pinn"))),
    repro_failed=not json.load(open(os.path.join(RES, "repro_check.json")))["all_bit_identical"])

# stall check
st = {a: get("stall", "poisson", "ritz", a, 20, 12000, schedule=False) for a in ("plain", "presolve")}
if st["plain"] and st["presolve"]:
    gp = [st["plain"][s]["gamma_end"] for s in sorted(st["plain"])]
    gq = [st["presolve"][s]["gamma_end"] for s in sorted(st["presolve"])]
    if len(gp) == 3 and len(gq) == 3:
        v = ("CONFIRMED" if sum(g < 0.1 for g in gq) >= 2 and sum(g >= 0.5 for g in gp) >= 2 else
             "REFUTED" if sum(g >= 0.5 for g in gq) >= 2 else "INCONCLUSIVE")
    else:
        v = "INCOMPLETE"
    decision["stall"] = dict(verdict=v, gamma_plain=gp, gamma_presolve=gq,
                             rel_l2_plain=[st["plain"][s]["rel_l2"] for s in sorted(st["plain"])],
                             rel_l2_presolve=[st["presolve"][s]["rel_l2"] for s in sorted(st["presolve"])],
                             T_half_plain=[st["plain"][s]["T_half"] for s in sorted(st["plain"])],
                             T_half_presolve=[st["presolve"][s]["T_half"] for s in sorted(st["presolve"])])

# exploratory, NOT pre-registered (later choice, notes/VERIFICATION.md section 4): centred input only; enters no verdict
ex_runs = []
for f in glob.glob(os.path.join(RES, "runs_posthoc_lift1.jsonl")):
    ex_runs += [json.loads(l) for l in open(f) if l.strip()]
exploratory = {}
for problem in PROBS:
    for method in ("ritz", "pinn"):
        base = get("main20", problem, method, "plain", 20, 4000)
        cell = {r["seed"]: r for r in ex_runs if r["problem"] == problem and r["method"] == method}
        if cell:
            exploratory[f"{PID[problem]}/d20/{method}/lift1/4000"] = dict(
                status="EXPLORATORY, not pre-registered", rel_l2=stats([cell[s]["rel_l2"] for s in sorted(cell)]),
                cpu_total_s=stats([cell[s]["cpu_total_s"] for s in sorted(cell)]),
                T_half=[cell[s]["T_half"] for s in sorted(cell)], paired=paired(base, cell))
json.dump(exploratory, open(os.path.join(RES, "summary_exploratory.json"), "w"), indent=1)
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
json.dump(decision, open(os.path.join(RES, "decision.json"), "w"), indent=1)


# ------------------------------------------------------------------ LaTeX tables
def fe(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    e = f"{x:.2e}"; m, ex = e.split("e")
    return f"${m}\\times10^{{{int(ex)}}}$"


def fr(p):
    if not p or not p.get("n"):
        return "--"
    return f"{p['median']:.2f} [{p['min']:.2f}, {p['max']:.2f}] {p['wins']}/{p['n']}"


L = ["% generated by scripts/analyze.py from results/summary.json; do not edit",
     "% src: research_highdim_remedies/results/summary.json",
     "\\begin{tabular}{llllll}", "\\hline",
     "problem & method & arm & budget & rel.\\ $L^2$ mean (sd) & $\\rho$ median [min, max] wins \\\\", "\\hline"]
for problem in PROBS:
    for method in ("ritz", "pinn"):
        for arm in ("plain", "presolve", "lift3c", "lift3u", "rep3"):
            for budget in ("4000", "eqcpu"):
                r = summary.get(f"{PID[problem]}/d20/{method}/{arm}/{budget}")
                if not r:
                    continue
                tag = PID[problem] + (" (f.-m.)" if problem == "poisson" else "")
                bud = "4000 it" if budget == "4000" else f"eq.\\ CPU ({'/'.join(map(str, r['iters']))} it)"
                L.append(f"{tag} & {'Deep Ritz' if method == 'ritz' else 'PINN'} & {arm} & {bud} & "
                         f"{fe(r['rel_l2']['mean'])} ({fe(r['rel_l2']['sd'])}) & {fr(r.get('paired'))} \\\\")
    L.append("\\hline")
L.append("\\end{tabular}")
open(os.path.join(RES, "tables.tex"), "w").write("\n".join(L) + "\n")
print(json.dumps({k: v["verdict"] for k, v in decision["per_problem"].items()}, indent=1))
print({k: v for k, v in decision.items() if k.startswith("general") or k in ("centring_clause", "failure_flags", "stall")})
