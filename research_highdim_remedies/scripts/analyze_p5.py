"""Analysis of the addendum PREREG_C1_P5.md (sections 6.1-6.4). Reads results/runs_*.jsonl; writes
results/summary_p5.json, results/decision_p5.json, results/attribution.json, results/tables_p5.tex.
The frozen analysis of PREREG_C1.md (scripts/analyze.py, results/decision.json) is not touched."""
import os, json, glob, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
PID = {"laplace": "P1", "poisson": "P2", "ridge": "P3", "cospair": "P4", "altridge": "P5"}

runs = []
for f in sorted(glob.glob(os.path.join(RES, "runs_*.jsonl"))):
    runs += [json.loads(l) for l in open(f) if l.strip()]


def get(phases, problem, method, arm, d, iters=4000):
    out = {}
    for r in runs:
        if (r["phase"] in phases and r["problem"] == problem and r["method"] == method and r["arm"] == arm
                and r["d"] == d and r["schedule"] and (iters is None or r["iters"] == iters)):
            out[r["seed"]] = r
    return out


def stats(v):
    return dict(n=len(v), mean=statistics.mean(v), sd=statistics.stdev(v) if len(v) > 1 else float("nan"),
                min=min(v), max=max(v)) if v else dict(n=0)


def paired(base, other):
    seeds = sorted(set(base) & set(other))
    rho = [base[s]["rel_l2"] / other[s]["rel_l2"] for s in seeds]
    if not rho:
        return dict(n=0)
    return dict(n=len(rho), seeds=seeds, rho=rho, median=statistics.median(rho), min=min(rho), max=max(rho),
                wins=sum(x > 1 for x in rho))


# ---------------------------------------------------------------- P5 cells
summary = {}
P5 = ("p5main20", "p5eqcpu", "p5dim")
for d, phase in ((20, "p5main20"), (10, "p5dim"), (5, "p5dim")):
    for method in ("ritz", "pinn"):
        base = get((phase,), "altridge", method, "plain", d)
        if not base:
            continue
        for arm in ("plain", "presolve", "lift3c", "lift3u", "rep3", "lift1"):
            for budget in ("4000", "eqcpu"):
                if budget == "eqcpu" and (d != 20 or arm not in ("lift3c", "presolve")):
                    continue
                cell = get((phase,) if budget == "4000" else ("p5eqcpu",), "altridge", method, arm, d,
                           4000 if budget == "4000" else None)
                if not cell:
                    continue
                e = [cell[s]["rel_l2"] for s in sorted(cell)]
                rec = dict(d=d, method=method, arm=arm, budget=budget, iters=sorted({cell[s]["iters"] for s in cell}),
                           rel_l2=stats(e), per_seed={s: cell[s]["rel_l2"] for s in sorted(cell)},
                           cpu_total_s=stats([cell[s]["cpu_total_s"] for s in sorted(cell)]),
                           gamma_end=[cell[s]["gamma_end"] for s in sorted(cell)],
                           load1=[cell[s].get("load1") for s in sorted(cell)])
                if arm != "plain":
                    rec["paired"] = paired(base, cell)
                summary[f"P5/d{d}/{method}/{arm}/{budget}"] = rec
json.dump(summary, open(os.path.join(RES, "summary_p5.json"), "w"), indent=1)

# ---------------------------------------------------------------- decision (PREREG_C1_P5.md 6.1, 6.2)
eqplan = json.load(open(os.path.join(RES, "eqcpu_plan_p5.json"))) if os.path.exists(os.path.join(RES, "eqcpu_plan_p5.json")) else {}


def verdict(remedy):
    helps_all, hurts, detail = True, False, {}
    for method in ("ritz", "pinn"):
        m = summary.get(f"P5/d20/{method}/{remedy}/4000", {}).get("paired", {"n": 0})
        if remedy == "lift3c" or (eqplan.get(f"altridge/{method}/presolve", {}).get("iters") not in (0, None)):
            q = summary.get(f"P5/d20/{method}/{remedy}/eqcpu", {}).get("paired", {"n": 0})
        else:
            q = m   # presolve CPU condition holds: the 4000-iteration runs count as equal CPU
        detail[method] = dict(at4000=dict(median=m.get("median"), wins=m.get("wins"), n=m.get("n")),
                              eqcpu=dict(median=q.get("median"), wins=q.get("wins"), n=q.get("n")))
        for c in (m, q):
            if c.get("n", 0) < 5:
                return dict(verdict="INCOMPLETE", detail=detail)
            if not (c["wins"] >= 4 and c["median"] >= 1.5):
                helps_all = False
            if c["median"] <= 1 / 1.1:
                hurts = True
    v = "HURTS" if hurts else ("HELPS" if helps_all else "NEUTRAL/MIXED")
    return dict(verdict=v, detail=detail)


frozen = json.load(open(os.path.join(RES, "decision.json")))
dec = {"P5": {r: verdict(r) for r in ("presolve", "lift3c")}}


def frozen_v(remedy, p):
    return frozen["per_problem"][f"{remedy}/{p}"]["verdict"]


ext = {}
for remedy in ("presolve", "lift3c"):
    try:
        vs = {p: frozen_v(remedy, p) for p in ("P1", "P3", "P4")}
    except Exception as exc:   # decision.json layout differs: record and stop here
        vs = {"error": repr(exc)}
    vs["P5"] = dec["P5"][remedy]["verdict"]
    general = (vs.get("P1") == "HELPS" and any(vs.get(p) == "HELPS" for p in ("P3", "P4", "P5"))
               and not any(vs.get(p) == "HURTS" for p in ("P1", "P3", "P4", "P5")))
    ext[remedy] = dict(verdicts=vs, general_on_P1_P5=general)
dec["extended_verdict_P1_P5"] = ext

# ---------------------------------------------------------------- lift beyond centring (6.3), all problems, d = 20
att = {}
for problem in ("laplace", "poisson", "ridge", "cospair", "altridge"):
    ph_l3 = ("p5main20",) if problem == "altridge" else ("main20",)
    ph_l1 = ("p5main20",) if problem == "altridge" else ("posthoc_lift1",)
    row = {}
    for method in ("ritz", "pinn"):
        l3 = get(ph_l3, problem, method, "lift3c", 20)
        l1 = get(ph_l1, problem, method, "lift1", 20)
        pl = get(ph_l3, problem, method, "plain", 20)
        l3eq = get(("p5eqcpu",) if problem == "altridge" else ("eqcpu",), problem, method, "lift3c", 20, None)
        r = dict(lift1_vs_plain=paired(pl, l1), lift3c_vs_plain=paired(pl, l3),
                 lift3c_vs_lift1=paired(l1, l3), lift3c_eqcpu_vs_lift1=paired(l1, l3eq),
                 cpu_lift1_over_plain=(statistics.median([l1[s]["cpu_train_s"] for s in l1]) /
                                       statistics.median([pl[s]["cpu_train_s"] for s in pl])) if l1 and pl else None)
        if method == "ritz":
            rp = get(ph_l3, problem, method, "rep3", 20)
            r["rep3_vs_lift1"] = paired(l1, rp)
            r["lift3c_vs_rep3"] = paired(rp, l3)
        row[method] = r
    c = [row[m]["lift3c_vs_lift1"] for m in ("ritz", "pinn")]
    if any(x.get("n", 0) < 5 for x in c):
        lab = "INCOMPLETE"
    elif any(x["median"] <= 1 / 1.1 for x in c):
        lab = "worse than centring"
    elif all(x["wins"] >= 4 and x["median"] >= 1.5 for x in c):
        lab = "adds to centring"
    else:
        lab = "no clear difference"
    row["label"] = lab
    att[PID[problem]] = row
json.dump(att, open(os.path.join(RES, "attribution.json"), "w"), indent=1)
dec["lift_beyond_centring"] = {p: att[p]["label"] for p in att}
json.dump(dec, open(os.path.join(RES, "decision_p5.json"), "w"), indent=1)


# ---------------------------------------------------------------- LaTeX table
def fmt(c):
    if not c or c.get("n", 0) == 0:
        return "--"
    return f"{c['median']:.2f} [{c['min']:.2f}, {c['max']:.2f}] {c['wins']}/{c['n']}"


lines = [r"\begin{tabular}{llll}", r"\hline",
         r"problem & method & $\mathrm{err(lift1)}/\mathrm{err(lift3c)}$ & $\mathrm{err(plain)}/\mathrm{err(lift1)}$ \\", r"\hline"]
for p in ("P1", "P2", "P3", "P4", "P5"):
    for m, mn in (("ritz", "Deep Ritz"), ("pinn", "PINN")):
        lines.append(f"{p} & {mn} & {fmt(att[p][m]['lift3c_vs_lift1'])} & {fmt(att[p][m]['lift1_vs_plain'])} \\\\")
lines += [r"\hline", r"\end{tabular}"]
p5 = [r"\begin{tabular}{lllll}", r"\hline", r"method & arm & budget & test rel.\ $L^2$, mean (s.d.) & $\rho$ median [min, max] wins \\", r"\hline"]
for k, v in summary.items():
    if v["d"] != 20:
        continue
    st = v["rel_l2"]
    p5.append(f"{'Deep Ritz' if v['method'] == 'ritz' else 'PINN'} & {v['arm']} & {v['budget'] if v['budget'] == '4000' else 'eq.\\ CPU (' + str(v['iters'][0]) + ')'} & "
              f"{st['mean']:.4f} ({st['sd']:.4f}) & {fmt(v.get('paired'))} \\\\")
p5 += [r"\hline", r"\end{tabular}"]
open(os.path.join(RES, "tables_p5.tex"), "w").write("% lift1 baseline, d = 20, seeds 10-14 (results/attribution.json)\n"
                                                     + "\n".join(lines) + "\n\n% P5, d = 20 (results/summary_p5.json)\n"
                                                     + "\n".join(p5) + "\n")
print(json.dumps(dec, indent=1))
for k, v in summary.items():
    print(k, f"{v['rel_l2']['mean']:.4e}", fmt(v.get("paired")))
for p in att:
    print(p, att[p]["label"], {m: fmt(att[p][m]["lift3c_vs_lift1"]) for m in ("ritz", "pinn")})
