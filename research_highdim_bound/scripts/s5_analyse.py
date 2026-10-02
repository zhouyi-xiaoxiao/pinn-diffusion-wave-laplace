"""Decisions H1-H4 of PREREG_T2.md from results/eval_confirm.jsonl, exactly as pre-registered.
Outputs: results/eval_summary.csv (one row per network and set), results/decisions.json,
results/table_networks.tex (fresh set, all 88 networks), results/share_by_d.csv."""
import csv, json, math, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from t2common import RES, load_jsonl, dump

rows = load_jsonl(os.path.join(RES, "eval_confirm.jsonl"))
assert len(rows) == 176, len(rows)
FIELDS = ["file", "family", "problem", "method", "d", "seed", "w", "iters", "confirmatory", "set", "err_abs", "se_err",
          "rel_l2", "B_int", "se_B_int", "B_bd", "se_B_bd", "bound", "se_bound", "D", "se_D", "z_D", "eta", "se_eta",
          "share_bd", "se_share", "sampled_max_err", "sampled_max_res", "sampled_max_bd", "sampled_maxprinciple_bound",
          "lap_check_rel"]
with open(os.path.join(RES, "eval_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n"); w.writeheader()
    for r in sorted(rows, key=lambda r: (r["set"], r["family"], r["problem"], r["method"], r["d"], r["w"], r["seed"])):
        w.writerow(r)

dec = {}
# H1
viol = [dict(file=r["file"], set=r["set"], z_D=r["z_D"]) for r in rows if r["D"] < -3 * r["se_D"]]
dec["H1"] = dict(rule="bound >= err - 3 se(D) for all 88 networks on both sets", violations=viol,
                 min_z_D={s: min(r["z_D"] for r in rows if r["set"] == s) for s in ("fresh", "hd")},
                 min_eta={s: min(r["eta"] for r in rows if r["set"] == s) for s in ("fresh", "hd")},
                 decision="holds" if not viol else "FAILS")
# H2
fresh = [r for r in rows if r["set"] == "fresh"]
h2 = [r for r in fresh if r["confirmatory"] and r["d"] >= 5]
fail2 = [dict(file=r["file"], eta=r["eta"], se_eta=r["se_eta"]) for r in h2 if r["eta"] > 3]
dec["H2"] = dict(rule="eta <= 3 (fresh, point estimate) for every confirmatory network with d >= 5",
                 n_tested=len(h2), failures=sorted(fail2, key=lambda x: -x["eta"]),
                 eta_range=[min(r["eta"] for r in h2), max(r["eta"] for r in h2)],
                 eta_range_main_d_ge5=[min(r["eta"] for r in h2 if r["family"] == "main"), max(r["eta"] for r in h2 if r["family"] == "main")],
                 decision="holds" if not fail2 else "FAILS")
# H3
DS = [2, 3, 5, 10, 20]
share = {}
out_rows = []
h3 = {}
for prob in ("laplace", "poisson"):
    for meth in ("pinn", "ritz"):
        for variant in ("primary_all_seeds", "sensitivity_confirmatory_only"):
            means = []
            for d in DS:
                sel = [r for r in fresh if r["family"] == "main" and r["problem"] == prob and r["method"] == meth and r["d"] == d
                       and (variant.startswith("primary") or r["confirmatory"])]
                m = sum(r["share_bd"] for r in sel) / len(sel)
                means.append(m)
                out_rows.append(dict(problem=prob, method=meth, variant=variant, d=d, n_seeds=len(sel), mean_share=m,
                                     min_share=min(r["share_bd"] for r in sel), max_share=max(r["share_bd"] for r in sel),
                                     mean_eta=sum(r["eta"] for r in sel) / len(sel), min_eta=min(r["eta"] for r in sel),
                                     max_eta=max(r["eta"] for r in sel),
                                     mean_rel_err=sum(r["rel_l2"] for r in sel) / len(sel)))
            nondec = all(means[i + 1] >= means[i] for i in range(len(means) - 1))
            h3[f"{prob}/{meth}/{variant}"] = dict(means=means, non_decreasing=nondec)
dec["H3"] = dict(rule="seed-mean share non-decreasing in d over 2,3,5,10,20 for each (problem, method); primary all seeds",
                 pairs=h3, decision="holds" if all(v["non_decreasing"] for k, v in h3.items() if "primary" in k) else "FAILS")
with open(os.path.join(RES, "share_by_d.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(out_rows)
# H4
h4 = [dict(problem=r["problem"], method=r["method"], w=r["w"], eta=r["eta"], se_eta=r["se_eta"], share=r["share_bd"],
           rel_l2=r["rel_l2"]) for r in fresh if r["family"] == "sweep"]
dec["H4"] = dict(rule="descriptive", rows=sorted(h4, key=lambda x: (x["problem"], x["method"], x["w"])))
# descriptive: long and equal-CPU vs main at d = 10
desc = []
for r in fresh:
    if r["family"] in ("long", "equaltime"):
        base = [b for b in fresh if b["family"] == "main" and b["problem"] == r["problem"] and b["method"] == r["method"]
                and b["d"] == r["d"] and b["seed"] == r["seed"]][0]
        pinn = [b for b in fresh if b["family"] == "main" and b["problem"] == r["problem"] and b["method"] == "pinn"
                and b["d"] == r["d"] and b["seed"] == r["seed"]][0]
        desc.append(dict(file=r["file"], eta=r["eta"], share=r["share_bd"], rel_l2=r["rel_l2"], bound=r["bound"],
                         main4000_eta=base["eta"], main4000_share=base["share_bd"], main4000_rel_l2=base["rel_l2"],
                         main4000_bound=base["bound"], pinn4000_rel_l2=pinn["rel_l2"], pinn4000_bound=pinn["bound"]))
dec["budget_descriptive"] = desc
# max-principle comparison
mp = [r["sampled_maxprinciple_bound"] / r["bound"] for r in fresh]
dec["maxprinciple_over_L2bound"] = dict(min=min(mp), max=max(mp))
# hd vs fresh agreement
pairs = defaultdict(dict)
for r in rows:
    pairs[r["file"]][r["set"]] = r
dec["fresh_vs_hd_eta_max_absdiff"] = max(abs(p["fresh"]["eta"] - p["hd"]["eta"]) for p in pairs.values())
dec["lap_check_max"] = max(r["lap_check_rel"] for r in rows)
dump("decisions.json", dec)
for k in ("H1", "H2", "H3"):
    print(k, dec[k]["decision"])
print(json.dumps({k: dec[k] for k in ("H1",)}, indent=0)[:600])
print("H2", dec["H2"]["n_tested"], dec["H2"]["eta_range"], dec["H2"]["eta_range_main_d_ge5"], dec["H2"]["failures"][:20])
for k, v in h3.items():
    print(k, [round(x, 3) for x in v["means"]], v["non_decreasing"])
for r in dec["H4"]["rows"]:
    print(r)
for r in desc:
    print(r)
print(dec["maxprinciple_over_L2bound"], dec["fresh_vs_hd_eta_max_absdiff"], dec["lap_check_max"])

# LaTeX table (fresh set): all 88 networks
def f3(x):
    return f"{x:.3g}"
lines = []
for r in sorted(fresh, key=lambda r: ({"main": 0, "long": 1, "equaltime": 2, "sweep": 3}[r["family"]], r["problem"], r["method"], r["d"], r["w"], r["seed"])):
    mark = "" if r["confirmatory"] else "$^\\dagger$"
    lines.append(f"{r['family']} & {'P1' if r['problem']=='laplace' else 'P2'} & {'PINN' if r['method']=='pinn' else 'DR'} & {r['d']} & {r['w']} & {r['seed']}{mark} & "
                 f"{r['err_abs']:.3e} ({r['se_err']:.1e}) & {r['B_int']:.3e} ({r['se_B_int']:.1e}) & {r['B_bd']:.3e} ({r['se_B_bd']:.1e}) & "
                 f"{r['bound']:.3e} & {r['eta']:.2f} ({r['se_eta']:.2f}) & {r['share_bd']:.3f} ({r['se_share']:.3f}) \\\\")
with open(os.path.join(RES, "table_networks.tex"), "w") as f:
    f.write("% generated by scripts/s5_analyse.py from results/eval_confirm.jsonl (set 'fresh'); dagger: non-confirmatory (identical to a pilot-E2 network)\n")
    f.write("\n".join(lines) + "\n")
