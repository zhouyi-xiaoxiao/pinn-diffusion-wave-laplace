"""Markdown tables of the study from the saved JSON results (no training). Output: results/tables.md"""
import os, json, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
PROBLEMS = ["heat1d", "wave1d", "wave2d", "laplace2d", "laplace3d"]
SAMPLERS = ["grid", "random", "resample", "sobol", "rad"]
L = []
e = lambda v: f"{v:.2e}"

S = R("benchmark_summary.json")
summ = {(r["problem"], r["arm"], r["sampler"]): r for r in S["summary"]}
L.append("### T1. Held-out errors, mean ± std over seeds (geometric mean in brackets)\n")
L.append("| problem | N_r | seeds | strategy | Adam: rel-L2 | Adam: L∞ | Adam→L-BFGS: rel-L2 | Adam→L-BFGS: L∞ |")
L.append("|---|---|---|---|---|---|---|---|")
for p in PROBLEMS:
    for s in SAMPLERS:
        a, b = summ.get((p, "adam", s)), summ.get((p, "adam_lbfgs", s))
        if a is None:
            continue
        L.append(f"| {p} | {S['config'][p]['n_r']} | {a['rel_l2_n']} | {s} | {e(a['rel_l2_mean'])} ± {e(a['rel_l2_std'])} [{e(a['rel_l2_geo_mean'])}] | "
                 f"{e(a['linf_mean'])} ± {e(a['linf_std'])} | {e(b['rel_l2_mean'])} ± {e(b['rel_l2_std'])} [{e(b['rel_l2_geo_mean'])}] | {e(b['linf_mean'])} ± {e(b['linf_std'])} |")

st = R("stats.json")
L.append("\n### T2. Adam vs Adam→L-BFGS at matched iteration count (paired over all strategy × seed pairs)\n")
L.append("| problem | pairs | geo-mean rel-L2 Adam | geo-mean rel-L2 Adam→L-BFGS | ratio | L-BFGS better in | Wilcoxon p |")
L.append("|---|---|---|---|---|---|---|")
for p in PROBLEMS:
    if p in st:
        o = st[p]["adam_vs_adam_lbfgs"]
        L.append(f"| {p} | {o['n_pairs']} | {e(o['adam_geo_mean_rel_l2'])} | {e(o['adam_lbfgs_geo_mean_rel_l2'])} | {o['geo_mean_ratio_lbfgs_over_adam']:.3f} | {o['n_pairs_lbfgs_better']}/{o['n_pairs']} | {o['wilcoxon_p_two_sided']:.1e} |")

L.append("\n### T3. Collocation strategy vs fixed i.i.d. random at matched N_r (paired by seed; ratio < 1 = better than random)\n")
L.append("| problem | arm | grid | resample | sobol | rad | Kruskal–Wallis p (5 strategies) |")
L.append("|---|---|---|---|---|---|---|")
for p in PROBLEMS:
    if p not in st:
        continue
    for arm in ("adam", "adam_lbfgs"):
        cells = []
        for s in ("grid", "resample", "sobol", "rad"):
            o = st[p]["sampler_vs_random"][f"{arm}:{s}"]
            pv = "n/a" if o["wilcoxon_p_two_sided"] is None else f"{o['wilcoxon_p_two_sided']:.3f}"
            cells.append(f"{o['geo_mean_ratio_to_random']:.2f}× ({o['n_seeds_better']}/{o['n']}, p={pv})")
        L.append(f"| {p} | {arm} | " + " | ".join(cells) + f" | {st[p]['kruskal_across_samplers'][arm]:.3g} |")

cl = R("classical.json")
L.append("\n### T4. Second-order finite-difference baselines on the same held-out grids\n")
L.append("| problem | n | stored grid values | rel-L2 | L∞ | observed order | time (s, best of 3) |")
L.append("|---|---|---|---|---|---|---|")
for r in cl:
    od = "" if r["observed_order_rel_l2"] is None else f"{r['observed_order_rel_l2']:.2f}"
    L.append(f"| {r['problem']} | {r['n']} | {r['grid_values']} | {e(r['rel_l2'])} | {e(r['linf'])} | {od} | {r['time_s_best_of_3']:.2e} |")

if os.path.exists(os.path.join(ROOT, "results", "timing_single.json")):
    tm = R("timing_single.json")
    L.append("\n### T5. Cost: one PINN run (single-model reference driver, CPU 4 threads) vs the cheapest FD solve that is at least as accurate\n")
    L.append("| problem | params | ms / Adam step | ms / L-BFGS iter | est. run time Adam (s) | est. run time Adam→L-BFGS (s) | best PINN geo-mean rel-L2 (Adam→L-BFGS) | cheapest FD with rel-L2 ≤ that | FD time (s) | time ratio |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in PROBLEMS:
        if p not in tm:
            continue
        best = min(summ[(p, "adam_lbfgs", s)]["rel_l2_geo_mean"] for s in SAMPLERS if (p, "adam_lbfgs", s) in summ)
        cand = [r for r in cl if r["problem"] == p and r["rel_l2"] <= best]
        t = tm[p]
        if cand:
            c = cand[0]
            L.append(f"| {p} | {t['n_params']} | {t['ms_per_adam_step']:.0f} | {t['ms_per_lbfgs_iter']:.0f} | {t['est_run_s_adam']:.0f} | {t['est_run_s_adam_lbfgs']:.0f} | {e(best)} | n={c['n']} ({e(c['rel_l2'])}) | {c['time_s_best_of_3']:.1e} | {t['est_run_s_adam_lbfgs'] / c['time_s_best_of_3']:.1e} |")
        else:
            L.append(f"| {p} | {t['n_params']} | {t['ms_per_adam_step']:.0f} | {t['ms_per_lbfgs_iter']:.0f} | {t['est_run_s_adam']:.0f} | {t['est_run_s_adam_lbfgs']:.0f} | {e(best)} | none in table | | |")
    L.append(f"\nLoad average during the timing run: start {tm['loadavg_start']}, end {tm['loadavg_end']} (12 cores).")

if os.path.exists(os.path.join(ROOT, "results", "budget_sweep.json")):
    bs = R("budget_sweep.json")
    L.append("\n### T6. Collocation-budget sweep (Adam→L-BFGS, rel-L2 mean ± std over 5 seeds)\n")
    L.append("| problem | N_r | grid | random | sobol |")
    L.append("|---|---|---|---|---|")
    for p in sorted({r["problem"] for r in bs}):
        for n in sorted({r["n_r"] for r in bs if r["problem"] == p}):
            cells = []
            for s in ("grid", "random", "sobol"):
                v = np.array([r["adam_lbfgs"]["rel_l2"] for r in bs if r["problem"] == p and r["n_r"] == n and r["sampler"] == s])
                cells.append(f"{e(v.mean())} ± {e(v.std(ddof=1))}" if len(v) else "")
            L.append(f"| {p} | {n} | " + " | ".join(cells) + " |")

open(os.path.join(ROOT, "results", "tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
