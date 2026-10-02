"""Aggregate the benchmark runs into CSV/JSON tables with mean +- std and paired statistics.

Outputs (results/):
  benchmark_runs.csv       one row per (problem, sampler, optimiser arm, seed)
  benchmark_summary.csv    mean, std (ddof=1), median, min, max, geometric mean and geometric
                           std of rel-L2 and L_inf per (problem, sampler, arm)
  benchmark_summary.json   same, machine readable
  stats.json               paired tests (same seed = same initial weights):
                             sampler vs i.i.d.-random baseline (Wilcoxon signed-rank on log error),
                             Adam vs Adam->L-BFGS, Kruskal-Wallis across samplers
"""
import sys, os, json, glob, csv
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBLEMS = ["heat1d", "wave1d", "wave2d", "laplace2d", "laplace3d"]
SAMPLERS = ["grid", "random", "resample", "sobol", "rad"]
ARMS = ["adam", "adam_lbfgs"]


def load_runs():
    rows, curves, cfgs, timings = [], {}, {}, {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*_s*-*.json"))):
        d = json.load(open(f))
        prob = d["runs"][0]["problem"]
        cfgs[prob] = d["config"]
        timings.setdefault(prob, []).append(d["timing"])
        for r in d["runs"]:
            for arm in ARMS:
                a = r[arm]
                rows.append({"problem": r["problem"], "sampler": r["sampler"], "arm": arm, "seed": r["seed"],
                             "rel_l2": a["rel_l2"], "linf": a["linf"], "rel_linf": a["rel_linf"],
                             "final_loss": a["final_loss"]})
                curves[(r["problem"], r["sampler"], arm, r["seed"])] = a["curve"]
    return rows, curves, cfgs, timings


def summarise(v):
    v = np.asarray(v, float)
    lg = np.log10(v)
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)), "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max()), "geo_mean": float(10 ** lg.mean()),
            "geo_std_factor": float(10 ** lg.std(ddof=1))}


def main():
    rows, curves, cfgs, timings = load_runs()
    out = os.path.join(ROOT, "results")
    with open(os.path.join(out, "benchmark_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)

    def sel(p, s, a, key="rel_l2"):
        r = sorted([r for r in rows if r["problem"] == p and r["sampler"] == s and r["arm"] == a], key=lambda r: r["seed"])
        return np.array([x[key] for x in r]), [x["seed"] for x in r]

    summary = []
    for p in PROBLEMS:
        for a in ARMS:
            for s in SAMPLERS:
                v, seeds = sel(p, s, a)
                if len(v) == 0:
                    continue
                e = {"problem": p, "arm": a, "sampler": s}
                for key in ("rel_l2", "linf"):
                    sm = summarise(sel(p, s, a, key)[0])
                    e.update({f"{key}_{k}": val for k, val in sm.items()})
                summary.append(e)
    with open(os.path.join(out, "benchmark_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(summary)
    json.dump({"config": cfgs, "timing": timings, "summary": summary}, open(os.path.join(out, "benchmark_summary.json"), "w"), indent=1)

    st = {}
    for p in PROBLEMS:
        if not any(r["problem"] == p for r in rows):
            continue
        st[p] = {"sampler_vs_random": {}, "adam_vs_adam_lbfgs": {}, "kruskal_across_samplers": {}}
        for a in ARMS:
            base, seeds = sel(p, "random", a)
            groups = []
            for s in SAMPLERS:
                v, sd = sel(p, s, a)
                groups.append(np.log10(v))
                if s == "random":
                    continue
                assert sd == seeds
                d = np.log10(v) - np.log10(base)
                try:
                    pval = float(stats.wilcoxon(d).pvalue)
                except ValueError:
                    pval = None
                st[p]["sampler_vs_random"][f"{a}:{s}"] = {
                    "geo_mean_ratio_to_random": float(10 ** d.mean()), "n_seeds_better": int((d < 0).sum()),
                    "n": int(len(d)), "wilcoxon_p_two_sided": pval}
            st[p]["kruskal_across_samplers"][a] = float(stats.kruskal(*groups).pvalue)
        # optimiser: paired over all (sampler, seed)
        va = np.concatenate([sel(p, s, "adam")[0] for s in SAMPLERS])
        vb = np.concatenate([sel(p, s, "adam_lbfgs")[0] for s in SAMPLERS])
        d = np.log10(vb) - np.log10(va)
        st[p]["adam_vs_adam_lbfgs"] = {"geo_mean_ratio_lbfgs_over_adam": float(10 ** d.mean()),
                                       "n_pairs_lbfgs_better": int((d < 0).sum()), "n_pairs": int(len(d)),
                                       "wilcoxon_p_two_sided": float(stats.wilcoxon(d).pvalue),
                                       "adam_geo_mean_rel_l2": float(10 ** np.log10(va).mean()),
                                       "adam_lbfgs_geo_mean_rel_l2": float(10 ** np.log10(vb).mean())}
    json.dump(st, open(os.path.join(out, "stats.json"), "w"), indent=1)

    # console table
    for p in PROBLEMS:
        for a in ARMS:
            for e in summary:
                if e["problem"] == p and e["arm"] == a:
                    print(f"{p:10s} {a:10s} {e['sampler']:9s} n={e['rel_l2_n']:2d} relL2 {e['rel_l2_mean']:.2e} +- {e['rel_l2_std']:.2e}"
                          f" (median {e['rel_l2_median']:.2e}, geo {e['rel_l2_geo_mean']:.2e} x/ {e['rel_l2_geo_std_factor']:.2f})"
                          f"  Linf {e['linf_mean']:.2e} +- {e['linf_std']:.2e}")
        if p in st:
            print("   ", p, "adam vs adam_lbfgs:", st[p]["adam_vs_adam_lbfgs"])
            for k, v in st[p]["sampler_vs_random"].items():
                print("      vs random", k, v)
            print("      kruskal:", st[p]["kruskal_across_samplers"])


if __name__ == "__main__":
    main()
