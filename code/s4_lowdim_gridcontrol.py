"""Control experiment of Section 4.3: what causes the penalty of the cell-centred grid on the heat problem?

The benchmark found the cell-centred 16 x 16 grid about four times worse than random points on
the heat problem after Adam -> L-BFGS.  The explanation offered is that this grid has no residual
point in the strip 0 < t < 1/32 next to the initial line.  This experiment tests that explanation
with the benchmark's own driver (stacked networks, same protocol, same seeds 0..9), by changing
the point set in one respect at a time:

  random      256 i.i.d. uniform points (reference; as in the benchmark)
  grid        cell-centred 16 x 16 grid (as in the benchmark): t, x in {(j + 1/2)/16}
  gridn       node-centred closed 16 x 16 grid: t, x in {j/15}; has points on t = 0, t = 1, x = 0, x = 1
  gridt0      the cell-centred grid shifted by half a cell in t ONLY: t in {j/16}, x in {(j + 1/2)/16};
              first column on the initial line, no point on x = 0 or x = 1
  gridplus0   the cell-centred grid PLUS one column of 16 points at t = 1/64 (inside the strip);
              272 points
  gridplusm   placebo: the cell-centred grid PLUS one column of 16 points at t = 1/2 (between two
              columns of the grid, far from the initial line); 272 points

Usage (from the repository root or from article/):
    python code/s4_lowdim_gridcontrol.py run A [device]     # stack A: random, grid, gridn, gridt0 (40 networks, N_r = 256)
    python code/s4_lowdim_gridcontrol.py run B [device]     # stack B: gridplus0, gridplusm       (20 networks, N_r = 272)
    python code/s4_lowdim_gridcontrol.py merge              # statistics -> data/s4_lowdim_gridcontrol.json, table rows

Outputs: data/s4_lowdim_gridcontrol_parts/stack_{A,B}.json (raw per-run records with training curves
and the squared error per time level of the evaluation grid), data/s4_lowdim_gridcontrol.json,
data/s4_lowdim_gridcontrol_table.tex (the rows are also written into sections/s4_lowdim.tex between the
lines "%% BEGIN GENERATED s4_lowdim:tab:gridcontrol" and "%% END GENERATED ...", if that file has them).
No timing claim is derived from these runs.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
ROOT = ARTICLE if os.path.isdir(os.path.join(ARTICLE, "research_benchmark")) else os.path.dirname(ARTICLE)  # repository root
BENCH = os.path.join(ROOT, "research_benchmark")
DATA = os.path.join(ARTICLE, "data")
PARTS = os.path.join(DATA, "s4_lowdim_gridcontrol_parts")
sys.path.insert(0, BENCH)

PROBLEM, N_ADAM, BRANCH_AT, M = "heat1d", 3000, 1500, 16
SEEDS = list(range(10))
STACKS = {"A": (["random", "grid", "gridn", "gridt0"], M * M),
          "B": (["gridplus0", "gridplusm"], M * M + M)}
LABEL = {"random": r"\strat{random}", "grid": r"\strat{grid-c}", "gridn": r"\strat{grid-n}",
         "gridt0": r"\strat{grid-c} shifted to $t=0$", "gridplus0": r"\strat{grid-c} + column, $t=1/64$",
         "gridplusm": r"\strat{grid-c} + column, $t=1/2$"}


def register_samplers():
    """Add the control point sets to the sampler table of the benchmark package (nothing in the
    package is modified on disk)."""
    from pinnbench import samplers as S

    cell = (np.arange(M) + 0.5) / M

    def mesh(t, x):
        T, X = np.meshgrid(t, x, indexing="ij")
        return np.stack([T.ravel(), X.ravel()], 1)

    class NodeGrid(S.Sampler):
        def initial(self):
            a = np.arange(M) / (M - 1)
            return self._to_tensor(mesh(a, a))

    class GridT0(S.Sampler):
        def initial(self):
            return self._to_tensor(mesh(np.arange(M) / M, cell))

    class GridPlus(S.Sampler):
        t_extra = None

        def initial(self):
            pts = np.concatenate([mesh(cell, cell), mesh(np.array([self.t_extra]), cell)])
            assert pts.shape[0] == self.n
            return self._to_tensor(pts)

    class GridPlus0(GridPlus):
        t_extra = 1.0 / 64

    class GridPlusM(GridPlus):
        t_extra = 0.5

    S.SAMPLERS.update({"gridn": NodeGrid, "gridt0": GridT0, "gridplus0": GridPlus0, "gridplusm": GridPlusM})


def run(stack, device):
    import torch
    torch.set_num_threads(4)
    from pinnbench.core import run_batched
    from pinnbench.problems import make_problem

    register_samplers()
    names, n_r = STACKS[stack]
    out_path = os.path.join(PARTS, f"stack_{stack}.json")
    if os.path.exists(out_path):
        print("skip (exists)", out_path)
        return
    os.makedirs(PARTS, exist_ok=True)
    specs = [(s, seed) for s in names for seed in SEEDS]
    t0 = time.time()
    print(f"=== stack {stack}: {len(specs)} stacked networks, N_r = {n_r}, device = {device}", flush=True)
    out, timing, preds = run_batched(PROBLEM, specs, n_r, N_ADAM, BRANCH_AT, device=device, log_every=100,
                                     verbose=True, return_preds=True)
    timing["wall_total_s_not_a_timing_claim"] = time.time() - t0
    # squared error per time level of the 201 x 201 evaluation grid, both arms (float64)
    P = make_problem(PROBLEM)
    X = P.eval_points()
    u = P.exact(X)
    n = int(round(np.sqrt(X.shape[0])))
    t_levels = X[:, 0].reshape(n, n)[:, 0]
    assert np.allclose(X[:, 0].reshape(n, n), t_levels[:, None])
    for i, o in enumerate(out):
        for arm in ("adam", "adam_lbfgs"):
            e = (preds[arm][i].astype(np.float64) - u).reshape(n, n)
            o[arm]["sq_error_per_time_level"] = (e ** 2).sum(1).tolist()
            k = np.unravel_index(np.abs(e).argmax(), e.shape)
            o[arm]["argmax_t"] = float(t_levels[k[0]])
            o[arm]["argmax_x"] = float(X[:, 1].reshape(n, n)[k])
    json.dump({"stack": stack, "problem": PROBLEM, "n_r": n_r, "n_adam": N_ADAM, "branch_at": BRANCH_AT,
               "seeds": SEEDS, "samplers": names, "timing": timing, "t_levels": t_levels.tolist(), "runs": out},
              open(out_path, "w"))
    for s in names:
        e = np.array([o["adam_lbfgs"]["rel_l2"] for o in out if o["sampler"] == s])
        print(f"   {s:10s} adam_lbfgs rel_l2 mean {e.mean():.3e} sd {e.std(ddof=1):.3e} geo {np.exp(np.log(e).mean()):.3e}", flush=True)
    print(f"=== stack {stack} done in {time.time() - t0:.1f} s", flush=True)


def merge():
    from scipy import stats

    register_samplers()
    runs, t_levels, meta = [], None, {}
    for stack in STACKS:
        d = json.load(open(os.path.join(PARTS, f"stack_{stack}.json")))
        runs += d["runs"]
        t_levels = np.array(d["t_levels"])
        meta[stack] = {"samplers": d["samplers"], "n_r": d["n_r"], "batch_size": d["timing"]["batch_size"],
                       "device": d["timing"]["device"], "lbfgs_batched_fevals": d["timing"]["lbfgs_batched_fevals"]}

    def val(s, arm="adam_lbfgs", key="rel_l2"):
        rr = sorted((r for r in runs if r["sampler"] == s), key=lambda r: r["seed"])
        assert [r["seed"] for r in rr] == SEEDS
        return np.array([r[arm][key] for r in rr])

    def geo(v):
        return float(np.exp(np.log(v).mean()))

    def paired(num, den):
        d = np.log(num) - np.log(den)
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        tq = stats.t.ppf(0.975, n - 1)
        return {"ratio": float(np.exp(d.mean())), "n_better": int((d < 0).sum()), "n": n,
                "p": float(stats.wilcoxon(d).pvalue), "per_seed_ratio": [float(x) for x in np.exp(d)],
                "ci95_t_on_log": [float(np.exp(d.mean() - tq * se)), float(np.exp(d.mean() + tq * se))]}

    names = [s for st in STACKS.values() for s in st[0]]
    N = {"note": "Control experiment of Section 4.3. Heat problem, benchmark driver and protocol (1500 Adam + 1500 L-BFGS iterations, "
                 "stacked networks), seeds 0-9. Errors on the held-out 201 x 201 grid.",
         "stacks": meta, "strategies": {}, "vs_random": {}, "vs_grid_c": {}}
    early = t_levels <= 0.05 + 1e-12
    strip = t_levels <= 1.0 / 32 + 1e-12
    for s in names:
        N["strategies"][s] = {}
        for arm in ("adam", "adam_lbfgs"):
            v = val(s, arm)
            prof = np.array(val(s, arm, "sq_error_per_time_level").tolist())            # (seeds, 201)
            sh05 = prof[:, early].sum(1) / prof.sum(1)
            sh32 = prof[:, strip].sum(1) / prof.sum(1)
            N["strategies"][s][arm] = {
                "n_points": STACKS["A" if s in STACKS["A"][0] else "B"][1],
                "mean": float(v.mean()), "std": float(v.std(ddof=1)), "geo_mean": geo(v),
                "min": float(v.min()), "max": float(v.max()), "per_seed": v.tolist(),
                "share_sq_error_t_le_0.05": {"per_seed": sh05.tolist(), "median": float(np.median(sh05)),
                                             "min": float(sh05.min()), "max": float(sh05.max())},
                "share_sq_error_t_le_1_32": {"per_seed": sh32.tolist(), "median": float(np.median(sh32)),
                                             "min": float(sh32.min()), "max": float(sh32.max())},
                "argmax_t_per_seed": val(s, arm, "argmax_t").tolist()}
    for arm in ("adam", "adam_lbfgs"):
        N["vs_random"][arm] = {s: paired(val(s, arm), val("random", arm)) for s in names if s != "random"}
        N["vs_grid_c"][arm] = {s: paired(val(s, arm), val("grid", arm)) for s in names if s not in ("random", "grid")}
    # the contrast of the design: the same 272 points except for the position of the extra column
    N["column_in_strip_vs_placebo"] = {arm: paired(val("gridplus0", arm), val("gridplusm", arm)) for arm in ("adam", "adam_lbfgs")}
    # agreement of the two re-run arms with the benchmark runs of the same seeds
    import csv
    rows = list(csv.DictReader(open(os.path.join(BENCH, "results", "benchmark_runs.csv"))))
    N["agreement_with_benchmark"] = {}
    for s in ("random", "grid"):
        b = np.array([float(r["rel_l2"]) for r in sorted((r for r in rows if r["problem"] == PROBLEM and r["sampler"] == s
                                                          and r["arm"] == "adam_lbfgs"), key=lambda r: int(r["seed"]))])
        q = val(s) / b
        N["agreement_with_benchmark"][s] = {"geo_ratio_control_over_benchmark": geo(q), "min": float(q.min()), "max": float(q.max()),
                                            "benchmark_geo_mean": geo(b), "control_geo_mean": geo(val(s))}
    json.dump(N, open(os.path.join(DATA, "s4_lowdim_gridcontrol.json"), "w"), indent=1)

    def sci(x):
        m, e = f"{x:.2e}".split("e")
        return f"$\\sci{{{m}}}{{{int(e)}}}$"

    def cell(q, with_p=True):
        return f"{q['ratio']:.2f} ({q['n_better']}/{q['n']}" + (f"; {q['p']:.3f})" if with_p else ")")

    tex = ["% rows of Table s4_lowdim:tab:gridcontrol, generated by code/s4_lowdim_gridcontrol.py merge"]
    for s in names:
        a = N["strategies"][s]["adam_lbfgs"]
        vr = "---" if s == "random" else cell(N["vs_random"]["adam_lbfgs"][s])
        vg = "---" if s in ("random", "grid") else cell(N["vs_grid_c"]["adam_lbfgs"][s], with_p=False)
        sh = a["share_sq_error_t_le_0.05"]
        m, e = f"{a['mean']:.2e}".split("e")
        sd = a["std"] / 10 ** int(e)
        tex.append(f"{LABEL[s]} & {a['n_points']} & {sci(a['geo_mean'])} & $({m}\\pm{sd:.2f})\\times10^{{{int(e)}}}$ & {vr} & {vg} & "
                   f"{100 * sh['median']:.0f} ({100 * sh['min']:.0f}--{100 * sh['max']:.0f}) \\\\")
    open(os.path.join(DATA, "s4_lowdim_gridcontrol_table.tex"), "w").write("\n".join(tex) + "\n")
    sys.path.insert(0, HERE)
    from inject import inject
    inject("s4_lowdim.tex", "s4_lowdim:tab:gridcontrol", "\n".join(tex[1:]))
    print("\n".join(tex))
    q = N["column_in_strip_vs_placebo"]["adam_lbfgs"]
    print(f"column at t = 1/64 against column at t = 1/2 (equal N): {cell(q)}  CI {q['ci95_t_on_log'][0]:.2f}-{q['ci95_t_on_log'][1]:.2f}  per seed "
          + " ".join(f"{x:.2f}" for x in q["per_seed_ratio"]))
    for arm in ("adam_lbfgs", "adam"):
        print(arm)
        for s, q in N["vs_random"][arm].items():
            print(f"  {s:10s}/random  {cell(q)}  CI {q['ci95_t_on_log'][0]:.2f}-{q['ci95_t_on_log'][1]:.2f}  per seed "
                  + " ".join(f"{x:.2f}" for x in q["per_seed_ratio"]))
        for s, q in N["vs_grid_c"][arm].items():
            print(f"  {s:10s}/grid-c  {cell(q)}  CI {q['ci95_t_on_log'][0]:.2f}-{q['ci95_t_on_log'][1]:.2f}")
    print("agreement with benchmark:", N["agreement_with_benchmark"])


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        import torch
        dev = sys.argv[3] if len(sys.argv) > 3 else ("mps" if torch.backends.mps.is_available() else "cpu")
        run(sys.argv[2], dev)
    elif len(sys.argv) >= 2 and sys.argv[1] == "merge":
        merge()
    else:
        print(__doc__)
