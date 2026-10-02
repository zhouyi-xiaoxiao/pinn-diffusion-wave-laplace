# -- added for the public repository: absolute paths of the development machine are replaced by
# -- paths relative to the repository root, through the helper _repo_path defined here.
import os as _os


def _repo_path(rel=""):
    d = _os.path.dirname(_os.path.abspath(__file__))
    while not (_os.path.isdir(_os.path.join(d, "research_benchmark")) and _os.path.isdir(_os.path.join(d, "research_highdim"))):
        p = _os.path.dirname(d)
        if p == d:
            raise RuntimeError("repository root not found")
        d = p
    return _os.path.join(d, rel) if rel else d
# -- end of the added lines

"""Markdown + LaTeX tables from results/summary.json, floors.json, cost_vs_d.json (pure stdlib)."""
import json
ROOT = _repo_path("research_highdim")
S = json.load(open(f"{ROOT}/results/summary.json"))
C = json.load(open(f"{ROOT}/results/cost_vs_d.json"))
g = lambda p, m, d: next(s for s in S if s["problem"] == p and s["method"] == m and s["d"] == d)
DS = sorted({s["d"] for s in S})


def sci(x):
    m, e = f"{x:.2e}".split("e"); return f"{m}e{int(e)}"


def tex(x):
    m, e = f"{x:.2e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"


md, tx = [], []
for p, name in [("laplace", "P1 (Laplace)"), ("poisson", "P2 (Poisson)")]:
    md += [f"\n**{name}** — test relative $L^2$ error, mean ± sample s.d. over 3 seeds (min–max in brackets)\n",
           "| d | PINN rel. L2 | Deep Ritz rel. L2 | PINN rel. H1-semi | Ritz rel. H1-semi | PINN centred L2 | Ritz centred L2 | best affine fit | Ritz/PINN |",
           "|---|---|---|---|---|---|---|---|---|"]
    tx += [f"% {name}", "\\begin{tabular}{rcccccc}", "\\toprule",
           "$d$ & PINN rel.\\ $L^2$ & Deep Ritz rel.\\ $L^2$ & PINN rel.\\ $H^1$ & Deep Ritz rel.\\ $H^1$ & affine floor & Ritz/PINN\\\\", "\\midrule"]
    for d in DS:
        a, b = g(p, "pinn", d), g(p, "ritz", d)
        md.append(f"| {d} | {sci(a['rel_l2_mean'])} ± {sci(a['rel_l2_std'])} [{sci(a['rel_l2_min'])}, {sci(a['rel_l2_max'])}] | "
                  f"{sci(b['rel_l2_mean'])} ± {sci(b['rel_l2_std'])} [{sci(b['rel_l2_min'])}, {sci(b['rel_l2_max'])}] | "
                  f"{sci(a['rel_h1semi_mean'])} | {sci(b['rel_h1semi_mean'])} | {sci(a['rel_l2_centered_mean'])} | {sci(b['rel_l2_centered_mean'])} | "
                  f"{a['affine_rel_l2']:.3f} | {b['rel_l2_mean']/a['rel_l2_mean']:.2f} |")
        tx.append(f"{d} & {tex(a['rel_l2_mean'])} $\\pm$ {tex(a['rel_l2_std'])} & {tex(b['rel_l2_mean'])} $\\pm$ {tex(b['rel_l2_std'])} & "
                  f"{tex(a['rel_h1semi_mean'])} & {tex(b['rel_h1semi_mean'])} & {a['affine_rel_l2']:.3f} & {b['rel_l2_mean']/a['rel_l2_mean']:.2f}\\\\")
    tx += ["\\bottomrule", "\\end{tabular}", ""]
md += ["\n**Cost per training iteration** — CPU ms (1 thread), median of 7 rounds × 10 iterations; ratio to Deep Ritz paired per round, median [min, max]\n",
       "| d | Deep Ritz | PINN forward-Laplacian | ratio | PINN nested autograd | ratio |", "|---|---|---|---|---|---|"]
c = lambda m, d: next(r for r in C if r["method"] == m and r["d"] == d)
for d in sorted({r["d"] for r in C}):
    r, f, n = c("ritz", d), c("pinn_forward", d), c("pinn_nested", d)
    md.append(f"| {d} | {r['cpu_ms_median']:.2f} | {f['cpu_ms_median']:.2f} | {f['ratio_to_ritz_median']:.2f} [{f['ratio_to_ritz_min']:.2f}, {f['ratio_to_ritz_max']:.2f}] | "
              f"{n['cpu_ms_median']:.2f} | {n['ratio_to_ritz_median']:.2f} [{n['ratio_to_ritz_min']:.2f}, {n['ratio_to_ritz_max']:.2f}] |")
# least-squares line cost = a + b d
for m in ["ritz", "pinn_forward", "pinn_nested"]:
    xs = [r["d"] for r in C if r["method"] == m]; ys = [r["cpu_ms_median"] for r in C if r["method"] == m]
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs); a = my - b * mx
    md.append(f"\nlinear fit {m}: cost ≈ {a:.2f} + {b:.3f}·d ms  (cost(100)/cost(2) = {ys[-1]/ys[0]:.1f})")
open(f"{ROOT}/results/tables.md", "w").write("\n".join(md) + "\n")
open(f"{ROOT}/results/table_main.tex", "w").write("\n".join(tx) + "\n")
print("\n".join(md))
