"""Figure: seed-paired error ratios rho = err(plain) / err(remedy) at d = 20 (figures/paired_ratios.pdf/.png).
Source: results/summary.json (scripts/analyze.py)."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = json.load(open(os.path.join(ROOT, "results", "summary.json")))
COL = {"ritz": "#1f6fb4", "pinn": "#d1495b"}
CATS = [("ritz", "presolve", "4000"), ("ritz", "presolve", "eqcpu"), ("ritz", "lift3c", "4000"),
        ("ritz", "lift3c", "eqcpu"), ("ritz", "lift3u", "4000"), ("ritz", "rep3", "4000"),
        ("pinn", "presolve", "4000"), ("pinn", "presolve", "eqcpu"), ("pinn", "lift3c", "4000"),
        ("pinn", "lift3c", "eqcpu")]
TITLE = {"P1": "P1: pair products", "P2": "P2: separable cosines (feature-matched)",
         "P3": "P3: ridge cos(2s) (held out)", "P4": "P4: cosine pairs (held out)"}
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
for ax, P in zip(axes.flat, ("P1", "P2", "P3", "P4")):
    cats = [c for c in CATS if f"{P}/d20/{c[0]}/{c[1]}/{c[2]}" in S]
    for j, (m, arm, bud) in enumerate(cats):
        r = S[f"{P}/d20/{m}/{arm}/{bud}"]["paired"]
        rho = np.array(r["rho"])
        x = j + np.linspace(-0.18, 0.18, len(rho))
        ax.scatter(x, rho, s=14, color=COL[m], alpha=0.75, zorder=3,
                   marker="o" if bud == "4000" else "D")
        ax.hlines(r["median"], j - 0.3, j + 0.3, color="black", lw=1.6, zorder=4)
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.axhline(1.5, color="0.5", lw=0.8, ls="--")
    ax.axhline(1 / 1.1, color="0.5", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{'Ritz' if m == 'ritz' else 'PINN'}\n{arm}\n{'4000 it' if b == '4000' else 'eq. CPU'}"
                        for m, arm, b in cats], fontsize=7)
    ax.set_title(TITLE[P], fontsize=9)
    ax.grid(axis="y", which="both", color="0.92", lw=0.5)
for ax in axes[:, 0]:
    ax.set_ylabel("paired ratio  err(plain) / err(arm)")
fig.text(0.5, 0.005, "d = 20, seeds 10-14; dots: seeds (circles 4000 iterations, diamonds equal CPU time); bar: median.\n"
         "Solid line 1 (no change), dashed 1.5 (threshold for 'helps'), dotted 1/1.1 (threshold for 'hurts'). "
         "Above 1: the arm is more accurate than plain.", ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.045, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "figures", f"paired_ratios.{ext}"), dpi=200)
print("written figures/paired_ratios.pdf/.png")
