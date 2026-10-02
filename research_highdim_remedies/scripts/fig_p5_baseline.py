"""Figure for the addendum PREREG_C1_P5.md (figures/p5_and_centred_baseline.pdf/.png).
(a) P5, d = 20: seed-paired ratios err(plain)/err(arm); (b) P5, Deep Ritz: lift3c and lift1 against plain at
d = 5, 10, 20; (c) the lift against the centred-input network: err(lift1)/err(lift3c) on P1-P5, d = 20.
Sources: results/summary_p5.json, results/attribution.json (scripts/analyze_p5.py)."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = json.load(open(os.path.join(ROOT, "results", "summary_p5.json")))
A = json.load(open(os.path.join(ROOT, "results", "attribution.json")))
COL = {"ritz": "#1f6fb4", "pinn": "#d1495b"}
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[2.3, 1.0, 1.6]))


def dots(ax, j, r, col, marker):
    rho = np.array(r["rho"])
    x = j + np.linspace(-0.18, 0.18, len(rho))
    ax.scatter(x, rho, s=14, color=col, alpha=0.8, zorder=3, marker=marker)
    ax.hlines(r["median"], j - 0.3, j + 0.3, color="black", lw=1.6, zorder=4)


def deco(ax):
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.axhline(1.5, color="0.5", lw=0.8, ls="--")
    ax.axhline(1 / 1.1, color="0.5", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", color="0.92", lw=0.5)


ax = axes[0]
cats = [(m, a, b) for m, arms in (("ritz", ("presolve", "lift3c", "lift3u", "rep3", "lift1")),
                                  ("pinn", ("presolve", "lift3c", "lift1")))
        for a in arms for b in ("4000", "eqcpu") if f"P5/d20/{m}/{a}/{b}" in S]
for j, (m, a, b) in enumerate(cats):
    dots(ax, j, S[f"P5/d20/{m}/{a}/{b}"]["paired"], COL[m], "o" if b == "4000" else "D")
deco(ax)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([f"{'Ritz' if m == 'ritz' else 'PINN'}\n{a}\n{'4000 it' if b == '4000' else 'eq. CPU'}"
                    for m, a, b in cats], fontsize=6.5)
ax.set_ylabel("paired ratio  err(plain) / err(arm)")
ax.set_title("(a) P5, d = 20, seeds 10-14", fontsize=9)

ax = axes[1]
cats = [(d, a) for d in (5, 10, 20) for a in ("lift3c", "lift1") if f"P5/d{d}/ritz/{a}/4000" in S]
for j, (d, a) in enumerate(cats):
    dots(ax, j, S[f"P5/d{d}/ritz/{a}/4000"]["paired"], "#1f6fb4" if a == "lift3c" else "#7aa6d6", "o")
deco(ax)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([f"d = {d}\n{a}" for d, a in cats], fontsize=7)
ax.set_ylabel("err(plain) / err(arm), Deep Ritz")
ax.set_title("(b) P5, Deep Ritz, 4000 it", fontsize=9)

ax = axes[2]
j, ticks = 0, []
for P in ("P1", "P2", "P3", "P4", "P5"):
    for m in ("ritz", "pinn"):
        r = A[P][m]["lift3c_vs_lift1"]
        if r.get("n", 0):
            dots(ax, j, r, COL[m], "o")
        r2 = A[P][m]["lift3c_eqcpu_vs_lift1"]
        if r2.get("n", 0):
            rho = np.array(r2["rho"])
            ax.scatter(j + 0.33 + np.zeros(len(rho)), rho, s=10, color=COL[m], alpha=0.5, marker="D", zorder=2)
        ticks.append(f"{P}{chr(10) + '(f.-m.)' if P == 'P2' else ''}\n{'Ritz' if m == 'ritz' else 'PINN'}")
        j += 1
deco(ax)
ax.set_xticks(range(len(ticks)))
ax.set_xticklabels(ticks, fontsize=6.5)
ax.set_ylabel("err(lift1) / err(lift3c)")
ax.set_title("(c) lift3c against the centred input N(2x - 1), d = 20", fontsize=9)
fig.text(0.5, 0.005, "Dots: seeds (circles 4000 iterations, diamonds lift3c at equal CPU time); bar: median of the 4000-iteration "
         "ratios. Solid 1, dashed 1.5, dotted 1/1.1. Above 1: the arm in the denominator is more accurate.",
         ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "figures", f"p5_and_centred_baseline.{ext}"), dpi=200)
print("written figures/p5_and_centred_baseline.pdf/.png")
