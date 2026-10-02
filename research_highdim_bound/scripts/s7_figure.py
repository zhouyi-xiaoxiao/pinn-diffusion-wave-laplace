"""Figure: (a) bounds on C_d^2 against d (U_d, L_d, the elementary lower bound 1/(2d), Payne's convex-domain bound
1/2, Rayleigh-Ritz lower bounds, the article's constant);
(b) efficiency eta = bound/error and (c) boundary share against d for the saved networks (fresh set).
Inputs: results/theorem2_checks.json, results/restricted_lanczos.json, results/eval_confirm.jsonl.
Outputs: figures/t2_bound.pdf, figures/t2_bound.png."""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from t2common import U, L, RES, FIG, load_jsonl

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)

# (a)
dd = np.unique(np.round(np.logspace(0, 2, 60)).astype(int))
ax[0].plot(dd, [U(d) for d in dd], "-", color="k", lw=1.4, label=r"upper bound $U_d$ (Theorem 2)")
ax[0].plot(dd, [L(d, 400_001) for d in dd], "--", color="k", lw=1.0, label=r"lower bound $L_d$ (Theorem 2)")
ax[0].plot(dd, [1 / (2 * d) for d in dd], "-.", color="0.55", lw=1.0, label=r"lower bound $1/(2d)$ (test function $h=1$)")
ax[0].plot(dd, [0.5 for d in dd], "-", color="#1b9e77", lw=1.2, label=r"Payne's bound $1/2$ (convex domains, width 1)")
ax[0].plot(dd, [4 * (1 + 1 / (d * math.pi ** 2)) for d in dd], ":", color="0.45", lw=1.2, label=r"square of the article's constant $2\sqrt{1+1/(d\pi^2)}$")
rr = json.load(open(os.path.join(RES, "restricted_lanczos.json")))["rows"]
best = {}
for r in rr:
    best[r["d"]] = max(best.get(r["d"], 0), r["lam"])
for r in json.load(open(os.path.join(RES, "theorem2_checks.json")))["A6_lanczos"]:
    best[r["d"]] = max(best.get(r["d"], 0), r["lam_max"])
ks = sorted(best)
ax[0].plot(ks, [best[k] for k in ks], "o", color="#7a3db8", ms=4, label="computed lower bounds (Rayleigh-Ritz)")
ax[0].plot([1], [0.5], "s", color="#7a3db8", ms=4)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("dimension $d$"); ax[0].set_ylabel(r"$C_d^2=\|T\|^2$")
ax[0].set_title("(a) stability constant", loc="left")
ax[0].set_ylim(0.004, 2000)
ax[0].legend(fontsize=6.0, frameon=False, loc="upper right")

# (b), (c)
rows = [r for r in load_jsonl(os.path.join(RES, "eval_confirm.jsonl")) if r["set"] == "fresh"]
styles = {("laplace", "pinn"): ("#d95f02", "o", "-", "P1, PINN"), ("laplace", "ritz"): ("#1f78b4", "o", "-", "P1, Deep Ritz"),
          ("poisson", "pinn"): ("#d95f02", "s", "--", "P2, PINN"), ("poisson", "ritz"): ("#1f78b4", "s", "--", "P2, Deep Ritz")}
DS = [2, 3, 5, 10, 20]
off = {("laplace", "pinn"): -0.06, ("laplace", "ritz"): -0.02, ("poisson", "pinn"): 0.02, ("poisson", "ritz"): 0.06}
for key, (c, mk, ls, lab) in styles.items():
    sel = [r for r in rows if r["family"] == "main" and (r["problem"], r["method"]) == key]
    for panel, fld in ((1, "eta"), (2, "share_bd")):
        xs = [r["d"] * math.exp(off[key]) for r in sel]
        ys = [r[fld] for r in sel]
        face = [c if r["confirmatory"] else "white" for r in sel]
        ax[panel].scatter(xs, ys, marker=mk, s=14, facecolors=face, edgecolors=c, linewidths=0.8, zorder=3)
        means = [np.mean([r[fld] for r in sel if r["d"] == d]) for d in DS]
        ax[panel].plot([d * math.exp(off[key]) for d in DS], means, ls, color=c, lw=1.0, label=lab)
for r in rows:
    if r["family"] in ("long", "equaltime"):
        c, mk, _, _ = styles[(r["problem"], r["method"])]
        for panel, fld in ((1, "eta"), (2, "share_bd")):
            ax[panel].scatter([r["d"] * 1.18], [r[fld]], marker="^" if r["family"] == "long" else "v", s=16,
                              facecolors="none", edgecolors=c, linewidths=0.8, zorder=3)
ax[1].axhline(3, color="0.5", lw=0.8, ls=":")
ax[1].axhline(1, color="0.5", lw=0.8)
ax[1].text(1.95, 1.1, r"dotted line: limit 3, pre-registered for $d\geq5$", fontsize=6.3, color="0.35")
ax[1].set_yscale("log"); ax[1].set_ylim(0.9, 30)
ax[1].set_yticks([1, 2, 3, 5, 10, 20]); ax[1].set_yticklabels(["1", "2", "3", "5", "10", "20"])
for p in (1, 2):
    ax[p].set_xscale("log"); ax[p].minorticks_off(); ax[p].set_xticks(DS); ax[p].set_xticklabels([str(d) for d in DS])
    ax[p].set_xlabel("dimension $d$")
ax[1].set_ylabel(r"$\eta=(B_{\rm int}+B_{\rm bd})\,/\,\|v-u^*\|$")
ax[1].set_title("(b) efficiency of the bound", loc="left")
ax[2].set_ylabel(r"boundary share $B_{\rm bd}/(B_{\rm int}+B_{\rm bd})$")
ax[2].set_ylim(0, 1.02)
ax[2].set_title("(c) where the bound sits", loc="left")
h, l = ax[2].get_legend_handles_labels()
from matplotlib.lines import Line2D
h += [Line2D([], [], marker="^", ls="", mfc="none", mec="k", ms=5), Line2D([], [], marker="v", ls="", mfc="none", mec="k", ms=5),
      Line2D([], [], marker="o", ls="", mfc="white", mec="k", ms=4)]
l += ["16 000 iterations, d = 10", "Deep Ritz at the PINN's CPU time, d = 10", "open: non-confirmatory (pilot)"]
ax[2].legend(h, l, fontsize=6.3, frameon=False, loc="lower right")
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, "t2_bound.pdf"))
fig.savefig(os.path.join(FIG, "t2_bound.png"), dpi=200)
print("saved")
