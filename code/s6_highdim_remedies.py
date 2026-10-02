"""Section 6.8: tables, numbers and figures of the two remedies at d = 20, from the stored results of
research_highdim_remedies/ (no training, a few seconds).

    python code/s6_highdim_remedies.py           # write the table rows into sections/s6_highdim.tex, the
                                                 # numbers and rows into data/, the figures into figures/
    python code/s6_highdim_remedies.py --check   # only check that the article and data/ agree with the results

Inputs (relative to the repository root)
    research_highdim_remedies/results/tables.tex, tables_p5.tex     tables of the study (scripts/analyze.py,
                                                                    scripts/analyze_p5.py)
    research_highdim_remedies/results/summary.json, summary_p5.json, attribution.json   (error columns, figures)
    research_highdim_remedies/results/runs_*.jsonl                  load average recorded with each run
    research_highdim_remedies/check/runs_reimpl2.jsonl              second round of the re-check
Outputs
    sections/s6_highdim.tex   rows between the GENERATED markers of Tables s6_highdim:tab:remedies (plain,
                              presolve, lift3c), :tab:altridge and :tab:lift1; ratio columns copied from the
                              study's tables (problem names changed, no digit changed), error columns written as
                              mean (s.d.) in units of 10^-2 from summary.json / summary_p5.json and checked
                              against the study's tables to the digits printed there
    sections/S8_details.tex   rows of Table sC_details:tab:controls (the controls lift3u and rep3, with plain)
    data/s6_highdim_remedies_tables.tex     the same rows
    data/s6_highdim_remedies_recheck.json   seed-paired ratios of the second round of the re-check, its weight
                                            selection and the runs of its plan that were not made; the range
                                            of the load average during the runs of the study
    figures/s6_remedies_ratios.pdf, figures/s6_remedies_altridge.pdf   (plot only, the article's style)
"""
import glob
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)
C = os.path.join(ROOT, "research_highdim_remedies")
RES = os.path.join(C, "results")
sys.path.insert(0, HERE)
from inject import inject  # noqa: E402

CHECK = "--check" in sys.argv
NAMES = [("P2 (f.-m.)", r"\Ptwo\ (f.-m.)"), ("P1", r"\Pone"), ("P2", r"\Ptwo"), ("P3", r"\Pridge"),
         ("P4", r"\Pcospair"), ("P5", r"\Paltridge")]
WORDS = dict(P1="LapD", P2="PoiD", P3="RidgeD", P4="CosPairD", P5="AltRidgeD", P6="SinLinD", P7="ExpCosD")


def rename(cell):
    c = cell.strip()
    for a, b in NAMES:
        if c == a:
            return b
    return c


def tabular_blocks(path):
    """Body rows of each tabular of a file (None for a horizontal rule), without the header row."""
    blocks, cur, inside = [], None, False
    for line in open(path, encoding="utf8"):
        s = line.strip()
        if s.startswith(r"\begin{tabular}"):
            cur, inside = [], True
            continue
        if s.startswith(r"\end{tabular}"):
            blocks.append(cur)
            inside = False
            continue
        if not inside:
            continue
        if s == r"\hline":
            cur.append(None)
        elif s.endswith(r"\\"):
            cur.append([c.strip() for c in s[:-2].split("&")])
    cleaned = []
    for b in blocks:
        idx = [i for i, r in enumerate(b) if r is None]
        body = b[idx[1] + 1:]
        while body and body[-1] is None:
            body.pop()
        cleaned.append(body)
    return cleaned


KEYP = {r"\Pone": "P1", r"\Ptwo\ (f.-m.)": "P2", r"\Pridge": "P3", r"\Pcospair": "P4", r"\Paltridge": "P5"}
CONTROLS = ("lift3u", "rep3")


def sig(x, n):
    """x with n significant digits, trailing zeros kept, no trailing decimal point."""
    t = f"{x:#.{n}g}"
    return t[:-1] if t.endswith(".") else t


def src_value(tok):
    """'$4.75\\times10^{-2}$' or '0.0227' -> (value, half a unit of the last printed digit)."""
    m = re.fullmatch(r"\$([\d.]+)\\times10\^\{(-?\d+)\}\$", tok.strip())
    mant, ex = (m.group(1), int(m.group(2))) if m else (tok.strip(), 0)
    dec = len(mant.split(".")[1]) if "." in mant else 0
    return float(mant) * 10 ** ex, 0.5 * 10 ** (ex - dec)


def err_cell(summary, key, src):
    """Mean (s.d.) of the test error in units of 10^-2 from the study's summary file, checked against the
    printed cell of the study's own table (the same value to the digits printed there)."""
    r = summary[key]["rel_l2"]
    mean, sd = r["mean"], r["sd"]
    m = re.fullmatch(r"(\S+) \((\S+)\)", src.strip())
    (pm, um), (ps, us) = src_value(m.group(1)), src_value(m.group(2))
    assert abs(mean - pm) <= um * 1.0001 and abs(sd - ps) <= us * 1.0001, (key, mean, sd, src)
    return f"{sig(100 * mean, 3)} ({sig(100 * sd, 2)})"


def budget_cell(c):
    """'4000 it' / '4000' -> '4000'; 'eq.\\ CPU (3250 it)' -> 'eq.\\ CPU (3250)'."""
    return c.replace(" it)", ")").replace(" it", "")


def key_of(problem, method, arm, budget):
    return f"{problem}/d20/{'ritz' if method == 'Deep Ritz' else 'pinn'}/{arm}/{'4000' if budget.startswith('4000') else 'eqcpu'}"


def table_remedies():
    """Rows of Table s6_highdim:tab:remedies (plain, presolve, lift3c) and of the table of the two controls
    lift3u and rep3 (Supplementary Section S8, Table sC_details:tab:controls)."""
    S = json.load(open(os.path.join(RES, "summary.json"), encoding="utf8"))
    main, ctrl = [], []
    for r in tabular_blocks(os.path.join(RES, "tables.tex"))[0]:
        if r is None:
            main.append(r"\midrule")
            continue
        r[0] = rename(r[0])
        key = key_of(KEYP[r[0]], r[1], r[2], r[3])
        cells = [r[0], r[1], r[2], budget_cell(r[3]), err_cell(S, key, r[4]), r[5]]
        if r[2] in CONTROLS:
            ctrl.append(cells)
        else:
            main.append(" & ".join(cells) + r" \\")
    # control table: plain rows repeated for reference
    plain = {}
    for r in tabular_blocks(os.path.join(RES, "tables.tex"))[0]:
        if r is not None and r[2] == "plain" and r[1] == "Deep Ritz":
            plain[rename(r[0])] = [rename(r[0]), r[1], "plain", budget_cell(r[3]),
                                   err_cell(S, key_of(KEYP[rename(r[0])], r[1], "plain", r[3]), r[4]), r[5]]
    out, prev = [], None
    for c in ctrl:
        if c[0] != prev:
            if prev is not None:
                out.append(r"\midrule")
            out.append(" & ".join(plain[c[0]]) + r" \\")
            prev = c[0]
        out.append(" & ".join(c) + r" \\")
    return main, out


def table_p5():
    lift1, alt = tabular_blocks(os.path.join(RES, "tables_p5.tex"))[:2]
    S = json.load(open(os.path.join(RES, "summary_p5.json"), encoding="utf8"))
    a = []
    for r in alt:
        if r is None:
            continue
        key = key_of("P5", r[0], r[1], r[2])
        a.append(" & ".join([r[0], r[1], budget_cell(r[2]), err_cell(S, key, r[3]), r[4]]) + r" \\")
    b = []
    for r in lift1:
        if r is None:
            continue
        r[0] = rename(r[0])
        b.append(" & ".join(r) + r" \\")
    return a, b


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf8") if l.strip()]


def recheck():
    """Seed-paired ratios of the second round of the re-check (check/runs_reimpl2.jsonl)."""
    allr = load_jsonl(os.path.join(C, "check", "runs_reimpl2.jsonl"))
    sweep = [r for r in allr if r["block"] == "sweep"]
    R = [r for r in allr if r["block"] != "sweep"]
    out = {"n_runs": len(R), "n_sweep_runs": len(sweep), "problems": {k: WORDS[k] for k in ("P1", "P6", "P7")}}
    sel = {}
    for p in sorted({r["p"] for r in sweep}):
        rs = [r for r in sweep if r["p"] == p]
        best = min(rs, key=lambda r: r["val_rel_l2"])
        sel[p] = {"beta": best["w"], "val_rel_l2": {str(r["w"]): r["val_rel_l2"] for r in rs}}
    out["beta_selected"] = sel

    def get(p, m, d, arm, it=None):
        return {r["seed"]: r for r in R if r["p"] == p and r["method"] == m and r["d"] == d and r["arm"] == arm
                and (it is None or r["iters"] == it)}
    for p, m, d in (("P6", "ritz", 20), ("P6", "ritz", 10), ("P7", "ritz", 20), ("P1", "pinn", 20), ("P6", "pinn", 20)):
        base = get(p, m, d, "plain")
        if not base:
            out[f"{p}/{m}/d{d}"] = "not run"
            continue
        c = {"plain_err": {s: r["rel_l2"] for s, r in sorted(base.items())}, "w": next(iter(base.values()))["w"]}
        for arm in ("presolve", "lift3c", "lift1"):
            for it in sorted({r["iters"] for r in R if r["p"] == p and r["method"] == m and r["d"] == d and r["arm"] == arm}):
                a = get(p, m, d, arm, it)
                seeds = sorted(set(a) & set(base))
                rho = [base[s]["rel_l2"] / a[s]["rel_l2"] for s in seeds]
                c[f"{arm}/{it}"] = {"seeds": seeds, "rho": rho, "median": statistics.median(rho),
                                    "wins": f"{sum(x > 1 for x in rho)}/{len(rho)}",
                                    "err": [a[s]["rel_l2"] for s in seeds]}
        l3, l1 = get(p, m, d, "lift3c", 4000), get(p, m, d, "lift1", 4000)
        if l3 and l1:
            rr = [l1[s]["rel_l2"] / l3[s]["rel_l2"] for s in sorted(set(l3) & set(l1))]
            c["lift1_over_lift3c"] = {"rho": rr, "median": statistics.median(rr)}
        out[f"{p}/{m}/d{d}"] = c
    # runs of the plan of the second round that are not in the file (plan: notes/VERIFICATION.md, section 4.3)
    planned = [("P1", "pinn", 20, arm, s) for arm in ("plain", "presolve", "lift3c") for s in (40, 41, 42)]
    planned += [("P6", "pinn", 20, arm, s) for arm in ("plain", "presolve", "lift3c", "lift1") for s in (40, 41, 42)]
    have = {(r["p"], r["method"], r["d"], r["arm"], r["seed"]) for r in R if r["iters"] == 4000}
    out["missing_runs_of_plan"] = [list(k) for k in planned if k not in have] + [["P1", "pinn", 20, "lift3c, equal CPU", "40-42"]]
    out["missing_runs_note"] = "the equal-CPU runs of lift3c for the PINN on LapD were planned after the 4000-iteration runs"
    # load average recorded with every run of the study
    loads = [r["load1"] for f in sorted(glob.glob(os.path.join(RES, "runs_*.jsonl"))) for r in load_jsonl(f)
             if r.get("load1") is not None]
    out["load1_range"] = [min(loads), max(loads)]
    out["load1_n_runs"] = len(loads)
    return out


def figures():
    """Figures s6_highdim:fig:remedies and s6_highdim:fig:altridge, drawn at text width with the article's style
    (plot only, from results/summary.json, summary_p5.json and attribution.json of the study).  Seed-paired ratios
    are plotted horizontally (one row per arm); circles: 4000 iterations, diamonds: equal CPU time; bars: medians."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from figstyle import apply_style, panel_label, BLUE, ORANGE, INK2
    apply_style(**{"axes.grid": False})
    COL = {"ritz": BLUE, "pinn": ORANGE}
    MN = {"ritz": "Deep Ritz", "pinn": "PINN"}
    S = json.load(open(os.path.join(RES, "summary.json"), encoding="utf8"))
    S5 = json.load(open(os.path.join(RES, "summary_p5.json"), encoding="utf8"))
    A = json.load(open(os.path.join(RES, "attribution.json"), encoding="utf8"))

    def row(ax, y, rho, med, col, eq):
        n = len(rho)
        ys = [y + (k - (n - 1) / 2) * 0.09 for k in range(n)]
        ax.scatter(rho, ys, s=11, color=col, alpha=0.85, marker="D" if eq else "o", zorder=3, linewidths=0)
        if med is not None:
            ax.plot([med, med], [y - 0.32, y + 0.32], color="k", lw=1.4, zorder=4)

    def deco(ax, labels, xlim):
        ax.axvline(1.0, color=INK2, lw=0.8)
        ax.axvline(1.5, color=INK2, lw=0.8, ls="--")
        ax.axvline(1 / 1.1, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if xlim[0] <= t <= xlim[1]]
        ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.grid(axis="x", color="#e4e3df", lw=0.5)

    # ---- Figure s6_highdim:fig:remedies: four problems at d = 20
    cats = [("ritz", "presolve", "4000"), ("ritz", "presolve", "eqcpu"), ("ritz", "lift3c", "4000"),
            ("ritz", "lift3c", "eqcpu"), ("ritz", "lift3u", "4000"), ("ritz", "rep3", "4000"),
            ("pinn", "presolve", "4000"), ("pinn", "lift3c", "4000"), ("pinn", "lift3c", "eqcpu")]
    labels = [f"{MN[m]}, {a}" + (", eq. CPU" if b == "eqcpu" else "") for m, a, b in cats]
    title = {"P1": "(a) LapD", "P2": "(b) PoiD (feature-matched)", "P3": "(c) RidgeD (held out)",
             "P4": "(d) CosPairD (held out)"}
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.2), sharey=True)
    fig.subplots_adjust(left=0.255, right=0.985, top=0.95, bottom=0.17, hspace=0.42, wspace=0.08)
    for ax, P in zip(axes.flat, ("P1", "P2", "P3", "P4")):
        for j, (m, a, b) in enumerate(cats):
            k = f"{P}/d20/{m}/{a}/{b}"
            if k in S:
                r = S[k]["paired"]
                row(ax, j, r["rho"], r["median"], COL[m], b == "eqcpu")
        deco(ax, labels, (0.1, 20))
        panel_label(ax, title[P])
    for ax in axes[1]:
        ax.set_xlabel("seed-paired ratio  err(plain) / err(arm)")
    h = [Line2D([], [], marker="o", ls="", color="k", ms=3.5, label="4000 iterations"),
         Line2D([], [], marker="D", ls="", color="k", ms=3.2, label="equal CPU time"),
         Line2D([], [], color="k", lw=1.4, label="median"),
         Line2D([], [], color=INK2, lw=0.8, ls="--", label="threshold 1.5 (helps)"),
         Line2D([], [], color=INK2, lw=0.8, ls=":", label="threshold 1/1.1 (hurts)"),
         Line2D([], [], color=INK2, lw=0.8, label="no change")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.6, 0.0))
    fig.savefig(os.path.join(ART, "figures", "s6_remedies_ratios.pdf"))
    plt.close(fig)
    print("written figures/s6_remedies_ratios.pdf")

    # ---- Figure s6_highdim:fig:altridge
    fig = plt.figure(figsize=(6.3, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], left=0.245, right=0.985, top=0.955, bottom=0.085,
                          hspace=0.36, wspace=0.85)
    ax_a, ax_c, ax_b, ax_l = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]),
                              fig.add_subplot(gs[1, 1]))
    cats = [(m, a, b) for m, arms in (("ritz", ("presolve", "lift3c", "lift3u", "rep3", "lift1")),
                                      ("pinn", ("presolve", "lift3c", "lift1")))
            for a in arms for b in ("4000", "eqcpu") if f"P5/d20/{m}/{a}/{b}" in S5]
    for j, (m, a, b) in enumerate(cats):
        r = S5[f"P5/d20/{m}/{a}/{b}"]["paired"]
        row(ax_a, j, r["rho"], r["median"], COL[m], b == "eqcpu")
    deco(ax_a, [f"{MN[m]}, {a}" + (", eq. CPU" if b == "eqcpu" else "") for m, a, b in cats], (0.1, 3))
    ax_a.set_xlabel("err(plain) / err(arm)")
    panel_label(ax_a, "(a) AltRidgeD, $d=20$, seeds 10-14")
    cats = [(d, a) for d in (5, 10, 20) for a in ("lift3c", "lift1") if f"P5/d{d}/ritz/{a}/4000" in S5]
    for j, (d, a) in enumerate(cats):
        r = S5[f"P5/d{d}/ritz/{a}/4000"]["paired"]
        row(ax_b, j, r["rho"], r["median"], BLUE, False)
    deco(ax_b, [f"$d={d}$, {a}" for d, a in cats], (0.1, 3))
    ax_b.set_xlabel("err(plain) / err(arm), Deep Ritz")
    panel_label(ax_b, "(b) AltRidgeD, Deep Ritz")
    labels, j = [], 0
    for P, name in (("P1", "LapD"), ("P2", "PoiD (f.-m.)"), ("P3", "RidgeD"), ("P4", "CosPairD"), ("P5", "AltRidgeD")):
        for m in ("ritz", "pinn"):
            r = A[P][m]["lift3c_vs_lift1"]
            if r.get("n", 0):
                row(ax_c, j, r["rho"], r["median"], COL[m], False)
            r2 = A[P][m]["lift3c_eqcpu_vs_lift1"]
            if r2.get("n", 0):
                ax_c.scatter(r2["rho"], [j + 0.3] * len(r2["rho"]), s=9, color=COL[m], alpha=0.6, marker="D",
                             zorder=2, linewidths=0)
            labels.append(f"{name}, {MN[m]}")
            j += 1
    deco(ax_c, labels, (0.2, 20))
    ax_c.set_xlabel("err(lift1) / err(lift3c)")
    panel_label(ax_c, "(c) lift3c against lift1, $d=20$")
    ax_l.axis("off")
    h = [Line2D([], [], marker="o", ls="", color=BLUE, ms=3.5, label="Deep Ritz"),
         Line2D([], [], marker="o", ls="", color=ORANGE, ms=3.5, label="PINN"),
         Line2D([], [], marker="o", ls="", color="k", ms=3.5, label="4000 iterations"),
         Line2D([], [], marker="D", ls="", color="k", ms=3.2, label="lift3c at equal CPU time"),
         Line2D([], [], color="k", lw=1.4, label="median"),
         Line2D([], [], color=INK2, lw=0.8, ls="--", label="threshold 1.5"),
         Line2D([], [], color=INK2, lw=0.8, ls=":", label="threshold 1/1.1"),
         Line2D([], [], color=INK2, lw=0.8, label="ratio 1")]
    ax_l.legend(handles=h, loc="center left", fontsize=7, ncol=1, borderaxespad=0.0, bbox_to_anchor=(-0.35, 0.5))
    fig.savefig(os.path.join(ART, "figures", "s6_remedies_altridge.pdf"))
    plt.close(fig)
    print("written figures/s6_remedies_altridge.pdf")


def main():
    alt, lift1 = table_p5()
    main_rows, ctrl_rows = table_remedies()
    blocks = {"s6_highdim:tab:remedies": main_rows, "s6_highdim:tab:altridge": alt,
              "s6_highdim:tab:lift1": lift1, "sC_details:tab:controls": ctrl_rows}
    where = {name: ("S8_details.tex" if name.startswith("sC_") else "s6_highdim.tex") for name in blocks}
    gen = "\n".join(f"% {name}\n" + "\n".join(r) for name, r in blocks.items()) + "\n"
    num = recheck()
    if CHECK:
        ok = True
        for name, r in blocks.items():
            same = inject(where[name], name, "\n".join(r), check=True)
            print(f"{'ok  ' if same else 'FAIL'} {name}")
            ok &= bool(same)
        ok &= open(os.path.join(ART, "data", "s6_highdim_remedies_tables.tex"), encoding="utf8").read() == gen
        stored = json.load(open(os.path.join(ART, "data", "s6_highdim_remedies_recheck.json"), encoding="utf8"))
        ok &= json.dumps(stored, sort_keys=True) == json.dumps(json.loads(json.dumps(num)), sort_keys=True)
        print("CHECK PASSED" if ok else "CHECK FAILED")
        sys.exit(0 if ok else 1)
    for name, r in blocks.items():
        inject(where[name], name, "\n".join(r))
    open(os.path.join(ART, "data", "s6_highdim_remedies_tables.tex"), "w", encoding="utf8").write(gen)
    json.dump(num, open(os.path.join(ART, "data", "s6_highdim_remedies_recheck.json"), "w", encoding="utf8"), indent=1)
    print("written data/s6_highdim_remedies_tables.tex, data/s6_highdim_remedies_recheck.json and the table rows")
    figures()


if __name__ == "__main__":
    main()
