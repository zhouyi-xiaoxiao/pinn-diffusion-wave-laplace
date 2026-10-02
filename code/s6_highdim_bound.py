"""Section 6.7: tables, numbers and figure of the stability constant, the computable error bound and the
exact penalty bias, from the stored results of research_highdim_bound/ (no training, a few seconds).

    python code/s6_highdim_bound.py           # write the table rows into sections/s6_highdim.tex, the
                                              # numbers and rows into data/, the figure into figures/
    python code/s6_highdim_bound.py --check   # only check that the article and data/ agree with the results

Inputs (relative to the repository root)
    research_highdim_bound/results/table_payne.tex, table_theorem2.tex   constants of Theorem 6.7 (the study's
                                                                        scripts s1, s1b, s9 make them)
    research_highdim_bound/results/table_eta_by_d.tex                    the bound on the main networks (s8)
    research_highdim_bound/results/bias_table.tex                        the exact penalty bias (s6)
    research_highdim_bound/results/decisions.json, eval_confirm.jsonl    pre-registered evaluation (s4, s5)
    research_highdim_bound/check/out/c4_fresh.jsonl, check/out_r2/r2_fresh.jsonl   retrained networks of the re-check
Outputs
    sections/s6_highdim.tex   rows between the GENERATED markers of Tables s6_highdim:tab:kappa, :tab:eta,
                              :tab:biasexact (copied from the study's tables; problem names and number format
                              changed, no digit changed)
    data/s6_highdim_bound_tables.tex     the same rows
    data/s6_highdim_bound_numbers.json   numbers quoted in the text: the constants re-evaluated from their
                                         formulas, the counts of networks on which the bound was evaluated,
                                         the second round of the re-check (r2_fresh.jsonl)
    figures/s6_bound.pdf                 Figure s6_highdim:fig:bound (plot only; also reads data/s6_highdim_torsion.json,
                                         written by code/s6_highdim_torsion.py)
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)
B = os.path.join(ROOT, "research_highdim_bound")
RES = os.path.join(B, "results")
sys.path.insert(0, HERE)
from inject import inject  # noqa: E402

CHECK = "--check" in sys.argv
# torsion-function bound m_d of the cube (code/s6_highdim_torsion.py), last column of Table s6_highdim:tab:kappa
TORS = {r["d"]: r["m_d"] for r in json.load(open(os.path.join(ART, "data", "s6_highdim_torsion.json"),
                                                 encoding="utf8"))["rows"]}
NAMES = [("P2 (f.-m.)", r"\Ptwo\ (f.-m.)"), ("P1", r"\Pone"), ("P2", r"\Ptwo"), ("P3", r"\Pridge"),
         ("P4", r"\Pcospair"), ("P5", r"\Paltridge")]


def rename(cell):
    c = cell.strip()
    for a, b in NAMES:
        if c == a:
            return b
    return c


def sci(tok):
    """'1.234e-02' -> '$\\sci{1.234}{-2}$' (same digits)."""
    m = re.fullmatch(r"(-?\d+\.\d+)e([+-]\d+)", tok.strip())
    return r"$\sci{%s}{%d}$" % (m.group(1), int(m.group(2))) if m else tok.strip()


def fmt_minmax(s):
    return re.sub(r"\((\S+)--(\S+)\)", r"[\1, \2]", s)


def rows(path):
    out = []
    for line in open(path, encoding="utf8"):
        if line.lstrip().startswith("%"):
            continue
        line = line.split("%")[0].rstrip()
        if line.strip().endswith(r"\\"):
            out.append([c.strip() for c in line.strip()[:-2].split("&")])
    return out


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf8") if l.strip()]


# ------------------------------------------------------------------ table rows
def table_kappa():
    payne = {int(r[0]): r for r in rows(os.path.join(RES, "table_payne.tex"))}
    thm = {int(r[0]): r for r in rows(os.path.join(RES, "table_theorem2.tex"))}
    out = []
    for d in sorted(payne):
        p, t = payne[d], thm[d]
        assert p[2] == t[1] and p[3] == t[3], (d, p, t)  # L_d and U_d agree between the two tables
        out.append(" & ".join([str(d), p[1], p[2], t[2], p[3], t[4], p[5], t[5], t[6], t[7], f"{TORS[d]:.4f}",
                                f"{math.sqrt(TORS[d] / U(d)):.3f}"]) + r" \\")
    return out


def table_eta():
    out, prev = [], None
    for r in rows(os.path.join(RES, "table_eta_by_d.tex")):
        key = (r[0], r[1])
        if prev is not None and key != prev:
            out.append(r"\addlinespace")
        prev = key
        cells = [rename(r[0]), r[1], r[2], sci(r[3]), sci(r[4]), sci(r[5]), fmt_minmax(r[6]), fmt_minmax(r[7])]
        out.append(" & ".join(cells) + r" \\")
    return out


def table_bias():
    out = []
    for r in rows(os.path.join(RES, "bias_table.tex")):
        out.append(" & ".join([r[0]] + [sci(c) for c in r[1:6]] + r[6:]) + r" \\")
    return out


# ------------------------------------------------------------------ numbers
def U(d):
    if d == 1:
        return 0.5
    a = math.sqrt(d - 1)
    return math.tanh(math.pi * a / 2) / (math.pi * a)


def L(d, jmax=2_000_001):
    s = 0.0
    for j in range(1, jmax, 2):
        s += j * j / (j * j + d - 1) ** 2
    return 4 / math.pi ** 2 * s


def numbers():
    num = {"source": "code/s6_highdim_bound.py"}
    # constants of Table s6_highdim:tab:kappa, re-evaluated from their formulas and compared at the printed digits
    k = []
    for r in table_kappa():
        c = [x.strip() for x in r[:-2].split("&")]
        d = int(c[0])
        ud, ld = U(d), (0.5 if d == 1 else L(d))
        ok = (f"{1 / (2 * d):.4f}" == c[1] and f"{ld:.4f}" == c[2] and f"{ud:.4f}" == c[4]
              and f"{math.sqrt(0.5 / ud):.3f}" == c[6] and f"{math.sqrt(ud):.3f}" == c[7]
              and f"{math.sqrt(2 * d * ud):.3f}" == c[8] and f"{2 * math.sqrt(1 + 1 / (d * math.pi ** 2)):.3f}" == c[9])
        k.append({"d": d, "U_d": ud, "L_d": ld, "half_over_d": 1 / (2 * d), "formulas_match_table": ok})
    num["kappa_constants"] = k
    num["L_vs_elementary"] = {str(d): {"L_d": L(d), "1/(2d)": 1 / (2 * d), "elementary_larger": 1 / (2 * d) > L(d)}
                              for d in range(2, 11)}
    num["sqrt_d_U_d_at_1e6"] = math.sqrt(1e6) * U(10 ** 6)
    # networks on which the bound was evaluated
    dec = json.load(open(os.path.join(RES, "decisions.json"), encoding="utf8"))
    conf = load_jsonl(os.path.join(RES, "eval_confirm.jsonl"))
    files = sorted({r["file"] for r in conf})
    c4 = load_jsonl(os.path.join(B, "check", "out", "c4_fresh.jsonl"))
    r2 = load_jsonl(os.path.join(B, "check", "out_r2", "r2_fresh.jsonl"))
    num["n_networks_bound_valid"] = {
        "saved_networks": len(files), "saved_H1_violations": len(dec["H1"]["violations"]),
        "recheck_round1": len(c4), "recheck_round1_valid": sum(r["bound"] > r["err"] for r in c4),
        "recheck_round1_eta_range": [min(r["eta"] for r in c4), max(r["eta"] for r in c4)],
        "recheck_round2": len(r2), "recheck_round2_valid": sum(r["bound"] > r["err"] for r in r2)}
    n = num["n_networks_bound_valid"]
    n["total"] = n["saved_networks"] + n["recheck_round1"] + n["recheck_round2"]
    n["all_valid"] = (n["saved_H1_violations"] == 0 and n["recheck_round1_valid"] == n["recheck_round1"]
                      and n["recheck_round2_valid"] == n["recheck_round2"])
    by_d = {}
    for r in [r for r in conf if r["set"] == "fresh"] + c4 + r2:
        by_d.setdefault(r["d"], []).append(r["eta"])
    num["eta_by_d_all_networks"] = {str(d): {"n": len(v), "min": min(v), "max": max(v)} for d, v in sorted(by_d.items())}
    num["eta_d10_d20_all_networks"] = [min(by_d[10] + by_d[20]), max(by_d[10] + by_d[20])]
    # interior term against the error on the main networks (Remark s6_highdim:rem:split: ||e_0|| <= B_int)
    fresh_main = [r for r in conf if r["set"] == "fresh" and r["family"] == "main"]
    num["B_int_over_error_main_networks"] = {
        str(d): {f"{p}_{m}": [min(r["B_int"] / r["err_abs"] for r in fresh_main if (r["d"], r["problem"], r["method"]) == (d, p, m)),
                             max(r["B_int"] / r["err_abs"] for r in fresh_main if (r["d"], r["problem"], r["method"]) == (d, p, m))]
                 for p in ("laplace", "poisson") for m in ("pinn", "ritz")} for d in (10, 20)}
    # weight sweep at d = 5 (pre-registered descriptive output H4) and the sampled maximum-principle quantity
    sw = [r for r in conf if r["set"] == "fresh" and r["family"] == "sweep"]
    num["sweep_d5"] = {"n": len(sw), "eta_range": [min(r["eta"] for r in sw), max(r["eta"] for r in sw)],
                       "n_eta_above_3": sum(r["eta"] > 3 for r in sw)}
    # unit square: the Rayleigh quotient of the re-check against the value computed by Antunes and Gazzola (2013)
    rq = json.load(open(os.path.join(B, "check", "out_r2", "r2_rr_quadrature.json"), encoding="utf8"))["K20_nq800"]
    ag = 1.96179 * math.sqrt(math.pi)  # refs.bib, antunes2013convex: 1.96179 for the square of area pi
    num["unit_square"] = {"rayleigh_quotient_harmonic_side": rq, "delta1_upper_from_it": 1 / rq,
                          "antunes_gazzola_rescaled": ag, "antunes_gazzola_rel_above": ag * rq - 1}
    # Observation sA_proofs:obs:sharp: the bound of Corollary (b) against the finite-difference Robin solution at d = 2
    fd = json.load(open(os.path.join(ART, "data", "sA_proofs_checks.json"), encoding="utf8"))["robin"]
    nrm = fd["norm_dn_ustar_P1_d2"]
    num["corollary_b_over_fd_bias_d2"] = {
        "sqrt_U2_N_over_2": math.sqrt(U(2)) * nrm / 2,
        "ratio": {b: math.sqrt(U(2)) * nrm / (2 * float(b)) / fd["fd_robin_P1_d2_N200"][b]["L2"] for b in ("10", "100", "1000")},
        "source": "data/sA_proofs_checks.json (robin: norm_dn_ustar_P1_d2, fd_robin_P1_d2_N200.L2)"}
    num["plotted_in_figure"] = {"main": len(fresh_main), "long_and_equaltime": sum(r["family"] in ("long", "equaltime")
                                                                                    for r in conf if r["set"] == "fresh"),
                                "sweep": len(sw)}
    # second round of the re-check: efficiencies per cell and the ranking of PINN against Deep Ritz at d = 20
    def cell(r):
        return f"{r['prob']}_{r['method']}{'_eqcpu' if r['iters'] > 4000 else ''}_d{r['d']}"
    cells = {}
    for r in r2:
        cells.setdefault(cell(r), []).append(r)
    num["recheck_r2"] = {
        "min_z": min(r["z"] for r in r2),
        "lap_check_rel_max": max(r["lap_check_rel"] for r in r2),
        "cells": {c: {"n": len(v), "eta": [min(x["eta"] for x in v), max(x["eta"] for x in v)],
                      "rel_err": [min(x["rel_err"] for x in v), max(x["rel_err"] for x in v)],
                      "iters": sorted({x["iters"] for x in v})} for c, v in sorted(cells.items())}}
    pinn = [r for r in r2 if cell(r) == "P1_pinn_d20"]
    for other, key in (("P1_ritz_d20", "equal_iterations"), ("P1_ritz_eqcpu_d20", "equal_cpu")):
        rit = [r for r in r2 if cell(r) == other]
        good = sum((p["bound"] < q["bound"]) == (p["err"] < q["err"]) for p in pinn for q in rit)
        num["recheck_r2"][f"pairs_ranked_like_error_{key}"] = f"{good}/{len(pinn) * len(rit)}"
    return num


# ------------------------------------------------------------------ figure
def figure():
    """Figure s6_highdim:fig:bound, drawn at text width with the article's style (plot only, stored results).

    (a) bounds of kappa_d^2: U_d, L_d, 1/(2d), Payne's 1/2, the torsion-function bound m_d of the cube
        (data/s6_highdim_torsion.json), the computed Rayleigh-Ritz values (results/restricted_lanczos.json,
        results/theorem2_checks.json) and (2 C_d)^2 of Proposition s3_methods:prop:robin(c);
    (b) efficiency and (c) boundary share of the bound on all 88 saved networks, new Monte-Carlo set
        (results/eval_confirm.jsonl): main runs (lines: seed means), 16 000-iteration and equal-compute runs
        at d = 10, and the 16 networks of the weight sweep at d = 5.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from figstyle import apply_style, panel_label, BLUE, ORANGE, GREY, INK2
    apply_style()
    C = {"pinn": ORANGE, "ritz": BLUE}
    MK = {"laplace": "o", "poisson": "s"}
    LS = {"laplace": "-", "poisson": "--"}
    PN = {"laplace": "LapD", "poisson": "PoiD"}
    MN = {"pinn": "PINN", "ritz": "Deep Ritz"}
    fig = plt.figure(figsize=(6.3, 5.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=0.42, wspace=0.34,
                          left=0.085, right=0.985, top=0.955, bottom=0.17)
    ax_a, ax_l = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_b, ax_c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    # (a)
    dd = sorted({int(round(x)) for x in [10 ** (k / 40) for k in range(81)]})
    tors = json.load(open(os.path.join(ART, "data", "s6_highdim_torsion.json"), encoding="utf8"))["rows"]
    best = {}
    for r in json.load(open(os.path.join(RES, "restricted_lanczos.json"), encoding="utf8"))["rows"]:
        best[r["d"]] = max(best.get(r["d"], 0), r["lam"])
    for r in json.load(open(os.path.join(RES, "theorem2_checks.json"), encoding="utf8"))["A6_lanczos"]:
        best[r["d"]] = max(best.get(r["d"], 0), r["lam_max"])
    best[1] = 0.5
    ks = sorted(best)
    curves = [
        (ax_a.plot(dd, [4 * (1 + 1 / (d * math.pi ** 2)) for d in dd], ":", color=GREY, lw=1.2)[0],
         r"$(2C_d)^2$, constant of the penalty-bias bound"),
        (ax_a.plot(dd, [0.5] * len(dd), "-", color="#1b9e77", lw=1.2)[0], r"Payne's bound $\frac{1}{2}$ (convex domains)"),
        (ax_a.plot([r["d"] for r in tors], [r["m_d"] for r in tors], "-.", color="#7a3db8", lw=1.0, marker="^", ms=3)[0],
         r"torsion-function bound $m_d$ of the cube"),
        (ax_a.plot(dd, [U(d) for d in dd], "-", color="k", lw=1.4)[0], r"upper bound $U_d$"),
        (ax_a.plot(ks, [best[k] for k in ks], "o", color="#b2182b", ms=3.2)[0], "computed Rayleigh-Ritz values"),
        (ax_a.plot(dd, [0.5 if d == 1 else L(d, 400_001) for d in dd], "--", color="k", lw=1.0)[0],
         r"lower bound $L_d$"),
        (ax_a.plot(dd, [1 / (2 * d) for d in dd], "-.", color=INK2, lw=0.9)[0], r"lower bound $1/(2d)$"),
    ]
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.set_xlabel("dimension $d$"); ax_a.set_ylabel(r"$\kappa_d^2$")
    ax_a.set_ylim(0.004, 6)
    panel_label(ax_a, "(a) bounds of the constant")
    ax_l.axis("off")
    ax_l.legend([c[0] for c in curves], [c[1] for c in curves], loc="center left", fontsize=7, handlelength=2.6,
                borderaxespad=0.0)
    # (b), (c)
    rows = [r for r in load_jsonl(os.path.join(RES, "eval_confirm.jsonl")) if r["set"] == "fresh"]
    DS = [2, 3, 5, 10, 20]
    off = {("laplace", "pinn"): -0.06, ("laplace", "ritz"): -0.02, ("poisson", "pinn"): 0.02, ("poisson", "ritz"): 0.06}
    for (p, m), o in off.items():
        sel = [r for r in rows if r["family"] == "main" and (r["problem"], r["method"]) == (p, m)]
        for ax, fld in ((ax_b, "eta"), (ax_c, "share_bd")):
            face = [C[m] if r["confirmatory"] else "white" for r in sel]
            ax.scatter([r["d"] * math.exp(o) for r in sel], [r[fld] for r in sel], marker=MK[p], s=11,
                       facecolors=face, edgecolors=C[m], linewidths=0.7, zorder=3)
            means = [sum(r[fld] for r in sel if r["d"] == d) / sum(1 for r in sel if r["d"] == d) for d in DS]
            ax.plot([d * math.exp(o) for d in DS], means, LS[p], color=C[m], lw=1.0)
    for r in rows:
        if r["family"] in ("long", "equaltime"):
            for ax, fld in ((ax_b, "eta"), (ax_c, "share_bd")):
                ax.scatter([r["d"] * 1.17], [r[fld]], marker="^" if r["family"] == "long" else "v", s=14,
                           facecolors="none", edgecolors=C[r["method"]], linewidths=0.8, zorder=3)
        elif r["family"] == "sweep":
            for ax, fld in ((ax_b, "eta"), (ax_c, "share_bd")):
                ax.scatter([r["d"] * 0.86], [r[fld]], marker=MK[r["problem"]], s=9, facecolors="none",
                           edgecolors=GREY, linewidths=0.7, zorder=3)
    ax_b.axhline(3, color=INK2, lw=0.8, ls=":")
    ax_b.axhline(1, color=INK2, lw=0.8)
    ax_b.set_yscale("log"); ax_b.set_ylim(0.9, 30)
    ax_b.set_yticks([1, 2, 3, 5, 10, 20]); ax_b.set_yticklabels(["1", "2", "3", "5", "10", "20"])
    ax_b.minorticks_off()
    ax_c.set_ylim(0, 1.02)
    for ax in (ax_b, ax_c):
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(DS); ax.set_xticklabels([str(d) for d in DS])
        ax.set_xlim(1.6, 25)
        ax.set_xlabel("dimension $d$")
    ax_b.set_ylabel(r"$\eta=(B_{\rm int}+B_{\rm bd})\,/\,\|v-u^*\|$")
    ax_c.set_ylabel(r"$B_{\rm bd}/(B_{\rm int}+B_{\rm bd})$")
    panel_label(ax_b, "(b) efficiency of the bound")
    panel_label(ax_c, "(c) boundary share of the bound")
    h = [Line2D([], [], color=C[m], ls=LS[p], marker=MK[p], ms=3.5, lw=1.0, label=f"{PN[p]}, {MN[m]}")
         for p in ("laplace", "poisson") for m in ("pinn", "ritz")]
    h += [Line2D([], [], marker="^", ls="", mfc="none", mec="k", ms=4, label="16 000 iterations, $d=10$"),
          Line2D([], [], marker="v", ls="", mfc="none", mec="k", ms=4, label="Deep Ritz at the PINN's CPU time, $d=10$"),
          Line2D([], [], marker="o", ls="", mfc="none", mec=GREY, ms=3.5, label="weight sweep, $d=5$ (16 networks)"),
          Line2D([], [], marker="o", ls="", mfc="white", mec="k", ms=3.5, label="open: non-confirmatory (pilot)"),
          Line2D([], [], color=INK2, ls=":", lw=0.8, label="pre-registered limit 3 (for $d\\geq5$)")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.0), handlelength=2.4,
               columnspacing=1.2)
    out = os.path.join(ART, "figures", "s6_bound.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("written figures/s6_bound.pdf")


def main():
    blocks = {"s6_highdim:tab:kappa": table_kappa(), "s6_highdim:tab:eta": table_eta(),
              "s6_highdim:tab:biasexact": table_bias()}
    gen = "\n".join(f"% {name}\n" + "\n".join(r) for name, r in blocks.items()) + "\n"
    num = numbers()
    ok = all(x["formulas_match_table"] for x in num["kappa_constants"]) and num["n_networks_bound_valid"]["all_valid"]
    if CHECK:
        for name, r in blocks.items():
            same = inject("s6_highdim.tex", name, "\n".join(r), check=True)
            print(f"{'ok  ' if same else 'FAIL'} {name}")
            ok &= bool(same)
        old = open(os.path.join(ART, "data", "s6_highdim_bound_tables.tex"), encoding="utf8").read()
        ok &= old == gen
        stored = json.load(open(os.path.join(ART, "data", "s6_highdim_bound_numbers.json"), encoding="utf8"))
        ok &= json.dumps(stored, sort_keys=True) == json.dumps(json.loads(json.dumps(num)), sort_keys=True)
        print("CHECK PASSED" if ok else "CHECK FAILED")
        sys.exit(0 if ok else 1)
    for name, r in blocks.items():
        inject("s6_highdim.tex", name, "\n".join(r))
    open(os.path.join(ART, "data", "s6_highdim_bound_tables.tex"), "w", encoding="utf8").write(gen)
    json.dump(num, open(os.path.join(ART, "data", "s6_highdim_bound_numbers.json"), "w", encoding="utf8"), indent=1)
    print("written data/s6_highdim_bound_tables.tex, data/s6_highdim_bound_numbers.json and the table rows")
    figure()
    if not ok:
        sys.exit("a constant does not match its formula, or the bound failed on a network")


if __name__ == "__main__":
    main()
