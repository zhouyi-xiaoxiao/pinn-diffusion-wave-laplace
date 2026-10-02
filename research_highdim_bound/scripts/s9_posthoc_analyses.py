"""Analyses made after the pre-registered decisions (notes/VERIFICATION.md, section 4). Reads only existing results; no network
is evaluated again and no number of the confirmatory evaluation changes.

(1) H2: boundary shares of failures and passes; is the failure set equal to the set of the 21 smallest shares?
    How many networks are swapped, and which ones lie in the overlap of the two share ranges?
(2) Efficiency eta by dimension over all 88 networks (fresh set), and per (problem, method, d) for the main family.
(3) d = 10 budget comparison: does the bound order PINN and Deep Ritz as the error does, (a) for the seed means,
    (b) for each seed index, (c) for every one of the 3 x 3 cross pairs?
(4) Comparison of U_d with Payne's convex-domain bound C_d^2 <= 1/2 (delta_1 >= 2/width, width 1 for the cube)
    and of L_d with the elementary bound C_d^2 >= |Omega|/|dOmega| = 1/(2d) (test function h = 1).
Output: results/posthoc_analyses.json, results/table_payne.tex."""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from t2common import U, L, RES, load_jsonl, dump

rows = load_jsonl(os.path.join(RES, "eval_confirm.jsonl"))
fresh = [r for r in rows if r["set"] == "fresh"]
assert len(fresh) == 88
out = {}

# (1) H2 share overlap
h2 = [r for r in fresh if r["confirmatory"] and r["d"] >= 5]
fails = [r for r in h2 if r["eta"] > 3]
passes = [r for r in h2 if r["eta"] <= 3]
by_share = sorted(h2, key=lambda r: r["share_bd"])
smallest = {r["file"] for r in by_share[:len(fails)]}
failset = {r["file"] for r in fails}
fs = [r["share_bd"] for r in fails]; ps = [r["share_bd"] for r in passes]
lo, hi = min(ps), max(fs)
overlap = [dict(file=r["file"], share=r["share_bd"], se_share=r["se_share"], eta=r["eta"], se_eta=r["se_eta"],
                fails=r["eta"] > 3) for r in by_share if lo <= r["share_bd"] <= hi]
out["H2_share"] = dict(n_tested=len(h2), n_fail=len(fails),
                       fail_share_range=[min(fs), max(fs)], pass_share_range=[min(ps), max(ps)],
                       fail_set_equals_smallest_share_set=failset == smallest,
                       failures_not_among_smallest=sorted(failset - smallest),
                       smallest_not_failing=sorted(smallest - failset),
                       n_swapped=len(failset - smallest), overlap_networks=overlap,
                       n_tested_with_share_below_0_5=sum(r["share_bd"] < 0.5 for r in h2),
                       all_with_share_below_0_5_fail=all(r["eta"] > 3 for r in h2 if r["share_bd"] < 0.5),
                       all_with_share_above_0_6_pass=all(r["eta"] <= 3 for r in h2 if r["share_bd"] > 0.6))

# (2) eta by dimension
eta_d = {}
for d in sorted({r["d"] for r in fresh}):
    sel = [r for r in fresh if r["d"] == d]
    m = max(sel, key=lambda r: r["eta"])
    eta_d[d] = dict(n=len(sel), min=min(r["eta"] for r in sel), max=m["eta"], argmax=m["file"])
out["eta_by_d_all88"] = eta_d
main = {}
for r in fresh:
    if r["family"] != "main":
        continue
    k = f'{r["problem"]}/{r["method"]}/d{r["d"]}'
    main.setdefault(k, []).append(r["eta"])
out["eta_main_by_cell"] = {k: dict(min=min(v), max=max(v), n=len(v)) for k, v in sorted(main.items())}

# (3) d = 10 budget ordering
def grab(prob, meth, fam, iters):
    s = [r for r in fresh if r["d"] == 10 and r["problem"] == prob and r["method"] == meth and r["family"] == fam
         and r["iters"] == iters and r["seed"] in (0, 1, 2)]
    return {r["seed"]: r for r in s}
cmp = {}
for prob in ("laplace", "poisson"):
    pinn = grab(prob, "pinn", "main", 4000)
    dr4 = grab(prob, "ritz", "main", 4000)
    eq = [r for r in fresh if r["d"] == 10 and r["problem"] == prob and r["method"] == "ritz" and r["family"] == "equaltime"]
    dreq = {r["seed"]: r for r in eq}
    for name, other in (("equal_iterations_4000", dr4), ("equal_cpu", dreq)):
        assert len(pinn) == 3 and len(other) == 3, (prob, name, len(pinn), len(other))
        mean = lambda dct, key: sum(r[key] for r in dct.values()) / 3
        same_mean = (mean(pinn, "err_abs") < mean(other, "err_abs")) == (mean(pinn, "bound") < mean(other, "bound"))
        by_seed = [(pinn[s]["err_abs"] < other[s]["err_abs"]) == (pinn[s]["bound"] < other[s]["bound"]) for s in (0, 1, 2)]
        cross = [(pinn[a]["err_abs"] < other[b]["err_abs"]) == (pinn[a]["bound"] < other[b]["bound"])
                 for a in (0, 1, 2) for b in (0, 1, 2)]
        err_sep = max(r["err_abs"] for r in pinn.values()) < min(r["err_abs"] for r in other.values()) or \
            min(r["err_abs"] for r in pinn.values()) > max(r["err_abs"] for r in other.values())
        bnd_sep = max(r["bound"] for r in pinn.values()) < min(r["bound"] for r in other.values()) or \
            min(r["bound"] for r in pinn.values()) > max(r["bound"] for r in other.values())
        cmp[f"{prob}/{name}"] = dict(
            pinn_err_mean=mean(pinn, "err_abs"), other_err_mean=mean(other, "err_abs"),
            pinn_bound_range=[min(r["bound"] for r in pinn.values()), max(r["bound"] for r in pinn.values())],
            other_bound_range=[min(r["bound"] for r in other.values()), max(r["bound"] for r in other.values())],
            pinn_err_range=[min(r["err_abs"] for r in pinn.values()), max(r["err_abs"] for r in pinn.values())],
            other_err_range=[min(r["err_abs"] for r in other.values()), max(r["err_abs"] for r in other.values())],
            means_ordered_alike=same_mean, by_seed_index_ordered_alike=by_seed,
            cross_pairs_ordered_alike=sum(cross), cross_pairs_total=9,
            error_ranges_separated=err_sep, bound_ranges_separated=bnd_sep)
out["d10_ordering"] = cmp

# (4) Payne and the elementary lower bound
pay = []
for d in (1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 50, 100):
    u, l = U(d), L(d)
    pay.append(dict(d=d, U=u, payne=0.5, U_over_payne=u / 0.5, sqrt_payne_over_sqrtU=math.sqrt(0.5 / u),
                    article_over_payne=2 * math.sqrt(1 + 1 / (d * math.pi ** 2)) / math.sqrt(0.5),
                    L=l, elementary_lower=1 / (2 * d), best_proved_lower=max(l, 1 / (2 * d)),
                    L_beats_elementary=l > 1 / (2 * d),
                    delta1_lower_from_U=1 / u, delta1_upper_elementary=2 * d))
out["payne_compare"] = pay
# (5) the unit square (d = 2): d_1(k Omega) = d_1(Omega)/k, so d_1((0,1)^2) = sqrt(pi) d_1((0, sqrt(pi))^2).
# Published upper bounds for the square of area pi (Bucur-Gazzola 2011, Sec. 4): Kuttler 1972 d_1 < 1.9889...,
# Ferrero-Gazzola-Weth 2005 d_1 < 1.96256. Our bounds: d_1 = 1/C_2^2 >= 1/U_2, and every computed lower bound
# lam <= C_2^2 gives d_1 <= 1/lam (results/theorem2_checks.json A6, K = 800: 0.287459).
lam800 = max(r["lam_max"] for r in json.load(open(os.path.join(RES, "theorem2_checks.json")))["A6_lanczos"] if r["d"] == 2)
out["unit_square"] = dict(kuttler1972_upper=math.sqrt(math.pi) * 1.9889, fgw2005_upper=math.sqrt(math.pi) * 1.96256,
                          lower_from_U2=1 / U(2), upper_from_lanczos=1 / lam800, lanczos_lam=lam800,
                          note="d_1 of the unit square; the Kuttler value is truncated in the source (1.9889...)")
# (6) Step 8 of Theorem 2 (lower bound 1/(2d) from h = 1), numerically: zeta_M = projection of 1 onto the sine
# modes with max k_i <= M (only all-odd k: a_k = 2^{d/2} prod 2/(pi k_i)); ||T zeta_M||^2 from eq. (TT) of the working document of the proofs;
# check (int zeta_M)^2 / (2d ||zeta_M||^2) <= ||T zeta_M||^2/||zeta_M||^2 <= U_d and that the first tends to 1/(2d).
import itertools
import numpy as np
step8 = []
for d, Ms in ((2, (11, 101, 401)), (3, (11, 41))):
    for M in Ms:
        k1 = np.arange(1, M + 1, 2, dtype=float)
        grids = np.meshgrid(*([k1] * d), indexing="ij")
        K2 = sum(g ** 2 for g in grids)
        a = 2 ** (d / 2) * np.prod([2 / (math.pi * g) for g in grids], axis=0)
        z2 = float((a ** 2).sum()); intz = float((a * 2 ** (d / 2) * np.prod([2 / (math.pi * g) for g in grids], axis=0)).sum())
        T2 = 0.0
        for i in range(d):
            O = (a * grids[i] / K2).sum(axis=i)   # all k_i odd: E_i = 0
            T2 += 4 / math.pi ** 2 * float((O ** 2).sum())
        step8.append(dict(d=d, M=M, int_zeta=intz, norm2=z2, h1_quotient=intz ** 2 / (2 * d * z2), rayleigh=T2 / z2, U=U(d),
                          ok=intz ** 2 / (2 * d * z2) <= T2 / z2 * (1 + 1e-12) <= U(d)))
out["step8_check"] = step8
dump("posthoc_analyses.json", out)
with open(os.path.join(RES, "table_payne.tex"), "w") as f:
    f.write("% generated by scripts/s9_posthoc_analyses.py\n")
    for p in pay:
        if p["d"] in (1, 2, 3, 5, 10, 20, 50, 100):
            f.write(f'{p["d"]} & {p["elementary_lower"]:.4f} & {p["L"]:.4f} & {p["U"]:.4f} & 0.5000 & '
                    f'{p["sqrt_payne_over_sqrtU"]:.3f} & {p["delta1_lower_from_U"]:.3f} \\\\\n')
print(json.dumps({k: v for k, v in out.items() if k != "eta_main_by_cell"}, indent=1, default=str)[:6000])
