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

"""Table of the exact penalty bias of P1 (Proposition 3) against
  - bound (b) of Corollary 2:          sqrt(U_d) N / (2 beta),
  - the article's bound (Prop. robin c): sqrt(1 + 1/(d pi^2)) N / beta,
  - the first-order term ||v1||/beta and Theorem 1's remainder bound K_d N beta^{-3/2},
  - the article's series values (data/s6_highdim_robin_bias.json; Monte Carlo over coordinates for d >= 5),
  - the polynomial Galerkin values (article data/s6_highdim_robin_bias.txt, largest degree per d;
    inputs/robin_bias_sparse.out),
  - the finite-difference values at d = 2 (beta ||w|| = 0.16673, 0.18279, 0.18476 for beta = 10, 100, 1000;
    article Observation sA_proofs:obs:sharp, verify_research_highdim/results/v_robin_fd.json).
Inputs: results/exact_bias.jsonl, results/exact_v1.json. Outputs: results/bias_table.json,
results/bias_table.csv, results/bias_table.tex, results/bias_galerkin_compare.csv."""
import csv, json, math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from t2common import U, RES, dump, load_jsonl

ART = _repo_path("data")
REF = _repo_path("research_highdim_bound/inputs")
rows = [r for r in load_jsonl(os.path.join(RES, "exact_bias.jsonl")) if r["nmax"] == 200_000]
low = {(r["d"], r["beta"]): r for r in load_jsonl(os.path.join(RES, "exact_bias.jsonl")) if r["nmax"] == 50_000}
v1 = {r["d"]: r for r in json.load(open(os.path.join(RES, "exact_v1.json")))["rows"]}
art = json.load(open(os.path.join(ART, "s6_highdim_robin_bias.json")))["series"]

# Galerkin values: largest p per d in the article's txt
gal = {}
for line in open(os.path.join(ART, "s6_highdim_robin_bias.txt")):
    m = re.match(r"galerkin d=\s*(\d+) p=\s*(\d+) n=\s*(\d+): (.*)", line)
    if m:
        d, p = int(m[1]), int(m[2])
        vals = {int(b): float(v) for b, v in re.findall(r"beta=(\d+): ([0-9.eE+-]+)", m[4])}
        if d not in gal or p > gal[d]["p"]:
            gal[d] = dict(p=p, n=int(m[3]), vals=vals)
sparse = []
for line in open(os.path.join(REF, "robin_bias_sparse.out")):
    m = re.match(r"d=\s*(\d+) p=\s*(\d+) r=\s*(\d+) n=\s*(\d+) (.*)", line)
    if m:
        vals = {int(b): float(v) for b, v in re.findall(r"b=(\d+): ([0-9.eE+-]+)", m[5])}
        sparse.append(dict(d=int(m[1]), p=int(m[2]), r=int(m[3]), n=int(m[4]), vals=vals))

FD = {10: 0.16673, 100: 0.18279, 1000: 0.18476}
out = []
for r in sorted(rows, key=lambda r: (r["d"], r["beta"])):
    d, b = r["d"], r["beta"]; m = d // 2
    us = r["ustar_norm"]; N = math.sqrt(4 * m / 3)
    bnd_b = math.sqrt(U(d)) * N / (2 * b)
    bnd_art = math.sqrt(1 + 1 / (d * math.pi ** 2)) * N / b
    Kd = (math.sqrt(d) + 2 / (math.pi * math.sqrt(d))) / (4 * math.sqrt(2))
    first = v1[d]["v1_norm"] / b
    row = dict(d=d, beta=b, rel_bias=r["rel_bias"], abs_bias=r["bias_abs"],
               rel_bias_trunc_diff=abs(r["rel_bias"] - low[(d, b)]["rel_bias"]) / r["rel_bias"] if (d, b) in low else None,
               rel_bound_b=bnd_b / us, ratio_bound_b=bnd_b / r["bias_abs"],
               rel_bound_article=bnd_art / us, ratio_bound_article=bnd_art / r["bias_abs"],
               rel_first_order=first / us, ratio_exact_over_first=r["bias_abs"] / first,
               thm1_lhs=abs(r["bias_abs"] - first), thm1_rhs=Kd * N * b ** -1.5,
               thm1_ok=abs(r["bias_abs"] - first) <= Kd * N * b ** -1.5)
    key = f"{d}/{b}"
    if key in art:
        row.update(article_series=art[key]["rel_l2"], article_series_mc_se=art[key].get("mc_rel_standard_error"),
                   exact_over_article_series=r["rel_bias"] / art[key]["rel_l2"])
    if d in gal and b in gal[d]["vals"]:
        row.update(galerkin_p=gal[d]["p"], galerkin=gal[d]["vals"][b], exact_over_galerkin=r["rel_bias"] / gal[d]["vals"][b])
    if d == 2 and b in FD:
        row.update(fd_beta_abs=FD[b], exact_beta_abs=b * r["bias_abs"], exact_over_fd=b * r["bias_abs"] / FD[b])
    out.append(row)
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})

comp = []
for s in sparse:
    for b, v in s["vals"].items():
        ex = [r for r in out if r["d"] == s["d"] and r["beta"] == b]
        if ex:
            comp.append(dict(source="inputs/robin_bias_sparse.out", d=s["d"], p=s["p"], r=s["r"], n=s["n"], beta=b,
                             galerkin=v, exact=ex[0]["rel_bias"], exact_over_galerkin=ex[0]["rel_bias"] / v))
limit = [dict(d=d, c_exact_beta1e5=[r for r in out if r["d"] == d and r["beta"] == 100000][0]["rel_bias"] * 100000,
              c_sine_series=v1[d]["c"], d_times_c=d * v1[d]["c"]) for d in sorted(v1)]
dump("bias_table.json", dict(rows=out, galerkin_sparse_compare=comp, limit=limit,
                             theorem1_all_ok=all(r["thm1_ok"] for r in out),
                             max_trunc_diff=max(r["rel_bias_trunc_diff"] for r in out if r["rel_bias_trunc_diff"] is not None)))
with open(os.path.join(RES, "bias_table.csv"), "w", newline="") as f:
    keys = sorted({k for r in out for k in r})
    w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(out)
with open(os.path.join(RES, "bias_galerkin_compare.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(comp[0]), lineterminator="\n"); w.writeheader(); w.writerows(comp)
print("limit", limit)
print("theorem 1 inequality in all cells:", all(r["thm1_ok"] for r in out))
# LaTeX rows for the main table
B5 = [1, 3, 10, 100, 1000]
with open(os.path.join(RES, "bias_table.tex"), "w") as f:
    f.write("% generated by scripts/s6_bias_table.py from results/exact_bias.jsonl; relative bias ||u_beta-u*||/||u*|| (exact sums) "
            "for beta = 1, 3, 10, 100, 1000; then bound(b)/exact and article bound/exact at beta = 1 and beta = 100\n")
    for d in [2, 3, 5, 10, 20, 50, 100]:
        cells = {r["beta"]: r for r in out if r["d"] == d}
        f.write(f"{d} & " + " & ".join(f"{cells[b]['rel_bias']:.3e}" for b in B5)
                + f" & {cells[1]['ratio_bound_b']:.2f} & {cells[100]['ratio_bound_b']:.2f} & {cells[1]['ratio_bound_article']:.1f} & {cells[100]['ratio_bound_article']:.1f} \\\\\n")
