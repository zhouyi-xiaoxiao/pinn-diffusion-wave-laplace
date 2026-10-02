"""Re-check of the exact-bias table (results/exact_bias.jsonl, s3_exact_bias.py, t-range [1e-11, 80]) with the wide
t-range of src/bias_quad.py (t from 1e-18 to 400, split at the peak, epsrel 1e-12), 2e5 roots per type, all 49 cells
d in {2,3,5,10,20,50,100} x beta in {1,3,10,100,1000,1e4,1e5} (later choice, notes/VERIFICATION.md section 4).
Output: results/exact_bias_wide.jsonl, results/exact_bias_wide_compare.json."""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2common import RES, append_jsonl, load_jsonl, dump
from s3_exact_bias import one_d, ustar_norm, DS, BETAS
from bias_quad import bias_sq_wide

J = os.path.join(RES, "exact_bias_wide.jsonl")
done = {(r["d"], r["beta"]) for r in load_jsonl(J)}
for beta in BETAS:
    o = one_d(2 * beta, 200_000)
    for d in DS:
        if (d, beta) in done:
            continue
        w2, err, tpk = bias_sq_wide(o, d)
        rec = dict(d=d, beta=beta, nmax=200_000, rel_bias=math.sqrt(w2) / ustar_norm(d), quad_err_sq_rel=err / w2, peak_t=tpk)
        append_jsonl(J, rec); print(rec, flush=True)
old = {(r["d"], r["beta"]): r["rel_bias"] for r in load_jsonl(os.path.join(RES, "exact_bias.jsonl"))
       if r["nmax"] == 200_000 and r["epsrel"] == 1e-9}
new = {(r["d"], r["beta"]): r["rel_bias"] for r in load_jsonl(J)}
cmp = [dict(d=d, beta=b, old=old[(d, b)], wide=new[(d, b)], wide_over_old=new[(d, b)] / old[(d, b)]) for (d, b) in sorted(new)]
mx = max(cmp, key=lambda c: abs(c["wide_over_old"] - 1))
tab = [c for c in cmp if c["beta"] <= 1000]
mxt = max(tab, key=lambda c: abs(c["wide_over_old"] - 1))
dump("exact_bias_wide_compare.json", dict(cells=cmp, max_rel_change_all=dict(cell=[mx["d"], mx["beta"]], rel=mx["wide_over_old"] - 1),
                                          max_rel_change_beta_le_1000=dict(cell=[mxt["d"], mxt["beta"]], rel=mxt["wide_over_old"] - 1)))
for c in cmp:
    print(c["d"], c["beta"], "%.6e %.6e %.3e" % (c["old"], c["wide"], c["wide_over_old"] - 1))
