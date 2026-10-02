"""Truncation extrapolation for d c(d) (later choice, notes/VERIFICATION.md section 4). With the wide t-range the result depends on the number
nmax of 1-D Robin roots per type: the modes beyond nmax are missing for t < 1/mu_max ~ (2 pi nmax)^-2, where the
integrand in log t behaves like t^{1/2}, so the relative truncation error is proportional to 1/nmax. We compute
d * beta * rel_bias at beta = 1e8, 1e9 for nmax = 2e5, 4e5, 8e5 and Richardson-extrapolate in 1/nmax:
R1 = 2 v(4e5) - v(2e5), R2 = 2 v(8e5) - v(4e5); the agreement of R1 and R2 tests the 1/nmax model.
Output: results/dc_truncation.jsonl, results/dc_truncation.json."""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2common import RES, append_jsonl, load_jsonl, dump
from s3_exact_bias import one_d, ustar_norm
from bias_quad import bias_sq_wide

J = os.path.join(RES, "dc_truncation.jsonl")
DS = [2, 5, 10, 20, 50, 100, 300, 1000, 3000, 10000]
done = {(r["d"], r["beta"], r["nmax"]) for r in load_jsonl(J)}
for beta in (1e8, 1e9):
    for nmax in (200_000, 400_000, 800_000):
        o = one_d(2 * beta, nmax)
        for d in DS:
            if (d, beta, nmax) in done:
                continue
            w2, err, tpk = bias_sq_wide(o, d)
            rec = dict(d=d, beta=beta, nmax=nmax, d_beta_rel=d * beta * math.sqrt(w2) / ustar_norm(d), quad_err_sq_rel=err / w2)
            append_jsonl(J, rec); print(rec, flush=True)
rows = load_jsonl(J)
out = []
for beta in (1e8, 1e9):
    for d in DS:
        v = {r["nmax"]: r["d_beta_rel"] for r in rows if r["d"] == d and r["beta"] == beta}
        R1 = 2 * v[400_000] - v[200_000]; R2 = 2 * v[800_000] - v[400_000]
        out.append(dict(d=d, beta=beta, v2e5=v[200_000], v4e5=v[400_000], v8e5=v[800_000], R1=R1, R2=R2,
                        R_rel_diff=abs(R2 / R1 - 1), d_c_estimate=R2))
dump("dc_truncation.json", out)
for r in out:
    print(r["d"], "%.0e" % r["beta"], "%.6f %.6f %.6f | R1 %.6f R2 %.6f diff %.1e" % (r["v2e5"], r["v4e5"], r["v8e5"], r["R1"], r["R2"], r["R_rel_diff"]))
