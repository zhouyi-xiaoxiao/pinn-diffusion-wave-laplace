"""Conjecture 'd c(d) converges' (the working document of the proofs): evidence at larger d, with a truncation study (later choice, notes/VERIFICATION.md section 4).

c(d) = lim_{beta -> inf} beta ||u_beta - u*|| / ||u*|| for P1.  We evaluate beta * rel_bias(d, beta) from the exact
quadrature of Proposition 3 (src/bias_quad.py: t from 1e-18 to 400, split at the peak; the t-range of
scripts/s3_exact_bias.py, [1e-11, 80], would cut off the small-t tail and give d c(d) 3.5 per cent too
low at d = 1e4, beta = 1e9: 1.5757 against 1.6329 after extrapolation in scripts/s10c_dc_truncation.py) for beta = 1e3 ... 1e9 and
two truncations (number of 1-D Robin roots per parity type, 2e5 and 4e5), for d in {100, 300, 1000, 3000, 10000}.
Theorem 1 gives |beta ||w|| - ||v1||| <= K_d N beta^{-1/2}, so the sequence must settle as beta grows; a value is
reported as resolved only if (i) the two truncations agree to 1e-4 relative and (ii) the last two beta values agree
to 1e-3 relative.  Output: results/dc_large_d.jsonl (one record per run) and results/dc_large_d.json (summary).
With the wide t-range the truncation in nmax matters for d >= 300 (criterion (i) fails there); it is removed by
the extrapolation in scripts/s10c_dc_truncation.py. Run serially: OMP_NUM_THREADS=1 nice -n 19 python scripts/s10_dc_large_d.py"""
import json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2common import RES, append_jsonl, load_jsonl, dump
from s3_exact_bias import one_d, ustar_norm
from bias_quad import bias_sq_wide

JSONL = os.path.join(RES, "dc_large_d.jsonl")
DS = [100, 300, 1000, 3000, 10000]
BETAS = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
NMAX = [200_000, 400_000]
T0 = time.time()
done = {(r["d"], r["beta"], r["nmax"]) for r in load_jsonl(JSONL)}
for nmax in NMAX:
    for d in DS:
        for beta in BETAS:
            if (d, beta, nmax) in done:
                continue
            if time.time() - T0 > 12 * 60:
                print("time budget reached; rerun to continue", flush=True); sys.exit(0)
            c0 = time.process_time()
            w2, err, tpk = bias_sq_wide(one_d(2 * beta, nmax), d)
            rel = math.sqrt(w2) / ustar_norm(d)
            rec = dict(d=d, beta=beta, nmax=nmax, rel_bias=rel, beta_rel=beta * rel, d_beta_rel=d * beta * rel,
                       quad_err_sq_rel=err / w2, peak_t=tpk, cpu_s=time.process_time() - c0)
            append_jsonl(JSONL, rec); print(rec, flush=True)

rows = load_jsonl(JSONL)
summ = []
for d in DS:
    get = lambda b, n: next((r["d_beta_rel"] for r in rows if r["d"] == d and r["beta"] == b and r["nmax"] == n), None)
    seq = {f"{b:.0e}": [get(b, n) for n in NMAX] for b in BETAS}
    tr = [abs(v[1] / v[0] - 1) for v in seq.values() if None not in v]
    last = [get(b, NMAX[1]) for b in BETAS[-2:]]
    ok_tr = bool(tr) and max(tr) < 1e-4
    ok_beta = None not in last and abs(last[1] / last[0] - 1) < 1e-3
    summ.append(dict(d=d, d_beta_rel_by_beta=seq, max_truncation_rel_diff=max(tr) if tr else None,
                     last_two_beta_rel_diff=abs(last[1] / last[0] - 1) if None not in last else None,
                     resolved=ok_tr and ok_beta, estimate=last[1] if ok_tr and ok_beta else None))
dump("dc_large_d.json", dict(note="d c(d) estimated by d * beta * rel_bias at the largest beta; see script docstring",
                             rows=summ))
print(json.dumps(summ, indent=1))
