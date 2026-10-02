"""Quadrature sensitivity of the d = 2 extension-side Rayleigh-Ritz value (r2_thm2_edges.py)."""
import json, sys
sys.argv = [sys.argv[0], "/dev/null"]
src = open(__file__.replace("r2_rr_quadrature.py", "r2_thm2_edges.py")).read()
pre = src.split("out = {}")[0]
fn = "def rr_d2" + src.split("def rr_d2")[1].split("rr = {}")[0]
ns = {}; exec(pre, ns); exec(fn, ns)
res = {f"K20_nq{nq}": ns["rr_d2"](20, nq) for nq in (200, 400, 800)}
print(res)
json.dump(res, open(__file__.replace("r2_rr_quadrature.py", "out_r2/r2_rr_quadrature.json"), "w"), indent=1)
