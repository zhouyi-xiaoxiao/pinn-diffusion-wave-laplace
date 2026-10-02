"""Convergence test of ||v1|| (Dirichlet heat-kernel formula of c2_bias.py) at large d: vary truncation and grid."""
import math, sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_bias as C
out = []
for nroot, ns, slo in [(100000, 4000, -40.0), (400000, 8000, -45.0), (400000, 16000, -45.0)]:
    mu, Dp, X, M = C.dirichlet_1d(nroot)
    for d in [100, 1000, 10000, 100000]:
        val, a, b = C.heat_integral(mu, Dp, X, M, d, s_lo=slo, ns=ns)
        v = math.sqrt(val / 4)
        rec = dict(nroot=nroot, ns=ns, s_lo=slo, d=d, v1=v, d_c=d * v / C.unorm_P1(d))
        out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(C.OUT, "c2b_v1_large_d.json"), "w"), indent=1)
