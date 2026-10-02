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

import sys, json, glob, time
sys.path.insert(0, _repo_path("research_highdim/scripts"))
import hd_core as H
T = _repo_path("research_highdim/results")
rec = {}
for f in glob.glob(T + "/runs*.jsonl"):
    for l in open(f):
        if l.strip():
            r = json.loads(l); rec[(r["phase"], r["problem"], r["method"], r["d"], r["seed"], float(r["w"]), r["iters"])] = r
for c in [("long", "laplace", "pinn", 10, 0, 1000.0, 16000)]:
    ph, p, m, d, s, w, it = c
    t0 = time.time(); net, final, curve = H.train(p, m, d, s, w, iters=it); r = rec[c]
    line = f"{c}: stored rel_l2 {r['rel_l2']:.9e}  rerun {final['rel_l2']:.9e}  reldiff {abs(final['rel_l2']/r['rel_l2']-1):.2e}  cpu {final['cpu_time_s']:.1f}s (stored {r.get('cpu_time_s', float('nan')):.1f}s) wall {time.time()-t0:.0f}s"
    print(line, flush=True)
    open(_repo_path("verify_research_highdim/results/v_determinism.txt"), "a").write(line + "\n")
