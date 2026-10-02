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

"""Re-evaluate the study's saved final weights (results/ckpt/*.pt) on a separately drawn held-out MC set (different generator/seed,
own exact-solution code) and compare with the rel_l2 stored in the study's run records."""
import sys, os, re, glob, json, torch, statistics as st, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_core as V
T = _repo_path("research_highdim/results")
rec = {}
for f in glob.glob(T + "/runs*.jsonl"):
    for l in open(f):
        if l.strip():
            r = json.loads(l); rec[(r["phase"], r["problem"], r["method"], r["d"], r["seed"], float(r["w"]), r["iters"])] = r
rows = []
for f in sorted(glob.glob(T + "/ckpt/*.pt")):
    m = re.match(r"(\w+?)_(laplace|poisson)_(pinn|ritz)_d(\d+)_s(\d+)_w([\d.e+]+)_it(\d+)\.pt", os.path.basename(f))
    ph, p, me, d, s, w, it = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5)), float(m.group(6)), int(m.group(7))
    sd = torch.load(f, map_location="cpu")
    net = V.make_net(d)
    net.load_state_dict({k.replace("net.", "", 1): v for k, v in sd.items()})
    recheck = V.test_metrics(net, p, d)
    r = rec[(ph, p, me, d, s, w, it)]
    rows.append(dict(phase=ph, problem=p, method=me, d=d, seed=s, w=w, iters=it, study=r["rel_l2"], recheck=recheck["rel_l2"],
                     study_h1=r["rel_h1semi"], recheck_h1=recheck["rel_h1"], study_c=r["rel_l2_centered"], recheck_c=recheck["rel_l2_centered"]))
json.dump(rows, open(_repo_path("verify_research_highdim/results/v_reeval.json"), "w"), indent=1)
dev = [abs(r["recheck"] / r["study"] - 1) for r in rows]; devh = [abs(r["recheck_h1"] / r["study_h1"] - 1) for r in rows]
print(f"{len(rows)} checkpoints; rel_l2 |recheck/study-1|: max {max(dev):.3f} median {st.median(dev):.4f}; H1: max {max(devh):.3f}")
worst = sorted(rows, key=lambda r: -abs(r["recheck"] / r["study"] - 1))[:5]
for r in worst: print("  worst:", r["phase"], r["problem"], r["method"], r["d"], r["seed"], f"study {r['study']:.4e} recheck {r['recheck']:.4e}")
G = collections.defaultdict(list)
for r in rows:
    if r["phase"] == "main": G[(r["problem"], r["d"], r["method"])].append(r["recheck"])
print("main cells on the re-check test set: mean (study mean) ; Ritz/PINN")
T2 = collections.defaultdict(list)
for r in rows:
    if r["phase"] == "main": T2[(r["problem"], r["d"], r["method"])].append(r["study"])
for p in ["laplace", "poisson"]:
    for d in [2, 3, 5, 10, 20]:
        a, b = G[(p, d, "pinn")], G[(p, d, "ritz")]
        print(f"  {p} d={d}: PINN {st.mean(a):.3e} ({st.mean(T2[(p,d,'pinn')]):.3e})  Ritz {st.mean(b):.3e} ({st.mean(T2[(p,d,'ritz')]):.3e})  ratio {st.mean(b)/st.mean(a):.2f}  PINN all below Ritz: {max(a)<min(b)}")
for ph in ["long", "equaltime"]:
    E = collections.defaultdict(list)
    for r in rows:
        if r["phase"] == ph: E[(r["problem"], r["method"], r["iters"])].append(r["recheck"])
    for k, v in E.items(): print(" ", ph, k, f"mean {st.mean(v):.3e} sd {st.stdev(v):.2e}", [f"{x:.3e}" for x in v])
