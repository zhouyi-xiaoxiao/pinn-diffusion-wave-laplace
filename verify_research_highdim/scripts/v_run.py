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

"""Separately written training runs (own implementation v_core; PINN Laplacian by nested autograd).
usage: v_run.py <worker-name>   -> results/v_runs_<name>.jsonl (resumable)"""
import sys, json, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_core as V
ROOT = _repo_path("verify_research_highdim")
W = {("laplace", "pinn"): 1000.0, ("laplace", "ritz"): 100.0, ("poisson", "pinn"): 1000.0, ("poisson", "ritz"): 1.0}
def J(tag, prob, method, d, seed, w=None, iters=4000):
    return dict(tag=tag, prob=prob, method=method, d=d, seed=seed, w=(W[(prob, method)] if w is None else w), iters=iters)
JOBS = {
 "B": [J("main", "laplace", m, 10, s) for s in (7, 8, 9) for m in ("ritz", "pinn")] +
      [J("main", "laplace", m, 2, s) for s in (7, 8, 9) for m in ("ritz", "pinn")] +
      [J("main", "laplace", m, 20, 7) for m in ("ritz", "pinn")],
 "C": [J("main", "poisson", m, 10, s) for s in (7, 8, 9) for m in ("ritz", "pinn")] +
      [J("main", "poisson", "ritz", 20, s) for s in (7, 8)] + [J("main", "poisson", "pinn", 20, 7)] +
      [J("main", "poisson", m, 2, s) for s in (7, 8, 9) for m in ("ritz", "pinn")],
 "D": [J("equaltime", "laplace", "ritz", 10, s, iters=10000) for s in (7, 8, 9)] +
      [J("equaltime", "poisson", "ritz", 10, s, iters=9000) for s in (7, 8, 9)] +
      [J("beta", "poisson", "ritz", 10, s, w=100.0) for s in (7, 8, 9)] +
      [J("beta", "poisson", "ritz", 10, s, w=10.0) for s in (7, 8, 9)] +
      [J("beta", "poisson", "ritz", 20, 7, w=100.0)] +
      [J("long", "laplace", "ritz", 10, s, iters=16000) for s in (7, 8)],
 "E": [J("robin", "laplace", "ritz", 2, 7, w=b) for b in (1.0, 10.0, 100.0, 1000.0)] +
      [J("sweep", "laplace", "ritz", 5, 7, w=b) for b in (1.0, 10.0, 100.0, 1000.0)] +
      [J("sweep", "poisson", "ritz", 5, 7, w=b) for b in (1.0, 1000.0)] +
      [J("sweep", "poisson", "pinn", 5, 7, w=b) for b in (1.0, 1000.0)] +
      [J("sweep", "laplace", "pinn", 5, 7, w=b) for b in (1.0, 1000.0)],
}
JOBS["F"] = [J("main", "laplace", "ritz", 20, s) for s in (8, 9, 10, 11)] + \
            [J("eq9k", "laplace", "ritz", 10, s, iters=9000) for s in (7, 8, 9)] + \
            [dict(J("main_fwd", "laplace", "pinn", 20, s), fwd=True) for s in (8, 9)]
JOBS["G"] = [dict(J("lam1e4", pr, "pinn", 10, s, w=1e4), fwd=True) for pr in ("laplace", "poisson") for s in (7, 8, 9)]
sys.path.insert(0, _repo_path("research_highdim/scripts"))
import hd_core as H
class Wrap:
    def __init__(self, seq): self.net = seq
    def __call__(self, x): return self.net(x)
_nested = V.loss_pinn
_fwd = lambda net, prob, xi, xb, w: H.loss_pinn(Wrap(net), prob, xi, xb, w)[0]
name = sys.argv[1]
out = f"{ROOT}/results/v_runs_{name}.jsonl"
done = set()
if os.path.exists(out):
    for l in open(out):
        r = json.loads(l); done.add((r["tag"], r["prob"], r["method"], r["d"], r["seed"], r["w"], r["iters"]))
os.makedirs(f"{ROOT}/results/ckpt", exist_ok=True)
for j in JOBS[name]:
    k = (j["tag"], j["prob"], j["method"], j["d"], j["seed"], j["w"], j["iters"])
    if k in done: continue
    V.loss_pinn = _fwd if j.get("fwd") else _nested
    t0 = time.time()
    net, res = V.train(j["prob"], j["method"], j["d"], j["seed"], j["w"], iters=j["iters"])
    res["tag"] = j["tag"]; res["wall_s"] = time.time() - t0
    torch.save(net.state_dict(), f"{ROOT}/results/ckpt/{j['tag']}_{j['prob']}_{j['method']}_d{j['d']}_s{j['seed']}_w{j['w']:g}_it{j['iters']}.pt")
    V.append(out, res)
    print(f"[{name}] {j['tag']} {j['prob']} {j['method']} d={j['d']} s={j['seed']} w={j['w']:g} it={j['iters']}: rel_l2={res['rel_l2']:.3e} h1={res['rel_h1']:.3e} bd={res['bd_rel_l2']:.3e} cpu={res['cpu_s']:.1f}s ({res['cpu_ms_it']:.2f} ms/it) wall={res['wall_s']:.0f}s", flush=True)
