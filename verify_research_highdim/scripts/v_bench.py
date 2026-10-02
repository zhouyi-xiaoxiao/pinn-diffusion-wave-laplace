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

"""Own cost benchmark: CPU ms / iteration (1 thread), Ritz (re-check code) vs PINN nested autograd (re-check code) vs PINN forward-Laplacian (study's function)."""
import sys, os, time, json, statistics as st, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_core as V
sys.path.insert(0, _repo_path("research_highdim/scripts"))
import hd_core as H
def step_fn(kind, d):
    torch.manual_seed(0); gen = torch.Generator().manual_seed(1)
    if kind == "fwd":
        net = H.MLP(d); lossf = lambda xi, xb: H.loss_pinn(net, "poisson", xi, xb, 1000.0)[0]
    elif kind == "nested":
        net = V.make_net(d); lossf = lambda xi, xb: V.loss_pinn(net, "poisson", xi, xb, 1000.0)
    else:
        net = V.make_net(d); lossf = lambda xi, xb: V.loss_ritz(net, "poisson", xi, xb, 1.0)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    def step():
        xi = V.interior(1024, d, gen); xb = V.boundary(1024, d, gen)
        opt.zero_grad(); L = lossf(xi, xb); L.backward(); opt.step()
    return step
res = {}
for d in [2, 5, 10, 20, 50, 100]:
    fns = {k: step_fn(k, d) for k in ("ritz", "fwd", "nested")}
    for f in fns.values():
        for _ in range(3): f()
    T = {k: [] for k in fns}; W = {k: [] for k in fns}
    nit = 10 if d <= 20 else 5
    for rnd in range(7):
        for k, f in fns.items():
            c0 = time.process_time(); w0 = time.perf_counter()
            for _ in range(nit): f()
            T[k].append(1e3 * (time.process_time() - c0) / nit); W[k].append(1e3 * (time.perf_counter() - w0) / nit)
    r = {k: st.median(v) for k, v in T.items()}
    rat_f = [a / b for a, b in zip(T["fwd"], T["ritz"])]; rat_n = [a / b for a, b in zip(T["nested"], T["ritz"])]
    res[d] = dict(cpu_ms=r, ratio_fwd=[min(rat_f), st.median(rat_f), max(rat_f)], ratio_nested=[min(rat_n), st.median(rat_n), max(rat_n)],
                  wall_ms={k: st.median(v) for k, v in W.items()})
    print(f"d={d:3d}: ritz {r['ritz']:.2f}  fwd {r['fwd']:.2f} (x{st.median(rat_f):.2f} [{min(rat_f):.2f},{max(rat_f):.2f}])  nested {r['nested']:.2f} (x{st.median(rat_n):.2f} [{min(rat_n):.2f},{max(rat_n):.2f}])  nested/fwd {r['nested']/r['fwd']:.2f}", flush=True)
    json.dump(res, open(_repo_path("verify_research_highdim/results/v_bench.json"), "w"), indent=1)
