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

"""Separately written evaluation of the bound of Corollary 2(c) on all 88 saved networks, with NEW Monte-Carlo
points (numpy generator, seeds 4_242_000 + d interior, 5_353_000 + d boundary), a tanh-MLP forward pass written
here from the state_dict, and the Laplacian as the trace of the Hessian from torch.func (vmap(hessian)),
not the forward-Laplacian of hd_core. Problems written from their description:
  P1 u* = sum_{p<=floor(d/2)} x_{2p-1} x_{2p}, f = 0;  P2 u* = d^{-1/2} sum cos(pi x_i), f = pi^2 u*.
Output: check/out/c3_eval.jsonl (one record per network).
"""
import json, math, os, re, sys, time
import numpy as np
import torch
from torch.func import hessian, vmap

torch.set_num_threads(1)
CK = _repo_path("research_highdim/results/ckpt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
PAT = re.compile(r"(main|long|equaltime|sweep)_(laplace|poisson)_(pinn|ritz)_d(\d+)_s(\d+)_w(\d+)_it(\d+)\.pt")
NI = int(os.environ.get("NI", "200000")); NB = int(os.environ.get("NB", "200000"))


def Ud(d):
    if d == 1:
        return 0.5
    a = math.sqrt(d - 1)
    return math.tanh(math.pi * a / 2) / (math.pi * a)


def ustar(prob, x):
    d = x.shape[1]
    if prob == "laplace":
        m = d // 2
        return sum(x[:, 2 * p] * x[:, 2 * p + 1] for p in range(m))
    return torch.cos(math.pi * x).sum(1) / math.sqrt(d)


def fsrc(prob, x):
    return torch.zeros(x.shape[0], dtype=x.dtype) if prob == "laplace" else math.pi ** 2 * ustar(prob, x)


def make_net(sd):
    Ws = [sd[f"net.{i}.weight"].double() for i in (0, 2, 4, 6)]
    bs = [sd[f"net.{i}.bias"].double() for i in (0, 2, 4, 6)]
    def f(x):  # x: (d,) -> scalar
        h = x
        for W, b in zip(Ws[:-1], bs[:-1]):
            h = torch.tanh(W @ h + b)
        return (Ws[-1] @ h + bs[-1])[0]
    return f


def pts(d):
    ri = np.random.default_rng(4_242_000 + d); rb = np.random.default_rng(5_353_000 + d)
    xi = ri.random((NI, d))
    xb = rb.random((NB, d))
    face = rb.integers(0, d, NB); side = rb.integers(0, 2, NB).astype(float)
    xb[np.arange(NB), face] = side
    return torch.from_numpy(xi), torch.from_numpy(xb)


def main():
    path = os.path.join(OUT, "c3_eval.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["file"] for l in open(path)}
    files = sorted(f for f in os.listdir(CK) if PAT.fullmatch(f))
    files.sort(key=lambda f: int(PAT.fullmatch(f)[4]))
    cache = {}
    for f in files:
        if f in done:
            continue
        fam, prob, meth, d, seed, w, it = PAT.fullmatch(f).groups(); d = int(d)
        if d not in cache:
            cache = {d: pts(d)}
        xi, xb = cache[d]
        sd = torch.load(os.path.join(CK, f), map_location="cpu")
        net = make_net(sd)
        c0 = time.process_time()
        lap_fn = vmap(lambda x: torch.diagonal(hessian(net)(x)).sum())
        val_fn = vmap(net)
        e2, r2, s2, u2 = [], [], [], []
        for xc in torch.split(xi, 2000):
            with torch.no_grad():
                u = val_fn(xc); lap = lap_fn(xc)
            ue = ustar(prob, xc)
            e2.append((u - ue) ** 2); r2.append((lap + fsrc(prob, xc)) ** 2); u2.append(ue ** 2)
        for xc in torch.split(xb, 20000):
            with torch.no_grad():
                s2.append((val_fn(xc) - ustar(prob, xc)) ** 2)
        e2 = torch.cat(e2).numpy(); r2 = torch.cat(r2).numpy(); s2 = torch.cat(s2).numpy(); u2 = torch.cat(u2).numpy()
        E = math.sqrt(e2.mean()); R = math.sqrt(r2.mean()); Sb = math.sqrt(2 * d * s2.mean())
        Bint = R / (d * math.pi ** 2); Bbd = math.sqrt(Ud(d)) * Sb; bound = Bint + Bbd
        # simple s.e. by delta method, independent samples
        seE = e2.std() / math.sqrt(NI) / (2 * E)
        seB = math.sqrt((r2.std() / math.sqrt(NI) / (2 * R) / (d * math.pi ** 2)) ** 2 +
                        (math.sqrt(Ud(d) * 2 * d) * s2.std() / math.sqrt(NB) / (2 * math.sqrt(s2.mean()))) ** 2)
        rec = dict(file=f, family=fam, problem=prob, method=meth, d=d, seed=int(seed), w=int(w), iters=int(it),
                   err=E, rel=E / math.sqrt(u2.mean()), B_int=Bint, B_bd=Bbd, bound=bound, eta=bound / E,
                   share=Bbd / bound, se_err=seE, se_bound=seB, z=(bound - E) / math.sqrt(seE ** 2 + seB ** 2),
                   cpu_s=time.process_time() - c0)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{f:48s} rel {rec['rel']:.3e} eta {rec['eta']:.3f} share {rec['share']:.3f} z {rec['z']:.0f} cpu {rec['cpu_s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
