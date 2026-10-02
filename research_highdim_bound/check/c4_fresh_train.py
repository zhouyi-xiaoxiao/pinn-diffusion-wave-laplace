"""Fresh-seed networks (trained here with a trainer written from the study's description, not hd_core.train),
then the bound of Corollary 2(c) evaluated with the separately written evaluator of c3_eval.py on new Monte-Carlo points.
Problems: P2 (as in the study) and P3, a problem NOT in the study:
  P3: u* = (1/m) sum_{p<=m} exp(pi (x_{2p-1} - 1)) sin(pi x_{2p}),  harmonic (f = 0), m = floor(d/2).
Methods: PINN  mean(Lap u + f)^2 + lambda mean(u - g)^2, lambda = 1000;
         Deep Ritz  mean(|grad u|^2/2 - f u) + beta * 2d * mean(u - g)^2, beta = 100 (P3) or 1 (P2).
Network 3 x 64 tanh, Adam lr 1e-3, steps x0.1 at 1/2 and 3/4 of the iterations, 1024 + 1024 fresh points
per iteration, seeds 11, 12, 13 (not used by the study).
Usage: c4_fresh_train.py problem method d seed iters   -> appends to check/out/c4_fresh.jsonl
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def ustar(prob, x):
    d = x.shape[1]; m = d // 2
    if prob == "p3":
        return sum(torch.exp(math.pi * (x[:, 2 * p] - 1)) * torch.sin(math.pi * x[:, 2 * p + 1]) for p in range(m)) / m
    if prob == "poisson":
        return torch.cos(math.pi * x).sum(1) / math.sqrt(d)
    raise ValueError


def fsrc(prob, x):
    return math.pi ** 2 * ustar(prob, x) if prob == "poisson" else torch.zeros(x.shape[0], dtype=x.dtype)


def bd_points(n, d, g):
    x = torch.rand(n, d, generator=g)
    face = torch.randint(0, d, (n,), generator=g); side = torch.randint(0, 2, (n,), generator=g).float()
    x[torch.arange(n), face] = side
    return x


def lap(u, x):
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    return sum(torch.autograd.grad(g[:, i].sum(), x, create_graph=True)[0][:, i] for i in range(x.shape[1])), g


def main():
    prob, meth, d, seed, iters = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    w = 1000.0 if meth == "pinn" else (100.0 if prob == "p3" else 1.0)
    torch.manual_seed(seed); g = torch.Generator().manual_seed(77_000 + seed)
    layers, k = [], d
    for _ in range(3):
        layers += [nn.Linear(k, 64), nn.Tanh()]; k = 64
    net = nn.Sequential(*layers, nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, [iters // 2, 3 * iters // 4], 0.1)
    c0 = time.process_time()
    for it in range(iters):
        xi = torch.rand(1024, d, generator=g).requires_grad_(True); xb = bd_points(1024, d, g)
        u = net(xi).squeeze(1)
        lb = ((net(xb).squeeze(1) - ustar(prob, xb)) ** 2).mean()
        if meth == "pinn":
            L, _ = lap(u, xi)
            loss = ((L + fsrc(prob, xi.detach())) ** 2).mean() + w * lb
        else:
            gr = torch.autograd.grad(u.sum(), xi, create_graph=True)[0]
            loss = (0.5 * (gr ** 2).sum(1) - fsrc(prob, xi.detach()) * u).mean() + w * 2 * d * lb
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    cpu = time.process_time() - c0
    sd = {f"net.{i}.{n}": p.detach().clone() for i, mod in enumerate(net) if isinstance(mod, nn.Linear) for n, p in mod.named_parameters()}
    # evaluation with the separately written evaluator of c3 (problem functions local to this file for p3)
    import c3_eval as E
    f = E.make_net(sd)
    from torch.func import hessian, vmap
    xi, xb = E.pts(d)
    lap_fn = vmap(lambda x: torch.diagonal(hessian(f)(x)).sum()); val_fn = vmap(f)
    e2 = []; r2 = []; u2 = []; s2 = []
    for xc in torch.split(xi, 2000):
        with torch.no_grad():
            uu = val_fn(xc); ll = lap_fn(xc)
        ue = ustar(prob, xc)
        e2.append((uu - ue) ** 2); r2.append((ll + fsrc(prob, xc)) ** 2); u2.append(ue ** 2)
    for xc in torch.split(xb, 20000):
        with torch.no_grad():
            s2.append((val_fn(xc) - ustar(prob, xc)) ** 2)
    e2 = torch.cat(e2).numpy(); r2 = torch.cat(r2).numpy(); u2 = torch.cat(u2).numpy(); s2 = torch.cat(s2).numpy()
    Er = math.sqrt(e2.mean()); Bint = math.sqrt(r2.mean()) / (d * math.pi ** 2); Bbd = math.sqrt(E.Ud(d) * 2 * d * s2.mean())
    rec = dict(problem=prob, method=meth, d=d, seed=seed, iters=iters, w=w, train_cpu_s=cpu, err=Er,
               rel=Er / math.sqrt(u2.mean()), B_int=Bint, B_bd=Bbd, bound=Bint + Bbd, eta=(Bint + Bbd) / Er,
               share=Bbd / (Bint + Bbd))
    with open(os.path.join(OUT, "c4_fresh.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
