"""Separately written implementation of the verification (written from scratch within the same project; does not import the study's hd_core)."""
import math, time, json, os, sys
import torch, torch.nn as nn
torch.set_num_threads(1)

def u_exact(prob, x):
    d = x.shape[1]
    if prob == "laplace":
        s = torch.zeros(x.shape[0], dtype=x.dtype)
        for k in range(d // 2):
            s = s + x[:, 2 * k] * x[:, 2 * k + 1]
        return s.unsqueeze(1)
    return (torch.cos(math.pi * x).sum(1) / math.sqrt(d)).unsqueeze(1)

def f_src(prob, x):
    if prob == "laplace":
        return torch.zeros(x.shape[0], 1, dtype=x.dtype)
    return math.pi ** 2 * u_exact(prob, x)

def grad_exact(prob, x):
    d = x.shape[1]
    if prob == "laplace":
        g = torch.zeros_like(x)
        for k in range(d // 2):
            g[:, 2 * k] = x[:, 2 * k + 1]; g[:, 2 * k + 1] = x[:, 2 * k]
        return g
    return -math.pi * torch.sin(math.pi * x) / math.sqrt(d)

def interior(n, d, gen, dtype=torch.float32):
    return torch.rand(n, d, generator=gen, dtype=dtype)

def boundary(n, d, gen, dtype=torch.float32):
    x = torch.rand(n, d, generator=gen, dtype=dtype)
    idx = torch.randint(0, 2 * d, (n,), generator=gen)      # face id in 0..2d-1
    x[torch.arange(n), idx // 2] = (idx % 2).to(dtype)
    return x

def make_net(d, width=64, depth=3):
    layers, k = [], d
    for _ in range(depth):
        layers += [nn.Linear(k, width), nn.Tanh()]; k = width
    layers += [nn.Linear(k, 1)]
    return nn.Sequential(*layers)

def lap_nested(net, x):
    x = x.clone().requires_grad_(True)
    u = net(x)
    g, = torch.autograd.grad(u.sum(), x, create_graph=True)
    lap = 0.0
    for i in range(x.shape[1]):
        gi, = torch.autograd.grad(g[:, i].sum(), x, create_graph=True)
        lap = lap + gi[:, i]
    return u, g, lap.unsqueeze(1)

def loss_pinn(net, prob, xi, xb, w):
    u, g, lap = lap_nested(net, xi)
    r = lap + f_src(prob, xi)
    return (r ** 2).mean() + w * ((net(xb) - u_exact(prob, xb)) ** 2).mean()

def loss_ritz(net, prob, xi, xb, w):
    d = xi.shape[1]
    x = xi.clone().requires_grad_(True)
    u = net(x)
    g, = torch.autograd.grad(u.sum(), x, create_graph=True)
    e = (0.5 * (g ** 2).sum(1, keepdim=True) - f_src(prob, xi) * u).mean()
    return e + w * 2 * d * ((net(xb) - u_exact(prob, xb)) ** 2).mean()

def test_metrics(net, prob, d, n=50_000, seed=777_000):
    gen = torch.Generator().manual_seed(seed + d)
    x = interior(n, d, gen)
    xb = boundary(20_000, d, gen)
    xg = x.clone().requires_grad_(True)
    u = net(xg)
    gu, = torch.autograd.grad(u.sum(), xg)
    u = u.detach(); ue = u_exact(prob, x); ge = grad_exact(prob, x)
    e2 = ((u - ue) ** 2).sum()
    with torch.no_grad():
        ub = net(xb)
    ueb = u_exact(prob, xb)
    return dict(rel_l2=(e2 / (ue ** 2).sum()).sqrt().item(),
                rel_l2_centered=(e2 / ((ue - ue.mean()) ** 2).sum()).sqrt().item(),
                rel_h1=(((gu - ge) ** 2).sum() / (ge ** 2).sum()).sqrt().item(),
                bd_rel_l2=(((ub - ueb) ** 2).sum() / (ueb ** 2).sum()).sqrt().item())

def train(prob, method, d, seed, w, iters=4000, n=1024, lr=1e-3, log_every=250, milestones=None):
    torch.manual_seed(50_000 + seed)
    gen = torch.Generator().manual_seed(60_000 + seed)
    net = make_net(d)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ms = milestones or [iters // 2, 3 * iters // 4]
    lossf = loss_pinn if method == "pinn" else loss_ritz
    genv = torch.Generator().manual_seed(888_000 + d)
    xv = interior(5000, d, genv); uv = u_exact(prob, xv)
    curve = []; cpu = 0.0
    for it in range(iters):
        if it % log_every == 0:
            with torch.no_grad():
                curve.append((it, (((net(xv) - uv) ** 2).sum() / (uv ** 2).sum()).sqrt().item()))
        if it in ms:
            for gpar in opt.param_groups: gpar["lr"] *= 0.1
        c0 = time.process_time()
        xi = interior(n, d, gen); xb = boundary(n, d, gen)
        opt.zero_grad()
        L = lossf(net, prob, xi, xb, w)
        L.backward(); opt.step()
        cpu += time.process_time() - c0
    out = test_metrics(net, prob, d)
    with torch.no_grad():
        curve.append((iters, (((net(xv) - uv) ** 2).sum() / (uv ** 2).sum()).sqrt().item()))
    out.update(prob=prob, method=method, d=d, seed=seed, w=w, iters=iters, cpu_s=cpu, cpu_ms_it=1e3 * cpu / iters, curve=curve)
    return net, out

def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
