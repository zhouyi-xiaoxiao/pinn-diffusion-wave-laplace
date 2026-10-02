"""Round-2 check: separately written trainer + evaluator for the computable bound
   ||v-u*|| <= ||Lap v + f||/(d pi^2) + sqrt(2 d U_d) sqrt(L_bd)   (Corollary (c), loss form).
Fresh seeds 31-33; cells not used in the first re-check: P1 d=20 (incl. Deep Ritz at the CPU time of
the PINN), P2 d=2, and a new non-separable Poisson problem P4 u* = sin(pi sum_i x_i / sqrt(d)), d=5.
Architecture/optimiser follow the study's description (tanh MLP 3x64, Adam 1e-3, steps at 1/2 and 3/4,
1024+1024 fresh points per iteration, PINN lambda=1000, Deep Ritz beta=100 (P1) / 1 (P2, P4)).
Laplacian: own forward propagation of (value, gradient, Laplacian) through the layers; checked against
autograd Hessian traces at evaluation."""
import json, math, sys, time
import torch
torch.set_num_threads(1)

def U(d):
    if d == 1: return 0.5
    a = math.sqrt(d - 1); return math.tanh(math.pi * a / 2) / (math.pi * a)

def u_exact(prob, x):
    d = x.shape[1]
    if prob == "P1":
        m = d // 2; return (x[:, :2*m:2] * x[:, 1:2*m:2]).sum(1)
    if prob == "P2":
        return torch.cos(math.pi * x).sum(1) / math.sqrt(d)
    if prob == "P4":
        return torch.sin(math.pi * x.sum(1) / math.sqrt(d))
def f_src(prob, x):
    if prob == "P1": return torch.zeros(x.shape[0], dtype=x.dtype)
    return math.pi ** 2 * u_exact(prob, x)   # both P2 and P4 satisfy -Lap u = pi^2 u

class Net(torch.nn.Module):
    def __init__(s, d, w=64, depth=3):
        super().__init__()
        dims = [d] + [w] * depth
        s.hidden = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        s.out = torch.nn.Linear(w, 1)
    def forward(s, x):
        for L in s.hidden: x = torch.tanh(L(x))
        return s.out(x).squeeze(1)
    def val_lap(s, x):
        # z: (n,w) values; G: (n,d,w) gradient wrt x; Q: (n,w) Laplacian
        z = x; G = None; Q = None
        for L in s.hidden:
            W = L.weight  # (out,in)
            a = z @ W.T + L.bias
            Ga = (torch.eye(x.shape[1], dtype=x.dtype).expand(x.shape[0], -1, -1) if G is None else G) @ W.T
            Qa = torch.zeros_like(a) if Q is None else Q @ W.T
            t = torch.tanh(a); s1 = 1 - t * t
            Q = s1 * Qa - 2 * t * s1 * (Ga * Ga).sum(1)
            G = Ga * s1.unsqueeze(1); z = t
        W = s.out.weight
        return (z @ W.T).squeeze(1) + s.out.bias, (Q @ W.T).squeeze(1)

def sample_bd(n, d, g, dtype=torch.float32):
    x = torch.rand(n, d, generator=g, dtype=dtype)
    face = torch.randint(0, d, (n,), generator=g); side = torch.randint(0, 2, (n,), generator=g).to(dtype)
    x[torch.arange(n), face] = side
    return x

def train(prob, method, d, seed, w, iters=None, cpu_budget=None):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(7_000 + seed)
    net = Net(d); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n_it = iters
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, [n_it // 2, 3 * n_it // 4], 0.1)
    c0 = time.process_time()
    for it in range(n_it):
        xi = torch.rand(1024, d, generator=g); xb = sample_bd(1024, d, g)
        opt.zero_grad()
        if method == "pinn":
            _, lap = net.val_lap(xi)
            loss = ((lap + f_src(prob, xi)) ** 2).mean() + w * ((net(xb) - u_exact(prob, xb)) ** 2).mean()
        else:
            xi.requires_grad_(True); u = net(xi)
            gr = torch.autograd.grad(u.sum(), xi, create_graph=True)[0]
            loss = (0.5 * (gr ** 2).sum(1) - f_src(prob, xi.detach()) * u).mean() + w * 2 * d * ((net(xb) - u_exact(prob, xb)) ** 2).mean()
        loss.backward(); opt.step(); sch.step()
    return net, time.process_time() - c0

def evaluate(net, prob, d, n=200_000):
    net = net.double()
    gi = torch.Generator().manual_seed(90_000 + d); gb = torch.Generator().manual_seed(95_000 + d)
    xi = torch.rand(n, d, generator=gi, dtype=torch.float64); xb = sample_bd(n, d, gb, torch.float64)
    with torch.no_grad():
        vals, laps = [], []
        for c in torch.split(xi, 20_000):
            v, l = net.val_lap(c); vals.append(v); laps.append(l)
        v = torch.cat(vals); lap = torch.cat(laps)
        e2 = (v - u_exact(prob, xi)) ** 2; r2 = (lap + f_src(prob, xi)) ** 2
        b2 = (net(xb) - u_exact(prob, xb)) ** 2; uu = u_exact(prob, xi) ** 2
    # autograd Hessian-trace check on 16 points
    xc = xi[:16].clone().requires_grad_(True)
    gr = torch.autograd.grad(net(xc).sum(), xc, create_graph=True)[0]
    ht = sum(torch.autograd.grad(gr[:, i].sum(), xc, retain_graph=True)[0][:, i] for i in range(d))
    lap_err = float(((ht - lap[:16]).abs().max() / lap[:16].abs().max()).item())
    m = lambda t: float(t.mean()); se = lambda t: float(t.std() / math.sqrt(len(t)))
    err = math.sqrt(m(e2)); Bi = math.sqrt(m(r2)) / (d * math.pi ** 2); Bb = math.sqrt(2 * d * U(d)) * math.sqrt(m(b2))
    se_err = se(e2) / (2 * err); se_bound = se(r2) / (2 * math.sqrt(m(r2))) / (d * math.pi ** 2) + math.sqrt(2 * d * U(d)) * se(b2) / (2 * math.sqrt(m(b2)))
    return dict(err=err, rel_err=err / math.sqrt(m(uu)), B_int=Bi, B_bd=Bb, bound=Bi + Bb, eta=(Bi + Bb) / err,
                share=Bb / (Bi + Bb), z=(Bi + Bb - err) / math.sqrt(se_err ** 2 + se_bound ** 2), lap_check_rel=lap_err)

if __name__ == "__main__":
    out = sys.argv[1]
    cells = [("P1", 20, "pinn", 1000.0), ("P1", 20, "ritz", 100.0), ("P2", 2, "pinn", 1000.0), ("P2", 2, "ritz", 1.0),
             ("P4", 5, "pinn", 1000.0), ("P4", 5, "ritz", 1.0)]
    done = set()
    try:
        for line in open(out):
            r = json.loads(line); done.add((r["prob"], r["d"], r["method"], r["seed"], r["iters"]))
    except FileNotFoundError: pass
    pinn_cpu = {}
    for seed in (31, 32, 33):
        for prob, d, method, w in cells:
            key = (prob, d, method, seed, 4000)
            if key in done: continue
            net, cpu = train(prob, method, d, seed, w, iters=4000)
            rec = dict(prob=prob, d=d, method=method, seed=seed, w=w, iters=4000, cpu_s=cpu, **evaluate(net, prob, d))
            open(out, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
    # equal CPU time: Deep Ritz on P1 d=20 for the CPU time of the PINN of the same seed
    rows = [json.loads(l) for l in open(out)]
    for seed in (31, 32, 33):
        p = [r for r in rows if r["prob"] == "P1" and r["method"] == "pinn" and r["seed"] == seed and r["iters"] == 4000][0]
        q = [r for r in rows if r["prob"] == "P1" and r["method"] == "ritz" and r["seed"] == seed and r["iters"] == 4000][0]
        it = int(round(4000 * p["cpu_s"] / q["cpu_s"] / 100.0)) * 100
        key = ("P1", 20, "ritz", seed, it)
        if key in done: continue
        net, cpu = train("P1", "ritz", 20, seed, 100.0, iters=it)
        rec = dict(prob="P1", d=20, method="ritz", seed=seed, w=100.0, iters=it, equal_cpu_to_pinn=p["cpu_s"], cpu_s=cpu, **evaluate(net, "P1", 20))
        open(out, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
