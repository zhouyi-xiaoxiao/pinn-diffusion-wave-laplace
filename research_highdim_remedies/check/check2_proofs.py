"""Second round of the re-check: proof checks with separately written code: Prop. 3(e) P5 floors, Prop. 5 (rep3 under Adam), Prop. 1 for P5 at d = 1, 2
with small and large penalty weight (brute force by tensor Gauss quadrature + generic minimiser), Lemma 1 at d = 1."""
import json, math, os
import numpy as np, torch
from scipy import optimize
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); out = {}

# ---- Prop 3(e): closed-form P5 floor vs Monte-Carlo least squares (own sampler), d = 1, 2, 5, 20, 50
def p5(X):
    d = X.shape[1]; e = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
    return np.cos(math.pi * ((2*X - 1) @ e) / math.sqrt(d) + 1.0)
def fl_closed(d):
    a = math.pi / math.sqrt(d); ph = lambda z: math.sin(z) / z; chi = (math.sin(a) - a*math.cos(a)) / a**2
    Eu2 = 0.5 * (1 + math.cos(2) * ph(2*a)**d)
    return math.sqrt(1 - (math.cos(1)**2 * ph(a)**(2*d) + 3*d*math.sin(1)**2 * chi**2 * ph(a)**(2*d-2)) / Eu2)
rng = np.random.default_rng(12345)
for d in (1, 2, 5, 20, 50):
    fls = []
    for rep in range(2):
        X = rng.random((400_000, d)); u = p5(X); A = np.hstack([np.ones((len(X), 1)), X])
        r = u - A @ np.linalg.lstsq(A, u, rcond=None)[0]; fls.append(float(np.linalg.norm(r) / np.linalg.norm(u)))
    out[f"P3e_floor_d{d}"] = dict(closed=fl_closed(d), mc=fls)

# ---- Lemma 1 at d = 1 (edge case) and d = 2 by brute force of face moments
for d in (1, 2):
    M = np.zeros((d+1, d+1))
    for k in range(d):
        for s in (0.0, 1.0):
            # Gauss-Legendre on the face (d-1 free coords), exact for degree <= 2
            g, w = np.polynomial.legendre.leggauss(3); g = (g + 1)/2; w = w/2
            grids = np.meshgrid(*([g]*(d-1)), indexing="ij") if d > 1 else []
            W = np.prod(np.meshgrid(*([w]*(d-1)), indexing="ij"), axis=0).ravel() if d > 1 else np.ones(1)
            pts = np.zeros((len(W), d)); j = 0
            for i in range(d):
                if i == k: pts[:, i] = s
                else: pts[:, i] = grids[j].ravel(); j += 1
            Phi = np.hstack([np.ones((len(W), 1)), pts]); M += (Phi * W[:, None]).T @ Phi
    a = 1 + 2*(d-1)/3; Mc = np.full((d+1, d+1), d/2); Mc[0, 0] = 2*d; Mc[0, 1:] = Mc[1:, 0] = d; np.fill_diagonal(Mc[1:, 1:], a)
    B = np.array([[2*d, d*math.sqrt(d)], [d*math.sqrt(d), a + (d-1)*d/2]])
    out[f"L1_d{d}"] = dict(maxdiff=float(np.abs(M - Mc).max()), detB=float(np.linalg.det(B)), detB_formula=d*(d+2)/3,
                          eig=sorted(np.linalg.eigvalsh(M).tolist()))

# ---- Prop 1 for P5, d = 1, 2, beta in {1e-3, 1, 1e3}: brute-force population loss by tensor Gauss + BFGS
def brute(d, method, w, n=40):
    g, wt = np.polynomial.legendre.leggauss(n); g = (g + 1)/2; wt = wt/2
    G = np.stack(np.meshgrid(*([g]*d), indexing="ij"), -1).reshape(-1, d); Wi = np.prod(np.stack(np.meshgrid(*([wt]*d), indexing="ij"), -1).reshape(-1, d), 1)
    fpts, fw = [], []
    for k in range(d):
        for s in (0.0, 1.0):
            if d == 1: P = np.array([[s]]); Wf = np.ones(1)
            else:
                P = np.zeros((n, d)); P[:, 1-k] = g; P[:, k] = s; Wf = wt
            fpts.append(P); fw.append(Wf)
    Xb = np.vstack(fpts); Wb = np.concatenate(fw)
    ui, ub = p5(G), p5(Xb); f = 4*math.pi**2 * ui
    def loss(c):
        pi = c[0] + G @ c[1:]; pb = c[0] + Xb @ c[1:]
        bd = np.sum(Wb * (pb - ub)**2)
        if method == "ritz": return 0.5*np.sum(c[1:]**2) - np.sum(Wi * f * pi) + w * bd
        return bd
    return optimize.minimize(loss, np.zeros(d+1), method="BFGS", options=dict(gtol=1e-12, maxiter=10000)).x
import reimpl as R
for d in (1, 2):
    for method, ws in (("ritz", (1e-3, 1.0, 1e3)), ("pinn", (1000.0,))):
        for w in ws:
            c = R.affine_min("P5", d, method, w); cb = brute(d, method, w)
            out[f"P1prop_P5_d{d}_{method}_w{w:g}"] = dict(closed=c.tolist(), brute=cb.tolist(), maxdiff=float(np.abs(c - cb).max()))

# ---- Prop 5: rep3 under Adam == N(t) with 3x lr on the first weight (own construction), float64
def mlp(din, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(din, 16), torch.nn.Tanh(), torch.nn.Linear(16, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).double()
res = {}
for d in (1, 3, 20):
    n3 = mlp(3*d, 7); n1 = mlp(d, 8)
    with torch.no_grad():
        W = n3[0].weight; n1[0].weight.copy_(W[:, :d] + W[:, d:2*d] + W[:, 2*d:]); n1[0].bias.copy_(n3[0].bias)
        for i in (2, 4): n1[i].weight.copy_(n3[i].weight); n1[i].bias.copy_(n3[i].bias)
    kw = dict(betas=(0.8, 0.95), eps=1e-6)
    o3 = torch.optim.Adam(n3.parameters(), lr=2e-3, **kw)
    o1 = torch.optim.Adam([{"params": [n1[0].weight], "lr": 6e-3}, {"params": [p for k, p in n1.named_parameters() if k != "0.weight"], "lr": 2e-3}], **kw)
    s3 = torch.optim.lr_scheduler.MultiStepLR(o3, [100, 150], 0.1); s1 = torch.optim.lr_scheduler.MultiStepLR(o1, [100, 150], 0.1)
    g = torch.Generator().manual_seed(1); worst = 0.0
    for it in range(200):
        x = torch.rand(256, d, generator=g, dtype=torch.float64, requires_grad=True); t = 2*x - 1
        for net, opt, inp in ((n3, o3, torch.cat([t, t, t], 1)), (n1, o1, t)):
            u = net(inp); gu = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            L = (0.5*(gu**2).sum(1) - torch.sin(3*x.sum(1)) * u.squeeze()).mean()   # a Ritz-type loss
            opt.zero_grad(); L.backward(); opt.step()
        s3.step(); s1.step()
        with torch.no_grad():
            xt = torch.rand(500, d, dtype=torch.float64, generator=torch.Generator().manual_seed(9)); tt = 2*xt - 1
            worst = max(worst, float((n3(torch.cat([tt, tt, tt], 1)) - n1(tt)).abs().max()))
    res[f"d{d}"] = worst
out["Prop5_maxdiff_200_steps"] = res
# init distribution: variance and kurtosis of W_eff (rep3) vs lift1 first weight; biases
d = 20; We, W1, b3, b1 = [], [], [], []
for s in range(200):
    torch.manual_seed(s); l3 = torch.nn.Linear(3*d, 64); torch.manual_seed(s); l1 = torch.nn.Linear(d, 64)
    We.append((l3.weight[:, :d] + l3.weight[:, d:2*d] + l3.weight[:, 2*d:]).detach().numpy().ravel()); W1.append(l1.weight.detach().numpy().ravel())
    b3.append(l3.bias.detach().numpy()); b1.append(l1.bias.detach().numpy())
kurt = lambda z: float(np.mean((z - z.mean())**4) / np.var(z)**2)
We, W1 = np.concatenate(We), np.concatenate(W1)
out["Prop5c_init_d20"] = dict(var_Weff=float(We.var()), var_lift1=float(W1.var()), theory=1/(3*d), kurt_Weff=kurt(We), kurt_lift1=kurt(W1),
                              kurt_uniform=1.8, max_abs_Weff=float(np.abs(We).max()), max_abs_lift1=float(np.abs(W1).max()),
                              var_b_rep3=float(np.concatenate(b3).var()), var_b_lift1=float(np.concatenate(b1).var()))
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "check2_proofs.json"), "w"), indent=1)
