"""the working document of the proofs, Proposition 5: rep3 = N(t, t, t) trained by Adam(lr) is the centred-input network N(t) whose
first-layer weight W1 + W2 + W3 is trained by Adam(3 lr) (other parameters Adam(lr)). Numerical check in float64:
both are trained for 300 steps of the study's Deep Ritz loss (P3, d = 20, beta = 100) on the same batches;
the maximum difference of the outputs on 2000 test points is reported, together with the initial standard
deviations of the effective first-layer weights. Writes results/check_rep3_equivalence.json."""
import sys, os, json, copy
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch
import c1core as cc
from c1core import hc
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
d, w, steps = 20, 100.0, 300
out = {}
torch.manual_seed(10)
rep = cc.Model(d, "rep3").double()
eq = cc.Model(d, "lift1").double()
with torch.no_grad():
    L0r, L0e = rep.mlp.net[0], eq.mlp.net[0]
    L0e.weight.copy_(L0r.weight[:, :d] + L0r.weight[:, d:2 * d] + L0r.weight[:, 2 * d:])
    L0e.bias.copy_(L0r.bias)
    for a, b in zip(list(rep.mlp.net)[1:], list(eq.mlp.net)[1:]):
        for pa, pb in zip(a.parameters(), b.parameters()):
            pb.copy_(pa)
opt_r = torch.optim.Adam(rep.mlp.parameters(), lr=1e-3)
first = [eq.mlp.net[0].weight]
rest = [p for p in eq.mlp.parameters() if p is not eq.mlp.net[0].weight]
opt_e = torch.optim.Adam([dict(params=first, lr=3e-3), dict(params=rest, lr=1e-3)])
gen = torch.Generator().manual_seed(1010)
xt = torch.rand(2000, d, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
diffs = []
for it in range(steps):
    xi = hc.sample_interior(1024, d, gen).double(); xb = hc.sample_boundary(1024, d, gen).double()
    for m, o in ((rep, opt_r), (eq, opt_e)):
        o.zero_grad()
        loss = cc.loss_ritz(m, "ridge", xi.clone(), xb, w)[0]
        loss.backward(); o.step()
    if (it + 1) % 50 == 0:
        with torch.no_grad():
            diffs.append(dict(it=it + 1, max_abs_diff=(rep(xt) - eq(xt)).abs().max().item(), max_abs_u=rep(xt).abs().max().item()))
out["trajectory"] = diffs
out["max_abs_diff"] = max(x["max_abs_diff"] for x in diffs)
# initial distribution of the effective first-layer weight: rep3 (sum of three U(+-1/sqrt(3d))) vs lift1 (U(+-1/sqrt(d)))
sd_r, sd_l, sb_r, sb_l = [], [], [], []
for s in range(10, 30):
    torch.manual_seed(s); r = cc.Model(d, "rep3"); torch.manual_seed(s); l = cc.Model(d, "lift1")
    W = r.mlp.net[0].weight
    sd_r.append((W[:, :d] + W[:, d:2 * d] + W[:, 2 * d:]).std().item()); sd_l.append(l.mlp.net[0].weight.std().item())
    sb_r.append(r.mlp.net[0].bias.std().item()); sb_l.append(l.mlp.net[0].bias.std().item())
mean = lambda v: sum(v) / len(v)
out["init_sd_effective_weight"] = dict(rep3=mean(sd_r), lift1=mean(sd_l), theory=(1 / (3 * d)) ** 0.5)
out["init_sd_bias"] = dict(rep3=mean(sb_r), lift1=mean(sb_l), theory_rep3=(1 / (9 * d)) ** 0.5, theory_lift1=(1 / (3 * d)) ** 0.5)
json.dump(out, open(os.path.join(ROOT, "results", "check_rep3_equivalence.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
