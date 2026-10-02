"""Prop. 4 check: one-sweep Laplacian of c1core.Model (all arms, with shift) against separately written nested autograd, float64,
d in {1, 2, 7, 20}, default and inflated (x3) weights."""
import sys, json, torch
sys.path.insert(0, "../src"); import c1core as cc
torch.set_num_threads(1)
res = {}
for d in (1, 2, 7, 20):
    for arm in ("plain", "presolve", "lift3c", "lift3u", "rep3", "lift1"):
        for scale in (1.0, 3.0):
            torch.manual_seed(d)
            shift = torch.randn(d + 1).double() if arm == "presolve" else None
            m = cc.Model(d, arm, shift=shift).double()
            with torch.no_grad():
                for p in m.mlp.parameters(): p.mul_(scale)
            x = torch.rand(64, d, dtype=torch.float64, requires_grad=True)
            u = m(x); g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            lap = sum(torch.autograd.grad(g[:, i].sum(), x, retain_graph=True)[0][:, i] for i in range(d)).unsqueeze(1)
            u2, lap2 = m.u_lap(x.detach())
            res[f"{arm}_d{d}_s{scale:g}"] = dict(u=float((u - u2).abs().max()), lap_rel=float((lap - lap2).abs().max() / lap.abs().max()))
print(max(v["lap_rel"] for v in res.values()), max(v["u"] for v in res.values()))
json.dump(res, open("check_lap.json", "w"), indent=1)
