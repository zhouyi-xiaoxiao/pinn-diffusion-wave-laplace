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

"""Sanity checks of the separately written core + cross-check of the study's forward_laplacian against the separately written nested Laplacian (float64)."""
import sys, math, torch
sys.path.insert(0, _repo_path("verify_research_highdim/scripts"))
import v_core as V
sys.path.insert(0, _repo_path("research_highdim/scripts"))
import hd_core as H
torch.manual_seed(0)
for d in [2, 3, 5, 10, 20]:
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(200, d, generator=gen, dtype=torch.float64)
    for prob in ["laplace", "poisson"]:
        xx = x.clone().requires_grad_(True)
        u = V.u_exact(prob, xx)
        g, = torch.autograd.grad(u.sum(), xx, create_graph=True)
        lap = sum(torch.autograd.grad(g[:, i].sum(), xx, create_graph=True)[0][:, i] for i in range(d)) if g.requires_grad and g.grad_fn is not None else torch.zeros(200, dtype=torch.float64)
        res = (lap.unsqueeze(1) + V.f_src(prob, x)).abs().max().item()
        gerr = (g.detach() - V.grad_exact(prob, x)).abs().max().item()
        # separately written exact vs study exact
        dif = (V.u_exact(prob, x) - H.exact(prob, x)).abs().max().item()
        print(f"d={d} {prob}: max|Lap u*+f|={res:.2e}  grad err={gerr:.2e}  |recheck-study exact|={dif:.2e}")
    # forward Laplacian of the study vs the separately written nested, float64
    net = H.MLP(d).double()
    h, J, L = H.forward_laplacian(net, x)
    u2, g2, lap2 = V.lap_nested(net.net, x)
    print(f"   fwd-vs-nested f64: |u|={ (h-u2).abs().max().item():.1e} lap={ (L-lap2).abs().max().item():.1e} rel={ ((L-lap2).abs().max()/lap2.abs().max()).item():.1e}")
    # float32 loss values, both problems
    net = H.MLP(d)
    xi = torch.rand(1024, d, generator=gen); xb = V.boundary(1024, d, gen)
    for prob in ["laplace", "poisson"]:
        a = H.loss_pinn(net, prob, xi.clone(), xb, 1000.0)[0].item()
        b = V.loss_pinn(net.net, prob, xi.clone(), xb, 1000.0).item()
        c = H.loss_ritz(net, prob, xi.clone(), xb, 100.0)[0].item()
        e = V.loss_ritz(net.net, prob, xi.clone(), xb, 100.0).item()
        print(f"   f32 {prob}: study-fwd PINN loss {a:.9g} vs the separately written nested {b:.9g}; study Ritz {c:.9g} vs recheck {e:.9g}")
# boundary sampler checks
gen = torch.Generator().manual_seed(3)
for d in [2, 5, 20]:
    xb = V.boundary(200000, d, gen)
    on = ((xb == 0) | (xb == 1))
    print(f"d={d}: every pt has exactly one bdry coord: {(on.sum(1)==1).float().mean().item():.4f}; face freq min/max {(on.float().mean(0)).min().item():.4f}/{(on.float().mean(0)).max().item():.4f} (expect {1/d:.4f}); frac at 1: {(xb==1).sum().item()/on.sum().item():.4f}")
    xh = H.sample_boundary(200000, d, gen); on = ((xh == 0) | (xh == 1))
    print(f"   study sampler: one bdry coord {(on.sum(1)==1).float().mean().item():.4f}; face freq {(on.float().mean(0)).min().item():.4f}/{(on.float().mean(0)).max().item():.4f}; frac at 1 {(xh==1).sum().item()/on.sum().item():.4f}")
# param count
print("params d=10:", sum(p.numel() for p in V.make_net(10).parameters()), "expected 64*10+8449 =", 64*10+8449)
# floors: analytic
for d in [2,3,5,10,20]:
    m=d//2
    print(f"d={d}: P1 const floor {math.sqrt(7/(7+9*m)):.4f}, affine floor {math.sqrt(1/(7+9*m)):.4f}; P2 affine floor {math.sqrt(1-96/math.pi**4):.4f}")
