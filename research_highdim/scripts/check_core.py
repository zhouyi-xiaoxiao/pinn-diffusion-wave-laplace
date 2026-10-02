"""Unit checks for hd_core: exact solutions, boundary sampler, forward-Laplacian == nested autograd."""
import torch, time
from hd_core import *
for prob in ["laplace", "poisson"]:
    for d in [2, 3, 5, 10]:
        g = torch.Generator().manual_seed(0)
        x = torch.rand(2000, d, generator=g, dtype=torch.float64).requires_grad_(True)
        u = exact(prob, x); lap, gr = laplacian(u, x)
        f = source(prob, x.detach()).double()
        print(f"{prob} d={d}: max|Δu*+f|={(lap+f).abs().max().item():.2e}  max|∇u*-analytic|={(gr-exact_grad(prob,x.detach())).abs().max().item():.2e}")
xb = sample_boundary(100000, 5, torch.Generator().manual_seed(1))
on = ((xb == 0) | (xb == 1))
print("boundary sampler: all on ∂Ω:", bool(on.any(1).all()), " face freq×d:", [round(v, 3) for v in (on.float().mean(0) * 5).tolist()])
for d in [2, 10]:
    torch.manual_seed(0)
    net = MLP(d).double()
    x = torch.rand(500, d, dtype=torch.float64)
    u1, g1, l1 = forward_laplacian(net, x)
    xr = x.clone().requires_grad_(True); u2 = net(xr); l2, g2 = laplacian(u2, xr)
    print(f"forward-vs-nested d={d} (float64): max|Δ diff|={(l1-l2).abs().max().item():.2e} max|∇ diff|={(g1-g2).abs().max().item():.2e} max|u diff|={(u1-u2).abs().max().item():.2e}")
    net32 = MLP(d); gen = torch.Generator().manual_seed(0)
    xi = sample_interior(1024, d, gen); xb = sample_boundary(1024, d, gen)
    la = loss_pinn(net32, "poisson", xi.clone(), xb, 10.0)[0]; lb = loss_pinn_nested(net32, "poisson", xi.clone(), xb, 10.0)[0]
    ga = torch.autograd.grad(la, list(net32.parameters())); gb = torch.autograd.grad(lb, list(net32.parameters()))
    print(f"   float32 loss: forward={la.item():.7f} nested={lb.item():.7f}; max param-grad diff={max((a-b).abs().max().item() for a,b in zip(ga,gb)):.2e}")
