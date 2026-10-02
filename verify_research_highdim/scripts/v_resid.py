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

"""PDE residual of the study's final d=20 / d=10 P2 networks: has the PINN 'learnt no curvature' or learnt the Laplacian but not the harmonic part?"""
import sys, os, glob, torch, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_core as V
T = _repo_path("research_highdim/results/ckpt")
for p in ["poisson"]:
    for d in [10, 20]:
        gen = torch.Generator().manual_seed(999 + d); x = V.interior(4000, d, gen); xb = V.boundary(20000, d, gen)
        f = V.f_src(p, x)
        for fn in sorted(glob.glob(f"{T}/main_{p}_*_d{d}_s*_it4000.pt")):
            net = V.make_net(d); net.load_state_dict({k.replace("net.", "", 1): v for k, v in torch.load(fn).items()})
            u, g, lap = V.lap_nested(net, x)
            lap = lap.detach(); ue = V.u_exact(p, x)
            # per-coordinate second derivatives vs truth for coordinate 0
            xx = x.clone().requires_grad_(True); uu = net(xx); gg, = torch.autograd.grad(uu.sum(), xx, create_graph=True)
            h00 = torch.autograd.grad(gg[:, 0].sum(), xx)[0][:, 0]
            t00 = -math.pi ** 2 * torch.cos(math.pi * x[:, 0]) / math.sqrt(d)
            with torch.no_grad(): ub = net(xb)
            ueb = V.u_exact(p, xb)
            print(f"{os.path.basename(fn):42s} ||Lap u+f||/||f|| = {((lap+f)**2).sum().sqrt()/(f**2).sum().sqrt():.3f}  "
                  f"rel err {((u.detach()-ue)**2).sum().sqrt()/(ue**2).sum().sqrt():.3f}  bd rel err {((ub-ueb)**2).sum().sqrt()/(ueb**2).sum().sqrt():.3f}  "
                  f"d2u/dx0^2: rel err {((h00-t00)**2).sum().sqrt()/(t00**2).sum().sqrt():.3f} corr {torch.corrcoef(torch.stack([h00,t00]))[0,1]:.3f}")
