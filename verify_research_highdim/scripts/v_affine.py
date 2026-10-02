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

"""Is the trained net at d=20 'only the linear part'?  Decompose the study's final networks into their best affine fit + remainder."""
import sys, os, glob, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_core as V
T = _repo_path("research_highdim/results/ckpt")
def affine_fit(x, y):
    A = torch.cat([x, torch.ones(x.shape[0], 1)], 1).double()
    c = torch.linalg.lstsq(A, y.double()).solution
    return (A @ c).float()
for p in ["poisson", "laplace"]:
    for d in [10, 20]:
        gen = torch.Generator().manual_seed(4242 + d); x = V.interior(50000, d, gen); ue = V.u_exact(p, x)
        aff_star = affine_fit(x, ue); nrm = (ue ** 2).sum().sqrt()
        for f in sorted(glob.glob(f"{T}/main_{p}_*_d{d}_s*_it4000.pt")):
            net = V.make_net(d); net.load_state_dict({k.replace("net.", "", 1): v for k, v in torch.load(f).items()})
            with torch.no_grad(): u = net(x)
            au = affine_fit(x, u)
            print(f"{os.path.basename(f):45s} rel err {((u-ue)**2).sum().sqrt()/nrm:.4f} | affine floor {((aff_star-ue)**2).sum().sqrt()/nrm:.4f} | "
                  f"non-affine part of u_theta / non-affine part of u*: {((u-au)**2).sum().sqrt()/((ue-aff_star)**2).sum().sqrt():.3f} | "
                  f"err of u_theta's affine part vs u*'s affine part (rel to ||u*||): {((au-aff_star)**2).sum().sqrt()/nrm:.4f} | "
                  f"corr(nonaffine u_theta, nonaffine u*) {torch.corrcoef(torch.cat([(u-au).T,(ue-aff_star).T]))[0,1]:.3f}")
