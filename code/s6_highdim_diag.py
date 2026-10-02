"""Diagnostics of the d = 20 networks of Section 6: what is the error made of?

For every final network at d = 20 -- the 4000-iteration networks of the main study
(research_highdim/results/ckpt) and the 16 000-iteration networks of Section 6.3
(data/s6_highdim_long_d20_ckpt) -- compute on fresh held-out points

  rel_l2            ||u_theta - u*|| / ||u*||                       (50 000 interior points)
  affine_floor      ||u* - A u*|| / ||u*||, A = least-squares affine fit on the same points
  nonaffine_ratio   ||u_theta - A u_theta|| / ||u* - A u*||         (size of the non-affine part)
  nonaffine_corr    correlation of (u_theta - A u_theta) with (u* - A u*)
  affine_err        ||A u_theta - A u*|| / ||u*||
  resid_ratio       ||Lap u_theta + f|| / ||f||   (P2 only, f = pi^2 u*; 4000 interior points)
  lap_rms           rms of Lap u_theta            (4000 interior points)
  bd_rel_l2         relative L2 error on 20 000 boundary points

This repeats, with a separate script and point set, the analysis of the verification code V-dim of the
4000-iteration networks (verify_research_highdim/results/v_affine.txt, v_resid.txt) and extends
it to the long runs.  No training, no timing.

Output: data/s6_highdim_diag_d20.json
"""
import json
import math
import sys
from pathlib import Path

ART = Path(__file__).resolve().parents[1]             # folder holding code/, data/, figures/
ROOT = ART if (ART / "research_benchmark").is_dir() else ART.parent   # repository root
sys.path.insert(0, str(ROOT / "research_highdim" / "scripts"))

import torch  # noqa: E402
from hd_core import MLP, exact, source, forward_laplacian, sample_interior, sample_boundary  # noqa: E402

D = 20
WEIGHT = {("laplace", "pinn"): 1000.0, ("laplace", "ritz"): 100.0,
          ("poisson", "pinn"): 1000.0, ("poisson", "ritz"): 1.0}
CK4 = ROOT / "research_highdim" / "results" / "ckpt"
CK16 = ART / "data" / "s6_highdim_long_d20_ckpt"
OUT = ART / "data" / "s6_highdim_diag_d20.json"


def affine_fit(x, y):
    A = torch.cat([x, torch.ones(x.shape[0], 1)], 1).double()
    c = torch.linalg.lstsq(A, y.double()).solution
    return (A @ c)


def nrm(v):
    return v.double().pow(2).sum().sqrt().item()


def diagnose(net, problem, x, xr, xb):
    with torch.no_grad():
        u = net(x).double()
    ue = exact(problem, x).double()
    au, ae = affine_fit(x, u), affine_fit(x, ue)
    na_u, na_e = u - au, ue - ae
    out = dict(rel_l2=nrm(u - ue) / nrm(ue), affine_floor=nrm(na_e) / nrm(ue),
               nonaffine_ratio=nrm(na_u) / nrm(na_e),
               nonaffine_corr=torch.corrcoef(torch.cat([na_u.T, na_e.T]))[0, 1].item(),
               affine_err=nrm(au - ae) / nrm(ue))
    with torch.no_grad():
        _, _, lap = forward_laplacian(net, xr)
        f = source(problem, xr)
        ub = net(xb)
    out["lap_rms"] = lap.double().pow(2).mean().sqrt().item()
    out["resid_ratio"] = (nrm(lap + f) / nrm(f)) if problem == "poisson" else None
    ueb = exact(problem, xb)
    out["bd_rel_l2"] = nrm(ub - ueb) / nrm(ueb)
    return out


def main():
    gen = torch.Generator().manual_seed(31_000 + D)
    x = sample_interior(50_000, D, gen)
    xr = sample_interior(4_000, D, gen)
    xb = sample_boundary(20_000, D, gen)
    rows = []
    for problem in ["laplace", "poisson"]:
        for method in ["pinn", "ritz"]:
            w = WEIGHT[(problem, method)]
            for iters, path_of in [(4000, lambda s: CK4 / f"main_{problem}_{method}_d{D}_s{s}_w{w:g}_it4000.pt"),
                                   (16000, lambda s: CK16 / f"{problem}_{method}_d{D}_s{s}_it16000.pt")]:
                for seed in [0, 1, 2]:
                    path = path_of(seed)
                    if not path.exists():
                        print("missing", path.relative_to(ROOT))
                        continue
                    net = MLP(D)
                    net.load_state_dict(torch.load(path))
                    r = dict(problem=problem, method=method, d=D, iters=iters, seed=seed,
                             ckpt=str(path.relative_to(ROOT)))
                    r.update(diagnose(net, problem, x, xr, xb))
                    rows.append(r)
                    rr = "  n/a" if r["resid_ratio"] is None else f"{r['resid_ratio']:.3f}"
                    print(f"{problem:8s} {method:4s} it={iters:5d} s={seed}: rel_l2={r['rel_l2']:.4f} "
                          f"floor={r['affine_floor']:.4f} nonaffine ratio={r['nonaffine_ratio']:.3f} "
                          f"corr={r['nonaffine_corr']:.3f} affine_err={r['affine_err']:.4f} resid={rr} "
                          f"lap_rms={r['lap_rms']:.3f} bd={r['bd_rel_l2']:.4f}")
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"{len(rows)} networks -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
