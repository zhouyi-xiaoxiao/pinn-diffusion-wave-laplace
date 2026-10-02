"""Evaluation of the computable bound (Corollary 2(c) of proof.tex) on saved networks.

For a network v (C^2 on the closed cube) and the problem data f, g = u* on the boundary:
    ||v - u*|| <= B_int + B_bd,   B_int = ||Lap v + f|| / (d pi^2),   B_bd = sqrt(U_d) ||v - g||_{L2(bd)},
||.||_{L2(bd)}^2 = 2d * (mean over uniform boundary points). Norms are Monte-Carlo estimates on
  set 'fresh': 2e5 interior points (seed 50_000 + d) and 2e5 boundary points (seed 60_000 + d), float64;
  set 'hd'   : hd_core.fixed_sets(d) test (5e4 interior) and test_bd (2e4 boundary) points.
Networks are evaluated in float64 (parameters cast from the saved float32 values); the Laplacian is
the exact one of hd_core.forward_laplacian, checked against nested autograd on 64 points per network.
Standard errors: delta method on per-point contributions, interior and boundary samples independent;
the s.e. of D = bound - error accounts for the correlation of B_int and the error (same points).
Usage: s4_eval_bound.py pilot     -> only the pilot-E networks of ../ideas_theory/results/nets_E (debug)
       s4_eval_bound.py confirm   -> all 88 checkpoints (only after PREREG_FREEZE.json exists)
Output: results/eval_<mode>.jsonl (one record per network and set; resumable)."""
import json, math, os, re, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from t2common import U, RES, ROOT, HD_SCRIPTS, CKPT_DIR, PILOT_E_NETS, append_jsonl, load_jsonl
sys.path.insert(0, HD_SCRIPTS)
import hd_core as hc

torch.set_num_threads(1)
PAT = re.compile(r"(main|long|equaltime|sweep)_(laplace|poisson)_(pinn|ritz)_d(\d+)_s(\d+)_w(\d+)_it(\d+)\.pt")
PATE = re.compile(r"(E1|E2)_(laplace|poisson)_d(\d+)_s(\d+)_(pinn|ritz)_w(\d+)\.pt")
N_FRESH = 200_000
BATCH = 4000


def fresh_sets(d):
    gi = torch.Generator().manual_seed(50_000 + d)
    gb = torch.Generator().manual_seed(60_000 + d)
    xi = torch.rand(N_FRESH, d, generator=gi, dtype=torch.float64)
    xb = torch.rand(N_FRESH, d, generator=gb, dtype=torch.float64)
    face = torch.randint(0, d, (N_FRESH,), generator=gb)
    side = torch.randint(0, 2, (N_FRESH,), generator=gb).to(torch.float64)
    xb[torch.arange(N_FRESH), face] = side
    return xi, xb


def load_net(path, d):
    obj = torch.load(path, map_location="cpu")
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    net = hc.MLP(d)
    net.load_state_dict(sd)
    return net.double().eval()


def lap_check(net, problem, x):
    x = x[:64].clone().requires_grad_(True)
    u = net(x)
    lap_nested, _ = hc.laplacian(u, x)
    with torch.no_grad():
        _, _, lap_fwd = hc.forward_laplacian(net, x.detach())
    return float((lap_nested.detach() - lap_fwd).abs().max() / (lap_fwd.abs().max() + 1e-300))


def per_point(net, problem, xi, xb):
    e2, r2, ue2, er = [], [], [], []
    for xc in torch.split(xi, BATCH):
        with torch.no_grad():
            u, _, lap = hc.forward_laplacian(net, xc)
            ue = hc.exact(problem, xc)
            r = lap + hc.source(problem, xc)
        e2.append(((u - ue) ** 2).squeeze(1)); r2.append((r ** 2).squeeze(1)); ue2.append((ue ** 2).squeeze(1))
        er.append(torch.stack([(u - ue).abs().squeeze(1), r.abs().squeeze(1)], 1))
    s2, eb = [], []
    for xc in torch.split(xb, BATCH):
        with torch.no_grad():
            s = net(xc) - hc.exact(problem, xc)
        s2.append((s ** 2).squeeze(1)); eb.append(s.abs().squeeze(1))
    er = torch.cat(er)
    return (torch.cat(e2), torch.cat(r2), torch.cat(ue2), torch.cat(s2),
            float(er[:, 0].max()), float(er[:, 1].max()), float(torch.cat(eb).max()))


def stats(d, e2, r2, ue2, s2):
    n, nb = len(e2), len(s2)
    me, mr, mu, ms = e2.mean().item(), r2.mean().item(), ue2.mean().item(), s2.mean().item()
    E = math.sqrt(me); R = math.sqrt(mr); S = math.sqrt(2 * d * ms); Ud = U(d)
    B_int = R / (d * math.pi ** 2); B_bd = math.sqrt(Ud) * S
    bound = B_int + B_bd
    # derivatives of each quantity with respect to the per-point means (delta method)
    dE = 1 / (2 * E) if E > 0 else 0.0
    dBint = 1 / (2 * R * d * math.pi ** 2) if R > 0 else 0.0
    dBbd = math.sqrt(Ud) * math.sqrt(2 * d) / (2 * math.sqrt(ms)) if ms > 0 else 0.0
    var_int = lambda g: g.var().item() / n
    var_bd = lambda g: g.var().item() / nb
    se_E = math.sqrt(var_int(dE * e2)); se_Bint = math.sqrt(var_int(dBint * r2)); se_Bbd = math.sqrt(var_bd(dBbd * s2))
    se_D = math.sqrt(var_int(dBint * r2 - dE * e2) + var_bd(dBbd * s2))
    # eta = bound / E: d log eta = d bound / bound - d E / E
    se_eta = (bound / E) * math.sqrt(var_int(dBint * r2 / bound - dE * e2 / E) + var_bd(dBbd * s2 / bound))
    # share = B_bd / bound: d share = (B_int dB_bd - B_bd dB_int) / bound^2
    se_share = math.sqrt(var_int(-B_bd * dBint * r2 / bound ** 2) + var_bd(B_int * dBbd * s2 / bound ** 2))
    u_norm = math.sqrt(mu)
    return dict(n_int=n, n_bd=nb, err_abs=E, se_err=se_E, rel_l2=E / u_norm, u_norm_mc=u_norm,
                res_l2=R, bd_l2=S, B_int=B_int, se_B_int=se_Bint, B_bd=B_bd, se_B_bd=se_Bbd,
                bound=bound, se_bound=math.sqrt(se_Bint ** 2 + se_Bbd ** 2), D=bound - E, se_D=se_D,
                z_D=(bound - E) / se_D if se_D > 0 else float("inf"),
                eta=bound / E, se_eta=se_eta, share_bd=B_bd / bound, se_share=se_share,
                loss_int=mr, loss_bd=ms, loss_form_check=math.sqrt(mr) / (d * math.pi ** 2) + math.sqrt(2 * d * Ud) * math.sqrt(ms))


def networks(mode):
    out = []
    if mode == "pilot":
        for f in sorted(os.listdir(PILOT_E_NETS)):
            m = PATE.fullmatch(f)
            if m:
                out.append(dict(file=f, path=os.path.join(PILOT_E_NETS, f), family="pilotE_" + m[1], problem=m[2],
                                d=int(m[3]), seed=int(m[4]), method=m[5], w=int(m[6]), iters=4000))
    else:
        inv = {r["file"]: r for r in json.load(open(os.path.join(RES, "inventory.json")))["rows"]}
        for f in sorted(os.listdir(CKPT_DIR)):
            m = PAT.fullmatch(f)
            if m:
                out.append(dict(file=f, path=os.path.join(CKPT_DIR, f), family=m[1], problem=m[2], method=m[3],
                                d=int(m[4]), seed=int(m[5]), w=int(m[6]), iters=int(m[7]),
                                confirmatory=inv[f]["confirmatory"]))
    return out


def main():
    mode = sys.argv[1]
    if mode == "confirm" and not os.path.exists(os.path.join(ROOT, "PREREG_FREEZE.json")):
        sys.exit("PREREG_FREEZE.json missing: freeze the pre-registration first")
    path = os.path.join(RES, f"eval_{mode}.jsonl")
    done = {(r["file"], r["set"]) for r in load_jsonl(path)}
    nets = networks(mode)
    cache = {}
    for nt in nets:
        d = nt["d"]
        if d not in cache:
            hs = hc.fixed_sets(d)
            cache = {d: dict(fresh=fresh_sets(d), hd=(hs["test"].double(), hs["test_bd"].double()))}
        net = load_net(nt["path"], d)
        for setname in ("fresh", "hd"):
            if (nt["file"], setname) in done:
                continue
            c0 = time.process_time()
            xi, xb = cache[d][setname]
            e2, r2, ue2, s2, max_e, max_r, max_bd = per_point(net, nt["problem"], xi, xb)
            rec = {k: v for k, v in nt.items() if k != "path"}
            rec.update(set=setname, lap_check_rel=lap_check(net, nt["problem"], xi))
            rec.update(stats(d, e2, r2, ue2, s2))
            rec.update(sampled_max_err=max_e, sampled_max_res=max_r, sampled_max_bd=max_bd,
                       sampled_maxprinciple_bound=max_bd + max_r / 8, cpu_s=time.process_time() - c0)
            append_jsonl(path, rec)
            print(f"{nt['file']:48s} {setname:5s} err {rec['err_abs']:.4e} B_int {rec['B_int']:.3e} B_bd {rec['B_bd']:.3e} "
                  f"eta {rec['eta']:.3f}+-{rec['se_eta']:.3f} z_D {rec['z_D']:.1f} share {rec['share_bd']:.3f} "
                  f"lapchk {rec['lap_check_rel']:.1e} cpu {rec['cpu_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
