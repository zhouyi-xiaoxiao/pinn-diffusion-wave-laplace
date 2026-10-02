"""Inventory of the saved networks (no evaluation of any norm, no bound).

Lists the 88 checkpoints of pinn/research_highdim/results/ckpt/ with their configuration and
SHA-256, and finds those whose parameters equal (to 1e-6 relative, max abs) a network used in
pilot E2 (../ideas_theory/results/nets_E/); those are non-confirmatory. Output:
results/inventory.json."""
import os, re, sys, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from t2common import CKPT_DIR, PILOT_E_NETS, sha256, dump

PAT = re.compile(r"(main|long|equaltime|sweep)_(laplace|poisson)_(pinn|ritz)_d(\d+)_s(\d+)_w(\d+)_it(\d+)\.pt")
PATE = re.compile(r"(E1|E2)_(laplace|poisson)_d(\d+)_s(\d+)_(pinn|ritz)_w(\d+)\.pt")


def flat(sd):
    return torch.cat([v.reshape(-1).double() for k, v in sorted(sd.items())])


def main():
    pil = {}
    for f in sorted(os.listdir(PILOT_E_NETS)):
        m = PATE.fullmatch(f)
        if m:
            pil[f] = dict(part=m[1], problem=m[2], d=int(m[3]), seed=int(m[4]), method=m[5], w=int(m[6]),
                          params=flat(torch.load(os.path.join(PILOT_E_NETS, f), map_location="cpu")))
    rows = []
    for f in sorted(os.listdir(CKPT_DIR)):
        m = PAT.fullmatch(f)
        if not m:
            print("unmatched", f); continue
        obj = torch.load(os.path.join(CKPT_DIR, f), map_location="cpu")
        sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        p = flat(sd)
        same_cfg, identical = [], []
        for pf, q in pil.items():
            cfg = (q["problem"], q["d"], q["seed"], q["method"], q["w"]) == (m[2], int(m[4]), int(m[5]), m[3], int(m[6]))
            if cfg and m[1] == "main":
                same_cfg.append(pf)
            if q["d"] == int(m[4]) and q["params"].numel() == p.numel():
                diff = (q["params"] - p).abs().max().item()
                if diff <= 1e-6 * max(1.0, p.abs().max().item()):
                    identical.append(dict(file=pf, max_abs_diff=diff))
        rows.append(dict(file=f, family=m[1], problem=m[2], method=m[3], d=int(m[4]), seed=int(m[5]),
                         w=int(m[6]), iters=int(m[7]), sha256=sha256(os.path.join(CKPT_DIR, f)),
                         keys=list(sd.keys()) if isinstance(sd, dict) else None,
                         pilot_same_config=same_cfg, pilot_identical=identical,
                         confirmatory=(not same_cfg and not identical)))
        print(f, rows[-1]["confirmatory"], identical, flush=True)
    out = dict(n=len(rows), n_confirmatory=sum(r["confirmatory"] for r in rows),
               pilot_E_files=sorted(pil), rows=rows)
    dump("inventory.json", out)
    print("n", out["n"], "confirmatory", out["n_confirmatory"])


if __name__ == "__main__":
    main()
