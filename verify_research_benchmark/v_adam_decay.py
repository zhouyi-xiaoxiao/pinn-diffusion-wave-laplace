"""Robustness check of the optimiser claim: is 'Adam -> L-BFGS beats Adam' partly an artefact of the
constant-lr Adam endpoint? Adam-only, 3000 steps, lr decayed exponentially 1e-3 -> 1e-5 (gamma per step),
same seeds/points as the main queue (random, N_r=256)."""
import json, os, time, numpy as np, torch
from vpinn import problem, make_mlp, points, loss_fn, evaluate
OUT = "v_adam_decay.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for name in ("heat1d", "laplace2d"):
    P = problem(name)
    for seed in (0, 1, 2):
        key = f"{name}|random|{seed}"
        if key in res: continue
        net = make_mlp(P.d, P.hidden, seed); Xr = points(P, "random", 256, seed)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-5 / 1e-3) ** (1 / 3000))
        t0 = time.time()
        for it in range(3000):
            opt.zero_grad(set_to_none=True); L = loss_fn(P, net, Xr); L.backward(); opt.step(); sch.step()
        r = evaluate(P, net)
        res[key] = {"rel_l2": r[0], "linf": r[1], "loss": float(L.detach()), "wall": time.time() - t0}
        json.dump(res, open(OUT, "w"), indent=1)
        print(key, res[key], flush=True)
print("DECAY DONE")
