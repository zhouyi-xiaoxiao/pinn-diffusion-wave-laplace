"""Check the single-forward-pass loss assembly (Problem.losses) against an independent,
term-by-term evaluation with separate forward passes, and stacked vs single models.
Output: checks/verify_losses.out"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(2)
from pinnbench.problems import make_problem, gradient, PROBLEMS
from pinnbench.core import MLP, BatchedMLP, total_loss

worst = 0.0
for name in PROBLEMS:
    P = make_problem(name)
    torch.manual_seed(0)
    nets = [MLP(P.d, P.hidden) for _ in range(3)]
    rng = np.random.default_rng(0)
    Xr = torch.as_tensor((P.lo + (P.hi - P.lo) * rng.random((200, P.d))).astype(np.float32))
    blocks = P.blocks()
    for k, net in enumerate(nets):
        tot, lres, comps = total_loss(P, net, Xr)
        Xg = Xr.clone().requires_grad_(True)
        ref_pde = (P.residual(net, Xg) ** 2).mean()
        diffs = [abs(float(lres) - float(ref_pde)) / float(ref_pde)]
        for nm, b, kind, tgt in P.terms(blocks):
            Xb = torch.as_tensor(blocks[b].astype(np.float32)).requires_grad_(True)
            u = net(Xb)
            val = u[:, 0] if kind == "u" else gradient(u, Xb)[:, kind]
            ref = ((val - torch.as_tensor(np.asarray(tgt, dtype=np.float32))) ** 2).mean()
            diffs.append(abs(float(comps[nm]) - float(ref)) / max(float(ref), 1e-12))
        worst = max(worst, max(diffs))
        if k == 0:
            print(f"{name}: single-pass vs term-by-term, max relative difference over {len(diffs)} terms = {max(diffs):.2e}")
    Pb = make_problem(name, batch=3)
    bn = BatchedMLP(nets)
    totb, _, _ = total_loss(Pb, bn, Xr.unsqueeze(0).repeat(3, 1, 1))
    singles = [float(total_loss(P, n, Xr)[0]) for n in nets]
    d = max(abs(float(totb[i]) - singles[i]) / singles[i] for i in range(3))
    worst = max(worst, d)
    print(f"{name}: stacked (S=3) vs single total loss, max relative difference = {d:.2e}")
print("WORST", worst)
