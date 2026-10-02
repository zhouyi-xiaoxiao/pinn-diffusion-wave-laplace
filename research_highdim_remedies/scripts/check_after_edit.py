"""After adding the exploratory arm lift1 to src/c1core.py (later choice, notes/VERIFICATION.md section 4): (a) the forward
Laplacian of lift1 against nested autograd in float64; (b) two finished counted runs re-trained with the edited
code give identical test errors (results/check_after_edit.json)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch, c1core as cc
torch.set_num_threads(1)
out = {}
for d in (5, 20):
    m = cc.Model(d, "lift1").double()
    x = torch.rand(128, d, dtype=torch.float64, generator=torch.Generator().manual_seed(7)).requires_grad_(True)
    u = m(x); g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    lap_ref = sum(torch.autograd.grad(g[:, i].sum(), x, create_graph=True)[0][:, i:i + 1] for i in range(d))
    _, lap = m.u_lap(x.detach())
    out[f"lift1/d{d}/max_rel_diff_lap"] = ((lap - lap_ref).abs().max() / lap_ref.abs().max()).item()
R = {(r["problem"], r["method"], r["arm"], r["seed"]): r for r in cc.load_jsonl(os.path.join(ROOT, "results", "runs_main20.jsonl"))}
for (problem, method, arm, seed) in (("laplace", "ritz", "lift3c", 10), ("laplace", "pinn", "plain", 10)):
    w = R[(problem, method, arm, seed)]["w"]
    _, rec, _ = cc.train(problem, method, 20, seed, w, arm=arm)
    out[f"{problem}/{method}/{arm}/s{seed}"] = dict(new=rec["rel_l2"], stored=R[(problem, method, arm, seed)]["rel_l2"],
                                                   identical=rec["rel_l2"] == R[(problem, method, arm, seed)]["rel_l2"])
print(out)
json.dump(out, open(os.path.join(ROOT, "results", "check_after_edit.json"), "w"), indent=1)
