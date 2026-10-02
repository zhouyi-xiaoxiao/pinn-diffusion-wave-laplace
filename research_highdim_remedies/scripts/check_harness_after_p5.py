"""After adding P5 to src/ and scripts/run_c1.py: one plain run (P1, Deep Ritz, d = 20, seed 10) and one lift1 run
(P3, PINN, d = 20, seed 10) must reproduce the stored records bit for bit. Writes results/check_harness_after_p5.json."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import c1core as cc
out = []
for phase, problem, method, arm, w in (("main20", "laplace", "ritz", "plain", 100.0), ("posthoc_lift1", "ridge", "pinn", "lift1", 1000.0)):
    ref = [r for r in cc.load_jsonl(os.path.join(ROOT, "results", f"runs_{phase}.jsonl"))
           if r["problem"] == problem and r["method"] == method and r["arm"] == arm and r["seed"] == 10 and r["d"] == 20 and r["iters"] == 4000][0]
    _, res, _ = cc.train(problem, method, 20, 10, w, arm=arm)
    out.append(dict(phase=phase, problem=problem, method=method, arm=arm, rel_new=res["rel_l2"], rel_stored=ref["rel_l2"],
                    loss_new=res["final_loss"], loss_stored=ref["final_loss"], identical=res["rel_l2"] == ref["rel_l2"] and res["final_loss"] == ref["final_loss"]))
    print(out[-1], flush=True)
json.dump(dict(all_identical=all(o["identical"] for o in out), runs=out), open(os.path.join(ROOT, "results", "check_harness_after_p5.json"), "w"), indent=1)
