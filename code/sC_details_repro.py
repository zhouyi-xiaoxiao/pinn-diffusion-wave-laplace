"""Bit-level reproducibility of the stacked GPU runs, and its limit (Supplementary Section S8.4).

Three questions, answered by execution:

  (1) Does a stack, repeated with unchanged code on the same machine and library versions, reproduce the stored
      results bit for bit?          ->  `rerun heat`, `rerun wave2d`, `rerun gridcontrol_B`, then `compare`
      heat    the benchmark stack of problem Heat   (50 networks: 5 strategies x seeds 0..9)
      wave2d  the benchmark stack of problem Wave2  (25 networks: 5 strategies x seeds 0..4)
      gridcontrol_B     stack B of the control experiment of Section 4.3          (20 networks: 2 point sets x seeds 0..9, N_r = 272)
  (2) Does the result of one network depend on the stack in which it is trained?
                                    ->  `stacksize mps`, `stacksize cpu`
      The network (random, seed 0) of problem Heat, 300 Adam + 300 L-BFGS iterations, is trained in stacks of
      two and of five networks, in two orders, and once more in the first stack.
  (3) The same question on stored results: stack A of the control experiment (40 networks) and the benchmark stack of Heat
      (50 networks) contain the same 20 (strategy, seed) pairs (random and cell-centred grid, seeds 0..9),
      trained by the same code.     ->  `compare` (no training)

Outputs
  data/sC_details_repro_runs/{heat,wave2d,gridcontrol_B}.json     compact records of the re-runs (errors, losses, curves)
  data/sC_details_repro_runs/stacksize_{mps,cpu}.json
  data/sC_details_repro.json, data/sC_details_repro.txt  the comparison quoted in the article
No timing claim is derived from these runs.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)
BENCH = os.path.join(ROOT, "research_benchmark")
DATA = os.path.join(ART, "data")
RUNS = os.path.join(DATA, "sC_details_repro_runs")
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)

SAMPLERS = ["grid", "random", "resample", "sobol", "rad"]          # order of research_benchmark/scripts/run_benchmark.py
STACKS = {
    "heat":   dict(problem="heat1d", n_r=256, stored=os.path.join(BENCH, "results", "runs", "heat1d_s0-10.json"),
                   specs=[(s, seed) for s in SAMPLERS for seed in range(0, 10)]),
    "wave2d": dict(problem="wave2d", n_r=512, stored=os.path.join(BENCH, "results", "runs", "wave2d_s0-5.json"),
                   specs=[(s, seed) for s in SAMPLERS for seed in range(0, 5)]),
    "gridcontrol_B":    dict(problem="heat1d", n_r=272, stored=os.path.join(DATA, "s4_lowdim_gridcontrol_parts", "stack_B.json"),
                   specs=[(s, seed) for s in ("gridplus0", "gridplusm") for seed in range(10)]),
}
N_ADAM, BRANCH_AT = 3000, 1500


def compact(run):
    out = {"sampler": run["sampler"], "seed": run["seed"]}
    for arm in ("adam", "adam_lbfgs"):
        a = run[arm]
        out[arm] = {"rel_l2": a["rel_l2"], "linf": a["linf"], "final_loss": a["final_loss"],
                    "curve": [[c["it"], c["loss"], c["test_rel_l2"]] for c in a["curve"]]}
    return out


def rerun(name, device):
    import torch
    torch.set_num_threads(4)
    from pinnbench.core import run_batched
    if name == "gridcontrol_B":
        import s4_lowdim_gridcontrol as gc
        gc.register_samplers()
    st = STACKS[name]
    t0 = time.time()
    print(f"=== re-run of stack {name}: {len(st['specs'])} networks, device {device}", flush=True)
    out, timing = run_batched(st["problem"], st["specs"], st["n_r"], N_ADAM, BRANCH_AT, device=device,
                              log_every=100, verbose=True)
    os.makedirs(RUNS, exist_ok=True)
    json.dump({"stack": name, "problem": st["problem"], "n_r": st["n_r"], "device": timing["device"],
               "batch_size": timing["batch_size"], "lbfgs_batched_fevals": timing["lbfgs_batched_fevals"],
               "torch_version": torch.__version__, "runs": [compact(o) for o in out]},
              open(os.path.join(RUNS, f"{name}.json"), "w"))
    print(f"=== done in {time.time() - t0:.0f} s (not a timing claim)", flush=True)


def stacksize(device):
    import torch
    torch.set_num_threads(2)
    from pinnbench.core import run_batched
    stacks = {"A: (random,0), (random,1)": [("random", 0), ("random", 1)],
              "B: (random,0), (grid,5)": [("random", 0), ("grid", 5)],
              "C: (random,0), (random,1), (grid,5), (sobol,2), (sobol,3)":
                  [("random", 0), ("random", 1), ("grid", 5), ("sobol", 2), ("sobol", 3)],
              "D: (grid,5), (random,0)": [("grid", 5), ("random", 0)],
              "A again": [("random", 0), ("random", 1)]}
    res = {}
    for label, specs in stacks.items():
        out, _ = run_batched("heat1d", specs, 256, 600, 300, device=device, log_every=100, verbose=False)
        r = [o for o in out if (o["sampler"], o["seed"]) == ("random", 0)][0]
        res[label] = {"stack_size": len(specs), "adam_rel_l2": r["adam"]["rel_l2"], "adam_final_loss": r["adam"]["final_loss"],
                      "adam_lbfgs_rel_l2": r["adam_lbfgs"]["rel_l2"], "adam_lbfgs_final_loss": r["adam_lbfgs"]["final_loss"]}
        print(device, label, "Adam %.10e" % r["adam"]["rel_l2"], "Adam->L-BFGS %.10e" % r["adam_lbfgs"]["rel_l2"], flush=True)
    os.makedirs(RUNS, exist_ok=True)
    json.dump({"device": device, "problem": "heat1d", "n_r": 256, "n_adam": 600, "branch_at": 300,
               "network": "(random, seed 0)", "torch_version": torch.__version__, "stacks": res},
              open(os.path.join(RUNS, f"stacksize_{device}.json"), "w"), indent=1)


def compare():
    out, lines = {"reruns": {}, "stack_size": {}}, []
    # (1) re-runs against the stored stacks
    for name, st in STACKS.items():
        p = os.path.join(RUNS, f"{name}.json")
        if not os.path.exists(p):
            continue
        new, old = json.load(open(p)), json.load(open(st["stored"]))
        n = same = curves = 0
        mx = 0.0
        for x, y in zip(old["runs"], new["runs"]):
            assert (x["sampler"], x["seed"]) == (y["sampler"], y["seed"])
            for arm in ("adam", "adam_lbfgs"):
                for q in ("rel_l2", "linf", "final_loss"):
                    n += 1
                    same += x[arm][q] == y[arm][q]
                    mx = max(mx, abs(x[arm][q] - y[arm][q]) / abs(x[arm][q]))
            curves += all([[c["it"], c["loss"], c["test_rel_l2"]] for c in x[arm]["curve"]] == y[arm]["curve"]
                          for arm in ("adam", "adam_lbfgs"))
        out["reruns"][name] = {"networks": len(old["runs"]), "final_values_compared": n, "final_values_identical": same,
                               "max_relative_difference": mx, "networks_with_identical_curves": curves,
                               "lbfgs_evaluations_stored": old["timing"]["lbfgs_batched_fevals"],
                               "lbfgs_evaluations_rerun": new["lbfgs_batched_fevals"],
                               "device_stored": old["timing"]["device"], "device_rerun": new["device"]}
        lines.append(f"re-run {name}: {same} of {n} final values identical (largest relative difference {mx:.1e}); "
                     f"{curves} of {len(old['runs'])} networks with identical logged curves in both arms; "
                     f"L-BFGS evaluations {old['timing']['lbfgs_batched_fevals']} -> {new['lbfgs_batched_fevals']}; "
                     f"device {old['timing']['device']} -> {new['device']}")
    # (2) one network in stacks of different size
    for dev in ("mps", "cpu"):
        p = os.path.join(RUNS, f"stacksize_{dev}.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))["stacks"]
        two = [k for k, v in s.items() if v["stack_size"] == 2]
        five = [k for k, v in s.items() if v["stack_size"] == 5]
        out["stack_size"][dev] = {
            "stacks": s,
            "identical_in_all_stacks_of_two": len({(s[k]["adam_rel_l2"], s[k]["adam_lbfgs_rel_l2"]) for k in two}) == 1,
            "identical_in_stacks_of_two_and_five": len({(v["adam_rel_l2"], v["adam_lbfgs_rel_l2"]) for v in s.values()}) == 1,
            "adam_two_vs_five": [s[two[0]]["adam_rel_l2"], s[five[0]]["adam_rel_l2"]],
            "adam_lbfgs_two_vs_five": [s[two[0]]["adam_lbfgs_rel_l2"], s[five[0]]["adam_lbfgs_rel_l2"]]}
        o = out["stack_size"][dev]
        lines.append(f"stack size, {dev}: identical in the four stacks of two: {o['identical_in_all_stacks_of_two']}; "
                     f"identical in the stacks of two and of five: {o['identical_in_stacks_of_two_and_five']}; "
                     f"Adam {o['adam_two_vs_five'][0]:.4e} (two) vs {o['adam_two_vs_five'][1]:.4e} (five); "
                     f"Adam->L-BFGS {o['adam_lbfgs_two_vs_five'][0]:.4e} vs {o['adam_lbfgs_two_vs_five'][1]:.4e}")
    # (3) stored results: control-experiment stack A (40 networks) against the benchmark stack of Heat (50 networks)
    a = json.load(open(os.path.join(DATA, "s4_lowdim_gridcontrol_parts", "stack_A.json")))
    b = json.load(open(STACKS["heat"]["stored"]))
    B = {(r["sampler"], r["seed"]): r for r in b["runs"]}
    cmp = {}
    for s in ("random", "grid"):
        ad = lb = 0
        first, ratio = [], []
        for r in a["runs"]:
            if r["sampler"] != s:
                continue
            q = B[(s, r["seed"])]
            ad += r["adam"]["rel_l2"] == q["adam"]["rel_l2"]
            lb += r["adam_lbfgs"]["rel_l2"] == q["adam_lbfgs"]["rel_l2"]
            diff = [c1["it"] for c1, c2 in zip(r["adam_lbfgs"]["curve"], q["adam_lbfgs"]["curve"]) if c1["loss"] != c2["loss"]]
            first.append(diff[0] if diff else None)
            ratio.append(r["adam_lbfgs"]["rel_l2"] / q["adam_lbfgs"]["rel_l2"])
        cmp[s] = {"networks": len(ratio), "adam_arm_identical": ad, "lbfgs_arm_identical": lb,
                  "first_logged_iteration_with_different_loss": sorted({f for f in first if f is not None}),
                  "ratio_control_over_benchmark_lbfgs_error": [min(ratio), max(ratio)], "per_seed_ratio": ratio}
        lines.append(f"Control stack A (40) vs benchmark Heat stack (50), {s}: Adam arm identical in {ad} of {len(ratio)}; "
                     f"L-BFGS arm identical in {lb} of {len(ratio)}; first logged iteration with a different loss "
                     f"{cmp[s]['first_logged_iteration_with_different_loss']}; ratio of the final L-BFGS errors "
                     f"{min(ratio):.2f} to {max(ratio):.2f}")
    allr = cmp["random"]["per_seed_ratio"] + cmp["grid"]["per_seed_ratio"]
    cmp["both"] = {"networks": len(allr), "adam_arm_identical": cmp["random"]["adam_arm_identical"] + cmp["grid"]["adam_arm_identical"],
                   "lbfgs_arm_identical": cmp["random"]["lbfgs_arm_identical"] + cmp["grid"]["lbfgs_arm_identical"],
                   "ratio_range": [min(allr), max(allr)], "stack_sizes": [a["timing"]["batch_size"], b["timing"]["batch_size"]]}
    out["gridcontrol_stack_A_vs_benchmark_heat"] = cmp
    json.dump(out, open(os.path.join(DATA, "sC_details_repro.json"), "w"), indent=1)
    open(os.path.join(DATA, "sC_details_repro.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if cmd == "rerun":
        rerun(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "mps")
    elif cmd == "stacksize":
        stacksize(sys.argv[2] if len(sys.argv) > 2 else "cpu")
    else:
        compare()
