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

"""Resumable driver. Usage:
  python run_study.py sweep   # penalty-weight selection at d=5, selection seed 100, by VALIDATION error
  python run_study.py main    # d in {2,3,5,10} x {pinn,ritz} x seeds {0,1,2} x problems, chosen weights
Every finished run is appended to results/runs.jsonl (key = phase/problem/method/d/seed/w/iters);
re-running skips finished keys. Model weights -> results/ckpt/.
"""
import sys, os, json, time, torch
sys.path.insert(0, os.path.dirname(__file__))
import glob
from hd_core import train, fixed_sets, baselines, append_jsonl
from hd_core import load_jsonl as _load_one

ROOT = _repo_path("research_highdim")
RUNS = f"{ROOT}/results/runs.jsonl"          # sweep + first main runs
OUT = RUNS                                   # file this process appends to (per-worker when run in parallel)


def load_jsonl(_=None):
    """All records from results/runs*.jsonl (one file per parallel worker, so appends never interleave)."""
    recs = []
    for f in sorted(glob.glob(f"{ROOT}/results/runs*.jsonl")):
        recs += _load_one(f)
    return recs
CKPT = f"{ROOT}/results/ckpt"
os.makedirs(CKPT, exist_ok=True)
ITERS = 4000
PROBLEMS = ["laplace", "poisson"]
WEIGHTS = [1.0, 10.0, 100.0, 1000.0]


def key(r):
    return (r["phase"], r["problem"], r["method"], r["d"], r["seed"], r["w"], r["iters"])


def run(phase, problem, method, d, seed, w, iters, done):
    rec = dict(phase=phase, problem=problem, method=method, d=d, seed=seed, w=w, iters=iters)
    if key(rec) in done:
        return
    t0 = time.time()
    net, final, curve = train(problem, method, d, seed, w, iters=iters)
    rec.update(final); rec["curve"] = curve; rec["wall_s"] = time.time() - t0
    rec.update({f"base_{k}": v for k, v in baselines(problem, d).items()})
    torch.save(net.state_dict(), f"{CKPT}/{phase}_{problem}_{method}_d{d}_s{seed}_w{w:g}_it{iters}.pt")
    append_jsonl(OUT, rec); done.add(key(rec))
    print(f"[{phase}] {problem:8s} {method:4s} d={d:2d} seed={seed} w={w:g}: rel_l2={final['rel_l2']:.3e} "
          f"h1={final['rel_h1semi']:.3e} bd={final['bd_rel_l2']:.3e} val={final['val_rel_l2']:.3e} "
          f"{final['ms_per_iter']:.1f} ms/it  wall={rec['wall_s']:.0f}s", flush=True)


def chosen_weights():
    """argmin validation error over the sweep, per (problem, method)."""
    best = {}
    for r in load_jsonl(RUNS):
        if r["phase"] != "sweep":
            continue
        k = (r["problem"], r["method"])
        if k not in best or r["val_rel_l2"] < best[k][1]:
            best[k] = (r["w"], r["val_rel_l2"])
    return best


if __name__ == "__main__":
    phase = sys.argv[1]
    done = {key(r) for r in load_jsonl(RUNS)}
    if phase == "sweep":
        for problem in PROBLEMS:
            for method in ["ritz", "pinn"]:
                for w in WEIGHTS:
                    run("sweep", problem, method, 5, 100, w, ITERS, done)
        print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in chosen_weights().items()}, indent=1))
    elif phase == "ext":
        # Extensions, one worker per seed:  `ext <seed>`
        #  (a) d=20 with the identical main protocol (stored with phase="main");
        #  (b) "long": P1, d=10, 4x budget (16000 its) for both methods -> is the error budget-limited?
        #  (c) "equaltime": Deep Ritz at d=10 given the SAME training CPU time as the 4000-it PINN runs:
        #      iters = mean PINN CPU time / mean Ritz CPU ms-per-iter over the main d=10 runs, rounded to 1000.
        seed = int(sys.argv[2])
        OUT = f"{ROOT}/results/runs_ext_s{seed}.jsonl"
        best = chosen_weights()
        for problem in PROBLEMS:
            for method in ["ritz", "pinn"]:
                run("main", problem, method, 20, seed, best[(problem, method)][0], ITERS, done)
        for method in ["ritz", "pinn"]:
            run("long", "laplace", method, 10, seed, best[("laplace", method)][0], 4 * ITERS, done)
        R = [r for r in load_jsonl() if r["phase"] == "main" and r["d"] == 10 and r["iters"] == ITERS]
        for problem in PROBLEMS:
            tp = [r["cpu_time_s"] for r in R if r["problem"] == problem and r["method"] == "pinn"]
            mr = [r["cpu_ms_per_iter"] for r in R if r["problem"] == problem and r["method"] == "ritz"]
            assert len(tp) == 3 and len(mr) == 3, (problem, len(tp), len(mr))
            iters = int(round((sum(tp) / len(tp)) / (sum(mr) / len(mr) / 1e3) / 1000.0)) * 1000
            print(f"{problem}: PINN mean train CPU {sum(tp)/len(tp):.1f}s, Ritz {sum(mr)/len(mr):.2f} CPU ms/it -> Ritz iters {iters}", flush=True)
            run("equaltime", problem, "ritz", 10, seed, best[(problem, "ritz")][0], iters, done)
    elif phase == "main":
        best = chosen_weights()
        print("chosen weights:", best, flush=True)
        seeds = [int(a) for a in sys.argv[2:]] or [0, 1, 2]   # e.g. `main 1` = one worker per seed
        if sys.argv[2:]:
            OUT = f"{ROOT}/results/runs_main_s{'_'.join(sys.argv[2:])}.jsonl"
        for seed in seeds:
            for problem in PROBLEMS:
                for d in [2, 3, 5, 10]:
                    for method in ["ritz", "pinn"]:
                        run("main", problem, method, d, seed, best[(problem, method)][0], ITERS, done)
