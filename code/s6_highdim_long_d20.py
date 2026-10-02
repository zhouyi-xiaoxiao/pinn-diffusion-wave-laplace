"""Longer runs at d = 20 (Section 6.3): is the d = 20 plateau a budget effect?

The main dimension study trains for 4000 iterations.  Here the same protocol is run with a
four-times longer schedule (16 000 iterations, learning-rate milestones at 50 % and 75 % of
the run, exactly as hd_core.train scales them) at d = 20, for both problems (P1 "laplace",
P2 "poisson"), both methods (PINN, Deep Ritz) and seeds 0, 1, 2, with the weights of the
main study (lambda = 1000; beta = 100 on P1 and beta = 1 on P2).

Training is done by research_highdim/scripts/hd_core.train, unmodified.

Usage (one single-thread worker per (problem, seed); at most three at once):
    python s6_highdim_long_d20.py check                # bit-for-bit rerun of one stored cell
    python s6_highdim_long_d20.py run laplace 0        # ... poisson 2
    python s6_highdim_long_d20.py merge                # parts -> data/s6_highdim_long_d20.jsonl

Outputs
    data/s6_highdim_long_d20_parts/<problem>_s<seed>.jsonl   (one record per run)
    data/s6_highdim_long_d20_ckpt/<problem>_<method>_d20_s<seed>_it16000.pt
    data/s6_highdim_long_d20.jsonl                           (merged, 12 records)

The CPU/wall times stored in the records are raw output of hd_core.train; they were taken
while other jobs were running on the machine and are NOT used for any claim in the article.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

ART = Path(__file__).resolve().parents[1]             # folder holding code/, data/, figures/
ROOT = ART if (ART / "research_benchmark").is_dir() else ART.parent   # repository root
sys.path.insert(0, str(ROOT / "research_highdim" / "scripts"))

import torch  # noqa: E402
from hd_core import train, baselines, fixed_sets, append_jsonl, load_jsonl  # noqa: E402

DATA = ART / "data"
PARTS = DATA / "s6_highdim_long_d20_parts"
CKPT = DATA / "s6_highdim_long_d20_ckpt"
MERGED = DATA / "s6_highdim_long_d20.jsonl"

D = 20
ITERS = 16_000
SEEDS = [0, 1, 2]
PROBLEMS = ["laplace", "poisson"]
METHODS = ["ritz", "pinn"]
# weights of the main study (research_highdim/results/runs.csv, phase "main")
WEIGHT = {("laplace", "pinn"): 1000.0, ("laplace", "ritz"): 100.0,
          ("poisson", "pinn"): 1000.0, ("poisson", "ritz"): 1.0}


def check_weights():
    """The weights above must be the ones stored for the 4000-iteration d = 20 runs."""
    with open(ROOT / "research_highdim" / "results" / "runs.csv") as f:
        for r in csv.DictReader(f):
            if r["phase"] == "main" and int(r["d"]) == D:
                assert float(r["w"]) == WEIGHT[(r["problem"], r["method"])], r


def run(problem, seed):
    check_weights()
    PARTS.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    out = PARTS / f"{problem}_s{seed}.jsonl"
    done = {(r["problem"], r["method"], r["seed"]) for r in load_jsonl(out)}
    sets = fixed_sets(D)
    for method in METHODS:
        if (problem, method, seed) in done:
            continue
        w = WEIGHT[(problem, method)]
        t0 = time.time()
        net, final, curve = train(problem, method, D, seed, w, iters=ITERS, sets=sets)
        rec = dict(phase="long20", problem=problem, method=method, d=D, seed=seed, w=w, iters=ITERS)
        rec.update(final)
        rec["curve"] = curve
        rec["wall_s"] = time.time() - t0
        rec.update({f"base_{k}": v for k, v in baselines(problem, D, sets).items()})
        rec["torch_version"] = torch.__version__
        torch.save(net.state_dict(), CKPT / f"{problem}_{method}_d{D}_s{seed}_it{ITERS}.pt")
        append_jsonl(out, rec)
        print(f"[long20] {problem:8s} {method:4s} d={D} seed={seed} w={w:g}: rel_l2={final['rel_l2']:.4e} "
              f"centred={final['rel_l2_centered']:.4e} h1={final['rel_h1semi']:.4e} "
              f"val={final['val_rel_l2']:.4e} wall={rec['wall_s']:.0f}s", flush=True)


def merge():
    recs = []
    for problem in PROBLEMS:
        for seed in SEEDS:
            recs += load_jsonl(PARTS / f"{problem}_s{seed}.jsonl")
    recs.sort(key=lambda r: (r["problem"], r["method"], r["seed"]))
    keys = [(r["problem"], r["method"], r["seed"]) for r in recs]
    assert len(set(keys)) == len(keys), "duplicate records"
    with open(MERGED, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"merged {len(recs)} records -> {MERGED.relative_to(ROOT)}")
    missing = [(p, m, s) for p in PROBLEMS for m in METHODS for s in SEEDS if (p, m, s) not in keys]
    print("missing:", missing or "none")


def check():
    """Sanity check of the import path: rerun one stored 4000-iteration cell bit for bit."""
    stored = None
    with open(ROOT / "research_highdim" / "results" / "runs.csv") as f:
        for r in csv.DictReader(f):
            if (r["phase"], r["problem"], r["method"], r["d"], r["seed"]) == ("main", "laplace", "ritz", "2", "0"):
                stored = float(r["rel_l2"])
    _, final, _ = train("laplace", "ritz", 2, 0, 100.0, iters=4000)
    msg = f"stored {stored!r}  rerun {final['rel_l2']!r}  identical: {stored == final['rel_l2']}"
    print(msg)
    with open(DATA / "s6_highdim_long_d20_check.txt", "w") as f:
        f.write("bit-for-bit check of hd_core.train (P1, Deep Ritz, d=2, seed 0, 4000 iterations)\n" + msg + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], int(sys.argv[3]))
    elif cmd == "merge":
        merge()
    elif cmd == "check":
        check()
    else:
        raise SystemExit(__doc__)
