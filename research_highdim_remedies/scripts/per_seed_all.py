"""Write results/per_seed_all.csv: one row per run of every phase that is reported (pre-registered,
exploratory and addendum), with its status and the seed-paired ratio err(plain)/err(arm).

No training; reads results/runs_*.jsonl only. Pairing: same problem, method, d, seed and schedule, with the
plain run of the matching phase family (main20 for main20/eqcpu/posthoc_lift1; p5main20 for p5main20/p5eqcpu;
d10, p5dim and stall within themselves). The repro runs (seeds 0-2) are listed without a ratio; their
comparison with the main study is results/repro_check.json. Runs that do not count (timing, sweep, sweepP5)
are not listed.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

PID = {"laplace": "P1", "poisson": "P2", "ridge": "P3", "cospair": "P4", "altridge": "P5"}
STATUS = {
    "repro": "prereg_C1 (reproduction of the main study)",
    "main20": "prereg_C1",
    "eqcpu": "prereg_C1",
    "stall": "prereg_C1",
    "d10": "prereg_C1 (secondary)",
    "posthoc_lift1": "exploratory (later choice, notes/VERIFICATION.md section 4)",
    "p5main20": "addendum_P5 (not blind)",
    "p5eqcpu": "addendum_P5 (not blind)",
    "p5dim": "addendum_P5 (not blind)",
}
FAMILY = {"main20": "main20", "eqcpu": "main20", "posthoc_lift1": "main20",
          "p5main20": "p5main20", "p5eqcpu": "p5main20",
          "d10": "d10", "p5dim": "p5dim", "stall": "stall"}
PHASES = ["repro", "main20", "eqcpu", "stall", "d10", "posthoc_lift1", "p5main20", "p5eqcpu", "p5dim"]

runs = []
for ph in PHASES:
    with open(os.path.join(RES, f"runs_{ph}.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            assert r["phase"] == ph
            runs.append(r)

plain = {}
for r in runs:
    if r["arm"] == "plain" and r["phase"] in FAMILY and FAMILY[r["phase"]] == r["phase"]:
        key = (r["phase"], r["problem"], r["method"], r["d"], r["seed"], r["schedule"])
        assert key not in plain, key
        plain[key] = r

cols = ["phase", "status", "problem_id", "problem", "feature_matched_for_lift", "method", "arm", "d", "seed",
        "w", "iters", "schedule", "budget", "rel_l2", "rel_l2_centered", "ratio_plain_over_arm",
        "plain_rel_l2", "cpu_total_s", "plain_cpu_total_s", "T_half", "gamma_end", "load1"]
rows = []
for r in runs:
    ph = r["phase"]
    budget = {"eqcpu": "equal CPU", "p5eqcpu": "equal CPU", "stall": "12000 it, constant lr",
              "repro": "4000 it (seeds 0-2)"}.get(ph, "4000 it")
    p = None
    if ph in FAMILY:
        p = plain.get((FAMILY[ph], r["problem"], r["method"], r["d"], r["seed"], r["schedule"]))
    ratio = "" if (p is None or r["arm"] == "plain") else p["rel_l2"] / r["rel_l2"]
    rows.append({
        "phase": ph, "status": STATUS[ph], "problem_id": PID[r["problem"]], "problem": r["problem"],
        "feature_matched_for_lift": "yes" if r["problem"] == "poisson" else "no",
        "method": r["method"], "arm": r["arm"], "d": r["d"], "seed": r["seed"], "w": r["w"],
        "iters": r["iters"], "schedule": r["schedule"], "budget": budget,
        "rel_l2": r["rel_l2"], "rel_l2_centered": r["rel_l2_centered"], "ratio_plain_over_arm": ratio,
        "plain_rel_l2": "" if p is None else p["rel_l2"], "cpu_total_s": r["cpu_total_s"],
        "plain_cpu_total_s": "" if p is None else p["cpu_total_s"], "T_half": r.get("T_half"),
        "gamma_end": r.get("gamma_end"), "load1": r.get("load1"),
    })
# every non-plain, non-repro row must be paired
missing = [(x["phase"], x["problem"], x["method"], x["arm"], x["d"], x["seed"]) for x in rows
           if x["arm"] != "plain" and x["phase"] != "repro" and x["ratio_plain_over_arm"] == ""]
assert not missing, missing
rows.sort(key=lambda x: (PHASES.index(x["phase"]), x["problem_id"], x["method"], x["d"], x["arm"], x["seed"]))
out = os.path.join(RES, "per_seed_all.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print(out, len(rows), "rows")
