"""Compare the separately written evaluation (out/c3_eval.jsonl, new MC points) and bias values (out/c2_bias.json)
with the package's results; recompute the pre-registered decisions H1-H3 from the new numbers.
Output: out/c5_compare.json and printed summary."""
import json, math, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(H)
recheck = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(H, "out", "c3_eval.jsonl"))}
study = {}
for l in open(os.path.join(PK, "results", "eval_confirm.jsonl")):
    r = json.loads(l)
    if r["set"] == "fresh":
        study[r["file"]] = r
res = {}
# agreement
z_eta = []; rel_eta = []
for f, m in recheck.items():
    t = study[f]
    se = math.hypot(t["se_eta"], t["se_eta"])  # same-size samples: comparable s.e.
    z_eta.append((m["eta"] - t["eta"]) / se); rel_eta.append(m["eta"] / t["eta"] - 1)
res["n"] = len(recheck)
res["eta_rel_diff_range"] = [min(rel_eta), max(rel_eta)]
res["eta_z_absmax"] = max(abs(z) for z in z_eta)
res["err_rel_diff_range"] = [min(recheck[f]["err"] / study[f]["err_abs"] - 1 for f in recheck), max(recheck[f]["err"] / study[f]["err_abs"] - 1 for f in recheck)]
print("n", res["n"], "eta rel diff", res["eta_rel_diff_range"], "max |z|", res["eta_z_absmax"], "err rel diff", res["err_rel_diff_range"])
# H1
res["H1_min_z"] = min(m["z"] for m in recheck.values()); res["H1_min_eta"] = min(m["eta"] for m in recheck.values())
res["H1_violations"] = [f for f, m in recheck.items() if m["bound"] < m["err"] - 3 * math.hypot(m["se_err"], m["se_bound"])]
print("H1 min z", res["H1_min_z"], "min eta", res["H1_min_eta"], "violations", res["H1_violations"])
# H2
inv = {r["file"]: r for r in json.load(open(os.path.join(PK, "results", "inventory.json")))["rows"]}
tested = [m for f, m in recheck.items() if inv[f]["confirmatory"] and m["d"] >= 5]
fails = sorted([(m["eta"], m["share"], m["file"]) for m in tested if m["eta"] > 3], reverse=True)
passes = [(m["eta"], m["share"], m["file"]) for m in tested if m["eta"] <= 3]
res["H2"] = dict(n_tested=len(tested), n_fail=len(fails), eta_max=max(m["eta"] for m in tested),
                 fail_share_range=[min(x[1] for x in fails), max(x[1] for x in fails)],
                 pass_share_range=[min(x[1] for x in passes), max(x[1] for x in passes)],
                 fails=[x[2] for x in fails])
# "exactly the networks with the smallest shares"?
order = sorted(tested, key=lambda m: m["share"])
first = {m["file"] for m in order[:len(fails)]}
res["H2"]["fails_equal_smallest_share_set"] = first == {x[2] for x in fails}
print("H2", {k: v for k, v in res["H2"].items() if k != "fails"})
# H3 (primary: main, seeds 0-2)
h3 = {}
for prob in ("laplace", "poisson"):
    for meth in ("pinn", "ritz"):
        means = []
        for d in (2, 3, 5, 10, 20):
            v = [m["share"] for m in recheck.values() if m["family"] == "main" and m["problem"] == prob and m["method"] == meth and m["d"] == d]
            means.append(float(np.mean(v)))
        h3[f"{prob}/{meth}"] = dict(means=means, non_decreasing=all(b >= a for a, b in zip(means, means[1:])))
res["H3"] = h3
for k, v in h3.items():
    print("H3", k, [round(x, 3) for x in v["means"]], v["non_decreasing"])
# eta by d (all networks) for the summary
for d in (2, 3, 5, 10, 20):
    e = [m["eta"] for m in recheck.values() if m["d"] == d]
    print("d", d, "eta range", round(min(e), 3), round(max(e), 3))
res["eta_by_d"] = {d: [min(m["eta"] for m in recheck.values() if m["d"] == d), max(m["eta"] for m in recheck.values() if m["d"] == d)] for d in (2, 3, 5, 10, 20)}
# bias: recheck vs package table
bt = json.load(open(os.path.join(PK, "results", "bias_table.json")))["rows"]
mb = {(r["d"], r["beta"]): r["rel_bias"] for r in json.load(open(os.path.join(H, "out", "c2_bias.json")))["bias"]}
rr = [mb[(r["d"], float(r["beta"]))] / r["rel_bias"] - 1 for r in bt if (r["d"], float(r["beta"])) in mb]
res["bias_rel_diff_range"] = [min(rr), max(rr)]; res["bias_cells"] = len(rr)
print("bias cells", len(rr), "rel diff", res["bias_rel_diff_range"])
json.dump(res, open(os.path.join(H, "out", "c5_compare.json"), "w"), indent=1)
