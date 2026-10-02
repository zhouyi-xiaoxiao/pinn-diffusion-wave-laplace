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

"""Resumable driver for PREREG_C1.md. One process, one torch thread; run under nice -n 19.
Every finished run is appended to results/runs_<phase>.jsonl and skipped if its key is present;
network weights go to results/ckpt/. Usage:
  python run_c1.py timing      # not counted: cost per iteration of each arm at d = 20 (seed 99, 100 its)
  python run_c1.py sweep       # not counted: Deep Ritz weight for P3/P4 by the study's protocol (d = 5, seed 100)
  python run_c1.py repro       # plain seeds 0-2 of P1/P2 at d = 20, compared bit for bit with the main study
  python run_c1.py main20      # primary block, seeds 10-14
  python run_c1.py eqcpu       # equal-CPU lift3c (and presolve if its CPU rule requires it)
  python run_c1.py stall       # constant learning rate, 12 000 its, P2, d = 20, Deep Ritz plain/presolve, seeds 10-12
  python run_c1.py rep3        # fan-in control, Deep Ritz, d = 20
  python run_c1.py d10         # secondary block, Deep Ritz, d = 10, seeds 10-12
Addendum PREREG_C1_P5.md (problem P5 "altridge", lift1 pre-registered for P5):
  python run_c1.py sweepP5     # not counted: Deep Ritz weight for P5 (d = 5, seed 100)
  python run_c1.py p5main20    # P5, d = 20, seeds 10-14
  python run_c1.py p5eqcpu     # P5, equal CPU
  python run_c1.py p5dim       # P5, Deep Ritz, d = 5 and 10, seeds 10-12
"""
import sys, os, json, time, glob, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import torch
import c1core as cc
import presolve as ps

RES = os.path.join(ROOT, "results")
CKPT = os.path.join(RES, "ckpt")
os.makedirs(CKPT, exist_ok=True)
MAIN = _repo_path("research_highdim/results")
PNAME = {"P1": "laplace", "P2": "poisson", "P3": "ridge", "P4": "cospair", "P5": "altridge"}
PINN_W = 1000.0
SEEDS = [10, 11, 12, 13, 14]


def ritz_w(problem):
    w = {"laplace": 100.0, "poisson": 1.0}      # the main study's choice (sweep at d = 5, seed 100)
    for name in ("sweep_selection.json", "sweep_selection_p5.json"):
        sel = os.path.join(RES, name)
        if os.path.exists(sel):
            w.update({k: float(v) for k, v in json.load(open(sel)).items()})
    return w[problem]


def weight(problem, method):
    return PINN_W if method == "pinn" else ritz_w(problem)


def key(r):
    return (r["phase"], r["problem"], r["method"], r["d"], r["seed"], r["arm"], r["w"], r["iters"], r["schedule"])


def all_runs(phase=None):
    recs = []
    for f in sorted(glob.glob(os.path.join(RES, "runs_*.jsonl"))):
        recs += cc.load_jsonl(f)
    return [r for r in recs if phase is None or r["phase"] == phase]


def run(phase, problem, method, d, seed, arm, iters=4000, schedule=True, done=None, save=True, log_every=50):
    w = weight(problem, method)
    rec = dict(phase=phase, problem=problem, method=method, d=d, seed=seed, arm=arm, w=w, iters=iters,
               schedule=schedule)
    if done is not None and key(rec) in done:
        return None
    shift, shift_cpu = None, 0.0
    if arm == "presolve":
        c0 = time.process_time()
        shift = ps.affine_minimiser(problem, d, method, w)
        shift_cpu = time.process_time() - c0
        rec["shift_coef"] = [float(v) for v in shift]
    t0 = time.time()
    model, res, curve = cc.train(problem, method, d, seed, w, arm=arm, iters=iters, schedule=schedule,
                                 shift=shift, shift_cpu=shift_cpu, log_every=log_every)
    rec.update(res); rec["curve"] = curve; rec["wall_s"] = time.time() - t0
    rec["load1"] = os.getloadavg()[0]
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if save:
        torch.save(model.state_dict(), os.path.join(
            CKPT, f"{phase}_{problem}_{method}_{arm}_d{d}_s{seed}_w{w:g}_it{iters}_{'sch' if schedule else 'const'}.pt"))
    cc.append_jsonl(os.path.join(RES, f"runs_{phase}.jsonl"), rec)
    if done is not None:
        done.add(key(rec))
    print(f"[{phase}] {problem:8s} {method:4s} {arm:8s} d={d:2d} s={seed:3d} it={iters:5d}: rel={res['rel_l2']:.4e} "
          f"cen={res['rel_l2_centered']:.4e} gamma={res['gamma_end']:.3f} Thalf={res['T_half']} "
          f"cpu={res['cpu_total_s']:.1f}s wall={rec['wall_s']:.0f}s load={rec['load1']:.0f}", flush=True)
    return rec


def round50(x):
    return int(max(50, 50 * round(x / 50.0)))


def eq_iters(problem, method, arm, base_phase="main20"):
    """Equal-CPU iteration count (PREREG_C1.md section 5): from medians over seeds 10-14 of the 4000-it runs."""
    R = [r for r in all_runs(base_phase) if r["problem"] == problem and r["method"] == method and r["iters"] == 4000]
    plain = [r["cpu_train_s"] for r in R if r["arm"] == "plain"]
    other = [r for r in R if r["arm"] == arm]
    if len(plain) < 5 or len(other) < 5:
        return None, None
    mp = statistics.median(plain)
    if arm == "lift3c":
        n = round50(4000 * mp / statistics.median([r["cpu_train_s"] for r in other]))
        return n, dict(median_cpu_plain=mp, median_cpu_arm=statistics.median([r["cpu_train_s"] for r in other]))
    if arm == "presolve":
        tot = statistics.median([r["cpu_total_s"] for r in other])
        info = dict(median_cpu_plain=mp, median_cpu_total_presolve=tot, ratio=tot / mp)
        if tot / mp <= 1.05:
            return 0, info     # 0: the 4000-iteration comparison counts as equal CPU
        sc = statistics.median([r["cpu_shift_s"] for r in other])
        tr = statistics.median([r["cpu_train_s"] for r in other])
        return round50(4000 * (mp - sc) / tr), info
    raise ValueError(arm)


if __name__ == "__main__":
    phase = sys.argv[1]
    done = {key(r) for r in all_runs()}
    if phase == "timing":
        out = []
        for method, arms in (("ritz", ("plain", "presolve", "lift3c", "lift3u", "rep3")), ("pinn", ("plain", "presolve", "lift3c"))):
            for arm in arms:
                r = run("timing", "laplace", method, 20, 99, arm, iters=100, done=done, save=False, log_every=100)
                if r: out.append((method, arm, r["cpu_ms_per_iter"]))
        print(out)
    elif phase == "sweep":
        for problem in ("ridge", "cospair"):
            for w in (1.0, 10.0, 100.0, 1000.0):
                rec = dict(phase="sweep", problem=problem, method="ritz", d=5, seed=100, arm="plain", w=w, iters=4000,
                           schedule=True)
                if key(rec) in done:
                    continue
                t0 = time.time()
                model, res, curve = cc.train(problem, "ritz", 5, 100, w, arm="plain", iters=4000)
                rec.update(res); rec["curve"] = curve; rec["wall_s"] = time.time() - t0
                cc.append_jsonl(os.path.join(RES, "runs_sweep.jsonl"), rec); done.add(key(rec))
                print(f"[sweep] {problem} w={w:g}: val={res['val_rel_l2']:.4e} test={res['rel_l2']:.4e}", flush=True)
        best = {}
        for r in all_runs("sweep"):
            if r["problem"] not in best or r["val_rel_l2"] < best[r["problem"]][1]:
                best[r["problem"]] = (r["w"], r["val_rel_l2"])
        json.dump({k: v[0] for k, v in best.items()}, open(os.path.join(RES, "sweep_selection.json"), "w"), indent=1)
        print("selected:", best)
    elif phase == "repro":
        ref = []
        for f in sorted(glob.glob(os.path.join(MAIN, "runs_ext_s*.jsonl"))):
            ref += [json.loads(l) for l in open(f) if l.strip()]
        ref = {(r["problem"], r["method"], r["seed"]): r for r in ref if r["phase"] == "main" and r["d"] == 20}
        report = []
        for seed in (0, 1, 2):
            for problem in ("laplace", "poisson"):
                for method in ("ritz", "pinn"):
                    rec = run("repro", problem, method, 20, seed, "plain", done=done)
                    if rec is None:
                        rec = [r for r in all_runs("repro") if r["problem"] == problem and r["method"] == method
                               and r["seed"] == seed][0]
                    m = ref[(problem, method, seed)]
                    sd_new = torch.load(os.path.join(CKPT, f"repro_{problem}_{method}_plain_d20_s{seed}_w{rec['w']:g}_it4000_sch.pt"))
                    sd_old = torch.load(os.path.join(MAIN, "ckpt", f"main_{problem}_{method}_d20_s{seed}_w{m['w']:g}_it4000.pt"))
                    same_w = all(torch.equal(sd_new["mlp." + k], v) for k, v in sd_old.items())
                    row = dict(problem=problem, method=method, seed=seed, rel_l2_new=rec["rel_l2"], rel_l2_main=m["rel_l2"],
                               final_loss_new=rec["final_loss"], final_loss_main=m["final_loss"],
                               identical_rel_l2=rec["rel_l2"] == m["rel_l2"],
                               identical_final_loss=rec["final_loss"] == m["final_loss"], identical_weights=same_w)
                    report.append(row); print(row, flush=True)
        ok = all(r["identical_rel_l2"] and r["identical_weights"] for r in report)
        json.dump(dict(all_bit_identical=ok, runs=report), open(os.path.join(RES, "repro_check.json"), "w"), indent=1)
        print("ALL BIT-IDENTICAL:", ok)
        if not ok:
            sys.exit(3)
    elif phase == "main20":
        for seed in SEEDS:
            for problem in ("laplace", "poisson", "ridge", "cospair"):
                for arm in ("plain", "presolve", "lift3c", "lift3u"):
                    run("main20", problem, "ritz", 20, seed, arm, done=done)
                for arm in ("plain", "presolve", "lift3c"):
                    run("main20", problem, "pinn", 20, seed, arm, done=done)
    elif phase == "eqcpu":
        plan = {}
        for problem in ("laplace", "poisson", "ridge", "cospair"):
            for method in ("ritz", "pinn"):
                for arm in ("lift3c", "presolve"):
                    n, info = eq_iters(problem, method, arm)
                    plan[f"{problem}/{method}/{arm}"] = dict(iters=n, **(info or {}))
        json.dump(plan, open(os.path.join(RES, "eqcpu_plan.json"), "w"), indent=1)
        print(json.dumps(plan, indent=1), flush=True)
        for seed in SEEDS:
            for problem in ("laplace", "poisson", "ridge", "cospair"):
                for method in ("ritz", "pinn"):
                    for arm in ("lift3c", "presolve"):
                        n = plan[f"{problem}/{method}/{arm}"]["iters"]
                        if n:   # None: inputs missing; 0: not needed
                            run("eqcpu", problem, method, 20, seed, arm, iters=n, done=done)
    elif phase == "stall":
        for seed in (10, 11, 12):
            for arm in ("plain", "presolve"):
                run("stall", "poisson", "ritz", 20, seed, arm, iters=12000, schedule=False, done=done)
    elif phase == "rep3":
        for seed in SEEDS:
            for problem in ("laplace", "poisson", "ridge", "cospair"):
                run("main20", problem, "ritz", 20, seed, "rep3", done=done)
    elif phase == "d10":
        for seed in (10, 11, 12):
            for problem in ("laplace", "poisson", "ridge", "cospair"):
                for arm in ("plain", "presolve", "lift3c"):
                    run("d10", problem, "ritz", 10, seed, arm, done=done)
    elif phase == "posthoc_lift1":
        # EXPLORATORY, NOT PRE-REGISTERED (later choice, notes/VERIFICATION.md section 4); enters no verdict
        for seed in SEEDS:
            for problem in ("laplace", "poisson", "ridge", "cospair"):
                for method in ("ritz", "pinn"):
                    run("posthoc_lift1", problem, method, 20, seed, "lift1", done=done)
    elif phase == "sweepP5":
        # not counted (PREREG_C1_P5.md section 3)
        for w in (1.0, 10.0, 100.0, 1000.0):
            rec = dict(phase="sweepP5", problem="altridge", method="ritz", d=5, seed=100, arm="plain", w=w,
                       iters=4000, schedule=True)
            if key(rec) in done:
                continue
            t0 = time.time()
            model, res, curve = cc.train("altridge", "ritz", 5, 100, w, arm="plain", iters=4000)
            rec.update(res); rec["curve"] = curve; rec["wall_s"] = time.time() - t0
            rec["load1"] = os.getloadavg()[0]
            cc.append_jsonl(os.path.join(RES, "runs_sweepP5.jsonl"), rec); done.add(key(rec))
            print(f"[sweepP5] w={w:g}: val={res['val_rel_l2']:.4e}", flush=True)
        R = all_runs("sweepP5")
        if len(R) == 4:
            best = min(R, key=lambda r: r["val_rel_l2"])
            json.dump({"altridge": best["w"]}, open(os.path.join(RES, "sweep_selection_p5.json"), "w"), indent=1)
            print("selected:", best["w"], flush=True)
    elif phase == "p5main20":
        assert os.path.exists(os.path.join(RES, "sweep_selection_p5.json"))
        for seed in SEEDS:
            for arm in ("plain", "presolve", "lift3c", "lift3u", "rep3", "lift1"):
                run("p5main20", "altridge", "ritz", 20, seed, arm, done=done)
            for arm in ("plain", "presolve", "lift3c", "lift1"):
                run("p5main20", "altridge", "pinn", 20, seed, arm, done=done)
    elif phase == "p5eqcpu":
        plan = {}
        for method in ("ritz", "pinn"):
            for arm in ("lift3c", "presolve"):
                n, info = eq_iters("altridge", method, arm, base_phase="p5main20")
                plan[f"altridge/{method}/{arm}"] = dict(iters=n, **(info or {}))
        json.dump(plan, open(os.path.join(RES, "eqcpu_plan_p5.json"), "w"), indent=1)
        print(json.dumps(plan, indent=1), flush=True)
        for seed in SEEDS:
            for method in ("ritz", "pinn"):
                for arm in ("lift3c", "presolve"):
                    n = plan[f"altridge/{method}/{arm}"]["iters"]
                    if n:
                        run("p5eqcpu", "altridge", method, 20, seed, arm, iters=n, done=done)
    elif phase == "p5dim":
        for d in (5, 10):
            for seed in (10, 11, 12):
                for arm in ("plain", "lift3c", "lift1"):
                    run("p5dim", "altridge", "ritz", d, seed, arm, done=done)
    else:
        raise SystemExit(__doc__)
