# How the results were checked

This note describes how the results of the article *Physics-informed neural networks and the Deep Ritz
method for diffusion, wave and Laplace problems: a reproducible benchmark and its failure modes*
(`paper.pdf`, Supplementary Material `supplement.pdf`) were checked, what each check found, and where its
scripts and outputs are. The article is authoritative; where this note and the article differ, the article
holds. Sections and tables with the prefix S are those of the Supplementary Material.

## Contents

1. How the checks were made
2. Verification of the low-dimensional benchmark (`research_benchmark/`)
3. Verification of the study in d dimensions (`research_highdim/`)
4. The stability constant, the error bound and the two remedies (`research_highdim_bound/`, `research_highdim_remedies/`)
5. Where the numbers are

---

## 1. How the checks were made

**Verification codes.** The low-dimensional benchmark and the study in d dimensions were verified by
re-aggregation of every stored run record, by re-runs with two separately written codes (called V-bench
and V-dim in the article), by re-derivation of the finite-difference errors, and by adversarial tests of the
main claims. A check ends in one of four outcomes: *confirmed*, *partially confirmed* (confirmed with a
restriction, stated below), *refuted*, or *not verifiable*. No key result was refuted.

**Pre-registration and re-checks.** The analyses of Sections 6.7 and 6.8 were pre-registered and then
re-checked twice with separately written code (section 4).

**Article-level checks.** `code/check_headline_numbers.py` recomputes the numbers of the abstract and of
Section 8 from the stored results; `code/check_tables.py` and the `--check` options of the table scripts
verify that every generated table in the LaTeX sources equals the stored results;
`code/sA_proofs_checks.py` checks the computational steps of the proofs symbolically and numerically;
`code/sC_details_check.py` checks the numbers quoted in Supplementary Section S8.

**Limits.** All checks belong to the same project and ran on the same shared 12-core machine, which was
heavily loaded throughout. They used separately written code and other seeds, but they are checks of the
code and of the analysis, not a replication by a third party; errors common to both sides (a shared
misreading of a problem, the single-precision library stack) are not excluded.

**Where the evidence is.**

| Folder | Content |
|---|---|
| `verify_research_benchmark/` | verification of the low-dimensional benchmark (V-bench: `vpinn.py`) |
| `verify_research_highdim/` | verification of the study in d dimensions (V-dim: `scripts/v_core.py`) |
| `research_highdim_bound/check/`, `research_highdim_remedies/check/` | re-checks of the analyses of Sections 6.7 and 6.8 |
| `data/checks/` | the numbers that the article takes from the records of the checks |

Trained networks are not included, also not the 88 networks on which the bound of Section 6.7 was
evaluated; their SHA-256 hashes, recorded before the evaluation, are in
`research_highdim_bound/results/inventory.json`. The scripts that need trained networks therefore do not run as
they are; their stored outputs are included, and the training scripts recreate the networks (`reproduce.md`).

---

## 2. Verification of the low-dimensional benchmark

### 2.1 What was checked

- **Re-aggregation.** `benchmark_runs.csv` equals the raw run files (350 rows, largest difference 0);
  every mean, standard deviation, geometric mean, paired ratio, count and p-value of the benchmark tables
  was recomputed and agrees to the printed digits (`verify_research_benchmark/v_stats.py`,
  `v_json.py`, `v_summary.py`). The median held-out error of the 25 Wave2 runs, 7.24e-2 after 1000 L-BFGS
  iterations and 4.58e-2 after 1500, and the range of the 50 Wave1 runs, 0.414 to 0.512, were recomputed.
- **Separately written code (V-bench, `vpinn.py`).** One network at a time on the CPU, derivatives
  propagated forward through the layers, the L-BFGS of PyTorch (strong Wolfe), same protocol and seeds
  where the configuration is shared; 40 configurations, among them a node-centred grid control
  (`grid-n`: a closed 16 x 16 grid that includes t = 0 and x = 0, 1).
- **Finite differences** re-derived from closed-form solutions of the discrete schemes and from a
  separate sparse solve (`v_fd.py`, `v_fd.out`).
- **Literature.** The two-mode wave problem Wave1 is that of Wang, Yu and Perdikaris (arXiv:2007.14527,
  Sec. 7.3, eqs. (7.5) to (7.9)); the caption of their Fig. 6 gives 4.518e-01 for the plain PINN after
  80 000 iterations with a five-layer network of width 500.

### 2.2 Key results of the benchmark and the outcome of the verification

KR1, KR2, ... number the key results (KR = key result).

| Code | Key result | Outcome |
|---|---|---|
| KR1 | Adam followed by L-BFGS beats Adam at a matched iteration count in 175 of 175 pairs | partially confirmed (39 of 40 in V-bench) |
| KR2 | No strategy is uniformly best; the cell-centred grid is worst on Heat and best on Lap2 | partially confirmed (items 1 and 5 below) |
| KR3 | Sobol' points are never worse than random points in the mean; RAD gives no significant gain | partially confirmed (not seed by seed) |
| KR4 | Per-iteration resampling hurts Adam-only training and is neutral after L-BFGS | partially confirmed (item 6) |
| KR5 | The strategy matters only when points are scarce | confirmed |
| KR6 | Wave2 and Lap3 are solved to a few per cent at 3000 iterations, not converged | confirmed |
| KR7 | The unit-weight PINN does not solve the two-mode wave problem | confirmed |
| KR8 | Finite differences are much faster at equal or better accuracy | partially confirmed (item 7) |
| KR9 | The stacked multi-network driver is equivalent to single-network training | equivalent in exact arithmetic; in floating point see Supplementary Sections S8.2 and S8.4 (caveats 3 and 4 below) |

The total compute of the study (about 95 minutes, section 2.6) could not be verified after the fact.

### 2.3 Points on which the verification restricts a statement

1. **The heat penalty belongs to the cell-centred grid (KR2).** A closed node-centred grid is not
   distinguishable from random points in V-bench (six seeds: ratio 1.26, better in 2 of 6, p = 0.44;
   node-centred against cell-centred 0.32). The cell-centred grid leaves 0 < t < 1/32 without points;
   the control experiment of Section 4.3 of the article establishes this mechanism directly.
2. **L-BFGS won 39 of 40 pairs in V-bench (KR1).** The exception is Heat, grid-c, seed 1 (Adam 1.303e-2
   against 1.316e-2), which is also the closest pair of the benchmark on Heat (ratio 0.86).
3. **Tests per seed (KR1).** The strategies of one seed share their initial weights, so the optimiser arms
   are tested per seed: p = 2/1024 (10 seeds) and 0.0625 (5 seeds), the smallest attainable values.
4. **The Adam arm ends on a noisy iterate (KR1).** Constant-rate Adam fluctuates by up to an order of
   magnitude between iterations 1500 and 3000; an Adam with exponential decay (1e-3 to 1e-5) did not close
   the gap to the L-BFGS arm (Heat 2.0e-2 to 3.1e-2, 0.22 to 1.00 times the constant-rate error of the same
   runs; Lap2 1.7e-2 to 2.3e-2, 2.3 to 5.3 times worse; Supplementary Section S3.1). The article compares with the last, the median and the
   best logged Adam iterate.
5. **The effects on Lap2 are weaker in V-bench (KR2, KR3).** Grid against random 0.75 and Sobol' against
   random 0.84 on three seeds, against 0.49 and 0.54 in the benchmark; the direction holds.
6. **Resampling (KR4).** The resampling penalty under Adam alone on Lap2 is 6.36 in the benchmark and 2.7
   in V-bench (three of three seeds); after the L-BFGS phase on a fixed set V-bench gives 1.56 on three
   seeds, so the neutrality is established only up to the spread of a few seeds.
7. **Finite differences are three to five orders of magnitude faster than a single network (KR8).** The
   single-network timing of the study was taken at a load average of 95 to 167; V-bench ran an
   Adam-then-L-BFGS arm in 20 to 83 s at a load average of about 20 to 45. The accuracies of the finite
   differences were reproduced, three of four to the printed digits. The stacked GPU time per network gives
   the smallest ratio, 4.8e2.
8. **Discrete and continuous norms.** The relative distance 1.353 between the decaying mode exp(-t) S and
   the solution of Wave2 (Section 7.1) is a discrete norm on the 21 time levels of the evaluation grid; the
   continuous value is 1.380.

### 2.4 Checks of the demonstrations of Section 7

- `code/s7_pitfalls.py` executes both differentiation patterns and the operator test on an exact solution
  and on a non-solution in double precision; `code/check_headline_numbers.py` checks that the PDE term of
  the training run is exactly zero at every iteration, in the run of 150 passes and in that of 1500.
- The proofs of the two propositions of Section 7 are checked symbolically by `code/sA_proofs_checks.py`
  (block `nonunique_notwave`).
- The demonstration of Section 7.2 (`research_benchmark/scripts/run_oneface.py`, seeds 0 to 4) was re-run
  for seeds 0 to 2 with the network and derivative code of V-bench (`verify_research_benchmark/v_oneface.py`,
  `v_oneface.json`): relative distances from the solution of Lap3 1.589, 12.84 and 13.36, against 1.616, 12.51
  and 13.22 in the stored runs, with both loss terms between 4.0e-4 and 8.7e-4. The demonstration of Section
  7.1 rests on one code (`code/s7_pitfalls.py`).

### 2.5 Caveats of the benchmark

1. **Budgets.** 256 to 1024 interior points and 3000 iterations per arm; the planned 1024 and 4096 points
   with 5000 iterations would have taken more than two hours on the loaded machine. Strategy effects belong
   to the scarce-point regime.
2. **Statistics.** Five seeds for Wave1, Wave2 and Lap3, for which the paired test cannot go below
   p = 0.0625; the p-values are not corrected for multiple comparisons, and the rule for calling an effect
   significant was fixed after the results were known.
3. **L-BFGS of the stacks.** The multi-seed runs use a stacked L-BFGS with Armijo backtracking, whose
   final errors were within 4 to 10 per cent (geometric mean) of those of `torch.optim.LBFGS` with a
   strong-Wolfe line search in the validation.
4. **Hardware.** The stacks ran in single precision on the Apple GPU backend; the result of one network
   depends on the size of its stack (Supplementary Section S8).
5. **Wall-clock time** is good to an order of magnitude only (load average 50 to 170 during the study).
6. **Wave2 is not converged and Wave1 is not solved**; no loss weighting, causal training, Fourier
   features or hard constraints were tried.
7. **Protocol.** Constant learning rate for Adam, no weight decay, no gradient clipping, unit loss weights.
8. **Lap3** errors are measured on an interior cell-centred 40^3 grid because the exact solution is
   discontinuous along two edges.
9. **Wave1** is a benchmark from the literature, run here with a 3 x 64 network.

### 2.6 Records of the study

- **Software at run time:** Python with PyTorch 2.14.1, NumPy 2.5.3, SciPy 1.18.1, matplotlib 3.11.2.
- **Validation of the stacked driver** (`scripts/validate_batched.py`, `results/validate_batched.json`,
  CPU): the losses of the stacked and the single-network Adam agree exactly at iteration 1, to 6.6e-8 at
  iteration 50 and to 1.5e-7 at iteration 100; the stacked L-BFGS ends within a geometric-mean factor
  1.10 (Heat) and 1.04 (Lap2) of the reference. The validation was done before the loss was re-assembled
  into a single forward pass; the re-assembly itself was checked afterwards (`checks/verify_losses.out`:
  largest relative difference 0.00e+00 for every term of every problem).
- **Single forward pass.** The first Heat block was produced with separate forward passes for the
  residual and each constraint set and is kept in `results/runs_superseded/`; the stored benchmark blocks
  use the single-pass loss, which avoids repeated recompilation on the GPU backend. For equal seeds the
  10-seed Heat run and the first block have identical losses at iteration 1 and final errors in the ratio
  0.98 in geometric mean (Supplementary Section S8.4); the first block enters no result.
- **Compute (approximate, training and solving only):** about 95 min in all, including pilots, the
  validation of the stacked driver, the benchmark blocks, the single-network timing, the budget sweep, the
  runs of Section 7.2 and 12 CPU minutes of heat runs with other networks, optimiser settings and point
  budgets, outside the design of the benchmark and not used in the article
  (`data/checks/study_records.json`).
  This total cannot be re-measured; the stored stack timings (241 to 790 s per stack) are consistent with it.
- **Compute of the verification:** about 59 min of training, every run under 3 min.

---

## 3. Verification of the study in d dimensions

### 3.1 What was checked

| Check | Content | Output |
|---|---|---|
| V1 | Exact solutions, a nested Laplacian and both boundary samplers written separately; the study's forward Laplacian equals the nested one to 1e-16 in double precision for d = 2 to 20; single-precision losses agree to 7 to 9 digits | `scripts/v_check.py`, `results/v_check.txt` |
| V2 | Constant and affine floors derived analytically (affine floor of LapD at d = 2, 3, 5, 10, 20: 0.250, 0.250, 0.200, 0.1387, 0.1015; of PoiD: 0.1203 for every d) | analytic |
| V3 | Re-aggregation of the 88 run records: every table entry, ratio, slope and fit; 5051 s summed wall-clock time, 2539 s recorded CPU time, longest run 255 s | `scripts/v_agg.py`, `results/v_agg.txt` |
| V4 | The 88 stored networks re-evaluated on a separate set of 50 000 test points: largest change 1.4 per cent, no ordering changed | `scripts/v_reeval.py`, `results/v_reeval.txt` |
| V5 | Finite-difference solve of the Robin problem of the penalised Deep Ritz energy at d = 2: distance 0.288, 5.00e-2, 5.48e-3, 5.54e-4 from the exact solution for beta = 1, 10, 100, 1000 | `scripts/v_robin_fd.py`, `results/v_robin_fd.json` |
| V6, V16 | Seven runs repeated with the study's code reproduce the stored errors bit for bit, among them a 16 000-iteration run and an equal-time run; their CPU time was 1.1 to 1.8 times the stored value | `scripts/v_determinism*.py`, `results/v_determinism.txt` |
| V7 | The problem LapD at d = 10 compared with Sec. 3.2 of E and Yu (2018) (`refs.bib`, entry eyu2018) | (this note) |
| V8 | Decomposition of the d = 20 PoiD networks into affine and non-affine parts | `scripts/v_affine.py`, `scripts/v_resid.py` |
| V9 | Deep Ritz at d = 2 trained with V-dim for beta = 1, 10, 100, 1000 against the Robin minimiser of V5 | `results/v_summary.txt` (rows `robin`) |
| V10 | Networks trained with V-dim (nested Laplacian, seeds 7 to 9, and 7 to 11 for Deep Ritz on LapD at d = 20) at d = 2, 10, 20 and at equal compute | `results/v_runs_*.jsonl`, `results/v_summary.txt` |
| V11 | Penalty weight of Deep Ritz on PoiD at d = 10 and 20 | `results/v_summary.txt` (rows `beta`) |
| V12 | Spot checks of the weight sweep at d = 5 (seed 7) | `results/v_summary.txt` (rows `sweep`) |
| V13 | PINN with lambda = 1e4 at d = 10 | `results/v_summary.txt` (rows `lam1e4`) |
| V14 | Cost per iteration re-measured at a load average of about 20 | `scripts/v_bench.py`, `results/v_bench.txt` |
| V15 | Compute of the verification: 73 runs, 2896 s of training CPU time (4469 s wall-clock, longest run 238 s), plus about 5 min for the repetitions and the cost benchmark | `results/v_summary.txt` |

### 3.2 Key results of the study and the outcome of the verification

| Code | Key result | Outcome |
|---|---|---|
| KR1 | At equal iterations the PINN is more accurate for every d <= 10 | confirmed |
| KR2 | At a fixed budget the error grows with d for both methods | confirmed |
| KR3 | At d = 20 on PoiD neither method beats the best affine function | partially confirmed (the Deep Ritz networks are nearly affine, the PINN networks are not; item 2) |
| KR4 | The growth with d is partly a budget effect | confirmed |
| KR5 | Deep Ritz cost is nearly flat in d, PINN cost linear | partially confirmed (item 4) |
| KR6 | At equal CPU time at d = 10 Deep Ritz is more accurate | confirmed (item 5) |
| KR7 | Deep Ritz is sensitive to beta on LapD and insensitive on PoiD; the PINN only mildly sensitive | partially confirmed (holds at d = 5 only; item 3) |

### 3.3 Points on which the verification restricts a statement

1. **Floors (V2).** The analytic floors agree with the Monte-Carlo values of the study (Section 2 and
   Supplementary Section S1 of the article).
2. **Mechanism at d = 20 (KR3, V8).** Deep Ritz has a non-affine part of only 0.18 to 0.20 times the
   true one (residual 0.90 of the source term). The PINN's non-affine part is 0.83 to 0.84 times the true one
   in size but correlates only 0.14 to 0.16 with it, and its residual is 0.35 to 0.37 of the source term: its error is
   a mis-shaped, nearly harmonic component that the boundary data do not pin down.
3. **Penalty weights at d = 10 (KR7, V11, V13).** On PoiD at d = 10 Deep Ritz with beta = 10 or 100
   stays on the affine plateau (0.119 to 0.126, against 0.071 to 0.096 with beta = 1), and the PINN with
   lambda = 1e4 does the same (0.111 to 0.120, against 0.021 to 0.026 with lambda = 1e3). The weight
   statements are statements about d = 5; larger weights do not help at d = 10.
4. **Cost (KR5, V14).** Process CPU time depends on the load on this machine (V6). At lower load Deep
   Ritz takes 4.8 and 5.1 ms per iteration at d = 50 and 100 instead of 8.0 and 8.4, so it is flatter in
   d, and the ratio of the forward-Laplacian PINN to Deep Ritz at d = 100 is about 13 rather than 10.7.
   Both measurements are given in the article.
5. **Equal compute (KR6).** The CPU seconds of runs made under different load are not comparable, so no
   statement is made about the CPU time actually used; with 9000 instead of 10 000 iterations on LapD the
   reversal still holds (9.24e-3, 9.41e-3, 9.65e-3, all below every PINN run). That the advantage of Deep
   Ritz per unit of compute widens with d is not shown and is not claimed.
6. **Seed spread (KR2).** Three seeds understate the spread: with seed 7 at d = 20 on LapD both methods
   end 19 to 23 per cent below the lower ends of the study's ranges, and five Deep Ritz seeds have a standard deviation
   about twice that of the study's three. The article gives the V-dim seeds.
7. **The Robin bias at d = 2 (V5, V9).** With beta = 100 the penalised minimiser is 5.48e-3 away from the
   exact solution, close to the PINN's whole error at d = 2; trained Deep Ritz networks land on the
   minimiser for beta = 1 and 10 (0.290 and 4.78e-2) and above it for beta = 100 and 1000.

### 3.4 Caveats and records of the study

- Three seeds per cell; comparative statements are limited to cells where the seed ranges do not overlap.
- PoiD has zero normal derivative on every face and favours the penalised Deep Ritz method; LapD is the
  representative case. Both solutions are simple in structure.
- The penalty weights were chosen at d = 5 with one run per value; the PINN optimum lies at the edge of
  the grid (lambda = 1000); the Deep Ritz choice beta = 1 on PoiD rests on a difference of 0.7 per cent.
- One architecture (three hidden layers of 64, tanh) and one learning-rate schedule for all d. The longer
  schedule was run for LapD at d = 10 and for both problems at d = 20.
- Boundary conditions are imposed by penalty only; dimensions above 20 were benchmarked for cost only.
- **Deviations from the plan, all because of the load:** one PyTorch thread per process instead of four
  (measured faster under load: at d = 10 the forward-Laplacian PINN took 8.9 ms per iteration with one
  thread and 18.7 ms with four); three single-thread workers, one per seed, in the main and extension
  phases; the PINN Laplacian of the main runs computed by forward propagation (the same loss, check V1).
  The first 13 of the 16 runs of the weight sweep predate these changes: they used four threads, carry no
  CPU timer, and the five PINN runs among them used the nested Laplacian.
- **Compute:** 88 runs, 5051 s of summed wall-clock time (inflated by the load, up to three workers in
  parallel), 2539 s of recorded training CPU time (without the 13 early sweep runs), longest run 255 s.
- **Load during the cost benchmark** (`results/cost_vs_d.csv`): load average about 90 to 110; a first run
  at 130 to 166 (`results/cost_vs_d_run1.csv`) gave the same ordering with times 1.5 to 2 times larger.

---

## 4. The stability constant, the error bound and the two remedies

The analyses of Sections 6.7 and 6.8 rest on two studies with a pre-registration, each re-checked twice
with separately written code within the same project, on the same machine (not a replication by a third
party). This section records the pre-registrations, the choices made after them and the re-checks. The
article states every result with the status these checks left it.

### 4.1 How the two studies were run

* **Pre-registration.** Before the first counted evaluation or run, the hypotheses, the networks or problems,
  the seeds, the budgets, the metric and the decision rules were written down, and the SHA-256 hash of the
  document was recorded with the time. The documents are released unchanged (their hashes below can be
  recomputed). Every choice made later was recorded with its time; sections 4.2 and 4.3 list them.
* **Re-checks.** Each study was re-checked in two rounds. Each round read the proofs, recomputed the quoted
  numbers from the stored files, and re-evaluated or retrained networks with separately written code and other
  seeds, among them networks for problems outside the pre-registered set. The plans of the re-runs
  of the remedies were written and hashed before their runs; they are summarised below. The scripts and
  outputs of the re-checks are in the `check/` folders of the two studies.
* **Outcome.** No gap was found in the proofs. Every number in the article is in, or recomputed from, a stored
  file. The results of the re-checks are reported in the article (Section 6.7, Supplementary Sections S5.6 and
  S8.5).

### 4.2 Stability constant and computable error bound (`research_highdim_bound/`)

**Pre-registration.** `PREREG_T2.md` (SHA-256 `1a0701517b8ea9739f6daac1a3ae2b86c8bfdd8c3059053e5043a300c3925c3a`)
and the index `PREREG.md` (`95b10ea75987e9ee34ff2fe891a2c33e1d249b1365cf6517d13ec52387a692d6`), frozen on
2026-10-01 at 18:00:07 BST together with the evaluation script `scripts/s4_eval_bound.py`
(`c95c4b1e3bd5f6a2a56d5b0f8f36011f146ef2088d83b3b94b840aa94f64c7e4`) and the list of networks
`results/inventory.json` (`6ee405b09aa5d1068a99a84ce1c8a7e6aa6fa566db59737ad2d22a5d2d9d9400`). Only the
evaluation of the 13 networks of an earlier pilot (`results/eval_pilot.jsonl`) existed at that time; the
confirmatory evaluation (`results/eval_confirm.jsonl`) ran after it. The proofs and the exact penalty bias
need no pre-registration (no network is involved). The two documents, the evaluation script and the list of
networks are published byte for byte; the published `src/t2common.py` differs from its frozen state by the helper
of item 2 below and by paths made relative to the repository root.

**Choices made after the freeze.**

1. The header of `PREREG_T2.md` gives the time of writing as 18:00 to 18:10; the document was complete and
   frozen at 18:00 BST, so the end time in the header is too late. Nothing else is affected.
2. A helper for writing JSON files was added to `src/t2common.py` after the freeze (its frozen hash
   `d72f5dd0767a763e69a328c28d9417c258d792e73b69f6a760891fb4c559f585` therefore no longer matches). The
   confirmatory evaluation had finished before (18:02:48) and does not use the helper.
3. The post-hoc conjecture that the networks failing H2 are exactly those with the smallest boundary shares
   does not hold: the share ranges of failures (0.219 to 0.5741) and passes (0.5726 to 0.986) overlap
   (`results/posthoc_analyses.json`, H2_share). No explanation of the failures is offered.
4. In the computation of the restricted Rayleigh-Ritz lower bounds, the last configuration (d = 100, two modes
   per coordinate, K = 21) gave 0.014193, below the one-mode value 0.020389, which is therefore the value at
   d = 100 (`results/restricted_lanczos.json`).
5. The values of d c(d) for the conjecture on the penalty bias come from a quadrature range that includes the
   small-t part of the integrand (a range without it gives 1.5757 instead of 1.6329 at d = 10 000), with an
   extrapolation in the number of one-dimensional roots (`scripts/s10_dc_large_d.py`,
   `s10b_dc_quadrature_check.py`, `s10c_dc_truncation.py`; `results/dc_truncation.json`). With the same wider
   range the 49 entries of the bias table change by at most 1.0e-8 relative for beta <= 1000
   (`results/exact_bias_wide_compare.json`).

**First re-check** (`check/c1_theory.py` to `check/c5_compare.py`, outputs in `check/out/`). The proofs
re-read; the constant from the harmonic side by Rayleigh-Ritz (`c1b_harmonic_basis.py`); the exact bias
re-implemented, with a Chebyshev collocation check at d = 2, 3 (`c2_bias.py`); all 88 networks re-evaluated
with a separately written evaluator, the Laplacian from the trace of the Hessian and new Monte-Carlo points:
the efficiency agreed to within -0.78 to +0.61 per cent and the decisions on H1 to H3 were the same
(`c3_eval.py`, `c5_compare.py`); 24 networks retrained with seeds 11 to 13, including a harmonic problem outside
the study: the bound held in all (`c4_fresh_train.py`, `run_c4.sh`).

**Second re-check** (`check/r2_*.py`, outputs in `check/out_r2/`). The proofs re-read line by line, without a
gap; that L_d improves on 1/(2d) from d = 9 on was checked (`r2_thm2_edges.py`). A Rayleigh-Ritz computation on
the harmonic side for the unit square gives kappa_2^2 >= 0.2877121, and the computed value of Antunes and
Gazzola (2013) was added to the comparison (`r2_rr_quadrature.py`). d c(d) was computed without mode
truncation up to d = 10^6, and a leading-order Laplace evaluation gives the limit sqrt(8/3); the statement
remains a conjecture because the asymptotic argument is not written out with error terms (`r2_dc_theta.py`).
21 further networks were retrained with another separately written trainer and seeds 31 to 33, including
Deep Ritz at the CPU time of the PINN at d = 20: the bound held in all, ranked all 9 pairs correctly at equal
CPU time and only 4 of 9 at equal iterations (`r2_fresh_train.py`).

### 4.3 Two remedies at d = 20 (`research_highdim_remedies/`)

**Pre-registration.** `PREREG_C1.md` (SHA-256
`5d110a6505f87d8ba1b98eb85180386eb74d969876f948c1004c40be90506dff`), frozen on 2026-10-01 at 17:59:54 BST,
before the first counted run (the cost measurement `results/runs_timing.jsonl` and the selection of the Deep
Ritz weights `results/runs_sweep.jsonl` precede it and do not count). The addendum for the fifth problem,
`PREREG_C1_P5.md` (`6a5a361db64ad5dd6ac4f2bfcb27f1484be075725221f8c4ef885d2002ee5c58`), was frozen at
20:12:30, before any run of that problem with the study's code; its outcome was known in direction from the
first re-check, which the addendum and the article state. `PREREG.md` is an index.

**Pilots before the freeze.** Exploratory pilots with seeds 0 to 2 (not part of the study and not published;
no result of the article rests on them) had tried both remedies on LapD and PoiD at d = 20 before
`PREREG_C1.md` was written. On PoiD the Legendre lift gave test errors 1.90e-2, 1.90e-2 and 1.92e-2 for Deep Ritz
(seeds 0 to 2) and 9.34e-3 for the PINN (seed 0), against 0.126 and 0.144 for the plain network; the affine
pre-solve gave 0.1265 against 0.126. On LapD the lift was better than plain in 3 of 3 seeds (factor 2.1 to 2.9 for
Deep Ritz, 2.6 to 3.3 for the PINN), the centred input and rep3 gave Deep Ritz no clear gain, and the affine
pre-solve gained a factor 2.3 to 2.5 (two seeds). The centred input had also been tried on PoiD for d = 4 to 16.
The direction of both remedies on LapD and PoiD was therefore known before the freeze; only RidgeD and CosPairD
were held out (a related ridge had been used in a theory pilot). The article states this in Sections 6.8 and 8.3
and Supplementary Section S5.9.

**Choices made after the freeze.**

1. (After the main runs, by the frozen rule of section 5 of `PREREG_C1.md`.) The CPU condition for counting the
   4000-iteration presolve runs as equal-CPU runs failed only for Deep Ritz on RidgeD (ratio 1.082, caused by
   variations of the per-iteration CPU time on the shared machine); that cell was rerun with 3700 iterations
   (`results/eqcpu_plan.json`).
2. (After the frozen analysis, written before the runs.) Because the reparametrisation control rep3 gained as
   much as the lift on three problems, an exploratory arm lift1, the plain network with the centred input
   2x - 1, was added on LapD, PoiD, RidgeD and CosPairD, both methods, seeds 10 to 14
   (`results/runs_posthoc_lift1.jsonl`, `results/summary_exploratory.json`). It enters no verdict.
3. (After the first re-check, written before the runs.) The addendum `PREREG_C1_P5.md` added AltRidgeD, on
   which the re-check had found the lift worse than plain; beta = 1000 was chosen by the study's protocol
   (`results/sweep_selection_p5.json`); lift1 is a pre-registered arm there. The frozen verdict on the first
   four problems is kept and reported next to the extended verdict.
4. (No runs.) The closed-form floors of AltRidgeD and the proposition on rep3 under Adam are in
   Supplementary Section S7.6; rep3 is a reparametrisation control.

**First re-check** (`check/recompute.py`, `check_proofs.py`, `check_lap.py`, `reimpl.py`,
`analyze_reimpl.py`; plan written before its runs and frozen at 19:41:52 with SHA-256
`bcd618304de5786f6d71e2150f6d2dbf37daa81d7040db5d18f9dc19b036cf86`; the plan is not published). Every quoted number recomputed from the run
files (`recompute.out`); the proofs checked numerically; the main cells re-run with separately written code
(nested reverse-mode Laplacian, own samplers, test sets and quadrature), seeds 30 to 32
(`runs_reimpl.jsonl`, `reimpl_summary.txt`): Deep Ritz on LapD, RidgeD and AltRidgeD and the PINN on AltRidgeD
at d = 20, Deep Ritz on AltRidgeD at d = 5 and 10. Its median ratios against plain lie on the same side of 1 as
those of the study's runs in every cell that both contain, except rep3 for Deep Ritz on AltRidgeD at d = 20
(1.21, two of three seeds better, against 0.93).

**Second re-check** (`check/recompute2.py`, `check2_proofs.py`, `reimpl2.py`, `analyze_r2.py`; plan written before
its runs and frozen at 21:59:14 with SHA-256 `e9525a5031ed209c46be5f386715cb75b1eacbc5f71e6a988895313d2c4e29b6`;
the plan is not published). Plan: separately written code, fresh seeds 40 to 42, the Deep Ritz weight chosen by
the study's protocol on that code's validation set; Deep Ritz on SinLinD (plain, presolve, lift3c, lift1, lift3c
at equal CPU, d = 10) and on ExpCosD (plain, presolve, lift3c, lift1); the PINN on LapD (plain, presolve,
lift3c, then lift3c at equal CPU) and on SinLinD (plain, presolve, lift3c, lift1); criteria fixed in advance
(an arm helps if it wins in 3 of 3 seeds with median ratio at least 1.5, hurts if the median is at most 1/1.1).
The sources, affine minimisers and floors of the two new problems were checked before training
(`selftest2.json`). The runs stopped after the second seed of the PINN on LapD: the third seed of that cell, its
equal-CPU runs and the PINN on SinLinD were not made (`data/s6_highdim_remedies_recheck.json`,
missing_runs_of_plan). The article reports the finished runs as descriptive results of a different
implementation (`runs_reimpl2.jsonl`, `reimpl2_summary.txt`).

### 4.4 What a reader should not conclude

* That the stability constant or the computable bound is a new method: the constant is classical, and Payne's
  dimension-free bound already gives most of the improvement over the elementary constant 2 C_d.
* That the bound estimates the error: it held on every network tested, but its efficiency is 1.9 to 3.4 at
  d = 10 and 20, up to 5.46 at d = 5 and up to 24.8 at d = 2.
* That H3 alone shows the error to be dominated by the boundary misfit (H3 is a statement about the bound). At
  d = 20 the decomposition of the remark in Section 6.3 does show it for the networks tested (the interior term
  is at most 0.36 of the error on every main network); at d = 10 it does not (up to 0.81).
* That U_d is sharp, or that the limit of d c(d) is proved.
* That either remedy is general at d = 20. By the frozen rule lift3c was a general remedy on the four
  pre-registered problems, of which only RidgeD and CosPairD had been held out from the pilots; it and presolve
  are less accurate than plain on AltRidgeD, added afterwards. On PoiD, the one problem of the pre-registered
  study on the plateau, presolve does not help, and lift3c, whose features contain the solution of PoiD up to
  3.9e-3, leaves the plateau for both methods in every seed. For Deep Ritz the gain
  of lift3c cannot be attributed to its features: the featureless control rep3 gains more (median ratio 8.10
  against 6.59). The centred input alone (lift1, exploratory) also leaves the plateau for both methods, with less
  gain; for the PINN, for which no featureless control was run, lift3c is 7.51 times more accurate than lift1
  (median, 5 of 5 seeds). On the other problems of the study the comparison is one of accuracy at a fixed
  budget; there lift3c is less accurate than plain on AltRidgeD, which is not a blind test.
* Anything beyond the network (tanh, three hidden layers of 64), optimiser, schedule, weights and budgets used;
  the CPU times were measured on a loaded shared machine.
* That the bound through the torsion function (Remark S9) improves on the theorem: it is proved, but it
  decreases with d more slowly than U_d.

---

## 5. Where the numbers are

Every number that the article takes from these records is stored, with its source, in `data/checks/`:

| File | Content |
|---|---|
| `data/checks/verification.json` | numbers of the verification of the two studies quoted in the article and not stored in a result file of the verification codes |
| `data/checks/study_records.json` | working records of the two studies: software, compute, deviations from the plan |
| `data/checks/machine_load.json` | load averages of the shared machine during the timings |

All other numbers come from result files of the experiments and of the verification codes, named in the
`% src:` comments of the LaTeX sources.
