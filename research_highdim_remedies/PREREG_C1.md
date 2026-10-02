# PREREG_C1: two cheap remedies for the affine plateau at d = 20

Written 2026-10-01 before any counted run. Frozen by SHA-256 in `PREREG_FREEZE.json`. Every later
choice goes to `POSTHOC_LOG.md`. Runs made before the freeze, none of which counts:
`results/runs_timing.jsonl` (cost per iteration, seed 99, 100 iterations) and `results/runs_sweep.jsonl`
(Deep Ritz weight for P3 and P4, see 2.3). The pilots in `../ideas_cost` and `../ideas_dynamics` used
seeds 0-2 and are not part of this study.

## 1. Question and status

Section 6 of the article reports that at d = 20 neither the PINN nor Deep Ritz improves on the best
affine fit of the Poisson solution (P2) within 4000 iterations. Two known, cheap ideas are tested on
that failure:

- presolve: subtract the exact minimiser of the method's own loss over affine functions (a fixed
  (d+1) x (d+1) linear solve) and let the network learn the correction. This is the cheapest instance of
  "Galerkin solution plus neural correction" (Ainsworth and Dong 2021; Aldirany et al. 2023).
- lift3c: feed the network the coordinate-wise Legendre features P_1, P_2, P_3 of t_i = 2x_i - 1 instead
  of x. This is the first layer of Chebyshev/Legendre-KAN and feature-embedding networks (Shukla et al.
  2024; Fazliani et al. 2025; Fourier features, Wang et al. 2021).

Neither is claimed as a new method. What is measured is whether either removes the d = 20 failure on
problems it was not built for, with controls for centring (lift3u) and fan-in (rep3), at equal
iterations and at equal CPU time. All outcomes are numerical observations; no law in d is claimed or
tested.

## 2. Fixed setting

### 2.1 Code and training (unchanged from the article's study)

`pinn/research_highdim/scripts/hd_core.py` is imported, not copied: network d_in-64-64-64-1 with tanh and
PyTorch default initialisation (seeded by `torch.manual_seed(seed)`), float32, Adam, learning rate 1e-3,
4000 iterations, MultiStepLR with factor 0.1 at 2000 and 3000 iterations, 1024 interior and 1024
boundary points redrawn every iteration from `Generator(1000 + seed)`, one torch thread. PINN loss
mean (Lap u + f)^2 + lambda mean_bd (u - g)^2 with lambda = 1000 and the exact Laplacian by forward
(Taylor-mode) propagation. Deep Ritz loss mean (1/2 |grad u|^2 - f u) + beta 2d mean_bd (u - g)^2.
Held-out sets: `hd_core.fixed_sets(d)` (test 50 000 interior and 20 000 boundary points, seed
10 000 + d; validation 5000 points, seed 20 000 + d), never used for training or for any choice.
Driver: `scripts/run_c1.py`; model code `src/c1core.py`; presolve `src/presolve.py`.

### 2.2 Problems (sources and floors checked before any training)

| id | u* | f = -Lap u* | role |
|---|---|---|---|
| P1 | sum_{k <= d/2} x_{2k-1} x_{2k} | 0 | fair pilot problem (lift gives no linear advantage) |
| P2 | d^{-1/2} sum_i cos(pi x_i) | pi^2 u* | the article's d = 20 failure; FEATURE-MATCHED: reported, excluded from the lift decision |
| P3 | cos(2s), s = d^{-1/2} sum_i (2x_i - 1) | 16 u* | non-separable ridge, never trained with a remedy. A related ridge, sin(2s) with s = sqrt(12/d) sum(x_i - 1/2), was trained in a theory pilot (`../ideas_theory`, pilot D) |
| P4 | sum_{k <= d/2} cos(pi x_{2k-1}) cos(pi x_{2k}) | 2 pi^2 u* | non-polynomial interactions |

Source check (`results/checks.json`, C1): f + Lap u* by nested reverse-mode autograd in float64, 256
points, d in {2, 5, 10, 20}: maximum absolute difference 4.3e-14 over all problems and d.

Floors: relative L2 error of the best function of each class (exact value from one-dimensional
integrals, `src/floors.py`; in brackets least squares on the held-out test set, the set on which errors
are reported; `results/checks.json`, C5):

| problem, d | best constant | best affine in x | best affine in the lift3 features |
|---|---|---|---|
| P1, 10 | 0.3669 (0.3682) | 0.1387 (0.1391) | 0.1387 (0.1391) |
| P1, 20 | 0.2686 (0.2695) | 0.1015 (0.1020) | 0.1015 (0.1020) |
| P2, 10 | 1.0000 (1.0000) | 0.1203 (0.1195) | 0.0039 (0.0039) |
| P2, 20 | 1.0000 (1.0000) | 0.1203 (0.1193) | 0.0039 (0.0039) |
| P3, 10 | 0.7151 (0.7178) | 0.7151 (0.7178) | 0.7018 (0.7037) |
| P3, 20 | 0.7136 (0.7150) | 0.7136 (0.7149) | 0.7072 (0.7087) |
| P4, 10 | 1.0000 (1.0000) | 1.0000 (0.9999) | 1.0000 (0.9996) |
| P4, 20 | 1.0000 (1.0000) | 1.0000 (0.9999) | 1.0000 (0.9996) |

The lift3c and lift3u feature spans are equal (same floors). Consequences recorded before training:
on P1, P3 and P4 the lift gives at most a 1 % lower linear floor, so any gain there is not explained by
the features containing the solution; on P2 the lift features contain the solution up to 0.39 %.

### 2.3 Penalty weights

PINN: lambda = 1000 for all problems (the study's value). Deep Ritz: beta = 100 on P1, beta = 1 on P2
(the study's values). For P3 and P4 the study has no value; it was selected by the study's own protocol
before this freeze (plain network, d = 5, seed 100, 4000 iterations, beta in {1, 10, 100, 1000}, argmin of
the validation error): beta = 100 on P3, beta = 1 on P4 (`results/runs_sweep.jsonl`,
`results/sweep_selection.json`). No d = 20 or d = 10 cell and no test-set value entered this choice.

## 3. Arms

| arm | network input / ansatz | methods |
|---|---|---|
| plain | u = N(x): the existing PINN (lambda = 1000) and Deep Ritz (beta of 2.3) | both |
| presolve | u = p(x) + N(x); p = exact minimiser over affine functions of the method's own population loss with the same weight; fixed; only N is trained | both |
| lift3c | u = N(P_1(t), P_2(t), P_3(t)), t = 2x - 1 coordinate-wise (3d inputs) | both |
| lift3u | u = N(x, P_2(t), P_3(t)) (linear feature not centred; centring control) | Deep Ritz |
| rep3 | u = N(t, t, t) (fan-in control) | Deep Ritz |

Presolve details (proof.tex, Proposition 1): with phi = (1, x), M = int_bd phi phi^T (closed form),
r_f = int f phi, r_g = int_bd g phi, E = diag(0, 1, ..., 1): Deep Ritz (E + 2 beta M) c = r_f + 2 beta r_g;
PINN M c = r_g (the L2(boundary) projection of g; independent of lambda and f). r_f and r_g by
64-point Gauss-Legendre products (exact up to round-off). Checks (`results/checks.json` C3, C4;
`results/check_presolve_quad.json`): M against quadrature 8.9e-15; against brute force on two independent
10^6-point Monte-Carlo samples the difference is of the size of the difference between the two samples
(e.g. P1, d = 20, Deep Ritz beta = 100: 2.7e-3 against 3.0e-3); against tensor-Gauss brute force for
d = 2, 3: <= 1.7e-13. Structural facts (proof.tex, Proposition 2): p = 0 for P4 for both methods (computed
|c| <= 1.2e-15), so the presolve arm on P4 is the plain network up to float32 round-off and is predicted
to equal plain (it is run anyway); p is a constant on P3 (c_i = 0 for i >= 1). The forward Laplacian of
every arm (with and without shift) agrees with nested autograd in float64 to a relative 2.5e-15
(`results/checks.json`, C2).

## 4. Seeds, budgets, design

Seeds 10, 11, 12, 13, 14 (seeds 0-2 were used by the pilots). The same seed gives every arm the same
sample stream; plain and presolve also share the network initialisation (the lifted arms have a 3d-input
first layer and therefore a different initialisation).

1. repro (harness check, run first): plain arm, P1 and P2, d = 20, both methods, seeds 0, 1, 2 (12 runs).
   Each must reproduce the main study (`research_highdim/results/runs_ext_s*.jsonl`,
   `results/ckpt/main_*_d20_*`) bit for bit: identical test relative L2 and identical final weights.
   If any differs, STOP: the harness is wrong; nothing below is run or reported as a result.
2. main20 (primary): d = 20, 4000 iterations, step schedule. Deep Ritz: plain, presolve, lift3c, lift3u
   x P1-P4 x seeds 10-14 (80 runs) plus rep3 x P1-P4 x seeds 10-14 (20 runs, block "rep3").
   PINN: plain, presolve, lift3c x P1-P4 x seeds 10-14 (60 runs).
3. eqcpu (primary): lift3c at equal CPU time, both methods x P1-P4 x seeds 10-14 (40 runs), see 5.
4. stall (secondary): P2, d = 20, Deep Ritz, plain and presolve, constant learning rate 1e-3, 12 000
   iterations, seeds 10, 11, 12 (6 runs).
5. d10 (secondary): d = 10, Deep Ritz, plain / presolve / lift3c x P1-P4 x seeds 10, 11, 12 (36 runs).

Run order: repro, main20, eqcpu, stall, rep3, d10. Drop order if time is short: d10 first, then rep3,
then stall. Every run < 15 min and < 4 GB (measured cost: Deep Ritz about 2.1-2.5 ms per iteration, PINN
about 7.7-8.1 ms per iteration at d = 20, `results/runs_timing.jsonl`), one process, nice 19, serial while
the 1-minute load is above 24; each run is appended to `results/runs_<phase>.jsonl` with its weights in
`results/ckpt/`, and finished runs are skipped on restart.

## 5. Equal CPU time

CPU = process CPU time of the training steps (sampling, loss, backward, optimiser step), excluding
logging and evaluation, as in the study; for presolve the one-off CPU time of computing p is added.
- lift3c: for each (method, problem), N_eq = 4000 x median_s CPU(plain) / median_s CPU(lift3c), medians
  over seeds 10-14 of the 4000-iteration main20 runs, rounded to a multiple of 50; the lift3c arm is
  rerun with N_eq iterations and the step schedule rescaled (milestones N_eq // 2, 3 N_eq // 4). The
  measured CPU of the rerun is reported next to the plain CPU.
- presolve: its per-iteration cost equals plain up to one affine evaluation. If for a (method, problem)
  median_s [CPU(presolve) + CPU(solve)] / median_s CPU(plain) <= 1.05, the 4000-iteration comparison
  counts as an equal-CPU comparison. Otherwise presolve is rerun with
  N_eq = 4000 x (median CPU(plain) - median CPU(solve)) / median CPU(presolve training), rounded to 50.

## 6. Metrics

Primary: relative L2 error on the held-out test set (50 000 points). Secondary: centred relative L2
(error over the fluctuation of u*), boundary relative L2, maximum error, CPU seconds. Per cell: mean,
standard deviation, min and max over seeds. Paired ratio per seed: rho = err(plain) / err(remedy) (> 1:
the remedy is better); per (method, problem, arm, budget): median, min, max of rho and the number of
seeds with rho > 1 ("wins"; rho = 1 is not a win). Logged every 50 iterations on the validation set
(descriptive only, no decision uses them): relative error, affine part of the error, gamma(t) = share of
the non-affine part of u* captured by the model (as in `../ideas_dynamics/scripts/dyn_core.py`);
T_half = first logged iteration with gamma >= 0.5, T10 likewise with 0.1.

## 7. Decision rule (frozen)

All decisions use d = 20, the primary metric, and seeds 10-14 only.

1. Remedy R in {presolve, lift3c} HELPS on problem p if, for BOTH methods, rho > 1 in >= 4 of 5 seeds and
   median rho >= 1.5 at 4000 iterations, AND the same holds at equal CPU time (lift3c: the eqcpu runs
   against plain at 4000 iterations; presolve: the 4000-iteration runs if the CPU condition of 5 holds,
   otherwise its eqcpu runs).
2. R HURTS on p if, for at least one method, median rho <= 1/1.1 at 4000 iterations or (when separate
   equal-CPU runs exist) at equal CPU.
3. Otherwise R is NEUTRAL/MIXED on p (reported with every cell).
4. R is reported as a GENERAL remedy for the plateau only if it HELPS on P1 and on at least one of P3, P4,
   and HURTS on none of P1, P3, P4. Otherwise it is reported as problem-specific, with every cell shown.
   P2 is reported for both remedies; for lift3c it is excluded from this decision (feature-matched); for
   presolve P2 is reported, and the decision of item 4 uses P1, P3, P4 as written.
5. Centring clause: if, for Deep Ritz at 4000 iterations on P1, lift3c has rho > 1 in >= 4 of 5 seeds and
   lift3u has rho < 1 in >= 4 of 5 seeds, or the reverse, the P1 effect of the lift is attributed to input
   centring/scale and no statement about Legendre features is made on P1. rep3 is reported descriptively
   next to this clause.
6. Failure (reported in full as a negative result): (a) neither remedy helps on P3 or P4; (b) the presolve
   gain on P1 disappears at equal CPU; (c) the repro check fails (then stop).

Secondary blocks (no effect on the verdicts above):
- stall: the stall seen in pilot D2 is called CONFIRMED if, at 12 000 iterations, presolve has gamma < 0.1
  in >= 2 of 3 seeds while plain has gamma >= 0.5 in >= 2 of 3 seeds; REFUTED if presolve reaches
  gamma >= 0.5 in >= 2 of 3 seeds; otherwise inconclusive. Test errors are reported.
- d10: paired ratios reported descriptively (3 seeds; no verdict).

## 8. Not claimed

No law in d, no statement for other networks, optimisers, budgets or weights, no claim that either idea
is new. If a remedy helps on P1 only, it is reported as problem-specific. Results from seeds 0-2 (pilots)
are never pooled with seeds 10-14.
