# PREREG_C1_P5: addendum to PREREG_C1.md, a fifth problem (P5) and a centred-input baseline

Written 2026-10-01 after the check of this study (`check/`), BEFORE any run of this addendum with the study's
code. Frozen by SHA-256 in `PREREG_FREEZE.json` (key `addendum_p5`). `PREREG_C1.md` is unchanged and its
verdicts (on P1-P4) stay as frozen; this addendum adds runs and one extended verdict next to them.

## 1. Why, and what is already known

The check re-ran parts of the study with separately written code (`check/reimpl.py`; same project, same
machine; fresh seeds 30-32) and introduced a further ridge problem P5. There, at d = 20, lift3c was less
accurate than plain Deep Ritz in 3/3 seeds (median paired ratio 0.46; 0.32 at equal CPU; 0.59 at d = 10;
0.42 at d = 5), presolve was less accurate (0.73, 0/3), and for the PINN lift3c was neutral (0.90, 1/3)
(`check/reimpl_summary.txt`). The check also noted that the plain network takes uncentred inputs in [0, 1]
while normalising inputs to [-1, 1] is standard practice, so the natural baseline for lift3c is the
centred-input network lift1 = N(2x - 1), which in the study exists only as an exploratory arm.

So this addendum is NOT a blind held-out test: P5 and the direction of its outcome are known from the
check's 3-seed runs. It is a confirmatory replication with the study's own code (`src/c1core.py`,
`scripts/run_c1.py`), the study's seeds 10-14 (never run on P5), the study's held-out test set
(`hd_core.fixed_sets`) and its weight-selection protocol, with lift1 run as a pre-registered arm.

## 2. Problem P5 (checks in `results/checks_p5.json`, run before any P5 training)

u* = cos(pi s + 1), s = d^{-1/2} sum_i e_i (2 x_i - 1), e_i = +1 for odd i and -1 for even i (1-based),
f = -Lap u* = 4 pi^2 u*, g = u* on the boundary.
- Source by nested autograd in float64, d in {2, 5, 10, 20}: |f + Lap u*| <= 2.2e-14 (C1).
- Affine minimisers (src/presolve.py, P5 as the real part of one complex product): against tensor-Gauss
  brute force for d = 2, 3, <= 6.7e-14 (C4b); against two 10^6-point Monte-Carlo samples at d = 20 the
  difference (<= 9.4e-3) is of the size of the difference between the two samples (<= 7.4e-3) (C4).
  Unlike P3, the affine minimiser of P5 is not constant (odd/even coefficients -0.195/+0.195 for the PINN).
- Floors at d = 20 (exact; test-set least squares in brackets): best affine in x 0.8965 (0.9004), best
  affine in the lift3 features 0.8951 (0.8988), best constant 0.9897 (0.9895) (C5). The lift features do
  not contain the solution (lower floor by 0.2 %).
- The edit that added P5 leaves P1-P4 unchanged: floors, affine minimisers and sources of P1-P4 equal the
  stored pre-edit values to <= 4.3e-14 (C6).

## 3. Weight

PINN: lambda = 1000. Deep Ritz: beta for P5 is selected by the study's protocol BEFORE any d = 20 or d = 10
P5 run: plain network, d = 5, seed 100, 4000 iterations, beta in {1, 10, 100, 1000}, argmin of the error on
the study's validation set (`hd_core.fixed_sets(5)["val"]`); stored in `results/sweep_selection_p5.json`.
These 4 runs (phase `sweepP5`) do not count.

## 4. Arms and blocks (seeds 10-14 unless stated; 4000 iterations and the step schedule unless stated)

Arms as in PREREG_C1.md section 3, plus lift1 = N(t), t = 2x - 1 (centred input, d inputs; the arm of
POSTHOC_LOG.md entry 2, now pre-registered for P5).

1. `p5main20` (primary), d = 20.
   Deep Ritz: plain, presolve, lift3c, lift3u, rep3, lift1 (30 runs). PINN: plain, presolve, lift3c, lift1 (20 runs).
2. `p5eqcpu` (primary): lift3c at equal CPU time for both methods by the rule of PREREG_C1.md section 5,
   with medians over the p5main20 runs (10 runs); presolve by the CPU rule of the same section (rerun only
   if its CPU condition fails).
3. `p5dim` (secondary): Deep Ritz, d = 5 and d = 10, plain / lift3c / lift1, seeds 10, 11, 12 (18 runs).
4. No new runs: the lift3c-versus-lift1 paired ratios on P1-P4 from the existing runs (`runs_main20.jsonl`,
   `runs_posthoc_lift1.jsonl`; same seeds 10-14, same sample streams).

Resources as in PREREG_C1.md section 4 (serial, nice 19, one thread, every run < 15 min, resumable).

## 5. Expectations (from the check, stated before the runs)

E1: lift3c HURTS Deep Ritz on P5 at d = 20 (median rho <= 1/1.1), at 4000 iterations and at equal CPU.
E2: presolve HURTS on P5 for Deep Ritz. E3: lift3c is less accurate than lift1 on P5 for both methods.

## 6. Decision rules (frozen)

1. Per remedy (presolve, lift3c) on P5: HELPS / HURTS / NEUTRAL exactly by items 1-3 of PREREG_C1.md
   section 7 (both methods, seeds 10-14, 4000 iterations and equal CPU).
2. Extended verdict, reported NEXT TO the frozen one (which stays as it is): a remedy is a "general remedy
   on P1-P5" only if it HELPS on P1 and on at least one of P3, P4, P5, and HURTS on none of P1, P3, P4, P5.
3. Lift beyond centring (descriptive per problem, both methods): on problem p the lift "adds to centring" if
   rho' = err(lift1)/err(lift3c) > 1 in >= 4 of 5 seeds with median >= 1.5 for both methods; it is "worse than
   centring" if the median rho' <= 1/1.1 for at least one method; otherwise "no clear difference". Applied
   to P1-P5 at d = 20, 4000 iterations.
4. p5dim: paired ratios reported descriptively (3 seeds, no verdict).
5. If the study's code does not reproduce E1 (lift3c not HURTING Deep Ritz on P5), that is reported as a
   disagreement with the check and both results are given.

## 7. Not claimed

No statement for other networks, optimisers, budgets, weights or problems. P5 is one further problem; it
does not make the problem set representative.
