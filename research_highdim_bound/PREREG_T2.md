# Pre-registration T2: the computable L2 error bound on the saved networks

Written 2026-10-01, 18:00-18:10 BST, before any confirmatory checkpoint was evaluated.
Frozen by `PREREG_FREEZE.json` (SHA-256 of this file, of `PREREG.md`, of the evaluation script and
of the inventory, with the time). Later choices go to `POSTHOC_LOG.md`.

## What has been seen before this file was written (disclosure)

- Pilot E2 (`../ideas_theory/results/pilot_E.jsonl`): the bound on 13 networks (Deep Ritz P1 d = 20 with
  beta in {3, 10, 20, 100}, seeds 0, 1; PINN P1 d = 20; PINN and Deep Ritz P2 d = 10 and 20, seed 0), on
  the hd_core held-out points: eta between 1.91 and 2.69, boundary term 5-50 times the interior term
  at d = 20.
- `results/inventory.json` (parameters only, no norm evaluated): 7 of the 88 checkpoints are
  bit-identical to pilot-E2 networks: main laplace pinn d20 s0, main laplace ritz d20 s0 and s1,
  main poisson pinn d10 s0, main poisson ritz d10 s0, main poisson pinn d20 s0, main poisson ritz
  d20 s0. They are NON-CONFIRMATORY. The other 81 are confirmatory.
- A debug run of `scripts/s4_eval_bound.py pilot` on the 13 pilot-E networks (files in
  `../ideas_theory/results/nets_E`, which include the 7 above) on both sets below
  (`results/eval_pilot.jsonl`): eta 1.93-2.70 on the fresh set; the forward Laplacian agrees with
  nested autograd to 1.3e-15 relative.
- The network-free results of this directory (Theorem 2 checks, exact bias) and the article's
  tables of errors for these networks (`pinn/research_highdim/results/summary.csv`), which show the
  error but not the bound.

## Statement being tested (proof.tex, Corollary 2(c); Theorem 2)

For every v in C^2 of the closed cube Omega = (0,1)^d and the Dirichlet problem -Lap u = f, u = g with
solution u* in C^2:
  ||v - u*|| <= B_int + B_bd,  B_int = ||Lap v + f|| / (d pi^2),  B_bd = sqrt(U_d) ||v - g||_{L2(bd)},
  U_d = tanh(pi sqrt(d-1)/2) / (pi sqrt(d-1)).
In the losses' normalisation (means over uniform points): B_int + B_bd = sqrt(L_int)/(d pi^2) +
sqrt(2 d U_d) sqrt(L_bd).

## Networks

All 88 files of `pinn/research_highdim/results/ckpt/` (SHA-256 in `results/inventory.json`):
- main: P1 ("laplace") and P2 ("poisson") x PINN (lambda = 1000) and Deep Ritz (beta = 100 on P1,
  beta = 1 on P2) x d in {2, 3, 5, 10, 20} x seeds 0, 1, 2, 4000 iterations (60);
- long: d = 10, P1, both methods, 16 000 iterations, seeds 0-2 (6);
- equaltime: d = 10, Deep Ritz, P1 (10 000 iterations) and P2 (9 000 iterations), seeds 0-2, the
  equal-CPU counterparts of the 4000-iteration PINN runs (6);
- sweep: d = 5, seed 100, P1 and P2 x PINN and Deep Ritz x weight in {1, 10, 100, 1000} (16).
Baselines are thus covered at equal iterations (main, long) and at equal CPU time (equaltime vs the
main PINN at d = 10). No network is trained for this study.

## Evaluation (no tuning; fixed now)

- Script `scripts/s4_eval_bound.py confirm` (hash frozen), float64 evaluation of the saved float32
  parameters, exact Laplacian by the forward Laplacian of hd_core (checked against nested autograd
  on 64 points per network).
- Primary set "fresh": 200 000 uniform interior points (torch generator seed 50 000 + d) and
  200 000 uniform boundary points (seed 60 000 + d; face and side uniform), float64.
- Secondary set "hd": hd_core.fixed_sets(d) test (50 000 interior) and test_bd (20 000 boundary).
- Reported per network and set: err = ||v - u*|| (MC), relative error, B_int, B_bd, bound, eta =
  bound/err, boundary share B_bd/(B_int + B_bd), each with its delta-method Monte-Carlo standard
  error; D = bound - err with s.e. that includes the covariance of B_int and err (same points);
  sampled maxima max|v - u*|, max|Lap v + f|, max_bd|v - g| and the sampled maximum-principle
  quantity max_bd|v - g| + max|Lap v + f|/8 (descriptive only; sampled maxima are lower estimates of
  the true suprema).

## Hypotheses and decision rules

H1 (validity). For every one of the 88 networks, on each set: bound >= err - 3 s.e.(D).
  A violation on either set means a bug or a false theorem: stop, investigate, write nothing about
  the bound until resolved; report the violation.
H2 (efficiency). eta <= 3 (point estimate, fresh set) for every confirmatory network with d >= 5.
  If it fails for any such network: the bound is reported as valid but loose (with the eta table),
  as a remark, not as a selling point. Networks with d in {2, 3} are reported, not tested.
H3 (where the error lives). For each (problem, method) in {P1, P2} x {PINN, Deep Ritz}, the seed mean
  (seeds 0-2, main networks, fresh set) of the share B_bd/(B_int + B_bd) is non-decreasing in d over
  d = 2, 3, 5, 10, 20 (point estimates; no tolerance). Primary: all three seeds (includes the 7
  non-confirmatory networks at d = 10, 20, whose shares pilot E2 already showed); sensitivity: the
  same with confirmatory networks only (seed means over the available seeds). Decision on the
  primary. If H3 fails for any of the four pairs, the sentence "the boundary term dominates in high
  dimension" is dropped; the four curves are reported as they are.
H4 (penalty weight, d = 5 sweep). eta and share against the weight in {1, 10, 100, 1000} for the four
  (problem, method) pairs: descriptive, no hypothesis.
Descriptive comparisons (no hypothesis): eta and share of the 16 000-iteration and the equal-CPU
networks against the 4000-iteration networks of the same seeds; the L2 bound against the sampled
maximum-principle quantity.

## Seeds, budgets, compute

Seeds are those of the saved networks (0-2; 100 for the sweep). Monte-Carlo seeds as above. One
worker, one thread, nice 19; about 3 s per network and set.

## Optional parts (only if time remains; each needs an addendum frozen before it runs)

Spearman correlation of bound and error along training (re-runs at d = 10, 20, three seeds,
checkpoints 250..4000); pilot F's boundary-point allocation with three seeds at equal CPU (a null
result is allowed).
