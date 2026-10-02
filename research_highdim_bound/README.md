# Stability constant and computable error bound on the unit cube (Section 6.7)

Scripts, stored results and re-checks behind Section 6.7 of the article and Section S7.5 of its Supplementary Material: the norm kappa_d of
the harmonic extension from L2 of the boundary to L2 of the cube, the error bound for any network computable
from the two terms of the PINN loss, its evaluation on the trained networks of `research_highdim/`, and the
exact penalty bias of the Laplace problem LapD.

## Names used in this folder

The scripts and result files use the names of the study's working document. In the article:

| here | article |
|---|---|
| C_d, the norm of the operator T | kappa_d (Section 6.7; Lemma S8) |
| Theorem 2 | Theorem 5 (bounds L_d, U_d of kappa_d^2) |
| Corollary 2 (a)-(c) | Corollary 7 (a)-(c) |
| Proposition on the biharmonic Steklov eigenvalue | Proposition 6 |
| Theorem 1 (first-order term of the bias) | Theorem S12 |
| Proposition 3 (exact bias) | Proposition S11 |
| Rayleigh-Ritz / restricted Lanczos lower bounds | Observation S10 |
| conjecture on d c(d) | Conjecture S4 |
| eq. (A.3), eq. (TT) (the operator T*T and the norm of T zeta in the sine basis) | equation (S8), Supplementary Section S7.5 |
| Remark 4 (maximum-principle bound) | the bound max abs(v - u*) <= max over the boundary of abs(v - g) + (1/8) max abs(Lap v + f) quoted in Section 6.7; not Remark 4 of the article |
| P1, P2 | LapD, PoiD |
| H1, H2, H3 | the hypotheses of `PREREG_T2.md`, as in Section 6.7 |

## Contents

| Path | Content |
|---|---|
| `PREREG.md`, `PREREG_T2.md` | pre-registration of the evaluation of the bound on the saved networks, unchanged since it was frozen (SHA-256 and time in `notes/VERIFICATION.md`, section 4.2) |
| `src/t2common.py`, `src/bias_quad.py` | shared definitions (U_d, L_d, loading of the networks of `research_highdim/`); quadrature of the exact bias |
| `scripts/s1_theorem2_checks.py`, `s1b_restricted_lanczos.py` | numerical checks of every step of the proof of Theorem 5; Rayleigh-Ritz lower bounds |
| `scripts/s2_corollary_checks.py` | checks of Corollary 7 on harmonic test functions and separable errors |
| `scripts/s3_exact_bias.py`, `s3b_bias_recheck.py`, `s6_bias_table.py` | exact penalty bias of LapD (Proposition S11), its check, and its table |
| `scripts/s10_dc_large_d.py`, `s10b_dc_quadrature_check.py`, `s10c_dc_truncation.py` | the limit d c(d) of Conjecture S4 |
| `scripts/s0_inventory.py`, `s4_eval_bound.py`, `s5_analyse.py`, `s8_tables.py`, `s7_figure.py` | inventory of the saved networks, the pre-registered evaluation, the decisions on H1 to H3, tables, figure |
| `scripts/s9_posthoc_analyses.py` | analyses made after the decisions (reported as such in the article) |
| `results/` | all outputs of these scripts |
| `inputs/robin_bias_sparse.out`, `inputs/robin_bias_sparse.py` | sparse Galerkin values of the bias computed in a check of the article, read by `s6_bias_table.py` for a comparison |
| `check/` | the two rounds of re-checks with separately written code (scripts `c1_*.py` to `c5_*.py`, `run_c4.sh`, `r2_*.py`; outputs in `check/out/`, `check/out_r2/`); described in `notes/VERIFICATION.md`, section 4.2 |

## Commands

From this folder (Python of `requirements.txt`; one thread; `s1b_restricted_lanczos.py` takes about 20 minutes
of CPU time, each other line a few minutes or less):

```sh
python scripts/s1_theorem2_checks.py && python scripts/s1b_restricted_lanczos.py
python scripts/s2_corollary_checks.py && python scripts/s9_posthoc_analyses.py
python scripts/s3_exact_bias.py && python scripts/s3b_bias_recheck.py && python scripts/s6_bias_table.py
python scripts/s10_dc_large_d.py && python scripts/s10b_dc_quadrature_check.py && python scripts/s10c_dc_truncation.py
python check/r2_thm2_edges.py check/out_r2/r2_thm2_edges.json && python check/r2_rr_quadrature.py && python check/r2_dc_theta.py check/out_r2/r2_dc_theta.json
```

In `results/restricted_lanczos.json` the last configuration of `s1b_restricted_lanczos.py` (d = 100 with two
modes per coordinate, K = 21) gives 0.014193, below the one-mode value 0.020389, which is therefore the value
used at d = 100.

The evaluation of the bound on the saved networks (`s0_inventory.py`, `s4_eval_bound.py confirm`) and its
re-evaluation (`check/c3_eval.py`) need the trained networks of `research_highdim/`, which are not
redistributed; re-run `research_highdim/scripts/run_study.py` (reproduce.md, section 3.3) to recreate them.
Their outputs are stored in `results/` and `check/out/`. The analysis of the stored evaluation runs without the
networks:

```sh
python scripts/s5_analyse.py && python scripts/s8_tables.py && python scripts/s7_figure.py
```

The 45 networks of the re-checks are trained from scratch by `check/run_c4.sh` (24 networks; run with `zsh`
from `check/`, it skips those already in `check/out/c4_fresh.jsonl`) and `check/r2_fresh_train.py <output
file>` (21 networks); training takes about 8 minutes of CPU time per round on the machine used, and the
evaluation of each network on 2 x 10^5 new points comes on top. The tables and the figure of the article are
made from the stored outputs by `code/s6_highdim_bound.py`.

## Not included

Trained networks; the working document of the proofs, which the pre-registration documents name as
`proof.tex` (Supplementary Section S7.5 holds the proofs); running notes, the reports of the re-checks and the
record of the choices made after the freeze, which the pre-registration documents name as `POSTHOC_LOG.md` and
which `notes/VERIFICATION.md` (section 4.2) summarises; a third-party article. The freeze record
`PREREG_FREEZE.json` (SHA-256 hashes and local time of the freeze) is included unchanged; the hash it records
for `src/t2common.py` no longer matches, for the reason given in `notes/VERIFICATION.md`, section 4.2, item 2.
The freeze is a record in this repository; no external registry was used.

## Published copies

The pre-registration documents, the evaluation script `scripts/s4_eval_bound.py` and `results/inventory.json` are
published byte for byte, so that their SHA-256 hashes (`notes/VERIFICATION.md`, section 4.2) can be checked. In
the other files, absolute paths of the development machine are replaced by paths relative to the repository root,
references to files that are not published (the working document of the proofs, the record of later choices,
the plans and records of the re-checks) are replaced by a short description and a pointer to
`notes/VERIFICATION.md`, and the wording of some comments, docstrings and variable names is edited; no number is changed. The
pre-registrations refer to earlier exploratory pilots (folders
`../ideas_theory`, `../ideas_cost`, `../ideas_dynamics`), which are not included; no result of the article rests
on them.
