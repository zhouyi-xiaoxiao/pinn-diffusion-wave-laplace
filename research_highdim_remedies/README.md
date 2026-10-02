# Two remedies for the plateau at d = 20 (Section 6.8)

Scripts, stored runs and re-checks behind Section 6.8 of the article and Section S7.6 of its Supplementary Material: a fixed affine part
computed by one linear solve (presolve) and coordinate-wise Legendre input features (lift3c), for the PINN and
the Deep Ritz method at d = 20, at equal iterations and at equal CPU time, with the controls lift3u, rep3 and lift1.

## Names used in this folder

| here | article |
|---|---|
| P1, P2 (`laplace`, `poisson`/`cospoisson`) | LapD, PoiD |
| P3 (`ridge`) | RidgeD |
| P4 (`cospair`/`prodcos`) | CosPairD |
| P5 (`altridge`) | AltRidgeD |
| P6, P7 (second round of the re-check) | SinLinD, ExpCosD |
| Lemma 1 (boundary Gram matrix) | Lemma S13 |
| Proposition 1 (affine minimisers) | Proposition S14 |
| Proposition 2 (structure of the affine minimisers) | Proposition S15 |
| Proposition 3 (floors; (e): AltRidgeD) | Proposition S16 |
| Proposition 4 (Laplacian of a lifted network) | Lemma S17 |
| Proposition 5 (rep3 under Adam) | Proposition S18 |

The network, optimiser, schedule, point sampling and test sets are imported from
`research_highdim/scripts/hd_core.py`.

## Contents

| Path | Content |
|---|---|
| `PREREG.md`, `PREREG_C1.md`, `PREREG_C1_P5.md` | pre-registration and its addendum for AltRidgeD, unchanged since they were frozen (SHA-256 and times in `notes/VERIFICATION.md`, section 4.3) |
| `src/c1core.py`, `src/presolve.py`, `src/floors.py` | the arms, the affine minimisers, the floors |
| `scripts/run_c1.py`, `scripts/queue.sh`, `scripts/queue_p5.sh` | training driver (one phase per call) and the order in which the phases were run |
| `scripts/check_all.py`, `check_p5.py`, `check_proof_formulas.py`, `check_p5_formulas.py`, `check_presolve_quad.py`, `check_rep3_equivalence.py`, `check_harness_after_p5.py`, `check_after_edit.py` | checks before training: sources, Laplacians, affine minimisers, floors, the formulas of the proofs, the rep3 equivalence, the unchanged code paths after AltRidgeD was added |
| `scripts/analyze.py`, `analyze_p5.py`, `per_seed_all.py` | decisions by the frozen rules, summaries and tables |
| `scripts/fig_ratios.py`, `fig_p5_baseline.py`, `fig_gamma.py` | figures (the article's versions are made by `code/s6_highdim_remedies.py`) |
| `results/runs_*.jsonl` | one record per run, with the load average of the machine at the time |
| `results/` (other files) | checks, decisions, summaries, tables |
| `check/` | the two rounds of re-checks with separately written code (`recompute*.py`, `check_proofs.py`, `check_lap.py`, `check2_proofs.py`, `reimpl.py`, `reimpl2.py`, `analyze_reimpl.py`, `analyze_r2.py`, `queue*.sh`) and their outputs; described in `notes/VERIFICATION.md`, section 4.3 |

## Commands

From this folder (Python of `requirements.txt`; one thread):

```sh
python scripts/check_all.py && python scripts/check_p5.py
python scripts/check_proof_formulas.py && python scripts/check_p5_formulas.py && python scripts/check_rep3_equivalence.py
python scripts/analyze.py && python scripts/analyze_p5.py      # decisions, summaries and tables from the stored runs
```

Re-running the training (some hours on one core):

```sh
for ph in timing sweep repro main20 eqcpu stall rep3 d10 posthoc_lift1 sweepP5 p5main20 p5eqcpu p5dim; do
  python scripts/run_c1.py $ph
done
```

(`scripts/queue.sh` and `scripts/queue_p5.sh` give the order that was used; they write their logs to `logs/`,
which must be created first.) `run_c1.py` skips runs whose records exist; move `results/runs_<phase>.jsonl`
aside to repeat a phase. The re-checks are re-run with `bash check/queue.sh sweepP5 R1 R2 R3 R4`, then
`bash check/queue.sh R5 R6 R7` (first round) and `bash check/queue_r2.sh sweeps B2 B3 B1 B1eq B4` (second
round; its last blocks were not run for the article), followed by `python check/analyze_reimpl.py` and
`python check/analyze_r2.py`.

## Not included

Trained networks; the working document of the proofs (Supplementary Section S7.6 holds the proofs); running
notes, training logs, the plans of the re-checks (their SHA-256 hashes are given in `notes/VERIFICATION.md`,
section 4.3) and their reports, and the record of the choices made after the freeze, which the
pre-registration documents name as `POSTHOC_LOG.md` and which `notes/VERIFICATION.md` (section 4.3) summarises;
the exploratory pilots that preceded the pre-registration (summarised in the same section). The freeze record
`PREREG_FREEZE.json` (SHA-256 hashes and local times of the freeze of `PREREG_C1.md` and of the addendum
`PREREG_C1_P5.md`) is included unchanged. The freeze is a record in this repository; no external registry was
used.

## Published copies

The pre-registration documents are
published byte for byte, so that their SHA-256 hashes (`notes/VERIFICATION.md`, section 4.3) can be checked. In
the other files, absolute paths of the development machine are replaced by paths relative to the repository root,
references to files that are not published (the working document of the proofs, the record of later choices,
the plans and records of the re-checks) are replaced by a short description and a pointer to
`notes/VERIFICATION.md`, and the wording of some comments and docstrings is edited; no number is changed. The
pre-registrations refer to earlier exploratory pilots (folders
`../ideas_theory`, `../ideas_cost`, `../ideas_dynamics`), which are not included; no result of the article rests
on them.
