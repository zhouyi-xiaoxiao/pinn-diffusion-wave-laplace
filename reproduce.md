# Reproducing the article and its Supplementary Material

All commands are run from the root of the repository unless a `cd` is shown. `python` is a Python
interpreter with the packages of `requirements.txt`.

There are three levels:

1. **Check** the numbers of the article against the stored results (seconds, no training).
2. **Regenerate** every table and figure from the stored per-run results (about two minutes, no training).
3. **Re-run the training** that produced the stored results (hours).

Levels 1 and 2 reproduce every number printed in the article and the Supplementary Material from the files in this repository (stored floating-point values may differ in the last digit; no printed number changes). Level 3 reproduces the
single-thread CPU runs and, on the same machine and software, the stacked GPU runs bit for bit; see "What to
expect" below.

## 0. Environment

```sh
python -m pip install -r requirements.txt
```

The results were produced with Python 3.14.6, PyTorch 2.14.1, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2,
SymPy 1.14.0 and mpmath 1.3.0 under macOS 26.7 on one 12-core Apple-silicon machine. The low-dimensional
benchmark was trained on the Metal (`mps`) backend of PyTorch; every script also runs on the CPU.

Building the PDFs needs a TeX distribution with `pdflatex`, `bibtex` and the package `xr-hyper` (TeX Live 2025
was used). The article and the Supplementary Material refer to each other; `sh build_pdfs.sh` runs
`pdflatex` and `bibtex` in the order that resolves every reference, checks the logs, and copies `main.pdf` to
`paper.pdf`, the file to which the links of the Supplementary Material point.

## 1. Check the article against the stored results

```sh
python code/check_headline_numbers.py   # the headline numbers; ends with "ALL HEADLINE CHECKS PASSED"
python code/sA_proofs_checks.py         # every computational step of the proofs; ends with "ALL CHECKS PASSED"
python code/s2_problems_symbolic.py     # exact solutions, symbolically; every printed expression is 0
python code/check_tables.py             # every generated table row against the article; ends with "ALL TABLE CHECKS PASSED"
python code/sC_details_check.py         # numbers quoted in Supplementary Section S8 against data/sC_details_numbers.json
python code/s6_highdim_bound.py --check     # Section 6.7 against research_highdim_bound/; ends with "CHECK PASSED"
python code/s6_highdim_remedies.py --check  # Section 6.8 against research_highdim_remedies/; ends with "CHECK PASSED"
```

Each quantitative statement in `sections/*.tex` (article: `s*.tex`; supplement: `S*.tex`) is accompanied by a comment `% src: <path>` next to it,
naming the file that holds the number.

## 2. Regenerate the tables and figures (no training)

Run in this order; later scripts read the outputs of earlier ones.

```sh
# aggregation of the raw runs of the two studies
(cd research_benchmark && python scripts/analyze.py && python scripts/make_tables.py && python scripts/make_figures.py)
(cd research_highdim   && python scripts/analyze.py && python scripts/derived.py && python scripts/make_tables.py)

# article
python code/closed_forms.py
python code/s2_problems_symbolic.py
python code/s2_problems_evalsets.py
python code/s3_methods_points.py
python code/s4_lowdim_tables.py
python code/s4_lowdim_gridcontrol.py merge
python code/s4_lowdim_figs.py
python code/s4_lowdim_timeslices.py merge
python code/s4_lowdim_timeslices.py plot
python code/s5_hard_cost.py
python code/s5_hard_tables.py
python code/s5_hard_wave1d_floor.py
python code/s5_hard_wave2d_long.py plot
python code/s5_hard_wave1d_long.py merge
python code/s5_hard_wave1d_long.py plot
python code/s5_hard_fd_lap3s.py
python code/s6_highdim_tables.py
python code/s6_highdim_robin_bias.py           # about two minutes on one core
python code/s6_highdim_figs.py
python code/s6_highdim_torsion.py             # Section 6.7: the torsion-function bound m_d (seconds)
python code/s6_highdim_bound.py               # Section 6.7: Tables S9 to S11, Figure 3 (from research_highdim_bound/)
python code/s6_highdim_remedies.py            # Section 6.8: Tables S12, S13 and S19, Figures S11 and S12 (from research_highdim_remedies/)
python code/s7_pitfalls.py --plot             # Section 7: Figure S13
python code/sA_proofs_checks.py
python code/sC_details_repro.py compare
python code/sC_details_tables.py
python code/sC_details_check.py
python code/check_tables.py
python code/check_headline_numbers.py

# article and Supplementary Material
sh build_pdfs.sh
```

The table scripts write LaTeX rows to `data/*_tables.tex`. Where a table in `sections/*.tex` is enclosed by the
comment lines `%% BEGIN GENERATED <name>` and `%% END GENERATED <name>`, the script also writes its rows there
(helper `code/inject.py`). The rows of Tables 9, S3, S4, S6 and S8 were pasted from the generated files;
`python code/check_tables.py` verifies that every generated row is in the article or the supplement. The figure scripts write to
`figures/`; all of them use the style of `code/figstyle.py`.

### Which script makes which figure

Figures 1 to 3 are in the article, Figures S1 to S13 in the Supplementary Material.

| Figure | File in `figures/` | Command | Reads |
|---|---|---|---|
| S1 | `s3_methods_points.pdf` | `python code/s3_methods_points.py` | `research_benchmark/pinnbench/` |
| S2 | `s4_lowdim_samplers.pdf` | `python code/s4_lowdim_figs.py samplers` | `research_benchmark/results/runs/*.json` |
| S3 | `s4_lowdim_curves.pdf` | `python code/s4_lowdim_figs.py curves` | the same |
| S4 | `s4_lowdim_errmaps.pdf` | `python code/s4_lowdim_figs.py errmaps` | `research_benchmark/results/runs/*_pred_seed0.npz` |
| S5 | `s4_lowdim_budget.pdf` | `python code/s4_lowdim_figs.py budget` | `research_benchmark/results/budget_sweep.json` |
| 1 | `s4_lowdim_timeslices.pdf` | `python code/s4_lowdim_timeslices.py plot` | `data/s4_lowdim_timeslices.json` |
| S6 | `s5_hard_wave2d.pdf` | `python code/s5_hard_wave2d_long.py plot` | `data/s5_hard_wave2d_long.json` |
| S7 | `s5_hard_wave1d.pdf` | `python code/s5_hard_wave1d_long.py plot` | `data/s5_hard_wave1d_long.json` |
| S8 | `s5_hard_cost.pdf` | `python code/s5_hard_cost.py` | `research_benchmark/results/{classical.json,benchmark_runs.csv,timing_single.json}`, `verify_research_benchmark/v_runs*.json` |
| 2, S9, S10 | `s6_highdim_error.pdf`, `s6_highdim_curves.pdf`, `s6_highdim_cost.pdf` | `python code/s6_highdim_figs.py` | `research_highdim/results/runs*.jsonl`, `research_highdim/results/cost_vs_d.json`, `verify_research_highdim/results/v_bench.json`, `data/s6_highdim_long_d20.jsonl`, `data/closed_forms.json` |
| 3 | `s6_bound.pdf` | `python code/s6_highdim_bound.py` (plot only; the study's own version of the figure is `research_highdim_bound/scripts/s7_figure.py`) | `research_highdim_bound/results/{theorem2_checks.json,restricted_lanczos.json,eval_confirm.jsonl}`, `data/s6_highdim_torsion.json` |
| S11, S12 | `s6_remedies_ratios.pdf`, `s6_remedies_altridge.pdf` | `python code/s6_highdim_remedies.py` (plot only; the study's own versions are `research_highdim_remedies/scripts/fig_ratios.py` and `fig_p5_baseline.py`) | `research_highdim_remedies/results/{summary.json,summary_p5.json,attribution.json}` |
| S13 | `s7_pitfalls.pdf` | `python code/s7_pitfalls.py --plot` | `data/s7_pitfalls_autograd.json`, `research_benchmark/results/oneface_laplace3d.json` |

### Which script makes which table

Tables 1 to 9 are in the article, Tables S1 to S19 in the Supplementary Material.

| Table | Content | Command | Output |
|---|---|---|---|
| 1 | test problems | written by hand; checked by `code/s2_problems_symbolic.py` and `code/s2_problems_evalsets.py` | `data/s2_problems_symbolic.txt`, `data/s2_problems_evalsets.json` |
| 2, 3 | networks, loss terms, collocation strategies | written by hand from `research_benchmark/pinnbench/{problems,samplers}.py`; the same settings are generated in Table S14 | |
| 4, 5, S1 | benchmark errors; strategies against random; budget sweep | `python code/s4_lowdim_tables.py` | `data/s4_lowdim_tables.tex`, `data/s4_lowdim_numbers.json` |
| 6 | control experiment for the cell-centred grid | `python code/s4_lowdim_gridcontrol.py merge` | `data/s4_lowdim_gridcontrol_table.tex`, `data/s4_lowdim_gridcontrol.json` |
| S2 | errors per time slice | `python code/s4_lowdim_timeslices.py merge` | `data/s4_lowdim_timeslices_table.tex`, `data/s4_lowdim_timeslices.json` |
| 7, 8 | harder problems; PINN against finite differences (three time measurements) | `python code/s5_hard_cost.py`, then `python code/s5_hard_tables.py` (`--check` to compare the article with the stored results) | `data/s5_hard_cost.json`, `data/s5_hard_tables.tex` |
| 9, S3 to S7 | dimension study, runs of the verification code, cost, choice of the weights | `python code/s6_highdim_tables.py` | `data/s6_highdim_tables.tex`, `data/s6_highdim_numbers.json` |
| S8 | penalty bias of Deep Ritz against the trained networks | `python code/s6_highdim_robin_bias.py` | `data/s6_highdim_robin_bias_table.tex`, `data/s6_highdim_robin_bias.json` |
| S9 to S11 | the constant of Theorem 5 (with the torsion-function bound); the computable bound on the main networks; the exact penalty bias | `python code/s6_highdim_torsion.py`, then `python code/s6_highdim_bound.py` (rows written into `sections/S5_highdim.tex`) | `data/s6_highdim_torsion.json`, `data/s6_highdim_bound_tables.tex`, `data/s6_highdim_bound_numbers.json` |
| S12, S13, S19 | the two remedies at d = 20; AltRidgeD and the centred-input baseline; the controls lift3u and rep3 | `python code/s6_highdim_remedies.py` (rows written into `sections/S5_highdim.tex` and `sections/S8_details.tex`) | `data/s6_highdim_remedies_tables.tex`, `data/s6_highdim_remedies_recheck.json` |
| S14 to S18 | settings, full errors, paired tests, finite differences, cost | `python code/sC_details_tables.py` | `data/sC_details_tables.tex`, `data/sC_details_numbers.json` |

## 3. Re-run the training

Scripts that train skip a run whose result file already exists. To repeat a run, move its stored result
file aside first. Run times below are those recorded on the loaded development machine; they are
indications only.

### 3.1 Low-dimensional benchmark (Sections 4 and 5)

```sh
cd research_benchmark
python scripts/verify_exact.py                       # exact solutions by finite differences
python scripts/verify_losses.py                      # single-pass loss against term-by-term loss
python scripts/run_classical.py                      # finite-difference baselines -> results/classical.json
python scripts/run_benchmark.py heat1d 0 10          # 50 stacked networks -> results/runs/heat1d_s0-10.json
python scripts/run_benchmark.py laplace2d 0 5        # 25 stacked networks
python scripts/run_benchmark.py laplace2d 5 10
python scripts/run_benchmark.py wave1d 0 5
python scripts/run_benchmark.py wave2d 0 5
python scripts/run_benchmark.py laplace3d 0 5
python scripts/run_budget_sweep.py heat1d 64,1024    # -> results/budget_sweep.json
python scripts/validate_batched.py                   # stacked driver against single-network driver
python scripts/time_single.py                        # time measurement A -> results/timing_single.json
python scripts/run_oneface.py                        # Section 7.2: data on one face only, five seeds -> results/oneface_laplace3d.json
cd ..
```

`run_benchmark.py`, `run_budget_sweep.py` and `code/s4_lowdim_gridcontrol.py run` take the device as an optional
last argument (`mps` or `cpu`; the default is `mps` where it is available). The six stacks of the benchmark took 241 to 790 s each.
Afterwards repeat the aggregation of step 2.

### 3.2 Longer runs, control experiments and demonstrations

```sh
# Section 4.3: control experiment for the cell-centred grid on the heat problem (60 stacked networks)
python code/s4_lowdim_gridcontrol.py run A     # random, grid-c, grid-n, grid-c shifted in t: 40 networks, 256 points
python code/s4_lowdim_gridcontrol.py run B     # grid-c with one further column at t = 1/64 or at t = 1/2: 20 networks
python code/s4_lowdim_gridcontrol.py merge

# Section 4.5: time slices of the heat problem (20 single-network runs, CPU)
python code/s4_lowdim_timeslices.py run random 0 10
python code/s4_lowdim_timeslices.py run sobol 0 10
python code/s4_lowdim_timeslices.py merge

# Section 5.1: longer L-BFGS phase on the wave problem Wave2 (one process per seed)
python code/s5_hard_wave2d_long.py run 0
python code/s5_hard_wave2d_long.py run 1
python code/s5_hard_wave2d_long.py run 2
python code/s5_hard_wave2d_long.py merge

# Section 5.2: three-dimensional Laplace problem with compatible data, and the control
python code/s5_hard_laplace3d_smooth.py
python code/s5_hard_fd_lap3s.py                # finite differences on Lap3s and Lap3 (no training)

# Section 5.3: longer runs on the two-mode wave problem Wave1 (one process per seed; 41 000 Adam iterations)
python code/s5_hard_wave1d_long.py run 0
python code/s5_hard_wave1d_long.py run 1
python code/s5_hard_wave1d_long.py run 2
python code/s5_hard_wave1d_long.py merge

# Section 6.3: 16 000 iterations at d = 20 (12 single-thread runs)
sh code/s6_highdim_long_d20_launch.sh python

# Section 7.1: a residual made identically zero by differentiating with respect to a slice of the input and
# replacing the missing derivative by zeros (well under a minute; a few minutes for the run with ten times as
# many passes)
python code/s7_pitfalls.py
python code/s7_pitfalls.py --passes 1500

# Supplementary Section S8.4: repeat three stacks with unchanged code and test the dependence on the stack
python code/sC_details_repro.py rerun heat mps
python code/sC_details_repro.py rerun wave2d mps
python code/sC_details_repro.py rerun gridcontrol_B mps
python code/sC_details_repro.py stacksize mps
python code/sC_details_repro.py stacksize cpu
python code/sC_details_repro.py compare
```

### 3.3 Dimension study (Section 6)

```sh
cd research_highdim
python scripts/check_core.py           # exact solutions; forward against nested Laplacian
python scripts/run_study.py sweep      # choice of the penalty weights at d = 5 (16 runs)
python scripts/run_study.py main 0     # d = 2, 3, 5, 10; one worker per seed
python scripts/run_study.py main 1
python scripts/run_study.py main 2
python scripts/run_study.py ext 0      # d = 20, 16 000 iterations at d = 10, equal CPU time; one worker per seed
python scripts/run_study.py ext 1
python scripts/run_study.py ext 2
python scripts/bench_cost.py           # cost per iteration for d = 2 to 100 -> results/cost_vs_d.json
cd ..
```

The 88 runs of this study sum to 5051 s of wall-clock time under load. `run_study.py` appends to
`results/runs*.jsonl` and skips keys that are already present.

### 3.4 Stability constant, error bound and remedies (Sections 6.7 and 6.8)

The two folders contain their own pre-registrations, scripts, stored results and re-checks; their `README.md`
files list every script. The network-free checks of the proofs and the exact penalty bias (minutes on one core,
except `s1b_restricted_lanczos.py`):

```sh
cd research_highdim_bound
python scripts/s1_theorem2_checks.py       # every step of the proof of Theorem 5
python scripts/s2_corollary_checks.py      # Corollary 7 on harmonic test functions and separable errors
python scripts/s3_exact_bias.py && python scripts/s3b_bias_recheck.py && python scripts/s6_bias_table.py   # Proposition S11, Table S11
python scripts/s10_dc_large_d.py && python scripts/s10b_dc_quadrature_check.py && python scripts/s10c_dc_truncation.py   # Conjecture S4
python scripts/s9_posthoc_analyses.py   # comparisons with Payne's bound and the unit square
python scripts/s1b_restricted_lanczos.py   # Rayleigh-Ritz lower bounds (about 20 minutes of CPU time)
python check/r2_thm2_edges.py check/out_r2/r2_thm2_edges.json && python check/r2_rr_quadrature.py && python check/r2_dc_theta.py check/out_r2/r2_dc_theta.json   # second re-check
cd ../research_highdim_remedies
python scripts/check_all.py && python scripts/check_p5.py            # sources, Laplacians, affine minimisers, floors
python scripts/check_proof_formulas.py && python scripts/check_p5_formulas.py && python scripts/check_rep3_equivalence.py
python scripts/analyze.py && python scripts/analyze_p5.py           # frozen decisions and tables from the stored runs
cd ..
python code/s6_highdim_torsion.py && python code/s6_highdim_bound.py && python code/s6_highdim_remedies.py
```

The evaluation of the bound on the saved networks (`research_highdim_bound/scripts/s0_inventory.py` and
`s4_eval_bound.py confirm`) needs the trained networks of section 3.3, which are not redistributed; the analysis
of its stored output runs without them:

```sh
cd research_highdim_bound
python scripts/s5_analyse.py && python scripts/s8_tables.py && python scripts/s7_figure.py   # decisions, tables, the study's figure
cd ..
```

The 45 networks of the re-check of Section 6.7 are trained from scratch, and can be retrained from this
repository (training takes about 8 minutes of CPU time per round on the machine used, the sum of the recorded
training times; the evaluation of each network on 2 x 10^5 new points comes on top):

```sh
cd research_highdim_bound/check
zsh run_c4.sh                                   # first round, 24 networks -> out/c4_fresh.jsonl
python r2_fresh_train.py out_r2/r2_fresh_new.jsonl   # second round, 21 networks, into a new file
cd ../..
```

`run_c4.sh` skips the networks already in `out/c4_fresh.jsonl` and `r2_fresh_train.py` those already in the
file it is given; rename `out/c4_fresh.jsonl` first to retrain all 24. The training runs of the remedies are
re-run with `python scripts/run_c1.py <phase>` in `research_highdim_remedies/` (phases and order in its
`README.md`; some hours on one core). The other re-checks are in the two `check/` folders.

### 3.5 Verification

`verify_research_benchmark/` and `verify_research_highdim/` contain the scripts and outputs of the
separately written re-computation; `notes/VERIFICATION.md` (sections 2 and 3) describes what each check does
and what it found. The scripts that only re-aggregate run without training:

```sh
(cd verify_research_benchmark && python v_stats.py && python v_summary.py)
(cd verify_research_highdim   && python scripts/v_agg.py && python scripts/v_robin_fd.py)
```

The re-run of the demonstration of Section 7.2 with the code of V-bench (three seeds, about 1.5 minutes each on
the CPU; seeds already in `v_oneface.json` are not trained again):

```sh
(cd verify_research_benchmark && python v_oneface.py)
```

## What to expect

* **Levels 1 and 2** reproduce the printed numbers. Stored floating-point values can differ in the last
  digit (for example the Monte-Carlo value `affine_mc` of `data/closed_forms.json`), and regenerated PDF
  figures can differ from the stored ones in embedded metadata.
* **Single-thread CPU runs** (Section 6) are reproducible bit for bit on the same software: the
  verification repeated seven of them and recovered the stored errors to all printed digits. None of the
  seven belongs to the 16 000-iteration runs at d = 20, which use the same code path but were not repeated.
* **Stacked runs on the GPU backend** (the 175 networks of the benchmark) use single precision. Repeated with
  unchanged code on the same machine and software, three stacks (Heat, 50 networks; Wave2, 25; stack B of the control experiment,
  20) reproduced every stored value bit for bit (Supplementary Section S8.4). The result of a network does,
  however, depend on the stack in which it is trained (its size and composition; the mechanism, presumably
  rounding, was not investigated): the same (strategy, seed) trained in a stack of 40 networks (four strategies,
  control experiment) and in one of 50 (five strategies, benchmark) gave L-BFGS errors that differ by a factor
  0.72 to 1.09. Other hardware was not tested;
  there, expect agreement in distribution, not digit for digit.
* **Timings** depend on the load of the machine. The article quotes three separate measurements wherever a
  cost of the low-dimensional problems is stated, and two for the dimension study, and uses only orders of
  magnitude.

## What cannot be re-run from this repository

* `code/s6_highdim_diag.py` (diagnostics of the d = 20 networks), the re-evaluation scripts of
  `verify_research_highdim/`, and the evaluation of the bound of Section 6.7 on the saved networks
  (`research_highdim_bound/scripts/s0_inventory.py`, `s4_eval_bound.py`; `check/c3_eval.py`) need the trained
  networks, which are not redistributed. Their outputs are stored (`data/s6_highdim_diag_d20.json`,
  `verify_research_highdim/results/`, `research_highdim_bound/results/eval_confirm.jsonl`,
  `research_highdim_bound/check/out/c3_eval.jsonl`). Re-running section 3.2 (16 000 iterations at d = 20)
  and section 3.3 recreates the networks.
* `code/sC_details_repro.py rerun` and `stacksize` need the Metal backend of PyTorch for a bit-level
  comparison with the stored stacks.
