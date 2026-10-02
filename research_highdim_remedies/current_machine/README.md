# Current-machine PINN replication

This separate, retrospective descriptive cohort covers all three seeds (40–42) for the PINN comparisons on LapD and SinLinD at dimension 20. It does not alter the earlier frozen-study or re-check records.

- `B1-runs.jsonl`: nine LapD runs (plain, presolve and lift3c at 4000 iterations) followed by three lift3c runs at a CPU-calibrated budget.
- `B4-runs.jsonl`: twelve SinLinD runs (plain, presolve, lift3c and lift1 at 4000 iterations).
- `checkpoints/`: all 24 trained state dictionaries, configurations and metrics; each record includes its SHA-256 hash.
- `B1-meta.json`, `B4-meta.json`: environment, source hashes, seeds, start/finish times and completion state.
- `summary.json`: independently reloaded checkpoint errors on the specified 50,000 test points and seed-paired comparisons.
- `run_cohort.py`: portable driver using the separately written `../check/reimpl.py` and `reimpl2.py` implementation.
- `analyze_cohort.py`: read-only by default; `--output` and `--table-output` explicitly write recomputed outputs.

The recorded environment is an Apple M4 Pro CPU, macOS 26.7 arm64, Python 3.14.6, PyTorch 2.14.1 and NumPy 2.5.3. Two processes ran concurrently, each with one PyTorch thread. The original architecture, loss weights, sampling generators, seed mapping, schedules and evaluation set are unchanged. The full source is linked by the metadata hashes. Timing uses process CPU time, not elapsed wall time.

## Outcomes

The error ratio is plain divided by the tested arm, paired by seed. On LapD, median ratios are 3.56 for presolve and 2.86 for lift3c, each improving all three seeds. The calibrated lift3c runs use 2500 iterations: the measured median CPU ratio is 1.105, so this is calibration with a 10.5% realised overshoot, not an exact equal-CPU comparison. The error ratio is 1.31 (three wins), below the 1.5 threshold for a clear gain.

On SinLinD, presolve worsens all three seeds (median ratio 0.87). lift3c and lift1 improve all three (2.75 and 2.57). These results do not support a uniformly beneficial remedy or feature mechanism.

## Reproduction

From the repository root:

```sh
python research_highdim_remedies/current_machine/analyze_cohort.py
python research_highdim_remedies/current_machine/analyze_cohort.py --output research_highdim_remedies/current_machine/summary.json --table-output data/current_machine_table.tex
```

For training into a fresh directory, use `run_cohort.py B1 --output-dir /tmp/pinn-new-cohort` and `run_cohort.py B4 --output-dir /tmp/pinn-new-cohort`, then `analyze_cohort.py --input-dir /tmp/pinn-new-cohort`. Existing completed records are skipped. A fresh run recalibrates its own CPU budget; it need not choose the released iteration count or reproduce timing on different hardware.
