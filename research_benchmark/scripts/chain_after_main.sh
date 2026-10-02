#!/bin/zsh
# -- added for the public repository: paths relative to the repository root
REPO="$(cd "$(dirname "$0")" && while [ ! -d research_benchmark ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
# Sequential GPU jobs after the first seed block (one process at a time on MPS).
cd "$REPO/research_benchmark"
PY=python
until grep -q "^DONE" results/logs/run_benchmark_s0-5_b.log; do sleep 5; done
mkdir -p results/runs_superseded
mv results/runs/heat1d_s0-5.json results/runs_superseded/heat1d_s0-5_separate-forward-passes.json
mv results/runs/heat1d_pred_seed0.npz results/runs_superseded/
$PY -u scripts/run_benchmark.py heat1d 0 10 mps > results/logs/run_benchmark_heat1d_s0-10.log 2>&1
$PY -u scripts/run_benchmark.py laplace2d 5 10 mps > results/logs/run_benchmark_laplace2d_s5-10.log 2>&1
$PY -u scripts/run_budget_sweep.py heat1d 64,1024 mps > results/logs/run_budget_sweep_heat1d.log 2>&1
echo CHAIN_DONE > results/logs/chain.done
