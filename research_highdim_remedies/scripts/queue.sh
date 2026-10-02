#!/bin/zsh
# -- added for the public repository: paths relative to the repository root
REPO="$(cd "$(dirname "$0")" && while [ ! -d research_benchmark ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
# Runs the counted phases in the pre-registered order, serially, at nice 19, one torch thread.
cd "$REPO/research_highdim_remedies"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=python
for ph in main20 eqcpu stall rep3 d10; do
  echo "=== $ph start $(date)" >> logs/queue.log
  nice -n 19 $PY scripts/run_c1.py $ph >> logs/$ph.log 2>&1 || { echo "=== $ph FAILED $(date)" >> logs/queue.log; exit 1; }
  echo "=== $ph done $(date)" >> logs/queue.log
done
