#!/bin/zsh
# -- added for the public repository: paths relative to the repository root
REPO="$(cd "$(dirname "$0")" && while [ ! -d research_benchmark ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
# PREREG_C1_P5.md blocks, serially, nice 19, one thread; each invocation < 14 min, resumable (finished runs skipped).
cd "$REPO/research_highdim_remedies"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=python
echo "=== harness check $(date) load $(sysctl -n vm.loadavg)" >> logs/queue_p5.log
timeout 840 nice -n 19 $PY scripts/check_harness_after_p5.py >> logs/harness_after_p5.log 2>&1 || { echo "=== harness FAILED" >> logs/queue_p5.log; exit 1; }
grep -q '"all_identical": true' results/check_harness_after_p5.json || { echo "=== harness NOT IDENTICAL, stop" >> logs/queue_p5.log; exit 1; }
for ph in sweepP5 p5main20 p5eqcpu p5dim; do
  for try in 1 2 3 4 5 6 7 8; do
    echo "=== $ph try $try $(date) load $(sysctl -n vm.loadavg)" >> logs/queue_p5.log
    timeout 840 nice -n 19 $PY scripts/run_c1.py $ph >> logs/$ph.log 2>&1 && break
  done
  echo "=== $ph done $(date)" >> logs/queue_p5.log
done
echo "=== queue_p5 finished $(date)" >> logs/queue_p5.log
