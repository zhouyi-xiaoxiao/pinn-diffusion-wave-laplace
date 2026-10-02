#!/bin/sh
# Launch the 16 000-iteration runs at d = 20 (Section 6.3): one single-thread worker per (problem, seed), three at a time.
# Usage: sh s6_highdim_long_d20_launch.sh /path/to/python
PY="${1:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$HERE/../data/s6_highdim_long_d20_logs"
mkdir -p "$LOGS"
for problem in laplace poisson; do
  for seed in 0 1 2; do
    "$PY" "$HERE/s6_highdim_long_d20.py" run "$problem" "$seed" > "$LOGS/${problem}_s${seed}.log" 2>&1 &
  done
  wait
done
"$PY" "$HERE/s6_highdim_long_d20.py" merge > "$LOGS/merge.log" 2>&1
