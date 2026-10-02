#!/bin/zsh
# -- added for the public repository: paths relative to the repository root
REPO="$(cd "$(dirname "$0")" && while [ ! -d research_benchmark ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
# serial driver for c4 fresh-seed runs; waits for c3 to finish first
cd "$REPO/research_highdim_bound/check"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
P=python
while pgrep -f c3_eval.py >/dev/null; do sleep 5; done
for spec in "poisson ritz 5" "poisson pinn 5" "p3 pinn 5" "p3 ritz 5" "p3 pinn 10" "p3 ritz 10" "poisson ritz 10" "poisson pinn 10"; do
  for s in 11 12 13; do
    set -- ${=spec}
    if grep -q "\"problem\": \"$1\", \"method\": \"$2\", \"d\": $3, \"seed\": $s," out/c4_fresh.jsonl 2>/dev/null; then continue; fi
    nice -n 19 timeout 840 $P c4_fresh_train.py $1 $2 $3 $s 4000 2>&1 | grep -v -i warn
  done
done
echo DONE
