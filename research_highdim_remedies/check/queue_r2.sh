#!/bin/bash
# serial queue for the plan of the second round (notes/VERIFICATION.md, section 4.3); each invocation < 14 min (resumable, finished runs skipped)
PY=python
cd "$(dirname "$0")"
for blk in "$@"; do
  for attempt in 1 2 3 4; do
    echo "== $blk attempt $attempt $(date +%T) load $(uptime | awk -F'averages: ' '{print $2}')"
    timeout 840 nice -n 19 $PY reimpl2.py $blk 2>&1 | grep --line-buffered -v -i -e warn -e detach -e cpu_total
    [ ${PIPESTATUS[0]} -eq 0 ] && break
  done
done
echo "== queue done $(date +%T)"
