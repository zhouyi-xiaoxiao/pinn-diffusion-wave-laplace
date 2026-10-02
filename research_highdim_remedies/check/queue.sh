#!/bin/bash
# serial queue; each invocation < 14 min (resumable, finished runs skipped)
PY=python
cd "$(dirname "$0")"
for blk in "$@"; do
  for attempt in 1 2 3; do
    echo "== $blk attempt $attempt $(date +%T) load $(uptime | awk -F'averages: ' '{print $2}')"
    timeout 840 nice -n 19 $PY reimpl.py $blk 2>&1 | grep --line-buffered -v -i warn
    [ ${PIPESTATUS[0]} -eq 0 ] && break
  done
done
echo "== queue done $(date +%T)"
