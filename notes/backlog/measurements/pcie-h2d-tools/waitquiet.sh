#!/bin/bash
# waits until the 1-minute load average is below $1 for two checks in a row (max $2 seconds)
t=0; ok=0
while [ $t -lt $2 ]; do
  l=$(cut -d' ' -f1 /proc/loadavg); if awk "BEGIN{exit !($l < $1)}"; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 2 ] && { echo "quiet: load $l after $t s"; exit 0; }
  sleep 20; t=$((t+20))
done
echo "still busy: load $(cat /proc/loadavg)"; exit 1
