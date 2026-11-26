#!/bin/bash
P=/tmp/claude-0/-root-hy4/c566ea29-7612-537f-abca-9a9da726328e/scratchpad/pcie
cd $P
echo "SESSION4 start $(date -Is) ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES load $(cat /proc/loadavg)"
for c in /sys/class/drm/card*/device; do [ -e $c/hwmon ] || continue; echo "SESSION4 card $(basename $(readlink -f $c)) power1_cap $(cat $c/hwmon/hwmon*/power1_cap) uW"; done
PROBE_PASSES=1 PROBE_PAIRS="10:-1;40:-1;75:-1;78:-1;79:-1;10:10;78:78;79:79;10:11;40:41;75:76;10:40;10:75;40:75;10:79;75:79;78:10" ./region_probe 80 8 3
echo "SESSION4 end $(date -Is) load $(cat /proc/loadavg)"
