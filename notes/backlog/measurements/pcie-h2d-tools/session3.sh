#!/bin/bash
P=/tmp/claude-0/-root-hy4/c566ea29-7612-537f-abca-9a9da726328e/scratchpad/pcie
cd $P
echo "SESSION3 start $(date -Is) ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES load $(cat /proc/loadavg)"
for c in /sys/class/drm/card*/device; do [ -e $c/hwmon ] || continue; echo "SESSION3 card $(basename $(readlink -f $c)) power1_cap $(cat $c/hwmon/hwmon*/power1_cap) uW"; done
echo "=== CPU read 8 threads alone"
CS_THREADS=8 CS_READONLY=1 ./cpu_stream 4 6 | grep -v '^CPUSTREAM-rep'
echo "=== both cards alone"
./h2d_bench --hosts engine --paths e0,kern --cases big --sets 01 --reps 7 | grep H2D
echo "=== both cards beside the 8-thread CPU read (overlap window printed)"
CS_THREADS=8 CS_READONLY=1 ./cpu_stream 4 120 > $P/bg3.log &
BG=$!
sleep 3
T0=$(python3 -c 'import time;print(time.clock_gettime(time.CLOCK_MONOTONIC))')
./h2d_bench --hosts engine --paths e0,kern --cases big --sets 01 --reps 40 | grep H2D | sed 's/^/[overlap] /'
T1=$(python3 -c 'import time;print(time.clock_gettime(time.CLOCK_MONOTONIC))')
sleep 2
kill $BG; wait $BG 2>/dev/null
echo "overlap window $T0 .. $T1"
awk -v a=$T0 -v b=$T1 '/CPUSTREAM-rep/{split($4,t,"="); if (t[2]>a+1.5 && t[2]<b) print "[overlap] cpu-read rep", $6; else print "[outside] cpu-read rep", $6}' $P/bg3.log
echo "SESSION3 load $(cat /proc/loadavg)"
echo "=== region probe 80 GiB, two passes, 8 threads"
PROBE_PASSES=2 ./region_probe 80 8 3
echo "SESSION3 end $(date -Is) load $(cat /proc/loadavg)"
