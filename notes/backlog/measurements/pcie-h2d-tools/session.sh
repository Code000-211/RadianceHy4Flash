#!/bin/bash
# GPU session for the pcie-h2d measurement; run as: flock /tmp/radiance-lanes/gpu.lock gpu2 session.sh
P=/tmp/claude-0/-root-hy4/c566ea29-7612-537f-abca-9a9da726328e/scratchpad/pcie
cd $P
echo "SESSION start $(date -Is) ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
echo "SESSION radiance processes: $(ps -eo pid,args | grep -E 'radiance|rad_serve|rad-serve|/build/bin/' | grep -v -E 'grep|pi-bolt|session.sh' | wc -l)"
for c in /sys/class/drm/card*/device; do [ -e $c/hwmon ] || continue; s=$(basename $(readlink -f $c)); echo "SESSION card $s power1_cap $(cat $c/hwmon/hwmon*/power1_cap) uW  link $(cat $c/current_link_speed) x$(cat $c/current_link_width)"; done
echo "SESSION thp enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled)  defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag)  hugetlb pool: $(grep HugePages_Total /proc/meminfo)"
# link-speed sampler on both root ports
( while :; do echo "$(date +%s.%N) $(cat /sys/bus/pci/devices/0000:40:01.1/current_link_speed | cut -d' ' -f1) $(cat /sys/bus/pci/devices/0000:c0:01.1/current_link_speed | cut -d' ' -f1)"; sleep 0.25; done ) > $P/linkspeed.log &
SAMP=$!
echo "SESSION load $(cat /proc/loadavg)"
echo "=== CPU STREAM (start)"
./cpu_stream 2 5
echo "SESSION load $(cat /proc/loadavg)"
echo "=== matrix (one process, both devices)"
./h2d_bench --reps 7
echo "SESSION load $(cat /proc/loadavg)"
echo "=== regthp repeat (order check)"
./h2d_bench --hosts regthp,engine,regthp --paths hip1,e0,kern --cases big,l058 --sets "0;01" --reps 7
echo "SESSION load $(cat /proc/loadavg)"
echo "=== two processes, one card each, barrier in /dev/shm"
IFS=, read -r A B <<< "$ROCR_VISIBLE_DEVICES"
N=pcieh2d.$$
ROCR_VISIBLE_DEVICES=$B ./h2d_bench --hosts engine,hipmalloc,reg4k --paths hip1,e0,e01,kern --cases big,l058 --sets 0 --reps 7 --shm $N --rank 1 --nproc 2 > $P/proc1.log 2>&1 &
R1=$!
ROCR_VISIBLE_DEVICES=$A ./h2d_bench --hosts engine,hipmalloc,reg4k --paths hip1,e0,e01,kern --cases big,l058 --sets 0 --reps 7 --shm $N --rank 0 --nproc 2
wait $R1
sed 's/^/[rank1] /' $P/proc1.log
echo "SESSION load $(cat /proc/loadavg)"
echo "=== CPU STREAM (end)"
./cpu_stream 2 5
echo "=== card DMA beside a 16-thread CPU read stream (contention)"
CS_THREADS=16 CS_READONLY=1 ./cpu_stream 4 40 > $P/bgstream.log &
BG=$!
sleep 1
./h2d_bench --hosts engine --paths e0,kern --cases big,l058 --sets "0;01" --reps 7 | sed 's/^/[cpu-load] /'
kill $BG 2>/dev/null; wait $BG 2>/dev/null
cat $P/bgstream.log | sed 's/^/[cpu-load] /'
echo "=== M-20 tool (m20_xpbench_test, build of /root/hy4/wt/m20)"
( cd /root/hy4/wt/m20 && timeout 300 build/tests/m20_xpbench_test m20_ ) 2>&1 | grep -E 'M20XP|ok|FAIL|SKIP'
echo "SESSION load $(cat /proc/loadavg)  $(grep MemAvailable /proc/meminfo)"
echo "=== region probe: 1 GiB THP chunks, CPU read vs both-card DMA per chunk"
./region_probe ${PROBE_GIB:-80} 16 3
echo "SESSION load $(cat /proc/loadavg)"
kill $SAMP
echo "SESSION link speeds seen (root ports 40:01.1 c0:01.1):"; awk '{print $2, $3}' $P/linkspeed.log | sort | uniq -c
for c in /sys/class/drm/card*/device; do [ -e $c/hwmon ] || continue; echo "SESSION end card $(basename $(readlink -f $c)) power1_cap $(cat $c/hwmon/hwmon*/power1_cap)"; done
echo "SESSION load $(cat /proc/loadavg)"
echo "SESSION end $(date -Is)"
