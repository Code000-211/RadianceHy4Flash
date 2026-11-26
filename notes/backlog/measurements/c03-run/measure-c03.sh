#!/usr/bin/env bash
# C-03 A/B: base (cf11ed50) vs head (this branch), three interleaved boots an arm, one boot a run,
# in two configs — DFly2 coupled (`cpv`, the house config) and no speculation (`nsv`).
#
# Run it under the lane lock, whole:
#   flock -w 7200 /tmp/hy4-lane.lock -c '/home/ubuntu/projects/wt-c03/notes/backlog/measurements/c03-run/measure-c03.sh'
#
# Every boot is a `systemd-run --user` transient unit on a PRIVATE port (:8199) with the model on
# both R9700s; the script stops the house engine first and stops each arm's engine before the next.
# It does NOT restore the house engine: under the lane handoff the parent restores hy4-u4 once the
# worker queue drains, so :8180 is left free on purpose (and is checked to be free).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/home/ubuntu/r9700-serving/hy4flash/c03-measure-$(date +%Y%m%d-%H%M)}
# A private port, not :8180: other workers' probes post to :8180 blindly, and during the first attempt
# their concurrent c8 streams landed in a measured boot and tripled the streamed bytes. Nothing else
# on the host knows this port, so a measured boot only ever serves this script's own requests.
PORT=${PORT:-8199}
BASE=/home/ubuntu/projects/wt-c03-base   # cf11ed50
HEAD=/home/ubuntu/projects/wt-c03        # this branch
PROBE=$OUT/probe
ROUNDS=${ROUNDS:-3}
ARMS=${ARMS:-"base head"}
CFGS=${CFGS:-"cpv nsv"}
PHASES=${PHASES:-c1,c8,greedy,pf20}
mkdir -p "$OUT" "$PROBE"
cp "$HEAD/scripts/backlog/probe.py" "$HEAD/scripts/backlog/prompts.py" "$PROBE"/
sed -i 's/"model": "hy4-flash"/"model": "qwen3.8-27b"/g' "$PROBE/probe.py"
export RAD_SRC=$HEAD
export RADIANCE_LOG=debug

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/status.log"; }

bin_of() { case $1 in base) echo "$BASE/build/bin/radiance";; head) echo "$HEAD/build/bin/radiance";; esac; }
home_of() { case $1 in base) echo "$BASE/build/radiance_home";; head) echo "$HEAD/build/radiance_home";; esac; }
cfg_of() { case $1 in cpv) echo "$HERE/c03-cpv.yaml";; nsv) echo "$HERE/c03-nsv.yaml";; esac; }

meta() {
  {
    echo "measured_by: krisf (Hermes agent)"
    echo "date: $(date -Is)"
    echo "host: $(hostname) kernel $(uname -r)"
    echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
    echo "lane: local hy4-flash lane, private port $PORT, house alias qwen3.8-27b / qwen3.8-27b-local"
    echo "gpus: $(amd-smi static --asic 2>/dev/null | grep -E 'MARKET_NAME' | sed 's/.*: //' | paste -sd'|')"
    echo "gpu_ids: $(for d in /sys/class/drm/card*/device; do [ -e "$d/vendor" ] || continue; [ "$(cat "$d/vendor" 2>/dev/null)" = "0x1002" ] && echo "$(basename "$(dirname "$d")"):$(cat "$d/device")"; done | paste -sd' ')"
    echo "tp: 2"
    for d in /sys/class/drm/card*/device; do
      [ -e "$d/vendor" ] || continue
      [ "$(cat "$d/vendor" 2>/dev/null)" = "0x1002" ] || continue
      for h in "$d"/hwmon/hwmon*; do
        v=$(cat "$h/power1_cap" 2>/dev/null || true)
        [ -n "$v" ] && echo "power_cap_$(basename "$(dirname "$d")"): $v uW"
      done
    done
    echo "rocm_version_file: $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
    echo "amd_smi: $(amd-smi version 2>/dev/null | head -2 | paste -sd' ')"
    echo "container: hy4-flash-iqr-v4-u4.rad path=/home/ubuntu/models/radiance/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u4.rad size=$(stat -c%s /home/ubuntu/models/radiance/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u4.rad) mtime=$(stat -c%y /home/ubuntu/models/radiance/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u4.rad)"
    echo "ngram_placement: disk"
    echo "configs: cpv=$HERE/c03-cpv.yaml nsv=$HERE/c03-nsv.yaml"
    echo "commit_base: $(git -C "$BASE" rev-parse HEAD) ($(git -C "$BASE" log -1 --format=%s))"
    echo "commit_head: $(git -C "$HEAD" rev-parse HEAD) ($(git -C "$HEAD" log -1 --format=%s))"
    echo "binary_base: $(sha256sum "$(bin_of base)" | cut -c1-16) $(bin_of base)"
    echo "binary_head: $(sha256sum "$(bin_of head)" | cut -c1-16) $(bin_of head)"
    echo "probe_phases: $PHASES"
    echo "env: RADIANCE_AQL_STAMP=50 RADIANCE_LOG_STEPS=1 RADIANCE_LOG=debug"
    echo "lock: flock -w 7200 /tmp/hy4-lane.lock (held for the whole run)"
  } | tee "$OUT/meta.txt"
}

port_free() { for _ in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0; sleep 2; done; return 1; }
wait_ready() {
  for i in $(seq 1 200); do
    if curl -s --max-time 3 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 'qwen3.8-27b'; then
      log "  READY ~$((i))s"; return 0
    fi
    st=$(systemctl --user is-active c03-meas 2>/dev/null || true)
    if [ "$st" != "active" ] && [ "$st" != "activating" ]; then
      log "  ENGINE DIED ($st)"; tail -15 "$OUT/$1-boot.log" 2>/dev/null | tee -a "$OUT/status.log"; return 1
    fi
    sleep 1
  done
  log "  !! not ready"; return 1
}

boot() {   # arm, round, cfgname
  local arm=$1 r=$2 cfg=$3
  systemctl --user stop hy4-u4 2>/dev/null
  systemctl --user stop c03-meas 2>/dev/null
  port_free || { log "  !! :$PORT held"; return 1; }
  systemctl --user reset-failed c03-meas 2>/dev/null || true
  local wd; wd=$(realpath "$(dirname "$(bin_of "$arm")")/..")
  log "  boot loadavg=$(cut -d' ' -f1-3 /proc/loadavg) swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{printf "%.0f/%.0f MiB", (t-f)/1024, t/1024}' /proc/meminfo)"
  systemd-run --user --unit=c03-meas --description="c03 meas $arm r$r $cfg" \
    -p LimitMEMLOCK=infinity -p WorkingDirectory="$wd" -p Restart=no \
    --setenv=RADIANCE_HOME="$(home_of "$arm")" \
    $( [ "${STAMP:-50}" != "0" ] && echo --setenv=RADIANCE_AQL_STAMP="$STAMP" \
                                     --setenv=RADIANCE_AQL_STAMP_OUT="$OUT/$arm-$cfg-r$r-stamp.csv" ) \
    --setenv=RADIANCE_LOG_STEPS=1 \
    --setenv=RADIANCE_LOG=debug \
    bash -c "exec '$(bin_of "$arm")' --config '$(cfg_of "$cfg")' >> '$OUT/$arm-$cfg-r$r-boot.log' 2>&1" 2>>"$OUT/status.log" | tee -a "$OUT/status.log" >/dev/null
  wait_ready "$arm-$cfg-r$r"
}

stop() {
  systemctl --user stop c03-meas 2>/dev/null
  for _ in $(seq 1 90); do systemctl --user is-active --quiet c03-meas || break; sleep 2; done
  port_free
}

probe_phase() {   # arm, cfg, r, phase
  local arm=$1 cfg=$2 r=$3 ph=$4 tag="$arm-$cfg-r$r"
  local off0 off1 k reps=1
  # c1 and c8 are repeated inside one log window: the step loop prints its 128-step phase line
  # (`ranks`) only once every 128 steps, and a 256-token DFly2 decode is ~73 steps, so one
  # standard phase would often not cross a line. Three standard phases back to back do.
  case $ph in c1|c8) reps=3;; esac
  off0=$(wc -l < "$OUT/$tag-boot.log" 2>/dev/null || echo 0)
  for k in $(seq 1 $reps); do
    timeout 900 python3 "$PROBE/probe.py" "$PORT" "$OUT/$tag-$ph-$k.json" "$ph" \
      > "$OUT/$tag-$ph-$k.probe.log" 2>&1
    log "  $tag $ph $k rc=$?"
  done
  off1=$(wc -l < "$OUT/$tag-boot.log" 2>/dev/null || echo 0)
  echo "$ph $off0 $off1" >> "$OUT/$tag.logwindow"
  return 0
}

run_boot() {  # arm, r, cfg
  local arm=$1 r=$2 cfg=$3 tag="$arm-$cfg-r$r"
  boot "$arm" "$r" "$cfg" || { stop; return 1; }
  # a discarded warm-up: one 24.9K-token prefill exercises every layer's experts broadly and two
  # c1 passes settle the heat map, so the measured passes do not carry the boot's page-in
  timeout 600 python3 "$PROBE/probe.py" "$PORT" "$OUT/$tag-warm.json" pf20 > "$OUT/$tag-warm.log" 2>&1
  timeout 600 python3 "$PROBE/probe.py" "$PORT" "$OUT/$tag-warm2.json" c1 >> "$OUT/$tag-warm.log" 2>&1
  timeout 600 python3 "$PROBE/probe.py" "$PORT" "$OUT/$tag-warm3.json" c1 >> "$OUT/$tag-warm.log" 2>&1
  : > "$OUT/$tag.logwindow"
  local ph rc=0
  for ph in ${PHASES//,/ }; do
    probe_phase "$arm" "$cfg" "$r" "$ph" || rc=1
  done
  curl -s --max-time 10 http://127.0.0.1:$PORT/metrics > "$OUT/$tag.metrics.txt" 2>/dev/null
  stop
  return $rc
}

meta
systemctl --user stop hy4-u4 2>/dev/null; port_free || { log "!! :$PORT held before the run"; exit 3; }

for r in $(seq 1 "$ROUNDS"); do
  for cfg in $CFGS; do
    # alternate the arm order every round: an interleaved base/head pair per round, with the
    # first boot of a round alternating between the arms so neither arm always pays the cold start
    if [ $(( (r + ${#cfg}) % 2 )) -eq 0 ]; then order=$ARMS; else order=$(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' '); fi
    for arm in $order; do
      log "== round $r $cfg $arm"
      run_boot "$arm" "$r" "$cfg" || log "   $arm r$r $cfg FAILED"
    done
  done
done

# Lane handoff (owner, 2026-11-26): do NOT restore the house engine. The parent restores hy4-u4
# once the whole worker queue has drained, so restoring here would only restart the engine between
# workers. Leave hy4-u4 stopped and :8180 free, and say so in the report.
log "leaving hy4-u4 stopped and :8180 free for the parent's restore (lane handoff)"
systemctl --user stop c03-meas 2>/dev/null
if ss -ltn 2>/dev/null | grep -qE ':(8180|'$PORT') '; then
  log "!! a lane port is still held after the run"
else
  log ":$PORT free"
fi
log "RESULTS IN $OUT"