#!/usr/bin/env bash
# X-03 (#130) lane measurement, one replicate: the same /tmp/wt-x03 build twice, with the rolling
# decode snapshot off (0) and on (256), on the local lane (port 8041, u4 model, TP2, DFly2
# speculation on). Caller holds /tmp/hy4-lane.lock; the house engine is stopped for the run and
# started again by the EXIT trap. Exit 0 only if both arms booted and each left a complete probe
# JSON (longresume with three runs, c1, c8, greedy); any arm that did not measure fails the run.
set -uo pipefail
REP=${1:?usage: x03-measure.sh <rep>}
cd /tmp/wt-x03 || exit 2
OUT=runs/x03/$REP
mkdir -p "$OUT"
BIN=./build/bin/radiance
export RADIANCE_HOME=$PWD/build/radiance_home
: > "$OUT/status"
say() { echo "$* $(date -u +%H:%M:%SZ)" | tee -a "$OUT/status"; }

restore() {
  say "restore: starting the house engine"
  systemctl --user start hy4-u4 2>>/tmp/wt-x03/"$OUT"/status \
    || (cd /home/ubuntu/r9700-serving/hy4flash && ./run-hy4-u4-disk.sh >>/tmp/wt-x03/"$OUT"/status 2>&1)
  for i in $(seq 1 120); do
    if curl -s --max-time 3 http://127.0.0.1:8180/v1/models | grep -q 'qwen3.8-27b-local'; then
      say "restore: hy4-u4 answers on :8180"; return
    fi
    sleep 5
  done
  say "restore: FAILED -- hy4-u4 does not answer on :8180"
}
trap restore EXIT

say "stop: hy4-u4"
systemctl --user stop hy4-u4
for i in $(seq 1 120); do ss -ltn 2>/dev/null | grep -q ':8180 ' || break; sleep 1; done
for i in $(seq 1 90); do
  f0=$(amd-smi metric --mem-usage 2>/dev/null | awk 'index($0,"GPU: 0"){on=1} on&&/FREE_VRAM:/{print $2; exit}')
  f1=$(amd-smi metric --mem-usage 2>/dev/null | awk 'index($0,"GPU: 1"){on=1} on&&/FREE_VRAM:/{print $2; exit}')
  if [ "${f0:-0}" -gt 20000 ] && [ "${f1:-0}" -gt 20000 ]; then say "lane free: GPU0 ${f0} MiB, GPU1 ${f1} MiB"; break; fi
  sleep 2
done

# Ready = the boot's own "listening" line and /v1/models naming the probe's model id, so neither a
# half-loaded engine nor a stale listener on the port is mistaken for it.
ready() {
  local port=$1 pid=$2 log=$3
  for i in $(seq 1 180); do
    if grep -q 'server: listening on' "$log" 2>/dev/null &&
       curl -s --max-time 2 "http://127.0.0.1:$port/v1/models" | grep -q '"id":"hy4-flash"'; then
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}

check_results() {
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ph = {p.get("name"): p for p in d["phases"]}
lr = ph.get("longresume")
assert lr and len(lr["runs"]) >= 3, "longresume missing or short"
for k in ("c1", "c8", "greedy"):
    assert k in ph, f"{k} missing"
r = lr["runs"]
print("turn2_prefill", [x["turn2"]["prefill_tokens"] for x in r])
print("turn2_sha256", [x["turn2"]["sha256"][:16] for x in r])
print("turn1_ckpt", [x["turn1"]["checkpoints_written"] for x in r])
print("c1", round(ph["c1"]["agg_tps"], 1), "c8", round(ph["c8"]["agg_tps"], 1))
PY
}

rc=0
run_arm() {
  local tag=$1 iv=$2 cfg=$OUT/$1.yaml
  mkdir -p "$OUT/$tag"
  sed "s/^    decode-checkpoint-interval: .*/    decode-checkpoint-interval: $iv/" x03-lane.yaml > "$cfg"
  say "arm $tag (interval $iv): boot"
  setsid "$BIN" --config "$cfg" > "$OUT/$tag/server.log" 2>&1 &
  local pid=$!
  if ! ready 8041 "$pid" "$OUT/$tag/server.log"; then
    say "arm $tag: FAILED TO BECOME READY -- see $OUT/$tag/server.log"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rc=1; return 1
  fi
  say "arm $tag: ready on :8041 (/v1/models has hy4-flash); probe longresume24000,c1,c8,greedy"
  RAD_RESUME_RUNS=3 RAD_RESUME_REPLY=1024 RAD_RESUME_NEW=64 \
    python3 scripts/backlog/probe.py 8041 "$OUT/$tag/probe.json" longresume24000,c1,c8,greedy \
    > "$OUT/$tag/probe.log" 2>&1
  local pr=$?
  say "arm $tag: probe rc=$pr"
  kill "$pid" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  if [ $pr -ne 0 ] || [ ! -s "$OUT/$tag/probe.json" ]; then
    say "arm $tag: NO MEASUREMENT (probe rc=$pr)"; rc=1; return 1
  fi
  if ! check_results "$OUT/$tag/probe.json" > "$OUT/$tag/result.txt" 2>&1; then
    say "arm $tag: INCOMPLETE MEASUREMENT -- $OUT/$tag/result.txt"; rc=1; return 1
  fi
  cat "$OUT/$tag/result.txt" | tee -a "$OUT/status"
  sleep 5
}

run_arm arm0 0
run_arm arm256 256

{
  echo "=== lane record $(date -u +%FT%TZ)"
  echo "model $(stat -c '%n %s %y' /home/ubuntu/models/radiance/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u4.rad)"
  echo "prefix64m $(head -c 67108864 /home/ubuntu/models/radiance/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u4.rad | sha256sum | cut -d' ' -f1)"
  for c in 0 1; do
    echo "card $c power1_cap=$(cat /sys/class/drm/card$((c+1))/device/hwmon/hwmon$((8+c))/power1_cap 2>/dev/null) default=$(cat /sys/class/drm/card$((c+1))/device/hwmon/hwmon$((8+c))/power1_cap_default 2>/dev/null)"
  done
  echo "kernel $(uname -r)"
  echo "binary $(stat -c '%s %y' build/bin/radiance)"
  git -C /tmp/wt-x03 log --oneline -3
} > "$OUT/lane-record.txt" 2>&1
say "replicate $REP done rc=$rc"
exit $rc