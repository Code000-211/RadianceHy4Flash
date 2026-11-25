#!/bin/bash
# lane-pair.sh <cfg.yaml> <phases> [ENV=VAL ...]
# Both iks lanes at once: the same config on iks-A and on iks-B, both servers up together, probe.py
# against each, then both are stopped. It is what proves the two u3 servers share iks's 112 GB
# (notes/backlog/CARDS.md §1), and it is the served A/B of a card that needs both pairs.
#
# Run it inside an iks tree, through lane.sh:
#   scripts/backlog/lane.sh iks-A 'scripts/backlog/lane-pair.sh nospec.yaml c1,c8'
#
# Each lane's server is a session of its own, so stopping it stops radiance and `gpu2` with it and
# the pair goes back to the scheduler. Everything lands in runs/<lane>/<tag>/ on iks; copy a file
# back with scp iks:moebius-gpu/<name>/src/runs/... .
set -o pipefail
here=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
. "$here/lanes.sh"
RAD_REPO=$(rad_here); export RAD_REPO

cfg=${1:?usage: lane-pair.sh <cfg.yaml> <phases> [ENV=VAL ...]}
ph=${2:?usage: lane-pair.sh <cfg.yaml> <phases> [ENV=VAL ...]}
shift 2
while [ $# -gt 0 ]; do export "${1?}"; shift; done
tag=${TAG:-pair-$(date -u +%m%d-%H%M)}
lanes="iks-A iks-B"

for lane in $lanes; do
  lane_env "$lane" || exit 2
  [ -r "$RAD_MODEL" ] || { echo "lane-pair.sh: $lane has no model at $RAD_MODEL" >&2; exit 2; }
done

start() {  # start <lane>: the lane's server under `gpu2`, its own session, pid file for lane_stop
  local lane=$1 D=$RAD_RUNS/$1/$tag
  mkdir -p "$D"
  ( lane_env "$lane"
    lane_power || echo "start: a card is not at the 210 W cap" >&2
    mkdir -p "$D"
    lane_render "$cfg" "$D/$(basename "$cfg")" || exit 2
    case $RAD_NGRAM in ram=*|disk=*) mkdir -p "${RAD_NGRAM#*=}" ;; esac
    lane_ngram_reset
    setsid bash -c 'lane=$1 D=$2; echo $$ > "$D/$lane.pid"
      exec gpu2 "$RAD_BIN" --config "$D/$(basename "$3")"' _ "$lane" "$D" "$cfg"
  ) >> "$D/$lane.server.log" 2>&1 &
  echo "$!"
}

wait_lane() {  # wait_lane <lane>: up to 30 min for /health (a first Engram fill takes minutes)
  local lane=$1 D=$RAD_RUNS/$1/$tag port pgid i
  port=$( lane_env "$lane" >/dev/null; echo "$RAD_PORT" )
  for i in $(seq 1 1800); do
    lane_health "$port" && { echo "$lane healthy on $port after ${i}s"; return 0; }
    pgid=$(cat "$D/$lane.pid" 2>/dev/null)
    if [ -z "$pgid" ]; then
      [ "$i" -ge 60 ] && { echo "$lane: it never started (no $D/$lane.pid)" >&2; return 1; }
    elif ! kill -0 "$pgid" 2>/dev/null; then
      echo "$lane: the server exited before /health" >&2; tail -30 "$D/$lane.server.log" >&2; return 1
    fi
    sleep 1
  done
  echo "$lane: no /health on $port after 1800s" >&2; tail -20 "$D/$lane.server.log" >&2
  return 1
}

stop_lane() {  # stop_lane <lane>: the whole session, so radiance and gpu2 end together
  local lane=$1 D=$RAD_RUNS/$1/$tag pgid i
  pgid=$(cat "$D/$lane.pid" 2>/dev/null) || return 0
  [ -n "$pgid" ] || return 0
  kill -TERM -"$pgid" 2>/dev/null
  for i in $(seq 1 90); do kill -0 "$pgid" 2>/dev/null || break; sleep 1; done
  kill -9 -"$pgid" 2>/dev/null
  echo "$lane stopped"
}

cleanup() { for lane in $lanes; do stop_lane "$lane"; done; }
trap cleanup EXIT INT TERM

lane_power || echo "lane-pair.sh: a card is not at the 210 W cap" >&2
for lane in $lanes; do start "$lane"; done
rc=0
for lane in $lanes; do wait_lane "$lane" || rc=1; done
if [ $rc -eq 0 ]; then
  echo "both lanes healthy together: iks-A $(lane_env iks-A >/dev/null; echo $RAD_PORT)," \
       "iks-B $(lane_env iks-B >/dev/null; echo $RAD_PORT)"
  grep -E "MemTotal|MemAvailable|Cached" /proc/meminfo
else
  echo "lane-pair.sh: not both lanes up; the probe of a lane that is up is still run" >&2
fi
for lane in $lanes; do
  lane_env "$lane" >/dev/null
  D=$RAD_RUNS/$lane/$tag
  if lane_health "$RAD_PORT"; then
    python3 "$here/probe.py" "$RAD_PORT" "$D/$tag.json" "$ph" 2>&1
  else
    echo "$lane: not serving, no probe" >&2
  fi
done
grep -E "MemTotal|MemAvailable|Cached" /proc/meminfo
for lane in $lanes; do echo "$lane: runs/$lane/$tag"; done
exit $rc
