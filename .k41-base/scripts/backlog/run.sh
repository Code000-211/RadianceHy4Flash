#!/bin/bash
# run.sh <tag> <cfg.yaml> <phases> [ENV=VAL ...] [-- extra radiance args]
# One server in the lane's setting, probe.py against it, then it is stopped; everything lands in
# runs/<lane>/<tag>/ (the lane's rendered config, the server log, the probe JSON).
#
# Run it through lane.sh, which sets the lane's port, model and tiers and ships the tree to iks:
#   scripts/backlog/lane.sh local 'scripts/backlog/run.sh c1 nospec.yaml c1,c8'
#   scripts/backlog/lane.sh iks-A 'scripts/backlog/run.sh c1 nospec.yaml c1,c8'
# or directly, which is the local lane:
#   scripts/backlog/run.sh c1 nospec.yaml c1,c8
#
# ENV=VAL pairs are exported for the server and the probe (RADIANCE_AQL_STAMP=1 leaves the run's
# stamps in runs/<lane>/<tag>/<tag>.stamps.csv, the input of layer.py, agg.py and attr.py);
# everything after `--` goes to radiance itself.
set -o pipefail
here=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
. "$here/lanes.sh"
RAD_REPO=$(rad_here); export RAD_REPO

tag=${1:?usage: run.sh <tag> <cfg.yaml> <phases> [ENV=VAL ...] [-- radiance args]}
cfg=$2; ph=$3; shift 3
lane_begin "${RAD_LANE:-local}" || exit 2
D=$RAD_LANE_DIR/$tag
mkdir -p "$D"
lanecfg=$D/$(basename "$cfg")
lane_render "$cfg" "$lanecfg" || exit 2
echo "run $tag: $D"
echo "cfg $lanecfg, phases $ph"

X=(); args=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; X=("$@"); break; fi
  export "${1?}"; shift
done
for a in "${X[@]}"; do args+=" $(printf '%q' "$a")"; done
if [ -n "$RADIANCE_AQL_STAMP" ]; then
  export RADIANCE_AQL_STAMP_OUT=${RADIANCE_AQL_STAMP_OUT:-$D/$tag.stamps.csv}
fi

# The server and the probe run inside the lane's card reservation, so no other job takes the pair
# between the probe's requests.
inner=$(
  printf '"$RAD_BIN" --config %q%s > %q 2>&1 &\nP=$!\n' \
    "$lanecfg" "$args" "$D/server.log"
  cat <<'EOF'
. "$RAD_REPO/scripts/backlog/lanes.sh"    # lane_health
for i in $(seq 1 900); do
  lane_health "$RAD_PORT" && break
  kill -0 $P 2>/dev/null || break
  sleep 1
done
date -u +'%Y-%m-%dT%H:%M:%SZ'
if kill -0 $P 2>/dev/null; then
  python3 "$RAD_REPO/scripts/backlog/probe.py" "$RAD_PORT" "$OUT.json" "$PH" 2>&1
else
  echo "run.sh: the server did not answer /health on port $RAD_PORT" >&2
fi
kill $P 2>/dev/null
for i in $(seq 1 90); do kill -0 $P 2>/dev/null || break; sleep 1; done
kill -9 $P 2>/dev/null; wait $P; echo "server-exit=$?"
EOF
)
export OUT=$D/$tag PH=$ph
"$here/lane.sh" --gpu "${RAD_LANE:-local}" "$inner" > "$D/run.out" 2>&1
rc=$?
cat "$D/run.out"
[ $rc -eq 0 ] && [ -s "$D/$tag.json" ] ||
  echo "run.sh: no $D/$tag.json -- read $D/run.out and $D/server.log" >&2
echo "done $(date -u +'%Y-%m-%dT%H:%M:%SZ') rc=$rc"
exit $rc
