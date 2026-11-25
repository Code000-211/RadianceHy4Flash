#!/bin/bash
# lane.sh: run a command in one of the backlog's lanes (notes/backlog/CARDS.md §1).
#
#   scripts/backlog/lane.sh lanes                        list the lanes
#   scripts/backlog/lane.sh <lane> <command>             run a shell command on the lane
#   scripts/backlog/lane.sh <lane> <command> [args...]   ... or run a program with arguments, as it is
#   scripts/backlog/lane.sh --gpu <lane> <command>       the same, on the lane's pair of cards
#   scripts/backlog/lane.sh --env [lane]                 print the lane's exports
#   scripts/backlog/lane.sh --in-tree <lane> <b64>       run a command inside an unpacked iks tree
#
# A lane sets RAD_LANE, RAD_PORT, RAD_MODEL, RAD_HOST_POOL_MIB, RAD_NGRAM, RAD_PREFIX_CACHE_MIB,
# RAD_BIN, RADIANCE_HOME, RAD_LANE_DIR (the lane's run directory, under runs/<lane>) and runs the
# command there:
#
#   local   here, as it is: the cards come from `gpu2` and from <lock dir>/gpu.lock, which one local
#           lane run holds at a time, so two of them never share the pair or the port.
#   iks-A   the tree is tarred to `ssh iks moebius-gpu/run.sh <name> <b64>`, unpacked into
#   iks-B   ~/moebius-gpu/<name>/src (the build survives between calls) and the command runs there;
#           its cards come from `gpu2` on iks. One iks call at a time, because the unpack replaces
#           the tree: `lane-pair.sh` runs both iks lanes together in one call.
#
# Example: scripts/backlog/lane.sh iks-A 'scripts/backlog/lane-pair.sh nospec.yaml c1,c8'
set -o pipefail
here=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
. "$here/lanes.sh"
RAD_REPO=$(rad_here); export RAD_REPO

usage() {
  sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

mode=run
case $1 in
  ''|-h|--help|help) usage; [ -n "$1" ] && exit 0; exit 2 ;;
  lanes) echo "$RAD_LANES" | tr ' ' '\n'; exit 0 ;;
  --env) lane_env "${2:-${RAD_LANE:-local}}" || exit 2
         # a remote lane's paths are under the home on that host, not this one's: show it as `~`
         vhome=${HOME:-/root}
         for v in RAD_LANE RAD_HOST RAD_REPO RAD_PORT RAD_MODEL RAD_HOST_POOL_MIB RAD_NGRAM \
                  RAD_PREFIX_CACHE_MIB RAD_BIN RADIANCE_HOME RAD_RUNS RAD_LANE_DIR \
                  RAD_LANE_LOCK_DIR RAD_CARDS; do
           val=${!v}
           [ "$RAD_HOST" = local ] || val=${val//"$vhome"/\~}
           printf 'export %s=%q\n' "$v" "$val"
         done
         exit 0 ;;
  --gpu|--in-tree) mode=$1; shift ;;
esac

lane=${1:?lane.sh: a lane is needed (lanes: $RAD_LANES)}; shift
lane_env "$lane" || exit 2

on_gpu() {  # the lane's pair, through this host's card scheduler
  local lock=$RAD_LANE_LOCK_DIR/gpu.lock
  mkdir -p "$RAD_LANE_LOCK_DIR"
  [ $# -eq 1 ] && set -- bash -c "$1"
  if [ "$RAD_HOST" = local ]; then
    exec flock "$lock" gpu2 "$@"
  fi
  exec gpu2 "$@"
}

run_here() {  # run the command here, in the lane's setting
  [ -x "$RAD_BIN" ] || echo "lane.sh: $RAD_BIN is missing -- build it first (build.sh or the CMake presets)" >&2
  if [ $# -eq 1 ]; then exec bash -c "$1"; fi
  exec "$@"
}

if [ "$mode" = --gpu ]; then
  on_gpu "$@"
elif [ "$mode" = --in-tree ]; then
  export RAD_IN_TREE=1
  run_here "$(printf %s "$1" | base64 -d)"
else
  case $RAD_HOST in
    local) run_here "$@" ;;
    iks)
      if [ -n "$RAD_IN_TREE" ]; then
        run_here "$@"
      else
        name=$(lane_name)
        case $name in ''|*/*|.*) echo "lane.sh: the tree name '$name' cannot be used on iks" >&2; exit 2 ;; esac
        if [ $# -eq 1 ]; then cmd=$1; else cmd=$(printf '%q ' "$@"); fi
        inner="scripts/backlog/lane.sh --in-tree $lane $(printf %s "$cmd" | base64 -w0)"
        echo "lane.sh: iks tree ~/moebius-gpu/$name/src, lane $lane" >&2
        # one iks call at a time: the unpack replaces ~/moebius-gpu/<name>/src
        mkdir -p "$RAD_LANE_LOCK_DIR"
        exec flock "$RAD_LANE_LOCK_DIR/iks.lock" bash -c '
          set -o pipefail
          git -C "$1" ls-files -z --cached --others --exclude-standard |
            tar --null -T - --ignore-failed-read -czf - |
            ssh iks moebius-gpu/run.sh "$2" "$3"' _ "$RAD_REPO" "$name" "$(printf %s "$inner" | base64 -w0)"
      fi
      ;;
  esac
fi
