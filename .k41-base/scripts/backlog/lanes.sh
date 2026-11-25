# shellcheck shell=bash
# lanes.sh: the backlog's lanes and their settings, sourced by lane.sh and run.sh.
#
# A lane is a TP2 pair of R9700 at 210 W with the u3 container (notes/backlog/CARDS.md §1):
#
#   local    this host, cards 0 and 1         port 8041   Engram in RAM (`ngram-placement: ram`)
#   iks-A    ssh iks, a free pair of its four port 8042   Engram in a shared 60 GiB /dev/shm file
#   iks-B    ssh iks, the other free pair     port 8043   Engram from a 60 GiB file on the NVMe
#
# Local values are the served config (scripts/backlog/*.yaml as they are); the iks lanes take a
# smaller host pool and the Engram out of host RAM, because iks has 112 GB and two u3 servers
# cannot both pin ~60 GiB (docs/GUIDE.md §4.3, notes/backlog/CARDS.md §1).
#
# The prefix cache's host tier (`prefix-cache-mib`) is pinned host memory beside `host-pool-mib` × tp
# and the Engram table when the latter is in RAM, so it is sized with the lane's host memory too:
# 16384 MiB is the served value (README); the iks lanes hold half of it, 8192 MiB, because their host
# pool is half as large on a 112 GB host that two u3 servers share. `RAD_PREFIX_CACHE_MIB=0
# scripts/backlog/lane.sh <lane> 'scripts/backlog/run.sh ...'` runs a lane with the tier off, for the
# before and after of an item about it.
#
# lane_env <lane> exports the lane's settings; everything else here is a helper over them.

RAD_LANES="local iks-A iks-B"

homedir() {  # $HOME, or the passwd entry when a tool runs without one
  local h
  printf -v h '%s' ~
  printf '%s\n' "${h:-$HOME}"
}

rad_here() {  # the tree this file lives in, however it was reached
  printf '%s\n' "${RAD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
}

lane_host() {  # lane_host <lane>: local or iks
  case $1 in
    local) printf 'local\n' ;;
    iks-*) printf 'iks\n' ;;
    *) echo "lanes.sh: no lane '$1' (lanes: $RAD_LANES)" >&2; return 2 ;;
  esac
}

lane_env() {  # lane_env [lane]: export RAD_* for the lane (default: $RAD_LANE, else local)
  local lane=${1:-${RAD_LANE:-local}}
  local host
  host=$(lane_host "$lane") || return 2
  RAD_REPO=$(rad_here)
  export RAD_LANE=$lane RAD_HOST=$host RAD_REPO
  export RAD_BIN=$RAD_REPO/build/bin/radiance
  export RADIANCE_HOME=$RAD_REPO/build/radiance_home
  export RAD_RUNS=${RAD_RUNS:-$RAD_REPO/runs}
  export RAD_LANE_DIR=$RAD_RUNS/$lane
  export RAD_LANE_LOCK_DIR=${RAD_LANE_LOCK_DIR:-/tmp/radiance-lanes}
  export RAD_CARDS=2
  case $lane in
    local)
      RAD_PORT=8041
      RAD_MODEL=${RAD_MODEL:-/srv/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad}
      RAD_HOST_POOL_MIB=39000
      RAD_NGRAM=ram
      RAD_PREFIX_CACHE_MIB=${RAD_PREFIX_CACHE_MIB:-16384}
      ;;
    iks-A)
      RAD_PORT=8042
      RAD_MODEL=${RAD_MODEL:-$(homedir)/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad}
      RAD_HOST_POOL_MIB=16000
      RAD_NGRAM="ram=/dev/shm/radiance-ngram"
      RAD_PREFIX_CACHE_MIB=${RAD_PREFIX_CACHE_MIB:-8192}
      ;;
    iks-B)
      RAD_PORT=8043
      RAD_MODEL=${RAD_MODEL:-$(homedir)/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad}
      RAD_HOST_POOL_MIB=16000
      RAD_NGRAM="disk=$(homedir)/ngram"
      RAD_PREFIX_CACHE_MIB=${RAD_PREFIX_CACHE_MIB:-8192}
      ;;
  esac
  export RAD_PORT RAD_MODEL RAD_HOST_POOL_MIB RAD_NGRAM RAD_PREFIX_CACHE_MIB
}

lane_name() {  # the tree's name under ~/moebius-gpu on iks (RAD_IKS_NAME overrides)
  if [ -n "$RAD_IKS_NAME" ]; then printf '%s\n' "$RAD_IKS_NAME"; return; fi
  local repo run git
  run=$(basename "${RAD_REPO%/}")
  git=$(git -C "$RAD_REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) ||
    git=$(git -C "$RAD_REPO" rev-parse --git-common-dir 2>/dev/null) || git=$RAD_REPO/.git
  repo=$(basename "$(dirname "$git")")
  [ "$repo" = "$run" ] && repo=radiance
  printf '%s-%s\n' "$repo" "$run"
}

lane_power() {  # lane_power: print each card's power cap; 1 if one is not the 210 W of the lane
  local bad=0 f w
  for f in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do
    [ -r "$f" ] || continue
    w=$(( $(cat "$f") / 1000000 ))
    echo "cap $w W  ($f)"
    [ "$w" -eq 210 ] || bad=1
  done
  return $bad
}

lane_begin() {  # lane_begin <lane>: the lane's env, its directories and the power cap read back
  lane_env "$1" || return 2
  echo "lane $RAD_LANE on $RAD_HOST, port $RAD_PORT, model $RAD_MODEL" \
       "host-pool $RAD_HOST_POOL_MIB MiB, ngram $RAD_NGRAM," \
       " prefix-cache $RAD_PREFIX_CACHE_MIB MiB"
  lane_power || echo "lane_begin: a card is not at the 210 W cap of the lane -- numbers from this run" \
                        "are not the lane's (notes/backlog/CARDS.md §1)" >&2
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  mkdir -p "$RAD_LANE_DIR"
  case $RAD_NGRAM in
    ram=*|disk=*) mkdir -p "${RAD_NGRAM#*=}" ;;
  esac
  lane_ngram_reset
}

lane_ngram_reset() {  # a `ram=DIR` table is one process's pinned copy: a copy left by an earlier run
  # is dead tmpfs that the next start counts against MemAvailable and refuses to start beside (iks has
  # 112 GB and no swap), so a lane clears it first. A `disk=DIR` table is reusable and is left alone.
  case $RAD_NGRAM in
    ram=*) local d=${RAD_NGRAM#ram=}
           if [ -d "$d" ]; then rm -f "$d"/ngram-*.tab "$d"/ngram-*.tab.lock; fi ;;
  esac
}

lane_render() {  # lane_render <cfg.yaml> <out.yaml>: the cfg with this lane's port, model and tiers
  local cfg=$1
  [ -r "$cfg" ] || cfg=$(dirname "${BASH_SOURCE[0]}")/$cfg    # a bare name is one of scripts/backlog/*.yaml
  [ -r "$cfg" ] || { echo "lane_render: no config $1" >&2; return 2; }
  python3 "$RAD_REPO/scripts/backlog/lane-cfg.py" "$cfg" "$2"
}

lane_health() {  # lane_health <port>: 0 if the server answers /health (python3: iks has no curl)
  python3 - "$1" <<'EOF' 2>/dev/null
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=3).read()
EOF
}
