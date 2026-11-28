#!/bin/bash
# Issue #337: one agent session whose prompts grow past ~110k tokens, the shape that faulted device
# 0's queue inside gemm_i8_kpar_kernel<4, 256, 1, r4dchain> twice on `e3834473` (TP2, u3 container,
# num-speculative-tokens 5, prefix cache hot, W2 on).
#
# One server with the issue's settings, then scripts/agentreplay.py grows a single conversation from
# SYSTEM prompt tokens to CONTEXT in turns of ~1800-2200 tokens and decodes REPLY tokens each, so the
# verify pass -- the kernel's M is 1 + num-speculative-tokens -- runs at 110k-130k rows of history
# for a few thousand steps, which is where the two reports died. The startup lines and the two
# settings the report names are printed for the arm to be compared with `e3834473`.
#
# Exit status: 0 if the log holds no `the AQL queue faulted` and no `GPU memory fault` (the shape did
# not fault here); 1 if it holds one, and then the report lines are printed and kept in the run
# directory (a reproduction); 2 if the server did not come up -- its startup message names the
# --host-pool-mib or --max-model-len that would fit this box, and POOL / LEN are the knobs; 3 if the
# replay failed and the log holds no fault, which is not a reproduction of anything.
#
# The replay exits non-zero when the server dies under it, which is what a fault does, so its failure
# is reported and the log is read anyway: the fault report is the point. `--summarise DIR` prints the
# same verdict over a run directory without a card, and tests/repro_337_script_test.sh is that.
#
# The cards are not chosen here: run it under the card scheduler, as the lane does for a GPU job
#     scripts/backlog/lane.sh --gpu local scripts/repro-337.sh <tag>
# or `gpu2 scripts/repro-337.sh <tag>` on a free pair. Nothing else should be on the cards.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
BIN="$BUILD/bin/radiance"
TAG="${1:-$(date -u +%Y%m%d-%H%M%S)}"
OUT="${OUT:-${RAD_RUNS:-$ROOT/runs}/${RAD_LANE:-local}/repro-337-$TAG}"
MODEL="${MODEL:-/srv/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad}"
PORT="${PORT:-8041}"
TP="${TP:-2}"
# The issue's two knobs to narrow a fault with: NUM_SPECULATIVE_TOKENS and draft-coupling. Both keep
# the issue's values unless the environment asks for another, so the default arm is the reported one.
SPEC="${SPEC:-5}"
COUPLING="${COUPLING:-on}"
# The issue's two numbers; this box's budget may not hold both (see exit status 2).
POOL="${POOL:-20000}"
LEN="${LEN:-262144}"
SYSTEM="${SYSTEM:-95000}"        # system prompt tokens: where the session starts
CONTEXT="${CONTEXT:-130000}"     # a turn stops the session when its prompt would pass this
TURN="${TURN:-1800,2200}"        # tool result tokens a turn adds, LO,HI
REPLY="${REPLY:-2048}"           # tokens a turn decodes: past one 1984-token checkpoint
TOOLS="${TOOLS:-12}"              # the issue's session is a Hermes client with 12 tools: each turn
                                 # is a chat turn with that tool set, and the call it asks for is
                                 # answered before the next one
SUB_EVERY="${SUB_EVERY:-3}"      # a parent runs a subagent every N turns
SUB_TURNS="${SUB_TURNS:-3}"
export RADIANCE_HOME="$BUILD/radiance_home"   # absolute: the engine will not find the kernels without it

# The verdict over one run directory: the rows the replay wrote and the fault lines in the log, then
# the report itself when the log holds one. 0: no fault. 1: a reproduction. Never raises on a run
# directory a dead server left half written, because that is the interesting one.
verdict() {
  local out="$1" faults
  python3 - "$out/run.jsonl" <<'EOF'
import json, sys
rows = []
try:
    with open(sys.argv[1]) as h:
        for line in h:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass                 # a row cut short by a server that died mid-request
except FileNotFoundError:
    pass
if not rows:
    print("repro-337: no requests recorded")
else:
    print("repro-337: %d requests, largest prompt %d tokens, %d with a prefix-cache hit"
          % (len(rows), max(r["prompt_tokens"] for r in rows),
             sum(1 for r in rows if r.get("cached_tokens"))))
EOF
  faults="$(grep -cE 'the AQL queue faulted|GPU memory fault' "$out/server.log" || true)"
  echo "repro-337: $faults fault line(s) in $out/server.log"
  if [ "$faults" != 0 ]; then
    grep -nE -A 24 'the AQL queue faulted|GPU memory fault' "$out/server.log" || true
    return 1
  fi
  return 0
}

if [ "${1:-}" = "--summarise" ]; then
  [ -n "${2:-}" ] || { echo "usage: $0 --summarise <run directory>" >&2; exit 2; }
  vrc=0
  verdict "$2" || vrc=$?
  exit "$vrc"
fi

[ -x "$BIN" ] || { echo "no engine at $BIN (build it first; this script builds nothing)" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "no model at $MODEL" >&2; exit 2; }
mkdir -p "$OUT"
CFG="$OUT/cfg.yaml"
cat > "$CFG" <<EOF
server:
  host: 127.0.0.1
  port: $PORT
  kernels: libr4d,libavx,spechub
memory:
  gpu-headroom-mib: 128
models:
  hy4-flash:
    model: $MODEL
    tp: $TP
    pin: true
    placement: expert_tiered
    host-pool-mib: $POOL
    ngram-placement: disk
    max-model-len: $LEN
    max-num-batched-tokens: 2048
    checkpoint-interval: 1984
    checkpoint-slots: 128
    max-num-seqs: 4
    mm-max-patches: 0
    prefix-resume: nearest
    prefix-cache-mib: 4096
    busy-prefill-chunk: auto
    kv-cache-dtype: bf16
    control-vector: on
    num-speculative-tokens: $SPEC
    draft-window: fixed
    draft-coupling: $COUPLING
    draft-verify: traversal
    draft-depth: adaptive
    draft-depth-stop: 0.3
    draft-depth-load: on
    draft-depth-load-from: 1
    draft-suffix: on
EOF

"$BIN" --config "$CFG" > "$OUT/server.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
up=0
for _ in $(seq 1 900); do
  kill -0 $PID 2>/dev/null || break
  # The readiness probe is python3's, not curl's: the replay needs python3 on any box this runs on,
  # and a box without curl (iks) would otherwise sleep out the whole loop with the server up.
  if python3 "$ROOT/scripts/agentreplay.py" --probe "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    up=1; break
  fi
  sleep 1
done
if [ "$up" != 1 ]; then
  echo "repro-337: the server did not come up; its last lines:" >&2
  tail -n 30 "$OUT/server.log" >&2
  echo "repro-337: POOL and LEN are the knobs the message above names" >&2
  exit 2
fi
grep -E 'kv: the cache can grow|activation arena: |[[]0[]] arena: ' "$OUT/server.log" | sed 's/^/repro-337: /' || true
echo "repro-337: W2 $([ "${RADIANCE_W2:-1}" = 0 ] && echo off || echo on), $TP card(s), $SPEC speculative token(s), draft-coupling $COUPLING, host pool $POOL MiB, max-model-len $LEN, server up"

# A faulting server takes every request down with it, and the replay exits non-zero when that
# happens: that is the run this script exists for, so its failure is reported and the log is read
# anyway. `|| rrc=$?` keeps `set -e` and `pipefail` from stopping the script here.
rrc=0
python3 "$ROOT/scripts/agentreplay.py" --url "http://127.0.0.1:$PORT" --parents 1 \
  --system "$SYSTEM" --context "$CONTEXT" --turn "$TURN" --reply "$REPLY" \
  --sub-every "$SUB_EVERY" --sub-turns "$SUB_TURNS" --tools "$TOOLS" --timeout 1800 \
  --out "$OUT/run.jsonl" 2>&1 | tee "$OUT/replay.log" || rrc=$?
echo "repro-337: replay rc=$rrc"

kill $PID 2>/dev/null || true
for _ in $(seq 1 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -9 $PID 2>/dev/null || true
wait $PID 2>/dev/null || true

vrc=0
verdict "$OUT" || vrc=$?
if [ "$vrc" = 0 ] && [ "$rrc" != 0 ]; then
  echo "repro-337: the replay failed (rc=$rrc) and the log holds no fault: not a reproduction" >&2
  vrc=3
fi
exit "$vrc"
