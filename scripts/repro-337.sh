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
# --host-pool-mib or --max-model-len that would fit this box, and POOL / LEN are the knobs.
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
# The issue's two numbers; this box's budget may not hold both (see exit status 2).
POOL="${POOL:-20000}"
LEN="${LEN:-262144}"
SYSTEM="${SYSTEM:-95000}"        # system prompt tokens: where the session starts
CONTEXT="${CONTEXT:-130000}"     # a turn stops the session when its prompt would pass this
TURN="${TURN:-1800,2200}"        # tool result tokens a turn adds, LO,HI
REPLY="${REPLY:-2048}"           # tokens a turn decodes: past one 1984-token checkpoint
SUB_EVERY="${SUB_EVERY:-3}"      # a parent runs a subagent every N turns
SUB_TURNS="${SUB_TURNS:-3}"
export RADIANCE_HOME="$BUILD/radiance_home"   # absolute: the engine will not find the kernels without it
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
    num-speculative-tokens: 5
    draft-window: fixed
    draft-coupling: on
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
  if curl -sf -m 2 "http://127.0.0.1:$PORT/health" > /dev/null; then up=1; break; fi
  sleep 1
done
if [ "$up" != 1 ]; then
  echo "repro-337: the server did not come up; its last lines:" >&2
  tail -n 30 "$OUT/server.log" >&2
  echo "repro-337: POOL and LEN are the knobs the message above names" >&2
  exit 2
fi
grep -E 'kv: the cache can grow|activation arena: |[[]0[]] arena: ' "$OUT/server.log" | sed 's/^/repro-337: /' || true
echo "repro-337: W2 $([ "${RADIANCE_W2:-1}" = 0 ] && echo off || echo on), $TP card(s), server up"

python3 "$ROOT/scripts/agentreplay.py" --url "http://127.0.0.1:$PORT" --parents 1 \
  --system "$SYSTEM" --context "$CONTEXT" --turn "$TURN" --reply "$REPLY" \
  --sub-every "$SUB_EVERY" --sub-turns "$SUB_TURNS" --timeout 1800 --out "$OUT/run.jsonl" \
  2>&1 | tee "$OUT/replay.log"
echo "repro-337: replay rc=$?"

kill $PID 2>/dev/null || true
for _ in $(seq 1 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -9 $PID 2>/dev/null || true
wait $PID 2>/dev/null || true

python3 - "$OUT/run.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print("repro-337: %d requests, largest prompt %d tokens, %d with a prefix-cache hit"
      % (len(rows), max(r["prompt_tokens"] for r in rows),
         sum(1 for r in rows if r.get("cached_tokens"))))
EOF
FAULTS="$(grep -cE 'the AQL queue faulted|GPU memory fault' "$OUT/server.log" || true)"
echo "repro-337: $FAULTS fault line(s) in $OUT/server.log"
if [ "$FAULTS" != 0 ]; then
  grep -nE -A 14 'the AQL queue faulted|GPU memory fault' "$OUT/server.log"
  exit 1
fi
exit 0
