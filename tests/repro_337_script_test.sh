#!/bin/sh
# repro_337_script_test.sh -- scripts/repro-337.sh reads a run directory after a fault, not before.
#
# The script exists for the case where the engine faults mid-request: the server exits, the replay
# it drives exits non-zero, and the run directory is what is left. Three things then have to hold,
# and this test is the one that says they do, without a card (the script's `--summarise` mode):
#
#  - the fault report in the log is printed and counted, and the exit status says "a reproduction"
#    (1) rather than the replay's own failure;
#  - a log with no fault reads as 0 and says so;
#  - a run directory a dead server left half written -- an empty run.jsonl, a row cut short, no
#    run.jsonl at all -- is summarised, not raised on.
#
#   tests/repro_337_script_test.sh
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
script="$root/scripts/repro-337.sh"
[ -x "$script" ] || [ -f "$script" ] || { echo "repro_337_script_test: no $script" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

row1='{"conversation": "p0", "kind": "parent", "prompt_tokens": 95935, "cached_tokens": 90100, "ttft_ms": 900, "prev_prompt_tokens": 0, "tool_calls": 1, "t": 1}'
row2='{"conversation": "p0", "kind": "parent", "prompt_tokens": 99153, "cached_tokens": 95000, "ttft_ms": 700, "prev_prompt_tokens": 95935, "tool_calls": 1, "t": 2}'
row3='{"conversation": "p0", "kind": "parent", "prompt_tokens": 112174, "cached_tokens": 108000, "ttft_ms": 1800, "prev_prompt_tokens": 99153, "tool_calls": 1, "t": 3}'

# 1. The reproduction: a fault report in the log, the replay's rows beside it.
mkdir "$tmp/fault"
printf '%s\n%s\n%s\n' "$row1" "$row2" "$row3" > "$tmp/fault/run.jsonl"
cat > "$tmp/fault/server.log" <<'LOG'
radiance: device 0: the AQL queue faulted: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
radiance: device 0:   packet 5372985 may have run kernel gemm_i8_kpar_kernel<4,256,1,r4dchain>
radiance:   argument +0 = 0x7cc1d6c36000: in pageable of owner 1 card 0
radiance: fatal: a queue on device 0 stopped for good; exiting
LOG
set +e
out=$("$script" --summarise "$tmp/fault" 2>&1)
rc=$?
set -e
[ "$rc" = 1 ] || { echo "repro_337_script_test: a fault log gave rc=$rc, wanted 1" >&2
                   echo "$out" >&2; exit 1; }
for want in "3 requests" "largest prompt 112174 tokens" "3 with a prefix-cache hit" \
            "1 fault line(s)" "the AQL queue faulted" "gemm_i8_kpar_kernel"; do
    printf '%s' "$out" | grep -qF "$want" || {
        echo "repro_337_script_test: the summary does not name '$want':" >&2
        echo "$out" >&2; exit 1; }
done

# 2. A run with no fault: 0, and the count says so.
mkdir "$tmp/clean"
printf '%s\n%s\n' "$row1" "$row2" > "$tmp/clean/run.jsonl"
printf 'radiance: server up\nradiance: 2 requests served\n' > "$tmp/clean/server.log"
set +e
out=$("$script" --summarise "$tmp/clean" 2>&1)
rc=$?
set -e
[ "$rc" = 0 ] || { echo "repro_337_script_test: a clean log gave rc=$rc, wanted 0" >&2
                   echo "$out" >&2; exit 1; }
printf '%s' "$out" | grep -qF "0 fault line(s)" || {
    echo "repro_337_script_test: a clean run does not say 0:" >&2; echo "$out" >&2; exit 1; }

# 3. Half written, as a dead server leaves it: an empty run.jsonl, then a row cut short, then no
#    run.jsonl at all. Each is summarised -- no traceback -- and the fault is still counted.
mkdir "$tmp/empty"
: > "$tmp/empty/run.jsonl"
cp "$tmp/fault/server.log" "$tmp/empty/server.log"
set +e
out=$("$script" --summarise "$tmp/empty" 2>&1)
rc=$?
set -e
[ "$rc" = 1 ] || { echo "repro_337_script_test: an empty run.jsonl gave rc=$rc, wanted 1" >&2
                   echo "$out" >&2; exit 1; }
printf '%s' "$out" | grep -qF "no requests recorded" || {
    echo "repro_337_script_test: an empty run.jsonl is not reported" >&2; echo "$out" >&2; exit 1; }
printf '%s' "$out" | grep -qi "traceback" && {
    echo "repro_337_script_test: an empty run.jsonl raised" >&2; echo "$out" >&2; exit 1; }

mkdir "$tmp/cut"
printf '%s\n%s' "$row1" '{"conversation": "p0", "kind": "par' > "$tmp/cut/run.jsonl"
cp "$tmp/fault/server.log" "$tmp/cut/server.log"
set +e
out=$("$script" --summarise "$tmp/cut" 2>&1)
rc=$?
set -e
[ "$rc" = 1 ] || { echo "repro_337_script_test: a cut row gave rc=$rc, wanted 1" >&2
                   echo "$out" >&2; exit 1; }
printf '%s' "$out" | grep -qF "1 requests" || {
    echo "repro_337_script_test: the row before the cut is not counted" >&2
    echo "$out" >&2; exit 1; }

mkdir "$tmp/none"
cp "$tmp/fault/server.log" "$tmp/none/server.log"
set +e
out=$("$script" --summarise "$tmp/none" 2>&1)
rc=$?
set -e
[ "$rc" = 1 ] || { echo "repro_337_script_test: no run.jsonl gave rc=$rc, wanted 1" >&2
                   echo "$out" >&2; exit 1; }
printf '%s' "$out" | grep -qF "no requests recorded" || {
    echo "repro_337_script_test: a missing run.jsonl is not reported" >&2
    echo "$out" >&2; exit 1; }

# 4. No directory at all is the one thing it refuses, rather than reporting an empty run.
set +e
out=$("$script" --summarise 2>&1)
rc=$?
set -e
[ "$rc" = 2 ] || { echo "repro_337_script_test: --summarise without a directory gave rc=$rc" >&2
                   echo "$out" >&2; exit 1; }

echo "repro_337_script_test: a fault is counted and printed, a clean run is not, half written runs"
echo "  are summarised (5 cases)"
