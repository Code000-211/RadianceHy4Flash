#!/bin/sh
# Replay every recorded kernel fixture against the libraries this build made, and fail if the two
# disagree. No model and no card: a fixture holds the reference library's answers at a container's
# geometries, so the check runs in seconds anywhere the build runs (T-05).
#
# This is the gate the stale gpu-mini junction fixtures needed. `rmsnorm_ihc_junction` was failing
# even against the reference's own recording (rel 7.0e-03 against an exact comparison), which no
# amount of re-running the test could fix -- only re-recording could.
#
#   scripts/kbench-fixtures.sh                 replay data/*.rkb
#   KN=/path/to/rad-kbench scripts/kbench-fixtures.sh
#
# Exit status is the number of fixtures that failed (0 = clean), so a CI step needs no parsing.
#
# Reaching a row nobody recorded is a coverage gap, not a failure, and the summary line says so.
# `diverged` means the recording and a library describe an operand differently -- a real finding,
# but one that needs an owner's decision about which description is right, so it is reported and
# counted apart from failures rather than folded into the exit status.

set -u

here=$(cd "$(dirname "$0")/.." && pwd)
kn=${KN:-$here/build/bin/rad-kbench}
export RADIANCE_HOME=${RADIANCE_HOME:-$here/build/radiance_home}

if [ ! -x "$kn" ]; then
    echo "kbench-fixtures: no rad-kbench at $kn (build it, or set KN)" >&2
    exit 1
fi
if [ ! -d "$RADIANCE_HOME/kernels" ]; then
    echo "kbench-fixtures: no kernel plugins under $RADIANCE_HOME/kernels (set RADIANCE_HOME)" >&2
    exit 1
fi

failed_fixtures=0
diverged_total=0

for fx in "$here"/data/*.rkb; do
    [ -f "$fx" ] || continue
    name=$(basename "$fx")
    out=$("$kn" --fixture "$fx" 2>&1) || true
    summary=$(printf '%s\n' "$out" | grep -E '^[0-9]+ checked,' | tail -1)
    if [ -z "$summary" ]; then
        echo "FAIL  $name: no summary line -- the replay did not finish"
        printf '%s\n' "$out" | tail -5 | sed 's/^/      /'
        failed_fixtures=$((failed_fixtures + 1))
        continue
    fi
    nfail=$(printf '%s\n' "$summary" | sed -n 's/.* \([0-9][0-9]*\) failed.*/\1/p')
    ndiv=$(printf '%s\n' "$summary" | sed -n 's/.* \([0-9][0-9]*\) diverged.*/\1/p')
    [ -n "$nfail" ] || nfail=0
    [ -n "$ndiv" ] || ndiv=0
    diverged_total=$((diverged_total + ndiv))

    if [ "$nfail" -gt 0 ]; then
        echo "FAIL  $name: $summary"
        printf '%s\n' "$out" | grep -E 'FAIL' | head -8 | sed 's/^/      /'
        failed_fixtures=$((failed_fixtures + 1))
    else
        echo "ok    $name: $summary"
    fi
    if [ "$ndiv" -gt 0 ]; then
        printf '%s\n' "$out" | sed -n '/described differently/,/^$/p' | head -6 | sed 's/^/      /'
    fi
done

if [ "$diverged_total" -gt 0 ]; then
    echo "note: $diverged_total case(s) diverged (an operand described differently by the recording and a library)." >&2
    echo "note: these need an owner's decision, so they do not fail the gate." >&2
fi

if [ "$failed_fixtures" -gt 0 ]; then
    echo "kbench-fixtures: $failed_fixtures fixture(s) disagree with the libraries; re-record with" >&2
    echo "  rad-kbench -m <model.rad> --max-cases 1 --record data/<name>.rkb" >&2
fi

exit "$failed_fixtures"