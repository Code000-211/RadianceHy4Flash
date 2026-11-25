#!/usr/bin/env python3
"""prof-steps.py <server.log> [op-substring]: one step's own op time, from the per-op dumps.

`RADIANCE_PROFILE_EVERY=1` makes the engine print the per-op table after every step, and the table is
cumulative for the run (`dump_profile`, `core/runtime/ctx.cpp`), so consecutive rank-0 tables
differenced give the step's own time: its row of the table total, and the ms and us/call of every op
whose name carries `op-substring` (default `moe_gemm`, the routed expert GEMMs a stager card is about).
Steps that carried prefill rows are printed with the rows they carried, read from the step log
(`RADIANCE_LOG_STEPS=1`) that has to be in the same server.log.

A scratch tool for the M-03b card: the A/B of a stager arm needs a *step's* routed-GEMM time, and the
table alone only gives the whole run's. Dumps are the rank-0 table only; TP2's rank 1 runs the same
code and the same table.
"""
import re, sys

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "moe_gemm"
step_re = re.compile(r"^I step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)")


def dumps(log):
    """(last step seen, {op: (calls, ms)}, table total ms) per rank-0 table, in file order"""
    last, cur, total = None, None, None
    for ln in open(log, errors="replace"):
        m = step_re.match(ln)
        if m:
            last = (int(m.group(1)), m.group(2), int(m.group(4)))
            continue
        if ln.startswith("per-op timing, rank 0"):
            cur, total = {}, None
            continue
        if cur is None:
            continue
        t = ln.split()
        if t and t[0] == "total" and len(t) > 1:
            yield last, cur, float(t[1])
            cur = None
        elif len(t) > 7 and t[0].isdigit() and t[3] in ("dev", "host"):
            cur[(t[6], t[-1])] = (int(t[0]), float(t[1]))   # op (with geometry) x weight name


prev, rows = None, []
for step, cur, total in dumps(path):
    if prev is not None:
        delta = {}
        for k, (calls, ms) in cur.items():
            pc, pm = prev["ops"].get(k, (0, 0.0))
            if calls != pc or abs(ms - pm) > 1e-9:
                delta[k] = (calls - pc, ms - pm)
        rows.append((step, delta, total - prev["total"]))
    prev = {"ops": cur, "total": total}

if not rows:
    sys.exit(f"{path}: no per-op table pair to difference; run with RADIANCE_PROFILE_EVERY=1")

for (n, phase, nrows), delta, tt in rows:
    if phase == "decode":
        continue
    hits = [(k, v) for k, v in delta.items() if want in k[0]]
    hms = sum(v[1] for _, v in hits)
    hcalls = sum(v[0] for _, v in hits)
    print(f"step {n} {phase} {nrows} rows: table {tt:.1f} ms; {want} {hms:.1f} ms over {hcalls} calls"
          + (f", {1000 * hms / hcalls:.2f} us/call" if hcalls else ""))
