#!/usr/bin/env python3
"""w2fit.py <name>=<records>:<a1.csv,a2.csv,..> ... : arms' recorded passes of one kind, normalised.

A verify pass's wall time moves from pass to pass and from run to run with things W2 has no part
in (the rows the drafter's depth rule gave the pass, the cards' clock under the power cap): one
run's 40-odd passes have a wall sd of ~6 %, more than the effect looked for. Each pass therefore
gets a yardstick measured inside it: the exclusive time of its trunk linears (iqr_m16_kernel and
gemm_i8_kpar_kernel, 376 launches a pass; none reads an expert, and W2 changes none of them).
Exclusive: a launch's end less the later of its start and the end of every launch before it in the
pass, because a chained launch's start stamp is when it was queued, not when it ran. The yardstick
follows the same rows and the same clock (correlation with the pass wall 0.65-0.97 within a run).

Per run: passes, mean wall, mean routed span (router end to the last routed down's end, over the 41
layers), mean yardstick, and wall / yardstick and span / yardstick over the run's passes. Per arm:
the mean and sd of the runs' ratios. Then every arm against the first.
"""
import csv, sys, statistics as st
from collections import defaultdict

def passes(f, records):
    P = defaultdict(list)
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if int(r["records"]) == records:
                P[int(r["pass"])].append((int(r["seq"]), r["kernel"], int(r["start_ns"]), int(r["end_ns"])))
    out = []
    for recs in P.values():
        recs.sort()
        ro = [r for r in recs if "router_sigmoid_scatter" in r[1]]
        dn = [r for r in recs if "moe_down_dec" in r[1]]
        if len(ro) != 41 or not dn: continue
        span = 0
        for i, r in enumerate(ro):
            nxt = ro[i + 1][0] if i + 1 < len(ro) else 1 << 60
            d = [x for x in dn if r[0] < x[0] < nxt]
            if d: span += d[-1][3] - r[3]
        x, hi = 0, 0
        for s, k, a, b in recs:
            if ("iqr_m16_kernel" in k or "gemm_i8_kpar_kernel" in k) and b > max(a, hi): x += b - max(a, hi)
            hi = max(hi, b)
        out.append((x / 1e6, span / 1e6, (max(r[3] for r in recs) - min(r[2] for r in recs)) / 1e6))
    return out

arms = []
for a in sys.argv[1:]:
    name, rest = a.split("=", 1)
    rec, files = rest.split(":", 1)
    runs = []
    for f in files.split(","):
        ps = passes(f, int(rec))
        if not ps: continue
        x = st.mean(p[0] for p in ps); sp = st.mean(p[1] for p in ps); w = st.mean(p[2] for p in ps)
        runs.append((len(ps), w, sp, x, w / x, sp / x))
        print(f"{name:6s} {f.split('/')[-1]:28s} {len(ps):3d} passes  wall {w:6.2f} ms  routed span {sp:6.2f} ms  "
              f"yardstick {x:5.3f} ms  wall/y {w / x:6.3f}  span/y {sp / x:6.3f}")
    arms.append((name, runs))
print()
base = arms[0]
def ms(v): return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
for name, runs in arms:
    w, sw = ms([r[4] for r in runs]); s, ss = ms([r[5] for r in runs])
    bw = st.mean(r[4] for r in base[1]); bs = st.mean(r[5] for r in base[1])
    y = st.mean(r[3] for r in base[1])
    print(f"{name:6s} {len(runs)} runs  wall/y {w:6.3f} (sd {sw:.3f})  span/y {s:6.3f} (sd {ss:.3f})  "
          f"plain wall {st.mean(r[1] for r in runs):6.2f} ms  span {st.mean(r[2] for r in runs):6.2f} ms"
          + ("" if name == base[0] else f"  vs {base[0]}: wall {100 * (w / bw - 1):+.2f} % = {(w - bw) * y:+.2f} ms a pass, "
                                         f"span {100 * (s / bs - 1):+.2f} % = {(s - bs) * y:+.2f} ms a pass"))
