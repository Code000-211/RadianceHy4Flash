#!/usr/bin/env python3
"""w2budget.py <name>=<records>:<a.csv,b.csv,..> ...   [--raw]

Per-layer time budget of one recorded pass kind (the DFly2 c8 verify pass: 1191 records with W2
off and with the inline form, 1314 with the tail form) from RADIANCE_AQL_STAMP files, an arm a
column. A routed layer is cut at its router launch and its critical path is split into
  router | router end -> gate_up start | gate_up (first launch) | gate_up end -> down start (the
  quantiser) | down (first launch) | tail (first down's end -> last down's end; the tail form) |
  last down end -> gather end | the rest of the layer (attention, trunk, ... to the next router).
Every pass is divided by w2fit.py's yardstick (the exclusive time of its trunk linears, which
follows the pass's rows and the cards' clock) and scaled by 5.05 ms, the yardstick's mean over
the runs of 2026-11-26; a run is the mean of its passes, an arm the mean of its runs (sd over
runs in brackets where there are several). ms a pass. --raw prints the unnormalised means too.
"""
import csv, sys, statistics as st
from collections import defaultdict

Y0 = 5.05
SEGS = ["router", "router end -> gate_up start", "gate_up (first launch)", "gate_up end -> down start",
        "down (first launch)", "tail (first down end -> last down end)", "last down end -> gather end",
        "rest of the layer", "= routed block (router start -> gather end)", "= pass wall"]
def kind(k):
    if "router_sigmoid_scatter" in k: return "router"
    if "moe_iqr_kernel" in k: return "gu"
    if "moe_down_dec" in k: return "dn"
    if "gather_kernel" in k: return "gather"
    return None
def run(path, records):
    passes = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["records"]) == records:
                passes[int(r["pass"])].append((int(r["seq"]), r["kernel"], int(r["start_ns"]), int(r["end_ns"])))
    norm, raw = defaultdict(list), defaultdict(list)
    for p, recs in passes.items():
        recs.sort()
        idx = [i for i, r in enumerate(recs) if kind(r[1]) == "router"]
        if len(idx) != 41: continue
        y, hi = 0, 0
        for s, k, a, b in recs:
            if ("iqr_m16_kernel" in k or "gemm_i8_kpar_kernel" in k) and b > max(a, hi): y += b - max(a, hi)
            hi = max(hi, b)
        acc = defaultdict(float)
        for li, i in enumerate(idx):
            j = idx[li + 1] if li + 1 < len(idx) else len(recs)
            s = recs[i:j]
            gu = [r for r in s if kind(r[1]) == "gu"]; dn = [r for r in s if kind(r[1]) == "dn"]
            ga = [r for r in s if kind(r[1]) == "gather"]
            if not gu or not dn or not ga: continue
            nxt = recs[j][2] if j < len(recs) else max(r[3] for r in s)
            acc[SEGS[0]] += s[0][3] - s[0][2]; acc[SEGS[1]] += gu[0][2] - s[0][3]
            acc[SEGS[2]] += gu[0][3] - gu[0][2]; acc[SEGS[3]] += dn[0][2] - gu[0][3]
            acc[SEGS[4]] += dn[0][3] - dn[0][2]; acc[SEGS[5]] += dn[-1][3] - dn[0][3]
            acc[SEGS[6]] += ga[0][3] - dn[-1][3]; acc[SEGS[7]] += nxt - ga[0][3]
            acc[SEGS[8]] += ga[0][3] - s[0][2]
        acc[SEGS[9]] = max(r[3] for r in recs) - min(r[2] for r in recs)
        for k in SEGS:
            norm[k].append(acc[k] / y * Y0); raw[k].append(acc[k] / 1e6)
        raw["yardstick"].append(y / 1e6)
    return ({k: st.mean(v) for k, v in norm.items()}, {k: st.mean(v) for k, v in raw.items()}, len(raw["yardstick"]))

arms = []
for a in sys.argv[1:]:
    if a.startswith("--"): continue
    name, rest = a.split("=", 1); rec, files = rest.split(":", 1)
    runs = [run(f, int(rec)) for f in files.split(",")]
    runs = [r for r in runs if r[2]]
    arms.append((name, runs))
    print(f"{name}: {len(runs)} runs, passes {[r[2] for r in runs]}, yardstick ms {[round(r[1]['yardstick'], 3) for r in runs]}")
def cell(vals):
    return f"{st.mean(vals):7.2f}" + (f" ({st.stdev(vals):.2f})" if len(vals) > 1 else "       ")
for title, which in (("normalised, ms a pass", 0),) + ((("raw means, ms a pass", 1),) if "--raw" in sys.argv else ()):
    print(f"\n{title:46s} " + " ".join(f"{n:>14s}" for n, _ in arms) + "   " +
          " ".join(f"{n + '-' + arms[0][0]:>10s}" for n, _ in arms[1:]))
    for k in SEGS:
        m = [[r[which][k] for r in runs] for _, runs in arms]
        print(f"{k:46s} " + " ".join(f"{cell(v):>14s}" for v in m) + "   " +
              " ".join(f"{st.mean(v) - st.mean(m[0]):+10.2f}" for v in m[1:]))
