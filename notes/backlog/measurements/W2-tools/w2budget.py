#!/usr/bin/env python3
"""w2budget.py <stamps.csv> <records> [--layers]

Per-layer time budget of one recorded pass kind (e.g. the DFly2 c8 verify pass: 1191 records with
W2 off, 1314 with the tail form, 1191 with the inline form) from a RADIANCE_AQL_STAMP file.
A routed layer is cut at its router launch; inside it the critical path is split into
  router | router end -> gate_up start | gate_up | gate_up end -> down start (the quantiser) | down |
  tail (resident down's end -> the tail down's end; the tail form only) | last down end -> gather end |
  the rest of the layer (attention, trunk linears, ... up to the next router's start).
Prints p50 / mean us a layer and the mean ms a pass of each segment, and the pass wall.
--layers adds the mean of gate_up, down and the tail by layer index.
"""
import csv, sys, statistics as st
from collections import defaultdict

path, records = sys.argv[1], int(sys.argv[2])
passes = defaultdict(list)
with open(path) as f:
    for r in csv.DictReader(f):
        if int(r["records"]) == records:
            passes[int(r["pass"])].append((int(r["seq"]), r["kernel"], int(r["start_ns"]), int(r["end_ns"])))
def kind(k):
    if "router_sigmoid_scatter" in k: return "router"
    if "moe_iqr_kernel" in k: return "gu"
    if "moe_down_dec" in k: return "dn"
    if "gated_quant" in k: return "gq"
    if "gather_kernel" in k: return "gather"
    return None
seg = defaultdict(list)
walls, by_layer = [], defaultdict(lambda: defaultdict(list))
for p, recs in sorted(passes.items()):
    recs.sort()
    idx = [i for i, r in enumerate(recs) if kind(r[1]) == "router"]
    if len(idx) < 2: continue
    walls.append((max(r[3] for r in recs) - min(r[2] for r in recs)) / 1e3)
    seg["before the first router"].append((recs[idx[0]][2] - min(r[2] for r in recs)) / 1e3 / len(idx))
    for li, i in enumerate(idx):
        j = idx[li + 1] if li + 1 < len(idx) else len(recs)
        s = recs[i:j]
        gu = [r for r in s if kind(r[1]) == "gu"]; dn = [r for r in s if kind(r[1]) == "dn"]
        ga = [r for r in s if kind(r[1]) == "gather"]
        if not gu or not dn or not ga: continue
        nxt = recs[j][2] if j < len(recs) else max(r[3] for r in s)
        L = {"router": s[0][3] - s[0][2], "router end -> gate_up start": gu[0][2] - s[0][3],
             "gate_up (first launch)": gu[0][3] - gu[0][2],
             "gate_up end -> down start": dn[0][2] - gu[0][3], "down (first launch)": dn[0][3] - dn[0][2],
             "tail (first down end -> last down end)": dn[-1][3] - dn[0][3],
             "last down end -> gather end": ga[0][3] - dn[-1][3],
             "rest of the layer": nxt - ga[0][3],
             "= routed block (router start -> gather end)": ga[0][3] - s[0][2],
             "= layer": nxt - s[0][2]}
        for k, v in L.items(): seg[k].append(v / 1e3)
        by_layer[li]["gu"].append(L["gate_up (first launch)"] / 1e3)
        by_layer[li]["dn"].append(L["down (first launch)"] / 1e3)
        by_layer[li]["tail"].append(L["tail (first down end -> last down end)"] / 1e3)
n = len(walls)
print(f"{path}: {n} passes of {records} records, wall p50 {st.median(walls)/1e3:.2f} mean {st.mean(walls)/1e3:.2f} ms")
print(f"{'segment':46s} {'p50 us':>8s} {'mean us':>8s} {'ms a pass':>10s}")
for k, v in seg.items():
    print(f"{k:46s} {st.median(v):8.1f} {st.mean(v):8.1f} {sum(v)/n/1e3:10.2f}")
if "--layers" in sys.argv:
    for li in sorted(by_layer):
        b = by_layer[li]
        print(f"layer {li:2d} gu {st.mean(b['gu']):6.1f} dn {st.mean(b['dn']):6.1f} tail {st.mean(b['tail']):6.1f}")
