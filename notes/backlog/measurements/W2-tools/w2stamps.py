#!/usr/bin/env python3
"""w2stamps.py <stamps.csv> <records> [--on]

The routed block of one recorded pass kind (`records` dispatches a pass, e.g. the DFly2 c8 verify
pass) from a RADIANCE_AQL_STAMP file: per routed layer, the router's end to the last routed down
GEMM's end ("routed span"), and the launches inside it. With the W2 form (--on) a layer has two
gate_up and two down launches; the first of each is the resident pass, the second the tail.
Prints medians over every layer of every pass of that kind, and the per-pass sums; --raw adds a
line a pass (wall, routed spans, the rest; ms).
"""
import csv, sys, statistics as st
from collections import defaultdict

path, records = sys.argv[1], int(sys.argv[2])
on = "--on" in sys.argv
passes = defaultdict(list)
with open(path) as f:
    for r in csv.DictReader(f):
        if int(r["records"]) != records:
            continue
        passes[int(r["pass"])].append((int(r["seq"]), r["kernel"], int(r["start_ns"]), int(r["end_ns"]),
                                        int(r["ordered"])))
def kind(k):
    if "router_sigmoid_scatter" in k: return "router"
    if "moe_iqr_kernel" in k: return "gu"
    if "moe_down_dec" in k: return "dn"
    if "gated_quant" in k: return "gq"
    return None
rows = defaultdict(list)       # name -> us per layer
per_pass = defaultdict(list)
ordered = []
n_pass = 0
for p, recs in sorted(passes.items()):
    recs.sort()
    n_pass += 1
    ordered.append(sum(r[4] for r in recs))
    acc = defaultdict(float)
    i = 0
    while i < len(recs):
        if kind(recs[i][1]) != "router":
            i += 1
            continue
        j = i + 1
        while j < len(recs) and kind(recs[j][1]) != "router":
            j += 1
        seg = recs[i:j]
        gu = [r for r in seg if kind(r[1]) == "gu"]
        dn = [r for r in seg if kind(r[1]) == "dn"]
        if not gu or not dn:
            i = j
            continue
        rend = seg[0][3]
        L = {}
        L["router"] = (seg[0][3] - seg[0][2]) / 1e3
        L["gu first"] = (gu[0][3] - gu[0][2]) / 1e3
        L["dn first"] = (dn[0][3] - dn[0][2]) / 1e3
        L["router end -> dn first start"] = (dn[0][2] - rend) / 1e3
        L["router end -> dn first end (window)"] = (dn[0][3] - rend) / 1e3
        if len(gu) > 1 and len(dn) > 1:
            L["gu tail"] = (gu[1][3] - gu[1][2]) / 1e3
            L["dn tail"] = (dn[1][3] - dn[1][2]) / 1e3
            L["tail (gu tail start -> dn tail end)"] = (dn[1][3] - gu[1][2]) / 1e3
        L["routed span (router end -> last dn end)"] = (dn[-1][3] - rend) / 1e3
        L["moe_iqr + moe_down_dec, all launches"] = sum(r[3] - r[2] for r in gu + dn) / 1e3
        for k, v in L.items():
            rows[k].append(v)
            acc[k] += v
        i = j
    # the whole pass, and what of it is not the routed block: the same in two arms when their
    # passes had the same heights (the rows a pass carries are not in the stamps)
    wall = (max(r[3] for r in recs) - min(r[2] for r in recs)) / 1e3
    acc["pass wall"] = wall
    acc["pass wall less routed spans and routers"] = wall - acc["routed span (router end -> last dn end)"] - acc["router"]
    for k, v in acc.items():
        per_pass[k].append(v)
print(f"{path}: {n_pass} passes of {records} records; ordered dispatches a pass: median "
      f"{st.median(ordered) if ordered else 0:.0f}")
print(f"{'':44s} {'us a layer p50':>15s} {'mean':>8s} {'ms a pass p50':>14s}")
for k in rows:
    print(f"{k:44s} {st.median(rows[k]):15.1f} {st.mean(rows[k]):8.1f} {st.median(per_pass[k]) / 1e3:14.2f}")
for k in ("pass wall", "pass wall less routed spans and routers"):
    v = per_pass[k]
    if v:
        print(f"{k:44s} {'':15s} {'':8s} {st.median(v) / 1e3:14.2f}   mean {st.mean(v) / 1e3:.2f} ms")
if "--raw" in sys.argv:      # one line a pass: wall, routed, rest (ms)
    for w, r, x in zip(per_pass["pass wall"], per_pass["routed span (router end -> last dn end)"],
                       per_pass["pass wall less routed spans and routers"]):
        print(f"pass {w / 1e3:.3f} {r / 1e3:.3f} {x / 1e3:.3f}")
