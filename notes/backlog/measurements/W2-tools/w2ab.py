#!/usr/bin/env python3
"""w2ab.py <runs-dir>... : one line a run from its probe JSON (and its stamps, where it has them).

Each argument is a run directory (runs/<lane>/<tag>) holding <tag>.json and maybe <tag>.stamps.csv.
Prints, per run: c1 tok/s and ms a step, c8 aggregate tok/s, its decode steps and tokens a step,
the greedy texts' sha256 heads, and for the widest recorded pass kind of the c8 phase (the DFly2
verify pass) the per-pass sums the stall is read from: moe_iqr + moe_down_dec over every launch,
the routed span (router end to the last routed down's end), the pass wall and the rest of the pass.
"""
import csv, json, os, sys, statistics as st
from collections import defaultdict

def kind(k):
    if "router_sigmoid_scatter" in k: return "router"
    if "moe_iqr_kernel" in k: return "gu"
    if "moe_down_dec" in k: return "dn"
    return None

def stamps(path, t0, t1):
    passes = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            s = int(r["start_ns"])
            if s < t0 or s > t1: continue
            passes[(int(r["records"]), int(r["pass"]))].append(
                (int(r["seq"]), r["kernel"], s, int(r["end_ns"]), int(r["ordered"])))
    by = defaultdict(list)
    for (rec, p), v in passes.items(): by[rec].append(v)
    wide = [k for k in by if len(by[k]) >= 10]
    if not wide: return None
    rec = max(wide)
    out = defaultdict(list)
    for recs in by[rec]:
        recs.sort()
        gu = [r for r in recs if kind(r[1]) == "gu"]
        dn = [r for r in recs if kind(r[1]) == "dn"]
        ro = [r for r in recs if kind(r[1]) == "router"]
        if not gu or not dn or len(ro) != 41: continue
        span = 0.0
        for i, r in enumerate(ro):
            nxt = ro[i + 1][0] if i + 1 < len(ro) else 1 << 60
            d = [x for x in dn if r[0] < x[0] < nxt]
            if d: span += d[-1][3] - r[3]
        wall = max(r[3] for r in recs) - min(r[2] for r in recs)
        out["moe"].append(sum(r[3] - r[2] for r in gu + dn) / 1e6)
        out["gu"].append(sum(r[3] - r[2] for r in gu) / 1e6)
        out["dn"].append(sum(r[3] - r[2] for r in dn) / 1e6)
        out["span"].append(span / 1e6)
        out["wall"].append(wall / 1e6)
        out["rest"].append((wall - span) / 1e6)
        out["ordered"].append(sum(r[4] for r in recs))
        out["disp"].append(len(recs))
    if not out["wall"]: return None
    res = {k: st.mean(v) for k, v in out.items()}
    res["n"] = len(out["wall"]); res["records"] = rec
    res["sem_span"] = st.stdev(out["span"]) / len(out["span"]) ** 0.5 if len(out["span"]) > 1 else 0
    res["med_wall"] = st.median(out["wall"]); res["med_span"] = st.median(out["span"]); res["med_moe"] = st.median(out["moe"])
    return res

for d in sys.argv[1:]:
    tag = os.path.basename(d.rstrip("/"))
    try: J = json.load(open(os.path.join(d, tag + ".json")))
    except Exception as e:
        print(f"{tag}: no probe JSON ({e})"); continue
    ph = {p["name"]: p for p in J.get("phases", [])}
    line = [f"{tag:22s}"]
    if "c1" in ph:
        c = ph["c1"]
        line.append(f"c1 {c.get('sum_req_dec_tps', 0):7.2f} tok/s {c.get('ms_step') or 0:6.3f} ms/step {c.get('tok_per_step', 0):5.2f} tok/step")
    if "c8" in ph:
        c = ph["c8"]
        line.append(f"c8 agg {c.get('agg_tps', 0):7.2f} sum {c.get('sum_req_dec_tps', 0):7.2f} tok/s steps {c.get('decode_steps', 0):5.0f} {c.get('tok_per_step', 0):6.2f} tok/step")
    if "greedy" in ph:
        line.append("greedy " + "/".join(t["sha256"][:8] for t in ph["greedy"]["texts"]))
    print("  ".join(line))
    sp = os.path.join(d, tag + ".stamps.csv")
    if "c8" in ph and os.path.exists(sp):
        t0, t1 = ph["c8"]["mono_ns"]
        r = stamps(sp, t0, t1)
        if r:
            print(f"{'':22s}  c8 verify pass kind {r['records']}: {r['n']} passes, {r['disp']:.0f} dispatches, {r['ordered']:.0f} ordered; "
                  f"ms a pass (mean): moe_iqr {r['gu']:.2f} + moe_down_dec {r['dn']:.2f} = {r['moe']:.2f}, routed span {r['span']:.2f} "
                  f"(sem {r['sem_span']:.2f}), wall {r['wall']:.2f}, rest {r['rest']:.2f}; medians: moe {r['med_moe']:.2f} span {r['med_span']:.2f} wall {r['med_wall']:.2f}")
