#!/usr/bin/env python3
"""attr.py stamps.csv run.json [--top N]: frontier attribution. Dispatches of a pass in record order; a dispatch is
charged end - max(start, latest end before it) (chained packets start early and wait for their chain, so raw
durations overlap). idle = gaps where nothing later-ending runs = wall - sum(charged). Per phase and pass kind."""
import csv, json, sys, collections, statistics as S, re
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 12
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["start_ns"] and r["end_ns"]]
R = json.load(open(sys.argv[2]))
P = collections.defaultdict(list)
for r in rows:
    r["s"], r["e"], r["q"] = int(r["start_ns"]), int(r["end_ns"]), int(r["seq"]); P[int(r["pass"])].append(r)
def short(k):
    k = k.replace("_ZN12_GLOBAL__N_1", "").replace("r4dchain_", "")
    m = re.match(r"(?:_Z)?\d*([A-Za-z_][A-Za-z0-9_]*?)(?:_kernel)?(?:I|E|$)", k)
    n = m.group(1) if m else k[:40]
    if "ar_ihc_junction" in k:
        t = re.search(r"ILi4E(Lb\d)E(Lb\d)E(Lb\d)E", k); n = "ar_ihc_junction_2rank" 
    if "iqr_m16" in k: n = "iqr_m16(shared/latent gemm)"
    if "moe_iqr" in k: n = "moe_iqr(routed gate_up)"
    return n
def coll(n): return n.startswith("ar_") or "all_reduce" in n or "all_gather" in n or "oneshot" in n
out = {}
for ph in R["phases"]:
    w = ph.get("mono_ns")
    if not w or not w[1]: continue   # a composite (longctx, idle, long8) or gap (stallN) phase: no window
    a, b = w
    ps = [v for v in P.values() if a <= min(x["s"] for x in v) and max(x["e"] for x in v) <= b]
    kinds = collections.defaultdict(list)
    for v in ps: kinds[int(v[0]["records"])].append(v)
    for rec, vs in sorted(kinds.items(), key=lambda kv: -len(kv[1]))[:5]:
        full = max(len(x) for x in vs); vs = [v for v in vs if len(v) == full]
        if len(vs) < 2: continue
        K = collections.defaultdict(float); N = collections.Counter(); walls = []; idle = []; ard = []; argap = []
        for v in vs:
            v.sort(key=lambda x: x["q"]); fr = v[0]["s"]; ch = 0
            for x in v:
                n = short(x["kernel"]); c = max(0, x["e"] - max(x["s"], fr)); g = max(0, x["s"] - fr)
                K[n] += c; N[n] += 1; ch += c
                if coll(n): ard.append(c / 1e3); argap.append(g / 1e3)
                fr = max(fr, x["e"])
            w = fr - v[0]["s"]; walls.append(w / 1e3); idle.append((w - ch) / 1e3)
        np_ = len(vs); W = S.mean(walls); ct = sum(t for k, t in K.items() if coll(k)) / np_ / 1e3
        nar = sum(n for k, n in N.items() if coll(k)) // np_
        print(f"\n== {ph['name']} tape {rec}: {np_} passes x {full} dispatches; wall {W/1e3:.2f} ms (p10 {sorted(walls)[len(walls)//10]/1e3:.2f} p90 {sorted(walls)[9*len(walls)//10]/1e3:.2f}); idle gaps {S.mean(idle)/1e3:.2f} ms = {100*S.mean(idle)/W:.1f} %; collectives x{nar}: {ct/1e3:.3f} ms = {100*ct/W:.1f} % of wall")
        if ard:
            ard.sort(); print(f"   collective charged us: p10 {ard[len(ard)//10]:.1f} p50 {S.median(ard):.1f} p90 {ard[9*len(ard)//10]:.1f} p99 {ard[99*len(ard)//100]:.1f} max {ard[-1]:.1f}; gap before it p50 {S.median(argap):.1f}")
        for k, t in sorted(K.items(), key=lambda kv: -kv[1])[:top]:
            m = t / np_ / 1e3
            print(f"   {k[:44]:44s} x{N[k]//np_:5d}  {m/1e3:8.3f} ms  {100*m/W:5.1f} %  each {m/max(1,N[k]//np_):8.1f} us{'  <- collective' if coll(k) else ''}")
        out[f"{ph['name']}/{rec}"] = {"passes": np_, "dispatches": full, "wall_us": W, "idle_us": S.mean(idle), "coll_us": ct, "coll_n": nar,
                                      "coll_p50_us": S.median(ard) if ard else 0, "coll_p90_us": ard[9*len(ard)//10] if ard else 0,
                                      "kern_us": {k: [t / np_ / 1e3, N[k] // np_] for k, t in K.items()}}
json.dump(out, open(sys.argv[1].replace(".csv", ".attr.json"), "w"), indent=1)
