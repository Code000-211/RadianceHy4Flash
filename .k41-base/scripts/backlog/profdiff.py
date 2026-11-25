#!/usr/bin/env python3
"""profdiff.py server.log [rank]: per-op tables of a rank are cumulative; print, for each dump whose delta total is
>= MINMS (default 150 ms: a prefill chunk), the delta summed by op (ms, share, calls, us/call); and the last small-step delta."""
import sys, re, collections
L = open(sys.argv[1], errors="replace").read().split("\n"); rank = sys.argv[2] if len(sys.argv) > 2 else "0"
MINMS = float(sys.argv[3]) if len(sys.argv) > 3 else 150
starts = [i for i, l in enumerate(L) if l.startswith(f"per-op timing, rank {rank}")]
def parse(st):
    T = {}
    for l in L[st + 1:]:
        m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(dev|host)\s+(\S+)\s+([\d.\-]+)\s+(\S+)(.*)", l)
        if m:
            c, ms, op, rest, w = int(m[1]), float(m[2]), m[7], m[8], m[6]
            idx = re.search(r"#(\d+) (\S+)", rest); key = (op, idx[1] if idx else rest)
            wn = idx[2] if idx else "-"
            if op == "moe_gemm_q": op += " gate_up" if "gate_up" in wn else " down" if "down" in wn else ""
            if op in ("gemm_nt_q", "gemm_nt", "gemm_nt_bias"): op += " " + re.sub(r"blk\.\d+\.", "", wn).replace(".weight", "")[:22]
            T[key] = (c, ms, op, (float(w) if w != "-" else 0) * c / 1e3)
        elif l.strip().startswith("total") or l.startswith("per-op timing"): break
    return T
prev = {}; n = 0
for st in starts:
    cur = parse(st); D = collections.defaultdict(lambda: [0, 0.0, 0.0]); tot = 0
    for k, (c, ms, op, w) in cur.items():
        pc, pms, _, pw = prev.get(k, (0, 0.0, op, 0.0))
        if c > pc: D[op][0] += c - pc; D[op][1] += ms - pms; D[op][2] += w - pw; tot += ms - pms
    prev = cur; n += 1
    if tot >= MINMS or (MINMS < 0 and tot > 1):
        print(f"\n== dump {n}: delta total {tot:.1f} ms, wait {sum(v[2] for v in D.values()):.1f} ms")
        for op, (c, ms, w) in sorted(D.items(), key=lambda kv: -kv[1][1])[:int(sys.argv[4]) if len(sys.argv) > 4 else 45]:
            print(f"  {op[:44]:44s} calls {c:5d} {ms:8.2f} ms {100*ms/tot:5.1f} % {1e3*ms/c:9.1f} us/call")
