#!/usr/bin/env python3
"""profagg.py server.log [rank]: the last per-op table of a rank summed by op name (ms, share, calls, wait)."""
import sys, re, collections
L = open(sys.argv[1], errors="replace").read().split("\n"); rank = sys.argv[2] if len(sys.argv) > 2 else "0"
st = [i for i, l in enumerate(L) if l.startswith(f"per-op timing, rank {rank}")][-1]
T = collections.defaultdict(lambda: [0, 0.0, 0.0]); tot = 0
for l in L[st:]:
    m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(dev|host)\s+(\S+)\s+([\d.\-]+)\s+(\S+)(.*)", l)
    if m:
        c, ms, w, op, rest = int(m[1]), float(m[2]), m[6], m[7], m[8]
        if op == "moe_gemm_q": op += " (routed gate_up)" if "gate_up" in rest else " (routed down)" if "down" in rest else ""
        if op in ("gemm_nt_q", "gemm_nt"): op += " " + (re.search(r"#\d+ blk\.\d+\.(\S+)", rest) or [0, "?"])[1][:22]
        T[op][0] += c; T[op][1] += ms; T[op][2] += (float(w) if w != "-" else 0) * c / 1e3
    elif l.strip().startswith("total"):
        tot = float(l.split()[1]); print(l.strip()); break
for op, (c, ms, w) in sorted(T.items(), key=lambda kv: -kv[1][1])[:28]:
    print(f"{op[:46]:46s} calls {c:6d}  {ms:9.1f} ms  {100*ms/tot:5.1f} %  {1e3*ms/c:9.1f} us/call  wait {w:7.1f} ms")
