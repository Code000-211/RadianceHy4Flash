#!/usr/bin/env python3
"""agg.py stamps.csv probe.json [--top N] [--match RE]: per probe phase and per kind of pass (tape length): passes, wall, busy union,
per-kernel exclusive time (end - max(start, previous end on its queue)), grouped into categories; and the card's idle
time between consecutive stamped passes is not visible (1 pass in N) -- compare wall with the probe's ms/step.
--match prints only the kernels whose name matches RE, with their total on a line of its own: the read of one named
term per step (T-03's `qsa_select` and `idx_score` at 60K, K-43's score plane) without the whole table.
A phase with no stamped window -- a composite one (longctx, idle, long8) or a gap probe (stallN) -- is named and
skipped; its waves are in the JSON as phases of their own, and those are what the stamped numbers come from."""
import csv, json, sys, collections, subprocess, re, statistics
import demangle
FILT = demangle.path()  # c++filt locally, the ROCm SDK's llvm-cxxfilt on iks
if not FILT: sys.exit("agg.py: no c++filt or llvm-cxxfilt, neither on PATH nor next to the tree's compiler")
rows = list(csv.DictReader(open(sys.argv[1])))
R = json.load(open(sys.argv[2]))
TOP = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
MATCH = re.compile(sys.argv[sys.argv.index("--match") + 1]) if "--match" in sys.argv else None
for r in rows:
    for k in ("pass", "seq", "start_ns", "end_ns", "records", "ordered", "chained", "queue"): r[k] = int(r[k] or 0)
rows = [r for r in rows if r["start_ns"] and r["end_ns"]]
names = sorted({r["kernel"] for r in rows})
dem = dict(zip(names, subprocess.run([FILT] + names, capture_output=True, text=True).stdout.split("\n")))
def short(n):
    d = re.sub(r"\(anonymous namespace\)::", "", dem[n]); d = re.sub(r"^void ", "", d)
    return re.sub(r"<.*", "", d.split("(")[0])[:46]
CATS = [("MoE routed gate_up (moe_iqr / moe_gemm)", r"moe_iqr|moe_gemm"),
        ("MoE routed down", r"moe_down"),
        ("MoE sort/gather/scatter/swiglu", r"moegather|moesort|swiglu|moe_"),
        ("router (f32 GEMM + sigmoid top-k)", r"router|gemm_nt_f32"),
        ("all-reduce + iHC junction", r"^ar_|ihc_|junction"),
        ("attention: sink rows + merge", r"mla_sink|merge_sink|attn_kv_sink|kv_split"),
        ("attention: project_kv absorb/expand, latent proj", r"project|expand|gate_mul|vit_gemm"),
        ("attention: cache row, rope, copies", r"row_front|rope|kv_store|copy"),
        ("indexer / selection", r"idx_|qsa_|dfly_sel"),
        ("engram", r"engram|embed_lookup_q"),
        ("trunk linears IQR (iqr_gemv / iqr_m16)", r"^iqr_|trellis"),
        ("trunk linears int8/w4 (gemm_i8_kpar)", r"gemm_i8|gemm_w4|gemm_q"),
        ("bf16 GEMMs (small)", r"gemm_bf16|gemm_"),
        ("activation quantisers", r"quant"),
        ("norms / silu", r"rmsnorm|silu|had_"),
        ("head (lm) + top-k + couple", r"logits|lm_wmma|mvm|sample|topk|couple|mtp_"),
        ]
def cat(k):
    for c, rx in CATS:
        if re.search(rx, k): return c
    return "other"
def union(iv):
    tot = 0; ca = cb = None
    for a, b in sorted(iv):
        if cb is None or a > cb:
            if cb is not None: tot += cb - ca
            ca, cb = a, b
        elif b > cb: cb = b
    return tot + (cb - ca if cb is not None else 0)
passes = collections.defaultdict(list)
for r in rows: passes[r["pass"]].append(r)
out = {}
for p in R["phases"]:
    w = p.get("mono_ns")
    if not w or not w[1] or "decode_steps" not in p:
        print(f"\n==== phase {p.get('name', '(composite)')}: no stamped window, skipped"
              f"{'; its waves are their own phases in the JSON' if any(v.get('mono_ns') for v in p.values() if isinstance(v, dict)) else ''}")
        continue
    a, b = w
    ps = [v for v in passes.values() if a <= min(x["start_ns"] for x in v) and max(x["end_ns"] for x in v) <= b]
    kinds = collections.Counter(v[0]["records"] for v in ps)
    print(f"\n==== phase {p['name']}: clients {p['clients']}, gen {p['gen']}, decode steps {p['decode_steps']:.0f}, tok/step {p['tok_per_step']:.2f}, "
          f"sum req tok/s {p['sum_req_dec_tps']:.1f}, ms/step(c1) {p['ms_step']}, hit {p['hit']:.3f}, miss/step {p['miss_units_per_step']:.0f}, ttft {p['ttft']}; pass kinds {dict(kinds)}")
    for rec, cnt in kinds.most_common(4):
        sel = [v for v in ps if v[0]["records"] == rec]
        mx = max(len(v) for v in sel); sel = [v for v in sel if len(v) >= 0.98 * mx]
        if not sel: continue
        n = len(sel)
        walls = [(max(x["end_ns"] for x in v) - min(x["start_ns"] for x in v)) / 1e3 for v in sel]
        busy = [union([(x["start_ns"], x["end_ns"]) for x in v]) / 1e3 for v in sel]
        agg = collections.defaultdict(lambda: [0, 0.0, 0.0]); qs = collections.Counter()
        for v in sel:
            last = {}
            for x in sorted(v, key=lambda x: (x["queue"], x["end_ns"])):
                q = x["queue"]; s = max(x["start_ns"], last.get(q, 0)); ex = max(0, x["end_ns"] - s); last[q] = max(last.get(q, 0), x["end_ns"])
                g = agg[short(x["kernel"])]; g[0] += 1; g[1] += ex / 1e3; g[2] += (x["end_ns"] - x["start_ns"]) / 1e3; qs[q] += ex / 1e3
        tot = sum(g[1] for g in agg.values()) / n
        print(f"\n -- pass kind {rec} records: {n} passes, {mx} dispatches; wall median {statistics.median(walls):.0f} us (min {min(walls):.0f}, max {max(walls):.0f}), busy median {statistics.median(busy):.0f}, "
              f"exclusive sum {tot:.0f}; per queue {{{', '.join(f'{q}: {t/n:.0f}' for q, t in qs.items())}}}")
        cs = collections.defaultdict(lambda: [0, 0.0])
        for k, g in agg.items(): c = cs[cat(k)]; c[0] += g[0] / n; c[1] += g[1] / n
        for c, g in sorted(cs.items(), key=lambda x: -x[1][1]):
            print(f"    [{c:46s}] {g[0]:7.1f} launches {g[1]:8.0f} us {100*g[1]/tot:5.1f}%")
        shown = [kv for kv in sorted(agg.items(), key=lambda x: -x[1][1]) if MATCH.search(kv[0])] if MATCH \
            else sorted(agg.items(), key=lambda x: -x[1][1])[:TOP]
        if MATCH:
            ln = sum(g[0] for _, g in shown) / n; us = sum(g[1] for _, g in shown) / n
            print(f"    terms /{MATCH.pattern}/: {len(shown)} kernels, {ln:.1f} launches {us:.1f} us a step "
                  f"({100 * us / tot:.1f}% of the exclusive sum)")
        for k, g in shown:
            print(f"      {k:46s} {g[0]/n:7.1f} x {g[1]/g[0]:8.1f} us = {g[1]/n:8.0f} us {100*g[1]/n/tot:5.1f}%   ({cat(k)})")
        out[f"{p['name']}/{rec}"] = {"n": n, "wall": statistics.median(walls), "busy": statistics.median(busy), "cats": {c: g for c, g in cs.items()}, "k": {k: [g[0] / n, g[1] / n] for k, g in agg.items()}}
json.dump(out, open(sys.argv[1].replace(".csv", ".agg.json"), "w"), indent=1)
