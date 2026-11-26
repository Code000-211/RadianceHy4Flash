#!/usr/bin/env python3
"""x06_fp5_agg.py <run.json> ...: X-06 (FP-5) table from x06_fp5.py outputs named <...>-on-r<n>.json / -off-r<n>.json.

Per phase and run: median decode tok/s over the phase's requests, tokens a decode step (decode tokens / decode steps),
draft acceptance, suffix steps / global wins; then the median over runs per arm. Parity: the sha256 of every T 0
request, ON against OFF, in order.
"""
import json, re, statistics as st, sys

runs = {}
for p in sys.argv[1:]:
    m = re.search(r"-(on|off)-r(\d+)\.json$", p)
    runs[(m.group(1), int(m.group(2)))] = json.load(open(p))
# Runs need not carry the same phases (session 2 ran only some of them): each phase is aggregated over the runs
# that have it, in first-seen order.
phases = []
for d in runs.values():
    phases += [ph for ph in d if ph not in phases]


def having(arm, ph):
    return sorted(k for k in runs if k[0] == arm and ph in runs[k])


def row(d, ph):
    x = d[ph]; m = x["metrics"]; rq = [r for g in x["reqs"] for r in g]
    dec = st.median(r["dec"] for r in rq)
    tps = m["decode_tokens_total"] / max(1, m["engine_steps_decode_total"])
    acc = m["draft_accepted_total"] / max(1, m["draft_tokens_total"])
    return dec, tps, acc, m


print("| phase | arm | decode tok/s (median of requests) r.. | median | tokens a decode step | draft accept | "
      "suffix steps / global wins / busy / adopted |")
print("|---|---|---|---|---|---|---|")
for ph in phases:
    for arm in ("on", "off"):
        ks = having(arm, ph)
        if not ks:
            continue
        rs = [row(runs[k], ph) for k in ks]
        decs = [r[0] for r in rs]
        m = rs[0][3]
        sfx = "; ".join(f"{int(r[3]['suffix_steps_total'])}/{int(r[3]['suffix_global_total'])}/"
                        f"{int(r[3]['suffix_global_busy_total'])}/{int(r[3]['suffix_adopted_total'])}" for r in rs)
        print(f"| {ph} | {arm} | {' / '.join(f'{d:.1f}' for d in decs)} | {st.median(decs):.1f} | "
              f"{st.median(r[1] for r in rs):.3f} | {st.median(r[2] for r in rs):.3f} | {sfx} |")

# greedy retries by retry index, and the 64K turn 2 TTFT
for arm in ("on", "off"):
    if having(arm, "rg"):
        by = [[r for k in having(arm, "rg") for g in runs[k]["rg"]["reqs"] for j, r in enumerate(g) if j == i] for i in range(3)]
        print(f"rg {arm}: decode tok/s by retry (median over prompts and runs): "
              + ", ".join(f"ask {i + 1} {st.median(x['dec'] for x in b):.1f}" for i, b in enumerate(by)))
    if having(arm, "t64"):
        t = [runs[k]["t64"]["reqs"][0] for k in having(arm, "t64")]
        print(f"t64 {arm}: turn 1 prompt {t[0][0]['prompt']} TTFT " + " / ".join(f"{x[0]['ttft']:.3f}" for x in t)
              + f"; turn 2 prompt {t[0][1]['prompt']} TTFT " + " / ".join(f"{x[1]['ttft']:.3f}" for x in t))

# parity: every T 0 request in order, per phase, over the runs that have the phase
for ph in phases:
    ks = sorted(k for k in runs if ph in runs[k])
    shas = {k: [r["sha"] for g in runs[k][ph]["reqs"] for r in g if r["temp"] == 0.0] for k in ks}
    ref = shas[ks[0]]
    if not ref:
        continue
    same = all(v == ref for v in shas.values())
    print(f"greedy parity {ph}: {len(ref)} T 0 texts per run, {len(shas)} runs, all equal: {same}")
    if not same:
        for k, v in sorted(shas.items()):
            print(k, " ".join(v))
# sampled: equal texts between arms, request by request, per phase
for ph in phases:
    a = [[r["sha"] for g in runs[k][ph]["reqs"] for r in g] for k in sorted(runs) if ph in runs[k]]
    n = len(a[0])
    eq = sum(1 for i in range(n) if len({x[i] for x in a}) == 1)
    print(f"{ph}: {eq} of {n} requests give the same text in all {len(a)} runs")
