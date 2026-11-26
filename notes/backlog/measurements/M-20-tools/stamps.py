#!/usr/bin/env python3
"""stamps.py STAMPS.csv PROBE.json -- per phase, the routed MoE block's kernels a layer (M-20):
router -> moe_iqr (gate_up) -> ... -> moe_down_dec (down), from RADIANCE_AQL_STAMP's rank-0 stamps.

For every full-model pass (records >= 1000) in a phase's window and every routed layer:
  t_gu   moe_iqr_kernel exclusive us
  t_dn   moe_down_dec_kernel exclusive us
  win    end of the router kernel -> start of moe_down_dec (the in-layer window a down copy
         posted right after routing has before the down GEMM wants it)
Prints p10 / p50 / p90 and the mean, per phase, and the pass wall."""
import csv, json, sys, collections, statistics

def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))] if v else float('nan')

rows = collections.defaultdict(list)
with open(sys.argv[1]) as fh:
    for r in csv.DictReader(fh):
        if not r['end_ns']: continue
        rows[int(r['pass'])].append((int(r['seq']), r['kernel'], int(r['start_ns']), int(r['end_ns']), int(r['records'])))
R = json.load(open(sys.argv[2]))
wins = [(p['name'], p['mono_ns'][0], p['mono_ns'][1]) for p in R['phases'] if p.get('mono_ns') and p['mono_ns'][1]]
out = {}
for name, a, b in wins:
    S = collections.defaultdict(list)
    npass = 0
    for pid, ks in rows.items():
        ks.sort()
        if ks[0][4] < 1000: continue
        if not (a <= ks[0][2] <= b): continue
        npass += 1
        S['wall'].append((max(k[3] for k in ks) - min(k[2] for k in ks)) / 1e3)
        last = 0; ex = []
        for seq, kn, s, e, rec in ks:
            st = max(s, last); ex.append((kn, s, e, (e - st) / 1e3)); last = max(last, e)
        rt_end = None; gu = 0
        tg = 0; td = 0
        for kn, s, e, x in ex:
            if 'router_sigmoid' in kn: rt_end = e
            elif 'moe_iqr_kernel' in kn: S['t_gu'].append(x); tg += x
            elif 'moe_down_dec' in kn:
                S['t_dn'].append(x); td += x
                if rt_end is not None: S['win'].append((s - rt_end) / 1e3)
                rt_end = None
        S['gu_pass'].append(tg); S['dn_pass'].append(td)
    if not npass: continue
    o = {'passes': npass}
    for k, v in S.items():
        o[k] = {'p10': round(q(v, .1), 1), 'p50': round(q(v, .5), 1), 'p90': round(q(v, .9), 1),
                'mean': round(statistics.mean(v), 1), 'n': len(v)}
    out[name] = o
for k, v in out.items():
    print(k, json.dumps(v))
