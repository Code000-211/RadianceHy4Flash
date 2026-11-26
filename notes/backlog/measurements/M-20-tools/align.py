#!/usr/bin/env python3
"""align.py TRACE STAMPS PROBE PHASE -- pairs a phase's stamped full-model passes with its trace
dispatches (same order; the offset with the best correlation of per-pass miss count against
per-pass gate_up+down time is taken), then fits per layer
    t_gu = a_gu + b_gu * misses,  t_dn = a_dn + b_dn * misses
(least squares over every (pass, layer)): b is the measured cost of one whole-unit miss in each
GEMM today, a the GEMM's time with no miss in the layer."""
import sys, csv, json, collections, statistics
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import rtrace

def fit(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    sxx = sum((a - mx) ** 2 for a in x); sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    b = sxy / sxx if sxx else 0.0
    return my - b * mx, b

def corr(x, y):
    if len(x) < 3: return 0.0
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** .5; sy = sum((b - my) ** 2 for b in y) ** .5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else 0.0

tr, st, pj, ph = sys.argv[1:5]
nl, ne, by, mv, T, R, D = rtrace.read(tr)
probe = json.load(open(pj))
rng = {n: (d0, d1) for n, d0, d1, *_ in rtrace.phases_of(D, probe)}
d0, d1 = rng[ph]
miss = collections.defaultdict(dict); nexp = collections.defaultdict(dict)
for disp, hs, layer, reps, missed, pairs in R:
    if d0 < disp <= d1:
        miss[disp][layer] = missed; nexp[disp][layer] = len(pairs)
disps = sorted(miss)
w = [p for p in probe['phases'] if p.get('name') == ph][0]['mono_ns']
passes = collections.defaultdict(list)
with open(st) as fh:
    for r in csv.DictReader(fh):
        if not r['end_ns']: continue
        passes[int(r['pass'])].append((int(r['seq']), r['kernel'], int(r['start_ns']), int(r['end_ns']), int(r['records'])))
P = []
for pid in sorted(passes):
    ks = sorted(passes[pid])
    if ks[0][4] < 1000 or not (w[0] <= ks[0][2] <= w[1]): continue
    last = 0; gu = []; dn = []; wn = []; rend = None
    for seq, kn, s, e, rec in ks:
        x = (e - max(s, last)) / 1e3; last = max(last, e)
        if 'router_sigmoid' in kn: rend = e
        elif 'moe_iqr_kernel' in kn: gu.append(x)
        elif 'moe_down_dec' in kn:
            dn.append(x); wn.append((s - rend) / 1e3 if rend else float('nan')); rend = None
    P.append((ks[0][2], gu, dn, wn))
P.sort()
layers = sorted({l for d in disps for l in miss[d]})
best = None
for off in range(-6, 7):
    xs, ys = [], []
    for i, (_, gu, dn, wn) in enumerate(P):
        j = i + off
        if 0 <= j < len(disps):
            xs.append(sum(miss[disps[j]].values())); ys.append(sum(gu) + sum(dn))
    c = corr(xs, ys)
    if best is None or c > best[0]: best = (c, off, len(xs))
c, off, n = best
print(f'{ph}: {len(P)} stamped passes, {len(disps)} dispatches; best offset {off} (pairs {n}), '
      f'corr(per-pass misses, gate_up+down time) = {c:.3f}')
xg, yg, yd, xe = [], [], [], []
yw = []
for i, (_, gu, dn, wn) in enumerate(P):
    j = i + off
    if not (0 <= j < len(disps)) or len(gu) != len(layers): continue
    for k, l in enumerate(layers):
        xg.append(miss[disps[j]].get(l, 0)); yg.append(gu[k]); yd.append(dn[k]); xe.append(nexp[disps[j]].get(l, 0)); yw.append(wn[k])
ag, bg = fit(xg, yg); ad, bd = fit(xd := xg, yd)
print(f'  per layer: gate_up = {ag:.1f} + {bg:.1f} us x misses (corr {corr(xg, yg):.3f});  '
      f'down = {ad:.1f} + {bd:.1f} us x misses (corr {corr(xg, yd):.3f});  n={len(xg)} layer passes, '
      f'mean misses {statistics.mean(xg):.2f}, mean distinct experts {statistics.mean(xe):.1f}')
aw, bw = fit(xg, yw)
print(f'  window router-end -> down-start = {aw:.1f} + {bw:.1f} us x misses (corr {corr(xg, yw):.3f})')
byk = collections.defaultdict(lambda: ([], [], []))
for m, g, d, ww in zip(xg, yg, yd, yw):
    byk[min(m, 12)][0].append(g); byk[min(m, 12)][1].append(d); byk[min(m, 12)][2].append(ww)
print('  misses: n  gate_up p50  down p50  window p50')
for k in sorted(byk):
    g, d, ww = byk[k]
    print(f'   {k:2d}{"+" if k == 12 else " "} {len(g):5d} {statistics.median(g):8.1f} {statistics.median(d):8.1f} {statistics.median(ww):8.1f}')
