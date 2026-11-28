#!/usr/bin/env python3
"""runahead.py -- S-11: does a two-micro-batch runahead pay for offloaded MoE decode?

Reads a `RADIANCE_ROUTE_TRACE` file (core/place/rtrace.h) written by a probe run of
`scripts/backlog/run.sh`, split into the run's probe phases, and answers two questions about the
schedule the design note (notes/backlog/design/S-11.md) proposes: the pass is cut into two
micro-batches of r = R/2 rows each, and each micro-batch's exact miss set, read off its own router,
is copied on the prefetch DMA queue while the *other* micro-batch runs its layer.

  Q1  Does a micro-batch's miss copy fit the window the other micro-batch's layer opens?
      copy time = missed bytes / the link rate the run shows; window = the other half's layer
      (T(R/2) / n_layer). Layers where copy > window cannot be hidden by any ordering.
  Q2  Is the fork cheaper than the stall it hides? The fork is measured, not modelled: two
      micro-batch passes of r rows cost 2 x T(r) against T(R) for the merged pass (`--steps`, the
      clean runs' step times), and the stall the runahead can hide is the in-place read of the
      misses the halves route (`--per-miss`, the per-miss marginal stall of a pass).

Per phase it prints the rows, the routing and the misses of the pass, the halves' expectations, the
fork, the hideable stall, the net, and the per-layer merge-back (the item's "split only on layers
with many misses") with the number of layers that would split.

Expectations of a half come from the trace's per-expert row counts, which is what its `R` records
hold, under the hypergeometric model that a half is a random half of the pass's rows: an expert with
count c is routed by at least one of the r rows with probability 1 - C(R-c, r)/C(R, r). The trace
does not hold which row went where, so this is an expectation, and it is the generous direction: it
assumes the two halves' misses are independent samples of the same routing.

  runahead.py TRACE PROBE.json --steps c2=..,c4=..,c8=.. --half c4=c2,c8=c4
              [--rows c4=4,c8=8] [--pool] [--min-rows N] [--fork MS] [--per-miss US] [--link-us US] [--k TOPK] [--layers N]
              [--unit B] [--json OUT]
  runahead.py --self-test

`--steps` is the measured step time by probe phase in ms, from the same run's clean arm (no stamps),
and `--half` names the phase a half of a phase's rows runs at: the split of a c8 step is
2 x T(c4) against T(c8) for the merged step (a DFly2 step is its draft and verify passes together,
so both sides are steps).
`--per-miss` is the marginal stall a miss read in place costs a pass, default 34 us (the W2 design's
own fit on this lane, notes/backlog/measurements/W2-prototype.md §5, against a link time of ~66 us a
unit). `--link-us` is the same miss's full link time, default 66 us, which sets the link rate Q1
uses. The defaults are conservative in the runahead's favour: 34 us a miss is the stall measured
once the in-layer window has already hidden part of the read.
"""
import array
import json
import struct
import sys

def read(path):
    """(n_layer, n_expert, bytes[nl*ne], movable[nl*ne], T, R, D) of a RADIANCE_ROUTE_TRACE."""
    d = open(path, 'rb').read()
    if d[:8] != b'RADRTRC1':
        raise SystemExit(f'{path}: not a route trace')
    nl, ne = struct.unpack_from('<ii', d, 8)
    n = nl * ne
    o = 16
    by = array.array('q')
    by.frombytes(d[o:o + 8 * n])
    o += 8 * n
    mv = d[o:o + n]
    o += n
    T, R, D = [], [], []
    while o < len(d):
        k = d[o:o + 1]
        o += 1
        if k == b'T':
            disp, = struct.unpack_from('<i', d, o)
            o += 4
            T.append((disp, d[o:o + n]))
            o += n
        elif k == b'R':
            disp, hs, layer, reps, missed, m = struct.unpack_from('<iihhii', d, o)
            o += 20
            pr = array.array('H')
            pr.frombytes(d[o:o + 4 * m])
            o += 4 * m
            R.append((disp, hs, layer, reps, missed, list(zip(pr[0::2], pr[1::2]))))
        elif k == b'D':
            D.append(struct.unpack_from('<iiQQQQQQQ', d, o))
            o += 8 + 56
        else:
            raise SystemExit(f'{path}: bad record {k!r} at {o - 1}')
    return nl, ne, by, mv, T, R, D


def tier_at(T, disp, n):
    """The residency the trace's snapshots give at a dispatch (the last snapshot at or before it)."""
    best = None
    for d, t in T:
        if d <= disp:
            best = t
        else:
            break
    return best if best is not None else (T[0][1] if T else bytes(n))


def hyper_at_least(R, c, r):
    """P(at least one of r rows of R routes an expert that c of the R rows route)."""
    if c <= 0 or r <= 0:
        return 0.0
    if R - c < r:
        return 1.0
    p = 1.0
    for i in range(r):
        p *= (R - c - i) / (R - i)
    return 1.0 - p


def pass_stats(tr, probe, k=8, layers=None, rows_of=None, pool=False, min_rows=2):
    """Per phase: the routing, the misses and the halves' expectations, from the trace.

    `counts` of a trace row is the credited routing histogram of one pass at one layer: the rows of
    the pass counted per expert, k of them a row (so they sum to rows x k). A phase's dispatches are
    those whose row count is the height the phase decodes at (`rows_of`, else the phase's
    `tok_per_step` rounded): the trace also holds the phase's prefill passes (whose "rows" are
    tokens) and the single-client steps of a staggered start, and those are not the shape a
    micro-batch fork would cut. With `pool` the whole trace is one phase instead - the shape for a DFly2
    run, whose verify height moves with the draft length, so no height selects its phases; pooled, a
    pass of `min_rows` to 64 rows is a decode pass (a prompt is 50 tokens and up); a run that
    starts its clients one by one has single-client passes at 2..6 rows, so raise it for one.

    The halves' model is per dispatch, at that dispatch's own height: a half is a random half of the
    pass's rows, and an expert the pass routes with c of its R rows is routed by the half with
    1 - C(R-c, r)/C(R, r), r = R/2. The snapshot says which of the pass's experts sat off the card;
    the record's own `missed` says how many did, and the two drift apart between snapshots (one is
    taken every 256 dispatches), so the snapshot's set is scaled to the record's count.
    """
    nl, ne, by, mv, T, R, D = read(tr)
    n = nl * ne
    unit = max(by) if by else 0
    by_disp = {}
    for rec in R:
        by_disp.setdefault(rec[0], []).append(rec)
    disp_rows = {}
    for disp in sorted(by_disp):
        per = sorted(sum(c for _, c in pr) // max(k, 1) for _, _, _, _, _, pr in by_disp[disp])
        disp_rows[disp] = per[len(per) // 2] if per else 0
    if pool:
        # a prompt is 50 tokens and up, so a pass of 2..64 rows is a decode pass of the run
        todo = [('all', [d for d in sorted(disp_rows) if min_rows <= disp_rows[d] <= 64])]
    else:
        todo = []
        for ph in [q for q in probe['phases'] if isinstance(q, dict) and 'stats' in q]:
            rows = (rows_of or {}).get(ph['name']) or int(round(ph.get('tok_per_step') or 0))
            keep = [d for d in sorted(disp_rows) if disp_rows[d] == rows]
            if keep:
                todo.append((ph['name'], keep))
    out = []
    for name, keep in todo:
        per_layer = {}
        for disp in keep:
            rows = max(1, disp_rows[disp])
            r = max(1, rows // 2)
            t = tier_at(T, disp, n)
            for _, hs, layer, reps, missed, pr in by_disp[disp]:
                if layers is not None and not (layers[0] <= layer < layers[1]):
                    continue
                L = per_layer.setdefault(layer, dict(distinct=0.0, missed=0.0, miss_bytes=0.0,
                                                     dist_half=0.0, miss_half=0.0))
                L['distinct'] += len(pr)
                L['missed'] += missed
                b, snap, ph = 0, 0, 0.0
                for e, c in pr:
                    p = hyper_at_least(rows, c, r)
                    L['dist_half'] += p
                    if t[layer * ne + e] != 0:              # 0 is Tier::VRAM in the snapshot
                        b = by[layer * ne + e] or b
                        snap += 1
                        ph += p
                if snap:
                    f = missed / snap
                    L['miss_bytes'] += f * snap * b
                    L['miss_half'] += f * ph
        if not per_layer:
            continue
        d_n = max(1, len(keep))
        for L in per_layer.values():                    # a phase holds one dispatch a step
            for f in ('distinct', 'missed', 'miss_bytes', 'dist_half', 'miss_half'):
                L[f] /= d_n
        rows = sorted(disp_rows[d] for d in keep)
        out.append(dict(name=name, rows=rows[len(rows) // 2], half=max(1, rows[len(rows) // 2] // 2),
                        dispatches=len(keep), dropped=0, layers=len(per_layer),
                        distinct=sum(L['distinct'] for L in per_layer.values()),
                        missed=sum(L['missed'] for L in per_layer.values()),
                        miss_bytes=sum(L['miss_bytes'] for L in per_layer.values()),
                        distinct_half=sum(L['dist_half'] for L in per_layer.values()),
                        missed_half=sum(L['miss_half'] for L in per_layer.values()),
                        per_layer=per_layer, unit=unit, steps=d_n))
    return out


def model(stats, steps, half_of, per_miss_us=34.0, link_us=66.0, fork=None):
    """The fork, the stall it can hide and the per-layer merge-back of one phase.

    `steps` is the measured step time by phase name and `half_of` names the phase a half of this one
    runs at, so the fork is 2*T(half) - T(merged), measured, not modelled. The stall a merged pass
    pays is its misses at the per-miss cost; the copies it would have to move are the same misses at
    the link cost, and the link is the larger of the two, so the stall is what a schedule can win.
    Per layer the fork costs the uniform share of the measured fork, and a layer is worth splitting
    when its own stall beats that share.
    """
    name = stats['name']
    want = half_of.get(name)
    t_merged = steps.get(name)
    t_half = steps.get(want)
    if fork is None:
        if t_merged is None or t_half is None:
            return None
        fork = 2.0 * t_half - t_merged              # ms a step, measured
    if t_half is None:
        t_half = t_merged                           # a pooled run: the window from the step itself
    if t_half is None:
        return None
    unit = stats['unit']
    link_rate = (unit / link_us) * 1e3 if unit and link_us else 0.0   # bytes / ms
    window = t_half / max(stats['layers'], 1)       # ms: the other half's time on one layer
    stall_ms = stats['missed'] * per_miss_us / 1000.0      # what the merged pass pays, a step
    link_ms = stats['missed'] * link_us / 1000.0           # the copies that would hide it
    hide = min(stall_ms, link_ms)
    share = fork / max(len(stats['per_layer']), 1)
    per_layer = []
    for layer, L in sorted(stats['per_layer'].items()):
        copy_ms = L['miss_half'] * unit / link_rate if link_rate else 0.0
        L_stall = L['missed'] * per_miss_us / 1000.0
        per_layer.append(dict(layer=layer, fits=copy_ms <= window, copy_ms=copy_ms,
                              window_ms=window, stall_ms=L_stall, share_ms=share,
                              split=L_stall > share, gain_ms=L_stall - share,
                              miss_half=L['miss_half'], missed=L['missed']))
    split = [x for x in per_layer if x['split']]
    return dict(rows=stats['rows'], half=stats['half'], half_phase=want, t_merged=t_merged,
                t_half=t_half, fork=fork, hide=hide, stall_ms=stall_ms, link_ms=link_ms,
                net=hide - fork, link_rate=link_rate, window=window, per_layer=per_layer,
                fits=sum(1 for x in per_layer if x['fits']), merge_n=len(split),
                merge_gain=sum(x['gain_ms'] for x in split),
                merge_net=sum(x['gain_ms'] for x in split),
                merge_split_gain=sum(x['stall_ms'] for x in split),
                merge_split_cost=sum(x['share_ms'] for x in split))


def self_test():
    """A made-up trace with known answers: the parser, the half expectations and the model."""
    import os
    import tempfile
    nl, ne = 2, 4
    by = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    # 24 dispatches, the same routing each: k = 2, 4 rows, so each layer's counts sum to 8
    rows = [(d, 0, layer, 1, missed, pr)
            for d in range(24)
            # layer 0: 2 rows on e0, 2 on e1, 4 on e2; layer 1: 4 rows on each of two experts
            for layer, missed, pr in ((0, 1, [(0, 2), (1, 2), (2, 4)]),
                                      (1, 0, [(2, 4), (3, 4)]))]
    tier = bytes([1, 0, 0, 0, 0, 0, 0, 0])          # expert 0 of layer 0 is off the card
    body = b'RADRTRC1' + struct.pack('<ii', nl, ne) + array.array('q', by).tobytes() + \
        bytes(8) + b'T' + struct.pack('<i', 0) + tier
    for d, hs, layer, reps, missed, pr in rows:
        body += b'R' + struct.pack('<iihhii', d, hs, layer, reps, missed, len(pr))
        body += b''.join(struct.pack('<HH', e, c) for e, c in pr)
    for i in range(24):
        body += b'D' + struct.pack('<iiQQQQQQQ', i, 0, 8 * (i + 1), 8, 0, 0, 0, 0, 0)
    probe = {'phases': [{'name': 'c4', 'stats': {'routed': 8}}]}
    fd, path = tempfile.mkstemp(suffix='.rt')
    os.write(fd, body)
    os.close(fd)
    try:
        st = pass_stats(path, probe, k=2, rows_of={'c4': 4})
        bad = []

        def want(name, got, exp, tol=1e-9):
            if abs(got - exp) > tol:
                bad.append(f'{name}: got {got}, want {exp}')

        if len(st) != 1:
            print('SELF-TEST FAILED: no phase', st)
            return 1
        s = st[0]
        want('rows', s['rows'], 4)
        want('half', s['half'], 2)
        want('distinct', s['distinct'], 5)
        want('missed', s['missed'], 1)
        want('miss_bytes', s['miss_bytes'], 1000)
        # an expert routed by 2 of 4 rows is routed by a half of 2 rows with 1 - C(2,2)/C(4,2) = 5/6;
        # one routed by all 4 is always routed
        want('distinct_half', s['distinct_half'], 5.0 / 6.0 + 5.0 / 6.0 + 3.0)
        want('missed_half', s['missed_half'], 5.0 / 6.0)
        st[0]['name'] = 'c4'
        mo = model(s, {'c2': 4.0, 'c4': 10.0}, {'c4': 'c2'}, per_miss_us=60.0, link_us=100.0)
        want('fork', mo['fork'], -2.0)                         # 2 x 4 - 10 ms
        want('hide', mo['hide'], 1 * 0.060)                    # a step's one miss at 60 us
        want('net', mo['net'], mo['hide'] - mo['fork'])
        want('link_rate', mo['link_rate'], 1e4)                 # 1000 B / 100 us -> 10 kB/ms
        want('window', mo['window'], 2.0)                      # 4 ms / 2 layers
        if bad:
            print('SELF-TEST FAILED:\n  ' + '\n  '.join(bad))
            return 1
        print('self-test: ok (parser, half expectations 5/6 and 3, fork -2.0 ms, hide 0.060 ms, '
              'link 10 kB/ms, window 2.0 ms)')
        return 0
    finally:
        os.unlink(path)


def steps_from(path, name=None):
    """The step time a probe run's phases measured (its `<tag>.json`, or the run's stdout).

    The decode window of the slowest-starting client over the steps the engine counted; probe.py's
    own ms_step is only filled for a single client.
    """
    txt = open(path).read()
    try:
        phases = json.loads(txt).get('phases', [])
    except json.JSONDecodeError:                    # a run's stdout: one phase per JSON line
        phases = [json.loads(l) for l in txt.splitlines() if l.startswith('{')]
    out = []
    for p in phases:
        if not isinstance(p, dict) or 'all_decoding_s' not in p:
            continue
        a, b = p['all_decoding_s']
        st = p.get('decode_steps') or 0
        if st:
            out.append((name or p['name'], 1000.0 * (b - a) / st, p['name']))
    return out


def main():
    a = sys.argv[1:]
    if a and a[0] == '--steps-from':
        pairs = []
        for f in a[1:]:
            for ph, ms, real in steps_from(f):
                print(f'  {f}: {real} {ms:.3f} ms/step')
                pairs.append(f'{ph}={ms:.3f}')
        print('--steps ' + ','.join(pairs))
        return 0
    if a and a[0] == '--self-test':
        return self_test()
    if len(a) < 2:
        raise SystemExit(__doc__.split('\n\n')[2])
    tr, pj = a[0], a[1]
    steps, half_of, rows_of, pool = {}, {}, {}, False
    per_miss, link_us, k, layers, unit, outp = 34.0, 66.0, 8, None, 0, None
    fork_given = None
    min_rows = 2
    for i, x in enumerate(a):
        if x == '--steps':
            for kv in a[i + 1].split(','):
                h, v = kv.split('=')
                steps[h] = float(v)
        if x == '--fork':
            fork_given = float(a[i + 1])
        if x == '--min-rows':
            min_rows = int(a[i + 1])
        if x == '--pool':
            pool = True
        if x == '--rows':
            for kv in a[i + 1].split(','):
                h, v = kv.split('=')
                rows_of[h] = int(v)
        if x == '--half':
            for kv in a[i + 1].split(','):
                h, v = kv.split('=')
                half_of[h] = v
        if x == '--per-miss':
            per_miss = float(a[i + 1])
        if x == '--link-us':
            link_us = float(a[i + 1])
        if x == '--k':
            k = int(a[i + 1])
        if x == '--layers':
            lo, hi = a[i + 1].split(':')
            layers = (int(lo), int(hi))
        if x == '--unit':
            unit = int(a[i + 1])
        if x == '--json':
            outp = a[i + 1]
    probe = json.load(open(pj))
    st = pass_stats(tr, probe, k=k, layers=layers, rows_of=rows_of, pool=pool, min_rows=min_rows)
    if unit:
        for s in st:
            s['unit'] = unit
    print(f'{tr}: {len(st)} phases; per-miss {per_miss} us, link {link_us} us a miss; '
          f'steps {steps} ms; halves {half_of}')
    keep = []
    for s in st:
        mo = model(s, steps, half_of, per_miss, link_us, fork=fork_given)
        line = (f"  {s['name']:>8}: {s['rows']:>3} rows ({s['half']} a half) x {s['dispatches']} steps, "
                f"{s['layers']} layers, {s['distinct']:.1f} routed experts and "
                f"{s['missed']:.2f} misses a step ({s['miss_bytes'] / 1e6:.2f} MB of link)")
        print(line)
        if mo is None:
            print(f"           (no step time for {s['name']} or its half: skipped)")
            continue
        print(f"           halves: {s['distinct_half']:.1f} routed, {s['missed_half']:.2f} misses a "
              f"half-step, x 2 = {2 * s['missed_half']:.2f}")
        src = (f"2 x {mo['half_phase']} {mo['t_half']:.2f} - {s['name']} {mo['t_merged']:.2f}"
               if mo['t_merged'] is not None else f"given, window from {s['name']} {mo['t_half']:.2f}")
        print(f"           fork {mo['fork']:+.2f} ms a step ({src}), hideable stall {mo['hide']:.2f} ms, "
              f"net {mo['net']:+.2f} ms")
        print(f"           window {mo['window'] * 1000:.0f} us a layer, link {mo['link_rate'] / 1e6:.1f} MB/ms; "
              f"{mo['fits']}/{len(mo['per_layer'])} layers fit their copy")
        print(f"           merge-back: {mo['merge_n']}/{len(mo['per_layer'])} layers split, "
              f"gain {mo['merge_gain']:.2f} ms, net {mo['merge_net']:+.2f} ms a step")
        top = sorted(mo['per_layer'], key=lambda x: -x['missed'])[:3]
        print("           hottest layers: " + ", ".join(
            f"L{x['layer']} {x['missed']:.2f} misses ({x['stall_ms'] * 1000:.0f} us stall, "
            f"share {x['share_ms'] * 1000:.0f} us, {'x' if x['split'] else '-'})" for x in top))
        keep.append(dict(phase=s['name'], **{x: mo[x] for x in
                                             ('rows', 'half', 'half_phase', 'fork', 'hide', 'net',
                                              'window', 'fits', 'merge_n', 'merge_gain',
                                              'merge_net')},
                         layers=len(mo['per_layer'])))
    if outp:
        json.dump(keep, open(outp, 'w'), indent=1)
        print(f'wrote {outp}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
