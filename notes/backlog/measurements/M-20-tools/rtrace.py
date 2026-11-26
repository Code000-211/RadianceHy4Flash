#!/usr/bin/env python3
"""rtrace.py -- reads a RADIANCE_ROUTE_TRACE file (core/place/rtrace.h) for the M-20 design.

  rtrace.py TRACE PROBE.json [--gu B] [--dn B]

Splits the trace into the probe's phases (from the cumulative `routed` of the D records against the
phases' /stats deltas, walking back from the end), then per phase:
  - live misses a dispatch and a layer (the heat engine's own `live_missed`, i.e. today's whole-unit
    misses read over the link), and the routed experts a layer;
  - the VRAM budget the phase ran with (units resident in the T snapshots x the unit's bytes);
  - a residency simulation: an online decayed-frequency heat (decay 0.95 a dispatch, credited
    like heat.observe), residency = the top of that ranking with no move limit, for
      (a) today's whole units: N = budget / unit bytes,
      (b) the split: every expert's gate_up ranked first, then downs, for a sweep of how much of the
          budget goes to gate_up; a routed expert is a full miss when its gate_up is not resident
          and a down-only miss when its gate_up is and its down is not.
"""
import array, json, struct, sys, collections

def read(path):
    d = open(path, 'rb').read()
    assert d[:8] == b'RADRTRC1'
    nl, ne = struct.unpack_from('<ii', d, 8); n = nl * ne; o = 16
    by = array.array('q'); by.frombytes(d[o:o + 8 * n]); o += 8 * n
    mv = d[o:o + n]; o += n
    T, R, D = [], [], []
    while o < len(d):
        k = d[o:o + 1]; o += 1
        if k == b'T':
            disp, = struct.unpack_from('<i', d, o); o += 4
            T.append((disp, d[o:o + n])); o += n
        elif k == b'R':
            disp, hs, layer, reps, missed, m = struct.unpack_from('<iihhii', d, o); o += 20
            pr = array.array('H'); pr.frombytes(d[o:o + 4 * m]); o += 4 * m
            R.append((disp, hs, layer, reps, missed, list(zip(pr[0::2], pr[1::2]))))
        elif k == b'D':
            v = struct.unpack_from('<iiQQQQQQQ', d, o); o += 8 + 56
            D.append(v)
        else:
            raise SystemExit(f'bad record {k!r} at {o - 1}')
    return nl, ne, by, mv, T, R, D

def phases_of(D, probe):
    """[(name, d0, d1)] dispatch ranges (d0 exclusive) from the D records' cumulative routed."""
    ph = [p for p in probe['phases'] if isinstance(p, dict) and 'stats' in p]
    disp = [x[0] for x in D]; routed = [x[2] for x in D]
    out = []; end = len(D) - 1
    # walk back from the end: each phase's routed delta
    for p in reversed(ph):
        want = p['stats'].get('routed', 0)
        top = routed[end]
        lo = end
        while lo > 0 and top - routed[lo - 1] < want: lo -= 1
        # lo is the first D index inside the phase
        out.append((p['name'], disp[lo - 1] if lo > 0 else -1, disp[end], top - (routed[lo - 1] if lo > 0 else 0), want))
        # skip the idle gap: the previous phase ends at the last D before lo whose routed moved
        end = lo - 1
        while end > 0 and routed[end] == routed[end - 1]: end -= 1
    return list(reversed(out))

def main():
    tr, pj = sys.argv[1], sys.argv[2]
    gu = 891136; dn = None; poll = None; outp = None
    a = sys.argv[3:]
    for i, x in enumerate(a):
        if x == '--gu': gu = int(a[i + 1])
        if x == '--dn': dn = int(a[i + 1])
        if x == '--poll': poll = a[i + 1]
        if x == '--out': outp = a[i + 1]
    nl, ne, by, mv, T, R, D = read(tr)
    unit = max(by); dn = dn or unit - gu
    probe = json.load(open(pj))
    rng = phases_of(D, probe)
    print(f'trace {tr}: {nl} layers x {ne} experts, unit {unit} B (gate_up {gu}, down {dn}), '
          f'{len(R)} rows, {len(T)} tier snapshots, {len(D)} dispatches')
    win = {p['name']: p['mono_ns'] for p in probe['phases'] if isinstance(p, dict) and p.get('mono_ns')}
    tv = [(disp, sum(1 for i in range(len(t)) if by[i] > 0 and t[i] == 0)) for disp, t in T]
    pb = {}
    if poll:
        P = [json.loads(l) for l in open(poll)]
        for name, (a0, a1) in win.items():
            v = [x['vram'] for x in P if a0 <= x['ns'] <= a1 and x.get('vram')]
            if v: pb[name] = (sum(v) / len(v), min(v), max(v), len(v))
    for name, d0, d1, got, want in rng:
        b = pb.get(name)
        print(f'  phase {name}: dispatches ({d0}, {d1}], routed {got} (probe {want}); vram units '
              + (f'mean {b[0]:.0f} min {b[1]} max {b[2]} ({b[3]} polls)' if b else 'from tier snapshots'))
    heat = [0.0] * (nl * ne)
    decay = float(next((a[i + 1] for i, x in enumerate(a) if x == '--decay'), 0.95))
    res = collections.defaultdict(lambda: collections.defaultdict(float))
    hist = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.Counter()))
    fr_sweep = ['today', 0.62, 0.65, 0.68, 0.70, 0.75, 'allgu']
    def phase_at(disp):
        for name, d0, d1, *_ in rng:
            if d0 < disp <= d1: return name
        return None
    def budget_at(disp, ph):
        if ph in pb: return int(pb[ph][0])
        best = None
        for dd, u in tv:
            if dd <= disp: best = u
        return best if best is not None else tv[0][1]
    order = [i for i in range(nl * ne) if by[i] > 0]
    pos = [10**9] * (nl * ne)
    rows_by_disp = collections.defaultdict(list)
    for r in R: rows_by_disp[r[0]].append(r)
    for disp in sorted(rows_by_disp):
        ph = phase_at(disp)
        rows = rows_by_disp[disp]
        if ph is not None:
            order.sort(key=lambda i: -heat[i])
            for k, i in enumerate(order): pos[i] = k
            Nu = budget_at(disp, ph)
            B = Nu * unit
            S = res[ph]
            S['dispatches'] += 1
            S['units'] += Nu
            sets = {}
            for f in fr_sweep:
                if f == 'today': Ng = Nd = Nu
                elif f == 'allgu':
                    Ng = nl * ne - 512; Nd = int(max(0.0, B - Ng * gu) // dn)
                else:
                    Ng = min(nl * ne - 512, int(f * B // gu)); Nd = int(max(0.0, B - Ng * gu) // dn)
                sets[f] = (Ng, Nd)
            S['Ng_allgu_fits'] = 1.0 if (nl * ne - 512) * gu <= B else 0.0
            for (_, hs, layer, reps, missed, pairs) in rows:
                S['rows'] += 1
                S['live_missed'] += missed
                S['routed_experts'] += len(pairs)
                for f, (Ng, Nd) in sets.items():
                    nf = nd_ = 0
                    for e, c in pairs:
                        p = pos[layer * ne + e]
                        if p >= Ng: nf += 1
                        elif p >= Nd: nd_ += 1
                    S[f'{f}_full'] += nf; S[f'{f}_dn'] += nd_
                    hist[ph][str(f)][(nf, nd_)] += 1
                hist[ph]['live'][(missed, 0)] += 1
        for (_, hs, layer, reps, missed, pairs) in rows:
            for e, c in pairs:
                heat[layer * ne + e] += c * (reps if reps > 1 else 1)
        for i in range(nl * ne): heat[i] *= decay
    out = {}
    for name, *_ in rng:
        S = res.get(name)
        if not S: continue
        nd = S['dispatches']; nr = S['rows']
        Nu = S['units'] / nd
        o = {'dispatches': nd, 'layer_rows': nr, 'units_resident_mean': Nu,
             'budget_gib': Nu * unit / 2**30, 'all_gate_up_gib': (nl * ne - 512) * gu / 2**30,
             'routed_experts_per_layer': S['routed_experts'] / nr,
             'live_missed_per_dispatch': S['live_missed'] / nd, 'live_missed_per_layer': S['live_missed'] / nr}
        for f in fr_sweep:
            if f == 'today': Ng = Nd = int(Nu)
            elif f == 'allgu': Ng = nl * ne - 512; Nd = int(max(0.0, Nu * unit - Ng * gu) // dn)
            else: Ng = min(nl * ne - 512, int(f * Nu * unit // gu)); Nd = int(max(0.0, Nu * unit - Ng * gu) // dn)
            o[str(f)] = {'gate_up_resident': Ng, 'down_resident': Nd, 'full_per_layer': S[f'{f}_full'] / nr,
                         'down_only_per_layer': S[f'{f}_dn'] / nr}
        out[name] = o
    print(json.dumps(out, indent=1))
    if outp:
        json.dump({'summary': out, 'hist': {ph: {f: [[k[0], k[1], v] for k, v in c.items()] for f, c in d.items()}
                                            for ph, d in hist.items()}}, open(outp, 'w'))

if __name__ == '__main__':
    main()
