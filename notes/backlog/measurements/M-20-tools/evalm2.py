#!/usr/bin/env python3
"""evalm2.py SIM.json PHASE b_gu b_dn W0 Tdn0 L0 tail [scale] -- four variants of one routed layer's
miss stall on the simulated (full, down-only) histogram of rtrace.py --out:

  today     (b_gu + b_dn) * n_full                      in-place link reads inside moe_iqr / moe_down
  M-20      split placement, down copies posted after the router, landing inside W0 (router end ->
            down start, no-miss intercept); a copy that has not landed is read in place (b_dn);
            full misses as today; + plan/patch launches (2 x 2.6 us) when the layer has a down miss
  Weg 2     whole-unit placement; resident experts run gate_up -> quant -> down, the missing units
            are copied by SDMA meanwhile (window W0 + Tdn0), then a tail gate_up/quant/down runs
            them from VRAM: max(0, L0 + T_unit(n) - W2) + tail
  M-20+W2   split placement; down-only misses wait like Weg 2 (T_down(n)); full misses copy whole
            units in the same list (T for n_full units + n_dn downs)
T(n) are the measured two-card list times (M20XP two-cards, p50). `scale` multiplies the result
of the simulated histogram (live misses / simulated misses of `today`), default 1."""
import json, sys
d = json.load(open(sys.argv[1])); ph = sys.argv[2]
bgu, bdn, W0, Tdn0, L0, tail = map(float, sys.argv[3:9]); scale = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
DN = [(0, 0.0), (1, 57.2), (2, 104.9), (4, 162.1), (8, 265.3), (16, 542.5), (32, 1073.6)]
UN = [(0, 0.0), (1, 90.6), (2, 172.4), (4, 339.4), (8, 665.4), (16, 1347.4), (32, 2679.2)]
def T(tab, n):
    if n <= 0: return 0.0
    for (n0, t0), (n1, t1) in zip(tab, tab[1:]):
        if n <= n1: return t0 + (t1 - t0) * (n - n0) / (n1 - n0)
    n0, t0 = tab[-1]; return t0 + (n - n0) * (tab[-1][1] - tab[-2][1]) / (tab[-1][0] - tab[-2][0])
def landed(t):
    k = 0
    while T(DN, k + 1) <= t and k < 512: k += 1
    return k
k = landed(W0 - L0); W2 = W0 + Tdn0; LAUNCH = 2.6
H = d['hist'][ph]; S = d['summary'][ph]
def avg(rows, fn):
    n = sum(c for *_, c in rows); return sum(fn(a, b) * c for a, b, c in rows) / n
base = avg(H['today'], lambda a, b: (bgu + bdn) * a)
w2 = avg(H['today'], lambda a, b: (max(0.0, L0 + T(UN, a) - W2) + tail) if a else 0.0)
print(f'{ph}: W0 {W0:.0f} us (M-20 lands {k} downs), W2 {W2:.0f} us; today {base * scale:6.1f} us a layer '
      f'{base * scale * 41 / 1e3:5.2f} ms a pass | Weg 2 alone {w2 * scale:6.1f} us ({(w2 - base) * scale * 41 / 1e3:+5.2f} ms)')
for f in ('0.62', '0.65', '0.68', '0.75', 'allgu'):
    if f not in H: continue
    m20 = avg(H[f], lambda a, b: (bgu + bdn) * a + bdn * max(0, b - k) + (2 * LAUNCH if b else 0.0))
    cmb = avg(H[f], lambda a, b: (max(0.0, L0 + T(UN, a) + T(DN, b) - W2) + tail) if (a or b) else 0.0)
    s = S[f]
    print(f'   split {f:>5} (gate_up {s["gate_up_resident"]}, down {s["down_resident"]}; full {s["full_per_layer"]:.2f} '
          f'down-only {s["down_only_per_layer"]:.2f}): M-20 {m20 * scale:6.1f} us ({(m20 - base) * scale * 41 / 1e3:+5.2f} ms) | '
          f'M-20+W2 {cmb * scale:6.1f} us ({(cmb - base) * scale * 41 / 1e3:+5.2f} ms)')
