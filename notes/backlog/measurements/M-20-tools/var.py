#!/usr/bin/env python3
"""var.py [L0_us] [fold] [tailmiss] -- M-20 variant model: per layer and per miss choose (1) in place, (2) early down copy,
(3) early whole-unit copy + tail (Weg 2); fluid sharing of the link between the in-place reads and
the SDMA list, calibrated on ccbench (in-place read kernel || SDMA copy, both cards / one card)."""
import json, sys, itertools
import os
SP = os.environ.get('M20_SIM_DIR', '/tmp/claude-0/-root-hy4/c566ea29-7612-537f-abca-9a9da726328e/scratchpad/m20') + '/'  # rtrace.py --out JSONs
G, D, U = 891136, 608768, 1499904
def interp(tab, x):
    if x <= 0: return 0.0
    for (x0, y0), (x1, y1) in zip(tab, tab[1:]):
        if x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = tab[-2], tab[-1]; return y1 + (x - x1) * (y1 - y0) / (x1 - x0)
RAM = {
 # copy list alone (rad_xp_list p50): TP2 both cards today / one card = fixed-RAM proxy
 'today': dict(DN=[(0,0),(1,57.2),(2,104.9),(4,162.1),(8,265.3),(16,542.5),(32,1073.6)],
               UN=[(0,0),(1,90.6),(2,172.4),(4,339.4),(8,665.4),(16,1347.4),(32,2679.2)],
               # in-place read kernel alone, n gate_ups (ccbench two-cards)
               K=[(0,0),(1,60.4),(2,94.9),(3,132.7),(4,172.5),(8,330.2)],
               ab={1:(0.60,0.57)}, abN=(0.80,0.28), bscale=1.0),
 'fixed': dict(DN=[(0,0),(1,25.7),(2,53.8),(4,101.7),(8,197.6),(16,391.0),(32,774.0)],
               UN=[(0,0),(1,63.3),(2,119.4),(4,244.6),(8,473.8),(16,963.4),(32,1900.6)],
               K=[(0,0),(1,49.5),(2,87.9),(3,120.4),(4,135.5),(8,261.5)],
               ab={1:(0.69,0.52)}, abN=(0.83,0.30), bscale=0.85),
}
def fluid(R, Cw, cs, a, b):
    """read flow of work R from t=0, copy of work Cw from t=cs; shares a/b while both run.
    returns (read end, function copy-progress->time)."""
    # phase 1: [0, cs) read alone
    t, r, c = 0.0, R, Cw; ev = []  # (t, copy done work) breakpoints
    if r <= cs: tr = r; t = cs; ev.append((cs, 0.0)); rate_c = 1.0; return tr, (lambda w: cs + w)
    r -= cs; t = cs; ev.append((t, 0.0))
    # both
    tb_r = r / a; tb_c = c / b
    if tb_r <= tb_c:
        t1 = t + tb_r; done = b * tb_r; tr = t1
        return tr, (lambda w, t=t, t1=t1, done=done: t + w / b if w <= done else t1 + (w - done))
    else:
        t1 = t + tb_c; rem = r - a * tb_c; tr = t1 + rem
        return tr, (lambda w, t=t: t + w / b)

SH = {  # Tgu0 b_gu Tdn0 b_dn W0 (us, measured, M-20 §3 / measurements §3)
 'DFly2 c1':  (87.9, 31.6, 53.2, 21.3, 100.7, 'c03', 'c1'),
 'DFly2 c8':  (443.0, 21.7, 147.0, 31.5, 473.0, 'c03', 'c8'),
 'DFly2 lctx1': (112.3, 10.6, 67.3, 7.8, 130.5, 'c03', 'lctx1-decode'),
 'DFly2 lctx4': (285.8, 10.2, 153.3, 9.6, 309.0, 'c03', 'lctx4-decode'),
 'nospec c1': (34.5, 45.5, 22.1, 35.6, 43.6, 'ns', 'c1'),
 'nospec c8': (170.1, 29.5, 105.7, 13.1, 175.7, 'ns', 'c8'),
 'nospec lctx1': (40.3, 24.2, 27.3, 12.7, 52.2, 'ns', 'lctx1-decode'),
 'nospec lctx4': (81.8, 42.3, 47.5, 30.8, 102.0, 'ns', 'lctx4-decode'),
}
HIST = {'c03': json.load(open(SP + 'sim-c03-2-d0.8.json'))['hist'], 'ns': json.load(open(SP + 'sim-nospec-2-d0.8.json'))['hist']}
PLAN, TAIL_L, TAIL_E = 2.6, 3 * 2.6, 4.2

def layer(sh, ram, n, a, b, L0):
    """layer time router end -> routed block end, n misses: a whole-unit copies (tail), b early downs, rest in place."""
    Tgu0, bgu, Tdn0, bdn, W0 = sh[:5]; P = RAM[ram]; s = P['bscale']; bgu *= s; bdn *= s
    c = n - a - b; nr = b + c   # gate_ups read in place
    al, be = P['ab'].get(nr, P['abN']) if nr else (1, 1)
    DN, UN, K = P['DN'], P['UN'], P['K']
    # list: b downs then a units; work in "alone" time
    Wd = interp(DN, b); Wl = Wd + (interp(UN, a + b * D / U) - interp(UN, b * D / U) if a else 0.0)
    R = interp(K, nr)
    eg = (1.0 if EFULL else min(1.0, bgu * nr / R)) if nr else 0.0
    if Wl > 0 and nr:
        Rend, prog = fluid(R, Wl, L0, al, be)
        dgu = eg * (Rend - R)
    else:
        prog = (lambda w: L0 + w); dgu = 0.0
    gu_end = Tgu0 + bgu * nr + dgu
    tds = gu_end + (W0 - Tgu0)
    # downs landed before down start
    landed = sum(1 for j in range(1, b + 1) if prog(interp(DN, j)) <= tds)
    miss_dn = c + (b - landed)
    # down phase: in-place downs; contention with the copy still running (units, or late downs)
    ddn = 0.0
    t_copy_end = prog(Wl) if Wl > 0 else 0.0
    if miss_dn and t_copy_end > tds:
        Rd = interp(K, miss_dn) * D / G; ed = min(1.0, bdn * miss_dn / Rd)
        al2, be2 = P['ab'].get(miss_dn, P['abN'])
        ov = min(t_copy_end - tds, Rd / al2)
        ddn = ed * ov * (1 - al2)
    dn_end = tds + Tdn0 + bdn * miss_dn + ddn
    end = dn_end
    if a:
        end = max(dn_end, t_copy_end) + TAIL_E + 0.5 * (a - 1)
    return end

def today(sh, ram, n):
    Tgu0, bgu, Tdn0, bdn, W0 = sh[:5]; s = RAM[ram]['bscale']
    return W0 + Tdn0 + (bgu + bdn) * s * n

import functools, os
EFULL = os.environ.get('EFULL') == '1'
@functools.lru_cache(maxsize=None)
def best_c(name, ram, n, mode, L0):
    return best(SH[name], ram, n, mode, L0)
def best(sh, ram, n, mode, L0):
    t0 = today(sh, ram, n)
    if n == 0: return t0
    opts = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            if mode == '2' and a: continue
            if mode == 'w2' and b: continue
            if mode == 'w2all' and (a != n): continue
            opts.append(layer(sh, ram, n, a, b, L0))
    return min(opts) if mode != 'w2all' else opts[0]

def run(L0=10.0, fold_plan=False, tail_every=True, out=None):
    rows = []
    for name, sh in SH.items():
        H = HIST[sh[5]][sh[6]]['live']; N = sum(c for *_, c in H)
        r = {'shape': name, 'mpl': sum(a * c for a, b, c in H) / N}
        for ram in ('today', 'fixed'):
            base = sum(today(sh, ram, a) * c for a, b, c in H) / N
            stall = sum((today(sh, ram, a) - today(sh, ram, 0)) * c for a, b, c in H) / N
            r[f'stall_{ram}'] = stall * 41 / 1e3
            for mode in ('2', '23', 'w2', 'w2all'):
                t = sum(best_c(name, ram, min(a, 24), mode, L0) * c if a <= 24 else (best_c(name, ram, 24, mode, L0) + today(sh, ram, a) - today(sh, ram, 24)) * c for a, b, c in H) / N
                fix = (0 if fold_plan else PLAN)
                if mode in ('23', 'w2', 'w2all'):
                    fix += TAIL_L if tail_every else TAIL_L * sum(c for a, b, c in H if a) / N
                r[f'{mode}_{ram}'] = (t - base + fix) * 41 / 1e3
        rows.append(r)
    return rows

if __name__ == '__main__':
    L0 = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    fold = len(sys.argv) > 2 and sys.argv[2] == 'fold'
    te = not (len(sys.argv) > 3 and sys.argv[3] == 'tailmiss')
    for r in run(L0, fold, te):
        print(f"{r['shape']:13s} m/l {r['mpl']:5.2f} | today stall {r['stall_today']:5.2f} / {r['stall_fixed']:5.2f} | "
              f"(2) {r['2_today']:+5.2f} / {r['2_fixed']:+5.2f} | (2)+(3) {r['23_today']:+5.2f} / {r['23_fixed']:+5.2f} | "
              f"W2 capped {r['w2_today']:+5.2f} / {r['w2_fixed']:+5.2f} | W2 all {r['w2all_today']:+5.2f} / {r['w2all_fixed']:+5.2f}")
