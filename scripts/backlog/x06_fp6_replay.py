#!/usr/bin/env python3
"""x06_fp6_replay.py <q.log> <p.jsonl> [--step-ms T] [--tps TAU] [--row-ms R]: X-06 (FP-6) offline sizing.

Joins draft 1's coupled law Q_1 (`Q <seq> <pos> <x> <count> id:q ...` lines from RADIANCE_DEBUG_FP6_LOG, one per
bound verify) with the target law P at the same output position (x06_fp6.py: raw top-20 logprobs per output token;
requests are matched in order of first appearance of their seq) and replays recursive rejection sampling with
k = 1, 2, 3 siblings at depth 1, drawn without replacement from Q_1 (Gumbel-top-k, abi/rad_mtp_rrs.h), as exact
expectations over the draw orders:

  acc_k = Pr(one of the first k children is accepted)       (acc_1 = sum min(P, Q), today's chain)
  extra_k = acc_k - acc_1                                    (tokens a step a depth-1 leaf sibling set adds: the
                                                              accepted sibling's own row samples one more token)

P at temperature T is exp(lp / T) over the top 20, normalised with the tail outside them bounded by
[0, R * p20^(1/T - 1)] (R = raw mass outside the top 20; the midpoint is used); with T = 1 P is exact. Q ids outside
P's top 20 get P = 0 (low end) or P = p20, the most they can have (high end); both ends are reported.

The net throughput change at c1 for k siblings (k - 1 extra rows at R ms each, the chain's measured TAU tokens a
step and T ms a step): (TAU + extra_k) / (T + (k - 1) R) / (TAU / T) - 1.
"""
import itertools, json, math, sys


def args():
    a = sys.argv[1:]
    o = {"step-ms": None, "tps": None, "row-ms": 1.0, "temp": None}
    pos = []
    i = 0
    while i < len(a):
        if a[i].startswith("--"):
            o[a[i][2:]] = float(a[i + 1]); i += 2
        else:
            pos.append(a[i]); i += 1
    return pos, o


def load_q(path):
    by_seq, order = {}, []
    for ln in open(path):
        f = ln.split()
        if not f or f[0] != "Q":
            continue
        seq, p, x, n = int(f[1]), int(f[2]), int(f[3]), int(f[4])
        sup = [(int(t.split(":")[0]), float(t.split(":")[1])) for t in f[5:5 + n]]
        if seq not in by_seq:
            by_seq[seq] = {}; order.append(seq)
        by_seq[seq][p] = (x, sup)
    return [by_seq[s] for s in order]


def p_law(top, T, hi):
    """P over the top ids at temperature T: (dict id -> P, the value an id outside the top 20 gets: 0 or p20)."""
    lps = [lp for _, lp in top]
    raw = [math.exp(lp) for lp in lps]
    R = max(0.0, 1.0 - sum(raw))
    w = [math.exp(lp / T) for lp in lps]
    tail = R * (raw[-1] ** (1.0 / T - 1.0)) if raw else 0.0
    tail = R if T == 1.0 else tail / 2
    z = sum(w) + tail
    return {i: wi / z for (i, _), wi in zip(top, w)}, (w[-1] / z if hi and w else 0.0)


def acc(Pd, sup, k):
    """Exact Pr(accept one of the first k Gumbel-top-k children of Q), RRS as rad_mtp_rrs_decide_over."""
    ids = [i for i, _ in sup]; q = [v for _, v in sup]
    s = sum(q); q = [v / s for v in q]
    n = len(ids); k = min(k, n)
    P, dflt = Pd
    p = [P.get(i, dflt) for i in ids]
    pout = max(0.0, 1.0 - sum(p))
    tot = 0.0
    for order in itertools.permutations(range(n), k):
        # probability of this draw order without replacement
        pr, left = 1.0, 1.0
        for j in order:
            pr *= q[j] / left; left -= q[j]
            if left <= 1e-15 and j != order[-1]:
                pr = 0.0
                break
        if pr == 0.0:
            continue
        r = p[:]; so = 1.0; qr = q[:]; reject = 1.0; a = 0.0
        for j in order:
            sq = sum(qr)
            if sq <= 0:
                break
            qx = qr[j] / sq
            aj = min(1.0, r[j] / qx) if qx > 0 else 0.0
            a += reject * aj; reject *= (1 - aj)
            if reject <= 0:
                break
            z = so * pout
            for m in range(n):
                d = r[m] - qr[m] / sq
                r[m] = d if d > 0 else 0.0
                z += r[m]
            if z <= 0:
                break
            r = [v / z for v in r]; so /= z; qr[j] = 0.0
        tot += pr * a
    return tot


def main():
    (qpath, ppath), o = args()
    Q = load_q(qpath)
    reqs = [json.loads(l) for l in open(ppath)]
    T = o["temp"] or reqs[0]["temp"]
    # Pair requests with seqs in order; positions: pick the offset that best matches Q's x to the emitted token.
    best = None
    for off in range(-3, 4):
        hit = tot = 0
        for qs, r in zip(Q, reqs):
            for pos, (x, sup) in qs.items():
                j = pos + off
                if 0 <= j < len(r["toks"]):
                    tot += 1; hit += r["toks"][j][0] == x
        if tot and (best is None or hit / tot > best[1]):
            best = (off, hit / tot, tot)
    off = best[0]
    S = {k: [0.0, 0.0] for k in (1, 2, 3)}; n = 0; miss_top = 0; nq = 0
    for qs, r in zip(Q, reqs):
        for pos, (x, sup) in qs.items():
            j = pos + off
            if not (0 <= j < len(r["toks"])) or len(sup) < 2:
                continue
            top = r["toks"][j][1]
            ids = {i for i, _ in top}
            miss_top += sum(1 for i, _ in sup if i not in ids); nq += len(sup)
            for hi, col in ((False, 0), (True, 1)):
                P = p_law(top, T, hi)
                for k in S:
                    S[k][col] += acc(P, sup, k)
            n += 1
    print(f"requests {min(len(Q), len(reqs))} of Q {len(Q)} / P {len(reqs)}; position offset {off} "
          f"(Q's x == emitted token {best[1]:.3f} over {best[2]}); rows replayed {n}; T {T}; "
          f"Q ids outside P's top 20: {miss_top}/{nq}")
    a1 = S[1][0] / n
    print("| k siblings | acc at depth 1 | extra tokens a step | (Q ids outside P's top 20 at p20) | net c1 at %.2f ms a row |"
          % o["row-ms"])
    print("|---|---|---|---|---|")
    for k in S:
        ak, akh = S[k][0] / n, S[k][1] / n
        net = ""
        if o["step-ms"] and o["tps"]:
            tau, t = o["tps"], o["step-ms"]
            net = "%+.2f %%" % (100 * (((tau + ak - a1) / (t + (k - 1) * o["row-ms"])) / (tau / t) - 1))
        print(f"| {k} | {ak:.4f} | {ak - a1:+.4f} | {akh - S[1][1] / n:+.4f} | {net} |")
    if o["step-ms"] and o["tps"]:
        tau, t = o["tps"], o["step-ms"]
        more = (tau - 1 - a1) / a1   # drafts the chain accepts after an accepted x1
        print(f"break-even row price at c1: k=2 {t * (S[2][0] / n - a1) / tau:.3f} ms, "
              f"k=3 {t * (S[3][0] / n - a1) / tau / 2:.3f} ms a row (leaf siblings); "
              f"a whole sibling branch as deep as the chain (upper bound, +{more:.2f} drafts after the sibling): "
              f"k=2 adds {(S[2][0] / n - a1) * (1 + more):+.4f} tokens a step")


main()
