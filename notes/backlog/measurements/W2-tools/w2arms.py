#!/usr/bin/env python3
"""w2arms.py <runs dir> [<runs dir> ...] [--tags REGEX] [--runs] [--flaky]

Arms of DFly2 c8 runs compared at equal miss counts. A run directory holds <tag>/<tag>.json (probe.py)
and <tag>/<tag>.stamps.csv (RADIANCE_AQL_STAMP); a tag is <round>-<arm>, the arm `off` is the reference.

A run's verify passes (1191 or 1314 records, whichever it has more of) are normalised by w2budget.py's
yardstick. Runs differ in their trajectory, and so in the missing units a step (105-180 here), which moves
the routed block by more than the arms do: a line `routed block = a + b x misses a step` is fitted through
the `off` runs and every run is read as its distance from that line at its own miss count. Per arm: the
mean distance (ms a verify pass; sd over its runs), for the routed block and for the pass wall.
--runs prints a line a run with its segments.

A run whose server.log budgets a hardware queue above 100 MiB (the start-up measurement that flips from
35.4 to 175.4 MiB in one start of about nine and leaves the card 307-338 fewer expert slots) is left out
and named; --flaky keeps it.
"""
import glob, json, os, re, sys, statistics as st
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "w2budget.py")).read().split("arms = []")[0]
ns = {}; exec(src, ns)
run, SEGS = ns["run"], ns["SEGS"]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
tags = re.compile(sys.argv[sys.argv.index("--tags") + 1]) if "--tags" in sys.argv else re.compile(".")
if "--tags" in sys.argv: args.remove(sys.argv[sys.argv.index("--tags") + 1])
rows, skipped = [], []
for root in args:
    for d in sorted(glob.glob(root + "/*"), key=os.path.getmtime):
        t = os.path.basename(d)
        f, jf = f"{d}/{t}.stamps.csv", f"{d}/{t}.json"
        if "-" not in t or not tags.search(t) or not os.path.exists(f) or not os.path.exists(jf): continue
        ph = [p for p in json.load(open(jf)).get("phases", []) if p.get("name") == "c8"]
        if not ph: continue
        q = re.search(r"hardware queues.* at ([0-9.]+) MiB", open(f"{d}/server.log", errors="replace").read())
        if q and float(q.group(1)) > 100 and "--flaky" not in sys.argv:
            skipped.append(t); continue
        best = max((run(f, rec) for rec in (1191, 1314)), key=lambda r: r[2])
        if not best[2]: continue
        n = best[0]
        rows.append(dict(tag=t, arm=t.split("-", 1)[1], miss=ph[0]["miss_units_per_step"], hit=ph[0]["hit"],
                         tps=ph[0]["agg_tps"], routed=n[SEGS[8]], wall=n[SEGS[9]], router=n[SEGS[0]],
                         gu=n[SEGS[1]] + n[SEGS[2]], dn=n[SEGS[3]] + n[SEGS[4]], tail=n[SEGS[5]], passes=best[2]))
if skipped: print("left out (queue budget above 100 MiB):", " ".join(skipped))
off = [r for r in rows if r["arm"] == "off"]
def fit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b
aR, bR = fit([r["miss"] for r in off], [r["routed"] for r in off])
aW, bW = fit([r["miss"] for r in off], [r["wall"] for r in off])
for r in rows:
    r["dR"] = r["routed"] - (aR + bR * r["miss"]); r["dW"] = r["wall"] - (aW + bW * r["miss"])
print(f"off: {len(off)} runs, misses a step {min(r['miss'] for r in off):.0f}-{max(r['miss'] for r in off):.0f}; "
      f"routed block = {aR:.2f} + {bR:.4f} x misses ms (residual sd {st.stdev(r['dR'] for r in off):.2f}), "
      f"wall = {aW:.2f} + {bW:.4f} x misses (sd {st.stdev(r['dW'] for r in off):.2f})")
if "--runs" in sys.argv:
    print(f"{'run':12s} {'tok/s':>6s} {'miss':>6s} {'hit':>6s} {'passes':>6s} {'router':>6s} {'->gu end':>8s} {'->dn end':>8s} "
          f"{'tail':>5s} {'routed':>7s} {'wall':>6s} {'routed-fit':>10s} {'wall-fit':>8s}")
    for r in rows:
        print(f"{r['tag']:12s} {r['tps']:6.1f} {r['miss']:6.1f} {r['hit']:.4f} {r['passes']:6d} {r['router']:6.2f} {r['gu']:8.2f} "
              f"{r['dn']:8.2f} {r['tail']:5.2f} {r['routed']:7.2f} {r['wall']:6.2f} {r['dR']:+10.2f} {r['dW']:+8.2f}")
print(f"{'arm':10s} {'runs':>4s} {'routed block vs off':>20s} {'wall vs off':>16s} {'misses':>7s} {'router':>6s} {'->gu end':>8s} {'->dn end':>8s} {'tail':>5s}")
arms = {}
for r in rows: arms.setdefault(r["arm"], []).append(r)
for a, v in arms.items():
    sd = lambda k: f" ({st.stdev(x[k] for x in v):.2f})" if len(v) > 1 else "       "
    m = lambda k: st.mean(x[k] for x in v)
    print(f"{a:10s} {len(v):4d} {m('dR'):+13.2f}{sd('dR')} {m('dW'):+9.2f}{sd('dW')} {m('miss'):7.1f} {m('router'):6.2f} {m('gu'):8.2f} {m('dn'):8.2f} {m('tail'):5.2f}")
