#!/usr/bin/env python3
# analyze.py -- C-03's run directory -> the rows C-03.md quotes.
#
#   python3 analyze.py <run-dir>
#
# Reads the per-boot artefacts measure-c03.sh left:
#   <arm>-<cfg>-r<n>-<phase>-<rep>.json   probe.py's phase JSON
#   <arm>-<cfg>-r<n>.logwindow            "phase first_line last_line" (server.log line offsets)
#   <arm>-<cfg>-r<n>-boot.log             server.log (RADIANCE_LOG_STEPS=1, log: debug)
#   <arm>-<cfg>-r<n>-stamp.csv            RADIANCE_AQL_STAMP=50
# and prints, per arm/config/phase: every raw served row, and the medians per boot and over boots;
# then the `ranks` phase lines with their window and the median per boot; then runs
# scripts/stampgaps.py on every stamp CSV and prints the decode-pass block.
import glob
import json
import os
import re
import statistics
import subprocess
import sys

RUN = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
STAMPGAPS = os.path.join(REPO, "scripts", "stampgaps.py")

RANKS_RE = re.compile(r"^\S+ step (decode|other): .* ranks ([0-9.]+) save")


def boots():
    out = []
    for d in sorted(glob.glob(os.path.join(RUN, "*-boot.log"))):
        base = os.path.basename(d)[:-len("-boot.log")]
        arm, cfg, r = base.split("-")
        out.append((arm, cfg, int(r[1:]), base))
    return out


def phases_of(tag):
    got = {}
    for p in glob.glob(os.path.join(RUN, tag + "-*.json")):
        m = re.match(re.escape(tag) + r"-(c1|c8|pf20|greedy)-(\d+)\.json$", os.path.basename(p))
        if not m:
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        ph = j.get("phases") or []
        if not ph:
            continue
        x = ph[0]
        got.setdefault(m.group(1), []).append(
            (int(m.group(2)), x.get("agg_tps"), x.get("ms_step"), (x.get("ttft") or [None])[0],
             x.get("gen"), x.get("prompt"), x.get("decode_steps")))
    for k in got:
        got[k].sort()
    return got


def windows(tag):
    w = {}
    p = os.path.join(RUN, tag + ".logwindow")
    if not os.path.exists(p):
        return w
    for ln in open(p):
        f = ln.split()
        if len(f) == 3:
            w[f[0]] = (int(f[1]), int(f[2]))
    return w


print("== served rows (one line per probe.py phase; c1/c8 are three standard phases a boot)")
print("arm,cfg,round,phase,rep,agg_tps,ms_step,ttft,gen,prompt,decode_steps")
served = {}
for arm, cfg, r, tag in boots():
    got = phases_of(tag)
    for ph in ("c1", "c8", "pf20", "greedy"):
        for rep, agg, ms, ttft, gen, prompt, steps in got.get(ph, []):
            print(f"{arm},{cfg},{r},{ph},{rep},{agg},{ms},{ttft},{gen},{prompt},{steps}")
            served.setdefault((arm, cfg, ph), []).append((r, rep, agg, ms, ttft))

print()
print("== medians per boot, then over the three boots (agg tps, ms/step, TTFT s)")
for cfg in ("cpv", "nsv"):
    for ph in ("c1", "c8", "pf20"):
        for arm in ("base", "head"):
            rows = served.get((arm, cfg, ph), [])
            if not rows:
                continue
            by_boot = {}
            for r, rep, agg, ms, ttft in rows:
                by_boot.setdefault(r, []).append((agg, ms, ttft))
            per = []
            for r in sorted(by_boot):
                a = [x[0] for x in by_boot[r] if x[0] is not None]
                m = [x[1] for x in by_boot[r] if x[1] is not None]
                t = [x[2] for x in by_boot[r] if x[2] is not None]
                per.append((r, statistics.median(a) if a else None,
                            statistics.median(m) if m else None,
                            statistics.median(t) if t else None))
            fmt = lambda xs: statistics.median(xs) if xs else float("nan")
            print(f"{cfg:3} {ph:4} {arm:4} boots " +
                  "  ".join(f"r{r}:{a:.3f}/{('%.3f' % m) if m else '-'}/{('%.3f' % t) if t else '-'}"
                            for r, a, m, t in per) +
                  f"   median {fmt([x[1] for x in per]):.3f} tps, "
                  f"{fmt([x[2] for x in per if x[2] is not None]):.3f} ms/step, "
                  f"{fmt([x[3] for x in per if x[3] is not None]):.3f} s TTFT")

print()
print("== host `ranks` phase, from the 128-step line in each phase window")
print("arm,cfg,round,phase,kind,ranks_ms")
rank_rows = {}
for arm, cfg, r, tag in boots():
    w = windows(tag)
    lines = open(os.path.join(RUN, tag + "-boot.log"), errors="replace").read().splitlines()
    for ph, (a, b) in sorted(w.items()):
        for ln in lines[a:b]:
            m = RANKS_RE.match(ln)
            if not m:
                continue
            kind, v = m.group(1), float(m.group(2))
            print(f"{arm},{cfg},{r},{ph},{kind},{v}")
            rank_rows.setdefault((arm, cfg, ph), []).append((r, v))
print()
print("== ranks medians (all lines in the window, ms)")
for cfg in ("cpv", "nsv"):
    for ph in ("c1", "c8"):
        for arm in ("base", "head"):
            rows = rank_rows.get((arm, cfg, ph), [])
            if not rows:
                continue
            by_boot = {}
            for r, v in rows:
                by_boot.setdefault(r, []).append(v)
            per = [(r, statistics.median(v)) for r, v in sorted(by_boot.items())]
            print(f"{cfg:3} {ph:4} {arm:4} " + "  ".join(f"r{r}:{v:.3f}" for r, v in per) +
                  f"   median {statistics.median([v for _, v in per]):.3f}")

print()
print("== AQL stamps (scripts/stampgaps.py, verbatim)")
for arm, cfg, r, tag in boots():
    csv = os.path.join(RUN, f"{arm}-{cfg}-r{r}-stamp.csv")
    if not os.path.exists(csv) or os.path.getsize(csv) == 0:
        print(f"{arm}-{cfg}-r{r}: NO STAMP CSV")
        continue
    out = subprocess.run([sys.executable, STAMPGAPS, csv, f"{arm}-{cfg}-r{r}"],
                         capture_output=True, text=True)
    txt = out.stdout
    open(os.path.join(RUN, f"{arm}-{cfg}-r{r}-stampgaps.txt"), "w").write(txt + out.stderr)
    print(txt.strip())
    print()