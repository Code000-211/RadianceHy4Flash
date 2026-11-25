#!/usr/bin/env python3
"""layer.py stamps.csv probe.json phase records [layer_index]: median exclusive us per record of one layer of the pass kind."""
import csv, json, sys, collections, subprocess, re, statistics
import demangle
FILT = demangle.path()  # c++filt locally, the ROCm SDK's llvm-cxxfilt on iks
if not FILT: sys.exit("layer.py: no c++filt or llvm-cxxfilt, neither on PATH nor next to the tree's compiler")
rows = list(csv.DictReader(open(sys.argv[1]))); R = json.load(open(sys.argv[2])); PH = sys.argv[3]; REC = int(sys.argv[4]); LI = int(sys.argv[5]) if len(sys.argv) > 5 else 6
for r in rows:
    for k in ("pass", "seq", "start_ns", "end_ns", "records", "queue"): r[k] = int(r[k] or 0)
w = [p for p in R["phases"] if p.get("name") == PH and p.get("mono_ns") and p["mono_ns"][1]]
if not w:  # a composite phase (longctx, idle, long8) has no window of its own: name what does
    sys.exit(f"layer.py: no stamped window for phase '{PH}'; {sys.argv[2]} has "
             f"{', '.join(p.get('name', '(composite)') for p in R['phases'])}")
p = w[0]; a, b = p["mono_ns"]
rows = [r for r in rows if r["records"] == REC and a <= r["start_ns"] <= b and r["end_ns"]]
names = sorted({r["kernel"] for r in rows})
dem = dict(zip(names, subprocess.run([FILT] + names, capture_output=True, text=True).stdout.split("\n")))
def short(n):
    d = re.sub(r"\(anonymous namespace\)::", "", dem[n]); d = re.sub(r"^void ", "", d)
    return d.split("(")[0][:70]
passes = collections.defaultdict(list)
for r in rows: passes[r["pass"]].append(r)
ex = collections.defaultdict(list); gap = collections.defaultdict(list); nm = {}
for v in passes.values():
    v.sort(key=lambda x: x["end_ns"]); last = None
    for x in v:
        s = max(x["start_ns"], last or 0); ex[x["seq"]].append((x["end_ns"] - s) / 1e3)
        gap[x["seq"]].append(((x["start_ns"] - last) / 1e3) if last else 0); last = max(last or 0, x["end_ns"]); nm[x["seq"]] = short(x["kernel"])
seqs = sorted(ex)
moe = [s for s in seqs if "moe_iqr" in nm[s] or "moe_gemm_q" in nm[s]]
lo, hi = moe[LI - 1], moe[LI]
tot = 0
for s in seqs:
    if lo < s <= hi:
        m = statistics.median(ex[s]); tot += m
        print(f"{s:5d} {nm[s]:70s} {m:7.1f} us   startgap {statistics.median(gap[s]):6.1f}")
print("layer total", round(tot, 1), "us; records", hi - lo)
