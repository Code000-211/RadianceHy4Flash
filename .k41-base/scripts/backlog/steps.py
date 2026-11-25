#!/usr/bin/env python3
"""steps.py <server.log> [--tail N]: the step log (RADIANCE_LOG_STEPS=1) as the backlog reads it.

Prints a digest of every step that carries prefill rows -- `MIXED n_tok` or `prefill n_tok` -- with
the number of decode-only steps that ran before it, so one interval looks like `MIXED 2057` beside
decoding sequences with the headroom and `MIXED 2033 | MIXED 17` without it. `--tail N` keeps the
last N entries, which is what a phase list that ends with the arrival phases needs. A scratch tool
for the S-01b card.
"""
import re, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a and a.startswith("--")}
log = args[0]
tail = int(opts.get("--tail", "0"))
line = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)")
digest, dec = [], 0
for ln in open(log, errors="replace"):
    m = line.search(ln)
    if not m:
        continue
    phase, n_tok = m.group(2), int(m.group(4))
    if phase == "decode":
        dec += 1
        continue
    if dec:
        digest.append(f"{dec}dec")
        dec = 0
    digest.append(f"{phase} {n_tok}")
if tail:
    digest = digest[-tail:]
print(f"{log}: " + (" | ".join(digest) if digest else "no prefill-bearing step"))
