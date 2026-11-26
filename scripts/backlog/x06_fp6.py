#!/usr/bin/env python3
"""x06_fp6.py <port> <out.jsonl> <temp> <n> [ntok] [effort]: X-06 (FP-6) data run, the target law P per output token.

Sends n sampled code requests one at a time (c1) with logprobs and top_logprobs 20 and writes one JSON line per
request: prompt tokens, and per output token its id and the top 20 (id, logprob) of the raw distribution.
Run it against a server started with RADIANCE_DEBUG_FP6_LOG=<file> (a local patch, not part of the PR), which logs
draft 1's coupled law Q_1 per verify; x06_fp6_replay.py joins the two by request order and output position.
"""
import json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import CODE_PROMPTS  # noqa: E402

PORT, OUT, TEMP, N = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
NTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 256
EFFORT = sys.argv[6] if len(sys.argv) > 6 else "none"
with open(OUT, "a") as f:
    for i in range(N):
        body = {"model": "hy4-flash", "messages": [{"role": "user", "content": CODE_PROMPTS[i % len(CODE_PROMPTS)]}],
                "max_tokens": NTOK, "temperature": TEMP, "top_p": 1.0, "seed": 7000 + i, "reasoning_effort": EFFORT,
                "ignore_eos": True, "logprobs": True, "top_logprobs": 20}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.time()
        o = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        ch = o["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        toks = [[e["id"], [[a["id"], a["logprob"]] for a in e["top_logprobs"]]] for e in lp]
        f.write(json.dumps({"i": i, "temp": TEMP, "effort": EFFORT, "prompt": o["usage"]["prompt_tokens"],
                            "gen": o["usage"]["completion_tokens"], "wall": round(time.time() - t0, 3),
                            "toks": toks}) + "\n")
        f.flush()
        print(f"{i} gen {o['usage']['completion_tokens']} lp {len(toks)} {time.time() - t0:.1f}s", flush=True)
