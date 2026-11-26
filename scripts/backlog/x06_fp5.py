#!/usr/bin/env python3
"""x06_fp5.py <port> <out.json> [phases]: X-06 (FP-5) workload, the global suffix cache's served A/B.

Run it once per server, against an arm with --draft-suffix-global-mib at its default (256) and one with 0.
Every phase records each request's decode tok/s, TTFT, the sha256 of its text, and the /metrics delta
over the phase (draft / suffix counters). Prompts come from prompts.py, so both arms send the same tokens.

phases (default all, in this order):
  rg    greedy retries, c1: 4 prompts, each asked 3 times at T 0 (256 tokens)
  rs    sampled retries, c1: 4 prompts x 3 at T 0.9, a new seed each time (256 tokens)
  r8    retries at c8: 8 clients, each asks its own prompt 3 times in a row at T 0.9 (192 tokens)
  c1    control, c1: 8 prompts never asked before at T 0.9 (256 tokens)
  c8    control, c8: 16 prompts never asked before at T 0.9, two waves of 8 (192 tokens)
  t64   a 64K conversation: turn 1 (~58K tokens, 8 out), turn 2 = turn 1 + answer + a question (64 out)
"""
import hashlib, json, os, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import CODE_PROMPTS, long_prompt  # noqa: E402

PORT, OUT = int(sys.argv[1]), sys.argv[2]
PH = sys.argv[3].split(",") if len(sys.argv) > 3 else ["rg", "rs", "r8", "c1", "c8", "t64"]
BASE = f"http://127.0.0.1:{PORT}"
KEYS = ("draft_tokens_total", "draft_accepted_total", "suffix_steps_total", "suffix_rows_total",
        "suffix_accepted_total", "suffix_global_total", "suffix_global_busy_total",
        "suffix_global_credited_total", "suffix_adopted_total", "decode_tokens_total",
        "engine_steps_decode_total", "engine_step_seconds_decode_total", "decode_rows_unspeculated_total",
        "draft_reached_pos1_total")


def metrics():
    m = {}
    for ln in urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        k, _, v = ln.rpartition(" ")
        n = k.split("{")[0]
        if n == "radiance:draft_position_reached_total":
            n = "radiance:draft_reached_pos1_total" if 'position="1"' in k else ""
        if n.startswith("radiance:"):
            try:
                m[n[9:]] = m.get(n[9:], 0) + float(v)
            except ValueError:
                pass
    return m


def request(messages, n, temp, seed):
    body = {"model": "hy4-flash", "messages": messages, "max_tokens": n, "temperature": temp, "top_p": 1.0,
            "seed": seed, "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True},
            "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; gen = pt = 0; text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            ln = raw.decode(errors="replace").strip()
            if not ln.startswith("data:"):
                continue
            d = ln[5:].strip()
            if d == "[DONE]":
                break
            ev = json.loads(d)
            if ev.get("usage"):
                gen = ev["usage"].get("completion_tokens", 0); pt = ev["usage"].get("prompt_tokens", 0)
            for ch in ev.get("choices", []):
                c = (ch.get("delta", {}).get("content") or "") + (ch.get("delta", {}).get("reasoning_content") or "")
                if c:
                    text.append(c)
                    if tf is None:
                        tf = time.time()
    t1 = time.time(); s = "".join(text)
    return {"gen": gen, "prompt": pt, "ttft": round((tf or t1) - t0, 4),
            "dec": round((gen - 1) / (t1 - tf), 3) if tf and gen > 1 else 0.0,
            "sha": hashlib.sha256(s.encode()).hexdigest()[:12], "temp": temp, "text": s}


def user(c):
    return [{"role": "user", "content": c}]


def parallel(jobs):
    out = [None] * len(jobs)

    def w(i):
        out[i] = [request(*a) for a in jobs[i]]
    th = [threading.Thread(target=w, args=(i,)) for i in range(len(jobs))]
    for t in th: t.start()
    for t in th: t.join()
    return out


def phase(name):
    if name == "rg":
        return [[request(user(CODE_PROMPTS[p]), 256, 0.0, 7) for _ in range(3)] for p in range(4)]
    if name == "rs":
        return [[request(user(CODE_PROMPTS[p]), 256, 0.9, 100 * p + k) for k in range(3)] for p in range(4, 8)]
    if name == "r8":
        return parallel([[(user(CODE_PROMPTS[i] + f"\n\n(session r8-{i})"), 192, 0.9, 500 + 10 * i + k)
                          for k in range(3)] for i in range(8)])
    if name == "c1":
        ps = [CODE_PROMPTS[p] for p in range(8, 12)] + [long_prompt(2500, 900 + i) for i in range(4)]
        return [[request(user(p), 256, 0.9, 1000 + i)] for i, p in enumerate(ps)]
    if name == "c8":
        out = []
        for w in range(2):
            out += parallel([[(user(long_prompt(3000, 1100 + 8 * w + i)), 192, 0.9, 1200 + 8 * w + i)]
                             for i in range(8)])
        return out
    if name == "t64":
        p = long_prompt(170000, 64)
        a = request(user(p), 8, 0.0, 64)
        b = request(user(p) + [{"role": "assistant", "content": a["text"]},
                               {"role": "user", "content": "Now list three concrete bugs with file and line."}],
                    64, 0.0, 65)
        return [[a, b]]
    raise SystemExit(f"unknown phase {name}")


res = {}
for ph in PH:
    m0 = metrics(); t0 = time.time()
    reqs = phase(ph)
    m1 = metrics()
    md = {k: m1.get(k, 0) - m0.get(k, 0) for k in KEYS}
    for g in reqs:
        for r in g:
            r.pop("text", None)
    res[ph] = {"wall": round(time.time() - t0, 2), "metrics": md, "reqs": reqs}
    decs = sorted(r["dec"] for g in reqs for r in g)
    print(f"{ph} wall {res[ph]['wall']} dec_med {decs[len(decs) // 2]} "
          + " ".join(f"{k[:-6]}={int(v)}" for k, v in md.items() if v), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
