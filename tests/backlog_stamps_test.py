#!/usr/bin/env python3
"""The backlog's stamp readers over a synthetic run: the phases of a long-context run and what they print.

No card. `probe.py` writes a `probe.json` whose phase list holds more than stamped windows: a composite
phase (`longctx`, `long8`) records the summary the phase printed next to the waves that made it, and a gap
probe (`stallN`) records a window of no span at all. `agg.py`, `attr.py` and `layer.py` iterate that list,
so if they take the window unconditionally a run of `c1,longctx` dies on the way to the numbers the card
asked for -- which is the shape of T-03, whose whole value is one stamped long-context run. So: the
windowless entries are skipped by name, the waves are still read, and `agg.py --match` reads one named term
per step out of the pass (`qsa_select` and `idx_score`, the two terms K-25 has no served number for).

The dispatches here are made up but their exclusive time is exact: queue 1 carries the three terms of a
decode step one after another, queue 2 two MoE dispatches, so the per-step us the tools print is the sum of
the spans written below."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "scripts" / "backlog"
sys.path.insert(0, str(TOOLS))
import demangle  # noqa: E402  (the same demangler lookup the tools use)

HDR = ["pass", "seq", "queue", "kernel", "ordered", "chained", "start_ns", "end_ns", "play", "records"]
RECORDS = 1335           # the stamped pass kind: a decode tape's record count, any fixed value here
SHORT = {"qsa": 2000, "idx": 500, "mla": 5500}        # ns of exclusive time on queue 1, short context
LONG = {"qsa": 20000, "idx": 6000, "mla": 100000}     # the same three at 60K context


def step(pgid, base, t):
    """One decode step at `base` ns: the three terms in series on queue 1, two MoE dispatches on queue 2.
    `t` is the exclusive time of each term in ns, so the per-step sums the tools print are known."""
    q1 = [("r4dqsa_select_kernel", base, base + t["qsa"]),
          ("r4didx_score_kernel", base + t["qsa"], base + t["qsa"] + t["idx"]),
          ("r4dmla_attn_sel_sink_kernel", base + t["qsa"] + t["idx"], base + t["qsa"] + t["idx"] + t["mla"])]
    q2 = [("r4dmoe_iqr_kernel", base, base + 100), ("r4dmoe_iqr_kernel", base + 100, base + 200)]
    return [dict(zip(HDR, [pgid, seq, 1 if seq <= len(q1) else 2, k, 1, 0, s, e, 0, RECORDS]))
            for seq, (k, s, e) in enumerate(q1 + q2, start=1)]


def phase(name, clients, lo, hi):
    """A `probe.py` phase entry: the fields agg.py prints beside the table, and its stamped window."""
    return {"name": name, "clients": clients, "gen": 128, "prompt": [58903], "decode_steps": 127.0,
            "tok_per_step": 1.0, "agg_tps": 20.0, "sum_req_dec_tps": 20.0, "ms_step": 49.0,
            "hit_tokens": 58903, "hit": 1.0, "miss_units_per_step": 0, "ttft": 18.6,
            "mono_ns": [lo, hi], "stats": {}, "metrics": {}}


class BacklogStampsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not demangle.path():
            raise unittest.SkipTest("no c++filt or llvm-cxxfilt")
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        rows = (step(7, 1000, SHORT) + step(8, 11000, SHORT)          # the short-context c1 phase
                + step(9, 2 * 10**12, LONG) + step(10, 2 * 10**12 + 200000, LONG))
        with (d / "stamps.csv").open("w") as f:
            f.write(",".join(HDR) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in HDR) + "\n")
        # What probe.py records: two windowed phases, the composite longctx entry (a summary with the
        # waves nested in it, no window of its own) and a stall probe whose window has no span.
        waves = {"prefill": phase("longctx-pre", 8, 4 * 10**12, 5 * 10**12),
                 "decode": phase("longctx-decode", 8, 6 * 10**12, 7 * 10**12)}
        cls.json = d / "probe.json"
        cls.json.write_text(json.dumps({"port": 30001, "phases": [
            phase("c1", 1, 0, 10 ** 12), phase("lctx1-decode", 1, 2 * 10**12, 3 * 10**12),
            dict(waves, **{"chars": 190000}),
            {"name": "stall1", "clients": 1, "typical_gap_ms": 4.0, "worst_gaps_ms": [9.0],
             "metrics": {}, "mono_ns": [0, 0]}]}))
        cls.csv = str(d / "stamps.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def runtool(self, name, *args):
        return subprocess.run([sys.executable, str(TOOLS / name), self.csv, str(self.json), *args],
                              capture_output=True, text=True)

    def test_readers_survive_a_windowless_phase(self):
        for name in ("agg.py", "attr.py"):
            r = self.runtool(name, *(("--top", "40") if name == "agg.py" else ()))
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr}")
        r = self.runtool("agg.py", "--top", "40")
        # the two windowless entries are named as skipped, not fatal, and say where their numbers are
        self.assertEqual(r.stdout.count("no stamped window, skipped"), 2, r.stdout)
        self.assertIn("stall1: no stamped window", r.stdout)
        self.assertIn("its waves are their own phases in the JSON", r.stdout)
        self.assertIn("==== phase c1", r.stdout)
        self.assertIn("==== phase lctx1-decode", r.stdout)
        # and the read is per windowed phase: the exclusive time of each term is its span, per step
        self.assertIn("r4dqsa_select_kernel", r.stdout)
        self.assertIn("r4dmla_attn_sel_sink_kernel", r.stdout)
        agg = json.loads(Path(self.csv.replace(".csv", ".agg.json")).read_text())
        self.assertEqual(sorted(agg), ["c1/1335", "lctx1-decode/1335"])
        self.assertAlmostEqual(agg["c1/1335"]["k"]["r4dqsa_select_kernel"][1], 2.0, places=6)
        self.assertAlmostEqual(agg["lctx1-decode/1335"]["k"]["r4dqsa_select_kernel"][1], 20.0, places=6)

    def test_match_reads_one_named_term_per_step(self):
        r = self.runtool("agg.py", "--match", "qsa_select|idx_score")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("terms /qsa_select|idx_score/: 2 kernels, 2.0 launches 2.5 us a step", r.stdout)
        self.assertIn("terms /qsa_select|idx_score/: 2 kernels, 2.0 launches 26.0 us a step", r.stdout)
        # only the matched kernels are listed, and the total is their sum and no one else's
        self.assertNotIn("mla_attn_sel_sink", r.stdout)
        self.assertEqual(r.stdout.count("terms /qsa_select|idx_score/"), 2)

    def test_layer_reads_a_windowed_phase_by_name(self):
        r = self.runtool("layer.py", "c1", str(RECORDS), "1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("moe_iqr", r.stdout)

    def test_layer_names_the_phases_a_run_actually_has(self):
        r = self.runtool("layer.py", "longctx", str(RECORDS), "1")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no stamped window for phase 'longctx'", r.stderr + r.stdout)
        self.assertIn("c1", r.stderr + r.stdout)
        self.assertIn("stall1", r.stderr + r.stdout)


class ProbePhasesTest(unittest.TestCase):
    """The phase the card runs is dispatched and sized from the lane's environment, and it is written
    where the next card will look for it: probe.py's own header and its error line."""

    @classmethod
    def setUpClass(cls):
        cls.src = (TOOLS / "probe.py").read_text()

    def test_lctx_is_a_phase_of_its_own(self):
        self.assertIn("lctxN", self.src.split('"""')[1])           # the header lists the phase
        self.assertIn("lctxN", self.src.split("no phase")[-1])     # and so does the unknown-phase error
        self.assertIn('ph.startswith("lctx")', self.src)
        self.assertIn('RAD_LCTX_CHARS', self.src)
        self.assertIn('RAD_LCTX_NTOK', self.src)

    def test_a_composite_phase_records_its_waves(self):
        self.assertIn("def record(", self.src)
        for ph in ("longctx()", "long8(", "idle(", "resume("):
            self.assertIn(f"record({ph}", self.src, f"{ph} is recorded through record()")

    def test_record_runs_and_lifts_the_waves(self):
        """record() itself, run on dicts: a composite phase keeps its summary and its stamped waves
        follow as phases of their own; a plain phase and a wave without stamps add nothing."""
        a = self.src.index("R = {")
        b = self.src.index('if "nowarm"')
        ns = {}
        exec(self.src[a:b], ns)
        pre = {"name": "longctx-pre", "mono_ns": [1, 2], "decode_steps": 0}
        dec = {"name": "longctx-decode", "mono_ns": [3, 4], "decode_steps": 5}
        bare = {"name": "x", "decode_steps": 1}
        ns["record"]({"name": "longctx", "prefill": pre, "decode": dec, "other": bare})
        names = [o["name"] for o in ns["R"]["phases"]]
        self.assertEqual(names, ["longctx", "longctx-pre", "longctx-decode"])
        ns["record"]({"name": "plain", "mono_ns": [1], "decode_steps": 2})
        self.assertEqual(ns["R"]["phases"][-1]["name"], "plain")
        self.assertEqual(len(ns["R"]["phases"]), 4)


class _FakeServer:
    """A stand-in for a served radiance: /health, /metrics and a streamed /v1/chat/completions that
    answers `ntok` tokens. It takes `die_after` completions and cuts every request after them, /health
    included, which is how an engine that stops in the middle of a wave looks from the probe."""

    def __init__(self, die_after=None):
        import http.server, threading
        fake = self
        self.taken, self.steps, self.die_after, self.lock = 0, 0, die_after, threading.Lock()
        self.prompts = []

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                with fake.lock:
                    dead = fake.die_after is not None and fake.taken >= fake.die_after
                if dead:
                    self.connection.close(); return
                body = b"ok"
                if self.path == "/metrics":
                    body = f"radiance:engine_steps_decode_total{{model=\"m\"}} {fake.steps}\n".encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                n = req["max_tokens"]
                with fake.lock:
                    fake.taken += 1
                    dead = fake.die_after is not None and fake.taken > fake.die_after
                    if not dead:
                        fake.prompts.append(req["messages"][0]["content"]); fake.steps += n
                if dead:
                    self.connection.close(); return
                self.send_response(200); self.end_headers()
                for _ in range(n):
                    self.wfile.write(b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n')
                use = {"usage": {"completion_tokens": n, "prompt_tokens": 7}, "choices": []}
                self.wfile.write(b"data: " + json.dumps(use).encode() + b"\n\ndata: [DONE]\n\n")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()


class StressPhaseTest(unittest.TestCase):
    """`stress` (issue #251): waves of concurrent requests whose failures are counted, not raised, so
    a run against an engine that stops says how far it got and that the server is gone."""

    def run_stress(self, fake, waves):
        import os
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "probe.json"
            env = dict(os.environ, RAD_STRESS_WAVES=str(waves), RAD_STRESS_CHARS="300", RAD_STRESS_NTOK="3")
            r = subprocess.run([sys.executable, str(TOOLS / "probe.py"), str(fake.port), str(out), "nowarm,stress"],
                               capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            return json.loads(out.read_text())["phases"][-1]

    def test_every_request_of_every_wave_is_counted(self):
        fake = _FakeServer()
        try:
            o = self.run_stress(fake, 4)
        finally:
            fake.close()
        self.assertEqual((o["name"], o["waves"], o["of"], o["clients"]), ("stress", 4, 4, 8))
        self.assertEqual((o["ok"], o["err"], o["server_dead"]), (32, 0, False))
        self.assertEqual(o["steps"], {"engine_steps_decode_total": 96.0})
        self.assertEqual([w["ok"] for w in o["per_wave"]], [8, 8, 8, 8])
        # A wave's prompts are fresh, and the wave after it sends the same eight again.
        p = fake.prompts
        self.assertEqual(sorted(p[0:8]), sorted(p[8:16]))
        self.assertEqual(sorted(p[16:24]), sorted(p[24:32]))
        self.assertEqual(len(set(p)), 16)

    def test_a_server_that_stops_ends_the_phase_with_its_count(self):
        fake = _FakeServer(die_after=20)       # it stops four requests into the third wave
        try:
            o = self.run_stress(fake, 6)
        finally:
            fake.close()
        self.assertTrue(o["server_dead"])
        self.assertEqual((o["ok"], o["err"]), (20, 4))
        self.assertEqual((o["waves"], o["of"]), (3, 6))   # and no wave is sent after it
        self.assertTrue(o["per_wave"][-1]["errors"])

    def test_stress_is_listed(self):
        src = (TOOLS / "probe.py").read_text()
        self.assertIn("stress", src.split('"""')[1])
        self.assertIn("stress", src.split("no phase")[-1])


if __name__ == "__main__":
    if not shutil.which("c++filt") and not shutil.which("llvm-cxxfilt"):
        sys.exit(77)
    unittest.main()
