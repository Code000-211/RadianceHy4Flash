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


if __name__ == "__main__":
    if not shutil.which("c++filt") and not shutil.which("llvm-cxxfilt"):
        sys.exit(77)
    unittest.main()
