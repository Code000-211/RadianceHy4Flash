#!/usr/bin/env python3
"""scripts/stampgaps.py over live-issued passes (T-01): the stamp file a prefill or mixed pass leaves.

No card. A live pass is written by core/device/aql.cpp (a_pass_stamp / live_stamp) in the format a
played pass uses, with its pass number from 2^40 up, play 0 and `records` the pass's dispatch count.
So the reader needs no change: passes group by that count, a pass the signal pool ran dry in (fewer
lines than its count) is left out and counted, and the gaps and the overlap between two queues come
out of the made-up times below exactly."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HDR = "pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records"
LIVE = 1 << 40


def live_pass(n, t0, drop=0):
    """Four dispatches: queue 1 runs a (0-1000 us) and, ordered, b after a 500 ns gap; queue 2 runs c
    beside a, then d ordered. `drop` removes the last lines, as an exhausted signal pool does."""
    k = [(1, "gemm_a", 0, 1000, 0), (1, "gemm_b", 1500, 2500, 1),
         (2, "attn_c", 200, 900, 0), (2, "attn_d", 1100, 1400, 1)]
    rows = [f"{LIVE + n},{i},{q},{name},{o},0,{t0 + a},{t0 + b},0,4"
            for i, (q, name, a, b, o) in enumerate(k)]
    return rows[:len(rows) - drop]


class StampgapsLiveTest(unittest.TestCase):
    def test_live_passes_read_like_played_ones(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "live.csv"
            rows = live_pass(1, 10_000) + live_pass(2, 50_000) + live_pass(3, 90_000, drop=2)
            p.write_text(HDR + "\n" + "\n".join(rows) + "\n")
            r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "stampgaps.py"), str(p)],
                               capture_output=True, text=True, check=True)
        out = r.stdout
        self.assertIn("3 stamped passes", out)
        self.assertIn("pass of 4 records", out)
        self.assertIn("passes complete / incomplete      2 / 1", out)
        self.assertIn("dispatches a pass                 4", out)
        # Wall 2500 ns; the union of the four spans is 2300 ns (holes 1000-1100 and 1400-1500), so 200 ns
        # idle, 8 %; the ordered starts wait 500 ns (queue 1) and 200 ns (queue 2), median 350 ns.
        self.assertIn("wall ms                           0.003", out)
        self.assertIn("idle ms / share                   0.000 / 8.0 %", out)
        self.assertIn("gap before an ordered start us    p50 0.35", out)


if __name__ == "__main__":
    unittest.main()
