#!/usr/bin/env python3
"""The Hy4 fixture recipe's clones: what a clone may resize, and what it must carry over (T-04).

`m_clone` copies a recorded case to another M, the shape D-10 asks the fixture for: every trunk row
at M 24, 40 and 48. It resizes the operands that are M rows tall and nothing else, so the weights,
and the encoding the container recorded with them (affine `fwht128` 8-bit codes and their scales),
are still the container's. A clone that drops the encoding offers a quantised row plain weights it
never receives, and the device then answers something other than the reference: T-04's first
recording did exactly that, and the card replay said so -- `gemm_nt_q` i8a8h at M 24/40/48 against
`gemm_i8a8_nt_m16` at rel_l2 1.4, and `logits_gemm` bf16 at M 24/40/48 against `logits_gemm_i8`
refusing "a dtype this kernel does not serve", because the row that serves the encoded geometry is
no longer the row the case matches.

`fresh` keeps drawing plain, and that is right for `heads_clone` and `wide_clone`: they resize the
weight itself, so the container's encoding no longer describes the operand it was recorded for.

No card. The source case is read out of the committed fixture, so the test speaks about the
recording the recipe actually produces."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "kernels_hy4.rkb"
_spec = importlib.util.spec_from_file_location("kernels_hy4", ROOT / "data" / "kernels_hy4.py")
recipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recipe)

M2 = 24
SALT = 0x51D00004          # any fixed salt: a clone draws its data fresh either way


def container_case():
    """A recorded case whose weight carries an encoding and whose activations are M rows tall."""
    if not FIXTURE.exists():
        raise unittest.SkipTest("data/kernels_hy4.rkb is not in this tree")
    for c in recipe.load(str(FIXTURE))["cases"]:
        if (c["op"] == "gemm_nt_q" and c["m"] == 8 and c["band"] == "M <= 8"
                and recipe.par(c, "dtype") == "i8a8h" and any(o["has_enc"] for o in c["opd"])):
            return c
    raise AssertionError("the fixture lost its recorded gemm_nt_q i8a8h case at M 8")


def a_heads_case():
    """A recorded case of a row `heads_clone` widens, with an encoded operand to lose."""
    if not FIXTURE.exists():
        raise unittest.SkipTest("data/kernels_hy4.rkb is not in this tree")
    for c in recipe.load(str(FIXTURE))["cases"]:
        if (c["op"] in recipe.HEADS and any(o["has_enc"] for o in c["opd"])
                and " heads " not in c["band"] and recipe.par(c, "heads")):
            return c
    raise AssertionError("the fixture lost its encoded case of a heads row")


class MClone(unittest.TestCase):
    def test_a_clone_at_another_m_keeps_the_encoding_the_container_recorded(self):
        c = container_case()
        d = recipe.m_clone(c, M2, SALT)
        self.assertEqual(d["m"], M2)
        self.assertEqual([o["has_enc"] for o in d["opd"]], [o["has_enc"] for o in c["opd"]])
        self.assertEqual([o.get("sel") for o in d["opd"]], [o.get("sel") for o in c["opd"]])
        self.assertEqual([o.get("enc") for o in d["opd"]], [o.get("enc") for o in c["opd"]])

    def test_a_clone_resizes_the_rows_that_are_m_tall_and_nothing_else(self):
        c = container_case()
        d = recipe.m_clone(c, M2, SALT)
        self.assertEqual(len(d["opd"]), len(c["opd"]))
        for o, s in zip(d["opd"], c["opd"]):
            m_tall = bool(s["shape"]) and s["shape"][0] == 8 and not (s["flags"] & 4) and s["role"] != 3
            want = [M2] + s["shape"][1:] if m_tall else s["shape"]
            self.assertEqual(o["shape"], want, f"role {s['role']} dtype {s['dtype']}")
            self.assertEqual(o["dtype"], s["dtype"])
            self.assertEqual(o["role"], s["role"])
            self.assertEqual(o["flags"], s["flags"])
        self.assertEqual(recipe.par(d, "M"), M2)
        self.assertEqual(d["band"], f"{c['band']} M {M2}")

    def test_a_clone_carries_no_reference_of_its_own(self):
        c = container_case()
        d = recipe.m_clone(c, M2, SALT)
        for o in d["opd"]:
            self.assertEqual(o["have_ref"], 0)
            self.assertNotIn("ref", o)
        self.assertNotEqual(d["seed"], c["seed"])
        self.assertNotEqual(d["opd"][0]["shape"], c["opd"][0]["shape"])

    def test_the_clone_leaves_the_case_it_came_from_as_it_was(self):
        c = container_case()
        before = (c["m"], c["band"], [list(o["shape"]) for o in c["opd"]])
        recipe.m_clone(c, M2, SALT)
        self.assertEqual((c["m"], c["band"], [list(o["shape"]) for o in c["opd"]]), before)


def a_wide_case():
    """A recorded case of a row `wide_clone` widens, with an encoded operand to lose."""
    if not FIXTURE.exists():
        raise unittest.SkipTest("data/kernels_hy4.rkb is not in this tree")
    for c in recipe.load(str(FIXTURE))["cases"]:
        if c["op"] in recipe.WIDE and recipe.par(c, "n") and any(o["has_enc"] for o in c["opd"]):
            return c
    raise AssertionError("the fixture lost its encoded case of a wide row")


class OtherClones(unittest.TestCase):
    def test_the_heads_clone_still_draws_plain_weights(self):
        c = a_heads_case()
        d = recipe.heads_clone(c, recipe.par(c, "heads") + 16, SALT)
        self.assertFalse(any(o["has_enc"] for o in d["opd"]))
        self.assertFalse(any("sel" in o or "enc" in o for o in d["opd"]))

    def test_the_wide_clone_still_draws_plain_weights(self):
        c = a_wide_case()
        d = recipe.wide_clone(c, recipe.par(c, "n") + 1024, SALT)
        self.assertFalse(any(o["has_enc"] for o in d["opd"]))
        self.assertFalse(any("sel" in o or "enc" in o for o in d["opd"]))
        self.assertEqual(recipe.par(d, "n"), recipe.par(c, "n") + 1024)


def trunk_cases(op, M):
    """The committed fixture's cases of `op` at M, one per recorded point."""
    if not FIXTURE.exists():
        raise unittest.SkipTest("data/kernels_hy4.rkb is not in this tree")
    return [c for c in recipe.load(str(FIXTURE))["cases"] if c["op"] == op and c["m"] == M]


def shapes(c):
    return [o["shape"] for o in c["opd"]]


def point(c):
    """A case's geometry and parameters, the part of it a clone decides -- its data and reference are
    drawn afresh, so they are not part of the point."""
    return (c["op"], c["m"], c["band"], repr(c["params"]),
            repr([(o["role"], o["dtype"], o["shape"], o["idx_max"], o["idx_const"], o["flags"],
                   o["absent"]) for o in c["opd"]]))


class VerifyHeights(unittest.TestCase):
    """D-10's M 24 / 40 / 48 rows: a clone is that row at that height, so every operand sits at the
    extent a launch at that height has. `m_clone` moves the M rows and leaves the rest, which is right
    only for a row whose every operand is M tall: `kv_store`'s cache and `qsa_work`'s work list are
    sized by the height through the derivation their reference makes, and the state, block table and
    count of `qsa_work` by the sequences the batch holds."""

    HEIGHTS = (24, 40, 48)

    def test_the_kv_store_clone_fills_a_cache_the_height_needs(self):
        cases = trunk_cases("kv_store", 8)
        self.assertTrue(cases, "the fixture lost its kv_store case at M 8")
        for c in cases:
            bs = recipe.par(c, "block_size")
            for M in self.HEIGHTS:
                d = recipe.t04_clone(c, M, SALT)
                nb = (M + bs - 1) // bs + 2      # ref_kv_store_shape: ceil(M / block_size) + 2 blocks
                (cache,) = [o for o in d["opd"] if len(o["shape"]) == 4]
                (slots,) = [o for o in d["opd"] if len(o["shape"]) == 1]
                self.assertEqual(cache["shape"][0], nb, f"block size {bs} M {M}")
                self.assertEqual(slots["idx_max"], nb * bs, f"block size {bs} M {M}")
                # the key and value rows are the batch's, and nothing else moved
                self.assertIn([M, recipe.par(c, "n_head_kv"), recipe.par(c, "head_dim")], shapes(d))
                self.assertEqual([o["dtype"] for o in d["opd"]], [o["dtype"] for o in c["opd"]])

    def test_the_committed_fixture_stores_into_a_cache_at_the_height(self):
        for M in self.HEIGHTS:
            cases = trunk_cases("kv_store", M)
            self.assertEqual(len(cases), 2, f"the fixture lost its kv_store cases at M {M}")
            for c in cases:
                bs = recipe.par(c, "block_size")
                nb = (M + bs - 1) // bs + 2
                (cache,) = [o for o in c["opd"] if len(o["shape"]) == 4]
                (slots,) = [o for o in c["opd"] if len(o["shape"]) == 1]
                self.assertEqual(cache["shape"][0], nb, f"block size {bs} M {M}")
                self.assertEqual(slots["idx_max"], nb * bs, f"block size {bs} M {M}")

    def qsa_work(self, M):
        c = trunk_cases("qsa_work", M)[0]
        r, seqs = recipe.par(c, "ratio"), recipe.par(c, "seqs")
        work = (M + r - 1) // r + seqs     # hyv4flash_common.h: work = ceil(T / ratio) + max_seqs
        S = min(seqs, M)                   # ref_qsa_work_shape: one row a sequence
        return c, work, S

    def test_the_qsa_work_clone_keeps_a_sequence_a_row_and_sizes_its_list_by_the_height(self):
        cases = trunk_cases("qsa_work", 8)
        self.assertTrue(cases, "the fixture lost its qsa_work case at M 8")
        (c,) = cases
        r, seqs = recipe.par(c, "ratio"), recipe.par(c, "seqs")
        n, ring = recipe.par(c, "n"), recipe.par(c, "ring")
        for M in self.HEIGHTS:
            d = recipe.t04_clone(c, M, SALT)
            work, S = (M + r - 1) // r + seqs, min(seqs, M)
            self.assertEqual(recipe.par(d, "work"), work, f"M {M}")
            self.assertEqual(recipe.par(d, "M"), M)
            self.assertEqual(shapes(d), [[M, n], [M], [S + 1], [S, 1, ring, n], [S, 1], [S, 2], [],
                                         [work * r, n], [work], [work], [S]], f"M {M}")
            self.assertEqual([o["idx_max"] for o in d["opd"]], [0, 0, 0, 0, S, 1 << 16, 0, 0, 0, 0, 0],
                             f"M {M}")
            self.assertEqual([o["role"] for o in d["opd"]], [o["role"] for o in c["opd"]])

    def test_the_committed_fixture_holds_the_qsa_work_rows_at_the_height(self):
        for M in self.HEIGHTS:
            c, work, S = self.qsa_work(M)
            self.assertEqual(recipe.par(c, "work"), work, f"M {M}")
            self.assertEqual(shapes(c)[:7], [[M, recipe.par(c, "n")], [M], [S + 1],
                                             [S, 1, recipe.par(c, "ring"), recipe.par(c, "n")],
                                             [S, 1], [S, 2], []], f"M {M}")
            self.assertEqual(shapes(c)[7:], [[work * recipe.par(c, "ratio"), recipe.par(c, "n")],
                                             [work], [work], [S]], f"M {M}")

    def test_the_committed_verify_heights_are_what_the_recipe_clones(self):
        if not FIXTURE.exists():
            raise unittest.SkipTest("data/kernels_hy4.rkb is not in this tree")
        cases = recipe.load(str(FIXTURE))["cases"]
        clones = [c for c in cases if c["m"] in self.HEIGHTS]
        self.assertTrue(clones, "the fixture lost D-10's verify heights")
        for c in clones:
            band, M = c["band"].rsplit(" M ", 1)
            src = [s for s in cases if s["op"] == c["op"] and s["m"] == 8 and s["band"] == band]
            self.assertTrue(src, f"no M 8 case under '{c['band']}'")
            self.assertIn(point(c), {point(recipe.t04_clone(s, int(M), SALT)) for s in src}, c["band"])


class GemmQA(unittest.TestCase):
    """`gemm_nt_q` at q_a's (N 1024, K 3072), M 2048 -- the Verify pair K-51 and audit F7.1 name."""

    def base(self):
        for c in recipe.load(str(FIXTURE))["cases"]:
            if (c["op"] == "gemm_nt_q" and c["m"] == 512 and recipe.par(c, "N") == 3072
                    and recipe.par(c, "K") == 1536):
                return c
        raise AssertionError("the fixture lost its N 3072 K 1536 gemm_nt_q case at M 512")

    def test_the_clone_is_q_a_s_shape_and_the_others_are_left_alone(self):
        base = self.base()
        (d,) = recipe.t04e_gemm_q_a(str(FIXTURE))
        self.assertEqual(d["op"], "gemm_nt_q")
        self.assertEqual(d["m"], 2048)
        self.assertEqual((recipe.par(d, "M"), recipe.par(d, "N"), recipe.par(d, "K")), (2048, 1024, 3072))
        self.assertNotEqual(d["seed"], base["seed"])
        self.assertIn("q_a", d["band"])
        self.assertEqual(len(d["opd"]), len(base["opd"]))
        encs = [o["has_enc"] for o in d["opd"]]
        self.assertEqual(encs, [o["has_enc"] for o in base["opd"]])
        self.assertTrue(any(encs))
        for o, b in zip(d["opd"], base["opd"]):
            self.assertEqual(o.get("enc"), b.get("enc"))
            self.assertEqual(o["dtype"], b["dtype"])
            self.assertEqual(o["role"], b["role"])
            self.assertEqual(o["idx_max"], b["idx_max"])
        shapes = [o["shape"] for o in d["opd"] if o["shape"]]
        self.assertIn([2048, 3072], shapes)                 # the activation
        self.assertIn([2048, 24], shapes)                   # its group scales
        self.assertIn([1024, 3072], shapes)                 # the weight
        self.assertIn([1024, 24], shapes)                   # the weight's group scales
        self.assertIn([2048, 1024], shapes)                 # the output
        # a kernel lays the selected planes out from their recorded extents: they move with the weight
        sels = [o["sel"] for o in d["opd"] if o["has_enc"]]
        self.assertEqual(sels, [[(0, 1024, 3072)], [(1, 1024, 24)]])

    def test_the_committed_fixture_holds_a_case_at_q_a_s_shape(self):
        if not FIXTURE.exists():
            raise unittest.SkipTest("data/kernels_hy4.rkb is not in this tree")
        hits = [c for c in recipe.load(str(FIXTURE))["cases"]
                if (c["op"] == "gemm_nt_q" and c["m"] == 2048 and recipe.par(c, "N") == 1024
                    and recipe.par(c, "K") == 3072)]
        self.assertEqual(len(hits), 1, "the fixture lost the q_a-shaped gemm_nt_q case")
        self.assertTrue(any(o["has_enc"] for o in hits[0]["opd"]))


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
