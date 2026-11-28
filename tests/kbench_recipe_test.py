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


class OtherClones(unittest.TestCase):
    def test_the_heads_clone_still_draws_plain_weights(self):
        c = a_heads_case()
        d = recipe.heads_clone(c, recipe.par(c, "heads") + 16, SALT)
        self.assertFalse(any(o["has_enc"] for o in d["opd"]))
        self.assertFalse(any("sel" in o or "enc" in o for o in d["opd"]))


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
