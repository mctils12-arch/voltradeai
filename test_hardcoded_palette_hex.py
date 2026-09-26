"""
Regression tests for scripts/hardcoded_palette_hex.py — the
hardcoded_palette_hex counter (D13, MASTER PROGRAM §0.7 DETECT duty).

Two things are pinned: (1) the detector's synthetic semantics — a palette hex
value inside a STRING literal counts, the same value inside a COMMENT does
not, and a non-palette hex color never counts; (2) the real live tree's
current count matches ci/counter_baseline.txt's pin (402), so a future
session cannot silently add more hardcoded palette hex without this test (and
the CI ratchet) catching it.
"""
import unittest

from scripts.hardcoded_palette_hex import (
    compute,
    find_hardcoded_hex,
    palette_hex_values,
)


class TestPaletteHexValues(unittest.TestCase):
    def test_live_design_md_has_the_eleven_hex_tokens(self):
        palette = palette_hex_values("DESIGN.md")
        # 13 color tokens documented; --bg-card/--bg-card-hover are rgba(),
        # not hex, and the two font tokens have no color value at all — 11
        # hex-valued rows remain.
        self.assertEqual(len(palette), 11)
        self.assertIn("#4d9fff", palette)  # --accent
        self.assertIn("#050a13", palette)  # --bg-primary
        self.assertNotIn("#ffffff", palette)


class TestFindHardcodedHex(unittest.TestCase):
    PALETTE = ["#4d9fff", "#ff5a6e"]

    def test_hex_inside_string_literal_counts(self):
        src = 'const c = "#4d9fff";\n'
        hits = find_hardcoded_hex(src, self.PALETTE)
        self.assertEqual(hits, [(1, "#4d9fff")])

    def test_hex_inside_template_literal_counts(self):
        src = "const style = `fill:${x}#ff5a6e`;\n"
        hits = find_hardcoded_hex(src, self.PALETTE)
        self.assertEqual(hits, [(1, "#ff5a6e")])

    def test_hex_inside_line_comment_does_not_count(self):
        src = "// legacy accent was #4d9fff before the token existed\n"
        hits = find_hardcoded_hex(src, self.PALETTE)
        self.assertEqual(hits, [])

    def test_hex_inside_trailing_comment_does_not_count(self):
        src = 'const c = "#00ff00"; // not #4d9fff, deliberately different\n'
        hits = find_hardcoded_hex(src, self.PALETTE)
        self.assertEqual(hits, [])

    def test_non_palette_hex_never_counts(self):
        src = 'const c = "#123456";\n'
        hits = find_hardcoded_hex(src, self.PALETTE)
        self.assertEqual(hits, [])

    def test_case_insensitive_match_counts(self):
        src = 'const c = "#4D9FFF";\n'
        hits = find_hardcoded_hex(src, self.PALETTE)
        self.assertEqual(hits, [(1, "#4d9fff")])

    def test_multiple_hits_same_line_all_count(self):
        src = 'const grad = ["#4d9fff", "#ff5a6e"];\n'
        hits = find_hardcoded_hex(src, self.PALETTE)
        self.assertEqual(len(hits), 2)

    def test_empty_palette_finds_nothing(self):
        hits = find_hardcoded_hex('const c = "#4d9fff";\n', [])
        self.assertEqual(hits, [])


class TestLiveRepoCount(unittest.TestCase):
    def test_live_tree_matches_pinned_baseline(self):
        result = compute()
        self.assertEqual(
            result["count"], 402,
            "hardcoded_palette_hex regressed or improved — if this is a real "
            "fix, lower ci/counter_baseline.txt's pin in the same PR; if it "
            "grew, remove the new hardcoded hex and reference the CSS var "
            "instead (DESIGN.md rule 5)",
        )


if __name__ == "__main__":
    unittest.main()
