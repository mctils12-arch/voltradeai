"""
Regression tests for scripts/design_token_drift.py — the design_token_drift
counter (MASTER PROGRAM T8.1 / D12), extracted this session out of
scripts/program_status.sh's inline heredoc so its own logic has direct
coverage instead of only ever running embedded and unverified (same gap
class gate2_stats.py/test_gate2_stats.py closed for the gate-2 statistics).

Two things are pinned: (1) the real DESIGN.md/client/src/index.css pair
drifts 0 today, matching ci/counter_baseline.txt's pin; (2) the detector's
actual, INTENTIONALLY one-directional semantics — a doc-vs-css mismatch or
a documented-but-undefined token counts, a css-only undocumented token does
not. That asymmetry was previously only a claim in a comment ("a mismatch
in either direction is drift") that the code beneath it never implemented;
this locks in the real behavior so a future "fix" can't silently make the
counter bidirectional and turn ~72 legitimate shadcn/ui internal tokens
(verified present in index.css, none of them a canonical author-facing
token DESIGN.md's own rule 5 asks page authors to use) into false-positive
drift.
"""
import os
import tempfile
import unittest

from scripts.design_token_drift import compute_drift


class TestRealRepoPair(unittest.TestCase):
    def test_live_design_md_and_index_css_have_zero_drift(self):
        result = compute_drift("client/src/index.css", "DESIGN.md")
        self.assertEqual(
            result["drift"], 0,
            f"design_token_drift regressed: missing={result['missing']} "
            f"mismatched={result['mismatched']}",
        )


class TestSyntheticSemantics(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.css_path = os.path.join(self.tmpdir, "index.css")
        self.md_path = os.path.join(self.tmpdir, "DESIGN.md")

    def write(self, css_root_body: str, md_table_rows: str):
        with open(self.css_path, "w") as f:
            f.write(f":root {{\n{css_root_body}\n}}\n")
        with open(self.md_path, "w") as f:
            f.write("| Token | Value | Use |\n|---|---|---|\n" + md_table_rows)

    def drift(self):
        return compute_drift(self.css_path, self.md_path)

    def test_matching_pair_is_clean(self):
        self.write(
            "  --bg-primary: #050a13;\n",
            "| `--bg-primary` | `#050a13` | bg |\n",
        )
        result = self.drift()
        self.assertEqual(result["drift"], 0)

    def test_documented_token_missing_from_css_counts(self):
        self.write(
            "  --bg-primary: #050a13;\n",
            "| `--bg-primary` | `#050a13` | bg |\n"
            "| `--ghost-token` | `#000000` | never defined |\n",
        )
        result = self.drift()
        self.assertEqual(result["drift"], 1)
        self.assertEqual(result["missing"], ["--ghost-token"])
        self.assertEqual(result["mismatched"], [])

    def test_documented_token_with_wrong_value_counts(self):
        self.write(
            "  --bg-primary: #111111;\n",
            "| `--bg-primary` | `#050a13` | bg |\n",
        )
        result = self.drift()
        self.assertEqual(result["drift"], 1)
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["mismatched"], [("--bg-primary", "#050a13", "#111111")])

    def test_css_only_undocumented_token_is_not_drift(self):
        self.write(
            "  --bg-primary: #050a13;\n"
            "  --shadcn-internal-plumbing: 213 53% 5%;\n",
            "| `--bg-primary` | `#050a13` | bg |\n",
        )
        result = self.drift()
        self.assertEqual(
            result["drift"], 0,
            "a css-only token with no DESIGN.md row must not count as drift",
        )


if __name__ == "__main__":
    unittest.main()
