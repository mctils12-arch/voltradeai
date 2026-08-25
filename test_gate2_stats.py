"""
Regression tests for gate2_stats.py — the shared ROOT VALIDATION LADDER
gate 2 (SIGNAL) statistical core (EDGE DOCTRINE #3 consolidation of
`_newey_west_diff_test`/`find_entry_index`, previously hand-duplicated in
cot_gate2_test.py, cftc_tff_gate2_test.py, and scripts/eia930_gate2.py).

Two things are pinned here: (1) the statistics themselves, mirroring the
core cases already covered per-consumer in test_cot_gate2.py etc., and
(2) that each consumer module's `_newey_west_diff_test`/`find_entry_index`
name is a re-export of THIS module's function (identity, not just
behavioral equivalence) — so a future session re-introducing a local copy
in any consumer breaks this test immediately rather than silently
diverging.
"""
import math
import unittest

from gate2_stats import find_entry_index, newey_west_diff_test


class TestFindEntryIndex(unittest.TestCase):
    def test_first_bar_strictly_after(self):
        dates = ["2026-01-01", "2026-01-03", "2026-01-05"]
        self.assertEqual(find_entry_index(dates, "2026-01-02"), 1)

    def test_no_lookahead_on_exact_match(self):
        dates = ["2026-01-01", "2026-01-03"]
        self.assertEqual(find_entry_index(dates, "2026-01-01"), 1)

    def test_none_when_nothing_after(self):
        dates = ["2026-01-01", "2026-01-03"]
        self.assertIsNone(find_entry_index(dates, "2026-01-03"))


class TestNeweyWestDiffTest(unittest.TestCase):
    def test_beta_equals_conditional_mean_difference(self):
        rows = [
            {"bucket": "extreme_high", "forward_returns": {20: 0.05}},
            {"bucket": "extreme_high", "forward_returns": {20: 0.07}},
            {"bucket": "mid", "forward_returns": {20: 0.01}},
            {"bucket": "extreme_low", "forward_returns": {20: -0.02}},
            {"bucket": "mid", "forward_returns": {20: 0.02}},
            {"bucket": "extreme_high", "forward_returns": {20: 0.03}},
        ]
        result = newey_west_diff_test(rows, 20, "extreme_high", lag=1)
        self.assertIsNotNone(result)
        bucket_mean = (0.05 + 0.07 + 0.03) / 3
        complement_mean = (0.01 - 0.02 + 0.02) / 3
        self.assertAlmostEqual(result["mean_diff_pct"] / 100,
                                bucket_mean - complement_mean, places=3)
        self.assertEqual(result["n"], 6)
        self.assertEqual(result["lag_weeks"], 1)

    def test_too_few_observations_returns_none_not_a_fake_number(self):
        rows = [
            {"bucket": "extreme_high", "forward_returns": {20: 0.05}},
            {"bucket": "mid", "forward_returns": {20: 0.01}},
        ]
        self.assertIsNone(newey_west_diff_test(rows, 20, "extreme_high"))

    def test_degenerate_bucket_dummy_returns_none(self):
        all_bucket = [{"bucket": "extreme_high", "forward_returns": {20: 0.01 * i}}
                      for i in range(10)]
        self.assertIsNone(newey_west_diff_test(all_bucket, 20, "extreme_high"))
        none_bucket = [{"bucket": "mid", "forward_returns": {20: 0.01 * i}}
                       for i in range(10)]
        self.assertIsNone(newey_west_diff_test(none_bucket, 20, "extreme_high"))

    def test_default_lag_is_horizon_over_5(self):
        rows = [{"bucket": "extreme_high" if i % 2 else "mid",
                  "forward_returns": {20: 0.01 * i}} for i in range(30)]
        result = newey_west_diff_test(rows, 20, "extreme_high")
        self.assertEqual(result["lag_weeks"], round(20 / 5))


class TestConsolidationIdentity(unittest.TestCase):
    """Pins that every consumer's `_newey_west_diff_test`/`find_entry_index`
    IS gate2_stats' function (re-exported), not a re-derived local copy."""

    def test_cot_gate2_test_reexports(self):
        import cot_gate2_test
        self.assertIs(cot_gate2_test._newey_west_diff_test, newey_west_diff_test)
        self.assertIs(cot_gate2_test.find_entry_index, find_entry_index)

    def test_cftc_tff_gate2_test_reexports(self):
        import cftc_tff_gate2_test
        self.assertIs(cftc_tff_gate2_test._newey_west_diff_test, newey_west_diff_test)
        self.assertIs(cftc_tff_gate2_test.find_entry_index, find_entry_index)

    def test_eia930_gate2_reexports(self):
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "eia930_gate2_identity_check",
            os.path.join(os.path.dirname(__file__), "scripts", "eia930_gate2.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertIs(mod._newey_west_diff_test, newey_west_diff_test)
        self.assertIs(mod.find_entry_index, find_entry_index)


if __name__ == "__main__":
    unittest.main()
