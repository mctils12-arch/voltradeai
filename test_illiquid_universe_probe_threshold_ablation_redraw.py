"""
Regression tests for scripts/illiquid_universe_probe_threshold_ablation_redraw.py
(INDEPENDENT RE-DRAW check on the STRICT_RSI mean_reversion threshold
variant). Pure/mocked tests only: no network calls, no dependency on live
backtest_v2/NASDAQ data — matches the existing test_illiquid_universe_
probe_redraw.py / test_illiquid_universe_probe_threshold_ablation.py
convention for this research-probe family.
"""
import importlib.util
import os
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "illiquid_universe_probe_threshold_ablation_redraw",
    os.path.join(os.path.dirname(__file__), "scripts",
                 "illiquid_universe_probe_threshold_ablation_redraw.py"))
redraw_abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(redraw_abl)


class TestSampleStrideIndependence(unittest.TestCase):
    def test_stride_differs_from_both_prior_draws(self):
        import illiquid_universe_probe as orig
        from illiquid_universe_probe_redraw import SAMPLE_STRIDE as REDRAW_STRIDE
        self.assertNotEqual(redraw_abl.SAMPLE_STRIDE, 40)  # 2026-07-24 original
        self.assertNotEqual(redraw_abl.SAMPLE_STRIDE, REDRAW_STRIDE)  # 2026-07-28 redraw (41)

    def test_reuses_strict_rsi_thresholds_unchanged(self):
        from illiquid_universe_probe_threshold_ablation import VARIANTS
        self.assertEqual(redraw_abl.VARIANTS["STRICT_RSI"], VARIANTS["STRICT_RSI"])


class TestDrawGroups(unittest.TestCase):
    def test_draw_groups_screens_and_flags_overlap(self):
        import illiquid_universe_probe as orig
        fake_candidates = list(orig.ILLIQUID[:1]) + ["FRESH1", "FRESH2"]

        def fake_fetch_bars(t, d):
            if t == orig.ILLIQUID[0]:
                return {"close": [1.0] * 600, "volume": [500_000] * 600}
            if t == "FRESH1":
                return {"close": [1.0] * 600, "volume": [500_000] * 600}
            return {"close": [1.0] * 600, "volume": [2_000_000] * 600}

        with patch.object(orig, "fetch_nasdaq_capital_market_candidates",
                           return_value=fake_candidates) as mock_fetch_candidates, \
             patch("backtest_v2.fetch_bars", side_effect=fake_fetch_bars):
            out = redraw_abl.draw_groups(sample_stride=99)

        mock_fetch_candidates.assert_called_once_with(sample_stride=99)
        self.assertIn(orig.ILLIQUID[0], out["overlap_with_2026_07_24_draw"])
        self.assertIn(orig.ILLIQUID[0], out["illiquid"])
        self.assertIn("FRESH1", out["illiquid"])
        self.assertIn("FRESH2", out["moderate"])

    def test_draw_groups_empty_candidates_yields_empty_buckets(self):
        import illiquid_universe_probe as orig
        with patch.object(orig, "fetch_nasdaq_capital_market_candidates", return_value=[]):
            out = redraw_abl.draw_groups(sample_stride=99)
        self.assertEqual(out["illiquid"], [])
        self.assertEqual(out["moderate"], [])
        self.assertEqual(out["overlap_with_2026_07_24_draw"], [])


class TestReusedMachineryWiring(unittest.TestCase):
    def test_run_variant_and_summarize_are_the_original_threshold_ablation_functions(self):
        import illiquid_universe_probe_threshold_ablation as orig_abl
        self.assertIs(redraw_abl.run_variant, orig_abl.run_variant)
        self.assertIs(redraw_abl.summarize_variant, orig_abl.summarize_variant)
        self.assertIs(redraw_abl.mean_reversion_spread, orig_abl.mean_reversion_spread)


if __name__ == "__main__":
    unittest.main()
