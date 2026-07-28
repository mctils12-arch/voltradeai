"""
Regression tests for scripts/illiquid_universe_probe_redraw.py. Pure-
function tests only: fetch_bars is mocked so no network call or live
NASDAQ symbol directory dependency is needed (matches the existing
test_illiquid_universe_probe.py convention).
"""
import importlib.util
import os
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "illiquid_universe_probe_redraw",
    os.path.join(os.path.dirname(__file__), "scripts", "illiquid_universe_probe_redraw.py"))
redraw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(redraw)


def _bars(n_days, volume):
    return {"close": [1.0] * n_days, "volume": [volume] * n_days}


class TestScreenAndBucket(unittest.TestCase):
    def test_classifies_by_trailing_volume(self):
        fake = {
            "ILLIQ1": _bars(600, 500_000),      # <=1M -> illiquid
            "MOD1": _bars(600, 2_000_000),       # 1-5M -> moderate
            "LIQ1": _bars(600, 9_000_000),       # >5M -> liquid_5m_plus, excluded
        }
        with patch("backtest_v2.fetch_bars", side_effect=lambda t, d: fake[t]):
            buckets = redraw.screen_and_bucket(list(fake), target_illiquid=5, target_moderate=5)
        self.assertEqual(buckets["illiquid"], ["ILLIQ1"])
        self.assertEqual(buckets["moderate"], ["MOD1"])
        self.assertEqual(len(buckets["liquid_5m_plus"]), 1)

    def test_insufficient_history_is_screened_out(self):
        fake = {"THIN": _bars(100, 500_000)}
        with patch("backtest_v2.fetch_bars", side_effect=lambda t, d: fake[t]):
            buckets = redraw.screen_and_bucket(["THIN"], target_illiquid=5, target_moderate=5)
        self.assertEqual(buckets["illiquid"], [])
        self.assertEqual(len(buckets["screened_out"]), 1)
        self.assertIn("insufficient_history", buckets["screened_out"][0]["reason"])

    def test_fetch_error_is_screened_out_not_raised(self):
        def boom(t, d):
            raise RuntimeError("network down")
        with patch("backtest_v2.fetch_bars", side_effect=boom):
            buckets = redraw.screen_and_bucket(["BAD"], target_illiquid=5, target_moderate=5)
        self.assertEqual(buckets["illiquid"], [])
        self.assertEqual(len(buckets["screened_out"]), 1)
        self.assertIn("fetch_error", buckets["screened_out"][0]["reason"])

    def test_stops_once_both_targets_met(self):
        fake = {f"I{i}": _bars(600, 500_000) for i in range(10)}
        fake.update({f"M{i}": _bars(600, 2_000_000) for i in range(10)})
        calls = []

        def counting_fetch(t, d):
            calls.append(t)
            return fake[t]

        candidates = [f"I{i}" for i in range(10)] + [f"M{i}" for i in range(10)]
        with patch("backtest_v2.fetch_bars", side_effect=counting_fetch):
            buckets = redraw.screen_and_bucket(candidates, target_illiquid=2, target_moderate=2)
        self.assertEqual(len(buckets["illiquid"]), 2)
        self.assertEqual(len(buckets["moderate"]), 2)
        # loop breaks as soon as both targets are met, before scanning every candidate
        self.assertLess(len(calls), len(candidates))

    def test_excess_beyond_target_bucket_goes_to_screened_out(self):
        fake = {f"I{i}": _bars(600, 500_000) for i in range(3)}
        with patch("backtest_v2.fetch_bars", side_effect=lambda t, d: fake[t]):
            buckets = redraw.screen_and_bucket(list(fake), target_illiquid=1, target_moderate=5)
        self.assertEqual(len(buckets["illiquid"]), 1)
        self.assertEqual(len(buckets["screened_out"]), 2)


class TestConstantsMatchOriginal(unittest.TestCase):
    def test_stride_differs_from_original_draw(self):
        import illiquid_universe_probe as orig
        self.assertNotEqual(redraw.SAMPLE_STRIDE, 40)
        self.assertNotEqual(redraw.SAMPLE_STRIDE, getattr(orig, "SAMPLE_STRIDE", 40))

    def test_liquid_group_unchanged_from_original(self):
        import illiquid_universe_probe as orig
        self.assertEqual(orig.LIQUID, orig.LIQUID)  # sanity: module import works
        # redraw.py intentionally reuses orig.LIQUID directly (no local copy)
        with open(os.path.join(os.path.dirname(__file__), "scripts",
                                "illiquid_universe_probe_redraw.py")) as f:
            src = f.read()
        self.assertIn("orig.LIQUID", src)


if __name__ == "__main__":
    unittest.main()
