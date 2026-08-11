"""
Regression tests for scripts/illiquid_universe_probe_universe_gate.py
(LADDER PATH step 5 — LOGIC-gate ablation of the pinned illiquid/moderate/
liquid mean_reversion candidate lists against the live bot's actual
scan_market() universe filter). Pure/mocked tests only: no network calls,
no dependency on live backtest_v2/Yahoo data — matches the existing
test_illiquid_universe_probe_threshold_ablation_redraw.py convention for
this research-probe family.
"""
import importlib.util
import os
import unittest

import bot_engine

_spec = importlib.util.spec_from_file_location(
    "illiquid_universe_probe_universe_gate",
    os.path.join(os.path.dirname(__file__), "scripts",
                 "illiquid_universe_probe_universe_gate.py"))
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


class TestLiveConstantsNotCopied(unittest.TestCase):
    """The whole point of this probe is that it can never silently drift
    from the live scan_market() filter — pin identity, not just value."""

    def test_min_price_is_the_live_bot_engine_constant(self):
        self.assertIs(gate.MIN_PRICE, bot_engine.MIN_PRICE)

    def test_min_volume_is_the_live_bot_engine_constant(self):
        self.assertIs(gate.MIN_VOLUME, bot_engine.MIN_VOLUME)


class TestPinnedTickerListsReused(unittest.TestCase):
    """Same discipline as illiquid_universe_probe_threshold_ablation_redraw.py:
    the candidate lists must be the SAME objects steps 1-4 evaluated, not a
    re-typed copy that could silently diverge."""

    def test_illiquid_is_the_original_pinned_list(self):
        import illiquid_universe_probe as orig
        self.assertIs(gate.ILLIQUID, orig.ILLIQUID)

    def test_moderate_is_the_original_pinned_list(self):
        import illiquid_universe_probe as orig
        self.assertIs(gate.MODERATE, orig.MODERATE)

    def test_liquid_is_the_original_pinned_list(self):
        import illiquid_universe_probe as orig
        self.assertIs(gate.LIQUID, orig.LIQUID)


class TestPassesUniverseGate(unittest.TestCase):
    def test_passes_when_both_at_exactly_threshold(self):
        # scan_market()'s own check is `c < MIN_PRICE or v < _min_vol` (strict
        # less-than fails), so exactly-at-threshold must PASS.
        self.assertTrue(gate.passes_universe_gate(5.0, 500_000, min_price=5.0, min_volume=500_000))

    def test_fails_just_under_price_threshold(self):
        self.assertFalse(gate.passes_universe_gate(4.99, 1_000_000, min_price=5.0, min_volume=500_000))

    def test_fails_just_under_volume_threshold(self):
        self.assertFalse(gate.passes_universe_gate(100.0, 499_999, min_price=5.0, min_volume=500_000))

    def test_fails_both_under(self):
        self.assertFalse(gate.passes_universe_gate(1.0, 1_000, min_price=5.0, min_volume=500_000))

    def test_passes_when_both_comfortably_over(self):
        self.assertTrue(gate.passes_universe_gate(50.0, 5_000_000, min_price=5.0, min_volume=500_000))

    def test_none_inputs_fail_closed(self):
        self.assertFalse(gate.passes_universe_gate(None, 1_000_000))
        self.assertFalse(gate.passes_universe_gate(50.0, None))


class TestCheckTicker(unittest.TestCase):
    def test_no_data_reports_error_and_fails_gate(self):
        result = gate.check_ticker("FAKE", fetch_fn=lambda t, d: {"close": [], "volume": []})
        self.assertFalse(result["passes_gate"])
        self.assertEqual(result["error"], "no data")

    def test_computes_trailing_window_average_not_single_day(self):
        # 21 days of volume, oldest-first; a lone low-volume day OUTSIDE the
        # trailing 20-day window must not drag the average down.
        volumes = [1] + [1_000_000] * 20
        bars = {"close": [10.0] * 21, "volume": volumes}
        result = gate.check_ticker("FAKE", fetch_fn=lambda t, d: bars, window=20)
        self.assertAlmostEqual(result["avg_volume"], 1_000_000, delta=1)
        self.assertTrue(result["passes_volume"])

    def test_uses_last_close_not_first(self):
        bars = {"close": [1.0, 2.0, 100.0], "volume": [1_000_000] * 3}
        result = gate.check_ticker("FAKE", fetch_fn=lambda t, d: bars)
        self.assertEqual(result["last_close"], 100.0)
        self.assertTrue(result["passes_price"])


class TestCheckGroup(unittest.TestCase):
    def test_aggregates_pass_count_and_rate(self):
        fake_bars = {
            "PASS1": {"close": [50.0], "volume": [1_000_000]},
            "PASS2": {"close": [50.0], "volume": [1_000_000]},
            "FAIL_PRICE": {"close": [1.0], "volume": [1_000_000]},
            "FAIL_VOL": {"close": [50.0], "volume": [1_000]},
        }

        def fake_fetch(t, d):
            return fake_bars[t]

        result = gate.check_group(list(fake_bars.keys()), fetch_fn=fake_fetch)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["n_pass"], 2)
        self.assertEqual(result["pass_rate_pct"], 50.0)
        self.assertEqual(result["n_price_only"], 1)  # FAIL_VOL: price ok, volume not
        self.assertEqual(result["n_volume_only"], 1)  # FAIL_PRICE: volume ok, price not


if __name__ == "__main__":
    unittest.main()
