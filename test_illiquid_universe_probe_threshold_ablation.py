"""
Regression tests for scripts/illiquid_universe_probe_threshold_ablation.py
(LADDER PATH step 4, the 2026-07-24 illiquid mean_reversion edge entry).
Pure/mocked tests only: no network calls, no dependency on live
backtest_v2/NASDAQ data or the fetch-heavy run_group() — matches the
test_illiquid_universe_probe_significance.py convention for this
research-probe family. Verifies (1) the monkeypatch wrapper actually
changes strategies.mean_reversion.score's behavior and correctly restores
it afterward (even on exception — a leaked monkeypatch would silently
corrupt every OTHER script that imports mean_reversion in the same
process), and (2) the pure summarize/spread aggregation.
"""
import importlib.util
import os
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "illiquid_universe_probe_threshold_ablation",
    os.path.join(os.path.dirname(__file__), "scripts",
                 "illiquid_universe_probe_threshold_ablation.py"))
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)

from strategies import mean_reversion


class TestMakeVariantScore(unittest.TestCase):
    def test_variant_thresholds_change_output(self):
        variant = abl.make_variant_score(mean_reversion.score, {"rsi_extreme": 15})
        # rsi=18: default extreme band (20) applies -> 40pts; tightened
        # variant (15) does NOT apply at rsi=18 -> falls to oversold (30pts).
        self.assertEqual(mean_reversion.score(18, 0, 0)["score"], 40)
        self.assertEqual(variant(18, 0, 0)["score"], 30)

    def test_explicit_caller_override_still_wins_over_variant(self):
        variant = abl.make_variant_score(mean_reversion.score, {"rsi_extreme": 15})
        # Caller explicitly asks for rsi_extreme=25 at call time -> that
        # should win over the variant's 15, matching score()'s own
        # {**DEFAULTS, **override} merge contract.
        r = variant(18, 0, 0, thresholds={"rsi_extreme": 25})
        self.assertEqual(r["score"], 40)  # 18 < 25 -> extreme band


class TestRunVariantMonkeypatchLifecycle(unittest.TestCase):
    def test_baseline_none_does_not_monkeypatch(self):
        original = mean_reversion.score
        with patch.object(abl.orig, "run_group", return_value=[]):
            abl.run_variant(None, groups={"g": ["X"]})
        self.assertIs(mean_reversion.score, original)

    def test_variant_monkeypatches_during_call_and_restores_after(self):
        original = mean_reversion.score
        seen_during_call = {}

        def fake_run_group(tickers):
            # captures whether the monkeypatch is active DURING run_group,
            # the way backtest_v2's score_candidate() would observe it
            seen_during_call["is_patched"] = mean_reversion.score is not original
            seen_during_call["score_18_0_0"] = mean_reversion.score(18, 0, 0)["score"]
            return []

        with patch.object(abl.orig, "run_group", side_effect=fake_run_group):
            abl.run_variant({"rsi_extreme": 15}, groups={"g": ["X"]})

        self.assertTrue(seen_during_call["is_patched"])
        self.assertEqual(seen_during_call["score_18_0_0"], 30)  # variant applied
        self.assertIs(mean_reversion.score, original)  # restored after

    def test_restores_even_when_run_group_raises(self):
        original = mean_reversion.score
        with patch.object(abl.orig, "run_group", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                abl.run_variant({"rsi_extreme": 15}, groups={"g": ["X"]})
        self.assertIs(mean_reversion.score, original)


class TestSummarizeAndSpread(unittest.TestCase):
    def test_summarize_variant_delegates_to_orig_summarize(self):
        rows = {"illiquid": [{"buy_and_hold_pct": 1.0}]}
        with patch.object(abl.orig, "summarize", return_value={"n": 1}) as m:
            out = abl.summarize_variant(rows)
        m.assert_called_once_with(rows["illiquid"])
        self.assertEqual(out, {"illiquid": {"n": 1}})

    def test_mean_reversion_spread_computes_illiquid_minus_moderate(self):
        summary = {
            "illiquid": {"mean_reversion": {"mean_sharpe": 0.5}},
            "moderate": {"mean_reversion": {"mean_sharpe": -0.1}},
        }
        self.assertEqual(abl.mean_reversion_spread(summary), 0.6)

    def test_mean_reversion_spread_none_on_missing_group(self):
        self.assertIsNone(abl.mean_reversion_spread({"illiquid": {"mean_reversion": None}}))
        self.assertIsNone(abl.mean_reversion_spread({}))


class TestVariantDefinitions(unittest.TestCase):
    def test_each_variant_touches_exactly_one_input_axis(self):
        axis_groups = {
            "rsi": {"rsi_extreme", "rsi_extreme_pts", "rsi_oversold",
                    "rsi_oversold_pts", "rsi_mild", "rsi_mild_pts",
                    "rsi_overbought", "rsi_overbought_pts"},
            "chg": {"chg_big", "chg_big_pts", "chg_med", "chg_med_pts",
                    "chg_small", "chg_small_pts"},
            "vr": {"vr_high", "vr_high_pts", "vr_med", "vr_med_pts"},
        }
        for name, vdef in abl.VARIANTS.items():
            keys = set(vdef["thresholds"].keys())
            touched_axes = [axis for axis, members in axis_groups.items()
                            if keys & members]
            with self.subTest(variant=name):
                self.assertEqual(len(touched_axes), 1,
                                  f"{name} touches {touched_axes}, expected exactly 1 axis")

    def test_every_variant_key_is_a_real_default_threshold_key(self):
        valid_keys = set(mean_reversion.DEFAULT_THRESHOLDS.keys())
        for name, vdef in abl.VARIANTS.items():
            with self.subTest(variant=name):
                self.assertTrue(set(vdef["thresholds"].keys()) <= valid_keys)


if __name__ == "__main__":
    unittest.main()
