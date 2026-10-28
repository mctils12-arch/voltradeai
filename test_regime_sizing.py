"""
One sizing path (docs/regime/FINDINGS.md step 6): the engine's exposure_multiplier
replaces the VIX-level scalar, the stress-index multiplier and the Tier-1 VXX ladder
— one regime multiplier per sizing call, never two — behind VOLTRADE_REGIME_SIZING
(per sleeve: stock, options, tier1). A stale state means 1.0 and is counted (never a
silent cut); exposure 0 blocks a stock entry; an options sleeve is never sized to zero.
Flag off: every sizer is bit-identical to before.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import regime_engine as re_
import position_sizing as ps
import options_execution as oe
import instrument_selector as isel
import tiered_strategy as ts

ROOT = os.path.dirname(os.path.abspath(__file__))
BOT_SRC = open(os.path.join(ROOT, "bot_engine.py")).read()
TODAY = "2026-10-28"


def state(**over):
    base = {"as_of": "2026-10-27", "composite_regime": "CALM_BULL", "exposure_multiplier": 0.75,
            "allowed_strategies": ["momentum_long", "csp", "dip_buy"]}
    base.update(over)
    return base


def eng(mult=0.5, **over):
    base = {"multiplier": mult, "source": "engine", "stale": False, "as_of": "2026-10-27", "composite": "VOLATILE_BULL", "sleeve": "stock"}
    base.update(over)
    return base


class TestFlag(unittest.TestCase):
    def test_sleeves(self):
        self.assertEqual(re_.sizing_sleeves({}), set())
        self.assertEqual(re_.sizing_sleeves({"VOLTRADE_REGIME_SIZING": "off"}), set())
        self.assertEqual(re_.sizing_sleeves({"VOLTRADE_REGIME_SIZING": "all"}), set(re_.SIZING_SLEEVES))
        self.assertEqual(re_.sizing_sleeves({"VOLTRADE_REGIME_SIZING": "stock, tier1, bogus"}), {"stock", "tier1"})


class TestMultiplier(unittest.TestCase):
    def test_stock_takes_the_exposure_multiplier_as_is(self):
        self.assertEqual(re_.sizing_multiplier(state(), "stock", today=TODAY)["multiplier"], 0.75)
        z = re_.sizing_multiplier(state(exposure_multiplier=0.0, composite_regime="BEAR", allowed_strategies=["csp_reduced"]), "stock", today=TODAY)
        self.assertEqual((z["multiplier"], z["source"]), (0.0, "engine"))

    def test_options_sleeves_are_floored_by_what_the_state_still_permits(self):
        for sleeve in ("options", "tier1"):
            self.assertEqual(re_.sizing_multiplier(state(exposure_multiplier=0.0), sleeve, today=TODAY)["multiplier"], 0.5)        # csp allowed
            self.assertEqual(re_.sizing_multiplier(state(exposure_multiplier=0.0, allowed_strategies=["csp_reduced"]), sleeve, today=TODAY)["multiplier"], 0.25)
            self.assertEqual(re_.sizing_multiplier(state(exposure_multiplier=0.0, allowed_strategies=[]), sleeve, today=TODAY)["multiplier"], 0.25)   # CRISIS: gates block, the sizer never zeroes
            self.assertEqual(re_.sizing_multiplier(state(exposure_multiplier=0.75), sleeve, today=TODAY)["multiplier"], 0.75)

    def test_stale_or_failed_state_means_one_never_a_silent_cut(self):
        for st in (state(as_of="2026-10-20"), state(error="boom"), state(stale=True)):
            r = re_.sizing_multiplier(st, "stock", today=TODAY)
            self.assertEqual((r["multiplier"], r["source"], r["stale"]), (1.0, "stale", True))
        fresh = re_.sizing_multiplier(state(as_of="2026-10-26"), "stock", today=TODAY)   # 2 market days old: still fresh
        self.assertEqual(fresh["source"], "engine")

    def test_missing_exposure_or_state_means_one(self):
        r = re_.sizing_multiplier(state(exposure_multiplier=None), "stock", today=TODAY)
        self.assertEqual((r["multiplier"], r["source"]), (1.0, "missing"))
        self.assertEqual(re_.sizing_multiplier(None, "stock")["source"], "none")


class TestCounters(unittest.TestCase):
    def setUp(self):
        re_._SIZING_COUNTS.update(day=None, by_sleeve={}, stale_fallbacks=0, zero_blocks=0, last=None)

    tearDown = setUp

    def test_counts_per_sleeve_and_value_reset_daily_and_flush(self):
        re_.note_sizing(eng(0.75), "stock", today=TODAY)
        re_.note_sizing(eng(0.75), "stock", today=TODAY)
        re_.note_sizing(eng(1.0, source="stale", stale=True), "options", today=TODAY)
        re_.note_sizing(eng(0.0), "stock", today=TODAY)
        c = re_.sizing_counts()
        self.assertEqual(c["by_sleeve"]["stock"], {"0.75": 2, "0.00": 1})
        self.assertEqual(c["by_sleeve"]["options"], {"1.00": 1})
        self.assertEqual((c["stale_fallbacks"], c["zero_blocks"], c["last"]["sleeve"]), (1, 1, "stock"))
        re_.note_sizing(eng(0.5), "tier1", today="2026-10-29")             # new day: reset
        c = re_.sizing_counts()
        self.assertEqual((c["day"], c["by_sleeve"], c["stale_fallbacks"]), ("2026-10-29", {"tier1": {"0.50": 1}}, 0))
        with tempfile.TemporaryDirectory() as d:
            out = re_.flush_sizing_counts(os.path.join(d, "sizing.json"))
            on_disk = json.load(open(os.path.join(d, "sizing.json")))
        self.assertEqual(on_disk["by_sleeve"], {"tier1": {"0.50": 1}})
        self.assertIn("written_at", on_disk)
        self.assertNotIn("error", out)

    def test_sizing_for_is_none_unless_the_engine_is_on_and_the_sleeve_is_flagged(self):
        env_off = {"VOLTRADE_REGIME_ENGINE": "0", "VOLTRADE_REGIME_SIZING": "all"}
        self.assertIsNone(re_.sizing_for("stock", env=env_off, loader=lambda: state(), today=TODAY))
        env_other = {"VOLTRADE_REGIME_ENGINE": "1", "VOLTRADE_REGIME_SIZING": "tier1"}
        self.assertIsNone(re_.sizing_for("stock", env=env_other, loader=lambda: state(), today=TODAY))
        env_on = {"VOLTRADE_REGIME_ENGINE": "1", "VOLTRADE_REGIME_SIZING": "stock"}
        re_._STATE_CACHE.update(at=0.0, state=None)
        r = re_.sizing_for("stock", env=env_on, loader=lambda: state(), today=TODAY)
        re_._STATE_CACHE.update(at=0.0, state=None)
        self.assertEqual((r["multiplier"], r["source"]), (0.75, "engine"))
        self.assertEqual(re_.sizing_counts()["by_sleeve"]["stock"], {"0.75": 1})


STATS = {"overall": {"win_rate": 0.6, "avg_win": 1.0, "avg_loss": 0.8}, "by_strategy": {}}


class TestStockSizer(unittest.TestCase):
    def run_sizer(self, engine):
        trade = {"ticker": "AAPL", "price": 50.0, "score": 70, "deep_score": 70, "volume": 5_000_000, "ewma_rv": 2.0, "side": "buy"}
        with tempfile.TemporaryDirectory() as d, \
             patch("position_sizing.check_halt_status", return_value={"halted": False, "status": "ok"}), \
             patch("position_sizing._get_historical_stats", return_value=STATS), \
             patch("position_sizing._kelly_fraction", return_value=0.04), \
             patch("position_sizing._volatility_scalar", return_value=1.0), \
             patch("position_sizing._confidence_scalar", return_value=1.0), \
             patch("position_sizing._earnings_scalar", return_value=1.0), \
             patch("position_sizing._time_scalar", return_value=1.0), \
             patch("position_sizing._portfolio_heat_scalar", return_value=1.0), \
             patch("position_sizing._liquidity_scalar", return_value=1.0), \
             patch("position_sizing.DATA_DIR", d), \
             patch("stress_index.get_stress_multiplier", return_value=0.5), \
             patch("stress_index.compute_stress_index", return_value={"level": "stressed", "index": 70}), \
             patch("regime_engine.sizing_for", side_effect=lambda sleeve, **kw: engine if sleeve == "stock" else None):
            return ps.calculate_position(trade, 100_000.0, [], macro={"vix": 22.0})

    def test_flag_off_keeps_the_vix_scalar_and_the_stress_cut(self):
        a = self.run_sizer(None)
        b = self.run_sizer(None)
        self.assertEqual(a, b)                                            # deterministic: the bit-identical baseline
        self.assertEqual((a["scalars"]["regime"], a["scalars"]["stress"]), (0.8, 0.5))
        self.assertEqual((a["scalars"]["regime_engine"], a["scalars"]["regime_source"]), (None, "vix_scalar"))
        self.assertAlmostEqual(a["position_pct"], 0.04 * 0.8 * 0.5, places=6)
        self.assertIn("Market: 80%", a["reasoning"])

    def test_flag_on_applies_the_engine_once_and_mutes_the_stress_cut(self):
        r = self.run_sizer(eng(0.5))
        self.assertEqual((r["scalars"]["regime"], r["scalars"]["stress"], r["scalars"]["stress_level"]), (0.5, 1.0, "regime_engine"))
        self.assertEqual((r["scalars"]["regime_engine"], r["scalars"]["regime_source"]), (0.5, "engine"))
        self.assertAlmostEqual(r["position_pct"], 0.04 * 0.5, places=6)   # kelly x engine, nothing else regime-related
        self.assertIn("Regime engine: 50% (VOLATILE_BULL)", r["reasoning"])
        self.assertFalse(r["blocked"])

    def test_exposure_zero_blocks_the_entry(self):
        r = self.run_sizer(eng(0.0, composite="BEAR"))
        self.assertTrue(r["blocked"])
        self.assertIn("exposure 0 in BEAR", r["block_reason"])

    def test_stale_state_sizes_at_one(self):
        r = self.run_sizer(eng(1.0, source="stale", stale=True))
        self.assertEqual((r["scalars"]["regime"], r["scalars"]["stress"], r["scalars"]["regime_source"]), (1.0, 1.0, "stale"))
        self.assertAlmostEqual(r["position_pct"], 0.04, places=6)


class TestOptionsSizers(unittest.TestCase):
    def run_oe(self, engine):
        trade = {"ticker": "AAPL", "price": 50.0, "score": 70, "volume": 5_000_000, "ewma_rv": 2.0}
        with patch("options_execution._get_historical_stats", return_value=STATS), \
             patch("options_execution._kelly_fraction", return_value=0.04), \
             patch("options_execution._volatility_scalar", return_value=1.0), \
             patch("options_execution._confidence_scalar", return_value=1.0), \
             patch("options_execution._earnings_scalar", return_value=1.0), \
             patch("options_execution._time_scalar", return_value=1.0), \
             patch("options_execution._portfolio_heat_scalar", return_value=1.0), \
             patch("options_execution._liquidity_scalar", return_value=1.0), \
             patch("regime_engine.sizing_for", side_effect=lambda sleeve, **kw: engine if sleeve == "options" else None):
            return oe._dynamic_options_size(trade, 100_000.0, [], macro={"vix": 22.0})

    def run_isel(self, engine):
        trade = {"ticker": "AAPL", "price": 50.0, "score": 70, "volume": 5_000_000, "ewma_rv": 2.0}
        with patch("instrument_selector._get_historical_stats", return_value=STATS), \
             patch("instrument_selector._kelly_fraction", return_value=0.04), \
             patch("instrument_selector._volatility_scalar", return_value=1.0), \
             patch("instrument_selector._confidence_scalar_safe", return_value=1.0), \
             patch("instrument_selector._earnings_scalar", return_value=1.0), \
             patch("instrument_selector._time_scalar", return_value=1.0), \
             patch("instrument_selector._portfolio_heat_scalar", return_value=1.0), \
             patch("instrument_selector._liquidity_scalar", return_value=1.0), \
             patch("regime_engine.sizing_for", side_effect=lambda sleeve, **kw: engine if sleeve == "options" else None):
            return isel._dynamic_options_size(trade, 100_000.0, [], macro={"vix": 22.0})

    def test_both_options_sizers_swap_the_vix_scalar_for_the_engine_only_under_the_flag(self):
        for run in (self.run_oe, self.run_isel):
            off = run(None)
            self.assertAlmostEqual(off, 0.04 * 0.8 * 0.5, places=6)        # kelly x VIX scalar (22 -> 0.80) x leverage discount
            on = run(eng(0.75, sleeve="options"))
            self.assertAlmostEqual(on, 0.04 * 0.75 * 0.5, places=6)
            self.assertAlmostEqual(run(eng(1.0, source="stale", stale=True, sleeve="options")), 0.04 * 0.5, places=6)


class TestTier1(unittest.TestCase):
    UNIVERSE = ["AMD", "AAPL", "MSFT", "NVDA"]

    def run_t1(self, engine, vxx=1.5, equity=100_000.0, peak=100_000.0):
        ctx = ts.TierContext(equity=equity, peak_equity=peak, buying_power=equity, positions=[], vxx_ratio=vxx, spy_vs_ma50=1.0)
        with patch("tiered_strategy._get_t1_universe", return_value=self.UNIVERSE), \
             patch("csp_universe._load_layer2_cache", return_value={}), \
             patch("regime_engine.sizing_for", side_effect=lambda sleeve, **kw: engine if sleeve == "tier1" else None):
            actions = ts.tier1_csp_core(ctx)
        return actions, ts.get_regime_caps(vxx, 1.0, equity)["MAX_OPTIONS_PCT"]

    def test_flag_off_keeps_the_vxx_ladder(self):
        actions, max_pct = self.run_t1(None)
        self.assertTrue(actions)
        self.assertAlmostEqual(actions[0].size_pct, max_pct * 0.30, places=9)          # vxx 1.5 >= the kill ratio rung

    def test_flag_on_replaces_the_vxx_ladder_and_keeps_the_drawdown_ladder(self):
        actions, max_pct = self.run_t1(eng(0.5, sleeve="tier1"))
        self.assertAlmostEqual(actions[0].size_pct, max_pct * 0.5, places=9)
        actions, max_pct = self.run_t1(eng(0.5, sleeve="tier1"), equity=88_000.0, peak=100_000.0)   # DD -12%: x 0.75 stays
        self.assertAlmostEqual(actions[0].size_pct, max_pct * 0.5 * 0.75, places=9)
        actions, max_pct = self.run_t1(eng(1.0, source="stale", stale=True, sleeve="tier1"))
        self.assertAlmostEqual(actions[0].size_pct, max_pct * 1.0, places=9)             # stale: 1.0, never a silent cut


class TestScanWiring(unittest.TestCase):
    def test_scan_result_carries_the_flushed_counters_only_when_a_sleeve_is_flagged(self):
        self.assertIn('"regime_sizing": _regime_sizing_counts if \'_regime_sizing_counts\' in dir() else None,', BOT_SRC)
        i_flush = BOT_SRC.index("if _re_szc.engine_allowed() and _re_szc.sizing_sleeves():\n            _regime_sizing_counts = _re_szc.flush_sizing_counts()")
        i_key = BOT_SRC.index('"regime_sizing": _regime_sizing_counts')
        self.assertLess(i_flush, i_key)


if __name__ == "__main__":
    unittest.main()
