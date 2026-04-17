#!/usr/bin/env python3
"""
VolTradeAI — Risk Controls Unit Tests (v1.0.29+)

Covers:
  1. Portfolio DD halt with peak tracking + one-way ratchet resume
  2. -20% per-position hard floor (phase-1 only, stocks only)
  3. 10% DD-gated hedge budget escalation

These tests run without live Alpaca/Finnhub connectivity — pure logic tests
using isolated JSON state files under /tmp.
"""
import os
import sys
import json
import tempfile
import unittest

# Resolve repo root from this file's location
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Isolate data dir per-test-class to avoid cross-test contamination
_TEST_DATA = tempfile.mkdtemp(prefix="vt_risk_controls_")
os.environ["VOLTRADE_DATA_DIR"] = _TEST_DATA


class TestPortfolioDrawdownHalt(unittest.TestCase):
    """Portfolio-level DD halt: peak tracking, halt trigger, one-way ratchet."""

    def setUp(self):
        # Reset DD state before each test
        import bot_engine
        self.bot = bot_engine
        if os.path.exists(bot_engine._DD_STATE_PATH):
            os.remove(bot_engine._DD_STATE_PATH)

    def test_first_run_seeds_peak_at_current_equity(self):
        """Fresh state: peak should be seeded from first equity update."""
        r = self.bot.update_equity_peak(100_000, regime="BULL")
        self.assertEqual(r["peak_equity"], 100_000)
        self.assertEqual(r["dd_pct"], 0.0)
        self.assertFalse(r["halted"])

    def test_peak_ratchets_up_only(self):
        """Peak should never decrease."""
        self.bot.update_equity_peak(100_000, regime="BULL")
        r = self.bot.update_equity_peak(110_000, regime="BULL")
        self.assertEqual(r["peak_equity"], 110_000)
        # Now drop equity — peak stays at 110k
        r2 = self.bot.update_equity_peak(105_000, regime="BULL")
        self.assertEqual(r2["peak_equity"], 110_000)
        self.assertAlmostEqual(r2["dd_pct"], 4.545, places=2)

    def test_halt_triggers_at_18_percent_dd(self):
        """DD ≥ 18% must halt trading."""
        self.bot.update_equity_peak(100_000, regime="BULL")
        # Drop to 81,000 → 19% DD
        r = self.bot.update_equity_peak(81_000, regime="BEAR")
        self.assertTrue(r["halted"])
        self.assertIn("19.0", r["halt_reason"])
        self.assertTrue(self.bot.is_trading_halted())

    def test_halt_does_not_trigger_below_18_percent(self):
        """DD at 17% must NOT halt."""
        self.bot.update_equity_peak(100_000, regime="BULL")
        r = self.bot.update_equity_peak(83_000, regime="BEAR")  # 17% DD
        self.assertFalse(r["halted"])
        self.assertFalse(self.bot.is_trading_halted())

    def test_ratchet_requires_both_regime_and_equity_recovery(self):
        """Halt must NOT resume on regime alone — equity must also recover."""
        self.bot.update_equity_peak(100_000, regime="BULL")
        self.bot.update_equity_peak(80_000, regime="PANIC")  # halted at 20% DD
        self.assertTrue(self.bot.is_trading_halted())
        # BULL regime but still 15% below peak → should NOT resume (> 5% gap)
        r = self.bot.update_equity_peak(85_000, regime="BULL")
        self.assertTrue(r["halted"], "Halt should persist when equity gap > 5%")

    def test_ratchet_resumes_when_both_conditions_met(self):
        """Halt resumes only when regime ∈ {BULL, NEUTRAL} AND equity within 5% of peak."""
        self.bot.update_equity_peak(100_000, regime="BULL")
        self.bot.update_equity_peak(80_000, regime="PANIC")
        self.assertTrue(self.bot.is_trading_halted())
        # Recover to 96k (4% gap) in BULL → should resume
        r = self.bot.update_equity_peak(96_000, regime="BULL")
        self.assertFalse(r["halted"])

    def test_ratchet_blocks_resume_in_bear_rally(self):
        """Equity recovers but regime is BEAR — must stay halted."""
        self.bot.update_equity_peak(100_000, regime="BULL")
        self.bot.update_equity_peak(80_000, regime="PANIC")
        # Equity recovers to 97k (3% gap) but regime still BEAR → stays halted
        r = self.bot.update_equity_peak(97_000, regime="BEAR")
        self.assertTrue(r["halted"], "Halt must not resume in BEAR regime")

    def test_state_persists_across_loads(self):
        """Halt state must survive process restart (persistence check)."""
        self.bot.update_equity_peak(100_000, regime="BULL")
        self.bot.update_equity_peak(80_000, regime="PANIC")
        # Simulate reload by reading state fresh
        s = self.bot.get_portfolio_dd_state()
        self.assertTrue(s["halted"])
        self.assertEqual(s["peak_equity"], 100_000)

    def test_disabled_flag_bypasses_halt(self):
        """DRAWDOWN_HALT_ENABLED=False must prevent halt trigger."""
        from system_config import BASE_CONFIG
        _orig = BASE_CONFIG.get("DRAWDOWN_HALT_ENABLED")
        try:
            BASE_CONFIG["DRAWDOWN_HALT_ENABLED"] = False
            self.bot.update_equity_peak(100_000, regime="BULL")
            r = self.bot.update_equity_peak(70_000, regime="PANIC")  # 30% DD
            self.assertFalse(r["halted"])
        finally:
            BASE_CONFIG["DRAWDOWN_HALT_ENABLED"] = _orig


class TestHedgeEscalation(unittest.TestCase):
    """10% DD-gated hedge budget escalation logic."""

    def setUp(self):
        import bot_engine
        self.bot = bot_engine
        if os.path.exists(bot_engine._DD_STATE_PATH):
            os.remove(bot_engine._DD_STATE_PATH)

    def test_no_escalation_below_10_percent_dd(self):
        self.bot.update_equity_peak(100_000, regime="BULL")
        self.bot.update_equity_peak(95_000, regime="BULL")  # 5% DD
        self.assertFalse(self.bot.should_escalate_hedge())

    def test_escalates_at_exactly_10_percent_dd(self):
        self.bot.update_equity_peak(100_000, regime="BULL")
        self.bot.update_equity_peak(90_000, regime="BULL")  # 10% DD
        self.assertTrue(self.bot.should_escalate_hedge())

    def test_escalates_above_10_percent_dd(self):
        self.bot.update_equity_peak(100_000, regime="BULL")
        self.bot.update_equity_peak(85_000, regime="BULL")  # 15% DD
        self.assertTrue(self.bot.should_escalate_hedge())

    def test_escalation_independent_of_regime(self):
        """DD escalation must fire even in BULL regime (the whole point)."""
        self.bot.update_equity_peak(100_000, regime="BULL")
        self.bot.update_equity_peak(89_000, regime="BULL")  # 11% DD in BULL
        self.assertTrue(self.bot.should_escalate_hedge())

    def test_no_escalation_with_no_state(self):
        """Cold start: no state → no escalation (safe default)."""
        # State file already removed in setUp
        self.assertFalse(self.bot.should_escalate_hedge())


class TestConfigValues(unittest.TestCase):
    """Sanity checks on the three risk-control config values."""

    def test_dd_halt_pct_is_18(self):
        from system_config import BASE_CONFIG
        self.assertEqual(BASE_CONFIG["DRAWDOWN_HALT_PCT"], 18.0)

    def test_hard_stop_pct_is_20(self):
        from system_config import BASE_CONFIG
        self.assertEqual(BASE_CONFIG["POSITION_HARD_STOP_PCT"], 20.0)

    def test_hedge_escalate_pct_is_10(self):
        from system_config import BASE_CONFIG
        self.assertEqual(BASE_CONFIG["DRAWDOWN_HEDGE_ESCALATE_PCT"], 10.0)

    def test_resume_regimes_include_bull_and_neutral(self):
        from system_config import BASE_CONFIG
        resume = set(BASE_CONFIG["DRAWDOWN_HALT_RESUME_REGIMES"])
        self.assertIn("BULL", resume)
        self.assertIn("NEUTRAL", resume)
        # Must NOT include stressed regimes
        self.assertNotIn("BEAR", resume)
        self.assertNotIn("PANIC", resume)

    def test_resume_equity_gap_is_5pct(self):
        from system_config import BASE_CONFIG
        self.assertEqual(BASE_CONFIG["DRAWDOWN_HALT_RESUME_EQUITY_PCT"], 5.0)

    def test_hedge_escalate_less_than_halt(self):
        """Hedge must escalate BEFORE halt fires — 10% < 18%."""
        from system_config import BASE_CONFIG
        self.assertLess(
            BASE_CONFIG["DRAWDOWN_HEDGE_ESCALATE_PCT"],
            BASE_CONFIG["DRAWDOWN_HALT_PCT"],
            "Hedge escalation must trigger before portfolio halt"
        )


class TestHardStopIntegration(unittest.TestCase):
    """Source-level checks that the -20% hard stop is wired into manage_positions."""

    def test_hard_stop_block_exists_in_manage_positions(self):
        """The phase-1 hard-floor block must be present in bot_engine.py."""
        import bot_engine
        with open(bot_engine.__file__) as f:
            src = f.read()
        self.assertIn("POSITION_HARD_STOP_PCT", src)
        self.assertIn("POSITION_HARD_STOP_ENABLED", src)
        self.assertIn("HARD FLOOR", src)
        # Phase guard must be present so floor only fires pre-scale-out
        self.assertIn("phase < 2", src)

    def test_options_skip_block_present(self):
        """Options positions must be skipped by the stock stop logic."""
        import bot_engine
        with open(bot_engine.__file__) as f:
            src = f.read()
        # Confirm options are explicitly skipped in manage_positions
        self.assertIn('us_option', src)
        self.assertIn('asset_class', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
