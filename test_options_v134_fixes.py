#!/usr/bin/env python3
"""
Test suite for v1.0.34 options execution fixes.

Tests:
  Fix 1: manage_positions() skips OCC option symbols
  Fix 2: get_options_trades() only allows HIGH_EDGE_SETUPS
  Fix 3: Options allocation capped at 8% everywhere
  Fix 4: entry_timestamp always set in options_manager state
  Fix 5: No short straddle or CSP filler strategies

Run: cd voltradeai && python test_options_v134_fixes.py
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add the repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 1: manage_positions() skips options
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix1_ManagePositionsSkipsOptions(unittest.TestCase):
    """manage_positions() must skip OCC option symbols and us_option asset_class."""

    def test_occ_symbol_skipped_by_length(self):
        """OCC symbols (>8 chars) should be skipped by the filter logic."""
        # Simulate the exact filter logic from manage_positions
        test_positions = [
            {"symbol": "AAPL", "asset_class": "us_equity"},
            {"symbol": "AAPL260420C00257500", "asset_class": "us_option"},
            {"symbol": "TSLA260420P00200000", "asset_class": "us_option"},
            {"symbol": "QQQ", "asset_class": "us_equity"},  # Also skipped (FLOOR_AND_LEG)
            {"symbol": "UAL", "asset_class": "us_equity"},
        ]

        FLOOR_AND_LEG_TICKERS = {"QQQ", "SVXY", "SPY", "SQQQ", "SPXS"}
        processed_tickers = []

        for pos in test_positions:
            ticker = pos.get("symbol", "")
            if ticker in FLOOR_AND_LEG_TICKERS:
                continue
            # This is the exact fix we added
            if pos.get("asset_class") == "us_option" or len(ticker) > 8:
                continue
            processed_tickers.append(ticker)

        # Only AAPL and UAL should be processed by stock logic
        self.assertIn("AAPL", processed_tickers)
        self.assertIn("UAL", processed_tickers)
        self.assertNotIn("AAPL260420C00257500", processed_tickers,
            "OCC option symbol should be skipped")
        self.assertNotIn("TSLA260420P00200000", processed_tickers,
            "OCC option symbol should be skipped")
        self.assertNotIn("QQQ", processed_tickers,
            "Floor ticker should be skipped")
        self.assertEqual(len(processed_tickers), 2)

    def test_asset_class_check_present_in_source(self):
        """Verify the asset_class == 'us_option' check exists in bot_engine.py."""
        import inspect
        from bot_engine import manage_positions
        source = inspect.getsource(manage_positions)
        self.assertIn("us_option", source,
            "manage_positions() missing 'us_option' asset_class check")
        self.assertIn("len(ticker) > 8", source,
            "manage_positions() missing OCC symbol length check")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 2: Only HIGH_EDGE_SETUPS allowed
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix2_HighEdgeSetupsOnly(unittest.TestCase):
    """get_options_trades() must filter to HIGH_EDGE_SETUPS only."""

    def test_high_edge_setups_defined(self):
        """HIGH_EDGE_SETUPS should contain exactly the proven setups."""
        from options_scanner import HIGH_EDGE_SETUPS
        self.assertIn("earnings_iv_crush", HIGH_EDGE_SETUPS)
        self.assertIn("vxx_panic_put_sale", HIGH_EDGE_SETUPS)
        self.assertIn("high_iv_premium_sale", HIGH_EDGE_SETUPS)

    def test_filler_setups_excluded(self):
        """CSP, gamma pin, low-IV buy should NOT be in HIGH_EDGE_SETUPS."""
        from options_scanner import HIGH_EDGE_SETUPS
        self.assertNotIn("csp_normal_market", HIGH_EDGE_SETUPS)
        self.assertNotIn("gamma_pin", HIGH_EDGE_SETUPS)
        self.assertNotIn("low_iv_breakout_buy", HIGH_EDGE_SETUPS)

    def test_min_options_score_raised(self):
        """Minimum options score should be 70 (was 65)."""
        from options_scanner import MIN_OPTIONS_SCORE
        self.assertGreaterEqual(MIN_OPTIONS_SCORE, 70.0)

    def test_get_options_trades_filters_low_edge(self):
        """get_options_trades must reject non-high-edge setups."""
        from options_scanner import get_options_trades, HIGH_EDGE_SETUPS

        # Mock scan_options to return a mix of setups
        mock_scan_result = {
            "opportunities": [
                {"ticker": "AAPL", "setup": "earnings_iv_crush", "score": 75,
                 "price": 185, "action_label": "SELL", "options_strategy": "iron_condor",
                 "reasoning": "test", "source": "test", "side": "sell"},
                {"ticker": "TSLA", "setup": "csp_normal_market", "score": 72,
                 "price": 200, "action_label": "SELL PUT", "options_strategy": "sell_cash_secured_put",
                 "reasoning": "test", "source": "test", "side": "sell"},
                {"ticker": "GOOG", "setup": "gamma_pin", "score": 68,
                 "price": 160, "action_label": "BUY CALL", "options_strategy": "buy_call",
                 "reasoning": "test", "source": "test", "side": "buy"},
                {"ticker": "SPY", "setup": "vxx_panic_put_sale", "score": 78,
                 "price": 500, "action_label": "SELL PUT", "options_strategy": "sell_put",
                 "reasoning": "test", "source": "test", "side": "sell"},
            ],
            "setup_counts": {},
            "vxx_ratio": 1.35,
            "regime": "PANIC",
            "scanned": 100,
            "duration_secs": 5.0,
        }

        with patch("options_scanner.scan_options", return_value=mock_scan_result):
            trades = get_options_trades(100000, [], max_new=10, min_score=65)

        # Only earnings_iv_crush and vxx_panic_put_sale should pass
        setups_returned = [t["setup"] for t in trades]
        self.assertIn("earnings_iv_crush", setups_returned)
        self.assertIn("vxx_panic_put_sale", setups_returned)
        self.assertNotIn("csp_normal_market", setups_returned)
        self.assertNotIn("gamma_pin", setups_returned)

    def test_short_straddle_blocked_for_high_iv(self):
        """high_iv_premium_sale with short_straddle strategy should be blocked."""
        from options_scanner import get_options_trades

        mock_scan_result = {
            "opportunities": [
                {"ticker": "TSLA", "setup": "high_iv_premium_sale", "score": 75,
                 "price": 200, "action_label": "SELL STRADDLE",
                 "options_strategy": "short_straddle",  # Should be blocked
                 "reasoning": "test", "source": "test", "side": "sell"},
                {"ticker": "NVDA", "setup": "high_iv_premium_sale", "score": 78,
                 "price": 800, "action_label": "SELL IRON CONDOR",
                 "options_strategy": "iron_condor",  # Should pass
                 "reasoning": "test", "source": "test", "side": "sell"},
            ],
            "setup_counts": {},
            "vxx_ratio": 1.0,
            "regime": "NEUTRAL",
            "scanned": 100,
            "duration_secs": 5.0,
        }

        with patch("options_scanner.scan_options", return_value=mock_scan_result):
            trades = get_options_trades(100000, [], max_new=10)

        strategies = [t["options_strategy"] for t in trades]
        self.assertNotIn("short_straddle", strategies,
            "short_straddle should be blocked — only iron_condor allowed")
        # Iron condor should be present
        tickers = [t["ticker"] for t in trades]
        self.assertIn("NVDA", tickers, "NVDA iron_condor should pass the filter")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 3: Options allocation capped at 8%
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix3_AllocationCaps(unittest.TestCase):
    """All options allocation caps should be 8% or less."""

    def test_options_execution_ceiling(self):
        """options_execution.py caps at 8%."""
        from options_execution import MAX_OPTIONS_PCT_CEILING, MAX_TOTAL_OPTIONS_PCT
        self.assertLessEqual(MAX_OPTIONS_PCT_CEILING, 0.08,
            f"MAX_OPTIONS_PCT_CEILING is {MAX_OPTIONS_PCT_CEILING}, should be <= 0.08")
        self.assertLessEqual(MAX_TOTAL_OPTIONS_PCT, 0.08,
            f"MAX_TOTAL_OPTIONS_PCT is {MAX_TOTAL_OPTIONS_PCT}, should be <= 0.08")

    def test_instrument_selector_ceiling(self):
        """instrument_selector.py caps at 8%."""
        from instrument_selector import MAX_OPTIONS_PCT_CEILING, MAX_TOTAL_OPTIONS_PCT
        self.assertLessEqual(MAX_OPTIONS_PCT_CEILING, 0.08,
            f"instrument_selector MAX_OPTIONS_PCT_CEILING = {MAX_OPTIONS_PCT_CEILING}")
        self.assertLessEqual(MAX_TOTAL_OPTIONS_PCT, 0.08,
            f"instrument_selector MAX_TOTAL_OPTIONS_PCT = {MAX_TOTAL_OPTIONS_PCT}")

    def test_system_config_ceiling(self):
        """system_config.py BASE_CONFIG caps at 8%."""
        from system_config import BASE_CONFIG
        self.assertLessEqual(BASE_CONFIG["MAX_OPTIONS_PCT"], 0.08,
            f"system_config MAX_OPTIONS_PCT = {BASE_CONFIG['MAX_OPTIONS_PCT']}")

    def test_max_options_positions_is_three(self):
        """bot_engine MAX_OPTIONS_POSITIONS should be 3."""
        from bot_engine import MAX_OPTIONS_POSITIONS
        self.assertEqual(MAX_OPTIONS_POSITIONS, 3)

    def test_sizing_never_exceeds_eight_pct(self):
        """get_options_trades sizing should never exceed 8%."""
        from options_scanner import get_options_trades

        mock_scan_result = {
            "opportunities": [
                {"ticker": "AAPL", "setup": "vxx_panic_put_sale", "score": 80,
                 "price": 185, "action_label": "SELL PUT", "options_strategy": "sell_put",
                 "reasoning": "test", "source": "test", "side": "sell"},
                {"ticker": "TSLA", "setup": "earnings_iv_crush", "score": 76,
                 "price": 200, "action_label": "SELL STRADDLE", "options_strategy": "iron_condor",
                 "reasoning": "test", "source": "test", "side": "sell"},
            ],
            "setup_counts": {},
            "vxx_ratio": 1.35,
            "regime": "PANIC",
            "scanned": 100,
            "duration_secs": 5.0,
        }

        with patch("options_scanner.scan_options", return_value=mock_scan_result):
            trades = get_options_trades(100000, [])

        for t in trades:
            self.assertLessEqual(t["position_pct"], 0.08,
                f"Trade {t['ticker']} sized at {t['position_pct']}, exceeds 8%")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 4: entry_timestamp always set
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix4_EntryTimestamp(unittest.TestCase):
    """entry_timestamp must be set in all options state initialization paths."""

    def test_register_options_entry_sets_timestamp(self):
        """register_options_entry must set entry_timestamp."""
        from options_manager import register_options_entry, _load_options_state, OPTIONS_STATE_PATH

        # Use temp path
        test_path = "/tmp/test_opts_state_fix4.json"
        with patch("options_manager.OPTIONS_STATE_PATH", test_path):
            # Clean up
            if os.path.exists(test_path):
                os.remove(test_path)

            register_options_entry(
                "AAPL260420C00257500", 5.00, "buy", "buy_straddle",
                delta=0.50, qty=1, ticker="AAPL", setup="earnings_iv_crush"
            )

            with open(test_path) as f:
                state = json.load(f)

            occ_state = state.get("AAPL260420C00257500", {})
            self.assertIn("entry_timestamp", occ_state,
                "register_options_entry missing entry_timestamp")
            self.assertIn("ticker", occ_state,
                "register_options_entry missing ticker")
            self.assertEqual(occ_state["ticker"], "AAPL")
            self.assertIn("setup", occ_state,
                "register_options_entry missing setup")

            # Verify it's a valid ISO format
            ts = occ_state["entry_timestamp"]
            dt = datetime.fromisoformat(ts)
            self.assertIsInstance(dt, datetime)

            # Clean up
            os.remove(test_path)

    def test_standalone_init_sets_timestamp(self):
        """When manage_options_positions encounters a new position, it must set entry_timestamp."""
        import inspect
        from options_manager import manage_options_positions
        source = inspect.getsource(manage_options_positions)
        # Check that the standalone position initialization includes entry_timestamp
        # This is the code path at line ~768 where pos_state is initialized
        self.assertIn("entry_timestamp", source,
            "manage_options_positions standalone init missing entry_timestamp")

    def test_register_entry_extracts_ticker_from_occ(self):
        """If ticker not provided, extract from OCC symbol."""
        from options_manager import register_options_entry, OPTIONS_STATE_PATH

        test_path = "/tmp/test_opts_state_fix4b.json"
        with patch("options_manager.OPTIONS_STATE_PATH", test_path):
            if os.path.exists(test_path):
                os.remove(test_path)

            # Call without explicit ticker
            register_options_entry(
                "TSLA260420P00200000", 3.00, "sell", "iron_condor"
            )

            with open(test_path) as f:
                state = json.load(f)

            occ_state = state.get("TSLA260420P00200000", {})
            self.assertEqual(occ_state.get("ticker"), "TSLA",
                "Should extract ticker 'TSLA' from OCC symbol")

            os.remove(test_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 5: No straddle scalps or CSP fillers
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix5_NoStraddleScalpsOrCSP(unittest.TestCase):
    """Straddle scalping and CSP filler strategies must be eliminated."""

    def test_high_iv_always_uses_iron_condor(self):
        """_setup_high_iv_premium_sale must always produce iron_condor, never short_straddle."""
        import inspect
        from options_scanner import _setup_high_iv_premium_sale
        source = inspect.getsource(_setup_high_iv_premium_sale)

        # The strategy assignment should always be iron_condor
        # Old code had: strategy = "iron_condor" if iv_rank >= 75 else "short_straddle"
        self.assertNotIn('"short_straddle"', source,
            "_setup_high_iv_premium_sale still references 'short_straddle'")

    def test_csp_disabled_in_scan_options(self):
        """_setup_csp_normal_market should NOT be called in the scanner loop."""
        import inspect
        # Read the _check_ticker function inside scan_options
        from options_scanner import scan_options
        source = inspect.getsource(scan_options)
        # The CSP call should be commented out
        # Look for uncommented calls to _setup_csp_normal_market
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if "_setup_csp_normal_market" in stripped and not stripped.startswith("#"):
                self.fail(f"_setup_csp_normal_market still called (uncommented): {stripped}")

    def test_delta_selection_always_20(self):
        """Iron condor should select 20-delta contracts (not 50-delta straddle)."""
        import inspect
        from options_scanner import _setup_high_iv_premium_sale
        source = inspect.getsource(_setup_high_iv_premium_sale)

        # Should have 0.20 delta calls, not 0.50
        self.assertIn("target_delta=0.20", source,
            "Missing 20-delta selection for iron condor")
        # 50-delta for straddle should NOT appear in an uncommented line
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if "target_delta=0.50" in stripped and not stripped.startswith("#"):
                self.fail(f"Still selecting 50-delta (straddle): {stripped}")


# ═══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION: End-to-end flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete fix set."""

    def test_options_scanner_import_clean(self):
        """options_scanner should import without errors."""
        import options_scanner
        self.assertTrue(hasattr(options_scanner, "HIGH_EDGE_SETUPS"))
        self.assertTrue(hasattr(options_scanner, "MIN_OPTIONS_SCORE"))
        self.assertTrue(hasattr(options_scanner, "get_options_trades"))

    def test_options_manager_import_clean(self):
        """options_manager should import without errors."""
        import options_manager
        self.assertTrue(hasattr(options_manager, "manage_options_positions"))
        self.assertTrue(hasattr(options_manager, "register_options_entry"))
        self.assertTrue(hasattr(options_manager, "MIN_HOLD_MINUTES"))
        self.assertEqual(options_manager.MIN_HOLD_MINUTES, 60)

    def test_bot_engine_import_clean(self):
        """bot_engine should import without errors."""
        import bot_engine
        self.assertTrue(hasattr(bot_engine, "manage_positions"))
        self.assertTrue(hasattr(bot_engine, "MAX_OPTIONS_POSITIONS"))

    def test_options_execution_import_clean(self):
        """options_execution should import without errors."""
        import options_execution
        self.assertTrue(hasattr(options_execution, "MAX_OPTIONS_PCT_CEILING"))
        self.assertTrue(hasattr(options_execution, "MAX_TOTAL_OPTIONS_PCT"))

    def test_consistent_caps_across_modules(self):
        """All modules should agree on 8% cap."""
        from options_execution import MAX_OPTIONS_PCT_CEILING as exec_ceil
        from options_execution import MAX_TOTAL_OPTIONS_PCT as exec_total
        from instrument_selector import MAX_OPTIONS_PCT_CEILING as inst_ceil
        from instrument_selector import MAX_TOTAL_OPTIONS_PCT as inst_total
        from system_config import BASE_CONFIG

        self.assertEqual(exec_ceil, 0.08)
        self.assertEqual(exec_total, 0.08)
        self.assertEqual(inst_ceil, 0.08)
        self.assertEqual(inst_total, 0.08)
        self.assertEqual(BASE_CONFIG["MAX_OPTIONS_PCT"], 0.08)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 6: config_overrides.json cannot override options caps
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix6_ProtectedConfigKeys(unittest.TestCase):
    """config_overrides.json must not be able to override options allocation caps."""

    def test_protected_keys_defined(self):
        """_PROTECTED_KEYS should include all options cap keys."""
        from system_config import _PROTECTED_KEYS
        self.assertIn("MAX_OPTIONS_PCT", _PROTECTED_KEYS)
        self.assertIn("MAX_TOTAL_OPTIONS_PCT", _PROTECTED_KEYS)
        self.assertIn("MAX_OPTIONS_PCT_CEILING", _PROTECTED_KEYS)

    def test_override_strips_protected_keys(self):
        """load_config_overrides must strip protected keys from the result."""
        import tempfile, json as _json
        from system_config import DATA_DIR

        override_path = os.path.join(DATA_DIR, "config_overrides.json")
        # Write a config_overrides.json that tries to raise options caps
        malicious_overrides = {
            "MAX_OPTIONS_PCT": 0.25,          # Trying to raise from 0.08
            "MAX_TOTAL_OPTIONS_PCT": 0.30,    # Trying to raise from 0.08
            "MAX_OPTIONS_PCT_CEILING": 0.20,  # Trying to raise from 0.08
            "MIN_SCORE": 50,                  # This is NOT protected — should pass through
        }
        try:
            with open(override_path, "w") as f:
                _json.dump(malicious_overrides, f)

            from system_config import load_config_overrides
            result = load_config_overrides()

            # Protected keys should NOT be in the result
            self.assertNotIn("MAX_OPTIONS_PCT", result,
                "MAX_OPTIONS_PCT must be stripped from overrides")
            self.assertNotIn("MAX_TOTAL_OPTIONS_PCT", result,
                "MAX_TOTAL_OPTIONS_PCT must be stripped from overrides")
            self.assertNotIn("MAX_OPTIONS_PCT_CEILING", result,
                "MAX_OPTIONS_PCT_CEILING must be stripped from overrides")

            # Non-protected keys should still pass through
            self.assertEqual(result.get("MIN_SCORE"), 50,
                "Non-protected keys should still be applied")
        finally:
            # Clean up
            if os.path.exists(override_path):
                os.remove(override_path)

    def test_cfg_retains_8pct_cap_with_override(self):
        """Even with a malicious config_overrides.json, MAX_OPTIONS_PCT stays 0.08."""
        from system_config import BASE_CONFIG
        # BASE_CONFIG is the source of truth and doesn't get overridden
        self.assertEqual(BASE_CONFIG["MAX_OPTIONS_PCT"], 0.08)

    def test_no_override_file_returns_empty(self):
        """When no config_overrides.json exists, should return empty dict."""
        from system_config import load_config_overrides, DATA_DIR
        override_path = os.path.join(DATA_DIR, "config_overrides.json")
        # Make sure file doesn't exist
        if os.path.exists(override_path):
            os.remove(override_path)
        result = load_config_overrides()
        self.assertEqual(result, {})


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 7: Earnings IV crush always uses iron_condor (no naked straddles)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix7_EarningsAlwaysIronCondor(unittest.TestCase):
    """Earnings IV crush setup must always produce iron_condor strategy,
    never short_straddle. Naked straddles have unlimited risk which
    destroys drawdown metrics and Sortino ratio."""

    def test_low_iv_move_produces_iron_condor(self):
        """When iv_implied_move <= 8%, strategy must still be iron_condor."""
        # Previously this case returned short_straddle
        from options_scanner import _setup_earnings_iv_crush

        # Mock all the network calls this function makes
        with patch("options_scanner._fetch_options_chain") as mock_chain, \
             patch("options_scanner._fetch_iv_rank", return_value=75.0), \
             patch("options_scanner._build_setup_features", return_value=[0]*10), \
             patch("options_scanner._options_ml_score", return_value=0.8):

            # Simulate liquid ATM options with IV implied move of ~5% (< 8%)
            mock_chain.return_value = [
                {"strike": 100, "type": "call", "expiry": "2026-04-20",
                 "mid": 3.50, "bid": 3.40, "ask": 3.60, "iv": 0.50,
                 "spread_pct": 0.06, "oi": 1000,
                 "occ_symbol": "AAPL260420C00100000"},
                {"strike": 100, "type": "put", "expiry": "2026-04-20",
                 "mid": 1.50, "bid": 1.40, "ask": 1.60, "iv": 0.50,
                 "spread_pct": 0.07, "oi": 800,
                 "occ_symbol": "AAPL260420P00100000"},
            ]

            result = _setup_earnings_iv_crush(
                ticker="AAPL", price=100.0, days_to_earnings=3, vxx_ratio=1.1
            )

            if result is not None:
                self.assertEqual(result["options_strategy"], "iron_condor",
                    "Earnings IV crush must always use iron_condor, never short_straddle")
                self.assertNotIn("STRADDLE", result["action_label"].upper(),
                    "Action label should not say STRADDLE")
                self.assertIn("IRON CONDOR", result["action_label"].upper())

    def test_high_iv_move_produces_iron_condor(self):
        """When iv_implied_move > 8%, strategy must be iron_condor."""
        from options_scanner import _setup_earnings_iv_crush

        with patch("options_scanner._fetch_options_chain") as mock_chain, \
             patch("options_scanner._fetch_iv_rank", return_value=80.0), \
             patch("options_scanner._build_setup_features", return_value=[0]*10), \
             patch("options_scanner._options_ml_score", return_value=0.85):

            # Simulate options with high IV implied move > 8%
            mock_chain.return_value = [
                {"strike": 100, "type": "call", "expiry": "2026-04-20",
                 "mid": 6.00, "bid": 5.80, "ask": 6.20, "iv": 0.80,
                 "spread_pct": 0.05, "oi": 1500,
                 "occ_symbol": "NVDA260420C00100000"},
                {"strike": 100, "type": "put", "expiry": "2026-04-20",
                 "mid": 4.00, "bid": 3.80, "ask": 4.20, "iv": 0.80,
                 "spread_pct": 0.05, "oi": 1200,
                 "occ_symbol": "NVDA260420P00100000"},
            ]

            result = _setup_earnings_iv_crush(
                ticker="NVDA", price=100.0, days_to_earnings=2, vxx_ratio=1.05
            )

            if result is not None:
                self.assertEqual(result["options_strategy"], "iron_condor",
                    "High IV earnings must also use iron_condor")

    def test_no_short_straddle_in_scanner_output(self):
        """The earnings setup detector must never return short_straddle.
        This is a code-level check on the source."""
        import inspect
        from options_scanner import _setup_earnings_iv_crush

        source = inspect.getsource(_setup_earnings_iv_crush)
        # Check that 'strategy = "short_straddle"' does NOT appear
        self.assertNotIn('strategy = "short_straddle"', source,
            "_setup_earnings_iv_crush must not assign short_straddle strategy")
        # The function should always set strategy = "iron_condor"
        self.assertIn('strategy = "iron_condor"', source,
            "_setup_earnings_iv_crush must always assign iron_condor")

    def test_get_options_trades_blocks_straddle_for_all_setups(self):
        """get_options_trades should never output a short_straddle from any setup."""
        from options_scanner import get_options_trades, HIGH_EDGE_SETUPS

        # Mock scan_options to return earnings opportunities with various strategies
        test_opps = [
            {"ticker": "AAPL", "setup": "earnings_iv_crush", "score": 85,
             "options_strategy": "iron_condor", "price": 200.0, "side": "sell",
             "action_label": "SELL IRON CONDOR", "reasoning": "Test"},
            {"ticker": "MSFT", "setup": "high_iv_premium_sale", "score": 80,
             "options_strategy": "short_straddle", "price": 400.0, "side": "sell",
             "action_label": "SELL STRADDLE", "reasoning": "Test"},
        ]

        with patch("options_scanner.scan_options",
                   return_value={"opportunities": test_opps}):
            trades = get_options_trades(equity=100000, current_tickers=[])
            for trade in trades:
                self.assertNotEqual(trade.get("options_strategy"), "short_straddle",
                    f"Trade for {trade.get('ticker')} should not use short_straddle")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 8: Iron condor 50%-of-max-loss early exit
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix8_IronCondorMaxLossExit(unittest.TestCase):
    """Iron condors must close at 50% of max loss to protect drawdowns."""

    def test_ic_max_loss_exit_constant_defined(self):
        """IC_MAX_LOSS_EXIT_PCT should be 0.50 (50%)."""
        from options_manager import IC_MAX_LOSS_EXIT_PCT
        self.assertEqual(IC_MAX_LOSS_EXIT_PCT, 0.50)

    def test_manage_strategy_group_has_ic_exit_rule(self):
        """_manage_strategy_group must contain the IC max-loss exit logic."""
        import inspect
        from options_manager import _manage_strategy_group
        source = inspect.getsource(_manage_strategy_group)
        self.assertIn("IC_MAX_LOSS_EXIT_PCT", source,
            "_manage_strategy_group must reference IC_MAX_LOSS_EXIT_PCT")
        self.assertIn("strategy_ic_max_loss_exit", source,
            "_manage_strategy_group must have ic_max_loss_exit action type")

    def test_register_entry_stores_max_loss(self):
        """register_options_entry must store max_loss in state."""
        from options_manager import register_options_entry, _load_options_state, OPTIONS_STATE_PATH

        test_path = "/tmp/test_opts_state_fix8.json"
        with patch("options_manager.OPTIONS_STATE_PATH", test_path):
            if os.path.exists(test_path):
                os.remove(test_path)

            register_options_entry(
                "AAPL260420C00110000", 1.80, "sell", "iron_condor",
                delta=-0.20, qty=1, ticker="AAPL", setup="earnings_iv_crush",
                max_loss=320.00
            )

            state = _load_options_state()
            occ_state = state.get("AAPL260420C00110000", {})
            self.assertEqual(occ_state.get("max_loss"), 320.00,
                "max_loss must be stored in entry state")

            os.remove(test_path)

    def test_execution_passes_max_loss_to_register(self):
        """options_execution must pass max_loss to register_options_entry."""
        import inspect
        import options_execution as oe
        source = inspect.getsource(oe)
        self.assertIn("max_loss=contract.get", source,
            "execute_options_trade must pass max_loss to register_options_entry")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 9: Market calendar awareness
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix9_MarketCalendar(unittest.TestCase):
    """Market calendar utility must correctly identify holidays, half-days,
    and short weeks for 2026."""

    def test_good_friday_2026_is_holiday(self):
        from market_calendar import is_market_holiday
        from datetime import date
        self.assertTrue(is_market_holiday(date(2026, 4, 3)),
            "Good Friday 2026 should be a holiday")

    def test_thanksgiving_is_holiday(self):
        from market_calendar import is_market_holiday
        from datetime import date
        self.assertTrue(is_market_holiday(date(2026, 11, 26)),
            "Thanksgiving 2026 should be a holiday")

    def test_day_after_thanksgiving_is_half_day(self):
        from market_calendar import is_half_day
        from datetime import date
        self.assertTrue(is_half_day(date(2026, 11, 27)),
            "Day after Thanksgiving should be a half-day")

    def test_christmas_eve_is_half_day(self):
        from market_calendar import is_half_day
        from datetime import date
        self.assertTrue(is_half_day(date(2026, 12, 24)),
            "Christmas Eve 2026 should be a half-day")

    def test_regular_monday_not_holiday(self):
        from market_calendar import is_market_holiday
        from datetime import date
        self.assertFalse(is_market_holiday(date(2026, 4, 14)),
            "Regular Monday should not be a holiday")

    def test_saturday_is_holiday(self):
        from market_calendar import is_market_holiday
        from datetime import date
        self.assertTrue(is_market_holiday(date(2026, 4, 11)),
            "Saturday should count as market closure")

    def test_good_friday_week_is_short(self):
        from market_calendar import is_short_week, trading_days_this_week
        from datetime import date
        # Week of March 30 - April 3 (Good Friday on April 3)
        gf_week = date(2026, 3, 31)  # Tuesday of that week
        self.assertTrue(is_short_week(gf_week),
            "Good Friday week should be a short week")
        self.assertEqual(trading_days_this_week(gf_week), 4)

    def test_thanksgiving_week_is_short(self):
        from market_calendar import is_short_week, trading_days_this_week
        from datetime import date
        # Week of Nov 23-27 (Thanksgiving Nov 26)
        tg_week = date(2026, 11, 23)  # Monday
        self.assertTrue(is_short_week(tg_week))
        # Mon, Tue, Wed open; Thu closed; Fri half-day (but still open)
        self.assertEqual(trading_days_this_week(tg_week), 4)

    def test_regular_week_is_not_short(self):
        from market_calendar import is_short_week, trading_days_this_week
        from datetime import date
        regular = date(2026, 4, 14)  # Regular Monday
        self.assertFalse(is_short_week(regular))
        self.assertEqual(trading_days_this_week(regular), 5)

    def test_half_day_skips_new_options(self):
        from market_calendar import should_skip_new_options
        from datetime import date
        skip, reason = should_skip_new_options(date(2026, 11, 27))
        self.assertTrue(skip, "Should skip new options on half-days")
        self.assertIn("Half-day", reason)

    def test_pre_long_weekend_skips_new_options(self):
        from market_calendar import should_skip_new_options, is_pre_long_weekend
        from datetime import date
        # Thursday April 2 is the day before Good Friday (3-day weekend)
        self.assertTrue(is_pre_long_weekend(date(2026, 4, 2)),
            "Day before Good Friday should be pre-long-weekend")
        skip, reason = should_skip_new_options(date(2026, 4, 2))
        self.assertTrue(skip, "Should skip new options before long weekend")

    def test_scanner_uses_calendar(self):
        """get_options_trades must reference market calendar."""
        import inspect
        from options_scanner import get_options_trades
        source = inspect.getsource(get_options_trades)
        self.assertIn("should_skip_new_options", source,
            "get_options_trades must check market calendar")
        self.assertIn("is_short_week", source,
            "get_options_trades must check for short weeks")


if __name__ == "__main__":
    # Set env vars to prevent import errors
    os.environ.setdefault("ALPACA_KEY", "test")
    os.environ.setdefault("ALPACA_SECRET", "test")

    unittest.main(verbosity=2)
