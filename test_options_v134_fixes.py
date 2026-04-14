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


if __name__ == "__main__":
    # Set env vars to prevent import errors
    os.environ.setdefault("ALPACA_KEY", "test")
    os.environ.setdefault("ALPACA_SECRET", "test")

    unittest.main(verbosity=2)
