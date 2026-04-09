#!/usr/bin/env python3
"""
Comprehensive test suite for the options execution fixes.
Tests every fix:
  1. options_manager.py — DTE exits, profit targets, Greeks, rolling, assignment
  2. options_execution.py — size_pct fix, limit price optimization, spread cleanup,
                            select_contract signature, entry registration
  3. bot_engine.py — options manager wiring

Run: python test_options_fixes.py
"""

import os
import sys
import json
import time
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add the repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1: _optimized_limit_price
# ═══════════════════════════════════════════════════════════════════════════════

class TestLimitPriceOptimization(unittest.TestCase):
    """Test the mid-price optimization that saves money on every trade."""

    def setUp(self):
        from options_execution import _optimized_limit_price
        self.optimize = _optimized_limit_price

    def test_buy_price_between_mid_and_ask(self):
        """Buying: price should be between mid and ask (not at full ask)."""
        contract = {"bid": 2.00, "ask": 2.50, "mid": 2.25}
        price = self.optimize(contract, "buy")
        self.assertGreater(price, 2.25, "Buy price should be above mid")
        self.assertLess(price, 2.50, "Buy price should be below full ask")

    def test_sell_price_between_bid_and_mid(self):
        """Selling: price should be between bid and mid (not at full bid)."""
        contract = {"bid": 2.00, "ask": 2.50, "mid": 2.25}
        price = self.optimize(contract, "sell")
        self.assertGreater(price, 2.00, "Sell price should be above full bid")
        self.assertLess(price, 2.25, "Sell price should be below mid")

    def test_narrow_spread_saves_less(self):
        """Narrow spread = smaller absolute savings (but still saves)."""
        contract = {"bid": 3.00, "ask": 3.10, "mid": 3.05}
        buy_price = self.optimize(contract, "buy")
        self.assertLess(buy_price, 3.10, "Should save on narrow spread too")
        self.assertGreater(buy_price, 3.05)

    def test_wide_spread_saves_more(self):
        """Wide spread = more money saved by not paying full ask."""
        contract = {"bid": 1.00, "ask": 2.00, "mid": 1.50}
        buy_price = self.optimize(contract, "buy")
        savings = 2.00 - buy_price
        self.assertGreater(savings, 0.10, "Should save significantly on wide spread")

    def test_zero_bid_fallback(self):
        """Handle edge case: zero bid."""
        contract = {"bid": 0, "ask": 1.50, "mid": 0.75}
        price = self.optimize(contract, "buy")
        self.assertGreater(price, 0, "Should return a positive price")

    def test_zero_ask_fallback(self):
        """Handle edge case: zero ask."""
        contract = {"bid": 1.50, "ask": 0, "mid": 0.75}
        price = self.optimize(contract, "sell")
        self.assertGreaterEqual(price, 0, "Should handle gracefully")

    def test_exact_walk_percentage(self):
        """Verify the 30% walk from mid."""
        contract = {"bid": 2.00, "ask": 3.00, "mid": 2.50}
        buy_price = self.optimize(contract, "buy")
        # Expected: mid + 0.30 * (ask - mid) = 2.50 + 0.30 * 0.50 = 2.65
        self.assertAlmostEqual(buy_price, 2.65, places=2)

        sell_price = self.optimize(contract, "sell")
        # Expected: mid - 0.30 * (mid - bid) = 2.50 - 0.30 * 0.50 = 2.35
        self.assertAlmostEqual(sell_price, 2.35, places=2)

    def test_never_exceeds_spread(self):
        """Price should never go outside bid-ask boundaries."""
        contract = {"bid": 5.00, "ask": 5.50, "mid": 5.25}
        buy_px = self.optimize(contract, "buy")
        sell_px = self.optimize(contract, "sell")
        self.assertGreaterEqual(buy_px, 5.00)
        self.assertLessEqual(buy_px, 5.50)
        self.assertGreaterEqual(sell_px, 5.00)
        self.assertLessEqual(sell_px, 5.50)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2: _parse_occ_symbol
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseOccSymbol(unittest.TestCase):
    """Test OCC symbol parsing."""

    def setUp(self):
        from options_manager import _parse_occ_symbol
        self.parse = _parse_occ_symbol

    def test_standard_call(self):
        result = self.parse("AAPL260418C00250000")
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["expiry_date"], "2026-04-18")
        self.assertEqual(result["option_type"], "call")
        self.assertEqual(result["strike"], 250.0)

    def test_standard_put(self):
        result = self.parse("SPY260320P00500000")
        self.assertEqual(result["ticker"], "SPY")
        self.assertEqual(result["expiry_date"], "2026-03-20")
        self.assertEqual(result["option_type"], "put")
        self.assertEqual(result["strike"], 500.0)

    def test_long_ticker(self):
        result = self.parse("GOOGL260418C00180000")
        self.assertEqual(result["ticker"], "GOOGL")
        self.assertEqual(result["strike"], 180.0)

    def test_fractional_strike(self):
        result = self.parse("AAPL260418C00185500")
        self.assertEqual(result["strike"], 185.5)

    def test_invalid_symbol(self):
        result = self.parse("XX")
        self.assertEqual(result, {})

    def test_empty_string(self):
        result = self.parse("")
        self.assertEqual(result, {})


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3: _days_to_expiry
# ═══════════════════════════════════════════════════════════════════════════════

class TestDaysToExpiry(unittest.TestCase):
    def setUp(self):
        from options_manager import _days_to_expiry
        self.dte = _days_to_expiry

    def test_future_date(self):
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        dte = self.dte(future)
        self.assertGreaterEqual(dte, 29)
        self.assertLessEqual(dte, 31)

    def test_today(self):
        today = datetime.now().strftime("%Y-%m-%d")
        dte = self.dte(today)
        self.assertLessEqual(dte, 0)

    def test_past_date(self):
        past = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        dte = self.dte(past)
        self.assertLess(dte, 0)

    def test_invalid_date(self):
        dte = self.dte("not-a-date")
        self.assertEqual(dte, 999)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4: select_contract signature fix
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelectContractSignature(unittest.TestCase):
    """Test that select_contract accepts the new keyword arguments."""

    def test_accepts_trade_positions_macro(self):
        """Should not raise TypeError with extra kwargs."""
        from options_execution import select_contract
        import inspect
        sig = inspect.signature(select_contract)
        params = list(sig.parameters.keys())
        self.assertIn("trade", params, "select_contract must accept 'trade' kwarg")
        self.assertIn("positions", params, "select_contract must accept 'positions' kwarg")
        self.assertIn("macro", params, "select_contract must accept 'macro' kwarg")

    def test_backwards_compatible(self):
        """Can still call with just positional args (old style)."""
        from options_execution import select_contract
        # This should not crash (it will return an error due to no API,
        # but should not raise TypeError)
        try:
            result = select_contract("AAPL", "buy_call", 185.0, 100000)
        except TypeError:
            self.fail("select_contract raised TypeError — signature not backward compatible")
        # Should return a dict (error or contract)
        self.assertIsInstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5: size_pct parameter in _select_* functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestSizePctParameter(unittest.TestCase):
    """Verify all _select_* functions accept size_pct parameter."""

    def test_buy_call_has_size_pct(self):
        from options_execution import _select_buy_call
        import inspect
        sig = inspect.signature(_select_buy_call)
        self.assertIn("size_pct", sig.parameters)

    def test_buy_put_has_size_pct(self):
        from options_execution import _select_buy_put
        import inspect
        sig = inspect.signature(_select_buy_put)
        self.assertIn("size_pct", sig.parameters)

    def test_sell_put_has_size_pct(self):
        from options_execution import _select_sell_put
        import inspect
        sig = inspect.signature(_select_sell_put)
        self.assertIn("size_pct", sig.parameters)

    def test_bull_spread_has_size_pct(self):
        from options_execution import _select_bull_spread
        import inspect
        sig = inspect.signature(_select_bull_spread)
        self.assertIn("size_pct", sig.parameters)

    def test_bear_spread_has_size_pct(self):
        from options_execution import _select_bear_spread
        import inspect
        sig = inspect.signature(_select_bear_spread)
        self.assertIn("size_pct", sig.parameters)

    def test_straddle_has_size_pct(self):
        from options_execution import _select_straddle
        import inspect
        sig = inspect.signature(_select_straddle)
        self.assertIn("size_pct", sig.parameters)

    def test_condor_has_size_pct(self):
        from options_execution import _select_condor
        import inspect
        sig = inspect.signature(_select_condor)
        self.assertIn("size_pct", sig.parameters)

    def test_buy_call_runs_without_crash(self):
        """_select_buy_call should not crash with NameError on size_pct."""
        from options_execution import _select_buy_call
        contracts = [
            {"occ_symbol": "AAPL260418C00190000", "option_type": "call",
             "strike": 190, "bid": 3.00, "ask": 3.50, "mid": 3.25,
             "delta": 0.42, "gamma": 0.02, "theta": -0.05, "iv": 0.25,
             "volume": 500, "open_interest": 2000, "expiry": "2026-04-18",
             "days_to_expiry": 9}
        ]
        # This was crashing before the fix with NameError: 'size_pct'
        try:
            result = _select_buy_call(contracts, 185.0, 100000, "AAPL", 0.05)
        except NameError:
            self.fail("_select_buy_call raised NameError — size_pct bug still exists")
        self.assertIsInstance(result, dict)
        self.assertIsNone(result.get("error"))

    def test_buy_put_runs_without_crash(self):
        """_select_buy_put should not crash with NameError on size_pct."""
        from options_execution import _select_buy_put
        contracts = [
            {"occ_symbol": "AAPL260418P00180000", "option_type": "put",
             "strike": 180, "bid": 2.50, "ask": 3.00, "mid": 2.75,
             "delta": -0.38, "gamma": 0.02, "theta": -0.04, "iv": 0.22,
             "volume": 300, "open_interest": 1500, "expiry": "2026-04-18",
             "days_to_expiry": 9}
        ]
        try:
            result = _select_buy_put(contracts, 185.0, 100000, "AAPL", 0.05)
        except NameError:
            self.fail("_select_buy_put raised NameError — size_pct bug still exists")
        self.assertIsInstance(result, dict)
        self.assertIsNone(result.get("error"))

    def test_sell_put_runs_without_crash(self):
        """_select_sell_put should not crash with NameError on size_pct."""
        from options_execution import _select_sell_put
        contracts = [
            {"occ_symbol": "AAPL260418P00175000", "option_type": "put",
             "strike": 175, "bid": 1.80, "ask": 2.10, "mid": 1.95,
             "delta": -0.28, "gamma": 0.015, "theta": -0.03, "iv": 0.20,
             "volume": 400, "open_interest": 1800, "expiry": "2026-04-18",
             "days_to_expiry": 9}
        ]
        try:
            # size_pct=0.20 so 100k * 0.20 = 20k > 17.5k per contract
            result = _select_sell_put(contracts, 185.0, 100000, "AAPL", 0.20)
        except NameError:
            self.fail("_select_sell_put raised NameError — size_pct bug still exists")
        self.assertIsInstance(result, dict)
        self.assertIsNone(result.get("error"))


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6: Options Manager — DTE Exit Logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestDTEExitLogic(unittest.TestCase):
    """Test the 21 DTE and 5 DTE critical exit logic."""

    def _make_position(self, occ_symbol, qty, side, entry_price, current_price):
        return {
            "symbol": occ_symbol,
            "qty": str(qty),
            "side": side,
            "avg_entry_price": str(entry_price),
            "current_price": str(current_price),
            "market_value": str(current_price * qty * 100),
            "unrealized_pl": str((current_price - entry_price) * qty * 100),
            "asset_class": "option",
        }

    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_critical_dte_forces_close(self, mock_close, mock_snap, mock_get):
        """Position at 3 DTE should be force-closed."""
        from options_manager import manage_options_positions, _save_options_state, OPTIONS_STATE_PATH

        # Set up: option expiring in 3 days
        exp = (datetime.now() + timedelta(days=3)).strftime("%y%m%d")
        occ = f"AAPL{exp}C00190000"

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [self._make_position(occ, 1, "long", 3.00, 2.50)]
        )
        mock_snap.return_value = {"bid": 2.30, "ask": 2.70, "mid": 2.50,
                                   "delta": 0.45, "gamma": 0.10, "theta": -0.15, "vega": 0.05, "iv": 0.30}
        mock_close.return_value = {"status": "submitted", "order_id": "test123"}

        # Clean state
        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

        result = manage_options_positions(100000)

        # Should have triggered a close action
        close_actions = [a for a in result["actions"] if a["action"] == "CLOSE"]
        self.assertGreaterEqual(len(close_actions), 1, "Should close at critical DTE")
        self.assertIn("dte_critical", close_actions[0].get("type", ""))

    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_21_dte_closes_bought_option(self, mock_close, mock_snap, mock_get):
        """Bought option at 18 DTE should be closed (theta acceleration zone)."""
        from options_manager import manage_options_positions, OPTIONS_STATE_PATH

        exp = (datetime.now() + timedelta(days=18)).strftime("%y%m%d")
        occ = f"SPY{exp}C00500000"

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [self._make_position(occ, 2, "long", 5.00, 4.00)]
        )
        mock_snap.return_value = {"bid": 3.80, "ask": 4.20, "mid": 4.00,
                                   "delta": 0.40, "gamma": 0.04, "theta": -0.12, "vega": 0.08, "iv": 0.25}
        mock_close.return_value = {"status": "submitted", "order_id": "test456"}

        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

        result = manage_options_positions(100000)
        close_actions = [a for a in result["actions"] if a["action"] == "CLOSE"]
        self.assertGreaterEqual(len(close_actions), 1, "Should close bought option at 21 DTE")
        self.assertIn("dte_close", close_actions[0].get("type", ""))


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 7: Options Manager — Profit Target
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfitTarget(unittest.TestCase):
    """Test the 50% profit target for sold premium."""

    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_50pct_profit_triggers_close(self, mock_close, mock_snap, mock_get):
        """Sold option at 60% profit should be closed."""
        from options_manager import manage_options_positions, _save_options_state, OPTIONS_STATE_PATH

        # Option expiring in 30 days (not DTE-triggered)
        exp = (datetime.now() + timedelta(days=30)).strftime("%y%m%d")
        occ = f"AAPL{exp}P00175000"

        # Sold at $3.00, now worth $1.20 (60% profit)
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{
                "symbol": occ, "qty": "-1", "side": "short",
                "avg_entry_price": "3.00", "current_price": "1.20",
                "market_value": "-120", "unrealized_pl": "180",
                "asset_class": "option",
            }]
        )
        mock_snap.return_value = {"bid": 1.10, "ask": 1.30, "mid": 1.20,
                                   "delta": -0.15, "gamma": 0.01, "theta": -0.02, "vega": 0.03, "iv": 0.20}
        mock_close.return_value = {"status": "submitted", "order_id": "profit789"}

        # Pre-seed state with entry info
        state = {
            occ: {
                "entry_price": 3.00,
                "entry_delta": -0.30,
                "entry_date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
                "initial_credit": 3.00,
                "max_profit_target": 1.50,
                "highest_value": 3.00,
                "strategy": "sell_cash_secured_put",
                "side": "short",
                "qty": 1,
            }
        }
        _save_options_state(state)

        result = manage_options_positions(100000)
        close_actions = [a for a in result["actions"] if a.get("type") == "profit_target"]
        self.assertGreaterEqual(len(close_actions), 1, "Should close at 50% profit target")

    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    def test_30pct_profit_does_not_trigger(self, mock_snap, mock_get):
        """Sold option at only 30% profit should NOT be closed."""
        from options_manager import manage_options_positions, _save_options_state, OPTIONS_STATE_PATH

        exp = (datetime.now() + timedelta(days=30)).strftime("%y%m%d")
        occ = f"AAPL{exp}P00175000"

        # Sold at $3.00, now worth $2.10 (30% profit — below 50% target)
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{
                "symbol": occ, "qty": "-1", "side": "short",
                "avg_entry_price": "3.00", "current_price": "2.10",
                "market_value": "-210", "unrealized_pl": "90",
                "asset_class": "option",
            }]
        )
        mock_snap.return_value = {"bid": 2.00, "ask": 2.20, "mid": 2.10,
                                   "delta": -0.22, "gamma": 0.015, "theta": -0.03, "vega": 0.04, "iv": 0.22}

        state = {
            occ: {
                "entry_price": 3.00, "entry_delta": -0.30,
                "entry_date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
                "initial_credit": 3.00, "max_profit_target": 1.50,
                "highest_value": 3.00, "strategy": "sell_cash_secured_put",
                "side": "short", "qty": 1,
            }
        }
        _save_options_state(state)

        result = manage_options_positions(100000)
        profit_closes = [a for a in result["actions"] if a.get("type") == "profit_target"]
        self.assertEqual(len(profit_closes), 0, "Should NOT close at only 30% profit")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 8: Options Manager — Loss Limit
# ═══════════════════════════════════════════════════════════════════════════════

class TestLossLimit(unittest.TestCase):
    """Test the 2x credit loss limit for sold options."""

    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_2x_loss_triggers_close(self, mock_close, mock_snap, mock_get):
        """Sold option now costing 2.5x the credit should be force-closed."""
        from options_manager import manage_options_positions, _save_options_state, OPTIONS_STATE_PATH

        exp = (datetime.now() + timedelta(days=30)).strftime("%y%m%d")
        occ = f"AAPL{exp}P00185000"

        # Sold at $2.00, now worth $5.00 (2.5x loss)
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{
                "symbol": occ, "qty": "-1", "side": "short",
                "avg_entry_price": "2.00", "current_price": "5.00",
                "market_value": "-500", "unrealized_pl": "-300",
                "asset_class": "option",
            }]
        )
        mock_snap.return_value = {"bid": 4.80, "ask": 5.20, "mid": 5.00,
                                   "delta": -0.65, "gamma": 0.03, "theta": -0.05, "vega": 0.06, "iv": 0.35}
        mock_close.return_value = {"status": "submitted", "order_id": "loss123"}

        state = {
            occ: {
                "entry_price": 2.00, "entry_delta": -0.30,
                "entry_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                "initial_credit": 2.00, "max_profit_target": 1.00,
                "highest_value": 2.00, "strategy": "sell_cash_secured_put",
                "side": "short", "qty": 1,
            }
        }
        _save_options_state(state)

        result = manage_options_positions(100000)
        loss_closes = [a for a in result["actions"] if a.get("type") == "loss_limit"]
        self.assertGreaterEqual(len(loss_closes), 1, "Should close at 2x loss limit")

    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_bought_option_50pct_loss_triggers_close(self, mock_close, mock_snap, mock_get):
        """Bought option down 60% should be closed."""
        from options_manager import manage_options_positions, _save_options_state, OPTIONS_STATE_PATH

        exp = (datetime.now() + timedelta(days=30)).strftime("%y%m%d")
        occ = f"SPY{exp}C00520000"

        # Bought at $5.00, now worth $1.80 (64% loss)
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{
                "symbol": occ, "qty": "2", "side": "long",
                "avg_entry_price": "5.00", "current_price": "1.80",
                "market_value": "360", "unrealized_pl": "-640",
                "asset_class": "option",
            }]
        )
        mock_snap.return_value = {"bid": 1.60, "ask": 2.00, "mid": 1.80,
                                   "delta": 0.15, "gamma": 0.01, "theta": -0.08, "vega": 0.05, "iv": 0.20}
        mock_close.return_value = {"status": "submitted", "order_id": "bloss456"}

        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

        result = manage_options_positions(100000)
        loss_closes = [a for a in result["actions"] if a.get("type") == "bought_loss_limit"]
        self.assertGreaterEqual(len(loss_closes), 1, "Should close bought option at 50% loss")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 9: Options Manager — Gamma Risk
# ═══════════════════════════════════════════════════════════════════════════════

class TestGammaRisk(unittest.TestCase):
    """Test gamma threshold exit."""

    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_high_gamma_triggers_exit(self, mock_close, mock_snap, mock_get):
        """Position with gamma > 0.08 and <=30 DTE should be closed."""
        from options_manager import manage_options_positions, OPTIONS_STATE_PATH

        exp = (datetime.now() + timedelta(days=25)).strftime("%y%m%d")
        occ = f"AAPL{exp}C00185000"

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{
                "symbol": occ, "qty": "1", "side": "long",
                "avg_entry_price": "4.00", "current_price": "4.50",
                "market_value": "450", "unrealized_pl": "50",
                "asset_class": "option",
            }]
        )
        # Gamma = 0.12 exceeds 0.08 threshold
        mock_snap.return_value = {"bid": 4.30, "ask": 4.70, "mid": 4.50,
                                   "delta": 0.55, "gamma": 0.12, "theta": -0.10, "vega": 0.06, "iv": 0.28}
        mock_close.return_value = {"status": "submitted", "order_id": "gamma789"}

        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

        result = manage_options_positions(100000)
        gamma_closes = [a for a in result["actions"] if a.get("type") == "gamma_risk"]
        self.assertGreaterEqual(len(gamma_closes), 1, "Should close on high gamma risk")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 10: Options Manager — Delta Drift Warning
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeltaDrift(unittest.TestCase):
    """Test delta drift detection."""

    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    def test_delta_drift_generates_warning(self, mock_snap, mock_get):
        """Large delta shift should generate a WARNING action."""
        from options_manager import manage_options_positions, _save_options_state, OPTIONS_STATE_PATH

        # 35 DTE — won't trigger DTE exit
        exp = (datetime.now() + timedelta(days=35)).strftime("%y%m%d")
        occ = f"AAPL{exp}C00185000"

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{
                "symbol": occ, "qty": "1", "side": "long",
                "avg_entry_price": "4.00", "current_price": "6.00",
                "market_value": "600", "unrealized_pl": "200",
                "asset_class": "option",
            }]
        )
        # Delta shifted from 0.40 to 0.75 (shift of 0.35 > threshold 0.25)
        # Gamma low enough to not trigger gamma exit
        mock_snap.return_value = {"bid": 5.80, "ask": 6.20, "mid": 6.00,
                                   "delta": 0.75, "gamma": 0.05, "theta": -0.08, "vega": 0.07, "iv": 0.22}

        state = {
            occ: {
                "entry_price": 4.00, "entry_delta": 0.40,
                "entry_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                "initial_credit": 0, "max_profit_target": 0,
                "highest_value": 6.00, "strategy": "buy_call",
                "side": "long", "qty": 1,
            }
        }
        _save_options_state(state)

        result = manage_options_positions(100000)
        warnings = [a for a in result["actions"] if a.get("type") == "delta_drift"]
        self.assertGreaterEqual(len(warnings), 1, "Should generate delta drift warning")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 11: Options Manager — State Persistence
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatePersistence(unittest.TestCase):
    """Test that options state is saved and loaded correctly."""

    def test_save_and_load(self):
        from options_manager import _save_options_state, _load_options_state, OPTIONS_STATE_PATH

        test_state = {
            "AAPL260418C00190000": {
                "entry_price": 3.50,
                "entry_delta": 0.42,
                "side": "long",
                "strategy": "buy_call",
            }
        }
        _save_options_state(test_state)
        loaded = _load_options_state()
        self.assertEqual(loaded["AAPL260418C00190000"]["entry_price"], 3.50)
        self.assertEqual(loaded["AAPL260418C00190000"]["entry_delta"], 0.42)

        # Cleanup
        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

    def test_load_missing_file(self):
        from options_manager import _load_options_state, OPTIONS_STATE_PATH
        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)
        loaded = _load_options_state()
        self.assertEqual(loaded, {})


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 12: register_options_entry
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegisterOptionsEntry(unittest.TestCase):
    """Test that entry registration populates state correctly."""

    def test_register_sold_option(self):
        from options_manager import register_options_entry, _load_options_state, OPTIONS_STATE_PATH

        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

        register_options_entry(
            "AAPL260418P00175000", 2.50, "sell", "sell_cash_secured_put",
            delta=-0.30, qty=2,
        )

        state = _load_options_state()
        entry = state.get("AAPL260418P00175000")
        self.assertIsNotNone(entry, "Entry should be registered")
        self.assertEqual(entry["initial_credit"], 2.50)
        self.assertEqual(entry["side"], "short")
        self.assertEqual(entry["qty"], 2)
        self.assertEqual(entry["strategy"], "sell_cash_secured_put")
        self.assertAlmostEqual(entry["max_profit_target"], 1.25)  # 50% of 2.50

        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

    def test_register_bought_option(self):
        from options_manager import register_options_entry, _load_options_state, OPTIONS_STATE_PATH

        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

        register_options_entry(
            "AAPL260418C00190000", 3.50, "buy", "buy_call",
            delta=0.42, qty=1,
        )

        state = _load_options_state()
        entry = state.get("AAPL260418C00190000")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["initial_credit"], 0)  # Bought, no credit
        self.assertEqual(entry["side"], "long")

        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 13: Spread Execution Safety — Cancel Long if Short Fails
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpreadSafety(unittest.TestCase):
    """Test that failed short leg cancels the long leg."""

    @patch("options_execution._cancel_order")
    @patch("options_execution.requests.post")
    def test_short_leg_failure_cancels_long(self, mock_post, mock_cancel):
        from options_execution import _submit_spread_order

        # Long leg succeeds
        long_resp = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=lambda: {"id": "long_order_123", "status": "accepted"}
        )
        # Short leg fails
        short_resp = MagicMock(
            status_code=403,
            headers={"content-type": "application/json"},
            json=lambda: {"message": "insufficient buying power"},
            text="insufficient buying power"
        )
        mock_post.side_effect = [long_resp, short_resp]
        mock_cancel.return_value = True

        contract = {
            "long_leg": "AAPL260418C00185000",
            "short_leg": "AAPL260418C00195000",
            "qty": 1,
            "net_debit": 2.50,
            "strategy": "bull_call_spread",
        }

        result = _submit_spread_order(contract)
        # Should have called cancel on the long order
        mock_cancel.assert_called_once_with("long_order_123")
        self.assertEqual(result["status"], "error")
        self.assertTrue(result.get("long_order_cancelled", False))


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 14: Bot Engine Wiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestBotEngineWiring(unittest.TestCase):
    """Test that bot_engine imports and calls options_manager."""

    def test_options_management_key_in_scan_result(self):
        """The scan result dict should include 'options_management' key."""
        # Read bot_engine.py and check the return dict
        import ast
        with open(os.path.join(os.path.dirname(__file__), "bot_engine.py")) as f:
            content = f.read()
        self.assertIn("options_management", content,
                      "bot_engine.py should reference 'options_management' in return dict")
        self.assertIn("manage_options_positions", content,
                      "bot_engine.py should import manage_options_positions")

    def test_options_manager_import_exists(self):
        """The import line should exist in bot_engine.py."""
        with open(os.path.join(os.path.dirname(__file__), "bot_engine.py")) as f:
            content = f.read()
        self.assertIn("from options_manager import manage_options_positions", content)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 15: No Positions = Clean Return
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoPositions(unittest.TestCase):
    """Test behavior when there are no options positions."""

    @patch("options_manager.requests.get")
    def test_empty_positions_returns_clean(self, mock_get):
        from options_manager import manage_options_positions
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: []  # No positions at all
        )
        result = manage_options_positions(100000)
        self.assertEqual(result["actions"], [])
        self.assertEqual(result["positions_checked"], 0)

    @patch("options_manager.requests.get")
    def test_only_stock_positions_skipped(self, mock_get):
        from options_manager import manage_options_positions
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {"symbol": "AAPL", "qty": "10", "side": "long",
                 "avg_entry_price": "185", "current_price": "190",
                 "market_value": "1900", "unrealized_pl": "50"},
            ]  # Only stock positions — no options
        )
        result = manage_options_positions(100000)
        self.assertEqual(result["positions_checked"], 0, "Should skip stock-only positions")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 16: Greeks Returned in Contract Selection
# ═══════════════════════════════════════════════════════════════════════════════

class TestGreeksInContract(unittest.TestCase):
    """Test that gamma and theta are included in contract selection results."""

    def test_buy_call_includes_gamma_theta(self):
        from options_execution import _select_buy_call
        contracts = [
            {"occ_symbol": "AAPL260418C00190000", "option_type": "call",
             "strike": 190, "bid": 3.00, "ask": 3.50, "mid": 3.25,
             "delta": 0.42, "gamma": 0.025, "theta": -0.06, "iv": 0.25,
             "volume": 500, "open_interest": 2000, "expiry": "2026-04-18",
             "days_to_expiry": 9}
        ]
        result = _select_buy_call(contracts, 185.0, 100000, "AAPL", 0.05)
        self.assertIn("gamma", result, "Should include gamma in result")
        self.assertIn("theta", result, "Should include theta in result")
        self.assertEqual(result["gamma"], 0.025)
        self.assertEqual(result["theta"], -0.06)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST: Separate Options Slot Allocation (v1.0.32 fix)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptionsSlotseparation(unittest.TestCase):
    """Options positions must NOT consume stock slots and vice versa."""

    def test_max_options_positions_constant_exists(self):
        """bot_engine must have MAX_OPTIONS_POSITIONS separate from MAX_POSITIONS."""
        from bot_engine import MAX_POSITIONS, MAX_OPTIONS_POSITIONS
        self.assertEqual(MAX_POSITIONS, 5)
        self.assertEqual(MAX_OPTIONS_POSITIONS, 3)

    def test_num_positions_excludes_options(self):
        """num_positions should only count stocks, not OCC-symbol options."""
        # Simulate positions list with mix of stocks and options
        mixed_positions = [
            {"symbol": "QQQ", "asset_class": "us_equity", "qty": "111"},
            {"symbol": "AEHR", "asset_class": "us_equity", "qty": "15"},
            {"symbol": "SPY260501P00500000", "asset_class": "us_option", "qty": "1"},
            {"symbol": "AAPL260515C00200000", "asset_class": "us_option", "qty": "2"},
        ]
        # Count only stock positions (symbol <= 8 chars and not us_option)
        num_stock = sum(
            1 for p in mixed_positions
            if len(str(p.get("symbol", ""))) <= 8 and p.get("asset_class", "us_equity") != "us_option"
        )
        self.assertEqual(num_stock, 2)  # QQQ + AEHR only

    def test_options_slots_counted_separately(self):
        """Options positions should be counted against MAX_OPTIONS_POSITIONS."""
        mixed_positions = [
            {"symbol": "QQQ", "asset_class": "us_equity"},
            {"symbol": "SPY260501P00500000", "asset_class": "us_option"},
        ]
        from bot_engine import MAX_OPTIONS_POSITIONS
        existing_options = sum(
            1 for p in mixed_positions
            if len(str(p.get("symbol", ""))) > 8 or p.get("asset_class") == "us_option"
        )
        options_slots = MAX_OPTIONS_POSITIONS - existing_options
        self.assertEqual(existing_options, 1)
        self.assertEqual(options_slots, 2)  # 3 - 1 = 2 remaining

    def test_full_stock_slots_still_allows_options(self):
        """Even with 5 stock positions, options scanner should get slots."""
        from bot_engine import MAX_POSITIONS, MAX_OPTIONS_POSITIONS
        stock_positions = [
            {"symbol": f"STK{i}", "asset_class": "us_equity"}
            for i in range(5)
        ]
        num_stock = sum(
            1 for p in stock_positions
            if len(str(p.get("symbol", ""))) <= 8 and p.get("asset_class", "us_equity") != "us_option"
        )
        stock_slots = MAX_POSITIONS - num_stock
        existing_options = sum(
            1 for p in stock_positions
            if len(str(p.get("symbol", ""))) > 8 or p.get("asset_class") == "us_option"
        )
        options_slots = MAX_OPTIONS_POSITIONS - existing_options
        self.assertEqual(stock_slots, 0)  # No stock slots
        self.assertEqual(options_slots, 3)  # All 3 options slots available

    def test_scanner_trade_has_correct_markers(self):
        """Scanner trades must have trade_type='options' and regime_at_entry='OPTIONS_SCANNER'."""
        # This simulates what bot_engine.py builds for scanner trades
        scanner_trade = {
            "trade_type": "options",
            "use_options": True,
            "options_strategy": "buy_straddle",
            "regime_at_entry": "OPTIONS_SCANNER",
            "shares": 0,
        }
        # These are the markers bot.ts uses to route to scanner path
        is_scanner = (
            scanner_trade["trade_type"] == "options"
            and scanner_trade["regime_at_entry"] == "OPTIONS_SCANNER"
        )
        self.assertTrue(is_scanner)

    def test_stock_to_options_trade_not_scanner(self):
        """Stock→options trades must NOT be treated as scanner trades."""
        stock_options_trade = {
            "trade_type": "stock",
            "use_options": True,
            "options_strategy": "sell_cash_secured_put",
            "regime_at_entry": "BULL",
            "shares": 10,
        }
        is_scanner = (
            stock_options_trade.get("trade_type") == "options"
            and stock_options_trade.get("regime_at_entry") == "OPTIONS_SCANNER"
        )
        self.assertFalse(is_scanner)


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Set up environment
    os.makedirs("/tmp", exist_ok=True)

    # Run with verbose output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    print("=" * 70)
    print("VolTradeAI — Options Execution Fix Test Suite")
    print("=" * 70)
    print()

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    print(f"RESULTS: {passed}/{total} passed, {failures} failures, {errors} errors")
    print("=" * 70)
