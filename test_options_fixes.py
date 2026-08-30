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
import inspect
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

    def test_sell_put_stretch_mode_sizing_matches_its_own_budget(self):
        """
        BUG FIX 2026-07-29 regression: when no strike fits the normal
        size_pct/2%-floor budget, _select_sell_put "stretches" to the
        smallest available strike if it's within 20% of equity (the
        2026-05-22 GRACEFUL DEGRADATION branch) — but the final
        max_contracts calc used to divide by the ORIGINAL tiny budget
        instead of the stretched one, guaranteeing max_contracts == 0 and
        an "Not enough capital" error on every single stretch-mode trade.

        Equity $110k, size_pct 0.005 (0.55% — a plausible Kelly-scaled
        value after several sub-1.0 scalars compound): normal budget =
        max(110000*0.005, 110000*0.02) = $2,200. Only strike available is
        $55 (needs $5,500) — exceeds $2,200 but is well inside the 20%
        stretch ceiling ($22,000), so stretch mode should fire and
        succeed with 1 contract, not fail.
        """
        from options_execution import _select_sell_put
        contracts = [
            {"occ_symbol": "PYPL260918P00055000", "option_type": "put",
             "strike": 55.0, "bid": 1.10, "ask": 1.30, "mid": 1.20,
             "delta": -0.29, "gamma": 0.02, "theta": -0.03, "iv": 0.35,
             "volume": 200, "open_interest": 900, "expiry": "2026-09-18",
             "days_to_expiry": 51}
        ]
        result = _select_sell_put(contracts, 68.0, 110000, "PYPL", 0.005)
        self.assertIsNone(
            result.get("error"),
            f"stretch-mode CSP should succeed, got: {result.get('error')}",
        )
        self.assertEqual(result.get("qty"), 1)
        self.assertEqual(result.get("strike"), 55.0)

    def test_sell_put_rejects_when_cash_available_below_equity_budget(self):
        """
        LIVE-CAPITAL CAP 2026-07-31 regression: without cash_available,
        _select_sell_put sizes purely off equity * size_pct and has no idea
        the account's real uncommitted cash is far lower — Alpaca then
        rejects the order live. Reproduces the exact live shape confirmed
        via /api/diag/audit?type=T2-FAIL 2026-07-30: "TLT: Alpaca rejected:
        insufficient options buying power for cash-secured put (required:
        7889.01, available: 362.36)" — repeated every scan cycle for hours
        because the equity-based budget never changed.

        Equity $105,397, size_pct 0.08 (8% MAX_OPTIONS_PCT) => equity-based
        budget ~$8,432, comfortably covering a $78 strike ($7,800/contract).
        With cash_available=362.36 (the live figure from the audit log),
        the fix must cap the budget below every available strike and fail
        CLEANLY here — instead of Alpaca finding out after submission.
        """
        from options_execution import _select_sell_put
        contracts = [
            {"occ_symbol": "TLT260918P00078000", "option_type": "put",
             "strike": 78.0, "bid": 1.10, "ask": 1.30, "mid": 1.20,
             "delta": -0.29, "gamma": 0.02, "theta": -0.03, "iv": 0.20,
             "volume": 500, "open_interest": 2000, "expiry": "2026-09-18",
             "days_to_expiry": 49}
        ]
        result = _select_sell_put(contracts, 82.0, 105397, "TLT", 0.08,
                                   cash_available=362.36)
        self.assertIsNotNone(
            result.get("error"),
            "cash-capped CSP should fail cleanly, not return a contract "
            "Alpaca will reject",
        )
        self.assertIn("cash_available", result["error"])
        # CSP CAPITAL ALLOCATION counterfactual logging (open_questions.md,
        # 2026-07-28/2026-08-06): the caller (server/bot.ts) needs the
        # resolved underlying price to log a shadow_portfolio record for
        # this rejection — it has no other zero-API-call way to get it.
        self.assertEqual(result.get("price"), 82.0)

    def test_sell_put_unaffected_by_generous_cash_available(self):
        """cash_available higher than the equity-based budget must not
        restrict anything below the pre-existing behavior."""
        from options_execution import _select_sell_put
        contracts = [
            {"occ_symbol": "AAPL260418P00175000", "option_type": "put",
             "strike": 175, "bid": 1.80, "ask": 2.10, "mid": 1.95,
             "delta": -0.28, "gamma": 0.015, "theta": -0.03, "iv": 0.20,
             "volume": 400, "open_interest": 1800, "expiry": "2026-04-18",
             "days_to_expiry": 9}
        ]
        result = _select_sell_put(contracts, 185.0, 100000, "AAPL", 0.20,
                                   cash_available=1_000_000)
        self.assertIsNone(result.get("error"))
        self.assertEqual(result.get("strike"), 175)

    def test_sell_put_stretch_mode_also_capped_by_cash_available(self):
        """The 20%-of-equity stretch ceiling must also respect
        cash_available — otherwise stretch mode picks a strike the account
        still cannot secure, and Alpaca rejects it exactly as before."""
        from options_execution import _select_sell_put
        contracts = [
            {"occ_symbol": "PYPL260918P00055000", "option_type": "put",
             "strike": 55.0, "bid": 1.10, "ask": 1.30, "mid": 1.20,
             "delta": -0.29, "gamma": 0.02, "theta": -0.03, "iv": 0.35,
             "volume": 200, "open_interest": 900, "expiry": "2026-09-18",
             "days_to_expiry": 51}
        ]
        # Same inputs as the stretch-mode-succeeds test above, but with
        # cash_available far below the $5,500 the stretched strike needs.
        result = _select_sell_put(contracts, 68.0, 110000, "PYPL", 0.005,
                                   cash_available=400.0)
        self.assertIsNotNone(
            result.get("error"),
            "stretch mode must not exceed cash_available",
        )
        self.assertEqual(result.get("price"), 68.0)

    def test_sell_put_no_otm_strikes_gives_honest_error_not_zero_strike(self):
        """
        BUG FIX 2026-08-23 regression: when the fetched chain has zero puts
        struck at or below the current price, `puts` (the strike<=price
        candidate list) is empty — `min(..., default=0)` used to silently
        paper over that, producing "smallest strike $0 needs $0 ... but max
        affordable per-position is $X ... Underlying too expensive" (seen
        live on KORU via /api/diag/audit?type=T2-FAIL: "smallest strike $0
        needs $0, but max affordable per-position is $2,189"). That message
        is false — the underlying wasn't too expensive, the chain simply had
        no OTM/ATM puts. This must now fail with an honest "no OTM puts
        available" message that names the real cause, and must never
        mention a $0 strike.
        """
        from options_execution import _select_sell_put
        # Every put in the chain is struck ABOVE the current price ($19.89,
        # matching the live KORU case) — none satisfy `strike <= price`.
        contracts = [
            {"occ_symbol": "KORU260918P00025000", "option_type": "put",
             "strike": 25.0, "bid": 5.10, "ask": 5.40, "mid": 5.25,
             "delta": -0.65, "gamma": 0.02, "theta": -0.03, "iv": 0.55,
             "volume": 50, "open_interest": 200, "expiry": "2026-09-18",
             "days_to_expiry": 37},
        ]
        result = _select_sell_put(contracts, 19.89, 109462, "KORU", 0.0136)
        self.assertIsNotNone(result.get("error"))
        self.assertNotIn("$0", result["error"])
        self.assertNotIn("too expensive", result["error"])
        self.assertIn("No OTM puts available", result["error"])


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

    @patch("ml_model_v2.track_fill")
    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_critical_dte_forces_close(self, mock_close, mock_snap, mock_get, mock_track_fill):
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

    @patch("ml_model_v2.track_fill")
    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_21_dte_closes_bought_option(self, mock_close, mock_snap, mock_get, mock_track_fill):
        """Bought option at 18 DTE should be closed (theta acceleration zone)."""
        from options_manager import manage_options_positions, _save_options_state, OPTIONS_STATE_PATH

        exp = (datetime.now() + timedelta(days=18)).strftime("%y%m%d")
        occ = f"SPY{exp}C00500000"

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [self._make_position(occ, 2, "long", 5.00, 4.00)]
        )
        mock_snap.return_value = {"bid": 3.80, "ask": 4.20, "mid": 4.00,
                                   "delta": 0.40, "gamma": 0.04, "theta": -0.12, "vega": 0.08, "iv": 0.25}
        mock_close.return_value = {"status": "submitted", "order_id": "test456"}

        # Pre-seed state with entry_timestamp > 60 min ago (v1.0.34: MIN_HOLD gate)
        _save_options_state({
            occ: {
                "entry_price": 5.00, "entry_delta": 0.50, "side": "long",
                "entry_date": "2026-04-12",
                "entry_timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "strategy": "buy_call", "highest_value": 5.00, "qty": 2,
            }
        })

        result = manage_options_positions(100000)
        close_actions = [a for a in result["actions"] if a["action"] == "CLOSE"]
        self.assertGreaterEqual(len(close_actions), 1, "Should close bought option at 21 DTE")
        self.assertIn("dte_close", close_actions[0].get("type", ""))


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 7: Options Manager — Profit Target
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfitTarget(unittest.TestCase):
    """Test the 50% profit target for sold premium."""

    @patch("ml_model_v2.track_fill")
    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_50pct_profit_triggers_close(self, mock_close, mock_snap, mock_get, mock_track_fill):
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

        # KNOWN BROKEN #12(c): this CLOSE must now be recorded into
        # trade_feedback with the real (Alpaca unrealized_pl-derived) pnl_pct,
        # not silently dropped.
        mock_track_fill.assert_called_once()
        fill_payload = mock_track_fill.call_args[0][0]
        self.assertEqual(fill_payload["ticker"], "AAPL")
        self.assertEqual(fill_payload["exit_reason"], "profit_target")
        self.assertAlmostEqual(fill_payload["exit_context"]["pnl_pct"], 60.0, places=1)

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

    @patch("ml_model_v2.track_fill")
    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_2x_loss_triggers_close(self, mock_close, mock_snap, mock_get, mock_track_fill):
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

    @patch("ml_model_v2.track_fill")
    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_bought_option_50pct_loss_triggers_close(self, mock_close, mock_snap, mock_get, mock_track_fill):
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

        # Pre-seed state with entry_timestamp > 60 min ago (v1.0.34: MIN_HOLD gate)
        _save_options_state({
            occ: {
                "entry_price": 5.00, "entry_delta": 0.50, "side": "long",
                "entry_date": "2026-04-12",
                "entry_timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "strategy": "buy_call", "highest_value": 5.00, "qty": 2,
            }
        })

        result = manage_options_positions(100000)
        loss_closes = [a for a in result["actions"] if a.get("type") == "bought_loss_limit"]
        self.assertGreaterEqual(len(loss_closes), 1, "Should close bought option at 50% loss")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 9: Options Manager — Gamma Risk
# ═══════════════════════════════════════════════════════════════════════════════

class TestGammaRisk(unittest.TestCase):
    """Test gamma threshold exit."""

    @patch("ml_model_v2.track_fill")
    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_high_gamma_triggers_exit(self, mock_close, mock_snap, mock_get, mock_track_fill):
        """Position with gamma > 0.08 and <=30 DTE should be closed."""
        from options_manager import manage_options_positions, _save_options_state, OPTIONS_STATE_PATH

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

        # Pre-seed state with entry_timestamp > 60 min ago (v1.0.34: MIN_HOLD gate)
        _save_options_state({
            occ: {
                "entry_price": 4.00, "entry_delta": 0.50, "side": "long",
                "entry_date": "2026-04-12",
                "entry_timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "strategy": "buy_call", "highest_value": 4.50, "qty": 1,
            }
        })

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

    @patch("ml_model_v2.track_fill")
    def test_register_sold_option(self, mock_track_fill):
        from options_manager import register_options_entry, _load_options_state, OPTIONS_STATE_PATH

        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

        register_options_entry(
            "AAPL260418P00175000", 2.50, "sell", "sell_cash_secured_put",
            delta=-0.30, qty=2, ticker="AAPL",
        )

        state = _load_options_state()
        entry = state.get("AAPL260418P00175000")
        self.assertIsNotNone(entry, "Entry should be registered")
        self.assertEqual(entry["initial_credit"], 2.50)
        self.assertEqual(entry["side"], "short")
        self.assertEqual(entry["qty"], 2)
        self.assertEqual(entry["strategy"], "sell_cash_secured_put")
        self.assertAlmostEqual(entry["max_profit_target"], 1.25)  # 50% of 2.50

        # KNOWN BROKEN #12(c): a standalone (non-multi-leg) entry must be
        # recorded into trade_feedback so its eventual exit has something
        # to match against.
        mock_track_fill.assert_called_once()
        fill_payload = mock_track_fill.call_args[0][0]
        self.assertEqual(fill_payload["ticker"], "AAPL")
        self.assertEqual(fill_payload["side"], "sell")
        self.assertEqual(fill_payload["fill_price"], 2.50)
        self.assertIsNone(fill_payload.get("exit_context"), "must be an ENTRY, not an exit")

        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

    @patch("ml_model_v2.track_fill")
    def test_register_bought_option(self, mock_track_fill):
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
#  TEST 13: Native Mleg Order Submission
# ═══════════════════════════════════════════════════════════════════════════════

class TestNativeMlegOrders(unittest.TestCase):
    """Test that all multi-leg strategies use Alpaca native mleg API."""

    def test_mleg_function_exists(self):
        """_submit_mleg_order must exist."""
        from options_execution import _submit_mleg_order
        self.assertTrue(callable(_submit_mleg_order))

    def test_all_multi_leg_use_mleg(self):
        """All multi-leg submit functions must call _submit_mleg_order."""
        import options_execution as oe
        for fn_name in ['_submit_spread_order', '_submit_buy_straddle_order',
                        '_submit_straddle_order', '_submit_condor_order']:
            src = inspect.getsource(getattr(oe, fn_name))
            self.assertIn('_submit_mleg_order', src, f'{fn_name} must use _submit_mleg_order')

    def test_no_old_polling_code(self):
        """Old polling/unwinding code must be fully removed."""
        import options_execution as oe
        full_src = inspect.getsource(oe)
        self.assertNotIn('_verify_multi_leg_fills', full_src)
        self.assertNotIn('_submit_multi_leg', full_src)

    @patch("options_execution.requests.post")
    def test_mleg_sends_correct_payload(self, mock_post):
        """mleg order must use order_class=mleg with legs array."""
        from options_execution import _submit_mleg_order
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"id": "test_123", "status": "pending_new"}
        )
        result = _submit_mleg_order(
            legs=[{"symbol": "SPY260424C00500000", "side": "buy", "ratio_qty": 1},
                  {"symbol": "SPY260424P00500000", "side": "buy", "ratio_qty": 1}],
            qty=1, limit_price=5.0, label="Test straddle"
        )
        # Verify the POST payload
        call_args = mock_post.call_args
        payload = call_args.kwargs.get('json') or call_args[1].get('json')
        self.assertEqual(payload['order_class'], 'mleg')
        self.assertEqual(payload['qty'], '1')
        self.assertEqual(len(payload['legs']), 2)
        self.assertEqual(payload['legs'][0]['position_intent'], 'buy_to_open')
        self.assertIn('pending_new', result.get('status', '') or result.get('detail', ''))

    @patch("options_execution.requests.post")
    def test_spread_rejection_returns_error(self, mock_post):
        """If Alpaca rejects the mleg order, return error status."""
        from options_execution import _submit_spread_order
        mock_post.return_value = MagicMock(
            status_code=403,
            json=lambda: {"message": "insufficient buying power"},
            text="insufficient buying power",
            headers={"content-type": "application/json"}
        )
        result = _submit_spread_order({
            "long_leg": "AAPL260418C00185000",
            "short_leg": "AAPL260418C00195000",
            "qty": 1, "net_debit": 2.50,
        })
        self.assertEqual(result["status"], "error")


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
        """bot_engine must have MAX_OPTIONS_POSITIONS separate from MAX_POSITIONS.

        2026-07-04: stopped pinning the VALUES (5/3) — they are tunable
        parameters (SIZING-FIX 2026-04-22 and ALPHA-TUNE 2026-04-21 moved
        them to 8/8 with dated comments) and pinning tunables in tests
        contradicts the constitution's RULE REVIEW authority. The MECHANISM
        this class exists for — separate caps for stock vs options slots —
        is what gets asserted.
        """
        import bot_engine
        import inspect
        self.assertIsInstance(bot_engine.MAX_POSITIONS, int)
        self.assertIsInstance(bot_engine.MAX_OPTIONS_POSITIONS, int)
        self.assertGreater(bot_engine.MAX_POSITIONS, 0)
        self.assertGreater(bot_engine.MAX_OPTIONS_POSITIONS, 0)
        # Separation must be structural: two independent assignments, so a
        # future refactor can't silently alias one cap to the other.
        src = inspect.getsource(bot_engine)
        self.assertIn("\nMAX_POSITIONS =", src)
        self.assertIn("\nMAX_OPTIONS_POSITIONS =", src)

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
        # anchored to the live constant, not a stale value pin (2026-07-04)
        self.assertEqual(options_slots, MAX_OPTIONS_POSITIONS - 1)

    def test_full_stock_slots_still_allows_options(self):
        """Even with ALL stock slots taken, options scanner should get slots."""
        from bot_engine import MAX_POSITIONS, MAX_OPTIONS_POSITIONS
        # fill the stock book to its cap, whatever the cap is tuned to
        stock_positions = [
            {"symbol": f"STK{i}", "asset_class": "us_equity"}
            for i in range(MAX_POSITIONS)
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
        # THE mechanism: a full stock book consumes zero options slots
        self.assertEqual(options_slots, MAX_OPTIONS_POSITIONS)

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
#  TEST: v1.0.33 Threshold Fixes + CSP Setup
# ═══════════════════════════════════════════════════════════════════════════════

class TestV1033ThresholdFixes(unittest.TestCase):
    """Verify v1.0.33 threshold changes are in the source code."""

    def test_earnings_spread_at_010(self):
        """Earnings IV crush spread limit should be 0.10 (v1.0.34: tightened from 0.15)."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod._setup_earnings_iv_crush)
        self.assertIn('> 0.10', src)

    def test_high_iv_spread_at_010(self):
        """High-IV premium sale spread limit should be 0.10 (v1.0.34: tightened from 0.15)."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod._setup_high_iv_premium_sale)
        self.assertIn('> 0.10', src)

    def test_high_iv_ivr_threshold_at_50(self):
        """High-IV premium sale must use IVR 50, not 70."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod._setup_high_iv_premium_sale)
        self.assertIn('iv_rank < 50', src)
        self.assertNotIn('iv_rank < 70', src)

    def test_low_iv_straddle_cost_at_5pct(self):
        """Low-IV breakout buy straddle cost limit should be 5%, not 3%."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod._setup_low_iv_breakout_buy)
        self.assertIn('straddle_pct >= 5.0', src)
        self.assertNotIn('straddle_pct >= 3.0', src)

    def test_low_iv_spread_widened_to_015(self):
        """Low-IV breakout buy spread limit should be 0.15, not 0.12."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod._setup_low_iv_breakout_buy)
        self.assertIn('> 0.15', src)
        self.assertNotIn('> 0.12', src)


class TestCSPNormalMarketSetup(unittest.TestCase):
    """Verify the new CSP normal-market setup exists and is correctly integrated."""

    def test_csp_function_exists(self):
        """_setup_csp_normal_market function must exist."""
        import options_scanner as os_mod
        self.assertTrue(hasattr(os_mod, '_setup_csp_normal_market'))
        self.assertTrue(callable(os_mod._setup_csp_normal_market))

    def test_csp_integrated_in_scan_options(self):
        """scan_options docstring should mention 6 setups, not 5."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod.scan_options)
        self.assertIn('6 options setup', src)

    def test_csp_integrated_in_check_ticker(self):
        """_check_ticker inner function should call _setup_csp_normal_market."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod.scan_options)
        self.assertIn('_setup_csp_normal_market', src)

    def test_csp_excluded_from_get_options_trades(self):
        """v1.0.34: get_options_trades must NOT pass through csp_normal_market (disabled filler)."""
        from options_scanner import HIGH_EDGE_SETUPS
        self.assertNotIn('csp_normal_market', HIGH_EDGE_SETUPS)

    def test_csp_returns_correct_strategy(self):
        """CSP setup must output sell_cash_secured_put as the strategy."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod._setup_csp_normal_market)
        self.assertIn('"sell_cash_secured_put"', src)
        self.assertIn('"csp_normal_market"', src)
        self.assertIn('"sell"', src)  # side must be sell

    def test_csp_ivr_band_15_to_50(self):
        """CSP should only fire for IVR 15-50 (moderate IV).

        Lower bound dropped from 20 → 15 on 2026-04-17 per backtest_scenario_c_wf
        (Alpaca commission-free options justify firing on calmer underlyings).
        """
        import options_scanner as os_mod
        src = inspect.getsource(os_mod._setup_csp_normal_market)
        self.assertIn('iv_rank < 15', src)
        self.assertIn('iv_rank > 50', src)

    def test_csp_uses_30_delta_put(self):
        """CSP should target 30-delta put."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod._setup_csp_normal_market)
        self.assertIn('target_delta=0.30', src)

    def test_csp_score_capped_at_75(self):
        """CSP max score should be capped at 75 (below high-conviction setups)."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod._setup_csp_normal_market)
        self.assertIn('min(75', src)

    def test_csp_rejects_high_vxx(self):
        """CSP should not fire when VXX ratio >= 1.15."""
        import options_scanner as os_mod
        src = inspect.getsource(os_mod._setup_csp_normal_market)
        self.assertIn('vxx_ratio >= 1.15', src)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 20: KNOWN BROKEN #12(c) — standalone options entries/exits now flow
#  into trade_feedback (research/open_questions.md)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptionsFeedbackWiring(unittest.TestCase):
    """options_manager.py's exits used to record NOTHING into trade_feedback
    (KNOWN BROKEN #12(c)). These tests pin the standalone (single-leg) wiring
    added to close that gap, and — just as importantly — that multi-leg
    strategies are deliberately excluded (they're closed as one combined
    mleg order and ticker-based _find_entry_record() matching can't safely
    attribute per-leg economics; see MULTI_LEG_STRATEGIES' docstring)."""

    @patch("ml_model_v2.track_fill")
    def test_multi_leg_entry_does_not_call_track_fill(self, mock_track_fill):
        from options_manager import register_options_entry, OPTIONS_STATE_PATH
        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

        register_options_entry(
            "AAPL260418C00190000", 1.00, "sell", "iron_condor",
            delta=-0.20, qty=1, ticker="AAPL",
        )

        mock_track_fill.assert_not_called()

        if os.path.exists(OPTIONS_STATE_PATH):
            os.remove(OPTIONS_STATE_PATH)

    @patch("ml_model_v2.track_fill")
    def test_record_options_exit_feedback_pnl_pct_matches_dollar_pnl(self, mock_track_fill):
        """Direct unit test of the pnl_pct math: real Alpaca unrealized_pl
        (dollars) against notional cost basis (entry_price * qty * 100),
        not a fabricated/synthetic number."""
        from options_manager import _record_options_exit_feedback

        # Sold 2 contracts at $2.00 credit (cost basis $400), now $120 profit
        # -> 120 / 400 * 100 = 30%
        _record_options_exit_feedback(
            ticker="XYZ", side="short", qty=2, entry_price=2.00,
            exit_price=1.40, unrealized_pnl=120.0, exit_reason="profit_target",
            entry_timestamp=(datetime.now() - timedelta(days=3)).isoformat(),
        )

        mock_track_fill.assert_called_once()
        payload = mock_track_fill.call_args[0][0]
        self.assertEqual(payload["ticker"], "XYZ")
        self.assertEqual(payload["side"], "buy")  # closing side reverses a short entry
        self.assertEqual(payload["exit_reason"], "profit_target")
        self.assertAlmostEqual(payload["exit_context"]["pnl_pct"], 30.0, places=3)
        self.assertEqual(payload["exit_context"]["days_held"], 3)
        self.assertIn("code_version", payload)

    @patch("ml_model_v2.track_fill")
    def test_record_options_exit_feedback_never_raises_on_bad_input(self, mock_track_fill):
        """Must never raise — this runs inside position management, which
        must never crash on a feedback-recording failure."""
        from options_manager import _record_options_exit_feedback
        mock_track_fill.side_effect = Exception("boom")
        try:
            _record_options_exit_feedback("XYZ", "short", 1, 2.00, 1.00, 100.0,
                                           "profit_target", None)
        except Exception as e:
            self.fail(f"_record_options_exit_feedback raised: {e}")

    @patch("ml_model_v2.track_fill")
    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._attempt_roll")
    @patch("options_manager._submit_close_order")
    def test_assignment_close_wired_to_feedback(self, mock_close, mock_roll, mock_snap,
                                                 mock_get, mock_track_fill):
        """A deep-ITM sold option near expiry whose roll attempt fails must
        force-close AND record the exit into trade_feedback (previously:
        neither the close nor any prior single-leg CLOSE path recorded
        anything at all)."""
        from options_manager import manage_options_positions, _save_options_state

        exp = (datetime.now() + timedelta(days=8)).strftime("%y%m%d")
        occ = f"AAPL{exp}P00200000"

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{
                "symbol": occ, "qty": "-1", "side": "short",
                "avg_entry_price": "3.00", "current_price": "6.00",
                "market_value": "-600", "unrealized_pl": "-300",
                "asset_class": "option",
            }]
        )
        # |delta| > 0.80 assignment threshold
        mock_snap.return_value = {"bid": 5.80, "ask": 6.20, "mid": 6.00,
                                   "delta": -0.90, "gamma": 0.05, "theta": -0.10, "vega": 0.04, "iv": 0.40}
        mock_roll.return_value = {"rolled": False, "detail": "no valid roll candidate"}
        mock_close.return_value = {"status": "submitted", "order_id": "assign999"}

        _save_options_state({
            occ: {
                "entry_price": 3.00, "entry_delta": -0.30, "side": "short",
                "entry_date": "2026-04-12",
                "entry_timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
                "strategy": "sell_cash_secured_put", "highest_value": 3.00, "qty": 1,
                "initial_credit": 3.00,
            }
        })

        result = manage_options_positions(100000)
        close_actions = [a for a in result["actions"] if a.get("type") == "assignment_close"]
        self.assertGreaterEqual(len(close_actions), 1, "Should force-close on failed assignment roll")

        mock_track_fill.assert_called_once()
        payload = mock_track_fill.call_args[0][0]
        self.assertEqual(payload["ticker"], "AAPL")
        self.assertEqual(payload["exit_reason"], "assignment_close")
        # Lost $300 on a $300 credit basis (1 contract * $3.00 * 100) = -100%
        self.assertAlmostEqual(payload["exit_context"]["pnl_pct"], -100.0, places=1)


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
