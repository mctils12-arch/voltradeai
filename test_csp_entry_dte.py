"""Regression tests for fix f3-csp-entry-dte (2026-10-13).

DEFECT: cash-secured puts were born inside options_manager's 21-DTE exit
zone. select_contract fetched CSP expiries from min_dte=7 and
_select_sell_put picked on delta alone, so a 7-21 DTE put was routinely
chosen; options_manager Check 3 (dte <= DTE_EXIT_THRESHOLD) then closed it
as soon as MIN_HOLD_MINUTES elapsed — a ~60-71 minute spread round-trip,
re-sold by T1 the next cycle (live: XLF/NOK/NU 2026-10-23/10-30 expiries).

FIX: the entry side yields (the 21-DTE exit is a risk rule and stays):
  * select_contract requests CSP expiries from CSP_ENTRY_MIN_DTE
    (= tiered_strategy.T1_TARGET_DTE_MIN, clamped > DTE_EXIT_THRESHOLD);
  * _select_sell_put drops any candidate with days_to_expiry <=
    DTE_EXIT_THRESHOLD (defence in depth).
"""
import unittest
from unittest.mock import patch

import options_execution
import options_manager
import tiered_strategy


def _put(occ, strike, delta, dte, expiry):
    return {
        "occ_symbol": occ, "ticker": "XLF", "option_type": "put",
        "strike": strike, "bid": 0.90, "ask": 0.98, "mid": 0.94,
        "bid_size": 50, "ask_size": 50,
        "delta": delta, "gamma": 0.03, "theta": -0.02, "iv": 0.22,
        "volume": 500, "open_interest": 2000,
        "expiry": expiry, "days_to_expiry": dte,
    }


# The live failure shape: an 11-DTE put sitting exactly on the -0.30 delta
# target beside a 35-DTE put slightly further from it. Delta-only selection
# picks the 11-DTE contract.
SHORT_DATED = _put("XLF261023P00050000", 50.0, -0.30, 11, "2026-10-23")
LONG_DATED = _put("XLF261117P00049000", 49.0, -0.27, 35, "2026-11-17")


class TestCspEntryDteInvariant(unittest.TestCase):
    def test_entry_min_dte_strictly_beyond_exit_threshold(self):
        self.assertGreater(
            options_execution.CSP_ENTRY_MIN_DTE,
            options_manager.DTE_EXIT_THRESHOLD,
            "CSP entries must be requested strictly outside the "
            "options_manager DTE exit zone, or they round-trip on fill",
        )

    def test_entry_min_dte_is_the_t1_target_window_min(self):
        # One shared constant — the T1 target window — not a new number.
        self.assertEqual(
            options_execution.CSP_ENTRY_MIN_DTE,
            max(tiered_strategy.T1_TARGET_DTE_MIN,
                options_manager.DTE_EXIT_THRESHOLD + 1),
        )

    def test_guard_tracks_the_live_exit_threshold(self):
        self.assertEqual(options_execution._DTE_EXIT_THRESHOLD,
                         options_manager.DTE_EXIT_THRESHOLD)


class TestSelectSellPutExitZoneGuard(unittest.TestCase):
    def test_never_selects_put_inside_exit_zone(self):
        result = options_execution._select_sell_put(
            [SHORT_DATED, LONG_DATED], 52.0, 100000, "XLF", 0.05)
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertGreater(result["days_to_expiry"],
                           options_manager.DTE_EXIT_THRESHOLD)
        self.assertEqual(result["occ_symbol"], LONG_DATED["occ_symbol"])

    def test_only_exit_zone_puts_fails_cleanly(self):
        result = options_execution._select_sell_put(
            [SHORT_DATED], 52.0, 100000, "XLF", 0.05)
        self.assertIsNotNone(result.get("error"))
        self.assertIn("exit threshold", result["error"])
        # Not a capital rejection — bot.ts must not log it as
        # rejected_capital counterfactual (it keys on these words).
        self.assertNotIn("budget", result["error"])
        self.assertNotIn("capital", result["error"])

    def test_boundary_dte_equal_to_threshold_is_rejected(self):
        at_threshold = _put("XLF261102P00050000", 50.0, -0.30,
                            options_manager.DTE_EXIT_THRESHOLD, "2026-11-02")
        result = options_execution._select_sell_put(
            [at_threshold], 52.0, 100000, "XLF", 0.05)
        self.assertIsNotNone(result.get("error"))

    def test_stretch_mode_cannot_pick_exit_zone_strike(self):
        # Smallest strike is short-dated; stretch mode used to fall back to
        # it regardless of expiry. Budget is tiny so stretch triggers.
        cheap_short = _put("XLF261023P00040000", 40.0, -0.10, 11, "2026-10-23")
        pricier_long = _put("XLF261117P00045000", 45.0, -0.15, 35, "2026-11-17")
        result = options_execution._select_sell_put(
            [cheap_short, pricier_long], 52.0, 100000, "XLF", 0.001)
        # Budget = max(100k*0.001, 2% floor) = $2,000 < both strikes, so the
        # stretch branch picks the smallest ELIGIBLE strike ($45, 35 DTE).
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertGreater(result["days_to_expiry"],
                           options_manager.DTE_EXIT_THRESHOLD)
        self.assertEqual(result["strike"], 45.0)


class TestSelectContractRequestsBeyondExitZone(unittest.TestCase):
    @patch("vol_surface.get_surface_score", return_value={})
    @patch("options_execution._dynamic_options_size", return_value=0.05)
    @patch("options_execution._fetch_option_chain")
    def test_csp_fetch_min_dte_and_selection(self, mock_fetch, _size, _surf):
        seen = {}

        def _fake_fetch(ticker, price, min_dte=7, option_type=None):
            seen["min_dte"] = min_dte
            return [SHORT_DATED, LONG_DATED]
        mock_fetch.side_effect = _fake_fetch

        with patch("options_execution._is_liquid", return_value=True):
            result = options_execution.select_contract(
                "XLF", "sell_cash_secured_put", 52.0, 100000)
        self.assertGreater(seen["min_dte"], options_manager.DTE_EXIT_THRESHOLD)
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertGreater(result["days_to_expiry"],
                           options_manager.DTE_EXIT_THRESHOLD)

    @patch("options_execution._fetch_option_chain", return_value=[])
    def test_other_strategies_keep_their_fetch_floor(self, mock_fetch):
        options_execution.select_contract("QQQ", "buy_put", 500.0, 100000)
        self.assertEqual(mock_fetch.call_args.kwargs.get("min_dte"), 7)
        options_execution.select_contract("QQQ", "covered_call", 500.0, 100000,
                                          trade={"shares_held": 100})
        self.assertEqual(mock_fetch.call_args.kwargs.get("min_dte"), 3)


if __name__ == "__main__":
    unittest.main()
