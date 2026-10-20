"""Entry DTE vs the options manager's 21-DTE exit zone — the root-cause
invariant (2026-10-20, RECURRENCE of the 2026-10-13 CSP defect).

LIVE FAILURE: iron condors on TSLA / META / GOOGL (2026-10-19/20) were
fetched from min_dte=7. On a big chain the 100-contract snapshot page fills
with the nearest weeklies, so _select_condor's "closest to 30 DTE"
preference only saw 7-14 DTE expiries; _manage_strategy_group's Exit Rule 4
(dte <= DTE_EXIT_THRESHOLD) then closed every condor on its first eligible
cycle. The 2026-10-13 fix (test_csp_entry_dte.py) had floored CSPs only.

INVARIANT: every strategy select_contract can open is either fetched from a
floor that clears the exit zone (as the manager counts DTE), or is named in
ENTRY_EXIT_ZONE_CONFLICTS with its reason. A new strategy fails here until
it is classified — the conflict can no longer reappear silently.
"""
import inspect
import re
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import options_execution
import options_manager


def _dispatched_strategies() -> set:
    src = inspect.getsource(options_execution.select_contract)
    return set(re.findall(r'(?:if|elif) strategy == "([a-z_]+)"', src))


class TestClassification(unittest.TestCase):
    def test_every_dispatched_strategy_is_classified_once(self):
        dispatched = _dispatched_strategies()
        self.assertGreaterEqual(len(dispatched), 10, "dispatch parse broke")
        floored = set(options_execution._EXIT_ZONE_FLOORED)
        conflicts = set(options_execution.ENTRY_EXIT_ZONE_CONFLICTS)
        self.assertEqual(floored & conflicts, set())
        self.assertEqual(dispatched - floored - conflicts, set(),
                         "strategy opens without an exit-zone decision")

    def test_every_grouped_multi_leg_strategy_is_floored_except_the_weekly(self):
        for s in options_manager.MULTI_LEG_STRATEGIES:
            if s == "qqq_iron_condor":
                self.assertIn(s, options_execution.ENTRY_EXIT_ZONE_CONFLICTS)
            else:
                self.assertIn(s, options_execution._EXIT_ZONE_FLOORED, s)

    def test_floor_clears_the_exit_zone_as_the_manager_counts_dte(self):
        # The fetch floors on the calendar date (today + min_dte); the
        # manager counts floor((expiry 00:00 ET - now) / 1 day), which is one
        # less during the session. The floor must clear the zone after that.
        self.assertGreater(options_execution.MULTI_LEG_ENTRY_MIN_DTE - 1,
                           options_manager.DTE_EXIT_THRESHOLD)
        self.assertEqual(options_execution.MULTI_LEG_ENTRY_MIN_DTE,
                         options_execution.CSP_ENTRY_MIN_DTE, "one shared floor")


def _chain(price: float, dtes) -> list:
    out = []
    for dte in dtes:
        exp = (datetime.now() + timedelta(days=dte)).strftime("%Y-%m-%d")
        for k in range(-12, 13):
            strike = round(price + k * 5.0, 2)
            for cp in ("call", "put"):
                otm = (strike - price) if cp == "call" else (price - strike)
                mid = max(0.4, 12.0 - otm * 0.25) * (1 + dte / 60)
                out.append({
                    "occ_symbol": f"TSLA{exp[2:4]}{exp[5:7]}{exp[8:10]}{cp[0].upper()}{int(strike * 1000):08d}",
                    "ticker": "TSLA", "option_type": cp, "strike": strike,
                    "bid": round(mid - 0.05, 2), "ask": round(mid + 0.05, 2), "mid": round(mid, 2),
                    "bid_size": 50, "ask_size": 50, "delta": 0.3 if cp == "call" else -0.3,
                    "gamma": 0.01, "theta": -0.05, "iv": 0.6, "volume": 500, "open_interest": 2000,
                    "expiry": exp, "days_to_expiry": dte,
                })
    return out


def _page_fill_fetch(seen):
    """Alpaca's snapshot page as it behaved live: limit=100 fills from the
    NEAREST expiry at or after min_dte, so only that one expiry comes back."""
    def fetch(ticker, price, min_dte=7, option_type=None):
        seen.append(min_dte)
        first = min(d for d in (9, 16, 30, 37, 44) if d >= min_dte)
        return _chain(price, [first])
    return fetch


class TestCondorEntryClearsExitZone(unittest.TestCase):
    @patch("vol_surface.get_surface_score", return_value={})
    @patch("options_execution._dynamic_options_size", return_value=0.04)
    @patch("options_execution._is_liquid", return_value=True)
    def test_live_shape_condor_is_born_outside_the_exit_zone(self, *_):
        seen = []
        with patch("options_execution._fetch_option_chain", side_effect=_page_fill_fetch(seen)):
            c = options_execution.select_contract("TSLA", "iron_condor", 445.0, 100000)
        self.assertIsNone(c.get("error"), c.get("error"))
        self.assertEqual(seen, [options_execution.MULTI_LEG_ENTRY_MIN_DTE])
        # judged by the manager's own DTE function, not the selector's field
        self.assertGreater(options_manager._days_to_expiry(c["expiry"]),
                           options_manager.DTE_EXIT_THRESHOLD,
                           "condor would be closed by Exit Rule 4 on its first cycle")

    @patch("options_execution._fetch_option_chain", return_value=[])
    def test_every_floored_strategy_requests_the_floor(self, mock_fetch):
        for s in options_execution._EXIT_ZONE_FLOORED:
            options_execution.select_contract("TSLA", s, 445.0, 100000)
            self.assertGreaterEqual(mock_fetch.call_args.kwargs.get("min_dte"),
                                    options_execution.MULTI_LEG_ENTRY_MIN_DTE, s)

    @patch("options_execution._fetch_option_chain", return_value=[])
    def test_weekly_qqq_condor_keeps_its_design(self, mock_fetch):
        options_execution.select_contract("QQQ", "qqq_iron_condor", 500.0, 100000)
        self.assertEqual(mock_fetch.call_args.kwargs.get("min_dte"), 3)


if __name__ == "__main__":
    unittest.main()
