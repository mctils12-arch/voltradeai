#!/usr/bin/env python3
"""
VolTradeAI — Floor-basket exemption regression tests (R-FIX 2026-07-17)

Covers a live bug: FLOOR_AND_LEG_TICKERS (the set of tickers manage_positions()
must NOT apply active stop-loss/take-profit/scale-out/time-stop logic to) was
hardcoded to {"QQQ", "SVXY", "SPY"} and never updated when system_config.py's
FLOOR_BASKET (SMH/KWEB/VXUS) shipped 2026-04-22, nor for DEFENSIVE_FLOOR_TICKER
(GLD). manage_positions() was therefore flagging live floor-basket ETFs for
active exits (confirmed live 2026-07-17: 5x erroneous "WS TIME STOP" sells of
VXUS + 1x "WS STOP LOSS" sell of SMH in a single session, each immediately
re-bought by the floor rebalancer on the next cycle).

Run: cd voltradeai && python3 -m pytest -q test_floor_basket_stops.py
"""
import os
import sys
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import bot_engine
import system_config


class TestFloorAndLegTickersMembership(unittest.TestCase):
    """FLOOR_AND_LEG_TICKERS must be derived from — and stay in sync with —
    system_config.BASE_CONFIG, not a hand-maintained literal that can drift."""

    def test_includes_every_live_floor_basket_member(self):
        basket_tickers = {t for t in system_config.BASE_CONFIG.get("FLOOR_BASKET", {}) if t != "CASH"}
        self.assertTrue(basket_tickers, "FLOOR_BASKET must not be empty for this test to be meaningful")
        missing = basket_tickers - bot_engine.FLOOR_AND_LEG_TICKERS
        self.assertFalse(
            missing,
            f"FLOOR_AND_LEG_TICKERS is missing live floor-basket members {missing} — "
            "manage_positions() would apply active stop/TP/time-stop logic to them.",
        )

    def test_includes_defensive_floor_ticker(self):
        defensive_ticker = system_config.BASE_CONFIG.get("DEFENSIVE_FLOOR_TICKER", "GLD")
        self.assertIn(defensive_ticker, bot_engine.FLOOR_AND_LEG_TICKERS)

    def test_pins_current_expected_set(self):
        # Direct pin on top of the derived-membership tests above — fails loudly
        # (rather than silently) if FLOOR_BASKET/DEFENSIVE_FLOOR_TICKER ever change,
        # prompting a human/session to re-verify the Node-side mirror in server/bot.ts.
        self.assertEqual(
            bot_engine.FLOOR_AND_LEG_TICKERS,
            {"QQQ", "SVXY", "SPY", "SMH", "KWEB", "VXUS", "GLD"},
        )


class TestManagePositionsSkipsFloorBasket(unittest.TestCase):
    """End-to-end: manage_positions() must not flag a floor-basket ETF for a
    TIME STOP even when the same conditions (days_held>=7, flat P&L) would
    flag a non-floor ticker."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp(prefix="vt_floor_basket_")
        self._data_dir_patch = patch.object(bot_engine, "DATA_DIR", self._tmp_dir)
        self._data_dir_patch.start()

        stale_entry_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        stop_state = {
            "VXUS": {"entry_date": stale_entry_date},
            "AAPL": {"entry_date": stale_entry_date},
        }
        with open(os.path.join(self._tmp_dir, "voltrade_stop_state.json"), "w") as f:
            json.dump(stop_state, f)

        self._atr_patch = patch.object(bot_engine, "_get_atr", return_value=2.0)
        self._atr_patch.start()

        self._macro_patch = patch(
            "macro_data.get_macro_snapshot",
            return_value={
                "vxx_ratio": 1.0,
                "spy_vs_ma50": 1.05,
                "spy_below_200_days": 0,
                "spy_above_200d": True,
            },
        )
        self._macro_patch.start()

    def tearDown(self):
        self._macro_patch.stop()
        self._atr_patch.stop()
        self._data_dir_patch.stop()

    def _flat_position(self, symbol):
        # Flat P&L (~0%), held 10 days per stop_state above — satisfies
        # manage_positions()'s time_stop condition (days_held>=7, |pnl|<2%)
        # for any ticker NOT in FLOOR_AND_LEG_TICKERS.
        return {
            "symbol": symbol,
            "asset_class": "us_equity",
            "side": "long",
            "qty": "10",
            "avg_entry_price": "100.00",
            "current_price": "100.10",
            "unrealized_plpc": "0.001",
            "market_value": "1001.0",
        }

    def test_floor_basket_ticker_not_flagged(self):
        with patch.object(
            bot_engine, "get_alpaca_positions",
            return_value=[self._flat_position("VXUS")],
        ):
            result = bot_engine.manage_positions()
        vxus_actions = [a for a in result.get("actions", []) if a.get("ticker") == "VXUS"]
        self.assertEqual(
            vxus_actions, [],
            "manage_positions() flagged an action for VXUS (a live FLOOR_BASKET "
            "member) — it must be fully skipped, not just excluded from execution.",
        )

    def test_control_non_floor_ticker_is_flagged(self):
        # Proves the test harness itself would have caught the bug: the exact
        # same held-10-days/flat-P&L position on a NON-floor ticker must still
        # produce a time_stop CLOSE action.
        with patch.object(
            bot_engine, "get_alpaca_positions",
            return_value=[self._flat_position("AAPL")],
        ):
            result = bot_engine.manage_positions()
        aapl_actions = [a for a in result.get("actions", []) if a.get("ticker") == "AAPL"]
        self.assertTrue(aapl_actions, "control ticker AAPL should have been flagged for TIME STOP")
        self.assertEqual(aapl_actions[0]["type"], "time_stop")


if __name__ == "__main__":
    unittest.main()
