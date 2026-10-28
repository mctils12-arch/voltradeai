#!/usr/bin/env python3
"""
VolTradeAI — third-leg (SVXY VRP harvest) risk regression tests (audit §6.8 N14/N20).

SVXY is a short-volatility ETF. _run_third_leg (bot_engine.py) bought it when
the VXX ratio sat in 1.05-1.25 and only ever exited on a NEUTRAL/BULL regime.
SVXY is in bot.ts FLOOR_AND_LEG_TICKERS, so it gets no websocket stop and no
POS-KILL, and Python manage_positions skips it too. These tests pin:
  1. size: LEG3_VRP_PCT is capped at the documented 15% sweet spot (was 40%).
No live Alpaca/network connectivity — every requests call is mocked.
"""
import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import system_config  # noqa: E402


class TestLeg3SizeCap(unittest.TestCase):
    def test_vrp_pct_capped_at_sweet_spot(self):
        vrp = float(system_config.BASE_CONFIG["LEG3_VRP_PCT"])
        self.assertGreater(vrp, 0.0, "VRP leg must stay enabled")
        self.assertLessEqual(
            vrp, 0.15,
            f"LEG3_VRP_PCT={vrp}: an unstopped short-vol leg above the 15% "
            "backtest sweet spot (was 0.40) — audit N14")


if __name__ == "__main__":
    unittest.main()
