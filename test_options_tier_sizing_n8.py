"""Regression tests — audit §6.8 N8 (tier sizing never reaches execution),
[REPAIR] 2026-10-27.

The tier dispatcher (server/bot.ts SELL_CSP) called
select_contract(ticker, "sell_cash_secured_put", 0, equity, cash_available)
with no size_pct / metadata, so T1's stress + drawdown de-risking, T2's
leverage and T1's target delta never reached contract selection —
_dynamic_options_size ran on a fabricated score of 70 instead. And
_select_sell_put could "stretch" one CSP's collateral to 20% of equity, past
the 8% MAX_OPTIONS_PCT per-position cap. These pin:
  * a caller-supplied size_pct is the sizing fraction (dynamic sizing is
    only the fallback), clamped to the per-position cap;
  * the stretch branch never exceeds the cap;
  * metadata target_delta reaches the CSP strike pick.
No network: the chain fetch and liquidity filter are patched.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options_execution  # noqa: E402


def _put(occ, strike, delta, dte=35, expiry="2026-12-18"):
    return {
        "occ_symbol": occ, "option_type": "put", "strike": strike,
        "bid": 1.00, "ask": 1.10, "mid": 1.05, "delta": delta,
        "gamma": 0.02, "theta": -0.03, "iv": 0.30,
        "volume": 500, "open_interest": 2000,
        "expiry": expiry, "days_to_expiry": dte,
    }


CHAIN = [
    _put("ABC261218P00045000", 45.0, -0.30),
    _put("ABC261218P00040000", 40.0, -0.20),
]


class TestCallerSizeReachesSelection(unittest.TestCase):
    def _run(self, **kwargs):
        seen = {}

        def _fake_sell_put(contracts, price, equity, ticker, size_pct=0.05,
                           cash_available=None, target_delta=None):
            seen["size_pct"] = size_pct
            seen["target_delta"] = target_delta
            return {"error": None, "qty": 1}

        with patch("options_execution._fetch_option_chain", return_value=list(CHAIN)), \
             patch("options_execution._is_liquid", return_value=True), \
             patch("vol_surface.get_surface_score", return_value={}), \
             patch("options_execution._dynamic_options_size", return_value=0.031) as dyn, \
             patch("options_execution._select_sell_put", side_effect=_fake_sell_put):
            options_execution.select_contract(
                "ABC", "sell_cash_secured_put", 50.0, 100_000, **kwargs)
        return seen, dyn

    def test_tier_size_pct_is_used_not_default_score_dynamic_size(self):
        seen, dyn = self._run(size_pct=0.048)
        self.assertAlmostEqual(seen["size_pct"], 0.048)
        dyn.assert_not_called()

    def test_tier_size_above_per_position_cap_is_clamped(self):
        # T2 can propose up to 2x MAX_OPTIONS_PCT; collateral never exceeds the cap.
        seen, _ = self._run(size_pct=0.16)
        self.assertAlmostEqual(seen["size_pct"], options_execution.MAX_OPTIONS_PCT_CEILING)

    def test_without_size_pct_dynamic_sizing_is_the_fallback(self):
        seen, dyn = self._run()
        self.assertAlmostEqual(seen["size_pct"], 0.031)
        dyn.assert_called_once()

    def test_metadata_target_delta_reaches_the_put_pick(self):
        seen, _ = self._run(size_pct=0.05, metadata={"target_delta": 0.20})
        self.assertAlmostEqual(seen["target_delta"], 0.20)


class TestSellPutCollateralCeiling(unittest.TestCase):
    def test_stretch_never_exceeds_max_options_pct(self):
        # $100k equity, 2% budget = $2,000. Smallest strike $150 needs $15,000
        # = 15% of equity: inside the old 20% stretch, outside the 8% cap.
        chain = [_put("XYZ261218P00150000", 150.0, -0.30)]
        result = options_execution._select_sell_put(chain, 160.0, 100_000, "XYZ", 0.02)
        self.assertIsNotNone(result.get("error"),
                             "a CSP needing 15% of equity must not be stretched into")
        self.assertIn("budget", result["error"])  # bot.ts logs it as rejected_capital

    def test_stretch_still_allowed_up_to_the_cap(self):
        # Smallest strike $70 needs $7,000 = 7% <= 8% cap: stretch still fires.
        chain = [_put("XYZ261218P00070000", 70.0, -0.30)]
        result = options_execution._select_sell_put(chain, 75.0, 100_000, "XYZ", 0.02)
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertEqual(result["qty"], 1)
        self.assertLessEqual(result["cash_required"],
                             100_000 * options_execution.MAX_OPTIONS_PCT_CEILING)

    def test_target_delta_selects_matching_put(self):
        result = options_execution._select_sell_put(
            list(CHAIN), 50.0, 100_000, "ABC", 0.08, target_delta=0.20)
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertEqual(result["strike"], 40.0)

    def test_default_target_delta_unchanged(self):
        result = options_execution._select_sell_put(list(CHAIN), 50.0, 100_000, "ABC", 0.08)
        self.assertEqual(result["strike"], 45.0)

    def test_cap_matches_system_config(self):
        from system_config import BASE_CONFIG
        self.assertEqual(options_execution.MAX_OPTIONS_PCT_CEILING, BASE_CONFIG["MAX_OPTIONS_PCT"])


class TestDispatcherPassesTierSizing(unittest.TestCase):
    def test_sell_csp_subprocess_passes_size_pct_and_metadata(self):
        here = os.path.dirname(os.path.abspath(__file__))
        bot = open(os.path.join(here, "server", "bot.ts")).read()
        start = bot.find("const cspPayload = {")
        self.assertGreater(start, 0)
        end = bot.find("submit_options_order(contract)", start)
        section = bot[start:end]
        self.assertIn("size_pct: action.size_pct,", section)
        self.assertIn("metadata: action.metadata,", section)
        self.assertIn("size_pct=data.get('size_pct')", section)
        self.assertIn("metadata=data.get('metadata')", section)

    def test_buy_put_hedge_subprocess_passes_size_pct_and_metadata(self):
        here = os.path.dirname(os.path.abspath(__file__))
        bot = open(os.path.join(here, "server", "bot.ts")).read()
        start = bot.find("const hedgePayload = {")
        self.assertGreater(start, 0)
        end = bot.find("submit_options_order(contract)", start)
        section = bot[start:end]
        self.assertIn("size_pct=data.get('size_pct')", section)
        self.assertIn("metadata=data.get('metadata')", section)


if __name__ == "__main__":
    unittest.main()
