#!/usr/bin/env python3
"""
VolTradeAI — third-leg exit vs floor-allocator ownership (GLD churn, 2026-10-12)

GLD had two owners: _manage_defensive_floor() holds DEFENSIVE_FLOOR_TICKER
(GLD) always-on at DEFENSIVE_FLOOR_BULL/NEUTRAL (8%), while _run_third_leg()'s
"exit VRP + legacy sector positions when regime recovers" block market-sold
ALL GLD every NEUTRAL/BULL cycle. Live 2026-10-12: 21 GLD bought/sold ~11
times. Same class as the 2026-04-10 QQQ churn — two allocators fighting
over one ticker.

These tests pin:
  1. Regression: GLD held in NEUTRAL -> _run_third_leg posts NO sell for GLD
     (and still exits SVXY, the ticker the exit genuinely owns).
  2. Ownership invariant: under the LIVE BASE_CONFIG, with every
     floor-owned ticker held, _run_third_leg sells none of them. The
     floor-owned set is derived here independently from BASE_CONFIG, so a
     future floor/leg collision fails CI rather than churning live.
No live Alpaca/network connectivity — every requests call is mocked.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import bot_engine  # noqa: E402
import system_config  # noqa: E402

# vxx_ratio 1.0 keeps the VRP-harvest BUY branch (1.05-1.25) from firing.
_MACRO = {
    "vxx_ratio": 1.0, "spy_vs_ma50": 1.0,
    "spy_below_200_days": 0, "spy_above_200d": True,
}


def _live_floor_owned():
    """Independent derivation of floor-allocator-owned tickers."""
    cfg = system_config.BASE_CONFIG
    owned = {cfg.get("FLOOR_TICKER", "QQQ")}
    if cfg.get("FLOOR_BASKET_ENABLED", False):
        owned |= {t for t in cfg.get("FLOOR_BASKET", {}) if t != "CASH"}
    if cfg.get("DEFENSIVE_FLOOR_ENABLED", False):
        owned.add(cfg.get("DEFENSIVE_FLOOR_TICKER", "GLD"))
    return owned


def _resp(payload):
    m = MagicMock()
    m.json.return_value = payload
    return m


def _run(held_syms, regime="NEUTRAL"):
    positions = [
        {"symbol": s, "qty": "21", "market_value": "1000", "unrealized_plpc": "0.01"}
        for s in held_syms
    ]

    def fake_get(url, *a, **kw):
        if url.endswith("/v2/account"):
            return _resp({"equity": "102000"})
        if url.endswith("/v2/positions"):
            return _resp(positions)
        if url.endswith("/v2/orders"):
            return _resp([])
        return _resp({})

    post = MagicMock(return_value=_resp({"id": "test-order"}))
    with patch("requests.get", side_effect=fake_get), \
         patch("requests.post", post), \
         patch("system_config.get_market_regime", return_value=regime):
        result = bot_engine._run_third_leg(dict(_MACRO))
    sells = [
        c.kwargs.get("json", {}).get("symbol")
        for c in post.call_args_list
        if (c.kwargs.get("json") or {}).get("side") == "sell"
    ]
    return result, sells


class TestThirdLegDoesNotSellFloorOwnedGLD(unittest.TestCase):
    def test_gld_held_in_neutral_not_sold_svxy_still_exits(self):
        self.assertTrue(system_config.BASE_CONFIG.get("DEFENSIVE_FLOOR_ENABLED"),
                        "test premise: defensive floor enabled in live config")
        self.assertEqual(system_config.BASE_CONFIG.get("DEFENSIVE_FLOOR_TICKER"), "GLD")
        result, sells = _run(["GLD", "SVXY"], regime="NEUTRAL")
        self.assertEqual(result.get("status"), "ok", result)
        self.assertNotIn("GLD", sells,
                         "third-leg exit sold GLD, which the defensive floor owns "
                         "(always-on in NEUTRAL/BULL) — allocator churn")
        self.assertIn("SVXY", sells, "third-leg exit must still exit SVXY on recovery")

    def test_gld_held_in_bull_not_sold(self):
        _, sells = _run(["GLD"], regime="BULL")
        self.assertNotIn("GLD", sells)


class TestFloorOwnershipInvariant(unittest.TestCase):
    """Third-leg exit set and floor-owned tickers must be disjoint (live config)."""

    def test_exit_set_disjoint_from_floor_owned(self):
        owned = _live_floor_owned()
        self.assertIn("GLD", owned)
        exit_set = bot_engine._third_leg_exit_tickers(system_config.BASE_CONFIG)
        self.assertFalse(exit_set & owned,
                         f"third-leg exit set collides with floor-owned tickers: "
                         f"{sorted(exit_set & owned)}")

    def test_behavioral_no_floor_owned_ticker_sold(self):
        owned = sorted(_live_floor_owned())
        for regime in ("NEUTRAL", "BULL"):
            _, sells = _run(owned + ["SVXY"], regime=regime)
            self.assertFalse(set(sells) & set(owned),
                             f"{regime}: third leg sold floor-owned {sorted(set(sells) & set(owned))}")
            self.assertIn("SVXY", sells)

    def test_disabled_defensive_floor_releases_gld_to_exit(self):
        cfg = dict(system_config.BASE_CONFIG, DEFENSIVE_FLOOR_ENABLED=False)
        self.assertIn("GLD", bot_engine._third_leg_exit_tickers(cfg))


if __name__ == "__main__":
    unittest.main()
