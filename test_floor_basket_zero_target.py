#!/usr/bin/env python3
"""
VolTradeAI — floor basket members orphaned at a 0% target (audit §6.8 N18).

_manage_spy_floor (bot_engine.py): at target 0% (BEAR/PANIC) the zero-target
branch sold only FLOOR_TICKER (QQQ), and the basket branch requires
target_pct > 0 — so SMH/KWEB/VXUS (~28-36% of equity) were held with no exit
through the whole bear phase, pushing invested % over the BEAR/PANIC caps and
firing the master kill. The basket rebalance sell also never unwound covered
calls, while the single-ticker path does.

Pinned here:
  1. a 0% target sells EVERY floor-owned ticker (QQQ + non-CASH members);
  2. each after its own covered-call unwind (buy-to-close BEFORE the sale);
  3. a failed unwind blocks only that ticker's sale and leaves the exit to
     retry next cycle (last_regime not persisted);
  4. members already orphaned when the fix ships (state already BEAR, no
     regime change) are exited once, then not re-sold every cycle;
  5. a basket rebalance sell that leaves < 100 shares unwinds calls first.
All Alpaca I/O is mocked.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import bot_engine  # noqa: E402

_MACRO = {
    # QQQ above both MAs: no trend override, no bars fetch.
    "qqq_price": 500.0, "qqq_ma50": 480.0, "qqq_ma200": 460.0,
    "vxx_ratio": 1.2, "spy_vs_ma50": 1.0,
    "spy_below_200_days": 0, "spy_above_200d": True,
}
BASKET = {"QQQ": 0.50, "SMH": 0.15, "KWEB": 0.10, "VXUS": 0.15, "CASH": 0.10}


def _cfg(**over):
    cfg = {
        "FLOOR_TICKER": "QQQ",
        "FLOOR_BASKET_ENABLED": True,
        "FLOOR_BASKET": dict(BASKET),
        "FLOOR_BASKET_DRIFT_THRESHOLD": 0.03,
        "SPY_FLOOR_BULL": 0.70, "SPY_FLOOR_NEUTRAL": 0.90,
        "SPY_FLOOR_CAUTION": 0.35, "SPY_FLOOR_BEAR": 0.0, "SPY_FLOOR_PANIC": 0.0,
        "TREND_EXIT_DEATH_CROSS_CAP": 0.20,
        "TREND_EXIT_EARLY_WARNING_CAP": 0.50,
    }
    cfg.update(over)
    return cfg


def _stock(sym, qty, mv):
    return {"symbol": sym, "qty": str(qty), "qty_available": str(qty),
            "market_value": str(mv), "asset_class": "us_equity"}


def _short_call(sym):
    return {"symbol": sym, "qty": "-1", "market_value": "-200", "asset_class": "us_option"}


HELD = [_stock("QQQ", 90, 45000), _stock("SMH", 60, 15000),
        _stock("KWEB", 330, 10000), _stock("VXUS", 250, 15000)]


def _ok(status=200):
    m = MagicMock()
    m.status_code = status
    m.text = "" if status < 400 else "rejected"
    m.json.return_value = {"id": "test-order"}
    return m


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_floor_zero_")
        for p in (patch.object(bot_engine, "DATA_DIR", self.tmp),
                  patch.object(bot_engine, "_is_regular_hours", return_value=True),
                  patch.object(bot_engine, "data_feed", return_value="iex")):
            p.start()
            self.addCleanup(p.stop)

    def seed(self, **state):
        with open(os.path.join(self.tmp, "voltrade_floor_state.json"), "w") as f:
            json.dump(state, f)

    def state(self):
        with open(os.path.join(self.tmp, "voltrade_floor_state.json")) as f:
            return json.load(f)

    def run_floor(self, positions, regime="BEAR", cfg=None, post_side_effect=None,
                  get=None):
        post = MagicMock(side_effect=post_side_effect, return_value=_ok())
        with patch("system_config.BASE_CONFIG", cfg or _cfg()), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=positions), \
             patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000"}), \
             patch.object(bot_engine.requests, "get", side_effect=get or self._no_get), \
             patch.object(bot_engine.requests, "post", post):
            result = bot_engine._manage_spy_floor(dict(_MACRO), {"regime": regime})
        return result, [c.kwargs.get("json") or {} for c in post.call_args_list]

    @staticmethod
    def _no_get(url, *a, **kw):
        raise AssertionError(f"unexpected GET {url}")


def _sold(posts):
    return {p["symbol"]: int(p["qty"]) for p in posts if p.get("side") == "sell"}


class TestZeroTargetExitsWholeBasket(_Base):
    def test_bear_regime_change_sells_every_member(self):
        self.seed(last_regime="NEUTRAL")
        _, posts = self.run_floor(HELD)
        self.assertEqual(_sold(posts), {"QQQ": 90, "SMH": 60, "KWEB": 330, "VXUS": 250},
                         "0% target must exit every basket member, not only QQQ")
        self.assertEqual(self.state()["last_regime"], "BEAR")

    def test_covered_calls_unwound_before_each_member_sale(self):
        self.seed(last_regime="NEUTRAL")
        held = HELD + [_short_call("SMH261120C00300000"), _short_call("KWEB261120C00035000")]
        _, posts = self.run_floor(held)
        order = [(p["symbol"], p["side"]) for p in posts]
        for occ, under in (("SMH261120C00300000", "SMH"), ("KWEB261120C00035000", "KWEB")):
            self.assertIn((occ, "buy"), order, f"{under} covered call not bought back")
            self.assertLess(order.index((occ, "buy")), order.index((under, "sell")),
                            f"{under} sold before its covered call was closed")

    def test_failed_unwind_blocks_only_that_member_and_retries(self):
        self.seed(last_regime="NEUTRAL")
        held = HELD + [_short_call("KWEB261120C00035000")]

        def post(url, json=None, **kw):
            return _ok(500 if json and json.get("symbol") == "KWEB261120C00035000" else 200)

        result, posts = self.run_floor(held, post_side_effect=post)
        sold = _sold(posts)
        self.assertNotIn("KWEB", sold, "KWEB sold with a naked short call left open")
        self.assertEqual(set(sold), {"QQQ", "SMH", "VXUS"})
        self.assertIn("cc_unwind_blocked", [a.get("type") for a in result["actions"]])
        self.assertEqual(self.state()["last_regime"], "NEUTRAL",
                         "blocked exit must not persist last_regime (retry next cycle)")

    def test_already_orphaned_members_exit_once_without_regime_change(self):
        # Shipped while already in BEAR: QQQ was sold at the transition, the
        # members were not. No regime change from here on.
        self.seed(last_regime="BEAR")
        orphans = HELD[1:]
        _, posts = self.run_floor(orphans)
        self.assertEqual(set(_sold(posts)), {"SMH", "KWEB", "VXUS"})
        _, posts2 = self.run_floor(orphans)
        self.assertEqual(posts2, [], "zero-target exit must run once per episode, not every cycle")

    def test_pending_sell_not_doubled_on_retry(self):
        self.seed(last_regime="NEUTRAL")
        held = [dict(_stock("SMH", 60, 15000), qty_available="0"), _stock("VXUS", 250, 15000)]
        _, posts = self.run_floor(held)
        self.assertEqual(_sold(posts), {"VXUS": 250})

    def test_single_ticker_mode_unchanged(self):
        self.seed(last_regime="NEUTRAL")
        cfg = _cfg(FLOOR_BASKET_ENABLED=False)
        _, posts = self.run_floor(HELD, cfg=cfg)
        self.assertEqual(_sold(posts), {"QQQ": 90})


class TestBasketRebalanceSellUnwindsCalls(_Base):
    def test_sell_leaving_under_100_shares_unwinds_first(self):
        # CAUTION target 35%: SMH target = 100k*0.35*0.15 = $5,250; held
        # 120 sh @ $250 = $30,000 -> sell 99 sh, leaving 21 (< 100).
        self.seed(last_regime="CAUTION")
        held = [_stock("SMH", 120, 30000), _short_call("SMH261120C00300000")]
        cfg = _cfg(FLOOR_BASKET={"SMH": 0.15, "CASH": 0.85})

        def get(url, *a, **kw):
            m = MagicMock()
            m.json.return_value = {"latestTrade": {"p": 250.0}}
            return m

        _, posts = self.run_floor(held, regime="CAUTION", cfg=cfg, get=get)
        order = [(p["symbol"], p["side"]) for p in posts]
        self.assertIn(("SMH261120C00300000", "buy"), order,
                      "basket sell left a naked short call (no unwind)")
        self.assertLess(order.index(("SMH261120C00300000", "buy")), order.index(("SMH", "sell")))


if __name__ == "__main__":
    unittest.main()
