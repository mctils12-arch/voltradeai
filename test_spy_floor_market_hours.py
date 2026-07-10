#!/usr/bin/env python3
"""
VolTradeAI — SPY Floor market-hours regression tests (KNOWN BROKEN #16)

_manage_spy_floor() (bot_engine.py) rebalances the passive floor basket
(QQQ/SMH/KWEB/VXUS/...) on every Tier-2 scan cycle (~5 min), around the
clock. Its basket-rebalance branch submitted raw "type": "market" orders
directly via requests.post with zero market-hours awareness — Alpaca
rejects/cancels a market order placed outside 9:30am-4:00pm ET, so the
next cycle's drift check saw the same gap and resubmitted, forever. Live
evidence (2026-07-10, /api/diag/orders): 50+ SMH buy-market orders
canceled over 3 days, ~5 minutes apart, entirely during extended hours.

These tests pin the fix: _is_regular_hours() gates every direct order
submission in this function, deferring (not silently dropping) the
rebalance to the next regular-hours cycle instead of looping forever.
No live Alpaca/network connectivity — every requests.get/post call this
function makes is mocked.
"""
import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import bot_engine  # noqa: E402

_MACRO = {
    # Keeps the QQQ trend-override branch from firing its own extra
    # requests.get('.../bars') call — price comfortably above both MAs.
    "qqq_price": 500.0, "qqq_ma50": 480.0, "qqq_ma200": 460.0,
    "vxx_ratio": 1.0, "spy_vs_ma50": 1.0,
    "spy_below_200_days": 0, "spy_above_200d": True,
}


def _snapshot_response(price):
    m = MagicMock()
    m.json.return_value = {"latestTrade": {"p": price}}
    return m


class _FloorTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vt_floor_hours_")
        self._patches = [
            patch.object(bot_engine, "DATA_DIR", self._tmpdir),
            patch.object(bot_engine, "data_feed", return_value="iex"),
            patch("system_config.get_market_regime", return_value="BULL"),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _floor_state_path(self):
        return os.path.join(self._tmpdir, "voltrade_floor_state.json")


class TestBasketRebalanceDeferredWhenClosed(_FloorTestBase):
    """FLOOR_BASKET_ENABLED path: buy/sell must defer, never POST, when closed."""

    def _base_config(self, **overrides):
        cfg = {
            "FLOOR_TICKER": "QQQ",
            "FLOOR_BASKET_ENABLED": True,
            "FLOOR_BASKET": {"SMH": 1.0},
            "FLOOR_BASKET_DRIFT_THRESHOLD": 0.03,
            "SPY_FLOOR_BULL": 0.5,
            "TREND_EXIT_DEATH_CROSS_CAP": 0.20,
            "TREND_EXIT_EARLY_WARNING_CAP": 0.50,
        }
        cfg.update(overrides)
        return cfg

    def test_buy_deferred_no_post_when_market_closed(self):
        with patch.object(bot_engine, "_is_regular_hours", return_value=False), \
             patch("system_config.BASE_CONFIG", self._base_config()), \
             patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000"}), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=[]), \
             patch.object(bot_engine.requests, "get", return_value=_snapshot_response(100.0)), \
             patch.object(bot_engine.requests, "post") as mock_post:
            result = bot_engine._manage_spy_floor(_MACRO)

        mock_post.assert_not_called()
        actions = result["basket"]["actions"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "buy_deferred")
        self.assertEqual(actions[0]["ticker"], "SMH")

    def test_sell_deferred_no_post_when_market_closed(self):
        # Currently holding SMH worth 80% of equity; BULL target is 50% —
        # this must attempt a sell, and the sell must also defer.
        held = [{"symbol": "SMH", "market_value": "80000"}]
        with patch.object(bot_engine, "_is_regular_hours", return_value=False), \
             patch("system_config.BASE_CONFIG", self._base_config()), \
             patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000"}), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=held), \
             patch.object(bot_engine.requests, "get", return_value=_snapshot_response(100.0)), \
             patch.object(bot_engine.requests, "post") as mock_post:
            result = bot_engine._manage_spy_floor(_MACRO)

        mock_post.assert_not_called()
        actions = result["basket"]["actions"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "sell_deferred")

    def test_buy_still_executes_when_market_open(self):
        """Regression guard: the fix must not disable trading during regular hours."""
        with patch.object(bot_engine, "_is_regular_hours", return_value=True), \
             patch("system_config.BASE_CONFIG", self._base_config()), \
             patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000"}), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=[]), \
             patch.object(bot_engine.requests, "get", return_value=_snapshot_response(100.0)), \
             patch.object(bot_engine.requests, "post") as mock_post:
            result = bot_engine._manage_spy_floor(_MACRO)

        mock_post.assert_called_once()
        posted_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(posted_json["symbol"], "SMH")
        self.assertEqual(posted_json["type"], "market")
        actions = result["basket"]["actions"]
        self.assertEqual(actions[0]["action"], "buy")


class TestLegacyRebalanceDeferredWhenClosed(_FloorTestBase):
    """FLOOR_BASKET_ENABLED=False (single-ticker legacy) path defers too."""

    def _base_config(self, **overrides):
        cfg = {
            "FLOOR_TICKER": "QQQ",
            "FLOOR_BASKET_ENABLED": False,
            "FLOOR_BASKET": {},
            "SPY_FLOOR_BULL": 0.5,
            "TREND_EXIT_DEATH_CROSS_CAP": 0.20,
            "TREND_EXIT_EARLY_WARNING_CAP": 0.50,
        }
        cfg.update(overrides)
        return cfg

    def _mock_get(self, url, headers=None, params=None, timeout=None):
        if url.endswith("/v2/account"):
            m = MagicMock(); m.json.return_value = {"equity": "100000"}; return m
        if url.endswith("/v2/stocks/snapshots"):
            m = MagicMock()
            m.json.return_value = {"QQQ": {"latestTrade": {"p": 500.0}}}
            return m
        raise AssertionError(f"unexpected GET {url}")

    def test_buy_deferred_no_post_when_market_closed(self):
        with patch.object(bot_engine, "_is_regular_hours", return_value=False), \
             patch("system_config.BASE_CONFIG", self._base_config()), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=[]), \
             patch.object(bot_engine.requests, "get", side_effect=self._mock_get), \
             patch.object(bot_engine.requests, "post") as mock_post:
            result = bot_engine._manage_spy_floor(_MACRO)

        mock_post.assert_not_called()
        kinds = [a.get("type") for a in result["actions"]]
        self.assertIn("floor_rebalance_deferred", kinds)


class TestZeroTargetExitDeferredPreservesRetry(_FloorTestBase):
    """target_pct<=0 exit-on-regime-change path: a deferred exit must NOT
    persist last_regime, so the next call still sees regime_changed=True
    and retries — otherwise the exit is silently forgotten forever."""

    def _base_config(self):
        return {
            "FLOOR_TICKER": "QQQ",
            "FLOOR_BASKET_ENABLED": False,
            "FLOOR_BASKET": {},
            "SPY_FLOOR_BULL": 0.0,   # target is zero -> exit branch
            "TREND_EXIT_DEATH_CROSS_CAP": 0.20,
            "TREND_EXIT_EARLY_WARNING_CAP": 0.50,
        }

    def test_deferred_exit_does_not_update_last_regime(self):
        # Seed state as if we were previously in a different regime, so
        # regime_changed=True on this call.
        with open(self._floor_state_path(), "w") as f:
            json.dump({"last_regime": "BEAR"}, f)

        held = [{"symbol": "QQQ", "qty": "10", "market_value": "5000"}]
        with patch.object(bot_engine, "_is_regular_hours", return_value=False), \
             patch("system_config.BASE_CONFIG", self._base_config()), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=held), \
             patch.object(bot_engine.requests, "post") as mock_post:
            result = bot_engine._manage_spy_floor(_MACRO)

        mock_post.assert_not_called()
        kinds = [a.get("type") for a in result["actions"]]
        self.assertIn("floor_exit_deferred", kinds)

        # last_regime must be unchanged (still BEAR, or file untouched) so
        # regime_changed re-evaluates True and the exit retries next call.
        with open(self._floor_state_path()) as f:
            state = json.load(f)
        self.assertEqual(state.get("last_regime"), "BEAR")


if __name__ == "__main__":
    unittest.main()
