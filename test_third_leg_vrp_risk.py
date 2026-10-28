#!/usr/bin/env python3
"""
VolTradeAI — third-leg (SVXY VRP harvest) risk regression tests (audit §6.8 N14/N20).

SVXY is a short-volatility ETF. _run_third_leg (bot_engine.py) bought it when
the VXX ratio sat in 1.05-1.25 and only ever exited on a NEUTRAL/BULL regime.
SVXY is in bot.ts FLOOR_AND_LEG_TICKERS, so it gets no websocket stop and no
POS-KILL, and Python manage_positions skips it too. These tests pin:
  1. size: LEG3_VRP_PCT is capped at the documented 15% sweet spot (was 40%).
  2. protective exits (N14): SVXY is sold IMMEDIATELY when the VXX ratio
     rises above the buy band's top (the vol spike it is short) or when its
     loss from entry reaches LEG3_VRP_STOP_PCT; re-entry then waits out a
     cooldown.
  3. dwell (N20): the buy (VXX band) and the take-profit exit (NEUTRAL/BULL)
     each need a zone confirmed on LEG3_REGIME_CONFIRM_CYCLES consecutive
     DISTINCT macro readings — VXX-ratio flicker at 1.05 no longer round-trips.
No live Alpaca/network connectivity — every requests call is mocked;
get_market_regime is REAL.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import bot_engine  # noqa: E402
import system_config  # noqa: E402

EQUITY = 100000.0
SVXY_PX = 50.0


def _macro(vxx_ratio, ts):
    """A REAL VXX reading (vxx_closes/vxx_latest present) stamped like a
    macro_data snapshot; SPY calm so the regime is driven by VXX alone."""
    return {
        "vxx_ratio": vxx_ratio, "spy_vs_ma50": 1.0,
        "spy_below_200_days": 0, "spy_above_200d": True,
        "vxx_closes": [20.0] * 30, "vxx_latest": 20.0 * vxx_ratio,
        "timestamp": float(ts),
    }


def _resp(payload):
    m = MagicMock()
    m.json.return_value = payload
    return m


class _Leg3Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_leg3_risk_")
        p = patch.object(bot_engine, "DATA_DIR", self.tmp)
        p.start()
        self.addCleanup(p.stop)
        self.ts = 1_000_000

    def seed(self, **state):
        with open(os.path.join(self.tmp, "voltrade_leg3_state.json"), "w") as f:
            json.dump(state, f)

    def state(self):
        with open(os.path.join(self.tmp, "voltrade_leg3_state.json")) as f:
            return json.load(f)

    def run_leg(self, vxx_ratio, svxy_plpc=None, same_reading=False):
        """One scan. svxy_plpc=None -> SVXY not held. Returns (result, posts)."""
        if not same_reading:
            self.ts += 300
        positions = []
        if svxy_plpc is not None:
            positions.append({"symbol": "SVXY", "qty": "300", "market_value": "15000",
                              "unrealized_plpc": str(svxy_plpc)})

        def fake_get(url, *a, **kw):
            if url.endswith("/v2/account"):
                return _resp({"equity": str(EQUITY)})
            if url.endswith("/v2/positions"):
                return _resp(positions)
            if url.endswith("/v2/orders"):
                return _resp([])
            if url.endswith("/v2/stocks/bars"):
                # VXX declining over 5 sessions (buy precondition).
                return _resp({"bars": {"VXX": [{"c": c} for c in (30, 29, 28, 27, 26, 25)]}})
            if url.endswith("/v2/stocks/snapshots"):
                return _resp({"SVXY": {"latestTrade": {"p": SVXY_PX}}})
            return _resp({})

        post = MagicMock(return_value=_resp({"id": "test-order"}))
        with patch("requests.get", side_effect=fake_get), patch("requests.post", post):
            result = bot_engine._run_third_leg(_macro(vxx_ratio, self.ts))
        posts = [c.kwargs.get("json") or {} for c in post.call_args_list]
        return result, posts


def _sells(posts):
    return [p for p in posts if p.get("side") == "sell" and p.get("symbol") == "SVXY"]


def _buys(posts):
    return [p for p in posts if p.get("side") == "buy" and p.get("symbol") == "SVXY"]


class TestLeg3SizeCap(unittest.TestCase):
    def test_vrp_pct_capped_at_sweet_spot(self):
        vrp = float(system_config.BASE_CONFIG["LEG3_VRP_PCT"])
        self.assertGreater(vrp, 0.0, "VRP leg must stay enabled")
        self.assertLessEqual(
            vrp, 0.15,
            f"LEG3_VRP_PCT={vrp}: an unstopped short-vol leg above the 15% "
            "backtest sweet spot (was 0.40) — audit N14")


class TestProtectiveExits(_Leg3Base):
    def test_vol_spike_above_band_exits_svxy_immediately(self):
        # Held from a confirmed BAND entry; VXX ratio jumps to 1.30 (BEAR, not
        # PANIC). Old code: regime BEAR -> no exit at all.
        self.seed(zone="BAND")
        result, posts = self.run_leg(1.30, svxy_plpc=-0.03)
        sells = _sells(posts)
        self.assertEqual(len(sells), 1, f"vol spike must exit SVXY now: {posts}")
        self.assertEqual(sells[0]["qty"], "300")
        exits = [a for a in result["actions"] if a.get("type") == "third_leg_exit"]
        self.assertEqual(exits[0]["trigger"], "vol_spike")
        self.assertGreater(self.state().get("last_protective_exit_ts", 0), 0)

    def test_spike_exit_ignores_dwell_even_from_recovery(self):
        self.seed(zone="RECOVERY")
        _, posts = self.run_leg(1.50, svxy_plpc=0.02)  # PANIC-level spike
        self.assertEqual(len(_sells(posts)), 1)

    def test_stop_loss_from_entry_exits_inside_band(self):
        self.seed(zone="BAND")
        stop = float(system_config.BASE_CONFIG["LEG3_VRP_STOP_PCT"])
        result, posts = self.run_leg(1.10, svxy_plpc=-(stop + 0.02))
        self.assertEqual(len(_sells(posts)), 1, f"stop must exit SVXY: {posts}")
        exits = [a for a in result["actions"] if a.get("type") == "third_leg_exit"]
        self.assertEqual(exits[0]["trigger"], "stop_loss")

    def test_small_loss_inside_band_holds(self):
        self.seed(zone="BAND")
        _, posts = self.run_leg(1.10, svxy_plpc=-0.04)
        self.assertEqual(posts, [], "no exit and no duplicate buy while held in band")

    def test_no_rebuy_during_cooldown_then_rebuy_after(self):
        self.seed(zone="BAND")
        self.run_leg(1.30, svxy_plpc=-0.03)              # spike exit, cooldown starts
        for _ in range(4):                               # vol calms back into band, confirmed
            _, posts = self.run_leg(1.10, svxy_plpc=None)
            self.assertEqual(_buys(posts), [], "re-bought SVXY inside the cooldown")
        self.assertEqual(self.state()["zone"], "BAND")
        st = self.state()
        hours = float(system_config.BASE_CONFIG["LEG3_VRP_REENTRY_COOLDOWN_HOURS"])
        st["last_protective_exit_ts"] = time.time() - (hours + 1) * 3600
        self.seed(**st)
        _, posts = self.run_leg(1.10, svxy_plpc=None)
        self.assertEqual(len(_buys(posts)), 1, "re-entry must resume after the cooldown")


class TestDwell(_Leg3Base):
    def test_recovery_exit_needs_confirmed_readings(self):
        # Old code sold on the FIRST NEUTRAL reading (vxx 1.04 < 1.05).
        self.seed(zone="BAND")
        n = int(system_config.BASE_CONFIG["LEG3_REGIME_CONFIRM_CYCLES"])
        self.assertGreaterEqual(n, 2, "test premise: a real dwell is configured")
        for i in range(n - 1):
            _, posts = self.run_leg(1.04, svxy_plpc=0.05)
            self.assertEqual(_sells(posts), [], f"sold on unconfirmed reading {i + 1}")
        _, posts = self.run_leg(1.04, svxy_plpc=0.05)
        self.assertEqual(len(_sells(posts)), 1, "confirmed recovery must take profit")

    def test_repeated_snapshot_does_not_advance_dwell(self):
        self.seed(zone="BAND")
        self.run_leg(1.04, svxy_plpc=0.05)
        for _ in range(5):  # same cached macro snapshot seen on later scans
            _, posts = self.run_leg(1.04, svxy_plpc=0.05, same_reading=True)
            self.assertEqual(_sells(posts), [])
        self.assertEqual(self.state()["zone_pending_count"], 1)

    def test_flicker_at_band_floor_never_round_trips(self):
        self.seed(zone="BAND")
        for vxx in (1.04, 1.06, 1.04, 1.06, 1.049, 1.051, 1.04, 1.06):
            _, posts = self.run_leg(vxx, svxy_plpc=0.01)
            self.assertEqual(_sells(posts), [], f"flicker sold SVXY at {vxx}")
        self.assertEqual(self.state()["zone"], "BAND")

    def test_buy_needs_confirmed_band(self):
        # Old code bought on the FIRST in-band reading after a calm regime.
        self.seed(zone="RECOVERY")
        n = int(system_config.BASE_CONFIG["LEG3_REGIME_CONFIRM_CYCLES"])
        for i in range(n - 1):
            _, posts = self.run_leg(1.10, svxy_plpc=None)
            self.assertEqual(_buys(posts), [], f"bought on unconfirmed band reading {i + 1}")
        _, posts = self.run_leg(1.10, svxy_plpc=None)
        buys = _buys(posts)
        self.assertEqual(len(buys), 1, "confirmed band must buy")
        vrp = float(system_config.BASE_CONFIG["LEG3_VRP_PCT"])
        self.assertEqual(int(buys[0]["qty"]), int(EQUITY * vrp / SVXY_PX))

    def test_fallback_macro_holds_confirmed_zone(self):
        self.seed(zone="BAND")
        fallback = {"vxx_ratio": 1.0, "spy_vs_ma50": 1.0, "spy_below_200_days": 0,
                    "spy_above_200d": True, "vxx_closes": []}
        for i in range(5):
            positions = [{"symbol": "SVXY", "qty": "300", "market_value": "15000",
                          "unrealized_plpc": "0.02"}]
            post = MagicMock(return_value=_resp({"id": "x"}))

            def fake_get(url, *a, **kw):
                if url.endswith("/v2/account"):
                    return _resp({"equity": str(EQUITY)})
                if url.endswith("/v2/positions"):
                    return _resp(positions)
                return _resp([] if url.endswith("/v2/orders") else {})

            with patch("requests.get", side_effect=fake_get), patch("requests.post", post):
                bot_engine._run_third_leg(dict(fallback, timestamp=float(i)))
            self.assertEqual(post.call_count, 0, "fallback VXX reading moved the leg")
        self.assertEqual(self.state()["zone"], "BAND")


if __name__ == "__main__":
    unittest.main()
