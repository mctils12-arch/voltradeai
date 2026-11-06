#!/usr/bin/env python3
"""
SVXY third leg — every decision leaves a record (rule call "SVXY 10% stop /
24h cooldown", decided KEEP on 2026-11-06 under the human's "fix all of this").

The stop and the cooldown had never fired live (0 SVXY lots since 2026-05-07)
and nothing could ever judge them: bot.ts never read scan_market's
`third_leg`, a buy the cooldown blocked left no trace, an exit carried no
price, and a failed order was a debug line. These pin what _run_third_leg now
returns for server/leg3Audit.ts:
  1. a buy that ONLY the cooldown blocks -> leg3_cooldown_block with the price
     it would have paid (no order);
  2. a protective exit -> third_leg_exit with price, qty and the broker's answer;
  3. a buy -> vrp_harvest with the broker's answer;
  4. a failed order -> third_leg_error (no longer silent); a broker-clock
     deferral is NOT an error.
No network: every requests call is mocked.
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
import broker_clock  # noqa: E402
import system_config  # noqa: E402

SVXY_PX = 50.0


def _resp(payload, ok=None):
    m = MagicMock()
    m.json.return_value = payload
    if ok is not None:
        m.ok = ok
    return m


class TestThirdLegAuditActions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_leg3_audit_")
        for p in (patch.object(bot_engine, "DATA_DIR", self.tmp),
                  patch.dict(os.environ, {"VOLTRADE_SHADOW_ENTRIES": "off"}),
                  patch.dict(system_config.BASE_CONFIG, {"CASH_RESERVE_ENABLED": False})):
            p.start()
            self.addCleanup(p.stop)
        self.ts = 2_000_000

    def seed(self, **state):
        with open(os.path.join(self.tmp, "voltrade_leg3_state.json"), "w") as f:
            json.dump(state, f)

    def run_leg(self, vxx_ratio, held_plpc=None, post=None):
        self.ts += 300
        positions = []
        if held_plpc is not None:
            positions.append({"symbol": "SVXY", "qty": "300", "market_value": "15000",
                              "unrealized_plpc": str(held_plpc), "current_price": "45.10"})

        def fake_get(url, *a, **kw):
            if url.endswith("/v2/account"):
                return _resp({"equity": "100000"})
            if url.endswith("/v2/positions"):
                return _resp(positions)
            if url.endswith("/v2/orders"):
                return _resp([])
            if url.endswith("/v2/stocks/bars"):
                return _resp({"bars": {"VXX": [{"c": c} for c in (30, 29, 28, 27, 26, 25)]}})
            if url.endswith("/v2/stocks/snapshots"):
                return _resp({"SVXY": {"latestTrade": {"p": SVXY_PX}}})
            return _resp({})

        post = post or MagicMock(return_value=_resp({"id": "ord-1"}, ok=True))
        macro = {"vxx_ratio": vxx_ratio, "spy_vs_ma50": 1.0, "spy_below_200_days": 0, "spy_above_200d": True,
                 "vxx_closes": [20.0] * 30, "vxx_latest": 20.0 * vxx_ratio, "timestamp": float(self.ts)}
        with patch("requests.get", side_effect=fake_get), patch("requests.post", post):
            result = bot_engine._run_third_leg(macro)
        return result, [c.kwargs.get("json") or {} for c in post.call_args_list]

    def test_buy_blocked_only_by_cooldown_is_recorded_with_its_price(self):
        exit_ts = time.time() - 3600
        self.seed(zone="BAND", last_protective_exit_ts=exit_ts)
        result, posts = self.run_leg(1.10)
        self.assertEqual([p for p in posts if p.get("side") == "buy"], [], "the cooldown must still block the buy")
        blocks = [a for a in result["actions"] if a.get("type") == "leg3_cooldown_block"]
        self.assertEqual(len(blocks), 1, result["actions"])
        b = blocks[0]
        self.assertEqual(b["symbol"], "SVXY")
        self.assertEqual(b["price"], SVXY_PX)
        self.assertEqual(b["vxx_ratio"], 1.1)
        hours = float(system_config.BASE_CONFIG["LEG3_VRP_REENTRY_COOLDOWN_HOURS"])
        self.assertAlmostEqual(b["cooldown_until"], exit_ts + hours * 3600, places=3)
        self.assertIn("re-entry cooldown", b["reason"])

    def test_no_block_record_outside_the_band_or_without_a_cooldown(self):
        self.seed(zone="BAND", last_protective_exit_ts=time.time() - 3600)
        result, _ = self.run_leg(1.02)  # below the band: no buy would happen anyway
        self.assertEqual([a for a in result["actions"] if a.get("type") == "leg3_cooldown_block"], [])
        self.seed(zone="BAND")  # no cooldown: a real buy, not a block
        result, posts = self.run_leg(1.10)
        self.assertEqual(len([p for p in posts if p.get("side") == "buy"]), 1)
        buys = [a for a in result["actions"] if a.get("type") == "vrp_harvest"]
        self.assertEqual(buys[0]["ok"], True, "the broker's answer rides the buy action")
        self.assertEqual([a for a in result["actions"] if a.get("type") == "leg3_cooldown_block"], [])

    def test_protective_exit_carries_price_qty_and_broker_answer(self):
        self.seed(zone="BAND")
        stop = float(system_config.BASE_CONFIG["LEG3_VRP_STOP_PCT"])
        result, posts = self.run_leg(1.10, held_plpc=-(stop + 0.01))
        exits = [a for a in result["actions"] if a.get("type") == "third_leg_exit"]
        self.assertEqual(len(exits), 1)
        e = exits[0]
        self.assertEqual((e["trigger"], e["price"], e["qty"], e["ok"]), ("stop_loss", 45.10, "300", True))
        rejected = MagicMock(return_value=_resp({"code": 40310000}, ok=False))
        self.seed(zone="BAND")
        result, _ = self.run_leg(1.30, held_plpc=-0.02, post=rejected)
        self.assertEqual([a["ok"] for a in result["actions"] if a.get("type") == "third_leg_exit"], [False])

    def test_unknown_response_is_none_never_true(self):
        self.assertIsNone(bot_engine._leg3_order_ok(MagicMock()))
        self.assertIsNone(bot_engine._leg3_order_ok(None))
        self.assertIs(bot_engine._leg3_order_ok(_resp({}, ok=True)), True)

    def test_failed_order_is_an_error_action_and_starts_no_cooldown(self):
        self.seed(zone="BAND")
        boom = MagicMock(side_effect=RuntimeError("connection reset"))
        result, _ = self.run_leg(1.30, held_plpc=-0.02, post=boom)
        errs = [a for a in result["actions"] if a.get("type") == "third_leg_error"]
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0]["stage"], "exit (vol_spike)")
        self.assertIn("connection reset", errs[0]["error"])
        with open(os.path.join(self.tmp, "voltrade_leg3_state.json")) as f:
            self.assertFalse(json.load(f).get("last_protective_exit_ts"), "a failed exit must not start the cooldown")

    def test_broker_clock_deferral_is_not_an_error(self):
        self.seed(zone="BAND")
        with patch.object(broker_clock, "check_order_allowed", side_effect=broker_clock.OrderBlocked("market closed")):
            result, posts = self.run_leg(1.30, held_plpc=-0.02)
        self.assertEqual(posts, [])
        self.assertEqual([a for a in result["actions"] if a.get("type") in ("third_leg_error", "third_leg_exit")], [])


if __name__ == "__main__":
    unittest.main()
