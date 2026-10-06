"""DD-halt trip confirmation + invalid-reading guard (2026-10-06, human-directed).

The 2026-09-09 portfolio DD halt tripped on ONE anomalous Alpaca paper-account
equity snapshot ($91,185 — the day's fills lost ~$415, the next day read
$101.5k) and then blocked every new entry for four weeks: the one-way ratchet
needs equity within 5% of peak, and nothing ever re-checked the trip itself.
A halt now still fires instantly, but the first reading >= 60 s later must
confirm it (>= 50% of the threshold) or it is released as a bad-data trip.
"""
import json
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("VOLTRADE_DATA_DIR", tempfile.mkdtemp(prefix="vt_dd_confirm_"))

import bot_engine  # noqa: E402


class _Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class TestDDHaltConfirmation(unittest.TestCase):
    def setUp(self):
        self.bot = bot_engine
        if os.path.exists(bot_engine._DD_STATE_PATH):
            os.remove(bot_engine._DD_STATE_PATH)
        self.clock = _Clock()
        self._orig_now = bot_engine._dd_now
        bot_engine._dd_now = self.clock
        bot_engine._last_dd_event = None

    def tearDown(self):
        bot_engine._dd_now = self._orig_now
        bot_engine._last_dd_event = None

    def _trip(self):
        self.bot.update_equity_peak(100_000, regime="BULL")
        r = self.bot.update_equity_peak(81_000, regime="BEAR")  # 19% -> trips
        self.assertTrue(r["halted"])
        self.assertFalse(r["halt_confirmed"])
        return r

    def test_real_drawdown_is_confirmed_and_ratchet_unchanged(self):
        self._trip()
        self.clock.t += 180  # next Tier-2 cycle
        r = self.bot.update_equity_peak(82_000, regime="BEAR")  # 18% — still deep
        self.assertTrue(r["halted"])
        self.assertTrue(r["halt_confirmed"])
        # once confirmed, a partial recovery (8% gap) in BULL does NOT release
        self.clock.t += 180
        r = self.bot.update_equity_peak(92_000, regime="BULL")
        self.assertTrue(r["halted"], "confirmed halt keeps the 5% one-way ratchet")
        # and the normal ratchet still resumes within 5% in BULL
        r = self.bot.update_equity_peak(96_000, regime="BULL")
        self.assertFalse(r["halted"])

    def test_bad_data_trip_is_released_and_recorded(self):
        self._trip()
        self.clock.t += 180
        r = self.bot.update_equity_peak(99_000, regime="BEAR")  # 1% — trip was bogus
        self.assertFalse(r["halted"], "re-read far below the trip releases it, regime aside")
        state = json.load(open(bot_engine._DD_STATE_PATH))
        rel = state["last_anomaly_release"]
        self.assertEqual(rel["trip_equity"], 81_000)
        self.assertEqual(rel["confirm_equity"], 99_000)
        self.assertIn("19.00%", rel["trip_reason"])
        self.assertEqual(bot_engine._last_dd_event["kind"], "anomaly_release")

    def test_reread_inside_window_does_not_decide(self):
        self._trip()
        self.clock.t += 30  # < 60 s: same-cycle noise, not a confirming read
        r = self.bot.update_equity_peak(99_000, regime="BEAR")
        self.assertTrue(r["halted"])
        self.assertFalse(r["halt_confirmed"])

    def test_production_legacy_halt_from_2026_09_09_is_released(self):
        # exact persisted shape + numbers from the live account on 2026-10-06
        legacy = {"peak_equity": 111736.69, "halted": True,
                  "halt_reason": "DD 18.39% >= 18.0% (peak=$111,737 cur=$91,185)",
                  "halt_started_at": "2026-09-09T19:58:00", "last_equity": 105947.52,
                  "last_updated": "2026-10-06T15:39:28"}
        os.makedirs(os.path.dirname(bot_engine._DD_STATE_PATH), exist_ok=True)
        json.dump(legacy, open(bot_engine._DD_STATE_PATH, "w"))
        r = self.bot.update_equity_peak(105947.52, regime="BEAR")  # 5.18% gap
        self.assertFalse(r["halted"], "legacy unconfirmed halt re-read at 5.18% (< 9%) is released")
        self.assertEqual(r["peak_equity"], 111736.69, "peak is never lowered")

    def test_legacy_halt_with_real_drawdown_is_confirmed_not_released(self):
        legacy = {"peak_equity": 100_000, "halted": True, "halt_reason": "DD 19.00% >= 18.0%",
                  "halt_started_at": "2026-09-01T00:00:00", "last_equity": 81_000}
        os.makedirs(os.path.dirname(bot_engine._DD_STATE_PATH), exist_ok=True)
        json.dump(legacy, open(bot_engine._DD_STATE_PATH, "w"))
        r = self.bot.update_equity_peak(88_000, regime="BULL")  # 12% — real
        self.assertTrue(r["halted"])
        self.assertTrue(r["halt_confirmed"])

    def test_invalid_equity_reading_never_trips_or_moves_peak(self):
        self.bot.update_equity_peak(100_000, regime="BULL")
        for bad in (0, 0.0, None, -5):
            r = self.bot.update_equity_peak(bad, regime="BEAR")
            self.assertFalse(r["halted"], f"equity {bad!r} must not trip the halt")
            self.assertTrue(r["invalid_reading"])
        self.assertFalse(self.bot.is_trading_halted())
        self.assertEqual(self.bot.get_portfolio_dd_state()["peak_equity"], 100_000)
        self.assertEqual(self.bot.get_portfolio_dd_state()["current_equity"], 100_000)


class TestReleaseIsAudited(unittest.TestCase):
    def test_scan_market_attaches_release_event_once(self):
        orig = bot_engine._scan_market_inner
        try:
            bot_engine._scan_market_inner = lambda: {"trades": [], "new_trades": []}
            bot_engine._last_dd_event = {"kind": "anomaly_release", "confirm_equity": 1}
            res = bot_engine.scan_market()
            self.assertEqual(res["dd_event"]["kind"], "anomaly_release")
            self.assertNotIn("dd_event", bot_engine.scan_market(), "attached once, then cleared")
        finally:
            bot_engine._scan_market_inner = orig
            bot_engine._last_dd_event = None

    def test_node_writes_a_dd_halt_audit_line_for_the_release(self):
        src = open(os.path.join(_REPO_ROOT, "server", "bot.ts")).read()
        self.assertIn('result.dd_event.kind === "anomaly_release"', src)
        self.assertIn("RELEASED as bad-data trip", src)


if __name__ == "__main__":
    unittest.main()
