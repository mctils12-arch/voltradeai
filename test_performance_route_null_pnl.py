"""/api/bot/performance's inline Python, run for real against a feedback file
that carries what the live file always carries while a position is open: an
entry record with pnl_pct None. [RULE-REVIEW] 2026-10-22: before the null
filter the script raised TypeError on 'None > 0', the route fell back to the
Alpaca today-only path and the KPI strip showed that instead of the history."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
BOT_TS = os.path.join(HERE, "server", "bot.ts")


def _performance_script(feedback_path: str) -> str:
    src = open(BOT_TS, encoding="utf-8").read()
    anchor = src.index('app.get("/api/bot/performance"')
    start = src.index('python3 -c "', anchor) + len('python3 -c "')
    end = src.index('"`', start)
    script = src[start:end]
    assert "from ml_model_v2 import fills_slippage_stats" in script and "'totalTrades': len(feedback)" in script, "performance script moved"
    # point the script at the fixture instead of storage_config's path
    imp = "    from storage_config import TRADE_FEEDBACK_PATH\n"
    assert script.count(imp) == 1
    return script.replace(imp, f"    TRADE_FEEDBACK_PATH = {feedback_path!r}\n")


def _run(records):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "voltrade_trade_feedback.json")
        with open(path, "w") as fh:
            json.dump(records, fh)
        proc = subprocess.run([sys.executable, "-c", _performance_script(path)], cwd=HERE, capture_output=True, text=True, timeout=120)
    return proc


FIXTURE = [
    {"ticker": "NVDA", "pnl_pct": None, "outcome": None, "entry_time": "2026-10-22T14:00:00Z"},          # open entry (track_fill)
    {"ticker": "AMD", "pnl_pct": None, "outcome": "orphan_exit", "exit_time": "2026-10-21T19:00:00Z"},  # orphan exit
    {"ticker": "SMH", "pnl_pct": 4.0, "outcome": "win", "strategy": "stocks"},
    {"ticker": "KWEB", "pnl_pct": -2.0, "outcome": "loss"},
    {"ticker": "QQQ", "pnl_pct": 1.0, "outcome": "win"},
    {"ticker": "GLD", "pnl_pct": 9.0, "outcome": "win", "_seed": True},                                   # backtest seed: never a trade
    {"ticker": "", "pnl_pct": 3.0, "outcome": "win"},                                                     # corrupt: no ticker
]


class TestPerformanceRouteNullPnl(unittest.TestCase):
    def test_null_pnl_records_do_not_crash_the_route_into_its_fallback(self):
        proc = _run(FIXTURE)
        self.assertEqual(proc.returncode, 0, f"the performance script failed — the route would have served the Alpaca fallback:\n{proc.stderr[-800:]}")
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        # the three closed, numeric, non-seeded records — not the open entry, not the orphan exit, not the seed
        self.assertEqual(out["totalTrades"], 3)
        self.assertEqual(out["seededTrades"], 1)
        self.assertAlmostEqual(out["winRate"], 66.7, places=1)
        self.assertAlmostEqual(out["avgGain"], 2.5, places=2)
        self.assertAlmostEqual(out["avgLoss"], -2.0, places=2)
        self.assertAlmostEqual(out["totalPnlPct"], 3.0, places=2)
        self.assertEqual({t["ticker"] for t in out["recentTrades"]}, {"SMH", "KWEB", "QQQ"})
        self.assertTrue(all(t.get("pnl_pct") is not None for t in out["recentTrades"]))

    def test_the_bot_log_number_says_what_it_excludes_and_when_it_last_moved(self):
        # [RULE-REVIEW] 2026-10-29 (the Expectancy tile): beside the number, the records the
        # filter dropped (orphan exits, open entries — never the seed), the last counted close
        # (exit_time first, the entry fill time for a record without one; naive and offset
        # timestamps ordered on their yyyy-mm-ddThh:mm:ss prefix) and the basis caption.
        records = [
            {"ticker": "NVDA", "pnl_pct": None, "outcome": None, "time_filled": "2026-10-28T14:00:00"},               # open entry
            {"ticker": "AMD", "pnl_pct": None, "outcome": "orphan_exit", "time_filled": "2026-10-27T19:00:00Z"},      # orphan exit
            {"ticker": "CRWV261120P00055000", "pnl_pct": None, "outcome": "orphan_exit", "time_filled": "2026-10-26T19:00:00Z"},
            {"ticker": "SMH", "pnl_pct": 4.0, "outcome": "win", "time_filled": "2026-09-30T14:00:00", "exit_time": "2026-10-14T19:59:02.123456"},
            {"ticker": "KWEB", "pnl_pct": -2.0, "outcome": "loss", "time_filled": "2026-10-01T14:00:00+00:00", "exit_time": "2026-10-09T15:30:00+00:00"},
            {"ticker": "PENG", "pnl_pct": -5.0, "outcome": "loss", "time_filled": "2026-10-20T14:00:00"},                 # closed, no exit_time recorded: its fill time stands in
            {"ticker": "GLD", "pnl_pct": 9.0, "outcome": "win", "_seed": True, "exit_time": "2026-10-28T19:00:00"},     # seed: neither counted nor excluded
        ]
        proc = _run(records)
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out["totalTrades"], 3)
        self.assertEqual(out["excluded"], {"orphanExits": 2, "openEntries": 1})
        self.assertEqual(out["lastClosedAt"], "2026-10-20T14:00:00", "PENG's fill time outranks SMH's 10-14 exit")
        self.assertEqual(out["basis"], "trigger price, not fill")
        # every pre-existing field is still there for the other consumers
        for key in ("winRate", "avgGain", "avgLoss", "totalPnlPct", "profitFactor", "byStrategy", "recentTrades", "realisticPnlPct", "avgSlippagePct", "totalSlippageCost", "slippageBasis", "slippageGapPct", "bestTrade", "worstTrade", "seededTrades", "totalFills"):
            self.assertIn(key, out)
        # no counted record carries a close time -> null, never a made-up date
        bare = _run([{"ticker": "SMH", "pnl_pct": 4.0, "outcome": "win"}])
        self.assertIsNone(json.loads(bare.stdout.strip().splitlines()[-1])["lastClosedAt"])

    def test_the_old_filter_did_crash_on_the_same_fixture(self):
        # the regression this pins: the pre-2026-10-22 filter kept the null records and the
        # next comparison raised — kept here so the fallback path can never again be the KPI
        feedback = [t for t in FIXTURE if not t.get("_seed")]
        feedback = [t for t in feedback if t.get("ticker", "").strip() and not (t.get("pnl_pct", 0) == 0 and t.get("outcome") is None)]
        with self.assertRaises(TypeError):
            [t for t in feedback if t.get("pnl_pct", 0) > 0]


if __name__ == "__main__":
    unittest.main()
