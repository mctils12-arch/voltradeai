#!/usr/bin/env python3
"""
BROKER CLOCK ORDER GATE (2026-11-05, human-directed): no Python order is sent
while Alpaca's clock says the market is closed.

The Tier 2 scan runs 04:00-20:00 ET every day, weekends and holidays included,
and bot_engine._is_regular_hours read only the time of day; the GLD defensive
floor, the third leg, the convexity overlay and intraday shorts had no hours
check at all. A DAY order sent while the market is closed is queued by Alpaca
for the next opening print.

Pinned here:
  - the reader (broker_clock.py): open; closed; a half-day close taking
    effect at 13:00 (the cached reading's own next_close); a failed read
    extrapolated from the last good reading; no reading at all -> allowed
    ("clock unknown"); the TTL cache; a broken response never raises;
  - the gate: OrderBlocked (a RuntimeError) while closed, one log line per
    symbol per ET day;
  - representative call sites send nothing while closed: the old basket floor
    (so a rollback to FLOOR_MODE "basket" is safe), the GLD defensive floor,
    the volatility-managed floor, an options_manager close (single leg and
    mleg), submit_options_order and its mleg twin, the covered-call buy-back,
    intraday short covers;
  - on the source: every order POST in the engine sits behind the gate.
All broker I/O is mocked; the clock is served through
broker_clock._fetch_clock_body (the root conftest stubs it to "no reading" for
every other test).
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from contextlib import ExitStack, contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import broker_clock  # noqa: E402
import bot_engine  # noqa: E402
import intraday_shorts  # noqa: E402
import options_execution  # noqa: E402
import options_manager as om  # noqa: E402

# Captured at import (collection), before the conftest fixture stubs it.
_REAL_FETCH = broker_clock._fetch_clock_body


def ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


# Thu 2026-11-05 is a normal session; Sat 2026-11-07 is a weekend; Fri
# 2026-11-27 closes early at 13:00 ET (market_calendar.py HALF_DAYS_2026).
OPEN_THU = {"timestamp": "2026-11-05T11:00:00.123456789-05:00", "is_open": True,
            "next_open": "2026-11-06T09:30:00-05:00", "next_close": "2026-11-05T16:00:00-05:00"}
CLOSED_SAT = {"timestamp": "2026-11-07T11:00:00-05:00", "is_open": False,
              "next_open": "2026-11-09T09:30:00-05:00", "next_close": "2026-11-09T16:00:00-05:00"}
CLOSED_MON_PRE = {"timestamp": "2026-11-09T09:29:00-05:00", "is_open": False,
                  "next_open": "2026-11-09T09:30:00-05:00", "next_close": "2026-11-09T16:00:00-05:00"}
HALF_DAY_OPEN = {"timestamp": "2026-11-27T12:59:00-05:00", "is_open": True,
                 "next_open": "2026-11-30T09:30:00-05:00", "next_close": "2026-11-27T13:00:00-05:00"}
HALF_DAY_CLOSED = {"timestamp": "2026-11-27T13:00:05-05:00", "is_open": False,
                   "next_open": "2026-11-30T09:30:00-05:00", "next_close": "2026-11-30T16:00:00-05:00"}


class _Served:
    """broker_clock._fetch_clock_body stand-in: one body per read (the last
    one repeats), counting reads."""

    def __init__(self, bodies):
        self.bodies, self.reads = list(bodies), 0

    def __call__(self):
        self.reads += 1
        return self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]


@contextmanager
def broker(*bodies, at=None):
    """Serve these clock bodies; with `at`, freeze both clocks the code reads
    (the gate's time.time and bot_engine's datetime.now) at that ET instant."""
    served = _Served(bodies)
    with ExitStack() as stack:
        stack.enter_context(patch.object(broker_clock, "_fetch_clock_body", served))
        if at is not None:
            frozen = datetime.fromisoformat(at)

            class _Frozen(datetime):
                @classmethod
                def now(cls, tz=None):
                    return frozen.astimezone(tz) if tz is not None else frozen.replace(tzinfo=None)

            stack.enter_context(patch.object(broker_clock, "time", MagicMock(time=lambda: frozen.timestamp())))
            stack.enter_context(patch.object(bot_engine, "datetime", _Frozen))
        broker_clock.reset_cache()
        try:
            yield served
        finally:
            broker_clock.reset_cache()


# ── the reader ──────────────────────────────────────────────────────────────

class TestReader(unittest.TestCase):
    def test_open_allows(self):
        with broker(OPEN_THU):
            ok, why = broker_clock.orders_allowed(ts("2026-11-05T11:00:01-05:00"))
        self.assertTrue(ok)
        self.assertIn("market open", why)
        self.assertIn("16:00 ET", why, "the reading's own close time is carried")

    def test_closed_refuses(self):
        with broker(CLOSED_SAT):
            st = broker_clock.clock_state(ts("2026-11-07T11:00:01-05:00"))
            ok, why = broker_clock.orders_allowed(ts("2026-11-07T11:00:02-05:00"))
        self.assertEqual((st["state"], st["source"]), ("closed", "live"))
        self.assertFalse(ok)
        self.assertIn("closed until Mon 2026-11-09 09:30 ET", why)

    def test_half_day_closes_at_13_00_not_a_ttl_later(self):
        with broker(HALF_DAY_OPEN, HALF_DAY_CLOSED) as served:
            self.assertTrue(broker_clock.orders_allowed(ts("2026-11-27T12:59:45-05:00"))[0])
            self.assertTrue(broker_clock.orders_allowed(ts("2026-11-27T12:59:55-05:00"))[0])
            self.assertEqual(served.reads, 1, "cached within the TTL")
            ok, why = broker_clock.orders_allowed(ts("2026-11-27T13:00:05-05:00"))
            self.assertEqual(served.reads, 2, "the 12:59:45 reading expires at its own next_close (13:00), "
                                              "not at the end of its 30 s TTL (13:00:15)")
        self.assertFalse(ok)
        self.assertIn("market closed", why)

    def test_half_day_close_holds_even_when_the_reread_fails(self):
        with broker(HALF_DAY_OPEN, None):
            self.assertTrue(broker_clock.orders_allowed(ts("2026-11-27T12:59:00-05:00"))[0])
            st = broker_clock.clock_state(ts("2026-11-27T13:00:10-05:00"))
        self.assertEqual((st["state"], st["source"]), ("closed", "extrapolated"),
                         "said open until 13:00 and reopens Monday: closed in between, on the broker's own schedule")
        self.assertIn("clock read failed", st["detail"])

    def test_unreadable_extrapolates_from_the_last_good_reading(self):
        # said closed until Monday 09:30: a failed read on Saturday stays closed
        with broker(CLOSED_SAT, None, None) as served:
            self.assertFalse(broker_clock.orders_allowed(ts("2026-11-07T11:00:00-05:00"))[0])
            st = broker_clock.clock_state(ts("2026-11-07T15:00:00-05:00"))
            self.assertEqual(served.reads, 2)
            self.assertEqual((st["state"], st["source"]), ("closed", "extrapolated"))
            # past its next_open but before its next_close: open on the broker's schedule
            st = broker_clock.clock_state(ts("2026-11-09T10:00:00-05:00"))
            self.assertEqual(st["state"], "open")
        # said open until 16:00: a failed read at 15:00 stays open
        with broker(OPEN_THU, None):
            broker_clock.clock_state(ts("2026-11-05T11:00:00-05:00"))
            st = broker_clock.clock_state(ts("2026-11-05T15:00:00-05:00"))
            self.assertEqual((st["state"], st["source"]), ("open", "extrapolated"))
            # out of date (past both of its times): unknown, so allowed
            st = broker_clock.clock_state(ts("2026-11-06T17:00:00-05:00"))
            self.assertEqual(st["state"], "unknown")
            self.assertTrue(broker_clock.orders_allowed(ts("2026-11-06T17:00:30-05:00"))[0])

    def test_no_usable_reading_allows_with_clock_unknown(self):
        for body in (None, [{"symbol": "AAPL"}], {"is_open": "yes"}, {"message": "forbidden"}, "garbage"):
            with broker(body):
                ok, why = broker_clock.orders_allowed(ts("2026-11-07T11:00:00-05:00"))
                broker_clock.check_order_allowed("AAPL", "test", now=ts("2026-11-07T11:00:01-05:00"))
            self.assertTrue(ok, body)
            self.assertIn("clock unknown", why, body)

    def test_a_live_reading_without_times_is_trusted_as_read(self):
        with broker({"is_open": False}):
            self.assertFalse(broker_clock.orders_allowed(ts("2026-11-07T11:00:00-05:00"))[0])
        with broker({"is_open": True, "next_close": "garbage"}):
            self.assertTrue(broker_clock.orders_allowed(ts("2026-11-05T11:00:00-05:00"))[0])

    def test_ttl_cache_one_read_per_window_failures_too(self):
        t0 = ts("2026-11-07T11:00:00-05:00")
        with broker(CLOSED_SAT) as served:
            for dt in (0, 5, 29):
                broker_clock.clock_state(t0 + dt)
            self.assertEqual(served.reads, 1)
            broker_clock.clock_state(t0 + 31)
            self.assertEqual(served.reads, 2, "re-read after BROKER_CLOCK_TTL_SECONDS")
        with broker(None) as served:
            for dt in (0, 10, 20):
                broker_clock.clock_state(t0 + dt)
            self.assertEqual(served.reads, 1, "a failed read is cached for the TTL too")
        with patch.dict("system_config.BASE_CONFIG", {"BROKER_CLOCK_TTL_SECONDS": 5}), broker(CLOSED_SAT) as served:
            broker_clock.clock_state(t0)
            broker_clock.clock_state(t0 + 6)
            self.assertEqual(served.reads, 2, "the TTL is read from system_config")

    def test_the_fetch_never_raises_and_never_needs_a_status_code(self):
        def get_raising(*_a, **_k):
            raise ConnectionError("down")

        bad_json = MagicMock()
        bad_json.json.side_effect = ValueError("not json")
        with patch.object(broker_clock.requests, "get", side_effect=get_raising):
            self.assertIsNone(_REAL_FETCH())
        with patch.object(broker_clock.requests, "get", return_value=bad_json):
            self.assertIsNone(_REAL_FETCH())
        good = MagicMock()
        good.json.return_value = dict(OPEN_THU)
        with patch.object(broker_clock.requests, "get", return_value=good) as get:
            self.assertEqual(_REAL_FETCH(), OPEN_THU)
        self.assertTrue(get.call_args.args[0].endswith("/v2/clock"))
        self.assertIsNone(broker_clock._reading_from_body([OPEN_THU], 0.0))

    def test_reset_forgets_a_closed_reading(self):
        with broker(CLOSED_SAT):
            self.assertFalse(broker_clock.orders_allowed(ts("2026-11-07T11:00:00-05:00"))[0])
            broker_clock.reset_cache()
            with patch.object(broker_clock, "_fetch_clock_body", return_value=None):
                self.assertTrue(broker_clock.orders_allowed(ts("2026-11-07T11:00:01-05:00"))[0],
                                "no reading survives a reset (none leaks into another test)")


class TestGate(unittest.TestCase):
    def test_blocked_raises_a_runtime_error_and_logs_once_per_symbol_per_day(self):
        t = ts("2026-11-07T11:00:00-05:00")
        with broker(CLOSED_SAT), self.assertLogs("broker_clock", level="WARNING") as logs:
            for i in range(3):
                with self.assertRaises(broker_clock.OrderBlocked) as ctx:
                    broker_clock.check_order_allowed("GLD", "defensive floor exit", now=t + i)
            self.assertIsInstance(ctx.exception, RuntimeError, "callers that re-raise RuntimeError keep blocking the sale")
            self.assertTrue(str(ctx.exception).startswith("ORDER-BLOCKED: market closed"))
            self.assertIn("defensive floor exit GLD not sent", str(ctx.exception))
            with self.assertRaises(broker_clock.OrderBlocked):
                broker_clock.check_order_allowed("QQQ", "floor sell", now=t + 5)
        blocked = [r for r in logs.output if "ORDER-BLOCKED-CLOSED" in r]
        self.assertEqual(len(blocked), 2, "one line for GLD, one for QQQ")
        with broker(CLOSED_SAT, CLOSED_SAT), self.assertLogs("broker_clock", level="WARNING") as logs2:
            for day in ("2026-11-07T11:00:00-05:00", "2026-11-08T11:00:00-05:00"):
                with self.assertRaises(broker_clock.OrderBlocked):
                    broker_clock.check_order_allowed("GLD", "x", now=ts(day))
        self.assertEqual(len([r for r in logs2.output if "ORDER-BLOCKED-CLOSED" in r]), 2, "again the next ET day")

    def test_open_and_unknown_pass(self):
        with broker(OPEN_THU):
            broker_clock.check_order_allowed("SPY", "floor buy", now=ts("2026-11-05T11:00:00-05:00"))
        with broker(None):
            broker_clock.check_order_allowed("SPY", "floor buy", now=ts("2026-11-07T11:00:00-05:00"))


# ── bot_engine: the hours check and the floors ─────────────────────────────

SAT_11 = "2026-11-07T11:00:00-05:00"
THU_11 = "2026-11-05T11:00:00-05:00"
_MACRO = {"qqq_price": 500.0, "qqq_ma50": 480.0, "qqq_ma200": 460.0, "vxx_ratio": 1.0,
          "spy_vs_ma50": 1.0, "spy_below_200_days": 0, "spy_above_200d": True}


def _snap(price):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"latestTrade": {"p": price}}
    return m


class TestRegularHours(unittest.TestCase):
    def test_the_time_of_day_alone_no_longer_passes_a_weekend(self):
        with broker(CLOSED_SAT, at=SAT_11):
            self.assertFalse(bot_engine._is_regular_hours(), "Saturday 11:00 ET: the broker says closed")
        with broker(OPEN_THU, at=THU_11):
            self.assertTrue(bot_engine._is_regular_hours())
        with broker(None, at=SAT_11):
            self.assertTrue(bot_engine._is_regular_hours(), "no reading: the time-of-day test alone, as before")
        with broker(OPEN_THU, at="2026-11-05T08:00:00-05:00") as served:
            self.assertFalse(bot_engine._is_regular_hours())
            self.assertEqual(served.reads, 0, "outside 09:30-16:00 the clock is not even read")


class _FloorBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_clockgate_")
        for p in (patch.object(bot_engine, "DATA_DIR", self.tmp),
                  patch.object(bot_engine, "data_feed", return_value="iex"),
                  patch("system_config.get_market_regime", return_value="BULL")):
            p.start()
            self.addCleanup(p.stop)

    def state(self, name):
        try:
            with open(os.path.join(self.tmp, name)) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def seed(self, name, **st):
        with open(os.path.join(self.tmp, name), "w") as f:
            json.dump(st, f)


class TestOldBasketFloor(_FloorBase):
    """FLOOR_MODE "basket" (the rollback): nothing is sent on a weekend, even
    at 11:00 ET, where the old time-of-day check said "regular hours"."""

    CFG = {"FLOOR_TICKER": "QQQ", "FLOOR_BASKET_ENABLED": True, "FLOOR_BASKET": {"SMH": 1.0},
           "FLOOR_BASKET_DRIFT_THRESHOLD": 0.03, "SPY_FLOOR_BULL": 0.5, "SPY_FLOOR_BEAR": 0.0,
           "TREND_EXIT_DEATH_CROSS_CAP": 0.20, "TREND_EXIT_EARLY_WARNING_CAP": 0.50}

    def run_floor(self, positions, regime="BULL"):
        with patch("system_config.BASE_CONFIG", dict(self.CFG)), \
             patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000"}), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=positions), \
             patch.object(bot_engine.requests, "get", return_value=_snap(100.0)), \
             patch.object(bot_engine.requests, "post") as post:
            result = bot_engine._manage_spy_floor(dict(_MACRO), {"regime": regime})
        return result, post

    def test_weekend_rebalance_is_deferred_not_sent(self):
        with broker(CLOSED_SAT, at=SAT_11):
            result, post = self.run_floor([])
        post.assert_not_called()
        self.assertEqual([a["action"] for a in result["basket"]["actions"]], ["buy_deferred"])
        with broker(OPEN_THU, at=THU_11):
            result, post = self.run_floor([])
        self.assertEqual(post.call_count, 1, "the same rebalance is sent while the market is open")
        self.assertEqual(result["basket"]["actions"][0]["action"], "buy")

    def test_weekend_zero_target_exit_is_deferred_and_retried(self):
        self.seed("voltrade_floor_state.json", last_regime="NEUTRAL")
        held = [{"symbol": "QQQ", "qty": "10", "qty_available": "10", "market_value": "5000"},
                {"symbol": "SMH", "qty": "20", "qty_available": "20", "market_value": "6000"}]
        with broker(CLOSED_SAT, at=SAT_11):
            result, post = self.run_floor(held, regime="BEAR")
        post.assert_not_called()
        self.assertIn("floor_exit_deferred", [a.get("type") for a in result["actions"]])
        self.assertEqual(self.state("voltrade_floor_state.json").get("last_regime"), "NEUTRAL",
                         "deferred, not consumed: the exit is retried on the next scan")

    def test_the_gate_backstops_a_wrong_hours_answer(self):
        """Even if the hours check says open, the POST itself is refused: the
        rebalance is recorded as failed, never as a buy."""
        with broker(CLOSED_SAT, at=SAT_11), patch.object(bot_engine, "_is_regular_hours", return_value=True):
            result, post = self.run_floor([])
        post.assert_not_called()
        action = result["basket"]["actions"][0]
        self.assertEqual(action["action"], "buy_failed")
        self.assertTrue(action["error"].startswith("ORDER-BLOCKED"))


class TestDefensiveFloor(_FloorBase):
    """The GLD floor (basket mode) used to send its orders at any hour."""

    CFG = {"DEFENSIVE_FLOOR_ENABLED": True, "DEFENSIVE_FLOOR_TICKER": "GLD", "FLOOR_TICKER": "QQQ",
           "DEFENSIVE_FLOOR_BULL": 0.0, "DEFENSIVE_FLOOR_NEUTRAL": 0.0, "DEFENSIVE_FLOOR_CAUTION": 0.10,
           "DEFENSIVE_FLOOR_BEAR": 0.30, "DEFENSIVE_FLOOR_PANIC": 0.40, "DEFENSIVE_FLOOR_DEATHCROSS_MIN": 0.25}
    STATE = "voltrade_defensive_floor_state.json"

    def get(self, url, *a, **kw):
        m = MagicMock()
        m.status_code = 200
        if url.endswith("/v2/account"):
            m.json.return_value = {"equity": "100000"}
        elif url.endswith("/v2/stocks/snapshots"):
            m.json.return_value = {"GLD": {"latestTrade": {"p": 380.0}}}
        else:
            raise AssertionError(f"unexpected GET {url}")
        return m

    def run_floor(self, regime, positions):
        with patch("system_config.BASE_CONFIG", dict(self.CFG)), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=positions), \
             patch.object(bot_engine.requests, "get", side_effect=self.get), \
             patch.object(bot_engine.requests, "post") as post:
            result = bot_engine._manage_defensive_floor(dict(_MACRO), {"regime": regime})
        return result, post

    def test_exit_on_regime_change_waits_for_the_open(self):
        self.seed(self.STATE, last_regime="BEAR")
        gld = [{"symbol": "GLD", "qty": "50", "market_value": "19000"}]
        with broker(CLOSED_SAT, at=SAT_11):
            result, post = self.run_floor("BULL", gld)
        post.assert_not_called()
        self.assertEqual(result["actions"][0]["type"], "defensive_exit_deferred")
        self.assertEqual(self.state(self.STATE)["last_regime"], "BEAR", "not consumed: retried next scan")
        with broker(OPEN_THU, at=THU_11):
            result, post = self.run_floor("BULL", gld)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs["json"]["side"], "sell")
        self.assertEqual(self.state(self.STATE)["last_regime"], "BULL")

    def test_rebalance_buy_waits_for_the_open(self):
        self.seed(self.STATE, last_regime="NEUTRAL")
        with broker(CLOSED_SAT, at=SAT_11):
            result, post = self.run_floor("BEAR", [])
        post.assert_not_called()
        self.assertEqual(result["actions"][0]["type"], "defensive_rebalance_deferred")
        self.assertEqual(self.state(self.STATE)["last_regime"], "NEUTRAL")
        with broker(CLOSED_MON_PRE, at="2026-11-09T09:29:00-05:00"):
            _, post = self.run_floor("BEAR", [])
        post.assert_not_called()
        with broker(OPEN_THU, at=THU_11):
            _, post = self.run_floor("BEAR", [])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs["json"]["side"], "buy")


class TestVolManagedFloor(unittest.TestCase):
    def test_weekend_pass_defers_before_any_order(self):
        orders = []
        closes = [500.0 * (1.004 if i % 2 else 0.996) for i in range(21)]
        with broker(CLOSED_SAT, at=SAT_11), \
             patch.object(bot_engine, "_daily_closes_completed", return_value=closes), \
             patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000"}), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=[]), \
             patch.object(bot_engine, "_floor_vm_order", side_effect=lambda *a: orders.append(a)), \
             patch.object(bot_engine, "_floor_vm_save_state"), \
             patch.object(bot_engine, "_et_now_hour", return_value=11.0):
            result = bot_engine._manage_vol_managed_floor({})
        self.assertEqual(orders, [])
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(result["actions"][-1], {"type": "floor_vm_deferred", "reason": "market closed (broker clock)"},
                         "11:00 ET is inside the trade window: the reason is the broker's clock")

    def test_its_order_is_refused_and_reported_failed_when_closed(self):
        with broker(CLOSED_SAT, at=SAT_11), patch.object(bot_engine.requests, "post") as post:
            out = bot_engine._floor_vm_order("SPY", 10, "buy", 500.1)
        post.assert_not_called()
        self.assertFalse(out["ok"])
        self.assertIsNone(out["order_id"])
        self.assertTrue(out["error"].startswith("ORDER-BLOCKED"))


class TestCoveredCallBuyBack(unittest.TestCase):
    def test_blocked_buy_back_blocks_the_stock_sale(self):
        pos = [{"symbol": "SMH261120C00300000", "asset_class": "us_option", "qty": "-1"}]
        with broker(CLOSED_SAT, at=SAT_11), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=pos), \
             patch.object(bot_engine.requests, "post") as post:
            with self.assertRaises(RuntimeError) as ctx:
                bot_engine._unwind_covered_calls("SMH")
        post.assert_not_called()
        self.assertIsInstance(ctx.exception, broker_clock.OrderBlocked)


# ── options ─────────────────────────────────────────────────────────────────

class TestOptionsManager(unittest.TestCase):
    def test_single_leg_close_is_refused_and_reported_as_an_error(self):
        with broker(CLOSED_SAT, at=SAT_11), patch.object(om.requests, "post") as post:
            out = om._submit_close_order("XLF261127P00055500", 1, "buy", 5.00)
        post.assert_not_called()
        self.assertEqual(out["status"], "error")
        self.assertTrue(out["detail"].startswith("ORDER-BLOCKED"))

    def test_group_close_is_deferred_without_the_leg_fallback(self):
        legs = [{"symbol": "GOOGL261127P00300000", "qty": "1", "side": "long"},
                {"symbol": "GOOGL261127P00310000", "qty": "-1", "side": "short"}]
        with broker(CLOSED_SAT, at=SAT_11), patch.object(om.requests, "post") as post, \
                patch.object(om, "_get_option_snapshot") as snap:
            out = om._close_strategy_mleg(legs, "GOOGL")
        post.assert_not_called()
        snap.assert_not_called()
        self.assertEqual(out["status"], "deferred")
        self.assertTrue(out["detail"].startswith("ORDER-BLOCKED"))

    def test_market_is_open_follows_the_shared_gate(self):
        own_open = MagicMock(status_code=200, json=lambda: {"is_open": True})
        saved = dict(om._CLOCK_CACHE)
        try:
            om._CLOCK_CACHE.update({"at": 0.0, "is_open": None})
            with broker(CLOSED_SAT, at=SAT_11), patch.object(om.requests, "get", return_value=own_open):
                self.assertFalse(om._market_is_open(), "the shared gate says closed: defer, never a refused close")
            om._CLOCK_CACHE.update({"at": 0.0, "is_open": None})
            with broker(None), patch.object(om.requests, "get", return_value=own_open):
                self.assertTrue(om._market_is_open(), "unknown shared clock: this module's own reading decides")
        finally:
            om._CLOCK_CACHE.clear()
            om._CLOCK_CACHE.update(saved)


class TestOptionsExecution(unittest.TestCase):
    def test_submit_options_order_sends_nothing_while_closed(self):
        contract = {"strategy": "sell_cash_secured_put", "occ_symbol": "NU260904P00014000",
                    "side": "sell", "qty": 1, "limit_price": 0.18}
        with broker(CLOSED_SAT, at=SAT_11), patch.object(options_execution, "requests") as req:
            out = options_execution.submit_options_order(contract)
        req.post.assert_not_called()
        self.assertEqual(out["status"], "error")
        self.assertTrue(out["detail"].startswith("ORDER-BLOCKED"))
        self.assertNotIn("order_id", out)

    def test_mleg_order_sends_nothing_while_closed(self):
        legs = [{"symbol": "SPY260424C00500000", "side": "buy", "ratio_qty": 1},
                {"symbol": "SPY260424P00500000", "side": "buy", "ratio_qty": 1}]
        with broker(CLOSED_SAT, at=SAT_11), patch.object(options_execution, "requests") as req:
            out = options_execution._submit_mleg_order(legs, 1, 5.0, "Test straddle")
        req.post.assert_not_called()
        self.assertEqual(out["status"], "error")
        self.assertIn("ORDER-BLOCKED", out["detail"])

    def test_open_market_unchanged(self):
        ok = MagicMock(status_code=200, headers={"content-type": "application/json"}, text="")
        ok.json.return_value = {"id": "o-1"}
        contract = {"strategy": "buy_put", "occ_symbol": "SPY260918P00500000", "side": "buy", "qty": 2, "limit_price": 3.45}
        with broker(OPEN_THU, at=THU_11), patch.object(options_execution, "requests") as req:
            req.post.return_value = ok
            out = options_execution.submit_options_order(contract)
        self.assertEqual(req.post.call_count, 1)
        self.assertEqual(out["status"], "submitted")


class TestIntradayShorts(unittest.TestCase):
    def test_cover_waits_and_the_trade_stays_open(self):
        tmp = tempfile.mkdtemp(prefix="vt_clockgate_shorts_")
        log = {"trades": [{"symbol": "XYZ", "shares": 10, "entry_price": 20.0, "status": "open"}]}
        snap = MagicMock()
        snap.json.return_value = {"XYZ": {"latestTrade": {"p": 19.0}}}
        with broker(CLOSED_SAT, at=SAT_11), \
             patch.object(intraday_shorts, "SHORTS_LOG_PATH", os.path.join(tmp, "s.json")), \
             patch.object(intraday_shorts, "_load_log", return_value=log), \
             patch.object(intraday_shorts, "_save_log"), \
             patch.object(intraday_shorts.requests, "get", return_value=snap), \
             patch.object(intraday_shorts.requests, "post") as post:
            out = intraday_shorts.close_open_shorts()
        post.assert_not_called()
        self.assertEqual(out["count"], 0)
        self.assertEqual(log["trades"][0]["status"], "open", "not recorded as covered")


# ── on the source: every order POST sits behind the gate ───────────────────

ENGINE_FILES = ("bot_engine.py", "intraday_shorts.py", "options_manager.py", "options_execution.py")


def _order_posts(tree):
    """(call, enclosing statement list, index) for every `<x>.post(<url with v2/orders>...)`."""
    found = []
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody", "handlers"):
            stmts = getattr(parent, field, None)
            if not isinstance(stmts, list):
                continue
            for i, stmt in enumerate(stmts):
                if not isinstance(stmt, ast.stmt) or isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                                                                        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                for node in ast.walk(stmt):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "post"
                            and node.args and "v2/orders" in ast.unparse(node.args[0])):
                        found.append((node, stmts, i))
    return found


def _is_gate_call(stmt):
    return any(isinstance(n, ast.Call) and ast.unparse(n.func) == "broker_clock.check_order_allowed"
               for n in ast.walk(stmt))


class TestEveryOrderPostIsGated(unittest.TestCase):
    def test_each_post_is_preceded_by_the_gate_in_its_own_block(self):
        total = 0
        for name in ENGINE_FILES:
            tree = ast.parse(open(os.path.join(_ROOT, name)).read())
            posts = _order_posts(tree)
            self.assertTrue(posts, f"{name}: no order POST found — the scan itself is broken")
            for call, stmts, i in posts:
                total += 1
                # the gate call is an earlier statement of the POST's own block
                # (_floor_vm_order: inside the try/except just ahead of it)
                self.assertTrue(any(_is_gate_call(s) for s in stmts[:i]),
                                f"{name}:{call.lineno}: an order POST without broker_clock.check_order_allowed before it")
        self.assertEqual(total, 21, "bot_engine 15 + intraday_shorts 2 + options_manager 2 + options_execution 2: "
                                    "a new order POST site must be gated and counted here")

    def test_no_other_runtime_module_posts_orders(self):
        tracked = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True, cwd=_ROOT).stdout.split()
        others = [f for f in tracked if not os.path.basename(f).startswith("test_")
                  and f not in ENGINE_FILES and os.path.basename(f) != "conftest.py"]
        self.assertGreater(len(others), 50, "git ls-files found the runtime modules")
        unreadable = []
        for name in others:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(open(os.path.join(_ROOT, name)).read())
            except (SyntaxError, UnicodeDecodeError, OSError) as e:
                unreadable.append(f"{name}: {type(e).__name__}")
                tree = None
            if tree is not None:
                self.assertEqual(_order_posts(tree), [], f"{name} posts orders outside the gated engine files")
        self.assertLess(len(unreadable), 5, f"too many modules could not be scanned: {unreadable}")


if __name__ == "__main__":
    unittest.main()
