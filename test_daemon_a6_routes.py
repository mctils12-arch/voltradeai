#!/usr/bin/env python3
"""
A6 DAEMON ROUTING (2026-11-03, memory audit fix A6) — the Python half.

server/bot.ts now sends Tier 1's 45 s stop-state sync and the per-candidate
research call through voltrade_daemon.py instead of a fresh interpreter each
time (server/daemonRouting.ts has the client-side rules and their tests). This
file pins the daemon side:

  * the two new routes resolve to the real entry points and dispatch() hands
    back EXACTLY what the replaced subprocess printed — manage_positions()'s
    dict, and analyze.analyze_ticker_rpc()'s normalised + NaN-cleaned dict;
  * an exception inside the routed method comes back as the {"status":
    "error", ..., "traceback"} envelope the client classifies as "never run
    this again", not as something that could trigger a second run;
  * RPCHandler waits no longer than the client itself will when the request
    carries `timeout_sec` (pythonRpc now sends its deadline), so a run of
    abandoned 30 s research calls cannot pin _inflight_sem slots for 300 s
    each and queue a Tier-1 call behind them. A request without timeout_sec
    behaves exactly as before.
"""
import json
import math
import os
import socket
import sys
import threading
import time
import types
import unittest

import voltrade_daemon
from voltrade_daemon import (
    MAX_INFLIGHT_REQUESTS, REQUEST_TIMEOUT_SEC, RPCDispatcher, RPCHandler,
    ThreadingUnixServer, _effective_join_sec,
)

REPO = os.path.dirname(os.path.abspath(__file__))

MANAGE_FIXTURE = {
    "actions": [
        {"ticker": "NVDA", "type": "TRAIL_STOP", "price": 181.25, "reason": "Phase 4 trail — 3.0× ATR"},
        {"ticker": "AMD", "type": "SCALE_OUT", "qty": 3, "frac": 0.1 + 0.2, "none": None},
    ],
    "positions": 7,
    "upgrade_candidates": [],
    "regime": "NEUTRAL_BULL",
}


class _StubModule:
    """Context manager: shadow `name` in sys.modules (and the daemon's lazy-import
    cache) with a stub module for the duration of one dispatch."""

    def __init__(self, name, **attrs):
        self.name = name
        self.stub = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(self.stub, k, v)

    def __enter__(self):
        self.saved_sys = sys.modules.get(self.name)
        self.saved_cache = voltrade_daemon._modules_loaded.pop(self.name, None)
        sys.modules[self.name] = self.stub
        return self.stub

    def __exit__(self, *exc):
        voltrade_daemon._modules_loaded.pop(self.name, None)
        if self.saved_sys is not None:
            sys.modules[self.name] = self.saved_sys
        else:
            sys.modules.pop(self.name, None)
        if self.saved_cache is not None:
            voltrade_daemon._modules_loaded[self.name] = self.saved_cache
        return False


class TestA6Routes(unittest.TestCase):
    def setUp(self):
        self.dispatcher = RPCDispatcher()

    def test_routes_target_the_real_entry_points(self):
        self.assertEqual(self.dispatcher._routes["manage_positions"], ("bot_engine", "manage_positions"))
        self.assertEqual(self.dispatcher._routes["analyze_ticker"], ("analyze", "analyze_ticker_rpc"))
        import analyze
        self.assertTrue(callable(analyze.analyze_ticker_rpc))

    def test_manage_positions_dispatch_returns_exactly_what_the_one_liner_prints(self):
        with _StubModule("bot_engine", manage_positions=lambda: MANAGE_FIXTURE):
            envelope = self.dispatcher.dispatch("manage_positions", {})
        self.assertEqual(envelope["status"], "ok")
        # The replaced subprocess printed json.dumps(manage_positions()) and bot.ts
        # JSON.parse'd it; the daemon path hands bot.ts json.loads(json.dumps(envelope))["result"].
        via_subprocess = json.loads(json.dumps(MANAGE_FIXTURE))
        via_daemon = json.loads(json.dumps(envelope))["result"]
        self.assertEqual(via_daemon, via_subprocess)

    def test_manage_positions_dispatch_takes_no_args_and_tolerates_the_empty_dict_bot_ts_sends(self):
        calls = []

        def manage_positions():
            calls.append(True)
            return {"actions": [], "positions": 0, "upgrade_candidates": []}

        with _StubModule("bot_engine", manage_positions=manage_positions):
            envelope = self.dispatcher.dispatch("manage_positions", {})
        self.assertEqual(envelope, {"status": "ok", "result": {"actions": [], "positions": 0, "upgrade_candidates": []}})
        self.assertEqual(calls, [True])

    def test_analyze_ticker_dispatch_normalises_the_symbol_and_cleans_nan_like_the_cli(self):
        import analyze
        seen = []
        fixture = {
            "ticker": "AAPL", "vrp": float("nan"), "spot": 1.5,
            "edge_factors": {"squeeze_score": float("inf"), "squeeze_desc": "x"},
            "earnings_intel": {"days_to_earnings": 3, "beat_pct": None, "options_edge": -float("inf")},
            "top_spreads": [{"score": float("nan")}],
        }

        def fake_analyze_ticker(symbol):
            seen.append(symbol)
            return fixture

        orig = analyze.analyze_ticker
        analyze.analyze_ticker = fake_analyze_ticker
        voltrade_daemon._modules_loaded.pop("analyze", None)
        try:
            envelope = self.dispatcher.dispatch("analyze_ticker", {"symbol": " aapl "})
            direct = analyze.analyze_ticker_rpc(" aapl ")  # what the CLI prints, same stub
        finally:
            analyze.analyze_ticker = orig
        self.assertEqual(seen, ["AAPL", "AAPL"], "the CLI's upper().strip() normalisation")
        self.assertEqual(envelope, {"status": "ok", "result": {
            "ticker": "AAPL", "vrp": None, "spot": 1.5,
            "edge_factors": {"squeeze_score": None, "squeeze_desc": "x"},
            "earnings_intel": {"days_to_earnings": 3, "beat_pct": None, "options_edge": None},
            "top_spreads": [{"score": None}],
        }})
        # ...and it is the very function the CLI prints, so the two cannot drift.
        self.assertEqual(envelope["result"], direct)
        # Valid JSON over the wire either way (NaN would not be).
        json.loads(json.dumps(envelope))

    def test_analyze_cli_prints_the_route_function(self):
        src = open(os.path.join(REPO, "analyze.py"), encoding="utf8").read()
        main_block = src[src.index('if __name__ == "__main__":'):]
        self.assertIn("print(json.dumps(analyze_ticker_rpc(cmd)))", main_block)
        self.assertNotIn("analyze_ticker(cmd)", main_block.replace("analyze_ticker_rpc(cmd)", ""),
                         "the non-SCAN CLI branch must not call analyze_ticker() directly any more")
        # analyze_ticker_rpc is analyze_ticker + _clean_nan, nothing else.
        self.assertIn("return _clean_nan(analyze_ticker(str(symbol).upper().strip()))", src)

    def test_an_exception_inside_the_routed_method_is_an_error_envelope_with_a_traceback(self):
        import analyze

        def boom(symbol):
            raise RuntimeError("yfinance exploded")

        orig = analyze.analyze_ticker
        analyze.analyze_ticker = boom
        voltrade_daemon._modules_loaded.pop("analyze", None)
        try:
            envelope = self.dispatcher.dispatch("analyze_ticker", {"symbol": "AAPL"})
        finally:
            analyze.analyze_ticker = orig
        self.assertEqual(envelope["status"], "error")
        self.assertIn("yfinance exploded", envelope["error_message"])
        self.assertIn("traceback", envelope, "server/daemonRouting.ts classifies a traceback-bearing envelope as 'error': never rerun")

        def manage_boom():
            raise KeyError("qty")

        with _StubModule("bot_engine", manage_positions=manage_boom):
            envelope = self.dispatcher.dispatch("manage_positions", {})
        self.assertEqual(envelope["status"], "error")
        self.assertIn("traceback", envelope)


class TestEffectiveJoinSec(unittest.TestCase):
    def test_default_when_the_client_sent_nothing_usable(self):
        for bad in (None, "15", "", True, False, 0, -5, float("nan"), [], {}):
            self.assertEqual(_effective_join_sec(bad), REQUEST_TIMEOUT_SEC, repr(bad))

    def test_the_clients_positive_deadline_capped_at_the_daemon_ceiling(self):
        self.assertEqual(_effective_join_sec(15), 15.0)
        self.assertEqual(_effective_join_sec(0.5), 0.5)
        self.assertEqual(_effective_join_sec(300), 300.0)
        self.assertEqual(_effective_join_sec(1e9), float(REQUEST_TIMEOUT_SEC))
        self.assertEqual(_effective_join_sec(REQUEST_TIMEOUT_SEC + 1), float(REQUEST_TIMEOUT_SEC))

    def test_never_raises(self):
        for weird in (object(), b"3", complex(1, 1), math.inf):
            _effective_join_sec(weird)


class TestClientDeadlineJoinLiveSocket(unittest.TestCase):
    """Over a real Unix socket + ThreadingUnixServer (same harness as
    test_daemon_active_dispatches.py): a request that carries timeout_sec gets
    its timeout answer — and gives its _inflight_sem slot back — at that
    deadline, while the dispatch thread itself keeps running (it cannot be
    killed; _active_dispatch_count still shows it)."""

    def setUp(self):
        self.socket_path = f"/tmp/test_a6_daemon_{os.getpid()}_{id(self)}.sock"
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        voltrade_daemon._active_dispatch_count = 0
        voltrade_daemon._active_dispatch_detail.clear()
        self._orig_dispatch = voltrade_daemon._dispatcher.dispatch
        self.server = ThreadingUnixServer(self.socket_path, RPCHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self):
        voltrade_daemon._dispatcher.dispatch = self._orig_dispatch
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=5)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        voltrade_daemon._active_dispatch_count = 0
        voltrade_daemon._active_dispatch_detail.clear()

    def _rpc(self, request, timeout=10):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(self.socket_path)
        s.sendall((json.dumps(request) + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        s.close()
        return json.loads(buf.split(b"\n")[0].decode())

    def test_timeout_sec_releases_the_handler_and_its_slot_at_the_client_deadline(self):
        release = threading.Event()
        entered = threading.Event()

        def slow_dispatch(method, args):
            entered.set()
            release.wait(timeout=10)
            return {"status": "ok", "result": {"late": True}}

        voltrade_daemon._dispatcher.dispatch = slow_dispatch
        t0 = time.time()
        resp = self._rpc({"method": "manage_positions", "args": {}, "timeout_sec": 0.5})
        elapsed = time.time() - t0
        self.assertTrue(entered.is_set(), "the dispatch did start")
        self.assertEqual(resp["status"], "error")
        self.assertEqual(resp["error_message"], "Request timed out after 0.5s")
        self.assertLess(elapsed, 5.0, f"answered at the client's deadline, not REQUEST_TIMEOUT_SEC ({elapsed:.2f}s)")
        # The slot is back: a Tier-1 call arriving now does not queue behind this one...
        deadline = time.time() + 2
        while voltrade_daemon._inflight_sem._value < MAX_INFLIGHT_REQUESTS and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(voltrade_daemon._inflight_sem._value, MAX_INFLIGHT_REQUESTS)
        # ...while the abandoned thread is still honestly counted.
        self.assertEqual(voltrade_daemon._active_dispatch_count, 1)
        release.set()
        deadline = time.time() + 5
        while voltrade_daemon._active_dispatch_count > 0 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(voltrade_daemon._active_dispatch_count, 0)

    def test_a_request_without_timeout_sec_still_completes_as_before(self):
        voltrade_daemon._dispatcher.dispatch = lambda method, args: {"status": "ok", "result": {"method": method, "args": args}}
        resp = self._rpc({"method": "ping", "args": {"x": 1}})
        self.assertEqual(resp, {"status": "ok", "result": {"method": "ping", "args": {"x": 1}}})

    def test_a_non_numeric_timeout_sec_is_ignored_not_fatal(self):
        voltrade_daemon._dispatcher.dispatch = lambda method, args: {"status": "ok", "result": 1}
        resp = self._rpc({"method": "ping", "args": {}, "timeout_sec": "soon"})
        self.assertEqual(resp, {"status": "ok", "result": 1})


if __name__ == "__main__":
    unittest.main()
