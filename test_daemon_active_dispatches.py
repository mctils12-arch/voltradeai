#!/usr/bin/env python3
"""
Regression tests for voltrade_daemon.py's active-dispatch-thread visibility
(DAEMON-TIMEOUT-VISIBILITY, 2026-07-10).

WHY THIS EXISTS: production audit logs showed TIER2-ERROR "daemon
run_full_scan failed: Daemon timeout" recurring 7x across ~90 minutes on
2026-07-10, with zero diagnostic detail — the classification logic in
server/bot.ts's tier2Intelligence catch block is built for subprocess
failures (stderr/stdout/code/signal) and produces a content-free
"code=? signal=none" line plus a Node-process memory snapshot when the
actual failure is a Unix-socket RPC timeout to voltrade_daemon.py.

Root cause investigation found that RPCHandler.handle()'s t.join(
REQUEST_TIMEOUT_SEC) returns (and releases _inflight_sem) whether or not
the spawned worker thread actually finished — Python threads cannot be
forcibly killed, so a dispatch() call that itself hangs past the timeout
keeps running in the background, invisibly consuming CPU/memory, after the
client has already been told the call failed. _inflight_sem being released
at handler-wait-timeout (not at actual thread completion) means it no
longer bounds real concurrent load once this happens — a plausible driver
of the cascading pattern (each further call also contends with growing
zombie thread pileup, each of which can also time out).

This file pins the fix: a dedicated counter that tracks dispatch threads
actually executing right now (including abandoned ones), surfaced on
_health() as active_dispatches, so a client hitting a daemon timeout can
immediately check whether the daemon is piling up abandoned work.

EXTENDED 2026-07-20 (KNOWN BROKEN #18 continuation, scheduled-routine
session): live TIER2-ERROR catches this session found active_dispatches==2
on every occurrence, but the bare count can't distinguish "healthy 2x
concurrency" from "an abandoned run_full_scan zombie thread from the PRIOR
timed-out cycle still running, competing for the shared alpaca_throttle
bucket with a fresh run_full_scan that just started" — the
self-perpetuating-cascade hypothesis three timeouts landing at exactly the
300s RPC boundary (19:29:20, 19:36:20, 19:57:13Z) suggest. _inc_active_dispatch
now takes the method name and returns a per-dispatch id; _health() gained
active_dispatch_detail: [{method, elapsed_sec}, ...] so the next occurrence's
audit line can answer "what, and for how long" directly.
"""
import os
import socket
import sys
import threading
import time
import types
import unittest

import voltrade_daemon
from voltrade_daemon import RPCDispatcher, RPCHandler, ThreadingUnixServer


class TestActiveDispatchCounter(unittest.TestCase):
    """Pure unit tests on the counter functions — no socket involved."""

    def setUp(self):
        # Counter/detail dict are module-global; reset around each test so
        # tests don't bleed into each other regardless of run order.
        voltrade_daemon._active_dispatch_count = 0
        voltrade_daemon._active_dispatch_detail.clear()

    def tearDown(self):
        voltrade_daemon._active_dispatch_count = 0
        voltrade_daemon._active_dispatch_detail.clear()

    def test_starts_at_zero(self):
        self.assertEqual(voltrade_daemon._active_dispatch_count, 0)
        self.assertEqual(voltrade_daemon._active_dispatch_snapshot(), [])

    def test_increment_and_decrement(self):
        id_a = voltrade_daemon._inc_active_dispatch("run_full_scan")
        self.assertEqual(voltrade_daemon._active_dispatch_count, 1)
        id_b = voltrade_daemon._inc_active_dispatch("run_full_scan")
        self.assertEqual(voltrade_daemon._active_dispatch_count, 2)
        voltrade_daemon._dec_active_dispatch(id_a)
        self.assertEqual(voltrade_daemon._active_dispatch_count, 1)
        voltrade_daemon._dec_active_dispatch(id_b)
        self.assertEqual(voltrade_daemon._active_dispatch_count, 0)

    def test_concurrent_increments_are_race_free(self):
        # If the +=1/-=1 weren't guarded by _active_dispatch_lock, this
        # would flake under real thread interleaving on some runs.
        threads = [threading.Thread(target=voltrade_daemon._inc_active_dispatch, args=("run_full_scan",))
                   for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(voltrade_daemon._active_dispatch_count, 50)
        self.assertEqual(len(voltrade_daemon._active_dispatch_snapshot()), 50)

    def test_health_reports_the_counter(self):
        dispatcher = RPCDispatcher()
        voltrade_daemon._active_dispatch_count = 3
        result = dispatcher._health({})
        self.assertEqual(result["active_dispatches"], 3)
        self.assertIn("rss_mb", result)
        self.assertIn("uptime_seconds", result)

    def test_detail_tracks_method_and_elapsed_time(self):
        """The exact shape a live zombie-pileup catch needs: entries
        distinguishable by method name and by how long each has run —
        this is what active_dispatches's bare count could never answer."""
        dispatch_id = voltrade_daemon._inc_active_dispatch("run_full_scan")
        snapshot = voltrade_daemon._active_dispatch_snapshot()
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0]["method"], "run_full_scan")
        self.assertGreaterEqual(snapshot[0]["elapsed_sec"], 0)
        voltrade_daemon._dec_active_dispatch(dispatch_id)
        self.assertEqual(voltrade_daemon._active_dispatch_snapshot(), [])

    def test_detail_distinguishes_concurrent_dispatches_by_method(self):
        """A zombie run_full_scan overlapping a fresh one of the SAME method
        must still show as two distinct entries — the count alone (2) can't
        tell a healthy pair apart from one abandoned + one live."""
        id_a = voltrade_daemon._inc_active_dispatch("run_full_scan")
        id_b = voltrade_daemon._inc_active_dispatch("select_contract")
        snapshot = voltrade_daemon._active_dispatch_snapshot()
        self.assertEqual({s["method"] for s in snapshot}, {"run_full_scan", "select_contract"})
        voltrade_daemon._dec_active_dispatch(id_a)
        snapshot = voltrade_daemon._active_dispatch_snapshot()
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0]["method"], "select_contract")
        voltrade_daemon._dec_active_dispatch(id_b)

    def test_health_surfaces_active_dispatch_detail(self):
        dispatcher = RPCDispatcher()
        dispatch_id = voltrade_daemon._inc_active_dispatch("select_contract")
        try:
            result = dispatcher._health({})
            self.assertIn("active_dispatch_detail", result)
            self.assertEqual(len(result["active_dispatch_detail"]), 1)
            self.assertEqual(result["active_dispatch_detail"][0]["method"], "select_contract")
        finally:
            voltrade_daemon._dec_active_dispatch(dispatch_id)

    def test_dec_unknown_id_is_a_safe_detail_noop(self):
        """An id with no matching detail entry (e.g. a double-decrement)
        must not raise and must leave the detail registry consistent."""
        voltrade_daemon._dec_active_dispatch(999999)
        self.assertEqual(voltrade_daemon._active_dispatch_snapshot(), [])


class TestLayer2PrefetchSnapshot(unittest.TestCase):
    """
    DAEMON-TIMEOUT-VISIBILITY 2026-07-21 (KNOWN BROKEN #18 continuation):
    _health() now also surfaces csp_universe.py's Layer 2 CSP prefetch
    diagnostics (cache_hit/completed/total/elapsed_sec/budget_exceeded),
    read live from sys.modules so a TIER2-ERROR daemon-timeout occurrence's
    health probe reflects the CURRENTLY-RUNNING (possibly still-hung) scan's
    own state, not just whatever the last scan that actually returned left
    in bot.ts's /api/diag/timings.
    """

    def setUp(self):
        self._had_csp_universe = "csp_universe" in sys.modules
        self._orig_csp_universe = sys.modules.get("csp_universe")

    def tearDown(self):
        if self._had_csp_universe:
            sys.modules["csp_universe"] = self._orig_csp_universe
        else:
            sys.modules.pop("csp_universe", None)

    def test_empty_when_module_never_loaded(self):
        sys.modules.pop("csp_universe", None)
        self.assertEqual(voltrade_daemon._layer2_prefetch_snapshot(), {})

    def test_empty_when_layer2_score_never_ran(self):
        fake = types.SimpleNamespace(get_last_layer2_prefetch_stats=lambda: {})
        sys.modules["csp_universe"] = fake
        self.assertEqual(voltrade_daemon._layer2_prefetch_snapshot(), {})

    def test_surfaces_stats_with_computed_age(self):
        checked_at = time.time() - 12.3
        fake = types.SimpleNamespace(get_last_layer2_prefetch_stats=lambda: {
            "cache_hit": False, "completed": 90, "total": 150,
            "elapsed_sec": 45.0, "budget_exceeded": True,
            "checked_at": checked_at,
        })
        sys.modules["csp_universe"] = fake
        snap = voltrade_daemon._layer2_prefetch_snapshot()
        self.assertEqual(snap["completed"], 90)
        self.assertEqual(snap["total"], 150)
        self.assertTrue(snap["budget_exceeded"])
        self.assertAlmostEqual(snap["age_sec"], 12.3, delta=0.5)

    def test_exception_from_csp_universe_propagates_not_swallowed(self):
        """No new silent broad-except: a genuine failure here must surface
        as a visible dispatch() error, not disappear into an empty dict —
        test_silent_except_ratchet.py pins the count of tolerated handlers
        and would catch a bare `except Exception` added to swallow this."""
        def _raise():
            raise RuntimeError("boom")
        fake = types.SimpleNamespace(get_last_layer2_prefetch_stats=_raise)
        sys.modules["csp_universe"] = fake
        with self.assertRaises(RuntimeError):
            voltrade_daemon._layer2_prefetch_snapshot()
        # dispatch()'s own outer try/except turns it into a visible error
        # response rather than a crash or a silently-empty health() result.
        dispatcher = RPCDispatcher()
        result = dispatcher.dispatch("health", {})
        self.assertEqual(result["status"], "error")
        self.assertIn("boom", result["error_message"])

    def test_health_includes_layer2_prefetch_key(self):
        sys.modules.pop("csp_universe", None)
        dispatcher = RPCDispatcher()
        result = dispatcher._health({})
        self.assertIn("layer2_prefetch", result)
        self.assertEqual(result["layer2_prefetch"], {})

    def test_health_surfaces_populated_layer2_prefetch(self):
        fake = types.SimpleNamespace(get_last_layer2_prefetch_stats=lambda: {
            "cache_hit": True, "completed": 0, "total": 0,
            "elapsed_sec": 0.0, "budget_exceeded": False,
            "checked_at": time.time(),
        }, get_last_layer1_stats=lambda: {})
        sys.modules["csp_universe"] = fake
        dispatcher = RPCDispatcher()
        result = dispatcher._health({})
        self.assertTrue(result["layer2_prefetch"]["cache_hit"])
        self.assertIn("age_sec", result["layer2_prefetch"])


class TestLayer1StatsSnapshot(unittest.TestCase):
    """DAEMON-TIMEOUT-VISIBILITY 2026-07-27 (KNOWN BROKEN #18 continuation):
    same live-mid-hang read as _layer2_prefetch_snapshot, mirrored for Layer
    1's own independent cache (csp_universe.get_last_layer1_stats(),
    v1.0.521) — every tier_engine_start-stuck occurrence checked this session
    read layer2_prefetch cache_hit=true, which never explained the hang;
    Layer 1 was invisible until now and could be the thing still running."""

    def setUp(self):
        self._had_csp_universe = "csp_universe" in sys.modules
        self._orig_csp_universe = sys.modules.get("csp_universe")

    def tearDown(self):
        if self._had_csp_universe:
            sys.modules["csp_universe"] = self._orig_csp_universe
        else:
            sys.modules.pop("csp_universe", None)

    def test_empty_when_module_never_loaded(self):
        sys.modules.pop("csp_universe", None)
        self.assertEqual(voltrade_daemon._layer1_stats_snapshot(), {})

    def test_empty_when_layer1_hard_gates_never_ran(self):
        fake = types.SimpleNamespace(get_last_layer1_stats=lambda: {})
        sys.modules["csp_universe"] = fake
        self.assertEqual(voltrade_daemon._layer1_stats_snapshot(), {})

    def test_surfaces_stats_with_computed_age(self):
        checked_at = time.time() - 8.7
        fake = types.SimpleNamespace(get_last_layer1_stats=lambda: {
            "cache_hit": False, "universe_size": 380, "elapsed_sec": 18.4,
            "checked_at": checked_at,
        })
        sys.modules["csp_universe"] = fake
        snap = voltrade_daemon._layer1_stats_snapshot()
        self.assertEqual(snap["universe_size"], 380)
        self.assertFalse(snap["cache_hit"])
        self.assertAlmostEqual(snap["age_sec"], 8.7, delta=0.5)

    def test_exception_from_csp_universe_propagates_not_swallowed(self):
        """No new silent broad-except — mirrors the layer2 snapshot's own
        pin: test_silent_except_ratchet.py would catch a bare
        `except Exception` added here to swallow this."""
        def _raise():
            raise RuntimeError("boom")
        fake = types.SimpleNamespace(get_last_layer2_prefetch_stats=lambda: {},
                                      get_last_layer1_stats=_raise)
        sys.modules["csp_universe"] = fake
        with self.assertRaises(RuntimeError):
            voltrade_daemon._layer1_stats_snapshot()
        dispatcher = RPCDispatcher()
        result = dispatcher.dispatch("health", {})
        self.assertEqual(result["status"], "error")
        self.assertIn("boom", result["error_message"])

    def test_health_includes_layer1_stats_key(self):
        sys.modules.pop("csp_universe", None)
        dispatcher = RPCDispatcher()
        result = dispatcher._health({})
        self.assertIn("layer1_stats", result)
        self.assertEqual(result["layer1_stats"], {})

    def test_health_surfaces_populated_layer1_stats(self):
        fake = types.SimpleNamespace(get_last_layer2_prefetch_stats=lambda: {},
                                      get_last_layer1_stats=lambda: {
            "cache_hit": True, "universe_size": 412, "elapsed_sec": 0.0,
            "checked_at": time.time(),
        })
        sys.modules["csp_universe"] = fake
        dispatcher = RPCDispatcher()
        result = dispatcher._health({})
        self.assertTrue(result["layer1_stats"]["cache_hit"])
        self.assertIn("age_sec", result["layer1_stats"])


class TestActiveDispatchLiveSocket(unittest.TestCase):
    """
    Integration test over a real Unix socket + ThreadingUnixServer, proving
    active_dispatches goes 0 -> 1+ while a dispatch is genuinely in flight
    and back to 0 once it completes — the exact signal a future TIER2-ERROR
    daemon-timeout investigation needs. Monkeypatches the module-level
    dispatcher's dispatch() to sleep, rather than touching any real route,
    so this cannot affect production routing behavior.
    """

    def setUp(self):
        self.socket_path = f"/tmp/test_voltrade_daemon_{os.getpid()}_{id(self)}.sock"
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

    def _rpc(self, method, args=None, timeout=5):
        import json
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(self.socket_path)
        s.sendall((json.dumps({"method": method, "args": args or {}}) + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        s.close()
        return json.loads(buf.split(b"\n")[0].decode())

    def test_active_dispatches_reflects_in_flight_call(self):
        release_event = threading.Event()
        entered_event = threading.Event()

        def slow_dispatch(method, args):
            entered_event.set()
            release_event.wait(timeout=5)
            return {"pong": True}

        voltrade_daemon._dispatcher.dispatch = slow_dispatch

        client_thread = threading.Thread(target=self._rpc, args=("ping",))
        client_thread.start()

        self.assertTrue(entered_event.wait(timeout=5), "slow dispatch never started")
        # Give _inc_active_dispatch a moment to run relative to entered_event.
        time.sleep(0.05)
        self.assertGreaterEqual(
            voltrade_daemon._active_dispatch_count, 1,
            "active_dispatches did not reflect the in-flight dispatch call",
        )

        release_event.set()
        client_thread.join(timeout=5)
        # Give _dec_active_dispatch a moment to run after dispatch() returns.
        deadline = time.time() + 2
        while voltrade_daemon._active_dispatch_count != 0 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(
            voltrade_daemon._active_dispatch_count, 0,
            "active_dispatches did not return to 0 after the dispatch completed",
        )

    def test_health_rpc_over_real_socket_includes_active_dispatches(self):
        response = self._rpc("health")
        self.assertEqual(response["status"], "ok")
        self.assertIn("active_dispatches", response["result"])
        # The health call is itself an in-flight dispatch at the moment
        # _health() runs (the counter is incremented before dispatch()
        # executes) — so a lone health probe on an otherwise-idle daemon
        # honestly reports 1, not 0. Anything else already tests the
        # decrement-after-completion path above.
        self.assertEqual(response["result"]["active_dispatches"], 1)
        self.assertEqual(len(response["result"]["active_dispatch_detail"]), 1)
        self.assertEqual(response["result"]["active_dispatch_detail"][0]["method"], "health")


if __name__ == "__main__":
    unittest.main()
