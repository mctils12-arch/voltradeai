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
"""
import os
import socket
import threading
import time
import unittest

import voltrade_daemon
from voltrade_daemon import RPCDispatcher, RPCHandler, ThreadingUnixServer


class TestActiveDispatchCounter(unittest.TestCase):
    """Pure unit tests on the counter functions — no socket involved."""

    def setUp(self):
        # Counter is module-global; reset around each test so tests don't
        # bleed into each other regardless of run order.
        voltrade_daemon._active_dispatch_count = 0

    def tearDown(self):
        voltrade_daemon._active_dispatch_count = 0

    def test_starts_at_zero(self):
        self.assertEqual(voltrade_daemon._active_dispatch_count, 0)

    def test_increment_and_decrement(self):
        voltrade_daemon._inc_active_dispatch()
        self.assertEqual(voltrade_daemon._active_dispatch_count, 1)
        voltrade_daemon._inc_active_dispatch()
        self.assertEqual(voltrade_daemon._active_dispatch_count, 2)
        voltrade_daemon._dec_active_dispatch()
        self.assertEqual(voltrade_daemon._active_dispatch_count, 1)
        voltrade_daemon._dec_active_dispatch()
        self.assertEqual(voltrade_daemon._active_dispatch_count, 0)

    def test_concurrent_increments_are_race_free(self):
        # If the +=1/-=1 weren't guarded by _active_dispatch_lock, this
        # would flake under real thread interleaving on some runs.
        threads = [threading.Thread(target=voltrade_daemon._inc_active_dispatch)
                   for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(voltrade_daemon._active_dispatch_count, 50)

    def test_health_reports_the_counter(self):
        dispatcher = RPCDispatcher()
        voltrade_daemon._active_dispatch_count = 3
        result = dispatcher._health({})
        self.assertEqual(result["active_dispatches"], 3)
        self.assertIn("rss_mb", result)
        self.assertIn("uptime_seconds", result)


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


if __name__ == "__main__":
    unittest.main()
