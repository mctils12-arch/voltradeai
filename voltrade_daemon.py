#!/usr/bin/env python3
"""
VolTradeAI — Python RPC Daemon
==============================

Eliminates the ~12s/scan overhead of spawning 27 fresh Python subprocesses
(each re-importing numpy/pandas/LightGBM). This daemon starts once, imports
everything once, and handles RPC calls over a Unix socket.

ARCHITECTURE:
  bot.ts                           voltrade_daemon.py
    │                                  │
    │ POST /rpc {method, args}         │
    ├─────────────────────────────────▶│
    │                                  ├── dispatch to method
    │                                  ├── return JSON result
    │◀─────────────────────────────────┤
    │                                  │

SAFETY:
  - This daemon is OPTIONAL. If the Unix socket isn't listening,
    bot.ts falls back to the original subprocess pattern (see
    DAEMON_INTEGRATION.md for the fallback wrapper).
  - The daemon auto-restarts on fatal errors (supervised by bot.ts or
    systemd).
  - Each request runs in its own thread — one slow call doesn't block
    others.
  - Memory monitoring: daemon self-kills if RSS exceeds 1 GB (bot.ts
    respawns it).

USAGE:
  # Start daemon:
  python3 voltrade_daemon.py &

  # Check status:
  curl --unix-socket /tmp/voltrade_daemon.sock http://localhost/health

  # Call a method:
  curl --unix-socket /tmp/voltrade_daemon.sock http://localhost/rpc \\
       -d '{"method":"ml_status","args":{}}'

SUPPORTED METHODS:
  - ml_status: get ML model status (was ml_status.py)
  - ml_toggle: enable/disable ML (was ml_toggle.py)
  - track_fill: record a fill for ML training (was ml_model_v2.track_fill)
  - check_halt: check if ticker is halted (was position_sizing.check_halt_status)
  - select_contract: pick an options contract (was options_execution.select_contract)
  - submit_options_order: submit options order (was options_execution.submit_options_order)
  - evaluate_and_execute: full options evaluation (was options_execution.evaluate_and_execute)

Each method accepts a JSON dict of args and returns a JSON dict.
Responses always include {"status": "ok" | "error", "result" | "error_message"}.
"""

import inspect
import json
import logging
import os
import resource
import socket
import socketserver
import sys
import threading
import time
import traceback

# ── Setup logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [daemon] %(levelname)s: %(message)s",
)
log = logging.getLogger("voltrade_daemon")


# ── Constants ────────────────────────────────────────────────────────────────
SOCKET_PATH = os.environ.get("VOLTRADE_DAEMON_SOCKET", "/tmp/voltrade_daemon.sock")
MAX_RSS_MB = int(os.environ.get("VOLTRADE_DAEMON_MAX_RSS_MB", "1024"))  # 1 GB
REQUEST_TIMEOUT_SEC = 300  # DAEMON-TIMEOUT 2026-04-23: was 60s, too tight for full scan. scan_market can legitimately take 90-180s on 11,400 tickers.

# Cap in-flight RPC dispatches. Each handler spawns its own worker thread for
# timeout enforcement, and heavy methods (run_full_scan) internally spawn their
# own thread pools. Without a cap the daemon can hit the container's thread
# limit ("RuntimeError: can't start new thread") when many requests pile up.
# Requests beyond the cap wait on the semaphore rather than failing fast —
# preferable to cascading thread-exhaustion across callers.
MAX_INFLIGHT_REQUESTS = int(os.environ.get("VOLTRADE_DAEMON_MAX_INFLIGHT", "8"))
_inflight_sem = threading.BoundedSemaphore(MAX_INFLIGHT_REQUESTS)

# DAEMON-TIMEOUT-VISIBILITY 2026-07-10: counts dispatch worker threads that
# are actually executing right now, including ones the handler already gave
# up waiting on (t.join(REQUEST_TIMEOUT_SEC) returns after 300s regardless of
# whether the target thread finished — Python threads cannot be forcibly
# killed, so a dispatch() call that itself hangs past the timeout keeps
# running in the background after the client has already been told "Request
# timed out"/"Daemon timeout"). _inflight_sem is released as soon as the
# handler stops waiting, NOT when the thread actually exits, so it does not
# reflect this. This counter does, and is surfaced on _health() so a client
# that just hit a daemon timeout can immediately check whether the daemon is
# actually piling up abandoned work (the root-cause question KNOWN BROKEN's
# prior TIER2-ERROR entries had no way to answer).
_active_dispatch_lock = threading.Lock()
_active_dispatch_count = 0

# DAEMON-TIMEOUT-VISIBILITY 2026-07-20 (KNOWN BROKEN #18 continuation): the
# count above answers "how many", never "what" or "for how long" — live
# TIER2-ERROR catches this session found active_dispatches==2 on every
# occurrence (never near MAX_INFLIGHT_REQUESTS=8), which the count alone
# can't distinguish from "one abandoned run_full_scan zombie thread still
# running from the PRIOR timed-out cycle, competing for the shared
# alpaca_throttle bucket with a fresh run_full_scan that just started" — a
# self-perpetuating cascade would look identical to healthy 2x concurrency
# in the count alone. This registry tracks method name + start time per
# active dispatch so _health() can surface exactly that.
_active_dispatch_detail = {}  # dispatch_id -> {"method": str, "started_at": float}
_active_dispatch_next_id = 0


def _inc_active_dispatch(method):
    """Register a dispatch as active. Returns an id for the matching _dec call."""
    global _active_dispatch_count, _active_dispatch_next_id
    with _active_dispatch_lock:
        _active_dispatch_count += 1
        _active_dispatch_next_id += 1
        dispatch_id = _active_dispatch_next_id
        _active_dispatch_detail[dispatch_id] = {"method": method, "started_at": time.time()}
    return dispatch_id


def _dec_active_dispatch(dispatch_id):
    global _active_dispatch_count
    with _active_dispatch_lock:
        _active_dispatch_count -= 1
        _active_dispatch_detail.pop(dispatch_id, None)


def _active_dispatch_snapshot():
    """Point-in-time [{method, elapsed_sec}, ...] for every dispatch still
    running, including ones the RPC handler already gave up waiting on."""
    now = time.time()
    with _active_dispatch_lock:
        return [
            {"method": d["method"], "elapsed_sec": round(now - d["started_at"], 1)}
            for d in _active_dispatch_detail.values()
        ]


# DAEMON-TIMEOUT-VISIBILITY 2026-07-21 (KNOWN BROKEN #18 continuation):
# csp_universe.py's get_last_layer2_prefetch_stats() (v1.0.418) exposes the
# CSP Layer 2 prefetch cache_hit/completed/total/elapsed_sec/budget_exceeded
# shape, but two prior sessions (v1.0.418, v1.0.454) both tried and failed
# to catch a live TIER2-ERROR and this reading in the SAME window by polling
# /api/diag/timings — that endpoint only reflects the LAST scan_market() call
# that actually RETURNED, never the state of a call still hung past its own
# 300s RPC timeout. This process's own health() RPC runs on a separate
# thread from the (possibly still-zombied) run_full_scan dispatch, so it can
# read csp_universe's module-level dict live, mid-hang, the moment
# server/bot.ts's daemon-timeout catch branch calls it — no stakeout needed.
def _layer2_prefetch_snapshot():
    """Live read of csp_universe's last-recorded Layer 2 prefetch stats,
    plus how stale that reading is. {} if csp_universe hasn't loaded yet
    (no scan has reached Layer 2 in this process) — never triggers an
    import itself, only reads sys.modules if already populated by whatever
    scan already ran. Deliberately no broad except here: dispatch()'s own
    outer try/except already turns any real failure into a visible
    {"status": "error"} RPC response instead of a silently swallowed one
    (test_silent_except_ratchet.py's pinned count forbids adding another)."""
    csp_universe = sys.modules.get("csp_universe")
    if csp_universe is None:
        return {}
    stats = csp_universe.get_last_layer2_prefetch_stats()
    if not stats:
        return {}
    checked_at = stats.get("checked_at")
    if checked_at:
        stats["age_sec"] = round(time.time() - checked_at, 1)
    return stats


# DAEMON-TIMEOUT-VISIBILITY 2026-07-27 (KNOWN BROKEN #18 continuation):
# same live-mid-hang read as _layer2_prefetch_snapshot above, for
# csp_universe.py's Layer 1 hard-gate stats (get_last_layer1_stats(),
# v1.0.521) — every tier_engine_start-stuck occurrence checked this
# session showed Layer 2 as a cache hit (0s), which never explained the
# hang; Layer 1 has its own independent 15-min cache and was never
# surfaced, so it could be the thing actually still running.
def _layer1_stats_snapshot():
    """Live read of csp_universe's last-recorded Layer 1 hard-gate stats,
    plus how stale that reading is. {} if csp_universe hasn't loaded yet."""
    csp_universe = sys.modules.get("csp_universe")
    if csp_universe is None:
        return {}
    stats = csp_universe.get_last_layer1_stats()
    if not stats:
        return {}
    checked_at = stats.get("checked_at")
    if checked_at:
        stats["age_sec"] = round(time.time() - checked_at, 1)
    return stats


# DAEMON-TIMEOUT-VISIBILITY 2026-07-28 (KNOWN BROKEN #18 continuation):
# live TIER2-ERROR occurrences today show `step6_trade_loop_and_options`
# (options_scanner.get_options_trades()) as the preamble's single slowest
# phase at 233-240s of a ~296-300s total — RECURRING despite v1.0.503's
# "ROOT CAUSE FOUND + FIXED" claim for this exact phase. Root cause: Setup 7
# calls vol_surface.get_surface_score() for every high_iv/anchor candidate,
# and that function's own spot/chain/bars fetches are NOT gated by
# alpaca_throttle — a previously uninstrumented cost. Same live-mid-hang
# read pattern as layer1/layer2 above.
def _surface_score_stats_snapshot():
    """Live read of vol_surface's scan-scoped get_surface_score() cost
    accumulator, plus how stale that reading is. {} if vol_surface hasn't
    loaded yet (no scan has reached Setup 7 in this process)."""
    vol_surface = sys.modules.get("vol_surface")
    if vol_surface is None:
        return {}
    stats = vol_surface.get_surface_score_scan_stats()
    if not stats:
        return {}
    checked_at = stats.get("checked_at")
    if checked_at:
        stats["age_sec"] = round(time.time() - checked_at, 1)
    return stats


# ── Heavy imports happen ONCE at daemon startup ──────────────────────────────
# This is the entire reason the daemon exists. Re-importing numpy/pandas/
# LightGBM in 27 different subprocess calls per scan cycle costs 12+ seconds.
# Here we pay the cost once and keep the modules resident.
log.info("Importing heavy modules...")
_t0 = time.time()
try:
    import numpy as np
    import pandas as pd
    import requests
    log.info(f"  numpy+pandas+requests: {time.time() - _t0:.2f}s")
except ImportError as e:
    log.error(f"Critical import failed: {e}")
    sys.exit(2)

# Lazy-import the VolTradeAI modules (they pull in more deps)
_modules_loaded = {}


def _lazy_import(name):
    """Import a module once, cache it, return it. Safe for missing deps."""
    if name in _modules_loaded:
        return _modules_loaded[name]
    try:
        mod = __import__(name)
        _modules_loaded[name] = mod
        return mod
    except ImportError as e:
        log.warning(f"Module {name} unavailable: {e}")
        return None


# ── RPC Dispatcher ───────────────────────────────────────────────────────────
class RPCDispatcher:
    """Maps RPC method names to actual Python callables."""

    def __init__(self):
        # Whitelist of allowed methods. Each entry: (module_name, attr_name)
        # If module_name is None, handler is a local method on this class.
        self._routes = {
            # Health / daemon control
            "health": (None, "_health"),
            "ping": (None, "_ping"),

            # ML control
            "ml_status": ("ml_status_impl", None),
            "ml_toggle": ("ml_toggle_impl", None),
            "track_fill": ("ml_model_v2", "track_fill"),

            # Trading helpers
            "check_halt": ("position_sizing", "check_halt_status"),
            "select_contract": ("options_execution", "select_contract"),
            "submit_options_order": ("options_execution", "submit_options_order"),
            "evaluate_and_execute": ("options_execution", "evaluate_and_execute"),

            # Risk management (added 2026-04-20 for monitoring endpoints)
            "risk_status": ("risk_kill_switch", "get_kill_switch_status"),
            "get_peak_equity": ("risk_kill_switch", "get_peak_equity"),
            "check_position_risk": ("risk_kill_switch", "check_position_risk"),
            "check_correlation_pre_trade": ("risk_kill_switch", "check_correlation_pre_trade"),

            # Regime / macro
            "macro_snapshot": ("macro_data", "get_macro_snapshot"),

            # Shadow portfolio stats
            "shadow_stats": ("shadow_portfolio", "get_shadow_stats"),

            # Cache inventory (added 2026-04-20)
            "cache_inventory": (None, "_cache_inventory"),

            # Scan — route to bot_engine.scan_market (fixed 2026-04-22,
            # previously pointed at non-existent "main_scan" which made
            # every daemon-routed scan return "Method main_scan not found"
            "run_full_scan": ("bot_engine", "scan_market"),
        }

    def dispatch(self, method: str, args: dict) -> dict:
        """Dispatch a method call and return JSON-serializable result."""
        # DAEMON-TRACE 2026-04-23: write heartbeat files at each phase
        # of daemon dispatch so we can see exactly where hang occurs.
        _dstate = {}
        _dpath = "/tmp/voltrade_daemon_trace.json"
        if method == "run_full_scan":
            try:
                import json as _dtj
                import os as _dto
                import time as _dtt
                if _dto.path.exists(_dpath):
                    try:
                        with open(_dpath) as _df:
                            _dstate = _dtj.load(_df)
                    except Exception:
                        _dstate = {}
                _dstate["last_request_received"] = _dtt.time()
                _dstate["last_method"] = "run_full_scan"
                _dstate["last_status"] = "received"
                with open(_dpath, "w") as _df:
                    _dtj.dump(_dstate, _df)
            except Exception:
                pass

        if method not in self._routes:
            return {"status": "error",
                    "error_message": f"Unknown method: {method}"}

        module_name, attr_name = self._routes[method]

        try:
            if module_name is None:
                # Local method
                fn = getattr(self, attr_name)
                result = fn(args)
            else:
                # External module method
                mod = _lazy_import(module_name)
                if mod is None:
                    # Module unavailable — inline impls for ml_status/ml_toggle
                    if method == "ml_status":
                        return {"status": "ok", "result": self._ml_status_fallback()}
                    elif method == "ml_toggle":
                        return {"status": "ok",
                                "result": self._ml_toggle_fallback(args)}
                    else:
                        return {"status": "error",
                                "error_message": f"Module {module_name} not loaded"}

                fn = getattr(mod, attr_name, None)
                if fn is None:
                    return {"status": "error",
                            "error_message":
                            f"Method {attr_name} not found in {module_name}"}

                if method == "run_full_scan":
                    # DAEMON-TRACE 2026-04-23: phase = calling_scan_market
                    try:
                        import json as _dtj
                        import time as _dtt
                        _dstate["last_status"] = "calling_scan_market"
                        _dstate["call_start"] = _dtt.time()
                        with open(_dpath, "w") as _df:
                            _dtj.dump(_dstate, _df)
                    except Exception:
                        pass

                # Call with args dict as kwargs or positional depending on
                # the method's signature.
                #
                # REPAIR 2026-07-20 (live break: run_full_scan/scan_market
                # failing every cycle with "scan_market() takes 0 positional
                # arguments but 1 was given"): the old code called fn(**args)
                # inside a `try: ... except TypeError:` and, on ANY TypeError,
                # blindly retried fn(args) assuming the first call failed
                # because of a calling-convention mismatch. But scan_market()
                # takes zero args, so fn(**{}) IS fn() — it can only raise a
                # TypeError from something inside the function's own
                # execution, never from the call itself. The retry then
                # called fn({}) — one positional arg into a zero-arg
                # function — which fails FOR A DIFFERENT REASON and
                # overwrites the real error with a misleading one. Every
                # real TypeError raised during a scan was being silently
                # replaced by this phantom "takes 0 positional arguments"
                # message, making the actual bug undiagnosable from the
                # audit log (confirmed live 2026-07-20: 30+ min of
                # consecutive scan failures, all reporting the same
                # misleading message).
                #
                # Fix: decide kwargs-vs-positional by validating the BINDING
                # against the function's real signature (inspect.signature
                # .bind — this does not call the function, so it cannot
                # raise a TypeError from the function body). Only fall back
                # to positional-dict calling when the binding genuinely
                # doesn't match. Once the calling convention is decided, the
                # real call is made OUTSIDE any except TypeError — so a
                # TypeError raised by the method's own logic propagates with
                # its real message to the outer handler below.
                if isinstance(args, dict):
                    try:
                        inspect.signature(fn).bind(**args)
                        _call_as_kwargs = True
                    except TypeError:
                        _call_as_kwargs = False
                else:
                    _call_as_kwargs = False

                # The actual call is OUTSIDE the bind-check's try/except, so
                # a TypeError raised by fn's own body is never mistaken for
                # a signature mismatch and never triggers the positional
                # retry.
                result = fn(**args) if _call_as_kwargs else fn(args)

                if method == "run_full_scan":
                    # DAEMON-TRACE 2026-04-23: phase = scan_market_returned
                    try:
                        import json as _dtj
                        import time as _dtt
                        _dstate["last_status"] = "scan_market_returned"
                        _dstate["call_end"] = _dtt.time()
                        _dstate["call_duration"] = _dtt.time() - _dstate.get("call_start", _dtt.time())
                        with open(_dpath, "w") as _df:
                            _dtj.dump(_dstate, _df)
                    except Exception:
                        pass

            # Ensure result is JSON-serializable
            return {"status": "ok", "result": result}

        except Exception as e:
            log.error(f"RPC {method} failed: {e}\n{traceback.format_exc()}")
            return {"status": "error",
                    "error_message": str(e)[:500],
                    "traceback": traceback.format_exc()[:2000]}

    def _health(self, args):
        """Health check — returns daemon status, memory, uptime."""
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux ru_maxrss is in KB; on macOS it's in bytes
        if sys.platform == "darwin":
            rss_kb //= 1024
        rss_mb = rss_kb / 1024
        return {
            "alive": True,
            "uptime_seconds": int(time.time() - _DAEMON_START),
            "rss_mb": round(rss_mb, 1),
            "max_rss_mb": MAX_RSS_MB,
            "modules_loaded": list(_modules_loaded.keys()),
            "active_dispatches": _active_dispatch_count,
            "active_dispatch_detail": _active_dispatch_snapshot(),
            "layer2_prefetch": _layer2_prefetch_snapshot(),
            "layer1_stats": _layer1_stats_snapshot(),
            "surface_score_stats": _surface_score_stats_snapshot(),
            "pid": os.getpid(),
        }

    def _ping(self, args):
        """Simplest possible call — for latency testing."""
        return {"pong": True, "t": time.time()}

    def _cache_inventory(self, args):
        """Return top cache files by size for operational visibility."""
        import glob
        result = []
        for pattern in ["/tmp/voltrade_*.json", "/tmp/voltrade_alt_cache/*.json",
                        "/data/voltrade/*.json"]:
            try:
                for f in glob.glob(pattern):
                    try:
                        size = os.path.getsize(f)
                        age_s = int(time.time() - os.path.getmtime(f))
                        result.append({
                            "path": f,
                            "size_kb": round(size / 1024, 1),
                            "age_seconds": age_s,
                        })
                    except OSError:
                        pass
            except Exception:
                pass
        result.sort(key=lambda x: -x["size_kb"])
        return result[:20]

    def _ml_status_fallback(self):
        """Inline reimplementation of ml_status.py for when module unavailable."""
        try:
            from storage_config import DATA_DIR, ML_MODEL_PATH
        except ImportError:
            DATA_DIR = os.environ.get("DATA_DIR", "/tmp")
            ML_MODEL_PATH = os.path.join(DATA_DIR, "voltrade_ml_v2.pkl")

        status_path = os.path.join(DATA_DIR, "ml_status.json")
        toggle_path = os.path.join(DATA_DIR, "ml_toggle.json")

        model_exists = os.path.exists(ML_MODEL_PATH)
        model_age_hours = None
        if model_exists:
            model_age_hours = round(
                (time.time() - os.path.getmtime(ML_MODEL_PATH)) / 3600, 1
            )

        enabled = False
        try:
            with open(toggle_path) as f:
                enabled = json.load(f).get("enabled", False)
        except Exception:
            pass

        last_train = {}
        try:
            with open(status_path) as f:
                last_train = json.load(f)
        except Exception:
            pass

        return {
            "model_exists": model_exists,
            "model_age_hours": model_age_hours,
            "enabled": enabled,
            "contributes_to_cagr": model_exists and enabled,
            "last_status": last_train.get("status", "unknown"),
        }

    def _ml_toggle_fallback(self, args):
        """Inline reimplementation of ml_toggle.py."""
        try:
            from storage_config import DATA_DIR
        except ImportError:
            DATA_DIR = os.environ.get("DATA_DIR", "/tmp")
        toggle_path = os.path.join(DATA_DIR, "ml_toggle.json")
        enabled = args.get("enabled", None)
        if enabled is None:
            # Just return current status
            try:
                with open(toggle_path) as f:
                    return {"enabled": json.load(f).get("enabled", False),
                            "status": "ok"}
            except Exception:
                return {"enabled": False, "status": "ok"}
        else:
            with open(toggle_path, "w") as f:
                json.dump({"enabled": bool(enabled)}, f)
            return {"enabled": bool(enabled), "status": "ok"}


# ── Unix Socket Server ───────────────────────────────────────────────────────
class RPCHandler(socketserver.StreamRequestHandler):
    """Handles a single RPC call over Unix socket."""

    def handle(self):
        try:
            # Read line-delimited JSON request
            line = self.rfile.readline().strip()
            if not line:
                return
            request = json.loads(line.decode("utf-8"))
            method = request.get("method", "")
            args = request.get("args", {})

            # Cap concurrent dispatches so we don't blow past the container's
            # thread limit under burst load. Fail fast with a clear error if we
            # can't acquire within a short window — caller should back off.
            acquired = _inflight_sem.acquire(timeout=REQUEST_TIMEOUT_SEC)  # DAEMON-TIMEOUT 2026-04-23
            if not acquired:
                response = {
                    "status": "error",
                    "error_message": f"Daemon busy: >{MAX_INFLIGHT_REQUESTS} in-flight requests",
                }
                try:
                    self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
                except Exception:
                    pass
                return

            try:
                # Dispatch with timeout
                result_holder = {"done": False, "response": None}

                def _run():
                    dispatch_id = _inc_active_dispatch(method)
                    try:
                        result_holder["response"] = _dispatcher.dispatch(method, args)
                    except Exception as e:
                        result_holder["response"] = {
                            "status": "error",
                            "error_message": str(e)[:500],
                        }
                    finally:
                        result_holder["done"] = True
                        _dec_active_dispatch(dispatch_id)

                try:
                    t = threading.Thread(target=_run, daemon=True)
                    t.start()
                    t.join(REQUEST_TIMEOUT_SEC)  # DAEMON-TIMEOUT 2026-04-23
                except RuntimeError as thread_err:
                    # "can't start new thread" — return a structured error so
                    # the client can back off rather than loop.
                    log.error(f"Thread creation failed: {thread_err}")
                    response = {
                        "status": "error",
                        "error_message": f"Daemon thread creation failed: {thread_err}",
                    }
                    try:
                        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
                    except Exception:
                        pass
                    return

                if not result_holder["done"]:
                    response = {
                        "status": "error",
                        "error_message": f"Request timed out after {REQUEST_TIMEOUT_SEC}s",
                    }
                else:
                    response = result_holder["response"]

                self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
            finally:
                _inflight_sem.release()

        except Exception as e:
            log.error(f"Handler error: {e}")
            try:
                self.wfile.write(
                    (json.dumps({"status": "error",
                                 "error_message": str(e)[:500]}) + "\n").encode("utf-8")
                )
            except Exception:
                pass


class ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Multi-threaded Unix socket server."""
    daemon_threads = True
    allow_reuse_address = True


# ── Memory self-monitoring ───────────────────────────────────────────────────
def _memory_watchdog():
    """Background thread — self-kill if memory exceeds limit."""
    while True:
        try:
            time.sleep(30)
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                rss_kb //= 1024
            rss_mb = rss_kb / 1024
            if rss_mb > MAX_RSS_MB:
                log.error(f"Memory {rss_mb:.0f} MB exceeds limit {MAX_RSS_MB} MB — exiting for respawn")
                os._exit(3)  # Hard exit — supervisor should respawn
        except Exception as e:
            log.warning(f"Watchdog error: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────
_DAEMON_START = time.time()
_dispatcher = RPCDispatcher()


def main():
    """Start the daemon and listen for RPC requests."""
    # Remove stale socket file
    if os.path.exists(SOCKET_PATH):
        try:
            os.unlink(SOCKET_PATH)
        except OSError as e:
            log.error(f"Cannot remove stale socket {SOCKET_PATH}: {e}")
            sys.exit(1)

    # ── ONE-SHOT FEEDBACK SEED 2026-05-03 (alpha audit) ───────────────────
    # If trade_feedback.json doesn't exist or is empty, seed it from the
    # 10-year backtest so the Kelly gate has real per-bucket data on day
    # one. Without this, the gate falls back to hardcoded defaults that
    # may be wrong/stale, and bucket statistics need 10 live trades to
    # take over — which can be weeks of trading.
    #
    # Seeding is IDEMPOTENT — the seeder uses (ticker, time_placed, exit_time,
    # raw_strategy) as record id and skips records already present, so this
    # runs every startup safely.
    try:
        from storage_config import TRADE_FEEDBACK_PATH as _TFP
    except Exception:
        _TFP = "/tmp/voltrade_trade_feedback.json"
    _seed_needed = False
    try:
        if not os.path.exists(_TFP):
            _seed_needed = True
        else:
            with open(_TFP) as _ff:
                _existing = json.load(_ff)
            _seed_needed = len(_existing) < 100  # treat near-empty as needs seeding
    except Exception:
        _seed_needed = True
    if _seed_needed and os.environ.get("VOLTRADE_AUTOSEED", "true").lower() != "false":
        log.info(f"trade_feedback.json missing or sparse → auto-seeding from backtest")
        try:
            import subprocess
            _seed_script = os.path.join(os.path.dirname(__file__),
                                         "seed_feedback_from_backtest.py")
            if os.path.exists(_seed_script):
                _r = subprocess.run([sys.executable, _seed_script], capture_output=True,
                                    text=True, timeout=120)
                if _r.returncode == 0:
                    log.info(f"  seed succeeded: {_r.stdout.strip().splitlines()[-1] if _r.stdout else 'ok'}")
                else:
                    log.warning(f"  seed script returned {_r.returncode}: {_r.stderr[:300]}")
            else:
                log.warning(f"  seed script not found at {_seed_script}")
        except Exception as _seed_err:
            log.warning(f"  auto-seed failed (non-fatal): {_seed_err}")

    # Start memory watchdog
    watchdog = threading.Thread(target=_memory_watchdog, daemon=True)
    watchdog.start()

    # Warm up common modules in background
    def _warmup():
        log.info("Warming up trading modules...")
        for mod_name in ("storage_config", "position_sizing",
                         "options_execution", "ml_model_v2", "bot_engine"):
            _lazy_import(mod_name)
        log.info(f"  modules loaded: {list(_modules_loaded.keys())}")
    threading.Thread(target=_warmup, daemon=True).start()

    # Start server
    log.info(f"Listening on {SOCKET_PATH}")
    server = ThreadingUnixServer(SOCKET_PATH, RPCHandler)

    # Make the socket world-accessible so non-root users can connect
    try:
        os.chmod(SOCKET_PATH, 0o666)
    except OSError:
        pass

    log.info(f"voltrade_daemon ready (pid={os.getpid()})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass


if __name__ == "__main__":
    main()
