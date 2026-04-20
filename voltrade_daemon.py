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
REQUEST_TIMEOUT_SEC = 60  # Hard timeout per request


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
            "shadow_stats": ("shadow_portfolio", "get_stats"),

            # Cache inventory (added 2026-04-20)
            "cache_inventory": (None, "_cache_inventory"),

            # Scan
            "run_full_scan": ("bot_engine", "main_scan"),
        }

    def dispatch(self, method: str, args: dict) -> dict:
        """Dispatch a method call and return JSON-serializable result."""
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

                # Call with args dict as kwargs or positional depending on
                # the method's signature. Try kwargs first.
                try:
                    result = fn(**args) if isinstance(args, dict) else fn(args)
                except TypeError:
                    # Some methods take a single positional dict
                    result = fn(args)

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

            # Dispatch with timeout
            result_holder = {"done": False, "response": None}

            def _run():
                try:
                    result_holder["response"] = _dispatcher.dispatch(method, args)
                except Exception as e:
                    result_holder["response"] = {
                        "status": "error",
                        "error_message": str(e)[:500],
                    }
                finally:
                    result_holder["done"] = True

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(REQUEST_TIMEOUT_SEC)

            if not result_holder["done"]:
                response = {
                    "status": "error",
                    "error_message": f"Request timed out after {REQUEST_TIMEOUT_SEC}s",
                }
            else:
                response = result_holder["response"]

            self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))

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
