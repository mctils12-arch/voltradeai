"""
alpaca_feed.py — central Alpaca market-data FEED selection with
entitlement fallback ([REPAIR 2026-07-06]).

WHY THIS EXISTS: on 2026-07-06 every market-data request with feed=sip
began returning HTTP 403 (SIP entitlement rejected; the trading API and
/v2/account stayed fine). 44 call sites across the stack hardcoded
feed=sip, so the Tier2 scan — and everything else touching the data
plane — went blind at the open while /api/health read "active"
(surfaced by the v1.0.148 scanner check; named by /api/diag/scanner).

DESIGN: one probe, one switch, zero per-site error wiring.
  - data_feed() returns the feed every call site should use.
  - First use (and every PROBE_TTL_S thereafter) probes a single
    1-symbol snapshot with feed=sip. 403 => the process downgrades to
    "delayed_sip": the FULL consolidated tape at a 15-minute delay,
    free on every account tier. Anything else (200, timeouts, 5xx)
    keeps/restores "sip" — only an explicit entitlement rejection
    downgrades, and a restored subscription upgrades back automatically
    at the next probe.
  - HONESTY: delayed_sip preserves consolidated VOLUME semantics — the
    dollar-volume floors and change-% math stay correct, just 15 min
    stale. feed=iex was deliberately REJECTED as the fallback: IEX-only
    prints undercount volume ~30-50x and would silently poison every
    volume filter (measurement integrity beats freshness). Candidate
    DISCOVERY tolerates 15-min staleness; executions price off live
    quotes via the trading API as before.
  - ALPACA_DATA_FEED env forces a feed and disables probing (tests,
    emergencies).
  - The downgrade is logged loudly ONCE per switch; scan results carry
    data_feed so the audit trail shows which tape produced them.

Pure-ish module: stdlib + requests only; no imports from trading code.
"""
import os
import sys
import time
import threading

PROBE_TTL_S = 600  # re-probe at most every 10 minutes
_DELAYED = "delayed_sip"
_PREFERRED = "sip"

_lock = threading.Lock()
_state = {"feed": _PREFERRED, "probed_at": 0.0, "downgraded_since": None}


def _default_probe(timeout: float = 6.0) -> int:
    """One tiny snapshots request with feed=sip; returns the HTTP status
    (0 on transport error — treated as inconclusive, never a downgrade)."""
    try:
        import requests
        r = requests.get(
            "https://data.alpaca.markets/v2/stocks/snapshots",
            params={"symbols": "SPY", "feed": _PREFERRED},
            headers={
                "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
                "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
            },
            timeout=timeout,
        )
        return r.status_code
    except Exception:
        return 0


def data_feed(now: float | None = None, probe=None) -> str:
    """The Alpaca data feed every call site should request. Cached;
    probes the SIP entitlement at most every PROBE_TTL_S. `probe` is
    injectable for tests (callable -> HTTP status int)."""
    forced = os.environ.get("ALPACA_DATA_FEED", "").strip()
    if forced:
        return forced
    t = now if now is not None else time.time()
    with _lock:
        if t - _state["probed_at"] < PROBE_TTL_S:
            return _state["feed"]
        _state["probed_at"] = t
    status = (probe or _default_probe)()
    with _lock:
        if status == 403 and _state["feed"] != _DELAYED:
            _state["feed"] = _DELAYED
            _state["downgraded_since"] = t
            # REPAIR 2026-07-07: this module is imported by one-shot
            # subprocesses whose ENTIRE stdout is JSON.parse'd by the caller
            # (ml_retrain_safe.py -> bot.ts tier3Strategic; manipulation_detect
            # scan). A fresh subprocess always starts with probed_at=0.0, so
            # while SIP stays 403 this branch fires on literally every
            # invocation, and a stdout print here breaks that JSON.parse every
            # time (confirmed live: TIER3-ML-ERROR "Unexpected token 'F',
            # \"[FEED] SIP \"... is not valid JSON" on 3/3 hourly retrains).
            # stderr is diagnostic output, not a caller-consumed contract —
            # Railway still captures it in logs — so log there instead.
            print(f"[FEED] SIP entitlement rejected (HTTP 403) — downgrading to "
                  f"{_DELAYED} (full consolidated tape, 15-min delay). Restore the "
                  f"Alpaca data subscription to return to real-time automatically.",
                  file=sys.stderr, flush=True)
        elif status == 200 and _state["feed"] != _PREFERRED:
            _state["feed"] = _PREFERRED
            _state["downgraded_since"] = None
            print("[FEED] SIP entitlement restored — back to real-time sip.",
                  file=sys.stderr, flush=True)
        # 0 / 5xx / 429: inconclusive — keep the current feed
        return _state["feed"]


def feed_status() -> dict:
    """For diagnostics/scan meta: current feed + downgrade info."""
    with _lock:
        return {
            "feed": os.environ.get("ALPACA_DATA_FEED", "").strip() or _state["feed"],
            "degraded": _state["feed"] == _DELAYED,
            "downgraded_since": _state["downgraded_since"],
        }


def _reset_for_tests() -> None:
    with _lock:
        _state["feed"] = _PREFERRED
        _state["probed_at"] = 0.0
        _state["downgraded_since"] = None
