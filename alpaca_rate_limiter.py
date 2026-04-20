"""
VolTradeAI — Alpaca API rate limiter
=====================================

Token-bucket rate limiter for Alpaca API calls. Alpaca paid tier is
200 requests/minute. During a full scan with parallel workers + options
chain fetches + fundamentals lookups, peak req/min can exceed that,
causing silent 429 errors that get swallowed by generic except blocks.

Usage:
    from alpaca_rate_limiter import alpaca_throttle

    # Before each Alpaca call:
    alpaca_throttle.acquire()
    resp = requests.get("https://data.alpaca.markets/...", ...)

This is conservative — uses 180/min by default (10% safety margin
below the 200/min limit) to handle burst traffic.

The pattern is copied from finnhub_data.py:51 _TokenBucket with
adjusted defaults for Alpaca's higher rate limit.
"""

import threading
import time


class _AlpacaTokenBucket:
    """
    Token bucket that allows at most `rate` tokens per `period` seconds.
    Thread-safe. Blocks the caller until a token is available.

    Same implementation as finnhub_data.py._TokenBucket — duplicated here
    rather than importing to avoid a dependency on finnhub_data.py
    (which has its own Finnhub-specific rate).
    """

    def __init__(self, rate: int = 180, period: float = 60.0):
        self._rate = rate
        self._period = period
        self._tokens = float(rate)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        """Block until a request token is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            # Refill tokens proportionally to elapsed time
            self._tokens = min(
                float(self._rate),
                self._tokens + elapsed * (self._rate / self._period),
            )
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # Need to wait for the next token
            wait_time = (1.0 - self._tokens) * (self._period / self._rate)

        time.sleep(wait_time)

        with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0)
            self._last_refill = time.monotonic()


# Module-level singleton — all callers share the same rate limiter
#
# 180/min is conservative — paid Alpaca is 200/min. Leaves 10% headroom
# for non-throttled calls elsewhere and handles brief bursts without
# blocking.
alpaca_throttle = _AlpacaTokenBucket(rate=180, period=60.0)


# Convenience wrapper so callers can swap in a single line change:
#   resp = requests.get(url)
#   →
#   resp = alpaca_get(url, headers=...)
def alpaca_get(url, **kwargs):
    """requests.get wrapper that throttles before the call."""
    import requests
    alpaca_throttle.acquire()
    return requests.get(url, **kwargs)


def alpaca_post(url, **kwargs):
    """requests.post wrapper that throttles before the call."""
    import requests
    alpaca_throttle.acquire()
    return requests.post(url, **kwargs)


def alpaca_delete(url, **kwargs):
    """requests.delete wrapper that throttles before the call."""
    import requests
    alpaca_throttle.acquire()
    return requests.delete(url, **kwargs)


if __name__ == "__main__":
    # Self-test: acquire 10 tokens rapidly, verify we don't exceed rate
    print("Testing alpaca_throttle...")
    start = time.monotonic()
    for i in range(10):
        alpaca_throttle.acquire()
        print(f"  acquire #{i+1} at t+{time.monotonic() - start:.3f}s")
    elapsed = time.monotonic() - start
    print(f"10 acquires in {elapsed:.3f}s")
    print(f"Rate observed: {10 / max(elapsed, 0.001):.1f} per sec "
          f"(max is {180/60:.1f} per sec)")
    print("OK" if elapsed < 1.0 else "WARN: throttled more than expected")



# ═══════════════════════════════════════════════════════════════════════
# GLOBAL AUTO-PATCH — apply throttle to ALL Alpaca requests automatically
# ═══════════════════════════════════════════════════════════════════════
# Call install_global_throttle() once at program startup. It monkey-patches
# requests.get/post/delete to auto-throttle any URL going to Alpaca,
# without requiring modification of the 36+ existing call sites.
#
# Usage (call ONCE, e.g. in bot_engine.py or a startup hook):
#     from alpaca_rate_limiter import install_global_throttle
#     install_global_throttle()
#
# Safe to call multiple times — will only patch once.

_patched = False


def install_global_throttle():
    """
    Monkey-patch requests.get/post/delete to throttle Alpaca URLs.
    Idempotent — safe to call from multiple entry points.
    """
    global _patched
    if _patched:
        return
    import requests

    _orig_get = requests.get
    _orig_post = requests.post
    _orig_delete = requests.delete

    def _is_alpaca(url):
        if not isinstance(url, str):
            return False
        return "alpaca.markets" in url

    def _throttled_get(url, **kwargs):
        if _is_alpaca(url):
            alpaca_throttle.acquire()
        return _orig_get(url, **kwargs)

    def _throttled_post(url, **kwargs):
        if _is_alpaca(url):
            alpaca_throttle.acquire()
        return _orig_post(url, **kwargs)

    def _throttled_delete(url, **kwargs):
        if _is_alpaca(url):
            alpaca_throttle.acquire()
        return _orig_delete(url, **kwargs)

    requests.get = _throttled_get
    requests.post = _throttled_post
    requests.delete = _throttled_delete
    _patched = True
