"""
KNOWN BROKEN #18, fourth-mechanism investigation (rate-limiter contention
during deep_score) — regression test for the credit_spread redundant-fetch
fix, 2026-07-19.

Prior session (2026-07-18) directly measured the root cause of the
recurring TIER2-ERROR "Daemon timeout" storm: KNOWN BROKEN #23's bars-feed
fix (v1.0.397) turned 14 previously-fast-failing `/v2/stocks/bars` call
sites into real, throttled network round-trips sharing one process-wide
180 req/min bucket (alpaca_rate_limiter.py, FROZEN). This session found
one of the worst offenders by READ BEFORE WRITE: deep_score()'s TLT/HYG
credit_spread fetch (bot_engine.py) is market-wide, not ticker-specific,
but was re-fetched fresh on every single deep_score() call — up to 15x
redundant identical Alpaca requests per scan cycle (one per deep-scored
candidate), all funneling through the same shared throttle bucket.

The fix mirrors the existing, already-proven _MOST_ACTIVES_CACHE pattern
(2026-05-18, same file, same bug class) — a 60s TTL module-level cache,
since a 21-day return spread cannot materially shift within one scan
cycle. This test proves the cache actually prevents the redundant fetch
(not just that the code contains a cache check) and that it doesn't
change the computed value or silently break deep_score.
"""
import unittest
from unittest.mock import patch

import bot_engine


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def _make_fake_get(call_counts):
    """requests.get stub: real TLT/HYG bars for the credit_spread request
    (counted), an empty-but-valid {} for every other endpoint deep_score's
    many other fetchers hit — enough for their own `.json().get(...)`
    fallback paths to degrade gracefully without a real network call."""
    def _fake_get(url, params=None, headers=None, timeout=None, **kw):
        params = params or {}
        if params.get("symbols") == "TLT,HYG":
            call_counts["credit_spread"] += 1
            return _FakeResp({"bars": {
                "TLT": [{"c": 100.0 + i * 0.05} for i in range(30)],
                "HYG": [{"c": 80.0 + i * 0.03} for i in range(30)],
            }})
        return _FakeResp({})
    return _fake_get


def _quick_result(ticker="NVDA"):
    return {
        "ticker": ticker, "price": 100.0, "close": 100.0, "prev_close": 99.0,
        "volume": 5_000_000, "change_pct": 1.0, "above_vwap": True,
        "range_pct": 1.2, "vwap_dist": 0.3, "reasons": [], "quick_score": 28,
    }


class TestCreditSpreadCache(unittest.TestCase):
    def setUp(self):
        # Force a cold cache before each test — deep_score() calls from a
        # prior test (or this module import) must not leak a warm cache in.
        bot_engine._CREDIT_SPREAD_CACHE = 0.0
        bot_engine._CREDIT_SPREAD_CACHE_TIME = 0.0
        # get_stock_details() shells out to analyze.py via subprocess — not
        # under test here (see test_get_stock_details_diag.py for that).
        # deep_score()'s own early-return guard (`if not detail or "error"
        # in detail: return quick_result`) only needs a truthy, error-free
        # dict to proceed into the enrichment/credit_spread block.
        self._gsd_patch = patch("bot_engine.get_stock_details", return_value={"price": 100.0})
        self._gsd_patch.start()
        self.addCleanup(self._gsd_patch.stop)

    def test_second_call_within_ttl_reuses_cache_not_a_second_fetch(self):
        call_counts = {"credit_spread": 0}
        with patch("bot_engine.requests.get", side_effect=_make_fake_get(call_counts)):
            r1 = bot_engine.deep_score("NVDA", _quick_result("NVDA"))
            r2 = bot_engine.deep_score("AAPL", _quick_result("AAPL"))

        self.assertEqual(
            call_counts["credit_spread"], 1,
            "credit_spread is market-wide (TLT/HYG), not ticker-specific — "
            "a second deep_score() call within the 60s TTL must reuse the "
            "cached value instead of re-issuing an identical Alpaca request",
        )
        cs1 = (r1.get("entry_features") or {}).get("credit_spread")
        cs2 = (r2.get("entry_features") or {}).get("credit_spread")
        self.assertIsNotNone(cs1)
        self.assertEqual(
            cs1, cs2,
            "cached credit_spread must produce the identical feature value "
            "for a second ticker scored in the same scan cycle",
        )

    def test_expired_cache_triggers_a_fresh_fetch(self):
        call_counts = {"credit_spread": 0}
        with patch("bot_engine.requests.get", side_effect=_make_fake_get(call_counts)):
            bot_engine.deep_score("NVDA", _quick_result("NVDA"))
            self.assertEqual(call_counts["credit_spread"], 1)

            # Simulate the cache going stale past its 60s TTL.
            bot_engine._CREDIT_SPREAD_CACHE_TIME -= 61

            bot_engine.deep_score("AAPL", _quick_result("AAPL"))
        self.assertEqual(
            call_counts["credit_spread"], 2,
            "an expired cache entry must trigger a real re-fetch, not stay "
            "permanently stuck on the first scan cycle's value",
        )

    def test_cache_holds_the_real_computed_spread_value(self):
        """Static-shape guard: confirms the cache variable actually gets
        populated with the same value the fetch computed, not left at its
        0.0 default forever (which would silently zero out a live ML
        feature for every candidate after the first)."""
        call_counts = {"credit_spread": 0}
        with patch("bot_engine.requests.get", side_effect=_make_fake_get(call_counts)):
            bot_engine.deep_score("NVDA", _quick_result("NVDA"))
        # TLT: (100 + 29*0.05)/(100 + 8*0.05) - 1 ; HYG mirrors it — just
        # assert it's the nonzero, deterministic value the fake bars imply,
        # not the 0.0 exception-path default.
        self.assertNotEqual(bot_engine._CREDIT_SPREAD_CACHE, 0.0)
        self.assertGreater(bot_engine._CREDIT_SPREAD_CACHE_TIME, 0.0)


if __name__ == "__main__":
    unittest.main()
