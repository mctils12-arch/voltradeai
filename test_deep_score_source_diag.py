"""
REPAIR 2026-07-06 pt.2 — regression test for deep_score()'s enrichment-fetch
blind spot.

KNOWN BROKEN #5's 2026-07-04 audit found deep_score()'s 5 parallel fetchers
(_fetch_macro/_fetch_intel/_fetch_alt/_fetch_social/_fetch_finnhub) swallow
their own exceptions via a bare `except Exception: return {}` with zero
trace, and only added CACHE-FRESHNESS monitoring for 2 of them (reddit_/fh_
prefixed files in diagnostics.py) — never the actual exception. This is the
exact silent-degradation shape _fetch_snap had before v1.0.148's fix. This
test pins _run_diag_fetch (bot_engine.py) — the pure helper each of the 5
fetchers now routes through — against both the capture and the pass-through
cases, and confirms it never resolves any network call itself.
"""
import unittest

from bot_engine import _run_diag_fetch


class TestRunDiagFetch(unittest.TestCase):
    def test_success_passes_value_through_untouched(self):
        diag = {}
        result = _run_diag_fetch("macro", lambda: {"vix": 18.2}, diag)
        self.assertEqual(result, {"vix": 18.2})
        self.assertEqual(diag, {}, "a successful fetch must not write anything into diag")

    def test_exception_captured_with_type_and_message_then_reraised(self):
        diag = {}
        with self.assertRaises(ValueError):
            _run_diag_fetch("intel", lambda: (_ for _ in ()).throw(ValueError("insider feed 500")), diag)
        self.assertIn("intel", diag)
        self.assertEqual(diag["intel"], "ValueError: insider feed 500")

    def test_diag_none_still_reraises_without_error(self):
        # deep_score() itself is always called with a real dict in prod, but
        # _diag defaults to None (backward-compatible signature) — must not
        # raise a *second*, unrelated exception (e.g. NoneType has no
        # __setitem__) while propagating the original one.
        with self.assertRaises(RuntimeError):
            _run_diag_fetch("alt", lambda: (_ for _ in ()).throw(RuntimeError("boom")), None)

    def test_message_truncated_to_150_chars(self):
        # Long tracebacks/response bodies (e.g. a raw HTML error page from a
        # rate-limited API) must not blow up the diag payload that eventually
        # rides on the token-gated /api/diag/scanner probe.
        diag = {}
        long_msg = "x" * 500
        with self.assertRaises(Exception):
            _run_diag_fetch("social", lambda: (_ for _ in ()).throw(Exception(long_msg)), diag)
        self.assertEqual(diag["social"], "Exception: " + "x" * 150)

    def test_distinct_sources_do_not_clobber_each_other(self):
        diag = {}
        with self.assertRaises(Exception):
            _run_diag_fetch("macro", lambda: (_ for _ in ()).throw(Exception("macro down")), diag)
        with self.assertRaises(Exception):
            _run_diag_fetch("finnhub", lambda: (_ for _ in ()).throw(Exception("finnhub down")), diag)
        self.assertEqual(diag, {"macro": "Exception: macro down", "finnhub": "Exception: finnhub down"})


class TestDeepScoreFetchersRouteThroughDiagHelper(unittest.TestCase):
    """Static-shape check (no network): every one of the 5 fetch closures'
    source code calls _run_diag_fetch with its own distinct source key, so a
    failure in any single source is individually attributable rather than
    collapsing to one generic 'enrichment failed' bucket."""

    def test_all_five_sources_wired(self):
        import inspect
        import bot_engine
        src = inspect.getsource(bot_engine.deep_score)
        for source_name in ("macro", "intel", "alt", "social", "finnhub"):
            self.assertIn(f'"{source_name}"', src,
                          f"deep_score must call _run_diag_fetch with source key {source_name!r}")
        self.assertEqual(src.count("_run_diag_fetch("), 5, "expected exactly 5 call sites, one per enrichment source")


if __name__ == "__main__":
    unittest.main()
