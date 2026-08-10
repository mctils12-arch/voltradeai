"""
REPAIR 2026-08-10 (KNOWN BROKEN #29 follow-up) — regression test for
deep_score()'s enrichment-fetch TIMEOUT blind spot.

Live evidence: `/api/diag/audit?type=TIER3-DIAG` showed "Multiple API
sources down: ['polygon', 'wikipedia', 'gdelt', 'fred']" firing on every
hourly check across a 40+ hour window, including 5+ hours AFTER v1.0.637
(the alt_data.py FRED/GDELT parallelization fix) deployed — so that fix
did not clear the symptom, or not fully. Checking `/api/diag/scanner`
live during this session found `dataSourceErrors: {}` on every poll,
which per KNOWN BROKEN #5 / REPAIR 2026-07-06 pt.2 should mean "no
enrichment fetcher raised an exception this cycle" — but `_run_diag_fetch`
(the helper each of deep_score()'s 5 fetchers routes through) only wraps
the fn() call itself. The `.result(timeout=15)` call that bounds each
fetcher lives OUTSIDE that wrapper, at deep_score()'s own call site, and
was caught by a bare `except Exception: pass` with zero capture. A
fetcher that times out (concurrent.futures.TimeoutError) is therefore
indistinguishable from one that never ran at all — dataSourceErrors
staying `{}` was never proof the fetchers succeeded, only proof none of
them raised an exception INSIDE fn() specifically. This test pins the
fix: deep_score()'s 5 result() call sites now capture TimeoutError (and
any other exception not already captured by _run_diag_fetch) into _diag,
without ever overwriting a more specific inner capture.
"""
import inspect
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import bot_engine


class TestDeepScoreSourceCodeCapturesResultTimeouts(unittest.TestCase):
    """Static-shape check (no network): each of the 5 `.result(timeout=15)`
    call sites must capture its exception into _diag, not silently discard
    it via a bare `except Exception: pass`."""

    def test_no_bare_pass_swallows_result_timeout(self):
        # Scoped to the enrichment-fetch block specifically: deep_score's
        # function body has other, unrelated bare 'except Exception: pass'
        # blocks (momentum/mean-rev sub-scores, credit-spread cache, etc.)
        # that are out of scope for this fix — only the 5 .result(timeout=15)
        # call sites are in play here.
        src = inspect.getsource(bot_engine.deep_score)
        self.assertNotIn(
            ".result(timeout=15)\n        except Exception:\n            pass", src,
            "a bare 'except Exception: pass' directly after a .result(timeout=15) "
            "call silently discards TimeoutError with zero trace — deep_score "
            "must capture it into _diag instead (KNOWN BROKEN #29 follow-up).",
        )

    def test_all_five_result_sites_write_into_diag_guarded_by_not_in(self):
        src = inspect.getsource(bot_engine.deep_score)
        for source_name in ("macro", "intel", "alt", "social", "finnhub"):
            self.assertIn(
                f'"{source_name}" not in _diag', src,
                f"{source_name}'s .result(timeout=15) except-clause must guard "
                f"with '\"{source_name}\" not in _diag' so it fills the gap for "
                f"TimeoutError without overwriting a more specific capture "
                f"_run_diag_fetch already recorded for the same source.",
            )


class TestResultTimeoutCaptureBehavior(unittest.TestCase):
    """Proves the bug class itself with the exact ThreadPoolExecutor/.result()
    idiom deep_score uses (no heavy network dependencies): the OLD bare
    'except Exception: pass' pattern loses all trace of a timeout; the FIXED
    guarded-capture pattern records it, and never clobbers an existing entry."""

    def _submit_slow(self, pool, seconds=0.5):
        def _slow():
            time.sleep(seconds)
            return "done"
        return pool.submit(_slow)

    def test_old_bare_except_pass_loses_timeout_info(self):
        diag = {}
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = self._submit_slow(pool)
            try:
                fut.result(timeout=0.05)
            except Exception:
                pass  # the exact old pattern
        finally:
            pool.shutdown(wait=False)
        self.assertEqual(diag, {}, "sanity check: the OLD pattern must leave no trace of the timeout")

    def test_fixed_pattern_captures_timeout_error_type_and_message(self):
        diag = {}
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = self._submit_slow(pool)
            try:
                fut.result(timeout=0.05)
            except Exception as e:
                if diag is not None and "alt" not in diag:
                    diag["alt"] = f"{type(e).__name__}: {str(e)[:150]}"
        finally:
            pool.shutdown(wait=False)
        self.assertIn("alt", diag)
        self.assertTrue(diag["alt"].startswith("TimeoutError"))

    def test_fixed_pattern_never_overwrites_an_existing_more_specific_entry(self):
        # Simulates _run_diag_fetch having already captured a real inner
        # exception (re-raised through .result() unchanged) before this
        # outer guard runs — the guard must not replace it with a
        # differently-worded duplicate.
        diag = {"alt": "ConnectionError: yfinance SSL handshake failed"}
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            def _raise():
                raise ConnectionError("yfinance SSL handshake failed")
            fut = pool.submit(_raise)
            try:
                fut.result(timeout=5)
            except Exception as e:
                if diag is not None and "alt" not in diag:
                    diag["alt"] = f"{type(e).__name__}: {str(e)[:150]}"
        finally:
            pool.shutdown(wait=False)
        self.assertEqual(diag["alt"], "ConnectionError: yfinance SSL handshake failed",
                          "must not overwrite an entry _run_diag_fetch already captured")


if __name__ == "__main__":
    unittest.main()
