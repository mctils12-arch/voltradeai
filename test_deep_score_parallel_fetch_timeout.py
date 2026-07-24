"""
REPAIR 2026-07-21 (KNOWN BROKEN #18 investigation, open_questions.md) —
regression test for deep_score()'s parallel-fetch shutdown hazard.

`with ThreadPoolExecutor(max_workers=5) as _pool:` calls
`pool.shutdown(wait=True)` on exit — an unbounded `.join()` on every worker
thread, regardless of whether that thread's own `.result(timeout=15)` call
already gave up on it. This silently defeated the block's own documented
"~3-4s, limited by the slowest source" contract: a single slow/hung fetcher
(e.g. a finnhub/pytrends call stalling past its own internal timeout) could
block deep_score's return for however long that thread actually took, not
the 15s ceiling the code appears to promise. Fix: manual
`_pool.shutdown(wait=False)` in a `finally` block — abandoned threads keep
running harmlessly in the background instead of blocking the caller.

Two tests: (1) a source-shape check pinning the fixed pattern in
deep_score() itself (mirrors test_deep_score_source_diag.py's convention),
(2) a real behavioral reproduction of the bug class using the exact same
ThreadPoolExecutor idiom, proving the old context-manager form blocks past
its own per-future timeout while the fixed form does not.
"""
import inspect
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import bot_engine


class TestDeepScoreParallelFetchShutdown(unittest.TestCase):
    def test_deep_score_uses_manual_nonblocking_shutdown(self):
        src = inspect.getsource(bot_engine.deep_score)
        self.assertNotIn(
            "with ThreadPoolExecutor(max_workers=5) as _pool:", src,
            "deep_score must not use the ThreadPoolExecutor context manager "
            "for its 5 enrichment fetchers — __exit__ calls shutdown(wait=True), "
            "an unbounded join that defeats the per-future .result(timeout=15) "
            "ceiling (KNOWN BROKEN #18).",
        )
        self.assertIn("_pool.shutdown(wait=False)", src,
                      "deep_score must explicitly shut down its enrichment-fetch "
                      "pool with wait=False so a hung fetcher is abandoned, not joined.")


class TestThreadPoolShutdownWaitBehavior(unittest.TestCase):
    """Proves the bug class itself (not deep_score's internals, which pull in
    heavy real network dependencies): the context-manager form blocks the
    caller past a future's own timeout(); shutdown(wait=False) does not."""

    SLOW_SECONDS = 1.5
    RESULT_TIMEOUT = 0.2

    def _slow_task(self):
        time.sleep(self.SLOW_SECONDS)
        return "done"

    def test_context_manager_form_blocks_past_result_timeout(self):
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self._slow_task)
            try:
                fut.result(timeout=self.RESULT_TIMEOUT)
            except Exception:
                pass
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(
            elapsed, self.SLOW_SECONDS * 0.9,
            "sanity check: the OLD pattern (context manager) should still "
            "block for close to the full slow-task duration on exit, "
            "confirming this reproduces the exact hazard being fixed.",
        )

    def test_manual_shutdown_wait_false_does_not_block_past_result_timeout(self):
        start = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(self._slow_task)
            try:
                fut.result(timeout=self.RESULT_TIMEOUT)
            except Exception:
                pass
        finally:
            pool.shutdown(wait=False)
        elapsed = time.monotonic() - start
        self.assertLess(
            elapsed, self.SLOW_SECONDS * 0.5,
            "the FIXED pattern must return promptly (bounded by "
            "RESULT_TIMEOUT, not the slow task's real duration) — this is "
            "the exact behavior deep_score's enrichment fetch now relies on.",
        )


if __name__ == "__main__":
    unittest.main()
