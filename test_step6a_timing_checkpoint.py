"""
KNOWN BROKEN #18 (open_questions.md item 18), 2026-07-26 — fourth
visibility instrument on the recurring TIER2-ERROR daemon-timeout storm.

Live production evidence (server/bot.ts's TIER2-ERROR audit line, active
since v1.0.481's scan_phase field went live 2026-07-25 18:34Z) shows every
occurrence lands with `last_phase=deep_score` at a consistent age of
256-275s — nearly the whole 300s daemon budget — with no checkpoint
between deep_score's own completion and the next existing one
(step6_trade_loop_and_options), even though a 2026-05-04 comment on that
same span already flagged it as a 507s gap worth instrumenting. This test
pins the new `step6a_trade_loop_and_covered_calls` checkpoint bot_engine.py
now logs between the per-candidate trade loop / covered-call sweep and
options_scanner.get_options_trades()'s full-market scan, so the next live
occurrence's scan_timings read tells us directly which half is slow
instead of guessing a fifth theory blind.

Pure source-text pin (matches server/tier2DaemonTimeoutVisibility.test.ts's
and test_deep_score_source_diag.py's convention) — no network calls, no
execution of _scan_market_inner itself.
"""
import inspect
import unittest

import bot_engine


class TestStep6aTimingCheckpoint(unittest.TestCase):
    def setUp(self):
        self.src = inspect.getsource(bot_engine._scan_market_inner)

    def test_new_checkpoint_is_logged(self):
        self.assertIn(
            '_timing_log("step6a_trade_loop_and_covered_calls")',
            self.src,
            "the new split checkpoint must be logged via the same _timing_log "
            "mechanism as every other scan phase",
        )

    def test_new_checkpoint_sits_between_covered_call_sweep_and_options_scanner(self):
        cc_sweep_idx = self.src.index("Covered call sweep error")
        checkpoint_idx = self.src.index('_timing_log("step6a_trade_loop_and_covered_calls")')
        options_scanner_idx = self.src.index("from options_scanner import get_options_trades")
        self.assertTrue(
            cc_sweep_idx < checkpoint_idx < options_scanner_idx,
            "checkpoint must land after the covered-call sweep and before "
            "get_options_trades() — otherwise it can't distinguish the two "
            "halves of the previously-unsplit gap",
        )

    def test_existing_step6_checkpoint_still_present_and_after_the_new_one(self):
        # The pre-existing checkpoint must still fire (unchanged name/position
        # relative to step 7) — this change only ADDS a split, never removes
        # the original end-of-step-6 marker downstream consumers already read.
        checkpoint_idx = self.src.index('_timing_log("step6a_trade_loop_and_covered_calls")')
        existing_idx = self.src.index('_timing_log("step6_trade_loop_and_options")')
        step7_idx = self.src.index("Step 7: Check position management")
        self.assertTrue(
            checkpoint_idx < existing_idx < step7_idx,
            "existing step6_trade_loop_and_options checkpoint must remain, "
            "positioned after the new split and before step 7",
        )


if __name__ == "__main__":
    unittest.main()
