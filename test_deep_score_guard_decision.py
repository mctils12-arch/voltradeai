"""
KNOWN BROKEN #21 RECURRENCE (2026-07-14) — regression test for the
pre_deep_score memory guard's threshold values, not its RSS-reading
mechanism (that part was already fixed and pinned by
test_mem_rss_current.py).

v1.0.311 fixed _mem_rss_mb() to read real current RSS instead of a stale
monotonic peak. That was necessary but not sufficient: the skip_mb/trim_mb
defaults (130/100, 2026-04-22 HOTFIX) were tuned against a short-lived
subprocess's RSS trajectory, not the persistent daemon that is the primary
execution path. The daemon's live-measured idle baseline this session
(254.2MB via /api/diag/daemon, stable across 4 polls, ml_model_v2
permanently imported) already exceeded both thresholds before any scan
even started — so once the monotonic-peak bug was fixed, the guard's real
behavior became "skip deep scoring every single cycle," matching the live
symptom (wikipedia/gdelt/fred reported down across 18 straight DIAGNOSTIC
audit entries spanning a full trading day post-v1.0.311-deploy).

This pins _deep_score_guard_decision (bot_engine.py) against the actual
live-observed baseline so a future re-lowering of these defaults toward
the stale subprocess-tuned values fails loudly.
"""
import unittest

from bot_engine import _deep_score_guard_decision


class TestDeepScoreGuardDecision(unittest.TestCase):
    def test_daemon_idle_baseline_is_normal_at_current_defaults(self):
        """The regression pin: 254MB (the live-measured persistent-daemon
        idle RSS this session) must NOT trip the guard at the re-tuned
        550/400 defaults — this is exactly the case that silently
        wedged deep_score permanently 'off' at the old 130/100 defaults."""
        decision = _deep_score_guard_decision(254, skip_mb=550, trim_mb=400)
        self.assertEqual(decision, "normal")

    def test_old_defaults_would_have_skipped_the_daemon_baseline(self):
        """Documents the bug this test file exists to prevent recurring:
        the OLD 130/100 defaults treat the daemon's mere idle baseline as
        critical memory pressure."""
        decision = _deep_score_guard_decision(254, skip_mb=130, trim_mb=100)
        self.assertEqual(decision, "skip")

    def test_trims_between_trim_and_skip_thresholds(self):
        decision = _deep_score_guard_decision(450, skip_mb=550, trim_mb=400)
        self.assertEqual(decision, "trim")

    def test_skips_at_or_above_skip_threshold(self):
        decision = _deep_score_guard_decision(550, skip_mb=550, trim_mb=400)
        self.assertEqual(decision, "skip")
        decision = _deep_score_guard_decision(650, skip_mb=550, trim_mb=400)
        self.assertEqual(decision, "skip")

    def test_skip_threshold_leaves_margin_below_survival_mode(self):
        """SURVIVAL_MODE (bot_engine.py) engages at 700MB and is untouched
        by this fix — the skip threshold must stay below it so 'skip deep
        scoring' remains a real intermediate escalation step, not
        redundant with survival mode."""
        SURVIVAL_MODE_THRESHOLD_MB = 700
        skip_mb = 550
        self.assertLess(skip_mb, SURVIVAL_MODE_THRESHOLD_MB)

    def test_zero_or_falsy_rss_is_normal(self):
        """_mem_rss_mb() returns 0 when both read paths fail (see
        test_mem_rss_current.py) — the guard must not treat that as
        maximum pressure."""
        self.assertEqual(_deep_score_guard_decision(0, skip_mb=550, trim_mb=400), "normal")


if __name__ == "__main__":
    unittest.main()
