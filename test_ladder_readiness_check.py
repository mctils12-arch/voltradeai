"""
Tests for scripts/ladder_readiness_check.py — the EDGE DOCTRINE #3
compiled-knowledge check for gateN_pending ROOT VALIDATION LADDER roots
(see that file's module docstring for why it exists: usaspending_contracts's
"unblocks 2026-08-15" condition alone was manually re-derived over a dozen
times across research/experiments.md sessions before this script existed).

Run: python3 -m pytest test_ladder_readiness_check.py -v
"""
import importlib.util
import os
import sys
import unittest
from datetime import date

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

_spec = importlib.util.spec_from_file_location(
    "ladder_readiness_check",
    os.path.join(REPO_ROOT, "scripts", "ladder_readiness_check.py"),
)
readiness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(readiness)


class TestDateTrigger(unittest.TestCase):
    def test_before_not_ready(self):
        trigger = {"type": "date", "not_before": "2026-08-15"}
        out = readiness.evaluate_trigger(trigger, date(2026, 8, 13))
        self.assertFalse(out["ready"])
        self.assertEqual(out["days"], 2)

    def test_exact_day_is_ready(self):
        trigger = {"type": "date", "not_before": "2026-08-15"}
        out = readiness.evaluate_trigger(trigger, date(2026, 8, 15))
        self.assertTrue(out["ready"])
        self.assertEqual(out["days"], 0)

    def test_after_ready_with_overdue_count(self):
        trigger = {"type": "date", "not_before": "2026-08-15"}
        out = readiness.evaluate_trigger(trigger, date(2026, 8, 20))
        self.assertTrue(out["ready"])
        self.assertEqual(out["days"], 5)


class TestArchiveDaysTrigger(unittest.TestCase):
    def test_not_enough_elapsed(self):
        trigger = {"type": "archive_days", "since": "2026-07-04", "min_days": 90}
        out = readiness.evaluate_trigger(trigger, date(2026, 8, 13))
        self.assertFalse(out["ready"])
        # 2026-07-04 -> 2026-08-13 is 40 days elapsed; 90 - 40 = 50 remaining
        self.assertEqual(out["days"], 50)

    def test_exactly_min_days_is_ready(self):
        trigger = {"type": "archive_days", "since": "2026-07-04", "min_days": 90}
        out = readiness.evaluate_trigger(trigger, date(2026, 10, 2))
        self.assertTrue(out["ready"])
        self.assertEqual(out["days"], 0)

    def test_past_min_days_reports_overdue(self):
        trigger = {"type": "archive_days", "since": "2026-07-04", "min_days": 90}
        out = readiness.evaluate_trigger(trigger, date(2026, 10, 12))
        self.assertTrue(out["ready"])
        self.assertEqual(out["days"], 10)


class TestWeeklyReportsTrigger(unittest.TestCase):
    def test_far_from_ready(self):
        trigger = {"type": "weekly_reports", "since": "2026-07-08", "min_count": 15, "cadence_days": 7}
        out = readiness.evaluate_trigger(trigger, date(2026, 8, 13))
        self.assertFalse(out["ready"])
        self.assertIn("ESTIMATE", out["detail"])

    def test_ready_once_enough_weeks_elapsed(self):
        # 15 weeks * 7 days = 105 days after 2026-07-08 -> 2026-10-21
        trigger = {"type": "weekly_reports", "since": "2026-07-08", "min_count": 15, "cadence_days": 7}
        out = readiness.evaluate_trigger(trigger, date(2026, 10, 21))
        self.assertTrue(out["ready"])

    def test_default_cadence_is_weekly(self):
        trigger = {"type": "weekly_reports", "since": "2026-07-08", "min_count": 1}
        out = readiness.evaluate_trigger(trigger, date(2026, 7, 15))
        self.assertTrue(out["ready"])  # exactly 7 days elapsed, 1 report estimated


class TestUnrecognizedTriggerFailsLoud(unittest.TestCase):
    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            readiness.evaluate_trigger({"type": "phase_of_the_moon"}, date(2026, 8, 13))


class TestCheckAllAgainstLiveLadder(unittest.TestCase):
    """Guards the real datacore/signal_ladder.json — proves the three roots this
    session wired up (usaspending_contracts, cftc_cot_positioning,
    sec_8k_earnings_language) are readable and evaluable, and that check_all()
    only returns roots that actually carry a trigger (not every root in the
    ladder)."""

    def test_known_gated_roots_present_and_evaluable(self):
        results = readiness.check_all(today=date(2026, 8, 13))
        ids = {r["id"] for r in results}
        for expected in ("usaspending_contracts", "cftc_cot_positioning", "sec_8k_earnings_language"):
            self.assertIn(expected, ids, f"{expected} should carry a readiness_trigger in signal_ladder.json")

    def test_usaspending_not_yet_ready_on_aug_13(self):
        results = readiness.check_all(today=date(2026, 8, 13))
        row = next(r for r in results if r["id"] == "usaspending_contracts")
        self.assertFalse(row["ready"])
        self.assertEqual(row["days"], 2)

    def test_usaspending_ready_on_aug_15(self):
        results = readiness.check_all(today=date(2026, 8, 15))
        row = next(r for r in results if r["id"] == "usaspending_contracts")
        self.assertTrue(row["ready"])

    def test_roots_without_trigger_are_omitted(self):
        results = readiness.check_all(today=date(2026, 8, 13))
        ids = {r["id"] for r in results}
        # grid_vision_tower_detector is 'killed' and carries no readiness_trigger —
        # this tool is not a dump of the whole ladder, only the gated-with-a-known-condition subset.
        self.assertNotIn("grid_vision_tower_detector", ids)

    def test_ladder_json_is_the_source_of_truth_not_a_copy(self):
        # Prove this test suite doesn't hardcode a duplicate of the ladder —
        # every result must trace back to a real entry with a source_note.
        results = readiness.check_all(today=date(2026, 8, 13))
        for r in results:
            self.assertTrue(r["source_note"], f"{r['id']} readiness_trigger is missing its source_note")


if __name__ == "__main__":
    unittest.main()
