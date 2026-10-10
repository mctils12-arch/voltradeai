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
    """Guards the real datacore/signal_ladder.json — proves the roots that
    carry a readiness_trigger are readable and evaluable, and that
    check_all() only returns roots that actually carry one (not every root
    in the ladder). Originally wired up for three roots (usaspending_
    contracts, cftc_cot_positioning, sec_8k_earnings_language); usaspending_
    contracts GRADUATED 2026-08-15 (gate2_fail — see open_questions.md's
    USASPENDING gate-2 final-run entry) and its readiness_trigger was
    removed from the live ladder (an already-resolved trigger would make
    check_all() report it "ready" forever, a staleness bug in the tool
    itself) — the two remaining live-ladder integration tests below were
    repointed at cftc_cot_positioning (still gate2_pending) to keep the
    same not-ready-today / eventually-ready coverage shape; TestDateTrigger
    above still fully covers the `date`-type trigger logic itself in
    isolation."""

    def test_known_gated_roots_present_and_evaluable(self):
        results = readiness.check_all(today=date(2026, 8, 15))
        ids = {r["id"] for r in results}
        for expected in ("cftc_cot_positioning", "sec_8k_earnings_language"):
            self.assertIn(expected, ids, f"{expected} should carry a readiness_trigger in signal_ladder.json")
        self.assertNotIn("usaspending_contracts", ids, "graduated 2026-08-15 — should no longer carry a trigger")

    def test_cftc_not_yet_ready_on_aug_15(self):
        results = readiness.check_all(today=date(2026, 8, 15))
        row = next(r for r in results if r["id"] == "cftc_cot_positioning")
        self.assertFalse(row["ready"])

    def test_cftc_ready_once_enough_weeks_elapsed(self):
        # matches TestWeeklyReportsTrigger.test_ready_once_enough_weeks_elapsed's
        # own math: 15 weeks * 7 days after 2026-07-08 -> 2026-10-21
        results = readiness.check_all(today=date(2026, 10, 21))
        row = next(r for r in results if r["id"] == "cftc_cot_positioning")
        self.assertTrue(row["ready"])

    def test_gnss_integrity_adsb_present_despite_being_gate2_pass(self):
        # 2026-09-26 WIDENING: readiness_trigger now also covers a gateN_pass
        # root's stated FOLLOW-ON artifact (here, the GNSS-jamming x defense-ETF
        # correlation probe gated on this root's own daily-archive depth) — not
        # just a gateN_pending root's own re-run condition. This root's status
        # is gate2_pass, so its presence here proves check_all() doesn't filter
        # by status.
        results = readiness.check_all(today=date(2026, 9, 26))
        row = next(r for r in results if r["id"] == "gnss_integrity_adsb")
        self.assertEqual(row["status"], "gate2_pass")
        self.assertFalse(row["ready"])

    def test_gnss_integrity_adsb_ready_once_archive_deep_enough(self):
        # CORRECTED 2026-10-10: the old fixture (since=2026-09-22, min_days=15,
        # "ready 2026-10-07") encoded the bug — 15 daily records is not the
        # pre-registered bar. since=2026-08-24, min_days=128 -> ready 2026-12-30.
        early = readiness.check_all(today=date(2026, 10, 10))
        row = next(r for r in early if r["id"] == "gnss_integrity_adsb")
        self.assertFalse(row["ready"])
        late = readiness.check_all(today=date(2026, 12, 30))
        row = next(r for r in late if r["id"] == "gnss_integrity_adsb")
        self.assertTrue(row["ready"])

    def test_gnss_trigger_covers_preregistered_destrided_bar(self):
        # Pre-registered design (open_questions.md 2026-09-22): z window 10
        # trading days, horizon 5, >=15 non-overlapping pairs. Recompute the
        # trading days needed (weekdays only = a LOWER bound, holidays only
        # add days) and require the trigger's calendar span to cover it.
        from datetime import timedelta
        z_window, horizon, min_pairs = 10, 5, 15
        needed = z_window + min_pairs * horizon + horizon
        trig = readiness.load_ladder_roots()["gnss_integrity_adsb"]["readiness_trigger"] \
            if hasattr(readiness, "load_ladder_roots") else None
        if trig is None:
            import json
            with open(os.path.join(REPO_ROOT, "datacore", "signal_ladder.json")) as f:
                data = json.load(f)
            roots = data["roots"] if "roots" in data else data
            root = roots["gnss_integrity_adsb"] if isinstance(roots, dict) else \
                next(r for r in roots if r.get("id") == "gnss_integrity_adsb")
            trig = root["readiness_trigger"]
        d, n = date.fromisoformat(trig["since"]), 0
        while n < needed:
            if d.weekday() < 5:
                n += 1
            if n < needed:
                d += timedelta(days=1)
        self.assertGreaterEqual(trig["min_days"], (d - date.fromisoformat(trig["since"])).days)

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
