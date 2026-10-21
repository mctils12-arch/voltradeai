"""Regression tests for cot_gate2_oos.py — pure functions, no network."""
import unittest

from cot_gate2_oos import MIN_EPISODES, oos_power


def row(d, bucket, fr=None):
    return {"report_date": d, "bucket": bucket, "forward_returns": {20: fr} if fr is not None else {}}


class TestOosPower(unittest.TestCase):
    def test_no_extreme_weeks_is_untestable(self):
        rows = [row("2026-07-14", "mid", 0.01), row("2026-07-21", "mid", 0.02)]
        r = oos_power(rows, "2026-07-08", "extreme_high")
        self.assertEqual((r["extreme_weeks"], r["episodes"], r["verdict"]), (0, 0, "UNTESTABLE"))

    def test_consecutive_extreme_weeks_are_one_episode(self):
        rows = [row("2026-07-14", "extreme_high", 0.0), row("2026-07-21", "extreme_high", 0.0),
                row("2026-07-28", "mid"), row("2026-08-04", "extreme_high")]
        r = oos_power(rows, "2026-07-08", "extreme_high")
        self.assertEqual((r["extreme_weeks"], r["episodes"], r["matured_extreme"]), (3, 2, 2))

    def test_in_sample_weeks_excluded_boundary_strict(self):
        rows = [row("2026-07-08", "extreme_high", 0.1), row("2026-07-14", "mid", 0.0)]
        self.assertEqual(oos_power(rows, "2026-07-08", "extreme_high")["extreme_weeks"], 0)

    def test_enough_separate_episodes_is_testable(self):
        rows = []
        for i in range(MIN_EPISODES):
            rows += [row(f"2026-08-{i * 2 + 1:02d}", "extreme_high", 0.0), row(f"2026-08-{i * 2 + 2:02d}", "mid")]
        self.assertEqual(oos_power(rows, "2026-07-08", "extreme_high")["verdict"], "TESTABLE")


if __name__ == "__main__":
    unittest.main()
