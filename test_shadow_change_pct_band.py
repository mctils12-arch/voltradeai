#!/usr/bin/env python3
"""
Regression tests for shadow_portfolio._change_pct_band_stats() /
get_shadow_stats()["win_rate_by_change_pct_band"].

WHY THIS EXISTS (research/open_questions.md KNOWN BROKEN #10): system_config.py's
MAX_CHANGE_PCT ("Skip stocks already up/down 35%+") is defined with a comment
claiming it's a live backtest-validated gate but is read nowhere in the actual
trading path — a dead key that reads as an active guardrail to anyone
consulting system_config.py. Item #10's own NEXT STEP, blocked for months on
shadow_portfolio's backfill labeling bug (fixed 2026-08-05, v1.0.596), was to
query the shadow archive directly: win rate for |change_pct_today| > threshold
"taken" candidates vs. the rest. Now that labeled history exists, this is that
query, exposed through the same get_shadow_stats()/diag "shadow" probe path
already built to answer it (see diag.ts's DIAG_PROBES "shadow" entry).
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import shadow_portfolio


def _rec(decision, change_pct, labels):
    """labels: dict like {"+5d": 1, "+10d": 0} — omitted horizons stay unlabeled."""
    outcomes = {"+5d": None, "+10d": None, "+20d": None}
    for h, lbl in labels.items():
        outcomes[h] = {"label": lbl}
    return {
        "ticker": "TEST", "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": 70.0, "decision": decision, "decision_reason": "",
        "entry_price": 100.0, "vxx_ratio": 1.0, "regime_label": "NEUTRAL",
        "features": {"change_pct_today": change_pct},
        "outcomes": outcomes, "code_version": "test-band",
    }


class TestChangePctBandStats(unittest.TestCase):
    def test_splits_by_live_threshold_from_system_config(self):
        from system_config import BASE_CONFIG
        threshold = float(BASE_CONFIG["MAX_CHANGE_PCT"])
        records = [
            _rec("taken", threshold + 5, {"+5d": 1}),
            _rec("taken", threshold - 5, {"+5d": 0}),
        ]
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=1)
        self.assertEqual(stats["threshold"], threshold)
        self.assertEqual(stats["by_band"]["over_threshold"]["candidate_count"], 1)
        self.assertEqual(stats["by_band"]["at_or_under_threshold"]["candidate_count"], 1)

    def test_boundary_is_at_or_under_not_over(self):
        # exactly at threshold must land in "at_or_under_threshold", matching
        # the ">" comparison the dead MAX_CHANGE_PCT config comment implies
        # ("skip stocks already up/down 35%+" reads as >=, but the query
        # itself only needs to be consistent and documented — pinned here so
        # a future edit can't silently flip the boundary without a test change).
        records = [_rec("taken", 35.0, {"+5d": 1})]
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=1)
        self.assertEqual(stats["by_band"]["at_or_under_threshold"]["candidate_count"], 1)
        self.assertEqual(stats["by_band"]["over_threshold"]["candidate_count"], 0)

    def test_negative_change_pct_uses_absolute_value(self):
        # a -40% candidate is just as "already moved" as a +40% one
        records = [_rec("taken", -40.0, {"+5d": 1})]
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=1)
        self.assertEqual(stats["by_band"]["over_threshold"]["candidate_count"], 1)

    def test_only_taken_records_count(self):
        # rejected/other decisions never got a real outcome under the bot's
        # actual exit rules — including them would corrupt the comparison
        records = [
            _rec("taken", 50.0, {"+5d": 1}),
            _rec("rejected_score", 50.0, {"+5d": 1}),
            _rec("rejected_heat", 50.0, {"+5d": 0}),
        ]
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=1)
        self.assertEqual(stats["by_band"]["over_threshold"]["candidate_count"], 1)

    def test_win_rate_omitted_below_min_n(self):
        records = [_rec("taken", 50.0, {"+5d": 1}), _rec("taken", 60.0, {"+5d": 0})]
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=5)
        self.assertEqual(stats["by_band"]["over_threshold"]["candidate_count"], 2)
        self.assertNotIn("+5d", stats["by_band"]["over_threshold"]["win_rate"])

    def test_win_rate_present_at_or_above_min_n(self):
        records = [_rec("taken", 50.0, {"+5d": 1 if i % 2 == 0 else 0}) for i in range(6)]
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=5)
        self.assertIn("+5d", stats["by_band"]["over_threshold"]["win_rate"])
        self.assertEqual(stats["by_band"]["over_threshold"]["win_rate"]["+5d"]["n"], 6)
        self.assertAlmostEqual(stats["by_band"]["over_threshold"]["win_rate"]["+5d"]["win_rate"], 50.0)

    def test_missing_change_pct_feature_defaults_to_zero(self):
        rec = _rec("taken", 0.0, {"+5d": 1})
        del rec["features"]["change_pct_today"]
        stats = shadow_portfolio._change_pct_band_stats([rec], min_n=1)
        self.assertEqual(stats["by_band"]["at_or_under_threshold"]["candidate_count"], 1)

    def test_distribution_empty_when_no_taken_records(self):
        records = [_rec("rejected_score", 50.0, {"+5d": 1})]
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=1)
        dist = stats["taken_distribution"]
        self.assertEqual(dist["n"], 0)
        self.assertIsNone(dist["max"])
        self.assertEqual(dist["count_over_20"], 0)
        self.assertEqual(dist["count_over_30"], 0)

    def test_distribution_reports_max_and_percentiles(self):
        records = [_rec("taken", v, {"+5d": 1}) for v in [1.0, 5.0, 12.0, 18.0, 25.0]]
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=1)
        dist = stats["taken_distribution"]
        self.assertEqual(dist["n"], 5)
        self.assertEqual(dist["max"], 25.0)
        self.assertEqual(dist["p50"], 12.0)

    def test_distribution_uses_absolute_value(self):
        records = [_rec("taken", -40.0, {"+5d": 1}), _rec("taken", 10.0, {"+5d": 1})]
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=1)
        dist = stats["taken_distribution"]
        self.assertEqual(dist["max"], 40.0)
        self.assertEqual(dist["count_over_30"], 1)

    def test_distribution_count_over_thresholds(self):
        records = ([_rec("taken", 15.0, {"+5d": 1})] * 3
                   + [_rec("taken", 25.0, {"+5d": 1})] * 2
                   + [_rec("taken", 35.0, {"+5d": 1})])
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=1)
        dist = stats["taken_distribution"]
        self.assertEqual(dist["n"], 6)
        self.assertEqual(dist["count_over_20"], 3)  # the two 25s + the 35
        self.assertEqual(dist["count_over_30"], 1)  # only the 35

    def test_distribution_only_counts_taken_records(self):
        records = [_rec("taken", 5.0, {"+5d": 1}), _rec("rejected_score", 45.0, {"+5d": 1})]
        stats = shadow_portfolio._change_pct_band_stats(records, min_n=1)
        self.assertEqual(stats["taken_distribution"]["n"], 1)
        self.assertEqual(stats["taken_distribution"]["max"], 5.0)


class TestGetShadowStatsWiring(unittest.TestCase):
    """Confirms get_shadow_stats() (the function the diag "shadow" probe
    calls verbatim, per server/diag.test.ts's pinned wiring test) actually
    surfaces the band stats end to end, using the real file-backed log."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_log_path = shadow_portfolio.SHADOW_LOG_PATH
        self._orig_lock_path = shadow_portfolio.SHADOW_LOCK_PATH
        shadow_portfolio.SHADOW_LOG_PATH = os.path.join(self._tmpdir.name, "shadow.json")
        shadow_portfolio.SHADOW_LOCK_PATH = shadow_portfolio.SHADOW_LOG_PATH + ".lock"

    def tearDown(self):
        shadow_portfolio.SHADOW_LOG_PATH = self._orig_log_path
        shadow_portfolio.SHADOW_LOCK_PATH = self._orig_lock_path
        self._tmpdir.cleanup()

    def test_key_present_and_shaped(self):
        records = [_rec("taken", 50.0, {"+5d": 1})]
        shadow_portfolio._save_shadow_log(records)
        stats = shadow_portfolio.get_shadow_stats()
        self.assertIn("win_rate_by_change_pct_band", stats)
        band = stats["win_rate_by_change_pct_band"]
        self.assertIn("threshold", band)
        self.assertIn("over_threshold", band["by_band"])
        self.assertIn("at_or_under_threshold", band["by_band"])
        self.assertIn("taken_distribution", band)
        self.assertEqual(band["taken_distribution"]["n"], 1)
        self.assertEqual(band["taken_distribution"]["max"], 50.0)


if __name__ == "__main__":
    unittest.main()
