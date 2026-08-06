"""
Regression tests for scripts/jodi_gate2_test.py — the ROOT VALIDATION
LADDER gate 2 (SIGNAL) screen for the JODI non-OECD closing-stock root.
Pure-function tests only: no network calls, no dependency on the live
JODI archive file or backtest_v2's Alpaca/Yahoo fetch.
"""
import importlib.util
import json
import os
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location(
    "jodi_gate2_test_mod",
    os.path.join(os.path.dirname(__file__), "scripts", "jodi_gate2_test.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_composite = _mod.build_composite
build_deltas = _mod.build_deltas
compute_forward_returns = _mod.compute_forward_returns
hac_significance = _mod.hac_significance
period_end_date = _mod.period_end_date
summarize = _mod.summarize
zscore_trailing = _mod.zscore_trailing


class TestPeriodEndDate(unittest.TestCase):
    def test_mid_year(self):
        self.assertEqual(period_end_date("2026-04"), "2026-04-30")

    def test_december_wraps_year(self):
        self.assertEqual(period_end_date("2025-12"), "2025-12-31")

    def test_february_non_leap(self):
        self.assertEqual(period_end_date("2026-02"), "2026-02-28")

    def test_february_leap(self):
        self.assertEqual(period_end_date("2024-02"), "2024-02-29")


class TestBuildDeltas(unittest.TestCase):
    def test_first_period_has_no_delta(self):
        deltas = build_deltas({"2020-01": 100.0, "2020-02": 120.0, "2020-03": 90.0})
        self.assertEqual(set(deltas), {"2020-02", "2020-03"})
        self.assertAlmostEqual(deltas["2020-02"], 20.0)
        self.assertAlmostEqual(deltas["2020-03"], -30.0)

    def test_unsorted_input_still_chronological(self):
        deltas = build_deltas({"2020-03": 90.0, "2020-01": 100.0, "2020-02": 120.0})
        self.assertAlmostEqual(deltas["2020-02"], 20.0)
        self.assertAlmostEqual(deltas["2020-03"], -30.0)


class TestZscoreTrailing(unittest.TestCase):
    def test_warmup_period_produces_nothing(self):
        deltas = {f"2020-{m:02d}": float(m) for m in range(1, 13)}  # 12 points
        z = zscore_trailing(deltas, window=36)
        self.assertEqual(z, {})

    def test_zero_variance_window_returns_none_not_zero(self):
        periods = [f"{2018 + i // 12}-{i % 12 + 1:02d}" for i in range(40)]
        deltas = {p: 5.0 for p in periods}  # constant deltas -> zero variance
        z = zscore_trailing(deltas, window=36)
        self.assertTrue(z)  # some periods past warmup exist
        self.assertTrue(all(v is None for v in z.values()))

    def test_positive_deviation_from_trailing_mean_is_positive_z(self):
        periods = [f"{2018 + i // 12}-{i % 12 + 1:02d}" for i in range(40)]
        deltas = {p: 0.0 for p in periods}
        last = periods[-1]
        deltas[last] = 10.0  # a real spike relative to a flat trailing window
        # window prior to `last` is all zeros except possibly earlier spikes;
        # give the window some variance so std > 0
        deltas[periods[10]] = 3.0
        z = zscore_trailing(deltas, window=36)
        self.assertGreater(z[last], 0)


class TestBuildComposite(unittest.TestCase):
    def _write_jodi(self, tmpdir, series):
        path = os.path.join(tmpdir, "primary_stocks.json")
        with open(path, "w") as f:
            json.dump({"series": series}, f)
        return path

    def test_requires_minimum_countries_present(self):
        # 40 months of gently varying levels for 4 countries; only 2 report
        # in the final month -> below MIN_COUNTRIES_FOR_COMPOSITE (3), so
        # that month must be excluded even though data exists.
        periods = [f"{2018 + i // 12}-{i % 12 + 1:02d}" for i in range(40)]
        series = {}
        for idx, c in enumerate(["SA", "NG", "DZ", "BN"]):
            pts = [[p, 1000.0 + idx * 10 + (i % 5)] for i, p in enumerate(periods)]
            if c in ("SA", "NG"):  # only these two report the final month
                pass
            else:
                pts = pts[:-1]
            series[f"{c}|TOTCRUDE"] = {"points": pts}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_jodi(tmpdir, series)
            composite = build_composite(jodi_path=path, window=12, min_countries=3)
            self.assertNotIn(periods[-1], composite)
            # an earlier month where all 4 report should be present
            self.assertIn(periods[20], composite)


class TestComputeForwardReturns(unittest.TestCase):
    def test_left_censoring_guard_drops_pre_ipo_period(self):
        composite = {
            "2009-01": {"z": 1.0, "n_countries": 4},   # publish ~2009-03, before bars start
            "2015-01": {"z": -1.0, "n_countries": 4},  # publish ~2015-03, well after bars start
        }
        bars = {
            "date": ["2010-06-02"] + [f"2015-{m:02d}-15" for m in range(1, 13)] * 6,
            "close": [10.0] + [10.0 + i * 0.1 for i in range(72)],
        }
        rows = compute_forward_returns(composite, bars, publish_lag_days=60, horizons=(20,))
        periods_seen = {r["period"] for r in rows}
        self.assertNotIn("2009-01", periods_seen)
        self.assertIn("2015-01", periods_seen)

    def test_right_censoring_drops_horizon_not_whole_row(self):
        composite = {"2020-01": {"z": 1.0, "n_countries": 4}}
        # publish_date for 2020-01 (period-end 2020-01-31 + 60d) = 2020-03-31;
        # bars start well before that so the LEFT-censoring guard doesn't
        # fire, but only 5 bars exist after publish date -> can't reach a
        # 20-bar-ahead exit (RIGHT-censoring of the horizon only).
        bars = {
            "date": ["2020-01-01", "2020-02-01", "2020-03-01",
                     "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-06", "2020-04-07"],
            "close": [9.0, 9.5, 9.8, 10.0, 10.1, 10.2, 10.3, 10.4],
        }
        rows = compute_forward_returns(composite, bars, publish_lag_days=60, horizons=(20,))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["forward_returns"], {})

    def test_bucket_sign_matches_z(self):
        composite = {
            "2020-01": {"z": 1.5, "n_countries": 4},
            "2020-02": {"z": -0.5, "n_countries": 4},
        }
        # bars span well before both periods' publish dates (2020-03-31 and
        # 2020-04-29) through year end, so neither is left-censored.
        bars = {
            "date": [f"2020-{m:02d}-{d:02d}" for m in range(1, 13) for d in (1, 15)],
            "close": [100.0 + i for i in range(24)],
        }
        rows = compute_forward_returns(composite, bars, publish_lag_days=60, horizons=(20,))
        buckets = {r["period"]: r["bucket"] for r in rows}
        self.assertEqual(buckets["2020-01"], "build")
        self.assertEqual(buckets["2020-02"], "draw")


class TestSummarizeAndHac(unittest.TestCase):
    def _rows(self):
        # 30 synthetic monthly rows, build bucket systematically lower
        # forward return than draw -> a detectable negative diff.
        rows = []
        for i in range(30):
            bucket = "build" if i % 2 == 0 else "draw"
            fr = -0.01 if bucket == "build" else 0.01
            rows.append({
                "period": f"2020-{(i % 12) + 1:02d}",
                "bucket": bucket,
                "forward_returns": {20: fr + (0.0001 * (i % 3))},
            })
        return rows

    def test_summarize_counts_and_means(self):
        summary = summarize(self._rows(), horizons=(20,))
        self.assertEqual(summary["20"]["build"]["n"], 15)
        self.assertEqual(summary["20"]["draw"]["n"], 15)
        self.assertLess(summary["20"]["build"]["mean_pct"], summary["20"]["draw"]["mean_pct"])

    def test_hac_significance_returns_negative_diff(self):
        sig = hac_significance(self._rows(), horizons=(20,))
        result = sig["20"]
        self.assertIsNotNone(result)
        self.assertLess(result["mean_diff_pct"], 0)


if __name__ == "__main__":
    unittest.main()
