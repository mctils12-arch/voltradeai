"""
Regression tests for scripts/midas_gate2.py — the ROOT VALIDATION LADDER
gate 2 (SIGNAL) screen for the MIDAS HFT-colonization filter hypothesis.
Pure-function tests only: no network calls (no live diag probe, no
backtest_v2 Yahoo/Alpaca fetch) — `fetch_bars_fn`/`find_entry_index_fn`
are always fakes here.
"""
import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "midas_gate2",
    os.path.join(os.path.dirname(__file__), "scripts", "midas_gate2.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

entry_date_for = _mod.entry_date_for
quarter_is_ready = _mod.quarter_is_ready
stratified_sample = _mod.stratified_sample
tercile_bucket = _mod.tercile_bucket
forward_returns_from_bars = _mod.forward_returns_from_bars
welch_vs_baseline = _mod.welch_vs_baseline
run_quarter = _mod.run_quarter


def _row(ticker, mcap_rank, cancel_to_trade, hidden=10.0, odd_lot=10.0, turn_rank=5):
    return {"ticker": ticker, "mcapRank": mcap_rank, "turnRank": turn_rank,
            "n_days": 40, "cancelToTrade": cancel_to_trade,
            "hiddenRatePct": hidden, "oddLotRatePct": odd_lot}


class TestEntryDate(unittest.TestCase):
    def test_adds_lag_days(self):
        self.assertEqual(entry_date_for("2025-12-31", 56), "2026-02-25")

    def test_default_lag_matches_evidence_derived_56(self):
        # 2026q2 (end 2026-06-30) was already archived live on 2026-08-25 --
        # only possible if the real lag was <= 56 days. This pins the
        # constant to that evidence rather than letting it silently drift.
        self.assertEqual(_mod.DEFAULT_PUBLISH_LAG_DAYS, 56)
        self.assertEqual(entry_date_for("2026-06-30", _mod.DEFAULT_PUBLISH_LAG_DAYS), "2026-08-25")


class TestQuarterIsReady(unittest.TestCase):
    def test_ready_when_forward_history_exists(self):
        self.assertTrue(quarter_is_ready("2025-12-31", "2026-08-25", lag_days=56))

    def test_not_ready_when_entry_date_is_today(self):
        # 2026q2's own +56d entry lands exactly on 2026-08-25 -- zero
        # forward days available, must be excluded, not force-tested.
        self.assertFalse(quarter_is_ready("2026-06-30", "2026-08-25", lag_days=56))

    def test_not_ready_when_entry_date_is_future(self):
        self.assertFalse(quarter_is_ready("2026-07-31", "2026-08-25", lag_days=56))


class TestStratifiedSample(unittest.TestCase):
    def test_proportional_by_stratum(self):
        rows = [_row(f"T{i}", 1, 10.0) for i in range(60)] + [_row(f"U{i}", 2, 10.0) for i in range(40)]
        sample = stratified_sample(rows, 20, seed=1)
        strata = [r["mcapRank"] for r in sample]
        # 60/40 population split -> roughly proportional in a 20-sample draw
        self.assertEqual(strata.count(1), 12)
        self.assertEqual(strata.count(2), 8)

    def test_deterministic_for_fixed_seed(self):
        rows = [_row(f"T{i}", 1, float(i)) for i in range(30)]
        a = [r["ticker"] for r in stratified_sample(rows, 10, seed=42)]
        b = [r["ticker"] for r in stratified_sample(rows, 10, seed=42)]
        self.assertEqual(a, b)

    def test_different_seed_can_differ(self):
        rows = [_row(f"T{i}", 1, float(i)) for i in range(30)]
        a = [r["ticker"] for r in stratified_sample(rows, 10, seed=1)]
        b = [r["ticker"] for r in stratified_sample(rows, 10, seed=2)]
        self.assertNotEqual(a, b)


class TestTercileBucket(unittest.TestCase):
    def test_splits_within_stratum_not_across(self):
        # mcapRank=1 stratum: values 1..9 -> low={1,2,3} high={7,8,9}
        # mcapRank=2 stratum: values 100..108 -> low={100,101,102} high={106,107,108}
        # A high-cancelToTrade value in stratum 1 (e.g. 9) must never be
        # compared against stratum 2's raw values -- confirms no cross-
        # stratum contamination.
        rows = [_row(f"A{v}", 1, float(v)) for v in range(1, 10)]
        rows += [_row(f"B{v}", 2, float(v)) for v in range(100, 109)]
        buckets = tercile_bucket(rows, "cancelToTrade")
        self.assertEqual(buckets["A1"], "low")
        self.assertEqual(buckets["A9"], "high")
        self.assertEqual(buckets["B100"], "low")
        self.assertEqual(buckets["B108"], "high")

    def test_mid_bucket_excluded_from_high_low(self):
        rows = [_row(f"A{v}", 1, float(v)) for v in range(1, 10)]
        buckets = tercile_bucket(rows, "cancelToTrade")
        mids = [t for t, b in buckets.items() if b == "mid"]
        self.assertEqual(len(mids), 3)


class TestForwardReturnsFromBars(unittest.TestCase):
    def _find_entry_index(self, dates, publish_date):
        for i, d in enumerate(dates):
            if d > publish_date:
                return i
        return None

    def test_no_lookahead_entry_strictly_after_publish_date(self):
        bars = {"date": ["2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27"],
                "close": [10.0, 10.0, 11.0, 12.0]}
        fr = forward_returns_from_bars(bars, "2026-02-25", (1, 2), self._find_entry_index)
        # entry must be 2026-02-26 (strictly after 25th), not the 25th itself
        self.assertAlmostEqual(fr[1], 12.0 / 11.0 - 1)

    def test_horizon_past_end_of_bars_is_omitted_not_zero_filled(self):
        bars = {"date": ["2026-02-25", "2026-02-26"], "close": [10.0, 11.0]}
        fr = forward_returns_from_bars(bars, "2026-02-24", (1, 60), self._find_entry_index)
        self.assertIn(1, fr)
        self.assertNotIn(60, fr)

    def test_empty_bars_returns_empty(self):
        fr = forward_returns_from_bars({"date": [], "close": []}, "2026-02-24", (5,), self._find_entry_index)
        self.assertEqual(fr, {})


class TestWelchVsBaseline(unittest.TestCase):
    def test_none_below_min_n(self):
        self.assertIsNone(welch_vs_baseline([0.01, 0.02, 0.03], [0.0] * 10))

    def test_detects_real_separation(self):
        sample = [0.20, 0.22, 0.19, 0.21, 0.23, 0.18]
        baseline = [0.0, 0.01, -0.01, 0.0, 0.01, -0.01, 0.0, 0.01]
        result = welch_vs_baseline(sample, baseline)
        self.assertIsNotNone(result)
        self.assertLess(result["p_value"], 0.01)
        self.assertGreater(result["mean_diff_pct"], 0)


class TestRunQuarterIntegration(unittest.TestCase):
    """End-to-end with fully injected fakes -- proves the orchestration
    wires sampling -> bucketing -> forward returns -> significance
    correctly, with zero network I/O."""

    def test_high_bucket_shows_negative_separation_in_a_synthetic_universe(self):
        rows = []
        for i in range(30):
            # cancelToTrade rises with i; price path is DESIGNED to fall
            # for high-cancelToTrade tickers and rise for low ones, so the
            # test has a known ground-truth direction to assert against.
            rows.append(_row(f"T{i}", 1, cancel_to_trade=float(i)))

        def fake_fetch_bars(ticker, days):
            idx = int(ticker[1:])
            drift = -0.30 if idx >= 20 else (0.30 if idx < 10 else 0.0)
            base = 100.0
            dates = [f"2026-02-{20+d:02d}" if d < 8 else f"2026-03-{d-7:02d}" for d in range(30)]
            closes = [base * (1 + drift * (d / 29)) for d in range(30)]
            return {"date": dates, "close": closes}

        def fake_find_entry_index(dates, publish_date):
            for i, d in enumerate(dates):
                if d > publish_date:
                    return i
            return None

        result = run_quarter("2025q4", "2025-12-31", rows, fake_fetch_bars,
                              fake_find_entry_index, n_sample=30, seed=1,
                              lag_days=56, horizons=(20,), as_of="2026-08-25")
        high20 = result["metrics"]["cancelToTrade"]["20"]["high"]
        low20 = result["metrics"]["cancelToTrade"]["20"]["low"]
        self.assertIsNotNone(high20)
        self.assertIsNotNone(low20)
        self.assertLess(high20["mean_pct"], low20["mean_pct"])


if __name__ == "__main__":
    unittest.main()
