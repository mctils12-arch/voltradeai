"""
Regression tests for regime_detector_compare.py — the OPEN RESEARCH
QUESTIONS screen comparing markov_regime.py vs a VXX-ratio-style heuristic
as forward-volatility predictors. Pure-function tests only: no network
calls, no dependency on FRED being reachable (mirrors the existing
cot_gate2_test.py / test_cot_gate2.py convention in this repo).
"""
import math
import unittest

from regime_detector_compare import (
    align_series,
    compare_detectors,
    compute_features,
    parse_fred_csv_text,
    _clean_pairs,
)


class TestParseFredCsvText(unittest.TestCase):
    def test_parses_valid_rows(self):
        text = "observation_date,SP500\n2026-01-02,100.0\n2026-01-05,101.5\n"
        self.assertEqual(parse_fred_csv_text(text), {"2026-01-02": 100.0, "2026-01-05": 101.5})

    def test_skips_missing_value_marker(self):
        # FRED uses a literal "." for holidays/no-observation rows — must
        # not be parsed as a float or silently coerced to 0.
        text = "observation_date,VIXCLS\n2026-01-02,15.0\n2026-01-03,.\n2026-01-04,16.2\n"
        result = parse_fred_csv_text(text)
        self.assertEqual(result, {"2026-01-02": 15.0, "2026-01-04": 16.2})
        self.assertNotIn("2026-01-03", result)

    def test_skips_malformed_rows(self):
        text = "observation_date,SP500\n2026-01-02,100.0\nnot,a,valid,row\n2026-01-05,101.0\n"
        result = parse_fred_csv_text(text)
        self.assertEqual(result, {"2026-01-02": 100.0, "2026-01-05": 101.0})

    def test_empty_body_returns_empty_dict(self):
        self.assertEqual(parse_fred_csv_text("observation_date,SP500\n"), {})


class TestAlignSeries(unittest.TestCase):
    def test_inner_join_sorted_ascending(self):
        spy = {"2026-01-03": 100.0, "2026-01-02": 99.0, "2026-01-05": 102.0}
        vix = {"2026-01-02": 15.0, "2026-01-03": 16.0, "2026-01-04": 17.0}
        dates, spy_v, vix_v = align_series(spy, vix)
        # 2026-01-04 (spy missing) and 2026-01-05 (vix missing) must be dropped
        self.assertEqual(dates, ["2026-01-02", "2026-01-03"])
        self.assertEqual(spy_v, [99.0, 100.0])
        self.assertEqual(vix_v, [15.0, 16.0])

    def test_no_overlap_returns_empty(self):
        dates, spy_v, vix_v = align_series({"2026-01-02": 1.0}, {"2026-02-02": 1.0})
        self.assertEqual((dates, spy_v, vix_v), ([], [], []))


class TestComputeFeatures(unittest.TestCase):
    def _flat_then_spike(self, n=100, spike_at=80, spike_mag=5.0):
        """Deterministic synthetic series: SPY flat at 100 with a tiny
        daily wiggle (so returns aren't all exactly zero, which would
        make MarkovRegime's threshold adaptation degenerate), VIX flat at
        15, except a forward volatility spike injected starting at
        `spike_at` (so fwd_vol features at the day BEFORE the spike should
        be elevated once the spike falls inside their forward window)."""
        spy = [100.0 + (0.05 if i % 2 == 0 else -0.05) for i in range(n)]
        for i in range(spike_at, min(spike_at + 5, n)):
            spy[i] = spy[i - 1] * (1.03 if i % 2 == 0 else 0.97)  # inject real vol
        vix = [15.0 + (i % 3) * 0.01 for i in range(n)]
        dates = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]
        return dates, spy, vix

    def test_drops_warmup_and_tail_rows(self):
        dates, spy, vix = self._flat_then_spike(n=100)
        rows = compute_features(dates, spy, vix)
        # warmup = max(50, MARKOV_LOOKBACK)=50, tail=20 -> valid range [50, 79]
        self.assertEqual(len(rows), 100 - 50 - 20)
        self.assertEqual(rows[0]["date"], dates[50])
        self.assertEqual(rows[-1]["date"], dates[79])

    def test_forward_vol_elevated_ahead_of_injected_spike(self):
        dates, spy, vix = self._flat_then_spike(n=100, spike_at=80)
        rows = compute_features(dates, spy, vix)
        by_date = {r["date"]: r for r in rows}
        # day index 75: forward 5-day window (76..80) touches the spike start
        # day index 60: forward 5-day window (61..65) is pure flat wiggle
        vol_near_spike = by_date[dates[75]]["fwd_vol_5"]
        vol_far_from_spike = by_date[dates[60]]["fwd_vol_5"]
        self.assertGreater(vol_near_spike, vol_far_from_spike)

    def test_markov_probs_are_valid_distribution(self):
        dates, spy, vix = self._flat_then_spike(n=100)
        rows = compute_features(dates, spy, vix)
        for r in rows:
            self.assertGreaterEqual(r["markov_p_bear"], 0.0)
            self.assertLessEqual(r["markov_p_bear"], 1.0)
            self.assertIn(r["markov_signal"], {"STRONG_BUY", "BUY", "NEUTRAL", "CAUTION", "SELL"})

    def test_vix_ratio_and_spy_vs_ma50_formulas(self):
        # constant series -> both ratios must be exactly 1.0 (no rounding
        # surprises, no accidental off-by-one in the trailing window slice)
        n = 90
        dates = [f"day{i}" for i in range(n)]
        spy = [100.0] * n
        vix = [20.0] * n
        rows = compute_features(dates, spy, vix)
        for r in rows:
            self.assertAlmostEqual(r["vix_ratio"], 1.0, places=9)
            self.assertAlmostEqual(r["spy_vs_ma50"], 1.0, places=9)


class TestCleanPairs(unittest.TestCase):
    def test_drops_none_and_nan(self):
        xa, ya = _clean_pairs([1.0, None, 3.0, float("nan")], [10.0, 20.0, None, 40.0])
        self.assertEqual(xa, [1.0])
        self.assertEqual(ya, [10.0])

    def test_empty_input(self):
        self.assertEqual(_clean_pairs([], []), ([], []))


class TestCompareDetectors(unittest.TestCase):
    def _rows(self, signal_vals, fwd5_vals, fwd20_vals=None):
        fwd20_vals = fwd20_vals or fwd5_vals
        return [
            {
                "date": f"day{i}",
                "vix_ratio": signal_vals[i],
                "spy_vs_ma50": -signal_vals[i],  # inverse, for a sign sanity check
                "markov_p_bear": 0.2,  # constant -> correlation undefined-ish but must not crash
                "fwd_vol_5": fwd5_vals[i],
                "fwd_vol_20": fwd20_vals[i],
            }
            for i in range(len(signal_vals))
        ]

    def test_perfect_monotonic_relationship_gives_rho_near_one(self):
        signal = list(range(30))
        fwd = list(range(30))  # perfectly increasing together
        rows = self._rows(signal, fwd)
        result = compare_detectors(rows)
        self.assertAlmostEqual(result["vix_ratio"]["fwd_vol_5"]["rho"], 1.0, places=6)
        self.assertAlmostEqual(result["spy_vs_ma50"]["fwd_vol_5"]["rho"], -1.0, places=6)

    def test_n_and_base_rate_reported(self):
        signal = list(range(15))
        fwd = [float(v) for v in range(15)]
        rows = self._rows(signal, fwd)
        result = compare_detectors(rows)
        self.assertEqual(result["n"], 15)
        self.assertAlmostEqual(result["fwd_vol_5_base_mean"], sum(fwd) / len(fwd), places=6)

    def test_below_minimum_n_reports_none_rho_not_a_crash(self):
        signal = list(range(5))
        fwd = [float(v) for v in range(5)]
        rows = self._rows(signal, fwd)
        result = compare_detectors(rows)
        self.assertIsNone(result["vix_ratio"]["fwd_vol_5"]["rho"])
        self.assertEqual(result["vix_ratio"]["fwd_vol_5"]["n"], 5)


if __name__ == "__main__":
    unittest.main()
