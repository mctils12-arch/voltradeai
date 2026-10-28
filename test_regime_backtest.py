"""
scripts/regime_backtest.py — the index-level regime-layer tool (Stage B step 1
of docs/regime/FINDINGS.md). Synthetic series only; nothing here touches the
network or the bar cache. The arithmetic pinned here is what RESEARCH.md §5's
numbers rest on: lag semantics, turnover cost, drawdown, flips, the monthly
Faber rule, the capped vol target, the current-schedule proxy's dwell, and the
±20% reversal detector.
"""
import math
import os
import sys
import unittest
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import regime_backtest as rb  # noqa: E402


def bdays(start="2020-01-01", n=10):
    d = date.fromisoformat(start)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


class TestEvaluate(unittest.TestCase):
    def test_constant_growth_cagr_no_drawdown(self):
        n = 505
        dates = bdays(n=n)
        closes = [100.0 * (1.001 ** i) for i in range(n)]
        r = rb.evaluate(dates, closes, [1.0] * n, lag=1, cost_bp=0.0)
        self.assertEqual(r["days"], n - 1)
        self.assertAlmostEqual(r["cagr"], 1.001 ** 252 - 1, places=6)
        self.assertEqual(r["mdd"], 0.0)
        self.assertEqual(r["flips_yr"], 0.0)
        self.assertEqual(r["tim"], 1.0)

    def test_lag_semantics_exposure_decided_at_close_i_applies_to_return_i_plus_lag(self):
        n = 12
        dates = bdays(n=n)
        closes = [100.0] * n
        j = 7
        closes[j] = 110.0                      # one +10% return, closes[j-1] -> closes[j]
        for k in range(j + 1, n):
            closes[k] = 110.0
        expo = [0.0] * n
        expo[j - 2] = 1.0                      # lag 2: decided at close j-2, applied to return j
        r = rb.evaluate(dates, closes, expo, lag=2, cost_bp=0.0)
        self.assertAlmostEqual((1 + r["cagr"]) ** (r["days"] / 252.0), 1.10, places=9)
        expo = [0.0] * n
        expo[j - 1] = 1.0                      # one session late: misses the jump
        r = rb.evaluate(dates, closes, expo, lag=2, cost_bp=0.0)
        self.assertAlmostEqual((1 + r["cagr"]) ** (r["days"] / 252.0), 1.0, places=9)

    def test_turnover_cost_charged_per_unit_moved(self):
        n = 8
        dates = bdays(n=n)
        closes = [100.0] * n                   # flat market: only costs move equity
        expo = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # two units of turnover
        r = rb.evaluate(dates, closes, expo, lag=1, cost_bp=50.0)
        growth = (1 + r["cagr"]) ** (r["days"] / 252.0)
        self.assertAlmostEqual(growth, (1 - 0.005) * (1 - 0.005), places=9)
        self.assertAlmostEqual(r["turnover_yr"] * (r["days"] / 252.0), 2.0, places=9)

    def test_max_drawdown_from_peak(self):
        dates = bdays(n=5)
        closes = [100.0, 110.0, 99.0, 120.0, 120.0]
        r = rb.evaluate(dates, closes, [1.0] * 5, lag=1, cost_bp=0.0)
        self.assertAlmostEqual(r["mdd"], 99.0 / 110.0 - 1.0, places=9)

    def test_flips_and_time_in_market_count_exposure_levels(self):
        dates = bdays(n=7)
        closes = [100.0] * 7
        expo = [1.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.5]
        r = rb.evaluate(dates, closes, expo, lag=1, cost_bp=0.0)
        # evaluated days: expo[0..5] (lag 1 -> last decision at i=5); levels 1,1,0,0,1,.5 -> 3 changes
        self.assertEqual(r["days"], 6)
        self.assertAlmostEqual(r["flips_yr"] * (6 / 252.0), 3.0, places=9)
        self.assertAlmostEqual(r["tim"], 4 / 6, places=9)

    def test_window_bounds_and_input_validation(self):
        dates = bdays(n=6)
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        r = rb.evaluate(dates, closes, [1.0] * 6, lag=1, cost_bp=0.0, start=dates[3], end=dates[4])
        self.assertEqual(r["days"], 2)
        with self.assertRaises(ValueError):
            rb.evaluate(dates, closes, [1.0] * 5, lag=1)
        with self.assertRaises(ValueError):
            rb.evaluate(dates, closes, [1.0] * 6, lag=0)

    def test_exposure_is_clamped_to_unit_interval(self):
        dates = bdays(n=4)
        closes = [100.0, 110.0, 121.0, 133.1]
        r = rb.evaluate(dates, closes, [3.0, 3.0, 3.0, 3.0], lag=1, cost_bp=0.0)
        self.assertAlmostEqual((1 + r["cagr"]) ** (r["days"] / 252.0), 1.1 ** 3, places=9)


class TestArms(unittest.TestCase):
    def test_sma200_daily_is_one_before_history_and_follows_the_average(self):
        n = 260
        dates = bdays(n=n)
        closes = [100.0] * 200 + [90.0] * 60      # drops below its 200-day average after day 200
        e = rb.arm_sma200_daily(dates, closes)
        self.assertEqual(e[:199], [1.0] * 199)    # no 200-day history yet -> invested
        self.assertEqual(e[199], 0.0)             # exactly AT the average is not above it (strict >, as macro_data)
        self.assertTrue(all(v == 0.0 for v in e[205:]))
        rising = [100.0 * (1.001 ** i) for i in range(260)]
        self.assertTrue(all(v == 1.0 for v in rb.arm_sma200_daily(dates, rising)))

    def test_faber_monthly_only_changes_state_at_month_end(self):
        dates = bdays("2023-01-02", 300)
        n = len(dates)
        closes = [100.0 * (1.0002 ** i) for i in range(n)]   # gently rising: price > its 10-month SMA
        flags = rb.month_end_flags(dates)
        # find the first month end that has >= 210 sessions of history, then crash the price mid-month after it
        first_ok = next(i for i, f in enumerate(flags) if f and i >= 209)
        crash = first_ok + 5
        for k in range(crash, n):
            closes[k] = 50.0
        e = rb.arm_faber_monthly(dates, closes, months=10)
        self.assertEqual(e[crash], 1.0)                      # mid-month: state held
        next_me = next(i for i in range(crash, n) if flags[i])
        self.assertEqual(e[next_me - 1], 1.0)
        self.assertEqual(e[next_me], 0.0)                    # decided at the month end
        self.assertTrue(all(v == 0.0 for v in e[next_me:]))

    def test_month_end_flags_mark_last_session_of_each_month(self):
        dates = ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"]
        self.assertEqual(rb.month_end_flags(dates), [False, True, False, True, True])

    def test_vol_target_caps_at_one_in_calm_markets_and_cuts_in_wild_ones(self):
        n = 60
        dates = bdays(n=n)
        calm = [100.0 * (1.0001 ** i) for i in range(n)]
        e = rb.arm_vol_target(dates, calm, target=0.15, window=20)
        self.assertEqual(e[-1], 1.0)
        wild = [100.0 * ((1.05 if i % 2 else 0.95) ** (i // 2 + 1)) for i in range(n)]
        e = rb.arm_vol_target(dates, wild, target=0.15, window=20, floor=0.3)
        self.assertLess(e[-1], 1.0)
        self.assertGreaterEqual(e[-1], 0.3)
        rv = rb.realized_vol(wild, n - 1, 20)
        self.assertAlmostEqual(e[-1], max(0.3, min(1.0, 0.15 / rv)), places=9)

    def test_faber_x_vol_target_respects_floor_and_multiplies(self):
        n = 300
        dates = bdays("2023-01-02", n)
        closes = [100.0] * n
        e = rb.arm_faber_x_vol_target(dates, closes, floor=0.5, target=0.15)
        self.assertTrue(all(0.5 <= v <= 1.0 for v in e))


class TestCurrentScheduleProxy(unittest.TestCase):
    def test_escalation_is_immediate_and_calming_needs_three_readings(self):
        seq = ["BULL", "BULL", "CAUTION", "BULL", "BULL", "BULL", "BULL"]
        calls = iter(seq)

        def classify(vr, spy_vs_ma50, below, above):
            return next(calls)

        n = len(seq)
        dates = bdays(n=n)
        closes = [100.0] * n
        expo, labels = rb.current_schedule_proxy(dates, closes, {}, classify=classify)
        self.assertEqual(labels, ["BULL", "BULL", "CAUTION", "CAUTION", "CAUTION", "BULL", "BULL"])
        self.assertEqual(expo[2], 0.35 / 0.70)
        self.assertEqual(expo[5], 1.0)

    def test_schedule_is_normalised_to_bull(self):
        seq = ["NEUTRAL"] * 3
        calls = iter(seq)
        expo, labels = rb.current_schedule_proxy(bdays(n=3), [100.0] * 3, {}, classify=lambda *a: next(calls))
        self.assertAlmostEqual(expo[0], 0.90 / 0.70, places=9)

    def test_proxy_feeds_the_live_classifier_the_vix_ratio_and_trend_inputs(self):
        seen = []

        def classify(vr, spy_vs_ma50, below, above):
            seen.append((round(vr, 4), round(spy_vs_ma50, 4), below, above))
            return "BULL"

        n = 205
        dates = bdays(n=n)
        closes = [100.0] * 199 + [101.0] + [90.0] * 5   # above its 200-day on day 199, then five closes below
        vix = {d: 20.0 for d in dates}
        vix[dates[-1]] = 40.0
        rb.current_schedule_proxy(dates, closes, vix, classify=classify)
        vr_last, ma_last, below_last, above_last = seen[-1]
        self.assertAlmostEqual(vr_last, 40.0 / ((29 * 20.0 + 40.0) / 30), places=4)
        self.assertLess(ma_last, 1.0)
        self.assertFalse(above_last)
        self.assertEqual(below_last, 5)


class TestSweepMachinery(unittest.TestCase):
    def test_perturbations_cover_each_numeric_parameter_both_ways(self):
        p = {"target": 0.15, "window": 20, "floor": 0.0, "label": "x"}
        variants = rb.perturbations(p, 0.20)
        self.assertEqual(len(variants), 6)
        tags = [t for t, _ in variants]
        self.assertIn("target+20%", tags)
        self.assertEqual(dict(variants)["window-20%"]["window"], 16)
        self.assertEqual(dict(variants)["floor-20%"]["floor"], 0.0)

    def test_reversal_detector_is_silent_on_identical_orderings_and_catches_a_swap(self):
        base = {"a": {"cagr": 0.10, "mdd": -0.2, "sharpe": 0.9}, "b": {"cagr": 0.08, "mdd": -0.3, "sharpe": 0.7}}
        self.assertEqual(rb.ordering_reversals(base, base), [])
        swapped = {"a": {"cagr": 0.07, "mdd": -0.2, "sharpe": 0.9}, "b": {"cagr": 0.08, "mdd": -0.3, "sharpe": 0.7}}
        rev = rb.ordering_reversals(base, swapped)
        self.assertEqual(rev, [("cagr", "a", "b")])

    def test_aligned_forward_fills_onto_the_anchor_dates(self):
        series = {"SPY": {"2024-01-02": 1.0, "2024-01-03": 2.0, "2024-01-04": 3.0},
                  "X": {"2024-01-02": 10.0, "2024-01-04": 30.0}}
        dates, closes = rb.aligned(series, "SPY")
        self.assertEqual(dates, ["2024-01-02", "2024-01-03", "2024-01-04"])
        self.assertEqual(closes["X"], [10.0, 10.0, 30.0])

    def test_run_arms_and_render_produce_the_tables(self):
        n = 320
        dates = bdays("2023-01-02", n)
        closes = {"SPY": [100.0 * (1.0005 ** i) for i in range(n)], "IWM": [50.0 * (1.0003 ** i) for i in range(n)]}
        arms = {"buy_and_hold": (rb.arm_buy_and_hold, {}), "vol_target_15": (rb.arm_vol_target, {"target": 0.15, "window": 20})}
        results, levels = rb.run_arms(dates, closes, arms, ["SPY", "IWM"], 2, 5.0, dates[0], dates[-1])
        self.assertIn("FULL", results["buy_and_hold"]["SPY"])
        self.assertIn("1.00", levels["buy_and_hold"])
        text = rb.render(results, levels, {"run_date": "2026-10-28", "symbols": ["SPY", "IWM"], "start": dates[0],
                                           "end": dates[-1], "lag": 2, "cost_bp": 5.0})
        self.assertIn("| buy_and_hold |", text)
        self.assertIn("NOT a strategy backtest", text)
        rows, rev, broken = rb.sweep(dates, closes, arms, ["SPY", "IWM"], 2, 5.0, dates[0], dates[-1])
        self.assertTrue(any(r["arm"] == "vol_target_15" for r in rows))
        self.assertTrue(all(isinstance(r["reversals"], int) for r in rows))
        self.assertTrue(all("claims_ok" in r for r in rows))
        self.assertIsInstance(broken, list)

    def test_pre_registered_claims_are_checked_literally(self):
        full = {"buy_and_hold": {"cagr": 0.11, "mdd": -0.55, "sharpe": 0.6},
                "vol_target_20": {"cagr": 0.112, "mdd": -0.40, "sharpe": 0.76},
                "faber_x_vt15_floor0": {"cagr": 0.07, "mdd": -0.24, "sharpe": 0.65},
                "faber_x_vt15_floor50": {"cagr": 0.086, "mdd": -0.26, "sharpe": 0.75},
                "current_proxy": {"cagr": 0.043, "mdd": -0.20, "sharpe": 0.56}}
        claims = rb.claims_hold(full)
        self.assertTrue(all(claims.values()), claims)
        full["current_proxy"]["cagr"] = 0.075            # no longer the lowest
        self.assertFalse(rb.claims_hold(full)["current_schedule_proxy_has_lowest_cagr"])
        full["faber_x_vt15_floor0"]["cagr"] = 0.12        # a trend-gated arm beating buy-and-hold
        self.assertFalse(rb.claims_hold(full)["trend_gated_arms_trail_buy_and_hold_cagr"])
        full["vol_target_20"]["cagr"] = 0.13              # the loose dial is no longer CAGR-neutral
        self.assertFalse(rb.claims_hold(full)["loose_vol_dial_is_cagr_neutral_within_1pp"])
        full["vol_target_20"]["mdd"] = -0.60              # a gated arm with a deeper drawdown than buy-and-hold
        self.assertFalse(rb.claims_hold(full)["gated_arms_cut_max_drawdown"])
        self.assertTrue(rb.claims_hold({"buy_and_hold": full["buy_and_hold"]})["never_fully_out_beats_fully_out_cagr"])


if __name__ == "__main__":
    unittest.main()
