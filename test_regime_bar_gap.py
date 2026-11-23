"""
KNOWN BROKEN #59 (2026-11-23): the regime engine read a bar history with a hole
in it as consecutive sessions. Production's SPY series from the IEX feed was
2020-07-27 onward plus ONE bar on 2018-11-01 (diag probe bars_coverage: 1,591
bars, first 2018-11-01; AAPL and IWM 1,590 from 2020-07-27). The 21-month step
entered the EWMA volatility as one day's return and sat in the percentile
history, so the engine's volatility percentile read about 0.09 low on average:
0.73 (NORMAL) on 2026-11-20 where the whole history reads 0.83-0.84 (CRISIS).

These tests pin the cut (regime_engine.drop_before_gap / contiguous_closes),
that fetch_closes and build_inputs (fresh and cached) both apply it, and that
a stray early bar no longer moves the volatility percentile.
"""
import math
import os
import random
import tempfile
import unittest
from datetime import date, timedelta

import regime_engine as re_


def bdays(start, n):
    d = date.fromisoformat(start)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def walk(n, seed=7, start=300.0, vol=0.01):
    rnd = random.Random(seed)
    px = [start]
    for _ in range(n - 1):
        px.append(px[-1] * math.exp(rnd.gauss(0.0003, vol)))
    return px


class TestDropBeforeGap(unittest.TestCase):
    def test_no_hole_returns_the_same_series_and_no_record(self):
        days = bdays("2020-07-27", 700)
        s = {d: 100.0 + i for i, d in enumerate(days)}
        kept, info = re_.drop_before_gap(s)
        self.assertIs(kept, s)
        self.assertIsNone(info)

    def test_the_production_shape_one_stray_bar_21_months_early(self):
        days = bdays("2020-07-27", 1590)
        s = {d: 300.0 + i for i, d in enumerate(days)}
        s["2018-11-01"] = 240.95
        kept, info = re_.drop_before_gap(s)
        self.assertEqual(sorted(kept), days)
        self.assertEqual(info, {"bars": 1, "first_kept": "2020-07-27", "last_dropped": "2018-11-01",
                                "gap_days": 634, "cut": True})

    def test_real_closures_are_not_holes(self):
        # 2001-09-10 -> 2001-09-17 is the longest closure since 1933: 7 calendar days apart.
        s = {"2001-09-07": 1.0, "2001-09-10": 2.0, "2001-09-17": 3.0, "2001-09-18": 4.0}
        kept, info = re_.drop_before_gap(s, min_keep=1)
        self.assertEqual(len(kept), 4)
        self.assertIsNone(info)
        self.assertLess(7, re_.MAX_BAR_GAP_DAYS)

    def test_the_last_hole_is_the_cut(self):
        a = bdays("2015-01-05", 20)
        b = bdays("2017-01-03", 20)
        c = bdays("2019-01-02", 600)
        s = {d: 1.0 for d in a + b + c}
        kept, info = re_.drop_before_gap(s)
        self.assertEqual(sorted(kept), c)
        self.assertEqual(info["bars"], 40)
        self.assertEqual(info["last_dropped"], b[-1])

    def test_a_cut_that_would_leave_too_little_history_is_not_made(self):
        old = bdays("2018-01-02", 900)
        new = bdays("2026-10-01", 30)          # a hole near the present
        s = {d: 1.0 for d in old + new}
        kept, info = re_.drop_before_gap(s)
        self.assertEqual(len(kept), 930)        # whole: a 30-bar series would read as calm
        self.assertFalse(info["cut"])
        self.assertEqual(info["bars"], 0)
        self.assertEqual(info["bars_after_hole"], 30)
        self.assertEqual(re_.MIN_BARS_AFTER_GAP, re_.DEFAULT_PARAMS["vol_pct_min_obs"])

    def test_dates_that_are_not_iso_days_leave_the_series_alone(self):
        s = {"d1": 1.0, "d2": 2.0}
        kept, info = re_.drop_before_gap(s)
        self.assertIs(kept, s)
        self.assertIsNone(info)

    def test_empty_and_none(self):
        self.assertEqual(re_.drop_before_gap({}), ({}, None))
        self.assertEqual(re_.drop_before_gap(None), (None, None))


class TestContiguousCloses(unittest.TestCase):
    def setUp(self):
        self.days = bdays("2020-07-27", 600)
        self.spy = {d: 300.0 + i for i, d in enumerate(self.days)}
        self.spy["2018-11-01"] = 240.95
        self.iwm = {d: 150.0 + i for i, d in enumerate(self.days)}

    def test_only_the_series_with_a_hole_is_cut_and_recorded(self):
        closes, sources = re_.contiguous_closes({"SPY": self.spy, "IWM": self.iwm}, {"SPY": "alpaca", "IWM": "alpaca"})
        self.assertNotIn("2018-11-01", closes["SPY"])
        self.assertEqual(len(closes["SPY"]), 600)
        self.assertEqual(closes["IWM"], self.iwm)
        self.assertEqual(list(sources["gap_dropped"]), ["SPY"])
        self.assertEqual(sources["SPY"], "alpaca")

    def test_no_hole_adds_no_key(self):
        closes, sources = re_.contiguous_closes({"IWM": self.iwm}, {"IWM": "alpaca"})
        self.assertEqual(sources, {"IWM": "alpaca"})

    def test_a_second_pass_changes_nothing(self):
        closes, sources = re_.contiguous_closes({"SPY": self.spy}, {"SPY": "alpaca"})
        again, sources2 = re_.contiguous_closes(closes, sources)
        self.assertEqual(again, closes)
        self.assertEqual(sources2, sources)

    def test_fetch_closes_cuts_what_the_feed_returns(self):
        def alp(syms, d):
            return {"SPY": sorted([k, v] for k, v in self.spy.items())}
        closes, sources = re_.fetch_closes(["SPY"], 3800, fetch_alpaca=alp, fetch_yahoo=lambda s, d: {})
        self.assertEqual(min(closes["SPY"]), "2020-07-27")
        self.assertEqual(sources["gap_dropped"]["SPY"]["gap_days"], 634)

    def test_build_inputs_cuts_a_cache_written_before_the_fix(self):
        import json
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as tmp:
            orig = re_._cache_dir
            re_._cache_dir = lambda: tmp
            try:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                with open(os.path.join(tmp, "regime_bars_SPY.json"), "w") as f:
                    json.dump({"day": today, "closes": {"SPY": self.spy, "IWM": self.iwm},
                               "sources": {"SPY": "alpaca", "IWM": "alpaca"}}, f)
                inp = re_.build_inputs("SPY", ebp=False)
                self.assertTrue(inp["cache_hit"])
                self.assertEqual(inp["dates"][0], "2020-07-27")
                self.assertEqual(len(inp["dates"]), 600)
                self.assertEqual(inp["sources"]["gap_dropped"]["SPY"]["bars"], 1)
            finally:
                re_._cache_dir = orig


class TestTheStrayBarNoLongerMovesTheReading(unittest.TestCase):
    """The mechanism, on a written-out series: one early bar far below the rest, read as
    the session before, depresses the volatility percentile for years."""

    def test_percentile_with_the_stray_cut_equals_the_clean_series(self):
        days = bdays("2020-07-27", 900)
        px = walk(900)
        clean = dict(zip(days, px))
        dirty = dict(clean)
        dirty["2018-11-01"] = px[0] * 0.75

        def last_pct(series):
            dates, al = re_.align({"SPY": series}, "SPY")
            return re_.signal_series(dates, al["SPY"], {})[-1]["vol_pct"]

        want = last_pct(clean)
        raw = last_pct(dirty)
        cut, _ = re_.contiguous_closes({"SPY": dirty}, {})
        self.assertIsNotNone(want)
        self.assertLess(raw, want - 0.02)                 # the bug: the stray bar lowers the reading
        self.assertEqual(last_pct(cut["SPY"]), want)      # the fix: the same as the clean history


class TestTheStateRecomputesOnceAfterTheDeploy(unittest.TestCase):
    def test_a_state_stamped_before_the_cut_is_not_served_as_current(self):
        # update_state serves the persisted state while its stamp equals PARAMS_VERSION and
        # the bar date has not advanced; a state computed on the holed series carries the
        # old stamp, so the first read after the deploy recomputes.
        self.assertNotEqual(re_.PARAMS_VERSION, "2026-10-28.1")
        days = bdays("2020-07-27", 40)
        inputs = {"dates": days, "gating": [100.0 + i for i in range(40)], "basket": {}, "vix": None, "vix3m": None,
                  "sources": {"SPY": "alpaca"}, "ebp": None, "cache_hit": False}
        with tempfile.TemporaryDirectory() as tmp:
            orig = re_._paths
            re_._paths = lambda: (os.path.join(tmp, "state.json"), os.path.join(tmp, "history.jsonl"))
            try:
                first = re_.update_state(build=lambda gating: inputs)
                self.assertNotIn("error", first)
                again = re_.update_state(build=lambda gating: inputs)
                self.assertTrue(again.get("cache_hit"))
                import json
                with open(os.path.join(tmp, "state.json")) as f:
                    old = json.load(f)
                old["params_version"] = "2026-10-28.1"
                with open(os.path.join(tmp, "state.json"), "w") as f:
                    json.dump(old, f)
                third = re_.update_state(build=lambda gating: inputs)
                self.assertFalse(third.get("cache_hit"))
                self.assertEqual(third["params_version"], re_.PARAMS_VERSION)
            finally:
                re_._paths = orig


if __name__ == "__main__":
    unittest.main()
