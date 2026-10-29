"""Tests for forecast.py — the weekly forecast screen (SHADOW, 2026-10-27).

Pinned: the range math, that a probability and a direction are ALWAYS emitted
(never a neutral default, never inflated), on-disk idempotency of the weekly
batch, the scorer (hit/miss, in-range, Brier, split-robust, pending, give-up,
append-only), no lookahead in the features or the validation split, and the
report's shape.
"""
import json
import math
import os
import random
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone

import forecast as fc

# Friday 2026-10-23 17:00 ET (after the close) — d0 is that Friday.
FRI_AFTER_CLOSE = datetime(2026, 10, 23, 21, 0, tzinfo=timezone.utc)
FRI_MIDDAY = datetime(2026, 10, 23, 16, 0, tzinfo=timezone.utc)   # 12:00 ET


def _bar(day, c):
    return {"t": f"{day}T04:00:00Z", "o": c, "h": c, "l": c, "c": c, "v": 1000}


def _trading_days(start, end):
    d, out = start, []
    while d <= end:
        if fc._is_trading_day(d):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def fake_fetch_factory(calls=None, drop=(), last_day=None):
    """Deterministic random-walk daily bars for any symbol (seeded by name)."""
    def fetch(symbols, start, end, deadline):
        if calls is not None:
            calls.append(list(symbols))
        days = _trading_days(date.fromisoformat(start), date.fromisoformat(last_day or end))
        out = {}
        for s in symbols:
            if s in drop:
                continue
            rng = random.Random(s)
            p, bars = 50.0 + rng.random() * 100, []
            for d in days:
                p *= math.exp(rng.gauss(0.0003, 0.015))
                bars.append(_bar(d, round(p, 4)))
            out[s] = bars
        return out, set(drop)
    return fetch


class TempStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "fc.jsonl")
        self.spath = os.path.join(self.dir, "fc_scores.jsonl")


class TestRangeMath(unittest.TestCase):
    def test_band_is_symmetric_in_log_space(self):
        lo, hi = fc.band(100.0, 0.05)
        self.assertAlmostEqual(lo, 100 * math.exp(-0.05), places=9)
        self.assertAlmostEqual(hi, 100 * math.exp(0.05), places=9)
        self.assertAlmostEqual(math.log(hi / 100), -math.log(lo / 100), places=12)

    def test_realized_vol_path_scales_by_sqrt_five(self):
        closes = [100.0]
        for k in range(30):
            closes.append(closes[-1] * math.exp(0.01 if k % 2 else -0.01))
        sd = fc.daily_vol(closes, len(closes) - 1)
        sig, src, ann = fc.sigma_for(closes, date(2026, 10, 23), date(2026, 10, 30), None, None)
        self.assertEqual(src, "realized_20d")
        self.assertAlmostEqual(sig, sd * math.sqrt(5), places=12)
        self.assertAlmostEqual(ann, sd * math.sqrt(252), places=12)

    def test_chain_iv_path_uses_calendar_days_over_365(self):
        closes = [100.0 + k for k in range(30)]
        sig, src, ann = fc.sigma_for(closes, date(2026, 10, 23), date(2026, 10, 30), 0.40, "2026-10-23")
        self.assertEqual(src, "chain_iv30")
        self.assertAlmostEqual(sig, 0.40 * math.sqrt(7 / 365), places=12)

    def test_stale_or_bad_iv_falls_back_to_realized(self):
        closes = [100.0 * math.exp(0.01 * ((-1) ** k)) for k in range(30)]
        for iv, day in ((0.4, "2026-09-01"), (0.0, "2026-10-23"), (float("nan"), "2026-10-23"), (0.4, None)):
            _, src, _ = fc.sigma_for(closes, date(2026, 10, 23), date(2026, 10, 30), iv, day)
            self.assertEqual(src, "realized_20d", (iv, day))


class TestFeatures(unittest.TestCase):
    def test_features_use_no_future_bars(self):
        rng = random.Random(7)
        c = [100.0]
        for _ in range(120):
            c.append(c[-1] * math.exp(rng.gauss(0, 0.02)))
        spy = [400.0 + k for k in range(121)]
        x = fc.feature_row(c, 80, spy, 80)
        c2 = c[:81] + [v * 3 for v in c[81:]]
        spy2 = spy[:81] + [1.0] * 40
        self.assertEqual(x, fc.feature_row(c2, 80, spy2, 80))

    def test_short_history_gives_none(self):
        self.assertIsNone(fc.feature_row([100.0] * 10, 9, [1.0] * 100, 60))


class TestValidationSplit(unittest.TestCase):
    def test_holdout_is_purged_and_time_ordered(self):
        fetch = fake_fetch_factory()
        raw, _ = fetch(list(fc.TRAIN_BASKET), "2023-09-01", "2026-10-23", None)
        series = {s: fc.completed_bars(b, FRI_AFTER_CLOSE) for s, b in raw.items()}
        samples = fc.build_samples(series, series["SPY"])
        model = fc.fit_and_validate(samples)
        self.assertIsNotNone(model)
        v = model["validation"]
        split = v["split_date"]
        train = [s for s in samples if s["label_date"] < split]
        test = [s for s in samples if s["date"] >= split]
        self.assertEqual((v["n_train"], v["n_test"]), (len(train), len(test)))
        self.assertTrue(all(s["label_date"] < split for s in train))
        self.assertGreater(v["n_train"], v["n_test"])
        self.assertEqual(len(model["coef"]), 1 + len(fc.FEATURES))
        self.assertEqual(sum(b["n"] for b in v["calibration"]), v["n_test"])

    def test_samples_do_not_overlap_within_a_name(self):
        fetch = fake_fetch_factory()
        raw, _ = fetch(["SPY", "XLE"], "2025-01-01", "2026-10-23", None)
        samples = fc.build_samples(raw, raw["SPY"])
        xle = sorted(s["date"] for s in samples if s["sym"] == "XLE")
        days = [b["t"][:10] for b in raw["XLE"]]
        idx = [days.index(d) for d in xle]
        self.assertTrue(all(b - a == fc.SAMPLE_STRIDE for a, b in zip(idx, idx[1:])))


class TestCalendar(unittest.TestCase):
    def test_friday_after_close_targets_next_week(self):
        self.assertEqual(fc.expected_base_date(FRI_AFTER_CLOSE), date(2026, 10, 23))
        self.assertEqual(fc.week_key(date(2026, 10, 23)), "2026-W44")
        self.assertEqual(fc.horizon_end(date(2026, 10, 23)), date(2026, 10, 30))

    def test_midday_today_bar_is_not_complete(self):
        bars = [_bar("2026-10-22", 10.0), _bar("2026-10-23", 11.0)]
        self.assertEqual([b["t"][:10] for b in fc.completed_bars(bars, FRI_MIDDAY)], ["2026-10-22"])
        self.assertEqual(len(fc.completed_bars(bars, FRI_AFTER_CLOSE)), 2)
        self.assertEqual(fc.expected_base_date(FRI_MIDDAY), date(2026, 10, 22))

    def test_horizon_skips_thanksgiving(self):
        # Fri 2026-11-20 + 5 sessions: 23, 24, 25, (26 closed), 27, 30
        self.assertEqual(fc.horizon_end(date(2026, 11, 20)), date(2026, 11, 30))


class TestGenerate(TempStore):
    def test_every_ticker_gets_a_real_probability_and_a_call(self):
        res = fc.generate_weekly(tickers=[{"symbol": "AAPL", "group": "position"}, "NVDA", "spy"],
                                 iv={"AAPL": 0.31}, iv_day="2026-10-23", now=FRI_AFTER_CLOSE,
                                 fetch=fake_fetch_factory(), path=self.path)
        self.assertEqual(res["status"], "ok", res)
        _, rows, _ = fc.load_store(self.path, self.spath)
        self.assertEqual([r["symbol"] for r in rows], ["SPY", "AAPL", "NVDA"])
        for r in rows:
            self.assertTrue(0.0 < r["p_up"] < 1.0)
            self.assertIn(r["direction"], ("up", "down"))
            self.assertEqual(r["direction"], "up" if r["p_up"] >= 0.5 else "down")
            self.assertAlmostEqual(r["p_call"], max(r["p_up"], 1 - r["p_up"]), places=4)
            self.assertLess(r["range_lo"], r["spot"])
            self.assertGreater(r["range_hi"], r["spot"])
            self.assertEqual(r["base_date"], "2026-10-23")
            self.assertEqual(r["horizon_end"], "2026-10-30")
            self.assertEqual(r["model_version"], fc.MODEL_VERSION)
        by = {r["symbol"]: r for r in rows}
        self.assertEqual(by["SPY"]["group"], "index")
        self.assertEqual(by["AAPL"]["vol_source"], "chain_iv30")
        self.assertEqual(by["NVDA"]["vol_source"], "realized_20d")
        self.assertEqual(by["NVDA"]["group"], "candidate")

    def test_probability_is_not_inflated(self):
        # p_up is exactly the model's sigmoid on the stored features
        fc.generate_weekly(tickers=["XOM"], now=FRI_AFTER_CLOSE, fetch=fake_fetch_factory(), path=self.path)
        batches, rows, _ = fc.load_store(self.path, self.spath)
        coef = batches[0]["model"]["coef"]
        for r in rows:
            x = [r["features"][k] for k in fc.FEATURES]
            self.assertAlmostEqual(r["p_up"], fc.predict(coef, x), places=3)

    def test_weekly_batch_is_idempotent_on_disk(self):
        calls = []
        f = fake_fetch_factory(calls)
        first = fc.generate_weekly(tickers=["MSFT"], now=FRI_AFTER_CLOSE, fetch=f, path=self.path)
        size = os.path.getsize(self.path)
        second = fc.generate_weekly(tickers=["MSFT", "TSLA"], now=FRI_AFTER_CLOSE + timedelta(hours=30),
                                    fetch=f, path=self.path)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second, {"status": "skipped", "reason": "exists", "week_key": "2026-W44"})
        self.assertEqual(os.path.getsize(self.path), size)
        self.assertEqual(len(calls), 1, "the existing-batch check must not touch the network")

    def test_first_run_midweek_is_flagged_late(self):
        tue = datetime(2026, 10, 27, 15, 0, tzinfo=timezone.utc)  # 11:00 ET, session open
        res = fc.generate_weekly(tickers=[], now=tue, fetch=fake_fetch_factory(), path=self.path)
        self.assertEqual(res["status"], "ok")
        b = fc.load_store(self.path, self.spath)[0][0]
        self.assertEqual(b["base_date"], "2026-10-26")
        self.assertEqual(b["week_key"], "2026-W44")
        self.assertFalse(b["on_time"])

    def test_friday_batch_is_on_time(self):
        fc.generate_weekly(tickers=[], now=FRI_AFTER_CLOSE, fetch=fake_fetch_factory(), path=self.path)
        self.assertTrue(fc.load_store(self.path, self.spath)[0][0]["on_time"])

    def test_spy_failure_writes_nothing(self):
        res = fc.generate_weekly(tickers=["AAPL"], now=FRI_AFTER_CLOSE,
                                 fetch=fake_fetch_factory(drop=("SPY",)), path=self.path)
        self.assertEqual(res["status"], "error")
        self.assertFalse(os.path.exists(self.path))

    def test_unfetched_name_is_skipped_with_a_reason(self):
        fc.generate_weekly(tickers=["AAPL", "ZZZZ"], now=FRI_AFTER_CLOSE,
                           fetch=fake_fetch_factory(drop=("ZZZZ",)), path=self.path)
        b = fc.load_store(self.path, self.spath)[0][0]
        self.assertEqual(b["skipped"], [{"symbol": "ZZZZ", "reason": "bars_not_fetched"}])

    def test_universe_normalization(self):
        u = fc.normalize_universe([{"symbol": "aapl", "group": "candidate"}, {"symbol": "AAPL", "group": "position"},
                                   "bad sym!", {"symbol": "SPY", "group": "candidate"}, {"symbol": "X", "group": "weird"}])
        self.assertEqual(u, [{"symbol": "SPY", "group": "index"}, {"symbol": "AAPL", "group": "position"}])
        self.assertEqual(len(fc.normalize_universe([f"T{chr(65 + k % 26)}{chr(65 + k // 26)}" for k in range(80)])),
                         fc.MAX_TICKERS)


class TestScorer(TempStore):
    FC = {"kind": "forecast", "id": "2026-W44:ABC", "week_key": "2026-W44", "symbol": "ABC", "group": "position",
          "base_date": "2026-10-23", "horizon_end": "2026-10-30", "spot": 100.0, "p_up": 0.6, "direction": "up",
          "p_call": 0.6, "range_lo": 95.0, "range_hi": 105.0, "vol_source": "realized_20d", "made_at": "x"}

    def _bars(self, closes, start="2026-10-21"):
        days = _trading_days(date.fromisoformat(start), date(2026, 12, 31))
        return [_bar(d, c) for d, c in zip(days, closes)]

    def test_hit_in_range_and_brier(self):
        bars = self._bars([99, 100, 100, 101, 102, 103, 104, 103])  # d0=10-23 at idx 2, d0+5 = 103
        s = fc.score_one(self.FC, bars)
        self.assertEqual((s["outcome_up"], s["hit"], s["in_range"]), (1, 1, 1))
        self.assertEqual(s["resolved_date"], "2026-10-30")
        self.assertAlmostEqual(s["brier"], (0.6 - 1) ** 2, places=9)
        self.assertAlmostEqual(s["realized_close"], 103.0, places=6)

    def test_miss_outside_range(self):
        s = fc.score_one(self.FC, self._bars([99, 100, 100, 99, 98, 97, 96, 90]))
        self.assertEqual((s["outcome_up"], s["hit"], s["in_range"]), (0, 0, 0))
        self.assertAlmostEqual(s["brier"], 0.36, places=9)

    def test_flat_close_is_not_up(self):
        s = fc.score_one(self.FC, self._bars([99, 100, 100, 101, 99, 100, 101, 100]))
        self.assertEqual((s["outcome_up"], s["hit"]), (0, 0))

    def test_split_after_the_call_cannot_fake_a_result(self):
        closes = [99, 100, 100, 101, 102, 103, 104, 103]
        a = fc.score_one(self.FC, self._bars(closes))
        b = fc.score_one(self.FC, self._bars([c / 4 for c in closes]))  # re-adjusted history
        self.assertEqual({k: a[k] for k in ("outcome_up", "hit", "in_range")},
                         {k: b[k] for k in ("outcome_up", "hit", "in_range")})
        self.assertAlmostEqual(a["realized_close"], b["realized_close"], places=6)

    def test_pending_until_the_fifth_session_closes(self):
        self.assertIsNone(fc.score_one(self.FC, self._bars([99, 100, 100, 101, 102, 103, 104])))
        self.assertIsNone(fc.score_one(self.FC, self._bars([1, 2, 3], start="2026-11-02")))

    def test_score_matured_appends_once(self):
        fc._append_lines(self.path, [self.FC, {**self.FC, "id": "2026-W44:GONE", "symbol": "GONE"}])
        bars = {"ABC": self._bars([97, 98, 99, 99, 99, 100, 101, 102, 103, 104, 103], start="2026-10-16")}

        def fetch(symbols, start, end, deadline):
            return {s: bars.get(s, []) for s in symbols}, set()
        now = datetime(2026, 10, 31, 2, 0, tzinfo=timezone.utc)
        st = fc.score_matured(now=now, fetch=fetch, path=self.path, scores_path=self.spath)
        self.assertEqual(st["resolved"], 1)
        self.assertEqual(st["pending"], 1)          # GONE: no bars yet, not past give-up
        st2 = fc.score_matured(now=now, fetch=fetch, path=self.path, scores_path=self.spath)
        self.assertEqual(st2["resolved"], 0)
        late = datetime(2026, 12, 15, 2, 0, tzinfo=timezone.utc)
        st3 = fc.score_matured(now=late, fetch=fetch, path=self.path, scores_path=self.spath)
        self.assertEqual(st3["gave_up"], 1)
        _, _, scores = fc.load_store(self.path, self.spath)
        self.assertEqual(scores["2026-W44:GONE"]["status"], "no_price_data")
        with open(self.spath) as f:
            self.assertEqual(len(f.read().strip().splitlines()), 2)

    def test_fetch_failure_is_not_a_strike(self):
        fc._append_lines(self.path, [self.FC])

        def fetch(symbols, start, end, deadline):
            return {}, set(symbols)
        late = datetime(2026, 12, 15, 2, 0, tzinfo=timezone.utc)
        st = fc.score_matured(now=late, fetch=fetch, path=self.path, scores_path=self.spath)
        self.assertEqual((st["fetch_errors"], st["gave_up"]), (1, 0))
        self.assertFalse(os.path.exists(self.spath))


class TestReport(TempStore):
    def test_empty_report_shape(self):
        r = fc.report(self.path, self.spath, now=FRI_AFTER_CLOSE)
        for k in ("generated_at", "model_version", "experimental", "shadow", "latest", "accuracy",
                  "calibration", "weeks", "open", "method", "caveats"):
            self.assertIn(k, r)
        self.assertIsNone(r["latest"])
        self.assertEqual(r["accuracy"]["n"], 0)
        self.assertTrue(r["accuracy"]["thin"])
        self.assertEqual(len(r["calibration"]["bins"]), len(fc.CAL_EDGES) - 1)
        json.dumps(r)

    def test_report_joins_scores_and_calibrates(self):
        fc.generate_weekly(tickers=["AAPL", "MSFT"], now=FRI_AFTER_CLOSE, fetch=fake_fetch_factory(), path=self.path)
        _, rows, _ = fc.load_store(self.path, self.spath)
        fc._append_lines(self.spath, [{"kind": "score", "id": rows[0]["id"], "status": "resolved", "outcome_up": 1,
                                       "hit": int(rows[0]["direction"] == "up"), "in_range": 1, "brier": 0.1,
                                       "realized_close": 1.0, "resolved_date": "2026-10-30"}])
        r = fc.report(self.path, self.spath, now=FRI_AFTER_CLOSE)
        self.assertEqual(r["latest"]["week_key"], "2026-W44")
        self.assertEqual(len(r["latest"]["forecasts"]), 3)
        self.assertEqual(r["latest"]["forecasts"][0]["symbol"], "SPY")
        scored = [f for f in r["latest"]["forecasts"] if f["score"]]
        self.assertEqual(len(scored), 1)
        self.assertEqual(r["accuracy"]["n"], 1)
        self.assertAlmostEqual(r["accuracy"]["brier"], (rows[0]["p_up"] - 1) ** 2, places=4)
        self.assertEqual(sum(b["n"] for b in r["calibration"]["bins"]), 1)
        self.assertEqual(r["weeks"][0]["resolved"], 1)
        self.assertEqual(r["open"], 2)
        self.assertIn("validation", r["latest"]["model"])

    def test_calibration_bins_edges(self):
        bins = fc.calibration_bins([(0.0, 0), (0.52, 1), (0.52, 0), (1.0, 1), (0.349, 1)])
        self.assertEqual([b["n"] for b in bins], [2, 0, 0, 0, 2, 0, 0, 1])
        self.assertEqual(bins[4]["freq"], 0.5)
        self.assertIsNone(bins[1]["freq"])


if __name__ == "__main__":
    unittest.main()
