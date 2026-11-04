"""Tests for forecast.py — the weekly forecast screen (SHADOW, 2026-10-27).

Pinned: the range math, that a probability and a direction are ALWAYS emitted
(never a neutral default, never inflated), on-disk idempotency of the weekly
batch, the scorer (hit/miss, in-range, Brier, split-robust, pending, give-up,
append-only), no lookahead in the features or the validation split, and the
report's shape.

Daily horizon (Part C, 2026-11-04): the target session and phase, the pre-open
batch (base-rate direction, fc-range-har-v1 band, waits for the options capture
and the d0 VIX close before any network call), intraday rest-of-day calls and
the live band, the daily scorer (split-robust, both horizons kept apart), the
daily report, and the pins to the two committed studies.
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



# ── daily horizon (Part C, 2026-11-04) ──────────────────────────────────────
# Friday 2026-10-23 20:00 ET (after the options capture window) -> D = Mon 10-26
FRI_EVENING = datetime(2026, 10, 24, 0, 0, tzinfo=timezone.utc)
FRI_1800_ET = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
MON_0830_ET = datetime(2026, 10, 26, 12, 30, tzinfo=timezone.utc)
MON_1100_ET = datetime(2026, 10, 26, 15, 0, tzinfo=timezone.utc)
MON_1605_ET = datetime(2026, 10, 26, 20, 5, tzinfo=timezone.utc)
VIX_FRI = {"value": 18.5, "day": "2026-10-23"}


class DailyStore(TempStore):
    def setUp(self):
        super().setUp()
        self.live = os.path.join(self.dir, "live.json")

    def gen(self, now, tickers=("AAPL",), fetch=None, **kw):
        return fc.generate_daily(tickers=list(tickers), vix=kw.pop("vix", VIX_FRI), now=now,
                                 fetch=fetch or fake_fetch_factory(), path=self.path, live_path=self.live, **kw)

    def daily(self):
        return fc.load_store(self.path, self.spath, horizon="daily")


class TestDailyCalendar(unittest.TestCase):
    def test_target_session_and_phase(self):
        self.assertEqual(fc.daily_target(FRI_EVENING), (date(2026, 10, 23), date(2026, 10, 26), "pre_open"))
        self.assertEqual(fc.daily_target(MON_0830_ET), (date(2026, 10, 23), date(2026, 10, 26), "pre_open"))
        self.assertEqual(fc.daily_target(MON_1100_ET), (date(2026, 10, 23), date(2026, 10, 26), "open"))
        self.assertEqual(fc.daily_target(MON_1605_ET)[2], "closing")
        sat = datetime(2026, 10, 24, 16, 0, tzinfo=timezone.utc)
        self.assertEqual(fc.daily_target(sat), (date(2026, 10, 23), date(2026, 10, 26), "pre_open"))
        # after the Monday bar is final the target rolls to Tuesday
        self.assertEqual(fc.daily_target(datetime(2026, 10, 26, 21, 0, tzinfo=timezone.utc))[:2], (date(2026, 10, 26), date(2026, 10, 27)))

    def test_half_day_clock_is_stretched_onto_the_regular_session(self):
        # Fri 2026-11-27 closes 13:00 ET; 11:15 ET is half way through it
        half = datetime(2026, 11, 27, 16, 15, tzinfo=timezone.utc)
        self.assertEqual(fc.daily_target(half)[2], "open")
        self.assertAlmostEqual(fc.session_minutes(half, date(2026, 11, 27)), 570 + 195, places=6)
        self.assertEqual(fc.daily_target(datetime(2026, 11, 27, 18, 5, tzinfo=timezone.utc))[2], "closing")


class TestDailyPureParts(unittest.TestCase):
    def test_base_rate_uses_closes_up_to_d0_only(self):
        days = _trading_days(date(2023, 1, 2), date(2026, 10, 30))
        up = {s: [_bar(d, 100.0 + k) for k, d in enumerate(days)] for s in fc.TRAIN_BASKET}   # every day up
        br = fc.base_rate(up, "2026-10-23")
        self.assertEqual(br["p"], 1.0)
        self.assertEqual(br["last"], "2026-10-23")
        self.assertEqual(br["n"], len(fc.TRAIN_BASKET) * fc.DAILY_BASE_RATE_SESSIONS)
        # a crash AFTER d0 does not move it
        crash = {s: b[:] for s, b in up.items()}
        for s in crash:
            crash[s] = [x if x["t"][:10] <= "2026-10-23" else _bar(x["t"][:10], 1.0) for x in crash[s]]
        self.assertEqual(fc.base_rate(crash, "2026-10-23"), br)
        self.assertIsNone(fc.base_rate({"SPY": up["SPY"][:30]}, "2026-10-23"))

    def test_clim_q_is_the_tau_quantile_of_abs_returns(self):
        c = [100.0]
        for k in range(400):
            c.append(c[-1] * math.exp(0.01 if k % 3 else -0.02))
        q = fc.clim_q(c)
        import numpy as np
        r = np.abs(np.diff(np.log(c[-(fc.DAILY_CLIM_SESSIONS + 1):])))
        self.assertAlmostEqual(q, float(np.quantile(r, fc.FR.RANGE_TAU)), places=12)
        self.assertIsNone(fc.clim_q(c[:100]))

    def test_daily_direction_is_the_base_rate_shrunk_by_time_left(self):
        self.assertEqual(fc.daily_p_up(0.545, 1.0), 0.545)
        self.assertAlmostEqual(fc.daily_p_up(0.545, 0.25), 0.5 + 0.045 * 0.5)
        self.assertEqual(fc.daily_p_up(0.545, 0.0), 0.5)
        self.assertEqual(fc.DAILY_DIRECTION_FEATURES, ())

    def test_latest_price_reads_the_session_bar_only(self):
        bars = [_bar("2026-10-23", 10.0), _bar("2026-10-26", 10.4)]
        self.assertEqual(fc._latest_price(bars, date(2026, 10, 26)), 10.4)
        self.assertIsNone(fc._latest_price(bars[:1], date(2026, 10, 26)))


class TestDailyGenerate(DailyStore):
    def test_preopen_batch_has_a_call_per_name_with_the_base_rate_and_the_har_band(self):
        res = self.gen(FRI_EVENING, tickers=[{"symbol": "AAPL", "group": "position"}, "NVDA"],
                       iv={"AAPL": 0.31}, iv_day="2026-10-23")
        self.assertEqual(res["status"], "ok", res)
        batches, rows, _ = self.daily()
        self.assertEqual([r["symbol"] for r in rows], ["SPY", "AAPL", "NVDA"])
        b = batches[0]
        self.assertEqual((b["session"], b["base_date"], b["timing"], b["on_time"]), ("2026-10-26", "2026-10-23", "pre_open", True))
        self.assertTrue(b["vix_used"])
        for r in rows:
            self.assertEqual(r["horizon"], "daily")
            self.assertEqual(r["id"], f"D:2026-10-26:{r['symbol']}")
            self.assertEqual(r["p_up"], b["base_rate"]["p"])            # no feature: the base rate exactly
            self.assertEqual(r["p_base"], r["p_up"])
            self.assertEqual(r["direction"], "up" if r["p_up"] >= 0.5 else "down")
            self.assertEqual(r["spot"], r["base_close"])
            self.assertAlmostEqual(r["range_lo"], r["spot"] * math.exp(-r["q"]), places=3)
            self.assertAlmostEqual(r["range_hi"], r["spot"] * math.exp(r["q"]), places=3)
            self.assertEqual(r["q"], r["q_1d"])
            self.assertEqual(r["range_model_version"], fc.FR.RANGE_MODEL_VERSION)
            self.assertEqual(r["model_version"], fc.DAILY_MODEL_VERSION)
            self.assertGreater(r["q_clim"], 0)
        by = {r["symbol"]: r for r in rows}
        self.assertEqual(by["SPY"]["range_source"], "har+vix")
        self.assertEqual(by["AAPL"]["range_source"], "har+chain_iv30")
        self.assertEqual(by["NVDA"]["range_source"], "har")
        self.assertIsNone(by["NVDA"]["iv"])
        # the band equals the range module on the same closes
        _, w, _ = fc.load_store(self.path, self.spath)
        self.assertEqual(w, [], "daily calls never appear in the weekly store view")

    def test_waits_for_the_options_capture_and_for_the_vix_close_without_touching_the_network(self):
        calls = []
        f = fake_fetch_factory(calls)
        self.assertEqual(self.gen(FRI_1800_ET, fetch=f)["reason"], "options_capture_window")
        self.assertEqual(self.gen(FRI_EVENING, fetch=f, vix={"value": 17.0, "day": "2026-10-22"})["reason"], "vix_not_published")
        self.assertEqual(self.gen(FRI_EVENING, fetch=f, vix=None)["status"], "waiting")
        self.assertEqual(calls, [])
        self.assertFalse(os.path.exists(self.path))
        # from 08:00 ET on the session it goes ahead without VIX, labeled
        res = self.gen(MON_0830_ET, fetch=f, vix=None)
        self.assertEqual(res["status"], "ok")
        b, rows, _ = self.daily()
        spy = next(r for r in rows if r["symbol"] == "SPY")
        self.assertEqual(spy["range_source"], "har")
        self.assertFalse(b[0]["vix_used"])

    def test_one_call_per_name_and_session_first_wins(self):
        calls = []
        f = fake_fetch_factory(calls)
        self.gen(FRI_EVENING, tickers=["AAPL"], fetch=f)
        size = os.path.getsize(self.path)
        self.assertEqual(self.gen(FRI_EVENING + timedelta(hours=1), tickers=["AAPL"], fetch=f)["reason"], "exists")
        self.assertEqual(os.path.getsize(self.path), size)
        self.assertEqual(len(calls), 1, "an existing pre-open batch is detected on disk, before any fetch")
        res = self.gen(FRI_EVENING + timedelta(hours=2), tickers=["AAPL", "MSFT"], fetch=f)   # a new name before the open
        self.assertEqual((res["status"], res["new_calls"], res["batch"]), ("ok", 1, "existing"))
        _, rows, _ = self.daily()
        self.assertEqual(sorted(r["symbol"] for r in rows), ["AAPL", "MSFT", "SPY"])
        self.assertEqual(next(r for r in rows if r["symbol"] == "MSFT")["timing"], "pre_open")

    def test_intraday_names_get_a_rest_of_day_call_at_the_live_price_and_the_live_band_refreshes(self):
        self.gen(FRI_EVENING, tickers=["AAPL"])
        res = self.gen(MON_1100_ET, tickers=["AAPL", "TSLA"])
        self.assertEqual((res["status"], res["new_calls"]), ("ok", 1), res)
        b, rows, _ = self.daily()
        t = next(r for r in rows if r["symbol"] == "TSLA")
        f = fc.FR.remaining_variance_fraction(fc.session_minutes(MON_1100_ET, date(2026, 10, 26)), "TSLA")
        self.assertEqual((t["timing"], t["on_time"]), ("intraday", False))
        self.assertAlmostEqual(t["remaining_fraction"], f, places=4)
        self.assertAlmostEqual(t["q"], t["q_1d"] * math.sqrt(f), places=5)
        self.assertAlmostEqual(t["p_up"], 0.5 + (b[0]["base_rate"]["p"] - 0.5) * math.sqrt(f), places=4)
        start = (MON_1100_ET - timedelta(days=fc.HISTORY_CALENDAR_DAYS)).date().isoformat()
        raw, _ = fake_fetch_factory()(["TSLA"], start, MON_1100_ET.date().isoformat(), None)   # the generator's own fetch
        live_px = raw["TSLA"][-1]["c"]
        self.assertEqual(raw["TSLA"][-1]["t"][:10], "2026-10-26")
        self.assertAlmostEqual(t["spot"], live_px, places=3)
        self.assertNotEqual(t["spot"], t["base_close"])
        live = fc.read_live(self.live)
        self.assertEqual(live["session"], "2026-10-26")
        self.assertEqual(sorted(live["rows"]), ["AAPL", "SPY", "TSLA"])
        a = next(r for r in rows if r["symbol"] == "AAPL")
        lr = live["rows"]["AAPL"]
        self.assertAlmostEqual(lr["q"], a["q_1d"] * math.sqrt(f), places=5)
        self.assertAlmostEqual(lr["lo"], lr["price"] * math.exp(-lr["q"]), places=3)
        # a second intraday hour adds nothing, refreshes the live file only
        size = os.path.getsize(self.path)
        res2 = self.gen(MON_1100_ET + timedelta(hours=1), tickers=["AAPL", "TSLA"])
        self.assertEqual((res2["new_calls"], res2["live_rows"]), (0, 3))
        self.assertEqual(os.path.getsize(self.path), size)

    def test_a_batch_first_made_after_the_open_is_intraday_and_late(self):
        res = self.gen(MON_1100_ET, tickers=["AAPL"])
        self.assertEqual(res["status"], "ok", res)
        b, rows, _ = self.daily()
        self.assertEqual((b[0]["timing"], b[0]["on_time"]), ("intraday", False))
        self.assertTrue(all(r["timing"] == "intraday" for r in rows))

    def test_nothing_is_made_while_the_session_bar_settles(self):
        calls = []
        self.assertEqual(self.gen(MON_1605_ET, fetch=fake_fetch_factory(calls))["reason"], "session_closing")
        self.assertEqual(calls, [])

    def test_missing_spy_base_bar_writes_nothing(self):
        res = self.gen(FRI_EVENING, fetch=fake_fetch_factory(drop=("SPY",)))
        self.assertEqual(res["status"], "error")
        self.assertFalse(os.path.exists(self.path))


class TestDailyScorer(DailyStore):
    PRE = {"kind": "forecast", "horizon": "daily", "id": "D:2026-10-26:ABC", "session": "2026-10-26", "symbol": "ABC",
           "group": "position", "timing": "pre_open", "on_time": True, "base_date": "2026-10-23", "base_close": 100.0,
           "spot": 100.0, "p_up": 0.54, "p_base": 0.54, "direction": "up", "q": 0.01, "q_1d": 0.01, "q_clim": 0.012,
           "range_lo": 100 * math.exp(-0.01), "range_hi": 100 * math.exp(0.01), "made_at": "x"}

    def _bars(self, pairs):
        return [_bar(d, c) for d, c in pairs]

    def test_preopen_hit_inside_and_scores(self):
        s = fc.score_daily_one(self.PRE, self._bars([("2026-10-22", 99.0), ("2026-10-23", 100.0), ("2026-10-26", 100.5)]))
        self.assertEqual((s["outcome_up"], s["hit"], s["in_range"]), (1, 1, 1))
        self.assertAlmostEqual(s["brier"], (0.54 - 1) ** 2, places=9)
        self.assertEqual(s["brier"], s["brier_base"])
        y = math.log(100.5 / 100.0)
        self.assertAlmostEqual(s["interval_score"], fc.FR.interval_score(0.01, y) * 100, places=4)
        self.assertAlmostEqual(s["interval_score_clim"], fc.FR.interval_score(0.012, y) * 100, places=4)
        self.assertEqual(s["horizon"], "daily")

    def test_miss_outside_flat_and_split_robust(self):
        s = fc.score_daily_one(self.PRE, self._bars([("2026-10-23", 100.0), ("2026-10-26", 97.0)]))
        self.assertEqual((s["outcome_up"], s["hit"], s["in_range"]), (0, 0, 0))
        flat = fc.score_daily_one(self.PRE, self._bars([("2026-10-23", 100.0), ("2026-10-26", 100.0)]))
        self.assertEqual(flat["outcome_up"], 0)
        a = fc.score_daily_one(self.PRE, self._bars([("2026-10-23", 100.0), ("2026-10-26", 100.4)]))
        b = fc.score_daily_one(self.PRE, self._bars([("2026-10-23", 25.0), ("2026-10-26", 25.1)]))   # re-adjusted history
        self.assertEqual({k: a[k] for k in ("outcome_up", "in_range")}, {k: b[k] for k in ("outcome_up", "in_range")})
        self.assertAlmostEqual(a["realized_close"], b["realized_close"], places=6)

    def test_intraday_call_is_judged_against_its_own_price(self):
        intr = {**self.PRE, "timing": "intraday", "spot": 101.0, "range_lo": 101 * math.exp(-0.006), "range_hi": 101 * math.exp(0.006), "q": 0.006}
        s = fc.score_daily_one(intr, self._bars([("2026-10-23", 100.0), ("2026-10-26", 100.8)]))   # up vs 10-23, down vs 101
        self.assertEqual((s["outcome_up"], s["hit"], s["in_range"]), (0, 0, 1))

    def test_pending_until_the_session_bar_exists(self):
        self.assertIsNone(fc.score_daily_one(self.PRE, self._bars([("2026-10-23", 100.0)])))

    def test_score_matured_scores_both_horizons_and_keeps_them_apart(self):
        fc._append_lines(self.path, [self.PRE, {**TestScorer.FC}])
        bars = {"ABC": self._bars([("2026-10-16", 97), ("2026-10-19", 98), ("2026-10-20", 99), ("2026-10-21", 99), ("2026-10-22", 99),
                                    ("2026-10-23", 100), ("2026-10-26", 101), ("2026-10-27", 102), ("2026-10-28", 103), ("2026-10-29", 104), ("2026-10-30", 103)])}

        def fetch(symbols, start, end, deadline):
            return {s: bars.get(s, []) for s in symbols}, set()
        st = fc.score_matured(now=datetime(2026, 10, 31, 2, 0, tzinfo=timezone.utc), fetch=fetch, path=self.path, scores_path=self.spath)
        self.assertEqual(st["resolved"], 1)
        self.assertEqual(st["daily"]["resolved"], 1)
        _, _, wscores = fc.load_store(self.path, self.spath)
        _, _, dscores = fc.load_store(self.path, self.spath, horizon="daily")
        self.assertEqual(list(wscores), ["2026-W44:ABC"])
        self.assertEqual(list(dscores), ["D:2026-10-26:ABC"])
        st2 = fc.score_matured(now=datetime(2026, 10, 31, 3, 0, tzinfo=timezone.utc), fetch=fetch, path=self.path, scores_path=self.spath)
        self.assertEqual((st2["resolved"], st2["daily"]["resolved"]), (0, 0))

    def test_daily_give_up_and_fetch_failure(self):
        fc._append_lines(self.path, [self.PRE])

        def nothing(symbols, start, end, deadline):
            return {}, set()

        def down(symbols, start, end, deadline):
            return {}, set(symbols)
        late = datetime(2026, 12, 15, 2, 0, tzinfo=timezone.utc)
        st = fc.score_matured(now=late, fetch=down, path=self.path, scores_path=self.spath)
        self.assertEqual((st["daily"]["fetch_errors"], st["daily"]["gave_up"]), (1, 0))
        st = fc.score_matured(now=late, fetch=nothing, path=self.path, scores_path=self.spath)
        self.assertEqual(st["daily"]["gave_up"], 1)
        _, _, sc = fc.load_store(self.path, self.spath, horizon="daily")
        self.assertEqual(sc["D:2026-10-26:ABC"]["status"], "no_price_data")


class TestDailyReport(DailyStore):
    def test_empty_daily_report_shape(self):
        r = fc.report(self.path, self.spath, now=FRI_EVENING, live_path=self.live)
        d = r["daily"]
        for k in ("model_version", "range_model_version", "direction_model", "direction_features", "latest", "live",
                  "accuracy", "calibration", "sessions", "open", "study", "method", "caveats"):
            self.assertIn(k, d)
        self.assertIsNone(d["latest"])
        self.assertTrue(d["accuracy"]["thin"])
        self.assertEqual(d["direction_features"], [])
        json.dumps(r)

    def test_daily_report_joins_scores_live_and_skill_references(self):
        self.gen(FRI_EVENING, tickers=["AAPL"])
        self.gen(MON_1100_ET, tickers=["AAPL"])
        _, rows, _ = self.daily()
        spy = next(x for x in rows if x["symbol"] == "SPY")
        fc._append_lines(self.spath, [{"kind": "score", "horizon": "daily", "id": spy["id"], "status": "resolved", "outcome_up": 1,
                                       "hit": 1, "in_range": 1, "brier": round((spy["p_up"] - 1) ** 2, 6), "brier_base": round((spy["p_up"] - 1) ** 2, 6),
                                       "interval_score": 2.0, "interval_score_clim": 2.5, "realized_close": 1.0, "resolved_date": "2026-10-26"}])
        d = fc.report(self.path, self.spath, now=MON_1100_ET, live_path=self.live)["daily"]
        self.assertEqual(d["latest"]["session"], "2026-10-26")
        self.assertEqual([c["symbol"] for c in d["latest"]["forecasts"]], ["SPY", "AAPL"])
        self.assertEqual(sum(d["latest"]["range_sources"].values()), d["latest"]["n_forecasts"])
        self.assertTrue(all(c["live"] for c in d["latest"]["forecasts"]))
        self.assertEqual(d["live"]["session"], "2026-10-26")
        a = d["accuracy"]
        self.assertEqual(a["n"], 1)
        self.assertEqual(a["bss_vs_base_rate"], 0.0)          # the call IS the base rate
        self.assertAlmostEqual(a["bss_vs_coinflip"], round(1 - a["brier"] / 0.25, 4))
        self.assertAlmostEqual(a["range_skill_vs_constant"], 1 - 2.0 / 2.5, places=4)
        self.assertEqual(sum(b["n"] for b in d["calibration"]["bins"]), 1)
        self.assertEqual(d["sessions"][0]["resolved"], 1)
        self.assertEqual(d["open"], 1)
        # the weekly half of the report never sees a daily call
        r = fc.report(self.path, self.spath, now=MON_1100_ET, live_path=self.live)
        self.assertIsNone(r["latest"])
        self.assertEqual(r["accuracy"]["n"], 0)

    def test_a_stale_live_file_is_not_attached(self):
        self.gen(FRI_EVENING, tickers=["AAPL"])
        fc._write_json_atomic(self.live, {"session": "2026-10-23", "made_at": "x", "rows": {"SPY": {"price": 1}}})
        d = fc.daily_report(self.path, self.spath, self.live)
        self.assertIsNone(d["live"])
        self.assertTrue(all(c["live"] is None for c in d["latest"]["forecasts"]))


class TestDailyStudyPins(unittest.TestCase):
    """The page's research numbers and the empty direction-feature tuple are
    exactly the committed studies' — a new model needs a new study."""

    def _js(self, name):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "research", name)) as fh:
            return json.load(fh)

    def test_direction_features_follow_the_direction_study(self):
        js = self._js("forecast_direction_study_2026-11-04.json")
        self.assertEqual(fc.DAILY_DIRECTION_FEATURES, tuple(js["shipped_features"]))
        self.assertEqual(fc.DAILY_STUDY["direction"]["passed"], sum(1 for t in js["tests"].values() if t.get("verdict") == "PASS"))
        self.assertEqual(fc.DAILY_STUDY["direction"]["tests"], js["spec"]["m_tests"])
        b = js["base"]["SPY_y"]
        self.assertAlmostEqual(fc.DAILY_STUDY["direction"]["spy_base_rate"], b["up_rate"], places=4)
        self.assertAlmostEqual(fc.DAILY_STUDY["direction"]["spy_brier_base_rate"], b["brier_base"], places=5)
        self.assertAlmostEqual(fc.DAILY_STUDY["direction"]["spy_skill_vs_coinflip"], b["skill_vs_coinflip"], places=4)

    def test_range_numbers_follow_the_range_study(self):
        js = self._js("forecast_daily_study_2026-11-04.json")
        r = fc.DAILY_STUDY["range"]
        spy = js["groups"]["index_h1"]["per_name_holdout"]["SPY"]
        self.assertAlmostEqual(r["spy_coverage"], spy["V4"]["coverage"], places=4)
        self.assertAlmostEqual(r["spy_skill_vs_constant"], 1 - spy["V4"]["is"] / spy["V_clim"]["is"], places=4)
        self.assertAlmostEqual(r["spy_skill_vs_current"], 1 - spy["V4"]["is"] / spy["V0"]["is"], places=4)
        iv = js["groups"]["stock_iv_h1"]["holdout"]
        self.assertAlmostEqual(r["iv_names_coverage"], iv["V4"]["coverage"], places=4)
        self.assertAlmostEqual(r["iv_names_skill_vs_constant"], 1 - iv["V4"]["is"] / iv["V_clim"]["is"], places=4)
        har = js["groups"]["har_h1"]["holdout"]
        self.assertAlmostEqual(r["names_coverage"], har["V3"]["coverage"], places=4)
        self.assertAlmostEqual(r["names_skill_vs_constant"], 1 - har["V3"]["is"] / har["V_clim"]["is"], places=4)
        # the shipped band is the study's chosen model in every group
        self.assertEqual([js["groups"][k]["chosen"] for k in ("index_h1", "stock_iv_h1", "har_h1")], ["V4", "V4", "V3"])
        self.assertTrue(all(js["groups"][k]["ship_rule_R2"] for k in ("index_h1", "stock_iv_h1", "har_h1")))


if __name__ == "__main__":
    unittest.main()
