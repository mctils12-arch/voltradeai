"""Unit tests for scripts/wikiattention_gate2.py — synthetic data only, no
network (this repo's Wikimedia/price fetches are exercised live by the
script's own __main__ path, not by CI). Mirrors test_hazard_rate_probe.py's
convention of pure-function coverage plus a couple of orchestration-shape
smoke checks."""
import importlib.util
import math
import os
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "wikiattention_gate2", os.path.join(os.path.dirname(__file__), "scripts", "wikiattention_gate2.py"))
wag2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wag2)


class TestZscoreSeries:
    def test_insufficient_window_is_none(self):
        views = [10] * 50  # < default window (90)
        z = wag2.zscore_series(views, window=90)
        assert all(v is None for v in z)

    def test_flat_baseline_then_spike(self):
        views = [10] * 90 + [500]
        z = wag2.zscore_series(views, window=90)
        assert z[89] is None  # zero-std trailing window -> None, never fabricated
        # index 90 has a full trailing window of flat 10s (std=0) too
        assert z[90] is None

    def test_real_spike_scores_high(self):
        # trailing window has real variance, then a genuine outlier
        import random
        random.seed(0)
        views = [100 + random.randint(-5, 5) for _ in range(90)] + [400]
        z = wag2.zscore_series(views, window=90)
        assert z[90] is not None
        assert z[90] > 3

    def test_no_lookahead(self):
        """Changing a FUTURE value must never change a past z-score."""
        import random
        random.seed(1)
        base = [100 + random.randint(-5, 5) for _ in range(120)]
        z1 = wag2.zscore_series(base, window=90)
        mutated = list(base)
        mutated[110] = 99999  # far-future spike
        z2 = wag2.zscore_series(mutated, window=90)
        assert z1[95] == z2[95]

    def test_none_view_stays_none(self):
        import random
        random.seed(2)
        views = [100 + random.randint(-5, 5) for _ in range(90)] + [None]
        z = wag2.zscore_series(views, window=90)
        assert z[90] is None


class TestSpikeDayIndices:
    def test_threshold(self):
        z = [None, 0.5, 2.0, 2.1, None, -3.0]
        assert wag2.spike_day_indices(z, threshold=2.0) == [2, 3]

    def test_empty(self):
        assert wag2.spike_day_indices([], threshold=2.0) == []


class TestDailyReturns:
    def test_basic(self):
        closes = [100, 110, 99]
        r = wag2.daily_returns(closes)
        assert r[0] is None
        assert math.isclose(r[1], 0.10, rel_tol=1e-9)
        assert math.isclose(r[2], -0.1, rel_tol=1e-9)

    def test_zero_prior_close_is_none(self):
        r = wag2.daily_returns([0, 5])
        assert r[1] is None


class TestForwardVolumeRatio:
    def test_ratio_above_baseline(self):
        volumes = [100] * 21 + [500, 500, 500]
        r = wag2.forward_volume_ratio(volumes, idx=20, horizon=3, baseline_window=20)
        assert r is not None
        assert math.isclose(r, 5.0, rel_tol=1e-6)

    def test_insufficient_forward_days_is_none(self):
        volumes = [100] * 20 + [500]
        assert wag2.forward_volume_ratio(volumes, idx=19, horizon=3, baseline_window=20) is None

    def test_insufficient_baseline_is_none(self):
        volumes = [100] * 5 + [500, 500, 500]
        assert wag2.forward_volume_ratio(volumes, idx=4, horizon=3, baseline_window=20) is None

    def test_spike_day_own_volume_excluded_from_its_baseline(self):
        # idx's own (huge) volume must not leak into the trailing baseline
        volumes = [100] * 20 + [999999] + [100] * 3
        r = wag2.forward_volume_ratio(volumes, idx=20, horizon=3, baseline_window=20)
        assert math.isclose(r, 1.0, rel_tol=1e-6)


class TestForwardRealizedVol:
    def test_basic_nonzero(self):
        returns = [None, 0.0, 0.0, 0.1, -0.1, 0.05]
        v = wag2.forward_realized_vol(returns, idx=1, horizon=3)
        assert v is not None
        assert v > 0

    def test_insufficient_forward_days_is_none(self):
        returns = [None, 0.0, 0.1]
        assert wag2.forward_realized_vol(returns, idx=1, horizon=5) is None

    def test_none_gaps_shrink_available_window(self):
        returns = [None, 0.0, None, 0.1]
        assert wag2.forward_realized_vol(returns, idx=0, horizon=3) is None


class TestAlignViewsToTradingDays:
    def test_present_and_missing(self):
        by_date = {"2026-01-02": 500, "2026-01-05": 600}
        dates = ["2026-01-02", "2026-01-03", "2026-01-05"]
        out = wag2.align_views_to_trading_days(by_date, dates)
        assert out == [500, None, 600]

    def test_never_zero_fills(self):
        out = wag2.align_views_to_trading_days({}, ["2026-01-02"])
        assert out == [None]  # honest absence, not 0


class TestWelchVsBaseline:
    def test_below_min_n_is_none(self):
        assert wag2.welch_vs_baseline([1, 2, 3], [1, 2, 3, 4, 5]) is None
        assert wag2.welch_vs_baseline([1, 2, 3, 4, 5], [1, 2, 3]) is None

    def test_clear_separation_significant(self):
        sample = [10.0 + 0.01 * i for i in range(20)]
        baseline = [1.0 + 0.01 * i for i in range(20)]
        r = wag2.welch_vs_baseline(sample, baseline)
        assert r is not None
        assert r["p_value"] < 0.01
        assert r["mean_diff"] > 0

    def test_identical_distributions_not_significant(self):
        sample = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98]
        baseline = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98]
        r = wag2.welch_vs_baseline(sample, baseline)
        assert r is not None
        assert r["p_value"] > 0.9


class TestEvaluateTicker:
    def _synthetic_series(self, n=400, spike_ats=(100, 140, 180, 220, 260, 300)):
        """Several engineered spike days (welch_vs_baseline needs n>=5 per
        side, so a single spike can never clear that floor — matching how
        run_gate2's own docstring notes real gate 2 needs pooling across
        many spike occurrences, not one ticker's one event)."""
        import random
        random.seed(3)
        dates = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]
        views = [100 + random.randint(-5, 5) for _ in range(n)]
        for s in spike_ats:
            views[s] = 800
        closes = [50.0]
        for i in range(1, n):
            drift = 0.002
            closes.append(closes[-1] * (1 + random.uniform(-drift, drift)))
        volumes = [1000 + random.randint(-50, 50) for _ in range(n)]
        for s in spike_ats:
            for i in range(s + 1, min(s + 4, n)):
                volumes[i] *= 5  # elevated forward volume after each spike
        return dates, views, closes, volumes

    def test_shape(self):
        dates, views, closes, volumes = self._synthetic_series()
        out = wag2.evaluate_ticker(dates, views, closes, volumes, horizons=(1, 3, 5))
        assert out["n_days"] == 400
        assert out["n_spike_days"] >= 5
        assert set(out["horizons"].keys()) == {1, 3, 5}
        for h, row in out["horizons"].items():
            assert "volume_ratio" in row
            assert "realized_vol" in row

    def test_engineered_spikes_show_elevated_forward_volume(self):
        dates, views, closes, volumes = self._synthetic_series()
        out = wag2.evaluate_ticker(dates, views, closes, volumes, horizons=(3,))
        row = out["horizons"][3]["volume_ratio"]
        assert row is not None
        assert row["mean"] > row["baseline_mean"]
        assert row["p_value"] < 0.05


class TestPoolGroup:
    def _two_ticker_per_ticker(self):
        # ticker A: clear elevated forward volume on spike days
        vol_spike_a = [5.0 + 0.01 * i for i in range(6)]
        vol_base_a = [1.0 + 0.01 * i for i in range(30)]
        # ticker B: no effect at all
        vol_spike_b = [1.0 + 0.01 * i for i in range(6)]
        vol_base_b = [1.0 + 0.01 * i for i in range(30)]
        return {
            "A": {"horizons": {3: {"_raw": {"vol_spike": vol_spike_a, "vol_base": vol_base_a,
                                             "rv_spike": [], "rv_base": []}}}},
            "B": {"horizons": {3: {"_raw": {"vol_spike": vol_spike_b, "vol_base": vol_base_b,
                                             "rv_spike": [], "rv_base": []}}}},
        }

    def test_pools_across_tickers(self):
        per_ticker = self._two_ticker_per_ticker()
        pooled = wag2.pool_group(per_ticker, ["A", "B"], [3])
        row = pooled[3]["volume_ratio"]
        assert row is not None
        assert row["n"] == 12  # 6 + 6
        assert row["n_baseline"] == 60  # 30 + 30
        assert pooled[3]["n_tickers_pooled"] == 2

    def test_missing_ticker_skipped_not_erroring(self):
        per_ticker = self._two_ticker_per_ticker()
        pooled = wag2.pool_group(per_ticker, ["A", "MISSING"], [3])
        assert pooled[3]["n_tickers_pooled"] == 1
        assert pooled[3]["volume_ratio"]["n"] == 6


class TestParseWikiResponse:
    def test_documented_shape(self):
        raw = {"items": [
            {"timestamp": "2026080100", "views": 4803},
            {"timestamp": "2026080200", "views": 4269},
        ]}
        out = wag2.parse_wiki_response(raw)
        assert out == {"2026-08-01": 4803, "2026-08-02": 4269}

    def test_malformed_items_dropped(self):
        raw = {"items": [
            {"timestamp": "2026080100", "views": 4803},
            {"timestamp": "bad", "views": 4269},
            {"timestamp": "2026080300"},  # no views
            {"timestamp": "2026080400", "views": "not-a-number"},
        ]}
        out = wag2.parse_wiki_response(raw)
        assert out == {"2026-08-01": 4803}

    def test_empty_or_missing_items(self):
        assert wag2.parse_wiki_response({}) == {}
        assert wag2.parse_wiki_response({"items": []}) == {}
        assert wag2.parse_wiki_response(None) == {}


class TestCapTierClassification:
    def test_mega_cap_set_matches_docstring(self):
        assert wag2.MEGA_CAP_TICKERS == ("NVDA", "AAPL", "TSLA", "AMD")

    def test_mega_cap_tickers_are_real_seed_tickers(self):
        articles = wag2._wiki_articles()
        for t in wag2.MEGA_CAP_TICKERS:
            assert t in articles
