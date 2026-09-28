"""Unit tests for scripts/wikiattention_gate3_momentum.py — synthetic data
only, no network (mirrors test_wikiattention_gate3.py's own convention: the
script's __main__/run_momentum_gate3 path is what exercises the live
Wikimedia/EDGAR/price/system_config fetches, not CI)."""
import importlib.util
import os

import pytest

_HERE = os.path.dirname(__file__)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wagm = _load("wikiattention_gate3_momentum", "wikiattention_gate3_momentum.py")


class TestSameDayReturn:
    def test_basic_up(self):
        closes = [100.0, 105.0]
        assert wagm.same_day_return(closes, 1) == pytest.approx(0.05)

    def test_basic_down(self):
        closes = [100.0, 90.0]
        assert wagm.same_day_return(closes, 1) == pytest.approx(-0.10)

    def test_idx_zero_returns_none(self):
        closes = [100.0, 105.0]
        assert wagm.same_day_return(closes, 0) is None

    def test_zero_prior_close_returns_none(self):
        closes = [0.0, 105.0]
        assert wagm.same_day_return(closes, 1) is None


class TestEvaluateTickerMomentum:
    def _synthetic(self, n=250, spike_idx=150, same_day_move=0.05, forward_move=0.03):
        import random
        random.seed(7)
        dates = [f"2026-{1 + (i // 28):02d}-{1 + (i % 28):02d}" for i in range(n)]
        views = [100 + random.randint(-5, 5) for _ in range(n)]
        views[spike_idx] = 900  # forces a z-score spike once the trailing window is full
        closes = [100.0 + 0.001 * i for i in range(n)]
        # spike day itself moves same_day_move relative to the prior close
        closes[spike_idx] = closes[spike_idx - 1] * (1 + same_day_move)
        # the day after continues (or reverses) by forward_move
        closes[spike_idx + 1] = closes[spike_idx] * (1 + forward_move)
        for i in range(spike_idx + 2, n):
            closes[i] = closes[spike_idx + 1] + 0.001 * (i - spike_idx - 1)
        return dates, views, closes

    def test_up_spike_lands_in_up_bucket(self):
        dates, views, closes = self._synthetic(same_day_move=0.05)
        out = wagm.evaluate_ticker_momentum(dates, views, closes, filing_dates=set(), window=90, horizons=(1,))
        assert out["n_up"] == 1
        assert out["n_down"] == 0
        assert len(out["horizons"][1]["_raw"]["ret_up"]) == 1
        assert len(out["horizons"][1]["_raw"]["ret_down"]) == 0

    def test_down_spike_lands_in_down_bucket(self):
        dates, views, closes = self._synthetic(same_day_move=-0.05)
        out = wagm.evaluate_ticker_momentum(dates, views, closes, filing_dates=set(), window=90, horizons=(1,))
        assert out["n_up"] == 0
        assert out["n_down"] == 1

    def test_news_contaminated_spike_excluded_from_both_buckets(self):
        dates, views, closes = self._synthetic(same_day_move=0.05)
        spike_date = dates[150]
        out = wagm.evaluate_ticker_momentum(dates, views, closes, filing_dates={spike_date}, window=90, horizons=(1,))
        assert out["n_up"] == 0
        assert out["n_down"] == 0

    def test_momentum_shows_up_as_positive_forward_return_in_up_bucket(self):
        dates, views, closes = self._synthetic(same_day_move=0.05, forward_move=0.03)
        out = wagm.evaluate_ticker_momentum(dates, views, closes, filing_dates=set(), window=90, horizons=(1,))
        assert out["horizons"][1]["_raw"]["ret_up"][0] == pytest.approx(0.03, abs=1e-3)

    def test_no_spikes_yields_zero_buckets(self):
        n = 10
        dates = [f"2026-01-{1 + i:02d}" for i in range(n)]
        views = [100] * n
        closes = [100.0] * n
        out = wagm.evaluate_ticker_momentum(dates, views, closes, filing_dates=set(), window=90, horizons=(1,))
        assert out["n_up"] == 0 and out["n_down"] == 0
        assert out["horizons"][1]["up_vs_down"] is None


class TestPoolBuckets:
    def test_pools_across_tickers(self):
        per_ticker = {
            "AAA": {"horizons": {1: {"_raw": {"ret_up": [0.1, 0.2], "ret_down": [-0.05]}}}},
            "BBB": {"horizons": {1: {"_raw": {"ret_up": [0.05], "ret_down": [-0.02, -0.01]}}}},
        }
        out = wagm.pool_buckets(per_ticker, ["AAA", "BBB"], (1,))
        assert out[1]["n_up"] == 3
        assert out[1]["n_down"] == 3
        assert out[1]["n_tickers_pooled"] == 2

    def test_missing_ticker_skipped_not_errored(self):
        per_ticker = {"AAA": {"horizons": {1: {"_raw": {"ret_up": [0.1], "ret_down": [-0.1]}}}}}
        out = wagm.pool_buckets(per_ticker, ["AAA", "ZZZ"], (1,))
        assert out[1]["n_tickers_pooled"] == 1


class TestApplyMomentumVerdict:
    def _pooled(self, n_up, n_down, mean_up, mean_down, p_value):
        welch = None
        if n_up >= 5 and n_down >= 5:
            welch = {
                "n": n_up, "n_baseline": n_down, "mean": mean_up, "baseline_mean": mean_down,
                "mean_diff": round(mean_up - mean_down, 4), "t_stat": 3.0, "p_value": p_value,
            }
        return {5: {"up_vs_down": welch, "n_up": n_up, "n_down": n_down, "n_tickers_pooled": 3}}

    def test_insufficient_data_below_min_n(self):
        pooled = self._pooled(n_up=10, n_down=10, mean_up=0.02, mean_down=-0.01, p_value=0.001)
        out = wagm.apply_momentum_verdict(pooled, round_trip_cost=0.004, min_n_per_bucket=15)
        assert out["primary_result"]["status"] == "insufficient_data"
        assert out["gate3_pass"] is False

    def test_momentum_classification_when_up_mean_exceeds_down_mean(self):
        pooled = self._pooled(n_up=20, n_down=20, mean_up=0.02, mean_down=-0.01, p_value=0.001)
        out = wagm.apply_momentum_verdict(pooled, round_trip_cost=0.001, min_n_per_bucket=15)
        assert out["per_horizon"][5]["classification"] == "momentum"

    def test_reversal_classification_when_down_mean_exceeds_up_mean(self):
        pooled = self._pooled(n_up=20, n_down=20, mean_up=-0.01, mean_down=0.02, p_value=0.001)
        out = wagm.apply_momentum_verdict(pooled, round_trip_cost=0.001, min_n_per_bucket=15)
        assert out["per_horizon"][5]["classification"] == "reversal"

    def test_not_significant_never_passes(self):
        pooled = self._pooled(n_up=20, n_down=20, mean_up=0.02, mean_down=-0.01, p_value=0.5)
        out = wagm.apply_momentum_verdict(pooled, round_trip_cost=0.001, min_n_per_bucket=15)
        assert out["per_horizon"][5]["significant"] is False
        assert out["gate3_pass"] is False

    def test_significant_but_cost_eats_the_edge_never_passes(self):
        # long leg 0.006, short leg -(-0.001)=0.001 -> pair gross 0.007, but
        # 2 legs x round_trip_cost=0.01 = 0.02 wipes it out
        pooled = self._pooled(n_up=20, n_down=20, mean_up=0.006, mean_down=-0.001, p_value=0.0001)
        out = wagm.apply_momentum_verdict(pooled, round_trip_cost=0.01, min_n_per_bucket=15)
        assert out["per_horizon"][5]["significant"] is True
        assert out["per_horizon"][5]["pair_profitable_net_of_cost"] is False
        assert out["gate3_pass"] is False

    def test_significant_and_profitable_passes(self):
        pooled = self._pooled(n_up=20, n_down=20, mean_up=0.05, mean_down=-0.05, p_value=0.0001)
        out = wagm.apply_momentum_verdict(pooled, round_trip_cost=0.004, min_n_per_bucket=15)
        assert out["per_horizon"][5]["horizon_pass"] is True
        assert out["gate3_pass"] is True

    def test_alpha_bar_reflects_family_size(self):
        pooled = self._pooled(n_up=20, n_down=20, mean_up=0.05, mean_down=-0.05, p_value=0.02)
        out_narrow = wagm.apply_momentum_verdict(pooled, round_trip_cost=0.004, min_n_per_bucket=15, family_size=3)
        out_wide = wagm.apply_momentum_verdict(pooled, round_trip_cost=0.004, min_n_per_bucket=15, family_size=1)
        assert out_narrow["per_horizon"][5]["significant"] is False  # 0.02 > 0.05/3
        assert out_wide["per_horizon"][5]["significant"] is True    # 0.02 < 0.05/1

    def test_primary_horizon_selects_the_pre_registered_horizon(self):
        pooled = {
            1: {"up_vs_down": {"n": 20, "n_baseline": 20, "mean": 0.05, "baseline_mean": -0.05,
                                "mean_diff": 0.10, "t_stat": 3.0, "p_value": 0.9},  # not significant at h=1
                "n_up": 20, "n_down": 20, "n_tickers_pooled": 3},
            5: {"up_vs_down": {"n": 20, "n_baseline": 20, "mean": 0.05, "baseline_mean": -0.05,
                                "mean_diff": 0.10, "t_stat": 3.0, "p_value": 0.0001},  # significant at h=5
                "n_up": 20, "n_down": 20, "n_tickers_pooled": 3},
        }
        out = wagm.apply_momentum_verdict(pooled, round_trip_cost=0.004, min_n_per_bucket=15, primary_horizon=5)
        assert out["primary_horizon"] == 5
        assert out["gate3_pass"] is True  # judged on h=5 only, not h=1


class TestNetworkFunctionsNotCalledAtImport:
    def test_run_momentum_gate3_exists_and_is_not_called_at_import(self):
        assert callable(wagm.run_momentum_gate3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
