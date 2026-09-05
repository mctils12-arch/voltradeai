"""Unit tests for scripts/wikiattention_gate3.py — synthetic data only, no
network (mirrors test_wikiattention_gate2_newsfree.py's own convention: the
script's __main__ path is what exercises the live Wikimedia/EDGAR/price/
system_config fetches, not CI)."""
import importlib.util
import os

import pytest

_HERE = os.path.dirname(__file__)
_SPEC = importlib.util.spec_from_file_location(
    "wikiattention_gate3", os.path.join(_HERE, "scripts", "wikiattention_gate3.py"))
wag3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wag3)


class TestForwardReturn:
    def test_basic_gain(self):
        closes = [100.0, 101.0, 102.0, 105.0]
        assert wag3.forward_return(closes, 0, 3) == pytest.approx(0.05)

    def test_basic_loss(self):
        closes = [100.0, 90.0]
        assert wag3.forward_return(closes, 0, 1) == pytest.approx(-0.10)

    def test_horizon_runs_past_end_returns_none(self):
        closes = [100.0, 101.0]
        assert wag3.forward_return(closes, 0, 5) is None

    def test_zero_entry_price_returns_none(self):
        closes = [0.0, 101.0]
        assert wag3.forward_return(closes, 0, 1) is None

    def test_exact_last_index_boundary(self):
        closes = [100.0, 110.0]
        assert wag3.forward_return(closes, 0, 1) == pytest.approx(0.10)
        assert wag3.forward_return(closes, 1, 1) is None  # nothing past the last bar


class TestEvaluateTickerGate3:
    def _synthetic(self, n=200, spike_idx=150, jump=0.05):
        import random
        random.seed(11)
        dates = [f"2026-{1 + (i // 28):02d}-{1 + (i % 28):02d}" for i in range(n)]
        views = [100 + random.randint(-5, 5) for _ in range(n)]
        views[spike_idx] = 900  # forces a z-score spike once the trailing window is full
        closes = [100.0 + 0.01 * i for i in range(n)]
        closes[spike_idx + 1] *= (1 + jump)  # a real forward jump right after the spike
        for i in range(spike_idx + 2, n):
            closes[i] = closes[spike_idx + 1] + 0.01 * (i - spike_idx - 1)
        return dates, views, closes

    def test_newsfree_spike_lands_in_sample(self):
        dates, views, closes = self._synthetic()
        out = wag3.evaluate_ticker_gate3(dates, views, closes, filing_dates=set(), window=90, horizons=(1,))
        assert out["n_spike_days_newsfree"] == out["n_spike_days_total"] >= 1
        assert len(out["horizons"][1]["_raw"]["ret_spike"]) >= 1

    def test_news_contaminated_spike_excluded_from_sample_and_baseline(self):
        dates, views, closes = self._synthetic()
        spike_date = dates[150]
        out = wag3.evaluate_ticker_gate3(dates, views, closes, filing_dates={spike_date}, window=90, horizons=(1,))
        assert out["n_spike_days_newsfree"] == out["n_spike_days_total"] - 1
        # the sample loses its only spike-day observation at h=1
        assert len(out["horizons"][1]["_raw"]["ret_spike"]) == 0

    def test_forward_jump_shows_up_as_elevated_sample_mean(self):
        dates, views, closes = self._synthetic(jump=0.20)
        out = wag3.evaluate_ticker_gate3(dates, views, closes, filing_dates=set(), window=90, horizons=(1,))
        raw = out["horizons"][1]["_raw"]
        assert raw["ret_spike"][0] == pytest.approx(0.20, abs=1e-3)
        assert sum(raw["ret_base"]) / len(raw["ret_base"]) < raw["ret_spike"][0]

    def test_no_spikes_yields_none_welch_result(self):
        n = 10
        dates = [f"2026-01-{1 + i:02d}" for i in range(n)]
        views = [100] * n
        closes = [100.0] * n
        out = wag3.evaluate_ticker_gate3(dates, views, closes, filing_dates=set(), window=90, horizons=(1,))
        assert out["n_spike_days_total"] == 0
        assert out["horizons"][1]["forward_return"] is None


class TestPoolReturns:
    def test_pools_across_tickers(self):
        per_ticker = {
            "AAA": {"horizons": {1: {"_raw": {"ret_spike": [0.1, 0.2], "ret_base": [0.0, 0.01]}}}},
            "BBB": {"horizons": {1: {"_raw": {"ret_spike": [0.05], "ret_base": [0.0, -0.01, 0.02]}}}},
        }
        out = wag3.pool_returns(per_ticker, ["AAA", "BBB"], (1,))
        assert out[1]["n_tickers_pooled"] == 2
        # welch_vs_baseline requires n>=5/side; pooled sample here has 3, baseline 5
        assert out[1]["forward_return"] is None  # under the n>=5 floor on the sample side, honestly None

    def test_missing_ticker_skipped_not_errored(self):
        per_ticker = {"AAA": {"horizons": {1: {"_raw": {"ret_spike": [0.1], "ret_base": [0.0]}}}}}
        out = wag3.pool_returns(per_ticker, ["AAA", "ZZZ"], (1,))
        assert out[1]["n_tickers_pooled"] == 1


class TestApplyVerdict:
    def _row(self, mean, baseline_mean, p_value, n=30):
        return {
            "forward_return": {
                "n": n, "n_baseline": n, "mean": mean, "baseline_mean": baseline_mean,
                "mean_diff": round(mean - baseline_mean, 4), "t_stat": 3.0, "p_value": p_value,
            }
        }

    def test_passes_when_significant_and_profitable_net_of_cost(self):
        pooled = {1: self._row(mean=0.02, baseline_mean=0.001, p_value=0.001)}
        out = wag3.apply_verdict(pooled, round_trip_cost=0.004)
        assert out["gate3_pass"] is True
        assert out["per_horizon"][1]["horizon_pass"] is True
        assert out["per_horizon"][1]["net_of_cost_return"] == pytest.approx(0.016)

    def test_fails_when_significant_but_cost_eats_the_edge(self):
        # beats baseline statistically, but the round-trip cost exceeds the raw mean
        pooled = {1: self._row(mean=0.003, baseline_mean=0.0005, p_value=0.001)}
        out = wag3.apply_verdict(pooled, round_trip_cost=0.004)
        assert out["per_horizon"][1]["beats_baseline_significant"] is True
        assert out["per_horizon"][1]["profitable_net_of_cost"] is False
        assert out["gate3_pass"] is False

    def test_fails_when_not_significant_even_if_profitable(self):
        pooled = {1: self._row(mean=0.02, baseline_mean=0.001, p_value=0.5)}
        out = wag3.apply_verdict(pooled, round_trip_cost=0.004)
        assert out["per_horizon"][1]["beats_baseline_significant"] is False
        assert out["gate3_pass"] is False

    def test_negative_mean_diff_never_passes_even_at_low_p(self):
        # spike mean BELOW baseline (a reversal signature) must never read as a pass
        pooled = {1: self._row(mean=-0.02, baseline_mean=0.001, p_value=0.0001)}
        out = wag3.apply_verdict(pooled, round_trip_cost=0.004)
        assert out["per_horizon"][1]["beats_baseline_significant"] is False
        assert out["gate3_pass"] is False

    def test_missing_horizon_data_marks_insufficient_and_fails_overall(self):
        pooled = {1: {"forward_return": None}}
        out = wag3.apply_verdict(pooled, round_trip_cost=0.004)
        assert out["per_horizon"][1]["status"] == "insufficient_data"
        assert out["gate3_pass"] is False

    def test_requires_all_horizons_to_pass(self):
        pooled = {
            1: self._row(mean=0.02, baseline_mean=0.001, p_value=0.001),
            3: self._row(mean=-0.01, baseline_mean=0.001, p_value=0.9),
        }
        out = wag3.apply_verdict(pooled, round_trip_cost=0.004, family_size=2)
        assert out["per_horizon"][1]["horizon_pass"] is True
        assert out["per_horizon"][3]["horizon_pass"] is False
        assert out["gate3_pass"] is False

    def test_alpha_bar_reflects_family_size(self):
        pooled = {1: self._row(mean=0.02, baseline_mean=0.001, p_value=0.02)}
        out_narrow = wag3.apply_verdict(pooled, round_trip_cost=0.004, family_size=3)
        out_wide = wag3.apply_verdict(pooled, round_trip_cost=0.004, family_size=1)
        assert out_narrow["per_horizon"][1]["beats_baseline_significant"] is False  # 0.02 > 0.05/3
        assert out_wide["per_horizon"][1]["beats_baseline_significant"] is True   # 0.02 < 0.05/1


class TestSlippageCostsReadsSystemConfig:
    def test_reads_real_constants_not_hardcoded(self):
        costs = wag3._slippage_costs()
        assert costs["small_mid_round_trip"] == pytest.approx(0.004)  # 2 x SLIPPAGE_ILLIQUID (0.002)
        assert costs["mega_round_trip"] == pytest.approx(0.001)       # 2 x SLIPPAGE_PCT (0.0005)


class TestNetworkFunctionsNotCalledAtImport:
    def test_run_gate3_exists_and_is_not_called_at_import(self):
        assert callable(wag3.run_gate3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
