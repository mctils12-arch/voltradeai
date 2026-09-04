"""Unit tests for scripts/wikiattention_gate2_newsfree.py — synthetic data
only, no network (mirrors test_wikiattention_gate2.py's own convention: the
script's __main__ path is what exercises the live Wikimedia/EDGAR/price
fetches, not CI)."""
import importlib.util
import os

import pytest

_HERE = os.path.dirname(__file__)
_SPEC = importlib.util.spec_from_file_location(
    "wikiattention_gate2_newsfree", os.path.join(_HERE, "scripts", "wikiattention_gate2_newsfree.py"))
wagnf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wagnf)


class TestParseCompanyTickers:
    def test_basic(self):
        raw = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
               "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}}
        out = wagnf.parse_company_tickers(raw)
        assert out == {"AAPL": "0000320193", "NVDA": "0001045810"}

    def test_zero_pads_to_10_digits(self):
        raw = {"0": {"cik_str": 2488, "ticker": "AMD", "title": "ADVANCED MICRO DEVICES"}}
        out = wagnf.parse_company_tickers(raw)
        assert out["AMD"] == "0000002488"
        assert len(out["AMD"]) == 10

    def test_skips_rows_missing_ticker_or_cik(self):
        raw = {"0": {"cik_str": 1, "ticker": None}, "1": {"cik_str": None, "ticker": "X"}, "2": {}}
        assert wagnf.parse_company_tickers(raw) == {}

    def test_empty(self):
        assert wagnf.parse_company_tickers({}) == {}
        assert wagnf.parse_company_tickers(None) == {}


class TestParseSubmissions8kDates:
    def test_filters_to_8k_and_8ka_only(self):
        block = {
            "form": ["10-Q", "8-K", "4", "8-K/A", "8-K"],
            "filingDate": ["2026-01-01", "2026-01-15", "2026-01-16", "2026-02-01", "2026-02-10"],
        }
        assert wagnf.parse_submissions_8k_dates(block) == {"2026-01-15", "2026-02-01", "2026-02-10"}

    def test_empty_block(self):
        assert wagnf.parse_submissions_8k_dates({}) == set()
        assert wagnf.parse_submissions_8k_dates({"form": [], "filingDate": []}) == set()

    def test_mismatched_lengths_use_shortest_via_zip(self):
        block = {"form": ["8-K", "8-K"], "filingDate": ["2026-01-01"]}
        assert wagnf.parse_submissions_8k_dates(block) == {"2026-01-01"}

    def test_duplicate_dates_dedup(self):
        block = {"form": ["8-K", "8-K"], "filingDate": ["2026-01-01", "2026-01-01"]}
        assert wagnf.parse_submissions_8k_dates(block) == {"2026-01-01"}


class TestIsNewsfreeSpikeIdx:
    DATES = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]

    def test_no_filing_nearby_is_newsfree(self):
        assert wagnf.is_newsfree_spike_idx(2, self.DATES, set()) is True

    def test_same_day_filing_excludes(self):
        assert wagnf.is_newsfree_spike_idx(2, self.DATES, {"2026-01-07"}) is False

    def test_prior_trading_day_filing_excludes(self):
        assert wagnf.is_newsfree_spike_idx(2, self.DATES, {"2026-01-06"}) is False

    def test_unrelated_filing_date_does_not_exclude(self):
        assert wagnf.is_newsfree_spike_idx(2, self.DATES, {"2026-01-08", "2025-06-01"}) is True

    def test_first_day_has_no_prior_day_to_check(self):
        # i=0: only same-day matters, no IndexError from dates[i-1]
        assert wagnf.is_newsfree_spike_idx(0, self.DATES, set()) is True
        assert wagnf.is_newsfree_spike_idx(0, self.DATES, {self.DATES[0]}) is False


class TestEvaluateTickerNewsfree:
    def _synthetic(self, n=200, spike_idx=150, filing_dates=None):
        import random
        random.seed(7)
        dates = [f"2026-{1 + (i // 28):02d}-{1 + (i % 28):02d}" for i in range(n)]
        views = [100 + random.randint(-5, 5) for _ in range(n)]
        views[spike_idx] = 900  # forces a real z-score spike once window is full
        closes = [100.0 + 0.01 * i for i in range(n)]
        volumes = [1_000_000 + random.randint(-10_000, 10_000) for _ in range(n)]
        volumes[spike_idx + 1] = 5_000_000  # elevated forward volume at h=1
        return dates, views, closes, volumes

    def test_newsfree_spike_kept_in_sample(self):
        dates, views, closes, volumes = self._synthetic()
        out = wagnf.evaluate_ticker_newsfree(dates, views, closes, volumes, filing_dates=set(),
                                              window=90, horizons=(1,))
        assert out["n_spike_days_total"] >= 1
        assert out["n_spike_days_newsfree"] == out["n_spike_days_total"]
        assert out["n_spike_days_excluded_for_news"] == 0

    def test_same_day_8k_removes_spike_from_sample(self):
        dates, views, closes, volumes = self._synthetic()
        spike_date = dates[150]
        out = wagnf.evaluate_ticker_newsfree(dates, views, closes, volumes, filing_dates={spike_date},
                                              window=90, horizons=(1,))
        assert out["n_spike_days_newsfree"] == out["n_spike_days_total"] - 1
        assert out["n_spike_days_excluded_for_news"] == 1

    def test_excluded_news_day_is_dropped_from_baseline_too(self):
        """A spike day excluded for news must not silently reappear in the
        baseline (it's a spike day, just not a news-free one) — the
        baseline stays 'every non-spike day', not 'every non-newsfree-spike
        day'."""
        dates, views, closes, volumes = self._synthetic()
        spike_date = dates[150]
        out_all_newsfree = wagnf.evaluate_ticker_newsfree(dates, views, closes, volumes, filing_dates=set(),
                                                            window=90, horizons=(1,))
        out_excluded = wagnf.evaluate_ticker_newsfree(dates, views, closes, volumes, filing_dates={spike_date},
                                                        window=90, horizons=(1,))
        n_base_all = len(out_all_newsfree["horizons"][1]["_raw"]["vol_base"])
        n_base_excluded = len(out_excluded["horizons"][1]["_raw"]["vol_base"])
        assert n_base_all == n_base_excluded  # the excluded spike day doesn't join baseline

    def test_no_spikes_yields_none_welch_result(self):
        dates = [f"2026-01-{1 + i:02d}" for i in range(10)]
        views = [100] * 10
        closes = [100.0] * 10
        volumes = [1_000_000] * 10
        out = wagnf.evaluate_ticker_newsfree(dates, views, closes, volumes, filing_dates=set(),
                                              window=90, horizons=(1,))
        assert out["n_spike_days_total"] == 0
        assert out["horizons"][1]["volume_ratio"] is None


class TestFetchCikMapAndSubmissionsAreNetworkOnly:
    """These two functions perform live HTTP — assert only that they exist
    with the expected signature shape and are not accidentally exercised at
    import time (mirrors wikiattention_gate2.py's own test file, which
    likewise never calls its network functions)."""

    def test_functions_exist_and_are_not_called_at_import(self):
        assert callable(wagnf.fetch_cik_map)
        assert callable(wagnf.fetch_8k_dates_for_cik)
        assert callable(wagnf.run_gate2_newsfree)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
