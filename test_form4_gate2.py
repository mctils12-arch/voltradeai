"""
Regression tests for form4_gate2_test.py — the ROOT VALIDATION LADDER gate 2
(SIGNAL) screen for the SEC bulk-historical Form 4 archive. Pure-function
tests only: no network calls, no dependency on sec_form4_bulk's live SEC
fetch or backtest_v2's Alpaca/Yahoo fetch.
"""
import unittest

from form4_gate2_test import (
    build_events,
    compute_forward_returns,
    compute_ticker_baseline,
    find_entry_index,
    select_tickers,
    significance,
    summarize,
)


def _rec(ticker, filing_date, owner_cik, dollar_value=10000.0, trans_code="P"):
    return {"ticker": ticker, "filing_date": filing_date, "owner_cik": owner_cik,
            "dollar_value": dollar_value, "trans_code": trans_code}


class TestFindEntryIndex(unittest.TestCase):
    def test_finds_first_bar_strictly_after_filing_date(self):
        dates = ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"]
        self.assertEqual(find_entry_index(dates, "2026-01-02"), 2)

    def test_no_bar_after_filing_date_returns_none(self):
        dates = ["2026-01-01", "2026-01-02"]
        self.assertIsNone(find_entry_index(dates, "2026-01-02"))

    def test_never_returns_filing_date_itself_no_lookahead(self):
        dates = ["2026-01-02", "2026-01-03"]
        self.assertEqual(find_entry_index(dates, "2026-01-02"), 1)


class TestBuildEvents(unittest.TestCase):
    def test_single_isolated_buy_is_not_a_cluster(self):
        records = [_rec("AAA", "2026-01-01", "owner1")]
        events = build_events(records)
        self.assertEqual(len(events["AAA"]), 1)
        self.assertFalse(events["AAA"][0]["cluster"])

    def test_two_distinct_owners_same_day_is_a_cluster(self):
        records = [_rec("AAA", "2026-01-01", "owner1"), _rec("AAA", "2026-01-01", "owner2")]
        events = build_events(records)
        self.assertEqual(len(events["AAA"]), 1)  # same-day collapses to one event
        self.assertTrue(events["AAA"][0]["cluster"])
        self.assertEqual(events["AAA"][0]["n_owners_same_day"], 2)

    def test_two_distinct_owners_within_window_is_a_cluster(self):
        records = [_rec("AAA", "2026-01-01", "owner1"), _rec("AAA", "2026-01-04", "owner2")]
        events = build_events(records)
        dates = {e["filing_date"]: e for e in events["AAA"]}
        self.assertFalse(dates["2026-01-01"]["cluster"])  # nothing preceded it
        self.assertTrue(dates["2026-01-04"]["cluster"])   # owner1's buy is within 5 days back

    def test_two_distinct_owners_outside_window_is_not_a_cluster(self):
        records = [_rec("AAA", "2026-01-01", "owner1"), _rec("AAA", "2026-01-20", "owner2")]
        events = build_events(records)
        dates = {e["filing_date"]: e for e in events["AAA"]}
        self.assertFalse(dates["2026-01-20"]["cluster"])

    def test_same_owner_repeated_buys_do_not_self_cluster(self):
        # Same owner buying twice within the window is not "2 insiders
        # agree" — it's one insider dollar-cost-averaging.
        records = [_rec("AAA", "2026-01-01", "owner1"), _rec("AAA", "2026-01-03", "owner1")]
        events = build_events(records)
        for e in events["AAA"]:
            self.assertFalse(e["cluster"])

    def test_never_uses_a_later_event_to_flag_an_earlier_one(self):
        # A buy on day 1 cannot become "clustered" because of a buy on day
        # 4 — clustering must be strictly backward-looking (no lookahead).
        records = [_rec("AAA", "2026-01-01", "owner1"), _rec("AAA", "2026-01-04", "owner2")]
        events = build_events(records)
        dates = {e["filing_date"]: e for e in events["AAA"]}
        self.assertFalse(dates["2026-01-01"]["cluster"])

    def test_wrong_trans_code_excluded_by_default(self):
        records = [_rec("AAA", "2026-01-01", "owner1", trans_code="S")]
        events = build_events(records, trans_code="P")
        self.assertEqual(events, {})

    def test_dollar_value_summed_across_same_day_owners(self):
        records = [_rec("AAA", "2026-01-01", "owner1", dollar_value=5000.0),
                   _rec("AAA", "2026-01-01", "owner2", dollar_value=7000.0)]
        events = build_events(records)
        self.assertEqual(events["AAA"][0]["dollar_value"], 12000.0)

    def test_events_sorted_chronologically_per_ticker(self):
        records = [_rec("AAA", "2026-02-01", "owner1"), _rec("AAA", "2026-01-01", "owner2")]
        events = build_events(records)
        self.assertEqual([e["filing_date"] for e in events["AAA"]], ["2026-01-01", "2026-02-01"])


class TestSelectTickers(unittest.TestCase):
    def test_single_ticker_slot_goes_to_cluster_when_only_one_available(self):
        events_by_ticker = {
            "SINGLE_BIG": [{"cluster": False, "dollar_value": 1_000_000}],
            "CLUSTER_SMALL": [{"cluster": True, "dollar_value": 100}],
        }
        selected = select_tickers(events_by_ticker, max_tickers=1)
        self.assertEqual(selected, ["CLUSTER_SMALL"])

    def test_cap_respected(self):
        events_by_ticker = {f"T{i}": [{"cluster": False, "dollar_value": i}] for i in range(10)}
        selected = select_tickers(events_by_ticker, max_tickers=3)
        self.assertEqual(len(selected), 3)

    def test_single_tickers_ranked_by_dollar_value_desc(self):
        events_by_ticker = {
            "LOW": [{"cluster": False, "dollar_value": 100}],
            "HIGH": [{"cluster": False, "dollar_value": 999}],
            "MID": [{"cluster": False, "dollar_value": 500}],
        }
        selected = select_tickers(events_by_ticker, max_tickers=3)
        self.assertEqual(set(selected), {"HIGH", "MID", "LOW"})

    def test_deterministic_across_repeated_calls(self):
        events_by_ticker = {f"T{i}": [{"cluster": i % 3 == 0, "dollar_value": i}] for i in range(20)}
        self.assertEqual(
            select_tickers(events_by_ticker, max_tickers=10),
            select_tickers(events_by_ticker, max_tickers=10),
        )

    def test_single_bucket_not_starved_when_cluster_pool_exceeds_cap(self):
        # Regression for the live-found selection bug (2026-07-19): when
        # cluster-eligible tickers alone exceed max_tickers, a naive
        # "cluster first" order gives the single bucket ZERO tickers,
        # silently corrupting what "single" means downstream. The split
        # must guarantee both buckets get a share.
        events_by_ticker = {}
        for i in range(20):
            events_by_ticker[f"C{i}"] = [{"cluster": True, "dollar_value": 100 - i}]
        for i in range(20):
            events_by_ticker[f"S{i}"] = [{"cluster": False, "dollar_value": 100 - i}]
        selected = select_tickers(events_by_ticker, max_tickers=10)
        n_cluster = sum(1 for t in selected if t.startswith("C"))
        n_single = sum(1 for t in selected if t.startswith("S"))
        self.assertGreater(n_single, 0)
        self.assertEqual(n_cluster + n_single, 10)

    def test_full_budget_used_when_one_bucket_is_scarce(self):
        events_by_ticker = {"ONLY_CLUSTER": [{"cluster": True, "dollar_value": 100}]}
        for i in range(20):
            events_by_ticker[f"S{i}"] = [{"cluster": False, "dollar_value": 100 - i}]
        selected = select_tickers(events_by_ticker, max_tickers=10)
        self.assertEqual(len(selected), 10)  # scarce cluster side doesn't waste budget


class TestComputeForwardReturns(unittest.TestCase):
    def test_basic_forward_return(self):
        bars = {"date": ["2026-01-01", "2026-01-02", "2026-01-03"], "close": [10.0, 11.0, 12.0]}
        events = [{"ticker": "AAA", "filing_date": "2026-01-01", "cluster": False}]
        # entry at idx 1 (11.0); horizon-1 return not in HORIZONS but exercise the mechanism
        rows = compute_forward_returns(events, bars)
        self.assertEqual(rows[0]["entry_date"], "2026-01-02")
        self.assertEqual(rows[0]["entry_idx"], 1)

    def test_no_bar_after_filing_date_leaves_entry_none(self):
        bars = {"date": ["2026-01-01"], "close": [10.0]}
        events = [{"ticker": "AAA", "filing_date": "2026-01-01", "cluster": False}]
        rows = compute_forward_returns(events, bars)
        self.assertIsNone(rows[0]["entry_idx"])
        self.assertEqual(rows[0]["forward_returns"], {})

    def test_horizon_beyond_available_bars_is_dropped_not_zero_filled(self):
        import form4_gate2_test as mod
        dates = [f"2026-01-{d:02d}" for d in range(1, 11)]
        closes = [float(i) for i in range(1, 11)]
        bars = {"date": dates, "close": closes}
        events = [{"ticker": "AAA", "filing_date": "2026-01-01", "cluster": False}]
        rows = compute_forward_returns(events, bars)
        for h in mod.HORIZONS:
            self.assertNotIn(h, rows[0]["forward_returns"])


class TestComputeTickerBaseline(unittest.TestCase):
    def test_baseline_excludes_event_adjacent_windows(self):
        import form4_gate2_test as mod
        n = 200
        bars = {"date": [f"d{i}" for i in range(n)], "close": [100.0 + i for i in range(n)]}
        event_idx = 100
        baseline = compute_ticker_baseline(bars, [event_idx])
        for h in mod.HORIZONS:
            # entries within h of the event on either side must be excluded
            excluded_range = range(max(0, event_idx - h), min(n, event_idx + h + 1))
            self.assertEqual(len(baseline[h]), max(0, n - h) - len(set(excluded_range) & set(range(n - h))))

    def test_no_events_uses_full_series(self):
        import form4_gate2_test as mod
        n = 100
        bars = {"date": [f"d{i}" for i in range(n)], "close": [100.0 + i for i in range(n)]}
        baseline = compute_ticker_baseline(bars, [])
        for h in mod.HORIZONS:
            self.assertEqual(len(baseline[h]), n - h)


class TestSummarizeAndSignificance(unittest.TestCase):
    def test_summarize_separates_buckets(self):
        import form4_gate2_test as mod
        rows = [
            {"cluster": True, "forward_returns": {20: 0.10}},
            {"cluster": False, "forward_returns": {20: 0.01}},
        ]
        baseline = {20: [0.0, 0.02], 60: []}
        summary = summarize(rows, baseline)
        self.assertEqual(summary["20"]["cluster"]["n"], 1)
        self.assertAlmostEqual(summary["20"]["cluster"]["mean_pct"], 10.0)
        self.assertEqual(summary["20"]["single"]["n"], 1)
        self.assertEqual(summary["20"]["baseline"]["n"], 2)

    def test_significance_returns_none_for_thin_samples(self):
        rows = [{"cluster": True, "forward_returns": {20: 0.10}}]
        baseline = {20: [0.01, 0.02], 60: []}
        sig = significance(rows, baseline)
        self.assertIsNone(sig["20"]["cluster"])

    def test_significance_computes_for_adequate_samples(self):
        rows = [{"cluster": True, "forward_returns": {20: 0.20}} for _ in range(10)]
        baseline = {20: [0.0] * 10, 60: []}
        sig = significance(rows, baseline)
        self.assertIsNotNone(sig["20"]["cluster"])
        self.assertEqual(sig["20"]["cluster"]["n"], 10)
        self.assertAlmostEqual(sig["20"]["cluster"]["mean_diff_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()
