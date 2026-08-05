"""
Regression tests for cftc_tff_tlt_disjoint_replication.py — the LADDER PATH
step (1) disjoint-window replication for the TLT leveraged-money-positioning
momentum candidate (open_questions.md, filed 2026-08-03). Pure orchestration
tests only: fetch_symbol_history_range and fetch_bars are injected/mocked,
no network calls. The statistical machinery itself (summarize,
hac_significance, compute_forward_returns) is already covered by
test_cftc_tff_gate2.py — these tests pin the wiring around it, not the math.
"""
import unittest
from unittest.mock import Mock, patch

from cftc_tff_tlt_disjoint_replication import WINDOW_END, run


def _bars(dates, closes):
    return {"date": dates, "close": closes, "open": closes, "high": closes,
            "low": closes, "volume": [0] * len(closes)}


class TestRun(unittest.TestCase):
    @patch("cftc_tff_tlt_disjoint_replication.fetch_symbol_history_range")
    def test_no_tff_data_short_circuits_before_any_bars_fetch(self, mock_fetch):
        mock_fetch.return_value = ([], 0)
        fetch_bars_fn = Mock()

        result = run(fetch_bars_fn)

        self.assertEqual(result, {"symbol": "TLT", "status": "no_tff_data"})
        fetch_bars_fn.assert_not_called()

    @patch("cftc_tff_tlt_disjoint_replication.fetch_symbol_history_range")
    def test_no_price_data_reports_status(self, mock_fetch):
        mock_fetch.return_value = (
            [{"report_date": "2020-09-01", "net_lev_money_pct_oi": 10.0}], 0,
        )

        def empty_bars(symbol, days):
            return {"date": []}

        result = run(empty_bars)
        self.assertEqual(result["status"], "no_price_data")

    @patch("cftc_tff_tlt_disjoint_replication.fetch_symbol_history_range")
    def test_ok_path_reports_window_bounds_and_delegates_stats(self, mock_fetch):
        records = [
            {"report_date": "2020-09-04", "net_lev_money_pct_oi": 5.0},
            {"report_date": "2020-09-11", "net_lev_money_pct_oi": -5.0},
        ]
        mock_fetch.return_value = (records, 2)

        dates = [f"2020-09-{d:02d}" for d in range(4, 30)]
        closes = [100.0 + i for i in range(len(dates))]

        def fake_fetch_bars(symbol, days):
            self.assertEqual(symbol, "TLT")
            self.assertGreater(days, 0)
            return _bars(dates, closes)

        result = run(fake_fetch_bars, window_end="2023-08-01")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["symbol"], "TLT")
        self.assertEqual(result["weeks"], 2)
        self.assertEqual(result["rejected_tff_records"], 2)
        self.assertEqual(result["window"], {"start": "2020-09-04", "end": "2020-09-11"})
        self.assertIn("summary", result)
        self.assertIn("significance", result)
        mock_fetch.assert_called_once()
        called_end_date = mock_fetch.call_args.args[1] if len(mock_fetch.call_args.args) > 1 \
            else mock_fetch.call_args.kwargs.get("end_date")
        self.assertEqual(called_end_date, "2023-08-01")

    def test_default_window_end_precedes_original_screen_start(self):
        # The original 7-symbol screen's window starts 2023-08 (experiments.md
        # 2026-08-03); this must stay strictly before it or the two windows
        # overlap and step (1) stops being a real out-of-sample test.
        self.assertEqual(WINDOW_END, "2023-08-01")


if __name__ == "__main__":
    unittest.main()
