"""
Regression tests for scripts/reconstruct_position_pnl.py — the independent,
published-market-close P&L reconstruction built for KNOWN BROKEN #42
(research/open_questions.md): does a book's own actual market performance on
a given day support the account's reported same-day pnl, or point to a data
anomaly? Pure-function tests only: backtest_v2.fetch_bars is monkeypatched,
no live network calls.
"""
import importlib.util
import os
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "reconstruct_position_pnl",
    os.path.join(os.path.dirname(__file__), "scripts", "reconstruct_position_pnl.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

is_option_symbol = _mod.is_option_symbol
load_positions = _mod.load_positions
reconstruct = _mod.reconstruct


def _bars(dates, closes):
    return {"date": dates, "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [1_000_000] * len(dates)}


class TestIsOptionSymbol(unittest.TestCase):
    def test_occ_option_detected(self):
        self.assertTrue(is_option_symbol("BAC261016P00057500"))
        self.assertTrue(is_option_symbol("HPE261016P00045000"))

    def test_equity_symbol_not_option(self):
        for sym in ("QQQ", "KWEB", "FCEL", "SMH", "VXUS", "BRK.B".replace(".", "")):
            self.assertFalse(is_option_symbol(sym))

    def test_short_root_option(self):
        # single-letter root, matches the real-world F (Ford) options shape
        self.assertTrue(is_option_symbol("F261016C00012000"))


class TestLoadPositions(unittest.TestCase):
    def test_plain_map(self):
        self.assertEqual(load_positions('{"QQQ": 51, "FCEL": 70}', None), {"QQQ": 51.0, "FCEL": 70.0})

    def test_positions_detail_payload(self):
        payload = '{"probe": "positions-detail", "positions": [{"symbol": "QQQ", "qty": "51"}, {"symbol": "FCEL", "qty": "70"}]}'
        self.assertEqual(load_positions(payload, None), {"QQQ": 51.0, "FCEL": 70.0})

    def test_positions_file(self):
        import json
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"SMH": 20}, f)
            path = f.name
        try:
            self.assertEqual(load_positions(None, path), {"SMH": 20.0})
        finally:
            os.unlink(path)


class TestReconstruct(unittest.TestCase):
    def test_sums_close_to_close_contribution(self):
        # QQQ: 51 shares, 718.36 -> 716.31 (-2.05) = -104.55
        # SMH: 20 shares, 573.73 -> 574.29 (+0.56) = +11.20
        fake_bars = {
            "QQQ": _bars(["2026-09-04", "2026-09-08", "2026-09-09"], [718.96, 718.36, 716.31]),
            "SMH": _bars(["2026-09-04", "2026-09-08", "2026-09-09"], [571.10, 573.73, 574.29]),
        }
        with patch.object(_mod.backtest_v2, "fetch_bars", side_effect=lambda sym, days, use_cache=True: fake_bars[sym]):
            result = reconstruct("2026-09-09", {"QQQ": 51, "SMH": 20})
        self.assertAlmostEqual(result["reconstructed_pnl"], -104.55 + 11.20, places=2)
        self.assertEqual(len(result["legs"]), 2)
        self.assertEqual(result["excluded_options"], [])
        self.assertEqual(result["excluded_no_data"], [])

    def test_excludes_option_legs_without_fetching_bars(self):
        with patch.object(_mod.backtest_v2, "fetch_bars") as mock_fetch:
            result = reconstruct("2026-09-09", {"BAC261016P00057500": -1})
        mock_fetch.assert_not_called()
        self.assertEqual(result["excluded_options"], ["BAC261016P00057500"])
        self.assertEqual(result["reconstructed_pnl"], 0.0)

    def test_reports_missing_date_without_crashing(self):
        fake_bars = _bars(["2026-09-04", "2026-09-08"], [718.96, 718.36])
        with patch.object(_mod.backtest_v2, "fetch_bars", return_value=fake_bars):
            result = reconstruct("2026-09-09", {"QQQ": 51})
        self.assertEqual(result["reconstructed_pnl"], 0.0)
        self.assertEqual(len(result["excluded_no_data"]), 1)
        self.assertEqual(result["excluded_no_data"][0]["symbol"], "QQQ")

    def test_reports_missing_prior_day_without_crashing(self):
        # target date is the very first bar in the lookback window: no prior
        # close exists to diff against, must exclude rather than divide by
        # a phantom baseline.
        fake_bars = _bars(["2026-09-09", "2026-09-10"], [716.31, 710.36])
        with patch.object(_mod.backtest_v2, "fetch_bars", return_value=fake_bars):
            result = reconstruct("2026-09-09", {"QQQ": 51})
        self.assertEqual(result["reconstructed_pnl"], 0.0)
        self.assertEqual(result["excluded_no_data"][0]["reason"], "no prior-day bar in lookback window")

    def test_fetch_exception_excluded_not_raised(self):
        with patch.object(_mod.backtest_v2, "fetch_bars", side_effect=RuntimeError("network down")):
            result = reconstruct("2026-09-09", {"QQQ": 51})
        self.assertEqual(result["reconstructed_pnl"], 0.0)
        self.assertIn("network down", result["excluded_no_data"][0]["reason"])

    def test_short_position_contribution_sign(self):
        # a short leg's qty is negative in the live positions-detail payload;
        # a price DROP should be a POSITIVE contribution to a short.
        fake_bars = _bars(["2026-09-08", "2026-09-09"], [10.0, 9.0])
        with patch.object(_mod.backtest_v2, "fetch_bars", return_value=fake_bars):
            result = reconstruct("2026-09-09", {"XYZ": -100})
        self.assertAlmostEqual(result["reconstructed_pnl"], 100.0, places=2)


if __name__ == "__main__":
    unittest.main()
