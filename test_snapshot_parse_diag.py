"""
REPAIR 2026-07-06 — regression test for the Tier2 scan-failure blind spot.

_fetch_snap's old `except Exception: return {}` (and its silent skip of any
non-per-symbol response body) meant a genuine Alpaca snapshot-API problem
produced the exact same uninformative "Could not fetch market data from
Alpaca" as a legitimately-empty batch, with no trace of WHY. This pins
_parse_snapshot_batch (bot_engine.py) — the pure function scan_market's
error path now consults for a debug_detail — against every shape that
caused the prod incident class: non-200 status, non-dict/error-shaped 200
body, and an all-empty-dailyBar body (entitlement/pre-open edge case).
"""
import unittest

from bot_engine import _parse_snapshot_batch


class TestParseSnapshotBatchDiag(unittest.TestCase):
    def test_normal_response_parses_with_no_diag(self):
        raw = {
            "AAPL": {
                "dailyBar": {"c": 150.0, "o": 148.0, "v": 1_000_000},
                "prevDailyBar": {"c": 149.0},
            }
        }
        out, detail = _parse_snapshot_batch(raw, 200)
        self.assertIn("AAPL", out)
        self.assertEqual(out["AAPL"], (150.0, 148.0, 1_000_000, 149.0))
        self.assertIsNone(detail)

    def test_non_200_status_captured(self):
        out, detail = _parse_snapshot_batch(
            {"code": 40310000, "message": "subscription does not permit this feed"}, 403
        )
        self.assertEqual(out, {})
        self.assertIn("403", detail)

    def test_error_shaped_200_body_captured(self):
        # Alpaca can return HTTP 200 with an error-shaped (non-per-symbol) body.
        raw = {"code": 42210000, "message": "invalid feed"}
        out, detail = _parse_snapshot_batch(raw, 200)
        self.assertEqual(out, {})
        self.assertIsNotNone(detail)
        self.assertIn("0/2 symbols", detail)

    def test_non_dict_response_captured(self):
        out, detail = _parse_snapshot_batch(["unexpected", "list"], 200)
        self.assertEqual(out, {})
        self.assertIn("non-dict", detail)

    def test_all_zero_dailybar_captured(self):
        # Entitlement/pre-open edge case: 200 + per-symbol shape, but every
        # dailyBar close is 0 — the exact silent-failure shape suspected in
        # the live incident this test file was written to chase.
        raw = {"XYZ": {"dailyBar": {"c": 0, "o": 0, "v": 0}, "prevDailyBar": {"c": 10}}}
        out, detail = _parse_snapshot_batch(raw, 200)
        self.assertEqual(out, {})
        self.assertIn("0/1 symbols", detail)

    def test_partial_batch_no_diag_when_some_symbols_usable(self):
        raw = {
            "AAPL": {"dailyBar": {"c": 150.0, "o": 148.0, "v": 1_000_000}, "prevDailyBar": {"c": 149.0}},
            "ZZZZ": {"dailyBar": {"c": 0, "o": 0, "v": 0}, "prevDailyBar": {"c": 1}},
        }
        out, detail = _parse_snapshot_batch(raw, 200)
        self.assertEqual(set(out.keys()), {"AAPL"})
        self.assertIsNone(detail, "a batch with at least one usable symbol is not a diagnosable failure")


if __name__ == "__main__":
    unittest.main()
