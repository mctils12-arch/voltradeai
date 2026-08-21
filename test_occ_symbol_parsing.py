#!/usr/bin/env python3
"""
REPAIR ratchet: options_execution._fetch_option_chain and
options_scanner._fetch_options_chain both parsed OCC option symbols
(root + YYMMDD + C/P + 8-digit strike) by slicing occ_symbol[len(ticker):]
— stripping len(ticker) characters off the FRONT to isolate the
date/type/strike suffix.

OCC's own adjusted-contract convention (a root gaining an extra character
after a corporate action, e.g. "IONQ1" instead of "IONQ") makes that slice
land one character short: the parsed "strike" substring absorbs the C/P
flag character itself (e.g. "P00037000" instead of "00037000"), and
int() throws. The exception unwinds to the function's outer try/except,
which reports the ENTIRE chain unavailable for that ticker, not just the
one malformed contract.

Live symptom reproduced here (from /api/diag/audit?type=T2-FAIL):
  IONQ: No options contracts available for this ticker
  (exception: invalid literal for int() with base 10: 'P00037000')

Fix: parse anchored from the END of the OCC symbol instead — these fields
are fixed-width from the right regardless of root length.

Run: python3 -m pytest test_occ_symbol_parsing.py -q
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options_execution
import options_scanner


def _fake_snapshot(bid=1.20, ask=1.40, iv=0.5, delta=-0.3, volume=100, oi=200):
    return {
        "latestQuote": {"bp": bid, "ap": ask, "bs": 5, "as": 5},
        "latestTrade": {"p": (bid + ask) / 2},
        "greeks": {"iv": iv, "delta": delta, "theta": -0.02, "gamma": 0.01},
        "dailyBar": {"v": volume},
        "openInterest": oi,
        "impliedVolatility": iv,
    }


class TestFetchOptionChainAdjustedRoot(unittest.TestCase):
    """options_execution._fetch_option_chain"""

    def setUp(self):
        options_execution._last_chain_error.clear()

    @patch("options_execution.requests.get")
    def test_adjusted_root_parses_correctly_not_dropped(self, mock_get):
        # Live shape: IONQ's adjusted-contract root "IONQ1", 260418 expiry,
        # put, $37.00 strike — the exact live crash this session reproduced.
        occ_symbol = "IONQ1260418P00037000"
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"snapshots": {occ_symbol: _fake_snapshot()}}
        )
        contracts = options_execution._fetch_option_chain("IONQ", 19.89, min_dte=7, option_type="put")
        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0]["strike"], 37.0)
        self.assertEqual(contracts[0]["expiry"], "2026-04-18")
        self.assertEqual(contracts[0]["option_type"], "put")
        # The whole-chain failure this bug caused must not fire
        self.assertNotIn("IONQ", options_execution._last_chain_error)

    @patch("options_execution.requests.get")
    def test_plain_root_still_parses_correctly(self, mock_get):
        occ_symbol = "AAPL260418C00250000"
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"snapshots": {occ_symbol: _fake_snapshot()}}
        )
        contracts = options_execution._fetch_option_chain("AAPL", 245.0, min_dte=7, option_type="call")
        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0]["strike"], 250.0)
        self.assertEqual(contracts[0]["expiry"], "2026-04-18")
        self.assertEqual(contracts[0]["option_type"], "call")


class TestFetchOptionsChainAdjustedRoot(unittest.TestCase):
    """options_scanner._fetch_options_chain"""

    def setUp(self):
        options_scanner._chain_cache.clear()
        options_scanner._chain_cache_ts.clear()

    @patch("options_scanner.requests.get")
    def test_adjusted_root_parses_correctly_not_dropped(self, mock_get):
        occ_symbol = "IONQ1260418P00037000"
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"snapshots": {occ_symbol: _fake_snapshot()}, "next_page_token": None},
        )
        contracts = options_scanner._fetch_options_chain("IONQ", 19.89)
        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0]["strike"], 37.0)
        self.assertEqual(contracts[0]["exp_date"], "2026-04-18")
        self.assertEqual(contracts[0]["opt_type"], "put")

    @patch("options_scanner.requests.get")
    def test_plain_root_still_parses_correctly(self, mock_get):
        occ_symbol = "AAPL260418C00250000"
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"snapshots": {occ_symbol: _fake_snapshot()}, "next_page_token": None},
        )
        contracts = options_scanner._fetch_options_chain("AAPL", 245.0)
        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0]["strike"], 250.0)
        self.assertEqual(contracts[0]["exp_date"], "2026-04-18")
        self.assertEqual(contracts[0]["opt_type"], "call")


if __name__ == "__main__":
    unittest.main()
