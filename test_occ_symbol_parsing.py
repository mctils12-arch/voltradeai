#!/usr/bin/env python3
"""
REPAIR 2026-08-06 ratchet (KNOWN BROKEN, live audit-log evidence):
options_execution._fetch_option_chain and options_scanner._fetch_options_chain
both parsed OCC option symbols by slicing off len(ticker) characters from the
FRONT ("sym_body = occ_symbol[len(ticker):]"). Adjusted-contract roots (OCC's
own convention after a corporate action, e.g. a root like "IONQ1..." instead
of the plain "IONQ") carry extra characters the plain ticker doesn't, which
shifts this slice by one and corrupts the C/P flag + strike digits.

Live evidence (2026-08-06 /api/diag/audit): a T2-FAIL entry for IONQ reading
"No options contracts available for this ticker (exception: invalid literal
for int() with base 10: 'P00037000')" — the smoking gun: sym_body[7:] landed
on "P00037000" (the C/P flag plus the full 8-digit strike) instead of the
bare 8-digit strike, because sym_body itself was one character too long.

Fix: parse strike/type/date anchored from the END of the OCC symbol (fixed-
width regardless of root length), the same pattern options_scanner.py's own
ATM-IV lookup already used successfully.

Run: python3 -m pytest test_occ_symbol_parsing.py -q
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options_execution
import options_scanner


def _snapshot(occ_symbol, bid=1.0, ask=1.2):
    return {
        occ_symbol: {
            "latestQuote": {"bp": bid, "ap": ask, "bs": 5, "as": 5},
            "latestTrade": {"p": (bid + ask) / 2},
            "dailyBar": {"v": 50},
            "openInterest": 250,
            "greeks": {},
        }
    }


class TestOptionsExecutionOccParsing(unittest.TestCase):
    def setUp(self):
        options_execution._last_chain_error.clear()

    @patch("options_execution.requests.get")
    def test_adjusted_root_with_extra_character_parses_correctly(self, mock_get):
        # Root "IONQ1" (one char longer than the plain ticker "IONQ") is the
        # exact shape that broke the front-anchored slice live.
        occ = "IONQ1260821P00037000"
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"snapshots": _snapshot(occ)}
        )
        contracts = options_execution._fetch_option_chain("IONQ", 38.0, min_dte=7, option_type="put")
        self.assertEqual(len(contracts), 1)
        c = contracts[0]
        self.assertEqual(c["strike"], 37.0)
        self.assertEqual(c["option_type"], "put")
        self.assertEqual(c["expiry"], "2026-08-21")
        self.assertNotIn("IONQ", options_execution._last_chain_error)

    @patch("options_execution.requests.get")
    def test_plain_root_still_parses_correctly(self, mock_get):
        occ = "AAPL260418C00250000"
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"snapshots": _snapshot(occ)}
        )
        contracts = options_execution._fetch_option_chain("AAPL", 245.0, min_dte=7, option_type="call")
        self.assertEqual(len(contracts), 1)
        c = contracts[0]
        self.assertEqual(c["strike"], 250.0)
        self.assertEqual(c["option_type"], "call")
        self.assertEqual(c["expiry"], "2026-04-18")


class TestOptionsScannerOccParsing(unittest.TestCase):
    def setUp(self):
        options_scanner._chain_cache.clear()
        options_scanner._chain_cache_ts.clear()

    @patch("options_scanner.requests.get")
    def test_adjusted_root_with_extra_character_parses_correctly(self, mock_get):
        occ = "IONQ1260821P00037000"
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"snapshots": _snapshot(occ, bid=1.0, ask=1.4)},
        )
        contracts = options_scanner._fetch_options_chain("IONQ", 38.0)
        self.assertEqual(len(contracts), 1)
        c = contracts[0]
        self.assertEqual(c["strike"], 37.0)
        self.assertEqual(c["opt_type"], "put")
        self.assertEqual(c["exp_date"], "2026-08-21")

    @patch("options_scanner.requests.get")
    def test_plain_root_still_parses_correctly(self, mock_get):
        occ = "AAPL260418C00250000"
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"snapshots": _snapshot(occ, bid=1.0, ask=1.4)},
        )
        contracts = options_scanner._fetch_options_chain("AAPL", 245.0)
        self.assertEqual(len(contracts), 1)
        c = contracts[0]
        self.assertEqual(c["strike"], 250.0)
        self.assertEqual(c["opt_type"], "call")
        self.assertEqual(c["exp_date"], "2026-04-18")


if __name__ == "__main__":
    unittest.main()
