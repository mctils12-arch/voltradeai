#!/usr/bin/env python3
"""
REPAIR 2026-07-24 ratchet: options_execution.py's _fetch_option_chain
used to swallow non-200 OPRA responses into a bare `logger.warning`,
making a real fetch failure (entitlement/rate-limit/network) and a
genuinely-empty options chain indistinguishable at the audit-log level —
both surfaced as the generic "No options contracts available for this
ticker" (see research/experiments.md 2026-07-24, research/open_questions.md
KNOWN BROKEN precedent from 2026-07-12/13).

These tests pin: a non-200 response is captured into `_last_chain_error`
and threaded into select_contract's returned error string; a genuinely
empty (200, zero snapshots) chain gets the "no HTTP error" reason instead
of a fabricated one; a subsequent successful fetch clears the stale error
so it never leaks into an unrelated later failure.

Run: python3 -m pytest test_options_chain_diagnosability.py -q
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options_execution


class TestChainFetchErrorCapture(unittest.TestCase):
    def setUp(self):
        options_execution._last_chain_error.clear()

    @patch("options_execution.requests.get")
    def test_non_200_response_captured_with_status_and_body(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=403, text='{"message":"not entitled to opra"}'
        )
        contracts = options_execution._fetch_option_chain("STM", 40.0, min_dte=7)
        self.assertEqual(contracts, [])
        self.assertIn("STM", options_execution._last_chain_error)
        reason = options_execution._last_chain_error["STM"]
        self.assertIn("403", reason)
        self.assertIn("not entitled to opra", reason)

    @patch("options_execution.requests.get")
    def test_exception_during_fetch_captured(self, mock_get):
        mock_get.side_effect = ConnectionError("boom")
        contracts = options_execution._fetch_option_chain("DRAM", 12.0, min_dte=7)
        self.assertEqual(contracts, [])
        self.assertIn("exception", options_execution._last_chain_error["DRAM"])
        self.assertIn("boom", options_execution._last_chain_error["DRAM"])

    @patch("options_execution.requests.get")
    def test_genuinely_empty_chain_leaves_no_error(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"snapshots": {}})
        contracts = options_execution._fetch_option_chain("HYG", 78.0, min_dte=7)
        self.assertEqual(contracts, [])
        self.assertNotIn("HYG", options_execution._last_chain_error)

    @patch("options_execution.requests.get")
    def test_success_clears_prior_error_for_same_ticker(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500, text="server error")
        options_execution._fetch_option_chain("QS", 5.0, min_dte=7)
        self.assertIn("QS", options_execution._last_chain_error)

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"snapshots": {}})
        options_execution._fetch_option_chain("QS", 5.0, min_dte=7)
        self.assertNotIn("QS", options_execution._last_chain_error)


class TestSelectContractSurfacesReason(unittest.TestCase):
    def setUp(self):
        options_execution._last_chain_error.clear()

    @patch("options_execution._fetch_option_chain")
    def test_fetch_failure_reason_reaches_select_contract_error(self, mock_fetch):
        def _fake_fetch(ticker, price, min_dte=7, option_type=None):
            options_execution._last_chain_error[ticker] = "HTTP 403: not entitled to opra"
            return []
        mock_fetch.side_effect = _fake_fetch

        result = options_execution.select_contract(
            "STM", "sell_cash_secured_put", 40.0, 100000
        )
        self.assertIn("error", result)
        self.assertIn("403", result["error"])
        self.assertIn("not entitled to opra", result["error"])

    @patch("options_execution._fetch_option_chain")
    def test_genuinely_empty_chain_gets_honest_reason_not_fabricated(self, mock_fetch):
        mock_fetch.return_value = []  # no entry set in _last_chain_error

        result = options_execution.select_contract(
            "HYG", "sell_cash_secured_put", 78.0, 100000
        )
        self.assertIn("error", result)
        self.assertNotIn("HTTP", result["error"])
        self.assertIn("zero contracts", result["error"])


if __name__ == "__main__":
    unittest.main()
