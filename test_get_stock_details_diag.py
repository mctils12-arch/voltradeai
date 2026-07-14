"""
KNOWN BROKEN #21 (open_questions.md, confirmed live 2026-07-14) — regression
test for get_stock_details()'s previously-silent failure path.

Live evidence this session: /api/diag/audit?type=DIAGNOSTIC showed
wikipedia/gdelt/fred reporting "down" continuously through a full Monday
trading day (2026-07-13, 12:13-22:45 UTC), while /api/diag/scanner's
dataSourceErrors — which DOES capture exceptions from deep_score's 5
enrichment fetchers (_run_diag_fetch, v1.0.150/155) — stayed empty the
whole time. Since every one of those 5 fetchers (including the "alt"
one that writes wiki_/fred_/gdelt_ cache files unconditionally, even on
total network failure) unconditionally writes its cache file whenever it
actually runs, the only way all three could be simultaneously absent for
10+ hours is if the enrichment block was never reached at all — i.e.
deep_score()'s early return (`if not detail or "error" in detail: return
quick_result`) fired for every scanned candidate, and get_stock_details()
swallowed whatever caused that with a bare `except Exception: pass`,
leaving zero trace anywhere.

This test pins get_stock_details()'s new optional _diag parameter: it must
capture WHY detail was None/error (subprocess exception, empty stdout, or
analyze.py's own {"error": ...} payload) without changing any return value,
mirroring the existing _run_diag_fetch convention used by deep_score's 5
enrichment fetchers.
"""
import subprocess
import unittest
from unittest.mock import patch, MagicMock

from bot_engine import get_stock_details


def _fake_result(stdout="", stderr="", returncode=0):
    r = MagicMock()
    r.stdout = stdout
    r.stderr = stderr
    r.returncode = returncode
    return r


class TestGetStockDetailsDiag(unittest.TestCase):
    def test_success_leaves_diag_untouched(self):
        diag = {}
        with patch("subprocess.run", return_value=_fake_result(stdout='{"price": 100}')):
            result = get_stock_details("AAPL", _diag=diag)
        self.assertEqual(result, {"price": 100})
        self.assertEqual(diag, {}, "a clean analyze.py result must not write anything into diag")

    def test_analyze_py_error_payload_captured(self):
        diag = {}
        with patch("subprocess.run", return_value=_fake_result(stdout='{"error": "No module named \'yfinance\'"}')):
            result = get_stock_details("AAPL", _diag=diag)
        self.assertEqual(result, {"error": "No module named 'yfinance'"})
        self.assertIn("stock_details", diag)
        self.assertIn("No module named", diag["stock_details"])

    def test_empty_stdout_captured_with_stderr_and_exit_code(self):
        diag = {}
        with patch("subprocess.run", return_value=_fake_result(stdout="", stderr="Traceback: boom", returncode=1)):
            result = get_stock_details("AAPL", _diag=diag)
        self.assertIsNone(result)
        self.assertIn("stock_details", diag)
        self.assertIn("exit=1", diag["stock_details"])
        self.assertIn("boom", diag["stock_details"])

    def test_subprocess_timeout_captured(self):
        diag = {}
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="analyze.py", timeout=30)):
            result = get_stock_details("AAPL", _diag=diag)
        self.assertIsNone(result)
        self.assertIn("stock_details", diag)
        self.assertTrue(diag["stock_details"].startswith("TimeoutExpired:"))

    def test_malformed_json_captured(self):
        diag = {}
        with patch("subprocess.run", return_value=_fake_result(stdout="not json")):
            result = get_stock_details("AAPL", _diag=diag)
        self.assertIsNone(result)
        self.assertIn("stock_details", diag)
        self.assertIn("JSONDecodeError", diag["stock_details"])

    def test_diag_none_never_raises_and_return_value_unchanged(self):
        """Backward-compatible default: callers that don't pass _diag get the
        exact same behavior as before this change, for every failure mode."""
        scenarios = [
            {"return_value": _fake_result(stdout='{"error": "x"}')},
            {"return_value": _fake_result(stdout="", stderr="boom", returncode=1)},
            {"side_effect": subprocess.TimeoutExpired(cmd="analyze.py", timeout=30)},
            {"return_value": _fake_result(stdout="not json")},
        ]
        for kwargs in scenarios:
            with patch("subprocess.run", **kwargs):
                with_diag = get_stock_details("AAPL", _diag={})
                without_diag = get_stock_details("AAPL")
            self.assertEqual(with_diag, without_diag)


class TestDeepScorePassesDiagThrough(unittest.TestCase):
    """Static-shape check (no network): deep_score must forward its own
    _diag into get_stock_details so a stalled analyze.py call is no longer
    invisible to /api/diag/scanner."""

    def test_deep_score_forwards_diag(self):
        import inspect
        import bot_engine
        src = inspect.getsource(bot_engine.deep_score)
        self.assertIn("get_stock_details(ticker, _diag=_diag)", src,
                      "deep_score must call get_stock_details with its own _diag forwarded")


if __name__ == "__main__":
    unittest.main()
