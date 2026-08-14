# pytest collection config — makes the constitutional local gate
# (`python3 -m pytest -q`, CLAUDE.md promotion rule 1) runnable repo-wide.
#
# Two root-level files carry the test_ prefix but are STANDALONE SCRIPTS
# with their own runners, not pytest suites. Collecting them breaks the
# entire gate (KNOWN BROKEN #6, resolved 2026-07-04):
#
# - test_auto_discovery.py: executes its whole discovery protocol at
#   import and ends with sys.exit() -> pytest INTERNALERROR that kills
#   collection for every other file. Run it directly:
#   `python3 test_auto_discovery.py`.
# - test_full_system.py: defines a module-level helper `def test(phase,
#   name, fn)` that pytest collects as a test and then fails to resolve
#   `phase` as a fixture. Run it directly: `python3 test_full_system.py`.
#
# Excluding them here removes NO assertions from the gate — neither file
# can execute under pytest at all; both remain runnable as scripts.
# test_collection_health.py is the regression guard that keeps future
# collection breakers out of the gate.
collect_ignore = [
    "test_auto_discovery.py",
    "test_full_system.py",
]

import pytest
from unittest.mock import MagicMock, patch


def _empty_yf_ticker(*_a, **_kw):
    fake = MagicMock()
    fake.history.return_value = MagicMock(empty=True)
    return fake


@pytest.fixture(scope="session", autouse=True)
def _hermetic_yfinance():
    """MASTER PROGRAM Q22: the gated suite is not hermetic — macro_data.py's
    get_macro_snapshot() calls yfinance live (^VIX, ^TNX, DX-Y.NYB) whenever
    its own 5-minute disk cache is cold, and every real call this session's
    diagnostic probe found reaches the network and fails there: 4 test files
    hit it WITHOUT mocking it themselves (test_deep_score_credit_spread_
    cache.py, test_gridvision_pod_run.py, test_tiered_strategy.py,
    test_voltrade_daemon.py — none import yfinance or patch it; they reach
    macro_data indirectly). All 1348 non-quarantined tests pass regardless,
    because every branch in macro_data.py already degrades to a documented
    default on a yfinance exception — so this fixture reproduces that exact
    same default-taking code path hermetically instead of via a live,
    always-failing network call. It changes reachability, not behavior.

    Session-scoped and a `with` context manager (not `.start()`), so a local
    test-level `patch("yfinance.Ticker", ...).start()` (e.g.
    test_macro_snapshot_spy_dedup.py) simply shadows this for its own
    duration; its `patch.stopall()` cleanup only unwinds patches started via
    `.start()` and cannot touch this one.
    """
    import yfinance as yf

    with patch.object(yf, "Ticker", side_effect=_empty_yf_ticker):
        yield
