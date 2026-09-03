"""
Regression tests for backtest_v2.py's regime_series() data-quality flag.

MEASUREMENT INTEGRITY (2026-09-03, [RULE-REVIEW]): `regime_series()`'s
docstring always promised "1.0 (degraded) when VXX data is missing", but the
old check (`"ok" if vxx_by_date else "degraded"`) only looked at whether VXX
was TOTALLY absent — a request window that starts before VXX's real data
history (this provider's VXX series only goes back to ~2018-01-25) got a
silent per-day fallback to a synthetic neutral vxx_ratio=1.0 for every day
before that, while `quality` still reported "ok" because *some* VXX data
existed somewhere in the fetch. Found by hazard_rate_probe.py's own "longer
window" GATE 2 follow-up: extending `--days` past ~3650 changed the shape of
the detected onset set in a way a real renewal process spanning a consistent
measurement shouldn't, traced to this silent VXX-coverage gap. This directly
matters for PROMOTION RULE 3 too — a `years=10` (3650-day) backtest, the
maximum `backtest.py` accepts, already crosses that 2018-01-25 boundary.

These tests use only synthetic, injected bars — no network, no keys
(CI-safe) — and pin: (1) full coverage stays "ok"; (2) VXX totally absent
stays "degraded" (the pre-fix behavior, preserved); (3) a large partial gap
(the real-world failure mode) is now "degraded"; (4) the normal ~1-day
trailing VXX-arrives-a-day-behind lag every live fetch already carries stays
"ok" — the fix must not cry wolf on ordinary, harmless staleness.
"""
import unittest
from datetime import date, timedelta

from backtest_v2 import regime_series, _VXX_COVERAGE_DEGRADED_THRESHOLD


def _series(n, start=date(2022, 1, 3), price=100.0):
    """Flat-price daily bars — only 'date'/'close' matter to regime_series."""
    dates, closes = [], []
    for i in range(n):
        dates.append((start + timedelta(days=i)).isoformat())
        closes.append(price)
    return {"date": dates, "open": list(closes), "high": list(closes),
            "low": list(closes), "close": list(closes), "volume": [1_000_000] * n}


def _drop_dates(series, indices_to_drop):
    """A copy of `series` with the bars at `indices_to_drop` removed —
    simulates VXX not covering some subset of the SPY date range."""
    keep = [i for i in range(len(series["date"])) if i not in set(indices_to_drop)]
    return {k: [v[i] for i in keep] for k, v in series.items()}


class TestRegimeSeriesQuality(unittest.TestCase):
    def test_full_vxx_coverage_is_ok(self):
        spy = _series(300)
        vxx = _series(300, price=15.0)
        _, quality = regime_series(spy, vxx)
        self.assertEqual(quality, "ok")

    def test_vxx_totally_absent_is_degraded(self):
        """Pre-fix behavior, preserved: no VXX data at all."""
        spy = _series(300)
        _, quality = regime_series(spy, None)
        self.assertEqual(quality, "degraded")
        _, quality_empty = regime_series(spy, {"date": [], "close": []})
        self.assertEqual(quality_empty, "degraded")

    def test_large_partial_gap_is_degraded(self):
        """The real-world failure mode: VXX covers the back half of the
        window (mirroring a request window starting before VXX's real
        history) but is missing entirely from the front half — 50% of days
        have no real VXX data, far above the 5% threshold."""
        spy = _series(300)
        vxx_partial = _drop_dates(_series(300, price=15.0), range(0, 150))
        _, quality = regime_series(spy, vxx_partial)
        self.assertEqual(quality, "degraded")

    def test_normal_trailing_lag_stays_ok(self):
        """VXX missing only the single most-recent day (the ordinary
        live-fetch edge case: VXX data arrives ~1 day behind SPY) must NOT
        be flagged degraded — that is not a real data-quality problem."""
        spy = _series(300)
        vxx_lagged = _drop_dates(_series(300, price=15.0), [299])
        _, quality = regime_series(spy, vxx_lagged)
        self.assertEqual(quality, "ok")

    def test_threshold_boundary(self):
        """Missing fraction must be STRICTLY greater than the threshold to
        degrade — pins the exact comparison operator, not just the
        direction, so a future edit can't silently flip <= to < unnoticed."""
        n = 200
        at_threshold = int(n * _VXX_COVERAGE_DEGRADED_THRESHOLD)  # 10 of 200 = 5%
        spy = _series(n)

        vxx_at = _drop_dates(_series(n, price=15.0), range(at_threshold))
        _, quality_at = regime_series(spy, vxx_at)
        self.assertEqual(quality_at, "ok",
                          "exactly at the threshold must not yet degrade")

        vxx_over = _drop_dates(_series(n, price=15.0), range(at_threshold + 1))
        _, quality_over = regime_series(spy, vxx_over)
        self.assertEqual(quality_over, "degraded",
                          "one day past the threshold must degrade")

    def test_labels_still_computed_length_matches_spy_even_when_degraded(self):
        """Degraded quality must not change the CONTRACT (one label per SPY
        day, synthetic-fallback vxx_ratio=1.0) — only the quality flag."""
        spy = _series(300)
        vxx_partial = _drop_dates(_series(300, price=15.0), range(0, 150))
        labels, quality = regime_series(spy, vxx_partial)
        self.assertEqual(len(labels), len(spy["date"]))
        self.assertEqual(quality, "degraded")


if __name__ == "__main__":
    unittest.main()
