"""earnings_surprise look-ahead removed from ml_model_v2 training
([RULE-REVIEW] 2026-10-27, audit §4.1 / §5.5).

Before: _train_model_impl fetched Finnhub's LATEST surprise per ticker and
_build_training_data stamped it on EVERY historical row of that ticker, so a
row dated years before the report carried its outcome. No point-in-time
source exists (Finnhub's free endpoint has no report dates, 4 quarters only),
so the feature is 0.0 on generic rows — the same constant serving produces.
"""
import inspect
import os
import sys
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ml_model_v2 as v2  # noqa: E402


def _bars(seed, n=140):
    rng = np.random.default_rng(seed)
    ds, d = [], date(2024, 1, 2)
    while len(ds) < n:
        if d.weekday() < 5:
            ds.append(d.isoformat())
        d += timedelta(days=1)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return [{"t": x + "T04:00:00Z", "o": float(p), "h": float(p * 1.01), "l": float(p * 0.99),
             "c": float(p), "v": 2_000_000} for x, p in zip(ds, c)]


def test_no_latest_surprise_fetch_and_no_per_ticker_injection():
    assert not hasattr(v2, "_fetch_earnings_surprises")
    assert "earnings_surprises" not in inspect.signature(v2._build_training_data).parameters
    src = inspect.getsource(v2._train_model_impl)
    assert "finnhub" not in src.lower() and "earnings_surpris" not in src


def test_generic_rows_carry_the_serve_time_constant():
    bars = {"AAA": _bars(1), "BBB": _bars(2), "SPY": _bars(3), "VXX": _bars(4)}
    X, y, _ = v2._build_training_data(bars)
    j = v2.FEATURE_COLS.index("earnings_surprise")
    assert np.all(X[:, j] == 0.0)
    # intel_score was 0.3 * earnings_surprise on generic rows — now 0 too
    assert np.all(X[:, v2.FEATURE_COLS.index("intel_score")] == 0.0)
