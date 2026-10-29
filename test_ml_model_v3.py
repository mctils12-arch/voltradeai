"""ml_model_v3.py — the shadow v3 candidate (2026-10-27).

Label: one exit-aligned label (live_exit_policy, the mirror of bot.ts
checkPositionOnTick) instead of v2's three (triple barrier 1.5/1.0 ATR 5d,
shadow +2/-4 10d, feedback > 1.5 ATR).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ml_features_v3 as f3  # noqa: E402
import ml_model_v3 as m3  # noqa: E402
from test_ml_features_v3 import _dates, _fixture, _series  # noqa: E402


def _path(prices, dates):
    """Bars whose OHLC follow a given list of (o, h, l, c)."""
    return [{"t": d + "T04:00:00Z", "o": o, "h": h, "l": l, "c": c, "v": 1e6}
            for d, (o, h, l, c) in zip(dates, prices)]


def test_constants_match_the_shadow_exit_aligned_ruler():
    sp = pytest.importorskip("shadow_portfolio")
    assert m3.LABEL_CAP_BARS == sp.XA_CAP_BARS
    assert m3.ROUND_TRIP_COST_PCT == sp.XA_ROUND_TRIP_COST_PCT
    assert m3.EMBARGO_DATES > m3.LABEL_CAP_BARS


def test_label_matches_the_shadow_ruler_on_fuzzed_paths():
    sp = pytest.importorskip("shadow_portfolio")
    rng = np.random.default_rng(4)
    ds = _dates(40)
    for trial in range(300):
        bars = _series(ds, 50, float(rng.choice([0.01, 0.02, 0.04])), 1000 + trial)
        atr = float(rng.uniform(0.8, 6.0))
        reg = str(rng.choice(["bullish", "bearish", "neutral"]))
        a = f3.to_arrays(bars)
        entry = float(a["o"][1])
        mine = m3.label_exit_aligned(entry, a, 1, atr, reg)
        theirs = sp._label_exit_aligned(entry, bars[1:], atr, reg, ds[1])
        assert mine is not None and theirs["label"] in (0, 1)
        assert mine["exit_reason"] == theirs["exit_reason"], trial
        assert mine["return_pct"] == pytest.approx(theirs["return_pct"], abs=1e-3), trial
        assert mine["net_return_pct"] == pytest.approx(mine["return_pct"] - m3.ROUND_TRIP_COST_PCT)
        assert mine["label"] == int(mine["net_return_pct"] > 0)


def test_label_reads_the_live_stop_not_the_v2_barriers():
    # ATR 2% -> live phase-1 stop = clamp(2*ATR, 1.5, 8) = 4%; v2's triple
    # barrier would have stopped at -1 ATR (-2%) and v2 shadow at -4%/+2%.
    ds = _dates(25)
    path = [(100, 100, 100, 100)] + [(100, 100.5, 97.0, 97.5)] + [(97.5, 99, 95.5, 96)] * 23
    a = f3.to_arrays(_path(path, ds))
    out = m3.label_exit_aligned(100.0, a, 0, 2.0, "neutral")
    assert out["exit_reason"] == "stop"
    assert out["return_pct"] == pytest.approx(-4.0, abs=1e-6)
    assert out["label"] == 0
    # +2% then drift: v2-shadow's +2% target would be a WIN; live scales a
    # third out at 1R = +4% only, so a +2..3% wander that fades is a time stop
    flat = [(100, 100, 100, 100)] + [(100, 103, 99.5, 101)] * 4 + [(101, 101.5, 100.5, 101)] * 20
    out = m3.label_exit_aligned(100.0, f3.to_arrays(_path(flat, ds)), 0, 2.0, "neutral")
    assert out["exit_reason"] == "time_stop"
    assert out["return_pct"] == pytest.approx(1.0, abs=1e-6)


def test_incomplete_path_is_dropped_not_labeled():
    ds = _dates(5)
    path = [(100, 100.5, 99.5, 100)] * 5
    assert m3.label_exit_aligned(100.0, f3.to_arrays(_path(path, ds)), 0, 0.5, "neutral") is None


def test_dataset_is_chronological_point_in_time_and_full_window():
    ds, stock, refs = _fixture(n=330, seed=7)
    data = m3.build_dataset({"AAA": stock, "BBB": _series(ds, 30, 0.03, 77), **refs})
    n = len(data["y"])
    assert n > 50
    assert list(data["dates"]) == sorted(data["dates"])
    assert set(np.unique(data["y"])) <= {0, 1}
    assert data["X"].shape == (n, len(f3.V3_FEATURE_COLS))
    assert all(e > d for d, e in zip(data["dates"], data["label_end_dates"]))
    # every row's full cap window exists: no row within LABEL_CAP_BARS of the end
    assert max(data["dates"]) <= ds[-1 - m3.LABEL_CAP_BARS]
    # features of a row equal the serve path at that date (label never feeds X)
    k = max(j for j in range(n) if data["tickers"][j] == "AAA")
    d = data["dates"][k]
    served = f3.serve_features(stock, refs, ds[ds.index(d) + 1])
    assert np.allclose(np.array(f3.row_vector(served), dtype=np.float32), data["X"][k], equal_nan=True)
