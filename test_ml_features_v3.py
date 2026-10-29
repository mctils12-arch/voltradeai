"""ml_features_v3.py — one feature definition for training and serving
([REPAIR] 2026-10-27, audit §4.1).

Pins: (1) the serving path reproduces the training row exactly on the same
fixture (train/serve parity); (2) no v3 feature is constant across a
fixture batch (the 9 constant-at-serve v2 features could not have passed);
(3) no row reads bars after its decision bar; (4) every v2 feature has a
recorded disposition.
"""
import math
import os
import sys
from datetime import date, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ml_features_v3 as f3  # noqa: E402


def _dates(n, start=date(2024, 1, 2)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _series(dates, base, vol, seed, volume=2_000_000):
    rng = np.random.default_rng(seed)
    c = base * np.exp(np.cumsum(rng.normal(0.0003, vol, len(dates))))
    out = []
    for d, x in zip(dates, c):
        o = x * (1 + rng.normal(0, vol / 3))
        hi = max(o, x) * (1 + abs(rng.normal(0, vol / 2)))
        lo = min(o, x) * (1 - abs(rng.normal(0, vol / 2)))
        out.append({"t": d + "T04:00:00Z", "o": float(o), "h": float(hi), "l": float(lo),
                    "c": float(x), "v": float(volume * (0.5 + rng.random()))})
    return out


def _fixture(n=340, seed=0):
    ds = _dates(n)
    refs = {"SPY": _series(ds, 450, 0.01, seed + 1, 5e7),
            "VXX": _series(ds, 40, 0.04, seed + 2, 1e7),
            "TLT": _series(ds, 90, 0.008, seed + 3, 1e7),
            "HYG": _series(ds, 77, 0.004, seed + 4, 1e7)}
    # the stock trades on a slightly different calendar (a missing session)
    # so the as-of reference alignment is exercised
    stock = [b for k, b in enumerate(_series(ds, 60, 0.025, seed + 5)) if k != 280]
    vxx_gap = [b for k, b in enumerate(refs["VXX"]) if k != 300]
    refs["VXX"] = vxx_gap
    return ds, stock, refs


def _same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


def test_serving_path_equals_training_path_on_the_same_fixture():
    ds, stock, refs = _fixture()
    rows = f3.training_rows(stock, f3.RefSeries(refs), eligible_only=False)
    assert len(rows) > 50
    checked = 0
    for i, d, train_feats in rows[-40::7] + [rows[-1]]:
        k = ds.index(d)
        today = ds[k + 1] if k + 1 < len(ds) else (date.fromisoformat(d) + timedelta(days=1)).isoformat()
        # the live fetch also returns today's in-progress bar (and the
        # reference fetch may run a little later) — serve must strip both
        live_stock = [b for b in stock if b["t"][:10] <= today]
        live_refs = {s: [b for b in refs[s] if b["t"][:10] <= today] for s in refs}
        served = f3.serve_features(live_stock, live_refs, today)
        assert served is not None
        assert served["_asof_date"] == d
        for col in f3.V3_FEATURE_COLS:
            assert _same(served[col], train_feats[col]), (d, col, served[col], train_feats[col])
        checked += 1
    assert checked >= 5


def test_no_feature_is_constant_across_a_fixture_batch():
    batch = []
    for seed in range(4):
        _, stock, refs = _fixture(seed=10 * seed)
        batch += [f3.row_vector(f) for _, _, f in f3.training_rows(stock, f3.RefSeries(refs))]
    X = np.array(batch)
    assert X.shape[1] == len(f3.V3_FEATURE_COLS)
    for j, col in enumerate(f3.V3_FEATURE_COLS):
        x = X[:, j]
        assert not np.isnan(x).all(), f"{col} is never computed"
        assert np.nanstd(x) > 0, f"{col} is constant across the batch (dead at serve or train)"


def test_rows_never_read_bars_after_the_decision_bar():
    ds, stock, refs = _fixture(seed=3)
    full = {d: f for _, d, f in f3.training_rows(stock, f3.RefSeries(refs), eligible_only=False)}
    cut = ds[300]
    trunc_stock = [b for b in stock if b["t"][:10] <= cut]
    trunc_refs = {s: [b for b in refs[s] if b["t"][:10] <= cut] for s in refs}
    part = {d: f for _, d, f in f3.training_rows(trunc_stock, f3.RefSeries(trunc_refs), eligible_only=False)}
    assert part
    for d, feats in part.items():
        for col in f3.V3_FEATURE_COLS:
            assert _same(feats[col], full[d][col]), (d, col)


def test_serve_returns_none_without_enough_completed_bars():
    ds, stock, refs = _fixture(n=f3.MIN_BARS)
    # MIN_BARS bars total, but the last is "today" and gets stripped
    assert f3.serve_features(stock, refs, ds[-1]) is None


def test_frac_diff_matches_the_v2_definition():
    v2 = pytest.importorskip("ml_model_v2")
    rng = np.random.default_rng(9)
    closes = list(50 * np.exp(np.cumsum(rng.normal(0, 0.02, 40))))
    assert f3.frac_diff(closes) == pytest.approx(v2._frac_diff(closes))


def test_every_v2_feature_has_a_disposition_and_dropped_ones_stay_out():
    v2 = pytest.importorskip("ml_model_v2")
    assert set(f3.V2_FEATURE_DISPOSITION) == set(v2.FEATURE_COLS)
    dropped = [k for k, v in f3.V2_FEATURE_DISPOSITION.items() if v.startswith("dropped")]
    assert set(dropped) >= {"earnings_surprise", "intel_score", "news_sentiment", "insider_signal",
                            "days_to_earnings", "cross_sec_rank", "market_breadth"}
    for k in dropped:
        assert k not in f3.V3_FEATURE_COLS
    assert len(set(f3.V3_FEATURE_COLS)) == len(f3.V3_FEATURE_COLS)


def test_indicator_reference_values():
    # RSI of a strictly rising series is 100; of a flat series 50
    assert f3._wilder_rsi(np.arange(1.0, 120.0)) == 100.0
    assert f3._wilder_rsi(np.full(101, 5.0)) == 50.0
    # ADX of a clean uptrend is high, of an alternating chop is low
    up = np.arange(100.0, 200.0)
    assert f3._wilder_adx(up + 0.5, up - 0.5, up) > 60
    chop = 100 + np.tile([1.0, -1.0], 50)
    assert f3._wilder_adx(chop + 0.5, chop - 0.5, chop) < 25
    # hv_rank of the most volatile window is 100
    rng = np.random.default_rng(1)
    calm = list(np.exp(np.cumsum(rng.normal(0, 0.005, 210))))
    wild = list(calm[-1] * np.exp(np.cumsum(rng.normal(0, 0.05, 15))))
    assert f3._hv_rank(np.array(calm + wild)) == 100.0
