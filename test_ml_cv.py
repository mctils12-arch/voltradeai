"""ml_cv.py — honest out-of-time CV ([RULE-REVIEW] 2026-10-27).

The legacy ml_model_v2 CV splits rows by index while rows are emitted ticker
by ticker, so test folds share calendar dates with training folds and early
stopping reads the test fold. These tests pin the replacement's guarantees
and show, on a synthetic date-only signal, that the legacy procedure reports
skill that does not exist out of time while the honest one does not.
"""
import os
import sys
from datetime import date, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ml_cv  # noqa: E402


@pytest.fixture(autouse=True)
def _single_thread_lightgbm(monkeypatch):
    """The legacy v2 params leave num_threads to OpenMP; on a CPU-throttled
    runner that is ~10x slower than one thread. Thread count changes speed
    only, never which rows a fold reads."""
    try:
        import lightgbm as lgb
    except ImportError:
        return
    real = lgb.train

    def train(params, *a, **k):
        return real({**params, "num_threads": 1}, *a, **k)
    monkeypatch.setattr(lgb, "train", train)


def _trading_dates(n, start=date(2023, 1, 2)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _ticker_major(n_tickers, n_dates, horizon=5):
    """Rows emitted ticker by ticker (the ml_model_v2 builder's order)."""
    ds = _trading_dates(n_dates + horizon)
    dates, ends, tick = [], [], []
    for t in range(n_tickers):
        for i in range(n_dates):
            dates.append(ds[i])
            ends.append(ds[i + horizon])
            tick.append(t)
    return dates, ends, tick


def test_folds_are_strictly_out_of_time_with_embargo_and_purge():
    dates, ends, _ = _ticker_major(4, 300, horizon=10)
    splits = ml_cv.purged_walk_forward_splits(dates, n_splits=5, embargo_dates=10,
                                              label_end_dates=ends)
    assert len(splits) >= 4
    uniq = sorted(set(dates))
    pos = {d: i for i, d in enumerate(uniq)}
    for tr, te in splits:
        tr_dates = {dates[i] for i in tr}
        te_dates = {dates[i] for i in te}
        assert not (tr_dates & te_dates), "a decision date appears in both train and test"
        gap = pos[min(te_dates)] - pos[max(tr_dates)]
        assert gap > 10, f"embargo violated: only {gap} trading dates between train and test"
        assert max(ends[i] for i in tr) < min(te_dates), "a training label window reaches into the test block"
        # every ticker's rows for the test dates are tested together (date blocks, not ticker blocks)
        assert len({i % 300 for i in te}) * 4 == len(te)


def test_legacy_row_index_folds_overlap_dates_on_ticker_major_rows():
    """Documents the bug being fixed: the legacy splitter's test rows share
    calendar dates with its training rows when rows are ticker-major."""
    import ml_model_v2 as v2
    dates, _, _ = _ticker_major(12, 250)
    X = np.zeros((len(dates), 2))
    y = np.zeros(len(dates))
    overlapped = False
    for tr, te in v2._walk_forward_cv(X, y, n_splits=5, embargo_days=5, n_tickers_approx=12):
        if {dates[i] for i in tr} & {dates[i] for i in te}:
            overlapped = True
    assert overlapped
    for tr, te in ml_cv.purged_walk_forward_splits(dates, embargo_dates=5):
        assert not ({dates[i] for i in tr} & {dates[i] for i in te})


def test_inner_validation_slice_comes_from_training_window_only():
    dates, ends, _ = _ticker_major(3, 200, horizon=5)
    splits = ml_cv.purged_walk_forward_splits(dates, n_splits=4, embargo_dates=5, label_end_dates=ends)
    tr, te = splits[-1]
    itr, iva = ml_cv.inner_validation_split(tr, dates, frac=0.2, embargo_dates=5, label_end_dates=ends)
    assert set(itr) <= set(tr) and set(iva) <= set(tr)
    assert not (set(iva) & set(te)), "inner validation must never contain test rows"
    assert max(dates[i] for i in iva) < min(dates[i] for i in te)
    assert max(dates[i] for i in itr) < min(dates[i] for i in iva)
    assert max(ends[i] for i in itr) < min(dates[i] for i in iva)
    uniq = sorted(set(dates))
    pos = {d: i for i, d in enumerate(uniq)}
    assert pos[min(dates[i] for i in iva)] - pos[max(dates[i] for i in itr)] > 5


def test_auc_and_losses_match_reference_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert ml_cv.auc(y, p) == pytest.approx(0.75)
    assert ml_cv.auc([1, 1], [0.2, 0.3]) is None
    m = ml_cv.binary_metrics(y, p)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert m["log_loss"] == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert m["base_rate"] == 0.5
    assert m["base_rate_log_loss"] == pytest.approx(np.log(2))
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    yy = rng.integers(0, 2, 500)
    pp = np.round(rng.random(500), 2)  # rounding forces ties
    assert ml_cv.auc(yy, pp) == pytest.approx(sk.roc_auc_score(yy, pp))


def test_quantile_returns_and_calibration_shapes():
    p = np.linspace(0, 1, 100)
    r = np.linspace(-1, 1, 100)
    q = ml_cv.quantile_returns(p, r, q=5)
    assert [row["q"] for row in q["quantiles"]] == [1, 2, 3, 4, 5]
    assert q["top_minus_bottom"] > 0
    assert q["all_mean"] == pytest.approx(0.0, abs=1e-9)
    cal = ml_cv.calibration_table((p > 0.5).astype(int), p, n_bins=10)
    assert len(cal) == 10 and sum(c["n"] for c in cal) == 100


def _date_only_signal(n_tickers=12, n_dates=150, seed=2):
    """A label decided by the DATE alone (a market move), plus a per-date
    fingerprint feature — the 11 date-only v2 features in miniature. There is
    no out-of-time skill to find: future dates' outcomes are independent."""
    rng = np.random.default_rng(seed)
    ds = _trading_dates(n_dates + 5)
    date_label = rng.integers(0, 2, n_dates)
    fingerprint = rng.normal(size=n_dates)
    import ml_model_v2 as v2
    rows, y, dates, ends = [], [], [], []
    for t in range(n_tickers):
        for i in range(n_dates):
            x = rng.normal(size=len(v2.FEATURE_COLS))
            x[0] = fingerprint[i]
            rows.append(x)
            y.append(date_label[i] if rng.random() < 0.9 else 1 - date_label[i])
            dates.append(ds[i])
            ends.append(ds[i + 5])
    return np.array(rows, dtype=np.float32), np.array(y, dtype=np.int8), dates, ends


def test_legacy_cv_reports_date_leak_skill_and_honest_cv_does_not():
    pytest.importorskip("lightgbm")
    pytest.importorskip("sklearn")
    X, y, dates, ends = _date_only_signal()
    legacy = ml_cv.legacy_cv(X, y, n_tickers=12)
    honest = ml_cv.honest_cv(X, y, dates, label_end_dates=ends, embargo_dates=5)
    assert legacy["pooled"]["auc"] > 0.65, legacy["pooled"]
    assert honest["pooled"]["auc"] < 0.56, honest["pooled"]
    assert honest["folds"], "honest CV produced no folds"
    for f in honest["folds"]:
        assert f["train_last_date"] < f["test_first_date"]
        assert f["early_stopping"] == "inner_validation"


def test_fit_never_reads_test_labels():
    lgb = pytest.importorskip("lightgbm")  # noqa: F841
    X, y, dates, ends = _date_only_signal(n_tickers=6, n_dates=150, seed=1)
    tr, te = ml_cv.purged_walk_forward_splits(dates, embargo_dates=5, label_end_dates=ends)[-1]
    m1, _ = ml_cv.fit_lgbm_inner_es(X, y, tr, dates, label_end_dates=ends, embargo_dates=5)
    y2 = y.copy()
    y2[te] = 1 - y2[te]  # flip every test label
    m2, _ = ml_cv.fit_lgbm_inner_es(X, y2, tr, dates, label_end_dates=ends, embargo_dates=5)
    assert np.allclose(m1.predict(X[te]), m2.predict(X[te]))


def test_v2_builder_return_meta_is_additive():
    import ml_model_v2 as v2
    rng = np.random.default_rng(5)
    ds = _trading_dates(140)

    def series(base):
        c = base * np.exp(np.cumsum(rng.normal(0, 0.02, len(ds))))
        return [{"t": d + "T04:00:00Z", "o": float(x), "h": float(x * 1.01), "l": float(x * 0.99),
                 "c": float(x), "v": 2_000_000} for d, x in zip(ds, c)]

    bars = {"AAA": series(50), "BBB": series(20), "SPY": series(400), "VXX": series(30)}
    plain = v2._build_training_data(bars, {})
    assert len(plain) == 3
    X, y, reg, meta = v2._build_training_data(bars, {}, return_meta=True)
    assert np.array_equal(X, plain[0]) and np.array_equal(y, plain[1]) and reg == plain[2]
    assert len(meta["dates"]) == len(meta["tickers"]) == len(meta["label_end_dates"]) == len(y)
    assert all(e > d for d, e in zip(meta["dates"], meta["label_end_dates"]))
    # the builder's ticker-major order is exactly what the legacy CV leaks on
    assert meta["tickers"][0] != meta["tickers"][-1]
    assert meta["dates"] != sorted(meta["dates"])
