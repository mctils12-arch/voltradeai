"""[RULE-REVIEW] 2026-11-06 — v2 training honesty.

The live v2 fit's last-10,000-row window was all shadow rows, one per SCAN
(production 2026-11-06: 10,845 labeled +10d rows = 209 candidate-days, 51.9x),
and its CV split rows by INDEX, so its reported accuracy and isotonic
calibration were in-sample. Pinned here:
  1. load_shadow_data keeps ONE row per (ticker, ET entry day): the
     first-seen labeled record (earliest timestamp), and can return each
     row's decision date and label end date;
  2. feedback rows carry their ET decision date and label end date;
  3. the dated fit (_fit_v2_honest): walk-forward test folds never share a
     date with the rows that trained or early-stopped their model, sit more
     than `embargo_dates` dates after them, and every training row's label
     window closed before the test block; the production holdout is the last
     dates, embargoed and purged the same way — whatever the input row order;
  4. the fast-mode window keeps the most recent WHOLE dates;
  5. end to end through _train_model_impl: switch ON dedupes and reports the
     honest record; switch OFF is the prior procedure;
  6. one config parser for the trainer and the signals table.
Every test fails on the pre-change code (the arguments, helpers and switch
did not exist)."""
import inspect
import json
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import ml_cv
import ml_model_v2 as v2
import shadow_portfolio as sp


@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "SHADOW_LOG_PATH", str(tmp_path / "shadow.json"))
    monkeypatch.setattr(sp, "SHADOW_LOCK_PATH", str(tmp_path / "shadow.json.lock"))
    monkeypatch.setattr(sp, "SHADOW_STATS_PATH", str(tmp_path / "shadow_stats.json"))
    sp._shadow_log_cache_reset()
    yield tmp_path
    sp._shadow_log_cache_reset()


def _srec(ticker, ts, label, rsi=50.0, exit_date="2026-10-15", features=True):
    out = {"label": label, "return_pct": 2.0 if label == 1 else -4.0,
           "exit_reason": "target" if label == 1 else "stop", "label_method": "path_dependent_v2"}
    if exit_date is not None:
        out["exit_date"] = exit_date
    return {"ticker": ticker, "timestamp": ts, "score": 60.0, "decision": "rejected_score",
            "entry_price": 10.0, "features": {"rsi_14": rsi, "atr_pct": 2.0} if features else {},
            "outcomes": {"+5d": None, "+10d": out, "+20d": None}}


def _write(path, recs):
    with open(path, "w") as fh:
        json.dump(recs, fh)
    sp._shadow_log_cache_reset()


LOG = [
    _srec("AAA", "2026-10-05T14:00:00+00:00", 1, rsi=11),   # ET 10-05 10:00 — first AAA/10-05
    _srec("AAA", "2026-10-05T16:00:00+00:00", 0, rsi=12),   # later scan, same ticker-day
    _srec("AAA", "2026-10-06T03:30:00+00:00", 0, rsi=13),   # ET 10-05 23:30: SAME ET day, other UTC day
    _srec("AAA", "2026-10-06T14:00:00+00:00", 0, rsi=14, exit_date=None),   # ET 10-06; no exit_date
    _srec("BBB", "2026-10-05T13:45:00+00:00", -1, rsi=21),  # unlabelable: never a row
    _srec("BBB", "2026-10-05T15:00:00+00:00", 1, rsi=22),   # first LABELED BBB/10-05
    _srec("BBB", "2026-10-05T17:00:00+00:00", 0, rsi=23, features=False),   # no features: never a row
    _srec("CCC", "2026-10-05T19:00:00+00:00", 0, rsi=31),
    _srec("CCC", "2026-10-05T14:30:00+00:00", 1, rsi=32),   # logged later, EARLIER scan: it wins
]
RSI = v2.FEATURE_COLS.index("rsi_14")


# ── 1. the shadow dedupe ─────────────────────────────────────────────────────

def test_dedupe_keeps_the_first_seen_labeled_record_per_ticker_et_day(tmp_log):
    _write(sp.SHADOW_LOG_PATH, LOG)
    X, y, w = sp.load_shadow_data(horizon_days=10, min_records=1)
    assert X[:, RSI].tolist() == [11, 14, 22, 32], "one row per (ticker, ET day): the earliest labeled scan"
    assert y.tolist() == [1, 0, 1, 1]
    assert len(w) == 4 and np.all(w > 0)


def test_dedupe_off_is_every_labeled_scan_row_as_before(tmp_log):
    _write(sp.SHADOW_LOG_PATH, LOG)
    X, y, _w = sp.load_shadow_data(horizon_days=10, min_records=1, dedupe=False)
    assert X[:, RSI].tolist() == [11, 12, 13, 14, 22, 31, 32]
    assert y.tolist() == [1, 0, 0, 0, 1, 0, 1]


def test_min_records_counts_candidate_days_not_scan_copies(tmp_log):
    _write(sp.SHADOW_LOG_PATH, LOG)
    assert sp.load_shadow_data(horizon_days=10, min_records=5) is None            # 4 candidate-days
    assert sp.load_shadow_data(horizon_days=10, min_records=5, dedupe=False) is not None   # 7 copies


def test_meta_carries_et_decision_dates_label_ends_and_counts(tmp_log):
    _write(sp.SHADOW_LOG_PATH, LOG)
    out = sp.load_shadow_data(horizon_days=10, min_records=1, return_meta=True)
    assert len(out) == 4
    meta = out[3]
    assert meta["dates"] == ["2026-10-05", "2026-10-06", "2026-10-05", "2026-10-05"]
    assert meta["tickers"] == ["AAA", "AAA", "BBB", "CCC"]
    # exit_date when the outcome has one; else timestamp + int(10 * 2.5) + 3 days (the backfill's bar window end)
    assert meta["label_end_dates"] == ["2026-10-15", "2026-11-03", "2026-10-15", "2026-10-15"]
    assert meta["rows_labeled"] == 7 and meta["deduplicated"] is True
    assert len(sp.load_shadow_data(horizon_days=10, min_records=1)) == 3, "default shape unchanged"


def test_label_window_buffer_is_the_backfills_own():
    m = re.search(r"HORIZON_CALENDAR_BUFFER\s*=\s*([0-9.]+)", inspect.getsource(sp.backfill_outcomes))
    assert m and float(m.group(1)) == sp.SHADOW_LABEL_WINDOW_BUFFER


def test_the_trainer_asks_for_deduplicated_dated_shadow_rows():
    src = inspect.getsource(v2._train_model_impl)
    assert "load_shadow_data(horizon_days=10, shadow_weight=0.3, dedupe=honest, return_meta=honest)" in src, \
        "the 0.3 shadow weight is unchanged; dedupe + dates follow the switch"


# ── 2. feedback rows carry dates ────────────────────────────────────────────

def _fb(i, placed, exit_time=None, pnl=3.0):
    t = {"ticker": f"T{i}", "entry_features": {"atr_pct": 2.0, "rsi_14": 40.0 + i}, "pnl_pct": pnl,
         "timestamp": placed, "time_placed": placed}
    if exit_time is not None:
        t["exit_time"] = exit_time
    return t


def test_feedback_meta_dates_are_et_session_dates():
    trades = [_fb(i, "2026-10-05T19:30:00Z", "2026-10-08T15:00:00Z") for i in range(14)]
    trades.append(_fb(99, "2026-10-07T01:00:00Z"))       # ET 10-06 21:00; still open
    out = v2._build_feedback_training_data(trades, return_meta=True)
    assert len(out) == 5
    meta = out[4]
    assert meta["dates"][:14] == ["2026-10-05"] * 14 and meta["label_end_dates"][:14] == ["2026-10-08"] * 14
    today_et = v2._et_session_date(datetime.now(timezone.utc).timestamp())
    assert meta["dates"][14] == "2026-10-06" and meta["label_end_dates"][14] == today_et, \
        "an open position's label window is bounded by today"
    assert len(v2._build_feedback_training_data(trades)) == 4, "default shape unchanged"
    assert v2._build_feedback_training_data(trades[:3], return_meta=True) == (None,) * 5


def test_et_session_date_reads_iso_epoch_and_bare_dates():
    assert v2._et_session_date("2026-10-06T03:30:00Z") == "2026-10-05"
    assert v2._et_session_date(datetime(2026, 10, 6, 14, 0, tzinfo=timezone.utc).timestamp()) == "2026-10-06"
    assert v2._et_session_date("2026-10-06") == "2026-10-06"
    for bad in (None, "", 0, -5, True, "not a date"):
        assert v2._et_session_date(bad) is None


# ── 3. the dated CV ──────────────────────────────────────────────────────────

N_TICKERS, N_DATES, E = 30, 120, 11
DATE_AXIS = [(datetime(2026, 1, 5) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(N_DATES + 15)]
ORD = {d: i for i, d in enumerate(DATE_AXIS)}


def _ticker_major(seed=3, n_shadow=60):
    """Generic-like rows emitted TICKER BY TICKER (the order _build_training_data
    produces) + a shadow block; column 0 is the row id."""
    rng = np.random.default_rng(seed)
    dates, ends = [], []
    for _t in range(N_TICKERS):
        for i in range(N_DATES):
            dates.append(DATE_AXIS[i]); ends.append(DATE_AXIS[i + 5])
    sh_dates, sh_ends = [], []
    for j in range(n_shadow):
        i = int(rng.integers(40, N_DATES))
        sh_dates.append(DATE_AXIS[i]); sh_ends.append(DATE_AXIS[i + 10])
    n = len(dates) + n_shadow
    X = rng.normal(size=(n, len(v2.FEATURE_COLS))).astype(np.float32)
    X[:, 0] = np.arange(n)
    y = (rng.random(n) < 0.45).astype(np.int8)
    blocks = [(dates, ends, "generic"), (sh_dates, sh_ends, "shadow")]
    return X, y, blocks, dates + sh_dates, ends + sh_ends


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(v2, "CALIBRATOR_PATH", str(tmp_path / "cal.pkl"))
    fits, preds = {}, []
    real_fit, real_pred = v2._train_single_lgbm, v2._predict_v2

    def fit(X_tr, X_te, y_tr, y_te, label="", sample_weight=None):
        m, s, a = real_fit(X_tr, X_te, y_tr, y_te, label, sample_weight=sample_weight)
        fits[id(m)] = {"label": label, "train": X_tr[:, 0].astype(int).tolist(),
                       "holdout": X_te[:, 0].astype(int).tolist(), "model": m}
        return m, s, a

    def pred(model, scaler, X_raw):
        preds.append((id(model), X_raw[:, 0].astype(int).tolist()))
        return real_pred(model, scaler, X_raw)

    monkeypatch.setattr(v2, "_train_single_lgbm", fit)
    monkeypatch.setattr(v2, "_predict_v2", pred)
    return fits, preds


def test_dated_cv_folds_never_share_a_date_and_respect_the_embargo_and_purge(recorded):
    fits, preds = recorded
    X, y, blocks, dates, ends = _ticker_major()
    reg = ["neutral"] * len(y)
    out = v2._fit_v2_honest(X, y, reg, np.ones(len(y), np.float32), blocks, fast_mode=False,
                            cfg=dict(ml_cv.V2_HONEST_TRAINING_DEFAULTS))
    assert out["status"] is None
    # the fit sorts rows by date: map its row ids (column 0) back to the input's dates
    d_of = {int(i): dates[int(i)] for i in X[:, 0]}
    e_of = {int(i): ends[int(i)] for i in X[:, 0]}
    assert len(preds) >= 3, "at least three walk-forward folds scored"
    seen_test = set()
    for mid, test_ids in preds:
        f = fits[mid]
        fitted = f["train"] + f["holdout"]
        t_ords = sorted({ORD[d_of[i]] for i in test_ids})
        assert t_ords == list(range(t_ords[0], t_ords[-1] + 1)), "a test fold is one contiguous block of dates"
        assert not ({d_of[i] for i in test_ids} & {d_of[i] for i in fitted}), "no date on both sides"
        assert t_ords[0] - max(ORD[d_of[i]] for i in fitted) > E, "embargo: > embargo_dates dates before the test block"
        assert max(e_of[i] for i in fitted) < DATE_AXIS[t_ords[0]], "purge: every fitted row's label closed first"
        # early stopping (and the LightGBM-vs-GB choice) used the training window's own last dates
        assert min(ORD[d_of[i]] for i in f["holdout"]) - max(ORD[d_of[i]] for i in f["train"]) > E
        assert max(e_of[i] for i in f["train"]) < d_of[min(f["holdout"], key=lambda i: ORD[d_of[i]])]
        assert not (seen_test & set(test_ids)), "test folds are disjoint"
        seen_test |= set(test_ids)
    prod = [f for f in fits.values() if f["label"] == "global"]
    assert len(prod) == 1
    p = prod[0]
    ho = sorted({ORD[d_of[i]] for i in p["holdout"]})
    assert ho[-1] == N_DATES - 1 and len(ho) == round(N_DATES * 0.2), "production early-stops on the last 20% of dates"
    assert ho[0] - max(ORD[d_of[i]] for i in p["train"]) > E
    assert max(e_of[i] for i in p["train"]) < DATE_AXIS[ho[0]]
    rep = out["honest_cv"]
    assert rep["rows_tested"] == len(seen_test) and rep["pooled"]["status"] == "OK"
    assert set(rep["pooled"]) >= {"auc", "log_loss", "brier", "accuracy", "base_rate", "base_rate_log_loss",
                                  "base_rate_brier", "majority_accuracy", "n_dates"}
    assert rep["pooled"]["majority_accuracy"] == pytest.approx(max(rep["pooled"]["base_rate"], 1 - rep["pooled"]["base_rate"]), abs=1e-4)
    assert rep["skill"] is False, "random labels: no skill"


def test_the_legacy_row_index_splitter_breaks_those_invariants_on_the_same_rows():
    """Why the change: on ticker-major rows the legacy folds put the same dates on both sides."""
    X, y, blocks, dates, _ends = _ticker_major()
    shared = 0
    for tr, te in v2._walk_forward_cv(X, y, n_splits=5, embargo_days=5, n_tickers_approx=N_TICKERS):
        shared += len({dates[i] for i in tr} & {dates[i] for i in te})
    assert shared > 0


# ── 4. the fast-mode window ──────────────────────────────────────────────────

def test_recent_whole_dates_keeps_complete_most_recent_dates():
    dates = ["2026-01-02"] * 3 + ["2026-01-05"] * 4 + ["2026-01-01"] * 2 + ["2026-01-06"] * 3
    pos = v2._recent_whole_dates(dates, 8)
    kept = [dates[i] for i in pos]
    assert kept == ["2026-01-05"] * 4 + ["2026-01-06"] * 3, "whole dates, newest first, chronological output"
    assert len(v2._recent_whole_dates(dates, 100)) == len(dates)
    assert len(v2._recent_whole_dates(["2026-01-06"] * 5, 3)) == 3, "one date over the cap keeps the cap"


def test_fast_mode_window_is_by_date_not_by_row(recorded, monkeypatch):
    monkeypatch.setattr(v2, "HONEST_WINDOW_ROWS", 1500)
    X, y, blocks, dates, _ends = _ticker_major(n_shadow=0)
    blocks = blocks[:1]
    out = v2._fit_v2_honest(X, y, ["neutral"] * len(y), None, blocks, fast_mode=True,
                            cfg=dict(ml_cv.V2_HONEST_TRAINING_DEFAULTS))
    ts = out["training_set"]
    assert ts["rows"] == 50 * N_TICKERS and ts["dates"] == 50, "the newest 50 whole dates of every ticker"
    assert ts["date_range"] == [DATE_AXIS[N_DATES - 50], DATE_AXIS[N_DATES - 1]]
    # the legacy cut on the same rows: the last 1,500 ROWS = whole histories of the last 12.5 tickers
    tail = dates[-1500:]
    assert len(set(tail)) == N_DATES


# ── 5. end to end through _train_model_impl ─────────────────────────────────

def _bars(seed, n=260, start=datetime(2025, 10, 1)):
    rng = np.random.default_rng(seed)
    ds, d = [], start
    while len(ds) < n:
        if d.weekday() < 5:
            ds.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return [{"t": x + "T04:00:00Z", "o": float(p * (1 + rng.normal(0, 0.003))), "h": float(p * 1.012),
             "l": float(p * 0.988), "c": float(p), "v": 2_000_000.0} for x, p in zip(ds, c)]


@pytest.fixture
def train_env(tmp_log, tmp_path, monkeypatch):
    bars = {f"S{i}": _bars(i) for i in range(10)}
    bars.update({"SPY": _bars(101), "VXX": _bars(102), "TLT": _bars(103), "HYG": _bars(104)})
    monkeypatch.setattr(v2, "_fetch_training_bars", lambda **kw: (json.loads(json.dumps(bars)), None))
    monkeypatch.setattr(v2, "_load_trade_feedback", lambda: [])
    monkeypatch.setattr(v2, "MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(v2, "CALIBRATOR_PATH", str(tmp_path / "cal.pkl"))
    rng = np.random.default_rng(9)
    recs = []
    days = [b["t"][:10] for b in bars["SPY"]][150:230]
    for k in range(60):                                   # 60 candidate-days x 4 scan copies
        day = days[k]
        lab = int(rng.random() < 0.45)
        for c in range(4):
            f = {col: float(rng.normal()) for col in v2.FEATURE_COLS}
            recs.append(_srec(f"S{k % 10}", f"{day}T{14 + c}:00:00+00:00", lab, exit_date=days[k + 10]
                              if k + 10 < len(days) else None) | {"features": f})
    _write(sp.SHADOW_LOG_PATH, recs)
    calls = []
    real = sp.load_shadow_data

    def spy_load(**kw):
        calls.append(kw)
        return real(**kw)
    monkeypatch.setattr(sp, "load_shadow_data", spy_load)
    return calls


def test_switch_on_trains_on_deduplicated_dated_rows_and_reports_the_honest_record(train_env, monkeypatch):
    monkeypatch.setitem(v2.BASE_CONFIG, "ML_V2_HONEST_TRAINING", dict(ml_cv.V2_HONEST_TRAINING_DEFAULTS))
    r = v2.train_model(fast_mode=True)
    assert r["status"] == "trained", r
    assert train_env[-1]["dedupe"] is True and train_env[-1]["return_meta"] is True
    assert r["training_set"]["shadow"] == {"rows_labeled": 240, "rows_used": 60}
    assert r["training_set"]["by_source"]["shadow"] == 60 and r["training_set"]["undated_dropped"] == 0
    assert r["cv_method"] == "purged_walk_forward_by_date"
    hc = r["honest_cv"]
    assert hc["pooled"]["status"] == "OK" and hc["skill"] in (True, False)
    assert hc["by_source"]["shadow"]["n"] <= 60
    assert "+shadow(60 first-seen ticker-days of 240 labeled scan rows)" in r["data_source"]
    import joblib
    bundle = joblib.load(v2.MODEL_PATH)
    assert bundle["cv_method"] == "purged_walk_forward_by_date" and bundle["honest_cv"]["pooled"] == hc["pooled"]


def test_switch_off_is_the_prior_procedure(train_env, monkeypatch):
    monkeypatch.setitem(v2.BASE_CONFIG, "ML_V2_HONEST_TRAINING", {**ml_cv.V2_HONEST_TRAINING_DEFAULTS, "enabled": False})
    r = v2.train_model(fast_mode=True)
    assert r["status"] == "trained", r
    assert train_env[-1]["dedupe"] is False and train_env[-1]["return_meta"] is False
    assert r["cv_method"] == "walk_forward_5fold" and "honest_cv" not in r and "training_set" not in r
    assert "shadow" not in r["data_source"]
    import joblib
    assert joblib.load(v2.MODEL_PATH)["cv_method"] == "walk_forward_5fold_purged"


# ── 6. one parser for the switch ─────────────────────────────────────────────

def test_the_shipped_config_is_the_parsers_defaults_and_on():
    from system_config import BASE_CONFIG
    assert BASE_CONFIG["ML_V2_HONEST_TRAINING"] == ml_cv.V2_HONEST_TRAINING_DEFAULTS
    assert ml_cv.V2_HONEST_TRAINING_DEFAULTS["enabled"] is True
    assert ml_cv.V2_HONEST_TRAINING_DEFAULTS["embargo_dates"] > 10, "embargo exceeds the +10-trading-day shadow label"


@pytest.mark.parametrize("raw,enabled,problem", [
    (None, True, False),
    ({"enabled": False}, False, False),
    ({"enabled": True, "embargo_dates": 21}, True, False),
    ("off", True, True),
    ({"enabled": "no"}, True, True),
    ({"enabled": False, "holdout_frac": 0.9}, True, True),
    ({"enabled": True, "embargo_dates": True}, True, True),
])
def test_config_parser_and_the_signals_table_agree(raw, enabled, problem, monkeypatch):
    import signal_attribution as sa
    cfg, prob = ml_cv.v2_honest_training_cfg(raw)
    assert cfg["enabled"] is enabled and bool(prob) is problem
    monkeypatch.setitem(v2.BASE_CONFIG, "ML_V2_HONEST_TRAINING", raw)
    assert v2._honest_training_cfg()["enabled"] is enabled
    assert sa.v2_fit_deduplicated(raw) is enabled
