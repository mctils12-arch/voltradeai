"""ml_cv.py — honest out-of-time cross-validation for the stock ML models.

[RULE-REVIEW] measurement code (CLAUDE.md MEASUREMENT INTEGRITY), 2026-10-27.

WHY (audit 2026-10-27 §5.3/§5.5, research/bot_cockpit_audit.md V15).
ml_model_v2's _walk_forward_cv / _purged_train_test_split split rows by ROW
INDEX. _build_training_data emits rows TICKER BY TICKER, and nothing sorts
them by date, so every "walk-forward" test fold is a block of whole tickers
covering the SAME calendar window as the training rows; the "5-day embargo"
is 5 x n_tickers ROWS (~1.7 tickers), not 5 days; and early stopping (plus
the LGB-vs-GB fallback) is chosen on the test fold itself. With 11 of 34
features being pure functions of the date, the model can learn "these
dates hit PT" from training tickers and get credit for it on test tickers
of the same dates. The reported accuracy is contemporaneous cross-sectional
fit, not out-of-time skill.

WHAT THIS MODULE DOES INSTEAD (pure numpy; LightGBM only inside
fit_lgbm_inner_es):
  1. Rows are addressed by DECISION DATE, never by row position.
  2. purged_walk_forward_splits: the unique decision dates are cut into
     n_splits+1 contiguous blocks; fold k tests block k and trains only on
     rows whose decision date is more than `embargo_dates` trading dates
     before the test block AND (when label end dates are given) whose label
     window closed strictly before the test block starts (purge).
     embargo_dates must be >= the label horizon (asserted by callers).
  3. fit_lgbm_inner_es: early stopping runs on an INNER validation slice
     carved from the LAST dates of the training window (with the same
     embargo/purge between inner-train and inner-validation). The test fold
     is never seen while fitting. The tree count found there is then refit
     on the whole training window.
  4. Metrics a probability model owes its reader: AUC, log-loss, Brier,
     accuracy, base rate, calibration table, quintile realized returns —
     not accuracy alone.

Bias direction of the change: the honest numbers can only be LOWER than the
leaky ones on the same data (removing leaks removes information); a
measurement change that made the existing model look better would be
suspect by default — this one cannot.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_N_SPLITS = 5
DEFAULT_INNER_VAL_FRAC = 0.2
DEFAULT_ES_ROUNDS = 30
DEFAULT_NUM_BOOST_ROUND = 300
_EPS = 1e-6


# ── Date bookkeeping ─────────────────────────────────────────────────────────

def _as_date_strings(dates: Sequence) -> np.ndarray:
    return np.array([str(d)[:10] for d in dates], dtype=object)


def date_ordinals(dates: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """(unique sorted dates, per-row ordinal into them)."""
    ds = _as_date_strings(dates)
    uniq = np.array(sorted(set(ds.tolist())), dtype=object)
    pos = {d: i for i, d in enumerate(uniq.tolist())}
    return uniq, np.array([pos[d] for d in ds.tolist()], dtype=np.int64)


def chronological_order(dates: Sequence) -> np.ndarray:
    """Stable argsort by decision date (ties keep their input order)."""
    ds = _as_date_strings(dates)
    return np.array(sorted(range(len(ds)), key=lambda i: ds[i]), dtype=np.int64)


def _end_ordinals(uniq: np.ndarray, label_end_dates: Optional[Sequence]) -> Optional[np.ndarray]:
    """Ordinal of each row's label end date on the decision-date axis. A label
    end that falls between decision dates maps to the first decision date on or
    after it (searchsorted 'left'), so `end_ord < test_start` means the label
    window closed strictly before the test block's first decision date."""
    if label_end_dates is None:
        return None
    ends = _as_date_strings(label_end_dates)
    return np.searchsorted(uniq.astype(str), ends.astype(str), side="left").astype(np.int64)


# ── Splitters ────────────────────────────────────────────────────────────────

def purged_walk_forward_splits(dates: Sequence,
                               n_splits: int = DEFAULT_N_SPLITS,
                               embargo_dates: int = 20,
                               label_end_dates: Optional[Sequence] = None,
                               min_train_dates: int = 20,
                               min_test_rows: int = 10,
                               ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward folds keyed by decision date.

    Fold k (1..n_splits) tests the rows of date block k and trains on rows
    whose date ordinal < test_start - embargo_dates and (if label_end_dates)
    whose label end ordinal < test_start. Index arrays point into the INPUT
    arrays (any row order) and are returned sorted chronologically.
    """
    if embargo_dates < 0:
        raise ValueError("embargo_dates must be >= 0")
    uniq, ords = date_ordinals(dates)
    end_ords = _end_ordinals(uniq, label_end_dates)
    n_dates = len(uniq)
    if n_dates < n_splits + 1:
        return []
    block = n_dates // (n_splits + 1)
    if block < 1:
        return []
    order = chronological_order(dates)
    ords_sorted = ords[order]
    splits = []
    for k in range(1, n_splits + 1):
        test_start = k * block
        test_end = (k + 1) * block if k < n_splits else n_dates
        train_cut = test_start - embargo_dates
        if train_cut < min_train_dates:
            continue
        tr_mask = ords_sorted < train_cut
        if end_ords is not None:
            tr_mask &= end_ords[order] < test_start
        te_mask = (ords_sorted >= test_start) & (ords_sorted < test_end)
        tr, te = order[tr_mask], order[te_mask]
        if len(te) < min_test_rows or len(tr) == 0:
            continue
        splits.append((tr, te))
    return splits


def inner_validation_split(idx: np.ndarray, dates: Sequence,
                           frac: float = DEFAULT_INNER_VAL_FRAC,
                           embargo_dates: int = 20,
                           label_end_dates: Optional[Sequence] = None,
                           min_rows: int = 30,
                           ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Carve an early-stopping slice from a TRAINING window only.

    The last `frac` of the window's unique dates is the inner validation set;
    inner-train keeps rows dated more than `embargo_dates` dates before it
    (and, with label_end_dates, whose label closed before it). None when
    either side would be too small — the caller then fits a fixed number of
    rounds instead of early-stopping (never on the test fold).
    """
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return None
    all_dates = _as_date_strings(dates)
    uniq, ords = date_ordinals(all_dates)
    end_ords = _end_ordinals(uniq, label_end_dates)
    win = sorted(set(ords[idx].tolist()))
    n_val = max(1, int(round(len(win) * frac)))
    if len(win) - n_val - embargo_dates < 1:
        return None
    val_start = win[len(win) - n_val]
    cut_pos = len(win) - n_val - embargo_dates
    train_cut = win[cut_pos]  # inner-train dates must be < this ordinal
    w_ords = ords[idx]
    tr_mask = w_ords < train_cut
    if end_ords is not None:
        tr_mask &= end_ords[idx] < val_start
    va_mask = w_ords >= val_start
    tr, va = idx[tr_mask], idx[va_mask]
    if len(tr) < min_rows or len(va) < max(10, min_rows // 3):
        return None
    return tr, va


# ── Model fitting with leak-free early stopping ─────────────────────────────

LGBM_V3_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.03, "num_leaves": 15,
    "min_child_samples": 100,
    "feature_fraction": 0.7, "bagging_fraction": 0.7,
    "bagging_freq": 5, "lambda_l2": 5.0,
    # num_threads=1: the job runs as one serialized subprocess (bot.ts
    # execPythonSerialized) next to the daemon; OpenMP spin-waits on a
    # CPU-throttled container made 2 threads ~10x SLOWER than 1 when measured
    # in the session sandbox (0.29s vs 3.4s for 50 rounds on 1.5k rows).
    "verbose": -1, "num_threads": 1, "seed": 7,
    "deterministic": True,
}


def fit_lgbm_inner_es(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                      dates: Sequence,
                      params: Optional[dict] = None,
                      sample_weight: Optional[np.ndarray] = None,
                      label_end_dates: Optional[Sequence] = None,
                      num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
                      es_rounds: int = DEFAULT_ES_ROUNDS,
                      inner_frac: float = DEFAULT_INNER_VAL_FRAC,
                      embargo_dates: int = 20,
                      feature_names: Optional[List[str]] = None):
    """Fit LightGBM on train_idx only. Early stopping uses an inner slice of
    the training window (inner_validation_split); the found tree count is
    refit on the whole window. Returns (booster, info)."""
    import lightgbm as lgb
    p = dict(LGBM_V3_PARAMS if params is None else params)
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    w = None if sample_weight is None else np.asarray(sample_weight)
    split = inner_validation_split(train_idx, dates, frac=inner_frac,
                                   embargo_dates=embargo_dates,
                                   label_end_dates=label_end_dates)
    best_iter = None
    inner_logloss = None
    if split is not None:
        itr, iva = split
        dtr = lgb.Dataset(X[itr], label=y[itr], weight=None if w is None else w[itr],
                          feature_name=names, free_raw_data=True)
        dva = lgb.Dataset(X[iva], label=y[iva], reference=dtr, free_raw_data=True)
        m = lgb.train(p, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                                 lgb.log_evaluation(period=-1)])
        best_iter = int(m.best_iteration or num_boost_round)
        try:
            inner_logloss = float(m.best_score["valid_0"]["binary_logloss"])
        except Exception:
            inner_logloss = None
    n_rounds = best_iter if best_iter else min(100, num_boost_round)
    dall = lgb.Dataset(X[train_idx], label=y[train_idx],
                       weight=None if w is None else w[train_idx],
                       feature_name=names, free_raw_data=True)
    model = lgb.train(p, dall, num_boost_round=max(1, n_rounds))
    return model, {"best_iteration": best_iter, "rounds": int(max(1, n_rounds)),
                   "inner_val_logloss": inner_logloss,
                   "early_stopping": "inner_validation" if split is not None else "none_fixed_rounds"}


# ── Metrics ──────────────────────────────────────────────────────────────────

def _avg_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auc(y: Sequence, p: Sequence) -> Optional[float]:
    """ROC AUC via the Mann-Whitney rank statistic (ties averaged). None when
    only one class is present."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = _avg_ranks(p)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(y: Sequence, p: Sequence) -> Dict[str, Optional[float]]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    if len(y) == 0:
        return {"n": 0, "auc": None, "log_loss": None, "brier": None, "accuracy": None,
                "base_rate": None, "base_rate_log_loss": None, "base_rate_brier": None}
    base = float(y.mean())
    b = min(max(base, _EPS), 1 - _EPS)
    return {
        "n": int(len(y)),
        "auc": auc(y, p),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
        "base_rate": base,
        # what a constant forecast of the base rate scores — the bar to beat
        "base_rate_log_loss": float(-(base * np.log(b) + (1 - base) * np.log(1 - b))),
        "base_rate_brier": float(base * (1 - base)),
        "mean_pred": float(p.mean()),
    }


def calibration_table(y: Sequence, p: Sequence, n_bins: int = 10) -> List[dict]:
    """Equal-count bins of predicted probability: mean prediction vs realized
    frequency per bin."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) < n_bins:
        return []
    order = np.argsort(p, kind="mergesort")
    out = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        out.append({"n": int(len(chunk)), "mean_pred": round(float(p[chunk].mean()), 4),
                    "realized": round(float(y[chunk].mean()), 4)})
    return out


def quantile_returns(p: Sequence, ret: Sequence, q: int = 5) -> dict:
    """Realized mean/median return per predicted-probability quantile, plus
    the top-minus-bottom spread and the all-rows mean (the base rate: random
    entry under the same exit rule and universe)."""
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(ret, dtype=np.float64)
    if len(p) < q:
        return {"quantiles": [], "top_minus_bottom": None, "all_mean": None}
    order = np.argsort(p, kind="mergesort")
    rows = []
    for i, chunk in enumerate(np.array_split(order, q)):
        rows.append({"q": i + 1, "n": int(len(chunk)),
                     "mean_pred": round(float(p[chunk].mean()), 4),
                     "mean_ret": round(float(r[chunk].mean()), 4),
                     "median_ret": round(float(np.median(r[chunk])), 4),
                     "win_rate": round(float((r[chunk] > 0).mean()), 4)})
    return {"quantiles": rows,
            "top_minus_bottom": round(rows[-1]["mean_ret"] - rows[0]["mean_ret"], 4),
            "all_mean": round(float(r.mean()), 4)}


def block_bootstrap_ci(y: Sequence, p: Sequence, dates: Sequence,
                       stat: Callable[[np.ndarray, np.ndarray], Optional[float]],
                       n_boot: int = 200, seed: int = 11, alpha: float = 0.05,
                       ) -> Optional[Tuple[float, float]]:
    """Percentile CI of stat(y, p) resampling whole DECISION DATES with
    replacement (same-day rows are correlated through the market, so rows are
    not independent draws)."""
    y = np.asarray(y)
    p = np.asarray(p)
    uniq, ords = date_ordinals(dates)
    if len(uniq) < 10:
        return None
    by_date = [np.where(ords == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by_date[k] for k in pick])
        v = stat(y[idx], p[idx])
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot // 2:
        return None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 4), round(float(hi), 4)


# ── Full honest CV run ──────────────────────────────────────────────────────

def honest_cv(X: np.ndarray, y: np.ndarray, dates: Sequence,
              label_end_dates: Optional[Sequence] = None,
              embargo_dates: int = 20,
              n_splits: int = DEFAULT_N_SPLITS,
              params: Optional[dict] = None,
              sample_weight: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None) -> dict:
    """Date-ordered purged walk-forward with inner-validation early stopping.
    Returns {"oos_prob": array (NaN where a row was never tested), "folds":
    [...], "pooled": binary_metrics over all OOS rows}."""
    splits = purged_walk_forward_splits(dates, n_splits=n_splits, embargo_dates=embargo_dates,
                                        label_end_dates=label_end_dates)
    oos = np.full(len(y), np.nan)
    ds = _as_date_strings(dates)
    folds = []
    for k, (tr, te) in enumerate(splits):
        if len(tr) < 200 or len(np.unique(y[tr])) < 2:
            continue
        model, info = fit_lgbm_inner_es(X, y, tr, dates, params=params,
                                        sample_weight=sample_weight,
                                        label_end_dates=label_end_dates,
                                        embargo_dates=embargo_dates,
                                        feature_names=feature_names)
        prob = model.predict(X[te])
        oos[te] = prob
        m = binary_metrics(y[te], prob)
        m.update({"fold": k, "train_rows": int(len(tr)), "test_rows": int(len(te)),
                  "train_last_date": str(max(ds[tr])), "test_first_date": str(min(ds[te])),
                  "test_last_date": str(max(ds[te])), **info})
        folds.append(m)
    tested = ~np.isnan(oos)
    return {"oos_prob": oos, "folds": folds,
            "pooled": binary_metrics(y[tested], oos[tested]) if tested.any() else binary_metrics([], []),
            "method": f"purged_walk_forward_by_date(n_splits={n_splits}, embargo_dates={embargo_dates}, "
                      f"purge={'label_end' if label_end_dates is not None else 'none'}, "
                      f"early_stopping=inner_validation_{DEFAULT_INNER_VAL_FRAC})"}


def legacy_cv(X: np.ndarray, y: np.ndarray, n_tickers: int) -> dict:
    """The CURRENT ml_model_v2 procedure, reproduced for a before/after on
    identical inputs: rows in the order given (the builder's ticker-major
    order), ml_model_v2._walk_forward_cv row-index folds with an embargo of
    5 x n_tickers ROWS, and _train_single_lgbm (early stopping on the test
    fold, GB fallback chosen on the test fold)."""
    import ml_model_v2 as v2
    splits = v2._walk_forward_cv(X, y, n_splits=5, embargo_days=5, n_tickers_approx=n_tickers)
    oos = np.full(len(y), np.nan)
    folds = []
    for k, (tr, te) in enumerate(splits):
        tr = np.asarray(tr)
        te = np.asarray(te)
        if len(tr) < 50 or len(te) < 10:
            continue
        m, s, acc = v2._train_single_lgbm(X[tr], X[te], y[tr], y[te], f"legacy{k}")
        if m is None:
            continue
        import pandas as pd
        xs = s.transform(pd.DataFrame(X[te], columns=v2.FEATURE_COLS))
        if hasattr(m, "feature_name_") or hasattr(m, "best_iteration"):
            prob = m.predict(pd.DataFrame(xs, columns=v2.FEATURE_COLS))
        else:
            prob = m.predict_proba(xs)[:, 1]
        oos[te] = prob
        bm = binary_metrics(y[te], prob)
        bm.update({"fold": k, "train_rows": int(len(tr)), "test_rows": int(len(te)),
                   "reported_accuracy": acc})
        folds.append(bm)
    tested = ~np.isnan(oos)
    return {"oos_prob": oos, "folds": folds,
            "pooled": binary_metrics(y[tested], oos[tested]) if tested.any() else binary_metrics([], []),
            "method": "legacy ml_model_v2._walk_forward_cv (row-index, ticker-major, 5*n_tickers-row embargo, "
                      "early stopping on the test fold)"}


def strip_arrays(result: dict) -> dict:
    """JSON-safe copy of a CV result (drops the per-row probability array)."""
    return {k: v for k, v in result.items() if k != "oos_prob"}
