"""ml_model_v3.py — the v3 CANDIDATE stock model (SHADOW ONLY).

Human Phase 4 "Fix the model" (2026-10-27). The live v2 model
(ml_model_v2.py, voltrade_ml_v2.pkl) keeps its influence on trading; this
module trains and evaluates a candidate SIDE BY SIDE and logs its score next
to v2's on every shadow record. Promotion is a later, separate decision.

The candidate differs from v2 in exactly the four audited ways:
  1. honest CV (ml_cv.py): folds by decision date, embargo >= label horizon,
     early stopping on an inner slice of the training window;
  2. no earnings-surprise look-ahead (dropped: no point-in-time source);
  3. one feature definition for train and serve (ml_features_v3.py);
  4. ONE label that matches the real exit rules (this file, below).

LABEL (exit_aligned_v1_next_open). Decision after the close of bar i
(features read bars <= i). Entry at the OPEN of bar i+1 — the honest
analogue of the bot buying intraday in the session after the scan's last
completed bar. The position is then walked bar by bar under the LIVE exit
policy, live_exit_policy.LivePosition (the pure mirror of server/bot.ts
checkPositionOnTick: 2xATR R unit, scale-outs in thirds at 1R and the
regime's 2nd R, breakeven, phase trails, conditional TP, 7-calendar-day
flat time stop), the same policy shadow_portfolio's exit-aligned ruler uses.
A path still open after LABEL_CAP_BARS bars is marked to that close
(censored). return_pct is the tranche-weighted gross return; the label is
1 iff the return NET of the round-trip cost is > 0. Rows whose path is cut
off by the end of the data are dropped, never labeled from a stub.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

import ml_features_v3 as f3
from live_exit_policy import LivePosition, regime_class, time_eligible, walk_bar

LABEL_METHOD = "exit_aligned_v1_next_open"
# == shadow_portfolio.XA_CAP_BARS (censoring cap of the exit-aligned ruler;
# live has no horizon). Drift-pinned by test_ml_model_v3.py.
LABEL_CAP_BARS = 20
# == shadow_portfolio.XA_ROUND_TRIP_COST_PCT (system_config SLIPPAGE_PCT
# 0.05% per side x 2). Drift-pinned by test_ml_model_v3.py.
ROUND_TRIP_COST_PCT = 0.10
# CV embargo in trading dates: the label reads at most LABEL_CAP_BARS bars
# after the decision bar (entry bar + 19 more), so one more than the cap
# guarantees no training label window touches a test decision date.
EMBARGO_DATES = LABEL_CAP_BARS + 1

logger = logging.getLogger("ml_model_v3")


def regime_class_at(feats: dict) -> str:
    """bot.ts isBullish/isBearish class at decision time, from the row's own
    vxx_ratio / spy_vs_ma50 through system_config.get_market_regime (the
    5-level regime manage_positions writes into stop_state)."""
    vx, sp = feats.get("vxx_ratio"), feats.get("spy_vs_ma50")
    if vx is None or sp is None or math.isnan(vx) or math.isnan(sp):
        return "neutral"
    try:
        from system_config import get_market_regime
        return regime_class(get_market_regime(float(vx), float(sp)))
    except ImportError as e:
        logger.warning(f"system_config unavailable, regime class neutral: {e}")
        return "neutral"


def label_exit_aligned(entry: float, fwd: Dict[str, np.ndarray], start: int,
                       atr_pct: float, regime_cls: str,
                       cap: int = LABEL_CAP_BARS) -> Optional[dict]:
    """Walk bars fwd[start : start+cap] (arrays from f3.to_arrays) from an
    entry at `entry` dated fwd['t'][start]. None when the data ends before
    the position exits or reaches the cap."""
    if entry <= 0 or not atr_pct or atr_pct <= 0 or math.isnan(atr_pct):
        return None
    n = len(fwd["c"])
    entry_date = str(fwd["t"][start])
    pos = LivePosition(entry, atr_pct, regime_cls)
    used = 0
    for k in range(start, min(start + cap, n)):
        used += 1
        te = time_eligible(entry_date, str(fwd["t"][k]))
        if walk_bar(pos, float(fwd["o"][k]), float(fwd["h"][k]), float(fwd["l"][k]),
                    float(fwd["c"][k]), te):
            return _result(pos, str(fwd["t"][k]), used, censored=False)
    if used < cap:
        return None
    last = start + cap - 1
    px = float(fwd["c"][last])
    pos.max_px, pos.min_px = max(pos.max_px, px), min(pos.min_px, px)
    pos._close(px, "open_at_cap")
    return _result(pos, str(fwd["t"][last]), used, censored=True)


def _result(pos: LivePosition, exit_date: str, bars_held: int, censored: bool) -> dict:
    gross = sum(w * (px - pos.entry) / pos.entry * 100.0 for w, px in pos.fills)
    net = gross - ROUND_TRIP_COST_PCT
    return {"label": 1 if net > 0 else 0, "return_pct": gross, "net_return_pct": net,
            "exit_reason": pos.exit_reason, "exit_date": exit_date,
            "bars_held": bars_held, "censored": censored}


def build_dataset(all_bars: Dict[str, Sequence[dict]],
                  tickers: Optional[Sequence[str]] = None) -> dict:
    """Features (ml_features_v3 training path) + exit-aligned labels for every
    eligible (ticker, bar) whose full LABEL_CAP_BARS forward window exists
    (pass COMPLETED bars only — strip today's session first). Rows are returned in
    chronological order (decision date, then ticker)."""
    refs = f3.RefSeries({s: all_bars.get(s) or [] for s in f3.REF_SYMBOLS})
    names = sorted(t for t in (tickers or all_bars) if t not in f3.REF_SYMBOLS and all_bars.get(t))
    X, y, dates, ends, tick, gross, net, reg, why = [], [], [], [], [], [], [], [], []
    for t in names:
        a = f3.to_arrays(all_bars[t])
        n = len(a["c"])
        if n < f3.MIN_BARS + 1:
            continue
        spy = refs.spy_asof(list(a["t"]))
        # Only decisions whose WHOLE cap window exists: keeping late rows
        # that happened to exit early (and dropping the still-open ones)
        # would bias the sample's tail toward fast stops/time stops.
        for i in range(f3.MAX_LOOKBACK, n - LABEL_CAP_BARS):
            if not f3.row_eligible(a, i):
                continue
            feats = f3.features_at(a, i, refs, spy)
            if feats is None:
                continue
            rc = regime_class_at(feats)
            lab = label_exit_aligned(float(a["o"][i + 1]), a, i + 1, feats["atr_pct"], rc)
            if lab is None:
                continue
            X.append(f3.row_vector(feats))
            y.append(lab["label"])
            dates.append(str(a["t"][i]))
            ends.append(lab["exit_date"])
            tick.append(t)
            gross.append(lab["return_pct"])
            net.append(lab["net_return_pct"])
            reg.append(rc)
            why.append(lab["exit_reason"])
    order = sorted(range(len(y)), key=lambda k: (dates[k], tick[k]))
    pick = lambda seq, dt=None: np.array([seq[k] for k in order], dtype=dt)  # noqa: E731
    return {
        "X": pick(X, np.float32).reshape(len(order), len(f3.V3_FEATURE_COLS)),
        "y": pick(y, np.int8),
        "dates": pick(dates, object), "label_end_dates": pick(ends, object),
        "tickers": pick(tick, object), "regime_class": pick(reg, object),
        "return_pct": pick(gross, np.float64), "net_return_pct": pick(net, np.float64),
        "exit_reason": pick(why, object),
        "feature_names": list(f3.V3_FEATURE_COLS), "label_method": LABEL_METHOD,
    }


# ══════════════════════════════════════════════════════════════════════════
#  TRAIN / EVALUATE (honest out-of-sample), ARTIFACT, SHADOW SCORE, JOB
# ══════════════════════════════════════════════════════════════════════════
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

import ml_cv  # noqa: E402

try:
    from system_config import BASE_CONFIG, DATA_DIR
except ImportError:  # pragma: no cover - system_config always ships
    BASE_CONFIG, DATA_DIR = {}, os.environ.get("VOLTRADE_DATA_DIR", "/data/voltrade")

V3_DIR = os.path.join(DATA_DIR, "ml_v3")
# The CANDIDATE artifact. Deliberately NOT ml_model_v2.MODEL_PATH: nothing
# that blends into the trading score ever loads this file.
CANDIDATE_MODEL_PATH = os.path.join(V3_DIR, "voltrade_ml_v3_candidate.pkl")
EVAL_PATH = os.path.join(V3_DIR, "ml_v3_eval.json")
STATE_PATH = os.path.join(V3_DIR, "job_state.json")
BARS_PATH = os.path.join(V3_DIR, "bars.json")
DATASET_PATH = os.path.join(V3_DIR, "dataset.pkl")
CV_PATH = os.path.join(V3_DIR, "cv.pkl")

MODEL_ROLE = "shadow_candidate"
MODEL_NAME = "lgbm_v3_candidate"
# Model configurations evaluated on the v3 dataset before the report was
# written (ml_cv.LGBM_V3_PARAMS, chosen a priori, never tuned on results).
# A session that tries more MUST raise this so the report's discount is honest.
VARIANTS_TRIED = 1

# Fixed training universe chosen independently of outcomes (liquid US names
# across sectors + the bot's core list). Survivorship-biased: every name is
# listed TODAY (stated in the report). Deliberately not "today's
# most-actives", which selects on the end of the sample.
UNIVERSE = (
    "AAPL MSFT NVDA AMD META GOOGL AMZN TSLA JPM BAC V MA COIN MSTR PLTR CRWD HOOD SOFI AFRM UPST "
    "NFLX INTC MU QCOM AVGO ORCL CRM ADBE PYPL SHOP UBER SNAP ROKU DKNG RIVN LCID NIO F GM XOM CVX "
    "OXY SLB HAL DVN FCX AA CLF NEM KO PEP WMT TGT COST HD LOW NKE SBUX MCD DIS T VZ PFE MRK JNJ "
    "ABBV LLY UNH CVS GILD BA CAT DE GE LMT RTX C WFC GS MS SCHW AAL DAL UAL CCL RCL MARA RIOT RBLX "
    "ZM DOCU TWLO NET SNOW DDOG MRNA BABA"
).split()


def _cfg(key: str, default):
    return BASE_CONFIG.get(key, default) if isinstance(BASE_CONFIG, dict) else default


def evaluate(ds: dict, n_splits: Optional[int] = None) -> dict:
    """Honest walk-forward OOS evaluation of the candidate on `ds`."""
    n_splits = int(n_splits or _cfg("ML_V3_CV_SPLITS", 5))
    res = ml_cv.honest_cv(ds["X"], ds["y"], ds["dates"], label_end_dates=ds["label_end_dates"],
                          embargo_dates=EMBARGO_DATES, n_splits=n_splits,
                          feature_names=list(ds["feature_names"]))
    return {"cv": res, "report": summarize(ds, res["oos_prob"], res["folds"], res["method"])}


def summarize(ds: dict, oos: np.ndarray, folds: List[dict], method: str) -> dict:
    """Every number research/ml_v3_eval.md reports, from OOS probabilities."""
    tested = ~np.isnan(oos)
    y, p = ds["y"][tested], oos[tested]
    net = ds["net_return_pct"][tested]
    dates = ds["dates"][tested]
    reg = ds["regime_class"][tested]
    alpha = 0.05 / max(1, VARIANTS_TRIED)  # Bonferroni across variants tried
    per_regime = {}
    for rc in ("bullish", "neutral", "bearish"):
        m = reg == rc
        if m.sum() >= 50:
            per_regime[rc] = {**ml_cv.binary_metrics(y[m], p[m]),
                              "mean_net_return_pct": float(net[m].mean()),
                              "quintiles": ml_cv.quantile_returns(p[m], net[m])}
    n_all = len(ds["y"])
    return {
        "model": MODEL_NAME, "role": MODEL_ROLE, "label_method": LABEL_METHOD,
        "feature_set": f3.FEATURE_SET_VERSION, "features": list(ds["feature_names"]),
        "cv_method": method, "embargo_dates": EMBARGO_DATES,
        "rows": int(n_all), "rows_tested": int(tested.sum()),
        "tickers": int(len(set(ds["tickers"].tolist()))),
        "date_range": [str(min(ds["dates"])), str(max(ds["dates"]))] if n_all else [None, None],
        "label_base_rate_all": float(ds["y"].mean()) if n_all else None,
        # random entry, same universe, same exit rule, same cost: the alpha bar
        "base_rate_mean_net_return_pct": float(net.mean()) if len(net) else None,
        "pooled": ml_cv.binary_metrics(y, p),
        "auc_ci95_date_block": ml_cv.block_bootstrap_ci(y, p, dates, ml_cv.auc),
        "variants_tried": VARIANTS_TRIED,
        "auc_ci_bonferroni": ml_cv.block_bootstrap_ci(y, p, dates, ml_cv.auc, alpha=alpha),
        "calibration_raw": ml_cv.calibration_table(y, p),
        "quintiles_net_return": ml_cv.quantile_returns(p, net),
        "top_minus_bottom_ci95_date_block": ml_cv.block_bootstrap_ci(
            net, p, dates, lambda r, pp: ml_cv.quantile_returns(pp, r)["top_minus_bottom"]),
        "per_regime": per_regime, "folds": folds,
        "survivorship_note": "universe = names listed today (no delisted names)",
    }


def train_final(ds: dict, oos: np.ndarray) -> dict:
    """Fit the candidate on ALL rows (inner-validation early stopping) plus an
    isotonic calibrator fitted on the honest OOS probabilities."""
    idx = np.arange(len(ds["y"]))
    model, info = ml_cv.fit_lgbm_inner_es(ds["X"], ds["y"], idx, ds["dates"],
                                          label_end_dates=ds["label_end_dates"],
                                          embargo_dates=EMBARGO_DATES,
                                          feature_names=list(ds["feature_names"]))
    calibrator = None
    tested = ~np.isnan(oos)
    if tested.sum() >= 200:
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(oos[tested], ds["y"][tested])
    gain = model.feature_importance(importance_type="gain")
    return {"model": model, "calibrator": calibrator, "fit": info,
            "feature_names": list(ds["feature_names"]), "label_method": LABEL_METHOD,
            "feature_set": f3.FEATURE_SET_VERSION, "role": MODEL_ROLE, "promoted": False,
            "trained_at": datetime.now(timezone.utc).isoformat(), "rows": int(len(ds["y"])),
            "feature_importance_gain": {k: round(float(v), 2)
                                        for k, v in zip(ds["feature_names"], gain)}}


def save_artifacts(bundle: dict, report: dict) -> None:
    import joblib
    os.makedirs(V3_DIR, exist_ok=True)
    joblib.dump(bundle, CANDIDATE_MODEL_PATH + ".tmp")
    os.replace(CANDIDATE_MODEL_PATH + ".tmp", CANDIDATE_MODEL_PATH)
    _save_json(EVAL_PATH, {**report, "feature_importance_gain": bundle.get("feature_importance_gain"),
                           "final_fit": bundle.get("fit"), "trained_at": bundle.get("trained_at")})


def run_pipeline(all_bars: Dict[str, Sequence[dict]], save: bool = True) -> dict:
    """One shot (no time budget): dataset -> honest CV -> final fit. Used by
    scripts/ml_v3_eval.py; production runs the staged run_job()."""
    ds = build_dataset(all_bars)
    ev = evaluate(ds)
    bundle = train_final(ds, ev["cv"]["oos_prob"])
    if save:
        save_artifacts(bundle, ev["report"])
    return {"report": ev["report"], "bundle": bundle, "dataset": ds, "oos": ev["cv"]["oos_prob"]}


# ── Bars (Alpaca, same endpoint/adjustment as deep_score's bar fetch) ───────

def _alpaca_daily_bars(symbols: Sequence[str], start: str, timeout: int = 20) -> Dict[str, list]:
    import requests
    from alpaca_feed import bars_feed
    try:
        from alpaca_rate_limiter import install_global_throttle
        install_global_throttle()
    except ImportError as e:
        logger.warning(f"rate limiter unavailable: {e}")
    headers = {"APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
               "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", "")}
    out: Dict[str, list] = {}
    token = None
    for _ in range(20):
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start,
                  "limit": 10000, "adjustment": "all", "feed": bars_feed()}
        if token:
            params["page_token"] = token
        r = requests.get("https://data.alpaca.markets/v2/stocks/bars", params=params,
                         headers=headers, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
        js = r.json()
        for sym, bars in (js.get("bars") or {}).items():
            out.setdefault(sym, []).extend(bars)
        token = js.get("next_page_token")
        if not token:
            break
    return out


def _today_et() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception as e:  # zoneinfo data missing: fixed EST offset
        logger.debug(f"zoneinfo unavailable ({e}); using UTC-5")
        return (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%d")


# ── Shadow score: deep_score LOGS this next to v2's score, never blends it ──

_REF_TTL_S, _REF_FAIL_TTL_S = 1800.0, 300.0
_cache_lock = threading.Lock()
_cache: dict = {"bundle": None, "mtime": 0.0, "refs": None, "refs_at": 0.0, "refs_err_at": 0.0}


def _load_bundle():
    mtime = os.path.getmtime(CANDIDATE_MODEL_PATH)
    with _cache_lock:
        if _cache["bundle"] is None or _cache["mtime"] != mtime:
            import joblib
            _cache["bundle"] = joblib.load(CANDIDATE_MODEL_PATH)
            _cache["mtime"] = mtime
        return _cache["bundle"]


def _ref_bars() -> Optional[dict]:
    """SPY/VXX/TLT/HYG daily bars, one multi-symbol call per 30 min."""
    now = time.time()
    with _cache_lock:
        if _cache["refs"] is not None and now - _cache["refs_at"] < _REF_TTL_S:
            return _cache["refs"]
        if now - _cache["refs_err_at"] < _REF_FAIL_TTL_S:
            return None
    start = (datetime.now(timezone.utc) - timedelta(days=420)).strftime("%Y-%m-%d")
    try:
        refs = _alpaca_daily_bars(list(f3.REF_SYMBOLS), start, timeout=8)
    except Exception as e:
        logger.warning(f"v3 reference bars unavailable: {e}")
        with _cache_lock:
            _cache["refs_err_at"] = now
        return None
    with _cache_lock:
        _cache["refs"], _cache["refs_at"] = refs, now
    return refs


def shadow_score(ticker: str, stock_bars: Optional[Sequence[dict]],
                 ref_bars: Optional[dict] = None, today_et: Optional[str] = None) -> Optional[dict]:
    """Candidate win probability for one deep_score candidate, for the shadow
    log ONLY. None — with no network call — when no candidate artifact exists
    or there are too few completed bars."""
    if not stock_bars or not os.path.exists(CANDIDATE_MODEL_PATH):
        return None
    bundle = _load_bundle()
    refs = ref_bars if ref_bars is not None else _ref_bars()
    if not refs:
        return None
    feats = f3.serve_features(stock_bars, refs, today_et or _today_et())
    if feats is None:
        return None
    row = np.array([[float(feats.get(k, math.nan)) for k in bundle["feature_names"]]])
    prob = float(bundle["model"].predict(row)[0])
    cal = bundle.get("calibrator")
    if cal is not None:
        prob = float(cal.predict([prob])[0])
    return {"v3_win_probability": round(prob, 4), "v3_model": MODEL_NAME,
            "v3_asof": feats.get("_asof_date")}


# ── Production job: staged, checkpointed, time-budgeted ─────────────────────
# bot.ts runs this through execPythonSerialized (hard 90s SIGKILL), so each
# call does what fits in time_budget_s, persists it, and the next call
# resumes: fetch -> build (per ticker) -> cv (per fold) -> final.

def _load_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, default=str)
    os.replace(path + ".tmp", path)


def _dump(path: str, obj) -> None:
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def _load(path: str):
    import joblib
    return joblib.load(path)


def _eval_age_days() -> Optional[float]:
    if not os.path.exists(EVAL_PATH):
        return None
    return (time.time() - os.path.getmtime(EVAL_PATH)) / 86400.0


def _merge(parts: List[dict]) -> dict:
    parts = [p for p in parts if len(p["y"])]
    if not parts:
        raise RuntimeError("empty v3 dataset")
    keys = ("X", "y", "dates", "label_end_dates", "tickers", "regime_class",
            "return_pct", "net_return_pct", "exit_reason")
    ds = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    order = np.array(sorted(range(len(ds["y"])), key=lambda i: (ds["dates"][i], ds["tickers"][i])))
    ds = {k: v[order] for k, v in ds.items()}
    ds["feature_names"] = parts[0]["feature_names"]
    ds["label_method"] = LABEL_METHOD
    return ds


def run_job(time_budget_s: float = 60.0, force: bool = False,
            fetch=None, symbols: Optional[Sequence[str]] = None) -> dict:
    """Advance the weekly candidate retrain by as much as fits in the budget.
    Returns {"stage", "done", ...}. Never touches the live v2 artifact.
    fetch/symbols are injection points for tests."""
    t0 = time.monotonic()
    left = lambda: time_budget_s - (time.monotonic() - t0)  # noqa: E731
    fetch = fetch or _alpaca_daily_bars
    universe = list(symbols or UNIVERSE)
    st = _load_json(STATE_PATH, {})
    age = _eval_age_days()
    if not st.get("stage") or st.get("stage") == "done":
        if not force and age is not None and age < float(_cfg("ML_V3_RETRAIN_DAYS", 7)):
            return {"stage": "idle", "done": True, "eval_age_days": round(age, 2)}
        st = {"stage": "fetch", "run_started": datetime.now(timezone.utc).isoformat(),
              "fetched": [], "built": []}
        for p in (BARS_PATH, DATASET_PATH, CV_PATH):
            if os.path.exists(p):
                os.remove(p)
    steps: List[str] = []
    try:
        if st["stage"] == "fetch":
            bars = _load_json(BARS_PATH, {})
            start = (datetime.now(timezone.utc)
                     - timedelta(days=int(_cfg("ML_V3_LOOKBACK_DAYS", 1400)))).strftime("%Y-%m-%d")
            todo = [s for s in list(f3.REF_SYMBOLS) + universe if s not in st["fetched"]]
            while todo and left() > 25:
                chunk, todo = todo[:10], todo[10:]
                bars.update(fetch(chunk, start))
                st["fetched"] += chunk
            steps.append(f"fetched {len(st['fetched'])}")
            _save_json(BARS_PATH, bars)
            if not todo:
                st["stage"] = "build"
            _save_json(STATE_PATH, st)
        if st["stage"] == "build" and left() > 15:
            today = _today_et()
            bars = {k: f3.strip_incomplete(v, today) for k, v in _load_json(BARS_PATH, {}).items()}
            refs = {s: bars.get(s, []) for s in f3.REF_SYMBOLS}
            parts = _load(DATASET_PATH) if os.path.exists(DATASET_PATH) else []
            todo = [t for t in universe if t in bars and t not in st["built"]]
            while todo and left() > 10:
                t = todo.pop(0)
                parts.append(build_dataset({t: bars[t], **refs}, [t]))
                st["built"].append(t)
            steps.append(f"built {len(st['built'])}")
            _dump(DATASET_PATH, parts)
            if not todo:
                st["stage"] = "cv"
                _dump(CV_PATH, {"oos": None, "folds": [], "next_fold": 0})
            _save_json(STATE_PATH, st)
        if st["stage"] in ("cv", "final") and left() > 15:
            ds = _merge(_load(DATASET_PATH))
            cvs = _load(CV_PATH)
            splits = ml_cv.purged_walk_forward_splits(
                ds["dates"], n_splits=int(_cfg("ML_V3_CV_SPLITS", 5)),
                embargo_dates=EMBARGO_DATES, label_end_dates=ds["label_end_dates"])
            if cvs["oos"] is None:
                cvs["oos"] = np.full(len(ds["y"]), np.nan)
            while st["stage"] == "cv" and left() > 15:
                k = cvs["next_fold"]
                if k >= len(splits):
                    st["stage"] = "final"
                    break
                tr, te = splits[k]
                model, info = ml_cv.fit_lgbm_inner_es(
                    ds["X"], ds["y"], tr, ds["dates"], label_end_dates=ds["label_end_dates"],
                    embargo_dates=EMBARGO_DATES, feature_names=list(ds["feature_names"]))
                prob = model.predict(ds["X"][te])
                cvs["oos"][te] = prob
                m = ml_cv.binary_metrics(ds["y"][te], prob)
                m.update({"fold": k, "train_rows": int(len(tr)), "test_rows": int(len(te)),
                          "test_first_date": str(min(ds["dates"][te])), **info})
                cvs["folds"].append(m)
                cvs["next_fold"] = k + 1
                steps.append(f"fold {k}")
                _dump(CV_PATH, cvs)
            _save_json(STATE_PATH, st)
            if st["stage"] == "final" and left() > 15:
                report = summarize(ds, cvs["oos"], cvs["folds"],
                                   f"purged_walk_forward_by_date(embargo_dates={EMBARGO_DATES}, "
                                   f"early_stopping=inner_validation)")
                save_artifacts(train_final(ds, cvs["oos"]), report)
                st["stage"] = "done"
                st["finished"] = datetime.now(timezone.utc).isoformat()
                _save_json(STATE_PATH, st)
                steps.append("final")
    except Exception as e:
        logger.warning(f"ml_v3 job failed at stage {st.get('stage')}: {e}")
        return {"stage": st.get("stage"), "done": False, "steps": steps,
                "error": f"{type(e).__name__}: {e}"[:300]}
    return {"stage": st["stage"], "done": st["stage"] == "done", "steps": steps,
            "seconds": round(time.monotonic() - t0, 1)}


def status() -> dict:
    """Compact read of the last evaluation + job state (daemon / diag)."""
    rep = _load_json(EVAL_PATH, None)
    st = _load_json(STATE_PATH, {})
    out = {"candidate_exists": os.path.exists(CANDIDATE_MODEL_PATH), "role": MODEL_ROLE,
           "promoted": False, "job_stage": st.get("stage"), "eval_age_days": _eval_age_days()}
    if rep:
        out.update({k: rep.get(k) for k in ("trained_at", "rows", "date_range", "pooled",
                                            "auc_ci95_date_block", "quintiles_net_return",
                                            "base_rate_mean_net_return_pct", "variants_tried",
                                            "per_regime")})
    return out


if __name__ == "__main__":  # python3 ml_model_v3.py job|status
    import sys
    _cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    print(json.dumps(run_job() if _cmd == "job" else status(), default=str))
