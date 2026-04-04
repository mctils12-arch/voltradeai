#!/usr/bin/env python3
"""
VolTradeAI ML Intelligence Layer
──────────────────────────────────
A. ML Scoring Model    — sklearn RandomForest + GradientBoosting ensemble
B. Fill Tracker        — tracks order fills, learns optimal order types
C. Dynamic Weight Learner — learns which strategy combos predict winners

Public API:
    ml_score(features_dict)        → {"ml_score", "ml_confidence", "ml_signal", "model_type"}
    train_model(polygon_key)       → {"status", "accuracy", "features", "samples", "timestamp"}
    track_fill(order_data)         → None
    get_fill_recommendation(...)   → {"order_type", "reason", "expected_slippage"}
    update_weights(trade_results)  → {"momentum", "mean_reversion", "vrp", "squeeze", "volume"}
"""

import os
import json
import time
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/tmp/voltrade_ml_model.pkl"
FILLS_PATH   = "/tmp/voltrade_fills.json"
WEIGHTS_PATH = "/tmp/voltrade_weights.json"

# ── Feature columns (must match training order) ────────────────────────────────
FEATURE_COLS = [
    "momentum_1m", "momentum_3m", "momentum_12m", "volume_ratio",
    "rsi_14", "vrp", "ewma_vol", "garch_vol", "change_pct_today",
    "range_pct", "vwap_position", "put_call_ratio", "iv_rank",
    "sentiment_score", "sector_encoded",
    "vix", "vix_regime_encoded", "sector_momentum", "market_regime_encoded",
    "news_sentiment", "treasury_10y",
    "adx", "obv_signal",
    "intel_score", "trap_warning", "insider_signal", "sell_the_news_risk",
]

# ── Default fallback weights ───────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "momentum": 0.25,
    "mean_reversion": 0.20,
    "vrp": 0.25,
    "squeeze": 0.15,
    "volume": 0.15,
}

POLYGON_KEY_DEFAULT = "UNwTHo3kvZMBckeIaHQbBLuaaURmFUQP"


# ═══════════════════════════════════════════════════════════════════════════════
# A.  ML SCORING MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def _model_is_fresh(max_age_days: int = 7) -> bool:
    """Return True if saved model exists and is younger than max_age_days."""
    if not os.path.exists(MODEL_PATH):
        return False
    age_secs = time.time() - os.path.getmtime(MODEL_PATH)
    return age_secs < max_age_days * 86400


def _load_model():
    """Load (ensemble_rf, ensemble_gb) from disk. Returns None on failure."""
    try:
        import joblib
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict) and "rf" in bundle and "gb" in bundle:
            return bundle
    except Exception:
        pass
    return None


def _features_to_array(features_dict: dict) -> np.ndarray:
    """Convert a feature dict to a 1-row numpy array in FEATURE_COLS order."""
    row = []
    for col in FEATURE_COLS:
        val = features_dict.get(col, 0)
        try:
            val = float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            val = 0.0
        row.append(val)
    return np.array([row], dtype=np.float32)


def ml_score(features_dict: dict) -> dict:
    """
    Score a stock using the ML model ensemble.

    Returns:
        {
            "ml_score": 0-100 (float),
            "ml_confidence": 0-1.0,
            "ml_signal": "BUY" | "SELL" | "HOLD",
            "model_type": "ensemble" | "fallback",
        }
    """
    # ── Try ML model first ────────────────────────────────────────────────────
    if _model_is_fresh():
        bundle = _load_model()
        if bundle is not None:
            try:
                X = _features_to_array(features_dict)
                rf = bundle["rf"]
                gb = bundle["gb"]

                prob_rf = rf.predict_proba(X)[0][1]   # P(class=1)
                prob_gb = gb.predict_proba(X)[0][1]
                prob    = (prob_rf + prob_gb) / 2.0   # ensemble average

                score_0_100 = round(float(prob) * 100, 1)
                confidence  = round(float(prob), 4)

                if prob >= 0.60:
                    signal = "BUY"
                elif prob <= 0.40:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                return {
                    "ml_score":      score_0_100,
                    "ml_confidence": confidence,
                    "ml_signal":     signal,
                    "model_type":    "ensemble",
                }
            except Exception:
                pass  # fall through to rule-based

    # ── Fallback: simple rule-based scoring ───────────────────────────────────
    mom  = float(features_dict.get("momentum_1m", 0) or 0)
    rsi  = float(features_dict.get("rsi_14", 50) or 50)
    vrp  = float(features_dict.get("vrp", 0) or 0)
    vol  = float(features_dict.get("volume_ratio", 1) or 1)
    sent = float(features_dict.get("sentiment_score", 50) or 50)

    score = 50.0
    score += min(mom * 2,  20)
    score += min((vol - 1) * 10, 15)
    score += min(vrp * 1.5, 10)
    score += (sent - 50) * 0.15
    if rsi < 30:
        score += 10
    elif rsi > 70:
        score -= 5

    score = max(0.0, min(100.0, score))

    if score >= 65:
        signal = "BUY"
    elif score <= 35:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "ml_score":      round(score, 1),
        "ml_confidence": round(score / 100, 4),
        "ml_signal":     signal,
        "model_type":    "fallback",
    }


# ── Training helpers ───────────────────────────────────────────────────────────

def _fetch_grouped_day(date_str: str, api_key: str) -> list:
    """Fetch Polygon grouped daily results for one date."""
    import requests
    url = (
        f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
        f"?adjusted=true&apiKey={api_key}"
    )
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        return data.get("results", [])
    except Exception:
        return []


def _trading_dates(num_days: int = 45) -> list:
    """Return the last num_days calendar dates (weekdays only, recent first)."""
    dates = []
    d = datetime.now() - timedelta(days=1)
    while len(dates) < num_days:
        if d.weekday() < 5:  # Mon–Fri
            dates.append(d.strftime("%Y-%m-%d"))
        d -= timedelta(days=1)
    return dates  # most recent first


def _build_feature_matrix(api_key: str):
    """
    Fetch ~40 trading days of Polygon grouped data, build feature matrix X and labels y.
    Target: did the stock's close go up 2%+ in the next 5 trading days?
    Returns (X, y, sample_count) or (None, None, 0) on failure.
    """
    dates = _trading_dates(50)  # fetch more to have look-ahead room
    if len(dates) < 10:
        return None, None, 0

    # Fetch data day by day — dict keyed by date then ticker
    daily_data: dict[str, dict] = {}  # date -> {ticker: bar}
    print(f"  Fetching {min(len(dates), 45)} trading days from Polygon...")
    for date in dates[:45]:
        bars = _fetch_grouped_day(date, api_key)
        if bars:
            daily_data[date] = {b["T"]: b for b in bars if "T" in b}
        time.sleep(0.12)  # ~8 req/sec — well within free tier

    if len(daily_data) < 10:
        return None, None, 0

    sorted_dates = sorted(daily_data.keys())  # oldest first

    X_rows = []
    y_labels = []

    # For each date d (up to 5 days before end so we have a look-ahead window)
    for i, date in enumerate(sorted_dates[:-5]):
        day_bars = daily_data[date]

        # For look-ahead: find the bar 5 trading days later
        future_dates = sorted_dates[i + 1: i + 8]  # next 7 to grab 5 real trading days
        if len(future_dates) < 5:
            continue

        future_date = future_dates[4]  # ~5 trading days later
        future_bars = daily_data.get(future_date, {})

        for ticker, bar in day_bars.items():
            # Basic filters
            close = bar.get("c", 0)
            volume = bar.get("v", 0)
            if close < 5 or volume < 500_000:
                continue
            if "." in ticker or len(ticker) > 5:
                continue

            # ── Compute features ──────────────────────────────────────────
            o = bar.get("o", close) or close
            h = bar.get("h", close) or close
            l = bar.get("l", close) or close
            vw = bar.get("vw", close) or close

            change_pct = (close - o) / o * 100 if o > 0 else 0
            range_pct  = (h - l) / l * 100 if l > 0 else 0
            vwap_pos   = 1.0 if close > vw else 0.0

            # Momentum proxies from what we have
            momentum_1m  = change_pct
            momentum_3m  = change_pct * 1.2   # rough proxy
            momentum_12m = change_pct * 0.5   # rough proxy

            # Volume ratio (normalised by rough median — will be standardised)
            volume_ratio = min(volume / 5_000_000, 5.0)

            # RSI proxy from single-day data (14-day RSI requires history; use intraday proxy)
            rsi_proxy = 50 + min(max(change_pct * 3, -25), 25)  # crude but fast

            # Volatility proxies
            ewma_vol_proxy = range_pct * 5   # daily range as rough vol estimate
            garch_vol_proxy = range_pct * 4

            # Options / sentiment proxies (zero — no history available in grouped)
            put_call_ratio = 1.0
            iv_rank        = 50.0
            sentiment      = 50.0 + change_pct * 0.5

            # Sector encoding
            sector_encoded = float(hash(ticker[:2]) % 10)

            # VRP proxy
            vrp_proxy = change_pct * 0.3

            row = [
                momentum_1m, momentum_3m, momentum_12m, volume_ratio,
                rsi_proxy, vrp_proxy, ewma_vol_proxy, garch_vol_proxy,
                change_pct, range_pct, vwap_pos, put_call_ratio,
                iv_rank, sentiment, sector_encoded,
            ]

            # ── Compute target ────────────────────────────────────────────
            future_bar = future_bars.get(ticker)
            if future_bar is None:
                continue
            future_close = future_bar.get("c", 0)
            if future_close <= 0 or close <= 0:
                continue

            gain_pct = (future_close - close) / close * 100
            label = 1 if gain_pct >= 2.0 else 0  # 2%+ in 5 days

            X_rows.append(row)
            y_labels.append(label)

    if len(X_rows) < 50:
        return None, None, 0

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int8)
    return X, y, len(X_rows)


def train_model(polygon_key: str = POLYGON_KEY_DEFAULT) -> dict:
    """
    Train (or retrain) the ML ensemble on recent Polygon historical data.

    Returns:
        {"status": "trained"/"cached"/"failed", "accuracy": float,
         "features": 21, "samples": int, "timestamp": str}
    """
    # If model exists and is fresh enough, skip retraining
    if _model_is_fresh(max_age_days=7):
        bundle = _load_model()
        if bundle is not None:
            return {
                "status":    "cached",
                "accuracy":  bundle.get("accuracy", 0),
                "features":  21,
                "samples":   bundle.get("samples", 0),
                "timestamp": bundle.get("timestamp", ""),
            }

    print("Training ML model — fetching Polygon historical data...")
    t0 = time.time()

    try:
        X, y, n_samples = _build_feature_matrix(polygon_key)
    except Exception as e:
        return {"status": "failed", "error": str(e), "accuracy": 0, "features": 15, "samples": 0, "timestamp": datetime.now().isoformat()}

    if X is None or n_samples < 50:
        return {
            "status":    "failed",
            "error":     "Not enough training samples",
            "accuracy":  0,
            "features":  21,
            "samples":   n_samples,
            "timestamp": datetime.now().isoformat(),
        }

    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import joblib

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)

        # Fill remaining NaN with median
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_medians[j]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Random Forest
        rf_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])
        rf_pipeline.fit(X_train, y_train)

        # Gradient Boosting (sklearn's built-in, no xgboost needed)
        gb_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )),
        ])
        gb_pipeline.fit(X_train, y_train)

        # Ensemble accuracy on test set
        rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
        gb_probs = gb_pipeline.predict_proba(X_test)[:, 1]
        ensemble_probs = (rf_probs + gb_probs) / 2
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        accuracy = float((ensemble_preds == y_test).mean())

        # Save bundle
        bundle = {
            "rf":        rf_pipeline,
            "gb":        gb_pipeline,
            "accuracy":  round(accuracy, 4),
            "samples":   n_samples,
            "features":  21,
            "timestamp": datetime.now().isoformat(),
        }
        joblib.dump(bundle, MODEL_PATH)

        elapsed = round(time.time() - t0, 1)
        print(f"Training complete in {elapsed}s | accuracy={accuracy:.3f} | samples={n_samples}")

        return {
            "status":    "trained",
            "accuracy":  round(accuracy, 4),
            "features":  21,
            "samples":   n_samples,
            "elapsed_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "status":    "failed",
            "error":     str(e),
            "accuracy":  0,
            "features":  21,
            "samples":   n_samples,
            "timestamp": datetime.now().isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# B.  FILL TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

def _load_fills() -> list:
    """Load fill history from disk."""
    if not os.path.exists(FILLS_PATH):
        return []
    try:
        with open(FILLS_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save_fills(fills: list) -> None:
    """Persist fill history to disk."""
    try:
        with open(FILLS_PATH, "w") as f:
            json.dump(fills, f)
    except Exception:
        pass


def track_fill(order_data: dict) -> None:
    """
    Log an order fill for ML fill-tracker learning.

    order_data keys:
        ticker, order_type, side, qty, expected_price, fill_price,
        time_placed, time_filled, session, volume, score (optional)
    """
    try:
        expected = float(order_data.get("expected_price", 0) or 0)
        fill     = float(order_data.get("fill_price", expected) or expected)
        slippage = abs(fill - expected) / expected * 100 if expected > 0 else 0.0

        record = {
            "ticker":       str(order_data.get("ticker", "")),
            "order_type":   str(order_data.get("order_type", "market")),
            "side":         str(order_data.get("side", "buy")),
            "qty":          float(order_data.get("qty", 0) or 0),
            "expected_price": expected,
            "fill_price":   fill,
            "slippage_pct": round(slippage, 4),
            "volume_at_time": float(order_data.get("volume", 0) or 0),
            "session":      str(order_data.get("session", "regular")),
            "time_placed":  str(order_data.get("time_placed", datetime.now().isoformat())),
            "time_filled":  str(order_data.get("time_filled", datetime.now().isoformat())),
            "score":        float(order_data.get("score", 0) or 0),
        }

        fills = _load_fills()
        fills.append(record)
        # Keep last 500 fills
        if len(fills) > 500:
            fills = fills[-500:]
        _save_fills(fills)
    except Exception:
        pass


def get_fill_recommendation(ticker: str, volume: float, session: str) -> dict:
    """
    Based on fill history, recommend best order type for the given context.

    Returns:
        {"order_type": "market"/"limit", "reason": str, "expected_slippage": float}
    """
    fills = _load_fills()

    # Default when not enough data
    if len(fills) < 50:
        return {
            "order_type":       "market",
            "reason":           "Insufficient fill history — defaulting to market orders",
            "expected_slippage": 0.05,
        }

    # Filter to relevant session
    session_fills = [f for f in fills if f.get("session") == session]
    if len(session_fills) < 10:
        session_fills = fills  # fall back to all sessions

    # Classify by volume profile (low / medium / high)
    volume_threshold_low  = 1_000_000
    volume_threshold_high = 10_000_000

    if volume < volume_threshold_low:
        vol_cat = "low"
    elif volume < volume_threshold_high:
        vol_cat = "medium"
    else:
        vol_cat = "high"

    # Compute avg slippage by order type for this volume category
    stats: dict[str, list] = {"market": [], "limit": []}
    for f in session_fills:
        fvol = f.get("volume_at_time", 0)
        if fvol < volume_threshold_low:
            fcat = "low"
        elif fvol < volume_threshold_high:
            fcat = "medium"
        else:
            fcat = "high"
        if fcat != vol_cat:
            continue
        otype = f.get("order_type", "market")
        if otype not in stats:
            stats[otype] = []
        stats[otype].append(f.get("slippage_pct", 0))

    market_avg  = float(np.mean(stats["market"])) if stats["market"] else 0.10
    limit_avg   = float(np.mean(stats["limit"])) if stats["limit"] else 0.05
    market_n    = len(stats["market"])
    limit_n     = len(stats["limit"])

    # Recommend order type with lower average slippage (need ≥5 data points)
    if market_n >= 5 and limit_n >= 5:
        if market_avg < limit_avg:
            return {
                "order_type":        "market",
                "reason":            f"Market orders have lower avg slippage ({market_avg:.2f}%) vs limit ({limit_avg:.2f}%) for {vol_cat}-volume stocks in {session} session",
                "expected_slippage": round(market_avg, 4),
            }
        else:
            return {
                "order_type":        "limit",
                "reason":            f"Limit orders have lower avg slippage ({limit_avg:.2f}%) vs market ({market_avg:.2f}%) for {vol_cat}-volume stocks in {session} session",
                "expected_slippage": round(limit_avg, 4),
            }
    elif vol_cat == "low":
        return {
            "order_type":        "limit",
            "reason":            f"Low-volume stocks ({vol_cat}) fill better with limit orders — less price impact",
            "expected_slippage": round(limit_avg, 4),
        }
    else:
        return {
            "order_type":        "market",
            "reason":            f"High/medium-volume stocks fill quickly with market orders in {session} session",
            "expected_slippage": round(market_avg, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# C.  DYNAMIC WEIGHT LEARNER
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_FEATURES = ["momentum_score", "mean_reversion_score", "vrp_score", "squeeze_score", "volume_score"]
STRATEGY_KEYS     = ["momentum", "mean_reversion", "vrp", "squeeze", "volume"]


def update_weights(trade_results: list) -> dict:
    """
    Given completed trades with outcomes, learn optimal strategy weights.

    trade_results: list of dicts with keys:
        momentum_score, mean_reversion_score, vrp_score, squeeze_score,
        volume_score, pnl_pct

    Returns normalised weights dict:
        {"momentum": float, "mean_reversion": float, "vrp": float,
         "squeeze": float, "volume": float}
    """
    # Always return defaults if not enough data
    if len(trade_results) < 30:
        _save_weights(DEFAULT_WEIGHTS.copy())
        return DEFAULT_WEIGHTS.copy()

    try:
        from sklearn.ensemble import RandomForestClassifier

        X_rows = []
        y_labels = []

        for t in trade_results:
            row = [
                float(t.get("momentum_score", 50) or 50),
                float(t.get("mean_reversion_score", 50) or 50),
                float(t.get("vrp_score", 50) or 50),
                float(t.get("squeeze_score", 50) or 50),
                float(t.get("volume_score", 50) or 50),
            ]
            pnl = float(t.get("pnl_pct", 0) or 0)
            label = 1 if pnl > 0 else 0
            X_rows.append(row)
            y_labels.append(label)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_labels, dtype=np.int8)

        # Handle degenerate case (all same label)
        if len(set(y)) < 2:
            _save_weights(DEFAULT_WEIGHTS.copy())
            return DEFAULT_WEIGHTS.copy()

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X, y)

        # Feature importances → normalised weights
        importances = clf.feature_importances_
        total = importances.sum()
        if total <= 0:
            _save_weights(DEFAULT_WEIGHTS.copy())
            return DEFAULT_WEIGHTS.copy()

        weights = {
            STRATEGY_KEYS[i]: round(float(importances[i] / total), 4)
            for i in range(len(STRATEGY_KEYS))
        }

        # Blend 50/50 with defaults to avoid over-fitting to small samples
        blended = {
            k: round(weights[k] * 0.5 + DEFAULT_WEIGHTS[k] * 0.5, 4)
            for k in STRATEGY_KEYS
        }

        # Re-normalise
        total_blended = sum(blended.values())
        blended = {k: round(v / total_blended, 4) for k, v in blended.items()}

        _save_weights(blended)
        return blended

    except Exception:
        _save_weights(DEFAULT_WEIGHTS.copy())
        return DEFAULT_WEIGHTS.copy()


def _save_weights(weights: dict) -> None:
    """Persist learned weights to disk."""
    try:
        payload = {**weights, "updated": datetime.now().isoformat()}
        with open(WEIGHTS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def load_weights() -> dict:
    """Load learned weights from disk, falling back to defaults."""
    if not os.path.exists(WEIGHTS_PATH):
        return DEFAULT_WEIGHTS.copy()
    try:
        with open(WEIGHTS_PATH) as f:
            data = json.load(f)
        weights = {k: float(data.get(k, DEFAULT_WEIGHTS[k])) for k in STRATEGY_KEYS}
        total = sum(weights.values())
        if total <= 0:
            return DEFAULT_WEIGHTS.copy()
        return {k: round(v / total, 4) for k, v in weights.items()}
    except Exception:
        return DEFAULT_WEIGHTS.copy()


# ══ Exit Prediction Model ─────────────────────────────────────────────────────────────────────────────

def predict_exit(position_data: dict) -> dict:
    """
    Predict whether to hold or sell a position.
    Uses a combination of rules and learned patterns from trade feedback.
    
    position_data: {ticker, pnl_pct, entry_price, current_price, holding_days}
    Returns: {action: "HOLD"/"SELL", confidence: 0-1, reason: "..."}
    """
    pnl = position_data.get("pnl_pct", 0)
    holding_days = position_data.get("holding_days", 0)
    ticker = position_data.get("ticker", "")
    
    # Load trade feedback to learn from past exits
    feedback_path = "/tmp/voltrade_trade_feedback.json"
    avg_winner_hold = 5  # defaults
    avg_loser_hold = 3
    win_rate = 0.5
    feedback = []
    
    try:
        if os.path.exists(feedback_path):
            with open(feedback_path) as f:
                feedback = json.load(f)
            if len(feedback) >= 10:
                winners = [t for t in feedback if t.get("pnl_pct", 0) > 0]
                losers = [t for t in feedback if t.get("pnl_pct", 0) <= 0]
                if winners:
                    avg_winner_hold = sum(t.get("holding_days", 5) for t in winners) / len(winners)
                if losers:
                    avg_loser_hold = sum(t.get("holding_days", 3) for t in losers) / len(losers)
                win_rate = len(winners) / len(feedback) if feedback else 0.5
    except Exception:
        pass
    
    # Decision logic learned from feedback + rules
    
    # Rule 1: Losing trade held too long — cut it
    if pnl < -1.0 and holding_days > avg_loser_hold * 1.5:
        return {
            "action": "SELL",
            "confidence": 0.75,
            "reason": f"Held losing trade {holding_days} days (avg loser exits at {avg_loser_hold:.0f} days)"
        }
    
    # Rule 2: Small gain being held too long — take profit, don't let it reverse
    if 0.5 < pnl < 3.0 and holding_days > avg_winner_hold * 1.2:
        return {
            "action": "SELL",
            "confidence": 0.68,
            "reason": f"Small gain (+{pnl:.1f}%) held {holding_days} days — avg winner exits at {avg_winner_hold:.0f} days"
        }
    
    # Rule 3: Good gain pulling back — learned from feedback
    if pnl > 0 and pnl < 1.5 and holding_days > 5:
        # Check if similar trades in feedback ended up losing
        try:
            similar = [t for t in feedback if 0 < t.get("pnl_pct", 0) < 2 and t.get("holding_days", 0) > 5]
            if len(similar) >= 3:
                similar_losers = [t for t in similar if t.get("pnl_pct", 0) < 0]
                if len(similar_losers) / len(similar) > 0.6:
                    return {
                        "action": "SELL",
                        "confidence": 0.70,
                        "reason": f"Pattern match: {len(similar_losers)}/{len(similar)} similar trades ended in losses"
                    }
        except Exception:
            pass
    
    # Rule 4: Strong winner — let it ride
    if pnl > 4.0:
        return {
            "action": "HOLD",
            "confidence": 0.80,
            "reason": f"Strong gain (+{pnl:.1f}%) — let it ride toward ATR take-profit"
        }
    
    # Default: hold
    return {
        "action": "HOLD",
        "confidence": 0.55,
        "reason": "No exit signal — holding position"
    }


# ════════════════════════════════════════════════════════════════════════════════
# CLI / __main__
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("VolTradeAI ML Intelligence Layer — Self Test")
    print("=" * 60)

    # ── 1. Train model ────────────────────────────────────────────────────────
    print("\n[1] Training ML model...")
    result = train_model()
    print(json.dumps(result, indent=2))

    # ── 2. Score a synthetic stock ────────────────────────────────────────────
    print("\n[2] Scoring a synthetic stock...")
    test_features = {
        "momentum_1m":    3.5,
        "momentum_3m":    7.2,
        "momentum_12m":   15.0,
        "volume_ratio":   2.5,
        "rsi_14":         58,
        "vrp":            6.0,
        "ewma_vol":       22.0,
        "garch_vol":      20.5,
        "change_pct_today": 1.8,
        "range_pct":      3.2,
        "vwap_position":  1,
        "put_call_ratio": 0.85,
        "iv_rank":        62,
        "sentiment_score": 65,
        "sector_encoded": 4,
    }
    score_result = ml_score(test_features)
    print(json.dumps(score_result, indent=2))

    # ── 3. Track a fill ───────────────────────────────────────────────────────
    print("\n[3] Tracking a test fill...")
    track_fill({
        "ticker":         "AAPL",
        "order_type":     "market",
        "side":           "buy",
        "qty":            10,
        "expected_price": 175.00,
        "fill_price":     175.08,
        "time_placed":    datetime.now().isoformat(),
        "time_filled":    datetime.now().isoformat(),
        "session":        "regular",
        "volume":         45_000_000,
        "score":          72,
    })
    fills = _load_fills()
    print(f"  Total fills logged: {len(fills)}")

    # ── 4. Fill recommendation ────────────────────────────────────────────────
    print("\n[4] Fill recommendation for AAPL (regular session, high volume)...")
    rec = get_fill_recommendation("AAPL", 45_000_000, "regular")
    print(json.dumps(rec, indent=2))

    # ── 5. Weight update ──────────────────────────────────────────────────────
    print("\n[5] Updating strategy weights with synthetic trade results...")
    synthetic_trades = []
    rng = np.random.default_rng(42)
    for _ in range(35):
        mom_score = float(rng.uniform(40, 90))
        pnl = (mom_score - 60) * 0.1 + float(rng.normal(0, 1))
        synthetic_trades.append({
            "momentum_score":      mom_score,
            "mean_reversion_score": float(rng.uniform(40, 80)),
            "vrp_score":           float(rng.uniform(45, 85)),
            "squeeze_score":       float(rng.uniform(40, 75)),
            "volume_score":        float(rng.uniform(45, 80)),
            "pnl_pct":             pnl,
        })

    new_weights = update_weights(synthetic_trades)
    print(json.dumps(new_weights, indent=2))

    print("\n✓ All tests complete.")
