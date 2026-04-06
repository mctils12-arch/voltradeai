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

import requests
import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

try:
    from storage_config import ML_MODEL_PATH, FILLS_PATH, WEIGHTS_PATH, TRADE_FEEDBACK_PATH
except ImportError:
    ML_MODEL_PATH = "/tmp/voltrade_ml_model.pkl"
    FILLS_PATH = "/tmp/voltrade_fills.json"
    WEIGHTS_PATH = "/tmp/voltrade_weights.json"
    TRADE_FEEDBACK_PATH = "/tmp/voltrade_trade_feedback.json"

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = ML_MODEL_PATH

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
    "wiki_spike_ratio", "short_pressure_signal", "congressional_signal",
    "geopolitical_risk", "yield_curve", "credit_spread", "unemployment",
    "patent_signal", "ftd_signal",
    "reddit_mentions", "reddit_sentiment", "reddit_wsb_mentions",
    "google_trends_spike", "google_trends_interest",
    "news_multi_articles", "news_multi_sentiment", "social_combined_signal",
    "forward_pe", "trailing_pe", "price_to_book", "ev_ebitda",
    "price_to_sales", "analyst_target_upside", "analyst_buy_pct", "num_analysts",
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

def _model_is_fresh(max_age_days: int = 1) -> bool:
    """Return True if saved model exists and is younger than max_age_days (default: 1 day for daily retrain)."""
    if not os.path.exists(MODEL_PATH):
        return False
    age_secs = time.time() - os.path.getmtime(MODEL_PATH)
    return age_secs < max_age_days * 86400


def _load_model():
    """Load (ensemble_rf, ensemble_gb) from disk. Returns None on failure."""
    try:
        import joblib
        bundle = joblib.load(MODEL_PATH)
        # Accept both LightGBM bundles (model_type key) and legacy sklearn bundles (rf/gb keys)
        if isinstance(bundle, dict) and ("model_type" in bundle or ("rf" in bundle and "gb" in bundle)):
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

                model_type = bundle.get("model_type", "sklearn_ensemble")

                if model_type == "lightgbm":
                    model = bundle["model"]
                    prob = float(model.predict(X)[0])
                    predicted_class = 1 if prob > 0.5 else 0
                else:
                    # Support both new bundle["model"] tuple and old bundle["rf"]/bundle["gb"] keys
                    if "model" in bundle:
                        rf, gb = bundle["model"]
                    else:
                        rf, gb = bundle["rf"], bundle["gb"]
                    prob_rf = rf.predict_proba(X)[:, 1][0]
                    prob_gb = gb.predict_proba(X)[:, 1][0]
                    prob = (prob_rf + prob_gb) / 2
                    predicted_class = 1 if prob > 0.5 else 0

                ml_s = prob * 100  # Scale 0-1 probability to 0-100 score

                signal = "BUY" if predicted_class == 1 and prob > 0.6 else "SELL" if predicted_class == 0 and prob < 0.4 else "HOLD"

                return {
                    "ml_score":      round(ml_s, 1),
                    "ml_confidence": round(prob if predicted_class == 1 else 1 - prob, 3),
                    "ml_signal":     signal,
                    "model_type":    model_type,
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
    logger.info(f"  Fetching {min(len(dates), 45)} trading days from Polygon...")
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


# ── Alpaca Historical Data for Training ───────────────────────────────────────

ALPACA_KEY = os.environ.get("ALPACA_KEY", "PKMDHJOVQEVIB4UHZXUYVTIDBU")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et")
ALPACA_DATA_URL = "https://data.alpaca.markets"

def _alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

def _fetch_alpaca_training_data(days=60, max_tickers=500):
    """
    Fetch historical data from Alpaca for ML training.
    Returns list of feature dicts with labels.
    Better than Polygon free tier: no rate limits, deeper history.
    """
    samples = []
    
    # Get most active stocks to train on (diverse universe)
    tickers = []
    try:
        resp = requests.get(
            f"{ALPACA_DATA_URL}/v1beta1/screener/stocks/most-actives?by=volume&top=50",
            headers=_alpaca_headers(), timeout=10
        )
        data = resp.json()
        tickers = [s["symbol"] for s in data.get("most_actives", []) if s.get("symbol")]
    except Exception:
        pass
    
    # Add core tickers to ensure coverage
    core = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "NFLX", 
            "JPM", "BAC", "GS", "XOM", "CVX", "PFE", "JNJ", "UNH", "HD", "WMT", "DIS",
            "SPY", "QQQ", "IWM", "BA", "CAT", "V", "MA", "CRM", "ADBE", "INTC"]
    for t in core:
        if t not in tickers:
            tickers.append(t)
    
    tickers = tickers[:max_tickers]  # Increased: batch API makes this fast
    
    start_date = (datetime.now() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    
    # Batch fetch bars (24 tickers per request instead of 1 at a time)
    all_bars: dict = {}  # ticker -> list of bars
    for i in range(0, len(tickers), 20):
        batch = tickers[i:i+20]
        try:
            url = (f"{ALPACA_DATA_URL}/v2/stocks/bars"
                   f"?symbols={','.join(batch)}&timeframe=1Day&start={start_date}&limit=1000&adjustment=all")
            resp = requests.get(url, headers=_alpaca_headers(), timeout=15)
            data = resp.json()
            batch_bars = data.get("bars", {})
            for sym, b in batch_bars.items():
                all_bars[sym] = b
        except Exception:
            continue
        time.sleep(0.1)

    for ticker in tickers:
        try:
            bars = all_bars.get(ticker, [])
            
            if len(bars) < 20:
                continue
            
            # Compute features from bars
            closes = [b["c"] for b in bars]
            volumes = [b["v"] for b in bars]
            highs = [b["h"] for b in bars]
            lows = [b["l"] for b in bars]
            opens = [b["o"] for b in bars]
            
            for i in range(20, len(bars) - 5):
                # Features
                c = closes[i]
                if c <= 0:
                    continue
                
                # Momentum
                mom_1m = (c - closes[i-20]) / closes[i-20] * 100 if closes[i-20] > 0 else 0
                mom_3m = (c - closes[max(0, i-60)]) / closes[max(0, i-60)] * 100 if closes[max(0, i-60)] > 0 else 0
                
                # Volume ratio
                avg_vol_20 = sum(volumes[i-20:i]) / 20 if sum(volumes[i-20:i]) > 0 else 1
                vol_ratio = volumes[i] / avg_vol_20 if avg_vol_20 > 0 else 1
                
                # RSI (14-day)
                gains = []
                losses = []
                for j in range(i-13, i+1):
                    change = closes[j] - closes[j-1] if j > 0 else 0
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))
                avg_gain = sum(gains) / 14
                avg_loss = sum(losses) / 14
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi = 100 - (100 / (1 + rs))
                
                # Change today
                change_pct = (c - opens[i]) / opens[i] * 100 if opens[i] > 0 else 0
                
                # Range
                range_pct = (highs[i] - lows[i]) / c * 100 if c > 0 else 0
                
                # VWAP position (simplified)
                vwap_pos = 1 if c > (highs[i] + lows[i]) / 2 else 0
                
                # Volatility (20-day)
                returns = [(closes[j] - closes[j-1]) / closes[j-1] for j in range(i-19, i+1) if closes[j-1] > 0]
                ewma_vol = (sum(r**2 for r in returns[-5:]) / 5) ** 0.5 * 100 if returns else 2
                
                # Label: did it go up 2%+ in next 5 days?
                future_price = closes[min(i+5, len(closes)-1)]
                label = 1 if (future_price - c) / c >= 0.02 else 0
                
                sample = {
                    "momentum_1m": round(mom_1m, 2),
                    "momentum_3m": round(mom_3m, 2),
                    "momentum_12m": round(mom_1m * 2, 2),  # Proxy
                    "volume_ratio": round(vol_ratio, 2),
                    "rsi_14": round(rsi, 1),
                    "vrp": 0,  # Not available from bars alone
                    "ewma_vol": round(ewma_vol, 2),
                    "garch_vol": round(ewma_vol * 1.1, 2),  # Proxy
                    "change_pct_today": round(change_pct, 2),
                    "range_pct": round(range_pct, 2),
                    "vwap_position": vwap_pos,
                    "put_call_ratio": 1.0,  # Default
                    "iv_rank": 50,  # Default
                    "sentiment_score": 0,  # Not available
                    "sector_encoded": hash(ticker) % 10,
                    "vix": 20,  # Default
                    "vix_regime_encoded": 1,
                    "sector_momentum": 0,
                    "market_regime_encoded": 1,
                    "news_sentiment": 0,
                    "treasury_10y": 4.25,
                    "adx": 20,  # Default
                    "obv_signal": 0,
                    "intel_score": 0,
                    "trap_warning": 0,
                    "insider_signal": 0,
                    "sell_the_news_risk": 0,
                    "wiki_spike_ratio": 1.0,
                    "short_pressure_signal": 0,
                    "congressional_signal": 0,
                    "geopolitical_risk": 0,
                    "yield_curve": 0.5,
                    "credit_spread": 1.0,
                    "unemployment": 4.0,
                    "patent_signal": 0,
                    "ftd_signal": 0,
                    "_label": label,
                }
                samples.append(sample)
            
            # Small delay to be polite to API
            time.sleep(0.1)
            
        except Exception:
            continue
    
    return samples


def train_model(polygon_key: str = POLYGON_KEY_DEFAULT) -> dict:
    """
    Train (or retrain) the ML ensemble on recent Polygon historical data.

    Returns:
        {"status": "trained"/"cached"/"failed", "accuracy": float,
         "features": 21, "samples": int, "timestamp": str}
    """
    # If model exists and is fresh enough, skip retraining
    # BUT force retrain if feature count changed (upgrade detection)
    if _model_is_fresh(max_age_days=1):
        bundle = _load_model()
        if bundle is not None:
            stored_features = len(bundle.get("feature_names", []))
            if stored_features > 0 and stored_features != len(FEATURE_COLS):
                logger.info(f"Feature count changed ({stored_features} -> {len(FEATURE_COLS)}), forcing retrain")
            else:
              return {
                "status":    "cached",
                "accuracy":  bundle.get("accuracy", 0),
                "features":  len(FEATURE_COLS),
                "samples":   bundle.get("samples", 0),
                "timestamp": bundle.get("timestamp", ""),
            }

    logger.info("Training ML model — fetching training data...")
    t0 = time.time()

    # Primary: Alpaca historical data (free, no rate limit, better quality)
    try:
        alpaca_samples = _fetch_alpaca_training_data(days=60)
        if len(alpaca_samples) >= 500:
            all_samples = alpaca_samples
            print(json.dumps({"status": "alpaca_data_loaded", "samples": len(all_samples)}))
        else:
            print(json.dumps({"status": "alpaca_insufficient", "samples": len(alpaca_samples), "falling_back": "polygon"}))
            all_samples = alpaca_samples  # Use what we have, Polygon may add more below
    except Exception as e:
        print(json.dumps({"status": "alpaca_failed", "error": str(e)}))
        all_samples = []

    # Convert Alpaca samples to X/y arrays if we have enough
    X, y, n_samples = None, None, 0
    if len(all_samples) >= 500:
        try:
            X_rows = []
            y_labels = []
            for s in all_samples:
                row = [float(s.get(col, 0) or 0) for col in FEATURE_COLS]
                X_rows.append(row)
                y_labels.append(int(s.get("_label", 0)))
            X = np.array(X_rows, dtype=np.float32)
            y = np.array(y_labels, dtype=np.int8)
            n_samples = len(X_rows)
        except Exception as e:
            print(json.dumps({"status": "alpaca_convert_failed", "error": str(e)}))
            X, y, n_samples = None, None, 0

    if len(all_samples) < 500:
        # Fallback: Polygon grouped daily data
        try:
            X_poly, y_poly, n_poly = _build_feature_matrix(polygon_key)
        except Exception as e:
            return {"status": "failed", "error": str(e), "accuracy": 0, "features": 15, "samples": 0, "timestamp": datetime.now().isoformat()}

        # Convert any Alpaca samples we have to arrays and merge with Polygon
        if len(all_samples) > 0 and X_poly is not None:
            try:
                X_rows = []
                y_labels = []
                for s in all_samples:
                    row = [float(s.get(col, 0) or 0) for col in FEATURE_COLS]
                    X_rows.append(row)
                    y_labels.append(int(s.get("_label", 0)))
                X_alp = np.array(X_rows, dtype=np.float32)
                y_alp = np.array(y_labels, dtype=np.int8)
                # Pad Polygon rows (15 features) to match FEATURE_COLS length
                if X_poly.shape[1] < len(FEATURE_COLS):
                    pad = np.zeros((X_poly.shape[0], len(FEATURE_COLS) - X_poly.shape[1]), dtype=np.float32)
                    X_poly = np.hstack([X_poly, pad])
                X = np.vstack([X_alp, X_poly])
                y = np.concatenate([y_alp, y_poly])
                n_samples = len(X)
                all_samples.extend([None] * n_poly)  # Track combined count
            except Exception:
                X, y, n_samples = X_poly, y_poly, n_poly
        elif X_poly is not None:
            # Pad Polygon rows (15 features) to match FEATURE_COLS length
            if X_poly.shape[1] < len(FEATURE_COLS):
                pad = np.zeros((X_poly.shape[0], len(FEATURE_COLS) - X_poly.shape[1]), dtype=np.float32)
                X_poly = np.hstack([X_poly, pad])
            X, y, n_samples = X_poly, y_poly, n_poly
        else:
            X, y, n_samples = None, None, 0

    if X is None or n_samples < 50:
        return {
            "status":    "failed",
            "error":     "Not enough training samples",
            "accuracy":  0,
            "features":  len(FEATURE_COLS),
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

        if HAS_LIGHTGBM:
            # LightGBM — 5-10x faster, better accuracy, standard in quant finance
            print(json.dumps({"status": "training_lightgbm", "samples": len(all_samples)}))

            train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "n_jobs": -1,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }

            model = lgb.train(
                params,
                train_data,
                num_boost_round=300,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
            )

            # Evaluate
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            accuracy = float((y_pred == y_test).mean())

            # Feature importance
            importance = dict(zip(FEATURE_COLS, model.feature_importance(importance_type="gain").tolist()))
            top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

            bundle = {"model": model, "model_type": "lightgbm", "features": FEATURE_COLS, "accuracy": accuracy, "top_features": top_features,
                      "samples": n_samples, "timestamp": datetime.now().isoformat()}
        else:
            # Fallback to sklearn if LightGBM not available
            print(json.dumps({"status": "training_sklearn_fallback"}))

            rf = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))])
            gb = Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))])

            rf.fit(X_train, y_train)
            gb.fit(X_train, y_train)

            y_pred_rf = rf.predict(X_test)
            y_pred_gb = gb.predict(X_test)
            y_pred = ((rf.predict_proba(X_test)[:, 1] + gb.predict_proba(X_test)[:, 1]) / 2 > 0.5).astype(int)
            accuracy = float((y_pred == y_test).mean())

            bundle = {"model": (rf, gb), "model_type": "sklearn_ensemble", "features": FEATURE_COLS, "accuracy": accuracy, "top_features": [],
                      "samples": n_samples, "timestamp": datetime.now().isoformat()}

        joblib.dump(bundle, MODEL_PATH)

        elapsed = round(time.time() - t0, 1)
        logger.info(f"Training complete in {elapsed}s | accuracy={accuracy:.3f} | samples={n_samples}")

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
    feedback_path = TRADE_FEEDBACK_PATH
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
