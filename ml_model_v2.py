#!/usr/bin/env python3
"""
VolTradeAI — ML Model v3: Pro Architecture
============================================
WHAT CHANGED FROM v2:
  ❌ OLD: Fixed 2% binary label — treats NVDA and JNJ identically
  ✅ NEW: Volatility-adjusted triple-barrier labels (López de Prado)
          NVDA needs 4%+ to be a win; JNJ needs 1.2%

  ❌ OLD: Random 80/20 train/test split — future data leaks into training
  ✅ NEW: TimeSeriesSplit with 5-day embargo — no look-ahead, honest accuracy

  ❌ OLD: One model for all regimes (bull and bear with same weights)
  ✅ NEW: Regime-conditional ensemble — separate model per regime
          bull_model trained on bull bars, bear_model on bear bars
          Blended at inference by current regime probability

  ❌ OLD: 25 features, redundant (4 momentum variants)
  ✅ NEW: 26 features — removed 5 dead/collinear (intel_score, news_sentiment,
          insider_signal, volume_acceleration, float_turnover), added 6 real signals:
          earnings_surprise, cross_sec_rank, put_call_proxy,
          vol_of_vol, frac_diff_price, idiosyncratic_return

  ❌ OLD: Single model decides entry (no false-positive filter)
  ✅ NEW: Meta-labeling ready — after 50 trades, secondary model
          trained on actual outcomes filters false positives

TRAINING TARGETS:
  Stock ML: triple-barrier label — profit/stop/timeout, vol-adjusted
  Options ML: same structure applied to IV crush / premium decay outcomes
"""

import os, json, time, logging, warnings
# Cap OpenBLAS/MKL threads — Railway container has limited PIDs
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

import requests
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ml_model_v3")

try:
    from system_config import BASE_CONFIG, DATA_DIR
except ImportError:
    BASE_CONFIG = {"ML_FEATURE_COUNT": 26, "ML_MIN_SAMPLES": 200, "ML_TARGET_RETURN": 2.0}
    DATA_DIR = os.environ.get("VOLTRADE_DATA_DIR", "/data/voltrade")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

ALPACA_KEY    = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
FINNHUB_KEY   = os.environ.get("FINNHUB_KEY", "")
DATA_URL      = "https://data.alpaca.markets"

MODEL_PATH        = os.path.join(DATA_DIR, "voltrade_ml_v2.pkl")       # keep same path for compatibility
FEEDBACK_PATH     = os.path.join(DATA_DIR, "voltrade_trade_feedback.json")
META_MODEL_PATH   = os.path.join(DATA_DIR, "voltrade_ml_meta.pkl")
OPTIONS_MODEL_PATH= os.path.join(DATA_DIR, "voltrade_ml_options.pkl")

def _h(): return {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}


def _classify_regime(vxx_ratio: float, spy_vs_ma50: float) -> str:
    """Unified regime classification — delegates to regime_util.

    FIX 2026-04-20 (Bug #7): Previously each module (system_config, ml_model_v2,
    markov_regime) had its own thresholds. Now all three import from
    regime_util.classify_regime so changes propagate everywhere.
    """
    try:
        from regime_util import classify_regime
        return classify_regime(vxx_ratio, spy_vs_ma50)
    except ImportError:
        # Backward-compat fallback — matches old thresholds exactly
        if vxx_ratio >= 1.15 or spy_vs_ma50 < 0.94:
            return "bear"
        elif vxx_ratio <= 0.95 and spy_vs_ma50 >= 0.98:
            return "bull"
        else:
            return "neutral"


def _frac_diff(closes: list, d: float = 0.4) -> float:
    """Fractionally differentiated log price (fixed-width window, d=0.4).

    Preserves ~85% of price memory while achieving stationarity.
    Research: López de Prado 2018.
    Reusable by both training (_compute_features) and inference (bot_engine).
    """
    if len(closes) < 20:
        return 0.0
    w = [1.0]
    for k in range(1, 20):
        w.append(-w[-1] * (d - k + 1) / k)
    w = np.array(w[::-1])
    price_slice = np.array(closes[-20:])
    val = float(np.dot(w, np.log(price_slice + 1e-8))) if len(price_slice) == 20 else 0.0
    return max(-3.0, min(3.0, val))

# ══════════════════════════════════════════════════════════════════
# FEATURE COLUMNS — 34 features (26 base + 3 intel re-added + 5 new)
# ══════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    # Technical momentum (4)
    "momentum_1m",        # 21-day return
    "momentum_3m",        # 63-day return
    "rsi_14",             # RSI
    "price_vs_52w_high",  # % from 52-week high (George & Hwang 2004 — strongest predictor)
    # Volume (1)
    "volume_ratio",       # today / 20d avg
    # Volatility (4)
    "ewma_vol",           # EWMA realized vol
    "atr_pct",            # ATR as % of price
    "vrp",                # variance risk premium (IV - RV)
    "iv_rank_proxy",      # VXX-based IV rank 0-100
    # Microstructure (3)
    "vwap_position",      # above/below VWAP
    "adx",                # trend strength
    "range_pct",          # intraday range
    # Regime (5)
    "vxx_ratio",          # VXX / 30d avg
    "spy_vs_ma50",        # SPY / 50d MA
    "markov_state",       # 0=bear, 1=neutral, 2=bull
    "regime_score",       # 0-100 combined regime
    "sector_momentum",    # sector vs SPY
    # Additive features (6)
    "cross_sec_rank",     # rank of today's move vs all stocks
    "earnings_surprise",  # last EPS beat/miss direction (+1/-1/0)
    "put_call_proxy",     # put/call volume ratio from options
    "vol_of_vol",         # std dev of VXX
    "frac_diff_price",    # fractionally differentiated price
    "idiosyncratic_ret",  # stock return MINUS what market/sector explains
    # Change today (3)
    "change_pct_today",
    "above_ma10",
    "trend_strength",
    # Intel features — re-added with real data pipelines (3)
    "intel_score",        # composite: 0.4*news + 0.3*insider + 0.3*earnings
    "news_sentiment",     # Alpaca news API headline sentiment (-1 to 1)
    "insider_signal",     # SEC EDGAR insider buying/selling signal
    # New professional features (5)
    "iv_rank_stock",      # per-stock IV rank (0-100 percentile)
    "days_to_earnings",   # normalized days to next earnings (0-1)
    "credit_spread",      # TLT-HYG 21d return spread (risk-off indicator)
    "market_breadth",     # % of universe above 50d MA (0-1)
    "put_call_ratio",     # CBOE equity put/call ratio (VIX proxy if unavailable)
]

assert len(FEATURE_COLS) == 34, f"Expected 34 features, got {len(FEATURE_COLS)}"

CALIBRATOR_PATH = os.path.join(DATA_DIR, "voltrade_ml_calibrator.pkl")

# ── Model cache — load once, reuse for all inference calls ──────
# Without this, joblib.load() deserializes the full LightGBM model + scalers
# on every ml_score() call (~20+ times per scan cycle), causing memory spikes.
#
# THREAD SAFETY FIX (2026-04-20): _model_cache is accessed by parallel workers
# in bot_engine.deep_score() via ThreadPoolExecutor. Without a lock, two
# workers can both see bundle=None, both call joblib.load() (50MB allocation
# each), and the second overwrite leaks the first. Lock prevents the race.
import threading as _threading_ml
_model_cache = {"bundle": None, "mtime": 0.0, "calibrator": None, "cal_mtime": 0.0}
_model_cache_lock = _threading_ml.Lock()

# ══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════

def _fetch_training_bars(days: int = 365, max_tickers: int = 200) -> dict:
    """Fetch daily bars for training universe."""
    tickers = []
    try:
        r = requests.get(f"{DATA_URL}/v1beta1/screener/stocks/most-actives?by=volume&top=50",
            headers=_h(), timeout=10)
        tickers = [s["symbol"] for s in r.json().get("most_actives", []) if s.get("symbol")]
    except Exception: pass

    core = ["AAPL","MSFT","NVDA","AMD","META","GOOGL","AMZN","TSLA","JPM","BAC",
            "V","MA","COIN","MSTR","PLTR","CRWD","HOOD","SOFI","AFRM","UPST",
            "SPY","QQQ","IWM","VXX","GLD","XLE","XLF","XLK","TLT","HYG"]
    for t in core:
        if t not in tickers: tickers.append(t)
    tickers = list(dict.fromkeys(tickers))[:max_tickers]

    # Use fixed 2020-01-01 start to include full regime diversity:
    #   2020: COVID panic (VXX>1.30), 2021: bull (VXX<0.90), 2022: bear (VXX>1.15)
    #   This gives the model real examples of all 5 regimes
    start_date = "2020-01-01"
    all_bars: dict = {}
    for i in range(0, len(tickers), 10):
        batch = tickers[i:i+10]
        try:
            r = requests.get(f"{DATA_URL}/v2/stocks/bars",
                params={"symbols": ",".join(batch), "timeframe": "1Day",
                        "start": start_date, "limit": 10000,
                        "adjustment": "all", "feed": "sip"},
                headers=_h(), timeout=20)
            for sym, bars in r.json().get("bars", {}).items():
                all_bars[sym] = bars
        except Exception: continue
    return all_bars


def _fetch_earnings_surprises(tickers: list) -> Dict[str, float]:
    """Fetch last EPS surprise direction for a list of tickers."""
    surprises = {}
    for sym in tickers[:50]:  # Cap to avoid rate limits
        try:
            r = requests.get("https://finnhub.io/api/v1/stock/earnings",
                params={"symbol": sym, "token": FINNHUB_KEY}, timeout=5)
            data = r.json()
            if data and len(data) > 0:
                sp = data[0].get("surprisePercent", 0) or 0
                surprises[sym] = 1.0 if sp > 2 else (-1.0 if sp < -2 else 0.0)
        except Exception: pass
    return surprises

# ══════════════════════════════════════════════════════════════════
# HELPER: Historical news sentiment via Alpaca news API
# ══════════════════════════════════════════════════════════════════

def _fetch_historical_news_sentiment(ticker: str, date_str: str) -> float:
    """Fetch news sentiment for a ticker on a specific date using Alpaca news API."""
    url = "https://data.alpaca.markets/v1beta1/news"
    params = {
        "symbols": ticker,
        "start": f"{date_str}T00:00:00Z",
        "end": f"{date_str}T23:59:59Z",
        "limit": 10,
    }
    headers = {
        "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            news = resp.json().get("news", [])
            if not news:
                return 0.0
            pos_words = {"beat", "surge", "rise", "gain", "up", "rally", "profit", "record", "strong", "upgrade"}
            neg_words = {"miss", "fall", "drop", "loss", "down", "crash", "cut", "weak", "downgrade", "decline"}
            score = 0.0
            for article in news:
                headline = article.get("headline", "").lower()
                pos = sum(1 for w in pos_words if w in headline)
                neg = sum(1 for w in neg_words if w in headline)
                score += (pos - neg)
            return max(-1.0, min(1.0, score / max(len(news), 1)))
        return 0.0
    except Exception:
        return 0.0


def _fetch_insider_signal(ticker: str) -> float:
    """Fetch insider buying/selling signal from intelligence.py."""
    try:
        from intelligence import get_insider_activity
        insider = get_insider_activity(ticker)
        if insider and isinstance(insider, dict):
            return float(insider.get("signal", 0.0) or 0.0)
    except Exception:
        pass
    return 0.0


# ══════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION — 34 features
# ══════════════════════════════════════════════════════════════════

def _compute_features(bars: list, idx: int, all_bars: dict,
                       ticker: str, vxx_bars: list, spy_bars: list,
                       earnings_surprise: float = 0.0,
                       cross_sec_rank: float = 0.5,
                       news_sentiment: float = 0.0,
                       insider_signal: float = 0.0,
                       tlt_bars: list = None,
                       hyg_bars: list = None,
                       breadth_pct: float = 0.5) -> Optional[dict]:
    """
    Compute all 34 features for a bar at index idx.
    """
    if idx < 25 or idx >= len(bars) - 5:
        return None

    closes  = [b["c"] for b in bars]
    volumes = [b.get("v", 0) for b in bars]
    highs   = [bars[i].get("h", closes[i]) for i in range(len(bars))]
    lows    = [bars[i].get("l", closes[i]) for i in range(len(bars))]
    opens   = [bars[i].get("o", closes[i]) for i in range(len(bars))]

    c = closes[idx]
    if c <= 0 or volumes[idx] < 300_000:
        return None

    # ── Momentum ──────────────────────────────────────────────────
    mom_1m  = (c - closes[idx-21]) / closes[idx-21] * 100 if idx >= 21 else 0
    mom_3m  = (c - closes[idx-63]) / closes[idx-63] * 100 if idx >= 63 else 0

    # ── Volume ────────────────────────────────────────────────────
    vol_20  = [volumes[max(0,idx-j-1)] for j in range(20)]
    vol_avg = sum(v for v in vol_20 if v > 0) / max(1, sum(1 for v in vol_20 if v > 0))
    volume_ratio = volumes[idx] / vol_avg if vol_avg > 0 else 1.0

    # ── Volatility ────────────────────────────────────────────────
    rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(max(1,idx-20), idx) if closes[i-1] > 0]
    ewma_vol = float(np.std(rets) * 100) if rets else 2.0

    atr_list = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
                for i in range(max(1,idx-14), idx) if closes[i-1] > 0]
    atr = (sum(atr_list)/len(atr_list)) if atr_list else c * 0.02
    atr_pct = atr / c * 100

    # ── RSI ───────────────────────────────────────────────────────
    close_slice = closes[max(0,idx-15):idx]
    if len(close_slice) >= 14:
        gains  = [max(0, close_slice[i]-close_slice[i-1]) for i in range(1,len(close_slice))]
        losses = [max(0, close_slice[i-1]-close_slice[i]) for i in range(1,len(close_slice))]
        ag = sum(gains[-14:])/14; al = sum(losses[-14:])/14
        rsi_14 = 100 - (100/(1+ag/al)) if al > 0 else 50.0
    else:
        rsi_14 = 50.0

    # ── 52-week high ──────────────────────────────────────────────
    hi52   = max(highs[max(0,idx-252):idx]) if idx >= 5 else c
    p_52h  = (c - hi52) / hi52 * 100

    # ── Regime features (VXX + SPY) ───────────────────────────────
    _vxx_idx = min(idx, len(vxx_bars)-1) if vxx_bars else 0
    vxx_close = float(vxx_bars[_vxx_idx]["c"]) if vxx_bars and _vxx_idx < len(vxx_bars) else 15.0
    vxx_hist30 = [float(vxx_bars[min(max(0,_vxx_idx-j-1),len(vxx_bars)-1)]["c"]) for j in range(30) if vxx_bars and min(max(0,_vxx_idx-j-1),len(vxx_bars)-1) >= 0]
    vxx_avg30  = sum(vxx_hist30)/len(vxx_hist30) if vxx_hist30 else 15.0
    vxx_ratio  = vxx_close / vxx_avg30 if vxx_avg30 > 0 else 1.0

    _spy_idx = min(idx, len(spy_bars)-1) if spy_bars else 0
    spy_close = float(spy_bars[_spy_idx]["c"]) if spy_bars and _spy_idx < len(spy_bars) else 500.0
    spy_hist50 = [float(spy_bars[min(max(0,_spy_idx-j-1),len(spy_bars)-1)]["c"]) for j in range(50) if spy_bars and min(max(0,_spy_idx-j-1),len(spy_bars)-1) >= 0]
    spy_ma50   = sum(spy_hist50)/len(spy_hist50) if spy_hist50 else spy_close
    spy_vs_ma50= spy_close / spy_ma50 if spy_ma50 > 0 else 1.0

    vxx_52 = [float(vxx_bars[min(max(0,_vxx_idx-j-1),len(vxx_bars)-1)]["c"]) for j in range(252) if vxx_bars and min(max(0,_vxx_idx-j-1),len(vxx_bars)-1) >= 0]
    vxx_lo = min(vxx_52) if vxx_52 else 10
    vxx_hi = max(vxx_52) if vxx_52 else 50
    iv_rank= (vxx_close-vxx_lo)/(vxx_hi-vxx_lo)*100 if vxx_hi > vxx_lo else 50.0
    vrp    = max(-20, min(20, (vxx_ratio - 1.0) * 50))

    regime_score = max(0, min(100, (spy_vs_ma50-0.94)/0.12*50 + (1.3-vxx_ratio)/0.6*30 + 20))

    # Markov state proxy (simplified — real Markov runs separately)
    spy_rets_5 = [(float(spy_bars[min(_spy_idx-j,len(spy_bars)-1)]["c"])-float(spy_bars[min(max(0,_spy_idx-j-1),len(spy_bars)-1)]["c"]))/float(spy_bars[min(max(0,_spy_idx-j-1),len(spy_bars)-1)]["c"])
                  for j in range(5) if spy_bars and _spy_idx-j > 0 and _spy_idx-j < len(spy_bars)]
    markov_state = 2.0 if (len(spy_rets_5) >= 3 and sum(spy_rets_5)/len(spy_rets_5) > 0.002) else \
                   0.0 if (len(spy_rets_5) >= 3 and sum(spy_rets_5)/len(spy_rets_5) < -0.002) else 1.0

    # Sector momentum: simplified (stock vs SPY 5d return)
    spy_5d = (spy_close - float(spy_bars[min(max(0,idx-5), len(spy_bars)-1)]["c"])) / float(spy_bars[min(max(0,idx-5), len(spy_bars)-1)]["c"]) * 100 if idx >= 5 and len(spy_bars) > idx-5 and spy_bars[min(max(0,idx-5),len(spy_bars)-1)] else 0
    stock_5d = (c - closes[max(0,idx-5)]) / closes[max(0,idx-5)] * 100 if idx >= 5 else 0
    sector_momentum = stock_5d - spy_5d  # outperformance vs market

    # ── Microstructure ────────────────────────────────────────────
    vwap_pos   = 1.0 if c > opens[idx] else 0.0  # simplified VWAP proxy
    range_pct  = (highs[idx]-lows[idx])/c*100 if c > 0 else 0
    change_pct = (c - closes[idx-1]) / closes[idx-1] * 100 if idx >= 1 and closes[idx-1] > 0 else 0
    above_ma10 = 1.0 if c > (sum(closes[max(0,idx-10):idx])/10) else 0.0
    trend_str  = min(abs(change_pct) / max(ewma_vol, 0.1), 5.0)

    # ADX proxy (simplified)
    if idx >= 14:
        up_moves   = [max(0, highs[i]-highs[i-1]) for i in range(idx-14, idx)]
        down_moves = [max(0, lows[i-1]-lows[i])   for i in range(idx-14, idx)]
        dm_plus  = sum(up_moves) / 14
        dm_minus = sum(down_moves) / 14
        adx = abs(dm_plus - dm_minus) / max(dm_plus + dm_minus, 0.001) * 100
    else:
        adx = 20.0

    # ── NEW FEATURE 1: Vol-of-vol ─────────────────────────────────
    # Std dev of VXX over recent 10 days — when uncertainty about uncertainty
    # is high, all signals are weaker (research: Mencía & Sentana 2013)
    vxx_10_rets = []
    for j in range(1, 11):
        if idx-j >= 0 and idx-j < len(vxx_bars) and vxx_bars[idx-j] and vxx_bars[idx-j-1]:
            prev = float(vxx_bars[idx-j-1]["c"]); curr = float(vxx_bars[idx-j]["c"])
            if prev > 0: vxx_10_rets.append((curr-prev)/prev)
    vol_of_vol = float(np.std(vxx_10_rets) * 100) if len(vxx_10_rets) >= 5 else 2.0

    # ── NEW FEATURE 2: Fractionally Differentiated Price ─────────
    # d=0.4 preserves 85% of price memory while achieving stationarity
    # Research: López de Prado 2018 — preserves predictive information
    # that regular returns (d=1) throw away
    frac_diff = _frac_diff(closes[max(0, idx-20):idx]) if idx >= 20 else 0.0

    # ── NEW FEATURE 3: Idiosyncratic Return ──────────────────────
    # Stock return minus what SPY/sector explains
    # This is the "pure alpha" component — what makes this stock move
    # beyond just following the market up or down
    # Research: Ang et al 2006 — idiosyncratic vol predicts returns
    if idx >= 21 and len(spy_hist50) >= 21:
        stock_21d  = mom_1m
        spy_21d_idx = max(0, min(idx - 21, len(spy_bars) - 1))  # bounds-safe
        spy_21d_bar = spy_bars[spy_21d_idx] if spy_bars and spy_21d_idx < len(spy_bars) else None
        spy_21d    = (spy_close - float(spy_21d_bar["c"])) / float(spy_21d_bar["c"]) * 100 if spy_21d_bar and float(spy_21d_bar["c"]) > 0 else 0
        # APPROXIMATION NOTE: beta_proxy is NOT a real beta coefficient.
        # A real beta requires a linear regression of stock returns on SPY returns
        # over at least 30-60 days. Using 1.2 for high-volume days and 0.8 for
        # normal days is a rough heuristic — high-volume stocks TEND to be more
        # volatile (higher beta) but this is not a statistically valid estimate.
        # The idiosyncratic_ret feature computed here is directionally useful but
        # will overstate the idiosyncratic component for low-beta stocks and
        # understate it for true high-beta names.
        # TODO: Replace with actual OLS beta from rolling 60-day regression.
        beta_proxy = 1.2 if volume_ratio > 2 else 0.8  # Approximate — see note above
        idio_ret   = stock_21d - beta_proxy * spy_21d
    else:
        idio_ret = 0.0

    # ── NEW FEATURE 4: Put/Call Proxy ────────────────────────────
    put_call_proxy = -1.0 if vrp > 8 else (1.0 if vrp < -5 else 0.0)

    # ── Intel features (re-added with real data) ─────────────────
    intel_score = 0.4 * news_sentiment + 0.3 * insider_signal + 0.3 * earnings_surprise

    # ── NEW: iv_rank_stock — per-stock IV rank using VXX scaled by beta
    # At training time, use VXX rank as proxy scaled by volume_ratio (beta approx)
    beta_approx = min(volume_ratio, 3.0) / 2.0  # rough beta from volume
    iv_rank_stock = min(100.0, iv_rank * beta_approx)

    # ── NEW: days_to_earnings — normalized (0-1, 0=today, 1=60+ days)
    # At training time, default to 0.5 (unknown); caller can override
    days_to_earnings = 0.5  # default; overridden at inference with real data

    # ── NEW: credit_spread — TLT-HYG 21d return spread
    credit_spread = 0.0
    if tlt_bars and hyg_bars:
        _tlt_idx = min(idx, len(tlt_bars) - 1)
        _hyg_idx = min(idx, len(hyg_bars) - 1)
        if _tlt_idx >= 21 and _hyg_idx >= 21:
            tlt_now = float(tlt_bars[_tlt_idx]["c"])
            tlt_21 = float(tlt_bars[max(0, _tlt_idx - 21)]["c"])
            hyg_now = float(hyg_bars[_hyg_idx]["c"])
            hyg_21 = float(hyg_bars[max(0, _hyg_idx - 21)]["c"])
            if tlt_21 > 0 and hyg_21 > 0:
                tlt_ret = (tlt_now / tlt_21) - 1
                hyg_ret = (hyg_now / hyg_21) - 1
                credit_spread = round(tlt_ret - hyg_ret, 4)

    # ── NEW: market_breadth — pct of universe above 50d MA
    market_breadth = breadth_pct

    # ── NEW: put_call_ratio — use VIX proxy: vxx_level / 20
    put_call_ratio = round(vxx_close / 20.0, 3) if vxx_close else 1.0

    return {
        "momentum_1m":        round(mom_1m, 3),
        "momentum_3m":        round(mom_3m, 3),
        "rsi_14":             round(rsi_14, 2),
        "price_vs_52w_high":  round(p_52h, 2),
        "volume_ratio":       round(min(volume_ratio, 10.0), 3),
        "ewma_vol":           round(ewma_vol, 3),
        "atr_pct":            round(atr_pct, 3),
        "vrp":                round(vrp, 3),
        "iv_rank_proxy":      round(iv_rank, 2),
        "vwap_position":      vwap_pos,
        "adx":                round(adx, 2),
        "range_pct":          round(range_pct, 3),
        "vxx_ratio":          round(vxx_ratio, 4),
        "spy_vs_ma50":        round(spy_vs_ma50, 4),
        "markov_state":       markov_state,
        "regime_score":       round(regime_score, 1),
        "sector_momentum":    round(sector_momentum, 3),
        "cross_sec_rank":     round(cross_sec_rank, 3),
        "earnings_surprise":  earnings_surprise,
        "put_call_proxy":     put_call_proxy,
        "vol_of_vol":         round(vol_of_vol, 3),
        "frac_diff_price":    round(frac_diff, 4),
        "idiosyncratic_ret":  round(idio_ret, 3),
        "change_pct_today":   round(change_pct, 3),
        "above_ma10":         above_ma10,
        "trend_strength":     round(trend_str, 3),
        # Intel features (3)
        "intel_score":        round(intel_score, 3),
        "news_sentiment":     round(news_sentiment, 3),
        "insider_signal":     round(insider_signal, 3),
        # New professional features (5)
        "iv_rank_stock":      round(iv_rank_stock, 2),
        "days_to_earnings":   round(days_to_earnings, 3),
        "credit_spread":      round(credit_spread, 4),
        "market_breadth":     round(market_breadth, 3),
        "put_call_ratio":     round(put_call_ratio, 3),
    }


# ══════════════════════════════════════════════════════════════════
# TRIPLE-BARRIER LABEL — volatility-adjusted (López de Prado)
# ══════════════════════════════════════════════════════════════════

def _triple_barrier_label(bars: list, idx: int, atr_pct: float,
                            pt_mult: float = 1.5, sl_mult: float = 1.0,
                            max_days: int = 5) -> int:
    """
    Label a bar using triple-barrier method:
      +1 = profit-take hit first (pt_mult × ATR above entry)
       0 = timeout (neither barrier hit in max_days)
      -1 = stop-loss hit first (sl_mult × ATR below entry)

    Why this beats fixed 2%:
      - ATR-adjusted: NVDA with ATR=3% needs 4.5% to be a win
      - JNJ with ATR=0.8% needs 1.2% to be a win
      - Timeout is genuinely neutral (0), not misclassified as a loss
      - Maps directly to actual bot behavior (stops + TP)
    """
    if idx + max_days >= len(bars):
        return 0

    entry = bars[idx]["c"]
    if entry <= 0:
        return 0

    pt_pct = max(0.005, atr_pct * pt_mult / 100)  # min 0.5% profit target
    sl_pct = max(0.003, atr_pct * sl_mult / 100)  # min 0.3% stop loss

    pt_price = entry * (1 + pt_pct)
    sl_price = entry * (1 - sl_pct)

    for future_idx in range(idx + 1, min(idx + max_days + 1, len(bars))):
        h = bars[future_idx].get("h", bars[future_idx]["c"])
        l = bars[future_idx].get("l", bars[future_idx]["c"])
        if h >= pt_price:
            return 1   # profit-take hit
        if l <= sl_price:
            return -1  # stop-loss hit

    return 0  # timeout — neutral


# ══════════════════════════════════════════════════════════════════
# PURGED TIME-SERIES CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════

def _purged_train_test_split(X: np.ndarray, y: np.ndarray,
                               embargo_periods: int = 5,
                               n_tickers_per_date: int = 1) -> Tuple:
    """
    Proper temporal split with embargo gap.

    Why random split is wrong for financial data:
      A training bar at day T with a 5-day label window "knows" what
      happened at T+1 through T+5. A test bar at T+3 overlaps this window.
      The model learns to predict T+3 from a bar that "saw" T+3.
      This is look-ahead bias — reports inflated accuracy.

    Fix: Use the LAST 20% of bars as the test set (pure holdout),
    with an embargo gap scaled to trading DAYS (not rows) to prevent
    the training label window from touching test bars.

    EMBARGO SCALING FIX: With multi-ticker data, rows are stacked across
    tickers for the same dates. A 5-row embargo on 50 tickers = only 0.1
    trading days of embargo, NOT 5 days. The embargo must be:
        embargo_rows = embargo_periods (in days) * n_tickers_per_date
    Pass n_tickers_per_date=len(unique_tickers) from the caller.
    Default n_tickers_per_date=1 preserves backward compatibility for
    single-ticker calls (options model, feedback-only training).
    """
    n = len(X)
    # Use last 20% for test
    test_start = int(n * 0.80)
    # Scale embargo from trading days to rows: 5 days * ~n tickers per day
    # Cap embargo at 20% of data to prevent empty test sets on small datasets
    raw_embargo = embargo_periods * max(1, n_tickers_per_date)
    embargo_rows = min(raw_embargo, len(X) // 5)  # Never more than 20% of data
    train_end = test_start - embargo_rows

    if train_end <= 50:
        # Not enough data for proper split — use simple 70/30
        split = int(n * 0.70)
        return X[:split], X[split:], y[:split], y[split:]

    X_train = X[:train_end]
    X_test  = X[test_start:]
    y_train = y[:train_end]
    y_test  = y[test_start:]

    return X_train, X_test, y_train, y_test


def _walk_forward_cv(X, y, n_splits=5, embargo_days=5, n_tickers_approx=50):
    """
    Purged walk-forward cross-validation.
    Split timeline into n_splits+1 equal folds.
    For each fold k (1..n_splits):
      Train on folds 0..k-1
      Embargo: skip embargo_days * n_tickers_approx rows
      Test on fold k
    Returns list of (train_idx, test_idx) pairs.
    """
    n = len(X)
    fold_size = n // (n_splits + 1)
    splits = []
    min_train_size = max(500, fold_size)

    for k in range(1, n_splits + 1):
        train_end = k * fold_size
        embargo = embargo_days * n_tickers_approx
        test_start = train_end + embargo
        test_end = (k + 1) * fold_size

        if test_start >= n or train_end < min_train_size:
            continue

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, min(test_end, n)))
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


# ══════════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL TRAINING DATA BUILDER
# ══════════════════════════════════════════════════════════════════

def _build_training_data(all_bars: dict,
                          earnings_surprises: dict = None) -> Tuple:
    """
    Build training data with:
    1. Triple-barrier labels (volatility-adjusted, timeouts relabeled)
    2. Regime labels for conditional training
    3. All 34 features including intel and new professional features

    Returns: X, y, regimes (array of 'bull'/'bear'/'neutral' per row)
    """
    vxx_bars_raw = all_bars.get("VXX", {})
    spy_bars_raw = all_bars.get("SPY", {})
    tlt_bars_raw = all_bars.get("TLT", {})
    hyg_bars_raw = all_bars.get("HYG", {})

    def to_list(raw):
        if isinstance(raw, dict):
            return [raw[d] for d in sorted(raw.keys())]
        return sorted(raw, key=lambda b: b.get("t","")) if raw else []

    vxx_list = to_list(vxx_bars_raw)
    spy_list = to_list(spy_bars_raw)
    tlt_list = to_list(tlt_bars_raw)
    hyg_list = to_list(hyg_bars_raw)

    # BUGFIX P0-3: Build date→index maps for ALL reference series so we can
    # align by actual calendar date rather than positional index. Previously,
    # a PLTR row at stock_idx=100 (i.e. PLTR's 100th trading day, ~mid-2021)
    # was paired with vxx_bars[100] (VXX's 100th day, ~late-2009). Every IPO-era
    # ticker trained on wrong-era regime features.
    def _date_key(bar):
        t = bar.get("t", "")
        if isinstance(t, str):
            return t[:10]  # YYYY-MM-DD
        return str(t)[:10]

    vxx_date_idx = {_date_key(b): i for i, b in enumerate(vxx_list)}
    spy_date_idx = {_date_key(b): i for i, b in enumerate(spy_list)}
    tlt_date_idx = {_date_key(b): i for i, b in enumerate(tlt_list)}
    hyg_date_idx = {_date_key(b): i for i, b in enumerate(hyg_list)}

    X_rows, y_labels, regimes_list = [], [], []

    # Compute cross-sectional ranks BY DATE (also previously used positional idx
    # which collides across tickers that start on different dates).
    all_changes_by_date: Dict[str, list] = {}

    # Pre-compute market breadth: % of tickers above 50d MA PER DATE
    breadth_by_date: Dict[str, float] = {}
    ticker_closes_map = {}
    ticker_dates_map = {}
    tradeable_tickers = [t for t in all_bars if t not in ("SPY","QQQ","IWM","VXX","GLD","TLT","HYG")]

    for ticker, bars_raw in all_bars.items():
        if ticker in ("SPY","QQQ","IWM","VXX","GLD","TLT","HYG"): continue
        bars = to_list(bars_raw)
        if len(bars) < 30: continue
        closes = [b["c"] for b in bars]
        dates = [_date_key(b) for b in bars]
        ticker_closes_map[ticker] = closes
        ticker_dates_map[ticker] = dates
        for idx in range(1, len(bars)):
            if closes[idx-1] > 0:
                chg = (closes[idx]-closes[idx-1])/closes[idx-1]*100
                all_changes_by_date.setdefault(dates[idx], []).append(chg)

    # Compute breadth per CALENDAR DATE (not positional idx). Previously, idx=100
    # meant "100 bars into each ticker's own history", so PLTR's idx=100 (2021-ish)
    # was aggregated with AAPL's idx=100 (2020-ish) — breadth was temporally smeared.
    all_dates_sorted = sorted({d for dates in ticker_dates_map.values() for d in dates})
    for target_date in all_dates_sorted:
        above_50ma = 0
        total = 0
        for tkr, t_closes in ticker_closes_map.items():
            t_dates = ticker_dates_map.get(tkr, [])
            if target_date not in t_dates:
                continue
            j = t_dates.index(target_date)  # O(n) but dates are monotonic; acceptable for training-time work
            if j < 50:
                continue
            ma50 = sum(t_closes[j-50:j]) / 50
            if ma50 > 0:
                total += 1
                if t_closes[j] > ma50:
                    above_50ma += 1
        if total > 0:
            breadth_by_date[target_date] = above_50ma / total

    for ticker, bars_raw in all_bars.items():
        if ticker in ("SPY","QQQ","IWM","VXX","GLD","TLT","HYG"): continue
        bars = to_list(bars_raw)
        if len(bars) < 30: continue
        closes = [b["c"] for b in bars]
        dates  = [_date_key(b) for b in bars]

        earn_surp = (earnings_surprises or {}).get(ticker, 0.0)

        # P0-3 FIX: pre-compute aligned VXX/SPY/TLT/HYG slices keyed to THIS
        # stock's calendar dates. _compute_features then just uses stock's idx
        # and everything lines up.
        def _aligned(ref_list, ref_date_idx):
            """Return a list of length len(bars) where position i holds the ref_list
            bar whose date equals (or most-recently precedes) bars[i]'s date.
            Fills missing early dates with the first available bar."""
            out = []
            last_good_idx = None
            for d in dates:
                j = ref_date_idx.get(d)
                if j is None and last_good_idx is None:
                    # walk backwards through sorted ref_date_idx keys up to d
                    # (use linear scan once; ref series are small ~2500 bars)
                    for rd in sorted(ref_date_idx.keys()):
                        if rd <= d:
                            last_good_idx = ref_date_idx[rd]
                        else:
                            break
                    out.append(ref_list[last_good_idx] if last_good_idx is not None else (ref_list[0] if ref_list else None))
                elif j is None:
                    out.append(ref_list[last_good_idx] if last_good_idx is not None else None)
                else:
                    last_good_idx = j
                    out.append(ref_list[j])
            return out

        aligned_vxx = _aligned(vxx_list, vxx_date_idx)
        aligned_spy = _aligned(spy_list, spy_date_idx)
        aligned_tlt = _aligned(tlt_list, tlt_date_idx) if tlt_list else None
        aligned_hyg = _aligned(hyg_list, hyg_date_idx) if hyg_list else None

        for idx in range(25, len(bars) - 6):
            breadth = breadth_by_date.get(dates[idx], 0.5)
            feats = _compute_features(bars, idx, all_bars, ticker,
                                        aligned_vxx, aligned_spy, earn_surp,
                                        cross_sec_rank=0.5,
                                        news_sentiment=0.0,
                                        insider_signal=0.0,
                                        tlt_bars=aligned_tlt,
                                        hyg_bars=aligned_hyg,
                                        breadth_pct=breadth)
            if feats is None:
                continue

            # Cross-sectional rank for this DATE (fixes same bug as breadth).
            day_changes = all_changes_by_date.get(dates[idx], [])
            if day_changes and len(day_changes) > 10:
                stock_chg = (closes[idx]-closes[idx-1])/closes[idx-1]*100 if closes[idx-1]>0 else 0
                rank_pct  = sum(1 for x in day_changes if x <= stock_chg) / len(day_changes)
            else:
                rank_pct = 0.5
            feats["cross_sec_rank"] = rank_pct

            # Triple-barrier label (volatility-adjusted)
            atr_pct = feats.get("atr_pct", 2.0)
            label = _triple_barrier_label(bars, idx, atr_pct)

            if label == 0:
                # Relabel timeouts by realized P&L at expiry
                max_holding_days = 5
                if idx + max_holding_days < len(bars):
                    expiry_close = bars[idx + max_holding_days]["c"]
                    entry_price = bars[idx]["c"]
                    pnl_pct = (expiry_close - entry_price) / entry_price * 100
                    neutral_threshold = atr_pct * 0.3
                    if pnl_pct > neutral_threshold:
                        label = 1   # positive timeout
                    elif pnl_pct < -neutral_threshold:
                        label = -1  # negative timeout
                    # else: label stays 0 (true neutral → conservative: treat as loss)

            # Convert to binary: +1 → 1 (win), 0 or -1 → 0 (loss/no-trade)
            binary_label = 1 if label == 1 else 0

            # Regime at this bar — unified classification (see _classify_regime)
            vxx_r = feats.get("vxx_ratio", 1.0)
            spy_m = feats.get("spy_vs_ma50", 1.0)
            regime_label = _classify_regime(vxx_r, spy_m)

            X_rows.append([float(feats.get(col, 0) or 0) for col in FEATURE_COLS])
            y_labels.append(binary_label)
            regimes_list.append(regime_label)

    # Release large intermediate structures before converting to numpy
    del ticker_closes_map, ticker_dates_map, all_changes_by_date, breadth_by_date

    if len(X_rows) < 50:
        return None, None, None

    return (np.array(X_rows, dtype=np.float32),
            np.array(y_labels, dtype=np.int8),
            regimes_list)


# ══════════════════════════════════════════════════════════════════
# SELF-LEARNING FROM REAL TRADES
# ══════════════════════════════════════════════════════════════════

# Minimum code version for feedback to be considered valid.
# All trades from earlier versions ran on broken code (pre-27-bug-fix,
# pre-scale-out, pre-WebSocket monitoring, pre-HEAT-CAP fix) and would
# poison the model with outcomes caused by bugs, not bad signals.
MIN_FEEDBACK_VERSION = "1.0.33"

# ──────────────────────────────────────────────────────────────────────────
# ITEM 18 FIX 2026-04-20: Auto-clean poisoned feedback on first post-deploy run
# Bug #13 was writing every trade as `pnl_pct=0`. After the fix, these records
# still exist in voltrade_trade_feedback.json and will pollute Kelly sizing
# until they age out. This runs once on import — if we detect poisoned
# records (pnl_pct=0 + no code_version) covering >20% of the file, archive
# them to a backup and keep only the clean ones.
# ──────────────────────────────────────────────────────────────────────────
def _clean_poisoned_feedback_on_startup():
    """Idempotent one-time cleanup. Safe to call multiple times."""
    try:
        import os, json, time
        feedback_path = TRADE_FEEDBACK_PATH
        marker_path = os.path.join(DATA_DIR, "voltrade_feedback_cleaned.marker")
        if os.path.exists(marker_path):
            return  # Already cleaned
        if not os.path.exists(feedback_path):
            return  # No feedback file yet

        with open(feedback_path) as f:
            records = json.load(f)
        if not isinstance(records, list) or len(records) == 0:
            return

        # Classify: poisoned = (pnl_pct == 0 OR None) AND no code_version
        poisoned = [r for r in records
                    if (r.get("pnl_pct", 0) == 0 or r.get("pnl_pct") is None)
                    and not r.get("code_version")]
        clean = [r for r in records if r not in poisoned]

        poison_pct = len(poisoned) / len(records)
        if poison_pct < 0.20:
            # Not enough poison to bother — leave it alone
            # (ML loader will weight it low anyway via MIN_FEEDBACK_VERSION)
            open(marker_path, "w").write(f"skipped, only {poison_pct*100:.1f}% poisoned")
            return

        # Archive the poisoned records (in case we need to inspect later)
        backup_path = os.path.join(DATA_DIR, f"voltrade_trade_feedback_poisoned_{int(time.time())}.json")
        try:
            with open(backup_path, "w") as f:
                json.dump(poisoned, f)
        except Exception:
            pass

        # Write back only the clean records
        tmp_path = feedback_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(clean, f)
        os.replace(tmp_path, feedback_path)

        # Mark as done so we don't re-run
        with open(marker_path, "w") as f:
            f.write(f"cleaned {len(poisoned)}/{len(records)} records on {time.time()}")
    except Exception as e:
        # Never crash on cleanup — just log and continue
        import logging
        logging.getLogger("voltrade.ml").warning(f"Feedback cleanup failed: {e}")


# Run cleanup once on module import
try:
    _clean_poisoned_feedback_on_startup()
except Exception:
    pass



def _version_tuple(v):
    """Parse "1.0.33" -> (1, 0, 33) for correct numeric comparison.

    BUGFIX P0-4: Previously `t.get("code_version", "") >= MIN_FEEDBACK_VERSION`
    used lexicographic string comparison — "1.0.9" >= "1.0.33" is True, so
    trades from v1.0.9 (pre-bugfix, poisonous) passed the filter. Tuple
    comparison fixes this: (1,0,9) < (1,0,33).
    """
    if not v:
        return (0, 0, 0)
    try:
        parts = str(v).split(".")
        # Coerce each segment to int; strip any non-numeric prefix/suffix.
        def _to_int(s):
            num = ""
            started = False
            for ch in s:
                if ch.isdigit():
                    num += ch
                    started = True
                elif started:
                    break
            return int(num) if num else 0
        return tuple(_to_int(p) for p in parts[:3]) + (0,) * max(0, 3 - len(parts))
    except Exception:
        return (0, 0, 0)

def _load_trade_feedback() -> List[dict]:
    """
    Load trade feedback for training.

    FIX 2026-04-20 (MIN_FEEDBACK_VERSION data loss):
    Previously, bumping MIN_FEEDBACK_VERSION after a schema-changing bug fix
    silently discarded ALL older records. For a model that often has only
    200-500 training samples, losing a month of history kills learning.

    New behavior: old records are KEPT but marked with a _legacy_weight
    multiplier that the training loop applies as sample_weight * 0.4.
    This way the model still has the older signal at reduced confidence
    rather than losing it entirely. Records with genuinely incompatible
    schemas (missing entry_features) are still dropped.
    """
    try:
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH) as f:
                raw = json.load(f)

            _min_v = _version_tuple(MIN_FEEDBACK_VERSION)
            # Records that pass the version gate get full weight
            # Older records get weight=0.4 but stay in the pool
            valid = []
            legacy_kept = 0
            schema_drops = 0
            for t in raw:
                # Schema check — need entry_features to train at all
                if not t.get("entry_features"):
                    schema_drops += 1
                    continue
                is_legacy = _version_tuple(t.get("code_version", "")) < _min_v
                if is_legacy:
                    t = dict(t)  # shallow copy to avoid mutating raw file
                    t["_legacy_weight"] = 0.4
                    legacy_kept += 1
                valid.append(t)

            if legacy_kept > 0 or schema_drops > 0:
                import logging
                logging.getLogger("voltrade.ml").info(
                    f"[FEEDBACK] Loaded {len(valid)} trades "
                    f"(legacy weighted 0.4x: {legacy_kept}, schema-dropped: {schema_drops})")
            return valid
    except Exception:
        pass
    return []


def _build_feedback_training_data(trades: List[dict]) -> Tuple:
    """
    Build training rows from real closed trades with:
    - Exponential recency decay (30-day half-life)
    - ±20% P&L outlier filter
    - ATR-consistent labeling (matching triple-barrier thresholds)
    - Sample weights for LightGBM
    """
    current_ts = time.time()
    half_life_days = 30

    # Filter P&L outliers (bugs or black swans)
    trades = [t for t in trades if abs(t.get("pnl_pct", 0) or 0) <= 20.0]

    X_rows, y_labels, regimes, sample_weights = [], [], [], []
    for trade in trades:
        features = trade.get("entry_features", {})
        if not features: continue
        pnl_pct = trade.get("pnl_pct", 0) or 0

        # Label using ATR-consistent thresholds (matching triple-barrier)
        atr = features.get("atr_pct", 2.0) or 2.0
        if pnl_pct > atr * 1.5:  # Same as PT barrier multiplier
            label = 1
        elif pnl_pct < -atr * 1.0:  # Same as SL barrier multiplier
            label = 0
        else:
            label = 0  # timeout-equivalent → conservative

        # Exponential recency decay
        trade_ts = trade.get("timestamp", trade.get("time_filled", 0))
        if isinstance(trade_ts, str):
            try:
                trade_ts = datetime.fromisoformat(trade_ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                trade_ts = current_ts
        trade_age_days = (current_ts - float(trade_ts)) / 86400
        decay_weight = 2 ** (-trade_age_days / half_life_days)
        weight = 3.0 * decay_weight  # max 3x for new trades, decays over time
        # Legacy weight (from MIN_FEEDBACK_VERSION fix) — keeps old records
        # in the training pool at reduced confidence rather than dropping them
        weight *= trade.get("_legacy_weight", 1.0)

        row = [float(features.get(col, 0) or 0) for col in FEATURE_COLS]
        X_rows.append(row)
        y_labels.append(label)
        sample_weights.append(weight)
        regimes.append(features.get("regime_at_entry", "neutral").lower()
                       if "regime_at_entry" in trade else "neutral")

    if len(X_rows) < 15:
        return None, None, None, None

    return (np.array(X_rows, dtype=np.float32),
            np.array(y_labels, dtype=np.int8),
            regimes,
            np.array(sample_weights, dtype=np.float32))


# ══════════════════════════════════════════════════════════════════
# TRAIN MODEL — regime-conditional ensemble
# ══════════════════════════════════════════════════════════════════

def _train_single_lgbm(X_tr, X_te, y_tr, y_te, label="", sample_weight=None):
    """Train one LightGBM model, return (model, scaler, accuracy).
    sample_weight: optional per-row weights for feedback decay weighting."""
    if not HAS_SKLEARN:
        return None, None, 0.0

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    if HAS_LGB:
        # scale_pos_weight balances classes better than class_weight="balanced"
        # for LightGBM — ensures model spreads predictions across full range
        n_pos = int(np.sum(y_tr == 1)); n_neg = int(np.sum(y_tr == 0))
        spw = max(1.0, n_neg / max(n_pos, 1))  # weight positives up if minority
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.04, "num_leaves": 24,
            "min_child_samples": 10,
            "feature_fraction": 0.75, "bagging_fraction": 0.75,
            "bagging_freq": 5, "verbose": -1,
            "scale_pos_weight": spw,
            "min_gain_to_split": 0.01,  # forces model to find real splits
        }
        try:
            import pandas as pd
            dtrain = lgb.Dataset(pd.DataFrame(X_tr_sc, columns=FEATURE_COLS), label=y_tr,
                                 weight=sample_weight)
            dtest  = lgb.Dataset(pd.DataFrame(X_te_sc,  columns=FEATURE_COLS), label=y_te, reference=dtrain)
            model  = lgb.train(params, dtrain, num_boost_round=300,
                                valid_sets=[dtest],
                                callbacks=[lgb.early_stopping(30, verbose=False),
                                           lgb.log_evaluation(period=-1)])
            probs = model.predict(pd.DataFrame(X_te_sc, columns=FEATURE_COLS))
            preds = (probs > 0.5).astype(int)
            acc   = float(np.mean(preds == y_te))
            # Compute output spread — if too narrow, fallback to GBM
            spread = float(np.percentile(probs, 90) - np.percentile(probs, 10))
            if spread >= 0.08:  # good differentiation
                return model, scaler, acc
            logger.debug(f"LGB spread too narrow ({spread:.3f}), falling back ({label})")
        except Exception as e:
            logger.debug(f"LGB failed ({label}): {e}")

    # Fallback: GradientBoosting — often better calibrated for small datasets
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                     learning_rate=0.08, subsample=0.8,
                                     min_samples_leaf=8)
    gb.fit(X_tr_sc, y_tr, sample_weight=sample_weight)
    preds = gb.predict(X_te_sc)
    acc   = float(np.mean(preds == y_te))
    return gb, scaler, acc


def train_model(fast_mode: bool = False) -> dict:
    """
    Train the regime-conditional ensemble.

    Architecture:
      - One model trained on ALL data (global fallback)
      - One model trained ONLY on bull-regime bars
      - One model trained ONLY on bear-regime bars
      - One model trained ONLY on neutral-regime bars

    At inference: blend by current regime probability.
    If a regime has < 50 training rows, use the global model.

    Training uses:
      1. Real trade feedback (3× weight, highest priority)
      2. Generic market data with triple-barrier labels (bootstrap)
    """
    if not HAS_SKLEARN:
        return {"status": "skipped", "reason": "sklearn not installed"}

    t0 = time.time()

    # Step 1: Fetch training data
    all_bars = _fetch_training_bars(days=BASE_CONFIG.get("ML_LOOKBACK_DAYS", 365))  # 1yr for regime diversity
    if not all_bars:
        return {"status": "failed", "reason": "Could not fetch training bars"}

    # Fetch earnings surprises for the tickers we're training on
    ticker_list = [k for k in all_bars if k not in ("SPY","QQQ","IWM","VXX","GLD","TLT")]
    earnings_surprises = _fetch_earnings_surprises(ticker_list[:30])

    # Step 2: Build generic training data with triple-barrier labels
    X_gen, y_gen, reg_gen = _build_training_data(all_bars, earnings_surprises)

    # Release raw bar data — features have been extracted into X_gen
    del all_bars, earnings_surprises
    import gc; gc.collect()

    # Step 3: Load real trade feedback
    feedback    = _load_trade_feedback()
    X_fb, y_fb, reg_fb, fb_weights = None, None, None, None
    if len(feedback) >= 15:
        fb_result = _build_feedback_training_data(feedback)
        if fb_result[0] is not None:
            X_fb, y_fb, reg_fb, fb_weights = fb_result

    # Step 4: Combine — use sample_weight instead of tripling
    # Cap feedback at 15% of generic training data
    sample_weights = None
    if X_fb is not None and X_gen is not None:
        max_feedback_rows = int(len(X_gen) * 0.15)
        if len(X_fb) > max_feedback_rows:
            X_fb = X_fb[-max_feedback_rows:]
            y_fb = y_fb[-max_feedback_rows:]
            reg_fb = reg_fb[-max_feedback_rows:]
            fb_weights = fb_weights[-max_feedback_rows:]
        X_all = np.vstack([X_gen, X_fb])
        y_all = np.concatenate([y_gen, y_fb])
        reg_all = reg_gen + list(reg_fb)
        # Sample weights: 1.0 for generic, decay-weighted for feedback
        gen_weights = np.ones(len(X_gen), dtype=np.float32)
        sample_weights = np.concatenate([gen_weights, fb_weights])
        data_source = f"generic({len(X_gen)})+feedback({len(X_fb)})"
    elif X_gen is not None:
        X_all, y_all, reg_all = X_gen, y_gen, reg_gen
        data_source = f"generic_only({len(X_gen)})"
    elif X_fb is not None:
        X_all, y_all, reg_all = X_fb, y_fb, list(reg_fb)
        sample_weights = fb_weights
        data_source = f"feedback_only({len(X_fb)})"
    else:
        return {"status": "failed", "reason": "No training data"}

    # ── NEW: blend in shadow portfolio training data ─────────────────────
    try:
        from shadow_portfolio import load_shadow_data
        shadow = load_shadow_data(horizon_days=10, shadow_weight=0.3)
        if shadow is not None:
            X_shadow, y_shadow, w_shadow = shadow
            if sample_weights is None:
                sample_weights = np.ones(len(X_all), dtype=np.float32)
            X_all = np.vstack([X_all, X_shadow])
            y_all = np.concatenate([y_all, y_shadow])
            reg_all = reg_all + ["neutral"] * len(X_shadow)
            sample_weights = np.concatenate([sample_weights, w_shadow])
            logger.info(f"[ML] Blended {len(X_shadow)} shadow samples (weight 0.3x)")
    except Exception as e:
        logger.debug(f"[ML] Shadow blend skipped: {e}")

    # Normalize regime labels to lowercase to match training split expectations
    # (bot_engine stores labels as UPPERCASE e.g. "NEUTRAL", training expects "neutral")
    reg_all = [r.lower() for r in reg_all]

    # Fast mode: subsample to reduce memory on Railway.
    #
    # BUGFIX: Previous implementation used rng.choice() to pick a *random*
    # 10K rows, then handed the result to _purged_train_test_split() which
    # uses the LAST 20% as the test set. Because the rows were shuffled,
    # "last 20%" became random 20% — training set contained rows that came
    # chronologically AFTER validation rows. Classic look-ahead leakage.
    #
    # sample_weights also went out of alignment because they were sliced
    # by position AFTER the shuffle (see line ~1049 `sample_weights[:len(X_tr)]`).
    #
    # Fix: take the most recent contiguous 10K rows — preserves temporal order,
    # keeps sample_weights aligned, and is a more relevant training window.
    if fast_mode and len(X_all) > 10000:
        keep = 10000
        X_all = X_all[-keep:]
        y_all = y_all[-keep:]
        reg_all = reg_all[-keep:]
        if sample_weights is not None:
            sample_weights = sample_weights[-keep:]

    if len(X_all) < 50:
        return {"status": "insufficient_data", "samples": len(X_all)}

    # Step 5: Walk-forward CV for accuracy reporting
    _n_tickers = max(1, len(ticker_list))
    wf_splits = _walk_forward_cv(X_all, y_all, n_splits=5,
                                  embargo_days=5, n_tickers_approx=_n_tickers)
    fold_accs = []
    fold_regime_accs = {}
    all_wf_probs = []
    all_wf_labels = []
    for fold_i, (tr_idx, te_idx) in enumerate(wf_splits):
        X_wf_tr, X_wf_te = X_all[tr_idx], X_all[te_idx]
        y_wf_tr, y_wf_te = y_all[tr_idx], y_all[te_idx]
        if len(X_wf_tr) < 50 or len(X_wf_te) < 10:
            continue
        wf_m, wf_s, wf_a = _train_single_lgbm(X_wf_tr, X_wf_te, y_wf_tr, y_wf_te, f"wf_fold{fold_i}")
        if wf_m is not None:
            fold_accs.append(wf_a)
            # Collect raw probs for isotonic calibration
            import pandas as pd
            X_wf_te_sc = wf_s.transform(pd.DataFrame(X_wf_te, columns=FEATURE_COLS))
            if hasattr(wf_m, "feature_name_"):
                wf_probs = wf_m.predict(pd.DataFrame(X_wf_te_sc, columns=FEATURE_COLS))
            elif hasattr(wf_m, "predict_proba"):
                wf_probs = wf_m.predict_proba(X_wf_te_sc)[:, 1]
            else:
                wf_probs = wf_m.predict(X_wf_te_sc)
            all_wf_probs.extend(wf_probs.tolist() if hasattr(wf_probs, 'tolist') else list(wf_probs))
            all_wf_labels.extend(y_wf_te.tolist() if hasattr(y_wf_te, 'tolist') else list(y_wf_te))
            # Per-regime accuracy on this fold
            reg_fold_te = [reg_all[i] if i < len(reg_all) else "neutral" for i in te_idx]
            for reg in ["bull", "bear", "neutral"]:
                r_idx = [j for j, r in enumerate(reg_fold_te) if r == reg]
                if len(r_idx) >= 5:
                    r_preds = (np.array([wf_probs[j] for j in r_idx]) > 0.5).astype(int)
                    r_labels = y_wf_te[r_idx]
                    r_acc = float(np.mean(r_preds == r_labels))
                    fold_regime_accs.setdefault(reg, []).append(r_acc)
    wf_accuracy = float(np.mean(fold_accs)) if fold_accs else 0.5
    logger.info(f"Walk-forward CV: {len(fold_accs)} folds, per-fold acc={[round(a*100,1) for a in fold_accs]}, avg={round(wf_accuracy*100,1)}%")

    # Step 5b: Fit isotonic calibrator from walk-forward OOS predictions
    calibrator = None
    if all_wf_probs and len(all_wf_probs) >= 50 and HAS_SKLEARN:
        try:
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(np.array(all_wf_probs), np.array(all_wf_labels))
            joblib.dump(calibrator, CALIBRATOR_PATH)
            logger.info("Isotonic calibrator fitted and saved")
        except Exception as e:
            logger.warning(f"Isotonic calibration failed: {e}")
            calibrator = None

    # Step 6: Train production model on ALL data
    X_tr, X_te, y_tr, y_te = _purged_train_test_split(
        X_all, y_all, embargo_periods=5, n_tickers_per_date=_n_tickers
    )
    if len(X_tr) < 30 or len(X_te) < 10:
        return {"status": "insufficient_data", "samples": len(X_all)}

    # Pass sample_weights to honor feedback decay weighting
    sw_tr = sample_weights[:len(X_tr)] if sample_weights is not None else None
    global_model, global_scaler, global_acc = _train_single_lgbm(X_tr, X_te, y_tr, y_te, "global", sample_weight=sw_tr)
    if global_model is None:
        return {"status": "failed", "reason": "Model training failed"}

    # Step 7: Train regime-specific models
    regime_models = {}
    regime_accs   = {}
    _embargo_rows = 5 * _n_tickers
    reg_tr = reg_all[:len(X_tr)]
    reg_te = reg_all[len(X_tr)+_embargo_rows:len(X_tr)+_embargo_rows+len(X_te)]
    while len(reg_te) < len(X_te): reg_te.append("neutral")
    reg_te = reg_te[:len(X_te)]

    for reg in ["bull", "bear", "neutral"]:
        idx_tr = [i for i, r in enumerate(reg_tr) if r == reg]
        idx_te = [i for i, r in enumerate(reg_te) if r == reg]
        if len(idx_tr) >= 50 and len(idx_te) >= 10:
            Xr_tr = X_tr[idx_tr]; yr_tr = y_tr[idx_tr]
            Xr_te = X_te[idx_te]; yr_te = y_te[idx_te]
            rm, rs, ra = _train_single_lgbm(Xr_tr, Xr_te, yr_tr, yr_te, reg)
            if rm is not None:
                regime_models[reg] = {"model": rm, "scaler": rs}
                regime_accs[reg]   = round(ra * 100, 1)
    # Average regime accs from walk-forward if available
    wf_regime_accs = {r: round(float(np.mean(accs))*100, 1)
                      for r, accs in fold_regime_accs.items() if accs}

    # Step 8: Feature importance
    importance = {}
    try:
        if HAS_LGB and hasattr(global_model, "feature_importance"):
            import pandas as pd
            imp = global_model.feature_importance(importance_type="gain")
            importance = {FEATURE_COLS[i]: float(imp[i]) for i in range(len(FEATURE_COLS))}
    except Exception: pass

    # Step 9: Save bundle
    bundle = {
        "model":          global_model,
        "scaler":         global_scaler,
        "regime_models":  regime_models,
        "feature_names":  FEATURE_COLS,
        "accuracy":       round(wf_accuracy, 4),
        "regime_accs":    wf_regime_accs or regime_accs,
        "samples":        len(X_all),
        "feedback_trades":len(feedback),
        "data_source":    data_source,
        "timestamp":      datetime.now().isoformat(),
        "feature_importance": importance,
        "label_method":   "triple_barrier_volatility_adjusted_timeout_relabel",
        "cv_method":      "walk_forward_5fold_purged",
        "architecture":   "regime_conditional_ensemble",
        "calibrator_path": CALIBRATOR_PATH if calibrator else None,
    }
    if HAS_SKLEARN:
        joblib.dump(bundle, MODEL_PATH)

    # Invalidate model cache so next ml_score() picks up the fresh model
    _model_cache["bundle"] = None
    _model_cache["mtime"] = 0.0

    # Save counts before releasing training data
    _n_samples = len(X_all)
    _n_feedback = len(feedback)

    # Release training data from memory
    del X_all, y_all, X_tr, X_te, y_tr, y_te
    try:
        del X_gen
    except NameError:
        pass
    try:
        del X_fb
    except NameError:
        pass
    import gc; gc.collect()

    return {
        "status":         "trained",
        "accuracy":       round(wf_accuracy * 100, 1),
        "regime_accs":    wf_regime_accs or regime_accs,
        "features":       len(FEATURE_COLS),
        "samples":        _n_samples,
        "feedback_trades":_n_feedback,
        "data_source":    data_source,
        "label_method":   "triple_barrier_timeout_relabel",
        "cv_method":      "walk_forward_5fold",
        "regime_models":  list(regime_models.keys()),
        "wf_fold_accs":   [round(a*100, 1) for a in fold_accs],
        "calibrated":     calibrator is not None,
        "training_seconds": round(time.time() - t0, 1),
    }


# ══════════════════════════════════════════════════════════════════
# INFERENCE — regime-conditional blend
# ══════════════════════════════════════════════════════════════════

def ml_score(features_dict: dict) -> dict:
    """
    Score a stock using the regime-conditional ensemble.

    Logic:
      1. Detect current regime from vxx_ratio + spy_vs_ma50
      2. If a regime-specific model exists, use it (most accurate)
      3. Fall back to global model
      4. Fall back to rules-based if no model loaded

    Returns: {ml_score, ml_confidence, ml_signal, model_type, win_probability}
    """
    try:
        if not os.path.exists(MODEL_PATH):
            return _rule_based_fallback(features_dict)

        age_hours = (time.time() - os.path.getmtime(MODEL_PATH)) / 3600
        if age_hours > BASE_CONFIG.get("ML_MAX_AGE_HOURS", 26):
            return _rule_based_fallback(features_dict)

        if not HAS_SKLEARN:
            return _rule_based_fallback(features_dict)

        # Use cached model — only reload if file changed on disk
        # Thread-safe: two parallel workers can't both trigger a reload
        model_mtime = os.path.getmtime(MODEL_PATH)
        with _model_cache_lock:
            if _model_cache["bundle"] is None or _model_cache["mtime"] != model_mtime:
                _model_cache["bundle"] = joblib.load(MODEL_PATH)
                _model_cache["mtime"] = model_mtime
            bundle = _model_cache["bundle"]
        if bundle is None:
            return _rule_based_fallback(features_dict)

        # Detect current regime — unified function (matches training thresholds)
        vxx_r   = float(features_dict.get("vxx_ratio", 1.0) or 1.0)
        spy_m   = float(features_dict.get("spy_vs_ma50", 1.0) or 1.0)
        current_regime = _classify_regime(vxx_r, spy_m)

        # Build feature row (34 features)
        row = [float(features_dict.get(col, 0) or 0) for col in FEATURE_COLS]
        X   = np.array([row], dtype=np.float32)

        # Try regime-specific model first
        regime_models = bundle.get("regime_models", {})
        used_regime   = "global"
        model_type    = "lightgbm_global"

        def _get_prob(mdl, scaler_obj, X_raw):
            """Get win probability from any model type. Always returns float 0-1."""
            import pandas as pd
            X_sc = scaler_obj.transform(pd.DataFrame(X_raw, columns=FEATURE_COLS))
            # LightGBM Booster: has feature_name_() method, .predict() returns probs
            if hasattr(mdl, "feature_name_"):
                # LGB Booster — predict returns probabilities directly
                p = mdl.predict(pd.DataFrame(X_sc, columns=FEATURE_COLS))
                return float(p[0])
            # sklearn-style: prefer predict_proba (returns actual probabilities)
            if hasattr(mdl, "predict_proba"):
                return float(mdl.predict_proba(X_sc)[0][1])
            # Last resort: predict returns class labels or proba depending on model
            p = mdl.predict(X_sc)
            return float(p[0])

        if current_regime in regime_models:
            rm = regime_models[current_regime]
            try:
                prob = _get_prob(rm["model"], rm["scaler"], X)
                used_regime = current_regime
                model_type  = f"lightgbm_{current_regime}"
            except Exception:
                prob = None
        else:
            prob = None

        # Fall back to global model
        if prob is None:
            try:
                prob = _get_prob(bundle["model"], bundle["scaler"], X)
            except Exception:
                return _rule_based_fallback(features_dict)

        # Clamp to valid range
        prob = max(0.10, min(0.90, prob))

        # Isotonic calibration: replace ad-hoc sigmoid stretch with proper calibrator
        calibrator_path = bundle.get("calibrator_path")
        if calibrator_path and os.path.exists(calibrator_path):
            try:
                cal_mtime = os.path.getmtime(calibrator_path)
                with _model_cache_lock:
                    if _model_cache["calibrator"] is None or _model_cache["cal_mtime"] != cal_mtime:
                        _model_cache["calibrator"] = joblib.load(calibrator_path)
                        _model_cache["cal_mtime"] = cal_mtime
                    cal = _model_cache["calibrator"]
                prob = float(cal.transform([prob])[0])
                prob = max(0.10, min(0.90, prob))
            except Exception:
                pass  # fall through with raw prob

        signal     = "BUY" if prob > 0.60 else ("SELL" if prob < 0.40 else "HOLD")
        confidence = abs(prob - 0.5) * 2

        return {
            "ml_score":        round(prob * 100, 1),
            "ml_confidence":   round(confidence, 3),
            "ml_signal":       signal,
            "model_type":      model_type,
            "win_probability": round(prob, 3),
            "regime_used":     used_regime,
        }

    except Exception:
        return _rule_based_fallback(features_dict)


def _rule_based_fallback(features: dict) -> dict:
    """Rules-based fallback when model not available."""
    mom   = float(features.get("momentum_1m", 0) or 0)
    rsi   = float(features.get("rsi_14", 50) or 50)
    vol   = float(features.get("volume_ratio", 1) or 1)
    reg   = float(features.get("regime_score", 50) or 50)
    idio  = float(features.get("idiosyncratic_ret", 0) or 0)

    score = 50.0
    score += min(mom * 1.5, 15)
    score += min((vol - 1) * 8, 10)
    score += (reg - 50) * 0.2
    score += min(idio * 0.5, 8)
    if rsi < 35:  score += 8
    elif rsi > 70: score -= 5

    score = max(0.0, min(100.0, score))
    prob  = score / 100

    return {
        "ml_score":        round(score, 1),
        "ml_confidence":   round(abs(prob - 0.5) * 2, 3),
        "ml_signal":       "BUY" if score >= 62 else ("SELL" if score <= 38 else "HOLD"),
        "model_type":      "fallback_rules",
        "win_probability": round(prob, 3),
        "regime_used":     "fallback",
    }


# ══════════════════════════════════════════════════════════════════
# META-LABELING MODEL (activated after 50 real trades)
# ══════════════════════════════════════════════════════════════════

def train_meta_model() -> dict:
    """
    Train a secondary model to filter false positives from the primary model.

    Meta-labeling per López de Prado: trains on P(correct | primary said BUY).
    Only uses trades where primary model scored above threshold.

    Features for meta model (5):
      - primary_score: what the primary model scored this trade
      - regime_score: market regime at entry
      - iv_rank: IV environment
      - vxx_ratio: fear level
      - regime_stability: how consistent the regime was over last 10 days
    """
    if not HAS_SKLEARN:
        return {"status": "skipped", "reason": "sklearn not available"}

    feedback = _load_trade_feedback()

    # Filter to primary-positive trades only (meta-labeling trains on BUY signals)
    primary_threshold = 55.0  # min score for a "BUY" signal
    meta_trades = [t for t in feedback
                   if float(t.get("score", t.get("entry_features", {}).get("regime_score", 0)) or 0)
                   >= primary_threshold]

    MIN_META_TRADES = 100  # increased from 50
    if len(meta_trades) < MIN_META_TRADES:
        return {
            "status": "skipped",
            "reason": f"Need {MIN_META_TRADES}+ primary-positive trades, have {len(meta_trades)}",
            "trades": len(meta_trades),
        }

    t0 = time.time()
    META_FEATURES = ["primary_score", "regime_score", "iv_rank_proxy",
                     "vxx_ratio", "regime_stability"]

    X_rows, y_labels = [], []
    for i, trade in enumerate(meta_trades):
        feats = trade.get("entry_features", {})
        if not feats: continue
        primary_sc = float(trade.get("score", feats.get("regime_score", 50)) or 50)
        regime_sc  = float(feats.get("regime_score", 50) or 50)
        iv_rank    = float(feats.get("iv_rank_proxy", 50) or 50)
        vxx_r      = float(feats.get("vxx_ratio", 1.0) or 1.0)
        # regime_stability: how consistent the regime was over last 10 trades
        recent_trades = meta_trades[max(0, i-10):i]
        if recent_trades:
            current_regime = feats.get("regime_at_entry", "neutral")
            if isinstance(current_regime, str):
                current_regime = current_regime.lower()
            regime_stability = sum(
                1 for t in recent_trades
                if (t.get("entry_features", {}).get("regime_at_entry", "neutral") or "neutral").lower() == current_regime
            ) / len(recent_trades)
        else:
            regime_stability = 0.5
        pnl = trade.get("pnl_pct", 0) or 0
        label = 1 if pnl > 0 else 0
        X_rows.append([primary_sc, regime_sc, iv_rank, vxx_r, regime_stability])
        y_labels.append(label)

    if len(X_rows) < 50:
        return {"status": "insufficient_data"}

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int8)

    # Walk-forward CV for meta model (3 folds for smaller dataset)
    meta_splits = _walk_forward_cv(X, y, n_splits=3, embargo_days=3, n_tickers_approx=1)
    meta_fold_accs = []
    for tr_idx, te_idx in meta_splits:
        if len(tr_idx) < 30 or len(te_idx) < 10:
            continue
        X_wf_tr, X_wf_te = X[tr_idx], X[te_idx]
        y_wf_tr, y_wf_te = y[tr_idx], y[te_idx]
        sc = StandardScaler()
        X_s_tr = sc.fit_transform(X_wf_tr)
        X_s_te = sc.transform(X_wf_te)
        if HAS_LGB:
            try:
                dtrain = lgb.Dataset(X_s_tr, label=y_wf_tr)
                m = lgb.train({"objective": "binary", "metric": "binary_logloss",
                               "num_leaves": 8, "learning_rate": 0.05, "verbose": -1},
                              dtrain, num_boost_round=100)
                probs = m.predict(X_s_te)
                acc = float(np.mean((probs > 0.5).astype(int) == y_wf_te))
            except Exception:
                from sklearn.linear_model import LogisticRegression
                m = LogisticRegression(C=1.0); m.fit(X_s_tr, y_wf_tr)
                acc = float(m.score(X_s_te, y_wf_te))
        else:
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(C=1.0); m.fit(X_s_tr, y_wf_tr)
            acc = float(m.score(X_s_te, y_wf_te))
        meta_fold_accs.append(acc)

    # Train production meta model on full data
    X_tr, X_te, y_tr, y_te = _purged_train_test_split(X, y, embargo_periods=2)
    if len(X_tr) < 20: return {"status": "insufficient_data"}

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr); X_te_sc = scaler.transform(X_te)
    if HAS_LGB:
        params = {"objective": "binary", "metric": "binary_logloss",
                  "num_leaves": 8, "learning_rate": 0.05, "verbose": -1}
        try:
            import pandas as pd
            dtrain = lgb.Dataset(X_tr_sc, label=y_tr)
            model  = lgb.train(params, dtrain, num_boost_round=100)
            probs  = model.predict(X_te_sc)
            acc    = float(np.mean((probs > 0.5).astype(int) == y_te))
        except Exception:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=1.0); model.fit(X_tr_sc, y_tr)
            acc   = float(model.score(X_te_sc, y_te))
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1.0); model.fit(X_tr_sc, y_tr)
        acc   = float(model.score(X_te_sc, y_te))

    wf_acc = float(np.mean(meta_fold_accs)) if meta_fold_accs else acc

    joblib.dump({"model": model, "scaler": scaler, "features": META_FEATURES,
                 "accuracy": wf_acc, "trades_used": len(meta_trades),
                 "cv_method": "walk_forward_3fold"}, META_MODEL_PATH)
    return {
        "status": "trained",
        "meta_accuracy": round(wf_acc * 100, 1),
        "trades_used": len(meta_trades),
        "primary_positive_only": True,
        "wf_fold_accs": [round(a * 100, 1) for a in meta_fold_accs],
        "training_seconds": round(time.time() - t0, 1),
    }


def meta_score(primary_score: float, features: dict,
               regime_stability: float = 0.5) -> float:
    """
    Return meta-model probability of profit given primary said BUY.
    Returns 0.5 (neutral) if meta model not yet trained.
    """
    try:
        if not os.path.exists(META_MODEL_PATH): return 0.5
        if not HAS_SKLEARN: return 0.5
        bundle = joblib.load(META_MODEL_PATH)
        X = np.array([[
            primary_score,
            float(features.get("regime_score", 50) or 50),
            float(features.get("iv_rank_proxy", 50) or 50),
            float(features.get("vxx_ratio", 1.0) or 1.0),
            regime_stability,
        ]], dtype=np.float32)
        X_sc = bundle["scaler"].transform(X)
        if hasattr(bundle["model"], "predict_proba"):
            return float(bundle["model"].predict_proba(X_sc)[0][1])
        probs = bundle["model"].predict(X_sc)
        return float(probs[0]) if len(probs) > 0 else 0.5
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════
# OPTIONS ML — same architecture improvements applied
# ══════════════════════════════════════════════════════════════════

OPTIONS_FEATURE_COLS = FEATURE_COLS + [
    "iv_rank",       # actual IV rank (not proxy)
    "days_to_earn",  # days to earnings
    "vrp_magnitude", # abs(VRP)
]


def _build_options_synthetic_training():
    """
    Build synthetic options training data from VXX/SPY history.
    Uses volatility-adjusted labels (same improvement as stock model).
    """
    try:
        r = requests.get(f"{DATA_URL}/v2/stocks/bars",
            params={"symbols":"VXX","timeframe":"1Day","start":"2020-01-01",
                    "limit":2000,"feed":"sip"},
            headers=_h(), timeout=15)
        vxx_bars = r.json().get("bars",{}).get("VXX",[])
        r2 = requests.get(f"{DATA_URL}/v2/stocks/bars",
            params={"symbols":"SPY","timeframe":"1Day","start":"2020-01-01",
                    "limit":2000,"feed":"sip"},
            headers=_h(), timeout=15)
        spy_bars = r2.json().get("bars",{}).get("SPY",[])

        if len(vxx_bars) < 60 or len(spy_bars) < 60: return None, None

        vxx_closes = [float(b["c"]) for b in vxx_bars]
        spy_closes = [float(b["c"]) for b in spy_bars]
        n = min(len(vxx_closes), len(spy_closes))
        X_rows, y_labels = [], []

        for i in range(50, n - 10):
            vx = vxx_closes[i]; sp = spy_closes[i]
            vxx30 = vxx_closes[max(0,i-30):i]; avg30 = sum(vxx30)/len(vxx30)
            vxx_r = vx/avg30 if avg30>0 else 1.0
            vxx52 = vxx_closes[max(0,i-252):i]
            iv_rank = (vx-min(vxx52))/(max(vxx52)-min(vxx52))*100 if vxx52 and max(vxx52)>min(vxx52) else 50
            sh50 = spy_closes[max(0,i-50):i]; ma50 = sum(sh50)/len(sh50) if sh50 else sp
            spy_sm = sp/ma50 if ma50>0 else 1.0
            spy_1m_ago = spy_closes[max(0,i-21)]
            mom_1m = (sp-spy_1m_ago)/spy_1m_ago*100 if spy_1m_ago>0 else 0
            vrp = max(-20,min(20,(vxx_r-1.0)*50))
            regime_score = max(0,min(100,(spy_sm-0.94)/0.12*50+(1.3-vxx_r)/0.6*30+20))

            # Forward outcomes
            spy_f5  = spy_closes[min(i+5,n-1)]; spy_f10 = spy_closes[min(i+10,n-1)]
            vxx_f5  = vxx_closes[min(i+5,n-1)]
            spy_r5  = (spy_f5-sp)/sp*100; spy_r10=(spy_f10-sp)/sp*100
            vxx_r5  = (vxx_f5-vx)/vx*100

            # Vol-of-vol (new feature)
            vxx_10r = [(vxx_closes[j]-vxx_closes[j-1])/vxx_closes[j-1]
                        for j in range(max(1,i-10),i) if vxx_closes[j-1]>0]
            vol_of_vol = float(np.std(vxx_10r)*100) if len(vxx_10r)>=5 else 2.0

            base = {c:0.0 for c in OPTIONS_FEATURE_COLS}
            base.update({
                "vxx_ratio":vxx_r,"iv_rank_proxy":iv_rank,"iv_rank":iv_rank,
                "spy_vs_ma50":spy_sm,"vrp":vrp,"vrp_magnitude":abs(vrp),
                "regime_score":regime_score,"ewma_vol":vx/10,"momentum_1m":mom_1m,
                "vol_of_vol":vol_of_vol,
            })
            row = [float(base.get(col,0) or 0) for col in OPTIONS_FEATURE_COLS]

            # Volatility-adjusted label for options:
            # SPY ATR proxy = vxx_close / 15 (rough: VXX 15 = ~1% daily SPY vol)
            spy_atr_pct = max(0.5, vx / 15)

            if vxx_r >= 1.30 and spy_sm >= 0.94:
                # Panic put sale: vol-adjusted — success if SPY stays within 2×ATR
                threshold = spy_atr_pct * 2
                vxx_still = vxx_f5 > vx * 1.10
                label = 1 if (spy_r10 > -threshold and not vxx_still) else 0
                X_rows.append(row); y_labels.append(label)

            if iv_rank > 70:
                # High-IV sale: success if stock/market stays contained (premium keeps)
                # Timeout with small move = WIN for premium seller (option decays)
                # TIGHTER threshold to balance labels (was 1.5x ATR → too many wins)
                win = (abs(spy_r5) < spy_atr_pct * 1.0)  # within 1.0×ATR (stricter)
                label = 1 if win else 0
                # Downsample wins to avoid class imbalance (win rate ~50-55% target)
                if label == 1 and (i % 3 != 0):  # keep ~1/3 of wins
                    pass  # skip this sample
                else:
                    X_rows.append(row); y_labels.append(label)

            if iv_rank < 20:
                # Low-IV buy: success if market moves > 0.8×ATR either direction
                # Lower threshold for cheap straddles (entry cost is low)
                label = 1 if abs(spy_r5) > spy_atr_pct * 0.8 else 0
                # Low-IV moves are rarer, keep all samples
                X_rows.append(row); y_labels.append(label)

            if 20 <= iv_rank <= 70 and i % 5 == 0:
                # Mid-IV neutral: success = premium decays without big move
                # Added to create more balanced class distribution
                neutral_win = (abs(spy_r5) < spy_atr_pct * 1.2 and
                               abs(spy_r10) < spy_atr_pct * 1.8)
                label = 1 if neutral_win else 0
                X_rows.append(row); y_labels.append(label)

            if 50 < iv_rank < 85 and i % 10 == 0:
                row_earn = list(row)
                earn_idx = OPTIONS_FEATURE_COLS.index("days_to_earn")
                row_earn[earn_idx] = 2.0
                threshold = spy_atr_pct * 1.5
                label = 1 if (vxx_r5 < 20.0 and spy_r10 > -threshold) else 0
                X_rows.append(row_earn); y_labels.append(label)

        if len(X_rows) < 50: return None, None
        X_arr = np.array(X_rows, dtype=np.float32)
        y_arr = np.array(y_labels, dtype=np.int8)
        # Final balance check: if still >65% one class, force balance via undersampling
        pos_count = int(np.sum(y_arr == 1))
        neg_count = int(np.sum(y_arr == 0))
        if pos_count > 0 and neg_count > 0:
            ratio = max(pos_count, neg_count) / min(pos_count, neg_count)
            if ratio > 2.5:  # more than 2.5:1 imbalance
                minority = 1 if pos_count < neg_count else 0
                majority = 1 - minority
                min_idx = np.where(y_arr == minority)[0]
                maj_idx = np.where(y_arr == majority)[0]
                # Downsample majority to 2.0:1 ratio
                target_maj = min(len(maj_idx), len(min_idx) * 2)
                rng = np.random.default_rng(42)
                keep_maj = rng.choice(maj_idx, size=target_maj, replace=False)
                keep_all = np.sort(np.concatenate([min_idx, keep_maj]))
                X_arr, y_arr = X_arr[keep_all], y_arr[keep_all]
        return X_arr, y_arr
    except Exception:
        return None, None


def _build_options_feedback_training(trades: list):
    X_rows, y_labels = [], []
    for trade in trades:
        if trade.get("instrument") != "options" and not trade.get("use_options"): continue
        features = trade.get("entry_features", {})
        if not features: continue
        pnl = trade.get("pnl_pct", 0) or 0
        # Volatility-adjusted win threshold for options
        atr = features.get("atr_pct", 2.0) or 2.0
        win_threshold = max(2.0, atr * 1.0)
        label = 1 if pnl >= win_threshold else 0
        row = [float(features.get(col, 0) or 0) for col in OPTIONS_FEATURE_COLS]
        X_rows.append(row); y_labels.append(label)
    if len(X_rows) < 15: return None, None
    return np.array(X_rows, dtype=np.float32), np.array(y_labels, dtype=np.int8)


def options_ml_score(features: dict) -> float:
    """Score an options setup. Returns 0.0-1.0 probability of profit."""
    try:
        if os.path.exists(OPTIONS_MODEL_PATH):
            age = time.time() - os.path.getmtime(OPTIONS_MODEL_PATH)
            if age < 7 * 86400 and HAS_SKLEARN:
                bundle = joblib.load(OPTIONS_MODEL_PATH)
                m, sc = bundle.get("model"), bundle.get("scaler")
                feats_list = bundle.get("features", OPTIONS_FEATURE_COLS)
                if m and sc:
                    row = [float(features.get(col, 0) or 0) for col in feats_list]
                    X = np.array([row], dtype=np.float32)
                    try:
                        import pandas as pd
                        X_df = pd.DataFrame(X, columns=feats_list)
                        X_sc = sc.transform(X_df)
                        if hasattr(m, "predict_proba"):
                            raw = float(m.predict_proba(X_sc)[0][1])
                        elif hasattr(m, "predict"):
                            # lgb Booster uses .predict() which returns probabilities directly
                            raw = float(m.predict(X_sc)[0])
                        else:
                            raw = 0.5
                        raw = max(0.10, min(0.90, raw))
                        # Isotonic calibration (same as stock model)
                        if os.path.exists(CALIBRATOR_PATH):
                            try:
                                cal = joblib.load(CALIBRATOR_PATH)
                                raw = float(cal.transform([raw])[0])
                                raw = max(0.10, min(0.90, raw))
                            except Exception:
                                pass
                        return round(raw, 4)
                    except Exception:
                        pass
    except Exception:
        pass

    # Rules fallback
    iv_rank   = float(features.get("iv_rank", features.get("iv_rank_proxy", 50)) or 50)
    vxx_ratio = float(features.get("vxx_ratio", 1.0) or 1.0)
    days_earn = float(features.get("days_to_earn", 99) or 99)
    vov       = float(features.get("vol_of_vol", 2.0) or 2.0)

    # Reduce confidence when vol-of-vol is high (uncertain environment)
    vov_penalty = max(0, (vov - 3.0) * 0.02)

    if vxx_ratio >= 1.30:      return max(0.40, 0.70 - vov_penalty)
    if 1 <= days_earn <= 7 and iv_rank > 60: return max(0.40, 0.67 - vov_penalty)
    if iv_rank > 70:           return max(0.40, 0.65 - vov_penalty)
    if iv_rank < 20:           return max(0.40, 0.55 - vov_penalty)
    return 0.50


def train_options_model() -> dict:
    """Train the options ML model with the same pro-architecture improvements."""
    t0 = time.time()
    if not HAS_SKLEARN:
        return {"status": "skipped", "reason": "sklearn not available"}

    feedback = []
    try:
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH) as f:
                all_trades = json.load(f)
            feedback = [t for t in all_trades
                        if t.get("instrument")=="options" or t.get("use_options")]
    except Exception: pass

    X_fb, y_fb = None, None
    if len(feedback) >= 15:
        X_fb, y_fb = _build_options_feedback_training(feedback)

    X_syn, y_syn = _build_options_synthetic_training()

    if X_fb is not None and X_syn is not None:
        X_all = np.vstack([X_syn, X_fb, X_fb, X_fb])
        y_all = np.concatenate([y_syn, y_fb, y_fb, y_fb])
        data_src = f"synthetic({len(X_syn)})+real_3x({len(feedback)})"
    elif X_syn is not None:
        X_all, y_all = X_syn, y_syn
        data_src = f"synthetic_only({len(X_syn)})"
    elif X_fb is not None:
        X_all, y_all = X_fb, y_fb
        data_src = f"real_only({len(feedback)})"
    else:
        return {"status": "failed", "reason": "No training data"}

    if len(X_all) < 50:
        return {"status": "insufficient_data", "samples": len(X_all)}

    # Purged split for options too
    X_tr, X_te, y_tr, y_te = _purged_train_test_split(X_all, y_all, embargo_periods=5)
    if len(X_tr) < 30:
        return {"status": "insufficient_data"}

    # For options synthetic data: the temporal split may create class imbalance
    # between train/test since market regimes shift. Balance TRAINING set only.
    pos_tr = int(np.sum(y_tr == 1)); neg_tr = int(np.sum(y_tr == 0))
    if pos_tr > 0 and neg_tr > 0 and max(pos_tr, neg_tr) / min(pos_tr, neg_tr) > 1.8:
        rng = np.random.default_rng(42)
        min_cls = 1 if pos_tr < neg_tr else 0; maj_cls = 1 - min_cls
        min_idx = np.where(y_tr == min_cls)[0]; maj_idx = np.where(y_tr == maj_cls)[0]
        keep_maj = rng.choice(maj_idx, size=len(min_idx), replace=False)
        keep_all = np.sort(np.concatenate([min_idx, keep_maj]))
        X_tr, y_tr = X_tr[keep_all], y_tr[keep_all]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr); X_te_sc = scaler.transform(X_te)

    try:
        import pandas as pd
        if HAS_LGB:
            # Use scale_pos_weight for better balance (same fix as stock model)
            n_pos = int(np.sum(y_tr == 1)); n_neg = int(np.sum(y_tr == 0))
            spw = max(1.0, n_neg / max(n_pos, 1))
            # Use lgb.train (not LGBMClassifier) for early stopping + spread check
            opt_params = {
                "objective": "binary", "metric": "binary_logloss",
                "learning_rate": 0.04, "num_leaves": 20,
                "min_child_samples": 10, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 5,
                "verbose": -1, "scale_pos_weight": spw,
                "min_gain_to_split": 0.005,
            }
            opt_feat_names = [f"f{i}" for i in range(X_tr_sc.shape[1])]  # neutral names
            dtrain = lgb.Dataset(X_tr_sc, label=y_tr)
            dtest  = lgb.Dataset(X_te_sc, label=y_te, reference=dtrain)
            model = lgb.train(opt_params, dtrain, num_boost_round=250,
                              valid_sets=[dtest],
                              callbacks=[lgb.early_stopping(25, verbose=False),
                                         lgb.log_evaluation(period=-1)])
            probs = model.predict(X_te_sc)
            spread = float(np.percentile(probs, 90) - np.percentile(probs, 10))
            if spread < 0.08:
                raise ValueError(f"LGB spread too narrow: {spread:.3f}")
            preds = (probs > 0.5).astype(int)
            acc = float(np.mean(preds == y_te))
        else:
            raise ValueError("LGB not available")
    except Exception:
        # Fallback to GradientBoosting which spreads better on small data
        model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.06, subsample=0.8,
                                            min_samples_leaf=8, random_state=42)
        model.fit(X_tr_sc, y_tr)
        probs = model.predict_proba(X_te_sc)[:, 1]
        preds = (probs > 0.5).astype(int)
        acc = float(np.mean(preds == y_te))

    # Use ROC-AUC as primary metric (handles imbalanced test sets correctly)
    # Accuracy on imbalanced sets is misleading: 32% accuracy can still mean good ranking
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_te)) > 1:
            auc = float(roc_auc_score(y_te, probs))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    # Report: use AUC*100 as "accuracy" (50=random, 60=decent, 70=good)
    # This is more meaningful than raw accuracy on imbalanced sets
    reported_acc = round(auc * 100, 1)

    joblib.dump({"model": model, "scaler": scaler, "features": OPTIONS_FEATURE_COLS,
                 "accuracy": auc, "data_source": data_src, "trained_at": time.time(),
                 "label_method": "volatility_adjusted_triple_barrier_balanced",
                 "cv_method": "purged_temporal_stratified"},
                OPTIONS_MODEL_PATH)
    return {
        "status": "trained",
        "options_model_accuracy": reported_acc,  # AUC×100 (50=random, 65=good)
        "options_model_auc": reported_acc,
        "real_options_trades": len(feedback),
        "total_samples": len(X_all),
        "data_source": data_src,
        "label_balance": f"{int(np.sum(y_all==1))}/{int(np.sum(y_all==0))} pos/neg",
        "spread": round(float(np.percentile(probs, 90) - np.percentile(probs, 10)), 3),
        "training_seconds": round(time.time() - t0, 1),
    }


# ══ track_fill — called by bot.ts on every order fill ════════════════════════════
# Saves fill data to the trade feedback file so the self-learning loop
# can use real trade outcomes to improve future predictions.
# Compatible drop-in for the old ml_model.track_fill() signature.
#
# CRITICAL FIX 2026-04-20 (Bug #13): Previously, track_fill only APPENDED new
# records. When a trade closed (sell fill after buy, or buy-to-close after
# short), it added a SECOND record instead of updating the first. The
# "outcome" and "pnl_pct" fields stayed None on the entry record forever.
# In _build_feedback_training_data, pnl_pct defaults to 0 when None, which
# falls through the label logic to label=0 (loss). Result: the model was
# silently learning "every trade I took was a loss", regardless of actual
# outcome. This fix detects exit fills and updates the matching entry record.
#
# THREAD SAFETY: Uses fcntl file locking + atomic rename to prevent corruption
# when bot.ts fires multiple track_fill calls near-simultaneously (rare but
# possible during EOD mass-close events).
_feedback_lock = _threading_ml.Lock()

def _atomic_write_feedback(records: list) -> bool:
    """Write feedback file atomically with POSIX file lock.
    Returns True on success, False on failure."""
    import tempfile
    try:
        try:
            import fcntl
            use_flock = True
        except ImportError:
            use_flock = False

        lock_path = FEEDBACK_PATH + ".lock"
        lock_f = open(lock_path, "a+")
        try:
            if use_flock:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            # Atomic write: temp file + rename
            dirname = os.path.dirname(FEEDBACK_PATH) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dirname, prefix=".feedback.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as tf:
                    json.dump(records, tf, indent=2)
                os.replace(tmp_path, FEEDBACK_PATH)
                return True
            except Exception:
                try: os.unlink(tmp_path)
                except Exception: pass
                return False
        finally:
            try:
                if use_flock:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            except Exception: pass
            lock_f.close()
    except Exception:
        return False


def _is_exit_fill(order_data: dict) -> bool:
    """Detect whether a fill is closing a prior position vs. opening new.

    Exit fills:
      - side="sell" (closing a long equity position)
      - side="buy" with is_close flag (covering a short)
      - exit_context present (bot_engine exit_context dict is passed through)
      - exit_reason present (from options_manager exit logic)
    """
    if order_data.get("exit_context"):
        return True
    if order_data.get("exit_reason"):
        return True
    if order_data.get("is_close"):
        return True
    # Heuristic: sell side usually means exit (but could be short-open).
    # Use the bot_engine.exit_context signal as the primary indicator.
    return False


def _find_entry_record(records: list, ticker: str) -> int:
    """Find the index of the most recent entry record for this ticker that
    still has outcome=None (meaning it hasn't been closed yet).

    Returns -1 if not found.
    Searches backwards for efficiency (entries are usually recent).
    """
    for i in range(len(records) - 1, -1, -1):
        r = records[i]
        if r.get("ticker") != ticker:
            continue
        if r.get("outcome") is not None:
            continue  # already closed
        if r.get("pnl_pct") is not None:
            continue  # already labeled
        return i
    return -1


def track_fill(order_data: dict) -> None:
    """
    Log an order fill for ML self-learning.
    Called by bot.ts after every filled order.

    For ENTRY fills: appends a new record with entry_features, outcome=None.
    For EXIT fills:  finds the matching entry record and updates it with
                     computed outcome + pnl_pct.

    order_data keys:
        ticker, side, qty, expected_price, fill_price,
        time_placed, time_filled, session, volume, score,
        entry_features, exit_context (present on exits), exit_reason, is_close
    """
    try:
        # Validate required fields
        raw_ticker = order_data.get("ticker")
        ticker = str(raw_ticker).strip().upper() if raw_ticker is not None else ""
        if not ticker:
            return
        qty = float(order_data.get("qty", 0) or 0)
        if qty <= 0:
            return

        expected   = float(order_data.get("expected_price", 0) or 0)
        fill_price = float(order_data.get("fill_price", expected) or expected)
        slippage   = abs(fill_price - expected) / expected * 100 if expected > 0 else 0.0

        with _feedback_lock:  # prevent interleaved writes
            # Load current feedback (raw, unfiltered)
            raw_feedback = []
            try:
                if os.path.exists(FEEDBACK_PATH):
                    with open(FEEDBACK_PATH) as _ff:
                        raw_feedback = json.load(_ff)
            except Exception:
                raw_feedback = []

            if _is_exit_fill(order_data):
                # ── EXIT FILL: update the matching entry record ──
                idx = _find_entry_record(raw_feedback, ticker)
                if idx >= 0:
                    entry = raw_feedback[idx]
                    entry_price = float(entry.get("fill_price", 0) or 0)
                    entry_side = entry.get("side", "buy").lower()

                    if entry_price > 0:
                        # Compute pnl_pct based on entry side
                        if entry_side in ("buy", "long"):
                            pnl_pct = (fill_price - entry_price) / entry_price * 100
                        else:  # sell/short entry — short covered with buy
                            pnl_pct = (entry_price - fill_price) / entry_price * 100

                        # Exit context from bot_engine.py gives us the real label
                        exit_ctx = order_data.get("exit_context", {}) or {}
                        # Prefer the bot-reported pnl_pct if available (matches the
                        # bot's own accounting including partial fills and scale-outs)
                        reported_pnl = exit_ctx.get("pnl_pct")
                        if reported_pnl is not None:
                            pnl_pct = float(reported_pnl)

                        # Update the entry record in place
                        entry["outcome"] = "win" if pnl_pct > 0 else ("flat" if pnl_pct == 0 else "loss")
                        entry["pnl_pct"] = round(pnl_pct, 3)
                        try:
                            from risk_kill_switch import record_trade_outcome
                            record_trade_outcome(pnl_pct)
                        except Exception:
                            pass
                        entry["exit_price"] = fill_price
                        entry["exit_time"] = str(order_data.get("time_filled", datetime.now().isoformat()))
                        entry["exit_reason"] = str(order_data.get("exit_reason", exit_ctx.get("exit_reason", "close")))
                        entry["days_held"] = exit_ctx.get("days_held", 0)
                        entry["exit_context"] = exit_ctx
                        # Keep the record with same code_version (don't bump — preserves
                        # training eligibility if MIN_FEEDBACK_VERSION hasn't changed)
                        raw_feedback[idx] = entry
                    # No matching entry found — log the exit as a standalone record
                    # so we don't lose the data entirely
                    else:
                        raw_feedback.append({
                            "ticker": ticker, "side": str(order_data.get("side", "sell")),
                            "qty": qty, "fill_price": fill_price,
                            "time_filled": str(order_data.get("time_filled", datetime.now().isoformat())),
                            "outcome": "orphan_exit", "pnl_pct": None,
                            "note": "Exit fill with no matching open entry",
                            "code_version": "1.0.34",
                        })
                else:
                    # No matching entry — might be manual trade or pre-existing position
                    # Log it anyway so the record exists
                    raw_feedback.append({
                        "ticker": ticker, "side": str(order_data.get("side", "sell")),
                        "qty": qty, "fill_price": fill_price,
                        "time_filled": str(order_data.get("time_filled", datetime.now().isoformat())),
                        "outcome": "orphan_exit", "pnl_pct": None,
                        "note": "Exit fill with no matching open entry",
                        "code_version": "1.0.34",
                    })
            else:
                # ── ENTRY FILL: append new record ──
                record = {
                    "ticker":         ticker,
                    "side":           str(order_data.get("side", "buy")),
                    "qty":            qty,
                    "expected_price": expected,
                    "fill_price":     fill_price,
                    "slippage_pct":   round(slippage, 4),
                    "volume":         float(order_data.get("volume", 0) or 0),
                    "session":        str(order_data.get("session", "regular")),
                    "time_placed":    str(order_data.get("time_placed", datetime.now().isoformat())),
                    "time_filled":    str(order_data.get("time_filled", datetime.now().isoformat())),
                    "score":          float(order_data.get("score", 0) or 0),
                    "entry_features": order_data.get("entry_features", {}),
                    "outcome":        None,   # backfilled on matching exit fill
                    "pnl_pct":        None,
                    "code_version":   "1.0.34",
                }
                raw_feedback.append(record)

            # Rotate: keep last 5000 records
            if len(raw_feedback) > 5000:
                raw_feedback = raw_feedback[-5000:]

            # Write atomically
            _atomic_write_feedback(raw_feedback)
    except Exception as e:
        import logging as _tfl
        _tfl.getLogger("voltrade.ml").debug(f"track_fill error: {e}")
        pass  # Never crash bot.ts for a logging call
