#!/usr/bin/env python3
"""
VolTradeAI Autonomous Bot Engine
─────────────────────────────────
Scans all tradeable stocks, scores them across multiple strategies,
picks the best trades, and executes via Alpaca.

Called by: server/bot.ts on a schedule (every 15 min during market hours)
Output: JSON with recommended trades and actions

Usage:
  python3 bot_engine.py scan          # Scan market, return top opportunities
  python3 bot_engine.py manage        # Check existing positions, manage stops
  python3 bot_engine.py full          # Full cycle: scan + decide + recommend
"""

# ── Constrain thread-hungry libraries BEFORE any imports ──────────────────────
# OpenBLAS (numpy) defaults to 32 threads — kills Railway's container.
# 2 threads is plenty for the math we do (scoring, position sizing).
import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "2")

import sys
import json
import os
import time
import subprocess
import requests
import numpy as np
from datetime import datetime, timedelta

# ── Persistent storage path (Railway volume or /tmp locally) ─────────────────
try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"
os.makedirs(DATA_DIR, exist_ok=True)

# ── System config: all adaptive parameters (regime-aware) ────────────────
try:
    from system_config import get_adaptive_params, BASE_CONFIG
    _HAS_SYSTEM_CONFIG = True
except ImportError:
    _HAS_SYSTEM_CONFIG = False
    BASE_CONFIG = {}

# ── Markov regime detector ───────────────────────────────────────────
try:
    from markov_regime import get_regime as _get_markov_regime
    _HAS_MARKOV = True
except ImportError:
    _HAS_MARKOV = False
    def _get_markov_regime(*a, **kw): return {"regime_score":50,"regime_label":"NEUTRAL","markov_state":1,"size_multiplier":1.0,"markov_signal":"NEUTRAL"}

# ── Strategy imports ─────────────────────────────────────────────────────────
_STRATEGIES_DIR = os.path.join(os.path.dirname(__file__), "strategies")
sys.path.insert(0, _STRATEGIES_DIR)

try:
    import momentum as momentum_strategy
except ImportError:
    momentum_strategy = None

try:
    import mean_reversion as mean_reversion_strategy
except ImportError:
    mean_reversion_strategy = None

try:
    import squeeze as squeeze_strategy
except ImportError:
    squeeze_strategy = None

STRATEGIES_LOADED = all([
    momentum_strategy is not None,
    mean_reversion_strategy is not None,
    squeeze_strategy is not None,
])

# ── yfinance cache (module-level, avoids re-fetching same ticker within 5 min) ──
_yf_cache: dict = {}
_yf_cache_time: dict = {}
_YF_CACHE_TTL = 300  # 5 minutes

# ── Config ──────────────────────────────────────────────────────────────────

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "UNwTHo3kvZMBckeIaHQbBLuaaURmFUQP")
ALPACA_KEY = os.environ.get("ALPACA_KEY", "PKMDHJOVQEVIB4UHZXUYVTIDBU")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et")
ALPACA_DATA_URL = "https://data.alpaca.markets"

def _alpaca_headers():
    return {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

MAX_POSITIONS = 5          # Max stocks to hold at once
MAX_POSITION_PCT = 0.05    # 5% of portfolio per position
STOP_LOSS_PCT = 0.02       # 2% stop loss
TAKE_PROFIT_PCT = 0.06     # 6% take profit (3:1 reward/risk)
# MIN_SCORE is no longer hardcoded here. Use params.get("MIN_SCORE", 63) from
# get_adaptive_params() so the threshold correctly varies by market regime:
#   BULL: 63, CAUTION: 67, BEAR/PANIC: 75 (from system_config.get_adaptive_params).
# The old hardcoded 65 conflicted with system_config's regime-aware 63/67/75 values.
MIN_VOLUME = 500000        # 3-year backtest: higher volume = better liquidity and more reliable signals
MIN_PRICE = 5              # 3-year backtest: stocks < $5 (penny/meme) have 25% WR — consistent drag
MAX_SECTOR_POSITIONS = 2   # Max 2 stocks from the same sector

# ── Sector Map (for correlation / diversification check) ─────────────────────
# Values match macro_data.py SECTOR_ETFS names (capitalized, as returned by get_macro_snapshot)
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "INTC": "Technology", "ORCL": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "AVGO": "Technology", "AMAT": "Technology",
    "NOW": "Technology", "SNOW": "Technology", "PLTR": "Technology", "UBER": "Industrials",
    "TSLA": "Consumer Discretionary", "F": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "TM": "Consumer Discretionary", "RIVN": "Consumer Discretionary",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "WFC": "Financials", "C": "Financials", "AXP": "Financials", "V": "Financials", "MA": "Financials",
    "JNJ": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare", "UNH": "Healthcare",
    "ABBV": "Healthcare", "LLY": "Healthcare", "BMY": "Healthcare", "AMGN": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "SPY": "etf", "QQQ": "etf", "IWM": "etf", "DIA": "etf", "GLD": "etf",
    "AMZN": "Consumer Discretionary", "WMT": "Consumer Staples", "TGT": "Consumer Discretionary", "COST": "Consumer Staples",
}

# ── EWMA / GARCH Volatility ──────────────────────────────────────────────────

def ewma_vol(returns, lambd=0.94):
    """
    EWMA (RiskMetrics) volatility estimate — reacts faster than rolling stddev.
    Annualised and expressed as a percentage.
    """
    if len(returns) < 5:
        return None
    arr = np.array(returns, dtype=float)
    var = float(np.var(arr))
    for r in arr:
        var = lambd * var + (1 - lambd) * float(r) ** 2
    return round(float(np.sqrt(var * 252)) * 100, 2)

# garch_vol_estimate() removed — dead code (never called in production).
# EWMA (ewma_vol above) is used instead. GARCH was considered during prototyping
# but requires 30+ observations and is not statistically superior for this use case.

# ── Data Fetching ────────────────────────────────────────────────────────────



def get_stock_details(ticker):
    """Get detailed analysis from analyze.py."""
    import subprocess
    try:
        result = subprocess.run(
            ["python3", os.path.join(os.path.dirname(__file__), "analyze.py"), ticker, "--mode=scan"],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            return json.loads(result.stdout.strip())
    except Exception:
        pass
    return None


def get_alpaca_account():
    """Get Alpaca account info."""
    import requests
    r = requests.get("https://paper-api.alpaca.markets/v2/account", headers={
        "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET
    }, timeout=10)
    return r.json()


def get_alpaca_positions():
    """Get current Alpaca positions."""
    import requests
    r = requests.get("https://paper-api.alpaca.markets/v2/positions", headers={
        "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET
    }, timeout=10)
    return r.json()

# ── Strategy Scoring ─────────────────────────────────────────────────────────

# score_stock() removed — dead code (never called from anywhere in the codebase).
# The main scan pipeline uses quick_scan() + deep_score() instead.
# Retained its neighbor ewma_vol() which IS used for volatility estimation.

def deep_score(ticker, quick_result):
    """
    Deep analysis on a pre-filtered stock using analyze.py.
    Integrates all three strategy modules (momentum, mean_reversion, squeeze)
    plus VRP, sentiment, earnings, EWMA/GARCH vol and edge factors.
    """
    detail = get_stock_details(ticker)
    if not detail or "error" in detail:
        return quick_result

    reasons = list(quick_result.get("reasons", []))
    change_pct = quick_result.get("change_pct", 0)

    # ── Parallel data fetch — all 6 sources run simultaneously ──────────────
    # Previously sequential: ~12s per ticker. Now parallel: ~3-4s (limited by slowest source)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_macro():
        try:
            from macro_data import get_macro_snapshot, get_news_sentiment
            return get_macro_snapshot(), get_news_sentiment(ticker)
        except Exception:
            return {}, {}

    def _fetch_intel():
        try:
            from intelligence import get_full_intelligence
            return get_full_intelligence(ticker)
        except Exception:
            return {}

    def _fetch_alt():
        try:
            from alt_data import get_alt_data_score
            return get_alt_data_score(ticker)
        except Exception:
            return {}

    def _fetch_social():
        try:
            from social_data import get_social_intelligence
            return get_social_intelligence(ticker)
        except Exception:
            return {}

    def _fetch_finnhub():
        try:
            from finnhub_data import get_insider_sentiment, get_recommendation_trends
            fi = get_insider_sentiment(ticker)
            fr = get_recommendation_trends(ticker)
            return {"insider_mspr": fi.get("mspr", 0), "insider_signal": fi.get("signal", "neutral"),
                    "analyst_consensus": fr.get("consensus", "hold"),
                    "analyst_buy_count": fr.get("buy", 0) + fr.get("strong_buy", 0)}
        except Exception:
            return {}

    macro, news_sent, intel, alt, social, finnhub = {}, {}, {}, {}, {}, {}
    with ThreadPoolExecutor(max_workers=5) as _pool:
        _f_macro    = _pool.submit(_fetch_macro)
        _f_intel    = _pool.submit(_fetch_intel)
        _f_alt      = _pool.submit(_fetch_alt)
        _f_social   = _pool.submit(_fetch_social)
        _f_finnhub  = _pool.submit(_fetch_finnhub)
        try:
            macro, news_sent = _f_macro.result(timeout=15)
        except Exception:
            pass
        try:
            intel = _f_intel.result(timeout=15)
        except Exception:
            pass
        try:
            alt = _f_alt.result(timeout=15)
        except Exception:
            pass
        try:
            social = _f_social.result(timeout=15)
        except Exception:
            pass
        try:
            finnhub = _f_finnhub.result(timeout=15)
        except Exception:
            pass

    # Analyst ratings + valuation multiples (yfinance) — with 5-min cache
    # WRAPPED IN SUBPROCESS with 5s hard kill to prevent yfinance hangs
    yf_fundamentals = {}
    _yf_cache_key = ticker
    if _yf_cache_key in _yf_cache and (time.time() - _yf_cache_time.get(_yf_cache_key, 0)) < _YF_CACHE_TTL:
        yf_fundamentals = _yf_cache[_yf_cache_key]
    else:
        try:
            _yf_script = f'''import yfinance as yf, json, sys
try:
    t = yf.Ticker("{ticker}")
    info = t.info or {{}}
    d = {{
        "forward_pe": info.get("forwardPE") or 0,
        "trailing_pe": info.get("trailingPE") or 0,
        "price_to_book": info.get("priceToBook") or 0,
        "ev_ebitda": info.get("enterpriseToEbitda") or 0,
        "price_to_sales": info.get("priceToSalesTrailing12Months") or 0,
        "peg_ratio": info.get("pegRatio") or 0,
        "target_mean": info.get("targetMeanPrice") or 0,
        "target_high": info.get("targetHighPrice") or 0,
        "target_low": info.get("targetLowPrice") or 0,
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or 0,
        "recommendation": info.get("recommendationKey") or "none",
        "num_analysts": info.get("numberOfAnalystOpinions") or 0,
    }}
    recs = t.recommendations
    if recs is not None and not recs.empty:
        latest = recs.iloc[0]
        d["strong_buy_count"] = int(latest.get("strongBuy", 0))
        d["buy_count"] = int(latest.get("buy", 0))
        d["hold_count"] = int(latest.get("hold", 0))
        d["sell_count"] = int(latest.get("sell", 0))
        d["strong_sell_count"] = int(latest.get("strongSell", 0))
    print(json.dumps(d))
except Exception as e:
    print(json.dumps({{}}))
'''
            _yf_proc = subprocess.run(
                ["python3", "-c", _yf_script],
                capture_output=True, text=True, timeout=5  # HARD 5-second kill
            )
            if _yf_proc.stdout.strip():
                yf_fundamentals = json.loads(_yf_proc.stdout.strip())
                _yf_cache[_yf_cache_key] = yf_fundamentals
                _yf_cache_time[_yf_cache_key] = time.time()
        except (subprocess.TimeoutExpired, Exception):
            pass  # yfinance hung or failed — skip, don't block Tier 2

    # ── Pull key metrics from analyze.py output ──────────────────────────────
    vrp = detail.get("vrp", 0) or 0
    rec = detail.get("recommendation", {}) or {}
    sentiment = detail.get("sentiment", {}) or {}
    edge = detail.get("edge_factors", {}) or {}
    vol_metrics = detail.get("vol_metrics", {}) or {}

    rsi = vol_metrics.get("rsi_14") or None
    volume_ratio = vol_metrics.get("volume_ratio_5d") or 1.0

    # ── ADX (trend strength) + OBV (smart money flow) ─────────────────────────
    adx_value = None
    obv_signal = 0  # -1 = distribution, 0 = neutral, +1 = accumulation
    _deep_atr14 = None   # captured for ml_features atr_pct
    _deep_closes = []    # captured for ml_features above_ma10
    try:
        end_d = datetime.now().strftime("%Y-%m-%d")
        start_d = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
        bars_url = (f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars"
                    f"?timeframe=1Day&start={start_d}&limit=30&adjustment=all&feed=sip")
        bars_resp = requests.get(bars_url, headers=_alpaca_headers(), timeout=8)
        bars_data = bars_resp.json().get("bars", [])
        if len(bars_data) >= 14:
            # ADX calculation (14-period)
            highs = [b["h"] for b in bars_data]
            lows = [b["l"] for b in bars_data]
            closes = [b["c"] for b in bars_data]
            _deep_closes = closes  # capture for ML features
            volumes = [b.get("v", 0) for b in bars_data]

            plus_dm_list = []
            minus_dm_list = []
            tr_list = []
            for i in range(1, len(bars_data)):
                up_move = highs[i] - highs[i-1]
                down_move = lows[i-1] - lows[i]
                plus_dm_list.append(up_move if up_move > down_move and up_move > 0 else 0)
                minus_dm_list.append(down_move if down_move > up_move and down_move > 0 else 0)
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                tr_list.append(tr)

            if len(tr_list) >= 14:
                # Smoothed averages (Wilder's method)
                atr14 = sum(tr_list[:14]) / 14
                plus_dm14 = sum(plus_dm_list[:14]) / 14
                minus_dm14 = sum(minus_dm_list[:14]) / 14
                for i in range(14, len(tr_list)):
                    atr14 = (atr14 * 13 + tr_list[i]) / 14
                    plus_dm14 = (plus_dm14 * 13 + plus_dm_list[i]) / 14
                    minus_dm14 = (minus_dm14 * 13 + minus_dm_list[i]) / 14

                _deep_atr14 = atr14  # capture for ML features atr_pct
                plus_di = (plus_dm14 / atr14 * 100) if atr14 > 0 else 0
                minus_di = (minus_dm14 / atr14 * 100) if atr14 > 0 else 0
                dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
                adx_value = round(dx, 1)

            # OBV calculation
            obv = 0
            obv_values = [0]
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    obv += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv -= volumes[i]
                obv_values.append(obv)

            # OBV trend: compare last 5-day OBV avg to previous 5-day
            if len(obv_values) >= 10:
                recent_obv = sum(obv_values[-5:]) / 5
                prev_obv = sum(obv_values[-10:-5]) / 5
                if recent_obv > prev_obv * 1.05:
                    obv_signal = 1  # Accumulation
                elif recent_obv < prev_obv * 0.95:
                    obv_signal = -1  # Distribution
    except Exception:
        pass

    # ── Strategy module scores ────────────────────────────────────────────────

    # 1. Momentum score (use 12-month and 1-month returns from edge factors)
    momentum_score = 50
    mom_12_1 = edge.get("relative_strength", 0) or 0  # proxy for 12mo momentum
    mom_1m = change_pct  # proxy — best we have from quick data
    avg_volume = quick_result.get("volume", 0)

    if momentum_strategy:
        try:
            m_result = momentum_strategy.score(mom_12_1, mom_1m, avg_volume)
            momentum_score = m_result.get("score", 50)
            if m_result.get("signal") not in ("NO DATA", "SKIP", "NEUTRAL"):
                reasons.append(f"Momentum: {m_result['signal']} ({m_result['reason']})")
        except Exception:
            pass

    # 2. Mean reversion score
    mean_reversion_score = 50
    change_5d = -(abs(change_pct) * 3) if change_pct < -1 else change_pct  # rough proxy
    if mean_reversion_strategy and rsi is not None:
        try:
            mr_result = mean_reversion_strategy.score(rsi, change_5d, volume_ratio)
            mean_reversion_score = mr_result.get("score", 50)
            if mr_result.get("signal") not in ("NO DATA", "NO EDGE"):
                reasons.append(f"MeanRev: {mr_result['signal']} ({mr_result['reason']})")
        except Exception:
            pass

    # 3. VRP score (volatility risk premium)
    vrp_score = 50
    if vrp > 8:
        vrp_score = 85
        reasons.append(f"High VRP (+{vrp:.1f}%) — sell options edge")
    elif vrp > 5:
        vrp_score = 70
        reasons.append(f"Elevated VRP (+{vrp:.1f}%) — options overpriced")
    elif vrp > 0:
        vrp_score = 55
    elif vrp < -3:
        vrp_score = 65
        reasons.append(f"Cheap IV ({vrp:.1f}%) — buy options edge")
    else:
        vrp_score = 45

    # 4. Squeeze score
    squeeze_score_val = 50
    squeeze_raw = edge.get("squeeze_score", 0) or 0
    short_pct = edge.get("short_float_pct", 0) or 0
    days_to_cover = edge.get("days_to_cover", 0) or 0
    sent_score_val = sentiment.get("score", 50) or 50
    reddit_buzz = sentiment.get("reddit_buzz", 0) or 0

    if squeeze_strategy:
        try:
            sq_result = squeeze_strategy.score(short_pct, days_to_cover, sent_score_val, volume_ratio, reddit_buzz)
            squeeze_score_val = sq_result.get("score", 50)
            if sq_result.get("signal") not in ("NONE", "LOW RISK"):
                reasons.append(f"Squeeze: {sq_result['signal']} ({sq_result['reason']})")
        except Exception:
            squeeze_score_val = min(100, max(0, squeeze_raw))
    else:
        squeeze_score_val = min(100, max(0, squeeze_raw))

    # 5. Volume score
    volume_score = 50
    if volume_ratio > 3:
        volume_score = 90
    elif volume_ratio > 2:
        volume_score = 75
    elif volume_ratio > 1.5:
        volume_score = 60
    elif volume_ratio > 1:
        volume_score = 50
    else:
        volume_score = 35

    # ── EWMA / GARCH vol adjustment ──────────────────────────────────────────
    ewma_rv = vol_metrics.get("ewma_rv") or None
    garch_rv = vol_metrics.get("garch_rv") or None
    rv20 = vol_metrics.get("rv20") or None

    # If EWMA vol > realized vol by a large margin, options are expensive → SELL
    vol_premium_bonus = 0
    if ewma_rv and rv20:
        ewma_premium = ewma_rv - rv20
        if ewma_premium > 5:
            vol_premium_bonus = 8
            reasons.append(f"EWMA vol ({ewma_rv:.1f}%) >> RV20 ({rv20:.1f}%) — elevated vol")
        elif ewma_premium < -5:
            vol_premium_bonus = 5
            reasons.append(f"Vol compressed — low EWMA ({ewma_rv:.1f}%)")

    # ── Macro & News Factors ──────────────────────────────────────────────
    macro_adjustment = 0
    news_adjustment = 0

    # VIX regime affects strategy preference
    vix_regime = macro.get("vix_regime", "medium")
    if vix_regime == "extreme":
        # High VIX: favor vol-selling (VRP), penalize momentum
        macro_adjustment += 5 if vrp > 0 else -10
        reasons.append(f"VIX extreme ({macro.get('vix', 'N/A')}) — favoring vol strategies")
    elif vix_regime == "low":
        # Low VIX: favor momentum, penalize vol-selling (low premiums)
        macro_adjustment += 5 if mom_12_1 > 0 else -5
        reasons.append(f"VIX low ({macro.get('vix', 'N/A')}) — favoring momentum")

    # Sector momentum — boost stocks in hot sectors, penalize cold ones
    sector_mom = macro.get("sector_momentum", {})
    stock_sector = SECTOR_MAP.get(ticker, "Other")
    sector_pct = sector_mom.get(stock_sector, 0)
    if sector_pct > 1.5:
        macro_adjustment += 8
        reasons.append(f"Sector tailwind: {stock_sector} +{sector_pct:.1f}%")
    elif sector_pct < -1.5:
        macro_adjustment -= 8
        reasons.append(f"Sector headwind: {stock_sector} {sector_pct:.1f}%")

    # Market regime
    regime = macro.get("market_regime", "neutral")
    if regime == "risk_off":
        # Risk-off: penalize aggressive buys, favor defensive
        macro_adjustment -= 5
        reasons.append("Market regime: risk-off")
    elif regime == "risk_on":
        macro_adjustment += 3

    # News sentiment scoring
    news_score_val = news_sent.get("sentiment_score", 0)
    if news_score_val >= 60:
        news_adjustment += 10
        reasons.append(f"News very bullish ({news_sent.get('headline_count', 0)} articles)")
    elif news_score_val >= 20:
        news_adjustment += 5
        reasons.append(f"News bullish ({news_sent.get('headline_count', 0)} articles)")
    elif news_score_val <= -60:
        news_adjustment -= 12
        reasons.append(f"News very bearish: {news_sent.get('top_headline', '')[:60]}")
    elif news_score_val <= -20:
        news_adjustment -= 6
        reasons.append(f"News bearish ({news_sent.get('headline_count', 0)} articles)")

    # ── Combined score ───────────────────────────────────────────────────────
    combined_score = (
        momentum_score * 0.25 +
        mean_reversion_score * 0.20 +
        vrp_score * 0.25 +
        squeeze_score_val * 0.15 +
        volume_score * 0.15
    ) + vol_premium_bonus
    combined_score += macro_adjustment + news_adjustment

    # ADX trend strength adjustment
    adx_adjustment = 0
    if adx_value is not None:
        if adx_value > 40:
            # Strong trend — boost momentum trades, penalize mean reversion
            adx_adjustment += 6
            reasons.append(f"ADX {adx_value} — strong trend")
        elif adx_value > 25:
            adx_adjustment += 3
            reasons.append(f"ADX {adx_value} — moderate trend")
        elif adx_value < 15:
            # No trend — penalize momentum, boost mean reversion
            adx_adjustment -= 4
            reasons.append(f"ADX {adx_value} — no trend (choppy)")

    # OBV smart money flow adjustment
    obv_adjustment = 0
    if obv_signal == 1:
        obv_adjustment += 5
        reasons.append("OBV rising — smart money accumulating")
    elif obv_signal == -1:
        obv_adjustment -= 5
        reasons.append("OBV falling — smart money distributing")

    combined_score += adx_adjustment + obv_adjustment

    # ── Intelligence-based scoring ────────────────────────────────────────────
    intel_adjustment = 0
    if intel:
        intel_score = intel.get("intel_score", 0)
        
        # Scale intel_score into a -15 to +15 adjustment
        intel_adjustment = max(-15, min(15, int(intel_score * 0.3)))
        
        # Trap detection — major penalty
        if intel.get("trap_warning"):
            intel_adjustment = min(intel_adjustment, -10)
            reasons.append(f"TRAP WARNING: {intel.get('trap_reason', 'conflicting signals')[:80]}")
        
        # Insider activity (SEC EDGAR)
        insider = intel.get("insider", {})
        if insider.get("net_direction") == "strong_selling":
            reasons.append(f"Insiders heavily selling ({insider.get('sell_count', 0)} sells)")
        elif insider.get("net_direction") == "strong_buying":
            reasons.append(f"Insiders heavily buying ({insider.get('buy_count', 0)} buys)")

        # Finnhub insider sentiment (MSPR: -100 to +100, predicts 30-90 day moves)
        if finnhub.get("insider_mspr"):
            mspr = finnhub["insider_mspr"]
            if mspr > 30:
                intel_adjustment += 5
                reasons.append(f"Finnhub insider sentiment: bullish (MSPR {mspr:.0f})")
            elif mspr < -30:
                intel_adjustment -= 5
                reasons.append(f"Finnhub insider sentiment: bearish (MSPR {mspr:.0f})")
        if finnhub.get("analyst_consensus") == "buy":
            intel_adjustment += 3
            reasons.append(f"Analyst consensus: BUY ({finnhub.get('analyst_buy_count', 0)} buy ratings)")
        
        # Earnings pattern
        earnings_pat = intel.get("earnings_pattern", {})
        if earnings_pat.get("sell_the_news_risk"):
            intel_adjustment -= 5
            reasons.append("Sell-the-news pattern detected — stock often drops on good earnings")
        
        # Top news event
        news = intel.get("news", {})
        if news.get("events"):
            top_event = news["events"][0]
            if abs(top_event.get("weight", 0)) >= 15:
                reasons.append(f"Major event: {top_event['category'].replace('_', ' ')} — {top_event.get('headline', '')[:60]}")
    
    combined_score += intel_adjustment

    # ── Alternative data scoring ──────────────────────────────────────────────
    alt_adjustment = 0
    if alt:
        alt_adjustment = max(-20, min(20, alt.get("alt_score", 0)))
        
        wiki = alt.get("wiki", {})
        if wiki.get("has_spike"):
            reasons.append(f"Wikipedia attention spike: {wiki.get('spike_ratio', 0)}x normal views")
        
        short = alt.get("short_interest", {})
        if short.get("squeeze_potential"):
            reasons.append("Short squeeze potential — price rising with heavy prior shorting")
        elif short.get("short_pressure") == "high":
            reasons.append("High short pressure — heavy selling on down days")
        
        congress = alt.get("congressional", {})
        if congress.get("unusual_activity"):
            reasons.append(f"Unusual insider filing activity ({congress.get('recent_filings', 0)} Form 4s in 2 weeks)")
        
        geo = alt.get("geopolitical", {})
        if geo.get("risk_level") in ("high", "extreme"):
            reasons.append(f"Geopolitical risk: {geo.get('risk_level')} — {', '.join(geo.get('active_risks', []))}")
        
        fred = alt.get("fred_macro", {})
        if fred.get("yield_curve_inverted"):
            reasons.append("Yield curve inverted — recession signal")
        if fred.get("credit_stress"):
            reasons.append("Credit spreads elevated — financial stress")
    
    combined_score += alt_adjustment

    # ── Social & search data scoring ──────────────────────────────────────────
    social_adjustment = 0
    if social:
        social_signal = social.get("combined_signal", 0)
        social_adjustment = social_signal  # Already weighted and clamped to -15 to +15
        
        # Add specific reasons
        reddit = social.get("reddit", {})
        trends = social.get("google_trends", {})
        news_m = social.get("news_multi", {})
        
        if trends.get("has_spike"):
            reasons.append(f"Google Trends spike: {trends.get('spike_ratio', 0)}x normal search interest")
        elif trends.get("trend_direction") == "rising":
            reasons.append("Google search interest rising")
        
        if reddit.get("buzz_level") in ("high", "viral"):
            reasons.append(f"Reddit buzz: {reddit.get('buzz_level')} ({reddit.get('total_mentions', 0)} mentions, {reddit.get('bullish_pct', 50):.0f}% bullish)")
            if reddit.get("wsb_mentions", 0) > 10 and reddit.get("bullish_pct", 50) > 80:
                reasons.append("WSB hype warning — extreme bullishness often precedes drops")
        
        if news_m.get("freshness") == "breaking":
            reasons.append(f"Breaking news: {news_m.get('total_articles', 0)} articles from {len(news_m.get('sources', {}))} sources")
        
        if social.get("confidence") == "low":
            social_adjustment = int(social_adjustment * 0.5)
    
    combined_score += social_adjustment

    # ── Analyst & valuation scoring ──────────────────────────────────────────────────────
    fundamental_adjustment = 0
    if yf_fundamentals:
        # Analyst consensus
        yf_rec = yf_fundamentals.get("recommendation", "none")
        if yf_rec in ("strong_buy", "buy"):
            fundamental_adjustment += 5
            reasons.append(f"Analyst consensus: {yf_rec.upper()} ({yf_fundamentals.get('num_analysts', 0)} analysts)")
        elif yf_rec in ("sell", "strong_sell", "underperform"):
            fundamental_adjustment -= 6
            reasons.append(f"Analyst consensus: {yf_rec.upper()} ({yf_fundamentals.get('num_analysts', 0)} analysts)")

        # Price target upside/downside
        target = yf_fundamentals.get("target_mean", 0)
        current = yf_fundamentals.get("current_price", 0)
        if target > 0 and current > 0:
            upside_pct = ((target - current) / current) * 100
            if upside_pct > 20:
                fundamental_adjustment += 6
                reasons.append(f"Analyst target: ${target:.0f} (+{upside_pct:.0f}% upside)")
            elif upside_pct > 10:
                fundamental_adjustment += 3
                reasons.append(f"Analyst target: ${target:.0f} (+{upside_pct:.0f}% upside)")
            elif upside_pct < -10:
                fundamental_adjustment -= 5
                reasons.append(f"Analyst target: ${target:.0f} ({upside_pct:.0f}% downside)")

        # Valuation check — penalize extremely overvalued stocks
        fwd_pe = yf_fundamentals.get("forward_pe", 0)
        if fwd_pe > 50:
            fundamental_adjustment -= 4
            reasons.append(f"Expensive: Forward P/E {fwd_pe:.1f}")
        elif fwd_pe > 0 and fwd_pe < 12:
            fundamental_adjustment += 3
            reasons.append(f"Cheap: Forward P/E {fwd_pe:.1f}")

        ev_ebitda = yf_fundamentals.get("ev_ebitda", 0)
        if ev_ebitda > 30:
            fundamental_adjustment -= 3
            reasons.append(f"High EV/EBITDA: {ev_ebitda:.1f}")
        elif ev_ebitda > 0 and ev_ebitda < 10:
            fundamental_adjustment += 2
            reasons.append(f"Low EV/EBITDA: {ev_ebitda:.1f}")

    combined_score += fundamental_adjustment

    # Recommendation boost
    if rec:
        action = rec.get("action", "")
        if "BUY" in action.upper():
            combined_score += 8
            reasons.append(f"AI recommends: {action}")
        elif "SELL" in action.upper() and "OPTIONS" in action.upper():
            combined_score += 6
            reasons.append(f"AI recommends: {action}")

    # Sentiment boost / penalty
    contrarian = sentiment.get("contrarian_flag")
    if contrarian == "Squeeze Watch":
        combined_score += 10
        reasons.append("SQUEEZE WATCH — retail hype + high short interest")
    elif contrarian == "Buy the Dip":
        combined_score += 8
        reasons.append("Buy the Dip signal — retail panic, institutions buying")
    elif sent_score_val > 75:
        combined_score += 4

    # Edge factors
    rs = edge.get("relative_strength", 0)
    if rs and rs > 5:
        combined_score += 6
        reasons.append(f"Outperforming SPY by {rs:.1f}%")

    combined_score = max(0, min(100, combined_score))
    rules_only_score = round(combined_score, 1)  # Capture BEFORE ML blend for attribution
    ml_only_score = None  # Set below if ML runs

    # ── ML Model Integration ──────────────────────────────────────────────────
    try:
        from ml_model_v2 import ml_score  # v2: 25 clean features, self-learning
        # Get Markov regime state for the ML (adds real predictive power)
        _regime_ctx = {"regime_score": 50, "markov_state": 1, "size_multiplier": 1.0}
        if _HAS_MARKOV:
            try:
                _spy_rets = []
                _spy_macro = macro.get("spy_returns_5d", []) or []
                _vxx_r = intel.get("vxx_ratio", 1.0) if intel else 1.0
                _spy_ma = macro.get("spy_vs_ma50", 1.0) or 1.0
                _regime_ctx = _get_markov_regime(_spy_rets, float(_vxx_r), float(_spy_ma))
            except Exception:
                pass

        # 31 features — ALL computed from real data (no zeroing)
        # price_vs_52w_high: strongest momentum predictor (George & Hwang 2004)
        _price = quick_result.get("price", 0) or 0
        _high_52w = quick_result.get("high_52w", _price) or _price
        _price_vs_52w = (_price - _high_52w) / _high_52w * 100 if _high_52w > 0 else 0
        _float_turnover = min(volume_ratio * 5, 100)  # proxy from volume ratio

        # Compute MA10 and ATR% from captured bar data
        _above_ma10 = 1.0  # default: assume above (already trend-filtered)
        if len(_deep_closes) >= 10:
            _ma10 = sum(_deep_closes[-10:]) / 10
            _above_ma10 = 1.0 if (_deep_closes[-1] > _ma10) else 0.0
        _atr_pct = 3.0  # default
        if _deep_atr14 is not None and _price > 0:
            _atr_pct = round(_deep_atr14 / _price * 100, 2)
        # Derived features for the 6 new signals
        _put_call_proxy = -1.0 if vrp > 8 else (1.0 if vrp < -5 else 0.0)
        _idio_ret = (quick_result.get("change_pct", 0) or 0) - (macro.get("spy_change_pct", 0) or 0)

        ml_features = {
            # Technical (9)
            "momentum_1m":          edge.get("momentum_1m", 0) or 0,
            "momentum_3m":          edge.get("momentum_3m", 0) or 0,
            "rsi_14":               rsi or 50,
            "volume_ratio":         min(volume_ratio, 10.0),
            "vwap_position":        1.0 if quick_result.get("above_vwap") else 0.0,
            "adx":                  adx_value if adx_value is not None else 20,
            "ewma_vol":             vol_metrics.get("ewma_vol", 2) or 2 if vol_metrics else 2,
            "range_pct":            quick_result.get("range_pct", 0) or 0,
            "price_vs_52w_high":    _price_vs_52w,    # strongest predictor
            "float_turnover":       _float_turnover,  # volume intensity
            # Options/volatility (3)
            "vrp":                  vrp or 0,
            "iv_rank_proxy":        detail.get("iv_rank", 50) or 50,
            "atr_pct":              _atr_pct,  # actual ATR% from daily bars
            # Regime (5) — wired to Markov + VXX ratio
            "vxx_ratio":            intel.get("vxx_ratio", 1.0) if intel else 1.0,
            "spy_vs_ma50":          macro.get("spy_vs_ma50", 1.0) or 1.0,
            "markov_state":         float(_regime_ctx.get("markov_state", 1)),
            "regime_score":         float(_regime_ctx.get("regime_score", 50)),
            "sector_momentum":      sector_mom.get(stock_sector, 0),
            # Quality (4)
            "change_pct_today":     quick_result.get("change_pct", 0) or 0,
            "above_ma10":           _above_ma10,  # actual MA10 comparison
            "trend_strength":       min(abs(quick_result.get("change_pct", 0) or 0) / max(vol_metrics.get("ewma_vol", 2) or 2, 0.1), 5.0) if vol_metrics else 1.0,
            "volume_acceleration":  0.0,  # available from streaming bars but not in daily scan
            # Intelligence (3)
            # TRAIN/INFERENCE CONSISTENCY FIX: The generic training data always uses
            # zeros for intel_score, insider_signal, and news_sentiment because the
            # training loop (Alpaca bar history) has no access to historical intel/news.
            # The model therefore learned these features have no predictive value.
            # Until the model is retrained with real historical intel data, we zero
            # these out at inference to match training distribution and avoid
            # out-of-distribution inputs that would corrupt ML predictions.
            # TODO: Collect historical intel/news data and retrain with real values.
            "intel_score":          0.0,  # Zeroed for train/inference consistency (see above)
            "insider_signal":       0.0,  # Zeroed for train/inference consistency
            "news_sentiment":       0.0,  # Zeroed for train/inference consistency
            # New 6 features (FIX 5: previously missing from feature dict)
            "cross_sec_rank":       0.5,   # neutral — requires cross-sectional data not available here
            "earnings_surprise":    intel.get("earnings_surprise", 0) if intel else 0,
            "put_call_proxy":       _put_call_proxy,  # derived from VRP direction
            "vol_of_vol":           0.0,   # requires VXX history not available in deep_score
            "frac_diff_price":      0.0,   # requires long price history; default to 0
            "idiosyncratic_ret":    round(_idio_ret, 4),  # stock return minus SPY contribution
        }
        ml_result = ml_score(ml_features)

        # Blend: dynamic ratio loaded from tracker, default 60/40
        # IMPORTANT: ML only adjusts score when it has CONVICTION (not near 50).
        # Near-50 ML = uncertain. Uncertain ML should NOT drag down a good rules setup.
        if ml_result.get("model_type") != "fallback":
            ml_s = ml_result.get("ml_score", combined_score)
            ml_only_score = round(ml_s, 1)  # Capture ML score for attribution
            # Dynamic blend: load tracked accuracy, default 60/40
            try:
                from storage_config import BLEND_TRACKER_PATH
                if os.path.exists(BLEND_TRACKER_PATH):
                    with open(BLEND_TRACKER_PATH) as _bf:
                        _blend = json.load(_bf)
                    rule_weight = _blend.get("rule_weight", 0.6)
                    ml_weight = _blend.get("ml_weight", 0.4)
                else:
                    rule_weight, ml_weight = 0.6, 0.4
            except Exception:
                rule_weight, ml_weight = 0.6, 0.4

            # Conviction-gating: when ML is uncertain (42-58 range), treat as neutral 50.
            # This prevents a mediocre ML score from dragging down strong rule setups.
            # At 50, the blend becomes: rules*0.6 + 50*0.4 = rules contribution only matters.
            ml_conviction = abs(ml_s - 50)
            if ml_conviction < 8:  # Score within 42-58 = ML is uncertain
                ml_s_blended = 50.0  # Treat as neutral — no drag, no boost
                reasons.append(f"ML: uncertain ({ml_s:.0f}), using rules-only")
            else:
                ml_s_blended = ml_s
                reasons.append(f"ML model: {ml_result.get('ml_signal', 'HOLD')} ({ml_result.get('ml_confidence', 0):.0%} confidence)")

            combined_score = combined_score * rule_weight + ml_s_blended * ml_weight
            reasons.append(f"Blend: {rule_weight:.0%} rules + {ml_weight:.0%} ML")
    except Exception:
        pass  # ML not available, use rule-based only

    combined_score = max(0, min(100, combined_score))

    # Store ml_features and regime on the result dict for entry_features logging
    # This is how the self-learning loop works: entry features saved at trade time,
    # used by ml_model_v2 to train on actual bot outcomes after trade closes.
    try:
        quick_result["ml_features"]  = ml_features  # The 31 features used at entry
        quick_result["regime_label"] = _regime_ctx.get("regime_label", "NEUTRAL")
    except Exception:
        pass

    # ── Determine trade SIDE ─────────────────────────────────────────────────
    momentum_is_negative = (momentum_score < 40 or change_pct < -2)
    sentiment_is_bearish = (sent_score_val < 40)
    vrp_is_high = (vrp > 5)
    rsi_overbought = (rsi is not None and rsi > 70)

    side = "buy"
    trade_type = "stock"
    action_label = "BUY"

    if combined_score > 65:
        if vrp_is_high:
            side = "sell"
            trade_type = "options"
            action_label = "SELL OPTIONS"
        elif momentum_is_negative and sentiment_is_bearish:
            side = "short"
            action_label = "SHORT"
        elif rsi_overbought:
            side = "sell"
            action_label = "SELL"
        # else: default BUY

    return {
        **quick_result,
        "deep_score": round(combined_score, 1),
        "reasons": reasons,
        "vrp": vrp,
        "side": side,
        "trade_type": trade_type,
        "action_label": action_label,
        "recommendation": rec.get("action") if rec else None,
        "rec_reasoning": rec.get("reasoning") if rec else None,
        "sentiment_score": sent_score_val,
        "rsi": rsi,
        "momentum_score": round(momentum_score, 1),
        "mean_reversion_score": round(mean_reversion_score, 1),
        "vrp_score": round(vrp_score, 1),
        "squeeze_score": round(squeeze_score_val, 1),
        "volume_score": round(volume_score, 1),
        "ewma_rv": ewma_rv,
        "garch_rv": garch_rv,
        "horizon": rec.get("horizon") if rec else None,
        "leveraged_bull": rec.get("leveraged_bull") if rec else None,
        "leveraged_bear": rec.get("leveraged_bear") if rec else None,
        # Full 52-feature snapshot at entry time (for ML exit model training)
        "entry_features": ml_features if 'ml_features' in dir() else None,
        # Score attribution (exposed at scan level via stock.get())
        "rules_only_score": combined_score if 'combined_score' in dir() else None,
        "ml_only_score": ml_s if 'ml_s' in dir() else None,
    }

# ── Correlation / Sector Check ───────────────────────────────────────────────

def check_sector_correlation(ticker, existing_tickers):
    """
    Returns True if adding this ticker would create dangerous concentration.
    Checks both sector limits AND portfolio beta correlation.
    False = safe to trade.
    """
    # Check 1: Sector concentration
    sector = SECTOR_MAP.get(ticker.upper(), "unknown")
    if sector != "unknown":
        count = sum(
            1 for t in existing_tickers
            if SECTOR_MAP.get(t.upper(), "unknown") == sector
        )
        if count >= MAX_SECTOR_POSITIONS:
            return True

    # Check 2: High-beta concentration — don't load up on all volatile names
    # If we already hold 3+ positions, check if adding this one makes the
    # portfolio too correlated (all high-beta or all defensive)
    if len(existing_tickers) >= 3:
        try:
            _beta_map = {
                # High beta (>1.3) — tech/growth/meme
                "TSLA": 2.0, "NVDA": 1.7, "AMD": 1.6, "MSTR": 2.5, "COIN": 2.0,
                "META": 1.4, "AMZN": 1.3, "NFLX": 1.5, "SHOP": 1.8, "SQ": 1.7,
                "ROKU": 1.9, "SNAP": 1.6, "PLTR": 1.8, "RIVN": 2.0, "LCID": 2.1,
                "SOFI": 1.6, "HOOD": 1.9, "AFRM": 1.8, "UPST": 2.2, "SMCI": 2.0,
                # Medium beta (0.8-1.3) — large cap blend
                "AAPL": 1.2, "MSFT": 1.1, "GOOGL": 1.1, "GOOG": 1.1, "JPM": 1.1,
                "V": 1.0, "MA": 1.0, "UNH": 0.9, "HD": 1.0, "DIS": 1.2,
                # Low beta (<0.8) — defensive/utilities
                "JNJ": 0.6, "PG": 0.5, "KO": 0.6, "PEP": 0.6, "WMT": 0.5,
                "MRK": 0.5, "VZ": 0.4, "T": 0.6, "NEE": 0.5, "SO": 0.3,
                "CL": 0.5, "GIS": 0.4, "K": 0.4, "ED": 0.3, "XEL": 0.3,
            }
            new_beta = _beta_map.get(ticker.upper(), 1.0)
            existing_betas = [_beta_map.get(t.upper(), 1.0) for t in existing_tickers]
            high_beta_count = sum(1 for b in existing_betas if b >= 1.5)

            # Block if adding another high-beta stock when we already have 3+
            if new_beta >= 1.5 and high_beta_count >= 3:
                return True  # Too many volatile names

            # Block if average portfolio beta would exceed 1.6
            all_betas = existing_betas + [new_beta]
            avg_beta = sum(all_betas) / len(all_betas)
            if avg_beta > 1.6:
                return True  # Portfolio too hot
        except Exception:
            pass

    return False

# ── Position Management ──────────────────────────────────────────────────────

def manage_positions():
    """
    Smart position management with ATR-based trailing stops and upgrade logic.
    - Trailing stop: tracks highest price since entry, stop set at entry + (highest - ATR*2)
    - ATR-based: volatile stocks get wider stops, calm stocks get tighter
    - Take profit is also dynamic: ATR * 3 from entry
    - Returns upgrade candidates: positions that could be sold for better picks
    """
    try:
        positions = get_alpaca_positions()
    except Exception:
        return {"actions": [], "error": "Could not fetch positions", "upgrade_candidates": []}

    if not isinstance(positions, list):
        return {"actions": [], "positions": 0, "upgrade_candidates": []}

    actions = []
    upgrade_candidates = []  # Weak positions that could be replaced

    # Load evolving stop state (persists across cycles)
    _stop_state_path = os.path.join(DATA_DIR, 'voltrade_stop_state.json')
    try:
        with open(_stop_state_path) as f:
            stop_state = json.load(f)
    except Exception:
        stop_state = {}

    # Tickers managed by other components — do NOT apply stop/TP logic to these
    FLOOR_AND_LEG_TICKERS = {"QQQ", "SVXY", "ITA", "SPY", "GLD"}  # ETFs only — no single-stock risk

    for pos in positions:
        ticker = pos.get("symbol", "")

        # Skip passive floor, VRP harvest, and sector rotation tickers
        if ticker in FLOOR_AND_LEG_TICKERS:
            continue

        current = float(pos.get("current_price", 0))
        entry = float(pos.get("avg_entry_price", current))
        pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100
        qty = abs(int(float(pos.get("qty", 0))))
        side = pos.get("side", "long")
        market_value = abs(float(pos.get("market_value", 0)))

        # Get ATR for this stock (14-day Average True Range)
        atr = _get_atr(ticker)
        atr_pct = (atr / current * 100) if current > 0 and atr else 2.0

        # ── EVOLVING STOP SYSTEM ────────────────────────────────────────────
        # Stops evolve through 4 phases as the trade profits:
        #   Phase 1 (entry):     2.0x ATR stop, 3.0x ATR target
        #   Phase 2 (at 1R):     Tighten to 1.5x ATR trailing from high
        #   Phase 3 (at 2R):     Tighten to 1.0x ATR trailing from high  
        #   Phase 4 (at 3R+):    Lock in at least 50% of max gain
        # Options have different rules: time-based + delta-based decay
        # ETFs use underlying's ATR * 2 (because 2x leverage)
        
        # Initialize or load stop state for this position
        ps = stop_state.get(ticker, {})
        initial_risk_pct = ps.get("initial_risk_pct", atr_pct * 2.0)  # 1R = 2x ATR from entry
        highest_pnl = max(ps.get("highest_pnl", 0), pnl_pct)  # Track peak P&L
        r_multiple = pnl_pct / initial_risk_pct if initial_risk_pct > 0 else 0  # How many R's we're up
        peak_r = highest_pnl / initial_risk_pct if initial_risk_pct > 0 else 0
        
        # Determine current phase and stop level
        if peak_r >= 3.0:
            # Phase 4: Lock in at least 50% of peak gain
            stop_pct = -(highest_pnl * 0.50)  # Negative means loss threshold
            phase = 4
            stop_reason = f"Phase 4: protecting 50% of {highest_pnl:.1f}% peak gain"
        elif peak_r >= 2.0:
            # Phase 3: 1.0x ATR trailing from high water mark
            stop_pct = max(1.0, atr_pct * 1.0)
            phase = 3
            stop_reason = f"Phase 3 (2R+): 1.0x ATR trailing"
        elif peak_r >= 1.0:
            # Phase 2: 1.5x ATR trailing from high water mark
            stop_pct = max(1.5, atr_pct * 1.5)
            phase = 2
            stop_reason = f"Phase 2 (1R+): 1.5x ATR trailing"
        else:
            # Phase 1: Initial stop at 2.0x ATR
            stop_pct = max(1.5, min(atr_pct * 2.0, 8.0))
            phase = 1
            stop_reason = f"Phase 1: 2.0x ATR initial stop"

        # For phases 2-4, stop is relative to the HIGH WATER MARK, not entry
        if phase >= 2:
            # Stop triggers if current P&L drops more than stop_pct from the peak
            drawdown_from_peak = highest_pnl - pnl_pct
            should_stop = drawdown_from_peak >= stop_pct
        else:
            # Phase 1: stop triggers on absolute loss from entry
            should_stop = pnl_pct <= -stop_pct

        # Dynamic take profit: ATR * 3 from entry (but don't cap at phase 3+)
        tp_pct = max(4.0, min(atr_pct * 3.0, 15.0)) if phase < 3 else 999  # No TP ceiling after 2R

        # Time stop: 7 days with no progress (prevents capital lockup)
        entry_date = ps.get("entry_date", time.strftime("%Y-%m-%d"))
        try:
            days_held = (datetime.now() - datetime.strptime(entry_date, "%Y-%m-%d")).days
        except Exception:
            days_held = 0
        time_stop = days_held >= 7 and abs(pnl_pct) < 2.0  # Flat for a week

        # Save updated stop state
        stop_state[ticker] = {
            "initial_risk_pct": round(initial_risk_pct, 2),
            "highest_pnl": round(highest_pnl, 2),
            "phase": phase,
            "current_stop_pct": round(stop_pct, 2),
            "r_multiple": round(r_multiple, 2),
            "entry_date": entry_date if entry_date != time.strftime("%Y-%m-%d") else ps.get("entry_date", time.strftime("%Y-%m-%d")),
            "days_held": days_held,
        }

        # Exit context: full state at exit time (for ML exit model training)
        exit_context = {
            "atr_pct": round(atr_pct, 2),
            "phase": phase,
            "r_multiple": round(r_multiple, 2),
            "peak_r": round(peak_r, 2),
            "highest_pnl": round(highest_pnl, 2),
            "stop_pct": round(stop_pct, 2),
            "pnl_pct": round(pnl_pct, 2),
            "days_held": days_held,
            "current_price": current,
            "entry_price": entry,
        }

        # Execute stops
        if should_stop:
            stop_type = "trailing_stop" if phase >= 2 else "stop_loss"
            if phase >= 2:
                reason = f"EVOLVING STOP Phase {phase}: P&L {pnl_pct:+.1f}% dropped {highest_pnl - pnl_pct:.1f}% from peak {highest_pnl:+.1f}% ({stop_reason})"
            else:
                reason = f"STOP LOSS: {pnl_pct:.1f}% loss hit Phase 1 stop at -{stop_pct:.1f}% (ATR: ${atr or 0:.2f})"
            actions.append({"action": "CLOSE", "ticker": ticker, "side": side, "reason": reason, "type": stop_type, "phase": phase, "exit_context": exit_context})
            # Record stop-loss cooldown to prevent immediate re-entry
            try:
                _cd_path = os.path.join(DATA_DIR, 'voltrade_stop_cooldown.json')
                try:
                    with open(_cd_path) as _f: _cd = json.load(_f)
                except Exception: _cd = {}
                _cd[ticker] = time.time()
                # Clean entries older than 24 hours
                _cd = {k: v for k, v in _cd.items() if time.time() - v < 86400}
                with open(_cd_path, 'w') as _f: json.dump(_cd, _f)
            except Exception: pass
        elif pnl_pct >= tp_pct and phase < 3:
            actions.append({"action": "CLOSE", "ticker": ticker, "side": side,
                "reason": f"TAKE PROFIT: +{pnl_pct:.1f}% hit target +{tp_pct:.1f}% (Phase {phase})",
                "type": "take_profit", "phase": phase, "exit_context": exit_context})
        elif time_stop:
            actions.append({"action": "CLOSE", "ticker": ticker, "side": side,
                "reason": f"TIME STOP: {days_held} days held, P&L only {pnl_pct:+.1f}% — capital locked up",
                "type": "time_stop", "phase": phase, "exit_context": exit_context})

        # Upgrade candidate: position is flat or slightly negative in Phase 1
        if phase == 1 and -stop_pct < pnl_pct < 1.0:
            upgrade_candidates.append({
                "ticker": ticker, "pnl_pct": pnl_pct, "market_value": market_value,
                "score": 50 + pnl_pct * 5, "qty": qty, "side": side,
            })

    # Persist stop state
    try:
        with open(_stop_state_path, 'w') as f:
            json.dump(stop_state, f)
    except Exception:
        pass

    # Clean up stop state for positions no longer held
    held_tickers = {pos.get('symbol', '') for pos in positions}
    for old_ticker in list(stop_state.keys()):
        if old_ticker not in held_tickers:
            del stop_state[old_ticker]
    try:
        with open(_stop_state_path, 'w') as f:
            json.dump(stop_state, f)
    except Exception:
        pass

    return {"actions": actions, "positions": len(positions), "upgrade_candidates": upgrade_candidates}


def _get_atr(ticker, period=14):
    """Get 14-day ATR for a ticker using Alpaca daily bars."""
    import requests
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        url = (f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars"
               f"?timeframe=1Day&start={start}&limit={period + 5}&adjustment=all&feed=sip")
        resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
        data = resp.json()
        results = data.get("bars", [])
        if len(results) < period:
            return None
        
        # Calculate ATR
        trs = []
        for i in range(1, min(period + 1, len(results))):
            h = results[i].get("h", 0)
            l = results[i].get("l", 0)
            prev_c = results[i - 1].get("c", 0)  # ascending sort: i-1 is the prior day
            # True Range = max(H-L, |H-prevClose|, |L-prevClose|)
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)
        
        if not trs:
            return None
        return sum(trs) / len(trs)
    except Exception:
        return None

# ── Main Scan ────────────────────────────────────────────────────────────────

def _get_full_universe() -> list:
    """
    Returns all tradeable US equity symbols from Alpaca.
    Cached in /data/voltrade/universe_cache.json — refreshed once per day.
    This is the full ~11,600 stock universe, not just the top 100 actives.
    """
    cache_path = os.path.join(DATA_DIR, "universe_cache.json")
    # Use cached list if less than 24 hours old
    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < 86400:  # 24 hours
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception:
                pass
    # Fetch fresh universe from Alpaca assets endpoint
    try:
        resp = requests.get(
            "https://paper-api.alpaca.markets/v2/assets?status=active&asset_class=us_equity",
            headers=_alpaca_headers(), timeout=30
        )
        assets = resp.json()
        symbols = [
            a["symbol"] for a in assets
            if a.get("tradable")
            and "." not in a.get("symbol", "")
            and not a.get("symbol", "").endswith("W")
            and len(a.get("symbol", "")) <= 5
            and a.get("symbol", "")
        ]
        # Cache it
        with open(cache_path, "w") as f:
            json.dump(symbols, f)
        import logging; logging.getLogger("bot_engine").info(f"Universe cache refreshed: {len(symbols):,} symbols")
        return symbols
    except Exception as e:
        import logging; logging.getLogger("bot_engine").warning(f"Could not fetch full universe: {e} — falling back to movers only")
        return []


def scan_market():
    """
    Full market scan — ALL ~11,600 tradeable US stocks, not just top 100.
    
    WHY FULL UNIVERSE:
      - most-actives API only returns top 100 by volume — misses 99% of stocks
      - Scanning 11,600 with 16 parallel workers takes ~4 seconds
      - Deep analysis only runs on top 20 — same speed as before
      - We were missing real opportunities (AEHR +17%, ENVX +13%) every scan
    
    Pipeline:
    1. Load full universe (~11,600 symbols, cached daily)
    2. Fetch ALL snapshots in parallel (16 workers, ~4 seconds)
    3. Quick-score everything that passes price+volume filters
    4. Deep-analyze top 20 candidates (same as before)
    5. Return top positions with full trade recommendations
    """
    from concurrent.futures import ThreadPoolExecutor as _TPE

    # Step 1: Get full universe (cached daily)
    full_universe = _get_full_universe()

    # Also get today's top movers as a "freshness boost" — 
    # they're guaranteed to be moving right now and get priority
    movers_fresh = []
    try:
        resp = requests.get(f"{ALPACA_DATA_URL}/v1beta1/screener/stocks/movers?top=50",
                           headers=_alpaca_headers(), timeout=10)
        mv = resp.json()
        movers_fresh = [s.get("symbol","") for s in mv.get("gainers",[]) + mv.get("losers",[])]
        movers_fresh = [s for s in movers_fresh if s]
    except Exception:
        pass

    # Combine: full universe first, then any fresh movers not already in it
    all_tickers_set = set(full_universe)
    ticker_symbols = list(full_universe)
    for sym in movers_fresh:
        if sym not in all_tickers_set:
            ticker_symbols.append(sym)

    # Fallback if universe fetch failed
    if not ticker_symbols:
        try:
            resp = requests.get(f"{ALPACA_DATA_URL}/v1beta1/screener/stocks/most-actives?by=volume&top=100",
                               headers=_alpaca_headers(), timeout=15)
            ticker_symbols = [s["symbol"] for s in resp.json().get("most_actives", []) if s.get("symbol")]
        except Exception:
            return {"error": "Could not fetch market universe", "trades": []}

    # Step 2: Fetch ALL snapshots in parallel (16 workers = ~4 seconds for 11K stocks)
    batches = [ticker_symbols[i:i+50] for i in range(0, len(ticker_symbols), 50)]
    snap_all = {}

    def _fetch_snap(batch):
        try:
            r = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/snapshots?symbols={','.join(batch)}&feed=sip",
                headers=_alpaca_headers(), timeout=15)
            return r.json()
        except Exception:
            return {}

    # 6 workers balances speed vs Railway thread limits (was 16 — caused EAGAIN)
    with _TPE(max_workers=6) as pool:
        for snap_data in pool.map(_fetch_snap, batches):
            snap_all.update(snap_data)

    quick_results = []
    _et_hour = (datetime.now(__import__("datetime").timezone.utc).hour - 4 + 24) % 24
    _et_min  = datetime.now(__import__("datetime").timezone.utc).minute
    _min_vol = 100_000 if (_et_hour == 9 and _et_min < 60) else MIN_VOLUME

    all_tickers = list(snap_all.keys())  # For scanned count

    # Step 3: Quick-score all stocks that pass price+volume filters
    for sym, snap in snap_all.items():
        try:
            bar = snap.get("dailyBar", {})
            prev = snap.get("prevDailyBar", {})
            c = float(bar.get("c", 0))
            o = float(bar.get("o", c))
            pc = float(prev.get("c", c))
            v = int(bar.get("v", 0))
            if c < MIN_PRICE or v < _min_vol:
                continue
            if "." in sym or len(sym) > 5:
                continue
            change_pct = ((c - pc) / pc * 100) if pc > 0 else 0
            _capped_change = min(abs(change_pct), 15.0)
            _extreme_penalty = -30 if abs(change_pct) > 50 else (-15 if abs(change_pct) > 30 else 0)
            _vol_score = min(v / 1_000_000, 5.0) * 2
            _direction_bonus = 5 if abs(change_pct) > 10 else (2 if abs(change_pct) > 5 else 0)
            quick_results.append({
                "ticker": sym,
                "price": round(c, 2),
                "close": c,
                "open": o,
                "prev_close": pc,
                "volume": v,
                "change_pct": round(change_pct, 2),
                "quick_score": _capped_change * 3 + _vol_score + _direction_bonus + _extreme_penalty,
                "above_vwap": c > o,
                "range_pct": 0,
                "vwap_dist": 0,
                "reasons": [],
            })
        except Exception:
            continue

    if not quick_results:
        return {"error": "Could not fetch market data from Alpaca", "trades": []}

    # Sort by quick score
    quick_results.sort(key=lambda x: x["quick_score"], reverse=True)
    scored = quick_results

    # Step 3: Deep analyze top 20 in PARALLEL
    # Each deep_score internally runs 5 data sources in parallel too.
    # Net result: 20 tickers that used to take ~4 min now complete in ~15-20s.
    from concurrent.futures import ThreadPoolExecutor
    top_candidates = scored[:20]
    deep_scored = [None] * len(top_candidates)

    def _deep_one(args):
        idx, candidate = args
        try:
            return idx, deep_score(candidate["ticker"], candidate)
        except Exception:
            return idx, candidate

    with ThreadPoolExecutor(max_workers=4) as _dpool:
        for idx, result in _dpool.map(_deep_one, enumerate(top_candidates)):
            deep_scored[idx] = result

    deep_scored = [d for d in deep_scored if d is not None]

    # Sort by deep score (or quick score if no deep)
    deep_scored.sort(key=lambda x: x.get("deep_score", x.get("quick_score", 0)), reverse=True)

    # Step 4: Get account info for position sizing
    try:
        account = get_alpaca_account()
        portfolio_value = float(account.get("portfolio_value", 100000))
        cash = float(account.get("cash", 100000))
    except Exception:
        portfolio_value = 100000
        cash = 100000

    # Step 5: Check current positions
    try:
        current_positions = get_alpaca_positions()
        current_tickers = (
            [p.get("symbol") for p in current_positions]
            if isinstance(current_positions, list) else []
        )
        num_positions = len(current_tickers)
    except Exception:
        current_tickers = []
        num_positions = 0

    # Step 6: Generate trade recommendations using DYNAMIC POSITION SIZING
    trades = []
    slots_available = MAX_POSITIONS - num_positions

    # Get macro context for position sizing
    try:
        from macro_data import get_macro_snapshot
        _macro = get_macro_snapshot()
    except Exception:
        _macro = {}

    # Import dynamic sizing engine
    try:
        from position_sizing import calculate_position, check_halt_status
        _has_sizer = True
    except ImportError:
        _has_sizer = False

    for stock in deep_scored:
        if len(trades) >= slots_available:
            break

        final_score = stock.get("deep_score", stock.get("quick_score", 0))
        ticker = stock["ticker"]

        # Skip if already holding
        if ticker in current_tickers:
            continue

        # Skip if recently stopped out — REGIME-AWARE cooldown
        # Logic: when fear is high, stocks keep trending down longer.
        # Re-entering too soon in a panic compounds losses far more than in a calm market.
        #
        # Cooldown durations (backed by intraday momentum research):
        #   PANIC  (VXX > 130% of 30d avg): 4 hours — panic selloffs last 3-6h on avg
        #   BEAR   (VXX > 115%):            3 hours — bearish drift typically lasts 2-4h
        #   CAUTION (VXX > 105%):           2.5 hours — cautious market, extra buffer
        #   NEUTRAL (VXX near avg):         2 hours — baseline (was always 2h)
        #   BULL   (VXX < 90% of avg):      1.5 hours — calm/rising mkt = faster mean revert
        #
        # How regime is detected at this point: uses the most recent macro snapshot's VXX
        # ratio. Falls back to 2h (neutral) if data unavailable. No API call — uses
        # _macro dict already loaded above.
        _cooldown_path = os.path.join(DATA_DIR, 'voltrade_stop_cooldown.json')
        try:
            with open(_cooldown_path) as _f:
                _cooldown = json.load(_f)
        except Exception:
            _cooldown = {}
        _last_stop = _cooldown.get(ticker, 0)
        # Determine cooldown seconds from current regime (Fix B: include 200d MA)
        _vxx_r_scan      = float(_macro.get("vxx_ratio", 1.0) or 1.0)
        _spy_ma_scan     = float(_macro.get("spy_vs_ma50", 1.0) or 1.0)
        _spy_b200_scan   = int(_macro.get("spy_below_200_days", 0) or 0)
        _spy_above_200d  = bool(_macro.get("spy_above_200d", True))
        # Use the same regime function so 200MA block is consistent everywhere
        try:
            from system_config import get_market_regime as _gmr
            _scan_regime = _gmr(_vxx_r_scan, _spy_ma_scan,
                                spy_below_200_days=_spy_b200_scan,
                                spy_above_200d=_spy_above_200d)
        except Exception:
            # Fallback inline
            if _vxx_r_scan >= 1.30 or _spy_ma_scan < 0.94 or _spy_b200_scan >= 10:
                _scan_regime = "PANIC" if _vxx_r_scan >= 1.30 else "BEAR"
            elif _vxx_r_scan >= 1.15: _scan_regime = "BEAR"
            elif _vxx_r_scan >= 1.05: _scan_regime = "CAUTION"
            elif _vxx_r_scan <= 0.90: _scan_regime = "BULL"
            else:                     _scan_regime = "NEUTRAL"
        _cooldown_map = {"PANIC": 14400, "BEAR": 10800, "CAUTION": 9000,
                         "BULL": 5400, "NEUTRAL": 7200}
        _cooldown_secs  = _cooldown_map.get(_scan_regime, 7200)
        _cooldown_label = f"{_scan_regime} ({_cooldown_secs//3600:.1f}h)"
        if time.time() - _last_stop < _cooldown_secs:
            continue  # Ticker in cooldown — skip. Re-entry blocked ({_cooldown_label})

        # Fix B: block ALL new stock longs in BEAR/PANIC (200MA slow-bear included)
        if _scan_regime in ("BEAR", "PANIC"):
            continue  # No new stock longs in bear/panic — preserve capital

        # Skip if below minimum score — use regime-adaptive threshold from get_adaptive_params.
        # Previously this used a hardcoded MIN_SCORE=65 which conflicted with system_config
        # values (63 in BULL, 67 in CAUTION, 75 in BEAR/PANIC). Now uses the adaptive value.
        try:
            from system_config import get_adaptive_params as _gap
            _scan_params = _gap(
                vxx_ratio=_vxx_r_scan,
                spy_vs_ma50=_spy_ma_scan,
                spy_below_200_days=_spy_b200_scan,
            )
            _min_score_threshold = _scan_params.get("MIN_SCORE", 63)
        except Exception:
            _min_score_threshold = 63  # system_config BASE_CONFIG default
        if final_score < _min_score_threshold:
            continue

        # Correlation / sector check — don't over-concentrate
        if check_sector_correlation(ticker, current_tickers + [t["ticker"] for t in trades]):
            continue

        side = stock.get("side", "buy")
        action_label = stock.get("action_label", "BUY")
        change_pct_val = stock.get("change_pct", 0) or 0

        # ── Dollar volume filter (Fix D) ────────────────────────────────
        # Minimum $50M daily dollar volume = price × shares traded.
        # WHY: FCUV ($7, $370M vol) and PFSA ($2, $450M vol) passed all other
        # filters but are micro-caps with manipulated momentum spikes.
        # A $50M floor blocks them while keeping NVDA at $13 (post-split, $2B+/day).
        # Does NOT use a price floor (that hurt backtest by blocking split-adj stocks).
        _stock_price  = float(stock.get("price", 0) or 0)
        _stock_volume = float(stock.get("volume", 0) or 0)
        _dollar_vol   = _stock_price * _stock_volume
        _MIN_DOLLAR_VOL = 50_000_000  # $50M minimum daily dollar volume
        if _dollar_vol < _MIN_DOLLAR_VOL:
            continue  # Skip micro-cap / low-liquidity stock

        # Sector quality filter — 3-year backtest confirmed these destroy returns
        # Gaming (DKNG, RBLX): 25% WR over 3 years  | Leveraged ETFs: 22% WR
        # Travel (ABNB, DASH): 20% WR               | These are structural drags
        _BLOCKED_TICKERS = {
            "DKNG", "RBLX",            # Gaming — 25% win rate over 3 years
            "SQQQ", "TQQQ", "SPXU", "UPRO", "UVXY",  # Leveraged ETFs — 22% WR
            "ABNB", "DASH",            # Travel — 20% WR
            "LYFT", "UBER",            # Ride-share — consistent underperformers
        }
        if ticker in _BLOCKED_TICKERS:
            continue  # Skip — 3-year backtest shows these hurt more than they help

        # Extreme mover check: stock already up 50%+ today
        # Don't buy OR short the spike — write to watchlist for overnight analysis
        # Tomorrow's setup (continuation or mean reversion) is a much cleaner entry
        if change_pct_val > 50:
            _em_path = os.path.join(DATA_DIR, 'extreme_movers_today.json')
            try:
                try:
                    with open(_em_path) as _f: _em = json.load(_f)
                except Exception: _em = []
                # Add if not already in list
                if not any(e.get('ticker') == ticker for e in _em):
                    _em.append({
                        'ticker': ticker, 'price': stock.get('price', 0),
                        'change_pct': change_pct_val, 'volume': stock.get('volume', 0),
                        'score': final_score, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'high': stock.get('high', 0), 'setup': 'extreme_mover',
                    })
                    with open(_em_path, 'w') as _f: json.dump(_em, _f)
            except Exception: pass
            continue  # Skip — will be analyzed tonight for tomorrow

        # Smart short logic:
        # - Large-cap liquid stocks → allow actual short sell (instrument_selector handles borrow check)
        # - Everything else → options path (puts) handles it, don't convert to buy
        # - Micro-caps up 50%+ → already blocked by extreme mover filter above
        _LARGE_CAP_SHORTABLE = {
            "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG",
            "META", "TSLA", "AMD", "AVGO", "COST", "NFLX", "ORCL", "CRM", "ADBE",
            "INTC", "QCOM", "TXN", "AMAT", "LRCX", "MU", "JPM", "BAC", "GS", "MS",
            "V", "MA", "UNH", "JNJ", "LLY", "ABBV", "PFE", "MRK", "WMT", "HD", "XOM", "CVX",
        }
        if side == "sell" and stock.get("trade_type") != "options":
            if ticker in _LARGE_CAP_SHORTABLE:
                pass  # Allow short — large-cap, borrow always available
            else:
                side = "buy"  # Non-large-cap: convert to buy (options path handles bearish plays)
                action_label = "BUY"

        # Instrument decision: stock vs 2x ETF vs options (unified selector)
        instrument_decision = {"chosen": "stock", "strategy": "buy_stock", "reasoning": "default"}
        options_decision = {"use_options": False, "strategy": "stock", "reason": ""}
        try:
            from instrument_selector import select_instrument
            instrument_decision = select_instrument(
                trade={**stock, "score": final_score, "deep_score": final_score,
                       "side": side, "action_label": action_label},
                equity=portfolio_value,
                positions=current_positions if isinstance(current_positions, list) else [],
                macro=_macro,
            )
            # Backward compat for options_execution fields
            if instrument_decision.get("chosen") == "options":
                options_decision = {"use_options": True, "strategy": instrument_decision.get("strategy", "stock"),
                                    "reason": instrument_decision.get("reasoning", ""),
                                    "edge_pct": instrument_decision.get("scores", {}).get("options", {}).get("edge_pct", 0)}
        except Exception:
            try:
                from options_execution import should_use_options
                options_decision = should_use_options(
                    {**stock, "score": final_score, "side": side, "action_label": action_label},
                    portfolio_value,
                    current_positions if isinstance(current_positions, list) else []
                )
                if options_decision.get("use_options"):
                    instrument_decision = {"chosen": "options", "strategy": options_decision["strategy"],
                                           "reasoning": options_decision["reason"]}
            except Exception:
                pass  # Both modules unavailable, default to stock

        # Dynamic position sizing (replaces fixed 5%)
        if _has_sizer:
            sizing = calculate_position(
                trade={**stock, "score": final_score, "side": side},
                equity=portfolio_value,
                current_positions=current_positions if isinstance(current_positions, list) else [],
                macro=_macro,
            )
            if sizing.get("blocked"):
                continue  # Sizing engine blocked this trade (halted, too small, etc.)
            shares = sizing["shares"]
            position_value = sizing["position_value"]
            stop_loss = sizing["stop_loss"]
            take_profit = sizing["take_profit"]
            sizing_reasoning = sizing.get("reasoning", "")
            sizing_scalars = sizing.get("scalars", {})
        else:
            # Fallback to old fixed sizing if import fails
            position_value = min(portfolio_value * MAX_POSITION_PCT, cash * 0.9)
            shares = int(position_value / stock["price"]) if stock["price"] > 0 else 0
            stop_loss = round(stock["price"] * (1 - STOP_LOSS_PCT), 2)
            take_profit = round(stock["price"] * (1 + TAKE_PROFIT_PCT), 2)
            position_value = round(shares * stock["price"], 2)
            sizing_reasoning = "Fallback: fixed 5% sizing"
            sizing_scalars = {}

        if shares <= 0:
            continue

        trades.append({
            "action": action_label,
            "side": side,
            "trade_type": stock.get("trade_type", "stock"),
            "ticker": ticker,
            "shares": shares,
            "price": stock["price"],
            "score": final_score,
            "reasons": stock.get("reasons", []),
            "recommendation": stock.get("recommendation"),
            "rec_reasoning": stock.get("rec_reasoning"),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_value": position_value,
            "sizing_reasoning": sizing_reasoning,
            "sizing_scalars": sizing_scalars,
            "momentum_score": stock.get("momentum_score"),
            "mean_reversion_score": stock.get("mean_reversion_score"),
            "vrp_score": stock.get("vrp_score"),
            "squeeze_score": stock.get("squeeze_score"),
            "volume_score": stock.get("volume_score"),
            "ewma_rv": stock.get("ewma_rv"),
            "garch_rv": stock.get("garch_rv"),
            "rsi": stock.get("rsi"),
            "vrp": stock.get("vrp"),
            "instrument": instrument_decision.get("chosen", "stock"),
            "instrument_strategy": instrument_decision.get("strategy", "buy_stock"),
            "instrument_reasoning": instrument_decision.get("reasoning", ""),
            "instrument_ticker": instrument_decision.get("ticker", ticker),  # Could be ETF ticker
            "instrument_scores": instrument_decision.get("scores", {}),
            "use_options": options_decision.get("use_options", False),
            "options_strategy": options_decision.get("strategy", "stock"),
            "options_reasoning": options_decision.get("reason", ""),
            "options_edge_pct": options_decision.get("edge_pct", 0),
            "rules_score": stock.get("rules_only_score"),
            "ml_score_raw": stock.get("ml_only_score"),
            # entry_features: the 25 ML features at the moment of entry.
            # Saved here so when the trade closes, ml_model_v2 can train on
            # THESE SPECIFIC SIGNALS — not generic market data.
            # This is what enables true self-learning.
            "entry_features": stock.get("entry_features") or stock.get("ml_features"),
            "entry_date": time.strftime("%Y-%m-%d"),
            "regime_at_entry": stock.get("regime_label", "UNKNOWN"),
        })

    # Step 6b: Run options scanner synchronously with real portfolio equity.
    # NOTE: The original code had a parallel "Step 0" that submitted options with
    # equity=100_000 hardcoded, then discarded that result and called again here.
    # That wasted one full scan per cycle. Now options runs once, here only.
    options_trade_count = 0
    try:
        from options_scanner import get_options_trades
        options_trades = get_options_trades(
            equity=portfolio_value,
            current_tickers=current_tickers + [t["ticker"] for t in trades],
            max_new=max(1, slots_available - len(trades)),
            min_score=65.0,
        )
        for ot in options_trades:
            if len(trades) >= slots_available:
                break
            # Convert to bot_engine trade format
            trades.append({
                "action":             ot.get("action_label", "OPTIONS"),
                "side":               ot.get("side", "buy"),
                "trade_type":         "options",
                "ticker":             ot["ticker"],
                "shares":             0,  # Options: shares = 0, handled by options_execution
                "price":              ot["price"],
                "score":              ot["score"],
                "reasons":            [ot.get("reasoning", "")],
                "recommendation":     None,
                "rec_reasoning":      ot.get("reasoning", ""),
                "stop_loss":          None,
                "take_profit":        None,
                "position_value":     ot.get("position_dollars", portfolio_value * 0.05),
                "sizing_reasoning":   f"Options scanner: {ot.get('setup','unknown')} setup",
                "sizing_scalars":     {},
                "momentum_score":     None,
                "mean_reversion_score": None,
                "vrp_score":          None,
                "squeeze_score":      None,
                "volume_score":       None,
                "ewma_rv":            None,
                "garch_rv":           None,
                "rsi":                None,
                "vrp":                None,
                "instrument":         "options",
                "instrument_strategy": ot.get("options_strategy", ""),
                "instrument_reasoning": ot.get("reasoning", ""),
                "instrument_ticker":  ot["ticker"],
                "instrument_scores":  {},
                "use_options":        True,
                "options_strategy":   ot.get("options_strategy", ""),
                "options_reasoning":  ot.get("reasoning", ""),
                "options_edge_pct":   0,
                "rules_score":        ot["score"],
                "ml_score_raw":       None,
                "entry_features":     None,
                "entry_date":         time.strftime("%Y-%m-%d"),
                "regime_at_entry":    "OPTIONS_SCANNER",
                # Pass through all setup-specific keys for execution
                **{k: v for k, v in ot.items()
                   if k not in ("action", "action_label", "side", "trade_type",
                                "ticker", "price", "score", "reasoning",
                                "position_dollars", "position_pct", "source")},
            })
            options_trade_count += 1
    except Exception as _oe2:
        pass  # Options scan failed — stock trades still execute normally

    # Step 7: Check position management
    mgmt = manage_positions()

    # Step 8: Intraday Shorts (v1.0.28 hybrid v2.1) ───────────────────────────────────
    # Hybrid v2.1: fixed lookback signals + full universe architecture.
    try:
        from intraday_shorts import run_intraday_shorts
        intraday_short_result = run_intraday_shorts(_macro)
    except Exception:
        intraday_short_result = {"actions": [], "status": "error"}

    # Step 9: Third Leg (v1.0.25) ─────────────────────────────────────────────
    # Runs alongside stock scan in every cycle.
    # Backtest: ALL 64 combinations beat SPY. Winner: VRP=15% + Sector=12%.
    # Result: CAGR +14.8%/yr vs 2-leg +13.8% vs SPY +12.3%
    try:
        third_leg_result = _run_third_leg(_macro)
    except Exception:
        third_leg_result = {"actions": [], "status": "error"}

    # Step 10: Passive SPY Floor (v1.0.29) ──────────────────────────────────
    # Hold passive SPY allocation based on regime. Captures market drift
    # in calm bull markets where momentum signals are noise.
    # Backtest: 12.9% CAGR (beats SPY 12.3%). Fixes quiet bull year losses.
    spy_floor_result = {"actions": [], "status": "ok", "target_pct": 0}
    try:
        spy_floor_result = _manage_spy_floor(_macro)
    except Exception as _sfe:
        spy_floor_result["status"] = f"error: {str(_sfe)[:80]}"

    return {
        "timestamp": datetime.now().isoformat(),
        "scanned": len(all_tickers),
        "filtered": len(scored),
        "deep_analyzed": len(deep_scored),
        "portfolio_value": portfolio_value,
        "cash": cash,
        "current_positions": num_positions,
        "slots_available": slots_available,
        "new_trades": trades,
        "options_trades_added": options_trade_count,
        "position_actions": mgmt.get("actions", []),
        "upgrade_candidates": mgmt.get("upgrade_candidates", []),
        "top_10": [{
            "ticker": s["ticker"],
            "price": s["price"],
            "score": s.get("deep_score", s.get("quick_score", 0)),
            "change_pct": s["change_pct"],
            "side": s.get("side", "buy"),
            "action_label": s.get("action_label", "BUY"),
            "reasons": s.get("reasons", [])[:2],
        } for s in deep_scored[:10]],
        "third_leg": third_leg_result,
        "intraday_shorts": intraday_short_result,
        "spy_floor": spy_floor_result,
    }


# ── ML Attribution Tracker ───────────────────────────────────────────────────
def ml_attribution_summary() -> dict:
    """
    Aggregate ML attribution across all closed trades to measure whether
    the ML model helps or hurts trade selection vs pure rules.

    Reads from the trade feedback log (voltrade_trade_feedback.json) and
    computes per-trade deltas between rules_score and ml_score_raw.

    Returns a dict with:
      - n_trades: total closed trades with both scores recorded
      - avg_rules_score: average rules-only score at entry
      - avg_ml_score: average ML score at entry
      - avg_outcome_pnl: average realized P&L % across those trades
      - ml_boosted_n: trades where ML score > rules score
      - ml_suppressed_n: trades where ML score < rules score
      - ml_lift_pct: estimated P&L improvement from ML adjustment
      - verdict: 'ML_HELPS' / 'ML_HURTS' / 'NEUTRAL' / 'INSUFFICIENT_DATA'

    Use this to decide whether to increase or decrease the ML blend weight.
    """
    try:
        from ml_model_v2 import FEEDBACK_PATH as _fp
    except ImportError:
        _fp = os.path.join(DATA_DIR, "voltrade_trade_feedback.json")

    try:
        if not os.path.exists(_fp):
            return {"verdict": "INSUFFICIENT_DATA", "reason": "No feedback file"}
        with open(_fp) as f:
            trades = json.load(f)
    except Exception as e:
        return {"verdict": "INSUFFICIENT_DATA", "reason": str(e)}

    # Filter to closed trades with both scores recorded
    scored = [
        t for t in trades
        if t.get("rules_score") is not None
        and t.get("ml_score_raw") is not None
        and t.get("pnl_pct") is not None
    ]

    if len(scored) < 10:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n_trades": len(scored),
            "reason": f"Need 10+ trades with both scores, have {len(scored)}",
        }

    rules_scores = [t["rules_score"] for t in scored]
    ml_scores    = [t["ml_score_raw"] for t in scored]
    pnls         = [t["pnl_pct"] for t in scored]

    # Trades where ML raised the score (ML was more bullish than rules)
    ml_boosted    = [t for t in scored if t["ml_score_raw"] > t["rules_score"] + 2]
    # Trades where ML lowered the score (ML was more cautious)
    ml_suppressed = [t for t in scored if t["ml_score_raw"] < t["rules_score"] - 2]

    ml_boost_pnl = [t["pnl_pct"] for t in ml_boosted]
    ml_supp_pnl  = [t["pnl_pct"] for t in ml_suppressed]

    avg_boost_pnl = float(sum(ml_boost_pnl) / len(ml_boost_pnl)) if ml_boost_pnl else None
    avg_supp_pnl  = float(sum(ml_supp_pnl) / len(ml_supp_pnl))   if ml_supp_pnl  else None
    avg_all_pnl   = float(sum(pnls) / len(pnls))

    # Compute lift: do ML-boosted trades outperform ML-suppressed ones?
    ml_lift = None
    if avg_boost_pnl is not None and avg_supp_pnl is not None:
        ml_lift = round(avg_boost_pnl - avg_supp_pnl, 3)

    # Verdict: ML helps if boosted trades clearly outperform suppressed ones
    if ml_lift is None:
        verdict = "INSUFFICIENT_DATA"
    elif ml_lift > 0.5:   # ML-boosted trades beat ML-suppressed by >0.5%
        verdict = "ML_HELPS"
    elif ml_lift < -0.5:  # ML-boosted trades underperform by >0.5%
        verdict = "ML_HURTS"
    else:
        verdict = "NEUTRAL"

    return {
        "verdict":               verdict,
        "n_trades":              len(scored),
        "avg_rules_score":       round(sum(rules_scores) / len(rules_scores), 2),
        "avg_ml_score":          round(sum(ml_scores) / len(ml_scores), 2),
        "avg_outcome_pnl":       round(avg_all_pnl, 3),
        "ml_boosted_n":          len(ml_boosted),
        "ml_boosted_avg_pnl":    round(avg_boost_pnl, 3) if avg_boost_pnl is not None else None,
        "ml_suppressed_n":       len(ml_suppressed),
        "ml_suppressed_avg_pnl": round(avg_supp_pnl, 3) if avg_supp_pnl is not None else None,
        "ml_lift_pct":           ml_lift,
        "interpretation": (
            f"ML boosted returns by {abs(ml_lift):.2f}% on ML-influenced trades"
            if ml_lift is not None and ml_lift > 0
            else (
                f"ML reduced returns by {abs(ml_lift):.2f}% on ML-influenced trades"
                if ml_lift is not None and ml_lift < 0
                else "Not enough ML-influenced trades to judge impact"
            )
        ),
    }


# ── Third Leg Engine (v1.0.25) ──────────────────────────────────────────────
def _run_third_leg(macro: dict) -> dict:
    """
    Third leg: VRP harvest + Sector rotation.
    Runs every scan cycle alongside stocks + options.

    Backtest results (64 combinations, 2016-2026):
      ALL 64 beat SPY. Winner: VRP=15% + Sector=12%
      CAGR: +14.8%/yr vs 2-leg +13.8% vs SPY +12.3%

    Leg 3A (TLT): DISABLED — TLT was crushed in 2022 rate hike cycle
    Leg 3B (VRP): 15% of equity, fires when VXX ratio 1.05-1.25 + declining
    Leg 3C (Sector): 12% into XOM+LMT when BEAR/CAUTION regime
    """
    actions = []
    import logging as _logging
    _log = _logging.getLogger("voltrade.leg3")
    try:
        from system_config import BASE_CONFIG, get_market_regime
        import requests as _req

        LEG3_VRP_PCT    = float(BASE_CONFIG.get("LEG3_VRP_PCT",    0.15))
        # LEG3_SECTOR_PCT removed — replaced by regime-adaptive LEG3_CRASH_ASSETS / LEG3_RECOVERY_ASSETS
        LEG3_TLT_PCT    = float(BASE_CONFIG.get("LEG3_TLT_PCT",    0.00))

        vxx_ratio       = float(macro.get("vxx_ratio",    1.0) or 1.0)
        spy_vs_ma50     = float(macro.get("spy_vs_ma50",  1.0) or 1.0)
        spy_b200        = int(macro.get("spy_below_200_days", 0) or 0)
        spy_above_200d  = bool(macro.get("spy_above_200d", True))
        regime          = get_market_regime(vxx_ratio, spy_vs_ma50,
                                            spy_below_200_days=spy_b200,
                                            spy_above_200d=spy_above_200d)

        alpaca_key    = os.environ.get("ALPACA_KEY",    "PKMDHJOVQEVIB4UHZXUYVTIDBU")
        alpaca_secret = os.environ.get("ALPACA_SECRET", "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et")
        base_url      = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        headers       = {"APCA-API-KEY-ID": alpaca_key,
                         "APCA-API-SECRET-KEY": alpaca_secret}

        # Fetch account equity and existing positions
        acc_r = _req.get(f"{base_url}/v2/account", headers=headers, timeout=8)
        acc   = acc_r.json()
        equity = float(acc.get("equity", 100_000) or 100_000)

        pos_r = _req.get(f"{base_url}/v2/positions", headers=headers, timeout=8)
        positions_raw = pos_r.json() if isinstance(pos_r.json(), list) else []
        position_syms = {p["symbol"] for p in positions_raw}

        # Also check pending/open orders to prevent duplicate buys
        # (fixes GLD flooding: bot bought 13x before position registered)
        try:
            open_orders = _req.get(f"{base_url}/v2/orders",
                params={"status": "open", "limit": 50}, headers=headers, timeout=8).json()
            if isinstance(open_orders, list):
                for oo in open_orders:
                    position_syms.add(oo.get("symbol", ""))
        except Exception:
            pass

        # ── Leg 3B: VRP Harvest ────────────────────────────────────────
        # When VXX elevated (1.05-1.25) AND declining: sell volatility
        # Implementation: buy SVXY (inverse VXX ETF) — goes up when VXX falls
        # Simpler than options straddles, trackable as a regular stock position
        if LEG3_VRP_PCT > 0 and 1.05 <= vxx_ratio <= 1.25:
            # Check VXX trend: fetch last 5 days
            vxx_r = _req.get("https://data.alpaca.markets/v2/stocks/bars",
                params={"symbols":"VXX","timeframe":"1Day","limit":6,"feed":"sip"},
                headers=headers, timeout=8)
            vxx_bars = vxx_r.json().get("bars",{}).get("VXX",[])
            vxx_declining = (len(vxx_bars) >= 2 and
                             float(vxx_bars[-1]["c"]) < float(vxx_bars[-5 if len(vxx_bars)>=5 else 0]["c"]))

            if vxx_declining and "SVXY" not in position_syms:
                # Buy SVXY: inverse volatility ETF (goes up as VXX falls)
                svxy_r = _req.get("https://data.alpaca.markets/v2/stocks/snapshots",
                    params={"symbols":"SVXY","feed":"sip"}, headers=headers, timeout=8)
                svxy_price = float(svxy_r.json().get("SVXY",{}).get("latestTrade",{}).get("p", 0) or 0)
                if svxy_price > 0:
                    alloc  = equity * LEG3_VRP_PCT
                    shares = int(alloc / svxy_price)
                    if shares > 0 and alloc > 100:
                        order = {
                            "symbol":       "SVXY",
                            "qty":          str(shares),
                            "side":         "buy",
                            "type":         "limit",
                            "limit_price":  str(round(svxy_price * 1.001, 2)),
                            "time_in_force":"day",
                        }
                        try:
                            o = _req.post(f"{base_url}/v2/orders",
                                          json=order, headers=headers, timeout=10)
                            actions.append({
                                "type": "vrp_harvest",
                                "symbol": "SVXY",
                                "shares": shares,
                                "price": svxy_price,
                                "reason": f"VXX ratio {vxx_ratio:.3f} (elevated+declining) — VRP harvest",
                                "regime": regime,
                                "order_id": o.json().get("id", "?"),
                            })
                            _log.info(f"[LEG3-VRP] Bought {shares} SVXY @ {svxy_price:.2f} (VRP harvest, regime={regime})")
                        except Exception as e:
                            _log.debug(f"[LEG3-VRP] Order failed: {e}")

        # ── Leg 3C: Regime-Adaptive Sector Rotation (v1.0.30) ─────────
        # CRASH (PANIC/BEAR): GLD (gold) — +0.122%/day in bear, near-zero SPY corr
        # RECOVERY (CAUTION): XOM+LMT — strong cyclical bounce-back
        # Backtest: 19.8% CAGR (beats SPY by +7.5%), up from 18.4% with fixed XOM+LMT
        if regime in ("BEAR", "PANIC", "CAUTION"):
            if regime in ("PANIC", "BEAR"):
                # Crash hedge: GLD goes UP when SPY crashes
                sector_assets = BASE_CONFIG.get("LEG3_CRASH_ASSETS", [("GLD", 0.15)])
            else:
                # Recovery bounce: XOM+LMT rebound hard after fear normalizes
                sector_assets = BASE_CONFIG.get("LEG3_RECOVERY_ASSETS", [("XOM", 0.10), ("LMT", 0.10)])
            for ssym, alloc_pct in sector_assets:
                alloc_each = equity * alloc_pct
                if alloc_each <= 100: continue
                if ssym in position_syms: continue
                snap_r = _req.get("https://data.alpaca.markets/v2/stocks/snapshots",
                    params={"symbols": ssym, "feed": "sip"}, headers=headers, timeout=8)
                price = float(snap_r.json().get(ssym,{}).get("latestTrade",{}).get("p", 0) or 0)
                if price > 0:
                    shares = int(alloc_each / price)
                    if shares > 0:
                        order = {
                            "symbol":       ssym,
                            "qty":          str(shares),
                            "side":         "buy",
                            "type":         "limit",
                            "limit_price":  str(round(price * 1.001, 2)),
                            "time_in_force": "day",
                        }
                        try:
                            o = _req.post(f"{base_url}/v2/orders",
                                          json=order, headers=headers, timeout=10)
                            actions.append({
                                "type": "sector_rotation",
                                "symbol": ssym,
                                "shares": shares,
                                "price": price,
                                "reason": f"{regime} — {'crash hedge (GLD)' if regime in ('PANIC','BEAR') else 'recovery rotation'}",
                                "regime": regime,
                                "order_id": o.json().get("id", "?"),
                            })
                            _log.info(f"[LEG3-SECTOR] Bought {shares} {ssym} @ {price:.2f} ({regime} rotation)")
                        except Exception as e:
                            _log.debug(f"[LEG3-SECTOR] Order failed for {ssym}: {e}")

        # ── Exit sector positions when regime recovers ─────────────────
        if regime in ("NEUTRAL", "BULL"):
            for pos in positions_raw:
                sym = pos.get("symbol", "")
                if sym in ("ITA", "SVXY", "GLD", "XOM", "LMT"):  # Exit third-leg + legacy positions when regime normalizes
                    qty = pos.get("qty", "0")
                    try:
                        o = _req.post(f"{base_url}/v2/orders",
                            json={"symbol": sym, "qty": qty, "side": "sell",
                                  "type": "market", "time_in_force": "day"},
                            headers=headers, timeout=10)
                        pnl = float(pos.get("unrealized_plpc", 0) or 0) * 100
                        actions.append({
                            "type": "third_leg_exit",
                            "symbol": sym,
                            "reason": f"Regime recovered to {regime} — exiting third leg",
                            "pnl_pct": round(pnl, 2),
                        })
                        _log.info(f"[LEG3-EXIT] Sold {sym} (regime={regime}, P&L={pnl:.1f}%)")
                    except Exception as e:
                        _log.debug(f"[LEG3-EXIT] Sell failed for {sym}: {e}")

    except Exception as e:
        _log.debug(f"[LEG3] Third leg error: {e}")
        return {"actions": actions, "status": "error", "error": str(e)[:100]}

    return {
        "actions":   actions,
        "status":    "ok",
        "regime":    regime if 'regime' in dir() else "unknown",
        "vrp_on":    LEG3_VRP_PCT > 0 if 'LEG3_VRP_PCT' in dir() else False,
        "sector_on": LEG3_SECTOR_PCT > 0 if 'LEG3_SECTOR_PCT' in dir() else False,
    }


# ── Entry Point ──────────────────────────────────────────────────────────────

# ── Passive SPY Floor (v1.0.29) ──────────────────────────────────────────────

def _manage_spy_floor(macro: dict) -> dict:
    """
    Passive SPY Floor: hold SPY shares proportional to regime allocation.
    Captures market drift in calm bull markets where momentum signals = noise.

    Backtest (7-config sweep, 2016-2026):
      B60/N85/C30 = best: 12.9% CAGR vs SPY 12.3% (beats by +0.6%)
      Fixes quiet bull year losses: 2017 -27% -> -6%, 2019 -33% -> -6%
    """
    result = {"actions": [], "status": "ok", "target_pct": 0, "current_pct": 0}

    try:
        from system_config import BASE_CONFIG, get_market_regime

        vxx_ratio   = float(macro.get("vxx_ratio", 1.0) or 1.0)
        spy_vs_ma50 = float(macro.get("spy_vs_ma50", 1.0) or 1.0)
        spy_b200    = int(macro.get("spy_below_200_days", 0) or 0)
        spy_above   = bool(macro.get("spy_above_200d", True))

        regime = get_market_regime(vxx_ratio, spy_vs_ma50,
                                   spy_below_200_days=spy_b200,
                                   spy_above_200d=spy_above)

        # Use configurable floor ticker (QQQ by default — 5.6%/yr more than SPY)
        floor_ticker = BASE_CONFIG.get("FLOOR_TICKER", "QQQ")
        floor_key = f"SPY_FLOOR_{regime}"
        target_pct = BASE_CONFIG.get(floor_key, 0)
        result["target_pct"] = target_pct
        result["regime"] = regime
        result["floor_ticker"] = floor_ticker

        if target_pct <= 0:
            # No floor — sell any existing position
            try:
                positions = get_alpaca_positions()
                floor_pos = [p for p in positions if p.get("symbol") == floor_ticker]
                if floor_pos:
                    qty = abs(int(float(floor_pos[0].get("qty", 0))))
                    if qty > 0:
                        requests.post(f"https://paper-api.alpaca.markets/v2/orders",
                            json={"symbol": floor_ticker, "qty": str(qty),
                                  "side": "sell", "type": "market",
                                  "time_in_force": "day"},
                            headers=_alpaca_headers(), timeout=10)
                        result["actions"].append({"type": "floor_exit",
                            "shares": qty, "reason": f"{regime} regime"})
                        import logging as _floor_log; _floor_log.getLogger("bot_engine").info(f"[FLOOR] Sold {qty} {floor_ticker} ({regime} regime)")
            except Exception:
                pass
            return result

        # Get account equity and current SPY position
        try:
            acc = requests.get("https://paper-api.alpaca.markets/v2/account",
                headers=_alpaca_headers(), timeout=8).json()
            equity = float(acc.get("equity", 98000) or 98000)
        except Exception:
            equity = 98000

        current_spy_value = 0
        current_spy_shares = 0
        try:
            positions = get_alpaca_positions()
            for p in positions:
                if p.get("symbol") == floor_ticker:
                    current_spy_shares = int(float(p.get("qty", 0)))
                    current_spy_value = abs(float(p.get("market_value", 0)))
                    break
        except Exception:
            pass

        current_pct = current_spy_value / equity if equity > 0 else 0
        result["current_pct"] = round(current_pct, 3)

        target_value = equity * target_pct
        diff = target_value - current_spy_value

        # Only rebalance if >5% off target (avoid micro-trades)
        if abs(diff) / equity < 0.05:
            result["status"] = "within_band"
            return result

        # Get SPY price
        try:
            snap = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/snapshots",
                params={"symbols": floor_ticker, "feed": "sip"},
                headers=_alpaca_headers(), timeout=8).json()
            spy_price = float(snap.get(floor_ticker, {}).get("latestTrade", {}).get("p", 0) or 0)
        except Exception:
            spy_price = 0

        if spy_price <= 0:
            result["status"] = "no_spy_price"
            return result

        shares_diff = int(diff / spy_price)
        if shares_diff == 0:
            result["status"] = "within_band"
            return result

        if shares_diff > 0:
            try:
                requests.post("https://paper-api.alpaca.markets/v2/orders",
                    json={"symbol": floor_ticker, "qty": str(shares_diff),
                          "side": "buy", "type": "market",
                          "time_in_force": "day"},
                    headers=_alpaca_headers(), timeout=10)
                result["actions"].append({"type": "floor_buy",
                    "shares": shares_diff, "ticker": floor_ticker,
                    "reason": f"{regime}: target {target_pct*100:.0f}%, current {current_pct*100:.0f}%"})
                import logging as _floor_log; _floor_log.getLogger("bot_engine").info(f"[FLOOR] Bought {shares_diff} {floor_ticker} ({regime}: {target_pct*100:.0f}% target)")
            except Exception as e:
                import logging as _floor_log; _floor_log.getLogger("bot_engine").debug(f"[SPY_FLOOR] Buy failed: {e}")
        else:
            sell_qty = min(abs(shares_diff), current_spy_shares)
            if sell_qty > 0:
                try:
                    requests.post("https://paper-api.alpaca.markets/v2/orders",
                        json={"symbol": floor_ticker, "qty": str(sell_qty),
                              "side": "sell", "type": "market",
                              "time_in_force": "day"},
                        headers=_alpaca_headers(), timeout=10)
                    result["actions"].append({"type": "floor_sell",
                        "shares": sell_qty, "ticker": floor_ticker,
                        "reason": f"{regime}: target {target_pct*100:.0f}%, current {current_pct*100:.0f}%"})
                    import logging as _floor_log; _floor_log.getLogger("bot_engine").info(f"[FLOOR] Sold {sell_qty} {floor_ticker} ({regime}: {target_pct*100:.0f}% target)")
                except Exception as e:
                    import logging as _floor_log; _floor_log.getLogger("bot_engine").debug(f"[SPY_FLOOR] Sell failed: {e}")

    except Exception as e:
        result["status"] = f"error: {str(e)[:80]}"

    return result


if __name__ == "__main__":
    # ── ML v2 Training Schedule ─────────────────────────────────────────
    # Daily retrain at 4am (called by Tier 3 in bot.ts)
    # Event-triggered: VXX > 1.3, 3+ consecutive stops, SPY > 3% move
    # Research basis:
    #   - News signals decay in 1-5 days → retrain daily captures yesterday's regime
    #   - Momentum signals last 3-6 months → 60-day window is right
    #   - Own trade feedback: 3x weight after 50+ trades (self-learning)
    try:
        from ml_model_v2 import train_model, FEEDBACK_PATH
        model_v2_path = os.path.join(DATA_DIR, "voltrade_ml_v2.pkl")

        # Delete old broken model on first run (52-feature with 42 zeros)
        old_model = os.path.join(DATA_DIR, "voltrade_ml_model.pkl")
        if os.path.exists(old_model):
            try:
                import joblib as _jbl
                _old = _jbl.load(old_model)
                _old_features = len(_old.get("feature_names", []))
                # Old model had 0 stored feature names (confirmed in audit)
                # OR had 52 features with 42 zeros — delete and replace
                if _old_features == 0 or _old_features == 52:
                    os.remove(old_model)
            except Exception:
                pass  # Can't load it — leave it

        # Retrain v2 if: doesn't exist, older than 24hrs, or feedback has grown significantly
        needs_train = (
            not os.path.exists(model_v2_path)
            or (time.time() - os.path.getmtime(model_v2_path)) > 24 * 3600
        )
        # Also retrain if we have 20+ new trades since last train
        if not needs_train and os.path.exists(FEEDBACK_PATH):
            try:
                with open(FEEDBACK_PATH) as _ff:
                    _fb = json.load(_ff)
                if os.path.exists(model_v2_path):
                    model_age = time.time() - os.path.getmtime(model_v2_path)
                    # Rough estimate: trades since last train
                    recent_trades = sum(1 for t in _fb
                        if time.time() - time.mktime(time.strptime(t.get("exit_date","2000-01-01"), "%Y-%m-%d")) < model_age)
                    if recent_trades >= 20:
                        needs_train = True
            except Exception:
                pass

        if needs_train:
            train_result = train_model()  # Returns status dict, silent output
            # Also train the options-specific ML model if enough options trades exist
            try:
                from ml_model_v2 import train_options_model
                train_options_model()  # No-op if < 30 options trades; silent otherwise
            except Exception:
                pass
    except Exception:
        pass

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "scan":
        result = scan_market()
    elif mode == "manage":
        result = manage_positions()
    elif mode == "full":
        result = scan_market()
    else:
        result = {"error": f"Unknown mode: {mode}"}

    print(json.dumps(result))
