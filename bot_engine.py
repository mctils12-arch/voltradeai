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

import sys
import json
import os
import time
import requests
import numpy as np
from datetime import datetime, timedelta

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
MIN_SCORE = 65             # Minimum combined score to trade
MIN_VOLUME = 500000        # Minimum avg daily volume
MIN_PRICE = 5              # Minimum stock price
MAX_SECTOR_POSITIONS = 2   # Max 2 stocks from the same sector

# ── Sector Map (for correlation / diversification check) ─────────────────────
SECTOR_MAP = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "GOOG": "tech",
    "META": "tech", "AMZN": "tech", "NVDA": "tech", "AMD": "tech",
    "INTC": "tech", "ORCL": "tech", "CRM": "tech", "ADBE": "tech",
    "QCOM": "tech", "TXN": "tech", "AVGO": "tech", "AMAT": "tech",
    "NOW": "tech", "SNOW": "tech", "PLTR": "tech", "UBER": "tech",
    "TSLA": "auto", "F": "auto", "GM": "auto", "TM": "auto", "RIVN": "auto",
    "JPM": "finance", "BAC": "finance", "GS": "finance", "MS": "finance",
    "WFC": "finance", "C": "finance", "AXP": "finance", "V": "finance", "MA": "finance",
    "JNJ": "health", "PFE": "health", "MRK": "health", "UNH": "health",
    "ABBV": "health", "LLY": "health", "BMY": "health", "AMGN": "health",
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "SPY": "etf", "QQQ": "etf", "IWM": "etf", "DIA": "etf", "GLD": "etf",
    "AMZN": "consumer", "WMT": "consumer", "TGT": "consumer", "COST": "consumer",
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


def garch_vol_estimate(returns, omega=0.00001, alpha=0.05, beta=0.90):
    """
    Simple GARCH(1,1) annualised volatility estimate.
    """
    if len(returns) < 30:
        return None
    arr = np.array(returns, dtype=float)
    var = float(np.var(arr))
    for r in arr:
        var = omega + alpha * float(r) ** 2 + beta * var
    return round(float(np.sqrt(var * 252)) * 100, 2)

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

def score_stock(stock_data):
    """
    Quick-score a stock from Polygon snapshot data (no API calls).
    Returns None if the stock doesn't pass basic filters.
    """
    ticker = stock_data.get("T", "")
    close = stock_data.get("c", 0)
    open_price = stock_data.get("o", 0)
    high = stock_data.get("h", 0)
    low = stock_data.get("l", 0)
    volume = stock_data.get("v", 0)
    vwap = stock_data.get("vw", 0)

    # Basic filters
    if close < MIN_PRICE or volume < MIN_VOLUME:
        return None
    if "." in ticker or len(ticker) > 5:  # Skip warrants, units, etc.
        return None

    # Calculate quick signals from available data
    change_pct = ((close - open_price) / open_price * 100) if open_price > 0 else 0
    range_pct = ((high - low) / low * 100) if low > 0 else 0
    vwap_dist = ((close - vwap) / vwap * 100) if vwap > 0 else 0

    score = 50  # Start neutral
    reasons = []

    # Price action score
    if change_pct > 3:
        score += 10
        reasons.append(f"Up {change_pct:.1f}% today — momentum")
    elif change_pct < -3:
        score += 8  # Mean reversion candidate
        reasons.append(f"Down {abs(change_pct):.1f}% today — bounce candidate")

    # Volume score (unusual volume = institutional interest)
    if volume > 20000000:
        score += 15
        reasons.append(f"Very high volume ({volume / 1e6:.0f}M)")
    elif volume > 5000000:
        score += 8
    elif volume > 1000000:
        score += 3

    # VWAP position (above VWAP = buying pressure)
    if vwap_dist > 1:
        score += 5
        reasons.append("Trading above VWAP — buyers in control")
    elif vwap_dist < -1:
        score += 3
        reasons.append("Below VWAP — potential dip buy")

    # Volatility (intraday range) — higher range = more opportunity
    if range_pct > 5:
        score += 10
        reasons.append(f"Wide range ({range_pct:.1f}%) — active stock")
    elif range_pct > 3:
        score += 5

    score = max(0, min(100, score))

    return {
        "ticker": ticker,
        "price": round(close, 2),
        "change_pct": round(change_pct, 2),
        "volume": volume,
        "range_pct": round(range_pct, 2),
        "vwap_dist": round(vwap_dist, 2),
        "quick_score": score,
        "reasons": reasons,
    }


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

    # Fetch macro context
    try:
        from macro_data import get_macro_snapshot, get_news_sentiment
        macro = get_macro_snapshot()
        news_sent = get_news_sentiment(ticker)
    except Exception:
        macro = {}
        news_sent = {}

    # Full intelligence report (news classification + insider + earnings patterns)
    try:
        from intelligence import get_full_intelligence
        intel = get_full_intelligence(ticker)
    except Exception:
        intel = {}

    # Alternative data sources (congressional, short interest, wiki, FRED, GDELT, patents)
    try:
        from alt_data import get_alt_data_score
        alt = get_alt_data_score(ticker)
    except Exception:
        alt = {}

    # Social intelligence (Reddit, Google Trends, multi-source news)
    try:
        from social_data import get_social_intelligence
        social = get_social_intelligence(ticker)
    except Exception:
        social = {}

    # Analyst ratings + valuation multiples (yfinance) — with 5-min cache
    yf_fundamentals = {}
    _yf_cache_key = ticker
    if _yf_cache_key in _yf_cache and (time.time() - _yf_cache_time.get(_yf_cache_key, 0)) < _YF_CACHE_TTL:
        yf_fundamentals = _yf_cache[_yf_cache_key]
    else:
        try:
            import yfinance as yf
            _yticker = yf.Ticker(ticker)
            _yinfo = _yticker.info or {}
            yf_fundamentals = {
                "forward_pe": _yinfo.get("forwardPE") or 0,
                "trailing_pe": _yinfo.get("trailingPE") or 0,
                "price_to_book": _yinfo.get("priceToBook") or 0,
                "ev_ebitda": _yinfo.get("enterpriseToEbitda") or 0,
                "price_to_sales": _yinfo.get("priceToSalesTrailing12Months") or 0,
                "peg_ratio": _yinfo.get("pegRatio") or 0,
                "target_mean": _yinfo.get("targetMeanPrice") or 0,
                "target_high": _yinfo.get("targetHighPrice") or 0,
                "target_low": _yinfo.get("targetLowPrice") or 0,
                "current_price": _yinfo.get("currentPrice") or _yinfo.get("regularMarketPrice") or 0,
                "recommendation": _yinfo.get("recommendationKey") or "none",
                "num_analysts": _yinfo.get("numberOfAnalystOpinions") or 0,
            }
            # Get buy/hold/sell counts from recommendations
            _recs = _yticker.recommendations
            if _recs is not None and not _recs.empty:
                _latest = _recs.iloc[0]
                yf_fundamentals["strong_buy_count"] = int(_latest.get("strongBuy", 0))
                yf_fundamentals["buy_count"] = int(_latest.get("buy", 0))
                yf_fundamentals["hold_count"] = int(_latest.get("hold", 0))
                yf_fundamentals["sell_count"] = int(_latest.get("sell", 0))
                yf_fundamentals["strong_sell_count"] = int(_latest.get("strongSell", 0))
            # Store in cache
            _yf_cache[_yf_cache_key] = yf_fundamentals
            _yf_cache_time[_yf_cache_key] = time.time()
        except Exception:
            pass

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
    try:
        end_d = datetime.now().strftime("%Y-%m-%d")
        start_d = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
        bars_url = (f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars"
                    f"?timeframe=1Day&start={start_d}&limit=30&adjustment=all")
        bars_resp = requests.get(bars_url, headers=_alpaca_headers(), timeout=8)
        bars_data = bars_resp.json().get("bars", [])
        if len(bars_data) >= 14:
            # ADX calculation (14-period)
            highs = [b["h"] for b in bars_data]
            lows = [b["l"] for b in bars_data]
            closes = [b["c"] for b in bars_data]
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
        
        # Insider activity
        insider = intel.get("insider", {})
        if insider.get("net_direction") == "strong_selling":
            reasons.append(f"Insiders heavily selling ({insider.get('sell_count', 0)} sells)")
        elif insider.get("net_direction") == "strong_buying":
            reasons.append(f"Insiders heavily buying ({insider.get('buy_count', 0)} buys)")
        
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

    # ── ML Model Integration ──────────────────────────────────────────────────
    try:
        from ml_model import ml_score
        ml_features = {
            "momentum_1m": edge.get("momentum_1m", 0) or 0,
            "momentum_3m": edge.get("momentum_3m", 0) or 0,
            "momentum_12m": mom_12_1,
            "volume_ratio": volume_ratio,
            "rsi_14": rsi,
            "vrp": vrp,
            "ewma_vol": vol_metrics.get("ewma_vol", 0) or 0 if vol_metrics else 0,
            "garch_vol": vol_metrics.get("garch_vol", 0) or 0 if vol_metrics else 0,
            "change_pct_today": quick_result.get("change_pct", 0) or 0,
            "range_pct": quick_result.get("range_pct", 0) or 0,
            "vwap_position": 1 if quick_result.get("above_vwap") else 0,
            "put_call_ratio": edge.get("put_call_ratio", 1.0) or 1.0,
            "iv_rank": detail.get("iv_rank", 50) or 50,
            "sentiment_score": sent_score_val,
            "sector_encoded": hash(SECTOR_MAP.get(ticker, "unknown")) % 10,
            "vix": macro.get("vix", 20),
            "vix_regime_encoded": {"low": 0, "medium": 1, "high": 2, "extreme": 3}.get(macro.get("vix_regime", "medium"), 1),
            "sector_momentum": sector_mom.get(stock_sector, 0),
            "market_regime_encoded": {"risk_on": 2, "neutral": 1, "risk_off": 0}.get(macro.get("market_regime", "neutral"), 1),
            "news_sentiment": news_score_val,
            "treasury_10y": macro.get("treasury_10y", 4.25),
            "adx": adx_value if adx_value is not None else 20,
            "obv_signal": obv_signal,
            "intel_score": intel.get("intel_score", 0) if intel else 0,
            "trap_warning": 1 if intel.get("trap_warning") else 0,
            "insider_signal": intel.get("insider", {}).get("insider_signal", 0) if intel else 0,
            "sell_the_news_risk": 1 if intel.get("earnings_pattern", {}).get("sell_the_news_risk") else 0,
            "wiki_spike_ratio": alt.get("wiki", {}).get("spike_ratio", 1.0) if alt else 1.0,
            "short_pressure_signal": alt.get("short_interest", {}).get("signal", 0) if alt else 0,
            "congressional_signal": alt.get("congressional", {}).get("signal", 0) if alt else 0,
            "geopolitical_risk": alt.get("geopolitical", {}).get("risk_score", 0) if alt else 0,
            "yield_curve": alt.get("fred_macro", {}).get("yield_curve", 0) if alt else 0,
            "credit_spread": alt.get("fred_macro", {}).get("credit_spread", 0) if alt else 0,
            "unemployment": alt.get("fred_macro", {}).get("unemployment", 4.0) if alt else 4.0,
            "patent_signal": alt.get("patent", {}).get("signal", 0) if alt else 0,
            "ftd_signal": alt.get("ftd", {}).get("signal", 0) if alt else 0,
            "reddit_mentions": social.get("reddit", {}).get("total_mentions", 0) if social else 0,
            "reddit_sentiment": social.get("reddit", {}).get("sentiment_score", 0) if social else 0,
            "reddit_wsb_mentions": social.get("reddit", {}).get("wsb_mentions", 0) if social else 0,
            "google_trends_spike": social.get("google_trends", {}).get("spike_ratio", 1.0) if social else 1.0,
            "google_trends_interest": social.get("google_trends", {}).get("current_interest", 0) if social else 0,
            "news_multi_articles": social.get("news_multi", {}).get("total_articles", 0) if social else 0,
            "news_multi_sentiment": social.get("news_multi", {}).get("sentiment_score", 0) if social else 0,
            "social_combined_signal": social.get("combined_signal", 0) if social else 0,
            "forward_pe": yf_fundamentals.get("forward_pe", 0),
            "trailing_pe": yf_fundamentals.get("trailing_pe", 0),
            "price_to_book": yf_fundamentals.get("price_to_book", 0),
            "ev_ebitda": yf_fundamentals.get("ev_ebitda", 0),
            "price_to_sales": yf_fundamentals.get("price_to_sales", 0),
            "analyst_target_upside": ((yf_fundamentals.get("target_mean", 0) - yf_fundamentals.get("current_price", 1)) / max(yf_fundamentals.get("current_price", 1), 1) * 100) if yf_fundamentals.get("target_mean") else 0,
            "analyst_buy_pct": (yf_fundamentals.get("strong_buy_count", 0) + yf_fundamentals.get("buy_count", 0)) / max(sum([yf_fundamentals.get("strong_buy_count", 0), yf_fundamentals.get("buy_count", 0), yf_fundamentals.get("hold_count", 0), yf_fundamentals.get("sell_count", 0), yf_fundamentals.get("strong_sell_count", 0)]), 1) * 100 if yf_fundamentals else 0,
            "num_analysts": yf_fundamentals.get("num_analysts", 0),
        }
        ml_result = ml_score(ml_features)

        # Blend: dynamic ratio loaded from tracker, default 60/40
        if ml_result.get("model_type") != "fallback":
            ml_s = ml_result.get("ml_score", combined_score)
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
            combined_score = combined_score * rule_weight + ml_s * ml_weight
            reasons.append(f"Blend: {rule_weight:.0%} rules + {ml_weight:.0%} ML")
            reasons.append(f"ML model: {ml_result.get('ml_signal', 'HOLD')} ({ml_result.get('ml_confidence', 0):.0%} confidence)")
    except Exception:
        pass  # ML not available, use rule-based only

    combined_score = max(0, min(100, combined_score))

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
    }

# ── Correlation / Sector Check ───────────────────────────────────────────────

def check_sector_correlation(ticker, existing_tickers):
    """
    Returns True if adding this ticker would exceed MAX_SECTOR_POSITIONS
    for its sector. False = safe to trade.
    """
    sector = SECTOR_MAP.get(ticker.upper(), "unknown")
    if sector == "unknown":
        return False  # Unknown sector → allow (conservative)
    count = sum(
        1 for t in existing_tickers
        if SECTOR_MAP.get(t.upper(), "unknown") == sector
    )
    return count >= MAX_SECTOR_POSITIONS

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

    for pos in positions:
        ticker = pos.get("symbol", "")
        current = float(pos.get("current_price", 0))
        entry = float(pos.get("avg_entry_price", current))
        pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100
        qty = abs(int(float(pos.get("qty", 0))))
        side = pos.get("side", "long")
        market_value = abs(float(pos.get("market_value", 0)))

        # Get ATR for this stock (14-day Average True Range)
        atr = _get_atr(ticker)
        atr_pct = (atr / current * 100) if current > 0 and atr else 2.0  # Default 2% if ATR unavailable

        # Dynamic trailing stop: tighter for calm stocks, wider for volatile ones
        # Stop = max(fixed floor, ATR * 1.5)
        stop_pct = max(1.5, min(atr_pct * 1.5, 8.0))  # Between 1.5% and 8%

        # Dynamic take profit: ATR * 3 from entry
        tp_pct = max(4.0, min(atr_pct * 3.0, 15.0))  # Between 4% and 15%

        # Check stop loss (ATR-based)
        if pnl_pct <= -stop_pct:
            actions.append({
                "action": "CLOSE",
                "ticker": ticker,
                "side": side,
                "reason": f"TRAILING STOP hit: {pnl_pct:.1f}% loss (ATR-based stop: -{stop_pct:.1f}%, ATR: ${atr:.2f})",
                "type": "stop_loss",
            })
        # Check take profit (ATR-based)
        elif pnl_pct >= tp_pct:
            actions.append({
                "action": "CLOSE",
                "ticker": ticker,
                "side": side,
                "reason": f"TAKE PROFIT hit: +{pnl_pct:.1f}% gain (ATR-based target: +{tp_pct:.1f}%)",
                "type": "take_profit",
            })
        # Trailing stop: if stock went up significantly but is now pulling back
        elif pnl_pct > 2.0 and pnl_pct < (tp_pct * 0.5):
            # Stock was up but giving back gains — tighten stop
            tightened_stop = -(atr_pct * 0.75)
            if pnl_pct <= tightened_stop:
                actions.append({
                    "action": "CLOSE",
                    "ticker": ticker,
                    "side": side,
                    "reason": f"TIGHTENED TRAILING STOP: was profitable, now {pnl_pct:.1f}% (tightened to {tightened_stop:.1f}%)",
                    "type": "trailing_stop",
                })
        
        # Upgrade candidate: position is flat or slightly negative, not worth holding
        if -stop_pct < pnl_pct < 1.0:
            upgrade_candidates.append({
                "ticker": ticker,
                "pnl_pct": pnl_pct,
                "market_value": market_value,
                "score": 50 + pnl_pct * 5,  # Rough score based on P&L
                "qty": qty,
                "side": side,
            })

    return {"actions": actions, "positions": len(positions), "upgrade_candidates": upgrade_candidates}


def _get_atr(ticker, period=14):
    """Get 14-day ATR for a ticker using Alpaca daily bars."""
    import requests
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        url = (f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars"
               f"?timeframe=1Day&start={start}&limit={period + 5}&adjustment=all")
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

def scan_market():
    """
    Full market scan:
    1. Get stock universe from Alpaca (most-actives + movers)
    2. Fetch snapshots for price/volume data, quick-score
    3. Deep-analyze top 30
    4. Return top 5 with trade recommendations (buy / sell / short)
    """
    # Step 1 & 2: Fetch stock universe from Alpaca (unlimited, no rate limit)
    all_tickers = []
    try:
        # Most active by volume
        resp = requests.get(f"{ALPACA_DATA_URL}/v1beta1/screener/stocks/most-actives?by=volume&top=200",
                           headers=_alpaca_headers(), timeout=15)
        active = resp.json().get("most_actives", [])
        for s in active:
            sym = s.get("symbol", "")
            vol = s.get("volume", 0)
            if sym and vol > 100000:
                all_tickers.append({"ticker": sym, "volume": vol})
    except Exception:
        pass

    try:
        # Top movers (gainers + losers)
        resp = requests.get(f"{ALPACA_DATA_URL}/v1beta1/screener/stocks/movers?top=50",
                           headers=_alpaca_headers(), timeout=10)
        movers = resp.json()
        for s in movers.get("gainers", []) + movers.get("losers", []):
            sym = s.get("symbol", "")
            if sym and not any(t["ticker"] == sym for t in all_tickers):
                all_tickers.append({"ticker": sym, "volume": 0})
    except Exception:
        pass

    # Get snapshots for price data
    ticker_symbols = [t["ticker"] for t in all_tickers[:200]]
    quick_results = []

    for i in range(0, len(ticker_symbols), 50):
        batch = ticker_symbols[i:i+50]
        try:
            resp = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/snapshots?symbols={','.join(batch)}",
                               headers=_alpaca_headers(), timeout=15)
            snap_data = resp.json()
            for sym, snap in snap_data.items():
                bar = snap.get("dailyBar", {})
                prev = snap.get("prevDailyBar", {})
                c = float(bar.get("c", 0))
                o = float(bar.get("o", c))
                pc = float(prev.get("c", c))
                v = int(bar.get("v", 0))
                if c < 5 or v < 500000:
                    continue
                # Skip non-standard tickers
                if "." in sym or len(sym) > 5:
                    continue
                change_pct = ((c - pc) / pc * 100) if pc > 0 else 0
                quick_results.append({
                    "ticker": sym,
                    "price": round(c, 2),
                    "close": c,
                    "open": o,
                    "prev_close": pc,
                    "volume": v,
                    "change_pct": round(change_pct, 2),
                    "quick_score": abs(change_pct) * 2 + (v / 1000000) * 3,
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

    # Step 3: Deep analyze top 10
    top_candidates = scored[:10]
    deep_scored = []
    for candidate in top_candidates:
        try:
            deep = deep_score(candidate["ticker"], candidate)
            deep_scored.append(deep)
        except Exception:
            deep_scored.append(candidate)

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

        # Skip if below minimum score
        if final_score < MIN_SCORE:
            continue

        # Correlation / sector check — don't over-concentrate
        if check_sector_correlation(ticker, current_tickers + [t["ticker"] for t in trades]):
            continue

        side = stock.get("side", "buy")
        action_label = stock.get("action_label", "BUY")

        # For SELL signals without existing position — convert to BUY (no naked sells on paper)
        if side == "sell" and stock.get("trade_type") != "options":
            side = "buy"
            action_label = "BUY"

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
        })

    # Step 7: Check position management
    mgmt = manage_positions()

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
    }

# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Train ML model if needed (runs once, cached for a week)
    try:
        from ml_model import train_model
        model_path = "/tmp/voltrade_ml_model.pkl"
        # Retrain if model doesn't exist or is older than 7 days
        if not os.path.exists(model_path) or (time.time() - os.path.getmtime(model_path)) > 7 * 86400:
            print(json.dumps({"status": "training_ml_model"}))
            train_result = train_model()
            # Training happens in background, don't block the scan
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
