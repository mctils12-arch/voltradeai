"""
Macro market data + news sentiment for VolTradeAI.
All free sources, cached to avoid rate limits.
"""
import json
import os
import time
import requests
# Alpaca rate limiter — prevents silent 429 errors during parallel scans
try:
    from alpaca_rate_limiter import alpaca_throttle
    _HAS_ALPACA_THROTTLE = True
except ImportError:
    _HAS_ALPACA_THROTTLE = False
    class _NoThrottle:
        def acquire(self): pass
    alpaca_throttle = _NoThrottle()
from datetime import datetime, timedelta

POLYGON_KEY = os.environ.get("POLYGON_KEY", "")
CACHE_PATH = "/tmp/voltrade_macro_cache.json"
CACHE_TTL = 300  # 5 minutes

# Sector ETF → sector name mapping
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLC": "Communications",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}

# Simple keyword sentiment scoring (no LLM needed — fast and free)
POSITIVE_WORDS = {"beat", "beats", "surge", "surges", "rally", "rallies", "upgrade", "upgrades",
                  "bullish", "record", "high", "profit", "growth", "strong", "outperform",
                  "buy", "raise", "raises", "positive", "boom", "soar", "soars", "jumps",
                  "exceeds", "exceeded", "above", "optimistic", "breakthrough", "approval",
                  "dividend", "buyback", "expands", "partnership", "deal", "acquisition"}
NEGATIVE_WORDS = {"miss", "misses", "drop", "drops", "crash", "crashes", "downgrade", "downgrades",
                  "bearish", "low", "loss", "losses", "weak", "underperform", "sell",
                  "cut", "cuts", "negative", "bust", "plunge", "plunges", "falls",
                  "below", "pessimistic", "recall", "lawsuit", "fraud", "investigation",
                  "bankruptcy", "layoff", "layoffs", "warning", "decline", "declines"}


def _load_cache():
    """Load cached macro data."""
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                cache = json.load(f)
            if time.time() - cache.get("timestamp", 0) < CACHE_TTL:
                return cache
    except Exception:
        pass
    return None


def _save_cache(data):
    """Save macro data to cache."""
    data["timestamp"] = time.time()
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def get_macro_snapshot() -> dict:
    """
    Get current macro market conditions.
    Returns: {
        vix: float, vix_regime: "low"/"medium"/"high"/"extreme",
        treasury_10y: float, yield_trend: "rising"/"falling"/"flat",
        dollar_index: float,
        sector_momentum: {sector: pct_change, ...},
        strongest_sector: str, weakest_sector: str,
        market_regime: "risk_on"/"risk_off"/"neutral"
    }
    """
    cached = _load_cache()
    if cached and "vix" in cached:
        return cached

    result = {}

    # VIX via yfinance
    try:
        import yfinance as yf
        vix_data = yf.Ticker("^VIX").history(period="5d")
        if not vix_data.empty:
            result["vix"] = round(float(vix_data["Close"].iloc[-1]), 2)
            vix_prev = float(vix_data["Close"].iloc[-2]) if len(vix_data) > 1 else result["vix"]
            result["vix_change"] = round(result["vix"] - vix_prev, 2)
        else:
            result["vix"] = 20.0  # default
            result["vix_change"] = 0
    except Exception:
        result["vix"] = 20.0
        result["vix_change"] = 0

    # VIX regime
    vix = result["vix"]
    if vix < 15:
        result["vix_regime"] = "low"
    elif vix < 20:
        result["vix_regime"] = "medium"
    elif vix < 30:
        result["vix_regime"] = "high"
    else:
        result["vix_regime"] = "extreme"

    # Treasury 10Y yield via yfinance
    try:
        import yfinance as yf
        tnx = yf.Ticker("^TNX").history(period="5d")
        if not tnx.empty:
            result["treasury_10y"] = round(float(tnx["Close"].iloc[-1]), 3)
            tnx_prev = float(tnx["Close"].iloc[-2]) if len(tnx) > 1 else result["treasury_10y"]
            diff = result["treasury_10y"] - tnx_prev
            result["yield_trend"] = "rising" if diff > 0.02 else "falling" if diff < -0.02 else "flat"
        else:
            result["treasury_10y"] = 4.25
            result["yield_trend"] = "flat"
    except Exception:
        result["treasury_10y"] = 4.25
        result["yield_trend"] = "flat"

    # Dollar index via yfinance
    try:
        import yfinance as yf
        dx = yf.Ticker("DX-Y.NYB").history(period="5d")
        if not dx.empty:
            result["dollar_index"] = round(float(dx["Close"].iloc[-1]), 2)
        else:
            result["dollar_index"] = 104.0
    except Exception:
        result["dollar_index"] = 104.0

    # Sector ETF momentum from Alpaca (single batch call instead of 11 Polygon calls)
    sector_momentum = {}
    try:
        etf_symbols = ",".join(SECTOR_ETFS.keys())
        alpaca_url = f"https://data.alpaca.markets/v2/stocks/snapshots?symbols={etf_symbols}&feed=sip"
        alpaca_headers = {
            "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
        }
        resp = alpaca_throttle.acquire()
        resp = requests.get(alpaca_url, headers=alpaca_headers, timeout=10)
        snap_data = resp.json()
        for etf, sector in SECTOR_ETFS.items():
            snap = snap_data.get(etf, {})
            bar = snap.get("dailyBar", {})
            prev = snap.get("prevDailyBar", {})
            c = float(bar.get("c", 0))
            pc = float(prev.get("c", c))
            pct = ((c - pc) / pc * 100) if pc > 0 else 0
            sector_momentum[sector] = round(pct, 2)
    except Exception:
        for sector in SECTOR_ETFS.values():
            sector_momentum[sector] = 0.0

    result["sector_momentum"] = sector_momentum

    # Strongest and weakest sectors
    if sector_momentum:
        sorted_sectors = sorted(sector_momentum.items(), key=lambda x: x[1])
        result["weakest_sector"] = sorted_sectors[0][0]
        result["strongest_sector"] = sorted_sectors[-1][0]
    else:
        result["strongest_sector"] = "Unknown"
        result["weakest_sector"] = "Unknown"

    # Market regime: risk_on if VIX low + sectors broadly green, risk_off if VIX high + red
    green_count = sum(1 for v in sector_momentum.values() if v > 0)
    total = len(sector_momentum) or 1
    if vix < 18 and green_count / total > 0.6:
        result["market_regime"] = "risk_on"
    elif vix > 25 or green_count / total < 0.3:
        result["market_regime"] = "risk_off"
    else:
        result["market_regime"] = "neutral"

    # ── Fix B: SPY 200-day MA slow-bear detector (v1.0.22) ──────────────────
    # Count consecutive trading days SPY has closed below its 200-day MA.
    # Passed to get_market_regime() as spy_below_200_days.
    # When >= 10, forces BEAR regime regardless of VXX level.
    # This catches the 2022-style slow grinding bear that VXX ratio missed.
    try:
        import requests as _req
        _alpaca_key    = os.environ.get("ALPACA_KEY", "")
        _alpaca_secret = os.environ.get("ALPACA_SECRET", "")
        _h = {"APCA-API-KEY-ID": _alpaca_key, "APCA-API-SECRET-KEY": _alpaca_secret}
        _r = _req.get("https://data.alpaca.markets/v2/stocks/bars",
            params={"symbols": "SPY", "timeframe": "1Day",
                    "start": (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d"),
                    "limit": 220, "feed": "sip"},
            headers=_h, timeout=8)
        _spy_bars = _r.json().get("bars", {}).get("SPY", [])
        if len(_spy_bars) >= 200:
            _closes  = [float(b["c"]) for b in _spy_bars]
            _ma200   = sum(_closes[-200:]) / 200
            # Count consecutive days below 200d MA (from most recent backwards)
            _streak  = 0
            for _c in reversed(_closes):
                if _c < _ma200:
                    _streak += 1
                else:
                    break
            spy_close = _closes[-1]
            spy_ma200 = _ma200
            result["spy_below_200_days"] = _streak
            result["spy_vs_ma200"]       = round(spy_close / spy_ma200, 4)
            result["spy_above_200d"]     = spy_close > spy_ma200
        else:
            result["spy_below_200_days"] = 0
            result["spy_vs_ma200"]       = 1.0
            result["spy_above_200d"]     = True
    except Exception:
        result["spy_below_200_days"] = 0
        result["spy_vs_ma200"]       = 1.0
        result["spy_above_200d"]     = True

    # ── SYSTEM FIX 2026-04-20: Add vxx_ratio and spy_vs_ma50 ──
    # These are load-bearing inputs for tiered_strategy, options_scanner,
    # ml_model_v2, and bot_engine regime logic. Previously NOT returned
    # here, causing every consumer to fall back to default 1.0 silently.
    # With this fix, all consumers get real values from one source.

    # VXX ratio — current VXX / 30-day average (panic gauge)
    try:
        _alpaca_h = {
            "APCA-API-KEY-ID":     os.environ.get("ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
        }
        _vxx_start = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
        alpaca_throttle.acquire()
        _vxx_r = requests.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            params={"symbols": "VXX", "timeframe": "1Day",
                    "start": _vxx_start, "limit": 40, "feed": "sip"},
            headers=_alpaca_h, timeout=8,
        )
        _vxx_bars = _vxx_r.json().get("bars", {}).get("VXX", [])
        if _vxx_bars and len(_vxx_bars) >= 5:
            _vxx_closes = [float(b["c"]) for b in _vxx_bars]
            _vxx_avg30 = sum(_vxx_closes[-30:]) / len(_vxx_closes[-30:])
            _vxx_latest = _vxx_closes[-1]
            result["vxx_ratio"] = round(_vxx_latest / _vxx_avg30, 4) if _vxx_avg30 > 0 else 1.0
            result["vxx_latest"] = round(_vxx_latest, 2)
            result["vxx_avg30"] = round(_vxx_avg30, 2)
            # OPTIMIZATION 2026-04-20: expose raw closes so bot_engine can
            # compute vol_of_vol (std of last 10 returns) without re-fetching.
            # Saves one Alpaca call per scan cycle.
            result["vxx_closes"] = [round(c, 3) for c in _vxx_closes]
        else:
            result["vxx_ratio"] = 1.0
            result["vxx_closes"] = []
    except Exception as _regime_err:
        import logging
        logging.getLogger("voltrade.macro").debug(f"regime detection failed: {_regime_err}")
        result["vxx_ratio"] = 1.0

    # SPY / 50-day MA — trend gauge
    try:
        # Reuse SPY bars from above if already fetched; else fetch
        _spy_closes_local = None
        if "spy_vs_ma200" in result:
            # We already fetched 300 days of SPY above for MA200 calc — reuse
            try:
                _spy_start = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")
                alpaca_throttle.acquire()
                _spy_r = requests.get(
                    "https://data.alpaca.markets/v2/stocks/bars",
                    params={"symbols": "SPY", "timeframe": "1Day",
                            "start": _spy_start, "limit": 220, "feed": "sip"},
                    headers=_alpaca_h, timeout=8,
                )
                _spy_bars = _spy_r.json().get("bars", {}).get("SPY", [])
                if _spy_bars and len(_spy_bars) >= 50:
                    _spy_closes_local = [float(b["c"]) for b in _spy_bars]
            except Exception:
                pass

        if _spy_closes_local:
            _spy_ma50 = sum(_spy_closes_local[-50:]) / 50
            _spy_last = _spy_closes_local[-1]
            result["spy_vs_ma50"] = round(_spy_last / _spy_ma50, 4) if _spy_ma50 > 0 else 1.0
            result["spy_ma50"] = round(_spy_ma50, 2)
            result["spy_close"] = round(_spy_last, 2)
            # OPTIMIZATION: expose last 60 closes for callers that need them
            result["spy_closes_60d"] = [round(c, 2) for c in _spy_closes_local[-60:]]
        else:
            result["spy_vs_ma50"] = 1.0
            result["spy_closes_60d"] = []
    except Exception:
        result["spy_vs_ma50"] = 1.0

    # HOTFIX 2026-04-22: compute and set "regime" key that downstream
    # consumers expect (probability_engine, stress_index, UI). Previously
    # only vix_regime and market_regime were set — regime was left as None.
    _vxx_r = float(result.get("vxx_ratio", 1.0) or 1.0)
    try:
        from system_config import get_market_regime as _gmr_macro
        _spy_ma = float(result.get("spy_vs_ma50", 1.0) or 1.0)
        _spy_b200 = int(result.get("spy_below_200_days", 0) or 0)
        _spy_above = bool(result.get("spy_above_200d", True))
        result["regime"] = _gmr_macro(_vxx_r, _spy_ma,
                                      spy_below_200_days=_spy_b200,
                                      spy_above_200d=_spy_above)
    except Exception as _reg_err:
        import logging
        logging.getLogger("voltrade.macro").debug(f"regime derivation failed: {_reg_err}")
        # Fallback: map VXX ratio to canonical regime labels
        if _vxx_r >= 1.30:
            result["regime"] = "PANIC"
        elif _vxx_r >= 1.15:
            result["regime"] = "BEAR"
        elif _vxx_r >= 1.05:
            result["regime"] = "CAUTION"
        elif _vxx_r <= 0.90:
            result["regime"] = "BULL"
        else:
            result["regime"] = "NEUTRAL"

    # DATA QUALITY FLAG (added 2026-04-20): tiers that depend on vxx/spy
    # should skip execution if we had to fall back to defaults (all 1.0s).
    # Consumers check macro.get("data_quality") == "degraded" to decide.
    _defaults_used = 0
    if result.get("vxx_ratio", 1.0) == 1.0 and not result.get("vxx_closes"):
        _defaults_used += 1
    if result.get("spy_vs_ma50", 1.0) == 1.0 and not result.get("spy_closes_60d"):
        _defaults_used += 1
    if not result.get("sector_momentum"):
        _defaults_used += 1
    if _defaults_used >= 2:
        result["data_quality"] = "degraded"
        result["data_quality_reason"] = f"{_defaults_used} critical fields at defaults"
    else:
        result["data_quality"] = "ok"

    # ITEM 17 FIX 2026-04-20: Data quality flag
    # Rates the snapshot's trustworthiness. Consumers can gate execution
    # on this — e.g. if quality == "degraded", tiered strategy should
    # skip Tier 2 leverage since regime detection may be unreliable.
    _quality_score = 0
    _quality_failures = []
    # Check each critical field
    if result.get("vix", 0) > 0:
        _quality_score += 25
    else:
        _quality_failures.append("vix")
    if result.get("vxx_ratio") is not None and result.get("vxx_closes"):
        _quality_score += 30
    else:
        _quality_failures.append("vxx_ratio")
    if result.get("spy_vs_ma50") is not None and result.get("spy_vs_ma200") is not None:
        _quality_score += 25
    else:
        _quality_failures.append("spy_ma")
    if result.get("sector_momentum") and len(result.get("sector_momentum", {})) >= 8:
        _quality_score += 20
    else:
        _quality_failures.append("sectors")

    if _quality_score >= 90:
        result["data_quality"] = "good"
    elif _quality_score >= 60:
        result["data_quality"] = "degraded"
    else:
        result["data_quality"] = "poor"
    result["data_quality_score"] = _quality_score
    result["data_quality_failures"] = _quality_failures

    _save_cache(result)
    return result


def get_news_sentiment(ticker: str) -> dict:
    """
    Score news sentiment for a ticker using Polygon news headlines.
    Returns: {
        sentiment_score: -100 to +100,
        sentiment_label: "very_bullish"/"bullish"/"neutral"/"bearish"/"very_bearish",
        headline_count: int,
        top_headline: str,
        positive_count: int,
        negative_count: int,
    }
    """
    try:
        alpaca_headers = {
            "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
        }
        url = f"https://data.alpaca.markets/v1beta1/news?limit=10&sort=desc&symbols={ticker}"
        resp = requests.get(url, headers=alpaca_headers, timeout=8)
        data = resp.json()
        articles = data.get("news", [])
        # Transform: Alpaca uses "headline" and "summary" instead of "title" and "description"
        articles = [{"title": a.get("headline", ""), "description": a.get("summary", "")} for a in articles]
    except Exception:
        return {"sentiment_score": 0, "sentiment_label": "neutral", "headline_count": 0,
                "top_headline": "", "positive_count": 0, "negative_count": 0}

    if not articles:
        return {"sentiment_score": 0, "sentiment_label": "neutral", "headline_count": 0,
                "top_headline": "", "positive_count": 0, "negative_count": 0}

    pos_count = 0
    neg_count = 0

    for article in articles:
        title = (article.get("title", "") + " " + article.get("description", "")).lower()
        words = set(title.split())

        pos_hits = len(words & POSITIVE_WORDS)
        neg_hits = len(words & NEGATIVE_WORDS)

        if pos_hits > neg_hits:
            pos_count += 1
        elif neg_hits > pos_hits:
            neg_count += 1

    total = len(articles)
    # Score: ranges from -100 to +100
    if total > 0:
        score = round(((pos_count - neg_count) / total) * 100)
    else:
        score = 0

    # Label
    if score >= 60:
        label = "very_bullish"
    elif score >= 20:
        label = "bullish"
    elif score <= -60:
        label = "very_bearish"
    elif score <= -20:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "sentiment_score": score,
        "sentiment_label": label,
        "headline_count": total,
        "top_headline": articles[0].get("title", "") if articles else "",
        "positive_count": pos_count,
        "negative_count": neg_count,
    }


def get_sector_for_ticker(ticker: str) -> str:
    """Get sector from Polygon ticker details."""
    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={POLYGON_KEY}"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        sic = data.get("results", {}).get("sic_description", "")
        # Map SIC descriptions to broad sectors
        sic_lower = sic.lower()
        if any(w in sic_lower for w in ["software", "computer", "electronic", "semiconductor", "internet"]):
            return "Technology"
        elif any(w in sic_lower for w in ["bank", "insurance", "investment", "finance", "credit"]):
            return "Financials"
        elif any(w in sic_lower for w in ["oil", "gas", "petroleum", "coal", "mining"]):
            return "Energy"
        elif any(w in sic_lower for w in ["pharma", "biotech", "medical", "health", "hospital"]):
            return "Healthcare"
        elif any(w in sic_lower for w in ["auto", "aerospace", "industrial", "machinery", "defense"]):
            return "Industrials"
        elif any(w in sic_lower for w in ["telecom", "media", "broadcast", "entertainment"]):
            return "Communications"
        elif any(w in sic_lower for w in ["retail", "restaurant", "hotel", "leisure", "apparel"]):
            return "Consumer Discretionary"
        elif any(w in sic_lower for w in ["food", "beverage", "tobacco", "household"]):
            return "Consumer Staples"
        elif any(w in sic_lower for w in ["electric", "water", "utility"]):
            return "Utilities"
        elif any(w in sic_lower for w in ["real estate", "reit"]):
            return "Real Estate"
        else:
            return "Other"
    except Exception:
        return "Other"


if __name__ == "__main__":
    print("=== Macro Snapshot ===")
    macro = get_macro_snapshot()
    print(json.dumps(macro, indent=2))
    print("\n=== News Sentiment (AAPL) ===")
    news = get_news_sentiment("AAPL")
    print(json.dumps(news, indent=2))
    print("\n=== Sector for AAPL ===")
    print(get_sector_for_ticker("AAPL"))
