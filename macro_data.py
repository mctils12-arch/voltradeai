"""
Macro market data + news sentiment for VolTradeAI.
All free sources, cached to avoid rate limits.
"""
import json
import os
import time
import requests

POLYGON_KEY = os.environ.get("POLYGON_KEY", "UNwTHo3kvZMBckeIaHQbBLuaaURmFUQP")
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
            "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", "PKMDHJOVQEVIB4UHZXUYVTIDBU"),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et"),
        }
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
            "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", "PKMDHJOVQEVIB4UHZXUYVTIDBU"),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et"),
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
