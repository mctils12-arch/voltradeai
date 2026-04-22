"""
VolTradeAI Alternative Data Sources
All free APIs with aggressive caching to avoid rate limits.
"""
import json
import os
import time
import re
import requests
from datetime import datetime, timedelta

POLYGON_KEY = os.environ.get("POLYGON_KEY", "")
CACHE_DIR = "/tmp/voltrade_alt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

USER_AGENT = "VolTradeAI research@voltradeai.com"

def _cache_get(key, ttl_seconds=3600):
    """Read from cache if fresh."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_seconds:
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _cache_set(key, data):
    """
    Write to cache atomically.
    FIX 2026-04-20 (Bug #21): Previously a naive open+write could race when
    parallel deep_score workers write the same key concurrently. Now uses
    tempfile + os.replace for atomic write.
    """
    import tempfile
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        dirname = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".cache.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except Exception:
            try: os.unlink(tmp)
            except Exception: pass
    except Exception:
        pass


# ── 1. Wikipedia Pageview Spikes ──────────────────────────────────────────────

# Map tickers to Wikipedia article names
TICKER_TO_WIKI = {
    "AAPL": "Apple_Inc.", "TSLA": "Tesla,_Inc.", "MSFT": "Microsoft",
    "AMZN": "Amazon_(company)", "GOOGL": "Alphabet_Inc.", "META": "Meta_Platforms",
    "NVDA": "Nvidia", "AMD": "Advanced_Micro_Devices", "NFLX": "Netflix",
    "DIS": "The_Walt_Disney_Company", "BA": "Boeing", "JPM": "JPMorgan_Chase",
    "GS": "Goldman_Sachs", "V": "Visa_Inc.", "MA": "Mastercard",
    "PFE": "Pfizer", "JNJ": "Johnson_%26_Johnson", "UNH": "UnitedHealth_Group",
    "XOM": "ExxonMobil", "CVX": "Chevron_Corporation",
}

def get_wiki_attention(ticker: str) -> dict:
    """
    Check if a stock's Wikipedia page has unusual attention.
    Spike = current views > 2x the 30-day average.
    Cached for 4 hours.
    """
    cache_key = f"wiki_{ticker}"
    cached = _cache_get(cache_key, ttl_seconds=14400)
    if cached:
        return cached

    result = {"has_spike": False, "spike_ratio": 1.0, "recent_views": 0, "avg_views": 0, "signal": 0}

    article = TICKER_TO_WIKI.get(ticker)
    if not article:
        # Try to construct from ticker — won't work for all but covers some
        _cache_set(cache_key, result)
        return result

    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=35)).strftime("%Y%m%d")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        data = resp.json()
        items = data.get("items", [])

        if len(items) >= 7:
            views = [i["views"] for i in items]
            avg_30d = sum(views[:-3]) / max(len(views) - 3, 1)
            recent_3d = sum(views[-3:]) / 3

            result["avg_views"] = int(avg_30d)
            result["recent_views"] = int(recent_3d)

            if avg_30d > 0:
                ratio = recent_3d / avg_30d
                result["spike_ratio"] = round(ratio, 2)
                if ratio > 2.5:
                    result["has_spike"] = True
                    result["signal"] = 15  # Strong attention signal
                elif ratio > 1.8:
                    result["has_spike"] = True
                    result["signal"] = 8
                elif ratio > 1.4:
                    result["signal"] = 3
                elif ratio < 0.5:
                    result["signal"] = -3  # Fading interest
    except Exception:
        pass

    _cache_set(cache_key, result)
    return result


# ── 2. FRED Macro Indicators (expanded) ──────────────────────────────────────

FRED_SERIES = {
    "unemployment": "UNRATE",           # Monthly unemployment rate
    "consumer_confidence": "UMCSENT",   # U of Michigan consumer sentiment
    "initial_claims": "ICSA",           # Weekly initial jobless claims
    "credit_spread": "BAMLC0A0CM",      # ICE BofA US Corporate Index Option-Adjusted Spread
    "inflation_expect": "MICH",         # Michigan inflation expectations
    "yield_curve": "T10Y2Y",            # 10Y-2Y treasury spread (inversion = recession signal)
    "fed_funds": "FEDFUNDS",            # Fed funds rate
}

def get_fred_macro() -> dict:
    """
    Fetch expanded macro indicators from FRED (CSV endpoint, no API key needed).
    Cached for 6 hours.
    """
    cache_key = "fred_macro_expanded"
    cached = _cache_get(cache_key, ttl_seconds=21600)
    if cached:
        return cached

    result = {}
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    for name, series_id in FRED_SERIES.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
            resp = requests.get(url, timeout=10)
            lines = resp.text.strip().split("\n")
            if len(lines) >= 2:
                # Get the most recent non-empty value
                for line in reversed(lines[1:]):
                    parts = line.split(",")
                    if len(parts) >= 2 and parts[1].strip() and parts[1].strip() != ".":
                        result[name] = float(parts[1].strip())
                        break
        except Exception as _fred_err:
            import logging
            logging.getLogger("voltrade.alt_data").debug(f"FRED fetch failed for {series_id}: {_fred_err}")

    # Derived signals
    yc = result.get("yield_curve")
    if yc is not None:
        result["yield_curve_inverted"] = yc < 0
        result["recession_signal"] = yc < -0.5  # Deep inversion = strong recession signal

    claims = result.get("initial_claims")
    if claims is not None:
        result["labor_market_weak"] = claims > 300000  # Above 300K = weakening

    spread = result.get("credit_spread")
    if spread is not None:
        result["credit_stress"] = spread > 4.0  # Above 4% = financial stress

    _cache_set(cache_key, result)
    return result


# ── 3. GDELT Global Risk Events ──────────────────────────────────────────────

RISK_QUERIES = [
    "tariff trade war sanctions",
    "military conflict war escalation",
    "bank crisis financial crisis",
    "pandemic outbreak disease",
    "oil supply disruption OPEC",
]

def get_geopolitical_risk() -> dict:
    """
    Check GDELT for geopolitical risk events that could move markets.
    Cached for 2 hours.
    """
    cache_key = "gdelt_risk"
    cached = _cache_get(cache_key, ttl_seconds=7200)
    if cached:
        return cached

    result = {
        "risk_level": "low",
        "risk_score": 0,
        "active_risks": [],
        "top_headline": "",
    }

    total_risk_articles = 0

    for query in RISK_QUERIES:
        try:
            url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={requests.utils.quote(query)}&mode=artlist&maxrecords=5&format=json&sourcelang=eng"
            resp = requests.get(url, timeout=8)
            data = resp.json()
            articles = data.get("articles", [])

            # Count recent articles (last 48 hours)
            recent = []
            cutoff = (datetime.now() - timedelta(hours=48)).strftime("%Y%m%d")
            for a in articles:
                date_str = str(a.get("seendate", ""))[:8]
                if date_str >= cutoff:
                    recent.append(a)

            if len(recent) >= 3:
                risk_type = query.split()[0]
                result["active_risks"].append(risk_type)
                total_risk_articles += len(recent)

                if not result["top_headline"] and recent:
                    result["top_headline"] = recent[0].get("title", "")[:100]
        except Exception:
            continue

        # Rate limit — don't hammer GDELT
        time.sleep(0.3)

    if total_risk_articles > 20:
        result["risk_level"] = "extreme"
        result["risk_score"] = -20
    elif total_risk_articles > 10:
        result["risk_level"] = "high"
        result["risk_score"] = -12
    elif total_risk_articles > 5:
        result["risk_level"] = "elevated"
        result["risk_score"] = -5
    else:
        result["risk_score"] = 0

    _cache_set(cache_key, result)
    return result


# ── 4. Short Interest Approximation ──────────────────────────────────────────

def get_short_interest(ticker: str) -> dict:
    """
    Approximate short interest from Polygon data and volume analysis.
    Real short interest from SEC/FINRA is delayed 2 weeks, so we use
    volume patterns as a proxy.
    Cached for 4 hours.
    """
    cache_key = f"short_{ticker}"
    cached = _cache_get(cache_key, ttl_seconds=14400)
    if cached:
        return cached

    result = {"short_pressure": "unknown", "signal": 0, "days_to_cover": 0}

    try:
        # Get recent volume and price data
        alpaca_headers = {
            "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
        }
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars?timeframe=1Day&start={start}&limit=30&adjustment=all&feed=sip"
        resp = requests.get(url, headers=alpaca_headers, timeout=8)
        bars = resp.json().get("bars", [])
        # Alpaca uses "c", "o", "h", "l", "v" same as Polygon

        if len(bars) >= 10:
            volumes = [b.get("v", 0) for b in bars]
            closes = [b.get("c", 0) for b in bars]

            avg_vol = sum(volumes) / len(volumes)
            recent_vol = sum(volumes[-5:]) / 5

            # Volume spike on down days = potential short activity
            down_day_vol = []
            up_day_vol = []
            for i in range(1, len(bars)):
                if closes[i] < closes[i-1]:
                    down_day_vol.append(volumes[i])
                else:
                    up_day_vol.append(volumes[i])

            avg_down_vol = sum(down_day_vol) / len(down_day_vol) if down_day_vol else 0
            avg_up_vol = sum(up_day_vol) / len(up_day_vol) if up_day_vol else 0

            # Short pressure indicator
            if avg_down_vol > avg_up_vol * 1.5 and avg_down_vol > 0:
                result["short_pressure"] = "high"
                result["signal"] = -8
                # Rough days to cover estimate
                result["days_to_cover"] = round(avg_down_vol / avg_vol * 3, 1)
            elif avg_down_vol > avg_up_vol * 1.2:
                result["short_pressure"] = "moderate"
                result["signal"] = -4
            elif avg_up_vol > avg_down_vol * 1.5:
                result["short_pressure"] = "low"
                result["signal"] = 5  # Short squeeze potential if other factors align
            else:
                result["short_pressure"] = "normal"
                result["signal"] = 0

            # Short squeeze detection: price rising + volume spiking + down-day volume was high
            if (closes[-1] > closes[-5] and  # Price rising
                recent_vol > avg_vol * 1.5 and  # Volume spiking
                avg_down_vol > avg_up_vol * 1.3):  # Was heavily shorted
                result["squeeze_potential"] = True
                result["signal"] = 10
            else:
                result["squeeze_potential"] = False
    except Exception:
        pass

    _cache_set(cache_key, result)
    return result


# ── 5. Congressional Trading (SEC EDGAR Form 4 + heuristics) ─────────────────

def get_congressional_signal(ticker: str) -> dict:
    """
    Check for politically connected trading activity.
    Uses SEC EDGAR search for unusual Form 4 filing spikes.
    Cached for 12 hours.
    """
    cache_key = f"congress_{ticker}"
    cached = _cache_get(cache_key, ttl_seconds=43200)
    if cached:
        return cached

    result = {"signal": 0, "recent_filings": 0, "unusual_activity": False}

    try:
        # Check for recent Form 4 filing volume (spike = insiders active)
        recent_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        older_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

        # Recent 2 weeks
        url_recent = (f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
                     f"&forms=4&dateRange=custom&startdt={recent_date}")
        resp_recent = requests.get(url_recent, headers={"User-Agent": USER_AGENT}, timeout=10)
        recent_count = resp_recent.json().get("hits", {}).get("total", {}).get("value", 0)

        # Previous 6 weeks (for comparison)
        url_older = (f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
                    f"&forms=4&dateRange=custom&startdt={older_date}&enddt={recent_date}")
        resp_older = requests.get(url_older, headers={"User-Agent": USER_AGENT}, timeout=10)
        older_count = resp_older.json().get("hits", {}).get("total", {}).get("value", 0)

        result["recent_filings"] = recent_count
        avg_biweekly = older_count / 3 if older_count > 0 else 1  # ~3 biweekly periods in 6 weeks

        # Spike detection
        if recent_count > avg_biweekly * 2.5 and recent_count > 5:
            result["unusual_activity"] = True
            result["signal"] = -6  # Unusual insider filings often precede bad news
        elif recent_count > avg_biweekly * 1.5:
            result["signal"] = -3
        elif recent_count == 0 and avg_biweekly > 3:
            result["signal"] = 2  # Unusually quiet = no one wants to sell

    except Exception:
        pass

    _cache_set(cache_key, result)
    return result


# ── 6. Patent Activity (USPTO) ───────────────────────────────────────────────

TICKER_TO_COMPANY = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
    "AMZN": "Amazon", "META": "Facebook OR Meta", "NVDA": "Nvidia",
    "TSLA": "Tesla", "PFE": "Pfizer", "JNJ": "Johnson",
    "MRNA": "Moderna", "IBM": "IBM", "INTC": "Intel",
    "QCOM": "Qualcomm", "AMD": "Advanced Micro Devices",
}

def get_patent_activity(ticker: str) -> dict:
    """
    Check recent patent filings for innovation signals.
    Most useful for biotech and tech stocks.
    Cached for 24 hours.
    """
    cache_key = f"patent_{ticker}"
    cached = _cache_get(cache_key, ttl_seconds=86400)
    if cached:
        return cached

    result = {"signal": 0, "recent_patents": 0, "has_innovation_spike": False}

    company = TICKER_TO_COMPANY.get(ticker)
    if not company:
        _cache_set(cache_key, result)
        return result

    try:
        url = f"https://developer.uspto.gov/ibd-api/v1/application/publications?searchText={requests.utils.quote(company)}&start=0&rows=20"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        total = data.get("recordTotalQuantity", 0)

        result["recent_patents"] = min(total, 100)

        # A spike in patent filings can signal innovation
        if total > 50:
            result["signal"] = 5
            result["has_innovation_spike"] = True
        elif total > 20:
            result["signal"] = 2
    except Exception:
        pass

    _cache_set(cache_key, result)
    return result


# ── 7. Fail-to-Deliver Approximation ─────────────────────────────────────────

def get_ftd_signal(ticker: str) -> dict:
    """
    Approximate fail-to-deliver risk from volume patterns.
    High FTD = potential forced buying or manipulation.
    Cached for 6 hours.
    """
    cache_key = f"ftd_{ticker}"
    cached = _cache_get(cache_key, ttl_seconds=21600)
    if cached:
        return cached

    result = {"signal": 0, "ftd_risk": "low"}

    try:
        # Use volume analysis as FTD proxy
        # Stocks with very high volume relative to float tend to have FTD issues
        alpaca_headers = {
            "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
        }
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars?timeframe=1Day&start={start}&limit=15&adjustment=all&feed=sip"
        resp = requests.get(url, headers=alpaca_headers, timeout=8)
        bars = resp.json().get("bars", [])

        if len(bars) >= 5:
            volumes = [b.get("v", 0) for b in bars[:5]]
            avg_recent = sum(volumes) / len(volumes)

            # Get ticker details for shares outstanding (Polygon reference endpoint — not rate-limited)
            detail_url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={os.environ.get('POLYGON_KEY', '')}"
            detail_resp = requests.get(detail_url, timeout=5)
            detail = detail_resp.json().get("results", {})
            shares_outstanding = detail.get("share_class_shares_outstanding") or detail.get("weighted_shares_outstanding") or 0

            if shares_outstanding > 0:
                turnover_ratio = avg_recent / shares_outstanding
                if turnover_ratio > 0.15:  # >15% of float trading daily
                    result["ftd_risk"] = "high"
                    result["signal"] = -5
                elif turnover_ratio > 0.08:
                    result["ftd_risk"] = "moderate"
                    result["signal"] = -2
    except Exception:
        pass

    _cache_set(cache_key, result)
    return result


# ── Combined Alternative Data Score ───────────────────────────────────────────

def get_alt_data_score(ticker: str) -> dict:
    """
    Run all alternative data sources for a ticker.
    Returns combined signal score and individual signals.
    """
    wiki = get_wiki_attention(ticker)
    short = get_short_interest(ticker)
    congress = get_congressional_signal(ticker)
    patent = get_patent_activity(ticker)
    ftd = get_ftd_signal(ticker)

    # These are global, not per-ticker
    fred = get_fred_macro()
    geo = get_geopolitical_risk()

    # Combined score from all alt data
    alt_score = (
        wiki.get("signal", 0) +
        short.get("signal", 0) +
        congress.get("signal", 0) +
        patent.get("signal", 0) +
        ftd.get("signal", 0) +
        geo.get("risk_score", 0)
    )

    # FRED macro adjustment
    fred_adj = 0
    if fred.get("yield_curve_inverted"):
        fred_adj -= 5
    if fred.get("labor_market_weak"):
        fred_adj -= 3
    if fred.get("credit_stress"):
        fred_adj -= 8
    alt_score += fred_adj

    return {
        "alt_score": alt_score,
        "wiki": wiki,
        "short_interest": short,
        "congressional": congress,
        "patent": patent,
        "ftd": ftd,
        "fred_macro": fred,
        "geopolitical": geo,
        "fred_adjustment": fred_adj,
    }


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"=== Alt Data Report for {ticker} ===")
    report = get_alt_data_score(ticker)
    print(json.dumps(report, indent=2, default=str))
