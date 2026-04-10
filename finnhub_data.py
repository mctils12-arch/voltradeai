"""
VolTradeAI Finnhub Integration
Provides insider sentiment, earnings calendar, earnings surprises,
analyst recommendation trends, and company peers from Finnhub's free API.

Rate limit: 60 calls/min — we self-limit to 55/min via a token bucket.
Set FINNHUB_KEY environment variable to your free API key from finnhub.io.
"""
import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta

FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"

CACHE_DIR = "/tmp/voltrade_alt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

USER_AGENT = "VolTradeAI research@voltradeai.com"


# ── Cache helpers (same pattern as alt_data.py) ───────────────────────────────

def _cache_get(key, ttl_seconds=3600):
    """Read from disk cache if fresh."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_seconds:
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _cache_set(key, data):
    """Write to disk cache."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ── Token bucket rate limiter (55 calls/min) ──────────────────────────────────

class _TokenBucket:
    """
    Token bucket that allows at most `rate` tokens per `period` seconds.
    Thread-safe. Blocks the caller until a token is available.
    """
    def __init__(self, rate: int = 55, period: float = 60.0):
        self._rate = rate
        self._period = period
        self._tokens = float(rate)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        """Block until a request token is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            # Refill tokens proportionally to elapsed time
            self._tokens = min(
                float(self._rate),
                self._tokens + elapsed * (self._rate / self._period),
            )
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # Need to wait for the next token
            wait_time = (1.0 - self._tokens) * (self._period / self._rate)

        time.sleep(wait_time)

        with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0 + wait_time * (self._rate / self._period))


_bucket = _TokenBucket(rate=55, period=60.0)


# ── Internal API call helper ──────────────────────────────────────────────────

def _finnhub_get(endpoint: str, params: dict = None) -> dict | list | None:
    """
    Make a rate-limited GET request to the Finnhub API.
    Returns parsed JSON or None on failure.
    """
    if not FINNHUB_KEY or FINNHUB_KEY == "YOUR_FINNHUB_KEY_HERE":
        return {"error": "FINNHUB_KEY environment variable not set. "
                         "Get a free key at https://finnhub.io/register"}

    _bucket.acquire()

    url = f"{FINNHUB_BASE}{endpoint}"
    query = {"token": FINNHUB_KEY}
    if params:
        query.update(params)

    try:
        resp = requests.get(
            url,
            params=query,
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        return {"error": f"HTTP {status} from Finnhub endpoint {endpoint}"}
    except requests.exceptions.RequestException as exc:
        return {"error": f"Request failed: {exc}"}


# ── 1. Insider Sentiment ──────────────────────────────────────────────────────

def get_insider_sentiment(ticker: str) -> dict:
    """
    Fetch insider sentiment for a ticker using Finnhub's Monthly Share Purchase
    Ratio (MSPR). MSPR ranges from -100 (all sells) to +100 (all buys) and is
    a leading indicator for 30–90 day price moves.
    Cached for 6 hours.
    """
    cache_key = f"fh_insider_{ticker.upper()}"
    cached = _cache_get(cache_key, ttl_seconds=21600)
    if cached:
        return cached

    default = {
        "mspr": 0.0,
        "change": 0.0,
        "signal": "neutral",
        "total_buys": 0,
        "total_sells": 0,
    }

    # Finnhub requires a date range; pull the last 12 months
    today = datetime.now()
    date_from = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    date_to = today.strftime("%Y-%m-%d")

    data = _finnhub_get(
        "/stock/insider-sentiment",
        {"symbol": ticker.upper(), "from": date_from, "to": date_to},
    )

    if not data or isinstance(data, dict) and "error" in data:
        result = {**default, **(data if isinstance(data, dict) else {})}
        _cache_set(cache_key, result)
        return result

    items = data.get("data", []) if isinstance(data, dict) else []

    if not items:
        _cache_set(cache_key, default)
        return default

    # Use the most recent month's data; also compute rolling totals
    items_sorted = sorted(items, key=lambda x: x.get("year", 0) * 100 + x.get("month", 0))
    latest = items_sorted[-1]

    # Aggregate buy/sell counts across available history
    total_buys = sum(max(i.get("change", 0), 0) for i in items)
    total_sells = sum(abs(min(i.get("change", 0), 0)) for i in items)

    mspr = latest.get("mspr", 0.0) or 0.0
    change = latest.get("change", 0.0) or 0.0

    if mspr > 20:
        signal = "bullish"
    elif mspr < -20:
        signal = "bearish"
    else:
        signal = "neutral"

    result = {
        "mspr": round(mspr, 2),
        "change": round(change, 2),
        "signal": signal,
        "total_buys": int(total_buys),
        "total_sells": int(total_sells),
    }

    _cache_set(cache_key, result)
    return result


# ── 2. Earnings Calendar ──────────────────────────────────────────────────────

def get_earnings_calendar(ticker: str = None) -> list:
    """
    Fetch upcoming earnings announcements from Finnhub.
    If `ticker` is provided, results are filtered to that symbol.
    Returns a list of earnings events for the next 30 days.
    Cached for 4 hours.
    """
    cache_key = f"fh_earnings_cal_{ticker.upper() if ticker else 'ALL'}"
    cached = _cache_get(cache_key, ttl_seconds=14400)
    if cached is not None:
        return cached

    today = datetime.now()
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=30)).strftime("%Y-%m-%d")

    params = {"from": date_from, "to": date_to}
    if ticker:
        params["symbol"] = ticker.upper()

    data = _finnhub_get("/calendar/earnings", params)

    if isinstance(data, dict) and "error" in data:
        return [data]

    raw_events = []
    if isinstance(data, dict):
        raw_events = data.get("earningsCalendar", [])
    elif isinstance(data, list):
        raw_events = data

    results = []
    for event in raw_events:
        try:
            eps_actual = event.get("epsActual")
            results.append({
                "symbol": event.get("symbol", ""),
                "date": event.get("date", ""),
                "hour": event.get("hour", ""),          # "bmo" or "amc"
                "eps_estimate": event.get("epsEstimate") or 0.0,
                "eps_actual": eps_actual,               # None if not yet reported
                "revenue_estimate": event.get("revenueEstimate") or 0.0,
            })
        except Exception:
            continue

    _cache_set(cache_key, results)
    return results


# ── 3. Earnings Surprise ──────────────────────────────────────────────────────

def get_earnings_surprise(ticker: str) -> dict:
    """
    Fetch the last 4 quarters of EPS actuals vs. estimates to measure how
    consistently the company beats or misses expectations.
    Cached for 12 hours.
    """
    cache_key = f"fh_earnings_surp_{ticker.upper()}"
    cached = _cache_get(cache_key, ttl_seconds=43200)
    if cached:
        return cached

    default = {
        "surprises": [],
        "avg_surprise_pct": 0.0,
        "beat_rate": 0.0,
    }

    data = _finnhub_get("/stock/earnings", {"symbol": ticker.upper(), "limit": 4})

    if isinstance(data, dict) and "error" in data:
        result = {**default, **data}
        _cache_set(cache_key, result)
        return result

    if not data or not isinstance(data, list):
        _cache_set(cache_key, default)
        return default

    surprises = []
    for item in data[:4]:
        actual = item.get("actual")
        estimate = item.get("estimate")
        period = item.get("period", "")

        # Format period as YYYY-QN
        try:
            dt = datetime.strptime(period, "%Y-%m-%d")
            q = (dt.month - 1) // 3 + 1
            period_label = f"{dt.year}-Q{q}"
        except Exception:
            period_label = period

        if actual is not None and estimate is not None and estimate != 0:
            surprise_pct = round((actual - estimate) / abs(estimate) * 100, 2)
        else:
            surprise_pct = 0.0

        surprises.append({
            "period": period_label,
            "actual": actual,
            "estimate": estimate,
            "surprise_pct": surprise_pct,
        })

    valid = [s for s in surprises if s["estimate"] is not None and s["estimate"] != 0]
    avg_surprise_pct = round(
        sum(s["surprise_pct"] for s in valid) / len(valid), 2
    ) if valid else 0.0
    beat_rate = round(
        sum(1 for s in valid if s["surprise_pct"] > 0) / len(valid), 2
    ) if valid else 0.0

    result = {
        "surprises": surprises,
        "avg_surprise_pct": avg_surprise_pct,
        "beat_rate": beat_rate,
    }

    _cache_set(cache_key, result)
    return result


# ── 4. Recommendation Trends ──────────────────────────────────────────────────

def get_recommendation_trends(ticker: str) -> dict:
    """
    Fetch the latest analyst recommendation breakdown from Finnhub.
    Returns counts across all rating categories and an overall consensus.
    Cached for 12 hours.
    """
    cache_key = f"fh_reco_{ticker.upper()}"
    cached = _cache_get(cache_key, ttl_seconds=43200)
    if cached:
        return cached

    default = {
        "buy": 0,
        "hold": 0,
        "sell": 0,
        "strong_buy": 0,
        "strong_sell": 0,
        "consensus": "hold",
        "total_analysts": 0,
    }

    data = _finnhub_get("/stock/recommendation", {"symbol": ticker.upper()})

    if isinstance(data, dict) and "error" in data:
        result = {**default, **data}
        _cache_set(cache_key, result)
        return result

    if not data or not isinstance(data, list):
        _cache_set(cache_key, default)
        return default

    # Use the most recent period (first entry — Finnhub returns newest first)
    latest = data[0]

    strong_buy = int(latest.get("strongBuy", 0) or 0)
    buy = int(latest.get("buy", 0) or 0)
    hold = int(latest.get("hold", 0) or 0)
    sell = int(latest.get("sell", 0) or 0)
    strong_sell = int(latest.get("strongSell", 0) or 0)

    total = strong_buy + buy + hold + sell + strong_sell

    # Weighted consensus score: strong_buy=2, buy=1, hold=0, sell=-1, strong_sell=-2
    if total > 0:
        score = (strong_buy * 2 + buy * 1 + hold * 0 + sell * -1 + strong_sell * -2) / total
        if score > 0.5:
            consensus = "buy"
        elif score < -0.5:
            consensus = "sell"
        else:
            consensus = "hold"
    else:
        consensus = "hold"

    result = {
        "buy": buy,
        "hold": hold,
        "sell": sell,
        "strong_buy": strong_buy,
        "strong_sell": strong_sell,
        "consensus": consensus,
        "total_analysts": total,
    }

    _cache_set(cache_key, result)
    return result


# ── 5. Company Peers ──────────────────────────────────────────────────────────

def get_company_peers(ticker: str) -> list:
    """
    Fetch a list of peer tickers in the same sector from Finnhub.
    Cached for 24 hours.
    """
    cache_key = f"fh_peers_{ticker.upper()}"
    cached = _cache_get(cache_key, ttl_seconds=86400)
    if cached is not None:
        return cached

    data = _finnhub_get("/stock/peers", {"symbol": ticker.upper()})

    if isinstance(data, dict) and "error" in data:
        return [data]

    if isinstance(data, list):
        peers = [p for p in data if p != ticker.upper()]
        _cache_set(cache_key, peers)
        return peers

    _cache_set(cache_key, [])
    return []


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    print(f"=== Finnhub Data Report for {ticker} ===\n")

    report = {
        "insider_sentiment": get_insider_sentiment(ticker),
        "earnings_calendar": get_earnings_calendar(ticker),
        "earnings_surprise": get_earnings_surprise(ticker),
        "recommendation_trends": get_recommendation_trends(ticker),
        "company_peers": get_company_peers(ticker),
    }

    print(json.dumps(report, indent=2, default=str))
