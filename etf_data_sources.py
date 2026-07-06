#!/usr/bin/env python3
"""
VolTradeAI — ETF Builder data layer (production sources)

This module uses the SAME data sources the bot uses for trading decisions:

  Price history & current prices  → Alpaca Market Data API (SIP feed, adjusted)
                                    Batch endpoints supported (up to ~50 symbols).
  Per-holding company metadata    → Polygon /v3/reference/tickers/{T}
                                    (sector via SIC, industry, list_date, market
                                    cap, shares outstanding, description, etc.)
  P/E, beta, dividend yield, 52w  → Finnhub /stock/metric?metric=all
  Ticker classification           → Polygon `type` field (ETF/CS/REIT/FUND/ETN)
                                    Authoritative — no string matching needed.
  Full ETF holdings list          → NASDAQ public API (same as the .com site)

  Static fallbacks                → ETF_INCEPTION (~40 well-known funds, used
                                    only if Polygon list_date is empty)

Why this matters: yfinance is increasingly unreliable for ETFs (firstTradeDate
returns None, history flakes, top_holdings caps at 10). The bot already pays
for Alpaca SIP + Polygon — same data sources hedge funds use. Reusing them
here means the ETF Builder gets institutional-grade data for free.

Graceful degradation: every function returns None on failure, and callers
chain fallbacks. Missing API keys → returns None silently. Old yfinance/Stooq
code paths in etf_analyzer.py serve as final fallback.
"""
import os
import json
import time
import tempfile
import math
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from alpaca_feed import data_feed  # [REPAIR 2026-07-06] central feed w/ SIP-403 fallback


CACHE_DIR = "/tmp/voltrade_etf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


# ─── Tiny cache helpers (atomic write — same pattern as the rest of the project)

def _cache_get(key: str, ttl_seconds: int):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_seconds:
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _cache_set(key: str, data) -> None:
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        dirname = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".cache.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
    except Exception:
        pass


def _safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x):
    v = _safe_float(x)
    return int(v) if v is not None else None


# ─── ENV / API config ────────────────────────────────────────────────────────

ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_DATA = "https://data.alpaca.markets"

POLYGON_KEY = os.environ.get("POLYGON_KEY", "") or os.environ.get("POLYGON_API_KEY", "")
POLYGON_BASE = "https://api.polygon.io"

FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"


def _alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }


def have_alpaca() -> bool:
    return bool(ALPACA_KEY and ALPACA_SECRET)


def have_polygon() -> bool:
    return bool(POLYGON_KEY)


def have_finnhub() -> bool:
    return bool(FINNHUB_KEY)


# ─── ALPACA: batch daily-bar history ────────────────────────────────────────

def fetch_bars_alpaca(symbols, days=1900):
    """Fetch daily OHLCV bars for many symbols in batched Alpaca requests.

    Args:
      symbols: iterable of ticker strings (any case).
      days: how many calendar days of history to request. Default 1900 ≈ 5y.

    Returns:
      dict {SYMBOL: [(date_str, close_float), ...]} where each list is sorted
      ascending. Symbols with no data are simply absent from the dict.
      Returns {} if Alpaca keys aren't set or if the call fails entirely.

    Uses Alpaca's multi-symbol `/v2/stocks/bars` endpoint (up to 50 symbols
    per call, paginated). Same pattern intraday_shorts.py uses for short
    scanning — known-good in production.
    """
    if not have_alpaca() or not symbols:
        return {}

    symbols = [s.upper().strip() for s in symbols if s]
    if not symbols:
        return {}

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_iso = start.strftime("%Y-%m-%dT00:00:00Z")
    end_iso = end.strftime("%Y-%m-%dT00:00:00Z")

    out: dict[str, list] = {}

    def _fetch_batch(batch):
        params = {
            "symbols": ",".join(batch),
            "timeframe": "1Day",
            "start": start_iso,
            "end": end_iso,
            "limit": 10000,
            "adjustment": "all",  # split + dividend
            "feed": data_feed(),
        }
        try:
            url = f"{ALPACA_DATA}/v2/stocks/bars"
            page_token = None
            collected = {}
            for _ in range(8):  # safety cap on pagination
                if page_token:
                    params["page_token"] = page_token
                r = requests.get(url, params=params, headers=_alpaca_headers(), timeout=20)
                if r.status_code != 200:
                    return collected
                data = r.json()
                bars_map = data.get("bars", {}) or {}
                for sym, bars in bars_map.items():
                    rows = collected.setdefault(sym, [])
                    for b in bars:
                        t = b.get("t")
                        c = b.get("c")
                        if not t or c is None:
                            continue
                        # t is "2025-01-15T05:00:00Z"; take the date part
                        rows.append([t[:10], float(c)])
                page_token = data.get("next_page_token")
                if not page_token:
                    break
            return collected
        except Exception:
            return {}

    # Batch up to 50 symbols per request, run a few batches concurrently.
    batches = [symbols[i:i + 50] for i in range(0, len(symbols), 50)]
    with ThreadPoolExecutor(max_workers=min(6, len(batches))) as ex:
        for partial in ex.map(_fetch_batch, batches):
            for sym, rows in partial.items():
                # Sort and dedupe by date just in case pagination duplicates
                seen = set()
                deduped = []
                for d, c in sorted(rows):
                    if d in seen:
                        continue
                    seen.add(d)
                    deduped.append([d, c])
                if deduped:
                    out[sym] = deduped

    return out


# ─── Period-return computation from a bar history ────────────────────────────

_PERIODS = ["1mo", "3mo", "6mo", "ytd", "1y", "5y"]
_PERIOD_DAYS = {"1mo": 30, "3mo": 91, "6mo": 182, "1y": 365, "5y": 1825}


def returns_from_history(history):
    """Slice a single bar history into period returns.

    Args:
      history: list of [date_str, close_float] sorted ascending (Alpaca shape).

    Returns:
      dict {"1mo": 0.05, "3mo": 0.12, ...} with None where there isn't
      enough data for the period.
    """
    if not history or len(history) < 2:
        return {p: None for p in _PERIODS}

    last_date_str, last_close = history[-1]
    if last_close <= 0:
        return {p: None for p in _PERIODS}

    now = datetime.now(timezone.utc)
    out = {}
    for p in _PERIODS:
        if p == "ytd":
            year = now.year
            sliced = [row for row in history if row[0].startswith(str(year))]
        else:
            cutoff = (now - timedelta(days=_PERIOD_DAYS[p])).strftime("%Y-%m-%d")
            sliced = [row for row in history if row[0] >= cutoff]
        if not sliced or len(sliced) < 2:
            out[p] = None
            continue
        first_close = sliced[0][1]
        if first_close <= 0:
            out[p] = None
            continue
        out[p] = (last_close / first_close) - 1.0
    return out


def latest_close_from_history(history):
    if history and len(history) > 0:
        return history[-1][1]
    return None


# ─── POLYGON: per-ticker reference details ──────────────────────────────────

def fetch_ticker_details_polygon(symbol: str) -> dict | None:
    """Pull full reference data for a ticker from Polygon.

    Returns a normalised dict with the keys we care about, or None.

    The Polygon `/v3/reference/tickers/{T}` response contains:
      - ticker, name, type (ETF/CS/REIT/FUND/ETN/...)
      - market, locale, primary_exchange, currency_name
      - active, cik, composite_figi
      - market_cap (for stocks; null for ETFs)
      - share_class_shares_outstanding, weighted_shares_outstanding
      - description, homepage_url, total_employees
      - sic_code, sic_description
      - list_date (inception/IPO)
      - address {city, state, postal_code}
      - branding {logo_url, icon_url}

    Cached 24h — reference data doesn't change daily.
    """
    if not have_polygon() or not symbol:
        return None

    symbol = symbol.upper().strip()
    cache_key = f"poly_ref_{symbol}"
    cached = _cache_get(cache_key, ttl_seconds=24 * 3600)
    if cached is not None:
        return cached if cached else None

    url = f"{POLYGON_BASE}/v3/reference/tickers/{symbol}"
    try:
        r = requests.get(url, params={"apiKey": POLYGON_KEY}, timeout=8)
        if r.status_code != 200:
            _cache_set(cache_key, {})
            return None
        payload = r.json()
        results = payload.get("results") or {}
        if not results:
            _cache_set(cache_key, {})
            return None

        addr = results.get("address") or {}
        city = addr.get("city") or ""
        state = addr.get("state") or ""
        hq_parts = [p for p in (city, state) if p]
        hq = ", ".join(hq_parts) if hq_parts else None

        out = {
            "symbol": symbol,
            "name": results.get("name"),
            "type": results.get("type"),  # "ETF" | "CS" | "REIT" | "FUND" | "ETN" | "ADRC" | ...
            "market": results.get("market"),
            "exchange": results.get("primary_exchange"),
            "currency": (results.get("currency_name") or "").upper() or None,
            "active": results.get("active"),
            "cik": results.get("cik"),
            "market_cap": _safe_int(results.get("market_cap")),
            "shares_outstanding": _safe_int(
                results.get("share_class_shares_outstanding")
                or results.get("weighted_shares_outstanding")
            ),
            "description": results.get("description"),
            "website": results.get("homepage_url"),
            "employees": _safe_int(results.get("total_employees")),
            "sic_code": results.get("sic_code"),
            "sic_description": results.get("sic_description"),
            "list_date": results.get("list_date"),  # YYYY-MM-DD
            "headquarters": hq,
            "country": "USA" if (results.get("locale") or "").lower() == "us" else None,
            "logo_url": (results.get("branding") or {}).get("logo_url"),
        }
        _cache_set(cache_key, out)
        return out
    except Exception:
        return None


# ─── FINNHUB: per-ticker valuation metrics ──────────────────────────────────

def fetch_metrics_finnhub(symbol: str) -> dict | None:
    """P/E, beta, dividend yield, 52w high/low etc. for a stock.

    Polygon doesn't provide these on its base reference endpoint — they live
    on `/v3/reference/financials` which is paid + slow. Finnhub's
    `/stock/metric?metric=all` returns the same data points instantly.

    Cached 6h.
    """
    if not have_finnhub() or not symbol:
        return None

    symbol = symbol.upper().strip()
    cache_key = f"finn_metric_{symbol}"
    cached = _cache_get(cache_key, ttl_seconds=6 * 3600)
    if cached is not None:
        return cached if cached else None

    url = f"{FINNHUB_BASE}/stock/metric"
    try:
        r = requests.get(url, params={"symbol": symbol, "metric": "all",
                                       "token": FINNHUB_KEY},
                         headers={"User-Agent": UA}, timeout=8)
        if r.status_code != 200:
            _cache_set(cache_key, {})
            return None
        payload = r.json()
        m = (payload.get("metric") or {}) if isinstance(payload, dict) else {}
        if not m:
            _cache_set(cache_key, {})
            return None
        out = {
            "trailing_pe": _safe_float(m.get("peTTM") or m.get("peNormalizedAnnual")),
            "forward_pe": _safe_float(m.get("peExclExtraTTM")),
            "beta": _safe_float(m.get("beta")),
            "dividend_yield": _safe_float(m.get("dividendYieldIndicatedAnnual")),
            "52w_high": _safe_float(m.get("52WeekHigh")),
            "52w_low": _safe_float(m.get("52WeekLow")),
            "market_cap": _safe_int(m.get("marketCapitalization") * 1_000_000 if m.get("marketCapitalization") else None),
        }
        # Dividend yield from Finnhub is in % (e.g. 1.5 = 1.5%) — normalize to decimal
        if out["dividend_yield"] is not None and out["dividend_yield"] > 0.2:
            out["dividend_yield"] = out["dividend_yield"] / 100.0
        _cache_set(cache_key, out)
        return out
    except Exception:
        return None


# ─── Combined per-holding metadata fetch ────────────────────────────────────

def fetch_holding_metadata(symbol: str) -> dict:
    """One-call wrapper that combines Polygon (reference) + Finnhub (metrics).

    Returns a flat dict with everything needed for the holding detail card.
    Missing data is None. Never raises.
    """
    out = {
        "symbol": symbol.upper(),
        "name": None,
        "type": None,
        "sector": None,
        "industry": None,
        "country": None,
        "exchange": None,
        "currency": None,
        "first_trade_date": None,
        "listing_age_years": None,
        "market_cap": None,
        "shares_outstanding": None,
        "trailing_pe": None,
        "forward_pe": None,
        "dividend_yield": None,
        "beta": None,
        "52w_high": None,
        "52w_low": None,
        "employees": None,
        "headquarters": None,
        "website": None,
        "long_business_summary": None,
    }

    poly = fetch_ticker_details_polygon(symbol)
    if poly:
        out["name"] = poly.get("name") or out["name"]
        out["type"] = poly.get("type")
        out["exchange"] = poly.get("exchange") or out["exchange"]
        out["currency"] = poly.get("currency") or out["currency"]
        out["country"] = poly.get("country") or out["country"]
        out["market_cap"] = poly.get("market_cap") or out["market_cap"]
        out["shares_outstanding"] = poly.get("shares_outstanding") or out["shares_outstanding"]
        out["first_trade_date"] = poly.get("list_date")
        if out["first_trade_date"]:
            try:
                dt = datetime.strptime(out["first_trade_date"], "%Y-%m-%d")
                age_days = (datetime.now() - dt).days
                out["listing_age_years"] = round(age_days / 365.25, 2)
            except Exception:
                pass
        out["employees"] = poly.get("employees") or out["employees"]
        out["headquarters"] = poly.get("headquarters") or out["headquarters"]
        out["website"] = poly.get("website") or out["website"]
        out["long_business_summary"] = poly.get("description") or out["long_business_summary"]
        # Map SIC to GICS-ish sector. Polygon provides sic_description but
        # not GICS sector directly. The SIC description is usable as industry,
        # and we can heuristically map to a coarser sector grouping.
        sic_desc = poly.get("sic_description")
        if sic_desc:
            out["industry"] = sic_desc
            out["sector"] = _sic_to_sector(poly.get("sic_code"), sic_desc)

    finn = fetch_metrics_finnhub(symbol)
    if finn:
        for k in ("trailing_pe", "forward_pe", "beta", "dividend_yield",
                  "52w_high", "52w_low"):
            if out.get(k) is None and finn.get(k) is not None:
                out[k] = finn[k]
        if out.get("market_cap") is None and finn.get("market_cap") is not None:
            out["market_cap"] = finn["market_cap"]

    return out


# ─── SIC → coarse sector mapping ────────────────────────────────────────────
# Polygon returns the SEC's SIC code + description. We use the SIC's first
# digit to assign a GICS-style sector bucket. Coarse but consistent and free.

_SIC_FIRST_DIGIT = {
    "0": "Consumer Defensive",      # Agriculture, forestry, fishing
    "1": "Energy",                  # Mining + construction
    "2": "Industrials",             # Manufacturing
    "3": "Industrials",             # Manufacturing (continued)
    "4": "Communication Services",  # Transport + comms + utilities
    "5": "Consumer Cyclical",       # Wholesale + retail
    "6": "Financial Services",      # Finance + insurance + real estate
    "7": "Technology",              # Services
    "8": "Healthcare",              # Health, legal, education
    "9": "Other",                   # Public administration
}

# Sub-overrides where the first-digit bucket is wrong. Ordered most-specific
# first so narrow ranges (e.g. 2830-2836 pharma) win over wider ones (2800-2899
# chemicals).
_SIC_RANGE_OVERRIDES = [
    (("2830", "2836"), "Healthcare"),       # pharma (subset of 2800-2899 chemicals)
    (("1311", "1389"), "Energy"),           # oil/gas extraction
    (("2900", "2999"), "Energy"),           # petroleum refining
    (("2800", "2899"), "Basic Materials"),  # chemicals (excludes pharma above)
    (("4812", "4813", "4899"), "Communication Services"),  # wireless / telecom
    (("4911", "4924", "4931", "4961"), "Utilities"),
    (("6020", "6099"), "Financial Services"),  # banks
    (("6311", "6411"), "Financial Services"),  # insurance
    (("6798", "6799"), "Real Estate"),         # REITs (more specific than 6500-6599 below)
    (("6500", "6599"), "Real Estate"),
    (("7370", "7389"), "Technology"),       # computer services
    (("8000", "8099"), "Healthcare"),
    (("5411", "5499"), "Consumer Defensive"),  # food stores
    (("5810", "5829"), "Consumer Cyclical"),   # restaurants
]


def _sic_to_sector(sic_code, sic_description):
    """Map a SIC code to a coarse sector. Falls back to first-digit bucketing."""
    if not sic_code:
        return None
    sic_str = str(sic_code).strip()
    if not sic_str:
        return None

    # Range overrides
    for codes, sector in _SIC_RANGE_OVERRIDES:
        if len(codes) == 2:
            lo, hi = codes
            if lo <= sic_str <= hi:
                return sector
        elif sic_str in codes:
            return sector

    # First-digit bucket
    return _SIC_FIRST_DIGIT.get(sic_str[0], None)


# ─── Static inception-date table (used only when Polygon list_date is empty) ─

# Polygon returns list_date for nearly all ETFs, so this is rarely needed —
# kept as a final safety net.
ETF_INCEPTION = {
    "SPY": "1993-01-22", "VOO": "2010-09-07", "IVV": "2000-05-15", "SPLG": "2005-11-15",
    "QQQ": "1999-03-10", "QQQM": "2020-10-13",
    "DIA": "1998-01-14",
    "IWM": "2000-05-22", "VTWO": "2010-09-20",
    "VTI": "2001-05-24", "ITOT": "2004-01-20",
    "VEA": "2007-07-20", "EFA": "2001-08-14",
    "VWO": "2005-03-04", "EEM": "2003-04-07",
    "AGG": "2003-09-22", "BND": "2007-04-03",
    "TLT": "2002-07-22", "SHY": "2002-07-22", "IEF": "2002-07-22",
    "GLD": "2004-11-18", "SLV": "2006-04-21",
    "XLK": "1998-12-16", "XLF": "1998-12-16", "XLE": "1998-12-16",
    "XLV": "1998-12-16", "XLY": "1998-12-16", "XLP": "1998-12-16",
    "XLI": "1998-12-16", "XLU": "1998-12-16", "XLB": "1998-12-16",
    "XLC": "2018-06-18", "XLRE": "2015-10-07",
    "ARKK": "2014-10-31", "ARKQ": "2014-09-30", "ARKW": "2014-09-29",
    "ARKG": "2014-10-31", "ARKF": "2019-02-04", "ARKX": "2021-03-30",
    "JEPI": "2020-05-20", "JEPQ": "2022-05-03",
    "VYM": "2006-11-10", "VIG": "2006-04-21", "SCHD": "2011-10-20",
    "VNQ": "2004-09-23", "REM": "2007-05-01", "MORT": "2011-08-16",
    "MBB": "2007-03-13",
    "VXUS": "2011-01-26", "BNDX": "2013-05-31",
    "VB": "2004-01-26", "VTV": "2004-01-26", "VUG": "2004-01-26",
    "VEU": "2007-03-02", "BSV": "2007-04-03", "BIV": "2007-04-03",
    "TIP": "2003-12-04",
    "HYG": "2007-04-04", "LQD": "2002-07-22",
    "EMB": "2007-12-17",
}


# ─── NASDAQ holdings API (public, no key) ────────────────────────────────────

def fetch_full_holdings_nasdaq(ticker: str):
    """Fetch the full holdings list for an ETF via NASDAQ's public website API.

    Same endpoint NASDAQ uses to populate the holdings table on their ETF
    detail pages. Returns a list of dicts:

        [{"symbol": "AAPL", "weight": 0.072, "name": "Apple Inc"}, ...]

    Returns None on failure → caller falls back to yfinance top-N.
    Cached 6h.
    """
    ticker = ticker.upper()
    cache_key = f"nasdaq_holdings_{ticker}"
    cached = _cache_get(cache_key, ttl_seconds=6 * 3600)
    if cached is not None:
        return cached if cached else None

    url = (f"https://api.nasdaq.com/api/quote/{ticker}/etfdetail/holdings"
           f"?assetclass=etf&limit=9999&offset=0")
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nasdaq.com",
        "Referer": f"https://www.nasdaq.com/market-activity/etf/{ticker.lower()}",
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            _cache_set(cache_key, [])
            return None
        payload = r.json()
    except Exception:
        return None

    # Try a couple of plausible JSON paths defensively
    rows = None
    try:
        data = payload.get("data") or {}
        nested = data.get("data") or data.get("holdings") or {}
        rows = nested.get("rows") or nested.get("data") or data.get("rows")
    except Exception:
        rows = None

    if not rows:
        _cache_set(cache_key, [])
        return None

    out = []
    for row in rows:
        try:
            sym = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
            if not sym or sym in ("N/A", "—", "-"):
                continue
            pct_raw = row.get("holdingsPercent") or row.get("weight") or row.get("percent")
            if pct_raw is None:
                continue
            pct_str = str(pct_raw).replace("%", "").replace(",", "").strip()
            pct = _safe_float(pct_str)
            if pct is None:
                continue
            weight = pct / 100.0 if pct > 1.5 else pct
            name = (row.get("companyName") or row.get("name") or row.get("issuerName") or "").strip() or None
            out.append({"symbol": sym, "weight": weight, "name": name})
        except Exception:
            continue

    if not out:
        _cache_set(cache_key, [])
        return None

    out.sort(key=lambda h: h.get("weight") or 0, reverse=True)
    _cache_set(cache_key, out)
    return out


# ─── Ticker classification — Polygon `type` is authoritative ────────────────

# Polygon ticker types (from their docs):
#   CS    = Common Stock
#   PFD   = Preferred Stock
#   ETF   = Exchange Traded Fund
#   ETN   = Exchange Traded Note
#   ETV   = Exchange Traded Vehicle
#   FUND  = Mutual Fund
#   REIT  = Real Estate Investment Trust (sometimes)
#   ADRC  = American Depositary Receipt - Common
#   ADRP  = American Depositary Receipt - Preferred
#   SP    = Structured Product
#   WARRANT, RIGHT, UNIT, ...

_POLY_TYPE_LABEL = {
    "ETF": "Exchange Traded Fund",
    "ETN": "Exchange Traded Note",
    "ETV": "Exchange Traded Vehicle",
    "FUND": "Mutual Fund",
    "CS": "Common Stock",
    "PFD": "Preferred Stock",
    "ADRC": "American Depositary Receipt",
    "ADRP": "American Depositary Receipt",
    "REIT": "Real Estate Investment Trust",
    "SP": "Structured Product",
    "WARRANT": "Warrant",
    "RIGHT": "Right",
    "UNIT": "Unit",
}


def classify_ticker(symbol: str):
    """Classify what `symbol` actually IS, using Polygon as the source of truth.

    Returns (classification_id, human_message, suggested_alt_etf) where
    classification_id is one of:
       "etf"          — it's an ETF (caller should proceed)
       "etn"          — exchange-traded note (close enough — caller may accept)
       "mutual_fund"  — mutual fund
       "stock"        — common stock
       "reit"         — REIT (equity)
       "mreit"        — mortgage REIT
       "adr"          — American depositary receipt
       "index"        — market index (^GSPC etc.)
       "crypto"       — cryptocurrency (BTC-USD etc.)
       "preferred"    — preferred share
       "warrant"      — warrant
       "unknown"      — couldn't classify

    The caller in etf_analyzer.py treats "etf", "etn", "etv" as proceed,
    and everything else as a friendly rejection.
    """
    symbol = symbol.upper().strip()

    # Index / crypto heuristics before any API call
    if symbol.startswith("^"):
        return ("index",
                f"{symbol} is a market index, not a tradable fund. "
                f"You can't invest directly in an index — look for an ETF that tracks it.",
                "SPY")
    if symbol.endswith("-USD") or symbol.endswith(".USD"):
        return ("crypto",
                f"{symbol} is a cryptocurrency, not an ETF.",
                None)

    # Polygon ticker reference — authoritative `type` field
    poly = fetch_ticker_details_polygon(symbol)
    if poly:
        t = (poly.get("type") or "").upper()
        name = poly.get("name") or symbol
        sic_code = str(poly.get("sic_code") or "")
        sic_desc = (poly.get("sic_description") or "").lower()

        if t in ("ETF", "ETN", "ETV"):
            return ("etf", "", None)

        if t == "FUND":
            return ("mutual_fund",
                    f"{name} is a mutual fund, not an ETF. "
                    f"Mutual funds price once a day at NAV and have different "
                    f"trading mechanics. The ETF Builder is for exchange-traded funds only.",
                    None)

        if t == "REIT":
            # Mortgage REIT vs equity REIT — check SIC range
            is_mreit = (sic_code.startswith("6798") or "mortgage" in (name or "").lower()
                        or "mortgage" in sic_desc)
            if is_mreit:
                return ("mreit",
                        f"{name} is a mortgage REIT (mREIT) — it's an individual company "
                        f"that owns mortgage-backed securities, not a fund. "
                        f"Use the regular Analyze view (Options & Volatility, Smart Money, "
                        f"or Structure) for company-level analysis.",
                        "REM")
            return ("reit",
                    f"{name} is a REIT (Real Estate Investment Trust) — an individual "
                    f"company that owns real estate, not a fund. "
                    f"Use Options & Volatility, Smart Money, or Structure for company analysis. "
                    f"For an ETF that holds REITs, try VNQ.",
                    "VNQ")

        if t in ("ADRC", "ADRP"):
            return ("adr",
                    f"{name} is an American Depositary Receipt (ADR) representing a "
                    f"foreign company's shares — not a fund. "
                    f"Use the regular Analyze view for company-level analysis.",
                    None)

        if t == "PFD":
            return ("preferred",
                    f"{name} is a preferred stock — fixed-income-like equity, not a fund. "
                    f"The ETF Builder doesn't analyse individual preferreds.",
                    None)

        if t == "WARRANT":
            return ("warrant",
                    f"{name} is a warrant — a derivative on a stock, not a fund.",
                    None)

        if t == "CS":
            # SIC-based REIT detection in case Polygon classified it as CS
            # (some REITs come through as CS — Polygon's tagging isn't perfect).
            if sic_code.startswith("6798") or "real estate investment trust" in sic_desc:
                is_mreit = "mortgage" in sic_desc or "mortgage" in (name or "").lower()
                if is_mreit:
                    return ("mreit",
                            f"{name} is a mortgage REIT (mREIT) — an individual company that "
                            f"owns mortgage-backed securities, not a fund. "
                            f"Use the regular Analyze view for company analysis.",
                            "REM")
                return ("reit",
                        f"{name} is a REIT (Real Estate Investment Trust) — an individual "
                        f"company that owns real estate, not a fund. "
                        f"For an ETF of REITs, try VNQ.",
                        "VNQ")

            sector = _sic_to_sector(sic_code, sic_desc) or ""
            sector_clause = f" — a {sector.lower()} company" if sector else ""
            return ("stock",
                    f"{name} is an individual stock{sector_clause}, not a fund. "
                    f"Use Options & Volatility, Smart Money, or Structure for "
                    f"company-level analysis.",
                    None)

        # Some other Polygon type we don't have a friendly name for
        human_type = _POLY_TYPE_LABEL.get(t, t.title() if t else "instrument")
        return ("unknown",
                f"{name} is a {human_type} — not an ETF. "
                f"The ETF Builder accepts ETFs only (SPY, QQQ, VTI, ARKK, etc.).",
                None)

    # Polygon couldn't find the ticker — return None so caller can decide
    return None


# Backwards-compatibility shim + offline-capable fallback classifier.
# Used by etf_analyzer.py when Polygon's classify_ticker returns None
# (no API key OR ticker not in Polygon's universe) AND yfinance reports
# the ticker isn't an ETF. The heuristics here work entirely from a
# yfinance info dict — no network needed.

_REIT_INDUSTRY_MARKERS = (
    "reit", "real estate investment trust",
    "residential real", "industrial real", "office real",
    "retail real", "specialty real", "diversified real", "hotel & motel real",
)


def classify_non_etf(info: dict):
    """Legacy entry point. Tries Polygon first (when configured), then falls
    back to heuristic classification of a yfinance info dict.

    Returns (classification_id, human_message, suggested_alt_etf).
    """
    sym = (info.get("symbol") or "").upper()

    # Try Polygon first if we have a key — it's authoritative when it works
    if have_polygon() and sym:
        result = classify_ticker(sym)
        if result is not None:
            cid, msg, suggested = result
            if cid == "etf":
                # Caller said it's not an ETF, but Polygon disagrees — fall through
                # to heuristic below rather than telling user something wrong
                pass
            else:
                return (cid, msg, suggested)

    # Heuristic fallback — uses only the yfinance info dict (offline-safe)
    qt = (info.get("quoteType") or "").upper()
    sector = (info.get("sector") or "")
    industry = (info.get("industry") or "")
    industry_l = industry.lower()
    name = info.get("longName") or info.get("shortName") or sym or ""

    if qt == "MUTUALFUND":
        return ("mutual_fund",
                f"{name or sym} is a mutual fund, not an ETF. "
                f"Mutual funds price once a day at NAV and have different trading mechanics. "
                f"The ETF Builder is for exchange-traded funds only.",
                None)

    if qt == "INDEX":
        return ("index",
                f"{name or sym} is a market index, not a tradable fund. "
                f"You can't invest directly in an index — look for an ETF that tracks it.",
                "SPY")

    if qt == "CRYPTOCURRENCY":
        return ("crypto",
                f"{name or sym} is a cryptocurrency, not an ETF.",
                None)

    if qt == "CURRENCY":
        return ("currency",
                f"{name or sym} is a currency pair, not an ETF.",
                None)

    # REIT detection
    is_reit = any(marker in industry_l for marker in _REIT_INDUSTRY_MARKERS)
    if not is_reit and "real estate" in (sector or "").lower():
        is_reit = True
    if not is_reit and "real estate investment" in name.lower():
        is_reit = True

    if is_reit:
        is_mreit = "mortgage" in industry_l or "mortgage" in name.lower()
        if is_mreit:
            return ("mreit",
                    f"{name or sym} is a mortgage REIT (mREIT) — it's an individual company "
                    f"that owns mortgage-backed securities, not a fund. "
                    f"Use the regular Analyze view (Options & Volatility, Smart Money, or "
                    f"Structure) for company-level analysis.",
                    "REM")
        return ("reit",
                f"{name or sym} is a REIT (Real Estate Investment Trust) — an individual "
                f"company that owns real estate, not a fund. "
                f"Use Options & Volatility, Smart Money, or Structure for company analysis. "
                f"For an ETF of REITs, try VNQ.",
                "VNQ")

    if qt == "EQUITY" or qt == "":
        sector_clause = f" — a {sector.lower()} company" if sector else ""
        return ("stock",
                f"{name or sym} is an individual stock{sector_clause}, not a fund. "
                f"Use Options & Volatility, Smart Money, or Structure for company-level analysis.",
                None)

    return ("unknown",
            f"{name or sym} doesn't look like an ETF. "
            f"The ETF Builder accepts ETFs only (SPY, QQQ, VTI, ARKK, etc.).",
            None)


# ─── Convenience: compute returns for a whole batch in one shot ──────────────

def compute_returns_for_symbols(symbols, history_days=1900):
    """Fetch bars from Alpaca for all `symbols` in batched requests, then
    compute multi-period returns and latest prices for each.

    Returns dict {SYMBOL: {"performance": {...}, "price": float | None,
                            "history_points": int}}.

    Symbols with no data still appear in the result with all-None values.
    """
    bars_map = fetch_bars_alpaca(symbols, days=history_days)
    out = {}
    for sym in (s.upper().strip() for s in symbols if s):
        hist = bars_map.get(sym)
        if hist:
            out[sym] = {
                "performance": returns_from_history(hist),
                "price": latest_close_from_history(hist),
                "history_points": len(hist),
            }
        else:
            out[sym] = {
                "performance": {p: None for p in _PERIODS},
                "price": None,
                "history_points": 0,
            }
    return out


# ─── Single-ticker convenience helpers (used by ETF-level calls) ─────────────

def compute_multi_period_returns(ticker: str) -> dict:
    """Returns dict for ONE ticker. Wraps the batch version."""
    result = compute_returns_for_symbols([ticker])
    return result.get(ticker.upper(), {}).get("performance") or {p: None for p in _PERIODS}


def latest_price(ticker: str):
    """Most recent close from Alpaca for one ticker."""
    result = compute_returns_for_symbols([ticker], history_days=10)
    return result.get(ticker.upper(), {}).get("price")


# Legacy alias kept for compatibility with the previous Stooq-based code
def latest_price_stooq(ticker: str):
    return latest_price(ticker)
