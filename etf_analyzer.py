#!/usr/bin/env python3
"""
VolTradeAI — ETF Builder / Analyzer

Pulls ETF metadata, holdings, sector breakdown, expense ratio, and
multi-period performance from Yahoo Finance (via yfinance). For each
holding it surfaces a rich snapshot — listing date, market cap, sector,
valuation multiples, beta, 52w range, plus an IPO/SPAC heuristic — and
computes contribution to ETF return so callers can highlight drags.

Output: single JSON object to stdout (same convention as analyze.py).

Limits (documented honestly in the response):
  - Yahoo Finance returns top-N holdings (typically 10) per ETF. Full
    holdings require scraping the issuer (SSGA, iShares, Vanguard).
  - Yahoo does not publish rebalance dates. We surface a typical
    schedule by family.
  - SPAC flag is name-pattern heuristic, not authoritative.
  - "Listing date" = firstTradeDateEpochUtc; for deSPACs this is the
    original SPAC IPO, not the operating company's public debut.

Usage:
  python3 etf_analyzer.py SPY
"""

import sys
import json
import math
import os
import time
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

try:
    import yfinance as yf  # noqa
except Exception as e:
    print(json.dumps({"error": f"yfinance not available: {e}"}))
    sys.exit(1)


# ── Cache (filesystem, mirrors finnhub_data.py / alt_data.py pattern) ────────

CACHE_DIR = "/tmp/voltrade_etf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


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


# ── JSON cleanliness helpers ─────────────────────────────────────────────────

def _clean_nan(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    return obj


def _safe_float(x):
    """Coerce yfinance numbers (which can be numpy types) to clean Python floats."""
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


# ── Family / index heuristics ────────────────────────────────────────────────

# Maps fund-family substrings to a "typical rebalance schedule" string. Yahoo
# does not publish actual rebalance dates; this is informational only.
_REBALANCE_BY_FAMILY = [
    ("SPDR", "Quarterly (S&P 500 reconstitution: 3rd Fri of Mar/Jun/Sep/Dec)"),
    ("iShares Core S&P", "Quarterly (S&P reconstitution)"),
    ("iShares Russell", "Annual (Russell reconstitution in June)"),
    ("Vanguard", "Quarterly (CRSP index reconstitution)"),
    ("Invesco QQQ", "Quarterly (Nasdaq-100 review)"),
    ("Invesco S&P", "Quarterly (S&P review)"),
    ("ARK", "Active — discretionary, no fixed schedule"),
    ("First Trust", "Quarterly review"),
    ("Schwab", "Quarterly (per underlying index)"),
    ("WisdomTree", "Annual reconstitution"),
    ("VanEck", "Quarterly review"),
]

# A small index map for the "tracks" field. yfinance does not always expose
# the underlying index cleanly, so we fall back to a few well-known cases.
_INDEX_BY_TICKER = {
    "SPY": "S&P 500", "VOO": "S&P 500", "IVV": "S&P 500", "SPLG": "S&P 500",
    "QQQ": "Nasdaq-100", "QQQM": "Nasdaq-100",
    "DIA": "Dow Jones Industrial Average",
    "IWM": "Russell 2000", "VTWO": "Russell 2000",
    "VTI": "CRSP US Total Market", "ITOT": "S&P Total Market",
    "VEA": "FTSE Developed All Cap ex US", "EFA": "MSCI EAFE",
    "VWO": "FTSE Emerging Markets", "EEM": "MSCI Emerging Markets",
    "AGG": "Bloomberg US Aggregate Bond", "BND": "Bloomberg US Aggregate Float Adj",
    "TLT": "ICE US Treasury 20+ Year", "SHY": "ICE US Treasury 1-3 Year",
    "GLD": "Gold spot (physical bullion)", "SLV": "Silver spot (physical bullion)",
    "XLK": "S&P 500 Information Technology",
    "XLF": "S&P 500 Financials",
    "XLE": "S&P 500 Energy",
    "XLV": "S&P 500 Health Care",
    "XLY": "S&P 500 Consumer Discretionary",
    "XLP": "S&P 500 Consumer Staples",
    "XLI": "S&P 500 Industrials",
    "XLU": "S&P 500 Utilities",
    "XLB": "S&P 500 Materials",
    "XLC": "S&P 500 Communication Services",
    "XLRE": "S&P 500 Real Estate",
    "ARKK": "Active — Disruptive Innovation",
    "JEPI": "Active — Equity Premium Income",
    "JEPQ": "Active — Nasdaq Equity Premium Income",
}


def _rebalance_for(family: str, ticker: str) -> str:
    if ticker.upper() in _INDEX_BY_TICKER and "Active" in _INDEX_BY_TICKER[ticker.upper()]:
        return "Active — discretionary, no fixed schedule"
    fam = family or ""
    for key, sched in _REBALANCE_BY_FAMILY:
        if key.lower() in fam.lower():
            return sched
    return "Not disclosed by Yahoo (check issuer prospectus for rebalance schedule)"


def _index_for(ticker: str, category: str) -> str:
    t = ticker.upper()
    if t in _INDEX_BY_TICKER:
        return _INDEX_BY_TICKER[t]
    if category:
        return f"(Tracks a {category.lower()} benchmark — see issuer prospectus)"
    return "Unknown"


def _management_style(info: dict, fund_overview: dict | None, ticker: str) -> str:
    """Heuristically classify active vs passive."""
    t = ticker.upper()
    # Known actives
    if t in {"ARKK", "ARKQ", "ARKW", "ARKG", "ARKF", "ARKX", "JEPI", "JEPQ",
             "DIVO", "QYLD", "RYLD", "XYLD", "NUSI", "BALI", "PFFA", "AVUV",
             "AVDV", "AVEM", "FFTY", "MOAT"}:
        return "Active"
    name = (info.get("longName") or "").lower()
    if "active" in name or "actively managed" in name:
        return "Active"
    family = (info.get("fundFamily") or "").lower()
    if "ark" in family or "actively managed" in family:
        return "Active"
    # Default: most ETFs by AUM are passive index trackers
    return "Passive (index-tracking)"


# ── SPAC heuristic ────────────────────────────────────────────────────────────

_SPAC_NAME_PATTERNS = [
    "acquisition corp", "acquisition corporation",
    "capital partners corp", "growth opportunities corp",
]


def _spac_heuristic(long_name: str | None, short_name: str | None) -> bool:
    if not long_name and not short_name:
        return False
    blob = ((long_name or "") + " " + (short_name or "")).lower()
    return any(p in blob for p in _SPAC_NAME_PATTERNS)


# ── Per-holding fetch ─────────────────────────────────────────────────────────

# Periods we surface in the UI. Yahoo accepts standard yfinance period codes.
_PERIODS = ["1mo", "3mo", "6mo", "ytd", "1y", "5y"]


def _period_return(hist_close, period: str):
    """Compute % return over the period from a daily-close Series."""
    if hist_close is None or len(hist_close) < 2:
        return None
    try:
        last = float(hist_close.iloc[-1])
        first = float(hist_close.iloc[0])
        if first <= 0:
            return None
        return (last / first - 1.0)
    except Exception:
        return None


def _fetch_period_returns(ticker_obj, periods=_PERIODS) -> dict:
    """One yfinance .history() call per period — pulled sequentially per holding
    but the holdings themselves are parallelized at the level above."""
    out = {}
    for p in periods:
        try:
            h = ticker_obj.history(period=p, auto_adjust=True)
            if h is None or h.empty:
                out[p] = None
                continue
            out[p] = _period_return(h["Close"])
        except Exception:
            out[p] = None
    return out


def _fetch_holding_snapshot(symbol: str, weight: float) -> dict:
    """Build the per-holding payload. Cached 1h per symbol to keep ETF rebuilds
    fast across users."""
    cache_key = f"holding_{symbol.upper()}"
    cached = _cache_get(cache_key, ttl_seconds=3600)
    if cached is not None:
        cached["weight"] = weight  # weight comes from the parent ETF, not cached
        return cached

    out = {
        "symbol": symbol,
        "weight": weight,
        "name": None,
        "price": None,
        "currency": None,
        "exchange": None,
        "country": None,
        "sector": None,
        "industry": None,
        "market_cap": None,
        "shares_outstanding": None,
        "first_trade_date": None,
        "listing_age_years": None,
        "is_spac_heuristic": False,
        "trailing_pe": None,
        "forward_pe": None,
        "dividend_yield": None,
        "beta": None,
        "52w_high": None,
        "52w_low": None,
        "52w_pos_pct": None,
        "employees": None,
        "headquarters": None,
        "website": None,
        "long_business_summary": None,
        "performance": {},
        "fetch_error": None,
    }

    try:
        t = yf.Ticker(symbol)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            try:
                info = t.get_info() or {}
            except Exception:
                info = {}

        out["name"] = info.get("longName") or info.get("shortName")
        out["price"] = _safe_float(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        out["currency"] = info.get("currency")
        out["exchange"] = info.get("exchange") or info.get("fullExchangeName")
        out["country"] = info.get("country")
        out["sector"] = info.get("sector")
        out["industry"] = info.get("industry")
        out["market_cap"] = _safe_int(info.get("marketCap"))
        out["shares_outstanding"] = _safe_int(info.get("sharesOutstanding"))
        out["trailing_pe"] = _safe_float(info.get("trailingPE"))
        out["forward_pe"] = _safe_float(info.get("forwardPE"))
        out["dividend_yield"] = _safe_float(info.get("dividendYield"))
        out["beta"] = _safe_float(info.get("beta"))
        out["52w_high"] = _safe_float(info.get("fiftyTwoWeekHigh"))
        out["52w_low"] = _safe_float(info.get("fiftyTwoWeekLow"))
        out["employees"] = _safe_int(info.get("fullTimeEmployees"))
        out["website"] = info.get("website")
        out["long_business_summary"] = info.get("longBusinessSummary")

        # Headquarters
        city = info.get("city") or ""
        state = info.get("state") or ""
        country = info.get("country") or ""
        hq_parts = [p for p in (city, state, country) if p]
        if hq_parts:
            out["headquarters"] = ", ".join(hq_parts)

        # 52-week position (0-1)
        h, l, p = out["52w_high"], out["52w_low"], out["price"]
        if h and l and p and h > l:
            out["52w_pos_pct"] = max(0.0, min(1.0, (p - l) / (h - l)))

        # First trade / listing date
        ftd = info.get("firstTradeDateEpochUtc")
        if ftd:
            try:
                dt = datetime.fromtimestamp(int(ftd), tz=timezone.utc)
                out["first_trade_date"] = dt.strftime("%Y-%m-%d")
                age_days = (datetime.now(timezone.utc) - dt).days
                out["listing_age_years"] = round(age_days / 365.25, 2)
            except Exception:
                pass

        # SPAC heuristic
        out["is_spac_heuristic"] = _spac_heuristic(
            info.get("longName"), info.get("shortName")
        )

        # Multi-period performance — only call history if we have a usable ticker
        out["performance"] = _fetch_period_returns(t)

    except Exception as e:
        out["fetch_error"] = str(e)[:200]

    _cache_set(cache_key, out)
    out["weight"] = weight  # ensure freshest weight
    return out


# ── ETF top-level fetch ───────────────────────────────────────────────────────

def _df_to_dict(df, key_col, val_col):
    """Convert a small yfinance DataFrame to a plain dict, robust to schema drift."""
    if df is None:
        return {}
    try:
        if hasattr(df, "empty") and df.empty:
            return {}
        # DataFrame path
        if hasattr(df, "iterrows"):
            out = {}
            for _, row in df.iterrows():
                k = row.get(key_col)
                v = row.get(val_col)
                if k is None and df.index.name == key_col:
                    k = row.name
                if k is not None:
                    out[str(k)] = _safe_float(v)
            return out
        # Dict path (some yfinance versions return dicts)
        if isinstance(df, dict):
            return {str(k): _safe_float(v) for k, v in df.items()}
    except Exception:
        pass
    return {}


def _extract_top_holdings(funds_data) -> list[tuple[str, float]]:
    """Return [(symbol, weight_0_to_1), ...] for the ETF's top holdings.

    yfinance has shifted this API across versions; we try a few shapes.
    """
    try:
        th = funds_data.top_holdings
    except Exception:
        return []

    holdings: list[tuple[str, float]] = []

    # Most current yfinance returns a DataFrame with columns:
    #   index = symbol, "Holding Percent" or "holdingPercent" column
    try:
        if hasattr(th, "iterrows"):
            cols = [c.lower() for c in th.columns]
            pct_col = None
            for c in th.columns:
                cl = c.lower().replace(" ", "")
                if "percent" in cl or "weight" in cl or cl == "pct":
                    pct_col = c
                    break
            for idx, row in th.iterrows():
                sym = str(idx).strip().upper()
                if not sym or sym == "NAN":
                    continue
                w = _safe_float(row[pct_col]) if pct_col else None
                if w is None:
                    # Try common fallbacks
                    for c in th.columns:
                        v = _safe_float(row[c])
                        if v is not None and 0 < v <= 1.5:
                            w = v
                            break
                if w is None:
                    continue
                # Normalize: yfinance sometimes returns 0-1 floats, sometimes 0-100
                if w > 1.5:
                    w = w / 100.0
                holdings.append((sym, w))
        elif isinstance(th, dict):
            for sym, w in th.items():
                wv = _safe_float(w)
                if wv is None:
                    continue
                if wv > 1.5:
                    wv = wv / 100.0
                holdings.append((str(sym).upper(), wv))
    except Exception:
        pass

    return holdings


def _extract_expense_ratio(funds_data, info: dict) -> float | None:
    """Pull expense ratio from fund_operations (preferred) or info."""
    # Try fund_operations DataFrame
    try:
        fo = funds_data.fund_operations
        if fo is not None and hasattr(fo, "loc"):
            # Common row name in current yfinance: "Annual Report Expense Ratio"
            for row_name in ("Annual Report Expense Ratio", "annualReportExpenseRatio",
                             "Expense Ratio", "expenseRatio"):
                if row_name in fo.index:
                    # First column is usually the ETF itself
                    try:
                        val = _safe_float(fo.loc[row_name].iloc[0])
                        if val is not None:
                            return val if val < 1 else val / 100.0
                    except Exception:
                        pass
    except Exception:
        pass

    # Fallback to info fields (older yfinance)
    for k in ("annualReportExpenseRatio", "netExpenseRatio", "expenseRatio"):
        v = _safe_float(info.get(k))
        if v is not None:
            return v if v < 1 else v / 100.0

    return None


def _extract_fund_overview(funds_data) -> dict:
    out = {"category_name": None, "family": None, "legal_type": None}
    try:
        fo = funds_data.fund_overview
        if fo is None:
            return out
        if isinstance(fo, dict):
            out["category_name"] = fo.get("categoryName")
            out["family"] = fo.get("family")
            out["legal_type"] = fo.get("legalType")
        else:
            # DataFrame fallback
            try:
                row = fo.iloc[0]
                out["category_name"] = row.get("categoryName")
                out["family"] = row.get("family")
                out["legal_type"] = row.get("legalType")
            except Exception:
                pass
    except Exception:
        pass
    return out


def analyze_etf(ticker: str) -> dict:
    ticker = ticker.upper().strip()
    cache_key = f"etf_{ticker}"
    cached = _cache_get(cache_key, ttl_seconds=900)  # 15 min for full payload
    if cached is not None:
        cached["from_cache"] = True
        return cached

    t = yf.Ticker(ticker)

    # ── Verify ticker is an ETF ────────────────────────────────────────────
    info = {}
    try:
        info = t.info or {}
    except Exception:
        try:
            info = t.get_info() or {}
        except Exception:
            info = {}

    quote_type = (info.get("quoteType") or "").upper()
    if quote_type and quote_type not in ("ETF", "MUTUALFUND"):
        return {
            "error": f"{ticker} is a {quote_type.title()}, not an ETF. "
                     f"Use the regular Analyze view for individual stocks."
        }

    # funds_data is the yfinance accessor for ETF / fund details
    try:
        fd = t.funds_data
    except Exception as e:
        return {"error": f"No fund data available for {ticker}: {e}"}

    fund_overview = _extract_fund_overview(fd)
    sector_weights = {}
    try:
        sw = fd.sector_weightings
        if isinstance(sw, dict):
            sector_weights = {k: _safe_float(v) for k, v in sw.items() if _safe_float(v)}
            # Normalize keys to title case
            sector_weights = {k.replace("_", " ").title(): v for k, v in sector_weights.items()}
    except Exception:
        pass

    raw_holdings = _extract_top_holdings(fd)
    expense_ratio = _extract_expense_ratio(fd, info)

    # ── ETF metadata ──────────────────────────────────────────────────────
    family = fund_overview.get("family") or info.get("fundFamily")
    category = fund_overview.get("category_name") or info.get("category")
    legal_type = fund_overview.get("legal_type") or info.get("legalType") or "ETF"

    # ETF current price (separate yfinance call is unnecessary — info has it)
    etf_price = _safe_float(
        info.get("regularMarketPrice")
        or info.get("currentPrice")
        or info.get("previousClose")
        or info.get("navPrice")
    )

    inception = None
    ftd = info.get("firstTradeDateEpochUtc")
    if ftd:
        try:
            inception = datetime.fromtimestamp(int(ftd), tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            pass

    etf_meta = {
        "ticker": ticker,
        "long_name": info.get("longName") or info.get("shortName"),
        "family": family,
        "category": category,
        "legal_type": legal_type,
        "exchange": info.get("exchange") or info.get("fullExchangeName"),
        "currency": info.get("currency"),
        "total_assets": _safe_int(info.get("totalAssets")),
        "nav_price": _safe_float(info.get("navPrice")),
        "current_price": etf_price,
        "expense_ratio": expense_ratio,
        "yield": _safe_float(info.get("yield")),
        "ytd_return": _safe_float(info.get("ytdReturn")),
        "three_year_avg_return": _safe_float(info.get("threeYearAverageReturn")),
        "five_year_avg_return": _safe_float(info.get("fiveYearAverageReturn")),
        "inception_date": inception,
        "management_style": _management_style(info, fund_overview, ticker),
        "tracks_index": _index_for(ticker, category),
        "rebalance_schedule": _rebalance_for(family or "", ticker),
        "long_business_summary": info.get("longBusinessSummary"),
    }

    # ── ETF historical performance ────────────────────────────────────────
    etf_performance = _fetch_period_returns(t)

    # ── Holdings: fetch in parallel ───────────────────────────────────────
    holdings_full = []
    holdings_coverage = sum(w for _, w in raw_holdings) if raw_holdings else 0.0

    if raw_holdings:
        # Cap concurrent threads — yfinance gets rate-limited if hammered
        max_workers = min(6, len(raw_holdings))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_fetch_holding_snapshot, sym, w): sym
                for sym, w in raw_holdings
            }
            for fut in as_completed(futures):
                try:
                    holdings_full.append(fut.result())
                except Exception as e:
                    sym = futures[fut]
                    holdings_full.append({
                        "symbol": sym,
                        "weight": next((w for s, w in raw_holdings if s == sym), 0),
                        "fetch_error": str(e)[:200],
                    })

        # Preserve original weight ordering (descending)
        weight_order = {sym: i for i, (sym, _) in enumerate(raw_holdings)}
        holdings_full.sort(key=lambda h: weight_order.get(h.get("symbol"), 999))

    # ── Compute contribution-to-return for each period and tag drags ──────
    # contribution_to_etf_return[period] = weight × holding_return[period]
    for h in holdings_full:
        perf = h.get("performance") or {}
        contrib = {}
        drag_flag = {}
        for p in _PERIODS:
            r = perf.get(p)
            etf_r = (etf_performance or {}).get(p)
            if r is not None and h.get("weight") is not None:
                contrib[p] = h["weight"] * r
            else:
                contrib[p] = None
            if r is not None and etf_r is not None:
                drag_flag[p] = r < etf_r
            else:
                drag_flag[p] = None
        h["contribution"] = contrib
        h["is_drag"] = drag_flag

    payload = {
        "ticker": ticker,
        "info": etf_meta,
        "sector_weights": sector_weights,
        "holdings": holdings_full,
        "holdings_coverage": round(holdings_coverage, 4),
        "holdings_returned": len(holdings_full),
        "holdings_truncated": holdings_coverage < 0.99,
        "etf_performance": etf_performance,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data_source_notes": (
            "ETF metadata + top holdings via Yahoo Finance (yfinance). "
            "Per-holding details cached 1h, ETF metadata cached 15min. "
            "Yahoo returns top-N holdings (typically 10) — not the complete fund. "
            "Rebalance schedules are typical-by-family; not the literal last rebalance date. "
            "SPAC flag is a name-pattern heuristic only. "
            "Listing date = firstTradeDateEpochUtc (the SPAC IPO date for deSPACs)."
        ),
        "from_cache": False,
    }

    _cache_set(cache_key, payload)
    return payload


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No ticker provided"}))
        sys.exit(1)

    cmd = sys.argv[1].upper().strip()
    try:
        result = analyze_etf(cmd)
        print(json.dumps(_clean_nan(result)))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
