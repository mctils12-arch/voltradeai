#!/usr/bin/env python3
"""
VolTradeAI — ETF Builder (production-source edition)

Uses the same data infrastructure as the bot:
  - Alpaca SIP for all price history & current prices (BATCH endpoint —
    pulls 5y daily bars for ALL holdings in 1-2 HTTP calls)
  - Polygon /v3/reference/tickers for per-holding metadata (sector via SIC,
    industry, list_date, market cap, shares outstanding, description, etc.)
  - Finnhub /stock/metric for valuation (P/E, beta, dividend yield, 52w range)
  - NASDAQ public API for the full holdings list (all 100 of QQQ, 500 of SPY)
  - yfinance kept ONLY for ETF-specific fund accessors (expense ratio,
    sector weightings, total assets) — its weakest spot (per-stock metadata)
    is no longer used.

Output: single JSON object to stdout.

Usage: python3 etf_analyzer.py SPY
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
    import yfinance as yf
except Exception as e:
    print(json.dumps({"error": f"yfinance not available: {e}"}))
    sys.exit(1)

import etf_data_sources as ds


# ─── Cache (filesystem) ──────────────────────────────────────────────────────

CACHE_DIR = "/tmp/voltrade_etf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_get(key, ttl_seconds):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_seconds:
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _cache_set(key, data):
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


# ─── JSON cleanliness ────────────────────────────────────────────────────────

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
    try:
        if x is None: return None
        v = float(x)
        if math.isnan(v) or math.isinf(v): return None
        return v
    except Exception:
        return None


def _safe_int(x):
    v = _safe_float(x)
    return int(v) if v is not None else None


# ─── Family / index heuristics ───────────────────────────────────────────────

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
    "VNQ": "MSCI US Investable Market Real Estate 25/50",
    "REM": "FTSE NAREIT All Mortgage Capped",
    "MORT": "MVIS US Mortgage REITs",
}


def _rebalance_for(family: str, ticker: str) -> str:
    if ticker.upper() in _INDEX_BY_TICKER and "Active" in _INDEX_BY_TICKER[ticker.upper()]:
        return "Active — discretionary, no fixed schedule"
    fam = family or ""
    for key, sched in _REBALANCE_BY_FAMILY:
        if key.lower() in fam.lower():
            return sched
    return "Check issuer prospectus for the official rebalance schedule"


def _index_for(ticker: str, category: str) -> str:
    t = ticker.upper()
    if t in _INDEX_BY_TICKER:
        return _INDEX_BY_TICKER[t]
    if category:
        return f"(Tracks a {category.lower()} benchmark)"
    return "Unknown"


def _management_style(info: dict, ticker: str) -> str:
    t = ticker.upper()
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
    return "Passive (index-tracking)"


# ─── SPAC heuristic ──────────────────────────────────────────────────────────

_SPAC_NAME_PATTERNS = [
    "acquisition corp", "acquisition corporation",
    "capital partners corp", "growth opportunities corp",
]


def _spac_heuristic(long_name, short_name) -> bool:
    if not long_name and not short_name:
        return False
    blob = ((long_name or "") + " " + (short_name or "")).lower()
    return any(p in blob for p in _SPAC_NAME_PATTERNS)


# ─── Holdings sources from yfinance (last-resort fallback) ──────────────────

def _extract_top_holdings_yfinance(funds_data):
    """Used only if NASDAQ holdings API is unavailable. Caps at top-10."""
    try:
        th = funds_data.top_holdings
    except Exception:
        return []

    holdings = []
    try:
        if hasattr(th, "iterrows"):
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
                    for c in th.columns:
                        v = _safe_float(row[c])
                        if v is not None and 0 < v <= 1.5:
                            w = v
                            break
                if w is None:
                    continue
                if w > 1.5:
                    w = w / 100.0
                holdings.append({"symbol": sym, "weight": w, "name": None})
        elif isinstance(th, dict):
            for sym, w in th.items():
                wv = _safe_float(w)
                if wv is None:
                    continue
                if wv > 1.5:
                    wv = wv / 100.0
                holdings.append({"symbol": str(sym).upper(), "weight": wv, "name": None})
    except Exception:
        pass
    return holdings


def _extract_expense_ratio(funds_data, info):
    """yfinance is still the best free source for expense ratio."""
    try:
        fo = funds_data.fund_operations
        if fo is not None and hasattr(fo, "loc"):
            for row_name in ("Annual Report Expense Ratio", "annualReportExpenseRatio",
                             "Expense Ratio", "expenseRatio"):
                if row_name in fo.index:
                    try:
                        val = _safe_float(fo.loc[row_name].iloc[0])
                        if val is not None:
                            return val if val < 1 else val / 100.0
                    except Exception:
                        pass
    except Exception:
        pass
    for k in ("annualReportExpenseRatio", "netExpenseRatio", "expenseRatio"):
        v = _safe_float(info.get(k))
        if v is not None:
            return v if v < 1 else v / 100.0
    return None


def _extract_fund_overview(funds_data):
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


# ─── Per-holding fetch (rich metadata for top-N) ────────────────────────────

_PERIODS = ["1mo", "3mo", "6mo", "ytd", "1y", "5y"]
TOP_N_RICH = 30  # how many holdings get full metadata (Polygon + Finnhub)


def _fetch_holding_full(symbol, weight, name_hint, perf_data):
    """Build a holding payload with rich metadata.

    perf_data = {"performance": {...}, "price": float, "history_points": int}
    pre-computed in the batch Alpaca call. So this function only needs to
    fetch the per-symbol metadata (Polygon + Finnhub).
    """
    cache_key = f"holding_full_v2_{symbol.upper()}"
    cached = _cache_get(cache_key, ttl_seconds=3600)

    if cached is None:
        meta = ds.fetch_holding_metadata(symbol)
        cached = meta
        _cache_set(cache_key, cached)

    out = dict(cached)
    # Always re-apply current weight, name hint, and live perf data
    out["symbol"] = symbol
    out["weight"] = weight
    if name_hint and not out.get("name"):
        out["name"] = name_hint
    out["price"] = perf_data.get("price")
    out["performance"] = perf_data.get("performance") or {p: None for p in _PERIODS}
    out["data_level"] = "full"

    # SPAC heuristic from the name we have now
    out["is_spac_heuristic"] = _spac_heuristic(out.get("name"), None)

    # 52w position percentage (if we have both bounds and a current price)
    h_hi, h_lo, p = _safe_float(out.get("52w_high")), _safe_float(out.get("52w_low")), out.get("price")
    if h_hi and h_lo and p and h_hi > h_lo:
        out["52w_pos_pct"] = max(0.0, min(1.0, (p - h_lo) / (h_hi - h_lo)))
    else:
        out["52w_pos_pct"] = None

    return out


def _fetch_holding_slim(symbol, weight, name_hint, perf_data):
    """Lightweight holding payload — symbol, weight, price, returns only.

    Used for the long tail beyond top-N. No metadata API calls.
    """
    return {
        "symbol": symbol,
        "weight": weight,
        "name": name_hint,
        "price": perf_data.get("price"),
        "performance": perf_data.get("performance") or {p: None for p in _PERIODS},
        "data_level": "slim",
    }


# ─── ETF top-level fetch ─────────────────────────────────────────────────────

def analyze_etf(ticker: str) -> dict:
    ticker = ticker.upper().strip()
    cache_key = f"etf_v3_{ticker}"  # bumped v2 → v3 because data sources changed
    cached = _cache_get(cache_key, ttl_seconds=900)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    # ── Classification: Polygon's `type` field is the source of truth ────
    # If Polygon says it's not ETF/ETN/ETV, return a friendly classified error.
    classification = ds.classify_ticker(ticker)
    if classification:
        cid, msg, suggested = classification
        if cid not in ("etf", "etn", "etv"):
            return {
                "error": msg,
                "error_classification": cid,
                "suggested_ticker": suggested,
                "queried_ticker": ticker,
            }

    # ── yfinance for ETF-specific fund data (expense ratio, sector weights) ──
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        try:
            info = t.get_info() or {}
        except Exception:
            info = {}

    # If we didn't get Polygon classification AND yfinance also reports
    # this isn't an ETF, classify via the legacy path
    if classification is None:
        quote_type = (info.get("quoteType") or "").upper()
        if quote_type and quote_type not in ("ETF", "MUTUALFUND"):
            info_with_symbol = dict(info)
            info_with_symbol["symbol"] = ticker
            cid, msg, suggested = ds.classify_non_etf(info_with_symbol)
            return {
                "error": msg,
                "error_classification": cid,
                "suggested_ticker": suggested,
                "queried_ticker": ticker,
            }

    try:
        fd = t.funds_data
    except Exception:
        fd = None

    fund_overview = _extract_fund_overview(fd) if fd else {"category_name": None, "family": None, "legal_type": None}
    sector_weights = {}
    if fd:
        try:
            sw = fd.sector_weightings
            if isinstance(sw, dict):
                sector_weights = {k.replace("_", " ").title(): _safe_float(v)
                                  for k, v in sw.items() if _safe_float(v)}
        except Exception:
            pass

    expense_ratio = _extract_expense_ratio(fd, info) if fd else None
    family = fund_overview.get("family") or info.get("fundFamily")
    category = fund_overview.get("category_name") or info.get("category")
    legal_type = fund_overview.get("legal_type") or info.get("legalType") or "ETF"

    # Inception: Polygon list_date is authoritative for nearly all ETFs.
    # Fall through to static table, then yfinance, then None.
    poly_etf = ds.fetch_ticker_details_polygon(ticker)
    inception = (poly_etf or {}).get("list_date")
    if not inception:
        inception = ds.ETF_INCEPTION.get(ticker)
    if not inception:
        ftd = info.get("firstTradeDateEpochUtc")
        if ftd:
            try:
                inception = datetime.fromtimestamp(int(ftd), tz=timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                pass

    # ── HOLDINGS LIST: full list from NASDAQ; fall back to yfinance top-N ──
    holdings_meta = ds.fetch_full_holdings_nasdaq(ticker)
    holdings_source = "nasdaq"
    if not holdings_meta:
        holdings_meta = _extract_top_holdings_yfinance(fd) if fd else []
        holdings_source = "yfinance_top_n" if holdings_meta else "none"

    # ── BATCH PRICE/RETURN FETCH for ALL holdings + the ETF itself ────────
    # One Alpaca batch call (chunked at 50 symbols/request internally) covers
    # the entire ETF in 1-2 round trips. Much faster than per-symbol.
    all_syms = [ticker] + [h["symbol"] for h in (holdings_meta or [])]
    perf_map = ds.compute_returns_for_symbols(all_syms, history_days=1900)

    # ETF-level price & performance from the batch
    etf_pkg = perf_map.get(ticker, {})
    etf_performance = etf_pkg.get("performance") or {p: None for p in _PERIODS}
    etf_price = etf_pkg.get("price") or _safe_float(
        info.get("regularMarketPrice")
        or info.get("currentPrice")
        or info.get("previousClose")
        or info.get("navPrice")
    )

    etf_meta = {
        "ticker": ticker,
        "long_name": (info.get("longName") or info.get("shortName")
                      or (poly_etf or {}).get("name")),
        "family": family,
        "category": category,
        "legal_type": legal_type,
        "exchange": (info.get("exchange") or info.get("fullExchangeName")
                     or (poly_etf or {}).get("exchange")),
        "currency": info.get("currency") or (poly_etf or {}).get("currency"),
        "total_assets": _safe_int(info.get("totalAssets")),
        "nav_price": _safe_float(info.get("navPrice")),
        "current_price": etf_price,
        "expense_ratio": expense_ratio,
        "yield": _safe_float(info.get("yield")),
        "ytd_return": _safe_float(info.get("ytdReturn")),
        "three_year_avg_return": _safe_float(info.get("threeYearAverageReturn")),
        "five_year_avg_return": _safe_float(info.get("fiveYearAverageReturn")),
        "inception_date": inception,
        "management_style": _management_style(info, ticker),
        "tracks_index": _index_for(ticker, category),
        "rebalance_schedule": _rebalance_for(family or "", ticker),
        "long_business_summary": (info.get("longBusinessSummary")
                                   or (poly_etf or {}).get("description")),
    }

    # ── Build per-holding records ─────────────────────────────────────────
    holdings_full = []
    holdings_coverage = sum(h["weight"] for h in (holdings_meta or [])) if holdings_meta else 0.0

    if holdings_meta:
        ranked = sorted(holdings_meta, key=lambda h: h.get("weight") or 0, reverse=True)
        top_n = ranked[:TOP_N_RICH]
        rest = ranked[TOP_N_RICH:]

        # Rich metadata for top-N (Polygon + Finnhub via threadpool)
        max_workers = min(8, max(1, len(top_n)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for h in top_n:
                pd_perf = perf_map.get(h["symbol"], {})
                futures.append(ex.submit(_fetch_holding_full,
                                          h["symbol"], h["weight"],
                                          h.get("name"), pd_perf))
            for fut in as_completed(futures):
                try:
                    holdings_full.append(fut.result())
                except Exception:
                    pass

        # Slim for the long tail — no metadata calls, just batch perf data
        for h in rest:
            pd_perf = perf_map.get(h["symbol"], {})
            holdings_full.append(_fetch_holding_slim(
                h["symbol"], h["weight"], h.get("name"), pd_perf))

        holdings_full.sort(key=lambda h: h.get("weight") or 0, reverse=True)

    # ── Contribution + drag flags per period ──────────────────────────────
    for h in holdings_full:
        perf = h.get("performance") or {}
        contrib = {}
        drag_flag = {}
        for p in _PERIODS:
            r = perf.get(p)
            etf_r = etf_performance.get(p)
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

    coverage = round(holdings_coverage, 4)
    is_complete = coverage >= 0.90

    payload = {
        "ticker": ticker,
        "info": etf_meta,
        "sector_weights": sector_weights,
        "holdings": holdings_full,
        "holdings_coverage": coverage,
        "holdings_returned": len(holdings_full),
        "holdings_truncated": not is_complete,
        "holdings_source": holdings_source,
        "etf_performance": etf_performance,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data_source_notes": (
            "Price history & current prices: Alpaca Market Data API (SIP feed, "
            "adjusted for splits & dividends). "
            "Per-holding company data: Polygon reference + Finnhub financial metrics. "
            "Full holdings list: NASDAQ ETF data. "
            "Fund-level data (expense ratio, sector weightings): issuer feeds. "
            "Cached: ETF payload 15min, per-holding metadata 24h, "
            "Finnhub metrics 6h, price history 12h."
        ),
        "data_sources_used": {
            "alpaca": ds.have_alpaca(),
            "polygon": ds.have_polygon(),
            "finnhub": ds.have_finnhub(),
        },
        "from_cache": False,
    }

    _cache_set(cache_key, payload)
    return payload


# ─── Entry point ─────────────────────────────────────────────────────────────

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
