"""
VolTradeAI Institutional Holdings Module
Parses SEC EDGAR 13F filings to detect institutional interest signals.
All data is free from SEC EDGAR with aggressive caching to avoid rate limits.
"""
import json
import os
import time
import re
import requests
from datetime import datetime, timedelta

CACHE_DIR = "/tmp/voltrade_alt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

USER_AGENT = "VolTradeAI research@voltradeai.com"

# ── Cache helpers (matches alt_data.py pattern) ───────────────────────────────

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


# ── Super-investor CIKs ───────────────────────────────────────────────────────

WHALE_CIKS = {
    "Berkshire Hathaway":  "0001067983",
    "Bridgewater Associates": "0001350694",
    "Renaissance Technologies": "0001037389",
    "Citadel Advisors":    "0001423053",
    "D.E. Shaw":           "0001009207",
    "Two Sigma":           "0001442145",
    "Millennium Management": "0001273087",
    "Point72":             "0001603466",
}

# Top 50 institutions by AUM (representative sample — CIKs for major filers)
TOP_INSTITUTION_CIKS = {
    "BlackRock":           "0001086364",
    "Vanguard":            "0000102909",
    "State Street":        "0000093751",
    "Fidelity":            "0000315066",
    "JPMorgan":            "0000019617",
    "Goldman Sachs":       "0000886982",
    "Morgan Stanley":      "0000895421",
    "T. Rowe Price":       "0001113169",
    "Capital Group":       "0000277751",
    "Wellington Management": "0000101855",
    "Berkshire Hathaway":  "0001067983",
    "Bridgewater Associates": "0001350694",
    "Renaissance Technologies": "0001037389",
    "Citadel Advisors":    "0001423053",
    "D.E. Shaw":           "0001009207",
    "Two Sigma":           "0001442145",
    "Millennium Management": "0001273087",
    "Point72":             "0001603466",
    "Invesco":             "0000914208",
    "Northern Trust":      "0000073124",
}


# ── Helper: current 13F quarter window ───────────────────────────────────────

def _current_13f_window():
    """
    Return (start_date, end_date) strings for the most recent completed
    13F filing quarter. 13F filings are due 45 days after quarter end,
    so we look at the last completed quarter.
    """
    now = datetime.now()
    # Most recent quarter end
    month = now.month
    if month <= 3:
        qtr_end = datetime(now.year - 1, 12, 31)
    elif month <= 6:
        qtr_end = datetime(now.year, 3, 31)
    elif month <= 9:
        qtr_end = datetime(now.year, 6, 30)
    else:
        qtr_end = datetime(now.year, 9, 30)

    # 13F filings appear ~45 days after quarter end
    filing_deadline = qtr_end + timedelta(days=45)
    if now < filing_deadline:
        # Previous quarter not yet filed — go back another quarter
        if qtr_end.month == 12:
            qtr_end = datetime(qtr_end.year - 1, 9, 30)
        elif qtr_end.month == 9:
            qtr_end = datetime(qtr_end.year, 6, 30)
        elif qtr_end.month == 6:
            qtr_end = datetime(qtr_end.year, 3, 31)
        else:
            qtr_end = datetime(qtr_end.year - 1, 12, 31)

    # Search window: from 45 days before quarter end to 60 days after
    start = (qtr_end - timedelta(days=45)).strftime("%Y-%m-%d")
    end = (qtr_end + timedelta(days=60)).strftime("%Y-%m-%d")
    return start, end


# ── Helper: fetch 13F filings for a ticker via EFTS ──────────────────────────

def _fetch_13f_hits(ticker: str, start: str, end: str) -> list:
    """
    Query SEC EFTS full-text search for 13F-HR filings mentioning ticker.
    Returns list of hit dicts from the EFTS response.
    """
    url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
        f"&dateRange=custom&startdt={start}&enddt={end}&forms=13F-HR"
    )
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("hits", {}).get("hits", [])
    return hits


# ── Helper: fetch CIK's most recent 13F filing list ──────────────────────────

def _fetch_cik_filings(cik: str, count: int = 5) -> list:
    """
    Use SEC EDGAR submissions API to get recent 13F filings for a CIK.
    Returns list of accession numbers.
    """
    # Normalize CIK to 10 digits
    cik_norm = cik.lstrip("0")
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])

    results = []
    for i, form in enumerate(forms):
        if form in ("13F-HR", "13F-HR/A") and len(results) < count:
            results.append({
                "accession": accessions[i],
                "date": dates[i],
                "form": form,
            })
    return results


# ── Helper: check if a 13F document mentions a ticker ────────────────────────

def _filing_mentions_ticker(accession: str, cik: str, ticker: str) -> bool:
    """
    Fetch the 13F filing index and check if ticker appears in the info table.
    Uses the SEC EDGAR XBRL/viewer document endpoint.
    Gracefully returns False on any failure.
    """
    try:
        # Build the document index URL
        acc_clean = accession.replace("-", "")
        cik_norm = cik.zfill(10)
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_norm}"
            f"/{acc_clean}/{accession}-index.json"
        )
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(index_url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return False

        index_data = resp.json()
        # Look for the information table document
        for doc in index_data.get("directory", {}).get("item", []):
            name = doc.get("name", "").lower()
            if "infotable" in name or "information_table" in name or name.endswith(".xml"):
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_norm}"
                    f"/{acc_clean}/{doc['name']}"
                )
                doc_resp = requests.get(doc_url, headers=headers, timeout=12)
                if doc_resp.status_code == 200:
                    # Simple text search — ticker appears in <nameOfIssuer> tags
                    content = doc_resp.text.upper()
                    return ticker.upper() in content
    except Exception:
        pass
    return False


# ── 1. get_institutional_signal ───────────────────────────────────────────────

def get_institutional_signal(ticker: str) -> dict:
    """
    Check if major institutions (top 50 by AUM) recently added/increased
    position in this ticker by parsing SEC EDGAR 13F-HR filings.

    Uses EFTS full-text search to find 13F filings mentioning the ticker,
    then cross-references with known top institution CIKs.

    Returns:
        {
            "institutional_interest": "increasing" | "decreasing" | "stable",
            "major_holders_added": int,
            "major_holders_reduced": int,
            "signal_strength": 0-100,
            "top_holders": [list of institution names],
        }

    Cached for 24 hours (quarterly data).
    """
    cache_key = f"inst13f_{ticker.upper()}"
    cached = _cache_get(cache_key, ttl_seconds=86400)
    if cached:
        return cached

    result = {
        "institutional_interest": "stable",
        "major_holders_added": 0,
        "major_holders_reduced": 0,
        "signal_strength": 0,
        "top_holders": [],
    }

    try:
        start, end = _current_13f_window()

        # ── Step 1: EFTS full-text search for recent 13F filings mentioning ticker
        hits = _fetch_13f_hits(ticker, start, end)

        if not hits:
            # Widen the window to 6 months to catch any filing
            wide_start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            wide_end = datetime.now().strftime("%Y-%m-%d")
            hits = _fetch_13f_hits(ticker, wide_start, wide_end)

        total_hits = len(hits)

        # ── Step 2: Cross-reference filing entities with known top institutions
        top_holders_found = []
        filer_names = [h.get("_source", {}).get("entity_name", "") for h in hits]

        for inst_name, inst_cik in TOP_INSTITUTION_CIKS.items():
            # Check if institution name appears in any filer
            inst_lower = inst_name.lower()
            for fname in filer_names:
                fname_lower = fname.lower()
                # Partial match: e.g. "blackrock" in "blackrock fund advisors"
                if any(word in fname_lower for word in inst_lower.split() if len(word) > 4):
                    if inst_name not in top_holders_found:
                        top_holders_found.append(inst_name)
                    break

        result["top_holders"] = top_holders_found[:10]

        # ── Step 3: Compare current vs prior quarter filing counts for signal direction
        # Current quarter
        curr_start, curr_end = _current_13f_window()
        curr_hits = _fetch_13f_hits(ticker, curr_start, curr_end)
        curr_count = len(curr_hits)

        # Prior quarter (go back ~90 more days)
        prior_end_dt = datetime.strptime(curr_start, "%Y-%m-%d") - timedelta(days=1)
        prior_start_dt = prior_end_dt - timedelta(days=135)
        prior_start = prior_start_dt.strftime("%Y-%m-%d")
        prior_end = prior_end_dt.strftime("%Y-%m-%d")
        prior_hits = _fetch_13f_hits(ticker, prior_start, prior_end)
        prior_count = len(prior_hits)

        # Determine direction
        if prior_count > 0:
            ratio = curr_count / prior_count
            if ratio > 1.15:
                result["institutional_interest"] = "increasing"
                result["major_holders_added"] = max(0, curr_count - prior_count)
            elif ratio < 0.85:
                result["institutional_interest"] = "decreasing"
                result["major_holders_reduced"] = max(0, prior_count - curr_count)
            else:
                result["institutional_interest"] = "stable"
        elif curr_count > 0:
            result["institutional_interest"] = "increasing"
            result["major_holders_added"] = curr_count
        else:
            result["institutional_interest"] = "stable"

        # ── Step 4: Signal strength (0-100)
        # Based on: number of major holders found + direction + total filing count
        base = min(len(top_holders_found) * 8, 50)  # Up to 50 pts from major holders
        filing_pts = min(curr_count * 2, 30)         # Up to 30 pts from filing volume

        direction_pts = 0
        if result["institutional_interest"] == "increasing":
            direction_pts = 20
        elif result["institutional_interest"] == "decreasing":
            direction_pts = -20

        raw_strength = base + filing_pts + direction_pts
        result["signal_strength"] = max(0, min(100, raw_strength))

    except Exception as e:
        result["error"] = str(e)

    _cache_set(cache_key, result)
    return result


# ── 2. get_whale_activity ─────────────────────────────────────────────────────

def get_whale_activity(ticker: str) -> dict:
    """
    Check if any "super investor" CIK filed a 13F mentioning this ticker.
    Uses SEC EFTS full-text search to find filings from known whale CIKs.

    Returns:
        {
            "whale_interest": bool,
            "whales": [list of names that hold it],
            "signal": "bullish" | "neutral",
        }

    Cached for 24 hours.
    """
    cache_key = f"whale13f_{ticker.upper()}"
    cached = _cache_get(cache_key, ttl_seconds=86400)
    if cached:
        return cached

    result = {
        "whale_interest": False,
        "whales": [],
        "signal": "neutral",
    }

    try:
        # Search window: last 6 months to catch most recent filing cycle
        start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")

        hits = _fetch_13f_hits(ticker, start, end)

        if not hits:
            _cache_set(cache_key, result)
            return result

        # Build a set of entity names from hits for quick lookup
        filer_names_lower = [
            h.get("_source", {}).get("entity_name", "").lower()
            for h in hits
        ]
        filer_ciks = [
            str(h.get("_source", {}).get("file_num", "")).replace("028-", "")
            for h in hits
        ]

        whales_found = []

        for whale_name, whale_cik in WHALE_CIKS.items():
            matched = False

            # Method 1: Check by entity name in search results
            whale_words = [w for w in whale_name.lower().split() if len(w) > 4]
            for fname in filer_names_lower:
                if any(word in fname for word in whale_words):
                    matched = True
                    break

            # Method 2: Directly check recent filings for this whale CIK
            if not matched:
                try:
                    filings = _fetch_cik_filings(whale_cik, count=3)
                    for filing in filings:
                        acc = filing["accession"]
                        if _filing_mentions_ticker(acc, whale_cik, ticker):
                            matched = True
                            break
                except Exception:
                    pass

            if matched:
                whales_found.append(whale_name)

        result["whales"] = whales_found
        result["whale_interest"] = len(whales_found) > 0
        result["signal"] = "bullish" if whales_found else "neutral"

    except Exception as e:
        result["error"] = str(e)

    _cache_set(cache_key, result)
    return result


# ── Combined institutional score ──────────────────────────────────────────────

def get_institutional_score(ticker: str) -> dict:
    """
    Run both institutional data functions and return a combined score.
    Returns institutional signal, whale activity, and combined signal.
    """
    inst = get_institutional_signal(ticker)
    whale = get_whale_activity(ticker)

    # Combined signal strength
    inst_score = inst.get("signal_strength", 0)
    whale_bonus = 20 if whale.get("whale_interest") else 0

    direction_mult = 1
    if inst.get("institutional_interest") == "decreasing":
        direction_mult = -1

    combined = (inst_score + whale_bonus) * direction_mult

    return {
        "combined_score": combined,
        "institutional": inst,
        "whale": whale,
    }


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"=== Institutional Data Report for {ticker} ===")
    report = get_institutional_score(ticker)
    print(json.dumps(report, indent=2, default=str))
