"""
AlphaDesk — SEC EDGAR adapter (free, public; no API key, only a descriptive
User-Agent per SEC fair-access policy).

The flow is: ticker -> CIK -> recent submissions -> the latest N material
filings -> (optionally) the primary document text. The parsing steps are pure
functions that take already-fetched JSON/HTML, so they can be unit-checked
offline (see `alphadesk selftest`); only `fetch_filings` touches the network.

Endpoints used:
  - Ticker->CIK map: https://www.sec.gov/files/company_tickers.json
  - Submissions:     https://data.sec.gov/submissions/CIK##########.json
  - Documents:       https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<doc>
"""
from __future__ import annotations

import json
import re
import urllib.request
from html import unescape
from typing import List, Optional

from .models import Filing

# Forms that carry a narrative worth reading. Periodic reports + current events,
# plus foreign private issuers (6-K current reports / 20-F annual) — without these
# a company like ZIM (which files 6-K/20-F, not 10-K/8-K) would return nothing and
# its merger 6-K would be missed. Amendments (…/A) are matched too; insider Form 4
# noise is excluded.
MATERIAL_FORMS = ("10-K", "10-Q", "8-K", "6-K", "20-F")

# Cap on extracted document text. The heuristic reasoner only keyword-scans the
# body, and full 10-Ks are megabytes — keep reports fast and memory-bounded.
TEXT_CAP = 60_000


# --------------------------------------------------------------------------- #
# Network (the only side-effecting function)
# --------------------------------------------------------------------------- #
def _http_get(url: str, ua: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# --------------------------------------------------------------------------- #
# Pure parsing helpers (offline-testable)
# --------------------------------------------------------------------------- #
def resolve_cik(ticker: str, tickers_json: dict) -> Optional[str]:
    """
    Map a ticker to a zero-padded 10-digit CIK using the company_tickers.json
    payload (a dict of {idx: {"cik_str", "ticker", "title"}}). Case-insensitive.
    Returns None if the ticker isn't found.
    """
    t = ticker.upper()
    for row in tickers_json.values():
        if str(row.get("ticker", "")).upper() == t:
            return str(row["cik_str"]).zfill(10)
    return None


def _doc_url(cik10: str, accession: str, primary_doc: str) -> str:
    """Build the archive URL for a filing's primary document."""
    cik_int = str(int(cik10))                 # drop leading zeros for the path
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}"


def parse_recent_filings(
    ticker: str,
    cik10: str,
    submissions_json: dict,
    limit: int = 5,
    forms: tuple = MATERIAL_FORMS,
) -> List[Filing]:
    """
    Build Filing objects (without body text) from a submissions payload, newest
    first, keeping only material forms. Pure: takes the parsed JSON, no network.

    The `filings.recent` block is column-oriented (parallel arrays); we zip the
    columns back into rows. Tolerant of missing columns.
    """
    recent = (submissions_json.get("filings", {}) or {}).get("recent", {}) or {}
    col_form = recent.get("form", [])
    col_date = recent.get("filingDate", [])
    col_acc = recent.get("accessionNumber", [])
    col_doc = recent.get("primaryDocument", [])
    col_desc = recent.get("primaryDocDescription", [])
    t = ticker.upper()

    def _allowed(form: str) -> bool:
        base = form.split("/")[0]            # "10-K/A" -> "10-K"
        return base in forms

    out: List[Filing] = []
    for i, form in enumerate(col_form):
        if not _allowed(form):
            continue
        acc = col_acc[i] if i < len(col_acc) else ""
        doc = col_doc[i] if i < len(col_doc) else ""
        desc = col_desc[i] if i < len(col_desc) else ""
        date = col_date[i] if i < len(col_date) else ""
        url = _doc_url(cik10, acc, doc) if (acc and doc) else ""
        out.append(Filing(
            ticker=t,
            form=form,
            filed_at=date,
            title=(desc or f"{t} {form}").strip(),
            url=url,
            text="",
        ))
        if len(out) >= limit:
            break
    return out


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def html_to_text(raw: str, cap: int = TEXT_CAP) -> str:
    """
    Reduce a filing document (HTML or plain text) to collapsed, unescaped plain
    text, truncated to `cap` chars. Good enough for the keyword heuristic and
    cheap to compute. Pure.
    """
    # Drop script/style blocks wholesale, then all remaining tags.
    no_blocks = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    text = _TAG_RE.sub(" ", no_blocks)
    text = unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:cap]


# --------------------------------------------------------------------------- #
# Network orchestrator
# --------------------------------------------------------------------------- #
def fetch_filings(
    ticker: str,
    ua: str,
    limit: int = 5,
    forms: tuple = MATERIAL_FORMS,
    fetch_text: bool = True,
    text_docs: int = 3,
    timeout: int = 20,
) -> List[Filing]:
    """
    Resolve CIK, pull recent submissions, and return the latest `limit` material
    filings. When `fetch_text` is set, download and extract plain text for up to
    `text_docs` of the newest documents (so the filing reasoner reads real
    content); the rest carry metadata only. Raises on hard failure so the caller
    can fall back to sample data.
    """
    tickers_json = json.loads(_http_get("https://www.sec.gov/files/company_tickers.json", ua, timeout))
    cik10 = resolve_cik(ticker, tickers_json)
    if not cik10:
        raise ValueError(f"CIK not found for {ticker}")

    sub = json.loads(_http_get(f"https://data.sec.gov/submissions/CIK{cik10}.json", ua, timeout))
    filings = parse_recent_filings(ticker, cik10, sub, limit=limit, forms=forms)

    if fetch_text:
        for f in filings[:text_docs]:
            if not f.url:
                continue
            try:
                raw = _http_get(f.url, ua, timeout).decode("utf-8", "ignore")
                f.text = html_to_text(raw)
            except Exception:
                # best-effort: a single unreadable document shouldn't sink the report
                pass
    return filings
