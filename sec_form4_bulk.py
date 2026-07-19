"""
sec_form4_bulk.py — free bulk-historical SEC Form 3/4/5 insider-transaction
dataset (EDGE DOCTRINE #1: BUILD DATA, DON'T BUY IT / BUILD-FIRST rule #2).

open_questions.md's "Insider Form 4 clustering as a signal" gate-2 hypothesis
was blocked on "accumulating enough live filing history" from the real-time
edgarForm4.ts feed (which only started polling 2026-07-03) OR "sourcing free
historical Form 4 index files from SEC's bulk data page... worth trying
before waiting on live accumulation, unexplored." This module is that
unexplored path: SEC publishes a PRE-PARSED, structured TSV dataset per
calendar quarter (SUBMISSION/REPORTINGOWNER/NONDERIV_TRANS/...), keyless,
going back to 2006 — no XML parsing, no per-filing HTTP request, no waiting.
Confirmed live 2026-07-19: https://www.sec.gov/data-research/sec-markets-data/
insider-transactions-data-sets lists one zip per CLOSED quarter (the current,
still-open quarter is simply absent from the listing — SEC's own
closed-quarter boundary, mirrors EPA CAMD's quarter-closed gate, no request
needed to discover it). Base URL path has changed over time
(.../structureddata/... for quarters through 2026q1, .../
datastandardsinnovation/... for 2026q2+) — this module SCRAPES the live
listing for exact URLs rather than guessing a static pattern, the same
lesson the COT gate-2 session logged for a different provider.

DATA-LADDER GATE 1 (DATA) ONLY: this module fetches, joins, and archives.
Gate 2 (does an officer/director open-market-buy cluster predict forward
excess return?) is form4_gate2_test.py, a separate, explicit statistical
screen — nothing here is wired into deep_score/tier decisions.

Scope, deliberately: NONDERIV_TRANS rows with TRANS_CODE in {P, S} (open-
market purchase / sale — code A grants and code M option exercises are not
discretionary and would dilute the signal, per the open_questions.md prior)
whose SUBMISSION.DOCUMENT_TYPE is a Form 4 (or 4/A), joined to REPORTINGOWNER
and restricted to filers whose RPTOWNER_RELATIONSHIP contains "Officer" or
"Director" (the open_questions.md prior explicitly excludes pure 10%-owner
funds from the primary hypothesis — a fund's "purchase" is a portfolio
decision, not the same information event as an insider's own money).
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import time
import zipfile
from datetime import datetime

import requests

from storage_config import FORM4_BULK_CHECKPOINT_PATH, FORM4_BULK_DIR

USER_AGENT = "VolTradeAI research@voltradeai.com"
INDEX_PAGE_URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"

TRACKED_CODES = {"P", "S"}
VALID_DOCUMENT_TYPES = {"4", "4/A"}

# How many of the most-recent CLOSED quarters to backfill (~2 years) —
# mirrors SEC MIDAS / EPA CAMD's chronological-backfill-window precedent.
BACKFILL_QUARTERS = 8

# Only re-scrape the index listing (and fetch at most one quarter) once per
# this many seconds even if called more often — Tier 3 runs hourly, this
# dataset updates quarterly, so this makes repeated invocation nearly free.
MIN_CHECK_INTERVAL_SEC = 12 * 3600

_QUARTER_RE = re.compile(r'href="(/files/[^"]*?(\d{4})q([1-4])_form345\.zip)"')
_DATE_RE = re.compile(r"^(\d{1,2})-([A-Z]{3})-(\d{4})$")
_MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


def _to_iso_date(raw: str | None) -> str | None:
    """SEC's structured TSVs use '31-MAR-2025' — convert to ISO 8601, or
    None if unparseable (never fabricate a date)."""
    if not raw:
        return None
    m = _DATE_RE.match(raw.strip())
    if not m:
        return None
    day, mon, year = m.groups()
    mon_num = _MONTHS.get(mon)
    if not mon_num:
        return None
    return f"{year}-{mon_num}-{int(day):02d}"


# SEC's raw ISSUERTRADINGSYMBOL field is free text, not a validated ticker —
# live-probed 2026-07-19 across 2025q1's SUBMISSION.tsv turned up "N/A",
# "NONE", "-", exchange-prefixed "NASDAQ:SVC", and dual-class/merged values
# like "GEF, GEF-B" or "BFA, BFB". A dual-class value is deliberately
# REJECTED rather than guessed at (never fabricate which class symbol was
# meant) — this trades a small amount of recall for never mislabeling a
# record with a symbol the filer didn't actually report.
_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,2})?$")
_TICKER_PLACEHOLDER_WORDS = {"NONE", "NA", "TBD", "UNKNOWN", "PENDING"}


def _is_valid_ticker(ticker: str) -> bool:
    return bool(_VALID_TICKER_RE.match(ticker)) and ticker not in _TICKER_PLACEHOLDER_WORDS


def list_available_quarters() -> dict[str, str]:
    """Scrape the live SEC index page for {"2026q1": full_zip_url, ...}.
    Only CLOSED quarters ever appear in this listing (verified live
    2026-07-19: the in-progress quarter 404s and is simply absent here) —
    so "listed" IS the closed-quarter gate, no separate check needed."""
    resp = requests.get(INDEX_PAGE_URL, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    quarters: dict[str, str] = {}
    for path, year, q in _QUARTER_RE.findall(resp.text):
        key = f"{year}q{q}"
        quarters[key] = "https://www.sec.gov" + path
    return quarters


def _parse_tsv(content: bytes) -> list[dict]:
    text = content.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def fetch_quarter(url: str) -> list[dict]:
    """Download one quarter's zip, join SUBMISSION + REPORTINGOWNER onto
    NONDERIV_TRANS, filter to Form 4 officer/director open-market P/S
    transactions. Returns clean records (one per owner per transaction —
    a joint filing with 2 reporting owners on 1 transaction row emits 2
    records, each correctly attributed). Raises on network failure; the
    caller decides whether that aborts the whole update."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=180)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    submissions = {r["ACCESSION_NUMBER"]: r for r in _parse_tsv(zf.read("SUBMISSION.tsv"))}
    owners_by_accession: dict[str, list[dict]] = {}
    for r in _parse_tsv(zf.read("REPORTINGOWNER.tsv")):
        owners_by_accession.setdefault(r["ACCESSION_NUMBER"], []).append(r)

    out = []
    for r in _parse_tsv(zf.read("NONDERIV_TRANS.tsv")):
        code = (r.get("TRANS_CODE") or "").strip()
        if code not in TRACKED_CODES:
            continue
        acc = r.get("ACCESSION_NUMBER")
        sub = submissions.get(acc)
        if not sub or sub.get("DOCUMENT_TYPE") not in VALID_DOCUMENT_TYPES:
            continue
        ticker = (sub.get("ISSUERTRADINGSYMBOL") or "").strip().upper()
        if not ticker or not _is_valid_ticker(ticker):
            continue
        filing_date = _to_iso_date(sub.get("FILING_DATE"))
        if not filing_date:
            continue
        try:
            shares = float(r.get("TRANS_SHARES") or 0)
            price = float(r.get("TRANS_PRICEPERSHARE") or 0)
        except ValueError:
            continue
        if shares <= 0 or price <= 0:
            continue

        for o in owners_by_accession.get(acc, []):
            rel = o.get("RPTOWNER_RELATIONSHIP") or ""
            is_officer = "Officer" in rel
            is_director = "Director" in rel
            is_ten_pct = "TenPercentOwner" in rel
            if not (is_officer or is_director):
                continue  # gate-2 hypothesis is officer/director specifically
            out.append({
                "accession": acc,
                "ticker": ticker,
                "issuer_cik": sub.get("ISSUERCIK"),
                "issuer_name": sub.get("ISSUERNAME"),
                "filing_date": filing_date,
                "trans_date": _to_iso_date(r.get("TRANS_DATE")),
                "trans_code": code,
                "shares": shares,
                "price": price,
                "dollar_value": round(shares * price, 2),
                "owner_cik": o.get("RPTOWNERCIK"),
                "owner_name": o.get("RPTOWNERNAME"),
                "is_officer": is_officer,
                "is_director": is_director,
                "is_ten_pct_owner": is_ten_pct,
            })
    return out


def _archive_path(quarter: str) -> str:
    return os.path.join(FORM4_BULK_DIR, f"{quarter}.json")


def is_archived(quarter: str) -> bool:
    return os.path.exists(_archive_path(quarter))


def archived_quarters() -> list[str]:
    if not os.path.isdir(FORM4_BULK_DIR):
        return []
    return sorted(
        f[:-5] for f in os.listdir(FORM4_BULK_DIR)
        if f.endswith(".json") and not f.endswith(".tmp.json")
    )


def archive_quarter(quarter: str, records: list[dict]) -> None:
    path = _archive_path(quarter)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"quarter": quarter, "record_count": len(records), "records": records}, f)
    os.replace(tmp, path)


def load_quarter(quarter: str) -> list[dict]:
    try:
        with open(_archive_path(quarter)) as f:
            return json.load(f).get("records", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def load_all_records(quarters: list[str] | None = None) -> list[dict]:
    """Concatenate every archived quarter (or a specific subset) into one
    list, sorted by filing_date. Internal-API getter for future gate-2/
    gate-3 consumers, mirrors cftc_cot.get_cot_snapshot's role."""
    target = quarters if quarters is not None else archived_quarters()
    out: list[dict] = []
    for q in target:
        out.extend(load_quarter(q))
    out.sort(key=lambda r: r["filing_date"])
    return out


def pick_quarters_to_fetch(available: dict[str, str]) -> list[str]:
    """Chronological backfill, one quarter per poll (mirrors EPA CAMD /
    SEC MIDAS precedent): fill in the most recent BACKFILL_QUARTERS closed
    quarters oldest-missing-first; once fully backfilled, steady state is
    just checking whether a newer quarter has closed since the last poll."""
    all_keys_sorted = sorted(available.keys())
    if not all_keys_sorted:
        return []
    target_window = all_keys_sorted[-BACKFILL_QUARTERS:]
    missing_in_window = [k for k in target_window if not is_archived(k)]
    if missing_in_window:
        return [missing_in_window[0]]
    newest = all_keys_sorted[-1]
    if not is_archived(newest):
        return [newest]
    return []


def _load_last_check() -> float:
    try:
        with open(FORM4_BULK_CHECKPOINT_PATH) as f:
            return json.load(f).get("last_check_ts", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def _save_last_check(ts: float) -> None:
    with open(FORM4_BULK_CHECKPOINT_PATH, "w") as f:
        json.dump({"last_check_ts": ts}, f)


def run_update(force: bool = False) -> dict:
    """Entry point for the bot's hourly Tier 3 maintenance call (mirrors
    cftc_cot.run_daily_update's wiring). Hits the network at most once per
    MIN_CHECK_INTERVAL_SEC regardless of call frequency, and even then
    downloads at most ONE quarter per invocation (a quarter's zip is
    several MB — a nightly-poll cadence, not a burst)."""
    now = time.time()
    last_check = _load_last_check()
    if not force and (now - last_check) < MIN_CHECK_INTERVAL_SEC:
        return {"status": "skipped", "reason": "checked recently",
                "age_hours": round((now - last_check) / 3600, 1)}

    try:
        available = list_available_quarters()
    except requests.RequestException as e:
        _save_last_check(now)
        return {"status": "error", "reason": f"index fetch failed: {e}"[:200]}

    targets = pick_quarters_to_fetch(available)
    results = {}
    for q in targets:
        try:
            records = fetch_quarter(available[q])
            archive_quarter(q, records)
            results[q] = {"status": "ok", "records": len(records)}
        except requests.RequestException as e:
            results[q] = {"status": "error", "reason": str(e)[:200]}
        except (KeyError, zipfile.BadZipFile) as e:
            results[q] = {"status": "error", "reason": f"parse: {e}"[:200]}

    _save_last_check(now)
    return {
        "status": "done" if targets else "up_to_date",
        "quarters_checked": targets,
        "results": results,
        "quarters_archived_total": len(archived_quarters()),
    }


if __name__ == "__main__":
    print(json.dumps(run_update(force=True), indent=2, default=str))
