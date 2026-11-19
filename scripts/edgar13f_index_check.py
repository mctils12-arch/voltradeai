#!/usr/bin/env python3
"""edgar13f_index_check.py — ladder gate 1 (DATA) for the 13F archive.

Question (research/open_questions.md 2026-11-17 item (k)): does the 13F
archive hold every 13F-HR / 13F-HR/A that EDGAR disseminated on a day?

Truth: EDGAR's daily form index, Archives/edgar/daily-index/<y>/QTR<q>/
form.<yyyymmdd>.idx — permanent, complete, one line per (form, filer).
Reading: the archive as a customer gets it, /api/data/filings13f/history
paged over the same window.

Per filed day it prints: index count, held count, held share, and by form
type. Read-only; no signal claim. Usage:

  python3 scripts/edgar13f_index_check.py --from 2026-11-12 --to 2026-11-18
  python3 scripts/edgar13f_index_check.py --days 7 --json out.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = "https://voltradeai-production.up.railway.app"
# SEC fair access wants "name contact"; pass --ua (or EDGAR_USER_AGENT) to send your own.
DEFAULT_UA = os.environ.get("EDGAR_USER_AGENT") or "research-client admin@example.com"
FORMS = ("13F-HR", "13F-HR/A")
ARCHIVE_START = "2026-07-05"

# "13F-HR      Some Manager LLC      1234567     20261116    edgar/data/1234567/0001234567-26-000004.txt"
_INDEX_LINE = re.compile(
    r"^(\S+)\s{2,}.*\s(\d{1,10})\s+(\d{8})\s+edgar/data/\d+/(\d{10}-\d{2}-\d{6})\.txt\s*$"
)


def daily_form_index_url(day: str) -> str:
    y, m, d = day.split("-")
    quarter = (int(m) - 1) // 3 + 1
    return f"https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{quarter}/form.{y}{m}{d}.idx"


def parse_form_index(text: str, forms=FORMS) -> dict[str, dict]:
    """accession -> {form, cik, filed} for the exact form types asked for.
    The form type is matched whole: "13F-NT" and "13F-HR/A" are not "13F-HR"."""
    out: dict[str, dict] = {}
    for line in text.split("\n"):
        m = _INDEX_LINE.match(line)
        if not m or m.group(1) not in forms:
            continue
        acc = m.group(4)
        if acc in out:
            continue
        ymd = m.group(3)
        out[acc] = {"form": m.group(1), "cik": m.group(2), "filed": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"}
    return out


def compare_day(day: str, index: dict[str, dict] | None, held: set[str]) -> dict:
    """One filed day: what the index lists against what the archive holds.
    index None = EDGAR published no index for the day (weekend / holiday)."""
    if index is None:
        return {"day": day, "index": None, "held": 0, "missing": 0, "share": None, "by_form": {}}
    by_form: dict[str, dict] = {}
    for acc, rec in index.items():
        b = by_form.setdefault(rec["form"], {"index": 0, "held": 0})
        b["index"] += 1
        if acc in held:
            b["held"] += 1
    n = len(index)
    h = sum(b["held"] for b in by_form.values())
    return {
        "day": day,
        "index": n,
        "held": h,
        "missing": n - h,
        "share": round(h / n, 4) if n else None,
        "by_form": by_form,
    }


def summarize(rows: list[dict], held: set[str], indexed: set[str]) -> dict:
    days = [r for r in rows if r["index"]]
    n = sum(r["index"] for r in days)
    h = sum(r["held"] for r in days)
    return {
        "index_days": len(days),
        "index": n,
        "held": h,
        "missing": n - h,
        "share": round(h / n, 4) if n else None,
        "complete_days": sum(1 for r in days if r["missing"] == 0),
        # held rows the index of the window does not list: filed before the
        # window and archived inside it, or a form the index names otherwise
        "held_not_in_window_index": len(held - indexed),
    }


def days_between(frm: str, to: str) -> list[str]:
    a, b = dt.date.fromisoformat(frm), dt.date.fromisoformat(to)
    return [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def next_page_url(route: str, page: dict) -> str | None:
    """The route's `next` is a query string ("?from=…&to=…"); None ends the walk."""
    nxt = page.get("next")
    if not page.get("truncated") or not nxt:
        return None
    q = nxt if "limit=" in nxt else f"{nxt}&limit=500"
    return route + q if q.startswith("?") else q


# ── network ────────────────────────────────────────────────────────────────

def _get(url: str, ua: str, timeout: float = 60.0) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""


def fetch_index(day: str, ua: str) -> dict[str, dict] | None:
    """None when EDGAR serves no index for the day. EDGAR answers 403 or 404
    for a day it has no index for, and 403 when it refuses a client, so a
    weekday None is printed as "closed or refused" (no_index_note), never as
    a day with no filings."""
    code, text = _get(daily_form_index_url(day), ua)
    if code in (403, 404):
        return None
    if code != 200:
        raise RuntimeError(f"index {day}: HTTP {code}")
    return parse_form_index(text)


def no_index_note(day: str) -> str:
    if dt.date.fromisoformat(day).weekday() >= 5:
        return "no index (weekend)"
    return "no index on a weekday: EDGAR closed that day, or refused this client"


def fetch_held(base: str, frm: str, to: str, ua: str) -> tuple[set[str], dict]:
    """Every accession the archive serves for archive days [frm, to], paged."""
    held: set[str] = set()
    pages = 0
    types: dict[str, int] = {}
    route = f"{base}/api/data/filings13f/history"
    url = f"{route}?from={frm}&to={to}&limit=500"
    while url:
        code, text = _get(url, ua, timeout=120.0)
        if code != 200:
            raise RuntimeError(f"history: HTTP {code} for {url}")
        page = json.loads(text)
        pages += 1
        for f in page.get("filings", []):
            acc = f.get("accession")
            if acc and acc not in held:
                held.add(acc)
                t = str(f.get("submissionType"))
                types[t] = types.get(t, 0) + 1
        url = next_page_url(route, page)
    return held, {"pages": pages, "submission_types": types}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="frm")
    ap.add_argument("--to")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--base", default=DEFAULT_BASE_URL)
    ap.add_argument("--ua", default=DEFAULT_UA)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    to = a.to or dt.datetime.now(dt.timezone.utc).date().isoformat()
    frm = a.frm or (dt.date.fromisoformat(to) - dt.timedelta(days=a.days - 1)).isoformat()

    # A filing is archived on or after the day it was filed, so the archive
    # is read from the window's first day to today.
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    held, meta = fetch_held(a.base, max(frm, ARCHIVE_START), today, a.ua)

    rows, indexed = [], set()
    for day in days_between(frm, to):
        idx = fetch_index(day, a.ua)
        if idx:
            indexed |= set(idx)
        rows.append(compare_day(day, idx, held))
        time.sleep(0.3)  # SEC fair access
    total = summarize(rows, held, indexed)

    print(f"13F archive vs EDGAR daily form index, filed days {frm}..{to} (archive read {meta['pages']} page(s), {len(held)} accessions)")
    print(f"{'filed day':<12}{'index':>7}{'held':>7}{'missing':>9}{'share':>8}   by form (held/index)")
    for r in rows:
        if r["index"] is None:
            print(f"{r['day']:<12}{'-':>7}{'-':>7}{'-':>9}{'-':>8}   {no_index_note(r['day'])}")
            continue
        forms = ", ".join(f"{k} {v['held']}/{v['index']}" for k, v in sorted(r["by_form"].items()))
        share = f"{100 * r['share']:.1f}%" if r["share"] is not None else "-"
        print(f"{r['day']:<12}{r['index']:>7}{r['held']:>7}{r['missing']:>9}{share:>8}   {forms}")
    share = f"{100 * total['share']:.1f}%" if total["share"] is not None else "-"
    print(f"{'TOTAL':<12}{total['index']:>7}{total['held']:>7}{total['missing']:>9}{share:>8}   "
          f"{total['complete_days']} of {total['index_days']} index days complete; "
          f"{total['held_not_in_window_index']} held accessions not in the window's index")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"from": frm, "to": to, "read_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       "archive": meta, "rows": rows, "total": total}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
