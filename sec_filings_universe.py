"""Lazy Prices PR 2a — point-in-time filer universe from EDGAR quarterly form.idx.

WHY: company_tickers.json lists only CURRENT filers, so a universe built from
it is survivorship-biased (every delisted/bankrupt/acquired company is gone —
exactly the ones whose text changes preceded a bad outcome). The quarterly
full-index lists EVERY 10-K/10-Q filed in a quarter, defunct filers included.

Raw material only: (form, company name, cik, filing date, accession). No
tickers (EDGAR has no point-in-time ticker map; the price-panel PR must
supply one) and no prices. Stdlib + sec_filings_text's EdgarClient only;
no trading-logic imports (SPINOUT-READY).

Raw quarterly index is cached write-once (a published quarter never changes
after its quarter ends; the in-progress quarter is never cached).
CLI:  python3 sec_filings_universe.py build --start 2010 --end 2026
"""
import argparse
import gzip
import json
import os
import re
import sys
from datetime import date
from typing import Optional

import sec_filings_text as sft

FORM_IDX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.idx"
# Annual/quarterly report family the Lazy Prices paper uses (+ pre-2008 small-business variants).
ANNUAL_FORMS = frozenset({"10-K", "10-K405", "10-KSB", "10-KT"})
QUARTERLY_FORMS = frozenset({"10-Q", "10-QSB", "10-QT"})
REPORT_FORMS = ANNUAL_FORMS | QUARTERLY_FORMS
_ACC_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# Layout-independent row grammar. The real form.idx header is NOT aligned with
# its data columns (data sits 5 columns right of the header text), so offsets
# are never read from the header: form = first token (forms contain no
# spaces), and CIK / date / file name are anchored from the right.
_ROW_RE = re.compile(r"^(?P<form>\S+)\s+(?P<name>.*?)\s+(?P<cik>\d+)\s+(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<file>edgar/\S+)\s*$")


def parse_form_idx(text: str, forms=REPORT_FORMS, include_amendments: bool = False) -> list:
    """form.idx -> rows for the requested report forms. Rows that do not match
    the grammar or lack a parseable accession in the file name are dropped."""
    lines = text.splitlines()
    hdr = next((i for i, ln in enumerate(lines) if ln.startswith("Form Type")), None)
    if hdr is None:
        raise ValueError("form.idx header not found")
    out = []
    for ln in lines[hdr + 2:]:
        if sft.base_form(ln[:17].split(" ")[0]) not in forms:
            continue  # cheap prefilter before the regex
        m = _ROW_RE.match(ln)
        if not m:
            continue
        form = m.group("form")
        if sft.is_amendment(form) and not include_amendments:
            continue
        acc = _ACC_RE.search(m.group("file"))
        if not acc:
            continue
        out.append({"form": form, "name": m.group("name"), "cik": sft.cik10(m.group("cik")),
                    "filingDate": m.group("date"), "accession": acc.group(1)})
    return out


def quarter_is_complete(year: int, q: int, today: Optional[date] = None) -> bool:
    today = today or date.today()
    end = date(year + (q == 4), (q * 3) % 12 + 1, 1)  # first day after the quarter
    # index for a quarter is final shortly after it ends; wait a week of slack
    return today >= date.fromordinal(end.toordinal() + 7)


def idx_cache_path(archive: str, year: int, q: int) -> str:
    return os.path.join(sft.stream_root(archive), "form_idx", f"{year}-Q{q}.idx.gz")


def fetch_quarter(year: int, q: int, client: sft.EdgarClient, archive: str,
                  today: Optional[date] = None) -> tuple:
    """-> (rows, from_cache). Complete quarters are cached write-once."""
    path = idx_cache_path(archive, year, q)
    if os.path.exists(path):
        return parse_form_idx(sft.read_gz(path).decode("latin-1")), True
    raw = client.get(FORM_IDX_URL.format(year=year, q=q))
    if quarter_is_complete(year, q, today):
        sft.write_once_gz(path, raw)
    return parse_form_idx(raw.decode("latin-1")), False


def summarize(rows: list, current_ciks: Optional[set] = None) -> dict:
    """Per-quarter universe facts. `current_ciks` (company_tickers.json) gives
    the survivorship quantification: the share of that quarter's filers that a
    current-filer universe would have silently dropped."""
    ciks = {r["cik"] for r in rows}
    s = {"filings": len(rows), "ciks": len(ciks),
         "annual": sum(1 for r in rows if sft.base_form(r["form"]) in ANNUAL_FORMS),
         "quarterly": sum(1 for r in rows if sft.base_form(r["form"]) in QUARTERLY_FORMS)}
    if current_ciks is not None:
        gone = ciks - current_ciks
        s["not_current_ciks"] = len(gone)
        s["not_current_share"] = round(len(gone) / len(ciks), 4) if ciks else None
    return s


def current_filer_ciks(client: sft.EdgarClient) -> set:
    data = client.get_json(sft.COMPANY_TICKERS_URL)
    return {sft.cik10(v["cik_str"]) for v in data.values()}


def build(start: int, end: int, archive: str, client: sft.EdgarClient,
          today: Optional[date] = None) -> dict:
    today = today or date.today()
    cur = current_filer_ciks(client)
    quarters, universe = {}, {}
    for y in range(start, end + 1):
        for q in (1, 2, 3, 4):
            if date(y, 3 * q - 2, 1) > today:
                continue
            rows, _ = fetch_quarter(y, q, client, archive, today)
            quarters[f"{y}-Q{q}"] = summarize(rows, cur)
            for r in rows:
                u = universe.setdefault(r["cik"], {"name": r["name"], "first": r["filingDate"], "last": r["filingDate"], "n": 0})
                u["first"], u["last"], u["n"] = min(u["first"], r["filingDate"]), max(u["last"], r["filingDate"]), u["n"] + 1
                if r["filingDate"] >= u["last"]:
                    u["name"] = r["name"]  # latest name wins
    out = {"built": today.isoformat(), "start": start, "end": end, "quarters": quarters,
           "ciks": len(universe), "not_current": sum(1 for c in universe if c not in cur),
           "universe": universe}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--start", type=int, default=2010)
    b.add_argument("--end", type=int, default=date.today().year)
    b.add_argument("--archive", default=None)
    b.add_argument("--out", default=None, help="write universe JSON here (default: <stream>/universe.json)")
    b.add_argument("--rate", type=float, default=sft.DEFAULT_RATE_PER_SEC)
    a = ap.parse_args(argv)
    sft.process_limiter().set_rate(a.rate)
    archive = a.archive or sft.archive_base_dir()
    res = build(a.start, a.end, archive, sft.EdgarClient())
    out = a.out or os.path.join(sft.stream_root(archive), "universe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"{res['ciks']} CIKs {a.start}-{a.end}; {res['not_current']} "
          f"({res['not_current'] / max(1, res['ciks']):.1%}) absent from company_tickers.json -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
