"""Lazy Prices PR 3b — point-in-time CIK -> ticker map from SEC Form 3/4/5 filings.

WHY: the gate-2 harness (sec_filings_gate2.py) keys prices by CIK, but the
2016+ price panel is keyed by ticker/asset. EDGAR has no point-in-time ticker
map (sec_filings_universe.py says so) and company_tickers.json lists only
CURRENT filers, i.e. a survivor-only join — exactly the bias the delisted-
inclusive panel was built to remove. Every Form 4 does carry the issuer's CIK
AND the trading symbol it traded under AT THAT FILING DATE, so the quarterly
bulk data Edge P2 already caches is a free, dated, delisting-inclusive map.

Honest limits (the product of this module is a COVERAGE NUMBER first):
  * An issuer only appears when an insider transaction was filed; quiet names
    and names before the cache start (2014q1) are unmapped. Unmapped => the
    10-K row is DROPPED AND COUNTED, never guessed.
  * A span is first..last filing date seen for (cik, ticker), padded by
    SLACK_DAYS because insiders trade irregularly. Identity, not information:
    using a filing dated after the 10-K only to learn which ticker the CIK
    wore is not return lookahead, but it is disclosed.
  * Two tickers valid on the same day (share classes, a rename inside the
    slack) => status "ambiguous", ticker None. Conservative by design.

Pure functions, stdlib only, no network, no trading imports (SPINOUT-READY).
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Iterable, Optional

SLACK_DAYS = 400   # > one year: the median insider files at least annually; rarer ones are dropped, not guessed


def norm_cik(cik) -> str:
    s = str(cik).strip().lstrip("0")
    return s or "0"


def build_map(rows: Iterable) -> dict:
    """rows: (ticker, issuer_cik, filing_date 'YYYY-MM-DD') triples.
    -> {cik: {ticker: [first, last, n]}}. Blank/odd tickers and undated rows are skipped."""
    m: dict = defaultdict(dict)
    for tkr, cik, d in rows:
        tkr = (tkr or "").strip().upper()
        if not tkr or not cik or not d or len(d) < 10:
            continue
        d = d[:10]
        e = m[norm_cik(cik)].setdefault(tkr, [d, d, 0])
        if d < e[0]:
            e[0] = d
        if d > e[1]:
            e[1] = d
        e[2] += 1
    return dict(m)


def _pad(d: str, days: int) -> str:
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


def resolve(m: dict, cik, day: str, slack_days: int = SLACK_DAYS) -> tuple:
    """-> (ticker | None, status). status: exact (inside a seen span) | padded
    (inside span +/- slack) | ambiguous | none. A tighter match always wins over a
    padded one; two equally tight matches are ambiguous, never picked by size."""
    spans = m.get(norm_cik(cik))
    if not spans:
        return None, "none"
    exact = [t for t, (a, b, _) in spans.items() if a <= day <= b]
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        return None, "ambiguous"
    padded = [t for t, (a, b, _) in spans.items() if _pad(a, -slack_days) <= day <= _pad(b, slack_days)]
    if len(padded) == 1:
        return padded[0], "padded"
    if len(padded) > 1:
        return None, "ambiguous"
    return None, "none"


def coverage(filing_rows: list, m: dict, panel_symbols: Optional[set] = None, slack_days: int = SLACK_DAYS) -> dict:
    """filing_rows: dicts with cik and a date (`accepted_at` or `filed`). Reports how
    many resolve, by status and by year, and (if given) how many resolved tickers
    exist in the price panel. This is the gate-1-style number for the join."""
    st, by_year, in_panel = Counter(), defaultdict(lambda: [0, 0]), 0
    for r in filing_rows:
        day = (r.get("accepted_at") or r.get("filingDate") or r.get("filed") or "")[:10]
        if not day:
            st["undated"] += 1
            continue
        tkr, status = resolve(m, r["cik"], day, slack_days)
        st[status] += 1
        y = by_year[day[:4]]
        y[1] += 1
        ok = tkr is not None and (panel_symbols is None or tkr in panel_symbols)
        if tkr is not None and panel_symbols is not None and not ok:
            st["resolved_not_in_panel"] += 1
        if ok:
            y[0] += 1
            in_panel += 1
    n = len(filing_rows)
    return {"n_rows": n, "status": dict(st), "joined": in_panel,
            "joined_frac": (in_panel / n) if n else None,
            "by_year": {k: {"joined": v[0], "rows": v[1], "frac": v[0] / v[1]} for k, v in sorted(by_year.items())}}


def rows_from_form4_cache(form4_dir: str, skipped: Optional[Counter] = None) -> Iterable:
    """Edge P2's compact cache: [ticker, issuer_cik, owner_cik, code, filing_date, trans_date, value].
    Unparseable lines are COUNTED into `skipped["bad_line"]` (never silently lost)."""
    for fn in sorted(os.listdir(form4_dir)):
        if not fn.endswith(".jsonl.gz"):
            continue
        with gzip.open(os.path.join(form4_dir, fn), "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    yield r[0], r[1], r[4]
                except (ValueError, IndexError, TypeError):
                    if skipped is not None:
                        skipped["bad_line"] += 1
                    continue


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--form4-dir", required=True)
    ap.add_argument("--rows", required=True, help="gate-2 rows JSON (cik + accepted_at)")
    ap.add_argument("--panel-symbols", help="file with one panel ticker per line (optional)")
    a = ap.parse_args(argv)
    skipped: Counter = Counter()
    m = build_map(rows_from_form4_cache(a.form4_dir, skipped))
    syms = None
    if a.panel_symbols:
        with open(a.panel_symbols) as f:
            syms = {ln.strip().upper() for ln in f if ln.strip()}
    with open(a.rows) as f:
        rows = json.load(f)
    rep = coverage(rows, m, syms)
    rep["ciks_mapped"] = len(m)
    rep["cache_lines_skipped"] = skipped["bad_line"]
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
