"""Lazy Prices gate-1 clause: spot-check resolved CIK -> ticker pairs against the filing's own cover page.

WHY: sec_issuer_pairs.py + sec_cik_symbol_map.py join 10-K filers to tickers from
Form 3/4/5 history (coverage measured 68-72%). Coverage says nothing about
CORRECTNESS. Ladder gate 1 (DATA) needs the join checked against an external
truth: the cover page of the 10-K itself, which lists the registrant's Trading
Symbol(s) (inline-XBRL `dei:TradingSymbol`, mandatory on cover pages since
2019-2021 phase-in). This tool samples RESOLVED rows at random (seeded, no
cherry-picking), fetches each filing's head, extracts the cover tickers, and
reports match / mismatch / no-cover-ticker.

Pre-registered rule (open_questions 2026-11-10): PASS if matches >= 90% of
rows that HAVE a readable cover ticker; KILL if < 80%. Rows with no readable
cover ticker are reported separately and excluded from the denominator
(disclosed, never silently dropped).

Pure functions are offline-testable; only `fetch_head` / `main` touch network.
No trading imports (SPINOUT-READY).
"""
import argparse
import html
import json
import random
import re
import sys
import time

_TAG_RE = re.compile(r"<[^>]+>")
_IX_SYMBOL_RE = re.compile(
    r"<ix:nonNumeric[^>]*name=\"dei:TradingSymbol\"[^>]*>(.*?)</ix:nonNumeric>", re.I | re.S)
_TICKER_TOKEN_RE = re.compile(r"^[A-Z]{1,6}(?:[.\-][A-Z]{1,2})?$")

PASS_FRAC = 0.90
KILL_FRAC = 0.80


def cover_tickers(head_text: str) -> list:
    """Tickers named by dei:TradingSymbol on the cover page, in order, de-duplicated.
    Empty list when the filing carries no inline-XBRL cover (pre-2019 or plain HTML)."""
    out = []
    for raw in _IX_SYMBOL_RE.findall(head_text or ""):
        t = html.unescape(_TAG_RE.sub("", raw)).strip().upper()
        for tok in re.split(r"[\s,/;]+", t):
            if _TICKER_TOKEN_RE.match(tok) and tok not in out:
                out.append(tok)
    return out


def _norm(t: str) -> str:
    return t.upper().replace(".", "").replace("-", "")


def judge(map_ticker: str, covers: list) -> str:
    """'match' | 'mismatch' | 'no_cover_ticker'. Share-class punctuation is ignored
    (BRK.B vs BRKB); a map ticker equal to ANY listed cover symbol is a match."""
    if not covers:
        return "no_cover_ticker"
    return "match" if _norm(map_ticker) in {_norm(c) for c in covers} else "mismatch"


def summarize(results: list) -> dict:
    """results: [{'verdict': ...}, ...] -> counts, match fraction of READABLE rows, gate verdict."""
    n = len(results)
    c = {"match": 0, "mismatch": 0, "no_cover_ticker": 0}
    for r in results:
        c[r["verdict"]] += 1
    readable = c["match"] + c["mismatch"]
    frac = (c["match"] / readable) if readable else None
    if frac is None:
        verdict = "INCONCLUSIVE_NO_READABLE_ROWS"
    elif frac >= PASS_FRAC:
        verdict = "PASSES_GATE_1_JOIN_CLAUSE"
    elif frac < KILL_FRAC:
        verdict = "KILLED_AT_GATE_1"
    else:
        verdict = "BETWEEN_BARS_NOT_PASSED"
    return {"n_sampled": n, **c, "readable": readable, "match_frac_of_readable": frac, "verdict": verdict}


def sample_resolved(rows: list, m: dict, resolve, k: int, seed: int) -> list:
    """Seeded uniform sample of rows that resolve to a unique ticker."""
    ok = []
    for r in rows:
        tkr, st = resolve(m, r["cik"], r["filingDate"])
        if tkr and st in ("exact", "padded"):
            ok.append({**r, "ticker": tkr})
    rng = random.Random(seed)
    rng.shuffle(ok)
    return ok[:k]


def fetch_head(cik: str, accession: str, ua: str, nbytes: int = 600000) -> str:
    import requests
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}.txt"
    r = requests.get(url, headers={"User-Agent": ua, "Range": f"bytes=0-{nbytes - 1}"}, timeout=60)
    r.raise_for_status()
    return r.content.decode("latin-1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--year", type=int, action="append", required=True)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import requests
    import sec_cik_symbol_map as cm
    import sec_filings_universe as fu
    import sec_form4_bulk as sb
    import sec_issuer_pairs as ip
    m = cm.build_map(ip.rows_from_pairs_cache(a.pairs_dir))
    rows = []
    for y in a.year:
        url = fu.FORM_IDX_URL.format(year=y, q=1)
        txt = requests.get(url, headers={"User-Agent": sb.USER_AGENT}, timeout=120).content.decode("latin-1")
        rows += fu.parse_form_idx(txt, forms=fu.ANNUAL_FORMS)
    pool = sample_resolved(rows, m, cm.resolve, a.k, a.seed)
    results = []
    for r in pool:
        try:
            covers = cover_tickers(fetch_head(r["cik"], r["accession"], sb.USER_AGENT))
            err = None
        except Exception as e:  # noqa: BLE001 — recorded per row
            covers, err = [], str(e)[:120]
        results.append({"cik": r["cik"], "name": r["name"], "filingDate": r["filingDate"],
                        "accession": r["accession"], "map_ticker": r["ticker"], "cover_tickers": covers,
                        "verdict": judge(r["ticker"], covers), "fetch_error": err})
        time.sleep(0.25)
    rep = {"rule": f"pass>={PASS_FRAC}, kill<{KILL_FRAC} of readable", "years_q1": a.year,
           "seed": a.seed, "pool_resolved_rows_total": len(rows), "summary": summarize(results), "rows": results}
    s = json.dumps(rep, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s + "\n")
    print(json.dumps(rep["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
