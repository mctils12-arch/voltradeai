"""Lazy Prices PR 3c — ALL-FORM issuer (ticker, cik, date) pairs from the SEC
insider bulk data sets, to widen the point-in-time CIK -> ticker map.

WHY: sec_cik_symbol_map.py (PR 3b) was fed from Edge P2's cache, which keeps only
officer/director open-market P/S transaction rows, so a name whose insiders never
bought/sold on the open market is unmapped. SUBMISSION.tsv in the SAME quarterly
zip lists EVERY Form 3/4/5 filing with the issuer's CIK and the symbol it traded
under on that date — no transaction filter needed to learn an identity. This is
a SEPARATE cache (own dir): Edge P2's cache and its inputs are untouched.

Honest limits: identity only (never a return input); a name with no insider
filing of any kind in the cached span stays unmapped (dropped and counted by the
caller); ISSUERTRADINGSYMBOL is free text, so the same validity filter as
sec_form4_bulk applies and dual-class values are rejected, never guessed.
No trading imports (SPINOUT-READY). Network only in `ensure_pairs_cache`.
"""
import csv
import gzip
import io
import json
import os
import re
import sys
import time
import zipfile
from typing import Iterable, List, Optional, Tuple

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,2})?$")
_PLACEHOLDERS = {"NONE", "NA", "TBD", "UNKNOWN", "PENDING"}
_MON = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
_DATE_RE = re.compile(r"^(\d{1,2})-([A-Za-z]{3})-(\d{4})$")


def _iso(raw: Optional[str]) -> Optional[str]:
    m = _DATE_RE.match((raw or "").strip())
    if not m or m.group(2).upper() not in _MON:
        return None
    return f"{m.group(3)}-{_MON[m.group(2).upper()]:02d}-{int(m.group(1)):02d}"


def valid_ticker(t: str) -> bool:
    return bool(_VALID_TICKER_RE.match(t)) and t not in _PLACEHOLDERS


def pairs_from_submission_tsv(content: bytes, skipped: Optional[dict] = None) -> List[Tuple[str, str, str]]:
    """-> distinct (ticker, issuer_cik, 'YYYY-MM-DD') from one SUBMISSION.tsv.
    Every row dropped is counted into `skipped` by reason, never silently lost."""
    sk = skipped if skipped is not None else {}
    out = set()
    text = content.decode("utf-8", errors="replace")
    for r in csv.DictReader(io.StringIO(text), delimiter="\t"):
        t = (r.get("ISSUERTRADINGSYMBOL") or "").strip().upper()
        cik = (r.get("ISSUERCIK") or "").strip()
        if not cik:
            sk["no_cik"] = sk.get("no_cik", 0) + 1
            continue
        if not t or not valid_ticker(t):
            sk["bad_ticker"] = sk.get("bad_ticker", 0) + 1
            continue
        d = _iso(r.get("FILING_DATE"))
        if not d:
            sk["bad_date"] = sk.get("bad_date", 0) + 1
            continue
        out.add((t, cik, d))
    return sorted(out)


def pairs_from_zip_bytes(blob: bytes, skipped: Optional[dict] = None) -> List[Tuple[str, str, str]]:
    zf = zipfile.ZipFile(io.BytesIO(blob))
    return pairs_from_submission_tsv(zf.read("SUBMISSION.tsv"), skipped)


def _path(d: str, q: str) -> str:
    return os.path.join(d, f"{q}.pairs.jsonl.gz")


def write_pairs(d: str, q: str, pairs: Iterable[Tuple[str, str, str]]) -> None:
    os.makedirs(d, exist_ok=True)
    tmp = _path(d, q) + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(list(p), separators=(",", ":")) + "\n")
    os.replace(tmp, _path(d, q))


def rows_from_pairs_cache(d: str, skipped: Optional[dict] = None) -> Iterable[Tuple[str, str, str]]:
    """Yield (ticker, cik, date) for build_map; unparseable lines are COUNTED."""
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".pairs.jsonl.gz"):
            continue
        with gzip.open(os.path.join(d, fn), "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    yield r[0], r[1], r[2]
                except (ValueError, IndexError, TypeError):
                    if skipped is not None:
                        skipped["bad_line"] = skipped.get("bad_line", 0) + 1


def ensure_pairs_cache(d: str, first: str = "2014q1", sleep_s: float = 0.3) -> Tuple[List[str], List[str]]:
    """Resumable download of each closed quarter's pairs. Network trouble on one
    quarter never aborts the rest. -> (cached quarters, failed quarters)."""
    import sec_form4_bulk as sb
    avail = sb.list_available_quarters()
    wanted = [q for q in sorted(avail) if q >= first]
    failed = []
    for q in wanted:
        if os.path.exists(_path(d, q)):
            continue
        try:
            import requests
            resp = requests.get(avail[q], headers={"User-Agent": sb.USER_AGENT}, timeout=180)
            resp.raise_for_status()
            write_pairs(d, q, pairs_from_zip_bytes(resp.content))
        except Exception as e:  # noqa: BLE001 — surfaced via the failed list
            print(f"[sec_issuer_pairs] {q} failed: {e}", file=sys.stderr)
            failed.append(q)
            continue
        time.sleep(sleep_s)
    return [q for q in wanted if os.path.exists(_path(d, q))], failed
