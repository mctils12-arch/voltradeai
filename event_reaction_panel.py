#!/usr/bin/env python3
"""
event_reaction_panel.py — the PRE-REGISTERED event-reaction study
(research/event_reaction_study.md §1; code: event_reaction.py) re-run on the
2016+ delisting-inclusive daily price panel (price_panel.py), server-side.

WHY. The 2026-10-29 run priced CURRENT tickers on Yahoo: a survivor panel.
Its own pre-registration names the delisting-inclusive panel as the fix (§1.2)
and forbids any positive-drift pass until it runs (§1.5 rule 4). That panel
lives only on the Railway volume (server/pricePanel.ts; Alpaca data under
derived-statistics-only terms), so the re-run runs there: Saturday window,
once per ISO week at most, own long-job lane (server/eventStudy.ts).

UNCHANGED — imported from event_reaction.py, never restated: the four event
definitions and their SEC sources, the acceptance-time day-0 rule, the 20-day
dedup, the $1 / 60-day / full-window inclusion rules, the windows, the
same-ticker random-date controls (seed, span, exclusion), winsorizing,
week-clustered SEs, liquidity terciles, regime and halves splits, the
Abdi-Ranaldo cost proxy, the 12-cell family and its Bonferroni bars (2.87;
3.02 for E1), the E4 30,000-filing seeded subsample rule, and the verdict
functions. DIFFERENT — declared below (DEVIATIONS) before any panel run.

OUTPUT (storage_config.EVENT_STUDY_DIR = DATA_DIR/event_study, the volume):
latest.json (last success: aggregate tables only — no price, no ticker, no
per-event row; Alpaca licence) and last_run.json (every run), KB-sized. The SEC
cache (compact derived rows of public EDGAR data, re-fetchable) lives in the
CONTAINER tmp dir, never the volume (short of space): <cache>/<build>-<ver>/,
~15-25 MB (measured: ~40-100 KB of insider rows per quarter), older builds' dirs deleted on start. Raw downloads (insider zips,
filing documents) are parsed in memory and never written.

SEC FAIR ACCESS: the repo's declared User-Agent and process-wide limiter
(sec_filings_text.EdgarClient), here at --rate 5 req/s (under SEC's 10, leaving
room for the live EDGAR feeds). One full fetch is ~40-45k requests (~2.5 h: 52
insider zips at ~3 s each, ~13-15k submissions pages, the 30,000 5.02 documents).

CLI: python3 event_reaction_panel.py --run [--panel-root DIR] [--out-dir DIR]
       [--cache-dir DIR] [--max-rss-mb N] [--rate R] [--workers N] [--nice N]
       [--code-version V]   -> one JSON summary line on stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import resource
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import event_reaction as er
import price_panel as pp
import sec_cik_symbol_map as csm
import sec_issuer_pairs as sip
from walkforward import DELIST_RETURN_MERGER, DELIST_RETURN_OTHER
from walkforward import delisting_return as panel_delisting_return

SCHEMA = "event_study_panel/v1"
STUDY_FILES = ("event_reaction.py", "event_reaction_panel.py")   # server/eventStudy.ts EVENT_STUDY_FILES
CACHE_VERSION = "c1"            # bump when a cached derived row's definition changes
DEFAULT_MAX_RSS_MB = 1200.0
DEFAULT_RATE = 5.0              # req/s; sec_filings_text clamps to SEC's hard 10
DEFAULT_WORKERS = 4
PAIRS_FROM = (2014, 1)          # dated-ticker map span (sec_issuer_pairs' measured configuration)
PURCHASES_FROM = (2015, 4)      # = event_reaction._quarters() default start (pinned by test)
CURRENT_EXCHANGES = ("NYSE", "Nasdaq", "CBOE", "TXSE")   # any current listing => not a delisted name
FETCH_CHUNK = 200               # bounded submission to the thread pool: an abort never drains a queue
MAX_FETCH_ERROR_SHARE = 0.02    # more failed SEC fetches than this => "insufficient", never a thinned "ok"
RSS_CHECK_EVERY = 500
_BUILD_CACHE_RE = re.compile(r"^b\d{8}-c\d+$")

PANEL_PROVISIONAL = "CANDIDATE SIGNAL (gate 2, PROVISIONAL — positive drift on a survivorship-reduced panel)"

PRIOR = (
    "Stated 2026-11-06, before any panel run. Survivorship flatters post-event returns (firms that died after "
    "the event are missing), so adding delisted names should move post-event excess CARs DOWN, most for small "
    "caps and for E3 restatements, whose worst cases are the delisted ones. Expected: E1 insider clusters — "
    "[0,1] stays about +2% (fast reaction), pooled [2,20] falls from +0.79% to about +0.3..+0.6%, no cell "
    "reaches |t| 3.02 (P 0.95). E2 Item 1.01 — [0,1] about 0; small-cap [2,20] about -1.2..-1.6% but the "
    "2016-2020 half still near 0, so the halves rule fails again (P 0.90 no pass). E3 Item 4.02 — [0,1] more "
    "negative, about -2..-3% (delisted restaters return); [2,20] about -0.5..-2% and still underpowered, MDE "
    "> 3% (P 0.95 no pass). E4 CEO/CFO exits — [0,1] about -1.2%; pooled [2,20] about -1.0..-1.5%; small "
    "caps the likeliest cell to cross 2.87 (P 0.10), though the 30,000-filing sample is a smaller share of a "
    "larger 5.02 population. Overall P(no CANDIDATE cell) about 0.73: the re-run most likely CONFIRMS 'no "
    "tradeable post-event drift'. WHAT WOULD CHANGE THE CONCLUSION: (a) any [2,20] cell passing all four "
    "pre-registered tests here — a negative cell becomes a gate-2 CANDIDATE (next: a gate-3 ablation with "
    "borrow cost, since the trade is a short), a positive cell a PROVISIONAL candidate (the panel is "
    "survivorship-reduced, not free); (b) the survivor subset on this panel failing to reproduce the "
    "2026-10-29 Yahoo cells within their standard errors — then the price source, not the delistings, moved "
    "the numbers, a gate-1 (DATA) fault to diagnose before any cell is read; (c) a type whose full-panel "
    "[2,20] mean moves by more than its MDE versus the survivor subset — survivorship mattered for that type, "
    "a finding for every survivor-panel result in the repo."
)

DEVIATIONS = [
    "PD1 price source: Alpaca daily bars, adjustment=all (splits + dividends), from the panel build named in "
    "`panel`, not Yahoo adjusted closes; trading calendar = the panel's SPY days.",
    "PD2 universe: the pre-registered survivor universe (current NYSE/Nasdaq/CBOE CIKs in SEC "
    "company_tickers_exchange.json, first-listed ticker, >=1 10-K/10-Q since 2015) PLUS delisted CIKs: not "
    "currently exchange-listed (absent from that file, or there only as OTC / no exchange), with a dated ticker "
    "in the SEC Form 3/4/5 data sets (sec_issuer_pairs) that matches a panel symbol, same operating filter. A "
    "delisted CIK's event uses its dated ticker on day 0 (sec_cik_symbol_map.resolve: exact, or within 400 "
    "days of a filing under it); ambiguous or unmapped events are dropped and counted. Delisted names with no "
    "insider filing since 2014 stay missing: survivorship-REDUCED, not free.",
    "PD3 delisting assumption (walkforward.delisting_return; research/price_panel_coverage.md caveat 3): a "
    "series whose last bar is more than 7 days before the panel as_of earns, on its first missing trading day, "
    "0% at a dated merger or rename and -30% otherwise (Shumway 1997); every later day earns SPY (abnormal "
    "return 0). A window counts only if its entry close is a real bar; an event or control date needs real "
    "bars on day -1 and day 0 (a filing made after the last trade is not a market reaction: counted, not "
    "measured).",
    "PD4 the $1 floor on day -1 reads the panel's RAW close (point-in-time); the Yahoo run read the "
    "back-adjusted close. Events whose inclusion would differ are counted (meta.floor_flips).",
    "PD5 the panel starts 2016-01-04 (Yahoo: 2014-06): day 0 needs 60 prior panel bars (from about "
    "2016-03-31); 2016 events have no size bucket (no 2015 breakpoints) and enter pooled rows only, so a "
    "cell's 2016-2020 half is 2017-2020; the SPY 200-day regime starts about 2016-10.",
    "PD6 rule 4 (survivorship direction) is scoped to the survivor panel in its own text ('needs the "
    "delisting-inclusive panel'): here a positive cell meeting tests 1-3 reads candidate_provisional "
    "(PROVISIONAL: the panel is survivorship-reduced, not free); negative cells are unchanged. The core "
    "function's verdict is kept beside it (verdict_core); the survivor subset keeps rule 4 as written.",
    "PD7 reused-ticker identity guards: a survivor event on a reused ticker's unmapped (raw) series counts only "
    "when the CIK's dated ticker confirms the symbol; such an event, and every delisted CIK's event, counts only "
    "on a series that overlaps the CIK's own first-to-last filing span under that symbol (so a ticker that "
    "belonged to another company before or after it is never used). Both drops are counted.",
    "PD8 E4's seeded 30,000-filing subsample (§1.2 rule, same seed) is drawn from the larger 5.02 population "
    "(delisted CIKs included): a different, proportionally smaller sample than the survivor run's.",
    "PD9 SEC inputs are fetched at run time (as published then); measurement stops at the panel as_of, so the "
    "day-0 range ends at the last full [0,20] window.",
]

CAVEATS = [
    "Alpaca market data: derived statistics only — no price, ticker or per-event return leaves this process.",
    "Survivorship-REDUCED, not free: delisted names outside Alpaca's asset list and corporate-actions feed, "
    "or without a dated ticker in insider filings (~30% of 10-K filers unmapped, skewed small and quiet), are "
    "missing; the remaining bias is still upward.",
    "Default-mapping series of bankruptcies without a dated corporate action may carry an OTC tail "
    "(inactive_no_event_date); the raw-close $1 floor removes most of it.",
    "E4 labels are ~68% precise on survivors (D5); on delisted names merger-completion 8-Ks also match the "
    "frozen rule; the day-0 bar requirement removes those filed after the last trade.",
]


# ── small helpers ────────────────────────────────────────────────────────────
class MemoryBudgetExceeded(RuntimeError):
    pass


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # Linux: KB


def check_rss(limit_mb: float, where: str) -> None:
    if rss_mb() > limit_mb:
        raise MemoryBudgetExceeded(f"RSS {rss_mb():.0f} MB > budget {limit_mb:.0f} MB at {where}")


def study_code_hash(root: Optional[str] = None) -> str:
    """sha256 over (name, NUL, bytes, NUL) of STUDY_FILES in order, first 16 hex.
    server/eventStudy.ts studyCodeHash computes the same digest (both test suites
    pin it on one fixture): the job re-runs when the study code changes, not on
    every unrelated deploy."""
    root = root or os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in STUDY_FILES:
        try:
            with open(os.path.join(root, name), "rb") as fh:
                data = fh.read()
        except OSError:
            return f"missing:{name}"
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
    return h.hexdigest()[:16]


def panel_symbol(ticker: str) -> str:
    """SEC/Yahoo class separator '-' -> the panel's (Alpaca's) '.' (BRK-B -> BRK.B)."""
    return str(ticker or "").strip().upper().replace("-", ".")


def clean(x):
    """JSON-safe copy: tuples -> lists, floats rounded to 6 places, NaN/inf -> None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float):
        return round(x, 6) if math.isfinite(x) else None
    return x


def _write_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(clean(payload), fh, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


def default_out_dir() -> str:
    import storage_config
    return storage_config.EVENT_STUDY_DIR


def default_cache_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "voltrade_event_study")


def prepare_cache(base: str, build_id: str) -> str:
    """<base>/<build>-<CACHE_VERSION>/; every other build's dir is deleted (disk stays bounded)."""
    os.makedirs(base, exist_ok=True)
    keep = f"{build_id}-{CACHE_VERSION}"
    for name in os.listdir(base):
        if name != keep and _BUILD_CACHE_RE.match(name):
            shutil.rmtree(os.path.join(base, name), ignore_errors=True)
    d = os.path.join(base, keep)
    os.makedirs(d, exist_ok=True)
    return d


def dir_bytes(d: str) -> int:
    """Bytes under d (the cache dir is written by this process only, so no race)."""
    return sum(os.path.getsize(os.path.join(dp, fn)) for dp, _dn, fns in os.walk(d) for fn in fns)


# ── SEC inputs (cached under the build's cache dir) ─────────────────────────
def split_listing(d: dict) -> dict:
    """company_tickers_exchange.json -> {"listed": {cik10: first ticker}} by
    EXACTLY event_reaction.stage_universe's rule (NYSE/Nasdaq/CBOE, first listed;
    pinned by test), and "on_exchange": every CIK with ANY current exchange
    listing (incl. TXSE) — those are not delisted names."""
    listed: Dict[str, str] = {}
    on_ex = set()
    for cik, _name, ticker, exch in d["data"]:
        if exch in er.EXCHANGES and ticker:
            listed.setdefault(f"{int(cik):010d}", ticker)
        if exch in CURRENT_EXCHANGES:
            on_ex.add(f"{int(cik):010d}")
    return {"listed": listed, "on_exchange": sorted(on_ex)}


def stage_listing(cache: str, client) -> dict:
    path = os.path.join(cache, "listing.json.gz")
    if os.path.exists(path):
        return er._jload(path)
    out = split_listing(client.get_json(er.TICKERS_EXCHANGE_URL))
    er._jdump(path, out)
    return out


def purchases_from_zip(zf: zipfile.ZipFile) -> List[dict]:
    """event_reaction.stage_purchases' per-quarter rule for EVERY issuer (the
    universe filter comes later, once the delisted CIKs are known): Form 4 (not
    4/A), reporting owner a Director or Officer, non-derivative code P acquired A.
    Pinned to the core by test: same rows for any universe."""
    subs = {}
    for r in er._tsv(zf, "SUBMISSION.tsv"):
        cik = (r.get("ISSUERCIK") or "").strip()
        if r.get("DOCUMENT_TYPE") == "4" and cik.isdigit():
            fd = er.parse_sec_date(r.get("FILING_DATE"))
            if fd:
                subs[r["ACCESSION_NUMBER"]] = (f"{int(cik):010d}", fd)
    owners = defaultdict(set)
    for r in er._tsv(zf, "REPORTINGOWNER.tsv"):
        acc = r.get("ACCESSION_NUMBER")
        rel = r.get("RPTOWNER_RELATIONSHIP") or ""
        if acc in subs and ("Director" in rel or "Officer" in rel):
            owners[acc].add(r.get("RPTOWNERCIK"))
    out = []
    for r in er._tsv(zf, "NONDERIV_TRANS.tsv"):
        acc = r.get("ACCESSION_NUMBER")
        if acc not in owners or r.get("TRANS_CODE") != "P" or r.get("TRANS_ACQUIRED_DISP_CD") != "A":
            continue
        td = er.parse_sec_date(r.get("TRANS_DATE"))
        if not td:
            continue
        issuer, fd = subs[acc]
        for o in sorted(owners[acc], key=str):
            out.append({"issuer": issuer, "owner": o, "trans_date": td, "filing_date": fd, "accession": acc})
    return out


def _f345_paths(cache: str, key: str) -> Tuple[str, str]:
    d = os.path.join(cache, "form345")
    return os.path.join(d, f"{key}.pairs.json.gz"), os.path.join(d, f"{key}.purchases.json.gz")


def stage_form345(cache: str, client) -> dict:
    """One download per quarter (PAIRS_FROM .. now) -> two compact caches; the zip
    itself is parsed in memory and never kept:
      <key>.pairs.json.gz      (ticker, issuer cik, filing date) of every Form 3/4/5 (sec_issuer_pairs)
      <key>.purchases.json.gz  purchases_from_zip rows (quarters >= PURCHASES_FROM only)
    A quarter that is not published (404) or fails is listed in "missing", never fatal here."""
    import sec_filings_text as sft
    os.makedirs(os.path.join(cache, "form345"), exist_ok=True)
    out: Dict[str, list] = {"pairs": [], "purchases": [], "missing": []}
    for (y, q) in er._quarters(start=PAIRS_FROM):
        key = f"{y}q{q}"
        want_p = (y, q) >= PURCHASES_FROM
        pa, pu = _f345_paths(cache, key)
        if os.path.exists(pa) and (not want_p or os.path.exists(pu)):
            out["pairs"].append(key)
            if want_p:
                out["purchases"].append(key)
            continue
        blob, why = None, "not published"
        for url in er.FORM345_URLS:
            try:
                blob = client.get(url.format(y=y, q=q))
                break
            except sft.EdgarHTTPError as e:
                why = str(e)[-120:]
        if blob is None:
            er.log(f"form345 {key}: unavailable ({why})")
            out["missing"].append(key)
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                if want_p:
                    rows = purchases_from_zip(zf)
                    er._jdump(pu, [[p["issuer"], p["owner"], p["trans_date"], p["filing_date"], p["accession"]] for p in rows])
                skipped: Dict[str, int] = {}
                er._jdump(pa, sip.pairs_from_submission_tsv(zf.read("SUBMISSION.tsv"), skipped))
        except (zipfile.BadZipFile, KeyError) as e:
            er.log(f"form345 {key}: unreadable zip ({type(e).__name__}: {e})")
            out["missing"].append(key)
            continue
        out["pairs"].append(key)
        if want_p:
            out["purchases"].append(key)
        er.log(f"form345 {key}: cached")
    return out


def clear_e4_errors(cache: str) -> int:
    """event_reaction.stage_502_text caches a failed document as an "error: ..."
    string and never retries it; drop those entries so this run re-fetches them.
    -> number dropped."""
    path = os.path.join(cache, "e4_class.json.gz")
    if not os.path.exists(path):
        return 0
    done = er._jload(path)
    keep = {a: v for a, v in done.items() if not isinstance(v, str)}
    if len(keep) != len(done):
        er._jdump(path, keep)
    return len(done) - len(keep)


def load_purchases(cache: str, keys: Iterable[str], universe: set) -> List[dict]:
    """Universe issuers only, deduped on (owner, accession, trans_date) keeping the
    first, as event_reaction.stage_purchases does."""
    rows, seen = [], set()
    for key in sorted(keys):
        for issuer, owner, td, fd, acc in er._jload(_f345_paths(cache, key)[1]):
            if issuer not in universe:
                continue
            k = (owner, acc, td)
            if k in seen:
                continue
            seen.add(k)
            rows.append({"issuer": issuer, "owner": owner, "trans_date": td, "filing_date": fd, "accession": acc})
    return rows


def load_pairs_map(cache: str, keys: Iterable[str]) -> dict:
    def rows():
        for key in sorted(keys):
            for t, cik, d in er._jload(_f345_paths(cache, key)[0]):
                yield t, cik, d
    return csm.build_map(rows())


def stage_submissions(cache: str, client, ciks: Iterable[str], need_f4: Dict[str, set], workers: int,
                      max_rss_mb: float) -> dict:
    """event_reaction._submissions_summary per CIK, resumable: checkpointed every
    FETCH_CHUNK CIKs; a cached CIK is re-fetched only if it now needs acceptance
    times for Form 4 accessions it was not asked for."""
    path = os.path.join(cache, "submissions.json.gz")
    done = er._jload(path) if os.path.exists(path) else {}
    ciks = sorted(set(ciks))

    def stale(c: str) -> bool:
        e = done.get(c)
        return e is None or not set(need_f4.get(c, ())) <= set(e["f4"])

    todo = [c for c in ciks if stale(c)]
    er.log(f"submissions: {len(ciks)} CIKs, {len(todo)} to fetch")
    errors: List[list] = []

    def work(cik: str):
        try:
            return cik, er._submissions_summary(client, cik, set(need_f4.get(cik, ()))), None
        except Exception as e:  # recorded + reported (meta.submissions_errors); the CIK drops from the sample
            return cik, None, f"{type(e).__name__}: {e}"[:200]

    with ThreadPoolExecutor(max(1, int(workers))) as ex:
        for start in range(0, len(todo), FETCH_CHUNK):
            for cik, summ, err in ex.map(work, todo[start:start + FETCH_CHUNK]):
                if err:
                    errors.append([cik, err])
                else:
                    done[cik] = {"s": summ, "f4": sorted(need_f4.get(cik, ()))}
            er._jdump(path, done)
            check_rss(max_rss_mb, f"submissions {start + FETCH_CHUNK}/{len(todo)}")
            if (start // FETCH_CHUNK) % 10 == 9:
                er.log(f"submissions {min(start + FETCH_CHUNK, len(todo))}/{len(todo)} (errors {len(errors)})")
    return {"by_cik": {c: done[c]["s"] for c in ciks if c in done}, "errors": errors}


# ── panel ────────────────────────────────────────────────────────────────────
def index_panel(build_dir: str) -> Tuple[Optional[pp.Series], Dict[str, List[Tuple[int, int, str]]]]:
    """One pass: the longest SPY series (the calendar) and, per symbol, every
    series' (first day, last day, mode)."""
    spy: Optional[pp.Series] = None
    idx: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for s in pp.iter_series(build_dir):
        idx[s.symbol].append((s.first_day, s.last_day, s.mode))
        if s.symbol == "SPY" and (spy is None or len(s.c) > len(spy.c)):
            spy = s
    return spy, dict(idx)


def series_bars(s: pp.Series) -> dict:
    return {"date": s.dates, "close": s.c.tolist(), "high": s.h.tolist(), "low": s.l.tolist(), "volume": s.v.tolist()}


def delisted_candidates(pairs_map: dict, on_exchange: set, sym_index: dict,
                        slack_days: int = csm.SLACK_DAYS) -> Dict[str, List[str]]:
    """cik10 -> panel symbols it wore, for CIKs NOT currently exchange-listed whose
    dated ticker span (+/- slack) overlaps a panel series of that symbol."""
    out: Dict[str, List[str]] = {}
    for cik, spans in pairs_map.items():
        if not str(cik).isdigit():
            continue
        c10 = f"{int(cik):010d}"
        if c10 in on_exchange:
            continue
        hits = set()
        for tkr, (a, b, _n) in spans.items():
            sym = panel_symbol(tkr)
            segs = sym_index.get(sym)
            if not segs:
                continue
            lo, hi = pp.day_num(a) - slack_days, pp.day_num(b) + slack_days
            if any(f <= hi and l >= lo for f, l, _m in segs):
                hits.add(sym)
        if hits:
            out[c10] = sorted(hits)
    return out


def map_events(raw_events: list, days: list, listed: Dict[str, str], pairs_map: dict,
               dropped: Counter) -> Dict[str, list]:
    """(cik, type, acceptance UTC, filing date) -> {panel symbol: [(type, i0, cik,
    origin, dated_confirms, (first, last) day the CIK filed under that symbol)]}.
    Survivors use their pre-registered current ticker; delisted CIKs their dated
    ticker on day 0 (PD2). Unresolvable events are counted in `dropped`."""
    by_sym: Dict[str, list] = defaultdict(list)
    for cik, et, acc_utc, fd in raw_events:
        i0 = er.event_day0(days, acc_utc, fd)
        if i0 is None:
            dropped[("beyond_panel_calendar", et)] += 1
            continue
        dated, status = csm.resolve(pairs_map, cik, days[i0])
        spans = pairs_map.get(csm.norm_cik(cik), {})
        if cik in listed:
            sym, origin = panel_symbol(listed[cik]), "survivor"
            confirms = dated is not None and panel_symbol(dated) == sym
        else:
            if dated is None:
                dropped[(f"identity_{status}", et)] += 1
                continue
            sym, origin, confirms = panel_symbol(dated), "delisted", True
        span = spans.get(dated) if (dated is not None and confirms) else None
        by_sym[sym].append((et, i0, cik, origin, confirms, (pp.day_num(span[0]), pp.day_num(span[1])) if span else None))
    return dict(by_sym)


def measure_panel(build_dir: str, days: list, spy_ret: list, spy_bull: list, as_of: str,
                  ev_by_sym: Dict[str, list], bp_symbols: set, max_rss_mb: float) -> Tuple[list, dict, Counter, Counter]:
    """Second pass: every series of a universe symbol through event_reaction.measure_ticker
    with the panel's delisting return and raw-close floor. -> (records, {year: [median $vol]},
    per-reason/type counts, per-type counts of events no series was trading for)."""
    pos = {d: i for i, d in enumerate(days)}
    n = len(days)
    recs: List[dict] = []
    yearly: Dict[str, list] = defaultdict(list)
    diag: Counter = Counter()
    matched = set()
    seen = 0
    for s in pp.iter_series(build_dir):
        seen += 1
        if seen % RSS_CHECK_EVERY == 0:
            check_rss(max_rss_mb, f"panel series {seen}")
        if s.symbol == "SPY":
            continue
        evs = ev_by_sym.get(s.symbol) or []
        in_bp = s.symbol in bp_symbols
        if not evs and not in_bp:
            continue
        close: List[Optional[float]] = [None] * n
        floor: List[Optional[float]] = [None] * n
        for d, c, f in zip(s.dates, s.c.tolist(), pp.pit_price(s).tolist()):
            i = pos.get(d)
            if i is not None:
                close[i], floor[i] = c, f
        r_del = panel_delisting_return(s) if s.ends_before(as_of) else None
        last = max((i for i, c in enumerate(close) if c is not None), default=None)
        by_type: Dict[str, list] = defaultdict(list)
        origin_of: Dict[Tuple[str, int], str] = {}
        for k, (et, i0, _cik, origin, confirms, span) in enumerate(evs):
            if i0 < 1 or close[i0 - 1] is None:
                continue                      # not this series (another one, or none, traded on day -1)
            matched.add((s.symbol, k))
            # PD7: the series must overlap the CIK's own filing span under this symbol, so a
            # ticker that belonged to another company before or after the CIK is never used
            overlaps = span is not None and s.first_day <= span[1] and s.last_day >= span[0]
            if origin == "survivor" and s.mode == "raw" and not (confirms and overlaps):
                diag[("reuse_unconfirmed", et)] += 1
                continue
            if origin == "delisted" and not overlaps:
                diag[("symbol_reassigned", et)] += 1
                continue
            if close[i0] is None:
                diag[("no_bar_on_day0_merger" if r_del == DELIST_RETURN_MERGER else "no_bar_on_day0", et)] += 1
            adj_ok, raw_ok = close[i0 - 1] >= er.MIN_PRICE, floor[i0 - 1] is not None and floor[i0 - 1] >= er.MIN_PRICE
            if adj_ok != raw_ok:
                diag[("floor_raw_admits" if raw_ok else "floor_raw_excludes", et)] += 1
            by_type[et].append(i0)
            if origin == "survivor" or (et, i0) not in origin_of:
                origin_of[(et, i0)] = origin
        r, ym = er.measure_ticker(s.symbol, days, spy_ret, spy_bull, series_bars(s), dict(by_type),
                                  delisting_return=r_del, floor_close=floor)
        for rec in r:
            i0 = pos[rec["day0"]]
            rec.pop("ticker", None)           # nothing downstream needs it; keeps tickers out of every output
            rec["origin"] = origin_of.get((rec["type"], i0), "survivor")
            dl = (last + 1 - i0) if (r_del is not None and last is not None) else None
            rec["delist_day"] = dl if (dl is not None and 0 <= dl <= 60) else None
            rec["delist_kind"] = ("merger" if r_del == DELIST_RETURN_MERGER else "other") if rec["delist_day"] is not None else None
        recs.extend(r)
        if in_bp:
            for y, v in ym.items():
                yearly[y].append(v)
    unmatched: Counter = Counter()
    for sym, evs in ev_by_sym.items():
        for k, ev in enumerate(evs):
            if (sym, k) not in matched:
                unmatched[ev[0]] += 1
    diag[("_series_seen", "")] = seen
    return recs, dict(yearly), diag, unmatched


# ── verdicts on the panel (PD6) ─────────────────────────────────────────────
def panel_verdicts(summary: dict) -> dict:
    """event_reaction.summarize output -> the same tables with rule 4 read for a
    delisting-inclusive panel: a positive cell meeting tests 1-3 becomes
    candidate_provisional; the core's labels are kept as verdict_core."""
    out = {}
    for et, blk in summary.items():
        tbl = blk["table"]
        cells = []
        for b in ("small", "mid", "large"):
            core = tbl[b]["verdict"]
            tbl[b]["verdict_core"] = core
            tbl[b]["verdict"] = "candidate_provisional" if core == "positive_unconfirmed" else core
            cells.append(tbl[b]["verdict"])
        t01 = tbl["pooled"]["0_1"].get("t")
        if "candidate" in cells:
            v = er.type_verdict(cells, t01)
        elif "candidate_provisional" in cells:
            v = PANEL_PROVISIONAL
        else:
            v = er.type_verdict(cells, t01)
        out[et] = {**blk, "verdict_core": blk["verdict"], "verdict": v}
    return out


def _by_type(c: Counter) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (reason, et), v in c.items():
        if not reason.startswith("_"):
            out[reason][et] = int(v)
    return dict(out)


# ── run ──────────────────────────────────────────────────────────────────────
def run(panel_root: Optional[str] = None, out_dir: Optional[str] = None, cache_dir: Optional[str] = None,
        max_rss_mb: float = DEFAULT_MAX_RSS_MB, code_version: Optional[str] = None, workers: int = DEFAULT_WORKERS,
        rate: float = DEFAULT_RATE, client=None, now: Optional[datetime] = None) -> dict:
    t0 = time.time()
    out_dir = out_dir or default_out_dir()
    now = now or datetime.now(timezone.utc)
    code = study_code_hash()
    last_run = os.path.join(out_dir, "last_run.json")

    def finish(res: dict) -> dict:
        res = {**res, "study_code": code, "code_version": code_version}
        _write_atomic(last_run, {**res, "finished_at": datetime.now(timezone.utc).isoformat()})
        return res

    build = pp.current_build(panel_root)
    if not build or not build["manifest"].get("as_of"):
        return finish({"status": "skipped", "reason": "no complete price-panel build (current.json absent)"})
    as_of = str(build["manifest"]["as_of"])
    cache = prepare_cache(cache_dir or default_cache_dir(), build["build_id"])

    spy, sym_index = index_panel(build["dir"])
    if spy is None or len(spy.c) < er.MIN_HISTORY + 60:
        return finish({"status": "insufficient", "reason": "panel has no usable SPY series (trading calendar)",
                       "build_id": build["build_id"]})
    days = spy.dates
    spy_ret, spy_bull = er.spy_series(days, series_bars(spy))
    check_rss(max_rss_mb, "panel index")

    if client is None:
        import sec_filings_text as sft
        sft.process_limiter().set_rate(rate)
        client = sft.EdgarClient()
    listing = stage_listing(cache, client)
    f345 = stage_form345(cache, client)
    as_q = (int(as_of[:4]), (int(as_of[5:7]) - 1) // 3 + 1)
    # every closed quarter before the as_of's quarter is required, except the
    # latest one (SEC publishes a quarter's data set weeks after it closes);
    # whatever is missing is listed in meta.form345.missing
    need_q = [f"{y}q{q}" for (y, q) in er._quarters(start=PAIRS_FROM) if (y, q) < as_q][:-1]
    gaps = [k for k in need_q if k in f345["missing"]]
    if gaps:
        return finish({"status": "insufficient", "build_id": build["build_id"],
                       "reason": f"SEC insider data sets missing for {gaps[:6]} (needed before the panel as_of {as_of})"})
    pairs_map = load_pairs_map(cache, f345["pairs"])
    listed: Dict[str, str] = listing["listed"]
    delisted = delisted_candidates(pairs_map, set(listing["on_exchange"]), sym_index)
    universe = set(listed) | set(delisted)
    purchases = load_purchases(cache, f345["purchases"], universe)
    clusters = er.insider_clusters(purchases)
    need_f4: Dict[str, set] = defaultdict(set)
    for c in clusters:
        need_f4[c["issuer"]].add(c["accession"])
    check_rss(max_rss_mb, "insider clusters")
    subs = stage_submissions(cache, client, universe, need_f4, workers, max_rss_mb)
    by_cik = subs["by_cik"]
    op = {cik for cik, s in by_cik.items() if s["has_10kq"]}

    raw_events = []
    f4_hits = 0
    for c in clusters:
        if c["issuer"] not in op:
            continue
        acc_utc = by_cik[c["issuer"]]["f4"].get(c["accession"])
        f4_hits += acc_utc is not None
        raw_events.append((c["issuer"], er.EVENT_TYPES[0], acc_utc, c["filing_date"]))
    f502 = []
    for cik in sorted(op):
        for f in by_cik[cik]["eightk"]:
            if "1.01" in f["items"]:
                raw_events.append((cik, er.EVENT_TYPES[1], f["acc_utc"], f["fd"]))
            if "4.02" in f["items"]:
                raw_events.append((cik, er.EVENT_TYPES[2], f["acc_utc"], f["fd"]))
            if "5.02" in f["items"]:
                f502.append({"cik": cik, **f})
    clear_e4_errors(cache)
    e4 = er.stage_502_text(cache, client, f502, workers)
    selected = set(e4["selected"])
    e4_errors = sum(1 for a in selected if isinstance(e4["class"].get(a), str))
    sub_share = len(subs["errors"]) / max(1, len(universe))
    e4_share = e4_errors / max(1, len(selected))
    if sub_share > MAX_FETCH_ERROR_SHARE or e4_share > MAX_FETCH_ERROR_SHARE:
        return finish({"status": "insufficient", "build_id": build["build_id"],
                       "reason": f"SEC fetch errors: submissions {len(subs['errors'])}/{len(universe)}, "
                                 f"5.02 documents {e4_errors}/{len(selected)} (> {MAX_FETCH_ERROR_SHARE:.0%}); "
                                 f"successes stay cached for the next attempt"})
    for f in f502:
        # only THIS run's seeded selection (§1.2): the cache may also hold labels of an
        # earlier selection made on a different 5.02 population within the same build
        if f["acc"] in selected and e4["class"].get(f["acc"]) is True:
            raw_events.append((f["cik"], er.EVENT_TYPES[3], f["acc_utc"], f["fd"]))
    origin = {c: ("survivor" if c in listed else "delisted") for c in universe}
    meta_sec = {
        "listed_ciks": len(listed), "on_exchange_ciks": len(listing["on_exchange"]), "pairs_ciks": len(pairs_map),
        "delisted_candidate_ciks": len(delisted), "universe_ciks": len(universe),
        "submissions_ok": len(by_cik), "submissions_errors": len(subs["errors"]),
        "operating_ciks": dict(Counter(origin[c] for c in op)),
        "form345": {"pairs_quarters": len(f345["pairs"]), "purchase_quarters": len(f345["purchases"]),
                    "missing": f345["missing"]},
        "purchase_rows": len(purchases), "e1_clusters_all": len(clusters), "e1_acceptance_found": f4_hits,
        "e4_502_filings": len(f502), "e4_docs_selected": len(selected), "e4_doc_errors": e4_errors,
        "e4_positive": sum(1 for a in selected if e4["class"].get(a) is True),
        "raw_events_by_type": dict(Counter(e[1] for e in raw_events)),
        "raw_events_by_origin": {o: dict(Counter(e[1] for e in raw_events if origin[e[0]] == o)) for o in ("survivor", "delisted")},
    }
    del purchases, clusters, need_f4, subs, by_cik, f502, e4, selected
    check_rss(max_rss_mb, "SEC stages")

    dropped: Counter = Counter()
    ev_by_sym = map_events(raw_events, days, listed, pairs_map, dropped)
    bp_symbols = {panel_symbol(t) for t in listed.values()} | {s for syms in delisted.values() for s in syms}
    del raw_events, pairs_map
    recs, yearly, diag, unmatched = measure_panel(build["dir"], days, spy_ret, spy_bull, as_of, ev_by_sym,
                                                  bp_symbols, max_rss_mb)
    if not recs:
        return finish({"status": "insufficient", "reason": "no measurable event on the panel", "build_id": build["build_id"]})
    breaks = {y: (er.percentile(v, 1 / 3), er.percentile(v, 2 / 3)) for y, v in yearly.items() if len(v) >= 30}
    summary = panel_verdicts(er.summarize(recs, breaks))
    surv = er.summarize([r for r in recs if r["origin"] == "survivor"], breaks)

    delist_in = defaultdict(Counter)
    for r in recs:
        if r["delist_day"] is not None:
            delist_in[r["type"]][f"{r['delist_kind']}_{'0_20' if r['delist_day'] <= 20 else '21_60'}"] += 1
    measured = Counter(r["type"] for r in recs)
    meta = {
        **meta_sec,
        "dropped": {**_by_type(dropped), **_by_type(Counter({k: v for k, v in diag.items() if not k[0].startswith("floor")
                                                             and not k[0].startswith("no_bar")}))},
        "no_series_trading_on_day_minus_1": dict(unmatched),
        "stopped_trading_at_event": _by_type(Counter({k: v for k, v in diag.items() if k[0].startswith("no_bar")})),
        "floor_flips": _by_type(Counter({k: v for k, v in diag.items() if k[0].startswith("floor")})),
        "measured_by_type": {t: int(measured.get(t, 0)) for t in er.EVENT_TYPES},
        "measured_by_origin": {o: dict(Counter(r["type"] for r in recs if r["origin"] == o)) for o in ("survivor", "delisted")},
        "delisting_return_in_window": {t: dict(v) for t, v in delist_in.items()},
        "day0_range": [min(r["day0"] for r in recs), max(r["day0"] for r in recs)],
        "size_breaks": breaks, "spy_days": [days[0], days[-1]], "panel_series_seen": int(diag[("_series_seen", "")]),
        "corrupt_panel_lines_skipped": len(pp.corrupt_lines()),
        "sec_cache_bytes": dir_bytes(cache),
    }
    verdicts = {t: summary[t]["verdict"] for t in er.EVENT_TYPES}
    payload = {
        "schema": SCHEMA, "status": "ok", "generated_at": now.isoformat(), "code_version": code_version,
        "study_code": code, "as_of": as_of,
        "panel": {"build_id": build["build_id"], "as_of": as_of, "completed_at": build.get("completed_at"),
                  "survivorship": "REDUCED (2016+; delisted names outside Alpaca assets/corporate actions or without a dated insider-filing ticker are missing)"},
        "preregistration": "research/event_reaction_study.md §1 (frozen 2026-10-29, sha256 fd7cda39...)",
        "bars": {"t": er.T_BAR, "t_e1": er.T_BAR_E1, "cells": 12, "primary": "excess CAR[2,20]", "min_move": er.MIN_MOVE},
        "delisting_assumption": {"merger_or_rename": DELIST_RETURN_MERGER, "other": DELIST_RETURN_OTHER,
                                 "after_delisting": "SPY (abnormal return 0)", "source": "Shumway (1997); walkforward.delisting_return"},
        "prior": PRIOR, "deviations": DEVIATIONS, "caveats": CAVEATS,
        "meta": meta, "verdicts": verdicts, "summary": summary,
        "survivor_subset": {"note": "events of the pre-registered survivor universe only, on this panel: separates the "
                                    "price-source change from the delisted names (diagnostic; rule 4 as written)",
                            "verdicts": {t: surv[t]["verdict"] for t in er.EVENT_TYPES}, "summary": surv},
        "elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": round(rss_mb(), 1),
    }
    _write_atomic(os.path.join(out_dir, "latest.json"), payload)
    return finish({"status": "ok", "build_id": build["build_id"], "as_of": as_of, "events": len(recs),
                   "elapsed_s": payload["elapsed_s"], "peak_rss_mb": payload["peak_rss_mb"], "verdicts": verdicts})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered event-reaction study on the delisting-inclusive panel.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--panel-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-rss-mb", type=float, default=DEFAULT_MAX_RSS_MB)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--code-version", default=None)
    ap.add_argument("--nice", type=int, default=0, help="raise this process's niceness first (the weekly job passes 19)")
    a = ap.parse_args(argv)
    if not a.run:
        ap.print_help()
        return 2
    if a.nice > 0:
        try:
            os.nice(a.nice)       # never compete with the trading loop / daemon for CPU
        except OSError as e:
            print(f"[event_study] could not lower priority: {e}", file=sys.stderr)
    out_dir = a.out_dir or default_out_dir()
    try:
        res = run(a.panel_root, out_dir, a.cache_dir, a.max_rss_mb, a.code_version, a.workers, a.rate)
    except MemoryBudgetExceeded as e:
        res = {"status": "error", "reason": f"memory budget: {e}"}
        _write_atomic(os.path.join(out_dir, "last_run.json"), {**res, "finished_at": datetime.now(timezone.utc).isoformat()})
    except Exception as e:   # noqa: BLE001 — one JSON line for the scheduler; the last good latest.json is untouched
        res = {"status": "error", "reason": f"{type(e).__name__}: {e}"[:300]}
        _write_atomic(os.path.join(out_dir, "last_run.json"), {**res, "finished_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(res, separators=(",", ":")))
    return 0 if res.get("status") in ("ok", "skipped", "insufficient") else 1


if __name__ == "__main__":
    sys.exit(main())
