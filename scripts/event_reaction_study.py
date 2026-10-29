#!/usr/bin/env python3
"""
scripts/event_reaction_study.py — PRE-REGISTERED event-reaction study
(research/event_reaction_study.md §1 is the pre-registration; this script
implements it and nothing else).

Four event types, survivor universe of exchange-listed SEC filers, 2016+:
  E1 insider purchase cluster (Form 4 code P, >=2 distinct officer/director
     buyers within 10 calendar days)   — SEC Insider Transactions Data Sets
  E2 8-K Item 1.01 material definitive agreement   — submissions JSON items
  E3 8-K Item 4.02 non-reliance (restatement)      — submissions JSON items
  E4 8-K Item 5.02 sudden CEO/CFO departure        — items + primary-doc text

Measures SPY-adjusted CARs over [0,1] [2,5] [2,20] [2,60] (and [0,20]) from
the first trading day the filing was PUBLIC (acceptance timestamp; after
16:00 ET -> next trading day), minus a same-ticker random-date control
(base rate), by liquidity tercile, regime and half-sample, with SEs
clustered by calendar week, and applies the pre-registered verdict bar.

Data hygiene: every raw SEC / Yahoo download is cached under --cache
(default: $TMPDIR/event_reaction_cache), never under the repo. SEC fair
access: the repo's declared User-Agent and process-wide <=10 req/s limiter
(sec_filings_text.EdgarClient). No trading logic is imported or touched.

Usage:
  python3 scripts/event_reaction_study.py fetch [--cache DIR]   # network only
  python3 scripts/event_reaction_study.py run [--cache DIR] [--workers 6]
  python3 scripts/event_reaction_study.py report [--cache DIR]
"""
import argparse
import bisect
import csv
import gzip
import io
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ET = ZoneInfo("America/New_York")
SEED = 20261029

# ── Pre-registered constants (research/event_reaction_study.md §1) ─────────
EVENT_TYPES = ("E1_insider_cluster", "E2_item_1_01", "E3_item_4_02", "E4_ceo_cfo_exit")
WINDOWS = {"0_1": (0, 1), "2_5": (2, 5), "2_20": (2, 20), "2_60": (2, 60), "0_20": (0, 20)}
PRIMARY = "2_20"
CLUSTER_DAYS = 10            # E1: distinct buyers within 10 calendar days
CLUSTER_COOLDOWN_DAYS = 30   # E1: one cluster per issuer per 30 calendar days
DEDUP_TRADING_DAYS = 20      # same ticker+type within 20 trading days -> keep first
N_CONTROLS = 5
CONTROL_SPAN = 250
CONTROL_EXCLUDE = 20
MIN_PRICE = 1.0
MIN_HISTORY = 60
SIZE_LOOKBACK = (-250, -6)
SPREAD_LOOKBACK = (-60, -1)
WINSOR = (0.01, 0.99)
MIN_MOVE = 0.01
T_BAR = 2.87                 # 0.05 / 12 cells, two-sided
T_BAR_E1 = 3.02              # 0.05 / (12 + 8 prior Form 4 cells), two-sided
HALF_T = 1.0
HALF_SPLIT_YEAR = 2020       # 2016-2020 vs 2021-latest
MAX_502_DOCS = 30000
SAMPLE_START = "2016-01-01"
EXCHANGES = ("NYSE", "Nasdaq", "CBOE")

FORM345_URLS = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip",
    "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{y}q{q}_form345.zip",
)
TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


# ════════════════════════════════════════════════════════════════════════════
# Pure functions (unit-tested in test_event_reaction_study.py)
# ════════════════════════════════════════════════════════════════════════════

def acceptance_to_et(accept_utc: str) -> datetime:
    """EDGAR submissions-JSON acceptanceDateTime is TRUE UTC (verified against
    index-headers ACCEPTANCE-DATETIME, which is ET). -> aware ET datetime."""
    dt = datetime.strptime(str(accept_utc)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return dt.astimezone(ET)


def first_trading_index_after(days: list, d: str) -> Optional[int]:
    """Index of the first trading day STRICTLY after ISO date d."""
    i = bisect.bisect_right(days, d)
    return i if i < len(days) else None


def event_day0(days: list, accept_utc: Optional[str] = None,
               filing_date: Optional[str] = None) -> Optional[int]:
    """Day-0 index (first trading day the filing was public).
    Acceptance known: same ET date if it is a trading day and the time is
    before 16:00 ET, else the next trading day. Acceptance unknown: the first
    trading day strictly after the filing date (conservative)."""
    if accept_utc:
        et = acceptance_to_et(accept_utc)
        d = et.date().isoformat()
        i = bisect.bisect_left(days, d)
        if i < len(days) and days[i] == d and (et.hour, et.minute) < (16, 0):
            return i
        return first_trading_index_after(days, d)
    if filing_date:
        return first_trading_index_after(days, filing_date)
    return None


def simple_returns(closes: list) -> list:
    """r[t] = c[t]/c[t-1]-1; None where either close is missing or <= 0."""
    out = [None] * len(closes)
    for t in range(1, len(closes)):
        a, b = closes[t - 1], closes[t]
        if a is not None and b is not None and a > 0 and b > 0:
            out[t] = b / a - 1.0
    return out


def car(stock_ret: list, mkt_ret: list, i0: int, a: int, b: int) -> Optional[float]:
    """Sum of (r_stock - r_mkt) over t in [i0+a, i0+b]. None if the window
    runs outside the data or any day in it is missing (full-coverage rule)."""
    lo, hi = i0 + a, i0 + b
    if lo < 1 or hi >= len(stock_ret) or hi >= len(mkt_ret):
        return None
    s = 0.0
    for t in range(lo, hi + 1):
        rs, rm = stock_ret[t], mkt_ret[t]
        if rs is None or rm is None:
            return None
        s += rs - rm
    return s


class CarIndex:
    """O(1) CAR lookups via prefix sums; identical results to car()."""

    def __init__(self, stock_ret: list, mkt_ret: list):
        n = min(len(stock_ret), len(mkt_ret))
        self.n = n
        self.cum = [0.0] * (n + 1)
        self.bad = [0] * (n + 1)
        for t in range(n):
            rs, rm = stock_ret[t], mkt_ret[t]
            ok = rs is not None and rm is not None
            self.cum[t + 1] = self.cum[t] + ((rs - rm) if ok else 0.0)
            self.bad[t + 1] = self.bad[t] + (0 if ok else 1)

    def car(self, i0: int, a: int, b: int) -> Optional[float]:
        lo, hi = i0 + a, i0 + b
        if lo < 1 or hi >= self.n:
            return None
        if self.bad[hi + 1] - self.bad[lo]:
            return None
        return self.cum[hi + 1] - self.cum[lo]


def parse_items(items: str) -> set:
    return {x.strip() for x in str(items or "").split(",") if x.strip()}


_ROLE_LONG = re.compile(r"\b(chief executive officer|chief financial officer|"
                        r"principal executive officer|principal financial officer)\b", re.I)
_ROLE_SHORT = re.compile(r"\b(CEO|CFO)\b")
_DEPART = re.compile(r"\b(resign\w*|terminat\w*|depart\w*|step(?:s|ping)? down|stepped down|"
                     r"separat\w*|ceas\w*|removed)\b", re.I)
_RETIRE = re.compile(r"\bretire\w*", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.;])\s+|\n+")


def classify_departure(text: str) -> bool:
    """E4 rule: some sentence names the CEO/CFO role AND a departure word AND
    does not contain retire*."""
    for s in _SENT_SPLIT.split(text or ""):
        if not (_ROLE_LONG.search(s) or _ROLE_SHORT.search(s)):
            continue
        if _DEPART.search(s) and not _RETIRE.search(s):
            return True
    return False


def _d(s: str) -> date:
    return date.fromisoformat(s)


def insider_clusters(purchases: list) -> list:
    """purchases: dicts {issuer, owner, trans_date, filing_date, accession}
    (ISO dates). Backward-only: a cluster fires on the first FILING at which
    >=2 distinct owners have purchase trans dates within the trailing
    CLUSTER_DAYS calendar days of that filing's FILING date, counting only
    purchases filed at or before it (a stale late-filed trade outside the
    window does not count). One per issuer per CLUSTER_COOLDOWN_DAYS.
    -> [{issuer, filing_date, accession}]"""
    by_issuer = defaultdict(list)
    for p in purchases:
        by_issuer[p["issuer"]].append(p)
    out = []
    for issuer, rows in by_issuer.items():
        by_acc = defaultdict(list)
        for r in rows:
            by_acc[(r["filing_date"], r["accession"])].append(r)
        seen, last = [], None
        for (fd, acc) in sorted(by_acc):
            group = by_acc[(fd, acc)]
            seen.extend(group)
            ref = _d(fd)
            lo = ref - timedelta(days=CLUSTER_DAYS)
            owners = {r["owner"] for r in seen if lo <= _d(r["trans_date"]) <= ref}
            if len(owners) >= 2 and (last is None or (_d(fd) - last).days >= CLUSTER_COOLDOWN_DAYS):
                out.append({"issuer": issuer, "filing_date": fd, "accession": acc})
                last = _d(fd)
    return out


def dedup_day0(idx_list: list, gap: int = DEDUP_TRADING_DAYS) -> list:
    """Sorted unique day-0 indices with any index within `gap` trading days
    after a KEPT index dropped (keep first)."""
    kept = []
    for i in sorted(set(idx_list)):
        if not kept or i - kept[-1] > gap:
            kept.append(i)
    return kept


def control_pool(i0: int, n_days: int, event_idx: list, span: int = CONTROL_SPAN,
                 exclude: int = CONTROL_EXCLUDE) -> list:
    """Candidate control day-0s: within +-span of i0, excluding days within
    +-exclude of ANY same-type event of the ticker (event_idx)."""
    ev = sorted(event_idx)
    out = []
    for j in range(max(1, i0 - span), min(n_days, i0 + span + 1)):
        k = bisect.bisect_left(ev, j - exclude)
        if k < len(ev) and ev[k] <= j + exclude:
            continue
        out.append(j)
    return out


def event_rng(ticker: str, i0: int, etype: str) -> random.Random:
    return random.Random(f"{SEED}|{ticker}|{i0}|{etype}")


def abdi_ranaldo(high: list, low: list, close: list) -> Optional[float]:
    """Abdi & Ranaldo (2017) effective spread from close/high/low:
    s^2 = 4 * mean[(c_t - eta_t)(c_t - eta_{t+1})], eta = mid of log H/L.
    Negative estimates floor at 0. Returns a proportional round-trip spread."""
    prods = []
    for t in range(len(close) - 1):
        vals = (high[t], low[t], close[t], high[t + 1], low[t + 1])
        if any(v is None or v <= 0 for v in vals):
            continue
        c = math.log(close[t])
        eta = (math.log(high[t]) + math.log(low[t])) / 2
        eta1 = (math.log(high[t + 1]) + math.log(low[t + 1])) / 2
        prods.append((c - eta) * (c - eta1))
    if len(prods) < 20:
        return None
    s2 = 4.0 * sum(prods) / len(prods)
    return math.sqrt(s2) if s2 > 0 else 0.0


def percentile(vals: list, q: float) -> float:
    """Linear-interpolated percentile, q in [0,1]."""
    s = sorted(vals)
    if not s:
        raise ValueError("empty")
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def winsorize(vals: list, lo_q: float = WINSOR[0], hi_q: float = WINSOR[1],
              bounds: Optional[tuple] = None) -> list:
    if not vals:
        return []
    lo, hi = bounds if bounds else (percentile(vals, lo_q), percentile(vals, hi_q))
    return [min(max(v, lo), hi) for v in vals]


def clustered_mean_t(vals: list, clusters: list) -> dict:
    """Mean with cluster-robust SE: V = G/(G-1) * sum_g (sum_i e_i)^2 / n^2."""
    n = len(vals)
    if n < 2:
        return {"n": n, "mean": vals[0] if vals else None, "se": None, "t": None, "g": n}
    m = sum(vals) / n
    sums = defaultdict(float)
    for v, c in zip(vals, clusters):
        sums[c] += v - m
    g = len(sums)
    if g < 2:
        return {"n": n, "mean": m, "se": None, "t": None, "g": g}
    var = (g / (g - 1)) * sum(s * s for s in sums.values()) / (n * n)
    se = math.sqrt(var)
    return {"n": n, "mean": m, "se": se, "t": (m / se) if se > 0 else None, "g": g}


def week_cluster(day: str) -> str:
    y, w, _ = _d(day).isocalendar()
    return f"{y}-W{w:02d}"


def size_bucket(v: Optional[float], breaks: Optional[tuple]) -> Optional[str]:
    if v is None or not breaks:
        return None
    lo, hi = breaks
    return "small" if v <= lo else ("mid" if v <= hi else "large")


def late_share(m_late: Optional[float], m_total: Optional[float], t_total: Optional[float]) -> Optional[float]:
    """mean[2,20] / mean[0,20], only when [0,20] has |t| >= 2 and the same sign."""
    if m_late is None or m_total is None or t_total is None or abs(t_total) < 2 or m_total == 0:
        return None
    if (m_late > 0) != (m_total > 0):
        return None
    return m_late / m_total


def cell_verdict(stat: dict, halves: tuple, spread: Optional[float], t_bar: float) -> str:
    """Pre-registered bar for one (type, bucket) cell on excess CAR[2,20].
    -> 'candidate' | 'positive_unconfirmed' | 'fail'."""
    m, t = stat.get("mean"), stat.get("t")
    if m is None or t is None or abs(m) < MIN_MOVE or abs(t) < t_bar:
        return "fail"
    for h in halves:
        if not h or h.get("mean") is None or h.get("t") is None:
            return "fail"
        if (h["mean"] > 0) != (m > 0) or abs(h["t"]) < HALF_T:
            return "fail"
    if spread is None or abs(m) <= spread:
        return "fail"
    return "candidate" if m < 0 else "positive_unconfirmed"


def type_verdict(cell_verdicts: list, pooled_01_t: Optional[float], t_bar: float = T_BAR) -> str:
    if "candidate" in cell_verdicts:
        return "CANDIDATE SIGNAL (gate 2)"
    if "positive_unconfirmed" in cell_verdicts:
        return "POSITIVE ON SURVIVORS — UNCONFIRMED"
    if pooled_01_t is not None and abs(pooled_01_t) >= t_bar:
        return "PRICED IN FAST / NO DRIFT"
    return "NO MEASURABLE REACTION"


_SEC_DMY_RE = re.compile(r"^(\d{2})-([A-Za-z]{3})-(\d{4})$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}


def parse_sec_date(s: str) -> Optional[str]:
    """'31-MAR-2016' (insider data sets) or '2016-03-31' -> ISO; None if
    the string is neither shape or not a real calendar date."""
    s = (s or "").strip()
    m = _SEC_DMY_RE.match(s)
    if m and m.group(2).upper() in _MONTHS:
        y, mo, d = int(m.group(3)), _MONTHS[m.group(2).upper()], int(m.group(1))
    else:
        m = _ISO_DATE_RE.match(s)
        if not m:
            return None
        y, mo, d = (int(x) for x in m.groups())
    try:
        return date(y, mo, d).isoformat()
    except ValueError:  # e.g. 31-FEB: not a real date -> row dropped
        return None


# ════════════════════════════════════════════════════════════════════════════
# I/O stages (network; cached under --cache)
# ════════════════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _jdump(path: str, obj) -> None:
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _jload(path: str):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def stage_universe(cache: str, client) -> dict:
    path = os.path.join(cache, "universe.json.gz")
    if os.path.exists(path):
        return _jload(path)
    d = client.get_json(TICKERS_EXCHANGE_URL)
    uni = {}
    for cik, _name, ticker, exch in d["data"]:
        if exch in EXCHANGES and ticker:
            uni.setdefault(f"{int(cik):010d}", ticker)
    _jdump(path, uni)
    return uni


def _quarters(start=(2015, 4)) -> list:
    today = date.today()
    out, (y, q) = [], start
    while date(y, 3 * q - 2, 1) <= today:
        out.append((y, q))
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def _form345_zip(cache: str, client, y: int, q: int) -> Optional[str]:
    path = os.path.join(cache, "form345", f"{y}q{q}.zip")
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import sec_filings_text as sft
    for url in FORM345_URLS:
        try:
            raw = client.get(url.format(y=y, q=q))
        except sft.EdgarHTTPError as e:
            log(f"form345 {y}q{q}: {url.split('/data/')[1]} -> {e}")
            continue
        with open(path + ".tmp", "wb") as f:
            f.write(raw)
        os.replace(path + ".tmp", path)
        return path
    return None


def _tsv(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as fh:
        rdr = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"),
                             delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in rdr:
            yield row


def stage_purchases(cache: str, client, universe: dict) -> list:
    """Officer/director open-market purchases for universe issuers, streamed
    one quarter at a time."""
    path = os.path.join(cache, "purchases.json.gz")
    if os.path.exists(path):
        return _jload(path)
    out, missing = [], []
    for (y, q) in _quarters():
        zp = _form345_zip(cache, client, y, q)
        if not zp:
            missing.append(f"{y}q{q}")
            continue
        with zipfile.ZipFile(zp) as zf:
            subs = {}
            for r in _tsv(zf, "SUBMISSION.tsv"):
                cik = (r.get("ISSUERCIK") or "").strip()
                if r.get("DOCUMENT_TYPE") == "4" and cik.isdigit() and f"{int(cik):010d}" in universe:
                    fd = parse_sec_date(r.get("FILING_DATE"))
                    if fd:
                        subs[r["ACCESSION_NUMBER"]] = (f"{int(cik):010d}", fd)
            owners = defaultdict(set)
            for r in _tsv(zf, "REPORTINGOWNER.tsv"):
                acc = r.get("ACCESSION_NUMBER")
                rel = r.get("RPTOWNER_RELATIONSHIP") or ""
                if acc in subs and ("Director" in rel or "Officer" in rel):
                    owners[acc].add(r.get("RPTOWNERCIK"))
            n0 = len(out)
            for r in _tsv(zf, "NONDERIV_TRANS.tsv"):
                acc = r.get("ACCESSION_NUMBER")
                if acc not in owners or r.get("TRANS_CODE") != "P" or r.get("TRANS_ACQUIRED_DISP_CD") != "A":
                    continue
                td = parse_sec_date(r.get("TRANS_DATE"))
                if not td:
                    continue
                issuer, fd = subs[acc]
                for o in owners[acc]:
                    out.append({"issuer": issuer, "owner": o, "trans_date": td,
                                "filing_date": fd, "accession": acc})
        log(f"form345 {y}q{q}: {len(out) - n0} purchase rows")
    # dedupe identical (owner, accession, trans_date) rows
    seen, uniq = set(), []
    for p in out:
        k = (p["owner"], p["accession"], p["trans_date"])
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    _jdump(path, uniq)
    if missing:
        log(f"form345 quarters unavailable: {missing}")
    return uniq


def _submissions_summary(client, cik: str, need_f4: set) -> dict:
    import sec_filings_text as sft
    base = client.get_json(sft.SUBMISSIONS_URL.format(cik10=cik))
    blocks = [base["filings"]["recent"]]
    for fmeta in base["filings"].get("files", []):
        if fmeta.get("filingTo", "9999") >= "2015-10-01":
            blocks.append(client.get_json(sft.SUBMISSIONS_PAGE_URL.format(name=fmeta["name"])))
    has_10kq, eightk, f4 = False, [], {}
    for b in blocks:
        n = len(b.get("form", []))
        for i in range(n):
            form, fd = b["form"][i], b["filingDate"][i]
            if fd < "2015-01-01":
                continue
            if form in ("10-K", "10-Q"):
                has_10kq = True
            elif form == "8-K" and fd >= "2015-12-01":
                items = parse_items(b["items"][i])
                hit = sorted(items & {"1.01", "4.02", "5.02"})
                if hit:
                    eightk.append({"acc": b["accessionNumber"][i], "fd": fd,
                                   "acc_utc": b["acceptanceDateTime"][i], "items": hit,
                                   "doc": b["primaryDocument"][i]})
            elif form == "4" and b["accessionNumber"][i] in need_f4:
                f4[b["accessionNumber"][i]] = b["acceptanceDateTime"][i]
    return {"has_10kq": has_10kq, "eightk": eightk, "f4": f4}


def stage_submissions(cache: str, client, universe: dict, need_f4_by_cik: dict, workers: int) -> dict:
    path = os.path.join(cache, "submissions.json.gz")
    if os.path.exists(path):
        return _jload(path)
    res, errors = {}, []
    ciks = sorted(universe)

    def work(cik):
        try:
            return cik, _submissions_summary(client, cik, need_f4_by_cik.get(cik, set())), None
        except Exception as e:  # recorded + reported; a missing CIK drops from the sample
            return cik, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(workers) as ex:
        for k, (cik, summ, err) in enumerate(ex.map(work, ciks)):
            if err:
                errors.append((cik, err))
            else:
                res[cik] = summ
            if k % 500 == 0:
                log(f"submissions {k}/{len(ciks)} (errors {len(errors)})")
    out = {"by_cik": res, "errors": errors}
    _jdump(path, out)
    return out


def stage_502_text(cache: str, client, filings: list, workers: int) -> dict:
    """acc -> bool (sudden CEO/CFO departure) for the (possibly subsampled)
    5.02 filings. Classification results cached, raw text is not."""
    import sec_filings_text as sft
    path = os.path.join(cache, "e4_class.json.gz")
    done = _jload(path) if os.path.exists(path) else {}
    if len(filings) > MAX_502_DOCS:
        filings = random.Random(SEED).sample(sorted(filings, key=lambda f: f["acc"]), MAX_502_DOCS)
    todo = [f for f in filings if f["acc"] not in done]
    log(f"5.02 docs: {len(filings)} selected, {len(todo)} to fetch")

    def work(f):
        url = sft.ARCHIVES_DOC_URL.format(cik=int(f["cik"]), acc_nodash=f["acc"].replace("-", ""), doc=f["doc"])
        try:
            lines, _ = sft.html_to_text(client.get(url))
            return f["acc"], classify_departure("\n".join(lines))
        except Exception as e:  # unfetchable doc -> excluded, counted in the report
            return f["acc"], f"error: {type(e).__name__}"

    with ThreadPoolExecutor(workers) as ex:
        for k, (acc, v) in enumerate(ex.map(work, todo)):
            done[acc] = v
            if k % 1000 == 0:
                log(f"5.02 {k}/{len(todo)}")
                _jdump(path, done)
    _jdump(path, done)
    return {"selected": [f["acc"] for f in filings], "class": done}


def _price_path(cache: str, ticker: str) -> str:
    return os.path.join(cache, "prices", f"{ticker}.json.gz")


def stage_prices(cache: str, tickers: list, workers: int) -> dict:
    import backtest_v2 as bt
    os.makedirs(os.path.join(cache, "prices"), exist_ok=True)
    days = (date.today() - date(2014, 6, 1)).days
    errors = {}

    def work(tk):
        p = _price_path(cache, tk)
        if os.path.exists(p):
            return tk, None
        try:
            bars = bt._yahoo_bars(tk, days)
        except Exception as e:  # recorded; ticker drops from sample
            return tk, f"{type(e).__name__}: {e}"[:200]
        _jdump(p, {k: bars[k] for k in ("date", "close", "high", "low", "volume")})
        return tk, None

    with ThreadPoolExecutor(workers) as ex:
        for k, (tk, err) in enumerate(ex.map(work, tickers)):
            if err:
                errors[tk] = err
            if k % 500 == 0:
                log(f"prices {k}/{len(tickers)} (errors {len(errors)})")
    return errors


# ════════════════════════════════════════════════════════════════════════════
# Measurement
# ════════════════════════════════════════════════════════════════════════════

def align(days: list, bars: dict) -> dict:
    pos = {d: i for i, d in enumerate(days)}
    n = len(days)
    out = {k: [None] * n for k in ("close", "high", "low", "volume")}
    for j, d in enumerate(bars["date"]):
        i = pos.get(d)
        if i is None:
            continue
        for k in out:
            out[k][i] = bars[k][j]
    return out


def measure_ticker(ticker: str, days: list, spy_ret: list, spy_bull: list, bars: dict,
                   events_by_type: dict) -> tuple:
    """-> (event records, {year: median dollar volume})."""
    a = align(days, bars)
    close = a["close"]
    ret = simple_returns(close)
    dv = [(c * v) if (c is not None and v is not None) else None for c, v in zip(close, a["volume"])]
    yearly = defaultdict(list)
    for d, x in zip(days, dv):
        if x is not None:
            yearly[d[:4]].append(x)
    year_med = {y: statistics.median(v) for y, v in yearly.items() if len(v) >= 200}
    first_valid = next((i for i, c in enumerate(close) if c is not None), None)
    ci = CarIndex(ret, spy_ret)
    recs = []
    for etype, idx_list in events_by_type.items():
        kept = dedup_day0(idx_list)
        all_idx = sorted(set(idx_list))
        for i0 in kept:
            if first_valid is None or i0 - first_valid < MIN_HISTORY or close[i0 - 1] is None \
                    or close[i0 - 1] < MIN_PRICE or days[i0] < SAMPLE_START:
                continue
            cars = {w: ci.car(i0, lo, hi) for w, (lo, hi) in WINDOWS.items()}
            if cars["0_20"] is None:
                continue
            rng = event_rng(ticker, i0, etype)
            pool = [j for j in control_pool(i0, len(days), all_idx)
                    if ci.car(j, 0, 20) is not None]
            ctrl = rng.sample(pool, min(N_CONTROLS, len(pool)))
            if not ctrl:
                continue
            excess = {}
            for w, (lo, hi) in WINDOWS.items():
                if cars[w] is None:
                    continue
                cv = [x for x in (ci.car(j, lo, hi) for j in ctrl) if x is not None]
                if cv:
                    excess[w] = cars[w] - sum(cv) / len(cv)
            s0, s1 = i0 + SIZE_LOOKBACK[0], i0 + SIZE_LOOKBACK[1]
            dvw = [x for x in dv[max(0, s0):s1 + 1] if x is not None]
            p0, p1 = i0 + SPREAD_LOOKBACK[0], i0 + SPREAD_LOOKBACK[1]
            recs.append({
                "type": etype, "ticker": ticker, "day0": days[i0],
                "regime": "bull" if spy_bull[i0 - 1] else ("bear" if spy_bull[i0 - 1] is False else None),
                "medvol": statistics.median(dvw) if len(dvw) >= 60 else None,
                "spread": abdi_ranaldo(a["high"][max(0, p0):p1 + 1], a["low"][max(0, p0):p1 + 1],
                                       close[max(0, p0):p1 + 1]),
                "car": cars, "excess": excess, "n_ctrl": len(ctrl),
            })
    return recs, year_med


def spy_series(days: list, spy: dict) -> tuple:
    a = align(days, spy)
    c = a["close"]
    ret = simple_returns(c)
    bull = [None] * len(days)
    for i in range(199, len(days)):
        w = c[i - 199:i + 1]
        if all(x is not None for x in w):
            bull[i] = c[i] > sum(w) / 200
    return ret, bull


def summarize(recs: list, breaks_by_year: dict) -> dict:
    """Tables per pre-registration §1.4/§1.5."""
    for r in recs:
        prev = str(int(r["day0"][:4]) - 1)
        r["bucket"] = size_bucket(r["medvol"], breaks_by_year.get(prev))
    out = {}
    for etype in EVENT_TYPES:
        rs = [r for r in recs if r["type"] == etype]
        t_bar = T_BAR_E1 if etype.startswith("E1") else T_BAR
        bounds = {}
        for w in WINDOWS:
            vals = [r["excess"][w] for r in rs if w in r["excess"]]
            if len(vals) >= 20:
                bounds[w] = (percentile(vals, WINSOR[0]), percentile(vals, WINSOR[1]))

        def stat(subset, w, field="excess"):
            sub = [r for r in subset if w in r[field] and r[field][w] is not None]
            if not sub:
                return {"n": 0}
            raw = [r[field][w] for r in sub]
            vals = winsorize(raw, bounds=bounds[w]) if (field == "excess" and w in bounds) else raw
            st = clustered_mean_t(vals, [week_cluster(r["day0"]) for r in sub])
            st["median"] = statistics.median(raw)
            st["hit"] = sum(1 for x in raw if x > 0) / len(raw)
            return st

        groups = {"pooled": rs}
        for b in ("small", "mid", "large"):
            groups[b] = [r for r in rs if r.get("bucket") == b]
        tbl = {}
        for g, sub in groups.items():
            row = {w: stat(sub, w) for w in WINDOWS}
            row["raw"] = {w: stat(sub, w, "car") for w in WINDOWS}
            row["halves"] = {
                "2016_2020": stat([r for r in sub if int(r["day0"][:4]) <= HALF_SPLIT_YEAR], PRIMARY),
                "2021_on": stat([r for r in sub if int(r["day0"][:4]) > HALF_SPLIT_YEAR], PRIMARY)}
            row["regime"] = {rg: {w: stat([r for r in sub if r["regime"] == rg], w) for w in ("0_1", PRIMARY)}
                             for rg in ("bull", "bear")}
            sp = [r["spread"] for r in sub if r["spread"] is not None]
            row["spread_median"] = statistics.median(sp) if sp else None
            row["late_share"] = late_share(row[PRIMARY].get("mean"), row["0_20"].get("mean"), row["0_20"].get("t"))
            if g != "pooled":
                row["verdict"] = cell_verdict(row[PRIMARY], (row["halves"]["2016_2020"], row["halves"]["2021_on"]),
                                              row["spread_median"], t_bar)
            tbl[g] = row
        out[etype] = {"n": len(rs), "t_bar": t_bar, "table": tbl,
                      "verdict": type_verdict([tbl[b]["verdict"] for b in ("small", "mid", "large")],
                                              tbl["pooled"]["0_1"].get("t"))}
    return out


def run(cache: str, workers: int, fetch_only: bool = False) -> Optional[dict]:
    import sec_filings_text as sft
    os.makedirs(cache, exist_ok=True)
    client = sft.EdgarClient()
    universe = stage_universe(cache, client)
    log(f"universe: {len(universe)} CIKs")
    purchases = stage_purchases(cache, client, universe)
    log(f"purchases: {len(purchases)} rows")
    clusters = insider_clusters(purchases)
    log(f"E1 clusters (pre-dedup, all years): {len(clusters)}")
    need_f4 = defaultdict(set)
    for c in clusters:
        need_f4[c["issuer"]].add(c["accession"])
    subs = stage_submissions(cache, client, universe, need_f4, workers)
    by_cik = subs["by_cik"]
    op = {cik for cik, s in by_cik.items() if s["has_10kq"]}
    log(f"submissions ok {len(by_cik)}, errors {len(subs['errors'])}, operating {len(op)}")

    raw_events = []  # (cik, type, accept_utc or None, filing_date)
    f4_acc_hits = 0
    for c in clusters:
        if c["issuer"] not in op:
            continue
        acc_utc = by_cik[c["issuer"]]["f4"].get(c["accession"])
        f4_acc_hits += acc_utc is not None
        raw_events.append((c["issuer"], EVENT_TYPES[0], acc_utc, c["filing_date"]))
    f502 = []
    for cik in op:
        for f in by_cik[cik]["eightk"]:
            if "1.01" in f["items"]:
                raw_events.append((cik, EVENT_TYPES[1], f["acc_utc"], f["fd"]))
            if "4.02" in f["items"]:
                raw_events.append((cik, EVENT_TYPES[2], f["acc_utc"], f["fd"]))
            if "5.02" in f["items"]:
                f502.append({"cik": cik, **f})
    e4 = stage_502_text(cache, client, f502, workers)
    e4_err = sum(1 for v in e4["class"].values() if isinstance(v, str))
    for f in f502:
        if e4["class"].get(f["acc"]) is True:
            raw_events.append((f["cik"], EVENT_TYPES[3], f["acc_utc"], f["fd"]))

    tickers = sorted({universe[c] for c in op} | {"SPY"})
    price_err = stage_prices(cache, tickers, max(2, workers // 2))
    if price_err:
        log(f"price errors: {len(price_err)} (e.g. {sorted(price_err)[:5]})")
    if fetch_only:
        log("fetch-only: downloads cached; measurement skipped")
        return None
    spy = _jload(_price_path(cache, "SPY"))
    days = spy["date"]
    spy_ret, spy_bull = spy_series(days, spy)

    ev_by_ticker = defaultdict(lambda: defaultdict(list))
    for cik, et, acc_utc, fd in raw_events:
        i0 = event_day0(days, acc_utc, fd)
        if i0 is not None:
            ev_by_ticker[universe[cik]][et].append(i0)

    recs, yearly = [], defaultdict(list)
    for k, tk in enumerate(tickers):
        p = _price_path(cache, tk)
        if tk == "SPY" or not os.path.exists(p):
            continue
        r, ym = measure_ticker(tk, days, spy_ret, spy_bull, _jload(p), ev_by_ticker.get(tk, {}))
        recs.extend(r)
        for y, v in ym.items():
            yearly[y].append(v)
        if k % 1000 == 0:
            log(f"measure {k}/{len(tickers)} recs {len(recs)}")
    breaks = {y: (percentile(v, 1 / 3), percentile(v, 2 / 3)) for y, v in yearly.items() if len(v) >= 30}
    summary = summarize(recs, breaks)
    meta = {
        "universe_ciks": len(universe), "submissions_ok": len(by_cik),
        "submissions_errors": len(subs["errors"]), "operating_ciks": len(op),
        "tickers_priced": sum(1 for t in tickers if t != "SPY" and os.path.exists(_price_path(cache, t))),
        "price_errors_this_pass": len(price_err),
        "purchase_rows": len(purchases), "e1_clusters_all": len(clusters),
        "e1_acceptance_found": f4_acc_hits,
        "e4_502_filings": len(f502), "e4_docs_selected": len(e4["selected"]),
        "e4_doc_errors": e4_err,
        "e4_positive": sum(1 for a in e4["selected"] if e4["class"].get(a) is True),
        "raw_events_by_type": {t: sum(1 for e in raw_events if e[1] == t) for t in EVENT_TYPES},
        "measured_by_type": {t: sum(1 for r in recs if r["type"] == t) for t in EVENT_TYPES},
        "day0_range": [min(r["day0"] for r in recs), max(r["day0"] for r in recs)] if recs else None,
        "size_breaks": breaks, "spy_days": [days[0], days[-1]],
    }
    result = {"meta": meta, "summary": summary}
    with open(os.path.join(cache, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    _jdump(os.path.join(cache, "records.json.gz"), recs)
    return result


def _fmt(st: dict, pct=True) -> str:
    if not st or st.get("n", 0) == 0 or st.get("mean") is None:
        return "—"
    t = f"{st['t']:+.2f}" if st.get("t") is not None else "n/a"
    return f"{100 * st['mean']:+.2f}% (t {t})"


def report(result: dict) -> str:
    lines = []
    s = result["summary"]
    lines.append("| Event | Bucket | n | CAR[0,1] | CAR[2,5] | CAR[2,20] | CAR[2,60] | median[2,20] | hit[2,20] | late share | spread | cell |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for et in EVENT_TYPES:
        for g in ("pooled", "small", "mid", "large"):
            row = s[et]["table"][g]
            p = row[PRIMARY]
            ls = f"{100 * row['late_share']:.0f}%" if row.get("late_share") is not None else "n/a"
            sp = f"{100 * row['spread_median']:.2f}%" if row.get("spread_median") is not None else "—"
            med = f"{100 * p['median']:+.2f}%" if p.get("n") else "—"
            hit = f"{100 * p['hit']:.0f}%" if p.get("n") else "—"
            lines.append(f"| {et} | {g} | {p.get('n', 0)} | {_fmt(row['0_1'])} | {_fmt(row['2_5'])} | "
                         f"{_fmt(p)} | {_fmt(row['2_60'])} | {med} | {hit} | {ls} | {sp} | {row.get('verdict', '')} |")
    lines.append("")
    lines.append("| Event | Verdict | t bar |")
    lines.append("|---|---|---|")
    for et in EVENT_TYPES:
        lines.append(f"| {et} | {s[et]['verdict']} | {s[et]['t_bar']} |")
    lines.append("")
    lines.append("Raw SPY-adjusted CAR (no same-ticker control), pooled / small:")
    lines.append("")
    lines.append("| Event | Group | raw CAR[0,1] | raw CAR[2,20] | raw CAR[2,60] |")
    lines.append("|---|---|---|---|---|")
    for et in EVENT_TYPES:
        for g in ("pooled", "small"):
            rw = s[et]["table"][g]["raw"]
            lines.append(f"| {et} | {g} | {_fmt(rw['0_1'])} | {_fmt(rw['2_20'])} | {_fmt(rw['2_60'])} |")
    lines.append("")
    lines.append("Halves and regime (excess CAR[2,20]; regime also [0,1]):")
    lines.append("")
    lines.append("| Event | Group | 2016-2020 | 2021+ | bull [0,1] | bull [2,20] | bear [0,1] | bear [2,20] |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for et in EVENT_TYPES:
        for g in ("pooled", "small", "mid", "large"):
            row = s[et]["table"][g]
            h, rg = row["halves"], row["regime"]
            lines.append(f"| {et} | {g} | {_fmt(h['2016_2020'])} | {_fmt(h['2021_on'])} | "
                         f"{_fmt(rg['bull']['0_1'])} | {_fmt(rg['bull'][PRIMARY])} | "
                         f"{_fmt(rg['bear']['0_1'])} | {_fmt(rg['bear'][PRIMARY])} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered event-reaction study")
    ap.add_argument("cmd", choices=("fetch", "run", "report"))
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "event_reaction_cache"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        run(a.cache, a.workers, fetch_only=True)
        return 0
    if a.cmd == "run":
        res = run(a.cache, a.workers)
    else:
        with open(os.path.join(a.cache, "result.json")) as f:
            res = json.load(f)
    print(json.dumps(res["meta"], indent=1, default=str))
    print(report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
