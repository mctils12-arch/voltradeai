"""
sec_filings_text.py — EDGAR 10-K / 10-Q full-text archive + section
extractor (Lazy Prices pipeline, PR 1 of N: DATA only).

Cohen, Malloy & Nguyen, "Lazy Prices" (J. Finance 2020): firms that CHANGE
the language of their periodic reports year-over-year subsequently
underperform, and the market is slow to notice. Testing that needs, per
company, the raw 10-K/10-Q text, the specific sections that carry the
effect (Risk Factors, Legal Proceedings, MD&A), the honest SIGNAL
timestamp (EDGAR acceptance time — not the filing date, not the period),
and a same-type prior-year pairing. This module builds exactly that and
nothing more: NO similarity scores (PR 2), NO prices, NO trading logic.

SPINOUT boundary (CLAUDE.md STANDING BEHAVIORS): this module imports
nothing from bot_engine / system_config / strategies / server. It is
ROOT-LEVEL Python on purpose — scripts/ is not in the Docker image
(KNOWN BROKEN #44), so a pipeline module there could never run on Railway.

LADDER GATE 1 (DATA): `python3 sec_filings_text.py sample --n 50 ...`
prints a DATA-GATE report (per-form per-section found rate, length
distribution, ToC-duplicate fallbacks, no-material-changes rate, pairing
coverage, acceptance-time verification, errors).

ARCHIVE (data is the moat — raw bytes are never deleted):
  <DATA_DIR>/datacore_archive/filings_10kq/raw/<cik10>/<accession>.<ext>.gz
      primary document, as served by EDGAR, gzip — WRITE-ONCE, atomic.
  <DATA_DIR>/datacore_archive/filings_10kq/raw/<cik10>/<accession>.index-headers.html.gz
      EDGAR's index-headers page (carries ACCEPTANCE-DATETIME) — WRITE-ONCE.
  <DATA_DIR>/datacore_archive/filings_10kq/sections/<cik10>/<accession>.json.gz
      extracted sections — a DERIVED, recomputable view over raw/; rewritten
      only when EXTRACTION_VERSION changes.
DATA_DIR resolves exactly like server/datacoreArchive.ts archiveBaseDir():
env DATA_DIR first, else storage_config.DATA_DIR (/data/voltrade or /tmp).

ACCEPTANCE TIME — VERIFIED FINDING (2026-10-16): the data.sec.gov
submissions JSON's `acceptanceDateTime` carries a "Z" suffix but is NOT
reliably UTC. Against EDGAR's own index-headers ACCEPTANCE-DATETIME (US
Eastern, the documented EDGAR clock) it is INCONSISTENTLY either true UTC or
true UTC + the ET offset (+4h EDT / +5h EST): 454 vs 263 of 717 filings in
the 2026-10-16 50-company gate run, mixed within the same filer agent, the
same company, every year 2023-2026, and both before/after 17:00 ET — no
usable rule, so the JSON value cannot be corrected, only replaced. By hand:
AAPL 8-K 0000320193-26-000018 header 2026-07-30 16:30:28 ET (= 20:30:28Z),
JSON 2026-07-31T00:30:28Z (+4h); AAPL 10-Q 0000320193-26-000006 header
2026-01-30 06:01:32 EST (= 11:01:32Z), JSON 16:01:32Z (+5h); MSFT 10-K
0001193125-26-323660 header 16:08:01 ET, JSON next-day 00:08:01Z (+4h); but
GE 10-Q 0000040545-24-000176 header 06:28:10 EDT, JSON 10:28:10Z (exact).
Taking the JSON at face value can move a 06:01 ET pre-market filing to
"10:01 ET", after the open — a session misassignment, not a rounding error.
So `accepted_at` (UTC) comes from the index-headers page (source
"index_headers"). Only if that page is unavailable is the raw JSON value
used as-is (source "submissions_json_unverified"): it is never EARLIER than
the truth in either observed mode, so the fallback can delay a signal but
never introduce lookahead. The raw JSON value is always kept verbatim in
`acceptance_datetime_raw`, and the gate report classifies every filing's
JSON mode (classify_json_acceptance) so the finding is re-measured on every
run rather than trusted forever.
"""
from __future__ import annotations

import argparse
import email.utils
import gzip
import html as html_lib
import http.client
import io
import json
import os
import random
import re
import socket
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Callable, Optional

# Same declared User-Agent as server/edgarForm4.ts / server/sec8kEarnings.ts —
# SEC's fair-access policy keys on it, so every datacore EDGAR client sends one.
SEC_UA = "voltradeai-datacore/1.0 (research@voltradeai.com)"

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVES_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
INDEX_HEADERS_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{acc}-index-headers.html"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# SEC fair access: max 10 requests/second per host/user. We default under it.
DEFAULT_RATE_PER_SEC = 8.0
HARD_CAP_PER_SEC = 10
MAX_RETRIES = 4
BACKOFF_BASE_SEC = 1.0
BACKOFF_MAX_SEC = 60.0
DEFAULT_TIMEOUT_SEC = 30.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

STREAM = "filings_10kq"
EXTRACTION_VERSION = "lp-sections-1.0.0"
SECTIONS_SCHEMA_VERSION = 1

DEFAULT_FORMS = ("10-K", "10-Q")
AMENDMENT_SUFFIX = "/A"


# ════════════════════════════════════════════════════════════════════════════
# Rate limiter — token bucket + hard sliding-window cap, process-wide.
# ════════════════════════════════════════════════════════════════════════════

_EPS = 1e-9  # float-absorption guard: a sub-nanosecond wait would never advance the clock


class RateLimiter:
    """Token bucket at `rate` req/s (burst 1 => evenly spaced), PLUS a hard
    guard: never more than `hard_cap` acquisitions inside any trailing 1.0s
    window. The bucket alone cannot promise that (a bucket of capacity B
    admits B + r*T in a window T), so the window guard is what makes the
    10 req/s SEC limit an invariant rather than an average. The configured
    rate is clamped to the cap. Thread-safe: the lock is held while waiting,
    which serializes waiters — exactly the behaviour a process-wide limiter
    wants. Clock and sleep are injectable so tests can prove the invariant
    on a fake clock."""

    def __init__(self, rate: float = DEFAULT_RATE_PER_SEC, hard_cap: int = HARD_CAP_PER_SEC,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.hard_cap = int(min(hard_cap, HARD_CAP_PER_SEC))
        self.rate = float(min(rate, self.hard_cap))
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens = 1.0
        self._last = clock()
        self._recent: deque = deque()

    def set_rate(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        with self._lock:
            self.rate = float(min(rate, self.hard_cap))

    def acquire(self) -> float:
        """Blocks until a request may be sent; returns the acquisition time."""
        with self._lock:
            while True:
                now = self._clock()
                self._tokens = min(1.0, self._tokens + (now - self._last) * self.rate)
                self._last = now
                while self._recent and now - self._recent[0] >= 1.0 - _EPS:
                    self._recent.popleft()
                wait_bucket = 0.0 if self._tokens >= 1.0 - _EPS else (1.0 - self._tokens) / self.rate
                wait_window = 0.0
                if len(self._recent) >= self.hard_cap:
                    wait_window = 1.0 - (now - self._recent[0])
                wait = max(wait_bucket, wait_window)
                if wait <= _EPS:
                    self._tokens -= 1.0
                    self._recent.append(now)
                    return now
                self._sleep(wait)


_PROCESS_LIMITER: Optional[RateLimiter] = None
_PROCESS_LIMITER_LOCK = threading.Lock()


def process_limiter() -> RateLimiter:
    """The ONE limiter every default EdgarClient in this process shares, so
    concurrent workers cannot jointly exceed the cap."""
    global _PROCESS_LIMITER
    with _PROCESS_LIMITER_LOCK:
        if _PROCESS_LIMITER is None:
            _PROCESS_LIMITER = RateLimiter()
        return _PROCESS_LIMITER


# ════════════════════════════════════════════════════════════════════════════
# HTTP client with retries.
# ════════════════════════════════════════════════════════════════════════════

class HttpResponse:
    __slots__ = ("status", "headers", "body")

    def __init__(self, status: int, headers: Optional[dict] = None, body: bytes = b""):
        self.status = int(status)
        self.headers = {str(k).lower(): v for k, v in (headers or {}).items()}
        self.body = body


class EdgarHTTPError(RuntimeError):
    def __init__(self, url: str, status: Optional[int], detail: str = ""):
        super().__init__(f"GET {url} -> {status if status is not None else 'error'} {detail}".strip())
        self.url = url
        self.status = status


# Exceptions treated as transient (timeouts, dropped connections).
TRANSIENT_EXCEPTIONS = (TimeoutError, socket.timeout, ConnectionError, urllib.error.URLError,
                        http.client.IncompleteRead, http.client.RemoteDisconnected)


def default_http_get(url: str, headers: dict, timeout: float) -> HttpResponse:
    """urllib GET returning an HttpResponse for ANY status (HTTPError is a
    response, not an exception, so the retry layer sees 429/5xx codes).
    Transparently gunzips Content-Encoding: gzip."""
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            hdrs = dict(r.headers.items())
            status = r.status
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, "read") else b""
        hdrs = dict(e.headers.items()) if e.headers else {}
        status = e.code
    if str({k.lower(): v for k, v in hdrs.items()}.get("content-encoding", "")).lower() == "gzip":
        body = gzip.decompress(body)
    return HttpResponse(status, hdrs, body)


def parse_retry_after(value, now: Optional[datetime] = None) -> Optional[float]:
    """Retry-After is either delta-seconds or an HTTP-date (RFC 9110). Returns
    seconds to wait (>= 0) or None if absent/unparseable."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return max(0.0, float(s))
    try:
        dt = email.utils.parsedate_to_datetime(s)
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (dt - now).total_seconds())


class EdgarClient:
    """GETs against SEC hosts: declared UA, shared rate limiter, retries on
    429/5xx/timeouts with exponential backoff (base * 2**attempt, capped)
    that honors Retry-After when the server sends a longer one. 404 and
    other 4xx are NOT retried (retrying a permanent error just burns the
    rate budget)."""

    def __init__(self, http_get: Optional[Callable[[str, dict, float], HttpResponse]] = None,
                 limiter: Optional[RateLimiter] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 max_retries: int = MAX_RETRIES, backoff_base: float = BACKOFF_BASE_SEC,
                 timeout: float = DEFAULT_TIMEOUT_SEC):
        self.http_get = http_get or default_http_get
        self.limiter = limiter or process_limiter()
        self.sleep = sleep
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.timeout = float(timeout)
        self.stats = {"requests": 0, "retries": 0, "errors": 0}
        self._stats_lock = threading.Lock()

    def _bump(self, key: str) -> None:
        with self._stats_lock:
            self.stats[key] += 1

    def backoff_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        delay = min(BACKOFF_MAX_SEC, self.backoff_base * (2 ** attempt))
        if retry_after is not None:
            delay = max(delay, min(retry_after, BACKOFF_MAX_SEC * 5))
        return delay

    def get(self, url: str) -> bytes:
        headers = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip"}
        last = "no attempt"
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            self._bump("requests")
            retry_after = None
            try:
                resp = self.http_get(url, headers, self.timeout)
            except TRANSIENT_EXCEPTIONS as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if 200 <= resp.status < 300:
                    return resp.body
                if resp.status not in RETRYABLE_STATUS:
                    self._bump("errors")
                    raise EdgarHTTPError(url, resp.status)
                last = f"HTTP {resp.status}"
                retry_after = parse_retry_after(resp.headers.get("retry-after"))
            if attempt < self.max_retries:
                self._bump("retries")
                self.sleep(self.backoff_delay(attempt, retry_after))
        self._bump("errors")
        raise EdgarHTTPError(url, None, f"failed after {self.max_retries + 1} attempts ({last})")

    def get_json(self, url: str):
        return json.loads(self.get(url))


# ════════════════════════════════════════════════════════════════════════════
# Filing listing (submissions JSON incl. every filings.files page).
# ════════════════════════════════════════════════════════════════════════════

def cik10(cik) -> str:
    return f"{int(str(cik).strip()):010d}"


def is_amendment(form: str) -> bool:
    return str(form or "").upper().endswith(AMENDMENT_SUFFIX)


def base_form(form: str) -> str:
    f = str(form or "").upper()
    return f[: -len(AMENDMENT_SUFFIX)] if f.endswith(AMENDMENT_SUFFIX) else f


def parse_submission_columns(cols: dict, cik: str, forms, since: str) -> list:
    """Columnar submissions block (filings.recent or a files page) -> rows.
    Keeps forms whose BASE form is in `forms` (so 10-K/A is listed when 10-K
    is requested) and filingDate >= since."""
    want = {base_form(f) for f in forms}
    acc = cols.get("accessionNumber") or []
    out = []
    for i, a in enumerate(acc):
        def col(name, _i=i):
            v = cols.get(name) or []
            return v[_i] if _i < len(v) else None
        form = col("form") or ""
        if base_form(form) not in want:
            continue
        fdate = col("filingDate") or ""
        if since and fdate < since:
            continue
        out.append({
            "cik": cik10(cik),
            "accession": a,
            "form": form,
            "filingDate": fdate,
            "acceptanceDateTime": col("acceptanceDateTime"),
            "reportDate": col("reportDate") or None,
            "primaryDocument": col("primaryDocument") or None,
            "isAmendment": is_amendment(form),
        })
    return out


def list_filings(cik, forms=DEFAULT_FORMS, since: str = "2010-01-01",
                 client: Optional[EdgarClient] = None) -> list:
    """Every 10-K/10-Q (and their /A amendments, flagged isAmendment) for a
    CIK filed on/after `since`, sorted by (filingDate, accession).

    filings.recent holds only the newest ~1000 filings; older ones live in
    the filings.files pages, which are ALL read except a page whose declared
    filingTo is before `since` (it cannot contain a qualifying filing — the
    skip changes cost, never output; a page without filingTo is read).
    Amendments are listed here; comparison helpers exclude them by default."""
    client = client or EdgarClient()
    data = client.get_json(SUBMISSIONS_URL.format(cik10=cik10(cik)))
    filings = data.get("filings") or {}
    rows = parse_submission_columns(filings.get("recent") or {}, cik, forms, since)
    for page in filings.get("files") or []:
        name = page.get("name")
        if not name:
            continue
        to = page.get("filingTo")
        if since and to and to < since:
            continue
        page_data = client.get_json(SUBMISSIONS_PAGE_URL.format(name=name))
        rows += parse_submission_columns(page_data, cik, forms, since)
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: (r["filingDate"], r["accession"])):
        if r["accession"] in seen:
            continue
        seen.add(r["accession"])
        uniq.append(r)
    return uniq


# ════════════════════════════════════════════════════════════════════════════
# Acceptance time (see module docstring for the verified JSON offset).
# ════════════════════════════════════════════════════════════════════════════

def _nth_sunday(year: int, month: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_sunday(year: int, month: int) -> date:
    d = (date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() + 1) % 7)


def eastern_utc_offset_hours(local: datetime) -> int:
    """Hours to ADD to US-Eastern local wall time to get UTC (4 in EDT, 5 in
    EST). Implemented from the statutory rules (Energy Policy Act 2005 from
    2007; 1987-2006 rule before) so it needs no tzdata in the container."""
    y = local.year
    if y >= 2007:
        start, end = _nth_sunday(y, 3, 2), _nth_sunday(y, 11, 1)
    else:
        start, end = _nth_sunday(y, 4, 1), _last_sunday(y, 10)
    dst_start = datetime(start.year, start.month, start.day, 2)
    dst_end = datetime(end.year, end.month, end.day, 2)
    return 4 if dst_start <= local.replace(tzinfo=None) < dst_end else 5


def _iso_z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


_HDR_ACCEPT_RE = re.compile(r"ACCEPTANCE-DATETIME(?:>|&gt;)\s*(\d{14})")


def parse_index_headers_acceptance(page: str) -> Optional[str]:
    """EDGAR index-headers ACCEPTANCE-DATETIME (YYYYMMDDHHMMSS, US Eastern)
    -> UTC ISO 'Z', or None."""
    m = _HDR_ACCEPT_RE.search(page or "")
    if not m:
        return None
    local = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    return _iso_z(local + timedelta(hours=eastern_utc_offset_hours(local)))


def parse_index_headers_et(page: str) -> Optional[str]:
    m = _HDR_ACCEPT_RE.search(page or "")
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%S")


def json_acceptance_conservative_utc(raw: Optional[str]) -> Optional[str]:
    """Fallback signal timestamp from the submissions JSON: the raw value
    read as UTC. Both observed JSON modes (exact UTC, UTC + ET offset) are
    >= the true acceptance instant, so this can only be late, never early."""
    if not raw:
        return None
    try:
        return _iso_z(datetime.strptime(str(raw)[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def classify_json_acceptance(raw: Optional[str], true_utc: Optional[str]) -> Optional[str]:
    """Which mode a filing's JSON acceptanceDateTime is in, vs the header
    truth: "utc", "utc_plus_et_offset", or "other" (None if not comparable)."""
    j = json_acceptance_conservative_utc(raw)
    if not j or not true_utc:
        return None
    jd = datetime.strptime(j, "%Y-%m-%dT%H:%M:%SZ")
    td = datetime.strptime(true_utc, "%Y-%m-%dT%H:%M:%SZ")
    delta_h = (jd - td).total_seconds() / 3600.0
    if delta_h == 0:
        return "utc"
    if delta_h == eastern_utc_offset_hours(td - timedelta(hours=5)):
        return "utc_plus_et_offset"
    return "other"


# ════════════════════════════════════════════════════════════════════════════
# Archive (write-once raw, atomic).
# ════════════════════════════════════════════════════════════════════════════

def archive_base_dir() -> str:
    """Mirror of server/datacoreArchive.ts archiveBaseDir(): env DATA_DIR wins,
    else storage_config.DATA_DIR, then /datacore_archive."""
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        from storage_config import DATA_DIR as data_dir  # noqa: N811 — not trading logic
    return os.path.join(data_dir, "datacore_archive")


def stream_root(archive: Optional[str] = None) -> str:
    return os.path.join(archive or archive_base_dir(), STREAM)


def doc_ext(primary_document: Optional[str]) -> str:
    ext = os.path.splitext(primary_document or "")[1].lower().lstrip(".")
    return ext if re.fullmatch(r"[a-z0-9]{1,5}", ext or "") else "bin"


def raw_path(archive: str, cik, accession: str, primary_document: Optional[str]) -> str:
    return os.path.join(stream_root(archive), "raw", cik10(cik), f"{accession}.{doc_ext(primary_document)}.gz")


def headers_path(archive: str, cik, accession: str) -> str:
    return os.path.join(stream_root(archive), "raw", cik10(cik), f"{accession}.index-headers.html.gz")


def sections_path(archive: str, cik, accession: str) -> str:
    return os.path.join(stream_root(archive), "sections", cik10(cik), f"{accession}.json.gz")


def _gz_bytes(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(data)
    return buf.getvalue()


def _atomic_write(path: str, payload: bytes, write_once: bool) -> bool:
    """mkstemp in the destination dir + fsync, then publish. write_once uses
    os.link, which fails if the destination exists — so two racing writers
    can never overwrite each other's raw bytes. Returns True if written."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    if write_once and os.path.exists(path):
        return False
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".lp.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if write_once:
            try:
                os.link(tmp, path)
            except FileExistsError:
                return False
        else:
            os.replace(tmp, path)
            tmp = None
        return True
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def write_once_gz(path: str, data: bytes) -> bool:
    """Raw archive write: gzip, atomic, never overwrites. True if written,
    False if the file already existed (the archived bytes win)."""
    return _atomic_write(path, _gz_bytes(data), write_once=True)


def read_gz(path: str) -> bytes:
    with gzip.open(path, "rb") as f:
        return f.read()


def write_sections(path: str, doc: dict) -> bool:
    """Sections are derived from raw/ — rewrite (atomically) only when the
    stored extraction version differs, OR when the stored copy carries a
    fallback acceptance timestamp and the new one is verified from the
    index-headers page (otherwise a filing first processed without its
    headers page would keep the up-to-5h-late JSON time forever). True if
    written."""
    if os.path.exists(path):
        try:
            old = json.loads(read_gz(path))
            same_version = (old.get("extraction") or {}).get("version") == doc["extraction"]["version"]
            upgrades_ts = (old.get("accepted_at_source") != "index_headers"
                           and doc.get("accepted_at_source") == "index_headers")
            if same_version and not upgrades_ts:
                return False
        except (OSError, ValueError, EOFError) as e:
            print(f"[sec_filings_text] unreadable sections file {path} ({e}); rewriting", file=sys.stderr)
    return _atomic_write(path, _gz_bytes(json.dumps(doc, ensure_ascii=False).encode("utf-8")), write_once=False)


# ════════════════════════════════════════════════════════════════════════════
# HTML -> text (pure).
# ════════════════════════════════════════════════════════════════════════════

# Elements whose whole subtree is dropped.
SKIP_TAGS = frozenset({"script", "style", "head", "title", "ix:header", "noscript", "xml"})
BLOCK_TAGS = frozenset({"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                        "table", "ul", "ol", "section", "article", "blockquote", "center",
                        "pre", "hr", "dl", "dt", "dd", "body"})
CELL_TAGS = frozenset({"td", "th"})
VOID_TAGS = frozenset({"br", "hr", "img", "meta", "link", "input", "col", "area", "base", "wbr"})

# TABLE RULE: a table is dropped when digits are more than this share of its
# alphanumeric characters (financial/numeric tables) UNLESS it contains an
# Item/Part heading (layout tables used for headings, and ToC tables, are
# kept so heading detection and ToC-defeat still see them).
TABLE_DIGIT_SHARE_DROP = 0.15
_TABLE_HEADING_RE = re.compile(r"(?:^|\n)\s*(?:items?\s*\d|part\s+i)", re.I)
_DISPLAY_NONE_RE = re.compile(r"display\s*:\s*none", re.I)
_SRC_NEWLINE_RE = re.compile(r"[\r\n]+")


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list = []
        self._skip_tag: Optional[str] = None
        self._skip_depth = 0
        self._table_depth = 0
        self._table_buf: list = []
        self.tables_dropped = 0
        self.tables_kept = 0
        self.hidden_dropped = 0

    def _emit(self, s: str) -> None:
        (self._table_buf if self._table_depth else self.out).append(s)

    def handle_starttag(self, tag, attrs):
        if self._skip_tag is not None:
            if tag == self._skip_tag and tag not in VOID_TAGS:
                self._skip_depth += 1
            return
        style = ""
        for k, v in attrs:
            if k == "style" and v:
                style = v
        if tag in SKIP_TAGS or (style and _DISPLAY_NONE_RE.search(style) and tag not in VOID_TAGS):
            if tag not in SKIP_TAGS:
                self.hidden_dropped += 1
            self._skip_tag, self._skip_depth = tag, 1
            return
        if tag == "table":
            if self._table_depth == 0:
                self._table_buf = []
            self._table_depth += 1
            self._emit("\n")
        elif tag in BLOCK_TAGS:
            self._emit("\n")
        elif tag in CELL_TAGS:
            self._emit(" ")

    def handle_startendtag(self, tag, attrs):
        if self._skip_tag is None and tag in BLOCK_TAGS:
            self._emit("\n")

    def handle_endtag(self, tag):
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skip_tag = None
            return
        if tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0:
                text = "".join(self._table_buf)
                alnum = [c for c in text if c.isalnum()]
                digits = sum(1 for c in alnum if c.isdigit())
                share = digits / len(alnum) if alnum else 0.0
                if share > TABLE_DIGIT_SHARE_DROP and not _TABLE_HEADING_RE.search(text):
                    self.tables_dropped += 1
                else:
                    self.tables_kept += 1
                    self.out.append(text)
                self.out.append("\n")
                self._table_buf = []
        elif tag in BLOCK_TAGS or tag in ("td", "th"):
            self._emit("\n" if tag in BLOCK_TAGS else " ")

    def handle_data(self, data):
        # A raw newline inside an HTML text node is ordinary whitespace, not a
        # line break: filer software hard-wraps source at ~130 columns, and
        # treating that as a break splits "Item\n1A. Risk Factors" into two
        # lines and hides the heading. Only block tags create lines.
        if self._skip_tag is None:
            self._emit(_SRC_NEWLINE_RE.sub(" ", data))


_SGML_DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
_SGML_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.S | re.I)


def _first_sgml_document(raw: str) -> str:
    """EDGAR .txt submissions wrap each document in <DOCUMENT>; exhibits are
    the 2nd..Nth documents. EXHIBIT RULE: only the first document (the
    10-K/10-Q body) is kept. Modern filings list exhibits as separate files
    and only primaryDocument is ever fetched, so they never enter here."""
    m = _SGML_DOC_RE.search(raw)
    if not m:
        return raw
    body = m.group(1)
    t = _SGML_TEXT_RE.search(body)
    return t.group(1) if t else body


def _normalize_ws(s: str) -> str:
    s = s.replace("\xa0", " ").replace("\u200b", "").replace("\t", " ").replace("\r", "\n")
    s = re.sub(r"[ \f\v]+", " ", s)
    return s


def html_to_text(raw) -> tuple:
    """Raw primary document (bytes or str; HTML/iXBRL or SGML text) ->
    (list of non-empty stripped lines, counts). Drops script/style/head,
    the inline-XBRL header (ix:header — hidden facts, contexts), any
    display:none subtree, numeric tables (TABLE RULE), and SGML exhibits."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            raw = raw.decode("latin-1")
    raw = _first_sgml_document(raw)
    is_html = bool(re.search(r"<(html|body|div|p|table|font)\b", raw[:20000], re.I))
    counts = {"tables_dropped": 0, "tables_kept": 0, "hidden_dropped": 0, "is_html": is_html}
    if is_html:
        p = _TextExtractor()
        p.feed(raw)
        p.close()
        text = "".join(p.out)
        counts.update(tables_dropped=p.tables_dropped, tables_kept=p.tables_kept,
                      hidden_dropped=p.hidden_dropped)
    else:
        text = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = _normalize_ws(text)
    lines = [ln.strip() for ln in text.split("\n")]
    return [ln for ln in lines if ln], counts


# ════════════════════════════════════════════════════════════════════════════
# Boilerplate rules (each documented, each counted).
# ════════════════════════════════════════════════════════════════════════════

# B1 PAGE NUMBERS: a line that is only a page number — "12", "- 12 -",
#    "Page 12", "F-3", "A-1", roman "ii"/"xiv" (i..xxxix), "12 | Page".
_PAGE_NUM_RE = re.compile(
    r"^(?:page\s*)?[-–—]?\s*(?:[A-Z]-?\s?)?\d{1,3}\s*[-–—]?(?:\s*\|?\s*page)?$"
    r"|^(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})$", re.I)  # roman i..xxxix only (not "Civil")
# B2 TOC BACK-LINKS: a line that is only "Table of Contents" / "Index" /
#    "Back to Contents" / "Return to Table of Contents".
_TOC_LINE_RE = re.compile(
    r"^(?:(?:back|return)\s+to\s+)?(?:table\s+of\s+contents|contents|index|toc)\.?$", re.I)
# B3 RUNNING HEADERS/FOOTERS: a line <= 120 chars that occurs >= REPEAT_MIN
#    times in the document. Ordinary lines are compared digit-normalized
#    ("Acme Corp | 2025 Form 10-K | #"), so page-numbered footers collapse to
#    one form. Item/Part-shaped lines are compared EXACTLY (lowercased) —
#    digit-normalizing them would merge "Item 2"/"Item 3"/"Item 4" into one
#    form and delete real headings; exact repeats >= REPEAT_MIN of one heading
#    ("PART I Item 1A" atop every page) are running headers. A true heading
#    occurs at most twice (ToC + body), which selection handles.
REPEAT_MIN = 5
REPEAT_MAX_LEN = 120


def _repeat_key(ln: str) -> str:
    low = ln.lower()
    if _HEADING_RE.match(ln) or _PART_RE.match(ln):
        return low
    return re.sub(r"\d+", "#", low)


def strip_boilerplate(lines: list) -> tuple:
    counts = {"page_number_lines": 0, "toc_link_lines": 0, "repeated_header_lines": 0}
    norm = [_repeat_key(ln) for ln in lines]
    freq: dict = {}
    for n, ln in zip(norm, lines):
        if len(ln) <= REPEAT_MAX_LEN:
            freq[n] = freq.get(n, 0) + 1
    out = []
    for ln, n in zip(lines, norm):
        if _PAGE_NUM_RE.match(ln):
            counts["page_number_lines"] += 1
            continue
        if _TOC_LINE_RE.match(ln):
            counts["toc_link_lines"] += 1
            continue
        if len(ln) <= REPEAT_MAX_LEN and freq.get(n, 0) >= REPEAT_MIN:
            counts["repeated_header_lines"] += 1
            continue
        out.append(ln)
    return out, counts


# ════════════════════════════════════════════════════════════════════════════
# Section location (pure).
# ════════════════════════════════════════════════════════════════════════════

_HEADING_RE = re.compile(
    r"^items?\s*(\d{1,2})\s*([a-d])?(?![a-z])\s*(?:and\s*\d{1,2}[a-d]?)?\s*[\.:\-–—]*\s*(.*)$", re.I)
_PART_RE = re.compile(r"^part\s+(iv|iii|ii|i|1|2|3|4)(?![a-z])\s*[\.:\-–—,]*\s*(.*)$", re.I)
HEADING_MAX_LEN = 200
TITLE_JOIN_MAX_LEN = 120  # "Item 1A." alone on a line takes the next line as title only if this short
# A line starting "Item 1A of Part I ..." is a cross-reference in prose.
_XREF_TITLE_RE = re.compile(r"^(?:of|in|to|and|or|under|above|below|herein|hereof|for|as|is|was|,|;|\()", re.I)
_PART_NUM = {"i": 1, "1": 1, "ii": 2, "2": 2, "iii": 3, "3": 3, "iv": 4, "4": 4}

# Title -> 10-Q part inference (titles are more reliable than tracking PART
# lines, which some filings omit in the body).
_Q_PART2_TITLE_RE = re.compile(
    r"legal\s+proceedings|risk\s+factors|unregistered|defaults\s+upon|mine\s+safety|other\s+information|exhibits", re.I)
_Q_PART1_TITLE_RE = re.compile(
    r"financial\s+statements|management.{0,3}s?\s+discussion|quantitative\s+and\s+qualitative|controls\s+and\s+procedures", re.I)

_MDNA_TITLE_RE = re.compile(r"management.{0,3}s?\s+discussion|md\s*&\s*a", re.I)
_RISK_TITLE_RE = re.compile(r"risk\s+factors", re.I)
_LEGAL_TITLE_RE = re.compile(r"legal\s+proceedings", re.I)

# (section key, part (10-Q only; None for 10-K), item code, title regex)
SECTION_SPECS = {
    "10-K": [
        ("risk_factors", None, "1A", _RISK_TITLE_RE),
        ("legal_proceedings", None, "3", _LEGAL_TITLE_RE),
        ("mdna", None, "7", _MDNA_TITLE_RE),
    ],
    "10-Q": [
        ("mdna", 1, "2", _MDNA_TITLE_RE),
        ("legal_proceedings", 2, "1", _LEGAL_TITLE_RE),
        ("risk_factors", 2, "1A", _RISK_TITLE_RE),
    ],
}

# Disclosure-style detectors, applied to the first DISCLOSURE_HEAD_CHARS of a
# section, and only when the section is shorter than DISCLOSURE_MAX_CHARS (a
# long substantive section that mentions "incorporated by reference" once is
# not a cross-reference stub). Text is always kept; these only flag.
DISCLOSURE_HEAD_CHARS = 1500
DISCLOSURE_MAX_CHARS = 5000
_NO_MATERIAL_RE = re.compile(r"no\s+material\s+changes?|not\s+materially\s+changed|"
                             r"no\s+(?:other\s+)?(?:significant|material)\s+(?:changes|updates)", re.I)
_NOT_APPLICABLE_RE = re.compile(r"not\s+applicable|\bn/a\b|(?:^|\n)\s*none\.?\s*(?:\n|$)|smaller\s+reporting\s+compan(?:y|ies).{0,120}not\s+required"
                                r"|not\s+required\s+(?:for|of)\s+(?:a\s+)?smaller\s+reporting\s+compan", re.I | re.S)
_INCORP_REF_RE = re.compile(r"incorporated\s+(?:herein\s+)?by\s+reference|"
                            r"(?:is|are)\s+(?:included|contained|set\s+forth|presented)\s+(?:in|under)|"
                            r"information\s+required\s+by\s+this\s+item|"
                            r"(?:appears|is\s+located|can\s+be\s+found|begins)\s+on\s+pages?|"
                            r"(?:refer\s+to|see)\s+(?:note|the\s+information|\"|“|part)", re.I)

MIN_BODY_CHARS = 200  # below this a chosen span is flagged short (likely ToC/stub)

# CROSS-REFERENCE-INDEX FORMAT (big banks, GE, ...): the 10-K body is
# organized by the company's own headings and Items appear only in a "Form
# 10-K cross-reference index" mapping each Item to page numbers ("Item 1A.
# Risk Factors 26-33", Citi's "1A. Risk Factors 50–64"). Item-heading location
# cannot work there by construction; such filings are FLAGGED
# (cross_reference_index_format) so downstream comparison can exclude them,
# rather than guessed at with title-only heuristics.
_XREF_INDEX_LINE_RE = re.compile(
    r"^(?:items?\s*)?(?:1a|3|7|2)\.?\s+(?:risk\s+factors|legal\s+proceedings|management.{0,3}s?\s+discussion)"
    r".{0,120}?\b\d{1,3}(?:\s*[-–—]\s*\d{1,3})?\s*$", re.I)


def _item_ordinal(num: str, letter: Optional[str]) -> float:
    return int(num) + (("abcd".index(letter.lower()) + 1) / 10.0 if letter else 0.0)


def find_headings(lines: list, form: str) -> list:
    """All Item headings as dicts {idx, code, ordinal, part, title, text}.
    A heading whose title is empty takes the next line as its title (Item
    number and title in separate blocks). 10-Q part = title inference, else
    the most recent PART line; 10-K part is always 0 (items are unique)."""
    quarterly = base_form(form) == "10-Q"
    heads, part = [], None
    for i, ln in enumerate(lines):
        if len(ln) > HEADING_MAX_LEN:
            continue
        pm = _PART_RE.match(ln)
        if pm and len(ln) <= 120:
            part = _PART_NUM[pm.group(1).lower()]
            inner = pm.group(2)
            hm_inner = _HEADING_RE.match(inner) if inner else None
            if not hm_inner:
                continue
            ln = inner
        hm = _HEADING_RE.match(ln)
        if not hm:
            continue
        title = hm.group(3).strip()
        if title and _XREF_TITLE_RE.match(title):
            continue
        joined = False
        if not title and i + 1 < len(lines) and len(lines[i + 1]) <= TITLE_JOIN_MAX_LEN \
                and not _HEADING_RE.match(lines[i + 1]) and not _PART_RE.match(lines[i + 1]):
            title, joined = lines[i + 1].strip(), True
        code = hm.group(1).lstrip("0") + (hm.group(2) or "").upper()
        hpart = 0
        if quarterly:
            if _Q_PART2_TITLE_RE.search(title):
                hpart = 2
            elif _Q_PART1_TITLE_RE.search(title):
                hpart = 1
            else:
                hpart = part or 1
        heads.append({"idx": i, "code": code, "ordinal": _item_ordinal(hm.group(1), hm.group(2)),
                      "part": hpart, "title": title, "text": ln if not joined else f"{ln} {title}",
                      "title_joined": joined})
    return heads


def disclosure_flags(text: str) -> dict:
    head = text[:DISCLOSURE_HEAD_CHARS]
    short = len(text) < DISCLOSURE_MAX_CHARS
    return {
        "no_material_changes": bool(short and _NO_MATERIAL_RE.search(head)),
        "not_applicable": bool(short and _NOT_APPLICABLE_RE.search(head)),
        "incorporated_by_reference": bool(short and _INCORP_REF_RE.search(head)),
    }


def _empty_section() -> dict:
    return {"found": False, "text": "", "char_count": 0, "start_heading": None,
            "end_heading": None, "end_found": False, "candidates": 0,
            "toc_duplicate_defeated": False, "short": False, "title_mismatch_fallback": False,
            "flags": {"no_material_changes": False, "not_applicable": False,
                      "incorporated_by_reference": False}}


def _spans(lines: list, heads: list, cands: list) -> list:
    spans = []
    for h in cands:
        key = (h["part"], h["ordinal"])
        end = next((e for e in heads if e["idx"] > h["idx"] and (e["part"], e["ordinal"]) > key), None)
        body_start = h["idx"] + (2 if h["title_joined"] else 1)
        body_end = end["idx"] if end else len(lines)
        text = "\n".join(lines[body_start:body_end]) if body_end > body_start else ""
        spans.append((end is not None, len(text), h, end, text))
    return spans


def locate_section(lines: list, heads: list, part: Optional[int], code: str, title_re,
                   code_only_fallback: bool = False) -> dict:
    """Candidate starts = headings with this item code (and part, 10-Q) whose
    title matches (or is empty). Each candidate's span runs to the next
    heading with a strictly greater (part, ordinal) key — so 7A/8 end MD&A,
    1B/1C/2 end Risk Factors, PART II items end 10-Q MD&A — or to document
    end (end_found False).

    TOC DEFEAT: a ToC row is a heading whose span is ~one line (page number)
    before the next ToC row. Among candidates, prefer those with a found end,
    then take the LONGEST span. When that is not the first candidate, a
    ToC/cross-reference duplicate was defeated and toc_duplicate_defeated is
    set (the gate report counts these).

    CODE-ONLY FALLBACK (10-K only, where every Item number is unique): if no
    title-matched candidate yields a substantive span, headings with the
    right code but a non-matching title (filer typos such as "MANAGEMENT'S
    DISCUSSN AND ANYALYSI") are tried; their title is NOT treated as a joined
    line. Used => title_mismatch_fallback True. Never used for 10-Q, whose
    Part I and Part II reuse Item numbers."""
    sec = _empty_section()
    cands, loose = [], []
    for h in heads:
        if h["code"] != code:
            continue
        if part is not None and h["part"] != part:
            continue
        if h["title"] and not title_re.search(h["title"]):
            loose.append(dict(h, title_joined=False))
            continue
        cands.append(h)
    sec["candidates"] = len(cands)
    spans = _spans(lines, heads, cands)
    if code_only_fallback and loose and not any(n >= MIN_BODY_CHARS for _, n, *_ in spans):
        loose_spans = _spans(lines, heads, loose)
        if any(n >= MIN_BODY_CHARS for _, n, *_ in loose_spans):
            spans, cands = loose_spans, loose
            sec["candidates"] = len(loose)
            sec["title_mismatch_fallback"] = True
    if not spans:
        return sec
    with_end = [s for s in spans if s[0]] or spans
    best = max(with_end, key=lambda s: s[1])
    has_end, n, h, end, text = best
    sec.update(found=bool(text), text=text, char_count=n, start_heading=h["text"],
               end_heading=end["text"] if end else None, end_found=has_end,
               toc_duplicate_defeated=len(cands) > 1 and best is not spans[0],
               short=n < MIN_BODY_CHARS, flags=disclosure_flags(text))
    return sec


def extract_sections(raw, form: str) -> dict:
    """Primary document -> {"sections": {...}, "extraction": {...}}. Pure."""
    lines, counts = html_to_text(raw)
    lines, bcounts = strip_boilerplate(lines)
    counts.update(bcounts)
    counts["lines"] = len(lines)
    counts["text_chars"] = sum(len(ln) for ln in lines)
    bf = base_form(form)
    specs = SECTION_SPECS.get(bf)
    flags = []
    sections = {}
    if specs is None:
        flags.append("unsupported_form")
    else:
        heads = find_headings(lines, bf)
        counts["headings"] = len(heads)
        if not heads:
            flags.append("no_item_headings")
        for key, part, code, title_re in specs:
            s = locate_section(lines, heads, part, code, title_re, code_only_fallback=(bf == "10-K"))
            sections[key] = s
            if not s["found"]:
                flags.append(f"{key}_not_found")
            elif s["short"]:
                flags.append(f"{key}_short")
            if s["found"] and not s["end_found"]:
                flags.append(f"{key}_end_not_found")
            if s["toc_duplicate_defeated"]:
                flags.append(f"{key}_toc_duplicate_defeated")
            if s["title_mismatch_fallback"]:
                flags.append(f"{key}_title_mismatch_fallback")
        # No target section substantive => no_substantive_sections. If in
        # addition a target Item heading WAS located but spans nothing or a
        # stub (page number), or the explicit "Item 1A. Risk Factors 26-33"
        # line form is present, that is the cross-reference-index signature
        # (GE, Citi; Synchrony puts page numbers on their own lines).
        if not any(sx["found"] and not sx["short"] for sx in sections.values()):
            flags.append("no_substantive_sections")
            if any(sx["candidates"] > 0 for sx in sections.values()) or \
                    any(_XREF_INDEX_LINE_RE.match(ln) for ln in lines):
                flags.append("cross_reference_index_format")
    return {"sections": sections,
            "extraction": {"version": EXTRACTION_VERSION, "flags": flags, "counts": counts}}


# ════════════════════════════════════════════════════════════════════════════
# Prior-filing pairing (pure).
# ════════════════════════════════════════════════════════════════════════════

PAIR_TOLERANCE_DAYS = 45


def _d(s: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)[:10]) if s else None
    except ValueError:
        return None


def pair_prior_filings(filings: list, tolerance_days: int = PAIR_TOLERANCE_DAYS,
                       include_amendments: bool = False) -> dict:
    """accession -> {"prior": accession|None, "gap_days": int|None} for each
    (non-amendment, unless include_amendments) filing. Prior = the same
    company's same BASE form (10-K->10-K, 10-Q->10-Q) whose period
    (reportDate; filingDate if absent) is within +/- tolerance_days of exactly
    one year (365 days) earlier — the previous year's 10-K, or the 10-Q for
    the same CALENDAR period a year back. After a fiscal-year change this can
    pair different fiscal quarters without flagging it, and 10-KT/10-QT
    transition reports are not paired (no fiscal-period field yet). Closest to 365 days wins; ties
    go to the earlier-filed. Amendments never serve as priors by default."""
    pool = [f for f in filings if include_amendments or not f.get("isAmendment")]
    out = {}
    for f in pool:
        p = _d(f.get("reportDate")) or _d(f.get("filingDate"))
        if p is None:
            out[f["accession"]] = {"prior": None, "gap_days": None}
            continue
        best = None
        for g in pool:
            if g is f or g.get("cik") != f.get("cik") or base_form(g["form"]) != base_form(f["form"]):
                continue
            q = _d(g.get("reportDate")) or _d(g.get("filingDate"))
            if q is None:
                continue
            gap = (p - q).days
            if abs(gap - 365) > tolerance_days:
                continue
            rank = (abs(gap - 365), g.get("filingDate") or "", g["accession"])
            if best is None or rank < best[0]:
                best = (rank, g, gap)
        out[f["accession"]] = {"prior": best[1]["accession"] if best else None,
                               "gap_days": best[2] if best else None}
    return out


# ════════════════════════════════════════════════════════════════════════════
# Per-filing pipeline.
# ════════════════════════════════════════════════════════════════════════════

def _fetch_or_load(client: EdgarClient, url: str, path: str) -> tuple:
    """Archived bytes win (write-once); else fetch and archive. -> (bytes, fetched)"""
    if os.path.exists(path):
        return read_gz(path), False
    body = client.get(url)
    write_once_gz(path, body)
    return body, True


def process_filing(client: EdgarClient, filing: dict, archive: str) -> dict:
    """Fetch (or load) raw + index headers, extract, write sections JSON.
    Returns the sections document plus a `_io` stats dict (not persisted)."""
    cik = filing["cik"]
    acc = filing["accession"]
    acc_nodash = acc.replace("-", "")
    doc = filing.get("primaryDocument")
    if not doc:
        raise ValueError(f"{acc}: no primaryDocument in submissions JSON")
    rp = raw_path(archive, cik, acc, doc)
    raw, fetched = _fetch_or_load(client, ARCHIVES_DOC_URL.format(cik=int(cik), acc_nodash=acc_nodash, doc=doc), rp)
    hp = headers_path(archive, cik, acc)
    hdr_page = None
    try:
        hdr_bytes, _ = _fetch_or_load(client, INDEX_HEADERS_URL.format(cik=int(cik), acc_nodash=acc_nodash, acc=acc), hp)
        hdr_page = hdr_bytes.decode("latin-1")
    except EdgarHTTPError as e:
        print(f"[sec_filings_text] index-headers unavailable for {acc}: {e}", file=sys.stderr)
    accepted_hdr = parse_index_headers_acceptance(hdr_page) if hdr_page else None
    accepted_json = json_acceptance_conservative_utc(filing.get("acceptanceDateTime"))
    ext = extract_sections(raw, filing["form"])
    out = {
        "schema_version": SECTIONS_SCHEMA_VERSION,
        "cik": cik10(cik),
        "accession": acc,
        "form": filing["form"],
        "is_amendment": bool(filing.get("isAmendment")),
        "filing_date": filing.get("filingDate"),
        "accepted_at": accepted_hdr or accepted_json,
        "accepted_at_source": "index_headers" if accepted_hdr else (
            "submissions_json_unverified" if accepted_json else None),
        "accepted_at_et": parse_index_headers_et(hdr_page) if hdr_page else None,
        "acceptance_datetime_raw": filing.get("acceptanceDateTime"),
        "period": filing.get("reportDate"),
        "primary_document": doc,
        "source_url": ARCHIVES_DOC_URL.format(cik=int(cik), acc_nodash=acc_nodash, doc=doc),
        "sections": ext["sections"],
        "extraction": ext["extraction"],
    }
    sp = sections_path(archive, cik, acc)
    write_sections(sp, out)
    out["_io"] = {"raw_fetched": fetched, "raw_gz_bytes": os.path.getsize(rp),
                  "raw_bytes": len(raw), "sections_gz_bytes": os.path.getsize(sp),
                  "json_mode": classify_json_acceptance(filing.get("acceptanceDateTime"), accepted_hdr)}
    return out


# ════════════════════════════════════════════════════════════════════════════
# Sample CLI + DATA-GATE report.
# ════════════════════════════════════════════════════════════════════════════

def unique_ciks_in_rank_order(tickers_json: dict) -> list:
    """company_tickers.json is keyed "0".."N" in SEC's (roughly market-cap)
    order; multiple tickers share a CIK — keep the first (highest-rank)."""
    seen, out = set(), []
    for k in sorted(tickers_json, key=lambda x: int(x)):
        v = tickers_json[k]
        c = int(v["cik_str"])
        if c in seen:
            continue
        seen.add(c)
        out.append({"cik": cik10(c), "ticker": v.get("ticker"), "title": v.get("title"), "rank": len(out)})
    return out


def stratified_strata(entries: list, n: int, seed: int) -> list:
    """Split the rank-ordered list into n contiguous strata (large caps ...
    micro caps) and return, per stratum, its members in a seeded random
    order — the sampler takes the first member that files 10-K/10-Q."""
    rng = random.Random(seed)
    strata = []
    size = len(entries) / n
    for s in range(n):
        lo, hi = int(round(s * size)), int(round((s + 1) * size))
        members = entries[lo:hi]
        order = list(range(len(members)))
        rng.shuffle(order)
        strata.append([members[i] for i in order])
    return strata


def _pct(vals: list, q: float):
    if not vals:
        return None
    v = sorted(vals)
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return int(round(v[lo] + (v[hi] - v[lo]) * (k - lo)))


def gate_report(docs: list, listings: dict, pairing: dict, errors: list, meta: dict) -> str:
    originals = [d for d in docs if not d["is_amendment"]]
    lines = ["=" * 78, "LAZY PRICES — DATA-GATE REPORT (ladder gate 1: DATA)", "=" * 78]
    for k, v in meta.items():
        lines.append(f"{k:<34} {v}")
    by_form: dict = {}
    for d in originals:
        by_form.setdefault(base_form(d["form"]), []).append(d)
    lines.append(f"{'filings archived (originals)':<34} {len(originals)} "
                 + " ".join(f"{f}={len(v)}" for f, v in sorted(by_form.items())))
    lines.append(f"{'amendments archived (excluded)':<34} {len(docs) - len(originals)}")
    lines.append("")
    lines.append(f"{'form':<5} {'section':<18} {'found':>9} {'noHead':>6} {'short':>6} {'noEnd':>6} {'tocDef':>6} "
                 f"{'p10':>7} {'median':>7} {'p90':>8} {'noMat':>6} {'n/a':>5} {'incRef':>6}")
    for f, ds in sorted(by_form.items()):
        for key, *_ in SECTION_SPECS[f]:
            ss = [d["sections"][key] for d in ds]
            found = [s for s in ss if s["found"]]
            cc = [s["char_count"] for s in found]
            n = len(ss)
            lines.append(
                f"{f:<5} {key:<18} {len(found):>4}/{n:<4} {sum(s['candidates'] == 0 for s in ss):>6} "
                f"{sum(s['short'] for s in found):>6} "
                f"{sum(not s['end_found'] for s in found):>6} {sum(s['toc_duplicate_defeated'] for s in found):>6} "
                f"{_pct(cc, .1) or 0:>7} {_pct(cc, .5) or 0:>7} {_pct(cc, .9) or 0:>8} "
                f"{sum(s['flags']['no_material_changes'] for s in found):>6} "
                f"{sum(s['flags']['not_applicable'] for s in found):>5} "
                f"{sum(s['flags']['incorporated_by_reference'] for s in found):>6}")
    lines.append("  (found = heading located + non-empty span; noHead = no matching Item heading at all (absent from")
    lines.append("   filing or unlocatable); short = <%d chars; noEnd = ran to document end;" % MIN_BODY_CHARS)
    lines.append("   tocDef = ToC/cross-ref duplicate defeated; noMat/n/a/incRef = disclosure flags)")
    lines.append("")
    tot = sum(1 for d in originals for s in d["sections"].values() if s["found"])
    toc = sum(1 for d in originals for s in d["sections"].values() if s["toc_duplicate_defeated"])
    lines.append(f"{'ToC-duplicate fallbacks used':<34} {toc} of {tot} found sections")
    qrf = [d["sections"]["risk_factors"] for d in by_form.get("10-Q", []) if d["sections"]["risk_factors"]["found"]]
    if qrf:
        nm = sum(s["flags"]["no_material_changes"] for s in qrf)
        lines.append(f"{'10-Q Item 1A no-material-changes':<34} {nm}/{len(qrf)} = {nm / len(qrf):.1%}")
    no_heads = sum(1 for d in originals if "no_item_headings" in d["extraction"]["flags"])
    lines.append(f"{'filings with NO Item headings':<34} {no_heads}")
    for flag in ("no_substantive_sections", "cross_reference_index_format"):
        lines.append(f"{'filings flagged ' + flag:<34} {sum(1 for d in originals if flag in d['extraction']['flags'])}")
    tm = sum(1 for d in originals for sx in d["sections"].values() if sx.get("title_mismatch_fallback"))
    lines.append(f"{'10-K code-only (title typo) fallback':<34} {tm}")
    lines.append("")
    paired_meta = [a for a in (d["accession"] for d in originals) if (pairing.get(a) or {}).get("prior")]
    archived = {d["accession"] for d in originals}
    paired_both = [a for a in paired_meta if pairing[a]["prior"] in archived]
    if originals:
        lines.append(f"{'pairing coverage (prior listed)':<34} {len(paired_meta)}/{len(originals)} = {len(paired_meta) / len(originals):.1%}")
        lines.append(f"{'pairing coverage (prior archived)':<34} {len(paired_both)}/{len(originals)} = {len(paired_both) / len(originals):.1%}")
    by_form_pair = {}
    for d in originals:
        bf = base_form(d["form"])
        t = by_form_pair.setdefault(bf, [0, 0])
        t[1] += 1
        t[0] += bool((pairing.get(d["accession"]) or {}).get("prior"))
    for bf, (a, b) in sorted(by_form_pair.items()):
        lines.append(f"{'  ' + bf + ' prior listed':<34} {a}/{b}")
    lines.append("")
    src = {}
    for d in docs:
        src[d["accepted_at_source"]] = src.get(d["accepted_at_source"], 0) + 1
    modes = {}
    for d in docs:
        if d["_io"]["json_mode"] is not None:
            modes[d["_io"]["json_mode"]] = modes.get(d["_io"]["json_mode"], 0) + 1
    lines.append(f"{'accepted_at source':<34} {src}")
    lines.append(f"{'JSON acceptanceDateTime mode':<34} {modes}  (vs index-headers truth)")
    rg = [d["_io"]["raw_gz_bytes"] for d in docs]
    sg = [d["_io"]["sections_gz_bytes"] for d in docs]
    rb = [d["_io"]["raw_bytes"] for d in docs]
    if rg:
        lines.append(f"{'raw bytes / filing (median, mean)':<34} {statistics.median(rb):,.0f}  {statistics.mean(rb):,.0f}")
        lines.append(f"{'raw .gz / filing (median, mean)':<34} {statistics.median(rg):,.0f}  {statistics.mean(rg):,.0f}")
        lines.append(f"{'sections .gz / filing (median, mean)':<34} {statistics.median(sg):,.0f}  {statistics.mean(sg):,.0f}")
    lines.append("")
    lines.append(f"{'errors':<34} {len(errors)}")
    for e in errors[:40]:
        lines.append(f"  - {e}")
    return "\n".join(lines)


def run_sample(n: int, since: str, archive: str, seed: int = 20261016, workers: int = 4,
               client: Optional[EdgarClient] = None, max_tries_per_stratum: int = 6,
               report_json: Optional[str] = None) -> str:
    client = client or EdgarClient()
    entries = unique_ciks_in_rank_order(client.get_json(COMPANY_TICKERS_URL))
    strata = stratified_strata(entries, n, seed)
    pair_since = (date.fromisoformat(since) - timedelta(days=365 + PAIR_TOLERANCE_DAYS + 30)).isoformat()
    chosen, listings, errors, replaced = [], {}, [], 0
    for si, members in enumerate(strata):
        for t, m in enumerate(members[:max_tries_per_stratum]):
            try:
                rows = list_filings(m["cik"], since=pair_since, client=client)
            except (EdgarHTTPError, ValueError) as e:
                errors.append(f"list {m['cik']} ({m['ticker']}): {e}")
                continue
            if any(r["filingDate"] >= since for r in rows):
                chosen.append(m)
                listings[m["cik"]] = rows
                replaced += t > 0
                break
    todo = [r for c in chosen for r in listings[c["cik"]] if r["filingDate"] >= since]
    docs = []
    lock = threading.Lock()

    def work(r):
        try:
            d = process_filing(client, r, archive)
            with lock:
                docs.append(d)
        except (EdgarHTTPError, ValueError, OSError) as e:
            with lock:
                errors.append(f"filing {r['accession']} ({r['form']}): {e}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(work, todo))
    pairing = {}
    for c in chosen:
        pairing.update(pair_prior_filings(listings[c["cik"]]))
    meta = {
        "since": since, "seed": seed, "archive": archive,
        "companies sampled": f"{len(chosen)} of {n} strata ({replaced} strata re-drawn: first pick filed no 10-K/10-Q)",
        "CIK universe (company_tickers)": len(entries),
        "rank range": f"{min((c['rank'] for c in chosen), default=0)}..{max((c['rank'] for c in chosen), default=0)}",
        "filings queued": len(todo),
        "http requests / retries / errors": f"{client.stats['requests']} / {client.stats['retries']} / {client.stats['errors']}",
        "wall time (s)": f"{time.time() - t0:.0f}",
    }
    report = gate_report(docs, listings, pairing, errors, meta)
    if report_json:
        slim = [{k: v for k, v in d.items() if k != "sections"} | {
            "sections": {k: {kk: vv for kk, vv in s.items() if kk != "text"} for k, s in d["sections"].items()}}
            for d in docs]
        with open(report_json, "w") as f:
            json.dump({"companies": chosen, "docs": slim, "pairing": pairing, "errors": errors}, f, indent=1)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="stratified CIK sample -> archive + DATA-GATE report")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--since", default="2023-01-01")
    s.add_argument("--archive", default=None, help="archive base dir (default: DATA_DIR/datacore_archive)")
    s.add_argument("--seed", type=int, default=20261016)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--rate", type=float, default=DEFAULT_RATE_PER_SEC)
    s.add_argument("--report-json", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "sample":
        process_limiter().set_rate(args.rate)
        client = EdgarClient()
        print(run_sample(args.n, args.since, args.archive or archive_base_dir(), seed=args.seed,
                         workers=args.workers, client=client, report_json=args.report_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
