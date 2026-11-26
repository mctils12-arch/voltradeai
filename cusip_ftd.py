#!/usr/bin/env python3
"""cusip_ftd.py: a DATED CUSIP-to-ticker record, read from the SEC's
fails-to-deliver files, and the gate-1 count of how much of the DTCC swap
event list it covers (open_questions.md (at2), 2026-11-26).

WHY IT EXISTS. The registered gate-2 test for `dtcc_sbsdr_equity_swaps`
(open_questions.md 2026-11-25) names an underlier by CUSIP; prices are keyed
by ticker and date. cusip_figi.py gives TODAY'S ticker. Of the 3,254
share-like event underliers measured on 2026-11-26, 216 have no US listing
today (bought or renamed since) and 247 are unknown to OpenFIGI: 463 names,
on the oldest events, and they are the takeover targets and the failures. A
test that joins by today's ticker drops them and becomes a survivor test.

THE SOURCE. sec.gov cnsfails{YYYYMM}{a|b}.zip (public domain; the file
contract is the one server/secFtd.ts verified on 2026-07-07 and has parsed
since): one pipe-delimited text file a half month,
  SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
with two trailer lines that are checked here as there. Every security with a
net fail balance on a settlement day prints its CUSIP beside its symbol AS
OF THAT DAY. That pair is a fact that existed on the day, from a source
that had no knowledge of what happened to the company afterwards.

WHAT IS KEPT. <root>/<period>.csv.gz, one line per distinct (cusip, symbol)
seen in the half month: cusip,symbol,first_day,last_day,days. No quantity,
no price, no description. A period the SEC does not serve is not recorded:
it is asked again on the next run (one to three requests).

WHAT IT CANNOT SAY. (1) A security with no fail inside the window has no
pair: absence is not a statement about the security. (2) A span says the
pair was printed on `days` days between first_day and last_day, not on every
day between them (a half month is at most 17 days). (3) The symbol is the
SEC's print; share-class punctuation differs between sources, so symbols
are compared with everything but letters and digits removed.

THE GATE-1 COUNT (`coverage`), DEFINED BEFORE ANY FILE WAS READ. No price
and no return is involved. For each event of the registered list:
  joined        the CUSIP has a pair whose span is within WINDOW_DAYS (45)
                calendar days of the event day, before or after.
  joined_back   it has one whose first day is on or before the event day
                and whose last day is at most 45 days before it: a pair a
                reader could have known on the day. The runner must use
                this side when it exists.
An UNDERLIER is counted at its FIRST event. Groups, by the security-type
record (cusip_figi.py): `kept_us` (common stock, ADR or REIT with a US
listing today), `kept_no_us` (the same types, no US listing today),
`unresolved_share_like` (unknown to OpenFIGI, issue number of two digits).
Other types are dropped by the registered filter and are not counted.
`agreement`: over joined events of `kept_us`, whether the SEC's symbol
equals OpenFIGI's ticker today. A disagreement is a rename or a reused
ticker, which is what the dated record is for; it is a count, not an error.
`by_half_year`: of the events in the three groups, the share with no join.
The registration's rule (written 2026-11-26, before this ran): if that
share differs by more than 3 points between the first and the last
half-year, the test's verdict is read with that stated beside it.

WHERE IT LIVES. The repo root: the image copies root-level Python only.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cusip_figi

# The SEC's fair-access rule asks for a declared client; this is the one
# string every SEC reader in the repository sends (sec_filings_text.SEC_UA,
# server/secFtd.ts). A test holds the two equal.
SEC_UA = "voltradeai-datacore/1.0 (research@voltradeai.com)"
URL_VARIANTS = (
    "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{p}.zip",
    "https://www.sec.gov/files/data/other/fails-deliver-data/cnsfails{p}.zip",
    "https://www.sec.gov/files/data/other/fails-deliver-data/cnsfails{p}_0.zip",
)
HEADER = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE"
WINDOW_DAYS = 45
# One request a second: a tenth of what the SEC allows.
WAIT_S = 1.0
# A period whose last day is this recent may not be published yet ("a" comes
# at about the end of its month, "b" about the 15th of the next): a refusal
# there is not a missing file.
PUBLISH_LAG_DAYS = 35
MAX_CONSECUTIVE_FAILURES = 3
SCHEMA = "cusip_ftd/1"

# (status, body). status 0 = no HTTP answer at all.
Get = Callable[[str], Tuple[int, bytes]]

_TRAILER_COUNT = re.compile(r"^Trailer record count\s+(\d+)")
_TRAILER_QTY = re.compile(r"^Trailer total quantity of shares\s+(\d+)")
_PERIOD = re.compile(r"^(\d{4})(\d{2})([ab])$")


def period_of(day: dt.date) -> str:
    """The half month a settlement day is filed under: 1st-14th is "a"."""
    return f"{day.year}{day.month:02d}{'a' if day.day <= 14 else 'b'}"


def period_bounds(period: str) -> Tuple[dt.date, dt.date]:
    m = _PERIOD.match(period)
    if not m:
        raise ValueError(f"not a period: {period!r}")
    y, mo = int(m.group(1)), int(m.group(2))
    if m.group(3) == "a":
        return dt.date(y, mo, 1), dt.date(y, mo, 14)
    nxt = dt.date(y + (mo == 12), mo % 12 + 1, 1)
    return dt.date(y, mo, 15), nxt - dt.timedelta(days=1)


def periods_between(first: dt.date, last: dt.date) -> List[str]:
    """Every half month that holds a day from `first` to `last`, oldest first."""
    out: List[str] = []
    d = first
    while d <= last:
        p = period_of(d)
        if not out or out[-1] != p:
            out.append(p)
        d = period_bounds(p)[1] + dt.timedelta(days=1)
    return out


def wanted_periods(first_event: dt.date, last_event: dt.date, today: dt.date) -> List[str]:
    """The periods a 45-day window around the events can touch, up to today."""
    w = dt.timedelta(days=WINDOW_DAYS)
    return periods_between(first_event - w, min(last_event + w, today))


def unzip_single_text(buf: bytes) -> str:
    """The one text member of an SEC zip. Anything else raises ValueError: a
    file of another shape must never parse as data."""
    try:
        with zipfile.ZipFile(io.BytesIO(buf)) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            if len(names) != 1:
                raise ValueError(f"{len(names)} members in the zip, expected 1")
            return z.read(names[0]).decode("latin-1")
    except zipfile.BadZipFile as e:
        raise ValueError(f"not a zip: {e}") from e


def parse(text: str) -> List[Tuple[str, str, str]]:
    """[(day_iso, cusip, symbol)]. Raises ValueError when the header is not
    the known one or either trailer disagrees with the rows read: a cut or
    changed file is refused whole (the same rule as server/secFtd.ts)."""
    lines = [ln for ln in re.split(r"\r?\n", text) if ln]
    if not lines or lines[0].strip() != HEADER:
        raise ValueError(f"unexpected header: {(lines[0] if lines else '')[:80]!r}")
    out: List[Tuple[str, str, str]] = []
    count: Optional[int] = None
    qty_trailer: Optional[int] = None
    qty_sum = 0
    for line in lines[1:]:
        if "|" not in line:
            m = _TRAILER_COUNT.match(line)
            if m:
                count = int(m.group(1))
                continue
            m = _TRAILER_QTY.match(line)
            if m:
                qty_trailer = int(m.group(1))
            continue
        p = line.split("|")
        if len(p) != 6 or not re.fullmatch(r"\d{8}", p[0]) or not re.fullmatch(r"-?\d+", p[3].strip()):
            continue                      # not a data row; the trailer count then refuses the file
        qty_sum += int(p[3])
        out.append((f"{p[0][:4]}-{p[0][4:6]}-{p[0][6:8]}", p[1].strip(), p[2].strip()))
    if count is None or count != len(out):
        raise ValueError(f"trailer count {count} against {len(out)} rows read")
    if qty_trailer is not None and qty_trailer != qty_sum:
        raise ValueError(f"trailer quantity {qty_trailer} against {qty_sum} summed")
    return out


def pairs(rows: Iterable[Tuple[str, str, str]]) -> Dict[Tuple[str, str], List[object]]:
    """{(cusip, symbol): [first_day, last_day, days]} for one period. A row
    with no symbol (or one with a comma: the kept file is comma-separated), or
    a CUSIP field that is not nine letters and digits, is not a pair."""
    days: Dict[Tuple[str, str], set] = {}
    for day, cusip, symbol in rows:
        if not symbol or "," in symbol or not cusip_figi.is_cusip(cusip):
            continue
        days.setdefault((cusip, symbol), set()).add(day)
    return {k: [min(v), max(v), len(v)] for k, v in days.items()}


def period_path(root: str, period: str) -> str:
    return os.path.join(root, period + ".csv.gz")


def write_period(root: str, period: str, prs: Dict[Tuple[str, str], List[object]]) -> None:
    os.makedirs(root, exist_ok=True)
    body = "".join(f"{c},{s},{v[0]},{v[1]},{v[2]}\n" for (c, s), v in sorted(prs.items()))
    tmp = period_path(root, period) + ".tmp"
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            gz.write(body.encode("ascii", "replace"))
    os.replace(tmp, period_path(root, period))


def held_periods(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(n[:-7] for n in os.listdir(root) if n.endswith(".csv.gz") and _PERIOD.match(n[:-7]))


def http_get(url: str) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except (urllib.error.URLError, OSError) as e:
        return 0, str(e)[:200].encode()


def fetch_period(period: str, get: Get = http_get, sleep: Callable[[float], None] = time.sleep
                 ) -> Tuple[str, object, int]:
    """("ok", zip bytes, requests) | ("unavailable", None, n): every known
    address answered 404 | ("error", reason, n). A 403 is the SEC refusing
    the client or the rate: the caller stops asking for this run."""
    n = 0
    reason = None
    for i, variant in enumerate(URL_VARIANTS):
        if i:
            sleep(WAIT_S)
        status, body = get(variant.format(p=period))
        n += 1
        if status == 200:
            return "ok", body, n
        if status == 404:
            continue
        if status == 403:
            return "error", "HTTP 403: the SEC refused the client or the rate", n
        reason = f"HTTP {status}" if status else f"no answer: {body[:80].decode('utf-8', 'replace')}"
    return ("error", reason, n) if reason else ("unavailable", None, n)


def build(root: str, wanted: Sequence[str], today: dt.date, budget_s: float = 200.0, get: Get = http_get,
          clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> Dict[str, object]:
    """Read the wanted periods not held, oldest first, within `budget_s`.
    Never raises on a network, HTTP or file-shape failure."""
    t0 = clock()
    have = set(held_periods(root))
    todo = [p for p in wanted if p not in have]
    recent_from = today - dt.timedelta(days=PUBLISH_LAG_DAYS)
    read = requests = errors = consecutive = not_published = 0
    unavailable: List[str] = []
    last_error = stopped = None
    for k, period in enumerate(todo):
        if clock() - t0 >= budget_s:
            stopped = "time budget"
            break
        if k:
            sleep(WAIT_S)
        status, payload, n = fetch_period(period, get, sleep)
        requests += n
        if status == "ok":
            try:
                prs = pairs(parse(unzip_single_text(payload)))  # type: ignore[arg-type]
                if not prs:
                    raise ValueError("no (cusip, symbol) pair in the file")
                write_period(root, period, prs)
                read += 1
                consecutive = 0
                continue
            except (ValueError, OSError) as e:
                status, payload = "error", f"{type(e).__name__}: {str(e)[:120]}"
        if status == "unavailable":
            consecutive = 0
            if period_bounds(period)[1] >= recent_from:
                not_published += 1
            else:
                unavailable.append(period)
            continue
        errors += 1
        consecutive += 1
        last_error = f"{period}: {payload}"
        if "403" in str(payload) or consecutive >= MAX_CONSECUTIVE_FAILURES:
            stopped = "refused by the SEC" if "403" in str(payload) else f"{consecutive} periods in a row could not be read"
            break
    have = set(held_periods(root))
    # A period the SEC may not have published yet is not owed.
    owed = [p for p in wanted if p not in have and period_bounds(p)[1] < recent_from]
    out: Dict[str, object] = {
        "schema": SCHEMA, "wanted": len(wanted), "held": sum(1 for p in wanted if p in have), "owed": len(owed),
        "read": read, "requests": requests, "errors": errors, "not_published": not_published,
        "unavailable": len(unavailable), "oldest_unavailable": min(unavailable, default=None),
        "first_period": min(have, default=None), "last_period": max(have, default=None),
        "stopped": stopped, "elapsed_s": round(clock() - t0, 1),
    }
    if last_error:
        out["last_error"] = last_error
    # An error is a run that asked for files and read none of them.
    out["status"] = "error" if (errors and read == 0) else "ok"
    return out


Span = Tuple[str, str, str, int]   # first_day, last_day, symbol, days


def read_spans(root: str, cusips: Iterable[str]) -> Dict[str, List[Span]]:
    """{cusip: spans in day order} over every held period, for the CUSIPs
    asked about only (all of them would be most of a million lines)."""
    want = set(cusips)
    out: Dict[str, List[Span]] = {}
    for period in held_periods(root):
        try:
            with gzip.open(period_path(root, period), "rt", encoding="ascii", errors="replace") as f:
                for line in f:
                    p = line.rstrip("\n").split(",")
                    if len(p) == 5 and p[0] in want:
                        out.setdefault(p[0], []).append((p[2], p[3], p[1], int(p[4])))
        except (OSError, ValueError, EOFError) as e:
            print(f"[cusip_ftd] {period} unreadable, left out: {e}", file=sys.stderr)
    for spans in out.values():
        spans.sort()
    return out


def _day(s: str) -> dt.date:
    return dt.date(int(s[:4]), int(s[5:7]), int(s[8:10]))


def symbol_on(spans: Sequence[Span], day_iso: str, window: int = WINDOW_DAYS) -> Optional[Dict[str, object]]:
    """The pair to use for an event on `day_iso`, or None. A pair known on
    the day (first day on or before it, last day at most `window` days
    before it) is always preferred to one that only appears afterwards;
    within a side the nearest span wins, and of two equally near the later."""
    day = _day(day_iso)
    back: Optional[Tuple[int, Span]] = None
    fwd: Optional[Tuple[int, Span]] = None
    for sp in spans:
        first, last = _day(sp[0]), _day(sp[1])
        if first <= day:
            gap = max(0, (day - last).days)
            if gap <= window and (back is None or gap <= back[0]):
                back = (gap, sp)
        else:
            gap = (first - day).days
            if gap <= window and (fwd is None or gap < fwd[0]):
                fwd = (gap, sp)
    pick = back or fwd
    if pick is None:
        return None
    return {"symbol": pick[1][2], "gap_days": pick[0], "known_on_the_day": back is not None,
            "first_day": pick[1][0], "last_day": pick[1][1]}


def norm_symbol(s: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def share_like(cusip: str) -> bool:
    """A two-digit issue number (characters 7 and 8): the form share lines
    take; notes and private placements carry a letter there."""
    return cusip_figi.is_cusip(cusip) and cusip[6:8].isdigit()


def group_of(cusip: str, entry: Optional[Dict[str, object]]) -> Optional[str]:
    cls, bucket = cusip_figi.tradable_class(entry)
    if cls:
        return "kept_us" if (entry or {}).get("us") else "kept_no_us"
    if bucket == "unresolved" and share_like(cusip):
        return "unresolved_share_like"
    return None


def half_year(day_iso: str) -> str:
    return f"{day_iso[:4]}H{1 if int(day_iso[5:7]) <= 6 else 2}"


GROUPS = ("kept_us", "kept_no_us", "unresolved_share_like")


def coverage(events: Iterable[Dict[str, object]], cache: Dict[str, Dict[str, object]],
             spans: Dict[str, List[Span]], window: int = WINDOW_DAYS) -> Dict[str, object]:
    """The gate-1 count defined in the module header. Counts only: the
    result names no security."""
    blank = lambda: {"underliers": 0, "underliers_joined": 0, "underliers_joined_back": 0,  # noqa: E731
                     "events": 0, "events_joined": 0, "events_joined_back": 0}
    groups = {g: blank() for g in GROUPS}
    halves: Dict[str, Dict[str, int]] = {}
    agree = {"events": 0, "same": 0, "different": 0, "underliers": 0, "underliers_same": 0}
    seen = set()
    not_counted = 0
    for e in sorted(events, key=lambda x: (str(x["date"]), str(x["cusip"]))):
        c, d = str(e["cusip"]), str(e["date"])
        entry = cache.get(c)
        g = group_of(c, entry)
        if g is None:
            not_counted += 1
            continue
        j = symbol_on(spans.get(c, ()), d, window)
        first = c not in seen
        seen.add(c)
        row = groups[g]
        h = halves.setdefault(half_year(d), {"events": 0, "no_join": 0})
        row["events"] += 1
        h["events"] += 1
        row["underliers"] += first
        if j is None:
            h["no_join"] += 1
            continue
        row["events_joined"] += 1
        row["underliers_joined"] += first
        if j["known_on_the_day"]:
            row["events_joined_back"] += 1
            row["underliers_joined_back"] += first
        if g == "kept_us":
            same = norm_symbol(j["symbol"]) == norm_symbol((entry or {}).get("ticker"))
            agree["events"] += 1
            agree["same" if same else "different"] += 1
            agree["underliers"] += first
            agree["underliers_same"] += first and same
    for h in halves.values():
        h["no_join_pct"] = round(100.0 * h["no_join"] / h["events"], 2) if h["events"] else None  # type: ignore[assignment]
    keys = sorted(halves)
    spread = None
    if len(keys) >= 2 and halves[keys[0]]["events"] and halves[keys[-1]]["events"]:
        spread = round(float(halves[keys[0]]["no_join_pct"]) - float(halves[keys[-1]]["no_join_pct"]), 2)
    return {
        "window_days": window, "groups": groups, "agreement_kept_us": agree,
        "by_half_year": {k: halves[k] for k in keys},
        # first half-year's share with no join minus the last one's, in points
        "no_join_first_minus_last_points": spread,
        "events_of_other_types": not_counted,
    }
