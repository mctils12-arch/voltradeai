#!/usr/bin/env python3
"""cusip_figi.py: what kind of security a CUSIP is, and its US ticker, asked
of OpenFIGI once per CUSIP and kept (open_questions.md (as2) step 1,
2026-11-26).

WHAT IT IS FOR. Item 3 of the registered gate-2 test for
`dtcc_sbsdr_equity_swaps` (open_questions.md 2026-11-25) keeps an underlier
only if its CUSIP "resolves to a US-listed common stock, ADR or REIT". The
vendor's file names an underlier by CUSIP and nothing else. Until this file
the only resolver was server/cusipResolver.ts, run by hand from a session
(scripts/dtcc_resolve_cusips.ts): its cache was never on the server and it
did not keep the security type. The test runs on the server in Python,
beside the price panel, so the record is built there, by the panel's own
nightly job (dtcc_trs_panel_job.py).

WHERE IT LIVES. The repo root: the image copies root-level Python only.

SOURCE. POST https://api.openfigi.com/v3/mapping, jobs of
{"idType": "ID_CUSIP", "idValue": <9 characters>}. Free and keyless: 25
requests a minute of at most 10 jobs (the response's `ratelimit-policy`
header, read 2026-11-26). OPENFIGI_API_KEY, if the environment holds one,
raises that to 100 jobs a request and a shorter wait; it is never needed.
The request carries CUSIPs and nothing else.

WHAT IS KEPT, one JSON line per CUSIP in an append-only file (a later line
for the same CUSIP wins):
  {"cusip", "asked_at", "found": true, "us": true|false, "listings": n,
   "ticker", "name", "figi", "composite_figi", "share_class_figi",
   "exch_code", "security_type", "security_type2", "market_sector"}
  {"cusip", "asked_at", "found": false, "reason": <OpenFIGI's own words>}
`us` says whether OpenFIGI lists the security under exchange code "US" (its
US composite); the kept fields are that listing's, or the first listing's
when there is none. A CUSIP OpenFIGI does not know is kept as not found and
is not asked again: without that every run would ask for it again.

WHAT IT CANNOT SAY. The ticker and the type are TODAY'S, not the event
day's: OpenFIGI's mapping has no date. A company that changed its ticker or
converted (a SPAC unit that became a common share) reads as it is now. The
test's reader must match prices by the price panel's asset and date
segment, never by this ticker alone (registration item 5).

THE TRADABLE CLASS (`tradable_class`). It makes item 3's words exact and is
part of the registration from here on:
  kept      market_sector is "Equity" and security_type is exactly
            "Common Stock" (class "common"), "ADR" ("adr") or "REIT"
            ("reit").
  dropped   everything else, counted under its own security_type: funds and
            exchange-traded products, preferreds, units, warrants, rights,
            partnerships, and anything outside the Equity sector (the file
            also carries swaps on convertible notes).
  unresolved  not found, counted apart.
`us` is NOT part of the class. HISTORY, stated because the rule changed with
type counts in hand: it was first written (2026-11-26, before any panel
CUSIP had been asked) with "`us` is true" as a condition. The first read of
the 4,544 event underliers showed what that condition measures: a company
bought or renamed since the event keeps its old CUSIP only on foreign
venues, so "US composite today" drops it. 5.7% of the events of 2025 H1 were
on such names against 1.0% of 2026 H2: the older the event, the more are
dropped, by something that happened after it. That is a filter on the
future (REASONING STANDARD 7). The condition was removed the same session,
before any price was read. Whether a name was US-listed ON THE EVENT DAY is
for the price panel to say (item 3's bars, price and dollar-volume lines),
not for today's OpenFIGI.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional, Tuple

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
CACHE_FILE = "cusip_figi.jsonl"
KEYLESS_BATCH, KEYLESS_WAIT_S = 10, 2.6     # 25 requests a minute, with margin
KEYED_BATCH, KEYED_WAIT_S = 100, 0.3        # 25 requests per 6 s, with margin
# Stop a run after this many requests in a row that got no answer.
MAX_CONSECUTIVE_FAILURES = 3

TRADABLE_TYPES = {"Common Stock": "common", "ADR": "adr", "REIT": "reit"}
# OpenFIGI's words for an idValue that cannot be a CUSIP (read 2026-11-26).
INVALID_ID_ERROR = "Invalid idValue"
NO_US_KEY = "kept, no US listing today"

# (status, body, headers). status 0 = no HTTP answer at all.
Post = Callable[[str, bytes, Dict[str, str]], Tuple[int, bytes, Dict[str, str]]]


def is_cusip(s: object) -> bool:
    return isinstance(s, str) and len(s) == 9 and s.isalnum() and s.isascii()


def read_cache(path: str) -> Dict[str, Dict[str, object]]:
    """{cusip: entry}; a later line wins. A line that is not JSON (a write cut
    by a kill) is counted on stderr and skipped: its CUSIP is asked again."""
    out: Dict[str, Dict[str, object]] = {}
    bad = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if isinstance(e, dict) and is_cusip(e.get("cusip")) and isinstance(e.get("found"), bool):
                    out[str(e["cusip"])] = e
                else:
                    bad += 1
    except FileNotFoundError:
        return out
    if bad:
        print(f"[cusip_figi] {bad} unreadable line(s) in {path} skipped", file=sys.stderr)
    return out


def pick_listing(data: List[Dict[str, object]]) -> Tuple[Optional[Dict[str, object]], bool]:
    """OpenFIGI answers with every venue a security trades on. The US
    composite (exchCode "US") is the listing the test means by "US-listed";
    with none, the first listing is kept and marked not US."""
    rows = [d for d in data if isinstance(d, dict)]
    if not rows:
        return None, False
    for d in rows:
        if d.get("exchCode") == "US":
            return d, True
    return rows[0], False


def entry_from_job(cusip: str, job: object, asked_at: str) -> Optional[Dict[str, object]]:
    """One job's answer as a cache line. None when the answer is neither a
    listing nor a plain "not found" (an error for this job: ask again)."""
    if not isinstance(job, dict):
        return None
    data = job.get("data")
    if isinstance(job.get("error"), str) and job["error"].startswith(INVALID_ID_ERROR):
        # Not a CUSIP at all (the file carries a few index and basket codes
        # of nine characters): an answer, kept, so it is not asked nightly.
        return {"cusip": cusip, "asked_at": asked_at, "found": False, "reason": job["error"][:80]}
    if isinstance(data, list):
        d, us = pick_listing(data)
        if d is None:
            return {"cusip": cusip, "asked_at": asked_at, "found": False, "reason": "empty data"}
        s = lambda k: (str(d[k]) if d.get(k) is not None else None)  # noqa: E731
        return {"cusip": cusip, "asked_at": asked_at, "found": True, "us": us, "listings": len(data),
                "ticker": s("ticker"), "name": s("name"), "figi": s("figi"),
                "composite_figi": s("compositeFIGI"), "share_class_figi": s("shareClassFIGI"),
                "exch_code": s("exchCode"), "security_type": s("securityType"),
                "security_type2": s("securityType2"), "market_sector": s("marketSector")}
    if "warning" in job:      # "No identifier found."
        return {"cusip": cusip, "asked_at": asked_at, "found": False, "reason": str(job["warning"])[:80]}
    return None               # any other {"error": ...}: this job failed; not an answer about the CUSIP


def tradable_class(entry: Optional[Dict[str, object]]) -> Tuple[Optional[str], str]:
    """(class, bucket). class is "common", "adr", "reit" or None; bucket is
    the name the count goes under. The rule is in the module header."""
    if entry is None:
        return None, "not asked"
    if not entry.get("found"):
        return None, "unresolved"
    st = str(entry.get("security_type") or "no type")
    if entry.get("market_sector") != "Equity":
        return None, f"{st} ({entry.get('market_sector') or 'no sector'})"
    cls = TRADABLE_TYPES.get(st)
    return cls, (cls if cls else st)


def class_counts(cache: Dict[str, Dict[str, object]], cusips: Iterable[str]) -> Dict[str, int]:
    """{bucket: count}, largest first. "kept, no US listing today" counts,
    inside the three kept classes, the CUSIPs OpenFIGI no longer lists on
    its US composite (bought, renamed or re-issued since): the test's reader
    cannot join those to prices by today's ticker."""
    out: Dict[str, int] = {}
    no_us = 0
    for c in cusips:
        e = cache.get(c)
        cls, bucket = tradable_class(e)
        out[bucket] = out.get(bucket, 0) + 1
        if cls and not (e or {}).get("us"):
            no_us += 1
    out = dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
    if no_us:
        out[NO_US_KEY] = no_us
    return out


def http_post(url: str, body: bytes, headers: Dict[str, str]) -> Tuple[int, bytes, Dict[str, str]]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read(), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300], {k.lower(): v for k, v in (e.headers or {}).items()}
    except (urllib.error.URLError, OSError) as e:
        return 0, str(e)[:200].encode(), {}


def resolve(cusips: Iterable[str], cache_path: str, budget_s: float = 600.0,
            api_key: Optional[str] = None, post: Post = http_post,
            clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
            now_iso: Optional[Callable[[], str]] = None) -> Dict[str, object]:
    """Ask OpenFIGI for the CUSIPs the cache does not hold, in the order
    given, until the time budget is spent. Each answered batch is appended
    before the next is asked, so a killed run loses one batch at most.
    Never raises on a network or HTTP failure: the summary says what happened."""
    t0 = clock()
    now_iso = now_iso or (lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    cache = read_cache(cache_path)
    seen, wanted = set(), []
    malformed = 0
    for c in cusips:
        if c in seen:
            continue
        seen.add(c)
        if not is_cusip(c):
            malformed += 1
        else:
            wanted.append(c)
    todo = [c for c in wanted if c not in cache]
    batch_n, wait_s = (KEYED_BATCH, KEYED_WAIT_S) if api_key else (KEYLESS_BATCH, KEYLESS_WAIT_S)
    headers = {"Content-Type": "application/json", "User-Agent": "voltradeai-datacore"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key

    asked = found = not_found = job_errors = requests = failures = consecutive = 0
    last_error = stopped = None
    if todo:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    i = 0
    while i < len(todo):
        if clock() - t0 >= budget_s:
            stopped = "time budget"
            break
        batch = todo[i:i + batch_n]
        body = json.dumps([{"idType": "ID_CUSIP", "idValue": c} for c in batch]).encode()
        status, raw, hdrs = post(OPENFIGI_URL, body, headers)
        requests += 1
        answer = None
        if status == 200:
            try:
                answer = json.loads(raw)
            except ValueError:
                answer = None
        if not isinstance(answer, list) or len(answer) != len(batch):
            failures += 1
            consecutive += 1
            last_error = (f"HTTP {status}" if status else "no answer") + f": {raw[:80].decode('utf-8', 'replace')}"
            if status == 429:
                # The limit is the vendor's; the header says how long. Wait
                # once if it fits the budget, otherwise leave it to the next run.
                try:
                    reset = float(hdrs.get("ratelimit-reset") or hdrs.get("retry-after") or 60)
                except ValueError:
                    reset = 60.0
                if clock() - t0 + reset >= budget_s:
                    stopped = "rate limited by OpenFIGI"
                    break
                sleep(reset)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                stopped = f"{consecutive} requests in a row got no usable answer"
                break
            if status != 429:
                sleep(wait_s)
            continue                      # the same batch again
        consecutive = 0
        at = now_iso()
        lines = []
        for c, job in zip(batch, answer):
            e = entry_from_job(c, job, at)
            if e is None:
                job_errors += 1
                continue
            cache[c] = e
            lines.append(json.dumps(e, sort_keys=True))
            asked += 1
            if e["found"]:
                found += 1
            else:
                not_found += 1
        if lines:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        i += len(batch)
        if i < len(todo):
            sleep(wait_s)

    held = sum(1 for c in wanted if c in cache)
    out: Dict[str, object] = {
        "wanted": len(wanted), "held": held, "owed": len(wanted) - held, "malformed": malformed,
        "asked": asked, "found": found, "not_found": not_found, "job_errors": job_errors,
        "requests": requests, "failed_requests": failures, "stopped": stopped,
        "keyed": bool(api_key), "elapsed_s": round(clock() - t0, 1),
    }
    if last_error:
        out["last_error"] = last_error
    # An error is a run that wanted answers and got none.
    out["status"] = "error" if (todo and requests and asked == 0) else "ok"
    return out
