#!/usr/bin/env python3
"""dtcc_trs_panel.py: one row per US underlier per vendor day, from DTCC's
dated equity-swap files (open_questions.md (ap2)(a), 2026-11-25).

WHAT IT IS FOR. The ladder's gate 2 for `dtcc_sbsdr_equity_swaps` (fresh
large total-return-swap notional on small US names against forward returns)
was recorded as "blocked on accumulating archive depth". It is not: the
vendor serves every dated file for two years after it is written. This
script turns those files into the event panel the registered test reads.

WHERE IT LIVES. The repo root, because the image copies root-level Python
only (the Dockerfile's `COPY *.py`) and the registered test will run on the
server beside the price panel, as edge_p2_insider.py does.

WHAT IT DOES NOT DO. It reads no price, no market value and no return, and
nothing in the trading bot or the server reads its output. It is the event
half of the test only, so the registration in research/open_questions.md
(2026-11-25, "DTCC equity swaps: GATE 2 PRE-REGISTERED") could be written
with this panel in hand and still be blind to outcomes.

SOURCE. https://kgc0418-tdw-data-0.s3.amazonaws.com/sec/eod/
SEC_CUMULATIVE_EQUITIES_YYYY_MM_DD.zip. One zip member, one CSV, 110
columns; each file holds the lifecycle events DISSEMINATED on its day
(measured 2026-11-24). A file is moved to cold storage two years after it
is written and a GET is then refused (measured 2026-11-25); the script
records that as `unavailable`, never as an empty day.

ROW FILTER, the server's own (server/dtccSwaps.ts isUsUnderlier): the
underlier is a single identifier (no semicolon list: baskets are counted
and dropped), its source is CUSIP, or ISIN with a "US" prefix. The key is
the 9-character CUSIP (characters 3 to 11 of a US ISIN).

COLUMNS OF A PANEL ROW (definitions are part of the registration; changing
one after a return has been computed voids the test):
  date        the vendor file's day (the dissemination day)
  cusip       9 characters
  newt_n      rows with Action type NEWT and Event type TRAD whose
              Execution Timestamp falls on the file's day or the day
              before ("fresh": a late-reported old trade is not news)
  newt_usd    sum of Notional amount-Leg 1 over those rows where the
              currency is USD; a capped value ("250,000,000+") adds its
              floor. Whole dollars. A value above the vendor's cap of
              250,000,000 that is NOT marked capped is a reporter's
              placeholder: the row is counted in newt_n, counted in the
              day's `newt_notional_over_cap`, and adds nothing here.
  capped_n    how many of those rows carried a capped notional
  fx_n        how many of those rows were not in USD (counted in newt_n,
              absent from newt_usd; no exchange rate is applied)
  late_n      NEWT TRAD rows executed earlier than the day before the file
  modi_n      rows with Action type MODI
  term_n      rows with Action type TERM
  other_n     every other row (CORR, EROR, REVI, NEWT that is not TRAD)

THE PUBLIC FILE DOES NOT SAY WHO BOUGHT. A row has no direction: the
receiver of the equity return is not disseminated. The panel is therefore
unsigned activity, and the registration says so.

USAGE
  python3 dtcc_trs_panel.py build --from 2024-11-24 --to 2026-11-24 \
      --work /path/to/workdir [--workers 4]
  python3 dtcc_trs_panel.py compact --work /path/to/workdir \
      --out datacore/dtcc_trs_panel
  python3 dtcc_trs_panel.py events --panel datacore/dtcc_trs_panel
`build` is resumable: a day already written under <work>/days is skipped.
`compact` writes one gzip CSV per calendar quarter and manifest.json.
`events` writes events.json.gz beside the panel: the registered event list.

THE EVENT DEFINITION (registered 2026-11-25, before any return was read;
every constant below is part of it):
  trading day   a weekday that is not an NYSE full-day closure.
  value         an underlier's newt_usd on a trading day, plus its newt_usd
                on the non-trading days since the previous trading day (a
                weekend file is published before Monday's, so nothing is
                read early).
  covered       a trading day whose own vendor file was read.
  event         on a covered trading day t with at least EVENT_MIN_COVERED
                covered days among the EVENT_LOOKBACK trading days before
                it: value >= EVENT_FLOOR_USD, and fewer than EVENT_MAX_RANK
                of those covered days had a value at least as large (a day
                with no row is 0). That is "above the underlier's own
                trailing 60-day 95th percentile" made exact.
  cooldown      after an event the same underlier gives no event for
                EVENT_COOLDOWN trading days (the test's horizon), so one
                burst of trading is one event and windows do not overlap.
  known         the trading day after t: the file for t is published after
                00:00Z the next calendar day. A test enters no earlier.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

BASE_URL = "https://kgc0418-tdw-data-0.s3.amazonaws.com/sec/eod"

REQUIRED_COLUMNS = (
    "Action type", "Event type", "Execution Timestamp",
    "Notional amount-Leg 1", "Notional currency-Leg 1",
    "Underlier ID-Leg 1", "Underlier ID source-Leg 1",
)

PANEL_FIELDS = ("date", "cusip", "newt_n", "newt_usd", "capped_n", "fx_n",
                "late_n", "modi_n", "term_n", "other_n")
_COUNTERS = PANEL_FIELDS[2:]

# The dissemination cap. Measured 2026-11-25 over all 731 files: a capped
# value is written "250,000,000+", and the only unmarked values above it are
# placeholders (100,000,000,000,000,000,000 and the like, on 62 of 731 days).
# Summing one would make every underlier it touches an "event".
NOTIONAL_CAP_USD = 250_000_000

EVENT_LOOKBACK = 60
EVENT_MIN_COVERED = 55
EVENT_MAX_RANK = 3
EVENT_FLOOR_USD = 1_000_000
EVENT_COOLDOWN = 20
EVENTS_FILE = "events.json.gz"

# NYSE full-day closures before market_calendar.py's first year (2026).
# 2025-01-09 is the National Day of Mourning for President Carter.
_NYSE_CLOSED_BEFORE_2026 = frozenset(dt.date.fromisoformat(d) for d in (
    "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27",
    "2025-12-25",
))


def nyse_closures() -> frozenset:
    """Closures from 2024-11 on: the list above plus market_calendar.py's
    (beside this file; read, never edited)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import market_calendar
    return _NYSE_CLOSED_BEFORE_2026 | frozenset(market_calendar.ALL_HOLIDAYS)


def trading_days(first: dt.date, last: dt.date, closed: frozenset) -> List[dt.date]:
    out = []
    d = first
    while d <= last:
        if d.weekday() < 5 and d not in closed:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def detect_events(rows: Iterable[Dict[str, object]], days_ok: Iterable[str],
                  closed: frozenset) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """The registered event list from panel rows. Reads no price."""
    ok = sorted(dt.date.fromisoformat(d) for d in days_ok)
    if not ok:
        return [], {"trading_days": 0, "covered": 0, "testable_days": 0, "events": 0}
    tdays = trading_days(ok[0], ok[-1], closed)
    t_index = {d: i for i, d in enumerate(tdays)}
    ok_set = set(ok)
    covered = [d in ok_set for d in tdays]

    def fold(d: dt.date) -> Optional[int]:
        # the trading day a vendor day's rows are counted on
        while d not in t_index:
            d += dt.timedelta(days=1)
            if d > tdays[-1]:
                return None
        return t_index[d]

    values: Dict[str, Dict[int, int]] = {}
    counts: Dict[str, Dict[int, int]] = {}
    for r in rows:
        usd = int(r["newt_usd"])
        if usd <= 0:
            continue
        i = fold(dt.date.fromisoformat(str(r["date"])))
        if i is None:
            continue
        v = values.setdefault(str(r["cusip"]), {})
        v[i] = v.get(i, 0) + usd
        n = counts.setdefault(str(r["cusip"]), {})
        n[i] = n.get(i, 0) + int(r["newt_n"])

    # a day is testable when it is covered and enough of its lookback is
    testable: Dict[int, List[int]] = {}
    for i in range(EVENT_LOOKBACK, len(tdays)):
        if not covered[i]:
            continue
        window = [j for j in range(i - EVENT_LOOKBACK, i) if covered[j]]
        if len(window) >= EVENT_MIN_COVERED:
            testable[i] = window

    events: List[Dict[str, object]] = []
    for cusip in sorted(values):
        v = values[cusip]
        blocked_until = -1
        for i in sorted(v):
            if i <= blocked_until or i not in testable or v[i] < EVENT_FLOOR_USD:
                continue
            window = testable[i]
            at_least = sum(1 for j in window if v.get(j, 0) >= v[i])
            if at_least >= EVENT_MAX_RANK:
                continue
            blocked_until = i + EVENT_COOLDOWN
            events.append({
                "cusip": cusip,
                "date": tdays[i].isoformat(),
                "known": tdays[i + 1].isoformat() if i + 1 < len(tdays) else None,
                "newt_usd": v[i],
                "newt_n": counts[cusip].get(i, 0),
                "prior_max_usd": max((v.get(j, 0) for j in window), default=0),
                "prior_days_with_new": sum(1 for j in window if v.get(j, 0) > 0),
            })
    events.sort(key=lambda e: (e["date"], e["cusip"]))
    summary = {"trading_days": len(tdays), "covered": sum(covered),
               "testable_days": len(testable),
               "first_testable": tdays[min(testable)].isoformat() if testable else None,
               "last_testable": tdays[max(testable)].isoformat() if testable else None,
               "underliers_with_new": len(values), "events": len(events)}
    return events, summary


def write_events(panel_dir: str) -> Dict[str, object]:
    with open(os.path.join(panel_dir, "manifest.json")) as f:
        manifest = json.load(f)
    events, summary = detect_events(read_panel(panel_dir), manifest["days"].keys(), nyse_closures())
    doc = {
        "what": "registered event list for the DTCC equity-swap gate-2 test; see dtcc_trs_panel.py",
        "definition": {"lookback": EVENT_LOOKBACK, "min_covered": EVENT_MIN_COVERED,
                       "max_rank": EVENT_MAX_RANK, "floor_usd": EVENT_FLOOR_USD,
                       "cooldown": EVENT_COOLDOWN},
        "panel": {"first_day": manifest["first_day"], "last_day": manifest["last_day"],
                  "days_ok": manifest["days_ok"]},
        "summary": summary,
        "events": events,
    }
    # gzip with mtime 0: the same panel gives the same bytes
    tmp = os.path.join(panel_dir, EVENTS_FILE + ".tmp")
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(json.dumps(doc, sort_keys=True).encode())
    os.replace(tmp, os.path.join(panel_dir, EVENTS_FILE))
    return doc


def read_events(panel_dir: str) -> Dict[str, object]:
    with gzip.open(os.path.join(panel_dir, EVENTS_FILE), "rt") as f:
        return json.load(f)


def vendor_url(day: dt.date) -> str:
    return f"{BASE_URL}/SEC_CUMULATIVE_EQUITIES_{day:%Y_%m_%d}.zip"


def us_cusip(underlier_id: str, source: str) -> Optional[str]:
    """The 9-character CUSIP of a single US underlier, else None.
    Mirrors server/dtccSwaps.ts isUsUnderlier: a semicolon in either field
    is a basket and is never attributed."""
    if ";" in source or ";" in underlier_id:
        return None
    if source == "CUSIP":
        return underlier_id if len(underlier_id) == 9 else None
    if source == "ISIN" and underlier_id.startswith("US") and len(underlier_id) == 12:
        return underlier_id[2:11]
    return None


def parse_notional(raw: str) -> Tuple[Optional[int], bool]:
    """(whole dollars or None, capped). "250,000,000+" is (250000000, True):
    the vendor's cap marks a value at or above the figure."""
    s = raw.strip()
    if not s:
        return None, False
    capped = s.endswith("+")
    if capped:
        s = s[:-1]
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return None, False
    if v != v or v in (float("inf"), float("-inf")) or v < 0:
        return None, False
    return int(round(v)), capped


def aggregate(header: List[str], rows: Iterable[List[str]], file_day: dt.date
              ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """Fold one vendor file's rows into {cusip: counters} and day statistics.
    Raises ValueError when a required column is missing (schema drift is a
    failed day, never a silently wrong one)."""
    ix = {c: i for i, c in enumerate(header)}
    missing = [c for c in REQUIRED_COLUMNS if c not in ix]
    if missing:
        raise ValueError("schema drift: missing columns: " + ", ".join(missing))
    i_act, i_evt, i_exe = ix["Action type"], ix["Event type"], ix["Execution Timestamp"]
    i_amt, i_cur = ix["Notional amount-Leg 1"], ix["Notional currency-Leg 1"]
    i_id, i_src = ix["Underlier ID-Leg 1"], ix["Underlier ID source-Leg 1"]
    ncols = len(header)
    fresh_days = {file_day.isoformat(), (file_day - dt.timedelta(days=1)).isoformat()}
    out: Dict[str, Dict[str, int]] = {}
    stats = {"rows": 0, "malformed": 0, "baskets": 0, "us_rows": 0,
             "newt_notional_unreadable": 0, "newt_notional_over_cap": 0, "columns": ncols}
    for row in rows:
        stats["rows"] += 1
        if len(row) != ncols:
            stats["malformed"] += 1
            continue
        uid, src = row[i_id], row[i_src]
        if ";" in src or ";" in uid:
            stats["baskets"] += 1
            continue
        cusip = us_cusip(uid, src)
        if cusip is None:
            continue
        stats["us_rows"] += 1
        c = out.get(cusip)
        if c is None:
            c = out[cusip] = dict.fromkeys(_COUNTERS, 0)
        act = row[i_act]
        if act == "NEWT" and row[i_evt] == "TRAD":
            if row[i_exe][:10] not in fresh_days:
                c["late_n"] += 1
                continue
            c["newt_n"] += 1
            if row[i_cur] != "USD":
                c["fx_n"] += 1
                continue
            amount, capped = parse_notional(row[i_amt])
            if amount is None:
                stats["newt_notional_unreadable"] += 1
                continue
            if amount > NOTIONAL_CAP_USD and not capped:
                # the vendor caps at 250,000,000 and marks it with "+"; an
                # unmarked larger figure is a reporter's placeholder
                stats["newt_notional_over_cap"] += 1
                continue
            c["newt_usd"] += amount
            if capped:
                c["capped_n"] += 1
        elif act == "MODI":
            c["modi_n"] += 1
        elif act == "TERM":
            c["term_n"] += 1
        else:
            c["other_n"] += 1
    return out, stats


def read_zip_rows(path: str) -> Tuple[List[str], Iterator[List[str]]]:
    """Header and a row iterator over the zip's single CSV member. Python's
    zipfile reads the streamed (data-descriptor) members the 2024 and 2025
    files use."""
    z = zipfile.ZipFile(path)
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if len(names) != 1:
        raise ValueError(f"expected one CSV member, found {len(names)}")
    f = io.TextIOWrapper(z.open(names[0]), encoding="utf-8", errors="replace", newline="")
    reader = csv.reader(f)
    header = next(reader)
    return header, reader


def write_day(path: str, day: dt.date, per_cusip: Dict[str, Dict[str, int]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        for cusip in sorted(per_cusip):
            c = per_cusip[cusip]
            w.writerow([day.isoformat(), cusip] + [c[k] for k in _COUNTERS])
    os.replace(tmp, path)


def build_day(day_iso: str, work: str) -> Dict[str, object]:
    """Fetch, fold and write one vendor day. Returns its manifest line."""
    day = dt.date.fromisoformat(day_iso)
    days_dir = os.path.join(work, "days")
    stat_path = os.path.join(days_dir, day_iso + ".json")
    if os.path.exists(stat_path):
        with open(stat_path) as f:
            return json.load(f)
    line: Dict[str, object] = {"date": day_iso}
    fd, zpath = tempfile.mkstemp(suffix=".zip", dir=work)
    os.close(fd)
    try:
        try:
            req = urllib.request.Request(vendor_url(day), headers={"User-Agent": "research-panel/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(zpath, "wb") as o:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    o.write(chunk)
            line["bytes"] = os.path.getsize(zpath)
        except urllib.error.HTTPError as e:
            # 403: not published, or in cold storage (InvalidObjectState).
            line.update(status="unavailable", http=e.code)
            _write_json(stat_path, line)
            return line
        header, rows = read_zip_rows(zpath)
        per_cusip, stats = aggregate(header, rows, day)
        write_day(os.path.join(days_dir, day_iso + ".csv"), day, per_cusip)
        line.update(status="ok", underliers=len(per_cusip), **stats)
        _write_json(stat_path, line)
        return line
    except Exception as e:  # recorded, retried on the next run (no stat file)
        line.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
        return line
    finally:
        if os.path.exists(zpath):
            os.unlink(zpath)


def _write_json(path: str, obj: object) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, sort_keys=True)
    os.replace(tmp, path)


def daterange(a: dt.date, b: dt.date) -> List[str]:
    return [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def quarter_of(day_iso: str) -> str:
    return f"{day_iso[:4]}Q{(int(day_iso[5:7]) - 1) // 3 + 1}"


def compact(work: str, out: str) -> Dict[str, object]:
    """Quarterly gzip CSVs plus manifest.json, from <work>/days."""
    days_dir = os.path.join(work, "days")
    stats = []
    for name in sorted(os.listdir(days_dir)):
        if name.endswith(".json"):
            with open(os.path.join(days_dir, name)) as f:
                stats.append(json.load(f))
    os.makedirs(out, exist_ok=True)
    by_q: Dict[str, List[str]] = {}
    for s in stats:
        if s.get("status") == "ok":
            by_q.setdefault(quarter_of(s["date"]), []).append(s["date"])
    files = {}
    for q, dates in sorted(by_q.items()):
        name = f"panel_{q}.csv.gz"
        n = 0
        # mtime=0 and a fixed name: the same days give the same bytes.
        with open(os.path.join(out, name), "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
                gz.write((",".join(PANEL_FIELDS) + "\n").encode())
                for d in sorted(dates):
                    with open(os.path.join(days_dir, d + ".csv"), "rb") as f:
                        data = f.read()
                    n += data.count(b"\n")
                    gz.write(data)
        files[name] = {"days": len(dates), "rows": n}
    ok = [s for s in stats if s.get("status") == "ok"]
    manifest = {
        "what": "US-underlier equity swap activity per underlier per vendor day; see dtcc_trs_panel.py",
        "source": BASE_URL + "/SEC_CUMULATIVE_EQUITIES_YYYY_MM_DD.zip",
        "fields": list(PANEL_FIELDS),
        "first_day": min((s["date"] for s in ok), default=None),
        "last_day": max((s["date"] for s in ok), default=None),
        "days_ok": len(ok),
        "days_unavailable": sorted(s["date"] for s in stats if s.get("status") == "unavailable"),
        "rows": sum(f["rows"] for f in files.values()),
        "files": files,
        "days": {s["date"]: {k: s[k] for k in ("rows", "us_rows", "underliers", "baskets", "malformed",
                                                "newt_notional_unreadable", "newt_notional_over_cap", "columns", "bytes") if k in s}
                 for s in ok},
    }
    _write_json(os.path.join(out, "manifest.json"), manifest)
    return manifest


def read_panel(out: str) -> Iterator[Dict[str, object]]:
    """Rows of the compacted panel, counters as int. The registered test's
    only way in."""
    with open(os.path.join(out, "manifest.json")) as f:
        names = sorted(json.load(f)["files"])
    for name in names:
        with gzip.open(os.path.join(out, name), "rt", newline="") as f:
            for row in csv.DictReader(f):
                for k in _COUNTERS:
                    row[k] = int(row[k])
                yield row


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--from", dest="start", required=True)
    b.add_argument("--to", dest="end", required=True)
    b.add_argument("--work", required=True)
    b.add_argument("--workers", type=int, default=4)
    c = sub.add_parser("compact")
    c.add_argument("--work", required=True)
    c.add_argument("--out", required=True)
    e = sub.add_parser("events")
    e.add_argument("--panel", required=True)
    a = p.parse_args(argv)
    if a.cmd == "events":
        print(json.dumps(write_events(a.panel)["summary"]))
        return 0
    if a.cmd == "build":
        os.makedirs(os.path.join(a.work, "days"), exist_ok=True)
        days = daterange(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end))
        counts: Dict[str, int] = {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for line in ex.map(build_day, days, [a.work] * len(days)):
                st = str(line.get("status"))
                counts[st] = counts.get(st, 0) + 1
                if st != "ok":
                    print(json.dumps(line), flush=True)
                elif sum(counts.values()) % 25 == 0:
                    print(f"{line['date']} {counts}", flush=True)
        print("done", counts, flush=True)
        return 1 if counts.get("error") else 0
    m = compact(a.work, a.out)
    print(json.dumps({k: m[k] for k in ("first_day", "last_day", "days_ok", "rows")},))
    print("unavailable:", len(m["days_unavailable"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
