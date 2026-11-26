#!/usr/bin/env python3
"""dtcc_trs_panel_job.py: the server builds and keeps the DTCC equity-swap
panel on its own volume (open_questions.md (aq1), first step; 2026-11-26).

WHY THE SERVER BUILDS IT. The registered gate-2 test for
`dtcc_sbsdr_equity_swaps` (open_questions.md 2026-11-25, "GATE 2
PRE-REGISTERED") needs the event panel beside the price panel, and prices
live only on the server's volume. The image ships root-level Python and not
datacore/, so the panel a session built is not on the server: this job
builds it there with the same script, `dtcc_trs_panel.py`, and checks the
event list it gets against the one that was registered.

WHY NOW AND WHY OLDEST FIRST. The vendor stops serving a dated file two
years after it is written, one file a day. The first testable event day
(2025-02-25) needs the 60 trading days before it, which are the oldest
files still readable. A build that starts later reads fewer of them.

WHAT ONE RUN DOES
  1. The days to hold: from max(PANEL_FLOOR, UTC today - 731 days) to UTC
     yesterday. A day with a stat file under <root>/days is held.
  2. Reads the days not held, oldest first, one at a time, until the time
     budget or the day limit is spent. The vendor zip is held in the
     container's temporary directory, never on the volume. Each day file is
     gzipped after it is written (about 100 MB of plain CSV for two years
     becomes about 25 MB).
  3. A day the vendor refuses is recorded as unavailable (cold storage)
     unless it is one of the newest RECENT_DAYS days, where a refusal means
     "not published yet": that one is asked again on the next run.
  4. When every day is held and something new was read (or no event list
     exists), writes the compacted panel and the event list under
     <root>/panel and compares the events up to the registered list's last
     day with the registered list, month by month.

  5. SECURITY TYPES (open_questions.md (as2) step 1, 2026-11-26). Once an
     event list exists, asks OpenFIGI what each panel CUSIP is (cusip_figi.py:
     common stock, fund, preferred, ...; the registered filter's item 3
     needs it) for the CUSIPs not asked before: the event underliers first,
     then the rest by how many panel days they have. It uses what is left of
     the run's total time, 10 CUSIPs a request, 25 requests a minute, and
     keeps every answer in <root>/figi/cusip_figi.jsonl. It reads no price.

  6. THE DATED TICKER (open_questions.md (at2), 2026-11-26). Today's ticker
     cannot carry a 2025 event to its prices when the company was bought or
     renamed since. Before the type step, the run reads the SEC
     fails-to-deliver half-month files it does not hold (cusip_ftd.py: the
     CUSIP beside the symbol as printed on each settlement day), for the
     periods a 45-day window around the events can touch, for at most
     FTD_BUDGET_S seconds. After the type step it counts how many events
     and underliers that record covers (the join's gate 1; no price).

WHAT IT DOES NOT DO. It reads no price and computes no return; the test
itself is not here. Nothing in the trading bot reads its output.

OUTPUT: <DATA_DIR>/dtcc_trs_panel/{days/, panel/, figi/, ftd/, last_run.json}. The
last stdout line is one JSON summary for the scheduler
(server/dtccTrsPanel.ts).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from typing import Callable, Dict, Iterable, List, Optional

import cusip_figi
import cusip_ftd
import dtcc_trs_panel as panel

SCHEMA = "dtcc_trs_panel_job/1"
OUT_SUBDIR = "dtcc_trs_panel"
# The first vendor day the registered panel holds.
PANEL_FLOOR = dt.date(2024, 11, 24)
# On UTC day T the file dated T-731 was the oldest readable one (measured
# 2026-11-25 13:35Z; at 00:12Z on 2026-11-26 the file dated T-732 still
# answered, so the move to cold storage is not at midnight). The job asks
# from T-732 and lets the vendor's answer decide.
READABLE_DAYS = 732
# A refusal on one of the newest days means "not published yet".
RECENT_DAYS = 4
# Stop a run after this many days in a row that could not be read.
MAX_CONSECUTIVE_ERRORS = 3
# The security-type record (cusip_figi.py) lives beside the panel.
FIGI_SUBDIR = "figi"
# Under this many seconds left in the run, the step waits for the next run.
FIGI_MIN_BUDGET_S = 20.0
# The dated CUSIP-to-ticker record (cusip_ftd.py) lives beside the panel.
FTD_SUBDIR = "ftd"
# The most one run spends reading SEC files: about 50 half-month files at the
# first run (one request a second), two a month after that. The type step
# gets the rest of the run.
FTD_BUDGET_S = 200.0

# The registered event list (datacore/dtcc_trs_panel/events.json.gz as
# committed 2026-11-25, v1.0.1554), as a digest: per month, the number of
# events and the first 16 hex characters of the SHA-256 of its lines
# "date,cusip,newt_usd\n" in (date, cusip) order. test_dtcc_trs_panel_job.py
# recomputes this from the committed file, so the two cannot drift apart.
REGISTERED_EVENTS = {
    "through": "2026-11-24",
    "events": 27869,
    "by_month": {
        "2025-02": [455, "48f53ee124f06759"], "2025-03": [1238, "234fabfe6f65a751"],
        "2025-04": [1334, "05024e8d4777f6c5"], "2025-05": [1114, "9ac69f7056e01ee1"],
        "2025-06": [1304, "f10c0b68b14fe0c5"], "2025-07": [1157, "e91e513214c6e762"],
        "2025-08": [1279, "7a7bbd0415729528"], "2025-09": [1437, "aa8769fa79782a87"],
        "2025-10": [1359, "453503b799643449"], "2025-11": [980, "c8b475485cf2f735"],
        "2025-12": [1488, "09fcfdaa638c72af"], "2026-01": [1315, "65e67d8ff37c4c9c"],
        "2026-02": [1294, "912ee9ff99d49631"], "2026-03": [1296, "448db7e2e1dad3d6"],
        "2026-04": [1224, "5fe92af428bd5e3d"], "2026-05": [1568, "4d62cac7352fdf3a"],
        "2026-06": [1642, "50522b1302054356"], "2026-07": [1149, "27d3e473caeeafc2"],
        "2026-08": [1345, "19a3c47cf5186d04"], "2026-09": [1229, "90df8ea66bc308c8"],
        "2026-10": [1456, "b72866500cfd36e2"], "2026-11": [1206, "8446ac87e66046b0"],
    },
}


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def digest_by_month(events: Iterable[Dict[str, object]], through: str) -> Dict[str, List[object]]:
    """{month: [count, digest]} over the events dated on or before `through`."""
    by: Dict[str, List[Dict[str, object]]] = {}
    for e in events:
        d = str(e["date"])
        if d <= through:
            by.setdefault(d[:7], []).append(e)
    out: Dict[str, List[object]] = {}
    for month in sorted(by):
        h = hashlib.sha256()
        for e in sorted(by[month], key=lambda x: (str(x["date"]), str(x["cusip"]))):
            h.update(f"{e['date']},{e['cusip']},{e['newt_usd']}\n".encode())
        out[month] = [len(by[month]), h.hexdigest()[:16]]
    return out


def compare_with_registered(events: Iterable[Dict[str, object]],
                            registered: Dict[str, object] = REGISTERED_EVENTS) -> Dict[str, object]:
    """Is the event list built here the registered one, up to its last day?
    A month that differs is named with both counts: an unequal list is a
    finding for the test's reader, never something this job repairs."""
    here = digest_by_month(events, str(registered["through"]))
    reg = registered["by_month"]
    differing = []
    for month in sorted(set(here) | set(reg)):
        a, b = here.get(month), reg.get(month)
        if a is None or b is None or list(a) != list(b):
            differing.append({"month": month, "here": a[0] if a else 0, "registered": b[0] if b else 0})
    return {
        "through": registered["through"],
        "registered": registered["events"],
        "here": sum(int(v[0]) for v in here.values()),
        "match": not differing,
        "months_differing": differing,
    }


def wanted_days(today: dt.date, floor: dt.date = PANEL_FLOOR) -> List[str]:
    first = max(floor, today - dt.timedelta(days=READABLE_DAYS))
    last = today - dt.timedelta(days=1)
    return panel.daterange(first, last) if last >= first else []


def held(days_dir: str, day_iso: str) -> bool:
    return os.path.exists(os.path.join(days_dir, day_iso + ".json"))


def gzip_day_file(days_dir: str, day_iso: str) -> None:
    """<day>.csv -> <day>.csv.gz (mtime 0), then the plain file is removed."""
    plain = os.path.join(days_dir, day_iso + ".csv")
    if not os.path.exists(plain):
        return
    tmp = plain + ".gz.tmp"
    with open(plain, "rb") as src, open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            shutil.copyfileobj(src, gz)
    os.replace(tmp, plain + ".gz")
    os.unlink(plain)


def day_counts(days_dir: str) -> Dict[str, object]:
    ok, unavailable, unreadable = [], [], []
    if os.path.isdir(days_dir):
        for name in os.listdir(days_dir):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(days_dir, name)) as f:
                    st = json.load(f).get("status")
            except (OSError, ValueError) as e:
                print(f"[dtcc_trs_panel_job] unreadable stat file {name}: {e}", file=sys.stderr)
                st = None
            (ok if st == "ok" else unavailable if st == "unavailable" else unreadable).append(name[:-5])
    return {"days_ok": len(ok), "first_day": min(ok, default=None), "last_day": max(ok, default=None),
            "days_unavailable": len(unavailable), "stat_unreadable": len(unreadable),
            "oldest_unavailable": min(unavailable, default=None), "newest_unavailable": max(unavailable, default=None)}


def dir_bytes(root: str) -> int:
    total = 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            fp = os.path.join(base, name)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def figi_wanted(root: str) -> Dict[str, object]:
    """The CUSIPs to ask about, in order: the event underliers (by their
    first event), then every other panel CUSIP by its number of panel rows,
    most first. Kept in <root>/figi/wanted.json and rebuilt only when the
    event list's file changes (a pass over the panel is about 2.5 million
    rows)."""
    panel_dir = os.path.join(root, "panel")
    events_path = os.path.join(panel_dir, panel.EVENTS_FILE)
    st = os.stat(events_path)
    key = f"{st.st_size}:{int(st.st_mtime)}"
    path = os.path.join(root, FIGI_SUBDIR, "wanted.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                doc = json.load(f)
            if doc.get("key") == key and isinstance(doc.get("cusips"), list):
                return doc
        except (OSError, ValueError) as e:
            print(f"[dtcc_trs_panel_job] {path} unreadable, rebuilt: {e}", file=sys.stderr)
    order: List[str] = []
    seen = set()
    for e in sorted(panel.read_events(panel_dir)["events"], key=lambda x: (str(x["date"]), str(x["cusip"]))):
        c = str(e["cusip"])
        if c not in seen:
            seen.add(c)
            order.append(c)
    n_event = len(order)
    rows: Dict[str, int] = {}
    for r in panel.read_panel(panel_dir):
        c = str(r["cusip"])
        if c not in seen:
            rows[c] = rows.get(c, 0) + 1
    order.extend(sorted(rows, key=lambda c: (-rows[c], c)))
    doc = {"key": key, "event_cusips": n_event, "cusips": order}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_json(path, doc)
    return doc


def figi_step(root: str, budget_s: float,
              resolve: Callable[..., Dict[str, object]] = cusip_figi.resolve) -> Dict[str, object]:
    """Ask for the panel CUSIPs not asked before, within `budget_s`; then
    count the classes of what is held. Never raises."""
    try:
        doc = figi_wanted(root)
        cusips = [str(c) for c in doc["cusips"]]
        n_event = int(doc["event_cusips"])
        cache_path = os.path.join(root, FIGI_SUBDIR, cusip_figi.CACHE_FILE)
        out = resolve(cusips, cache_path, budget_s=budget_s, api_key=os.environ.get("OPENFIGI_API_KEY") or None)
        cache = cusip_figi.read_cache(cache_path)
        out["event_cusips"] = n_event
        out["event_cusips_held"] = sum(1 for c in cusips[:n_event] if c in cache)
        out["classes_event_cusips"] = cusip_figi.class_counts(cache, [c for c in cusips[:n_event] if c in cache])
        out["classes_all"] = cusip_figi.class_counts(cache, [c for c in cusips if c in cache])
        return out
    except Exception as e:  # noqa: BLE001 — the panel is kept; the run reads "error" with the reason
        return {"status": "error", "last_error": f"{type(e).__name__}: {str(e)[:160]}"}


def ftd_fetch_step(root: str, budget_s: float, today: Optional[dt.date] = None,
                   build: Callable[..., Dict[str, object]] = cusip_ftd.build) -> Dict[str, object]:
    """Read the SEC half-month files the events' 45-day windows can touch
    and that are not held yet, within `budget_s`. Never raises."""
    try:
        today = today or dt.datetime.now(dt.timezone.utc).date()
        days = [str(e["date"]) for e in panel.read_events(os.path.join(root, "panel"))["events"]]
        if not days:
            return {"status": "ok", "wanted": 0, "held": 0, "owed": 0, "read": 0, "requests": 0}
        wanted = cusip_ftd.wanted_periods(dt.date.fromisoformat(min(days)), dt.date.fromisoformat(max(days)), today)
        out = build(os.path.join(root, FTD_SUBDIR), wanted, today, budget_s=budget_s)
        out["bytes"] = dir_bytes(os.path.join(root, FTD_SUBDIR))
        return out
    except Exception as e:  # noqa: BLE001 — the panel is kept; the run reads "error" with the reason
        return {"status": "error", "last_error": f"{type(e).__name__}: {str(e)[:160]}"}


def ftd_coverage_step(root: str) -> Dict[str, object]:
    """How much of the event list the dated record covers (cusip_ftd.coverage:
    counts only, no price), from what is on the volume now. Never raises."""
    try:
        events = panel.read_events(os.path.join(root, "panel"))["events"]
        cache = cusip_figi.read_cache(os.path.join(root, FIGI_SUBDIR, cusip_figi.CACHE_FILE))
        spans = cusip_ftd.read_spans(os.path.join(root, FTD_SUBDIR), {str(e["cusip"]) for e in events})
        return cusip_ftd.coverage(events, cache, spans)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:160]}"}


def run(root: str, today: Optional[dt.date] = None, budget_s: float = 600.0, max_days: int = 400,
        max_rss_mb: float = 600.0, code_version: str = "unknown", tmp_dir: Optional[str] = None,
        build: Callable[..., Dict[str, object]] = panel.build_day,
        clock: Callable[[], float] = time.time, total_s: float = 0.0,
        figi: Callable[[str, float], Dict[str, object]] = figi_step,
        ftd_fetch: Callable[[str, float], Dict[str, object]] = ftd_fetch_step,
        ftd_coverage: Callable[[str], Dict[str, object]] = ftd_coverage_step) -> Dict[str, object]:
    t0 = clock()
    today = today or dt.datetime.now(dt.timezone.utc).date()
    days_dir = os.path.join(root, "days")
    panel_dir = os.path.join(root, "panel")
    os.makedirs(days_dir, exist_ok=True)
    tmp_dir = tmp_dir or os.path.join(tempfile.gettempdir(), "voltrade_dtcc_trs_panel")
    os.makedirs(tmp_dir, exist_ok=True)
    # a zip left by a killed run
    for name in os.listdir(tmp_dir):
        if name.endswith(".zip"):
            try:
                os.unlink(os.path.join(tmp_dir, name))
            except OSError as e:
                print(f"[dtcc_trs_panel_job] could not remove {name}: {e}", file=sys.stderr)

    wanted = wanted_days(today)
    todo = [d for d in wanted if not held(days_dir, d)]
    recent_from = (today - dt.timedelta(days=RECENT_DAYS)).isoformat()
    read = unavailable_now = not_published = errors = consecutive = 0
    last_error = None
    stopped = None
    for day in todo:
        if read + unavailable_now + not_published + errors >= max_days:
            stopped = "day limit"
            break
        if clock() - t0 >= budget_s:
            stopped = "time budget"
            break
        if _rss_mb() > max_rss_mb:
            stopped = f"peak memory {_rss_mb():.0f} MB over the {max_rss_mb:.0f} MB budget"
            break
        line = build(day, root, tmp_dir)
        status = line.get("status")
        if status == "ok":
            gzip_day_file(days_dir, day)
            read += 1
            consecutive = 0
        elif status == "unavailable":
            consecutive = 0
            if day >= recent_from:
                # not published yet: forget the answer so the next run asks again
                stat = os.path.join(days_dir, day + ".json")
                if os.path.exists(stat):
                    os.unlink(stat)
                not_published += 1
            else:
                unavailable_now += 1
        else:
            errors += 1
            consecutive += 1
            last_error = f"{day}: {str(line.get('error'))[:120]}"
            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                stopped = f"{consecutive} days in a row could not be read"
                break

    pending = sum(1 for d in wanted if not held(days_dir, d))
    # days the vendor has not published yet are not a backlog
    backlog = sum(1 for d in wanted if d < recent_from and not held(days_dir, d))
    out: Dict[str, object] = {
        "schema": SCHEMA, "code_version": code_version,
        "wanted": len(wanted), "read": read, "unavailable_new": unavailable_now,
        "not_published": not_published, "errors": errors, "pending": pending, "backlog": backlog,
        "stopped": stopped,
    }
    if last_error:
        out["last_error"] = last_error
    out.update(day_counts(days_dir))

    events_path = os.path.join(panel_dir, panel.EVENTS_FILE)
    events_summary = None
    if backlog == 0 and out["days_ok"] and (read or not os.path.exists(events_path)):
        try:
            panel.compact(root, panel_dir)
            doc = panel.write_events(panel_dir)
            events_summary = {"events": doc["summary"]["events"],
                              "first_testable": doc["summary"].get("first_testable"),
                              "last_testable": doc["summary"].get("last_testable"),
                              "registered": compare_with_registered(doc["events"])}
            _write_json(os.path.join(panel_dir, "events_check.json"), events_summary)
        except Exception as e:  # noqa: BLE001 — the day files are kept; the run reads "error"
            out["events_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    elif os.path.exists(os.path.join(panel_dir, "events_check.json")):
        try:
            with open(os.path.join(panel_dir, "events_check.json")) as f:
                events_summary = json.load(f)
        except (OSError, ValueError) as e:
            out["events_error"] = f"events_check.json unreadable: {str(e)[:120]}"
    out["events"] = events_summary

    # The dated ticker record first, for a bounded part of what is left of the
    # run's total time (0 = off); the type step keeps at least its minimum.
    ftd_left = min(FTD_BUDGET_S, total_s - (clock() - t0) - FIGI_MIN_BUDGET_S)
    if total_s > 0 and ftd_left >= FIGI_MIN_BUDGET_S and os.path.exists(events_path):
        out["ftd"] = ftd_fetch(root, ftd_left)
    else:
        out["ftd"] = None

    # Security types, with what is left of the run's total time (0 = off).
    figi_left = total_s - (clock() - t0)
    if total_s > 0 and figi_left >= FIGI_MIN_BUDGET_S and os.path.exists(events_path):
        out["figi"] = figi(root, figi_left)
    else:
        out["figi"] = None

    # The join's gate-1 count, from what the two steps above left on the volume.
    if out["ftd"] is not None and out["ftd"].get("held"):
        out["ftd"]["coverage"] = ftd_coverage(root)

    figi_failed = bool(out["figi"]) and out["figi"].get("status") == "error"
    ftd_failed = bool(out["ftd"]) and out["ftd"].get("status") == "error"
    out["status"] = "error" if (errors or out.get("events_error") or figi_failed or ftd_failed) else "ok"
    out["bytes"] = dir_bytes(root)
    out["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out["elapsed_s"] = round(clock() - t0, 1)
    out["peak_rss_mb"] = round(_rss_mb(), 1)
    _write_json(os.path.join(root, "last_run.json"), out)
    return out


def _write_json(path: str, obj: object) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, sort_keys=True)
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--max-rss-mb", type=float, default=600.0)
    ap.add_argument("--budget-s", type=float, default=600.0)
    ap.add_argument("--max-days", type=int, default=400)
    ap.add_argument("--total-s", type=float, default=0.0,
                    help="the whole run's seconds; what the day reads leave goes to the security-type step (0: no such step)")
    ap.add_argument("--code-version", default="unknown")
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args(argv)
    if a.nice:
        try:
            os.nice(a.nice)
        except OSError as e:
            print(f"[dtcc_trs_panel_job] os.nice failed: {e}", file=sys.stderr)
    data_dir = os.environ.get("DATA_DIR") or ("/data/voltrade" if os.path.exists("/data") else "/tmp")
    root = a.out_root or os.path.join(data_dir, OUT_SUBDIR)
    try:
        out = run(root, budget_s=a.budget_s, max_days=a.max_days, max_rss_mb=a.max_rss_mb, code_version=a.code_version,
                  total_s=a.total_s)
    except Exception as e:  # noqa: BLE001 — one JSON line for the scheduler; day files already written are kept
        out = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
    print(json.dumps(out))
    return 0 if out["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
