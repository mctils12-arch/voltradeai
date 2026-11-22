#!/usr/bin/env python3
"""filings13f_clusters.py — which securities several focused 13F managers
newly reported in the same quarter, read from the 13F archive on the volume.

ROOT: sec_edgar_13f_institutional_clustering (datacore/signal_ladder.json).
The hypothesis was filed 2026-07-05, before the archive's first row: several
capacity-constrained managers newly entering the same thinly held name precedes
60-90 day outperformance. It could not be looked at until the archive held two
consecutive quarters per manager; the Q3-2026 season closed 2026-11-16.

THIS FILE IS THE DETECTOR ONLY. It reads no price and computes no return. It
writes the list of cluster events, each dated by the day the K-th new entrant's
filing became public, so the list exists before any forward return does. The
gate-2 test that will read this list is pre-registered in
research/open_questions.md (2026-11-22) and is a later change.

DEFINITIONS (fixed here, before any outcome exists; changing one is a new
variant and is counted as one):
  manager      managerCik (the filer's CIK when absent).
  period       periodOfReport, a calendar quarter end. The base period of P is
               the quarter end before it.
  full filing  a 13F-HR or 13F-HR/A the archive holds with its holdings table
               (the archive keeps tables of at most 250 rows; larger managers
               are summary-only and are not in this study, by design).
  position     an archived holdings row with a 9-character CUSIP, no put/call
               marker and share type SH. A manager's holdings for a period are
               the UNION over its full filings for that period (an amendment
               may add rows or restate; the union never invents an entry).
  base filings only those filed on or before P's quarter end count as the
               base, so nothing filed during P's own season changes what was
               "held before".
  new entry    manager M holds security S in P; M has a base; and M's base has
               neither S's issuer (first six CUSIP characters) nor a holding
               with the same normalized issuer name (a CUSIP change is not a
               purchase).
  prior holders  managers with a base whose base holds S's issuer.
  cluster      a security with at least MIN_NEW_ENTRANTS new entries in P.
  near miss    a thinly held security (class "thin" below) with at least one
               new entry and fewer than MIN_NEW_ENTRANTS, dated by its last new
               entry: the comparison set for the test (same breadth, same
               season, fewer entrants).
  event date   the filing date of the MIN_NEW_ENTRANTS-th earliest new entry
               (filing dates are days; a test must enter no earlier than the
               next trading day's close).
  class        "thin"     1 <= prior holders <= THIN_PRIOR_MAX  (the hypothesis)
               "no_prior" prior holders == 0 (IPO, spin-off or new identifier:
                          every holder is new by construction; not the hypothesis)
               "broad"    prior holders > THIN_PRIOR_MAX (recorded, not tested)

OUTPUT (DATA_DIR/filings13f_clusters; public SEC filings data only):
  latest.json            counts per period pair, the newest pair's largest
                         clusters, coverage, definitions
  events_<P>.json        every cluster of period P as computed this run
  controls_<P>.json      the thin near misses of period P as computed this run
  first_seen.jsonl       append-only: one line the first time a (period,
                         security) is a cluster, with the day we detected it
  last_run.json          status of the last attempt

The last stdout line is one JSON object for the scheduler
(server/filings13fClusters.ts).

  python3 filings13f_clusters.py --run [--nice 19] [--max-rss-mb 700]
      [--archive-dir DIR] [--out-dir DIR] [--code-version V]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import re
import resource
import sys
import time

SCHEMA = "filings13f_clusters/1"
OUT_SUBDIR = "filings13f_clusters"
ARCHIVE_KIND = "filings13f"

# Day files older than this are not read. Memory is then set by three filing
# seasons and does not grow with the archive (measured 2026-11-22 on a synthetic
# archive of 19,497 filings in three quarters: 9 s, 426 MB peak). A quarter
# pair is recomputed while its base quarter's whole season is inside the
# window, about 110 days past the pair's filing deadline; after that its
# events and controls files stay on disk as last written.
LOOKBACK_DAYS = 250
MIN_NEW_ENTRANTS = 3
THIN_PRIOR_MAX = 10
TOP_N = 25
FORMS = ("13F-HR", "13F-HR/A")

_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_QUARTER_ENDS = {"03-31", "06-30", "09-30", "12-31"}
_NAME_NOISE = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|LP|LLC|SA|NV|AG|"
    r"HLDGS?|HOLDINGS?|GROUP|GRP|TR|TRUST|THE|NEW|CL|CLASS|COM|ORD|SHS|ADR|SPONSORED|DEL)\b")


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def prev_quarter_end(period: str) -> str:
    y, m = int(period[:4]), int(period[5:7])
    return {3: f"{y - 1}-12-31", 6: f"{y}-03-31", 9: f"{y}-06-30", 12: f"{y}-09-30"}[m]


def normalize_issuer(name) -> str:
    s = re.sub(r"[^A-Z0-9 ]", " ", str(name or "").upper())
    s = _NAME_NOISE.sub(" ", s)
    return " ".join(s.split())


def default_archive_dir() -> str:
    data_dir = os.environ.get("DATA_DIR") or ("/data/voltrade" if os.path.exists("/data") else "/tmp")
    return os.path.join(data_dir, "datacore_archive", ARCHIVE_KIND)


def window_start(today: dt.date) -> str:
    return (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()


def iter_archive_rows(dir_path: str, counts: dict, since_day: str = ""):
    """Every row of every day file from `since_day` on, oldest file first. A
    line that is not JSON is counted, never raised: one bad line must not hide
    a quarter."""
    try:
        names = sorted(n for n in os.listdir(dir_path) if _DAY_FILE.match(n))
    except FileNotFoundError:
        return
    for name in names:
        if name[:10] < since_day:
            counts["day_files_before_window"] += 1
            continue
        fp = os.path.join(dir_path, name)
        opener = gzip.open if name.endswith(".gz") else open
        counts["day_files"] += 1
        day = name[:10]
        counts["first_day"] = min(counts["first_day"] or day, day)
        counts["last_day"] = max(counts["last_day"] or day, day)
        try:
            with opener(fp, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        counts["malformed_lines"] += 1
                        continue
                    if isinstance(row, dict):
                        yield row
                    else:
                        counts["malformed_lines"] += 1
        except (OSError, EOFError) as e:  # a day file cut short by a restart: keep what was read, say so
            counts["unreadable_files"] += 1
            print(f"[filings13f_clusters] {name}: {type(e).__name__}: {e}", file=sys.stderr)


def new_counts() -> dict:
    return {"day_files": 0, "day_files_before_window": 0, "first_day": None, "last_day": None,
            "malformed_lines": 0, "unreadable_files": 0, "rows": 0, "duplicate_accessions": 0, "no_manager_or_period": 0, "not_quarter_end": 0,
            "other_form": 0, "summary_only": 0, "full_filings": 0, "positions": 0,
            "rows_not_positions": 0, "filed_before_period_end": 0}


def load_holdings(rows, counts: dict) -> tuple[dict, dict]:
    """(periods, names). periods[P][manager] = {cusip9: (first_filed_day, value)}
    (union over the manager's full filings for P; the earliest filing that
    lists a security dates it; the value is that filing's). names[cusip9] is
    one issuer name per security, the first one read: a name per position is
    what a million positions cannot afford in memory."""
    periods: dict = {}
    names: dict = {}
    seen = set()
    for r in rows:
        counts["rows"] += 1
        acc = r.get("accession")
        if acc:
            if acc in seen:
                counts["duplicate_accessions"] += 1
                continue
            seen.add(acc)
        mgr = str(r.get("managerCik") or r.get("cik") or "").lstrip("0")
        period = str(r.get("periodOfReport") or "")
        filed = str(r.get("filedAt") or "")
        if not mgr or not _DATE.match(period) or not _DATE.match(filed):
            counts["no_manager_or_period"] += 1
            continue
        period, filed = period[:10], filed[:10]
        if period[5:] not in _QUARTER_ENDS:
            counts["not_quarter_end"] += 1
            continue
        form = r.get("form") or r.get("submissionType")
        if form is not None and form not in FORMS:
            counts["other_form"] += 1
            continue
        if filed <= period:
            counts["filed_before_period_end"] += 1
            continue
        holdings = r.get("holdings")
        if r.get("holdingsOmitted") or not isinstance(holdings, list) or not holdings:
            counts["summary_only"] += 1
            continue
        counts["full_filings"] += 1
        filing: dict = {}  # one filing's rows per security (several rows when other managers are split out)
        for h in holdings:
            if not isinstance(h, dict):
                counts["rows_not_positions"] += 1
                continue
            cusip = str(h.get("cusip") or "").strip().upper()
            if len(cusip) != 9 or h.get("putCall") or str(h.get("sharesType") or "SH").upper() != "SH":
                counts["rows_not_positions"] += 1
                continue
            cusip = sys.intern(cusip)
            value = h.get("value") if isinstance(h.get("value"), (int, float)) else 0
            filing[cusip] = filing.get(cusip, 0) + value
            if cusip not in names and h.get("issuer"):
                names[cusip] = str(h.get("issuer"))
        book = periods.setdefault(period, {}).setdefault(mgr, {})
        for cusip, value in filing.items():
            cur = book.get(cusip)
            if cur is None:
                counts["positions"] += 1
                book[cusip] = (filed, value)
            elif filed < cur[0]:
                book[cusip] = (filed, value)
    return periods, names


def classify(prior_holders: int) -> str:
    if prior_holders == 0:
        return "no_prior"
    return "thin" if prior_holders <= THIN_PRIOR_MAX else "broad"


def detect_pair(periods: dict, names: dict, period: str) -> dict | None:
    """Clusters of `period` against its base quarter. None when the archive has
    no base filings for it."""
    base_p = prev_quarter_end(period)
    cur = periods.get(period) or {}
    base_books = periods.get(base_p) or {}
    if not base_books or not cur:
        return None

    norm_cache: dict = {}
    issuer_cache: dict = {}

    def norm(cusip: str) -> str:
        n = norm_cache.get(cusip)
        if n is None:
            n = norm_cache[cusip] = normalize_issuer(names.get(cusip))
        return n

    def issuer_of(cusip: str) -> str:
        i = issuer_cache.get(cusip)
        if i is None:
            i = issuer_cache[cusip] = sys.intern(cusip[:6])
        return i

    # a base filing filed after `period` ended is not a base (see module doc)
    prior_holders: dict = {}
    base_keys: dict = {}
    late_base_managers = 0
    for mgr, book in base_books.items():
        kept = [c for c, v in book.items() if v[0] <= period]
        if not kept:
            late_base_managers += 1
            continue
        issuers = {issuer_of(c) for c in kept}
        held_names = {n for n in (norm(c) for c in kept) if n}
        base_keys[mgr] = (issuers, held_names)
        for i in issuers:
            prior_holders[i] = prior_holders.get(i, 0) + 1
    if not base_keys:
        return None

    entrants: dict = {}
    unknown_base: dict = {}
    exits: dict = {}
    both = 0
    for mgr, book in cur.items():
        keys = base_keys.get(mgr)
        if keys is None:
            for c in book:
                unknown_base[c] = unknown_base.get(c, 0) + 1
            continue
        both += 1
        issuers, held_names = keys
        held_now = {issuer_of(c) for c in book}
        for i in issuers - held_now:
            exits[i] = exits.get(i, 0) + 1
        for c, (filed, value) in book.items():
            if issuer_of(c) in issuers:
                continue
            n = norm(c)
            if n and n in held_names:
                continue
            entrants.setdefault(c, []).append((filed, value))

    events, controls = [], []
    for c, lst in entrants.items():
        lst.sort()
        ph = prior_holders.get(c[:6], 0)
        if len(lst) < MIN_NEW_ENTRANTS:
            if classify(ph) == "thin":  # the comparison set: same breadth, fewer entrants
                controls.append({"cusip": c, "issuer": names.get(c), "date": lst[-1][0],
                                 "new_entrants": len(lst), "prior_holders": ph})
            continue
        events.append({
            "period": period, "cusip": c, "issuer": names.get(c),
            "event_date": lst[MIN_NEW_ENTRANTS - 1][0],
            "new_entrants": len(lst), "prior_holders": ph, "class": classify(ph),
            "first_entry_filed": lst[0][0], "last_entry_filed": lst[-1][0],
            "new_value_usd": int(sum(v for _, v in lst)),
            "exits": exits.get(c[:6], 0),
            "holders_without_base": unknown_base.get(c, 0),
        })
    events.sort(key=lambda e: (-e["new_entrants"], e["prior_holders"], e["cusip"]))
    by_class = {"thin": 0, "no_prior": 0, "broad": 0}
    for e in events:
        by_class[e["class"]] += 1
    return {
        "period": period, "base_period": base_p,
        "managers": {"base": len(base_keys), "period": len(cur), "both": both,
                     "period_without_base": len(cur) - both, "base_filed_late_ignored": late_base_managers},
        "securities_with_a_new_entry": len(entrants),
        "clusters": len(events), "clusters_by_class": by_class,
        "thin_near_misses": len(controls),
        "event_dates": {"first": min((e["event_date"] for e in events), default=None),
                        "last": max((e["event_date"] for e in events), default=None)},
        "events": events,
        "controls": sorted(controls, key=lambda r: r["cusip"]),
    }


def detect(periods: dict, names: dict, since_day: str = "") -> list[dict]:
    """Every quarter pair whose base quarter ended inside the window: the base
    season opens the day after the quarter ends, so the window holds all of it."""
    out = []
    for period in sorted(periods):
        if prev_quarter_end(period) < since_day:
            continue
        pair = detect_pair(periods, names, period)
        if pair:
            out.append(pair)
    return out


def _read_json(fp: str):
    try:
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        print(f"[filings13f_clusters] unreadable {fp}: {e}", file=sys.stderr)
        return None


def _write_atomic(fp: str, payload) -> None:
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    tmp = f"{fp}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, fp)


def read_first_seen(fp: str) -> set:
    keys = set()
    if not os.path.exists(fp):
        return keys
    torn = 0
    with open(fp, encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                keys.add((j["period"], j["cusip"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                torn += 1  # a line cut by a restart; the next run appends that event again
    if torn:
        print(f"[filings13f_clusters] first_seen.jsonl: {torn} unreadable line(s) skipped", file=sys.stderr)
    return keys


def append_first_seen(fp: str, pairs: list[dict], now_iso: str, archive_last_day, code_version: str) -> int:
    known = read_first_seen(fp)
    lines = []
    for pair in pairs:
        for e in pair["events"]:
            if (e["period"], e["cusip"]) in known:
                continue
            lines.append(json.dumps({
                "period": e["period"], "cusip": e["cusip"], "issuer": e["issuer"], "event_date": e["event_date"],
                "new_entrants": e["new_entrants"], "prior_holders": e["prior_holders"], "class": e["class"],
                "detected_at": now_iso, "archive_last_day": archive_last_day, "code_version": code_version,
            }, separators=(",", ":")))
    if lines:
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return len(lines)


def definitions() -> dict:
    return {
        "min_new_entrants": MIN_NEW_ENTRANTS, "thin_prior_max": THIN_PRIOR_MAX,
        "new_entry": "the manager holds the security this quarter and its base-quarter filing has neither the "
                     "issuer (first six CUSIP characters) nor the same normalized issuer name",
        "event_date": f"filing date of new entry number {MIN_NEW_ENTRANTS}, in filing-date order",
        "near_miss": f"thinly held, 1 to {MIN_NEW_ENTRANTS - 1} new entries, dated by the last one: the comparison set",
        "classes": {"thin": f"1 to {THIN_PRIOR_MAX} prior holders (the hypothesis)",
                    "no_prior": "no prior holder: IPO, spin-off or new identifier",
                    "broad": f"more than {THIN_PRIOR_MAX} prior holders"},
        "managers": "13F filers whose archived table has at most 250 rows; larger managers are not in the study",
        "claim": "none: a count of as-filed records; no price was read and no return computed",
    }


def run(out_dir: str, archive_dir: str | None, code_version: str, max_rss_mb: float,
        today: dt.date | None = None) -> dict:
    t0 = time.time()
    d = archive_dir or default_archive_dir()
    counts = new_counts()
    since = window_start(today or dt.datetime.now(dt.timezone.utc).date())
    periods, names = load_holdings(iter_archive_rows(d, counts, since), counts)
    if counts["rows"] == 0:
        reason = f"no filings13f rows found ({d})"
        _write_atomic(os.path.join(out_dir, "last_run.json"),
                      {"status": "skipped", "reason": reason,
                       "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        return {"status": "skipped", "reason": reason}
    if _rss_mb() > max_rss_mb:
        raise MemoryError(f"peak RSS {_rss_mb():.0f} MB > budget {max_rss_mb:.0f} MB after loading "
                          f"{counts['full_filings']} filings")
    pairs = detect(periods, names, since)
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for pair in pairs:
        head = {"schema": SCHEMA, "generated_at": now_iso, "code_version": code_version}
        _write_atomic(os.path.join(out_dir, f"events_{pair['period']}.json"),
                      {**head, **{k: v for k, v in pair.items() if k != "controls"}})
        _write_atomic(os.path.join(out_dir, f"controls_{pair['period']}.json"),
                      {**head, "period": pair["period"], "base_period": pair["base_period"],
                       "controls": pair["controls"]})
    appended = append_first_seen(os.path.join(out_dir, "first_seen.jsonl"), pairs, now_iso,
                                 counts["last_day"], code_version)
    newest = pairs[-1] if pairs else None
    summary_pairs = [{k: v for k, v in p.items() if k not in ("events", "controls")} for p in pairs]
    latest = {
        "schema": SCHEMA, "generated_at": now_iso, "code_version": code_version,
        "archive": counts, "window": {"lookback_days": LOOKBACK_DAYS, "since": since},
        "periods_full_filings": {p: len(m) for p, m in sorted(periods.items())},
        "pairs": summary_pairs,
        "newest_pair_top_thin": [e for e in (newest["events"] if newest else []) if e["class"] == "thin"][:TOP_N],
        "first_seen_appended": appended,
        "definitions": definitions(),
        "elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": round(_rss_mb(), 1),
    }
    _write_atomic(os.path.join(out_dir, "latest.json"), latest)
    thin = sum(p["clusters_by_class"]["thin"] for p in pairs)
    _write_atomic(os.path.join(out_dir, "last_run.json"),
                  {"status": "ok", "finished_at": now_iso, "pairs": len(pairs), "thin_clusters": thin})
    return {"status": "ok", "pairs": len(pairs), "full_filings": counts["full_filings"],
            "clusters": sum(p["clusters"] for p in pairs), "thin_clusters": thin, "first_seen_appended": appended,
            "elapsed_s": latest["elapsed_s"], "peak_rss_mb": latest["peak_rss_mb"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--max-rss-mb", type=float, default=700.0)
    ap.add_argument("--code-version", default="unknown")
    ap.add_argument("--archive-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    if a.nice:
        try:
            os.nice(a.nice)
        except OSError as e:
            print(f"[filings13f_clusters] os.nice failed: {e}", file=sys.stderr)
    data_dir = os.environ.get("DATA_DIR") or ("/data/voltrade" if os.path.exists("/data") else "/tmp")
    out_dir = a.out_dir or os.path.join(data_dir, OUT_SUBDIR)
    try:
        out = run(out_dir, a.archive_dir, a.code_version, a.max_rss_mb)
    except Exception as e:  # noqa: BLE001 — one JSON line for the scheduler; the last good latest.json is untouched
        out = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        try:
            _write_atomic(os.path.join(out_dir, "last_run.json"),
                          {"status": "error", "reason": out["reason"],
                           "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        except OSError as e2:
            print(f"[filings13f_clusters] could not write last_run.json: {e2}", file=sys.stderr)
    print(json.dumps(out))
    return 0 if out["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
