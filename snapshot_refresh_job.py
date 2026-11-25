#!/usr/bin/env python3
"""
snapshot_refresh_job.py — the nightly refresh of the four whole-file snapshots,
run by the server (server/snapshotRefresh.ts) on the Railway volume.

WHY (research/open_questions.md (ah4), 2026-11-25): the four snapshots
(EIA weekly, CPC degree days, STB EP724 rail, JODI stocks) were refreshed only
when a session ran scripts/snapshot_refresh.py and committed the result. The
21-day staleness trigger fired three times in seven weeks (10-12 at 99 days,
11-02, 11-24) and each time a session spent its time on a download. This
module runs the same four refreshers (now at the repo root, because the
production image ships root *.py and never scripts/) and writes the result
beside the archives, where the server reads it.

WHAT IT DOES, per source, in order:
  1. reads the copy the server would serve today: the newer, by embedded build
     time, of the volume copy and the copy bundled in the image;
  2. if that copy was built less than the source's minimum age ago, does
     nothing ("fresh": a restart in the same night fetches nothing again);
  3. otherwise runs the refresher's own main() with its output pointed at a
     staging file on the volume;
  4. accepts the staged file only if its build time is newer, it holds at least
     90% as many series as the served copy, and its newest data point is not
     earlier than the served copy's. A refused file is deleted and the served
     copy stays ("rejected");
  5. renames the staged file over the volume copy (same directory: atomic).

A source that cannot be fetched or parsed is "error" with its reason; the
other sources still run. Nothing here touches the bundled file: it stays the
cold-start copy and the rollback (VOLTRADE_SNAPSHOT_REFRESH=0 makes the server
ignore the volume copies).

OUTPUT: <DATA_DIR>/snapshot_refresh/<repo-relative path of each snapshot> and
<DATA_DIR>/snapshot_refresh/last_run.json. The last stdout line is one JSON
object (the scheduler reads it). Raw public-domain / attributed source data,
unchanged in content from what a session-run refresh writes; no derived value.

VINTAGES (open_questions.md (ao3), 2026-11-25): the volume keeps the newest copy
only, and EIA, CPC, the STB and JODI all revise what they published. What a
source said on a given night cannot be fetched again later, so before an
accepted file replaces the served one, every point that differs between the two
is appended to <DATA_DIR>/snapshot_refresh/vintages/:
  <key>.jsonl   one row per changed point: {"seen", "prev", "series", "period",
                "kind", "old", "new"}. kind is "new" (the period was not in the
                served copy: a first print), "revised" (both hold it, the values
                differ) or "removed". `seen` is the new file's build time, `prev`
                the served copy's. JODI values are [value, assessment code].
  builds.jsonl  one row per accepted file, written AFTER its point rows: the
                counts, the series that appeared or left, both build times. A
                night with no row here is a night no file was accepted; a row
                with zero counts is a night nothing changed. Point rows count
                only when their (key, seen) has a row here: a run killed between
                the two writes repeats its point rows the next night under a
                new `seen` (read_vintage skips the orphans).
  full/<key>_<prev build>.json.gz   written INSTEAD of point rows when more than
                VINTAGE_MAX_ROWS points changed (a source re-based its history):
                the whole previous copy, so nothing is lost and the row log
                stays small.
The record starts on the night this shipped; the builds before it are in git
(one per session commit) and are not back-filled here. A failed write never
stops the refresh: the source still reads "refreshed", its `vintage` carries
the error and the run's status is "error", so the job row says so.
VOLTRADE_SNAPSHOT_VINTAGES=0 stops the record only. Raw source values, nothing
derived; nothing reads this record yet.

Run:
  python3 snapshot_refresh_job.py --run [--only eia,cpc,rail,jodi] [--out-root DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gzip
import importlib
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_SUBDIR = "snapshot_refresh"
SCHEMA = "snapshot_refresh/1"
# A refreshed file must hold at least this share of the served copy's series.
MIN_SERIES_SHARE = 0.9

VINTAGE_SUBDIR = "vintages"
VINTAGE_SCHEMA = "snapshot_vintage/1"
# More changed points than this in one build is a re-based history, not a
# revision: the previous copy is kept whole instead (the largest snapshot is
# 1.4 MB, about 0.2 MB gzipped). Measured 2026-11-25: one night's refresh changed
# at most a few hundred points per source.
VINTAGE_MAX_ROWS = 20000

# key, refresher module (repo root), the snapshot's repo-relative path, and the
# youngest served copy that is refreshed again. 20 h: one run a night, also when
# the previous night's run started late. JODI publishes monthly.
SOURCES = (
    {"key": "eia", "module": "eia_weekly", "rel": "datacore/eia/weekly_series.json", "min_age_h": 20.0},
    {"key": "cpc", "module": "cpc_degree_days", "rel": "datacore/cpc/degree_days.json", "min_age_h": 20.0},
    {"key": "rail", "module": "stb_rail", "rel": "datacore/rail/ep724_carloads.json", "min_age_h": 20.0},
    {"key": "jodi", "module": "jodi_oil", "rel": "datacore/jodi/primary_stocks.json", "min_age_h": 144.0},
)
SOURCE_KEYS = tuple(s["key"] for s in SOURCES)


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def parse_iso(s):
    """An ISO time as an aware UTC datetime, or None."""
    if not isinstance(s, str) or not s:
        return None
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def built_of(doc):
    """The snapshot's embedded build time (`built`, or JODI's `built_at`)."""
    if not isinstance(doc, dict):
        return None
    return parse_iso(doc.get("built") or doc.get("built_at"))


def summarize(key: str, doc: dict) -> dict:
    """{"n_series", "latest"} of one snapshot. `latest` is the newest data
    point's own date string (a week, a day or a month, per source): compared
    only against the same source's other copy. Raises on a shape it does not
    know, which the caller reports as a refusal."""
    series = doc["series"]
    if not isinstance(series, dict) or not series:
        raise ValueError("no series")
    if key == "eia":
        latest = max(str(s["last"]) for s in series.values())
    elif key == "cpc":
        latest = max(str(doc["axes"][k][-1]) for k in ("H", "C") if doc["axes"].get(k))
    elif key == "rail":
        latest = str(doc["weeks"][-1])
    elif key == "jodi":
        latest = str(doc["latest_period"])
    else:
        raise ValueError(f"unknown source {key}")
    return {"n_series": len(series), "latest": latest}


def validate(key: str, new_doc, served_doc):
    """(ok, reason). `served_doc` None means there is no copy to compare with
    (never the case in production: the image bundles one)."""
    new_built = built_of(new_doc)
    if new_built is None:
        return False, "no parseable build time"
    try:
        new_sum = summarize(key, new_doc)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return False, f"unexpected shape: {type(e).__name__}: {str(e)[:80]}"
    if served_doc is None:
        return True, "no served copy to compare"
    served_built = built_of(served_doc)
    if served_built is not None and new_built <= served_built:
        return False, "not newer than the served copy"
    try:
        served_sum = summarize(key, served_doc)
    except (KeyError, IndexError, TypeError, ValueError):
        return True, "served copy has an unknown shape"
    if new_sum["n_series"] < MIN_SERIES_SHARE * served_sum["n_series"]:
        return False, f"{new_sum['n_series']} series against {served_sum['n_series']} served"
    if new_sum["latest"] < served_sum["latest"]:
        return False, f"latest {new_sum['latest']} is earlier than the served {served_sum['latest']}"
    return True, "ok"


def vintages_enabled(env=None) -> bool:
    """Unset is on; set, only 1/true/on/yes is on (the rule of the refresh's own
    flag, server/lazyAssets.ts snapshotRefreshEnabled)."""
    v = str((env if env is not None else os.environ).get("VOLTRADE_SNAPSHOT_VINTAGES", "1")).strip().lower()
    return v in ("1", "true", "on", "yes")


def flatten(key: str, doc: dict) -> dict:
    """{series: {period: value}} of one snapshot, every point it holds. A null
    in a week- or day-aligned array is a period the source did not report and
    is left out. Raises on a shape it does not know."""
    series = doc["series"]
    out = {}
    if key == "eia":
        for name, s in series.items():
            out[name] = {str(p[0]): p[1] for p in s["points"] if p[1] is not None}
    elif key == "jodi":
        for name, s in series.items():
            out[name] = {str(p[0]): list(p[1:]) for p in s["points"]}
    elif key in ("cpc", "rail"):
        for name, vals in series.items():
            axis = doc["axes"][name.rsplit("|", 1)[1]] if key == "cpc" else doc["weeks"]
            if len(vals) != len(axis):
                raise ValueError(f"{name}: {len(vals)} values against {len(axis)} periods")
            out[name] = {str(p): v for p, v in zip(axis, vals) if v is not None}
    else:
        raise ValueError(f"unknown source {key}")
    return out


def diff_points(old: dict, new: dict) -> list:
    """Every point that differs between two flatten() results, as
    (series, period, kind, old, new), in a stable order."""
    rows = []
    for name in sorted(set(old) | set(new)):
        a, b = old.get(name, {}), new.get(name, {})
        for period in sorted(set(a) | set(b)):
            if period not in a:
                rows.append((name, period, "new", None, b[period]))
            elif period not in b:
                rows.append((name, period, "removed", a[period], None))
            elif a[period] != b[period]:
                rows.append((name, period, "revised", a[period], b[period]))
    return rows


def _append_lines(fp: str, lines: list) -> None:
    """One write call, flushed to the disk before returning."""
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in lines))
        fh.flush()
        os.fsync(fh.fileno())


def record_vintage(key: str, served_doc, new_doc, out_root: str) -> dict:
    """Append what changed between the served copy and the accepted one (see
    VINTAGES in the header). Never raises: {"error": reason} instead."""
    try:
        prev, seen = built_of(served_doc), built_of(new_doc)
        if prev is None or seen is None:
            return {"error": "a copy has no build time"}
        prev_s, seen_s = prev.isoformat(timespec="seconds"), seen.isoformat(timespec="seconds")
        old, new = flatten(key, served_doc), flatten(key, new_doc)
        rows = diff_points(old, new)
        counts = {k: sum(1 for r in rows if r[2] == k) for k in ("new", "revised", "removed")}
        vdir = os.path.join(out_root, VINTAGE_SUBDIR)
        build = {"schema": VINTAGE_SCHEMA, "key": key, "seen": seen_s, "prev": prev_s, **counts,
                 "series_added": sorted(set(new) - set(old)), "series_removed": sorted(set(old) - set(new))}
        if len(rows) > VINTAGE_MAX_ROWS:
            name = f"{key}_{prev.strftime('%Y%m%dT%H%M%SZ')}.json.gz"
            fp = os.path.join(vdir, "full", name)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with gzip.open(fp + ".tmp", "wt", encoding="utf-8") as fh:
                json.dump(served_doc, fh, separators=(",", ":"))
            os.replace(fp + ".tmp", fp)
            build["full_prev"] = name
        elif rows:
            _append_lines(os.path.join(vdir, f"{key}.jsonl"), [
                {"seen": seen_s, "prev": prev_s, "series": r[0], "period": r[1], "kind": r[2], "old": r[3], "new": r[4]}
                for r in rows])
        _append_lines(os.path.join(vdir, "builds.jsonl"), [build])
        out = dict(counts)
        if "full_prev" in build:
            out["full_prev"] = build["full_prev"]
        return out
    except Exception as e:  # noqa: BLE001 — the record must never stop a refresh; the reason is reported on the job row
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


def vintage_bytes(out_root: str):
    """Bytes the record holds on disk, or None when it cannot be read. The job
    row carries it, so a reader without the volume can see the record grow."""
    total = 0
    try:
        for root, _dirs, files in os.walk(os.path.join(out_root, VINTAGE_SUBDIR)):
            for name in files:
                total += os.path.getsize(os.path.join(root, name))
    except OSError as e:
        print(f"[snapshot_refresh] could not size the vintage record: {e}", file=sys.stderr)
        return None
    return total


def read_vintage(key: str, out_root: str) -> dict:
    """{"builds": [...], "rows": [...]} for one source: the build rows, and the
    point rows of those builds only (see builds.jsonl in the header). Lines
    that are not JSON are counted in "bad_lines", never raised."""
    vdir = os.path.join(out_root, VINTAGE_SUBDIR)
    bad = 0

    def lines(fp):
        nonlocal bad
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        bad += 1
                        continue
                    if isinstance(row, dict):
                        yield row
        except FileNotFoundError:
            return

    builds = [b for b in lines(os.path.join(vdir, "builds.jsonl")) if b.get("key") == key]
    committed = {b.get("seen") for b in builds}
    rows = [r for r in lines(os.path.join(vdir, f"{key}.jsonl")) if r.get("seen") in committed]
    return {"builds": builds, "rows": rows, "bad_lines": bad}


def _read_json(fp: str):
    try:
        with open(fp, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        print(f"[snapshot_refresh] unreadable {fp}: {e}", file=sys.stderr)
        return None


def _write_atomic(fp: str, payload) -> None:
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, fp)


def bundled_path(rel: str, roots=None):
    """Where the image (or the working tree) holds the bundled copy: the repo
    root in a checkout, dist/ in the image (script/build.ts stages it there)."""
    for root in (roots or (HERE, os.path.join(HERE, "dist"))):
        fp = os.path.join(root, rel)
        if os.path.exists(fp):
            return fp
    return None


def served_copy(rel: str, out_root: str, bundled_roots=None):
    """The newer of the volume copy and the bundled copy, with its origin."""
    best, origin = None, None
    for name, fp in (("bundled", bundled_path(rel, bundled_roots)), ("refreshed", os.path.join(out_root, rel))):
        doc = _read_json(fp) if fp else None
        b = built_of(doc)
        if b is None:
            continue
        if best is None or b > built_of(best):
            best, origin = doc, name
    return best, origin


def _call_refresher(module_name: str, staging: str, load=importlib.import_module) -> None:
    """Run one refresher's main() writing to `staging`. Its progress lines go
    to stderr so stdout keeps one JSON line. Raises whatever it raises; a
    SystemExit with a message or a non-zero code is an error."""
    mod = load(module_name)
    mod.OUT = staging
    argv = sys.argv
    sys.argv = [module_name]  # jodi_oil.main() reads argv[1] as a local zip
    try:
        with contextlib.redirect_stdout(sys.stderr):
            rc = mod.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            raise RuntimeError(str(e.code)[:160]) from None
        rc = 0
    finally:
        sys.argv = argv
    if rc not in (None, 0):
        raise RuntimeError(f"refresher returned {rc}")


def refresh_one(src: dict, out_root: str, now: dt.datetime, call=_call_refresher, bundled_roots=None) -> dict:
    """One source, start to finish. Never raises."""
    key, rel = src["key"], src["rel"]
    final = os.path.join(out_root, rel)
    staging = final + ".staging"
    served, origin = served_copy(rel, out_root, bundled_roots)
    served_built = built_of(served)
    if served_built is not None:
        age_h = (now - served_built).total_seconds() / 3600.0
        if age_h < src["min_age_h"]:
            return {"status": "fresh", "origin": origin, "built": served_built.isoformat(timespec="seconds"),
                    "reason": f"served copy is {age_h:.1f} h old (under {src['min_age_h']:.0f} h)"}
    try:
        os.makedirs(os.path.dirname(final), exist_ok=True)
        call(src["module"], staging)
        new_doc = _read_json(staging)
        ok, reason = validate(key, new_doc, served)
        if not ok:
            return {"status": "rejected", "reason": reason}
        # Before the served copy is replaced: after it, what changed is gone.
        vintage = record_vintage(key, served, new_doc, out_root) if served is not None and vintages_enabled() else None
        os.replace(staging, final)
        s = summarize(key, new_doc)
        out = {"status": "refreshed", "built": built_of(new_doc).isoformat(timespec="seconds"),
               "latest": s["latest"], "n_series": s["n_series"], "bytes": os.path.getsize(final)}
        if vintage is not None:
            out["vintage"] = vintage
        return out
    except Exception as e:  # noqa: BLE001 — one source's outage must not stop the others; the reason is reported
        return {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:140]}"}
    finally:
        for leftover in (staging, staging + ".tmp.xlsx"):
            try:
                os.unlink(leftover)
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"[snapshot_refresh] could not remove {leftover}: {e}", file=sys.stderr)


def overall_status(sources: dict) -> str:
    """error when any source failed or was refused, or a refreshed source's
    vintage record could not be written; ok otherwise."""
    return "error" if any(r["status"] in ("error", "rejected") or "error" in (r.get("vintage") or {})
                          for r in sources.values()) else "ok"


def run(out_root: str, only=None, code_version: str = "unknown", max_rss_mb: float = 600.0,
        now=None, call=_call_refresher, bundled_roots=None) -> dict:
    t0 = time.time()
    now = now or dt.datetime.now(dt.timezone.utc)
    sources = {}
    for src in SOURCES:
        if only and src["key"] not in only:
            continue
        if _rss_mb() > max_rss_mb:
            sources[src["key"]] = {"status": "error", "reason": f"not started: peak memory {_rss_mb():.0f} MB is over the {max_rss_mb:.0f} MB budget"}
            continue
        sources[src["key"]] = refresh_one(src, out_root, now, call=call, bundled_roots=bundled_roots)
    out = {
        "schema": SCHEMA, "status": overall_status(sources), "code_version": code_version,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "refreshed": sum(1 for r in sources.values() if r["status"] == "refreshed"),
        "sources": sources, "elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": round(_rss_mb(), 1),
        "vintage_bytes": vintage_bytes(out_root),
    }
    _write_atomic(os.path.join(out_root, "last_run.json"), out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--max-rss-mb", type=float, default=600.0)
    ap.add_argument("--code-version", default="unknown")
    ap.add_argument("--only", default="")
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args(argv)
    only = [x for x in a.only.split(",") if x] or None
    bad = [x for x in (only or []) if x not in SOURCE_KEYS]
    if bad:
        print(json.dumps({"status": "error", "reason": f"unknown --only {bad}"}))
        return 1
    if a.nice:
        try:
            os.nice(a.nice)
        except OSError as e:
            print(f"[snapshot_refresh] os.nice failed: {e}", file=sys.stderr)
    data_dir = os.environ.get("DATA_DIR") or ("/data/voltrade" if os.path.exists("/data") else "/tmp")
    out_root = a.out_root or os.path.join(data_dir, OUT_SUBDIR)
    try:
        out = run(out_root, only, a.code_version, a.max_rss_mb)
    except Exception as e:  # noqa: BLE001 — one JSON line for the scheduler; served copies are untouched
        out = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
    print(json.dumps(out))
    return 0 if out["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
