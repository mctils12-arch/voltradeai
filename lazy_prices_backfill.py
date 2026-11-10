"""Lazy Prices gate-2 INPUT backfill — point-in-time 10-K similarity rows
accumulated on the production volume, with the heavy bytes kept OFF it.

WHY THIS EXISTS (2026-11-10): the weekly gate-2 job (server/lazyPrices.ts ->
lazy_prices_run.py) read the `filings_10kq` archive, and that archive does not
exist on the Railway volume — the PR-1 `sample` CLI only ever ran in sandboxes.
It also could not be built there: a 10-K primary document is 5-30 MB of inline
XBRL (1-4 MB gzipped), so the ~50k annual reports of the insider-visible
universe since 2015 are 50-200 GB raw and 2-4 GB as extracted sections, against
a volume with ~1 GB free whose archives outrank every derived store (CLAUDE.md
priority 1). And `sec_filings_similarity.pit_rows_from_archive` holds every
section text in memory at once, far past the 1.5 GB long-job budget at scale.

WHAT GATE 2 ACTUALLY NEEDS is one ~600-byte ROW per paired filing (per-section
cosine / Jaccard / point-in-time tf-idf cosine + identifiers) and, for the
tf-idf, the per-section DOCUMENT-FREQUENCY table of every filing accepted
earlier. Both are small. So this module streams the universe in FILING-DATE
ORDER and persists only:

  <DATA_DIR>/lazy_prices/backfill/
    plan.jsonl.gz       every 10-K (no /A) of every CIK in the dated CIK->ticker
                        map, listed from the submissions JSON; rebuilt when
                        older than PLAN_STALE_DAYS so new filings join the tail
    plan_meta.json      plan progress (CIKs listed so far): the listing is
                        itself resumable
    rows/YYYY-MM.jsonl.gz  scored rows, SAME SCHEMA as pit_rows_from_archive
                        (+ filing_date); month files rewritten atomically at
                        every checkpoint
    pit_state.json.gz   per-section df Counter + n = the corpus of every
                        processed filing on CLOSED days
    checkpoint.json     cursor day, whether it is closed, the open day's finished
                        accessions and df delta (a kill mid-day resumes exactly),
                        per-day processed counts, counters, bounded failures
    status.json         the last run's summary (the diag probe reads it)

Raw documents, index-headers pages and extracted sections live in CONTAINER
TMP (the sec_filings_text layout under a tmp archive root): raw bytes are
removed the moment the sections are extracted; sections stay ~410 days of
filing date behind the cursor as the prior-year cache and are simply re-fetched
if a deploy wipes tmp (EDGAR is the archive of record for public filings;
re-fetching costs requests, never correctness). datacore/manifests/
filings_10kq.json still describes the sample CLI's write-once volume layout —
that CLI is unchanged; this job never writes raw bytes to the volume.

POINT-IN-TIME RULE (day-granular, strictly conservative): a filing is scored
against the df table of filings whose FILING DATE is strictly earlier than its
own. pit_rows_from_archive uses exact acceptance timestamps, so it also admits
same-day filings accepted earlier; this job admits none of them. With filings
on distinct days the two produce identical rows (tested). A whole day shares
one frozen idf, so a day's fetch+extract runs in parallel threads and the df
table is updated only when the day CLOSES. A day closes only once every filing
on it is processed AND the day is at least two days older than the plan
listing (EDGAR keeps accepting until 22:00 ET; a newer plan may add filings).
A filing that enters the plan BEHIND a closed day (a CIK newly mapped, a late
EDGAR entry) is never scored — scoring it against a table that already holds
later filings would be lookahead — it is counted (`behind_cursor_skipped`).

Honest limits: no return is computed here and no signal is claimed; rows carry
CIK + accession (SEC public records, no Alpaca data) and never reach /data or
/api/v1. Nothing here touches a trade path. No trading imports (SPINOUT-READY).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import resource
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional

import sec_filings_similarity as sim
import sec_filings_text as sft

BACKFILL_SUBDIR = os.path.join("lazy_prices", "backfill")
SINCE_DEFAULT = "2015-01-01"        # first filing day scored; the panel is 2016+, 2015 warms the corpus and supplies priors
PRIOR_LOOKBACK_DAYS = 365 + sft.PAIR_TOLERANCE_DAYS   # the plan lists this far before SINCE (prior-only rows)
PLAN_STALE_DAYS = 7
PLAN_CHECKPOINT_EVERY = 250         # CIKs listed between plan writes
CLOSE_LAG_DAYS = 2                  # a day closes only when the plan listing is this many days newer
CHUNK = 48                          # filings fetched+extracted in parallel before scoring; one checkpoint per chunk
WORKERS = 4
SECTIONS_KEEP_DAYS = PRIOR_LOOKBACK_DAYS + 30   # tmp sections older than this behind the cursor are pruned
MIN_FREE_BYTES = 256 * 1024 * 1024  # volume floor: the job writes ~50 MB of derived rows/state in total
MAX_FAILED_KEPT = 5000
MIN_PIT_CORPUS = sim.MIN_PIT_CORPUS
FORMS = ("10-K",)                   # the pre-registered primary is 10-K Item 1A; 10-Q would triple the work for a non-primary


# ── small file helpers ──────────────────────────────────────────────────────

def _write_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_json(path: str, obj) -> None:
    _write_atomic(path, json.dumps(obj, default=str).encode("utf-8"))


def _write_json_gz(path: str, obj) -> None:
    _write_atomic(path, gzip.compress(json.dumps(obj, default=str).encode("utf-8"), compresslevel=6))


def _write_jsonl_gz(path: str, rows: Iterable[dict]) -> None:
    body = "".join(json.dumps(r, default=str) + "\n" for r in rows).encode("utf-8")
    _write_atomic(path, gzip.compress(body, compresslevel=6))


def _read_json(path: str, default=None):
    try:
        with open(path, "rb") as f:
            return json.loads(f.read())
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as e:
        print(f"[lazy_prices_backfill] unreadable {path}: {e}", file=sys.stderr)
        return default


def _read_json_gz(path: str, default=None):
    try:
        with gzip.open(path, "rb") as f:
            return json.loads(f.read())
    except FileNotFoundError:
        return default
    except (OSError, ValueError, EOFError) as e:
        print(f"[lazy_prices_backfill] unreadable {path}: {e}", file=sys.stderr)
        return default


def _read_jsonl_gz(path: str) -> list:
    out: list = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except FileNotFoundError:
        return []
    except (OSError, ValueError, EOFError) as e:
        print(f"[lazy_prices_backfill] unreadable {path}: {e}", file=sys.stderr)
    return out


def data_dir() -> str:
    return os.environ.get("DATA_DIR") or ("/data/voltrade" if os.path.exists("/data") else "/tmp")


def default_root() -> str:
    return os.path.join(data_dir(), BACKFILL_SUBDIR)


def default_tmp_archive() -> str:
    return os.path.join(tempfile.gettempdir(), "voltrade_lazy_prices_tmp_archive")


def free_bytes(path: str) -> Optional[int]:
    try:
        p = path
        while p and not os.path.isdir(p):
            p = os.path.dirname(p)
        st = os.statvfs(p or ".")
        return st.f_bavail * st.f_frsize
    except OSError:
        return None


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024.0 if sys.platform != "darwin" else ru / (1024.0 * 1024.0)


def _shift(day: str, days: int) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=days)).isoformat()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ── the plan: every 10-K of every mapped CIK, in filing-date order ────────────

def plan_row(f: dict) -> dict:
    """The fields process_filing / pairing need, nothing else."""
    return {"cik": sft.cik10(f["cik"]), "accession": f["accession"], "form": f["form"],
            "filingDate": f.get("filingDate"), "reportDate": f.get("reportDate"),
            "acceptanceDateTime": f.get("acceptanceDateTime"), "primaryDocument": f.get("primaryDocument"),
            "isAmendment": bool(f.get("isAmendment"))}


def _plan_paths(root: str):
    return os.path.join(root, "plan.jsonl.gz"), os.path.join(root, "plan_meta.json")


def build_plan(root: str, ciks: Iterable[str], since: str, list_filings: Callable[[str, str], list],
               deadline: Optional[float] = None, checkpoint_every: int = PLAN_CHECKPOINT_EVERY,
               reset: bool = False) -> dict:
    """Resumable listing. `list_filings(cik, list_since)` -> submissions rows
    (sft.list_filings in production, injected in tests). Amendments are dropped
    here (never paired, never corpus). Returns plan_meta; `complete` is False
    when the deadline stopped the listing (the next run continues it)."""
    plan_path, meta_path = _plan_paths(root)
    meta = {} if reset else (_read_json(meta_path, {}) or {})
    list_since = _shift(since, -PRIOR_LOOKBACK_DAYS)
    if meta.get("since") != since or meta.get("list_since") != list_since or meta.get("complete"):
        meta = {"since": since, "list_since": list_since, "done_ciks": [], "complete": False,
                "started_at": _now_iso(), "filings": 0, "list_errors": 0}
        rows: list = []
    else:
        rows = _read_jsonl_gz(plan_path)
    done = set(meta.get("done_ciks") or [])
    todo = [c for c in sorted({str(c) for c in ciks}, key=lambda c: (len(c), c)) if c not in done]
    n_since_ckpt = 0

    def save(complete: bool) -> None:
        rows.sort(key=lambda r: (r["filingDate"] or "", r["accession"]))
        meta.update({"done_ciks": sorted(done), "complete": complete, "filings": len(rows), "updated_at": _now_iso()})
        if complete:
            meta["completed_at"] = meta["updated_at"]
        _write_jsonl_gz(plan_path, rows)
        _write_json(meta_path, meta)

    for cik in todo:
        if deadline is not None and time.time() >= deadline:
            save(False)
            return meta
        try:
            fl = list_filings(cik, list_since)
        except Exception as e:  # noqa: BLE001 — one CIK's listing failure never aborts the plan
            meta["list_errors"] = int(meta.get("list_errors") or 0) + 1
            print(f"[lazy_prices_backfill] list {cik}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            done.add(cik)   # retried on the next plan REBUILD, not every run
            continue
        rows.extend(plan_row(f) for f in fl
                    if not f.get("isAmendment") and sft.base_form(f["form"]) in FORMS and f.get("filingDate"))
        done.add(cik)
        n_since_ckpt += 1
        if n_since_ckpt >= checkpoint_every:
            save(False)
            n_since_ckpt = 0
    save(True)
    return meta


def plan_is_stale(meta: Optional[dict], now: Optional[dt.datetime] = None, stale_days: int = PLAN_STALE_DAYS) -> bool:
    if not meta or not meta.get("complete") or not meta.get("completed_at"):
        return True
    try:
        age = (now or dt.datetime.now(dt.timezone.utc)) - dt.datetime.fromisoformat(str(meta["completed_at"]).replace("Z", "+00:00"))
    except ValueError:
        return True
    return age.days >= stale_days


def load_plan(root: str) -> list:
    return _read_jsonl_gz(_plan_paths(root)[0])


def plan_listing_day(meta: Optional[dict]) -> Optional[str]:
    ts = (meta or {}).get("completed_at")
    return str(ts)[:10] if ts else None


# ── checkpoint / state ───────────────────────────────────────────────────────

def _empty_checkpoint(since: str) -> dict:
    return {"schema": "lazy_prices_backfill/1", "since": since, "cursor_day": None, "closed": True,
            "day_done": [], "day_df": {}, "day_n": {}, "day_counts": {}, "counts": Counter(), "failed": {}}


def load_checkpoint(root: str, since: str) -> dict:
    ck = _read_json(os.path.join(root, "checkpoint.json"))
    if not ck or ck.get("since") != since:
        return _empty_checkpoint(since)
    base = _empty_checkpoint(since)
    base.update(ck)
    base["counts"] = Counter(ck.get("counts") or {})
    return base


def load_pit_state(root: str) -> dict:
    st = _read_json_gz(os.path.join(root, "pit_state.json.gz"), {}) or {}
    return {k: sim.PitIdf.from_state(v) for k, v in st.items()}


def save_state(root: str, ck: dict, idf: dict, rows_by_month: dict, dirty_months: Iterable[str]) -> None:
    for m in sorted(set(dirty_months)):
        _write_jsonl_gz(os.path.join(root, "rows", f"{m}.jsonl.gz"), rows_by_month.get(m) or [])
    _write_json_gz(os.path.join(root, "pit_state.json.gz"), {k: v.state() for k, v in idf.items()})
    out = dict(ck)
    out["counts"] = dict(ck["counts"])
    _write_json(os.path.join(root, "checkpoint.json"), out)


def load_rows(root: str) -> list:
    """Every scored row (all month files), oldest first — what lazy_prices_run reads."""
    d = os.path.join(root, "rows")
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith(".jsonl.gz"))
    except FileNotFoundError:
        return []
    rows: list = []
    for n in names:
        rows.extend(_read_jsonl_gz(os.path.join(d, n)))
    return rows


def rows_status(root: str) -> dict:
    """Headline counts for the diag probe / weekly runner (no identifiers)."""
    d = os.path.join(root, "rows")
    try:
        months = sorted(n[:-len(".jsonl.gz")] for n in os.listdir(d) if n.endswith(".jsonl.gz"))
    except FileNotFoundError:
        months = []
    ck = _read_json(os.path.join(root, "checkpoint.json")) or {}
    meta = _read_json(_plan_paths(root)[1]) or {}
    return {"months": len(months), "first_month": months[0] if months else None, "last_month": months[-1] if months else None,
            "cursor_day": ck.get("cursor_day"), "cursor_closed": ck.get("closed"), "counts": ck.get("counts") or {},
            "plan": {"complete": meta.get("complete"), "filings": meta.get("filings"), "ciks": len(meta.get("done_ciks") or []),
                     "completed_at": meta.get("completed_at"), "list_errors": meta.get("list_errors")},
            "status": _read_json(os.path.join(root, "status.json"))}


# ── per-filing fetch + extract (tmp archive), raw bytes dropped ───────────────

# Best-effort tmp housekeeping never aborts a run, but it is never silent either:
# every oddity is counted here and reported in the run summary / status.json.
IO_NOISE: Counter = Counter()


def _best_effort_remove(path: str, what: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        IO_NOISE[f"{what}_already_absent"] += 1
    except OSError as e:
        IO_NOISE[f"{what}_remove_failed"] += 1
        print(f"[lazy_prices_backfill] could not remove {what} {path}: {e}", file=sys.stderr)


def make_fetcher(client, tmp_archive: str) -> Callable[[dict], dict]:
    """process_filing against the TMP archive; the multi-MB raw gz and the
    headers page are removed as soon as the sections exist. Sections stay as
    the prior-year cache (pruned by the caller)."""
    def fetch(filing: dict) -> dict:
        sp = sft.sections_path(tmp_archive, filing["cik"], filing["accession"])
        if os.path.exists(sp):
            try:
                return json.loads(sft.read_gz(sp))
            except (OSError, ValueError, EOFError) as e:
                IO_NOISE["sections_cache_unreadable"] += 1     # re-fetched below; the cache is disposable
                print(f"[lazy_prices_backfill] unreadable cached sections {sp}: {e}; refetching", file=sys.stderr)
        doc = sft.process_filing(client, filing, tmp_archive)
        doc.pop("_io", None)
        _best_effort_remove(sft.raw_path(tmp_archive, filing["cik"], filing["accession"], filing.get("primaryDocument")), "raw")
        _best_effort_remove(sft.headers_path(tmp_archive, filing["cik"], filing["accession"]), "headers")
        return doc
    return fetch


def prune_sections(tmp_archive: str, plan_dates: dict, cursor_day: Optional[str], keep_days: int = SECTIONS_KEEP_DAYS) -> int:
    """Remove tmp sections whose filing date is more than keep_days behind the
    cursor (they can no longer be anyone's prior). Returns files removed."""
    root = os.path.join(sft.stream_root(tmp_archive), "sections")
    if not cursor_day or not os.path.isdir(root):
        return 0
    floor = _shift(cursor_day, -keep_days)
    removed = 0
    for cik in os.listdir(root):
        d = os.path.join(root, cik)
        try:
            names = os.listdir(d)
        except OSError as e:
            IO_NOISE["prune_listdir_failed"] += 1
            print(f"[lazy_prices_backfill] prune: cannot list {d}: {e}", file=sys.stderr)
            continue
        for n in names:
            acc = n[:-len(".json.gz")] if n.endswith(".json.gz") else None
            fd = plan_dates.get(acc) if acc else None
            if fd is not None and fd < floor:
                before = IO_NOISE["sections_remove_failed"]
                _best_effort_remove(os.path.join(d, n), "sections")
                removed += IO_NOISE["sections_remove_failed"] == before
        try:
            if not os.listdir(d):
                os.rmdir(d)
        except OSError as e:
            IO_NOISE["prune_rmdir_failed"] += 1
            print(f"[lazy_prices_backfill] prune: cannot remove {d}: {e}", file=sys.stderr)
    return removed


# ── the streaming scorer ─────────────────────────────────────────────────────

def _usable(idf: dict, min_corpus: int) -> dict:
    return {k: v for k, v in idf.items() if v.n >= min_corpus} if min_corpus > 0 else dict(idf)


def score_pair(cur: dict, prior: dict, gap_days, idf: dict, min_corpus: int) -> dict:
    """Same row as pit_rows_from_archive writes, plus filing_date (the month key)."""
    row = sim.pair_similarity(cur, prior, _usable(idf, min_corpus))
    row["gap_days"] = gap_days
    row["idf_pit"] = True
    row["idf_min_corpus"] = min_corpus
    row["idf_n"] = {k: v.n for k, v in idf.items()}
    row["filing_date"] = cur.get("filing_date")
    return row


def doc_token_sets(doc: dict) -> dict:
    return {k: set(sim.tokenize(sec["text"])) for k, sec in (doc.get("sections") or {}).items()
            if not sim.exclusion_reason(sec)}


class Backfill:
    """Streams the plan in filing-date order; see the module docstring."""

    def __init__(self, root: str, tmp_archive: str, fetch: Callable[[dict], dict], since: str = SINCE_DEFAULT,
                 min_corpus: int = MIN_PIT_CORPUS, workers: int = WORKERS, chunk: int = CHUNK,
                 max_rss_mb: float = 1500.0, min_free_bytes: int = MIN_FREE_BYTES, close_before: Optional[str] = None):
        self.root, self.tmp_archive, self.fetch, self.since = root, tmp_archive, fetch, since
        self.min_corpus, self.workers, self.chunk = min_corpus, max(1, workers), max(1, chunk)
        self.max_rss_mb, self.min_free_bytes = max_rss_mb, min_free_bytes
        # days >= close_before stay OPEN at the end of a run (a newer plan may still add filings to them)
        self.close_before = close_before
        self.ck = load_checkpoint(root, since)
        self.idf = load_pit_state(root)
        self.rows_by_month: dict = {}
        self.dirty_months: set = set()
        self.stop_reason: Optional[str] = None
        self.by_acc: dict = {}
        self.plan_dates: dict = {}
        self.pairs: dict = {}

    # -- plan views
    def _prepare(self, plan: list) -> list:
        self.by_acc = {r["accession"]: r for r in plan}
        self.plan_dates = {r["accession"]: r["filingDate"] for r in plan}
        by_cik: dict = {}
        for r in plan:
            by_cik.setdefault(r["cik"], []).append(r)
        self.pairs = {}
        for rows in by_cik.values():
            self.pairs.update(sft.pair_prior_filings(rows))
        failed = self.ck["failed"]
        done_today = set(self.ck["day_done"])
        cursor, closed = self.ck["cursor_day"], self.ck["closed"]
        todo, per_day = [], Counter()
        for r in plan:
            fd = r["filingDate"]
            if fd < self.since:
                continue
            per_day[fd] += 1
            if r["accession"] in failed:
                continue
            if cursor and (fd < cursor or (fd == cursor and closed)):
                continue      # a closed day: processed, or arrived late (counted below)
            if cursor and fd == cursor and r["accession"] in done_today:
                continue
            todo.append(r)
        # late arrivals behind closed days: plan filings minus what those days processed (recomputed, not accumulated)
        behind = 0
        for day, n in per_day.items():
            if cursor and (day < cursor or (day == cursor and closed)):
                behind += max(0, n - int(self.ck["day_counts"].get(day) or 0))
        self.ck["counts"]["behind_cursor_skipped"] = behind
        return todo

    def _month_rows(self, month: str) -> list:
        if month not in self.rows_by_month:
            self.rows_by_month[month] = _read_jsonl_gz(os.path.join(self.root, "rows", f"{month}.jsonl.gz"))
        return self.rows_by_month[month]

    def _append_row(self, row: dict) -> None:
        month = (row.get("filing_date") or "0000-00")[:7]
        self._month_rows(month).append(row)
        self.dirty_months.add(month)

    def _checkpoint(self) -> None:
        save_state(self.root, self.ck, self.idf, self.rows_by_month, self.dirty_months)
        self.dirty_months.clear()
        keep = (self.ck["cursor_day"] or "")[:7]
        for m in list(self.rows_by_month):
            if m != keep:
                del self.rows_by_month[m]

    def _guards(self, deadline: Optional[float]) -> bool:
        if deadline is not None and time.time() >= deadline:
            self.stop_reason = "budget"
            return False
        rss = peak_rss_mb()
        if rss > self.max_rss_mb:
            self.stop_reason = f"rss {rss:.0f} MB > {self.max_rss_mb:.0f} MB"
            return False
        fb = free_bytes(self.root)
        if fb is not None and fb < self.min_free_bytes:
            self.stop_reason = f"volume floor: {fb // 1_000_000} MB free < {self.min_free_bytes // 1_000_000} MB"
            return False
        return True

    def _open_day(self, day: str) -> None:
        self.ck.update({"cursor_day": day, "closed": False, "day_done": [], "day_df": {}, "day_n": {}})

    def _close_day(self) -> None:
        """Fold the finished day's df delta into the corpus: its filings become
        knowable for every LATER day, never for their own."""
        for key, df in self.ck["day_df"].items():
            self.idf.setdefault(key, sim.PitIdf()).add_delta(df, int(self.ck["day_n"].get(key) or 0))
        self.ck["day_counts"][self.ck["cursor_day"]] = len(self.ck["day_done"])
        self.ck.update({"closed": True, "day_done": [], "day_df": {}, "day_n": {}})
        self.ck["counts"]["days_closed"] += 1

    def _may_close(self, day: str) -> bool:
        return self.close_before is None or day < self.close_before

    def _fetch_pair(self, filing: dict):
        """-> (cur_doc|None, prior_doc|None, prior_acc|None, gap_days, error|None). Thread-safe: reads only."""
        p = self.pairs.get(filing["accession"]) or {"prior": None, "gap_days": None}
        try:
            cur = self.fetch(filing)
        except Exception as e:  # noqa: BLE001 — a bad filing is recorded, never fatal
            return None, None, p["prior"], p["gap_days"], f"{type(e).__name__}: {str(e)[:120]}"
        prior = None
        if p["prior"]:
            try:
                prior = self.fetch(self.by_acc[p["prior"]])
            except Exception as e:  # noqa: BLE001
                return cur, None, p["prior"], p["gap_days"], f"prior {type(e).__name__}: {str(e)[:120]}"
        return cur, prior, p["prior"], p["gap_days"], None

    def _record_failure(self, acc: str, reason: str) -> None:
        if len(self.ck["failed"]) < MAX_FAILED_KEPT:
            self.ck["failed"][acc] = reason
        self.ck["counts"]["failed"] += 1

    def _absorb(self, filing: dict, cur: dict, prior, prior_acc, gap, err) -> None:
        """Score one fetched filing against the day's frozen idf and stage its
        token sets in the open day's df delta."""
        acc = filing["accession"]
        if prior_acc is None:
            self.ck["counts"]["unpaired"] += 1
        elif prior is None:
            self._record_failure(acc, err or "prior unavailable")
        else:
            self._append_row(score_pair(cur, prior, gap, self.idf, self.min_corpus))
            self.ck["counts"]["rows"] += 1
        for key, toks in doc_token_sets(cur).items():
            df = self.ck["day_df"].setdefault(key, {})
            for t in toks:
                df[t] = df.get(t, 0) + 1
            self.ck["day_n"][key] = int(self.ck["day_n"].get(key) or 0) + 1
        self.ck["counts"]["filings"] += 1

    def run(self, plan: list, deadline: Optional[float] = None, max_filings: Optional[int] = None) -> dict:
        t0 = time.time()
        todo = self._prepare(plan)
        processed, i = 0, 0
        while i < len(todo):
            if not self._guards(deadline):
                break
            if max_filings is not None and processed >= max_filings:
                self.stop_reason = "max_filings"
                break
            day = todo[i]["filingDate"]
            if self.ck["cursor_day"] != day:
                prev = self.ck["cursor_day"]
                if prev is not None and not self.ck["closed"]:
                    # an open earlier day left by a previous run is fully processed (nothing of it is in todo),
                    # but it may close only once the plan is new enough that no late filing can still join it;
                    # until then no later day can be scored either (its idf must include this day)
                    if not self._may_close(prev):
                        self.stop_reason = f"waiting for a plan listed after {prev} before closing it"
                        break
                    self._close_day()
                self._open_day(day)
            j = i
            while j < len(todo) and todo[j]["filingDate"] == day and j - i < self.chunk:
                j += 1
            if max_filings is not None:
                j = min(j, i + max(0, max_filings - processed))
            chunk = todo[i:j]
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                results = list(ex.map(self._fetch_pair, chunk))
            for filing, (cur, prior, prior_acc, gap, err) in zip(chunk, results):
                if cur is None:
                    self._record_failure(filing["accession"], err or "unknown")
                else:
                    self._absorb(filing, cur, prior, prior_acc, gap, err)
                self.ck["day_done"].append(filing["accession"])
            processed += len(chunk)
            i = j
            day_complete = i >= len(todo) or todo[i]["filingDate"] != day
            if day_complete and self._may_close(day):
                self._close_day()
            self._checkpoint()       # a kill before the next chunk resumes from day_done / day_df exactly
        else:
            self.stop_reason = self.stop_reason or "plan exhausted"
        self.ck["counts"]["pruned_sections"] += prune_sections(self.tmp_archive, self.plan_dates, self.ck["cursor_day"])
        self.ck["counts"]["last_processed"] = processed
        self._checkpoint()
        return {"processed": processed, "cursor_day": self.ck["cursor_day"], "cursor_closed": self.ck["closed"],
                "remaining_in_plan": len(todo) - i, "stop_reason": self.stop_reason, "counts": dict(self.ck["counts"]),
                "io_noise": dict(IO_NOISE), "elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": round(peak_rss_mb())}


# ── the one-shot entry point for the nightly job ─────────────────────────────

def run_once(root: str, pairs_dir: str, tmp_archive: str, since: str, budget_s: float, max_rss_mb: float,
             code_version: str, max_filings: Optional[int] = None, ciks: Optional[Iterable[str]] = None,
             client=None, list_filings: Optional[Callable[[str, str], list]] = None,
             now: Optional[dt.datetime] = None) -> dict:
    """One bounded run: (re)build the plan if stale, then score until the budget.
    Returns the one-line summary; never raises for a prerequisite problem."""
    t0 = time.time()
    deadline = t0 + budget_s
    fb = free_bytes(root)
    if fb is not None and fb < MIN_FREE_BYTES:
        return {"status": "skipped", "reason": f"only {fb // 1_000_000} MB free on the volume (< {MIN_FREE_BYTES // 1_000_000} MB)"}
    os.makedirs(root, exist_ok=True)
    if ciks is None:
        import sec_cik_symbol_map as cmap
        import sec_issuer_pairs as sip
        _ok, failed_q = sip.ensure_pairs_cache(pairs_dir)
        ciks = sorted(cmap.build_map(sip.rows_from_pairs_cache(pairs_dir)).keys(), key=lambda c: (len(c), c))
        if not ciks:
            return {"status": "skipped", "reason": f"CIK map empty ({len(failed_q)} pairs quarters failed)"}
    ciks = list(ciks)
    client = client or sft.EdgarClient()
    lf = list_filings or (lambda cik, s: sft.list_filings(cik, forms=FORMS, since=s, client=client))
    meta = _read_json(_plan_paths(root)[1], {}) or {}
    plan_built = False
    if meta.get("since") != since or plan_is_stale(meta, now):
        meta = build_plan(root, ciks, since, lf, deadline=deadline, reset=bool(meta.get("complete")))
        plan_built = True
        if not meta.get("complete"):
            out = {"status": "ok", "phase": "plan", "plan_complete": False, "plan_ciks_listed": len(meta.get("done_ciks") or []),
                   "plan_ciks_total": len(ciks), "plan_filings": meta.get("filings"), "elapsed_s": round(time.time() - t0, 1),
                   "code_version": code_version}
            _write_json(os.path.join(root, "status.json"), {**out, "finished_at": _now_iso()})
            return out
    plan = load_plan(root)
    listing_day = plan_listing_day(meta)
    close_before = _shift(listing_day, -CLOSE_LAG_DAYS) if listing_day else None
    bf = Backfill(root, tmp_archive, make_fetcher(client, tmp_archive), since=since, max_rss_mb=max_rss_mb, close_before=close_before)
    res = bf.run(plan, deadline=deadline, max_filings=max_filings)
    out = {"status": "ok", "phase": "score", "plan_built": plan_built, "plan_filings": len(plan),
           "plan_ciks": len(meta.get("done_ciks") or []), **res, "code_version": code_version,
           "http": dict(getattr(client, "stats", None) or {})}
    _write_json(os.path.join(root, "status.json"), {**out, "finished_at": _now_iso()})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--budget-seconds", type=float, default=3.5 * 3600)
    ap.add_argument("--max-rss-mb", type=float, default=1500.0)
    ap.add_argument("--max-filings", type=int, default=None)
    ap.add_argument("--since", default=SINCE_DEFAULT)
    ap.add_argument("--code-version", default="unknown")
    ap.add_argument("--root", default=None, help="volume dir (default DATA_DIR/lazy_prices/backfill)")
    ap.add_argument("--pairs-dir", default=os.path.join(tempfile.gettempdir(), "voltrade_lazy_prices_pairs"))
    ap.add_argument("--tmp-archive", default=None, help="container tmp archive root for raw/sections")
    ap.add_argument("--ciks", default=None, help="comma-separated CIKs (smoke runs only; default = the dated CIK map)")
    ap.add_argument("--rate", type=float, default=sft.DEFAULT_RATE_PER_SEC)
    a = ap.parse_args(argv)
    if a.nice:
        try:
            os.nice(a.nice)
        except OSError as e:
            print(f"[lazy_prices_backfill] os.nice failed: {e}", file=sys.stderr)
    sft.process_limiter().set_rate(a.rate)
    root = a.root or default_root()
    try:
        out = run_once(root, a.pairs_dir, a.tmp_archive or default_tmp_archive(), a.since, a.budget_seconds,
                       a.max_rss_mb, a.code_version, max_filings=a.max_filings,
                       ciks=[c.strip() for c in a.ciks.split(",") if c.strip()] if a.ciks else None)
    except Exception as e:  # noqa: BLE001 — one JSON line for the scheduler; the checkpoint on disk is whatever was last saved
        out = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        try:
            _write_json(os.path.join(root, "status.json"), {**out, "finished_at": _now_iso()})
        except OSError as e2:
            print(f"[lazy_prices_backfill] could not write status.json: {e2}", file=sys.stderr)
    print(json.dumps(out, default=str))
    return 0 if out["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
