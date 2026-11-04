#!/usr/bin/env python3
"""
test_shadow_log_cache.py — [REPAIR] 2026-11-03 (memory fix A4): the shadow-log
read cache in shadow_portfolio._load_shadow_log / _save_shadow_log.

The loader used to json-parse the whole ~45 MB log on every call, several
times per candidate during a scan (11 call sites). It now parses once per
file change and serves copies; this file is the proof that nothing a caller
can observe changed.

WHAT IS PINNED
  1. IDENTITY — cache on, cache off and a direct json.load of the file give
     the same records (same values, same types, same key order) over a
     fixture carrying every persisted shape (_build_record with ml /
     score_breakdown / regime_snapshot / shadow_entry, +Nd outcomes, retry
     counters, exit-aligned labels, a NaN feature), after EVERY step of a
     scripted scan, and the file a scripted scan leaves behind is
     byte-identical with the cache on and off.
  2. ONE PARSE PER SCAN — a scripted scan against an unchanged file parses
     it once, counted by an independent json.load counter (not the cache's
     own counters), and N readonly loads parse once.
  3. FRESHNESS — a save, an atomic replace by "another process", an in-place
     rewrite and a deletion are each seen by the very next load.
  4. SOUNDNESS — a caller that modifies records and then fails to save
     (exception mid-batch, failed os.replace) is never observable through
     the cache, from the same thread or another; a list from a dropped
     generation can never be retained; concurrent writers and readers
     always see the file. The in-module readonly readers are pure.
  5. ROTATION at MAX_SHADOW_RECORDS is unchanged (in-place del == slice).
  6. FLAG — VOLTRADE_SHADOW_LOG_CACHE=0 restores the pre-cache path exactly
     (every call parses, plain lists); shadow_log_cache_stats() has the
     keys the health diagnostic reads.
  7. IDLE RELEASE — the parsed list is dropped after
     VOLTRADE_SHADOW_LOG_CACHE_TTL_S seconds without a use (default 120):
     a release is unobservable (the next load re-parses the same file),
     a use inside the TTL keeps it, TTL 0 never releases, and the default
     still gives one parse per scan.
"""
import copy
import fcntl
import json
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

import shadow_portfolio as sp

FIXED_NOW = datetime(2026, 11, 3, 15, 30, 0, tzinfo=timezone.utc)


class _FixedDatetime(datetime):
    """datetime whose now() is pinned, so two runs of the same scan write
    byte-identical records (timestamps are the only nondeterministic field)."""
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz is not None else FIXED_NOW.replace(tzinfo=None)


# ── helpers ──────────────────────────────────────────────────────────────────
def strict_equal(a, b, path="$"):
    """Structural identity: same type at every node (int is not float, list is
    list), same dict key ORDER, NaN equals NaN. Returns None or the first
    mismatch as a string."""
    if type(a) is not type(b) and not (isinstance(a, list) and isinstance(b, list)):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            return f"{path}: keys {list(a.keys())} != {list(b.keys())}"
        for k in a:
            m = strict_equal(a[k], b[k], f"{path}.{k}")
            if m:
                return m
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            m = strict_equal(x, y, f"{path}[{i}]")
            if m:
                return m
        return None
    if isinstance(a, float):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return None
        return f"{path}: {a!r} != {b!r}"
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def assert_strict_equal(a, b):
    m = strict_equal(a, b)
    assert m is None, m


def read_file():
    with open(sp.SHADOW_LOG_PATH) as f:
        return json.load(f)


def atomic_replace(records):
    """What _xa_merge_save and any other process do: mkstemp + os.replace."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(sp.SHADOW_LOG_PATH), prefix=".ext.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(records, f)
    os.replace(tmp, sp.SHADOW_LOG_PATH)


def fixture_records(n=120):
    """Every shape the log persists, so the identity check covers each one:
    _build_record output with every optional block, +Nd outcomes (filled and
    terminal), retry counters, exit-aligned labels, a NaN feature."""
    out = []
    decisions = ["taken", "rejected_score", "rejected_heat", "rejected_masterkill",
                 "rejected_other", "shadow_mode_entry"]
    for i in range(n):
        feats = {"rsi": 30.0 + i % 50, "change_pct_today": (i % 40) - 20.0, "atr_pct": 2.0 + (i % 7) / 10,
                 "spy_vs_ma50": 1.0 + (i % 5) / 100, "volume_ratio": 1 + i % 3}
        if i == 7:
            feats["obv_slope"] = float("nan")
        ml = {"win_probability": 0.4 + (i % 10) / 50, "model_type": "lgbm", "gated": i % 2 == 0,
              "final_score": 60.0 + i % 30} if i % 4 else None
        bd = {"momentum": 1.5, "vrp": -0.5, "regime_dip": 1, "rules_raw": 61.2} if i % 3 else None
        snap = {"state": "BULL", "p_bull": 0.71, "inputs": {"vix": 14.2}} if i % 5 == 0 else None
        se = {"sleeve": "stocks", "day": "2026-10-30", "qty": 10, "notional": 1200.5} if decisions[i % 6] == "shadow_mode_entry" else None
        rec = sp._build_record(f"T{i:03d}", feats, 55.0 + i % 40, decisions[i % 6], f"reason {i}",
                               10.0 + i, 1.0 + (i % 4) / 10, ["NEUTRAL", "BULL", "BEAR"][i % 3],
                               "buy", "BUY", "stock", ml, bd, snap, se)
        rec["timestamp"] = (FIXED_NOW - timedelta(days=i % 90, minutes=i)).isoformat()
        if i % 2 == 0:
            rec["outcomes"]["+5d"] = {"return_pct": round((i % 9) - 4.0, 3), "label": int(i % 9 > 4),
                                      "exit_price": round(10.0 + i + 0.25, 4), "exit_date": "2026-10-10",
                                      "exit_reason": "target", "label_method": "path_dependent_v2"}
            rec["outcomes"]["+10d"] = {"return_pct": -1.0, "label": 0, "exit_price": 9.9, "exit_date": "2026-10-12",
                                       "exit_reason": "stop", "label_method": "path_dependent_v2"}
        if i % 11 == 0:
            rec["_bf_attempts"] = {"+20d": 2}
        if i % 13 == 0:
            rec["outcomes"]["+20d"] = {"return_pct": None, "label": -1, "exit_price": None, "exit_date": None,
                                       "exit_reason": "no_price_data", "label_method": "path_dependent_v2"}
        if i % 3 == 0:
            rec[sp.XA_KEY] = {"label": 1, "return_pct": 2.5, "r_multiple": 0.625, "exit_reason": "trail",
                              "exit_date": "2026-10-20", "bars_held": 6, "censored": False, "scale_outs": 2,
                              "atr_pct": 2.1, "risk_pct": 4.2, "stop_pct": 4.2, "regime_class": "bullish",
                              "mfe_pct": 3.1, "mae_pct": -0.4, "label_method": sp.XA_LABEL_METHOD,
                              "atr_src": "feature", "atr_pct_prior": 2.05}
        elif i % 3 == 1:
            rec[sp.XA_ATTEMPTS_KEY] = 1
        out.append(rec)
    return out


def scripted_scan(n=40, check=None):
    """What one scan does to the log, in order: deep_score logs every
    candidate, the sizing / sector / spread filters relabel some and no-op
    on others, the Node admission loop relabels in one batch, entry-shadow
    records a batch. `check` runs after every call."""
    def _c():
        if check:
            check()
    for i in range(n):
        dec = "taken" if i % 3 == 0 else "rejected_score"
        sp.log_candidate(f"S{i:03d}", {"rsi": 40.0 + i, "change_pct_today": i / 7.0}, 60.0 + i % 20, dec,
                         "scripted", 10.0 + i, ml={"win_probability": 0.5, "model_type": "lgbm"},
                         score_breakdown={"momentum": 1.5})
        _c()
        if i % 6 == 0:
            assert sp.update_last_decision(f"S{i:03d}", "rejected_heat", "sector block") is True
            _c()
        if i % 5 == 1:
            still_taken = i % 3 == 0 and i % 6 != 0        # otherwise not "taken": a no-op that releases its lease
            assert sp.update_last_decision(f"S{i:03d}", "rejected_other", "spread") is still_taken
            _c()
    res = sp.update_decisions_batch([{"ticker": "S003", "decision": "rejected_size_cap", "reason": "cap"},
                                     {"ticker": "NOPE", "decision": "rejected_x", "reason": "none"}])
    assert (res["updated"], res["skipped"], res["saved"]) == (1, 1, True)
    _c()
    res = sp.log_shadow_entries([{"ticker": "S009", "sleeve": "stocks", "relabel": True, "score": 69.0, "price": 19.0,
                                  "details": {"source": "t", "day": "2026-11-03", "qty": 1}},
                                 {"ticker": "NEW1", "sleeve": "stocks", "score": 50.0, "price": 5.0,
                                  "details": {"source": "t", "day": "2026-11-03"}}])
    assert res["saved"] is True and len(res["results"]) == 2
    _c()


class _CountingJson:
    """Stands in for shadow_portfolio.json: counts json.load calls (the parses)
    and forwards everything else untouched."""
    def __init__(self, real):
        self._real = real
        self.load_calls = 0

    def load(self, *a, **k):
        self.load_calls += 1
        return self._real.load(*a, **k)

    def __getattr__(self, name):
        return getattr(self._real, name)


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "SHADOW_LOG_PATH", str(tmp_path / "shadow.json"))
    monkeypatch.setattr(sp, "SHADOW_LOCK_PATH", str(tmp_path / "shadow.json.lock"))
    monkeypatch.setattr(sp, "SHADOW_STATS_PATH", str(tmp_path / "shadow_stats.json"))
    monkeypatch.delenv(sp.SHADOW_LOG_CACHE_ENV, raising=False)
    monkeypatch.delenv(sp.SHADOW_LOG_CACHE_TTL_ENV, raising=False)
    monkeypatch.setattr(sp, "datetime", _FixedDatetime)
    sp._shadow_log_cache_reset()
    yield tmp_path
    sp._shadow_log_cache_reset()


@pytest.fixture
def counting_json(monkeypatch):
    cj = _CountingJson(json)
    monkeypatch.setattr(sp, "json", cj)
    return cj


def assert_served_equals_file():
    served = sp._load_shadow_log(readonly=True)
    assert_strict_equal(served, read_file())


# ── 1. identity ──────────────────────────────────────────────────────────────
def test_cache_on_off_and_the_file_agree_on_every_record(tmp_log, monkeypatch):
    atomic_replace(fixture_records())
    file_recs = read_file()

    monkeypatch.setenv(sp.SHADOW_LOG_CACHE_ENV, "0")
    off = sp._load_shadow_log()
    assert type(off) is list, "flag off: the pre-cache plain list"
    monkeypatch.delenv(sp.SHADOW_LOG_CACHE_ENV)
    sp._shadow_log_cache_reset()
    parsed = sp._load_shadow_log(readonly=True)      # the parse
    served = sp._load_shadow_log(readonly=True)      # the hit
    st = sp.shadow_log_cache_stats()
    assert (st["reloads"], st["hits"]) == (1, 1)

    assert_strict_equal(off, file_recs)
    assert_strict_equal(parsed, file_recs)
    assert_strict_equal(served, file_recs)
    assert isinstance(served, list) and len(served) == len(file_recs)
    assert math.isnan(served[7]["features"]["obv_slope"]), "NaN survives the round trip in every path"


def test_served_records_equal_the_file_after_every_step_of_a_scan(tmp_log):
    atomic_replace(fixture_records())
    scripted_scan(n=40, check=assert_served_equals_file)
    st = sp.shadow_log_cache_stats()
    assert st["reloads"] == 1, f"one parse for the whole scan, got {st}"
    assert st["leases"] == 0 and st["records"] == len(read_file())


def test_the_file_a_scan_leaves_is_byte_identical_with_the_cache_on_and_off(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "datetime", _FixedDatetime)
    outputs = {}
    for flag in ("1", "0"):
        d = tmp_path / f"flag{flag}"
        d.mkdir()
        monkeypatch.setattr(sp, "SHADOW_LOG_PATH", str(d / "shadow.json"))
        monkeypatch.setattr(sp, "SHADOW_LOCK_PATH", str(d / "shadow.json.lock"))
        monkeypatch.setenv(sp.SHADOW_LOG_CACHE_ENV, flag)
        sp._shadow_log_cache_reset()
        atomic_replace(fixture_records())
        scripted_scan(n=30)
        with open(sp.SHADOW_LOG_PATH, "rb") as f:
            outputs[flag] = f.read()
    sp._shadow_log_cache_reset()
    assert outputs["1"] == outputs["0"]
    assert len(json.loads(outputs["1"])) == 120 + 30 + 1


def test_every_persisted_shape_round_trips_json_exactly(tmp_log):
    """The cache retains the list a save wrote without re-parsing it, so the
    in-memory records must be exactly what json.load would give: this pins
    that every constructor of persisted data (the fixture covers them all)
    is JSON-stable, with the strict walker (types, key order, NaN)."""
    recs = fixture_records()
    assert_strict_equal(json.loads(json.dumps(recs)), recs)


# ── 2. one parse per scan ────────────────────────────────────────────────────
def test_a_scan_parses_the_log_once(tmp_log, counting_json):
    atomic_replace(fixture_records())
    scripted_scan(n=40)
    assert counting_json.load_calls == 1, f"{counting_json.load_calls} parses for one scan"
    st = sp.shadow_log_cache_stats()
    assert st["reloads"] == 1 and st["hits"] >= 40 and st["retained"] >= 40


def test_n_readonly_loads_parse_once(tmp_log, counting_json):
    atomic_replace(fixture_records(20))
    first = sp._load_shadow_log(readonly=True)
    for _ in range(50):
        assert_strict_equal(sp._load_shadow_log(readonly=True), first)
    assert counting_json.load_calls == 1
    assert sp.shadow_log_cache_stats()["hits"] == 50


def test_each_load_returns_the_callers_own_list(tmp_log):
    atomic_replace(fixture_records(5))
    a = sp._load_shadow_log(readonly=True)
    b = sp._load_shadow_log(readonly=True)
    a.append({"ticker": "JUNK"})
    del a[:2]
    assert len(b) == 5 and len(sp._load_shadow_log(readonly=True)) == 5, "list-level edits never reach the cache"


# ── 3. freshness ─────────────────────────────────────────────────────────────
def test_save_then_load_returns_the_new_records_without_a_parse(tmp_log, counting_json):
    atomic_replace(fixture_records(10))
    recs = sp._load_shadow_log()
    recs.append(sp._build_record("NEWX", {"rsi": 1.0}, 50.0, "rejected_score", "x", 3.0))
    assert sp._save_shadow_log(recs) is True
    served = sp._load_shadow_log(readonly=True)
    assert served[-1]["ticker"] == "NEWX" and len(served) == 11
    assert_strict_equal(served, read_file())
    assert counting_json.load_calls == 1, "the save retained what it wrote; no second parse"


def test_an_atomic_replace_by_another_process_is_picked_up(tmp_log):
    atomic_replace(fixture_records(10))
    assert len(sp._load_shadow_log(readonly=True)) == 10
    atomic_replace(fixture_records(12))
    served = sp._load_shadow_log(readonly=True)
    assert len(served) == 12
    assert_strict_equal(served, read_file())
    assert sp.shadow_log_cache_stats()["reloads"] == 2


def test_an_in_place_rewrite_is_picked_up(tmp_log):
    atomic_replace(fixture_records(10))
    assert len(sp._load_shadow_log(readonly=True)) == 10
    with open(sp.SHADOW_LOG_PATH, "w") as f:   # same inode, different size
        json.dump(fixture_records(3), f)
    st = os.stat(sp.SHADOW_LOG_PATH)
    os.utime(sp.SHADOW_LOG_PATH, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # distinct mtime on any kernel
    served = sp._load_shadow_log(readonly=True)
    assert len(served) == 3
    assert_strict_equal(served, read_file())


def test_a_deleted_log_serves_empty_and_the_next_save_is_seen(tmp_log):
    atomic_replace(fixture_records(4))
    assert len(sp._load_shadow_log(readonly=True)) == 4
    os.remove(sp.SHADOW_LOG_PATH)                       # the CLI `clear` path
    empty = sp._load_shadow_log()
    assert empty == [] and sp.shadow_log_cache_stats()["records"] is None
    sp.log_candidate("AFTER", {"rsi": 2.0}, 50.0, "taken", "x", 5.0)
    served = sp._load_shadow_log(readonly=True)
    assert [r["ticker"] for r in served] == ["AFTER"]
    assert_strict_equal(served, read_file())


def test_an_undecodable_log_still_serves_empty(tmp_log, caplog):
    with open(sp.SHADOW_LOG_PATH, "w") as f:
        f.write("{not json")
    with caplog.at_level("WARNING", logger="voltrade.shadow"):
        assert sp._load_shadow_log() == []
    assert "undecodable" in caplog.text, "the pre-cache loader hid this; now it is logged"


# ── 4. soundness ─────────────────────────────────────────────────────────────
def _fresh_taken(ticker="AMD"):
    rec = sp._build_record(ticker, {"rsi": 55.0}, 72.0, "taken", "score ok", 120.0)
    rec["timestamp"] = (FIXED_NOW - timedelta(seconds=30)).isoformat()
    return rec


def test_edits_followed_by_an_exception_never_reach_a_reader(tmp_log):
    """log_shadow_entries relabels the AMD record, then the second entry raises
    inside _build_record before the save. Two mechanisms keep the reader on the
    file: the relabel replaced the record in the caller's own list (the shared
    dict is untouched) and the caller's lease is still outstanding."""
    atomic_replace(fixture_records(10) + [_fresh_taken()])
    sp._load_shadow_log(readonly=True)                   # warm the cache
    res = sp.log_shadow_entries([
        {"ticker": "AMD", "sleeve": "stocks", "relabel": True, "details": {"source": "x", "day": "2026-11-03"}},
        {"ticker": "BAD", "sleeve": "stocks", "score": "not-a-number"},
    ])
    assert res["saved"] is False and "error" in res
    served = sp._load_shadow_log(readonly=True)
    assert served[-1]["decision"] == "taken"
    assert_strict_equal(served, read_file())
    assert sp.shadow_log_cache_stats()["reloads"] == 2, "the outstanding lease forced a re-parse"


def test_in_flight_edits_are_invisible_to_other_threads_until_saved(tmp_log):
    atomic_replace(fixture_records(10) + [_fresh_taken()])
    cached_before = sp._load_shadow_log(readonly=True)
    mine = sp._load_shadow_log()                         # this thread's lease
    assert sp._relabel_latest_taken(mine, "AMD", "rejected_heat", "sector", FIXED_NOW, 120.0)
    assert cached_before[-1]["decision"] == "taken", "copy-on-write: the shared dict was not edited"
    assert mine[-1] is not cached_before[-1] and mine[-1]["decision"] == "rejected_heat"
    seen = {}

    def other():
        seen["before"] = sp._load_shadow_log(readonly=True)[-1]["decision"]

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert seen["before"] == "taken", "another thread sees the file, not my unsaved edit"
    assert sp._save_shadow_log(mine) is True
    assert sp._load_shadow_log(readonly=True)[-1]["decision"] == "rejected_heat"
    assert_strict_equal(sp._load_shadow_log(readonly=True), read_file())


def test_a_failed_save_invalidates_the_cache(tmp_log, monkeypatch):
    atomic_replace(fixture_records(10))
    sp._load_shadow_log(readonly=True)
    real_replace = os.replace
    calls = []

    def failing_replace(src, dst):
        calls.append(dst)
        raise OSError("disk full")

    monkeypatch.setattr(sp.os, "replace", failing_replace)
    sp.log_candidate("LOST", {"rsi": 1.0}, 50.0, "taken", "x", 1.0)   # swallows the failure
    monkeypatch.setattr(sp.os, "replace", real_replace)
    assert calls, "the save was attempted"
    served = sp._load_shadow_log(readonly=True)
    assert all(r["ticker"] != "LOST" for r in served)
    assert_strict_equal(served, read_file())
    assert not [f for f in os.listdir(tmp_log) if f.startswith(".shadow.")], "no orphaned temp file"


def test_a_list_from_a_dropped_generation_is_never_retained(tmp_log):
    atomic_replace(fixture_records(10))
    old = sp._load_shadow_log()                          # generation 1, leased
    atomic_replace(fixture_records(12))                  # another process writes
    assert len(sp._load_shadow_log(readonly=True)) == 12  # generation 2
    before = sp.shadow_log_cache_stats()
    sp._shadow_log_cache_release(old)                    # stale: a no-op
    assert sp.shadow_log_cache_stats()["lease_releases"] == before["lease_releases"]
    old.append(sp._build_record("STALE", {"rsi": 1.0}, 50.0, "taken", "x", 1.0))
    assert sp._save_shadow_log(old) is True              # the file is now `old` (11 records)
    assert sp.shadow_log_cache_stats()["invalidations"] == before["invalidations"] + 1
    served = sp._load_shadow_log(readonly=True)          # parsed, not served from stale memory
    assert len(served) == 11 and served[-1]["ticker"] == "STALE"
    assert_strict_equal(served, read_file())


def test_the_readonly_readers_are_pure_and_keep_the_cache_servable(tmp_log):
    atomic_replace(fixture_records())
    before = copy.deepcopy(sp._load_shadow_log(readonly=True))
    sp.get_shadow_stats()
    sp.get_shadow_band_report()
    sp.get_shadow_band_report_deduped()
    sp.load_shadow_data(horizon_days=5, min_records=1)
    st = sp.shadow_log_cache_stats()
    assert st["leases"] == 0 and st["reloads"] == 1, "pure readers take no lease and parse nothing"
    assert_strict_equal(sp._load_shadow_log(readonly=True), before)
    assert_strict_equal(before, read_file())


def test_a_readonly_snapshot_is_stable_while_a_scan_edits_the_log(tmp_log):
    """The hot-path relabels replace records instead of editing them in place,
    so a snapshot served before the scan is still that file image afterwards
    (its dicts are shared with the cache — the scan must not touch them)."""
    atomic_replace(fixture_records())
    snapshot = sp._load_shadow_log(readonly=True)
    frozen = copy.deepcopy(snapshot)
    scripted_scan(n=20)
    assert_strict_equal(snapshot, frozen)
    assert sp.shadow_log_cache_stats()["reloads"] == 1, "and the scan still parsed nothing"
    assert_strict_equal(sp._load_shadow_log(readonly=True), read_file())


def _bars_after(ts_iso, entry, n=12):
    day = datetime.fromisoformat(ts_iso).date()
    return [{"t": (day + timedelta(days=k)).isoformat() + "T05:00:00Z", "o": entry, "h": entry * 1.01,
             "l": entry * 0.99, "c": entry * 1.005, "v": 1000} for k in range(1, n + 1)]


def test_backfill_outcomes_edits_a_private_parse_and_invalidates_the_cache(tmp_log, monkeypatch):
    """backfill_outcomes writes outcomes INTO record dicts; it works on its own
    parse, so a snapshot served before it never changes, and its save
    invalidates the cache so the next load is the file."""
    recs = fixture_records(6)
    for r in recs:
        r["timestamp"] = (FIXED_NOW - timedelta(days=60)).isoformat()
        r["outcomes"] = {"+5d": None, "+10d": None, "+20d": None}
        r.pop("_bf_attempts", None)
    atomic_replace(recs)
    snapshot = sp._load_shadow_log(readonly=True)
    frozen = copy.deepcopy(snapshot)
    monkeypatch.setattr(sp, "_fetch_historical_bars_batch",
                        lambda tickers, start, end: {t: _bars_after(recs[0]["timestamp"], 10.0 + int(t[1:])) for t in tickers})
    stats = sp.backfill_outcomes(max_records=100)
    assert stats["updated"] == 18 and stats["total_records"] == 6
    assert_strict_equal(snapshot, frozen), "the private parse shares no dicts with the snapshot"
    st = sp.shadow_log_cache_stats()
    assert st["invalidations"] >= 1, "an untagged save invalidates"
    # backfill_outcomes ends with get_shadow_stats(), a readonly load that re-parsed the saved file
    assert st["reloads"] == 3 and st["records"] == 6, st
    served = sp._load_shadow_log(readonly=True)
    assert all(isinstance(r["outcomes"]["+5d"], dict) for r in served)
    assert_strict_equal(served, read_file())
    assert sp.shadow_log_cache_stats()["leases"] == 0


def test_concurrent_writers_and_readers_always_see_the_file(tmp_log):
    atomic_replace(fixture_records(30))
    errors = []
    stop = threading.Event()

    def writer(k):
        try:
            for i in range(25):
                sp.log_candidate(f"W{k}{i:02d}", {"rsi": float(i)}, 60.0, "taken" if i % 2 else "rejected_score", "c", 10.0)
                if i % 5 == 0:
                    sp.update_decisions_batch([{"ticker": f"W{k}{i:02d}", "decision": "rejected_x", "reason": "r"}])
                if i % 7 == 0:
                    sp.get_shadow_stats()
        except Exception as e:  # pragma: no cover - surfaced below
            errors.append(f"writer {k}: {e!r}")
        finally:
            stop.set()

    def checker():
        try:
            n = 0
            while not stop.is_set() or n < 5:
                n += 1
                with open(sp.SHADOW_LOCK_PATH, "a+") as lf:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)   # the writers' own lock: the file is still while held
                    try:
                        m = strict_equal(sp._load_shadow_log(readonly=True), read_file())
                    finally:
                        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                if m:
                    errors.append(m)
                    return
        except Exception as e:  # pragma: no cover
            errors.append(f"checker: {e!r}")

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(3)] + \
              [threading.Thread(target=checker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors[:3]
    assert_strict_equal(sp._load_shadow_log(readonly=True), read_file())
    assert sp.shadow_log_cache_stats()["leases"] == 0


# ── 5. rotation ──────────────────────────────────────────────────────────────
def test_rotation_at_the_cap_keeps_exactly_the_newest_records(tmp_log, monkeypatch):
    monkeypatch.setattr(sp, "MAX_SHADOW_RECORDS", 5)
    seed = fixture_records(5)
    atomic_replace(seed)
    for i in range(3):
        sp.log_candidate(f"R{i}", {"rsi": 1.0}, 50.0, "rejected_score", "x", 1.0)
    expected = [r["ticker"] for r in seed] + ["R0", "R1", "R2"]
    assert [r["ticker"] for r in read_file()] == expected[-5:]
    assert_strict_equal(sp._load_shadow_log(readonly=True), read_file())
    res = sp.log_shadow_entries([{"ticker": "E1", "sleeve": "stocks", "score": 1.0, "price": 1.0,
                                  "details": {"source": "t", "day": "2026-11-03"}}])
    assert res["saved"] is True
    assert [r["ticker"] for r in read_file()] == (expected + ["E1"])[-5:]
    assert sp.shadow_log_cache_stats()["reloads"] == 1, "in-place rotation keeps the generation: no re-parse"


def test_in_place_rotation_equals_the_old_slice():
    for n in range(6, 10):
        lst = list(range(n))
        sliced = lst[-5:]
        del lst[:len(lst) - 5]
        assert lst == sliced


# ── 6. flag + diagnostics + list semantics ───────────────────────────────────
def test_flag_off_parses_every_call_and_returns_plain_lists(tmp_log, counting_json, monkeypatch):
    atomic_replace(fixture_records(6))
    monkeypatch.setenv(sp.SHADOW_LOG_CACHE_ENV, "0")
    for _ in range(4):
        recs = sp._load_shadow_log()
        assert type(recs) is list
    assert counting_json.load_calls == 4
    sp.log_candidate("OFF", {"rsi": 1.0}, 50.0, "taken", "x", 1.0)
    assert counting_json.load_calls == 5 and read_file()[-1]["ticker"] == "OFF"
    st = sp.shadow_log_cache_stats()
    assert st["enabled"] is False and st["hits"] == 0 and st["records"] is None


def test_cache_stats_have_the_diagnostic_keys(tmp_log):
    for k in ("enabled", "hits", "misses", "reloads", "records", "last_stat_ns"):
        assert k in sp.shadow_log_cache_stats()
    atomic_replace(fixture_records(7))
    sp._load_shadow_log(readonly=True)
    st = sp.shadow_log_cache_stats()
    assert st["enabled"] is True and st["records"] == 7
    assert st["last_stat_ns"] == os.stat(sp.SHADOW_LOG_PATH).st_mtime_ns
    assert st["path"] == sp.SHADOW_LOG_PATH


def test_the_returned_list_behaves_exactly_like_a_list(tmp_log):
    atomic_replace(fixture_records(4))
    recs = sp._load_shadow_log(readonly=True)
    plain = read_file()
    assert isinstance(recs, list) and recs == plain
    assert json.dumps(recs) == json.dumps(plain), "the save writes the same bytes as a plain list"
    assert copy.deepcopy(recs) == recs
    assert type(recs[1:]) is list
    recs.append({"x": 1})
    del recs[:1]
    assert type(recs) is sp._ShadowRecords and len(recs) == 4


# ── 7. idle release ──────────────────────────────────────────────────────────
def _until(pred, timeout=8.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_a_forced_idle_release_is_unobservable(tmp_log):
    atomic_replace(fixture_records())
    first = sp._load_shadow_log(readonly=True)
    c = sp._SHADOW_CACHE
    with c.lock:
        timer = c.timer
        assert timer is not None, "a use arms the idle timer"
        timer.cancel()
        c.last_used -= sp._shadow_log_cache_ttl_s() + 1      # pretend the TTL of idleness elapsed
    sp._shadow_cache_idle_fire(timer)                          # exactly what the timer thread runs
    st = sp.shadow_log_cache_stats()
    assert st["records"] is None and st["releases"] == 1 and st["timer_pending"] is False
    assert isinstance(st["last_release_at"], float) and st["idle_s"] is None
    again = sp._load_shadow_log(readonly=True)
    assert_strict_equal(again, first)
    assert_strict_equal(again, read_file())
    st = sp.shadow_log_cache_stats()
    assert st["reloads"] == 2 and st["generation"] == 2 and st["records"] == len(first) and st["timer_pending"]


def test_a_release_during_an_in_flight_edit_still_lands_the_save(tmp_log):
    """Generation and leases survive a release, so a caller that loaded before
    the release saves normally and its list is retained."""
    atomic_replace(fixture_records(10) + [_fresh_taken()])
    mine = sp._load_shadow_log()
    assert sp._relabel_latest_taken(mine, "AMD", "rejected_heat", "sector", FIXED_NOW, 120.0)
    c = sp._SHADOW_CACHE
    with c.lock:
        timer = c.timer
        timer.cancel()
        c.last_used -= sp._shadow_log_cache_ttl_s() + 1
    sp._shadow_cache_idle_fire(timer)
    assert sp.shadow_log_cache_stats()["records"] is None
    assert sp._save_shadow_log(mine) is True
    st = sp.shadow_log_cache_stats()
    assert st["records"] == 11 and st["leases"] == 0 and st["retained"] == 1
    assert_strict_equal(sp._load_shadow_log(readonly=True), read_file())
    assert sp.shadow_log_cache_stats()["reloads"] == 1, "the retained list served; no parse"


def test_the_timer_releases_only_after_the_ttl_of_idleness(tmp_log, monkeypatch):
    monkeypatch.setenv(sp.SHADOW_LOG_CACHE_TTL_ENV, "1.5")
    atomic_replace(fixture_records(20))
    sp._load_shadow_log(readonly=True)
    assert sp.shadow_log_cache_stats()["timer_pending"] is True
    time.sleep(0.6)
    sp._load_shadow_log(readonly=True)                          # a use inside the TTL
    time.sleep(0.6)                                             # 1.2 s since the first use, 0.6 since the last
    st = sp.shadow_log_cache_stats()
    assert st["records"] == 20 and st["releases"] == 0, f"used inside the TTL: kept ({st})"
    assert _until(lambda: sp.shadow_log_cache_stats()["records"] is None), "idle past the TTL: released"
    st = sp.shadow_log_cache_stats()
    assert st["releases"] == 1 and st["timer_pending"] is False and st["last_release_at"] is not None
    assert_strict_equal(sp._load_shadow_log(readonly=True), read_file())
    assert sp.shadow_log_cache_stats()["reloads"] == 2


def test_ttl_zero_never_releases(tmp_log, monkeypatch):
    monkeypatch.setenv(sp.SHADOW_LOG_CACHE_TTL_ENV, "0")
    atomic_replace(fixture_records(5))
    sp._load_shadow_log(readonly=True)
    st = sp.shadow_log_cache_stats()
    assert st["ttl_s"] == 0.0 and st["timer_pending"] is False, "no timer is ever armed"
    c = sp._SHADOW_CACHE
    with c.lock:
        c.last_used -= 10_000
    sp._shadow_cache_idle_fire(None)                            # even a stray fire releases nothing
    st = sp.shadow_log_cache_stats()
    assert st["records"] == 5 and st["releases"] == 0


def test_default_ttl_still_gives_one_parse_per_scan(tmp_log, counting_json):
    assert sp._shadow_log_cache_ttl_s() == 120.0
    atomic_replace(fixture_records())
    scripted_scan(n=40)
    assert counting_json.load_calls == 1
    st = sp.shadow_log_cache_stats()
    assert st["ttl_s"] == 120.0 and st["releases"] == 0 and st["timer_pending"] is True
    assert st["idle_s"] is not None and st["idle_s"] < 120


def test_a_bad_ttl_falls_back_to_the_default_and_logs(tmp_log, monkeypatch, caplog):
    monkeypatch.setenv(sp.SHADOW_LOG_CACHE_TTL_ENV, "soon")
    with caplog.at_level("WARNING", logger="voltrade.shadow"):
        assert sp._shadow_log_cache_ttl_s() == 120.0
    assert "not a number" in caplog.text
    monkeypatch.setenv(sp.SHADOW_LOG_CACHE_TTL_ENV, "-5")
    assert sp._shadow_log_cache_ttl_s() == 0.0, "negative means never"
