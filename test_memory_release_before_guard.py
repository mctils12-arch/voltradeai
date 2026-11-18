"""
KNOWN BROKEN #55 (2026-11-18) — freed heap goes back to the OS before the
deep-score memory guard reads RSS.

The first reading taken 5.4 h after a deploy with scans running
(/api/diag/daemon, 2026-11-18 20:16Z): RSS 647 MB; the C heap held 216.9 MB
freed-but-retained against 80.3 MB in use; the guard had skipped deep scoring
on 24 scans and trimmed it on 27 of 53. glibc returns only the top of an
arena by itself, so memory a scan freed on a worker thread stayed resident
and the guard judged it as held.

These tests pin: the release really lowers RSS for memory freed on worker
threads (the regression: without it the same RSS stays past the skip line),
it runs before both guards read memory, it has an off switch, it never
raises, and the daemon and the session health check report what it returned.
"""
import ctypes
import gc
import importlib.util
import inspect
import os
import random
import threading

import pytest

import bot_engine
import voltrade_daemon as vd

_spec = importlib.util.spec_from_file_location(
    "session_health_check_release",
    os.path.join(os.path.dirname(__file__), "scripts", "session_health_check.py"))
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)


def _has_malloc_trim():
    try:
        ctypes.CDLL(None).malloc_trim
        return os.path.exists("/proc/self/status")
    except (OSError, AttributeError):
        return False


@pytest.fixture(autouse=True)
def _fresh_release_stats():
    saved = bot_engine._MEMORY_RELEASE_STATS
    bot_engine._MEMORY_RELEASE_STATS = {
        "calls": 0, "released_mb_total": 0, "unavailable": 0, "last": None}
    yield
    bot_engine._MEMORY_RELEASE_STATS = saved




class _TrimFn:
    """Stands in for a ctypes function: accepts argtypes/restype."""
    def __init__(self, sink):
        self.sink = sink

    def __call__(self, pad):
        self.sink.append(pad)
        return 1


class _Libc:
    def __init__(self):
        self.pads = []
        self.malloc_trim = _TrimFn(self.pads)


# ── the regression: memory freed on worker threads stays resident ────────

@pytest.mark.skipif(not _has_malloc_trim(), reason="needs glibc malloc_trim and /proc")
def test_memory_freed_on_worker_threads_is_returned_and_the_guard_decision_changes():
    keep = []
    baseline = bot_engine._mem_rss_mb()

    def work(seed):
        # ~120 MB of touched mid-size blocks per thread, freed with
        # scattered survivors so the arena's top cannot shrink by itself.
        r = random.Random(seed)
        blobs = [b"x" * r.randint(600, 60_000) for _ in range(4000)]
        keep.extend(blobs[::100])
        del blobs

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    gc.collect()
    retained = bot_engine._mem_rss_mb()
    assert retained - baseline >= 100, (baseline, retained)   # the break: freed, still resident
    row = bot_engine._release_freed_memory(env={})
    assert row is not None
    assert row["before_mb"] == retained
    assert row["released_mb"] >= 0.6 * (retained - baseline), (baseline, row)
    assert bot_engine._mem_rss_mb() <= retained - 60
    # The guard, at the daemon's thresholds, on the 2026-11-18 reading:
    # 647 MB skips; with the 216.9 MB of retained heap returned it does not.
    assert bot_engine._deep_score_guard_decision(647, 550, 400) == "skip"
    assert bot_engine._deep_score_guard_decision(647 - 216, 550, 400) == "trim"
    assert len(keep) == 80                          # survivors untouched


# ── behaviour pinned without a real heap ─────────────────────────────────

def test_release_calls_malloc_trim_zero_and_counts_what_came_back():
    libc = _Libc()
    readings = iter([647, 431])
    row = bot_engine._release_freed_memory(libc=libc, env={}, rss_fn=lambda: next(readings), now=9.0)
    assert libc.pads == [0]
    assert row["before_mb"] == 647 and row["after_mb"] == 431 and row["released_mb"] == 216
    assert row["at"] == 9.0 and row["ms"] >= 0
    st = bot_engine.get_memory_release_stats()
    assert st["calls"] == 1 and st["released_mb_total"] == 216 and st["last"] == row
    st["last"]["released_mb"] = -1                  # a copy
    assert bot_engine.get_memory_release_stats()["last"]["released_mb"] == 216


def test_a_rise_in_rss_is_reported_as_zero_released_not_negative():
    readings = iter([400, 405])
    row = bot_engine._release_freed_memory(libc=_Libc(), env={}, rss_fn=lambda: next(readings))
    assert row["released_mb"] == 0


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_the_off_switch_does_nothing(value):
    libc = _Libc()
    assert bot_engine._release_freed_memory(libc=libc, env={"VOLTRADE_MALLOC_TRIM": value}) is None
    assert libc.pads == []
    assert bot_engine.get_memory_release_stats() == {}


def test_on_by_default_and_when_set_to_one():
    for env in ({}, {"VOLTRADE_MALLOC_TRIM": "1"}):
        libc = _Libc()
        assert bot_engine._release_freed_memory(libc=libc, env=env, rss_fn=lambda: 300) is not None
        assert libc.pads == [0]


def test_a_c_library_without_malloc_trim_never_raises_and_is_counted():
    class NoTrim:
        pass
    assert bot_engine._release_freed_memory(libc=NoTrim(), env={}, rss_fn=lambda: 300) is None
    st = bot_engine.get_memory_release_stats()
    assert st["unavailable"] == 1 and st["calls"] == 0 and st["last"] is None


def test_no_stats_before_the_first_release():
    assert bot_engine.get_memory_release_stats() == {}


# ── wiring ───────────────────────────────────────────────────────────────

def test_the_scan_releases_before_either_memory_guard_reads_rss():
    src = inspect.getsource(bot_engine._scan_market_inner)
    release = src.index("_release_freed_memory()")
    assert release < src.index('_timing_log("quick_scan")')          # timed inside quick_scan
    assert release < src.index("VOLTRADE_SURVIVAL_THRESHOLD_MB")     # the 700 MB guard
    assert release < src.index('_log_mem_phase("pre_deep_score")')   # the 550/400 guard
    assert src.count("_release_freed_memory()") == 1


def test_the_guard_thresholds_are_unchanged():
    src = inspect.getsource(bot_engine._scan_market_inner)
    assert '"VOLTRADE_MEM_SKIP_DEEP_MB", "550"' in src
    assert '"VOLTRADE_MEM_TRIM_DEEP_MB", "400"' in src
    assert '"VOLTRADE_SURVIVAL_THRESHOLD_MB", "700"' in src


def test_the_daemon_serves_the_release_with_the_guard(monkeypatch):
    saved = bot_engine._DEEP_GUARD_STATS
    bot_engine._DEEP_GUARD_STATS = {
        "counts": {d: 0 for d in bot_engine._DEEP_GUARD_DECISIONS},
        "consecutive_not_normal": 0, "last": None}
    try:
        readings = iter([647, 431])
        bot_engine._release_freed_memory(libc=_Libc(), env={}, rss_fn=lambda: next(readings))
        bot_engine._record_deep_guard("trim", 431, 400)
        snap = vd._deep_guard_snapshot()
        assert snap["release"]["last"]["released_mb"] == 216
        assert snap["release"]["calls"] == 1
        assert snap["last"]["decision"] == "trim"
    finally:
        bot_engine._DEEP_GUARD_STATS = saved


def test_the_health_check_prints_what_was_returned():
    guard = {"counts": {"normal": 3, "trim": 0, "skip": 0, "survival": 0},
             "consecutive_not_normal": 0,
             "last": {"decision": "normal", "rss_mb": 380},
             "release": {"calls": 3, "released_mb_total": 500, "unavailable": 0,
                         "last": {"before_mb": 596, "after_mb": 380, "released_mb": 216, "ms": 14.2}}}
    f = hc._deep_guard_verdict(guard, "daemon rss now=380MB")
    text = str(f)
    assert "216MB" in text and "596->380MB" in text and "500MB over 3 scan(s)" in text


def test_the_health_check_says_when_malloc_trim_is_unavailable():
    guard = {"counts": {"normal": 1}, "consecutive_not_normal": 0,
             "last": {"decision": "normal", "rss_mb": 380},
             "release": {"calls": 0, "released_mb_total": 0, "unavailable": 4, "last": None}}
    assert "malloc_trim unavailable on 4 scan(s)" in str(hc._deep_guard_verdict(guard, "x"))


def test_a_daemon_that_reports_no_release_reads_as_before():
    guard = {"counts": {"normal": 1}, "consecutive_not_normal": 0,
             "last": {"decision": "normal", "rss_mb": 380}}
    assert "returned to the OS" not in str(hc._deep_guard_verdict(guard, "x"))
