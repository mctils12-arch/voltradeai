"""
KNOWN BROKEN #55 (2026-11-17) — the daemon's memory, measured.

Live on 2026-11-17 ~20:20Z the daemon held ~640 MB whether idle or scanning
(278 MB at 13:03Z, same process family, 6.4 h of uptime), which is past the
deep-score guard's 550 MB skip line. /api/diag/timings showed the deep_score
phase taking 0 s: deep scoring was skipped on every scan and candidates went
to the trade loop on quick_score alone. Nothing readable said so (the guard
wrote to stderr only), and nothing said what the memory was made of.

These tests pin the instrument: the guard's decisions are counted and served
by the daemon's health RPC, every scan phase records the memory held at its
end, and the health RPC reports the C heap's in-use vs freed-but-retained
split, the interpreter's block count against its start, and the longest
module-level containers. The session health check then alarms on what the
guard DID, not on a guess from one memory figure.
"""
import importlib.util
import inspect
import os
import types

import pytest

import bot_engine
import voltrade_daemon as vd

_spec = importlib.util.spec_from_file_location(
    "session_health_check_mem",
    os.path.join(os.path.dirname(__file__), "scripts", "session_health_check.py"))
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)


@pytest.fixture(autouse=True)
def _fresh_guard_stats():
    saved = bot_engine._DEEP_GUARD_STATS
    bot_engine._DEEP_GUARD_STATS = {
        "counts": {d: 0 for d in bot_engine._DEEP_GUARD_DECISIONS},
        "consecutive_not_normal": 0, "last": None}
    yield
    bot_engine._DEEP_GUARD_STATS = saved


# ── bot_engine: the guard's decisions are counted ────────────────────────

def test_no_stats_before_the_first_scan():
    assert bot_engine.get_deep_guard_stats() == {}


def test_each_decision_is_counted_and_the_run_of_non_normal_scans_tracked():
    bot_engine._record_deep_guard("normal", 300, None, now=1.0)
    bot_engine._record_deep_guard("trim", 420, 400, now=2.0)
    bot_engine._record_deep_guard("skip", 640, 550, now=3.0)
    st = bot_engine._record_deep_guard("survival", 710, 700, now=4.0)
    assert st["counts"] == {"normal": 1, "trim": 1, "skip": 1, "survival": 1}
    assert st["scans"] == 4
    assert st["consecutive_not_normal"] == 3
    assert st["last"] == {"decision": "survival", "rss_mb": 710, "threshold_mb": 700, "at": 4.0}
    st = bot_engine._record_deep_guard("normal", 300, None, now=5.0)
    assert st["consecutive_not_normal"] == 0


def test_an_unknown_decision_is_refused_not_counted():
    with pytest.raises(ValueError):
        bot_engine._record_deep_guard("skipped", 640)
    assert bot_engine.get_deep_guard_stats() == {}


def test_the_snapshot_is_a_copy():
    st = bot_engine._record_deep_guard("skip", 640, 550, now=1.0)
    st["counts"]["skip"] = 99
    st["last"]["decision"] = "normal"
    again = bot_engine.get_deep_guard_stats()
    assert again["counts"]["skip"] == 1 and again["last"]["decision"] == "skip"


def test_the_scan_records_the_guard_and_the_survival_return():
    """Wiring: scan_market's own guard and its 700 MB survival return both
    record, with the decision the guard function returned (the live value of
    2026-11-17, 640 MB against 550/400, is a skip)."""
    src = inspect.getsource(bot_engine._scan_market_inner)
    guard = src.index("_guard_decision = _deep_score_guard_decision(")
    assert "_record_deep_guard(" in src[guard:guard + 400]
    assert '_record_deep_guard("survival", _surv_mb, _survival_threshold_mb)' in src
    assert bot_engine._deep_score_guard_decision(640, 550, 400) == "skip"


def test_every_scan_phase_row_carries_the_memory_held_at_its_end():
    src = inspect.getsource(bot_engine._scan_market_inner)
    row = src.index('_phase_times.append({"phase": phase')
    assert '"rss_mb": _mem_rss_mb()' in src[row:row + 200]


# ── voltrade_daemon: what the memory is made of ──────────────────────────

class _FakeLibc:
    def __init__(self, **fields):
        info = vd._Mallinfo2(**fields)
        self.mallinfo2 = _Settable(lambda: info)


class _Settable:
    """A callable that accepts the `restype` assignment ctypes functions take."""
    def __init__(self, fn):
        self._fn = fn
        self.restype = None

    def __call__(self):
        return self._fn()


def test_malloc_snapshot_splits_in_use_from_freed_but_retained():
    mb = 1024 * 1024
    libc = _FakeLibc(arena=500 * mb, uordblks=120 * mb, fordblks=380 * mb,
                     hblkhd=40 * mb, keepcost=2 * mb)
    assert vd._malloc_snapshot(libc) == {
        "in_use_mb": 160.0, "free_retained_mb": 380.0, "mmapped_mb": 40.0,
        "heap_mb": 500.0, "releasable_top_mb": 2.0}


def test_malloc_snapshot_is_none_without_mallinfo2():
    assert vd._malloc_snapshot(object()) is None


def test_malloc_snapshot_reads_this_process():
    snap = vd._malloc_snapshot()
    if snap is None:
        pytest.skip("no glibc mallinfo2 on this platform")
    assert snap["in_use_mb"] > 0 and snap["free_retained_mb"] >= 0


def _mod(name, path, **attrs):
    m = types.ModuleType(name)
    m.__file__ = path
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def test_largest_containers_names_the_longest_repo_module_globals():
    repo = vd._REPO_DIR
    mods = {
        "bot_engine": _mod("bot_engine", os.path.join(repo, "bot_engine.py"),
                           _BARS_CACHE={i: i for i in range(5000)}, _small={1: 1},
                           _seen=set(range(900)), NAME="x" * 10000),
        "csp_universe": _mod("csp_universe", os.path.join(repo, "csp_universe.py"),
                             _ROWS=list(range(12000))),
        "pandas": _mod("pandas", "/usr/lib/python3/site-packages/pandas/__init__.py",
                       _big=list(range(99999))),
        "vendored": _mod("vendored", os.path.join(repo, "node_modules", "x", "site-packages", "y.py"),
                         _big=list(range(99999))),
        "builtin": types.ModuleType("builtin"),
    }
    assert vd._largest_containers(mods) == [
        {"name": "csp_universe._ROWS", "type": "list", "len": 12000},
        {"name": "bot_engine._BARS_CACHE", "type": "dict", "len": 5000},
        {"name": "bot_engine._seen", "type": "set", "len": 900},
    ]
    assert len(vd._largest_containers(mods, top=1)) == 1


def test_health_reports_the_guard_and_the_memory_against_start():
    bot_engine._record_deep_guard("skip", 640, 550)
    vd.sys.modules["bot_engine"] = bot_engine
    h = vd._dispatcher._health({})
    assert h["deep_guard"]["counts"]["skip"] == 1
    assert h["deep_guard"]["last"]["decision"] == "skip"
    assert h["deep_guard"]["last"]["age_sec"] >= 0
    mem = h["memory"]
    assert set(mem) == {"rss_current_mb", "python_blocks", "malloc", "threads",
                        "at_start", "largest_containers"}
    assert mem["python_blocks"] > 0 and mem["threads"] >= 1
    assert set(mem["at_start"]) == {"rss_current_mb", "python_blocks", "malloc", "threads"}
    # the fields the 2026-11-16 reading relies on are still there
    assert "rss_peak_mb" in h and "rss_current_mb" in h


def test_health_reports_an_empty_guard_before_any_scan():
    assert vd._deep_guard_snapshot() == {} or vd._deep_guard_snapshot().get("last") is None


# ── session health check: the verdict is what the guard did ──────────────

def _guard(last, rss, **counts):
    c = {"normal": 0, "trim": 0, "skip": 0, "survival": 0, **counts}
    return {"counts": c, "scans": sum(c.values()),
            "consecutive_not_normal": counts.get("_run", 0) or (0 if last == "normal" else c[last]),
            "last": {"decision": last, "rss_mb": rss, "threshold_mb": None, "at": 1.0}}


def test_the_2026_11_17_reading_alarms_as_a_measured_skip():
    f = hc.check_daemon_memory({
        "alive": True, "rss_mb": 776.7, "rss_current_mb": 642.8,
        "deep_guard": _guard("skip", 640, normal=41, skip=37),
        "memory": {"python_blocks": 2400000, "at_start": {"python_blocks": 300000},
                   "malloc": {"in_use_mb": 410.0, "free_retained_mb": 150.0}}})
    assert f["severity"] == hc.ALARM
    d = f["detail"]
    assert "SKIPPED" in d and "37 scan(s) in a row" in d
    assert "normal=41" in d and "skip=37" in d
    assert "quick_score alone" in d
    assert "C heap in use 410.0MB, freed but retained 150.0MB" in d
    assert "python blocks 2400000 (start 300000)" in d


def test_a_skipping_guard_alarms_even_when_the_idle_reading_is_low():
    """The memory figure is read between scans; the guard reads it inside
    one. A low idle figure must not hide a skip."""
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 700, "rss_current_mb": 380,
                                "deep_guard": _guard("skip", 560, skip=3)})
    assert f["severity"] == hc.ALARM


def test_survival_mode_counts_as_skipped():
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 800, "rss_current_mb": 720,
                                "deep_guard": _guard("survival", 720, survival=2)})
    assert f["severity"] == hc.ALARM


def test_a_normal_guard_is_ok_even_when_the_idle_reading_is_past_the_line():
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 700, "rss_current_mb": 560,
                                "deep_guard": _guard("normal", 380, normal=12)})
    assert f["severity"] == hc.OK


def test_a_trimming_guard_warns():
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 500, "rss_current_mb": 430,
                                "deep_guard": _guard("trim", 430, normal=5, trim=2)})
    assert f["severity"] == hc.WARN and "trimmed" in f["detail"]


def test_earlier_skips_in_this_process_warn_after_the_guard_recovers():
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 700, "rss_current_mb": 300,
                                "deep_guard": _guard("normal", 300, normal=5, skip=4)})
    assert f["severity"] == hc.WARN and "4 earlier scan(s)" in f["detail"]


def test_a_daemon_with_no_guard_report_is_judged_on_memory_as_before():
    for guard in (None, {}, {"counts": {}, "last": None}):
        f = hc.check_daemon_memory({"alive": True, "rss_mb": 700, "rss_current_mb": 560,
                                    "deep_guard": guard})
        assert f["severity"] == hc.ALARM and "recurrence" in f["detail"]
