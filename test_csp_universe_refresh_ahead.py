"""
(am1) part (i), 2026-11-25 — KNOWN BROKEN #60's structural half: the CSP
universe's recompute leaves the scan's clock.

THE BREAK THESE TESTS WOULD HAVE CAUGHT: csp_universe recomputed its two
layers inside whichever scan first read them past their 15 minutes. Measured
on production 2026-11-24/25 (/api/diag/timings?history=1): the tier engine
took 31-75 s on those scans and under 10 s on the others, about one scan in
three; on 2026-11-24, with the options step ending near 250 s, three such
scans ran past the daemon's 300 s and returned nothing.

The regression test is `test_a_scan_is_answered_from_the_cache_while_the_
recompute_runs_elsewhere`: the recompute is held open on an Event and the
scan's read must return before it is released.
"""
import json
import threading
import time

import pytest

import csp_universe
import options_scanner
from csp_universe import refresh_decision


SNAP = {
    "AAA": {"dailyBar": {"c": 50.0, "v": 5_000_000}},
    "BBB": {"dailyBar": {"c": 80.0, "v": 3_000_000}},
    "CCC": {"dailyBar": {"c": 20.0, "v": 9_000_000}},
}


def _write_caches(tmp_path, age_layer1, age_layer2, scores=None):
    now = time.time()
    with open(tmp_path / "universe.json", "w") as f:
        json.dump({"candidates": [["OLD", 30.0, 2_000_000, 60_000_000.0]], "count": 1,
                   "_cached_at": now - age_layer1}, f)
    if scores is None:
        scores = [{"ticker": "OLD", "score": 61.0, "price": 30.0}]
    with open(tmp_path / "scores.json", "w") as f:
        json.dump({"scores": scores, "count": len(scores), "_cached_at": now - age_layer2}, f)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Both cache files in tmp_path, no network, a counted and gateable
    recompute. `gate` is set: a test that wants the recompute held clears it."""
    monkeypatch.setattr(csp_universe, "UNIVERSE_CACHE_PATH", str(tmp_path / "universe.json"))
    monkeypatch.setattr(csp_universe, "SCORES_CACHE_PATH", str(tmp_path / "scores.json"))
    monkeypatch.setattr(csp_universe, "_REFRESH_LOCK", threading.Lock())
    monkeypatch.setattr(csp_universe, "_BACKGROUND_REFRESH", {})
    monkeypatch.setattr(csp_universe, "_LAST_LAYER1_STATS", {})
    monkeypatch.setattr(csp_universe, "_LAST_LAYER2_PREFETCH", {})
    monkeypatch.setattr(csp_universe, "CSP_ANCHOR_TICKERS", [])
    monkeypatch.setattr(csp_universe, "_long_lived_process", lambda: True)
    monkeypatch.delenv("VOLTRADE_CSP_BACKGROUND_REFRESH", raising=False)
    monkeypatch.setattr(csp_universe, "_fetch_account_equity", lambda: 100_000.0)
    monkeypatch.setattr(csp_universe, "leveraged_symbols", lambda: set())
    monkeypatch.setattr(csp_universe, "_score_iv_rank", lambda t, c: 50.0)
    monkeypatch.setattr(csp_universe, "_score_vrp", lambda t, c: 50.0)
    monkeypatch.setattr(csp_universe, "_score_historical", lambda t: 50.0)
    monkeypatch.setattr(options_scanner, "_fetch_earnings_calendar_with_status",
                        lambda days_ahead=30: ({}, True))

    state = {"gate": threading.Event(), "threads": [], "snap": SNAP, "tmp": tmp_path}
    state["gate"].set()

    def _fetch():
        state["threads"].append(threading.current_thread().name)
        assert state["gate"].wait(10), "test never released the recompute"
        return state["snap"]
    monkeypatch.setattr(csp_universe, "_fetch_snap_data_for_universe", _fetch)
    yield state
    state["gate"].set()
    _wait_idle()


def _wait_idle(timeout=10.0):
    """Until no refresh holds the lock."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if csp_universe._REFRESH_LOCK.acquire(blocking=False):
            csp_universe._REFRESH_LOCK.release()
            return
        time.sleep(0.01)
    raise AssertionError("background refresh never finished")


# ── the decision, pure ───────────────────────────────────────────────────────

@pytest.mark.parametrize("age1,age2,expected", [
    (None, None, "compute"),              # first scan after a boot with no files
    (300.0, None, "compute"),
    (None, 300.0, "compute"),
    (0.0, 0.0, "fresh"),
    (479.9, 420.0, "fresh"),
    (480.0, 420.0, "serve_and_refresh"),  # Layer 1 is the older one and decides
    (420.0, 480.0, "serve_and_refresh"),
    (899.0, 840.0, "serve_and_refresh"),  # inside the old 15 minutes: refreshed AHEAD
    (1000.0, 940.0, "serve_and_refresh"),  # past them, a refresh pending: still served
    (1199.9, 1100.0, "serve_and_refresh"),
    (1200.0, 1100.0, "compute"),          # too old to serve: the scan computes
    (90_000.0, 90_000.0, "compute"),      # yesterday's file is never served
])
def test_refresh_decision(age1, age2, expected):
    assert refresh_decision(age1, age2, 480.0, 1200.0) == expected


def test_config_values_keep_the_refresh_inside_the_cache_lifetime():
    """The refresh must start before the caches expire, and a ranking is never
    served older than twice its lifetime."""
    from system_config import BASE_CONFIG
    ahead = BASE_CONFIG["CSP_SCORES_REFRESH_AHEAD_SEC"]
    stale_max = BASE_CONFIG["CSP_SCORES_SERVE_STALE_MAX_SEC"]
    assert BASE_CONFIG["CSP_SCORES_BACKGROUND_REFRESH"] is True
    assert 0 < ahead < csp_universe.LAYER1_CACHE_TTL
    assert csp_universe.LAYER2_CACHE_TTL <= stale_max <= 2 * csp_universe.LAYER2_CACHE_TTL
    assert csp_universe._refresh_config() == (True, float(ahead), float(stale_max))


# ── the regression ───────────────────────────────────────────────────────────

def test_a_scan_is_answered_from_the_cache_while_the_recompute_runs_elsewhere(env):
    _write_caches(env["tmp"], age_layer1=960.0, age_layer2=900.0)   # both past 15 min
    env["gate"].clear()                                             # recompute held open

    t0 = time.time()
    tickers = csp_universe.get_top_csp_candidates(n=10)
    took = time.time() - t0

    # The scan got the cached ranking and did not wait for the recompute.
    assert tickers == ["OLD"]
    assert took < 2.0
    assert csp_universe._REFRESH_LOCK.locked()                      # ...which is running
    # What the scan's own reading says (the audit line and the timing record
    # classify a scan by it): served from cache, a refresh started.
    l2 = csp_universe.get_last_layer2_prefetch_stats()
    assert l2["cache_hit"] is True and l2["refresh"] == "started"
    assert 899.0 <= l2["served_age_sec"] <= 905.0
    assert csp_universe.get_last_layer1_stats()["cache_hit"] is True
    assert csp_universe.get_last_background_refresh()["status"] == "running"

    env["gate"].set()
    _wait_idle()

    # The recompute ran once, on its own thread, and rewrote both files.
    assert env["threads"] == ["csp-universe-refresh"]
    rec = csp_universe.get_last_background_refresh()
    assert rec["status"] == "ok" and rec["scored"] == 3
    assert rec["layer1"]["cache_hit"] is False and rec["layer1"]["universe_size"] == 3
    assert rec["layer2"]["cache_hit"] is False and rec["layer2"]["total"] == 3
    assert rec["finished_at"] >= rec["started_at"]
    for name in ("universe.json", "scores.json"):
        with open(env["tmp"] / name) as f:
            assert time.time() - json.load(f)["_cached_at"] < 10
    # The scan's reading was not overwritten by the background thread's.
    assert csp_universe.get_last_layer2_prefetch_stats()["cache_hit"] is True

    # The next scan reads the new ranking through the ordinary fresh path.
    assert sorted(csp_universe.get_top_csp_candidates(n=10)) == ["AAA", "BBB", "CCC"]
    assert env["threads"] == ["csp-universe-refresh"]              # nothing recomputed
    assert "refresh" not in csp_universe.get_last_layer2_prefetch_stats()


def test_refresh_starts_ahead_of_expiry(env):
    """Caches 8-15 min old were plain cache hits; the next scan then paid. Now
    the read at that age starts the refresh so the next scan reads new files."""
    _write_caches(env["tmp"], age_layer1=600.0, age_layer2=540.0)
    assert csp_universe.get_top_csp_candidates(n=10) == ["OLD"]
    _wait_idle()
    assert env["threads"] == ["csp-universe-refresh"]
    assert csp_universe.get_last_background_refresh()["status"] == "ok"


def test_young_caches_start_nothing(env):
    _write_caches(env["tmp"], age_layer1=300.0, age_layer2=240.0)
    assert csp_universe.get_top_csp_candidates(n=10) == ["OLD"]
    assert env["threads"] == []
    assert csp_universe.get_last_background_refresh() == {}


def test_one_refresh_at_a_time(env):
    _write_caches(env["tmp"], age_layer1=960.0, age_layer2=900.0)
    env["gate"].clear()
    assert csp_universe.get_top_csp_candidates(n=10) == ["OLD"]
    assert csp_universe.get_top_csp_candidates(n=10) == ["OLD"]     # a second scan meanwhile
    assert csp_universe.get_last_layer2_prefetch_stats()["refresh"] == "in_flight"
    env["gate"].set()
    _wait_idle()
    assert env["threads"] == ["csp-universe-refresh"]


def test_a_scan_that_must_compute_waits_for_a_running_refresh_and_reads_its_result(env):
    """No second computation beside a running one: both would draw on the same
    Alpaca bucket for the same answer."""
    _write_caches(env["tmp"], age_layer1=960.0, age_layer2=900.0)
    env["gate"].clear()
    assert csp_universe.get_top_csp_candidates(n=10) == ["OLD"]     # starts the refresh
    _write_caches(env["tmp"], age_layer1=5000.0, age_layer2=5000.0)  # now too old to serve
    out = {}
    scan = threading.Thread(target=lambda: out.update(t=csp_universe.get_top_csp_candidates(n=10)))
    scan.start()
    time.sleep(0.2)
    assert scan.is_alive()                                          # waiting on the lock
    env["gate"].set()
    scan.join(10)
    assert sorted(out["t"]) == ["AAA", "BBB", "CCC"]
    assert env["threads"] == ["csp-universe-refresh"]              # computed once


# ── where the old path must remain ───────────────────────────────────────────

def test_caches_too_old_to_serve_are_computed_on_the_scan(env):
    _write_caches(env["tmp"], age_layer1=1300.0, age_layer2=1240.0)
    assert sorted(csp_universe.get_top_csp_candidates(n=10)) == ["AAA", "BBB", "CCC"]
    assert env["threads"] == ["MainThread"]
    assert csp_universe.get_last_layer2_prefetch_stats()["cache_hit"] is False
    assert csp_universe.get_last_background_refresh() == {}


def test_no_cache_is_computed_on_the_scan(env):
    assert sorted(csp_universe.get_top_csp_candidates(n=10)) == ["AAA", "BBB", "CCC"]
    assert env["threads"] == ["MainThread"]


def test_a_short_lived_process_never_starts_a_thread(env, monkeypatch):
    """The subprocess fallback exits when its scan returns: a background
    thread there would die mid-refresh, every time."""
    monkeypatch.setattr(csp_universe, "_long_lived_process", lambda: False)
    _write_caches(env["tmp"], age_layer1=600.0, age_layer2=540.0)
    assert csp_universe.get_top_csp_candidates(n=10) == ["OLD"]     # plain cache hit
    _write_caches(env["tmp"], age_layer1=960.0, age_layer2=900.0)
    assert sorted(csp_universe.get_top_csp_candidates(n=10)) == ["AAA", "BBB", "CCC"]
    assert env["threads"] == ["MainThread"]


def test_long_lived_process_is_the_daemon_only(monkeypatch):
    import sys
    import types
    fake = types.ModuleType("__main__")
    fake.__file__ = "/app/voltrade_daemon.py"
    monkeypatch.setitem(sys.modules, "__main__", fake)
    assert csp_universe._long_lived_process() is True
    fake.__file__ = "/app/bot_engine.py"
    assert csp_universe._long_lived_process() is False
    del fake.__file__                                               # python3 -c "..."
    assert csp_universe._long_lived_process() is False


@pytest.mark.parametrize("value", ["0", "off", "false", "no", ""])
def test_env_off_switch(env, monkeypatch, value):
    monkeypatch.setenv("VOLTRADE_CSP_BACKGROUND_REFRESH", value)
    _write_caches(env["tmp"], age_layer1=960.0, age_layer2=900.0)
    assert sorted(csp_universe.get_top_csp_candidates(n=10)) == ["AAA", "BBB", "CCC"]
    assert env["threads"] == ["MainThread"]


def test_config_off_switch(env, monkeypatch):
    import system_config
    monkeypatch.setitem(system_config.BASE_CONFIG, "CSP_SCORES_BACKGROUND_REFRESH", False)
    _write_caches(env["tmp"], age_layer1=960.0, age_layer2=900.0)
    assert sorted(csp_universe.get_top_csp_candidates(n=10)) == ["AAA", "BBB", "CCC"]
    assert env["threads"] == ["MainThread"]


def test_an_empty_ranking_keeps_its_fifteen_minutes(env, monkeypatch):
    """KB #45: a failed earnings lookup writes an EMPTY ranking (fail closed).
    It is not served past its lifetime and it is not refreshed early: the
    behaviour before this change, exactly."""
    _write_caches(env["tmp"], age_layer1=600.0, age_layer2=540.0, scores=[])
    monkeypatch.setattr(csp_universe, "CSP_ANCHOR_TICKERS", ["SPY"])
    assert csp_universe.get_top_csp_candidates(n=10) == ["SPY"]
    assert env["threads"] == []                                     # cache hit, no refresh
    _write_caches(env["tmp"], age_layer1=960.0, age_layer2=900.0, scores=[])
    got = csp_universe.get_top_csp_candidates(n=10)
    assert env["threads"] == ["MainThread"]                         # recomputed on the scan
    assert sorted(got) == ["AAA", "BBB", "CCC", "SPY"]


# ── a refresh that fails ─────────────────────────────────────────────────────

def test_a_failed_snapshot_fetch_does_not_replace_the_ranking(env):
    """Layer 1 answers a failed fetch with bare anchors and writes nothing.
    Scoring those would stamp an anchors-only ranking as fresh."""
    _write_caches(env["tmp"], age_layer1=960.0, age_layer2=900.0)
    env["snap"] = {}
    assert csp_universe.get_top_csp_candidates(n=10) == ["OLD"]
    _wait_idle()
    assert csp_universe.get_last_background_refresh()["status"] == "layer1_not_refreshed"
    with open(env["tmp"] / "scores.json") as f:
        kept = json.load(f)
    assert [s["ticker"] for s in kept["scores"]] == ["OLD"]
    assert time.time() - kept["_cached_at"] > 890                   # not re-stamped
    # The lock is free: the next scan starts another attempt.
    env["snap"] = SNAP
    assert csp_universe.get_top_csp_candidates(n=10) == ["OLD"]
    _wait_idle()
    assert csp_universe.get_last_background_refresh()["status"] == "ok"


def test_an_exception_in_the_refresh_releases_the_lock(env, monkeypatch):
    _write_caches(env["tmp"], age_layer1=960.0, age_layer2=900.0)

    def _boom(*a, **k):
        raise RuntimeError("scoring blew up")
    monkeypatch.setattr(csp_universe, "_layer2_score", _boom)
    assert csp_universe.get_top_csp_candidates(n=10) == ["OLD"]
    _wait_idle()
    rec = csp_universe.get_last_background_refresh()
    assert rec["status"] == "error" and "scoring blew up" in rec["error"]
    assert not csp_universe._REFRESH_LOCK.locked()


# ── wiring ───────────────────────────────────────────────────────────────────

def test_force_skips_a_fresh_cache(env):
    _write_caches(env["tmp"], age_layer1=10.0, age_layer2=10.0)
    assert csp_universe._layer1_hard_gates() == [("OLD", 30.0, 2_000_000, 60_000_000.0)]
    assert len(csp_universe._layer1_hard_gates(force=True)) == 3
    _write_caches(env["tmp"], age_layer1=10.0, age_layer2=10.0)
    cands = [("AAA", 50.0, 5_000_000, 250_000_000.0)]
    assert csp_universe._layer2_score(cands)[0]["ticker"] == "OLD"
    assert csp_universe._layer2_score(cands, force=True)[0]["ticker"] == "AAA"


def test_the_last_refresh_reaches_the_timings_and_the_snapshot(env, monkeypatch):
    from tiered_strategy import TierContext, TieredStrategy
    monkeypatch.setattr("tiered_strategy._get_t1_universe", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(csp_universe, "_BACKGROUND_REFRESH",
                        {"status": "ok", "elapsed_sec": 41.2, "started_at": time.time() - 60})
    ctx = TierContext(equity=100_000.0, peak_equity=100_000.0, buying_power=100_000.0,
                      positions=[], vxx_ratio=1.0, spy_vs_ma50=1.0)
    timings = TieredStrategy().run_tiers(ctx)["tier_timings"]
    assert timings["csp_background_refresh"]["status"] == "ok"
    assert timings["csp_background_refresh"]["elapsed_sec"] == 41.2
    assert csp_universe.get_universe_snapshot()["background_refresh"]["status"] == "ok"
