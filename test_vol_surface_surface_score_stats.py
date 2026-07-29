"""
KNOWN BROKEN #18 continuation (2026-07-28): live TIER2-ERROR occurrences on
2026-07-28 (13:36:33Z, 13:42:33Z) show `step6_trade_loop_and_options` (the
options scanner, options_scanner.get_options_trades()) as the preamble's
single slowest phase at 233-240s of a ~296-300s total daemon-timeout budget
— RECURRING despite v1.0.503's "ROOT CAUSE FOUND + FIXED" claim for this
exact phase (which removed dead Setup 4/5 fetches and predicted a ~140s
post-fix cost for ~421 candidates at the shared alpaca_throttle bucket's
3 req/sec).

ROOT CAUSE (this session): options_scanner.py's Setup 7 ("VRP-driven iron
condor", REAL-ALPHA-TUNE 2026-04-22) calls vol_surface.get_surface_score()
unconditionally for every high_iv/anchor candidate. That function's own
build_surface() does an UNTHROTTLED spot-price fetch, an UNTHROTTLED
options-chain fetch (vol_surface.py's own _fetch_options_chain, up to 10
pages), and an UNTHROTTLED 90-day bars fetch — none of those three network
calls go through alpaca_throttle, so none of them were ever counted by any
of this item's five prior sessions' "candidates/3s" cost model. This test
suite covers the new scan-scoped accumulator (reset/record/get) that makes
that cost visible, mirroring csp_universe.py's layer1_stats/layer2_prefetch
convention — pure diagnostics, no behavior change, real network calls not
exercised here (same tradeoff test_csp_universe_layer1_stats.py's own suite
already makes).
"""
import time

import vol_surface
from vol_surface import (
    reset_surface_score_scan_stats,
    get_surface_score_scan_stats,
    _record_surface_score_call,
    surface_score_scan_budget_exceeded,
    record_surface_score_budget_skip,
)


def test_scan_stats_empty_before_any_reset(monkeypatch):
    monkeypatch.setattr(vol_surface, "_SURFACE_SCORE_SCAN_STATS", {})
    assert get_surface_score_scan_stats() == {}


def test_reset_zeroes_the_accumulator_and_stamps_checked_at():
    reset_surface_score_scan_stats()
    stats = get_surface_score_scan_stats()
    assert stats["calls"] == 0
    assert stats["cache_misses"] == 0
    assert stats["cumulative_elapsed_sec"] == 0.0
    assert stats["skipped_budget"] == 0
    assert "checked_at" in stats


def test_record_accumulates_calls_and_elapsed_across_multiple_invocations():
    reset_surface_score_scan_stats()
    _record_surface_score_call(cache_hit=False, elapsed_sec=2.5)
    _record_surface_score_call(cache_hit=False, elapsed_sec=1.5)
    _record_surface_score_call(cache_hit=True, elapsed_sec=0.01)

    stats = get_surface_score_scan_stats()
    assert stats["calls"] == 3
    assert stats["cache_misses"] == 2  # only the two cache_hit=False calls
    assert stats["cumulative_elapsed_sec"] == 4.01


def test_record_before_any_reset_is_a_noop_not_a_crash(monkeypatch):
    monkeypatch.setattr(vol_surface, "_SURFACE_SCORE_SCAN_STATS", {})
    _record_surface_score_call(cache_hit=False, elapsed_sec=5.0)  # must not raise
    assert get_surface_score_scan_stats() == {}


def test_reset_clears_a_prior_scan_cycle_stats():
    reset_surface_score_scan_stats()
    _record_surface_score_call(cache_hit=False, elapsed_sec=10.0)
    assert get_surface_score_scan_stats()["calls"] == 1

    reset_surface_score_scan_stats()
    stats = get_surface_score_scan_stats()
    assert stats["calls"] == 0
    assert stats["cumulative_elapsed_sec"] == 0.0


def test_get_surface_score_records_a_call_wrapping_the_real_function(monkeypatch):
    """get_surface_score() itself must feed the accumulator regardless of
    whether the inner computation succeeds or raises — the whole point is
    to measure cost even for tickers that end up erroring out (e.g. no
    options data), same as the live production path would encounter."""
    reset_surface_score_scan_stats()

    def _fake_inner(ticker):
        time.sleep(0.01)
        return {"ticker": ticker, "surface_score": 70.0}

    monkeypatch.setattr(vol_surface, "_get_surface_score_inner", _fake_inner)
    monkeypatch.setattr(vol_surface, "_surface_cache", {})

    result = vol_surface.get_surface_score("AAPL")

    assert result == {"ticker": "AAPL", "surface_score": 70.0}
    stats = get_surface_score_scan_stats()
    assert stats["calls"] == 1
    assert stats["cache_misses"] == 1  # _surface_cache was empty -> cache miss
    assert stats["cumulative_elapsed_sec"] > 0.0


def test_get_surface_score_records_cache_hit_when_surface_cache_is_fresh(monkeypatch):
    reset_surface_score_scan_stats()
    monkeypatch.setattr(vol_surface, "_get_surface_score_inner", lambda t: {"ticker": t})
    monkeypatch.setattr(vol_surface, "_surface_cache", {
        "AAPL": {"timestamp": time.time(), "surface": {}}
    })

    vol_surface.get_surface_score("AAPL")

    stats = get_surface_score_scan_stats()
    assert stats["calls"] == 1
    assert stats["cache_misses"] == 0  # fresh cache entry -> counted as a hit


def test_get_surface_score_still_records_even_when_inner_raises(monkeypatch):
    reset_surface_score_scan_stats()
    monkeypatch.setattr(vol_surface, "_surface_cache", {})

    def _raise(ticker):
        raise RuntimeError("network down")
    monkeypatch.setattr(vol_surface, "_get_surface_score_inner", _raise)

    try:
        vol_surface.get_surface_score("AAPL")
        assert False, "expected RuntimeError to propagate"
    except RuntimeError:
        pass

    stats = get_surface_score_scan_stats()
    assert stats["calls"] == 1  # the finally-block still recorded the call


# KNOWN BROKEN #18 continuation (2026-07-29): two straight days of live
# TIER2-ERROR occurrences show cumulative_elapsed_sec (312-365s/scan) alone
# exceeding the 300s daemon RPC timeout — this accumulator answered "how
# much does it cost" but did nothing to CAP it. These tests cover the new
# scan-wide budget guard options_scanner.py's Setup 7 block now checks
# before every get_surface_score() call.

def test_budget_not_exceeded_below_threshold():
    reset_surface_score_scan_stats()
    _record_surface_score_call(cache_hit=False, elapsed_sec=10.0)
    assert surface_score_scan_budget_exceeded(60.0) is False


def test_budget_exceeded_once_cumulative_elapsed_reaches_threshold():
    reset_surface_score_scan_stats()
    _record_surface_score_call(cache_hit=False, elapsed_sec=45.0)
    _record_surface_score_call(cache_hit=False, elapsed_sec=15.0)  # cumulative = 60.0
    assert surface_score_scan_budget_exceeded(60.0) is True


def test_budget_check_never_trips_before_any_reset(monkeypatch):
    """Same safe-no-signal-yet default as get_surface_score_scan_stats()
    itself — a process that hasn't reset this cycle must not spuriously
    block every Setup 7 call."""
    monkeypatch.setattr(vol_surface, "_SURFACE_SCORE_SCAN_STATS", {})
    assert surface_score_scan_budget_exceeded(0.0) is False


def test_record_budget_skip_increments_the_counter():
    reset_surface_score_scan_stats()
    record_surface_score_budget_skip()
    record_surface_score_budget_skip()
    stats = get_surface_score_scan_stats()
    assert stats["skipped_budget"] == 2
    assert stats["calls"] == 0  # a skip is not a call — must not double-count


def test_record_budget_skip_before_any_reset_is_a_noop_not_a_crash(monkeypatch):
    monkeypatch.setattr(vol_surface, "_SURFACE_SCORE_SCAN_STATS", {})
    record_surface_score_budget_skip()  # must not raise
    assert get_surface_score_scan_stats() == {}
