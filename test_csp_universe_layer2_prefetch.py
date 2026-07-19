"""
KNOWN BROKEN #18 continuation (2026-07-19): direct instrument for csp_universe.py's
Layer 2 scoring prefetch.

Live evidence this session: v1.0.416 (this morning's dedup fix for deep_score's
credit_spread + macro_data's SPY-bars double-fetch) did NOT stop the TIER2-ERROR
"Daemon timeout" storm — it recurred at essentially the same ~8-10min cadence
hours after deploy, with active_dispatches staying at 2 every time (refuting the
daemon-concurrency-overload theory). A live /api/diag/timings pull during the
storm showed tier1_sec=42.94s (vs. a ~1s baseline) — suspiciously close to
_layer2_score()'s own PREFETCH_BUDGET_S=45.0 hard cap. The math: Layer 2's
deep_score_limit=150 candidates x 2 network calls each (_score_iv_rank +
_score_vrp) = up to 300 calls competing for the shared alpaca_throttle bucket
(180 req/min = 3 req/s, alpaca_rate_limiter.py, FROZEN) inside a 45s wall
budget — needing ~6.7 req/s, more than double the throttle's sustainable rate,
structurally independent of any other traffic. This is a NEW, untested-until-now
theory (a different mechanism than the two already-refuted event-loop-lag
theories or the shadowFleet O(n^2) fix that DID resolve mechanism #3) — per
this item's own discipline, it gets an instrument before a guessed fix, not a
blind patch.

These tests cover build_prefetch_stats() (the pure function) and _layer2_score's
wiring of it, NOT the real ThreadPoolExecutor timing (spinning up a genuine 45s
budget-exceeded case would make this suite too slow for CI — the same tradeoff
eventLoopLag.test.ts and dbWriteTiming.test.ts made for their own instruments).
"""
import json
import time

import pytest

import csp_universe
from csp_universe import build_prefetch_stats, get_last_layer2_prefetch_stats


def test_build_prefetch_stats_cache_hit_never_flags_budget_exceeded():
    stats = build_prefetch_stats(cache_hit=True)
    assert stats == {
        "cache_hit": True,
        "completed": 0,
        "total": 0,
        "elapsed_sec": 0.0,
        "budget_exceeded": False,
    }


def test_build_prefetch_stats_fresh_compute_all_completed():
    stats = build_prefetch_stats(cache_hit=False, completed=150, total=150, elapsed_sec=12.3)
    assert stats["budget_exceeded"] is False
    assert stats["completed"] == stats["total"] == 150
    assert stats["elapsed_sec"] == 12.3


def test_build_prefetch_stats_fresh_compute_budget_exceeded():
    """The exact live-observed shape: completed < total after PREFETCH_BUDGET_S
    wall-clock ran out — this is the reading that would confirm the throttle-
    overcommit theory the next time TIER2-ERROR fires."""
    stats = build_prefetch_stats(cache_hit=False, completed=62, total=150, elapsed_sec=45.02)
    assert stats["budget_exceeded"] is True
    assert stats["completed"] < stats["total"]


def test_build_prefetch_stats_no_candidates_is_not_budget_exceeded():
    """0/0 (Layer 1 returned nothing) must not read as a budget blowup."""
    stats = build_prefetch_stats(cache_hit=False, completed=0, total=0, elapsed_sec=0.0)
    assert stats["budget_exceeded"] is False


def test_get_last_layer2_prefetch_stats_empty_before_any_call(monkeypatch):
    monkeypatch.setattr(csp_universe, "_LAST_LAYER2_PREFETCH", {})
    assert get_last_layer2_prefetch_stats() == {}


def test_layer2_score_cache_hit_records_cache_hit_true(tmp_path, monkeypatch):
    scores_path = tmp_path / "scores_cache.json"
    scores_path.write_text(json.dumps({
        "scores": [{"ticker": "AAPL", "score": 80.0}],
        "_cached_at": time.time(),
    }))
    monkeypatch.setattr(csp_universe, "SCORES_CACHE_PATH", str(scores_path))

    result = csp_universe._layer2_score([("AAPL", 200.0, 1000, 200000.0)])

    assert result == [{"ticker": "AAPL", "score": 80.0}]
    stats = get_last_layer2_prefetch_stats()
    assert stats["cache_hit"] is True
    assert stats["budget_exceeded"] is False
    assert "checked_at" in stats


def test_layer2_score_stale_cache_falls_through_to_fresh_compute_all_completed(tmp_path, monkeypatch):
    """Cache miss (stale _cached_at) with a small, fast (mocked) candidate
    list must record cache_hit=False and completed==total — the "healthy"
    reading. A production reading of completed < total on the SAME shape
    of entry is what would confirm the throttle-overcommit theory."""
    scores_path = tmp_path / "scores_cache.json"
    scores_path.write_text(json.dumps({
        "scores": [],
        "_cached_at": time.time() - csp_universe.LAYER2_CACHE_TTL - 1,
    }))
    monkeypatch.setattr(csp_universe, "SCORES_CACHE_PATH", str(scores_path))
    monkeypatch.setattr(csp_universe, "_score_iv_rank",
                         lambda ticker, cache: cache.setdefault(ticker, 65.0) and 70.0)
    monkeypatch.setattr(csp_universe, "_score_vrp",
                         lambda ticker, cache: cache.setdefault(ticker, {"vrp": {"vrp_20d": 0.05}}) and 60.0)
    monkeypatch.setattr("options_scanner._fetch_earnings_calendar", lambda days_ahead=30: {})

    candidates = [
        ("AAPL", 200.0, 1_000_000, 200_000_000.0),
        ("MSFT", 300.0, 900_000, 270_000_000.0),
    ]
    result = csp_universe._layer2_score(candidates)

    assert len(result) == 2
    stats = get_last_layer2_prefetch_stats()
    assert stats["cache_hit"] is False
    assert stats["completed"] == stats["total"] == 2
    assert stats["budget_exceeded"] is False


def test_layer2_score_no_candidates_records_empty_stats_not_budget_exceeded(tmp_path, monkeypatch):
    scores_path = tmp_path / "scores_cache.json"
    monkeypatch.setattr(csp_universe, "SCORES_CACHE_PATH", str(scores_path))  # doesn't exist

    result = csp_universe._layer2_score([])

    assert result == []
    stats = get_last_layer2_prefetch_stats()
    assert stats["cache_hit"] is False
    assert stats["total"] == 0
    assert stats["budget_exceeded"] is False


def test_tier_engine_surfaces_csp_layer2_prefetch_in_tier_timings(monkeypatch):
    """Wiring pin: TieredStrategy.run_tiers()'s tier_timings must carry the
    csp_universe reading forward — this is how it reaches /api/diag/timings
    without any new endpoint (bot_engine.py already surfaces tier_timings
    verbatim from tier_result). Pure visibility: must not change actions/
    tier_stats shape. _get_t1_universe is patched (same convention as
    test_tiered_strategy.py) so tier1_csp_core never reaches the real,
    network-hitting csp_universe path — this test is about the wiring,
    not re-proving _layer2_score's own behavior."""
    from tiered_strategy import TierContext, TieredStrategy

    monkeypatch.setattr("tiered_strategy._get_t1_universe", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(csp_universe, "_LAST_LAYER2_PREFETCH",
                         {"cache_hit": True, "completed": 0, "total": 0,
                          "elapsed_sec": 0.0, "budget_exceeded": False,
                          "checked_at": time.time()})

    ctx = TierContext(
        equity=100_000.0, peak_equity=100_000.0, buying_power=100_000.0,
        positions=[], vxx_ratio=1.0, spy_vs_ma50=1.0,
    )
    result = TieredStrategy().run_tiers(ctx)

    assert "csp_layer2_prefetch" in result["tier_timings"]
    assert result["tier_timings"]["csp_layer2_prefetch"]["cache_hit"] is True
    # unchanged shape otherwise
    assert "actions" in result and "tier_stats" in result
