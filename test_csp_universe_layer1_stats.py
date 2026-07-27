"""
KNOWN BROKEN #18 continuation (2026-07-27): direct instrument for
csp_universe.py's Layer 1 hard-gate stage.

Live evidence this session: every tier_engine_start-stuck TIER2-ERROR
occurrence since v1.0.503's on_phase instrumentation shipped (11/11 checked,
2026-07-26 16:33Z through 2026-07-27 13:44Z, including one from earlier
today) read layer2_prefetch cache_hit=true (elapsed=0s) — Layer 2 was never
the explanation for these hangs. tier1_csp_core's OTHER expensive call,
_get_t1_universe() -> get_top_csp_candidates() -> _layer1_hard_gates(), has
its own independent 15-min cache (UNIVERSE_CACHE_PATH, separate from Layer
2's SCORES_CACHE_PATH) that was never instrumented at all — its cache-miss
path does a full-universe live snapshot fetch (_fetch_snap_data_for_universe,
batched Alpaca calls) plus an account-equity fetch, invisible cost that could
easily explain a tier1-localized hang while Layer 2 reads as a fast cache hit.

These tests cover build_layer1_stats() (the pure function) and
_layer1_hard_gates()'s wiring of it across all four return paths (cache hit,
snap_data fetch failure, empty snap_data, full live compute) — NOT the real
network fetch, same tradeoff test_csp_universe_layer2_prefetch.py's own
suite already makes.
"""
import json
import time

import pytest

import csp_universe
from csp_universe import build_layer1_stats, get_last_layer1_stats


def test_build_layer1_stats_cache_hit_shape():
    stats = build_layer1_stats(cache_hit=True, universe_size=412, elapsed_sec=0.0)
    assert stats == {
        "cache_hit": True,
        "universe_size": 412,
        "elapsed_sec": 0.0,
    }


def test_build_layer1_stats_live_compute_shape():
    stats = build_layer1_stats(cache_hit=False, universe_size=380, elapsed_sec=18.4)
    assert stats["cache_hit"] is False
    assert stats["universe_size"] == 380
    assert stats["elapsed_sec"] == 18.4


def test_get_last_layer1_stats_empty_before_any_call(monkeypatch):
    monkeypatch.setattr(csp_universe, "_LAST_LAYER1_STATS", {})
    assert get_last_layer1_stats() == {}


def test_layer1_hard_gates_cache_hit_records_cache_hit_true_and_size(tmp_path, monkeypatch):
    universe_path = tmp_path / "universe_cache.json"
    universe_path.write_text(json.dumps({
        "candidates": [["AAPL", 200.0, 1_000_000, 200_000_000.0],
                        ["MSFT", 300.0, 900_000, 270_000_000.0]],
        "count": 2,
        "_cached_at": time.time(),
    }))
    monkeypatch.setattr(csp_universe, "UNIVERSE_CACHE_PATH", str(universe_path))

    result = csp_universe._layer1_hard_gates()

    assert result == [("AAPL", 200.0, 1_000_000, 200_000_000.0),
                       ("MSFT", 300.0, 900_000, 270_000_000.0)]
    stats = get_last_layer1_stats()
    assert stats["cache_hit"] is True
    assert stats["universe_size"] == 2
    assert stats["elapsed_sec"] < 1.0  # cache read, no network
    assert "checked_at" in stats


def test_layer1_hard_gates_snap_data_fetch_failure_records_cache_miss_zero_size(tmp_path, monkeypatch):
    universe_path = tmp_path / "universe_cache.json"  # doesn't exist -> cache miss
    monkeypatch.setattr(csp_universe, "UNIVERSE_CACHE_PATH", str(universe_path))

    def _raise():
        raise RuntimeError("network down")
    monkeypatch.setattr(csp_universe, "_fetch_snap_data_for_universe", _raise)

    result = csp_universe._layer1_hard_gates()

    assert result == []
    stats = get_last_layer1_stats()
    assert stats["cache_hit"] is False
    assert stats["universe_size"] == 0


def test_layer1_hard_gates_empty_snap_data_records_anchor_fallback_size(tmp_path, monkeypatch):
    universe_path = tmp_path / "universe_cache.json"  # doesn't exist -> cache miss
    monkeypatch.setattr(csp_universe, "UNIVERSE_CACHE_PATH", str(universe_path))
    monkeypatch.setattr(csp_universe, "_fetch_snap_data_for_universe", lambda: {})

    result = csp_universe._layer1_hard_gates()

    assert len(result) == len(csp_universe.CSP_ANCHOR_TICKERS)
    stats = get_last_layer1_stats()
    assert stats["cache_hit"] is False
    assert stats["universe_size"] == len(csp_universe.CSP_ANCHOR_TICKERS)


def test_layer1_hard_gates_live_compute_records_cache_miss_and_real_size(tmp_path, monkeypatch):
    universe_path = tmp_path / "universe_cache.json"  # doesn't exist -> cache miss
    monkeypatch.setattr(csp_universe, "UNIVERSE_CACHE_PATH", str(universe_path))
    monkeypatch.setattr(csp_universe, "_fetch_account_equity", lambda: 100_000.0)

    snap_data = {
        "AAPL": {"dailyBar": {"c": 200.0, "v": 5_000_000}},
        "MSFT": {"dailyBar": {"c": 300.0, "v": 4_000_000}},
    }
    result = csp_universe._layer1_hard_gates(snap_data=snap_data)

    assert len(result) >= 2
    stats = get_last_layer1_stats()
    assert stats["cache_hit"] is False
    assert stats["universe_size"] == len(result)
    assert "checked_at" in stats
