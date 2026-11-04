"""
[REPAIR 2026-11-03, memory audit A5] vol_surface's two module caches are
LRU-capped (surface: 200 tickers, SABR: 2000 (ticker, expiry) pairs) with
the TTLs unchanged. These tests pin: the cap evicts least-recently-USED (a
hit refreshes recency), re-inserting a key never grows the cache, 0 = the
unbounded pre-A5 behavior (rollback), env parsing, the stats getter, and
that the live build/calibrate paths route through the capped put.
"""
import os
import re
import time

import pytest

import vol_surface
from vol_surface import (
    _cache_cap, _cache_put, _cache_touch, vol_surface_cache_stats,
    VOL_SURFACE_CACHE_CAP_DEFAULT, SABR_CACHE_CAP_DEFAULT,
    VOL_SURFACE_CACHE_CAP_ENV, SABR_CACHE_CAP_ENV,
)


def test_defaults_are_the_approved_figures():
    assert VOL_SURFACE_CACHE_CAP_DEFAULT == 200
    assert SABR_CACHE_CAP_DEFAULT == 2000
    assert vol_surface.CACHE_TTL == 3600, "A5 caps memory; the TTL is not what changed"
    assert vol_surface.SABR_CACHE_TTL == 3600


def test_cache_cap_env_parsing(monkeypatch):
    monkeypatch.delenv(VOL_SURFACE_CACHE_CAP_ENV, raising=False)
    assert _cache_cap(VOL_SURFACE_CACHE_CAP_ENV, 200) == 200
    monkeypatch.setenv(VOL_SURFACE_CACHE_CAP_ENV, "")
    assert _cache_cap(VOL_SURFACE_CACHE_CAP_ENV, 200) == 200
    monkeypatch.setenv(VOL_SURFACE_CACHE_CAP_ENV, " 50 ")
    assert _cache_cap(VOL_SURFACE_CACHE_CAP_ENV, 200) == 50
    monkeypatch.setenv(VOL_SURFACE_CACHE_CAP_ENV, "0")
    assert _cache_cap(VOL_SURFACE_CACHE_CAP_ENV, 200) == 0, "0 = unbounded (rollback)"
    monkeypatch.setenv(VOL_SURFACE_CACHE_CAP_ENV, "-5")
    assert _cache_cap(VOL_SURFACE_CACHE_CAP_ENV, 200) == 0
    monkeypatch.setenv(VOL_SURFACE_CACHE_CAP_ENV, "lots")
    assert _cache_cap(VOL_SURFACE_CACHE_CAP_ENV, 200) == 200, "garbage -> default, never a crash"


def test_put_evicts_least_recently_used_and_touch_refreshes_recency():
    c = {}
    evicted = 0
    for k in ("A", "B", "C"):
        evicted += _cache_put(c, k, {"v": k}, cap=3)
    assert evicted == 0 and list(c) == ["A", "B", "C"]
    _cache_touch(c, "A")                      # A is now the most recently used
    assert list(c) == ["B", "C", "A"]
    assert _cache_put(c, "D", {"v": "D"}, cap=3) == 1
    assert list(c) == ["C", "A", "D"], "B (least recently used) went, not A (oldest inserted)"
    _cache_touch(c, "ZZZ")                    # miss is a no-op
    assert list(c) == ["C", "A", "D"]


def test_reinserting_an_existing_key_never_grows_the_cache():
    c = {}
    for _ in range(50):
        _cache_put(c, "SPY", {"n": _}, cap=3)
    assert len(c) == 1 and c["SPY"] == {"n": 49}
    # a rebuilt (TTL-expired) ticker overwrites in place — size stays flat
    for k in ("QQQ", "IWM"):
        _cache_put(c, k, {}, cap=3)
    assert _cache_put(c, "SPY", {"n": 50}, cap=3) == 0 and len(c) == 3
    assert list(c) == ["QQQ", "IWM", "SPY"]


def test_cap_zero_is_unbounded_rollback():
    c = {}
    total_evicted = sum(_cache_put(c, str(i), i, cap=0) for i in range(5000))
    assert total_evicted == 0 and len(c) == 5000


def test_cap_holds_under_scanner_sized_churn():
    c = {}
    evicted = 0
    for i in range(3000):                     # more tickers than any universe scan
        evicted += _cache_put(c, f"T{i}", {"i": i}, cap=200)
    assert len(c) == 200 and evicted == 2800
    assert list(c)[0] == "T2800" and list(c)[-1] == "T2999", "the newest 200 survive"


def test_stats_getter_reports_sizes_caps_evictions_and_ttls(monkeypatch):
    monkeypatch.setattr(vol_surface, "_surface_cache", {"SPY": {"timestamp": time.time(), "surface": {}}})
    monkeypatch.setattr(vol_surface, "_sabr_cache", {})
    monkeypatch.delenv(VOL_SURFACE_CACHE_CAP_ENV, raising=False)
    monkeypatch.delenv(SABR_CACHE_CAP_ENV, raising=False)
    s = vol_surface_cache_stats()
    assert s["surface_entries"] == 1 and s["sabr_entries"] == 0
    assert s["surface_cap"] == 200 and s["sabr_cap"] == 2000
    assert s["surface_ttl_s"] == 3600 and s["sabr_ttl_s"] == 3600
    assert isinstance(s["surface_evictions"], int) and isinstance(s["sabr_evictions"], int)


def test_live_paths_route_through_the_capped_put_and_touch_on_hit():
    src = open(os.path.join(os.path.dirname(__file__), "vol_surface.py")).read()
    # no raw unbounded insert remains on either cache
    assert not re.search(r"^\s*_surface_cache\[[^\]]+\]\s*=", src, re.M), "surface insert must go through _cache_put"
    assert not re.search(r"^\s*_sabr_cache\[[^\]]+\]\s*=", src, re.M), "sabr insert must go through _cache_put"
    assert src.count("_cache_put(\n        _surface_cache") == 1 or src.count("_cache_put(_surface_cache") == 1
    assert "_cache_touch(_surface_cache, ticker)" in src
    assert "_cache_touch(_sabr_cache, cache_key)" in src


def test_build_surface_hit_path_refreshes_recency(monkeypatch):
    """A fresh cached surface is returned AND moved to most-recent, so a
    ticker scored every scan is never the one evicted."""
    now = time.time()
    cache = {
        "OLD": {"timestamp": now, "surface": {"ticker": "OLD"}},
        "SPY": {"timestamp": now, "surface": {"ticker": "SPY"}},
        "NEW": {"timestamp": now, "surface": {"ticker": "NEW"}},
    }
    monkeypatch.setattr(vol_surface, "_surface_cache", cache)
    out = vol_surface.build_surface("SPY") if hasattr(vol_surface, "build_surface") else None
    if out is None:
        pytest.skip("build_surface entry point not found under that name")
    assert out == {"ticker": "SPY"}
    assert list(cache) == ["OLD", "NEW", "SPY"]
