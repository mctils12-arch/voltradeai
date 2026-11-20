"""
Tier engine: a failed VXX fetch must not read as a calm market (2026-11-20).

Live, 2026-11-17: the tier engine's exposure cap blocked every scan of the day
under CAUTION / BEAR except two whose VXX fetch had failed. macro_data's
substitute vxx_ratio 1.0 read as NEUTRAL on those scans and both sold a put.
These tests pin: the substitute is never used as a reading while the guard is
on; the last real reading stands in while it is recent; with none, the scan is
blocked; and bot_engine resolves the reading before the kill switches see it.
"""

import json
import os
import re

import pytest

import tier_macro_guard as g
import tiered_strategy

REAL = {"vxx_ratio": 1.22, "vxx_closes": [20.0] * 30, "vxx_latest": 24.4, "spy_vs_ma50": 1.0}
# What macro_data.get_macro_snapshot() returns when the VXX fetch fails.
FAILED = {"vxx_ratio": 1.0, "vxx_closes": [], "spy_vs_ma50": 1.0, "data_quality_failures": ["vxx_ratio"]}
T0 = 1_800_000_000.0


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "last.json")


def _r(macro, now, path, **kw):
    kw.setdefault("max_age_s", 3600)
    kw.setdefault("enabled", True)
    return g.resolve_vxx(macro, now=now, path=path, **kw)


def test_a_real_reading_is_used_and_remembered(path):
    out = _r(REAL, T0, path)
    assert out == {"vxx_ratio": 1.22, "source": "live", "age_s": 0.0, "block": False, "reason": ""}
    assert json.load(open(path)) == {"vxx_ratio": 1.22, "at": T0}


def test_a_failed_fetch_uses_the_last_real_reading_not_one(path):
    _r(REAL, T0, path)
    out = _r(FAILED, T0 + 240, path)
    assert out["source"] == "held" and out["vxx_ratio"] == 1.22 and out["block"] is False
    assert out["age_s"] == 240.0
    assert "1.22" in out["reason"] and "4 min ago" in out["reason"]


def test_the_2026_11_17_scan_is_classified_by_the_held_reading(path):
    """The defect itself: 1.0 gave NEUTRAL and a 95% cap; the reading four
    minutes earlier was CAUTION (60%)."""
    _r({**REAL, "vxx_ratio": 1.08}, T0, path)
    held = _r(FAILED, T0 + 240, path)["vxx_ratio"]
    before = tiered_strategy.get_regime_caps(FAILED["vxx_ratio"], 1.0)
    after = tiered_strategy.get_regime_caps(held, 1.0)
    assert before["regime"] == "NEUTRAL" and before["MAX_TOTAL_EXPOSURE"] == 0.95
    assert after["regime"] == "CAUTION" and after["MAX_TOTAL_EXPOSURE"] == 0.60


def test_a_held_reading_above_the_tier_kill_still_kills_tiers_2_to_4(path):
    _r({**REAL, "vxx_ratio": 1.45}, T0, path)
    held = _r(FAILED, T0 + 60, path)["vxx_ratio"]
    ctx = tiered_strategy.TierContext(equity=100_000, vxx_ratio=held)
    assert tiered_strategy.regime_kill_for_tier(ctx, 2) is True
    assert tiered_strategy.regime_kill_for_tier(tiered_strategy.TierContext(equity=100_000, vxx_ratio=1.0), 2) is False


def test_a_reading_older_than_the_limit_blocks_the_scan(path):
    _r(REAL, T0, path)
    out = _r(FAILED, T0 + 3601, path)
    assert out["source"] == "none" and out["block"] is True
    assert out["vxx_ratio"] == 1.0   # what the kill switches received before the guard
    assert "60 min old" in out["reason"] and "no new tier entries" in out["reason"]
    # exactly at the limit is still held
    assert _r(FAILED, T0 + 3600, path)["source"] == "held"


def test_no_remembered_reading_blocks_the_scan(path):
    out = _r(FAILED, T0, path)
    assert out["source"] == "none" and out["block"] is True and "none remembered" in out["reason"]


@pytest.mark.parametrize("macro", [None, {}, "x", {"vxx_ratio": None}, {"vxx_ratio": float("nan"), "vxx_closes": [1.0]},
                                   {"vxx_ratio": 0, "vxx_closes": [1.0]}, {"vxx_ratio": -1.2, "vxx_closes": [1.0]}])
def test_an_unusable_snapshot_is_a_failed_fetch(path, macro):
    _r(REAL, T0, path)
    assert _r(macro, T0 + 10, path)["source"] == "held"


def test_a_real_reading_of_exactly_one_is_a_reading(path):
    out = _r({**REAL, "vxx_ratio": 1.0}, T0, path)
    assert out["source"] == "live" and out["vxx_ratio"] == 1.0


def test_a_cached_snapshot_ages_from_when_it_was_taken(path):
    """macro_data serves one snapshot for 300 s; a hold counts from its stamp."""
    _r({**REAL, "timestamp": T0 - 280}, T0, path)
    assert json.load(open(path))["at"] == T0 - 280
    assert _r(FAILED, T0 + 20, path)["age_s"] == 300.0
    # a stamp from the future is not believed
    _r({**REAL, "timestamp": T0 + 9999}, T0 + 30, path)
    assert json.load(open(path))["at"] == T0 + 30


def test_a_corrupt_memory_file_is_no_reading(path):
    for body in ("{not json", "[]", '{"vxx_ratio": "x", "at": 1}', '{"vxx_ratio": 1.2}'):
        with open(path, "w") as f:
            f.write(body)
        assert _r(FAILED, T0, path)["source"] == "none"


def test_an_unwritable_memory_still_uses_the_reading_in_hand(tmp_path):
    out = _r(REAL, T0, str(tmp_path / "missing_dir" / "last.json"))
    assert out["source"] == "live" and out["vxx_ratio"] == 1.22


def test_the_off_switch_restores_the_old_reading(path):
    _r(REAL, T0, path)
    out = _r(FAILED, T0 + 60, path, enabled=False)
    assert out == {"vxx_ratio": 1.0, "source": "disabled", "age_s": None, "block": False, "reason": ""}
    assert g.guard_enabled({"VOLTRADE_TIER_VXX_GUARD": "off"}) is False
    assert g.guard_enabled({"VOLTRADE_TIER_VXX_GUARD": "on"}) is True
    assert g.guard_enabled({}) is True   # system_config default


def test_config_carries_the_switch_and_a_bounded_hold():
    from system_config import BASE_CONFIG
    assert BASE_CONFIG["TIER_VXX_FALLBACK_GUARD"] is True
    assert 300 <= BASE_CONFIG["TIER_VXX_HOLD_MAX_AGE_S"] <= 23400


def test_the_floor_and_the_tier_engine_share_one_definition_of_a_failed_fetch():
    import bot_engine
    for macro in (REAL, FAILED, None, {}, {"vxx_ratio": 1.0, "vxx_latest": 20.0}):
        assert bot_engine._floor_macro_vxx_is_fallback(macro) == g.vxx_is_fallback(macro)


def test_bot_engine_resolves_the_reading_before_the_kill_switches_and_blocks_on_none():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_engine.py")).read()
    block = src[src.index("_timing_log(\"tier_engine_start\")"):src.index("\"tier_engine_breakdown\"")]
    resolve = block.index("_tmg.resolve_vxx(")
    kill = block.index("check_kill_switches(")
    ctx = block.index("ctx = TierContext(")
    assert resolve < kill < ctx
    # the one reading feeds both: no second read of the raw snapshot's vxx_ratio here
    assert not re.search(r"_macro\.get\(\s*[\"']vxx_ratio[\"']", block)
    assert "vxx_ratio=vxx_r" in block[kill:kill + 400] and "vxx_ratio=vxx_r" in block[ctx:ctx + 1200]
    gate = block.index('if not kill_status["killed"] and tier_macro["block"]:')
    assert kill < gate < ctx
    assert '"tier_macro": tier_macro' in src
