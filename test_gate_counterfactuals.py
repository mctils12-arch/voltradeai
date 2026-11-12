"""
[RULE-REVIEW] 2026-11-12 — counterfactual logging for the tier engine's gates
(tiered_strategy's gate shadow, shadow_portfolio.log_gate_rows) and the report
that reads it back (gate_counterfactuals_probe.py, diag probe
`gate_counterfactuals`).

LOGGING AND REPORTING ONLY. The pins below are of three kinds:
  1. the rows: a master kill's Tier-1 records carry the tier and the dollars;
     the Tier 2 / Tier 3 actions the engine would have proposed are recorded;
     the allocator's scale-down records what it cut under the constraint that
     bound; each row once per name per ET day;
  2. the gates are untouched: what a kill returns and what the allocator
     sizes are the same with the shadow on, failing, or over budget;
  3. the report: per gate and tier, name-days, the outcome distribution, the
     same-period base rates and the dollar estimate, hand-computed.
Offline: every file lives in tmp_path.
"""
import contextlib
import json
import os
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

import position_mgmt as pm
import shadow_portfolio as sp
import system_config
import tiered_strategy as ts

try:
    import gate_counterfactuals_probe as gcp
except ImportError:   # a tree without the probe: the file still collects and each report test fails on its own
    gcp = None

ROOT = os.path.dirname(os.path.abspath(__file__))
UNIVERSE = ["AMD", "AAPL", "MSFT", "NVDA", "AMZN"]
PRICES = {"AMD": 50.0, "AAPL": 60.0, "MSFT": 70.0, "QQQ": 600.0}

# The thresholds these tests are written against, pinned for the test run. They pin LOGGING: a later tune of a
# live cap, limit or reserve (the decision this evidence exists for) must not have to touch them.
CAPS = {"MAX_TOTAL_EXPOSURE": 0.95, "MAX_OPTIONS_PCT": 0.08, "MAX_OPTIONS_POSITIONS": 6, "MAX_POSITION_PCT": 0.15,
        "regime": "BULL"}
TIER_CONSTANTS = {"PORTFOLIO_DRAWDOWN_KILL": -0.20, "DAILY_LOSS_LIMIT_PCT": -0.03, "T2_LEVERAGE_DRAWDOWN_GATE": -0.08,
                  "REGIME_KILL_VXX_RATIO": 1.40, "T2_LEVERAGE_FACTOR": 3.0, "T2_REQUIRES_PM": True,
                  "T3_LEVERAGE_FACTOR": 2.0, "T3_MAX_ALLOCATION_PCT": 0.40, "T3_VXX_MAX": 0.95}
CONFIG = {"FLOOR_MODE": "vol_managed_spy", "FLOOR_VM_CAPITAL_PCT": 0.85, "CASH_RESERVE_ENABLED": True,
          "CASH_RESERVE_PCT_BULL": 0.05, "CASH_RESERVE_PCT_NEUTRAL": 0.08, "CASH_RESERVE_PCT_CAUTION": 0.15}


@pytest.fixture(autouse=True)
def iso(tmp_path, monkeypatch):
    import regime_engine
    monkeypatch.setattr(sp, "SHADOW_LOG_PATH", str(tmp_path / "shadow.json"))
    monkeypatch.setattr(sp, "SHADOW_LOCK_PATH", str(tmp_path / "shadow.json.lock"))
    if hasattr(sp, "_GATE_ROWS_SEEN"):          # absent on a tree without the change: tests then fail on their own
        monkeypatch.setitem(sp._GATE_ROWS_SEEN, "day", None)
        monkeypatch.setitem(sp._GATE_ROWS_SEEN, "keys", set())
    monkeypatch.setattr(pm, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pm, "POSITION_MGMT_LOG_PATH", str(tmp_path / "pm.jsonl"))
    monkeypatch.setattr(regime_engine, "risk_budget_for", lambda *a, **k: None, raising=False)
    for name, value in TIER_CONSTANTS.items():
        monkeypatch.setattr(ts, name, value)
    for tier, on in {1: True, 2: True, 3: True, 4: False}.items():
        monkeypatch.setitem(ts.TIERS_ENABLED, tier, on)
    for key, value in CONFIG.items():
        monkeypatch.setitem(system_config.BASE_CONFIG, key, value)
    pm._dedup_seen.clear()
    sp._shadow_log_cache_reset()
    getattr(ts, "_GATE_SHADOW_LAST", {}).clear()
    yield tmp_path
    sp._shadow_log_cache_reset()


def saved():
    if not os.path.exists(sp.SHADOW_LOG_PATH):
        return []
    with open(sp.SHADOW_LOG_PATH) as f:
        return json.load(f)


def stock(sym, mv, px=100.0):
    return {"symbol": sym, "market_value": mv, "current_price": px, "asset_class": "us_equity"}


def ctx(positions, equity=100_000.0, peak=100_000.0, vxx=0.9, spy=1.03, regime="BULL", pm_ok=True, daily=0.0):
    return ts.TierContext(equity=equity, peak_equity=peak, buying_power=4 * equity, positions=positions,
                          vxx_ratio=vxx, spy_vs_ma50=spy, regime=regime, portfolio_margin=pm_ok,
                          daily_pnl_pct=daily)


def csp(ticker, size, tier=1):
    return ts.TierAction(tier=tier, action="SELL_CSP", ticker=ticker, strategy="csp", size_pct=size, reason="t1",
                         metadata={"vxx_ratio": 0.9, "regime_label": "BULL"})


def layer2(prices):
    return {"scores": [{"ticker": t, "price": p} for t, p in prices.items()]}


def fixed_caps():
    """The tier engine reads CAPS (a BULL regime with a 95% cap), whatever system_config holds."""
    return patch("tiered_strategy.get_regime_caps", side_effect=lambda *a, **k: dict(CAPS))


@contextlib.contextmanager
def tier_world(prices=None, reserve_room=None):
    """CAPS; Tier 1 proposes the first three UNIVERSE names at 8% each; the
    cash reserve's room is fixed (None = reserve off)."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(fixed_caps())
        stack.enter_context(patch("tiered_strategy._get_t1_universe", return_value=UNIVERSE))
        stack.enter_context(patch("csp_universe._load_layer2_cache",
                                  return_value=layer2(PRICES if prices is None else prices)))
        stack.enter_context(patch("tiered_strategy._reserve_room_pct", return_value=reserve_room))
        yield


AT_THE_CAP = [stock("SPY", 85_000, 700.0), stock("XOM", 11_000)]     # 96% invested >= CAPS' 95%
WITH_ROOM = [stock("SPY", 20_000, 700.0)]                           # 20% invested: 75% under CAPS' cap
OVER_ANY_CAP = [stock("SPY", 150_000, 700.0)]                       # 150% invested: over every regime's cap


# ══ 1. the record's gate block ═══════════════════════════════════════════════

def test_gate_block_is_whitelisted_and_absent_on_every_other_record():
    block = sp._sanitize_gate({
        "tier": 2, "action": "scaled", "order": "SELL_CSP", "regime": "NEUTRAL", "equity": 100_000, "size_pct": 0.123456789,
        "kept_pct": 0.05, "cut_dollars": 7345.678, "net_dollars": 100.004, "net_frac": 0.0136,
        "cap_room": 0.1, "reserve_room": None, "also": ["cash_reserve"] * 9, "t2": "x" * 99, "t3": None,
        "ticker": "LEAK", "anything": {"else": 1}})
    assert block == {"v": 1, "tier": 2, "action": "scaled", "order": "SELL_CSP", "regime": "NEUTRAL", "equity": 100_000.0,
                     "size_pct": 0.123457, "kept_pct": 0.05, "cut_dollars": 7345.68, "net_dollars": 100.0,
                     "net_frac": 0.0136, "cap_room": 0.1, "also": ["cash_reserve"] * 4, "t2": "x" * 24}
    for bad in (None, {}, "x", {"tier": 9, "action": "dropped"}, {"tier": 1, "action": "killed"},
                {"tier": 1.5, "action": "dropped"}, {"tier": True, "action": "dropped"}, {"action": "dropped"}):
        assert sp._sanitize_gate(bad) is None
    assert "aside" not in sp._sanitize_gate({"tier": 1, "action": "dropped", "aside": 1})   # only log_gate_rows sets it

    sp.log_candidate("AMD", {}, 0.0, "rejected_masterkill", "Portfolio invested 96% >= 95%", entry_price=50.0,
                     gate={"tier": 1, "action": "dropped", "cut_dollars": 8000.0})
    sp.log_candidate("AAPL", {"rsi": 51}, 70.0, "taken", "", entry_price=60.0)
    amd, aapl = saved()
    assert amd["gate"] == {"v": 1, "tier": 1, "action": "dropped", "cut_dollars": 8000.0}
    assert "gate" not in aapl      # every record that is not a gated tier action keeps its shape


def test_log_gate_rows_writes_once_and_keeps_the_first_per_name_per_et_day():
    def row(ticker="AMD", decision="rejected_masterkill", reason="Daily loss limit: -3.5% <= -3%", tier=2,
            action="dropped", price=50.0):
        return {"ticker": ticker, "decision": decision, "reason": reason, "entry_price": price,
                "vxx_ratio": 0.9, "regime_label": "BULL", "side": "sell", "action_label": "SELL CSP",
                "trade_type": "options", "gate": {"tier": tier, "action": action, "cut_dollars": 8000.0}}

    with patch.object(sp, "_save_shadow_log", wraps=sp._save_shadow_log) as save:
        out = sp.log_gate_rows([row("AMD"), row("AAPL"), row("AMD"),            # the third repeats the first
                                row("MSFT", price=0.0), row("NVDA", decision="taken"),
                                {"ticker": "INTC", "decision": "rejected_masterkill", "entry_price": 30.0},
                                "not a row"])
    assert out == {"received": 7, "logged": 2, "duplicates": 1, "skipped": 4, "saved": True}
    assert save.call_count == 1                                                  # one save for the whole batch
    recs = saved()
    assert [(r["ticker"], r["decision"], r["gate"]["tier"]) for r in recs] == [
        ("AMD", "rejected_masterkill", 2), ("AAPL", "rejected_masterkill", 2)]
    assert recs[0]["features"] == {} and recs[0]["score"] == 0.0               # never an ML training row
    assert recs[0]["outcomes"] == {"+5d": None, "+10d": None, "+20d": None}     # the nightly backfill labels it

    # the next scan proposes the same names: nothing is written and the log is not even loaded
    with patch.object(sp, "_load_shadow_log", wraps=sp._load_shadow_log) as load:
        again = sp.log_gate_rows([row("AMD"), row("AAPL")])
    assert again == {"received": 2, "logged": 0, "duplicates": 2, "skipped": 0, "saved": False}
    assert load.call_count == 0
    # a different tier, action, gate or decision is a different row
    more = sp.log_gate_rows([row("AMD", tier=1), row("AMD", action="scaled"),
                             row("AMD", reason="Portfolio invested 96.0% >= 95% (regime=BULL)"),
                             row("AMD", decision="rejected_cash_reserve", reason="cash reserve leaves no room")])
    assert more["logged"] == 4 and len(saved()) == 6


def test_log_gate_rows_finds_todays_rows_on_file_after_a_restart_and_never_raises():
    row = {"ticker": "AMD", "decision": "rejected_cash_reserve", "reason": "cash reserve leaves no room",
           "entry_price": 50.0, "gate": {"tier": 1, "action": "scaled", "cut_dollars": 1000.0}}
    assert sp.log_gate_rows([row])["logged"] == 1
    sp._GATE_ROWS_SEEN.update(day=None, keys=set())                              # a new process remembers nothing
    assert sp.log_gate_rows([dict(row)]) == {"received": 1, "logged": 0, "duplicates": 1, "skipped": 0, "saved": False}
    assert len(saved()) == 1
    assert sp.log_gate_rows(None)["logged"] == 0 and sp.log_gate_rows([])["received"] == 0
    with patch.object(sp, "_load_shadow_log", side_effect=RuntimeError("disk full")):
        out = sp.log_gate_rows([dict(row, ticker="AAPL")])
    assert out["logged"] == 0 and "disk full" in out["error"]
    # a save that did not land is retried by the next scan (the key was not remembered)
    with patch.object(sp, "_save_shadow_log", return_value=False):
        assert sp.log_gate_rows([dict(row, ticker="MSFT")])["logged"] == 0
    assert sp.log_gate_rows([dict(row, ticker="MSFT")])["logged"] == 1


def test_a_gate_row_never_hides_the_scanner_candidates_own_record_from_a_gate_relabel():
    """bot.ts relabels the freshest "taken" record of a name its admission gates refuse
    (update_decisions_batch), and stops at the most recent record for that name. A gate row is
    appended later in the same scan; it did not exist before this change, so the relabel walks past it."""
    sp.log_candidate("AMD", {"rsi": 55.0}, 71.0, "taken", "", entry_price=50.0)        # deep_score, this scan
    sp.log_gate_rows([{"ticker": "AMD", "decision": "rejected_masterkill", "entry_price": 50.0,
                       "reason": "Portfolio invested 80.0% + proposed 16.0% > 95% (regime=BULL): allocator scaled to 94%",
                       "gate": {"tier": 1, "action": "scaled", "cut_dollars": 500.0}}])   # the allocator, same scan
    assert saved()[1]["gate"]["aside"] == 1
    out = sp.update_decisions_batch([{"ticker": "AMD", "decision": "rejected_stock_slots", "reason": "no stock slots"}])
    assert (out["updated"], out["saved"]) == (1, True)
    assert [x["decision"] for x in saved()] == ["rejected_stock_slots", "rejected_masterkill"]
    assert saved()[1]["gate"]["action"] == "scaled"                                    # the gate row itself is untouched


# ══ 2. the master kill's shadow ══════════════════════════════════════════════

def test_master_gates_holding_agrees_with_master_kill_switch():
    """On system_config's own caps (no CAPS here): the three conditions, one by one, are the switch's."""
    order = ("drawdown", "daily_loss", "exposure_ceiling")
    for peak in (100_000.0, 130_000.0):                      # 0% and -23% from peak
        for daily in (0.0, -0.035):
            for positions in (WITH_ROOM, OVER_ANY_CAP):
                c = ctx(positions, peak=peak, daily=daily)
                holding = ts._master_gates_holding(c)
                killed, reason = ts.master_kill_switch(c)
                assert list(holding) == list(order)
                assert killed == any(holding.values())
                if killed:
                    first = next(g for g in order if holding[g])
                    assert sp._masterkill_reason_category(reason) == first


def test_exposure_ceiling_kill_tier1_records_carry_tier_dollars_and_what_tiers_2_and_3_would_do():
    c = ctx(AT_THE_CAP)
    with tier_world(reserve_room=0.0):
        killed, reason = ts.master_kill_switch(c)
        assert killed and reason == "Portfolio invested 96.0% >= 95% (regime=BULL)"
        result = ts.TieredStrategy().run_tiers(c)

    assert (result["killed"], result["actions"], result["tier_stats"]) == (True, [], {})   # the kill is the kill
    recs = saved()
    assert [(x["ticker"], x["decision"], x["decision_reason"]) for x in recs] == [
        (t, "rejected_masterkill", reason) for t in ("AMD", "AAPL", "MSFT")]
    for x in recs:
        g = x["gate"]
        assert (g["tier"], g["action"], g["order"], g["regime"]) == (1, "dropped", "SELL_CSP", "BULL")
        assert (g["size_pct"], g["kept_pct"], g["cut_dollars"], g["equity"]) == (0.08, 0.0, 8000.0, 100_000.0)
        # Tier 2 and Tier 3 check the same cap themselves: no row for them can exist under this gate
        assert (g["t2"], g["t3"]) == ("no_headroom", "floor_owns_spy")
        # and the cash reserve had no room either: none of the cut would have gone through
        assert (g["cap_room"], g["reserve_room"], g["net_frac"], g["net_dollars"]) == (0.0, 0.0, 0.0, 0.0)
        assert g["also"] == ["cash_reserve"]
        assert sp._masterkill_reason_category(x["decision_reason"]) == "exposure_ceiling"
    shadow = result["tier_timings"]["gate_shadow"]
    assert (shadow["gate"], shadow["tier1_rows"], shadow["tier2"], shadow["tier3"], shadow["rows"]) == (
        "exposure_ceiling", 3, "no_headroom", "floor_owns_spy", None)


def test_kill_shadow_on_the_real_regime_caps_records_the_regime_whose_cap_killed():
    """No CAPS: system_config's caps and the regime risk budget's path, whatever their values are today."""
    c = ctx(OVER_ANY_CAP)
    caps = ts.get_regime_caps(c.vxx_ratio, c.spy_vs_ma50, c.equity, positions=c.positions)
    with patch("tiered_strategy._get_t1_universe", return_value=UNIVERSE), \
         patch("csp_universe._load_layer2_cache", return_value=layer2(PRICES)):
        result = ts.TieredStrategy().run_tiers(c)
    assert result["killed"] is True and f"(regime={caps['regime']})" in result["kill_reason"]
    recs = saved()
    assert len(recs) == result["tier_timings"]["gate_shadow"]["tier1_rows"] > 0
    for x in recs:
        g = x["gate"]
        assert (g["tier"], g["action"], g["regime"], g["cap_room"], g["t2"]) == (1, "dropped", caps["regime"], 0.0, "no_headroom")
        assert g["cut_dollars"] == pytest.approx(g["size_pct"] * c.equity, abs=0.01) and g["size_pct"] > 0
        assert gcp.regime_of(x) == caps["regime"]


def test_exposure_ceiling_cut_is_all_net_when_no_other_gate_holds():
    with tier_world(reserve_room=None):
        ts.log_masterkill_csp_shadow(ctx(AT_THE_CAP), "Portfolio invested 96.0% >= 95% (regime=BULL)")
    g = saved()[0]["gate"]
    assert (g["net_frac"], g["net_dollars"]) == (1.0, 8000.0)
    assert "also" not in g and "reserve_room" not in g


def test_daily_loss_kill_records_the_tier2_and_tier3_actions_it_also_blocked():
    c = ctx(WITH_ROOM, daily=-0.035)
    with tier_world(reserve_room=0.44), patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "basket"}):
        result = ts.TieredStrategy().run_tiers(c)
        again = ts.TieredStrategy().run_tiers(c)             # the next scan, same day

    assert result["kill_reason"] == "Daily loss limit: -3.5% <= -3%"
    assert (result["killed"], result["actions"], result["tier_stats"]) == (True, [], {})   # nothing is returned
    assert result["tier_timings"]["gate_shadow"]["rows"] == {
        "received": 5, "logged": 5, "duplicates": 0, "skipped": 0, "saved": True}
    assert again["tier_timings"]["gate_shadow"]["rows"]["duplicates"] == 5      # once per name per day
    recs = saved()
    upper = [x for x in recs if x["gate"]["tier"] != 1]
    assert len(recs) == 3 + 5 + 3 and len(upper) == 5        # the Tier-1 records still repeat per scan, as before
    by = {(x["ticker"], x["gate"]["tier"]): x for x in upper}
    assert set(by) == {("AMD", 2), ("AAPL", 2), ("MSFT", 2), ("SPY", 3), ("QQQ", 3)}
    for x in upper:
        assert (x["decision"], x["decision_reason"]) == ("rejected_masterkill", "Daily loss limit: -3.5% <= -3%")
        assert sp._masterkill_reason_category(x["decision_reason"]) == "daily_loss"
        assert x["gate"]["action"] == "dropped" and (x["gate"]["t2"], x["gate"]["t3"]) == ("levered", "proposed")
        # 24% Tier 1 + 24% Tier-2 leverage + 40% Tier 3 = 88% proposed; the allocator's rooms (75% under
        # the cap, 44% above the cash reserve) would have let 44% through: half of every cut
        assert (x["gate"]["cap_room"], x["gate"]["reserve_room"], x["gate"]["net_frac"]) == (0.75, 0.44, 0.5)
        assert x["gate"]["also"] == ["exposure_ceiling", "cash_reserve"]
    amd2, spy3, qqq3 = by[("AMD", 2)], by[("SPY", 3)], by[("QQQ", 3)]
    # Tier 2 would have levered each 8% put to 16%: its row is the 8% it ADDS, at the Tier-1 price
    assert (amd2["gate"]["size_pct"], amd2["gate"]["cut_dollars"], amd2["gate"]["net_dollars"]) == (0.08, 8000.0, 4000.0)
    assert (amd2["entry_price"], amd2["action_label"], amd2["trade_type"]) == (50.0, "SELL CSP", "options")
    # Tier 3 would have bought SPY and QQQ at 20% each; SPY is priced from the held position
    assert (spy3["gate"]["order"], spy3["gate"]["size_pct"], spy3["gate"]["cut_dollars"]) == ("BUY", 0.2, 20_000.0)
    assert (spy3["entry_price"], qqq3["entry_price"], spy3["action_label"], spy3["trade_type"]) == (700.0, 600.0, "BUY", "stock")
    t1 = [x for x in recs if x["gate"]["tier"] == 1][0]["gate"]
    assert (t1["cut_dollars"], t1["net_dollars"], t1["t2"], t1["t3"]) == (8000.0, 4000.0, "levered", "proposed")


def test_a_kill_another_master_gate_would_also_have_made_has_no_net_dollars():
    c = ctx(AT_THE_CAP, daily=-0.035)                        # the daily loss fires first; the ceiling holds too
    assert ts.master_kill_switch(c)[1].startswith("Daily loss limit")
    with tier_world(reserve_room=0.5):
        ts.log_masterkill_csp_shadow(c, ts.master_kill_switch(c)[1])
    g = saved()[0]["gate"]
    assert g["also"] == ["exposure_ceiling"] and (g["net_frac"], g["net_dollars"]) == (0.0, 0.0)


def test_tier2_and_tier3_statuses_name_the_gate_that_stops_the_real_function():
    with fixed_caps():
        t1 = [csp("AMD", 0.08), csp("AAPL", 0.08), csp("MSFT", 0.08)]

        def t2(c):
            status, added = ts._tier2_shadow(c, t1)
            real = ts.tier2_leverage_multiplier(c, t1)
            grew = any(x.tier == 2 and x.size_pct > x.metadata["original_size"] + 1e-9 for x in real)
            assert (status == "levered") == grew == bool(added)          # the real function's answer, never the mirror's
            return status

        assert t2(ctx(WITH_ROOM)) == "levered"
        assert t2(ctx(WITH_ROOM, pm_ok=False)) == "no_portfolio_margin"
        assert t2(ctx(WITH_ROOM, peak=110_000.0)) == "drawdown_gate"     # -9.1% from peak
        assert t2(ctx(WITH_ROOM, vxx=1.45)) == "vol_kill"
        assert t2(ctx(AT_THE_CAP)) == "no_headroom"
        assert t2(ctx([stock("SPY", 85_000)])) == "leverage_x1"          # 10% room under the cap < Tier 1's 24%
        with patch.dict(ts.TIERS_ENABLED, {2: False}):
            assert t2(ctx(WITH_ROOM)) == "disabled"
        assert ts._tier2_shadow(ctx(WITH_ROOM), []) == ("no_tier1_action", [])

        def t3(c):
            status, acts = ts._tier3_shadow(c)
            assert (status == "proposed") == bool(ts.tier3_trend_capture(c)) == bool(acts)
            return status

        assert t3(ctx(WITH_ROOM)) == "floor_owns_spy"                    # production: the vol-managed floor
        with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "basket"}):
            assert t3(ctx(WITH_ROOM)) == "proposed"
            assert t3(ctx(WITH_ROOM, vxx=1.45)) == "vol_kill"
            assert t3(ctx(WITH_ROOM, peak=110_000.0)) == "drawdown_gate"
            assert t3(ctx(WITH_ROOM, vxx=1.0)) == "vxx_above_max"
            assert t3(ctx(WITH_ROOM, spy=1.0)) == "spy_not_above_ma50"
            assert t3(ctx(WITH_ROOM, regime="NEUTRAL")) == "regime_not_bull"
            assert t3(ctx(AT_THE_CAP)) == "no_headroom"
            with patch.dict(ts.TIERS_ENABLED, {3: False}):
                assert t3(ctx(WITH_ROOM)) == "disabled"


def test_net_fraction_is_the_share_of_a_cut_the_other_gates_would_have_let_through():
    assert ts._net_fraction(0.24, 0.0, None) == 1.0                  # no other constraint
    assert ts._net_fraction(0.24, 0.0, 0.0) == 0.0                   # the other gate has no room at all
    assert ts._net_fraction(0.24, 0.0, 0.12) == pytest.approx(0.5)
    assert ts._net_fraction(0.24, 0.0, 0.50) == 1.0                  # more room than was asked for
    assert ts._net_fraction(0.24, 0.10, 0.15) == pytest.approx(0.05 / 0.14)   # a scale-down: 10% kept, 15% possible
    assert ts._net_fraction(0.24, 0.10, 0.05) == 0.0                 # the other gate is tighter than what was kept
    assert ts._net_fraction(0.10, 0.10, None) == 0.0                 # nothing was cut


def test_a_failing_shadow_plan_costs_no_tier1_record_and_never_raises():
    with tier_world(), patch("tiered_strategy._tier2_shadow", side_effect=RuntimeError("boom")):
        logged = ts.log_masterkill_csp_shadow(ctx(AT_THE_CAP), "Portfolio invested 96.0% >= 95% (regime=BULL)")
    assert logged == 3
    assert [x["ticker"] for x in saved()] == ["AMD", "AAPL", "MSFT"] and all("gate" not in x for x in saved())
    assert "RuntimeError: boom" in ts.get_gate_shadow_summary()["error"]

    with tier_world(), patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "basket"}), \
         patch.object(sp, "log_gate_rows", side_effect=RuntimeError("disk full")):
        result = ts.TieredStrategy().run_tiers(ctx(WITH_ROOM, daily=-0.035))      # must not raise
    assert result["killed"] is True and result["actions"] == []
    assert "disk full" in result["tier_timings"]["gate_shadow"]["error"]


def test_the_extra_rows_are_skipped_past_the_time_budget_and_make_no_network_call():
    c = ctx(WITH_ROOM, daily=-0.035)
    with tier_world(reserve_room=0.44), patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "basket"}), \
         patch.object(ts, "GATE_SHADOW_BUDGET_S", -1.0):
        ts.log_masterkill_csp_shadow(c, "Daily loss limit: -3.5% <= -3%")
    assert ts.get_gate_shadow_summary()["rows"] == {"skipped": "time budget"}
    assert [x["gate"]["tier"] for x in saved()] == [1, 1, 1]                      # the Tier-1 records are kept

    # the plan and the row writer read only what is already in hand
    t1 = [csp("AMD", 0.08), csp("AAPL", 0.08), csp("MSFT", 0.08)]
    with patch("socket.socket.connect", side_effect=AssertionError("network call")) as connect, fixed_caps(), \
         patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "basket"}), \
         patch("tiered_strategy._layer2_price_lookup", return_value=dict(PRICES)):
        plan = ts._masterkill_gate_plan(c, "Daily loss limit: -3.5% <= -3%", t1)
        ts._log_masterkill_upper_tiers(c, plan, ["AMD", "AAPL"], time.monotonic())
    assert connect.call_count == 0 and "error" not in plan
    # MSFT's Tier-1 record was not written (no price), so its Tier-2 row is not either
    assert ts.get_gate_shadow_summary()["rows"]["logged"] == 4
    assert sorted((x["ticker"], x["gate"]["tier"]) for x in saved() if x["gate"]["tier"] != 1) == [
        ("AAPL", 2), ("AMD", 2), ("QQQ", 3), ("SPY", 3)]


def test_reserve_room_is_the_number_the_allocator_reads_without_its_counterfactual_row():
    c = ctx([stock("SPY", 85_000)], vxx=1.0, spy=1.0)
    budget = {"budget": 0.5, "band": "reduced", "summary": "half the room"}
    with patch("position_mgmt.floor_plan_target_ratio", return_value=0.85), \
         patch("position_mgmt.regime_budget", return_value=budget), \
         patch("regime_engine.log_counterfactual") as cf:
        live = pm.live_headroom(c.equity, c.positions, "NEUTRAL")       # what _enforce_exposure_cap calls
        assert cf.call_count == 1                                        # ... and the row it writes
        room = ts._reserve_room_pct(c, "NEUTRAL")
        assert cf.call_count == 1                                        # the shadow's read writes none
    assert room == pytest.approx(live["headroom_dollars"] / c.equity) == pytest.approx(0.07)   # 100 - 85 - 8
    with patch.dict(system_config.BASE_CONFIG, {"CASH_RESERVE_ENABLED": False}):
        assert ts._reserve_room_pct(c, "NEUTRAL") is None


# ══ 3. the allocator's cut ═══════════════════════════════════════════════════

def run_allocator(c, actions, regime, floor_ratio):
    with patch("position_mgmt.floor_plan_target_ratio", return_value=floor_ratio), \
         patch("tiered_strategy.get_regime_caps", return_value={"MAX_TOTAL_EXPOSURE": 0.95, "regime": regime}), \
         patch("tiered_strategy._layer2_price_lookup", return_value={"AMD": 50.0, "INTC": 30.0, "XLE": 90.0}):
        return ts._enforce_exposure_cap(c, actions)


def test_scale_down_under_the_cap_records_what_was_cut_as_an_exposure_ceiling_row():
    c = ctx([stock("SPY", 80_000, 700.0)], vxx=1.0, spy=1.0)
    with patch.dict(system_config.BASE_CONFIG, {"CASH_RESERVE_ENABLED": False}):
        out = run_allocator(c, [csp("AMD", 0.08), csp("INTC", 0.08)], "BULL", 0.80)
    assert [round(x.size_pct, 6) for x in out] == [0.075, 0.075]                  # 16% proposed, 15% under the cap
    recs = saved()
    assert [(x["ticker"], x["decision"]) for x in recs] == [("AMD", "rejected_masterkill"), ("INTC", "rejected_masterkill")]
    for x in recs:
        assert x["decision_reason"] == ("Portfolio invested 80.0% + proposed 16.0% > 95% (regime=BULL): "
                                        "allocator scaled to 94%")
        assert sp._masterkill_reason_category(x["decision_reason"]) == "exposure_ceiling"
        g = x["gate"]
        assert (g["tier"], g["action"], g["order"], g["regime"]) == (1, "scaled", "SELL_CSP", "BULL")
        assert (g["size_pct"], g["kept_pct"], g["cut_dollars"]) == (0.08, 0.075, 500.0)
        assert (g["net_frac"], g["net_dollars"], g["cap_room"]) == (1.0, 500.0, 0.15)     # the reserve is off: all net
        assert "also" not in g and "reserve_room" not in g
        assert (x["action_label"], x["trade_type"], x["regime_label"]) == ("SELL CSP", "options", "BULL")
    assert (recs[0]["entry_price"], recs[1]["entry_price"]) == (50.0, 30.0)


def test_cap_bound_cut_the_cash_reserve_would_also_have_made_has_no_net_dollars():
    c = ctx([stock("SPY", 80_000, 700.0)], vxx=1.0, spy=1.0)
    run_allocator(c, [csp("AMD", 0.08), csp("INTC", 0.08)], "BULL", 0.80)     # cap room 15% = reserve room 100-80-5
    g = saved()[0]["gate"]
    assert saved()[0]["decision"] == "rejected_masterkill"
    assert (g["cap_room"], g["reserve_room"], g["net_frac"], g["net_dollars"], g["also"]) == (0.15, 0.15, 0.0, 0.0, ["cash_reserve"])


def test_scale_down_bound_by_the_cash_reserve_is_a_cash_reserve_row_not_an_exposure_ceiling_one():
    c = ctx([stock("SPY", 85_000, 700.0)], vxx=1.0, spy=1.0)
    out = run_allocator(c, [csp("AMD", 0.08)], "NEUTRAL", 0.85)                   # 100 - 85 - 8 = 7% vs 10% under the cap
    assert out[0].size_pct == pytest.approx(0.07) and "[cash reserve]" in out[0].reason
    (x,) = saved()
    assert (x["decision"], x["decision_reason"]) == ("rejected_cash_reserve", "cash reserve leaves room for 88% of the proposed size")
    assert sp.gate_reason(x) == "cash_reserve"
    g = x["gate"]
    assert (g["tier"], g["action"], g["size_pct"], g["kept_pct"], g["cut_dollars"]) == (1, "scaled", 0.08, 0.07, 1000.0)
    assert g["regime"] == "NEUTRAL"                                              # the regime whose reserve bound
    assert (g["cap_room"], g["reserve_room"], g["net_frac"], g["net_dollars"]) == (0.1, 0.07, 1.0, 1000.0)
    # the next scan cuts the same name again: one row per name per ET day
    run_allocator(ctx([stock("SPY", 85_000, 700.0)], vxx=1.0, spy=1.0), [csp("AMD", 0.08)], "NEUTRAL", 0.85)
    assert len(saved()) == 1


def test_cash_reserve_drop_records_carry_the_tier_and_dollars_and_cover_non_csp_actions():
    c = ctx([stock("SPY", 85_000, 700.0)], vxx=1.0, spy=1.0)
    buy = ts.TierAction(tier=3, action="BUY", ticker="XLE", strategy="trend_long", size_pct=0.10, reason="t3")
    out = run_allocator(c, [csp("AMD", 0.05), buy], "CAUTION", 0.85)              # 100 - 85 - 15: no room
    assert out == []
    recs = {x["ticker"]: x for x in saved()}
    assert set(recs) == {"AMD", "XLE"} and all(x["decision"] == "rejected_cash_reserve" for x in recs.values())
    amd, xle = recs["AMD"]["gate"], recs["XLE"]["gate"]
    assert (amd["tier"], amd["action"], amd["cut_dollars"]) == (1, "dropped", 5000.0)
    assert (xle["tier"], xle["action"], xle["order"], xle["cut_dollars"]) == (3, "dropped", "BUY", 10_000.0)
    # 15% proposed; the cap alone would have let 10% through
    assert amd["net_frac"] == xle["net_frac"] == pytest.approx(0.6667, abs=1e-4)
    assert amd["also"] == ["exposure_ceiling"]
    assert (recs["XLE"]["entry_price"], recs["XLE"]["action_label"], recs["XLE"]["trade_type"]) == (90.0, "BUY", "stock")


def test_allocator_sizes_and_reasons_do_not_depend_on_the_shadow():
    def sizes():
        pm._dedup_seen.clear()
        c = ctx([stock("SPY", 85_000, 700.0)], vxx=1.0, spy=1.0)
        out = run_allocator(c, [csp("AMD", 0.08), csp("INTC", 0.06)], "NEUTRAL", 0.85)
        return [(x.ticker, x.tier, x.size_pct, x.reason) for x in out]

    real = sizes()
    assert [round(x[2], 6) for x in real] == [0.04, 0.03]                          # 14% proposed, 7% above the reserve
    with patch.object(ts, "_log_allocator_gate_rows", lambda *a, **k: {}):         # the shadow off
        assert sizes() == real
    with patch.object(ts, "_gate_block", lambda *a, **k: 1 / 0):                   # the shadow failing
        assert sizes() == real
    with patch.object(sp, "log_gate_rows", side_effect=RuntimeError("disk full")):
        assert sizes() == real
    blocks = ts._allocator_gate_blocks(ctx(WITH_ROOM), {"MAX_TOTAL_EXPOSURE": object()}, [csp("AMD", 0.08)],
                                       [0.08], [0.04], "scaled", None, False, 0.0)
    assert blocks == ("exposure_ceiling", [None])                                 # a bad input is no block, not a raise


def test_an_action_the_allocator_left_whole_gets_no_row():
    c = ctx([stock("SPY", 50_000, 700.0)], vxx=1.0, spy=1.0)
    out = run_allocator(c, [csp("AMD", 0.08)], "BULL", 0.50)
    assert out[0].size_pct == 0.08 and saved() == []


# ══ 4. the report ════════════════════════════════════════════════════════════

A, B, C, E, F = "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-20", "2026-10-08"
CEILING = "Portfolio invested 96.0% >= 95% (regime=BULL)"
CEILING_CAUTION = "Portfolio invested 67.9% >= 60% (regime=CAUTION)"      # the live 2026-11-06 reason
DAILY = "Daily loss limit: -3.5% <= -3%"


def rec(ticker, day, decision, reason="", out5=None, gate=None, hour=14, features=None):
    r = {"ticker": ticker, "timestamp": f"{day}T{hour:02d}:00:00+00:00", "score": 0.0, "decision": decision,
         "decision_reason": reason, "entry_price": 50.0, "vxx_ratio": 1.0, "regime_label": "BULL",
         "side": "", "action_label": "", "trade_type": "", "features": features or {},
         "outcomes": {"+5d": out5, "+10d": None, "+20d": None}, "code_version": sp.CODE_VERSION,
         "ml": None, "score_breakdown": None}
    if gate is not None:
        r["gate"] = gate
    return r


def o(ret, exit_date, why):
    return {"return_pct": ret, "label": 1 if why == "target" else 0, "exit_price": 1.0, "exit_date": exit_date,
            "exit_reason": why, "label_method": "path_dependent_v2"}


def blk(tier, action, cut, net, **kw):
    return {"v": 1, "tier": tier, "action": action, "cut_dollars": cut, "net_dollars": net, **kw}


def fixture_log():
    legacy = [   # written before this change: no gate block
        rec("XAAA", A, "rejected_masterkill", CEILING, o(-4.0, "2026-10-07", "stop")),
        rec("XAAA", A, "rejected_masterkill", CEILING, o(-3.9, "2026-10-07", "stop"), hour=15),   # the next scan's copy
        rec("XBBB", A, "rejected_masterkill", CEILING, o(2.0, "2026-10-06", "target")),
        rec("XCCC", A, "rejected_masterkill", CEILING_CAUTION, o(2.0, "2026-10-07", "target")),
        rec("XAAA", C, "rejected_masterkill", CEILING, o(2.0, "2026-10-09", "target")),      # still inside XAAA's first trade
        rec("XBBB", C, "rejected_masterkill", CEILING_CAUTION, o(1.0, "2026-10-14", "timeout")),   # XBBB's first exited 10-06
        rec("XCCC", E, "rejected_masterkill", CEILING, None),                                # not old enough yet
        rec("XDDD", A, "rejected_masterkill", CEILING, {"return_pct": None, "label": -1, "exit_reason": "no_price_data"}),
    ]
    both = {"regime": "NEUTRAL", "t2": "levered", "t3": "proposed", "also": ["exposure_ceiling", "cash_reserve"]}
    new = [
        rec("XEEE", F, "rejected_masterkill", DAILY, o(2.0, "2026-10-09", "target"), blk(1, "dropped", 8000.0, 4000.0, **both)),
        rec("XFFF", F, "rejected_masterkill", DAILY, o(-4.0, "2026-10-12", "stop"), blk(1, "dropped", 8000.0, 4000.0, **both)),
        rec("XEEE", F, "rejected_masterkill", DAILY, o(2.0, "2026-10-09", "target"), blk(2, "dropped", 8000.0, 4000.0, **both)),
        rec("XQQQ", F, "rejected_masterkill", DAILY, o(1.0, "2026-10-15", "timeout"), blk(3, "dropped", 20000.0, 10000.0, **both)),
        rec("XGGG", F, "rejected_masterkill", "Portfolio invested 80.0% + proposed 16.0% > 95% (regime=BULL): allocator scaled to 94%",
            o(-4.0, "2026-10-12", "stop"), blk(1, "scaled", 500.0, 0.0, also=["cash_reserve"])),
        rec("XHHH", F, "rejected_cash_reserve", "cash reserve leaves no room", None),        # bot.ts's dispatcher: no block
        rec("XJJJ", F, "rejected_cash_reserve", "cash reserve: no room", o(2.0, "2026-10-09", "target"),
            features={"rsi": 61.0}),                                                          # the scanner's own stock entry
        rec("XIII", F, "rejected_cash_reserve", "cash reserve leaves room for 88% of the proposed size",
            o(2.0, "2026-10-09", "target"), blk(1, "scaled", 1000.0, 1000.0)),
    ]
    scanner = [
        rec("ZTA", A, "taken", "", o(2.0, "2026-10-06", "target")),
        rec("ZTB", A, "taken", "", o(-4.0, "2026-10-06", "stop")),
        rec("ZTC", A, "taken", "", o(2.0, "2026-10-06", "target")),
        rec("ZTC", A, "taken", "", o(-4.0, "2026-10-06", "stop"), hour=16),                   # a later copy: not first seen
        rec("ZTD", B, "taken", "", o(-4.0, "2026-10-07", "stop")),
        rec("ZTE", C, "taken", "", o(1.0, "2026-10-14", "timeout")),
        rec("ZTF", C, "taken", "", o(2.0, "2026-10-08", "target")),
        rec("ZTG", E, "taken", "", o(2.0, "2026-10-21", "target")),
        rec("ZRA", A, "rejected_score", "", o(-4.0, "2026-10-06", "stop")),
        rec("ZRB", C, "rejected_score", "", o(2.0, "2026-10-08", "target")),
    ]
    return legacy + new + scanner


def group(report, gate, tier, action):
    (g,) = [x for x in report["gates"][gate]["groups"] if (x["tier"], x["action"]) == (tier, action)]
    return g


def test_report_counts_name_days_per_gate_tier_and_action():
    rep = gcp.build_report(fixture_log())
    assert list(rep["gates"]) == ["exposure_ceiling", "daily_loss", "drawdown", "cash_reserve"]
    ceiling = rep["gates"]["exposure_ceiling"]
    assert (ceiling["rows"], ceiling["name_days"], ceiling["names"]) == (9, 8, 5)
    legacy = group(rep, "exposure_ceiling", "1", "dropped")       # pre-change records: Tier-1 CSPs, dropped
    assert (legacy["rows"], legacy["name_days"], legacy["names"], legacy["sized_name_days"]) == (8, 7, 4, 0)
    assert (legacy["first_day"], legacy["last_day"]) == (A, E)
    scaled = group(rep, "exposure_ceiling", "1", "scaled")
    assert (scaled["rows"], scaled["name_days"], scaled["sized_name_days"], scaled["other_gates"]) == (1, 1, 1, {"cash_reserve": 1})
    daily = rep["gates"]["daily_loss"]
    assert (daily["rows"], daily["name_days"], daily["names"]) == (4, 3, 3)      # XEEE's tier-1 and tier-2 rows: one name-day
    assert [(x["tier"], x["action"], x["rows"]) for x in daily["groups"]] == [
        ("1", "dropped", 2), ("2", "dropped", 1), ("3", "dropped", 1)]
    assert (daily["tier2_status"], daily["tier3_status"]) == ({"levered": 2}, {"proposed": 2})
    assert group(rep, "daily_loss", "1", "dropped")["other_gates"] == {"exposure_ceiling": 2, "cash_reserve": 2}
    assert (rep["gates"]["drawdown"]["rows"], rep["gates"]["drawdown"]["groups"]) == (0, [])
    reserve = rep["gates"]["cash_reserve"]
    # the reserve is account-wide: the scanner's own blocked stock entry is kept apart from the tier actions
    assert [(x["tier"], x["action"], x["rows"]) for x in reserve["groups"]] == [
        ("1", "scaled", 1), ("scanner", "dropped", 1), ("unrecorded", "dropped", 1)]
    assert rep["log"] == {"records": 26, "first_day": A, "last_day": E, "gate_rows": 16, "taken_rows": 8}


def test_report_splits_each_group_by_the_regime_whose_thresholds_applied():
    """Each regime has its own exposure cap, so a pooled exposure_ceiling result cannot move any one of them."""
    assert gcp.regime_of(rec("X", A, "rejected_masterkill", CEILING_CAUTION)) == "CAUTION"      # the reason's own text
    assert gcp.regime_of(rec("X", A, "rejected_masterkill", DAILY)) == "unrecorded"            # pre-change: not stored
    assert gcp.regime_of(rec("X", A, "rejected_masterkill", DAILY, gate=blk(1, "dropped", 1.0, 0.0, regime="BEAR"))) == "BEAR"
    assert gcp.regime_of(rec("X", A, "rejected_cash_reserve", "cash reserve leaves no room")) == "BULL"   # its regime_label
    rep = gcp.build_report(fixture_log())
    legacy = group(rep, "exposure_ceiling", "1", "dropped")["by_regime"]
    assert {k: v["name_days"] for k, v in legacy.items()} == {"BULL": 5, "CAUTION": 2}
    # BULL: XAAA-A (-4), XBBB-A (+2), XAAA-C (+2, inside XAAA-A's trade); CAUTION: XCCC-A (+2), XBBB-C (+1)
    assert legacy["BULL"]["+5d"] == {"labeled": 3, "positions": 2, "position_mean_return_pct": -1.0, "sized_positions": 0,
                                     "cut_dollars": 0.0, "prevention_pnl_dollars": 0.0, "net_prevention_pnl_dollars": 0.0}
    assert (legacy["CAUTION"]["+5d"]["labeled"], legacy["CAUTION"]["+5d"]["positions"],
            legacy["CAUTION"]["+5d"]["position_mean_return_pct"]) == (2, 2, 1.5)
    daily = group(rep, "daily_loss", "1", "dropped")["by_regime"]
    assert list(daily) == ["NEUTRAL"]
    assert (daily["NEUTRAL"]["+5d"]["cut_dollars"], daily["NEUTRAL"]["+5d"]["prevention_pnl_dollars"],
            daily["NEUTRAL"]["+5d"]["net_prevention_pnl_dollars"]) == (16000.0, 160.0, 80.0)
    assert list(group(rep, "exposure_ceiling", "1", "scaled")["by_regime"]) == ["BULL"]          # from "(regime=BULL)"


def test_report_outcome_distribution_and_same_period_base_rates_are_per_first_seen_name_day():
    cell = group(gcp.build_report(fixture_log()), "exposure_ceiling", "1", "dropped")["+5d"]
    # XAAA-A (-4, the 14:00 record, not the 15:00 copy), XBBB-A (+2), XCCC-A (+2), XAAA-C (+2), XBBB-C (+1)
    assert (cell["labeled"], cell["wins"], cell["losses"], cell["win_rate_pct"]) == (5, 3, 2, 60.0)
    assert (cell["pending"], cell["unlabelable"]) == (1, 1)
    assert cell["return_pct"] == {"mean": 0.6, "median": 2.0, "p10": -4.0, "p90": 2.0, "min": -4.0, "max": 2.0}
    assert cell["exits"] == {"stop": 1, "target": 3, "timeout": 1}
    # the scanner's taken candidates first seen on the SAME ET days (A and C): ZTA ZTB ZTC ZTE ZTF
    assert cell["base_taken_same_days"] == {"n": 5, "win_rate_pct": 60.0, "mean_return_pct": 0.6, "median_return_pct": 2.0}
    # over the same SPAN (A..C) ZTD on day B joins
    assert cell["base_taken_same_span"] == {"n": 6, "win_rate_pct": 50.0, "mean_return_pct": -0.167, "median_return_pct": 1.5}
    # every other logged candidate on the same days: the five taken plus ZRA and ZRB
    assert cell["base_scanned_same_days"] == {"n": 7, "win_rate_pct": 57.1, "mean_return_pct": 0.143, "median_return_pct": 2.0}
    assert cell["mean_minus_taken_same_days_pct"] == 0.0
    other = group(gcp.build_report(fixture_log()), "exposure_ceiling", "1", "dropped")["+20d"]
    assert (other["labeled"], other["pending"], other["unlabelable"]) == (0, 7, 0)


def test_report_dollars_count_one_position_per_name_at_a_time_and_only_sized_records():
    rep = gcp.build_report(fixture_log())
    legacy = group(rep, "exposure_ceiling", "1", "dropped")["+5d"]["dollars"]
    # XAAA-C sits inside XAAA-A's trade (exit 10-07): 4 positions out of 5 labeled name-days; none carries a size
    assert (legacy["positions"], legacy["sized_positions"], legacy["unsized_positions"]) == (4, 0, 4)
    assert (legacy["cut_dollars"], legacy["prevention_pnl_dollars"]) == (0.0, 0.0)
    assert str(legacy["prevention_pnl_dollars"]) == "0.0"                         # not "-0.0"
    assert legacy["position_mean_return_pct"] == 0.25                             # (-4 + 2 + 2 + 1) / 4: $2.50 per $1,000

    t1 = group(rep, "daily_loss", "1", "dropped")["+5d"]
    assert t1["labeled"] == 2 and "win_rate_pct" not in t1                        # below min_n: counts only
    assert t1["dollars"] == {
        "estimate": True, "positions": 2, "sized_positions": 2, "unsized_positions": 0, "position_mean_return_pct": -1.0,
        "cut_dollars": 16000.0, "would_have_made_dollars": -160.0, "prevention_pnl_dollars": 160.0,   # 8000 x +2% and x -4%
        "net_cut_dollars": 8000.0, "net_would_have_made_dollars": -80.0, "net_prevention_pnl_dollars": 80.0}
    t2 = group(rep, "daily_loss", "2", "dropped")["+5d"]["dollars"]
    assert (t2["cut_dollars"], t2["would_have_made_dollars"], t2["prevention_pnl_dollars"]) == (8000.0, 160.0, -160.0)
    t3 = group(rep, "daily_loss", "3", "dropped")["+5d"]["dollars"]
    assert (t3["cut_dollars"], t3["would_have_made_dollars"], t3["net_prevention_pnl_dollars"]) == (20000.0, 200.0, -100.0)
    assert rep["gates"]["daily_loss"]["dollars_all_tiers"]["+5d"] == {
        "estimate": True, "positions": 4, "sized_positions": 4, "unsized_positions": 0,
        "cut_dollars": 44000.0, "would_have_made_dollars": 200.0, "prevention_pnl_dollars": -200.0,   # the gate blocked a gain
        "net_cut_dollars": 22000.0, "net_would_have_made_dollars": 100.0, "net_prevention_pnl_dollars": -100.0}
    ceiling = rep["gates"]["exposure_ceiling"]["dollars_all_tiers"]["+5d"]
    assert (ceiling["positions"], ceiling["sized_positions"], ceiling["unsized_positions"]) == (5, 1, 4)
    assert (ceiling["cut_dollars"], ceiling["prevention_pnl_dollars"], ceiling["net_cut_dollars"]) == (500.0, 20.0, 0.0)
    assert rep["dollar_basis"].startswith("ESTIMATE")


def test_report_is_aggregate_only_fits_the_diag_sanitizer_and_does_not_modify_records():
    log = fixture_log()
    before = json.dumps(log, sort_keys=True)
    rep = gcp.build_report(log)
    assert json.dumps(log, sort_keys=True) == before
    text = json.dumps(rep)
    for ticker in {r["ticker"] for r in log}:
        assert ticker not in text                       # no name leaves the probe
    assert "entry_price" not in text and "timestamp" not in text

    def depth(v, d=0):          # server/diag.ts sanitizeDiag replaces anything deeper than 8 with "[depth]"
        if isinstance(v, dict):
            return max([depth(x, d + 1) for x in v.values()] or [d])
        if isinstance(v, list):
            return max([depth(x, d + 1) for x in v] or [d])
        return d

    assert depth(rep) <= 8
    assert all(len(s) < 2000 for s in rep["limits"] + [rep["dollar_basis"]] + list(rep["reading"].values()))


def test_probe_reads_the_log_read_only_and_reports_a_failure_instead_of_raising():
    assert gcp.gate_counterfactuals_report()["gates"]["exposure_ceiling"]["rows"] == 0     # no log yet
    with open(sp.SHADOW_LOG_PATH, "w") as f:
        json.dump(fixture_log(), f)
    folder = os.path.dirname(sp.SHADOW_LOG_PATH)
    raw, files = open(sp.SHADOW_LOG_PATH, "rb").read(), sorted(os.listdir(folder))
    rep = gcp.gate_counterfactuals_report()
    assert rep["gates"]["exposure_ceiling"]["rows"] == 9 and "elapsed_s" in rep
    assert open(sp.SHADOW_LOG_PATH, "rb").read() == raw and sorted(os.listdir(folder)) == files   # nothing written
    assert sp.shadow_log_cache_stats()["leases"] == 0                                  # a read-only load
    json.dumps(rep)                                                                    # the RPC result is JSON
    with open(sp.SHADOW_LOG_PATH, "w") as f:
        f.write("{not json")
    assert gcp.gate_counterfactuals_report()["log"]["records"] == 0                    # a corrupt log reads as empty
    with patch.object(sp, "_load_shadow_log", side_effect=RuntimeError("volume gone")):
        assert gcp.gate_counterfactuals_report() == {"error": "RuntimeError: volume gone"}


def test_probe_module_imports_neither_bot_engine_nor_the_tier_engine():
    code = ("import sys, gate_counterfactuals_probe; "
            "print(sorted(m for m in ('bot_engine', 'tiered_strategy', 'numpy', 'pandas') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_daemon_route_and_the_subprocess_fallback_resolve_the_same_entry_point():
    import inspect
    from voltrade_daemon import RPCDispatcher
    d = RPCDispatcher()
    assert d._routes["gate_counterfactuals"] == ("gate_counterfactuals_probe", "gate_counterfactuals_report")
    inspect.signature(gcp.gate_counterfactuals_report).bind()            # bot.ts sends no args
    reply = d.dispatch("gate_counterfactuals", {})
    assert reply["status"] == "ok" and set(gcp.GATE_ORDER[:4]) <= set(reply["result"]["gates"])
    bot = open(os.path.join(ROOT, "server", "bot.ts")).read()
    fallback = ('python3 -c "import json; from gate_counterfactuals_probe import gate_counterfactuals_report; '
                'print(json.dumps(gate_counterfactuals_report(), default=str))"')
    assert fallback in bot
    out = subprocess.run(["bash", "-c", fallback], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "gates" in json.loads(out.stdout) or "error" in json.loads(out.stdout)


# ══ 5. MEASUREMENT INTEGRITY: the summary that already existed ═══════════════

def test_existing_masterkill_summary_reads_pre_change_records_exactly_as_before():
    legacy = [r for r in fixture_log() if r["decision"] == "rejected_masterkill" and "gate" not in r]
    assert sp._win_rate_by_masterkill_reason(legacy) == {
        "exposure_ceiling": {"candidate_count": 8, "win_rate": {"+5d": {"win_rate": 50.0, "n": 6}}}}


def test_existing_masterkill_summary_counts_the_new_rows_too_which_is_why_the_probe_splits_them():
    log = [r for r in fixture_log() if r["decision"] == "rejected_masterkill"]
    flat = sp._win_rate_by_masterkill_reason(log)
    # the flat bucket now also holds the scaled row (a name that WAS still proposed) under exposure_ceiling,
    # and the tier-2 row that repeats its tier-1 row's outcome under daily_loss
    assert flat["exposure_ceiling"]["candidate_count"] == 9
    assert flat["daily_loss"]["candidate_count"] == 4
    rep = gcp.build_report(log)
    assert [(x["tier"], x["action"], x["rows"]) for x in rep["gates"]["exposure_ceiling"]["groups"]] == [
        ("1", "dropped", 8), ("1", "scaled", 1)]
