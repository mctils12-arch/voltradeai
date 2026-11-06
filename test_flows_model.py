#!/usr/bin/env python3
"""
The institutional flow model (2026-11-06; flows_model.py, its floor and
satellite hooks in bot_engine.py, the plan() multiplier in floor_vol_managed.py).

Pins:
- the rebalancing math: the paper's own worked example (stocks +10%, bonds
  flat -> 66/106 - 0.60 = 2.26 points), resets, the threshold average;
- every pressure is in [-1, 1]; the floor multiplier is in
  [1 - FLOWS_FLOOR_MAX_CUT, 1.0] and NEVER above 1.0 (no input raises the
  floor's target or adds leverage); a stale/missing/disabled estimate is 1.0;
- the satellite tilt is bounded and capped at the sizer's per-position max;
- refresh_daily writes the file and one history line, is cached, never raises;
- the floor pass applies and logs the multiplier (result, state file), the
  reduce-only pass still never buys, and the scan carries the sector tilt.

Run: python3 -m pytest -q test_flows_model.py
"""
import json
import math
import os
import random
import sys
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import calendar_effects as ce
import floor_vol_managed as fvm
import flows_model as fm
import system_config

D = date


# ── synthetic history on real sessions ──────────────────────────────────────

def _sessions(start, end):
    out, d = [], start
    while d <= end:
        if ce.is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _walk(days, seed, drift=0.0003, vol=0.01, start=100.0, shock=None):
    rng = random.Random(seed)
    px, out = start, []
    for d in days:
        r = drift + vol * rng.gauss(0, 1)
        if shock and shock[0] <= d <= shock[1]:
            r += shock[2]
        px *= 1 + r
        out.append((d.isoformat(), round(px, 4)))
    return out


def _book(end, seed=1, spy_shock=None, sector_drift=None):
    days = _sessions(end - timedelta(days=700), end)
    closes = {"SPY": _walk(days, seed, shock=spy_shock), "IEF": _walk(days, seed + 1, drift=0.0001, vol=0.003)}
    for k, (_, etf) in enumerate(fm.SECTORS):
        closes[etf] = _walk(days, seed + 10 + k, drift=(sector_drift or {}).get(etf, 0.0003), vol=0.013)
    return closes


# ── the rebalancing math ─────────────────────────────────────────────────────

def test_drift_is_the_papers_worked_example():
    out = fm.drift_path([100.0, 110.0], [100.0, 100.0], [False, False])
    assert out[0] == 0.0
    assert math.isclose(out[1], 66 / 106 - 0.60, rel_tol=1e-12)
    assert round(out[1] * 100, 2) == 2.26


def test_a_reset_restarts_the_drift_at_target():
    eq, bd = [100.0, 110.0, 110.0, 121.0], [100.0] * 4
    no_reset = fm.drift_path(eq, bd, [False] * 4)
    reset = fm.drift_path(eq, bd, [False, True, False, False])
    assert math.isclose(reset[1], no_reset[1]), "the signal is the PRE-reset distance"
    assert reset[2] == 0.0, "flat day after a reset: back at target"
    assert math.isclose(reset[3], 66 / 106 - 0.60), "drifts again from 60%"
    assert no_reset[3] > no_reset[1]


def test_threshold_signal_averages_the_bands():
    eq, bd = [100.0, 110.0, 121.0], [100.0] * 3
    daily = fm.threshold_path(eq, bd, deltas=(0.0,))
    assert math.isclose(daily[1], 66 / 106 - 0.6) and math.isclose(daily[2], 66 / 106 - 0.6), \
        "delta 0 resets every day: each day's own drift"
    never = fm.threshold_path(eq, bd, deltas=(1.0,))
    assert never == fm.drift_path(eq, bd, [False] * 3), "an unreachable band never resets"
    both = fm.threshold_path(eq, bd, deltas=(0.0, 1.0))
    assert math.isclose(both[2], (daily[2] + never[2]) / 2)
    assert fm.THRESHOLD_DELTAS[0] == 0.0 and fm.THRESHOLD_DELTAS[-1] == 0.02 and len(fm.THRESHOLD_DELTAS) == 9


def test_month_end_flags_and_signal():
    days = [D(2026, 10, 29), D(2026, 10, 30), D(2026, 11, 2), D(2026, 11, 30)]
    me, qe = fm.month_end_flags(days, last_is_month_end=True)
    assert me == [False, True, False, True] and qe == [False, False, False, False]
    me, qe = fm.month_end_flags([D(2026, 9, 29), D(2026, 9, 30), D(2026, 10, 1)], last_is_month_end=False)
    assert me == [False, True, False] and qe == [False, True, False]
    assert fm.rebalance_signal(0.01, 0.03, False) == 0.01
    assert fm.rebalance_signal(0.01, 0.03, True) == 0.04


def test_tax_loss_scores_only_losers():
    ytd = {"A": -0.20, "B": -0.10, "C": -0.05, "D": -0.01, "E": 0.30}
    assert fm.tax_loss_scores(ytd, selling=True) == {"A": -1.0, "B": -2 / 3, "C": -1 / 3, "D": 0.0, "E": 0.0}
    assert fm.tax_loss_scores(ytd, selling=False)["A"] == 1.0
    up = {"A": 0.01, "B": 0.05, "C": None}
    assert fm.tax_loss_scores(up, selling=True) == {"A": 0.0, "B": 0.0, "C": 0.0}, "no loss to harvest"


def test_rotation_score_bounds():
    rs = fm.rotation_score({"z126": 4.0, "z21": 3.0})
    assert rs["momentum"] == 1.0 and rs["crowding"] == 0.5
    assert math.isclose(rs["rotation"], 1.0 - 0.25 * 0.5)
    assert fm.rotation_score({"z126": -9.0, "z21": -9.0})["rotation"] == -1.0
    assert fm.rotation_score(None) is None


# ── the estimate ─────────────────────────────────────────────────────────────

CONTRACT_EQUITY = {"pressure", "direction", "confidence", "drivers"}
CONTRACT_SECTOR = {"sector", "etf", "pressure", "direction", "drivers"}


def test_estimate_shape_and_bounds_over_many_dates():
    closes = _book(D(2026, 11, 5), seed=3)
    seen_dirs = set()
    for k in range(0, 120, 7):
        as_of = ce.add_sessions(D(2026, 11, 5), -k)
        doc = fm.compute_flows(closes, as_of=as_of)
        assert doc["ok"] and doc["label"] == "ESTIMATE" and doc["asOf"] == as_of.isoformat()
        assert set(doc) >= {"asOf", "horizonDays", "equity", "sectors", "calendar", "notes"}
        e = doc["equity"]
        assert set(e) >= CONTRACT_EQUITY and -1.0 <= e["pressure"] <= 1.0 and 0.0 <= e["confidence"] <= 1.0
        assert e["direction"] == fm.direction_of(e["pressure"])
        seen_dirs.add(e["direction"])
        for d in e["drivers"]:
            assert set(d) >= {"name", "value", "reason"} and isinstance(d["reason"], str) and d["reason"]
            assert -1.0 <= d["value"] <= 1.0
        assert len(doc["sectors"]) == len(fm.SECTORS)
        for s in doc["sectors"]:
            assert set(s) >= CONTRACT_SECTOR and -1.0 <= s["pressure"] <= 1.0
        for c in doc["calendar"]:
            assert set(c) >= {"date", "event", "effect"}
        assert all(isinstance(n, str) for n in doc["notes"])
        m = fm.floor_multiplier(doc, today=as_of)["multiplier"]
        assert 0.9 <= m <= 1.0
    assert len(seen_dirs) >= 2, "the synthetic book should move the estimate around"


def test_stocks_beating_bonds_into_month_end_reads_as_selling():
    # SPY ~+8% in the first half of November 2026, bonds flat: the 60/40 drift
    # is well over a point overweight stocks with the month end 4 sessions out.
    shock = (D(2026, 11, 2), D(2026, 11, 13), 0.008)
    closes = _book(D(2026, 11, 20), seed=5, spy_shock=shock)
    doc = fm.compute_flows(closes, as_of=D(2026, 11, 20))
    reb = next(d for d in doc["equity"]["drivers"] if d["key"] == "rebalance")
    assert doc["signals"]["month_drift"] > 0.01
    assert reb["value"] < -0.3 and "sell stocks" in reb["reason"] and "2026-11-30" in reb["reason"]
    assert doc["equity"]["pressure"] < -0.15 and doc["equity"]["direction"] == "sell"
    fl = fm.floor_multiplier(doc, today=D(2026, 11, 20))
    assert fl["applied"] and 0.9 <= fl["multiplier"] < 1.0 and f"x{fl['multiplier']:.2f}" in fl["reason"]
    # and the reverse: bonds beat stocks -> buying, and the floor is NOT raised
    closes = _book(D(2026, 11, 20), seed=5, spy_shock=(D(2026, 11, 2), D(2026, 11, 13), -0.008))
    doc = fm.compute_flows(closes, as_of=D(2026, 11, 20))
    assert doc["equity"]["pressure"] > 0
    assert fm.floor_multiplier(doc, today=D(2026, 11, 20))["multiplier"] == 1.0


def test_missing_inputs_degrade_never_fabricate():
    closes = _book(D(2026, 11, 5), seed=7)
    no_bond = {k: v for k, v in closes.items() if k != "IEF"}
    doc = fm.compute_flows(no_bond, as_of=D(2026, 11, 5))
    assert doc["ok"] and "IEF" in doc["inputs"]["missing"]
    reb = next(d for d in doc["equity"]["drivers"] if d["key"] == "rebalance")
    assert reb["value"] == 0.0 and reb["reason"].startswith("Not computed")
    assert doc["signals"]["month_drift"] is None
    bad = fm.compute_flows({"SPY": closes["SPY"][:10]})
    assert bad["ok"] is False and "need SPY" in bad["error"]


def test_tax_window_weights_follow_the_evidence():
    # October 2026 with three sectors down on the year: the selling driver is a quarter weight
    end = D(2026, 10, 14)
    drifts = {"XLE": -0.0015, "XLU": -0.0012, "XLP": -0.001}
    doc = fm.compute_flows(_book(end, seed=11, sector_drift=drifts), as_of=end)
    taxed = {s["etf"]: next(d for d in s["drivers"] if d["key"] == "tax_loss")
             for s in doc["sectors"] if any(d["key"] == "tax_loss" for d in s["drivers"])}
    assert taxed, "the YTD-worst sectors with losses carry the tax-loss driver in October"
    for d in taxed.values():
        assert d["value"] < 0 and d["weight"] == pytest.approx(0.3 * 0.25)
        assert "opposite sign" in d["reason"]
    dec = D(2026, 12, 10)
    doc = fm.compute_flows(_book(dec, seed=11, sector_drift=drifts), as_of=dec)
    full = [d for s in doc["sectors"] for d in s["drivers"] if d["key"] == "tax_loss"]
    assert full and all(d["weight"] == pytest.approx(0.3) for d in full)


# ── the hooks ────────────────────────────────────────────────────────────────

def _doc(p, as_of="2026-11-20", sectors=None):
    return {"ok": True, "asOf": as_of, "equity": {"pressure": p, "drivers": [
        {"key": "rebalance", "name": "x", "value": p, "weight": 1.0, "reason": "because"}]},
            "sectors": sectors or []}


def test_floor_multiplier_never_above_one_and_bounded():
    today = D(2026, 11, 20)
    for p in [x / 20.0 for x in range(-20, 21)]:
        m = fm.floor_multiplier(_doc(p), today=today)["multiplier"]
        assert 0.9 <= m <= 1.0
        if p >= -0.10:
            assert m == 1.0, "inside the deadband (or buying): no change"
    assert fm.floor_multiplier(_doc(-1.0), today=today)["multiplier"] == 0.9
    assert fm.floor_multiplier(_doc(-1.0), {"FLOWS_FLOOR_MAX_CUT": 0.2}, today=today)["multiplier"] == 0.8
    assert fm.floor_multiplier(_doc(-1.0), {"FLOWS_FLOOR_MAX_CUT": 5.0}, today=today)["multiplier"] == 0.5, \
        "a mistyped cut is clipped to half the floor, never more"
    assert fm.floor_multiplier(_doc(-1.0), {"FLOWS_FLOOR_MAX_CUT": -1.0}, today=today)["multiplier"] == 1.0


def test_floor_multiplier_ignores_stale_missing_or_disabled():
    for doc, cfg, why in ((_doc(-1.0, as_of="2026-11-10"), None, "days old"),
                          (None, None, "no flows estimate"),
                          ({"ok": False}, None, "no flows estimate"),
                          (_doc(-1.0), {"FLOWS_FLOOR_ENABLED": False}, "off"),
                          (_doc(-1.0), {"FLOWS_ENABLED": False}, "off")):
        r = fm.floor_multiplier(doc, cfg, today=D(2026, 11, 20))
        assert r["multiplier"] == 1.0 and not r["applied"] and why in r["reason"], (doc, cfg, r)


def test_plan_multiplier_cuts_and_never_raises():
    px, calm = 500.0, [500.0]
    for i in range(20):
        px *= 1 + (0.004 if i % 2 == 0 else -0.004)
        calm.append(px)
    cfg = {"FLOOR_VM_WINDOW": 20, "FLOOR_VM_TARGET_VOL": 0.16, "FLOOR_VM_CAP": 1.5,
           "FLOOR_VM_BAND": 0.10, "FLOOR_VM_CAPITAL_PCT": 0.85}
    rule = fvm.plan(calm, 100_000, 0, 0, 0, cfg)
    assert rule["target_exposure"] == 1.5 and rule["rule_target_exposure"] == 1.5 and rule["exposure_multiplier"] == 1.0
    cut = fvm.plan(calm, 100_000, 0, 0, 0, cfg, exposure_multiplier=0.9)
    assert cut["rule_target_exposure"] == 1.5 and cut["target_exposure"] == 1.35
    h = fvm.holdings_for(1.35, 85_000)
    assert cut["target_base_dollars"] == round(h["base"], 2) and cut["target_levered_dollars"] == round(h["levered"], 2)
    assert cut["target_levered_dollars"] < rule["target_levered_dollars"], "less SSO: leverage only falls"
    for bad in (1.5, 2.0, float("nan"), float("inf"), "x", None):
        p = fvm.plan(calm, 100_000, 0, 0, 0, cfg, exposure_multiplier=bad)
        assert p["target_exposure"] == 1.5 and p["exposure_multiplier"] == 1.0, bad
    assert fvm.plan(calm, 100_000, 0, 0, 0, cfg, exposure_multiplier=-3)["target_exposure"] == 0.0
    short = fvm.plan(calm[:5], 100_000, 0, 0, 0, cfg, exposure_multiplier=0.9)
    assert not short["ok"] and short["exposure_multiplier"] == 0.9 and not short["rebalance"]


def test_sector_adjustment_bounds_and_aliases():
    secs = [{"sector": "Technology", "etf": "XLK", "pressure": -1.0, "direction": "sell", "drivers": []},
            {"sector": "Financials", "etf": "XLF", "pressure": 1.0, "direction": "buy", "drivers": []},
            {"sector": "Semiconductors", "etf": "SMH", "pressure": 0.5, "direction": "buy", "drivers": []}]
    doc = _doc(0.0, sectors=secs)
    today = D(2026, 11, 20)
    assert fm.sector_size_adjustment(doc, "Technology", "MSFT", today=today)["multiplier"] == 0.9
    fin = fm.sector_size_adjustment(doc, "Financial Services", "JPM", today=today)
    assert fin["multiplier"] == 1.1 and fin["sector"] == "Financials" and fin["etf"] == "XLF"
    semi = fm.sector_size_adjustment(doc, "Technology", "NVDA", today=today)
    assert semi["sector"] == "Semiconductors" and semi["multiplier"] == 1.05
    for sector, cfg in (("Martian Mining", None), ("Technology", {"FLOWS_SECTOR_SIZING_ENABLED": False})):
        r = fm.sector_size_adjustment(doc, sector, "ZZZ", cfg, today=today)
        assert r["multiplier"] == 1.0 and not r["applied"]
    assert fm.sector_size_adjustment(_doc(0.0, as_of="2026-10-01", sectors=secs), "Technology", "MSFT",
                                     today=today)["multiplier"] == 1.0, "stale estimate"


def test_sized_shares_caps_the_raise():
    assert fm.sized_shares(100, 50.0, 100_000, 0.9, 0.08) == 90
    assert fm.sized_shares(100, 50.0, 100_000, 1.1, 0.08) == 110
    assert fm.sized_shares(150, 50.0, 100_000, 1.1, 0.08) == 160, "capped at 8% of equity = 160 shares"
    assert fm.sized_shares(100, 50.0, 60_000, 1.1, 0.08) == 100, "a raise never goes below the sizer's number"
    assert fm.sized_shares(7, 50.0, 100_000, 0.9, 0.08) == 6
    assert fm.sized_shares(0, 50.0, 100_000, 1.1, 0.08) == 0


# ── I/O ──────────────────────────────────────────────────────────────────────

def test_last_completed_session_and_completed_only():
    ny = datetime
    assert fm.last_completed_session(ny(2026, 11, 5, 15, 0)) == D(2026, 11, 4)
    assert fm.last_completed_session(ny(2026, 11, 5, 17, 0)) == D(2026, 11, 5)
    assert fm.last_completed_session(ny(2026, 11, 7, 12, 0)) == D(2026, 11, 6), "Saturday"
    assert fm.last_completed_session(ny(2026, 11, 26, 18, 0)) == D(2026, 11, 25), "Thanksgiving"
    rows = {"SPY": [("2026-11-04", 1.0), ("2026-11-05", 2.0)]}
    assert fm.completed_only(rows, D(2026, 11, 4)) == {"SPY": [("2026-11-04", 1.0)]}


def test_refresh_writes_once_per_session_and_never_raises(tmp_path):
    closes = _book(D(2026, 11, 5), seed=13)
    calls = []

    def fetch(symbols, days):
        calls.append(list(symbols))
        return {s: closes[s] for s in symbols if s in closes}

    now = datetime(2026, 11, 5, 18, 0)
    r = fm.refresh_daily(data_dir=str(tmp_path), fetch=fetch, now_et=now)
    assert r["status"] == "computed" and r["asOf"] == "2026-11-05" and len(calls) == 1
    assert set(calls[0]) == set(fm.universe(fm._cfg()))
    doc = json.loads((tmp_path / fm.FLOWS_FILE).read_text())
    assert doc["ok"] and doc["floor"]["multiplier"] <= 1.0 and doc["config"]["FLOWS_FLOOR_MAX_CUT"] == 0.10
    hist = (tmp_path / fm.FLOWS_HISTORY_FILE).read_text().strip().splitlines()
    assert len(hist) == 1
    row = json.loads(hist[0])
    assert row["asOf"] == "2026-11-05" and set(row["sectors"]) == {e for _, e in fm.SECTORS}
    assert set(row["equity"]["drivers"]) >= {"rebalance", "threshold", "month_end"}
    again = fm.refresh_daily(data_dir=str(tmp_path), fetch=fetch, now_et=now)
    assert again["status"] == "cached" and len(calls) == 1, "same session: one file read, no fetch"

    def boom(symbols, days):
        raise ConnectionError("alpaca down")

    bad = fm.refresh_daily(force=True, data_dir=str(tmp_path), fetch=boom, now_et=now)
    assert bad["status"] == "error" and "ConnectionError" in bad["reason"]
    assert json.loads((tmp_path / fm.FLOWS_FILE).read_text())["asOf"] == "2026-11-05", "last good file kept"
    assert fm.report(str(tmp_path))["asOf"] == "2026-11-05"
    assert fm.report(str(tmp_path / "nowhere"))["ok"] is False


def test_refresh_waits_for_the_new_sessions_bars(tmp_path):
    """Alpaca serves the 16:00 ET session only after UTC midnight: a recompute
    that sees no new session rewrites nothing and appends no history."""
    closes = _book(D(2026, 11, 4), seed=17)
    fetch = lambda symbols, days: {s: closes[s] for s in symbols if s in closes}
    first = fm.refresh_daily(data_dir=str(tmp_path), fetch=fetch, now_et=datetime(2026, 11, 4, 21, 0))
    assert first["status"] == "computed" and first["asOf"] == "2026-11-04"
    before = (tmp_path / fm.FLOWS_FILE).read_text()
    stamped = json.loads(before)
    stamped["computedAt"] = "2026-11-04T01:00:00Z"            # older than the recency guard
    (tmp_path / fm.FLOWS_FILE).write_text(json.dumps(stamped))
    r = fm.refresh_daily(data_dir=str(tmp_path), fetch=fetch, now_et=datetime(2026, 11, 5, 17, 0))
    assert r["status"] == "no_new_session" and "2026-11-05 is not in yet" in r["reason"]
    assert json.loads((tmp_path / fm.FLOWS_FILE).read_text())["computedAt"] == "2026-11-04T01:00:00Z"
    assert len((tmp_path / fm.FLOWS_HISTORY_FILE).read_text().strip().splitlines()) == 1


def test_refresh_respects_the_master_switch(tmp_path):
    with patch.dict(system_config.BASE_CONFIG, {"FLOWS_ENABLED": False}):
        r = fm.refresh_daily(data_dir=str(tmp_path), fetch=lambda s, d: {})
    assert r["status"] == "off" and not (tmp_path / fm.FLOWS_FILE).exists()


# ── wiring ───────────────────────────────────────────────────────────────────

def test_config_switches_and_defaults_agree():
    c = system_config.BASE_CONFIG
    for key in ("FLOWS_ENABLED", "FLOWS_FLOOR_ENABLED", "FLOWS_SECTOR_SIZING_ENABLED"):
        assert c[key] is True, f"{key}: ON by default (human directive 2026-11-05)"
    assert (c["FLOWS_FLOOR_MAX_CUT"], c["FLOWS_FLOOR_DEADBAND"], c["FLOWS_SECTOR_SIZE_K"]) == (0.10, 0.10, 0.10)
    cfg_keys = {k for k in c if k.startswith("FLOWS_")}
    assert cfg_keys == set(fm.DEFAULTS), "every flows tunable lives in system_config; DEFAULTS mirrors them all"
    for key in cfg_keys:
        assert fm.DEFAULTS[key] == c[key], f"flows_model.DEFAULTS and system_config disagree on {key}"
    assert fm._cfg() == {k: c[k] for k in cfg_keys}, "production reads every value from system_config"


def test_daemon_registers_the_entry_points():
    import voltrade_daemon
    routes = voltrade_daemon.RPCDispatcher()._routes
    assert routes["flows_refresh"] == ("flows_model", "refresh_daily")
    assert routes["flows_report"] == ("flows_model", "report")


def _bot_engine():
    import bot_engine
    return bot_engine


def test_floor_pass_applies_and_logs_the_multiplier():
    be = _bot_engine()
    px, calm = 500.0, [500.0]
    for i in range(20):
        px *= 1 + (0.004 if i % 2 == 0 else -0.004)
        calm.append(px)
    orders, saved = [], []
    quotes = {"SPY": (499.9, 500.1), "SSO": (99.95, 100.05)}
    adj = {"multiplier": 0.9, "applied": True, "reason": "Expected net equity selling: floor target x0.90."}
    with patch.object(be, "_floor_flows_adjustment", return_value=adj), \
         patch.object(be, "_daily_closes_completed", return_value=calm), \
         patch.object(be, "_floor_vm_broker_clock", return_value=(True, "open")), \
         patch.object(be, "get_alpaca_account", return_value={"equity": "100000"}), \
         patch.object(be, "get_alpaca_positions", return_value=[]), \
         patch.object(be, "_latest_quote", side_effect=lambda s: quotes[s]), \
         patch.object(be, "_open_order_symbols", return_value=set()), \
         patch.object(be, "_floor_vm_order", side_effect=lambda s, q, side, px_: orders.append((s, q, side)) or
                      {"ok": True, "order_id": "o", "error": None}), \
         patch.object(be, "_unwind_covered_calls", return_value=[]), \
         patch.object(be, "_floor_vm_save_state", side_effect=saved.append), \
         patch.object(be, "_is_regular_hours", return_value=True), \
         patch.object(be, "_et_now_hour", return_value=11.0):
        r = be._manage_vol_managed_floor({})
    assert r["rule_target_exposure"] == 1.5 and r["target_exposure"] == 1.35 and r["flows"] == adj
    h = fvm.holdings_for(1.35, 85_000)
    assert sorted(orders) == [("SPY", int(h["base"] // 500.1), "buy"), ("SSO", int(h["levered"] // 100.05), "buy")]
    assert saved and saved[0]["flows"] == adj and saved[0]["exposure_multiplier"] == 0.9


def test_reduce_only_with_a_flows_cut_still_never_buys():
    be = _bot_engine()
    px, calm = 500.0, [500.0]
    for i in range(20):
        px *= 1 + (0.004 if i % 2 == 0 else -0.004)
        calm.append(px)
    held = fvm.holdings_for(1.5, 85_000)
    pos = [{"symbol": "SPY", "qty": str(held["base"] / 500), "qty_available": str(held["base"] / 500),
            "market_value": str(held["base"])},
           {"symbol": "SSO", "qty": str(held["levered"] / 100), "qty_available": str(held["levered"] / 100),
            "market_value": str(held["levered"])}]
    orders = []
    with patch.object(be, "_floor_flows_adjustment", return_value={"multiplier": 0.9, "applied": True, "reason": "r"}), \
         patch.object(be, "_daily_closes_completed", return_value=calm), \
         patch.object(be, "_floor_vm_broker_clock", return_value=(True, "open")), \
         patch.object(be, "get_alpaca_account", return_value={"equity": "100000"}), \
         patch.object(be, "get_alpaca_positions", return_value=pos), \
         patch.object(be, "_latest_quote", side_effect=lambda s: {"SPY": (499.9, 500.1), "SSO": (99.95, 100.05)}[s]), \
         patch.object(be, "_open_order_symbols", return_value=set()), \
         patch.object(be, "_floor_vm_order", side_effect=lambda s, q, side, px_: orders.append((s, q, side)) or
                      {"ok": True, "order_id": "o", "error": None}), \
         patch.object(be, "_unwind_covered_calls", return_value=[]), \
         patch.object(be, "_floor_vm_save_state"), \
         patch.object(be, "_is_regular_hours", return_value=True), \
         patch.object(be, "_et_now_hour", return_value=11.0):
        r = be._manage_vol_managed_floor({}, reduce_only=True)
    assert r["target_exposure"] == 1.35 and r["rebalance"] is True
    assert orders and all(side == "sell" for _, _, side in orders), orders
    assert [s for s, _, _ in orders][0] == "SSO", "de-lever first"


def test_floor_adjustment_reads_the_cached_file(tmp_path):
    be = _bot_engine()
    today = datetime.now(be.ZoneInfo("America/New_York")).date()
    doc = _doc(-1.0, as_of=today.isoformat())
    doc["ok"] = True
    (tmp_path / fm.FLOWS_FILE).write_text(json.dumps(doc))
    with patch.object(be, "DATA_DIR", str(tmp_path)):
        be._FLOWS_DOC_CACHE.update(mtime=None, doc=None)
        r = be._floor_flows_adjustment(system_config.BASE_CONFIG)
        assert r["multiplier"] == 0.9 and r["applied"]
        (tmp_path / fm.FLOWS_FILE).write_text("{not json")
        os.utime(tmp_path / fm.FLOWS_FILE, (1, 1))
        r = be._floor_flows_adjustment(system_config.BASE_CONFIG)
        assert r["multiplier"] == 1.0, "an unreadable estimate is no adjustment, never an error"
        os.remove(tmp_path / fm.FLOWS_FILE)
        assert be._floor_flows_adjustment(system_config.BASE_CONFIG)["multiplier"] == 1.0
    be._FLOWS_DOC_CACHE.update(mtime=None, doc=None)


def test_satellite_sizing_hook():
    be = _bot_engine()
    today = datetime.now(be.ZoneInfo("America/New_York")).date()
    doc = _doc(0.0, as_of=today.isoformat(), sectors=[
        {"sector": "Energy", "etf": "XLE", "pressure": -1.0, "direction": "sell", "drivers": []}])
    with patch.object(be, "_flows_doc", return_value=doc):
        r = be._flows_sector_sizing("XOM", None, 100, 100.0, 1_000_000)
    assert r["applied"] and r["shares"] == 90 and r["shares_before"] == 100 and r["position_value"] == 9000.0
    assert r["sector"] == "Energy" and r["etf"] == "XLE" and "x0.90" in r["reason"]
    with patch.object(be, "_flows_doc", side_effect=OSError("disk")):
        r = be._flows_sector_sizing("XOM", None, 100, 100.0, 1_000_000)
    assert not r["applied"] and r["shares"] == 100 and "flows unavailable" in r["reason"]


def test_scan_carries_the_sector_tilt():
    src = open(os.path.join(_ROOT, "bot_engine.py")).read()
    scan = src[src.index("def _scan_market_inner("):]
    scan = scan[:scan.index("\ndef ")]
    i_size = scan.index("sizing = calculate_position(")
    i_flow = scan.index("_flows_adj = _flows_sector_sizing(ticker, stock.get(\"sector\"), shares")
    i_zero = scan.index("if shares <= 0:")
    assert i_size < i_flow < i_zero, "after the sizer, before the zero-share check"
    assert '"flows": _flows_adj,' in scan, "every trade records its flow tilt"
    assert '"flows_sector": _flows_adj["multiplier"]' in scan
