"""
Position management (refocus spec section 4, 2026-11-06; position_mgmt.py):
the cash reserve, the drawdown risk budget and capital rotation, plus their
wiring into position_sizing, the options sizers, the tier engine, the scan and
the daemon. Offline: every file the module reads lives in tmp_path.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import position_mgmt as pm
import position_sizing as ps
import options_execution as oe
import instrument_selector as isel
import tiered_strategy as ts
from system_config import BASE_CONFIG

ROOT = os.path.dirname(os.path.abspath(__file__))
BOT_SRC = open(os.path.join(ROOT, "bot_engine.py")).read()
CFG = dict(BASE_CONFIG)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    import regime_engine
    monkeypatch.setattr(pm, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pm, "POSITION_MGMT_LOG_PATH", str(tmp_path / "pm.jsonl"))
    # no live regime risk budget unless a test sets one (works before and after
    # regime_engine.risk_budget_for exists)
    monkeypatch.setattr(regime_engine, "risk_budget_for", lambda *a, **k: None, raising=False)
    pm._dedup_seen.clear()
    yield


def rows(tmp_path):
    p = tmp_path / "pm.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def stock(sym, mv, qty=10, **kw):
    return {"symbol": sym, "qty": str(qty), "market_value": str(mv), "asset_class": "us_equity", "side": "long", **kw}


def opt(sym, qty, mv):
    return {"symbol": sym, "qty": str(qty), "market_value": str(mv), "asset_class": "us_option"}


# ── 1. Cash reserve ─────────────────────────────────────────────────────────

class TestReservePct:
    def test_config_per_regime_rises_risk_off(self):
        pcts = [pm.cash_reserve_pct(r, CFG) for r in ("BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC")]
        assert pcts == [0.05, 0.08, 0.15, 0.25, 0.35]
        assert pcts == sorted(pcts)

    def test_unknown_regime_uses_neutral_and_off_switch_is_zero(self):
        assert pm.cash_reserve_pct("SIDEWAYS", CFG) == 0.08
        assert pm.normalize_regime("bear") == ("BEAR", True)
        assert pm.normalize_regime(None) == ("NEUTRAL", False)
        assert pm.cash_reserve_pct("PANIC", {**CFG, "CASH_RESERVE_ENABLED": False}) == 0.0


class TestCommitments:
    def test_cash_secured_put_commits_its_collateral_not_its_premium(self):
        c = pm.option_commitments([opt("CMCSA261218P00024000", -1, -97)])
        assert c == {"CMCSA261218P00024000": 2400.0}

    def test_put_spread_commits_its_width(self):
        c = pm.option_commitments([opt("XYZ261218P00050000", -2, -300), opt("XYZ261218P00045000", 2, 100)])
        assert c["XYZ261218P00050000"] == pytest.approx(1000.0)     # (50 - 45) x 100 x 2
        assert c["XYZ261218P00045000"] == 0.0                         # its premium is inside the width

    def test_covered_call_commits_nothing_and_a_naked_call_commits_strike(self):
        covered = pm.option_commitments([stock("ABC", 5000, qty=100), opt("ABC261218C00060000", -1, -40)])
        assert covered["ABC261218C00060000"] == 0.0
        naked = pm.option_commitments([opt("ABC261218C00060000", -1, -40)])
        assert naked["ABC261218C00060000"] == 6000.0

    def test_iron_condor_commits_both_widths_and_a_lone_long_its_premium(self):
        legs = [opt("QQQ261218P00500000", -1, -200), opt("QQQ261218P00490000", 1, 100),
                opt("QQQ261218C00560000", -1, -150), opt("QQQ261218C00570000", 1, 60),
                opt("NVDA261218C00200000", 2, 900)]
        c = pm.option_commitments(legs)
        assert c["QQQ261218P00500000"] == 1000.0 and c["QQQ261218C00560000"] == 1000.0
        assert c["NVDA261218C00200000"] == 900.0

    def test_open_orders_count_opening_dollars_only(self):
        positions = [stock("AAPL", 2000)]
        orders = [
            {"symbol": "MSFT", "side": "buy", "qty": "10", "filled_qty": "4", "limit_price": "400"},  # 6 x 400
            {"symbol": "AAPL", "side": "sell", "qty": "10", "limit_price": "200"},                    # an exit: 0
            {"symbol": "DRAM261120P00045000", "side": "sell", "qty": "1", "limit_price": "1.2"},      # CSP: 4500
            {"symbol": "SPY", "side": "buy", "qty": "5", "limit_price": "600"},                       # floor: skipped
        ]
        assert pm.open_order_commitments(orders, positions, skip={"SPY"}) == pytest.approx(2400 + 4500)


def plan_file(tmp_path, ratio, age_s=0, ok=True, equity=100_000.0):
    at = (datetime.now() - timedelta(seconds=age_s)).isoformat()
    (tmp_path / pm.FLOOR_VM_STATE_FILE).write_text(json.dumps({
        "at": at, "equity": equity, "ok": ok, "target_base_dollars": equity * min(ratio, 0.85),
        "target_levered_dollars": 0.0}))


class TestHeadroom:
    def test_floor_target_protects_cash_the_floor_is_about_to_deploy(self, tmp_path):
        # floor holds 30k but its own plan targets 60k: satellites may not spend the other 30k
        snap = pm.satellite_headroom(100_000, [stock("SPY", 30_000)], "BULL", CFG, floor_plan_ratio=0.60)
        assert snap["floor_committed"] == 60_000 and snap["floor_target_source"] == "the floor's own latest plan"
        assert snap["headroom_dollars"] == pytest.approx(100_000 - 60_000 - 5_000)
        # no plan on file: the whole 85% sleeve is protected
        snap = pm.satellite_headroom(100_000, [stock("SPY", 30_000)], "BULL", CFG, floor_plan_ratio=None)
        assert snap["floor_committed"] == 85_000 and snap["headroom_dollars"] == pytest.approx(10_000)

    def test_floor_value_above_its_target_still_counts_and_legacy_tickers_are_floor(self):
        snap = pm.satellite_headroom(100_000, [stock("SPY", 50_000), stock("GLD", 26_000), stock("AAPL", 3_000)],
                                     "NEUTRAL", CFG, floor_plan_ratio=0.40)
        assert snap["floor_market_value"] == 76_000 and snap["floor_committed"] == 76_000
        assert snap["satellite_committed"] == 3_000
        assert snap["headroom_dollars"] == pytest.approx(100_000 - 76_000 - 3_000 - 8_000)

    def test_risk_off_reserve_counts_the_floors_idle_cash(self):
        # BEAR, vol high: floor plan 0.54 of equity -> 46% cash, 25% reserve -> ~21% for satellites
        snap = pm.satellite_headroom(100_000, [stock("SPY", 54_000)], "BEAR", CFG, floor_plan_ratio=0.54)
        assert snap["reserve_pct"] == 0.25 and snap["headroom_dollars"] == pytest.approx(21_000)

    def test_reserve_shortfall_blocks_without_selling_and_says_so(self):
        snap = pm.satellite_headroom(100_000, [stock("SPY", 85_000), stock("AAPL", 9_000)], "CAUTION", CFG,
                                     floor_plan_ratio=0.85)
        assert snap["headroom_dollars"] == 0.0
        assert snap["reserve_shortfall"] == pytest.approx(15_000 - 6_000)
        assert "never sold for the reserve" in snap["summary"]

    def test_basket_mode_uses_the_regimes_floor_share(self):
        basket = {**CFG, "FLOOR_MODE": "basket"}
        snap = pm.satellite_headroom(100_000, [], "BULL", basket)
        # SPY_FLOOR_BULL 0.70 x (1 - basket CASH 0.10) + DEFENSIVE_FLOOR_BULL 0.08
        assert snap["floor_committed"] == pytest.approx(100_000 * (0.70 * 0.90 + 0.08))

    def test_disabled_reserve_reports_enabled_false(self):
        snap = pm.satellite_headroom(100_000, [], "PANIC", {**CFG, "CASH_RESERVE_ENABLED": False}, floor_plan_ratio=0.5)
        assert snap["enabled"] is False and snap["reserve_pct"] == 0.0
        assert "switched off" in snap["summary"]

    def test_floor_plan_file_freshness(self, tmp_path):
        assert pm.floor_plan_target_ratio() is None                    # no file
        plan_file(tmp_path, 0.6)
        assert pm.floor_plan_target_ratio() == pytest.approx(0.6)
        plan_file(tmp_path, 0.6, age_s=pm.FLOOR_PLAN_MAX_AGE_S + 60)
        assert pm.floor_plan_target_ratio() is None                    # stale
        plan_file(tmp_path, 0.6, ok=False)
        assert pm.floor_plan_target_ratio() is None                    # the plan itself failed


class TestRiskBudget:
    """The regime engine's risk budget (regime_engine.risk_budget_for) acts on the
    satellites' room above the floor only: reserve = max(label, (1 - b) x room)."""
    FLOOR = [stock("SPY", 85_000)]

    def snap(self, regime="BULL", rb=None, ratio=0.85, positions=None, cfg=CFG):
        return pm.satellite_headroom(100_000, self.FLOOR if positions is None else positions, regime, cfg,
                                     floor_plan_ratio=ratio, risk_budget=rb)

    def test_no_budget_or_a_full_budget_leaves_the_label_reserve(self):
        a, b = self.snap(), self.snap(rb={"budget": 1.0, "summary": "calm"})
        assert a["headroom_dollars"] == b["headroom_dollars"] == pytest.approx(10_000)
        assert (a["risk_budget"], a["reserve_binding"]) == (None, "label")
        assert (b["risk_budget"], b["reserve_binding"]) == (1.0, "label")

    def test_budget_reserves_its_share_of_the_room_above_the_floor_max_not_sum(self):
        s = self.snap(rb={"budget": 0.5, "summary": "VIX 24 and breadth weak"})
        # room above the floor 15k; (1 - 0.5) x 15k = 7.5k > the 5k BULL label reserve
        assert s["room_above_floor"] == 15_000 and s["reserve_dollars"] == pytest.approx(7_500)
        assert s["headroom_dollars"] == pytest.approx(15_000 * 0.5)                  # room_eff = room x budget
        assert s["reserve_binding"] == "risk_budget" and s["floor_committed"] == 85_000
        assert "VIX 24 and breadth weak" in s["reserve_reason"] and "VIX 24 and breadth weak" in s["summary"]
        # a label reserve above the budget's share binds instead (max, not sum)
        s = self.snap("BEAR", rb={"budget": 0.8}, ratio=0.50, positions=[stock("SPY", 50_000)])
        assert (s["reserve_dollars"], s["reserve_binding"]) == (25_000, "label")      # 25k > (1-0.8) x 50k = 10k

    def test_zero_budget_closes_the_satellite_room_but_never_asks_the_floor_to_sell(self):
        s = self.snap(rb={"budget": 0.0, "summary": "crisis"})
        assert s["headroom_dollars"] == 0.0 and s["reserve_shortfall"] == 0.0
        assert "never sold for the reserve" in s["summary"]

    def test_reserve_off_ignores_the_budget(self):
        s = self.snap(rb={"budget": 0.2}, cfg={**CFG, "CASH_RESERVE_ENABLED": False})
        assert s["enabled"] is False and s["risk_budget"] is None

    def test_regime_budget_reader_is_defensive(self, monkeypatch):
        import regime_engine
        monkeypatch.delattr(regime_engine, "risk_budget_for", raising=False)
        assert pm.regime_budget() is None                                            # the API is not merged yet
        for ret, want in (({"budget": 0.6, "summary": "x"}, {"budget": 0.6, "summary": "x"}),
                          (None, None), ({"budget": None}, None), ("junk", None)):
            monkeypatch.setattr(regime_engine, "risk_budget_for", lambda r=ret: r, raising=False)
            assert pm.regime_budget() == want

        def boom():
            raise RuntimeError("state unreadable")
        monkeypatch.setattr(regime_engine, "risk_budget_for", boom, raising=False)
        assert pm.regime_budget() is None

    def test_live_headroom_applies_the_budget_and_logs_one_counterfactual(self, monkeypatch):
        import regime_engine
        calls = []
        monkeypatch.setattr(regime_engine, "risk_budget_for", lambda: {"budget": 0.5, "summary": "stressed"}, raising=False)
        monkeypatch.setattr(regime_engine, "log_counterfactual", lambda *a, **k: calls.append((a, k)) or True)
        monkeypatch.setattr(pm, "floor_plan_target_ratio", lambda *a, **k: 0.85)
        s = pm.live_headroom(100_000, self.FLOOR, "BULL", source="test.source")
        assert s["headroom_dollars"] == pytest.approx(7_500)
        (args, kw), = calls
        assert args[:2] == ("sizing", "satellite_reserve") and args[4] == "test.source"
        assert (args[2]["headroom_dollars"], args[3]["headroom_dollars"], args[3]["multiplier"]) == (10_000, 7_500, 0.75)
        assert kw == {"sleeve": "satellite_reserve"}
        monkeypatch.setattr(regime_engine, "risk_budget_for", lambda: None, raising=False)
        calls.clear()
        assert pm.live_headroom(100_000, self.FLOOR, "BULL")["headroom_dollars"] == pytest.approx(10_000)
        assert calls == []                                                           # no budget, nothing to compare

    def test_budget_reason_reaches_the_sizing_reason_and_never_the_per_trade_multiplier(self, monkeypatch):
        import inspect
        import regime_engine
        monkeypatch.setattr(regime_engine, "risk_budget_for", lambda: {"budget": 0.5, "summary": "VIX 24"}, raising=False)
        monkeypatch.setattr(pm, "floor_plan_target_ratio", lambda *a, **k: 0.85)
        sized = {"ticker": "A", "shares": 100, "price": 100.0, "side": "buy", "position_value": 10_000.0,
                 "position_pct": 0.1, "reasoning": "base", "blocked": False, "block_reason": None, "scalars": {}}
        with patch("position_sizing.calculate_position", return_value=dict(sized)):
            out = ps.size_portfolio([{"ticker": "A", "score": 90, "price": 100}], 100_000, self.FLOOR, {"regime": "BULL"})
        assert out[0]["position_value"] == 7_500 and "regime risk budget x0.50" in out[0]["reasoning"]
        assert "VIX 24" in out[0]["reasoning"]
        # the budget is a reserve on total room — the per-trade sizer never reads it (no double cut)
        assert "risk_budget" not in inspect.getsource(ps.calculate_position)


class TestFitEntry:
    def test_allow_trim_block(self):
        assert pm.fit_entry(4_000, 100, 5_000, 100_000, CFG)["action"] == "allow"
        t = pm.fit_entry(8_000, 100, 5_050, 100_000, CFG)
        assert (t["action"], t["shares"], t["value"]) == ("trim", 50, 5_000.0)
        b = pm.fit_entry(8_000, 100, 900, 100_000, CFG)                 # 9 shares = $900 < 1% of equity
        assert b["action"] == "block" and "smallest worthwhile" in b["reason"]
        assert pm.fit_entry(8_000, 100, 0, 100_000, CFG)["action"] == "block"

    def test_log_rows_for_trims_and_blocks_only_deduplicated(self, tmp_path):
        snap = pm.satellite_headroom(100_000, [], "BULL", CFG, floor_plan_ratio=0.85)
        fit = pm.fit_entry(20_000, 100, snap["headroom_dollars"], 100_000, CFG)
        assert pm.log_reserve_decision(fit, "nvda", 20_000, snap, "test") is True
        assert pm.log_reserve_decision(fit, "nvda", 20_000, snap, "test") is False      # same key inside 6 h
        assert pm.log_reserve_decision(pm.fit_entry(1, 1, 10, 100_000, CFG), "x", 1, snap, "test") is False
        r = rows(tmp_path)
        assert len(r) == 1 and r[0]["kind"] == "cash_reserve" and r[0]["action"] == "trim"
        assert r[0]["ticker"] == "NVDA" and r[0]["wanted_value"] == 20_000 and r[0]["regime"] == "BULL"
        assert r[0]["summary"].startswith("NVDA: cut from $20,000 to $10,000")


class TestSizePortfolio:
    def sized(self, value):
        return {"ticker": "X", "shares": int(value // 100), "price": 100.0, "side": "buy", "position_value": value,
                "position_pct": value / 100_000, "reasoning": "base", "blocked": False, "block_reason": None,
                "stop_loss": 94, "take_profit": 110, "scalars": {}}

    def test_batch_spends_only_the_room_above_the_reserve(self, tmp_path):
        trades = [{"ticker": "A", "score": 90, "price": 100}, {"ticker": "B", "score": 80, "price": 100},
                  {"ticker": "C", "score": 70, "price": 100}]
        positions = [stock("SPY", 85_000)]
        with patch("position_sizing.calculate_position", side_effect=lambda t, *a, **k: {**self.sized(6_000), "ticker": t["ticker"]}), \
             patch("position_mgmt.floor_plan_target_ratio", return_value=0.85):
            out = ps.size_portfolio(trades, 100_000, positions, {"regime": "BULL"})
        a, b, c = out
        assert not a["blocked"] and a["position_value"] == 6_000       # 10k room above the 5% reserve
        assert not b["blocked"] and b["position_value"] == 4_000 and "Cash reserve: cut" in b["reasoning"]
        assert c["blocked"] and c["block_reason"].startswith("Cash reserve:")
        assert sum(r["position_value"] for r in out if not r["blocked"]) <= 10_000
        assert {r["action"] for r in rows(tmp_path)} == {"trim", "block"}


# ── 2. Drawdown risk budget ─────────────────────────────────────────────────

class TestDrawdownBudget:
    def test_linear_between_start_and_end(self):
        pts = {dd: pm.dd_risk_scalar(dd, CFG) for dd in (None, 0, 9.9, 10, 12, 14, 16, 18, 30, -12)}
        assert pts == {None: 1.0, 0: 1.0, 9.9: 1.0, 10: 1.0, 12: 0.875, 14: 0.75, 16: 0.625, 18: 0.5, 30: 0.5, -12: 0.875}
        assert pm.dd_risk_scalar(15, {**CFG, "DD_RISK_BUDGET_ENABLED": False}) == 1.0

    def test_end_is_the_python_halt(self):
        assert CFG["DD_RISK_BUDGET_END_PCT"] == CFG["DRAWDOWN_HALT_PCT"]

    def test_combine_takes_the_tighter_rule_unless_told_to_replace(self):
        assert pm.combine_drawdown_scalars(0.6, 0.875, CFG) == (0.6, "ladder")
        assert pm.combine_drawdown_scalars(0.75, 0.6875, CFG) == (0.6875, "budget")
        assert pm.combine_drawdown_scalars(0.6, 0.875, {**CFG, "DD_RISK_BUDGET_REPLACES_LADDERS": True}) == (0.875, "budget")

    def test_current_drawdown_reads_the_node_peak(self, tmp_path):
        assert pm.current_drawdown_pct(90_000) is None
        (tmp_path / pm.EQUITY_PEAK_FILE).write_text(json.dumps({"equityPeak": 100_000}))
        assert pm.current_drawdown_pct(88_000) == pytest.approx(12.0)
        (tmp_path / pm.EQUITY_PEAK_FILE).write_text(json.dumps({"peak_equity": 100_000}))
        assert pm.current_drawdown_pct(101_000) == 0.0

    def test_log_once_per_day_while_cutting(self, tmp_path):
        assert pm.log_dd_budget("aapl", "t", 12.0, 0.875, "budget") is True
        assert pm.log_dd_budget("aapl", "t", 12.0, 0.875, "budget") is False
        assert pm.log_dd_budget("msft", "t", 5.0, 1.0, "budget") is False
        r = rows(tmp_path)
        assert len(r) == 1 and r[0]["kind"] == "dd_budget" and "12.0% below its peak" in r[0]["summary"]


class TestDrawdownInSizers:
    TRADE = {"ticker": "AAPL", "price": 100.0, "score": 75, "volume": 50_000_000, "deep_score": 75,
             "side": "buy", "trade_type": "etf"}

    def scalars(self, tmp_path, monkeypatch, equity, cfg_over=None):
        monkeypatch.setattr(ps, "DATA_DIR", str(tmp_path))
        fb = tmp_path / "fb.json"
        fb.write_text(json.dumps([{"strategy": "momentum", "pnl_pct": 3.0}] * 12 + [{"strategy": "momentum", "pnl_pct": -1.5}] * 4))
        monkeypatch.setattr(ps, "TRADE_FEEDBACK_PATH", str(fb))
        (tmp_path / "voltrade_equity_peak.json").write_text(json.dumps({"equityPeak": 100_000.0}))
        for k, v in (cfg_over or {}).items():
            monkeypatch.setitem(BASE_CONFIG, k, v)
        with patch("position_sizing.check_halt_status", return_value={"halted": False, "status": "ok"}):
            return ps.calculate_position(dict(self.TRADE), equity, [], {})["scalars"]

    def test_stock_sizer_keeps_the_tighter_ladder(self, tmp_path, monkeypatch):
        s = self.scalars(tmp_path, monkeypatch, 88_000.0)                # -12%: ladder 0.6, budget 0.875
        assert (s["drawdown_scalar"], s["drawdown_ladder"], s["drawdown_budget"], s["drawdown_rule"]) == (0.6, 0.6, 0.875, "ladder")

    def test_stock_sizer_budget_alone_when_it_replaces_the_ladder(self, tmp_path, monkeypatch):
        s = self.scalars(tmp_path, monkeypatch, 88_000.0, {"DD_RISK_BUDGET_REPLACES_LADDERS": True})
        assert (s["drawdown_scalar"], s["drawdown_rule"]) == (0.875, "budget")

    def options_size(self, mod, tmp_path, equity):
        (tmp_path / pm.EQUITY_PEAK_FILE).write_text(json.dumps({"equityPeak": 100_000.0}))
        stats = {"overall": {"win_rate": 0.6, "avg_win": 1.0, "avg_loss": 0.8}, "by_strategy": {}}
        names = ["_get_historical_stats", "_kelly_fraction", "_volatility_scalar", "_earnings_scalar", "_time_scalar",
                 "_portfolio_heat_scalar", "_liquidity_scalar", "_regime_scalar"]
        vals = [stats, 0.04, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        conf = "_confidence_scalar" if mod is oe else "_confidence_scalar_safe"
        patches = [patch(f"{mod.__name__}.{n}", return_value=v) for n, v in zip(names, vals)]
        patches += [patch(f"{mod.__name__}.{conf}", return_value=1.0),
                    patch("regime_engine.sizing_for", return_value=None)]
        for p in patches:
            p.start()
        try:
            return mod._dynamic_options_size({"ticker": "AAPL", "price": 50.0, "score": 70}, equity, [], macro={})
        finally:
            for p in patches:
                p.stop()

    def test_both_options_sizers_now_follow_the_budget(self, tmp_path):
        for mod in (oe, isel):
            assert self.options_size(mod, tmp_path, 100_000.0) == pytest.approx(0.04 * 0.5)       # no drawdown
            assert self.options_size(mod, tmp_path, 86_000.0) == pytest.approx(0.04 * 0.5 * 0.75)  # -14%
            assert self.options_size(mod, tmp_path, 70_000.0) == pytest.approx(0.04 * 0.5 * 0.5)   # past the end


class TestTierEngine:
    def ctx(self, positions, equity=100_000.0, peak=100_000.0, vxx=1.0):
        return ts.TierContext(equity=equity, peak_equity=peak, buying_power=equity, positions=positions,
                              vxx_ratio=vxx, spy_vs_ma50=1.0)

    def csp(self, t, size):
        return ts.TierAction(tier=1, action="SELL_CSP", ticker=t, strategy="csp", size_pct=size, reason="t1", metadata={})

    def test_reserve_scales_tier_actions_to_its_room(self, tmp_path):
        ctx = self.ctx([stock("SPY", 80_000)])
        with patch("position_mgmt.floor_plan_target_ratio", return_value=0.80), \
             patch("tiered_strategy.get_regime_caps", return_value={"MAX_TOTAL_EXPOSURE": 0.95, "regime": "BULL"}):
            out = ts._enforce_exposure_cap(ctx, [self.csp("AMD", 0.08), self.csp("INTC", 0.08)])
        # regime cap room 15%, reserve room 100 - 80 - 5 = 15% -> 16% proposed scaled to 15%
        assert sum(a.size_pct for a in out) == pytest.approx(0.15)
        out2 = None
        with patch("position_mgmt.floor_plan_target_ratio", return_value=0.85), \
             patch("tiered_strategy.get_regime_caps", return_value={"MAX_TOTAL_EXPOSURE": 0.95, "regime": "NEUTRAL"}):
            out2 = ts._enforce_exposure_cap(self.ctx([stock("SPY", 85_000)]), [self.csp("AMD", 0.08)])
        # NEUTRAL: 100 - 85 - 8 = 7% room (regime cap would allow 10%) -> the reserve binds
        assert out2[0].size_pct == pytest.approx(0.07) and "[cash reserve]" in out2[0].reason
        assert any(r["kind"] == "cash_reserve" and r["action"] == "trim" for r in rows(tmp_path))

    def test_no_room_drops_the_actions_instead_of_passing_size_zero(self, tmp_path):
        with patch("position_mgmt.floor_plan_target_ratio", return_value=0.85), \
             patch("tiered_strategy.get_regime_caps", return_value={"MAX_TOTAL_EXPOSURE": 0.95, "regime": "CAUTION"}), \
             patch("tiered_strategy.log_csp_gate_shadow") as shadow:
            out = ts._enforce_exposure_cap(self.ctx([stock("SPY", 85_000)]), [self.csp("AMD", 0.05)])
        assert out == []
        assert shadow.call_args[0][0][0]["decision"] == "rejected_cash_reserve"
        assert rows(tmp_path)[0]["action"] == "block"

    def test_tier1_drawdown_factor_is_the_tighter_rule(self):
        captured = {}
        orig = pm.combine_drawdown_scalars

        def spy(ladder, budget, cfg=None):
            captured["args"] = (ladder, budget)
            return orig(ladder, budget, cfg)
        with patch("position_mgmt.combine_drawdown_scalars", side_effect=spy), \
             patch("tiered_strategy._get_t1_universe", return_value=[]), \
             patch("regime_engine.sizing_for", return_value=None):
            ts.tier1_csp_core(self.ctx([], equity=85_500.0))               # -14.5%: ladder 0.75, budget 0.71875
        assert captured["args"] == (0.75, pytest.approx(0.71875, abs=1e-4))   # rounded to 4 places


# ── 3. Capital rotation ─────────────────────────────────────────────────────

NOW = datetime(2026, 11, 6, 15, 0)
STATS = {"overall": {"win_rate": 0.55, "avg_win": 2.0, "avg_loss": 2.0},
         "by_strategy": {"etf": {"win_rate": 0.6, "avg_win": 5.0, "avg_loss": 2.0, "trades": 40},     # +2.2%
                         "stocks": {"win_rate": 0.49, "avg_win": 4.19, "avg_loss": 4.25, "trades": 0}}}  # -0.11%


def held(sym, value, pnl_pct, cost=None, qty=10):
    cost = value / (1 + pnl_pct / 100) if cost is None else cost
    return {"symbol": sym, "qty": str(qty), "side": "long", "asset_class": "us_equity", "market_value": str(value),
            "cost_basis": str(cost), "unrealized_pl": str(value - cost), "unrealized_plpc": str(pnl_pct / 100),
            "current_price": str(value / qty), "avg_entry_price": str(cost / qty)}


def stop_row(days, atr=2.0, high=None, peak_pnl=0.0):
    row = {"entry_date": (NOW - timedelta(days=days)).strftime("%Y-%m-%d"), "current_stop_pct": atr,
           "initial_risk_pct": atr * 2, "highest_pnl": peak_pnl}
    if high is not None:
        row["highest_price"] = high
    return row


CAND = {"ticker": "TNA", "buy_ticker": "TNA", "buy_price": 40.0, "score": 80, "bucket": "etf",
        "wanted_value": 5_000.0, "instrument": "etf"}
FLAT_TAX = (lambda gain, days: gain * 0.24)


class TestRotation:
    def plan(self, positions, stop_state, cands=(CAND,), cfg=CFG, tax=FLAT_TAX):
        return pm.plan_rotations(positions, stop_state, list(cands), STATS, cfg=cfg, tax_fn=tax,
                                 tax_source="test 24%", now=NOW)

    def test_run_its_course_position_is_swapped_for_a_better_candidate(self, tmp_path):
        # up 6% vs a 6% target (3 x 2% ATR), 2.5% off its 108.00 high (> 1 ATR): faded
        p = held("AAPL", 1_053.0, 6.0)
        out = self.plan([p], {"AAPL": stop_row(6, high=108.0, peak_pnl=8.7)})
        assert len(out["rotations"]) == 1
        r = out["rotations"][0]
        assert (r["sell"]["ticker"], r["buy"]["ticker"], r["held_state"]) == ("AAPL", "TNA", "run_its_course")
        assert r["ev_hold_pct"] == 0.0 and r["gain_pct"] >= CFG["ROTATION_MIN_EDGE_PCT"]
        assert r["buy"]["value"] <= 1_053.0                               # never buys more than the sale frees
        assert "Sell AAPL" in r["summary"] and "estimates, not measurements" in r["summary"]
        assert [x["action"] for x in rows(tmp_path)] == ["planned"]

    def test_working_and_young_positions_are_not_sold(self, tmp_path):
        positions = [held("MSFT", 1_020.0, 2.0), held("NVDA", 970.0, -3.0)]
        out = self.plan(positions, {"MSFT": stop_row(6, high=10.21), "NVDA": stop_row(2)})
        assert out["rotations"] == []
        assert any("under the 4-day minimum hold" in x["reason"] for x in out["rejected"])

    def test_floor_options_and_covered_call_stocks_are_never_sold(self, tmp_path):
        positions = [held("SPY", 60_000.0, -5.0), held("GLD", 26_000.0, -2.0),
                     opt("CMCSA261218P00024000", -1, -97),
                     held("KO", 6_500.0, -4.0, qty=100), opt("KO261218C00070000", -1, -30)]
        out = self.plan(positions, {s: stop_row(30) for s in ("SPY", "GLD", "KO")})
        assert out["rotations"] == []
        sells = {x["sell"] for x in out["rejected"]}
        assert sells == {"KO"} and "covered call" in out["rejected"][0]["reason"]

    def test_negative_ev_candidate_and_thin_edges_are_rejected_with_reasons(self, tmp_path):
        loser = held("PFE", 950.0, -5.0)
        out = self.plan([loser], {"PFE": stop_row(5)}, cands=[{**CAND, "ticker": "BAC", "buy_ticker": "BAC", "bucket": "stocks"}])
        assert out["rotations"] == [] and "lost money on average" in out["rejected"][0]["reason"]
        thin = {**CFG, "ROTATION_MIN_EDGE_PCT": 5.0}
        out = self.plan([loser], {"PFE": stop_row(5)}, cfg=thin)
        assert out["rotations"] == [] and "under the 5.00% hurdle" in out["rejected"][0]["reason"]
        assert {x["action"] for x in rows(tmp_path)} == {"rejected"}

    def test_a_loser_keeps_its_time_decayed_bucket_value(self):
        out = self.plan([held("PFE", 950.0, -5.0)], {"PFE": stop_row(5)})
        r = out["rotations"][0]
        # PFE trades in the stocks bucket (-0.11% a trade) with 5 of 10 days left
        assert r["held_state"] == "loser" and r["ev_hold_pct"] == pytest.approx(-0.11 * 0.5, abs=0.01)
        assert r["tax_now"] < 0                                              # realizing the loss saves tax

    def test_long_term_status_close_by_makes_holding_win(self):
        # a big winner held 360 days: rotating now pays 24% on the gain; holding 10 more days pays 15%
        tax = pm.default_tax_fn(CFG)
        w = held("COST", 2_000.0, 100.0)
        out = self.plan([w], {"COST": stop_row(360, high=210.0, peak_pnl=110.0)}, tax=tax)
        assert out["rotations"] == []
        assert "hurdle" in out["rejected"][0]["reason"] and out["rejected"][0]["tax_now"] > 0

    def test_one_swap_per_scan_and_off_switch(self):
        positions = [held("AAPL", 1_053.0, 6.0), held("PFE", 950.0, -5.0)]
        ss = {"AAPL": stop_row(6, high=108.0, peak_pnl=8.7), "PFE": stop_row(5)}
        cands = [CAND, {**CAND, "ticker": "SOXL", "buy_ticker": "SOXL"}]
        assert len(self.plan(positions, ss, cands=cands)["rotations"]) == 1
        off = self.plan(positions, ss, cands=cands, cfg={**CFG, "ROTATION_ENABLED": False})
        assert off == {"rotations": [], "rejected": [], "considered": 0, "tax_source": "test 24%", "enabled": False}

    def test_rotation_value_without_an_edge_loses_the_costs(self):
        v = pm.rotation_value(10_000, 10_000, 0.0, 30, 0.5, 0.5, 0.10, 10, FLAT_TAX)
        assert v["gain_dollars"] < 0 and v["cost_dollars"] == pytest.approx(20.0, abs=0.05)

    def test_tax_plug_prefers_tax_filter_when_present(self, monkeypatch):
        import types
        fake = types.ModuleType("tax_filter")
        fake.estimate_tax = lambda gain, days: 0.5 * gain
        monkeypatch.setitem(__import__("sys").modules, "tax_filter", fake)
        fn, name = pm.resolve_tax_fn(CFG)
        assert name == "tax_filter.estimate_tax" and fn(100.0, 10) == 50.0
        monkeypatch.setitem(__import__("sys").modules, "tax_filter", None)
        fn, name = pm.resolve_tax_fn(CFG)
        assert fn(100.0, 10) == pytest.approx(24.0) and fn(100.0, 400) == pytest.approx(15.0)
        assert "24% short-term / 15% long-term" in name


# ── Wiring / mirrors ────────────────────────────────────────────────────────

class TestWiring:
    def test_scan_loop_fits_the_reserve_before_the_etf_plan_and_relabels_blocks(self):
        i_fit = BOT_SRC.index('_fit = _pm.fit_entry(position_value, stock["price"], _reserve_left, portfolio_value)')
        i_etf = BOT_SRC.index("instrument_decision, _etf_fields = _resolve_etf_execution(\n            instrument_decision, ticker, position_value)")
        assert i_fit < i_etf
        assert '_note_gate(ticker, "rejected_cash_reserve"' in BOT_SRC
        assert '_reserve_left -= float(_etf_fields.get("etf_position_value") or position_value)' in BOT_SRC
        assert '"cash_reserve": ({**_reserve' in BOT_SRC and '"rotation_plan": locals().get("_rotation_plan")' in BOT_SRC
        assert "_rotation_plan = _pm.plan_rotations(" in BOT_SRC

    def test_third_leg_buy_is_reserve_fitted(self):
        i = BOT_SRC.index("alloc  = equity * vrp_pct")
        j = BOT_SRC.index("shares = int(alloc / svxy_price)")
        assert "_reserve_fit(alloc, svxy_price" in BOT_SRC[i:j]

    def run_third_leg(self, tmp_path, positions, plan_ratio):
        """One _run_third_leg pass in the buy band (VXX 1.10 -> CAUTION, 15% reserve)."""
        import bot_engine as be
        from unittest.mock import MagicMock

        def resp(payload):
            m = MagicMock()
            m.json.return_value = payload
            return m

        def fake_get(url, *a, **kw):
            if url.endswith("/v2/account"):
                return resp({"equity": "100000"})
            if url.endswith("/v2/positions"):
                return resp(positions)
            if url.endswith("/v2/orders"):
                return resp([])
            if url.endswith("/v2/stocks/bars"):
                return resp({"bars": {"VXX": [{"c": c} for c in (30, 29, 28, 27, 26, 25)]}})
            if url.endswith("/v2/stocks/snapshots"):
                return resp({"SVXY": {"latestTrade": {"p": 50.0}}})
            return resp({})
        post = MagicMock(return_value=resp({"id": "o"}))
        dwell = {"zone": "BAND", "buy_ok": True, "spike": False, "recovery_exit_ok": False}
        with patch.object(be, "DATA_DIR", str(tmp_path)), patch.dict(os.environ, {"VOLTRADE_SHADOW_ENTRIES": "off"}), \
             patch("requests.get", side_effect=fake_get), patch("requests.post", post), \
             patch.object(be, "_resolve_third_leg_dwell", return_value=dwell), \
             patch("position_mgmt.floor_plan_target_ratio", return_value=plan_ratio), \
             patch("broker_clock.check_order_allowed", return_value=None):
            res = be._run_third_leg({"vxx_ratio": 1.10, "spy_vs_ma50": 1.0, "spy_below_200_days": 0, "spy_above_200d": True})
        return res, [c.kwargs.get("json") or {} for c in post.call_args_list]

    def test_third_leg_svxy_buy_is_blocked_or_cut_by_the_reserve(self, tmp_path):
        floor = [stock("SPY", 70_000)]
        # floor plan 0.70 + 15% CAUTION reserve -> 15k room: the 15% SVXY leg fits whole
        res, posts = self.run_third_leg(tmp_path, floor, 0.70)
        assert [p["qty"] for p in posts if p.get("symbol") == "SVXY"] == ["300"]
        # plan 0.80 -> 5k room: cut to 100 shares, logged
        res, posts = self.run_third_leg(tmp_path, floor, 0.80)
        assert [p["qty"] for p in posts if p.get("symbol") == "SVXY"] == ["100"]
        assert any(a.get("type") == "cash_reserve_trim" for a in res["actions"])
        # plan 0.85 -> no room: no order at all
        res, posts = self.run_third_leg(tmp_path, floor, 0.85)
        assert [p for p in posts if p.get("symbol") == "SVXY"] == []
        assert any(a.get("type") == "cash_reserve_block" for a in res["actions"])
        assert {r["source"] for r in rows(tmp_path)} == {"bot_engine._run_third_leg"}

    def test_reserve_fit_helper_blocks_trims_and_fails_open(self, monkeypatch):
        import bot_engine as be
        monkeypatch.setattr(pm, "floor_plan_target_ratio", lambda *a, **k: 0.85)
        pos = [stock("SPY", 85_000)]
        assert be._reserve_fit(15_000, 50, 100_000, pos, [], "BULL", "t", "SVXY")["action"] == "trim"   # 10k of room
        assert be._reserve_fit(5_000, 50, 100_000, pos, [], "PANIC", "t", "SVXY")["action"] == "block"
        with patch("position_mgmt.satellite_headroom", side_effect=TypeError("boom")):
            assert be._reserve_fit(5_000, 50, 100_000, pos, [], "BULL", "t", "SVXY")["action"] == "allow"

    def test_rotation_candidate_prices_an_etf_as_the_etf(self):
        import bot_engine as be
        dec = {"chosen": "etf", "ticker": "TSLL", "sizing": {"position_value": 2_000}}
        with patch("instrument_selector.plan_etf_execution",
                   return_value={"ok": True, "etf_shares": 100, "etf_price": 12.5, "etf_position_value": 1250, "sizing_source": "t"}):
            c = be._rotation_candidate({"price": 250.0}, "TSLA", 80, 2_000.0, dec, fetch_quote=lambda s: {"bp": 12.4, "ap": 12.5})
        assert (c["buy_ticker"], c["buy_price"], c["bucket"], c["instrument"]) == ("TSLL", 12.5, "etf", "etf")
        c = be._rotation_candidate({"price": 250.0, "momentum_score": 80}, "TSLA", 80, 2_000.0, {"chosen": "stock"})
        assert (c["buy_ticker"], c["bucket"]) == ("TSLA", "stocks")

    def test_daemon_route_and_rpc(self, tmp_path):
        import voltrade_daemon
        d = voltrade_daemon.RPCDispatcher()
        assert d._routes["cash_reserve_snapshot"] == ("position_mgmt", "snapshot_rpc")
        out = pm.snapshot_rpc(equity=100_000, positions=[stock("SPY", 85_000)], open_orders=[], regime="BULL")
        assert out["regime"] == "BULL" and out["headroom_dollars"] == pytest.approx(10_000)

    def test_mirrors_match_their_sources(self):
        led = open(os.path.join(ROOT, "server", "taxLedger.ts")).read()
        m = re.search(r"DEFAULT_TAX_RATES: TaxRates = \{ shortTerm: ([\d.]+), longTerm: ([\d.]+)", led)
        assert (float(m.group(1)), float(m.group(2))) == (CFG["ROTATION_TAX_RATE_SHORT_TERM"], CFG["ROTATION_TAX_RATE_LONG_TERM"])
        ts_src = open(os.path.join(ROOT, "server", "positionMgmt.ts")).read()
        assert 'POSITION_MGMT_LOG_FILE = "voltrade_position_mgmt.jsonl"' in ts_src
        import storage_config
        assert storage_config.POSITION_MGMT_LOG_PATH.endswith("voltrade_position_mgmt.jsonl")
        assert BOT_SRC.count('FLOOR_VM_STATE_FILE = "voltrade_floor_vm_state.json"') == 1
        assert pm.FLOOR_VM_STATE_FILE == "voltrade_floor_vm_state.json"
        assert CFG["ROTATION_MIN_HOLD_DAYS"] == 4                          # instrument_selector min_hold_days / SHORT_HOLD_DAYS


class TestLog:
    def test_log_rotates_and_recent_reads_the_tail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pm, "LOG_MAX_BYTES", 200)
        for i in range(6):
            pm.log_decision("rotation", "planned", f"T{i}", "x" * 60)
        assert (tmp_path / "pm.jsonl.1").exists()
        recent = pm.recent_decisions("rotation", "planned", since_seconds=3600)
        assert recent and all(r["kind"] == "rotation" for r in recent)
        with open(tmp_path / "pm.jsonl", "a") as f:
            f.write("{torn line\n")
        assert pm.recent_decisions("rotation") == recent
