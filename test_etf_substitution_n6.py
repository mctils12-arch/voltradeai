"""Regression tests — audit §6.8 N6 (ETF substitution), [REPAIR] 2026-10-27.

The leveraged-ETF substitution was mis-sized (Node bought floor(underlying
shares / 2) ETF shares), mis-priced (limit from the UNDERLYING's price),
discarded instrument_selector's ETF dollar sizing, and never applied
BLOCKED_TICKERS (whose "3x leveraged ETFs" class includes IWM→TNA and
XLF→FAS) to the substituted ETF. These pin the Python half of the fix:
  * plan_etf_execution sizes by dollars and prices off the ETF's own ask;
  * _score_etf rules out BLOCKED_TICKERS / >=3x substitutes;
  * bot_engine._resolve_etf_execution hands bot.ts the ETF order inputs or
    downgrades the choice to the stock.
No network: quotes are injected.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import instrument_selector as isel  # noqa: E402


# ── plan_etf_execution ──────────────────────────────────────────────────────

def test_plan_uses_selector_dollars_and_etf_ask_not_underlying_shares():
    # Underlying XLF-like: 200 shares @ $50 = $10,000 stock notional.
    # Selector ETF sizing = $5,000; FAS ask = $150 -> 33 shares, NOT 200/2=100.
    plan = isel.plan_etf_execution(
        {"type": "etf", "dollar_amount": 5000.0}, 10_000.0, {"bp": 149.9, "ap": 150.0})
    assert plan["ok"] is True
    assert plan["etf_shares"] == 33
    assert plan["etf_price"] == 150.0
    assert plan["etf_position_value"] == pytest.approx(33 * 150.0)
    assert plan["sizing_source"] == "selector"


def test_plan_falls_back_to_stock_notional_without_selector_sizing():
    plan = isel.plan_etf_execution(None, 3_000.0, {"bp": 29.9, "ap": 30.0})
    assert plan["ok"] is True
    assert plan["etf_shares"] == 100
    assert plan["sizing_source"] == "stock_notional"


def test_plan_never_exceeds_stock_notional_the_node_gates_checked():
    plan = isel.plan_etf_execution(
        {"type": "etf", "dollar_amount": 50_000.0}, 4_000.0, {"bp": 9.9, "ap": 10.0})
    assert plan["ok"] is True
    assert plan["etf_position_value"] <= 4_000.0
    assert plan["sizing_source"].endswith("_capped_at_stock_notional")


def test_plan_respects_a_selector_sizing_block():
    plan = isel.plan_etf_execution(
        {"type": "etf", "blocked": True, "block_reason": "negative Kelly"}, 4_000.0,
        {"bp": 9.9, "ap": 10.0})
    assert plan["ok"] is False
    assert "negative Kelly" in plan["reason"]


def test_plan_refuses_without_an_etf_quote():
    for q in (None, {}, {"bp": 10.0, "ap": 0}):
        plan = isel.plan_etf_execution({"type": "etf", "dollar_amount": 5000.0}, 10_000.0, q)
        assert plan["ok"] is False


def test_plan_refuses_when_dollars_buy_zero_shares():
    plan = isel.plan_etf_execution({"type": "etf", "dollar_amount": 100.0}, 100.0, {"ap": 150.0})
    assert plan["ok"] is False


# ── BLOCKED_TICKERS on the substitute ───────────────────────────────────────

def _etf_trade(ticker):
    return {"ticker": ticker, "side": "buy", "score": 75, "deep_score": 75,
            "expected_hold_days": 3, "above_ma200": 1.05, "ewma_rv": 2.0,
            "macro": {"vix": 18, "vxx_ratio": 1.0}}


@pytest.mark.parametrize("underlying,etf", [("IWM", "TNA"), ("XLF", "FAS")])
def test_three_x_substitutes_are_blocked(underlying, etf):
    res = isel._score_etf(_etf_trade(underlying), {}, 100_000)
    assert res is not None and res["etf_ticker"] == etf
    assert res.get("blocked") is True, f"{etf} is 3x and must be ruled out"
    assert res["score"] == 0.0


def test_blocked_tickers_list_applies_to_the_substitute(monkeypatch):
    monkeypatch.setitem(isel.LEVERAGED_ETFS, "QQQX", {"bull": "TQQQ", "bear": "SQQQ"})
    res = isel._score_etf(_etf_trade("QQQX"), {}, 100_000)
    assert res.get("blocked") is True
    assert "BLOCKED_TICKERS" in res["reason"]


def test_two_x_substitute_is_not_blocked_control():
    res = isel._score_etf(_etf_trade("TSLA"), {}, 100_000)
    assert res is not None and res["etf_ticker"] == "TSLL"
    assert not res.get("blocked")
    assert res["score"] > 0


def test_three_x_annotation_matches_config_category():
    from analyze import LEVERAGED_ETFS
    from system_config import BASE_CONFIG
    assert BASE_CONFIG["BLOCKED_ETF_MIN_LEVERAGE"] == 3
    three_x = {u for u, m in LEVERAGED_ETFS.items() if m.get("leverage", 2) >= 3}
    assert three_x == {"IWM", "XLF"}


# ── bot_engine wiring ───────────────────────────────────────────────────────

def test_resolve_etf_execution_emits_order_inputs():
    import bot_engine
    decision = {"chosen": "etf", "ticker": "TSLL", "strategy": "buy_etf_bull",
                "reasoning": "x", "sizing": {"type": "etf", "dollar_amount": 2_000.0}}
    out, fields = bot_engine._resolve_etf_execution(
        decision, "TSLA", 4_000.0, fetch_quote=lambda s: {"bp": 9.9, "ap": 10.0})
    assert out["chosen"] == "etf"
    assert fields == {"etf_shares": 200, "etf_price": 10.0,
                      "etf_position_value": 2000.0, "etf_sizing_source": "selector"}


def test_resolve_etf_execution_downgrades_to_stock_without_a_quote():
    import bot_engine
    decision = {"chosen": "etf", "ticker": "TSLL", "strategy": "buy_etf_bull",
                "reasoning": "x", "sizing": {"type": "etf", "dollar_amount": 2_000.0}}
    out, fields = bot_engine._resolve_etf_execution(
        decision, "TSLA", 4_000.0, fetch_quote=lambda s: None)
    assert out["chosen"] == "stock" and out["ticker"] == "TSLA"
    assert "ETF TSLL -> STOCK" in out["reasoning"]
    assert fields["etf_shares"] is None


def test_resolve_etf_execution_leaves_stock_choice_alone():
    import bot_engine
    decision = {"chosen": "stock", "ticker": "TSLA", "strategy": "buy_stock"}
    out, fields = bot_engine._resolve_etf_execution(
        decision, "TSLA", 4_000.0, fetch_quote=lambda s: pytest.fail("no quote needed"))
    assert out is decision
    assert set(fields.values()) == {None}


def test_scan_trade_carries_etf_fields():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_engine.py")).read()
    assert "_resolve_etf_execution(\n            instrument_decision, ticker, position_value)" in src
    assert "**_etf_fields," in src
