# Regression test for the 2026-08-05 [RULE-REVIEW] visibility fix.
#
# instrument_selector.py:138's MAX_TOTAL_OPTIONS_PCT comment asked for a
# "revisit at 30 days of live data" (set 2026-05-03) that never happened —
# 94 days later, /api/diag/audit and a grep of experiments.md turned up
# zero live occurrences of the exposure-cap rejection, because the reason
# select_instrument() rules options out (exposure cap, score floor, hours,
# strategy-not-viable) was only ever a Python logger.info() call: when
# stock/etf was chosen instead, select_instrument()'s returned "reasoning"
# used ONLY the chosen candidate's own reasoning string (instrument_selector
# .py line ~1456, `reasoning = stock_score["reasoning"]`), so the
# options-specific "[RULED OUT: ...]" tag appended to options_score's own
# reasoning never reached server/bot.ts's INSTRUMENT audit line — the same
# class of silent-rule gap open_questions.md item #20's TIER-KILL fix
# closed for master_kill_switch.
#
# Fix: select_instrument() now tracks WHY options was ruled out in a
# dedicated `options_ruled_out_reason` variable, returns it as its own
# field, and folds it into `full_reasoning` (which already flows into
# bot_engine.py's `instrument_reasoning` -> server/bot.ts's existing
# INSTRUMENT audit() call — no new audit call site needed).
import instrument_selector
from instrument_selector import select_instrument, MAX_TOTAL_OPTIONS_PCT, MIN_OPTIONS_SCORE


def _neutralize_intelligence_and_sizing(monkeypatch):
    """Keep the test offline and deterministic: stub every helper that
    would otherwise hit the network (VXX fetch, intelligence gathering) or
    require a fully-populated trade dict (sizing/stop-config builders)."""
    monkeypatch.setattr(instrument_selector, "get_instrument_intelligence", lambda *a, **kw: {})
    monkeypatch.setattr(instrument_selector, "_build_sizing", lambda *a, **kw: {})
    monkeypatch.setattr(instrument_selector, "_build_stop_config", lambda *a, **kw: {})
    monkeypatch.setattr(instrument_selector, "_is_regular_hours", lambda: True)
    monkeypatch.setattr("macro_data.get_macro_snapshot", lambda: {"vxx_ratio": 1.0, "vxx_latest": 25.0})


def _mock_trade(score=80):
    return {
        "ticker": "AAPL", "deep_score": score, "score": score, "price": 200.0,
        "side": "buy", "action_label": "BUY", "volume": 10_000_000,
        "expected_hold_days": 3, "vrp": 5.0, "ewma_rv": 1.8, "rsi": 52,
    }


def test_options_exposure_cap_ruled_out_reaches_reasoning(monkeypatch):
    """The scenario the stale comment could never actually check: options
    exposure at/above MAX_TOTAL_OPTIONS_PCT. Stock wins by default, but the
    reason options was passed over must now be visible in both the
    dedicated field and the audited reasoning string."""
    _neutralize_intelligence_and_sizing(monkeypatch)
    monkeypatch.setattr(instrument_selector, "_score_stock",
                         lambda trade, intel: {"score": 50.0, "strategy": "buy_stock", "reasoning": "stock ok"})
    monkeypatch.setattr(instrument_selector, "_score_etf", lambda *a, **kw: None)
    monkeypatch.setattr(instrument_selector, "_score_options",
                         lambda *a, **kw: {"score": 90.0, "strategy": "sell_cash_secured_put", "reasoning": "options ok"})
    monkeypatch.setattr(instrument_selector, "_options_exposure_pct", lambda *a, **kw: MAX_TOTAL_OPTIONS_PCT + 0.04)

    result = select_instrument(_mock_trade(), 100_000, [], {})

    assert result["chosen"] == "stock"
    expected_reason = f"options exposure {MAX_TOTAL_OPTIONS_PCT + 0.04:.1%} at max {MAX_TOTAL_OPTIONS_PCT:.0%}"
    assert result["options_ruled_out_reason"] == expected_reason
    assert f"Options ruled out: {expected_reason}" in result["reasoning"]


def test_options_score_floor_ruled_out_reaches_reasoning(monkeypatch):
    """A second ruled-out cause (score below MIN_OPTIONS_SCORE, exposure
    fine) must produce its own distinct reason, not the exposure message —
    guards against the fix collapsing every ruled-out cause to one string."""
    _neutralize_intelligence_and_sizing(monkeypatch)
    monkeypatch.setattr(instrument_selector, "_score_stock",
                         lambda trade, intel: {"score": 50.0, "strategy": "buy_stock", "reasoning": "stock ok"})
    monkeypatch.setattr(instrument_selector, "_score_etf", lambda *a, **kw: None)
    monkeypatch.setattr(instrument_selector, "_score_options",
                         lambda *a, **kw: {"score": 90.0, "strategy": "sell_cash_secured_put", "reasoning": "options ok"})
    monkeypatch.setattr(instrument_selector, "_options_exposure_pct", lambda *a, **kw: 0.0)

    low_score_trade = _mock_trade(score=MIN_OPTIONS_SCORE - 1)
    result = select_instrument(low_score_trade, 100_000, [], {})

    assert result["chosen"] == "stock"
    expected_reason = f"score {MIN_OPTIONS_SCORE - 1} < {MIN_OPTIONS_SCORE}"
    assert result["options_ruled_out_reason"] == expected_reason
    assert f"Options ruled out: {expected_reason}" in result["reasoning"]


def test_options_chosen_no_ruled_out_reason(monkeypatch):
    """Sanity/regression: when options wins outright, there is nothing to
    surface — options_ruled_out_reason stays None and no stray "Options
    ruled out" text leaks into the reasoning of a trade that WAS options."""
    _neutralize_intelligence_and_sizing(monkeypatch)
    monkeypatch.setattr(instrument_selector, "_score_stock",
                         lambda trade, intel: {"score": 50.0, "strategy": "buy_stock", "reasoning": "stock ok"})
    monkeypatch.setattr(instrument_selector, "_score_etf", lambda *a, **kw: None)
    monkeypatch.setattr(instrument_selector, "_score_options",
                         lambda *a, **kw: {"score": 90.0, "strategy": "sell_cash_secured_put", "reasoning": "options ok"})
    monkeypatch.setattr(instrument_selector, "_options_exposure_pct", lambda *a, **kw: 0.0)

    result = select_instrument(_mock_trade(score=80), 100_000, [], {})

    assert result["chosen"] == "options"
    assert result["options_ruled_out_reason"] is None
    assert "Options ruled out" not in result["reasoning"]
