"""lookup_ticker: the pure pieces, and a drift sentinel that reads the scan's
own source so the lookup's quick pass can never silently diverge from it."""
import os
import re

from lookup_ticker import (
    COOLDOWN_SECS, chain_summary_from_contracts, evaluate_gates, quick_result_from_snapshot, summarize_deep,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def test_quick_pass_matches_the_scan_formula():
    # (close, open, volume, prev_close): +4 % on 2.5 M shares
    q = quick_result_from_snapshot("ABCD", (104.0, 101.0, 2_500_000, 100.0))
    assert q["change_pct"] == 4.0
    assert q["dollar_volume"] == 260_000_000.0
    # capped change × 3 + vol score (min(2.5, 5) × 2) + no direction bonus (≤ 5 %) + no penalty
    assert abs(q["quick_score"] - (4.0 * 3 + 5.0)) < 1e-9
    assert q["above_vwap"] is True and q["reasons"] == []
    assert q["floors"] == {"price_ok": True, "volume_ok": True, "symbol_ok": True, "min_price": 5, "min_volume": 500_000}
    # a 60 % mover: capped at 15, +5 direction, −30 extreme
    big = quick_result_from_snapshot("XYZ", (16.0, 10.0, 12_000_000, 10.0))
    assert abs(big["quick_score"] - (15 * 3 + 10 + 5 - 30)) < 1e-9
    # the floors flag instead of skipping: a $3 stock on 50 k shares
    penny = quick_result_from_snapshot("PNY", (3.0, 3.0, 50_000, 3.1))
    assert penny["floors"]["price_ok"] is False and penny["floors"]["volume_ok"] is False
    # the opening half hour lowers the volume floor, as the scan does
    assert quick_result_from_snapshot("PNY", (30.0, 30.0, 150_000, 30.0), opening_half_hour=True)["floors"]["volume_ok"] is True
    # a dotted or long symbol is outside the scan's universe pattern
    assert quick_result_from_snapshot("BRK.B", (400.0, 400.0, 1_000_000, 399.0))["floors"]["symbol_ok"] is False


def test_drift_sentinel_the_scan_still_uses_the_same_quick_formula():
    src = open(os.path.join(HERE, "bot_engine.py"), encoding="utf8").read()
    for piece in (
        '"quick_score": _capped_change * 3 + _vol_score + _direction_bonus + _extreme_penalty',
        "_capped_change = min(abs(change_pct), 15.0)",
        "_extreme_penalty = -30 if abs(change_pct) > 50 else (-15 if abs(change_pct) > 30 else 0)",
        "_vol_score = min(v / 1_000_000, 5.0) * 2",
        "_direction_bonus = 5 if abs(change_pct) > 10 else (2 if abs(change_pct) > 5 else 0)",
        "_min_vol = 100_000 if _opening_half_hour else MIN_VOLUME",
        '_cooldown_map = {"PANIC": 14400, "BEAR": 10800, "CAUTION": 9000,',
        '"BULL": 5400, "NEUTRAL": 7200}',
        'if _scan_regime in ("BEAR", "PANIC"):',
    ):
        assert piece in src, f"the scan no longer carries: {piece!r} — update lookup_ticker to match"
    assert COOLDOWN_SECS == {"PANIC": 14400, "BEAR": 10800, "CAUTION": 9000, "BULL": 5400, "NEUTRAL": 7200}


def _floors(ok=True):
    return {"price_ok": ok, "volume_ok": ok, "symbol_ok": True, "min_price": 5, "min_volume": 500_000}


def test_gates_run_in_the_scans_order_and_stop_at_the_first_failure():
    d = evaluate_gates(score=71.2, floors=_floors(), regime="NEUTRAL", min_score=63, cooldown_left_s=0, sector_blocked=False, max_sector=3,
                       sizing={"blocked": False, "shares": 40, "position_value": 7_300.0})
    assert [g["gate"] for g in d["gates"]] == ["universe", "regime", "cooldown", "admission", "sector", "sizing"]
    assert d["stopped_at"] is None and d["verdict"] == "would place it"
    assert d["gates"][5]["why"] == "40 sh · $7,300"
    # below the threshold: admission is where the scan stops, even though sizing would have refused too
    d2 = evaluate_gates(score=58.2, floors=_floors(), regime="NEUTRAL", min_score=65, cooldown_left_s=0, sector_blocked=True, max_sector=3,
                        sizing={"blocked": True, "block_reason": "Negative expectancy for strategy 'stocks' (WR=28%); refusing to size"})
    assert d2["stopped_at"] == "admission" and d2["verdict"] == "stopped at admission"
    assert d2["gates"][3]["why"] == "score 58.2 vs MIN_SCORE 65"
    assert d2["gates"][4]["pass"] is False and "max 3/sector" in d2["gates"][4]["why"]
    assert d2["gates"][5]["why"].startswith("Negative expectancy")
    # a bear tape blocks every new long before the score is even looked at
    d3 = evaluate_gates(score=90, floors=_floors(), regime="BEAR", min_score=75, cooldown_left_s=0, sector_blocked=False, max_sector=3, sizing=None)
    assert d3["stopped_at"] == "regime" and d3["gates"][1]["why"] == "BEAR: no new stock longs"
    assert d3["gates"][5]["pass"] is None  # sizing unavailable is undecided, never a pass
    # the stop cooldown
    d4 = evaluate_gates(score=90, floors=_floors(), regime="BULL", min_score=63, cooldown_left_s=1800, sector_blocked=False, max_sector=3, sizing={"blocked": False, "shares": 1, "position_value": 100})
    assert d4["stopped_at"] == "cooldown" and "30 min of the BULL cooldown left" in d4["gates"][2]["why"]
    # outside the universe
    d5 = evaluate_gates(score=90, floors=_floors(False), regime="BULL", min_score=63, cooldown_left_s=0, sector_blocked=False, max_sector=3, sizing=None)
    assert d5["stopped_at"] == "universe" and "price under $5" in d5["gates"][0]["why"] and "volume under 500,000" in d5["gates"][0]["why"]
    # every gate passing but one unavailable: undecided, not a buy
    d6 = evaluate_gates(score=90, floors=_floors(), regime="BULL", min_score=63, cooldown_left_s=0, sector_blocked=None, max_sector=None, sizing={"blocked": False, "shares": 2, "position_value": 200})
    assert d6["stopped_at"] is None and d6["verdict"].startswith("undecided")


def test_summarize_deep_keeps_the_card_fields_and_the_score():
    s = summarize_deep({"deep_score": 71.25, "quick_score": 40, "reasons": ["Momentum 1m +9.8%"], "sector": "Semiconductors", "internal_df": "huge"})
    assert s["score"] == 71.25 and s["reasons"] == ["Momentum 1m +9.8%"] and "internal_df" not in s
    assert summarize_deep({"quick_score": 12.5})["score"] == 12.5


def _c(opt_type, strike, delta, days_out, bid=1.0, ask=1.1, iv=0.5, oi=500, price=100.0):
    return {"occ_symbol": f"T{opt_type[0].upper()}{strike}", "opt_type": opt_type, "strike": strike, "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "iv": iv,
            "delta": delta, "delta_real": True, "oi": oi, "volume": 10, "days_out": days_out, "exp_date": f"2026-11-{10 + days_out % 18:02d}", "moneyness": strike / price, "spread_pct": (ask - bid) / ask}


def test_chain_summary_uses_the_scanners_delta_picks_and_the_near_the_money_iv():
    contracts = [
        _c("put", 90, -0.24, 30, bid=2.4, ask=2.6, iv=0.61), _c("put", 85, -0.15, 30, iv=0.66), _c("call", 110, 0.26, 30, bid=1.9, ask=2.0, iv=0.58),
        _c("call", 100, 0.52, 30, iv=0.55), _c("put", 100, -0.48, 30, iv=0.57), _c("put", 95, -0.35, 9, iv=0.9),  # the 9-day contract is outside the 21–45 window
    ]
    s = chain_summary_from_contracts(contracts, 100.0)
    assert s["contracts"] == 6 and s["window_days"] == "21–45"
    assert s["csp"]["strike"] == 90 and s["csp"]["premium_pct"] == 2.4
    assert s["cc"]["strike"] == 110
    assert 55 <= s["atm_iv"] <= 62  # the near-the-money contracts in the window, not the 9-day 90 %
    assert chain_summary_from_contracts([], 100.0) == {"contracts": 0, "atm_iv": None, "csp": None, "cc": None, "expiries": []}


def test_daemon_route_resolves_to_the_lookup_entry_point():
    # READ BEFORE WRITE #4: a Python entry point bot.ts calls is registered in the daemon's table
    import importlib
    import voltrade_daemon
    routes = voltrade_daemon.RPCDispatcher()._routes
    assert routes["lookup_ticker"] == ("lookup_ticker", "lookup")
    mod = importlib.import_module("lookup_ticker")
    assert callable(getattr(mod, "lookup"))
