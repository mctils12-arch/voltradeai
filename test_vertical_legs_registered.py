#!/usr/bin/env python3
"""
[REPAIR] 2026-11-06: evaluate_and_execute registers BOTH legs of a vertical.

Every vertical builder in options_execution (bull_call_spread,
bull_put_credit_spread, calendar, diagonal) names its legs `long_leg` /
`short_leg` as bare OCC strings. The entry registration walked only
short_call / long_call / short_put / long_put / call_leg / put_leg, so
NEITHER leg was registered: options_manager managed two "unknown" single
legs, and a credit put spread's long wing could be stopped out alone (the
50% bought-option stop), leaving a naked short put, while the structure's
group profit target, max-loss exit and 21-DTE close never ran.

Run: python3 -m pytest -q test_vertical_legs_registered.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import options_execution as oe
import options_manager as om

PUT_SHORT = "XLF261218P00052000"
PUT_LONG = "XLF261218P00050000"
CALL_LONG = "AAPL261218C00250000"
CALL_SHORT = "AAPL261218C00260000"


def test_a_credit_put_spread_registers_both_legs_on_their_sides():
    contract = {"strategy": "bull_put_credit_spread", "short_leg": PUT_SHORT, "long_leg": PUT_LONG,
                "limit_price": 0.62, "short_delta": -0.28, "long_delta": -0.19, "max_loss": 138.0}
    legs = oe._contract_leg_entries(contract, "sell")
    assert [(occ, side) for occ, _px, side, _d in legs] == [(PUT_SHORT, "sell"), (PUT_LONG, "buy")]
    assert [d for *_rest, d in legs] == [-0.28, -0.19]
    assert all(px == 0.62 for _occ, px, _s, _d in legs), "no per-leg quote: the old contract-limit fallback"


def test_a_debit_call_spread_prices_each_leg_from_its_quote():
    contract = {"strategy": "bull_call_spread", "long_leg": CALL_LONG, "short_leg": CALL_SHORT, "limit_price": 3.10,
                "leg_quotes": [{"symbol": CALL_LONG, "side": "buy", "bid": 5.00, "ask": 5.20},
                               {"symbol": CALL_SHORT, "side": "sell", "bid": 2.10, "ask": 2.30}]}
    legs = {occ: (px, side) for occ, px, side, _d in oe._contract_leg_entries(contract, "buy")}
    assert legs == {CALL_LONG: (5.20, "buy"), CALL_SHORT: (2.10, "sell")}, "buy at the ask, sell at the bid"


def test_dict_legs_and_single_legs_are_unchanged():
    condor = {"strategy": "iron_condor", "limit_price": 1.4,
              "short_call": {"occ_symbol": "SPY261218C00700000", "bid": 1.1, "ask": 1.2, "delta": 0.2},
              "long_call": {"occ_symbol": "SPY261218C00710000", "bid": 0.5, "ask": 0.6, "delta": 0.1},
              "short_put": {"occ_symbol": "SPY261218P00600000", "bid": 1.3, "ask": 1.4, "delta": -0.2},
              "long_put": {"occ_symbol": "SPY261218P00590000", "bid": 0.7, "ask": 0.8, "delta": -0.1}}
    legs = oe._contract_leg_entries(condor, "sell")
    assert [(occ[-9:], px, side) for occ, px, side, _d in legs] == [
        ("C00700000", 1.1, "sell"), ("C00710000", 0.6, "buy"), ("P00600000", 1.3, "sell"), ("P00590000", 0.8, "buy")]
    assert oe._contract_leg_entries({"occ_symbol": "XLF261218P00052000", "limit_price": 0.5}, "sell") == []


def test_the_registration_block_uses_the_shared_leg_walk():
    src = open(os.path.join(_ROOT, "options_execution.py")).read()
    assert "leg_occs = _contract_leg_entries(contract, entry_side)" in src
    assert '"short_leg", "long_leg")' in src


def test_the_manager_groups_both_vertical_strategies():
    for strategy in ("bull_call_spread", "bull_put_credit_spread", "bear_put_spread"):
        assert strategy in om.MULTI_LEG_STRATEGIES, f"{strategy} legs must be managed as one group"
