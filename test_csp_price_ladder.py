"""2026-11-19: the first day the tier engine reached the order stage in weeks
it sold nothing. SELL 1 AAL261231P00011500 @ 0.29 (15:30Z) and @ 0.30 (18:50Z)
each rested 12 minutes unfilled and were swept. The limit was always mid minus
30% of the half-spread, so a second attempt asked what a buyer had just
declined. The dispatcher's price ladder (server/cspSweepCooldown.ts) now sends
`price_walk` for the second and third attempt of a day; these tests pin the
Python half: the walk moves the SELL limit toward the bid, never past it, and
nothing changes for callers that send none.
"""

import os
import re

import pytest

import options_execution as oe

SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "options_execution.py"), encoding="utf-8").read()
QUOTE = {"bid": 0.26, "ask": 0.34, "mid": 0.30}


def test_default_is_the_same_price_as_before():
    # mid 0.30 - 30% x (0.30 - 0.26) = 0.288 -> 0.29, the limit of 2026-11-19
    assert oe._optimized_limit_price(QUOTE, "sell") == 0.29
    assert oe._optimized_limit_price(QUOTE, "sell", walk=None) == 0.29
    assert oe._optimized_limit_price(QUOTE, "buy") == 0.31


def test_each_step_asks_less_and_the_last_is_the_bid():
    prices = [oe._optimized_limit_price(QUOTE, "sell", walk=w) for w in (0.30, 0.65, 1.0)]
    assert prices == [0.29, 0.27, 0.26]
    assert prices == sorted(prices, reverse=True) and prices[-1] == QUOTE["bid"]


@pytest.mark.parametrize("walk", [5, 1.0001, 100.0])
def test_a_walk_can_never_sell_below_the_bid(walk):
    assert oe._optimized_limit_price(QUOTE, "sell", walk=walk) == QUOTE["bid"]


@pytest.mark.parametrize("walk", [0, -1, 0.1, 0.2999])
def test_a_walk_can_never_ask_more_than_the_default(walk):
    assert oe._optimized_limit_price(QUOTE, "sell", walk=walk) == 0.29


@pytest.mark.parametrize("walk", ["abc", float("nan"), [], {}])
def test_a_bad_walk_is_the_default(walk):
    assert oe._clean_price_walk(walk) is None
    assert oe._optimized_limit_price(QUOTE, "sell", walk=walk) == 0.29


def test_a_one_sided_quote_is_handled_as_before():
    assert oe._optimized_limit_price({"bid": 0, "ask": 0.34, "mid": 0.17}, "sell", walk=1.0) == 0.17


def _chain(price=12.0):
    return [{"occ_symbol": "AAL261231P00011500", "option_type": "put", "strike": 11.5, "expiry": "2026-12-31",
             "bid": 0.26, "ask": 0.34, "mid": 0.30, "delta": -0.30, "iv": 0.5, "days_to_expiry": 42,
             "volume": 500, "open_interest": 5000}]


def test_the_put_selector_uses_the_walk_and_records_the_quote():
    first = oe._select_sell_put(_chain(), 12.0, 88_000.0, "AAL", 0.05, cash_available=14_000.0)
    third = oe._select_sell_put(_chain(), 12.0, 88_000.0, "AAL", 0.05, cash_available=14_000.0, price_walk=1.0)
    assert not first.get("error") and not third.get("error")
    assert (first["limit_price"], first["price_walk"]) == (0.29, 0.30)
    assert (third["limit_price"], third["price_walk"]) == (0.26, 1.0)
    assert (third["bid"], third["ask"]) == (0.26, 0.34)
    assert first["occ_symbol"] == third["occ_symbol"] and first["qty"] == third["qty"]      # only the price moved


def test_select_contract_reads_the_walk_from_the_tier_metadata_only():
    assert re.search(r'_price_walk = \(metadata or \{\}\)\.get\("price_walk"\) if isinstance\(metadata, dict\) else None', SRC)
    assert re.search(r"_select_sell_put\(liquid, price, equity, ticker, size_pct, cash_available=cash_available,\s*"
                     r"target_delta=_target_delta, price_walk=_price_walk\)", SRC)
    # every other strategy keeps the default: no other caller passes a walk
    assert len(re.findall(r"_optimized_limit_price\([^)\n]*walk=", SRC)) == 1


def test_the_lower_limit_is_still_judged_by_the_fair_value_check_and_the_tax_filter():
    sel = SRC[SRC.index("def select_contract("):SRC.index("def _select_sell_put(")]
    pick = sel.index("price_walk=_price_walk")
    assert pick < sel.index("option_fair_value.apply(result, strategy, ticker, price, surface_data, liquid)")
    assert pick < sel.index("_tax_gate_contract(result, ticker, strategy, metadata, price")
