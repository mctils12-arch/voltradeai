"""REPAIR 2026-11-18: the covered-call sweep (bot_engine.scan_market, step 6c)
offers a covered call on every stock position of 100+ shares. The floor's cash
park (BOXX, system_config FLOOR_VM_CASH_TICKER) reached 110 shares and the
pre-market scans of 2026-11-17 and 2026-11-18 queued "SELL COVERED CALL BOXX"
(audit MORNING-OPTIONS-SKIP, 14:30:36Z both days: dropped only because the
morning queue has no options path). The floor's cash rule sells those shares
whenever account cash must cover a short put's collateral, which would leave
the call naked. No such order was ever placed (no BOXX option order exists).
"""

import os
import re

import bot_engine
from system_config import BASE_CONFIG

SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_engine.py")).read()


def test_the_park_is_recognised_and_nothing_else_is():
    assert bot_engine.is_floor_cash_park("BOXX") is True
    assert bot_engine.is_floor_cash_park(" boxx ") is True
    for t in ("SPY", "SSO", "QQQ", "AAPL", "", None):
        assert bot_engine.is_floor_cash_park(t) is False


def test_no_park_configured_means_no_ticker_is_skipped():
    cfg = {**BASE_CONFIG, "FLOOR_VM_CASH_TICKER": ""}
    assert bot_engine.is_floor_cash_park("BOXX", cfg) is False
    # a park that clashes with a floor ticker is not a park (floor_cash_ticker's own rule)
    assert bot_engine.is_floor_cash_park("SPY", {**BASE_CONFIG, "FLOOR_VM_CASH_TICKER": "SPY"}) is False


def test_the_sweep_skips_the_park_before_it_can_append_a_covered_call():
    start = SRC.index("# Step 6c: Covered Call Sweep")
    end = SRC.index('_timing_log("step6a_trade_loop_and_covered_calls")', start)
    sweep = SRC[start:end]
    skip = sweep.index("if is_floor_cash_park(_cc_ticker):")
    append = sweep.index('"action": "SELL COVERED CALL"')
    assert skip < append
    assert re.search(r"if is_floor_cash_park\(_cc_ticker\):\s*\n\s*continue", sweep)


def test_the_floors_equity_holdings_stay_eligible():
    # the sweep exists so the passive floor's shares can carry covered calls; only the cash park is excluded
    start = SRC.index("# Step 6c: Covered Call Sweep")
    end = SRC.index('_timing_log("step6a_trade_loop_and_covered_calls")', start)
    assert "FLOOR_AND_LEG_TICKERS" not in SRC[start:end]
