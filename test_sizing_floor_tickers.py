#!/usr/bin/env python3
"""
[REPAIR] 2026-11-05: position sizing exempts the CONFIGURED floor, not QQQ alone.

position_sizing excluded FLOOR_TICKER (QQQ) from portfolio heat and from the
deployed-capital budget. Under FLOOR_MODE "vol_managed_spy" the floor is
SPY + SSO, about 85% of equity, so every new stock/ETF/option trade would have
been sized as if 85% were already at risk (two extra positions in the count
scalar, the 0.70x / 0.50x deployed scalars, almost no deploy budget). Under
"basket" (the rollback) the exemption stays QQQ alone, exactly as before.

Run: python3 -m pytest -q test_sizing_floor_tickers.py
"""
import os
import sys
from unittest.mock import patch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import position_sizing as ps
import system_config


def _pos(sym, value):
    return {"symbol": sym, "market_value": str(value)}


SATELLITE = [_pos("AAPL", 5_000)]


def test_the_vm_floor_is_exempt_from_heat():
    assert system_config.BASE_CONFIG["FLOOR_MODE"] == "vol_managed_spy", "the shipped mode"
    assert ps._passive_floor_tickers() == {"SPY", "SSO"}
    with_floor = [_pos("SPY", 64_000), _pos("SSO", 12_000)] + SATELLITE
    assert ps._portfolio_heat_scalar(with_floor, 100_000) == ps._portfolio_heat_scalar(SATELLITE, 100_000), \
        "the floor's 76% must not shrink a satellite trade"
    assert ps._portfolio_heat_scalar(with_floor, 100_000) == 0.9, "one active position: 0.9x, deployed 5%"


def test_the_basket_rollback_keeps_the_old_exemption():
    with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "basket"}):
        assert ps._passive_floor_tickers() == {system_config.BASE_CONFIG.get("FLOOR_TICKER", "QQQ")}
        book = [_pos("QQQ", 70_000), _pos("SPY", 10_000)] + SATELLITE
        # exactly the pre-2026-11-05 behaviour: QQQ exempt, everything else counted
        assert ps._portfolio_heat_scalar(book, 100_000) == ps._portfolio_heat_scalar(
            [_pos("SPY", 10_000)] + SATELLITE, 100_000)
        assert ps._portfolio_heat_scalar(book, 100_000) == 0.8


def test_a_heavy_vm_floor_without_the_fix_would_have_halved_sizes():
    """The defect this repairs, measured: counting SPY+SSO at 85% of equity."""
    book = [_pos("SPY", 70_000), _pos("SSO", 15_000)] + SATELLITE
    with patch.object(ps, "_passive_floor_tickers", return_value={"QQQ"}):   # the old behaviour
        old = ps._portfolio_heat_scalar(book, 100_000)
    new = ps._portfolio_heat_scalar(book, 100_000)
    assert round(old, 3) == 0.49 and new == 0.9


def test_size_portfolio_budget_excludes_the_configured_floor():
    src = open(os.path.join(_ROOT, "position_sizing.py")).read()
    fn = src[src.index("def size_portfolio("):]
    end = fn.find("\ndef ")          # size_portfolio may be the module's last function
    fn = fn if end < 0 else fn[:end]
    assert "_passive_floor_tickers()" in fn and "!= _FLOOR_TICKER" not in fn
