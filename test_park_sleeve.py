"""2026-11-19 (owner: "why hold that much cash on the side why not sell out of
positions"): the satellites' sleeve, 15% of equity, sat in plain cash for trades
that were not there. Live that day (read through /api/diag/account and
/api/diag/positions-detail): equity $87,949.59, SPY $59,035.08, BOXX
$13,099.02, cash $15,815.49, no option position. The T-bill fund held only the
FLOOR's idle capital; the cash step's answer was "within the band": hold.

Now the fund also holds the unused sleeve; cash keeps 2% of equity + every short
put's collateral, and the step sells the fund when cash is short of that.
"""

import math
import os
from unittest.mock import patch

import pytest

import floor_vol_managed as fvm
import system_config
from test_floor_cash_parking import _Broker, _pos, _run

CFG = dict(system_config.BASE_CONFIG)
EQ, SPY, BOXX, CASH = 87_949.59, 59_035.08, 13_099.02, 15_815.49


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("VOLTRADE_PARK_SLEEVE", raising=False)


def plan(mode_cfg, collateral=0.0, held=BOXX, cash=CASH, core=SPY, equity=EQ):
    i = fvm.parking_inputs(equity, collateral, mode_cfg)
    return i, fvm.cash_plan(i["capital"], core, held, cash, i["reserve"], mode_cfg, idle_capital=i["idle_capital"])


def test_the_setting_ships_on_with_a_two_percent_buffer():
    assert CFG["FLOOR_VM_CASH_PARK_SLEEVE"] is True and CFG["FLOOR_VM_CASH_BUFFER_PCT"] == 0.02
    assert fvm.park_sleeve_enabled(CFG) is True


def test_the_live_book_of_2026_11_19_parks_the_idle_sleeve():
    i, p = plan(CFG)
    assert i["mode"] == "sleeve"
    assert i["reserve"] == pytest.approx(EQ * 0.02) and i["buffer"] == pytest.approx(1_758.99, abs=0.01)
    # everything that is not SPY or the 2% buffer: 87,949.59 - 59,035.08 - 1,758.99
    assert p["idle"] == pytest.approx(27_155.52, abs=0.01)
    assert p["room"] == pytest.approx(BOXX + CASH - 1_758.99, abs=0.01)
    assert p["target"] == pytest.approx(27_155.52, abs=0.01)
    assert p["action"] == "buy" and p["dollars"] == pytest.approx(14_056.50, abs=0.01)
    assert "outside the core, the cash buffer and put collateral is idle" in p["reason"]


def test_one_setting_restores_the_floor_only_rule_and_its_answer_that_day():
    off = {**CFG, "FLOOR_VM_CASH_PARK_SLEEVE": False}
    i, p = plan(off)
    assert i["mode"] == "floor" and i["reserve"] == pytest.approx(EQ * 0.15)
    assert p["idle"] == pytest.approx(EQ * 0.85 - SPY, abs=0.01)             # 15,722.07: the floor's idle capital only
    assert p["action"] == "hold" and p["reason"] == "within the band of its target"


def test_the_env_switch_rolls_back_without_a_deploy(monkeypatch):
    monkeypatch.setenv("VOLTRADE_PARK_SLEEVE", "0")
    assert fvm.park_sleeve_enabled(CFG) is False and fvm.parking_inputs(EQ, 0, CFG)["mode"] == "floor"
    monkeypatch.setenv("VOLTRADE_PARK_SLEEVE", "1")
    assert fvm.park_sleeve_enabled({**CFG, "FLOOR_VM_CASH_PARK_SLEEVE": False}) is True
    monkeypatch.setenv("VOLTRADE_PARK_SLEEVE", "maybe")                       # ignored: the config decides
    assert fvm.park_sleeve_enabled(CFG) is True


def test_a_short_puts_collateral_is_never_parked_and_a_new_put_takes_the_money_back():
    # one put sold after the sleeve is parked: 4 x $11.50 strike = $4,600 of collateral
    i, p = plan(CFG, collateral=4_600.0, held=27_155.52, cash=1_758.99 + 116.0)   # + the premium
    assert i["reserve"] == pytest.approx(1_758.99 + 4_600.0, abs=0.01)
    assert p["room"] < p["idle"] + 4_600.0
    assert p["action"] == "sell" and p["dollars"] == pytest.approx(4_600.0 - 116.0, abs=0.02)
    assert p["reason"].startswith("account cash below its reserve")
    # and with the collateral known before parking, the fund is never bought into it
    _i2, p2 = plan(CFG, collateral=4_600.0)
    assert p2["target"] == pytest.approx(BOXX + CASH - 1_758.99 - 4_600.0, abs=0.01)


def test_cash_spent_by_the_core_comes_back_from_the_fund():
    # E* rose and the core bought $20,000 of SPY on buying power: cash is negative until the fund is sold
    i, p = plan(CFG, held=27_155.52, cash=1_758.99 - 20_000.0, core=SPY + 20_000.0)
    assert p["action"] == "sell" and p["dollars"] == pytest.approx(20_000.0, abs=0.02)


def test_the_target_never_exceeds_what_is_really_free():
    for collateral in (0.0, 1_150.0, 14_000.0, 60_000.0):
        for cash in (-10_000.0, 0.0, 1_758.99, CASH, 50_000.0):
            for held in (0.0, BOXX, 27_155.52):
                for core in (0.0, SPY, EQ * 0.85, EQ * 1.2):
                    i, p = plan(CFG, collateral=collateral, held=held, cash=cash, core=core)
                    # the plan rounds to cents
                    assert 0 <= p["target"] <= max(0.0, held + cash - i["reserve"]) + 0.01
                    assert p["target"] <= max(0.0, i["idle_capital"] - core) + 0.01
                    if p["action"] == "buy":
                        # after the buy, cash still covers the buffer and every put's collateral
                        assert cash - p["dollars"] >= i["reserve"] - 0.01


def test_the_buffer_cannot_exceed_the_old_sleeve_and_bad_inputs_are_zero():
    big = fvm.parking_inputs(100_000.0, 0.0, {**CFG, "FLOOR_VM_CASH_BUFFER_PCT": 0.9})
    assert big["buffer"] == pytest.approx(15_000.0)                           # capped at 1 - FLOOR_VM_CAPITAL_PCT
    z = fvm.parking_inputs(None, None, CFG)
    assert (z["capital"], z["reserve"], z["idle_capital"]) == (0.0, 0.0, 0.0)
    # collateral is kept in cash through the reserve, not by shrinking what may be parked
    c = fvm.parking_inputs(100_000.0, 50_000.0, CFG)
    assert c["idle_capital"] == pytest.approx(98_000.0) and c["capital"] == pytest.approx(85_000.0)
    assert c["reserve"] == pytest.approx(52_000.0)


def test_the_band_is_still_the_floors():
    _i, p = plan(CFG)
    assert p["band"] == pytest.approx(max(2_000.0, 0.05 * EQ * 0.85), abs=0.01)


def test_engine_pass_buys_the_fund_with_the_idle_sleeve_at_the_ask():
    b = _Broker([_pos("SPY", 118.07016, 500.0), _pos("BOXX", 110.0, 119.0818)], cash=CASH, equity=EQ)
    plan_, core, _ = _run(b, cfg={"FLOOR_VM_CASH_PARK_SLEEVE": True})
    assert plan_["parks"] == "sleeve" and plan_["cash_buffer"] == pytest.approx(1_758.99, abs=0.01)
    assert plan_["action"] == "buy" and plan_["status"] == "traded"
    (sym, qty, side, px), = b.orders
    assert (sym, side, px) == ("BOXX", "buy", 118.92)
    assert qty == int(plan_["dollars"] // 118.92) and qty * px <= CASH - 1_758.99
    assert core["actions"][-1]["type"] == "floor_vm_cash_buy"


def test_engine_pass_in_floor_mode_holds_as_it_did_that_day():
    b = _Broker([_pos("SPY", 118.07016, 500.0), _pos("BOXX", 110.0, 119.0818)], cash=CASH, equity=EQ)
    plan_, _core, _ = _run(b, cfg={"FLOOR_VM_CASH_PARK_SLEEVE": False})
    assert plan_["parks"] == "floor" and plan_["action"] == "hold" and b.orders == []
