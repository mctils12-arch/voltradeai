"""REPAIR 2026-11-18: the floor's idle cash is parked in a T-bill fund (BOXX,
system_config FLOOR_VM_CASH_TICKER) and the cash reserve counted that fund as a
satellite position.

Live case, 2026-11-18 10:55 ET (read through /api/diag/positions-detail and
/api/diag/account): equity $89,112.40, SPY $60,197.01, BOXX $13,099.90, cash
$15,815.49, BEAR (25% reserve), the floor's plan at 67.4% of equity, regime
risk budget 0.4749. The tier engine proposed three cash-secured puts (WULF,
NU, VALE) and the reserve refused all three with $0 of room. The reserve's own
rule (system_config, CASH RESERVE: "in risk-off regimes the floor's own
de-risked cash counts toward the reserve") leaves $6,637.29.
"""

import pytest

import position_mgmt as pm
from system_config import BASE_CONFIG

CFG = dict(BASE_CONFIG)
EQUITY = 89_112.40
BUDGET = {"budget": 0.4749, "summary": "Risk budget 0.47 (defensive)"}


def stock(sym, mv, qty=10):
    return {"symbol": sym, "qty": str(qty), "market_value": str(mv), "asset_class": "us_equity", "side": "long"}


BOOK = [stock("BOXX", 13_099.90, 110), stock("SPY", 60_197.0118, 82)]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pm, "POSITION_MGMT_LOG_PATH", str(tmp_path / "pm.jsonl"))
    monkeypatch.delenv("VOLTRADE_CASH_RESERVE_PARK", raising=False)
    pm._dedup_seen.clear()
    yield


def snap(book=BOOK, cfg=CFG, regime="BEAR", ratio=0.674, budget=BUDGET, orders=()):
    return pm.satellite_headroom(EQUITY, book, regime, cfg, open_orders=orders,
                                 floor_plan_ratio=ratio, risk_budget=budget)


def test_the_setting_exists_and_is_on():
    assert CFG["CASH_RESERVE_PARK_IS_CASH"] is True
    assert pm.cash_park_tickers(CFG) == {"BOXX"}
    assert "BOXX" not in pm.floor_tickers(CFG)        # the park has no floor target of its own


def test_the_live_book_of_2026_11_18_has_room_again():
    s = snap()
    assert s["satellite_committed"] == 0.0
    assert s["cash_park_value"] == pytest.approx(13_099.90) and s["cash_park_counted_as"] == "cash"
    assert s["floor_committed"] == pytest.approx(60_197.01, abs=0.01)
    assert s["reserve_dollars"] == pytest.approx(EQUITY * 0.25, abs=0.01) and s["reserve_binding"] == "label"
    # equity - floor - 25% reserve = 89,112.40 - 60,197.01 - 22,278.10
    assert s["headroom_dollars"] == pytest.approx(6_637.29, abs=0.01)
    assert "parked in BOXX is the floor's idle cash and counts as cash" in s["summary"]


def test_one_setting_restores_the_old_count():
    s = snap(cfg={**CFG, "CASH_RESERVE_PARK_IS_CASH": False})
    assert s["satellite_committed"] == pytest.approx(13_099.90)
    assert s["headroom_dollars"] == 0.0 and s["cash_park_counted_as"] == "satellite"
    assert s["reserve_shortfall"] == pytest.approx(22_278.10 - 15_815.49, abs=0.01)
    assert "counted as a satellite position" in s["summary"]


def test_the_env_switch_rolls_back_without_a_deploy(monkeypatch):
    monkeypatch.setenv("VOLTRADE_CASH_RESERVE_PARK", "satellite")
    assert pm.park_is_cash(CFG) is False and snap()["headroom_dollars"] == 0.0
    monkeypatch.setenv("VOLTRADE_CASH_RESERVE_PARK", "cash")
    assert pm.park_is_cash({**CFG, "CASH_RESERVE_PARK_IS_CASH": False}) is True
    monkeypatch.setenv("VOLTRADE_CASH_RESERVE_PARK", "nonsense")          # ignored: the config decides
    assert pm.park_is_cash(CFG) is True
    assert pm.park_is_cash({**CFG, "CASH_RESERVE_PARK_IS_CASH": False}) is False


def test_room_is_never_more_than_cash_plus_the_park():
    # cash 15,815.49 + park 13,099.90 = 28,915.39 is everything that is not SPY
    for regime in ("BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC"):
        s = snap(regime=regime, budget=None)
        assert s["cash_after_commitments"] == pytest.approx(28_915.39, abs=0.01)
        assert s["headroom_dollars"] <= 28_915.39
    # a floor still below its plan takes its shortfall out of that first
    s = snap(ratio=0.80, budget=None)
    assert s["floor_committed"] == pytest.approx(EQUITY * 0.80, abs=0.01)
    assert s["headroom_dollars"] == pytest.approx(max(0.0, EQUITY * 0.20 - EQUITY * 0.25))


def test_the_risk_budget_and_real_satellites_still_bind():
    # a budget of 0.10 reserves 90% of the room above the floor: 0.9 x 28,915.39 = 26,023.85
    s = snap(budget={"budget": 0.10, "summary": "x"})
    assert s["reserve_binding"] == "risk_budget"
    assert s["headroom_dollars"] == pytest.approx(28_915.39 - 26_023.85, abs=0.02)
    # a real satellite is still a satellite
    s = snap(book=BOOK + [stock("AAPL", 5_000.0)])
    assert s["satellite_committed"] == pytest.approx(5_000.0)


def test_a_working_buy_of_the_park_commits_nothing_but_a_satellite_buy_does():
    orders = [{"symbol": "BOXX", "side": "buy", "qty": "20", "filled_qty": "0", "limit_price": "119.10"},
              {"symbol": "AAPL", "side": "buy", "qty": "10", "filled_qty": "0", "limit_price": "200"}]
    s = snap(orders=orders)
    assert s["pending_orders"] == pytest.approx(2_000.0)
    off = snap(orders=orders, cfg={**CFG, "CASH_RESERVE_PARK_IS_CASH": False})
    assert off["pending_orders"] == pytest.approx(2_000.0 + 20 * 119.10)


def test_basket_mode_has_no_park():
    basket = {**CFG, "FLOOR_MODE": "basket"}
    assert pm.cash_park_tickers(basket) == set()
    s = pm.satellite_headroom(EQUITY, BOOK, "BEAR", basket)
    # under the basket neither SPY nor BOXX is a floor holding: both are satellites, as before
    assert s["satellite_committed"] == pytest.approx(13_099.90 + 60_197.01, abs=0.01)
    assert s["cash_park_value"] == 0.0 and s["cash_park_counted_as"] is None


def test_no_park_held_changes_nothing():
    book = [stock("SPY", 60_197.0118, 82)]
    on, off = snap(book=book), snap(book=book, cfg={**CFG, "CASH_RESERVE_PARK_IS_CASH": False})
    keys = [k for k in on if k not in ("summary",)]
    assert {k: on[k] for k in keys} == {k: off[k] for k in keys}
    assert on["cash_park_value"] == 0.0 and on["cash_park_counted_as"] is None


def test_rotation_never_offers_the_park_as_a_weak_holding():
    from datetime import datetime, timedelta
    now = datetime(2026, 11, 18, 15, 0)
    boxx = {"symbol": "BOXX", "qty": "110", "side": "long", "asset_class": "us_equity",
            "market_value": "13099.90", "cost_basis": "13150.00", "unrealized_pl": "-50.10",
            "unrealized_plpc": "-0.0038", "current_price": "119.09", "avg_entry_price": "119.55"}
    stops = {"BOXX": {"entry_date": (now - timedelta(days=12)).strftime("%Y-%m-%d"),
                      "current_stop_pct": 2.0, "initial_risk_pct": 4.0, "highest_pnl": 0.0}}
    stats = {"overall": {"win_rate": 0.55, "avg_win": 2.0, "avg_loss": 2.0},
             "by_strategy": {"etf": {"win_rate": 0.6, "avg_win": 5.0, "avg_loss": 2.0, "trades": 40}}}
    cand = {"ticker": "TNA", "buy_ticker": "TNA", "buy_price": 40.0, "score": 80, "bucket": "etf",
            "wanted_value": 5_000.0, "instrument": "etf"}
    out = pm.plan_rotations([boxx], stops, [cand], stats, cfg=CFG, tax_fn=lambda gain, days: gain * 0.24,
                            tax_source="test 24%", now=now)
    assert out["rotations"] == []
    assert all(x.get("sell") != "BOXX" for x in out["rejected"])     # skipped like the floor: nothing to log
