"""
Tests for safety_limits_view.safety_limits — the read-only view of the
engine's safety limits behind GET /api/bot/safety-limits ([PRODUCT]
2026-11-20).

Pins: the limits are the constants themselves (never restated); the losing
streak and the day's worst loss come from the kill switch's OWN state file
(the figures check_kill_switches and record_trade_outcome write), not from
get_kill_switch_status()'s two never-written files; a day figure from another
date is marked not-today; a missing state file reports None, not 0; nothing
is written. Every path is redirected to a tmp dir.
"""
import json
import os
from datetime import datetime

import pytest

import kill_state_probe as ksp
import risk_kill_switch as rks
import safety_limits_view as slv


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(rks, "KILLSWITCH_PATH", str(tmp_path / "voltrade_killswitch.json"))
    monkeypatch.setattr(rks, "MANUAL_KILL_PATH", str(tmp_path / "voltrade_MANUAL_KILL"))
    monkeypatch.setattr(rks, "KILL_HISTORY_PATH", str(tmp_path / "voltrade_kill_history.json"))
    monkeypatch.setattr(rks, "_state_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(ksp, "DD_STATE_PATH", str(tmp_path / ksp.DD_STATE_FILENAME))
    return tmp_path


NOW = datetime(2026, 11, 20, 22, 17, 0)


def _state(iso, **over):
    s = {"killed": False, "kill_reason": "", "killed_at": None, "peak_equity": 99962.18, "consecutive_losses": 2,
         "last_loss_ts": "2026-11-20T15:01:00", "daily_loss_date": "2026-11-20", "daily_loss_pct": -0.0241038812694185}
    s.update(over)
    with open(rks.KILLSWITCH_PATH, "w") as f:
        json.dump(s, f)
    return s


def test_limits_are_the_constants_themselves(iso):
    lim = slv.safety_limits(now=NOW)["limits"]
    assert lim["kill_pct"] == round(rks.PORTFOLIO_DD_KILL * 100, 2)
    assert lim["daily_loss_pct"] == round(rks.DAILY_LOSS_LIMIT * 100, 2)
    assert lim["streak_losses"] == rks.CONSECUTIVE_LOSS_KILL
    assert lim["vxx_ratio_stop"] == rks.REGIME_KILL_VXX
    assert lim["min_free_bp_pct"] == round(rks.MIN_FREE_BP * 100, 2)
    assert lim["correlated_group_cap_pct"] == round(rks.CORRELATION_CAP * 100, 2)
    assert lim["position_stop_pct"] == round(rks.POSITION_KILL_LOSS_PCT * 100, 2)
    assert lim["position_warn_pct"] == round(rks.POSITION_WARN_LOSS_PCT * 100, 2)
    assert lim["resume_recovery_pct"] == round(rks.RESUME_REQUIRES_EQUITY_RECOVERY * 100, 2)
    assert lim["resume_cooldown_hours"] == rks.RESUME_COOLDOWN_HOURS
    from system_config import BASE_CONFIG
    assert lim["engine_halt_pct"] == -abs(float(BASE_CONFIG["DRAWDOWN_HALT_PCT"]))
    assert lim["engine_halt_enabled"] is bool(BASE_CONFIG["DRAWDOWN_HALT_ENABLED"])
    # the 24 h pause is a literal inside check_kill_switches: pin that it is still there
    import inspect
    assert "hours_since < 24" in inspect.getsource(rks.check_kill_switches)
    assert lim["streak_pause_hours"] == 24


def test_readings_come_from_the_kill_switchs_own_state_file(iso):
    _state(iso)
    v = slv.safety_limits(equity=86035.68, now=NOW)
    assert v["state_read"] is True
    assert v["streak"] == {"losses": 2, "last_loss_at": "2026-11-20T15:01:00"}
    assert v["day"] == {"worst_pct": -2.41, "date": "2026-11-20", "is_today": True}
    assert v["kill_switch"]["killed"] is False
    assert v["kill_switch"]["peak_equity"] == 99962.18
    assert v["kill_switch"]["drawdown_pct"] == -13.93
    assert v["kill_switch"]["kill_equity"] == 79969.74
    assert v["kill_switch"]["resume_equity"] == 94964.07
    # the function the old monitoring route serves answers 0 for both from files nothing writes:
    # this is the difference the view exists for
    old = rks.get_kill_switch_status()
    assert old["consecutive_losses"] == 0 and old["daily_pnl_pct"] == 0


def test_the_engines_own_writers_are_what_the_view_reads(iso):
    """record_trade_outcome and check_kill_switches write the figures; the view reads them back."""
    rks.record_trade_outcome(-0.02)
    rks.record_trade_outcome(-0.01)
    rks.check_kill_switches(equity=97000.0, peak_equity=100000.0, positions=[], daily_pnl_pct=-0.012, vxx_ratio=1.0, buying_power=100000.0)
    v = slv.safety_limits(equity=97000.0)
    assert v["streak"]["losses"] == 2
    assert v["day"]["worst_pct"] == -1.2 and v["day"]["is_today"] is True
    assert v["kill_switch"]["drawdown_pct"] == -3.0


def test_a_day_figure_from_another_date_is_not_todays(iso):
    _state(iso, daily_loss_date="2026-11-19", daily_loss_pct=-0.031)
    day = slv.safety_limits(now=NOW)["day"]
    assert day == {"worst_pct": -3.1, "date": "2026-11-19", "is_today": False}


def test_no_state_file_is_none_not_zero(iso):
    v = slv.safety_limits(equity=90000.0, now=NOW)
    assert v["state_read"] is False
    assert v["streak"] == {"losses": None, "last_loss_at": None}
    assert v["day"] == {"worst_pct": None, "date": None, "is_today": False}
    assert v["kill_switch"]["killed"] is None and v["kill_switch"]["peak_equity"] is None
    assert v["engine_halt"] is None
    assert v["recent_events"] == []
    # a corrupt file is the same answer, never an exception
    with open(rks.KILLSWITCH_PATH, "w") as f:
        f.write("{not json")
    assert slv.safety_limits(now=NOW)["state_read"] is False


def test_latched_kill_halt_and_events(iso):
    _state(iso, killed=True, kill_reason="Portfolio DD -20.0% <= -20%", killed_at="2026-11-06T15:32:44", peak_equity=111736.69)
    with open(ksp.DD_STATE_PATH, "w") as f:
        json.dump({"peak_equity": 99956.88, "halted": True, "halt_reason": "DD 18.2% > 18%", "halt_started_at": "2026-11-20T15:00:00", "last_equity": 85981.38}, f)
    with open(rks.KILL_HISTORY_PATH, "w") as f:
        json.dump([{"timestamp": "2026-11-06T15:32:44.341096", "reason": "DRAWDOWN", "details": {"equity": 89370.65, "peak": 111736.69, "dd_pct": -0.200167375639998}},
                   {"nope": 1}, "junk"], f)
    open(rks.MANUAL_KILL_PATH, "w").close()
    v = slv.safety_limits(equity=89370.65, now=NOW)
    assert v["kill_switch"]["killed"] is True
    assert v["kill_switch"]["reason"] == "Portfolio DD -20.0% <= -20%"
    assert v["kill_switch"]["killed_at"] == "2026-11-06T15:32:44"
    assert v["kill_switch"]["manual_kill_file"] is True
    assert v["engine_halt"] == {"halted": True, "reason": "DD 18.2% > 18%", "halted_at": "2026-11-20T15:00:00", "peak_equity": 99956.88, "drawdown_pct": -10.59}
    assert v["recent_events"] == [{"at": "2026-11-06T15:32:44.341096", "reason": "DRAWDOWN", "equity": 89370.65, "drawdown_pct": -20.02}]


def test_nothing_is_written(iso):
    _state(iso)
    before = {n: os.path.getmtime(os.path.join(iso, n)) for n in os.listdir(iso)}
    slv.safety_limits(equity=86035.68, now=NOW)
    after = {n: os.path.getmtime(os.path.join(iso, n)) for n in os.listdir(iso)}
    assert before == after
    json.dumps(slv.safety_limits(equity=86035.68, now=NOW))  # the body is plain JSON
