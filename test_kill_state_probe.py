"""
Tests for kill_state_probe.kill_state_report — the read-only view of the three
Python-side drawdown latches (KNOWN BROKEN #49, 2026-11-10).

Pins: a latched risk_kill_switch is reported with the peak it measures from
and the equity levels that peak implies; the shared tiered peak and the
bot_engine DD halt are read without importing bot_engine; missing or corrupt
files report None instead of raising; nothing is written. Every path is
redirected to a tmp dir.
"""
import json
import os

import pytest

import kill_state_probe as ksp
import risk_kill_switch as rks


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(rks, "KILLSWITCH_PATH", str(tmp_path / "voltrade_killswitch.json"))
    monkeypatch.setattr(rks, "MANUAL_KILL_PATH", str(tmp_path / "voltrade_MANUAL_KILL"))
    monkeypatch.setattr(rks, "KILL_HISTORY_PATH", str(tmp_path / "voltrade_kill_history.json"))
    monkeypatch.setattr(rks, "_state_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(ksp, "DD_STATE_PATH", str(tmp_path / ksp.DD_STATE_FILENAME))
    return tmp_path


def _latched(peak=110727.04, reason="Portfolio DD -20.0% <= -20%"):
    rks._save_state({"killed": True, "kill_reason": reason, "killed_at": "2026-11-06T15:31:02",
                     "peak_equity": peak, "consecutive_losses": 0})


def test_dd_state_path_mirrors_bot_engine():
    import bot_engine
    assert os.path.basename(bot_engine._DD_STATE_PATH) == ksp.DD_STATE_FILENAME


def test_empty_volume_reports_nothing_latched_without_raising(iso):
    r = ksp.kill_state_report()
    assert r["risk_kill_switch"]["killed"] is False
    assert r["risk_kill_switch"]["peak_equity"] == 0.0
    assert r["risk_kill_switch"]["kill_equity"] is None
    assert r["shared_peak"]["peak_equity"] == 0.0
    assert r["bot_engine_dd"] is None
    assert r["equity"] is None


def test_latched_kill_reports_its_peak_and_the_levels_it_implies(iso):
    _latched()
    (iso / "voltrade_peak_equity.json").write_text(json.dumps({"peak_equity": 110727.04}))
    r = ksp.kill_state_report(equity=90293.63)
    k = r["risk_kill_switch"]
    assert k["killed"] is True and k["kill_reason"].startswith("Portfolio DD")
    assert k["can_auto_resume"] is False
    assert k["peak_equity"] == 110727.04
    # -20% from the stale peak trips at ~$88.6K and resumes only at 95% (~$105.2K)
    assert k["kill_equity"] == round(110727.04 * 0.8, 2)
    assert k["resume_equity"] == round(110727.04 * 0.95, 2)
    assert k["drawdown_pct"] == pytest.approx(-18.454, abs=0.01)
    assert r["shared_peak"]["peak_equity"] == 110727.04
    assert r["shared_peak"]["drawdown_pct"] == k["drawdown_pct"]
    assert r["constants"]["PORTFOLIO_DD_KILL"] == rks.PORTFOLIO_DD_KILL


def test_bot_engine_dd_file_is_read_without_importing_bot_engine(iso, monkeypatch):
    import sys
    (iso / ksp.DD_STATE_FILENAME).write_text(json.dumps({
        "peak_equity": 99962.18, "halted": False, "halt_reason": "", "halt_started_at": None,
        "last_equity": 90293.63, "last_updated": "2026-11-10T20:00:00",
    }))
    monkeypatch.delitem(sys.modules, "bot_engine", raising=False)
    monkeypatch.setitem(sys.modules, "bot_engine", None)  # an import would now fail loudly
    r = ksp.kill_state_report(equity=90293.63)
    dd = r["bot_engine_dd"]
    assert dd["peak_equity"] == 99962.18 and dd["halted"] is False
    assert dd["drawdown_pct"] == pytest.approx(-9.67, abs=0.01)


def test_corrupt_files_report_none_not_an_exception(iso):
    (iso / ksp.DD_STATE_FILENAME).write_text("{not json")
    (iso / "voltrade_peak_equity.json").write_text("[]")
    r = ksp.kill_state_report()
    assert r["bot_engine_dd"] is None
    assert r["shared_peak"]["peak_equity"] == 0.0


def test_manual_kill_file_is_reported(iso):
    (iso / "voltrade_MANUAL_KILL").write_text("")
    assert ksp.kill_state_report()["risk_kill_switch"]["manual_kill_file"] is True


def test_report_writes_nothing(iso):
    _latched()
    before = {p.name: p.read_bytes() for p in iso.iterdir()}
    ksp.kill_state_report(equity=90000)
    after = {p.name: p.read_bytes() for p in iso.iterdir()}
    assert after == before


def test_recent_kills_are_capped_at_five(iso):
    rks.KILL_HISTORY_PATH = str(iso / "voltrade_kill_history.json")
    (iso / "voltrade_kill_history.json").write_text(json.dumps(
        [{"timestamp": f"t{i}", "reason": "DRAWDOWN", "details": {}} for i in range(12)]))
    assert len(ksp.kill_state_report()["risk_kill_switch"]["recent_kills"]) == 5


def test_drawdown_pct_helper_rejects_unusable_pairs():
    assert ksp.drawdown_pct(None, 100) is None
    assert ksp.drawdown_pct(90, 0) is None
    assert ksp.drawdown_pct("x", 100) is None
    assert ksp.drawdown_pct(80, 100) == -20.0
