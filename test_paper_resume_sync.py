"""
Tests for paper_resume_sync.rebaseline_python_halts — the Python half of the
paper-account drawdown-kill AUTO-RESUME (server/killSwitchAutoResume.ts,
human-directed 2026-09-28, KNOWN BROKEN #42/#43).

Pins: the bot_engine DD halt is re-baselined only on a re-baselining basis;
risk_kill_switch's persisted DD kill is cleared only through its own
reset_kill_state() and NEVER when a manual kill file exists; nothing is
written for a non-paper base URL, a bad equity, or report-only calls; the
shared tiered peak is written DOWN with the others on a re-baselining basis
(2026-11-11, KNOWN BROKEN #49 — the human's wishlist option (a); until then it
was reported, never lowered) and left untouched otherwise. Every path is redirected to a
tmp dir — no real state file is touched.
"""
import json
import os

import pytest

import bot_engine
import paper_resume_sync as prs
import risk_kill_switch as rks
import tiered_strategy


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_engine, "_DD_STATE_PATH", str(tmp_path / "voltrade_portfolio_dd.json"))
    monkeypatch.setattr(rks, "KILLSWITCH_PATH", str(tmp_path / "voltrade_killswitch.json"))
    monkeypatch.setattr(rks, "MANUAL_KILL_PATH", str(tmp_path / "voltrade_MANUAL_KILL"))
    monkeypatch.setattr(rks, "KILL_HISTORY_PATH", str(tmp_path / "voltrade_kill_history.json"))
    monkeypatch.setattr(rks, "_state_path", lambda name: str(tmp_path / name))
    # the shared peak's writer in tiered_strategy resolves the same file by its own constant
    monkeypatch.setattr(tiered_strategy, "PEAK_EQUITY_PATH", str(tmp_path / "voltrade_peak_equity.json"))
    monkeypatch.setattr(bot_engine, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    return tmp_path


def _halt_bot_engine(peak=110727.04, cur=90748.0):
    bot_engine._save_dd_state({
        "peak_equity": peak, "last_equity": cur, "last_updated": "2026-09-10T03:12:00",
        "halted": True, "halt_reason": "DD 18.04% >= 18.0%", "halt_started_at": "2026-09-10T03:12:00",
    })


def _kill_rks(reason="Portfolio DD -20.1% <= -20%"):
    rks._save_state({"killed": True, "kill_reason": reason, "killed_at": "2026-09-10T03:12:00",
                     "peak_equity": 110727.04, "consecutive_losses": 0})


def _shared_peak(tmp_path, peak=110727.04):
    (tmp_path / "voltrade_peak_equity.json").write_text(json.dumps({"peak_equity": peak}))


def test_data_anomaly_rebaselines_bot_engine_dd_halt(iso):
    _halt_bot_engine()
    r = prs.rebaseline_python_halts(equity=101094.07, basis="DATA_ANOMALY")
    assert r["applied"] is True
    assert r["bot_engine_dd"]["before"]["halted"] is True
    assert r["bot_engine_dd"]["before"]["peak_equity"] == 110727.04
    assert r["bot_engine_dd"]["rebaselined"] is True
    s = bot_engine._load_dd_state()
    assert s["peak_equity"] == 101094.07 and s["halted"] is False
    assert bot_engine.is_trading_halted() is False
    # the halt MECHANISM is intact: a real -18% from the new baseline re-trips it
    trip = bot_engine.update_equity_peak(101094.07 * 0.81, regime="BEAR")
    assert trip["halted"] is True


def test_paper_rebase_resets_portfolio_dd_kill_via_its_own_reset(iso):
    _kill_rks()
    r = prs.rebaseline_python_halts(equity=101094.07, basis="PAPER_REBASE")
    assert r["risk_kill_switch"]["killed"] is True
    assert r["risk_kill_switch"]["reset"] is True
    assert rks._load_state()["killed"] is False


def test_manual_kill_file_is_never_cleared(iso):
    _kill_rks(reason="MANUAL kill file present")
    open(rks.MANUAL_KILL_PATH, "w").close()
    r = prs.rebaseline_python_halts(equity=101094.07, basis="DATA_ANOMALY")
    assert r["risk_kill_switch"]["reset"] is False
    assert os.path.exists(rks.MANUAL_KILL_PATH), "reset_kill_state would delete it — must never be called"
    assert rks._load_state()["killed"] is True
    assert any("MANUAL" in n for n in r["notes"])


def test_manual_file_present_even_with_dd_reason_is_not_cleared(iso):
    _kill_rks()
    open(rks.MANUAL_KILL_PATH, "w").close()
    prs.rebaseline_python_halts(equity=101094.07, basis="PAPER_REBASE")
    assert os.path.exists(rks.MANUAL_KILL_PATH)
    assert rks._load_state()["killed"] is True


def test_recovered_is_report_only(iso):
    _halt_bot_engine()
    _kill_rks()
    r = prs.rebaseline_python_halts(equity=103000.0, basis="RECOVERED")
    assert r["applied"] is False
    assert bot_engine._load_dd_state()["peak_equity"] == 110727.04
    assert bot_engine._load_dd_state()["halted"] is True
    assert rks._load_state()["killed"] is True


def test_apply_false_writes_nothing(iso):
    _halt_bot_engine()
    r = prs.rebaseline_python_halts(equity=101094.07, basis="DATA_ANOMALY", apply=False)
    assert r["applied"] is False
    assert bot_engine._load_dd_state()["halted"] is True


def test_non_paper_base_url_writes_nothing(iso, monkeypatch):
    monkeypatch.setattr(bot_engine, "ALPACA_BASE_URL", "https://api.alpaca.markets")
    _halt_bot_engine()
    _kill_rks()
    r = prs.rebaseline_python_halts(equity=101094.07, basis="DATA_ANOMALY")
    assert r["applied"] is False
    assert bot_engine._load_dd_state()["halted"] is True
    assert rks._load_state()["killed"] is True
    assert any("paper" in n for n in r["notes"])


@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf"), "garbage", None])
def test_bad_equity_is_refused(iso, bad):
    _halt_bot_engine()
    r = prs.rebaseline_python_halts(equity=bad, basis="DATA_ANOMALY")
    assert r["applied"] is False
    assert bot_engine._load_dd_state()["halted"] is True


def test_unknown_basis_is_refused(iso):
    _halt_bot_engine()
    r = prs.rebaseline_python_halts(equity=101094.07, basis="WHATEVER")
    assert r["applied"] is False
    assert bot_engine._load_dd_state()["halted"] is True


def test_non_dd_rks_kill_is_left_alone(iso):
    _kill_rks(reason="auto-resumed")  # anything that is not a Portfolio DD kill
    prs.rebaseline_python_halts(equity=101094.07, basis="DATA_ANOMALY")
    assert rks._load_state()["killed"] is True


def test_shared_tiered_peak_is_written_down_on_a_rebaselining_basis(iso):
    # 2026-11-11 (KNOWN BROKEN #49): this test pinned "reported, never lowered"
    # until the human chose wishlist option (a) — lower the shared peak with
    # the other re-baselined peaks. The cases that must NOT lower it are pinned
    # in the next test.
    _shared_peak(iso)
    r = prs.rebaseline_python_halts(equity=101094.07, basis="DATA_ANOMALY")
    assert r["shared_peak"]["peak_equity"] == 110727.04, "the report keeps the value it found"
    assert r["shared_peak"]["changed"] is True
    assert r["shared_peak"]["after"] == pytest.approx(101094.07)
    assert r["shared_peak"]["drawdown_pct_vs_equity"] == pytest.approx(-8.7, abs=0.01)
    assert json.loads((iso / "voltrade_peak_equity.json").read_text())["peak_equity"] == pytest.approx(101094.07)
    # the max-ratchet resumes from the new baseline
    assert tiered_strategy.update_peak_equity(100000.0) == pytest.approx(101094.07)
    assert tiered_strategy.update_peak_equity(102000.0) == pytest.approx(102000.0)


@pytest.mark.parametrize("basis,apply", [("RECOVERED", True), ("DATA_ANOMALY", False), ("PAPER_REBASE", False)])
def test_shared_tiered_peak_untouched_when_nothing_is_rebaselined(iso, basis, apply):
    _shared_peak(iso)
    r = prs.rebaseline_python_halts(equity=101094.07, basis=basis, apply=apply)
    assert r["shared_peak"]["changed"] is False
    assert json.loads((iso / "voltrade_peak_equity.json").read_text())["peak_equity"] == 110727.04
    assert any("report only" in n for n in r["notes"])


def test_shared_tiered_peak_untouched_off_paper(iso, monkeypatch):
    _shared_peak(iso)
    monkeypatch.setattr(bot_engine, "ALPACA_BASE_URL", "https://api.alpaca.markets")
    r = prs.rebaseline_python_halts(equity=101094.07, basis="PAPER_REBASE")
    assert r["shared_peak"]["changed"] is False
    assert json.loads((iso / "voltrade_peak_equity.json").read_text())["peak_equity"] == 110727.04


def test_report_is_json_serializable(iso):
    _halt_bot_engine()
    _kill_rks()
    json.dumps(prs.rebaseline_python_halts(equity=101094.07, basis="DATA_ANOMALY"))


def test_daemon_route_registered():
    """READ BEFORE WRITE rule 4: a Python entry point bot.ts calls must be in
    the daemon method table (the subprocess fallback stays in bot.ts)."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "voltrade_daemon.py")).read()
    assert '"paper_resume_sync": ("paper_resume_sync", "rebaseline_python_halts")' in src
