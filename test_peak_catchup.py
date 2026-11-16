"""
Tests for the ONE-TIME shared-peak catch-up (2026-11-11, KNOWN BROKEN #49 —
the human's pick of research/wishlist.md option (a)):
tiered_strategy.rebaseline_peak_equity (the only writer that LOWERS the shared
peak) and paper_resume_sync.catch_up_shared_peak.

The live case these pin: Node re-baselined its peak to $99,962.18 on
2026-09-28; the shared Python peak stayed at $111,736.69; risk_kill_switch
latched "Portfolio DD -20.0% <= -20%" on 2026-11-06 at equity $89,370.65 and
cannot clear (resume gate $106,149.86). After the catch-up the shared peak is
$99,962.18, the latch is cleared through risk_kill_switch's own
reset_kill_state(), and the kill MECHANISM still fires at -20% of the new peak
($79,969.74). A manual kill is never touched. Every path is a tmp file.
"""
import json
import os

import pytest

import bot_engine
import paper_resume_sync as prs
import risk_kill_switch as rks
import tiered_strategy

STALE_PEAK = 111736.69
NODE_PEAK = 99962.18
EQUITY = 91229.62


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_engine, "_DD_STATE_PATH", str(tmp_path / "voltrade_portfolio_dd.json"))
    monkeypatch.setattr(rks, "KILLSWITCH_PATH", str(tmp_path / "voltrade_killswitch.json"))
    monkeypatch.setattr(rks, "MANUAL_KILL_PATH", str(tmp_path / "voltrade_MANUAL_KILL"))
    monkeypatch.setattr(rks, "KILL_HISTORY_PATH", str(tmp_path / "voltrade_kill_history.json"))
    monkeypatch.setattr(rks, "_state_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(tiered_strategy, "PEAK_EQUITY_PATH", str(tmp_path / "voltrade_peak_equity.json"))
    monkeypatch.setattr(bot_engine, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    return tmp_path


def _shared(tmp_path, peak=STALE_PEAK):
    (tmp_path / "voltrade_peak_equity.json").write_text(json.dumps({"peak_equity": peak}))


def _shared_now(tmp_path):
    return json.loads((tmp_path / "voltrade_peak_equity.json").read_text())["peak_equity"]


def _latch(reason="Portfolio DD -20.0% <= -20%"):
    rks._save_state({"killed": True, "kill_reason": reason, "killed_at": "2026-11-06T15:32:44.339708",
                     "peak_equity": STALE_PEAK, "consecutive_losses": 1})


# ── tiered_strategy.rebaseline_peak_equity ──────────────────────────────────
def test_write_down_lowers_and_the_ratchet_resumes_from_it(iso):
    _shared(iso)
    r = tiered_strategy.rebaseline_peak_equity(NODE_PEAK, "PAPER_REBASE")
    assert r["changed"] is True and r["before"] == STALE_PEAK and r["after"] == NODE_PEAK
    stored = json.loads((iso / "voltrade_peak_equity.json").read_text())
    assert stored["peak_equity"] == NODE_PEAK and stored["rebaselined_from"] == STALE_PEAK
    assert tiered_strategy.update_peak_equity(EQUITY) == NODE_PEAK
    assert tiered_strategy.update_peak_equity(NODE_PEAK + 100) == NODE_PEAK + 100


@pytest.mark.parametrize("target", [STALE_PEAK, STALE_PEAK + 1, 2e6])
def test_write_down_never_raises_the_peak(iso, target):
    _shared(iso)
    r = tiered_strategy.rebaseline_peak_equity(target, "PAPER_REBASE")
    assert r["changed"] is False and _shared_now(iso) == STALE_PEAK


@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf"), "x", None])
def test_write_down_refuses_nonsense(iso, bad):
    _shared(iso)
    r = tiered_strategy.rebaseline_peak_equity(bad, "PAPER_REBASE")
    assert r["changed"] is False and r["note"].startswith("refused") and _shared_now(iso) == STALE_PEAK


# ── paper_resume_sync.catch_up_shared_peak ──────────────────────────────────
def test_live_case_peak_written_down_latch_reset_mechanism_intact(iso):
    _shared(iso)
    _latch()
    r = prs.catch_up_shared_peak(NODE_PEAK, EQUITY, settle_s=0)
    assert r["ok"] is True and r["applied"] is True
    assert r["shared_peak"]["before"] == STALE_PEAK and r["shared_peak"]["after"] == NODE_PEAK
    assert r["risk_kill_switch"]["killed"] is True and r["risk_kill_switch"]["reset"] is True
    assert r["levels"]["kill_equity"] == pytest.approx(NODE_PEAK * 0.8, abs=0.01)
    assert _shared_now(iso) == NODE_PEAK
    assert rks._load_state()["killed"] is False
    # the next scan's path: update_peak_equity -> check_kill_switches (FROZEN mechanism, untouched)
    st = rks.check_kill_switches(EQUITY, peak_equity=tiered_strategy.update_peak_equity(EQUITY))
    assert st["killed"] is False, "-8.7% from the re-baselined peak is not a -20% drawdown"
    # ...and the -20% kill still fires from the new peak
    low = NODE_PEAK * 0.79
    st = rks.check_kill_switches(low, peak_equity=tiered_strategy.update_peak_equity(low))
    assert st["killed"] is True and st["kill_reason"].startswith("Portfolio DD")
    assert json.dumps(r)  # serializable for the daemon / audit line


def test_manual_kill_is_never_touched(iso):
    _shared(iso)
    _latch(reason="MANUAL kill file present")
    open(rks.MANUAL_KILL_PATH, "w").close()
    r = prs.catch_up_shared_peak(NODE_PEAK, EQUITY, settle_s=0)
    assert os.path.exists(rks.MANUAL_KILL_PATH), "reset_kill_state would delete it — never called"
    assert rks._load_state()["killed"] is True and r["risk_kill_switch"]["reset"] is False
    assert any("MANUAL" in n for n in r["notes"])


def test_non_drawdown_kill_is_left_alone(iso):
    _shared(iso)
    _latch(reason="3 consecutive losses; pause 24h")
    r = prs.catch_up_shared_peak(NODE_PEAK, EQUITY, settle_s=0)
    assert rks._load_state()["killed"] is True and r["risk_kill_switch"]["reset"] is False
    assert _shared_now(iso) == NODE_PEAK, "the peak itself is still aligned"


def test_off_paper_writes_nothing(iso, monkeypatch):
    _shared(iso)
    _latch()
    monkeypatch.setattr(bot_engine, "ALPACA_BASE_URL", "https://api.alpaca.markets")
    r = prs.catch_up_shared_peak(NODE_PEAK, EQUITY, settle_s=0)
    assert r["ok"] is False and r["applied"] is False
    assert _shared_now(iso) == STALE_PEAK and rks._load_state()["killed"] is True


def test_report_only_writes_nothing(iso):
    _shared(iso)
    _latch()
    r = prs.catch_up_shared_peak(NODE_PEAK, EQUITY, apply=False, settle_s=0)
    assert r["ok"] is False and _shared_now(iso) == STALE_PEAK and rks._load_state()["killed"] is True


@pytest.mark.parametrize("target,equity", [(EQUITY - 1, EQUITY), (0, EQUITY), (NODE_PEAK, -1), ("x", EQUITY), (float("nan"), EQUITY)])
def test_refuses_a_target_that_is_not_a_peak(iso, target, equity):
    _shared(iso)
    _latch()
    r = prs.catch_up_shared_peak(target, equity, settle_s=0)
    assert r["ok"] is False and r["notes"][0].startswith("refused")
    assert _shared_now(iso) == STALE_PEAK and rks._load_state()["killed"] is True


def test_already_aligned_is_ok_and_writes_nothing(iso):
    _shared(iso, peak=NODE_PEAK)
    r = prs.catch_up_shared_peak(NODE_PEAK, EQUITY, settle_s=0)
    assert r["ok"] is True and r["applied"] is False and r["shared_peak"]["changed"] is False


def test_a_concurrent_ratchet_is_caught_and_retried(iso, monkeypatch):
    """A Tier-2 scan's update_peak_equity that read the old file before the
    write-down puts the stale high back: the settle read-back sees it and the
    write-down is retried once; the latch is cleared only after it holds."""
    _shared(iso)
    _latch()
    real = tiered_strategy.rebaseline_peak_equity
    calls = {"n": 0}

    def racing(new_peak, basis):
        out = real(new_peak, basis)
        calls["n"] += 1
        if calls["n"] == 1:  # the scan's stale write lands right after ours
            (iso / "voltrade_peak_equity.json").write_text(json.dumps({"peak_equity": STALE_PEAK}))
        return out

    monkeypatch.setattr(tiered_strategy, "rebaseline_peak_equity", racing)
    r = prs.catch_up_shared_peak(NODE_PEAK, EQUITY, settle_s=0)
    assert calls["n"] == 2 and r["shared_peak"]["attempts"] == 2
    assert r["ok"] is True and _shared_now(iso) == NODE_PEAK and rks._load_state()["killed"] is False


def test_a_peak_that_never_settles_leaves_the_latch(iso, monkeypatch):
    _shared(iso)
    _latch()
    monkeypatch.setattr(tiered_strategy, "rebaseline_peak_equity",
                        lambda new_peak, basis: {"changed": False, "note": "write failed: disk"})
    r = prs.catch_up_shared_peak(NODE_PEAK, EQUITY, settle_s=0)
    assert r["ok"] is False and rks._load_state()["killed"] is True
    assert any("did not settle" in n for n in r["notes"])


def test_daemon_route_registered():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "voltrade_daemon.py"), encoding="utf-8").read()
    assert '"paper_resume_catch_up": ("paper_resume_sync", "catch_up_shared_peak")' in src
