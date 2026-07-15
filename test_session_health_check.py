"""
test_session_health_check.py — pins the pure classifiers in
scripts/session_health_check.py against fixture payloads shaped like the
real /api/health, /api/diag/daemon, /api/diag/audit, and /api/diag/ml
responses, including the exact KNOWN BROKEN #21/#18/#12 signatures traced
live in research/open_questions.md.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import session_health_check as hc  # noqa: E402


def _health(dark=False, status="ok", detail="", bad_subsystem=None):
    checks = {
        "server": {"status": "ok"}, "database": {"status": "ok"},
        "alpaca": {"status": "ok"}, "python": {"status": "ok"},
        "scanner": {"status": "ok"}, "licensing": {"status": "ok"},
        "bot": {"status": "active", "liveness": (
            {"dark": True, "marketHours": 3.0, "wallHours": 25.0, "detail": detail}
            if dark else {"dark": False}
        )},
    }
    if bad_subsystem:
        checks[bad_subsystem] = {"status": "error"}
    return {"status": status, "checks": checks}


def _diag_entry(msg):
    return {"time": "2026-07-14T19:57:31.571Z", "type": "DIAGNOSTIC", "message": msg}


DOWN_MSG = "[MEDIUM] Multiple API sources down: ['wikipedia', 'gdelt', 'fred'] | [WARN] stale"
CLEAN_MSG = "[OK] all sources fresh"


# ── check_liveness ────────────────────────────────────────────────────────

def test_liveness_ok_when_not_dark():
    f = hc.check_liveness(_health(dark=False))
    assert f["severity"] == hc.OK


def test_liveness_alarm_when_dark():
    f = hc.check_liveness(_health(dark=True, detail="broker unreadable"))
    assert f["severity"] == hc.ALARM
    assert "broker unreadable" in f["detail"]


def test_liveness_alarm_when_no_response():
    f = hc.check_liveness(None)
    assert f["severity"] == hc.ALARM


def test_liveness_alarm_on_bad_top_level_status():
    f = hc.check_liveness(_health(dark=False, status="error"))
    assert f["severity"] == hc.ALARM


# ── check_health_subsystems ──────────────────────────────────────────────

def test_subsystems_ok():
    assert hc.check_health_subsystems(_health())["severity"] == hc.OK


def test_subsystems_warn_on_degraded():
    f = hc.check_health_subsystems(_health(bad_subsystem="alpaca"))
    assert f["severity"] == hc.WARN
    assert "alpaca=error" in f["detail"]


def test_subsystems_ignores_bot_status_field():
    # bot's own status vocabulary ("active") is not the ok/error vocabulary
    # the other subsystems use — must not be misclassified as degraded.
    f = hc.check_health_subsystems(_health())
    assert "bot=" not in f["detail"]


# ── check_alt_data_enrichment (KNOWN BROKEN #21) ─────────────────────────

def test_alt_data_too_few_entries_is_ok_not_a_verdict():
    f = hc.check_alt_data_enrichment([_diag_entry(DOWN_MSG)])
    assert f["severity"] == hc.OK
    assert "too few" in f["detail"]


def test_alt_data_all_down_is_warn_known_broken_21():
    entries = [_diag_entry(DOWN_MSG) for _ in range(8)]
    f = hc.check_alt_data_enrichment(entries)
    assert f["severity"] == hc.WARN
    assert "KNOWN BROKEN #21" in f["detail"]


def test_alt_data_all_clean_is_ok():
    entries = [_diag_entry(CLEAN_MSG) for _ in range(8)]
    f = hc.check_alt_data_enrichment(entries)
    assert f["severity"] == hc.OK


def test_alt_data_intermittent_is_ok_not_permanent_latch():
    entries = [_diag_entry(DOWN_MSG)] * 3 + [_diag_entry(CLEAN_MSG)] * 5
    f = hc.check_alt_data_enrichment(entries)
    assert f["severity"] == hc.OK
    assert "intermittent" in f["detail"]


# ── check_daemon_memory (KNOWN BROKEN #21 v1.0.314 thresholds) ──────────

def test_daemon_memory_ok_under_trim():
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 254.2})
    assert f["severity"] == hc.OK


def test_daemon_memory_warn_over_trim_under_skip():
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 420})
    assert f["severity"] == hc.WARN


def test_daemon_memory_alarm_over_skip_recurrence():
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 560})
    assert f["severity"] == hc.ALARM
    assert "recurrence" in f["detail"]


def test_daemon_memory_warn_when_unreachable():
    assert hc.check_daemon_memory({"alive": False})["severity"] == hc.WARN
    assert hc.check_daemon_memory(None)["severity"] == hc.WARN


def test_daemon_memory_respects_custom_thresholds():
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 120}, skip_mb=100, trim_mb=50)
    assert f["severity"] == hc.ALARM


# ── check_tier2_daemon_timeouts (KNOWN BROKEN #18) ───────────────────────

def _t2_entry(msg):
    return {"time": "t", "type": "TIER2-ERROR", "message": msg}


def test_tier2_timeouts_ok_below_cluster_threshold():
    entries = [_t2_entry("Scan failed via daemon: Daemon timeout — active_dispatches=2")]
    f = hc.check_tier2_daemon_timeouts(entries)
    assert f["severity"] == hc.OK


def test_tier2_timeouts_warn_when_clustered():
    entries = [
        _t2_entry("Scan failed via daemon: Daemon timeout — active_dispatches=2"),
        _t2_entry("Scan failed via daemon: Daemon timeout — active_dispatches=3"),
        _t2_entry("Scan failed via daemon: Daemon timeout — active_dispatches=9"),
    ]
    f = hc.check_tier2_daemon_timeouts(entries)
    assert f["severity"] == hc.WARN
    assert "'2'" in f["detail"] or "2" in f["detail"]
    assert "'9'" in f["detail"] or "9" in f["detail"]


def test_tier2_timeouts_ignores_non_timeout_errors():
    entries = [_t2_entry("Scan failed (code=20 signal=none): This operation was aborted")] * 5
    f = hc.check_tier2_daemon_timeouts(entries)
    assert f["severity"] == hc.OK


# ── check_ml_feedback (KNOWN BROKEN #12) ─────────────────────────────────

def test_ml_feedback_ok_when_healthy():
    ml = {
        "model_age_hours": 2, "retrain_overdue": False,
        "feedback_live_count": 20,
        "live_performance": {"total_trades": 12},
        "live_outcome_breakdown": {"win": 7, "loss": 5},
    }
    assert hc.check_ml_feedback(ml)["severity"] == hc.OK


def test_ml_feedback_warn_on_retrain_overdue():
    ml = {"retrain_overdue": True, "model_age_hours": 400}
    f = hc.check_ml_feedback(ml)
    assert f["severity"] == hc.WARN
    assert "retrain_overdue" in f["detail"]


def test_ml_feedback_warn_on_all_orphan_exit_known_broken_12():
    ml = {
        "retrain_overdue": False, "model_age_hours": 32,
        "feedback_live_count": 15,
        "live_performance": {"total_trades": 0},
        "live_outcome_breakdown": {"orphan_exit": 15},
    }
    f = hc.check_ml_feedback(ml)
    assert f["severity"] == hc.WARN
    assert "KNOWN BROKEN #12" in f["detail"]


def test_ml_feedback_ok_when_no_live_records_yet():
    ml = {
        "retrain_overdue": False, "model_age_hours": 1,
        "feedback_live_count": 0,
        "live_performance": {"total_trades": 0},
        "live_outcome_breakdown": {},
    }
    assert hc.check_ml_feedback(ml)["severity"] == hc.OK


def test_ml_feedback_warn_on_missing_response():
    assert hc.check_ml_feedback(None)["severity"] == hc.WARN


# ── run_all_checks / overall_exit_code ───────────────────────────────────

def test_overall_exit_code_ok():
    findings = [hc.finding(hc.OK, "a", "x"), hc.finding(hc.OK, "b", "y")]
    assert hc.overall_exit_code(findings) == 0


def test_overall_exit_code_warn():
    findings = [hc.finding(hc.OK, "a", "x"), hc.finding(hc.WARN, "b", "y")]
    assert hc.overall_exit_code(findings) == 1


def test_overall_exit_code_alarm_wins_over_warn():
    findings = [hc.finding(hc.WARN, "a", "x"), hc.finding(hc.ALARM, "b", "y")]
    assert hc.overall_exit_code(findings) == 2


def test_overall_exit_code_empty_is_ok():
    assert hc.overall_exit_code([]) == 0


def test_run_all_checks_returns_six_findings():
    findings = hc.run_all_checks(_health(), {"alive": True, "rss_mb": 100}, [], [], {})
    assert len(findings) == 6
    assert all("severity" in f and "label" in f and "detail" in f for f in findings)


def test_run_all_checks_end_to_end_reproduces_live_known_broken_21_snapshot():
    # Mirrors this session's live read (2026-07-14 19:57Z window, pre-v1.0.314
    # confirmation): all-down DIAGNOSTIC entries + daemon rss under the new
    # 550MB threshold. Expect a WARN (not ALARM) — the memory fix already
    # shipped and daemon rss looks fine; the enrichment blind spot is the
    # thing still worth a human/future-session look.
    entries = [_diag_entry(DOWN_MSG) for _ in range(8)]
    findings = hc.run_all_checks(
        _health(), {"alive": True, "rss_mb": 165.6}, entries, [], {"feedback_live_count": 0},
    )
    by_label = {f["label"]: f for f in findings}
    assert by_label["alt_data_enrichment"]["severity"] == hc.WARN
    assert by_label["daemon_memory"]["severity"] == hc.OK
    assert hc.overall_exit_code(findings) == 1
