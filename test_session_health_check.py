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


def _health(dark=False, status="ok", detail="", bad_subsystem=None, process=None):
    checks = {
        "server": {"status": "ok"}, "database": {"status": "ok"},
        "alpaca": {"status": "ok"}, "python": {"status": "ok"},
        "scanner": {"status": "ok"}, "licensing": {"status": "ok"},
        "bot": {"status": "active", "liveness": (
            {"dark": True, "marketHours": 3.0, "wallHours": 25.0, "detail": detail}
            if dark else {"dark": False}
        )},
        "process": process if process is not None else {
            "unhandledRejections": 0, "uncaughtExceptions": 0, "handlerFaults": 0,
            "lastRejection": None, "lastException": None,
        },
    }
    if bad_subsystem:
        checks[bad_subsystem] = {"status": "error"}
    return {"status": status, "checks": checks}


# ── check_restarts (2026-11-18: eight boots in 56 s and no row saying why) ──

def _restarts(**kw):
    base = {"status": "ok", "detail": None, "boots_10m": 1, "boots_24h": 14, "burst": False,
            "last_boot_at": "2026-11-18T13:48:35.000Z", "unclean_24h": 0, "last_unclean": None,
            "recorder_faults": 0, "gates_top_level_status": False}
    base.update(kw)
    h = _health()
    h["checks"]["restarts"] = base
    return h


_KILLED = {"bootAt": "2026-11-18T13:48:35.000Z", "version": "1.0.1463", "lastAliveAt": "2026-11-18T14:51:10.000Z",
           "livedS": 3755, "goneS": 7, "stage": "serving",
           "sample": {"heapUsedMB": 1500, "rssMB": 2100, "cgroupUsageMB": 2700}, "detectedAt": "2026-11-18T14:51:17.000Z"}


def test_restarts_ok_when_every_boot_followed_a_requested_stop():
    f = hc.check_restarts(_restarts())
    assert f["severity"] == hc.OK and "14 boot(s) in 24h" in f["detail"]


def test_restarts_alarm_on_a_burst_with_what_the_killed_process_looked_like():
    f = hc.check_restarts(_restarts(status="warning", burst=True, boots_10m=8, unclean_24h=8, last_unclean=_KILLED))
    assert f["severity"] == hc.ALARM
    assert "RESTART BURST: 8 boots in the last 10 minutes" in f["detail"]
    assert "3755s after its boot, stage 'serving'" in f["detail"]
    assert "heap 1500 MB / rss 2100 MB / container 2700 MB" in f["detail"]
    assert "only in Railway's deploy log" in f["detail"]


def test_restarts_warn_on_one_killed_process_after_the_burst_is_over():
    f = hc.check_restarts(_restarts(status="warning", unclean_24h=1, last_unclean=_KILLED))
    assert f["severity"] == hc.WARN and "1 process(es) killed without a shutdown signal" in f["detail"]


def test_restarts_warn_when_the_recorder_cannot_write_or_the_block_is_missing():
    assert hc.check_restarts(_restarts(recorder_faults=4))["severity"] == hc.WARN
    f = hc.check_restarts(_health())
    assert f["severity"] == hc.WARN and "predates the exit recorder" in f["detail"]
    assert hc.check_restarts(None)["severity"] == hc.WARN


def test_restarts_is_one_of_the_checks_main_runs():
    import inspect
    assert "check_restarts(health)," in inspect.getsource(hc)


def _diag_entry(msg):
    return {"time": "2026-07-14T19:57:31.571Z", "type": "DIAGNOSTIC", "message": msg}


DOWN_MSG = "[MEDIUM] Multiple API sources down: ['wikipedia', 'gdelt', 'fred'] | [WARN] stale"
CLEAN_MSG = "[OK] all sources fresh"


# ── check_liveness ────────────────────────────────────────────────────────

# ── check_engine_kill (KNOWN BROKEN #49) ─────────────────────────────────

def _ek(status="ok", kind=None, disagree=False, age=3, node=False):
    h = _health()
    h["checks"]["bot"]["engineKill"] = {
        "status": status, "kind": kind, "ageMinutes": age, "nodeKillSwitch": node,
        "disagree": disagree, "degrade": disagree,
    }
    return h


def test_engine_kill_alarm_when_engine_latched_and_node_reads_active():
    f = hc.check_engine_kill(_ek(status="latched", kind="PORTFOLIO_DD", disagree=True, age=7))
    assert f["severity"] == hc.ALARM
    assert "PORTFOLIO_DD" in f["detail"] and "/api/diag/kill_state" in f["detail"]
    assert "reset_kill_state" in f["fix"]


def test_engine_kill_ok_when_both_sides_latched_or_not_latched():
    h = _ek(status="latched", kind="DAILY_LOSS", disagree=False, node=True)
    h["checks"]["bot"]["status"] = "killed"
    assert hc.check_engine_kill(h)["severity"] == hc.OK
    assert hc.check_engine_kill(_ek(status="ok"))["severity"] == hc.OK
    assert hc.check_engine_kill(_ek(status="unknown"))["severity"] == hc.OK


def test_engine_kill_warns_when_the_server_predates_the_field():
    f = hc.check_engine_kill(_health())
    assert f["severity"] == hc.WARN and "KILL-SWITCH" in f["detail"]
    assert hc.check_engine_kill(None)["severity"] == hc.WARN


def test_engine_kill_is_part_of_the_standard_run():
    findings = hc.run_all_checks(_ek(status="latched", kind="PORTFOLIO_DD", disagree=True),
                                 {"alive": True, "rss_mb": 200}, [], [], None)
    assert any(f["label"] == "engine_kill" and f["severity"] == hc.ALARM for f in findings)
    assert hc.overall_exit_code(findings) == 2


def test_liveness_ok_when_not_dark():
    f = hc.check_liveness(_health(dark=False))
    assert f["severity"] == hc.OK


def test_liveness_alarm_when_dark():
    f = hc.check_liveness(_health(dark=True, detail="broker unreadable"))
    assert f["severity"] == hc.ALARM
    assert "broker unreadable" in f["detail"]


def test_liveness_alarm_names_the_latch_cause():
    h = _health(dark=True, detail="halted")
    h["checks"]["bot"]["status"] = "killed"
    h["checks"]["bot"]["autoResume"] = {
        "latchKind": "DRAWDOWN_KILL", "outcome": "MARKET_CLOSED",
        "reason": "market closed", "nextCheck": "2026-10-16T09:30:00-04:00",
    }
    f = hc.check_liveness(h)
    assert f["severity"] == hc.ALARM
    assert "latch=DRAWDOWN_KILL" in f["detail"]
    assert "MARKET_CLOSED" in f["detail"]


def test_liveness_alarm_flags_missing_latch_kind_on_killed_bot():
    h = _health(dark=True, detail="halted")
    h["checks"]["bot"]["status"] = "killed"
    f = hc.check_liveness(h)
    assert "latch kind not reported" in f["detail"]


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


# ── check_process_faults (KNOWN BROKEN #41's crash visibility, v1.0.917) ──

def test_process_faults_ok_when_clean():
    f = hc.check_process_faults(_health())
    assert f["severity"] == hc.OK


def test_process_faults_alarm_on_unhandled_rejection():
    f = hc.check_process_faults(_health(process={
        "unhandledRejections": 2, "uncaughtExceptions": 0, "handlerFaults": 0,
        "lastRejection": {"at": "2026-09-16T12:00:00Z", "detail": "TypeError: boom"},
        "lastException": None,
    }))
    assert f["severity"] == hc.ALARM
    assert "2 unhandled promise rejection" in f["detail"]
    assert "TypeError: boom" in f["detail"]


def test_process_faults_alarm_on_uncaught_exception():
    f = hc.check_process_faults(_health(process={
        "unhandledRejections": 0, "uncaughtExceptions": 1, "handlerFaults": 0,
        "lastRejection": None,
        "lastException": {"at": "2026-09-16T12:00:00Z", "detail": "RangeError: nope"},
    }))
    assert f["severity"] == hc.ALARM
    assert "1 uncaught exception" in f["detail"]
    assert "RangeError: nope" in f["detail"]


def test_process_faults_warn_on_handler_fault_alone():
    f = hc.check_process_faults(_health(process={
        "unhandledRejections": 0, "uncaughtExceptions": 0, "handlerFaults": 1,
        "lastRejection": None, "lastException": None,
    }))
    assert f["severity"] == hc.WARN
    assert "handler fault" in f["detail"]


def test_process_faults_warn_when_block_missing_pre_v1_0_917():
    h = _health()
    del h["checks"]["process"]
    f = hc.check_process_faults(h)
    assert f["severity"] == hc.WARN
    assert "pre-v1.0.917" in f["detail"]


def test_process_faults_warn_when_no_health_response():
    f = hc.check_process_faults(None)
    assert f["severity"] == hc.WARN


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


# 2026-11-16: the guard reads current memory; the daemon's rss_mb is a peak.

def test_daemon_memory_judged_on_current_not_peak():
    """The live reading of 2026-11-16: peak 695.7 MB after 6.5 h, enrichment
    fresh. A current reading under the trim line is OK whatever the peak."""
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 695.7, "rss_current_mb": 310.0})
    assert f["severity"] == hc.OK
    assert "310.0" in f["detail"] and "peak 695.7" in f["detail"]


def test_daemon_memory_current_over_skip_still_alarms():
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 700, "rss_current_mb": 560})
    assert f["severity"] == hc.ALARM
    assert "recurrence" in f["detail"]


def test_daemon_memory_current_in_trim_band_warns():
    f = hc.check_daemon_memory({"alive": True, "rss_mb": 700, "rss_current_mb": 420})
    assert f["severity"] == hc.WARN


def test_daemon_memory_unreadable_current_falls_back_to_peak():
    """rss_current_mb None (no /proc) or absent (older daemon): judged on
    rss_mb exactly as before — a missing reading never reads as healthy."""
    for d in ({"alive": True, "rss_mb": 560, "rss_current_mb": None},
              {"alive": True, "rss_mb": 560}):
        assert hc.check_daemon_memory(d)["severity"] == hc.ALARM


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


# (af2), 2026-11-24: the line is a share of the scans started in one stated
# window. It used to count the newest 50 TIER2-ERROR rows however old (26
# days on the day this was written: "48 timeouts in window" with none that
# day) and print no number of scans beside them.

from datetime import datetime as _dt, timezone as _tz

_T2_NOW = _dt(2026, 11, 24, 16, 0, tzinfo=_tz.utc)
_T2_SINCE = "2026-11-23T16:00:00.000Z"
_T2_TIMEOUT = "Scan failed via daemon: daemon run_full_scan failed: Daemon timeout — daemon rss=652.8MB active_dispatches=2 [run_full_scan:300s]"


def _row(time, msg, type_="TIER2"):
    return {"time": time, "type": type_, "message": msg}


def _scan_rows(times):
    rows = []
    for t in times:
        rows.append(_row(t, "Starting scan (interval: 3min based on market time)"))
        rows.append(_row(t, "Starting intelligence scan..."))
        rows.append(_row(t, "Scanned 12171 stocks, 0 trade candidates (via daemon)"))
    return rows


def test_tier2_window_since_is_24h_back_floored_to_the_minute():
    assert hc.tier2_window_since(_dt(2026, 11, 24, 16, 11, 9, 5, tzinfo=_tz.utc)) == "2026-11-23T16:11:00.000Z"


def test_tier2_timeouts_older_than_the_window_are_not_counted():
    """The production shape of 2026-11-24: weeks-old timeouts, none in the window."""
    errors = [_row("2026-10-30T15:00:00.000Z", _T2_TIMEOUT, "TIER2-ERROR")] * 40
    f = hc.check_tier2_daemon_timeouts(errors, since=_T2_SINCE, now=_T2_NOW,
                                       scans_started=140, scans_returned=140)
    assert f["severity"] == hc.OK
    assert f["detail"] == "0 of 140 scans started in the last 24.0 h ended in a daemon timeout (0.0%); 140 returned"


def test_tier2_timeouts_read_as_a_share_of_the_scans_the_server_counted():
    """2026-11-23 afternoon: 4 of 23 is one in six, and the line says so."""
    errors = [_row(f"2026-11-23T1{h}:30:00.000Z", _T2_TIMEOUT, "TIER2-ERROR") for h in (7, 8, 9, 9)]
    f = hc.check_tier2_daemon_timeouts(errors, since=_T2_SINCE, now=_T2_NOW,
                                       scans_started=23, scans_returned=19)
    assert f["severity"] == hc.WARN
    assert "4 of 23 scans started in the last 24.0 h ended in a daemon timeout (17.4%); 19 returned" in f["detail"]
    assert "active_dispatches seen: 2)" in f["detail"]
    assert "returns nothing" in f["detail"] and "KNOWN BROKEN #60" in f["detail"]
    assert "non-blocking" not in f["detail"] and "#18" not in f["detail"]
    assert "timings" in f["fix"]


def test_tier2_scans_counted_from_rows_when_the_server_serves_no_total():
    rows = _scan_rows(["2026-11-24T15:00:00.000Z", "2026-11-24T15:03:00.000Z", "2026-11-24T15:06:00.000Z"])
    rows.append(_row("2026-11-24T15:09:00.000Z", "Starting intelligence scan..."))   # still running
    f = hc.check_tier2_daemon_timeouts([], since=_T2_SINCE, now=_T2_NOW, tier2_entries=rows)
    assert f["severity"] == hc.OK
    assert f["detail"].startswith("0 of 4 scans started in the last 24.0 h") and f["detail"].endswith("3 returned")


def test_tier2_window_shrinks_to_the_rows_held_when_the_answer_is_cut():
    """A cut answer (the probe returns 200 rows at most) covers less than
    the window asked for: timeouts before its oldest row are left out and
    the hours printed are the hours the rows cover."""
    rows = _scan_rows(["2026-11-24T14:00:00.000Z", "2026-11-24T15:00:00.000Z"])
    errors = [_row("2026-11-24T14:30:00.000Z", _T2_TIMEOUT, "TIER2-ERROR")] * 3 \
        + [_row("2026-11-23T19:00:00.000Z", _T2_TIMEOUT, "TIER2-ERROR")] * 5
    f = hc.check_tier2_daemon_timeouts(errors, since=_T2_SINCE, now=_T2_NOW, tier2_entries=rows, limit=len(rows))
    assert f["severity"] == hc.WARN
    assert "3 of 2 scans started in the last 2.0 h" in f["detail"]
    whole = hc.check_tier2_daemon_timeouts(errors, since=_T2_SINCE, now=_T2_NOW, tier2_entries=rows)
    assert "8 of 2 scans started in the last 24.0 h" in whole["detail"]


def test_tier2_no_scan_in_window_and_other_errors_are_said():
    errors = [_row("2026-11-24T09:07:53.000Z", "fetch failed", "TIER2-ERROR")]
    f = hc.check_tier2_daemon_timeouts(errors, since=_T2_SINCE, now=_T2_NOW, scans_started=0, scans_returned=0)
    assert f["severity"] == hc.OK
    assert f["detail"] == "0 daemon timeout(s) and no scan started in the last 24.0 h; 1 scan error(s) of another kind"


def test_tier2_line_says_when_the_scans_were_not_read():
    errors = [_row("2026-11-24T15:00:00.000Z", _T2_TIMEOUT, "TIER2-ERROR")] * 3
    f = hc.check_tier2_daemon_timeouts(errors, since=_T2_SINCE, now=_T2_NOW)
    assert f["severity"] == hc.WARN and "scans started not read" in f["detail"] and "%" not in f["detail"]


def test_gather_tier2_scans_asks_for_counts_only_from_a_server_that_serves_them():
    asked = []

    def new_server(url):
        asked.append(url)
        if "prefix=Starting%20intelligence%20scan" in url:
            return {"probe": "audit", "total": 140, "entries": []}
        if "prefix=Scanned%20" in url:
            return {"probe": "audit", "total": 139, "entries": []}
        return {"probe": "audit", "total": 420, "entries": [_row("t", "x")]}

    got = hc.gather_tier2_scans("http://x", "tok", _T2_SINCE, fetch=new_server)
    assert (got["started"], got["returned"], len(got["entries"])) == (140, 139, 1)
    assert all(f"since={_T2_SINCE}" in u and "type=TIER2&" in u for u in asked) and len(asked) == 3

    asked.clear()

    def old_server(url):
        asked.append(url)
        return {"probe": "audit", "entries": [_row("t", "x")]}

    got = hc.gather_tier2_scans("http://x", "tok", _T2_SINCE, fetch=old_server)
    assert (got["started"], got["returned"]) == (None, None) and len(asked) == 1
    assert hc.gather_tier2_scans("http://x", "tok", _T2_SINCE, fetch=lambda u: None) == \
        {"entries": None, "started": None, "returned": None}


def test_run_all_checks_passes_the_scan_counts_to_the_tier2_line():
    findings = hc.run_all_checks(
        _health(), {"alive": True, "rss_mb": 100}, [], [], {},
        tier2_scans={"entries": [], "started": 12, "returned": 12}, tier2_since=hc.tier2_window_since())
    line = next(f for f in findings if f["label"] == "tier2_daemon_timeouts")
    assert line["detail"].startswith("0 of 12 scans started in the last 24.0 h")


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


# ── check_deploy_freshness (PRODUCTION DEPLOY FREEZE, wishlist.md) ───────

def test_deploy_freshness_ok_when_versions_match():
    f = hc.check_deploy_freshness("1.0.493", "1.0.493")
    assert f["severity"] == hc.OK


def test_deploy_freshness_warn_on_mismatch():
    f = hc.check_deploy_freshness("1.0.493", "1.0.475")
    assert f["severity"] == hc.WARN
    assert "1.0.493" in f["detail"] and "1.0.475" in f["detail"]
    assert "PRODUCTION DEPLOY FREEZE" in f["detail"]


def test_deploy_freshness_warn_when_server_version_missing():
    f = hc.check_deploy_freshness("1.0.493", None)
    assert f["severity"] == hc.WARN
    assert "server_version" in f["detail"]


def test_deploy_freshness_warn_when_local_version_missing():
    f = hc.check_deploy_freshness(None, "1.0.475")
    assert f["severity"] == hc.WARN
    assert "package.json" in f["detail"]


# ── compute_outage_state / check_outage_duration (KNOWN BROKEN #41) ─────

def test_compute_outage_state_reachable_with_no_prior_outage_stays_clear():
    assert hc.compute_outage_state(True, "2026-09-15T00:00:00+00:00", {}) == {"down_since": None}


def test_compute_outage_state_first_detected_down_sets_onset():
    state = hc.compute_outage_state(False, "2026-09-10T20:18:00+00:00", {})
    assert state == {"down_since": "2026-09-10T20:18:00+00:00"}


def test_compute_outage_state_still_down_preserves_original_onset():
    prior = {"down_since": "2026-09-10T20:18:00+00:00"}
    state = hc.compute_outage_state(False, "2026-09-15T02:35:00+00:00", prior)
    assert state == prior


def test_compute_outage_state_recovery_clears_and_records_last_outage():
    prior = {"down_since": "2026-09-10T20:18:00+00:00"}
    state = hc.compute_outage_state(True, "2026-09-15T12:00:00+00:00", prior)
    assert state["down_since"] is None
    assert state["last_recovered_utc"] == "2026-09-15T12:00:00+00:00"
    assert state["last_outage_started_utc"] == "2026-09-10T20:18:00+00:00"


def test_check_outage_duration_ok_when_no_outage():
    f = hc.check_outage_duration({"down_since": None}, "2026-09-15T00:00:00+00:00")
    assert f["severity"] == hc.OK


def test_check_outage_duration_ok_when_state_missing():
    f = hc.check_outage_duration(None, "2026-09-15T00:00:00+00:00")
    assert f["severity"] == hc.OK


def test_check_outage_duration_alarm_reports_hours_and_crosses_amendment_1():
    state = {"down_since": "2026-09-10T20:18:00+00:00"}
    f = hc.check_outage_duration(state, "2026-09-15T02:35:00+00:00")
    assert f["severity"] == hc.ALARM
    assert "102." in f["detail"]  # ~102.3h between the two timestamps
    assert "Amendment 1" in f["detail"]


def test_check_outage_duration_alarm_under_24h_omits_amendment_1_note():
    state = {"down_since": "2026-09-15T00:00:00+00:00"}
    f = hc.check_outage_duration(state, "2026-09-15T01:00:00+00:00")
    assert f["severity"] == hc.ALARM
    assert "Amendment 1" not in f["detail"]


def test_outage_state_roundtrips_through_disk(tmp_path):
    path = str(tmp_path / "outage_state.json")
    hc.save_outage_state({"down_since": "2026-09-10T20:18:00+00:00"}, path=path)
    assert hc.load_outage_state(path=path) == {"down_since": "2026-09-10T20:18:00+00:00"}


def test_load_outage_state_missing_file_returns_empty_dict():
    assert hc.load_outage_state(path="/nonexistent/outage_state.json") == {}


# ── compute_liveness_notify_state / check_liveness_notification ──────────
# (LIVENESS ALARM re-notify policy — the site stays reachable, the loop
# itself is dark, e.g. the 2026-09-10 latched kill switch. Distinct from
# compute_outage_state above, which is for the site being unreachable.)

def test_liveness_notify_state_not_dark_with_no_prior_stays_clear():
    got = hc.compute_liveness_notify_state(False, 0, "2026-09-25T00:00:00+00:00", {})
    assert got == {"dark_since": None, "last_notified_utc": None, "last_notified_wall_hours": None}


def test_liveness_notify_state_first_dark_reading_notifies():
    got = hc.compute_liveness_notify_state(True, 12.8, "2026-09-10T15:00:00+00:00", {})
    assert got == {
        "dark_since": "2026-09-10T15:00:00+00:00",
        "last_notified_utc": "2026-09-10T15:00:00+00:00",
        "last_notified_wall_hours": 12.8,
    }


def test_liveness_notify_state_below_doubling_does_not_renotify():
    prior = {
        "dark_since": "2026-09-10T03:12:26+00:00",
        "last_notified_utc": "2026-09-21T16:06:26+00:00",
        "last_notified_wall_hours": 276.9,
    }
    got = hc.compute_liveness_notify_state(True, 359.5, "2026-09-25T02:37:00+00:00", prior)
    assert got == prior


def test_liveness_notify_state_at_doubling_renotifies():
    prior = {
        "dark_since": "2026-09-10T03:12:26+00:00",
        "last_notified_utc": "2026-09-21T16:06:26+00:00",
        "last_notified_wall_hours": 276.9,
    }
    got = hc.compute_liveness_notify_state(True, 560.0, "2026-10-03T00:00:00+00:00", prior)
    assert got["last_notified_utc"] == "2026-10-03T00:00:00+00:00"
    assert got["last_notified_wall_hours"] == 560.0
    assert got["dark_since"] == prior["dark_since"]


def test_liveness_notify_state_recovery_clears():
    prior = {
        "dark_since": "2026-09-10T03:12:26+00:00",
        "last_notified_utc": "2026-09-21T16:06:26+00:00",
        "last_notified_wall_hours": 276.9,
    }
    got = hc.compute_liveness_notify_state(False, 0, "2026-09-26T00:00:00+00:00", prior)
    assert got == {"dark_since": None, "last_notified_utc": None, "last_notified_wall_hours": None}


def test_check_liveness_notification_ok_when_not_dark():
    f = hc.check_liveness_notification({}, {"dark_since": None}, 0)
    assert f["severity"] == hc.OK


def test_check_liveness_notification_warns_on_fresh_notify():
    prior = {}
    new = {"dark_since": "2026-09-10T15:00:00+00:00", "last_notified_utc": "2026-09-10T15:00:00+00:00",
           "last_notified_wall_hours": 12.8}
    f = hc.check_liveness_notification(prior, new, 12.8)
    assert f["severity"] == hc.WARN
    assert "NOTIFY-WORTHY" in f["detail"]


def test_check_liveness_notification_ok_when_already_notified_this_threshold():
    state = {
        "dark_since": "2026-09-10T03:12:26+00:00",
        "last_notified_utc": "2026-09-21T16:06:26+00:00",
        "last_notified_wall_hours": 276.9,
    }
    f = hc.check_liveness_notification(state, state, 359.5)
    assert f["severity"] == hc.OK
    assert "do not repeat" in f["detail"]


def test_read_local_package_version_reads_real_repo_version():
    # Integration-shaped on purpose: proves the relative path from
    # scripts/session_health_check.py to the repo root's package.json is
    # correct, not just that the parsing logic works on a fixture.
    import json as _json
    repo_root = os.path.join(os.path.dirname(__file__))
    with open(os.path.join(repo_root, "package.json"), encoding="utf-8") as fh:
        expected = _json.load(fh)["version"]
    assert hc.read_local_package_version() == expected


def test_read_local_package_version_missing_file_returns_none():
    assert hc.read_local_package_version(repo_root="/nonexistent/path") is None


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


def test_run_all_checks_returns_ten_findings():
    # 7 -> 8 on 2026-09-16 (check_deploy_gate, KNOWN BROKEN #41); 8 -> 9 same
    # day (check_process_faults, that fix's own queued NEXT step); 9 -> 10 on
    # 2026-11-10 (check_engine_kill, KNOWN BROKEN #49); 10 -> 11 on 2026-11-18
    # (check_restarts, the exit recorder; the test keeps its name).
    findings = hc.run_all_checks(_health(), {"alive": True, "rss_mb": 100}, [], [], {})
    assert len(findings) == 11
    assert all("severity" in f and "label" in f and "detail" in f for f in findings)
    assert {f["label"] for f in findings} >= {"deploy_freshness", "deploy_gate", "process_faults", "restarts"}
    assert findings[0]["label"] == "deploy_gate", "the deploy gate is the first line — a 503 there explains every other finding"


# ── check_deploy_gate (KNOWN BROKEN #41: the 503 that vetoed 30 deploys) ─

def _serving(ok=True, failing=None):
    return {"ok": ok, "gates": ["server", "database"], "failing": failing or [], "note": "n"}


def test_deploy_gate_ok_on_2xx_with_serving_block():
    h = _health(); h["serving"] = _serving()
    f = hc.check_deploy_gate(h)
    assert f["severity"] == hc.OK
    assert "would pass" in f["detail"]


def test_deploy_gate_the_sept10_case_pre_healthgate_build_alarms_and_names_every_degraded_check():
    # What production answered from 2026-09-10T15:30Z until the fix: HTTP 503,
    # status degraded, liveness dark, NO `serving` block (old handler). The
    # old _fetch_json dropped this as "fetch failed"; now it is the first
    # ALARM and it says what gates the deploy.
    h = _health(dark=True, status="degraded", detail="LIVENESS ALARM: trading loop dark for 26.0 market hours")
    h["checks"]["bot"]["status"] = "killed"
    h["_http_status"] = 503
    f = hc.check_deploy_gate(h)
    assert f["severity"] == hc.ALARM
    assert "503" in f["detail"]
    assert "pre-healthGate.ts" in f["detail"]
    assert "bot" in f["detail"], "the degraded check must be named"
    assert "rejected by Railway" in f["detail"]


def test_deploy_gate_503_with_serving_block_names_the_failing_gate():
    h = _health(status="degraded", bad_subsystem="database")
    h["_http_status"] = 503
    h["serving"] = _serving(ok=False, failing=["database"])
    f = hc.check_deploy_gate(h)
    assert f["severity"] == hc.ALARM
    assert "database" in f["detail"]
    assert "REJECT" in f["detail"]


def test_deploy_gate_degraded_but_2xx_is_ok_alarms_stay_with_their_own_checks():
    # The post-fix contract: liveness dark, status degraded, HTTP 200 —
    # the alarm is check_liveness's job; the gate is fine.
    h = _health(dark=True, status="degraded", detail="dark")
    h["serving"] = _serving()
    assert hc.check_deploy_gate(h)["severity"] == hc.OK
    assert hc.check_liveness(h)["severity"] == hc.ALARM


def test_deploy_gate_2xx_without_serving_block_warns_old_build():
    f = hc.check_deploy_gate(_health())
    assert f["severity"] == hc.WARN
    assert "pre-healthGate.ts" in f["detail"]


def test_deploy_gate_no_payload_alarms():
    assert hc.check_deploy_gate(None)["severity"] == hc.ALARM


def test_deploy_gate_contradiction_serving_false_with_2xx_alarms():
    h = _health(); h["serving"] = _serving(ok=False, failing=["server"])
    f = hc.check_deploy_gate(h)
    assert f["severity"] == hc.ALARM
    assert "contradiction" in f["detail"]


# ── _fetch_json keeps an HTTP error's JSON body when it is a health payload ─

class _FakeHTTPError(hc.urllib.error.HTTPError):
    def __init__(self, code, body):
        super().__init__("http://x/api/health", code, "err", {}, None)
        self._body = body

    def read(self):
        return self._body


def test_fetch_json_keeps_503_health_payload_with_status(monkeypatch):
    payload = b'{"status":"degraded","checks":{"server":{"status":"ok"}}}'

    def fake_urlopen(url, timeout=0):
        raise _FakeHTTPError(503, payload)

    monkeypatch.setattr(hc.urllib.request, "urlopen", fake_urlopen)
    got = hc._fetch_json("http://x/api/health")
    assert got is not None
    assert got["_http_status"] == 503
    assert got["status"] == "degraded"


def test_fetch_json_drops_railway_edge_fallback_502(monkeypatch):
    # Railway's own 502 page: {"status":"error","code":502,"message":...} — no
    # `checks`, so it is NOT a health payload and must stay None (the outage
    # tracker keys off None).
    def fake_urlopen(url, timeout=0):
        raise _FakeHTTPError(502, b'{"status":"error","code":502,"message":"Application failed to respond"}')

    monkeypatch.setattr(hc.urllib.request, "urlopen", fake_urlopen)
    assert hc._fetch_json("http://x/api/health") is None


def test_fetch_json_drops_non_json_error_bodies(monkeypatch):
    def fake_urlopen(url, timeout=0):
        raise _FakeHTTPError(500, b"<html>nope</html>")

    monkeypatch.setattr(hc.urllib.request, "urlopen", fake_urlopen)
    assert hc._fetch_json("http://x/api/health") is None


def test_run_all_checks_end_to_end_reproduces_live_known_broken_21_snapshot():
    # Mirrors this session's live read (2026-07-14 19:57Z window, pre-v1.0.314
    # confirmation): all-down DIAGNOSTIC entries + daemon rss under the new
    # 550MB threshold. Expect a WARN (not ALARM) — the memory fix already
    # shipped and daemon rss looks fine; the enrichment blind spot is the
    # thing still worth a human/future-session look.
    entries = [_diag_entry(DOWN_MSG) for _ in range(8)]
    findings = hc.run_all_checks(
        _health(), {"alive": True, "rss_mb": 165.6}, entries, [], {"feedback_live_count": 0},
        local_version="1.0.493", server_version="1.0.493",
    )
    by_label = {f["label"]: f for f in findings}
    assert by_label["alt_data_enrichment"]["severity"] == hc.WARN
    assert by_label["daemon_memory"]["severity"] == hc.OK
    assert by_label["deploy_freshness"]["severity"] == hc.OK
    assert hc.overall_exit_code(findings) == 1


def test_run_all_checks_end_to_end_reproduces_live_deploy_freeze_snapshot():
    # Mirrors this session's live read (2026-07-25): main package.json at
    # 1.0.493, live server_version at 1.0.475 — the PRODUCTION DEPLOY
    # FREEZE tracked in wishlist.md. Everything else healthy, so this
    # finding alone should be the thing that trips exit code 1.
    findings = hc.run_all_checks(
        _health(), {"alive": True, "rss_mb": 165.6}, [], [], {"feedback_live_count": 0},
        local_version="1.0.493", server_version="1.0.475",
    )
    by_label = {f["label"]: f for f in findings}
    assert by_label["deploy_freshness"]["severity"] == hc.WARN
    assert hc.overall_exit_code(findings) == 1


# ── compute_outage_state keeps the last outage's record (found 2026-09-18) ─

def test_outage_state_healthy_run_preserves_last_outage_record():
    prior = {"down_since": None, "last_outage_started_utc": "2026-09-10T20:18:00+00:00",
             "last_recovered_utc": "2026-09-16T02:40:06+00:00"}
    got = hc.compute_outage_state(True, "2026-09-18T13:08:00+00:00", prior)
    assert got["down_since"] is None
    assert got["last_outage_started_utc"] == "2026-09-10T20:18:00+00:00"
    assert got["last_recovered_utc"] == "2026-09-16T02:40:06+00:00"


def test_outage_state_new_outage_keeps_previous_record_until_recovery():
    prior = {"down_since": None, "last_outage_started_utc": "2026-09-10T20:18:00+00:00",
             "last_recovered_utc": "2026-09-16T02:40:06+00:00"}
    down = hc.compute_outage_state(False, "2026-09-20T10:00:00+00:00", prior)
    assert down["down_since"] == "2026-09-20T10:00:00+00:00"
    assert down["last_recovered_utc"] == "2026-09-16T02:40:06+00:00"
    still = hc.compute_outage_state(False, "2026-09-20T11:00:00+00:00", down)
    assert still == down
    back = hc.compute_outage_state(True, "2026-09-20T12:00:00+00:00", still)
    assert back == {"down_since": None, "last_outage_started_utc": "2026-09-20T10:00:00+00:00",
                    "last_recovered_utc": "2026-09-20T12:00:00+00:00"}


def test_outage_state_fresh_install_has_no_history_keys():
    assert hc.compute_outage_state(True, "2026-09-18T13:08:00+00:00", None) == {"down_since": None}
    assert hc.compute_outage_state(False, "2026-09-18T13:08:00+00:00", {}) == {"down_since": "2026-09-18T13:08:00+00:00"}


# ── data-plane ops checks (2026-10-14, ops item 4) ───────────────────────

NOW_MS = 1_792_000_000_000


def _plan_status(connected=True, last_err=None, silent_min=0.1, routes=22725, placed=21258, errs=0, dropped=0):
    return {
        "swimConfigured": True, "swimConnected": connected,
        "swim": {
            "messagesReceived": 163402, "messagesProcessed": 163007, "processErrors": errs, "parseErrors": 0,
            "droppedOverflow": dropped, "lastMessageAt": NOW_MS - silent_min * 60000, "lastError": last_err,
            "stallRestarts": 0,
            "routeShape": {"parserShadow": {"routes": routes, "placed": placed, "withUnresolved": 300}},
        },
    }


def test_swim_ok_reports_rate_since_boot():
    f = hc.check_swim(_plan_status(), uptime_s=1896, now_ms=NOW_MS)
    assert f["severity"] == hc.OK
    assert "310,257 msg/h" in f["detail"] and "0 watchdog restarts" in f["detail"]


def test_swim_disconnected_auth_error_names_the_password_rotation():
    f = hc.check_swim(_plan_status(connected=False, last_err="session connect failed: Unauthorized (401)"), now_ms=NOW_MS)
    assert f["severity"] == hc.ALARM
    assert "SWIM_PASSWORD" in f["fix"] and "swimConnected=true" in f["fix"]
    g = hc.check_swim(_plan_status(connected=False, last_err="session disconnected"), now_ms=NOW_MS)
    assert g["severity"] == hc.ALARM and "retries on its own" in g["fix"]


def test_swim_connected_but_silent_is_an_alarm():
    f = hc.check_swim(_plan_status(silent_min=40), now_ms=NOW_MS)
    assert f["severity"] == hc.ALARM and "silent for 40 min" in f["detail"]


def test_swim_unconfigured_and_unreachable():
    assert hc.check_swim(None)["severity"] == hc.WARN
    assert hc.check_swim({"swimConfigured": False})["severity"] == hc.WARN


def test_swim_parse_errors_and_overflow_warn_above_1pct():
    assert hc.check_swim(_plan_status(errs=5000), uptime_s=1896, now_ms=NOW_MS)["severity"] == hc.WARN
    assert hc.check_swim(_plan_status(dropped=5000), uptime_s=1896, now_ms=NOW_MS)["severity"] == hc.WARN
    assert hc.check_swim(_plan_status(dropped=395), uptime_s=1896, now_ms=NOW_MS)["severity"] == hc.OK


def test_filed_routes_share_and_small_samples():
    assert hc.check_filed_routes(_plan_status())["severity"] == hc.OK
    assert hc.check_filed_routes(_plan_status(routes=50, placed=1))["severity"] == hc.OK  # too few to judge
    f = hc.check_filed_routes(_plan_status(routes=1000, placed=500))
    assert f["severity"] == hc.WARN and "NASR" in f["fix"]


def _cov(n=3086, mcp=58.5, rejected=None, rows=9346):
    return {"aircraft": n, "rows_ingested": rows, "rejected": rejected or {"oat": 2},
            "latest_pct": {"nav_altitude_mcp": mcp, "nav_qnh": 60.4, "nav_modes": 24.4, "wd": 4.2},
            "archive": {"daily": {"days": 3, "latest": "2026-10-16"}}}


def test_intent_coverage():
    f = hc.check_intent_coverage(_cov())
    assert f["severity"] == hc.OK and "daily rollups 3" in f["detail"]
    assert hc.check_intent_coverage(_cov(mcp=5))["severity"] == hc.WARN
    assert hc.check_intent_coverage(_cov(n=100, mcp=5))["severity"] == hc.OK  # sample too small
    assert hc.check_intent_coverage(_cov(rejected={"oat": 500}))["severity"] == hc.WARN
    assert hc.check_intent_coverage(None)["severity"] == hc.WARN


def _offload(free, err=None):
    return {"volume": {"freeBytes": free}, "estimatedBytesPerDay": 64_674_900, "lastError": err}


def test_storage_thresholds_and_auto_pause_note():
    f = hc.check_storage(_offload(1_624_317_952))
    assert f["severity"] == hc.OK and "auto-paused" in f["detail"] and "d to the 1 GiB guard" in f["detail"]
    assert hc.check_storage(_offload(1_100_000_000))["severity"] == hc.ALARM
    w = hc.check_storage(_offload(3 * hc.GIB, err="PUT 403"))
    assert w["severity"] == hc.WARN and "R2_" in w["fix"]
    assert "auto-paused" not in hc.check_storage(_offload(3 * hc.GIB))["detail"]


def _gov(bg_n=93, r429=38, fg_n=40, fg_ok=38, rps=0.4, eff=None):
    return {"bg_rps": rps, **({"effective_bg_rps": eff} if eff is not None else {}), "last_hour": {
        "bg": {"results": bg_n, "ok": bg_n - r429, "rate_limited": r429},
        "fg": {"results": fg_n, "ok": fg_ok}}}


def test_adsb_upstream_429_share_names_the_env_fix():
    f = hc.check_adsb_upstream(_gov())
    assert f["severity"] == hc.WARN and "GLOBAL_SWEEP_RPS=0.2" in f["fix"]
    assert hc.check_adsb_upstream(_gov(r429=5))["severity"] == hc.OK
    assert hc.check_adsb_upstream(_gov(r429=5, fg_ok=10))["severity"] == hc.WARN  # 75% failover
    assert hc.check_adsb_upstream(None)["severity"] == hc.OK  # older build: nothing to judge


def test_latency_p95():
    assert hc.p95([100, 200, 300, 400, 5000]) == 5000
    assert hc.check_data_latency([120] * 10)["severity"] == hc.OK
    assert hc.check_data_latency([120] * 9 + [9000])["severity"] == hc.WARN
    assert hc.check_data_latency([])["severity"] == hc.WARN


def test_alert_message_is_silent_when_all_ok_and_one_message_otherwise():
    ok = hc.run_ops_checks(_plan_status(), _cov(), _offload(3 * hc.GIB), _gov(r429=0), [100] * 10,
                           uptime_s=1896, now_ms=NOW_MS)
    assert all(f["severity"] == hc.OK for f in ok)
    assert hc.alert_message(ok) is None
    bad = hc.run_ops_checks(_plan_status(connected=False, last_err="login failed"), _cov(), _offload(3 * hc.GIB),
                            _gov(), [100] * 10, now_ms=NOW_MS)
    msg = hc.alert_message(bad)
    assert msg.startswith("VolTradeAI ops: 2 item(s) need action")
    assert msg.index("[ALARM]") < msg.index("[WARN]"), "worst first"
    assert "FIX:" in msg


def test_adsb_upstream_self_pacing_is_not_a_human_step_until_the_floor():
    f = hc.check_adsb_upstream(_gov(eff=0.1))
    assert f["severity"] == hc.OK and "self-pacing at 0.1 of 0.4" in f["detail"]
    g = hc.check_adsb_upstream(_gov(eff=0.04))
    assert g["severity"] == hc.WARN and "GLOBAL_SWEEP_RPS=0.2" in g["fix"] and "floor" in g["detail"]


# ── 2026-11-16: a starved sweep passed this check while the worldwide snapshot aged out ──
# Read live that day from /api/data/aircraft/global: the sweep sent 59 requests in the hour (47 ok, 5
# rate-limited) where its self-paced 0.19 req/s allowed ~684; backoff_s 885; 86 consecutive failures.
def _gov_1116(**over):
    g = {"bg_rps": 0.4, "effective_bg_rps": 0.19, "backoff_s": 885, "consecutive_failures": 86, "last_hour": {
        "bg": {"results": 59, "ok": 47, "rate_limited": 5, "network_error": 7},
        "fg": {"results": 191, "ok": 110, "rate_limited": 49, "network_error": 32}}}
    g.update(over)
    return g


def test_a_starved_sweep_is_a_warning_once_the_container_has_a_full_hour():
    f = hc.check_adsb_upstream(_gov_1116(), uptime_s=3 * 3600)
    assert f["severity"] == hc.WARN
    assert "sweep STARVED: 59 requests in the last hour where its pace of 0.19 req/s allows ~684" in f["detail"]
    assert "held 885 s, 86 consecutive failures" in f["detail"]
    assert "viewer_pause_s" in f["fix"] and "/api/data/aircraft/global" in f["fix"]
    # 25% of 684 is 171: at 171 requests it is not starved, at 170 it is
    assert hc.SWEEP_STARVED_SHARE == 0.25
    ok = _gov_1116(); ok["last_hour"]["bg"].update(results=171, ok=171, rate_limited=0)
    assert "STARVED" not in hc.check_adsb_upstream(ok, uptime_s=3 * 3600)["detail"]
    low = _gov_1116(); low["last_hour"]["bg"].update(results=170, ok=170, rate_limited=0)
    assert "STARVED" in hc.check_adsb_upstream(low, uptime_s=3 * 3600)["detail"]


def test_starvation_is_not_judged_without_a_full_hour_or_without_a_pace():
    assert "STARVED" not in hc.check_adsb_upstream(_gov_1116(), uptime_s=3599)["detail"]
    assert "STARVED" not in hc.check_adsb_upstream(_gov_1116())["detail"], "no uptime given: the older behaviour"
    off = _gov_1116(bg_rps=0, effective_bg_rps=0)
    assert "STARVED" not in hc.check_adsb_upstream(off, uptime_s=3 * 3600)["detail"], "a sweep that is switched off is not starved"
    # without the self-paced figure the configured rate is the pace: 0.4 req/s allows 1,440
    older = _gov_1116(); del older["effective_bg_rps"]
    assert "pace of 0.4 req/s allows ~1440" in hc.check_adsb_upstream(older, uptime_s=3 * 3600)["detail"]


def test_run_ops_checks_passes_the_uptime_to_the_upstream_check():
    args = (_plan_status(), _cov(), _offload(3 * hc.GIB), _gov_1116(), [100] * 10)
    late = [f for f in hc.run_ops_checks(*args, uptime_s=3 * 3600, now_ms=NOW_MS) if f["label"] == "adsb.lol upstream"][0]
    assert late["severity"] == hc.WARN and "STARVED" in late["detail"]
    young = [f for f in hc.run_ops_checks(*args, uptime_s=600, now_ms=NOW_MS) if f["label"] == "adsb.lol upstream"][0]
    assert "STARVED" not in young["detail"]


# ── adsb.lol's daily quota (2026-11-17): a sweep that does not ask adsb.lol by design is not starved ──

def _quota_gov(**quota):
    g = _gov(bg_n=1, r429=1, fg_n=3, fg_ok=0)   # the live reading after a boot: 1 sweep request, the quota's 429
    g["effective_bg_rps"] = 0.19
    g["quota"] = {"limit": 500, "remaining": 0, "exhausted": True, "mode": "watched_only", "key_configured": False, **quota}
    return g


def test_adsb_upstream_quota_spent_or_small_is_not_starved_and_not_a_429_warning():
    # live 2026-11-17 23:26Z: limit 500, remaining 0, watched_only, no key; uptime over an hour
    f = hc.check_adsb_upstream(_quota_gov(), uptime_s=3 * 3600)
    assert f["severity"] == hc.OK, f
    assert "STARVED" not in f["detail"]
    assert "500/day" in f["detail"] and "spent until its reset" in f["detail"]
    assert "ADSBLOL_API_KEY" in f["detail"]
    # the quota not spent yet today, still small: the same answer, worded for what it is
    g = hc.check_adsb_upstream(_quota_gov(remaining=312, exhausted=False), uptime_s=3 * 3600)
    assert g["severity"] == hc.OK and "small: watched planes only" in g["detail"] and "312 left" in g["detail"]


def test_adsb_upstream_a_configured_key_with_a_small_or_spent_quota_is_a_warning():
    f = hc.check_adsb_upstream(_quota_gov(key_configured=True), uptime_s=3 * 3600)
    assert f["severity"] == hc.WARN
    assert "key" in f["fix"].lower()


def test_adsb_upstream_full_quota_is_judged_as_before_including_starved():
    # a large quota (a key), mode full: the starved rule still applies
    g = _quota_gov(limit=7200, remaining=5000, exhausted=False, mode="full", key_configured=True)
    f = hc.check_adsb_upstream(g, uptime_s=3 * 3600)
    assert f["severity"] == hc.WARN and "STARVED" in f["detail"]
    # and a build without the quota block (older server) is judged exactly as before
    old = _gov(bg_n=1, r429=1, fg_n=3, fg_ok=0)
    old["effective_bg_rps"] = 0.19
    assert "STARVED" in hc.check_adsb_upstream(old, uptime_s=3 * 3600)["detail"]


# ── EDGAR archives (2026-11-17, open_questions.md 2026-11-16 item (2)) ────
from datetime import date as _date, datetime as _dt, timezone as _tz  # noqa: E402


def _edgar_probe(form4=None, f13=None, e8k=None):
    def row(stream, over):
        base = {"stream": stream, "polls_ok": 6, "polls_failed": 0, "consecutive_failures": 0,
                "last_ok_at": None, "last_failure_at": None, "last_error": None, "rows_archived": 40,
                "last_archive_write_at": None}
        base.update(over or {})
        return base
    return {"probe": "edgar_poll", "streams": [
        row("filings", form4), row("filings13f", f13), row("earnings8k", e8k)]}


def _utc(s):
    return _dt.fromisoformat(s).replace(tzinfo=_tz.utc)


_MON_1116_FAILING = {
    "last_archive_write_at": "2026-11-16T12:40:00.000Z",  # 07:40 ET, the last row of that Monday
    "polls_ok": 0, "polls_failed": 9, "consecutive_failures": 9,
    "last_error": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&count=75&output=atom -> 400",
}


def test_edgar_monday_1116_alarms_at_each_session_that_ran_after_11_et():
    # The sessions of 2026-11-16 ran at 13:05Z, 16:11Z, 18:07Z, 20:16Z and 00:10Z.
    probe = _edgar_probe(form4=_MON_1116_FAILING)
    early = hc.check_edgar_archives(probe, now=_utc("2026-11-16T13:05:00"))
    assert early["severity"] == hc.WARN  # 08:05 ET: 5 minutes of window, but 9 failed polls are named
    assert "9 consecutive failed polls" in early["detail"] and "-> 400" in early["detail"]
    for at in ("2026-11-16T16:11:00", "2026-11-16T18:07:00", "2026-11-16T20:16:00", "2026-11-17T00:10:00"):
        f = hc.check_edgar_archives(probe, now=_utc(at))
        assert f["severity"] == hc.ALARM, at
        assert "Form 4: no row archived for" in f["detail"] and "2026-11-16" in f["detail"]
        assert "-> 400" in f["detail"]
    # 00:19Z on 11-17 = 19:19 ET: "11.6 h without a file write" was the registry's reading; in-window it is 11.3 h
    f = hc.check_edgar_archives(probe, now=_utc("2026-11-17T00:19:00"))
    assert "no row archived for 11.3 h" in f["detail"]


def test_edgar_quiet_hours_are_counted_inside_the_window_only():
    day = _date(2026, 11, 16)
    # written at 07:40 ET, read at 11:11 ET: quiet since 08:00, not since 07:40
    d, q = hc.edgar_quiet_hours(_utc("2026-11-16T12:40:00"), _utc("2026-11-16T16:11:00"))
    assert d == day and abs(q - (3 + 11 / 60)) < 0.01
    # read at 23:00 ET: the window closed at 20:00
    d, q = hc.edgar_quiet_hours(_utc("2026-11-16T12:40:00"), _utc("2026-11-17T04:00:00"))
    assert d == day and abs(q - 12.0) < 0.01
    # a write after the window closed clears it (the fix deployed that evening)
    d, q = hc.edgar_quiet_hours(_utc("2026-11-17T01:30:00"), _utc("2026-11-17T04:00:00"))
    assert d == day and q == 0.0
    # before 08:00 ET on Tuesday the verdict is still about Monday
    d, q = hc.edgar_quiet_hours(_utc("2026-11-16T12:40:00"), _utc("2026-11-17T12:00:00"))
    assert d == day and abs(q - 12.0) < 0.01
    # no plain day file at all: the whole window so far is quiet
    d, q = hc.edgar_quiet_hours(None, _utc("2026-11-16T18:00:00"))
    assert d == day and abs(q - 5.0) < 0.01


def test_edgar_healthy_day_weekend_and_overnight_are_ok():
    ok = _edgar_probe(form4={"last_archive_write_at": "2026-11-17T18:55:00.000Z"})
    f = hc.check_edgar_archives(ok, now=_utc("2026-11-17T19:05:00"))
    assert f["severity"] == hc.OK and "Form 4 last row 0.2 h ago" in f["detail"]
    # Saturday afternoon, last row Friday 19:50 ET: Friday's window ended 10 minutes quiet
    wk = _edgar_probe(form4={"last_archive_write_at": "2026-11-14T00:50:00.000Z"})
    assert hc.check_edgar_archives(wk, now=_utc("2026-11-14T19:00:00"))["severity"] == hc.OK
    # 02:00 ET Tuesday after a normal Monday
    assert hc.check_edgar_archives(
        _edgar_probe(form4={"last_archive_write_at": "2026-11-17T00:45:00.000Z"}),
        now=_utc("2026-11-17T07:00:00"))["severity"] == hc.OK
    # 2.9 h quiet inside the window is below the line
    f = hc.check_edgar_archives(
        _edgar_probe(form4={"last_archive_write_at": "2026-11-17T15:00:00.000Z"}), now=_utc("2026-11-17T17:54:00"))
    assert f["severity"] == hc.OK


def test_edgar_days_with_no_filings_expected_are_skipped():
    assert hc.edgar_filing_day(_date(2026, 11, 16)) is True
    assert hc.edgar_filing_day(_date(2026, 11, 14)) is False          # Saturday
    assert hc.edgar_filing_day(_date(2026, 11, 26)) is False          # Thanksgiving (NYSE calendar)
    assert hc.edgar_filing_day(_date(2026, 11, 11)) is False          # Veterans Day: EDGAR closed, NYSE open
    assert hc.edgar_filing_day(_date(2026, 10, 12)) is False          # Columbus Day
    assert hc._edgar_closed_nyse_open(_date(2028, 11, 10)) is True    # 11-11-2028 is a Saturday: observed Friday
    assert hc._edgar_closed_nyse_open(_date(2028, 11, 11)) is False
    assert hc._edgar_closed_nyse_open(_date(2029, 11, 12)) is True    # 11-11-2029 is a Sunday: observed Monday
    assert hc._edgar_closed_nyse_open(_date(2026, 10, 5)) is False    # first Monday of October
    # Veterans Day itself: nothing written all day, polls fine, read at 15:00 ET -> judged on Tuesday 11-10
    f = hc.check_edgar_archives(
        _edgar_probe(form4={"last_archive_write_at": "2026-11-11T00:40:00.000Z"}), now=_utc("2026-11-11T20:00:00"))
    assert f["severity"] == hc.OK


def test_edgar_silent_with_healthy_polls_still_alarms_and_says_so():
    probe = _edgar_probe(form4={"last_archive_write_at": "2026-11-17T13:10:00.000Z", "polls_ok": 20})
    f = hc.check_edgar_archives(probe, now=_utc("2026-11-17T19:00:00"))
    assert f["severity"] == hc.ALARM and "polls succeed but archive nothing" in f["detail"]
    # a container that has not completed one good poll
    probe = _edgar_probe(form4={"last_archive_write_at": "2026-11-17T13:10:00.000Z", "polls_ok": 0})
    f = hc.check_edgar_archives(probe, now=_utc("2026-11-17T19:00:00"))
    assert f["severity"] == hc.ALARM and "no poll has succeeded since boot" in f["detail"]


def test_edgar_13f_and_8k_failures_warn_then_alarm_and_one_failure_is_noise():
    fresh = {"last_archive_write_at": "2026-11-17T18:55:00.000Z"}
    now = _utc("2026-11-17T19:05:00")
    one = hc.check_edgar_archives(_edgar_probe(form4=fresh, f13={"consecutive_failures": 1, "last_error": "x -> 503"}), now=now)
    assert one["severity"] == hc.OK
    two = hc.check_edgar_archives(_edgar_probe(form4=fresh, f13={"consecutive_failures": 2, "last_error": "x -> 503"}), now=now)
    assert two["severity"] == hc.WARN and "13F: 2 consecutive failed polls, last error: x -> 503" in two["detail"]
    many = hc.check_edgar_archives(_edgar_probe(form4=fresh, e8k={"consecutive_failures": 12, "last_error": "y -> 400"}), now=now)
    assert many["severity"] == hc.ALARM and "8-K earnings: 12 consecutive failed polls" in many["detail"]
    assert many.get("fix")


def test_edgar_probe_missing_or_misshapen_is_a_warning_not_a_crash():
    for bad in (None, {}, {"streams": "no"}, []):
        assert hc.check_edgar_archives(bad, now=_utc("2026-11-17T19:05:00"))["severity"] == hc.WARN
    part = {"streams": [{"stream": "filings", "last_archive_write_at": "not a date", "polls_ok": 3}]}
    f = hc.check_edgar_archives(part, now=_utc("2026-11-17T13:30:00"))  # 08:30 ET: half an hour of window
    assert f["severity"] == hc.WARN and "13F: not in the probe" in f["detail"]
    # an unreadable write time counts as no write: by 11:30 ET that is an alarm
    assert hc.check_edgar_archives(part, now=_utc("2026-11-17T16:30:00"))["severity"] == hc.ALARM


def test_edgar_polls_saved_by_a_feed_retry_are_named_and_an_older_server_reads_as_before():
    # [REPAIR 2026-11-19] server/edgarFeedFetch.ts retries the feed request; the
    # probe counts the polls that needed it. An older server has no such key.
    saved = _edgar_probe(form4={"last_archive_write_at": "2026-11-17T18:55:00.000Z",
                                "polls_ok": 40, "polls_saved_by_retry": 9, "feed_attempts_failed": 10})
    f = hc.check_edgar_archives(saved, now=_utc("2026-11-17T19:05:00"))
    assert f["severity"] == hc.OK
    assert "Form 4 last row 0.2 h ago, polls 40 ok / 0 failed (9 of the ok saved by a feed retry)" in f["detail"]
    assert "13F last row none in the day files, polls 6 ok / 0 failed;" in f["detail"]  # no key, no note
    none = _edgar_probe(form4={"last_archive_write_at": "2026-11-17T18:55:00.000Z", "polls_saved_by_retry": 0})
    assert "saved by a feed retry" not in hc.check_edgar_archives(none, now=_utc("2026-11-17T19:05:00"))["detail"]


# ── polled archives gone quiet (2026-11-20) ──────────────────────────────
def _stream(name, health, newest, age_hours, note=None):
    return {"stream": name, "health": health, "newest_file": newest, "age_hours": age_hours, "health_note": note}


def test_quiet_archive_is_an_alarm_the_day_it_goes_stale():
    """The live inventory of 2026-11-20: fires had been stale since 11-04 and no check said so."""
    inv = {"streams": [
        _stream("aircraft", "live", "2026-11-20-18.jsonl", 0),
        _stream("fires", "stale", "2026-11-02.jsonl.gz", 402.0,
                "newest file 402.0h old — exceeds cadence-derived threshold ~26h"),
        _stream("airquality", "stale", "2026-10-06.jsonl.gz", 1025.9),
        _stream("cropconditions", "stale", "2026-10-26.jsonl.gz", 569.9),
        _stream("eiaweekly", "stale", "datacore/eia/weekly_series.json", 436.6),
    ]}
    f = hc.check_quiet_archives(inv)
    assert f["severity"] == hc.ALARM
    assert "fires: newest file 2026-11-02.jsonl.gz, 16.8 d old" in f["detail"]
    assert "exceeds cadence-derived threshold ~26h" in f["detail"]
    # a bundled snapshot is another check's (research_state_check.py); a known-quiet stream is named with its reason
    assert "eiaweekly" not in f["detail"]
    assert "airquality (Google Air Quality API is not enabled" in f["detail"]
    assert "never refills" in f["fix"]


def test_quiet_archives_known_reasons_alone_do_not_alarm():
    inv = {"streams": [
        _stream("aircraft", "live", "2026-11-20-18.jsonl", 0),
        _stream("fires", "live", "2026-11-20.jsonl", 0.2),
        _stream("airquality", "stale", "2026-10-06.jsonl.gz", 1025.9),
        _stream("eiaweekly", "stale", "datacore/eia/weekly_series.json", 436.6),
    ]}
    f = hc.check_quiet_archives(inv)
    assert f["severity"] == hc.OK
    assert f["detail"].startswith("2 of 3 polled streams inside their cadence")
    assert "airquality" in f["detail"]


def test_quiet_archives_unreadable_inventory_is_a_warning_not_ok():
    assert hc.check_quiet_archives(None)["severity"] == hc.WARN
    assert hc.check_quiet_archives({"streams": []})["severity"] == hc.WARN
    assert hc.check_quiet_archives({"warming_up": True})["severity"] == hc.WARN


# ── a silent source is not a dead poller (2026-11-27, open_questions.md (av5)) ──
_NOW_MS = 1795796400000  # 2026-11-27T16:20:00Z


def _gen_stale():
    return _stream("gridgeneration", "stale", "2026-11-25.jsonl", 51.0,
                   "newest file 51.0h old — exceeds cadence-derived threshold ~26h")


def _poll_route(**over):
    """The route as production served it on 2026-11-27 16:11Z, 47 minutes after its poll."""
    poll = {"at": _NOW_MS - 47 * 60_000, "newest_period": "2026-11-25T07", "rows_archived": 0,
            "first_failed": [], "catch_up": []}
    poll.update(over)
    return {"kind": "raw", "count": 12, "poll": poll}


def _quiet(route, stream=None):
    s = stream or _gen_stale()
    inv = {"streams": [_stream("aircraft", "live", "2026-11-27-16.jsonl", 0), s]}
    return hc.check_quiet_archives(inv, {s["stream"]: route}, now_ms=_NOW_MS)


def test_silent_source_with_a_healthy_poller_is_a_warning_that_names_the_source():
    """The measured case: EIA published nothing for 57 hours; the poller read 12 of 12 every two hours."""
    f = _quiet(_poll_route())
    assert f["severity"] == hc.WARN
    assert "gridgeneration: the source is silent, not the poller" in f["detail"]
    assert "EIA's newest hour is 2026-11-25T07 UTC, 57 h behind" in f["detail"]
    assert "last poll 47 min ago read every respondent" in f["detail"]
    assert "newest file 2026-11-25.jsonl, 2.1 d old" in f["detail"]
    assert "nothing to repair on our side" in f["fix"]


def test_same_stale_file_without_the_route_is_the_alarm_it_was():
    inv = {"streams": [_gen_stale()]}
    for routes in (None, {}):
        f = hc.check_quiet_archives(inv, routes, now_ms=_NOW_MS)
        assert f["severity"] == hc.ALARM
        assert "gridgeneration: newest file 2026-11-25.jsonl, 2.1 d old" in f["detail"]


def test_each_line_that_fails_keeps_the_alarm_and_says_why():
    cases = [
        (None, "served no poll state"),
        ({"warming_up": True, "count": 0}, "served no poll state"),
        ({"poll": None}, "served no poll state"),
        (_poll_route(at=None), "has no time"),
        (_poll_route(at=_NOW_MS - 6 * 3_600_000), "the poller last ran 6.0 h ago"),
        (_poll_route(first_failed=["PJM", "ERCO"]), "could not read 2 respondent(s): PJM, ERCO"),
        (_poll_route(first_failed=None), "does not say which respondents were read"),
        (_poll_route(catch_up=[{"respondent": "NW", "outcome": "capped", "extra_pages": 10, "oldest_read": "2026-11-01T00"}]),
         "did not reach the archive for NW (capped)"),
        (_poll_route(catch_up=[{"respondent": "NW", "outcome": "failed", "extra_pages": 1, "oldest_read": None}]),
         "did not reach the archive for NW (failed)"),
        (_poll_route(newest_period=None), "EIA returned no hour at all"),
        # the source is publishing and the archive is a day behind it: the fires break's shape
        (_poll_route(newest_period="2026-11-27T15"), "published hours are not being archived"),
    ]
    for route, why in cases:
        f = _quiet(route)
        assert f["severity"] == hc.ALARM, why
        assert why in f["detail"], f["detail"]
        assert "never refills" in f["fix"]


def test_a_read_back_that_joined_the_archive_is_not_a_hole():
    f = _quiet(_poll_route(catch_up=[{"respondent": "NW", "outcome": "joined", "extra_pages": 1, "oldest_read": "2026-11-22T00"}]))
    assert f["severity"] == hc.WARN


def test_poll_age_boundary_and_a_silence_near_the_read_backs_reach():
    assert _quiet(_poll_route(at=_NOW_MS - int(4.9 * 3_600_000)))["severity"] == hc.WARN
    assert _quiet(_poll_route(at=_NOW_MS - int(5.1 * 3_600_000)))["severity"] == hc.ALARM
    old = _stream("gridgeneration", "stale", "2026-11-06.jsonl.gz", 500.0)
    f = _quiet(_poll_route(newest_period="2026-11-06T23"), old)        # 497 h: past 480 h
    assert f["severity"] == hc.ALARM and "nearing the reach" in f["detail"]
    ok = _stream("gridgeneration", "stale", "2026-11-08.jsonl.gz", 450.0)
    assert _quiet(_poll_route(newest_period="2026-11-08T23"), ok)["severity"] == hc.WARN   # 449 h


def test_a_dead_poller_beside_a_silent_source_is_an_alarm_that_says_both():
    inv = {"streams": [
        _gen_stale(),
        _stream("fires", "stale", "2026-11-02.jsonl.gz", 402.0),
        _stream("griddemand", "stale", "2026-11-25.jsonl", 60.0),
    ]}
    routes = {"gridgeneration": _poll_route(), "griddemand": _poll_route(first_failed=["US48"])}
    f = hc.check_quiet_archives(inv, routes, now_ms=_NOW_MS)
    assert f["severity"] == hc.ALARM
    assert "fires: newest file 2026-11-02.jsonl.gz" in f["detail"]
    assert "griddemand: newest file 2026-11-25.jsonl" in f["detail"] and "US48" in f["detail"]
    assert "gridgeneration: the source is silent, not the poller" in f["detail"]


def test_a_stream_with_no_poll_route_is_never_excused_by_one():
    """Only the streams whose route serves the poll state can read as a silent source."""
    fires = _stream("fires", "stale", "2026-11-02.jsonl.gz", 402.0)
    f = hc.check_quiet_archives({"streams": [fires]}, {"fires": _poll_route()}, now_ms=_NOW_MS)
    assert f["severity"] == hc.ALARM
    assert set(hc.SOURCE_POLL_ROUTES) == {"gridgeneration", "griddemand"}


def test_poll_routes_are_requested_only_for_their_stale_streams():
    asked = []

    def fetch(url):
        asked.append(url)
        return _poll_route()

    inv = {"streams": [_gen_stale(), _stream("griddemand", "live", "2026-11-26.jsonl", 19.7),
                       _stream("fires", "stale", "2026-11-02.jsonl.gz", 402.0)]}
    got = hc.gather_poll_routes("https://x", inv, fetch=fetch)
    assert asked == ["https://x/api/data/grid-generation"] and list(got) == ["gridgeneration"]
    assert hc.gather_poll_routes("https://x", None, fetch=fetch) == {}
    assert hc.gather_poll_routes("https://x", {"streams": [_stream("fires", "live", "a.jsonl", 0)]}, fetch=fetch) == {}
    assert len(asked) == 1


def test_the_two_poll_routes_exist_and_serve_the_poll_state():
    """The check reads a field of two routes; a rename on the server would turn the WARN back
    into an ALARM silently, so the names are pinned to the code that serves them."""
    here = os.path.dirname(os.path.abspath(__file__))
    routes = open(os.path.join(here, "server", "routes.ts"), encoding="utf-8").read()
    for path, _who in hc.SOURCE_POLL_ROUTES.values():
        at = routes.index(f'app.get("{path}"')
        assert "poll: latest" in routes[at:at + 2500], path
    state = open(os.path.join(here, "server", "eia930CatchUp.ts"), encoding="utf-8").read()
    block = state[state.index("export interface PollState"):]
    for field in ("at: number", "newest_period:", "first_failed:", "catch_up:"):
        assert field in block[:block.index("}")], field


def test_edgar_latest_poll_unread_documents_are_said_and_an_older_server_reads_as_before():
    # [PIPELINE 2026-11-24, open_questions.md (ai3)] the probe's `documents`
    # block: a poll that read part of its page is an ok poll, and the part it
    # could not read is said on the line.
    fresh = {"last_archive_write_at": "2026-11-17T18:55:00.000Z"}
    now = _utc("2026-11-17T19:05:00")
    part = _edgar_probe(form4=fresh, f13={"documents": {
        "at": "2026-11-17T19:00:00.000Z", "candidates": 40, "returned": 37, "skipped": 0, "failed": 3,
        "first_error": "index.json -> 404"}})
    f = hc.check_edgar_archives(part, now=now)
    assert f["severity"] == hc.OK  # the verdict is still the run of failed polls
    assert ("13F last row none in the day files, polls 6 ok / 0 failed; "
            "latest poll could not read 3 of 40 filings (first error: index.json -> 404)") in f["detail"]
    clean = _edgar_probe(form4=fresh, e8k={"documents": {"candidates": 15, "returned": 9, "skipped": 6, "failed": 0,
                                                         "first_error": None}})
    assert "could not read" not in hc.check_edgar_archives(clean, now=now)["detail"]
    for odd in (None, "x", [], {"failed": None}):
        assert hc.edgar_documents_note(odd) == ""
    assert "could not read" not in hc.check_edgar_archives(_edgar_probe(form4=fresh), now=now)["detail"]
    # the FIX text no longer says 13F has no completeness pass (it has had one since 2026-11-19)
    bad = hc.check_edgar_archives(_edgar_probe(form4=fresh, f13={"consecutive_failures": 2, "last_error": "x"}), now=now)
    assert "Form 4 and 13F completeness passes" in bad["fix"]
