#!/usr/bin/env python3
"""
session_health_check.py — compile the MEMORY PROTOCOL's manual "check system
health" step into a script (EDGE DOCTRINE #3: COMPILE KNOWLEDGE INTO CODE —
never reason the same diagnosis twice).

Every autonomous session starts by reading CLAUDE.md's KNOWN BROKEN section
and then hand-running a handful of `curl .../api/diag/*?token=$DIAG_TOKEN`
one-liners to check: is the loop dark (LIVENESS ALARM)? is deep_score's
alt-data enrichment blind (KNOWN BROKEN #21)? are daemon timeouts clustering
(KNOWN BROKEN #18)? is the ML feedback loop recording real outcomes (KNOWN
BROKEN #12)? is the live site actually running this checkout's code, or is
it stuck behind the PRODUCTION DEPLOY FREEZE (research/wishlist.md, since
2026-07-22)? That reasoning has now been done by hand across several
sessions (see research/open_questions.md's #21 update chain, 2026-07-13/14,
and wishlist.md's CI-outage entries for the deploy-freshness comparison)
— this script is the second occurrence becoming code.

This is READ-ONLY: it fetches from the deployed site's token-gated
/api/diag/* surface (server/diag.ts) and /api/health, then classifies. It
never touches the trading loop, never authenticates as the owner, and
carries no side effects.

Every classifier below is a PURE function of already-parsed JSON — testable
without a network call (see test_session_health_check.py) and without
duplicating the canonical liveness computation in server/liveness.ts (this
script reads /api/health's already-computed `checks.bot.liveness` rather
than re-deriving dark/market-hours/wall-hours itself, so it can never drift
from the one true implementation RULE REVIEW would otherwise flag).

Usage:
  python3 scripts/session_health_check.py [--base-url URL] [--json]
  DIAG_TOKEN must be set in the environment (silently no-ops the diag-only
  checks if absent — /api/health alone still runs, keyless).

Exit code: 0 = all clear, 1 = WARN present (non-blocking, log and move on),
2 = ALARM present (top-of-report per CLAUDE.md Amendment 1 — surface first).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

DEFAULT_BASE_URL = "https://voltradeai-production.up.railway.app"
DEFAULT_TIMEOUT_S = 20
DEFAULT_OUTAGE_STATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "research", "outage_state.json"
)
DEFAULT_LIVENESS_NOTIFY_STATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "research", "liveness_notify_state.json"
)

OK, WARN, ALARM = "OK", "WARN", "ALARM"
_SEVERITY_RANK = {OK: 0, WARN: 1, ALARM: 2}


def finding(severity, label, detail, fix=None):
    f = {"severity": severity, "label": label, "detail": detail}
    if fix:
        f["fix"] = fix
    return f


# ── pure classifiers (no network) ────────────────────────────────────────

# railway.json (FROZEN): healthcheckPath=/api/health, healthcheckTimeout=60.
# A non-2xx answer REJECTS a new container — the deploy fails and whatever was
# serving before (or Railway's 502 page) keeps serving. server/healthGate.ts
# owns which checks may do that; this reads its `serving` block back.
def check_deploy_gate(health):
    """KNOWN BROKEN #41 (2026-09-10 -> 2026-09-16): every fresh container
    answered Railway's probe with 503 for five days because a monitoring
    alarm (liveness) was mapped onto the HTTP code, and no session noticed
    because this script treated the 503 as 'no response'. Now: an app-level
    non-2xx /api/health is its own ALARM naming what is gating it, and a
    payload whose `serving.ok` is false says so even when reached via 200
    (it cannot be, by construction — belt and braces)."""
    if not isinstance(health, dict):
        return finding(ALARM, "deploy_gate", "no /api/health payload — cannot tell whether a new container would pass Railway's healthcheck")
    code = health.get("_http_status")
    serving = health.get("serving") if isinstance(health.get("serving"), dict) else None
    if code is not None and not (200 <= int(code) < 300):
        failing = (serving or {}).get("failing")
        if failing is None:
            degraded = sorted(
                k for k, v in (health.get("checks") or {}).items()
                if isinstance(v, dict) and v.get("status") not in (None, "ok")
            )
            return finding(
                ALARM, "deploy_gate",
                f"/api/health answered HTTP {code} with NO `serving` block — a pre-healthGate.ts build is live, "
                f"so EVERY degraded check gates the deploy: {degraded or '(none named)'}; any merged fix will be "
                f"rejected by Railway until whatever is degraded clears or a human redeploys a build that carries "
                f"server/healthGate.ts (KNOWN BROKEN #41)",
            )
        return finding(
            ALARM, "deploy_gate",
            f"/api/health answered HTTP {code}: Railway would REJECT a new container — gating checks failing: "
            f"{failing} (gates={(serving or {}).get('gates')}); merged fixes cannot reach production until this clears",
        )
    if serving is not None and serving.get("ok") is False:
        return finding(ALARM, "deploy_gate", f"serving.ok=false with a 2xx answer — contradiction, inspect server/healthGate.ts (failing={serving.get('failing')})")
    if serving is None:
        return finding(WARN, "deploy_gate", "2xx but no `serving` block — a pre-healthGate.ts build is live; a future degraded status would veto deploys (KNOWN BROKEN #41)")
    return finding(OK, "deploy_gate", f"2xx, serving gates={serving.get('gates')} all ok — a new container would pass Railway's healthcheck")


def check_liveness(health):
    """CLAUDE.md Amendment 1 LIVENESS ALARM: the loop going dark is a
    TOP-OF-REPORT alarm, never a dashboard-only discovery. /api/health
    already computes this (server/liveness.ts) — read it, don't re-derive."""
    if not isinstance(health, dict):
        return finding(ALARM, "liveness", "no /api/health response — cannot confirm the loop is alive")
    bot = (health.get("checks") or {}).get("bot") or {}
    lv = bot.get("liveness") or {}
    if lv.get("dark"):
        ar = bot.get("autoResume") or {}
        cause = ""
        if ar.get("latchKind"):
            cause = (f" CAUSE: latch={ar.get('latchKind')}, auto-resume {ar.get('outcome')} "
                     f"({ar.get('reason')}), next check {ar.get('nextCheck')}.")
        elif bot.get("status") == "killed":
            cause = " CAUSE: killed, latch kind not reported (pre-v1.0.1089 server?)."
        return finding(
            ALARM, "liveness",
            f"LOOP DARK: {lv.get('detail', '(no detail)')} "
            f"(marketHours={lv.get('marketHours')}, wallHours={lv.get('wallHours')}){cause}",
        )
    status = health.get("status")
    if status not in ("ok", "degraded"):
        return finding(ALARM, "liveness", f"/api/health status={status!r} (expected ok/degraded)")
    return finding(OK, "liveness", "loop alive, not dark")


def check_health_subsystems(health):
    """Non-liveness /api/health subsystems (server/db/alpaca/python/scanner/
    licensing) — any non-'ok' status is a WARN, not an ALARM (liveness owns
    the alarm tier per Amendment 1)."""
    checks = (health or {}).get("checks") or {}
    bad = []
    for name, c in checks.items():
        if name == "bot":
            continue
        status = c.get("status") if isinstance(c, dict) else None
        if status not in ("ok", None):
            bad.append(f"{name}={status}")
    if bad:
        return finding(WARN, "subsystems", "degraded: " + ", ".join(sorted(bad)))
    return finding(OK, "subsystems", "server/db/alpaca/python/scanner/licensing all ok")


def check_process_faults(health):
    """KNOWN BROKEN #41's 2026-09-16 crash-visibility fix (server/
    crashVisibility.ts, v1.0.917) put unhandledRejection/uncaughtException
    counts on /api/health's `checks.process` block — this is that NEXT
    step, queued by the same session: ALARM on the counters instead of
    leaving them to be read by hand. `stats` there is in-process memory,
    not persisted, so a nonzero reading is always about the CURRENT
    container's own uptime — read `lastRejection`/`lastException` first on
    any restart-loop investigation, per that fix's own NEXT note."""
    if not isinstance(health, dict):
        return finding(WARN, "process_faults", "no /api/health response — cannot read crash-handler counters")
    proc = (health.get("checks") or {}).get("process")
    if not isinstance(proc, dict):
        return finding(WARN, "process_faults", "no checks.process block — pre-v1.0.917 build, crash visibility unavailable")
    rejections = proc.get("unhandledRejections") or 0
    exceptions = proc.get("uncaughtExceptions") or 0
    faults = proc.get("handlerFaults") or 0
    notes = []
    if rejections > 0:
        last = proc.get("lastRejection") or {}
        notes.append(f"{rejections} unhandled promise rejection(s), last at {last.get('at')}: {last.get('detail')}")
    if exceptions > 0:
        last = proc.get("lastException") or {}
        notes.append(f"{exceptions} uncaught exception(s) (process exits after each), last at {last.get('at')}: {last.get('detail')}")
    if notes:
        return finding(ALARM, "process_faults", "; ".join(notes))
    if faults > 0:
        return finding(WARN, "process_faults", f"{faults} crash-handler fault(s) — a FATAL-* audit line may be missing, trace incomplete")
    return finding(OK, "process_faults", "no unhandled rejections/exceptions this container's uptime")


def check_alt_data_enrichment(diagnostic_entries, min_entries_for_verdict=3):
    """KNOWN BROKEN #21: deep_score's 5-fetcher enrichment block (wikipedia/
    gdelt/fred are the 3 with a cache-freshness alarm) going permanently
    unreached. Persisted 'Multiple API sources down' across EVERY entry in
    the window (not just some) is the confirmed signature from the
    2026-07-13/14 trace — a few stragglers during a quiet window is not."""
    entries = diagnostic_entries or []
    if len(entries) < min_entries_for_verdict:
        return finding(
            OK, "alt_data_enrichment",
            f"only {len(entries)} DIAGNOSTIC entries in window — too few to judge "
            "(market closed, or quiet cadence); not a verdict either way",
        )
    down_count = sum(
        1 for e in entries
        if "Multiple API sources down" in str(e.get("message", ""))
        and all(s in str(e.get("message", "")) for s in ("wikipedia", "gdelt", "fred"))
    )
    if down_count == len(entries):
        return finding(
            WARN, "alt_data_enrichment",
            f"ALL {len(entries)} DIAGNOSTIC entries in window report wikipedia/gdelt/fred "
            "down — matches KNOWN BROKEN #21's confirmed signature (deep_score never "
            "reached). Check /api/diag/scanner's dataSourceErrors.stock_details next.",
        )
    if down_count > 0:
        return finding(
            OK, "alt_data_enrichment",
            f"{down_count}/{len(entries)} entries report wikipedia/gdelt/fred down — "
            "intermittent, not the permanent-latch signature",
        )
    return finding(OK, "alt_data_enrichment", "wikipedia/gdelt/fred reporting fresh")


def check_daemon_memory(daemon, skip_mb=550, trim_mb=400):
    """KNOWN BROKEN #21's v1.0.314 fix re-based the deep-score skip/trim
    guard to 550/400MB against a live idle baseline of ~254MB. If the
    daemon's current RSS is already past skip_mb, the same permanent-latch
    failure mode is reproducing at the new thresholds (RECURRENCE ESCALATES
    — do not re-tune blind, read this and the alt_data_enrichment finding
    together before touching the constants again)."""
    if not isinstance(daemon, dict) or not daemon.get("alive"):
        return finding(WARN, "daemon_memory", "daemon not reachable/alive via /api/diag/daemon")
    rss = daemon.get("rss_mb")
    if rss is None:
        return finding(WARN, "daemon_memory", "rss_mb missing from /api/diag/daemon response")
    if rss >= skip_mb:
        return finding(
            ALARM, "daemon_memory",
            f"daemon rss={rss}MB >= skip_mb={skip_mb}MB — deep_score guard likely "
            "latched off again (same shape as the pre-v1.0.314 recurrence)",
        )
    if rss >= trim_mb:
        return finding(
            WARN, "daemon_memory",
            f"daemon rss={rss}MB >= trim_mb={trim_mb}MB — deep_score running in trimmed mode",
        )
    return finding(OK, "daemon_memory", f"daemon rss={rss}MB, well under trim_mb={trim_mb}MB")


def check_tier2_daemon_timeouts(tier2_error_entries, cluster_threshold=3):
    """KNOWN BROKEN #18: daemon run_full_scan timeouts clustering (was
    hypothesized as zombie-thread pileup, refuted 2026-07-12 — event-loop
    lag is the confirmed mechanism, non-blocking). Just count + surface
    active_dispatches readings for whichever session picks up the thread
    next; do not diagnose further here."""
    entries = tier2_error_entries or []
    timeouts = [e for e in entries if "timeout" in str(e.get("message", "")).lower()]
    if len(timeouts) >= cluster_threshold:
        dispatches = [
            e["message"].split("active_dispatches=")[1].split()[0]
            for e in timeouts if "active_dispatches=" in str(e.get("message", ""))
        ]
        return finding(
            WARN, "tier2_daemon_timeouts",
            f"{len(timeouts)} daemon timeouts in window (active_dispatches seen: "
            f"{dispatches or 'none captured'}) — see KNOWN BROKEN #18, non-blocking",
        )
    return finding(OK, "tier2_daemon_timeouts", f"{len(timeouts)} timeout(s) in window, below cluster threshold")


def check_ml_feedback(ml):
    """KNOWN BROKEN #12: exit paths beyond the WS monitor still record
    nothing (item 12c, open). retrain_overdue is a harder, unconditional
    signal (Tier 3 is supposed to keep the model fresh)."""
    if not isinstance(ml, dict):
        return finding(WARN, "ml_feedback", "no /api/diag/ml response")
    findings = []
    if ml.get("retrain_overdue"):
        findings.append(f"retrain_overdue (model_age_hours={ml.get('model_age_hours')})")
    live_perf = ml.get("live_performance") or {}
    live_count = ml.get("feedback_live_count") or 0
    breakdown = ml.get("live_outcome_breakdown") or {}
    non_orphan = sum(v for k, v in breakdown.items() if k != "orphan_exit")
    if live_count > 0 and live_perf.get("total_trades") == 0 and non_orphan == 0:
        findings.append(
            f"{live_count} live feedback records, all orphan_exit / zero attributed "
            "trades — matches KNOWN BROKEN #12(c)'s open exit-path gap"
        )
    if findings:
        return finding(WARN, "ml_feedback", "; ".join(findings))
    return finding(OK, "ml_feedback", f"model_age_hours={ml.get('model_age_hours')}, no known-broken signature")


def check_deploy_freshness(local_version, server_version):
    """PRODUCTION DEPLOY FREEZE (research/wishlist.md, tracked since the
    2026-07-22T14:09Z GitHub Actions outage): Railway's 'Wait for CI' gate
    means a merged main HEAD does not reach the live site while Actions
    can't allocate runners. At least 4 separate sessions since then have
    hand-diagnosed this the same way — compare this checkout's
    package.json version against the live /api/data/layers server_version
    — before trusting any "should be live now" note in past PRs. That
    repeated manual comparison is exactly EDGE DOCTRINE #3's "second
    occurrence becomes a script": this is the second-plus occurrence.
    NOT the LIVENESS ALARM (the loop is still running — just possibly
    stale code), so a mismatch is WARN, not ALARM, matching this script's
    existing severity scheme."""
    if not server_version:
        return finding(WARN, "deploy_freshness", "no server_version returned by /api/data/layers")
    if not local_version:
        return finding(WARN, "deploy_freshness", "could not read this checkout's package.json version")
    if server_version != local_version:
        return finding(
            WARN, "deploy_freshness",
            f"LIVE SITE IS STALE: server_version={server_version} vs this checkout's "
            f"package.json={local_version} — if this checkout is at/near main HEAD, this "
            "matches the PRODUCTION DEPLOY FREEZE tracked in research/wishlist.md "
            "(Railway's 'Wait for CI' gate stuck behind the GitHub Actions outage)",
        )
    return finding(OK, "deploy_freshness", f"server_version={server_version} matches this checkout's package.json")


def _hours_between(earlier_iso, later_iso):
    earlier = datetime.fromisoformat(earlier_iso.replace("Z", "+00:00"))
    later = datetime.fromisoformat(later_iso.replace("Z", "+00:00"))
    return (later - earlier).total_seconds() / 3600.0


def compute_outage_state(health_reachable, now_iso, prior_state=None):
    """Persist the first-detected-down timestamp across sessions so 'how long
    has this outage lasted' stops being re-derived by hand from old prose —
    the KNOWN BROKEN #41 production outage (research/open_questions.md) had
    at least six separate sessions each hand-computing wall-clock duration
    from a copied-forward '2026-09-10T20:18Z' onset string ('~30 wall-clock
    hours', '~78.3 wall-clock hours', ...). EDGE DOCTRINE #3: the second
    occurrence of the same reasoning becomes code, not another manual
    subtraction. Pure function of (reachable-now, now, prior persisted
    state) — the only I/O is load_outage_state/save_outage_state below.
    """
    prior = dict(prior_state or {})
    down_since = prior.get("down_since")
    # The last outage's record is HISTORY, not state: every branch below
    # carries it forward. Found 2026-09-18: the healthy-and-was-healthy
    # branch returned a bare {down_since: None} and overwrote the 126.4h
    # record of the 2026-09-10 -> 09-16 outage on the very next run.
    history = {k: prior[k] for k in ("last_outage_started_utc", "last_recovered_utc") if prior.get(k)}
    if health_reachable:
        if down_since:
            return {"down_since": None, "last_recovered_utc": now_iso, "last_outage_started_utc": down_since}
        return {"down_since": None, **history}
    if down_since:
        return prior
    return {"down_since": now_iso, **history}


def check_outage_duration(state, now_iso):
    """Reports the persisted outage's running duration. Deliberately
    orthogonal to check_liveness (which already ALARMs the instant
    /api/health is unreachable) — this adds the 'and it has now been
    running for N hours' context that check_liveness's single-snapshot
    view cannot carry across sessions on its own."""
    down_since = (state or {}).get("down_since")
    if not down_since:
        return finding(OK, "outage_duration", "site reachable, no outage in progress")
    hours = _hours_between(down_since, now_iso)
    alarm_note = (
        " — past both CLAUDE.md Amendment 1 LIVENESS ALARM thresholds (2 market hours / 24 wall hours)"
        if hours >= 24 else ""
    )
    return finding(
        ALARM, "outage_duration",
        f"site unreachable continuously since {down_since} — {hours:.1f}h and counting{alarm_note}",
    )


def compute_liveness_notify_state(dark, wall_hours, now_iso, prior_state=None):
    """Persists the notify-threshold state for the LIVENESS ALARM (the loop
    dark while the SITE ITSELF stays reachable — e.g. a latched kill switch,
    as opposed to compute_outage_state's site-unreachable case). Before this,
    every session decided by hand whether to (re-)notify by grepping old
    experiments.md/wishlist.md prose for "already notified twice, don't
    repeat" — the 2026-09-10 kill-switch incident alone has at least six
    sessions independently re-deriving the same "notify on first alarm, then
    only once wall-clock duration has roughly doubled since the last
    notification" policy from scratch by reading history. EDGE DOCTRINE #3:
    the second occurrence of the same reasoning becomes code, not another
    manual re-derivation. Pure function — the only I/O is
    load_outage_state/save_outage_state below (reused as-is: both are
    generic path-parameterized JSON state helpers, not outage-specific)."""
    prior = dict(prior_state or {})
    if not dark:
        if prior.get("dark_since"):
            return {"dark_since": None, "last_notified_utc": None, "last_notified_wall_hours": None}
        return {"dark_since": None, "last_notified_utc": None, "last_notified_wall_hours": None}
    last_notified_hours = prior.get("last_notified_wall_hours")
    should_notify = last_notified_hours is None or wall_hours >= 2 * last_notified_hours
    return {
        "dark_since": prior.get("dark_since") or now_iso,
        "last_notified_utc": now_iso if should_notify else prior.get("last_notified_utc"),
        "last_notified_wall_hours": wall_hours if should_notify else last_notified_hours,
    }


def check_liveness_notification(prior_state, new_state, wall_hours):
    """Reports whether THIS run crossed the doubling threshold above and is
    therefore a fresh notify-worthy checkpoint for a session (or an
    automated caller) deciding whether to send a human notification about
    the ongoing dark loop. WARN, not ALARM: check_liveness already carries
    ALARM for "the loop is dark" itself on every run — this is the softer,
    distinct "and this specific run is new information" signal, so a
    RECOVERED loop (or one already notified at this duration) does not
    re-trip the alarm tier."""
    if not new_state.get("dark_since"):
        return finding(OK, "liveness_notify", "loop not dark, nothing to notify")
    notified_this_run = new_state.get("last_notified_utc") not in (None, (prior_state or {}).get("last_notified_utc"))
    if notified_this_run:
        prior_hours = (prior_state or {}).get("last_notified_wall_hours")
        return finding(
            WARN, "liveness_notify",
            f"NOTIFY-WORTHY: loop dark {wall_hours:.1f}h wall-clock — first alarm, or doubled since the "
            f"last notification at {prior_hours}h. A session with PushNotification access should send one.",
        )
    return finding(
        OK, "liveness_notify",
        f"loop dark {wall_hours:.1f}h wall-clock, already notified at {new_state.get('last_notified_wall_hours')}h "
        "— no new notify threshold crossed, do not repeat",
    )


def load_outage_state(path=DEFAULT_OUTAGE_STATE_PATH):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_outage_state(state, path=DEFAULT_OUTAGE_STATE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


# ── DATA-PLANE OPS CHECKS (human-directed 2026-10-14, ops item 4) ────────
# One daily pass over the live feeds the bot does not own: SWIM flight plans,
# filed-route placement, aircraft intent coverage, archive storage, the
# adsb.lol upstream and /data API latency. Each reads a PUBLIC endpoint (no
# token) and every non-OK finding carries the exact `fix`, so --alert can
# send one short message only when something needs doing. Safe auto-fixes
# live in the server, not here: the SWIM stall watchdog restarts a dead
# consumer (swimConnector.ts) and sweep archiving pauses itself below its
# 2 GiB gate (globalAircraft.ts sweepArchiveAllowed) — these checks report
# them, never duplicate them.

GIB = 1_073_741_824
SWIM_SILENT_ALARM_MIN = 15
SWIM_AUTH_RE = re.compile(r"auth|login|password|credential|unauthori[sz]ed|\b401\b|\b403\b", re.I)


def check_swim(plan_status, uptime_s=None, now_ms=None):
    label = "SWIM SFDPS (flight plans)"
    if plan_status is None:
        return finding(WARN, label, "plan-status unreachable", "check the site is up (/api/health)")
    if not plan_status.get("swimConfigured"):
        return finding(WARN, label, "SFDPS env not configured — no FAA flight plans",
                       "set SWIM_SFDPS_URL/VPN/QUEUE + SWIM_USER/SWIM_PASSWORD in Railway")
    sw = plan_status.get("swim") or {}
    err = sw.get("lastError") or ""
    if not plan_status.get("swimConnected"):
        if SWIM_AUTH_RE.search(err):
            fix = ("login rejected: rotate the password in the SCDS portal, paste it into Railway "
                   "SWIM_PASSWORD, redeploy; confirm plan-status swimConnected=true")
        else:
            fix = ("the consumer retries on its own (30 s doubling to 15 min); if still down at the next "
                   "check, redeploy")
        return finding(ALARM, label, f"disconnected — lastError: {err or 'none'}", fix)
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    last = sw.get("lastMessageAt")
    if isinstance(last, (int, float)):
        silent_min = (now_ms - last) / 60000
        if silent_min > SWIM_SILENT_ALARM_MIN:
            return finding(ALARM, label, f"connected but silent for {silent_min:.0f} min",
                           "the stall watchdog restarts it within 10 min; if this persists, redeploy and "
                           "check the queue subscription in the SCDS portal")
    recv = sw.get("messagesReceived") or 0
    processed = sw.get("messagesProcessed") or 0
    errs = (sw.get("processErrors") or 0) + (sw.get("parseErrors") or 0)
    dropped = sw.get("droppedOverflow") or 0
    rate = f"{recv / (uptime_s / 3600):,.0f} msg/h" if uptime_s and uptime_s >= 600 else "rate n/a (just booted)"
    detail = (f"connected, {rate} since boot, {errs} process/parse errors, {dropped} overflow drops, "
              f"{sw.get('stallRestarts', 0)} watchdog restarts")
    if processed >= 1000 and errs / processed > 0.01:
        return finding(WARN, label, detail, "parse errors above 1%: SFDPS schema change? see swim.lastError")
    if recv >= 1000 and dropped / recv > 0.01:
        return finding(WARN, label, detail, "overflow drops above 1%: the drain loop is falling behind")
    return finding(OK, label, detail)


def check_filed_routes(plan_status, min_routes=200, min_share=0.7):
    """FAA-filed plans must turn into real route points (the parser's live
    shadow counters, swimSfdps.ts parserShadow)."""
    label = "Filed routes -> route points"
    ps = (((plan_status or {}).get("swim") or {}).get("routeShape") or {}).get("parserShadow") or {}
    routes, placed = ps.get("routes") or 0, ps.get("placed") or 0
    if routes < min_routes:
        return finding(OK, label, f"{routes} routes parsed since boot — too few to judge")
    share = placed / routes
    # most "unresolved" tokens are honest gaps, not failures: an amendment
    # from the present position ("./.", the leg from wherever the aircraft
    # was is unknown) or an unexpanded SID/STAR — so they are reported, and
    # only the placed share is judged
    detail = (f"{placed}/{routes} placed ({share:.0%}); {ps.get('amended', 0)} amended from present position, "
              f"{ps.get('withUnresolved', 0)} carry a gap token")
    if share < min_share:
        return finding(WARN, label, detail,
                       "placement fell below 70%: check the NASR fixes/airways cycle and CIFP load "
                       "(plan-status swim.routeShape.unresolvedTokens)")
    return finding(OK, label, detail)


def check_intent_coverage(cov, min_aircraft=300, mcp_floor=25.0):
    label = "Aircraft intent coverage"
    if cov is None:
        return finding(WARN, label, "intent-coverage unreachable", "check the site is up (/api/health)")
    n = cov.get("aircraft") or 0
    lp = cov.get("latest_pct") or {}
    rows = cov.get("rows_ingested") or 0
    rejected = sum((cov.get("rejected") or {}).values())
    detail = (f"{n} airborne; selected alt {lp.get('nav_altitude_mcp', 0)}%, QNH {lp.get('nav_qnh', 0)}%, "
              f"mode bits {lp.get('nav_modes', 0)}%, wind {lp.get('wd', 0)}%")
    daily = ((cov.get("archive") or {}).get("daily") or {})
    if daily:
        detail += f"; daily rollups {daily.get('days', 0)} (latest {daily.get('latest')})"
    if n < min_aircraft:
        return finding(OK, label, detail + " — sample too small to judge")
    if (lp.get("nav_altitude_mcp") or 0) < mcp_floor:
        return finding(WARN, label, detail,
                       "selected-altitude coverage collapsed: compare a raw adsb.lol row against "
                       "aircraftIntent.parseReadsbIntent (upstream schema drift?)")
    if rows >= 1000 and rejected / rows > 0.01:
        return finding(WARN, label, detail + f"; {rejected} rejected values",
                       "over 1% of rows rejected: see intent-coverage.rejected for the field")
    return finding(OK, label, detail)


def check_storage(offload, guard_bytes=GIB, alarm_bytes=1_200_000_000):
    label = "Archive storage"
    if offload is None:
        return finding(WARN, label, "offload-status unreachable", "check the site is up (/api/health)")
    vol = offload.get("volume") or {}
    free = vol.get("freeBytes")
    per_day = offload.get("estimatedBytesPerDay") or 0
    if not isinstance(free, (int, float)):
        return finding(WARN, label, "volume free space unreadable", "check offload-status.volume")
    detail = f"{free / GIB:.2f} GiB free, ~{per_day / 1e6:.0f} MB/day archived"
    if free < 2 * GIB:
        detail += "; sweep archiving auto-paused below its 2 GiB gate"
    if per_day > 0:
        detail += f"; {max(0.0, (free - guard_bytes) / per_day):.0f} d to the 1 GiB guard if eviction stopped"
    if offload.get("lastError"):
        return finding(WARN, label, detail + f"; offload error: {offload['lastError']}",
                       "R2 offload failing: check R2_* env in Railway; nothing is evicted until uploads verify")
    if free < alarm_bytes:
        return finding(ALARM, label, detail,
                       "below 1.2 GB: shorten VESSEL_HOT_TIER_DAYS / localHotDays or enlarge the Railway volume")
    return finding(OK, label, detail)


def check_adsb_upstream(governor, min_results=30, max_429_share=0.25, max_failover_share=0.5):
    label = "adsb.lol upstream"
    lh = (governor or {}).get("last_hour")
    if not lh:
        return finding(OK, label, "no last-hour governor stats (older build)")
    bg, fg = lh.get("bg") or {}, lh.get("fg") or {}
    bg_n, fg_n = bg.get("results") or 0, fg.get("results") or 0
    r429 = bg.get("rate_limited") or 0
    fg_fail = fg_n - (fg.get("ok") or 0)
    detail = (f"sweep {bg_n} req/h, {r429} rate-limited; viewport {fg_n} req/h, {fg_fail} failed over "
              f"to the next provider")
    if bg_n >= min_results and r429 / bg_n > max_429_share:
        rps = governor.get("bg_rps") or 0.4
        eff = governor.get("effective_bg_rps")
        if eff is not None and eff > rps * 0.1 + 1e-9:
            # the sweep paces itself (AIMD, adsbGovernor.ts): still converging,
            # nothing for a human to do yet
            return finding(OK, label, detail + f" ({r429 / bg_n:.0%} 429s; sweep self-pacing at {eff} of {rps} req/s)")
        return finding(WARN, label, detail + f" ({r429 / bg_n:.0%} 429s at the self-pacing floor)",
                       f"set GLOBAL_SWEEP_RPS={max(0.05, round(rps / 2, 2))} in Railway (now {rps}) and redeploy")
    if fg_n >= min_results and fg_fail / fg_n > max_failover_share:
        return finding(WARN, label, detail + f" ({fg_fail / fg_n:.0%} failover)",
                       "adsb.lol failing for viewers; the chain still serves airplanes.live/adsb.fi — act only if "
                       "it persists a second day (adsb.lol status page)")
    return finding(OK, label, detail)


def p95(samples):
    xs = sorted(x for x in samples if isinstance(x, (int, float)))
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def check_data_latency(samples_ms, warn_ms=2500):
    label = "/data API latency"
    v = p95(samples_ms or [])
    if v is None:
        return finding(WARN, label, "no successful /data API sample", "check the site is up (/api/health)")
    detail = f"p95 {v:.0f} ms over {len(samples_ms)} requests"
    if v > warn_ms:
        return finding(WARN, label, detail, "slow /data API: check /api/health memory + process blocks")
    return finding(OK, label, detail)


def run_ops_checks(plan_status, intent_cov, offload, governor, latency_ms, uptime_s=None, now_ms=None):
    return [
        check_swim(plan_status, uptime_s=uptime_s, now_ms=now_ms),
        check_filed_routes(plan_status),
        check_intent_coverage(intent_cov),
        check_storage(offload),
        check_adsb_upstream(governor),
        check_data_latency(latency_ms),
    ]


def alert_message(findings):
    """ONE short message, only when something needs doing (None = stay quiet)."""
    bad = [f for f in findings if f["severity"] != OK]
    if not bad:
        return None
    bad.sort(key=lambda f: -_SEVERITY_RANK[f["severity"]])
    lines = [f"VolTradeAI ops: {len(bad)} item(s) need action"]
    for f in bad:
        lines.append(f"- [{f['severity']}] {f['label']}: {f['detail']}" + (f" -> FIX: {f['fix']}" if f.get("fix") else ""))
    return "\n".join(lines)


def run_all_checks(health, daemon, diagnostic_entries, tier2_error_entries, ml,
                    local_version=None, server_version=None):
    return [
        check_deploy_gate(health),
        check_liveness(health),
        check_health_subsystems(health),
        check_process_faults(health),
        check_alt_data_enrichment(diagnostic_entries),
        check_daemon_memory(daemon),
        check_tier2_daemon_timeouts(tier2_error_entries),
        check_ml_feedback(ml),
        check_deploy_freshness(local_version, server_version),
    ]


def overall_exit_code(findings):
    worst = max((_SEVERITY_RANK[f["severity"]] for f in findings), default=0)
    return {0: 0, 1: 1, 2: 2}[worst]


# ── network fetch (thin, not unit-tested directly) ───────────────────────

def _fetch_json(url, timeout=DEFAULT_TIMEOUT_S):
    """Returns the decoded JSON body, or None when nothing usable came back.

    An HTTP error that CARRIES a JSON body (the app's own 503 from
    /api/health) is returned with `_http_status` attached instead of being
    dropped — KNOWN BROKEN #41: for the first ~5 hours of the 2026-09-10
    outage the app itself was answering 503 with a payload that named the
    exact check gating it, and this function threw it away as
    "fetch failed", indistinguishable from Railway's 502 edge fallback.
    Railway's fallback carries `"status":"error"` and no `checks`, so the
    two remain told apart by check_deploy_gate below."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = None
        try:
            body = json.loads(e.read().decode("utf-8"))
        except (ValueError, OSError, AttributeError):
            body = None
        if isinstance(body, dict) and "checks" in body:
            print(f"[session_health_check] {url} answered HTTP {e.code} WITH a health payload — keeping it", file=sys.stderr)
            body["_http_status"] = e.code
            return body
        print(f"[session_health_check] fetch failed for {url}: {e}", file=sys.stderr)
        return None
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        print(f"[session_health_check] fetch failed for {url}: {e}", file=sys.stderr)
        return None


def read_local_package_version(repo_root=None):
    """The version this checkout would deploy if merged/built right now —
    package.json's `version` field (PROMOTION RULE 4's read-and-increment
    field), the same value server/routes.ts echoes back as `server_version`
    on /api/data/layers once a deploy has actually happened."""
    root = repo_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    path = os.path.join(root, "package.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("version")
    except (OSError, ValueError):
        return None


def _timed_get_ms(url, timeout=DEFAULT_TIMEOUT_S):
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            r.read()
        return (time.monotonic() - t0) * 1000
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def gather_ops(base_url, health, samples=5):
    plan_status = _fetch_json(f"{base_url}/api/data/aircraft/plan-status")
    intent_cov = _fetch_json(f"{base_url}/api/data/aircraft/intent-coverage")
    offload = _fetch_json(f"{base_url}/api/data/archive/offload-status")
    glob = _fetch_json(f"{base_url}/api/data/aircraft/global?lamin=0&lamax=1&lomin=0&lomax=1")
    governor = (((glob or {}).get("coverage") or {}).get("sweep") or {}).get("governor")
    latency = []
    for _ in range(samples):
        for path in ("/api/data/layers", "/api/data/aircraft/global?lamin=30&lamax=50&lomin=-100&lomax=-70"):
            ms = _timed_get_ms(f"{base_url}{path}")
            if ms is not None:
                latency.append(ms)
    uptime_s = (((health or {}).get("checks") or {}).get("server") or {}).get("uptime_s")
    return plan_status, intent_cov, offload, governor, latency, uptime_s


def gather(base_url, diag_token):
    health = _fetch_json(f"{base_url}/api/health")
    layers = _fetch_json(f"{base_url}/api/data/layers")
    server_version = (layers or {}).get("server_version")
    daemon = diagnostic_entries = tier2_error_entries = ml = None
    if diag_token:
        daemon = _fetch_json(f"{base_url}/api/diag/daemon?token={diag_token}")
        audit_diag = _fetch_json(f"{base_url}/api/diag/audit?type=DIAGNOSTIC&limit=50&token={diag_token}")
        diagnostic_entries = (audit_diag or {}).get("entries", [])
        audit_t2 = _fetch_json(f"{base_url}/api/diag/audit?type=TIER2-ERROR&limit=50&token={diag_token}")
        tier2_error_entries = (audit_t2 or {}).get("entries", [])
        ml = _fetch_json(f"{base_url}/api/diag/ml?token={diag_token}")
    return health, daemon, diagnostic_entries, tier2_error_entries, ml, server_version


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--base-url", default=os.environ.get("VOLTRADE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--json", action="store_true", help="emit findings as JSON instead of text")
    ap.add_argument("--alert", action="store_true",
                    help="print ONE short message only when something needs action (silent when all OK)")
    ap.add_argument("--no-ops", action="store_true", help="skip the data-plane ops checks")
    args = ap.parse_args()

    diag_token = os.environ.get("DIAG_TOKEN", "")
    health, daemon, diagnostic_entries, tier2_error_entries, ml, server_version = gather(args.base_url, diag_token)
    local_version = read_local_package_version()
    findings = run_all_checks(
        health, daemon, diagnostic_entries, tier2_error_entries, ml,
        local_version=local_version, server_version=server_version,
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    prior_outage_state = load_outage_state()
    outage_state = compute_outage_state(health is not None, now_iso, prior_outage_state)
    if outage_state != prior_outage_state:
        save_outage_state(outage_state)
    findings.append(check_outage_duration(outage_state, now_iso))

    lv = ((health or {}).get("checks") or {}).get("bot", {}).get("liveness") or {}
    dark = bool(lv.get("dark"))
    wall_hours = lv.get("wallHours") or 0
    prior_notify_state = load_outage_state(path=DEFAULT_LIVENESS_NOTIFY_STATE_PATH)
    notify_state = compute_liveness_notify_state(dark, wall_hours, now_iso, prior_notify_state)
    if notify_state != prior_notify_state:
        save_outage_state(notify_state, path=DEFAULT_LIVENESS_NOTIFY_STATE_PATH)
    findings.append(check_liveness_notification(prior_notify_state, notify_state, wall_hours))

    if not args.no_ops:
        plan_status, intent_cov, offload, governor, latency, uptime_s = gather_ops(args.base_url, health)
        findings.extend(run_ops_checks(plan_status, intent_cov, offload, governor, latency, uptime_s=uptime_s))

    if args.alert:
        msg = alert_message(findings)
        if msg:
            print(msg)
        return overall_exit_code(findings)

    if args.json:
        print(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "findings": findings}, indent=2))
    else:
        if not diag_token:
            print("[session_health_check] DIAG_TOKEN not set — only /api/health checks ran\n")
        for f in findings:
            print(f"[{f['severity']:5s}] {f['label']}: {f['detail']}" + (f"  FIX: {f['fix']}" if f.get("fix") else ""))

    return overall_exit_code(findings)


if __name__ == "__main__":
    sys.exit(main())
