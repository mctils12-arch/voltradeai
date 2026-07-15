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
BROKEN #12)? That reasoning has now been done by hand across several
sessions (see research/open_questions.md's #21 update chain, 2026-07-13/14)
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
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

DEFAULT_BASE_URL = "https://voltradeai-production.up.railway.app"
DEFAULT_TIMEOUT_S = 20

OK, WARN, ALARM = "OK", "WARN", "ALARM"
_SEVERITY_RANK = {OK: 0, WARN: 1, ALARM: 2}


def finding(severity, label, detail):
    return {"severity": severity, "label": label, "detail": detail}


# ── pure classifiers (no network) ────────────────────────────────────────

def check_liveness(health):
    """CLAUDE.md Amendment 1 LIVENESS ALARM: the loop going dark is a
    TOP-OF-REPORT alarm, never a dashboard-only discovery. /api/health
    already computes this (server/liveness.ts) — read it, don't re-derive."""
    if not isinstance(health, dict):
        return finding(ALARM, "liveness", "no /api/health response — cannot confirm the loop is alive")
    bot = (health.get("checks") or {}).get("bot") or {}
    lv = bot.get("liveness") or {}
    if lv.get("dark"):
        return finding(
            ALARM, "liveness",
            f"LOOP DARK: {lv.get('detail', '(no detail)')} "
            f"(marketHours={lv.get('marketHours')}, wallHours={lv.get('wallHours')})",
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


def run_all_checks(health, daemon, diagnostic_entries, tier2_error_entries, ml):
    return [
        check_liveness(health),
        check_health_subsystems(health),
        check_alt_data_enrichment(diagnostic_entries),
        check_daemon_memory(daemon),
        check_tier2_daemon_timeouts(tier2_error_entries),
        check_ml_feedback(ml),
    ]


def overall_exit_code(findings):
    worst = max((_SEVERITY_RANK[f["severity"]] for f in findings), default=0)
    return {0: 0, 1: 1, 2: 2}[worst]


# ── network fetch (thin, not unit-tested directly) ───────────────────────

def _fetch_json(url, timeout=DEFAULT_TIMEOUT_S):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        print(f"[session_health_check] fetch failed for {url}: {e}", file=sys.stderr)
        return None


def gather(base_url, diag_token):
    health = _fetch_json(f"{base_url}/api/health")
    daemon = diagnostic_entries = tier2_error_entries = ml = None
    if diag_token:
        daemon = _fetch_json(f"{base_url}/api/diag/daemon?token={diag_token}")
        audit_diag = _fetch_json(f"{base_url}/api/diag/audit?type=DIAGNOSTIC&limit=50&token={diag_token}")
        diagnostic_entries = (audit_diag or {}).get("entries", [])
        audit_t2 = _fetch_json(f"{base_url}/api/diag/audit?type=TIER2-ERROR&limit=50&token={diag_token}")
        tier2_error_entries = (audit_t2 or {}).get("entries", [])
        ml = _fetch_json(f"{base_url}/api/diag/ml?token={diag_token}")
    return health, daemon, diagnostic_entries, tier2_error_entries, ml


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--base-url", default=os.environ.get("VOLTRADE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--json", action="store_true", help="emit findings as JSON instead of text")
    args = ap.parse_args()

    diag_token = os.environ.get("DIAG_TOKEN", "")
    health, daemon, diagnostic_entries, tier2_error_entries, ml = gather(args.base_url, diag_token)
    findings = run_all_checks(health, daemon, diagnostic_entries, tier2_error_entries, ml)

    if args.json:
        print(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "findings": findings}, indent=2))
    else:
        if not diag_token:
            print("[session_health_check] DIAG_TOKEN not set — only /api/health checks ran\n")
        for f in findings:
            print(f"[{f['severity']:5s}] {f['label']}: {f['detail']}")

    return overall_exit_code(findings)


if __name__ == "__main__":
    sys.exit(main())
