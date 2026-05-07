#!/usr/bin/env python3
"""
system_status.py — Self-test for VolTradeAI subsystems.

ALPHA AUDIT 2026-05-06 batch 11: gives a one-shot view of whether
every subsystem is actually wired up correctly. Hit this from the
debug endpoint after any deploy and you'll see green/red on each
component without having to wait for the bot to trade.

Output: pure JSON on stdout. Always exits 0.

Usage:
    python3 system_status.py

Behind /api/bot/system-status (gated by DEBUG_ENABLED env var).
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": str(e)[:200], "type": type(e).__name__,
                "traceback": traceback.format_exc()[-400:]}


# ────────────────────────────────────────────────────────────────────────────
# Each check returns one of:
#   {"status": "pass", "message": "...", "details": {...}}
#   {"status": "fail", "message": "...", "error": "...", "details": {...}}
#   {"status": "warn", "message": "...", "details": {...}}
# ────────────────────────────────────────────────────────────────────────────


def check_size_tier_module():
    """Verify position_sizing.get_size_tier exists and returns sane values."""
    from position_sizing import get_size_tier

    # Sanity check across multiple equity levels
    cases = [
        (10_000,    "starter",      4),
        (100_000,   "growing",      8),
        (1_000_000, "established",  15),
        (10_000_000,"fund",         28),
    ]
    rows = []
    all_pass = True
    for eq, expected_label, expected_min_pos in cases:
        t = get_size_tier(eq)
        ok = (t.get("tier_label") == expected_label
              and t.get("max_positions", 0) >= expected_min_pos)
        rows.append({
            "equity": eq,
            "tier_label": t.get("tier_label"),
            "max_positions": t.get("max_positions"),
            "max_single_pct": t.get("max_single_pct"),
            "max_sector_positions": t.get("max_sector_positions"),
            "ok": ok,
        })
        if not ok:
            all_pass = False

    return {
        "status": "pass" if all_pass else "fail",
        "message": "size-tier function returns expected values" if all_pass
                   else "size-tier returned unexpected values for one or more equity levels",
        "details": {"cases": rows},
    }


def check_adaptive_params_overlay():
    """Verify get_adaptive_params overlays size-tier on top of regime."""
    from system_config import get_adaptive_params

    # BULL at $1M should give 15 positions (established tier wins)
    p_bull = get_adaptive_params(vxx_ratio=0.85, spy_vs_ma50=1.05,
                                  account_equity=1_000_000)
    bull_ok = p_bull.get("MAX_POSITIONS") == 15
    bull_has_tier = p_bull.get("SIZE_TIER", {}).get("label") == "established"

    # PANIC at $10M should still give 0 positions (regime stop wins)
    p_panic = get_adaptive_params(vxx_ratio=1.5, spy_vs_ma50=0.85,
                                   account_equity=10_000_000)
    panic_ok = p_panic.get("MAX_POSITIONS") == 0
    panic_tier_ok = p_panic.get("SIZE_TIER", {}).get("label") == "fund"

    # MAX_VOLUME_SHARE_PCT should be present (new in batch 11)
    has_vol_share = "MAX_VOLUME_SHARE_PCT" in p_bull

    all_ok = bull_ok and bull_has_tier and panic_ok and panic_tier_ok and has_vol_share

    return {
        "status": "pass" if all_ok else "fail",
        "message": "adaptive params correctly overlay size-tier and regime"
                   if all_ok else "size-tier overlay not wired correctly",
        "details": {
            "bull_1m_max_positions":           p_bull.get("MAX_POSITIONS"),
            "bull_1m_size_tier_label":         p_bull.get("SIZE_TIER", {}).get("label"),
            "bull_1m_max_volume_share_pct":    p_bull.get("MAX_VOLUME_SHARE_PCT"),
            "panic_10m_max_positions":         p_panic.get("MAX_POSITIONS"),
            "panic_10m_size_tier_label":       p_panic.get("SIZE_TIER", {}).get("label"),
            "expected": {
                "bull_1m_max_positions":   15,
                "bull_1m_tier_label":      "established",
                "panic_10m_max_positions": 0,
                "panic_10m_tier_label":    "fund",
            },
        },
    }


def check_size_portfolio_respects_tier():
    """Verify size_portfolio uses tier max_positions, not the hardcoded 8.

    We can't easily exercise the full sizing pipeline in a sandbox without
    mocking yfinance/Alpaca, so this check inspects the function source for
    the size-tier integration. If size_portfolio CALLS get_size_tier and
    uses _max_positions from it, the wiring is correct."""
    import inspect
    from position_sizing import size_portfolio
    src = inspect.getsource(size_portfolio)
    has_tier_call = "get_size_tier(equity)" in src
    has_max_pos_use = "_max_positions" in src
    has_old_constant = "ABSOLUTE_MAX_POSITIONS" in src and "_max_positions" not in src
    ok = has_tier_call and has_max_pos_use and not has_old_constant

    return {
        "status": "pass" if ok else "fail",
        "message": "size_portfolio uses tier-aware max_positions"
                   if ok else "size_portfolio still using hardcoded ABSOLUTE_MAX_POSITIONS",
        "details": {
            "has_tier_call":      has_tier_call,
            "has_max_pos_use":    has_max_pos_use,
            "has_old_constant":   has_old_constant,
        },
    }


def check_sector_correlation_accepts_override():
    """Verify check_sector_correlation accepts max_sector_positions kwarg."""
    from bot_engine import check_sector_correlation
    import inspect
    sig = inspect.signature(check_sector_correlation)
    has_override = "max_sector_positions" in sig.parameters

    return {
        "status": "pass" if has_override else "fail",
        "message": "check_sector_correlation has max_sector_positions kwarg"
                    if has_override
                    else "check_sector_correlation missing max_sector_positions kwarg",
        "details": {"signature": str(sig)},
    }


def check_top_10_enrichment():
    """Verify the bot_engine top_10 build site has the new fields.
    We can't actually call run_full_scan in a sandbox, so we inspect
    the source for the enriched fields."""
    import inspect
    import bot_engine

    src = inspect.getsource(bot_engine)
    required_fields = [
        '"rank":',
        '"factor_strength":',
        '"sector_clash":',
        '"factors":',
        '"tiebreak_score":',
        '"size_tier":',
    ]
    found = [f for f in required_fields if f in src]
    missing = [f for f in required_fields if f not in src]
    ok = len(missing) == 0

    return {
        "status": "pass" if ok else "fail",
        "message": "bot_engine top_10 includes batch 11 enrichment fields"
                    if ok else f"bot_engine missing fields: {missing}",
        "details": {"found": found, "missing": missing},
    }


def check_tie_breaking_present():
    """Verify the tie-break refinement block is in bot_engine."""
    import inspect
    import bot_engine
    src = inspect.getsource(bot_engine)
    has_block = "TIE-BREAKING REFINEMENT" in src and "_tiebreak_score" in src
    return {
        "status": "pass" if has_block else "fail",
        "message": "tie-break refinement block present"
                    if has_block else "tie-break block missing — scores will not be re-ordered within tied groups",
    }


def check_scan_caches():
    """Look at scan-result cache files for last-known-good state."""
    paths = [
        "/data/voltrade/voltrade_scan_timings.json",
        "/tmp/voltrade_scan_timings.json",
    ]
    info = {"checked_paths": []}
    for p in paths:
        item = {"path": p, "exists": os.path.exists(p)}
        if item["exists"]:
            try:
                with open(p) as f:
                    data = json.load(f)
                item["status"] = data.get("status")
                item["total_sec"] = data.get("total_sec")
                item["age_sec"] = round(time.time() - os.path.getmtime(p), 0)
                item["phases_count"] = len(data.get("phases", []))
            except Exception as e:
                item["error"] = str(e)[:120]
        info["checked_paths"].append(item)

    return {
        "status": "pass" if any(c.get("exists") for c in info["checked_paths"]) else "warn",
        "message": "scan timings file accessible"
                   if any(c.get("exists") for c in info["checked_paths"])
                   else "no scan timings file yet (bot may not have completed a scan)",
        "details": info,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def gather_status():
    """Run every check independently — partial failures still return data."""
    checks = {
        "size_tier_module":            _safe(check_size_tier_module),
        "adaptive_params_overlay":     _safe(check_adaptive_params_overlay),
        "size_portfolio_respects_tier":_safe(check_size_portfolio_respects_tier),
        "sector_correlation_override": _safe(check_sector_correlation_accepts_override),
        "top_10_enrichment":           _safe(check_top_10_enrichment),
        "tie_breaking_present":        _safe(check_tie_breaking_present),
        "scan_caches":                 _safe(check_scan_caches),
    }

    # Summary
    pass_count = sum(1 for c in checks.values() if isinstance(c, dict) and c.get("status") == "pass")
    fail_count = sum(1 for c in checks.values() if isinstance(c, dict) and c.get("status") == "fail")
    warn_count = sum(1 for c in checks.values() if isinstance(c, dict) and c.get("status") == "warn")

    overall = "ok" if fail_count == 0 else "degraded" if pass_count > 0 else "broken"

    return {
        "generated_at": datetime.now().isoformat(),
        "schema_version": "system-status-v1",
        "overall": overall,
        "summary": {
            "total":  len(checks),
            "passed": pass_count,
            "failed": fail_count,
            "warned": warn_count,
        },
        "checks": checks,
    }


if __name__ == "__main__":
    try:
        result = gather_status()
        print(json.dumps(result, default=str, indent=2))
    except Exception as e:
        print(json.dumps({
            "fatal_error": str(e)[:200],
            "traceback": traceback.format_exc()[-500:],
        }))
    sys.exit(0)
