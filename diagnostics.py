"""
VolTradeAI Self-Diagnostic & Safety System
- Weekly loss limits
- Cache freshness monitoring
- Disk cleanup
- Model health tracking
- Anomaly detection (ML-based)
- Auto-fix: identifies problem source and adjusts behavior
"""
import json
import os
import time
import glob
from datetime import datetime, timedelta

try:
    from storage_config import (DATA_DIR, ML_MODEL_PATH, TRADE_FEEDBACK_PATH,
                                 HEALTH_LOG_PATH, INSIDER_CACHE_PATH, EVENT_MEMORY_PATH,
                                 CACHE_DIR as ALT_CACHE_DIR)
except ImportError:
    DATA_DIR = "/tmp"
    ML_MODEL_PATH = "/tmp/voltrade_ml_v2.pkl"  # v1.0.26: updated to ml_model_v2 path
    TRADE_FEEDBACK_PATH = "/tmp/voltrade_trade_feedback.json"
    HEALTH_LOG_PATH = "/tmp/voltrade_health_log.json"
    INSIDER_CACHE_PATH = "/tmp/voltrade_insider_cache.json"
    EVENT_MEMORY_PATH = "/tmp/voltrade_event_memory.json"
    ALT_CACHE_DIR = "/tmp/voltrade_alt_cache"

DIAG_PATH = os.path.join(DATA_DIR, "voltrade_diagnostics.json")
HEALTH_PATH = HEALTH_LOG_PATH
CACHE_DIR = ALT_CACHE_DIR
MAX_CACHE_SIZE_MB = 50  # Max total cache size
MAX_FILE_SIZE_MB = 5    # Max single file size
WEEKLY_LOSS_LIMIT = -8.0  # Pause if -8% in a week

# ── Safety Limits ─────────────────────────────────────────────────────────────

def check_weekly_loss(equity_history: list) -> dict:
    """
    Check if the AI engine has lost more than the weekly limit.
    equity_history: list of {"date": "YYYY-MM-DD", "equity": float}
    Returns: {"safe": bool, "weekly_pnl_pct": float, "action": "continue"/"pause"/"reduce"}
    """
    if not equity_history or len(equity_history) < 2:
        return {"safe": True, "weekly_pnl_pct": 0, "action": "continue"}

    # Get equity from 7 days ago vs now
    now_equity = equity_history[-1].get("equity", 0)
    week_ago = None
    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    for entry in equity_history:
        if entry.get("date", "") >= cutoff:
            week_ago = entry.get("equity", now_equity)
            break

    if not week_ago or week_ago == 0:
        return {"safe": True, "weekly_pnl_pct": 0, "action": "continue"}

    weekly_pnl_pct = ((now_equity - week_ago) / week_ago) * 100

    if weekly_pnl_pct <= WEEKLY_LOSS_LIMIT:
        return {
            "safe": False,
            "weekly_pnl_pct": round(weekly_pnl_pct, 2),
            "action": "pause",
            "reason": f"Weekly loss limit hit: {weekly_pnl_pct:.1f}% (limit: {WEEKLY_LOSS_LIMIT}%)"
        }
    elif weekly_pnl_pct <= WEEKLY_LOSS_LIMIT / 2:  # -4%
        return {
            "safe": True,
            "weekly_pnl_pct": round(weekly_pnl_pct, 2),
            "action": "reduce",
            "reason": f"Approaching weekly limit: {weekly_pnl_pct:.1f}% — reducing position sizes by 50%"
        }
    else:
        return {
            "safe": True,
            "weekly_pnl_pct": round(weekly_pnl_pct, 2),
            "action": "continue"
        }


# ── Cache Freshness Monitor ───────────────────────────────────────────────────

EXPECTED_CACHE_FRESHNESS = {
    "macro": {"file": "/tmp/voltrade_macro_cache.json", "max_age_hours": 1, "critical": True},
    "insider": {"file": INSIDER_CACHE_PATH, "max_age_hours": 2, "critical": False},
    "fred": {"file": "/tmp/voltrade_alt_cache/fred_macro_expanded.json", "max_age_hours": 8, "critical": False},
    "gdelt": {"file": "/tmp/voltrade_alt_cache/gdelt_risk.json", "max_age_hours": 4, "critical": False},
    "ml_model": {"file": ML_MODEL_PATH, "max_age_hours": 168, "critical": True},  # 7 days
    "event_memory": {"file": EVENT_MEMORY_PATH, "max_age_hours": 48, "critical": False},
    "trade_feedback": {"file": TRADE_FEEDBACK_PATH, "max_age_hours": 168, "critical": False},
}

def check_cache_freshness() -> dict:
    """
    Check if all data caches are fresh enough.
    Returns: {"all_fresh": bool, "stale_caches": [...], "position_size_multiplier": 0.5-1.0}
    """
    stale = []
    critical_stale = False

    for name, config in EXPECTED_CACHE_FRESHNESS.items():
        path = config["file"]
        max_age = config["max_age_hours"] * 3600

        if not os.path.exists(path):
            stale.append({"name": name, "status": "missing", "critical": config["critical"]})
            if config["critical"]:
                critical_stale = True
            continue

        age = time.time() - os.path.getmtime(path)
        if age > max_age * 1.5:  # 50% past expected freshness
            hours_old = round(age / 3600, 1)
            stale.append({
                "name": name,
                "status": f"stale ({hours_old}h old, expected <{config['max_age_hours']}h)",
                "critical": config["critical"],
            })
            if config["critical"]:
                critical_stale = True

    # Position size multiplier: reduce when data is stale
    if critical_stale:
        multiplier = 0.5  # Cut positions in half
    elif len(stale) > 3:
        multiplier = 0.7  # Reduce by 30%
    elif len(stale) > 0:
        multiplier = 0.85  # Slight reduction
    else:
        multiplier = 1.0

    return {
        "all_fresh": len(stale) == 0,
        "stale_caches": stale,
        "critical_stale": critical_stale,
        "position_size_multiplier": multiplier,
    }


# ── Disk Cleanup ──────────────────────────────────────────────────────────────

def cleanup_disk() -> dict:
    """
    Clean up /tmp cache files to prevent disk overflow.
    - Cap individual files at MAX_FILE_SIZE_MB
    - Cap total cache at MAX_CACHE_SIZE_MB
    - Remove oldest cache entries when over limit
    """
    cleaned = []
    total_size = 0

    # 1. Check all voltrade files in /tmp
    voltrade_files = glob.glob("/tmp/voltrade_*.json") + glob.glob("/tmp/voltrade_*.pkl")
    cache_files = glob.glob(os.path.join(CACHE_DIR, "*.json")) if os.path.exists(CACHE_DIR) else []
    all_files = voltrade_files + cache_files

    for filepath in all_files:
        try:
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            total_size += size

            # Cap individual files
            if size > MAX_FILE_SIZE_MB:
                # For JSON files, truncate by keeping only recent entries
                if filepath.endswith(".json"):
                    try:
                        with open(filepath) as f:
                            data = json.load(f)
                        if isinstance(data, list) and len(data) > 100:
                            data = data[-100:]  # Keep last 100
                            with open(filepath, "w") as f:
                                json.dump(data, f)
                            cleaned.append(f"Truncated {os.path.basename(filepath)} to 100 entries")
                        elif isinstance(data, dict) and len(data) > 200:
                            # Keep only the 200 most recently modified keys
                            keys = list(data.keys())[-200:]
                            data = {k: data[k] for k in keys}
                            with open(filepath, "w") as f:
                                json.dump(data, f)
                            cleaned.append(f"Truncated {os.path.basename(filepath)} to 200 keys")
                    except Exception:
                        pass
        except Exception:
            continue

    # 2. If total exceeds limit, remove oldest cache files
    if total_size > MAX_CACHE_SIZE_MB:
        cache_files_with_age = []
        for filepath in cache_files:
            try:
                age = time.time() - os.path.getmtime(filepath)
                cache_files_with_age.append((filepath, age))
            except Exception:
                continue

        # Sort by age (oldest first) and remove until under limit
        cache_files_with_age.sort(key=lambda x: -x[1])
        for filepath, age in cache_files_with_age:
            if total_size <= MAX_CACHE_SIZE_MB * 0.7:  # Clean to 70% capacity
                break
            try:
                size = os.path.getsize(filepath) / (1024 * 1024)
                os.remove(filepath)
                total_size -= size
                cleaned.append(f"Removed old cache: {os.path.basename(filepath)}")
            except Exception:
                continue

    return {
        "total_size_mb": round(total_size, 2),
        "files_cleaned": len(cleaned),
        "actions": cleaned,
    }


# ── Model Health Tracking ─────────────────────────────────────────────────────

def check_model_health() -> dict:
    """
    Check the ML model's health and performance.
    Tracks win rate, prediction accuracy, and retrain status.
    """
    result = {
        "model_exists": os.path.exists(ML_MODEL_PATH),
        "model_age_hours": 0,
        "retrain_needed": False,
        "retrain_overdue": False,
        "performance": {},
    }

    if result["model_exists"]:
        age = time.time() - os.path.getmtime(ML_MODEL_PATH)
        result["model_age_hours"] = round(age / 3600, 1)
        result["retrain_needed"] = age > 7 * 86400  # > 7 days
        result["retrain_overdue"] = age > 10 * 86400  # > 10 days

    # Check trade feedback for win rate
    try:
        feedback_path = TRADE_FEEDBACK_PATH
        if os.path.exists(feedback_path):
            with open(feedback_path) as f:
                trades = json.load(f)
            if trades:
                winners = [t for t in trades if t.get("pnl_pct", 0) > 0]
                losers = [t for t in trades if t.get("pnl_pct", 0) <= 0]
                result["performance"] = {
                    "total_trades": len(trades),
                    "win_rate": round(len(winners) / len(trades) * 100, 1) if trades else 0,
                    "avg_win": round(sum(t.get("pnl_pct", 0) for t in winners) / len(winners), 2) if winners else 0,
                    "avg_loss": round(sum(t.get("pnl_pct", 0) for t in losers) / len(losers), 2) if losers else 0,
                }

                # Win rate dropping? Flag it
                recent = trades[-20:] if len(trades) >= 20 else trades
                recent_winners = [t for t in recent if t.get("pnl_pct", 0) > 0]
                recent_win_rate = len(recent_winners) / len(recent) * 100 if recent else 0
                result["performance"]["recent_win_rate_20"] = round(recent_win_rate, 1)

                # If recent win rate is significantly below lifetime, flag degradation
                if (result["performance"]["win_rate"] > 0 and
                    recent_win_rate < result["performance"]["win_rate"] - 15):
                    result["performance"]["degradation_detected"] = True
                    result["retrain_needed"] = True  # Force retrain
    except Exception:
        pass

    return result


# ── Self-Diagnostic ML System ─────────────────────────────────────────────────

def run_diagnostics() -> dict:
    """
    Full self-diagnostic scan.
    Checks all systems, identifies problems, and recommends fixes.
    Returns a health report with actionable recommendations.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "problems": [],
        "warnings": [],
        "auto_fixes": [],
        "recommendations": [],
    }

    # 1. Cache freshness
    cache = check_cache_freshness()
    if cache["critical_stale"]:
        report["problems"].append({
            "system": "cache",
            "severity": "high",
            "message": f"Critical data sources stale: {[s['name'] for s in cache['stale_caches'] if s['critical']]}",
            "auto_fix": "reduce_position_size",
            "fix_params": {"multiplier": cache["position_size_multiplier"]},
        })
    elif cache["stale_caches"]:
        report["warnings"].append({
            "system": "cache",
            "message": f"{len(cache['stale_caches'])} data sources stale: {[s['name'] for s in cache['stale_caches']]}",
        })

    # 2. Model health
    model = check_model_health()
    if model["retrain_overdue"]:
        report["problems"].append({
            "system": "ml_model",
            "severity": "high",
            "message": f"ML model is {model['model_age_hours']:.0f} hours old (>10 days) — predictions degrading",
            "auto_fix": "force_retrain",
        })
    elif model["retrain_needed"]:
        report["warnings"].append({
            "system": "ml_model",
            "message": "ML model retrain needed",
        })

    perf = model.get("performance", {})
    if perf.get("degradation_detected"):
        report["problems"].append({
            "system": "ml_model",
            "severity": "high",
            "message": f"Win rate degradation: lifetime {perf.get('win_rate', 0)}% → recent {perf.get('recent_win_rate_20', 0)}%",
            "auto_fix": "force_retrain_and_reduce",
            "fix_params": {"multiplier": 0.5},
        })
    elif perf.get("total_trades", 0) > 20 and perf.get("win_rate", 50) < 40:
        report["problems"].append({
            "system": "ml_model",
            "severity": "medium",
            "message": f"Low win rate: {perf.get('win_rate', 0)}% over {perf.get('total_trades', 0)} trades",
            "auto_fix": "increase_score_threshold",
            "fix_params": {"min_score": 75},  # Only trade high-confidence picks
        })

    # 3. Disk usage
    disk = cleanup_disk()
    if disk["total_size_mb"] > MAX_CACHE_SIZE_MB * 0.8:
        report["warnings"].append({
            "system": "disk",
            "message": f"Cache using {disk['total_size_mb']}MB of {MAX_CACHE_SIZE_MB}MB limit",
        })
    if disk["files_cleaned"] > 0:
        report["auto_fixes"].extend(disk["actions"])

    # 4. API health — check if key data sources responded recently
    api_checks = {
        "polygon": os.path.exists("/tmp/voltrade_macro_cache.json"),
        "sec_edgar": os.path.exists("/tmp/voltrade_insider_cache.json"),
        "wikipedia": any(f.startswith("wiki_") for f in os.listdir(CACHE_DIR)) if os.path.exists(CACHE_DIR) else False,
        "gdelt": os.path.exists(os.path.join(CACHE_DIR, "gdelt_risk.json")) if os.path.exists(CACHE_DIR) else False,
        "fred": os.path.exists(os.path.join(CACHE_DIR, "fred_macro_expanded.json")) if os.path.exists(CACHE_DIR) else False,
    }
    failed_apis = [name for name, ok in api_checks.items() if not ok]
    if len(failed_apis) >= 3:
        report["problems"].append({
            "system": "api",
            "severity": "medium",
            "message": f"Multiple API sources down: {failed_apis}",
            "auto_fix": "reduce_position_size",
            "fix_params": {"multiplier": 0.6},
        })
    elif failed_apis:
        report["warnings"].append({
            "system": "api",
            "message": f"Some API sources unavailable: {failed_apis}",
        })

    # 5. Determine overall status
    high_problems = [p for p in report["problems"] if p.get("severity") == "high"]
    if high_problems:
        report["overall_status"] = "degraded"
    elif report["problems"]:
        report["overall_status"] = "warning"

    # 6. Generate recommendations
    if report["overall_status"] == "degraded":
        report["recommendations"].append("AI engine should reduce position sizes or pause until issues resolve")
    if model.get("retrain_needed"):
        report["recommendations"].append("Force ML model retrain to incorporate recent data")
    if perf.get("degradation_detected"):
        report["recommendations"].append("Win rate dropping — model may be overfitting, need more diverse training data")

    # Update dynamic blend ratio based on performance
    try:
        from storage_config import BLEND_TRACKER_PATH
        blend_path = BLEND_TRACKER_PATH
    except ImportError:
        blend_path = "/tmp/voltrade_blend_tracker.json"

    try:
        if perf.get("total_trades", 0) >= 20:
            win_rate = perf.get("win_rate", 50)
            recent_wr = perf.get("recent_win_rate_20", 50)

            # If recent performance is good, trust ML more
            if recent_wr > 60:
                ml_w = min(0.55, 0.4 + (recent_wr - 50) * 0.003)  # Max 55% ML
            elif recent_wr < 40:
                ml_w = max(0.2, 0.4 - (50 - recent_wr) * 0.004)  # Min 20% ML
            else:
                ml_w = 0.4

            rule_w = round(1.0 - ml_w, 2)
            ml_w = round(ml_w, 2)

            with open(blend_path, "w") as f:
                json.dump({"rule_weight": rule_w, "ml_weight": ml_w, "updated": datetime.now().isoformat()}, f)

            report["blend_weights"] = {"rule": rule_w, "ml": ml_w}
    except Exception:
        pass

    # Save report
    _save_report(report)

    return report


def get_auto_fix_params() -> dict:
    """
    Run diagnostics and return actionable parameters the AI engine should use.
    Called every scan cycle.
    Returns: {
        "position_size_multiplier": 0.5-1.0,
        "min_score_threshold": 65-80,
        "should_pause": bool,
        "force_retrain": bool,
        "problems_summary": str,
    }
    """
    report = run_diagnostics()

    params = {
        "position_size_multiplier": 1.0,
        "min_score_threshold": 65,
        "should_pause": False,
        "force_retrain": False,
        "problems_summary": "",
    }

    for problem in report["problems"]:
        fix = problem.get("auto_fix", "")
        fix_params = problem.get("fix_params", {})

        if fix == "reduce_position_size":
            params["position_size_multiplier"] = min(
                params["position_size_multiplier"],
                fix_params.get("multiplier", 0.7)
            )
        elif fix == "force_retrain":
            params["force_retrain"] = True
        elif fix == "force_retrain_and_reduce":
            params["force_retrain"] = True
            params["position_size_multiplier"] = min(
                params["position_size_multiplier"],
                fix_params.get("multiplier", 0.5)
            )
        elif fix == "increase_score_threshold":
            params["min_score_threshold"] = max(
                params["min_score_threshold"],
                fix_params.get("min_score", 75)
            )

    # Build summary
    summaries = []
    for p in report["problems"]:
        summaries.append(f"[{p['severity'].upper()}] {p['message']}")
    for w in report["warnings"]:
        summaries.append(f"[WARN] {w['message']}")
    params["problems_summary"] = " | ".join(summaries) if summaries else "All systems healthy"

    return params


def _save_report(report):
    """Save diagnostic report to health log."""
    try:
        history = []
        if os.path.exists(HEALTH_PATH):
            with open(HEALTH_PATH) as f:
                history = json.load(f)

        history.append(report)
        history = history[-100:]  # Keep last 100 reports

        with open(HEALTH_PATH, "w") as f:
            json.dump(history, f)
    except Exception:
        pass


if __name__ == "__main__":
    print("=== VolTradeAI Self-Diagnostic Report ===")
    report = run_diagnostics()
    print(json.dumps(report, indent=2, default=str))
    print("\n=== Auto-Fix Parameters ===")
    params = get_auto_fix_params()
    print(json.dumps(params, indent=2))
