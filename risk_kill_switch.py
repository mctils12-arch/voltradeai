#!/usr/bin/env python3
"""
VolTradeAI — Risk Kill Switch
==============================
Hard portfolio-level and regime kill switches. Runs before any tier considers
trading. When fired, blocks ALL new entries and can optionally liquidate all
positions.

KILL SWITCHES
  1. Portfolio drawdown kill: -20% from peak equity → fire
  2. Daily loss limit:        -3% in one session → block entries today
  3. Correlation cap:         > 60% BP in correlated names → reject new corr
  4. Margin buffer breach:    < 20% free BP → force-reduce
  5. Regime kill:             VXX ratio > 1.40 → T2-T4 off, T1 minimum
  6. Consecutive loss kill:   5 consecutive losers → pause 24h
  7. Manual kill:             operator-set killswitch file → block all

PERSISTENCE
  Kill state lives in /data/voltrade/voltrade_killswitch.json.
  Atomic writes. Survives deploys. Manual reset requires editing the file or
  calling reset_kill_state().

INTEGRATION
  from risk_kill_switch import check_kill_switches, record_trade_outcome
  status = check_kill_switches(equity, peak, positions, daily_pnl_pct, vxx_ratio)
  if status["killed"]:
      return []  # block all new entries
"""

from __future__ import annotations
import os
import json
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

logger = logging.getLogger("voltrade.killswitch")

# ── File paths ───────────────────────────────────────────────────────────────
KILLSWITCH_PATH = os.path.join(DATA_DIR, "voltrade_killswitch.json")
KILL_HISTORY_PATH = os.path.join(DATA_DIR, "voltrade_kill_history.json")
MANUAL_KILL_PATH = os.path.join(DATA_DIR, "voltrade_MANUAL_KILL")  # touch file

# ── Risk thresholds ──────────────────────────────────────────────────────────
PORTFOLIO_DD_KILL = -0.20           # -20% from peak
DAILY_LOSS_LIMIT = -0.03            # -3% in one day
CONSECUTIVE_LOSS_KILL = 5           # 5 losers in a row
CORRELATION_CAP = 0.60              # max 60% BP in correlated names
MIN_FREE_BP = 0.00                  # USER: 100% invested — no BP buffer; margin calls have no cushion
REGIME_KILL_VXX = 1.40              # VXX ratio above this

# Recovery gates (must ALL pass to auto-resume after kill)
RESUME_REQUIRES_EQUITY_RECOVERY = 0.95  # equity must recover to 95% of peak
RESUME_REQUIRES_REGIME_CALM = True      # regime must be NEUTRAL/BULL
RESUME_COOLDOWN_HOURS = 4               # min 4h after kill before auto-resume

# Correlation groups (trade the same direction during stress)
CORRELATION_GROUPS = {
    "MEGA_CAP_TECH":  {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA"},
    "SEMIS":          {"AMD", "NVDA", "INTC", "AVGO", "QCOM", "AMAT", "LRCX", "TXN"},
    "FIN_MEGA":       {"JPM", "BAC", "WFC", "C", "GS", "MS"},
    "OIL_MAJOR":      {"XOM", "CVX", "COP"},
    "CONSUMER":       {"WMT", "COST", "HD", "NKE", "MCD", "SBUX"},
    "BIOTECH":        {"AMGN", "REGN", "VRTX", "GILD"},
    "INDEX_ETF":      {"SPY", "QQQ", "IWM", "DIA"},
}


def _which_group(ticker: str) -> Optional[str]:
    """Return the correlation group name for a ticker, or None."""
    t = ticker.upper()
    for group, members in CORRELATION_GROUPS.items():
        if t in members:
            return group
    return None


# ── Persistent state ─────────────────────────────────────────────────────────
def _load_state() -> Dict:
    try:
        if os.path.exists(KILLSWITCH_PATH):
            with open(KILLSWITCH_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "killed": False,
        "kill_reason": "",
        "killed_at": None,
        "peak_equity": 0.0,
        "consecutive_losses": 0,
        "last_loss_ts": None,
        "daily_loss_date": None,
        "daily_loss_pct": 0.0,
    }


def _save_state(state: Dict) -> None:
    try:
        dirname = os.path.dirname(KILLSWITCH_PATH) or "."
        fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".ks.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, KILLSWITCH_PATH)
        except Exception:
            try: os.unlink(tmp)
            except Exception: pass
    except Exception as e:
        logger.error(f"killswitch state save failed: {e}")


def _log_kill_event(reason: str, details: Dict) -> None:
    """Append to kill history for post-mortem analysis."""
    try:
        events = []
        if os.path.exists(KILL_HISTORY_PATH):
            try:
                with open(KILL_HISTORY_PATH) as f:
                    events = json.load(f)
            except Exception:
                events = []
        events.append({
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "details": details,
        })
        # Keep last 500 events
        events = events[-500:]
        with open(KILL_HISTORY_PATH, "w") as f:
            json.dump(events, f, indent=2, default=str)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def check_kill_switches(
    equity: float,
    peak_equity: float = None,
    positions: Optional[List[Dict]] = None,
    daily_pnl_pct: float = 0.0,
    vxx_ratio: float = 1.0,
    buying_power: float = 0.0,
) -> Dict[str, Any]:
    """
    Master check before any new trades. Returns structured status.

    Args:
        equity:         current account equity
        peak_equity:    all-time peak (auto-tracked if None)
        positions:      current Alpaca positions list
        daily_pnl_pct:  today's P&L as fraction of equity
        vxx_ratio:      VXX / 30d avg VXX
        buying_power:   Alpaca-reported BP

    Returns:
        {
            "killed": bool,              # overall kill state
            "kill_reason": str,
            "blocks_entries": bool,      # subset — daily limit blocks entries but
                                          # doesn't kill everything
            "tier_blocks": {
                "t1": bool, "t2": bool, "t3": bool, "t4": bool
            },
            "warnings": [str, ...],
            "can_auto_resume": bool,
        }
    """
    state = _load_state()
    positions = positions or []
    warnings = []

    # Update peak equity
    if peak_equity is None:
        peak_equity = max(state.get("peak_equity", 0), equity)
    else:
        peak_equity = max(peak_equity, equity, state.get("peak_equity", 0))
    state["peak_equity"] = peak_equity

    # Compute drawdown
    dd_pct = (equity - peak_equity) / max(peak_equity, 1)

    # ── 1. Manual kill (operator touched the killswitch file) ────────────
    if os.path.exists(MANUAL_KILL_PATH):
        if not state.get("killed"):
            state["killed"] = True
            state["kill_reason"] = "MANUAL kill file present"
            state["killed_at"] = datetime.now().isoformat()
            _save_state(state)
            _log_kill_event("MANUAL", {"equity": equity})
        return {
            "killed": True,
            "kill_reason": "MANUAL kill file present",
            "blocks_entries": True,
            "tier_blocks": {"t1": True, "t2": True, "t3": True, "t4": True},
            "warnings": warnings,
            "can_auto_resume": False,
        }

    # ── 2. Portfolio drawdown kill ───────────────────────────────────────
    if dd_pct <= PORTFOLIO_DD_KILL:
        if not state.get("killed"):
            state["killed"] = True
            state["kill_reason"] = f"Portfolio DD {dd_pct*100:.1f}% <= {PORTFOLIO_DD_KILL*100:.0f}%"
            state["killed_at"] = datetime.now().isoformat()
            _save_state(state)
            _log_kill_event("DRAWDOWN", {
                "equity": equity, "peak": peak_equity, "dd_pct": dd_pct,
            })
        return {
            "killed": True,
            "kill_reason": state["kill_reason"],
            "blocks_entries": True,
            "tier_blocks": {"t1": True, "t2": True, "t3": True, "t4": True},
            "warnings": warnings,
            "can_auto_resume": False,  # requires equity recovery + manual review
        }

    # ── 3. Consecutive loss kill ─────────────────────────────────────────
    consec_losses = int(state.get("consecutive_losses", 0) or 0)
    if consec_losses >= CONSECUTIVE_LOSS_KILL:
        last_loss = state.get("last_loss_ts")
        if last_loss:
            try:
                loss_time = datetime.fromisoformat(str(last_loss).replace("Z", "+00:00"))
                hours_since = (datetime.now() - loss_time.replace(tzinfo=None)).total_seconds() / 3600
                if hours_since < 24:
                    return {
                        "killed": True,
                        "kill_reason": f"{consec_losses} consecutive losses; pause 24h",
                        "blocks_entries": True,
                        "tier_blocks": {"t1": True, "t2": True, "t3": True, "t4": False},
                        "warnings": warnings,
                        "can_auto_resume": True,
                    }
                else:
                    # Auto-reset after 24h
                    state["consecutive_losses"] = 0
                    _save_state(state)
            except Exception:
                pass

    # ── 4. Check for auto-resume from prior kill ─────────────────────────
    if state.get("killed"):
        killed_at = state.get("killed_at")
        can_resume = False
        if killed_at:
            try:
                kt = datetime.fromisoformat(str(killed_at).replace("Z", "+00:00"))
                hours = (datetime.now() - kt.replace(tzinfo=None)).total_seconds() / 3600
                # All resume gates must pass
                cooldown_ok = hours >= RESUME_COOLDOWN_HOURS
                equity_ok = equity >= peak_equity * RESUME_REQUIRES_EQUITY_RECOVERY
                regime_ok = vxx_ratio < 1.20  # below stress threshold
                can_resume = cooldown_ok and equity_ok and regime_ok
            except Exception:
                pass

        if can_resume:
            state["killed"] = False
            state["kill_reason"] = "auto-resumed"
            _save_state(state)
            _log_kill_event("RESUME", {"equity": equity, "dd_pct": dd_pct})
            logger.info("Kill switch auto-resumed (cooldown + equity + regime)")
        else:
            return {
                "killed": True,
                "kill_reason": state.get("kill_reason", "prior kill"),
                "blocks_entries": True,
                "tier_blocks": {"t1": True, "t2": True, "t3": True, "t4": True},
                "warnings": warnings,
                "can_auto_resume": can_resume,
            }

    # ── 5. Daily loss limit (blocks new entries, doesn't kill) ────────────
    today_str = datetime.now().strftime("%Y-%m-%d")
    state_date = state.get("daily_loss_date")
    if state_date != today_str:
        state["daily_loss_date"] = today_str
        state["daily_loss_pct"] = daily_pnl_pct
    else:
        state["daily_loss_pct"] = min(state.get("daily_loss_pct", 0), daily_pnl_pct)

    daily_limit_hit = state["daily_loss_pct"] <= DAILY_LOSS_LIMIT
    if daily_limit_hit:
        warnings.append(f"Daily loss limit hit: {state['daily_loss_pct']*100:.1f}%")

    # ── 6. Regime kill (tier-selective) ──────────────────────────────────
    regime_killed_t234 = vxx_ratio >= REGIME_KILL_VXX
    if regime_killed_t234:
        warnings.append(f"Regime kill for T2-T4 (VXX ratio {vxx_ratio:.2f})")

    # ── 7. Correlation cap (checked per-trade, surfaced as warning here) ──
    corr_warning = _check_correlation(positions, buying_power or equity)
    if corr_warning:
        warnings.append(corr_warning)

    # ── 8. Margin buffer ─────────────────────────────────────────────────
    if buying_power > 0:
        bp_used = sum(abs(float(p.get("market_value", 0))) for p in positions)
        free_bp_pct = 1.0 - (bp_used / max(buying_power, 1))
        if free_bp_pct < MIN_FREE_BP:
            warnings.append(f"Free BP below {MIN_FREE_BP*100:.0f}%: {free_bp_pct*100:.1f}%")

    # Save updated state
    _save_state(state)

    return {
        "killed": False,
        "kill_reason": "",
        "blocks_entries": daily_limit_hit,
        "tier_blocks": {
            "t1": daily_limit_hit,  # entries blocked for today
            "t2": daily_limit_hit or regime_killed_t234,
            "t3": daily_limit_hit or regime_killed_t234,
            "t4": regime_killed_t234,  # still blocks hedge in extreme vol
        },
        "warnings": warnings,
        "can_auto_resume": True,
    }


def _check_correlation(positions: List[Dict], total_bp: float) -> Optional[str]:
    """Check if any correlation group exceeds its allocation cap."""
    if total_bp <= 0:
        return None
    group_exposure: Dict[str, float] = {}
    for p in positions:
        sym = p.get("symbol", "").upper()
        val = abs(float(p.get("market_value", 0)))
        group = _which_group(sym)
        if group:
            group_exposure[group] = group_exposure.get(group, 0) + val
    for group, exposure in group_exposure.items():
        pct = exposure / total_bp
        if pct > CORRELATION_CAP:
            return f"Correlation cap breach: {group} at {pct*100:.1f}% (max {CORRELATION_CAP*100:.0f}%)"
    return None


def record_trade_outcome(pnl_pct: float) -> None:
    """Call this from ml_model_v2.track_fill when a trade closes.
    Updates consecutive-loss counter used by the kill switch."""
    state = _load_state()
    if pnl_pct < 0:
        state["consecutive_losses"] = int(state.get("consecutive_losses", 0) or 0) + 1
        state["last_loss_ts"] = datetime.now().isoformat()
    else:
        # Win → reset streak
        state["consecutive_losses"] = 0
    _save_state(state)


def is_ticker_blocked_by_correlation(
    ticker: str, positions: List[Dict], total_bp: float,
    new_size_pct: float
) -> bool:
    """Check if adding `new_size_pct` of `ticker` would breach correlation cap.
    Called per-candidate-trade."""
    group = _which_group(ticker)
    if not group:
        return False
    current_group_bp = sum(
        abs(float(p.get("market_value", 0))) for p in positions
        if _which_group(p.get("symbol", "")) == group
    )
    if total_bp <= 0:
        return False
    projected_pct = (current_group_bp + new_size_pct * total_bp) / total_bp
    return projected_pct > CORRELATION_CAP


def reset_kill_state() -> None:
    """Manual reset — operator-invoked after reviewing a kill event."""
    _save_state({
        "killed": False,
        "kill_reason": "",
        "killed_at": None,
        "peak_equity": 0.0,  # will re-peak on next check
        "consecutive_losses": 0,
        "last_loss_ts": None,
        "daily_loss_date": None,
        "daily_loss_pct": 0.0,
    })
    # Remove manual kill file if present
    try:
        if os.path.exists(MANUAL_KILL_PATH):
            os.unlink(MANUAL_KILL_PATH)
    except Exception:
        pass
    logger.warning("Kill switch state manually reset")


def get_kill_status() -> Dict:
    """Read-only status for dashboards."""
    state = _load_state()
    manual_flag = os.path.exists(MANUAL_KILL_PATH)
    # Load recent kill events
    events = []
    try:
        if os.path.exists(KILL_HISTORY_PATH):
            with open(KILL_HISTORY_PATH) as f:
                events = json.load(f)[-10:]  # last 10
    except Exception:
        pass
    return {
        "killed": state.get("killed", False),
        "kill_reason": state.get("kill_reason", ""),
        "killed_at": state.get("killed_at"),
        "peak_equity": state.get("peak_equity", 0),
        "consecutive_losses": state.get("consecutive_losses", 0),
        "daily_loss_pct": state.get("daily_loss_pct", 0),
        "manual_flag_present": manual_flag,
        "recent_kills": events,
    }


# ── CLI for operators ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        print(json.dumps(get_kill_status(), indent=2, default=str))
    elif cmd == "reset":
        reset_kill_state()
        print("Kill switch reset.")
    elif cmd == "kill":
        # Create manual kill file
        try:
            with open(MANUAL_KILL_PATH, "w") as f:
                f.write(f"killed at {datetime.now().isoformat()}\n")
            print(f"Manual kill file created: {MANUAL_KILL_PATH}")
        except Exception as e:
            print(f"Failed: {e}")
    elif cmd == "test":
        # Smoke test
        r = check_kill_switches(
            equity=100000, peak_equity=100000, positions=[],
            daily_pnl_pct=-0.01, vxx_ratio=1.1, buying_power=200000,
        )
        print(json.dumps(r, indent=2))
    else:
        print("Usage: python3 risk_kill_switch.py [status | reset | kill | test]")



# ═══════════════════════════════════════════════════════════════════════
# ADDITIONS 2026-04-20: Position-level kill + correlation + liquidation
# ═══════════════════════════════════════════════════════════════════════

# Per-position loss limit (tighter than system-wide STOP_LOSS_PCT of 15%)
# Catches runaway single-name losses faster than portfolio-DD kill.
POSITION_KILL_LOSS_PCT = -0.25   # close position if down 25%
POSITION_WARN_LOSS_PCT = -0.15   # warn if down 15%


def check_position_risk(position: dict) -> dict:
    """
    Check a single position for per-position risk limits.
    Returns {'action': 'liquidate'|'warn'|'none', 'reason': str, 'pnl_pct': float}

    Usage (from bot.ts tier 1 reflex loop):
        for pos in positions:
            risk = check_position_risk(pos)
            if risk['action'] == 'liquidate':
                close_position(pos['symbol'])
                audit('POS-KILL', f"{pos['symbol']}: {risk['reason']}")
    """
    try:
        avg_entry = float(position.get("avg_entry_price", 0) or 0)
        current = float(position.get("current_price", avg_entry) or avg_entry)
        qty = float(position.get("qty", 0) or 0)
        if avg_entry <= 0 or qty == 0:
            return {"action": "none", "reason": "invalid_data", "pnl_pct": 0.0}

        # Compute pnl% (handles both long and short positions)
        if qty > 0:  # long
            pnl_pct = (current - avg_entry) / avg_entry
        else:  # short (qty < 0)
            pnl_pct = (avg_entry - current) / avg_entry

        pnl_pct = round(pnl_pct, 4)

        if pnl_pct <= POSITION_KILL_LOSS_PCT:
            return {
                "action": "liquidate",
                "reason": f"position_kill: pnl {pnl_pct*100:.1f}% <= {POSITION_KILL_LOSS_PCT*100:.0f}%",
                "pnl_pct": pnl_pct,
            }
        elif pnl_pct <= POSITION_WARN_LOSS_PCT:
            return {
                "action": "warn",
                "reason": f"position_warn: pnl {pnl_pct*100:.1f}%",
                "pnl_pct": pnl_pct,
            }
        return {"action": "none", "reason": "healthy", "pnl_pct": pnl_pct}
    except Exception as e:
        return {"action": "none", "reason": f"error:{e}", "pnl_pct": 0.0}


# ─── Correlation pre-trade check ───────────────────────────────────────
# Before adding a new position, check what fraction of existing book
# shares sector/tier exposure. Prevents over-concentration.

# Simple sector lookup — extend as needed
_SECTOR_MAP = {
    # Tech
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "AMD": "tech", "META": "tech",
    "GOOGL": "tech", "TSLA": "tech", "AMZN": "tech", "ORCL": "tech", "CRM": "tech",
    "INTC": "tech", "AVGO": "tech", "QCOM": "tech", "CSCO": "tech", "ADBE": "tech",
    # Financials
    "JPM": "fin", "BAC": "fin", "WFC": "fin", "C": "fin", "GS": "fin", "MS": "fin",
    # Index/broad
    "SPY": "broad", "QQQ": "broad", "IWM": "broad", "DIA": "broad",
    # Vol
    "VXX": "vol", "UVXY": "vol", "SVXY": "vol",
}


def get_sector(ticker: str) -> str:
    """Rough sector classification. Returns 'other' for unknown tickers."""
    return _SECTOR_MAP.get(ticker.upper(), "other")


def check_correlation_pre_trade(new_ticker: str, positions: list,
                                max_sector_pct: float = 0.40) -> dict:
    """
    Check if adding new_ticker would over-concentrate the book.
    Returns {'allowed': bool, 'reason': str, 'sector_pct_after': float}

    max_sector_pct = 40% means no sector can exceed 40% of total exposure.
    """
    new_sector = get_sector(new_ticker)
    if new_sector == "other":
        # Unknown ticker — allow but note it
        return {"allowed": True, "reason": "unknown_sector", "sector_pct_after": 0.0}

    try:
        total_exposure = sum(
            abs(float(p.get("market_value", 0) or 0)) for p in positions
        )
        if total_exposure <= 0:
            return {"allowed": True, "reason": "empty_book", "sector_pct_after": 0.0}

        sector_exposure = sum(
            abs(float(p.get("market_value", 0) or 0))
            for p in positions
            if get_sector(p.get("symbol", "")) == new_sector
        )
        # Assume new position will be ~5% of total (rough — actual size varies)
        estimated_new = total_exposure * 0.05
        sector_pct_after = (sector_exposure + estimated_new) / (total_exposure + estimated_new)

        if sector_pct_after > max_sector_pct:
            return {
                "allowed": False,
                "reason": f"{new_sector} sector would be {sector_pct_after*100:.0f}% > {max_sector_pct*100:.0f}% cap",
                "sector_pct_after": round(sector_pct_after, 3),
            }

        return {
            "allowed": True,
            "reason": "ok",
            "sector_pct_after": round(sector_pct_after, 3),
        }
    except Exception as e:
        # On error, allow (fail-open — don't block trades on our bug)
        return {"allowed": True, "reason": f"error:{e}", "sector_pct_after": 0.0}


# ─── Liquidation-on-kill option ────────────────────────────────────────
# If portfolio DD hits -25% (not just -20% block threshold), force-close
# all positions rather than just blocking new trades. User opt-in via
# env var VOLTRADE_LIQUIDATE_ON_KILL=true.

import os as _os

def should_liquidate_all(dd_pct: float) -> bool:
    """
    Check if we should force-liquidate the book.
    Returns True only if:
      1. VOLTRADE_LIQUIDATE_ON_KILL=true in env
      2. dd_pct <= -25% (beyond normal -20% block threshold)
    """
    if _os.environ.get("VOLTRADE_LIQUIDATE_ON_KILL", "").lower() != "true":
        return False
    return dd_pct <= -0.25


def liquidation_reason(dd_pct: float) -> str:
    """Human-readable reason for audit logs."""
    return f"LIQUIDATE_ALL: portfolio DD {dd_pct*100:.1f}% <= -25% with opt-in active"



def get_kill_switch_status() -> dict:
    """
    Return current kill-switch state for monitoring dashboards.
    Non-mutating — safe to call frequently.
    """
    try:
        peak = get_peak_equity()
    except Exception:
        peak = 0.0
    try:
        consec = _get_consecutive_losses()
    except Exception:
        consec = 0
    try:
        daily_pnl = _get_daily_pnl()
    except Exception:
        daily_pnl = 0.0
    return {
        "peak_equity": peak,
        "consecutive_losses": consec,
        "consecutive_loss_limit": CONSECUTIVE_LOSS_KILL,  # HOTFIX 2026-04-22: was undefined CONSECUTIVE_LOSS_LIMIT
        "daily_pnl_pct": round(daily_pnl * 100, 2),
        "daily_loss_limit_pct": round(DAILY_LOSS_LIMIT * 100, 2),  # HOTFIX 2026-04-22: was undefined DAILY_LOSS_KILL
        "portfolio_dd_kill_pct": round(PORTFOLIO_DD_KILL * 100, 2),
        "vxx_regime_kill_threshold": REGIME_KILL_VXX,  # HOTFIX 2026-04-22: was undefined VXX_REGIME_KILL
        "min_free_bp": MIN_FREE_BP,
    }


def _get_consecutive_losses() -> int:
    """Best-effort read of consecutive loss count from state file."""
    try:
        import json
        path = _state_path("consec_losses.json")
        if os.path.exists(path):
            with open(path) as f:
                return int(json.load(f).get("count", 0))
    except Exception:
        pass
    return 0


def _get_daily_pnl() -> float:
    """Best-effort read of today's P&L percent from state file."""
    try:
        import json
        from datetime import datetime
        path = _state_path("daily_pnl.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            today = datetime.utcnow().strftime("%Y-%m-%d")
            if data.get("date") == today:
                return float(data.get("pnl_pct", 0))
    except Exception:
        pass
    return 0.0


def _state_path(filename: str) -> str:
    """Return path to a state file in storage."""
    try:
        from storage_config import DATA_DIR
        return os.path.join(DATA_DIR, filename)
    except ImportError:
        return os.path.join("/tmp", filename)
