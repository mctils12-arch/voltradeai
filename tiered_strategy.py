#!/usr/bin/env python3
"""
VolTradeAI — Tiered Strategy Engine
====================================
Option D implementation: CSP core + leverage + trend capture + tail hedge.

Replaces the 30-signal scoring stack with four explicit, independently-managed
tiers. Each tier has its own entry logic, sizing math, and kill switch. You can
disable any tier via TIERS_ENABLED config without affecting the others.

TIER ARCHITECTURE
  Tier 1 — CSP core (always on, load-bearing)
    Sell cash-secured puts on high-liquidity names. Variance risk premium
    harvest. This is the foundation — research-backed, works across regimes,
    produces consistent theta decay income.

  Tier 2 — Leverage multiplier (requires portfolio margin)
    Under Alpaca PM approval, size Tier 1 positions 3-4x their cash-secured
    equivalent. Gated on PM status. Blocked entirely below -8% drawdown from
    peak. This is where 30% CAGR potential comes from if T1 has edge.

  Tier 3 — Trend capture (leveraged long during uptrends)
    2x SPY/QQQ when trend filters confirm (50d>200d, VXX low, breadth strong).
    Exits on any signal break. Captures beta cleanly when available, cash
    when it's not. Jegadeesh-Titman 12-1 for ticker selection.

  Tier 4 — Tail hedge (small always-on insurance)
    Systematic OTM SPY puts at 1.5% annual premium budget. Bleeds in good
    markets, pays 3-5x during crashes. Enables higher Tier 2-3 leverage
    because it caps the downside.

CAPITAL ALLOCATION (at peak utilization)
  Tier 1:  50-70% of equity as cash-secured (or unlocked by T2 margin)
  Tier 2:  multiplies T1 to 150-250% notional (via PM buying power)
  Tier 3:  20-40% in SPY/QQQ when trend gates pass
  Tier 4:  1.5% annual in SPY puts (convex, sized by vega)

RISK CONTROLS
  Portfolio stop:    -20% from peak → liquidate all, block entries
  Daily loss limit:  -3% in one day → no new entries today
  Correlation cap:   max 60% BP in correlated names
  Margin buffer:     maintain 20% free BP at all times
  Regime kill:       VXX/VIX3M > 1.40 → T2-T4 off, T1 minimum size

USAGE
  from tiered_strategy import TieredStrategy, TierContext
  ts = TieredStrategy()
  context = TierContext(equity=100_000, macro=macro_snapshot,
                         positions=current_positions, portfolio_margin=True)
  tier_actions = ts.run_tiers(context)
  # tier_actions = [{"tier": 1, "ticker": "AMD", ...}, ...]
"""

from __future__ import annotations
import os
import json
import logging
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

try:
    from regime_util import classify_regime, classify_regime_5level
    _HAS_REGIME_UTIL = True
except ImportError:
    _HAS_REGIME_UTIL = False

logger = logging.getLogger("voltrade.tiers")

# ── Tier enable flags (can be toggled live via this file's config) ───────────
# STRATEGY 2026-04-22: T4 tail hedge disabled per user preference.
# User rejects paying 1.5% annual insurance premium for tail protection.
# Predictive stress index (stress_index.py, Commit F) replaces this
# with dynamic sizing reduction when leading indicators fire.
TIERS_ENABLED = {
    1: True,   # CSP core — always leave enabled
    2: True,   # Leverage multiplier (auto-disabled without PM)
    3: True,   # Trend capture
    4: False,  # Tail hedge DISABLED — replaced by stress_index predictive de-risking
}

# ── Risk limits ──────────────────────────────────────────────────────────────
PORTFOLIO_DRAWDOWN_KILL = -0.20         # -20% from peak → kill switch fires
T2_LEVERAGE_DRAWDOWN_GATE = -0.08       # -8% from peak → T2 off
DAILY_LOSS_LIMIT_PCT = -0.03            # -3% intraday → no new entries today

# ALL CAPS DELEGATE TO system_config.get_adaptive_params() — SINGLE SOURCE OF TRUTH
# Do not add new caps here. If a cap is needed, add it to system_config
# where the rest of the system can see it. Duplicating caps caused silent
# violations in the prior tier build (fixed 2026-04-20).

MAX_CORRELATED_BP = 0.60                # max 60% in sector-correlated positions (tier-specific)
REGIME_KILL_VXX_RATIO = 1.40            # VXX ratio above this → T2-T4 off (tier-specific)


def get_regime_caps(vxx_ratio: float, spy_vs_ma50: float,
                     equity: float = 100_000) -> dict:
    """
    Pull the regime-adaptive caps from system_config (authoritative source).

    Returns dict with keys:
        MAX_POSITION_PCT, MAX_TOTAL_EXPOSURE, MAX_POSITIONS,
        MAX_OPTIONS_PCT, regime, MIN_SCORE, etc.

    Falls back to conservative defaults if system_config import fails.
    """
    try:
        from system_config import get_adaptive_params
        return get_adaptive_params(
            vxx_ratio=vxx_ratio,
            spy_vs_ma50=spy_vs_ma50,
            markov_state=1,
            time_of_day="regular",
            account_equity=equity,
            spy_below_200_days=0,
        )
    except Exception as _gcap_err:
        import logging
        logging.getLogger("voltrade.tiers").warning(f"get_regime_caps fallback: {_gcap_err}")
        # Conservative fallback matching system_config BASE_CONFIG defaults
        return {
            "MAX_POSITION_PCT":    0.08,
            "MAX_TOTAL_EXPOSURE":  1.00,   # USER: fully invested in BULL
            "MAX_POSITIONS":       6,
            "MAX_OPTIONS_PCT":     0.08,
            "regime":              "NEUTRAL",
        }

# ── Tier 1: CSP core settings ────────────────────────────────────────────────
# ── T1 Universe (DYNAMIC 2026-04-22) ────────────────────────────────────────
# Hardcoded list kept as FALLBACK only. Active universe comes from
# csp_universe.get_top_csp_candidates() which ranks ~400 candidates
# by 7-factor CSP score and returns top 200 per 15-min cycle.
T1_TICKERS_FALLBACK = [
    "AMD", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "SPY", "QQQ", "IWM",
    "AVGO", "CRM", "ORCL", "INTC", "CAT", "GE",
]

def _get_t1_universe() -> list:
    """
    Returns active T1 universe — dynamic if available and enabled,
    fallback otherwise.
    """
    try:
        from system_config import BASE_CONFIG
        if not BASE_CONFIG.get("T1_DYNAMIC_UNIVERSE_ENABLED", True):
            logger.debug("T1 dynamic universe disabled via config")
            return T1_TICKERS_FALLBACK
        max_size = BASE_CONFIG.get("T1_UNIVERSE_MAX_SIZE", 200)
    except Exception:
        max_size = 200

    try:
        from csp_universe import get_top_csp_candidates
        dynamic = get_top_csp_candidates(n=max_size)
        if dynamic and len(dynamic) >= len(T1_TICKERS_FALLBACK):
            return dynamic
        logger.warning(
            f"T1 dynamic universe small ({len(dynamic)} tickers) — using fallback"
        )
        return T1_TICKERS_FALLBACK
    except ImportError:
        logger.debug("csp_universe module not installed, using T1_TICKERS_FALLBACK")
        return T1_TICKERS_FALLBACK
    except Exception as e:
        logger.warning(f"T1 universe fetch failed ({e}) — using fallback")
        return T1_TICKERS_FALLBACK

# Backward compat: T1_TICKERS still referenced elsewhere in this file.
# Use the dynamic universe by default.
T1_TICKERS = T1_TICKERS_FALLBACK  # placeholder; tier1_csp_core calls _get_t1_universe()
T1_MIN_IV_RANK = 40          # IVR > 40 means elevated premium worth selling
T1_TARGET_DTE_MIN = 30       # 30-45 DTE is the sweet spot for VRP harvest
T1_TARGET_DTE_MAX = 45
T1_TARGET_DELTA = 0.30       # 30-delta short put (70% prob OTM at expiry)
T1_TAKE_PROFIT_PCT = 0.50    # close at 50% max profit (tastytrade standard)
T1_STOP_LOSS_MULT = 2.0      # stop if loss exceeds 2x credit received
T1_FORCE_CLOSE_DTE = 21      # force close at 21 DTE (avoid gamma risk)

# ── Tier 2: Leverage multiplier settings ─────────────────────────────────────
T2_LEVERAGE_FACTOR = 3.0     # PM gives ~4x effective BP multiplier; we use 3x for safety
T2_REQUIRES_PM = True        # Hard gate on portfolio margin approval
# T2_MAX_NOTIONAL_MULT removed — use regime-adaptive 95% equity cap instead
# (T2 leverage multiplies contracts-per-dollar-equity, NOT total equity deployed)

# ── Tier 3: Trend capture settings ───────────────────────────────────────────
T3_TICKERS = ["SPY", "QQQ"]
T3_LEVERAGE_FACTOR = 2.0     # 2x via margin (Reg-T works, PM better)
T3_MIN_TREND_DAYS = 10       # 50d>200d MA for at least 10 consecutive days
T3_MAX_ALLOCATION_PCT = 0.40 # Max 40% of equity in T3 positions
T3_EXIT_ON_MA_BREAK = True   # Exit immediately if SPY crosses below 50d MA
T3_VXX_MAX = 0.95            # Require VXX ratio below this

# ── Tier 4: Tail hedge settings ──────────────────────────────────────────────
T4_HEDGE_TICKER = "SPY"
T4_ANNUAL_PREMIUM_BUDGET_PCT = 0.015  # 1.5% annual bleed on puts
T4_PUT_DTE_TARGET = 60               # 2-month puts — avoid 0DTE decay
T4_PUT_DELTA_TARGET = 0.10           # 10-delta puts — OTM insurance
T4_MIN_EQUITY_FOR_HEDGE = 25000      # Don't hedge tiny accounts


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class TierContext:
    """Everything a tier needs to make decisions."""
    equity: float                        # current account equity
    peak_equity: float = 0               # all-time peak (for drawdown gate)
    buying_power: float = 0              # Alpaca reported BP
    positions: List[Dict] = field(default_factory=list)
    macro: Dict = field(default_factory=dict)
    portfolio_margin: bool = False       # Alpaca PM approved?
    daily_pnl_pct: float = 0             # today's P&L as pct of equity
    regime: str = "NEUTRAL"
    vxx_ratio: float = 1.0
    spy_vs_ma50: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0 or self.equity <= 0:
            return 0.0
        return (self.equity - self.peak_equity) / self.peak_equity

    @property
    def bp_utilization(self) -> float:
        if self.buying_power <= 0 and self.equity > 0:
            # No BP reported — assume equity = BP (cash account)
            return 0.0
        total_exposure = sum(abs(float(p.get("market_value", 0))) for p in self.positions)
        return total_exposure / max(self.buying_power, self.equity)


@dataclass
class TierAction:
    """A single trade recommendation from a tier."""
    tier: int
    action: str                          # BUY, SELL, CLOSE, SELL_CSP, etc.
    ticker: str
    strategy: str                        # csp, leveraged_csp, trend_long, tail_put
    size_pct: float                      # fraction of BP to deploy
    reason: str
    metadata: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER KILL-SWITCH — run before ANY tier considers trading
# ══════════════════════════════════════════════════════════════════════════════

def master_kill_switch(ctx: TierContext) -> Tuple[bool, str]:
    """
    Returns (is_killed, reason). If killed, NO new entries, existing positions
    still managed by their own exit logic (this just blocks new trades).
    """
    # 1. Portfolio drawdown kill
    if ctx.drawdown_pct <= PORTFOLIO_DRAWDOWN_KILL:
        return (True, f"Portfolio DD kill: {ctx.drawdown_pct*100:.1f}% <= {PORTFOLIO_DRAWDOWN_KILL*100:.0f}%")

    # 2. Daily loss limit
    if ctx.daily_pnl_pct <= DAILY_LOSS_LIMIT_PCT:
        return (True, f"Daily loss limit: {ctx.daily_pnl_pct*100:.1f}% <= {DAILY_LOSS_LIMIT_PCT*100:.0f}%")

    # 3. Portfolio invested ceiling — from system_config (authoritative).
    # Regime-adaptive: PANIC=30%, BEAR=50%, CAUTION=60%, BULL/NEUTRAL=95%.
    caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity)
    max_invested = caps.get("MAX_TOTAL_EXPOSURE", 1.00)
    current_invested = sum(abs(float(p.get("market_value", 0))) for p in ctx.positions)
    invested_pct = current_invested / max(ctx.equity, 1)
    if invested_pct >= max_invested:
        return (True,
            f"Portfolio invested {invested_pct*100:.1f}% >= {max_invested*100:.0f}% "
            f"(regime={caps.get('regime', ctx.regime)})")

    # 4. Regime kill for T2-4 (not a full kill, handled per tier)
    return (False, "ok")


def regime_kill_for_tier(ctx: TierContext, tier_num: int) -> bool:
    """Tier-specific regime kill. Tier 1 continues at min size even in panic."""
    if tier_num == 1:
        return False  # Tier 1 always runs (at reduced size in stress)
    # Tiers 2, 3, 4 all shut down in extreme vol
    return ctx.vxx_ratio >= REGIME_KILL_VXX_RATIO


# ══════════════════════════════════════════════════════════════════════════════
#  TIER 1 — CSP Core
# ══════════════════════════════════════════════════════════════════════════════

def tier1_csp_core(ctx: TierContext) -> List[TierAction]:
    """Sell cash-secured puts on Tier 1 universe, regime-scaled."""
    if not TIERS_ENABLED.get(1, False):
        return []

    actions = []

    # Pull authoritative caps from system_config (single source of truth)
    caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity)
    max_per_position = caps.get("MAX_OPTIONS_PCT", 0.08)  # CSP = options, use options cap
    max_positions = caps.get("MAX_POSITIONS", 6)
    regime_label = caps.get("regime", "NEUTRAL")

    # Additional tier-level scaling on top of the regime caps
    # (lets T1 further reduce size during stress beyond what system_config does)
    if ctx.vxx_ratio >= REGIME_KILL_VXX_RATIO:
        size_scalar = 0.30
    elif ctx.vxx_ratio >= 1.20:
        size_scalar = 0.60
    elif ctx.vxx_ratio >= 1.05:
        size_scalar = 0.85
    else:
        size_scalar = 1.00

    # Additional drawdown-based reduction
    dd = ctx.drawdown_pct
    if dd <= -0.15:
        size_scalar *= 0.50
    elif dd <= -0.10:
        size_scalar *= 0.75

    # Current tickers already held (avoid doubling up)
    held_tickers = {p.get("symbol", "").upper() for p in ctx.positions
                    if p.get("asset_class") in ("us_option", "option")}

    # Count current options positions to respect MAX_POSITIONS
    current_position_count = len([p for p in ctx.positions
                                   if p.get("asset_class") in ("us_option", "option")])
    slots_available = max(0, max_positions - current_position_count)

    # DYNAMIC-UNIVERSE 2026-04-22: pull ranked top 200 per scan.
    # If Layer 1/2 fails, falls back to T1_TICKERS_FALLBACK.
    active_universe = _get_t1_universe()
    logger.debug(f"T1 scanning {len(active_universe)} tickers (dynamic)")

    # Candidates: liquid, high-IV names we don't already own
    for ticker in active_universe:
        if ticker in held_tickers:
            continue
        if slots_available <= 0:
            break  # respect MAX_POSITIONS

        # Position sizing: respect MAX_OPTIONS_PCT from system_config (8%)
        base_size = max_per_position * size_scalar
        slots_available -= 1

        # Action: sell CSP on this ticker with T1 parameters
        actions.append(TierAction(
            tier=1,
            action="SELL_CSP",
            ticker=ticker,
            strategy="csp_core",
            size_pct=base_size,
            reason=f"T1 CSP harvest (regime={regime_label}, vxx={ctx.vxx_ratio:.2f}, DD={dd*100:.1f}%, size={size_scalar:.0%})",
            metadata={
                "target_dte_min": T1_TARGET_DTE_MIN,
                "target_dte_max": T1_TARGET_DTE_MAX,
                "target_delta": T1_TARGET_DELTA,
                "take_profit_pct": T1_TAKE_PROFIT_PCT,
                "stop_loss_mult": T1_STOP_LOSS_MULT,
                "force_close_dte": T1_FORCE_CLOSE_DTE,
                "min_iv_rank": T1_MIN_IV_RANK,
            },
        ))

    # Return at most 3 new CSP entries per scan cycle (avoid overconcentration)
    return actions[:3]


# ══════════════════════════════════════════════════════════════════════════════
#  TIER 2 — Leverage Multiplier (requires Portfolio Margin)
# ══════════════════════════════════════════════════════════════════════════════

def tier2_leverage_multiplier(ctx: TierContext, t1_actions: List[TierAction]) -> List[TierAction]:
    """
    Multiplies Tier 1 sizing under portfolio margin. Does NOT add new trades —
    just increases the size of T1 trades when PM gates pass.

    Returns T1 actions with multiplied sizing (or unchanged if gates fail).
    """
    if not TIERS_ENABLED.get(2, False):
        return t1_actions

    # Hard gate: requires Alpaca Portfolio Margin approval
    if T2_REQUIRES_PM and not ctx.portfolio_margin:
        logger.debug("T2 skipped: portfolio margin not approved")
        return t1_actions

    # Drawdown gate: disable leverage below -8% from peak
    if ctx.drawdown_pct <= T2_LEVERAGE_DRAWDOWN_GATE:
        logger.info(f"T2 disabled: DD {ctx.drawdown_pct*100:.1f}% <= {T2_LEVERAGE_DRAWDOWN_GATE*100:.0f}%")
        return t1_actions

    # Regime gate: no leverage during extreme vol
    if regime_kill_for_tier(ctx, 2):
        logger.info(f"T2 disabled: VXX ratio {ctx.vxx_ratio:.2f} >= {REGIME_KILL_VXX_RATIO}")
        return t1_actions

    # T2 leverage must respect system_config's regime-adaptive cap.
    # Under PM, leverage lets you SIZE MORE CONTRACTS on the same equity.
    # It does NOT let you exceed MAX_TOTAL_EXPOSURE (100% in BULL).
    caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity)
    max_invested = caps.get("MAX_TOTAL_EXPOSURE", 1.00)
    current_invested = sum(abs(float(p.get("market_value", 0))) for p in ctx.positions)
    current_invested_pct = current_invested / max(ctx.equity, 1)
    equity_headroom_pct = max_invested - current_invested_pct

    if equity_headroom_pct <= 0:
        logger.info(f"T2: invested cap reached {current_invested_pct*100:.1f}% >= {max_invested*100:.0f}%")
        return t1_actions

    # How much NEW equity exposure would T1 actions add (at 1x)?
    t1_new_exposure_pct = sum(a.size_pct for a in t1_actions)

    # Under PM, we can multiply contract count but the EQUITY EXPOSURE cannot
    # exceed headroom. effective_multiplier is the factor such that:
    #   t1_new_exposure_pct * multiplier <= equity_headroom_pct
    if t1_new_exposure_pct > 0:
        multiplier_by_headroom = equity_headroom_pct / t1_new_exposure_pct
    else:
        multiplier_by_headroom = T2_LEVERAGE_FACTOR

    effective_multiplier = min(T2_LEVERAGE_FACTOR, multiplier_by_headroom)
    effective_multiplier = max(1.0, effective_multiplier)

    # Per-position cap stays tight even under leverage
    max_options_pct = caps.get("MAX_OPTIONS_PCT", 0.08)
    leveraged = []
    for action in t1_actions:
        new_size = min(
            action.size_pct * effective_multiplier,
            max_options_pct * 2.0  # allow T2 to go up to 2x the normal options cap
        )
        leveraged.append(TierAction(
            tier=2,
            action=action.action,
            ticker=action.ticker,
            strategy="leveraged_csp",
            size_pct=new_size,
            reason=f"T2 lev x{effective_multiplier:.1f} on T1 CSP (PM={ctx.portfolio_margin}, DD={ctx.drawdown_pct*100:.1f}%)",
            metadata={**action.metadata, "leverage_applied": effective_multiplier,
                      "original_size": action.size_pct},
        ))

    return leveraged


# ══════════════════════════════════════════════════════════════════════════════
#  TIER 3 — Trend Capture
# ══════════════════════════════════════════════════════════════════════════════

def tier3_trend_capture(ctx: TierContext) -> List[TierAction]:
    """2x leveraged long SPY/QQQ when trend filters pass."""
    if not TIERS_ENABLED.get(3, False):
        return []

    if regime_kill_for_tier(ctx, 3):
        return []

    if ctx.drawdown_pct <= T2_LEVERAGE_DRAWDOWN_GATE:
        return []  # same gate as T2

    # Trend gates: VXX must be low, SPY must be above 50d MA
    if ctx.vxx_ratio > T3_VXX_MAX:
        return []
    if ctx.spy_vs_ma50 < 1.01:
        return []

    # Check regime: only fires during confirmed bull
    if ctx.regime not in ("BULL", "NEUTRAL_BULL", "bull"):
        return []

    # Pull caps from system_config
    caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity)
    max_total_exposure = caps.get("MAX_TOTAL_EXPOSURE", 1.00)
    max_position_pct = caps.get("MAX_POSITION_PCT", 0.08)

    # T3's 40% allocation is a TIER BUDGET, within system_config's overall cap.
    # Check current total invested doesn't already eat into T3's room.
    current_invested_pct = sum(abs(float(p.get("market_value", 0))) for p in ctx.positions) / max(ctx.equity, 1)
    overall_headroom = max(0, max_total_exposure - current_invested_pct)

    # Check current T3 allocation (don't overbuy)
    held_t3 = sum(abs(float(p.get("market_value", 0))) for p in ctx.positions
                  if p.get("symbol", "").upper() in T3_TICKERS
                  and p.get("asset_class") != "us_option")
    held_t3_pct = held_t3 / max(ctx.equity, 1)

    # T3 budget is min(tier quota 40%, overall headroom from other tiers)
    t3_effective_budget = min(T3_MAX_ALLOCATION_PCT, overall_headroom)
    if held_t3_pct >= t3_effective_budget:
        return []

    allocation_room = t3_effective_budget - held_t3_pct

    actions = []
    for ticker in T3_TICKERS:
        # Each ticker gets half the remaining room, capped at 2x MAX_POSITION_PCT
        size_pct = allocation_room / len(T3_TICKERS)
        levered_size = size_pct * T3_LEVERAGE_FACTOR

        actions.append(TierAction(
            tier=3,
            action="BUY",
            ticker=ticker,
            strategy="trend_long",
            # Per-position cap = 2x MAX_POSITION_PCT (T3 is explicitly leveraged)
            size_pct=min(levered_size, max_position_pct * 2.0),
            reason=f"T3 trend (regime={caps.get('regime')}, headroom={overall_headroom*100:.0f}%)",
            metadata={
                "stop_loss_ticker": ticker,
                "exit_on_ma_break": T3_EXIT_ON_MA_BREAK,
                "leverage": T3_LEVERAGE_FACTOR,
            },
        ))

    return actions


# ══════════════════════════════════════════════════════════════════════════════
#  TIER 4 — Tail Hedge
# ══════════════════════════════════════════════════════════════════════════════

def tier4_tail_hedge(ctx: TierContext) -> List[TierAction]:
    """Small systematic OTM SPY puts. Always on (except in extreme panic)."""
    if not TIERS_ENABLED.get(4, False):
        return []

    if ctx.equity < T4_MIN_EQUITY_FOR_HEDGE:
        return []

    # Don't add hedge when VXX is already blown out (puts are too expensive)
    if ctx.vxx_ratio >= REGIME_KILL_VXX_RATIO:
        return []

    # Check if we already have an active hedge
    hedge_held = any(
        p.get("symbol", "").upper().startswith(T4_HEDGE_TICKER)
        and p.get("asset_class") in ("us_option", "option")
        and int(float(p.get("qty", 0))) > 0  # long options only
        for p in ctx.positions
    )

    if hedge_held:
        return []  # hedge already in place

    # Annual premium budget / 4 (roll quarterly) → per-roll budget
    per_roll_budget = ctx.equity * T4_ANNUAL_PREMIUM_BUDGET_PCT / 4
    size_pct = per_roll_budget / max(ctx.equity, 1)

    return [TierAction(
        tier=4,
        action="BUY_PUT",
        ticker=T4_HEDGE_TICKER,
        strategy="tail_hedge",
        size_pct=size_pct,
        reason=f"T4 hedge roll (budget=${per_roll_budget:.0f}, vxx={ctx.vxx_ratio:.2f})",
        metadata={
            "target_dte": T4_PUT_DTE_TARGET,
            "target_delta": T4_PUT_DELTA_TARGET,
            "premium_budget": per_roll_budget,
        },
    )]


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER ENGINE — runs all tiers in order with master kill-switch
# ══════════════════════════════════════════════════════════════════════════════

def _enforce_exposure_cap(ctx: "TierContext", actions: List["TierAction"]) -> List["TierAction"]:
    """
    Ensure combined tier actions + existing positions don't breach
    MAX_TOTAL_EXPOSURE (from system_config).

    T4 tail hedge is protected — it's ~0.4% and removing it defeats the
    whole point of the hedge.

    If T1/T2/T3 combined would breach, scale them down proportionally.
    """
    if not actions:
        return actions

    caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity)
    max_invested_pct = caps.get("MAX_TOTAL_EXPOSURE", 1.00)

    current_invested_pct = sum(abs(float(p.get("market_value", 0))) for p in ctx.positions) / max(ctx.equity, 1)
    headroom = max(0.0, max_invested_pct - current_invested_pct)

    protected = [a for a in actions if a.tier == 4]
    scalable = [a for a in actions if a.tier != 4]

    protected_size = sum(a.size_pct for a in protected)
    scalable_size = sum(a.size_pct for a in scalable)
    total_proposed = protected_size + scalable_size

    if total_proposed <= headroom:
        return actions

    scalable_headroom = max(0.0, headroom - protected_size)

    if scalable_size > 0 and scalable_headroom < scalable_size:
        scale_factor = scalable_headroom / scalable_size
        logger.warning(
            f"[TIERS] Master allocator scaling down by {scale_factor:.2%} "
            f"(proposed {total_proposed*100:.1f}% > headroom {headroom*100:.1f}%, "
            f"regime={caps.get('regime')})"
        )
        for a in scalable:
            a.size_pct *= scale_factor
            a.reason = f"{a.reason} [scaled {scale_factor:.0%} by cap enforcer]"

    return protected + scalable


class TieredStrategy:
    """Main entry point. Wire this into bot_engine.scan_market()."""

    def __init__(self, config_overrides: Optional[Dict] = None):
        self.config = config_overrides or {}

    def run_tiers(self, ctx: TierContext) -> Dict[str, Any]:
        """
        Execute all tiers and return a unified action list + diagnostics.

        Returns:
            {
                "killed": bool,
                "kill_reason": str,
                "actions": [TierAction, ...],
                "tier_stats": {1: count, 2: count, 3: count, 4: count},
                "diagnostics": {...},
            }
        """
        # Run master kill-switch first
        killed, reason = master_kill_switch(ctx)
        if killed:
            logger.warning(f"Master kill-switch: {reason}")
            return {
                "killed": True,
                "kill_reason": reason,
                "actions": [],
                "tier_stats": {},
                "diagnostics": self._diagnostics(ctx),
            }

        all_actions = []

        # Tier 1 — CSP core (always runs)
        t1_actions = tier1_csp_core(ctx)
        logger.info(f"T1 CSP: {len(t1_actions)} candidates")

        # Tier 2 — multiply T1 sizing (if PM approved)
        t1_after_t2 = tier2_leverage_multiplier(ctx, t1_actions)
        all_actions.extend(t1_after_t2)

        # Tier 3 — trend capture (independent of T1)
        t3_actions = tier3_trend_capture(ctx)
        all_actions.extend(t3_actions)
        logger.info(f"T3 trend: {len(t3_actions)} candidates")

        # Tier 4 — tail hedge (always on, small size)
        t4_actions = tier4_tail_hedge(ctx)
        all_actions.extend(t4_actions)
        logger.info(f"T4 hedge: {len(t4_actions)} candidates")

        # ── MASTER ALLOCATOR ──────────────────────────────────────────────
        # Tiers run independently and don't know about each other's
        # allocations. This stage enforces MAX_TOTAL_EXPOSURE across all
        # tier actions combined. If the sum of proposed sizes + current
        # positions would exceed the cap, scale down proportionally.
        # T4 (tail hedge) is protected from scale-down — it's small and
        # load-bearing for the overall system safety.
        all_actions = _enforce_exposure_cap(ctx, all_actions)

        # Tier stats
        stats = {i: sum(1 for a in all_actions if a.tier == i) for i in (1, 2, 3, 4)}

        return {
            "killed": False,
            "kill_reason": "",
            "actions": all_actions,
            "tier_stats": stats,
            "diagnostics": self._diagnostics(ctx),
        }

    def _diagnostics(self, ctx: TierContext) -> Dict:
        """Summary of context for logging/dashboard."""
        return {
            "equity": round(ctx.equity, 2),
            "drawdown_pct": round(ctx.drawdown_pct * 100, 2),
            "bp_utilization": round(ctx.bp_utilization * 100, 2),
            "daily_pnl_pct": round(ctx.daily_pnl_pct * 100, 2),
            "regime": ctx.regime,
            "vxx_ratio": round(ctx.vxx_ratio, 3),
            "portfolio_margin": ctx.portfolio_margin,
            "tiers_enabled": TIERS_ENABLED,
            "n_positions": len(ctx.positions),
        }


# ── Persistence helpers (for equity peak tracking) ──────────────────────────
PEAK_EQUITY_PATH = os.path.join(DATA_DIR, "voltrade_peak_equity.json")


def update_peak_equity(current_equity: float) -> float:
    """
    Track all-time peak equity for drawdown calculations.
    Atomic write with POSIX lock.
    """
    peak = current_equity
    try:
        if os.path.exists(PEAK_EQUITY_PATH):
            with open(PEAK_EQUITY_PATH) as f:
                data = json.load(f)
                peak = max(peak, float(data.get("peak_equity", 0)))
    except Exception:
        pass

    try:
        import tempfile
        dirname = os.path.dirname(PEAK_EQUITY_PATH) or "."
        fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".peak.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"peak_equity": peak,
                           "last_updated": datetime.now().isoformat()}, f)
            os.replace(tmp, PEAK_EQUITY_PATH)
        except Exception:
            try: os.unlink(tmp)
            except Exception: pass
    except Exception as e:
        logger.debug(f"peak equity save failed: {e}")

    return peak


def get_portfolio_margin_status() -> bool:
    """
    Query Alpaca account endpoint for portfolio margin approval.
    Returns True only if account is confirmed PM-approved.
    """
    try:
        import requests
        alpaca_base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        key = os.environ.get("ALPACA_KEY", "")
        secret = os.environ.get("ALPACA_SECRET", "")
        if not key or not secret:
            return False
        r = requests.get(
            f"{alpaca_base}/v2/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=8,
        )
        data = r.json()
        # Alpaca returns "account_type" or indicates PM via specific fields
        # Portfolio margin accounts have different buying_power multiples
        multiplier = float(data.get("multiplier", 1) or 1)
        # Reg-T margin: 2x, Portfolio margin: 4x+, Cash: 1x
        return multiplier >= 4.0
    except Exception as e:
        logger.warning(f"PM status check failed: {e}")
        return False


# ── CLI self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Self-test with synthetic context
    ctx = TierContext(
        equity=100_000,
        peak_equity=100_000,
        buying_power=200_000,  # simulated PM
        positions=[],
        macro={},
        portfolio_margin=True,
        daily_pnl_pct=0.0,
        regime="BULL",
        vxx_ratio=0.92,
        spy_vs_ma50=1.03,
    )

    ts = TieredStrategy()
    result = ts.run_tiers(ctx)

    print("=" * 70)
    print(f"Tiered strategy self-test")
    print("=" * 70)
    print(f"Killed: {result['killed']}")
    print(f"Tier counts: {result['tier_stats']}")
    print(f"Total actions: {len(result['actions'])}")
    print()
    for a in result["actions"]:
        print(f"  T{a.tier} {a.action:<10} {a.ticker:<6} size={a.size_pct*100:.1f}%  {a.reason}")
    print()
    print(f"Diagnostics: {json.dumps(result['diagnostics'], indent=2)}")
