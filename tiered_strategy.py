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
from typing import Optional, List, Dict, Any, Tuple, Callable

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
                     equity: float = 100_000, positions: Optional[List[Dict]] = None) -> dict:
    """
    Pull the regime-adaptive caps from system_config (authoritative source).

    Returns dict with keys:
        MAX_POSITION_PCT, MAX_TOTAL_EXPOSURE, MAX_POSITIONS,
        MAX_OPTIONS_PCT, regime, MIN_SCORE, etc.

    positions (2026-11-06, regime risk budget): when given — the tier engine's
    exposure-cap call sites pass ctx.positions — MAX_TOTAL_EXPOSURE becomes
    regime_engine.budget_exposure_cap's value: the room ABOVE the passive floor
    times the risk budget (never above the legacy cap, never below the floor).
    The legacy value rides along as MAX_TOTAL_EXPOSURE_LEGACY and the full
    result as RISK_BUDGET_CAP. None (every other caller): unchanged.

    Falls back to conservative defaults if system_config import fails.
    """
    try:
        from system_config import get_adaptive_params
        caps = get_adaptive_params(
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
        caps = {
            "MAX_POSITION_PCT":    0.08,
            "MAX_TOTAL_EXPOSURE":  1.00,   # USER: fully invested in BULL
            "MAX_POSITIONS":       6,
            "MAX_OPTIONS_POSITIONS": 6,
            "MAX_OPTIONS_PCT":     0.08,
            "regime":              "NEUTRAL",
        }
    if positions is not None:
        caps = _apply_floor_exemption(caps, positions, equity)
        caps = _apply_risk_budget(caps, positions, equity)
    return caps


def exposure_cap_floor_mode(env=None) -> str:
    """'exempt' or 'legacy': system_config EXPOSURE_CAP_FLOOR_MODE, overridden by env
    VOLTRADE_EXPOSURE_CAP_FLOOR (the no-deploy rollback). Anything unreadable is 'legacy'."""
    e = os.environ if env is None else env
    v = str(e.get("VOLTRADE_EXPOSURE_CAP_FLOOR", "") or "").strip().lower()
    if v in ("legacy", "exempt"):
        return v
    try:
        from system_config import BASE_CONFIG
        v = str(BASE_CONFIG.get("EXPOSURE_CAP_FLOOR_MODE", "legacy")).strip().lower()
    except Exception as e:
        logger.warning(f"exposure cap floor mode unreadable, using legacy: {e}")
        return "legacy"
    return v if v in ("legacy", "exempt") else "legacy"


def _apply_floor_exemption(caps: dict, positions: List[Dict], equity: float, env=None) -> dict:
    """EXPOSURE CAP vs THE FLOOR (2026-11-17, EXPLORATORY; see system_config
    EXPOSURE_CAP_FLOOR_MODE). In a regime whose cap is below the deployment ceiling
    (PANIC / BEAR / CAUTION) the regime's haircut applies to the room ABOVE the passive
    floor, not to the floor, which has its own volatility rule:

        cap = regime_cap + floor_share x (1 - regime_cap / ceiling)

    so satellite room = (ceiling - floor_share) x regime_cap / ceiling. Bounded:
    regime_cap <= cap <= ceiling (system_config MAX_TOTAL_EXPOSURE, 0.95): it never adds
    margin, and a regime at the ceiling (BULL / NEUTRAL) is unchanged. The regime's own
    value rides along as MAX_TOTAL_EXPOSURE_REGIME and the detail as FLOOR_EXEMPT_CAP.
    Never raises: on any failure the caps come back unchanged (the legacy cap)."""
    try:
        if exposure_cap_floor_mode(env) != "exempt":
            return caps
        from system_config import BASE_CONFIG
        ceiling = float(BASE_CONFIG.get("MAX_TOTAL_EXPOSURE", 0.95))
        regime_cap = float(caps.get("MAX_TOTAL_EXPOSURE", ceiling))
        eq = float(equity or 0.0)
        if eq <= 0 or ceiling <= 0 or regime_cap >= ceiling or regime_cap < 0:
            return caps
        import regime_engine as _re_fx
        floor = _re_fx.passive_floor_tickers()
        passive = sum(abs(float(p.get("market_value") or 0.0)) for p in (positions or [])
                      if str(p.get("symbol", "")).upper() in floor) / eq
        if not (passive > 0):
            return caps
        share = min(passive, ceiling)
        cap = min(ceiling, regime_cap + share * (1.0 - regime_cap / ceiling))
        out = dict(caps)
        out["MAX_TOTAL_EXPOSURE_REGIME"] = regime_cap
        out["MAX_TOTAL_EXPOSURE"] = round(cap, 4)
        out["FLOOR_EXEMPT_CAP"] = {
            "regime_cap": regime_cap, "ceiling": ceiling, "floor_pct": round(passive, 4),
            "cap": round(cap, 4), "satellite_room": round(max(0.0, cap - passive), 4),
        }
        return out
    except Exception as e:
        logger.debug(f"floor exemption unavailable: {e}")
        return caps


def _floor_note(caps: dict) -> str:
    """' [regime cap 50% applied above the floor's 67.4%: 81.9%]' when the floor exemption
    set the cap, else ''. Appended to the cap's log/kill text next to the budget note."""
    fx = caps.get("FLOOR_EXEMPT_CAP") or {}
    if not fx:
        return ""
    return (f" [regime cap {float(fx.get('regime_cap') or 0.0) * 100:.0f}% applied above the floor's "
            f"{float(fx.get('floor_pct') or 0.0) * 100:.1f}%: {float(fx.get('cap') or 0.0) * 100:.1f}%]")


def _apply_risk_budget(caps: dict, positions: List[Dict], equity: float) -> dict:
    """REGIME RISK BUDGET 2026-11-06: MAX_TOTAL_EXPOSURE with the budget applied to the
    satellite headroom (regime_engine.budget_exposure_cap; flag VOLTRADE_REGIME_BUDGET,
    default on; logged there as a counterfactual sizing row). Never raises: on any failure
    the caps come back unchanged."""
    try:
        import regime_engine as _re_rb
        rb = _re_rb.budget_exposure_cap(caps.get("MAX_TOTAL_EXPOSURE", 1.00), positions, equity)
    except Exception as e:
        logger.debug(f"risk budget cap unavailable: {e}")
        return caps
    caps = dict(caps)
    caps["RISK_BUDGET_CAP"] = rb
    if rb.get("applied"):
        caps["MAX_TOTAL_EXPOSURE_LEGACY"] = caps.get("MAX_TOTAL_EXPOSURE", 1.00)
        caps["MAX_TOTAL_EXPOSURE"] = rb["cap"]
    return caps


def _budget_note(caps: dict) -> str:
    """' [risk budget 0.62 cut the cap from 95% to 90.5%]' when the budget acted on the cap,
    else ''. Appended to the cap's log/kill text so the shadow records and the TIER-KILL
    audit can tell a budget-made kill from a legacy one."""
    rb = caps.get("RISK_BUDGET_CAP") or {}
    if not rb.get("applied") or rb.get("cap") == rb.get("cap_legacy"):
        return ""
    return (f" [risk budget {float(rb.get('budget') or 0.0):.2f} cut the cap from "
            f"{float(rb.get('cap_legacy') or 0.0) * 100:.0f}% to {float(rb.get('cap') or 0.0) * 100:.1f}%]")

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
# AUDIT 2026-05-18: I initially proposed switching T3 to SSO/QLD for "real
# leverage" but researched leveraged-ETF behavior shows volatility decay makes
# SSO/QLD WORSE for any holding period over a few days. Per research (multiple
# 2025-2026 sources): "Leveraged ETFs are designed for tactical, short-term
# trading. Their daily reset turns market volatility into a persistent drag."
# T3's typical hold is days-to-weeks with T3_EXIT_ON_MA_BREAK as the exit
# gate — that's enough exposure to volatility decay to hurt.
#
# The original SPY/QQQ design at 40% allocation gives 40% beta-1 exposure
# without decay. Math: 20% in SSO vs 40% in SPY produces ~equivalent EXPECTED
# return on quiet days, but in chop SSO underperforms by the decay amount.
# Keep T3 as direct SPY/QQQ. The "2x leverage" framing in the variable name
# is misleading (it's just a 40% beta-1 cash position) but the BEHAVIOR is
# correct given the holding pattern.
#
# Note: this DOES overlap with the QQQ floor (basket has 50% QQQ * 70% floor
# = 35% QQQ from floor + up to 40% from T3 = potentially 75% QQQ). The
# original code's _enforce_exposure_cap is supposed to handle that overlap;
# audit shows it only enforces within tier actions, not across subsystems.
# Real fix would be a unified allocator — flagged for follow-up. For now,
# T3 stays direct because the leveraged alternative is empirically worse.
T3_TICKERS = ["SPY", "QQQ"]   # direct, not leveraged (decay hurts)
T3_LEVERAGE_FACTOR = 2.0      # multiplier on size_pct — 20% allocation × 2 = 40% (cash beta-1 exposure)
T3_MIN_TREND_DAYS = 10        # 50d>200d MA for at least 10 consecutive days
T3_MAX_ALLOCATION_PCT = 0.40  # Max 40% of equity in T3 positions
T3_EXIT_ON_MA_BREAK = True    # Exit immediately if SPY crosses below 50d MA
T3_VXX_MAX = 0.95             # Require VXX ratio below this

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
    # 2026-11-06: the room above the passive floor is scaled by the regime risk budget.
    # 2026-11-17: in PANIC/BEAR/CAUTION the regime's haircut applies above the floor
    # only (_apply_floor_exemption; never above the 95% ceiling).
    caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity, positions=ctx.positions)
    max_invested = caps.get("MAX_TOTAL_EXPOSURE", 1.00)
    current_invested = sum(abs(float(p.get("market_value", 0))) for p in ctx.positions)
    invested_pct = current_invested / max(ctx.equity, 1)
    if invested_pct >= max_invested:
        return (True,
            f"Portfolio invested {invested_pct*100:.1f}% >= {max_invested*100:.0f}% "
            f"(regime={caps.get('regime', ctx.regime)}){_floor_note(caps)}{_budget_note(caps)}")

    # 4. Regime kill for T2-4 (not a full kill, handled per tier)
    return (False, "ok")


def regime_kill_for_tier(ctx: TierContext, tier_num: int) -> bool:
    """Tier-specific regime kill. Tier 1 continues at min size even in panic."""
    if tier_num == 1:
        return False  # Tier 1 always runs (at reduced size in stress)
    # Tiers 2, 3, 4 all shut down in extreme vol
    return ctx.vxx_ratio >= REGIME_KILL_VXX_RATIO


def log_masterkill_csp_shadow(ctx: TierContext, kill_reason: str) -> int:
    """
    RULE REVIEW evidence gate for open_questions.md item #20: master_kill_switch
    blocks ALL 4 tiers on a whole-portfolio exposure breach, including Tier 1
    CSP, which holds none of the stock/ETF exposure that typically trips it
    (see tier1_csp_core's own budget being capped independently by
    MAX_OPTIONS_PCT). Whether that's correct or needlessly starves CSP is a
    RULE REVIEW threshold/design call that CLAUDE.md requires counterfactual
    evidence for before changing anything — this function only BUILDS that
    evidence, never acts on it.

    Computes what tier1_csp_core() would have proposed this cycle (pure
    function of ctx — no orders, no side effects) and logs each candidate via
    shadow_portfolio's existing rejected_* bucket machinery under a new
    "rejected_masterkill" decision, reusing its outcome-backfill/get_shadow_stats
    pipeline rather than building a parallel one. A future session reads
    get_shadow_stats()["win_rate_by_decision"]["rejected_masterkill"] once
    enough history accumulates (same 90-day-ish pattern as the spread-filter
    and correlation-block buckets already tracked this way).

    Caveat carried into decision_reason: the win/loss label comes from
    shadow_portfolio's stock price-path proxy (its existing +2%/-4% PT/SL
    labeler), not real CSP premium economics — a directional proxy for "would
    this underlying have been calm", not exact options P&L. Best-effort and
    silent on failure: logging must never affect a live kill-switch decision.

    2026-11-12 (gate counterfactuals, see the block below log_csp_gate_shadow):
    each Tier-1 record now carries a `gate` block (tier 1, dropped, the dollars
    its size_pct stood for, the share no other gate would have stopped), and
    the Tier 2 / Tier 3 actions the engine would have proposed past this kill
    are computed (for logging only, never returned) and recorded the same way.
    What the kill blocks is unchanged.

    Returns the number of Tier-1 shadow candidates logged (0 on any failure or
    if Tier 1 had nothing to propose).
    """
    started = time.monotonic()
    _GATE_SHADOW_LAST.clear()
    plan: Dict[str, Any] = {}
    logged_tickers: List[str] = []
    try:
        shadow_actions = tier1_csp_core(ctx)
        plan = _masterkill_gate_plan(ctx, kill_reason, shadow_actions)
        if shadow_actions:
            # DECISION JOURNAL 2026-11-06: the blocked CSPs as skip decisions (logging only).
            try:
                import decision_journal
                decision_journal.record_tier_kill(shadow_actions, kill_reason, regime=ctx.regime, vxx_ratio=ctx.vxx_ratio)
            except Exception as _dj_err:
                logger.warning(f"decision journal (master kill) failed, shadow log unaffected: {_dj_err}")

            from csp_universe import _load_layer2_cache
            _layer2 = _load_layer2_cache() or {}
            price_lookup = {
                e["ticker"]: float(e["price"])
                for e in _layer2.get("scores", [])
                if e.get("ticker") and e.get("price", 0) > 0
            }

            import shadow_portfolio
            _t1_blocks = plan.get("tier1_blocks") or {}
            for action in shadow_actions:
                entry_price = price_lookup.get(action.ticker, 0.0)
                if entry_price <= 0:
                    continue  # backfill_outcomes can't label a record with no price
                shadow_portfolio.log_candidate(
                    ticker=action.ticker,
                    features={},
                    score=0.0,
                    decision="rejected_masterkill",
                    decision_reason=kill_reason,
                    entry_price=entry_price,
                    vxx_ratio=ctx.vxx_ratio,
                    regime_label=ctx.regime or "unknown",
                    gate=_t1_blocks.get(action.ticker),
                )
                logged_tickers.append(action.ticker)
    except Exception as e:
        logger.debug(f"log_masterkill_csp_shadow failed: {e}")
        logged_tickers = []
    _log_masterkill_upper_tiers(ctx, plan, logged_tickers, started)
    return len(logged_tickers)


def _layer2_price_lookup() -> dict:
    """ticker -> last layer-2 price: the source log_masterkill_csp_shadow reads."""
    from csp_universe import _load_layer2_cache
    _layer2 = _load_layer2_cache() or {}
    return {
        e["ticker"]: float(e["price"])
        for e in _layer2.get("scores", [])
        if e.get("ticker") and e.get("price", 0) > 0
    }


def log_csp_gate_shadow(rejections: list) -> dict:
    """
    [RULE-REVIEW] 2026-10-26: bot.ts's tier dispatcher drops a SELL_CSP action
    at two gates — OPTIONS-SLOT-FULL and the swept/resting-order CSP-SKIP (plus
    the remainder after a kill-switch break) — and, unlike a stock candidate, a
    CSP never had a shadow record to relabel: tier1_csp_core proposes it and
    deep_score never sees it. Same convention as log_masterkill_csp_shadow: one
    log_candidate per dropped action, price from the layer-2 cache (skipped
    when unknown — backfill_outcomes cannot label it), regime off the action's
    metadata. The stock price-path labeler is a directional proxy for "would
    this underlying have stayed calm", not CSP premium economics. Never raises.

    rejections: [{"ticker", "decision", "reason", "metadata": {...}}, ...]
    A rejection may carry "gate" (2026-11-12, the allocator's cash-reserve
    drops: tier, dropped, dollars cut); it is stored on the record as given.
    """
    out = {"received": 0, "logged": 0, "skipped_no_price": 0, "skipped": 0}
    try:
        if not isinstance(rejections, list) or not rejections:
            return out
        out["received"] = len(rejections)
        # DECISION JOURNAL 2026-11-06: the real refusals (rejected_*) as skip
        # decisions; shadow-only would_reject_* verdicts are not decisions.
        try:
            import decision_journal
            decision_journal.record_tier_rejections(rejections)
        except Exception as _dj_err:
            logger.warning(f"decision journal (CSP refusals) failed, shadow log unaffected: {_dj_err}")
        import shadow_portfolio
        prices = _layer2_price_lookup()
        for r in rejections:
            if not isinstance(r, dict) or not r.get("ticker") or not r.get("decision"):
                out["skipped"] += 1
                continue
            ticker = str(r["ticker"]).upper()
            price = float(prices.get(ticker, 0.0) or 0.0)
            if price <= 0:
                out["skipped_no_price"] += 1
                continue
            meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
            shadow_portfolio.log_candidate(
                ticker=ticker, features={}, score=0.0,
                decision=str(r["decision"]),
                decision_reason=str(r.get("reason") or "")[:200],
                entry_price=price,
                vxx_ratio=float(meta.get("vxx_ratio", 1.0) or 1.0),
                regime_label=str(meta.get("regime_label") or "unknown"),
                side="sell", action_label="SELL CSP", trade_type="options",
                regime_snapshot=meta.get("regime_snapshot"),   # regime engine ML snapshot when present (logging only)
                gate=r.get("gate"),   # gate counterfactuals 2026-11-12 (logging only; None from every other caller)
            )
            out["logged"] += 1
        return out
    except Exception as e:
        logger.debug(f"log_csp_gate_shadow failed: {e}")
        out["error"] = str(e)[:200]
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  GATE COUNTERFACTUALS ([RULE-REVIEW] 2026-11-12) — logging only
# ══════════════════════════════════════════════════════════════════════════════
# RULE REVIEW wants each gate's prevention P&L before its threshold may move.
# The tier engine's master gates had three blind spots:
#   (i)   a master kill logged only the Tier-1 CSPs; Tier 2 and Tier 3 were
#         never computed after it, so nothing recorded what they would have done;
#   (ii)  the allocator's proportional scale-down wrote a WARNING and no record
#         of what it cut;
#   (iii) no record carried a size, so nothing could be costed in dollars.
# The helpers below add those rows to the shadow log (shadow_portfolio) under
# the existing decision "rejected_masterkill" and its reason categories
# (exposure_ceiling / daily_loss / drawdown), or "rejected_cash_reserve" when
# the cash reserve was the constraint that bound. Each row carries a `gate`
# block: the tier, dropped or scaled, the dollars cut, and the share of that
# cut no OTHER gate would have stopped at that moment (variables interact: the
# exposure cap and the cash reserve are two limits on the same room above the
# floor, so dollars one of them cut are often dollars the other would have cut
# too). gate_counterfactuals_probe.py reads the rows back.
#
# Nothing here changes what a gate blocks, a size, or an order. Every helper
# reads its arguments, calls the tier functions for their return value only
# (their result is never handed to the caller), and writes shadow rows. None
# raises into the tier engine. None makes a network call the killed path did
# not already make: prices come from the layer-2 cache or a held position,
# the cash reserve from the files position_mgmt reads.
GATE_SHADOW_BUDGET_S = 15.0   # the Tier-2/3 rows are skipped when the kill path's shadow has already run this long
_GATE_SHADOW_LAST: Dict[str, Any] = {}


def get_gate_shadow_summary() -> Dict[str, Any]:
    """What the last master kill's shadow recorded: the gate, the Tier-1 rows
    logged, what Tier 2 / Tier 3 would have done, the extra rows written and
    the seconds it took. run_tiers puts it in the killed result's
    tier_timings, which reaches /api/diag/timings. Read-only."""
    return dict(_GATE_SHADOW_LAST)


def _invested_pct(ctx: TierContext) -> float:
    """Invested share of equity, by the expression every exposure check here uses."""
    return sum(abs(float(p.get("market_value", 0))) for p in ctx.positions) / max(ctx.equity, 1)


def _master_gates_holding(ctx: TierContext, caps: Optional[dict] = None) -> Dict[str, bool]:
    """Each master_kill_switch condition evaluated on its own, in the switch's
    order (the switch itself returns at the first that holds, so a
    daily-loss kill never says whether the exposure ceiling would have killed
    too). Keys are shadow_portfolio's reason categories. Logging only;
    test_gate_counterfactuals.py pins it to master_kill_switch."""
    if caps is None:
        caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity, positions=ctx.positions)
    return {
        "drawdown": ctx.drawdown_pct <= PORTFOLIO_DRAWDOWN_KILL,
        "daily_loss": ctx.daily_pnl_pct <= DAILY_LOSS_LIMIT_PCT,
        "exposure_ceiling": _invested_pct(ctx) >= caps.get("MAX_TOTAL_EXPOSURE", 1.00),
    }


def _reserve_room_pct(ctx: TierContext, regime) -> Optional[float]:
    """The cash reserve's room for new satellite capital as a fraction of
    equity: the number _enforce_exposure_cap takes from
    position_mgmt.live_headroom, read here through satellite_headroom with the
    same inputs so a killed scan adds no row to the regime counterfactual log
    (live_headroom writes one). None when the reserve is off or unreadable."""
    try:
        import position_mgmt as _pm
        snap = _pm.satellite_headroom(ctx.equity, ctx.positions, regime,
                                      pending_dollars=_pm.pending_scale_in_dollars(),
                                      floor_plan_ratio=_pm.floor_plan_target_ratio(),
                                      risk_budget=_pm.regime_budget())
        if snap.get("enabled") and ctx.equity > 0:
            return float(snap["headroom_dollars"]) / ctx.equity
    except (ImportError, KeyError, TypeError, ValueError) as e:
        logger.debug(f"gate shadow: cash reserve room unavailable: {e}")
    return None


def _net_fraction(proposed: float, kept: float, other_room: Optional[float]) -> float:
    """The share (0..1) of a gate's cut that no other gate would have stopped.
    proposed / kept: the actions' total size before and after the gate;
    other_room: the room the other constraints leave for those same actions
    (None: no other constraint). The allocator scales proportionally, so the
    share is the same for every action of the scan."""
    cut = proposed - kept
    if cut <= 1e-12:
        return 0.0
    passable = proposed if other_room is None else min(proposed, max(0.0, other_room))
    return max(0.0, min(1.0, (passable - kept) / cut))


def _gate_block(tier: int, action: str, order: str, regime, equity: float, size_pct: float, kept_pct: float,
                net_frac: float, cap_room: Optional[float], reserve_room: Optional[float],
                also=(), t2: Optional[str] = None, t3: Optional[str] = None) -> Dict[str, Any]:
    """One record's `gate` block (shadow_portfolio._sanitize_gate lists the fields).
    regime: the system_config regime whose cap and reserve applied (caps["regime"])."""
    cut = max(0.0, size_pct - kept_pct) * equity
    return {"tier": tier, "action": action, "order": order, "regime": regime, "equity": equity,
            "size_pct": size_pct, "kept_pct": kept_pct, "cut_dollars": cut,
            "net_frac": net_frac, "net_dollars": cut * net_frac,
            "cap_room": cap_room, "reserve_room": reserve_room, "also": list(also), "t2": t2, "t3": t3}


def _tier2_shadow(ctx: TierContext, t1_actions: List["TierAction"]) -> Tuple[str, List[Tuple["TierAction", float]]]:
    """(status, [(action, the size Tier 2 would have ADDED to it)]) from the
    real tier2_leverage_multiplier. status "levered" when it adds size; else
    the first of its own gates, in its order, that explains why it does not."""
    if not t1_actions:
        return "no_tier1_action", []
    out = tier2_leverage_multiplier(ctx, t1_actions)
    added = [(a, a.size_pct - float(a.metadata.get("original_size", a.size_pct))) for a in out if a.tier == 2]
    added = [(a, d) for a, d in added if d > 1e-9]
    if added:
        return "levered", added
    if not TIERS_ENABLED.get(2, False):
        return "disabled", []
    if T2_REQUIRES_PM and not ctx.portfolio_margin:
        return "no_portfolio_margin", []
    if ctx.drawdown_pct <= T2_LEVERAGE_DRAWDOWN_GATE:
        return "drawdown_gate", []
    if regime_kill_for_tier(ctx, 2):
        return "vol_kill", []
    if any(a.tier == 2 for a in out):
        return "leverage_x1", []      # headroom under the cap is below Tier 1's own proposal
    return "no_headroom", []          # its own check of the same exposure cap


def _tier3_shadow(ctx: TierContext) -> Tuple[str, List["TierAction"]]:
    """(status, the actions the real tier3_trend_capture would have proposed).
    status "proposed", else the first of its own gates, in its order, that
    explains the empty list."""
    acts = tier3_trend_capture(ctx)
    if acts:
        return "proposed", acts
    from system_config import BASE_CONFIG as _cfg
    if not TIERS_ENABLED.get(3, False):
        return "disabled", []
    if _cfg.get("FLOOR_MODE", "basket") == "vol_managed_spy":
        return "floor_owns_spy", []
    if regime_kill_for_tier(ctx, 3):
        return "vol_kill", []
    if ctx.drawdown_pct <= T2_LEVERAGE_DRAWDOWN_GATE:
        return "drawdown_gate", []
    if ctx.vxx_ratio > T3_VXX_MAX:
        return "vxx_above_max", []
    if ctx.spy_vs_ma50 < 1.01:
        return "spy_not_above_ma50", []
    if ctx.regime not in ("BULL", "NEUTRAL_BULL", "bull"):
        return "regime_not_bull", []
    return "no_headroom", []          # its own check of the same exposure cap, or its budget is already held


def _masterkill_gate_plan(ctx: TierContext, kill_reason: str, t1_actions: List["TierAction"]) -> Dict[str, Any]:
    """What the engine would have proposed past a master kill, for logging
    only: the Tier-1 records' `gate` blocks ("tier1_blocks", by ticker), the
    Tier 2 / Tier 3 rows ("upper", unpriced), and the statuses. The dollars a
    row cut are its size x equity; net_frac is the share that would have
    passed the OTHER gates (the other master conditions, the allocator's cap
    room and reserve room). Pure apart from the reads its callees make; never
    raises (an empty plan leaves the Tier-1 records as they were)."""
    plan: Dict[str, Any] = {"gate": None, "tier2": None, "tier3": None, "tier1_blocks": {}, "upper": []}
    try:
        import shadow_portfolio
        gate = shadow_portfolio._masterkill_reason_category(kill_reason)
        plan["gate"] = gate
        if ctx.equity <= 0:
            plan["skipped"] = "no equity reading"
            return plan
        caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity, positions=ctx.positions)
        holding = _master_gates_holding(ctx, caps)
        t2_status, t2_added = _tier2_shadow(ctx, t1_actions)
        t3_status, t3_actions = _tier3_shadow(ctx)
        plan.update(tier2=t2_status, tier3=t3_status)
        cap_room = max(0.0, caps.get("MAX_TOTAL_EXPOSURE", 1.00) - _invested_pct(ctx))
        reserve_room = _reserve_room_pct(ctx, caps.get("regime") or ctx.regime)
        proposed = (sum(a.size_pct for a in t1_actions) + sum(d for _, d in t2_added)
                    + sum(a.size_pct for a in t3_actions))
        others = [g for g, on in holding.items() if on and g != gate]
        if others:
            other_room: Optional[float] = 0.0          # another master condition would have killed the scan
        elif gate == "exposure_ceiling":
            other_room = reserve_room                  # the cap absent, the cash reserve is what is left
        else:
            other_room = cap_room if reserve_room is None else min(cap_room, reserve_room)
        net_frac = _net_fraction(proposed, 0.0, other_room)
        also = list(others)
        if gate != "exposure_ceiling" and "exposure_ceiling" not in also and cap_room < proposed:
            also.append("exposure_ceiling")
        if reserve_room is not None and reserve_room < proposed:
            also.append("cash_reserve")
        plan.update(net_frac=round(net_frac, 4), proposed_pct=round(proposed, 6), also=also)

        def block(tier: int, order: str, size: float) -> Dict[str, Any]:
            return _gate_block(tier, "dropped", order, caps.get("regime"), ctx.equity, size, 0.0, net_frac,
                               cap_room, reserve_room, also, t2_status, t3_status)

        plan["tier1_blocks"] = {a.ticker: block(1, a.action, a.size_pct) for a in t1_actions}
        base = {"decision": "rejected_masterkill", "reason": kill_reason, "vxx_ratio": ctx.vxx_ratio,
                "regime_label": ctx.regime or "unknown"}
        plan["upper"] = (
            [{**base, "ticker": a.ticker, "side": "sell", "action_label": "SELL CSP", "trade_type": "options",
              "gate": block(2, a.action, d)} for a, d in t2_added]
            + [{**base, "ticker": a.ticker, "side": "buy", "action_label": "BUY", "trade_type": "stock",
                "gate": block(3, a.action, a.size_pct)} for a in t3_actions])
    except Exception as e:  # noqa: BLE001 — logged; a failed plan must not cost the Tier-1 records
        logger.warning(f"[TIERS] gate shadow plan failed (kill unaffected): {type(e).__name__}: {e}")
        plan["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return plan


def _gate_row_prices(ctx: TierContext) -> Dict[str, float]:
    """ticker -> entry price for a gate row, from data already in hand: a held
    stock or ETF position's current price, then the layer-2 cache (the Tier-1
    records' source, which wins where both exist). No network."""
    prices: Dict[str, float] = {}
    for p in ctx.positions:
        if p.get("asset_class") in ("us_option", "option"):
            continue
        px = _as_float(p.get("current_price"))
        sym = str(p.get("symbol") or "").upper()
        if sym and px > 0:
            prices[sym] = px
    try:
        prices.update(_layer2_price_lookup())
    except Exception as e:  # noqa: BLE001 — logged; rows without a price are counted, not written
        logger.debug(f"gate shadow: layer-2 prices unavailable: {e}")
    return prices


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _log_masterkill_upper_tiers(ctx: TierContext, plan: Dict[str, Any], tier1_tickers: List[str],
                                started: float) -> None:
    """Write the plan's Tier 2 / Tier 3 rows (one save, first per name per ET
    day) and publish the summary. A Tier-2 row is the size Tier 2 would have
    added to a Tier-1 put, so it is written only for a name whose Tier-1
    record was (tier1_tickers). Skipped when the kill path's shadow has
    already run GATE_SHADOW_BUDGET_S: the rows are once-a-day, a later scan
    writes them. Never raises."""
    summary: Dict[str, Any] = {"gate": plan.get("gate"), "tier1_rows": len(tier1_tickers),
                               "tier2": plan.get("tier2"), "tier3": plan.get("tier3"),
                               "net_frac": plan.get("net_frac"), "also": plan.get("also"), "rows": None}
    for k in ("error", "skipped"):
        if plan.get(k):
            summary[k] = plan[k]
    try:
        upper = [r for r in (plan.get("upper") or []) if r["gate"]["tier"] != 2 or r["ticker"] in tier1_tickers]
        if upper and time.monotonic() - started > GATE_SHADOW_BUDGET_S:
            summary["rows"] = {"skipped": "time budget"}
        elif upper:
            import shadow_portfolio
            prices = _gate_row_prices(ctx)
            summary["rows"] = shadow_portfolio.log_gate_rows(
                [{**r, "entry_price": prices.get(str(r["ticker"]).upper(), 0.0)} for r in upper])
    except Exception as e:  # noqa: BLE001 — logged; logging must never reach the kill decision
        logger.warning(f"[TIERS] gate shadow rows failed (kill unaffected): {type(e).__name__}: {e}")
        summary["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    summary["sec"] = round(time.monotonic() - started, 2)
    _GATE_SHADOW_LAST.update(summary)


def _allocator_gate_blocks(ctx: TierContext, caps: dict, actions: List["TierAction"], proposed: List[float],
                           kept: List[float], action: str, reserve: Optional[dict], reserve_bound: bool,
                           protected_size: float) -> Tuple[str, List[Optional[Dict[str, Any]]]]:
    """(gate, one `gate` block per action; None where nothing was cut) for what
    _enforce_exposure_cap took from `actions`: `proposed` is each action's
    size before the allocator, `kept` after it. The gate is the constraint
    that bound: "cash_reserve" when the reserve's room was the smaller one,
    else "exposure_ceiling". net_frac is the share of the cut the OTHER of the
    two would have let through. Logging only; never raises (no blocks on a
    failure)."""
    gate = "cash_reserve" if reserve_bound else "exposure_ceiling"
    try:
        if ctx.equity <= 0:
            return gate, [None] * len(actions)
        cap_room = max(0.0, float(caps.get("MAX_TOTAL_EXPOSURE", 1.00)) - _invested_pct(ctx))
        reserve_room = (float(reserve["headroom_dollars"]) / ctx.equity
                        if isinstance(reserve, dict) and reserve.get("enabled") else None)
        other = cap_room if reserve_bound else reserve_room
        other_room = None if other is None else max(0.0, other - protected_size)
        total, total_kept = sum(proposed), sum(kept)
        net_frac = _net_fraction(total, total_kept, other_room)
        also = []
        if other_room is not None and other_room < total:
            also.append("exposure_ceiling" if reserve_bound else "cash_reserve")
        return gate, [
            _gate_block(a.tier, action, a.action, caps.get("regime"), ctx.equity, p, k, net_frac, cap_room,
                        reserve_room, also)
            if p - k > 1e-9 else None
            for a, p, k in zip(actions, proposed, kept)]
    except Exception as e:  # noqa: BLE001 — logged; the allocator's result does not depend on this
        logger.warning(f"[TIERS] allocator gate blocks failed (sizes unaffected): {type(e).__name__}: {e}")
        return gate, [None] * len(actions)


def _log_allocator_scale_shadow(ctx: TierContext, caps: dict, scalable: List["TierAction"], proposed: List[float],
                                reserve: Optional[dict], reserve_bound: bool, protected_size: float,
                                scale_factor: float, invested_pct: float, total_proposed: float,
                                cap_pct: float) -> None:
    """The shadow rows of _enforce_exposure_cap's proportional scale-down,
    called after the sizes are final (`proposed` is what each action asked for
    before it). A cut the regime cap made reads "Portfolio invested ..." so
    shadow_portfolio files it under exposure_ceiling, the master kill's own
    category for the same cap. Logging only; never raises."""
    try:
        gate, blocks = _allocator_gate_blocks(ctx, caps, scalable, proposed, [a.size_pct for a in scalable],
                                              "scaled", reserve, reserve_bound, protected_size)
        if reserve_bound:
            reason = f"cash reserve leaves room for {scale_factor:.0%} of the proposed size"
        else:
            reason = (f"Portfolio invested {invested_pct * 100:.1f}% + proposed {total_proposed * 100:.1f}% > "
                      f"{cap_pct * 100:.0f}% (regime={caps.get('regime')}): allocator scaled to "
                      f"{scale_factor:.0%}{_budget_note(caps)}")
        _log_allocator_gate_rows(ctx, caps, list(zip(scalable, blocks)), gate, reason)
    except Exception as e:  # noqa: BLE001 — logged; the allocator's result does not depend on this
        logger.warning(f"[TIERS] allocator scale shadow failed (sizes unaffected): {type(e).__name__}: {e}")


def _log_allocator_gate_rows(ctx: TierContext, caps: dict, pairs: List[Tuple["TierAction", Optional[Dict[str, Any]]]],
                             gate: str, reason: str) -> Dict[str, Any]:
    """Shadow rows for the allocator's cut: one per (action, block) pair that
    has a block, under "rejected_masterkill" (reason category
    exposure_ceiling) when the regime cap bound and "rejected_cash_reserve"
    when the cash reserve did. One save; first per name per ET day
    (shadow_portfolio.log_gate_rows). Logging only; never raises."""
    try:
        rows = []
        prices: Optional[Dict[str, float]] = None
        for a, block in pairs:
            if block is None:
                continue
            if prices is None:
                prices = _gate_row_prices(ctx)
            meta = a.metadata if isinstance(a.metadata, dict) else {}
            csp = a.action == "SELL_CSP"
            rows.append({
                "ticker": a.ticker,
                "decision": "rejected_cash_reserve" if gate == "cash_reserve" else "rejected_masterkill",
                "reason": reason,
                "entry_price": prices.get(str(a.ticker).upper(), 0.0),
                "vxx_ratio": meta.get("vxx_ratio", ctx.vxx_ratio),
                "regime_label": meta.get("regime_label") or caps.get("regime") or ctx.regime or "unknown",
                "side": "sell" if csp else "buy",
                "action_label": "SELL CSP" if csp else str(a.action),
                "trade_type": "options" if csp or a.action == "BUY_PUT" else "stock",
                "gate": block,
            })
        if not rows:
            return {"received": 0, "logged": 0}
        import shadow_portfolio
        return shadow_portfolio.log_gate_rows(rows)
    except Exception as e:  # noqa: BLE001 — logged; the allocator's result does not depend on this
        logger.warning(f"[TIERS] allocator gate rows failed (sizes unaffected): {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


# ══════════════════════════════════════════════════════════════════════════════
#  TIER 1 — CSP Core
# ══════════════════════════════════════════════════════════════════════════════

def _option_underlying(position: Dict[str, Any]) -> str:
    """Underlying ticker of an option position dict (upper-cased).

    Prefers Alpaca's ``underlying_symbol`` field when the position carries
    one; otherwise parses the OCC symbol with options_manager's end-anchored
    parser (root = everything before the fixed 15-char date+C/P+strike
    suffix). A symbol that is not parseable OCC is returned as-is, which
    preserves the pre-fix behaviour for any non-OCC input.
    """
    underlying = str(position.get("underlying_symbol") or "").strip().upper()
    if underlying:
        return underlying
    symbol = str(position.get("symbol") or "").strip().upper()
    if not symbol:
        return ""
    try:
        from options_manager import _parse_occ_symbol
        parsed = _parse_occ_symbol(symbol)
    except Exception:
        parsed = {}
    return str(parsed.get("ticker") or symbol).upper()


# ── REGIME ENGINE step 5 (docs/regime/FINDINGS.md): the CSP guards ──────────
_T1_EARNINGS_CACHE = {"at": 0.0, "days_ahead": 0, "cal": {}, "ok": False, "coverage": {}}
T1_EARNINGS_CACHE_S = 900.0   # csp_universe refreshes its layers every 15 min: same cadence, one Finnhub call
# KB #45 (c), 2026-11-04 (live: an IREN put sold the day before earnings). A name ABSENT from
# a calendar that failed, or that Finnhub cut (it drops a busy window's NEAREST dates), is
# confirmed with one per-symbol lookup before it may be sold. The lookups are bounded per scan
# (the scan runs inside the daemon's 300 s budget), and successful answers are cached like the
# window calendar.
T1_EARNINGS_CONFIRM_MAX = 8
_T1_SYMBOL_EARNINGS_CACHE: Dict[Tuple[str, int], Tuple[float, Optional[int]]] = {}
_T1_EARNINGS_DIAG: Dict[str, Any] = {}
_T1_EARNINGS_DIAG_KEYS = (
    "engine_active", "window_days", "lookup_ok", "truncated", "authoritative", "calendar_entries",
    "window_from", "window_to", "earliest_date_returned", "latest_date_returned", "at_row_cap",
    "near_edge_gap", "row_cap", "http_status", "calendar_fetched_at", "symbol_lookups",
    "symbol_cache_hits", "symbol_lookup_failures", "symbol_budget_skipped", "confirmed_in_window",
    "earnings_unverified",
)


class _T1EarningsLookup(tuple):
    """(calendar, lookup_ok): unpacks and compares like the plain pair every caller and test
    uses, plus .coverage, the window fetch's coverage record (options_scanner)."""

    def __new__(cls, cal, ok, coverage=None):
        obj = super().__new__(cls, (cal, ok))
        obj.coverage = dict(coverage or {})
        return obj


def _t1_earnings_calendar(days_ahead: int):
    """({ticker: days_to_earnings}, lookup_ok) from options_scanner's Finnhub calendar, cached
    15 minutes per window. It also carries .coverage, the record of the fetch that produced it
    (truncated, calendar_entries, window, earliest/latest date returned). On a failed lookup
    (lookup_ok=False) or a cut page (coverage['truncated']), an ABSENT name is not evidence of
    "no earnings": tier1_csp_core confirms it per symbol and fails closed when it cannot
    (KB #45 (c), 2026-11-04; this used to fail open)."""
    now = time.time()
    c = _T1_EARNINGS_CACHE
    if c["days_ahead"] == days_ahead and now - c["at"] < T1_EARNINGS_CACHE_S:
        return _T1EarningsLookup(c["cal"], c["ok"], c.get("coverage"))
    coverage = {}
    try:
        import options_scanner as _osc
        seq0 = _osc.earnings_calendar_seq()
        cal, ok = _osc._fetch_earnings_calendar_with_status(days_ahead=days_ahead)
        rec = _osc.earnings_calendar_coverage(days_ahead)
        if rec.get("seq", 0) > seq0:       # the record THIS fetch wrote (a substituted fetcher writes none)
            coverage = rec
    except Exception as e:
        logger.debug(f"T1 earnings calendar unavailable: {e}")
        cal, ok = {}, False
    c.update(at=now, days_ahead=days_ahead, cal=dict(cal or {}), ok=bool(ok), coverage=coverage)
    return _T1EarningsLookup(c["cal"], c["ok"], c["coverage"])


def _t1_confirm_state() -> Dict[str, Any]:
    """Per-scan counters (and the breaker) of the per-symbol confirmations."""
    return {"symbol_lookups": 0, "symbol_cache_hits": 0, "symbol_lookup_failures": 0,
            "symbol_budget_skipped": 0, "confirmed_in_window": 0, "earnings_unverified": 0,
            "broken": False}


def _t1_confirm_earnings(ticker: str, days_ahead: int, scan: Dict[str, Any]):
    """(days_to_earnings or None, source) for ONE candidate the window calendar cannot vouch for.
    source 'symbol': Finnhub's per-symbol calendar answered (None = no report inside the window).
    source 'unverified': no answer, because the lookup failed, this scan's T1_EARNINGS_CONFIRM_MAX
    lookups are spent, or an earlier lookup this scan failed. The caller then fails CLOSED.
    Answers are cached T1_EARNINGS_CACHE_S; failures are not, so the next scan asks again."""
    key = (str(ticker).upper(), int(days_ahead))
    now = time.time()
    hit = _T1_SYMBOL_EARNINGS_CACHE.get(key)
    if hit is not None and now - hit[0] < T1_EARNINGS_CACHE_S:
        scan["symbol_cache_hits"] += 1
        return hit[1], "symbol"
    if scan["broken"] or scan["symbol_lookups"] >= T1_EARNINGS_CONFIRM_MAX:
        scan["symbol_budget_skipped"] += 1
        return None, "unverified"
    scan["symbol_lookups"] += 1
    try:
        from options_scanner import _fetch_symbol_earnings_with_status
        days, ok = _fetch_symbol_earnings_with_status(ticker, days_ahead=days_ahead)
    except Exception as e:
        logger.debug(f"T1 per-symbol earnings lookup {ticker} unavailable: {e}")
        days, ok = None, False
    if not ok:
        scan["symbol_lookup_failures"] += 1
        scan["broken"] = True      # Finnhub is failing: the scan's other absent names are refused without a call
        return None, "unverified"
    if len(_T1_SYMBOL_EARNINGS_CACHE) > 512:
        for k, v in list(_T1_SYMBOL_EARNINGS_CACHE.items()):     # a snapshot: two scans can overlap in the daemon
            if now - v[0] >= T1_EARNINGS_CACHE_S:
                _T1_SYMBOL_EARNINGS_CACHE.pop(k, None)
    _T1_SYMBOL_EARNINGS_CACHE[key] = (now, days)
    return days, "symbol"


def _t1_publish_earnings_diag(rg: Dict[str, Any], scan: Dict[str, Any]) -> None:
    """Keep what this Tier-1 run's earnings guard saw, for get_t1_earnings_coverage()."""
    active = rg.get("engine") is not None
    cov = rg.get("earnings_coverage") or {}
    d = {k: None for k in _T1_EARNINGS_DIAG_KEYS}
    d.update({k: cov.get(k) for k in ("calendar_entries", "window_from", "window_to", "earliest_date_returned",
                                      "latest_date_returned", "at_row_cap", "near_edge_gap", "row_cap",
                                      "http_status")})
    d.update({k: v for k, v in scan.items() if k != "broken"})
    d.update(engine_active=active, window_days=rg.get("earnings_window"), calendar_fetched_at=cov.get("at"),
             lookup_ok=bool(rg.get("earnings_ok")) if active else None,
             truncated=bool(rg.get("earnings_truncated")) if active else None,
             authoritative=(bool(rg.get("earnings_ok")) and not rg.get("earnings_truncated")) if active else None,
             at=time.time())
    _T1_EARNINGS_DIAG.update(d)      # every key, every run: a full overwrite, never a half-empty read


def get_t1_earnings_coverage() -> Dict[str, Any]:
    """KB #45 (c) diagnostic: what the Tier-1 earnings guard saw on its last run. That covers
    the window calendar's rows, window, earliest/latest date returned, truncated, lookup_ok,
    and whether an absent name could be trusted (authoritative). It also counts the per-symbol
    confirmations: lookups, cache hits, failures, budget-skipped, confirmed reports inside the
    window, and refusals for an unverified date. `windows` holds every calendar window this
    process fetched (21 d scanner, 30 d csp_universe Layer 2, 47 d T1). run_tiers puts it in
    tier_timings, which reaches /api/diag/timings the same way csp_layer2_prefetch does.
    Read-only; never raises."""
    out: Dict[str, Any] = {k: None for k in _T1_EARNINGS_DIAG_KEYS}
    out.update(_T1_EARNINGS_DIAG)
    at = out.pop("at", None)
    out["age_sec"] = round(time.time() - at, 1) if at else None
    try:
        import options_scanner as _osc
        fields = ("calendar_entries", "window_from", "window_to", "earliest_date_returned",
                  "latest_date_returned", "truncated", "lookup_ok", "http_status")
        out["windows"] = {k: {f: v.get(f) for f in fields} for k, v in _osc.earnings_calendar_coverage().items()}
    except Exception as e:
        out["windows"] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    return out


def _t1_regime_context() -> dict:
    """The regime engine's state-level verdict on NEW cash-secured puts for this scan
    (step 5), or an inert context when the engine is off or has no state. Never raises."""
    out = {"engine": None, "state": None, "verdict": None, "gates": set(), "size_mult": 1.0,
           "target_delta": None, "earnings": {}, "earnings_ok": False, "earnings_truncated": False,
           "earnings_window": None, "earnings_coverage": {}}
    try:
        import regime_engine as _re
        if not _re.engine_allowed():
            return out
        state = _re.cached_state()
        if not isinstance(state, dict) or not state.get("as_of"):
            return out
        gates = _re.options_gates()
        verdict = _re.csp_verdict(state, {"dte_max": T1_TARGET_DTE_MAX})
        acting = _re.apply_csp_gates(verdict, gates)
        window = T1_TARGET_DTE_MAX + int(_re.OPTIONS_PARAMS["earnings_buffer_days"])
        lookup = _t1_earnings_calendar(window)
        cal, ok = lookup
        coverage = getattr(lookup, "coverage", None) or {}
        out.update(engine=_re, state=state, verdict=verdict, gates=gates, size_mult=float(acting["size_mult"]),
                   target_delta=acting["target_delta"], earnings=cal, earnings_ok=ok,
                   earnings_truncated=bool(coverage.get("truncated")), earnings_window=window,
                   earnings_coverage=coverage)
    except Exception as e:
        logger.debug(f"T1 regime context unavailable: {e}")
    return out


T1_DEALER_GAMMA_SOURCE = "tiered_strategy.tier1_csp_core"


def _t1_dealer_gamma_decision() -> Dict[str, Any]:
    """dealer_gamma's verdict on a new Tier-1 short put under this scan's state:
    allow | would_skip | skip. Never raises: a failure reads as allow, logged."""
    try:
        import dealer_gamma
        return dealer_gamma.decide("sell_cash_secured_put", "sell", dealer_gamma.get_state())
    except Exception as e:  # noqa: BLE001 — logged; the filter fails open
        logger.warning(f"T1 dealer gamma unavailable ({type(e).__name__}: {e}) - CSPs allowed")
        return {"action": "allow", "reason": f"dealer gamma unavailable ({type(e).__name__})"}


def _t1_dealer_gamma_log(decision: Dict[str, Any], ticker: str, price: Optional[float], vxx_ratio: float,
                         regime_label: str, rejections: List[Dict[str, Any]]) -> None:
    """Record one Tier-1 skip / would-skip (dealer_gamma.record: JSON log, audit feed,
    once per ET day per name) and queue its shadow record for log_csp_gate_shadow
    (bounded by dealer_gamma.shadow_slot). Never raises."""
    try:
        import dealer_gamma
        rec = dealer_gamma.record(decision, ticker, T1_DEALER_GAMMA_SOURCE, setup="csp_core",
                                  strategy="sell_cash_secured_put", price=price)
        if rec and dealer_gamma.shadow_slot():
            rejections.append({
                "ticker": ticker,
                "decision": dealer_gamma.SKIP_DECISION if decision.get("action") == "skip"
                else dealer_gamma.WOULD_SKIP_DECISION,
                "reason": str(decision.get("reason") or "")[:200],
                "metadata": {"vxx_ratio": vxx_ratio, "regime_label": regime_label},
            })
    except Exception as e:  # noqa: BLE001 — logged; logging must never break the tier engine
        logger.warning(f"T1 dealer gamma log failed for {ticker}: {type(e).__name__}: {e}")


def _t1_dealer_gamma_finish(returned: List["TierAction"], decision: Dict[str, Any], vxx_ratio: float,
                            regime_label: str, prices: Dict[str, float],
                            rejections: List[Dict[str, Any]]) -> None:
    """The verdict rides each returned SELL_CSP's metadata; a would-skip is recorded
    for each of them (the names the scan actually proposes)."""
    note = {k: decision.get(k) for k in ("action", "rank252", "asOf", "lowGamma", "active", "reason")}
    for a in returned:
        a.metadata["dealer_gamma"] = dict(note)
        if decision.get("action") == "would_skip":
            _t1_dealer_gamma_log(decision, a.ticker, prices.get(a.ticker), vxx_ratio, regime_label, rejections)


def tier1_csp_core(ctx: TierContext) -> List[TierAction]:
    """Sell cash-secured puts on Tier 1 universe, regime-scaled."""
    if not TIERS_ENABLED.get(1, False):
        return []

    actions = []

    # Pull authoritative caps from system_config (single source of truth)
    caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity)
    max_per_position = caps.get("MAX_OPTIONS_PCT", 0.08)  # CSP = options, use options cap
    # BUG FIX 2026-07-11: was caps.get("MAX_POSITIONS", 6) — that key is the
    # STOCK position cap and is zeroed in PANIC/BEAR/NEUTRAL to block new
    # stock longs, which silently zeroed CSP too in exactly the regimes
    # system_config.py's own comments say CSP should keep running in. Use
    # the dedicated options cap instead (see system_config.py BASE_CONFIG).
    max_positions = caps.get("MAX_OPTIONS_POSITIONS", 6)
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
    _dd_factor = 0.50 if dd <= -0.15 else (0.75 if dd <= -0.10 else 1.00)
    # DD RISK BUDGET 2026-11-06 (position_mgmt): the linear 10% -> 18% budget
    # and this ladder are ONE drawdown factor — the tighter of the two.
    try:
        import position_mgmt as _pm_dd
        _dd_factor, _ = _pm_dd.combine_drawdown_scalars(_dd_factor, _pm_dd.dd_risk_scalar(-dd * 100.0))
    except ImportError as _pm_dd_err:
        logger.warning(f"T1 drawdown budget unavailable, ladder only: {_pm_dd_err}")
    _vxx_rung = size_scalar
    size_scalar *= _dd_factor

    # REGIME ENGINE step 6 (docs/regime/FINDINGS.md; flag VOLTRADE_REGIME_SIZING=tier1): the
    # engine's options multiplier REPLACES the VXX ladder above (one regime multiplier, never
    # two); the drawdown ladder stays (a risk mechanism, not a regime read). Stale state -> 1.0.
    _rg_sz = None
    try:
        import regime_engine as _re_sz
        _rg_sz = _re_sz.sizing_for("tier1")
        # COUNTERFACTUAL (2026-10-28 live flip, logging only): the legacy VXX rung next to the
        # engine's multiplier (the drawdown factor multiplies both and is not part of either).
        _re_sz.counterfactual_sizing("tier1", "tier1", {"multiplier": _vxx_rung, "vxx_rung": _vxx_rung},
                                     _rg_sz, "tiered_strategy.tier1_csp_core")
    except Exception as _re_sz_err:
        logger.debug(f"T1 regime sizing unavailable: {_re_sz_err}")
    if _rg_sz is not None:
        size_scalar = float(_rg_sz["multiplier"]) * _dd_factor

    # REGIME ENGINE step 5 (docs/regime/FINDINGS.md; flag VOLTRADE_REGIME_OPTIONS):
    # the engine's verdict on NEW short puts. SHADOW by default — the flags ride
    # the action metadata and every guard that WOULD act logs a
    # would_reject_regime_<g> shadow record (prevention-P&L first); only the
    # gates named in the flag block or scale for real. A blocked name is simply
    # not proposed — nothing here touches an order path. The size multiplier
    # rides size_scalar exactly like the VXX ladder above (the affordability
    # cap + size_pct; the contract count itself is select_contract's until the
    # master plan's N8 threads trade context through — AUDIT §2 T7).
    _rg = _t1_regime_context()
    if _rg["size_mult"] != 1.0:
        size_scalar *= _rg["size_mult"]
    _rg_target_delta = _rg["target_delta"] if _rg["target_delta"] is not None else T1_TARGET_DELTA
    _rg_seen: Dict[str, int] = {}
    _rg_rejections: List[Dict[str, Any]] = []
    # KB #45 (c), 2026-11-04: an ABSENT name reads as "no report" only on a calendar that was
    # fetched AND not cut. Otherwise each name about to be sold is confirmed per symbol below.
    _rg_earn_authoritative = bool(_rg["earnings_ok"]) and not _rg["earnings_truncated"]
    _rg_confirm = _t1_confirm_state()

    # Current tickers already held (avoid doubling up)
    # BUG FIX 2026-10-13 (f5-t1-occ-dedupe): option positions carry OCC
    # symbols ("DRAM261120P00045000"), but the loop below tests the plain
    # underlying ticker ("DRAM") against this set — so the de-dupe never
    # matched and T1 re-proposed SELL_CSP on underlyings it already held,
    # every scan. Map each option position to its UNDERLYING first.
    held_tickers = {_option_underlying(p) for p in ctx.positions
                    if p.get("asset_class") in ("us_option", "option")}
    held_tickers.discard("")

    # Count current options positions to respect MAX_OPTIONS_POSITIONS
    current_position_count = len([p for p in ctx.positions
                                   if p.get("asset_class") in ("us_option", "option")])
    slots_available = max(0, max_positions - current_position_count)

    # DYNAMIC-UNIVERSE 2026-04-22: pull ranked top 200 per scan.
    # If Layer 1/2 fails, falls back to T1_TICKERS_FALLBACK.
    active_universe = _get_t1_universe()
    logger.debug(f"T1 scanning {len(active_universe)} tickers (dynamic)")

    # AFFORDABILITY PRE-FILTER 2026-05-22: skip tickers whose typical CSP
    # cash collateral would exceed the per-position budget. A 30-delta CSP
    # typically picks a strike ~5% OTM, so cash_required ≈ price × 0.95 × 100.
    # We need: equity × max_per_position × size_scalar >= cash_required.
    # Equivalently: price <= (equity × max_per_position × size_scalar) / 95.
    # Without this, T1 hands the dispatcher candidates that consistently
    # fail at the strike-affordability step (per audit log: LRCX/WDC/MSTR).
    affordable_budget = ctx.equity * max_per_position * size_scalar
    max_underlying_price_for_csp = affordable_budget / 95.0  # ~strike × 100 with 5% OTM
    logger.debug(
        f"T1 affordability gate: max underlying price = ${max_underlying_price_for_csp:.0f} "
        f"(budget=${affordable_budget:.0f})"
    )

    # Snapshot lookup for affordability gate. Falls back to ranked-universe
    # cache or skips the gate gracefully if no price data available.
    _price_lookup = {}
    try:
        from csp_universe import _load_layer2_cache
        _layer2 = _load_layer2_cache() or {}
        for entry in _layer2.get("scores", []):
            if entry.get("ticker") and entry.get("price", 0) > 0:
                _price_lookup[entry["ticker"]] = float(entry["price"])
    except Exception:
        pass  # gate becomes a no-op if no price data

    # EDGE P1 DEALER GAMMA (2026-11-06, dealer_gamma.py): one state-level verdict on a new
    # short put — allow, would_skip (logged only: gate 3 found skipping CSPs hurt) or skip
    # (DEALER_GAMMA_ENFORCE_SHORT_PUTS). Inactive without fresh data.
    _dg_t1 = _t1_dealer_gamma_decision()
    _dg_skips = 0
    _dg_rejections: List[Dict[str, Any]] = []

    # Candidates: liquid, high-IV names we don't already own
    for ticker in active_universe:
        if ticker in held_tickers:
            continue
        if slots_available <= 0:
            break  # respect MAX_OPTIONS_POSITIONS

        # AFFORDABILITY GATE 2026-05-22: skip if underlying too expensive.
        # If we have no price data we let the dispatcher decide (defense in depth
        # is in _select_sell_put's pre-filter and in csp_universe's Layer 1 cap).
        _u_price = _price_lookup.get(ticker)
        if _u_price and _u_price > max_underlying_price_for_csp:
            logger.debug(
                f"T1 skip {ticker}: price ${_u_price:.0f} > affordable ${max_underlying_price_for_csp:.0f}"
            )
            continue

        # REGIME ENGINE step 5: this name's verdict (state flags + its earnings date).
        _rg_meta = None
        if _rg["engine"] is not None:
            _rg_days = _rg["earnings"].get(ticker)
            _rg_earn_src = "calendar"
            _rg_v = _rg["engine"].csp_verdict(
                _rg["state"], {"days_to_earnings": _rg_days, "dte_max": T1_TARGET_DTE_MAX})
            if _rg_days is None and not _rg_earn_authoritative \
                    and not _rg["engine"].apply_csp_gates(_rg_v, _rg["gates"])["block"]:
                # KB #45 (c): the calendar cannot vouch for this ABSENT name (the lookup failed,
                # or Finnhub cut the window's nearest dates: the live IREN put). Establish its
                # date before selling it. When nobody can, the earnings guard refuses it.
                _rg_days, _rg_earn_src = _t1_confirm_earnings(ticker, _rg["earnings_window"], _rg_confirm)
                if _rg_earn_src == "unverified":
                    _rg_confirm["earnings_unverified"] += 1
                elif _rg_days is not None:
                    _rg_confirm["confirmed_in_window"] += 1
                _rg_v = _rg["engine"].csp_verdict(
                    _rg["state"], {"days_to_earnings": _rg_days, "dte_max": T1_TARGET_DTE_MAX,
                                   "earnings_unverified": _rg_earn_src == "unverified"})
            _rg_act = _rg["engine"].apply_csp_gates(_rg_v, _rg["gates"])
            _rg_meta = {"flags": _rg_v["flags"], "would_block": _rg_v["would_block"], "gates": sorted(_rg["gates"]),
                        "size_mult": _rg_v["size_mult"], "target_delta": _rg_v["target_delta"],
                        "earnings_lookup_ok": _rg["earnings_ok"], "composite": _rg_v["composite"],
                        "vol_state": _rg_v["vol_state"], "as_of": _rg_v["as_of"],
                        "earnings_source": _rg_earn_src, "earnings_calendar_truncated": _rg["earnings_truncated"]}
            if _rg_v["would_block"]:
                _g = (_rg_act["blocked_by"] or _rg_v["would_block"])[0]
                _decision = f"rejected_regime_{_g}" if _rg_act["block"] else f"would_reject_regime_{_g}"
                if _g == "earnings" and _rg_earn_src == "unverified":
                    _decision += "_unverified"   # its own prevention-P&L bucket: refused on missing data, not a known date
                # COUNTERFACTUAL (2026-10-28 live flip, logging only): legacy would propose this
                # name (no regime guard); the engine blocked or would block it. A name that passes
                # is logged where its action is appended below.
                _rg["engine"].counterfactual_options(ticker, _rg_v, _rg["gates"], "tiered_strategy.tier1_csp_core",
                                                     legacy_delta=T1_TARGET_DELTA, shadow_decision=_decision)
                _rg_seen[_decision] = _rg_seen.get(_decision, 0) + 1
                # bounded like the 3 actions a scan returns, once per name/guard/day
                if _rg_seen[_decision] <= 3 and _rg["engine"].shadow_once_per_day(f"t1:{ticker}:{_decision}"):
                    _rg_rejections.append({
                        "ticker": ticker, "decision": _decision, "reason": str(_rg_v["reasons"].get(_g, _g)),
                        "metadata": {"vxx_ratio": ctx.vxx_ratio, "regime_label": regime_label, "regime_guards": _rg_meta,
                                     # ML snapshot (2026-10-28 live flip; logging only) for the shadow record
                                     "regime_snapshot": _rg["engine"].ml_snapshot(_rg["state"], kind="options",
                                                                                  verdict=_rg_v, gates=_rg["gates"])},
                    })
                if _rg_act["block"]:
                    if _g in ("crisis", "vrp") and _rg_seen[_decision] >= 3:
                        break      # state-level: every name is blocked; the 3 that would have been proposed are logged
                    continue       # the name is removed, the slot is kept

        # EDGE P1 DEALER GAMMA: an enforced skip refuses the name before it takes a slot;
        # state-level like the crisis gate, so 3 logged refusals end the walk.
        if _dg_t1.get("action") == "skip":
            _dg_skips += 1
            _t1_dealer_gamma_log(_dg_t1, ticker, _u_price, ctx.vxx_ratio, regime_label, _dg_rejections)
            if _dg_skips >= 3:
                break
            continue

        # Position sizing: respect MAX_OPTIONS_PCT from system_config (8%)
        base_size = max_per_position * size_scalar
        slots_available -= 1
        # COUNTERFACTUAL (2026-10-28 live flip, logging only): a proposed name both systems take;
        # the row carries the engine's size multiplier / delta vs legacy's x1 / default delta.
        if _rg_meta is not None and not _rg_v["would_block"]:
            _rg["engine"].counterfactual_options(ticker, _rg_v, _rg["gates"], "tiered_strategy.tier1_csp_core",
                                                 legacy_delta=T1_TARGET_DELTA)

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
                "target_delta": _rg_target_delta,
                "take_profit_pct": T1_TAKE_PROFIT_PCT,
                "stop_loss_mult": T1_STOP_LOSS_MULT,
                "force_close_dte": T1_FORCE_CLOSE_DTE,
                "min_iv_rank": T1_MIN_IV_RANK,
                # CSP CAPITAL ALLOCATION counterfactual logging (open_questions.md,
                # filed 2026-07-28/2026-08-06): the live dispatcher (server/bot.ts)
                # needs these to label a capital-starved rejection's shadow record
                # the same way log_masterkill_csp_shadow already does — without
                # them it would have to re-derive regime context it has no access
                # to at the point options_execution.select_contract fails.
                "vxx_ratio": ctx.vxx_ratio,
                "regime_label": regime_label,
                # REGIME ENGINE step 5: the engine's verdict on this name (None when the
                # engine is off); regime_size_mult is what the gates applied to size_scalar.
                "regime_guards": _rg_meta,
                "regime_size_mult": _rg["size_mult"],
            },
        ))

    # EDGE P1 DEALER GAMMA: the verdict rides each returned SELL_CSP; a would-skip is
    # recorded for the names actually returned, its shadow record joins the batch below.
    _t1_dealer_gamma_finish(actions[:3], _dg_t1, ctx.vxx_ratio, regime_label, _price_lookup, _dg_rejections)
    _rg_rejections.extend(_dg_rejections)

    # REGIME ENGINE step 5: the guards' shadow records — each guard earns a
    # prevention-P&L (per-decision bucket on /api/diag/shadow) before promotion.
    if _rg_rejections:
        log_csp_gate_shadow(_rg_rejections)
    _t1_publish_earnings_diag(_rg, _rg_confirm)

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
    caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity, positions=ctx.positions)
    max_invested = caps.get("MAX_TOTAL_EXPOSURE", 1.00)
    current_invested = sum(abs(float(p.get("market_value", 0))) for p in ctx.positions)
    current_invested_pct = current_invested / max(ctx.equity, 1)
    equity_headroom_pct = max_invested - current_invested_pct

    if equity_headroom_pct <= 0:
        logger.info(f"T2: invested cap reached {current_invested_pct*100:.1f}% >= {max_invested*100:.0f}%{_budget_note(caps)}")
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

    # 2026-11-05: the volatility-managed floor (system_config FLOOR_MODE
    # "vol_managed_spy") owns SPY and sells QQQ as legacy. T3 buying either
    # would fight it (T3 buys, the floor trims or sells, T3 buys again), and
    # levered SPY in calm uptrends is what that floor already holds.
    from system_config import BASE_CONFIG as _t3_cfg
    if _t3_cfg.get("FLOOR_MODE", "basket") == "vol_managed_spy":
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

    # Pull caps from system_config (2026-11-06: satellite room x the regime risk budget)
    caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity, positions=ctx.positions)
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
            # 2026-11-06 (tax_filter.py): insurance — the tax filter logs it,
            # never prices it against SPY (a hedge's expected return is negative by design)
            "tax_role": "hedge",
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

    # 2026-11-06: the room above the passive floor is scaled by the regime risk budget.
    caps = get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity, positions=ctx.positions)
    max_invested_pct = caps.get("MAX_TOTAL_EXPOSURE", 1.00)

    current_invested_pct = sum(abs(float(p.get("market_value", 0))) for p in ctx.positions) / max(ctx.equity, 1)
    headroom = max(0.0, max_invested_pct - current_invested_pct)

    # CASH RESERVE 2026-11-06 (position_mgmt.satellite_headroom): the room the
    # regime's cash reserve leaves after the floor's commitment, current
    # satellites (short puts at their collateral) and working orders also caps
    # the tiers' new capital. The tighter of the two caps applies.
    reserve_bound, reserve = False, None
    try:
        import position_mgmt as _pm
        reserve = _pm.live_headroom(ctx.equity, ctx.positions, caps.get("regime") or ctx.regime,
                                    source="tiered_strategy._enforce_exposure_cap")
        if reserve.get("enabled") and ctx.equity > 0:
            reserve_pct = float(reserve["headroom_dollars"]) / ctx.equity
            if reserve_pct < headroom:
                headroom, reserve_bound = reserve_pct, True
    except (ImportError, KeyError, TypeError, ValueError) as _rsv_err:
        logger.warning(f"[TIERS] cash reserve unavailable, regime cap only: {_rsv_err}")

    protected = [a for a in actions if a.tier == 4]
    scalable = [a for a in actions if a.tier != 4]

    protected_size = sum(a.size_pct for a in protected)
    scalable_size = sum(a.size_pct for a in scalable)
    total_proposed = protected_size + scalable_size

    if total_proposed <= headroom:
        return actions

    if reserve_bound and max(0.0, headroom - protected_size) <= 0.0 and scalable:
        # No room at all above the reserve: drop the scalable actions rather than
        # pass size 0 on (select_contract would read 0 as "size it yourself").
        dropped = [a for a in scalable]
        logger.warning(f"[TIERS] cash reserve leaves no room: {len(dropped)} tier action(s) dropped "
                       f"({reserve.get('summary') if reserve else ''})")
        for a in dropped:
            _pm.log_decision("cash_reserve", "block", a.ticker,
                             f"{a.ticker}: Tier {a.tier} {a.action} dropped — {reserve.get('summary')}",
                             dedup_key=f"cr|tiers|{a.ticker}", dedup_seconds=6 * 3600,
                             source="tiered_strategy._enforce_exposure_cap", tier=a.tier,
                             wanted_pct=round(a.size_pct, 4), regime=reserve.get("regime"))
        # GATE COUNTERFACTUALS 2026-11-12 (logging only): each dropped action's tier and
        # dollars ride its shadow record; a dropped non-CSP action gets a record too.
        _gate, _blocks = _allocator_gate_blocks(ctx, caps, dropped, [a.size_pct for a in dropped],
                                                [0.0] * len(dropped), "dropped", reserve, True, protected_size)
        log_csp_gate_shadow([{"ticker": a.ticker, "decision": "rejected_cash_reserve",
                              "reason": "cash reserve leaves no room", "metadata": a.metadata, "gate": b}
                             for a, b in zip(dropped, _blocks) if a.action == "SELL_CSP"])
        _log_allocator_gate_rows(ctx, caps, [(a, b) for a, b in zip(dropped, _blocks) if a.action != "SELL_CSP"],
                                 _gate, "cash reserve leaves no room")
        return protected

    scalable_headroom = max(0.0, headroom - protected_size)

    if scalable_size > 0 and scalable_headroom < scalable_size:
        scale_factor = scalable_headroom / scalable_size
        logger.warning(
            f"[TIERS] Master allocator scaling down by {scale_factor:.2%} "
            f"(proposed {total_proposed*100:.1f}% > headroom {headroom*100:.1f}%, "
            f"regime={caps.get('regime')}){_budget_note(caps)}"
        )
        _proposed = [a.size_pct for a in scalable]   # gate counterfactuals (logging only): the sizes before the cut
        for a in scalable:
            a.size_pct *= scale_factor
            a.reason = f"{a.reason} [scaled {scale_factor:.0%} by cap enforcer]{_budget_note(caps)}"
            if reserve_bound:
                a.reason += " [cash reserve]"
                _pm.log_decision("cash_reserve", "trim", a.ticker,
                                 f"{a.ticker}: Tier {a.tier} {a.action} scaled to {scale_factor:.0%} of its size so "
                                 f"the cash reserve stays whole — {reserve.get('summary')}",
                                 dedup_key=f"cr|tiers-trim|{a.ticker}", dedup_seconds=6 * 3600,
                                 source="tiered_strategy._enforce_exposure_cap", tier=a.tier,
                                 scale=round(scale_factor, 4), regime=reserve.get("regime"))
        # GATE COUNTERFACTUALS 2026-11-12 (logging only; the sizes above are final): what the
        # scale-down cut, as shadow rows under the constraint that bound.
        _log_allocator_scale_shadow(ctx, caps, scalable, _proposed, reserve, reserve_bound, protected_size,
                                    scale_factor, current_invested_pct, total_proposed, max_invested_pct)

    return protected + scalable


class TieredStrategy:
    """Main entry point. Wire this into bot_engine.scan_market()."""

    def __init__(self, config_overrides: Optional[Dict] = None):
        self.config = config_overrides or {}

    def run_tiers(self, ctx: TierContext, on_phase: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Execute all tiers and return a unified action list + diagnostics.

        on_phase: optional callback invoked as each internal tier finishes
        (KNOWN BROKEN #18 continuation, 2026-07-27) — "tier1", "tier2",
        "tier3", "tier4", "allocator", in that order. run_tiers() is one
        synchronous call from bot_engine.py's perspective; if it hangs
        past the daemon's 300s budget, nothing about WHICH tier was still
        running got recorded anywhere — the caller's own TIMING-DISK file
        only gained a "tier_engine_breakdown" entry on successful return
        (bot_engine.py ~line 3782), so a killed mid-flight run left that
        whole span as dark as this item's very first defect. on_phase lets
        the caller (bot_engine.py) write a phase checkpoint to the SAME
        on-disk file progressively, the same pattern already used for
        every other phase boundary in the scan pipeline. Never allowed to
        raise into this function — a visibility hook must not be able to
        break the tier engine it's observing (same discipline as
        log_masterkill_csp_shadow below).

        Returns:
            {
                "killed": bool,
                "kill_reason": str,
                "actions": [TierAction, ...],
                "tier_stats": {1: count, 2: count, 3: count, 4: count},
                "tier_timings": {"tier1_sec": float, "tier2_sec": float, ...},
                "diagnostics": {...},
            }
        """
        def _notify(phase: str) -> None:
            if on_phase is None:
                return
            try:
                on_phase(phase)
            except Exception as e:
                logger.debug(f"run_tiers on_phase({phase!r}) callback raised, ignored: {e}")

        # ALPHA AUDIT 2026-05-04 batch 5: per-tier timing for bottleneck
        # diagnosis. Production scans are timing out at 300s with the
        # tier_engine_complete phase as the culprit, but we don't know
        # WHICH tier inside the engine is slow. Capturing per-tier wall
        # time so the next snapshot shows e.g. "tier1: 250s" or
        # "tier3: 280s" and we can target the actual hot path.
        import time as _t
        tier_timings = {}
        _t_start = _t.time()

        # Run master kill-switch first
        killed, reason = master_kill_switch(ctx)
        if killed:
            logger.warning(f"Master kill-switch: {reason}")
            # RULE REVIEW evidence gate (open_questions.md item #20) — log what
            # Tier 1 CSP would have proposed, never executed. Visibility only;
            # cannot affect the kill decision above or the actions returned.
            log_masterkill_csp_shadow(ctx, reason)
            return {
                "killed": True,
                "kill_reason": reason,
                "actions": [],
                "tier_stats": {},
                # gate_shadow (2026-11-12): what the kill's shadow recorded, on the same path
                # to /api/diag/timings as the per-tier timings (visibility only).
                "tier_timings": {"killed_at_sec": round(_t.time() - _t_start, 2),
                                 "gate_shadow": get_gate_shadow_summary()},
                "diagnostics": self._diagnostics(ctx),
            }

        all_actions = []

        # Tier 1 — CSP core (always runs)
        _t_phase = _t.time()
        t1_actions = tier1_csp_core(ctx)
        tier_timings["tier1_sec"] = round(_t.time() - _t_phase, 2)
        logger.info(f"T1 CSP: {len(t1_actions)} candidates [{tier_timings['tier1_sec']}s]")

        # KNOWN BROKEN #18 continuation (2026-07-19): surface whether this
        # tier1_sec reading was a CSP Layer 2 scoring cache hit (near-instant)
        # or a fresh compute (up to 150 tickers x 2 Alpaca-bound calls in a
        # 45s wall budget) so a slow tier1_sec is diagnosable instead of a
        # guess — see csp_universe.get_last_layer2_prefetch_stats() docstring.
        try:
            from csp_universe import get_last_layer2_prefetch_stats
        except ImportError:
            pass
        else:
            tier_timings["csp_layer2_prefetch"] = get_last_layer2_prefetch_stats()
        # KB #45 (c), 2026-11-04: the Tier-1 earnings guard's view of the calendar (rows, window,
        # earliest/latest date returned, truncated, lookup_ok, per-symbol confirmations), on the
        # same path to /api/diag/timings as csp_layer2_prefetch.
        tier_timings["t1_earnings_coverage"] = get_t1_earnings_coverage()
        _notify("tier1")

        # Tier 2 — multiply T1 sizing (if PM approved)
        _t_phase = _t.time()
        t1_after_t2 = tier2_leverage_multiplier(ctx, t1_actions)
        tier_timings["tier2_sec"] = round(_t.time() - _t_phase, 2)
        all_actions.extend(t1_after_t2)
        _notify("tier2")

        # Tier 3 — trend capture (independent of T1)
        _t_phase = _t.time()
        t3_actions = tier3_trend_capture(ctx)
        tier_timings["tier3_sec"] = round(_t.time() - _t_phase, 2)
        all_actions.extend(t3_actions)
        logger.info(f"T3 trend: {len(t3_actions)} candidates [{tier_timings['tier3_sec']}s]")
        _notify("tier3")

        # Tier 4 — tail hedge (always on, small size)
        _t_phase = _t.time()
        t4_actions = tier4_tail_hedge(ctx)
        tier_timings["tier4_sec"] = round(_t.time() - _t_phase, 2)
        all_actions.extend(t4_actions)
        logger.info(f"T4 hedge: {len(t4_actions)} candidates [{tier_timings['tier4_sec']}s]")
        _notify("tier4")

        # ── MASTER ALLOCATOR ──────────────────────────────────────────────
        # Tiers run independently and don't know about each other's
        # allocations. This stage enforces MAX_TOTAL_EXPOSURE across all
        # tier actions combined. If the sum of proposed sizes + current
        # positions would exceed the cap, scale down proportionally.
        # T4 (tail hedge) is protected from scale-down — it's small and
        # load-bearing for the overall system safety.
        _t_phase = _t.time()
        all_actions = _enforce_exposure_cap(ctx, all_actions)
        tier_timings["allocator_sec"] = round(_t.time() - _t_phase, 2)
        _notify("allocator")

        tier_timings["total_sec"] = round(_t.time() - _t_start, 2)

        # Tier stats
        stats = {i: sum(1 for a in all_actions if a.tier == i) for i in (1, 2, 3, 4)}

        # Log a summary of where time went so it shows up in stderr/logs
        # even before the diagnostics dict reaches the snapshot endpoint.
        logger.info(
            f"[TIER-TIMING] total={tier_timings['total_sec']}s "
            f"t1={tier_timings.get('tier1_sec')}s "
            f"t2={tier_timings.get('tier2_sec')}s "
            f"t3={tier_timings.get('tier3_sec')}s "
            f"t4={tier_timings.get('tier4_sec')}s "
            f"alloc={tier_timings.get('allocator_sec')}s"
        )

        return {
            "killed": False,
            "kill_reason": "",
            "actions": all_actions,
            "tier_stats": stats,
            "tier_timings": tier_timings,
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


def rebaseline_peak_equity(new_peak: float, basis: str) -> Dict[str, Any]:
    """WRITE-DOWN of the shared peak (PEAK_EQUITY_PATH, voltrade_peak_equity.json)
    to a validated equity — the ONLY function that lowers it.

    2026-11-11 (KNOWN BROKEN #49, research/wishlist.md option (a)): the paper
    re-baseline (human-directed 2026-09-28) lowered Node's equityPeak and
    bot_engine's DD-halt peak but could not lower this file (only max-ratchet
    writers existed), so tiered_strategy's master kill / T2 gate and
    risk_kill_switch's -20% kill kept measuring from the pre-re-baseline high.
    Callers: paper_resume_sync only (a paper re-baseline, or its one-time
    catch-up). update_peak_equity stays the only writer that RAISES it.

    Refuses a non-finite or non-positive value and never raises the peak: a
    target at or above the stored peak changes nothing. Returns
    {"before", "after", "changed", "basis", "note"}; never raises.
    """
    out: Dict[str, Any] = {"before": None, "after": None, "changed": False, "basis": basis, "note": ""}
    try:
        target = float(new_peak)
    except (TypeError, ValueError):
        out["note"] = "refused: not a number"
        return out
    if not math.isfinite(target) or target <= 0:
        out["note"] = "refused: not a credible positive number"
        return out
    before = 0.0
    try:
        if os.path.exists(PEAK_EQUITY_PATH):
            with open(PEAK_EQUITY_PATH) as f:
                before = float(json.load(f).get("peak_equity", 0) or 0)
    except Exception as e:
        out["note"] = f"refused: stored peak unreadable ({e})"
        return out
    out["before"] = before
    out["after"] = before
    if target >= before:
        out["note"] = "no change: the target is not below the stored peak (this function never raises it)"
        return out
    try:
        import tempfile
        dirname = os.path.dirname(PEAK_EQUITY_PATH) or "."
        fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".peak.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"peak_equity": target,
                           "last_updated": datetime.now().isoformat(),
                           "rebaselined_from": before,
                           "rebaselined_basis": str(basis)}, f)
            os.replace(tmp, PEAK_EQUITY_PATH)
        except Exception:
            try: os.unlink(tmp)
            except OSError: pass  # the temp file may not exist; the write error below is the one reported
            raise
    except Exception as e:
        out["note"] = f"write failed: {e}"
        return out
    out["after"] = target
    out["changed"] = True
    out["note"] = f"shared peak written down {before:.2f} -> {target:.2f} ({basis})"
    logger.warning(f"[peak] {out['note']}")
    return out


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
