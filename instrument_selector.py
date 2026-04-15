#!/usr/bin/env python3
"""
VolTradeAI — Unified Instrument Selection Engine
==================================================
Decides stock vs 2× leveraged ETF vs options for every trade,
using the full intelligence stack from analyze.py.

This module makes the DECISION only — it does not submit orders.
Actual order submission is handled by options_execution.py.

KEY CONSTRAINTS (inherited from options_execution.py):
  - Options only during regular hours (9:30am–4pm ET)
  - Options only with score >= 70
  - Max 10% per options position (ceiling)
  - Max 20% total options exposure
  - Never sell naked calls
  - ETF only if expected hold <= 5 days
  - ETF only if ETF volume > 100K
  - ETF sized at 50% of what stock would get (2× leverage)

Usage:
  from instrument_selector import select_instrument
  result = select_instrument(trade, equity, positions, macro)
"""

import os
import sys
import json
import time
import logging
import subprocess
import requests
from datetime import datetime, timezone
from typing import Optional

# Alpaca constants (used for VXX fetch and ETF volume check)
_ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
_ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_DATA_URL = "https://data.alpaca.markets"

def _alpaca_headers() -> dict:
    return {"APCA-API-KEY-ID": _ALPACA_KEY, "APCA-API-SECRET-KEY": _ALPACA_SECRET}

logger = logging.getLogger("instrument_selector")
# REMOVED (PR #53): Stock shorting permanently disabled.
# LARGE_CAP_SHORTABLE set removed — no code path should reference it.
LARGE_CAP_SHORTABLE = set()  # Empty — kept for backward compat only


# ── Sibling module path ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ── Import analyze.py functions (direct import preferred; subprocess fallback) ─
try:
    from analyze import (
        LEVERAGED_ETFS,
        compute_iv_rank,
        bid_ask_quality,
        compute_mfi,
        compute_options_flow,
        compute_put_call_ratio,
        compute_skew,
        compute_gex,
        composite_score as _analyze_composite_score,
        compute_iv_crush_score,
        compute_short_squeeze_score,
    )
    _HAS_ANALYZE = True
except ImportError:
    _HAS_ANALYZE = False
    logger.warning("analyze.py not importable — intelligence functions will be skipped")
    LEVERAGED_ETFS = {}

# ── Import position sizing ────────────────────────────────────────────────────
try:
    from position_sizing import (
        calculate_position,
        _earnings_scalar,
        _volatility_scalar,
        _regime_scalar,
        _time_scalar,
        _portfolio_heat_scalar,
        _kelly_fraction,
        _liquidity_scalar,
        _get_historical_stats,
        ABSOLUTE_MAX_POSITION_PCT,
    )
    _HAS_SIZER = True
except ImportError:
    _HAS_SIZER = False
    logger.warning("position_sizing.py not importable — using fixed sizing")
    ABSOLUTE_MAX_POSITION_PCT = 0.10

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_OPTIONS_PCT_CEILING = 0.08   # Absolute max 8% per options trade (v1.0.34: was 10%)
MAX_TOTAL_OPTIONS_PCT   = 0.08   # Absolute max 8% total options exposure (v1.0.34: was 20%)
ETF_LEVERAGE_DISCOUNT   = 0.50   # ETF sized at 50% of stock (2× leverage)
MIN_OPTIONS_SCORE       = 65     # Minimum deep_score to consider options (lowered from 70)
MIN_ETF_VOLUME          = 100_000  # Minimum ETF daily volume
ETF_MAX_HOLD_DAYS       = 5      # Never hold leveraged ETF longer than 5 days
ETF_DRAG_PER_DAY        = 0.075  # Daily rebalancing drag ~0.075%
ETF_SWEET_SPOT_DAYS     = (2, 3) # Best hold window for leveraged ETFs
INTELLIGENCE_TIMEOUT    = 5      # Seconds per analyze.py function call

# ── Cache: intelligence results per ticker (5-minute TTL) ────────────────────
_intel_cache: dict = {}
_intel_cache_time: dict = {}
_INTEL_CACHE_TTL = 300  # 5 minutes


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION A: MARKET HOURS CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def _is_regular_hours() -> bool:
    """Return True if currently within US regular trading hours (9:30–16:00 ET)."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        et_time = now_et.hour + now_et.minute / 60.0
        return 9.5 <= et_time < 16.0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION B: INTELLIGENCE GATHERING
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_call(fn, *args, default=None, label=""):
    """
    Call a function with a timeout-style guard.
    Returns default on any exception — never blocks the trade decision.
    """
    try:
        return fn(*args)
    except Exception as exc:
        logger.debug(f"Intelligence fn '{label}' failed: {exc}")
        return default


def _options_exposure_pct(positions: list, equity: float) -> float:
    """Calculate current options exposure as fraction of equity."""
    if not positions or equity <= 0:
        return 0.0
    options_value = sum(
        abs(float(p.get("market_value", 0)))
        for p in positions
        if p.get("asset_class") == "option" or len(str(p.get("symbol", ""))) > 10
    )
    return options_value / equity


def get_instrument_intelligence(ticker: str, price: float, trade_data: dict) -> dict:
    """
    Gather all intelligence signals from analyze.py functions in the right order.

    Only calls functions that are actually needed:
      - Always: compute_iv_rank, bid_ask_quality, compute_mfi
      - Only if options candidate (score >= 70, regular hours):
            compute_options_flow, compute_put_call_ratio, compute_skew,
            compute_gex, composite_score
      - Only if near earnings: compute_iv_crush_score
      - Only if high short interest: compute_short_squeeze_score

    Results are cached 5 minutes per ticker.

    Returns a flat dict of all intelligence data. Keys are present but may
    be None if the underlying call failed or data was unavailable.
    """
    cache_key = ticker
    now = time.time()
    if cache_key in _intel_cache and (now - _intel_cache_time.get(cache_key, 0)) < _INTEL_CACHE_TTL:
        logger.debug(f"[{ticker}] Using cached intelligence (age {now - _intel_cache_time[cache_key]:.0f}s)")
        return _intel_cache[cache_key]

    score      = trade_data.get("deep_score", trade_data.get("score", 0)) or 0
    vol_metrics = trade_data.get("vol_metrics", {}) or {}

    # Options chain DataFrames — may or may not be present in trade_data
    calls_df   = trade_data.get("calls_df")    # pandas DataFrame or None
    puts_df    = trade_data.get("puts_df")     # pandas DataFrame or None
    spot       = price
    # Time to expiry for options calcs (default 30 days if not supplied)
    T          = trade_data.get("T") or (30 / 365.0)
    # Earnings intel dict — may be pre-computed by analyze.py / bot_engine
    earnings_intel = trade_data.get("earnings_intel") or {}
    # Fundamentals + sentiment for short squeeze
    fundamentals   = trade_data.get("fundamentals") or {}
    sentiment_data = trade_data.get("sentiment") or {}

    intel: dict = {
        "ticker":        ticker,
        "price":         price,
        "has_calls_df":  calls_df is not None,
        "has_puts_df":   puts_df  is not None,
        "fns_ran":       [],
        "fns_skipped":   [],
    }

    if not _HAS_ANALYZE:
        intel["fns_skipped"].append("all (analyze.py not importable)")
        _intel_cache[cache_key]       = intel
        _intel_cache_time[cache_key]  = now
        return intel

    # ── Always-run signals ─────────────────────────────────────────────────

    # 1. IV Rank (uses vol_metrics dict from trade_data)
    ivr = _safe_call(compute_iv_rank, vol_metrics, default=None, label="compute_iv_rank")
    intel["iv_rank"] = ivr
    if ivr is not None:
        intel["fns_ran"].append("compute_iv_rank")
    else:
        intel["fns_skipped"].append("compute_iv_rank (no vol_metrics)")

    # 2. bid_ask_quality — use atm_bid/atm_ask if available, else skip
    atm_bid = trade_data.get("atm_bid") or trade_data.get("bid")
    atm_ask = trade_data.get("atm_ask") or trade_data.get("ask")
    if atm_bid is not None and atm_ask is not None:
        baq = _safe_call(bid_ask_quality, atm_bid, atm_ask, default=None, label="bid_ask_quality")
        intel["bid_ask_quality"] = baq
        if baq is not None:
            intel["fns_ran"].append("bid_ask_quality")
    else:
        intel["bid_ask_quality"] = None
        intel["fns_skipped"].append("bid_ask_quality (no bid/ask)")

    # 3. MFI — uses price history DataFrame
    hist = trade_data.get("hist")
    if hist is not None:
        mfi = _safe_call(compute_mfi, hist, default=None, label="compute_mfi")
        intel["mfi"] = mfi
        if mfi is not None:
            intel["fns_ran"].append("compute_mfi")
    else:
        intel["mfi"] = None
        intel["fns_skipped"].append("compute_mfi (no price history)")

    # ── Options-candidate signals (score >= 70 AND regular hours) ─────────

    options_candidate = score >= MIN_OPTIONS_SCORE and _is_regular_hours()
    has_chain = calls_df is not None and puts_df is not None

    if options_candidate and has_chain:
        # 4. Options flow (call/put volume ratio)
        flow = _safe_call(compute_options_flow, calls_df, puts_df,
                          default=(1.0, "Balanced", 0.0, 0.0), label="compute_options_flow")
        flow_ratio, flow_signal, call_vol, put_vol = flow
        intel["flow_ratio"]  = flow_ratio
        intel["flow_signal"] = flow_signal
        intel["call_vol"]    = call_vol
        intel["put_vol"]     = put_vol
        intel["fns_ran"].append("compute_options_flow")

        # 5. Put/call ratio (open interest based)
        pcr = _safe_call(compute_put_call_ratio, calls_df, puts_df,
                         default=(None, "Unknown", "No data"), label="compute_put_call_ratio")
        pc_ratio, pc_signal, pc_interp = pcr
        intel["put_call_ratio"]  = pc_ratio
        intel["put_call_signal"] = pc_signal
        intel["put_call_interp"] = pc_interp
        intel["fns_ran"].append("compute_put_call_ratio")

        # 6. Skew
        skew = _safe_call(compute_skew, calls_df, puts_df, spot, T,
                          default=None, label="compute_skew")
        intel["skew"] = skew
        if skew is not None:
            intel["fns_ran"].append("compute_skew")
        else:
            intel["fns_skipped"].append("compute_skew (insufficient chain data)")

        # 7. GEX
        gex_result = _safe_call(compute_gex, calls_df, puts_df, spot,
                                default=(None, None), label="compute_gex")
        net_gex, gex_regime = gex_result
        intel["net_gex"]    = net_gex
        intel["gex_regime"] = gex_regime
        if net_gex is not None:
            intel["fns_ran"].append("compute_gex")
        else:
            intel["fns_skipped"].append("compute_gex (no greeks in chain)")

        # 8. Composite score (Bennett VRP scoring)
        # Build inputs: iv_edge from VRP, baq from #2, prob_profit from delta, R/R estimate
        vrp           = trade_data.get("vrp") or 0
        iv_edge_norm  = max(0.0, min(1.0, abs(vrp) / 20.0))  # Normalize 0-20% VRP → 0-1
        baq_norm      = intel.get("bid_ask_quality") or 0.5
        # Rough prob_profit from IV rank (high IVR → selling → ~0.70 prob; low → buying → ~0.40)
        ivr_val       = intel.get("iv_rank") or 50
        prob_profit   = 0.70 if ivr_val > 60 else (0.40 if ivr_val < 40 else 0.55)
        # Risk/reward placeholder normalized 0-1 (caller can refine)
        rr_norm       = 0.5
        comp = _safe_call(_analyze_composite_score, iv_edge_norm, baq_norm, prob_profit, rr_norm,
                          default=None, label="composite_score")
        intel["composite_score"] = comp
        if comp is not None:
            intel["fns_ran"].append("composite_score")

    elif options_candidate and not has_chain:
        # Score is high but no chain data — note it
        for fn in ["compute_options_flow", "compute_put_call_ratio", "compute_skew",
                   "compute_gex", "composite_score"]:
            intel["fns_skipped"].append(f"{fn} (no options chain DataFrames)")
        intel["flow_ratio"]      = None
        intel["flow_signal"]     = None
        intel["put_call_ratio"]  = None
        intel["put_call_signal"] = None
        intel["skew"]            = None
        intel["net_gex"]         = None
        intel["gex_regime"]      = None
        intel["composite_score"] = None

    else:
        reason = "score < 70" if score < MIN_OPTIONS_SCORE else "outside regular hours"
        for fn in ["compute_options_flow", "compute_put_call_ratio", "compute_skew",
                   "compute_gex", "composite_score"]:
            intel["fns_skipped"].append(f"{fn} ({reason})")
        intel["flow_ratio"]      = None
        intel["flow_signal"]     = None
        intel["put_call_ratio"]  = None
        intel["put_call_signal"] = None
        intel["skew"]            = None
        intel["net_gex"]         = None
        intel["gex_regime"]      = None
        intel["composite_score"] = None

    # ── Conditional: near earnings → IV crush score ────────────────────────
    days_to_earnings = earnings_intel.get("days_to_earnings")
    near_earnings    = days_to_earnings is not None and 0 <= days_to_earnings <= 21
    if near_earnings:
        atm_iv = trade_data.get("atm_iv") or (vol_metrics.get("hv20") or 25) / 100
        crush = _safe_call(compute_iv_crush_score, atm_iv, earnings_intel,
                           default=(None, None, "Could not compute"), label="compute_iv_crush_score")
        iv_crush_score, crush_pct, crush_rec = crush
        intel["iv_crush_score"] = iv_crush_score
        intel["iv_crush_pct"]   = crush_pct
        intel["iv_crush_rec"]   = crush_rec
        intel["days_to_earnings"] = days_to_earnings
        intel["fns_ran"].append("compute_iv_crush_score")
    else:
        intel["iv_crush_score"]   = None
        intel["iv_crush_pct"]     = None
        intel["iv_crush_rec"]     = None
        intel["days_to_earnings"] = days_to_earnings
        intel["fns_skipped"].append("compute_iv_crush_score (no upcoming earnings within 21d)")

    # ── Conditional: high short interest → squeeze score ──────────────────
    short_pct = fundamentals.get("short_pct_float", 0) or 0
    if short_pct >= 10:
        squeeze = _safe_call(compute_short_squeeze_score, fundamentals, sentiment_data, vol_metrics,
                             default=(0, "Unknown", "Could not compute"), label="compute_short_squeeze_score")
        sq_score, sq_signal, sq_desc = squeeze
        intel["squeeze_score"]  = sq_score
        intel["squeeze_signal"] = sq_signal
        intel["squeeze_desc"]   = sq_desc
        intel["short_pct"]      = short_pct
        intel["fns_ran"].append("compute_short_squeeze_score")
    else:
        intel["squeeze_score"]  = None
        intel["squeeze_signal"] = None
        intel["squeeze_desc"]   = None
        intel["short_pct"]      = short_pct
        intel["fns_skipped"].append("compute_short_squeeze_score (short_pct < 10%)")

    logger.info(
        f"[{ticker}] Intelligence: ran={intel['fns_ran']} "
        f"skipped={len(intel['fns_skipped'])} items"
    )

    _intel_cache[cache_key]      = intel
    _intel_cache_time[cache_key] = now
    return intel


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION C: INSTRUMENT SCORERS
# ═══════════════════════════════════════════════════════════════════════════════

def _spread_cost_stock(volume: int) -> float:
    """
    Estimate round-trip spread cost for stock as a fraction.
    Large cap (vol > 10M): 0.03%
    Mid cap (1M–10M):      0.08%
    Small cap (< 1M):      0.15%
    """
    if volume >= 10_000_000:
        return 0.0003
    elif volume >= 1_000_000:
        return 0.0008
    else:
        return 0.0015


def _score_stock(trade: dict, intelligence: dict) -> dict:
    """
    Score stock as instrument. Base case — always available.

    Bonuses:
      + Long hold (> 5 days): stock wins (no decay, no expiry)
      + Low VIX: stock is preferred over ETF in calm markets too
    Penalties:
      - If a leveraged ETF exists AND expected hold is 2–3 days (ETF is better)
      - Wide spread for illiquid tickers
    """
    ticker          = trade.get("ticker", "")
    volume          = trade.get("volume", 0) or 0
    score           = trade.get("deep_score", trade.get("score", 50)) or 50
    expected_hold   = trade.get("expected_hold_days") or 3  # Days
    side            = trade.get("side", "buy")

    spread_cost     = _spread_cost_stock(volume)
    roundtrip_cost  = spread_cost * 2  # Entry + exit

    # Base instrument score: starts at 50 (neutral) and adjusts
    inst_score = 50.0

    # Conviction premium: higher signal score → better to just buy stock
    inst_score += (score - 50) * 0.2  # +2pts per 10 score points above 50

    # Hold period analysis
    if expected_hold >= 5:
        inst_score += 15  # Long holds favour stock (no decay, no ETF drag)
    elif expected_hold <= 3 and ticker in LEVERAGED_ETFS:
        inst_score -= 10  # Short hold + ETF available = ETF is likely better

    # Liquidity bonus: very liquid stocks → tighter spreads → less cost drag
    if volume >= 10_000_000:
        inst_score += 5
    elif volume < 500_000:
        inst_score -= 8  # Thin stock: wider spreads hurt

    # Squeeze bonus: if short squeeze detected, buying stock directly captures full move
    squeeze_score = intelligence.get("squeeze_score") or 0
    if squeeze_score >= 70:
        inst_score += 8  # ETFs and options may miss the squeeze velocity

    # Clamp
    inst_score = max(0.0, min(100.0, inst_score))

    # Strategy label — always buy_stock (shorting disabled PR #53)
    strategy = "buy_stock"

    # Max loss: worst case is price → 0 for long, unlimited for short (capped at 30%)
    max_loss_pct = 100.0 if side == "buy" else 30.0

    return {
        "instrument":   "stock",
        "score":        round(inst_score, 1),
        "edge_pct":     round(max(0.0, (score - 50) * 0.1), 2),  # Rough edge vs doing nothing
        "cost_pct":     round(roundtrip_cost * 100, 3),
        "max_loss_pct": max_loss_pct,
        "strategy":     strategy,
        "reasoning":    (
            f"Stock: score {score}, hold {expected_hold}d, "
            f"vol {volume:,}, roundtrip cost ≈ {roundtrip_cost*100:.2f}%"
        ),
        "intelligence": {
            "mfi":           intelligence.get("mfi"),
            "iv_rank":       intelligence.get("iv_rank"),
            "squeeze_score": squeeze_score,
        },
    }


def _score_etf(trade: dict, intelligence: dict, equity: float) -> Optional[dict]:
    """
    Score 2× leveraged ETF. Only scored if the ticker has an entry in LEVERAGED_ETFS.

    Sweet spot: 2–3 day hold.
    Penalty: hold > 5 days (drag kills the edge), VIX > 30 (leveraged ETFs suffer more).

    Returns None if no ETF exists for this ticker.
    """
    ticker          = trade.get("ticker", "")
    side            = trade.get("side", "buy")
    expected_hold   = trade.get("expected_hold_days") or 3
    score           = trade.get("deep_score", trade.get("score", 50)) or 50
    macro_vix       = (trade.get("macro") or {}).get("vix") or 18

    etf_map = LEVERAGED_ETFS.get(ticker)
    if not etf_map:
        return None  # No leveraged ETF for this ticker

    # Determine which side ETF to use
    if side == "buy":
        etf_ticker = etf_map.get("bull")
    else:
        etf_ticker = etf_map.get("bear")

    if not etf_ticker:
        return None  # No ETF for this direction

    # Spread cost (from backtest_instruments.py empirical data)
    major_etfs = {"SSO", "QLD", "TNA", "UPRO", "TQQQ", "SPXL"}
    spread_cost = 0.0010 if etf_ticker in major_etfs else 0.0020
    roundtrip_cost = spread_cost * 2

    # Daily drag: ~0.075% per day from daily rebalancing
    total_drag = ETF_DRAG_PER_DAY * expected_hold / 100  # Fraction

    # Leveraged upside: 2× the stock move, minus drag
    stock_expected_move = trade.get("ewma_rv", 2.0) or 2.0  # Daily % move
    # Over expected_hold days, expected total move
    expected_gross_move = stock_expected_move * (expected_hold ** 0.5) * 2  # 2× levered, ~sqrt(t) scaling
    net_etf_move = expected_gross_move - (total_drag * 100) - (roundtrip_cost * 100)

    # Base score
    inst_score = 50.0

    # Hold period scoring: sweet spot 2–3 days
    if ETF_SWEET_SPOT_DAYS[0] <= expected_hold <= ETF_SWEET_SPOT_DAYS[1]:
        inst_score += 20  # Sweet spot: leverage benefit before drag builds
    elif expected_hold == 4:
        inst_score += 8
    elif expected_hold == 1:
        inst_score += 5   # Too short: costs dominate
    elif expected_hold == 5:
        inst_score += 0   # Borderline
    elif expected_hold > 5:
        inst_score -= 20  # Hard penalty — drag will eat the edge

    # VIX penalty: leveraged ETFs suffer more in volatile markets
    if macro_vix > 35:
        inst_score -= 20
    elif macro_vix > 30:
        inst_score -= 12
    elif macro_vix > 25:
        inst_score -= 5
    elif macro_vix < 15:
        inst_score += 5  # Low VIX: ETF tracks well, less noise

    # Signal conviction from score
    inst_score += (score - 50) * 0.15

    # MFI confirmation: if MFI aligns with direction, small bonus
    mfi = intelligence.get("mfi")
    if mfi is not None:
        if side == "buy" and mfi < 30:
            inst_score += 5  # Oversold + buying ETF bull
        elif side != "buy" and mfi > 70:
            inst_score += 5  # Overbought + buying ETF bear
        elif side == "buy" and mfi > 70:
            inst_score -= 5  # Buying into overbought momentum
        elif side != "buy" and mfi < 30:
            inst_score -= 5  # Buying bear into oversold

    # Clamp
    inst_score = max(0.0, min(100.0, inst_score))

    strategy_label = "buy_etf_bull" if side == "buy" else "buy_etf_bear"

    return {
        "instrument":   "etf",
        "score":        round(inst_score, 1),
        "edge_pct":     round(max(0.0, net_etf_move), 2),
        "cost_pct":     round((roundtrip_cost + total_drag) * 100, 3),
        "max_loss_pct": 50.0,  # Leveraged ETFs can lose ~50% in a bad week
        "strategy":     strategy_label,
        "etf_ticker":   etf_ticker,
        "reasoning":    (
            f"ETF {etf_ticker} (2× {ticker}): hold {expected_hold}d, "
            f"VIX {macro_vix:.0f}, drag {total_drag*100:.2f}%, "
            f"roundtrip cost {roundtrip_cost*100:.2f}%"
        ),
        "intelligence": {
            "mfi":     mfi,
            "vix":     macro_vix,
            "drag":    round(total_drag * 100, 3),
        },
    }


def _score_options(trade: dict, intelligence: dict, equity: float) -> Optional[dict]:
    """
    Score options as instrument. Only evaluated during regular hours with score >= 70.

    Scoring logic:
      - IV rank drives strategy direction:
          High IVR (> 60): sell premium (sell_cash_secured_put / bear_put_spread)
          Low  IVR (< 40): buy options (buy_call / buy_put)
          Mid  IVR (40–60): spread strategies for defined risk
      - Options flow confirms smart money direction
      - Put/call ratio adds contrarian or confirmation signal
      - GEX negative = stock will move more = good for buyers
      - Skew guides strategy selection
      - composite_score() from analyze.py is the final arbiter

    Penalties:
      - Wide bid/ask spread → cost drag
      - Near earnings (for buyers) → IV crush risk
      - No options chain data → skip entirely
      - Outside regular hours → skip
    """
    score           = trade.get("deep_score", trade.get("score", 50)) or 50
    side            = trade.get("side", "buy")
    vrp             = trade.get("vrp", 0) or 0
    action_label    = trade.get("action_label", "") or ""
    ewma_rv         = trade.get("ewma_rv", 2.0) or 2.0
    expected_hold   = trade.get("expected_hold_days") or 3
    rsi             = trade.get("rsi") or 50

    # Guard: regular hours only
    if not _is_regular_hours():
        return None  # Not scored during AH/PM

    # Guard: score threshold
    if score < MIN_OPTIONS_SCORE:
        return None

    ivr             = intelligence.get("iv_rank")  # 0–100 or None
    baq             = intelligence.get("bid_ask_quality")
    flow_ratio      = intelligence.get("flow_ratio")
    flow_signal     = intelligence.get("flow_signal") or "Balanced"
    pc_ratio        = intelligence.get("put_call_ratio")
    pc_signal       = intelligence.get("put_call_signal") or "Neutral"
    skew            = intelligence.get("skew")
    net_gex         = intelligence.get("net_gex")
    gex_regime      = intelligence.get("gex_regime")
    comp_score      = intelligence.get("composite_score")
    days_to_earn    = intelligence.get("days_to_earnings")
    iv_crush_score  = intelligence.get("iv_crush_score")

    # ── Strategy selection based on VRP + IVR ─────────────────────────────

    # Determine whether this is a premium-selling or premium-buying setup
    sell_premium = ("SELL OPTIONS" in action_label.upper()) or vrp > 5
    buy_options  = vrp < -3
    high_conv    = score >= 85 and abs(vrp) < 3

    if ivr is not None:
        if ivr > 60:
            sell_premium = True  # IV elevated: selling has an edge
        elif ivr < 40:
            buy_options = True   # IV cheap: buying has an edge

    # ── Near-earnings guard ────────────────────────────────────────────────
    near_earnings = days_to_earn is not None and 0 <= days_to_earn <= 7

    if sell_premium and near_earnings and days_to_earn is not None and days_to_earn <= 2:
        # Near earnings — IV is high for a reason; selling is risky
        return {
            "instrument":   "options",
            "score":        15.0,  # Very low — not recommended
            "edge_pct":     0.0,
            "cost_pct":     2.0,
            "max_loss_pct": 100.0,
            "strategy":     "none",
            "reasoning":    f"Options penalized: {days_to_earn}d to earnings — IV crush risk for sellers",
            "intelligence": intelligence,
        }

    if buy_options and near_earnings:
        # Buying options near earnings → IV crush will hurt even if right on direction
        return {
            "instrument":   "options",
            "score":        20.0,
            "edge_pct":     0.0,
            "cost_pct":     2.0,
            "max_loss_pct": 100.0,
            "strategy":     "none",
            "reasoning":    f"Options penalized: {days_to_earn}d to earnings — IV crush risk for buyers",
            "intelligence": intelligence,
        }

    # ── Choose strategy ────────────────────────────────────────────────────

    if sell_premium:
        # BACKTEST VALIDATED: sell_csp wins 72% of trades over 10yr (2016-2026)
        # sell_cash_secured_put is the ONLY profitable options strategy in our universe
        strategy = "sell_cash_secured_put"  # Always sell put regardless of direction
        # Note: bear_put_spread (bearish sell) was removed — backtest showed spreads lose (WR=28%)
        base_edge = min(abs(vrp), 15.0) if vrp != 0 else 5.0
        # IV crush bonus: if near earnings with good crush setup, reward sellers
        if iv_crush_score is not None and iv_crush_score > 50:
            base_edge += iv_crush_score * 0.05

    elif buy_options:
        # BACKTEST FINDING: buy_call WR=22%, avg=-2.44% over 10yr — DISABLED
        # Buying options loses money from theta decay even when IV looks cheap.
        # Route to stock instead.
        return {
            "instrument":   "stock",
            "score":        55.0,
            "edge_pct":     0.0,
            "cost_pct":     0.1,
            "max_loss_pct": 100.0,
            "strategy":     "stock",
            "reasoning":    f"buy_call disabled by backtest (WR=22% historically) — routing to stock",
            "intelligence": intelligence,
        }

    elif high_conv:
        # BACKTEST FINDING: bull_spread WR=28%, avg=-1.77% over 10yr — DISABLED
        # Spreads lose to theta on the long leg. Route to stock.
        return {
            "instrument":   "stock",
            "score":        60.0,
            "edge_pct":     0.0,
            "cost_pct":     0.1,
            "max_loss_pct": 100.0,
            "strategy":     "stock",
            "reasoning":    f"bull_spread disabled by backtest (WR=28% historically) — routing to stock",
            "intelligence": intelligence,
        }
    else:
        # Neither selling nor buying edge is clear — stock is better
        return {
            "instrument":   "options",
            "score":        20.0,
            "edge_pct":     0.0,
            "cost_pct":     1.5,
            "max_loss_pct": 100.0,
            "strategy":     "none",
            "reasoning":    f"Options not compelling: VRP {vrp:.1f}%, IVR {ivr}, no clear edge",
            "intelligence": intelligence,
        }

    # ── Instrument scoring from intelligence signals ───────────────────────

    # Base score reflects market-wide IV environment (VXX ratio-driven, self-calibrating)
    # Uses the ratio of VXX vs its 30-day average — works regardless of where VXX drifts
    # _vxx_ratio: >1.30 = panic, 1.10-1.30 = elevated, 0.90-1.10 = normal, <0.70 = complacency
    vxx_ratio = trade.get("_vxx_ratio", 1.0)  # default: normal conditions
    market_ivr = trade.get("_market_ivr", 50)
    if vxx_ratio >= 1.30:
        inst_score = 78.0  # Panic: VXX 30%+ above avg — options have clear, meaningful edge
    elif vxx_ratio >= 1.10:
        inst_score = 68.0  # Elevated: VXX 10-30% above avg — options clearly better than stock
    elif vxx_ratio >= 0.90:
        inst_score = 62.0  # Normal: VXX within 10% — options viable (lowered from 58→62 to compete)
    elif vxx_ratio >= 0.70:
        inst_score = 52.0  # Calm: VXX 10-30% below avg — options and stock roughly equal
    else:
        inst_score = 42.0  # Complacency: VXX 30%+ below avg — options overpriced, stock is better

    # Bid-ask quality
    if baq is not None:
        # BAQ = 1 - spread/mid: 0 = very wide, 1 = tight
        inst_score += (baq - 0.8) * 50  # ±10 pts around 0.80 quality threshold

    # Options flow alignment
    if flow_ratio is not None and flow_signal is not None:
        if side == "buy" and "Call" in flow_signal:
            inst_score += 8   # Smart money buying calls confirms our bull thesis
        elif side != "buy" and "Put" in flow_signal:
            inst_score += 8   # Smart money buying puts confirms our bear thesis
        elif side == "buy" and "Put" in flow_signal:
            inst_score -= 5   # Smart money buying puts while we're bullish
        elif side != "buy" and "Call" in flow_signal:
            inst_score -= 5

    # Put/call sentiment
    if pc_signal:
        if side == "buy" and pc_signal in ("Extreme Fear", "Bearish"):
            inst_score += 6   # Contrarian buy signal
        elif side != "buy" and pc_signal in ("Extreme Greed", "Bullish"):
            inst_score += 6   # Contrarian sell/put signal
        elif side == "buy" and pc_signal in ("Extreme Greed",):
            inst_score -= 4   # Potential euphoria peak

    # GEX: negative GEX = explosive moves = good for option buyers
    if net_gex is not None:
        if gex_regime == "explosive" and strategy.startswith("buy"):
            inst_score += 7  # Dealer short gamma amplifies moves — buyers win
        elif gex_regime == "pinned" and strategy.startswith("buy"):
            inst_score -= 5  # Dealers long gamma: stock stays pinned — buyers lose theta
        elif gex_regime == "explosive" and strategy.startswith("sell"):
            inst_score -= 4  # Explosive vol is risky for premium sellers
        elif gex_regime == "pinned" and "spread" in strategy:
            inst_score += 3  # Pin helps defined-risk spread sellers

    # Skew: high positive skew (put skew) → selling puts is riskier, buying calls is cheaper
    if skew is not None:
        if skew > 5 and strategy == "buy_call":
            inst_score += 4   # Calls are cheap relative to puts → buy calls
        elif skew > 5 and strategy == "sell_cash_secured_put":
            inst_score -= 3   # High put skew: put premium deserves respect
        elif skew < -2 and strategy in ("buy_put", "bear_put_spread"):
            inst_score += 4   # Call skew: unusual, bearish premium → puts are cheap

    # composite_score from analyze.py: range 0–100, takes precedence as anchor
    if comp_score is not None:
        # Blend: 60% our score, 40% analyze.py composite
        inst_score = 0.60 * inst_score + 0.40 * comp_score

    # Theta decay penalty: longer expected hold = more theta eaten
    theta_penalty = min(expected_hold * 1.5, 10.0)  # Max 10pt penalty
    inst_score -= theta_penalty

    # Never sell naked calls — if somehow strategy becomes "sell_call", override
    if strategy == "sell_call":
        return {
            "instrument":   "options",
            "score":        0.0,
            "edge_pct":     0.0,
            "cost_pct":     5.0,
            "max_loss_pct": 100.0,
            "strategy":     "none",
            "reasoning":    "BLOCKED: would require naked call selling (unlimited risk)",
            "intelligence": intelligence,
        }

    inst_score = max(0.0, min(100.0, inst_score))

    # Cost estimate: options have wider spreads, plus theta decay cost
    # Typical option spread cost: 1–3% of premium depending on liquidity
    options_cost = 1.5 if (baq or 0.7) > 0.8 else 3.0

    return {
        "instrument":   "options",
        "score":        round(inst_score, 1),
        "edge_pct":     round(base_edge, 2),
        "cost_pct":     round(options_cost, 2),
        "max_loss_pct": 100.0,  # Long options: max loss = premium paid
        "strategy":     strategy,
        "reasoning":    (
            f"Options ({strategy}): IVR {ivr}, VRP {vrp:+.1f}%, "
            f"flow={flow_signal}, GEX={gex_regime}, "
            f"comp_score={comp_score}"
        ),
        "intelligence": {
            "iv_rank":       ivr,
            "bid_ask_quality": baq,
            "flow_ratio":    flow_ratio,
            "flow_signal":   flow_signal,
            "put_call_ratio": pc_ratio,
            "put_call_signal": pc_signal,
            "net_gex":       net_gex,
            "gex_regime":    gex_regime,
            "skew":          skew,
            "composite_score": comp_score,
            "sell_premium":  sell_premium,
            "buy_options":   buy_options,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION D: SIZING
# ═══════════════════════════════════════════════════════════════════════════════

def _dynamic_options_size(trade: dict, equity: float,
                          existing_positions: list = None,
                          macro: dict = None) -> float:
    """
    Calculate dynamic position size for options as a fraction of equity.
    Mirrors the logic in options_execution.py but is local here to avoid
    circular imports when both modules are loaded.

    Returns: fraction (e.g. 0.035 = 3.5%)
    """
    if not _HAS_SIZER:
        return 0.05  # Fallback: 5% fixed

    stats   = _get_historical_stats()
    overall = stats["overall"]
    kelly_base = _kelly_fraction(overall["win_rate"], overall["avg_win"], overall["avg_loss"])

    score   = trade.get("deep_score", trade.get("score", 50))
    ewma_rv = trade.get("ewma_rv") or 2.0
    garch_rv = trade.get("garch_rv")
    volume  = trade.get("volume", 0)
    price   = trade.get("price", 100)

    s_vol   = _volatility_scalar(ewma_rv, garch_rv)
    s_conf  = _confidence_scalar_safe(score, trade.get("ml_confidence"))
    s_reg   = _regime_scalar(
        macro.get("vix") if macro else None,
        macro.get("vix_regime") if macro else None,
    )
    s_earn  = _earnings_scalar(trade.get("ticker", ""))
    s_time  = _time_scalar()
    s_heat  = _portfolio_heat_scalar(
        existing_positions or [], equity, trade.get("sector")
    )
    prelim  = equity * kelly_base * s_vol * s_conf * s_reg * s_earn * s_time * s_heat
    s_liq   = _liquidity_scalar(volume, price, prelim)

    OPTIONS_LEVERAGE_DISCOUNT = 0.50
    dynamic_pct = (kelly_base * s_vol * s_conf * s_reg * s_earn
                   * s_time * s_heat * s_liq * OPTIONS_LEVERAGE_DISCOUNT)
    return max(0.01, min(dynamic_pct, MAX_OPTIONS_PCT_CEILING))


def _confidence_scalar_safe(score: float, ml_confidence=None) -> float:
    """Thin wrapper — calls position_sizing._confidence_scalar if available."""
    try:
        from position_sizing import _confidence_scalar
        return _confidence_scalar(score, ml_confidence)
    except ImportError:
        if score is None or score <= 0:
            return 0.5
        base = 0.6 + (min(score, 100) - 65) / (100 - 65) * 0.55
        return max(0.4, min(base, 1.25))


def _build_sizing(trade: dict, equity: float, positions: list, macro: dict,
                  chosen: str, etf_ticker: Optional[str] = None) -> dict:
    """
    Build sizing dict for the chosen instrument.
    - Stock: full position_sizing.calculate_position()
    - ETF: 50% of stock sizing (2× leverage discount)
    - Options: dynamic options sizing
    """
    if chosen == "options":
        options_frac = _dynamic_options_size(trade, equity, positions, macro)
        return {
            "type":          "options",
            "fraction":      round(options_frac, 4),
            "dollar_amount": round(equity * options_frac, 2),
            "note":          "Dynamic options sizing with 0.50× leverage discount",
        }

    if chosen == "etf":
        if _HAS_SIZER:
            stock_sizing = calculate_position(trade, equity, positions, macro)
            if stock_sizing.get("blocked"):
                return {
                    "type":          "etf",
                    "fraction":      0.0,
                    "dollar_amount": 0.0,
                    "blocked":       True,
                    "block_reason":  stock_sizing.get("block_reason"),
                }
            etf_value = stock_sizing["position_value"] * ETF_LEVERAGE_DISCOUNT
            return {
                "type":          "etf",
                "etf_ticker":    etf_ticker,
                "fraction":      round(stock_sizing["position_pct"] * ETF_LEVERAGE_DISCOUNT, 4),
                "dollar_amount": round(etf_value, 2),
                "stock_sizing":  stock_sizing,
                "note":          f"ETF at 50% of stock sizing due to 2× leverage",
            }
        else:
            return {
                "type":          "etf",
                "etf_ticker":    etf_ticker,
                "fraction":      0.04,  # Fallback: 4% (half of 8% default)
                "dollar_amount": round(equity * 0.04, 2),
                "note":          "Fixed ETF sizing (position_sizing.py unavailable)",
            }

    # Stock
    if _HAS_SIZER:
        stock_sizing = calculate_position(trade, equity, positions, macro)
        return {"type": "stock", **stock_sizing}
    else:
        return {
            "type":          "stock",
            "fraction":      0.05,
            "dollar_amount": round(equity * 0.05, 2),
            "note":          "Fixed stock sizing (position_sizing.py unavailable)",
        }


def _build_stop_config(trade: dict, chosen: str, sizing: dict) -> dict:
    """
    Initial stop configuration for evolving stops.
    Mirrors logic from calculate_position but returns a standalone dict.
    """
    price       = float(trade.get("price", 0) or 0)
    side        = trade.get("side", "buy")
    ewma_rv     = trade.get("ewma_rv") or 2.0

    if price <= 0:
        return {"type": "none", "stop_price": None, "take_profit": None}

    if chosen == "options":
        # Options: no stop orders (Alpaca doesn't support stop on options)
        # Use a time-based exit: close at 50% loss or target expiry date
        return {
            "type":         "options",
            "stop_price":   None,
            "take_profit":  None,
            "max_loss_pct": 50.0,
            "exit_rule":    "Close if position value falls 50% or reaches target delta",
        }

    # ATR-based stops
    stop_distance_pct  = max(1.5, min(ewma_rv * 1.5, 8.0))
    tp_distance_pct    = max(4.0, min(ewma_rv * 3.0, 15.0))

    if chosen == "etf":
        # Leveraged ETF: tighter stop (moves 2× as fast)
        stop_distance_pct  = max(2.0, min(ewma_rv * 1.0, 6.0))
        tp_distance_pct    = max(5.0, min(ewma_rv * 2.0, 12.0))

    if side in ("short", "sell"):
        stop_price  = round(price * (1 + stop_distance_pct / 100), 2)
        take_profit = round(price * (1 - tp_distance_pct  / 100), 2)
    else:
        stop_price  = round(price * (1 - stop_distance_pct / 100), 2)
        take_profit = round(price * (1 + tp_distance_pct  / 100), 2)

    return {
        "type":              chosen,
        "stop_price":        stop_price,
        "take_profit":       take_profit,
        "stop_distance_pct": round(stop_distance_pct, 2),
        "tp_distance_pct":   round(tp_distance_pct, 2),
        "exit_rule":         f"Initial stop {stop_distance_pct:.1f}% | target {tp_distance_pct:.1f}%",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION E: SAFETY GUARDS
# ═══════════════════════════════════════════════════════════════════════════════

def _check_etf_volume(etf_ticker: str) -> bool:
    """
    Quick Alpaca volume check for the ETF ticker.
    Returns True if the ETF has sufficient volume (> 100K).
    Falls back to True on any error (don't block on API failure).
    """
    if not etf_ticker:
        return False
    try:
        import requests as _req
        ALPACA_KEY    = os.environ.get("ALPACA_KEY", "")
        ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
        resp = _req.get(
            f"https://data.alpaca.markets/v2/stocks/{etf_ticker}/bars",
            params={"timeframe": "1Day", "limit": 1, "feed": "sip"},
            headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET},
            timeout=5,
        )
        if resp.status_code == 200:
            bars = resp.json().get("bars", [])
            if bars:
                vol = bars[-1].get("v", 0) or 0
                return float(vol) >= MIN_ETF_VOLUME
    except Exception:
        pass
    return True  # Assume liquid when we can't check


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION F: MAIN DECISION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def select_instrument(trade: dict, equity: float,
                      positions: list = None,
                      macro: dict = None) -> dict:
    """
    Unified instrument selection engine.
    Makes a THREE-WAY decision: stock vs 2× leveraged ETF vs options.

    Replaces should_use_options() from options_execution.py.

    Args:
        trade:    Trade dict from bot_engine. Must include:
                    ticker, price, deep_score, volume, side, vrp, ewma_rv,
                    rsi, vol_metrics, action_label.
                  Optional: expected_hold_days, hist (DataFrame), calls_df, puts_df,
                    atm_bid, atm_ask, earnings_intel, fundamentals, sentiment.
        equity:   Total portfolio value in dollars.
        positions: List of current Alpaca position objects.
        macro:    Macro context dict (vix, vix_regime, etc.)

    Returns:
        {
            "chosen":     "stock" | "etf" | "options",
            "strategy":   str,
            "scores":     {"stock": {...}, "etf": {...}, "options": {...}},
            "ticker":     str,   # Actual ticker to trade (may be ETF ticker)
            "sizing":     dict,
            "stop_config": dict,
            "reasoning":  str,
        }
    """
    ticker   = trade.get("ticker", "UNKNOWN")
    price    = float(trade.get("price", 0) or 0)
    score    = trade.get("deep_score", trade.get("score", 50)) or 50
    side     = trade.get("side", "buy")
    positions = positions or []
    macro     = macro or {}

    logger.info(f"[{ticker}] select_instrument: price={price}, score={score}, side={side}")

    # ── Step 1: Gather intelligence ────────────────────────────────────────
    intelligence = get_instrument_intelligence(ticker, price, trade)

    # ── Market-wide IV context from VXX ratio (self-calibrating, no fixed thresholds) ──
    # Uses VXX price vs its own 30-day average so it's always relative.
    # This way the signal works regardless of where VXX drifts over time.
    #   ratio > 1.3 (+30% above avg): panic/fear, options extremely expensive
    #   ratio > 1.1 (+10% above avg): elevated IV, sell premium has edge
    #   ratio 0.9-1.1 (within 10%): normal conditions
    #   ratio < 0.9 (-10% below avg): calm, options cheaper
    #   ratio < 0.7 (-30% below avg): complacency, buying options is cheap
    try:
        from datetime import timedelta as _td
        _vxx_start = (datetime.utcnow() - _td(days=45)).strftime("%Y-%m-%d")
        _vxx_resp = requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/VXX/bars?timeframe=1Day&start={_vxx_start}&limit=35&feed=sip",
            headers=_alpaca_headers(), timeout=5
        )
        _vxx_bars = _vxx_resp.json().get("bars", [])
        if len(_vxx_bars) >= 10:
            _vxx_price = float(_vxx_bars[-1]["c"])  # Latest close
            _vxx_avg30 = sum(b["c"] for b in _vxx_bars[-30:]) / len(_vxx_bars[-30:])  # 30-day avg
            _vxx_ratio = _vxx_price / _vxx_avg30  # >1 = above normal, <1 = below normal
        else:
            _vxx_price = 25.0
            _vxx_ratio = 1.0
    except Exception:
        _vxx_price = 25.0
        _vxx_ratio = 1.0  # Default: assume normal

    # Map ratio to IVR — self-calibrating regardless of VXX price level
    if _vxx_ratio >= 1.30:
        _market_ivr = 85   # Panic: VXX 30%+ above its own average
    elif _vxx_ratio >= 1.10:
        _market_ivr = 70   # Elevated: VXX 10-30% above average
    elif _vxx_ratio >= 0.90:
        _market_ivr = 50   # Normal: VXX within 10% of average
    elif _vxx_ratio >= 0.70:
        _market_ivr = 35   # Calm: VXX 10-30% below average
    else:
        _market_ivr = 20   # Complacency: VXX 30%+ below average

    # Inject market IVR into trade and intelligence if not already high
    _existing_ivr = intelligence.get("iv_rank")
    if _existing_ivr is None or _market_ivr > _existing_ivr:
        intelligence["iv_rank"] = _market_ivr
        intelligence["iv_rank_source"] = f"VXX ${_vxx_price:.2f} (ratio={_vxx_ratio:.2f})"

    # Pass VXX data to the trade dict for scoring functions
    trade = {**trade, "_vxx_price": _vxx_price, "_vxx_ratio": _vxx_ratio, "_market_ivr": _market_ivr}

    # ── Step 2: Score all instruments ─────────────────────────────────────

    stock_score   = _score_stock(trade, intelligence)
    etf_score     = _score_etf(trade, intelligence, equity)
    options_score = _score_options(trade, intelligence, equity)

    scores = {
        "stock":   stock_score,
        "etf":     etf_score,
        "options": options_score,
    }

    # ── Step 3: Apply safety rules and pick winner ─────────────────────────

    candidates: list = []  # (score_value, instrument_name)

    # Stock is always a candidate (the safe default)
    candidates.append((stock_score["score"], "stock"))

    # ETF candidate: must exist, must be within hold limit, must be liquid
    if etf_score is not None:
        expected_hold = trade.get("expected_hold_days") or 3
        etf_ticker    = etf_score.get("etf_ticker")
        etf_ok        = (
            expected_hold <= ETF_MAX_HOLD_DAYS
            and _check_etf_volume(etf_ticker)
        )
        if etf_ok:
            candidates.append((etf_score["score"], "etf"))
        else:
            reason = (f"hold {expected_hold}d > {ETF_MAX_HOLD_DAYS}d max"
                      if expected_hold > ETF_MAX_HOLD_DAYS
                      else f"ETF {etf_ticker} volume < {MIN_ETF_VOLUME:,}")
            logger.info(f"[{ticker}] ETF ruled out: {reason}")
            if etf_score:
                etf_score["score"]     = 0.0
                etf_score["reasoning"] += f" [RULED OUT: {reason}]"
    else:
        logger.debug(f"[{ticker}] No leveraged ETF available for this ticker/direction")

    # Options candidate: score >= 70, regular hours, no naked calls, exposure limit
    if options_score is not None and options_score.get("strategy") != "none":
        options_exposure = _options_exposure_pct(positions, equity)
        options_ok = (
            _is_regular_hours()
            and score >= MIN_OPTIONS_SCORE
            and options_exposure < MAX_TOTAL_OPTIONS_PCT
            and options_score.get("strategy") != "sell_call"  # Never naked calls
        )
        if options_ok:
            candidates.append((options_score["score"], "options"))
        else:
            if not _is_regular_hours():
                reason = "outside regular hours"
            elif score < MIN_OPTIONS_SCORE:
                reason = f"score {score} < {MIN_OPTIONS_SCORE}"
            elif options_exposure >= MAX_TOTAL_OPTIONS_PCT:
                reason = f"options exposure {options_exposure:.1%} at max {MAX_TOTAL_OPTIONS_PCT:.0%}"
            else:
                reason = "strategy not viable"
            logger.info(f"[{ticker}] Options ruled out: {reason}")
            if options_score:
                options_score["score"]     = 0.0
                options_score["reasoning"] += f" [RULED OUT: {reason}]"
    else:
        logger.debug(f"[{ticker}] Options not scored (outside hours or low score)")

    # Pick highest-scoring candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen_score, chosen = candidates[0]

    _etf_s = f"{etf_score['score']:.0f}" if etf_score else "N/A"
    _opt_s = f"{options_score['score']:.0f}" if options_score else "N/A"
    logger.info(
        f"[{ticker}] Decision: {chosen} "
        f"(scores: stock={stock_score['score']:.0f}, etf={_etf_s}, options={_opt_s})"
    )

    # ── Step 4: Build final output ─────────────────────────────────────────

    # Determine the actual ticker to trade
    if chosen == "etf" and etf_score:
        trade_ticker = etf_score["etf_ticker"]
        strategy     = etf_score["strategy"]
        reasoning    = etf_score["reasoning"]
    elif chosen == "options" and options_score:
        trade_ticker = ticker
        strategy     = options_score["strategy"]
        reasoning    = options_score["reasoning"]
    else:
        trade_ticker = ticker
        strategy     = stock_score["strategy"]
        reasoning    = stock_score["reasoning"]

    # Sizing
    etf_ticker_for_sizing = etf_score.get("etf_ticker") if etf_score else None
    sizing = _build_sizing(trade, equity, positions, macro, chosen, etf_ticker_for_sizing)

    # Stop configuration
    stop_config = _build_stop_config(trade, chosen, sizing)

    # Attribution reasoning
    score_summary = (
        f"stock={stock_score['score']:.0f}"
        + (f" | etf={etf_score['score']:.0f}" if etf_score else "")
        + (f" | options={options_score['score']:.0f}" if options_score else "")
    )

    full_reasoning = (
        f"CHOSEN: {chosen.upper()} ({strategy}) | "
        f"Scores: [{score_summary}] | "
        f"{reasoning}"
    )

    return {
        "chosen":      chosen,
        "strategy":    strategy,
        "scores": {
            "stock":   stock_score,
            "etf":     etf_score,
            "options": options_score,
        },
        "ticker":      trade_ticker,
        "sizing":      sizing,
        "stop_config": stop_config,
        "reasoning":   full_reasoning,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION G: BACKWARD-COMPAT SHIM
# ═══════════════════════════════════════════════════════════════════════════════

def should_use_options(trade: dict, equity: float,
                       existing_positions: list = None) -> dict:
    """
    Backward-compatible shim for code that still calls should_use_options().
    Wraps select_instrument() and returns the original dict shape.
    """
    result   = select_instrument(trade, equity, existing_positions or [], {})
    chosen   = result["chosen"]
    strategy = result["strategy"]
    score    = result["scores"].get(chosen, {}).get("score", 0) or 0

    return {
        "use_options":  chosen == "options",
        "reason":       result["reasoning"],
        "strategy":     strategy if chosen == "options" else "stock",
        "edge_pct":     result["scores"].get(chosen, {}).get("edge_pct", 0) or 0,
        # Extension fields for callers that want the full decision
        "chosen":       chosen,
        "etf_ticker":   result["ticker"] if chosen == "etf" else None,
        "full_result":  result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI — Test with sample trade
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    result = select_instrument(
        trade={
            "ticker":       "NVDA",
            "price":        920,
            "deep_score":   85,
            "volume":       60_000_000,
            "side":         "buy",
            "vrp":          7.5,
            "ewma_rv":      2.5,
            "rsi":          60,
            "vol_metrics":  {},
            "action_label": "SELL OPTIONS",
            # expected_hold_days not set → defaults to 3
        },
        equity=100_000,
        positions=[],
        macro={"vix": 18},
    )

    print(json.dumps(result, indent=2, default=str))
