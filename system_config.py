#!/usr/bin/env python3
"""
VolTradeAI — System Configuration
===================================
Every parameter that was previously hardcoded is here as an adaptive variable.
Values change based on market regime (VXX ratio), time of day, and account size.

DESIGN PRINCIPLE: No magic numbers anywhere else in the codebase.
If you're tempted to hardcode a number, add it here with a comment explaining
what research or backtest result determined the value.

Usage:
    from system_config import cfg, get_adaptive_params
    params = get_adaptive_params(vxx_ratio=1.15, spy_trend="bull", time_of_day="open")
"""

import os
import json
import time
import math
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Base Configuration (research-backed defaults) ─────────────────────────────
# All values here are starting points. get_adaptive_params() returns regime-adjusted values.

BASE_CONFIG = {

    # ── TIMING ────────────────────────────────────────────────────────────────
    # Tier 1 (stop monitoring): every 30 seconds
    # Research: ATR stops need near-real-time. 45s was arbitrary — 30s is better.
    "TIER1_INTERVAL_MS": 30_000,

    # Tier 2 scan intervals by time of day (ms)
    # Research: Gao et al 2018 — 40% of daily volume in first 30min. Scan more aggressively.
    "TIER2_OPEN_MS":        60_000,   # 9:30-10:30 ET: every 60 seconds
    "TIER2_MIDMORNING_MS":  180_000,  # 10:30-12:00 ET: every 3 minutes
    "TIER2_MIDDAY_MS":      420_000,  # 12:00-14:00 ET: every 7 minutes
    "TIER2_POWER_HOUR_MS":  120_000,  # 14:00-16:00 ET: every 2 minutes (was 3min)
    "TIER2_AFTERHOURS_MS":  900_000,  # After close: every 15 minutes
    "TIER2_PREMARKET_MS":   300_000,  # Pre-market: every 5 minutes (was 10min)
    "TIER2_OVERNIGHT_MS":   1_800_000, # Overnight: every 30 minutes

    # Tier 3 strategic interval
    "TIER3_INTERVAL_MS": 3_600_000,  # 1 hour

    # ── STREAMING SIGNAL DETECTION ────────────────────────────────────────────
    # Volume spike threshold for real-time signal detection (Tier 0 WebSocket)
    # Research: Gervais et al 2001 — volume 2-3x average is statistically significant
    "STREAM_VOL_SPIKE_RATIO":  2.5,   # 2.5x average = signal
    "STREAM_BAR_CHANGE_PCT":   0.8,   # 0.8% move in a 1-min bar = directional
    "STREAM_MOMENTUM_PCT":     0.3,   # 5-bar trend confirmation
    "STREAM_COOLDOWN_MS":      600_000, # 10 minutes between signals per ticker

    # ── SIGNAL QUALITY ────────────────────────────────────────────────────────
    # v1.0.23: score=63 optimized (was 65). 324-combo backtest shows 63 adds
    # more true positives than false positives in bull/neutral regimes.
    # Bear/panic regimes use their own higher thresholds (75+) regardless.
    "MIN_SCORE":            63,    # Optimized: 63 (was 65)
    "SCORE_BAND_MAX":       75,    # Scores above this are often fake breakouts
    "SCORE_BAND_OPTIMAL_LO": 65,   # Sweet spot confirmed by 10-year backtest
    "SCORE_BAND_OPTIMAL_HI": 74,

    # ── STOCK QUALITY FILTERS ─────────────────────────────────────────────────
    # 3-year backtest: penny stocks (<$5) = 25% WR. Volume 500K = liquidity minimum.
    "MIN_PRICE":            5.0,   # Minimum stock price
    "MIN_VOLUME":           500_000,  # Minimum daily volume
    "OPEN_HOUR_MIN_VOLUME": 1_000_000, # First 30min: need even more volume
    "MAX_CHANGE_PCT":       35.0,  # Skip stocks already up/down 35%+ (easy money gone)
    "MAX_SPREAD_PCT":       0.5,   # Skip if bid-ask spread > 0.5% (execution risk)

    # ── POSITION SIZING ────────────────────────────────────────────────────────
    # Kelly criterion determines optimal size. These are the hard limits.
    "MAX_POSITION_PCT":     0.15,  # Never more than 15% in one position
    "MIN_POSITION_PCT":     0.04,  # Never less than 4% (not worth friction)
    "MAX_TOTAL_EXPOSURE":   0.80,  # Never more than 80% of portfolio invested
    "MAX_TOTAL_CAPITAL_PCT": 1.00,  # Never deploy more than 100% of equity across all components
    "MAX_SECTOR_POSITIONS": 2,     # Max 2 from the same sector
    "MAX_POSITIONS":        6,     # Max total open positions
    "MAX_OPTIONS_PCT":      0.10,  # Max 10% per options position
    "OPTIONS_SCALE":        2.0,   # v1.0.23 optimized: 2x options sizing (was 1x)

    # ── PASSIVE SPY FLOOR (v1.0.29) ────────────────────────────────────
    # Problem: in calm bull years (2017, 2019, 2023) momentum signals don't work.
    # The bot sat in cash while SPY gained 20-30%. Quiet markets = noise, not signal.
    # Fix: hold a passive SPY allocation in calm regimes to capture the market drift.
    # This is standard "beta + alpha" — passive floor + active overlay.
    #
    # 10-year backtest (8-config sweep):
    #   QQQ floor + VRP30/Sector25 = best: 18.7% CAGR vs SPY 12.3% (beats by +6.4%)
    #   QQQ outperforms SPY by ~5.6%/yr — captures tech/growth outperformance
    #   $100K → $578K in 10yr (SPY: $327K). Beats SPY in 7/11 years.
    #   2017: -27.7% → +20.2% | 2019: -32.5% → +49.1% | 2023: -46.7% → +52.4%
    "FLOOR_TICKER":         "QQQ", # QQQ instead of SPY — 5.6%/yr more return
    "SPY_FLOOR_BULL":       0.70,  # 70% passive QQQ in BULL
    "SPY_FLOOR_NEUTRAL":    0.90,  # 90% passive QQQ in NEUTRAL (signals = noise here)
    "SPY_FLOOR_CAUTION":    0.35,  # 35% passive QQQ in CAUTION
    "SPY_FLOOR_BEAR":       0.00,  # 0% in BEAR (third leg takes over)
    "SPY_FLOOR_PANIC":      0.00,  # 0% in PANIC (all defensive)

    # ── THIRD LEG (v1.0.25) ────────────────────────────────────────────
    # 64-combo backtest winner: VRP=15% + Sector=12%, TLT=0%
    # Result: CAGR +14.8%/yr (vs 2-leg +13.8%, SPY +12.3%)
    # Max DD improved: 74.4% -> 72.9%
    # ALL 64 combinations beat SPY -- third leg is additive regardless of sizing
    #
    # Leg 3A: TLT bond rotation -- disabled (0%) by backtest
    #   TLT hurt in 2022 (rate hikes crushed bonds at same time as stocks)
    #   TLT helped in 2020 but not enough to offset 2022 drag
    "LEG3_TLT_PCT":        0.00,  # % of equity into TLT in BEAR regime (0=disabled)
    #
    # Leg 3B: VRP Harvest -- 15% is the sweet spot
    #   Sell volatility when VXX ratio 1.05-1.25 (elevated but declining)
    #   Earns when fear bleeds out -- exactly the post-crash recovery window
    "LEG3_VRP_PCT":        0.40,  # % of equity for VRP harvest (regime-adaptive sweep: 40%)
    #
    # Leg 3C: Sector Rotation -- 12% into XOM+LMT in BEAR/CAUTION
    #   XOM: -0.04 corr to SPY, +45% last 6mo, earns in geopolitical stress
    #   LMT: +0.12 corr, defense spending immune to tariff/rate cycles
    # Leg 3C: Regime-adaptive sector rotation (v1.0.30)
    #   During CRASHES (PANIC/BEAR): hold GLD (gold) — +0.122%/day in bear, near-zero SPY corr
    #   During RECOVERY (CAUTION): hold XOM+LMT — strong cyclical bounce-back
    #   Backtest: 19.8% CAGR, beats SPY by +7.5% (vs 18.4% with fixed XOM+LMT)
    # Crash: GLD (gold ETF) — goes UP during crashes (+0.122%/day, near-zero SPY corr)
    # Recovery: ITA (iShares US Aerospace & Defense ETF, 35 companies) — strong bounce-back
    #   Why ETFs not stocks: single stock risk (bad quarter, buyout, split) eliminated
    #   ITA holds LMT, RTX, NOC, GD, BA, HII, etc. — diversified defense exposure
    #   Backtest: ITA beats XOM+LMT combo (20.3% vs 20.0% CAGR, 0.911 vs 0.906 Sharpe)
    "LEG3_CRASH_ASSETS":   [("GLD", 0.15)],      # Gold ETF 15% during PANIC/BEAR
    "LEG3_RECOVERY_ASSETS": [("ITA", 0.20)],      # Defense ETF 20% during CAUTION

    # ── STOP LOSS & TAKE PROFIT ────────────────────────────────────────────────
    # v1.0.23 optimized values (324-combo backtest, best risk-adjusted vs SPY):
    # tp=12%, sl=6%, hold=10d, score=63, bull_size=15%, opts=2x
    # Result: CAGR +13.8%/yr vs SPY +12.3%/yr, Sharpe 1.22, only 2 years worse than -20%
    # Optimization ran 324 combinations; this is the most consistent winner.
    "ATR_STOP_MULTIPLIER":  1.5,   # 1.5x 10-day ATR
    "MIN_STOP_PCT":         2.0,   # Never tighter than 2%
    "MAX_STOP_PCT":         6.0,   # Optimized: 6% stop (was 8%) — tighter = faster capital recycle
    "MIN_TP_PCT":           12.0,  # Optimized: 12% TP (was 8%) — let winners run further
    "MAX_TP_PCT":           25.0,  # Maximum TP cap
    "ATR_TP_MULTIPLIER":    4.0,   # TP = 4x stop distance (was 3x) — 2:1 min R:R
    "TIME_STOP_DAYS":       10,    # Optimized: 10-day hold (was 7) — more time to work

    # ── ML MODEL ──────────────────────────────────────────────────────────────
    # 31 features: 25 original + 6 new (cross_sec_rank, earnings_surprise,
    # put_call_proxy, vol_of_vol, frac_diff_price, idiosyncratic_ret)
    "ML_FEATURE_COUNT":     31,
    "ML_RETRAIN_TRADES":    20,    # Retrain after every 20 real trades
    "ML_MIN_SAMPLES":       300,   # Minimum samples to train (don't train on < 300)
    "ML_LOOKBACK_DAYS":     60,    # Training window: 60 days of history
    "ML_TARGET_HORIZON":    5,     # Predict outcome 5 trading days ahead
    "ML_TARGET_RETURN":     2.0,   # Label = 1 if stock returns 2%+ in horizon
    "ML_CONFIDENCE_THRESH": 0.60,  # Minimum ML confidence to trade
    "ML_MAX_AGE_HOURS":     24,    # Retrain if model is older than 24 hours

    # ── MARKET REGIME ─────────────────────────────────────────────────────────
    # Regime detection uses VXX ratio + SPY MA + Markov state
    "REGIME_SPY_MA_PERIOD":   50,   # 50-day MA for trend
    "REGIME_BEAR_THRESHOLD":  0.96, # SPY < 96% of 50-day MA = bearish
    "REGIME_VXX_BULL":         0.90, # VXX below 90% of avg = bull confirmed (no Markov needed)
    "REGIME_VXX_CAUTION":     1.05, # VXX 5%+ above avg = caution zone
    "REGIME_VXX_ELEVATED":    1.15, # VXX 15%+ above 30-day avg = elevated fear (was 1.10)
    "REGIME_VXX_PANIC":       1.30, # VXX 30%+ above avg = panic mode
    "MARKOV_STATES":          3,    # Bull / Neutral / Bear
    "MARKOV_LOOKBACK_DAYS":   3,    # Order-3 Markov (last 3 days)

    # ── RISK MANAGEMENT ────────────────────────────────────────────────────────
    "DAILY_LOSS_LIMIT_PCT":   5.0,  # Halt trading if down 5% on the day
    "DRAWDOWN_HALT_PCT":      15.0, # Halt if drawdown from peak > 15%
    "STOP_COOLDOWN_SECONDS":  7200, # 2 hours: no re-entry after a stop fires
    "CIRCUIT_BREAKER_STOPS":  3,    # 3 consecutive stops = circuit breaker
    "CIRCUIT_BREAKER_HOURS":  1,    # Pause 1 hour after circuit breaker

    # ── BLOCKED TICKERS (backtest-confirmed underperformers) ──────────────────
    # 3-year backtest: these had 20-25% WR with consistent negative EV
    "BLOCKED_TICKERS": [
        "DKNG", "RBLX",                      # Gaming: 25% WR
        "SQQQ", "TQQQ", "SPXU", "UPRO",     # 3x leveraged ETFs: 22% WR
        "UVXY",                               # VIX products: decay kills them
        # NOTE: SVXY deliberately excluded from BLOCKED_TICKERS.
        # SVXY is used by the VRP harvest leg (Leg 3B) as the primary
        # instrument for selling volatility premium. Blocking it would
        # prevent the third leg from executing its core strategy.
        "ABNB", "DASH",                       # Travel: 20% WR
        "LYFT",                               # Ride-share: structural drag
    ],

    # ── SLIPPAGE & COSTS ──────────────────────────────────────────────────────
    # Conservative estimates for realistic backtest
    "SLIPPAGE_PCT":      0.0005,  # 0.05% per side (liquid large caps)
    "SLIPPAGE_ILLIQUID": 0.002,   # 0.2% for names under 1M daily volume
    "OPTIONS_SLIPPAGE":  0.01,    # 1% on options mid price (wide spreads)

    # ── CACHE TTLs ────────────────────────────────────────────────────────────
    "CACHE_YF_SECONDS":     300,    # yfinance data: 5 min cache
    "CACHE_INTEL_SECONDS":  300,    # Intelligence data: 5 min cache
    "CACHE_MACRO_SECONDS":  300,    # Macro snapshot: 5 min cache
    "CACHE_SOCIAL_HOURS":   2,      # Social/Reddit: 2 hour cache
    "CACHE_ALT_DATA_HOURS": 4,      # Alt data: 4 hour cache
    "CACHE_INSIDER_HOURS":  1,      # Insider: 1 hour cache

    # ── SCAN UNIVERSE SIZE ────────────────────────────────────────────────────
    "SCREENER_TOP_N":       100,    # Top N from most-actives
    "SCREENER_MOVERS_N":    50,     # Top N movers
    "DEEP_SCORE_TOP_N":     20,     # Deep-analyze top N candidates
    "DEEP_SCORE_WORKERS":   8,      # Parallel workers for deep scoring
    "STREAM_TICKERS_N":     60,     # Tickers to subscribe in streaming feed
}

# ── Adaptive Parameter Logic ──────────────────────────────────────────────────

def get_market_regime(vxx_ratio: float, spy_vs_ma50: float,
                      markov_state: int = 1,
                      spy_below_200_days: int = 0,
                      spy_above_200d: bool = True) -> str:
    """
    Classify market regime from 0 (panic/bear) to 4 (euphoria/bull).
    vxx_ratio: VXX / 30-day average (>1 = fear)
    spy_vs_ma50: SPY / 50-day MA (>1 = above MA)
    markov_state: 0=bear, 1=neutral, 2=bull
    spy_below_200_days: consecutive trading days SPY has closed below its
        200-day MA. When >= 10, forces BEAR regardless of VXX level.
        This catches slow grinding bear markets (e.g. 2022) where VXX
        rises with its own 30d avg so the ratio stays near 1.0.
        Backtest result: +2.9% CAGR, no false positives over 10 years.
    """
    # Fix B: 200-day MA slow-bear detector (v1.0.22)
    # If SPY has been below its 200d MA for 10+ consecutive days, force BEAR.
    # VXX panic can still upgrade to PANIC from here.
    if spy_below_200_days >= 10:
        if vxx_ratio >= BASE_CONFIG["REGIME_VXX_PANIC"]:
            return "PANIC"  # Full panic on top of slow bear
        return "BEAR"       # Slow bear — block new stock longs

    if vxx_ratio >= BASE_CONFIG["REGIME_VXX_PANIC"] or spy_vs_ma50 < 0.94:
        return "PANIC"      # VXX >30% above avg OR SPY >6% below 50d MA
    elif vxx_ratio >= BASE_CONFIG["REGIME_VXX_ELEVATED"] or spy_vs_ma50 < BASE_CONFIG["REGIME_BEAR_THRESHOLD"]:
        return "BEAR"       # VXX >15% above avg OR SPY below 96% of 50d MA
    elif vxx_ratio >= BASE_CONFIG["REGIME_VXX_CAUTION"]:
        return "CAUTION"    # VXX 5-15% above avg — elevated but not BEAR
    elif vxx_ratio <= 0.90 and spy_vs_ma50 > 1.01:
        return "BULL"       # VXX below 90% of avg AND SPY above MA
    # v1.0.29: Wider BULL — SPY above 200d MA + VXX calm = BULL
    # Previously this was NEUTRAL, causing bot to miss quiet bull markets.
    # Backtest: helps regime correctly classify 2017, 2019, 2023 bull years.
    elif spy_above_200d and vxx_ratio <= 1.02:
        return "BULL"       # Calm + uptrend = BULL (wider detection)
    elif spy_vs_ma50 > 1.0 and markov_state >= 1:
        return "NEUTRAL"    # Standard conditions
    else:
        return "CAUTION"    # SPY below MA but VXX not elevated — wait for clarity


def get_adaptive_params(
    vxx_ratio: float = 1.0,
    spy_vs_ma50: float = 1.0,
    markov_state: int = 1,
    time_of_day: str = "regular",   # "open" / "regular" / "power_hour" / "afterhours"
    account_equity: float = 100_000,
    spy_below_200_days: int = 0,    # Fix B: consecutive days SPY below 200d MA
) -> dict:
    """
    Return the full adaptive parameter set for current conditions.
    All callers should use this function, not hardcoded values.
    """
    regime = get_market_regime(vxx_ratio, spy_vs_ma50, markov_state, spy_below_200_days)
    p = BASE_CONFIG.copy()

    # ── Regime adjustments ────────────────────────────────────────────────────
    if regime == "PANIC":
        # No new stock longs in PANIC. Options engine takes over (VXX spike = prime time for puts).
        p["MAX_POSITIONS"]          = 0      # No new stock longs in PANIC
        p["MAX_POSITION_PCT"]       = 0.06
        p["MIN_SCORE"]              = 75     # Would need to be extremely high conviction
        p["MAX_STOP_PCT"]           = 5.0    # Tighter stops in panic
        p["MAX_TOTAL_EXPOSURE"]     = 0.30   # 70% cash in panic
        p["STREAM_VOL_SPIKE_RATIO"] = 3.0    # Higher bar for signals
        p["TIME_STOP_DAYS"]         = 3      # Exit faster in panic
        p["regime"]                 = "PANIC"

    elif regime == "BEAR":
        # Research: Faber 2007 — go to cash when market in downtrend (SPY < 10mo MA).
        # Backtest result: blocking new longs in BEAR +1.4% total return.
        # We still hold existing positions (don't panic-sell), just don't open new ones.
        # Options engine continues running — BEAR is prime time for premium selling.
        p["MAX_POSITIONS"]          = 0      # No new stock longs in BEAR
        p["MAX_POSITION_PCT"]       = 0.08
        p["MIN_SCORE"]              = 75     # Very high bar if we do enter
        p["MAX_TOTAL_EXPOSURE"]     = 0.50
        p["STREAM_VOL_SPIKE_RATIO"] = 2.8
        p["TIME_STOP_DAYS"]         = 5
        p["regime"]                 = "BEAR"

    elif regime == "BULL":
        # Aggressive: full size, momentum longs, buy calls
        # NOTE: Stop/TP/time-stop improvements looked good in backtest but
        # those results were inflated by survivorship bias (36 pre-selected
        # winners). Keeping conservative values until live data confirms edge.
        p["MAX_POSITIONS"]          = 8
        p["MAX_POSITION_PCT"]       = 0.15
        p["MIN_SCORE"]              = 63     # More setups allowed in bull
        p["MAX_TOTAL_EXPOSURE"]     = 0.90   # Near fully invested
        p["STREAM_VOL_SPIKE_RATIO"] = 2.2    # Lower bar — more signals
        p["ATR_STOP_MULTIPLIER"]    = 2.0    # Standard stop multiplier
        p["TIME_STOP_DAYS"]         = 10     # Standard hold period
        p["regime"]                 = "BULL"

    elif regime == "NEUTRAL":
        # v1.0.29: NO active stock trades in NEUTRAL.
        # 10-year backtest: momentum signals are noise in calm markets.
        # Passive SPY floor (85%) captures the market drift instead.
        # Active trades in NEUTRAL had net negative P&L over 10 years.
        p["MAX_POSITIONS"]          = 0      # No stock trades in NEUTRAL
        p["MAX_TOTAL_EXPOSURE"]     = 0.90   # SPY floor handles exposure
        p["regime"] = "NEUTRAL"

    else:  # CAUTION
        p["MAX_POSITIONS"]          = 4
        p["MAX_POSITION_PCT"]       = 0.10
        p["MIN_SCORE"]              = 67
        p["MAX_TOTAL_EXPOSURE"]     = 0.60
        p["regime"]                 = "CAUTION"

    # ── Time-of-day adjustments ────────────────────────────────────────────────
    if time_of_day == "open":
        p["TIER2_INTERVAL_MS"]      = p["TIER2_OPEN_MS"]
        p["MIN_VOLUME"]             = p["OPEN_HOUR_MIN_VOLUME"]  # 1M at open
    elif time_of_day == "power_hour":
        p["TIER2_INTERVAL_MS"]      = p["TIER2_POWER_HOUR_MS"]
    elif time_of_day == "afterhours":
        p["TIER2_INTERVAL_MS"]      = p["TIER2_AFTERHOURS_MS"]
    elif time_of_day == "premarket":
        p["TIER2_INTERVAL_MS"]      = p["TIER2_PREMARKET_MS"]
    elif time_of_day == "overnight":
        p["TIER2_INTERVAL_MS"]      = p["TIER2_OVERNIGHT_MS"]
    else:
        p["TIER2_INTERVAL_MS"]      = p["TIER2_MIDMORNING_MS"]

    # ── Account size adjustments ──────────────────────────────────────────────
    # Larger accounts need more diversification, smaller can concentrate
    if account_equity < 25_000:
        # PDT rule: max 3 day trades per 5 days. Fewer positions needed.
        p["MAX_POSITIONS"]      = min(p["MAX_POSITIONS"], 3)
        p["MAX_POSITION_PCT"]   = min(p["MAX_POSITION_PCT"], 0.30)
    elif account_equity > 500_000:
        # Scale down per-position % to avoid market impact
        p["MAX_POSITION_PCT"]   = min(p["MAX_POSITION_PCT"], 0.08)
        p["MAX_POSITIONS"]      = min(p["MAX_POSITIONS"] + 2, 12)

    return p


def load_config_overrides() -> dict:
    """Load any manual overrides from /data/voltrade/config_overrides.json"""
    override_path = os.path.join(DATA_DIR, "config_overrides.json")
    try:
        if os.path.exists(override_path):
            with open(override_path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# Default config (used when regime data isn't available)
cfg = BASE_CONFIG.copy()
cfg.update(load_config_overrides())
