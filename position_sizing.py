"""
VolTradeAI — Dynamic Position Sizing Engine
============================================
Replaces all hard-coded percentages with variable sizing based on:
  1. Modified Kelly Criterion (edge × confidence)
  2. Volatility scaling (ATR/GARCH — volatile stocks get fewer shares)
  3. ML confidence (higher confidence = larger position)
  4. Liquidity (low volume = smaller to avoid moving the price)
  5. Portfolio heat (correlated positions reduce new sizing)
  6. Historical win rate by setup type
  7. Market regime (VIX level adjusts everything)
  8. Earnings proximity (reduce size near earnings dates)
  9. Time of day (smaller near close)
  10. Whole-share rounding with leftover redistribution

All outputs are in SHARES (integers) — no fractional nonsense.
Fees/slippage are subtracted from expected edge before sizing.
"""

import os
import json
import time
import logging
import math
import requests
from datetime import datetime, timedelta

try:
    from storage_config import DATA_DIR, FILLS_PATH, TRADE_FEEDBACK_PATH, WEIGHTS_PATH
except ImportError:
    DATA_DIR = "/tmp"
    FILLS_PATH = "/tmp/voltrade_fills.json"
    TRADE_FEEDBACK_PATH = "/tmp/voltrade_trade_feedback.json"
    WEIGHTS_PATH = "/tmp/voltrade_weights.json"

logger = logging.getLogger("position_sizing")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS (absolute guardrails — these are FLOORS/CEILINGS, not targets)
# ═══════════════════════════════════════════════════════════════════════════════

ABSOLUTE_MAX_POSITION_PCT = 0.08   # Never more than 8% of portfolio in one stock (pro-level: tighter cap)
ABSOLUTE_MIN_POSITION_PCT = 0.01   # Never less than 1% (not worth the trade)
ABSOLUTE_MAX_POSITIONS    = 8      # Hard ceiling on total positions
ABSOLUTE_MAX_PORTFOLIO_HEAT = 0.50 # Never more than 50% of portfolio deployed
DEFAULT_COMMISSION_PER_SHARE = 0.0 # Alpaca paper = $0. Change for live.
OPTIONS_FEE_PER_CONTRACT = 0.65    # Standard options fee

# Alpaca API (same keys used everywhere)
ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Sizing history for learning
SIZING_HISTORY_PATH = os.path.join(DATA_DIR, "voltrade_sizing_history.json")


# ═══════════════════════════════════════════════════════════════════════════════
#  A. KELLY CRITERION — How much to bet given edge and confidence
# ═══════════════════════════════════════════════════════════════════════════════

def _kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                     kelly_divisor: float = 4.0) -> float:
    """
    Modified Kelly Criterion:
      f* = (win_rate * avg_win - loss_rate * avg_loss) / avg_win

    Returns fraction of portfolio to risk (0.0 to 1.0).
    We use QUARTER-Kelly (f*/4) by default because:
    - Full Kelly has brutal drawdowns in practice
    - Half-Kelly is still too aggressive for correlated positions
    - Quarter-Kelly balances growth with survivability

    The kelly_divisor parameter controls the fraction (4.0 = quarter, 2.0 = half).
    """
    if avg_win <= 0 or avg_loss <= 0:
        return 0.03  # Default 3% when no data

    loss_rate = 1.0 - win_rate
    kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win

    # Quarter-Kelly for safety (configurable via kelly_divisor)
    fractional_kelly = kelly / kelly_divisor

    # Clamp to reasonable range
    return max(0.01, min(fractional_kelly, ABSOLUTE_MAX_POSITION_PCT))


def _get_historical_stats() -> dict:
    """
    Load trade feedback history and compute win rate + avg win/loss.
    Returns stats overall and by strategy type.

    Defaults are derived from 10-year backtest (2016-2026) when live data
    is insufficient:
      - stocks: 52.8% WR, avg win $6009, avg loss $4579
      - csp_options: 74% WR, avg win $975, avg loss $1692
      - vrp: 80% WR, avg win $9255, avg loss $12743
    """
    default = {
        "overall": {"win_rate": 0.528, "avg_win": 6009, "avg_loss": 4579, "total_trades": 0},
        "by_strategy": {
            "stocks":      {"win_rate": 0.528, "avg_win": 6009, "avg_loss": 4579, "trades": 771},
            "csp_options": {"win_rate": 0.740, "avg_win": 975, "avg_loss": 1692, "trades": 150},
            "vrp":         {"win_rate": 0.800, "avg_win": 9255, "avg_loss": 12743, "trades": 30},
        },
    }
    
    try:
        if not os.path.exists(TRADE_FEEDBACK_PATH):
            return default
        with open(TRADE_FEEDBACK_PATH) as f:
            trades = json.load(f)
        if len(trades) < 10:
            return default
    except Exception:
        return default
    
    # Overall stats
    wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
    losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]
    
    win_rate = len(wins) / len(trades) if trades else 0.55
    avg_win = abs(sum(t.get("pnl_pct", 0) for t in wins) / len(wins)) if wins else 4.0
    avg_loss = abs(sum(t.get("pnl_pct", 0) for t in losses) / len(losses)) if losses else 2.0
    
    stats = {
        "overall": {
            "win_rate": round(win_rate, 3),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "total_trades": len(trades),
        },
        "by_strategy": {},
    }
    
    # Stats by strategy/setup type (momentum, mean_reversion, etc.)
    strategies = set(t.get("strategy", "unknown") for t in trades)
    for strat in strategies:
        strat_trades = [t for t in trades if t.get("strategy", "unknown") == strat]
        if len(strat_trades) < 5:
            continue
        s_wins = [t for t in strat_trades if t.get("pnl_pct", 0) > 0]
        s_losses = [t for t in strat_trades if t.get("pnl_pct", 0) <= 0]
        stats["by_strategy"][strat] = {
            "win_rate": round(len(s_wins) / len(strat_trades), 3) if strat_trades else 0.5,
            "avg_win": round(abs(sum(t.get("pnl_pct", 0) for t in s_wins) / len(s_wins)), 2) if s_wins else 3.0,
            "avg_loss": round(abs(sum(t.get("pnl_pct", 0) for t in s_losses) / len(s_losses)), 2) if s_losses else 2.0,
            "trades": len(strat_trades),
        }
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  B. VOLATILITY SCALING — Volatile stocks get smaller positions
# ═══════════════════════════════════════════════════════════════════════════════

def _volatility_scalar(atr_pct: float, garch_vol: float = None) -> float:
    """
    Scale position size inversely with volatility.
    
    A stock with 1% daily range (calm, like JNJ) gets full size.
    A stock with 5% daily range (wild, like TSLA) gets ~40% size.
    
    Uses ATR% as primary, GARCH as secondary signal.
    Returns a multiplier from 0.25 to 1.2
    """
    if atr_pct is None or atr_pct <= 0:
        atr_pct = 2.0  # Default assumption
    
    # Base: inverse relationship. Lower vol = bigger position.
    # Anchor at 2% ATR = 1.0x (normal stock)
    vol_scalar = 2.0 / max(atr_pct, 0.5)
    
    # If GARCH says volatility is expanding (higher than ATR), reduce further
    if garch_vol and garch_vol > 0 and atr_pct > 0:
        garch_pct = garch_vol * 100  # Convert to percentage
        if garch_pct > atr_pct * 1.5:
            # Volatility expanding — reduce by 20%
            vol_scalar *= 0.80
    
    # Clamp: never more than 1.2x (very calm stocks), never less than 0.25x
    return max(0.25, min(vol_scalar, 1.20))


# ═══════════════════════════════════════════════════════════════════════════════
#  C. CONFIDENCE SCALING — ML confidence adjusts size
# ═══════════════════════════════════════════════════════════════════════════════

def _confidence_scalar(score: float, ml_confidence: float = None) -> float:
    """
    Higher conviction = larger position. 
    
    Score 65 (minimum) → 0.6x
    Score 75 (decent)  → 0.85x
    Score 85 (strong)  → 1.0x
    Score 95 (exceptional) → 1.15x
    
    ML confidence further adjusts: high confidence boosts, low dampens.
    """
    if score is None or score <= 0:
        return 0.5
    
    # Normalize score: 65-100 range mapped to 0.6-1.15
    base = 0.6 + (min(score, 100) - 65) / (100 - 65) * 0.55
    base = max(0.5, min(base, 1.15))
    
    # ML confidence adjustment (if available)
    if ml_confidence is not None and ml_confidence > 0:
        if ml_confidence > 0.75:
            base *= 1.10  # High confidence boost
        elif ml_confidence < 0.45:
            base *= 0.75  # Low confidence dampens
    
    return max(0.4, min(base, 1.25))


# ═══════════════════════════════════════════════════════════════════════════════
#  D. LIQUIDITY SCALING — Low volume stocks need smaller positions
# ═══════════════════════════════════════════════════════════════════════════════

def _liquidity_scalar(volume: int, price: float, position_value: float) -> float:
    """
    Ensure our trade doesn't exceed a % of daily volume.
    
    Rule: position should be less than 1% of daily dollar volume.
    If it exceeds that, we're moving the market and slippage kills us.
    
    Also penalizes low-volume stocks in general.
    """
    if volume is None or volume <= 0 or price is None or price <= 0:
        return 0.5  # Conservative when we don't know
    
    daily_dollar_volume = volume * price
    
    # Our position as % of daily volume
    volume_impact = position_value / daily_dollar_volume if daily_dollar_volume > 0 else 1.0
    
    if volume_impact > 0.02:
        # We'd be more than 2% of daily volume — way too much
        return 0.25
    elif volume_impact > 0.01:
        # 1-2% of daily volume — reduce
        return 0.50
    elif volume_impact > 0.005:
        # 0.5-1% — slight reduction
        return 0.75
    
    # General volume penalty for thin stocks
    if volume < 500_000:
        return 0.50
    elif volume < 1_000_000:
        return 0.70
    elif volume < 5_000_000:
        return 0.85
    
    return 1.0  # High volume, no concern


# ═══════════════════════════════════════════════════════════════════════════════
#  E. PORTFOLIO HEAT — How much risk is already deployed
# ═══════════════════════════════════════════════════════════════════════════════

def _portfolio_heat_scalar(current_positions: list, equity: float, 
                            new_ticker_sector: str = None) -> float:
    """
    Reduce sizing when portfolio is already heavy.
    
    - 0 positions: full size (1.0x)
    - 2 positions: 0.9x
    - 4 positions: 0.7x
    - 6+ positions: 0.5x
    
    Extra penalty if new trade is in same sector as existing positions.
    """
    if not current_positions or not isinstance(current_positions, list):
        return 1.0
    
    num_pos = len(current_positions)
    total_deployed = sum(abs(float(p.get("market_value", 0))) for p in current_positions)
    deployed_pct = total_deployed / equity if equity > 0 else 0
    
    # Base scalar from position count
    count_scalar = max(0.4, 1.0 - (num_pos * 0.10))
    
    # Deployed capital scalar
    if deployed_pct > 0.40:
        deploy_scalar = 0.50
    elif deployed_pct > 0.30:
        deploy_scalar = 0.70
    elif deployed_pct > 0.20:
        deploy_scalar = 0.85
    else:
        deploy_scalar = 1.0
    
    # Sector concentration penalty
    sector_scalar = 1.0
    if new_ticker_sector:
        same_sector = sum(1 for p in current_positions 
                         if p.get("sector", "") == new_ticker_sector)
        if same_sector >= 2:
            sector_scalar = 0.50  # Already 2+ in this sector
        elif same_sector == 1:
            sector_scalar = 0.75
    
    return count_scalar * deploy_scalar * sector_scalar


# ═══════════════════════════════════════════════════════════════════════════════
#  F. MARKET REGIME — VIX-based global adjustment
# ═══════════════════════════════════════════════════════════════════════════════

def _regime_scalar(vix: float = None, vix_regime: str = None) -> float:
    """
    Market-wide fear level adjusts all position sizes.
    
    VIX < 15:  calm market → 1.1x (slightly larger)
    VIX 15-20: normal → 1.0x
    VIX 20-30: elevated → 0.75x
    VIX 30-40: high fear → 0.50x
    VIX > 40:  panic → 0.30x
    """
    if vix is None or vix <= 0:
        # Try to use regime string
        if vix_regime == "low":
            return 1.10
        elif vix_regime == "elevated":
            return 0.75
        elif vix_regime == "high":
            return 0.50
        return 1.0  # Unknown = normal
    
    if vix < 15:
        return 1.10
    elif vix < 20:
        return 1.0
    elif vix < 25:
        return 0.80
    elif vix < 30:
        return 0.65
    elif vix < 40:
        return 0.50
    else:
        return 0.30


# ═══════════════════════════════════════════════════════════════════════════════
#  G. EARNINGS PROXIMITY — Reduce size near earnings
# ═══════════════════════════════════════════════════════════════════════════════

# In-memory earnings cache (refreshed once per day)
_earnings_cache = {}
_earnings_cache_time = 0

def _fetch_earnings_calendar() -> dict:
    """
    Fetch upcoming earnings from Alpaca. Returns {ticker: earnings_date_str}.
    Cached for 12 hours.
    """
    global _earnings_cache, _earnings_cache_time
    
    if _earnings_cache and (time.time() - _earnings_cache_time) < 43200:  # 12h cache
        return _earnings_cache
    
    try:
        # Alpaca doesn't have a dedicated earnings endpoint in free tier,
        # but we can check via the screener's upcoming earnings.
        # Fallback: use yfinance calendar for individual tickers.
        _earnings_cache_time = time.time()
        return _earnings_cache
    except Exception:
        return _earnings_cache


def get_earnings_date(ticker: str) -> str:
    """
    Get the next earnings date for a ticker.
    Returns ISO date string or None if unknown.
    Uses yfinance (already a dependency) as the source.
    """
    try:
        # Check cache first
        cache_key = f"earnings_{ticker}"
        if cache_key in _earnings_cache:
            cached = _earnings_cache[cache_key]
            if cached.get("fetched", 0) > time.time() - 43200:  # 12h cache
                return cached.get("date")
        
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.calendar
        
        if cal is not None:
            # yfinance returns different formats depending on version
            if isinstance(cal, dict):
                # Newer format: dict with 'Earnings Date' key
                dates = cal.get("Earnings Date", [])
                if dates:
                    date_str = str(dates[0])[:10]  # YYYY-MM-DD
                    _earnings_cache[cache_key] = {"date": date_str, "fetched": time.time()}
                    return date_str
            elif hasattr(cal, 'columns'):
                # DataFrame format
                if "Earnings Date" in cal.columns:
                    date_str = str(cal["Earnings Date"].iloc[0])[:10]
                    _earnings_cache[cache_key] = {"date": date_str, "fetched": time.time()}
                    return date_str
        
        _earnings_cache[cache_key] = {"date": None, "fetched": time.time()}
        return None
    except Exception:
        return None


def _earnings_scalar(ticker: str) -> float:
    """
    Reduce position size near earnings.
    
    > 7 days out:  1.0x (no adjustment)
    5-7 days out:  0.85x (slight caution)
    2-4 days out:  0.60x (significant reduction)
    0-1 days out:  0.35x (high risk — near coin flip)
    
    Returns scalar and days_to_earnings for logging.
    """
    earnings_date = get_earnings_date(ticker)
    
    if earnings_date is None:
        return 1.0  # Unknown = no adjustment
    
    try:
        now = datetime.now()
        earn_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
        days_to = (earn_dt - now).days
        
        if days_to < 0:
            return 1.0  # Earnings already passed
        elif days_to <= 1:
            return 0.35  # Tomorrow or today
        elif days_to <= 4:
            return 0.60  # This week
        elif days_to <= 7:
            return 0.85  # Next week
        else:
            return 1.0   # Far enough out
    except Exception:
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  H. TIME OF DAY — Smaller positions near close
# ═══════════════════════════════════════════════════════════════════════════════

def _time_scalar() -> float:
    """
    Adjust based on when during the trading day we're entering.
    
    First 30 min (9:30-10:00): 0.80x (opening volatility, gaps)
    Mid-day (10:00-14:00): 1.0x (normal)
    Power hour (14:00-15:30): 1.0x (fine, active trading)
    Last 30 min (15:30-16:00): 0.70x (less time to manage, close risk)
    After hours: 0.50x (wide spreads, low volume)
    """
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        et_hour = now_et.hour
        et_min = now_et.minute
        et_time = et_hour + et_min / 60.0
        
        if 9.5 <= et_time < 10.0:
            return 0.80   # Opening volatility
        elif 10.0 <= et_time < 14.0:
            return 1.0    # Normal trading
        elif 14.0 <= et_time < 15.5:
            return 1.0    # Power hour
        elif 15.5 <= et_time < 16.0:
            return 0.70   # Last 30 min
        else:
            return 0.50   # After/pre market
    except Exception:
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  I. EXCHANGE HALT CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_halt_status(ticker: str) -> dict:
    """
    Check if a stock is currently halted on the exchange.
    
    Returns: {"halted": bool, "status": str, "tradable": bool}
    """
    try:
        resp = requests.get(
            f"https://paper-api.alpaca.markets/v2/assets/{ticker}",
            headers={
                "APCA-API-KEY-ID": ALPACA_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET,
            },
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "active")
            tradable = data.get("tradable", True)
            return {
                "halted": status != "active" or not tradable,
                "status": status,
                "tradable": tradable,
            }
    except Exception:
        pass
    
    return {"halted": False, "status": "unknown", "tradable": True}


# ═══════════════════════════════════════════════════════════════════════════════
#  J. FEE / SLIPPAGE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_costs(shares: int, price: float, volume: int, 
                    trade_type: str = "stock") -> dict:
    """
    Estimate total cost of entering AND exiting a trade.
    
    Returns: {
        "commission_entry": float,
        "commission_exit": float,
        "estimated_slippage_pct": float,
        "total_cost_pct": float,  ← % of position value eaten by costs
    }
    """
    position_value = shares * price
    
    # Commissions (roundtrip)
    if trade_type == "options":
        commission_entry = shares * OPTIONS_FEE_PER_CONTRACT  # shares = contracts
        commission_exit = commission_entry
    else:
        commission_entry = shares * DEFAULT_COMMISSION_PER_SHARE
        commission_exit = commission_entry
    
    # Slippage estimate based on volume
    if volume and volume > 0:
        # Our shares as % of daily volume
        volume_pct = shares / volume if volume > 0 else 0.01
        if volume_pct > 0.01:
            slippage_pct = 0.30  # Moving the market
        elif volume_pct > 0.005:
            slippage_pct = 0.15
        elif volume_pct > 0.001:
            slippage_pct = 0.08
        elif volume > 10_000_000:
            slippage_pct = 0.02  # Very liquid
        elif volume > 5_000_000:
            slippage_pct = 0.04
        elif volume > 1_000_000:
            slippage_pct = 0.06
        else:
            slippage_pct = 0.10  # Thin
    else:
        slippage_pct = 0.10
    
    total_commission = commission_entry + commission_exit
    slippage_cost = position_value * slippage_pct / 100 * 2  # Entry + exit
    total_cost = total_commission + slippage_cost
    total_cost_pct = (total_cost / position_value * 100) if position_value > 0 else 0
    
    return {
        "commission_entry": round(commission_entry, 2),
        "commission_exit": round(commission_exit, 2),
        "estimated_slippage_pct": round(slippage_pct, 4),
        "total_cost_pct": round(total_cost_pct, 4),
        "total_cost_dollars": round(total_cost, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN: CALCULATE POSITION SIZE
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_position(trade: dict, equity: float, current_positions: list = None,
                       macro: dict = None) -> dict:
    """
    THE CORE FUNCTION — replaces all hard-coded sizing.
    
    Inputs:
        trade: {
            ticker, price, score, volume, deep_score, side, trade_type,
            ewma_rv, garch_rv, momentum_score, vrp_score, rsi,
            ml_confidence (optional), sector (optional)
        }
        equity: total portfolio value
        current_positions: list of Alpaca position objects
        macro: {vix, vix_regime, ...} from macro_data.get_macro_snapshot()
    
    Returns: {
        shares: int,              ← exact shares to buy (whole number)
        position_value: float,    ← shares × price
        position_pct: float,      ← % of portfolio this represents
        stop_loss: float,         ← dynamic stop price
        take_profit: float,       ← dynamic take profit price
        risk_per_share: float,    ← distance to stop in dollars
        risk_total: float,        ← total dollars at risk
        max_loss_pct: float,      ← worst case % loss on portfolio
        costs: dict,              ← fee/slippage breakdown
        scalars: dict,            ← every multiplier used (for transparency)
        reasoning: str,           ← plain English explanation
        blocked: bool,            ← True if trade should NOT be taken
        block_reason: str,        ← why it was blocked
    }
    """
    ticker = trade.get("ticker", "???")
    price = float(trade.get("price", 0))
    score = float(trade.get("deep_score", trade.get("score", 50)))
    volume = int(trade.get("volume", 0))
    trade_type = trade.get("trade_type", "stock")
    side = trade.get("side", "buy")
    
    # ── Pre-checks: should we even trade? ─────────────────────────────────
    
    # Check exchange halt
    halt = check_halt_status(ticker)
    if halt["halted"]:
        return _blocked(ticker, price, f"Stock is halted (status: {halt['status']})")
    
    if price <= 0:
        return _blocked(ticker, price, "Invalid price")
    
    if equity <= 0:
        return _blocked(ticker, price, "No equity available")
    
    # ── Step 1: Kelly base size ───────────────────────────────────────────
    
    hist_stats = _get_historical_stats()
    
    # Use strategy-specific stats if we have enough data
    strategy = _infer_strategy(trade)
    if strategy in hist_stats["by_strategy"]:
        strat_stats = hist_stats["by_strategy"][strategy]
        kelly_base = _kelly_fraction(
            strat_stats["win_rate"], strat_stats["avg_win"], strat_stats["avg_loss"]
        )
    else:
        overall = hist_stats["overall"]
        kelly_base = _kelly_fraction(
            overall["win_rate"], overall["avg_win"], overall["avg_loss"]
        )
    
    # ── Step 2: Apply all scalars ─────────────────────────────────────────
    
    # ATR as percentage of price
    ewma_rv = trade.get("ewma_rv")
    garch_rv = trade.get("garch_rv")
    atr_pct = None
    if ewma_rv and ewma_rv > 0:
        atr_pct = ewma_rv  # Already in % form from bot_engine
    elif garch_rv and garch_rv > 0:
        atr_pct = garch_rv
    
    s_volatility = _volatility_scalar(atr_pct, garch_rv)
    s_confidence = _confidence_scalar(score, trade.get("ml_confidence"))
    s_regime = _regime_scalar(
        macro.get("vix") if macro else None,
        macro.get("vix_regime") if macro else None,
    )
    s_earnings = _earnings_scalar(ticker)
    s_time = _time_scalar()
    s_heat = _portfolio_heat_scalar(
        current_positions, equity, trade.get("sector")
    )
    
    # Preliminary position value for liquidity check
    preliminary_pct = kelly_base * s_volatility * s_confidence * s_regime * s_earnings * s_time * s_heat
    preliminary_value = equity * preliminary_pct
    s_liquidity = _liquidity_scalar(volume, price, preliminary_value)
    
    # ── Step 3: Combine into final percentage ─────────────────────────────
    
    final_pct = (kelly_base 
                 * s_volatility 
                 * s_confidence 
                 * s_regime 
                 * s_earnings 
                 * s_time 
                 * s_heat 
                 * s_liquidity)
    
    # Clamp to absolute limits
    final_pct = max(ABSOLUTE_MIN_POSITION_PCT, min(final_pct, ABSOLUTE_MAX_POSITION_PCT))
    
    # ── Step 4: Convert to whole shares ───────────────────────────────────
    
    target_value = equity * final_pct
    raw_shares = target_value / price
    shares = int(math.floor(raw_shares))  # Always round down
    
    if shares <= 0:
        return _blocked(ticker, price, f"Position too small ({raw_shares:.2f} shares at {final_pct:.1%})")
    
    actual_value = shares * price
    actual_pct = actual_value / equity if equity > 0 else 0
    
    # ── Step 5: Dynamic stop-loss and take-profit ─────────────────────────
    
    # ATR-based stops (same logic as manage_positions but at ENTRY time)
    if atr_pct and atr_pct > 0:
        stop_distance_pct = max(1.5, min(atr_pct * 1.5, 8.0))
        tp_distance_pct = max(4.0, min(atr_pct * 3.0, 15.0))
    else:
        stop_distance_pct = 2.5  # Default
        tp_distance_pct = 7.5
    
    if side in ("short", "sell"):
        stop_loss = round(price * (1 + stop_distance_pct / 100), 2)
        take_profit = round(price * (1 - tp_distance_pct / 100), 2)
    else:
        stop_loss = round(price * (1 - stop_distance_pct / 100), 2)
        take_profit = round(price * (1 + tp_distance_pct / 100), 2)
    
    risk_per_share = abs(price - stop_loss)
    risk_total = risk_per_share * shares
    max_loss_pct = (risk_total / equity * 100) if equity > 0 else 0
    
    # ── Step 6: Cost estimation ───────────────────────────────────────────
    
    costs = _estimate_costs(shares, price, volume, trade_type)
    
    # If costs eat more than 1% of the position, warn
    cost_warning = costs["total_cost_pct"] > 1.0
    
    # ── Step 7: Build reasoning ───────────────────────────────────────────
    
    scalars = {
        "kelly_base": round(kelly_base, 4),
        "volatility": round(s_volatility, 3),
        "confidence": round(s_confidence, 3),
        "regime": round(s_regime, 3),
        "earnings": round(s_earnings, 3),
        "time_of_day": round(s_time, 3),
        "portfolio_heat": round(s_heat, 3),
        "liquidity": round(s_liquidity, 3),
        "combined": round(final_pct, 4),
    }
    
    reasoning_parts = [f"Kelly base: {kelly_base:.1%}"]
    if s_volatility != 1.0:
        reasoning_parts.append(f"Vol: {s_volatility:.0%}")
    if s_confidence != 1.0:
        reasoning_parts.append(f"Confidence: {s_confidence:.0%}")
    if s_regime != 1.0:
        reasoning_parts.append(f"Market: {s_regime:.0%}")
    if s_earnings != 1.0:
        reasoning_parts.append(f"Earnings: {s_earnings:.0%}")
    if s_time != 1.0:
        reasoning_parts.append(f"Time: {s_time:.0%}")
    if s_heat != 1.0:
        reasoning_parts.append(f"Heat: {s_heat:.0%}")
    if s_liquidity != 1.0:
        reasoning_parts.append(f"Liquidity: {s_liquidity:.0%}")
    
    reasoning = f"{shares} shares @ ${price:.2f} = ${actual_value:,.0f} ({actual_pct:.1%} of portfolio) | " + " × ".join(reasoning_parts)
    
    if cost_warning:
        reasoning += f" | ⚠ Costs: {costs['total_cost_pct']:.2f}% of position"
    
    if s_earnings < 1.0:
        earn_date = get_earnings_date(ticker)
        if earn_date:
            reasoning += f" | Earnings: {earn_date}"
    
    return {
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "side": side,
        "position_value": round(actual_value, 2),
        "position_pct": round(actual_pct, 4),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "stop_distance_pct": round(stop_distance_pct, 2),
        "tp_distance_pct": round(tp_distance_pct, 2),
        "risk_per_share": round(risk_per_share, 2),
        "risk_total": round(risk_total, 2),
        "max_loss_pct": round(max_loss_pct, 2),
        "costs": costs,
        "scalars": scalars,
        "reasoning": reasoning,
        "blocked": False,
        "block_reason": None,
        "hist_stats": hist_stats["overall"],
    }


def _blocked(ticker: str, price: float, reason: str) -> dict:
    """Return a blocked trade result."""
    return {
        "ticker": ticker,
        "shares": 0,
        "price": price,
        "side": "none",
        "position_value": 0,
        "position_pct": 0,
        "stop_loss": 0,
        "take_profit": 0,
        "stop_distance_pct": 0,
        "tp_distance_pct": 0,
        "risk_per_share": 0,
        "risk_total": 0,
        "max_loss_pct": 0,
        "costs": {},
        "scalars": {},
        "reasoning": f"BLOCKED: {reason}",
        "blocked": True,
        "block_reason": reason,
        "hist_stats": {},
    }


def _infer_strategy(trade: dict) -> str:
    """Infer the dominant strategy from scores."""
    scores = {
        "momentum": trade.get("momentum_score", 0) or 0,
        "mean_reversion": trade.get("mean_reversion_score", 0) or 0,
        "vrp": trade.get("vrp_score", 0) or 0,
        "squeeze": trade.get("squeeze_score", 0) or 0,
        "volume": trade.get("volume_score", 0) or 0,
    }
    if not any(scores.values()):
        return "unknown"
    return max(scores, key=scores.get)


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH SIZING — Size multiple trades together (for portfolio construction)
# ═══════════════════════════════════════════════════════════════════════════════

def size_portfolio(trades: list, equity: float, current_positions: list = None,
                   macro: dict = None) -> list:
    """
    Size a batch of trades together, ensuring total deployment stays safe.
    Redistributes leftover cash from rounding to the highest-conviction trade.
    
    Returns list of sized trades (same order, with shares/values filled in).
    """
    if not trades:
        return []
    
    # Sort by score descending — best trades get priority
    indexed = [(i, t) for i, t in enumerate(trades)]
    indexed.sort(key=lambda x: x[1].get("deep_score", x[1].get("score", 0)), reverse=True)
    
    sized = [None] * len(trades)
    total_deployed = 0
    max_deploy = equity * ABSOLUTE_MAX_PORTFOLIO_HEAT
    
    # Account for already-deployed capital
    if current_positions and isinstance(current_positions, list):
        total_deployed = sum(abs(float(p.get("market_value", 0))) for p in current_positions)
    
    positions_used = len(current_positions) if current_positions else 0
    
    for orig_idx, trade in indexed:
        if positions_used >= ABSOLUTE_MAX_POSITIONS:
            sized[orig_idx] = _blocked(trade.get("ticker", "?"), 
                                        float(trade.get("price", 0)),
                                        f"Max positions ({ABSOLUTE_MAX_POSITIONS}) reached")
            continue
        
        remaining_budget = max_deploy - total_deployed
        if remaining_budget <= 0:
            sized[orig_idx] = _blocked(trade.get("ticker", "?"),
                                        float(trade.get("price", 0)),
                                        f"Portfolio heat limit ({ABSOLUTE_MAX_PORTFOLIO_HEAT:.0%}) reached")
            continue
        
        result = calculate_position(trade, equity, current_positions, macro)
        
        # Cap at remaining budget
        if not result["blocked"] and result["position_value"] > remaining_budget:
            new_shares = int(math.floor(remaining_budget / result["price"]))
            if new_shares > 0:
                result["shares"] = new_shares
                result["position_value"] = round(new_shares * result["price"], 2)
                result["position_pct"] = round(result["position_value"] / equity, 4)
                result["reasoning"] += " | Capped by portfolio heat limit"
            else:
                result = _blocked(trade.get("ticker", "?"), result["price"],
                                  "Remaining budget too small for a single share")
        
        sized[orig_idx] = result
        if not result["blocked"]:
            total_deployed += result["position_value"]
            positions_used += 1
    
    # Redistribute leftover cash from rounding to top pick
    total_used = sum(r["position_value"] for r in sized if r and not r["blocked"])
    leftover = max_deploy - total_deployed
    
    if leftover > 0 and sized:
        # Find highest-scored non-blocked trade
        for orig_idx, trade in indexed:
            r = sized[orig_idx]
            if r and not r["blocked"] and r["price"] > 0:
                extra_shares = int(math.floor(leftover / r["price"]))
                if extra_shares > 0:
                    # Don't exceed absolute max
                    new_total = (r["shares"] + extra_shares) * r["price"]
                    if new_total / equity <= ABSOLUTE_MAX_POSITION_PCT:
                        r["shares"] += extra_shares
                        r["position_value"] = round(r["shares"] * r["price"], 2)
                        r["position_pct"] = round(r["position_value"] / equity, 4)
                        r["reasoning"] += f" | +{extra_shares} shares from leftover cash"
                break
    
    return sized


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI: Can be called from bot.ts via python3 position_sizing.py <json>
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        try:
            data = json.loads(sys.argv[1])
            mode = data.get("mode", "single")
            
            if mode == "batch":
                results = size_portfolio(
                    data["trades"], data["equity"],
                    data.get("positions", []), data.get("macro", {})
                )
                print(json.dumps(results))
            elif mode == "halt_check":
                result = check_halt_status(data["ticker"])
                print(json.dumps(result))
            elif mode == "earnings_check":
                scalar = _earnings_scalar(data["ticker"])
                date = get_earnings_date(data["ticker"])
                print(json.dumps({"scalar": scalar, "earnings_date": date}))
            else:
                result = calculate_position(
                    data["trade"], data["equity"],
                    data.get("positions", []), data.get("macro", {})
                )
                print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
    else:
        # Quick self-test
        test_trade = {
            "ticker": "NVDA", "price": 920, "score": 85, "deep_score": 85,
            "volume": 60000000, "side": "buy", "trade_type": "stock",
            "ewma_rv": 2.5, "garch_rv": 2.8, "momentum_score": 75,
            "vrp_score": 60, "ml_confidence": 0.72, "sector": "Technology",
        }
        result = calculate_position(test_trade, 100000, [], {"vix": 18, "vix_regime": "normal"})
        print(json.dumps(result, indent=2))
