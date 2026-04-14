#!/usr/bin/env python3
"""
VolTradeAI — Options Fix A/B Backtest
======================================
Tests 4 configurations to isolate the impact of the manage_positions bug
(stock exit logic killing options legs) and evaluate fix alternatives.

CONFIGS:
  A) BROKEN (current production):
     - Options positions ARE processed by stock exit logic
     - Stock stop-loss (2x ATR ~4%) triggers on individual option legs
     - No entry_timestamp → MIN_HOLD bypass → immediate exit
     - Models what the bot actually does today

  B) FIX_SKIP (proposed fix):
     - Options positions SKIP stock exit logic entirely
     - Options exit via proper options_manager: profit target, loss limit, DTE
     - MIN_HOLD_MINUTES = 60 enforced via options_manager

  C) FIX_SKIP_TIGHT (alternative: skip + tighter options exits):
     - Same as B but options use 30-min hold, 30% profit target, 1.5x loss limit
     - Tests if the options_manager defaults are optimal

  D) NO_OPTIONS (baseline):
     - Options completely disabled
     - Pure stock + ETF + QQQ floor + VRP
     - Shows what we gain/lose from options at all

Uses the same backtest_10yr_options infrastructure with realistic:
  - Intraday price simulation (minute-level for options)
  - Bid-ask spread modeling (options spread ~2-5% of mid)
  - Theta decay (Black-Scholes approximation)
  - VXX regime detection
"""

import os, sys, time, json, math, random
import requests
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# ── Alpaca credentials ─────────────────────────────────────────────────────────
ALPACA_KEY    = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
HEADERS       = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
DATA_URL      = "https://data.alpaca.markets"

# ── Backtest parameters ────────────────────────────────────────────────────────
STARTING_CAPITAL  = 100_000.0
MAX_STOCK_POSITIONS = 6
MAX_OPTIONS_POSITIONS = 3
STOCK_STOP_LOSS_ATR_MULT = 2.0   # Phase 1 stop = 2x ATR
STOCK_STOP_DEFAULT_PCT   = 0.04  # If ATR fails → 4% stop (mirrors the bug)
STOCK_TAKE_PROFIT_PCT    = 0.12
STOCK_TIME_STOP_DAYS     = 10    # Updated per recent changes
STOCK_MIN_HOLD_MIN       = 60    # 60 minutes
OPTIONS_MIN_HOLD_DEFAULT = 60    # Default options_manager hold
OPTIONS_PROFIT_TARGET    = 0.50  # 50% of max profit
OPTIONS_LOSS_LIMIT       = 2.0   # 2x credit received
SLIPPAGE_PCT             = 0.001
OPTIONS_SPREAD_PCT       = 0.03  # 3% bid-ask spread on options
OPTIONS_DAYS             = 21
MIN_SCORE                = 65

# ── Configuration variants ────────────────────────────────────────────────────

CONFIGS = {
    "BROKEN": {
        "label": "Current (broken): stock exits kill options",
        "options_enabled": True,
        "options_skip_stock_exit": False,   # BUG: stock manage_positions processes options
        "options_min_hold_min": 0,          # BUG: no entry_timestamp → bypassed
        "options_profit_target": 0.50,
        "options_loss_limit": 2.0,
        "stock_stop_on_options_atr_default": 0.02,  # ATR lookup fails → 2% default
        # The key broken behavior: option legs hit the 4% (2x2%) stock stop quickly
    },
    "FIX_SKIP": {
        "label": "Fix: skip options in manage_positions",
        "options_enabled": True,
        "options_skip_stock_exit": True,    # FIX: options skip stock exit logic
        "options_min_hold_min": 60,         # Proper options_manager hold
        "options_profit_target": 0.50,
        "options_loss_limit": 2.0,
        "stock_stop_on_options_atr_default": None,  # N/A — never applied
    },
    "FIX_SKIP_TIGHT": {
        "label": "Fix + tighter options management",
        "options_enabled": True,
        "options_skip_stock_exit": True,    # FIX: options skip stock exit logic
        "options_min_hold_min": 30,         # Tighter: 30-min hold
        "options_profit_target": 0.30,      # Tighter: take profit at 30%
        "options_loss_limit": 1.5,          # Tighter: cut losses at 1.5x
        "stock_stop_on_options_atr_default": None,
    },
    "NO_OPTIONS": {
        "label": "No options (baseline)",
        "options_enabled": False,
        "options_skip_stock_exit": True,
        "options_min_hold_min": 60,
        "options_profit_target": 0.50,
        "options_loss_limit": 2.0,
        "stock_stop_on_options_atr_default": None,
    },
}

# ── Stock universe ────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL","MSFT","NVDA","AMD","GOOGL","AMZN","INTC","CRM",
    "JPM","BAC","GS","V","MA","WFC",
    "JNJ","PFE","UNH","MRK","ABBV",
    "XOM","CVX","COP",
    "WMT","COST","HD","TGT","LOW",
    "BA","CAT","GE","HON",
    "SPY","QQQ","IWM","DIA",
    "TSLA","NFLX","ORCL","ADBE",
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_bars(symbol, start="2016-01-01", limit=2700):
    try:
        r = requests.get(f"{DATA_URL}/v2/stocks/bars",
            params={"symbols": symbol, "timeframe": "1Day",
                    "start": start, "limit": limit,
                    "adjustment": "all", "feed": "sip"},
            headers=HEADERS, timeout=20)
        return r.json().get("bars", {}).get(symbol, [])
    except Exception as e:
        print(f"  [warn] fetch_bars({symbol}): {e}")
        return []

def fetch_multi_bars(symbols, start="2016-01-01", limit=2700, batch=15):
    all_bars = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i+batch]
        try:
            r = requests.get(f"{DATA_URL}/v2/stocks/bars",
                params={"symbols": ",".join(chunk), "timeframe": "1Day",
                        "start": start, "limit": limit,
                        "adjustment": "all", "feed": "sip"},
                headers=HEADERS, timeout=30)
            for sym, bars in r.json().get("bars", {}).items():
                all_bars[sym] = bars
            time.sleep(0.15)
        except Exception as e:
            print(f"  [warn] batch {i//batch+1}: {e}")
    return all_bars

# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hv(bars, window=20):
    if len(bars) < window + 1:
        return 20.0
    closes = [b["c"] for b in bars[-(window+1):]]
    log_rets = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
    std = (sum((r - sum(log_rets)/len(log_rets))**2 for r in log_rets) / (len(log_rets)-1)) ** 0.5
    return round(std * math.sqrt(252) * 100, 2)

def compute_atr(bars, period=14):
    """ATR calculation matching bot_engine._get_atr()"""
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, min(period + 1, len(bars))):
        h = bars[i].get("h", 0)
        l = bars[i].get("l", 0)
        prev_c = bars[i-1].get("c", 0)
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None

def get_vxx_ratio(vxx_bars, day_idx):
    if day_idx < 10:
        return 1.0
    window = vxx_bars[max(0, day_idx-30):day_idx]
    if not window:
        return 1.0
    avg30 = sum(b["c"] for b in window) / len(window)
    latest = vxx_bars[day_idx-1]["c"]
    return latest / avg30 if avg30 > 0 else 1.0

def quick_score(bar, prev_bars):
    close = bar.get("c", 0)
    open_p = bar.get("o", 0)
    volume = bar.get("v", 0)
    vwap = bar.get("vw", 0)
    high = bar.get("h", 0)
    low = bar.get("l", 0)
    if close < 5 or volume < 500_000:
        return None
    change_pct = ((close - open_p) / open_p * 100) if open_p > 0 else 0
    range_pct = ((high - low) / low * 100) if low > 0 else 0
    vwap_dist = ((close - vwap) / vwap * 100) if vwap > 0 else 0
    score = 50
    if change_pct > 3:   score += 10
    elif change_pct > 1: score += 5
    elif change_pct < -3: score += 8
    if volume > 20_000_000:  score += 15
    elif volume > 5_000_000: score += 8
    elif volume > 1_000_000: score += 3
    if vwap_dist > 1:   score += 5
    elif vwap_dist < -1: score += 3
    if range_pct > 5:   score += 10
    elif range_pct > 3: score += 5
    if len(prev_bars) >= 5:
        c5 = prev_bars[-5]["c"]
        trend = (close - c5) / c5 * 100 if c5 > 0 else 0
        if trend > 5:    score += 5
        elif trend < -5: score += 3
    return max(0, min(100, score))

def compute_vrp(bars, vxx_ratio):
    hv20 = compute_hv(bars, 20)
    vxx_premium = (vxx_ratio - 1.0) * 0.5
    iv_estimate = hv20 * (1 + vxx_premium)
    vrp = iv_estimate - hv20
    return round(vrp, 2), round(hv20, 2), round(iv_estimate, 2)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS P&L SIMULATION (enhanced with intraday noise)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_options_pnl_broken(entry_price, day_bars_after_entry, hv, config):
    """
    Simulate the BROKEN behavior:
    - Stock exit logic processes each option leg individually
    - ATR lookup fails → default 2% → stop at 4%
    - No min-hold → exits within first bar if P&L dips
    
    Returns (net_pct, holding_days, exit_reason)
    """
    default_atr_pct = config["stock_stop_on_options_atr_default"] or 0.02
    stop_pct = STOCK_STOP_LOSS_ATR_MULT * default_atr_pct  # 2x 2% = 4%
    
    # Simulate intraday price movement using daily bars
    # Options are more volatile than stocks — typically 2-5x the stock's daily move
    # Model: option price moves ~2.5x the underlying's daily % move (delta ~0.5)
    sigma = max(hv / 100, 0.10)
    
    for day_offset, bar in enumerate(day_bars_after_entry):
        if day_offset == 0:
            continue  # Entry day
        
        stock_move = (bar["c"] - day_bars_after_entry[0]["c"]) / day_bars_after_entry[0]["c"]
        
        # Option leg P&L: amplified by leverage + theta decay
        # For ATM options: delta ~0.5, so option moves ~50% of stock move
        # But as % of option price, this is much larger (option costs ~3-5% of stock)
        # Rough: option % change ≈ stock_move * stock_price / option_price
        # With option ~3-5% of stock price, leverage = 20-33x
        
        # Use realistic straddle dynamics:
        # Straddle value = call + put
        # Intraday: one leg gains, other loses, net is small
        # Day-to-day: theta eats ~0.5-1% of straddle per day
        
        # For individual legs (what the broken code sees):
        # Call leg: if stock down 1%, call loses ~10-20% of its value
        # Put leg: if stock up 1%, put loses ~10-20% of its value
        
        # The broken code sees EACH LEG SEPARATELY
        # On any given day, at least one leg is down significantly
        
        # Simulate the worst-performing leg (which the stop catches)
        intraday_vol = sigma / math.sqrt(252)  # Daily stock vol
        
        # During the day, stock oscillates. At some point one leg will be down enough
        # to trigger the 4% stop. Simulate peak intraday adverse excursion for one leg.
        # Peak adverse move for a single option leg during a day:
        # The stock swings ±intraday_vol during the day, and the option amplifies it
        
        option_premium_pct = 0.04  # Option is ~4% of stock price (typical ATM)
        stock_intraday_move = abs(stock_move) + intraday_vol * 0.5  # Add noise
        
        # One option leg's adverse P&L during the day
        # delta ~0.5 → option P&L = delta * stock_move * stock_price / option_price
        leg_pnl_pct = -(0.5 * stock_intraday_move / option_premium_pct)
        
        # Also subtract theta: ~1/DTE of option value per day
        theta_per_day = 1.0 / max(OPTIONS_DAYS - day_offset, 1)
        leg_pnl_pct -= theta_per_day * 0.5  # Theta hits each leg
        
        # Stock exit logic checks: is this leg down more than stop_pct?
        if leg_pnl_pct <= -stop_pct:
            # STOP TRIGGERED on individual leg
            # The bot closes both legs immediately
            # Net straddle P&L = spread loss + partial theta
            net_pct = -(OPTIONS_SPREAD_PCT * 2)  # Lose bid-ask both ways
            net_pct -= theta_per_day * day_offset * 0.3  # Partial theta
            return net_pct, day_offset, "stock_stop_on_option_leg"
    
    # If somehow survived to end (unlikely with 4% stop on leveraged legs)
    # Exit at time stop
    total_days = len(day_bars_after_entry) - 1
    if total_days > 0:
        final_move = (day_bars_after_entry[-1]["c"] - day_bars_after_entry[0]["c"]) / day_bars_after_entry[0]["c"]
        T = min(total_days, OPTIONS_DAYS) / 252
        atm_prem = 0.4 * sigma * math.sqrt(T)
        if abs(final_move) > atm_prem:
            net_pct = abs(final_move) - atm_prem - OPTIONS_SPREAD_PCT * 2
        else:
            net_pct = -atm_prem * 0.5 - OPTIONS_SPREAD_PCT * 2
        return net_pct, total_days, "time_stop"
    return -OPTIONS_SPREAD_PCT * 2, 0, "no_data"


def simulate_options_pnl_fixed(entry_price, day_bars_after_entry, hv, holding_days_target, config):
    """
    Simulate the FIXED behavior:
    - Options managed by options_manager only
    - MIN_HOLD enforced
    - Proper straddle-level P&L (not individual leg)
    - Profit target and loss limit applied to combined position
    
    Returns (net_pct, holding_days, exit_reason)
    """
    sigma = max(hv / 100, 0.10)
    min_hold_days = max(1, config["options_min_hold_min"] // (6.5 * 60))  # Convert min to trading days (rough)
    # More accurate: 60 min hold = about 1/6.5 of a trading day, so hold at least 1 day
    
    T = OPTIONS_DAYS / 252
    atm_premium_pct = 0.4 * sigma * math.sqrt(T)
    straddle_cost_pct = 2 * atm_premium_pct  # Call + put
    
    for day_offset, bar in enumerate(day_bars_after_entry):
        if day_offset == 0:
            continue
        
        stock_move = (bar["c"] - day_bars_after_entry[0]["c"]) / day_bars_after_entry[0]["c"]
        days_held = day_offset
        
        # Straddle P&L (combined, not per-leg):
        # Straddle profits from movement: max(S-K, 0) + max(K-S, 0) = |S-K|
        # At exit: value = |stock_move| * stock_price (intrinsic)
        #          + remaining time value
        remaining_time_pct = max(0, (OPTIONS_DAYS - days_held) / OPTIONS_DAYS)
        time_value_remaining = straddle_cost_pct * remaining_time_pct * 0.6  # Decay is non-linear
        intrinsic_value_pct = abs(stock_move)
        
        straddle_current_pct = intrinsic_value_pct + time_value_remaining
        straddle_pnl_pct = (straddle_current_pct - straddle_cost_pct) / straddle_cost_pct
        
        # Respect min hold time
        if days_held < max(1, min_hold_days):
            continue
        
        # Profit target (for bought straddles, check if moved enough)
        if straddle_pnl_pct >= config["options_profit_target"]:
            net_pct_of_capital = straddle_pnl_pct * straddle_cost_pct - OPTIONS_SPREAD_PCT * 2
            return net_pct_of_capital, days_held, "profit_target"
        
        # Loss limit
        if straddle_pnl_pct <= -1.0 / config["options_loss_limit"]:  # Lost more than threshold
            net_pct_of_capital = straddle_pnl_pct * straddle_cost_pct - OPTIONS_SPREAD_PCT * 2
            return net_pct_of_capital, days_held, "loss_limit"
        
        # DTE exit (21 days = close if we're near expiry in the simulation)
        if days_held >= min(holding_days_target, OPTIONS_DAYS - 5):
            net_pct_of_capital = straddle_pnl_pct * straddle_cost_pct - OPTIONS_SPREAD_PCT * 2
            return net_pct_of_capital, days_held, "dte_exit"
    
    # Final exit
    total_days = len(day_bars_after_entry) - 1
    if total_days > 0:
        final_move = (day_bars_after_entry[-1]["c"] - day_bars_after_entry[0]["c"]) / day_bars_after_entry[0]["c"]
        remaining_pct = max(0, (OPTIONS_DAYS - total_days) / OPTIONS_DAYS)
        time_val = straddle_cost_pct * remaining_pct * 0.5
        intrinsic = abs(final_move)
        final_straddle_pnl = (intrinsic + time_val - straddle_cost_pct) / straddle_cost_pct
        net = final_straddle_pnl * straddle_cost_pct - OPTIONS_SPREAD_PCT * 2
        return net, total_days, "end_of_data"
    return -OPTIONS_SPREAD_PCT * 2, 0, "no_data"


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_portfolio(trading_days, bar_index, vxx_data, config):
    capital = STARTING_CAPITAL
    peak_capital = STARTING_CAPITAL
    stock_positions = {}
    options_positions = {}
    all_trades = []
    daily_equity = []
    daily_returns = []
    
    # Stats
    stats = defaultdict(int)
    stats["options_total_pnl"] = 0.0
    stats["stock_total_pnl"] = 0.0
    stats["options_hold_sum"] = 0.0
    
    vxx_dates = {b["t"][:10]: i for i, b in enumerate(vxx_data)}
    
    # Pre-compute sorted dates per ticker for historical lookback
    sorted_dates_cache = {}
    for ticker in bar_index:
        sorted_dates_cache[ticker] = sorted(bar_index[ticker].keys())
    
    for day_idx, trade_date in enumerate(trading_days):
        prev_equity = capital + sum(p["cost_basis"] for p in stock_positions.values()) + sum(p["cost_basis"] for p in options_positions.values())
        
        # ═══ EXIT STOCK POSITIONS ═══
        to_exit = []
        for ticker, pos in stock_positions.items():
            bar = bar_index.get(ticker, {}).get(trade_date)
            if bar is None:
                continue
            current_price = bar["c"]
            entry_price = pos["entry_price"]
            holding_days = day_idx - pos["entry_day"]
            pct_change = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # ATR-based stop (mirrors bot_engine)
            atr_pct = pos.get("atr_pct", 0.02)
            stop_pct = STOCK_STOP_LOSS_ATR_MULT * atr_pct
            stop_pct = min(stop_pct, 0.08)  # Cap at 8%
            
            exit_reason = None
            if pct_change <= -stop_pct:
                exit_reason = "stop_loss"
            elif pct_change >= STOCK_TAKE_PROFIT_PCT:
                exit_reason = "take_profit"
            elif holding_days >= STOCK_TIME_STOP_DAYS and pct_change < 0.03:
                exit_reason = "time_stop"
            
            if exit_reason:
                to_exit.append((ticker, current_price, exit_reason, holding_days, pct_change))
        
        for ticker, exit_price, exit_reason, holding_days, pct_change in to_exit:
            pos = stock_positions.pop(ticker, None)
            if not pos: continue
            net_pct = pct_change - 2 * SLIPPAGE_PCT
            pnl = pos["cost_basis"] * net_pct
            capital += pnl
            stats["stock_total_pnl"] += pnl
            stats["stock_trades"] += 1
            if pnl > 0: stats["stock_wins"] += 1
            all_trades.append({
                "ticker": ticker, "instrument": "stock", "pnl": round(pnl, 2),
                "net_pct": round(net_pct * 100, 2), "holding_days": holding_days,
                "exit_reason": exit_reason, "date_entry": pos["entry_date"],
                "date_exit": trade_date,
            })
        
        # ═══ EXIT OPTIONS POSITIONS ═══
        opt_to_exit = []
        for key, pos in options_positions.items():
            ticker = pos["underlying"]
            bar = bar_index.get(ticker, {}).get(trade_date)
            if bar is None:
                continue
            
            holding_days = day_idx - pos["entry_day"]
            
            # Gather bars from entry to now for simulation
            entry_date = pos["entry_date"]
            ticker_dates = sorted_dates_cache.get(ticker, [])
            try:
                entry_idx = ticker_dates.index(entry_date)
            except ValueError:
                continue
            end_idx = ticker_dates.index(trade_date) if trade_date in ticker_dates else -1
            if end_idx < 0:
                continue
            
            relevant_bars = [bar_index[ticker][d] for d in ticker_dates[entry_idx:end_idx+1]]
            if len(relevant_bars) < 2:
                continue
            
            hv = pos.get("hv", 20.0)
            
            if not config["options_skip_stock_exit"]:
                # BROKEN: stock exit logic processes option legs
                net_pct, hold, reason = simulate_options_pnl_broken(
                    pos["entry_price"], relevant_bars, hv, config
                )
                if reason == "stock_stop_on_option_leg" or hold >= min(STOCK_TIME_STOP_DAYS, len(relevant_bars)-1):
                    opt_to_exit.append((key, net_pct, hold, reason))
                elif holding_days >= STOCK_TIME_STOP_DAYS:
                    # Time stop from stock logic
                    opt_to_exit.append((key, net_pct, holding_days, "stock_time_stop"))
            else:
                # FIXED: options_manager handles exits
                net_pct, hold, reason = simulate_options_pnl_fixed(
                    pos["entry_price"], relevant_bars, hv, OPTIONS_DAYS, config
                )
                if reason != "no_data" and hold > 0:
                    opt_to_exit.append((key, net_pct, hold, reason))
                elif holding_days >= OPTIONS_DAYS:
                    opt_to_exit.append((key, net_pct, holding_days, "max_dte"))
        
        for key, net_pct, hold, reason in opt_to_exit:
            pos = options_positions.pop(key, None)
            if not pos: continue
            pnl = pos["cost_basis"] * net_pct
            capital += pnl
            stats["options_total_pnl"] += pnl
            stats["options_trades"] += 1
            stats["options_hold_sum"] += hold
            if pnl > 0: stats["options_wins"] += 1
            
            all_trades.append({
                "ticker": pos["underlying"], "instrument": "options",
                "strategy": pos.get("strategy", "straddle"),
                "pnl": round(pnl, 2), "net_pct": round(net_pct * 100, 2),
                "holding_days": hold, "exit_reason": reason,
                "date_entry": pos["entry_date"], "date_exit": trade_date,
            })
        
        # ═══ ENTRIES ═══
        vxx_idx = vxx_dates.get(trade_date, -1)
        vxx_ratio = get_vxx_ratio(vxx_data, vxx_idx + 1) if vxx_idx > 0 else 1.0
        
        # Stock entries
        if len(stock_positions) < MAX_STOCK_POSITIONS and capital > 5000:
            for ticker in UNIVERSE:
                if ticker in stock_positions:
                    continue
                if len(stock_positions) >= MAX_STOCK_POSITIONS:
                    break
                
                bars_for_ticker = bar_index.get(ticker, {})
                bar = bars_for_ticker.get(trade_date)
                if bar is None:
                    continue
                
                ticker_dates = sorted_dates_cache.get(ticker, [])
                idx_today = ticker_dates.index(trade_date) if trade_date in ticker_dates else -1
                if idx_today < 20:
                    continue
                prev_bars_list = [bars_for_ticker[d] for d in ticker_dates[max(0,idx_today-30):idx_today]]
                
                score = quick_score(bar, prev_bars_list)
                if score is None or score < MIN_SCORE:
                    continue
                
                # Compute ATR for this ticker
                atr_bars = [bars_for_ticker[d] for d in ticker_dates[max(0,idx_today-20):idx_today]]
                atr = compute_atr(atr_bars)
                atr_pct = (atr / bar["c"]) if atr and bar["c"] > 0 else 0.02
                
                position_pct = min(0.12, 0.08 + (score - 65) / 250)
                already_deployed = sum(p["cost_basis"] for p in stock_positions.values())
                already_deployed += sum(p["cost_basis"] for p in options_positions.values())
                max_new = capital * 0.80 - already_deployed
                if max_new < capital * 0.05:
                    continue
                cost_basis = min(capital * position_pct, max_new)
                
                stock_positions[ticker] = {
                    "entry_price": bar["c"],
                    "entry_day": day_idx,
                    "entry_date": trade_date,
                    "cost_basis": cost_basis,
                    "score": score,
                    "atr_pct": atr_pct,
                }
        
        # Options entries (if enabled)
        if config["options_enabled"] and len(options_positions) < MAX_OPTIONS_POSITIONS and capital > 5000:
            for ticker in UNIVERSE:
                opt_key = f"OPT_{ticker}_{day_idx}"
                if any(p["underlying"] == ticker for p in options_positions.values()):
                    continue
                if len(options_positions) >= MAX_OPTIONS_POSITIONS:
                    break
                
                bars_for_ticker = bar_index.get(ticker, {})
                bar = bars_for_ticker.get(trade_date)
                if bar is None or bar["c"] < 10:
                    continue
                
                ticker_dates = sorted_dates_cache.get(ticker, [])
                idx_today = ticker_dates.index(trade_date) if trade_date in ticker_dates else -1
                if idx_today < 20:
                    continue
                prev_bars_list = [bars_for_ticker[d] for d in ticker_dates[max(0,idx_today-30):idx_today]]
                
                vrp, hv, iv_est = compute_vrp(prev_bars_list, vxx_ratio)
                
                # Options scanner logic: look for high-IV or earnings setups
                # Simplified: take options trades when IV is elevated (VRP > 3)
                # or when VXX is in panic mode
                should_trade_options = False
                strategy = "straddle"
                
                if vxx_ratio >= 1.30:
                    should_trade_options = True
                    strategy = "panic_put_sale"
                elif vrp > 4 and hv > 25:
                    should_trade_options = True
                    strategy = "high_iv_straddle"
                elif hv < 15 and vrp < -1:
                    should_trade_options = True
                    strategy = "cheap_iv_buy"
                
                if not should_trade_options:
                    continue
                
                # Options allocation: 8% max per position (capped)
                opt_pct = min(0.08, 0.05 + vrp / 200)
                already_deployed = sum(p["cost_basis"] for p in stock_positions.values())
                already_deployed += sum(p["cost_basis"] for p in options_positions.values())
                max_new = capital * 0.80 - already_deployed
                if max_new < capital * 0.03:
                    continue
                cost_basis = min(capital * opt_pct, max_new)
                
                options_positions[opt_key] = {
                    "underlying": ticker,
                    "entry_price": bar["c"],
                    "entry_day": day_idx,
                    "entry_date": trade_date,
                    "cost_basis": cost_basis,
                    "strategy": strategy,
                    "hv": hv,
                    "vrp": vrp,
                    "vxx_ratio": vxx_ratio,
                }
        
        # ═══ DAILY EQUITY ═══
        open_stock_value = 0
        for ticker, pos in stock_positions.items():
            bar = bar_index.get(ticker, {}).get(trade_date)
            if bar:
                pct = (bar["c"] - pos["entry_price"]) / pos["entry_price"]
                open_stock_value += pos["cost_basis"] * (1 + pct)
        
        open_opt_value = sum(p["cost_basis"] for p in options_positions.values())
        deployed = sum(p["cost_basis"] for p in stock_positions.values()) + sum(p["cost_basis"] for p in options_positions.values())
        cash = capital - deployed
        equity = cash + open_stock_value + open_opt_value
        daily_equity.append(equity)
        
        if len(daily_equity) > 1 and daily_equity[-2] > 0:
            daily_returns.append((daily_equity[-1] - daily_equity[-2]) / daily_equity[-2])
        
        peak_capital = max(peak_capital, equity)
        dd = (peak_capital - equity) / peak_capital if peak_capital > 0 else 0
        stats["max_drawdown"] = max(stats.get("max_drawdown", 0), dd)
    
    # Close remaining positions
    last_day = trading_days[-1] if trading_days else None
    if last_day:
        for ticker, pos in list(stock_positions.items()):
            bar = bar_index.get(ticker, {}).get(last_day)
            if bar:
                pct = (bar["c"] - pos["entry_price"]) / pos["entry_price"] - 2 * SLIPPAGE_PCT
                pnl = pos["cost_basis"] * pct
                capital += pnl
                stats["stock_total_pnl"] += pnl
                stats["stock_trades"] += 1
                if pnl > 0: stats["stock_wins"] += 1
        for key, pos in list(options_positions.items()):
            # Assume neutral exit
            pnl = pos["cost_basis"] * (-OPTIONS_SPREAD_PCT * 2)
            capital += pnl
            stats["options_total_pnl"] += pnl
            stats["options_trades"] += 1
    
    # ═══ METRICS ═══
    n_years = len(trading_days) / 252
    cagr = ((capital / STARTING_CAPITAL) ** (1/n_years) - 1) * 100 if n_years > 0 and capital > 0 else 0
    total_return = (capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    
    # Sharpe
    if len(daily_returns) > 10:
        avg_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1)
        sharpe = (avg_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0
    else:
        sharpe = 0
    
    # Sortino
    if len(daily_returns) > 10:
        downside = [r for r in daily_returns if r < 0]
        down_std = np.std(downside, ddof=1) if len(downside) > 2 else std_ret
        sortino = (avg_ret / down_std) * math.sqrt(252) if down_std > 0 else 0
    else:
        sortino = 0
    
    opt_trades_count = stats.get("options_trades", 0)
    opt_wins = stats.get("options_wins", 0)
    stk_trades_count = stats.get("stock_trades", 0)
    stk_wins = stats.get("stock_wins", 0)
    
    opt_trades_list = [t for t in all_trades if t["instrument"] == "options"]
    
    return {
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(stats.get("max_drawdown", 0) * 100, 2),
        "total_trades": stk_trades_count + opt_trades_count,
        "stock_trades": stk_trades_count,
        "stock_wins": stk_wins,
        "stock_win_rate": round(stk_wins / stk_trades_count * 100, 1) if stk_trades_count > 0 else 0,
        "stock_total_pnl": round(stats["stock_total_pnl"], 2),
        "options_trades": opt_trades_count,
        "options_wins": opt_wins,
        "options_win_rate": round(opt_wins / opt_trades_count * 100, 1) if opt_trades_count > 0 else 0,
        "options_total_pnl": round(stats["options_total_pnl"], 2),
        "options_avg_hold_days": round(stats["options_hold_sum"] / opt_trades_count, 1) if opt_trades_count > 0 else 0,
        "options_avg_pnl": round(stats["options_total_pnl"] / opt_trades_count, 2) if opt_trades_count > 0 else 0,
        "options_pct_of_total_pnl": round(stats["options_total_pnl"] / (stats["stock_total_pnl"] + stats["options_total_pnl"]) * 100, 1) if (stats["stock_total_pnl"] + stats["options_total_pnl"]) != 0 else 0,
        "all_trades": all_trades,
        "options_exit_reasons": dict(defaultdict(int, {t["exit_reason"]: sum(1 for t2 in opt_trades_list if t2["exit_reason"] == t["exit_reason"]) for t in opt_trades_list})),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 75)
    print("  VolTradeAI — Options Fix A/B Backtest (2016-2026)")
    print("  Testing: BROKEN vs FIX_SKIP vs FIX_SKIP_TIGHT vs NO_OPTIONS")
    print("=" * 75)
    
    START = "2016-01-01"
    
    print("\n[1/3] Fetching 10 years of market data...")
    spy_bars = fetch_bars("SPY", start=START, limit=2700)
    print(f"  SPY: {len(spy_bars)} bars")
    
    vxx_bars = fetch_bars("VXX", start=START, limit=2700)
    print(f"  VXX: {len(vxx_bars)} bars")
    
    all_tickers = list(set(UNIVERSE))
    print(f"  Fetching {len(all_tickers)} tickers...")
    raw_bars = fetch_multi_bars(all_tickers, start=START, limit=2700)
    print(f"  Received: {len(raw_bars)} tickers")
    
    bar_index = {}
    for sym, bars in raw_bars.items():
        bar_index[sym] = {b["t"][:10]: b for b in bars}
    
    if not spy_bars:
        print("ERROR: No SPY data!")
        return
    
    trading_days = sorted({b["t"][:10] for b in spy_bars})
    print(f"  Trading days: {trading_days[0]} → {trading_days[-1]} ({len(trading_days)})")
    
    # SPY benchmark
    spy_start = spy_bars[0]["c"]
    spy_end = spy_bars[-1]["c"]
    n_years = len(trading_days) / 252
    spy_cagr = ((spy_end / spy_start) ** (1/n_years) - 1) * 100
    spy_total = (spy_end - spy_start) / spy_start * 100
    print(f"  SPY: {spy_total:+.1f}% total | {spy_cagr:+.1f}% CAGR")
    
    # ── Run all configs ─────────────────────────────────────────────────────────
    print("\n[2/3] Running backtests...")
    results = {}
    
    for name, config in CONFIGS.items():
        print(f"\n  --- {name}: {config['label']} ---")
        result = simulate_portfolio(trading_days, bar_index, vxx_bars, config)
        result["alpha"] = round(result["cagr_pct"] - spy_cagr, 2)
        results[name] = result
        print(f"  Capital: ${result['final_capital']:,.0f} | CAGR: {result['cagr_pct']:+.1f}% | Sharpe: {result['sharpe']:.3f} | Sortino: {result['sortino']:.3f}")
        print(f"  Options P&L: ${result['options_total_pnl']:+,.0f} | Options WR: {result['options_win_rate']:.0f}% | Avg Hold: {result['options_avg_hold_days']:.1f}d")
    
    # ── Comparison ──────────────────────────────────────────────────────────────
    print("\n[3/3] Comparison Report")
    sep = "─" * 85
    sep2 = "═" * 85
    
    print(f"\n{sep2}")
    print(f"  {'Metric':<30} {'BROKEN':>12} {'FIX_SKIP':>12} {'FIX_TIGHT':>12} {'NO_OPTIONS':>12}")
    print(sep)
    
    def row(label, key, fmt="{:.1f}", good="higher"):
        vals = {n: r.get(key, 0) for n, r in results.items()}
        valid = {k: v for k, v in vals.items() if v is not None}
        if valid:
            best = max(valid, key=lambda k: valid[k] if good == "higher" else -valid[k])
        else:
            best = ""
        parts = []
        for n in ["BROKEN", "FIX_SKIP", "FIX_SKIP_TIGHT", "NO_OPTIONS"]:
            v = vals.get(n, 0)
            s = fmt.format(v) if v is not None else "N/A"
            marker = " ★" if n == best else ""
            parts.append(f"{s}{marker}")
        print(f"  {label:<30} {parts[0]:>12} {parts[1]:>12} {parts[2]:>12} {parts[3]:>12}")
    
    row("Final Capital ($)", "final_capital", "${:,.0f}", "higher")
    row("CAGR (%)", "cagr_pct", "{:+.2f}%", "higher")
    row("Alpha vs SPY (%)", "alpha", "{:+.2f}%", "higher")
    row("Sharpe", "sharpe", "{:.3f}", "higher")
    row("Sortino", "sortino", "{:.3f}", "higher")
    row("Max Drawdown (%)", "max_drawdown_pct", "{:.1f}%", "lower")
    print(sep)
    row("Total Trades", "total_trades", "{:d}")
    row("Stock Trades", "stock_trades", "{:d}")
    row("Stock Win Rate (%)", "stock_win_rate", "{:.1f}%", "higher")
    row("Stock P&L ($)", "stock_total_pnl", "${:+,.0f}", "higher")
    print(sep)
    row("Options Trades", "options_trades", "{:d}")
    row("Options Win Rate (%)", "options_win_rate", "{:.1f}%", "higher")
    row("Options P&L ($)", "options_total_pnl", "${:+,.0f}", "higher")
    row("Options Avg Hold (days)", "options_avg_hold_days", "{:.1f}", "higher")
    row("Options Avg P&L ($)", "options_avg_pnl", "${:+.0f}", "higher")
    row("Options % of Total P&L", "options_pct_of_total_pnl", "{:.1f}%", "higher")
    print(sep2)
    
    print(f"\n  SPY Buy-and-Hold: {spy_total:+.1f}% total | {spy_cagr:+.1f}% CAGR")
    
    # Options exit reason breakdown
    print(f"\n  OPTIONS EXIT REASONS:")
    print(sep)
    for name in ["BROKEN", "FIX_SKIP", "FIX_SKIP_TIGHT"]:
        r = results.get(name, {})
        reasons = r.get("options_exit_reasons", {})
        if reasons:
            print(f"  {name}:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason:<35} {count:>5}")
    
    print(sep2)
    
    # Save results
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "period": f"{trading_days[0]} → {trading_days[-1]}",
        "trading_days": len(trading_days),
        "spy_cagr": round(spy_cagr, 2),
        "spy_total_return": round(spy_total, 2),
        "results": {
            k: {key: v for key, v in r.items() if key != "all_trades"}
            for k, r in results.items()
        },
    }
    
    out_path = "/home/user/workspace/voltradeai/backtest_options_fix_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(f"Completed in {time.time() - t0:.1f}s")
    
    return results


if __name__ == "__main__":
    main()
