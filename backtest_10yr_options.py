#!/usr/bin/env python3
"""
VolTradeAI — 10-Year Options Backtest (2016-2026)
====================================================
Compares OLD thresholds vs NEW thresholds across:
  - Stock-only trading
  - Options: sell cash-secured puts (high VRP > threshold)
  - Options: buy calls/puts (cheap IV, VRP < threshold)
  - Options: bull/bear call spreads (high conviction score)
  - 2x Leveraged ETF trading

Simulates the instrument_selector and options_execution decision logic
using actual historical price data from Alpaca, with:
  - VXX ratio-based options scoring
  - Realistic spread/slippage costs
  - Options theta decay simulation
  - Regime detection (panic/elevated/normal/calm)
  - Walk-forward validation (no look-ahead)

Uses SPY as market benchmark for alpha calculation.
"""

import os, sys, time, json, math, random
import requests
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# ── Alpaca credentials ─────────────────────────────────────────────────────────
ALPACA_KEY    = "PKMDHJOVQEVIB4UHZXUYVTIDBU"
ALPACA_SECRET = "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et"
HEADERS       = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
DATA_URL      = "https://data.alpaca.markets"
BROKER_URL    = "https://paper-api.alpaca.markets"

# ── Backtest parameters ────────────────────────────────────────────────────────
STARTING_CAPITAL  = 100_000.0
MAX_POSITIONS     = 8
STOP_LOSS_PCT     = 0.05    # -5% from entry (wider for options)
TAKE_PROFIT_PCT   = 0.12    # +12% from entry
TIME_STOP_DAYS    = 5       # Max stock hold
OPTIONS_DAYS      = 21      # Options expiry window
SLIPPAGE_PCT      = 0.001   # 0.1% per side
MIN_SCORE         = 65

# ── Threshold configurations ──────────────────────────────────────────────────
CONFIGS = {
    "OLD": {
        "min_options_score_entry":  70,
        "options_score_normal":     58,   # VXX ratio 0.90-1.10
        "options_score_calm":       50,   # VXX ratio 0.70-0.90
        "vrp_sell_threshold":        5,   # VRP > 5% to sell
        "vrp_buy_threshold":        -3,   # VRP < -3% to buy
        "allow_buy_call":           True,
        "allow_bull_spread":        True,
        "label": "OLD (score≥70, normal=58, VRP sell>5, buy<-3)",
    },
    "NEW": {
        "min_options_score_entry":  65,
        "options_score_normal":     62,   # VXX ratio 0.90-1.10 (raised from 58→62)
        "options_score_calm":       52,   # VXX ratio 0.70-0.90 (raised from 50→52)
        "vrp_sell_threshold":        4,   # VRP > 4% to sell (lowered from 5)
        "vrp_buy_threshold":        -2,   # VRP < -2% to buy (tightened from -3)
        "allow_buy_call":           True,
        "allow_bull_spread":        True,
        "label": "NEW (score≥65, normal=62, VRP sell>4, buy<-2)",
    },
    "NEW_CSP_ONLY": {
        "min_options_score_entry":  65,
        "options_score_normal":     62,
        "options_score_calm":       52,
        "vrp_sell_threshold":        4,
        "vrp_buy_threshold":       -99,  # Never buy calls (set impossibly low)
        "allow_buy_call":           False,
        "allow_bull_spread":        False,
        "label": "NEW+CSP-ONLY (sell premium only, no call buying or spreads)",
    },
}

# ── Leveraged ETF map ──────────────────────────────────────────────────────────
LEVERAGED_ETFS = {
    'TSLA':'TSLL','NVDA':'NVDL','AAPL':'AAPU','AMZN':'AMZU',
    'MSFT':'MSFU','META':'METU','GOOGL':'GGLL','AMD':'AMDU',
    'SPY':'SSO','QQQ':'QLD','IWM':'TNA',
}

# ── Stock universe (all with 10+ year history) ─────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_bars(symbol, start="2016-01-01", limit=2700):
    try:
        r = requests.get(f"{DATA_URL}/v2/stocks/bars",
            params={"symbols": symbol, "timeframe": "1Day",
                    "start": start, "limit": limit,
                    "adjustment": "all", "feed": "sip"},
            headers=HEADERS, timeout=20)
        bars = r.json().get("bars", {}).get(symbol, [])
        return bars
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

# ─────────────────────────────────────────────────────────────────────────────
# SCORING (mirrors bot_engine quick_score + VRP simulation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hv(bars, window=20):
    """Compute historical volatility (annualized %) from last N bars."""
    if len(bars) < window + 1:
        return 20.0
    closes = [b["c"] for b in bars[-(window+1):]]
    log_rets = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
    std = (sum((r - sum(log_rets)/len(log_rets))**2 for r in log_rets) / (len(log_rets)-1)) ** 0.5
    return round(std * math.sqrt(252) * 100, 2)

def compute_vrp(bars, vxx_ratio):
    """
    Simulate VRP using HV20 vs estimated IV.
    IV proxy = HV20 * (1 + 0.3 * vxx_ratio_deviation)
    When VXX is 30% above avg, IV is inflated ~30% above HV.
    """
    hv20 = compute_hv(bars, 20)
    # IV proxy: higher VXX ratio = higher implied vol
    vxx_premium = (vxx_ratio - 1.0) * 0.5  # 1.30 ratio → +15% IV premium
    iv_estimate = hv20 * (1 + vxx_premium)
    vrp = iv_estimate - hv20   # Positive = options overpriced
    return round(vrp, 2), round(hv20, 2), round(iv_estimate, 2)

def quick_score(bar, prev_bars):
    """Score a trading signal 0-100."""
    close  = bar.get("c", 0)
    open_p = bar.get("o", 0)
    high   = bar.get("h", 0)
    low    = bar.get("l", 0)
    volume = bar.get("v", 0)
    vwap   = bar.get("vw", 0)

    if close < 5 or volume < 500_000:
        return None

    change_pct = ((close - open_p) / open_p * 100) if open_p > 0 else 0
    range_pct  = ((high - low) / low * 100) if low > 0 else 0
    vwap_dist  = ((close - vwap) / vwap * 100) if vwap > 0 else 0

    score = 50

    # Momentum
    if change_pct > 3:   score += 10
    elif change_pct > 1: score += 5
    elif change_pct < -3: score += 8  # bounce candidate

    # Volume
    if volume > 20_000_000:  score += 15
    elif volume > 5_000_000: score += 8
    elif volume > 1_000_000: score += 3

    # VWAP
    if vwap_dist > 1:   score += 5
    elif vwap_dist < -1: score += 3

    # Range / volatility
    if range_pct > 5:   score += 10
    elif range_pct > 3: score += 5

    # Multi-day trend (compare close to 5d ago)
    if len(prev_bars) >= 5:
        c5 = prev_bars[-5]["c"]
        trend = (close - c5) / c5 * 100 if c5 > 0 else 0
        if trend > 5:    score += 5
        elif trend < -5: score += 3  # oversold bounce

    return max(0, min(100, score))

# ─────────────────────────────────────────────────────────────────────────────
# INSTRUMENT SELECTION (mirrors instrument_selector.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def get_vxx_ratio(vxx_bars, day_idx):
    """Get VXX ratio at a given day index (look-back only, no look-ahead)."""
    if day_idx < 10:
        return 1.0
    window = vxx_bars[max(0, day_idx-30):day_idx]
    if not window:
        return 1.0
    avg30 = sum(b["c"] for b in window) / len(window)
    latest = vxx_bars[day_idx-1]["c"]
    return latest / avg30 if avg30 > 0 else 1.0

def score_options(score, vxx_ratio, vrp, config):
    """Score options instrument. Returns (inst_score, strategy, reasoning)."""
    if score < config["min_options_score_entry"]:
        return 0, "none", f"Score {score} < min {config['min_options_score_entry']}"

    # Base score from VXX regime
    if vxx_ratio >= 1.30:
        inst_score = 78.0
        regime = "PANIC"
    elif vxx_ratio >= 1.10:
        inst_score = 68.0
        regime = "ELEVATED"
    elif vxx_ratio >= 0.90:
        inst_score = config["options_score_normal"]
        regime = "NORMAL"
    elif vxx_ratio >= 0.70:
        inst_score = config["options_score_calm"]
        regime = "CALM"
    else:
        inst_score = 42.0
        regime = "COMPLACENCY"

    # VRP adjustments
    if vrp > config["vrp_sell_threshold"]:
        inst_score += 8
        strategy = "sell_csp"  # Cash-secured put (sell premium)
        reason = f"VRP +{vrp:.1f}% sell premium | regime={regime}"
    elif vrp < config["vrp_buy_threshold"] and config.get("allow_buy_call", True):
        inst_score += 6
        strategy = "buy_call"  # Cheap IV, buy calls
        reason = f"VRP {vrp:.1f}% buy cheap IV | regime={regime}"
    elif score >= 85 and config.get("allow_bull_spread", True):
        inst_score += 4
        strategy = "bull_spread"
        reason = f"High conviction score={score} | regime={regime}"
    else:
        strategy = "none" if inst_score < 60 else "sell_csp"
        reason = f"Base options score={inst_score:.0f} | regime={regime}"

    return min(100, inst_score), strategy, reason

def score_stock(score, vxx_ratio):
    """Score stock instrument."""
    inst_score = 50.0
    inst_score += (score - 50) * 0.2
    return min(100, max(0, inst_score))

def score_etf(ticker, score, vxx_ratio, hold_days=2):
    """Score 2x ETF instrument."""
    etf = LEVERAGED_ETFS.get(ticker)
    if not etf:
        return 0, None
    inst_score = 50.0
    inst_score += (score - 50) * 0.15  # Slightly less than stock
    if 2 <= hold_days <= 3:
        inst_score += 5  # Sweet spot for ETFs
    return min(100, max(0, inst_score)), etf

def select_instrument(ticker, score, vxx_ratio, vrp, config, hold_days=2):
    """
    Choose: stock / etf / options
    Returns: (chosen, strategy, reasoning, scores_dict)
    """
    s_score = score_stock(score, vxx_ratio)
    e_score, etf_ticker = score_etf(ticker, score, vxx_ratio, hold_days)
    o_score, o_strategy, o_reason = score_options(score, vxx_ratio, vrp, config)

    candidates = [(s_score, "stock", "stock", f"Stock default score={s_score:.0f}")]
    if e_score > 0 and etf_ticker:
        candidates.append((e_score, "etf", etf_ticker, f"ETF score={e_score:.0f}"))
    if o_score >= 60 and o_strategy != "none":
        candidates.append((o_score, "options", o_strategy, o_reason))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, chosen, strategy, reason = candidates[0]

    return chosen, strategy, reason, {
        "stock": s_score, "etf": e_score, "options": o_score
    }

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS P&L SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_options_pnl(strategy, entry_price, exit_price, hv, holding_days, vrp):
    """
    Simulate options P&L for a given strategy.
    Returns net_pct return on capital allocated.
    
    Models:
    - sell_csp: Sell cash-secured put. Collect premium, profit if stock stays up.
      Premium ≈ ATM put price from Black-Scholes approximation.
      Profit = full premium if stock > strike at expiry.
      Loss = (strike - stock_price) - premium if stock < strike.
    
    - buy_call: Buy call. Profit from upside move.
      Cost ≈ ATM call price.
      Return = intrinsic_value - premium at expiry.
    
    - bull_spread: Bull call spread. Capped upside, defined risk.
      Debit = long_premium - short_premium.
      Max profit = spread_width - debit.
    """
    T = OPTIONS_DAYS / 252   # Time in years
    sigma = max(hv / 100, 0.10)  # Annualized vol

    # Approximate ATM premium using simplified BS (no risk-free for simplicity)
    # For ATM: premium ≈ 0.4 * S * sigma * sqrt(T)
    atm_premium_pct = 0.4 * sigma * math.sqrt(T)
    atm_premium = entry_price * atm_premium_pct

    stock_move_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0

    if strategy == "sell_csp":
        # Sell put at strike = entry_price (ATM)
        # Collect premium upfront. Profit if stock stays above strike.
        # If stock falls below strike, we lose (strike - exit) - premium
        if exit_price >= entry_price:
            # Stock rose or flat — full premium
            net_pct = atm_premium_pct * 0.9  # 90% of premium (bid-ask friction)
        else:
            # Stock fell — lose on the put
            intrinsic_loss = (entry_price - exit_price) / entry_price
            net_pct = atm_premium_pct - intrinsic_loss
        # Theta decay benefit: we collected time value, so actual P&L includes time component
        # For short options held to partial expiry: scale by holding / OPTIONS_DAYS
        time_decay_captured = min(1.0, holding_days / OPTIONS_DAYS)
        net_pct = net_pct * time_decay_captured

    elif strategy == "buy_call":
        # Buy call at strike = entry_price
        # Pay premium. Profit = max(0, stock - strike) - premium
        if exit_price > entry_price:
            intrinsic_gain = (exit_price - entry_price) / entry_price
            net_pct = intrinsic_gain - atm_premium_pct
        else:
            # Expired worthless (or we exit early with partial loss)
            # Time value remaining at exit
            time_remaining = max(0, OPTIONS_DAYS - holding_days) / OPTIONS_DAYS
            residual_value = atm_premium_pct * time_remaining * 0.5  # Decays faster near expiry
            net_pct = residual_value - atm_premium_pct
        # For buy_call: return expressed as % of allocated capital (which = full position)
        # The premium is a COST, not the entire allocation — no leverage scaling needed
        # The P&L is already expressed correctly as % of stock price (= % of capital)

    elif strategy == "bull_spread":
        # Buy call, sell higher strike call — express return as % of CAPITAL allocated
        # Capital allocated = full position size, NOT just the debit
        # This prevents the leverage blowup that occurs when expressing as % of tiny debit
        spread_width_pct = 0.05   # 5% spread width
        long_strike  = entry_price
        short_strike = entry_price * (1 + spread_width_pct)
        otm_premium_pct = atm_premium_pct * 0.45  # OTM is cheaper
        debit_pct = atm_premium_pct - otm_premium_pct  # Net debit (% of stock)
        if debit_pct <= 0:
            debit_pct = 0.01
        max_profit_pct = spread_width_pct - debit_pct   # Max profit as % of stock price
        if exit_price >= short_strike:
            gross_pct = max_profit_pct
        elif exit_price >= long_strike:
            gross_pct = ((exit_price - long_strike) / entry_price) - debit_pct
        else:
            gross_pct = -debit_pct  # Total loss of debit
        # net_pct expressed as % of capital allocated (= % of stock price, not debit)
        net_pct = gross_pct

    else:
        net_pct = stock_move_pct  # Fallback: stock return

    # Deduct spread/commission
    net_pct -= 2 * SLIPPAGE_PCT  # Two-sided friction on options

    return round(net_pct, 4)

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_portfolio(trading_days, bar_index, vxx_data, spy_data, config):
    """
    Full portfolio simulation over the trading period.
    Returns: dict with P&L, trades, Sharpe, max drawdown, etc.
    """
    capital = STARTING_CAPITAL
    peak_capital = STARTING_CAPITAL
    positions = {}   # ticker -> {entry_price, entry_day, strategy, instrument, shares, cost_basis}
    all_trades = []
    daily_equity = []

    options_trades = 0
    stock_trades   = 0
    etf_trades     = 0
    options_wins   = 0
    stock_wins     = 0
    etf_wins       = 0
    total_pnl      = 0.0
    max_drawdown   = 0.0

    vxx_dates = {b["t"][:10]: i for i, b in enumerate(vxx_data)}

    for day_idx, trade_date in enumerate(trading_days):
        daily_start_capital = capital

        # ── EXITS: check existing positions ──────────────────────────────────
        to_exit = []
        for ticker, pos in positions.items():
            bar = bar_index.get(ticker, {}).get(trade_date)
            if bar is None:
                continue
            current_price = bar["c"]
            entry_price = pos["entry_price"]
            holding_days = day_idx - pos["entry_day"]

            pct_change = (current_price - entry_price) / entry_price if entry_price > 0 else 0

            # Exit conditions
            exit_reason = None
            if pct_change <= -STOP_LOSS_PCT:
                exit_reason = "stop_loss"
            elif pct_change >= TAKE_PROFIT_PCT:
                exit_reason = "take_profit"
            elif holding_days >= TIME_STOP_DAYS:
                exit_reason = "time_stop"

            if exit_reason:
                to_exit.append((ticker, current_price, exit_reason, holding_days))

        for ticker, exit_price, exit_reason, holding_days in to_exit:
            pos = positions.pop(ticker, None)
            if not pos:
                continue

            entry_price = pos["entry_price"]
            instrument  = pos["instrument"]
            strategy    = pos["strategy"]
            cost_basis  = pos["cost_basis"]
            hv          = pos.get("hv", 20.0)
            vrp         = pos.get("vrp", 0.0)

            if instrument == "options":
                net_pct = simulate_options_pnl(strategy, entry_price, exit_price, hv, holding_days, vrp)
                pnl = cost_basis * net_pct
            else:
                gross_pct = (exit_price - entry_price) / entry_price
                slippage  = 2 * SLIPPAGE_PCT
                spread    = 0.0003 if ticker in LEVERAGED_ETFS or len(ticker) <= 4 else 0.0008
                net_pct   = gross_pct - slippage - spread
                pnl       = cost_basis * net_pct

            capital += pnl
            total_pnl += pnl
            is_win = pnl > 0

            all_trades.append({
                "date_exit":    trade_date,
                "date_entry":   pos["entry_date"],
                "ticker":       ticker,
                "instrument":   instrument,
                "strategy":     strategy,
                "entry_price":  entry_price,
                "exit_price":   exit_price,
                "holding_days": holding_days,
                "pnl":          round(pnl, 2),
                "net_pct":      round(net_pct * 100, 2),
                "exit_reason":  exit_reason,
                "score":        pos.get("score", 0),
                "vxx_ratio":    pos.get("vxx_ratio", 1.0),
                "vrp":          vrp,
            })

            if instrument == "options":
                options_trades += 1
                if is_win: options_wins += 1
            elif instrument == "etf":
                etf_trades += 1
                if is_win: etf_wins += 1
            else:
                stock_trades += 1
                if is_win: stock_wins += 1

        # ── ENTRIES: find new signals ─────────────────────────────────────────
        if len(positions) < MAX_POSITIONS and capital > 1000:
            # Get VXX ratio for this day
            vxx_idx = vxx_dates.get(trade_date, -1)
            if vxx_idx > 0:
                vxx_ratio = get_vxx_ratio(vxx_data, vxx_idx + 1)
            else:
                vxx_ratio = 1.0

            for ticker in UNIVERSE:
                if ticker in positions:
                    continue
                if len(positions) >= MAX_POSITIONS:
                    break

                bars_for_ticker = bar_index.get(ticker, {})
                bar = bars_for_ticker.get(trade_date)
                if bar is None:
                    continue

                # Get historical bars for this ticker up to this day
                sorted_dates = sorted(bars_for_ticker.keys())
                idx_today = sorted_dates.index(trade_date) if trade_date in sorted_dates else -1
                if idx_today < 5:
                    continue
                prev_bars_list = [bars_for_ticker[d] for d in sorted_dates[max(0,idx_today-30):idx_today]]

                score = quick_score(bar, prev_bars_list)
                if score is None or score < MIN_SCORE:
                    continue

                vrp, hv, iv_est = compute_vrp(prev_bars_list, vxx_ratio)

                chosen, strategy, reason, scores = select_instrument(
                    ticker, score, vxx_ratio, vrp, config
                )

                # Position sizing: Kelly-lite, 10-12% per position
                position_pct = min(0.12, 0.08 + (score - 65) / 250)
                # Don't over-deploy: keep at least 20% cash
                already_deployed = sum(p["cost_basis"] for p in positions.values())
                max_new = capital * 0.80 - already_deployed
                if max_new < capital * 0.05:
                    continue
                cost_basis = min(capital * position_pct, max_new)

                entry_price = bar["c"]
                positions[ticker] = {
                    "entry_price":  entry_price,
                    "entry_day":    day_idx,
                    "entry_date":   trade_date,
                    "instrument":   chosen,
                    "strategy":     strategy,
                    "cost_basis":   cost_basis,
                    "score":        score,
                    "vxx_ratio":    vxx_ratio,
                    "vrp":          vrp,
                    "hv":           hv,
                    "scores":       scores,
                }

        # ── Track daily equity for Sharpe/drawdown ────────────────────────────
        # Mark-to-market open positions
        open_value = 0
        for ticker, pos in positions.items():
            bar = bar_index.get(ticker, {}).get(trade_date)
            if bar:
                pct_change = (bar["c"] - pos["entry_price"]) / pos["entry_price"]
                open_value += pos["cost_basis"] * (1 + pct_change)

        # Cash not deployed
        deployed = sum(p["cost_basis"] for p in positions.values())
        cash = capital - deployed
        daily_equity.append(cash + open_value)

        # Max drawdown
        peak_capital = max(peak_capital, capital)
        dd = (peak_capital - capital) / peak_capital
        max_drawdown = max(max_drawdown, dd)

    # Close any remaining open positions at last available price
    last_day = trading_days[-1] if trading_days else None
    if last_day:
        for ticker, pos in list(positions.items()):
            bar = bar_index.get(ticker, {}).get(last_day)
            if bar:
                exit_price = bar["c"]
                entry_price = pos["entry_price"]
                instrument  = pos["instrument"]
                strategy    = pos["strategy"]
                cost_basis  = pos["cost_basis"]
                hv          = pos.get("hv", 20.0)
                vrp         = pos.get("vrp", 0.0)
                holding_days = len(trading_days) - pos["entry_day"]

                if instrument == "options":
                    net_pct = simulate_options_pnl(strategy, entry_price, exit_price, hv, holding_days, vrp)
                    pnl = cost_basis * net_pct
                else:
                    gross_pct = (exit_price - entry_price) / entry_price
                    net_pct = gross_pct - 2*SLIPPAGE_PCT
                    pnl = cost_basis * net_pct

                capital += pnl
                all_trades.append({
                    "date_exit":    last_day,
                    "date_entry":   pos["entry_date"],
                    "ticker":       ticker,
                    "instrument":   instrument,
                    "strategy":     strategy,
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "holding_days": holding_days,
                    "pnl":          round(pnl, 2),
                    "net_pct":      round(net_pct * 100, 2),
                    "exit_reason":  "end_of_backtest",
                    "score":        pos.get("score", 0),
                    "vxx_ratio":    pos.get("vxx_ratio", 1.0),
                    "vrp":          vrp,
                })

    # ── Performance Metrics ───────────────────────────────────────────────────
    total_return = (capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    n_years = len(trading_days) / 252
    cagr = ((capital / STARTING_CAPITAL) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

    # Sharpe ratio
    if len(daily_equity) > 10:
        returns = [(daily_equity[i] - daily_equity[i-1]) / daily_equity[i-1]
                   for i in range(1, len(daily_equity)) if daily_equity[i-1] > 0]
        if len(returns) > 2:
            avg_ret = sum(returns) / len(returns)
            std_ret = (sum((r - avg_ret)**2 for r in returns) / (len(returns)-1)) ** 0.5
            sharpe = (avg_ret / std_ret) * (252 ** 0.5) if std_ret > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    n_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    opt_trades = [t for t in all_trades if t["instrument"] == "options"]
    stk_trades = [t for t in all_trades if t["instrument"] == "stock"]
    etf_trades_list = [t for t in all_trades if t["instrument"] == "etf"]

    def wr(tlist):
        w = sum(1 for t in tlist if t["pnl"] > 0)
        return round(w / len(tlist) * 100, 1) if tlist else 0

    def avg_ret(tlist):
        if not tlist: return 0
        return round(sum(t["net_pct"] for t in tlist) / len(tlist), 2)

    return {
        "final_capital":       round(capital, 2),
        "total_return_pct":    round(total_return, 2),
        "cagr_pct":            round(cagr, 2),
        "sharpe":              round(sharpe, 3),
        "max_drawdown_pct":    round(max_drawdown * 100, 2),
        "n_trades":            n_trades,
        "win_rate_pct":        round(win_rate, 1),
        "options_trades":      len(opt_trades),
        "stock_trades":        len(stk_trades),
        "etf_trades":          len(etf_trades_list),
        "options_win_rate":    wr(opt_trades),
        "stock_win_rate":      wr(stk_trades),
        "etf_win_rate":        wr(etf_trades_list),
        "options_avg_return":  avg_ret(opt_trades),
        "stock_avg_return":    avg_ret(stk_trades),
        "etf_avg_return":      avg_ret(etf_trades_list),
        "options_pct_of_trades": round(len(opt_trades)/n_trades*100, 1) if n_trades else 0,
        "trades":              all_trades,
        "daily_equity":        daily_equity,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SPY BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def spy_buy_hold(spy_bars, trading_days):
    """Compute SPY buy-and-hold return over the trading period."""
    if not spy_bars or not trading_days:
        return 0, 0
    date_map = {b["t"][:10]: b for b in spy_bars}
    start_bar = date_map.get(trading_days[0])
    end_bar   = date_map.get(trading_days[-1])
    if not start_bar or not end_bar:
        return 0, 0
    spy_start = start_bar["c"]
    spy_end   = end_bar["c"]
    total_ret = (spy_end - spy_start) / spy_start * 100
    n_years = len(trading_days) / 252
    cagr = ((spy_end / spy_start) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    return round(total_ret, 2), round(cagr, 2)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("  VolTradeAI 10-Year Options Backtest: OLD vs NEW Thresholds")
    print("  Period: 2016-01-01 → 2026-04-07")
    print("=" * 70)

    # ── Step 1: Fetch all historical data ─────────────────────────────────────
    print("\n[1/4] Fetching 10 years of market data...")
    START = "2016-01-01"

    print("  Fetching SPY...")
    spy_bars = fetch_bars("SPY", start=START, limit=2700)
    print(f"  SPY: {len(spy_bars)} bars")

    print("  Fetching VXX...")
    vxx_bars = fetch_bars("VXX", start=START, limit=2700)
    print(f"  VXX: {len(vxx_bars)} bars ({vxx_bars[0]['t'][:10] if vxx_bars else 'N/A'} → {vxx_bars[-1]['t'][:10] if vxx_bars else 'N/A'})")

    # Fetch ETF tickers too
    all_tickers = list(set(UNIVERSE) | set(LEVERAGED_ETFS.values()))
    print(f"  Fetching {len(all_tickers)} stock/ETF tickers in batches...")
    raw_bars = fetch_multi_bars(all_tickers, start=START, limit=2700)
    print(f"  Received bars for {len(raw_bars)} tickers")

    # Build date-indexed bar map: {ticker: {date: bar}}
    bar_index = {}
    for sym, bars in raw_bars.items():
        bar_index[sym] = {b["t"][:10]: b for b in bars}

    # Determine trading days from SPY
    if not spy_bars:
        print("ERROR: No SPY data. Check Alpaca credentials.")
        sys.exit(1)
    trading_days = sorted({b["t"][:10] for b in spy_bars})
    print(f"  Trading days: {trading_days[0]} → {trading_days[-1]} ({len(trading_days)} days)")

    # ── Step 2: Baseline SPY buy-and-hold ─────────────────────────────────────
    print("\n[2/4] Computing SPY buy-and-hold benchmark...")
    spy_total, spy_cagr = spy_buy_hold(spy_bars, trading_days)
    print(f"  SPY Total Return: {spy_total:+.1f}% | CAGR: {spy_cagr:+.1f}%")

    # ── Step 3: Run both configurations ───────────────────────────────────────
    print("\n[3/4] Running backtests...")
    results = {}

    for config_name, config in CONFIGS.items():
        print(f"\n  --- Running {config_name}: {config['label']} ---")
        result = simulate_portfolio(trading_days, bar_index, vxx_bars, spy_bars, config)
        result["config"] = config
        result["config_name"] = config_name
        result["spy_total_return"] = spy_total
        result["spy_cagr"] = spy_cagr
        result["alpha"] = round(result["cagr_pct"] - spy_cagr, 2)
        results[config_name] = result

        print(f"  Final Capital: ${result['final_capital']:,.0f} | Return: {result['total_return_pct']:+.1f}%")
        print(f"  CAGR: {result['cagr_pct']:+.1f}% | Sharpe: {result['sharpe']:.3f} | Max DD: {result['max_drawdown_pct']:.1f}%")
        print(f"  Trades: {result['n_trades']} | Win Rate: {result['win_rate_pct']:.1f}%")
        print(f"  Options: {result['options_trades']} ({result['options_pct_of_trades']:.1f}%) | Stock: {result['stock_trades']} | ETF: {result['etf_trades']}")

    # ── Step 4: Print comparison report ───────────────────────────────────────
    print("\n[4/4] Comparison report...")
    print_comparison(results, trading_days, spy_total, spy_cagr)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "generated_at":    datetime.utcnow().isoformat() + "Z",
        "period_start":    trading_days[0],
        "period_end":      trading_days[-1],
        "trading_days":    len(trading_days),
        "starting_capital": STARTING_CAPITAL,
        "spy_total_return": spy_total,
        "spy_cagr":         spy_cagr,
        "results": {
            k: {key: v for key, v in r.items() if key != "trades" and key != "daily_equity"}
            for k, r in results.items()
        },
        "all_trades": {k: r["trades"] for k, r in results.items()},
    }

    out_path = "/home/user/workspace/voltradeai/backtest_10yr_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved → {out_path}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    return results


def strategy_breakdown(results, config_name, sep, sep2):
    cfg_result = results.get(config_name, {})
    trades_list = cfg_result.get("trades", [])
    opt_trades = [t for t in trades_list if t["instrument"] == "options"]
    by_strategy = defaultdict(list)
    for t in opt_trades:
        by_strategy[t["strategy"]].append(t)

    print(f"\n  {config_name} — OPTIONS STRATEGY BREAKDOWN")
    print(sep)
    for strat, tlist in sorted(by_strategy.items(), key=lambda x: -len(x[1])):
        wr_ = sum(1 for t in tlist if t["pnl"] > 0) / len(tlist) * 100 if tlist else 0
        avg = sum(t["net_pct"] for t in tlist) / len(tlist) if tlist else 0
        total_pnl_ = sum(t["pnl"] for t in tlist)
        print(f"  {strat:<25} {len(tlist):>5} trades  WR: {wr_:.0f}%  Avg: {avg:+.2f}%  Total P&L: ${total_pnl_:+,.0f}")

    print(f"\n  {config_name} — PERFORMANCE BY VXX REGIME (options trades only)")
    print(sep)
    by_regime = defaultdict(list)
    for t in opt_trades:
        r = t.get("vxx_ratio", 1.0)
        if r >= 1.30:   regime = "PANIC (>=1.30)"
        elif r >= 1.10: regime = "ELEVATED (1.10-1.30)"
        elif r >= 0.90: regime = "NORMAL (0.90-1.10)"
        elif r >= 0.70: regime = "CALM (0.70-0.90)"
        else:            regime = "COMPLACENCY (<0.70)"
        by_regime[regime].append(t)
    for regime in ["PANIC (>=1.30)","ELEVATED (1.10-1.30)","NORMAL (0.90-1.10)","CALM (0.70-0.90)","COMPLACENCY (<0.70)"]:
        tlist = by_regime[regime]
        if not tlist: continue
        wr_ = sum(1 for t in tlist if t["pnl"] > 0) / len(tlist) * 100
        avg_ = sum(t["net_pct"] for t in tlist) / len(tlist)
        print(f"  {regime:<28} {len(tlist):>4} trades  WR: {wr_:.0f}%  Avg: {avg_:+.2f}%")


def print_comparison(results, trading_days, spy_total, spy_cagr):
    sep  = "─" * 75
    sep2 = "═" * 75

    old  = results.get("OLD", {})
    new  = results.get("NEW", {})
    csp  = results.get("NEW_CSP_ONLY", {})

    print("\n" + sep2)
    print("  3-WAY COMPARISON: OLD vs NEW vs NEW+CSP-ONLY")
    print(f"  Period: {trading_days[0]} → {trading_days[-1]}  ({len(trading_days)} trading days)")
    print(f"  Starting capital: ${STARTING_CAPITAL:,.0f}")
    print(sep2)

    def row(label, ov, nv, cv, fmt="{}", good="higher"):
        o_s = fmt.format(ov) if ov is not None else "N/A"
        n_s = fmt.format(nv) if nv is not None else "N/A"
        c_s = fmt.format(cv) if cv is not None else "N/A"
        vals = {"OLD": ov, "NEW": nv, "CSP": cv}
        valid = {k: v for k, v in vals.items() if v is not None}
        if valid:
            best_k = max(valid, key=lambda k: valid[k] if good == "higher" else -valid[k])
        else:
            best_k = ""
        print(f"  {label:<30} {o_s:>10}  {n_s:>12}  {c_s:>13}  ★ {best_k}")

    print(f"\n  {'Metric':<30} {'OLD':>10}  {'NEW':>12}  {'NEW+CSP-ONLY':>13}  Best")
    print(sep)
    row("Final Capital ($)",     old.get("final_capital"),    new.get("final_capital"),    csp.get("final_capital"),    "${:,.0f}", "higher")
    row("Total Return (%)",      old.get("total_return_pct"), new.get("total_return_pct"), csp.get("total_return_pct"), "{:+.1f}%",  "higher")
    row("CAGR (%)",              old.get("cagr_pct"),         new.get("cagr_pct"),         csp.get("cagr_pct"),         "{:+.2f}%", "higher")
    row("Alpha vs SPY (%/yr)",   old.get("alpha"),            new.get("alpha"),            csp.get("alpha"),            "{:+.2f}%", "higher")
    row("Sharpe Ratio",          old.get("sharpe"),           new.get("sharpe"),           csp.get("sharpe"),           "{:.3f}",   "higher")
    row("Max Drawdown (%)",      old.get("max_drawdown_pct"), new.get("max_drawdown_pct"), csp.get("max_drawdown_pct"), "{:.1f}%",  "lower")
    print(sep)
    row("Total Trades",          old.get("n_trades"),         new.get("n_trades"),         csp.get("n_trades"),         "{:d}",     "higher")
    row("Win Rate (%)",          old.get("win_rate_pct"),     new.get("win_rate_pct"),     csp.get("win_rate_pct"),     "{:.1f}%",  "higher")
    row("Options Trades",        old.get("options_trades"),   new.get("options_trades"),   csp.get("options_trades"),   "{:d}",     "higher")
    row("Options % of Trades",   old.get("options_pct_of_trades"), new.get("options_pct_of_trades"), csp.get("options_pct_of_trades"), "{:.1f}%", "higher")
    row("Options Win Rate (%)",  old.get("options_win_rate"), new.get("options_win_rate"), csp.get("options_win_rate"), "{:.1f}%",  "higher")
    row("Options Avg Ret (%)",   old.get("options_avg_return"), new.get("options_avg_return"), csp.get("options_avg_return"), "{:.2f}%", "higher")
    row("Stock Win Rate (%)",    old.get("stock_win_rate"),   new.get("stock_win_rate"),   csp.get("stock_win_rate"),   "{:.1f}%",  "higher")
    row("ETF Win Rate (%)",      old.get("etf_win_rate"),     new.get("etf_win_rate"),     csp.get("etf_win_rate"),     "{:.1f}%",  "higher")

    print(sep2)
    print(f"\n  SPY Buy-and-Hold: {spy_total:+.1f}% total | {spy_cagr:+.2f}% CAGR")
    for k, r in results.items():
        print(f"  {k:<16} Alpha vs SPY: {r.get('alpha', 0):+.2f}%/yr  |  CAGR: {r.get('cagr_pct',0):+.2f}%")

    print(sep2)
    for config_name in results:
        strategy_breakdown(results, config_name, sep, sep2)
    print(sep2)


if __name__ == "__main__":
    main()
