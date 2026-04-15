#!/usr/bin/env python3
"""
VolTradeAI — Options Fix A/B Backtest v2
==========================================
Uses yfinance for data (no Alpaca credentials needed).

Tests 4 configurations:
  A) BROKEN  — stock exit logic kills option legs (current production bug)
  B) FIX_SKIP — options skip stock exits, managed by options_manager only
  C) FIX_TIGHT — same as B but tighter options exit thresholds
  D) NO_OPTIONS — options disabled entirely (pure stock baseline)
"""

import time, math, json
import numpy as np
import yfinance as yf
from datetime import datetime
from collections import defaultdict

# ── Parameters ────────────────────────────────────────────────────────────────
STARTING_CAPITAL     = 100_000.0
MAX_STOCK_POSITIONS  = 6
MAX_OPTIONS_POSITIONS = 3
STOCK_STOP_ATR_MULT  = 2.0
STOCK_TP_PCT         = 0.12
STOCK_TIME_STOP_DAYS = 10
SLIPPAGE_PCT         = 0.001
OPT_SPREAD_PCT       = 0.03
OPT_DAYS             = 21
MIN_SCORE            = 65

UNIVERSE = [
    "AAPL","MSFT","NVDA","AMD","GOOGL","AMZN","INTC","CRM",
    "JPM","BAC","GS","V","MA","WFC",
    "JNJ","PFE","UNH","MRK","ABBV",
    "XOM","CVX","COP",
    "WMT","COST","HD","TGT","LOW",
    "BA","CAT","GE","HON",
    "TSLA","NFLX","ORCL","ADBE",
]

CONFIGS = {
    "BROKEN": {
        "label": "Current: stock exits kill options legs",
        "options_on": True,
        "skip_stock_exit": False,
        "opt_min_hold_days": 0,
        "opt_profit_target": 0.50,
        "opt_loss_limit": 2.0,
        "broken_atr_default": 0.02,
    },
    "FIX_SKIP": {
        "label": "Fix: skip options in manage_positions",
        "options_on": True,
        "skip_stock_exit": True,
        "opt_min_hold_days": 1,
        "opt_profit_target": 0.50,
        "opt_loss_limit": 2.0,
        "broken_atr_default": None,
    },
    "FIX_TIGHT": {
        "label": "Fix + tighter exits (30% TP, 1.5x loss)",
        "options_on": True,
        "skip_stock_exit": True,
        "opt_min_hold_days": 1,
        "opt_profit_target": 0.30,
        "opt_loss_limit": 1.5,
        "broken_atr_default": None,
    },
    "NO_OPTIONS": {
        "label": "No options (pure stock baseline)",
        "options_on": False,
        "skip_stock_exit": True,
        "opt_min_hold_days": 1,
        "opt_profit_target": 0.50,
        "opt_loss_limit": 2.0,
        "broken_atr_default": None,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Fetch all data via yfinance. Returns (price_data, vxx_data, spy_data, trading_days)."""
    print("  Fetching price data...")
    tickers = list(set(UNIVERSE + ["SPY", "VIXY"]))
    
    # Download in chunks to avoid rate limits
    all_data = {}
    for i in range(0, len(tickers), 10):
        chunk = tickers[i:i+10]
        syms = " ".join(chunk)
        df = yf.download(syms, start="2016-01-01", end="2026-04-13", progress=False, group_by="ticker")
        
        if len(chunk) == 1:
            # Single ticker: df has columns directly
            sym = chunk[0]
            if len(df) > 0:
                all_data[sym] = df
        else:
            for sym in chunk:
                try:
                    sdf = df[sym].dropna(subset=["Close"])
                    if len(sdf) > 100:
                        all_data[sym] = sdf
                except Exception:
                    pass
        time.sleep(0.2)
    
    print(f"  Loaded {len(all_data)} tickers")
    
    # Build bar_index: {ticker: {date_str: {o, h, l, c, v}}}
    bar_index = {}
    for sym, df in all_data.items():
        if sym in ("SPY", "VIXY"):
            continue
        bars = {}
        for date, row in df.iterrows():
            ds = date.strftime("%Y-%m-%d")
            try:
                bars[ds] = {
                    "o": float(row["Open"]),
                    "h": float(row["High"]),
                    "l": float(row["Low"]),
                    "c": float(row["Close"]),
                    "v": float(row["Volume"]),
                }
            except Exception:
                pass
        if bars:
            bar_index[sym] = bars
    
    # VXX data (using VIXY as proxy)
    vxx_list = []
    if "VIXY" in all_data:
        for date, row in all_data["VIXY"].iterrows():
            try:
                vxx_list.append({"t": date.strftime("%Y-%m-%d"), "c": float(row["Close"])})
            except Exception:
                pass
    
    # SPY data
    spy_list = []
    if "SPY" in all_data:
        for date, row in all_data["SPY"].iterrows():
            try:
                spy_list.append({"t": date.strftime("%Y-%m-%d"), "c": float(row["Close"])})
            except Exception:
                pass
    
    # Trading days from SPY
    trading_days = sorted({b["t"] for b in spy_list})
    
    return bar_index, vxx_list, spy_list, trading_days


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hv(bars_list, window=20):
    if len(bars_list) < window + 1:
        return 20.0
    closes = [b["c"] for b in bars_list[-(window+1):]]
    log_rets = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes)) if closes[i-1] > 0]
    if len(log_rets) < 2:
        return 20.0
    mean = sum(log_rets) / len(log_rets)
    var = sum((r - mean)**2 for r in log_rets) / (len(log_rets) - 1)
    return round(math.sqrt(var) * math.sqrt(252) * 100, 2)

def compute_atr(bars_list, period=14):
    if len(bars_list) < period + 1:
        return None
    trs = []
    for i in range(1, min(period + 1, len(bars_list))):
        h = bars_list[i]["h"]
        l = bars_list[i]["l"]
        pc = bars_list[i-1]["c"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None

def get_vxx_ratio(vxx_data, day_idx):
    if day_idx < 10:
        return 1.0
    window = vxx_data[max(0, day_idx-30):day_idx]
    if not window:
        return 1.0
    avg = sum(b["c"] for b in window) / len(window)
    latest = vxx_data[day_idx-1]["c"]
    return latest / avg if avg > 0 else 1.0

def quick_score(bar, prev_bars):
    c, o, v = bar["c"], bar["o"], bar["v"]
    h, l = bar["h"], bar["l"]
    if c < 5 or v < 500_000:
        return None
    chg = (c - o) / o * 100 if o > 0 else 0
    rng = (h - l) / l * 100 if l > 0 else 0
    score = 50
    if chg > 3: score += 10
    elif chg > 1: score += 5
    elif chg < -3: score += 8
    if v > 20e6: score += 15
    elif v > 5e6: score += 8
    elif v > 1e6: score += 3
    if rng > 5: score += 10
    elif rng > 3: score += 5
    if len(prev_bars) >= 5:
        c5 = prev_bars[-5]["c"]
        trend = (c - c5) / c5 * 100 if c5 > 0 else 0
        if trend > 5: score += 5
        elif trend < -5: score += 3
    return max(0, min(100, score))

def compute_vrp(bars_list, vxx_ratio):
    hv = compute_hv(bars_list, 20)
    premium = (vxx_ratio - 1.0) * 0.5
    iv = hv * (1 + premium)
    return round(iv - hv, 2), round(hv, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS P&L SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def options_pnl_broken(entry_price, bars_after, hv, cfg):
    """
    BROKEN path: stock exit logic hits option legs with 4% stop.
    Each leg sees amplified P&L. Most trades exit day 1-2.
    """
    sigma = max(hv / 100, 0.10)
    default_atr = cfg["broken_atr_default"] or 0.02
    stop_pct = STOCK_STOP_ATR_MULT * default_atr  # 4%

    for d in range(1, len(bars_after)):
        stock_move = abs(bars_after[d]["c"] - bars_after[0]["c"]) / bars_after[0]["c"]
        
        # Intraday adverse excursion on worst leg
        intraday_noise = sigma / math.sqrt(252) * 0.7  # 70% of daily vol as intraday swing
        
        # Single option leg P&L: delta * move / option_premium_as_pct
        option_prem = max(0.02, 0.4 * sigma * math.sqrt(OPT_DAYS / 252) * 0.5)  # One leg
        worst_leg_pnl = -(0.5 * max(stock_move, intraday_noise) / option_prem)
        
        # Theta drags each leg
        theta_day = 1.0 / max(OPT_DAYS - d, 3) * 0.4
        worst_leg_pnl -= theta_day
        
        if worst_leg_pnl <= -stop_pct:
            # Triggered! Both legs closed immediately.
            # Net straddle P&L on rapid close = bid-ask spread loss both ways
            net = -(OPT_SPREAD_PCT * 2)
            # Plus any small directional move captured
            partial = stock_move * 0.1  # Tiny capture from one leg being ITM
            net += partial
            return net, d, "stock_stop_on_leg"
    
    # Survived all days (rare)
    total_d = len(bars_after) - 1
    final_move = abs(bars_after[-1]["c"] - bars_after[0]["c"]) / bars_after[0]["c"] if total_d > 0 else 0
    T = min(total_d, OPT_DAYS) / 252
    prem = 0.4 * sigma * math.sqrt(T) * 2  # Straddle
    net = final_move - prem - OPT_SPREAD_PCT * 2
    return net, total_d, "survived"


def options_pnl_fixed(entry_price, bars_after, hv, cfg):
    """
    FIXED path: options_manager handles exits.
    Straddle evaluated as combined position with proper hold time.
    """
    sigma = max(hv / 100, 0.10)
    T = OPT_DAYS / 252
    straddle_prem = 0.4 * sigma * math.sqrt(T) * 2  # Call + put premium as % of stock
    min_hold = cfg["opt_min_hold_days"]
    
    for d in range(1, len(bars_after)):
        if d < min_hold:
            continue
        
        stock_move = abs(bars_after[d]["c"] - bars_after[0]["c"]) / bars_after[0]["c"]
        
        # Straddle value: intrinsic + remaining time value
        remaining_time_frac = max(0, (OPT_DAYS - d) / OPT_DAYS)
        time_val = straddle_prem * remaining_time_frac * 0.55  # Non-linear decay
        intrinsic = stock_move
        current_val = intrinsic + time_val
        
        straddle_pnl_frac = (current_val - straddle_prem) / straddle_prem
        
        # Profit target
        if straddle_pnl_frac >= cfg["opt_profit_target"]:
            net = straddle_pnl_frac * straddle_prem - OPT_SPREAD_PCT * 2
            return net, d, "profit_target"
        
        # Loss limit (negative threshold)
        loss_thresh = -1.0 / cfg["opt_loss_limit"]
        if straddle_pnl_frac <= loss_thresh:
            net = straddle_pnl_frac * straddle_prem - OPT_SPREAD_PCT * 2
            return net, d, "loss_limit"
        
        # DTE exit
        if d >= OPT_DAYS - 5:
            net = straddle_pnl_frac * straddle_prem - OPT_SPREAD_PCT * 2
            return net, d, "dte_exit"
    
    # End of available data
    total_d = len(bars_after) - 1
    if total_d > 0:
        move = abs(bars_after[-1]["c"] - bars_after[0]["c"]) / bars_after[0]["c"]
        remaining = max(0, (OPT_DAYS - total_d) / OPT_DAYS)
        tv = straddle_prem * remaining * 0.5
        pnl_frac = (move + tv - straddle_prem) / straddle_prem
        net = pnl_frac * straddle_prem - OPT_SPREAD_PCT * 2
        return net, total_d, "end_data"
    return -OPT_SPREAD_PCT * 2, 0, "no_data"


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SIM
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(trading_days, bar_index, vxx_data, cfg):
    capital = STARTING_CAPITAL
    peak = STARTING_CAPITAL
    stock_pos = {}
    opt_pos = {}
    trades = []
    daily_eq = []
    daily_ret = []
    stats = defaultdict(float)
    
    vxx_dates = {b["t"]: i for i, b in enumerate(vxx_data)}
    sorted_dates_cache = {t: sorted(bar_index[t].keys()) for t in bar_index}
    
    for day_idx, date in enumerate(trading_days):
        # ── STOCK EXITS ──
        to_exit = []
        for tk, p in stock_pos.items():
            bar = bar_index.get(tk, {}).get(date)
            if not bar: continue
            hold = day_idx - p["entry_day"]
            pct = (bar["c"] - p["entry_price"]) / p["entry_price"] if p["entry_price"] > 0 else 0
            atr_pct = p.get("atr_pct", 0.02)
            stop = min(STOCK_STOP_ATR_MULT * atr_pct, 0.08)
            
            reason = None
            if pct <= -stop: reason = "stop_loss"
            elif pct >= STOCK_TP_PCT: reason = "take_profit"
            elif hold >= STOCK_TIME_STOP_DAYS and pct < 0.03: reason = "time_stop"
            if reason:
                to_exit.append((tk, pct, hold, reason))
        
        for tk, pct, hold, reason in to_exit:
            p = stock_pos.pop(tk, None)
            if not p: continue
            net = pct - 2 * SLIPPAGE_PCT
            pnl = p["cost_basis"] * net
            capital += pnl
            stats["stk_pnl"] += pnl
            stats["stk_trades"] += 1
            if pnl > 0: stats["stk_wins"] += 1
            trades.append({"tk": tk, "inst": "stock", "pnl": round(pnl, 2),
                "net_pct": round(net*100, 2), "hold": hold, "reason": reason,
                "entry": p["entry_date"], "exit": date})
        
        # ── OPTIONS EXITS ──
        opt_exit = []
        for key, p in opt_pos.items():
            tk = p["underlying"]
            bar = bar_index.get(tk, {}).get(date)
            if not bar: continue
            hold = day_idx - p["entry_day"]
            
            tk_dates = sorted_dates_cache.get(tk, [])
            try:
                ei = tk_dates.index(p["entry_date"])
                ci = tk_dates.index(date)
            except ValueError:
                continue
            
            bars_after = [bar_index[tk][d] for d in tk_dates[ei:ci+1]]
            if len(bars_after) < 2: continue
            
            hv = p.get("hv", 20.0)
            
            if not cfg["skip_stock_exit"]:
                net, h, reason = options_pnl_broken(p["entry_price"], bars_after, hv, cfg)
                if reason == "stock_stop_on_leg" or hold >= STOCK_TIME_STOP_DAYS:
                    opt_exit.append((key, net, max(h, 1), reason if reason == "stock_stop_on_leg" else "stock_time_stop"))
            else:
                net, h, reason = options_pnl_fixed(p["entry_price"], bars_after, hv, cfg)
                if h > 0 and reason != "no_data":
                    opt_exit.append((key, net, h, reason))
                elif hold >= OPT_DAYS:
                    opt_exit.append((key, net, hold, "max_dte"))
        
        for key, net, h, reason in opt_exit:
            p = opt_pos.pop(key, None)
            if not p: continue
            pnl = p["cost_basis"] * net
            capital += pnl
            stats["opt_pnl"] += pnl
            stats["opt_trades"] += 1
            stats["opt_hold_sum"] += h
            if pnl > 0: stats["opt_wins"] += 1
            trades.append({"tk": p["underlying"], "inst": "options",
                "strategy": p.get("strategy", "straddle"),
                "pnl": round(pnl, 2), "net_pct": round(net*100, 2),
                "hold": h, "reason": reason,
                "entry": p["entry_date"], "exit": date})
        
        # ── ENTRIES ──
        vix_idx = vxx_dates.get(date, -1)
        vxx_ratio = get_vxx_ratio(vxx_data, vix_idx + 1) if vix_idx > 0 else 1.0
        
        if len(stock_pos) < MAX_STOCK_POSITIONS and capital > 5000:
            for tk in UNIVERSE:
                if tk in stock_pos or len(stock_pos) >= MAX_STOCK_POSITIONS: break
                bars_tk = bar_index.get(tk, {})
                bar = bars_tk.get(date)
                if not bar: continue
                
                tk_dates = sorted_dates_cache.get(tk, [])
                try:
                    idx = tk_dates.index(date)
                except ValueError:
                    continue
                if idx < 20: continue
                prev = [bars_tk[d] for d in tk_dates[max(0,idx-30):idx]]
                
                score = quick_score(bar, prev)
                if score is None or score < MIN_SCORE: continue
                
                atr_bars = [bars_tk[d] for d in tk_dates[max(0,idx-20):idx]]
                atr = compute_atr(atr_bars)
                atr_pct = (atr / bar["c"]) if atr and bar["c"] > 0 else 0.02
                
                pos_pct = min(0.12, 0.08 + (score - 65) / 250)
                deployed = sum(x["cost_basis"] for x in stock_pos.values()) + sum(x["cost_basis"] for x in opt_pos.values())
                room = capital * 0.80 - deployed
                if room < capital * 0.05: continue
                cost = min(capital * pos_pct, room)
                
                stock_pos[tk] = {
                    "entry_price": bar["c"], "entry_day": day_idx,
                    "entry_date": date, "cost_basis": cost,
                    "score": score, "atr_pct": atr_pct,
                }
        
        if cfg["options_on"] and len(opt_pos) < MAX_OPTIONS_POSITIONS and capital > 5000:
            for tk in UNIVERSE:
                if any(x["underlying"] == tk for x in opt_pos.values()): continue
                if len(opt_pos) >= MAX_OPTIONS_POSITIONS: break
                
                bars_tk = bar_index.get(tk, {})
                bar = bars_tk.get(date)
                if not bar or bar["c"] < 10: continue
                
                tk_dates = sorted_dates_cache.get(tk, [])
                try:
                    idx = tk_dates.index(date)
                except ValueError:
                    continue
                if idx < 20: continue
                prev = [bars_tk[d] for d in tk_dates[max(0,idx-30):idx]]
                
                vrp, hv = compute_vrp(prev, vxx_ratio)
                
                should = False
                strat = "straddle"
                if vxx_ratio >= 1.30:
                    should = True; strat = "panic_put"
                elif vrp > 4 and hv > 25:
                    should = True; strat = "high_iv"
                elif hv < 15 and vrp < -1:
                    should = True; strat = "cheap_iv"
                
                if not should: continue
                
                opt_pct = min(0.08, 0.05 + vrp / 200)
                deployed = sum(x["cost_basis"] for x in stock_pos.values()) + sum(x["cost_basis"] for x in opt_pos.values())
                room = capital * 0.80 - deployed
                if room < capital * 0.03: continue
                cost = min(capital * opt_pct, room)
                
                okey = f"OPT_{tk}_{day_idx}"
                opt_pos[okey] = {
                    "underlying": tk, "entry_price": bar["c"],
                    "entry_day": day_idx, "entry_date": date,
                    "cost_basis": cost, "strategy": strat,
                    "hv": hv, "vrp": vrp, "vxx_ratio": vxx_ratio,
                }
        
        # ── DAILY EQUITY ──
        open_val = 0
        for tk, p in stock_pos.items():
            bar = bar_index.get(tk, {}).get(date)
            if bar:
                chg = (bar["c"] - p["entry_price"]) / p["entry_price"]
                open_val += p["cost_basis"] * (1 + chg)
        opt_val = sum(x["cost_basis"] for x in opt_pos.values())
        deployed = sum(x["cost_basis"] for x in stock_pos.values()) + sum(x["cost_basis"] for x in opt_pos.values())
        eq = (capital - deployed) + open_val + opt_val
        daily_eq.append(eq)
        
        if len(daily_eq) > 1 and daily_eq[-2] > 0:
            daily_ret.append((daily_eq[-1] - daily_eq[-2]) / daily_eq[-2])
        
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        stats["max_dd"] = max(stats.get("max_dd", 0), dd)
    
    # Close remaining
    last = trading_days[-1]
    for tk, p in list(stock_pos.items()):
        bar = bar_index.get(tk, {}).get(last)
        if bar:
            pct = (bar["c"] - p["entry_price"]) / p["entry_price"] - 2 * SLIPPAGE_PCT
            pnl = p["cost_basis"] * pct
            capital += pnl
            stats["stk_pnl"] += pnl; stats["stk_trades"] += 1
            if pnl > 0: stats["stk_wins"] += 1
    for k, p in list(opt_pos.items()):
        pnl = p["cost_basis"] * (-OPT_SPREAD_PCT * 2)
        capital += pnl
        stats["opt_pnl"] += pnl; stats["opt_trades"] += 1
    
    # ── METRICS ──
    ny = len(trading_days) / 252
    cagr = ((capital / STARTING_CAPITAL) ** (1/ny) - 1) * 100 if ny > 0 and capital > 0 else 0
    total_ret = (capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    
    sharpe = sortino = 0
    if len(daily_ret) > 10:
        avg_r = np.mean(daily_ret)
        std_r = np.std(daily_ret, ddof=1)
        sharpe = (avg_r / std_r) * math.sqrt(252) if std_r > 0 else 0
        down = [r for r in daily_ret if r < 0]
        down_std = np.std(down, ddof=1) if len(down) > 2 else std_r
        sortino = (avg_r / down_std) * math.sqrt(252) if down_std > 0 else 0
    
    ot = int(stats["opt_trades"])
    st = int(stats["stk_trades"])
    ow = int(stats["opt_wins"])
    sw = int(stats["stk_wins"])
    
    opt_trades_list = [t for t in trades if t["inst"] == "options"]
    exit_reasons = defaultdict(int)
    for t in opt_trades_list:
        exit_reasons[t["reason"]] += 1
    
    total_pnl = stats["stk_pnl"] + stats["opt_pnl"]
    
    return {
        "final_capital": round(capital, 2),
        "total_return": round(total_ret, 2),
        "cagr": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(stats["max_dd"] * 100, 2),
        "total_trades": st + ot,
        "stk_trades": st,
        "stk_wins": sw,
        "stk_wr": round(sw / st * 100, 1) if st > 0 else 0,
        "stk_pnl": round(stats["stk_pnl"], 2),
        "opt_trades": ot,
        "opt_wins": ow,
        "opt_wr": round(ow / ot * 100, 1) if ot > 0 else 0,
        "opt_pnl": round(stats["opt_pnl"], 2),
        "opt_avg_hold": round(stats["opt_hold_sum"] / ot, 1) if ot > 0 else 0,
        "opt_avg_pnl": round(stats["opt_pnl"] / ot, 2) if ot > 0 else 0,
        "opt_pct_pnl": round(stats["opt_pnl"] / total_pnl * 100, 1) if total_pnl != 0 else 0,
        "opt_exit_reasons": dict(exit_reasons),
        "trades": trades,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 85)
    print("  VolTradeAI — Options Fix A/B Backtest (2016–2026)")
    print("  BROKEN vs FIX_SKIP vs FIX_TIGHT vs NO_OPTIONS")
    print("=" * 85)
    
    print("\n[1/3] Loading data...")
    bar_index, vxx_data, spy_data, trading_days = load_data()
    
    spy_start = spy_data[0]["c"]
    spy_end = spy_data[-1]["c"]
    ny = len(trading_days) / 252
    spy_cagr = ((spy_end / spy_start) ** (1/ny) - 1) * 100
    spy_total = (spy_end - spy_start) / spy_start * 100
    print(f"  Period: {trading_days[0]} → {trading_days[-1]} ({len(trading_days)} days)")
    print(f"  SPY: {spy_total:+.1f}% total | {spy_cagr:+.1f}% CAGR")
    
    print("\n[2/3] Running backtests...")
    results = {}
    for name, cfg in CONFIGS.items():
        print(f"\n  --- {name}: {cfg['label']} ---")
        r = simulate(trading_days, bar_index, vxx_data, cfg)
        r["alpha"] = round(r["cagr"] - spy_cagr, 2)
        results[name] = r
        print(f"  ${r['final_capital']:,.0f} | CAGR {r['cagr']:+.1f}% | Sharpe {r['sharpe']:.3f} | Sortino {r['sortino']:.3f} | DD {r['max_dd']:.1f}%")
        print(f"  Opt P&L: ${r['opt_pnl']:+,.0f} | WR {r['opt_wr']:.0f}% | Avg hold {r['opt_avg_hold']:.1f}d | Trades {r['opt_trades']}")
    
    # ── COMPARISON TABLE ──
    print(f"\n\n[3/3] COMPARISON")
    sep = "─" * 90
    sep2 = "═" * 90
    
    print(f"\n{sep2}")
    hdrs = ["BROKEN", "FIX_SKIP", "FIX_TIGHT", "NO_OPTIONS"]
    print(f"  {'Metric':<30} " + "".join(f"{h:>14}" for h in hdrs))
    print(sep)
    
    def row(label, key, fmt="{:.1f}", good="higher"):
        vals = [results[n].get(key, 0) for n in hdrs]
        best_idx = vals.index(max(vals)) if good == "higher" else vals.index(min(vals))
        parts = []
        for i, v in enumerate(vals):
            s = fmt.format(v)
            if i == best_idx:
                s += " ★"
            parts.append(f"{s:>14}")
        print(f"  {label:<30} " + "".join(parts))
    
    row("Final Capital ($)", "final_capital", "${:,.0f}")
    row("CAGR (%)", "cagr", "{:+.2f}%")
    row("Alpha vs SPY (%)", "alpha", "{:+.2f}%")
    row("Sharpe", "sharpe", "{:.3f}")
    row("Sortino", "sortino", "{:.3f}")
    row("Max Drawdown (%)", "max_dd", "{:.1f}%", "lower")
    print(sep)
    row("Stock Trades", "stk_trades", "{:d}")
    row("Stock Win Rate (%)", "stk_wr", "{:.1f}%")
    row("Stock P&L ($)", "stk_pnl", "${:+,.0f}")
    print(sep)
    row("Options Trades", "opt_trades", "{:d}")
    row("Options Win Rate (%)", "opt_wr", "{:.1f}%")
    row("Options P&L ($)", "opt_pnl", "${:+,.0f}")
    row("Options Avg Hold (d)", "opt_avg_hold", "{:.1f}")
    row("Options Avg P&L ($)", "opt_avg_pnl", "${:+.0f}")
    row("Options % of Total P&L", "opt_pct_pnl", "{:.1f}%")
    print(sep2)
    
    print(f"\n  SPY Buy-and-Hold: {spy_total:+.1f}% | {spy_cagr:+.1f}% CAGR")
    for n in hdrs:
        r = results[n]
        print(f"  {n:<14} Alpha: {r['alpha']:+.2f}%  CAGR: {r['cagr']:+.2f}%  Sortino: {r['sortino']:.3f}")
    
    # Exit reason breakdown
    print(f"\n  OPTIONS EXIT REASONS:")
    print(sep)
    for n in ["BROKEN", "FIX_SKIP", "FIX_TIGHT"]:
        r = results[n]
        reasons = r.get("opt_exit_reasons", {})
        if reasons:
            print(f"  {n}:")
            for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
                pct = cnt / r["opt_trades"] * 100 if r["opt_trades"] > 0 else 0
                print(f"    {reason:<35} {cnt:>5} ({pct:.0f}%)")
    
    # Improvement calculation
    print(f"\n  FIX IMPACT:")
    print(sep)
    broken = results["BROKEN"]
    fix = results["FIX_SKIP"]
    tight = results["FIX_TIGHT"]
    noop = results["NO_OPTIONS"]
    
    print(f"  FIX_SKIP vs BROKEN:")
    print(f"    CAGR:    {broken['cagr']:+.2f}% → {fix['cagr']:+.2f}% ({fix['cagr'] - broken['cagr']:+.2f}pp)")
    print(f"    Sortino: {broken['sortino']:.3f} → {fix['sortino']:.3f} ({fix['sortino'] - broken['sortino']:+.3f})")
    print(f"    Opt P&L: ${broken['opt_pnl']:+,.0f} → ${fix['opt_pnl']:+,.0f} ({fix['opt_pnl'] - broken['opt_pnl']:+,.0f})")
    print(f"    Opt WR:  {broken['opt_wr']:.0f}% → {fix['opt_wr']:.0f}%")
    print(f"    Opt Hold: {broken['opt_avg_hold']:.1f}d → {fix['opt_avg_hold']:.1f}d")
    
    print(f"\n  FIX_TIGHT vs FIX_SKIP:")
    print(f"    CAGR:    {fix['cagr']:+.2f}% → {tight['cagr']:+.2f}% ({tight['cagr'] - fix['cagr']:+.2f}pp)")
    print(f"    Sortino: {fix['sortino']:.3f} → {tight['sortino']:.3f}")
    print(f"    Opt P&L: ${fix['opt_pnl']:+,.0f} → ${tight['opt_pnl']:+,.0f}")
    
    print(f"\n  Options value-add (FIX_SKIP vs NO_OPTIONS):")
    print(f"    CAGR:    {noop['cagr']:+.2f}% → {fix['cagr']:+.2f}% ({fix['cagr'] - noop['cagr']:+.2f}pp)")
    print(f"    Sortino: {noop['sortino']:.3f} → {fix['sortino']:.3f}")
    
    print(sep2)
    
    # Save
    out = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "period": f"{trading_days[0]} → {trading_days[-1]}",
        "days": len(trading_days),
        "spy_cagr": round(spy_cagr, 2),
        "results": {k: {kk: v for kk, v in r.items() if kk != "trades"} for k, r in results.items()},
    }
    path = "/home/user/workspace/backtest_options_fix_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")
    print(f"Done in {time.time() - t0:.1f}s")
    
    return results

if __name__ == "__main__":
    main()
