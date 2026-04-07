#!/usr/bin/env python3
"""
VolTradeAI v1.0.28 — Full System 10-Year Backtest
Tests the COMPLETE system: stocks + options + third leg + intraday shorts (hybrid v2.1)

Uses cached data from /tmp/backtest_cache.json to avoid API calls.
Runs two variants:
  A) Full system WITHOUT intraday shorts (baseline)
  B) Full system WITH intraday shorts (v2.1 hybrid)
Then compares both to SPY buy-and-hold.
"""
import os, sys, json, math, time
sys.path.insert(0, '/tmp/vt_test')
import numpy as np
from datetime import datetime

print("=" * 70)
print("VolTradeAI v1.0.28 — Full System Backtest (2016–2026)")
print("Stocks + Options + Third Leg + Intraday Shorts (hybrid v2.1)")
print("=" * 70)

# ── Load cached data ──────────────────────────────────────────────
print("\n[1] Loading cached market data...")
t0 = time.time()

CACHE_PATH = "/tmp/backtest_cache.json"
with open(CACHE_PATH) as f:
    raw_cache = json.load(f)

# Normalize cache structure (can be flat or nested)
if "spy" in raw_cache:
    # Nested format: {spy: [...], vxx: [...], stocks: {sym: [...]}, ...}
    spy_bars = raw_cache["spy"]
    vxx_bars = raw_cache.get("vxx", [])
    stock_data_raw = raw_cache.get("stocks", {})
    leg3_raw = raw_cache.get("leg3", {})
    midsmall_raw = raw_cache.get("midsmall", {})
else:
    # Flat format: {SPY: [...], VXX: [...], AAPL: [...], ...}
    spy_bars = raw_cache.get("SPY", [])
    vxx_bars = raw_cache.get("VXX", [])
    stock_data_raw = {k: v for k, v in raw_cache.items() if k not in ("SPY", "VXX")}
    leg3_raw = {}
    midsmall_raw = {}

spy_by_date = {b["t"][:10]: b for b in spy_bars}
vxx_by_date = {b["t"][:10]: b for b in vxx_bars}

# Stock universe
STOCK_UNIVERSE = ["AAPL","MSFT","NVDA","AMD","GOOGL","AMZN","INTC","CRM",
                  "JPM","BAC","GS","V","MA","MSTR","TSLA","XOM","JNJ","WMT","QQQ","IWM"]

# Third leg assets
THIRD_LEG_ASSETS = {k: v for k, v in leg3_raw.items()}

# All stock data
stock_data = {}
for sym in STOCK_UNIVERSE:
    if sym in stock_data_raw:
        stock_data[sym] = stock_data_raw[sym]

# Short candidates = all stocks + mid/small caps
short_candidates_data = {}
for sym, bars in stock_data_raw.items():
    short_candidates_data[sym] = bars
for sym, bars in midsmall_raw.items():
    short_candidates_data[sym] = bars

# Build date lookups
stock_by_date = {sym: {b["t"][:10]: b for b in bars} for sym, bars in stock_data.items()}
short_by_date = {sym: {b["t"][:10]: b for b in bars} for sym, bars in short_candidates_data.items()}
third_by_date = {sym: {b["t"][:10]: b for b in bars} for sym, bars in THIRD_LEG_ASSETS.items()}

all_dates = sorted([d for d in spy_by_date.keys() if d >= "2016-01-01"])

print(f"  SPY bars:     {len(spy_bars)}")
print(f"  VXX bars:     {len(vxx_bars)}")
print(f"  Long stocks:  {len(stock_data)}/{len(STOCK_UNIVERSE)}")
print(f"  Short stocks: {len(short_candidates_data)}")
print(f"  Third leg:    {len(THIRD_LEG_ASSETS)} assets")
print(f"  Test period:  {all_dates[0]} → {all_dates[-1]} ({len(all_dates)} days)")
print(f"  Load time:    {time.time()-t0:.1f}s")

# ── Regime detection (with Fix B: 200d MA) ────────────────────────
def get_regime(vxx_ratio, spy_vs_ma50, spy_below_200_days=0):
    """v1.0.22+: Fix B adds slow-bear detection via 200d MA."""
    if spy_below_200_days >= 10:
        if vxx_ratio >= 1.30 or spy_vs_ma50 < 0.94: return "PANIC"
        return "BEAR"  # Fix B: persistent weakness = BEAR
    if vxx_ratio >= 1.30 or spy_vs_ma50 < 0.94: return "PANIC"
    if vxx_ratio >= 1.15: return "BEAR"
    if vxx_ratio >= 1.05: return "CAUTION"
    if vxx_ratio <= 0.90: return "BULL"
    return "NEUTRAL"

def get_adaptive_params(regime):
    """v1.0.23 optimized: score=63, TP=12%, SL=6%, hold=10d."""
    if regime == "PANIC":   return {"max_pos": 0, "min_score": 99, "size_pct": 0.06, "tp": 12, "sl": 6, "hold": 10}
    if regime == "BEAR":    return {"max_pos": 0, "min_score": 99, "size_pct": 0.08, "tp": 12, "sl": 6, "hold": 10}
    if regime == "CAUTION": return {"max_pos": 4, "min_score": 67, "size_pct": 0.10, "tp": 12, "sl": 6, "hold": 10}
    if regime == "BULL":    return {"max_pos": 8, "min_score": 63, "size_pct": 0.15, "tp": 12, "sl": 6, "hold": 10}
    return {"max_pos": 6, "min_score": 63, "size_pct": 0.12, "tp": 12, "sl": 6, "hold": 10}

# ── Stock scoring (same as backtest_v2.py) ────────────────────────
def quick_score(sym, date_idx, dates):
    d = dates[date_idx]
    bar = stock_by_date.get(sym, {}).get(d)
    if not bar or date_idx < 10: return 0
    c = float(bar.get("c", 0))
    v = int(bar.get("v", 0))
    if c < 2 or v < 100_000: return 0

    prev5_bar = None
    for j in range(date_idx-5, max(0, date_idx-7), -1):
        pb = stock_by_date.get(sym, {}).get(dates[j])
        if pb: prev5_bar = pb; break
    if not prev5_bar: return 0
    pc5 = float(prev5_bar.get("c", c))
    if pc5 <= 0: return 0
    mom5 = (c - pc5) / pc5 * 100
    if abs(mom5) > 30: return 0

    prev20_bar = None
    for j in range(date_idx-20, max(0, date_idx-22), -1):
        mb = stock_by_date.get(sym, {}).get(dates[j])
        if mb: prev20_bar = mb; break
    mom20 = (c - float(prev20_bar.get("c", c))) / float(prev20_bar.get("c", c)) * 100 if prev20_bar else 0

    avg_vol5, vol_count = 0, 0
    for j in range(date_idx-5, date_idx):
        if 0 <= j < len(dates):
            vb = stock_by_date.get(sym, {}).get(dates[j])
            if vb: avg_vol5 += int(vb.get("v", 0)); vol_count += 1
    avg_vol5 = avg_vol5 / max(vol_count, 1)
    vol_ratio = v / max(avg_vol5, 1)

    score_mom5 = min(mom5 * 5, 35) if mom5 > 0 else 0
    score_mom20 = min(max(mom20 * 1.5, 0), 20) if mom20 > 0 else 0
    score_vol = min((vol_ratio - 1.0) * 12, 20) if vol_ratio > 1.1 else 0
    return round(min(30 + score_mom5 + score_mom20 + score_vol, 100), 1)

# ── Intraday short scoring (v2.1 hybrid: fixed windows) ──────────
def score_short(sym, date_idx, dates, spy_ret_10d):
    """v2.1 hybrid: fixed 5/10/20 lookbacks, ATR-relative thresholds."""
    d = dates[date_idx]
    # Need 25+ bars of history
    bars_list = []
    for j in range(max(0, date_idx - 30), date_idx + 1):
        b = short_by_date.get(sym, {}).get(dates[j])
        if b: bars_list.append(b)
    if len(bars_list) < 25: return None

    closes = [float(b.get("c", 0)) for b in bars_list]
    highs = [float(b.get("h", b.get("c", 0))) for b in bars_list]
    vols = [int(b.get("v", 0)) for b in bars_list]
    c = closes[-1]
    if c <= 0: return None

    # Dollar volume filter
    dollar_vol = c * vols[-1]
    if dollar_vol < 50_000_000: return None

    # ATR
    trs = []
    for i in range(1, len(bars_list)):
        h = float(bars_list[i].get("h", bars_list[i].get("c", 0)))
        l = float(bars_list[i].get("l", bars_list[i].get("c", 0)))
        pc = float(bars_list[i-1].get("c", 0))
        if pc > 0: trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < 14: return None
    atr = float(np.mean(trs[-14:]))
    if atr <= 0: return None
    atr_pct = atr / c * 100

    signals = {}

    # 1. Momentum collapse (5-day)
    if len(closes) > 5 and closes[-5] > 0:
        mom = (c - closes[-5]) / closes[-5] * 100
        signals["momentum"] = float(max(0, min(1, -mom / (atr_pct * 2))))
    else:
        signals["momentum"] = 0.0

    # 2. Failed breakout (20-day high)
    if len(highs) >= 20:
        h_roll = max(highs[-20:])
        h_recent = max(highs[-5:])
        drop = (c - h_recent) / h_recent * 100 if h_recent > 0 else 0
        near_high = h_recent >= h_roll * 0.97
        signals["failed_breakout"] = float(max(0, min(1, -drop / atr_pct))) if near_high else 0.0
    else:
        signals["failed_breakout"] = 0.0

    # 3. Volume distribution (10-day)
    if len(vols) >= 10:
        avg_v = np.mean(vols[-10:])
        vr = vols[-1] / max(avg_v, 1)
        day_ret = (c - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 and closes[-2] > 0 else 0
        signals["distribution"] = float(max(0, min(1, (vr - 1) * 0.5))) if day_ret < -(atr_pct * 0.3) else 0.0
    else:
        signals["distribution"] = 0.0

    # 4. Relative weakness vs SPY (10-day)
    if len(closes) >= 10 and closes[-10] > 0:
        stock_ret = (c - closes[-10]) / closes[-10] * 100
        relative = stock_ret - spy_ret_10d
        signals["rel_weakness"] = float(max(0, min(1, -relative / (atr_pct * 3))))
    else:
        signals["rel_weakness"] = 0.0

    # 5. Trend breakdown (10/20 MA)
    if len(closes) >= 20:
        ma_s = np.mean(closes[-10:])
        ma_l = np.mean(closes[-20:])
        below = (c < ma_s) + (c < ma_l)
        pct_below = (c - ma_l) / ma_l * 100 if ma_l > 0 else 0
        signals["trend_break"] = float(max(0, min(1, below * 0.3 + max(0, -pct_below / atr_pct) * 0.4)))
    else:
        signals["trend_break"] = 0.0

    # 6. Gap down
    if len(bars_list) >= 2:
        prev_c = float(bars_list[-2].get("c", c))
        today_o = float(bars_list[-1].get("o", c))
        gap = (today_o - prev_c) / prev_c * 100 if prev_c > 0 else 0
        gap_threshold = -(atr_pct * 0.2)
        signals["gap_down"] = float(max(0, min(1, -gap / atr_pct))) if gap < gap_threshold else 0.0
    else:
        signals["gap_down"] = 0.0

    # Equal weighted composite
    raw_score = sum(signals.values()) / 6.0
    noise_floor = min(0.4, atr_pct / 15)
    active_count = sum(1 for v in signals.values() if v > noise_floor)

    score = round(raw_score * 100, 1)
    if active_count < 2 or score < 15: return None

    return {"score": score, "active": active_count, "atr_pct": atr_pct}

# ── Options simulation ────────────────────────────────────────────
def simulate_options(setup, vxx_ratio, spy_fwd, vxx_fwd):
    if setup == "vxx_panic_put_sale":
        spy_ret = (spy_fwd - 1.0) * 100
        win = spy_ret > -3.0 and vxx_fwd < vxx_ratio * 1.10
        return win, 6.0 if win else -5.0
    elif setup == "high_iv_premium_sale":
        spy_move = abs((spy_fwd - 1.0) * 100)
        win = vxx_fwd < 1.0 and spy_move < 3.0
        return win, 8.0 if win else -6.0
    elif setup == "low_iv_breakout_buy":
        spy_move = abs((spy_fwd - 1.0) * 100)
        win = spy_move > 1.0
        return win, (spy_move * 2.5 - 1.5) if win else -1.5
    return False, 0

# ── Third leg (v1.0.25: VRP harvest + sector rotation) ───────────
def run_third_leg(regime, equity, date_idx, dates):
    """Third leg: allocate 15% to VRP (SVXY proxy via inverse VXX) and 12% to sector (XOM+LMT)."""
    pnl = 0
    d = dates[date_idx]
    if regime not in ("BEAR", "PANIC", "CAUTION"): return 0  # Only in stressed regimes

    # VRP harvest (15% of equity): Short vol via inverse VXX path
    # In backtest: use next-day VXX return, inverted
    if date_idx + 1 < len(dates):
        vxx_today = vxx_by_date.get(d, {})
        vxx_tmrw = vxx_by_date.get(dates[date_idx + 1], {})
        if vxx_today and vxx_tmrw:
            vxx_c = float(vxx_today.get("c", 0))
            vxx_n = float(vxx_tmrw.get("c", 0))
            if vxx_c > 0:
                vxx_ret = (vxx_n - vxx_c) / vxx_c
                # VRP = short vol → profit when VXX falls
                vrp_pnl = equity * 0.15 * (-vxx_ret)
                pnl += vrp_pnl

    # Sector rotation (12% split: 6% XOM + 6% LMT)
    for sym, alloc in [("XOM", 0.06), ("LMT", 0.06)]:
        if date_idx + 1 < len(dates):
            today = third_by_date.get(sym, {}).get(d)
            tmrw = third_by_date.get(sym, {}).get(dates[date_idx + 1])
            if today and tmrw:
                c = float(today.get("c", 0))
                n = float(tmrw.get("c", 0))
                if c > 0:
                    pnl += equity * alloc * ((n - c) / c)

    return pnl


# ══════════════════════════════════════════════════════════════════
# ── Run backtest (two variants) ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def run_backtest(include_shorts=False, include_third_leg=True, label=""):
    EQUITY_START = 100_000.0
    equity = EQUITY_START
    positions = {}
    trades_log = []
    daily_equity = []
    cooldown_log = {}
    SLIPPAGE = 0.001
    peak_equity = equity
    max_drawdown = 0.0
    total_trades = 0
    winning_trades = 0
    options_pnl_total = 0.0
    options_trades = []
    options_wins = 0
    third_leg_pnl = 0.0
    short_pnl_total = 0.0
    short_trades = 0
    short_wins = 0

    # Track SPY below 200d MA (for Fix B)
    spy_below_200_count = 0

    for day_idx, date in enumerate(all_dates):
        spy_bar = spy_by_date.get(date, {})
        vxx_bar = vxx_by_date.get(date, {})
        if not spy_bar: continue

        spy_close = float(spy_bar.get("c", 0))
        vxx_close = float(vxx_bar.get("c", 15)) if vxx_bar else 15.0

        # VXX ratio
        vxx_hist = []
        for j in range(max(0, day_idx-30), day_idx):
            vb = vxx_by_date.get(all_dates[j])
            if vb: vxx_hist.append(float(vb.get("c", 15)))
        vxx_avg30 = sum(vxx_hist) / len(vxx_hist) if vxx_hist else 15.0
        vxx_ratio = vxx_close / vxx_avg30 if vxx_avg30 > 0 else 1.0

        # SPY vs 50d MA
        spy_hist50 = []
        for j in range(max(0, day_idx-50), day_idx):
            sb = spy_by_date.get(all_dates[j])
            if sb: spy_hist50.append(float(sb.get("c", spy_close)))
        spy_ma50 = sum(spy_hist50) / len(spy_hist50) if spy_hist50 else spy_close
        spy_vs_ma50 = spy_close / spy_ma50 if spy_ma50 > 0 else 1.0

        # SPY 200d MA (Fix B)
        spy_hist200 = []
        for j in range(max(0, day_idx-200), day_idx):
            sb = spy_by_date.get(all_dates[j])
            if sb: spy_hist200.append(float(sb.get("c", spy_close)))
        spy_ma200 = sum(spy_hist200) / len(spy_hist200) if spy_hist200 else spy_close
        if spy_close < spy_ma200:
            spy_below_200_count += 1
        else:
            spy_below_200_count = 0

        regime = get_regime(vxx_ratio, spy_vs_ma50, spy_below_200_count)
        params = get_adaptive_params(regime)

        # IV rank
        vxx_52 = []
        for j in range(max(0, day_idx-252), day_idx):
            vb = vxx_by_date.get(all_dates[j])
            if vb: vxx_52.append(float(vb.get("c", 15)))
        vxx_lo = min(vxx_52) if vxx_52 else 10
        vxx_hi = max(vxx_52) if vxx_52 else 50
        iv_rank = (vxx_close - vxx_lo) / (vxx_hi - vxx_lo) * 100 if vxx_hi > vxx_lo else 50.0

        # ── Manage existing positions ─────────────────────────────
        to_close = []
        for sym, pos in positions.items():
            sym_bar = stock_by_date.get(sym, {}).get(date)
            if not sym_bar: continue
            curr = float(sym_bar.get("c", pos["entry_price"]))
            pnl_pct = (curr - pos["entry_price"]) / pos["entry_price"] * 100
            days_held = day_idx - pos["entry_day"]

            tp_pct = params["tp"]
            sl_pct = -params["sl"]

            if pnl_pct >= tp_pct:
                to_close.append((sym, "TP", pnl_pct))
            elif pnl_pct <= sl_pct:
                to_close.append((sym, "SL", pnl_pct))
                cooldown_log[sym] = day_idx
            elif days_held >= params["hold"]:
                to_close.append((sym, "TIME", pnl_pct))

        for sym, reason, pnl_pct in to_close:
            pos = positions.pop(sym)
            actual_cost = pos.get("actual_cost", pos["shares"] * pos["entry_price"])
            entry_value = pos["shares"] * pos["entry_price"]
            exit_value = entry_value * (1 + pnl_pct / 100) * (1 - SLIPPAGE)
            equity += exit_value
            total_trades += 1
            if pnl_pct > 0: winning_trades += 1
            pnl_amt = exit_value - actual_cost
            trades_log.append({"date": date, "sym": sym, "pnl_pct": round(pnl_pct, 2),
                                "pnl_amt": round(pnl_amt, 2), "regime": regime})

        # ── Options ───────────────────────────────────────────────
        fwd5 = min(day_idx + 5, len(all_dates) - 1)
        fwd10 = min(day_idx + 10, len(all_dates) - 1)
        fwd_spy = spy_by_date.get(all_dates[fwd5], {})
        fwd_vxx = vxx_by_date.get(all_dates[fwd5], {})
        spy_fwd5 = float(fwd_spy.get("c", spy_close)) / spy_close if spy_close > 0 else 1.0
        vxx_fwd5 = float(fwd_vxx.get("c", vxx_close)) / vxx_close if vxx_close > 0 else 1.0
        opts_size = equity * 0.05

        if vxx_ratio >= 1.30 and spy_vs_ma50 >= 0.94 and len(options_trades) % 2 == 0:
            fwd10_spy = spy_by_date.get(all_dates[fwd10], {})
            spy_fwd10 = float(fwd10_spy.get("c", spy_close)) / spy_close if spy_close > 0 else 1.0
            win, pnl_pct = simulate_options("vxx_panic_put_sale", vxx_ratio, spy_fwd10, vxx_fwd5)
            pnl_amt = opts_size * pnl_pct / 100
            equity += pnl_amt; options_pnl_total += pnl_amt
            if win: options_wins += 1
            options_trades.append({"date": date, "setup": "vxx_panic_put_sale", "win": win})

        if iv_rank > 70 and regime != "PANIC" and day_idx % 5 == 0:
            win, pnl_pct = simulate_options("high_iv_premium_sale", vxx_ratio, spy_fwd5, vxx_fwd5)
            pnl_amt = opts_size * pnl_pct / 100
            equity += pnl_amt; options_pnl_total += pnl_amt
            if win: options_wins += 1
            options_trades.append({"date": date, "setup": "high_iv_premium_sale", "win": win})

        if iv_rank < 20 and regime in ("BULL", "NEUTRAL") and day_idx % 7 == 0:
            win, pnl_pct = simulate_options("low_iv_breakout_buy", vxx_ratio, spy_fwd5, vxx_fwd5)
            pnl_amt = opts_size * pnl_pct / 100
            equity += pnl_amt; options_pnl_total += pnl_amt
            if win: options_wins += 1
            options_trades.append({"date": date, "setup": "low_iv_breakout_buy", "win": win})

        # ── Third leg (v1.0.25) ───────────────────────────────────
        if include_third_leg:
            tl_pnl = run_third_leg(regime, equity, day_idx, all_dates)
            equity += tl_pnl
            third_leg_pnl += tl_pnl

        # ── Intraday shorts (v2.1 hybrid) ────────────────────────
        # CRITICAL: Score using YESTERDAY's data to avoid look-ahead bias.
        # Signal fires at yesterday's close → short at today's open → cover at today's close.
        if include_shorts and regime in ("BEAR", "PANIC", "CAUTION") and day_idx >= 2:
            # SPY 10-day return (using yesterday's data)
            spy_ret_10d = 0
            if day_idx >= 11:
                spy_10ago = spy_by_date.get(all_dates[day_idx - 11], {})
                spy_yest = spy_by_date.get(all_dates[day_idx - 1], {})
                if spy_10ago and spy_yest:
                    spy_10c = float(spy_10ago.get("c", 1))
                    spy_yc = float(spy_yest.get("c", 1))
                    if spy_10c > 0:
                        spy_ret_10d = (spy_yc - spy_10c) / spy_10c * 100

            # Score using YESTERDAY's data (day_idx - 1)
            short_scored = []
            for sym in short_candidates_data.keys():
                sig = score_short(sym, day_idx - 1, all_dates, spy_ret_10d)
                if sig:
                    short_scored.append((sym, sig))

            short_scored.sort(key=lambda x: x[1]["score"], reverse=True)

            # Pick top 2 with sector diversification
            used_sectors = set()
            picks = []
            for sym, sig in short_scored:
                if len(picks) >= 2: break
                sec = sym[0]  # Rough sector proxy
                if sec in used_sectors: continue
                picks.append((sym, sig))
                used_sectors.add(sec)

            for sym, sig in picks:
                # TODAY's bar for actual execution
                sym_bar = short_by_date.get(sym, {}).get(date)
                if not sym_bar: continue
                today_open = float(sym_bar.get("o", 0))
                today_close = float(sym_bar.get("c", 0))
                if today_open <= 0 or today_close <= 0: continue

                # Position size: 3% base
                pos_pct = min(0.06, 0.03 * max(0.5, sig["score"] / 50))
                alloc = equity * pos_pct

                # Intraday short: sell at open, buy to cover at close
                short_ret = (today_open - today_close) / today_open * 100
                pnl_amt = alloc * (short_ret / 100) * (1 - SLIPPAGE * 2)  # 2x slippage for round trip
                equity += pnl_amt
                short_pnl_total += pnl_amt
                short_trades += 1
                if short_ret > 0: short_wins += 1

        # ── New stock trades ──────────────────────────────────────
        if len(positions) < params["max_pos"]:
            candidates = []
            for sym in STOCK_UNIVERSE:
                if sym in positions: continue
                if sym in cooldown_log and day_idx - cooldown_log[sym] < 1: continue
                sc = quick_score(sym, day_idx, all_dates)
                if sc >= params["min_score"]:
                    candidates.append((sym, sc))

            candidates.sort(key=lambda x: x[1], reverse=True)
            slots = params["max_pos"] - len(positions)

            for sym, sc in candidates[:slots]:
                sym_bar = stock_by_date.get(sym, {}).get(date)
                if not sym_bar: continue
                price = float(sym_bar.get("c", 0))
                if price < 5: continue
                target_value = equity * params["size_pct"]
                shares = int(target_value / price)
                if shares <= 0: continue
                actual_cost = shares * price * (1 + SLIPPAGE)  # What we actually pay
                positions[sym] = {"entry_price": price, "shares": shares,
                                   "entry_day": day_idx, "score": sc, "regime": regime,
                                   "actual_cost": actual_cost}
                equity -= actual_cost

        # ── Track equity ──────────────────────────────────────────
        unrealized = 0
        for sym, pos in positions.items():
            sym_bar = stock_by_date.get(sym, {}).get(date)
            if sym_bar:
                curr = float(sym_bar.get("c", pos["entry_price"]))
                unrealized += (curr - pos["entry_price"]) * pos["shares"]
        total_eq = equity + unrealized
        daily_equity.append(total_eq)

        if total_eq > peak_equity: peak_equity = total_eq
        dd = (peak_equity - total_eq) / peak_equity * 100
        if dd > max_drawdown: max_drawdown = dd

    # ── Results ───────────────────────────────────────────────────
    final = daily_equity[-1] if daily_equity else EQUITY_START
    total_return = (final - EQUITY_START) / EQUITY_START * 100
    years = len(all_dates) / 252
    cagr = ((final / EQUITY_START) ** (1 / max(years, 1)) - 1) * 100
    returns = np.diff(daily_equity) / np.array(daily_equity[:-1])
    sharpe = (returns.mean() / returns.std() * math.sqrt(252)) if returns.std() > 0 else 0
    wr = winning_trades / max(total_trades, 1) * 100
    opts_wr = options_wins / max(len(options_trades), 1) * 100

    # Annual breakdown
    annual = {}
    for year in range(2016, 2027):
        yr_dates = [d for d in all_dates if d.startswith(str(year))]
        if len(yr_dates) < 20: continue
        yr_start_idx = all_dates.index(yr_dates[0])
        yr_end_idx = all_dates.index(yr_dates[-1])
        if yr_start_idx > 0:
            yr_start_eq = daily_equity[yr_start_idx - 1]
        else:
            yr_start_eq = EQUITY_START
        yr_end_eq = daily_equity[yr_end_idx]
        yr_ret = (yr_end_eq - yr_start_eq) / yr_start_eq * 100

        spy_yr_s = float(spy_by_date.get(yr_dates[0], {}).get("c", 1))
        spy_yr_e = float(spy_by_date.get(yr_dates[-1], {}).get("c", 1))
        spy_yr = (spy_yr_e - spy_yr_s) / spy_yr_s * 100 if spy_yr_s > 0 else 0
        annual[year] = {"bot": round(yr_ret, 1), "spy": round(spy_yr, 1),
                        "alpha": round(yr_ret - spy_yr, 1)}

    # SPY total
    spy_start = float(spy_by_date.get(all_dates[0], {}).get("c", 1))
    spy_end = float(spy_by_date.get(all_dates[-1], {}).get("c", 1))
    spy_total = (spy_end - spy_start) / spy_start * 100
    spy_cagr = ((spy_end / spy_start) ** (1 / years) - 1) * 100

    return {
        "label": label,
        "final": round(final, 2), "total_return": round(total_return, 1),
        "cagr": round(cagr, 1), "sharpe": round(float(sharpe), 3),
        "max_dd": round(max_drawdown, 1),
        "stock_trades": total_trades, "stock_wr": round(wr, 1),
        "options_trades": len(options_trades), "options_wr": round(opts_wr, 1),
        "options_pnl": round(options_pnl_total, 0),
        "third_leg_pnl": round(third_leg_pnl, 0),
        "short_trades": short_trades,
        "short_wr": round(short_wins / max(short_trades, 1) * 100, 1),
        "short_pnl": round(short_pnl_total, 0),
        "spy_total": round(spy_total, 1), "spy_cagr": round(spy_cagr, 1),
        "annual": annual,
    }


# ══════════════════════════════════════════════════════════════════
print("\n[2] Running Variant A: Full system WITHOUT intraday shorts...")
t1 = time.time()
result_a = run_backtest(include_shorts=False, include_third_leg=True,
                        label="v1.0.28 (no shorts)")
print(f"  Done in {time.time()-t1:.1f}s")

print("\n[3] Running Variant B: Full system WITH intraday shorts (v2.1 hybrid)...")
t2 = time.time()
result_b = run_backtest(include_shorts=True, include_third_leg=True,
                        label="v1.0.28 + shorts v2.1")
print(f"  Done in {time.time()-t2:.1f}s")

# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"RESULTS COMPARISON — VolTradeAI v1.0.28")
print(f"{'='*70}")
print(f"\n  {'Metric':<25} {'No Shorts':>15} {'With Shorts':>15} {'SPY':>12}")
print(f"  {'-'*67}")
print(f"  {'Final Equity':<25} {'$'+str(int(result_a['final'])):>15} {'$'+str(int(result_b['final'])):>15} {'—':>12}")
print(f"  {'Total Return':<25} {result_a['total_return']:>+14.1f}% {result_b['total_return']:>+14.1f}% {result_a['spy_total']:>+11.1f}%")
print(f"  {'CAGR':<25} {result_a['cagr']:>+14.1f}% {result_b['cagr']:>+14.1f}% {result_a['spy_cagr']:>+11.1f}%")
print(f"  {'Sharpe':<25} {result_a['sharpe']:>15.3f} {result_b['sharpe']:>15.3f} {'—':>12}")
print(f"  {'Max Drawdown':<25} {result_a['max_dd']:>14.1f}% {result_b['max_dd']:>14.1f}% {'—':>12}")
print(f"  {'Stock Trades':<25} {result_a['stock_trades']:>15} {result_b['stock_trades']:>15} {'—':>12}")
print(f"  {'Stock WR':<25} {result_a['stock_wr']:>14.1f}% {result_b['stock_wr']:>14.1f}% {'—':>12}")
print(f"  {'Options P&L':<25} {'$'+str(int(result_a['options_pnl'])):>15} {'$'+str(int(result_b['options_pnl'])):>15} {'—':>12}")
print(f"  {'Third Leg P&L':<25} {'$'+str(int(result_a['third_leg_pnl'])):>15} {'$'+str(int(result_b['third_leg_pnl'])):>15} {'—':>12}")
print(f"  {'Short Trades':<25} {'N/A':>15} {result_b['short_trades']:>15} {'—':>12}")
print(f"  {'Short WR':<25} {'N/A':>15} {result_b['short_wr']:>14.1f}% {'—':>12}")
print(f"  {'Short P&L':<25} {'N/A':>15} {'$'+str(int(result_b['short_pnl'])):>15} {'—':>12}")

print(f"\n  Annual Breakdown:")
print(f"  {'Year':<6} {'No Shorts':>12} {'W/ Shorts':>12} {'SPY':>10} {'Alpha (A)':>12} {'Alpha (B)':>12}")
print(f"  {'-'*64}")
for year in sorted(result_a["annual"].keys()):
    a = result_a["annual"][year]
    b = result_b["annual"].get(year, a)
    print(f"  {year:<6} {a['bot']:>+11.1f}% {b['bot']:>+11.1f}% {a['spy']:>+9.1f}% {a['alpha']:>+11.1f}% {b['alpha']:>+11.1f}%")

beats_spy_a = result_a["cagr"] > result_a["spy_cagr"]
beats_spy_b = result_b["cagr"] > result_b["spy_cagr"]
shorts_help = result_b["cagr"] > result_a["cagr"]

print(f"\n  VERDICT:")
print(f"  Bot (no shorts) vs SPY: {'BEATS' if beats_spy_a else 'LAGS'} ({result_a['cagr']:+.1f}% vs {result_a['spy_cagr']:+.1f}%)")
print(f"  Bot (w/ shorts) vs SPY: {'BEATS' if beats_spy_b else 'LAGS'} ({result_b['cagr']:+.1f}% vs {result_b['spy_cagr']:+.1f}%)")
print(f"  Shorts impact on CAGR:  {'POSITIVE' if shorts_help else 'NEGATIVE' if result_b['cagr'] < result_a['cagr'] else 'NEUTRAL'} ({result_b['cagr'] - result_a['cagr']:+.1f}%)")
print(f"{'='*70}")

# Save results
combined = {"variant_a": result_a, "variant_b": result_b,
            "spy_cagr": result_a["spy_cagr"], "spy_total": result_a["spy_total"],
            "beats_spy_a": beats_spy_a, "beats_spy_b": beats_spy_b,
            "shorts_help": shorts_help}
with open("/tmp/backtest_v1028_results.json", "w") as f:
    json.dump(combined, f, indent=2)
print(f"\nResults saved to /tmp/backtest_v1028_results.json")
