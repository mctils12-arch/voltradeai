#!/usr/bin/env python3
"""
VolTradeAI Full-System Backtest (v2)
=====================================
Tests the ENTIRE bot system, not just ML picking:
  1. ML stock picker (real ml_model_v2)
  2. QQQ regime floor (BULL/NEUTRAL/CAUTION allocations)
  3. VRP harvest via SVXY (VXX 1.05-1.25 + declining → buy SVXY)
  4. GLD bear rotation (regime=BEAR/PANIC → sell QQQ, buy GLD)
  5. Sector concentration cap (max 25% in any one sector)
  6. Kelly-style position sizing from ml_score
  7. Simulated options overlay: CSP + CC on QQQ (Black-Scholes)

Uses precomputed feature/score cache from /tmp/backtest_sweep_features.pkl
to run each config in <1 second.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
import itertools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

os.environ["VOLTRADE_DATA_DIR"] = "/tmp/voltrade_ml_data"

CACHE_PATH = "/tmp/backtest_real_ml_cache.pkl"
SCORE_CACHE_PATH = "/tmp/backtest_sweep_features.pkl"
INITIAL_EQUITY = 100_000
START_DATE = "2016-01-01"
END_DATE = "2026-04-01"
SLIPPAGE_PCT = 0.0015

# ═══════════════════════════════════════════════════════════════════════════════
# Sector mapping (static, covers our universe)
# ═══════════════════════════════════════════════════════════════════════════════
SECTOR_MAP = {
    # Tech
    **{t: "tech" for t in ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","AVGO","ORCL","CRM","ADBE","NFLX","AMD","INTC","CSCO","QCOM","TXN","IBM","ACN","INTU","AMAT","LRCX","KLAC","ASML","MU","MCHP","ADI","MRVL","NXPI","ON","SHOP","PLTR","SNOW","CRWD","DDOG","NET","WDAY","TEAM","NOW","ZS","MDB","OKTA","VEEV","ZM","DOCU","TWLO","RBLX","U","PATH"]},
    # Financials
    **{t: "fin" for t in ["JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","SPGI","V","MA","PYPL","COF","USB","PNC","TFC","AIG","MET","PRU","SQ","COIN","HOOD","SOFI","AFRM","UPST"]},
    # Healthcare
    **{t: "hc" for t in ["UNH","JNJ","PFE","ABBV","LLY","MRK","TMO","ABT","BMY","AMGN","DHR","MDT","CVS","ELV","CI","HUM","SYK","BSX","ISRG","GILD","REGN","VRTX","BIIB","MRNA","BNTX","NVAX","ILMN","INCY"]},
    # Consumer
    **{t: "cons" for t in ["WMT","COST","PG","KO","PEP","MCD","NKE","SBUX","HD","LOW","TGT","DIS","CMCSA","TMUS","VZ","T","CHTR","BKNG","MAR","ABNB","EBAY","UBER","LYFT","ROKU","PINS","SNAP","SPOT","DKNG","PENN","OPEN"]},
    # Energy/Industrial
    **{t: "ind" for t in ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","KMI","BA","CAT","HON","UNP","GE","RTX","LMT","NOC","DE","MMM","TSLA"]},
    # Materials/Utilities/REIT
    **{t: "util" for t in ["LIN","SHW","FCX","NEM","GOLD","NEE","DUK","SO","PLD","AMT","CCI","EQIX","SPG","O","VICI","DLR","PSA","EXR","WELL","AVB"]},
    # High-beta/special
    **{t: "spec" for t in ["MSTR","GME","AMC","BBBY","SPWR","ENPH","FSLR","RIVN","LCID","NIO","FUBO","UVXY","SQQQ","TQQQ","SOXS","SOXL","LABU","LABD","TNA","BYND"]},
}

# ═══════════════════════════════════════════════════════════════════════════════
# QQQ floor + GLD rotation
# ═══════════════════════════════════════════════════════════════════════════════
def get_floor_allocation(regime, qqq_mult=1.0, use_gld_rotation=True):
    """Return (qqq_pct, gld_pct). GLD substitutes in BEAR/PANIC if enabled."""
    if regime == "BULL":
        return (min(0.95, 0.70 * qqq_mult), 0.0)
    if regime == "NEUTRAL":
        return (min(0.95, 0.90 * qqq_mult), 0.0)
    if regime == "CAUTION":
        return (min(0.95, 0.35 * qqq_mult), 0.0)
    if regime == "BEAR":
        if use_gld_rotation:
            return (0.0, 0.30)   # PR #70 GLD bear rotation
        return (0.0, 0.0)
    if regime == "PANIC":
        if use_gld_rotation:
            return (0.0, 0.15)
        return (0.0, 0.0)
    return (0.70, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# VRP signal (from bot_engine logic)
# ═══════════════════════════════════════════════════════════════════════════════
def vrp_signal(vxx_bars, vxx_idx, vrp_pct=0.15):
    """
    Returns (enter, exit) SVXY signals.
    Enter: VXX ratio 1.05-1.25 AND VXX declining over last 5 days
    Exit: regime shifts or VXX ratio outside band
    """
    if vxx_idx < 25:
        return False, False
    vxx_20 = np.mean([b["c"] for b in vxx_bars[vxx_idx-20:vxx_idx]])
    vxx_now = vxx_bars[vxx_idx]["c"]
    if vxx_20 <= 0:
        return False, False
    vxx_ratio = vxx_now / vxx_20
    # Declining: current lower than 5 days ago
    vxx_5 = vxx_bars[vxx_idx-5]["c"] if vxx_idx >= 5 else vxx_now
    declining = vxx_now < vxx_5

    enter = (1.05 <= vxx_ratio <= 1.25) and declining
    exit = (vxx_ratio > 1.30) or (vxx_ratio < 0.95)
    return enter, exit


# ═══════════════════════════════════════════════════════════════════════════════
# Options overlay (Black-Scholes simulation)
# ═══════════════════════════════════════════════════════════════════════════════
def bs_price_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    from math import log, sqrt, exp
    from scipy.stats import norm
    if sigma <= 0 or T <= 0:
        return max(K - S, 0)
    d1 = (log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_price_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    from math import log, sqrt, exp
    from scipy.stats import norm
    if sigma <= 0 or T <= 0:
        return max(S - K, 0)
    d1 = (log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)


def realized_vol(bars, idx, lookback=20):
    """20-day realized volatility, annualized."""
    if idx < lookback:
        return 0.20
    closes = np.array([b["c"] for b in bars[idx-lookback:idx+1]])
    rets = np.diff(closes) / closes[:-1]
    return float(np.std(rets) * np.sqrt(252))


# ═══════════════════════════════════════════════════════════════════════════════
# Regime
# ═══════════════════════════════════════════════════════════════════════════════
def get_regime(vxx_bars, spy_bars, vxx_idx, spy_idx):
    if vxx_idx < 30 or spy_idx < 200:
        return "NEUTRAL"
    vxx_20 = np.mean([b["c"] for b in vxx_bars[vxx_idx-20:vxx_idx]])
    vxx_ratio = vxx_bars[vxx_idx]["c"] / vxx_20 if vxx_20 > 0 else 1.0
    spy_closes = [b["c"] for b in spy_bars[:spy_idx+1]]
    spy_now = spy_closes[-1]
    ma50 = np.mean(spy_closes[-50:])
    ma200 = np.mean(spy_closes[-200:])
    spy_vs_ma50 = spy_now / ma50 if ma50 > 0 else 1.0
    below_200 = sum(1 for c in spy_closes[-30:] if c < ma200)

    if vxx_ratio > 1.35 and spy_vs_ma50 < 0.95:
        return "PANIC"
    if vxx_ratio > 1.20 or (spy_vs_ma50 < 0.95 and below_200 > 10):
        return "BEAR"
    if vxx_ratio > 1.10 or spy_vs_ma50 < 0.98:
        return "CAUTION"
    if vxx_ratio < 0.90 and spy_vs_ma50 > 1.02:
        return "BULL"
    return "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════════════
# Worker init
# ═══════════════════════════════════════════════════════════════════════════════
_GLOBAL = {}

def _df_to_alpaca_bars(df):
    return [
        {"t": idx.strftime("%Y-%m-%d"),
         "o": float(r["Open"]), "h": float(r["High"]),
         "l": float(r["Low"]), "c": float(r["Close"]),
         "v": int(r["Volume"]) if not pd.isna(r["Volume"]) else 0}
        for idx, r in df.iterrows()
    ]


def _init():
    global _GLOBAL
    if _GLOBAL:
        return
    print(f"[init] loading data + ML scores...", flush=True)
    data = pickle.load(open(CACHE_PATH, "rb"))
    all_bars = {}
    for t, df in data.items():
        key = "VXX" if t == "^VIX" else t
        if len(df) >= 30:
            all_bars[key] = _df_to_alpaca_bars(df)
    _GLOBAL["data"] = data
    _GLOBAL["all_bars"] = all_bars
    _GLOBAL["spy_bars"] = all_bars["SPY"]
    _GLOBAL["vxx_bars"] = all_bars.get("VXX", [])
    _GLOBAL["qqq_bars"] = all_bars.get("QQQ", [])
    _GLOBAL["gld_bars"] = all_bars.get("GLD", [])
    _GLOBAL["svxy_bars"] = all_bars.get("SVXY", [])
    _GLOBAL["spy_date_idx"] = {b["t"]: i for i, b in enumerate(_GLOBAL["spy_bars"])}
    _GLOBAL["vxx_date_idx"] = {b["t"]: i for i, b in enumerate(_GLOBAL["vxx_bars"])}
    _GLOBAL["qqq_date_idx"] = {b["t"]: i for i, b in enumerate(_GLOBAL["qqq_bars"])}
    _GLOBAL["gld_date_idx"] = {b["t"]: i for i, b in enumerate(_GLOBAL["gld_bars"])}
    _GLOBAL["svxy_date_idx"] = {b["t"]: i for i, b in enumerate(_GLOBAL["svxy_bars"])}

    from backtest_real_ml import CORE_UNIVERSE
    _GLOBAL["stock_tickers"] = [t for t in CORE_UNIVERSE if t in all_bars]
    _GLOBAL["tkr_date_idx"] = {t: {b["t"]: i for i, b in enumerate(all_bars[t])}
                                for t in _GLOBAL["stock_tickers"]}

    # Load precomputed ML scores
    if os.path.exists(SCORE_CACHE_PATH):
        cached = pickle.load(open(SCORE_CACHE_PATH, "rb"))
        _GLOBAL["feature_scores"] = cached["scores"]
        print(f"[init] loaded {len(cached['scores'])} ML scores from cache", flush=True)
    else:
        print("[init] ERROR: no ML score cache; run backtest_sweep.py first", flush=True)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main backtest
# ═══════════════════════════════════════════════════════════════════════════════
def run_full_system(cfg):
    _init()
    all_bars = _GLOBAL["all_bars"]
    spy_bars = _GLOBAL["spy_bars"]
    vxx_bars = _GLOBAL["vxx_bars"]
    qqq_bars = _GLOBAL["qqq_bars"]
    gld_bars = _GLOBAL["gld_bars"]
    svxy_bars = _GLOBAL["svxy_bars"]
    spy_date_idx = _GLOBAL["spy_date_idx"]
    vxx_date_idx = _GLOBAL["vxx_date_idx"]
    qqq_date_idx = _GLOBAL["qqq_date_idx"]
    gld_date_idx = _GLOBAL["gld_date_idx"]
    svxy_date_idx = _GLOBAL["svxy_date_idx"]
    stock_tickers = _GLOBAL["stock_tickers"]
    tkr_date_idx = _GLOBAL["tkr_date_idx"]
    feature_scores = _GLOBAL["feature_scores"]

    data = _GLOBAL["data"]
    spy_df = data["SPY"]
    spy_dates = [d for d in spy_df.index if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)]

    # Config
    pos_size_pct    = cfg["pos_size_pct"]
    max_positions   = cfg["max_positions"]
    hold_days       = cfg["hold_days"]
    stop_loss_pct   = cfg.get("stop_loss_pct")
    take_profit_pct = cfg.get("take_profit_pct")
    ml_threshold    = cfg["ml_threshold"]
    qqq_mult        = cfg.get("qqq_floor_multiplier", 1.0)
    use_gld         = cfg.get("use_gld_rotation", True)
    use_vrp         = cfg.get("use_vrp", True)
    vrp_pct         = cfg.get("vrp_pct", 0.15)
    use_sector_cap  = cfg.get("use_sector_cap", True)
    sector_cap_pct  = cfg.get("sector_cap_pct", 0.25)
    use_kelly       = cfg.get("use_kelly", True)
    use_options     = cfg.get("use_options", True)
    cc_pct_qqq      = cfg.get("cc_pct_qqq", 0.30)   # % of QQQ shares to CC
    csp_pct_equity  = cfg.get("csp_pct_equity", 0.15) # % equity for CSP

    # State
    cash = INITIAL_EQUITY
    qqq_shares = 0
    gld_shares = 0
    svxy_shares = 0
    positions = {}       # stock positions
    options_open = []    # list of {ticker, type, strike, expiry_days_left, shares_hedged, premium, opened_idx}
    equity_curve = []
    trades = []
    opt_pnl_total = 0.0
    vrp_pnl_total = 0.0
    regime_log = []

    for day_i, current_date in enumerate(spy_dates):
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in spy_date_idx:
            continue
        spy_idx = spy_date_idx[date_str]
        vxx_idx = vxx_date_idx.get(date_str, 0)
        regime = get_regime(vxx_bars, spy_bars, vxx_idx, spy_idx)
        regime_log.append(regime)

        qqq_idx = qqq_date_idx.get(date_str)
        gld_idx = gld_date_idx.get(date_str)
        svxy_idx = svxy_date_idx.get(date_str)
        if qqq_idx is None: continue

        qqq_price = qqq_bars[qqq_idx]["c"]
        gld_price = gld_bars[gld_idx]["c"] if gld_idx is not None else 0
        svxy_price = svxy_bars[svxy_idx]["c"] if svxy_idx is not None else 0

        # ─── Mark stock positions ───
        to_close_stocks = []
        stock_mv = 0.0
        for tkr, pos in positions.items():
            idx_map = tkr_date_idx.get(tkr, {})
            tidx = idx_map.get(date_str)
            if tidx is None:
                for i in range(len(all_bars[tkr])-1, -1, -1):
                    if all_bars[tkr][i]["t"] <= date_str:
                        tidx = i; break
            if tidx is None: continue
            cur_price = all_bars[tkr][tidx]["c"]
            stock_mv += pos["shares"] * cur_price
            ret = (cur_price / pos["entry_price"]) - 1
            if stop_loss_pct is not None and ret <= stop_loss_pct:
                to_close_stocks.append((tkr, "stop"))
            elif take_profit_pct is not None and ret >= take_profit_pct:
                to_close_stocks.append((tkr, "tp"))
            elif (current_date - pos["entry_date"]).days >= hold_days:
                to_close_stocks.append((tkr, "time"))

        for tkr, reason in to_close_stocks:
            pos = positions.pop(tkr)
            tbars = all_bars[tkr]
            tidx = tkr_date_idx[tkr].get(date_str)
            if tidx is None:
                for i in range(len(tbars)-1, -1, -1):
                    if tbars[i]["t"] <= date_str:
                        tidx = i; break
            exit_price = tbars[tidx]["c"] * (1 - SLIPPAGE_PCT)
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            cash += pos["shares"] * exit_price
            stock_mv -= pos["shares"] * tbars[tidx]["c"]
            trades.append({"ticker": tkr, "pnl": pnl, "reason": reason,
                           "ret_pct": (exit_price/pos["entry_price"]-1)*100})

        # ─── ML picker: new entries every 5 days ───
        if day_i % 5 == 0 and len(positions) < max_positions and regime not in ("PANIC", "BEAR"):
            scored = []
            for tkr in stock_tickers:
                if tkr in positions: continue
                key = (tkr, date_str)
                if key not in feature_scores: continue
                score = feature_scores[key] / 100.0
                if score >= ml_threshold:
                    tidx = tkr_date_idx[tkr].get(date_str)
                    if tidx is None: continue
                    scored.append((tkr, tidx, score))
            scored.sort(key=lambda x: -x[2])

            # Sector cap check
            current_sectors = {}
            for tkr, pos in positions.items():
                sec = SECTOR_MAP.get(tkr, "other")
                idx_map = tkr_date_idx.get(tkr, {})
                tidx = idx_map.get(date_str)
                if tidx is None: continue
                cur_price = all_bars[tkr][tidx]["c"]
                current_sectors[sec] = current_sectors.get(sec, 0) + pos["shares"] * cur_price

            slots = max_positions - len(positions)
            equity = cash + stock_mv + qqq_shares * qqq_price + gld_shares * gld_price + svxy_shares * svxy_price
            for tkr, tidx, score in scored[:slots*3]:   # scan extra in case of sector rejections
                if len(positions) >= max_positions: break
                # Sector cap
                sec = SECTOR_MAP.get(tkr, "other")
                sec_exposure = current_sectors.get(sec, 0) / equity if equity > 0 else 0
                if use_sector_cap and sec_exposure >= sector_cap_pct:
                    continue

                # Kelly sizing: higher score → bigger size
                if use_kelly:
                    # ml_score is win_prob. Edge = 2*p - 1 (assuming 1:1 payoff)
                    edge = max(0, 2*score - 1)
                    kelly_frac = min(edge, pos_size_pct)   # cap at pos_size_pct
                    alloc_pct = kelly_frac
                else:
                    alloc_pct = pos_size_pct

                tbars = all_bars[tkr]
                entry_price = tbars[tidx]["c"] * (1 + SLIPPAGE_PCT)
                dollar_alloc = equity * alloc_pct
                shares = int(dollar_alloc / entry_price)
                if shares <= 0: continue
                cost = shares * entry_price
                if cost > cash: continue
                cash -= cost
                positions[tkr] = {"entry_price": entry_price, "entry_date": current_date,
                                  "shares": shares, "ml_score": score, "sector": sec}
                stock_mv += shares * tbars[tidx]["c"]
                current_sectors[sec] = current_sectors.get(sec, 0) + shares * entry_price

        # ─── VRP harvest ───
        if use_vrp and svxy_price > 0:
            enter_vrp, exit_vrp = vrp_signal(vxx_bars, vxx_idx)
            if enter_vrp and svxy_shares == 0 and regime in ("NEUTRAL", "CAUTION"):
                equity = cash + stock_mv + qqq_shares * qqq_price + gld_shares * gld_price
                target = equity * vrp_pct
                shares = int(target / svxy_price)
                if shares > 0 and shares * svxy_price <= cash:
                    cash -= shares * svxy_price * (1 + SLIPPAGE_PCT)
                    svxy_shares = shares
            elif (exit_vrp or regime in ("BEAR", "PANIC")) and svxy_shares > 0:
                exit_val = svxy_shares * svxy_price * (1 - SLIPPAGE_PCT)
                # P&L tracking
                vrp_pnl_total += exit_val - (svxy_shares * svxy_price)  # rough
                cash += exit_val
                svxy_shares = 0

        # ─── Options overlay: CSP (cash-secured puts) on QQQ ───
        # + Covered calls on QQQ holdings
        # Open monthly (every 21 days), sell puts/calls with ~30 DTE at 5-10% OTM
        if use_options and day_i % 21 == 0 and regime not in ("PANIC", "BEAR"):
            iv = realized_vol(qqq_bars, qqq_idx, lookback=30)
            T = 30 / 365.0  # 30 days to expiry
            r = 0.04        # ~4% risk-free

            # Cash-Secured Put: sell OTM put, collect premium
            csp_strike = qqq_price * 0.95   # 5% OTM
            csp_premium = bs_price_put(qqq_price, csp_strike, T, r, iv)
            equity = cash + stock_mv + qqq_shares * qqq_price + gld_shares * gld_price + svxy_shares * svxy_price
            csp_dollars = equity * csp_pct_equity
            csp_contracts = int(csp_dollars / (csp_strike * 100))  # each contract = 100 shares collateral
            if csp_contracts > 0 and csp_contracts * csp_strike * 100 <= cash:
                cash += csp_contracts * csp_premium * 100   # collect premium
                options_open.append({
                    "type": "csp", "strike": csp_strike, "expiry_idx": qqq_idx + 21,
                    "contracts": csp_contracts, "premium": csp_premium,
                    "opened_price": qqq_price, "opened_idx": qqq_idx,
                    "collateral": csp_contracts * csp_strike * 100,
                })
                cash -= csp_contracts * csp_strike * 100   # reserve collateral

            # Covered Call on QQQ holdings (if we have 100+ shares)
            if qqq_shares >= 100:
                cc_shares_to_cover = int(qqq_shares * cc_pct_qqq / 100) * 100
                cc_contracts = cc_shares_to_cover // 100
                if cc_contracts > 0:
                    cc_strike = qqq_price * 1.05   # 5% OTM
                    cc_premium = bs_price_call(qqq_price, cc_strike, T, r, iv)
                    cash += cc_contracts * cc_premium * 100
                    options_open.append({
                        "type": "cc", "strike": cc_strike, "expiry_idx": qqq_idx + 21,
                        "contracts": cc_contracts, "premium": cc_premium,
                        "opened_price": qqq_price, "opened_idx": qqq_idx,
                        "shares_hedged": cc_shares_to_cover,
                    })

        # ─── Settle expired options ───
        opts_still = []
        for opt in options_open:
            if qqq_idx < opt["expiry_idx"]:
                opts_still.append(opt); continue
            # Expiration
            final_qqq = qqq_price
            if opt["type"] == "csp":
                collateral = opt["collateral"]
                if final_qqq < opt["strike"]:
                    # Put in money → assignment. Loss = (strike - final) * contracts * 100
                    loss = (opt["strike"] - final_qqq) * opt["contracts"] * 100
                    cash += collateral - loss
                    opt_pnl_total += opt["contracts"] * opt["premium"] * 100 - loss
                else:
                    cash += collateral
                    opt_pnl_total += opt["contracts"] * opt["premium"] * 100
            elif opt["type"] == "cc":
                if final_qqq > opt["strike"]:
                    # Call in money → cap gain at strike
                    # Forgone upside: (final - strike) * contracts * 100
                    forgone = (final_qqq - opt["strike"]) * opt["contracts"] * 100
                    opt_pnl_total += opt["contracts"] * opt["premium"] * 100 - forgone
                else:
                    opt_pnl_total += opt["contracts"] * opt["premium"] * 100
        options_open = opts_still

        # ─── Floor rebalance monthly (QQQ/GLD) ───
        if day_i % 21 == 0:
            equity = cash + stock_mv + qqq_shares * qqq_price + gld_shares * gld_price + svxy_shares * svxy_price
            qqq_pct, gld_pct = get_floor_allocation(regime, qqq_mult=qqq_mult, use_gld_rotation=use_gld)
            target_qqq_val = equity * qqq_pct
            target_qqq = int(target_qqq_val / qqq_price) if qqq_price > 0 else 0
            target_gld_val = equity * gld_pct
            target_gld = int(target_gld_val / gld_price) if gld_price > 0 else 0

            # Rebalance QQQ
            diff_qqq = target_qqq - qqq_shares
            if diff_qqq != 0:
                trade_cost = diff_qqq * qqq_price * (1 + SLIPPAGE_PCT if diff_qqq > 0 else 1 - SLIPPAGE_PCT)
                if diff_qqq > 0 and trade_cost <= cash:
                    cash -= trade_cost; qqq_shares += diff_qqq
                elif diff_qqq < 0:
                    cash += -trade_cost; qqq_shares += diff_qqq
            # Rebalance GLD
            diff_gld = target_gld - gld_shares
            if diff_gld != 0:
                trade_cost = diff_gld * gld_price * (1 + SLIPPAGE_PCT if diff_gld > 0 else 1 - SLIPPAGE_PCT)
                if diff_gld > 0 and trade_cost <= cash:
                    cash -= trade_cost; gld_shares += diff_gld
                elif diff_gld < 0:
                    cash += -trade_cost; gld_shares += diff_gld

        # Record equity
        equity = cash + stock_mv + qqq_shares * qqq_price + gld_shares * gld_price + svxy_shares * svxy_price
        equity_curve.append(equity)

    # Close everything at end
    final_date = spy_dates[-1].strftime("%Y-%m-%d")
    final_qqq = qqq_bars[qqq_date_idx.get(final_date, len(qqq_bars)-1)]["c"]
    final_gld = gld_bars[gld_date_idx.get(final_date, len(gld_bars)-1)]["c"] if gld_bars else 0
    final_svxy = svxy_bars[svxy_date_idx.get(final_date, len(svxy_bars)-1)]["c"] if svxy_bars else 0

    for tkr, pos in list(positions.items()):
        tbars = all_bars[tkr]
        for i in range(len(tbars)-1, -1, -1):
            if tbars[i]["t"] <= final_date:
                exit_price = tbars[i]["c"] * (1 - SLIPPAGE_PCT)
                cash += pos["shares"] * exit_price
                trades.append({"ticker": tkr, "pnl": (exit_price-pos["entry_price"])*pos["shares"], "reason": "end",
                               "ret_pct": (exit_price/pos["entry_price"]-1)*100})
                break
    cash += qqq_shares * final_qqq
    cash += gld_shares * final_gld
    cash += svxy_shares * final_svxy
    final_equity = cash

    # Stats
    equities = np.array(equity_curve)
    if len(equities) < 2:
        return {"config": cfg, "error": "no curve"}
    returns = np.diff(equities) / equities[:-1]
    years = (spy_dates[-1] - spy_dates[0]).days / 365.25
    cagr = (final_equity / INITIAL_EQUITY) ** (1/years) - 1 if final_equity > 0 else -1
    sharpe = np.mean(returns) / np.std(returns) * math.sqrt(252) if np.std(returns) > 0 else 0
    neg = returns[returns < 0]
    sortino = np.mean(returns) / np.std(neg) * math.sqrt(252) if len(neg) > 0 and np.std(neg) > 0 else 0
    peak = np.maximum.accumulate(equities)
    dd = (equities - peak) / peak
    max_dd = float(dd.min())

    n_trades = len(trades)
    n_win = sum(1 for t in trades if t["pnl"] > 0)

    return {
        "config": cfg,
        "cagr_pct": round(cagr*100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(abs(max_dd)*100, 2),
        "final_equity": round(final_equity, 0),
        "n_trades": n_trades,
        "win_rate_pct": round(n_win/max(1,n_trades)*100, 1),
        "options_pnl": round(opt_pnl_total, 0),
        "vrp_pnl": round(vrp_pnl_total, 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # Focused grid: hand-picked around known-good regions (~100 configs)
    pos_sizes   = [0.05, 0.08, 0.10]
    max_pos     = [3, 5]
    hold_days   = [10, 20]
    stop_losses = [-0.05, -0.10]
    take_profs  = [None, 0.20]
    thresholds  = [0.55, 0.60]
    qqq_mults   = [1.0, 1.3]
    gld_opts    = [True]
    vrp_opts    = [True]
    sector_opts = [True]
    kelly_opts  = [False]   # Kelly tended to underallocate; default off
    options_opts = [True, False]

    if args.smoke:
        configs = [{
            "pos_size_pct": 0.05, "max_positions": 5, "hold_days": 10,
            "stop_loss_pct": -0.08, "take_profit_pct": 0.20,
            "ml_threshold": 0.55, "qqq_floor_multiplier": 1.0,
            "use_gld_rotation": True, "use_vrp": True, "vrp_pct": 0.15,
            "use_sector_cap": True, "sector_cap_pct": 0.25,
            "use_kelly": True, "use_options": True,
            "cc_pct_qqq": 0.30, "csp_pct_equity": 0.10,
        }]
    else:
        configs = []
        for ps, mp, hd, sl, tp, th, qm, gld, vrp, sec, kel, opt in itertools.product(
                pos_sizes, max_pos, hold_days, stop_losses, take_profs,
                thresholds, qqq_mults, gld_opts, vrp_opts, sector_opts, kelly_opts, options_opts):
            if ps * mp > 0.85: continue
            configs.append({
                "pos_size_pct": ps, "max_positions": mp, "hold_days": hd,
                "stop_loss_pct": sl, "take_profit_pct": tp,
                "ml_threshold": th, "qqq_floor_multiplier": qm,
                "use_gld_rotation": gld, "use_vrp": vrp, "vrp_pct": 0.15,
                "use_sector_cap": sec, "sector_cap_pct": 0.25,
                "use_kelly": kel, "use_options": opt,
                "cc_pct_qqq": 0.30, "csp_pct_equity": 0.10,
            })

    print(f"[sweep] {len(configs)} configurations")

    t0 = time.time()
    results = []
    # Single-process for reliability since cache is in-memory
    _init()
    for i, cfg in enumerate(configs):
        try:
            r = run_full_system(cfg)
            results.append(r)
            if (i+1) % 20 == 0 or i < 5 or args.smoke:
                c = r.get("config", {})
                print(f"[{i+1}/{len(configs)}] CAGR={r.get('cagr_pct',0):.1f}% DD={r.get('max_dd_pct',0):.1f}% Sortino={r.get('sortino',0):.2f} opt_pnl={r.get('options_pnl',0):.0f}", flush=True)
        except Exception as e:
            print(f"[{i+1}] ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()

    print(f"\n[sweep] done in {(time.time()-t0)/60:.1f} min")

    SPY_BASELINE_DD = 33.72
    qualifying = [r for r in results if "error" not in r and r.get("max_dd_pct", 100) < SPY_BASELINE_DD]
    print(f"{len(qualifying)}/{len(results)} configs meet DD < {SPY_BASELINE_DD}%")

    by_cagr = sorted(qualifying, key=lambda r: -r["cagr_pct"])[:15]
    print(f"\n{'='*120}\nTOP 15 BY CAGR (DD < SPY's {SPY_BASELINE_DD}%)\n{'='*120}")
    print(f"{'CAGR':>7} {'MaxDD':>7} {'Sortino':>8} {'Sharpe':>7} {'Trades':>7} {'Win%':>6} {'OptPnL':>9}  Config")
    for r in by_cagr:
        c = r["config"]
        tag = f"sz={c['pos_size_pct']:.0%} np={c['max_positions']} h={c['hold_days']}d sl={c['stop_loss_pct']} tp={c['take_profit_pct']} th={c['ml_threshold']} qqq={c['qqq_floor_multiplier']}x gld={c.get('use_gld_rotation',False):d} vrp={c.get('use_vrp',False):d} sec={c.get('use_sector_cap',False):d} kel={c.get('use_kelly',False):d} opt={c.get('use_options',False):d}"
        print(f"{r['cagr_pct']:>6.1f}% {r['max_dd_pct']:>6.1f}% {r['sortino']:>8.2f} {r['sharpe']:>7.2f} {r['n_trades']:>7} {r['win_rate_pct']:>5.1f}% {r['options_pnl']:>9.0f}  {tag}")

    by_sortino = sorted(qualifying, key=lambda r: -r["sortino"])[:15]
    print(f"\n{'='*120}\nTOP 15 BY SORTINO (DD < SPY's {SPY_BASELINE_DD}%)\n{'='*120}")
    print(f"{'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'Trades':>7} {'Win%':>6}  Config")
    for r in by_sortino:
        c = r["config"]
        tag = f"sz={c['pos_size_pct']:.0%} np={c['max_positions']} h={c['hold_days']}d sl={c['stop_loss_pct']} tp={c['take_profit_pct']} th={c['ml_threshold']} qqq={c['qqq_floor_multiplier']}x gld={c.get('use_gld_rotation',False):d} vrp={c.get('use_vrp',False):d} sec={c.get('use_sector_cap',False):d} kel={c.get('use_kelly',False):d} opt={c.get('use_options',False):d}"
        print(f"{r['sortino']:>8.2f} {r['cagr_pct']:>6.1f}% {r['max_dd_pct']:>6.1f}% {r['sharpe']:>7.2f} {r['n_trades']:>7} {r['win_rate_pct']:>5.1f}%  {tag}")

    out_path = "/home/user/workspace/voltradeai/backtest_full_system_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "all_results": results,
            "qualifying": qualifying,
            "top15_by_cagr": by_cagr,
            "top15_by_sortino": by_sortino,
            "spy_baseline_dd_pct": SPY_BASELINE_DD,
            "spy_baseline_cagr_pct": 14.01,
            "swept_at": datetime.utcnow().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
