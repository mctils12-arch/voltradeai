#!/usr/bin/env python3
"""
VolTradeAI Full-System Backtest v3 — "Path to 45% CAGR"
========================================================
Builds on v2 (ML picker + QQQ floor + VRP + GLD rotation + sector cap)
with these alpha upgrades:

  U1. Daily scan (instead of every-5-days) — catches moves 5× faster
  U2. Top-K concentration — take top N ML scores globally, not threshold fill
  U3. Conviction sizing — size = base * (1 + conviction_gain * (score-0.5))
  U4. Leveraged floor — TQQQ replaces QQQ in BULL regime (optional mix)
  U5. Bear short overlay — SQQQ allocation in BEAR/PANIC (real bot does
      intraday shorts; this is the daily-bar proxy)
  U6. Leveraged names eligible as stock picks (TQQQ/SOXL/TNA/LABU in ML pool)

All upgrades are toggleable via cfg so we can isolate each effect.
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

# Exclude from ML-pick pool (we'll use these as floor/hedge instruments only)
FLOOR_TICKERS = {"QQQ", "TQQQ", "SQQQ", "SPY", "SPXL", "SPXU", "UVXY", "SOXS", "LABD"}

# Leveraged/high-beta names — allowed as ML picks but optionally down-weighted
LEVERAGED_PICK_POOL = {"SOXL", "TNA", "LABU"}

SECTOR_MAP = {
    **{t: "tech" for t in ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","AVGO","ORCL","CRM","ADBE","NFLX","AMD","INTC","CSCO","QCOM","TXN","IBM","ACN","INTU","AMAT","LRCX","KLAC","ASML","MU","MCHP","ADI","MRVL","NXPI","ON","SHOP","PLTR","SNOW","CRWD","DDOG","NET","WDAY","TEAM","NOW","ZS","MDB","OKTA","VEEV","ZM","DOCU","TWLO","RBLX","U","PATH"]},
    **{t: "fin" for t in ["JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","SPGI","V","MA","PYPL","COF","USB","PNC","TFC","AIG","MET","PRU","SQ","COIN","HOOD","SOFI","AFRM","UPST"]},
    **{t: "hc" for t in ["UNH","JNJ","PFE","ABBV","LLY","MRK","TMO","ABT","BMY","AMGN","DHR","MDT","CVS","ELV","CI","HUM","SYK","BSX","ISRG","GILD","REGN","VRTX","BIIB","MRNA","BNTX","NVAX","ILMN","INCY"]},
    **{t: "cons" for t in ["WMT","COST","PG","KO","PEP","MCD","NKE","SBUX","HD","LOW","TGT","DIS","CMCSA","TMUS","VZ","T","CHTR","BKNG","MAR","ABNB","EBAY","UBER","LYFT","ROKU","PINS","SNAP","SPOT","DKNG","PENN","OPEN"]},
    **{t: "ind" for t in ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","KMI","BA","CAT","HON","UNP","GE","RTX","LMT","NOC","DE","MMM","TSLA"]},
    **{t: "util" for t in ["LIN","SHW","FCX","NEM","GOLD","NEE","DUK","SO","PLD","AMT","CCI","EQIX","SPG","O","VICI","DLR","PSA","EXR","WELL","AVB"]},
    **{t: "spec" for t in ["MSTR","GME","AMC","BBBY","SPWR","ENPH","FSLR","RIVN","LCID","NIO","FUBO","UVXY","SQQQ","TQQQ","SOXS","SOXL","LABU","LABD","TNA","BYND"]},
}


def get_floor_allocation(regime, qqq_mult=1.0, use_gld_rotation=True,
                         use_tqqq_bull=False, tqqq_fraction=0.5):
    """
    Return dict of {symbol: pct}.
    When use_tqqq_bull=True: BULL regime splits floor between QQQ and TQQQ.
      tqqq_fraction = what % of floor goes to TQQQ (0 = all QQQ, 1 = all TQQQ).
    """
    if regime == "BULL":
        total = min(0.95, 0.70 * qqq_mult)
        if use_tqqq_bull:
            return {"QQQ": total * (1 - tqqq_fraction), "TQQQ": total * tqqq_fraction}
        return {"QQQ": total}
    if regime == "NEUTRAL":
        return {"QQQ": min(0.95, 0.90 * qqq_mult)}
    if regime == "CAUTION":
        return {"QQQ": min(0.95, 0.35 * qqq_mult)}
    if regime == "BEAR":
        out = {}
        if use_gld_rotation:
            out["GLD"] = 0.30
        return out
    if regime == "PANIC":
        out = {}
        if use_gld_rotation:
            out["GLD"] = 0.15
        return out
    return {"QQQ": 0.70}


def vrp_signal(vxx_bars, vxx_idx, vrp_pct=0.15):
    if vxx_idx < 25:
        return False, False
    vxx_20 = np.mean([b["c"] for b in vxx_bars[vxx_idx-20:vxx_idx]])
    vxx_now = vxx_bars[vxx_idx]["c"]
    if vxx_20 <= 0:
        return False, False
    vxx_ratio = vxx_now / vxx_20
    vxx_5 = vxx_bars[vxx_idx-5]["c"] if vxx_idx >= 5 else vxx_now
    declining = vxx_now < vxx_5
    enter = (1.05 <= vxx_ratio <= 1.25) and declining
    exit = (vxx_ratio > 1.30) or (vxx_ratio < 0.95)
    return enter, exit


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
    _GLOBAL["tqqq_bars"] = all_bars.get("TQQQ", [])
    _GLOBAL["sqqq_bars"] = all_bars.get("SQQQ", [])
    for sym in ["spy","vxx","qqq","gld","svxy","tqqq","sqqq"]:
        bars = _GLOBAL[f"{sym}_bars"]
        _GLOBAL[f"{sym}_date_idx"] = {b["t"]: i for i, b in enumerate(bars)}

    from backtest_real_ml import CORE_UNIVERSE
    _GLOBAL["stock_tickers"] = [t for t in CORE_UNIVERSE if t in all_bars]
    _GLOBAL["tkr_date_idx"] = {t: {b["t"]: i for i, b in enumerate(all_bars[t])}
                                for t in _GLOBAL["stock_tickers"]}

    if os.path.exists(SCORE_CACHE_PATH):
        cached = pickle.load(open(SCORE_CACHE_PATH, "rb"))
        _GLOBAL["feature_scores"] = cached["scores"]
        print(f"[init] loaded {len(cached['scores'])} ML scores from cache", flush=True)
    else:
        print("[init] ERROR: no ML score cache", flush=True)
        sys.exit(1)


def _rebalance_to_target(cash, shares, price, target_pct, equity):
    """Adjust shares toward target_pct * equity. Matches v2 semantics exactly."""
    if price <= 0:
        return cash, shares
    target_val = equity * target_pct
    target_shares = int(target_val / price)
    diff = target_shares - shares
    if diff == 0:
        return cash, shares
    # v2 used 1+SLIPPAGE on buys AND sells (multiplied by diff which carries sign)
    # but computed trade_cost as diff*price*(1+slip if buy else 1-slip)
    if diff > 0:
        trade_cost = diff * price * (1 + SLIPPAGE_PCT)
        if trade_cost <= cash:
            cash -= trade_cost
            shares += diff
        # else: skip (matches v2)
    else:
        trade_cost = diff * price * (1 - SLIPPAGE_PCT)  # diff is negative
        cash += -trade_cost  # positive proceeds
        shares += diff
    return cash, shares


def run_full_system(cfg):
    _init()
    all_bars = _GLOBAL["all_bars"]
    spy_bars = _GLOBAL["spy_bars"]
    vxx_bars = _GLOBAL["vxx_bars"]
    qqq_bars = _GLOBAL["qqq_bars"]
    gld_bars = _GLOBAL["gld_bars"]
    svxy_bars = _GLOBAL["svxy_bars"]
    tqqq_bars = _GLOBAL["tqqq_bars"]
    sqqq_bars = _GLOBAL["sqqq_bars"]
    spy_date_idx = _GLOBAL["spy_date_idx"]
    vxx_date_idx = _GLOBAL["vxx_date_idx"]
    qqq_date_idx = _GLOBAL["qqq_date_idx"]
    gld_date_idx = _GLOBAL["gld_date_idx"]
    svxy_date_idx = _GLOBAL["svxy_date_idx"]
    tqqq_date_idx = _GLOBAL["tqqq_date_idx"]
    sqqq_date_idx = _GLOBAL["sqqq_date_idx"]
    stock_tickers = _GLOBAL["stock_tickers"]
    tkr_date_idx = _GLOBAL["tkr_date_idx"]
    feature_scores = _GLOBAL["feature_scores"]

    data = _GLOBAL["data"]
    spy_df = data["SPY"]
    spy_dates = [d for d in spy_df.index if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)]

    # Config with defaults
    pos_size_pct    = cfg["pos_size_pct"]
    max_positions   = cfg["max_positions"]
    hold_days       = cfg["hold_days"]
    stop_loss_pct   = cfg.get("stop_loss_pct")
    take_profit_pct = cfg.get("take_profit_pct")
    ml_threshold    = cfg.get("ml_threshold", 0.55)
    qqq_mult        = cfg.get("qqq_floor_multiplier", 1.0)
    use_gld         = cfg.get("use_gld_rotation", True)
    use_vrp         = cfg.get("use_vrp", True)
    vrp_pct         = cfg.get("vrp_pct", 0.15)
    use_sector_cap  = cfg.get("use_sector_cap", True)
    sector_cap_pct  = cfg.get("sector_cap_pct", 0.25)

    # v3 upgrades
    scan_every_days      = cfg.get("scan_every_days", 5)
    use_top_k            = cfg.get("use_top_k", False)
    top_k_min_score      = cfg.get("top_k_min_score", 0.50)
    use_conviction_size  = cfg.get("use_conviction_size", False)
    conviction_gain      = cfg.get("conviction_gain", 1.0)
    use_tqqq_bull        = cfg.get("use_tqqq_bull", False)
    tqqq_fraction        = cfg.get("tqqq_fraction", 0.5)
    use_sqqq_bear        = cfg.get("use_sqqq_bear", False)
    sqqq_bear_pct        = cfg.get("sqqq_bear_pct", 0.30)
    allow_leveraged_picks = cfg.get("allow_leveraged_picks", True)

    # State
    cash = INITIAL_EQUITY
    # Floor holdings dict (ticker -> shares)
    floor_shares = {"QQQ": 0, "GLD": 0, "TQQQ": 0, "SQQQ": 0}
    svxy_shares = 0
    positions = {}
    equity_curve = []
    trades = []
    vrp_pnl_total = 0.0
    regime_log = []

    # Helper bars/idx lookup
    floor_bars = {"QQQ": qqq_bars, "GLD": gld_bars, "TQQQ": tqqq_bars, "SQQQ": sqqq_bars}
    floor_idx = {"QQQ": qqq_date_idx, "GLD": gld_date_idx, "TQQQ": tqqq_date_idx, "SQQQ": sqqq_date_idx}

    def price_of(sym, date_str):
        idx = floor_idx[sym].get(date_str)
        if idx is None:
            return 0
        return floor_bars[sym][idx]["c"]

    for day_i, current_date in enumerate(spy_dates):
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in spy_date_idx:
            continue
        spy_idx = spy_date_idx[date_str]
        vxx_idx = vxx_date_idx.get(date_str, 0)
        regime = get_regime(vxx_bars, spy_bars, vxx_idx, spy_idx)
        regime_log.append(regime)

        qqq_price = price_of("QQQ", date_str) or (qqq_bars[-1]["c"] if qqq_bars else 0)
        gld_price = price_of("GLD", date_str)
        tqqq_price = price_of("TQQQ", date_str)
        sqqq_price = price_of("SQQQ", date_str)
        svxy_idx = svxy_date_idx.get(date_str)
        svxy_price = svxy_bars[svxy_idx]["c"] if svxy_idx is not None else 0

        # Mark stock positions
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

        # ─── ML picker ───
        do_scan = (day_i % scan_every_days == 0) and len(positions) < max_positions \
                  and regime not in ("PANIC", "BEAR")
        if do_scan:
            scored = []
            for tkr in stock_tickers:
                if tkr in positions: continue
                if tkr in FLOOR_TICKERS: continue
                if (not allow_leveraged_picks) and tkr in LEVERAGED_PICK_POOL:
                    continue
                key = (tkr, date_str)
                if key not in feature_scores: continue
                score = feature_scores[key] / 100.0
                min_score = top_k_min_score if use_top_k else ml_threshold
                if score >= min_score:
                    tidx = tkr_date_idx[tkr].get(date_str)
                    if tidx is None: continue
                    scored.append((tkr, tidx, score))
            scored.sort(key=lambda x: -x[2])

            # Sector tally
            current_sectors = {}
            for tkr, pos in positions.items():
                sec = SECTOR_MAP.get(tkr, "other")
                tidx = tkr_date_idx.get(tkr, {}).get(date_str)
                if tidx is None: continue
                cur_price = all_bars[tkr][tidx]["c"]
                current_sectors[sec] = current_sectors.get(sec, 0) + pos["shares"] * cur_price

            slots = max_positions - len(positions)
            floor_val = sum(floor_shares[s] * (price_of(s, date_str) or 0) for s in floor_shares)
            equity = cash + stock_mv + floor_val + svxy_shares * svxy_price

            # Top-K mode: take only best slots picks. Standard: fill slots by rank.
            candidates = scored[:slots * 3]  # scan extra for sector rejections

            for tkr, tidx, score in candidates:
                if len(positions) >= max_positions: break
                sec = SECTOR_MAP.get(tkr, "other")
                sec_exposure = current_sectors.get(sec, 0) / equity if equity > 0 else 0
                if use_sector_cap and sec_exposure >= sector_cap_pct:
                    continue

                # Conviction sizing: size scales with how much score exceeds 0.5
                if use_conviction_size:
                    edge = max(0, score - 0.5)  # e.g. 0.60 → 0.10
                    alloc_pct = pos_size_pct * (1 + conviction_gain * (edge / 0.1))
                    alloc_pct = min(alloc_pct, pos_size_pct * 2.5)  # cap
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
                floor_val = sum(floor_shares[s] * (price_of(s, date_str) or 0) for s in floor_shares)
                equity = cash + stock_mv + floor_val
                target = equity * vrp_pct
                shares = int(target / svxy_price)
                if shares > 0 and shares * svxy_price <= cash:
                    cash -= shares * svxy_price * (1 + SLIPPAGE_PCT)
                    svxy_shares = shares
            elif (exit_vrp or regime in ("BEAR", "PANIC")) and svxy_shares > 0:
                exit_val = svxy_shares * svxy_price * (1 - SLIPPAGE_PCT)
                vrp_pnl_total += exit_val - (svxy_shares * svxy_price)
                cash += exit_val
                svxy_shares = 0

        # ─── Floor rebalance monthly ───
        if day_i % 21 == 0:
            floor_val = sum(floor_shares[s] * (price_of(s, date_str) or 0) for s in floor_shares)
            equity = cash + stock_mv + floor_val + svxy_shares * svxy_price

            allocs = get_floor_allocation(
                regime, qqq_mult=qqq_mult, use_gld_rotation=use_gld,
                use_tqqq_bull=use_tqqq_bull, tqqq_fraction=tqqq_fraction,
            )

            # Add SQQQ bear short if configured
            if use_sqqq_bear and regime in ("BEAR", "PANIC"):
                allocs["SQQQ"] = sqqq_bear_pct

            # Zero out any floor ticker not in current allocation
            for sym in list(floor_shares.keys()):
                if sym not in allocs:
                    allocs[sym] = 0.0

            # Rebalance each
            for sym, target_pct in allocs.items():
                p = price_of(sym, date_str)
                if p <= 0:
                    continue
                cash, floor_shares[sym] = _rebalance_to_target(
                    cash, floor_shares[sym], p, target_pct, equity
                )

        # Record equity
        floor_val = sum(floor_shares[s] * (price_of(s, date_str) or 0) for s in floor_shares)
        equity = cash + stock_mv + floor_val + svxy_shares * svxy_price
        equity_curve.append(equity)

    # Close at end
    final_date = spy_dates[-1].strftime("%Y-%m-%d")
    for tkr, pos in list(positions.items()):
        tbars = all_bars[tkr]
        for i in range(len(tbars)-1, -1, -1):
            if tbars[i]["t"] <= final_date:
                exit_price = tbars[i]["c"] * (1 - SLIPPAGE_PCT)
                cash += pos["shares"] * exit_price
                trades.append({"ticker": tkr, "pnl": (exit_price-pos["entry_price"])*pos["shares"], "reason": "end",
                               "ret_pct": (exit_price/pos["entry_price"]-1)*100})
                break
    for sym, sh in floor_shares.items():
        if sh > 0:
            p = price_of(sym, final_date)
            if p <= 0:
                # fallback to last
                bars = floor_bars[sym]
                p = bars[-1]["c"] if bars else 0
            cash += sh * p
    if svxy_shares > 0:
        p = svxy_bars[-1]["c"] if svxy_bars else 0
        cash += svxy_shares * p
    final_equity = cash

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
        "vrp_pnl": round(vrp_pnl_total, 0),
    }


# ─── Config grid builder ──────────────────────────────────────────────────────
def build_configs(mode):
    if mode == "baseline":
        # Exact v2 winner — sanity check
        return [{
            "name": "v2-winner-baseline",
            "pos_size_pct": 0.10, "max_positions": 5, "hold_days": 10,
            "stop_loss_pct": -0.10, "take_profit_pct": None, "ml_threshold": 0.55,
            "qqq_floor_multiplier": 1.3, "use_gld_rotation": True, "use_vrp": True,
            "use_sector_cap": True,
            "scan_every_days": 5, "use_top_k": False, "use_conviction_size": False,
            "use_tqqq_bull": False, "use_sqqq_bear": False,
        }]
    if mode == "ablation":
        # Each upgrade individually against baseline
        base = {
            "pos_size_pct": 0.10, "max_positions": 5, "hold_days": 10,
            "stop_loss_pct": -0.10, "take_profit_pct": None, "ml_threshold": 0.55,
            "qqq_floor_multiplier": 1.3, "use_gld_rotation": True, "use_vrp": True,
            "use_sector_cap": True,
        }
        configs = []
        configs.append({**base, "name": "baseline"})
        configs.append({**base, "name": "U1_daily_scan", "scan_every_days": 1})
        configs.append({**base, "name": "U2_top_k", "use_top_k": True, "top_k_min_score": 0.50})
        configs.append({**base, "name": "U3_conviction_size", "use_conviction_size": True, "conviction_gain": 1.5})
        configs.append({**base, "name": "U4_tqqq_bull_50", "use_tqqq_bull": True, "tqqq_fraction": 0.5})
        configs.append({**base, "name": "U4_tqqq_bull_100", "use_tqqq_bull": True, "tqqq_fraction": 1.0})
        configs.append({**base, "name": "U5_sqqq_bear_30", "use_sqqq_bear": True, "sqqq_bear_pct": 0.30})
        configs.append({**base, "name": "U5_sqqq_bear_50", "use_sqqq_bear": True, "sqqq_bear_pct": 0.50})
        return configs
    if mode == "stack":
        # Stacked: start with best, add each upgrade cumulatively
        base = {
            "pos_size_pct": 0.10, "max_positions": 5, "hold_days": 10,
            "stop_loss_pct": -0.10, "take_profit_pct": None, "ml_threshold": 0.55,
            "qqq_floor_multiplier": 1.3, "use_gld_rotation": True, "use_vrp": True,
            "use_sector_cap": True,
        }
        configs = []
        configs.append({**base, "name": "S0_baseline"})
        c = {**base, "scan_every_days": 1, "name": "S1_+daily"}
        configs.append(c)
        c = {**c, "use_top_k": True, "top_k_min_score": 0.50, "name": "S2_+topk"}
        configs.append(c)
        c = {**c, "use_conviction_size": True, "conviction_gain": 1.5, "name": "S3_+conviction"}
        configs.append(c)
        c = {**c, "use_tqqq_bull": True, "tqqq_fraction": 0.5, "name": "S4_+tqqq50"}
        configs.append(c)
        c = {**c, "tqqq_fraction": 1.0, "name": "S4b_tqqq100"}
        configs.append(c)
        c = {**c, "use_sqqq_bear": True, "sqqq_bear_pct": 0.30, "name": "S5_+sqqq30"}
        configs.append(c)
        c = {**c, "sqqq_bear_pct": 0.50, "name": "S5b_sqqq50"}
        configs.append(c)
        # Aggressive variant: more positions, bigger size
        c = {**c, "max_positions": 8, "pos_size_pct": 0.08, "name": "S6_more_positions"}
        configs.append(c)
        c = {**c, "max_positions": 10, "pos_size_pct": 0.08, "name": "S7_even_more"}
        configs.append(c)
        return configs
    if mode == "sweep":
        # Focused sweep around the best stacked config
        configs = []
        for scan in [1, 3, 5]:
            for tqqq_frac in [0.0, 0.3, 0.5, 1.0]:
                for sqqq_pct in [0.0, 0.25, 0.50]:
                    for cg in [0.0, 1.5, 3.0]:
                        for maxp in [5, 8]:
                            for size in [0.08, 0.10]:
                                configs.append({
                                    "pos_size_pct": size, "max_positions": maxp, "hold_days": 10,
                                    "stop_loss_pct": -0.10, "take_profit_pct": None, "ml_threshold": 0.55,
                                    "qqq_floor_multiplier": 1.3, "use_gld_rotation": True, "use_vrp": True,
                                    "use_sector_cap": True,
                                    "scan_every_days": scan,
                                    "use_top_k": True, "top_k_min_score": 0.50,
                                    "use_conviction_size": cg > 0, "conviction_gain": cg,
                                    "use_tqqq_bull": tqqq_frac > 0, "tqqq_fraction": tqqq_frac,
                                    "use_sqqq_bear": sqqq_pct > 0, "sqqq_bear_pct": sqqq_pct,
                                    "name": f"sc{scan}_tq{int(tqqq_frac*100)}_sq{int(sqqq_pct*100)}_cg{cg}_mp{maxp}_sz{size}",
                                })
        return configs
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="ablation",
                        choices=["baseline", "ablation", "stack", "sweep"])
    parser.add_argument("--out", default="/home/user/workspace/voltradeai/backtest_v3_results.json")
    args = parser.parse_args()

    _init()
    configs = build_configs(args.mode)
    print(f"[main] running {len(configs)} configs in mode={args.mode}", flush=True)

    t0 = time.time()
    results = []
    for i, cfg in enumerate(configs):
        ti = time.time()
        r = run_full_system(cfg)
        elapsed = time.time() - ti
        name = cfg.get("name", f"cfg{i}")
        cagr = r.get("cagr_pct", "ERR")
        dd = r.get("max_dd_pct", "ERR")
        sortino = r.get("sortino", "ERR")
        trades = r.get("n_trades", "ERR")
        print(f"[{i+1}/{len(configs)}] {name:<30s} CAGR={cagr:>6}%  DD={dd:>5}%  Sortino={sortino:>5}  N={trades:<5} ({elapsed:.1f}s)", flush=True)
        results.append(r)

    total = time.time() - t0
    print(f"\n[done] {len(results)} configs in {total:.1f}s", flush=True)

    # Sort by CAGR and report top 10
    ranked = sorted(results, key=lambda r: r.get("cagr_pct", -999), reverse=True)
    print("\n═══ TOP 10 by CAGR ═══")
    print(f"{'Rank':<5}{'Name':<35}{'CAGR':>8}{'DD':>8}{'Sortino':>9}{'Trades':>8}")
    for i, r in enumerate(ranked[:10]):
        name = r["config"].get("name", "")[:33]
        print(f"{i+1:<5}{name:<35}{r.get('cagr_pct',0):>7}%{r.get('max_dd_pct',0):>7}%{r.get('sortino',0):>9}{r.get('n_trades',0):>8}")

    with open(args.out, "w") as f:
        json.dump({"mode": args.mode, "results": results,
                   "generated_at": datetime.now().isoformat(),
                   "n_configs": len(results)}, f, indent=2)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
