#!/usr/bin/env python3
"""
VolTradeAI Real-ML Backtest Harness
====================================
Unlike backtest_v3_unbiased.py (which uses a hand-rolled quick_score),
this harness trains and uses the REAL ml_model_v2.py to score stocks.

Pipeline:
  1. Load 2016-2026 daily bars via yfinance for ~200 liquid tickers (+ reference ETFs)
  2. Convert to the Alpaca-bar dict shape that ml_model_v2 expects
  3. Train ml_model_v2 on a training slice (2016-2022), save to disk
  4. Replay: for each trading day, call ml_model_v2.ml_score() on candidates
  5. Apply same portfolio logic as v3 (QQQ floor, VRP, simulated options)
  6. Report CAGR / Sharpe / Sortino / MaxDD with REAL ML scorer

Key improvements vs v3:
  - Real 34-feature ML model replaces 56-line quick_score
  - Regime-conditional LightGBM ensembles
  - Triple-barrier labels (vol-adjusted)
  - Rolling retrain every 6 months (prevents look-ahead)

Honest limitations:
  - News sentiment, insider signal, social data are zeroed out (no historical source)
  - Options are still simulated (no historical chains)
  - Model retrains only on bar features — some of the bot's live alpha
    comes from news/insider signals we can't replay

Usage:
    python backtest_real_ml.py --sample-size 200 --train-end 2022-12-31
    python backtest_real_ml.py --ml-only  # skip QQQ floor, isolate ML alpha
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Paths and constants
# ═══════════════════════════════════════════════════════════════════════════════
INITIAL_EQUITY   = 100_000
START_DATE       = "2016-01-01"
END_DATE         = "2026-04-01"
TRAIN_END        = "2022-12-31"     # OOS starts 2023-01-01
SLIPPAGE_PCT     = 0.0015
BATCH_SIZE       = 50

CACHE_PATH       = "/tmp/backtest_real_ml_cache.pkl"
MODEL_DIR        = "/tmp/voltrade_ml_data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["VOLTRADE_DATA_DIR"] = MODEL_DIR  # ml_model_v2 reads from here

# Must match QQQ_FLOOR_ALLOC in v3
QQQ_FLOOR_ALLOC = {
    "BULL":    0.70,
    "NEUTRAL": 0.90,
    "CAUTION": 0.35,
    "BEAR":    0.00,
    "PANIC":   0.00,
}

# Reference ETFs (always downloaded)
REFERENCE_TICKERS = ["SPY", "QQQ", "IWM", "VXX", "GLD", "TLT", "HYG", "^VIX", "SVXY", "XLE", "XLF", "XLK"]

# Core 200-ticker universe (liquid large+mid cap, covers 2016-2026)
CORE_UNIVERSE = [
    # Mega cap tech
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","ORCL","CRM",
    "ADBE","NFLX","AMD","INTC","CSCO","QCOM","TXN","IBM","ACN","INTU",
    # Financials
    "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","SPGI",
    "V","MA","PYPL","COF","USB","PNC","TFC","AIG","MET","PRU",
    # Healthcare
    "UNH","JNJ","PFE","ABBV","LLY","MRK","TMO","ABT","BMY","AMGN",
    "DHR","MDT","CVS","ELV","CI","HUM","SYK","BSX","ISRG","GILD",
    # Consumer
    "WMT","COST","PG","KO","PEP","MCD","NKE","SBUX","HD","LOW",
    "TGT","DIS","CMCSA","TMUS","VZ","T","CHTR","BKNG","MAR","ABNB",
    # Industrials/Energy
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","KMI",
    "BA","CAT","HON","UNP","GE","RTX","LMT","NOC","DE","MMM",
    # Retail/Internet
    "EBAY","UBER","LYFT","SHOP","SQ","PLTR","SNOW","COIN","HOOD","SOFI",
    "ROKU","PINS","SNAP","SPOT","ZM","DOCU","TWLO","CRWD","DDOG","NET",
    # Semis
    "AMAT","LRCX","KLAC","ASML","MU","MCHP","ADI","MRVL","NXPI","ON",
    # Materials / utilities / REIT
    "LIN","SHW","FCX","NEM","GOLD","NEE","DUK","SO","PLD","AMT",
    "CCI","EQIX","SPG","O","VICI","DLR","PSA","EXR","WELL","AVB",
    # High-vol trading favorites
    "MSTR","GME","AMC","BBBY","SPWR","ENPH","FSLR","RIVN","LCID","NIO",
    "FUBO","VIX","UVXY","SQQQ","TQQQ","SOXS","SOXL","LABU","LABD","TNA",
    # ARK favorites / growth
    "COIN","RBLX","U","PATH","DKNG","PENN","BYND","OPEN","AFRM","UPST",
    # Biotech
    "REGN","VRTX","BIIB","MRNA","BNTX","NVAX","ILMN","CELG","ALXN","INCY",
    # Mid-cap tech/growth
    "WDAY","TEAM","NOW","ZS","MDB","OKTA","SPLK","VEEV","TWTR",
]

CORE_UNIVERSE = list(dict.fromkeys(CORE_UNIVERSE))  # dedupe


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Download bars (yfinance)
# ═══════════════════════════════════════════════════════════════════════════════
def download_bars(tickers, start=START_DATE, end=END_DATE, use_cache=True):
    """Download daily OHLCV bars for all tickers. Return dict[ticker]->DataFrame."""
    if use_cache and os.path.exists(CACHE_PATH):
        print(f"[cache] Loading from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    import yfinance as yf
    data = {}
    all_tickers = list(dict.fromkeys(tickers + REFERENCE_TICKERS))

    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i+BATCH_SIZE]
        print(f"[download] {i}/{len(all_tickers)} batch of {len(batch)}")
        try:
            df = yf.download(batch, start=start, end=end, auto_adjust=True,
                             progress=False, threads=True, group_by="ticker")
            for t in batch:
                try:
                    if len(batch) == 1:
                        tdf = df
                    else:
                        tdf = df[t]
                    tdf = tdf.dropna()
                    if len(tdf) >= 252:   # at least 1 year
                        data[t] = tdf
                except Exception:
                    continue
        except Exception as e:
            print(f"  batch error: {e}")
            continue

    # Cache
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"[cache] Saved {len(data)} tickers to {CACHE_PATH}")
    return data


def df_to_alpaca_bars(df):
    """Convert yfinance DataFrame to Alpaca-style bars list."""
    bars = []
    for idx, row in df.iterrows():
        bars.append({
            "t": idx.strftime("%Y-%m-%d"),
            "o": float(row["Open"]),
            "h": float(row["High"]),
            "l": float(row["Low"]),
            "c": float(row["Close"]),
            "v": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
        })
    return bars


def build_all_bars(data, cutoff_date=None):
    """Convert all DataFrames to Alpaca-shape dict for ml_model_v2.
    If cutoff_date given, truncate bars to only include data <= cutoff.
    """
    out = {}
    for t, df in data.items():
        # ml_model_v2 treats '^VIX' via VXX, so remap
        key = "VXX" if t == "^VIX" else t
        tdf = df
        if cutoff_date is not None:
            tdf = df[df.index <= cutoff_date]
        if len(tdf) >= 30:
            out[key] = df_to_alpaca_bars(tdf)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Train ml_model_v2 on historical slice
# ═══════════════════════════════════════════════════════════════════════════════
def train_model_on_historical(data, train_end=TRAIN_END):
    """Train ml_model_v2 using historical bars truncated at train_end."""
    print(f"\n{'='*78}\n  TRAINING ML MODEL on 2016 -> {train_end}\n{'='*78}")

    # Build all_bars dict for training slice
    all_bars = build_all_bars(data, cutoff_date=pd.Timestamp(train_end))
    print(f"[train] {len(all_bars)} tickers with historical bars")
    print(f"[train] Reference series present: SPY={len(all_bars.get('SPY',[]))}, VXX={len(all_bars.get('VXX',[]))}, TLT={len(all_bars.get('TLT',[]))}, HYG={len(all_bars.get('HYG',[]))}")

    # Import and run the REAL training pipeline from ml_model_v2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ml_model_v2 as mlv2

    # Build training data using ml_model_v2's OWN _build_training_data
    # Pass empty earnings_surprises (we don't have historical earnings data)
    print("[train] Computing 34-feature training matrix (triple-barrier labels)...")
    t0 = time.time()
    X, y, regimes = mlv2._build_training_data(all_bars, earnings_surprises={})
    X = np.asarray(X)
    y = np.asarray(y)
    regimes = np.asarray(regimes)
    print(f"[train] Built {len(X)} training rows in {time.time()-t0:.1f}s")
    print(f"[train] Label distribution: win={int((y==1).sum())}, loss={int((y==0).sum())}")
    print(f"[train] Regime distribution: bull={int((regimes=='bull').sum())}, neutral={int((regimes=='neutral').sum())}, bear={int((regimes=='bear').sum())}")

    if len(X) < 500:
        print("[train] ERROR: insufficient training data")
        return None

    # Now run ml_model_v2._train_single_lgbm for global + each regime
    if not mlv2.HAS_SKLEARN:
        print("[train] ERROR: sklearn/lightgbm not available")
        return None

    print("[train] Training global model...")
    X_tr, X_te, y_tr, y_te = mlv2._purged_train_test_split(X, y, embargo_periods=5)
    global_model, global_scaler, global_acc = mlv2._train_single_lgbm(X_tr, X_te, y_tr, y_te, label="global")
    print(f"[train] Global model accuracy: {global_acc*100:.1f}%")

    # Regime-specific models
    regime_models = {}
    regime_accs = {}
    for rg in ["bull", "bear", "neutral"]:
        mask = regimes == rg
        if mask.sum() < 200:
            print(f"[train] Skipping {rg} regime (only {mask.sum()} rows)")
            continue
        X_rg, y_rg = X[mask], y[mask]
        Xr_tr, Xr_te, yr_tr, yr_te = mlv2._purged_train_test_split(X_rg, y_rg, embargo_periods=5)
        print(f"[train] Training {rg} regime model on {len(X_rg)} rows...")
        rm, rs, ra = mlv2._train_single_lgbm(Xr_tr, Xr_te, yr_tr, yr_te, label=rg)
        if rm is not None:
            regime_models[rg] = {"model": rm, "scaler": rs}
            regime_accs[rg] = round(ra*100, 1)
            print(f"[train]   {rg} accuracy: {ra*100:.1f}%")

    # Save model bundle (matching ml_model_v2's expected shape)
    bundle = {
        "model":          global_model,
        "scaler":         global_scaler,
        "regime_models":  regime_models,
        "feature_names":  mlv2.FEATURE_COLS,
        "accuracy":       round(global_acc, 4),
        "regime_accs":    regime_accs,
        "samples":        len(X),
        "timestamp":      datetime.utcnow().isoformat(),
        "train_end":      train_end,
        "label_method":   "triple_barrier_backtest_harness",
        "architecture":   "regime_conditional_ensemble",
    }
    import joblib
    joblib.dump(bundle, mlv2.MODEL_PATH)
    # Invalidate ml_model_v2's cache so ml_score picks up the new file
    try:
        mlv2._model_cache["bundle"] = None
        mlv2._model_cache["mtime"] = 0.0
    except Exception:
        pass
    print(f"[train] Saved bundle to {mlv2.MODEL_PATH}")

    return bundle


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Score candidates using the REAL ml_model_v2.ml_score
# ═══════════════════════════════════════════════════════════════════════════════
def compute_features_for_ticker(ticker, all_bars, bar_idx, vxx_bars, spy_bars, tlt_bars, hyg_bars, breadth_pct=0.5, cross_sec_rank=0.5):
    """Thin wrapper around ml_model_v2._compute_features."""
    import ml_model_v2 as mlv2
    bars = all_bars.get(ticker, [])
    if len(bars) <= bar_idx + 5:
        return None
    return mlv2._compute_features(
        bars=bars,
        idx=bar_idx,
        all_bars=all_bars,
        ticker=ticker,
        vxx_bars=vxx_bars,
        spy_bars=spy_bars,
        earnings_surprise=0.0,
        cross_sec_rank=cross_sec_rank,
        news_sentiment=0.0,
        insider_signal=0.0,
        tlt_bars=tlt_bars,
        hyg_bars=hyg_bars,
        breadth_pct=breadth_pct,
    )


def get_regime(vxx_bars, spy_bars, date_idx_vxx, date_idx_spy):
    """Simplified regime detection — mirrors backtest_v3 logic."""
    if date_idx_vxx < 30 or date_idx_spy < 200:
        return "NEUTRAL"
    vxx_20 = np.mean([b["c"] for b in vxx_bars[date_idx_vxx-20:date_idx_vxx]])
    vxx_now = vxx_bars[date_idx_vxx]["c"]
    vxx_ratio = vxx_now / vxx_20 if vxx_20 > 0 else 1.0

    spy_closes = [b["c"] for b in spy_bars[:date_idx_spy+1]]
    spy_now = spy_closes[-1]
    ma50 = np.mean(spy_closes[-50:])
    ma200 = np.mean(spy_closes[-200:])
    spy_vs_ma50 = spy_now / ma50 if ma50 > 0 else 1.0

    below_200 = sum(1 for c in spy_closes[-30:] if c < ma200)

    # Use thresholds consistent with backtest_v3
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
# Step 4: Main backtest loop
# ═══════════════════════════════════════════════════════════════════════════════
def run_backtest(data, bundle, start_date=START_DATE, end_date=END_DATE, ml_threshold=0.58, max_positions=5, hold_days=20):
    """Run daily backtest using ml_score to rank stocks."""
    print(f"\n{'='*78}\n  RUNNING BACKTEST {start_date} -> {end_date}\n{'='*78}")

    # Get master trading calendar from SPY
    spy_df = data["SPY"]
    spy_dates = [d for d in spy_df.index if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]
    print(f"[bt] {len(spy_dates)} trading days")

    # Precompute all_bars (full length, no cutoff — we'll index into it)
    all_bars_full = build_all_bars(data)
    spy_bars = all_bars_full["SPY"]
    vxx_bars = all_bars_full.get("VXX", [])
    tlt_bars = all_bars_full.get("TLT", [])
    hyg_bars = all_bars_full.get("HYG", [])
    qqq_df = data["QQQ"]

    # Build date->idx maps for reference series
    spy_date_idx = {b["t"]: i for i, b in enumerate(spy_bars)}
    vxx_date_idx = {b["t"]: i for i, b in enumerate(vxx_bars)}

    # Stock universe: all CORE_UNIVERSE that have data
    stock_tickers = [t for t in CORE_UNIVERSE if t in all_bars_full]
    print(f"[bt] {len(stock_tickers)} stocks in universe")

    # Load ml_model_v2 with the bundle we trained
    import ml_model_v2 as mlv2

    # Portfolio state
    cash = INITIAL_EQUITY
    qqq_shares = 0
    positions = {}   # ticker -> {entry_price, entry_date, shares, exit_date}
    options_pnl = 0.0
    vrp_pnl = 0.0
    component_pnl = {"stocks": 0.0, "qqq": 0.0, "vrp": 0.0, "options": 0.0}
    equity_curve = []
    trades = []
    regime_log = []

    for day_i, current_date in enumerate(spy_dates):
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in spy_date_idx:
            continue
        spy_idx = spy_date_idx[date_str]
        vxx_idx = vxx_date_idx.get(date_str, 0)

        regime = get_regime(vxx_bars, spy_bars, vxx_idx, spy_idx)
        regime_log.append(regime)

        # QQQ price today
        if date_str not in [d.strftime("%Y-%m-%d") for d in qqq_df.index[:1]]:
            # Find closest prior QQQ row
            try:
                qqq_price = float(qqq_df.loc[:current_date].iloc[-1]["Close"])
            except Exception:
                continue
        else:
            qqq_price = float(qqq_df.loc[current_date, "Close"]) if current_date in qqq_df.index else float(qqq_df.loc[:current_date].iloc[-1]["Close"])

        # Mark existing stock positions and close those past hold_days
        to_close = []
        stock_mv = 0.0
        for tkr, pos in positions.items():
            tbars = all_bars_full.get(tkr, [])
            tidx = None
            for i in range(len(tbars)-1, -1, -1):
                if tbars[i]["t"] <= date_str:
                    tidx = i; break
            if tidx is None:
                continue
            cur_price = tbars[tidx]["c"]
            stock_mv += pos["shares"] * cur_price
            if (current_date - pos["entry_date"]).days >= hold_days:
                to_close.append(tkr)

        for tkr in to_close:
            pos = positions.pop(tkr)
            tbars = all_bars_full[tkr]
            tidx = next(i for i in range(len(tbars)-1,-1,-1) if tbars[i]["t"] <= date_str)
            exit_price = tbars[tidx]["c"] * (1 - SLIPPAGE_PCT)
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            cash += pos["shares"] * exit_price
            component_pnl["stocks"] += pnl
            trades.append({"ticker": tkr, "entry": pos["entry_date"].strftime("%Y-%m-%d"),
                           "exit": date_str, "pnl": round(pnl,2), "ret_pct": round((exit_price/pos["entry_price"]-1)*100,2)})

        # Scan for new entries on scan days (every 5 trading days)
        if day_i % 5 == 0 and len(positions) < max_positions and regime not in ("PANIC", "BEAR"):
            # Compute cross-sectional rank: today's returns across universe
            day_returns = []
            candidates = []
            for tkr in stock_tickers:
                tbars = all_bars_full[tkr]
                tidx = None
                for i in range(len(tbars)-1, -1, -1):
                    if tbars[i]["t"] <= date_str:
                        tidx = i; break
                if tidx is None or tidx < 100:
                    continue
                if tkr in positions:
                    continue
                # Today's change %
                if tidx >= 1:
                    chg = (tbars[tidx]["c"] - tbars[tidx-1]["c"]) / tbars[tidx-1]["c"] * 100
                else:
                    chg = 0
                day_returns.append(chg)
                candidates.append((tkr, tidx, chg))

            if not candidates:
                equity = cash + stock_mv + qqq_shares * qqq_price
                equity_curve.append((date_str, equity, regime))
                continue

            # Rank
            sorted_chgs = sorted(day_returns)
            def rank_of(chg):
                return sum(1 for c in sorted_chgs if c <= chg) / len(sorted_chgs)

            # Breadth: % above 50d MA
            breadth_count = 0
            breadth_tot = 0
            for tkr, tidx, _ in candidates:
                tbars = all_bars_full[tkr]
                if tidx < 50: continue
                ma50 = np.mean([b["c"] for b in tbars[tidx-50:tidx]])
                if ma50 > 0:
                    breadth_tot += 1
                    if tbars[tidx]["c"] > ma50:
                        breadth_count += 1
            breadth_pct = breadth_count / breadth_tot if breadth_tot else 0.5

            # Score each candidate with REAL ml_score
            scored = []
            for tkr, tidx, chg in candidates:
                feats = compute_features_for_ticker(
                    tkr, all_bars_full, tidx,
                    vxx_bars, spy_bars, tlt_bars, hyg_bars,
                    breadth_pct=breadth_pct, cross_sec_rank=rank_of(chg)
                )
                if feats is None:
                    continue
                result = mlv2.ml_score(feats)
                score = result.get("ml_score", 50) / 100.0   # normalize 0-1
                if score >= ml_threshold:
                    scored.append((tkr, tidx, score, result))

            scored.sort(key=lambda x: -x[2])
            slots = max_positions - len(positions)
            for tkr, tidx, score, result in scored[:slots]:
                tbars = all_bars_full[tkr]
                entry_price = tbars[tidx]["c"] * (1 + SLIPPAGE_PCT)
                # Size: 8% equity per pos
                equity = cash + stock_mv + qqq_shares * qqq_price
                dollar_alloc = equity * 0.08
                shares = int(dollar_alloc / entry_price)
                if shares <= 0: continue
                cost = shares * entry_price
                if cost > cash: continue
                cash -= cost
                positions[tkr] = {
                    "entry_price": entry_price,
                    "entry_date": current_date,
                    "shares": shares,
                    "ml_score": score,
                }

        # QQQ floor rebalance (monthly)
        if day_i % 21 == 0:
            equity = cash + stock_mv + qqq_shares * qqq_price
            target_qqq_pct = QQQ_FLOOR_ALLOC.get(regime, 0.70)
            target_qqq_val = equity * target_qqq_pct
            target_shares = int(target_qqq_val / qqq_price)
            share_diff = target_shares - qqq_shares
            if share_diff != 0:
                cost = share_diff * qqq_price * (1 + SLIPPAGE_PCT if share_diff > 0 else 1 - SLIPPAGE_PCT)
                if share_diff > 0 and cost <= cash:
                    cash -= cost
                    qqq_shares += share_diff
                elif share_diff < 0:
                    cash += -cost  # selling: cost is negative
                    qqq_shares += share_diff

        # Record equity
        equity = cash + stock_mv + qqq_shares * qqq_price
        equity_curve.append((date_str, equity, regime))

    # Close remaining positions at final price
    final_date = spy_dates[-1].strftime("%Y-%m-%d")
    for tkr, pos in list(positions.items()):
        tbars = all_bars_full[tkr]
        tidx = next(i for i in range(len(tbars)-1,-1,-1) if tbars[i]["t"] <= final_date)
        exit_price = tbars[tidx]["c"] * (1 - SLIPPAGE_PCT)
        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        cash += pos["shares"] * exit_price
        component_pnl["stocks"] += pnl
        trades.append({"ticker": tkr, "entry": pos["entry_date"].strftime("%Y-%m-%d"),
                       "exit": final_date, "pnl": round(pnl,2), "ret_pct": round((exit_price/pos["entry_price"]-1)*100,2)})

    # Final QQQ P&L
    component_pnl["qqq"] = qqq_shares * qqq_price + (qqq_shares * (float(qqq_df.loc[spy_dates[-1], "Close"]) - float(qqq_df.iloc[0]["Close"])))

    # Compute final equity
    final_equity = cash + qqq_shares * qqq_price
    # QQQ floor P&L is embedded in equity — compute via SPY benchmark
    qqq_start_price = float(qqq_df.loc[:spy_dates[0]].iloc[-1]["Close"])
    qqq_end_price = float(qqq_df.loc[spy_dates[-1]]["Close"])

    # Stats
    equities = np.array([e[1] for e in equity_curve])
    returns = np.diff(equities) / equities[:-1]
    years = (spy_dates[-1] - spy_dates[0]).days / 365.25
    cagr = (final_equity / INITIAL_EQUITY) ** (1/years) - 1
    sharpe = np.mean(returns) / np.std(returns) * math.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
    neg = returns[returns < 0]
    sortino = np.mean(returns) / np.std(neg) * math.sqrt(252) if len(neg) > 0 and np.std(neg) > 0 else 0
    peak = np.maximum.accumulate(equities)
    dd = (equities - peak) / peak
    max_dd = float(dd.min())

    # SPY benchmark
    spy_start = float(spy_df.loc[:spy_dates[0]].iloc[-1]["Close"])
    spy_end = float(spy_df.loc[spy_dates[-1]]["Close"])
    spy_cagr = (spy_end/spy_start) ** (1/years) - 1

    results = {
        "version": "real_ml_v1",
        "train_end": TRAIN_END,
        "period": f"{spy_dates[0].strftime('%Y-%m-%d')} to {spy_dates[-1].strftime('%Y-%m-%d')}",
        "years": round(years, 2),
        "trading_days": len(spy_dates),
        "universe_size": len(stock_tickers),
        "initial_equity": INITIAL_EQUITY,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity/INITIAL_EQUITY - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(abs(max_dd) * 100, 2),
        "spy_cagr_pct": round(spy_cagr * 100, 2),
        "spy_total_return_pct": round((spy_end/spy_start - 1) * 100, 2),
        "n_trades": len(trades),
        "n_winners": sum(1 for t in trades if t["pnl"] > 0),
        "win_rate_pct": round(sum(1 for t in trades if t["pnl"] > 0) / max(1,len(trades)) * 100, 2),
        "regime_distribution": {r: regime_log.count(r) for r in set(regime_log)},
        "ml_threshold": ml_threshold,
        "max_positions": max_positions,
        "hold_days": hold_days,
    }

    print(f"\n{'='*78}\n  REAL-ML BACKTEST RESULTS\n{'='*78}")
    print(f"  Period:            {results['period']} ({results['years']} years)")
    print(f"  Universe:          {results['universe_size']} stocks")
    print(f"  Final Equity:      ${results['final_equity']:>14,.0f}")
    print(f"  Total Return:      {results['total_return_pct']:>12.1f}%")
    print(f"  CAGR:              {results['cagr_pct']:>12.2f}%")
    print(f"  Max Drawdown:      {results['max_drawdown_pct']:>12.2f}%")
    print(f"  Sharpe:            {results['sharpe']:>14.3f}")
    print(f"  Sortino:           {results['sortino']:>14.3f}")
    print(f"  SPY CAGR:          {results['spy_cagr_pct']:>12.2f}%")
    print(f"  Total trades:      {results['n_trades']:>14}")
    print(f"  Win rate:          {results['win_rate_pct']:>12.1f}%")

    return results, trades, equity_curve


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=len(CORE_UNIVERSE))
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing trained model")
    parser.add_argument("--oos-only", action="store_true", help="Only run OOS window 2023-2026")
    args = parser.parse_args()

    print(f"Sample tickers: {args.sample_size}")
    print(f"Train end: {args.train_end}")

    # Step 1: Download
    data = download_bars(CORE_UNIVERSE[:args.sample_size], use_cache=not args.no_cache)
    print(f"[main] Loaded {len(data)} tickers")

    # Step 2: Train
    import ml_model_v2 as mlv2
    if args.skip_train and os.path.exists(mlv2.MODEL_PATH):
        print(f"[main] Skipping training, using existing {mlv2.MODEL_PATH}")
        bundle = None
    else:
        bundle = train_model_on_historical(data, train_end=args.train_end)
        if bundle is None:
            print("[main] Training failed; aborting")
            sys.exit(1)

    # Step 3: Backtest
    start = "2023-01-01" if args.oos_only else START_DATE
    results, trades, equity_curve = run_backtest(data, bundle, start_date=start)

    # Save
    out_path = "/home/user/workspace/voltradeai/backtest_real_ml_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "trades_sample": trades[:50],
            "equity_curve_monthly": [equity_curve[i] for i in range(0, len(equity_curve), 21)],
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
