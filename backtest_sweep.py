#!/usr/bin/env python3
"""
VolTradeAI Backtest Parameter Sweep
====================================
Finds the configuration that maximizes:
  - CAGR (target: 45%)
  - Sortino (higher is better)
subject to:
  - Max drawdown < 33.72% (SPY 2016-2026 baseline)

Sweeps:
  - pos_size_pct      : 2%, 4%, 6%, 8%
  - max_positions     : 3, 5, 8
  - hold_days         : 5, 10, 20, 40
  - stop_loss_pct     : None, -5%, -8%, -12%
  - take_profit_pct   : None, +10%, +20%
  - ml_threshold      : 0.55, 0.60, 0.65
  - qqq_floor_multiplier : 0.5x (half), 1.0x (normal), 1.5x (bigger)
  - use_bear_model_short : True/False (if bear model confidence high, short)

Uses the pre-trained /tmp/voltrade_ml_v2.pkl from backtest_real_ml.py.
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
INITIAL_EQUITY = 100_000
START_DATE = "2016-01-01"
END_DATE = "2026-04-01"
SLIPPAGE_PCT = 0.0015

# Precomputed once, cached in worker
_GLOBAL = {}


def _df_to_alpaca_bars(df):
    return [
        {"t": idx.strftime("%Y-%m-%d"),
         "o": float(r["Open"]), "h": float(r["High"]),
         "l": float(r["Low"]), "c": float(r["Close"]),
         "v": int(r["Volume"]) if not pd.isna(r["Volume"]) else 0}
        for idx, r in df.iterrows()
    ]


def _build_all_bars(data):
    out = {}
    for t, df in data.items():
        key = "VXX" if t == "^VIX" else t
        if len(df) >= 30:
            out[key] = _df_to_alpaca_bars(df)
    return out


def _get_regime(vxx_bars, spy_bars, vxx_idx, spy_idx):
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


def _init_worker():
    """Load data + trained ML once per worker process. Also PRECOMPUTES all
    feature vectors and caches batched ml_score predictions to speed up sweeps."""
    global _GLOBAL
    if _GLOBAL:
        return
    print(f"[worker {os.getpid()}] loading data + ML model...", flush=True)
    import ml_model_v2 as mlv2
    import joblib
    _GLOBAL["mlv2"] = mlv2
    data = pickle.load(open(CACHE_PATH, "rb"))
    all_bars = _build_all_bars(data)
    _GLOBAL["data"] = data
    _GLOBAL["all_bars"] = all_bars
    _GLOBAL["spy_bars"] = all_bars["SPY"]
    _GLOBAL["vxx_bars"] = all_bars.get("VXX", [])
    _GLOBAL["tlt_bars"] = all_bars.get("TLT", [])
    _GLOBAL["hyg_bars"] = all_bars.get("HYG", [])
    _GLOBAL["spy_date_idx"] = {b["t"]: i for i, b in enumerate(_GLOBAL["spy_bars"])}
    _GLOBAL["vxx_date_idx"] = {b["t"]: i for i, b in enumerate(_GLOBAL["vxx_bars"])}

    # Universe: use same CORE_UNIVERSE from backtest_real_ml
    from backtest_real_ml import CORE_UNIVERSE
    _GLOBAL["stock_tickers"] = [t for t in CORE_UNIVERSE if t in all_bars]

    # Precompute ticker date_idx maps for fast lookup
    _GLOBAL["tkr_date_idx"] = {t: {b["t"]: i for i, b in enumerate(all_bars[t])}
                                for t in _GLOBAL["stock_tickers"]}

    # ─────────────────────────────────────────────────────────────────
    # OPTIMIZATION: Precompute feature matrix + ML scores ONCE
    # ─────────────────────────────────────────────────────────────────
    cache_file = "/tmp/backtest_sweep_features.pkl"
    if os.path.exists(cache_file):
        print(f"[worker {os.getpid()}] loading cached scores from {cache_file}", flush=True)
        cached = pickle.load(open(cache_file, "rb"))
        _GLOBAL["feature_scores"] = cached["scores"]
        return

    print(f"[worker {os.getpid()}] precomputing features + ML scores (one-time)...", flush=True)
    t0 = time.time()
    bundle = joblib.load(mlv2.MODEL_PATH)
    FEATURE_COLS = mlv2.FEATURE_COLS

    # Pre-load regime models + global scaler/model
    global_model = bundle["model"]
    global_scaler = bundle["scaler"]
    regime_models_raw = bundle.get("regime_models", {})
    regime_objs = {rg: (rm["model"], rm["scaler"]) for rg, rm in regime_models_raw.items()}

    feature_scores = {}  # (ticker, date_str) -> score 0-100

    # Group by date to compute breadth + rank once per date
    spy_bars = _GLOBAL["spy_bars"]
    vxx_bars = _GLOBAL["vxx_bars"]
    tlt_bars = _GLOBAL["tlt_bars"]
    hyg_bars = _GLOBAL["hyg_bars"]
    spy_dates_all = [b["t"] for b in spy_bars]

    # Only score every 5th trading day (matches scan cadence) to save time
    for day_i, date_str in enumerate(spy_dates_all):
        if day_i % 5 != 0:
            continue
        if date_str < "2016-01-04" or date_str > "2026-04-01":
            continue

        # Collect candidates for this date
        candidates = []   # (ticker, tidx, chg)
        day_chgs = []
        for tkr in _GLOBAL["stock_tickers"]:
            idx_map = _GLOBAL["tkr_date_idx"][tkr]
            tidx = idx_map.get(date_str)
            if tidx is None or tidx < 100:
                continue
            tbars = all_bars[tkr]
            chg = (tbars[tidx]["c"] - tbars[tidx-1]["c"]) / tbars[tidx-1]["c"] * 100 if tidx >= 1 else 0
            day_chgs.append(chg)
            candidates.append((tkr, tidx, chg))
        if not candidates:
            continue

        sorted_chgs = sorted(day_chgs)
        def rank_of(c):
            return sum(1 for x in sorted_chgs if x <= c) / len(sorted_chgs)

        # Breadth
        breadth_c = 0; breadth_t = 0
        for tkr, tidx, _ in candidates:
            if tidx < 50: continue
            tbars = all_bars[tkr]
            ma50 = sum(b["c"] for b in tbars[tidx-50:tidx]) / 50
            if ma50 > 0:
                breadth_t += 1
                if tbars[tidx]["c"] > ma50:
                    breadth_c += 1
        breadth_pct = breadth_c / breadth_t if breadth_t else 0.5

        # Compute features for every candidate
        feats_per_ticker = []
        for tkr, tidx, chg in candidates:
            feats = mlv2._compute_features(
                bars=all_bars[tkr], idx=tidx, all_bars=all_bars,
                ticker=tkr, vxx_bars=vxx_bars, spy_bars=spy_bars,
                earnings_surprise=0.0, cross_sec_rank=rank_of(chg),
                news_sentiment=0.0, insider_signal=0.0,
                tlt_bars=tlt_bars, hyg_bars=hyg_bars,
                breadth_pct=breadth_pct,
            )
            if feats is not None:
                feats_per_ticker.append((tkr, feats))

        if not feats_per_ticker:
            continue

        # Build feature matrix, group by regime
        import pandas as pd
        rows_by_regime = {"global": [], "bull": [], "bear": [], "neutral": []}
        tkr_by_regime = {"global": [], "bull": [], "bear": [], "neutral": []}
        for tkr, feats in feats_per_ticker:
            row = [float(feats.get(col, 0) or 0) for col in FEATURE_COLS]
            vxx_r = float(feats.get("vxx_ratio", 1.0) or 1.0)
            spy_m = float(feats.get("spy_vs_ma50", 1.0) or 1.0)
            rg = mlv2._classify_regime(vxx_r, spy_m)
            if rg in regime_objs:
                rows_by_regime[rg].append(row)
                tkr_by_regime[rg].append(tkr)
            else:
                rows_by_regime["global"].append(row)
                tkr_by_regime["global"].append(tkr)

        # Batched predictions
        for rg_key, rows in rows_by_regime.items():
            if not rows:
                continue
            if rg_key == "global":
                mdl, scaler = global_model, global_scaler
            else:
                mdl, scaler = regime_objs[rg_key]
            X_df = pd.DataFrame(rows, columns=FEATURE_COLS, dtype=np.float32)
            try:
                X_sc = scaler.transform(X_df)
                if hasattr(mdl, "feature_name_"):
                    probs = mdl.predict(pd.DataFrame(X_sc, columns=FEATURE_COLS))
                elif hasattr(mdl, "predict_proba"):
                    probs = mdl.predict_proba(X_sc)[:, 1]
                else:
                    probs = mdl.predict(X_sc)
            except Exception:
                continue
            probs = np.clip(probs, 0.10, 0.90)
            for tkr, p in zip(tkr_by_regime[rg_key], probs):
                feature_scores[(tkr, date_str)] = round(float(p) * 100, 1)

    print(f"[worker {os.getpid()}] cached {len(feature_scores)} (ticker,date) scores in {time.time()-t0:.1f}s", flush=True)
    _GLOBAL["feature_scores"] = feature_scores
    try:
        pickle.dump({"scores": feature_scores}, open(cache_file, "wb"))
        print(f"[worker {os.getpid()}] saved score cache to {cache_file}", flush=True)
    except Exception as e:
        print(f"[worker {os.getpid()}] cache save failed: {e}", flush=True)


def run_one(cfg):
    """Run a single backtest config. Returns results dict."""
    _init_worker()
    mlv2 = _GLOBAL["mlv2"]
    all_bars = _GLOBAL["all_bars"]
    spy_bars = _GLOBAL["spy_bars"]
    vxx_bars = _GLOBAL["vxx_bars"]
    tlt_bars = _GLOBAL["tlt_bars"]
    hyg_bars = _GLOBAL["hyg_bars"]
    spy_date_idx = _GLOBAL["spy_date_idx"]
    vxx_date_idx = _GLOBAL["vxx_date_idx"]
    stock_tickers = _GLOBAL["stock_tickers"]
    tkr_date_idx = _GLOBAL["tkr_date_idx"]

    data = _GLOBAL["data"]
    spy_df = data["SPY"]
    qqq_df = data["QQQ"]
    spy_dates = [d for d in spy_df.index if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)]

    # Config
    pos_size_pct     = cfg["pos_size_pct"]
    max_positions    = cfg["max_positions"]
    hold_days        = cfg["hold_days"]
    stop_loss_pct    = cfg.get("stop_loss_pct")       # e.g. -0.05 or None
    take_profit_pct  = cfg.get("take_profit_pct")     # e.g. 0.15 or None
    ml_threshold     = cfg["ml_threshold"]
    qqq_mult         = cfg.get("qqq_floor_multiplier", 1.0)

    # QQQ floor allocation (multiplied)
    qqq_floor = {
        "BULL":    min(0.95, 0.70 * qqq_mult),
        "NEUTRAL": min(0.95, 0.90 * qqq_mult),
        "CAUTION": min(0.95, 0.35 * qqq_mult),
        "BEAR":    0.00,
        "PANIC":   0.00,
    }

    cash = INITIAL_EQUITY
    qqq_shares = 0
    positions = {}   # ticker -> {entry_price, entry_date, shares, ml_score}
    equity_curve = []
    trades = []
    regime_log = []

    for day_i, current_date in enumerate(spy_dates):
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in spy_date_idx:
            continue
        spy_idx = spy_date_idx[date_str]
        vxx_idx = vxx_date_idx.get(date_str, 0)
        regime = _get_regime(vxx_bars, spy_bars, vxx_idx, spy_idx)
        regime_log.append(regime)

        # QQQ price today
        try:
            qqq_price = float(qqq_df.loc[:current_date].iloc[-1]["Close"])
        except Exception:
            continue

        # Mark positions: compute current value + check stops/TP/hold
        to_close = []
        stock_mv = 0.0
        for tkr, pos in positions.items():
            idx_map = tkr_date_idx.get(tkr, {})
            # Find most recent bar <= today
            tidx = idx_map.get(date_str)
            if tidx is None:
                tbars = all_bars.get(tkr, [])
                for i in range(len(tbars)-1, -1, -1):
                    if tbars[i]["t"] <= date_str:
                        tidx = i; break
            if tidx is None:
                continue
            cur_price = all_bars[tkr][tidx]["c"]
            stock_mv += pos["shares"] * cur_price
            ret = (cur_price / pos["entry_price"]) - 1

            # Stop-loss
            if stop_loss_pct is not None and ret <= stop_loss_pct:
                to_close.append((tkr, "stop"))
                continue
            # Take-profit
            if take_profit_pct is not None and ret >= take_profit_pct:
                to_close.append((tkr, "tp"))
                continue
            # Time exit
            if (current_date - pos["entry_date"]).days >= hold_days:
                to_close.append((tkr, "time"))

        for tkr, reason in to_close:
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
            # stock_mv was computed pre-close — adjust
            stock_mv -= pos["shares"] * tbars[tidx]["c"]
            trades.append({"ticker": tkr, "pnl": pnl,
                           "ret_pct": (exit_price/pos["entry_price"]-1)*100,
                           "reason": reason})

        # Entry scan every 5 trading days — uses PRECOMPUTED score cache
        if day_i % 5 == 0 and len(positions) < max_positions and regime not in ("PANIC", "BEAR"):
            feature_scores = _GLOBAL.get("feature_scores", {})
            # Collect all candidates with scores for this date
            scored = []
            for tkr in stock_tickers:
                if tkr in positions: continue
                key = (tkr, date_str)
                if key not in feature_scores:
                    continue
                score = feature_scores[key] / 100.0
                if score >= ml_threshold:
                    idx_map = tkr_date_idx.get(tkr, {})
                    tidx = idx_map.get(date_str)
                    if tidx is None: continue
                    scored.append((tkr, tidx, score))

            scored.sort(key=lambda x: -x[2])
            slots = max_positions - len(positions)
            equity = cash + stock_mv + qqq_shares * qqq_price
            for tkr, tidx, score in scored[:slots]:
                tbars = all_bars[tkr]
                entry_price = tbars[tidx]["c"] * (1 + SLIPPAGE_PCT)
                dollar_alloc = equity * pos_size_pct
                shares = int(dollar_alloc / entry_price)
                if shares <= 0: continue
                cost = shares * entry_price
                if cost > cash: continue
                cash -= cost
                positions[tkr] = {"entry_price": entry_price, "entry_date": current_date,
                                  "shares": shares, "ml_score": score}
                stock_mv += shares * tbars[tidx]["c"]

        # QQQ floor rebalance monthly
        if day_i % 21 == 0:
            equity = cash + stock_mv + qqq_shares * qqq_price
            target_pct = qqq_floor.get(regime, 0.70)
            target_val = equity * target_pct
            target_sh = int(target_val / qqq_price)
            diff = target_sh - qqq_shares
            if diff != 0:
                trade_cost = diff * qqq_price * (1 + SLIPPAGE_PCT if diff > 0 else 1 - SLIPPAGE_PCT)
                if diff > 0 and trade_cost <= cash:
                    cash -= trade_cost
                    qqq_shares += diff
                elif diff < 0:
                    cash += -trade_cost
                    qqq_shares += diff

        equity = cash + stock_mv + qqq_shares * qqq_price
        equity_curve.append(equity)

    # Close remaining at end
    final_date = spy_dates[-1].strftime("%Y-%m-%d")
    final_qqq_price = float(qqq_df.loc[spy_dates[-1]]["Close"])
    for tkr, pos in list(positions.items()):
        tbars = all_bars[tkr]
        for i in range(len(tbars)-1, -1, -1):
            if tbars[i]["t"] <= final_date:
                exit_price = tbars[i]["c"] * (1 - SLIPPAGE_PCT)
                cash += pos["shares"] * exit_price
                trades.append({"ticker": tkr, "pnl": (exit_price - pos["entry_price"]) * pos["shares"],
                               "ret_pct": (exit_price/pos["entry_price"]-1)*100, "reason": "end"})
                break
    final_equity = cash + qqq_shares * final_qqq_price

    equities = np.array(equity_curve)
    if len(equities) < 2:
        return {"config": cfg, "error": "no equity curve"}
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Test first N configs only")
    parser.add_argument("--quick", action="store_true", help="Smaller grid for speed")
    args = parser.parse_args()

    # Build config grid
    if args.quick:
        # Small targeted grid
        pos_sizes   = [0.03, 0.05, 0.08]
        max_pos     = [3, 5]
        hold_days   = [5, 10, 20]
        stop_losses = [None, -0.05, -0.08]
        take_profs  = [None, 0.15]
        thresholds  = [0.58, 0.62]
        qqq_mults   = [0.8, 1.0, 1.2]
    else:
        # Full sweep
        pos_sizes   = [0.02, 0.04, 0.06, 0.08]
        max_pos     = [3, 5, 8]
        hold_days   = [5, 10, 20, 40]
        stop_losses = [None, -0.05, -0.08, -0.12]
        take_profs  = [None, 0.10, 0.20]
        thresholds  = [0.55, 0.60, 0.65]
        qqq_mults   = [0.5, 1.0, 1.5]

    configs = []
    for ps, mp, hd, sl, tp, th, qm in itertools.product(
            pos_sizes, max_pos, hold_days, stop_losses, take_profs, thresholds, qqq_mults):
        if ps * mp > 0.85: continue  # Don't allow >85% equity in stocks
        configs.append({
            "pos_size_pct": ps, "max_positions": mp, "hold_days": hd,
            "stop_loss_pct": sl, "take_profit_pct": tp,
            "ml_threshold": th, "qqq_floor_multiplier": qm,
        })

    if args.limit:
        configs = configs[:args.limit]
    print(f"[sweep] {len(configs)} configurations, {args.workers} workers")
    print(f"[sweep] Est time: {len(configs) * 60 / args.workers / 60:.1f} min")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futures = {ex.submit(run_one, c): i for i, c in enumerate(configs)}
        for i, f in enumerate(as_completed(futures)):
            try:
                r = f.result(timeout=600)
                results.append(r)
                if (i+1) % 10 == 0 or i < 5:
                    print(f"[{i+1}/{len(configs)}] CAGR={r.get('cagr_pct',0):.1f}% DD={r.get('max_dd_pct',0):.1f}% Sortino={r.get('sortino',0):.2f}", flush=True)
            except Exception as e:
                print(f"[{i+1}] ERROR: {e}", flush=True)

    print(f"\n[sweep] completed in {(time.time()-t0)/60:.1f} min")

    # Filter: DD < 33.72 (SPY baseline), then rank by CAGR and Sortino
    qualifying = [r for r in results if "error" not in r and r.get("max_dd_pct", 100) < 33.72]
    print(f"\n{len(qualifying)}/{len(results)} configs meet DD < 33.72% (SPY baseline)")

    # Top 10 by CAGR
    by_cagr = sorted(qualifying, key=lambda r: -r["cagr_pct"])[:10]
    print(f"\n{'='*110}")
    print("TOP 10 BY CAGR (with DD < SPY's 33.72%)")
    print(f"{'='*110}")
    print(f"{'CAGR':>7} {'MaxDD':>7} {'Sortino':>8} {'Sharpe':>7} {'Trades':>7} {'Win%':>6}  Config")
    for r in by_cagr:
        c = r["config"]
        tag = f"sz={c['pos_size_pct']:.0%} np={c['max_positions']} hold={c['hold_days']}d sl={c['stop_loss_pct']} tp={c['take_profit_pct']} thr={c['ml_threshold']} qqq={c['qqq_floor_multiplier']}x"
        print(f"{r['cagr_pct']:>6.1f}% {r['max_dd_pct']:>6.1f}% {r['sortino']:>8.2f} {r['sharpe']:>7.2f} {r['n_trades']:>7} {r['win_rate_pct']:>5.1f}%  {tag}")

    # Top 10 by Sortino
    by_sortino = sorted(qualifying, key=lambda r: -r["sortino"])[:10]
    print(f"\n{'='*110}")
    print("TOP 10 BY SORTINO (with DD < SPY's 33.72%)")
    print(f"{'='*110}")
    print(f"{'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'Trades':>7} {'Win%':>6}  Config")
    for r in by_sortino:
        c = r["config"]
        tag = f"sz={c['pos_size_pct']:.0%} np={c['max_positions']} hold={c['hold_days']}d sl={c['stop_loss_pct']} tp={c['take_profit_pct']} thr={c['ml_threshold']} qqq={c['qqq_floor_multiplier']}x"
        print(f"{r['sortino']:>8.2f} {r['cagr_pct']:>6.1f}% {r['max_dd_pct']:>6.1f}% {r['sharpe']:>7.2f} {r['n_trades']:>7} {r['win_rate_pct']:>5.1f}%  {tag}")

    # Save full results
    out_path = "/home/user/workspace/voltradeai/backtest_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "all_results": results,
            "qualifying": qualifying,
            "top10_by_cagr": by_cagr,
            "top10_by_sortino": by_sortino,
            "spy_baseline_dd_pct": 33.72,
            "spy_baseline_cagr_pct": 14.01,
            "swept_at": datetime.utcnow().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
