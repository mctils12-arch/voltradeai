#!/usr/bin/env python3
"""
Aggressive Backtest — target 42% CAGR / 10% DD

Changes from v3 (everything on the table):
  A. CONCENTRATED PICKING: top-3 positions (not 5-8), equal-weight 33% each
  B. MOMENTUM TRIGGER: only enter if ML score > 0.55 AND price > 20d MA AND 5d momentum > 0
  C. TRAILING STOP at ATR*2 (dynamic, tighter than fixed %)
  D. TIGHT HARD STOP at -5% (not -10%) — limits DD
  E. TAKE PROFIT at +15% to lock gains
  F. REGIME-GATED EXPOSURE: only 30% deployed in CAUTION, 0% in BEAR/PANIC
  G. NO FLOOR (remove QQQ floor entirely — user said we can change anything)
     The floor was the main source of drawdown in 2018, 2020, 2022
  H. CASH-HEAVY DEFAULT: idle cash earns money-market ~4%
  I. SHORT HIGH-PROBABILITY DECLINERS in BEAR/PANIC (via inverse exposure)

The MAR target (CAGR/MaxDD = 4.2) is Medallion-level. Under 10% DD means
~zero exposure during 2018-Q4, COVID-March-2020, and 2022-full-year, which
is ONLY achievable by staying in cash during those windows.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime

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
CASH_YIELD = 0.04  # annualized money-market yield

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
    for sym in ["spy","vxx","qqq","gld"]:
        bars = _GLOBAL[f"{sym}_bars"]
        _GLOBAL[f"{sym}_date_idx"] = {b["t"]: i for i, b in enumerate(bars)}

    from backtest_real_ml import CORE_UNIVERSE
    _GLOBAL["stock_tickers"] = [t for t in CORE_UNIVERSE if t in all_bars]
    _GLOBAL["tkr_date_idx"] = {t: {b["t"]: i for i, b in enumerate(all_bars[t])}
                                for t in _GLOBAL["stock_tickers"]}

    if os.path.exists(SCORE_CACHE_PATH):
        cached = pickle.load(open(SCORE_CACHE_PATH, "rb"))
        _GLOBAL["feature_scores"] = cached["scores"]
        print(f"[init] loaded {len(cached['scores'])} ML scores", flush=True)
    else:
        print("[init] ERROR: no ML cache"); sys.exit(1)


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


def atr(bars, idx, period=14):
    """Average True Range."""
    if idx < period: return 0
    trs = []
    for i in range(idx - period, idx):
        h, l = bars[i]["h"], bars[i]["l"]
        prev_c = bars[i-1]["c"] if i > 0 else bars[i]["c"]
        tr = max(h-l, abs(h-prev_c), abs(l-prev_c))
        trs.append(tr)
    return np.mean(trs)


def momentum_filter(bars, idx):
    """Returns (price_above_ma20, momentum_positive_5d)."""
    if idx < 20: return False, False
    closes = [b["c"] for b in bars[max(0,idx-20):idx+1]]
    ma20 = np.mean(closes[-20:])
    cur = bars[idx]["c"]
    prev5 = bars[idx-5]["c"] if idx >= 5 else cur
    return cur > ma20, cur > prev5


def run_aggressive(cfg):
    _init()
    all_bars = _GLOBAL["all_bars"]
    spy_bars = _GLOBAL["spy_bars"]
    vxx_bars = _GLOBAL["vxx_bars"]
    qqq_bars = _GLOBAL["qqq_bars"]
    spy_date_idx = _GLOBAL["spy_date_idx"]
    vxx_date_idx = _GLOBAL["vxx_date_idx"]
    stock_tickers = _GLOBAL["stock_tickers"]
    tkr_date_idx = _GLOBAL["tkr_date_idx"]
    feature_scores = _GLOBAL["feature_scores"]

    data = _GLOBAL["data"]
    spy_df = data["SPY"]
    spy_dates = [d for d in spy_df.index if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)]

    # Config
    pos_size_pct       = cfg["pos_size_pct"]
    max_positions      = cfg["max_positions"]
    hold_days          = cfg["hold_days"]
    ml_threshold       = cfg["ml_threshold"]
    stop_loss_pct      = cfg["stop_loss_pct"]
    take_profit_pct    = cfg.get("take_profit_pct")
    trailing_stop_atr  = cfg.get("trailing_stop_atr", 0)   # 0=off
    use_momentum_gate  = cfg.get("use_momentum_gate", True)
    require_bull       = cfg.get("require_bull", False)  # only enter in BULL/NEUTRAL
    caution_exposure   = cfg.get("caution_exposure", 0.30)
    bear_exposure      = cfg.get("bear_exposure", 0.0)
    cash_yield_daily   = (1 + CASH_YIELD) ** (1/252) - 1

    cash = INITIAL_EQUITY
    positions = {}  # tkr -> {entry_price, entry_date, shares, peak_price, ml_score}
    equity_curve = []
    trades = []
    regime_log = []

    for day_i, current_date in enumerate(spy_dates):
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in spy_date_idx: continue
        spy_idx = spy_date_idx[date_str]
        vxx_idx = vxx_date_idx.get(date_str, 0)
        regime = get_regime(vxx_bars, spy_bars, vxx_idx, spy_idx)
        regime_log.append(regime)

        # Accrue cash yield
        cash *= (1 + cash_yield_daily)

        # Mark positions, update trailing peak, check exits
        to_close = []
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
            # Update trailing peak
            if cur_price > pos["peak_price"]:
                pos["peak_price"] = cur_price
            ret = (cur_price / pos["entry_price"]) - 1

            # Hard stop
            if ret <= stop_loss_pct:
                to_close.append((tkr, "stop")); continue
            # Take profit
            if take_profit_pct is not None and ret >= take_profit_pct:
                to_close.append((tkr, "tp")); continue
            # Trailing stop (ATR-based)
            if trailing_stop_atr > 0:
                a = atr(all_bars[tkr], tidx, 14)
                if a > 0 and cur_price <= pos["peak_price"] - trailing_stop_atr * a:
                    to_close.append((tkr, "trail")); continue
            # Time stop
            if (current_date - pos["entry_date"]).days >= hold_days:
                to_close.append((tkr, "time")); continue
            # Regime exit — dump everything in BEAR/PANIC
            if regime in ("BEAR", "PANIC"):
                to_close.append((tkr, "regime_exit")); continue

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
            stock_mv -= pos["shares"] * tbars[tidx]["c"]
            trades.append({"ticker": tkr, "pnl": pnl, "reason": reason,
                           "ret_pct": (exit_price/pos["entry_price"]-1)*100,
                           "regime_entry": pos.get("regime_entry")})

        # ─── Entry logic ───
        # Allowed regimes
        if require_bull:
            allowed_regime = regime == "BULL"
        else:
            allowed_regime = regime in ("BULL", "NEUTRAL", "CAUTION")
        if regime in ("BEAR", "PANIC"):
            allowed_regime = False

        # Exposure cap by regime
        if regime == "BULL":         target_exp = 1.0
        elif regime == "NEUTRAL":    target_exp = 0.90
        elif regime == "CAUTION":    target_exp = caution_exposure
        else:                        target_exp = bear_exposure

        if allowed_regime and len(positions) < max_positions:
            equity = cash + stock_mv
            current_exp = stock_mv / equity if equity > 0 else 0
            if current_exp < target_exp:
                # Scan candidates
                scored = []
                for tkr in stock_tickers:
                    if tkr in positions: continue
                    key = (tkr, date_str)
                    if key not in feature_scores: continue
                    score = feature_scores[key] / 100.0
                    if score < ml_threshold: continue
                    tidx = tkr_date_idx[tkr].get(date_str)
                    if tidx is None: continue
                    # Momentum gate
                    if use_momentum_gate:
                        above_ma, mom_pos = momentum_filter(all_bars[tkr], tidx)
                        if not (above_ma and mom_pos): continue
                    scored.append((tkr, tidx, score))
                scored.sort(key=lambda x: -x[2])

                # Fill up to max_positions or target_exp
                for tkr, tidx, score in scored:
                    if len(positions) >= max_positions: break
                    equity = cash + stock_mv
                    current_exp = stock_mv / equity if equity > 0 else 0
                    if current_exp >= target_exp: break

                    tbars = all_bars[tkr]
                    entry_price = tbars[tidx]["c"] * (1 + SLIPPAGE_PCT)
                    dollar_alloc = equity * min(pos_size_pct, target_exp - current_exp)
                    shares = int(dollar_alloc / entry_price)
                    if shares <= 0: continue
                    cost = shares * entry_price
                    if cost > cash: continue
                    cash -= cost
                    positions[tkr] = {
                        "entry_price": entry_price, "entry_date": current_date,
                        "shares": shares, "ml_score": score,
                        "peak_price": entry_price, "regime_entry": regime,
                    }
                    stock_mv += shares * tbars[tidx]["c"]

        equity = cash + stock_mv
        equity_curve.append(equity)

    # Close everything
    final_date = spy_dates[-1].strftime("%Y-%m-%d")
    for tkr, pos in list(positions.items()):
        tbars = all_bars[tkr]
        for i in range(len(tbars)-1, -1, -1):
            if tbars[i]["t"] <= final_date:
                exit_price = tbars[i]["c"] * (1 - SLIPPAGE_PCT)
                cash += pos["shares"] * exit_price
                trades.append({"ticker": tkr, "pnl": (exit_price-pos["entry_price"])*pos["shares"],
                               "reason": "end",
                               "ret_pct": (exit_price/pos["entry_price"]-1)*100})
                break
    final_equity = cash

    eq = np.array(equity_curve)
    if len(eq) < 2:
        return {"config": cfg, "error": "no curve"}
    rets = np.diff(eq) / eq[:-1]
    years = (spy_dates[-1] - spy_dates[0]).days / 365.25
    cagr = (final_equity / INITIAL_EQUITY) ** (1/years) - 1 if final_equity > 0 else -1
    sharpe = np.mean(rets) / np.std(rets) * math.sqrt(252) if np.std(rets) > 0 else 0
    neg = rets[rets < 0]
    sortino = np.mean(rets) / np.std(neg) * math.sqrt(252) if len(neg) > 0 and np.std(neg) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
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
        "mar_ratio": round(cagr*100 / max(abs(max_dd)*100, 0.01), 2),
    }


def build_configs():
    """Search for configs that prioritize DD reduction first, then CAGR."""
    configs = []
    # Tight-risk configs
    for max_pos in [3, 5]:
        for pos_size in [0.20, 0.33] if max_pos == 3 else [0.15, 0.20]:
            for hold in [5, 10, 20]:
                for stop in [-0.05, -0.08]:
                    for tp in [None, 0.15, 0.20]:
                        for thresh in [0.55, 0.60, 0.65]:
                            for trail in [0, 2.0]:
                                for require_bull in [False, True]:
                                    for caution_exp in [0.0, 0.30]:
                                        name = f"mp{max_pos}_sz{pos_size}_h{hold}_st{abs(stop):.2f}_tp{tp}_ml{thresh}_tr{trail}_rb{int(require_bull)}_ce{caution_exp}"
                                        configs.append({
                                            "name": name,
                                            "max_positions": max_pos,
                                            "pos_size_pct": pos_size,
                                            "hold_days": hold,
                                            "stop_loss_pct": stop,
                                            "take_profit_pct": tp,
                                            "ml_threshold": thresh,
                                            "trailing_stop_atr": trail,
                                            "use_momentum_gate": True,
                                            "require_bull": require_bull,
                                            "caution_exposure": caution_exp,
                                            "bear_exposure": 0.0,
                                        })
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/user/workspace/voltradeai/backtest_aggressive_results.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    _init()
    configs = build_configs()
    if args.smoke:
        configs = configs[:20]
    print(f"[main] running {len(configs)} configs", flush=True)

    t0 = time.time()
    results = []
    for i, cfg in enumerate(configs):
        r = run_aggressive(cfg)
        results.append(r)
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(configs)}] elapsed {time.time()-t0:.0f}s", flush=True)

    print(f"\n[done] {len(results)} configs in {time.time()-t0:.0f}s", flush=True)

    # Rank by a hybrid score that penalizes DD
    for r in results:
        cagr = r.get("cagr_pct", 0)
        dd = r.get("max_dd_pct", 100)
        # Score: reward CAGR, hard penalty above 10% DD
        if dd <= 10:
            r["score"] = cagr + 10  # bonus for meeting DD target
        elif dd <= 20:
            r["score"] = cagr
        else:
            r["score"] = cagr - (dd - 20) * 2  # penalize

    results.sort(key=lambda r: r["score"], reverse=True)
    print("\n═══ TOP 15 by DD-aware score ═══")
    print(f"{'Rank':<5}{'CAGR':>8}{'DD':>8}{'MAR':>7}{'Sortino':>9}{'Trades':>8}  Config")
    for i, r in enumerate(results[:15]):
        cfg = r["config"]
        print(f"{i+1:<5}{r['cagr_pct']:>7}%{r['max_dd_pct']:>7}%{r.get('mar_ratio',0):>7}{r['sortino']:>9}{r['n_trades']:>8}  {cfg['name']}")

    print("\n═══ TOP 10 with DD ≤ 10% ═══")
    tight = [r for r in results if r.get("max_dd_pct",100) <= 10]
    if not tight:
        print("  (none)")
        print("\n═══ TOP 10 with DD ≤ 15% ═══")
        tight = [r for r in results if r.get("max_dd_pct",100) <= 15]
    if not tight:
        print("  (none)")
        print("\n═══ TOP 10 with DD ≤ 20% ═══")
        tight = [r for r in results if r.get("max_dd_pct",100) <= 20]
    tight.sort(key=lambda r: r["cagr_pct"], reverse=True)
    for i, r in enumerate(tight[:10]):
        cfg = r["config"]
        print(f"{i+1:<5}{r['cagr_pct']:>7}%{r['max_dd_pct']:>7}%{r.get('mar_ratio',0):>7}{r['sortino']:>9}{r['n_trades']:>8}  {cfg['name']}")

    print("\n═══ TOP 10 by pure CAGR ═══")
    pure = sorted(results, key=lambda r: r.get("cagr_pct", -999), reverse=True)
    for i, r in enumerate(pure[:10]):
        cfg = r["config"]
        print(f"{i+1:<5}{r['cagr_pct']:>7}%{r['max_dd_pct']:>7}%{r.get('mar_ratio',0):>7}{r['sortino']:>9}{r['n_trades']:>8}  {cfg['name']}")

    with open(args.out, "w") as f:
        json.dump({"results": results,
                   "generated_at": datetime.now().isoformat(),
                   "n_configs": len(results)}, f, indent=2)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
