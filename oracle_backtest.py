#!/usr/bin/env python3
"""
Oracle backtest: what would a PERFECT picker achieve on our universe?
This sets the theoretical ceiling. No ML can do better than this.

Each rebalance period, picks the top-N best performers for the NEXT period
(cheating — uses future data). Then we know the absolute upper bound.
"""
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

CACHE_PATH = "/tmp/backtest_real_ml_cache.pkl"
START = "2016-01-01"
END = "2026-04-01"

data = pickle.load(open(CACHE_PATH, "rb"))
# Build price matrix
tickers = [t for t in data if len(data[t]) >= 2000 and t not in ("^VIX",)]
print(f"Universe: {len(tickers)} tickers")

# Align to SPY dates
spy = data["SPY"]
dates = [d for d in spy.index if pd.Timestamp(START) <= d <= pd.Timestamp(END)]
print(f"Dates: {len(dates)}  {dates[0].date()} to {dates[-1].date()}")

# Price matrix
prices = pd.DataFrame(index=dates)
for t in tickers:
    s = data[t]["Close"].reindex(dates).ffill()
    prices[t] = s

prices = prices.dropna(axis=1, thresh=len(dates)*0.5)  # drop columns >50% missing
tickers = list(prices.columns)
print(f"After cleanup: {len(tickers)} tickers")

def simulate_oracle(hold_days=10, top_n=5, leverage=1.0, initial=100000):
    """Every hold_days, pick top N best-performing tickers over NEXT hold_days."""
    cash = initial
    equity_curve = [initial]
    i = 0
    while i < len(dates) - hold_days:
        # Forward returns over next hold_days
        start_prices = prices.iloc[i].dropna()
        end_prices = prices.iloc[i + hold_days].dropna()
        common = start_prices.index.intersection(end_prices.index)
        rets = (end_prices[common] / start_prices[common] - 1)
        # Pick top N
        top = rets.nlargest(top_n)
        # Equal weight, with leverage
        avg_ret = top.mean() * leverage
        cash = cash * (1 + avg_ret)
        # Record curve (linear interp for smoothness)
        for k in range(1, hold_days + 1):
            equity_curve.append(cash * (1 + avg_ret * (k - hold_days) / hold_days))
        i += hold_days

    equity = np.array(equity_curve)
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (equity[-1] / initial) ** (1/years) - 1
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = abs(dd.min())
    returns = np.diff(equity) / equity[:-1]
    neg = returns[returns < 0]
    sortino = np.mean(returns) / np.std(neg) * np.sqrt(252) if len(neg) else 0
    return {
        "hold_days": hold_days, "top_n": top_n, "leverage": leverage,
        "cagr": cagr * 100, "max_dd": max_dd * 100,
        "final": equity[-1], "sortino": sortino,
    }

print("\nORACLE (perfect foresight, top-N picks held for N days):")
print(f"{'hold':<5}{'topN':<5}{'lev':<5}{'CAGR':>8}{'MaxDD':>8}{'Final':>14}{'Sortino':>9}")
for hold in [5, 10, 20]:
    for top_n in [1, 3, 5, 10]:
        for lev in [1.0]:
            r = simulate_oracle(hold, top_n, lev)
            print(f"{r['hold_days']:<5}{r['top_n']:<5}{r['leverage']:<5}{r['cagr']:>7.1f}%{r['max_dd']:>7.1f}%{r['final']:>14,.0f}{r['sortino']:>9.2f}")

# What if picker is only 50% right? Mix top with random
print("\nREALISTIC (50% hit rate on oracle + 50% random):")
rng = np.random.default_rng(42)
for hit_rate in [0.3, 0.5, 0.7, 0.9]:
    cash = 100000
    curve = [cash]
    i = 0
    while i < len(dates) - 10:
        start_p = prices.iloc[i].dropna()
        end_p = prices.iloc[i+10].dropna()
        common = start_p.index.intersection(end_p.index)
        rets = (end_p[common] / start_p[common] - 1)
        n_oracle = int(5 * hit_rate)
        n_random = 5 - n_oracle
        top = list(rets.nlargest(n_oracle).values) if n_oracle > 0 else []
        if n_random > 0:
            rand_picks = rng.choice(rets.values, n_random, replace=False)
            top.extend(rand_picks)
        avg = np.mean(top)
        cash *= (1 + avg)
        for k in range(10): curve.append(cash)
        i += 10
    eq = np.array(curve)
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (eq[-1] / 100000) ** (1/years) - 1
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    print(f"  hit_rate={hit_rate:.1%}: CAGR={cagr*100:.1f}%  MaxDD={abs(dd.min())*100:.1f}%")
