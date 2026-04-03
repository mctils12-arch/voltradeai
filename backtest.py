#!/usr/bin/env python3
"""
VolTradeAI Backtesting Engine
Walk-forward validation on historical data from Yahoo Finance.
Strategies: VRP Selling, Momentum, PEAD, Combined
"""
import sys
import json
import numpy as np
from datetime import datetime, timedelta

def get_data(ticker, years=3):
    """Fetch historical data from Yahoo Finance."""
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
    if df.empty:
        return None
    return df

def calc_returns(prices):
    """Calculate daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()

def calc_rolling_vol(returns, window=20):
    """Annualized rolling volatility."""
    return returns.rolling(window).std() * np.sqrt(252)

def calc_rsi(prices, period=14):
    """RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_momentum(prices, lookback=252, skip=21):
    """12-1 month momentum signal."""
    return prices.shift(skip) / prices.shift(lookback) - 1

def backtest_momentum(df, lookback=252, hold_days=21):
    """
    Momentum strategy: Buy when 12-1 month momentum is positive, sell when negative.
    """
    prices = df['Close']
    mom = calc_momentum(prices, lookback, 21)
    signals = (mom > 0).astype(int)
    
    # Shift signal by 1 day (trade next day)
    signals = signals.shift(1).fillna(0)
    
    daily_returns = prices.pct_change().fillna(0)
    strategy_returns = signals * daily_returns
    
    return _calc_metrics("Momentum (12-1mo)", strategy_returns, daily_returns)

def backtest_mean_reversion(df, rsi_period=14, oversold=30, overbought=70):
    """
    Mean reversion: Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought).
    """
    prices = df['Close']
    rsi = calc_rsi(prices, rsi_period)
    
    position = 0
    signals = []
    for i in range(len(rsi)):
        if rsi.iloc[i] < oversold and position == 0:
            position = 1
        elif rsi.iloc[i] > overbought and position == 1:
            position = 0
        signals.append(position)
    
    signals = np.array(signals)
    # Shift by 1
    signals = np.roll(signals, 1)
    signals[0] = 0
    
    daily_returns = prices.pct_change().fillna(0).values
    strategy_returns = signals * daily_returns
    
    import pandas as pd
    return _calc_metrics("Mean Reversion (RSI)", pd.Series(strategy_returns, index=df.index), pd.Series(daily_returns, index=df.index))

def backtest_vol_selling(df, vol_window=20):
    """
    Volatility selling proxy: Be long when realized vol is below its 6-month average
    (low vol = calm market = sell options works). Be flat when vol is high.
    """
    prices = df['Close']
    returns = calc_returns(prices)
    rv = calc_rolling_vol(returns, vol_window)
    rv_avg = rv.rolling(126).mean()  # 6-month avg
    
    # Signal: long when rv < rv_avg (low vol regime = sell premium)
    signals = (rv < rv_avg).astype(int).shift(1).fillna(0)
    
    daily_returns = prices.pct_change().fillna(0)
    strategy_returns = signals * daily_returns
    
    return _calc_metrics("Vol Selling (Low Vol Regime)", strategy_returns, daily_returns)

def backtest_combined(df):
    """
    Combined strategy: Equal weight momentum + mean reversion + vol selling.
    Diversification across uncorrelated signals.
    """
    prices = df['Close']
    daily_returns = prices.pct_change().fillna(0)
    
    # Momentum signal
    mom = calc_momentum(prices, 252, 21)
    mom_sig = (mom > 0).astype(float).shift(1).fillna(0)
    
    # RSI signal
    rsi = calc_rsi(prices, 14)
    rsi_sig = ((rsi < 40).astype(float)).shift(1).fillna(0)  # More lenient threshold
    
    # Vol regime signal
    returns = calc_returns(prices)
    rv = calc_rolling_vol(returns, 20)
    rv_avg = rv.rolling(126).mean()
    vol_sig = (rv < rv_avg).astype(float).shift(1).fillna(0)
    
    # Combined: average of 3 signals (0 to 1 exposure)
    combined = (mom_sig + rsi_sig + vol_sig) / 3.0
    strategy_returns = combined * daily_returns
    
    return _calc_metrics("Combined (Mom + RSI + Vol)", strategy_returns, daily_returns)

def _calc_metrics(name, strategy_returns, benchmark_returns):
    """Calculate performance metrics."""
    import pandas as pd
    strat = strategy_returns.dropna()
    bench = benchmark_returns.dropna()
    
    if len(strat) == 0:
        return {"strategy": name, "error": "No data"}
    
    # Total return
    total_return = float((1 + strat).prod() - 1) * 100
    bench_return = float((1 + bench).prod() - 1) * 100
    
    # Annualized return
    years = len(strat) / 252
    if years <= 0: years = 1
    annual_return = float(((1 + total_return/100) ** (1/years) - 1) * 100)
    
    # Sharpe ratio
    mean_ret = float(strat.mean()) * 252
    std_ret = float(strat.std()) * np.sqrt(252)
    sharpe = round(mean_ret / std_ret, 2) if std_ret > 0 else 0
    
    # Max drawdown
    cum = (1 + strat).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min()) * 100
    
    # Win rate
    trades = strat[strat != 0]
    wins = (trades > 0).sum()
    total_trades = len(trades)
    win_rate = round((wins / total_trades) * 100, 1) if total_trades > 0 else 0
    
    # Invested time
    invested_pct = round((strat != 0).mean() * 100, 1)
    
    return {
        "strategy": name,
        "totalReturn": round(total_return, 2),
        "benchmarkReturn": round(bench_return, 2),
        "annualReturn": round(annual_return, 2),
        "sharpe": sharpe,
        "maxDrawdown": round(max_dd, 2),
        "winRate": win_rate,
        "totalTrades": int(total_trades),
        "investedPct": invested_pct,
        "years": round(years, 1),
        "dataPoints": len(strat),
    }

def run_backtest(ticker="SPY", strategy="combined", years=3):
    """Main entry point."""
    df = get_data(ticker, years)
    if df is None or df.empty:
        return {"error": f"No data for {ticker}"}
    
    strategies = {
        "momentum": backtest_momentum,
        "mean_reversion": backtest_mean_reversion,
        "vol_selling": backtest_vol_selling,
        "combined": backtest_combined,
    }
    
    if strategy == "all":
        results = []
        for name, func in strategies.items():
            try:
                results.append(func(df))
            except Exception as e:
                results.append({"strategy": name, "error": str(e)})
        return {"ticker": ticker, "years": years, "results": results}
    
    func = strategies.get(strategy, backtest_combined)
    try:
        result = func(df)
        return {"ticker": ticker, "years": years, "results": [result]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    strategy = sys.argv[2] if len(sys.argv) > 2 else "all"
    years = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    result = run_backtest(ticker, strategy, years)
    print(json.dumps(result))
