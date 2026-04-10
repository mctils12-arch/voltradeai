#!/usr/bin/env python3
"""
VolTradeAI — Full System Backtest
===================================
Simulates the ENTIRE VolTradeAI trading engine over the last 30 trading days
using real historical data from Alpaca.

Period:  2026-02-20 to 2026-04-04
Capital: $100,000
"""

import sys
import os
import json
import math
import time
import requests
from datetime import datetime, date
from collections import defaultdict

# ── Path setup so we can import from voltrade/ ──────────────────────────────
VOLTRADE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VOLTRADE_DIR)

from bot_engine import score_stock
from position_sizing import calculate_position

# ── Alpaca credentials ────────────────────────────────────────────────────
ALPACA_KEY    = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
DATA_URL      = "https://data.alpaca.markets"
BROKER_URL    = "https://paper-api.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

# ── Backtest parameters ───────────────────────────────────────────────────
STARTING_CAPITAL   = 100_000.0
MAX_POSITIONS      = 8
MIN_SCORE          = 65
STOP_LOSS_PCT      = 0.03   # -3 % from entry
TAKE_PROFIT_PCT    = 0.08   # +8 % from entry
TIME_STOP_DAYS     = 5      # Max hold days
DEFAULT_VIX        = 20.0

BACKTEST_START = "2026-02-20"
BACKTEST_END   = "2026-04-04"

# ── Universe (80 tickers) ─────────────────────────────────────────────────
UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","NFLX","AVGO",
    "CRM","ORCL","ADBE","INTC","QCOM","MU","AMAT","LRCX","KLAC","MRVL",
    "SNPS","CDNS","ARM","SMCI","PLTR","COIN","MSTR","SQ","SHOP","ROKU",
    "SNAP","HOOD","SOFI","AFRM","UPST","RIVN","LCID","NIO","BABA","JD",
    "PDD","LI","SPY","QQQ","IWM","DIA","XLE","XLF","XLK","XLV",
    "JPM","BAC","GS","MS","WFC","V","MA","JNJ","PFE","UNH","MRK","ABBV",
    "LLY","WMT","COST","HD","TGT","LOW","XOM","CVX","COP","SLB","OXY",
    "BA","CAT","DE","GE","HON","T","VZ","TMUS",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ALPACA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_trading_calendar(start: str, end: str) -> list[str]:
    """Fetch Alpaca calendar and return list of trading day strings (YYYY-MM-DD)."""
    url = f"{BROKER_URL}/v2/calendar"
    params = {"start": start, "end": end}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return [d["date"] for d in resp.json()]


def fetch_bars_batch(symbols: list[str], start: str, end: str,
                     chunk_size: int = 50) -> dict:
    """
    Fetch daily OHLCV for multiple symbols in batches.
    Returns: {ticker: {date_str: {o, h, l, c, v, vw}}}
    """
    all_bars: dict = defaultdict(dict)

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        params = {
            "symbols":    ",".join(chunk),
            "timeframe":  "1Day",
            "start":      start,
            "end":        end,
            "limit":      1000,
            "adjustment": "split",
            "feed":       "sip",
        }
        url = f"{DATA_URL}/v2/stocks/bars"
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  [WARN] bars batch {i//chunk_size+1} HTTP {resp.status_code}: {resp.text[:200]}")
            continue

        data = resp.json()
        bars_map = data.get("bars", {})
        for ticker, bars in bars_map.items():
            for bar in bars:
                # Alpaca timestamps look like "2026-02-20T00:00:00Z"
                day = bar["t"][:10]
                all_bars[ticker][day] = {
                    "o":  bar.get("o", 0),
                    "h":  bar.get("h", 0),
                    "l":  bar.get("l", 0),
                    "c":  bar.get("c", 0),
                    "v":  bar.get("v", 0),
                    "vw": bar.get("vw", bar.get("c", 0)),  # fallback to close
                }

        # Respect rate limits — short pause between batches
        time.sleep(0.3)

    return dict(all_bars)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PORTFOLIO STATE
# ─────────────────────────────────────────────────────────────────────────────

class Portfolio:
    def __init__(self, starting_cash: float):
        self.cash       = starting_cash
        self.equity     = starting_cash
        self.positions  = {}   # ticker → {shares, entry_price, entry_date, score}
        self.trades     = []   # full trade log
        self.eq_curve   = []   # [{date, equity, daily_pnl, positions_held}]

    def open_position(self, ticker, shares, price, trade_date, score):
        cost = shares * price
        if cost > self.cash:
            return False
        self.cash -= cost
        self.positions[ticker] = {
            "ticker":      ticker,
            "shares":      shares,
            "entry_price": price,
            "entry_date":  trade_date,
            "score":       score,
        }
        self.trades.append({
            "date":      trade_date,
            "action":    "BUY",
            "ticker":    ticker,
            "shares":    shares,
            "price":     round(price, 2),
            "score":     score,
            "pnl":       None,
            "pnl_pct":   None,
            "days_held": None,
            "reason":    "entry",
        })
        return True

    def close_position(self, ticker, price, trade_date, reason):
        if ticker not in self.positions:
            return
        pos = self.positions.pop(ticker)
        proceeds  = pos["shares"] * price
        cost      = pos["shares"] * pos["entry_price"]
        pnl       = proceeds - cost
        pnl_pct   = (pnl / cost) * 100 if cost > 0 else 0

        # Days held
        entry_dt  = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
        exit_dt   = datetime.strptime(trade_date, "%Y-%m-%d")
        days_held = (exit_dt - entry_dt).days

        self.cash += proceeds
        self.trades.append({
            "date":      trade_date,
            "action":    "SELL",
            "ticker":    ticker,
            "shares":    pos["shares"],
            "price":     round(price, 2),
            "score":     pos["score"],
            "pnl":       round(pnl, 2),
            "pnl_pct":   round(pnl_pct, 2),
            "days_held": days_held,
            "reason":    reason,
        })

    def mark_to_market(self, prices: dict):
        """Revalue open positions at current prices."""
        pos_value = sum(
            pos["shares"] * prices.get(t, pos["entry_price"])
            for t, pos in self.positions.items()
        )
        return self.cash + pos_value

    def snapshot(self, trade_date: str, prev_equity: float, prices: dict):
        self.equity = self.mark_to_market(prices)
        daily_pnl   = self.equity - prev_equity
        self.eq_curve.append({
            "date":            trade_date,
            "equity":          round(self.equity, 2),
            "daily_pnl":       round(daily_pnl, 2),
            "positions_held":  len(self.positions),
        })
        return self.equity


# ─────────────────────────────────────────────────────────────────────────────
# 3.  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(portfolio: Portfolio) -> dict:
    sells = [t for t in portfolio.trades if t["action"] == "SELL"]
    if not sells:
        return {}

    wins   = [t for t in sells if (t["pnl"] or 0) > 0]
    losses = [t for t in sells if (t["pnl"] or 0) <= 0]

    win_rate  = len(wins) / len(sells) * 100 if sells else 0
    avg_win   = (sum(t["pnl_pct"] for t in wins)   / len(wins))   if wins   else 0
    avg_loss  = (sum(t["pnl_pct"] for t in losses) / len(losses)) if losses else 0

    total_win  = sum(t["pnl"] for t in wins   if t["pnl"])
    total_loss = abs(sum(t["pnl"] for t in losses if t["pnl"]))
    profit_factor = (total_win / total_loss) if total_loss > 0 else float("inf")

    # Max drawdown from equity curve
    equities = [e["equity"] for e in portfolio.eq_curve]
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    final_equity  = portfolio.equity
    total_return  = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    # Approximate Sharpe (using daily returns)
    daily_returns = []
    for i in range(1, len(portfolio.eq_curve)):
        prev = portfolio.eq_curve[i-1]["equity"]
        curr = portfolio.eq_curve[i]["equity"]
        if prev > 0:
            daily_returns.append((curr - prev) / prev)

    if len(daily_returns) > 1:
        import statistics
        mean_r  = statistics.mean(daily_returns)
        std_r   = statistics.stdev(daily_returns)
        sharpe  = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0
    else:
        sharpe = 0

    return {
        "total_trades":   len(sells),
        "win_rate":       round(win_rate, 1),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "profit_factor":  round(profit_factor, 2) if profit_factor != float("inf") else 9999,
        "max_drawdown":   round(-max_dd, 2),
        "final_equity":   round(final_equity, 2),
        "total_return":   round(total_return, 2),
        "sharpe":         round(sharpe, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PRINTING / REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_report(portfolio: Portfolio, metrics: dict):
    print()
    print("=" * 70)
    print("  VOLTRADEAI FULL SYSTEM BACKTEST")
    print(f"  Period:          {BACKTEST_START} to {BACKTEST_END}")
    print(f"  Starting Capital: ${STARTING_CAPITAL:,.0f}")
    print("=" * 70)

    print("\nEQUITY CURVE:")
    print(f"{'Date':<12} {'Equity':>12} {'Daily P&L':>12} {'Positions':>10}")
    print("-" * 50)
    for row in portfolio.eq_curve:
        sign = "+" if row["daily_pnl"] >= 0 else ""
        print(
            f"{row['date']:<12} "
            f"${row['equity']:>11,.2f} "
            f"{sign}${row['daily_pnl']:>10,.2f} "
            f"{row['positions_held']:>10}"
        )

    print("\nPERFORMANCE METRICS:")
    print("-" * 40)
    print(f"  Total Trades:     {metrics.get('total_trades', 0)}")
    print(f"  Win Rate:         {metrics.get('win_rate', 0):.1f}%")
    print(f"  Average Win:      +{metrics.get('avg_win', 0):.2f}%")
    print(f"  Average Loss:     {metrics.get('avg_loss', 0):.2f}%")
    pf = metrics.get('profit_factor', 0)
    pf_str = f"{pf:.2f}" if pf < 9999 else "∞"
    print(f"  Profit Factor:    {pf_str}")
    print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):.2f}%")
    print(f"  Final Equity:     ${metrics.get('final_equity', 0):,.2f}")
    print(f"  Total Return:     {'+' if metrics.get('total_return',0)>=0 else ''}{metrics.get('total_return', 0):.2f}%")
    print(f"  Sharpe Ratio:     {metrics.get('sharpe', 0):.2f}")

    print("\nALL TRADES:")
    print(f"{'Date':<12} {'Action':<6} {'Ticker':<6} {'Shares':>6} {'Price':>9} {'Score':>6} {'P&L':>10} {'P&L%':>7} {'Days':>5} {'Reason'}")
    print("-" * 80)
    for t in portfolio.trades:
        pnl_str  = f"${t['pnl']:+,.2f}"  if t["pnl"]  is not None else "-"
        pct_str  = f"{t['pnl_pct']:+.1f}%" if t["pnl_pct"] is not None else "-"
        days_str = str(t["days_held"])    if t["days_held"] is not None else "-"
        reason   = t.get("reason", "")
        print(
            f"{t['date']:<12} "
            f"{t['action']:<6} "
            f"{t['ticker']:<6} "
            f"{t['shares']:>6} "
            f"${t['price']:>8.2f} "
            f"{t['score']:>6} "
            f"{pnl_str:>10} "
            f"{pct_str:>7} "
            f"{days_str:>5}  "
            f"{reason}"
        )

    # Position sizing analysis
    buy_trades = [t for t in portfolio.trades if t["action"] == "BUY"]
    if buy_trades:
        position_values = [t["shares"] * t["price"] for t in buy_trades]
        avg_pct  = sum(v / STARTING_CAPITAL * 100 for v in position_values) / len(position_values)
        max_val  = max(position_values)
        min_val  = min(position_values)
        max_pct  = max_val / STARTING_CAPITAL * 100
        min_pct  = min_val / STARTING_CAPITAL * 100
        print("\nPOSITION SIZING ANALYSIS:")
        print("-" * 40)
        print(f"  Avg position size:  {avg_pct:.1f}% of portfolio")
        print(f"  Largest position:   ${max_val:,.0f} ({max_pct:.1f}%)")
        print(f"  Smallest position:  ${min_val:,.0f} ({min_pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest():
    t0 = time.time()
    print("VolTradeAI Full System Backtest — loading data...")

    # ── Step 1: Get trading calendar ──────────────────────────────────────
    print(f"  Fetching trading calendar {BACKTEST_START} → {BACKTEST_END}...")
    trading_days = get_trading_calendar(BACKTEST_START, BACKTEST_END)
    print(f"  → {len(trading_days)} trading days found")

    # ── Step 2: Fetch all bars in batches ─────────────────────────────────
    print(f"  Fetching OHLCV bars for {len(UNIVERSE)} tickers (batches of 50)...")
    all_bars = fetch_bars_batch(UNIVERSE, BACKTEST_START, BACKTEST_END, chunk_size=50)
    tickers_with_data = [t for t in UNIVERSE if t in all_bars and len(all_bars[t]) > 0]
    print(f"  → Got data for {len(tickers_with_data)}/{len(UNIVERSE)} tickers")
    print(f"  Data load completed in {time.time()-t0:.1f}s")
    print()

    # ── Step 3: Run daily simulation ──────────────────────────────────────
    portfolio  = Portfolio(STARTING_CAPITAL)
    prev_equity = STARTING_CAPITAL

    for day_idx, trade_date in enumerate(trading_days):
        # Collect today's prices for all tickers
        day_prices: dict[str, dict] = {}
        for ticker in tickers_with_data:
            bar = all_bars.get(ticker, {}).get(trade_date)
            if bar and bar["c"] > 0:
                day_prices[ticker] = bar

        if not day_prices:
            print(f"  {trade_date}: no price data — skipping")
            continue

        # ── A. Manage existing positions first ────────────────────────────
        to_close = []
        for ticker, pos in list(portfolio.positions.items()):
            bar = day_prices.get(ticker)
            if bar is None:
                continue
            close_price = bar["c"]
            entry       = pos["entry_price"]

            # Days held (count of trading days from entry)
            entry_idx  = trading_days.index(pos["entry_date"]) if pos["entry_date"] in trading_days else 0
            days_held  = day_idx - entry_idx

            pct_change = (close_price - entry) / entry

            if pct_change <= -STOP_LOSS_PCT:
                to_close.append((ticker, close_price, "stop_loss"))
            elif pct_change >= TAKE_PROFIT_PCT:
                to_close.append((ticker, close_price, "take_profit"))
            elif days_held >= TIME_STOP_DAYS:
                to_close.append((ticker, close_price, "time_stop"))

        for ticker, close_price, reason in to_close:
            portfolio.close_position(ticker, close_price, trade_date, reason)

        # ── B. Score all tickers for today ───────────────────────────────
        scored = []
        for ticker, bar in day_prices.items():
            stock_data = {
                "T":  ticker,
                "c":  bar["c"],
                "o":  bar["o"],
                "h":  bar["h"],
                "l":  bar["l"],
                "v":  bar["v"],
                "vw": bar["vw"],
            }
            result = score_stock(stock_data)
            if result is not None:
                scored.append(result)

        # Sort by quick_score descending, take top 10
        scored.sort(key=lambda x: x["quick_score"], reverse=True)
        top10 = scored[:10]

        # ── C. Try to enter new positions ────────────────────────────────
        for candidate in top10:
            if len(portfolio.positions) >= MAX_POSITIONS:
                break

            ticker = candidate["ticker"]
            score  = candidate["quick_score"]

            if score < MIN_SCORE:
                continue
            if ticker in portfolio.positions:
                continue  # Already holding

            close_price = candidate["price"]

            # Build trade dict for position_sizing
            trade_dict = {
                "ticker":     ticker,
                "price":      close_price,
                "score":      score,
                "deep_score": score,
                "volume":     candidate["volume"],
                "side":       "buy",
                "trade_type": "stock",
            }

            # Current positions formatted as Alpaca-style list
            alpaca_positions = [
                {
                    "symbol":        t,
                    "qty":           str(p["shares"]),
                    "current_price": str(day_prices.get(t, {}).get("c", p["entry_price"])),
                    "market_value":  str(p["shares"] * day_prices.get(t, {}).get("c", p["entry_price"])),
                }
                for t, p in portfolio.positions.items()
            ]

            sizing = calculate_position(
                trade        = trade_dict,
                equity       = portfolio.equity if portfolio.equity > 0 else STARTING_CAPITAL,
                current_positions = alpaca_positions,
                macro        = {"vix": DEFAULT_VIX, "vix_regime": "normal"},
            )

            if sizing.get("blocked"):
                continue

            shares = sizing.get("shares", 0)
            if shares <= 0:
                continue

            cost = shares * close_price
            if cost > portfolio.cash:
                # Try with fewer shares if we can afford at least 1
                affordable = int(portfolio.cash / close_price)
                if affordable < 1:
                    continue
                shares = affordable

            portfolio.open_position(ticker, shares, close_price, trade_date, score)

        # ── D. Snapshot equity at end of day ─────────────────────────────
        eod_prices = {t: day_prices[t]["c"] for t in day_prices}
        prev_equity = portfolio.snapshot(trade_date, prev_equity, eod_prices)

    # ── Step 4: Close any remaining positions at last day's close ─────────
    last_day = trading_days[-1]
    last_prices = {}
    for ticker in tickers_with_data:
        bar = all_bars.get(ticker, {}).get(last_day)
        if bar and bar["c"] > 0:
            last_prices[ticker] = bar["c"]

    for ticker in list(portfolio.positions.keys()):
        price = last_prices.get(ticker, portfolio.positions[ticker]["entry_price"])
        portfolio.close_position(ticker, price, last_day, "end_of_backtest")

    # ── Step 5: Compute metrics & report ──────────────────────────────────
    metrics = compute_metrics(portfolio)
    print_report(portfolio, metrics)

    print(f"\n[Backtest completed in {time.time()-t0:.1f}s]")

    # ── Step 6: Save JSON results ──────────────────────────────────────────
    results = {
        "equity_curve": portfolio.eq_curve,
        "trades":       portfolio.trades,
        "metrics":      metrics,
    }
    out_path = os.path.join(VOLTRADE_DIR, "backtest_full_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    run_backtest()
