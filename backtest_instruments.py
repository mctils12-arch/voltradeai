#!/usr/bin/env python3
"""
VolTradeAI Historical Backtest Engine
======================================
Pulls last 30 trading days from Alpaca, simulates the scanner (top movers +
most actives), runs a simplified quick_score, and for every stock scoring ≥ 65
records what would have happened holding 1 / 2 / 3 / 5 days in both the stock
and its 2× leveraged ETF equivalent.

Outputs:
  - Progress to stdout while running
  - Summary table to stdout
  - Detailed JSON to /home/user/workspace/voltrade/backtest_results.json
"""

import os
import json
import time
import sys
from datetime import datetime, timedelta, date
from collections import defaultdict
import requests

# ── Alpaca credentials ────────────────────────────────────────────────────────
ALPACA_KEY    = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
DATA_URL      = "https://data.alpaca.markets"
HEADERS       = {
    "APCA-API-KEY-ID":     ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

# ── Strategy parameters ───────────────────────────────────────────────────────
MIN_SCORE  = 65
MIN_PRICE  = 5
MIN_VOLUME = 500_000
HOLD_DAYS  = [1, 2, 3, 5]

# ── Leveraged ETF map ─────────────────────────────────────────────────────────
LEVERAGED_ETFS = {
    'TSLA': 'TSLL', 'NVDA': 'NVDL', 'AAPL': 'AAPU', 'AMZN': 'AMZU',
    'MSFT': 'MSFU', 'META': 'METU', 'GOOGL': 'GGLL', 'AMD':  'AMDU',
    'COIN': 'CONL', 'MSTR': 'MSTX', 'PLTR': 'PTIR',  'SMCI': 'SMCU',
    'SOFI': 'SOFL', 'SPY':  'SSO',  'QQQ':  'QLD',   'IWM':  'TNA',
}

# Large-cap list for spread estimation
LARGE_CAPS = {
    'AAPL','MSFT','GOOGL','GOOG','AMZN','NVDA','META','TSLA',
    'BRK.B','JPM','UNH','V','MA','XOM','JNJ','WMT','PG','LLY',
    'HD','MRK','AVGO','ABBV','KO','PEP','COST','CVX','MCD','TMO',
    'ACN','BAC','CRM','AMD','NFLX','DIS','ADBE','WFC','QCOM','TXN',
    'SPY','QQQ','IWM','DIA','GLD',
}

# Major ETF tickers (lower spread)
MAJOR_ETFS = {'SSO','QLD','TNA','UPRO','TQQQ','SPXL'}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def alpaca_get(path, params=None, retries=3):
    """GET from Alpaca DATA endpoint with retry."""
    url = DATA_URL + path
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                print(f"  [rate-limit] sleeping 2s …")
                time.sleep(2)
            else:
                print(f"  [warn] {path} → HTTP {r.status_code}: {r.text[:120]}")
                return None
        except Exception as e:
            print(f"  [error] {path}: {e}")
            time.sleep(1)
    return None


def get_trading_days(n=35):
    """
    Return the last n calendar days worth of dates and let Alpaca bars tell us
    which ones were actual trading days.  We ask for a bit more than 30 so that
    after skipping weekends / holidays we still have ≥ 30 real trading days.
    """
    end   = date.today()
    start = end - timedelta(days=n)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca screener: movers + most-actives
# ─────────────────────────────────────────────────────────────────────────────

def fetch_screener_universe():
    """
    Pull today's most-active and top-mover tickers from Alpaca screener.
    (These endpoints return real-time data; we use them purely to build
    the universe of tickers the bot *would* watch over the back-test window.)
    """
    universe = set()

    # Most actives
    data = alpaca_get("/v1beta1/screener/stocks/most-actives", {"by": "volume", "top": 50})
    if data and "most_actives" in data:
        for item in data["most_actives"]:
            sym = item.get("symbol", "")
            if sym and "." not in sym and len(sym) <= 5:
                universe.add(sym)
    time.sleep(0.1)

    # Top gainers
    data = alpaca_get("/v1beta1/screener/stocks/movers", {"top": 50})
    if data:
        for key in ("gainers", "losers"):
            for item in data.get(key, []):
                sym = item.get("symbol", "")
                if sym and "." not in sym and len(sym) <= 5:
                    universe.add(sym)
    time.sleep(0.1)

    # Always include ETF leveraged-pair stocks too
    universe.update(LEVERAGED_ETFS.keys())

    print(f"  Universe size: {len(universe)} tickers")
    return list(universe)


# ─────────────────────────────────────────────────────────────────────────────
# Historical bars fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_bars_bulk(tickers, start, end, batch_size=50):
    """
    Fetch daily bars for a list of tickers using the multi-bar endpoint.
    Returns dict: {ticker: [bar, ...]} sorted by ascending date.
    """
    all_bars = defaultdict(list)
    tickers = list(tickers)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        params = {
            "symbols":   ",".join(batch),
            "timeframe": "1Day",
            "start":     start,
            "end":       end,
            "limit":     10000,
            "adjustment": "split",
            "feed":       "sip",
        }
        data = alpaca_get("/v2/stocks/bars", params)
        if data and "bars" in data:
            for sym, bars in data["bars"].items():
                all_bars[sym].extend(bars)
        time.sleep(0.1)
        print(f"  Fetched bars batch {i // batch_size + 1} "
              f"({min(i + batch_size, len(tickers))}/{len(tickers)} tickers)")

    # Sort each ticker's bars chronologically
    for sym in all_bars:
        all_bars[sym].sort(key=lambda b: b["t"])

    return all_bars


# ─────────────────────────────────────────────────────────────────────────────
# Quick scoring (mirrors bot_engine.score_stock logic, Alpaca-bar edition)
# ─────────────────────────────────────────────────────────────────────────────

def quick_score(bar):
    """
    Score a single daily bar dict from Alpaca.
    Alpaca bar fields: o, h, l, c, v, vw, t
    Returns score (0-100) or None if filtered out.
    """
    close  = bar.get("c", 0)
    open_p = bar.get("o", 0)
    high   = bar.get("h", 0)
    low    = bar.get("l", 0)
    volume = bar.get("v", 0)
    vwap   = bar.get("vw", 0)

    if close < MIN_PRICE or volume < MIN_VOLUME:
        return None

    change_pct = ((close - open_p) / open_p * 100) if open_p > 0 else 0
    range_pct  = ((high - low) / low * 100) if low > 0 else 0
    vwap_dist  = ((close - vwap) / vwap * 100) if vwap > 0 else 0

    score = 50

    # Price action
    if change_pct > 3:
        score += 10
    elif change_pct < -3:
        score += 8   # bounce candidate

    # Volume
    if volume > 20_000_000:
        score += 15
    elif volume > 5_000_000:
        score += 8
    elif volume > 1_000_000:
        score += 3

    # VWAP position
    if vwap_dist > 1:
        score += 5
    elif vwap_dist < -1:
        score += 3

    # Intraday range
    if range_pct > 5:
        score += 10
    elif range_pct > 3:
        score += 5

    return max(0, min(100, score))


# ─────────────────────────────────────────────────────────────────────────────
# Spread estimation
# ─────────────────────────────────────────────────────────────────────────────

def spread_stock(ticker):
    """Round-trip spread cost estimate for a stock (as a fraction, not %)."""
    return 0.0003 if ticker in LARGE_CAPS else 0.0008   # 0.03% or 0.08%


def spread_etf(ticker):
    """Round-trip spread cost estimate for a leveraged ETF."""
    return 0.0010 if ticker in MAJOR_ETFS else 0.0020   # 0.10% or 0.20%


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest logic
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest():
    print("\n" + "="*60)
    print("  VolTradeAI Historical Backtest Engine")
    print("="*60)

    # ── Step 1: Determine date range ─────────────────────────────────────────
    start_str, end_str = get_trading_days(n=45)   # extra buffer
    print(f"\n[1/5] Date window: {start_str} → {end_str}")

    # ── Step 2: Build universe ───────────────────────────────────────────────
    print("\n[2/5] Building scanner universe …")
    universe = fetch_screener_universe()

    # Also collect all leveraged ETF tickers so we can fetch their bars too
    etf_tickers = list(set(LEVERAGED_ETFS.values()))
    all_tickers = list(set(universe) | set(etf_tickers))
    print(f"  Total tickers to fetch bars for: {len(all_tickers)}")

    # ── Step 3: Fetch historical bars ────────────────────────────────────────
    print("\n[3/5] Fetching historical daily bars from Alpaca …")
    all_bars = fetch_bars_bulk(all_tickers, start_str, end_str, batch_size=50)
    print(f"  Bars received for {len(all_bars)} tickers")

    # Build index: ticker → {date_str: bar}
    bar_index = {}
    for sym, bars in all_bars.items():
        bar_index[sym] = {}
        for b in bars:
            d = b["t"][:10]   # "YYYY-MM-DD"
            bar_index[sym][d] = b

    # Determine actual trading days present in the data (use SPY as reference)
    reference = "SPY" if "SPY" in bar_index else list(bar_index.keys())[0]
    trading_days = sorted(bar_index[reference].keys())

    # Keep only last 30 trading days
    trading_days = trading_days[-30:]
    print(f"  Trading days used: {trading_days[0]} → {trading_days[-1]}"
          f" ({len(trading_days)} days)")

    # ── Step 4: Simulate scanner + score + record trades ─────────────────────
    print("\n[4/5] Simulating scanner and scoring …")

    trades = []
    skipped_etf_no_data = 0
    total_signals = 0

    for day_idx, trade_date in enumerate(trading_days):
        day_signals = []

        for ticker in universe:
            if ticker not in bar_index:
                continue
            bar = bar_index[ticker].get(trade_date)
            if bar is None:
                continue

            score = quick_score(bar)
            if score is None or score < MIN_SCORE:
                continue

            # Skip leveraged ETFs themselves as signal tickers
            if ticker in LEVERAGED_ETFS.values():
                continue

            day_signals.append((ticker, score, bar))

        total_signals += len(day_signals)

        # For each signal, calculate multi-day returns
        for ticker, score, entry_bar in day_signals:
            entry_price_stock = entry_bar["c"]
            etf_ticker        = LEVERAGED_ETFS.get(ticker)

            # ETF entry price
            entry_price_etf = None
            if etf_ticker and etf_ticker in bar_index:
                etf_entry_bar = bar_index[etf_ticker].get(trade_date)
                if etf_entry_bar:
                    entry_price_etf = etf_entry_bar["c"]

            # Compute returns for each hold period
            stock_returns = {}
            etf_returns   = {}
            future_dates  = trading_days[day_idx + 1:]   # days after entry

            for hold in HOLD_DAYS:
                # Stock return
                if len(future_dates) >= hold:
                    exit_date  = future_dates[hold - 1]
                    exit_bar   = bar_index.get(ticker, {}).get(exit_date)
                    if exit_bar:
                        ret = (exit_bar["c"] - entry_price_stock) / entry_price_stock * 100
                        stock_returns[hold] = round(ret, 4)

                # ETF return
                if etf_ticker and entry_price_etf and len(future_dates) >= hold:
                    exit_date  = future_dates[hold - 1]
                    etf_exit_bar = bar_index.get(etf_ticker, {}).get(exit_date)
                    if etf_exit_bar:
                        ret = (etf_exit_bar["c"] - entry_price_etf) / entry_price_etf * 100
                        etf_returns[hold] = round(ret, 4)
                    else:
                        skipped_etf_no_data += 1

            # Only record if we have at least 1-day returns
            if 1 not in stock_returns:
                continue

            # Spread costs (one-way each side = round-trip)
            sp_stock = spread_stock(ticker) * 100   # convert to %
            sp_etf   = (spread_etf(etf_ticker) * 100
                        if etf_ticker else None)

            # Net returns (subtract round-trip spread from gross return)
            # Use 2-day hold as the "primary" benchmark for net_return fields
            primary_hold = 2
            gross_s = stock_returns.get(primary_hold)
            gross_e = etf_returns.get(primary_hold)

            net_stock = round(gross_s - sp_stock, 4) if gross_s is not None else None
            net_etf   = round(gross_e - sp_etf,   4) if (gross_e is not None and sp_etf is not None) else None
            etf_adv   = round(net_etf - net_stock, 4) if (net_etf is not None and net_stock is not None) else None

            trade = {
                "date":                 trade_date,
                "ticker":               ticker,
                "etf_ticker":           etf_ticker,
                "score":                score,
                "entry_price_stock":    round(entry_price_stock, 4),
                "entry_price_etf":      round(entry_price_etf, 4) if entry_price_etf else None,
                # gross returns
                "stock_return_1d":      stock_returns.get(1),
                "stock_return_2d":      stock_returns.get(2),
                "stock_return_3d":      stock_returns.get(3),
                "stock_return_5d":      stock_returns.get(5),
                "etf_return_1d":        etf_returns.get(1),
                "etf_return_2d":        etf_returns.get(2),
                "etf_return_3d":        etf_returns.get(3),
                "etf_return_5d":        etf_returns.get(5),
                # spread estimates (round-trip %)
                "spread_estimate_stock": round(sp_stock, 4),
                "spread_estimate_etf":  round(sp_etf, 4) if sp_etf else None,
                # net returns at 2-day hold (primary)
                "net_return_stock":     net_stock,
                "net_return_etf":       net_etf,
                "etf_advantage":        etf_adv,
            }
            trades.append(trade)

        if (day_idx + 1) % 5 == 0 or day_idx == len(trading_days) - 1:
            print(f"  Processed {day_idx + 1}/{len(trading_days)} days, "
                  f"{len(trades)} trades accumulated …")

    print(f"\n  Total scanner signals: {total_signals}")
    print(f"  Trades with 1d+ data: {len(trades)}")
    print(f"  ETF comparisons skipped (no data): {skipped_etf_no_data}")

    # ── Step 5: Compute summary statistics ───────────────────────────────────
    print("\n[5/5] Computing summary statistics …")

    summary = compute_summary(trades)

    # ── Print results ─────────────────────────────────────────────────────────
    print_summary(summary, trades)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output_path = "/home/user/workspace/voltrade/backtest_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "generated_at":    datetime.utcnow().isoformat() + "Z",
            "window_start":    trading_days[0] if trading_days else None,
            "window_end":      trading_days[-1] if trading_days else None,
            "trading_days":    len(trading_days),
            "universe_size":   len(universe),
            "total_signals":   total_signals,
            "total_trades":    len(trades),
            "summary":         summary,
            "trades":          trades,
        }, f, indent=2)
    print(f"\nDetailed results saved → {output_path}")
    return trades, summary


# ─────────────────────────────────────────────────────────────────────────────
# Summary computation
# ─────────────────────────────────────────────────────────────────────────────

def avg(lst):
    lst = [x for x in lst if x is not None]
    return round(sum(lst) / len(lst), 4) if lst else None

def win_rate(lst):
    lst = [x for x in lst if x is not None]
    if not lst: return None
    return round(sum(1 for x in lst if x > 0) / len(lst) * 100, 1)

def pct_etf_wins(adv_list):
    adv_list = [x for x in adv_list if x is not None]
    if not adv_list: return None
    return round(sum(1 for x in adv_list if x > 0) / len(adv_list) * 100, 1)


def compute_summary(trades):
    """Aggregate performance statistics across all trades."""
    # Separate trades that have ETF data
    paired = [t for t in trades if t["etf_ticker"] and t["etf_return_1d"] is not None]
    stock_only = trades   # All trades have stock data by construction

    summary = {}

    for hold in HOLD_DAYS:
        sk = f"stock_return_{hold}d"
        ek = f"etf_return_{hold}d"

        s_rets = [t.get(sk) for t in stock_only if t.get(sk) is not None]
        e_rets = [t.get(ek) for t in paired      if t.get(ek) is not None]

        # ETF advantage at this hold period (net)
        spread_s = [t["spread_estimate_stock"] for t in paired if t.get(ek) is not None]
        spread_e = [t["spread_estimate_etf"]   for t in paired
                    if t.get(ek) is not None and t["spread_estimate_etf"] is not None]

        net_s = [r - sp for r, sp in zip(e_rets, spread_s)
                 if r is not None and sp is not None]
        net_e = [r - sp for r, sp in zip(e_rets, spread_e)
                 if r is not None and sp is not None] if len(spread_e) == len(e_rets) else []

        # Advantage list: net_etf - net_stock (at this hold period)
        adv_list = []
        for t in paired:
            sr = t.get(sk)
            er = t.get(ek)
            if sr is None or er is None: continue
            sp_s = t["spread_estimate_stock"]
            sp_e = t["spread_estimate_etf"]
            if sp_e is None: continue
            adv_list.append(round((er - sp_e) - (sr - sp_s), 4))

        summary[f"{hold}d"] = {
            "stock_trades":          len(s_rets),
            "etf_paired_trades":     len(e_rets),
            "avg_stock_return_pct":  avg(s_rets),
            "avg_etf_return_pct":    avg(e_rets),
            "stock_win_rate_pct":    win_rate(s_rets),
            "etf_win_rate_pct":      win_rate(e_rets),
            "etf_beat_stock_pct":    pct_etf_wins(adv_list),
            "avg_etf_advantage_pct": avg(adv_list),
        }

    # Per-ticker breakdown
    ticker_stats = defaultdict(lambda: {"stock_rets_2d": [], "etf_rets_2d": [], "scores": []})
    for t in trades:
        tk = t["ticker"]
        ticker_stats[tk]["scores"].append(t["score"])
        sr = t.get("stock_return_2d")
        er = t.get("etf_return_2d")
        if sr is not None:
            ticker_stats[tk]["stock_rets_2d"].append(sr)
        if er is not None:
            ticker_stats[tk]["etf_rets_2d"].append(er)

    top_tickers = []
    for tk, data in ticker_stats.items():
        if not data["stock_rets_2d"]: continue
        top_tickers.append({
            "ticker":           tk,
            "appearances":      len(data["scores"]),
            "avg_score":        avg(data["scores"]),
            "avg_stock_2d":     avg(data["stock_rets_2d"]),
            "avg_etf_2d":       avg(data["etf_rets_2d"]) if data["etf_rets_2d"] else None,
            "etf":              LEVERAGED_ETFS.get(tk),
        })
    top_tickers.sort(key=lambda x: x.get("avg_stock_2d") or -999, reverse=True)

    summary["top_tickers_by_2d_return"] = top_tickers[:15]

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Formatted output
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, suffix="%", width=8):
    if val is None:
        return " " * width + "  N/A"
    return f"{val:+{width}.2f}{suffix}"


def print_summary(summary, trades):
    sep  = "─" * 78
    sep2 = "═" * 78

    print("\n" + sep2)
    print("  BACKTEST RESULTS SUMMARY  (score ≥ 65 entries, 30 trading days)")
    print(sep2)

    # Hold-period table
    print(f"\n{'Hold':>6}  {'Trades':>7}  {'Paired':>7}  "
          f"{'Avg Stock':>10}  {'Avg ETF':>9}  "
          f"{'Stock WR':>9}  {'ETF WR':>7}  "
          f"{'ETF>Stock':>10}  {'ETF Adv':>8}")
    print(sep)
    for hold in HOLD_DAYS:
        s = summary.get(f"{hold}d", {})
        print(f"  {hold}d    "
              f"{s.get('stock_trades', 0):>7}  "
              f"{s.get('etf_paired_trades', 0):>7}  "
              f"{_fmt(s.get('avg_stock_return_pct'), '%', 7)}   "
              f"{_fmt(s.get('avg_etf_return_pct'), '%', 6)}  "
              f"{_fmt(s.get('stock_win_rate_pct'), '%', 6)}  "
              f"{_fmt(s.get('etf_win_rate_pct'), '%', 5)}  "
              f"{_fmt(s.get('etf_beat_stock_pct'), '%', 7)}  "
              f"{_fmt(s.get('avg_etf_advantage_pct'), '%', 6)}")

    # Top tickers
    print(f"\n{sep}")
    print("  TOP 15 TICKERS BY AVG 2-DAY RETURN (stock)")
    print(f"{'Ticker':>8}  {'Times':>5}  {'Score':>6}  "
          f"{'Stock 2d':>9}  {'ETF':>6}  {'ETF 2d':>8}")
    print(sep)
    for row in summary.get("top_tickers_by_2d_return", []):
        etf_str  = row.get("etf") or "─"
        etf_2d   = _fmt(row.get("avg_etf_2d"), "%", 6) if row.get("etf") else "   N/A"
        print(f"  {row['ticker']:>6}  "
              f"{row['appearances']:>5}  "
              f"{row.get('avg_score') or 0:>6.1f}  "
              f"{_fmt(row.get('avg_stock_2d'), '%', 7)}  "
              f"{etf_str:>6}  "
              f"{etf_2d}")

    # Key findings
    print(f"\n{sep2}")
    print("  KEY FINDINGS")
    print(sep2)
    for hold in HOLD_DAYS:
        s = summary.get(f"{hold}d", {})
        beat = s.get("etf_beat_stock_pct")
        adv  = s.get("avg_etf_advantage_pct")
        wr_s = s.get("stock_win_rate_pct")
        wr_e = s.get("etf_win_rate_pct")
        n    = s.get("etf_paired_trades", 0)
        if beat is not None and n > 0:
            verdict = "ETF BETTER" if beat > 50 else "STOCK BETTER"
            print(f"  [{hold}d hold, {n} paired trades]  "
                  f"ETF beat stock in {beat:.1f}% of trades  "
                  f"| avg edge = {adv:+.2f}%  "
                  f"| WR stock={wr_s:.1f}%, etf={wr_e:.1f}%  "
                  f"→ {verdict}")
        else:
            n_s = s.get("stock_trades", 0)
            print(f"  [{hold}d hold, {n_s} stock-only trades]  "
                  f"No ETF comparisons available for this period")

    print(sep2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    trades, summary = run_backtest()
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s  ({len(trades)} trades analysed)\n")
