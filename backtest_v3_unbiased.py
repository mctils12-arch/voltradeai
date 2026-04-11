#!/usr/bin/env python3
"""
VolTradeAI Backtest v3 — Survivorship-Bias-Free
=================================================
Uses a broad US stock universe (active + delisted) from Alpha Vantage,
randomly sampled ~1000-1500 tickers stratified by exchange, with yfinance
OHLCV data. Builds a point-in-time investable universe each year.

Same scoring/trading logic as v2 (quick_score, VRP, QQQ floor, options)
but NO intraday shorts and NO sector rotation (removed per prior analysis).

Usage:
    python backtest_v3_unbiased.py
    python backtest_v3_unbiased.py --sample-size 800
    python backtest_v3_unbiased.py --no-cache
"""

import argparse
import csv
import io
import json
import math
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
INITIAL_EQUITY = 100_000
START_DATE = "2016-01-01"
END_DATE = "2026-04-01"
SLIPPAGE_PCT = 0.0015          # 0.15% per side
OPTIONS_COST_PER_CONTRACT = 0.65
OPTIONS_MAX_EQUITY_PCT = 0.08  # Cap options at 8% of equity

CACHE_PATH = "/tmp/backtest_v3_cache.pkl"
UNIVERSE_CACHE_PATH = "/tmp/backtest_v3_universe.pkl"

# Minimum sample size for the broad universe
DEFAULT_SAMPLE_SIZE = 1200
BATCH_SIZE = 50  # yfinance download batch size

# Universe filters
MIN_DOLLAR_VOLUME = 5_000_000   # $5M daily dollar volume
MIN_PRICE = 5.0                 # $5 minimum price
MIN_TRADING_DAYS = 60           # At least 60 days of data

# Common stock asset types from Alpha Vantage
COMMON_STOCK_TYPES = {"Stock", "Common Stock"}
VALID_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "NYSE ARCA", "NYSE MKT", "BATS"}

# QQQ floor allocation by regime
QQQ_FLOOR_ALLOC = {
    "BULL":    0.70,
    "NEUTRAL": 0.90,
    "CAUTION": 0.35,
    "BEAR":    0.00,
    "PANIC":   0.00,
}

# ETFs needed for system components (always downloaded)
SYSTEM_ETFS = ["SPY", "QQQ", "^VIX", "VXX", "SVXY"]


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE BUILDING — Alpha Vantage listing status
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_listing_status(state="active"):
    """Fetch active or delisted US stocks from Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo"
    if state == "delisted":
        url += "&state=delisted"

    print(f"[UNIVERSE] Fetching {state} listings from Alpha Vantage...")
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        print(f"[UNIVERSE] Got {len(rows)} {state} listings")
        return rows
    except (URLError, Exception) as e:
        print(f"[UNIVERSE] Error fetching {state} listings: {e}")
        return []


def build_universe(sample_size=DEFAULT_SAMPLE_SIZE):
    """
    Build survivorship-bias-free universe:
    1. Fetch active + delisted tickers from Alpha Vantage
    2. Filter to common stocks on major exchanges
    3. Stratified random sample by exchange
    """
    # Check cache first
    if os.path.exists(UNIVERSE_CACHE_PATH):
        age_hours = (datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(UNIVERSE_CACHE_PATH))).total_seconds() / 3600
        if age_hours < 72:  # Cache for 3 days
            print("[UNIVERSE] Loading cached universe...")
            with open(UNIVERSE_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if len(cached.get("sampled_tickers", [])) > 0:
                print(f"[UNIVERSE] Cached: {len(cached['all_tickers'])} total, "
                      f"{len(cached['sampled_tickers'])} sampled")
                return cached

    active = fetch_listing_status("active")
    delisted = fetch_listing_status("delisted")

    all_listings = []
    seen_symbols = set()

    for row in active + delisted:
        symbol = row.get("symbol", "").strip()
        exchange = row.get("exchange", "").strip()
        asset_type = row.get("assetType", "").strip()
        name = row.get("name", "").strip()

        # Skip if no symbol
        if not symbol:
            continue

        # Skip duplicates
        if symbol in seen_symbols:
            continue

        # Filter: only common stocks on major exchanges
        if asset_type not in COMMON_STOCK_TYPES:
            continue
        if exchange not in VALID_EXCHANGES:
            continue

        # Skip warrants, units, rights, preferred (by symbol pattern)
        if any(c in symbol for c in [".", "-", "+"]):
            continue
        if len(symbol) > 5:
            continue

        seen_symbols.add(symbol)
        all_listings.append({
            "symbol": symbol,
            "exchange": exchange,
            "name": name,
            "status": row.get("status", "Active"),
            "ipoDate": row.get("ipoDate", ""),
            "delistingDate": row.get("delistingDate", ""),
        })

    print(f"[UNIVERSE] Total common stocks after filtering: {len(all_listings)}")

    # Stratified sampling by exchange
    by_exchange = defaultdict(list)
    for listing in all_listings:
        by_exchange[listing["exchange"]].append(listing)

    print("[UNIVERSE] Exchange distribution:")
    for exch, listings in sorted(by_exchange.items(), key=lambda x: -len(x[1])):
        print(f"  {exch}: {len(listings)}")

    # Sample proportionally from each exchange
    total = len(all_listings)
    sampled = []
    for exch, listings in by_exchange.items():
        n = max(1, int(sample_size * len(listings) / total))
        n = min(n, len(listings))
        sampled.extend(random.sample(listings, n))

    # If we're under target, add more from largest exchanges
    while len(sampled) < sample_size and len(sampled) < total:
        remaining = [l for l in all_listings if l not in sampled]
        if not remaining:
            break
        sampled.append(random.choice(remaining))

    sampled_tickers = [s["symbol"] for s in sampled]
    random.shuffle(sampled_tickers)

    print(f"[UNIVERSE] Sampled {len(sampled_tickers)} tickers (target: {sample_size})")

    result = {
        "all_tickers": [l["symbol"] for l in all_listings],
        "sampled_tickers": sampled_tickers,
        "listings": {l["symbol"]: l for l in all_listings},
    }

    # Cache
    with open(UNIVERSE_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD & CACHE
# ═══════════════════════════════════════════════════════════════════════════════
def download_data(tickers, use_cache=True):
    """Download OHLCV data via yfinance with aggressive pickle caching."""

    # Check cache
    if use_cache and os.path.exists(CACHE_PATH):
        age_hours = (datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(CACHE_PATH))).total_seconds() / 3600
        if age_hours < 48:
            print("[DATA] Loading from pickle cache...")
            with open(CACHE_PATH, "rb") as f:
                cached_data = pickle.load(f)
            # Check how many of our tickers are already cached
            cached_tickers = set(cached_data.keys())
            needed_tickers = set(tickers) - cached_tickers
            if len(needed_tickers) == 0:
                print(f"[DATA] All {len(tickers)} tickers found in cache")
                return cached_data
            elif len(needed_tickers) < len(tickers) * 0.3:
                print(f"[DATA] {len(cached_tickers & set(tickers))} cached, "
                      f"need {len(needed_tickers)} more")
                # Download only missing tickers
                new_data = _download_tickers(list(needed_tickers))
                cached_data.update(new_data)
                # Re-cache
                with open(CACHE_PATH, "wb") as f:
                    pickle.dump(cached_data, f)
                return cached_data
            else:
                print(f"[DATA] Cache has {len(cached_tickers & set(tickers))}/{len(tickers)} "
                      f"tickers, re-downloading all")

    # Full download
    data = _download_tickers(tickers)

    # Cache to pickle
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"[DATA] Cached {len(data)} tickers to {CACHE_PATH}")

    return data


def _download_tickers(tickers):
    """Download tickers from yfinance in batches."""
    import yfinance as yf

    data = {}
    total = len(tickers)
    start_time = time.time()

    print(f"[DATA] Downloading {total} tickers from yfinance...")

    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        elapsed = time.time() - start_time
        rate = (i / elapsed) if elapsed > 0 and i > 0 else 0
        eta = ((total - i) / rate / 60) if rate > 0 else 0

        print(f"  Batch {batch_num}/{total_batches} ({len(data)}/{total} downloaded, "
              f"ETA: {eta:.1f}min)")

        batch_str = " ".join(batch)
        try:
            df_batch = yf.download(
                batch_str,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )

            if len(batch) == 1:
                tk = batch[0]
                if df_batch is not None and not df_batch.empty:
                    df_clean = df_batch.dropna(subset=["Close"])
                    if len(df_clean) >= MIN_TRADING_DAYS:
                        data[tk] = df_clean
            else:
                for tk in batch:
                    try:
                        df_tk = df_batch[tk].dropna(subset=["Close"])
                        if len(df_tk) >= MIN_TRADING_DAYS:
                            data[tk] = df_tk
                    except (KeyError, TypeError):
                        pass  # No data for this ticker, skip silently
        except Exception as e:
            print(f"    Error downloading batch: {e}")
            # Try individual downloads as fallback
            for tk in batch:
                try:
                    df_single = yf.download(
                        tk, start=START_DATE, end=END_DATE,
                        auto_adjust=True, progress=False,
                    )
                    if df_single is not None and not df_single.empty:
                        df_clean = df_single.dropna(subset=["Close"])
                        if len(df_clean) >= MIN_TRADING_DAYS:
                            data[tk] = df_clean
                except Exception:
                    pass

    elapsed = time.time() - start_time
    print(f"[DATA] Downloaded {len(data)} tickers with valid data in {elapsed:.0f}s")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# POINT-IN-TIME UNIVERSE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════
def build_investable_universe(data, date_idx, dates, listings_info=None):
    """
    Build the investable universe for a given date.
    Filters: price > $5, daily dollar volume > $5M (20-day avg), has data.
    """
    investable = []
    date = dates[date_idx]

    for sym, df in data.items():
        # Skip system ETFs
        if sym in SYSTEM_ETFS:
            continue

        # Check if this ticker has data at this date
        if date not in df.index:
            # Check if the date is within the ticker's data range
            if date < df.index[0] or date > df.index[-1]:
                continue
            # Forward-fill: find the most recent available date
            mask = df.index <= date
            if not mask.any():
                continue

        # Get aligned data
        try:
            loc = df.index.get_indexer([date], method="ffill")[0]
            if loc < 0 or loc >= len(df):
                continue
        except Exception:
            continue

        close_val = float(df.iloc[loc]["Close"])
        vol_val = float(df.iloc[loc]["Volume"])

        # Price filter
        if close_val < MIN_PRICE:
            continue

        # Dollar volume filter (20-day average)
        start_loc = max(0, loc - 20)
        if loc - start_loc < 5:  # Need at least 5 days of data
            continue

        recent_closes = df.iloc[start_loc:loc + 1]["Close"].values.astype(float)
        recent_vols = df.iloc[start_loc:loc + 1]["Volume"].values.astype(float)
        dollar_vols = recent_closes * recent_vols
        avg_dollar_vol = np.nanmean(dollar_vols)

        if avg_dollar_vol < MIN_DOLLAR_VOLUME:
            continue

        # Check if delisted before this date
        if listings_info and sym in listings_info:
            delist_date = listings_info[sym].get("delistingDate", "")
            if delist_date:
                try:
                    delist_dt = pd.Timestamp(delist_date)
                    if date > delist_dt:
                        continue
                except Exception:
                    pass

        investable.append(sym)

    return investable


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION (matches production system_config.py — same as v2)
# ═══════════════════════════════════════════════════════════════════════════════
def get_regime(vxx_ratio, spy_vs_ma50, spy_below_200_days=0):
    """
    Market regime detection — mirrors production get_market_regime()
    from system_config.py (thresholds matched to BASE_CONFIG).
    vxx_ratio: VXX / 30-day avg (>1 = fear)
    spy_vs_ma50: SPY / 50-day MA
    spy_below_200_days: consecutive days SPY < 200d MA
    """
    if spy_below_200_days >= 10:
        if vxx_ratio >= 1.30 or spy_vs_ma50 < 0.94:
            return "PANIC"
        return "BEAR"
    if vxx_ratio >= 1.30 or spy_vs_ma50 < 0.94:
        return "PANIC"
    if vxx_ratio >= 1.15 or spy_vs_ma50 < 0.96:
        return "BEAR"
    if vxx_ratio >= 1.05:
        return "CAUTION"
    if vxx_ratio <= 0.90 and spy_vs_ma50 > 1.01:
        return "BULL"
    return "NEUTRAL"


def get_adaptive_params(regime):
    """Regime-adaptive position parameters — matched to production system_config.py."""
    params = {
        "PANIC":   {"max_pos": 0, "min_score": 75, "size_pct": 0.06, "tp": 0.12, "sl": 0.05, "hold": 3},
        "BEAR":    {"max_pos": 0, "min_score": 75, "size_pct": 0.08, "tp": 0.12, "sl": 0.05, "hold": 5},
        "CAUTION": {"max_pos": 4, "min_score": 67, "size_pct": 0.10, "tp": 0.12, "sl": 0.06, "hold": 10},
        "BULL":    {"max_pos": 8, "min_score": 63, "size_pct": 0.15, "tp": 0.12, "sl": 0.06, "hold": 10},
        "NEUTRAL": {"max_pos": 0, "min_score": 75, "size_pct": 0.12, "tp": 0.12, "sl": 0.06, "hold": 10},
    }
    return params.get(regime, params["NEUTRAL"])


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTION (identical to v2 quick_score)
# ═══════════════════════════════════════════════════════════════════════════════
def quick_score(closes, volumes, idx):
    """
    Momentum score for long candidates.
    Returns 0-100 score based on 5d/20d momentum + volume ratio.
    """
    if idx < 20:
        return 0
    c = closes[idx]
    v = volumes[idx]
    if c < 2 or v < 100_000:
        return 0

    # 5-day momentum
    pc5 = closes[idx - 5] if idx >= 5 else c
    if pc5 <= 0:
        return 0
    mom5 = (c - pc5) / pc5 * 100
    if abs(mom5) > 30:
        return 0  # Skip gapped stocks

    # 20-day momentum
    pc20 = closes[idx - 20] if idx >= 20 else c
    mom20 = (c - pc20) / pc20 * 100 if pc20 > 0 else 0

    # Volume ratio (current vs 5-day avg)
    avg_vol5 = np.mean(volumes[max(0, idx - 5):idx]) if idx >= 5 else v
    vol_ratio = v / max(avg_vol5, 1)

    # Score components
    score_mom5 = min(mom5 * 5, 35) if mom5 > 0 else 0
    score_mom20 = min(max(mom20 * 1.5, 0), 20) if mom20 > 0 else 0
    score_vol = min((vol_ratio - 1.0) * 12, 20) if vol_ratio > 1.1 else 0

    return round(min(30 + score_mom5 + score_mom20 + score_vol, 100), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS SIMULATION (same as v2)
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_options_pnl(entry_price, exit_price, hv, holding_days):
    """
    SIMULATED options P&L using Black-Scholes heuristic.
    Strategy: sell cash-secured put (CSP) at-the-money.
    """
    T = 30 / 252
    sigma = max(hv / 100, 0.10)
    atm_premium_pct = 0.4 * sigma * math.sqrt(T)
    premium_collected = atm_premium_pct * 0.9

    stock_move_pct = (exit_price - entry_price) / entry_price

    if stock_move_pct >= 0:
        theta_fraction = min(holding_days / 30, 1.0)
        pnl_pct = premium_collected * theta_fraction
    else:
        pnl_pct = stock_move_pct + premium_collected

    pnl_pct -= 2 * SLIPPAGE_PCT
    return pnl_pct


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT P&L TRACKER (same as v2)
# ═══════════════════════════════════════════════════════════════════════════════
class ComponentTracker:
    """Tracks P&L for a single component independently."""

    def __init__(self, name):
        self.name = name
        self.daily_pnl = []
        self.trades = []
        self.total_pnl = 0.0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        self.cumulative = 0.0

    def record_daily(self, date, pnl):
        self.daily_pnl.append((date, pnl))
        self.total_pnl += pnl
        self.cumulative += pnl
        equity = INITIAL_EQUITY + self.cumulative
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def record_trade(self, trade_dict):
        self.trades.append(trade_dict)

    def annual_pnl(self):
        by_year = defaultdict(float)
        for date, pnl in self.daily_pnl:
            by_year[date.year] += pnl
        return dict(by_year)

    def win_rate(self):
        if not self.trades:
            return 0
        wins = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        return wins / len(self.trades) * 100

    def avg_win(self):
        wins = [t["pnl"] for t in self.trades if t.get("pnl", 0) > 0]
        return np.mean(wins) if wins else 0

    def avg_loss(self):
        losses = [t["pnl"] for t in self.trades if t.get("pnl", 0) <= 0]
        return np.mean(losses) if losses else 0


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION MANAGER (same as v2)
# ═══════════════════════════════════════════════════════════════════════════════
class Position:
    """Represents an open position with stop/TP tracking."""

    def __init__(self, symbol, entry_price, shares, entry_date, entry_idx,
                 stop_pct, tp_pct, max_hold):
        self.symbol = symbol
        self.entry_price = entry_price
        self.shares = shares
        self.entry_date = entry_date
        self.entry_idx = entry_idx
        self.stop_pct = stop_pct
        self.tp_pct = tp_pct
        self.max_hold = max_hold
        self.highest_price = entry_price

    def check_exit(self, current_price, current_idx):
        """Check if position should be exited. Returns (should_exit, reason)."""
        days_held = current_idx - self.entry_idx
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        self.highest_price = max(self.highest_price, current_price)

        if pnl_pct <= -self.stop_pct:
            return True, "stop_loss"
        if pnl_pct >= self.tp_pct:
            return True, "take_profit"
        if days_held >= self.max_hold:
            return True, "time_stop"
        # Trailing stop
        if pnl_pct > self.stop_pct:
            trail = self.highest_price * (1 - self.stop_pct * 0.5)
            if current_price < trail:
                return True, "trailing_stop"

        return False, ""

    def calc_pnl(self, exit_price):
        """Calculate P&L including slippage."""
        raw_pnl = (exit_price - self.entry_price) * self.shares
        slippage = self.entry_price * self.shares * SLIPPAGE_PCT + exit_price * self.shares * SLIPPAGE_PCT
        return raw_pnl - slippage


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST ENGINE — Survivorship-Bias-Free
# ═══════════════════════════════════════════════════════════════════════════════
class BacktestV3Engine:
    """
    Runs VolTradeAI backtest with broad, survivorship-bias-free universe.
    Components: Stock Picking (broad universe), VRP/SVXY, QQQ Floor, Options.
    NO intraday shorts, NO sector rotation.
    """

    def __init__(self):
        self.equity = INITIAL_EQUITY
        self.peak_equity = INITIAL_EQUITY

        self.trackers = {
            "stocks":  ComponentTracker("Stock Picking (Broad)"),
            "vrp":     ComponentTracker("VRP / SVXY"),
            "qqq":     ComponentTracker("QQQ Floor"),
            "options": ComponentTracker("Options (Simulated)"),
        }

        self.stock_positions = []
        self.vrp_active = False
        self.vrp_entry_price = 0
        self.vrp_entry_idx = 0
        self.vrp_shares = 0
        self.qqq_shares = 0
        self.qqq_entry_price = 0

        self.regime_history = []
        self.spy_below_200_count = 0
        self.equity_curve = []

        # Universe tracking
        self.universe_size_history = []
        self.current_universe = []
        self.last_universe_year = None

    def run(self, data, listings_info=None):
        """Execute the full backtest."""
        spy = data.get("SPY")
        if spy is None:
            print("ERROR: No SPY data available")
            return

        dates = spy.index.tolist()
        print(f"\n[BACKTEST] Running survivorship-bias-free mode")
        print(f"[BACKTEST] Period: {dates[0].date()} to {dates[-1].date()} | {len(dates)} trading days")
        print(f"[BACKTEST] Total tickers with data: {len(data)}")

        # Pre-compute SPY arrays
        spy_close = spy["Close"].values.astype(float)
        spy_open = spy["Open"].values.astype(float)

        # VIX data
        vix_df = data.get("^VIX")
        vix_close = self._align_to_dates(vix_df, dates, "Close") if vix_df is not None else np.full(len(dates), 20.0)

        # VXX data
        vxx_df = data.get("VXX")
        vxx_close = self._align_to_dates(vxx_df, dates, "Close") if vxx_df is not None else None

        # SVXY data
        svxy_df = data.get("SVXY")
        svxy_close = self._align_to_dates(svxy_df, dates, "Close") if svxy_df is not None else None
        svxy_open = self._align_to_dates(svxy_df, dates, "Open") if svxy_df is not None else None

        # QQQ data
        qqq_df = data.get("QQQ")
        qqq_close = self._align_to_dates(qqq_df, dates, "Close") if qqq_df is not None else None
        qqq_open = self._align_to_dates(qqq_df, dates, "Open") if qqq_df is not None else None

        # Pre-compute stock data arrays for ALL tickers (aligned to SPY dates)
        print("[BACKTEST] Aligning stock data to SPY dates...")
        stock_data = {}
        align_count = 0
        for sym, df in data.items():
            if sym in SYSTEM_ETFS:
                continue
            sd = {
                "close": self._align_to_dates(df, dates, "Close"),
                "open":  self._align_to_dates(df, dates, "Open"),
                "high":  self._align_to_dates(df, dates, "High"),
                "volume": self._align_to_dates(df, dates, "Volume"),
            }
            # Only keep tickers with enough non-NaN data
            valid_count = np.sum(~np.isnan(sd["close"]))
            if valid_count >= MIN_TRADING_DAYS:
                stock_data[sym] = sd
                align_count += 1
        print(f"[BACKTEST] Aligned {align_count} tickers to trading calendar")

        # Pre-compute VXX ratio and regime for each day
        vxx_ratio_arr = np.full(len(dates), 1.0)
        regime_arr = ["NEUTRAL"] * len(dates)

        for i in range(len(dates)):
            if vxx_close is not None and not np.isnan(vxx_close[i]) and vxx_close[i] > 0:
                if i >= 30:
                    vxx_30d_avg = np.nanmean(vxx_close[max(0, i - 30):i])
                    vxx_ratio_arr[i] = vxx_close[i] / vxx_30d_avg if vxx_30d_avg > 0 else 1.0
                else:
                    vxx_ratio_arr[i] = 1.0
            else:
                if i >= 20 and not np.isnan(vix_close[i]):
                    vix_20d_avg = np.nanmean(vix_close[max(0, i - 20):i])
                    vxx_ratio_arr[i] = vix_close[i] / vix_20d_avg if vix_20d_avg > 0 else 1.0
                else:
                    vxx_ratio_arr[i] = 1.0

            if i >= 50:
                spy_ma50 = np.mean(spy_close[i - 50:i])
                spy_vs_ma50 = spy_close[i] / spy_ma50 if spy_ma50 > 0 else 1.0
            else:
                spy_vs_ma50 = 1.0

            if i >= 200:
                spy_ma200 = np.mean(spy_close[i - 200:i])
                if spy_close[i] < spy_ma200:
                    self.spy_below_200_count += 1
                else:
                    self.spy_below_200_count = 0
            else:
                self.spy_below_200_count = 0

            regime_arr[i] = get_regime(vxx_ratio_arr[i], spy_vs_ma50, self.spy_below_200_count)

        print(f"[REGIME] Distribution: " + ", ".join(
            f"{r}={sum(1 for x in regime_arr if x == r)}"
            for r in ["BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC"]
        ))

        # ───────────────────────────────────────────────────────────────
        # MAIN SIMULATION LOOP
        # ───────────────────────────────────────────────────────────────
        for i in range(50, len(dates)):
            date = dates[i]
            regime = regime_arr[i]
            self.regime_history.append((date, regime))
            daily_total_pnl = 0

            # Rebuild investable universe annually (or on first day)
            current_year = date.year
            if self.last_universe_year != current_year:
                self._rebuild_universe(i, dates, stock_data, listings_info)
                self.last_universe_year = current_year

            # ── 1. STOCK PICKING (broad universe) ────────────────────
            pnl = self._run_stocks(i, dates, regime, stock_data)
            self.trackers["stocks"].record_daily(date, pnl)
            daily_total_pnl += pnl

            # ── 2. VRP / SVXY ────────────────────────────────────────
            pnl = self._run_vrp(i, dates, regime, vxx_ratio_arr, vix_close,
                                svxy_close, svxy_open, vxx_close)
            self.trackers["vrp"].record_daily(date, pnl)
            daily_total_pnl += pnl

            # ── 3. QQQ FLOOR ─────────────────────────────────────────
            pnl = self._run_qqq_floor(i, dates, regime, qqq_close, qqq_open)
            self.trackers["qqq"].record_daily(date, pnl)
            daily_total_pnl += pnl

            # ── 4. OPTIONS (SIMULATED) ───────────────────────────────
            pnl = self._run_options(i, dates, regime, stock_data, vix_close)
            self.trackers["options"].record_daily(date, pnl)
            daily_total_pnl += pnl

            # Update total equity
            self.equity += daily_total_pnl
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            self.equity_curve.append((date, self.equity))

        self.regime_arr = regime_arr
        self.dates = dates

    def _rebuild_universe(self, idx, dates, stock_data, listings_info):
        """Rebuild the investable universe based on liquidity/price filters."""
        date = dates[idx]
        universe = []

        for sym, sd in stock_data.items():
            close_val = sd["close"][idx]
            vol_val = sd["volume"][idx]

            if np.isnan(close_val) or close_val <= 0:
                continue
            if close_val < MIN_PRICE:
                continue

            # 20-day average dollar volume
            start_i = max(0, idx - 20)
            closes_window = sd["close"][start_i:idx + 1]
            vols_window = sd["volume"][start_i:idx + 1]
            valid_mask = ~np.isnan(closes_window) & ~np.isnan(vols_window)
            if np.sum(valid_mask) < 5:
                continue
            dollar_vols = closes_window[valid_mask] * vols_window[valid_mask]
            avg_dv = np.mean(dollar_vols)
            if avg_dv < MIN_DOLLAR_VOLUME:
                continue

            # Check delist date
            if listings_info and sym in listings_info:
                delist_date = listings_info[sym].get("delistingDate", "")
                if delist_date:
                    try:
                        delist_dt = pd.Timestamp(delist_date)
                        if date > delist_dt:
                            continue
                    except Exception:
                        pass

            universe.append(sym)

        self.current_universe = universe
        self.universe_size_history.append((date, len(universe)))
        print(f"[UNIVERSE] {date.date()}: {len(universe)} investable stocks "
              f"(price>${MIN_PRICE}, dvol>${MIN_DOLLAR_VOLUME/1e6:.0f}M)")

    def _align_to_dates(self, df, dates, col):
        """Align a DataFrame column to the master date index, forward-filling gaps."""
        if df is None:
            return np.full(len(dates), np.nan)
        arr = np.full(len(dates), np.nan)
        for i, d in enumerate(dates):
            if d in df.index:
                val = df.loc[d, col]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                arr[i] = float(val)
        # Forward-fill NaN
        for i in range(1, len(arr)):
            if np.isnan(arr[i]):
                arr[i] = arr[i - 1]
        return arr

    # ─── COMPONENT: Stock Picking (Broad Universe) ──────────────────
    def _run_stocks(self, idx, dates, regime, stock_data):
        """Run stock picking against the full investable universe."""
        pnl = 0.0
        params = get_adaptive_params(regime)
        date = dates[idx]

        # Check exits on existing positions
        exits_to_process = []
        for pos in self.stock_positions:
            sd = stock_data.get(pos.symbol)
            if sd is None:
                continue
            current_price = sd["close"][idx]
            if np.isnan(current_price) or current_price <= 0:
                continue
            should_exit, reason = pos.check_exit(current_price, idx)
            if should_exit:
                if idx + 1 < len(dates):
                    exit_price = sd["open"][idx + 1]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = current_price
                else:
                    exit_price = current_price
                trade_pnl = pos.calc_pnl(exit_price)
                pnl += trade_pnl
                self.trackers["stocks"].record_trade({
                    "symbol": pos.symbol, "side": "long",
                    "entry": pos.entry_price, "exit": exit_price,
                    "pnl": trade_pnl, "reason": reason,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(date.date()),
                    "days_held": idx - pos.entry_idx,
                })
                exits_to_process.append(pos)

        for pos in exits_to_process:
            self.stock_positions.remove(pos)

        # New entries — score ALL stocks in current universe
        if params["max_pos"] > 0 and len(self.stock_positions) < params["max_pos"]:
            candidates = []
            held_syms = {p.symbol for p in self.stock_positions}

            for sym in self.current_universe:
                if sym in held_syms:
                    continue
                sd = stock_data.get(sym)
                if sd is None:
                    continue
                closes = sd["close"]
                vols = sd["volume"]
                if np.isnan(closes[idx]) or closes[idx] <= 0:
                    continue
                score = quick_score(closes, vols, idx)
                if score >= params["min_score"]:
                    candidates.append((sym, score))

            # Sort by score descending, take top candidates
            candidates.sort(key=lambda x: -x[1])
            slots = params["max_pos"] - len(self.stock_positions)

            for sym, score in candidates[:slots]:
                sd = stock_data.get(sym)
                if idx + 1 < len(dates):
                    entry_price = sd["open"][idx + 1]
                    if np.isnan(entry_price) or entry_price <= 0:
                        continue
                else:
                    continue

                alloc = self.equity * params["size_pct"]
                shares = int(alloc / entry_price)
                if shares <= 0:
                    continue

                entry_price_slipped = entry_price * (1 + SLIPPAGE_PCT)

                pos = Position(
                    symbol=sym,
                    entry_price=entry_price_slipped, shares=shares,
                    entry_date=date, entry_idx=idx,
                    stop_pct=params["sl"], tp_pct=params["tp"],
                    max_hold=params["hold"],
                )
                self.stock_positions.append(pos)

        return pnl

    # ─── COMPONENT: VRP / SVXY (same as v2) ─────────────────────────
    def _run_vrp(self, idx, dates, regime, vxx_ratio_arr, vix_close,
                 svxy_close, svxy_open, vxx_close):
        """VRP harvest: buy SVXY when VXX elevated+declining, with circuit breaker."""
        pnl = 0.0
        vxx_ratio = vxx_ratio_arr[idx]
        current_vix = vix_close[idx] if not np.isnan(vix_close[idx]) else 20

        # Circuit Breaker: no VRP when VIX > 40
        if current_vix > 40:
            if self.vrp_active and svxy_close is not None:
                exit_price = svxy_close[idx]
                if not np.isnan(exit_price) and exit_price > 0:
                    raw = (exit_price - self.vrp_entry_price) * self.vrp_shares
                    slippage = (self.vrp_entry_price + exit_price) * self.vrp_shares * SLIPPAGE_PCT
                    trade_pnl = raw - slippage
                    pnl += trade_pnl
                    self.trackers["vrp"].record_trade({
                        "type": "vrp_exit", "reason": "circuit_breaker_vix>40",
                        "pnl": trade_pnl,
                        "days_held": idx - self.vrp_entry_idx,
                    })
                self.vrp_active = False
            return pnl

        # Exit: regime recovered
        if self.vrp_active and regime in ("NEUTRAL", "BULL"):
            if svxy_close is not None and not np.isnan(svxy_close[idx]):
                exit_price = svxy_close[idx]
                if idx + 1 < len(dates) and svxy_open is not None and not np.isnan(svxy_open[idx + 1]):
                    exit_price = svxy_open[idx + 1]
                raw = (exit_price - self.vrp_entry_price) * self.vrp_shares
                slippage = (self.vrp_entry_price + exit_price) * self.vrp_shares * SLIPPAGE_PCT
                trade_pnl = raw - slippage
                pnl += trade_pnl
                self.trackers["vrp"].record_trade({
                    "type": "vrp_exit", "reason": "regime_recovered",
                    "pnl": trade_pnl,
                    "days_held": idx - self.vrp_entry_idx,
                })
            self.vrp_active = False

        # Stop-loss: SVXY drops >15% from entry
        if self.vrp_active and svxy_close is not None:
            if not np.isnan(svxy_close[idx]) and svxy_close[idx] > 0:
                vrp_return = (svxy_close[idx] - self.vrp_entry_price) / self.vrp_entry_price
                if vrp_return < -0.15:
                    if idx + 1 < len(dates) and svxy_open is not None and not np.isnan(svxy_open[idx + 1]):
                        exit_price = svxy_open[idx + 1]
                    else:
                        exit_price = svxy_close[idx]
                    raw = (exit_price - self.vrp_entry_price) * self.vrp_shares
                    slippage = (self.vrp_entry_price + exit_price) * self.vrp_shares * SLIPPAGE_PCT
                    trade_pnl = raw - slippage
                    pnl += trade_pnl
                    self.trackers["vrp"].record_trade({
                        "type": "vrp_exit", "reason": "stop_loss",
                        "pnl": trade_pnl,
                        "days_held": idx - self.vrp_entry_idx,
                    })
                    self.vrp_active = False

        # Entry: VXX ratio 1.05-1.25 AND VXX declining (5-day)
        if not self.vrp_active and 1.05 <= vxx_ratio <= 1.25:
            vxx_declining = False
            if vxx_close is not None and idx >= 5:
                if (not np.isnan(vxx_close[idx]) and not np.isnan(vxx_close[idx - 5]) and
                        vxx_close[idx - 5] > 0):
                    vxx_declining = vxx_close[idx] < vxx_close[idx - 5]
            elif idx >= 5:
                if not np.isnan(vix_close[idx]) and not np.isnan(vix_close[idx - 5]):
                    vxx_declining = vix_close[idx] < vix_close[idx - 5]

            if vxx_declining:
                if current_vix >= 25:
                    vrp_alloc_pct = 0.30
                else:
                    vrp_alloc_pct = 0.15

                if svxy_close is not None and not np.isnan(svxy_close[idx]) and svxy_close[idx] > 0:
                    if idx + 1 < len(dates) and svxy_open is not None and not np.isnan(svxy_open[idx + 1]):
                        entry_price = svxy_open[idx + 1]
                    else:
                        entry_price = svxy_close[idx]

                    if not np.isnan(entry_price) and entry_price > 0:
                        alloc = self.equity * vrp_alloc_pct
                        shares = int(alloc / entry_price)
                        if shares > 0:
                            self.vrp_active = True
                            self.vrp_entry_price = entry_price * (1 + SLIPPAGE_PCT)
                            self.vrp_entry_idx = idx
                            self.vrp_shares = shares

        # Approximate VRP returns when no SVXY data
        if self.vrp_active and (svxy_close is None or np.isnan(svxy_close[idx])):
            if idx >= 1 and not np.isnan(vix_close[idx]) and not np.isnan(vix_close[idx - 1]):
                vix_daily_ret = (vix_close[idx] - vix_close[idx - 1]) / vix_close[idx - 1]
                approx_svxy_ret = -0.5 * vix_daily_ret
                vrp_daily_pnl = self.vrp_entry_price * self.vrp_shares * approx_svxy_ret
                pnl += vrp_daily_pnl

        return pnl

    # ─── COMPONENT: QQQ Floor (same as v2) ──────────────────────────
    def _run_qqq_floor(self, idx, dates, regime, qqq_close, qqq_open):
        """Passive QQQ allocation by regime."""
        if qqq_close is None:
            return 0.0

        pnl = 0.0
        target_alloc = QQQ_FLOOR_ALLOC.get(regime, 0.0)
        current_qqq_price = qqq_close[idx]
        if np.isnan(current_qqq_price) or current_qqq_price <= 0:
            return 0.0

        target_value = self.equity * target_alloc
        target_shares = int(target_value / current_qqq_price)

        current_value = self.qqq_shares * current_qqq_price
        if abs(current_value - target_value) > self.equity * 0.05 or (target_shares == 0 and self.qqq_shares > 0):
            delta_shares = target_shares - self.qqq_shares
            if delta_shares != 0:
                if idx + 1 < len(dates) and qqq_open is not None and not np.isnan(qqq_open[idx + 1]):
                    exec_price = qqq_open[idx + 1]
                else:
                    exec_price = current_qqq_price
                if not np.isnan(exec_price) and exec_price > 0:
                    slippage_cost = abs(delta_shares) * exec_price * SLIPPAGE_PCT
                    pnl -= slippage_cost
                    self.qqq_shares = target_shares
                    self.qqq_entry_price = exec_price

        if self.qqq_shares > 0 and idx > 0:
            yest_price = qqq_close[idx - 1]
            if not np.isnan(yest_price) and yest_price > 0:
                daily_ret = (current_qqq_price - yest_price) / yest_price
                pnl += self.qqq_shares * yest_price * daily_ret

        return pnl

    # ─── COMPONENT: Options (same as v2 but uses broad universe) ────
    def _run_options(self, idx, dates, regime, stock_data, vix_close):
        """SIMULATED options P&L — CSP on strong-scoring stocks from broad universe."""
        pnl = 0.0
        current_vix = vix_close[idx] if not np.isnan(vix_close[idx]) else 20
        if current_vix < 18:
            return 0.0

        options_alloc = self.equity * OPTIONS_MAX_EQUITY_PCT

        # Pick top-scoring stock from current universe for CSP
        best_sym = None
        best_score = 0
        # Sample up to 50 from universe for options scoring (performance)
        universe_sample = self.current_universe[:50] if len(self.current_universe) > 50 else self.current_universe

        for sym in universe_sample:
            sd = stock_data.get(sym)
            if sd is None:
                continue
            closes = sd["close"]
            vols = sd["volume"]
            if np.isnan(closes[idx]) or closes[idx] <= 0:
                continue
            score = quick_score(closes, vols, idx)
            if score > best_score:
                best_score = score
                best_sym = sym

        if best_sym is None or best_score < 50:
            return 0.0

        sd = stock_data.get(best_sym)
        entry_price = sd["close"][idx]

        hold_days = 5
        if idx + hold_days < len(dates):
            exit_price = sd["close"][idx + hold_days]
            if np.isnan(exit_price) or exit_price <= 0:
                return 0.0

            if idx >= 20:
                rets = np.diff(sd["close"][idx - 20:idx + 1]) / sd["close"][idx - 20:idx]
                rets = rets[~np.isnan(rets)]
                hv = float(np.std(rets) * np.sqrt(252) * 100) if len(rets) > 5 else current_vix
            else:
                hv = current_vix

            options_return = simulate_options_pnl(entry_price, exit_price, hv, hold_days)
            trade_pnl = options_alloc * options_return

            if idx % 5 == 0:
                pnl += trade_pnl
                if abs(trade_pnl) > 0:
                    self.trackers["options"].record_trade({
                        "symbol": best_sym,
                        "strategy": "CSP_simulated",
                        "pnl": trade_pnl,
                        "hold_days": hold_days,
                        "iv": round(hv, 1),
                    })

        return pnl

    # ─── REPORTING ───────────────────────────────────────────────────
    def generate_report(self, data):
        """Generate console report and JSON output."""
        spy_close = data["SPY"]["Close"].values.astype(float)
        dates = self.dates

        spy_start = spy_close[50]
        spy_end = spy_close[-1]
        spy_total_ret = (spy_end - spy_start) / spy_start
        num_years = len(dates) / 252

        total_pnl = sum(t.total_pnl for t in self.trackers.values())
        final_equity = INITIAL_EQUITY + total_pnl
        total_ret = total_pnl / INITIAL_EQUITY
        sys_cagr = (final_equity / INITIAL_EQUITY) ** (1 / num_years) - 1 if final_equity > 0 else 0
        spy_cagr = (1 + spy_total_ret) ** (1 / num_years) - 1

        peak = INITIAL_EQUITY
        max_dd = 0
        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        daily_returns = []
        for i in range(1, len(self.equity_curve)):
            prev_eq = self.equity_curve[i - 1][1]
            curr_eq = self.equity_curve[i][1]
            if prev_eq > 0:
                daily_returns.append((curr_eq - prev_eq) / prev_eq)

        daily_returns = np.array(daily_returns)
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            downside = daily_returns[daily_returns < 0]
            sortino = np.mean(daily_returns) / np.std(downside) * np.sqrt(252) if len(downside) > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append("  VolTradeAI Backtest v3 — SURVIVORSHIP-BIAS-FREE")
        lines.append(f"  Period: {dates[50].date()} to {dates[-1].date()} ({len(dates) - 50} trading days, {num_years:.1f} years)")
        lines.append(f"  Universe: ~{len(self.current_universe)} investable stocks (from broad sample)")
        lines.append("  Components: Stock Picking (broad), VRP/SVXY, QQQ Floor, Options (sim)")
        lines.append("  Removed: Intraday Shorts, Sector Rotation")
        lines.append("=" * 80)
        lines.append("")
        lines.append("  OVERALL PERFORMANCE")
        lines.append("  " + "-" * 40)
        lines.append(f"  Starting Equity:   ${INITIAL_EQUITY:>12,.0f}")
        lines.append(f"  Final Equity:      ${final_equity:>12,.0f}")
        lines.append(f"  Total Return:      {total_ret * 100:>11.1f}%")
        lines.append(f"  CAGR:              {sys_cagr * 100:>11.1f}%")
        lines.append(f"  Max Drawdown:      {max_dd * 100:>11.1f}%")
        lines.append(f"  Sharpe Ratio:      {sharpe:>11.2f}")
        lines.append(f"  Sortino Ratio:     {sortino:>11.2f}")
        lines.append(f"  SPY CAGR:          {spy_cagr * 100:>11.1f}%")
        lines.append(f"  SPY Total Return:  {spy_total_ret * 100:>11.1f}%")
        lines.append("")

        lines.append("  COMPONENT P&L ATTRIBUTION")
        lines.append("  " + "-" * 76)
        lines.append(f"  {'Component':<25} {'Total P&L':>12} {'% of Total':>10} {'Trades':>8} {'Win Rate':>10} {'Max DD':>10}")
        lines.append("  " + "-" * 76)

        for key, tracker in self.trackers.items():
            pct_of_total = (tracker.total_pnl / total_pnl * 100) if total_pnl != 0 else 0
            simulated_tag = " *" if key == "options" else ""
            lines.append(
                f"  {tracker.name + simulated_tag:<25} "
                f"${tracker.total_pnl:>11,.0f} "
                f"{pct_of_total:>9.1f}% "
                f"{len(tracker.trades):>7} "
                f"{tracker.win_rate():>9.1f}% "
                f"{tracker.max_drawdown * 100:>9.1f}%"
            )

        lines.append("  " + "-" * 76)
        lines.append(f"  {'TOTAL':<25} ${total_pnl:>11,.0f} {'100.0%':>10}")
        lines.append("")
        lines.append("  * Options P&L is SIMULATED using Black-Scholes heuristic (no real chain data)")
        lines.append("")

        # Universe size over time
        lines.append("  INVESTABLE UNIVERSE SIZE (annual)")
        lines.append("  " + "-" * 40)
        for date, size in self.universe_size_history:
            lines.append(f"  {date.date()}: {size:>5} stocks")
        lines.append("")

        # Annual breakdown
        lines.append("  ANNUAL RETURNS BY COMPONENT")
        lines.append("  " + "-" * 90)

        all_years = sorted(set(d.year for d, _ in self.equity_curve))

        active = list(self.trackers.items())
        header = f"  {'Year':<6}"
        for _, tracker in active:
            short_name = tracker.name[:12]
            header += f" {short_name:>12}"
        header += f" {'Total':>12} {'SPY':>12}"
        lines.append(header)
        lines.append("  " + "-" * 90)

        for year in all_years:
            row = f"  {year:<6}"
            year_total = 0
            for key, tracker in active:
                annual = tracker.annual_pnl()
                yr_pnl = annual.get(year, 0)
                year_total += yr_pnl
                row += f" ${yr_pnl:>11,.0f}"
            row += f" ${year_total:>11,.0f}"

            spy_year_dates = [d for d in dates if d.year == year]
            if len(spy_year_dates) >= 2:
                sy_start_idx = dates.index(spy_year_dates[0])
                sy_end_idx = dates.index(spy_year_dates[-1])
                spy_yr_ret = (spy_close[sy_end_idx] - spy_close[sy_start_idx]) / spy_close[sy_start_idx] * 100
                row += f"  {spy_yr_ret:>10.1f}%"
            else:
                row += f"  {'N/A':>10}"
            lines.append(row)

        lines.append("")

        # Regime distribution
        lines.append("  REGIME DISTRIBUTION")
        lines.append("  " + "-" * 40)
        regime_counts = defaultdict(int)
        for _, r in self.regime_history:
            regime_counts[r] += 1
        total_days = len(self.regime_history)
        for regime in ["BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC"]:
            count = regime_counts.get(regime, 0)
            pct = count / total_days * 100 if total_days > 0 else 0
            lines.append(f"  {regime:<12} {count:>5} days  ({pct:>5.1f}%)")
        lines.append("")

        report = "\n".join(lines)
        print(report)

        # JSON Output
        results = {
            "version": "v3_survivorship_bias_free",
            "period": f"{dates[50].date()} to {dates[-1].date()}",
            "trading_days": len(dates) - 50,
            "years": round(num_years, 1),
            "universe_size": len(self.current_universe),
            "overall": {
                "starting_equity": INITIAL_EQUITY,
                "final_equity": round(final_equity, 2),
                "total_return_pct": round(total_ret * 100, 2),
                "cagr_pct": round(sys_cagr * 100, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "sharpe": round(sharpe, 3),
                "sortino": round(sortino, 3),
                "spy_cagr_pct": round(spy_cagr * 100, 2),
                "spy_total_return_pct": round(spy_total_ret * 100, 2),
            },
            "components": {},
            "annual": {},
            "regime_distribution": {r: regime_counts.get(r, 0) for r in ["BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC"]},
            "universe_history": [(str(d.date()), s) for d, s in self.universe_size_history],
        }

        for key, tracker in self.trackers.items():
            results["components"][key] = {
                "name": tracker.name,
                "total_pnl": round(tracker.total_pnl, 2),
                "pct_of_total": round(tracker.total_pnl / total_pnl * 100, 2) if total_pnl != 0 else 0,
                "trade_count": len(tracker.trades),
                "win_rate_pct": round(tracker.win_rate(), 2),
                "avg_win": round(tracker.avg_win(), 2),
                "avg_loss": round(tracker.avg_loss(), 2),
                "max_drawdown_pct": round(tracker.max_drawdown * 100, 2),
                "simulated": key == "options",
            }

        for year in all_years:
            year_data = {}
            for key, tracker in self.trackers.items():
                annual = tracker.annual_pnl()
                year_data[key] = round(annual.get(year, 0), 2)
            year_data["total"] = round(sum(year_data.values()), 2)
            results["annual"][str(year)] = year_data

        return report, results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="VolTradeAI Backtest v3 — Survivorship-Bias-Free"
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Number of tickers to sample (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download of all data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("  VolTradeAI Backtest v3 — Survivorship-Bias-Free")
    print("=" * 80)
    print(f"  Sample size: {args.sample_size}")
    print(f"  Random seed: {args.seed}")
    print(f"  Cache: {'disabled' if args.no_cache else 'enabled'}")
    print("")

    # Step 1: Build broad universe
    universe = build_universe(sample_size=args.sample_size)
    sampled_tickers = universe["sampled_tickers"]
    listings_info = universe.get("listings", {})

    # Step 2: Download data (sampled tickers + system ETFs)
    all_download = list(set(sampled_tickers + SYSTEM_ETFS))
    data = download_data(all_download, use_cache=not args.no_cache)

    if "SPY" not in data:
        print("FATAL: Could not download SPY data")
        sys.exit(1)

    # Step 3: Run backtest
    engine = BacktestV3Engine()
    engine.run(data, listings_info=listings_info)

    # Step 4: Generate report
    report, results = engine.generate_report(data)

    # Save JSON results
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "backtest_v3_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return report, results


if __name__ == "__main__":
    main()
