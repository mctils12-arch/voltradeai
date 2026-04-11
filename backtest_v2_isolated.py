#!/usr/bin/env python3
"""
VolTradeAI Backtest v2 — Isolated Component P&L Attribution
============================================================
Self-contained backtest that mirrors v1.0.30+ production logic.
Each component (stocks, VRP, sector rotation, QQQ floor, shorts, options)
is tracked independently with its own P&L.

Usage:
    python backtest_v2_isolated.py --all
    python backtest_v2_isolated.py --stocks-only
    python backtest_v2_isolated.py --vrp-only
    python backtest_v2_isolated.py --etf-only
    python backtest_v2_isolated.py --shorts-only
    python backtest_v2_isolated.py --options-only
    python backtest_v2_isolated.py --no-options
"""

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

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

CACHE_PATH = "/tmp/backtest_yf_cache.json"

# Stock universe — 20 stocks with full 2016+ history
STOCK_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "AMZN", "INTC", "CRM",
    # Finance
    "JPM", "BAC", "GS", "V", "MA",
    # High-vol
    "MSTR", "TSLA",
    # Stable
    "XOM", "JNJ", "WMT",
    # ETFs used as stocks
    "QQQ", "IWM",
]

# Sector ETFs for rotation testing
SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLC"]

# All tickers we need to download
ALL_TICKERS = (
    STOCK_UNIVERSE + SECTOR_ETFS +
    ["SPY", "^VIX", "VXX", "SVXY", "GLD", "ITA"]
)

# QQQ floor allocation by regime (from system_config.py)
QQQ_FLOOR_ALLOC = {
    "BULL":    0.70,
    "NEUTRAL": 0.90,
    "CAUTION": 0.35,
    "BEAR":    0.00,
    "PANIC":   0.00,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD & CACHE
# ═══════════════════════════════════════════════════════════════════════════════
def download_data():
    """Download all market data via yfinance, with disk cache."""
    if os.path.exists(CACHE_PATH):
        age_hours = (os.time() if False else
                     (datetime.now() - datetime.fromtimestamp(os.path.getmtime(CACHE_PATH))).total_seconds() / 3600)
        if age_hours < 24:
            print("[DATA] Loading from cache...")
            with open(CACHE_PATH, "r") as f:
                raw = json.load(f)
            # Convert back to DataFrames
            data = {}
            for tk, records in raw.items():
                if records:
                    df = pd.DataFrame(records)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    data[tk] = df
            print(f"[DATA] Loaded {len(data)} tickers from cache")
            return data

    import yfinance as yf

    # Deduplicate tickers
    tickers = list(set(ALL_TICKERS))
    print(f"[DATA] Downloading {len(tickers)} tickers from yfinance...")

    data = {}
    # Download in batches to avoid timeouts
    batch_size = 10
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = " ".join(batch)
        print(f"  Batch {i // batch_size + 1}: {batch_str}")
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
                # Single ticker: columns are just Open/High/Low/Close/Volume
                tk = batch[0]
                if not df_batch.empty:
                    df_batch = df_batch.dropna(subset=["Close"])
                    data[tk] = df_batch
            else:
                for tk in batch:
                    try:
                        df_tk = df_batch[tk].dropna(subset=["Close"])
                        if not df_tk.empty:
                            data[tk] = df_tk
                    except (KeyError, TypeError):
                        print(f"    Warning: No data for {tk}")
        except Exception as e:
            print(f"    Error downloading batch: {e}")

    # Cache to disk
    cache_data = {}
    for tk, df in data.items():
        df_reset = df.reset_index()
        df_reset["Date"] = df_reset["Date"].astype(str)
        cache_data[tk] = df_reset.to_dict(orient="records")

    with open(CACHE_PATH, "w") as f:
        json.dump(cache_data, f)
    print(f"[DATA] Cached {len(data)} tickers to {CACHE_PATH}")

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION (matches production system_config.py)
# ═══════════════════════════════════════════════════════════════════════════════
def get_regime(vxx_ratio, spy_vs_ma50, spy_below_200_days=0):
    """
    Market regime detection — mirrors production get_market_regime().
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
    if vxx_ratio >= 1.15:
        return "BEAR"
    if vxx_ratio >= 1.05:
        return "CAUTION"
    if vxx_ratio <= 0.90:
        return "BULL"
    return "NEUTRAL"


def get_adaptive_params(regime):
    """Regime-adaptive position parameters for stock picking."""
    params = {
        "PANIC":   {"max_pos": 0, "min_score": 99, "size_pct": 0.06, "tp": 0.12, "sl": 0.06, "hold": 10},
        "BEAR":    {"max_pos": 0, "min_score": 99, "size_pct": 0.08, "tp": 0.12, "sl": 0.06, "hold": 10},
        "CAUTION": {"max_pos": 4, "min_score": 67, "size_pct": 0.10, "tp": 0.12, "sl": 0.06, "hold": 10},
        "BULL":    {"max_pos": 8, "min_score": 63, "size_pct": 0.15, "tp": 0.12, "sl": 0.06, "hold": 10},
        "NEUTRAL": {"max_pos": 0, "min_score": 99, "size_pct": 0.12, "tp": 0.12, "sl": 0.06, "hold": 10},
    }
    return params.get(regime, params["NEUTRAL"])


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
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


def score_short(closes, highs, volumes, idx, spy_ret_10d):
    """
    Short candidate scoring — 6 signals combined.
    Returns dict with score, active signal count, and ATR%, or None.
    """
    if idx < 25:
        return None
    c = closes[idx]
    if c <= 0:
        return None

    # Dollar volume filter
    dollar_vol = c * volumes[idx]
    if dollar_vol < 50_000_000:
        return None

    # Calculate ATR (14-bar)
    trs = []
    for i in range(max(1, idx - 14), idx + 1):
        h = highs[i]
        l_val = closes[i] - (highs[i] - closes[i])  # Approximate low
        pc = closes[i - 1] if i > 0 else c
        if pc > 0:
            trs.append(max(h - l_val, abs(h - pc), abs(l_val - pc)))
    if len(trs) < 10:
        return None
    atr = float(np.mean(trs[-14:]))
    if atr <= 0:
        return None
    atr_pct = atr / c * 100

    signals = {}

    # 1. Momentum collapse (5-day)
    if idx >= 5 and closes[idx - 5] > 0:
        mom = (c - closes[idx - 5]) / closes[idx - 5] * 100
        signals["momentum"] = float(max(0, min(1, -mom / (atr_pct * 2))))
    else:
        signals["momentum"] = 0.0

    # 2. Failed breakout (20-day high)
    if idx >= 20:
        h_roll = max(highs[idx - 20:idx + 1])
        h_recent = max(highs[idx - 5:idx + 1])
        drop = (c - h_recent) / h_recent * 100 if h_recent > 0 else 0
        near_high = h_recent >= h_roll * 0.97
        signals["failed_breakout"] = float(max(0, min(1, -drop / atr_pct))) if near_high else 0.0
    else:
        signals["failed_breakout"] = 0.0

    # 3. Volume distribution (10-day)
    if idx >= 10:
        avg_v = np.mean(volumes[idx - 10:idx])
        vr = volumes[idx] / max(avg_v, 1)
        day_ret = (c - closes[idx - 1]) / closes[idx - 1] * 100 if closes[idx - 1] > 0 else 0
        signals["distribution"] = float(max(0, min(1, (vr - 1) * 0.5))) if day_ret < -(atr_pct * 0.3) else 0.0
    else:
        signals["distribution"] = 0.0

    # 4. Relative weakness vs SPY (10-day)
    if idx >= 10 and closes[idx - 10] > 0:
        stock_ret = (c - closes[idx - 10]) / closes[idx - 10] * 100
        relative = stock_ret - spy_ret_10d
        signals["rel_weakness"] = float(max(0, min(1, -relative / (atr_pct * 3))))
    else:
        signals["rel_weakness"] = 0.0

    # 5. Trend breakdown (10/20 MA)
    if idx >= 20:
        ma_s = np.mean(closes[idx - 10:idx + 1])
        ma_l = np.mean(closes[idx - 20:idx + 1])
        below = (c < ma_s) + (c < ma_l)
        pct_below = (c - ma_l) / ma_l * 100 if ma_l > 0 else 0
        signals["trend_break"] = float(max(0, min(1, below * 0.3 + max(0, -pct_below / atr_pct) * 0.4)))
    else:
        signals["trend_break"] = 0.0

    # 6. Gap down
    if idx >= 1:
        prev_c = closes[idx - 1]
        # Use open if available, otherwise approximate
        gap = (closes[idx] - prev_c) / prev_c * 100 if prev_c > 0 else 0
        gap_threshold = -(atr_pct * 0.2)
        signals["gap_down"] = float(max(0, min(1, -gap / atr_pct))) if gap < gap_threshold else 0.0
    else:
        signals["gap_down"] = 0.0

    # Composite score
    raw_score = sum(signals.values()) / 6.0
    noise_floor = min(0.4, atr_pct / 15)
    active_count = sum(1 for v in signals.values() if v > noise_floor)

    score = round(raw_score * 100, 1)
    if active_count < 2 or score < 15:
        return None

    return {"score": score, "active": active_count, "atr_pct": atr_pct}


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS SIMULATION (clearly labeled as simulated)
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_options_pnl(entry_price, exit_price, hv, holding_days):
    """
    SIMULATED options P&L using Black-Scholes heuristic.
    Strategy: sell cash-secured put (CSP) at-the-money.
    Returns estimated P&L as a fraction of allocated capital.
    """
    T = 30 / 252  # 30-day option
    sigma = max(hv / 100, 0.10)

    # Simplified ATM premium: 0.4 * S * sigma * sqrt(T)
    atm_premium_pct = 0.4 * sigma * math.sqrt(T)
    premium_collected = atm_premium_pct * 0.9  # 90% of theoretical (bid-ask)

    stock_move_pct = (exit_price - entry_price) / entry_price

    # CSP: collect premium, risk downside
    if stock_move_pct >= 0:
        # Stock stayed above strike — keep full premium
        theta_fraction = min(holding_days / 30, 1.0)
        pnl_pct = premium_collected * theta_fraction
    else:
        # Stock fell — loss is (strike - exit) minus premium
        pnl_pct = stock_move_pct + premium_collected

    # Apply friction
    pnl_pct -= 2 * SLIPPAGE_PCT
    return pnl_pct


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT P&L TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
class ComponentTracker:
    """Tracks P&L for a single component independently."""

    def __init__(self, name):
        self.name = name
        self.daily_pnl = []       # (date, pnl_dollars)
        self.trades = []          # list of completed trade dicts
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
        """Return dict of year -> total P&L."""
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
# POSITION MANAGER (for stock picking and shorts)
# ═══════════════════════════════════════════════════════════════════════════════
class Position:
    """Represents an open position with stop/TP tracking."""

    def __init__(self, symbol, side, entry_price, shares, entry_date, entry_idx,
                 stop_pct, tp_pct, max_hold, component):
        self.symbol = symbol
        self.side = side              # "long" or "short"
        self.entry_price = entry_price
        self.shares = shares
        self.entry_date = entry_date
        self.entry_idx = entry_idx
        self.stop_pct = stop_pct
        self.tp_pct = tp_pct
        self.max_hold = max_hold
        self.component = component    # "stocks" or "shorts"
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.partial_exits = 0        # For scale-out tracking

    def check_exit(self, current_price, current_idx, current_date):
        """Check if position should be exited. Returns (should_exit, reason)."""
        days_held = current_idx - self.entry_idx

        if self.side == "long":
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            self.highest_price = max(self.highest_price, current_price)

            # Stop loss
            if pnl_pct <= -self.stop_pct:
                return True, "stop_loss"
            # Take profit
            if pnl_pct >= self.tp_pct:
                return True, "take_profit"
            # Time stop
            if days_held >= self.max_hold:
                return True, "time_stop"
            # Trailing stop (after 1R profit, trail at 50% of gains)
            if pnl_pct > self.stop_pct:
                trail = self.highest_price * (1 - self.stop_pct * 0.5)
                if current_price < trail:
                    return True, "trailing_stop"
        else:
            # Short position
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            self.lowest_price = min(self.lowest_price, current_price)

            if pnl_pct <= -self.stop_pct:
                return True, "stop_loss"
            if pnl_pct >= self.tp_pct:
                return True, "take_profit"
            if days_held >= self.max_hold:
                return True, "time_stop"

        return False, ""

    def calc_pnl(self, exit_price):
        """Calculate P&L including slippage."""
        if self.side == "long":
            raw_pnl = (exit_price - self.entry_price) * self.shares
        else:
            raw_pnl = (self.entry_price - exit_price) * self.shares
        # Slippage on entry + exit
        slippage = self.entry_price * self.shares * SLIPPAGE_PCT + exit_price * self.shares * SLIPPAGE_PCT
        return raw_pnl - slippage


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class BacktestEngine:
    """
    Runs the full VolTradeAI backtest with isolated component tracking.
    """

    def __init__(self, mode="all"):
        self.mode = mode
        self.equity = INITIAL_EQUITY
        self.peak_equity = INITIAL_EQUITY

        # Component trackers
        self.trackers = {
            "stocks":  ComponentTracker("Stock Picking"),
            "vrp":     ComponentTracker("VRP / SVXY"),
            "sector":  ComponentTracker("Sector Rotation"),
            "qqq":     ComponentTracker("QQQ Floor"),
            "shorts":  ComponentTracker("Intraday Shorts"),
            "options": ComponentTracker("Options (Simulated)"),
        }

        # Active positions
        self.stock_positions = []    # List of Position objects
        self.short_positions = []
        self.vrp_active = False
        self.vrp_entry_price = 0
        self.vrp_entry_idx = 0
        self.vrp_shares = 0
        self.sector_positions = {}   # sym -> (entry_price, shares, entry_idx)
        self.qqq_shares = 0
        self.qqq_entry_price = 0

        # Regime tracking
        self.regime_history = []
        self.spy_below_200_count = 0

        # Daily equity curve
        self.equity_curve = []

    def enabled(self, component):
        """Check if a component is enabled in current mode."""
        if self.mode == "all":
            return True
        if self.mode == "no-options":
            return component != "options"
        if self.mode == "no-shorts-no-sector":
            return component not in ("shorts", "sector")
        mode_map = {
            "stocks-only":  "stocks",
            "vrp-only":     "vrp",
            "etf-only":     None,   # sector + qqq
            "shorts-only":  "shorts",
            "options-only": "options",
        }
        if self.mode == "etf-only":
            return component in ("sector", "qqq")
        return mode_map.get(self.mode) == component

    def run(self, data):
        """Execute the full backtest."""
        spy = data.get("SPY")
        if spy is None:
            print("ERROR: No SPY data available")
            return

        # Prepare aligned date index from SPY
        dates = spy.index.tolist()
        print(f"\n[BACKTEST] Running {self.mode} mode | {dates[0].date()} to {dates[-1].date()} | {len(dates)} trading days")

        # Pre-compute arrays for fast access
        spy_close = spy["Close"].values.astype(float)
        spy_open = spy["Open"].values.astype(float)

        # VIX data (for VXX proxy)
        vix_df = data.get("^VIX")
        vix_close = self._align_to_dates(vix_df, dates, "Close") if vix_df is not None else np.full(len(dates), 20.0)

        # VXX data (may have gaps pre-2018)
        vxx_df = data.get("VXX")
        vxx_close = self._align_to_dates(vxx_df, dates, "Close") if vxx_df is not None else None

        # SVXY data
        svxy_df = data.get("SVXY")
        svxy_close = self._align_to_dates(svxy_df, dates, "Close") if svxy_df is not None else None
        svxy_open = self._align_to_dates(svxy_df, dates, "Open") if svxy_df is not None else None

        # GLD, ITA data
        gld_df = data.get("GLD")
        gld_close = self._align_to_dates(gld_df, dates, "Close") if gld_df is not None else None
        gld_open = self._align_to_dates(gld_df, dates, "Open") if gld_df is not None else None

        ita_df = data.get("ITA")
        ita_close = self._align_to_dates(ita_df, dates, "Close") if ita_df is not None else None
        ita_open = self._align_to_dates(ita_df, dates, "Open") if ita_df is not None else None

        # QQQ data
        qqq_df = data.get("QQQ")
        qqq_close = self._align_to_dates(qqq_df, dates, "Close") if qqq_df is not None else None
        qqq_open = self._align_to_dates(qqq_df, dates, "Open") if qqq_df is not None else None

        # Pre-compute stock data arrays
        stock_data = {}
        for sym in STOCK_UNIVERSE + SECTOR_ETFS:
            df = data.get(sym)
            if df is not None:
                stock_data[sym] = {
                    "close": self._align_to_dates(df, dates, "Close"),
                    "open":  self._align_to_dates(df, dates, "Open"),
                    "high":  self._align_to_dates(df, dates, "High"),
                    "volume": self._align_to_dates(df, dates, "Volume"),
                }

        # Pre-compute VXX ratio and regime for each day
        vxx_ratio_arr = np.full(len(dates), 1.0)
        regime_arr = ["NEUTRAL"] * len(dates)

        for i in range(len(dates)):
            # VXX ratio: use VXX data if available, otherwise VIX as proxy
            if vxx_close is not None and not np.isnan(vxx_close[i]) and vxx_close[i] > 0:
                if i >= 30:
                    vxx_30d_avg = np.nanmean(vxx_close[max(0, i - 30):i])
                    vxx_ratio_arr[i] = vxx_close[i] / vxx_30d_avg if vxx_30d_avg > 0 else 1.0
                else:
                    vxx_ratio_arr[i] = 1.0
            else:
                # VIX proxy: VXX ratio ≈ VIX / VIX_20day_SMA
                if i >= 20 and not np.isnan(vix_close[i]):
                    vix_20d_avg = np.nanmean(vix_close[max(0, i - 20):i])
                    vxx_ratio_arr[i] = vix_close[i] / vix_20d_avg if vix_20d_avg > 0 else 1.0
                else:
                    vxx_ratio_arr[i] = 1.0

            # SPY vs 50d MA
            if i >= 50:
                spy_ma50 = np.mean(spy_close[i - 50:i])
                spy_vs_ma50 = spy_close[i] / spy_ma50 if spy_ma50 > 0 else 1.0
            else:
                spy_vs_ma50 = 1.0

            # SPY below 200d MA counter
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
        for i in range(50, len(dates)):  # Start at 50 to have enough history
            date = dates[i]
            regime = regime_arr[i]
            self.regime_history.append((date, regime))
            daily_total_pnl = 0

            # ── 1. STOCK PICKING ──────────────────────────────────────
            if self.enabled("stocks"):
                pnl = self._run_stocks(i, dates, regime, stock_data, spy_close, spy_open)
                self.trackers["stocks"].record_daily(date, pnl)
                daily_total_pnl += pnl

            # ── 2. VRP / SVXY ─────────────────────────────────────────
            if self.enabled("vrp"):
                pnl = self._run_vrp(i, dates, regime, vxx_ratio_arr, vix_close,
                                     svxy_close, svxy_open, vxx_close)
                self.trackers["vrp"].record_daily(date, pnl)
                daily_total_pnl += pnl

            # ── 3. SECTOR ROTATION ────────────────────────────────────
            if self.enabled("sector"):
                pnl = self._run_sector(i, dates, regime, gld_close, gld_open,
                                        ita_close, ita_open)
                self.trackers["sector"].record_daily(date, pnl)
                daily_total_pnl += pnl

            # ── 4. QQQ FLOOR ──────────────────────────────────────────
            if self.enabled("qqq"):
                pnl = self._run_qqq_floor(i, dates, regime, qqq_close, qqq_open)
                self.trackers["qqq"].record_daily(date, pnl)
                daily_total_pnl += pnl

            # ── 5. INTRADAY SHORTS ────────────────────────────────────
            if self.enabled("shorts"):
                pnl = self._run_shorts(i, dates, regime, stock_data, spy_close, spy_open)
                self.trackers["shorts"].record_daily(date, pnl)
                daily_total_pnl += pnl

            # ── 6. OPTIONS (SIMULATED) ────────────────────────────────
            if self.enabled("options"):
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

    # ─── HELPER: Align data to SPY date index ────────────────────────
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

    # ─── COMPONENT: Stock Picking ────────────────────────────────────
    def _run_stocks(self, idx, dates, regime, stock_data, spy_close, spy_open):
        """Run stock picking logic for one day. Returns daily P&L."""
        pnl = 0.0
        params = get_adaptive_params(regime)
        date = dates[idx]

        # Check exits on existing positions (use next-day open for execution)
        exits_to_process = []
        for pos in self.stock_positions:
            sym_data = stock_data.get(pos.symbol)
            if sym_data is None:
                continue
            current_price = sym_data["close"][idx]
            if np.isnan(current_price) or current_price <= 0:
                continue
            should_exit, reason = pos.check_exit(current_price, idx, date)
            if should_exit:
                # Execute at next-day open
                if idx + 1 < len(dates):
                    exit_price = sym_data["open"][idx + 1]
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

        # Mark-to-market existing positions (unrealized P&L not counted in daily)
        # Only realized P&L contributes

        # New entries — score all stocks and pick best candidates
        if params["max_pos"] > 0 and len(self.stock_positions) < params["max_pos"]:
            candidates = []
            held_syms = {p.symbol for p in self.stock_positions}
            for sym in STOCK_UNIVERSE:
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
                # Execute at next-day open
                if idx + 1 < len(dates):
                    entry_price = sd["open"][idx + 1]
                    if np.isnan(entry_price) or entry_price <= 0:
                        continue
                else:
                    continue

                # Position sizing
                alloc = self.equity * params["size_pct"]
                shares = int(alloc / entry_price)
                if shares <= 0:
                    continue

                # Apply entry slippage
                entry_price_slipped = entry_price * (1 + SLIPPAGE_PCT)

                pos = Position(
                    symbol=sym, side="long",
                    entry_price=entry_price_slipped, shares=shares,
                    entry_date=date, entry_idx=idx,
                    stop_pct=params["sl"], tp_pct=params["tp"],
                    max_hold=params["hold"], component="stocks",
                )
                self.stock_positions.append(pos)

        return pnl

    # ─── COMPONENT: VRP / SVXY ───────────────────────────────────────
    def _run_vrp(self, idx, dates, regime, vxx_ratio_arr, vix_close,
                  svxy_close, svxy_open, vxx_close):
        """VRP harvest: buy SVXY when VXX elevated+declining, with circuit breaker."""
        pnl = 0.0
        vxx_ratio = vxx_ratio_arr[idx]
        current_vix = vix_close[idx] if not np.isnan(vix_close[idx]) else 20

        # VRP Circuit Breaker: no VRP when VIX > 40
        if current_vix > 40:
            # Force exit if holding
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

        # Check for exit: regime recovered to NEUTRAL/BULL
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

        # Check for VRP stop-loss (SVXY drops >15% from entry)
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
            # Check VXX declining over 5 days
            vxx_declining = False
            if vxx_close is not None and idx >= 5:
                if (not np.isnan(vxx_close[idx]) and not np.isnan(vxx_close[idx - 5]) and
                        vxx_close[idx - 5] > 0):
                    vxx_declining = vxx_close[idx] < vxx_close[idx - 5]
            elif idx >= 5:
                # Use VIX as proxy
                if not np.isnan(vix_close[idx]) and not np.isnan(vix_close[idx - 5]):
                    vxx_declining = vix_close[idx] < vix_close[idx - 5]

            if vxx_declining:
                # Allocation: 15% normal, 30% when VIX 25-40 (higher risk = higher alloc)
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

        # If no SVXY data, approximate VRP returns from inverse VIX daily changes
        if self.vrp_active and (svxy_close is None or np.isnan(svxy_close[idx])):
            if idx >= 1 and not np.isnan(vix_close[idx]) and not np.isnan(vix_close[idx - 1]):
                vix_daily_ret = (vix_close[idx] - vix_close[idx - 1]) / vix_close[idx - 1]
                # SVXY ≈ -0.5x VIX daily (it's -0.5x, not -1x)
                approx_svxy_ret = -0.5 * vix_daily_ret
                vrp_daily_pnl = self.vrp_entry_price * self.vrp_shares * approx_svxy_ret
                pnl += vrp_daily_pnl

        return pnl

    # ─── COMPONENT: Sector Rotation ──────────────────────────────────
    def _run_sector(self, idx, dates, regime, gld_close, gld_open,
                     ita_close, ita_open):
        """
        v1.0.30 sector rotation:
        - PANIC/BEAR → GLD (gold crash hedge), 15% allocation
        - CAUTION → ITA (defense recovery rotation), 20% allocation
        - NEUTRAL/BULL → exit all sector positions
        """
        pnl = 0.0

        # Exit logic: when regime recovers to NEUTRAL/BULL, close sector positions
        if regime in ("NEUTRAL", "BULL"):
            for sym in list(self.sector_positions.keys()):
                entry_price, shares, entry_idx = self.sector_positions[sym]
                if sym == "GLD" and gld_close is not None:
                    exit_p = gld_open[idx] if (gld_open is not None and not np.isnan(gld_open[idx])) else gld_close[idx]
                elif sym == "ITA" and ita_close is not None:
                    exit_p = ita_open[idx] if (ita_open is not None and not np.isnan(ita_open[idx])) else ita_close[idx]
                else:
                    continue
                if np.isnan(exit_p) or exit_p <= 0:
                    continue
                raw = (exit_p - entry_price) * shares
                slippage = (entry_price + exit_p) * shares * SLIPPAGE_PCT
                trade_pnl = raw - slippage
                pnl += trade_pnl
                self.trackers["sector"].record_trade({
                    "symbol": sym, "pnl": trade_pnl,
                    "reason": f"regime_exit_{regime}",
                    "days_held": idx - entry_idx,
                })
                del self.sector_positions[sym]
            return pnl

        # Entry logic
        if regime in ("PANIC", "BEAR") and "GLD" not in self.sector_positions:
            if gld_close is not None and not np.isnan(gld_close[idx]):
                if idx + 1 < len(dates) and gld_open is not None and not np.isnan(gld_open[idx + 1]):
                    entry_price = gld_open[idx + 1] * (1 + SLIPPAGE_PCT)
                else:
                    entry_price = gld_close[idx] * (1 + SLIPPAGE_PCT)
                alloc = self.equity * 0.15
                shares = int(alloc / entry_price)
                if shares > 0 and not np.isnan(entry_price) and entry_price > 0:
                    self.sector_positions["GLD"] = (entry_price, shares, idx)
                    # Exit ITA if switching to crash mode
                    if "ITA" in self.sector_positions:
                        ep, sh, ei = self.sector_positions["ITA"]
                        if ita_close is not None and not np.isnan(ita_close[idx]):
                            raw = (ita_close[idx] - ep) * sh
                            slip = (ep + ita_close[idx]) * sh * SLIPPAGE_PCT
                            pnl += raw - slip
                            self.trackers["sector"].record_trade({
                                "symbol": "ITA", "pnl": raw - slip,
                                "reason": "regime_switch_to_crash",
                                "days_held": idx - ei,
                            })
                        del self.sector_positions["ITA"]

        if regime == "CAUTION" and "ITA" not in self.sector_positions:
            if ita_close is not None and not np.isnan(ita_close[idx]):
                if idx + 1 < len(dates) and ita_open is not None and not np.isnan(ita_open[idx + 1]):
                    entry_price = ita_open[idx + 1] * (1 + SLIPPAGE_PCT)
                else:
                    entry_price = ita_close[idx] * (1 + SLIPPAGE_PCT)
                alloc = self.equity * 0.20
                shares = int(alloc / entry_price)
                if shares > 0 and not np.isnan(entry_price) and entry_price > 0:
                    self.sector_positions["ITA"] = (entry_price, shares, idx)
                    # Exit GLD if switching from crash to recovery
                    if "GLD" in self.sector_positions:
                        ep, sh, ei = self.sector_positions["GLD"]
                        if gld_close is not None and not np.isnan(gld_close[idx]):
                            raw = (gld_close[idx] - ep) * sh
                            slip = (ep + gld_close[idx]) * sh * SLIPPAGE_PCT
                            pnl += raw - slip
                            self.trackers["sector"].record_trade({
                                "symbol": "GLD", "pnl": raw - slip,
                                "reason": "regime_switch_to_recovery",
                                "days_held": idx - ei,
                            })
                        del self.sector_positions["GLD"]

        # Mark-to-market for daily P&L on held sector positions
        for sym in self.sector_positions:
            entry_price, shares, entry_idx = self.sector_positions[sym]
            if sym == "GLD" and gld_close is not None:
                today_c = gld_close[idx]
                yest_c = gld_close[idx - 1] if idx > 0 else entry_price
            elif sym == "ITA" and ita_close is not None:
                today_c = ita_close[idx]
                yest_c = ita_close[idx - 1] if idx > 0 else entry_price
            else:
                continue
            if np.isnan(today_c) or np.isnan(yest_c) or yest_c <= 0:
                continue
            daily_ret = (today_c - yest_c) / yest_c
            pnl += shares * yest_c * daily_ret

        return pnl

    # ─── COMPONENT: QQQ Floor ────────────────────────────────────────
    def _run_qqq_floor(self, idx, dates, regime, qqq_close, qqq_open):
        """
        Passive QQQ allocation in NEUTRAL/BULL regimes.
        BULL: 70%, NEUTRAL: 90%, CAUTION: 35%, BEAR/PANIC: 0%
        """
        if qqq_close is None:
            return 0.0

        pnl = 0.0
        target_alloc = QQQ_FLOOR_ALLOC.get(regime, 0.0)
        current_qqq_price = qqq_close[idx]
        if np.isnan(current_qqq_price) or current_qqq_price <= 0:
            return 0.0

        target_value = self.equity * target_alloc
        target_shares = int(target_value / current_qqq_price)

        # Rebalance if off by more than 5%
        current_value = self.qqq_shares * current_qqq_price
        if abs(current_value - target_value) > self.equity * 0.05 or (target_shares == 0 and self.qqq_shares > 0):
            delta_shares = target_shares - self.qqq_shares

            if delta_shares != 0:
                # Execute at next-day open if available
                if idx + 1 < len(dates) and qqq_open is not None and not np.isnan(qqq_open[idx + 1]):
                    exec_price = qqq_open[idx + 1]
                else:
                    exec_price = current_qqq_price

                if not np.isnan(exec_price) and exec_price > 0:
                    # Slippage cost for the rebalance
                    slippage_cost = abs(delta_shares) * exec_price * SLIPPAGE_PCT
                    pnl -= slippage_cost
                    self.qqq_shares = target_shares
                    self.qqq_entry_price = exec_price

        # Daily mark-to-market P&L on QQQ holdings
        if self.qqq_shares > 0 and idx > 0:
            yest_price = qqq_close[idx - 1]
            if not np.isnan(yest_price) and yest_price > 0:
                daily_ret = (current_qqq_price - yest_price) / yest_price
                pnl += self.qqq_shares * yest_price * daily_ret

        return pnl

    # ─── COMPONENT: Intraday Shorts ──────────────────────────────────
    def _run_shorts(self, idx, dates, regime, stock_data, spy_close, spy_open):
        """Short candidates from the full universe using score_short()."""
        pnl = 0.0

        # Only short in CAUTION/BEAR/PANIC — momentum collapse signals
        if regime in ("BULL", "NEUTRAL"):
            # Exit any existing shorts
            for pos in list(self.short_positions):
                sd = stock_data.get(pos.symbol)
                if sd is None:
                    continue
                exit_price = sd["close"][idx]
                if np.isnan(exit_price) or exit_price <= 0:
                    continue
                trade_pnl = pos.calc_pnl(exit_price)
                pnl += trade_pnl
                self.trackers["shorts"].record_trade({
                    "symbol": pos.symbol, "side": "short",
                    "pnl": trade_pnl, "reason": "regime_exit",
                    "days_held": idx - pos.entry_idx,
                })
            self.short_positions.clear()
            return pnl

        # Check exits on existing shorts
        exits = []
        for pos in self.short_positions:
            sd = stock_data.get(pos.symbol)
            if sd is None:
                continue
            current_price = sd["close"][idx]
            if np.isnan(current_price) or current_price <= 0:
                continue
            should_exit, reason = pos.check_exit(current_price, idx, dates[idx])
            if should_exit:
                if idx + 1 < len(dates):
                    exit_price = sd["open"][idx + 1]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = current_price
                else:
                    exit_price = current_price
                trade_pnl = pos.calc_pnl(exit_price)
                pnl += trade_pnl
                self.trackers["shorts"].record_trade({
                    "symbol": pos.symbol, "side": "short",
                    "pnl": trade_pnl, "reason": reason,
                    "days_held": idx - pos.entry_idx,
                })
                exits.append(pos)

        for pos in exits:
            self.short_positions.remove(pos)

        # New short entries
        if len(self.short_positions) < 3:  # Max 3 shorts at a time
            # SPY 10-day return for relative weakness scoring
            spy_ret_10d = 0
            if idx >= 10 and spy_close[idx - 10] > 0:
                spy_ret_10d = (spy_close[idx] - spy_close[idx - 10]) / spy_close[idx - 10] * 100

            candidates = []
            held_syms = {p.symbol for p in self.short_positions}
            for sym in STOCK_UNIVERSE:
                if sym in held_syms:
                    continue
                sd = stock_data.get(sym)
                if sd is None:
                    continue
                closes = sd["close"]
                highs = sd["high"]
                vols = sd["volume"]
                if np.isnan(closes[idx]) or closes[idx] <= 0:
                    continue
                result = score_short(closes, highs, vols, idx, spy_ret_10d)
                if result is not None and result["score"] >= 25:
                    candidates.append((sym, result))

            candidates.sort(key=lambda x: -x[1]["score"])
            slots = 3 - len(self.short_positions)

            for sym, result in candidates[:slots]:
                sd = stock_data.get(sym)
                if idx + 1 < len(dates):
                    entry_price = sd["open"][idx + 1]
                    if np.isnan(entry_price) or entry_price <= 0:
                        continue
                else:
                    continue

                alloc = self.equity * 0.08  # 8% per short position
                shares = int(alloc / entry_price)
                if shares <= 0:
                    continue

                entry_price_slipped = entry_price * (1 - SLIPPAGE_PCT)  # Short entry is at lower price

                pos = Position(
                    symbol=sym, side="short",
                    entry_price=entry_price_slipped, shares=shares,
                    entry_date=dates[idx], entry_idx=idx,
                    stop_pct=0.08, tp_pct=0.10,
                    max_hold=7, component="shorts",
                )
                self.short_positions.append(pos)

        return pnl

    # ─── COMPONENT: Options (Simulated) ──────────────────────────────
    def _run_options(self, idx, dates, regime, stock_data, vix_close):
        """
        SIMULATED options P&L — sell cash-secured puts on strong-scoring stocks.
        Capped at 8% of equity. Clearly labeled as simulated.
        """
        pnl = 0.0

        # Only sell puts in CAUTION/BEAR/PANIC when IV is elevated
        current_vix = vix_close[idx] if not np.isnan(vix_close[idx]) else 20
        if current_vix < 18:
            return 0.0  # Low IV — options premium not worth it

        # Cap allocation
        options_alloc = self.equity * OPTIONS_MAX_EQUITY_PCT

        # Pick top-scoring stock for CSP (use highest quick_score)
        best_sym = None
        best_score = 0
        for sym in STOCK_UNIVERSE[:10]:  # Top 10 liquid stocks
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

        # Simulate 5-day hold for CSP
        hold_days = 5
        if idx + hold_days < len(dates):
            exit_price = sd["close"][idx + hold_days]
            if np.isnan(exit_price) or exit_price <= 0:
                return 0.0

            # Historical volatility (20-day)
            if idx >= 20:
                rets = np.diff(sd["close"][idx - 20:idx + 1]) / sd["close"][idx - 20:idx]
                rets = rets[~np.isnan(rets)]
                hv = float(np.std(rets) * np.sqrt(252) * 100) if len(rets) > 5 else current_vix
            else:
                hv = current_vix

            options_return = simulate_options_pnl(entry_price, exit_price, hv, hold_days)
            trade_pnl = options_alloc * options_return

            # Only count this every 5 days to avoid double-counting
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

        # SPY benchmark
        spy_start = spy_close[50]  # Aligned with our start
        spy_end = spy_close[-1]
        spy_total_ret = (spy_end - spy_start) / spy_start
        num_years = len(dates) / 252
        spy_cagr = (1 + spy_total_ret) ** (1 / num_years) - 1

        # System totals
        total_pnl = sum(t.total_pnl for t in self.trackers.values())
        final_equity = INITIAL_EQUITY + total_pnl
        total_ret = total_pnl / INITIAL_EQUITY
        sys_cagr = (final_equity / INITIAL_EQUITY) ** (1 / num_years) - 1 if final_equity > 0 else 0

        # Max drawdown
        peak = INITIAL_EQUITY
        max_dd = 0
        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe and Sortino
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

        # ── Console Report ──
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"  VolTradeAI Backtest v2 — {self.mode.upper()} MODE")
        lines.append(f"  Period: {dates[50].date()} to {dates[-1].date()} ({len(dates) - 50} trading days, {num_years:.1f} years)")
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

        # Per-component P&L table
        lines.append("  COMPONENT P&L ATTRIBUTION")
        lines.append("  " + "-" * 76)
        lines.append(f"  {'Component':<25} {'Total P&L':>12} {'% of Total':>10} {'Trades':>8} {'Win Rate':>10} {'Max DD':>10}")
        lines.append("  " + "-" * 76)

        for key, tracker in self.trackers.items():
            if not self.enabled(key):
                continue
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
        if any(k == "options" and self.enabled(k) for k in self.trackers):
            lines.append("  * Options P&L is SIMULATED using Black-Scholes heuristic (no real chain data)")
            lines.append("")

        # Annual breakdown
        lines.append("  ANNUAL RETURNS BY COMPONENT")
        lines.append("  " + "-" * 90)

        # Gather all years
        all_years = sorted(set(
            d.year for d, _ in self.equity_curve
        ))

        # Header
        active = [(k, t) for k, t in self.trackers.items() if self.enabled(k)]
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

            # SPY annual return
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

        # ── JSON Output ──
        results = {
            "mode": self.mode,
            "period": f"{dates[50].date()} to {dates[-1].date()}",
            "trading_days": len(dates) - 50,
            "years": round(num_years, 1),
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
        }

        for key, tracker in self.trackers.items():
            if not self.enabled(key):
                continue
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
                if not self.enabled(key):
                    continue
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
        description="VolTradeAI Backtest v2 — Isolated Component P&L Attribution"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Full system with all components")
    group.add_argument("--stocks-only", action="store_true", help="Just stock picking")
    group.add_argument("--vrp-only", action="store_true", help="Just VRP/SVXY")
    group.add_argument("--etf-only", action="store_true", help="Just sector rotation + QQQ floor")
    group.add_argument("--shorts-only", action="store_true", help="Just intraday shorts")
    group.add_argument("--options-only", action="store_true", help="Just options simulation")
    group.add_argument("--no-options", action="store_true", help="Everything except options")
    group.add_argument("--no-shorts-no-sector", action="store_true", help="Everything except intraday shorts and sector rotation")

    args = parser.parse_args()

    # Determine mode
    mode = "all"
    for m in ["all", "stocks_only", "vrp_only", "etf_only", "shorts_only", "options_only", "no_options", "no_shorts_no_sector"]:
        if getattr(args, m, False):
            mode = m.replace("_", "-")
            break

    # Download data
    data = download_data()
    if "SPY" not in data:
        print("FATAL: Could not download SPY data")
        sys.exit(1)

    # Run backtest
    engine = BacktestEngine(mode=mode)
    engine.run(data)

    # Generate report
    report, results = engine.generate_report(data)

    # Save JSON results
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return report, results


if __name__ == "__main__":
    main()
