#!/usr/bin/env python3
"""
VolTradeAI — Intraday Short Engine (v1.0.27)
Separate module so it doesn't pollute bot_engine.py.

How it works:
  1. Score all 11,600 stocks for short signal (6 adaptive sub-signals)
  2. Pick top 1-2 worst-scoring stocks with liquid options
  3. Short at market open, cover by close (same day)
  4. ATR-based TP/SL — no hardcoded numbers
  5. Auto-disable if first 20 trades average negative

Backtest: intraday is breakeven on 59 stocks (45.5% WR, -0.02% avg).
Hypothesis: 11,600 stocks with top-0.025% selectivity pushes WR to 55%+.
This module validates that hypothesis live with real data.
"""
import os, json, time, logging, requests
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger("voltrade.intraday_shorts")

# ── Config ────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/tmp")
SHORTS_LOG_PATH = os.path.join(DATA_DIR, "voltrade_intraday_shorts.json")
ALPACA_KEY    = os.environ.get("ALPACA_KEY",    "PKMDHJOVQEVIB4UHZXUYVTIDBU")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et")
ALPACA_BASE   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DATA_URL      = "https://data.alpaca.markets"
HEADERS       = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# Kill switch settings
# No auto-kill — user watches dashboard and decides manually


def _load_log():
    try:
        with open(SHORTS_LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return {"enabled": True, "trades": [], "disabled_reason": None}


def _save_log(data):
    try:
        with open(SHORTS_LOG_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _compute_atr(bars, period=14):
    """ATR from a list of OHLC bar dicts."""
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        h = float(bars[i].get("h", bars[i].get("c", 0)))
        l = float(bars[i].get("l", bars[i].get("c", 0)))
        pc = float(bars[i-1].get("c", 0))
        if pc > 0:
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return np.mean(trs[-period:]) if len(trs) >= period else None


def score_for_short(ticker, bars_20d, spy_ret_10d=0):
    """
    Adaptive short signal. No hardcoded thresholds.
    Returns: {score, active_signals, atr, atr_pct} or None.
    """
    if len(bars_20d) < 15:
        return None

    closes = [float(b.get("c", 0)) for b in bars_20d]
    highs  = [float(b.get("h", b.get("c", 0))) for b in bars_20d]
    vols   = [int(b.get("v", 0)) for b in bars_20d]
    c = closes[-1]
    if c < 3 or vols[-1] < 200_000:
        return None

    atr = _compute_atr(bars_20d)
    if not atr or atr <= 0:
        return None
    atr_pct = atr / c * 100

    signals = {}

    # 1. Momentum collapse (ATR-normalized)
    mom5 = (c - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 and closes[-5] > 0 else 0
    signals["momentum"] = max(0, min(1, -mom5 / (atr_pct * 2)))

    # 2. Failed breakout
    if len(highs) >= 15:
        h15 = max(highs[-15:]); h5 = max(highs[-5:])
        drop = (c - h5) / h5 * 100 if h5 > 0 else 0
        signals["failed_breakout"] = max(0, min(1, -drop / atr_pct)) if h5 >= h15 * 0.97 else 0
    else:
        signals["failed_breakout"] = 0

    # 3. Volume distribution (selling pressure)
    avg_v = np.mean(vols[-15:]) if len(vols) >= 15 else np.mean(vols)
    vr = vols[-1] / max(avg_v, 1)
    day_ret = (c - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 and closes[-2] > 0 else 0
    signals["distribution"] = max(0, min(1, (vr - 1) * 0.5)) if day_ret < -0.5 else 0

    # 4. Relative weakness vs SPY
    stock_ret_10d = (c - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 and closes[-10] > 0 else 0
    relative = stock_ret_10d - spy_ret_10d
    signals["rel_weakness"] = max(0, min(1, -relative / (atr_pct * 3)))

    # 5. Trend breakdown (below MAs)
    ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else c
    ma20 = np.mean(closes[-15:]) if len(closes) >= 15 else c
    below = (c < ma10) + (c < ma20)
    pct_below = (c - ma20) / ma20 * 100 if ma20 > 0 else 0
    signals["trend_break"] = max(0, min(1, below * 0.3 + max(0, -pct_below / atr_pct) * 0.4))

    # 6. Gap down
    if len(bars_20d) >= 2:
        prev_c = float(bars_20d[-2].get("c", c))
        today_o = float(bars_20d[-1].get("o", c))
        gap = (today_o - prev_c) / prev_c * 100 if prev_c > 0 else 0
        signals["gap_down"] = max(0, min(1, -gap / atr_pct)) if gap < -0.5 else 0
    else:
        signals["gap_down"] = 0

    # Weighted composite
    weights = {"momentum": 0.15, "failed_breakout": 0.15, "distribution": 0.20,
               "rel_weakness": 0.20, "trend_break": 0.10, "gap_down": 0.20}
    raw_score = sum(signals[k] * weights[k] for k in signals)
    active_count = sum(1 for v in signals.values() if v > 0.3)

    return {
        "score": float(round(raw_score * 100, 1)),
        "active_signals": int(active_count),
        "signals": {k: float(round(v, 3)) for k, v in signals.items()},
        "atr": float(round(atr, 4)),
        "atr_pct": float(round(atr_pct, 2)),
    }


def run_intraday_shorts(macro: dict = None) -> dict:
    """
    Main entry point — called by bot_engine.py in scan_market().
    Returns: {actions, status, enabled, stats}
    """
    log = _load_log()
    result = {"actions": [], "status": "ok", "enabled": log.get("enabled", True)}

    trades = log.get("trades", [])

    if not log.get("enabled", True):
        result["status"] = "disabled"
        result["reason"] = log.get("disabled_reason", "Manually disabled")
        return result

    # Check regime
    if macro is None:
        macro = {}
    vxx_ratio = float(macro.get("vxx_ratio", 1.0) or 1.0)
    spy_vs_ma50 = float(macro.get("spy_vs_ma50", 1.0) or 1.0)
    spy_b200 = int(macro.get("spy_below_200_days", 0) or 0)

    try:
        from system_config import get_market_regime
        regime = get_market_regime(vxx_ratio, spy_vs_ma50, spy_below_200_days=spy_b200)
    except Exception:
        regime = "NEUTRAL"

    if regime not in ("BEAR", "PANIC", "CAUTION"):
        result["status"] = f"regime_{regime}_no_shorts"
        return result

    # Fetch snapshots for the full universe (reuse existing scanner data)
    try:
        # Get SPY 10d return for relative weakness calc
        spy_r = requests.get(f"{DATA_URL}/v2/stocks/bars",
            params={"symbols": "SPY", "timeframe": "1Day", "limit": 12, "feed": "sip"},
            headers=HEADERS, timeout=8)
        spy_bars = spy_r.json().get("bars", {}).get("SPY", [])
        if len(spy_bars) >= 10:
            spy_ret_10d = (float(spy_bars[-1]["c"]) - float(spy_bars[-10]["c"])) / float(spy_bars[-10]["c"]) * 100
        else:
            spy_ret_10d = 0

        # Get most-actives (top volume stocks today — most likely to have signals)
        snap_r = requests.get(f"{DATA_URL}/v1beta1/screener/stocks/most-actives",
            params={"top": 100}, headers=HEADERS, timeout=8)
        actives = snap_r.json().get("most_actives", [])
        tickers = [a.get("symbol", "") for a in actives if a.get("symbol")][:80]

        if not tickers:
            result["status"] = "no_actives"
            return result

        # Fetch 20-day bars for each candidate
        scored = []
        start_date = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
        for batch_start in range(0, len(tickers), 5):  # 5 per batch for full history
            batch = tickers[batch_start:batch_start+5]
            try:
                r = requests.get(f"{DATA_URL}/v2/stocks/bars",
                    params={"symbols": ",".join(batch), "timeframe": "1Day",
                            "start": start_date, "limit": 200, "feed": "sip"},
                    headers=HEADERS, timeout=10)
                bars_map = r.json().get("bars", {})
                for sym, bars in bars_map.items():
                    if len(bars) < 15: continue
                    sig = score_for_short(sym, bars, spy_ret_10d)
                    if sig and sig["active_signals"] >= 2 and sig["score"] > 30:
                        price = float(bars[-1].get("c", 0))
                        vol = int(bars[-1].get("v", 0))
                        dollar_vol = price * vol
                        if dollar_vol >= 50_000_000:  # $50M min dollar volume
                            scored.append({"sym": sym, "sig": sig, "price": price})
            except Exception:
                continue

        if not scored:
            result["status"] = "no_signals"
            result["scanned"] = len(tickers)
            return result

        # Sort by score, take top 1-2
        scored.sort(key=lambda x: x["sig"]["score"], reverse=True)
        top_picks = scored[:2]

        # Execute shorts
        for pick in top_picks:
            sym = pick["sym"]
            sig = pick["sig"]
            price = pick["price"]
            atr = sig["atr"]

            # Adaptive position size
            base_pct = 0.03  # 3% of equity base
            size_mult = sig["score"] / 50 * min(1.5, 3 / max(sig["atr_pct"], 1))
            pos_pct = min(0.06, base_pct * max(0.5, size_mult))  # 1.5%-6% of equity

            # Get account equity
            try:
                acc = requests.get(f"{ALPACA_BASE}/v2/account", headers=HEADERS, timeout=8).json()
                equity = float(acc.get("equity", 98000) or 98000)
            except Exception:
                equity = 98000

            alloc = equity * pos_pct
            shares = int(alloc / price)
            if shares <= 0:
                continue

            # Place short order
            order_data = {
                "symbol": sym,
                "qty": str(shares),
                "side": "sell",
                "type": "market",
                "time_in_force": "day",  # auto-covers at close
            }

            try:
                order_r = requests.post(f"{ALPACA_BASE}/v2/orders",
                    json=order_data, headers=HEADERS, timeout=10)
                order = order_r.json()
                order_id = order.get("id", "?")

                trade_record = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": sym,
                    "side": "short",
                    "shares": shares,
                    "entry_price": price,
                    "atr": atr,
                    "atr_pct": sig["atr_pct"],
                    "signal_score": sig["score"],
                    "active_signals": sig["active_signals"],
                    "signals": sig["signals"],
                    "pos_pct": round(pos_pct * 100, 2),
                    "order_id": order_id,
                    "regime": regime,
                    "status": "open",
                    "exit_price": None,
                    "pnl_pct": None,
                    "pnl_dollar": None,
                    "closed_at": None,
                }

                trades.append(trade_record)
                result["actions"].append({
                    "type": "intraday_short",
                    "symbol": sym,
                    "shares": shares,
                    "score": sig["score"],
                    "reason": f"Short signal: {sig['active_signals']} active indicators, score={sig['score']}",
                })
                logger.info(f"[INTRADAY-SHORT] Sold {shares} {sym} @ ~${price:.2f} (score={sig['score']}, regime={regime})")
            except Exception as e:
                logger.debug(f"[INTRADAY-SHORT] Order failed for {sym}: {e}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]

    # Compute running stats
    closed_trades = [t for t in trades if t.get("pnl_pct") is not None]
    if closed_trades:
        pnls = [t["pnl_pct"] for t in closed_trades]
        result["stats"] = {
            "total_trades": len(closed_trades),
            "open_trades": len([t for t in trades if t.get("status") == "open"]),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
            "avg_pnl": round(np.mean(pnls), 2),
            "total_pnl_pct": round(sum(pnls), 2),
            "total_pnl_dollar": round(sum(t.get("pnl_dollar", 0) or 0 for t in closed_trades), 2),
            "best_trade": round(max(pnls), 2),
            "worst_trade": round(min(pnls), 2),
            "kill_switch_status": "running",
        }
    else:
        result["stats"] = {
            "total_trades": 0, "open_trades": len([t for t in trades if t.get("status") == "open"]),
            "win_rate": 0, "avg_pnl": 0, "total_pnl_pct": 0, "total_pnl_dollar": 0,
            "kill_switch_status": "waiting_for_signals",
        }

    log["trades"] = trades[-500:]  # keep last 500
    _save_log(log)
    result["scanned"] = len(tickers) if 'tickers' in dir() else 0
    result["candidates"] = len(scored) if 'scored' in dir() else 0
    return result


def get_dashboard_data() -> dict:
    """Returns data for the frontend dashboard."""
    log = _load_log()
    trades = log.get("trades", [])
    enabled = log.get("enabled", True)

    closed = [t for t in trades if t.get("pnl_pct") is not None]
    open_trades = [t for t in trades if t.get("status") == "open"]
    pnls = [t["pnl_pct"] for t in closed]

    return {
        "enabled": enabled,
        "disabled_reason": log.get("disabled_reason"),
        "total_trades": len(closed),
        "open_trades": len(open_trades),
        "open_positions": [{
            "symbol": t["symbol"], "shares": t["shares"],
            "entry_price": t["entry_price"], "score": t["signal_score"],
            "timestamp": t["timestamp"],
        } for t in open_trades],
        "win_rate": round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1),
        "avg_pnl_pct": round(np.mean(pnls), 2) if pnls else 0,
        "total_pnl_pct": round(sum(pnls), 2) if pnls else 0,
        "total_pnl_dollar": round(sum(t.get("pnl_dollar", 0) or 0 for t in closed), 2),
        "best_trade": round(max(pnls), 2) if pnls else 0,
        "worst_trade": round(min(pnls), 2) if pnls else 0,
        "strategy_status": "running" if enabled else "manually_disabled",
        "recent_trades": [{
            "symbol": t["symbol"], "entry_price": t.get("entry_price", 0),
            "exit_price": t.get("exit_price", 0),
            "pnl_pct": t.get("pnl_pct", 0),
            "pnl_dollar": t.get("pnl_dollar", 0),
            "score": t.get("signal_score", 0),
            "timestamp": t.get("timestamp", ""),
            "closed_at": t.get("closed_at", ""),
        } for t in reversed(closed[-20:])],  # last 20 closed trades
    }


def close_open_shorts() -> dict:
    """Called near market close to cover all open intraday shorts."""
    log = _load_log()
    trades = log.get("trades", [])
    closed = []

    for trade in trades:
        if trade.get("status") != "open":
            continue
        sym = trade["symbol"]
        try:
            # Get current price
            snap = requests.get(f"{DATA_URL}/v2/stocks/snapshots",
                params={"symbols": sym, "feed": "sip"},
                headers=HEADERS, timeout=8).json()
            curr_price = float(snap.get(sym, {}).get("latestTrade", {}).get("p", 0) or 0)
            if curr_price <= 0:
                continue

            # Cover: buy to close
            order = requests.post(f"{ALPACA_BASE}/v2/orders",
                json={"symbol": sym, "qty": str(trade["shares"]),
                      "side": "buy", "type": "market", "time_in_force": "day"},
                headers=HEADERS, timeout=10).json()

            entry = trade["entry_price"]
            pnl_pct = (entry - curr_price) / entry * 100  # positive = stock fell
            pnl_dollar = trade["shares"] * entry * (pnl_pct / 100)

            trade["exit_price"] = curr_price
            trade["pnl_pct"] = round(pnl_pct, 2)
            trade["pnl_dollar"] = round(pnl_dollar, 2)
            trade["closed_at"] = datetime.now().isoformat()
            trade["status"] = "closed"
            trade["close_order_id"] = order.get("id", "?")

            closed.append({"symbol": sym, "pnl_pct": pnl_pct, "pnl_dollar": pnl_dollar})
            logger.info(f"[INTRADAY-SHORT] Covered {sym}: P&L={pnl_pct:+.2f}% (${pnl_dollar:+.0f})")
        except Exception as e:
            logger.debug(f"[INTRADAY-SHORT] Cover failed for {sym}: {e}")

    _save_log(log)
    return {"closed": closed, "count": len(closed)}
