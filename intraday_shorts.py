#!/usr/bin/env python3
"""
VolTradeAI — Intraday Short Engine v2.1 (v1.0.28)
Hybrid: v1.0.27 fixed signals + v1.0.28 full-universe architecture.

Architecture from v1.0.28:
  - Scans full universe via 2-stage pipeline
  - Stage 1: snapshot pre-filter (0.01s) — down stocks + volume spikes
  - Stage 2: history fetch + 6-indicator scoring
  - Sector diversification (max 1 per sector)
  - No kill switch — user monitors manually

Signal engine from v1.0.27 (validated as superior):
  - Fixed 5-day momentum, 10/20 MA, 10-bar volume windows
  - Equal signal weights (no learned weights — not enough data)
  - ATR-relative thresholds (adaptive to each stock's volatility)
  - Dollar volume $50M floor (same as long scanner)

Why: Element-by-element testing showed adaptive lookbacks (v1.0.28)
degraded WR from 47.4% → 43.1% and avg P&L from +0.181% → +0.000%.
Fixed windows give cleaner, more consistent signals.
"""
import os, json, time, logging, requests
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("voltrade.intraday_shorts")

DATA_DIR = os.environ.get("DATA_DIR", "/tmp")
SHORTS_LOG_PATH    = os.path.join(DATA_DIR, "voltrade_intraday_shorts.json")
ALPACA_KEY    = os.environ.get("ALPACA_KEY",    "PKMDHJOVQEVIB4UHZXUYVTIDBU")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et")
ALPACA_BASE   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DATA_URL      = "https://data.alpaca.markets"
HEADERS       = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# Minimum $50M daily dollar volume (same as long scanner)
MIN_DOLLAR_VOL = 50_000_000

# Sub-signal names
SIGNAL_NAMES = ["momentum", "failed_breakout", "distribution",
                "rel_weakness", "trend_break", "gap_down"]

# Fixed equal weights — validated as best via element testing
# (learned weights were neutral; not enough closed trades to learn from)
FIXED_WEIGHTS = {n: round(1.0 / len(SIGNAL_NAMES), 4) for n in SIGNAL_NAMES}

# Fixed lookback windows — validated as superior to adaptive
# (adaptive lookbacks degraded WR by 4.3% and avg P&L by 0.181%)
FIXED_LOOKBACKS = {
    "mom": 5,         # 5-day momentum
    "ma_short": 10,   # 10-day short MA
    "ma_long": 20,    # 20-day long MA
    "vol": 10,        # 10-day volume average
}


# ── Persistent storage ────────────────────────────────────────────
def _load_log():
    try:
        with open(SHORTS_LOG_PATH) as f: return json.load(f)
    except Exception:
        return {"enabled": True, "trades": [], "disabled_reason": None}

def _save_log(data):
    try:
        with open(SHORTS_LOG_PATH, "w") as f: json.dump(data, f, indent=2)
    except Exception: pass


# ── Signal scoring (fixed windows, ATR-relative thresholds) ──────
def _compute_atr(bars, period=14):
    if len(bars) < period + 1: return None
    trs = []
    for i in range(1, len(bars)):
        h = float(bars[i].get("h", bars[i].get("c", 0)))
        l = float(bars[i].get("l", bars[i].get("c", 0)))
        pc = float(bars[i - 1].get("c", 0))
        if pc > 0: trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs[-period:])) if len(trs) >= period else None


def score_for_short(ticker, bars, spy_ret_10d=0):
    """
    Fixed-window short signal scoring — validated v1.0.27 approach.
    Uses ATR-relative thresholds (adaptive to each stock's volatility)
    but FIXED lookback windows (5d momentum, 10/20 MAs, 10d volume).
    Returns {score, signals, active, atr, atr_pct} or None.
    """
    if len(bars) < 25: return None  # Need 20+ bars for 20-day MA

    closes = [float(b.get("c", 0)) for b in bars]
    highs  = [float(b.get("h", b.get("c", 0))) for b in bars]
    lows   = [float(b.get("l", b.get("c", 0))) for b in bars]
    vols   = [int(b.get("v", 0)) for b in bars]
    c = closes[-1]

    # Dollar volume filter
    dollar_vol = c * vols[-1]
    if dollar_vol < MIN_DOLLAR_VOL: return None
    if c <= 0: return None

    atr = _compute_atr(bars)
    if not atr or atr <= 0: return None
    atr_pct = atr / c * 100

    lb = FIXED_LOOKBACKS
    signals = {}

    # 1. Momentum collapse (5-day)
    md = lb["mom"]
    if len(closes) > md and closes[-md] > 0:
        mom = (c - closes[-md]) / closes[-md] * 100
        signals["momentum"] = float(max(0, min(1, -mom / (atr_pct * 2))))
    else:
        signals["momentum"] = 0.0

    # 2. Failed breakout (20-day rolling high)
    hw = lb["ma_long"]
    if len(highs) >= hw:
        h_roll = max(highs[-hw:])
        h_recent = max(highs[-lb["mom"]:]) if len(highs) >= lb["mom"] else c
        drop = (c - h_recent) / h_recent * 100 if h_recent > 0 else 0
        near_high = h_recent >= h_roll * 0.97
        signals["failed_breakout"] = float(max(0, min(1, -drop / atr_pct))) if near_high else 0.0
    else:
        signals["failed_breakout"] = 0.0

    # 3. Volume distribution (10-day average)
    vw = lb["vol"]
    if len(vols) >= vw:
        avg_v = np.mean(vols[-vw:])
        vr = vols[-1] / max(avg_v, 1)
        day_ret = (c - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 and closes[-2] > 0 else 0
        # Volume spike on a down day = institutional selling
        signals["distribution"] = float(max(0, min(1, (vr - 1) * 0.5))) if day_ret < -(atr_pct * 0.3) else 0.0
    else:
        signals["distribution"] = 0.0

    # 4. Relative weakness vs SPY (10-day)
    rw = lb["ma_short"]
    if len(closes) >= rw and closes[-rw] > 0:
        stock_ret = (c - closes[-rw]) / closes[-rw] * 100
        relative = stock_ret - spy_ret_10d
        signals["rel_weakness"] = float(max(0, min(1, -relative / (atr_pct * 3))))
    else:
        signals["rel_weakness"] = 0.0

    # 5. Trend breakdown (10/20 MAs)
    ms, ml = lb["ma_short"], lb["ma_long"]
    if len(closes) >= ml:
        ma_s = np.mean(closes[-ms:])
        ma_l = np.mean(closes[-ml:])
        below = (c < ma_s) + (c < ma_l)
        pct_below = (c - ma_l) / ma_l * 100 if ma_l > 0 else 0
        signals["trend_break"] = float(max(0, min(1, below * 0.3 + max(0, -pct_below / atr_pct) * 0.4)))
    else:
        signals["trend_break"] = 0.0

    # 6. Gap down (ATR-relative threshold)
    if len(bars) >= 2:
        prev_c = float(bars[-2].get("c", c))
        today_o = float(bars[-1].get("o", c))
        gap = (today_o - prev_c) / prev_c * 100 if prev_c > 0 else 0
        gap_threshold = -(atr_pct * 0.2)  # 20% of ATR
        signals["gap_down"] = float(max(0, min(1, -gap / atr_pct))) if gap < gap_threshold else 0.0
    else:
        signals["gap_down"] = 0.0

    # Equal-weighted composite
    raw_score = sum(signals.get(k, 0) * FIXED_WEIGHTS.get(k, 0.167) for k in SIGNAL_NAMES)
    # Count active signals (noise floor based on stock's own volatility)
    noise_floor = min(0.4, atr_pct / 15)
    active_count = sum(1 for v in signals.values() if v > noise_floor)

    return {
        "score": float(round(raw_score * 100, 1)),
        "active_signals": int(active_count),
        "signals": {k: float(round(v, 3)) for k, v in signals.items()},
        "atr": float(round(atr, 4)),
        "atr_pct": float(round(atr_pct, 2)),
    }


# ── Two-stage full universe scan ─────────────────────────────────
def _stage1_prefilter(snapshot_data):
    """
    Stage 1: Pre-filter from snapshot data (no extra API calls).
    Finds stocks that are DOWN today with elevated volume.
    Takes ~0.01s for any universe size.
    """
    candidates = []
    for sym, snap in snapshot_data.items():
        if not isinstance(snap, dict): continue
        db = snap.get("dailyBar", {})
        pb = snap.get("prevDailyBar", {})
        if not db or not pb: continue
        c  = float(db.get("c", 0))
        pc = float(pb.get("c", 0))
        v  = int(db.get("v", 0))
        pv = int(pb.get("v", 0))
        if pc <= 0 or c <= 0: continue

        dollar_vol = c * v
        if dollar_vol < MIN_DOLLAR_VOL: continue

        day_ret = (c - pc) / pc * 100
        vol_ratio = v / max(pv, 1)

        # Pre-filter: stock is down today OR had a volume spike on a down day
        if day_ret < -0.5 or (day_ret < 0 and vol_ratio > 1.5):
            candidates.append({
                "symbol": sym, "price": c, "day_ret": day_ret,
                "vol_ratio": vol_ratio, "dollar_vol": dollar_vol,
            })

    # Sort by day_ret (most negative first) — worst performers are best short candidates
    candidates.sort(key=lambda x: x["day_ret"])
    return candidates[:500]  # Cap at 500 to keep history fetch fast


def _stage2_score(candidates, spy_ret_10d):
    """
    Stage 2: Fetch 20-day history and score with full 6-indicator signal.
    Uses parallel workers for speed.
    """
    start_date = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
    scored = []

    def _fetch_and_score(batch_syms):
        results = []
        try:
            r = requests.get(f"{DATA_URL}/v2/stocks/bars",
                params={"symbols": ",".join(batch_syms), "timeframe": "1Day",
                        "start": start_date, "limit": 200, "feed": "sip"},
                headers=HEADERS, timeout=12)
            bars_map = r.json().get("bars", {})
            for sym, bars in bars_map.items():
                if len(bars) < 25: continue
                sig = score_for_short(sym, bars, spy_ret_10d)
                if sig and sig["active_signals"] >= 2 and sig["score"] > 15:
                    price = float(bars[-1].get("c", 0))
                    results.append({"sym": sym, "sig": sig, "price": price})
        except Exception:
            pass
        return results

    # Batch into groups of 5 (Alpaca multi-symbol pagination)
    syms = [c["symbol"] for c in candidates]
    batches = [syms[i:i+5] for i in range(0, len(syms), 5)]

    with ThreadPoolExecutor(max_workers=16) as pool:
        for batch_results in pool.map(_fetch_and_score, batches):
            scored.extend(batch_results)

    scored.sort(key=lambda x: x["sig"]["score"], reverse=True)
    return scored


# ── Sector diversification ────────────────────────────────────────
KNOWN_SECTORS = {
    'AAPL':'Tech','MSFT':'Tech','NVDA':'Tech','AMD':'Tech','GOOGL':'Tech',
    'AMZN':'Cons','TSLA':'Auto','JPM':'Fin','BAC':'Fin','GS':'Fin',
    'XOM':'Energy','CVX':'Energy','JNJ':'Health','WMT':'Cons','META':'Tech',
}

def _get_sector(sym):
    if sym in KNOWN_SECTORS: return KNOWN_SECTORS[sym]
    return f"group_{sym[0]}"


# ── Main entry point ──────────────────────────────────────────────
def run_intraday_shorts(macro=None, snapshot_data=None):
    """
    Called by bot_engine.py in scan_market().
    If snapshot_data is passed (from the long scanner), uses it for Stage 1.
    Otherwise fetches most-actives as fallback.
    """
    log = _load_log()
    result = {"actions": [], "status": "ok", "enabled": log.get("enabled", True)}
    trades = log.get("trades", [])

    if not log.get("enabled", True):
        result["status"] = "disabled"
        return result

    if macro is None: macro = {}
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

    try:
        # SPY 10d return for relative weakness
        spy_r = requests.get(f"{DATA_URL}/v2/stocks/bars",
            params={"symbols": "SPY", "timeframe": "1Day", "limit": 12, "feed": "sip"},
            headers=HEADERS, timeout=8)
        spy_bars = spy_r.json().get("bars", {}).get("SPY", [])
        spy_ret_10d = ((float(spy_bars[-1]["c"]) - float(spy_bars[-10]["c"]))
                       / float(spy_bars[-10]["c"]) * 100) if len(spy_bars) >= 10 else 0

        # Stage 1: Pre-filter from snapshots
        if snapshot_data and len(snapshot_data) > 100:
            candidates = _stage1_prefilter(snapshot_data)
            result["stage1_source"] = "shared_snapshots"
        else:
            # Fallback: fetch most-actives
            snap_r = requests.get(f"{DATA_URL}/v1beta1/screener/stocks/most-actives",
                params={"top": 100}, headers=HEADERS, timeout=8)
            actives = snap_r.json().get("most_actives", [])
            candidates = []
            for a in actives:
                sym = a.get("symbol", "")
                if sym:
                    candidates.append({"symbol": sym, "price": float(a.get("price", 0) or 0),
                                       "day_ret": float(a.get("change", 0) or 0),
                                       "vol_ratio": 1.5, "dollar_vol": MIN_DOLLAR_VOL})
            candidates = [c for c in candidates if c["day_ret"] < 0][:80]
            result["stage1_source"] = "most_actives_fallback"

        result["stage1_candidates"] = len(candidates)
        if not candidates:
            result["status"] = "no_candidates_after_prefilter"
            return result

        # Stage 2: Full scoring with history
        scored = _stage2_score(candidates, spy_ret_10d)
        result["stage2_scored"] = len(scored)

        if not scored:
            result["status"] = "no_signals"
            return result

        # Pick top 2 with sector diversification
        used_sectors = set()
        picks = []
        for item in scored:
            if len(picks) >= 2: break
            sec = _get_sector(item["sym"])
            if sec in used_sectors: continue
            picks.append(item)
            used_sectors.add(sec)

        # Execute
        for pick in picks:
            sym = pick["sym"]
            sig = pick["sig"]
            price = pick["price"]

            # Position size: 3% base, scaled by signal strength and inverse volatility
            base = 0.03
            strength = sig["score"] / 50
            vol_adj = min(1.5, 3.0 / max(sig["atr_pct"], 1))
            pos_pct = min(0.06, base * max(0.5, strength * vol_adj))

            try:
                acc = requests.get(f"{ALPACA_BASE}/v2/account", headers=HEADERS, timeout=8).json()
                equity = float(acc.get("equity", 98000) or 98000)
            except Exception:
                equity = 98000

            alloc = equity * pos_pct
            shares = int(alloc / price)
            if shares <= 0: continue

            try:
                order_r = requests.post(f"{ALPACA_BASE}/v2/orders",
                    json={"symbol": sym, "qty": str(shares), "side": "sell",
                          "type": "market", "time_in_force": "day"},
                    headers=HEADERS, timeout=10)
                order = order_r.json()

                trade_record = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": sym, "side": "short", "shares": shares,
                    "entry_price": price, "atr": sig["atr"],
                    "atr_pct": sig["atr_pct"], "signal_score": sig["score"],
                    "active_signals": sig["active_signals"],
                    "signals": sig["signals"],
                    "pos_pct": round(pos_pct * 100, 2),
                    "order_id": order.get("id", "?"), "regime": regime,
                    "status": "open", "exit_price": None,
                    "pnl_pct": None, "pnl_dollar": None, "closed_at": None,
                }
                trades.append(trade_record)
                result["actions"].append({
                    "type": "intraday_short", "symbol": sym,
                    "shares": shares, "score": sig["score"],
                    "active_signals": sig["active_signals"],
                    "reason": f"Short: score={sig['score']}, {sig['active_signals']} signals, ATR={sig['atr_pct']:.1f}%",
                })
                logger.info(f"[SHORT] Sold {shares} {sym} @ ~${price:.2f} (score={sig['score']}, regime={regime})")
            except Exception as e:
                logger.debug(f"[SHORT] Order failed for {sym}: {e}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]

    # Stats
    closed_trades = [t for t in trades if t.get("pnl_pct") is not None]
    pnls = [t["pnl_pct"] for t in closed_trades]
    result["stats"] = {
        "total_trades": len(closed_trades),
        "open_trades": len([t for t in trades if t.get("status") == "open"]),
        "win_rate": round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1) if pnls else 0,
        "avg_pnl": round(float(np.mean(pnls)), 2) if pnls else 0,
        "total_pnl_pct": round(sum(pnls), 2) if pnls else 0,
        "total_pnl_dollar": round(sum(t.get("pnl_dollar", 0) or 0 for t in closed_trades), 2),
        "strategy_status": "running",
    }

    log["trades"] = trades[-500:]
    _save_log(log)
    return result


def get_dashboard_data():
    """Returns data for the frontend."""
    log = _load_log()
    trades = log.get("trades", [])
    enabled = log.get("enabled", True)
    closed = [t for t in trades if t.get("pnl_pct") is not None]
    open_t = [t for t in trades if t.get("status") == "open"]
    pnls = [t["pnl_pct"] for t in closed]

    return {
        "enabled": enabled,
        "disabled_reason": log.get("disabled_reason"),
        "total_trades": len(closed),
        "open_trades": len(open_t),
        "open_positions": [{"symbol": t["symbol"], "shares": t["shares"],
                            "entry_price": t["entry_price"], "score": t["signal_score"],
                            "timestamp": t["timestamp"]} for t in open_t],
        "win_rate": round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1) if pnls else 0,
        "avg_pnl_pct": round(float(np.mean(pnls)), 2) if pnls else 0,
        "total_pnl_pct": round(sum(pnls), 2) if pnls else 0,
        "total_pnl_dollar": round(sum(t.get("pnl_dollar", 0) or 0 for t in closed), 2),
        "best_trade": round(max(pnls), 2) if pnls else 0,
        "worst_trade": round(min(pnls), 2) if pnls else 0,
        "strategy_status": "running" if enabled else "manually_disabled",
        "recent_trades": [{"symbol": t["symbol"], "entry_price": t.get("entry_price", 0),
                           "exit_price": t.get("exit_price", 0),
                           "pnl_pct": t.get("pnl_pct", 0), "pnl_dollar": t.get("pnl_dollar", 0),
                           "score": t.get("signal_score", 0), "timestamp": t.get("timestamp", ""),
                           "closed_at": t.get("closed_at", "")} for t in reversed(closed[-20:])],
    }


def close_open_shorts():
    """Called near market close to cover all open intraday shorts."""
    log = _load_log()
    trades = log.get("trades", [])
    closed = []
    for trade in trades:
        if trade.get("status") != "open": continue
        sym = trade["symbol"]
        try:
            snap = requests.get(f"{DATA_URL}/v2/stocks/snapshots",
                params={"symbols": sym, "feed": "sip"},
                headers=HEADERS, timeout=8).json()
            curr = float(snap.get(sym, {}).get("latestTrade", {}).get("p", 0) or 0)
            if curr <= 0: continue
            requests.post(f"{ALPACA_BASE}/v2/orders",
                json={"symbol": sym, "qty": str(trade["shares"]),
                      "side": "buy", "type": "market", "time_in_force": "day"},
                headers=HEADERS, timeout=10)
            entry = trade["entry_price"]
            pnl_pct = (entry - curr) / entry * 100
            pnl_dollar = trade["shares"] * entry * (pnl_pct / 100)
            trade.update({"exit_price": curr, "pnl_pct": round(pnl_pct, 2),
                          "pnl_dollar": round(pnl_dollar, 2),
                          "closed_at": datetime.now().isoformat(), "status": "closed"})
            closed.append({"symbol": sym, "pnl_pct": pnl_pct, "pnl_dollar": pnl_dollar})
            logger.info(f"[SHORT] Covered {sym}: P&L={pnl_pct:+.2f}% (${pnl_dollar:+.0f})")
        except Exception as e:
            logger.debug(f"[SHORT] Cover failed for {sym}: {e}")
    _save_log(log)
    return {"closed": closed, "count": len(closed)}
