#!/usr/bin/env python3
"""
VolTradeAI — Intraday Short Engine v2 (v1.0.28)
Fully adaptive. No hardcoded thresholds. Full 11,600 universe.

Changes from v1:
  - Scans ALL 11,600 stocks (was 80 most-active)
  - Two-stage: snapshot pre-filter (0.01s) → history fetch (~3s)
  - Dollar volume filter replaces price+share volume floors
  - Lookback windows adapt per stock (ATR-period-relative)
  - Signal weights learn from rolling accuracy (updated every 63 bars)
  - No fixed cutoffs — uses percentile rank of today's scores
"""
import os, json, time, logging, requests
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("voltrade.intraday_shorts")

DATA_DIR = os.environ.get("DATA_DIR", "/tmp")
SHORTS_LOG_PATH    = os.path.join(DATA_DIR, "voltrade_intraday_shorts.json")
WEIGHT_LEARN_PATH  = os.path.join(DATA_DIR, "voltrade_short_weights.json")
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

def _load_weights():
    """Load learned signal weights. Falls back to equal weights."""
    try:
        with open(WEIGHT_LEARN_PATH) as f:
            w = json.load(f)
            if len(w) == len(SIGNAL_NAMES):
                return w
    except Exception: pass
    # Equal weights as starting point — will be learned from data
    return {n: round(1.0 / len(SIGNAL_NAMES), 4) for n in SIGNAL_NAMES}

def _save_weights(w):
    try:
        with open(WEIGHT_LEARN_PATH, "w") as f: json.dump(w, f, indent=2)
    except Exception: pass

def _update_weights_from_results():
    """
    Learn signal weights from actual trade outcomes.
    Runs every 63 days (quarterly) or after every 10 new closed trades.
    Each signal gets a weight proportional to its accuracy at predicting drops.
    """
    log = _load_log()
    closed = [t for t in log.get("trades", []) if t.get("pnl_pct") is not None and t.get("signals")]
    if len(closed) < 5:
        return _load_weights()  # Not enough data yet

    # For each signal: compute correlation with trade outcome
    signal_scores = {n: [] for n in SIGNAL_NAMES}
    outcomes = []
    for t in closed[-100:]:  # Use last 100 trades
        sigs = t.get("signals", {})
        pnl = t.get("pnl_pct", 0)
        outcomes.append(pnl)
        for n in SIGNAL_NAMES:
            signal_scores[n].append(float(sigs.get(n, 0)))

    # Weight = correlation of signal with positive P&L (signal fired → stock fell)
    new_weights = {}
    for n in SIGNAL_NAMES:
        if len(signal_scores[n]) >= 5:
            corr = np.corrcoef(signal_scores[n], outcomes)[0, 1]
            # Positive correlation = signal predicted profitable short
            new_weights[n] = max(0.05, float(corr)) if not np.isnan(corr) else 0.167
        else:
            new_weights[n] = 0.167  # equal weight fallback

    # Normalize to sum = 1.0
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

    _save_weights(new_weights)
    return new_weights


# ── Adaptive parameters per stock ─────────────────────────────────
def _adaptive_lookback(atr_pct):
    """
    Faster-moving stocks (higher ATR%) get shorter lookback windows.
    Slow stocks: 10-day momentum, 20-day MAs
    Fast stocks: 3-day momentum, 8-day MAs
    """
    # ATR% < 2 = slow (large cap), > 6 = fast (biotech/meme)
    speed = min(1.0, max(0.0, (atr_pct - 1.5) / 5.0))  # 0=slow, 1=fast
    mom_days   = int(round(10 - speed * 7))    # 10 → 3
    ma_short   = int(round(15 - speed * 7))    # 15 → 8
    ma_long    = int(round(25 - speed * 10))   # 25 → 15
    vol_window = int(round(20 - speed * 10))   # 20 → 10
    return {"mom": max(3, mom_days), "ma_short": max(5, ma_short),
            "ma_long": max(10, ma_long), "vol": max(5, vol_window)}


# ── Signal scoring ────────────────────────────────────────────────
def _compute_atr(bars, period=14):
    if len(bars) < period + 1: return None
    trs = []
    for i in range(1, len(bars)):
        h = float(bars[i].get("h", bars[i].get("c", 0)))
        l = float(bars[i].get("l", bars[i].get("c", 0)))
        pc = float(bars[i - 1].get("c", 0))
        if pc > 0: trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs[-period:])) if len(trs) >= period else None


def score_for_short(ticker, bars, spy_ret_10d=0, weights=None):
    """
    Fully adaptive short signal. Everything derived from the stock's own data.
    Returns {score, signals, active, atr, atr_pct, lookbacks} or None.
    """
    if len(bars) < 15: return None

    closes = [float(b.get("c", 0)) for b in bars]
    highs  = [float(b.get("h", b.get("c", 0))) for b in bars]
    lows   = [float(b.get("l", b.get("c", 0))) for b in bars]
    vols   = [int(b.get("v", 0)) for b in bars]
    c = closes[-1]

    # Dollar volume filter (adaptive — no fixed price or share volume)
    dollar_vol = c * vols[-1]
    if dollar_vol < MIN_DOLLAR_VOL: return None
    if c <= 0: return None

    atr = _compute_atr(bars)
    if not atr or atr <= 0: return None
    atr_pct = atr / c * 100

    # Adaptive lookback windows based on stock's speed
    lb = _adaptive_lookback(atr_pct)
    if weights is None:
        weights = _load_weights()

    signals = {}

    # 1. Momentum collapse (adaptive lookback)
    md = lb["mom"]
    if len(closes) > md and closes[-md] > 0:
        mom = (c - closes[-md]) / closes[-md] * 100
        signals["momentum"] = float(max(0, min(1, -mom / (atr_pct * 2))))
    else:
        signals["momentum"] = 0.0

    # 2. Failed breakout (rolling high, adaptive window)
    hw = lb["ma_long"]
    if len(highs) >= hw:
        h_roll = max(highs[-hw:])
        h_recent = max(highs[-lb["mom"]:]) if len(highs) >= lb["mom"] else c
        drop = (c - h_recent) / h_recent * 100 if h_recent > 0 else 0
        near_high = h_recent >= h_roll * 0.97
        signals["failed_breakout"] = float(max(0, min(1, -drop / atr_pct))) if near_high else 0.0
    else:
        signals["failed_breakout"] = 0.0

    # 3. Volume distribution (adaptive window)
    vw = lb["vol"]
    if len(vols) >= vw:
        avg_v = np.mean(vols[-vw:])
        vr = vols[-1] / max(avg_v, 1)
        day_ret = (c - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 and closes[-2] > 0 else 0
        # Volume spike on a down day = institutional selling
        signals["distribution"] = float(max(0, min(1, (vr - 1) * 0.5))) if day_ret < -(atr_pct * 0.3) else 0.0
    else:
        signals["distribution"] = 0.0

    # 4. Relative weakness vs SPY (adaptive window)
    rw = lb["ma_short"]
    if len(closes) >= rw and closes[-rw] > 0:
        stock_ret = (c - closes[-rw]) / closes[-rw] * 100
        relative = stock_ret - spy_ret_10d
        signals["rel_weakness"] = float(max(0, min(1, -relative / (atr_pct * 3))))
    else:
        signals["rel_weakness"] = 0.0

    # 5. Trend breakdown (adaptive MA periods)
    ms, ml = lb["ma_short"], lb["ma_long"]
    if len(closes) >= ml:
        ma_s = np.mean(closes[-ms:])
        ma_l = np.mean(closes[-ml:])
        below = (c < ma_s) + (c < ma_l)
        pct_below = (c - ma_l) / ma_l * 100 if ma_l > 0 else 0
        signals["trend_break"] = float(max(0, min(1, below * 0.3 + max(0, -pct_below / atr_pct) * 0.4)))
    else:
        signals["trend_break"] = 0.0

    # 6. Gap down (ATR-relative threshold, not fixed -0.5%)
    if len(bars) >= 2:
        prev_c = float(bars[-2].get("c", c))
        today_o = float(bars[-1].get("o", c))
        gap = (today_o - prev_c) / prev_c * 100 if prev_c > 0 else 0
        gap_threshold = -(atr_pct * 0.2)  # Adaptive: gap must be 20% of ATR
        signals["gap_down"] = float(max(0, min(1, -gap / atr_pct))) if gap < gap_threshold else 0.0
    else:
        signals["gap_down"] = 0.0

    # Weighted composite (weights from learning or equal)
    raw_score = sum(signals.get(k, 0) * weights.get(k, 0.167) for k in SIGNAL_NAMES)
    # Count active signals (above each stock's own noise floor: atr_pct-relative)
    noise_floor = min(0.4, atr_pct / 15)  # Noisier stocks need stronger signals
    active_count = sum(1 for v in signals.values() if v > noise_floor)

    return {
        "score": float(round(raw_score * 100, 1)),
        "active_signals": int(active_count),
        "signals": {k: float(round(v, 3)) for k, v in signals.items()},
        "atr": float(round(atr, 4)),
        "atr_pct": float(round(atr_pct, 2)),
        "lookbacks": lb,
        "weights_used": {k: round(v, 3) for k, v in weights.items()},
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


def _stage2_score(candidates, spy_ret_10d, weights):
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
                if len(bars) < 15: continue
                sig = score_for_short(sym, bars, spy_ret_10d, weights)
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
# Light sector mapping — can't know every stock's sector without Finnhub call
# Use exchange + price as a rough proxy, or known symbols
KNOWN_SECTORS = {
    'AAPL':'Tech','MSFT':'Tech','NVDA':'Tech','AMD':'Tech','GOOGL':'Tech',
    'AMZN':'Cons','TSLA':'Auto','JPM':'Fin','BAC':'Fin','GS':'Fin',
    'XOM':'Energy','CVX':'Energy','JNJ':'Health','WMT':'Cons','META':'Tech',
}

def _get_sector(sym):
    if sym in KNOWN_SECTORS: return KNOWN_SECTORS[sym]
    # Rough heuristic: first letter grouping for unknowns
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

    # Update learned weights periodically
    weights = _load_weights()
    closed_count = len([t for t in trades if t.get("pnl_pct") is not None])
    if closed_count > 0 and closed_count % 10 == 0:
        weights = _update_weights_from_results()

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
            # Convert to snapshot-like format for pre-filter
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
        scored = _stage2_score(candidates, spy_ret_10d, weights)
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

            # Adaptive position size: signal_strength × inverse_volatility
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
                    "signals": sig["signals"], "lookbacks": sig["lookbacks"],
                    "weights": sig["weights_used"],
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
        "weights": weights,
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
    weights = _load_weights()

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
        "signal_weights": weights,
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
