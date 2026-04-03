#!/usr/bin/env python3
"""
VolTradeAI Autonomous Bot Engine
─────────────────────────────────
Scans all tradeable stocks, scores them across multiple strategies,
picks the best trades, and executes via Alpaca.

Called by: server/bot.ts on a schedule (every 15 min during market hours)
Output: JSON with recommended trades and actions

Usage:
  python3 bot_engine.py scan          # Scan market, return top opportunities
  python3 bot_engine.py manage        # Check existing positions, manage stops
  python3 bot_engine.py full          # Full cycle: scan + decide + recommend
"""

import sys
import json
import os
import time
import numpy as np
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────────────────────

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "UNwTHo3kvZMBckeIaHQbBLuaaURmFUQP")
ALPACA_KEY = os.environ.get("ALPACA_KEY", "PKMDHJOVQEVIB4UHZXUYVTIDBU")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et")

MAX_POSITIONS = 5          # Max stocks to hold at once
MAX_POSITION_PCT = 0.05    # 5% of portfolio per position
STOP_LOSS_PCT = 0.02       # 2% stop loss
TAKE_PROFIT_PCT = 0.06     # 6% take profit (3:1 reward/risk)
MIN_SCORE = 65             # Minimum combined score to trade
MIN_VOLUME = 500000        # Minimum avg daily volume
MIN_PRICE = 5              # Minimum stock price

# ── Data Fetching ───────────────────────────────────────────────────────────

def get_polygon_snapshot():
    """Get all US stocks from Polygon Grouped Daily (instant, 1 API call)."""
    import requests
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    # Try yesterday, if weekend try Friday
    for days_back in range(1, 5):
        date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}?adjusted=true&apiKey={POLYGON_KEY}"
        try:
            r = requests.get(url, timeout=15)
            data = r.json()
            if data.get("resultsCount", 0) > 0:
                return data.get("results", [])
        except:
            continue
    return []

def get_stock_details(ticker):
    """Get detailed analysis from analyze.py."""
    import subprocess
    try:
        result = subprocess.run(
            ["python3", os.path.join(os.path.dirname(__file__), "analyze.py"), ticker, "--mode=scan"],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            return json.loads(result.stdout.strip())
    except:
        pass
    return None

def get_alpaca_account():
    """Get Alpaca account info."""
    import requests
    r = requests.get("https://paper-api.alpaca.markets/v2/account", headers={
        "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET
    }, timeout=10)
    return r.json()

def get_alpaca_positions():
    """Get current Alpaca positions."""
    import requests
    r = requests.get("https://paper-api.alpaca.markets/v2/positions", headers={
        "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET
    }, timeout=10)
    return r.json()

# ── Strategy Scoring ────────────────────────────────────────────────────────

def score_stock(stock_data):
    """
    Score a stock across all strategies. Returns combined score 0-100.
    
    stock_data from Polygon: {T: ticker, o: open, h: high, l: low, c: close, v: volume, vw: vwap}
    """
    ticker = stock_data.get("T", "")
    close = stock_data.get("c", 0)
    open_price = stock_data.get("o", 0)
    high = stock_data.get("h", 0)
    low = stock_data.get("l", 0)
    volume = stock_data.get("v", 0)
    vwap = stock_data.get("vw", 0)
    
    # Basic filters
    if close < MIN_PRICE or volume < MIN_VOLUME:
        return None
    if "." in ticker or len(ticker) > 5:  # Skip warrants, units, etc.
        return None
    
    # Calculate quick signals from available data
    change_pct = ((close - open_price) / open_price * 100) if open_price > 0 else 0
    range_pct = ((high - low) / low * 100) if low > 0 else 0
    vwap_dist = ((close - vwap) / vwap * 100) if vwap > 0 else 0
    
    # Quick scoring (no API calls needed — just math on Polygon data)
    score = 50  # Start neutral
    reasons = []
    
    # Price action score
    if change_pct > 3:
        score += 10
        reasons.append(f"Up {change_pct:.1f}% today — momentum")
    elif change_pct < -3:
        score += 8  # Mean reversion candidate
        reasons.append(f"Down {abs(change_pct):.1f}% today — bounce candidate")
    
    # Volume score (unusual volume = institutional interest)
    # We don't have avg volume from Polygon snapshot, so use absolute thresholds
    if volume > 20000000:
        score += 15
        reasons.append(f"Very high volume ({volume/1e6:.0f}M)")
    elif volume > 5000000:
        score += 8
    elif volume > 1000000:
        score += 3
    
    # VWAP position (above VWAP = buying pressure)
    if vwap_dist > 1:
        score += 5
        reasons.append("Trading above VWAP — buyers in control")
    elif vwap_dist < -1:
        score += 3
        reasons.append("Below VWAP — potential dip buy")
    
    # Volatility (intraday range) — higher range = more opportunity
    if range_pct > 5:
        score += 10
        reasons.append(f"Wide range ({range_pct:.1f}%) — active stock")
    elif range_pct > 3:
        score += 5
    
    # Cap at 100
    score = max(0, min(100, score))
    
    return {
        "ticker": ticker,
        "price": round(close, 2),
        "change_pct": round(change_pct, 2),
        "volume": volume,
        "range_pct": round(range_pct, 2),
        "vwap_dist": round(vwap_dist, 2),
        "quick_score": score,
        "reasons": reasons,
    }

def deep_score(ticker, quick_result):
    """
    Deep analysis on a pre-filtered stock using analyze.py.
    Adds VRP, sentiment, earnings, edge factors to the score.
    """
    detail = get_stock_details(ticker)
    if not detail or "error" in detail:
        return quick_result
    
    score = quick_result.get("quick_score", 50)
    reasons = list(quick_result.get("reasons", []))
    
    # VRP signal (max +20)
    vrp = detail.get("vrp", 0)
    if vrp and vrp > 5:
        score += 15
        reasons.append(f"High VRP (+{vrp:.1f}%) — sell options edge")
    elif vrp and vrp < -3:
        score += 10
        reasons.append(f"Cheap IV ({vrp:.1f}%) — buy options edge")
    
    # Recommendation
    rec = detail.get("recommendation", {})
    if rec:
        action = rec.get("action", "")
        if "BUY" in action.upper():
            score += 15
            reasons.append(f"AI recommends: {action}")
        elif "SELL" in action.upper() and "OPTIONS" in action.upper():
            score += 10
            reasons.append(f"AI recommends: {action}")
    
    # Sentiment
    sentiment = detail.get("sentiment", {})
    if sentiment:
        sent_score = sentiment.get("score", 50)
        contrarian = sentiment.get("contrarian_flag")
        if contrarian == "Squeeze Watch":
            score += 15
            reasons.append("SQUEEZE WATCH — retail hype + high short interest")
        elif contrarian == "Buy the Dip":
            score += 10
            reasons.append("Buy the Dip signal — retail panic, institutions buying")
        elif sent_score > 75:
            score += 5
    
    # Edge factors
    edge = detail.get("edge_factors", {})
    if edge:
        squeeze = edge.get("squeeze_score", 0)
        if squeeze and squeeze > 70:
            score += 15
            reasons.append(f"Squeeze score: {squeeze}/100")
        
        rs = edge.get("relative_strength", 0)
        if rs and rs > 5:
            score += 8
            reasons.append(f"Outperforming SPY by {rs:.1f}%")
    
    score = max(0, min(100, score))
    
    return {
        **quick_result,
        "deep_score": score,
        "reasons": reasons,
        "vrp": vrp,
        "recommendation": rec.get("action") if rec else None,
        "rec_reasoning": rec.get("reasoning") if rec else None,
        "sentiment_score": sentiment.get("score") if sentiment else None,
        "horizon": rec.get("horizon") if rec else None,
        "leveraged_bull": rec.get("leveraged_bull") if rec else None,
        "leveraged_bear": rec.get("leveraged_bear") if rec else None,
    }

# ── Position Management ─────────────────────────────────────────────────────

def manage_positions():
    """Check existing positions, recommend closes for stop-loss or take-profit."""
    try:
        positions = get_alpaca_positions()
    except:
        return {"actions": [], "error": "Could not fetch positions"}
    
    if not isinstance(positions, list):
        return {"actions": [], "positions": 0}
    
    actions = []
    for pos in positions:
        ticker = pos.get("symbol", "")
        entry = float(pos.get("avg_entry_price", 0))
        current = float(pos.get("current_price", 0))
        pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100
        
        if pnl_pct <= -STOP_LOSS_PCT * 100:
            actions.append({
                "action": "CLOSE",
                "ticker": ticker,
                "reason": f"STOP LOSS hit: {pnl_pct:.2f}% loss (limit: -{STOP_LOSS_PCT*100:.0f}%)",
                "type": "stop_loss",
            })
        elif pnl_pct >= TAKE_PROFIT_PCT * 100:
            actions.append({
                "action": "CLOSE",
                "ticker": ticker,
                "reason": f"TAKE PROFIT hit: +{pnl_pct:.2f}% gain (target: +{TAKE_PROFIT_PCT*100:.0f}%)",
                "type": "take_profit",
            })
    
    return {"actions": actions, "positions": len(positions)}

# ── Main Scan ───────────────────────────────────────────────────────────────

def scan_market():
    """
    Full market scan:
    1. Get all stocks from Polygon (instant)
    2. Quick-score all of them (math only, no API calls)
    3. Deep-analyze top 15
    4. Return top 5 with trade recommendations
    """
    # Step 1: Get all stocks
    all_stocks = get_polygon_snapshot()
    if not all_stocks:
        return {"error": "Could not fetch market data", "trades": []}
    
    # Step 2: Quick score all stocks
    scored = []
    for stock in all_stocks:
        result = score_stock(stock)
        if result and result["quick_score"] >= 55:
            scored.append(result)
    
    # Sort by quick score
    scored.sort(key=lambda x: x["quick_score"], reverse=True)
    
    # Step 3: Deep analyze top 15
    top_candidates = scored[:15]
    deep_scored = []
    for candidate in top_candidates:
        try:
            deep = deep_score(candidate["ticker"], candidate)
            deep_scored.append(deep)
        except:
            deep_scored.append(candidate)
    
    # Sort by deep score (or quick score if no deep)
    deep_scored.sort(key=lambda x: x.get("deep_score", x.get("quick_score", 0)), reverse=True)
    
    # Step 4: Get account info for position sizing
    try:
        account = get_alpaca_account()
        portfolio_value = float(account.get("portfolio_value", 100000))
        cash = float(account.get("cash", 100000))
    except:
        portfolio_value = 100000
        cash = 100000
    
    # Step 5: Check current positions
    try:
        current_positions = get_alpaca_positions()
        current_tickers = [p.get("symbol") for p in current_positions] if isinstance(current_positions, list) else []
        num_positions = len(current_tickers)
    except:
        current_tickers = []
        num_positions = 0
    
    # Step 6: Generate trade recommendations
    trades = []
    slots_available = MAX_POSITIONS - num_positions
    
    for stock in deep_scored:
        if len(trades) >= slots_available:
            break
        
        final_score = stock.get("deep_score", stock.get("quick_score", 0))
        ticker = stock["ticker"]
        
        # Skip if already holding
        if ticker in current_tickers:
            continue
        
        # Skip if below minimum score
        if final_score < MIN_SCORE:
            continue
        
        # Position size
        position_value = min(portfolio_value * MAX_POSITION_PCT, cash * 0.9)
        shares = int(position_value / stock["price"]) if stock["price"] > 0 else 0
        
        if shares <= 0:
            continue
        
        trades.append({
            "action": "BUY",
            "ticker": ticker,
            "shares": shares,
            "price": stock["price"],
            "score": final_score,
            "reasons": stock.get("reasons", []),
            "recommendation": stock.get("recommendation"),
            "rec_reasoning": stock.get("rec_reasoning"),
            "stop_loss": round(stock["price"] * (1 - STOP_LOSS_PCT), 2),
            "take_profit": round(stock["price"] * (1 + TAKE_PROFIT_PCT), 2),
            "position_value": round(shares * stock["price"], 2),
        })
    
    # Step 7: Check position management
    mgmt = manage_positions()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "scanned": len(all_stocks),
        "filtered": len(scored),
        "deep_analyzed": len(deep_scored),
        "portfolio_value": portfolio_value,
        "cash": cash,
        "current_positions": num_positions,
        "slots_available": slots_available,
        "new_trades": trades,
        "position_actions": mgmt.get("actions", []),
        "top_10": [{
            "ticker": s["ticker"],
            "price": s["price"],
            "score": s.get("deep_score", s.get("quick_score", 0)),
            "change_pct": s["change_pct"],
            "reasons": s.get("reasons", [])[:2],
        } for s in deep_scored[:10]],
    }

# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    if mode == "scan":
        result = scan_market()
    elif mode == "manage":
        result = manage_positions()
    elif mode == "full":
        result = scan_market()
    else:
        result = {"error": f"Unknown mode: {mode}"}
    
    print(json.dumps(result))
