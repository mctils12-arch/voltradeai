"""
VolTradeAI Manipulation Detection Module
Detects institutional traps, spoofing patterns, and unusual activity.
Uses Alpaca data only — no extra API calls.
"""
import json
import os
import requests
from datetime import datetime, timedelta

ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_DATA = "https://data.alpaca.markets"

def _headers():
    return {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}


def scan_for_manipulation() -> dict:
    """
    Scan for manipulation patterns across the most active stocks.
    Returns alerts for suspicious activity.
    """
    alerts = []
    flagged_tickers = set()
    
    try:
        # Get most active stocks
        resp = requests.get(f"{ALPACA_DATA}/v1beta1/screener/stocks/most-actives?by=volume&top=50",
                           headers=_headers(), timeout=10)
        actives = resp.json().get("most_actives", [])
    except Exception:
        return {"alerts": [], "error": "Could not fetch market data"}
    
    # Get snapshots for all active stocks
    tickers = [s["symbol"] for s in actives if s.get("symbol")][:50]
    if not tickers:
        return {"alerts": [], "scanned": 0}
    
    snapshots = {}
    try:
        resp = requests.get(f"{ALPACA_DATA}/v2/stocks/snapshots?symbols={','.join(tickers)}&feed=sip",
                           headers=_headers(), timeout=15)
        snapshots = resp.json()
    except Exception:
        return {"alerts": [], "error": "Could not fetch snapshots"}
    
    for ticker in tickers:
        snap = snapshots.get(ticker, {})
        bar = snap.get("dailyBar", {})
        prev = snap.get("prevDailyBar", {})
        quote = snap.get("latestQuote", {})
        
        c = float(bar.get("c", 0))
        o = float(bar.get("o", c))
        h = float(bar.get("h", c))
        l = float(bar.get("l", c))
        v = int(bar.get("v", 0))
        pc = float(prev.get("c", c))
        pv = int(prev.get("v", 1))
        
        if c <= 0 or pv <= 0:
            continue
        
        change_pct = ((c - pc) / pc) * 100
        vol_ratio = v / pv
        day_range = ((h - l) / c) * 100 if c > 0 else 0
        
        # ── Pattern 1: Volume Anomaly (high volume, flat price) ──
        # Institutions accumulating/distributing quietly
        if vol_ratio > 3.0 and abs(change_pct) < 1.0:
            alerts.append({
                "ticker": ticker,
                "pattern": "volume_anomaly",
                "severity": "high" if vol_ratio > 5 else "medium",
                "message": f"{ticker}: Volume {vol_ratio:.1f}x normal but price flat ({change_pct:+.1f}%) — possible institutional accumulation/distribution",
                "vol_ratio": round(vol_ratio, 1),
                "change_pct": round(change_pct, 2),
            })
            flagged_tickers.add(ticker)
        
        # ── Pattern 2: Wide Bid-Ask Spread ──
        # Low liquidity = easier to manipulate
        bid = float(quote.get("bp", 0))
        ask = float(quote.get("ap", 0))
        if bid > 0 and ask > 0:
            spread_pct = ((ask - bid) / bid) * 100
            if spread_pct > 1.0 and v > 500000:
                alerts.append({
                    "ticker": ticker,
                    "pattern": "wide_spread",
                    "severity": "medium",
                    "message": f"{ticker}: Wide bid-ask spread ({spread_pct:.2f}%) despite high volume — possible manipulation risk",
                    "spread_pct": round(spread_pct, 2),
                })
        
        # ── Pattern 3: Price Rejection (stop-loss hunt) ──
        # Spiked up/down then reversed — institutions triggering stops
        if day_range > 5:
            # Big intraday range
            upper_wick = ((h - max(c, o)) / c) * 100 if c > 0 else 0
            lower_wick = ((min(c, o) - l) / c) * 100 if c > 0 else 0
            
            if upper_wick > 3 and change_pct < 0:
                alerts.append({
                    "ticker": ticker,
                    "pattern": "price_rejection_top",
                    "severity": "high",
                    "message": f"{ticker}: Spiked to ${h:.2f} then rejected — possible bull trap / stop hunt (closed {change_pct:+.1f}%)",
                    "high": h, "close": c, "wick_pct": round(upper_wick, 1),
                })
                flagged_tickers.add(ticker)
            
            if lower_wick > 3 and change_pct > 0:
                alerts.append({
                    "ticker": ticker,
                    "pattern": "price_rejection_bottom",
                    "severity": "medium",
                    "message": f"{ticker}: Dipped to ${l:.2f} then recovered — possible bear trap / stop hunt (closed {change_pct:+.1f}%)",
                    "low": l, "close": c, "wick_pct": round(lower_wick, 1),
                })
        
        # ── Pattern 4: Extreme Move on Low Volume ──
        # Price moved big but volume is below average — fake move
        if abs(change_pct) > 5 and vol_ratio < 0.5:
            alerts.append({
                "ticker": ticker,
                "pattern": "low_vol_spike",
                "severity": "high",
                "message": f"{ticker}: {change_pct:+.1f}% move on only {vol_ratio:.1f}x average volume — suspicious, likely to reverse",
                "change_pct": round(change_pct, 2),
                "vol_ratio": round(vol_ratio, 1),
            })
            flagged_tickers.add(ticker)
        
        # ── Pattern 5: Sell-the-News (earnings day check) ──
        # Big positive move followed by reversal — check if near earnings
        if change_pct < -3 and vol_ratio > 2:
            # Sharp drop on high volume — could be post-earnings dump
            alerts.append({
                "ticker": ticker,
                "pattern": "high_vol_drop",
                "severity": "medium",
                "message": f"{ticker}: Sharp drop {change_pct:.1f}% on {vol_ratio:.1f}x volume — avoid catching falling knife",
                "change_pct": round(change_pct, 2),
            })
    
    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda a: severity_order.get(a.get("severity", "low"), 2))
    
    return {
        "alerts": alerts[:20],  # Top 20 alerts
        "flagged_tickers": list(flagged_tickers),
        "scanned": len(tickers),
        "timestamp": datetime.now().isoformat(),
    }


def is_ticker_flagged(ticker: str) -> dict:
    """Quick check if a specific ticker has manipulation signals."""
    try:
        resp = requests.get(f"{ALPACA_DATA}/v2/stocks/{ticker}/snapshot?feed=sip", headers=_headers(), timeout=5)
        snap = resp.json()
        bar = snap.get("dailyBar", {})
        prev = snap.get("prevDailyBar", {})
        quote = snap.get("latestQuote", {})
        
        c = float(bar.get("c", 0))
        pc = float(prev.get("c", c))
        v = int(bar.get("v", 0))
        pv = int(prev.get("v", 1))
        
        flags = []
        vol_ratio = v / pv if pv > 0 else 1
        change_pct = ((c - pc) / pc * 100) if pc > 0 else 0
        
        if vol_ratio > 3 and abs(change_pct) < 1:
            flags.append("volume_anomaly")
        if abs(change_pct) > 5 and vol_ratio < 0.5:
            flags.append("low_vol_spike")
        
        bid = float(quote.get("bp", 0))
        ask = float(quote.get("ap", 0))
        if bid > 0 and ask > 0 and ((ask - bid) / bid * 100) > 1:
            flags.append("wide_spread")
        
        return {"flagged": len(flags) > 0, "flags": flags, "vol_ratio": round(vol_ratio, 1), "change_pct": round(change_pct, 2)}
    except Exception:
        return {"flagged": False, "flags": [], "error": "check_failed"}


if __name__ == "__main__":
    print("=== Manipulation Scan ===")
    result = scan_for_manipulation()
    print(json.dumps(result, indent=2))
