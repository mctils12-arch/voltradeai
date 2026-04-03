"""
Mean Reversion Strategy — scores oversold stocks 0-100 for bounce potential.
"""

def score(rsi, change_pct_5d, volume_ratio):
    if rsi is None: return {"score": 0, "signal": "NO DATA", "reason": "No data"}
    
    s = 0
    if rsi < 20: s += 40
    elif rsi < 30: s += 30
    elif rsi < 40: s += 15
    elif rsi > 70: s -= 15
    
    if change_pct_5d and change_pct_5d < -10: s += 30
    elif change_pct_5d and change_pct_5d < -5: s += 20
    elif change_pct_5d and change_pct_5d < -3: s += 10
    
    if volume_ratio and volume_ratio > 2: s += 20
    elif volume_ratio and volume_ratio > 1.5: s += 10
    
    s = max(0, min(100, s))
    sig = "STRONG BUY" if s >= 65 else "BUY" if s >= 45 else "WATCH" if s >= 25 else "NO EDGE"
    return {"score": s, "signal": sig, "reason": f"RSI: {rsi:.0f}, 5d drop: {(change_pct_5d or 0):.1f}%, vol: {(volume_ratio or 1):.1f}x"}
