"""
Momentum Strategy — scores a stock 0-100 based on 12-1 month price momentum.
"""

def score(mom_12_1, mom_1m, avg_volume):
    if mom_12_1 is None: return {"score": 0, "signal": "NO DATA", "reason": "No data"}
    if avg_volume and avg_volume < 100000: return {"score": 0, "signal": "SKIP", "reason": "Too illiquid"}
    
    s = 0
    if mom_12_1 > 50: s += 60
    elif mom_12_1 > 30: s += 50
    elif mom_12_1 > 15: s += 40
    elif mom_12_1 > 5: s += 25
    elif mom_12_1 > 0: s += 10
    
    if mom_1m and mom_1m > 5: s += 25
    elif mom_1m and mom_1m > 0: s += 10
    
    if avg_volume and avg_volume > 2000000: s += 15
    elif avg_volume and avg_volume > 500000: s += 8
    
    s = max(0, min(100, s))
    sig = "STRONG BUY" if s >= 70 else "BUY" if s >= 50 else "NEUTRAL" if s >= 30 else "AVOID"
    return {"score": s, "signal": sig, "reason": f"12mo: +{mom_12_1:.1f}%, 1mo: +{(mom_1m or 0):.1f}%"}
