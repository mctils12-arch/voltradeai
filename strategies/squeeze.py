"""
Short Squeeze Strategy — scores squeeze potential 0-100.
"""

def score(short_pct, days_to_cover, sentiment_score, volume_ratio, reddit_buzz):
    s = 0
    sp = short_pct or 0
    dc = days_to_cover or 0
    sent = sentiment_score or 50
    vr = volume_ratio or 1
    rb = reddit_buzz or 0
    
    if sp >= 30: s += 40
    elif sp >= 20: s += 30
    elif sp >= 15: s += 20
    elif sp >= 10: s += 10
    
    if dc >= 10: s += 20
    elif dc >= 5: s += 12
    elif dc >= 3: s += 5
    
    if sent >= 75: s += 20
    elif sent >= 60: s += 10
    
    if vr >= 3: s += 10
    elif vr >= 2: s += 5
    
    if rb >= 8: s += 10
    elif rb >= 5: s += 5
    
    s = max(0, min(100, s))
    sig = "SQUEEZE ALERT" if s >= 70 else "POSSIBLE" if s >= 50 else "LOW RISK" if s >= 30 else "NONE"
    return {"score": s, "signal": sig, "reason": f"Short: {sp:.1f}%, Cover: {dc:.0f}d, Sentiment: {sent:.0f}"}
