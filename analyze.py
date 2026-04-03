#!/usr/bin/env python3
"""
VolTrade — Volatility Trading Algorithm Engine
Based on concepts from "Trading Volatility" by Colin Bennett.
Real-time options + fundamental data via Yahoo Finance.
"""

import sys
import json
import math
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta


# ── Math helpers ──────────────────────────────────────────────────────────────

def norm_cdf(x):
    from scipy.stats import norm
    return float(norm.cdf(x))

def norm_pdf(x):
    from scipy.stats import norm
    return float(norm.pdf(x))

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0.001 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return {"price": round(intrinsic, 2), "delta": 1.0 if S > K else 0.0,
                "gamma": 0, "vega": 0, "theta": 0, "prob_itm": 100.0 if S > K else 0.0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = norm_cdf(d1)
        prob_itm = norm_cdf(d2) * 100
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1
        prob_itm = norm_cdf(-d2) * 100
    gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * norm_pdf(d1) * math.sqrt(T) / 100
    theta = (-(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * (norm_cdf(d2) if option_type == 'call' else norm_cdf(-d2))) / 365
    return {
        "price":    round(max(price, 0), 4),
        "delta":    round(delta, 4),
        "gamma":    round(gamma, 6),
        "vega":     round(vega, 4),
        "theta":    round(theta, 4),
        "prob_itm": round(prob_itm, 2),
    }

def realized_vol(hist, window=20):
    closes = list(hist['Close'].dropna())
    if len(closes) < window + 1:
        return None
    closes = closes[-(window + 1):]
    returns = [math.log(closes[i+1] / closes[i]) for i in range(len(closes) - 1)]
    mean = sum(returns) / len(returns)
    var  = sum((r - mean)**2 for r in returns) / (len(returns) - 1)
    return round(math.sqrt(var * 252) * 100, 2)

def bid_ask_quality(bid, ask):
    mid = (bid + ask) / 2
    if mid <= 0:
        return 0
    spread_pct = (ask - bid) / mid
    return max(0.0, 1.0 - spread_pct)

def composite_score(iv_edge, baq, prob_profit, risk_reward):
    """Bennett VRP scoring: 40% IV edge, 25% bid-ask, 25% prob profit, 10% R/R."""
    return round((iv_edge * 0.40 + baq * 0.25 + prob_profit * 0.25 + risk_reward * 0.10) * 100, 1)

def days_to_expiry(exp_str):
    exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
    return max((exp_date - datetime.now()).days, 0)

def T_from_days(d):
    return max(d / 365.0, 1 / 365.0)


# ── Historical volume & Bennett metrics ───────────────────────────────────────

def compute_volume_metrics(hist):
    """
    Bennett-inspired volume & volatility metrics from price history.
    Returns: avg_volume, vol_ratio (recent/avg), iv_percentile proxy via HV range.
    """
    if hist.empty or len(hist) < 30:
        return {}

    volumes = list(hist['Volume'].dropna())
    closes  = list(hist['Close'].dropna())

    # Average daily volume (90-day)
    avg_vol_90 = sum(volumes[-90:]) / len(volumes[-90:]) if len(volumes) >= 90 else sum(volumes) / len(volumes)
    # Recent 5-day volume vs 90-day average (volume spike indicator)
    recent_vol_5 = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else avg_vol_90
    vol_ratio = round(recent_vol_5 / avg_vol_90, 2) if avg_vol_90 > 0 else 1.0

    # HV over multiple windows for percentile estimation
    def _hv(n):
        if len(closes) < n + 1:
            return None
        c = closes[-(n + 1):]
        rets = [math.log(c[i+1] / c[i]) for i in range(len(c) - 1)]
        mean = sum(rets) / len(rets)
        var  = sum((r - mean)**2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var * 252) * 100

    hv10  = _hv(10)
    hv20  = _hv(20)
    hv30  = _hv(30)
    hv60  = _hv(60)

    # Rolling 252-day HV to estimate percentile of current 20d HV
    hv_series = []
    for i in range(20, min(len(closes), 252)):
        chunk = closes[i - 20:i + 1]
        if len(chunk) < 21:
            continue
        rets = [math.log(chunk[j+1] / chunk[j]) for j in range(len(chunk) - 1)]
        mean = sum(rets) / len(rets)
        var  = sum((r - mean)**2 for r in rets) / (len(rets) - 1)
        hv_series.append(math.sqrt(var * 252) * 100)

    hv_pct = None
    if hv_series and hv20 is not None:
        below = sum(1 for h in hv_series if h <= hv20)
        hv_pct = round(below / len(hv_series) * 100, 1)

    # Term structure slope: (HV30 - HV10) / HV20 — measures vol term structure steepness
    ts_slope = None
    if hv10 and hv30 and hv20:
        ts_slope = round((hv30 - hv10) / (hv20 + 0.001), 3)

    # Vol-of-vol: std dev of daily returns' absolute values (roughness of vol)
    if len(closes) >= 22:
        abs_rets = [abs(math.log(closes[i+1] / closes[i])) for i in range(len(closes) - 1)]
        recent   = abs_rets[-20:]
        mean_r   = sum(recent) / len(recent)
        vov = round(math.sqrt(sum((r - mean_r)**2 for r in recent) / (len(recent) - 1)) * math.sqrt(252) * 100, 2)
    else:
        vov = None

    # Average True Range % (ATR%) — 14-day
    highs  = list(hist['High'].dropna())
    lows   = list(hist['Low'].dropna())
    if len(highs) >= 15 and len(lows) >= 15 and len(closes) >= 15:
        trs = []
        for i in range(1, 15):
            tr = max(highs[-15+i] - lows[-15+i],
                     abs(highs[-15+i] - closes[-15+i-1]),
                     abs(lows[-15+i]  - closes[-15+i-1]))
            trs.append(tr)
        atr_pct = round((sum(trs) / len(trs)) / closes[-1] * 100, 2)
    else:
        atr_pct = None

    result = {}
    if avg_vol_90  is not None: result["avg_volume_90d"]    = int(avg_vol_90)
    if vol_ratio   is not None: result["volume_ratio_5d"]   = vol_ratio
    if hv10        is not None: result["hv10"]              = round(hv10, 1)
    if hv20        is not None: result["hv20"]              = round(hv20, 1)
    if hv30        is not None: result["hv30"]              = round(hv30, 1)
    if hv60        is not None: result["hv60"]              = round(hv60, 1)
    if hv_pct      is not None: result["hv_percentile"]     = hv_pct
    if ts_slope    is not None: result["term_structure_slope"] = ts_slope
    if vov         is not None: result["vol_of_vol"]        = vov
    if atr_pct     is not None: result["atr_pct"]           = atr_pct
    return result


# ── Valuation assessment ───────────────────────────────────────────────────────

# Sector median P/E benchmarks (approximate; used when yfinance sector P/E unavailable)
SECTOR_PE_MEDIANS = {
    "Technology":            28.0,
    "Consumer Cyclical":     22.0,
    "Communication Services": 20.0,
    "Healthcare":            23.0,
    "Financial Services":    14.0,
    "Basic Materials":       15.0,
    "Industrials":           20.0,
    "Consumer Defensive":    20.0,
    "Energy":                13.0,
    "Utilities":             17.0,
    "Real Estate":           30.0,
    "Unknown":               20.0,
}

def assess_valuation(info, spot):
    """
    Multi-factor valuation assessment combining:
    - P/E vs sector median
    - PEG ratio (P/E divided by growth)
    - Price-to-Book
    - Price-to-Sales
    - DCF discount (using analyst target price as proxy)
    Returns: dict with overall verdict, score 0-100, and per-factor breakdown.
    """
    signals = []
    scores  = []

    sector = info.get("sector") or "Unknown"
    sector_pe = SECTOR_PE_MEDIANS.get(sector, 20.0)

    # ── Factor 1: Forward P/E vs sector ────────────────────────────────────────
    fpe = info.get("forwardPE") or info.get("trailingPE")
    if fpe and fpe > 0:
        ratio = fpe / sector_pe
        if ratio < 0.80:
            score = 85; label = "cheap"
        elif ratio < 0.95:
            score = 65; label = "fair-low"
        elif ratio < 1.10:
            score = 50; label = "fair"
        elif ratio < 1.35:
            score = 35; label = "fair-high"
        else:
            score = 15; label = "expensive"
        scores.append(score)
        signals.append({
            "factor": "P/E vs Sector",
            "value":  round(fpe, 1),
            "benchmark": sector_pe,
            "verdict": label,
            "detail": f"P/E {fpe:.1f} vs {sector} median {sector_pe:.0f}x ({ratio:.2f}x)"
        })

    # ── Factor 2: PEG ratio ─────────────────────────────────────────────────────
    peg = info.get("pegRatio")
    if peg and peg > 0:
        if peg < 0.75:
            score = 90; label = "very cheap"
        elif peg < 1.0:
            score = 70; label = "cheap"
        elif peg < 1.5:
            score = 50; label = "fair"
        elif peg < 2.5:
            score = 30; label = "expensive"
        else:
            score = 10; label = "very expensive"
        scores.append(score)
        signals.append({
            "factor":  "PEG Ratio",
            "value":   round(peg, 2),
            "benchmark": 1.0,
            "verdict": label,
            "detail":  f"PEG {peg:.2f} (1.0 = fairly valued)"
        })

    # ── Factor 3: Price-to-Book ─────────────────────────────────────────────────
    pb = info.get("priceToBook")
    if pb and pb > 0:
        if pb < 1.0:
            score = 85; label = "cheap"
        elif pb < 2.0:
            score = 65; label = "fair"
        elif pb < 4.0:
            score = 45; label = "fair-high"
        elif pb < 8.0:
            score = 25; label = "expensive"
        else:
            score = 10; label = "very expensive"
        scores.append(score)
        signals.append({
            "factor":  "Price-to-Book",
            "value":   round(pb, 2),
            "benchmark": 2.0,
            "verdict": label,
            "detail":  f"P/B {pb:.2f}x"
        })

    # ── Factor 4: Price-to-Sales ────────────────────────────────────────────────
    ps = info.get("priceToSalesTrailing12Months")
    if ps and ps > 0:
        if ps < 1.0:
            score = 85; label = "cheap"
        elif ps < 3.0:
            score = 60; label = "fair"
        elif ps < 8.0:
            score = 35; label = "expensive"
        else:
            score = 10; label = "very expensive"
        scores.append(score)
        signals.append({
            "factor":  "Price-to-Sales",
            "value":   round(ps, 2),
            "benchmark": 3.0,
            "verdict": label,
            "detail":  f"P/S {ps:.2f}x (trailing 12m)"
        })

    # ── Factor 5: Analyst target vs current price (DCF proxy) ──────────────────
    target = info.get("targetMeanPrice")
    if target and target > 0 and spot > 0:
        upside = (target - spot) / spot * 100
        if upside > 25:
            score = 90; label = "strongly undervalued"
        elif upside > 10:
            score = 72; label = "undervalued"
        elif upside > -5:
            score = 50; label = "fairly valued"
        elif upside > -15:
            score = 30; label = "overvalued"
        else:
            score = 10; label = "strongly overvalued"
        scores.append(score)
        signals.append({
            "factor":  "Analyst Target",
            "value":   round(target, 2),
            "benchmark": round(spot, 2),
            "verdict": label,
            "detail":  f"Consensus target ${target:.2f} ({'+' if upside >= 0 else ''}{upside:.1f}% vs current)"
        })

    # ── Factor 6: Earnings yield vs risk-free ───────────────────────────────────
    eps = info.get("trailingEps") or info.get("forwardEps")
    if eps and eps > 0 and spot > 0:
        earn_yield = eps / spot * 100
        rf_rate    = 5.0  # approximate 10Y treasury
        excess     = earn_yield - rf_rate
        if excess > 3:
            score = 85; label = "cheap vs bonds"
        elif excess > 1:
            score = 65; label = "slightly cheap vs bonds"
        elif excess > -1:
            score = 50; label = "fairly valued vs bonds"
        elif excess > -3:
            score = 30; label = "expensive vs bonds"
        else:
            score = 10; label = "very expensive vs bonds"
        scores.append(score)
        signals.append({
            "factor":  "Earnings Yield",
            "value":   round(earn_yield, 2),
            "benchmark": rf_rate,
            "verdict": label,
            "detail":  f"Earnings yield {earn_yield:.2f}% vs {rf_rate:.1f}% risk-free"
        })

    if not scores:
        return None   # no valuation data available

    overall_score = round(sum(scores) / len(scores), 1)

    if overall_score >= 72:
        verdict = "Undervalued"
        color   = "green"
    elif overall_score >= 52:
        verdict = "Fairly Valued"
        color   = "neutral"
    else:
        verdict = "Overvalued"
        color   = "red"

    return {
        "verdict":       verdict,
        "color":         color,
        "score":         overall_score,
        "sector":        sector,
        "signals":       signals,
        "factors_count": len(signals),
    }


# ── Strike selection helpers ───────────────────────────────────────────────────

def nearest_strikes(df, spot, n_each_side=6, pct_band=0.08):
    lo = spot * (1 - pct_band)
    hi = spot * (1 + pct_band)
    in_band = df[(df['strike'] >= lo) & (df['strike'] <= hi)].copy()
    below = in_band[in_band['strike'] <= spot].nlargest(n_each_side, 'strike')
    above = in_band[in_band['strike'] >  spot].nsmallest(n_each_side, 'strike')
    import pandas as pd
    return pd.concat([below, above]).sort_values('strike').reset_index(drop=True)

def find_atm_row(df, spot):
    if df.empty:
        return None
    df = df.copy()
    df['_dist'] = abs(df['strike'] - spot)
    return df.nsmallest(1, '_dist').iloc[0]


# ── Skew calculation ───────────────────────────────────────────────────────────

def compute_skew(calls, puts, spot, T):
    """
    Bennett put-call skew: IV(0.9*spot put) - IV(1.1*spot call).
    Positive = put skew (typical), negative = call skew (unusual, bullish premium).
    """
    otm_put_target  = spot * 0.90
    otm_call_target = spot * 1.10

    if puts.empty or calls.empty:
        return None

    puts_copy  = puts.copy()
    calls_copy = calls.copy()
    puts_copy['_dist']  = abs(puts_copy['strike']  - otm_put_target)
    calls_copy['_dist'] = abs(calls_copy['strike'] - otm_call_target)

    p_row = puts_copy.nsmallest(1, '_dist')
    c_row = calls_copy.nsmallest(1, '_dist')

    if p_row.empty or c_row.empty:
        return None

    iv_put  = float(p_row.iloc[0].get('impliedVolatility', 0) or 0) * 100
    iv_call = float(c_row.iloc[0].get('impliedVolatility', 0) or 0) * 100

    if iv_put <= 0 or iv_call <= 0:
        return None

    return round(iv_put - iv_call, 2)


# ── Spread builders ────────────────────────────────────────────────────────────

def build_bull_call_spread(calls, spot, T, r, rv20, exp):
    spreads = []
    near = nearest_strikes(calls, spot, n_each_side=5, pct_band=0.06)
    if near.empty:
        return spreads

    for i, row in near.iterrows():
        K_long = float(row['strike'])
        bid_l  = float(row.get('bid', 0) or 0)
        ask_l  = float(row.get('ask', 0) or 0)
        iv_l   = float(row.get('impliedVolatility', 0) or 0)
        if ask_l <= 0 or iv_l <= 0:
            continue

        candidates = calls[calls['strike'] > K_long].nsmallest(3, 'strike')
        for _, sc_row in candidates.iterrows():
            K_short = float(sc_row['strike'])
            if K_short > spot * 1.08:
                break
            bid_s = float(sc_row.get('bid', 0) or 0)
            if bid_s <= 0:
                continue

            net_debit    = round((ask_l + bid_l) / 2 - bid_s, 2)
            spread_width = round(K_short - K_long, 2)
            max_profit   = round(spread_width - net_debit, 2)
            if net_debit <= 0 or max_profit <= 0:
                continue

            bs   = black_scholes(spot, K_long, T, r, iv_l, 'call')
            baq  = bid_ask_quality(bid_l, ask_l)
            iv_e = max(0.0, min((iv_l * 100 - rv20) / 100.0, 1.0))
            pp   = bs["prob_itm"] / 100
            rr   = min(max_profit / (net_debit + 0.01), 5) / 5

            sc = composite_score(iv_e, baq, pp, rr)
            spreads.append({
                "type":          "Bull Call Spread",
                "expiry":        exp,
                "days_to_expiry": round(T * 365),
                "long_strike":   round(K_long, 2),
                "short_strike":  round(K_short, 2),
                "net_debit":     net_debit,
                "max_profit":    max_profit,
                "max_loss":      net_debit,
                "prob_profit":   round(bs["prob_itm"], 1),
                "iv":            round(iv_l * 100, 1),
                "delta":         bs["delta"],
                "score":         sc,
                "description":   f"Buy ${K_long:.0f}C / Sell ${K_short:.0f}C — expires {exp}",
            })
    return spreads


def build_bear_put_spread(puts, spot, T, r, rv20, exp):
    spreads = []
    near = nearest_strikes(puts, spot, n_each_side=5, pct_band=0.06)
    if near.empty:
        return spreads

    for i, row in near.iterrows():
        K_long = float(row['strike'])
        bid_l  = float(row.get('bid', 0) or 0)
        ask_l  = float(row.get('ask', 0) or 0)
        iv_l   = float(row.get('impliedVolatility', 0) or 0)
        if ask_l <= 0 or iv_l <= 0:
            continue

        candidates = puts[puts['strike'] < K_long].nlargest(3, 'strike')
        for _, sc_row in candidates.iterrows():
            K_short = float(sc_row['strike'])
            if K_short < spot * 0.92:
                break
            bid_s = float(sc_row.get('bid', 0) or 0)
            if bid_s <= 0:
                continue

            net_debit    = round((ask_l + bid_l) / 2 - bid_s, 2)
            spread_width = round(K_long - K_short, 2)
            max_profit   = round(spread_width - net_debit, 2)
            if net_debit <= 0 or max_profit <= 0:
                continue

            bs   = black_scholes(spot, K_long, T, r, iv_l, 'put')
            baq  = bid_ask_quality(bid_l, ask_l)
            iv_e = max(0.0, min((iv_l * 100 - rv20) / 100.0, 1.0))
            pp   = bs["prob_itm"] / 100
            rr   = min(max_profit / (net_debit + 0.01), 5) / 5

            sc = composite_score(iv_e, baq, pp, rr)
            spreads.append({
                "type":          "Bear Put Spread",
                "expiry":        exp,
                "days_to_expiry": round(T * 365),
                "long_strike":   round(K_long, 2),
                "short_strike":  round(K_short, 2),
                "net_debit":     net_debit,
                "max_profit":    max_profit,
                "max_loss":      net_debit,
                "prob_profit":   round(bs["prob_itm"], 1),
                "iv":            round(iv_l * 100, 1),
                "delta":         bs["delta"],
                "score":         sc,
                "description":   f"Buy ${K_long:.0f}P / Sell ${K_short:.0f}P — expires {exp}",
            })
    return spreads


def build_short_straddle(calls, puts, spot, T, r, rv20, exp):
    spreads = []
    atm_c = find_atm_row(calls, spot)
    atm_p = find_atm_row(puts,  spot)
    if atm_c is None or atm_p is None:
        return spreads

    K        = float(atm_c['strike'])
    call_bid = float(atm_c.get('bid', 0) or 0)
    put_bid  = float(atm_p.get('bid', 0) or 0)
    iv_c     = float(atm_c.get('impliedVolatility', 0) or 0)

    if call_bid <= 0 or put_bid <= 0 or iv_c <= 0:
        return spreads

    total_credit = round(call_bid + put_bid, 2)
    bep_up   = round(K + total_credit, 2)
    bep_down = round(K - total_credit, 2)
    bs_up    = black_scholes(spot, bep_up,   T, r, iv_c, 'call')
    bs_down  = black_scholes(spot, bep_down, T, r, iv_c, 'put')
    pp = max(0, 1.0 - bs_up["prob_itm"] / 100 - bs_down["prob_itm"] / 100)

    baq  = bid_ask_quality(call_bid, float(atm_c.get('ask', call_bid * 1.1) or call_bid * 1.1))
    iv_e = max(0.0, min((iv_c * 100 - rv20) / 100.0, 1.0))
    rr   = min(total_credit / (K * 0.05 + 0.01), 1.0)
    sc   = composite_score(iv_e, baq, pp, rr)

    spreads.append({
        "type":          "Short Straddle",
        "expiry":        exp,
        "days_to_expiry": round(T * 365),
        "long_strike":   K,
        "short_strike":  K,
        "net_debit":     -total_credit,
        "max_profit":    total_credit,
        "max_loss":      None,
        "prob_profit":   round(pp * 100, 1),
        "iv":            round(iv_c * 100, 1),
        "delta":         0.0,
        "score":         sc,
        "description":   (
            f"Sell ${K:.0f}C + ${K:.0f}P for ${total_credit:.2f} credit — "
            f"profit between ${bep_down:.0f}–${bep_up:.0f}"
        ),
    })
    return spreads


def build_iron_condor(calls, puts, spot, T, r, rv20, exp):
    spreads = []
    atm_c_row = find_atm_row(calls, spot)
    iv_approx = float(atm_c_row.get('impliedVolatility', 0.20) or 0.20) if atm_c_row is not None else 0.20
    one_sd = spot * iv_approx * math.sqrt(T)

    short_call_target = spot + one_sd * 0.8
    short_put_target  = spot - one_sd * 0.8

    call_candidates = calls[calls['strike'] > spot].copy()
    put_candidates  = puts[puts['strike']  < spot].copy()
    if call_candidates.empty or put_candidates.empty:
        return spreads

    call_candidates['_dist'] = abs(call_candidates['strike'] - short_call_target)
    put_candidates['_dist']  = abs(put_candidates['strike']  - short_put_target)

    sc_row = call_candidates.nsmallest(1, '_dist').iloc[0]
    sp_row = put_candidates.nsmallest(1, '_dist').iloc[0]

    K_sc = float(sc_row['strike'])
    K_sp = float(sp_row['strike'])

    lc_candidates = calls[calls['strike'] > K_sc].nsmallest(1, 'strike')
    lp_candidates = puts[puts['strike']  < K_sp].nlargest(1, 'strike')
    if lc_candidates.empty or lp_candidates.empty:
        return spreads

    lc_row = lc_candidates.iloc[0]
    lp_row = lp_candidates.iloc[0]
    K_lc = float(lc_row['strike'])
    K_lp = float(lp_row['strike'])

    sc_bid = float(sc_row.get('bid', 0) or 0)
    sp_bid = float(sp_row.get('bid', 0) or 0)
    lc_ask = float(lc_row.get('ask', 0) or 0)
    lp_ask = float(lp_row.get('ask', 0) or 0)

    if sc_bid <= 0 or sp_bid <= 0 or lc_ask <= 0 or lp_ask <= 0:
        return spreads

    net_credit = round((sc_bid + sp_bid) - (lc_ask + lp_ask), 2)
    call_width = round(K_lc - K_sc, 2)
    put_width  = round(K_sp - K_lp, 2)
    max_loss   = round(max(call_width, put_width) - net_credit, 2)

    if net_credit <= 0 or max_loss <= 0:
        return spreads

    iv_sc = float(sc_row.get('impliedVolatility', iv_approx) or iv_approx)
    bs_sc = black_scholes(spot, K_sc, T, r, iv_sc, 'call')
    iv_sp = float(sp_row.get('impliedVolatility', iv_approx) or iv_approx)
    bs_sp = black_scholes(spot, K_sp, T, r, iv_sp, 'put')
    pp = max(0.0, 1.0 - bs_sc["prob_itm"] / 100 - bs_sp["prob_itm"] / 100)

    baq  = bid_ask_quality(sc_bid, float(sc_row.get('ask', sc_bid * 1.1) or sc_bid * 1.1))
    iv_e = max(0.0, min((iv_sc * 100 - rv20) / 100.0, 1.0))
    rr   = min(net_credit / (max_loss + 0.01), 1.0)
    sc   = composite_score(iv_e, baq, pp, rr)

    spreads.append({
        "type":          "Iron Condor",
        "expiry":        exp,
        "days_to_expiry": round(T * 365),
        "long_strike":   K_lp,
        "short_strike":  K_sc,
        "net_debit":     -net_credit,
        "max_profit":    net_credit,
        "max_loss":      max_loss,
        "prob_profit":   round(pp * 100, 1),
        "iv":            round(iv_sc * 100, 1),
        "delta":         0.0,
        "score":         sc,
        "description":   (
            f"Sell ${K_sp:.0f}P/${K_sc:.0f}C, Buy ${K_lp:.0f}P/${K_lc:.0f}C — "
            f"profit if stays ${K_sp:.0f}–${K_sc:.0f}"
        ),
    })
    return spreads


# ── Module 1: Sentiment Analysis ─────────────────────────────────────────────
# Requires: requests (stdlib), pytrends (pip install pytrends)

def get_sentiment(ticker):
    """
    Composite sentiment score (0-100) from StockTwits, Reddit, and Google Trends.
    Returns: {score, signal, bull_pct, reddit_buzz, trend_spike, contrarian_flag}
    """
    import requests

    # ── Default fallback neutral values ──────────────────────────────────────
    bull_pct     = 50.0
    reddit_buzz  = 0.5
    trend_spike  = 1.0

    # ── 1. StockTwits ─────────────────────────────────────────────────────────
    try:
        st_url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        st_resp = requests.get(st_url, timeout=8,
                               headers={'User-Agent': 'VolTrade/1.0'})
        if st_resp.status_code == 200:
            st_data = st_resp.json()
            messages = st_data.get('messages', [])
            # Last 30 messages
            messages = messages[:30]
            bull_count = sum(
                1 for m in messages
                if isinstance(m.get('entities', {}).get('sentiment'), dict)
                and m['entities']['sentiment'].get('basic', '') == 'Bullish'
            )
            bear_count = sum(
                1 for m in messages
                if isinstance(m.get('entities', {}).get('sentiment'), dict)
                and m['entities']['sentiment'].get('basic', '') == 'Bearish'
            )
            total_sentiment = bull_count + bear_count
            if total_sentiment > 0:
                bull_pct = round(bull_count / total_sentiment * 100, 1)
            else:
                bull_pct = 50.0
    except Exception:
        bull_pct = 50.0

    # ── 2. Reddit ─────────────────────────────────────────────────────────────
    try:
        reddit_url = (
            f"https://www.reddit.com/search.json?"
            f"q={ticker}&sort=new&limit=25&type=link"
        )
        rd_resp = requests.get(reddit_url, timeout=8,
                               headers={'User-Agent': 'VolTrade/1.0'})
        if rd_resp.status_code == 200:
            rd_data = rd_resp.json()
            posts = rd_data.get('data', {}).get('children', [])
            ticker_upper = ticker.upper()
            mention_count = sum(
                1 for p in posts
                if ticker_upper in p.get('data', {}).get('title', '').upper()
            )
            # buzz_score: fraction of 25 posts mentioning ticker
            reddit_buzz = round(mention_count / 25.0, 3)
        else:
            reddit_buzz = 0.5
    except Exception:
        reddit_buzz = 0.5

    # ── 3. Google Trends ─────────────────────────────────────────────────────
    try:
        # pip install pytrends
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        pytrends.build_payload([ticker], timeframe='now 30-d', geo='US')
        iot = pytrends.interest_over_time()
        if not iot.empty and ticker in iot.columns:
            vals = iot[ticker].dropna().tolist()
            if len(vals) >= 14:
                last7  = sum(vals[-7:])  / 7.0
                prior7 = sum(vals[-14:-7]) / 7.0
                if prior7 > 0:
                    trend_spike = round(last7 / prior7, 3)
                else:
                    trend_spike = 1.0
            else:
                trend_spike = 1.0
        else:
            trend_spike = 1.0
    except ImportError:
        # pytrends not installed — use neutral
        trend_spike = 1.0
    except Exception:
        trend_spike = 1.0

    # ── 4. Composite score ────────────────────────────────────────────────────
    # bull_pct (40%), reddit_buzz normalized to 0-100 (30%), trend_spike (30%)
    # trend_spike: cap at 3x (=100), 1x = 33, below 1 tapers toward 0
    bull_component   = bull_pct                                     # already 0-100
    reddit_component = min(reddit_buzz * 100 / 0.5, 100.0)         # normalize: 0.5 buzz = 100
    reddit_component = max(reddit_component, 0.0)
    # trend_spike: 1.0 → 33, 2.0 → 67, 3.0+ → 100, <1 proportionally lower
    trend_component  = min(max((trend_spike / 3.0) * 100.0, 0.0), 100.0)

    score = round(
        bull_component   * 0.40 +
        reddit_component * 0.30 +
        trend_component  * 0.30,
        1
    )

    # ── 5. Signal string ──────────────────────────────────────────────────────
    if score >= 70:
        signal = "Extremely Bullish"
    elif score >= 55:
        signal = "Bullish"
    elif score >= 45:
        signal = "Neutral"
    elif score >= 30:
        signal = "Bearish"
    else:
        signal = "Extremely Bearish"

    return {
        "score":          score,
        "signal":         signal,
        "bull_pct":       bull_pct,
        "reddit_buzz":    reddit_buzz,
        "trend_spike":    trend_spike,
        "contrarian_flag": None,   # filled in after short_pct_float is available
    }


def _apply_contrarian_flag(sentiment, short_pct_float, fundamentals_strong):
    """
    Apply contrarian flag after short % data is available.
    Mutates sentiment dict in-place and returns it.
    """
    score = sentiment.get('score', 50)
    spf   = short_pct_float or 0
    flag  = None
    if score >= 70 and spf > 15:
        flag = "Squeeze Watch"
    elif score >= 70:
        flag = "Sell the Hype"
    elif score <= 30 and fundamentals_strong:
        flag = "Buy the Dip"
    elif score <= 25:
        flag = "Fade the Fear"
    sentiment['contrarian_flag'] = flag
    return sentiment


# ── Module 2: Earnings Intelligence ───────────────────────────────────────────

def get_earnings_intelligence(ticker, ticker_obj, spot, atm_iv):
    """
    Deep earnings analysis: timing, historical beat/miss, post-move size,
    IV vs historical comparison, recommendation.
    Returns comprehensive earnings dict.
    """
    try:
        import math as _math
        import numpy as np
        from datetime import datetime as _dt, timedelta as _td
        import pandas as pd

        result = {
            "timing":               None,
            "days_to_earnings":     None,
            "beat_count":           0,
            "miss_count":           0,
            "beat_pct":             None,
            "avg_surprise_pct":     None,
            "avg_move_pct":         None,
            "max_drop_pct":         None,
            "iv_implied_move":      None,
            "historical_avg_move":  None,
            "options_edge":         None,
            "gap_and_hold":         None,
            "earnings_recommendation": None,
            "caution_flag":         None,
        }

        # ── 1. Report timing from calendar ──────────────────────────────────
        try:
            cal = ticker_obj.calendar
            earnings_dt = None
            earnings_time_str = None

            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get('Earnings Date')
                    et = cal.get('Earnings Time')
                else:
                    ed = cal.get('Earnings Date', [None])
                    et = cal.get('Earnings Time', [None])

                if ed is not None:
                    if hasattr(ed, '__iter__') and not isinstance(ed, str):
                        ed = list(ed)
                        if ed: ed = ed[0]
                    if hasattr(ed, 'to_pydatetime'):
                        earnings_dt = ed.to_pydatetime()
                    elif hasattr(ed, 'strftime'):
                        earnings_dt = ed
                    elif isinstance(ed, str):
                        try:
                            earnings_dt = _dt.strptime(ed[:10], '%Y-%m-%d')
                        except Exception:
                            pass

                if et is not None:
                    if hasattr(et, '__iter__') and not isinstance(et, str):
                        et = list(et)
                        if et: et = str(et[0])
                    if et:
                        earnings_time_str = str(et).lower()

            if earnings_dt is not None:
                now = _dt.now()
                # Strip timezone info if present for comparison
                if hasattr(earnings_dt, 'tzinfo') and earnings_dt.tzinfo is not None:
                    earnings_dt_naive = earnings_dt.replace(tzinfo=None)
                else:
                    earnings_dt_naive = earnings_dt
                days_to_earn = max((earnings_dt_naive.date() - now.date()).days, 0)
                result['days_to_earnings'] = days_to_earn

                # Determine AMC/BMO/During based on time string or hour
                if earnings_time_str:
                    if any(x in earnings_time_str for x in ['after', 'amc', 'post']):
                        result['timing'] = 'AMC'
                    elif any(x in earnings_time_str for x in ['before', 'bmo', 'pre']):
                        result['timing'] = 'BMO'
                    else:
                        result['timing'] = 'During Market'
                elif hasattr(earnings_dt, 'hour') and earnings_dt.hour > 0:
                    hour = earnings_dt.hour
                    if hour >= 16:   # 4pm+
                        result['timing'] = 'AMC'
                    elif hour < 9 or (hour == 9 and earnings_dt.minute < 30):
                        result['timing'] = 'BMO'
                    else:
                        result['timing'] = 'During Market'
                else:
                    result['timing'] = 'AMC'  # default assumption
        except Exception:
            pass

        # ── 2. Historical EPS beat/miss data ────────────────────────────────
        beat_count     = 0
        miss_count     = 0
        surprises      = []
        earnings_dates = []

        try:
            # Try earnings_history first (newer yfinance)
            eh = getattr(ticker_obj, 'earnings_history', None)
            if eh is not None and not eh.empty:
                for _, row in eh.iterrows():
                    eps_est  = row.get('epsEstimate', None)
                    eps_act  = row.get('epsActual',   None)
                    if eps_est is None or eps_act is None:
                        continue
                    try:
                        eps_est = float(eps_est)
                        eps_act = float(eps_act)
                    except Exception:
                        continue
                    if eps_est == 0:
                        continue
                    surprise_pct = (eps_act - eps_est) / abs(eps_est) * 100
                    surprises.append(surprise_pct)
                    if eps_act >= eps_est:
                        beat_count += 1
                    else:
                        miss_count += 1
                    # Track date if available
                    if hasattr(row, 'name') and row.name is not None:
                        try:
                            earnings_dates.append(pd.Timestamp(row.name))
                        except Exception:
                            pass
        except Exception:
            pass

        if not surprises:
            try:
                # Fallback: quarterly_earnings
                qe = getattr(ticker_obj, 'quarterly_earnings', None)
                if qe is not None and not qe.empty:
                    if 'Reported EPS' in qe.columns and 'Surprise(%)' in qe.columns:
                        for _, row in qe.iterrows():
                            surp = row.get('Surprise(%)', None)
                            if surp is None:
                                continue
                            try:
                                surp = float(surp)
                            except Exception:
                                continue
                            surprises.append(surp)
                            if surp >= 0:
                                beat_count += 1
                            else:
                                miss_count += 1
            except Exception:
                pass

        # Limit to last 8 quarters
        surprises  = surprises[:8]
        total_q    = beat_count + miss_count
        if total_q > 0:
            beat_pct = round(beat_count / total_q * 100, 1)
        else:
            beat_pct = None

        avg_surprise = round(sum(surprises) / len(surprises), 2) if surprises else None

        result['beat_count']      = beat_count
        result['miss_count']      = miss_count
        result['beat_pct']        = beat_pct
        result['avg_surprise_pct'] = avg_surprise

        # ── 3. Post-earnings price moves ────────────────────────────────────
        try:
            hist_2y = ticker_obj.history(period='2y')
            if not hist_2y.empty and len(earnings_dates) >= 2:
                closes = hist_2y['Close']
                post_beat_moves = []
                post_miss_moves = []
                all_moves       = []
                gap_and_hold_signals = []

                for ed in earnings_dates[:8]:
                    try:
                        ed_ts = pd.Timestamp(ed).tz_localize(None)
                        # Find index at or after earnings date
                        future = closes.index[closes.index.tz_localize(None) >= ed_ts]
                        before = closes.index[closes.index.tz_localize(None) <  ed_ts]
                        if len(future) < 4 or len(before) < 1:
                            continue

                        pre_close  = float(closes.loc[before[-1]])
                        day1_close = float(closes.loc[future[0]])
                        day3_close = float(closes.loc[future[2]]) if len(future) >= 3 else day1_close

                        move_1d = (day1_close - pre_close) / pre_close * 100
                        move_3d = (day3_close - pre_close) / pre_close * 100
                        all_moves.append(abs(move_1d))

                        # Try to match this date to beat/miss
                        # Use surprises order (same order as earnings_dates)
                        idx = earnings_dates.index(ed)
                        if idx < len(surprises):
                            if surprises[idx] >= 0:
                                post_beat_moves.append(move_1d)
                                # Gap and hold: 3d > 1d return?
                                gap_and_hold_signals.append(move_3d > move_1d)
                            else:
                                post_miss_moves.append(move_1d)
                    except Exception:
                        continue

                if all_moves:
                    result['avg_move_pct'] = round(sum(all_moves) / len(all_moves), 2)
                    result['max_drop_pct'] = round(min(
                        [m for m in all_moves], default=0
                    ) * -1 if post_miss_moves else min(all_moves) * -1, 2)

                if gap_and_hold_signals:
                    hold_pct = sum(1 for x in gap_and_hold_signals if x) / len(gap_and_hold_signals)
                    result['gap_and_hold'] = hold_pct >= 0.6  # True if holds 60%+ of the time
        except Exception:
            pass

        # ── 4. IV vs historical move comparison ─────────────────────────────
        try:
            days_to_earn = result.get('days_to_earnings')
            avg_move     = result.get('avg_move_pct')
            if days_to_earn is not None and days_to_earn >= 0 and atm_iv is not None and avg_move is not None:
                iv_frac       = atm_iv / 100.0
                iv_implied    = round(iv_frac / _math.sqrt(252) * _math.sqrt(max(days_to_earn, 1)) * 100, 2)
                options_edge  = round(avg_move - iv_implied, 2)
                result['iv_implied_move']     = iv_implied
                result['historical_avg_move'] = avg_move
                result['options_edge']        = options_edge
        except Exception:
            pass

        # ── 5. Recommendation ───────────────────────────────────────────────
        try:
            timing       = result.get('timing', 'AMC')
            days_to_earn = result.get('days_to_earnings', 999)
            beat_pct_v   = result.get('beat_pct', 0) or 0
            opts_edge    = result.get('options_edge', 0) or 0
            atm_iv_val   = atm_iv or 30
            # rough "expensive" threshold: if iv_implied > avg_move * 1.2
            iv_impl      = result.get('iv_implied_move') or 0
            avg_mv       = result.get('avg_move_pct') or 0
            iv_expensive = iv_impl > avg_mv * 1.2 if avg_mv > 0 else False

            rec = None
            caution = None

            if days_to_earn < 7:
                caution = "CAUTION: Binary event"

            if timing == 'AMC' and beat_pct_v >= 75 and opts_edge > 0:
                rec = "Buy Call before close, sell at open"
            elif timing == 'AMC' and iv_expensive:
                rec = "Sell Straddle — collect IV crush"
            elif timing == 'BMO' and beat_pct_v >= 75:
                rec = "Buy Call day before, close at open"
            elif iv_expensive:
                rec = "Sell Straddle — IV elevated before earnings"

            if caution and rec:
                rec = f"{rec} | {caution}"
            elif caution:
                rec = caution

            result['earnings_recommendation'] = rec
            result['caution_flag']            = caution
        except Exception:
            pass

        return result

    except Exception:
        return {
            "timing":               None,
            "days_to_earnings":     None,
            "beat_count":           0,
            "miss_count":           0,
            "beat_pct":             None,
            "avg_surprise_pct":     None,
            "avg_move_pct":         None,
            "max_drop_pct":         None,
            "iv_implied_move":      None,
            "historical_avg_move":  None,
            "options_edge":         None,
            "gap_and_hold":         None,
            "earnings_recommendation": None,
            "caution_flag":         None,
        }


# ── Module 3: Recommendation Engine ───────────────────────────────────────────

# Leveraged ETF mapping
LEVERAGED_ETFS = {
    'TSLA': {'bull': 'TSLL',  'bear': 'TSLS'},
    'NVDA': {'bull': 'NVDL',  'bear': 'NVDS'},
    'SPY':  {'bull': 'SSO',   'bear': 'SDS'},
    'QQQ':  {'bull': 'QLD',   'bear': 'QID'},
    'AAPL': {'bull': 'AAPU',  'bear': 'AAPD'},
    'AMZN': {'bull': 'AMZU',  'bear': 'AMZD'},
    'MSFT': {'bull': 'MSFU',  'bear': 'MSFD'},
    'META': {'bull': 'METU',  'bear': 'METD'},
    'GOOGL':{'bull': 'GGLL',  'bear': 'GGLS'},
    'AMD':  {'bull': 'AMDU',  'bear': 'AMDS'},
    'RKLB': {'bull': 'RKLZ',  'bear': 'RKLX'},
    'COIN': {'bull': 'CONL',  'bear': 'CONS'},
    'PLTR': {'bull': 'PTIR',  'bear': None},
    'MSTR': {'bull': 'MSTX',  'bear': 'MSTS'},
    'SOXL': {'bull': 'SOXL',  'bear': 'SOXS'},
    'IWM':  {'bull': 'TNA',   'bear': 'TZA'},
    'GLD':  {'bull': 'UGL',   'bear': 'GLL'},
    'TLT':  {'bull': 'UBT',   'bear': 'TBT'},
    'XLE':  {'bull': 'ERX',   'bear': 'ERY'},
    'XLF':  {'bull': 'FAS',   'bear': 'FAZ'},
    'BITI': {'bull': 'BITX',  'bear': 'BITI'},
    'MARA': {'bull': 'MARU',  'bear': None},
    'RIOT': {'bull': None,    'bear': None},
    'SMCI': {'bull': 'SMCU',  'bear': 'SMCD'},
    'BABA': {'bull': 'BABAF', 'bear': None},
    'NIO':  {'bull': 'NIOL',  'bear': None},
    'SOFI': {'bull': 'SOFL',  'bear': None},
    'HOOD': {'bull': 'HOODL', 'bear': None},
    'GME':  {'bull': 'GMBL',  'bear': None},
    'AMC':  {'bull': None,    'bear': None},
}


def get_recommendation(ticker, spot, atm_iv, vrp, valuation, sentiment, earnings_intel,
                       fundamentals, vol_metrics):
    """
    Single best action recommendation based on all available signals.
    Returns: {action, signal, reasoning, horizon, alt, leveraged_bull, leveraged_bear}
    """
    try:
        earnings_within_7d   = (earnings_intel or {}).get('days_to_earnings', 999) is not None \
                               and (earnings_intel or {}).get('days_to_earnings', 999) <= 7
        squeeze_setup        = (sentiment or {}).get('contrarian_flag') == 'Squeeze Watch'
        iv_cheap             = vrp < -2
        iv_expensive         = vrp > 5
        val_score            = (valuation or {}).get('score', 50) if valuation else 50
        fundamentals_strong  = valuation is not None and val_score >= 60
        fundamentals_weak    = valuation is not None and val_score <= 35
        sentiment_score      = (sentiment or {}).get('score', 50)
        has_leveraged        = ticker in LEVERAGED_ETFS
        beat_pct_v           = (earnings_intel or {}).get('beat_pct', 0) or 0
        opts_edge            = (earnings_intel or {}).get('options_edge', 0) or 0

        action   = "WAIT"
        signal   = "⚪ No Clear Edge"
        reasoning= "Mixed signals — no strong directional or volatility edge right now"
        horizon  = "Monitor"
        alt      = None

        if squeeze_setup:
            action    = "BUY CALLS"
            signal    = "🚀 Squeeze Watch"
            reasoning = "Short interest elevated + retail buzz spiking — squeeze setup forming"
            horizon   = "7–14 days"
            if has_leveraged and LEVERAGED_ETFS[ticker].get('bull'):
                alt = f"Or buy {LEVERAGED_ETFS[ticker]['bull']} for leveraged exposure"

        elif earnings_within_7d and beat_pct_v >= 75 and opts_edge > 0:
            action    = "BUY CALL SPREAD"
            signal    = "📊 Earnings Edge"
            reasoning = (f"Consistent beater ({beat_pct_v:.0f}%) + options underpriced "
                         f"vs historical move")
            horizon   = "Hold through earnings"

        elif earnings_within_7d and iv_expensive:
            action    = "AVOID — WAIT"
            signal    = "⚠️ Earnings Risk"
            reasoning = ("Earnings in <7 days, IV inflated — wait for post-earnings "
                         "IV crush to sell premium")
            horizon   = "Wait"

        elif iv_expensive and sentiment_score >= 65 and fundamentals_strong:
            action    = "SELL PREMIUM"
            signal    = "💰 Sell the Hype"
            reasoning = (f"Retail euphoric, IV elevated (+{vrp:.1f}% over RV) — "
                         f"sell overpriced options, collect credit")
            horizon   = "21–45 days"

        elif iv_expensive and fundamentals_weak and sentiment_score >= 55:
            action    = "BUY PUTS"
            signal    = "☠️ Danger Zone"
            reasoning = ("Weak fundamentals + retail still bullish + high IV — "
                         "stock vulnerable, buy puts or bear spread")
            horizon   = "21–45 days"
            if has_leveraged and LEVERAGED_ETFS[ticker].get('bear'):
                alt = f"Or buy {LEVERAGED_ETFS[ticker]['bear']} for leveraged short"

        elif iv_cheap and fundamentals_strong and sentiment_score <= 45:
            action    = "BUY STOCK"
            signal    = "📈 Buy the Dip"
            reasoning = ("Retail bearish but fundamentals strong + IV cheap — "
                         "stock undervalued, buy shares or calls cheap")
            horizon   = "30–60 days"
            if has_leveraged and LEVERAGED_ETFS[ticker].get('bull'):
                alt = f"Or buy {LEVERAGED_ETFS[ticker]['bull']} for 2x leveraged upside"

        elif sentiment_score <= 30 and iv_cheap:
            action    = "BUY CALLS"
            signal    = "📉 Fade the Fear"
            reasoning = ("Extreme retail fear + cheap IV — buy calls before the "
                         "crowd realizes they're wrong")
            horizon   = "14–30 days"

        elif iv_expensive and not earnings_within_7d:
            action    = "SELL PREMIUM"
            signal    = "💰 Sell the Hype"
            reasoning = (f"IV significantly elevated (+{vrp:.1f}% over RV) — "
                         f"premium rich, sell straddle or iron condor")
            horizon   = "21–45 days"

        elif fundamentals_strong and iv_cheap:
            action    = "BUY STOCK"
            signal    = "✅ Clear Buy"
            reasoning = ("Strong fundamentals + cheap IV — straightforward long opportunity")
            horizon   = "30–90 days"
            if has_leveraged and LEVERAGED_ETFS[ticker].get('bull'):
                alt = f"Or buy {LEVERAGED_ETFS[ticker]['bull']} for leveraged upside"

        # Build leveraged ETF info
        lev_bull = None
        lev_bear = None
        if has_leveraged:
            lev_bull = LEVERAGED_ETFS[ticker].get('bull')
            lev_bear = LEVERAGED_ETFS[ticker].get('bear')

        return {
            "action":         action,
            "signal":         signal,
            "reasoning":      reasoning,
            "horizon":        horizon,
            "alt":            alt,
            "leveraged_bull": lev_bull,
            "leveraged_bear": lev_bear,
        }

    except Exception:
        return {
            "action":         "WAIT",
            "signal":         "⚪ No Clear Edge",
            "reasoning":      "Analysis unavailable",
            "horizon":        "Monitor",
            "alt":            None,
            "leveraged_bull": None,
            "leveraged_bear": None,
        }


# ── Edge Factor Helpers ────────────────────────────────────────────────────────

def compute_gex(calls, puts, spot):
    """
    Net Dealer Gamma Exposure (GEX).
    Positive GEX = dealers long gamma = stock pinned, low volatility expected.
    Negative GEX = dealers short gamma = stock accelerates moves.
    Falls back to Black-Scholes gamma if 'gamma' column unavailable.
    """
    try:
        def _row_gamma(row, option_type, spot):
            """Get or approximate gamma for a row."""
            g = row.get('gamma', None)
            if g is not None and float(g) > 0:
                return float(g)
            # Approximate via Black-Scholes
            iv   = float(row.get('impliedVolatility', 0) or 0)
            K    = float(row.get('strike', spot))
            T    = max(float(row.get('daysToExpiration', 30)) / 365.0, 1/365.0) \
                   if 'daysToExpiration' in row.index else 30/365.0
            if iv <= 0 or T <= 0:
                return 0.0
            bs = black_scholes(spot, K, T, 0.05, iv, option_type)
            return bs['gamma']

        call_gex = 0.0
        put_gex  = 0.0

        if calls is not None and not calls.empty:
            for _, row in calls.iterrows():
                g  = _row_gamma(row, 'call', spot)
                oi = float(row.get('openInterest', 0) or 0)
                call_gex += g * oi * 100 * spot**2 * 0.01

        if puts is not None and not puts.empty:
            for _, row in puts.iterrows():
                g  = _row_gamma(row, 'put', spot)
                oi = float(row.get('openInterest', 0) or 0)
                put_gex += g * oi * 100 * spot**2 * 0.01

        net_gex    = round(call_gex - put_gex, 2)
        gex_regime = "pinned" if net_gex > 0 else "explosive"
        return net_gex, gex_regime
    except Exception:
        return None, None


def compute_iv_rank(vol_metrics):
    """
    IVR = (current_IV - 52wk_low_IV) / (52wk_high_IV - 52wk_low_IV) * 100
    Uses HV series from vol_metrics as proxy for IV range.
    IVR > 50 = IV elevated; IVR < 50 = IV cheap.
    """
    try:
        hv10 = vol_metrics.get('hv10')
        hv20 = vol_metrics.get('hv20')
        hv60 = vol_metrics.get('hv60')
        if hv20 is None:
            return None
        # Use hv10 (recent) as current, hv60 as the 52-week range proxy
        # Build a rough range from available windows
        all_hvs = [v for v in [hv10, hv20, hv60] if v is not None]
        hv_low  = min(all_hvs)
        hv_high = max(all_hvs)
        current = hv20  # 20-day as current reference
        if hv_high - hv_low <= 0:
            return 50.0
        ivr = round((current - hv_low) / (hv_high - hv_low) * 100, 1)
        return max(0.0, min(ivr, 100.0))
    except Exception:
        return None


def compute_rsi(hist, period=14):
    """
    14-day RSI from price history.
    RSI > 70 = overbought; RSI < 30 = oversold.
    """
    try:
        closes = list(hist['Close'].dropna())
        if len(closes) < period + 1:
            return None
        closes = closes[-(period + 1):]
        gains  = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs  = avg_gain / avg_loss
        rsi = round(100 - (100 / (1 + rs)), 2)
        return rsi
    except Exception:
        return None


def compute_mfi(hist, period=14):
    """
    Money Flow Index — volume-weighted RSI.
    MFI > 80 = overbought with heavy volume; MFI < 20 = oversold.
    """
    try:
        closes  = list(hist['Close'].dropna())
        highs   = list(hist['High'].dropna())
        lows    = list(hist['Low'].dropna())
        volumes = list(hist['Volume'].dropna())
        n       = min(len(closes), len(highs), len(lows), len(volumes))
        if n < period + 1:
            return None

        # Align all to same length
        closes  = closes[-n:]
        highs   = highs[-n:]
        lows    = lows[-n:]
        volumes = volumes[-n:]

        typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(n)]
        money_flows    = [typical_prices[i] * volumes[i] for i in range(n)]

        pos_flow = []
        neg_flow = []
        for i in range(1, n):
            if typical_prices[i] > typical_prices[i-1]:
                pos_flow.append(money_flows[i])
                neg_flow.append(0)
            else:
                pos_flow.append(0)
                neg_flow.append(money_flows[i])

        # Use last `period` periods
        pos_sum = sum(pos_flow[-period:])
        neg_sum = sum(neg_flow[-period:])

        if neg_sum == 0:
            return 100.0
        mfr = pos_sum / neg_sum
        mfi = round(100 - (100 / (1 + mfr)), 2)
        return mfi
    except Exception:
        return None


def compute_insider_net(ticker_obj):
    """
    Net insider buying/selling in last 90 days.
    Positive = net buying (bullish); Negative = net selling (bearish).
    Returns net shares transacted.
    """
    try:
        from datetime import datetime as _dt, timedelta as _td
        ins = getattr(ticker_obj, 'insider_transactions', None)
        if ins is None or ins.empty:
            return None
        cutoff = _dt.now() - _td(days=90)
        net_shares = 0
        for _, row in ins.iterrows():
            try:
                start_date = row.get('startDate', None)
                if start_date is None:
                    continue
                if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                if hasattr(start_date, 'to_pydatetime'):
                    start_date = start_date.to_pydatetime()
                if start_date < cutoff:
                    continue
                shares = int(row.get('shares', 0) or 0)
                trans  = str(row.get('transactionText', '') or '').lower()
                if 'sale' in trans or 'sell' in trans:
                    net_shares -= shares
                elif 'purchase' in trans or 'buy' in trans:
                    net_shares += shares
            except Exception:
                continue
        return net_shares
    except Exception:
        return None


def compute_options_flow(calls, puts):
    """
    Call/Put volume ratio for today's options flow.
    > 2.0 = Heavy Call Buying; < 0.5 = Heavy Put Buying.
    """
    try:
        call_vol = float(calls['volume'].sum()) if calls is not None and not calls.empty else 0.0
        put_vol  = float(puts['volume'].sum())  if puts  is not None and not puts.empty  else 0.0
        flow_ratio = round(call_vol / put_vol, 3) if put_vol > 0 else 1.0
        if flow_ratio > 2.0:
            flow_signal = "Heavy Call Buying"
        elif flow_ratio < 0.5:
            flow_signal = "Heavy Put Buying"
        else:
            flow_signal = "Balanced"
        return flow_ratio, flow_signal, call_vol, put_vol
    except Exception:
        return 1.0, "Balanced", 0.0, 0.0


# ── New Edge Factors ─────────────────────────────────────────────────────────

def compute_put_call_ratio(calls_df, puts_df):
    """
    Put/Call ratio from open interest.
    > 1.5 = extreme fear (contrarian buy)
    < 0.5 = extreme greed (contrarian sell)
    Returns: ratio, signal, interpretation
    """
    try:
        call_oi = float(calls_df['openInterest'].fillna(0).sum()) if calls_df is not None and not calls_df.empty else 0
        put_oi  = float(puts_df['openInterest'].fillna(0).sum())  if puts_df  is not None and not puts_df.empty  else 0
        if call_oi == 0:
            return None, "Unknown", "No open interest data"
        ratio = round(put_oi / call_oi, 2)
        if ratio > 1.5:
            signal = "Extreme Fear"
            interp = "Heavy put buying — contrarian BUY signal. Market expects big drop."
        elif ratio > 1.0:
            signal = "Bearish"
            interp = "More puts than calls — cautious/bearish positioning."
        elif ratio > 0.7:
            signal = "Neutral"
            interp = "Normal put/call balance — no extreme positioning."
        elif ratio > 0.5:
            signal = "Bullish"
            interp = "More calls than puts — bullish positioning."
        else:
            signal = "Extreme Greed"
            interp = "Heavy call buying — contrarian SELL signal. Euphoria may be peaking."
        return ratio, signal, interp
    except Exception:
        return None, "Unknown", "Could not compute"


def compute_iv_crush_score(atm_iv, earnings_intel_data):
    """
    IV Crush Score — how much IV is likely to drop after earnings.
    Higher = better opportunity to sell IV before earnings.
    Returns: score 0-100, expected crush %, recommendation
    """
    try:
        days_to_earn = earnings_intel_data.get('days_to_earnings')
        iv_implied   = earnings_intel_data.get('iv_implied_move')
        hist_move    = earnings_intel_data.get('historical_avg_move')
        beat_pct     = earnings_intel_data.get('beat_pct', 50) or 50

        if days_to_earn is None or days_to_earn < 0 or days_to_earn > 21:
            return None, None, "No upcoming earnings within 21 days"

        if iv_implied is None or hist_move is None:
            return None, None, "Insufficient data"

        # IV crush = difference between IV implied move and historical actual move
        crush_pct = round(iv_implied - hist_move, 1)
        
        # Score: higher = more IV overpricing = better to sell
        score = min(100, max(0, int((crush_pct / max(hist_move, 1)) * 100)))

        if crush_pct > 5:
            rec = f"Strong IV Crush setup — IV implying {iv_implied:.1f}% move but history shows {hist_move:.1f}%. Sell straddle {days_to_earn}d before earnings."
        elif crush_pct > 2:
            rec = f"Moderate IV Crush — sell iron condor or strangle before earnings."
        elif crush_pct < -2:
            rec = f"IV is CHEAP vs history — consider buying a straddle before earnings."
        else:
            rec = "IV fairly priced vs earnings history. No strong crush play."

        return score, crush_pct, rec
    except Exception:
        return None, None, "Could not compute"


def compute_short_squeeze_score(fundamentals, sentiment_data, vol_metrics):
    """
    Short Squeeze Score 0-100.
    High score = high squeeze potential.
    Factors: short float %, days to cover, sentiment, volume spike, price momentum.
    """
    try:
        short_pct    = fundamentals.get('short_pct_float', 0) or 0
        days_cover   = fundamentals.get('short_ratio', 0) or 0
        sentiment    = sentiment_data.get('score', 50) or 50
        vol_spike    = vol_metrics.get('volume_ratio_5d', 1) or 1
        reddit_buzz  = sentiment_data.get('reddit_buzz', 0) or 0

        score = 0

        # Short float (max 40 pts) — higher short = more fuel
        if short_pct >= 30:   score += 40
        elif short_pct >= 20: score += 30
        elif short_pct >= 15: score += 20
        elif short_pct >= 10: score += 12
        elif short_pct >= 5:  score += 5

        # Days to cover (max 20 pts) — longer = shorts more trapped
        if days_cover >= 10:  score += 20
        elif days_cover >= 7: score += 14
        elif days_cover >= 5: score += 8
        elif days_cover >= 3: score += 4

        # Sentiment (max 20 pts) — retail buying into shorts
        if sentiment >= 75:   score += 20
        elif sentiment >= 60: score += 12
        elif sentiment >= 50: score += 6

        # Volume spike (max 10 pts) — unusual activity
        if vol_spike >= 3:    score += 10
        elif vol_spike >= 2:  score += 6
        elif vol_spike >= 1.5: score += 3

        # Reddit buzz (max 10 pts)
        if reddit_buzz >= 8:  score += 10
        elif reddit_buzz >= 5: score += 6
        elif reddit_buzz >= 3: score += 3

        score = min(100, score)

        if score >= 70:
            signal = "High Squeeze Risk"
            desc   = f"Short float {short_pct:.1f}%, {days_cover:.1f} days to cover — shorts are trapped. Retail FOMO building."
        elif score >= 50:
            signal = "Moderate Squeeze Potential"
            desc   = f"Short float {short_pct:.1f}% with rising sentiment. Watch for volume catalyst."
        elif score >= 30:
            signal = "Low Squeeze Risk"
            desc   = f"Some short interest but not enough for a major squeeze."
        else:
            signal = "No Squeeze Risk"
            desc   = f"Low short interest — stock moves on fundamentals."

        return score, signal, desc
    except Exception:
        return 0, "Unknown", "Could not compute"


def compute_gamma_pin(calls_df, puts_df, spot):
    """
    Find the gamma pin level — the strike with highest total open interest.
    This is the 'magnet' price where market makers force the stock to gravitate.
    Returns: pin_strike, total_oi, distance_pct
    """
    try:
        if calls_df is None or puts_df is None:
            return None, None, None

        # Combine OI by strike
        oi_map = {}
        for df in [calls_df, puts_df]:
            for _, row in df.iterrows():
                strike = float(row.get('strike', 0))
                oi     = float(row.get('openInterest', 0) or 0)
                if strike > 0:
                    oi_map[strike] = oi_map.get(strike, 0) + oi

        if not oi_map:
            return None, None, None

        # Find strike with max OI
        pin_strike = max(oi_map, key=oi_map.get)
        total_oi   = int(oi_map[pin_strike])
        distance_pct = round((pin_strike - spot) / spot * 100, 1)

        return round(pin_strike, 2), total_oi, distance_pct
    except Exception:
        return None, None, None


def compute_relative_strength(hist, spy_hist=None):
    """
    Relative strength vs SPY over last 20 days.
    > 0 = outperforming market
    < 0 = underperforming market
    Returns: rs_score, signal
    """
    try:
        if len(hist) < 20:
            return None, "Insufficient data"

        stock_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100

        # If no SPY data, just use absolute return as proxy
        if spy_hist is None or len(spy_hist) < 20:
            if stock_return > 5:   signal = "Strong"
            elif stock_return > 0: signal = "Positive"
            elif stock_return > -5: signal = "Weak"
            else:                  signal = "Very Weak"
            return round(stock_return, 1), signal

        spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-20] - 1) * 100
        rs = round(stock_return - spy_return, 1)

        if rs > 10:    signal = "Strong Outperformer"
        elif rs > 3:   signal = "Outperforming Market"
        elif rs > -3:  signal = "In line with Market"
        elif rs > -10: signal = "Underperforming"
        else:          signal = "Weak — Lagging Market"

        return rs, signal
    except Exception:
        return None, "Unknown"


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyze_ticker(ticker_symbol):
    import yfinance as yf
    import time
    ticker_symbol = ticker_symbol.upper().strip()

    # Retry up to 3 times on rate limit
    for attempt in range(3):
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist   = ticker.history(period="1y")
            if not hist.empty:
                break
            if attempt < 2:
                time.sleep(3)
        except Exception as e:
            if "Too Many Requests" in str(e) or "rate" in str(e).lower():
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
            raise
    else:
        hist = ticker.history(period="1y")

    if hist.empty:
        return {"error": f"No data found for '{ticker_symbol}'. Please check the ticker symbol."}

    info = ticker.info
    spot = float(hist['Close'].iloc[-1])
    r    = 0.05  # risk-free rate

    # Realized volatility
    rv10 = realized_vol(hist, 10) or 20.0
    rv20 = realized_vol(hist, 20) or 20.0
    rv30 = realized_vol(hist, 30) or 20.0

    # Volume + Bennett metrics
    vol_metrics = compute_volume_metrics(hist)

    # Valuation assessment
    valuation = assess_valuation(info, spot)

    # ── Options expirations ────────────────────────────────────────────────────
    try:
        all_expirations = ticker.options or []
    except Exception:
        all_expirations = []

    if not all_expirations:
        return {"error": f"'{ticker_symbol}' has no options market. Most stocks under $5 or with low volume don't have options. Try: SPY, QQQ, AAPL, TSLA, NVDA, MSFT, AMZN, META, GOOGL, AMD, COIN, GME", "no_options": True}

    valid_exps = []
    for exp in all_expirations:
        d = days_to_expiry(exp)
        if d >= 3:
            valid_exps.append((d, exp))

    if not valid_exps:
        for exp in all_expirations:
            d = days_to_expiry(exp)
            if d >= 1:
                valid_exps.append((d, exp))

    valid_exps.sort(key=lambda x: x[0])
    exps_to_scan = [e for _, e in valid_exps[:8]]

    all_spreads = []
    vol_surface = []
    atm_iv      = None
    skew        = None
    first_chain_exp = None   # for skew calculation

    for exp in exps_to_scan:
        d = days_to_expiry(exp)
        T = T_from_days(d)

        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls.dropna(subset=['strike', 'bid', 'ask', 'impliedVolatility'])
            puts  = chain.puts.dropna(subset=['strike', 'bid', 'ask', 'impliedVolatility'])

            calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
            puts  = puts[(puts['bid']  > 0) & (puts['ask']  > 0)]

            if calls.empty and puts.empty:
                continue

            atm_c = find_atm_row(calls, spot)
            if atm_c is not None:
                iv_here  = float(atm_c['impliedVolatility']) * 100
                vrp_here = round(iv_here - rv20, 1)
                vol_surface.append({
                    "expiry": exp,
                    "days":   d,
                    "atm_iv": round(iv_here, 1),
                    "vrp":    vrp_here,
                })
                if atm_iv is None and d >= 7:
                    atm_iv = round(iv_here, 1)
                elif atm_iv is None:
                    atm_iv = round(iv_here, 1)

            # Compute skew on first valid chain with ≥14 days (most meaningful)
            if skew is None and d >= 14:
                skew = compute_skew(calls, puts, spot, T)
                first_chain_exp = exp

            all_spreads += build_bull_call_spread(calls, spot, T, r, rv20, exp)
            all_spreads += build_bear_put_spread(puts,  spot, T, r, rv20, exp)
            all_spreads += build_short_straddle(calls, puts, spot, T, r, rv20, exp)
            all_spreads += build_iron_condor(calls, puts, spot, T, r, rv20, exp)

        except Exception:
            continue

    # ── Rank and trim ──────────────────────────────────────────────────────────
    all_spreads.sort(key=lambda x: x["score"], reverse=True)

    seen = set()
    top_spreads = []
    for s in all_spreads:
        key = (s["type"], s["expiry"])
        if key not in seen:
            seen.add(key)
            top_spreads.append(s)
        if len(top_spreads) == 6:
            break

    # ── VRP regime ────────────────────────────────────────────────────────────
    if atm_iv is None:
        atm_iv = rv20 * 1.1

    vrp = round(atm_iv - rv20, 1)
    if vrp > 5:
        vrp_regime = "high"
        vrp_signal = "Sell vol — implied vol is overpriced vs realized"
    elif vrp < -2:
        vrp_regime = "low"
        vrp_signal = "Buy vol — implied vol is underpriced vs realized"
    else:
        vrp_regime = "neutral"
        vrp_signal = "Neutral — implied vol is near fair value"

    # ── 52-week range ──────────────────────────────────────────────────────────
    high_52 = round(float(hist['High'].max()), 2) if not hist.empty else None
    low_52  = round(float(hist['Low'].min()),  2) if not hist.empty else None

    # ── Price change ──────────────────────────────────────────────────────────
    prev_close       = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else spot
    price_change     = round(spot - prev_close, 2)
    price_change_pct = round((price_change / prev_close) * 100, 2)

    company_name = info.get('longName') or info.get('shortName') or ticker_symbol

    # ── Edge factors: GEX, IVR, RSI, MFI, Flow, Insider ─────────────────────
    # Use first valid options chain for GEX and flow calculations
    _gex_calls = None
    _gex_puts  = None
    try:
        _first_exp = exps_to_scan[0] if exps_to_scan else None
        if _first_exp:
            _chain    = ticker.option_chain(_first_exp)
            _gex_calls = _chain.calls.dropna(subset=['strike', 'impliedVolatility'])
            _gex_puts  = _chain.puts.dropna(subset=['strike', 'impliedVolatility'])
    except Exception:
        pass

    try:
        net_gex, gex_regime = compute_gex(_gex_calls, _gex_puts, spot)
    except Exception:
        net_gex, gex_regime = None, None

    try:
        ivr = compute_iv_rank(vol_metrics)
    except Exception:
        ivr = None

    try:
        rsi = compute_rsi(hist, period=14)
    except Exception:
        rsi = None

    try:
        mfi = compute_mfi(hist, period=14)
    except Exception:
        mfi = None

    try:
        flow_ratio, flow_signal, _cvol, _pvol = compute_options_flow(_gex_calls, _gex_puts)
    except Exception:
        flow_ratio, flow_signal = 1.0, "Balanced"

    try:
        insider_net = compute_insider_net(ticker)
    except Exception:
        insider_net = None

    # Add edge factors to vol_metrics
    if net_gex    is not None: vol_metrics['gex']          = net_gex
    if gex_regime is not None: vol_metrics['gex_regime']   = gex_regime
    if ivr        is not None: vol_metrics['iv_rank']       = ivr
    vol_metrics['flow_ratio']   = flow_ratio
    vol_metrics['flow_signal']  = flow_signal
    if rsi        is not None: vol_metrics['rsi_14']        = rsi
    if mfi        is not None: vol_metrics['mfi']           = mfi
    if insider_net is not None: vol_metrics['insider_net']  = insider_net

    # ── Real-time price & fundamentals ───────────────────────────────────────
    def _safe(v, mult=1, digits=2):
        """Return rounded float or None."""
        try:
            if v is None: return None
            return round(float(v) * mult, digits)
        except Exception:
            return None

    def _pct(v, digits=1):
        """Convert 0.xx fraction to percentage string, or return None."""
        try:
            if v is None: return None
            return round(float(v) * 100, digits)
        except Exception:
            return None

    # Market-day price data (live during trading hours)
    open_price   = _safe(info.get('regularMarketOpen'))
    day_high     = _safe(info.get('regularMarketDayHigh'))
    day_low      = _safe(info.get('regularMarketDayLow'))
    day_volume   = info.get('regularMarketVolume') or info.get('volume')
    market_cap   = info.get('marketCap')
    beta         = _safe(info.get('beta'), digits=3)
    avg_vol_10d  = info.get('averageVolume10days') or info.get('averageDailyVolume10Day')
    avg_vol_3m   = info.get('averageVolume')

    # Valuation multiples
    trailing_pe  = _safe(info.get('trailingPE'))
    forward_pe   = _safe(info.get('forwardPE'))
    peg_ratio    = _safe(info.get('pegRatio'), digits=2)
    price_book   = _safe(info.get('priceToBook'), digits=2)
    price_sales  = _safe(info.get('priceToSalesTrailing12Months'), digits=2)
    ev_ebitda    = _safe(info.get('enterpriseToEbitda'), digits=1)

    # EPS
    trailing_eps = _safe(info.get('trailingEps'), digits=2)
    forward_eps  = _safe(info.get('forwardEps'), digits=2)

    # Dividend
    # Yahoo Finance returns dividendYield as a decimal (e.g. 0.0041 for 0.41%)
    # but sometimes already as a percentage — clamp to reasonable range
    _dy_raw = info.get('dividendYield')
    div_yield = None
    if _dy_raw is not None:
        _dy = float(_dy_raw)
        # If > 1 it's already in percent form (e.g. 1.5 = 1.5%)
        # If < 1 it's in decimal form (e.g. 0.015 = 1.5%)
        div_yield = round(_dy * 100 if _dy < 1 else _dy, 2)
    div_rate     = _safe(info.get('dividendRate'), digits=2)

    # Analyst estimates
    target_mean  = _safe(info.get('targetMeanPrice'))
    target_high  = _safe(info.get('targetHighPrice'))
    target_low   = _safe(info.get('targetLowPrice'))
    rec_key      = info.get('recommendationKey')          # 'buy','hold','sell'…
    num_analysts = info.get('numberOfAnalystOpinions')
    upside_pct   = _safe((target_mean - spot) / spot * 100, digits=1) if (target_mean and spot) else None

    # Financial health
    gross_margin  = _pct(info.get('grossMargins'))
    op_margin     = _pct(info.get('operatingMargins'))
    net_margin    = _pct(info.get('profitMargins'))
    roe           = _pct(info.get('returnOnEquity'))
    roa           = _pct(info.get('returnOnAssets'))
    debt_equity   = _safe(info.get('debtToEquity'), digits=1)
    current_ratio = _safe(info.get('currentRatio'), digits=2)
    quick_ratio   = _safe(info.get('quickRatio'), digits=2)
    fcf           = info.get('freeCashflow')
    op_cashflow   = info.get('operatingCashflow')
    total_revenue = info.get('totalRevenue')
    ebitda        = info.get('ebitda')

    # Growth
    rev_growth    = _pct(info.get('revenueGrowth'))
    earn_growth   = _pct(info.get('earningsGrowth'))
    earn_qtr_growth = _pct(info.get('earningsQuarterlyGrowth'))

    # Float & short
    shares_out    = info.get('sharesOutstanding')
    float_shares  = info.get('floatShares')
    short_ratio   = _safe(info.get('shortRatio'), digits=1)
    short_pct_float = _pct(info.get('shortPercentOfFloat'))

    # Next earnings
    earnings_date = None
    try:
        cal = ticker.calendar
        if cal is not None:
            # calendar can be a dict or DataFrame
            if isinstance(cal, dict):
                ed = cal.get('Earnings Date')
            else:
                ed = cal.get('Earnings Date', [None])
            if ed is not None:
                if hasattr(ed, '__iter__') and not isinstance(ed, str):
                    ed = list(ed)
                    if ed: ed = ed[0]
                if hasattr(ed, 'strftime'):
                    earnings_date = ed.strftime('%Y-%m-%d')
                elif isinstance(ed, str):
                    earnings_date = ed[:10]
    except Exception:
        pass

    fundamentals = {
        # Live price
        "open":           open_price,
        "day_high":       day_high,
        "day_low":        day_low,
        "day_volume":     int(day_volume) if day_volume else None,
        "avg_vol_10d":    int(avg_vol_10d) if avg_vol_10d else None,
        "avg_vol_3m":     int(avg_vol_3m) if avg_vol_3m else None,
        "market_cap":     int(market_cap) if market_cap else None,
        "beta":           beta,
        # Multiples
        "trailing_pe":    trailing_pe,
        "forward_pe":     forward_pe,
        "peg_ratio":      peg_ratio,
        "price_book":     price_book,
        "price_sales":    price_sales,
        "ev_ebitda":      ev_ebitda,
        "trailing_eps":   trailing_eps,
        "forward_eps":    forward_eps,
        # Income
        "total_revenue":  int(total_revenue) if total_revenue else None,
        "ebitda":         int(ebitda) if ebitda else None,
        "fcf":            int(fcf) if fcf else None,
        "op_cashflow":    int(op_cashflow) if op_cashflow else None,
        # Margins
        "gross_margin":   gross_margin,
        "op_margin":      op_margin,
        "net_margin":     net_margin,
        # Returns
        "roe":            roe,
        "roa":            roa,
        # Leverage
        "debt_equity":    debt_equity,
        "current_ratio":  current_ratio,
        "quick_ratio":    quick_ratio,
        # Growth
        "rev_growth":     rev_growth,
        "earn_growth":    earn_growth,
        "earn_qtr_growth": earn_qtr_growth,
        # Dividend
        "div_yield":      div_yield,
        "div_rate":       div_rate,
        # Analyst
        "target_mean":    target_mean,
        "target_high":    target_high,
        "target_low":     target_low,
        "rec_key":        rec_key,
        "num_analysts":   int(num_analysts) if num_analysts else None,
        "upside_pct":     upside_pct,
        # Float / short
        "shares_out":     int(shares_out) if shares_out else None,
        "float_shares":   int(float_shares) if float_shares else None,
        "short_ratio":    short_ratio,
        "short_pct_float": short_pct_float,
        # Calendar
        "earnings_date":  earnings_date,
    }
    # Strip None-only keys to keep payload lean but keep keys with 0 values
    fundamentals = {k: v for k, v in fundamentals.items() if v is not None}

    # ── Module 1: Sentiment ──────────────────────────────────────────────────
    try:
        sentiment_data = get_sentiment(ticker_symbol)
    except Exception:
        sentiment_data = {
            "score": 50, "signal": "Neutral", "bull_pct": 50.0,
            "reddit_buzz": 0.5, "trend_spike": 1.0, "contrarian_flag": None
        }

    # Apply contrarian flag now that short_pct_float is available
    try:
        _spf             = fundamentals.get('short_pct_float', 0) or 0
        _val_strong      = valuation is not None and (valuation.get('score', 50) >= 60)
        sentiment_data   = _apply_contrarian_flag(sentiment_data, _spf, _val_strong)
    except Exception:
        pass

    # ── Module 2: Earnings Intelligence ─────────────────────────────────────
    try:
        earnings_intel_data = get_earnings_intelligence(
            ticker_symbol, ticker, spot, atm_iv
        )
    except Exception:
        earnings_intel_data = {
            "timing": None, "days_to_earnings": None,
            "beat_count": 0, "miss_count": 0, "beat_pct": None,
            "avg_surprise_pct": None, "avg_move_pct": None,
            "max_drop_pct": None, "iv_implied_move": None,
            "historical_avg_move": None, "options_edge": None,
            "gap_and_hold": None, "earnings_recommendation": None,
            "caution_flag": None,
        }

    # ── New Edge Factors ──────────────────────────────────────────────────

    # Put/Call Ratio
    edge_factors = {}
    try:
        _first_calls = _gex_calls
        _first_puts  = _gex_puts
        pcr, pcr_signal, pcr_interp = compute_put_call_ratio(_first_calls, _first_puts)
        if pcr is not None:
            edge_factors['put_call_ratio'] = pcr
            edge_factors['put_call_signal'] = pcr_signal
            edge_factors['put_call_interp'] = pcr_interp
    except Exception:
        pass

    # IV Crush Score
    try:
        iv_crush_score, iv_crush_pct, iv_crush_rec = compute_iv_crush_score(atm_iv, earnings_intel_data)
        if iv_crush_score is not None:
            edge_factors['iv_crush_score'] = iv_crush_score
            edge_factors['iv_crush_pct'] = iv_crush_pct
            edge_factors['iv_crush_rec'] = iv_crush_rec
    except Exception:
        pass

    # Short Squeeze Score
    try:
        squeeze_score, squeeze_signal, squeeze_desc = compute_short_squeeze_score(
            fundamentals, sentiment_data, vol_metrics
        )
        edge_factors['squeeze_score'] = squeeze_score
        edge_factors['squeeze_signal'] = squeeze_signal
        edge_factors['squeeze_desc'] = squeeze_desc
    except Exception:
        pass

    # Gamma Pin Level
    try:
        pin_strike, pin_oi, pin_dist = compute_gamma_pin(_gex_calls, _gex_puts, spot)
        if pin_strike is not None:
            edge_factors['gamma_pin'] = pin_strike
            edge_factors['gamma_pin_oi'] = pin_oi
            edge_factors['gamma_pin_dist_pct'] = pin_dist
    except Exception:
        pass

    # Relative Strength
    try:
        _spy = yf.Ticker("SPY").history(period="1mo")
        rs_score, rs_signal = compute_relative_strength(hist, _spy)
        if rs_score is not None:
            edge_factors['relative_strength'] = rs_score
            edge_factors['relative_strength_signal'] = rs_signal
    except Exception:
        pass

    # ── Module 3: Recommendation Engine ──────────────────────────────────
    try:
        recommendation_data = get_recommendation(
            ticker_symbol, spot, atm_iv, vrp, valuation,
            sentiment_data, earnings_intel_data, fundamentals, vol_metrics
        )
    except Exception:
        recommendation_data = {
            "action": "WAIT", "signal": "⚪ No Clear Edge",
            "reasoning": "Analysis unavailable", "horizon": "Monitor",
            "alt": None, "leveraged_bull": None, "leveraged_bear": None,
        }

    result = {
        "ticker":           ticker_symbol,
        "company_name":     company_name,
        "spot":             round(spot, 2),
        "price_change":     price_change,
        "price_change_pct": price_change_pct,
        "atm_iv":           atm_iv,
        "rv10":             rv10,
        "rv20":             rv20,
        "rv30":             rv30,
        "vrp":              vrp,
        "vrp_regime":       vrp_regime,
        "vrp_signal":       vrp_signal,
        "high_52":          high_52,
        "low_52":           low_52,
        "vol_surface":      vol_surface[:6],
        "top_spreads":      top_spreads,
        "spread_count":     len(all_spreads),
        "vol_metrics":      vol_metrics,
        "skew":             skew,
        "valuation":        valuation,
        "fundamentals":     fundamentals,
        # ── New modules ────────────────────────────────────────────────
        "sentiment":        sentiment_data,
        "earnings_intel":   earnings_intel_data,
        "recommendation":   recommendation_data,
        "edge_factors":     edge_factors,
    }

    return result


# ── Market scanner ─────────────────────────────────────────────────────────────

# Broad universe: S&P 500 / Nasdaq 100 representative + high-vol names
SCAN_UNIVERSE = [
    # Mega-cap tech
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","ORCL","CRM",
    # Finance
    "JPM","BAC","WFC","GS","MS","BRK-B","V","MA","AXP","BLK",
    # Healthcare
    "UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","ABT","DHR","ISRG",
    # Industrials & energy
    "XOM","CVX","COP","SLB","CAT","DE","GE","HON","RTX","LMT",
    # Consumer
    "WMT","HD","COST","TGT","NKE","MCD","SBUX","DIS","NFLX","BKNG",
    # ETFs (high liquidity, great for vol strategies)
    "SPY","QQQ","IWM","GLD","TLT","XLF","XLE","XLK","ARKK","VXX",
    # High-vol / options-popular
    "PLTR","MSTR","RIVN","LCID","GME","AMC","SOFI","HOOD","UPST","SQ",
]

def quick_scan(ticker_symbol):
    """
    Lightweight scan of a ticker for the market scanner.
    Returns key metrics without full spread analysis.
    Score is based on VRP, HV percentile, volume ratio, and top spread score.
    """
    import yfinance as yf
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist   = ticker.history(period="6mo")
        if hist.empty or len(hist) < 30:
            return None

        spot = float(hist['Close'].iloc[-1])
        rv20 = realized_vol(hist, 20) or 20.0
        rv10 = realized_vol(hist, 10) or rv20
        r    = 0.05

        vol_metrics = compute_volume_metrics(hist)

        try:
            all_exps = ticker.options or []
        except Exception:
            return None

        if not all_exps:
            return None

        valid_exps = [(days_to_expiry(e), e) for e in all_exps if days_to_expiry(e) >= 3]
        if not valid_exps:
            return None
        valid_exps.sort()
        exps_to_scan = [e for _, e in valid_exps[:3]]   # quick: 3 expirations

        atm_iv    = None
        top_score = 0.0
        best_type = None

        for exp in exps_to_scan:
            d = days_to_expiry(exp)
            T = T_from_days(d)
            try:
                chain = ticker.option_chain(exp)
                calls = chain.calls.dropna(subset=['strike', 'bid', 'ask', 'impliedVolatility'])
                puts  = chain.puts.dropna(subset=['strike', 'bid', 'ask', 'impliedVolatility'])
                calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
                puts  = puts[(puts['bid']  > 0) & (puts['ask']  > 0)]

                atm_c = find_atm_row(calls, spot)
                if atm_c is not None:
                    iv_here = float(atm_c['impliedVolatility']) * 100
                    if atm_iv is None and d >= 7:
                        atm_iv = iv_here

                spreads = (
                    build_short_straddle(calls, puts, spot, T, r, rv20, exp) +
                    build_iron_condor(calls, puts, spot, T, r, rv20, exp)
                )
                for s in spreads:
                    if s["score"] > top_score:
                        top_score = s["score"]
                        best_type = s["type"]
            except Exception:
                continue

        if atm_iv is None:
            return None

        vrp = round(atm_iv - rv20, 1)
        hv_pct   = vol_metrics.get("hv_percentile")
        vol_ratio = vol_metrics.get("volume_ratio_5d", 1.0)

        # Composite scan score (different from per-spread score)
        # High VRP = big edge for vol selling. High HV percentile = elevated vol. Volume spike = interest.
        vrp_score   = min(max(vrp / 20.0, 0), 1.0)                       # 0-1
        hv_score    = (hv_pct / 100.0) if hv_pct is not None else 0.5    # 0-1
        vol_score   = min(vol_ratio / 3.0, 1.0)                          # 0-1, cap at 3x spike
        spread_score = top_score / 100.0                                  # 0-1

        scan_score = round((vrp_score * 0.35 + spread_score * 0.35 + hv_score * 0.20 + vol_score * 0.10) * 100, 1)

        # Price change
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else spot
        price_change_pct = round((spot - prev_close) / prev_close * 100, 2)

        # ── Sentiment score for scanner (fast, lightweight) ─────────────────────
        sentiment_score  = None
        sentiment_signal = None
        try:
            _sent = get_sentiment(ticker_symbol)
            sentiment_score  = _sent.get('score')
            sentiment_signal = _sent.get('signal')
        except Exception:
            pass

        # ── Recommendation signal for scanner ─────────────────────────────
        rec_action = None
        rec_signal = None
        try:
            _val = None  # no full valuation in scanner for speed
            _ei  = {'days_to_earnings': 999, 'beat_pct': None, 'options_edge': None}
            _sent_obj = {'score': sentiment_score or 50, 'contrarian_flag': None}
            _rec = get_recommendation(
                ticker_symbol, spot, atm_iv, vrp, _val,
                _sent_obj, _ei, {}, vol_metrics
            )
            rec_action = _rec.get('action')
            rec_signal = _rec.get('signal')
        except Exception:
            pass

        return {
            "ticker":           ticker_symbol,
            "spot":             round(spot, 2),
            "price_change_pct": price_change_pct,
            "atm_iv":           round(atm_iv, 1),
            "rv20":             rv20,
            "vrp":              vrp,
            "hv_percentile":    hv_pct,
            "volume_ratio":     vol_ratio,
            "best_strategy":    best_type,
            "top_spread_score": round(top_score, 1),
            "scan_score":       scan_score,
            # New fields
            "sentiment_score":  sentiment_score,
            "sentiment_signal": sentiment_signal,
            "rec_action":       rec_action,
            "rec_signal":       rec_signal,
        }
    except Exception:
        return None


def scan_market(max_tickers=50, top_n=10):
    """
    Scan the universe, return top_n tickers ranked by scan_score.
    """
    import concurrent.futures
    universe = SCAN_UNIVERSE[:max_tickers]
    results  = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(quick_scan, t): t for t in universe}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x["scan_score"], reverse=True)
    return results[:top_n]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No ticker provided"}))
        sys.exit(1)

    cmd = sys.argv[1].upper().strip()

    if cmd == "SCAN":
        try:
            top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            result = scan_market(top_n=top_n)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)
    else:
        try:
            result = analyze_ticker(cmd)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)
