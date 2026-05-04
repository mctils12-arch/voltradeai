#!/usr/bin/env python3
"""
insights.py — Smart-money + structure data for a single ticker.

ALPHA AUDIT 2026-05-03 (batch 3): a focused alternative view to
analyze.py's options-heavy display. Designed to surface the data
that exists in the bot's data layer but isn't shown in a focused
way to the user — insider activity, institutional positioning,
float/short structure, support/resistance levels, and options
smart-money signals (gamma pin, P/C ratio, max pain).

Backend for /api/insights/<ticker>. Invocation:
    python3 insights.py <TICKER>

Output: JSON-only on stdout (the route handler JSON.parses it).
Failure: still returns valid JSON with `error` field, exit 0.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from typing import Any

# Heavy imports are deferred so a partial-data ticker still returns
# something useful instead of crashing the whole endpoint.
try:
    import yfinance as yf
except Exception:
    yf = None


def _safe_get(fn, *args, **kwargs):
    """Call fn, never raise — return None on any failure."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _round(x, n=2):
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, n)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 1: Insider activity (Finnhub MSPR + monthly buy/sell counts)
# ═══════════════════════════════════════════════════════════════════════════════

def _insider_section(ticker: str) -> dict:
    """Pull insider sentiment from finnhub_data.get_insider_sentiment.

    Output:
        {
            'mspr':            -100..+100 (Monthly Share Purchase Ratio)
            'change':          net change in latest month
            'signal':          'bullish' | 'bearish' | 'neutral'
            'total_buys':      cumulative buy count last 12mo
            'total_sells':     cumulative sell count last 12mo
            'plain_english':   short user-facing summary
        }
    """
    try:
        from finnhub_data import get_insider_sentiment
        data = get_insider_sentiment(ticker) or {}
    except Exception:
        return {"available": False}

    mspr = data.get("mspr")
    signal = data.get("signal", "neutral")
    buys = data.get("total_buys", 0)
    sells = data.get("total_sells", 0)

    # Plain-english label
    if signal == "bullish":
        plain = (
            f"Insiders bought net {int(buys - sells)} shares in the last 12 months "
            f"(MSPR {mspr:+.1f}). Bullish — insiders are voting with their wallets."
        )
    elif signal == "bearish":
        plain = (
            f"Insiders sold net {int(sells - buys)} shares in the last 12 months "
            f"(MSPR {mspr:+.1f}). Bearish — insiders may know something."
        )
    else:
        plain = "No strong insider signal in either direction."

    return {
        "available":    True,
        "mspr":         _round(mspr, 1),
        "change":       _round(data.get("change"), 1),
        "signal":       signal,
        "total_buys":   int(buys or 0),
        "total_sells":  int(sells or 0),
        "plain_english": plain,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 2: Institutional positioning (13F)
# ═══════════════════════════════════════════════════════════════════════════════

def _institutional_section(ticker: str) -> dict:
    """13F-derived signal from institutional_data."""
    try:
        from institutional_data import (
            get_institutional_signal,
            get_whale_activity,
        )
        inst = get_institutional_signal(ticker) or {}
        whale = get_whale_activity(ticker) or {}
    except Exception:
        return {"available": False}

    direction = inst.get("institutional_interest", "stable")
    strength = inst.get("signal_strength", 0)
    holders = inst.get("top_holders", [])
    whales = whale.get("whales", [])

    # Plain-english
    if direction == "increasing" and strength >= 60:
        plain = (
            f"Institutional interest is INCREASING — {len(holders)} major "
            f"holders detected including {', '.join(holders[:3])}."
            + (f" Notable whales: {', '.join(whales[:2])}." if whales else "")
        )
    elif direction == "decreasing":
        plain = (
            f"Institutional interest is DECREASING — major holders are "
            f"reducing positions."
        )
    elif holders:
        plain = (
            f"Stable institutional ownership. Holders include "
            f"{', '.join(holders[:3])}."
            + (f" Whales: {', '.join(whales[:2])}." if whales else "")
        )
    else:
        plain = "No major institutional activity detected this quarter."

    return {
        "available":         True,
        "direction":         direction,
        "signal_strength":   int(strength),
        "major_holders":     holders,
        "holders_added":     int(inst.get("major_holders_added", 0)),
        "holders_reduced":   int(inst.get("major_holders_reduced", 0)),
        "whales":            whales,
        "whale_interest":    bool(whale.get("whale_interest", False)),
        "plain_english":     plain,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 3: Float + short structure (yfinance)
# ═══════════════════════════════════════════════════════════════════════════════

def _float_section(ticker: str) -> dict:
    """Shares outstanding, float, short interest. Source: yfinance."""
    if yf is None:
        return {"available": False, "reason": "yfinance unavailable"}

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {"available": False, "reason": "yfinance fetch failed"}

    shares_out = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    float_shares = info.get("floatShares")
    short_pct_float = info.get("shortPercentOfFloat")
    short_ratio = info.get("shortRatio")  # days to cover
    short_count = info.get("sharesShort")
    short_prior = info.get("sharesShortPriorMonth")
    held_insiders = info.get("heldPercentInsiders")
    held_institutions = info.get("heldPercentInstitutions")

    # Squeeze indicator: high short% + small float
    squeeze_setup = None
    if short_pct_float and short_pct_float > 0.20 and float_shares and float_shares < 100_000_000:
        squeeze_setup = "high"
    elif short_pct_float and short_pct_float > 0.10:
        squeeze_setup = "moderate"
    else:
        squeeze_setup = "low"

    # Short-rising flag
    short_rising = None
    if short_count and short_prior and short_prior > 0:
        delta = (short_count - short_prior) / short_prior
        if delta > 0.10:
            short_rising = "rising_fast"  # +10% MoM
        elif delta > 0.03:
            short_rising = "rising"
        elif delta < -0.10:
            short_rising = "covering_fast"
        elif delta < -0.03:
            short_rising = "covering"
        else:
            short_rising = "stable"

    plain_parts = []
    if float_shares and shares_out:
        insider_lock_pct = (1 - float_shares / shares_out) * 100
        if insider_lock_pct > 30:
            plain_parts.append(
                f"{insider_lock_pct:.0f}% of shares are held by insiders/restricted "
                f"— smaller float means bigger price moves."
            )
    if short_pct_float is not None:
        sp = short_pct_float * 100
        if sp > 20:
            plain_parts.append(
                f"Short interest is {sp:.1f}% of float — that's a squeeze setup if a catalyst hits."
            )
        elif sp > 10:
            plain_parts.append(f"Short interest {sp:.1f}% of float (moderate).")
    if short_ratio:
        plain_parts.append(f"Days to cover: {short_ratio:.1f}.")
    plain = " ".join(plain_parts) if plain_parts else "Standard share structure, no unusual setup."

    return {
        "available":           True,
        "shares_outstanding":  int(shares_out) if shares_out else None,
        "float_shares":        int(float_shares) if float_shares else None,
        "insider_locked_pct":  _round((1 - float_shares / shares_out) * 100, 1)
                                if (shares_out and float_shares and shares_out > 0) else None,
        "held_insiders_pct":   _round((held_insiders or 0) * 100, 2),
        "held_institutions_pct": _round((held_institutions or 0) * 100, 2),
        "short_count":         int(short_count) if short_count else None,
        "short_prior":         int(short_prior) if short_prior else None,
        "short_pct_float":     _round((short_pct_float or 0) * 100, 2),
        "short_ratio_days":    _round(short_ratio, 2),
        "squeeze_setup":       squeeze_setup,
        "short_trend":         short_rising,
        "plain_english":       plain,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 4: Buying/selling pressure (OBV, MFI, volume vs avg)
# ═══════════════════════════════════════════════════════════════════════════════

def _flow_section(ticker: str) -> dict:
    """Order-flow proxies from price+volume bars. No paid feed required."""
    if yf is None:
        return {"available": False}

    try:
        hist = yf.Ticker(ticker).history(period="60d", interval="1d")
        if hist is None or hist.empty or len(hist) < 20:
            return {"available": False, "reason": "insufficient history"}
    except Exception:
        return {"available": False, "reason": "history fetch failed"}

    closes = hist["Close"].values
    highs = hist["High"].values
    lows = hist["Low"].values
    vols = hist["Volume"].values

    # OBV (On-Balance Volume) — cumulative
    obv_series = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv_series.append(obv_series[-1] + vols[i])
        elif closes[i] < closes[i - 1]:
            obv_series.append(obv_series[-1] - vols[i])
        else:
            obv_series.append(obv_series[-1])

    # OBV trend: last 5 vs prior 5
    obv_signal = "neutral"
    if len(obv_series) >= 10:
        recent = sum(obv_series[-5:]) / 5
        prior = sum(obv_series[-10:-5]) / 5
        if prior != 0:
            ratio = recent / prior if prior > 0 else (recent < 0 and abs(recent) < abs(prior))
            if recent > prior * 1.05:
                obv_signal = "accumulation"
            elif recent < prior * 0.95:
                obv_signal = "distribution"

    # MFI (Money Flow Index, 14-period)
    mfi = None
    if len(closes) >= 15:
        typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
        money_flows = [tp * vols[i] for i, tp in enumerate(typical_prices)]
        positive_flow = 0.0
        negative_flow = 0.0
        for i in range(-14, 0):
            if i + 1 == 0:  # Skip last because we need a successor for direction
                break
            if typical_prices[i] > typical_prices[i - 1]:
                positive_flow += money_flows[i]
            elif typical_prices[i] < typical_prices[i - 1]:
                negative_flow += money_flows[i]
        if (positive_flow + negative_flow) > 0:
            money_ratio = positive_flow / (negative_flow + 1e-9)
            mfi = 100 - (100 / (1 + money_ratio))

    # Volume relative to 20-day average
    avg_vol_20 = sum(vols[-20:]) / 20 if len(vols) >= 20 else None
    last_vol = vols[-1] if len(vols) > 0 else None
    vol_ratio = last_vol / avg_vol_20 if (avg_vol_20 and avg_vol_20 > 0) else None

    # Up-day vs down-day volume last 30 days
    up_vol = sum(vols[i] for i in range(-30, 0) if i + len(vols) >= 0
                  and closes[i] > closes[i - 1]) if len(closes) >= 31 else 0
    down_vol = sum(vols[i] for i in range(-30, 0) if i + len(vols) >= 0
                    and closes[i] < closes[i - 1]) if len(closes) >= 31 else 0
    pressure_ratio = up_vol / down_vol if down_vol > 0 else None

    plain_parts = []
    if obv_signal == "accumulation":
        plain_parts.append("OBV shows ACCUMULATION — smart money buying on volume.")
    elif obv_signal == "distribution":
        plain_parts.append("OBV shows DISTRIBUTION — smart money selling on volume.")
    if mfi is not None:
        if mfi > 80:
            plain_parts.append(f"MFI {mfi:.0f} — overbought (institutional distribution risk).")
        elif mfi < 20:
            plain_parts.append(f"MFI {mfi:.0f} — oversold (potential reversal).")
        else:
            plain_parts.append(f"MFI {mfi:.0f} — neutral.")
    if pressure_ratio:
        if pressure_ratio > 1.5:
            plain_parts.append(f"Up-day volume is {pressure_ratio:.1f}x down-day volume — buyers in control.")
        elif pressure_ratio < 0.7:
            plain_parts.append(f"Down-day volume is {1/pressure_ratio:.1f}x up-day volume — sellers in control.")

    return {
        "available":      True,
        "obv_signal":     obv_signal,
        "mfi":            _round(mfi, 1),
        "volume_ratio":   _round(vol_ratio, 2),
        "up_down_volume_ratio": _round(pressure_ratio, 2),
        "plain_english":  " ".join(plain_parts) if plain_parts else "Mixed flow signals.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 5: Support / resistance levels (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def _support_resistance_section(ticker: str) -> dict:
    """
    Computes multiple S/R levels from price history:
      - 20-day, 50-day, 52-week highs and lows
      - Classic floor pivot points (P, S1/S2, R1/R2) from prior day OHLC
      - Volume Profile POC (Point of Control) — most-traded price level
        over last 60 days, computed from daily bar typical prices weighted
        by volume

    Output:
        {
          spot, levels[], poc, plain_english
        }
    """
    if yf is None:
        return {"available": False}

    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d")
        if hist is None or hist.empty or len(hist) < 20:
            return {"available": False, "reason": "insufficient history"}
    except Exception:
        return {"available": False, "reason": "history fetch failed"}

    closes = list(hist["Close"].values)
    highs = list(hist["High"].values)
    lows = list(hist["Low"].values)
    vols = list(hist["Volume"].values)
    spot = closes[-1]

    levels = []

    # Recent highs/lows (resistance above, support below)
    def add_level(label, price, kind):
        if price is None or price <= 0:
            return
        if abs(price - spot) / spot > 0.50:
            return  # Skip levels >50% away — not relevant
        levels.append({
            "label":    label,
            "price":    _round(price, 2),
            "kind":     kind,  # 'support' | 'resistance' | 'pivot'
            "distance_pct": _round((price - spot) / spot * 100, 2),
        })

    # 20-day
    if len(closes) >= 20:
        add_level("20-day high", max(highs[-20:]), "resistance")
        add_level("20-day low", min(lows[-20:]), "support")
    # 50-day
    if len(closes) >= 50:
        add_level("50-day high", max(highs[-50:]), "resistance")
        add_level("50-day low", min(lows[-50:]), "support")
    # 52-week (use entire 1y window)
    add_level("52-week high", max(highs), "resistance")
    add_level("52-week low", min(lows), "support")

    # Pivot points from yesterday's OHLC
    # P = (H + L + C) / 3
    # R1 = 2P - L,    S1 = 2P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    if len(closes) >= 2:
        h_y, l_y, c_y = highs[-2], lows[-2], closes[-2]
        p = (h_y + l_y + c_y) / 3
        r1 = 2 * p - l_y
        s1 = 2 * p - h_y
        r2 = p + (h_y - l_y)
        s2 = p - (h_y - l_y)
        add_level("Pivot",    p,  "pivot")
        add_level("R1",       r1, "resistance")
        add_level("R2",       r2, "resistance")
        add_level("S1",       s1, "support")
        add_level("S2",       s2, "support")

    # Volume Profile POC — bin the last 60 days into N price bins, weight
    # by volume per bin, find the bin with highest total volume.
    poc = None
    poc_volume_pct = None
    if len(closes) >= 60:
        h60 = highs[-60:]
        l60 = lows[-60:]
        v60 = vols[-60:]
        c60 = closes[-60:]
        price_min = min(l60)
        price_max = max(h60)
        if price_max > price_min:
            n_bins = 30
            bin_width = (price_max - price_min) / n_bins
            bins = [0.0] * n_bins
            for i in range(len(c60)):
                # Approximate distribution of volume across the bar's range
                bar_low = l60[i]
                bar_high = h60[i]
                bar_vol = v60[i]
                bar_range = bar_high - bar_low
                if bar_range <= 0:
                    # Concentrate at typical price
                    typical = (bar_high + bar_low + c60[i]) / 3
                    idx = min(n_bins - 1, max(0, int((typical - price_min) / bin_width)))
                    bins[idx] += bar_vol
                else:
                    # Spread evenly across bins covered by the bar
                    start_bin = max(0, int((bar_low - price_min) / bin_width))
                    end_bin = min(n_bins - 1, int((bar_high - price_min) / bin_width))
                    bins_covered = end_bin - start_bin + 1
                    per_bin = bar_vol / bins_covered
                    for b in range(start_bin, end_bin + 1):
                        bins[b] += per_bin
            total_vol = sum(bins)
            if total_vol > 0:
                poc_idx = bins.index(max(bins))
                poc = price_min + (poc_idx + 0.5) * bin_width
                poc_volume_pct = (bins[poc_idx] / total_vol) * 100
                add_level("POC (60d)", poc, "pivot")

    # Sort levels by distance to spot
    levels.sort(key=lambda x: abs(x["distance_pct"]))

    # Find nearest support and nearest resistance
    above = [l for l in levels if l["price"] > spot]
    below = [l for l in levels if l["price"] < spot]
    nearest_resistance = min(above, key=lambda x: x["price"]) if above else None
    nearest_support = max(below, key=lambda x: x["price"]) if below else None

    plain_parts = []
    if nearest_resistance:
        plain_parts.append(
            f"Nearest resistance: {nearest_resistance['label']} at "
            f"${nearest_resistance['price']:.2f} ({nearest_resistance['distance_pct']:+.1f}%)."
        )
    if nearest_support:
        plain_parts.append(
            f"Nearest support: {nearest_support['label']} at "
            f"${nearest_support['price']:.2f} ({nearest_support['distance_pct']:+.1f}%)."
        )
    if poc:
        if abs(poc - spot) / spot < 0.02:
            plain_parts.append(f"POC at ${poc:.2f} — price is at fair value (most-traded zone).")

    return {
        "available":          True,
        "spot":               _round(spot, 2),
        "levels":             levels,
        "nearest_support":    nearest_support,
        "nearest_resistance": nearest_resistance,
        "poc":                _round(poc, 2),
        "poc_volume_pct":     _round(poc_volume_pct, 1),
        "plain_english":      " ".join(plain_parts) if plain_parts else "Multiple levels — see chart.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 6: Options smart money (gamma pin, P/C, max pain)
# ═══════════════════════════════════════════════════════════════════════════════

def _options_section(ticker: str) -> dict:
    """
    Reuses analyze.compute_gamma_pin and compute_put_call_ratio. Adds
    max-pain (the strike that minimizes total option-writer payout).
    """
    if yf is None:
        return {"available": False}

    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return {"available": False, "reason": "no options chain"}
        # Use the nearest monthly expiration (skip weeklies < 5 DTE)
        target_exp = expirations[0]
        for exp in expirations:
            try:
                d = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
                if 14 <= d <= 60:
                    target_exp = exp
                    break
            except Exception:
                continue
        chain = t.option_chain(target_exp)
        calls = chain.calls
        puts = chain.puts
    except Exception:
        return {"available": False, "reason": "options chain fetch failed"}

    if calls is None or puts is None or calls.empty or puts.empty:
        return {"available": False, "reason": "empty chain"}

    # Spot
    try:
        spot = float(t.history(period="1d")["Close"].iloc[-1])
    except Exception:
        spot = 0.0

    # Gamma pin
    pin = None
    pin_oi = None
    pin_dist = None
    try:
        from analyze import compute_gamma_pin
        pin, pin_oi, pin_dist = compute_gamma_pin(calls, puts, spot)
    except Exception:
        pass

    # Put/call ratio
    pcr = None
    pcr_signal = None
    pcr_interp = None
    try:
        from analyze import compute_put_call_ratio
        pcr, pcr_signal, pcr_interp = compute_put_call_ratio(calls, puts)
    except Exception:
        pass

    # Max pain — strike that minimizes total option writer payout.
    # max_pain(K) = sum_calls max(K - strike, 0) * OI + sum_puts max(strike - K, 0) * OI
    # Wait, that formula is the WRITER's loss at expiration if spot=K.
    # writer_loss(K) = sum_calls (max(K-strike, 0) * OI) + sum_puts (max(strike-K, 0) * OI)
    # max-pain strike is the one that MINIMIZES this. (Writers WIN at max-pain.)
    max_pain = None
    try:
        all_strikes = sorted(set(list(calls["strike"].values) + list(puts["strike"].values)))
        best_k = None
        best_loss = float("inf")
        for k in all_strikes:
            loss = 0.0
            for _, row in calls.iterrows():
                strike = float(row.get("strike", 0))
                oi = float(row.get("openInterest", 0) or 0)
                loss += max(k - strike, 0) * oi
            for _, row in puts.iterrows():
                strike = float(row.get("strike", 0))
                oi = float(row.get("openInterest", 0) or 0)
                loss += max(strike - k, 0) * oi
            if loss < best_loss:
                best_loss = loss
                best_k = k
        max_pain = best_k
    except Exception:
        pass

    # Total open interest
    total_call_oi = int(calls["openInterest"].fillna(0).sum()) if not calls.empty else 0
    total_put_oi = int(puts["openInterest"].fillna(0).sum()) if not puts.empty else 0

    plain_parts = []
    if pin and spot > 0:
        plain_parts.append(
            f"Gamma pin at ${pin:.2f} ({pin_dist:+.1f}% from spot) — strike with most open interest, "
            f"market makers pull price toward it on op-ex."
        )
    if max_pain and spot > 0:
        mp_dist = (max_pain - spot) / spot * 100
        plain_parts.append(f"Max pain ${max_pain:.2f} ({mp_dist:+.1f}%) — option writers' optimal pin.")
    if pcr is not None:
        plain_parts.append(f"Put/Call ratio {pcr:.2f} — {pcr_signal}.")

    return {
        "available":     True,
        "expiration":    target_exp,
        "spot":          _round(spot, 2),
        "gamma_pin":     _round(pin, 2),
        "gamma_pin_oi":  pin_oi,
        "gamma_pin_dist_pct": _round(pin_dist, 2),
        "max_pain":      _round(max_pain, 2),
        "max_pain_dist_pct": _round((max_pain - spot) / spot * 100, 2)
                            if (max_pain and spot > 0) else None,
        "put_call_ratio": _round(pcr, 2),
        "put_call_signal": pcr_signal,
        "put_call_interp": pcr_interp,
        "total_call_oi": total_call_oi,
        "total_put_oi":  total_put_oi,
        "plain_english": " ".join(plain_parts) if plain_parts else "Standard options activity.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def get_insights(ticker: str) -> dict:
    """Build the full insights payload. Each section degrades gracefully
    so a partial failure (e.g. Finnhub down) still returns useful data."""
    ticker = ticker.upper().strip()

    # Validate ticker shape
    if not ticker or not all(c.isalpha() or c == "." for c in ticker):
        return {"error": "Invalid ticker"}

    # Quick spot price for the page header
    spot = None
    company = ticker
    if yf is not None:
        try:
            t = yf.Ticker(ticker)
            hist1d = t.history(period="2d", interval="1d")
            if hist1d is not None and not hist1d.empty:
                spot = float(hist1d["Close"].iloc[-1])
            info = t.info or {}
            company = info.get("longName") or info.get("shortName") or ticker
        except Exception:
            pass

    return {
        "ticker":         ticker,
        "company_name":   company,
        "spot":           _round(spot, 2),
        "generated_at":   datetime.now().isoformat(),
        "insider":        _safe_get(_insider_section, ticker) or {"available": False},
        "institutional":  _safe_get(_institutional_section, ticker) or {"available": False},
        "float":          _safe_get(_float_section, ticker) or {"available": False},
        "flow":           _safe_get(_flow_section, ticker) or {"available": False},
        "support_resistance": _safe_get(_support_resistance_section, ticker) or {"available": False},
        "options":        _safe_get(_options_section, ticker) or {"available": False},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: insights.py <TICKER>"}))
        sys.exit(0)

    ticker = sys.argv[1]
    try:
        result = get_insights(ticker)
        # Produce stable JSON. default=str catches any datetime/Decimal slips.
        print(json.dumps(result, default=str))
    except Exception as e:
        print(json.dumps({"error": str(e)[:200], "ticker": ticker}))
    sys.exit(0)
