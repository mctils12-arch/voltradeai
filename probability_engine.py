#!/usr/bin/env python3
"""
Multidimensional Probability Engine (2026-04-22)
=================================================
Instead of single-dimension stress (risk-on/risk-off), track multiple
axes of market state. Each dimension returns signal in [-1, +1]:
  -1 = strongly unfavorable for the leading side
   0 = neutral
  +1 = strongly favorable for the leading side

Dimensions (MVP):
  1. us_vs_intl       — relative momentum SPY vs ACWX
  2. tech_vs_defensive — momentum XLK vs (XLP+XLU+XLV)/3
  3. growth_vs_value  — momentum IVW vs IVE
  4. large_vs_small   — momentum SPY vs IWM
  5. stocks_vs_commodities — momentum SPY vs DBC
  6. short_vs_long_duration — short bond yield vs long (inflation regime)

Output: composite recommendations for tilting basket allocations.
Starts ADVISORY — logs signals but doesn't rebalance automatically.
Can be wired into basket weights later if signals prove predictive.
"""
import json
import os
import logging
import time
import math
from typing import Dict, Any, List

logger = logging.getLogger("voltrade.probability")

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/tmp"

CACHE_PATH = os.path.join(DATA_DIR, "voltrade_probability_cache.json")
CACHE_TTL_SECONDS = 900  # 15 min


def _momentum_pct(ticker: str, lookback_days: int = 90) -> float:
    """Simple momentum: price change over lookback period, annualized."""
    try:
        import os as _os
        import requests as _rq
        from datetime import datetime as _dt, timedelta as _td
        ALPACA_KEY = _os.environ.get("ALPACA_KEY", "")
        ALPACA_SECRET = _os.environ.get("ALPACA_SECRET", "")
        if not ALPACA_KEY:
            return 0.0
        start = (_dt.now() - _td(days=lookback_days + 10)).strftime("%Y-%m-%d")
        r = _rq.get(
            f"https://data.alpaca.markets/v2/stocks/bars",
            params={"symbols": ticker, "timeframe": "1Day", "start": start, "limit": 120, "feed": "sip"},
            headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET},
            timeout=10,
        )
        bars = r.json().get("bars", {}).get(ticker, [])
        if len(bars) < 30:
            return 0.0
        closes = [float(b["c"]) for b in bars]
        start_price = closes[max(0, len(closes) - lookback_days)]
        end_price = closes[-1]
        if start_price <= 0:
            return 0.0
        return (end_price - start_price) / start_price
    except Exception as e:
        logger.debug(f"momentum fetch failed for {ticker}: {e}")
        return 0.0


def _signal_from_diff(diff: float, threshold: float = 0.05) -> float:
    """Convert a momentum diff to a [-1, +1] signal.
    diff > threshold → +1, diff < -threshold → -1, linear in between."""
    if diff >= threshold:
        return 1.0
    if diff <= -threshold:
        return -1.0
    return diff / threshold


def _dim_us_vs_intl() -> Dict[str, Any]:
    """Positive = US outperforming international."""
    us_mom = _momentum_pct("SPY", 90)
    intl_mom = _momentum_pct("ACWX", 90)
    diff = us_mom - intl_mom
    signal = _signal_from_diff(diff, 0.05)
    return {
        "dimension": "us_vs_intl",
        "signal": round(signal, 3),
        "us_momentum_90d": round(us_mom * 100, 2),
        "intl_momentum_90d": round(intl_mom * 100, 2),
        "diff": round(diff * 100, 2),
        "interpretation": "US outperforming" if signal > 0.3 else ("intl outperforming" if signal < -0.3 else "balanced"),
    }


def _dim_tech_vs_defensive() -> Dict[str, Any]:
    """Positive = tech outperforming defensives."""
    tech = _momentum_pct("XLK", 90)
    xlp = _momentum_pct("XLP", 90)
    xlu = _momentum_pct("XLU", 90)
    xlv = _momentum_pct("XLV", 90)
    defensive = (xlp + xlu + xlv) / 3
    diff = tech - defensive
    signal = _signal_from_diff(diff, 0.05)
    return {
        "dimension": "tech_vs_defensive",
        "signal": round(signal, 3),
        "tech_momentum_90d": round(tech * 100, 2),
        "defensive_momentum_90d": round(defensive * 100, 2),
        "diff": round(diff * 100, 2),
        "interpretation": "risk-on (tech winning)" if signal > 0.3 else ("risk-off (defensives winning)" if signal < -0.3 else "balanced"),
    }


def _dim_growth_vs_value() -> Dict[str, Any]:
    """Positive = growth outperforming value."""
    growth = _momentum_pct("IVW", 90)
    value = _momentum_pct("IVE", 90)
    diff = growth - value
    signal = _signal_from_diff(diff, 0.03)
    return {
        "dimension": "growth_vs_value",
        "signal": round(signal, 3),
        "growth_momentum_90d": round(growth * 100, 2),
        "value_momentum_90d": round(value * 100, 2),
        "diff": round(diff * 100, 2),
        "interpretation": "growth winning (low-rate regime)" if signal > 0.3 else ("value winning (rising-rate regime)" if signal < -0.3 else "balanced"),
    }


def _dim_large_vs_small() -> Dict[str, Any]:
    """Positive = large caps outperforming small caps (risk-off signal)."""
    large = _momentum_pct("SPY", 90)
    small = _momentum_pct("IWM", 90)
    diff = large - small
    signal = _signal_from_diff(diff, 0.04)
    return {
        "dimension": "large_vs_small",
        "signal": round(signal, 3),
        "large_momentum_90d": round(large * 100, 2),
        "small_momentum_90d": round(small * 100, 2),
        "diff": round(diff * 100, 2),
        "interpretation": "large outperforming (risk-off)" if signal > 0.3 else ("small outperforming (risk-on)" if signal < -0.3 else "balanced"),
    }


def _dim_stocks_vs_commodities() -> Dict[str, Any]:
    """Positive = stocks outperforming commodities (low inflation regime)."""
    stocks = _momentum_pct("SPY", 90)
    commodities = _momentum_pct("DBC", 90)
    diff = stocks - commodities
    signal = _signal_from_diff(diff, 0.05)
    return {
        "dimension": "stocks_vs_commodities",
        "signal": round(signal, 3),
        "stocks_momentum_90d": round(stocks * 100, 2),
        "commodities_momentum_90d": round(commodities * 100, 2),
        "diff": round(diff * 100, 2),
        "interpretation": "stocks winning (low inflation)" if signal > 0.3 else ("commodities winning (inflation regime)" if signal < -0.3 else "balanced"),
    }


def _dim_short_vs_long_duration() -> Dict[str, Any]:
    """Positive = short-term bonds outperforming long-term (rising rate/inflation regime)."""
    short = _momentum_pct("SHY", 90)
    long_b = _momentum_pct("TLT", 90)
    diff = short - long_b
    signal = _signal_from_diff(diff, 0.04)
    return {
        "dimension": "short_vs_long_duration",
        "signal": round(signal, 3),
        "short_momentum_90d": round(short * 100, 2),
        "long_momentum_90d": round(long_b * 100, 2),
        "diff": round(diff * 100, 2),
        "interpretation": "short winning (rising-rate/inflation)" if signal > 0.3 else ("long winning (deflation/easing)" if signal < -0.3 else "balanced"),
    }


def compute_all_dimensions() -> Dict[str, Any]:
    """Compute all dimension signals. Cached 15 min."""
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                cached = json.load(f)
            if time.time() - cached.get("_cached_at", 0) < CACHE_TTL_SECONDS:
                return cached
        except Exception:
            pass

    dimensions = [
        _dim_us_vs_intl(),
        _dim_tech_vs_defensive(),
        _dim_growth_vs_value(),
        _dim_large_vs_small(),
        _dim_stocks_vs_commodities(),
        _dim_short_vs_long_duration(),
    ]

    risk_on_signals = sum(1 for d in dimensions if d["signal"] > 0.3)
    risk_off_signals = sum(1 for d in dimensions if d["signal"] < -0.3)
    inflation_signals = 0
    if dimensions[4]["signal"] < -0.3:
        inflation_signals += 1
    if dimensions[5]["signal"] > 0.3:
        inflation_signals += 1

    if risk_on_signals >= 4:
        regime_summary = "strongly risk-on"
        recommendation = "tilt toward tech, growth, small-cap"
    elif risk_off_signals >= 4:
        regime_summary = "strongly risk-off"
        recommendation = "tilt toward defensives, value, large-cap"
    elif inflation_signals >= 2:
        regime_summary = "inflation regime"
        recommendation = "tilt toward commodities, short duration, real assets"
    else:
        regime_summary = "mixed signals"
        recommendation = "maintain balanced basket weights"

    suggested_tilts = {}
    for d in dimensions:
        sig = d["signal"]
        if d["dimension"] == "us_vs_intl":
            suggested_tilts["us_weight_delta"] = round(sig * 0.05, 3)
            suggested_tilts["intl_weight_delta"] = round(-sig * 0.05, 3)
        elif d["dimension"] == "tech_vs_defensive":
            suggested_tilts["tech_weight_delta"] = round(sig * 0.03, 3)
            suggested_tilts["defensive_weight_delta"] = round(-sig * 0.03, 3)
        elif d["dimension"] == "stocks_vs_commodities":
            suggested_tilts["commodity_weight_delta"] = round(-sig * 0.03, 3)

    result = {
        "dimensions": dimensions,
        "summary": {
            "regime": regime_summary,
            "recommendation": recommendation,
            "risk_on_count": risk_on_signals,
            "risk_off_count": risk_off_signals,
            "inflation_count": inflation_signals,
        },
        "suggested_tilts": suggested_tilts,
        "mode": "advisory",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "_cached_at": time.time(),
    }

    try:
        import tempfile
        dirname = os.path.dirname(CACHE_PATH) or "."
        fd, tmp = tempfile.mkstemp(dir=dirname, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp, CACHE_PATH)
    except Exception as e:
        logger.warning(f"probability cache write failed: {e}")

    return result


if __name__ == "__main__":
    r = compute_all_dimensions()
    print(f"Regime: {r['summary']['regime']}")
    print(f"Recommendation: {r['summary']['recommendation']}")
    print(f"Risk-on signals: {r['summary']['risk_on_count']}/6")
    print(f"Risk-off signals: {r['summary']['risk_off_count']}/6")
    print(f"Inflation signals: {r['summary']['inflation_count']}/2")
    print()
    for d in r["dimensions"]:
        print(f"  {d['dimension']}: {d['signal']:+.2f} — {d['interpretation']}")
