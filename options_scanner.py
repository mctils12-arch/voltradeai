#!/usr/bin/env python3
"""
VolTradeAI — Standalone Options Scanner
=========================================
Finds options setups that CANNOT be found by the stock scanner alone.

WHY THIS EXISTS:
  The main stock scanner only considers options AFTER a stock already scores 70+.
  That means it can only answer: "this stock looks good — should we trade it with
  options instead of shares?"

  But that misses entire categories of options alpha:

  1. HIGH-IV PREMIUM SELLING  → Works best on stocks that are VOLATILE but NOT moving
     today (low quick_score). A straddle on TSLA before earnings is great even if
     TSLA isn't on today's mover list.

  2. EARNINGS IV CRUSH        → The setup is the CALENDAR (earnings in 2-7 days), not
     the stock's daily movement. These stocks are often flat the day before earnings.

  3. VXX SPIKE / MARKET PANIC → When VXX > 1.30, selling puts on the SPY or QQQ is a
     high-probability trade (market overpays for protection). This has zero connection
     to which stocks moved today.

  4. CHEAP IV / BREAKOUT BUY  → Stocks sitting in a tight range with IV at 52-week
     lows are cheap lottery tickets. Options are underpriced. Stock scanner misses these
     because nothing is moving.

  5. GAMMA PIN (0DTE / 1DTE)  → Stocks with massive open interest at a nearby strike
     get "pinned" there by market makers hedging. Completely independent of daily moves.

HOW IT WORKS:
  - Runs in parallel with the main scan_market() call in bot_engine.py
  - Uses the same Alpaca SIP + OPRA data feeds
  - Returns options-specific trade recommendations in the same format as bot_engine.py
  - Results are merged with stock scan results and compete for the same position slots
  - ML model in ml_model_v2.py is extended to include options-specific features

SETUPS SCANNED:
  SETUP 1: Earnings IV Crush    — sell straddle/iron condor 2-7 days before earnings
  SETUP 2: VXX Panic Put Sale   — sell SPY/QQQ puts when VXX spikes above 1.30
  SETUP 3: High-IV Premium Sale — sell premium on high-IV stocks (IVR > 70)
  SETUP 4: Low-IV Breakout Buy  — buy calls/puts on stocks with IV at 52-week low
  SETUP 5: Gamma Pin            — trade toward the pinned strike on expiry day

Author: VolTradeAI System
"""

import os
import sys
import json
import time
import logging
import requests
# Alpaca rate limiter — prevents silent 429 errors during parallel scans
try:
    from alpaca_rate_limiter import alpaca_throttle
    _HAS_ALPACA_THROTTLE = True
except ImportError:
    _HAS_ALPACA_THROTTLE = False
    class _NoThrottle:
        def acquire(self): pass
    alpaca_throttle = _NoThrottle()
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

logger = logging.getLogger("options_scanner")

# Import options ML scorer — falls back to rules if model not trained yet
def _options_ml_score(features: dict) -> float:
    """
    Call options_ml_score from ml_model_v2.
    Returns 0.0-1.0 probability of profit for this options setup.
    Falls back to rules-based win rates if model unavailable.
    """
    try:
        from ml_model_v2 import options_ml_score
        return options_ml_score(features)
    except Exception:
        # Hardcoded fallback win rates (research-backed)
        iv_rank   = float(features.get("iv_rank", features.get("iv_rank_proxy", 50)) or 50)
        vxx_ratio = float(features.get("vxx_ratio", 1.0) or 1.0)
        days_earn = float(features.get("days_to_earn", 99) or 99)
        if vxx_ratio >= 1.30:      return 0.70  # Panic put sale: ~70% WR
        if 1 <= days_earn <= 7 and iv_rank > 60: return 0.67  # Earnings crush: ~67%
        if iv_rank > 70:           return 0.65  # High-IV sale: ~65%
        if iv_rank < 20:           return 0.55  # Low-IV buy: ~55%
        return 0.50


def _build_setup_features(
    vxx_ratio: float,
    spy_vs_ma50: float,
    iv_rank: float,
    vrp: float,
    days_to_earn: float = 99.0,
    momentum_1m: float = 0.0,
) -> dict:
    """
    Build the 28-feature dict that options_ml_score() expects.

    These are the same features the model was trained on (both synthetic and
    real feedback). The regime features (vxx_ratio, iv_rank, spy_vs_ma50) are
    the most predictive for options setups — they determine which regime we're
    in and whether premium selling or premium buying has the edge.

    Stock-specific features (RSI, momentum, float) are set to neutral defaults
    since the options scanner operates at the market/IV level, not per-stock.
    """
    regime_score = max(0.0, min(100.0,
        (spy_vs_ma50 - 0.94) / 0.12 * 50 + (1.3 - vxx_ratio) / 0.6 * 30 + 20))
    return {
        # Regime features (MOST important for options — drive the ML prediction)
        "vxx_ratio":        vxx_ratio,
        "iv_rank_proxy":    iv_rank,
        "iv_rank":          iv_rank,
        "spy_vs_ma50":      spy_vs_ma50,
        "regime_score":     regime_score,
        "vrp":              vrp,
        "vrp_magnitude":    abs(vrp),
        "days_to_earn":     days_to_earn,
        # Stock features set to neutral (not applicable at market/IV level)
        "momentum_1m":      momentum_1m,
        "momentum_3m":      0.0,
        "rsi_14":           50.0,
        "volume_ratio":     1.0,
        "vwap_position":    0.0,
        "adx":              20.0,
        "ewma_vol":         vxx_ratio * 15,  # VXX level ≈ market vol proxy
        "range_pct":        0.0,
        "price_vs_52w_high": 0.0,
        "atr_pct":          0.0,
        "markov_state":     1.0,
        "sector_momentum":  0.0,
        "change_pct_today": 0.0,
        "above_ma10":       1.0 if spy_vs_ma50 > 1.0 else 0.0,
        "trend_strength":   0.0,
        # New features added to OPTIONS_FEATURE_COLS (cross_sec_rank, earnings_surprise,
        # put_call_proxy, vol_of_vol, frac_diff_price, idiosyncratic_ret).
        # These are stock-specific signals not available at market/IV level, so we
        # default them to neutral values (0.0 / 0.5) to match training distribution.
        "cross_sec_rank":   0.5,   # neutral rank — no per-stock cross-sectional data here
        "earnings_surprise": 0.0,  # unknown at market-level scan; set by _build_setup_features callers if available
        "put_call_proxy":   -1.0 if vrp > 8 else (1.0 if vrp < -5 else 0.0),  # derived from VRP direction
        "vol_of_vol":       0.0,   # requires VXX history not pre-fetched here; neutral default
        "frac_diff_price":  0.0,   # requires long price series; neutral default
        "idiosyncratic_ret": 0.0,  # no per-stock return available at market-IV level
    }

# ── Alpaca credentials ────────────────────────────────────────────────────────
_ALPACA_KEY    = os.environ.get("ALPACA_KEY", "")
_ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_DATA    = "https://data.alpaca.markets"
ALPACA_TRADE   = "https://paper-api.alpaca.markets"  # Use paper endpoint — never live

# ── Finnhub for earnings calendar ─────────────────────────────────────────────
FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")

# ── Data directory ────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("VOLTRADE_DATA_DIR", "/data/voltrade")

def _headers() -> dict:
    return {"APCA-API-KEY-ID": _ALPACA_KEY, "APCA-API-SECRET-KEY": _ALPACA_SECRET}

def _et_now_hour() -> float:
    """Return current ET hour (with fractional minutes), DST-aware."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    return now_et.hour + now_et.minute / 60.0

def _is_regular_hours() -> bool:
    """True if within 9:30am-4:00pm ET."""
    try:
        et_time = _et_now_hour()
        return 9.5 <= et_time < 16.0
    except Exception:
        return False

def _et_now() -> datetime:
    """Current time in ET, DST-aware."""
    return datetime.now(ZoneInfo("America/New_York"))

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

# OPTIMIZATION 2026-04-20: cache VXX/SPY fetches within a scan cycle
# Previously these were called 10+ times per scan (once per setup detector)
# each hitting Alpaca for the same bars. 60s cache eliminates redundant I/O.
_regime_cache: dict = {}
_regime_cache_ts: dict = {}
_REGIME_CACHE_TTL = 60  # seconds

def _get_vxx_ratio_raw() -> float:
    """
    VXX / 30-day average VXX. > 1.0 = above-average fear.
    > 1.30 = panic. Used to trigger VXX Panic Put Sale setup.
    Uncached — internal use only. Callers should use _get_vxx_ratio().
    """
    try:
        start = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
        r = alpaca_throttle.acquire()
        r = requests.get(
            f"{ALPACA_DATA}/v2/stocks/bars",
            params={"symbols": "VXX", "timeframe": "1Day",
                    "start": start, "limit": 40, "feed": "sip"},
            headers=_headers(), timeout=8
        )
        bars = r.json().get("bars", {}).get("VXX", [])
        if not bars or len(bars) < 5:
            return 1.0
        closes = [float(b["c"]) for b in bars]
        avg30  = sum(closes[-30:]) / len(closes[-30:])
        latest = closes[-1]
        return round(latest / avg30, 4) if avg30 > 0 else 1.0
    except Exception:
        return 1.0


def _get_vxx_ratio() -> float:
    """Cached wrapper — 60s TTL. Prevents redundant Alpaca calls."""
    import time
    now = time.monotonic()
    if "vxx_ratio" in _regime_cache and now - _regime_cache_ts.get("vxx_ratio", 0) < _REGIME_CACHE_TTL:
        return _regime_cache["vxx_ratio"]
    val = _get_vxx_ratio_raw()
    _regime_cache["vxx_ratio"] = val
    _regime_cache_ts["vxx_ratio"] = now
    return val


def _get_spy_vs_ma50_raw() -> float:
    """SPY price / SPY 50-day MA. > 1.0 = above MA (healthy)."""
    try:
        start = (datetime.now() - timedelta(days=80)).strftime("%Y-%m-%d")
        alpaca_throttle.acquire()
        r = requests.get(
            f"{ALPACA_DATA}/v2/stocks/bars",
            params={"symbols": "SPY", "timeframe": "1Day",
                    "start": start, "limit": 60, "feed": "sip"},
            headers=_headers(), timeout=8
        )
        bars = r.json().get("bars", {}).get("SPY", [])
        if not bars or len(bars) < 10:
            return 1.0
        closes = [float(b["c"]) for b in bars]
        ma50   = sum(closes[-50:]) / len(closes[-50:]) if len(closes) >= 50 else sum(closes) / len(closes)
        return round(closes[-1] / ma50, 4) if ma50 > 0 else 1.0
    except Exception:
        return 1.0


def _get_spy_vs_ma50() -> float:
    """Cached wrapper — 60s TTL."""
    import time
    now = time.monotonic()
    if "spy_vs_ma50" in _regime_cache and now - _regime_cache_ts.get("spy_vs_ma50", 0) < _REGIME_CACHE_TTL:
        return _regime_cache["spy_vs_ma50"]
    val = _get_spy_vs_ma50_raw()
    _regime_cache["spy_vs_ma50"] = val
    _regime_cache_ts["spy_vs_ma50"] = now
    return val


# OPTIMIZATION 2026-04-20: Module-level cache for options chains.
# Setup detectors often fetch the same ticker's chain repeatedly within
# a scan cycle (CSP + earnings + panic put detectors all fetch SPY's
# chain). Cache keyed by (ticker, min_days, max_days) with 3-min TTL.
_chain_cache: dict = {}
_chain_cache_ts: dict = {}
_CHAIN_CACHE_TTL = 180  # 3 min — options quotes change but not dramatically

def _fetch_options_chain(ticker: str, price: float,
                          min_days: int = 7, max_days: int = 60) -> list:
    """
    Fetch live options chain from Alpaca OPRA feed.
    Cached for 3 min to avoid redundant paginated fetches.
    Returns ALL contracts in the expiry/strike window, paginated.

    STRIKE RANGE: ±30% of current price.
      Why ±30% not ±15%: Iron condor long wings (protection legs) sit at
      ~15% OTM. A ±15% band would clip them. ±30% gives full coverage while
      avoiding deep DITM/DOTM contracts that have no liquidity.

    PAGINATION: Alpaca caps at 1000 per page. SPY has ~3,700 contracts in a
      30-day window. We paginate up to 5 pages to get the full chain.

    DELTA: OPRA provides delta for ~70% of contracts. For the other 30%,
      we compute a moneyness-based proxy:
        - At-the-money (strike ≈ price):      delta ≈ 0.50
        - 5% OTM call or put:                 delta ≈ 0.30
        - 10% OTM:                            delta ≈ 0.20 (iron condor short wing)
        - 15% OTM:                            delta ≈ 0.12
        - 20% OTM:                            delta ≈ 0.07 (iron condor long wing)
      These are conservative approximations — real delta varies with IV and DTE.
    """
    # Cache check
    cache_key = f"{ticker}_{min_days}_{max_days}"
    import time as _t
    _now_mono = _t.monotonic()
    if cache_key in _chain_cache and _now_mono - _chain_cache_ts.get(cache_key, 0) < _CHAIN_CACHE_TTL:
        return _chain_cache[cache_key]

    try:
        now     = datetime.now()
        min_exp = (now + timedelta(days=min_days)).strftime("%Y-%m-%d")
        max_exp = (now + timedelta(days=max_days)).strftime("%Y-%m-%d")
        # ±30% strike band — covers all meaningful strikes including condor wings
        min_k = price * 0.70
        max_k = price * 1.30

        all_snapshots: dict = {}
        next_token: Optional[str] = None
        pages = 0

        while pages < 5:  # Max 5 pages = 5,000 contracts (more than enough for any single name)
            params: dict = {
                "feed": "opra", "limit": 1000,
                "expiration_date_gte": min_exp,
                "expiration_date_lte": max_exp,
                "strike_price_gte":   str(min_k),
                "strike_price_lte":   str(max_k),
            }
            if next_token:
                params["page_token"] = next_token

            alpaca_throttle.acquire()

            r = requests.get(
                f"{ALPACA_DATA}/v1beta1/options/snapshots/{ticker}",
                params=params, headers=_headers(), timeout=12
            )
            if r.status_code != 200:
                break
            data = r.json()
            all_snapshots.update(data.get("snapshots", {}))
            pages += 1
            next_token = data.get("next_page_token")
            if not next_token:
                break  # No more pages

        if not all_snapshots:
            return []

        contracts = []
        for occ, snap in all_snapshots.items():
            quote  = snap.get("latestQuote", {})
            greeks = snap.get("greeks", {})
            body   = occ[len(ticker):]
            if len(body) < 15:
                continue

            exp_raw  = "20" + body[:6]
            exp_date = f"{exp_raw[:4]}-{exp_raw[4:6]}-{exp_raw[6:8]}"
            opt_type = "call" if body[6] == "C" else "put"
            strike   = int(body[7:]) / 1000

            bid = float(quote.get("bp", 0) or 0)
            ask = float(quote.get("ap", 0) or 0)
            mid = (bid + ask) / 2 if (bid + ask) > 0 else 0

            # Skip contracts with no market (no bid/ask = no liquidity)
            if mid <= 0:
                continue

            # Delta: use OPRA value if present, otherwise moneyness proxy
            raw_delta = greeks.get("delta")
            if raw_delta is not None and raw_delta != "N/A":
                try:
                    delta = float(raw_delta)
                except (ValueError, TypeError):
                    delta = 0.0
            else:
                delta = 0.0

            # Moneyness-based delta proxy when OPRA delta unavailable
            # Formula: approximate Black-Scholes delta from moneyness
            # Call delta increases as strike decreases (ITM). Put delta: mirror.
            if delta == 0.0 and price > 0:
                moneyness = strike / price  # < 1 = OTM call / ITM put, > 1 = ITM call / OTM put
                if opt_type == "call":
                    if moneyness <= 0.85:   delta =  0.90  # Deep ITM
                    elif moneyness <= 0.90: delta =  0.75
                    elif moneyness <= 0.95: delta =  0.55
                    elif moneyness <= 1.00: delta =  0.50  # ATM
                    elif moneyness <= 1.05: delta =  0.35
                    elif moneyness <= 1.10: delta =  0.22
                    elif moneyness <= 1.15: delta =  0.14
                    elif moneyness <= 1.20: delta =  0.08
                    else:                   delta =  0.04  # Deep OTM
                else:  # put (mirror of call)
                    if moneyness >= 1.15:   delta = -0.90
                    elif moneyness >= 1.10: delta = -0.75
                    elif moneyness >= 1.05: delta = -0.55
                    elif moneyness >= 1.00: delta = -0.50
                    elif moneyness >= 0.95: delta = -0.35
                    elif moneyness >= 0.90: delta = -0.22
                    elif moneyness >= 0.85: delta = -0.14
                    elif moneyness >= 0.80: delta = -0.08
                    else:                   delta = -0.04

            iv     = float(greeks.get("iv", 0) or 0)
            oi     = int(snap.get("openInterest", 0) or 0)
            volume = int(snap.get("volume", 0) or 0)

            try:
                days_out = (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.now()).days
            except ValueError:
                continue

            contracts.append({
                "occ_symbol":  occ,
                "ticker":      ticker,
                "exp_date":    exp_date,
                "opt_type":    opt_type,
                "strike":      strike,
                "bid":         bid,
                "ask":         ask,
                "mid":         mid,
                "iv":          iv,
                "delta":       delta,
                "delta_real":  (raw_delta is not None and raw_delta != "N/A"),  # True if from OPRA
                "oi":          oi,
                "volume":      volume,
                "days_out":    days_out,
                "moneyness":   round(strike / price, 4) if price > 0 else 1.0,
                "spread_pct":  (ask - bid) / ask if ask > 0 else 1.0,
            })

        contracts.sort(key=lambda x: (x["exp_date"], x["strike"]))
        # Cache the result (optimization 2026-04-20)
        _chain_cache[cache_key] = contracts
        _chain_cache_ts[cache_key] = _now_mono
        # Keep cache bounded: drop oldest when > 50 entries
        if len(_chain_cache) > 50:
            _oldest_key = min(_chain_cache_ts, key=_chain_cache_ts.get)
            _chain_cache.pop(_oldest_key, None)
            _chain_cache_ts.pop(_oldest_key, None)
        return contracts

    except Exception as e:
        logger.debug(f"[{ticker}] Options chain fetch failed: {e}")
        return []


def _find_by_delta(contracts: list, opt_type: str, target_delta: float,
                   tolerance: float = 0.08) -> Optional[dict]:
    """
    Find the best contract closest to a target delta.

    Used by all setup selectors instead of "closest to ATM".
    Delta-based selection is the industry standard because:
      - 50-delta = ATM (straddle center)
      - 20-delta = standard iron condor short wing (~1 SD OTM)
      - 10-delta = standard condor long wing (protection leg)
      - 30-delta = tighter condor or covered call placement

    Falls back to moneyness proxy when OPRA delta is unavailable.

    Args:
        contracts: List of contract dicts from _fetch_options_chain()
        opt_type:  "call" or "put"
        target_delta: Absolute value (e.g. 0.20 for 20-delta)
        tolerance: How far from target we'll accept (default ±0.08)

    Returns:
        Best matching contract or None
    """
    filtered = [
        c for c in contracts
        if c["opt_type"] == opt_type
        and abs(abs(c["delta"]) - target_delta) <= tolerance
        and c["bid"] > 0          # Must have a real bid (liquidity)
        and c["spread_pct"] < 0.15  # Max 15% bid/ask spread
    ]
    if not filtered:
        # Widen tolerance as fallback
        filtered = [
            c for c in contracts
            if c["opt_type"] == opt_type
            and abs(abs(c["delta"]) - target_delta) <= tolerance * 2
            and c["bid"] > 0
        ]
    if not filtered:
        return None
    # Among candidates, prefer: highest OI (liquidity) then tightest spread
    return max(filtered, key=lambda c: (c["oi"], -c["spread_pct"]))


def _fetch_iv_rank(ticker: str) -> Optional[float]:
    """
    Compute per-stock IV rank combining REAL implied vol (from ATM options)
    with historical vol context.

    FIX 2026-04-20 (Bug #4): Previous version computed HV rank and called it
    "IV rank" — misleading. True IV rank uses implied vol from option prices;
    HV rank uses historical realized vol. They correlate but diverge exactly
    at the inflection points (pre-earnings, vol spikes) where the difference
    matters most for VRP trades.

    New method:
      1. Pull nearest-expiry ATM options chain (real IV via mid-price)
      2. Compare current ATM IV against trailing 90-day HV as a normalized
         "IV/HV ratio" — >1.3 means market implying much more future vol
         than realized (rich premium, good for selling)
      3. Compute HV-rank as before for the 52-week percentile context
      4. Combine: iv_rank = weighted(iv_hv_ratio_rank, hv_rank)

    Falls back to pure HV rank if options chain fetch fails.
    Returns 0-100 where >70 means genuinely rich premium.
    """
    try:
        import numpy as _np

        # Fetch 400 days of bars for HV calculation + recent price
        start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        alpaca_throttle.acquire()
        r = requests.get(
            f"{ALPACA_DATA}/v2/stocks/bars",
            params={"symbols": ticker, "timeframe": "1Day",
                    "start": start, "limit": 300, "feed": "sip"},
            headers=_headers(), timeout=10
        )
        bars = r.json().get("bars", {}).get(ticker, [])
        if len(bars) < 60:
            return None

        closes = [float(b["c"]) for b in bars]
        rets = [_np.log(closes[i]/closes[i-1]) for i in range(1, len(closes)) if closes[i-1] > 0]
        if len(rets) < 50:
            return None

        # Rolling 30-day HV (annualized)
        hvs = []
        for i in range(30, len(rets)):
            hv = _np.std(rets[i-30:i]) * _np.sqrt(252) * 100
            hvs.append(hv)
        if not hvs:
            return None
        current_hv = hvs[-1]
        hv_lo = min(hvs)
        hv_hi = max(hvs)
        hv_rank = 50.0 if hv_hi <= hv_lo else (current_hv - hv_lo) / (hv_hi - hv_lo) * 100

        # ── Now get REAL implied vol from nearest ATM option ──────────────
        current_price = closes[-1]
        try:
            # Find nearest expiry (7-45 DTE)
            min_exp = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            max_exp = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
            # Get options chain snapshot from Alpaca
            alpaca_throttle.acquire()
            chain_r = requests.get(
                f"{ALPACA_DATA}/v1beta1/options/snapshots/{ticker}",
                params={"expiration_date_gte": min_exp,
                        "expiration_date_lte": max_exp,
                        "limit": 100, "feed": "indicative"},
                headers=_headers(), timeout=10
            )
            snapshots = chain_r.json().get("snapshots", {})
            # Find ATM call (strike closest to spot) with valid IV
            atm_iv = None
            best_dist = float("inf")
            for occ_sym, snap in snapshots.items():
                # OCC format: TICKER + YYMMDD + C/P + strike*1000 (8 digits)
                try:
                    strike_part = occ_sym[-8:]
                    strike = float(strike_part) / 1000
                except Exception:
                    continue
                dist = abs(strike - current_price) / current_price
                if dist > 0.05:  # skip far-OTM/ITM, want within 5% of spot
                    continue
                iv = snap.get("impliedVolatility") or snap.get("iv")
                if iv is None:
                    # fall back: compute IV from mid-price via Black-Scholes
                    # (expensive, skip for now — use snapshot IV only)
                    continue
                if dist < best_dist:
                    best_dist = dist
                    atm_iv = float(iv) * 100  # Alpaca returns IV as fraction

            if atm_iv is not None and current_hv > 0:
                # IV/HV ratio — true premium-richness signal
                iv_hv_ratio = atm_iv / current_hv
                # Map ratio to 0-100: ratio=1.0 → 50, ratio=1.5 → 75, ratio=0.7 → 30
                iv_hv_rank = max(0.0, min(100.0, (iv_hv_ratio - 0.5) / 1.5 * 100))
                # Combine: 70% IV/HV ratio (the true premium signal), 30% HV rank (context)
                combined = iv_hv_rank * 0.7 + hv_rank * 0.3
                return round(float(combined), 1)
        except Exception:
            pass  # options chain fetch failed — fall back to HV rank

        # Fallback: return HV rank (clearly marked in docstring)
        return round(float(hv_rank), 1)
    except Exception:
        return None



def _fetch_earnings_calendar(days_ahead: int = 10) -> Dict[str, int]:
    """
    Fetch earnings calendar from Finnhub.
    Returns {ticker: days_to_earnings} for all companies reporting within days_ahead.
    """
    result: Dict[str, int] = {}
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        end   = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={"from": today, "to": end, "token": FINNHUB_KEY},
            timeout=10
        )
        for item in r.json().get("earningsCalendar", []):
            sym = item.get("symbol", "")
            dt  = item.get("date", "")
            if sym and dt:
                try:
                    days = (datetime.strptime(dt, "%Y-%m-%d") - datetime.now()).days
                    if 0 <= days <= days_ahead:
                        result[sym] = days
                except Exception:
                    pass
    except Exception:
        pass
    return result


def _fetch_price(ticker: str) -> Optional[float]:
    """Get latest price for ticker."""
    try:
        alpaca_throttle.acquire()
        r = requests.get(
            f"{ALPACA_DATA}/v2/stocks/snapshots?symbols={ticker}&feed=sip",
            headers=_headers(), timeout=6
        )
        snap = r.json().get(ticker, {})
        bar  = snap.get("latestTrade", snap.get("minuteBar", {}))
        return float(bar.get("p", bar.get("c", 0)) or 0) or None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: SETUP DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

def _setup_earnings_iv_crush(
    ticker: str, price: float, days_to_earnings: int,
    vxx_ratio: float
) -> Optional[dict]:
    """
    SETUP 1: EARNINGS IV CRUSH

    What is it:
      Before earnings, option sellers push IV up because nobody knows which way
      the stock will move. After earnings, IV collapses back to normal — this is
      called "IV crush". If the amount IV is overpricing the expected move is LARGER
      than the actual average move this company makes post-earnings, selling a
      straddle (sell call + sell put) BEFORE earnings is profitable.

    Best conditions:
      - 2-7 days before earnings (not too early — IV is highest right before)
      - Historical move < IV-implied move (the stock moves LESS than options expect)
      - Stock has liquid options (bid/ask spread < 5%)
      - NOT in a panic regime (VXX > 1.30 = general market vol is high already)

    Risk:
      - If the stock makes a huge unexpected move (news, product failure), the loss
        is theoretically unlimited on one side. We use iron condors (capped loss) for
        anything with a history of big moves.
    """
    if not _is_regular_hours():
        return None  # Options only trade regular hours (exchange rule)
    if vxx_ratio >= 1.30:
        return None  # Panic mode: don't add risk, market IV already elevated
    # Fix 1 (v1.0.23): accept days=0 (earnings tomorrow morning)
    if not (0 <= days_to_earnings <= 7):
        return None

    # For days=0 (report tomorrow bmo), fetch contracts expiring within 2 weeks
    chain_min = max(days_to_earnings, 1)
    contracts = _fetch_options_chain(ticker, price, min_days=chain_min, max_days=days_to_earnings + 14)
    if not contracts:
        return None

    # Select ATM contracts using 50-delta (the industry standard for straddles)
    # 50-delta = at-the-money = maximum premium, maximum IV capture
    # We use _find_by_delta() instead of "closest to price" because:
    #   - A stock at $150.50 has $150 and $151 strikes — closest-to-price
    #     picks one arbitrarily. 50-delta picks the one the market prices as ATM.
    best_call = _find_by_delta(contracts, "call", target_delta=0.50, tolerance=0.10)
    best_put  = _find_by_delta(contracts, "put",  target_delta=0.50, tolerance=0.10)

    if not best_call or not best_put:
        return None

    # Reject if spreads are too wide (liquidity check)
    # FIX (2026-04-10): tightened from 0.15 to 0.10 — every straddle in Apr 3-10
    # lost money to the bid-ask spread. 10% is the max tolerable for round-trip.
    if best_call["spread_pct"] > 0.10 or best_put["spread_pct"] > 0.10:
        return None

    # IV-implied move: straddle price / stock price = expected % move
    straddle_price = best_call["mid"] + best_put["mid"]
    iv_implied_move_pct = (straddle_price / price) * 100 if price > 0 else 0

    if iv_implied_move_pct < 2.0:
        return None  # Straddle too cheap — not worth the risk

    # FIX (2026-04-10): Minimum expected profit must exceed 2x spread cost.
    # Every straddle in Apr 3-10 was opened and closed within minutes, losing
    # to the bid-ask spread. If the expected profit from IV crush doesn't cover
    # 2x the round-trip spread cost, it's not worth entering.
    _call_spread = best_call.get("ask", 0) - best_call.get("bid", 0)
    _put_spread = best_put.get("ask", 0) - best_put.get("bid", 0)
    _total_spread_cost = _call_spread + _put_spread  # Round-trip spread loss
    # Expected profit from IV crush: ~30-50% of straddle price for typical earnings
    _expected_profit = straddle_price * 0.30  # Conservative: 30% of premium
    if _expected_profit < _total_spread_cost * 2:
        return None  # Expected profit doesn't cover 2x spread cost

    # Minimum IV: must be at least 40% annualized (otherwise not enough premium)
    avg_iv = (best_call["iv"] + best_put["iv"]) / 2
    if avg_iv < 0.40:
        return None

    # Strategy: always iron condor for earnings plays (v1.0.34)
    #
    # Previous: short_straddle when iv_implied_move <= 8%, iron_condor above.
    # Problem:  short straddles have unlimited downside risk. One earnings
    #           surprise (stock gaps 15-20%) wipes out months of premium.
    #           This destroys drawdown metrics and Sortino ratio.
    # Research: SteadyOptions backtest — iron condors had 0.05 Sharpe vs
    #           strangles 0.64 on SPY, BUT iron condors had defined max loss.
    #           For a system targeting low drawdowns + high Sortino, capping
    #           downside per trade is more important than maximizing premium.
    # Decision: iron_condor only. Gives up ~20-30% of premium per trade but
    #           eliminates tail-risk blowups that kill compounding.
    strategy = "iron_condor"
    if iv_implied_move_pct > 8.0:
        action_label = "SELL IRON CONDOR (earnings IV crush — wide wings)"
    else:
        action_label = "SELL IRON CONDOR (earnings IV crush — tight wings)"

    # Score formula:
    #  Base 60 + IV premium bonus (how much IV is above normal) + liquidity bonus
    iv_rank = _fetch_iv_rank(ticker) or 50.0
    iv_bonus = min(20, (iv_rank - 50) * 0.4) if iv_rank > 50 else 0
    liq_bonus = 5 if (best_call["oi"] > 500 and best_put["oi"] > 500) else 0
    urgency_bonus = 8 if days_to_earnings <= 2 else (4 if days_to_earnings <= 4 else 0)
    rules_score = min(95, 60 + iv_bonus + liq_bonus + urgency_bonus)

    # ML blend: 60% rules + 40% ML (same ratio as stock model)
    # Features: IV rank, VXX ratio, SPY MA, days_to_earnings, VRP
    # Research basis: IV rank + days_to_earn are the two strongest predictors of
    # earnings crush profitability (Goyal & Saretto 2009, Muravyev 2016)
    vrp_est = max(-20.0, min(20.0, (vxx_ratio - 1.0) * 50))
    ml_features = _build_setup_features(
        vxx_ratio=vxx_ratio, spy_vs_ma50=_get_spy_vs_ma50(),
        iv_rank=iv_rank, vrp=vrp_est, days_to_earn=float(days_to_earnings))
    ml_prob = _options_ml_score(ml_features)
    ml_score_val = ml_prob * 100
    score = round(min(95, rules_score * 0.60 + ml_score_val * 0.40), 1)

    return {
        "ticker":          ticker,
        "setup":           "earnings_iv_crush",
        "score":           score,
        "price":           price,
        "action_label":    action_label,
        "options_strategy": strategy,
        "days_to_earnings": days_to_earnings,
        "iv_implied_move": round(iv_implied_move_pct, 2),
        "straddle_price":  round(straddle_price, 2),
        "avg_iv":          round(avg_iv * 100, 1),
        "iv_rank":         iv_rank,
        "best_call_occ":   best_call["occ_symbol"],
        "best_put_occ":    best_put["occ_symbol"],
        "reasoning": (
            f"Earnings in {days_to_earnings}d. Options implying {iv_implied_move_pct:.1f}% move. "
            f"IV rank {iv_rank:.0f}/100. Straddle premium ${straddle_price:.2f} ({iv_implied_move_pct:.1f}% of stock). "
            f"Strategy: {strategy}."
        ),
        "source": "options_scanner",
        "side": "sell",  # Selling premium
    }


def _setup_vxx_panic_put_sale(vxx_ratio: float, spy_vs_ma50: float) -> Optional[dict]:
    """
    SETUP 2: VXX PANIC PUT SALE (sell puts on SPY or QQQ)

    What is it:
      When VXX (the fear index) spikes 30%+ above its 30-day average, the options
      market is OVERPRICING protection. Put buyers are panicking and paying too much.
      This is the best time to SELL puts on SPY or QQQ.

      Why? The market tends to mean-revert. After a spike in fear, stocks usually
      stabilize or bounce within a few days. The person who sold the puts keeps the
      premium if SPY/QQQ doesn't crash further.

    Best conditions:
      - VXX ratio > 1.30 (significant panic)
      - SPY still above its 50-day MA (short-term fear, not structural bear)
      - We sell cash-secured puts at a strike 5-8% BELOW current SPY price
        (gives us a cushion — the stock can fall 5-8% and we still profit)

    Risk:
      - If the market keeps crashing below our strike, we take a loss.
      - That's why we only do this when SPY is still above the MA (not a full bear).
    """
    if not _is_regular_hours():
        return None  # Options only trade regular hours (exchange rule)
    if vxx_ratio < 1.30:
        return None  # Not panicky enough
    if spy_vs_ma50 < 0.94:
        return None  # SPY already broken down — too risky to sell puts

    spy_price = _fetch_price("SPY")
    if not spy_price or spy_price < 100:
        return None

    # Fetch SPY options chain — look for puts 5-8% OTM, 14-30 days out
    contracts = _fetch_options_chain("SPY", spy_price, min_days=7, max_days=30)
    if not contracts:
        # Also try QQQ
        qqq_price = _fetch_price("QQQ")
        if qqq_price:
            contracts = _fetch_options_chain("QQQ", qqq_price, min_days=7, max_days=30)
            spy_price = qqq_price or spy_price

    if not contracts:
        return None

    # Find the best OTM put: 5-8% below current price, 14-30 days out
    target_strike_min = spy_price * 0.92
    target_strike_max = spy_price * 0.95
    puts = [c for c in contracts
            if c["opt_type"] == "put"
            and target_strike_min <= c["strike"] <= target_strike_max
            and 14 <= c["days_out"] <= 30
            and c["bid"] >= 0.50  # Minimum $0.50 premium worth selling
            and c["spread_pct"] < 0.06]

    if not puts:
        return None

    # Choose the put with highest premium-to-strike ratio
    best_put = max(puts, key=lambda c: c["mid"] / c["strike"])

    cushion_pct = round((spy_price - best_put["strike"]) / spy_price * 100, 1)
    premium_pct = round(best_put["mid"] / spy_price * 100, 2)

    # Rules score: panic severity bonus + premium quality
    # VXX panic put sale is historically the strongest options setup (~70% WR)
    # Research: Whaley 2000 (VXX mean-reversion), Cremers & Weinbaum 2010 (put selling)
    panic_bonus = min(20, (vxx_ratio - 1.30) * 100)
    rules_score = min(90, 65 + panic_bonus)

    # ML blend: 60% rules + 40% ML
    # The ML was trained directly on historical VXX panic days — it has 750+ examples
    # of days where vxx_ratio >= 1.30 and knows which ones SPY recovered from
    spy_ma = _get_spy_vs_ma50()
    vrp_est = max(-20.0, min(20.0, (vxx_ratio - 1.0) * 50))
    ml_features = _build_setup_features(
        vxx_ratio=vxx_ratio, spy_vs_ma50=spy_ma,
        iv_rank=min(100, (vxx_ratio - 0.8) / 0.6 * 100),  # IV rank proxy from VXX ratio
        vrp=vrp_est, days_to_earn=99.0)
    ml_prob = _options_ml_score(ml_features)
    ml_score_val = ml_prob * 100
    score = round(min(90, rules_score * 0.60 + ml_score_val * 0.40), 1)

    return {
        "ticker":           "SPY",
        "setup":            "vxx_panic_put_sale",
        "score":            score,
        "price":            spy_price,
        "action_label":     "SELL PUT SPREAD (VXX panic — market overpaying for protection)",
        # P0-10 FIX: VXX panic fires when vxx_ratio >= 1.30, i.e. market is
        # genuinely panicking. Selling a NAKED cash-secured put there ties
        # up massive buying power (strike × 100 per contract) and leaves us
        # exposed to unbounded downside on a 1987-style gap. Switching to
        # a defined-risk bull put credit spread keeps the same positive-carry
        # thesis (we collect premium) but caps the tail at spread width × 100.
        "options_strategy": "bull_put_credit_spread",
        "vxx_ratio":        vxx_ratio,
        "put_strike":       best_put["strike"],
        "put_exp":          best_put["exp_date"],
        "put_mid":          best_put["mid"],
        "cushion_pct":      cushion_pct,
        "premium_pct":      premium_pct,
        "occ_symbol":       best_put["occ_symbol"],
        "reasoning": (
            f"VXX at {vxx_ratio:.2f}x 30d avg (PANIC). SPY still above 50d MA ({spy_vs_ma50:.3f}). "
            f"Selling SPY {best_put['strike']} put expiring {best_put['exp_date']} "
            f"for ${best_put['mid']:.2f} premium ({premium_pct}% of stock). "
            f"Cushion: {cushion_pct}% — SPY must fall {cushion_pct}% more before we lose."
        ),
        "source": "options_scanner",
        "side": "sell",
    }


def _setup_high_iv_premium_sale(ticker: str, price: float, vxx_ratio: float) -> Optional[dict]:
    """
    SETUP 3: HIGH-IV PREMIUM SELLING

    What is it:
      Some stocks regularly have high implied volatility — biotechs waiting on FDA
      decisions, earnings plays, high-momentum stocks (TSLA, NVDA). When a stock's
      IV is in the top 30% of its own 52-week range (IV rank > 70), the options
      MARKET is paying too much for that stock's options. Selling those options
      collects that overpricing as premium.

      This is different from the stock scanner — we're not saying "this stock will
      go up." We're saying "this stock's options are expensive and will decay."

    Best conditions:
      - IV rank > 70 (IV is in the top 30% of its own 52-week history)
      - No earnings within 14 days (IV is high for a regular reason, not just earnings)
      - Liquid options (OI > 1000 ATM)
      - NOT in a general panic (we don't want to add risk when everything is falling)
    """
    if not _is_regular_hours():
        return None  # Options only trade regular hours (exchange rule)
    if vxx_ratio >= 1.25:
        return None  # Market-wide panic — individual stock IV is noise

    iv_rank = _fetch_iv_rank(ticker)
    if iv_rank is None:
        return None

    # Fix 2 (v1.0.23): VRatio-based IV gate
    # After a volatility spike (like Apr 2026 tariff crash), the 52-week IV rank
    # drops to ~16% because the crash becomes the new 52w high.
    # BUT: VXX ratio > 1.05 means current IV is STILL elevated vs its own 30d avg.
    # That window — high VXX ratio + declining absolute IV — is actually the
    # BEST time to sell premium: vol is coming off a spike, mean-reversion favors sellers.
    #
    # v1.0.32 FIX: Lowered from IVR 70 to IVR 50. Research:
    #   - Tastytrade: Selling premium at IVR 40+ is consistently profitable
    #   - Bakshi & Kapadia 2003: IV-RV gap (VRP) exists at IVR 50+
    #   - IVR 50 = options priced in top half of annual range = above-average premium
    #   - Old threshold of 70 only triggered ~15% of trading days
    #   - New threshold of 50 triggers ~40% of trading days — much more active
    #
    # Condition: either IV rank > 50 (normal) OR VXX ratio in 1.05-1.25 range (transitional)
    vxx_elevated = 1.05 <= vxx_ratio <= 1.25  # Vol elevated but not panic
    if iv_rank < 50 and not vxx_elevated:
        return None  # Neither condition met — skip
    # When using VRatio gate, lower the profit target slightly (less premium available)
    _vratio_mode = vxx_elevated and iv_rank < 50

    contracts = _fetch_options_chain(ticker, price, min_days=14, max_days=45)
    if not contracts:
        return None

    # Filter to 14+ DTE contracts for the chain search
    dte_contracts = [c for c in contracts if c["days_out"] >= 14]
    if not dte_contracts:
        return None

    # v1.0.34: Always use iron condor (capped risk, wider profit zone).
    # Short straddles disabled — lost money to spreads/gamma in live trading.
    # Iron condor: sell 20-delta call + 20-delta put (short wings)
    # Buy 10-delta call + 10-delta put (long wings = protection)
    # 20-delta = ~1 standard deviation OTM — high probability of expiring worthless
    best_call = _find_by_delta(dte_contracts, "call", target_delta=0.20)
    best_put  = _find_by_delta(dte_contracts, "put",  target_delta=0.20)

    if not best_call or not best_put:
        return None

    # Minimum OI check (liquidity)
    if best_call["oi"] < 300 or best_put["oi"] < 300:
        return None

    # Minimum bid/ask quality
    # FIX (2026-04-10): tightened from 0.15 to 0.10 — straddle scalping lost money
    # every single day in Apr 3-10 due to wide spreads eating all edge.
    if best_call["spread_pct"] > 0.10 or best_put["spread_pct"] > 0.10:
        return None

    avg_iv = (best_call["iv"] + best_put["iv"]) / 2
    straddle_price = best_call["mid"] + best_put["mid"]
    iv_implied_pct = straddle_price / price * 100 if price > 0 else 0

    if avg_iv < 0.30:
        return None  # IV not high enough despite rank (small float distortion)

    # FIX (2026-04-10): Minimum expected profit must exceed 2x spread cost.
    _hi_call_spread = best_call.get("ask", 0) - best_call.get("bid", 0)
    _hi_put_spread = best_put.get("ask", 0) - best_put.get("bid", 0)
    _hi_total_spread = _hi_call_spread + _hi_put_spread
    _hi_expected_profit = straddle_price * 0.30  # Conservative: 30% premium capture
    if _hi_expected_profit < _hi_total_spread * 2:
        return None  # Not enough edge to cover spread

    # Strategy: iron condor ONLY (v1.0.34 fix)
    # Short straddles at IVR 50-74 were the rapid-fire scalps that lost
    # money every day in Apr 3-10 — the premium captured never exceeded
    # the bid-ask spread cost + gamma risk. Iron condors have capped risk
    # and wider profit zone, matching the MULTILEG backtest config (27.8% CAGR).
    strategy = "iron_condor"
    action_label = f"SELL {strategy.replace('_', ' ').upper()} (IV rank {iv_rank:.0f}/100 — premium selling)"

    # Research: Bakshi & Kapadia 2003 — selling options earns a premium when IV > RV.
    # The higher the IV rank, the larger the VRP (IV - RV gap) = the more you're paid.
    # v1.0.32: Rescaled bonus from IVR 50 base (was 70)
    #   IVR 50 → base score 62 + 0 = 62
    #   IVR 65 → 62 + 7.5 = 69.5
    #   IVR 80 → 62 + 15 = 77 (capped)
    iv_bonus = min(15, (iv_rank - 50) * 0.50)
    rules_score = min(88, 62 + iv_bonus)

    # ML blend: 60% rules + 40% ML
    # ML trained on high-IV days from 3 years of history — knows which IV levels
    # actually led to VXX contraction (premium sellers winning) vs continued IV expansion
    spy_ma = _get_spy_vs_ma50()
    vrp_est = max(-20.0, min(20.0, (vxx_ratio - 1.0) * 50))
    ml_features = _build_setup_features(
        vxx_ratio=vxx_ratio, spy_vs_ma50=spy_ma,
        iv_rank=float(iv_rank), vrp=vrp_est, days_to_earn=99.0)
    ml_prob = _options_ml_score(ml_features)
    ml_score_val = ml_prob * 100
    score = round(min(88, rules_score * 0.60 + ml_score_val * 0.40), 1)

    return {
        "ticker":           ticker,
        "setup":            "high_iv_premium_sale",
        "score":            score,
        "price":            price,
        "action_label":     action_label,
        "options_strategy": strategy,
        "iv_rank":          iv_rank,
        "avg_iv":           round(avg_iv * 100, 1),
        "iv_implied_pct":   round(iv_implied_pct, 2),
        "straddle_price":   round(straddle_price, 2),
        "best_call_occ":    best_call["occ_symbol"],
        "best_put_occ":     best_put["occ_symbol"],
        "reasoning": (
            f"IV rank {iv_rank:.0f}/100 — top {100 - iv_rank:.0f}% of 52wk range. "
            f"ATM IV {avg_iv*100:.0f}%. Straddle premium ${straddle_price:.2f} "
            f"({iv_implied_pct:.1f}% of stock). OI: calls {best_call['oi']:,} / puts {best_put['oi']:,}."
        ),
        "source": "options_scanner",
        "side": "sell",
    }


def _setup_low_iv_breakout_buy(ticker: str, price: float, vxx_ratio: float) -> Optional[dict]:
    """
    SETUP 4: LOW-IV BREAKOUT BUY (buy cheap options before a move)

    What is it:
      When a stock's IV is at its 52-week LOW (IV rank < 20), options are CHEAP.
      The market is NOT expecting this stock to move. If it does move — in either
      direction — the options buyer wins both from the direction AND from IV expanding
      back toward normal levels.

      This is the opposite of premium selling. We buy when cheap, sell when expensive.
      A straddle here (buy call + buy put) is a bet that the stock WILL move, regardless
      of direction. Good before catalyst events that the market hasn't priced in yet.

    Best conditions:
      - IV rank < 20 (options at 52-week lows = cheapest they've been all year)
      - Stock in a tight price range (low ATR = hasn't moved recently = coiled)
      - Reasonable premium (straddle costs < 3% of stock price — otherwise even if
        it moves, the entry cost eats all the profit)

    Risk:
      - If the stock stays flat, the options lose value through time decay (theta).
      - That's why we pick short-term options (14-21 days) to limit theta exposure.
    """
    if not _is_regular_hours():
        return None
    if vxx_ratio >= 1.20:
        return None  # In elevated fear, "low IV" stocks may just be quiet for a reason

    iv_rank = _fetch_iv_rank(ticker)
    if iv_rank is None or iv_rank > 20:
        return None  # IV not cheap enough

    # Look for short-dated options (14-21 days) — minimize theta burn
    contracts = _fetch_options_chain(ticker, price, min_days=10, max_days=25)
    if not contracts:
        return None

    # Buy ATM straddle at 50-delta — maximum sensitivity to a big move in either direction
    # When IV is at 52-week lows, ATM options are the cheapest they've been all year.
    # We want 50-delta because that's where the most gamma lives — fastest P&L if it moves.
    best_call = _find_by_delta(contracts, "call", target_delta=0.50, tolerance=0.10)
    best_put  = _find_by_delta(contracts, "put",  target_delta=0.50, tolerance=0.10)

    if not best_call or not best_put:
        return None

    straddle_price  = best_call["mid"] + best_put["mid"]
    straddle_pct    = straddle_price / price * 100 if price > 0 else 99

    # Only worth buying if straddle costs < 5% of stock
    # v1.0.33: raised from 3% to 5% — in calm markets most liquid straddles sit 3-5%.
    # At 5%, a $100 stock costs $5 straddle → need ~5% move to break even.
    # With IVR < 20, implied vol is near 52-week lows, so realized moves tend to exceed IV.
    if straddle_pct >= 5.0:
        return None

    # Bid/ask quality check
    # v1.0.33: widened from 0.12 to 0.15 — consistent across all setups
    if best_call["spread_pct"] > 0.15 or best_put["spread_pct"] > 0.15:
        return None  # Too illiquid — entry cost will eat the edge

    avg_iv   = (best_call["iv"] + best_put["iv"]) / 2
    # The breakeven move needed: straddle cost / stock price
    breakeven_pct = straddle_pct

    # Research: Carr & Wu 2016 — buying options when IV is at 52-week low captures
    # the subsequent mean-reversion of IV back to normal levels (vol of vol effect).
    # IV rank 20 → 0 bonus; IV rank 0 → 15 bonus = score 75
    cheapness_bonus = min(15, (20 - iv_rank) * 0.75)
    rules_score = min(82, 60 + cheapness_bonus)

    # ML blend: 60% rules + 40% ML
    # ML trained on low-IV days — knows which combo of low IV + regime conditions
    # historically produced large enough moves to profit from straddle buying
    spy_ma = _get_spy_vs_ma50()
    vrp_est = max(-20.0, min(20.0, (vxx_ratio - 1.0) * 50))
    ml_features = _build_setup_features(
        vxx_ratio=vxx_ratio, spy_vs_ma50=spy_ma,
        iv_rank=float(iv_rank), vrp=vrp_est, days_to_earn=99.0)
    ml_prob = _options_ml_score(ml_features)
    ml_score_val = ml_prob * 100
    score = round(min(82, rules_score * 0.60 + ml_score_val * 0.40), 1)

    return {
        "ticker":           ticker,
        "setup":            "low_iv_breakout_buy",
        "score":            score,
        "price":            price,
        "action_label":     f"BUY STRADDLE (IV rank {iv_rank:.0f}/100 — options at 52wk low)",
        "options_strategy": "buy_straddle",
        "iv_rank":          iv_rank,
        "avg_iv":           round(avg_iv * 100, 1),
        "straddle_price":   round(straddle_price, 2),
        "straddle_pct":     round(straddle_pct, 2),
        "breakeven_pct":    round(breakeven_pct, 2),
        "best_call_occ":    best_call["occ_symbol"],
        "best_put_occ":     best_put["occ_symbol"],
        "reasoning": (
            f"IV rank {iv_rank:.0f}/100 — options at 52-week low. "
            f"ATM IV {avg_iv*100:.0f}%. Straddle costs ${straddle_price:.2f} "
            f"({straddle_pct:.1f}% of stock). Breakeven: stock moves >{breakeven_pct:.1f}% either way."
        ),
        "source": "options_scanner",
        "side": "buy",
    }


def _setup_gamma_pin(ticker: str, price: float) -> Optional[dict]:
    """
    SETUP 5: GAMMA PIN (expiry day — trade toward the pinned strike)

    What is it:
      On expiration day (Friday, or 0DTE), market makers who sold options must
      constantly buy/sell shares to "delta hedge" — keep their risk neutral.
      When there is a HUGE amount of open interest at a nearby strike, the
      market makers' hedging activity actually pulls the stock TOWARD that strike
      (called "pinning"). The stock gets magnetically attracted to the big OI strike.

      Example: If AAPL has 50,000 call contracts at $175 and it's currently at $177
      on a Friday, AAPL will likely drift toward $175 by end of day.

    Best conditions:
      - Today is Friday (or 0DTE / 1DTE day)
      - One strike has MUCH higher OI than surrounding strikes (3x+)
      - That strike is within 2% of current price
      - We buy a call spread (if pin strike is above current) or put spread (if below)
        targeting the pin by end of day.

    Risk:
      - If the market moves sharply before the pin can form, the trade loses.
      - We use 0DTE or 1DTE options to minimize cost (they're very cheap).
    """
    if not _is_regular_hours():
        return None
    et_now = _et_now()
    # Only run near expiry days (Friday = weekday 4, or any 0DTE check)
    is_friday = et_now.weekday() == 4
    is_afternoon = et_now.hour >= 12  # After noon ET — pin effect strongest 1pm-3pm
    if not (is_friday and is_afternoon):
        return None

    # Fetch 0DTE/1DTE options only
    contracts = _fetch_options_chain(ticker, price, min_days=0, max_days=2)
    if not contracts or len(contracts) < 4:
        return None

    # Aggregate OI by strike (calls + puts together)
    oi_by_strike: dict = {}
    for c in contracts:
        oi_by_strike[c["strike"]] = oi_by_strike.get(c["strike"], 0) + c["oi"]

    if not oi_by_strike:
        return None

    # Find the strike with highest total OI
    max_oi_strike = max(oi_by_strike, key=oi_by_strike.get)
    max_oi        = oi_by_strike[max_oi_strike]

    # Check that this strike is within 2% of current price
    if abs(max_oi_strike - price) / price > 0.02:
        return None

    # Check that this strike dominates (3x higher OI than average of nearby strikes)
    nearby_oi = [oi for s, oi in oi_by_strike.items()
                 if s != max_oi_strike and abs(s - price) / price < 0.05]
    avg_nearby = sum(nearby_oi) / len(nearby_oi) if nearby_oi else 0
    if avg_nearby > 0 and max_oi < avg_nearby * 3:
        return None  # No dominant pin strike

    # Direction: if pin is above → stock pulled up → buy call
    #            if pin is below → stock pulled down → buy put
    direction = "call" if max_oi_strike >= price else "put"
    pin_dist_pct = round((max_oi_strike - price) / price * 100, 2)

    # Find appropriate contract
    target_contracts = [c for c in contracts if c["opt_type"] == direction
                        and c["strike"] == max_oi_strike
                        and c["bid"] >= 0.05]
    if not target_contracts:
        return None
    best = target_contracts[0]

    # Research: Ni, Pearson & Poteshman 2005 (U Chicago) — proved empirically that
    # stocks with heavy OI at nearby strikes are magnetically pulled toward that strike
    # by market maker delta hedging. Effect is strongest in the last 2 hours of trading.
    # OI 1000 → +1pt, OI 10000 → +10pt, OI 15000+ → capped at +15pt
    oi_bonus = min(15, max_oi / 1000)
    rules_score = min(80, 58 + oi_bonus)

    # Gamma pin doesn't rely on IV level — it's purely a market structure phenomenon.
    # ML contribution here is limited (the regime context still matters though —
    # pins work less reliably in high-vol / panic regimes because large moves override them)
    vxx_ratio_now = _get_vxx_ratio()
    spy_ma_now    = _get_spy_vs_ma50()
    ml_features = _build_setup_features(
        vxx_ratio=vxx_ratio_now, spy_vs_ma50=spy_ma_now,
        iv_rank=50.0, vrp=0.0, days_to_earn=99.0)
    ml_prob = _options_ml_score(ml_features)
    ml_score_val = ml_prob * 100
    # Smaller ML weight for gamma pin (40% rules, 20% ML, 40% held at rules base)
    # because the pin itself is the signal — regime is secondary
    score = round(min(80, rules_score * 0.80 + ml_score_val * 0.20), 1)

    return {
        "ticker":           ticker,
        "setup":            "gamma_pin",
        "score":            score,
        "price":            price,
        "action_label":     f"BUY {direction.upper()} (gamma pin toward ${max_oi_strike:.0f} — {max_oi:,} OI)",
        "options_strategy": f"buy_{direction}",
        "pin_strike":       max_oi_strike,
        "pin_oi":           max_oi,
        "pin_dist_pct":     pin_dist_pct,
        "occ_symbol":       best["occ_symbol"],
        "contract_mid":     best["mid"],
        "reasoning": (
            f"Expiry day. Strike ${max_oi_strike:.0f} has {max_oi:,} OI "
            f"({max_oi/avg_nearby:.1f}x avg nearby strikes). "
            f"Pin is {abs(pin_dist_pct):.1f}% {'above' if pin_dist_pct > 0 else 'below'} current price. "
            f"Buying {direction} at ${best['mid']:.2f}."
        ),
        "source": "options_scanner",
        "side": "buy",
    }


def _setup_csp_normal_market(ticker: str, price: float, vxx_ratio: float) -> Optional[dict]:
    """
    SETUP 6: CASH-SECURED PUT IN NORMAL MARKETS (v1.0.33)

    What is it:
      When no other setup triggers — VXX is calm, IV isn't extreme, no earnings —
      we can still sell cash-secured puts on high-quality, liquid stocks.
      This is the "wheel" strategy entry: collect premium from selling OTM puts
      on stocks we'd be happy owning at a lower price.

    Why it works:
      - At 30-delta, ~70% probability of expiring worthless (premium kept)
      - Even when assigned, the effective buy price = strike minus premium received
      - Theta decay is the primary edge — time is on the seller's side
      - Research: CBOE PUT Index (selling ATM puts on SPX) outperformed SPX
        from 1986-2024 with lower volatility (Figelman 2008, CBOE research).

    Best conditions:
      - IVR between 20-50 (moderate — enough premium to be worth it)
      - VXX ratio < 1.15 (calm/neutral market, not panicking)
      - Liquid options (OI > 300)
      - 30-45 DTE (theta decay sweet spot)
      - Stock price > $20 (liquid options chains)

    Why 30-delta (not ATM):
      - 30-delta = ~5-8% OTM — gives a buffer before assignment
      - Higher POP (probability of profit) than ATM
      - Less margin/capital required per contract
      - Sweet spot between premium collected and probability of assignment
    """
    if not _is_regular_hours():
        return None
    if vxx_ratio >= 1.15:
        return None  # Elevated fear — other setups handle this better

    iv_rank = _fetch_iv_rank(ticker)
    if iv_rank is None:
        return None

    # Only fire in the "normal" IVR band: 15-50
    # Below 15 → low_iv_breakout_buy handles it (buy cheap options)
    # Above 50 → high_iv_premium_sale handles it (sell expensive options)
    #
    # IVR lower bound dropped from 20 → 15 (2026-04-17, backtest_scenario_c_wf).
    # Rationale: Alpaca is commission-free on options, so CSPs fire profitably
    # on calmer underlyings. Backtest equivalent (wheel_min_vix 15→12) produced
    # CSP P&L of -$4,416 → +$10,238 in OOS 2023-2026.
    if iv_rank < 15 or iv_rank > 50:
        return None

    # Fetch 30-45 DTE options (theta decay sweet spot for short premium)
    contracts = _fetch_options_chain(ticker, price, min_days=25, max_days=50)
    if not contracts:
        return None

    # Find 30-delta put: ~5-8% OTM, ~70% POP
    best_put = _find_by_delta(contracts, "put", target_delta=0.30, tolerance=0.10)
    if not best_put:
        return None

    # Liquidity checks
    if best_put["oi"] < 300:
        return None  # Not enough open interest
    if best_put["spread_pct"] > 0.15:
        return None  # Spread too wide
    if best_put["mid"] < 0.30:
        return None  # Premium too small to be worth the capital tie-up

    avg_iv = best_put["iv"]
    if avg_iv < 0.15:
        return None  # Not enough IV to generate meaningful premium
    # IV floor dropped from 0.20 → 0.15 (2026-04-17). Matches cc_min_iv change
    # from 0.25 → 0.18 in backtest harness.

    # Calculate key metrics
    premium = best_put["mid"]
    strike = best_put["strike"]
    otm_pct = round((price - strike) / price * 100, 2)  # How far OTM
    premium_yield = round(premium / strike * 100, 2)     # Premium as % of strike
    days_out = best_put["days_out"]
    # Annualized yield: (premium / strike) * (365 / DTE)
    annual_yield = round(premium_yield * (365 / max(days_out, 1)), 1)

    # Score: base 58 + adjustments
    #   IVR bonus: higher IVR within the band = more premium = higher score
    #   Range: IVR 20 → +0, IVR 35 → +3.75, IVR 50 → +7.5
    ivr_bonus = (iv_rank - 20) * 0.25
    #   OTM bonus: further OTM = safer = slight bonus (max 3)
    otm_bonus = min(3.0, otm_pct * 0.4)
    #   Premium yield bonus: higher yield = better trade (max 4)
    yield_bonus = min(4.0, premium_yield * 2.0)
    rules_score = min(75, 58 + ivr_bonus + otm_bonus + yield_bonus)

    # ML blend (lighter weight — CSP is a steady-income trade, less regime-sensitive)
    spy_ma = _get_spy_vs_ma50()
    vrp_est = max(-20.0, min(20.0, (vxx_ratio - 1.0) * 50))
    ml_features = _build_setup_features(
        vxx_ratio=vxx_ratio, spy_vs_ma50=spy_ma,
        iv_rank=float(iv_rank), vrp=vrp_est, days_to_earn=99.0)
    ml_prob = _options_ml_score(ml_features)
    ml_score_val = ml_prob * 100
    score = round(min(75, rules_score * 0.70 + ml_score_val * 0.30), 1)

    return {
        "ticker":           ticker,
        "setup":            "csp_normal_market",
        "score":            score,
        "price":            price,
        "action_label":     f"SELL PUT ${strike:.0f} (CSP — IV rank {iv_rank:.0f}, {otm_pct:.0f}% OTM, {days_out}d)",
        "options_strategy": "sell_cash_secured_put",
        "iv_rank":          iv_rank,
        "avg_iv":           round(avg_iv * 100, 1),
        "premium":          round(premium, 2),
        "strike":           strike,
        "otm_pct":          otm_pct,
        "premium_yield":    premium_yield,
        "annual_yield":     annual_yield,
        "best_put_occ":     best_put["occ_symbol"],
        "reasoning": (
            f"Normal market CSP. IV rank {iv_rank:.0f}/100 (moderate). "
            f"Sell ${strike:.0f} put ({otm_pct:.1f}% OTM) for ${premium:.2f} premium. "
            f"Yield: {premium_yield:.1f}% in {days_out}d ({annual_yield:.0f}% ann). "
            f"OI: {best_put['oi']:,}."
        ),
        "source": "options_scanner",
        "side": "sell",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: CANDIDATE UNIVERSE FOR OPTIONS SCAN
# ══════════════════════════════════════════════════════════════════════════════

# ── Anchor tickers: always in the final OPRA fetch regardless of pre-filter ──
# These are checked every scan even if snapshot data doesn't flag them,
# because they're the most reliably options-liquid names in the market.
_OPTIONS_ANCHOR = [
    "SPY", "QQQ", "IWM", "VXX", "UVXY",           # Index ETFs (highest OI of any options)
    "AAPL", "MSFT", "NVDA", "META", "GOOGL",        # Mega-cap tech (always top-5 options vol)
    "AMZN", "TSLA", "AMD", "COIN", "MSTR",
    "XLE", "XLF", "XLK", "XBI", "GLD",             # Sector ETFs
]

# Cache for the pre-filtered candidate list (refreshed every scan cycle)
_options_candidates_cache: list = []
_options_candidates_time: float = 0.0
_CANDIDATES_CACHE_TTL = 60  # 1 minute — candidates change during the day


def _get_options_candidates(snap_data: dict = None) -> list:
    """
    Two-stage filter to get all options-eligible stocks from the full 11,635 universe.

    WHY TWO STAGES:
      Fetching an OPRA options chain costs ~0.2s per ticker.
      Doing that for all 11,635 stocks = 38 minutes. Not acceptable.
      But snapshot data (price, volume, daily change) is fast — all 11,635 in ~4s.
      We use snapshot data to pre-filter down to ~600-800 REAL candidates,
      then only fetch chains for those. Total time: ~10-15 seconds.

    STAGE 1 — Snapshot pre-filter (runs on all 11,635, already done by stock scanner):
      PASS criteria (stock is options-eligible):
        - Price >= $10  (stocks under $10 have illiquid/no options chains)
        - Volume >= 1M  (minimum daily volume for options market maker quoting)
      Then classify into setup types:
        - High-IV candidate:  abs(daily change) > 2%   → IV likely elevated → sell premium
        - Low-IV candidate:   abs(daily change) < 0.3% → stock quiet → IV may be cheap
        - Always include:     anchor tickers regardless of today's move

    STAGE 2 — Return the candidate list (caller fetches OPRA chains):
      Only these ~600-800 tickers get their options chain fetched.
      That's the same number as before (was fixed at 33), just actually correct.

    Args:
        snap_data: Optional pre-fetched snapshot dict {symbol: snap}. If provided,
                   skips re-fetching (reuses stock scanner's data). If None, fetches fresh.

    Returns:
        List of (ticker, price, setup_type) tuples where setup_type is one of:
        'high_iv', 'low_iv', 'anchor', 'any'
    """
    global _options_candidates_cache, _options_candidates_time
    now = time.time()
    if _options_candidates_cache and (now - _options_candidates_time) < _CANDIDATES_CACHE_TTL:
        return _options_candidates_cache

    # Get the full universe (same daily-cached list the stock scanner uses)
    full_universe: list = []
    try:
        import sys, os
        _here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, _here)
        from bot_engine import _get_full_universe
        full_universe = _get_full_universe()
    except Exception:
        pass

    if not full_universe:
        # Fallback: fetch from Alpaca assets endpoint directly
        try:
            alpaca_throttle.acquire()
            r = requests.get(
                f"{os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')}/v2/assets?status=active&asset_class=us_equity",
                headers=_headers(), timeout=20)
            assets = r.json()
            full_universe = [
                a["symbol"] for a in assets
                if a.get("tradable") and "." not in a.get("symbol", "")
                and not a.get("symbol", "").endswith("W")
                and len(a.get("symbol", "")) <= 5
            ]
        except Exception:
            full_universe = _OPTIONS_ANCHOR  # Last resort

    # ── STAGE 1: Snapshot fetch (all 11,635 in ~4s with 16 workers) ──────────
    # If snap_data already provided (from bot_engine), skip the fetch entirely.
    if not snap_data:
        snap_data = {}
        batches = [full_universe[i:i+50] for i in range(0, len(full_universe), 50)]

        def _fetch_snap(batch):
            try:
                alpaca_throttle.acquire()
                r = requests.get(
                    f"{ALPACA_DATA}/v2/stocks/snapshots",
                    params={"symbols": ",".join(batch), "feed": "sip"},
                    headers=_headers(), timeout=12)
                return r.json()
            except Exception:
                return {}

        with ThreadPoolExecutor(max_workers=4) as pool:  # MEM FIX: was 16 — Railway 512MB OOM
            for result in pool.map(_fetch_snap, batches):
                snap_data.update(result)

    # ── STAGE 2: Pre-filter — classify each stock ─────────────────────────────
    high_iv_candidates: list = []  # abs(change) > 2% → IV likely elevated
    low_iv_candidates:  list = []  # abs(change) < 0.3% AND vol > 1M → IV may be cheap

    for sym, snap in snap_data.items():
        if "." in sym or len(sym) > 5:
            continue
        bar  = snap.get("dailyBar", {})
        prev = snap.get("prevDailyBar", {})
        c    = float(bar.get("c", 0) or 0)
        pc   = float(prev.get("c", c) or c)
        v    = int(bar.get("v", 0) or 0)

        # Hard floor: options don't exist or are illiquid below these levels
        if c < 10.0 or v < 1_000_000:
            continue

        chg = abs((c - pc) / pc * 100) if pc > 0 else 0

        if chg > 2.0:
            # Stock moved significantly today → IV is elevated → sell premium candidate
            high_iv_candidates.append((sym, c, "high_iv"))
        elif chg < 0.3:
            # Stock is very quiet today → IV may be at lows → buy cheap options candidate
            low_iv_candidates.append((sym, c, "low_iv"))

    # Sort by movement magnitude (biggest movers first = highest IV)
    high_iv_candidates.sort(key=lambda x: x[0], reverse=False)

    # Cap each tier to keep total OPRA calls manageable
    # Even with 16 workers, 700+ chain fetches takes ~10s — acceptable.
    # 1000+ starts adding latency. Cap at 400 per tier.
    MAX_PER_TIER = 400
    high_iv_top = high_iv_candidates[:MAX_PER_TIER]
    low_iv_top  = low_iv_candidates[:MAX_PER_TIER]

    # ── Combine: anchor + high_iv + low_iv (deduplicated) ────────────────────
    seen: set = set()
    result: list = []

    # Anchors always first
    for sym in _OPTIONS_ANCHOR:
        if sym not in seen:
            seen.add(sym)
            # Get price from snap_data if available
            p = 0.0
            snap = snap_data.get(sym, {})
            if snap:
                bar = snap.get("dailyBar", {})
                p   = float(bar.get("c", 0) or 0)
            result.append((sym, p, "anchor"))

    for sym, price, setup_type in high_iv_top + low_iv_top:
        if sym not in seen:
            seen.add(sym)
            result.append((sym, price, setup_type))

    _options_candidates_cache = result
    _options_candidates_time  = now
    logger.info(
        f"Options candidates: {len(result)} total "
        f"({len(high_iv_top)} high-IV, {len(low_iv_top)} low-IV, {len(_OPTIONS_ANCHOR)} anchors) "
        f"from {len(full_universe):,} universe"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: MAIN SCAN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def scan_options() -> dict:
    """
    Main entry point. Runs all 6 options setup detectors in parallel.

    Returns:
        {
            "opportunities": [list of options trade dicts, sorted by score],
            "setup_counts":  {setup_name: count},
            "vxx_ratio":     float,
            "regime":        str,
            "scanned":       int,
            "duration_secs": float,
        }

    Each opportunity dict has the same structure as bot_engine trade dicts so
    they can be merged and compete for the same position slots.
    """
    t0 = time.time()
    opportunities = []

    if not _is_regular_hours():
        return {
            "opportunities": [],
            "setup_counts": {},
            "vxx_ratio": 1.0,
            "regime": "CLOSED",
            "scanned": 0,
            "duration_secs": 0,
            "message": "Options scanner only runs during regular market hours (9:30am-4pm ET)"
        }

    # ── Get macro context (shared across all setup checks) ────────────────
    vxx_ratio   = _get_vxx_ratio()
    spy_vs_ma50 = _get_spy_vs_ma50()

    if vxx_ratio >= 1.30 and spy_vs_ma50 < 0.94:
        regime = "PANIC"
    elif vxx_ratio >= 1.15:
        regime = "BEAR"
    elif vxx_ratio >= 1.05:
        regime = "CAUTION"
    elif vxx_ratio <= 0.90:
        regime = "BULL"
    else:
        regime = "NEUTRAL"

    # ── Vol Surface: Market-wide VRP signal (v1.0.33) ─────────────────────
    # Use SPY surface to determine if options are overpriced (sell) or cheap (buy)
    market_vrp = 0.0
    market_surface_score = 50
    try:
        from vol_surface import get_vrp, get_surface_score
        spy_vrp = get_vrp("SPY")
        market_vrp = spy_vrp.get("vrp_20d", 0)
        spy_surface = get_surface_score("SPY")
        market_surface_score = spy_surface.get("surface_score", 50)
    except Exception:
        pass  # Surface is advisory, don't block scanning

    # ── SETUP 2: VXX Panic Put Sale ───────────────────────────────────────
    # Run this first (market-level, doesn't need per-ticker chain fetch)
    panic_setup = _setup_vxx_panic_put_sale(vxx_ratio, spy_vs_ma50)
    if panic_setup:
        opportunities.append(panic_setup)

    # ── SETUP 1: Earnings IV Crush ────────────────────────────────────────
    # Fetch earnings calendar once, then check each earnings ticker
    # Fix 3 (v1.0.23): 21-day window catches full earnings cycle (was 8).
    # Finnhub returns all majors (JPM, GS, NFLX, etc.) when window is wide enough.
    # Self-updating — no hardcoded dates needed.
    earnings_cal = _fetch_earnings_calendar(days_ahead=21)
    earnings_tickers = [t for t in earnings_cal.keys()
                        if t and "." not in t and len(t) <= 5]

    def _check_earnings(tkr: str) -> Optional[dict]:
        days = earnings_cal.get(tkr, 99)
        # Fix 1 (v1.0.23): days >= 0 not >= 1.
        # days=0 means earnings tomorrow morning (report date is calendar
        # day ahead but datetime.now() is same calendar day).
        # e.g. DAL reports Apr 8 bmo; at 11am Apr 7 days=0 but setup is valid.
        if not (0 <= days <= 7):
            return None
        price = _fetch_price(tkr)
        # Fix: raise minimum price to $10 — earnings crush needs liquid options
        if not price or price < 10:
            return None
        return _setup_earnings_iv_crush(tkr, price, days, vxx_ratio)

    with ThreadPoolExecutor(max_workers=3) as pool:  # MEM FIX: was 8
        for result in pool.map(_check_earnings, earnings_tickers[:30]):
            if result:
                opportunities.append(result)

    # ── SETUPS 3, 4, 5: Per-ticker scans ─────────────────────────────────
    # Two-stage approach — same as the stock scanner:
    #   Stage 1: Snapshot all 11,635 stocks (~4s, 16 workers) → pre-filter to
    #            ~600-800 options candidates (price>$10, vol>1M, moved today OR quiet)
    #   Stage 2: Fetch OPRA chain only for those candidates (~10s, 16 workers)
    #
    # This covers the FULL market — not just 33 or 50 or 100 hardcoded tickers.
    # Any stock with IV rank 92 and earnings in 3 days will be found regardless
    # of whether it was ever on a fixed list.
    candidates = _get_options_candidates()  # [(ticker, price, setup_type), ...]

    def _check_ticker(candidate: tuple) -> List[dict]:
        tkr, price, setup_hint = candidate
        if not price or price < 10:
            return []
        found = []
        # Setup 3: High-IV premium sale
        if setup_hint in ("high_iv", "anchor", "any"):
            r3 = _setup_high_iv_premium_sale(tkr, price, vxx_ratio)
            if r3:
                found.append(r3)
        # Setup 4: Low-IV breakout buy
        if setup_hint in ("low_iv", "anchor", "any"):
            r4 = _setup_low_iv_breakout_buy(tkr, price, vxx_ratio)
            if r4:
                found.append(r4)
        # Setup 5: Gamma pin (only on Fridays after noon — checked inside)
        r5 = _setup_gamma_pin(tkr, price)
        if r5:
            found.append(r5)
        # Setup 7 (ALPHA-TUNE 2026-04-21): VRP-driven iron condor on high-IV
        # names without earnings. Uses vol_surface VRP/skew signals to identify
        # names where options are chronically overpriced. Defined-risk condor
        # (not short strangle) so single-name gaps don't destroy the book.
        if setup_hint in ("high_iv", "anchor"):
            try:
                from vol_surface import get_surface_score
                srf = get_surface_score(tkr)
                vrp_val = srf.get("vrp", {}).get("vrp_20d", 0)
                surf_score = srf.get("surface_score", 0)
                days_to_earn = earnings_cal.get(tkr, 99)
                if vrp_val > 0.05 and surf_score > 65 and days_to_earn > 10:
                    r7 = _setup_high_iv_premium_sale(tkr, price, vxx_ratio)
                    if r7 and not any(o.get("ticker") == tkr for o in found):
                        r7["setup"] = "vrp_condor"
                        r7["reasoning"] = f"VRP condor: vrp_20d={vrp_val*100:.1f}%, surface={surf_score:.0f}. " + r7.get("reasoning", "")
                        found.append(r7)
            except Exception:
                pass  # Surface is advisory — don't block scanning
        # Setup 6: Cash-secured put in normal markets — DISABLED (v1.0.34)
        # Backtest: CSP filler strategy at IVR 20-50 had negative P&L.
        # Premium too small to overcome spread cost + assignment risk.
        # Kept commented for reference:
        # if setup_hint in ("anchor", "any", "high_iv", "low_iv"):
        #     r6 = _setup_csp_normal_market(tkr, price, vxx_ratio)
        #     if r6:
        #         found.append(r6)
        return found

    # 16 workers — each candidate needs 1-2 OPRA chain fetches (~0.2s each)
    # 700 candidates / 16 workers = ~9 seconds total
    with ThreadPoolExecutor(max_workers=4) as pool:  # MEM FIX: was 16 — Railway 512MB OOM
        for results in pool.map(_check_ticker, candidates):
            opportunities.extend(results)

    # ── Sort by score, deduplicate by ticker+setup ────────────────────────
    seen = set()
    unique_opps = []
    for opp in sorted(opportunities, key=lambda x: x["score"], reverse=True):
        key = f"{opp['ticker']}_{opp['setup']}"
        if key not in seen:
            seen.add(key)
            unique_opps.append(opp)

    # Count by setup type
    setup_counts: dict = {}
    for opp in unique_opps:
        s = opp["setup"]
        setup_counts[s] = setup_counts.get(s, 0) + 1

    duration = round(time.time() - t0, 2)

    return {
        "opportunities":  unique_opps,
        "setup_counts":   setup_counts,
        "vxx_ratio":      vxx_ratio,
        "spy_vs_ma50":    spy_vs_ma50,
        "regime":         regime,
        "scanned":        len(candidates) + len(earnings_tickers[:30]),
        "duration_secs":  duration,
        "best_score":     unique_opps[0]["score"] if unique_opps else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: MERGE INTO BOT_ENGINE FORMAT
# ══════════════════════════════════════════════════════════════════════════════

# ── HIGH-EDGE SETUPS ONLY (v1.0.34 fix) ─────────────────────────────────
# Backtest: only these setups had positive P&L at 8% allocation.
# Earnings IV crush, VXX panic puts, and iron condors (IVR 75+) are the
# proven edge. Everything else (straddle scalps, CSP fillers, low-IV buys,
# gamma pins) lost money consistently in Apr 3-13 live trading and in the
# MULTILEG backtest config.
HIGH_EDGE_SETUPS = {
    "earnings_iv_crush",     # 65-68% WR, well-studied (Tastytrade research)
    "vxx_panic_put_sale",    # ~70% WR, strongest statistical edge
    "high_iv_premium_sale",  # Only iron_condor variant (IVR 75+), capped risk
}
# Minimum score for options trades — higher bar than stocks (65)
# because options have built-in time decay + spread cost headwinds.
MIN_OPTIONS_SCORE = 70.0


def get_options_trades(
    equity: float,
    current_tickers: list,
    max_new: int = 2,
    min_score: float = 70.0,
) -> list:
    """
    Run the options scanner and return ready-to-use trade recommendations
    in the same format as bot_engine.py scan_market() output.

    v1.0.34: Restricted to HIGH_EDGE_SETUPS only. CSP fillers, straddle
    scalps, low-IV buys, and gamma pins are disabled — they were net
    negative in both backtesting and live trading.

    Args:
        equity:          Total account equity (for position sizing)
        current_tickers: Currently held tickers (skip if already in portfolio)
        max_new:         Max number of new options trades to return per scan
        min_score:       Minimum score threshold (default 70 — higher bar for options)

    Returns:
        List of trade dicts ready for options_execution.execute_options_trade()
    """
    # ── Market Calendar Check (v1.0.34) ──────────────────────────────
    # Skip opening new options on half-days and before long weekends.
    # Half-days have reduced liquidity and wider spreads. Pre-long-weekend
    # means unmonitored gamma risk over 3+ consecutive closed days.
    try:
        from market_calendar import should_skip_new_options, is_short_week, trading_days_this_week
        skip, skip_reason = should_skip_new_options()
        if skip:
            return []  # No new options trades today

        # Short week awareness: raise minimum score by 5 points
        # Fewer trading days = less time for theta to work, more gap risk
        _short_week = is_short_week()
        _trading_days = trading_days_this_week()
    except ImportError:
        _short_week = False
        _trading_days = 5

    scan_result = scan_options()
    opps = scan_result.get("opportunities", [])

    # Enforce minimum score floor
    min_score = max(min_score, MIN_OPTIONS_SCORE)
    # Short-week penalty: raise the bar when fewer trading days available
    if _short_week:
        min_score = max(min_score, MIN_OPTIONS_SCORE + 5)

    trades = []
    for opp in opps:
        if len(trades) >= max_new:
            break

        # ── HIGH-EDGE GATE: Only allow proven setups ─────────────────
        setup = opp.get("setup", "")
        if setup not in HIGH_EDGE_SETUPS:
            continue

        # For high_iv_premium_sale, only allow iron_condor (capped risk).
        # Short straddles at IVR 50-74 were the rapid-fire scalps that
        # lost money every day in Apr 3-10.
        if setup == "high_iv_premium_sale":
            strategy = opp.get("options_strategy", "")
            if strategy not in ("iron_condor", "sell_iron_condor"):
                continue

        # ── Vol Surface Score Adjustment (v1.0.33) ─────────────────────
        # Boost or penalize based on market-wide VRP and surface
        try:
            from vol_surface import get_surface_score
            srf = get_surface_score(opp["ticker"])
            vrp_val = srf.get("vrp", {}).get("vrp_20d", 0)
            side = opp.get("side", "buy")
            # If selling premium and VRP is positive (options overpriced), boost score
            if side == "sell" and vrp_val > 0.02:
                opp["score"] = min(opp["score"] + 5, 95)
                opp["reasoning"] = opp.get("reasoning", "") + f" [Surface: VRP +{vrp_val*100:.1f}%, premium selling favored]"
            # If selling premium but VRP is negative (options cheap), penalize
            elif side == "sell" and vrp_val < -0.02:
                opp["score"] = max(opp["score"] - 8, 40)
                opp["reasoning"] = opp.get("reasoning", "") + f" [Surface: VRP {vrp_val*100:.1f}%, options cheap — penalized]"
            # If buying premium and VRP is negative (options cheap), boost
            elif side == "buy" and vrp_val < -0.02:
                opp["score"] = min(opp["score"] + 5, 95)
                opp["reasoning"] = opp.get("reasoning", "") + f" [Surface: VRP {vrp_val*100:.1f}%, cheap options — boosted]"
            # If buying premium but VRP is positive (options expensive), penalize
            elif side == "buy" and vrp_val > 0.02:
                opp["score"] = max(opp["score"] - 5, 40)
                opp["reasoning"] = opp.get("reasoning", "") + f" [Surface: VRP +{vrp_val*100:.1f}%, options expensive for buyers]"
        except Exception:
            pass  # Surface is advisory

        if opp["score"] < min_score:
            continue
        if opp["ticker"] in current_tickers:
            continue

        # Size: 4-6% of equity per options trade (capped by setup type)
        # Kelly criterion sizing: f* = (p*b - q) / b
        # For 57% WR, 1.5:1 R:R: f* = (0.57*1.5 - 0.43) / 1.5 = 0.282 (full Kelly)
        # Half-Kelly = 14% (too aggressive). Quarter-Kelly = 7-8% (practical).
        # Research: Thorp 2006 — quarter to half Kelly maximizes long-run growth.
        # Backtest result: 8% sizing +2.9% vs baseline, options P&L +63%.
        # v1.0.34: Only high-edge setups reach here (others filtered above).
        # All capped at 8% per the MAX_OPTIONS_PCT_CEILING.
        base_size = 0.06  # Default: conservative 6%
        if opp["setup"] == "vxx_panic_put_sale":
            base_size = 0.08  # 8% — strongest edge (~70% WR), well-defined max loss
        elif opp["setup"] == "earnings_iv_crush":
            base_size = 0.07  # 7% — 65-68% WR, well-studied setup
        elif opp["setup"] == "high_iv_premium_sale":
            base_size = 0.06  # 6% — iron condor, capped risk
        position_dollars = round(equity * base_size, 2)

        trades.append({
            "ticker":             opp["ticker"],
            "action":             "BUY",  # All options trades go through execute_options_trade
            "action_label":       opp["action_label"],
            "score":              opp["score"],
            "deep_score":         opp["score"],
            "price":              opp["price"],
            "position_dollars":   position_dollars,
            "position_pct":       base_size,
            "options_strategy":   opp["options_strategy"],
            "setup":              opp["setup"],
            "reasoning":          opp.get("reasoning", ""),
            "source":             "options_scanner",
            "side":               opp.get("side", "buy"),
            # Pass through all setup-specific keys for execution
            **{k: v for k, v in opp.items()
               if k not in ("ticker", "action_label", "score", "price",
                            "options_strategy", "setup", "reasoning", "source", "side")},
        })

    return trades


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    result = scan_options()
    print(json.dumps({
        "regime":        result["regime"],
        "vxx_ratio":     result["vxx_ratio"],
        "opportunities": len(result["opportunities"]),
        "setup_counts":  result["setup_counts"],
        "duration_secs": result["duration_secs"],
        "top_3": [
            {"ticker": o["ticker"], "setup": o["setup"],
             "score": o["score"], "reasoning": o["reasoning"]}
            for o in result["opportunities"][:3]
        ]
    }, indent=2))
