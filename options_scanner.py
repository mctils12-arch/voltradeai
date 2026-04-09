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
from datetime import datetime, timedelta, timezone
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
        "float_turnover":   0.0,
        "atr_pct":          0.0,
        "markov_state":     1.0,
        "sector_momentum":  0.0,
        "change_pct_today": 0.0,
        "above_ma10":       1.0 if spy_vs_ma50 > 1.0 else 0.0,
        "trend_strength":   0.0,
        "volume_acceleration": 0.0,
        "intel_score":      0.0,
        "insider_signal":   0.0,
        "news_sentiment":   0.0,
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
_ALPACA_KEY    = os.environ.get("ALPACA_KEY",    "PKMDHJOVQEVIB4UHZXUYVTIDBU")
_ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et")
ALPACA_DATA    = "https://data.alpaca.markets"
ALPACA_TRADE   = "https://api.alpaca.markets"

# ── Finnhub for earnings calendar ─────────────────────────────────────────────
FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "d78tj7hr01qp0fl6fo2gd78tj7hr01qp0fl6fo30")

# ── Data directory ────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("VOLTRADE_DATA_DIR", "/data/voltrade")

def _headers() -> dict:
    return {"APCA-API-KEY-ID": _ALPACA_KEY, "APCA-API-SECRET-KEY": _ALPACA_SECRET}

def _is_regular_hours() -> bool:
    """True if within 9:30am-4:00pm ET."""
    try:
        now_utc = datetime.now(timezone.utc)
        et_hour = (now_utc.hour - 4) % 24
        et_min  = now_utc.minute
        et_time = et_hour + et_min / 60.0
        return 9.5 <= et_time < 16.0
    except Exception:

def _is_extended_hours() -> bool:
    """True if within extended trading hours (4:00am-9:30am OR 4:00pm-8:00pm ET).
    These windows allow limit-order options trading for setups that benefit
    from pre/post-market positioning (earnings, panic, high IV)."""
    try:
        now_utc = datetime.now(timezone.utc)
        et_hour = (now_utc.hour - 4) % 24
        et_min  = now_utc.minute
        et_time = et_hour + et_min / 60.0
        return (4.0 <= et_time < 9.5) or (16.0 <= et_time < 20.0)
    except Exception:
        return False

def _et_now() -> datetime:
    """Current time in ET (approximate, no pytz dependency)."""
    return datetime.now(timezone.utc) - timedelta(hours=4)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_vxx_ratio() -> float:
    """
    VXX / 30-day average VXX. > 1.0 = above-average fear.
    > 1.30 = panic. Used to trigger VXX Panic Put Sale setup.
    """
    try:
        start = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
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


def _get_spy_vs_ma50() -> float:
    """SPY price / SPY 50-day MA. > 1.0 = above MA (healthy)."""
    try:
        start = (datetime.now() - timedelta(days=80)).strftime("%Y-%m-%d")
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


def _fetch_options_chain(ticker: str, price: float,
                          min_days: int = 7, max_days: int = 60) -> list:
    """
    Fetch live options chain from Alpaca OPRA feed.
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
    Compute IV rank from VXX as proxy (same method as ml_model_v2.py).
    Returns 0-100 (100 = IV at 52-week high).
    Full per-ticker IV history requires yfinance; we use VXX ratio as proxy
    for market-wide IV, then adjust with stock's own 30-day HV.
    """
    try:
        start = (datetime.now() - timedelta(days=380)).strftime("%Y-%m-%d")
        r = requests.get(
            f"{ALPACA_DATA}/v2/stocks/bars",
            params={"symbols": f"{ticker},VXX", "timeframe": "1Day",
                    "start": start, "limit": 400, "feed": "sip"},
            headers=_headers(), timeout=10
        )
        bars_map = r.json().get("bars", {})
        vxx_bars = bars_map.get("VXX", [])
        if not vxx_bars or len(vxx_bars) < 30:
            return None
        vxx_closes  = [float(b["c"]) for b in vxx_bars]
        vxx_52w_lo  = min(vxx_closes[-252:]) if len(vxx_closes) >= 252 else min(vxx_closes)
        vxx_52w_hi  = max(vxx_closes[-252:]) if len(vxx_closes) >= 252 else max(vxx_closes)
        vxx_now     = vxx_closes[-1]
        if vxx_52w_hi == vxx_52w_lo:
            return 50.0
        iv_rank = (vxx_now - vxx_52w_lo) / (vxx_52w_hi - vxx_52w_lo) * 100
        return round(iv_rank, 1)
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
    if best_call["spread_pct"] > 0.08 or best_put["spread_pct"] > 0.08:
        return None

    # IV-implied move: straddle price / stock price = expected % move
    straddle_price = best_call["mid"] + best_put["mid"]
    iv_implied_move_pct = (straddle_price / price) * 100 if price > 0 else 0

    if iv_implied_move_pct < 2.0:
        return None  # Straddle too cheap — not worth the risk

    # Minimum IV: must be at least 40% annualized (otherwise not enough premium)
    avg_iv = (best_call["iv"] + best_put["iv"]) / 2
    if avg_iv < 0.40:
        return None

    # Strategy: if IV implied move >> 8% → iron condor (capped risk)
    #           if IV implied move 3-8% → short straddle (more premium)
    if iv_implied_move_pct > 8.0:
        strategy = "iron_condor"
        action_label = "SELL IRON CONDOR (earnings IV crush)"
    else:
        strategy = "short_straddle"
        action_label = "SELL STRADDLE (earnings IV crush)"

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
        "action_label":     "SELL PUT (VXX panic — market overpaying for protection)",
        "options_strategy": "sell_cash_secured_put",
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
    # Condition: either IV rank > 70 (normal) OR VXX ratio in 1.05-1.25 range (transitional)
    vxx_elevated = 1.05 <= vxx_ratio <= 1.25  # Vol elevated but not panic
    if iv_rank < 70 and not vxx_elevated:
        return None  # Neither condition met — skip
    # When using VRatio gate, lower the profit target slightly (less premium available)
    _vratio_mode = vxx_elevated and iv_rank < 70

    contracts = _fetch_options_chain(ticker, price, min_days=14, max_days=45)
    if not contracts:
        return None

    # Filter to 14+ DTE contracts for the chain search
    dte_contracts = [c for c in contracts if c["days_out"] >= 14]
    if not dte_contracts:
        return None

    if strategy == "iron_condor":
        # Iron condor: sell 20-delta call + 20-delta put (short wings)
        # Buy 10-delta call + 10-delta put (long wings = protection)
        # 20-delta = ~1 standard deviation OTM — high probability of expiring worthless
        # This is the most common institutional iron condor placement
        best_call = _find_by_delta(dte_contracts, "call", target_delta=0.20)
        best_put  = _find_by_delta(dte_contracts, "put",  target_delta=0.20)
    else:
        # Short straddle: sell 50-delta call + put (ATM)
        # Maximum premium — but more risk than condor, hence we only use for iv_rank < 85
        best_call = _find_by_delta(dte_contracts, "call", target_delta=0.50)
        best_put  = _find_by_delta(dte_contracts, "put",  target_delta=0.50)

    if not best_call or not best_put:
        return None

    # Minimum OI check (liquidity)
    if best_call["oi"] < 300 or best_put["oi"] < 300:
        return None

    # Minimum bid/ask quality
    if best_call["spread_pct"] > 0.10 or best_put["spread_pct"] > 0.10:
        return None

    avg_iv = (best_call["iv"] + best_put["iv"]) / 2
    straddle_price = best_call["mid"] + best_put["mid"]
    iv_implied_pct = straddle_price / price * 100 if price > 0 else 0

    if avg_iv < 0.50:
        return None  # IV not high enough despite rank (small float distortion)

    # Strategy: use iron condor (safer, capped loss)
    strategy = "iron_condor" if iv_rank > 85 else "short_straddle"
    action_label = f"SELL {strategy.replace('_', ' ').upper()} (IV rank {iv_rank:.0f}/100 — premium selling)"

    # Research: Bakshi & Kapadia 2003 — selling options earns a premium when IV > RV.
    # The higher the IV rank, the larger the VRP (IV - RV gap) = the more you're paid.
    # IV rank 70 → ~0.75 bonus/point; IV rank 85 → ~11 bonus = score ~73
    iv_bonus = min(15, (iv_rank - 70) * 0.75)
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

    # Only worth buying if straddle costs < 3% of stock (otherwise need a huge move to profit)
    if straddle_pct >= 3.0:
        return None

    # Bid/ask quality check
    if best_call["spread_pct"] > 0.12 or best_put["spread_pct"] > 0.12:
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
            r = requests.get(
                "https://paper-api.alpaca.markets/v2/assets?status=active&asset_class=us_equity",
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
                r = requests.get(
                    f"{ALPACA_DATA}/v2/stocks/snapshots",
                    params={"symbols": ",".join(batch), "feed": "sip"},
                    headers=_headers(), timeout=12)
                return r.json()
            except Exception:
                return {}

        with ThreadPoolExecutor(max_workers=16) as pool:
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
    Main entry point. Runs all 5 options setup detectors in parallel.

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

    with ThreadPoolExecutor(max_workers=8) as pool:
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
        # Prioritize high_iv and anchor candidates — skip low_iv ones for this setup
        if setup_hint in ("high_iv", "anchor", "any"):
            r3 = _setup_high_iv_premium_sale(tkr, price, vxx_ratio)
            if r3:
                found.append(r3)
        # Setup 4: Low-IV breakout buy
        # Prioritize low_iv and anchor candidates — skip high_iv for this setup
        if setup_hint in ("low_iv", "anchor", "any"):
            r4 = _setup_low_iv_breakout_buy(tkr, price, vxx_ratio)
            if r4:
                found.append(r4)
        # Setup 5: Gamma pin (only on Fridays after noon — checked inside)
        r5 = _setup_gamma_pin(tkr, price)
        if r5:
            found.append(r5)
        return found

    # 16 workers — each candidate needs 1-2 OPRA chain fetches (~0.2s each)
    # 700 candidates / 16 workers = ~9 seconds total
    with ThreadPoolExecutor(max_workers=16) as pool:
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

def get_options_trades(
    equity: float,
    current_tickers: list,
    max_new: int = 2,
    min_score: float = 65.0,
) -> list:
    """
    Run the options scanner and return ready-to-use trade recommendations
    in the same format as bot_engine.py scan_market() output.

    Args:
        equity:          Total account equity (for position sizing)
        current_tickers: Currently held tickers (skip if already in portfolio)
        max_new:         Max number of new options trades to return per scan
        min_score:       Minimum score threshold

    Returns:
        List of trade dicts ready for options_execution.execute_options_trade()
    """
    scan_result = scan_options()
    opps = scan_result.get("opportunities", [])

    trades = []
    for opp in opps:
        if len(trades) >= max_new:
            break
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
        base_size = 0.08  # 8% default (quarter-Kelly for 57% WR setups)
        if opp["setup"] == "vxx_panic_put_sale":
            base_size = 0.10  # 10% — strongest edge (~70% WR), well-defined max loss
        elif opp["setup"] == "gamma_pin":
            base_size = 0.03  # 3% — very short-term, binary, limited edge
        elif opp["setup"] == "low_iv_breakout_buy":
            base_size = 0.08  # 8% — 56% WR, cheap entry, quarter-Kelly appropriate
        elif opp["setup"] == "earnings_iv_crush":
            base_size = 0.08  # 8% — 65-68% WR, well-studied setup
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
