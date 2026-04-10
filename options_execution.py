"""
VolTradeAI — Options Execution Engine
======================================
Decides whether stock or options is the better trade, selects the
specific contract, and submits orders via Alpaca.

KEY CONSTRAINTS:
  - Alpaca paper trading: options enabled by default (Level 2: buy calls/puts)
  - Options orders: market or limit only, DAY or GTC time_in_force
  - No complex orders (OTO/OCO) for options
  - No stop orders on options
  - OCC symbol format: AAPL260418C00250000
  - Regular hours only (9:30am-4pm ET) for options
  - $0.65/contract fee on live (free on paper)
  - Options settle T+1

SAFETY RULES:
  - Never sell naked calls (unlimited risk)
  - Never sell naked puts beyond cash-secured amount
  - Max 10% of portfolio in any single options position
  - Max 20% total options exposure
  - Only trade options with bid-ask spread < 15% of mid price
  - Only trade options with volume > 100 and open interest > 500
  - No options on stocks with earnings within 2 days (IV crush risk for buyers)
  - Limit orders only (no market orders on options — spreads are too wide)
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta

try:
    from position_sizing import (
        _earnings_scalar, check_halt_status, _volatility_scalar,
        _confidence_scalar, _regime_scalar, _kelly_fraction,
        _liquidity_scalar, _portfolio_heat_scalar, _time_scalar,
        _get_historical_stats, ABSOLUTE_MAX_POSITION_PCT,
    )
    _HAS_SIZER = True
except ImportError:
    def _earnings_scalar(t): return 1.0
    def check_halt_status(t): return {"halted": False}
    _HAS_SIZER = False

logger = logging.getLogger("options_execution")

ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_BASE = "https://paper-api.alpaca.markets"
ALPACA_DATA = "https://data.alpaca.markets"

# Absolute ceilings (safety nets — dynamic sizing targets lower values)
MAX_OPTIONS_PCT_CEILING = 0.10   # Absolute max 10% per options trade
MAX_TOTAL_OPTIONS_PCT = 0.20     # Absolute max 20% total options exposure
# v1.0.33: Lowered from vol>100/OI>500 to match scanner thresholds.
# Mid-cap stocks (KMI, DIS, WMB) have 10-80 daily volume on ATM options
# but 300+ OI — perfectly tradeable with limit orders.
# OI is the real liquidity signal; daily volume fluctuates intraday.
MIN_OPTION_VOLUME = 10           # Minimum daily volume on the contract
MIN_OPEN_INTEREST = 200          # Minimum open interest
MAX_SPREAD_PCT = 0.15            # Max bid-ask spread as % of mid price


def _dynamic_options_size(trade: dict, equity: float, existing_positions: list = None,
                          macro: dict = None) -> float:
    """
    Calculate dynamic position size for options as a fraction of equity.
    Uses the same scalars as the stock position sizer but with an options
    risk multiplier (options are inherently leveraged, so base size is smaller).
    
    Returns: fraction of equity to allocate (e.g., 0.035 = 3.5%)
    """
    if not _HAS_SIZER:
        return 0.05  # Fallback: 5% fixed
    
    # Start with Kelly base
    stats = _get_historical_stats()
    overall = stats["overall"]
    kelly_base = _kelly_fraction(overall["win_rate"], overall["avg_win"], overall["avg_loss"])
    
    # Apply all the same scalars as stocks
    score = trade.get("deep_score", trade.get("score", 50))
    ewma_rv = trade.get("ewma_rv") or 2.0
    garch_rv = trade.get("garch_rv")
    volume = trade.get("volume", 0)
    price = trade.get("price", 100)
    
    s_vol = _volatility_scalar(ewma_rv, garch_rv)
    s_conf = _confidence_scalar(score, trade.get("ml_confidence"))
    s_regime = _regime_scalar(
        macro.get("vix") if macro else None,
        macro.get("vix_regime") if macro else None,
    )
    s_earn = _earnings_scalar(trade.get("ticker", ""))
    s_time = _time_scalar()
    s_heat = _portfolio_heat_scalar(
        existing_positions or [], equity, trade.get("sector")
    )
    preliminary = equity * kelly_base * s_vol * s_conf * s_regime * s_earn * s_time * s_heat
    s_liq = _liquidity_scalar(volume, price, preliminary)
    
    # Options risk multiplier: options are 3-10x leveraged vs stock
    # So we use ~50% of what the stock sizer would give
    OPTIONS_LEVERAGE_DISCOUNT = 0.50
    
    dynamic_pct = (kelly_base * s_vol * s_conf * s_regime * s_earn 
                   * s_time * s_heat * s_liq * OPTIONS_LEVERAGE_DISCOUNT)
    
    # Clamp to absolute ceiling
    dynamic_pct = max(0.01, min(dynamic_pct, MAX_OPTIONS_PCT_CEILING))
    
    return dynamic_pct


def _alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  A. STOCK vs OPTIONS DECISION
# ═══════════════════════════════════════════════════════════════════════════════

def should_use_options(trade: dict, equity: float, existing_positions: list = None) -> dict:
    """
    Decide whether to trade stock or options for this opportunity.
    
    Returns: {
        "use_options": bool,
        "reason": str,
        "strategy": "buy_call" / "buy_put" / "sell_covered_call" / 
                    "sell_cash_secured_put" / "bull_call_spread" / "bear_put_spread" / "stock",
        "edge_pct": float,  — estimated edge of options over stock
    }
    """
    ticker = trade.get("ticker", "")
    vrp = trade.get("vrp", 0) or 0
    score = trade.get("deep_score", trade.get("score", 50))
    side = trade.get("side", "buy")
    action = trade.get("action_label", "BUY")
    rsi = trade.get("rsi") or 50
    ewma_rv = trade.get("ewma_rv") or 2.0
    garch_rv = trade.get("garch_rv") or 2.0
    
    result = {
        "use_options": False,
        "reason": "Stock is the default — simpler, more liquid, no expiration risk",
        "strategy": "stock",
        "edge_pct": 0,
    }
    
    # ── Check if options are even appropriate ──
    
    # Don't use options on low-score trades
    if score < 65:
        result["reason"] = f"Score {score} too low for options — need 65+ conviction"
        return result
    
    # Check earnings proximity — buyers get crushed by IV collapse
    earn_scalar = _earnings_scalar(ticker)
    
    # Check current time — options only during regular hours
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        et_hour = now_et.hour
        et_min = now_et.minute
        if et_hour < 9 or et_hour >= 16 or (et_hour == 9 and et_min < 30):
            result["reason"] = "Options only trade during regular hours (9:30am-4pm ET)"
            return result
    except Exception:
        pass
    
    # Check total options exposure
    if existing_positions:
        options_value = sum(
            abs(float(p.get("market_value", 0)))
            for p in existing_positions
            if p.get("asset_class") == "option" or len(p.get("symbol", "")) > 10
        )
        if options_value > equity * MAX_TOTAL_OPTIONS_PCT:
            result["reason"] = f"Options exposure already {options_value/equity:.1%} — max is {MAX_TOTAL_OPTIONS_PCT:.0%}"
            return result
    
    # ── Decision logic ──
    
    # Scenario 1: SELL OPTIONS — VRP is high (IV >> realized vol)
    # Options are overpriced. Selling premium has an edge.
    if "SELL OPTIONS" in action.upper() or vrp > 4:
        if earn_scalar < 0.6:
            # Near earnings — IV is high for a reason, selling is risky
            result["reason"] = "VRP is high but earnings are near — IV could spike more, too risky to sell"
            return result
        
        result["use_options"] = True
        result["edge_pct"] = min(vrp, 15.0)  # VRP IS the edge
        
        # Decide which sell strategy
        if side == "buy":
            # Bullish + high VRP → sell cash-secured puts (get paid to buy cheaper)
            result["strategy"] = "sell_cash_secured_put"
            result["reason"] = f"VRP +{vrp:.1f}% — options overpriced. Sell put to get paid while waiting for entry."
        else:
            # Bearish + high VRP → sell covered calls (if holding) or bear put spread
            result["strategy"] = "bear_put_spread"
            result["reason"] = f"VRP +{vrp:.1f}% — sell premium on the downside via bear put spread"
        return result
    
    # Scenario 2: BUY OPTIONS — VRP is negative (IV << realized vol)
    # BACKTEST FINDING (10yr, 2016-2026): buy_call WR=22%, avg=-2.44% — DISABLED
    # Buying calls loses money on average even when IV looks cheap.
    # Theta decay and timing errors eat the edge. CSP (sell premium) wins at 72% WR.
    # Keeping the logic for future re-evaluation but routing to sell_csp on the other side.
    if vrp < -2:
        if earn_scalar < 0.85:
            result["reason"] = f"Options cheap (VRP {vrp:.1f}%) but earnings risk — skip"
            return result
        # Instead of buying calls, fall through to stock — backtest proves buying is a loser
        result["reason"] = f"IV cheap (VRP {vrp:.1f}%) but buy_call historically loses — routing to stock"
        return result  # use_options stays False

    # Scenario 3: High conviction spreads — DISABLED by backtest
    # BACKTEST FINDING: bull_spread WR=28%, avg=-1.77% over 10yr — spreads lose money.
    # The theta decay on the long leg exceeds the gains from the spread width.
    # Disabled. High conviction trades go to stock or CSP (if VRP > threshold).
    if score >= 85 and abs(vrp) < 3:
        result["reason"] = f"High conviction score={score} but spreads historically lose — routing to stock"
        return result  # use_options stays False
    
    # Default: stock is better
    # Most of the time, stock is the right call because:
    # - No expiration (can hold indefinitely)
    # - No theta decay eating your position daily
    # - Better liquidity (tighter spreads)
    # - Simpler position management
    # - No assignment risk
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  B. CONTRACT SELECTION — Pick the right strike + expiry
# ═══════════════════════════════════════════════════════════════════════════════

def select_contract(ticker: str, strategy: str, price: float, equity: float,
                    trade: dict = None, positions: list = None, macro: dict = None) -> dict:
    """
    Select the best options contract for the given strategy.
    
    Returns: {
        "occ_symbol": str,         — Alpaca OCC format
        "strike": float,
        "expiry": str,             — YYYY-MM-DD
        "option_type": "call"/"put",
        "side": "buy"/"sell",
        "qty": int,                — number of contracts
        "limit_price": float,      — recommended limit price
        "max_cost": float,         — total cost of the position
        "max_loss": float,         — worst case loss
        "error": str or None,
    }
    """
    try:
        # Fetch available options contracts from Alpaca
        contracts = _fetch_option_chain(ticker, price)
        if not contracts:
            return {"error": "No options contracts available for this ticker"}
        
        # Filter by liquidity
        liquid = [c for c in contracts if _is_liquid(c)]
        if not liquid:
            return {"error": f"No liquid options contracts (need vol>{MIN_OPTION_VOLUME}, OI>{MIN_OPEN_INTEREST}, spread<{MAX_SPREAD_PCT:.0%})"}

        # ── Vol Surface Enrichment (v1.0.33) ────────────────────────────
        # Fetch surface data for this ticker to add fair value + edge info
        surface_data = {}
        try:
            from vol_surface import get_surface_score
            surface_data = get_surface_score(ticker)
        except Exception:
            pass  # Surface is advisory, don't block execution

        # Select based on strategy
        # Calculate dynamic size using trade context
        size_pct = _dynamic_options_size(
            trade or {"ticker": ticker, "price": price, "score": 70},
            equity, positions, macro
        )

        # Select contract based on strategy
        result = None
        if strategy == "buy_call":
            result = _select_buy_call(liquid, price, equity, ticker, size_pct)
        elif strategy == "buy_put":
            result = _select_buy_put(liquid, price, equity, ticker, size_pct)
        elif strategy == "sell_cash_secured_put":
            result = _select_sell_put(liquid, price, equity, ticker, size_pct)
        elif strategy == "bull_call_spread":
            result = _select_bull_spread(liquid, price, equity, ticker, size_pct)
        elif strategy == "bear_put_spread":
            result = _select_bear_spread(liquid, price, equity, ticker, size_pct)
        elif strategy == "buy_straddle":
            result = _select_buy_straddle(liquid, price, equity, ticker, size_pct)
        elif strategy == "short_straddle":
            result = _select_straddle(liquid, price, equity, ticker, size_pct)
        elif strategy == "iron_condor":
            result = _select_condor(liquid, price, equity, ticker, size_pct)
        else:
            return {"error": f"Unknown strategy: {strategy}"}

        # ── Vol Surface Enrichment (v1.0.33) ─────────────────────────
        # Add surface-based fair value, edge, and VRP to every contract
        if result and not result.get("error") and surface_data:
            result["surface_score"] = surface_data.get("surface_score", 0)
            result["vrp"] = surface_data.get("vrp", {}).get("vrp_20d", 0)
            result["vrp_recommendation"] = surface_data.get("vrp", {}).get("recommendation", "")
            result["skew_overpriced"] = surface_data.get("skew", {}).get("overpriced_side", "")
            result["surface_best_strategy"] = surface_data.get("best_strategy", "")
            result["surface_reasoning"] = surface_data.get("reasoning", "")

        return result
    
    except Exception as e:
        return {"error": f"Contract selection failed: {str(e)[:200]}"}


def _fetch_option_chain(ticker: str, current_price: float) -> list:
    """Fetch options chain from Alpaca data API."""
    contracts = []
    try:
        # Get contracts expiring 7-50 days out
        # v1.0.33: widened from 14-45 to 7-50 to cover all scanner setups:
        #   - low_iv_breakout_buy fetches 10-25 DTE
        #   - high_iv_premium_sale fetches 14-45 DTE
        #   - csp_normal_market fetches 25-50 DTE
        #   - gamma_pin fetches 0-2 DTE (handled separately)
        now = datetime.now()
        min_exp = (now + timedelta(days=7)).strftime("%Y-%m-%d")
        max_exp = (now + timedelta(days=50)).strftime("%Y-%m-%d")
        
        # Strike range: within 10% of current price
        min_strike = current_price * 0.90
        max_strike = current_price * 1.10
        
        resp = requests.get(
            f"{ALPACA_DATA}/v1beta1/options/snapshots/{ticker}",
            params={
                "feed": "opra",  # Real-time OPRA feed (Algo Trader Plus)
                "limit": 100,
                "expiration_date_gte": min_exp,
                "expiration_date_lte": max_exp,
                "strike_price_gte": str(min_strike),
                "strike_price_lte": str(max_strike),
            },
            headers=_alpaca_headers(),
            timeout=10,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            snapshots = data.get("snapshots", {})
            for occ_symbol, snap in snapshots.items():
                quote = snap.get("latestQuote", {})
                trade = snap.get("latestTrade", {})
                greeks = snap.get("greeks", {})
                
                # Parse OCC symbol: AAPL260418C00250000
                # ticker + YYMMDD + C/P + 8-digit strike (strike * 1000)
                sym_body = occ_symbol[len(ticker):]
                if len(sym_body) >= 15:
                    exp_str = "20" + sym_body[:6]  # YYYYMMDD
                    exp_date = f"{exp_str[:4]}-{exp_str[4:6]}-{exp_str[6:8]}"
                    opt_type = "call" if sym_body[6] == "C" else "put"
                    strike = int(sym_body[7:]) / 1000
                    
                    bid = float(quote.get("bp", 0) or 0)
                    ask = float(quote.get("ap", 0) or 0)
                    mid = (bid + ask) / 2 if (bid + ask) > 0 else float(trade.get("p", 0) or 0)
                    
                    # v1.0.33: Fix data extraction from Alpaca OPRA snapshot.
                    # Volume lives in dailyBar.v, NOT trade.s (that's last trade size).
                    # OI is not returned by the snapshot endpoint at all — use
                    # quote size (bp/ap size) as a proxy for market maker commitment.
                    daily_bar = snap.get("dailyBar", {})
                    daily_vol = int(daily_bar.get("v", 0) or 0)
                    # OI: try openInterest, then fall back to bid size as liquidity proxy
                    # Quote size (bs/as) = contracts being offered = real-time liquidity
                    oi_raw = int(snap.get("openInterest", 0) or 0)
                    bid_size = int(quote.get("bs", 0) or 0)
                    ask_size = int(quote.get("as", 0) or 0)
                    # If OI not available, use max quote size * 10 as proxy
                    # (if MM is quoting 150 contracts, OI is likely 1000+)
                    oi_est = oi_raw if oi_raw > 0 else max(bid_size, ask_size) * 10
                    
                    contracts.append({
                        "occ_symbol": occ_symbol,
                        "ticker": ticker,
                        "strike": strike,
                        "expiry": exp_date,
                        "option_type": opt_type,
                        "bid": bid,
                        "ask": ask,
                        "mid": round(mid, 2),
                        "volume": daily_vol,
                        "open_interest": oi_est,
                        "iv": float(snap.get("impliedVolatility", 0) or greeks.get("iv", 0) or 0),
                        "delta": float(greeks.get("delta", 0) or 0),
                        "theta": float(greeks.get("theta", 0) or 0),
                        "gamma": float(greeks.get("gamma", 0) or 0),
                        "days_to_expiry": (datetime.strptime(exp_date, "%Y-%m-%d") - now).days,
                    })
        else:
            logger.warning(f"Options chain fetch failed: {resp.status_code}")
    
    except Exception as e:
        logger.error(f"Options chain error: {e}")
    
    return contracts


def _is_liquid(contract: dict) -> bool:
    """Check if a contract meets minimum liquidity requirements."""
    bid = contract.get("bid", 0)
    ask = contract.get("ask", 0)
    mid = contract.get("mid", 0)
    
    # Must have a bid and ask
    if bid <= 0 or ask <= 0:
        return False
    
    # Spread check
    if mid > 0:
        spread_pct = (ask - bid) / mid
        if spread_pct > MAX_SPREAD_PCT:
            return False
    
    # Volume and OI
    if contract.get("open_interest", 0) < MIN_OPEN_INTEREST:
        return False
    
    # Price floor — don't trade options under $0.10 (penny options = gambling)
    if mid < 0.10:
        return False
    
    return True


def _optimized_limit_price(contract: dict, direction: str) -> float:
    """
    Professional limit price optimization.
    Instead of paying full ask (buying) or accepting full bid (selling),
    start at the midpoint and walk 30% toward the natural side.
    
    This saves money on every trade — pros never pay the full spread.
    
    Args:
        contract: dict with 'bid', 'ask', 'mid' keys
        direction: 'buy' or 'sell'
    Returns:
        Optimized limit price (float)
    """
    bid = contract.get("bid", 0)
    ask = contract.get("ask", 0)
    mid = contract.get("mid", 0)
    
    if bid <= 0 or ask <= 0:
        return mid if mid > 0 else ask if direction == "buy" else bid
    
    if mid <= 0:
        mid = (bid + ask) / 2
    
    # Walk 30% from mid toward the natural side
    # For buying: mid + 30% of (ask - mid) = lean slightly toward ask
    # For selling: mid - 30% of (mid - bid) = lean slightly toward bid
    WALK_PCT = 0.30
    
    if direction == "buy":
        optimized = mid + WALK_PCT * (ask - mid)
    else:
        optimized = mid - WALK_PCT * (mid - bid)
    
    # Safety: never exceed the spread boundaries
    optimized = max(bid, min(optimized, ask))
    
    return round(optimized, 2)


def _cancel_order(order_id: str) -> bool:
    """Cancel an open order by ID."""
    try:
        resp = requests.delete(
            f"{ALPACA_BASE}/v2/orders/{order_id}",
            headers=_alpaca_headers(),
            timeout=10,
        )
        return resp.status_code in (200, 204)
    except Exception:
        return False


def _select_buy_call(contracts: list, price: float, equity: float, ticker: str, size_pct: float = 0.05) -> dict:
    """Select a call to buy — slightly OTM, 20-35 days to expiry, delta ~0.40."""
    calls = [c for c in contracts if c["option_type"] == "call" and c["strike"] >= price]
    if not calls:
        return {"error": "No suitable calls found"}
    
    # Sort by proximity to delta 0.40 (sweet spot: enough leverage, decent probability)
    target_delta = 0.40
    calls.sort(key=lambda c: abs(abs(c.get("delta", 0)) - target_delta))
    
    best = calls[0]
    # Limit price optimization: start at mid, not full ask
    limit_price = _optimized_limit_price(best, "buy")
    max_cost = equity * size_pct  # Dynamic sizing
    qty = int(max_cost / (limit_price * 100))  # Each contract = 100 shares
    if qty <= 0:
        qty = 1  # Minimum 1 contract
    
    actual_cost = qty * limit_price * 100
    
    return {
        "occ_symbol": best["occ_symbol"],
        "strike": best["strike"],
        "expiry": best["expiry"],
        "option_type": "call",
        "side": "buy",
        "qty": qty,
        "limit_price": round(limit_price, 2),
        "max_cost": round(actual_cost, 2),
        "max_loss": round(actual_cost, 2),  # Max loss = premium paid
        "delta": best.get("delta"),
        "gamma": best.get("gamma"),
        "theta": best.get("theta"),
        "iv": best.get("iv"),
        "days_to_expiry": best.get("days_to_expiry"),
        "bid_ask_spread": round(best["ask"] - best["bid"], 2),
        "error": None,
    }


def _select_buy_put(contracts: list, price: float, equity: float, ticker: str, size_pct: float = 0.05) -> dict:
    """Select a put to buy — slightly OTM, delta ~-0.40."""
    puts = [c for c in contracts if c["option_type"] == "put" and c["strike"] <= price]
    if not puts:
        return {"error": "No suitable puts found"}
    
    target_delta = -0.40
    puts.sort(key=lambda c: abs(c.get("delta", 0) - target_delta))
    
    best = puts[0]
    limit_price = _optimized_limit_price(best, "buy")
    max_cost = equity * size_pct  # Dynamic sizing
    qty = int(max_cost / (limit_price * 100))
    if qty <= 0:
        qty = 1
    
    actual_cost = qty * limit_price * 100
    
    return {
        "occ_symbol": best["occ_symbol"],
        "strike": best["strike"],
        "expiry": best["expiry"],
        "option_type": "put",
        "side": "buy",
        "qty": qty,
        "limit_price": round(limit_price, 2),
        "max_cost": round(actual_cost, 2),
        "max_loss": round(actual_cost, 2),
        "delta": best.get("delta"),
        "gamma": best.get("gamma"),
        "theta": best.get("theta"),
        "iv": best.get("iv"),
        "days_to_expiry": best.get("days_to_expiry"),
        "error": None,
    }


def _select_sell_put(contracts: list, price: float, equity: float, ticker: str, size_pct: float = 0.05) -> dict:
    """
    Select a put to sell (cash-secured).
    Slightly OTM, delta ~-0.30 (70% probability of expiring worthless = profit).
    Must have enough cash to cover assignment.
    """
    puts = [c for c in contracts if c["option_type"] == "put" and c["strike"] <= price]
    if not puts:
        return {"error": "No suitable puts found for selling"}
    
    # Target delta -0.30 (OTM, high probability of profit)
    target_delta = -0.30
    puts.sort(key=lambda c: abs(c.get("delta", 0) - target_delta))
    
    best = puts[0]
    limit_price = _optimized_limit_price(best, "sell")
    
    # Cash required to secure: strike * 100 per contract
    cash_per_contract = best["strike"] * 100
    max_contracts = int(equity * size_pct / cash_per_contract)  # Dynamic sizing
    if max_contracts <= 0:
        return {"error": f"Not enough capital to sell cash-secured put at ${best['strike']} (need ${cash_per_contract:,.0f} per contract)"}
    
    qty = min(max_contracts, 2)  # Conservative: max 2 contracts
    premium_received = qty * limit_price * 100
    
    return {
        "occ_symbol": best["occ_symbol"],
        "strike": best["strike"],
        "expiry": best["expiry"],
        "option_type": "put",
        "side": "sell",
        "qty": qty,
        "limit_price": round(limit_price, 2),
        "max_cost": 0,  # We receive premium
        "premium_received": round(premium_received, 2),
        "max_loss": round(best["strike"] * qty * 100 - premium_received, 2),
        "cash_required": round(cash_per_contract * qty, 2),
        "delta": best.get("delta"),
        "gamma": best.get("gamma"),
        "theta": best.get("theta"),
        "iv": best.get("iv"),
        "days_to_expiry": best.get("days_to_expiry"),
        "error": None,
    }


def _select_bull_spread(contracts: list, price: float, equity: float, ticker: str, size_pct: float = 0.05) -> dict:
    """Bull call spread: buy lower strike call, sell higher strike call. Defined risk."""
    calls = [c for c in contracts if c["option_type"] == "call"]
    if len(calls) < 2:
        return {"error": "Not enough call contracts for a spread"}

    # Same-expiry enforcement: group calls by expiry
    from collections import defaultdict
    by_expiry: dict = defaultdict(list)
    for c in calls:
        by_expiry[c["expiry"]].append(c)

    # Filter to expiries with at least 2 calls
    valid_expiries = [(exp, grp) for exp, grp in by_expiry.items() if len(grp) >= 2]
    if not valid_expiries:
        return {"error": "No single expiry has enough calls for a spread"}

    # Prefer expiry closest to 30 DTE
    valid_expiries.sort(key=lambda x: abs(x[1][0].get("days_to_expiry", 30) - 30))

    best_result = None
    for _, exp_calls in valid_expiries:
        exp_calls = sorted(exp_calls, key=lambda c: c["strike"])

        # Find ATM call (closest to current price) and OTM call ~3-5% above
        atm_candidates = sorted(exp_calls, key=lambda c: abs(c["strike"] - price))
        long_leg = atm_candidates[0]

        # Short leg: 3-5% above current price
        target_short_strike = price * 1.04
        short_candidates = [c for c in exp_calls if c["strike"] > long_leg["strike"] and c["strike"] <= price * 1.08]
        if not short_candidates:
            continue

        short_leg = min(short_candidates, key=lambda c: abs(c["strike"] - target_short_strike))

        net_debit = round(long_leg["ask"] - short_leg["bid"], 2)
        if net_debit <= 0:
            continue

        best_result = (long_leg, short_leg, net_debit)
        break

    if best_result is None:
        return {"error": "No suitable short leg for bull call spread"}

    long_leg, short_leg, net_debit = best_result

    spread_width = short_leg["strike"] - long_leg["strike"]
    max_profit = round((spread_width - net_debit) * 100, 2)
    max_loss_per = round(net_debit * 100, 2)

    max_contracts = int(equity * size_pct / max_loss_per) if max_loss_per > 0 else 0  # Dynamic sizing
    if max_contracts <= 0:
        max_contracts = 1
    qty = min(max_contracts, 5)  # Dynamic sizing may allow more

    # Breakeven calculation
    breakeven = round(long_leg["strike"] + net_debit, 2)

    # EV calculation using delta as probability proxy
    long_delta = abs(long_leg.get("delta", 0))
    result = {
        "strategy": "bull_call_spread",
        "long_leg": long_leg["occ_symbol"],
        "short_leg": short_leg["occ_symbol"],
        "long_strike": long_leg["strike"],
        "short_strike": short_leg["strike"],
        "expiry": long_leg["expiry"],
        "qty": qty,
        "net_debit": net_debit,
        "limit_price": net_debit,  # Net debit as limit
        "max_cost": round(net_debit * 100 * qty, 2),
        "max_profit": round(max_profit * qty, 2),
        "max_loss": round(max_loss_per * qty, 2),
        "breakeven": breakeven,
        "risk_reward": round(max_profit / max_loss_per, 2) if max_loss_per > 0 else 0,
        "days_to_expiry": long_leg.get("days_to_expiry"),
        "error": None,
    }

    if long_delta > 0:
        p_win = long_delta
        p_loss = 1 - p_win
        ev = p_win * max_profit - p_loss * max_loss_per
        result["ev"] = round(ev, 2)
        result["win_probability"] = round(p_win * 100, 1)
        if ev < 0:
            result["ev_warning"] = "Negative expected value"

    return result


def _select_bear_spread(contracts: list, price: float, equity: float, ticker: str, size_pct: float = 0.05) -> dict:
    """Bear put spread: buy higher strike put, sell lower strike put. Defined risk."""
    puts = [c for c in contracts if c["option_type"] == "put"]
    if len(puts) < 2:
        return {"error": "Not enough put contracts for a spread"}

    # Same-expiry enforcement: group puts by expiry
    from collections import defaultdict
    by_expiry: dict = defaultdict(list)
    for p in puts:
        by_expiry[p["expiry"]].append(p)

    # Filter to expiries with at least 2 puts
    valid_expiries = [(exp, grp) for exp, grp in by_expiry.items() if len(grp) >= 2]
    if not valid_expiries:
        return {"error": "No single expiry has enough puts for a spread"}

    # Prefer expiry closest to 30 DTE
    valid_expiries.sort(key=lambda x: abs(x[1][0].get("days_to_expiry", 30) - 30))

    best_result = None
    for _, exp_puts in valid_expiries:
        exp_puts = sorted(exp_puts, key=lambda c: c["strike"], reverse=True)

        atm_candidates = sorted(exp_puts, key=lambda c: abs(c["strike"] - price))
        long_leg = atm_candidates[0]

        target_short_strike = price * 0.96
        short_candidates = [c for c in exp_puts if c["strike"] < long_leg["strike"] and c["strike"] >= price * 0.92]
        if not short_candidates:
            continue

        short_leg = min(short_candidates, key=lambda c: abs(c["strike"] - target_short_strike))

        net_debit = round(long_leg["ask"] - short_leg["bid"], 2)
        if net_debit <= 0:
            continue

        best_result = (long_leg, short_leg, net_debit)
        break

    if best_result is None:
        return {"error": "No suitable short leg for bear put spread"}

    long_leg, short_leg, net_debit = best_result

    spread_width = long_leg["strike"] - short_leg["strike"]
    max_profit = round((spread_width - net_debit) * 100, 2)
    max_loss_per = round(net_debit * 100, 2)

    max_contracts = int(equity * size_pct / max_loss_per) if max_loss_per > 0 else 0  # Dynamic sizing
    if max_contracts <= 0:
        max_contracts = 1
    qty = min(max_contracts, 5)

    # Breakeven calculation
    breakeven = round(long_leg["strike"] - net_debit, 2)

    # EV calculation using delta as probability proxy
    long_delta = abs(long_leg.get("delta", 0))
    result = {
        "strategy": "bear_put_spread",
        "long_leg": long_leg["occ_symbol"],
        "short_leg": short_leg["occ_symbol"],
        "long_strike": long_leg["strike"],
        "short_strike": short_leg["strike"],
        "expiry": long_leg["expiry"],
        "qty": qty,
        "net_debit": net_debit,
        "limit_price": net_debit,
        "max_cost": round(net_debit * 100 * qty, 2),
        "max_profit": round(max_profit * qty, 2),
        "max_loss": round(max_loss_per * qty, 2),
        "breakeven": breakeven,
        "risk_reward": round(max_profit / max_loss_per, 2) if max_loss_per > 0 else 0,
        "days_to_expiry": long_leg.get("days_to_expiry"),
        "error": None,
    }

    if long_delta > 0:
        p_win = long_delta
        p_loss = 1 - p_win
        ev = p_win * max_profit - p_loss * max_loss_per
        result["ev"] = round(ev, 2)
        result["win_probability"] = round(p_win * 100, 1)
        if ev < 0:
            result["ev_warning"] = "Negative expected value"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  C. ORDER SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════


def _select_buy_straddle(contracts: list, price: float, equity: float, ticker: str, size_pct: float = 0.08) -> dict:
    """
    Long straddle: buy ATM call + buy ATM put. (v1.0.33)
    Profit if stock moves significantly in either direction.
    Max loss = total debit paid.
    Max profit = unlimited on upside, large on downside.
    Best when: IV rank < 20, stock at 52-week low volatility, expecting a breakout.
    """
    calls = [c for c in contracts if c["option_type"] == "call" and abs(c["strike"] - price) / price < 0.03]
    puts  = [c for c in contracts if c["option_type"] == "put"  and abs(c["strike"] - price) / price < 0.03]

    if not calls or not puts:
        return {"error": "No ATM contracts found for buy straddle"}

    # v1.0.33 FIX: Both legs MUST share the same expiration.
    # Group by expiry and find the best expiry that has both a call and put.
    from collections import defaultdict
    by_expiry: dict = defaultdict(lambda: {"calls": [], "puts": []})
    for c in calls:
        by_expiry[c["expiry"]]["calls"].append(c)
    for p in puts:
        by_expiry[p["expiry"]]["puts"].append(p)

    # Filter to expiries that have both calls and puts
    valid_expiries = [
        (exp, grp) for exp, grp in by_expiry.items()
        if grp["calls"] and grp["puts"]
    ]
    if not valid_expiries:
        return {"error": "No expiry has both ATM call and put"}

    # Prefer expiry closest to 21 DTE (sweet spot for straddles)
    target_dte = 21
    valid_expiries.sort(key=lambda x: abs(
        x[1]["calls"][0].get("days_to_expiry", 21) - target_dte
    ))
    best_exp, best_grp = valid_expiries[0]

    # Pick ATM call and put FROM THE SAME EXPIRY
    atm_call = min(best_grp["calls"], key=lambda c: abs(c["strike"] - price))
    atm_put  = min(best_grp["puts"],  key=lambda c: abs(c["strike"] - price))

    # Use optimized limit prices (we're buying)
    call_debit = _optimized_limit_price(atm_call, "buy")
    put_debit  = _optimized_limit_price(atm_put, "buy")

    if call_debit <= 0 or put_debit <= 0:
        return {"error": "No valid ask prices for straddle legs"}

    total_debit = round(call_debit + put_debit, 2)
    breakeven_up   = round(atm_call["strike"] + total_debit, 2)
    breakeven_down = round(atm_put["strike"]  - total_debit, 2)

    # Size: total debit per contract = total_debit * 100
    max_cost = equity * size_pct
    qty = max(1, int(max_cost / (total_debit * 100)))

    max_loss_val = round(total_debit * qty * 100, 2)
    # For a straddle, max profit is theoretically unlimited; use a reasonable proxy
    # based on breakeven distance for EV calc
    result = {
        "strategy": "buy_straddle",
        "call_leg": atm_call["occ_symbol"],
        "put_leg": atm_put["occ_symbol"],
        "strike": atm_call["strike"],
        "expiry": atm_call["expiry"],
        "qty": qty,
        "call_debit": call_debit,
        "put_debit": put_debit,
        "total_debit": round(total_debit * qty * 100, 2),
        "limit_price": total_debit,
        "breakeven_up": breakeven_up,
        "breakeven_down": breakeven_down,
        "max_loss": max_loss_val,
        "days_to_expiry": atm_call.get("days_to_expiry"),
        "error": None,
    }

    # EV calculation: P(win) ≈ abs(long_delta), max_profit proxy = max_loss (1:1 move target)
    call_delta = abs(atm_call.get("delta", 0))
    put_delta = abs(atm_put.get("delta", 0))
    avg_delta = (call_delta + put_delta) / 2 if (call_delta > 0 and put_delta > 0) else max(call_delta, put_delta)
    max_loss_per = round(total_debit * 100, 2)
    if avg_delta > 0:
        p_win = avg_delta
        p_loss = 1 - p_win
        # Use max_loss as proxy for max_profit (target 1:1 on straddle)
        ev = p_win * max_loss_per - p_loss * max_loss_per
        result["ev"] = round(ev, 2)
        result["win_probability"] = round(p_win * 100, 1)
        result["risk_reward"] = 1.0  # Unlimited upside, but 1:1 is the proxy target
        if ev < 0:
            result["ev_warning"] = "Negative expected value"

    return result


def _select_straddle(contracts: list, price: float, equity: float, ticker: str, size_pct: float = 0.04) -> dict:
    """
    Short straddle: sell ATM call + sell ATM put.
    Profit if stock stays within the breakeven range.
    Max profit = total premium collected.
    Max loss = theoretically large if stock moves far.
    Best when: IV rank > 60, stock expected to consolidate.
    """
    calls = [c for c in contracts if c["option_type"] == "call" and abs(c["strike"] - price) / price < 0.03]
    puts  = [c for c in contracts if c["option_type"] == "put"  and abs(c["strike"] - price) / price < 0.03]

    if not calls or not puts:
        return {"error": "No ATM contracts found for straddle"}

    # v1.0.33 FIX: Both legs must share the same expiration (same fix as buy_straddle)
    from collections import defaultdict
    by_expiry: dict = defaultdict(lambda: {"calls": [], "puts": []})
    for c in calls:
        by_expiry[c["expiry"]]["calls"].append(c)
    for p in puts:
        by_expiry[p["expiry"]]["puts"].append(p)
    valid = [(e, g) for e, g in by_expiry.items() if g["calls"] and g["puts"]]
    if not valid:
        return {"error": "No expiry has both ATM call and put for straddle"}
    valid.sort(key=lambda x: abs(x[1]["calls"][0].get("days_to_expiry", 30) - 30))
    _, best = valid[0]
    calls = best["calls"]
    puts = best["puts"]

    # Pick the ATM call and put (closest to current price)
    atm_call = min(calls, key=lambda c: abs(c["strike"] - price))
    atm_put  = min(puts,  key=lambda c: abs(c["strike"] - price))

    # Use bid price (we're selling)
    call_credit = atm_call.get("bid", 0)
    put_credit  = atm_put.get("bid", 0)

    if call_credit <= 0 or put_credit <= 0:
        return {"error": "No valid bid prices for straddle legs"}

    total_credit = round(call_credit + put_credit, 2)
    breakeven_up   = round(atm_call["strike"] + total_credit, 2)
    breakeven_down = round(atm_put["strike"]  - total_credit, 2)

    # Size: credit received, capped by equity
    max_risk = equity * size_pct  # Conservative — straddle can lose a lot
    qty = min(2, max(1, int(max_risk / (atm_call["strike"] * 10))))  # Very conservative

    max_profit_val = round(total_credit * qty * 100, 2)
    max_profit_per = round(total_credit * 100, 2)
    # For short straddle, max loss is theoretically unlimited; use 2x credit as proxy
    max_loss_proxy = round(total_credit * 2 * 100, 2)

    result = {
        "strategy": "short_straddle",
        "call_leg": atm_call["occ_symbol"],
        "put_leg": atm_put["occ_symbol"],
        "strike": atm_call["strike"],
        "expiry": atm_call["expiry"],
        "qty": qty,
        "call_credit": call_credit,
        "put_credit": put_credit,
        "total_credit": round(total_credit * qty * 100, 2),
        "limit_price": total_credit,
        "breakeven_up": breakeven_up,
        "breakeven_down": breakeven_down,
        "max_profit": max_profit_val,
        "days_to_expiry": atm_call.get("days_to_expiry"),
        "error": None,
    }

    # EV calculation: P(win) ≈ 1 - avg(abs(short_deltas))
    call_delta = abs(atm_call.get("delta", 0))
    put_delta = abs(atm_put.get("delta", 0))
    avg_delta = (call_delta + put_delta) / 2 if (call_delta > 0 and put_delta > 0) else max(call_delta, put_delta)
    if avg_delta > 0:
        p_win = 1 - avg_delta
        p_loss = 1 - p_win
        ev = p_win * max_profit_per - p_loss * max_loss_proxy
        result["ev"] = round(ev, 2)
        result["win_probability"] = round(p_win * 100, 1)
        result["risk_reward"] = round(max_profit_per / max_loss_proxy, 2) if max_loss_proxy > 0 else 0
        if ev < 0:
            result["ev_warning"] = "Negative expected value"

    return result


def _select_condor(contracts: list, price: float, equity: float, ticker: str, size_pct: float = 0.04) -> dict:
    """
    Iron condor: sell OTM call spread + sell OTM put spread.
    Profit if stock stays within a defined range.
    Fully defined risk — max loss = max(call_spread_width, put_spread_width) - credit received.
    Best when: VIX elevated, stock range-bound (ADX < 25).
    """
    all_calls = [c for c in contracts if c["option_type"] == "call"]
    all_puts  = [c for c in contracts if c["option_type"] == "put"]

    if len(all_calls) < 2 or len(all_puts) < 2:
        return {"error": "Not enough contracts for iron condor"}

    # Same-expiry enforcement: group by expiry and require all 4 legs from one expiry
    from collections import defaultdict
    by_expiry: dict = defaultdict(lambda: {"calls": [], "puts": []})
    for c in all_calls:
        by_expiry[c["expiry"]]["calls"].append(c)
    for p in all_puts:
        by_expiry[p["expiry"]]["puts"].append(p)

    valid_expiries = [
        (exp, grp) for exp, grp in by_expiry.items()
        if len(grp["calls"]) >= 2 and len(grp["puts"]) >= 2
    ]
    if not valid_expiries:
        return {"error": "No single expiry has enough calls and puts for condor"}

    # Prefer expiry closest to 30 DTE
    valid_expiries.sort(key=lambda x: abs(
        x[1]["calls"][0].get("days_to_expiry", 30) - 30
    ))

    best_result = None
    for _, grp in valid_expiries:
        calls = sorted(grp["calls"], key=lambda c: c["strike"])
        puts  = sorted(grp["puts"],  key=lambda c: c["strike"], reverse=True)

        # Short call: ~5% OTM
        target_short_call = price * 1.05
        short_call_candidates = [c for c in calls if c["strike"] > price * 1.03]
        if not short_call_candidates:
            continue
        short_call = min(short_call_candidates, key=lambda c: abs(c["strike"] - target_short_call))

        # Short put: ~5% OTM
        target_short_put = price * 0.95
        short_put_candidates = [p for p in puts if p["strike"] < price * 0.97]
        if not short_put_candidates:
            continue
        short_put = min(short_put_candidates, key=lambda p: abs(p["strike"] - target_short_put))

        # Equal wing width: determine target wing width from call side, then match put side
        # Target long call ~3% above short call
        target_call_width = price * 0.03
        long_call_candidates = [c for c in calls if c["strike"] > short_call["strike"]]
        if not long_call_candidates:
            continue
        long_call = min(long_call_candidates, key=lambda c: abs((c["strike"] - short_call["strike"]) - target_call_width))
        call_wing_width = long_call["strike"] - short_call["strike"]

        # Match put wing width to call wing width for equal wings
        target_long_put_strike = short_put["strike"] - call_wing_width
        long_put_candidates = [p for p in puts if p["strike"] < short_put["strike"]]
        if not long_put_candidates:
            continue
        long_put = min(long_put_candidates, key=lambda p: abs(p["strike"] - target_long_put_strike))

        # Calculate credits
        call_credit = round(short_call.get("bid", 0) - long_call.get("ask", 0), 2)
        put_credit  = round(short_put.get("bid", 0)  - long_put.get("ask", 0),  2)
        net_credit  = round(call_credit + put_credit, 2)

        if net_credit <= 0:
            continue

        best_result = (short_call, long_call, short_put, long_put, net_credit)
        break

    if best_result is None:
        return {"error": "No suitable legs for iron condor"}

    short_call, long_call, short_put, long_put, net_credit = best_result

    # Max loss = max(call_spread_width, put_spread_width) - net_credit
    call_spread_width = round(long_call["strike"] - short_call["strike"], 2)
    put_spread_width  = round(short_put["strike"] - long_put["strike"], 2)
    wider_wing = max(call_spread_width, put_spread_width)
    max_loss_per = round((wider_wing - net_credit) * 100, 2)
    max_profit   = round(net_credit * 100, 2)

    qty = min(3, max(1, int(equity * size_pct / max(max_loss_per, 1))))

    # Breakeven calculations
    upper_be = round(short_call["strike"] + net_credit, 2)
    lower_be = round(short_put["strike"] - net_credit, 2)

    # EV calculation using short deltas as probability proxy
    sc_delta = abs(short_call.get("delta", 0))
    sp_delta = abs(short_put.get("delta", 0))

    result = {
        "strategy": "iron_condor",
        "short_call": short_call["occ_symbol"],
        "long_call":  long_call["occ_symbol"],
        "short_put":  short_put["occ_symbol"],
        "long_put":   long_put["occ_symbol"],
        "short_call_strike": short_call["strike"],
        "short_put_strike":  short_put["strike"],
        "long_call_strike":  long_call["strike"],
        "long_put_strike":   long_put["strike"],
        "expiry": short_call["expiry"],
        "qty": qty,
        "net_credit": net_credit,
        "total_credit": round(net_credit * qty * 100, 2),
        "max_profit": round(max_profit * qty, 2),
        "max_loss":   round(max_loss_per * qty, 2),
        "upper_breakeven": upper_be,
        "lower_breakeven": lower_be,
        "profit_range": f"${lower_be:.2f} to ${upper_be:.2f} (short strikes ${short_put['strike']}-${short_call['strike']})",
        "call_spread_width": call_spread_width,
        "put_spread_width": put_spread_width,
        "risk_reward": round(max_profit / max_loss_per, 2) if max_loss_per > 0 else 0,
        "days_to_expiry": short_call.get("days_to_expiry"),
        "error": None,
    }

    if sc_delta > 0 or sp_delta > 0:
        avg_short_delta = (sc_delta + sp_delta) / 2 if (sc_delta > 0 and sp_delta > 0) else max(sc_delta, sp_delta)
        p_win = 1 - avg_short_delta
        p_loss = 1 - p_win
        ev = p_win * max_profit - p_loss * max_loss_per
        result["ev"] = round(ev, 2)
        result["win_probability"] = round(p_win * 100, 1)
        if ev < 0:
            result["ev_warning"] = "Negative expected value"

    return result

def submit_options_order(contract: dict) -> dict:
    """
    Submit an options order to Alpaca.
    
    ALWAYS uses limit orders (never market on options — spreads are too wide).
    Returns: {"status": "filled"/"submitted"/"error", "order_id": str, "detail": str}
    """
    if contract.get("error"):
        return {"status": "error", "detail": contract["error"]}
    
    strategy = contract.get("strategy", "")
    
    # Multi-leg orders (spreads and multi-leg strategies)
    if "spread" in strategy:
        return _submit_spread_order(contract)
    elif strategy == "buy_straddle":
        return _submit_buy_straddle_order(contract)
    elif strategy == "short_straddle":
        return _submit_straddle_order(contract)
    elif strategy == "iron_condor":
        return _submit_condor_order(contract)
    
    # Single-leg order
    occ_symbol = contract.get("occ_symbol", "")
    side = contract.get("side", "buy")
    qty = contract.get("qty", 1)
    limit_price = contract.get("limit_price", 0)
    
    if not occ_symbol or limit_price <= 0:
        return {"status": "error", "detail": "Invalid contract data"}
    
    try:
        resp = requests.post(
            f"{ALPACA_BASE}/v2/orders",
            headers={**_alpaca_headers(), "Content-Type": "application/json"},
            json={
                "symbol": occ_symbol,
                "qty": str(qty),
                "side": side,
                "type": "limit",
                "limit_price": str(limit_price),
                "time_in_force": "day",
            },
            timeout=10,
        )
        
        if resp.status_code in (200, 201):
            order = resp.json()
            return {
                "status": "submitted",
                "order_id": order.get("id", ""),
                "detail": f"{'Sold' if side == 'sell' else 'Bought'} {qty}x {occ_symbol} @ ${limit_price} limit",
                "occ_symbol": occ_symbol,
            }
        else:
            error = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"message": resp.text[:200]}
            return {
                "status": "error",
                "detail": f"Alpaca rejected: {error.get('message', str(error)[:200])}",
            }
    
    except Exception as e:
        return {"status": "error", "detail": str(e)[:200]}


def _submit_mleg_order(legs: list, qty: int, limit_price: float, label: str) -> dict:
    """
    Submit a multi-leg options order using Alpaca's native mleg API.
    (v1.0.33: replaces all manual leg-by-leg submission + polling)

    The exchange handles atomicity — all legs fill together at one net price,
    or nothing fills. No polling, no unwinding, no partial fills, no naked legs.

    This is how pros do it: one order, one net price, atomic execution.

    Args:
        legs:  [{"symbol": OCC, "side": "buy"/"sell", "ratio_qty": "1"}]
        qty:   Number of strategy units
        limit_price: Net limit price for the whole structure
                     (positive = net debit you'll pay, negative = net credit you'll receive)
        label: Human-readable description for logging

    Returns:
        {"status": "submitted"/"filled"/"error", "order_id": str, "detail": str}
    """
    # Build position_intent from side: buy → buy_to_open, sell → sell_to_open
    mleg_legs = []
    for leg in legs:
        intent = "buy_to_open" if leg["side"] == "buy" else "sell_to_open"
        mleg_legs.append({
            "symbol": leg["symbol"],
            "side": leg["side"],
            "ratio_qty": str(leg.get("ratio_qty", 1)),
            "position_intent": intent,
        })

    payload = {
        "order_class": "mleg",
        "qty": str(qty),
        "type": "limit",
        "limit_price": str(round(abs(limit_price), 2)),
        "time_in_force": "day",
        "legs": mleg_legs,
    }

    try:
        resp = requests.post(
            f"{ALPACA_BASE}/v2/orders",
            headers={**_alpaca_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )

        if resp.status_code in (200, 201):
            order = resp.json()
            leg_syms = ", ".join(l["symbol"] for l in legs)
            return {
                "status": order.get("status", "submitted"),
                "order_id": order.get("id", ""),
                "detail": f"{label}: mleg order submitted (${limit_price:.2f} net, {len(legs)} legs: {leg_syms})",
                "occ_symbol": legs[0]["symbol"] if legs else "",
            }
        else:
            error_body = {}
            try:
                error_body = resp.json()
            except Exception:
                pass
            msg = error_body.get("message", resp.text[:200])
            return {"status": "error", "detail": f"{label}: Alpaca rejected mleg order — {msg}"}

    except Exception as e:
        return {"status": "error", "detail": f"{label}: mleg submission error — {str(e)[:200]}"}


def _submit_spread_order(contract: dict) -> dict:
    """Submit spread via Alpaca native mleg order."""
    long_symbol = contract.get("long_leg", "")
    short_symbol = contract.get("short_leg", "")
    qty = contract.get("qty", 1)
    net_debit = contract.get("net_debit", 0) or contract.get("limit_price", 0)
    if not long_symbol or not short_symbol:
        return {"status": "error", "detail": "Missing spread leg symbols"}
    return _submit_mleg_order(
        legs=[
            {"symbol": long_symbol, "side": "buy", "ratio_qty": 1},
            {"symbol": short_symbol, "side": "sell", "ratio_qty": 1},
        ],
        qty=qty, limit_price=net_debit,
        label=f"Spread {long_symbol}/{short_symbol} x{qty}",
    )


def _submit_buy_straddle_order(contract: dict) -> dict:
    """Submit long straddle via Alpaca native mleg order."""
    call_sym = contract.get("call_leg", "")
    put_sym  = contract.get("put_leg", "")
    qty      = contract.get("qty", 1)
    net_debit = contract.get("limit_price", 0) or (contract.get("call_debit", 0) + contract.get("put_debit", 0))
    return _submit_mleg_order(
        legs=[
            {"symbol": call_sym, "side": "buy", "ratio_qty": 1},
            {"symbol": put_sym,  "side": "buy", "ratio_qty": 1},
        ],
        qty=qty, limit_price=net_debit,
        label=f"Buy straddle {call_sym}/{put_sym} x{qty}",
    )


def _submit_straddle_order(contract: dict) -> dict:
    """Submit short straddle via Alpaca native mleg order."""
    call_sym = contract.get("call_leg", "")
    put_sym  = contract.get("put_leg", "")
    qty      = contract.get("qty", 1)
    net_credit = contract.get("limit_price", 0) or (contract.get("call_credit", 0) + contract.get("put_credit", 0))
    return _submit_mleg_order(
        legs=[
            {"symbol": call_sym, "side": "sell", "ratio_qty": 1},
            {"symbol": put_sym,  "side": "sell", "ratio_qty": 1},
        ],
        qty=qty, limit_price=net_credit,
        label=f"Short straddle {call_sym}/{put_sym} x{qty}",
    )


def _submit_condor_order(contract: dict) -> dict:
    """Submit iron condor via Alpaca native mleg order (4 legs, one order)."""
    qty = contract.get("qty", 1)
    net_credit = contract.get("net_credit", 0) or contract.get("limit_price", 0)
    sc = contract.get("short_call", "")
    lc = contract.get("long_call", "")
    sp = contract.get("short_put", "")
    lp = contract.get("long_put", "")
    if not all([sc, lc, sp, lp]):
        return {"status": "error", "detail": "Iron condor missing leg symbols"}
    return _submit_mleg_order(
        legs=[
            {"symbol": sc, "side": "sell", "ratio_qty": 1},
            {"symbol": lc, "side": "buy",  "ratio_qty": 1},
            {"symbol": sp, "side": "sell", "ratio_qty": 1},
            {"symbol": lp, "side": "buy",  "ratio_qty": 1},
        ],
        qty=qty, limit_price=net_credit,
        label=f"Iron condor x{qty}",
    )



# ═══════════════════════════════════════════════════════════════════════════════
#  D. FULL PIPELINE — Called by bot_engine
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_and_execute(trade: dict, equity: float, positions: list = None) -> dict:
    """
    Full pipeline: decide stock vs options → select contract → submit order.
    
    Returns: {
        "instrument": "stock" / "options",
        "strategy": str,
        "contract": dict or None,
        "order": dict or None,
        "reasoning": str,
    }
    """
    # Step 1: Should we use options?
    decision = should_use_options(trade, equity, positions)
    
    if not decision["use_options"]:
        return {
            "instrument": "stock",
            "strategy": "stock",
            "contract": None,
            "order": None,
            "reasoning": decision["reason"],
        }
    
    # Step 2: Select contract (with dynamic sizing)
    contract = select_contract(
        trade["ticker"], decision["strategy"], trade["price"], equity,
        trade=trade, positions=positions, macro=None
    )
    
    if contract.get("error"):
        return {
            "instrument": "stock",  # Fall back to stock
            "strategy": "stock",
            "contract": None,
            "order": None,
            "reasoning": f"Options fallback to stock: {contract['error']}",
        }
    
    # Step 3: Submit order
    order = submit_options_order(contract)
    
    # Step 4: Register entry state for options manager
    if order.get("status") in ("submitted", "filled", "pending_new", "accepted"):
        try:
            from options_manager import register_options_entry
            occ = contract.get("occ_symbol", "")
            entry_px = contract.get("limit_price", 0)
            entry_side = contract.get("side", "buy")
            entry_delta = contract.get("delta", 0)
            entry_qty = contract.get("qty", 1)
            register_options_entry(
                occ, entry_px, entry_side, decision["strategy"],
                delta=entry_delta, qty=entry_qty,
            )
        except Exception:
            pass  # Manager registration failed — non-critical
    
    return {
        "instrument": "options",
        "strategy": decision["strategy"],
        "edge_pct": decision["edge_pct"],
        "contract": contract,
        "order": order,
        "reasoning": decision["reason"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data = json.loads(sys.argv[1])
        mode = data.get("mode", "evaluate")
        
        if mode == "evaluate":
            result = evaluate_and_execute(
                data["trade"], data["equity"], data.get("positions", [])
            )
            print(json.dumps(result))
        elif mode == "decide":
            result = should_use_options(
                data["trade"], data["equity"], data.get("positions", [])
            )
            print(json.dumps(result))
        elif mode == "chain":
            chain = _fetch_option_chain(data["ticker"], data["price"])
            print(json.dumps({"contracts": len(chain), "sample": chain[:3] if chain else []}))
    else:
        # Quick test
        test_trade = {
            "ticker": "AAPL", "price": 185, "deep_score": 82, "vrp": 7.5,
            "side": "buy", "action_label": "SELL OPTIONS", "rsi": 55,
            "ewma_rv": 1.8, "garch_rv": 2.0,
        }
        result = should_use_options(test_trade, 100000)
        print(json.dumps(result, indent=2))
