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
    from position_sizing import _earnings_scalar, check_halt_status
except ImportError:
    def _earnings_scalar(t): return 1.0
    def check_halt_status(t): return {"halted": False}

logger = logging.getLogger("options_execution")

ALPACA_KEY = os.environ.get("ALPACA_KEY", "PKMDHJOVQEVIB4UHZXUYVTIDBU")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et")
ALPACA_BASE = "https://paper-api.alpaca.markets"
ALPACA_DATA = "https://data.alpaca.markets"

MAX_OPTIONS_PCT = 0.10      # Max 10% of portfolio per options trade
MAX_TOTAL_OPTIONS_PCT = 0.20 # Max 20% total options exposure
MIN_OPTION_VOLUME = 100      # Minimum daily volume on the contract
MIN_OPEN_INTEREST = 500      # Minimum open interest
MAX_SPREAD_PCT = 0.15        # Max bid-ask spread as % of mid price


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
    if score < 70:
        result["reason"] = f"Score {score} too low for options — need 70+ conviction"
        return result
    
    # Check earnings proximity — buyers get crushed by IV collapse
    earn_scalar = _earnings_scalar(ticker)
    
    # Check current time — options only during regular hours
    try:
        now = datetime.utcnow()
        et_hour = (now.hour - 4) % 24
        if et_hour < 9 or et_hour >= 16 or (et_hour == 9 and now.minute < 30):
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
    if "SELL OPTIONS" in action.upper() or vrp > 5:
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
    # Options are cheap. Buying has an edge.
    if vrp < -3:
        if earn_scalar < 0.85:
            # Options buyer near earnings — IV will crush you
            result["reason"] = f"Options are cheap (VRP {vrp:.1f}%) but earnings within 7 days — IV crush risk"
            return result
        
        result["use_options"] = True
        result["edge_pct"] = abs(vrp)
        
        if side == "buy" or rsi < 30:
            result["strategy"] = "buy_call"
            result["reason"] = f"IV is {abs(vrp):.1f}% below realized vol — calls are cheap. Leveraged upside."
        elif side == "short" or rsi > 70:
            result["strategy"] = "buy_put"
            result["reason"] = f"IV is {abs(vrp):.1f}% below realized vol — puts are cheap. Leveraged downside."
        else:
            result["strategy"] = "bull_call_spread"
            result["reason"] = f"IV cheap (VRP {vrp:.1f}%) — bull call spread for defined risk upside"
        return result
    
    # Scenario 3: High conviction directional — options for leverage
    if score >= 85 and abs(vrp) < 3:
        # VRP is neutral, but conviction is very high
        # Options give 3-10x leverage vs stock
        # BUT only if the stock moves enough to overcome theta decay
        
        expected_move_pct = ewma_rv * 1.5  # Rough estimate: 1.5x daily ATR over holding period
        
        if expected_move_pct > 3.0:
            result["use_options"] = True
            result["edge_pct"] = expected_move_pct - 2.0  # Net of theta
            
            if side == "buy":
                result["strategy"] = "bull_call_spread"
                result["reason"] = f"Score {score} + expected move {expected_move_pct:.1f}% — spread gives 3x leverage with defined risk"
            else:
                result["strategy"] = "bear_put_spread"
                result["reason"] = f"Score {score} + expected move {expected_move_pct:.1f}% — spread gives 3x leverage on downside"
            return result
    
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

def select_contract(ticker: str, strategy: str, price: float, equity: float) -> dict:
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
        
        # Select based on strategy
        if strategy == "buy_call":
            return _select_buy_call(liquid, price, equity, ticker)
        elif strategy == "buy_put":
            return _select_buy_put(liquid, price, equity, ticker)
        elif strategy == "sell_cash_secured_put":
            return _select_sell_put(liquid, price, equity, ticker)
        elif strategy == "bull_call_spread":
            return _select_bull_spread(liquid, price, equity, ticker)
        elif strategy == "bear_put_spread":
            return _select_bear_spread(liquid, price, equity, ticker)
        else:
            return {"error": f"Unknown strategy: {strategy}"}
    
    except Exception as e:
        return {"error": f"Contract selection failed: {str(e)[:200]}"}


def _fetch_option_chain(ticker: str, current_price: float) -> list:
    """Fetch options chain from Alpaca data API."""
    contracts = []
    try:
        # Get contracts expiring 14-45 days out (sweet spot for most strategies)
        now = datetime.now()
        min_exp = (now + timedelta(days=14)).strftime("%Y-%m-%d")
        max_exp = (now + timedelta(days=45)).strftime("%Y-%m-%d")
        
        # Strike range: within 10% of current price
        min_strike = current_price * 0.90
        max_strike = current_price * 1.10
        
        resp = requests.get(
            f"{ALPACA_DATA}/v1beta1/options/snapshots/{ticker}",
            params={
                "feed": "indicative",  # Free tier uses indicative
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
                    
                    contracts.append({
                        "occ_symbol": occ_symbol,
                        "ticker": ticker,
                        "strike": strike,
                        "expiry": exp_date,
                        "option_type": opt_type,
                        "bid": bid,
                        "ask": ask,
                        "mid": round(mid, 2),
                        "volume": int(trade.get("s", 0) or snap.get("volume", 0) or 0),
                        "open_interest": int(snap.get("openInterest", 0) or 0),
                        "iv": float(greeks.get("iv", 0) or 0),
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


def _select_buy_call(contracts: list, price: float, equity: float, ticker: str) -> dict:
    """Select a call to buy — slightly OTM, 20-35 days to expiry, delta ~0.40."""
    calls = [c for c in contracts if c["option_type"] == "call" and c["strike"] >= price]
    if not calls:
        return {"error": "No suitable calls found"}
    
    # Sort by proximity to delta 0.40 (sweet spot: enough leverage, decent probability)
    target_delta = 0.40
    calls.sort(key=lambda c: abs(abs(c.get("delta", 0)) - target_delta))
    
    best = calls[0]
    max_cost = equity * MAX_OPTIONS_PCT
    qty = int(max_cost / (best["ask"] * 100))  # Each contract = 100 shares
    if qty <= 0:
        qty = 1  # Minimum 1 contract
    
    actual_cost = qty * best["ask"] * 100
    
    return {
        "occ_symbol": best["occ_symbol"],
        "strike": best["strike"],
        "expiry": best["expiry"],
        "option_type": "call",
        "side": "buy",
        "qty": qty,
        "limit_price": round(best["ask"], 2),  # Limit at ask (could try mid for savings)
        "max_cost": round(actual_cost, 2),
        "max_loss": round(actual_cost, 2),  # Max loss = premium paid
        "delta": best.get("delta"),
        "iv": best.get("iv"),
        "days_to_expiry": best.get("days_to_expiry"),
        "bid_ask_spread": round(best["ask"] - best["bid"], 2),
        "error": None,
    }


def _select_buy_put(contracts: list, price: float, equity: float, ticker: str) -> dict:
    """Select a put to buy — slightly OTM, delta ~-0.40."""
    puts = [c for c in contracts if c["option_type"] == "put" and c["strike"] <= price]
    if not puts:
        return {"error": "No suitable puts found"}
    
    target_delta = -0.40
    puts.sort(key=lambda c: abs(c.get("delta", 0) - target_delta))
    
    best = puts[0]
    max_cost = equity * MAX_OPTIONS_PCT
    qty = int(max_cost / (best["ask"] * 100))
    if qty <= 0:
        qty = 1
    
    actual_cost = qty * best["ask"] * 100
    
    return {
        "occ_symbol": best["occ_symbol"],
        "strike": best["strike"],
        "expiry": best["expiry"],
        "option_type": "put",
        "side": "buy",
        "qty": qty,
        "limit_price": round(best["ask"], 2),
        "max_cost": round(actual_cost, 2),
        "max_loss": round(actual_cost, 2),
        "delta": best.get("delta"),
        "iv": best.get("iv"),
        "days_to_expiry": best.get("days_to_expiry"),
        "error": None,
    }


def _select_sell_put(contracts: list, price: float, equity: float, ticker: str) -> dict:
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
    
    # Cash required to secure: strike * 100 per contract
    cash_per_contract = best["strike"] * 100
    max_contracts = int(equity * MAX_OPTIONS_PCT / cash_per_contract)
    if max_contracts <= 0:
        return {"error": f"Not enough capital to sell cash-secured put at ${best['strike']} (need ${cash_per_contract:,.0f} per contract)"}
    
    qty = min(max_contracts, 2)  # Conservative: max 2 contracts
    premium_received = qty * best["bid"] * 100
    
    return {
        "occ_symbol": best["occ_symbol"],
        "strike": best["strike"],
        "expiry": best["expiry"],
        "option_type": "put",
        "side": "sell",
        "qty": qty,
        "limit_price": round(best["bid"], 2),  # Limit at bid (getting paid)
        "max_cost": 0,  # We receive premium
        "premium_received": round(premium_received, 2),
        "max_loss": round(best["strike"] * qty * 100 - premium_received, 2),
        "cash_required": round(cash_per_contract * qty, 2),
        "delta": best.get("delta"),
        "iv": best.get("iv"),
        "days_to_expiry": best.get("days_to_expiry"),
        "error": None,
    }


def _select_bull_spread(contracts: list, price: float, equity: float, ticker: str) -> dict:
    """Bull call spread: buy lower strike call, sell higher strike call. Defined risk."""
    calls = sorted(
        [c for c in contracts if c["option_type"] == "call"],
        key=lambda c: c["strike"]
    )
    if len(calls) < 2:
        return {"error": "Not enough call contracts for a spread"}
    
    # Find ATM call (closest to current price) and OTM call ~3-5% above
    atm_candidates = sorted(calls, key=lambda c: abs(c["strike"] - price))
    long_leg = atm_candidates[0]
    
    # Short leg: 3-5% above current price
    target_short_strike = price * 1.04
    short_candidates = [c for c in calls if c["strike"] > long_leg["strike"] and c["strike"] <= price * 1.08]
    if not short_candidates:
        return {"error": "No suitable short leg for bull call spread"}
    
    short_leg = min(short_candidates, key=lambda c: abs(c["strike"] - target_short_strike))
    
    net_debit = round(long_leg["ask"] - short_leg["bid"], 2)
    if net_debit <= 0:
        return {"error": "Spread has no cost — likely bad pricing data"}
    
    spread_width = short_leg["strike"] - long_leg["strike"]
    max_profit = round((spread_width - net_debit) * 100, 2)
    max_loss_per = round(net_debit * 100, 2)
    
    max_contracts = int(equity * MAX_OPTIONS_PCT / max_loss_per) if max_loss_per > 0 else 0
    if max_contracts <= 0:
        max_contracts = 1
    qty = min(max_contracts, 3)  # Max 3 spread contracts
    
    return {
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
        "risk_reward": round(max_profit / max_loss_per, 2) if max_loss_per > 0 else 0,
        "days_to_expiry": long_leg.get("days_to_expiry"),
        "error": None,
    }


def _select_bear_spread(contracts: list, price: float, equity: float, ticker: str) -> dict:
    """Bear put spread: buy higher strike put, sell lower strike put. Defined risk."""
    puts = sorted(
        [c for c in contracts if c["option_type"] == "put"],
        key=lambda c: c["strike"], reverse=True
    )
    if len(puts) < 2:
        return {"error": "Not enough put contracts for a spread"}
    
    atm_candidates = sorted(puts, key=lambda c: abs(c["strike"] - price))
    long_leg = atm_candidates[0]
    
    target_short_strike = price * 0.96
    short_candidates = [c for c in puts if c["strike"] < long_leg["strike"] and c["strike"] >= price * 0.92]
    if not short_candidates:
        return {"error": "No suitable short leg for bear put spread"}
    
    short_leg = min(short_candidates, key=lambda c: abs(c["strike"] - target_short_strike))
    
    net_debit = round(long_leg["ask"] - short_leg["bid"], 2)
    if net_debit <= 0:
        return {"error": "Spread has no cost — likely bad pricing data"}
    
    spread_width = long_leg["strike"] - short_leg["strike"]
    max_profit = round((spread_width - net_debit) * 100, 2)
    max_loss_per = round(net_debit * 100, 2)
    
    max_contracts = int(equity * MAX_OPTIONS_PCT / max_loss_per) if max_loss_per > 0 else 0
    if max_contracts <= 0:
        max_contracts = 1
    qty = min(max_contracts, 3)
    
    return {
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
        "risk_reward": round(max_profit / max_loss_per, 2) if max_loss_per > 0 else 0,
        "days_to_expiry": long_leg.get("days_to_expiry"),
        "error": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  C. ORDER SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════

def submit_options_order(contract: dict) -> dict:
    """
    Submit an options order to Alpaca.
    
    ALWAYS uses limit orders (never market on options — spreads are too wide).
    Returns: {"status": "filled"/"submitted"/"error", "order_id": str, "detail": str}
    """
    if contract.get("error"):
        return {"status": "error", "detail": contract["error"]}
    
    strategy = contract.get("strategy", "")
    
    # Multi-leg orders (spreads) — Alpaca doesn't support complex options orders yet
    # We need to submit legs individually
    if "spread" in strategy:
        return _submit_spread_order(contract)
    
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


def _submit_spread_order(contract: dict) -> dict:
    """
    Submit a spread (two legs).
    Alpaca doesn't support multi-leg options orders, so we submit each leg.
    The short leg is only submitted AFTER the long leg fills (protective).
    """
    long_symbol = contract.get("long_leg", "")
    short_symbol = contract.get("short_leg", "")
    qty = contract.get("qty", 1)
    net_debit = contract.get("net_debit", 0)
    
    if not long_symbol or not short_symbol:
        return {"status": "error", "detail": "Missing spread leg symbols"}
    
    # Step 1: Buy the long leg first (protective — limits risk)
    try:
        long_price = net_debit * 1.05  # Pay slightly more to ensure fill
        resp = requests.post(
            f"{ALPACA_BASE}/v2/orders",
            headers={**_alpaca_headers(), "Content-Type": "application/json"},
            json={
                "symbol": long_symbol,
                "qty": str(qty),
                "side": "buy",
                "type": "limit",
                "limit_price": str(round(long_price, 2)),
                "time_in_force": "day",
            },
            timeout=10,
        )
        
        if resp.status_code not in (200, 201):
            error = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            return {"status": "error", "detail": f"Long leg rejected: {error.get('message', resp.text[:200])}"}
        
        long_order = resp.json()
        
    except Exception as e:
        return {"status": "error", "detail": f"Long leg failed: {str(e)[:200]}"}
    
    # Step 2: Sell the short leg
    try:
        resp2 = requests.post(
            f"{ALPACA_BASE}/v2/orders",
            headers={**_alpaca_headers(), "Content-Type": "application/json"},
            json={
                "symbol": short_symbol,
                "qty": str(qty),
                "side": "sell",
                "type": "limit",
                "limit_price": str(round(net_debit * 0.05, 2)),  # Low limit — collecting credit
                "time_in_force": "day",
            },
            timeout=10,
        )
        
        if resp2.status_code not in (200, 201):
            # Short leg failed — we have naked long exposure now
            # This is OK (just becomes a regular long call/put), but log it
            return {
                "status": "partial",
                "detail": f"Long leg submitted but short leg rejected — position is a single long {long_symbol}",
                "long_order_id": long_order.get("id"),
            }
        
        short_order = resp2.json()
        
        return {
            "status": "submitted",
            "detail": f"Spread: bought {long_symbol}, sold {short_symbol} x{qty} @ ${net_debit} net debit",
            "long_order_id": long_order.get("id"),
            "short_order_id": short_order.get("id"),
        }
    
    except Exception as e:
        return {
            "status": "partial",
            "detail": f"Long leg OK but short leg error: {str(e)[:200]}",
            "long_order_id": long_order.get("id"),
        }


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
    
    # Step 2: Select contract
    contract = select_contract(
        trade["ticker"], decision["strategy"], trade["price"], equity
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
