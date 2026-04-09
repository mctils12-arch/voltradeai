"""
VolTradeAI — Options Position Manager
======================================
Professional-grade options exit management. Runs every cycle alongside
manage_positions() to handle options-specific concerns that stock logic
cannot address:

  1. DTE-based exit     — Close/roll at 21 DTE (Tastytrade research: cuts
                          largest losses ~50%, avg losses ~60%)
  2. Profit target      — Close sold premium at 50% of max profit
  3. Greeks monitoring   — Track delta drift, gamma risk, theta decay
  4. Rolling logic       — Roll threatened sold options for credit
  5. Assignment risk     — Detect and act on ITM options near expiry
  6. Gamma risk          — Exit when gamma exceeds threshold near expiry

REFERENCES:
  - Tastytrade 21 DTE research: largest losses cut ~50%
  - Tastytrade 50% profit management: higher win rate, less variance
  - Colin Bennett "Trading Volatility" Ch. 6: Gamma risk near expiry
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

logger = logging.getLogger("options_manager")

ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_BASE = "https://paper-api.alpaca.markets"
ALPACA_DATA = "https://data.alpaca.markets"

# ── Configuration ────────────────────────────────────────────────────────────
DTE_EXIT_THRESHOLD = 21          # Close/roll at 21 DTE
DTE_CRITICAL = 5                 # Force close at 5 DTE (no roll attempt)
PROFIT_TARGET_PCT = 0.50         # Close sold premium at 50% of max profit
LOSS_LIMIT_MULTIPLIER = 2.0      # Close if loss exceeds 2x credit received
GAMMA_THRESHOLD = 0.08           # Exit if per-contract gamma exceeds this
DELTA_DRIFT_THRESHOLD = 0.25     # Alert/act if delta shifted >0.25 from entry
ASSIGNMENT_DELTA_THRESHOLD = 0.80  # ITM risk when |delta| > 0.80
ROLL_MIN_CREDIT = 0.05           # Minimum net credit for a roll to be worthwhile

# Persistent state file for options positions
OPTIONS_STATE_PATH = os.path.join(DATA_DIR, "voltrade_options_state.json")


def _alpaca_headers():
    key = ALPACA_KEY or os.environ.get("ALPACA_KEY", "")
    secret = ALPACA_SECRET or os.environ.get("ALPACA_SECRET", "")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _load_options_state() -> dict:
    """Load persistent options position state."""
    try:
        if os.path.exists(OPTIONS_STATE_PATH):
            with open(OPTIONS_STATE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_options_state(state: dict):
    """Save persistent options position state."""
    try:
        with open(OPTIONS_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save options state: {e}")


def _get_option_snapshot(occ_symbol: str) -> dict:
    """Fetch current snapshot for an options contract from Alpaca."""
    try:
        # Extract ticker from OCC symbol (everything before the date portion)
        # OCC format: AAPL260418C00250000
        # Find where digits start for the date
        ticker = ""
        for i, ch in enumerate(occ_symbol):
            if ch.isdigit():
                ticker = occ_symbol[:i]
                break
        if not ticker:
            return {}

        resp = requests.get(
            f"{ALPACA_DATA}/v1beta1/options/snapshots/{ticker}",
            params={"feed": "opra", "symbols": occ_symbol},
            headers=_alpaca_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            snap = data.get("snapshots", {}).get(occ_symbol, {})
            if snap:
                quote = snap.get("latestQuote", {})
                greeks = snap.get("greeks", {})
                return {
                    "bid": float(quote.get("bp", 0) or 0),
                    "ask": float(quote.get("ap", 0) or 0),
                    "mid": round((float(quote.get("bp", 0) or 0) + float(quote.get("ap", 0) or 0)) / 2, 2),
                    "delta": float(greeks.get("delta", 0) or 0),
                    "gamma": float(greeks.get("gamma", 0) or 0),
                    "theta": float(greeks.get("theta", 0) or 0),
                    "vega": float(greeks.get("vega", 0) or 0),
                    "iv": float(greeks.get("iv", 0) or 0),
                }
    except Exception as e:
        logger.warning(f"Snapshot fetch failed for {occ_symbol}: {e}")
    return {}


def _parse_occ_symbol(occ_symbol: str) -> dict:
    """
    Parse an OCC symbol into components.
    Format: AAPL260418C00250000
    Returns: {ticker, expiry_date, option_type, strike}
    """
    try:
        # Find where the date digits start
        ticker = ""
        for i, ch in enumerate(occ_symbol):
            if ch.isdigit():
                ticker = occ_symbol[:i]
                break
        if not ticker or len(occ_symbol) < len(ticker) + 15:
            return {}

        body = occ_symbol[len(ticker):]
        exp_str = "20" + body[:6]
        expiry_date = f"{exp_str[:4]}-{exp_str[4:6]}-{exp_str[6:8]}"
        option_type = "call" if body[6] == "C" else "put"
        strike = int(body[7:]) / 1000

        return {
            "ticker": ticker,
            "expiry_date": expiry_date,
            "option_type": option_type,
            "strike": strike,
        }
    except Exception:
        return {}


def _days_to_expiry(expiry_date: str) -> int:
    """Calculate days until expiry."""
    try:
        exp = datetime.strptime(expiry_date, "%Y-%m-%d")
        return (exp - datetime.now()).days
    except Exception:
        return 999


def _submit_close_order(occ_symbol: str, qty: int, side: str, limit_price: float) -> dict:
    """
    Submit a closing order for an options position.
    Side should be the CLOSING side (opposite of the opening side).
    Uses mid-price optimization: starts at mid, with fallback walk.
    """
    if limit_price <= 0:
        return {"status": "error", "detail": "Invalid limit price"}

    try:
        resp = requests.post(
            f"{ALPACA_BASE}/v2/orders",
            headers={**_alpaca_headers(), "Content-Type": "application/json"},
            json={
                "symbol": occ_symbol,
                "qty": str(qty),
                "side": side,
                "type": "limit",
                "limit_price": str(round(limit_price, 2)),
                "time_in_force": "day",
            },
            timeout=10,
        )
        if resp.status_code in (200, 201):
            order = resp.json()
            return {
                "status": "submitted",
                "order_id": order.get("id", ""),
                "detail": f"Close {side} {qty}x {occ_symbol} @ ${limit_price:.2f}",
            }
        else:
            err = resp.text[:200]
            return {"status": "error", "detail": f"Alpaca rejected: {err}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)[:200]}


def _cancel_order(order_id: str) -> bool:
    """Cancel an order by ID."""
    try:
        resp = requests.delete(
            f"{ALPACA_BASE}/v2/orders/{order_id}",
            headers=_alpaca_headers(),
            timeout=10,
        )
        return resp.status_code in (200, 204)
    except Exception:
        return False


def _attempt_roll(occ_symbol: str, qty: int, current_side: str,
                  parsed: dict, equity: float) -> dict:
    """
    Attempt to roll an options position to a later expiry for a net credit.

    Steps:
      1. Close current position
      2. Open new position at same delta, 30-45 DTE out
      3. Only proceed if we receive a net credit (for sold options)

    Returns: {rolled: bool, detail: str, new_symbol: str or None}
    """
    try:
        ticker = parsed["ticker"]
        option_type = parsed["option_type"]
        strike = parsed["strike"]
        current_expiry = parsed["expiry_date"]

        # Get current price of the option we're closing
        snap = _get_option_snapshot(occ_symbol)
        if not snap or snap.get("mid", 0) <= 0:
            return {"rolled": False, "detail": "Cannot get current option price for roll"}

        # Determine closing price (what we pay/receive to close)
        if current_side == "sell":
            # We sold this option — to close we buy it back
            close_price = snap["ask"]  # We pay ask to buy back
        else:
            # We bought this option — to close we sell it
            close_price = snap["bid"]  # We receive bid

        # Find new contract: same strike, 30-45 days further out
        now = datetime.now()
        new_min_exp = (now + timedelta(days=30)).strftime("%Y-%m-%d")
        new_max_exp = (now + timedelta(days=50)).strftime("%Y-%m-%d")

        resp = requests.get(
            f"{ALPACA_DATA}/v1beta1/options/snapshots/{ticker}",
            params={
                "feed": "opra",
                "expiration_date_gte": new_min_exp,
                "expiration_date_lte": new_max_exp,
                "strike_price_gte": str(strike - 1),
                "strike_price_lte": str(strike + 1),
                "limit": 20,
            },
            headers=_alpaca_headers(),
            timeout=10,
        )

        if resp.status_code != 200:
            return {"rolled": False, "detail": f"Chain fetch failed: {resp.status_code}"}

        candidates = resp.json().get("snapshots", {})
        if not candidates:
            return {"rolled": False, "detail": "No roll candidates found"}

        # Find the best new contract (same type, closest strike)
        best_new = None
        best_credit = -999
        for new_occ, new_snap in candidates.items():
            new_parsed = _parse_occ_symbol(new_occ)
            if not new_parsed:
                continue
            if new_parsed["option_type"] != option_type:
                continue

            new_quote = new_snap.get("latestQuote", {})
            new_bid = float(new_quote.get("bp", 0) or 0)
            new_ask = float(new_quote.get("ap", 0) or 0)

            if new_bid <= 0 or new_ask <= 0:
                continue

            if current_side == "sell":
                # Rolling a sold option: close (buy back) + open new sell
                # Net credit = new_bid (sell new) - close_price (buy back old)
                net_credit = new_bid - close_price
            else:
                # Rolling a bought option: close (sell old) + open new buy
                # Net cost = new_ask (buy new) - close_price (sell old)
                net_credit = close_price - new_ask

            if net_credit > best_credit:
                best_credit = net_credit
                best_new = {
                    "occ_symbol": new_occ,
                    "bid": new_bid,
                    "ask": new_ask,
                    "mid": round((new_bid + new_ask) / 2, 2),
                    "net_credit": round(net_credit, 2),
                    "parsed": new_parsed,
                }

        if not best_new:
            return {"rolled": False, "detail": "No valid roll candidates with acceptable pricing"}

        # For sold options, only roll if we get a net credit
        if current_side == "sell" and best_new["net_credit"] < ROLL_MIN_CREDIT:
            return {
                "rolled": False,
                "detail": f"Roll would cost ${abs(best_new['net_credit']):.2f} — not a credit, skipping",
            }

        # Execute the roll: close old, open new
        # Step 1: Close
        close_side = "buy" if current_side == "sell" else "sell"
        close_result = _submit_close_order(occ_symbol, qty, close_side, close_price)
        if close_result["status"] == "error":
            return {"rolled": False, "detail": f"Close leg failed: {close_result['detail']}"}

        # Step 2: Open new
        if current_side == "sell":
            new_limit = best_new["bid"]
            new_side = "sell"
        else:
            new_limit = best_new["ask"]
            new_side = "buy"

        open_result = _submit_close_order(best_new["occ_symbol"], qty, new_side, new_limit)

        if open_result["status"] == "error":
            return {
                "rolled": False,
                "detail": f"Closed old but new leg failed: {open_result['detail']} — position is flat",
                "closed_old": True,
            }

        return {
            "rolled": True,
            "detail": (f"Rolled {occ_symbol} → {best_new['occ_symbol']} "
                       f"for ${best_new['net_credit']:+.2f}/contract net credit"),
            "new_symbol": best_new["occ_symbol"],
            "net_credit": best_new["net_credit"],
        }

    except Exception as e:
        return {"rolled": False, "detail": f"Roll error: {str(e)[:200]}"}


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-LEG STRATEGY MANAGEMENT (v1.0.33)
# ═══════════════════════════════════════════════════════════════════════════════


def _close_strategy_mleg(legs: list, ticker: str) -> dict:
    """
    Close a multi-leg strategy using a single mleg order.
    Each leg is reversed: bought legs are sold_to_close, sold legs are bought_to_close.
    Falls back to individual leg closes if mleg fails.
    """
    mleg_legs = []
    qty = 0

    for pos in legs:
        occ = pos.get("symbol", "")
        pos_qty = abs(int(float(pos.get("qty", 0))))
        side = pos.get("side", "long")

        if pos_qty > qty:
            qty = pos_qty

        # Reverse the position
        close_side = "sell" if side == "long" else "buy"
        intent = "sell_to_close" if side == "long" else "buy_to_close"

        mleg_legs.append({
            "symbol": occ,
            "side": close_side,
            "ratio_qty": "1",
            "position_intent": intent,
        })

    if not mleg_legs:
        return {"status": "error", "detail": "No legs to close"}

    # Try mleg close first
    try:
        payload = {
            "order_class": "mleg",
            "qty": str(max(qty, 1)),
            "type": "market",  # Use market for closing to ensure execution
            "time_in_force": "day",
            "legs": mleg_legs,
        }

        resp = requests.post(
            f"{ALPACA_BASE}/v2/orders",
            headers={**_alpaca_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )

        if resp.status_code in (200, 201):
            order = resp.json()
            return {"status": "submitted", "order_id": order.get("id", ""), "method": "mleg", "detail": f"mleg close: {len(mleg_legs)} legs"}
    except Exception:
        pass

    # Fallback: close legs individually
    results = []
    for pos in legs:
        occ = pos.get("symbol", "")
        pos_qty = abs(int(float(pos.get("qty", 0))))
        side = pos.get("side", "long")
        close_side = "sell" if side == "long" else "buy"
        current = abs(float(pos.get("current_price", 0)))

        result = _submit_close_order(occ, pos_qty, close_side, current)
        results.append(result)

    return {"status": "submitted", "method": "individual_legs", "detail": f"Closed {len(results)} legs individually (mleg failed)", "leg_results": results}


def _manage_strategy_group(group: dict, equity: float, state: dict) -> list:
    """
    Manage a multi-leg strategy as ONE unit.

    Calculates combined P&L across all legs, then applies exit rules
    to the WHOLE strategy, not individual legs.

    Returns: list of actions taken
    """
    actions = []
    legs = group["legs"]
    strategy = group["strategy"]
    ticker = group["ticker"]

    if not legs:
        return actions

    # Calculate COMBINED P&L across all legs
    total_entry_cost = 0
    total_current_value = 0
    total_unrealized_pnl = 0
    min_dte = 999
    all_occ_symbols = []

    for pos in legs:
        occ = pos.get("symbol", "")
        all_occ_symbols.append(occ)
        qty = abs(int(float(pos.get("qty", 0))))
        side = pos.get("side", "long")
        entry = abs(float(pos.get("avg_entry_price", 0)))
        current = abs(float(pos.get("current_price", 0)))
        pnl = float(pos.get("unrealized_pl", 0))

        total_entry_cost += entry * qty * 100
        total_current_value += current * qty * 100
        total_unrealized_pnl += pnl

        # Parse DTE from OCC symbol
        parsed = _parse_occ_symbol(occ)
        if parsed:
            dte = _days_to_expiry(parsed["expiry_date"])
            min_dte = min(min_dte, dte)

    # Combined P&L percentage
    if total_entry_cost > 0:
        combined_pnl_pct = total_unrealized_pnl / total_entry_cost
    else:
        combined_pnl_pct = 0

    # ── Exit Rule 1: CRITICAL DTE — Close entire strategy ─────────
    if min_dte <= DTE_CRITICAL:
        result = _close_strategy_mleg(legs, ticker)
        actions.append({
            "action": "CLOSE_STRATEGY",
            "ticker": ticker,
            "strategy": strategy,
            "legs": all_occ_symbols,
            "reason": f"CRITICAL DTE: {min_dte} days — closing entire {strategy}",
            "type": "strategy_dte_critical",
            "combined_pnl": round(total_unrealized_pnl, 2),
            "order": result,
        })
        # Clean state
        for occ in all_occ_symbols:
            state.pop(occ, None)
        return actions

    # ── Exit Rule 2: PROFIT TARGET (50% for credit strategies) ────
    is_credit_strategy = strategy in ("short_straddle", "iron_condor", "csp_normal_market")
    is_debit_strategy = strategy in ("buy_straddle", "bull_call_spread", "bear_put_spread")

    if is_credit_strategy and total_entry_cost > 0:
        # For credit strategies: profit = credit received - cost to close
        profit_pct = -combined_pnl_pct  # Flip sign for credit strategies
        if profit_pct >= PROFIT_TARGET_PCT:
            result = _close_strategy_mleg(legs, ticker)
            actions.append({
                "action": "CLOSE_STRATEGY",
                "ticker": ticker,
                "strategy": strategy,
                "legs": all_occ_symbols,
                "reason": f"PROFIT TARGET: {profit_pct:.0%} of max profit — closing {strategy}",
                "type": "strategy_profit_target",
                "combined_pnl": round(total_unrealized_pnl, 2),
                "order": result,
            })
            for occ in all_occ_symbols:
                state.pop(occ, None)
            return actions

    # ── Exit Rule 3: LOSS LIMIT ───────────────────────────────────
    if is_credit_strategy and total_entry_cost > 0:
        if combined_pnl_pct <= -LOSS_LIMIT_MULTIPLIER:  # Lost 2x credit
            result = _close_strategy_mleg(legs, ticker)
            actions.append({
                "action": "CLOSE_STRATEGY",
                "ticker": ticker,
                "strategy": strategy,
                "legs": all_occ_symbols,
                "reason": f"LOSS LIMIT: {combined_pnl_pct:.0%} loss on {strategy} — cutting",
                "type": "strategy_loss_limit",
                "combined_pnl": round(total_unrealized_pnl, 2),
                "order": result,
            })
            for occ in all_occ_symbols:
                state.pop(occ, None)
            return actions

    if is_debit_strategy and total_entry_cost > 0:
        if combined_pnl_pct <= -0.50:  # Lost 50% of debit
            result = _close_strategy_mleg(legs, ticker)
            actions.append({
                "action": "CLOSE_STRATEGY",
                "ticker": ticker,
                "strategy": strategy,
                "legs": all_occ_symbols,
                "reason": f"LOSS LIMIT: {combined_pnl_pct:.0%} loss on {strategy} — cutting debit position",
                "type": "strategy_loss_limit",
                "combined_pnl": round(total_unrealized_pnl, 2),
                "order": result,
            })
            for occ in all_occ_symbols:
                state.pop(occ, None)
            return actions

    # ── Exit Rule 4: 21 DTE Management ────────────────────────────
    if min_dte <= DTE_EXIT_THRESHOLD:
        result = _close_strategy_mleg(legs, ticker)
        actions.append({
            "action": "CLOSE_STRATEGY",
            "ticker": ticker,
            "strategy": strategy,
            "legs": all_occ_symbols,
            "reason": f"21 DTE MANAGEMENT: {min_dte} DTE — closing {strategy}",
            "type": "strategy_dte_exit",
            "combined_pnl": round(total_unrealized_pnl, 2),
            "order": result,
        })
        for occ in all_occ_symbols:
            state.pop(occ, None)
        return actions

    # ── Exit Rule 5: SABR Edge Evaporation ────────────────────────
    try:
        from vol_surface import get_surface_score
        surface = get_surface_score(ticker)
        vrp = surface.get("vrp", {}).get("vrp_20d", 0)

        # If we're selling premium but VRP flipped negative (options now cheap)
        if is_credit_strategy and vrp < -0.03:
            result = _close_strategy_mleg(legs, ticker)
            actions.append({
                "action": "CLOSE_STRATEGY",
                "ticker": ticker,
                "strategy": strategy,
                "legs": all_occ_symbols,
                "reason": f"EDGE EVAPORATED: VRP flipped to {vrp*100:.1f}% — premium selling edge gone",
                "type": "sabr_edge_exit",
                "combined_pnl": round(total_unrealized_pnl, 2),
                "vrp": round(vrp * 100, 1),
                "order": result,
            })
            for occ in all_occ_symbols:
                state.pop(occ, None)
            return actions

        # If we're buying premium but VRP flipped positive (options now expensive)
        if is_debit_strategy and vrp > 0.05:
            # Don't auto-close, but flag the edge deterioration
            actions.append({
                "action": "WARNING",
                "ticker": ticker,
                "strategy": strategy,
                "reason": f"EDGE WARNING: VRP at +{vrp*100:.1f}% — options expensive, debit position edge weakening",
                "type": "sabr_edge_warning",
                "vrp": round(vrp * 100, 1),
            })
    except Exception:
        pass  # SABR is advisory

    return actions


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN: manage_options_positions()
# ═══════════════════════════════════════════════════════════════════════════════

def manage_options_positions(equity: float = 100000) -> dict:
    """
    Professional options position management. Called every scan cycle.

    Checks every open options position for:
      1. DTE exit (21 DTE threshold)
      2. Profit target (50% of max profit for sold premium)
      3. Loss limit (2x credit received)
      4. Gamma risk (accelerating near expiry)
      5. Delta drift (position moved significantly from entry)
      6. Assignment risk (deep ITM near expiry)

    Returns: {
        "actions": [...],           — list of close/roll actions taken
        "positions_checked": int,
        "options_state": dict,      — updated state for all positions
    }
    """
    actions = []
    state = _load_options_state()

    # Fetch all positions from Alpaca
    try:
        resp = requests.get(
            f"{ALPACA_BASE}/v2/positions",
            headers=_alpaca_headers(),
            timeout=10,
        )
        if resp.status_code != 200:
            return {"actions": [], "error": f"Position fetch failed: {resp.status_code}"}
        all_positions = resp.json()
    except Exception as e:
        return {"actions": [], "error": str(e)[:200]}

    # Filter to options positions only (OCC symbols are >10 chars)
    options_positions = [
        p for p in all_positions
        if (len(p.get("symbol", "")) > 10 or
            p.get("asset_class", "") == "option")
    ]

    if not options_positions:
        # Clean stale state entries
        if state:
            _save_options_state({})
        return {"actions": [], "positions_checked": 0, "options_state": {}}

    positions_checked = 0

    # ── Group multi-leg positions by strategy ─────────────────────────
    # Read options state to identify which legs belong together
    options_meta = {}
    try:
        meta_path = os.path.join(DATA_DIR, 'voltrade_options_state.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                options_meta = json.load(f)
    except Exception:
        pass

    # Group positions by underlying ticker + strategy
    # All legs of the same strategy on the same ticker are managed as ONE unit
    strategy_groups = {}  # {"KMI_buy_straddle": [pos1, pos2], ...}
    standalone_positions = []  # Positions without a multi-leg group

    for pos in options_positions:
        occ = pos.get("symbol", "")
        meta = options_meta.get(occ, {})
        strategy = meta.get("strategy", "")
        ticker = meta.get("ticker", "")

        if strategy in ("buy_straddle", "short_straddle", "iron_condor",
                         "bull_call_spread", "bear_put_spread"):
            key = f"{ticker}_{strategy}"
            if key not in strategy_groups:
                strategy_groups[key] = {"legs": [], "strategy": strategy, "ticker": ticker, "setup": meta.get("setup", "")}
            strategy_groups[key]["legs"].append(pos)
        else:
            standalone_positions.append(pos)

    # ── Phase 1: Manage multi-leg strategy groups ─────────────────────
    for group_key, group in strategy_groups.items():
        group_actions = _manage_strategy_group(group, equity, state)
        actions.extend(group_actions)
        positions_checked += len(group["legs"])

    # Remove grouped positions from consideration in Phase 2
    grouped_symbols = set()
    for group in strategy_groups.values():
        for leg in group["legs"]:
            grouped_symbols.add(leg.get("symbol", ""))

    # ── Phase 2: Manage standalone positions ──────────────────────────
    for pos in options_positions:
        occ_symbol = pos.get("symbol", "")
        if occ_symbol in grouped_symbols:
            continue  # Already managed as part of a strategy group

        qty = abs(int(float(pos.get("qty", 0))))
        side = pos.get("side", "long")  # "long" = we bought, "short" = we sold
        entry_price = abs(float(pos.get("avg_entry_price", 0)))
        current_price = abs(float(pos.get("current_price", 0)))
        market_value = abs(float(pos.get("market_value", 0)))
        unrealized_pnl = float(pos.get("unrealized_pl", 0))

        if qty <= 0:
            continue

        positions_checked += 1

        # Parse OCC symbol
        parsed = _parse_occ_symbol(occ_symbol)
        if not parsed:
            continue

        dte = _days_to_expiry(parsed["expiry_date"])

        # Get current Greeks
        snap = _get_option_snapshot(occ_symbol)
        current_delta = snap.get("delta", 0)
        current_gamma = snap.get("gamma", 0)
        current_theta = snap.get("theta", 0)

        # Load or initialize state for this position
        pos_state = state.get(occ_symbol, {})
        if not pos_state:
            pos_state = {
                "entry_price": entry_price,
                "entry_delta": current_delta,
                "entry_date": time.strftime("%Y-%m-%d"),
                "initial_credit": entry_price if side == "short" else 0,
                "max_profit_target": entry_price * PROFIT_TARGET_PCT if side == "short" else 0,
                "highest_value": current_price,
                "strategy": "unknown",
                "side": side,
                "qty": qty,
            }
        else:
            # Update high water mark
            pos_state["highest_value"] = max(
                pos_state.get("highest_value", 0),
                current_price
            )

        state[occ_symbol] = pos_state

        # ── Check 1: CRITICAL DTE — Force close ─────────────────────────
        if dte <= DTE_CRITICAL:
            close_side = "sell" if side == "long" else "buy"
            # Use mid price for closing
            limit_px = snap.get("mid", current_price) if snap else current_price
            if limit_px <= 0:
                limit_px = current_price
            result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
            actions.append({
                "action": "CLOSE",
                "ticker": parsed["ticker"],
                "occ_symbol": occ_symbol,
                "reason": f"CRITICAL DTE: {dte} days to expiry — force close (< {DTE_CRITICAL} DTE)",
                "type": "dte_critical",
                "dte": dte,
                "pnl": round(unrealized_pnl, 2),
                "order": result,
            })
            # Remove from state
            state.pop(occ_symbol, None)
            continue

        # ── Check 2: Assignment Risk ─────────────────────────────────────
        if (side == "short" and
            abs(current_delta) > ASSIGNMENT_DELTA_THRESHOLD and
                dte <= 10):
            # Deep ITM sold option near expiry — high assignment risk
            # Try to roll first
            roll_result = _attempt_roll(occ_symbol, qty, "sell", parsed, equity)
            if roll_result.get("rolled"):
                actions.append({
                    "action": "ROLL",
                    "ticker": parsed["ticker"],
                    "occ_symbol": occ_symbol,
                    "reason": (f"ASSIGNMENT RISK: |delta|={abs(current_delta):.2f} > "
                               f"{ASSIGNMENT_DELTA_THRESHOLD}, {dte} DTE — rolled for credit"),
                    "type": "assignment_roll",
                    "dte": dte,
                    "delta": current_delta,
                    "roll_detail": roll_result["detail"],
                })
                # Update state to new symbol
                new_sym = roll_result.get("new_symbol")
                if new_sym:
                    state[new_sym] = {**pos_state, "entry_date": time.strftime("%Y-%m-%d")}
                state.pop(occ_symbol, None)
                continue
            else:
                # Roll failed — force close
                close_side = "buy"  # Buy back the sold option
                limit_px = snap.get("ask", current_price) if snap else current_price
                result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                actions.append({
                    "action": "CLOSE",
                    "ticker": parsed["ticker"],
                    "occ_symbol": occ_symbol,
                    "reason": (f"ASSIGNMENT RISK: |delta|={abs(current_delta):.2f}, "
                               f"{dte} DTE — roll failed ({roll_result['detail']}), closing"),
                    "type": "assignment_close",
                    "dte": dte,
                    "delta": current_delta,
                    "order": result,
                })
                state.pop(occ_symbol, None)
                continue

        # ── Check 3: DTE Exit Threshold (21 DTE) ────────────────────────
        if dte <= DTE_EXIT_THRESHOLD:
            if side == "short":
                # Sold option at 21 DTE — try to roll
                roll_result = _attempt_roll(occ_symbol, qty, "sell", parsed, equity)
                if roll_result.get("rolled"):
                    actions.append({
                        "action": "ROLL",
                        "ticker": parsed["ticker"],
                        "occ_symbol": occ_symbol,
                        "reason": f"21 DTE MANAGEMENT: {dte} DTE — rolled to later expiry for credit",
                        "type": "dte_roll",
                        "dte": dte,
                        "roll_detail": roll_result["detail"],
                    })
                    new_sym = roll_result.get("new_symbol")
                    if new_sym:
                        state[new_sym] = {**pos_state, "entry_date": time.strftime("%Y-%m-%d")}
                    state.pop(occ_symbol, None)
                    continue
                else:
                    # Roll failed — close
                    close_side = "buy"
                    limit_px = snap.get("ask", current_price) if snap else current_price
                    result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                    actions.append({
                        "action": "CLOSE",
                        "ticker": parsed["ticker"],
                        "occ_symbol": occ_symbol,
                        "reason": f"21 DTE EXIT: {dte} DTE — roll failed, closing ({roll_result['detail']})",
                        "type": "dte_close",
                        "dte": dte,
                        "order": result,
                    })
                    state.pop(occ_symbol, None)
                    continue
            else:
                # Bought option at 21 DTE — close to avoid theta decay acceleration
                close_side = "sell"
                limit_px = snap.get("bid", current_price) if snap else current_price
                if limit_px <= 0:
                    limit_px = current_price
                result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                actions.append({
                    "action": "CLOSE",
                    "ticker": parsed["ticker"],
                    "occ_symbol": occ_symbol,
                    "reason": f"21 DTE EXIT: {dte} DTE — closing bought option (theta acceleration zone)",
                    "type": "dte_close_bought",
                    "dte": dte,
                    "order": result,
                })
                state.pop(occ_symbol, None)
                continue

        # ── Check 4: Profit Target (50% for sold premium) ───────────────
        if side == "short":
            initial_credit = pos_state.get("initial_credit", entry_price)
            if initial_credit > 0 and current_price > 0:
                # For sold options: profit = credit received - current cost to close
                profit_pct = (initial_credit - current_price) / initial_credit
                if profit_pct >= PROFIT_TARGET_PCT:
                    # Hit 50% profit target — close and redeploy capital
                    close_side = "buy"
                    limit_px = snap.get("mid", current_price) if snap else current_price
                    if limit_px <= 0:
                        limit_px = current_price
                    result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                    actions.append({
                        "action": "CLOSE",
                        "ticker": parsed["ticker"],
                        "occ_symbol": occ_symbol,
                        "reason": (f"PROFIT TARGET: {profit_pct:.0%} of max profit reached "
                                   f"(target: {PROFIT_TARGET_PCT:.0%}) — closing to redeploy capital"),
                        "type": "profit_target",
                        "profit_pct": round(profit_pct * 100, 1),
                        "credit_received": initial_credit,
                        "current_cost": current_price,
                        "order": result,
                    })
                    state.pop(occ_symbol, None)
                    continue

        # ── Check 5: Loss Limit (2x credit for sold, 100% for bought) ──
        if side == "short":
            initial_credit = pos_state.get("initial_credit", entry_price)
            if initial_credit > 0 and current_price > 0:
                loss_ratio = current_price / initial_credit
                if loss_ratio >= LOSS_LIMIT_MULTIPLIER:
                    close_side = "buy"
                    limit_px = snap.get("ask", current_price) if snap else current_price
                    result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                    actions.append({
                        "action": "CLOSE",
                        "ticker": parsed["ticker"],
                        "occ_symbol": occ_symbol,
                        "reason": (f"LOSS LIMIT: Cost to close ${current_price:.2f} is "
                                   f"{loss_ratio:.1f}x the credit received ${initial_credit:.2f} "
                                   f"— capping loss at {LOSS_LIMIT_MULTIPLIER}x"),
                        "type": "loss_limit",
                        "loss_ratio": round(loss_ratio, 2),
                        "order": result,
                    })
                    state.pop(occ_symbol, None)
                    continue
        else:
            # Bought option: close if lost more than 50% of premium paid
            if entry_price > 0 and current_price > 0:
                loss_pct = (entry_price - current_price) / entry_price
                if loss_pct >= 0.50:
                    close_side = "sell"
                    limit_px = snap.get("bid", current_price) if snap else current_price
                    if limit_px <= 0:
                        limit_px = current_price
                    result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                    actions.append({
                        "action": "CLOSE",
                        "ticker": parsed["ticker"],
                        "occ_symbol": occ_symbol,
                        "reason": (f"LOSS LIMIT: Bought option down {loss_pct:.0%} from entry "
                                   f"(${entry_price:.2f} → ${current_price:.2f}) — cutting loss"),
                        "type": "bought_loss_limit",
                        "loss_pct": round(loss_pct * 100, 1),
                        "order": result,
                    })
                    state.pop(occ_symbol, None)
                    continue

        # ── Check 6: Gamma Risk ──────────────────────────────────────────
        if abs(current_gamma) > GAMMA_THRESHOLD and dte <= 30:
            close_side = "sell" if side == "long" else "buy"
            limit_px = snap.get("mid", current_price) if snap else current_price
            if limit_px <= 0:
                limit_px = current_price
            result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
            actions.append({
                "action": "CLOSE",
                "ticker": parsed["ticker"],
                "occ_symbol": occ_symbol,
                "reason": (f"GAMMA RISK: gamma={current_gamma:.3f} exceeds {GAMMA_THRESHOLD} "
                           f"with {dte} DTE — position too sensitive to price moves"),
                "type": "gamma_risk",
                "gamma": current_gamma,
                "dte": dte,
                "order": result,
            })
            state.pop(occ_symbol, None)
            continue

        # ── Check 7: Delta Drift ─────────────────────────────────────────
        entry_delta = pos_state.get("entry_delta", 0)
        if entry_delta != 0 and current_delta != 0:
            delta_shift = abs(current_delta - entry_delta)
            if delta_shift > DELTA_DRIFT_THRESHOLD:
                # Don't auto-close, but log as a warning action
                actions.append({
                    "action": "WARNING",
                    "ticker": parsed["ticker"],
                    "occ_symbol": occ_symbol,
                    "reason": (f"DELTA DRIFT: delta moved {entry_delta:.2f} → {current_delta:.2f} "
                               f"(shift: {delta_shift:.2f} > {DELTA_DRIFT_THRESHOLD})"),
                    "type": "delta_drift",
                    "entry_delta": entry_delta,
                    "current_delta": current_delta,
                    "delta_shift": round(delta_shift, 3),
                })

        # Update state with current Greeks
        pos_state["current_delta"] = current_delta
        pos_state["current_gamma"] = current_gamma
        pos_state["current_theta"] = current_theta
        pos_state["current_price"] = current_price
        pos_state["dte"] = dte
        pos_state["last_checked"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state[occ_symbol] = pos_state

    # Clean stale state entries (positions no longer held)
    held_symbols = {p.get("symbol", "") for p in options_positions}
    for old_sym in list(state.keys()):
        if old_sym not in held_symbols:
            del state[old_sym]

    _save_options_state(state)

    return {
        "actions": actions,
        "positions_checked": positions_checked,
        "options_state": state,
    }


def register_options_entry(occ_symbol: str, entry_price: float, side: str,
                           strategy: str, delta: float = 0, qty: int = 1):
    """
    Called when a new options position is opened. Records the entry state
    so the manager can track profit targets, delta drift, etc.
    """
    state = _load_options_state()
    state[occ_symbol] = {
        "entry_price": entry_price,
        "entry_delta": delta,
        "entry_date": time.strftime("%Y-%m-%d"),
        "initial_credit": entry_price if side == "sell" else 0,
        "max_profit_target": entry_price * PROFIT_TARGET_PCT if side == "sell" else 0,
        "highest_value": entry_price,
        "strategy": strategy,
        "side": "short" if side == "sell" else "long",
        "qty": qty,
    }
    _save_options_state(state)
