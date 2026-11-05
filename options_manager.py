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
import re
import json
import time
import logging
import requests

# [REPAIR 2026-07-28] KNOWN BROKEN #25 follow-up: this module was one
# of the call sites still hardcoding the opra feed param after the
# entitlement died (~07-23); alpaca_feed.options_feed() probes the
# entitlement and falls back to the free "indicative" feed, restoring
# automatically if the subscription returns. Same wiring as
# options_execution.py (v1.0.498).
import alpaca_feed
# BROKER CLOCK ORDER GATE 2026-11-05: no order while Alpaca's clock says closed
import broker_clock
from datetime import datetime, timedelta

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

logger = logging.getLogger("options_manager")

ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # FIX: was hardcoded — broke paper/live switching
ALPACA_DATA = "https://data.alpaca.markets"

# ── Configuration ────────────────────────────────────────────────────────────
DTE_EXIT_THRESHOLD = 21          # Close/roll at 21 DTE
DTE_CRITICAL = 5                 # Force close at 5 DTE (no roll attempt)
PROFIT_TARGET_PCT = 0.50         # Close sold premium at 50% of max profit
LOSS_LIMIT_MULTIPLIER = 2.0      # Close if loss exceeds 2x credit received
IC_MAX_LOSS_EXIT_PCT = 0.50      # Close iron condor at 50% of max loss (v1.0.34)
GAMMA_THRESHOLD = 0.08           # Exit if per-contract gamma exceeds this
DELTA_DRIFT_THRESHOLD = 0.25     # Alert/act if delta shifted >0.25 from entry
ASSIGNMENT_DELTA_THRESHOLD = 0.80  # ITM risk when |delta| > 0.80
ROLL_MIN_CREDIT = 0.05           # Minimum net credit for a roll to be worthwhile
MIN_HOLD_MINUTES = 60            # Minimum hold time before ANY exit (except DTE critical)

# Persistent state file for options positions
OPTIONS_STATE_PATH = os.path.join(DATA_DIR, "voltrade_options_state.json")

# KNOWN BROKEN #12(c) (research/open_questions.md): multi-leg strategies
# (spreads/straddles/condors) are managed as ONE combined unit by
# _manage_strategy_group() and closed via a single mleg order — there is no
# clean way to attribute that combined fill back to one options_manager
# ticker-keyed trade_feedback record without the per-leg collision problem
# described in that item (multiple legs share one underlying ticker, and
# _find_entry_record() matches by ticker alone). Standalone single-leg
# positions (CSP, covered call, naked long calls/puts — the strategies NOT
# in this set) have no such ambiguity and are wired into trade_feedback
# below. Multi-leg attribution stays a deliberate follow-up, not silently
# dropped — this is the SAME list manage_options_positions() already uses
# to decide grouping, named here so the two can never drift apart.
MULTI_LEG_STRATEGIES = (
    "buy_straddle", "short_straddle", "iron_condor",
    "bull_call_spread", "bear_put_spread",
    "bull_put_credit_spread", "qqq_iron_condor",
)

_CODE_VERSION_CACHE = None


def _code_version() -> str:
    """Read the app's package.json version once and cache it, for
    trade_feedback's code_version attribution field (PROMOTION RULES #4).
    options_manager.py runs inside the Python daemon/subprocess, which has
    no access to server/bot.ts's pkgVersion — package.json is the shared
    source of truth both sides already read from independently. Falls back
    to the same literal ml_model_v2.track_fill() itself falls back to for
    any caller that can't determine a version, so an unreadable file never
    breaks feedback recording, only its version attribution."""
    global _CODE_VERSION_CACHE
    if _CODE_VERSION_CACHE is not None:
        return _CODE_VERSION_CACHE
    try:
        pkg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "package.json")
        with open(pkg_path) as f:
            _CODE_VERSION_CACHE = str(json.load(f).get("version") or "1.0.34")
    except Exception:
        _CODE_VERSION_CACHE = "1.0.34"
    return _CODE_VERSION_CACHE


def _record_options_exit_feedback(ticker: str, side: str, qty: int, entry_price: float,
                                   exit_price: float, unrealized_pnl: float,
                                   exit_reason: str, entry_timestamp: str = None,
                                   order_result: dict = None) -> None:
    """Record a standalone single-leg options CLOSE into trade_feedback so
    it becomes a real win/open/loss/flat record instead of the permanent
    silence KNOWN BROKEN #12(c) documented — options_manager.py's exits
    previously recorded nothing at all. pnl_pct is derived from Alpaca's own
    live unrealized_pl (already real, quote-based dollar P&L — this is NOT
    a new synthetic pricing model, just plumbing an already-accurate number
    into the existing feedback pipeline) against the position's notional
    cost basis (entry_price * qty * 100, the standard option contract
    multiplier). entry_features is deliberately {} — no ML feature snapshot
    is captured at options entry today, so these records are visible in
    /api/diag/ml's outcome breakdown but correctly excluded from ML
    training by ml_model_v2's own entry_features schema gate (honest, not
    fabricated). Best-effort: any failure is logged and swallowed, never
    raised — this must not affect position management.

    [RULE-REVIEW] 2026-10-27 (ORPHAN EXITS, audit §7.2.4): `order_result` is
    the _submit_close_order() return value. A close Alpaca REJECTED is not an
    exit: the position stays open, its state entry is popped anyway, and the
    next cycle re-submits — each rejected attempt used to write one more
    exit record, and every one after the first found no open entry and
    became a permanent orphan_exit. Only an accepted close records now.
    Callers that pass no order_result (tests, legacy) keep the old behavior."""
    if order_result is not None and str(order_result.get("status", "")) != "submitted":
        logger.info(f"options exit feedback skipped for {ticker} ({exit_reason}): "
                    f"close order not accepted ({str(order_result.get('detail', ''))[:120]})")
        return
    try:
        from ml_model_v2 import track_fill
        cost_basis = abs(float(entry_price or 0)) * qty * 100
        pnl_pct = (float(unrealized_pnl or 0) / cost_basis * 100) if cost_basis > 0 else 0.0
        days_held = 0
        if entry_timestamp:
            try:
                days_held = max(0, (datetime.now() - datetime.fromisoformat(entry_timestamp)).days)
            except (ValueError, TypeError) as e:
                logger.warning(f"Bad entry_timestamp {entry_timestamp!r} for {ticker}: {e}")
        now_iso = datetime.now().isoformat()
        track_fill({
            "ticker": ticker,
            "side": "buy" if side == "short" else "sell",  # closing side reverses the opening side
            "qty": qty,
            "expected_price": exit_price,
            "fill_price": exit_price,
            "exit_reason": exit_reason,
            "exit_context": {
                "pnl_pct": round(pnl_pct, 3),
                "exit_reason": exit_reason,
                "days_held": days_held,
            },
            "time_filled": now_iso,
            "code_version": _code_version(),
        })
    except Exception as e:
        logger.warning(f"track_fill exit record failed for {ticker}: {e}")


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
            params={"feed": alpaca_feed.options_feed(), "symbols": occ_symbol},
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
                    # IV FIELD REPAIR 2026-08-06: impliedVolatility is
                    # top-level on Alpaca snapshots, not inside greeks
                    # (options_execution.py:626 pattern).
                    "iv": float(snap.get("impliedVolatility", 0) or greeks.get("iv", 0) or 0),
                }
    except Exception as e:
        logger.warning(f"Snapshot fetch failed for {occ_symbol}: {e}")
    return {}



def rolled_position_state(pos_state: dict, new_credit, now_date=None, now_iso=None) -> dict:
    """Fresh state for a ROLLED short leg (repair 2026-08-06, full-code-review
    finding, adversarially verified): the old leg's initial_credit /
    max_profit_target / highest_value must NOT survive the roll — profit
    targets would compute against a basis the new leg never had. Identity
    fields (ticker, qty, strategy tags) carry over; economics restart at the
    price the new leg actually sold for. Pure function, unit-tested."""
    fresh = {
        **pos_state,
        "entry_date": now_date or time.strftime("%Y-%m-%d"),
        "entry_timestamp": now_iso or datetime.now().isoformat(),
    }
    nc = float(new_credit or 0)
    if nc > 0:
        fresh["entry_price"] = nc
        fresh["initial_credit"] = nc
        fresh["max_profit_target"] = nc * PROFIT_TARGET_PCT
        fresh["highest_value"] = nc
    return fresh


def _parse_occ_symbol(occ_symbol: str) -> dict:
    """
    Parse an OCC symbol into components.
    Format: AAPL260418C00250000
    Returns: {ticker, expiry_date, option_type, strike}

    KNOWN BROKEN #31 scope note (options_execution.py/options_scanner.py
    carried the identical bug, fixed 2026-08-21 v1.0.760): parsing used to
    scan forward for the first digit to split ticker from date, which
    misreads an OCC adjusted-root's own digit (e.g. IONQ1 after a
    corporate action, instead of IONQ) as the date's start — shifting the
    whole suffix by one character and corrupting expiry/type/strike, or
    throwing outright. The date/type/strike suffix is fixed-width (15
    chars: 6-digit date + 1-char C/P + 8-digit strike) regardless of root
    length, so anchoring from the END avoids the ambiguity entirely; the
    ticker is simply whatever remains in front of it.
    """
    try:
        if len(occ_symbol) < 16:
            return {}

        ticker = occ_symbol[:-15]
        if not ticker:
            return {}
        date_digits = occ_symbol[-15:-9]
        type_char = occ_symbol[-9]
        strike_digits = occ_symbol[-8:]

        exp_str = "20" + date_digits
        expiry_date = f"{exp_str[:4]}-{exp_str[4:6]}-{exp_str[6:8]}"
        option_type = "call" if type_char == "C" else "put"
        strike = int(strike_digits) / 1000

        return {
            "ticker": ticker,
            "expiry_date": expiry_date,
            "option_type": option_type,
            "strike": strike,
        }
    except Exception:
        return {}


def _days_to_expiry(expiry_date: str) -> int:
    """Calculate days until expiry (timezone-aware, uses ET market close)."""
    try:
        from zoneinfo import ZoneInfo
        exp = datetime.strptime(expiry_date, "%Y-%m-%d").replace(
            tzinfo=ZoneInfo("America/New_York"))
        now_et = datetime.now(ZoneInfo("America/New_York"))
        return (exp - now_et).days
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
        broker_clock.check_order_allowed(occ_symbol, f"option close ({side})")
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
                "feed": alpaca_feed.options_feed(),
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
            # ROLL BASIS REPAIR 2026-08-06: the price the NEW leg was
            # submitted at — the fresh position's credit basis. Without it
            # the caller cloned the OLD leg's economics into the new state.
            "new_credit": new_limit,
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

    REPAIR 2026-11-02 (GOOGL Nov-27 iron condor, research/experiments.md):
    the fallback used to submit every leg at once at `current_price` and
    swallow the mleg rejection. Live, Alpaca refused the wing sells while the
    short legs were still open ("account not eligible to trade uncovered
    option contracts" / "insufficient options buying power for cash-secured
    put"), the after-hours marks were 0 ("Invalid limit price"), and nobody
    ever learned why the mleg was rejected. Now: the rejection text travels
    in `mleg_error`; the fallback buys back the SHORT legs first at the ask
    and defers the long wings until no short leg remains (then sells them at
    the bid); a leg without a usable quote is deferred, never errored. The
    caller (_request_group_close) keeps the group's state so the deferred
    legs are closed on a later tick. The mleg POST itself is unchanged.
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
    mleg_error = ""
    try:
        broker_clock.check_order_allowed(ticker, "strategy close (mleg)")
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
        mleg_error = f"HTTP {resp.status_code}: {str(getattr(resp, 'text', ''))[:160]}"
    except broker_clock.OrderBlocked as blocked:
        # BROKER CLOCK ORDER GATE 2026-11-05: while the broker's clock says
        # closed no leg can be sent either, so the whole close is reported as
        # deferred (the shape _request_group_close uses for a closed market)
        # and pursued on a later tick; the per-leg fallback is not attempted.
        return {"status": "deferred", "method": "none", "detail": str(blocked)[:200]}
    except Exception as e:
        mleg_error = str(e)[:160]
    # Fallback: close legs individually, in the order the broker accepts —
    # short legs first (buy_to_close at the ask); long wings only once no
    # short leg is left (sell_to_close at the bid).
    shorts = [pos for pos in legs if pos.get("side", "long") == "short"]
    longs = [pos for pos in legs if pos.get("side", "long") != "short"]
    batch = shorts if shorts else longs
    deferred = [pos.get("symbol", "") for pos in longs] if shorts else []
    results = []
    for pos in batch:
        occ = pos.get("symbol", "")
        pos_qty = abs(int(float(pos.get("qty", 0))))
        side = pos.get("side", "long")
        close_side = "sell" if side == "long" else "buy"
        snap = _get_option_snapshot(occ) or {}
        current = abs(float(pos.get("current_price", 0) or 0))
        px = float(snap.get("ask") or 0) if close_side == "buy" else float(snap.get("bid") or 0)
        if px <= 0:
            px = current
        if px <= 0:
            deferred.append(occ)
            results.append({"symbol": occ, "status": "deferred", "detail": "no usable quote this tick"})
            continue
        result = _submit_close_order(occ, pos_qty, close_side, px)
        result["symbol"] = occ
        results.append(result)
    submitted = sum(1 for r in results if r.get("status") == "submitted")
    status = "submitted" if submitted and not deferred else ("partial" if submitted else "error")
    return {
        "status": status, "method": "individual_legs", "mleg_error": mleg_error,
        "detail": (f"mleg rejected ({mleg_error}); {submitted}/{len(batch)} leg close(s) submitted"
                   + (f", {len(deferred)} deferred (shorts first / no quote)" if deferred else "")),
        "leg_results": results, "deferred": deferred,
    }


# ── REPAIR 2026-11-02: a requested strategy close is PURSUED until its legs are gone ──
# Before, every group exit submitted one close and popped the legs' state on
# the spot — a rejected/canceled close (the GOOGL condor: canceled day-limit
# fallbacks on 10-22, a deploy's "cancel all open orders") left four
# ungrouped legs that the single-leg rules then tried to close one at a
# time, which the broker partly refuses for a spread. Now the legs keep
# their strategy and carry a close request; each later tick re-submits when
# nothing is working at the broker, waits while something is, and never
# submits outside regular hours (DAY orders die at the next restart).
CLOSE_RETRY_SECONDS = 180      # a submitted close with nothing working at the broker is re-submitted after this
_CLOCK_CACHE = {"at": 0.0, "is_open": None, "et": None, "et_at": 0.0}
_CLOCK_TTL_SECONDS = 60


def _market_is_open() -> bool:
    """Alpaca /v2/clock, cached. A failed read counts as OPEN: a close is never
    withheld by a clock outage.

    BROKER CLOCK ORDER GATE 2026-11-05: also False while the shared order
    gate (broker_clock.orders_allowed) says closed. Every option close POST
    now passes that gate, so a close this function let through while the gate
    refuses it would be reported as a failed order (its state popped) instead
    of deferred. The gate follows the broker's own close time (a reading's
    next_close), and extrapolates from its last good reading when a read
    fails; with no usable reading it allows, so a clock outage still never
    withholds a close."""
    return _own_clock_is_open() and broker_clock.orders_allowed()[0]


def _own_clock_is_open() -> bool:
    """This module's own cached /v2/clock read (it also keeps the broker's
    timestamp for the opening window)."""
    now = time.time()
    if _CLOCK_CACHE["is_open"] is not None and now - _CLOCK_CACHE["at"] < _CLOCK_TTL_SECONDS:
        return bool(_CLOCK_CACHE["is_open"])
    is_open = True
    et = None
    try:
        resp = requests.get(f"{ALPACA_BASE}/v2/clock", headers=_alpaca_headers(), timeout=8)
        if resp.status_code == 200:
            body = resp.json()
            is_open = bool(body.get("is_open", True))
            et = _parse_clock_timestamp(body.get("timestamp"))   # the opening window reads it
    except Exception as e:
        logger.debug(f"clock read failed, assuming open: {e}")
    _CLOCK_CACHE.update({"at": now, "is_open": is_open, "et": et, "et_at": now})
    return is_open


def _open_orders_for(symbols) -> list:
    """Open orders at the broker for these contracts (a read; [] on failure so
    a failed read never blocks a close — the broker's own qty hold rejects a
    true duplicate, which the retry path then sees as 'working')."""
    syms = [s for s in symbols if s]
    if not syms:
        return []
    try:
        resp = requests.get(
            f"{ALPACA_BASE}/v2/orders",
            params={"status": "open", "symbols": ",".join(syms), "limit": 100},
            headers=_alpaca_headers(), timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.debug(f"open-orders read failed: {e}")
    return []


def _stamp_group_close(state: dict, occ_symbols: list, rule: str, reason: str, result: dict) -> None:
    """Record the close request on every leg WITHOUT dropping the leg's
    strategy/entry fields — the grouping must survive a failed close."""
    now_iso = datetime.now().isoformat()
    for occ in occ_symbols:
        st = state.get(occ)
        if not isinstance(st, dict):
            st = {}
            state[occ] = st
        st.setdefault("close_requested_at", now_iso)
        st["close_rule"] = rule
        st["close_reason"] = str(reason)[:200]
        st["close_attempts"] = int(st.get("close_attempts", 0) or 0) + (1 if result.get("status") != "deferred" else 0)
        st["close_last_at"] = now_iso
        st["close_last_result"] = {k: result.get(k) for k in ("status", "method", "order_id", "mleg_error", "detail", "deferred") if k in result}


def _request_group_close(legs: list, ticker: str, state: dict, rule: str, reason: str) -> dict:
    """The one path every strategy exit takes: submit the group close during
    regular hours (deferred otherwise), and stamp the request on the legs —
    the legs' state is never popped here; the end-of-run cleanup removes a
    leg's state only once the broker no longer holds it."""
    occs = [pos.get("symbol", "") for pos in legs]
    if not _market_is_open():
        result = {"status": "deferred", "method": "none", "detail": "market closed — close requested, submitted on the next regular-hours tick"}
    elif _in_opening_window():
        result = {"status": "deferred", "method": "none", "detail": "opening window — close requested, submitted after 09:45 ET"}
    else:
        result = _close_strategy_mleg(legs, ticker)
    _stamp_group_close(state, occs, rule, reason, result)
    return result


def _pursue_group_close(group: dict, state: dict) -> list:
    """A group whose legs carry a close request: wait while a close order is
    working at the broker (or one was just submitted), otherwise re-submit."""
    legs = group["legs"]
    ticker = group["ticker"]
    strategy = group["strategy"]
    occs = [pos.get("symbol", "") for pos in legs]
    first = next((state.get(o, {}) for o in occs if isinstance(state.get(o), dict) and state[o].get("close_requested_at")), {})
    rule = first.get("close_rule", "strategy_close")
    reason = first.get("close_reason", "")
    attempts = max(int((state.get(o) or {}).get("close_attempts", 0) or 0) for o in occs)
    last_at = max((str((state.get(o) or {}).get("close_last_at", "")) for o in occs), default="")
    working = _open_orders_for(occs)
    if working:
        return [{"action": "WAIT", "ticker": ticker, "strategy": strategy, "legs": occs, "type": "strategy_close_working",
                 "reason": f"{len(working)} close order(s) working at the broker for {strategy} — not resubmitting", "attempts": attempts}]
    try:
        since = (datetime.now() - datetime.fromisoformat(last_at)).total_seconds() if last_at else CLOSE_RETRY_SECONDS + 1
    except Exception:
        since = CLOSE_RETRY_SECONDS + 1
    if since < CLOSE_RETRY_SECONDS and (first.get("close_last_result") or {}).get("status") in ("submitted", "partial"):
        return [{"action": "WAIT", "ticker": ticker, "strategy": strategy, "legs": occs, "type": "strategy_close_settling",
                 "reason": f"close submitted {int(since)}s ago — giving it {CLOSE_RETRY_SECONDS}s before resubmitting", "attempts": attempts}]
    result = _request_group_close(legs, ticker, state, rule, reason)
    return [{"action": "CLOSE_STRATEGY", "ticker": ticker, "strategy": strategy, "legs": occs,
             "reason": f"{rule}: retry #{attempts + 1} — {reason}"[:220], "type": rule, "retry": attempts + 1, "order": result}]


def _max_loss_from_legs(legs: list, net_credit: float) -> float:
    """Max loss at expiry of a defined-risk CREDIT structure, from its legs:
    widest wing (strike difference) x 100 x qty - the net credit received —
    the formula the entry builders use (options_execution: max_loss_per x qty).

    REPAIR 2026-11-04: the iron-condor 50%-of-max-loss exit read max_loss only
    from the legs' state and did nothing when none was recorded. Only shapes
    the formula is exact for are priced: a credit vertical (one short and one
    long of one type, the long further out of the money) or an iron condor (a
    put and a call credit vertical whose short strikes do not cross), every
    leg the same quantity, root and expiry. Anything else returns 0, so the
    rule stays off as it was, rather than acting on a guess.
    """
    strikes = {"put": {"short": [], "long": []}, "call": {"short": [], "long": []}}
    qtys, series = set(), set()
    for pos in legs:
        parsed = _parse_occ_symbol(pos.get("symbol", ""))
        qty = abs(int(float(pos.get("qty", 0))))
        if not parsed or qty <= 0:
            return 0.0
        qtys.add(qty)
        series.add((parsed["ticker"], parsed["expiry_date"]))
        side = "short" if pos.get("side", "long") == "short" else "long"
        strikes[parsed["option_type"]][side].append(parsed["strike"])
    if len(qtys) != 1 or len(series) != 1:
        return 0.0
    widths = []
    for opt_type, sides in strikes.items():
        if not sides["short"] and not sides["long"]:
            continue
        if len(sides["short"]) != 1 or len(sides["long"]) != 1:
            return 0.0
        short_k, long_k = sides["short"][0], sides["long"][0]
        width = short_k - long_k if opt_type == "put" else long_k - short_k
        if width <= 0:
            return 0.0  # a debit vertical: its max loss is the debit, not this formula
        widths.append(width)
    if not widths:
        return 0.0
    if len(widths) == 2 and strikes["put"]["short"][0] > strikes["call"]["short"][0]:
        return 0.0  # short strikes cross: both sides can finish in the money
    max_loss = max(widths) * 100 * qtys.pop() - net_credit
    return round(max_loss, 2) if max_loss > 0 else 0.0


def _manage_strategy_group(group: dict, equity: float, state: dict, earnings: dict = None) -> list:
    """
    Manage a multi-leg strategy as ONE unit.

    Calculates combined P&L across all legs, then applies exit rules
    to the WHOLE strategy, not individual legs.

    `earnings` is manage_options_positions' _earnings_close_context() for the
    pass (None: the earnings close is not judged).

    Returns: list of actions taken
    """
    actions = []
    legs = group["legs"]
    strategy = group["strategy"]
    ticker = group["ticker"]

    if not legs:
        return actions

    # Calculate COMBINED P&L across all legs
    total_current_value = 0
    total_unrealized_pnl = 0
    # NET-CREDIT FIX 2026-05-03 (audit Finding #2):
    # For credit strategies (verticals, condors), max_profit = net credit
    # received = sum(short_premiums) - sum(long_premiums). The previous
    # denominator (gross entry cost summing both sides) made the 50% target
    # fire at 75% of max profit on 4-leg structures.
    net_credit = 0.0               # positive when we received cash at entry
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

        total_current_value += current * qty * 100
        total_unrealized_pnl += pnl

        # Net credit: short legs add, long legs subtract.
        # For pure credit structures (CSP, naked put, short straddle) all
        # legs are short, so net_credit equals the gross premium of the legs
        # (and for an all-long structure -net_credit does): no behavior
        # change there. For spreads/condors the long-leg debit nets against
        # short-leg credit, giving the true max profit.
        if side == "short":
            net_credit += entry * qty * 100
        else:
            net_credit -= entry * qty * 100

        # Parse DTE from OCC symbol
        parsed = _parse_occ_symbol(occ)
        if parsed:
            dte = _days_to_expiry(parsed["expiry_date"])
            min_dte = min(min_dte, dte)

    # ── CLOSING MODE (REPAIR 2026-11-02): a close already requested for this
    # group is pursued until the broker no longer holds the legs — no rule
    # below is re-judged, and the legs are never handed to the single-leg path.
    if any(isinstance(state.get(occ), dict) and state[occ].get("close_requested_at") for occ in all_occ_symbols):
        return _pursue_group_close(group, state)

    # ── Exit Rule 0: MINIMUM HOLD TIME (except DTE critical) ──────
    # Don't close positions within the first hour — the trade thesis
    # needs time to play out. Every straddle in Apr 3-10 was closed
    # within minutes, losing to the bid-ask spread every time.
    _min_hold_ok = True
    for occ in all_occ_symbols:
        _occ_state = state.get(occ, {})
        _entry_ts = _occ_state.get("entry_timestamp")
        if _entry_ts:
            try:
                _entry_dt = datetime.fromisoformat(_entry_ts)
                _held_minutes = (datetime.now() - _entry_dt).total_seconds() / 60
                if _held_minutes < MIN_HOLD_MINUTES:
                    _min_hold_ok = False
                    break
            except Exception:
                pass

    # ── Exit Rule 1: CRITICAL DTE — Force close (ignores hold time) ────
    if min_dte <= DTE_CRITICAL:
        result = _request_group_close(legs, ticker, state, "strategy_dte_critical", f"CRITICAL DTE: {min_dte} days — closing entire {strategy}")
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
        return actions

    # ── Exit Rule 1b: EARNINGS CLOSE (2026-11-05) — ignores hold time ─────
    # A group with a sold leg is closed whole on the last session before the
    # underlying reports, if the report falls before that leg's expiry (the
    # EARNINGS CLOSE block above manage_options_positions). A defined-risk
    # group opened FOR its report (_earnings_close_exempt) is left to the
    # other rules.
    for pos in legs:
        parsed_leg = _parse_occ_symbol(pos.get("symbol", "")) if pos.get("side", "long") == "short" else None
        earn = _earnings_close_due_for(earnings, parsed_leg["ticker"], _expiry_date(parsed_leg)) if parsed_leg else None
        if not earn:
            continue
        if _earnings_close_exempt(group):
            break
        reason = _earnings_close_reason(earnings, parsed_leg["ticker"], *earn, f"the whole {strategy}")
        result = _request_group_close(legs, ticker, state, "strategy_earnings_close", reason)
        actions.append({
            "action": "CLOSE_STRATEGY",
            "ticker": ticker or parsed_leg["ticker"],
            "strategy": strategy,
            "legs": all_occ_symbols,
            "reason": reason,
            "type": "strategy_earnings_close",
            "combined_pnl": round(total_unrealized_pnl, 2),
            "report_date": earn[0].isoformat(),
            "report_hour": earn[1],
            "close_session": earn[2].isoformat(),
            "order": result,
        })
        return actions

    # If minimum hold time not met, skip all other exit rules
    if not _min_hold_ok:
        return actions

    # ── Exit Rule 2: PROFIT TARGET (50% for credit strategies) ────
    #
    # P0-8 FIX: Expanded credit-strategy list so bull_put_credit_spread,
    # sell_cash_secured_put, covered_call, and qqq_iron_condor now honor
    # the profit target as well (previously fell into standalone path or
    # nothing at all).
    is_credit_strategy = strategy in (
        "short_straddle", "iron_condor", "csp_normal_market",
        "bull_put_credit_spread", "sell_cash_secured_put",
        "covered_call", "qqq_iron_condor",
    )
    is_debit_strategy = strategy in ("buy_straddle", "bull_call_spread", "bear_put_spread")

    if is_credit_strategy and net_credit > 0:
        # NET-CREDIT FIX 2026-05-03 (audit Finding #2):
        # For credit strategies, max_profit = net_credit received. The
        # legacy denominator (total_entry_cost = sum of |entry|*qty across
        # all legs) overcounted on spreads/condors because long legs were
        # added in instead of subtracted. With $1.00 short legs and $0.20
        # long legs on a 4-leg condor, entry_cost = $240 but max profit =
        # net credit = $160. The 50% gate was firing at $120 (75% of max).
        # Now using net_credit as denominator, the gate fires at $80 = 50%.
        profit_pct = total_unrealized_pnl / net_credit
        if profit_pct >= PROFIT_TARGET_PCT:
            result = _request_group_close(legs, ticker, state, "strategy_profit_target", f"PROFIT TARGET: {profit_pct:.0%} of max profit (net credit ${net_credit:.0f}) — closing {strategy}")
            actions.append({
                "action": "CLOSE_STRATEGY",
                "ticker": ticker,
                "strategy": strategy,
                "legs": all_occ_symbols,
                "reason": f"PROFIT TARGET: {profit_pct:.0%} of max profit (net credit ${net_credit:.0f}) — closing {strategy}",
                "type": "strategy_profit_target",
                "combined_pnl": round(total_unrealized_pnl, 2),
                "net_credit": round(net_credit, 2),
                "order": result,
            })
            return actions

    # ── Exit Rule 2b: IRON CONDOR 50%-OF-MAX-LOSS EARLY EXIT (v1.0.34) ──
    # For iron condors, we know the exact max loss (wing_width - credit).
    # If unrealized loss reaches 50% of that, close early. Rationale:
    #   - At 50% max loss, risk/reward of holding gets unfavorable
    #   - One earnings blowout shouldn't ride to full max loss
    #   - Tastylive research: managing losers at a defined threshold
    #     improves Sortino ratio by capping the left tail
    #   - Applies to ALL iron condors, not just earnings plays
    if strategy in ("iron_condor", "qqq_iron_condor") and total_unrealized_pnl < 0:
        # Try to get max_loss from state (set at entry via register_options_entry)
        group_max_loss = 0
        for occ in all_occ_symbols:
            occ_state = state.get(occ, {})
            ml = occ_state.get("max_loss", 0)
            if ml > 0:
                group_max_loss = max(group_max_loss, ml)
                break  # All legs share the same max_loss

        # REPAIR 2026-11-04: with no max_loss in any leg's state (live: the
        # GOOGL Nov-27 condor's state was re-initialised as "unknown") this
        # rule used to skip silently. The legs carry what it needs.
        max_loss_note = ""
        if group_max_loss <= 0:
            group_max_loss = _max_loss_from_legs(legs, net_credit)
            max_loss_note = " (from the legs; none recorded)"

        if group_max_loss > 0:
            loss_pct_of_max = abs(total_unrealized_pnl) / group_max_loss
            if loss_pct_of_max >= IC_MAX_LOSS_EXIT_PCT:
                reason = (
                    f"IC MAX-LOSS EXIT: Unrealized loss ${abs(total_unrealized_pnl):.2f} is "
                    f"{loss_pct_of_max:.0%} of max loss ${group_max_loss:.2f}{max_loss_note} — "
                    f"closing before full max loss (threshold: {IC_MAX_LOSS_EXIT_PCT:.0%})"
                )
                result = _request_group_close(legs, ticker, state, "strategy_ic_max_loss_exit", reason)
                actions.append({
                    "action": "CLOSE_STRATEGY",
                    "ticker": ticker,
                    "strategy": strategy,
                    "legs": all_occ_symbols,
                    "reason": reason,
                    "type": "strategy_ic_max_loss_exit",
                    "combined_pnl": round(total_unrealized_pnl, 2),
                    "max_loss": group_max_loss,
                    "loss_pct_of_max": round(loss_pct_of_max * 100, 1),
                    "order": result,
                })
                return actions

    # ── Exit Rule 3: LOSS LIMIT ───────────────────────────────────
    # REPAIR 2026-11-04: measured against the NET premium, as the comments
    # always said. Both rules used to divide the loss by the GROSS premium of
    # every leg (shorts and longs added together), several times the net on a
    # spread, which typically put the trigger beyond the structure's own max
    # loss (widest wing x 100 x qty - net credit; a debit vertical's is its
    # net debit) — the GOOGL Nov-27 condor: 2x gross $9,700 vs a $1,090 max
    # loss, so the rule could never fire. Thresholds unchanged. All legs
    # short (short straddle, a lone short put) or all long (long straddle):
    # net == gross, identical decisions. A credit structure that collected
    # more than a third of its widest wing still cannot lose 2x its credit;
    # on a condor the 50%-of-max-loss exit above is the stop that acts.
    if is_credit_strategy and net_credit > 0:
        pnl_of_credit = total_unrealized_pnl / net_credit
        if pnl_of_credit <= -LOSS_LIMIT_MULTIPLIER:  # Lost 2x credit
            reason = (f"LOSS LIMIT: loss ${-total_unrealized_pnl:.0f} is {-pnl_of_credit:.0%} of the net credit "
                      f"${net_credit:.0f} (limit {LOSS_LIMIT_MULTIPLIER:.0%}) on {strategy} — cutting")
            result = _request_group_close(legs, ticker, state, "strategy_loss_limit", reason)
            actions.append({
                "action": "CLOSE_STRATEGY",
                "ticker": ticker,
                "strategy": strategy,
                "legs": all_occ_symbols,
                "reason": reason,
                "type": "strategy_loss_limit",
                "combined_pnl": round(total_unrealized_pnl, 2),
                "order": result,
            })
            return actions

    net_debit = -net_credit        # positive when we paid cash at entry
    if is_debit_strategy and net_debit > 0:
        pnl_of_debit = total_unrealized_pnl / net_debit
        if pnl_of_debit <= -0.50:  # Lost 50% of debit
            reason = (f"LOSS LIMIT: loss ${-total_unrealized_pnl:.0f} is {-pnl_of_debit:.0%} of the net debit "
                      f"${net_debit:.0f} on {strategy} — cutting debit position")
            result = _request_group_close(legs, ticker, state, "strategy_loss_limit", reason)
            actions.append({
                "action": "CLOSE_STRATEGY",
                "ticker": ticker,
                "strategy": strategy,
                "legs": all_occ_symbols,
                "reason": reason,
                "type": "strategy_loss_limit",
                "combined_pnl": round(total_unrealized_pnl, 2),
                "order": result,
            })
            return actions

    # ── Exit Rule 4: 21 DTE Management ────────────────────────────
    if min_dte <= DTE_EXIT_THRESHOLD:
        result = _request_group_close(legs, ticker, state, "strategy_dte_exit", f"21 DTE MANAGEMENT: {min_dte} DTE — closing {strategy}")
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
        return actions

    # ── Exit Rule 5: SABR Edge Evaporation ────────────────────────
    try:
        from vol_surface import get_surface_score
        surface = get_surface_score(ticker)
        vrp = surface.get("vrp", {}).get("vrp_20d", 0)

        # If we're selling premium but VRP flipped negative (options now cheap)
        if is_credit_strategy and vrp < -0.03:
            result = _request_group_close(legs, ticker, state, "sabr_edge_exit", f"EDGE EVAPORATED: VRP flipped to {vrp*100:.1f}% — premium selling edge gone")
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

# HALT SCOPE (2026-10-27, audit N15/N9): a roll is close + SELL-TO-OPEN a new
# contract — the open half is a new entry. While a halt blocks entries the
# caller passes allow_rolls=False and every roll point takes its existing
# "roll failed -> close" branch instead, so the position is exited, never
# re-opened.
ROLL_SKIPPED_HALTED_DETAIL = "roll skipped: entries halted (exits only)"
# ENTRY SHADOW MODE (2026-10-28, entry_shadow.py): the same close-only path
# while the options sleeve is shadowed — the close stays live, the would-be
# re-open is logged by the caller (entry_shadow.roll_skip_entries).
ROLL_SKIPPED_SHADOW_DETAIL = "roll skipped: options entries in shadow mode (close only)"


def _roll_or_skip(allow_rolls: bool, occ_symbol: str, qty: int, current_side: str,
                  parsed: dict, equity: float, skip_detail: str = ROLL_SKIPPED_HALTED_DETAIL) -> dict:
    if not allow_rolls:
        return {"rolled": False, "detail": skip_detail}
    return _attempt_roll(occ_symbol, qty, current_side, parsed, equity)


# ── REPAIR 2026-11-04: a single-leg close waits for regular hours, like a group close ──
# Options trade only in regular hours, so a mark read outside them is not a
# price. 2026-10-29 07:31 ET (11:31:35Z): the sold-option LOSS LIMIT rule read
# XLF261127P00055500's pre-market mark ("Cost to close $5.00 is 6.9x the
# credit") and submitted a $5.00 DAY limit at the pre-market ask; it filled at
# the open at $1.61 against a fair value near $0.49. _request_group_close
# already deferred while the market was closed; the single-leg rules (Phase 2
# of manage_options_positions) submitted at any hour. Now every Phase-2 close,
# and every roll (a close plus a sell-to-open), is decided and submitted only
# while _market_is_open() says so (cached; a failed clock read counts as open).
# Outside regular hours the leg is left as it is: no order, no roll attempt,
# no exit feedback, its state kept (popping it would restart the hold gate at
# the open). The run reports a CLOSE whose order is {"status": "deferred"},
# the group path's shape, so the OPTIONS-EXIT audit row reads "order
# deferred". Unlike a group close nothing is persisted: the closed-market
# marks that tripped the rule are exactly what cannot be trusted, so the rule
# is judged again, on regular-hours marks, at the next regular-hours tick.
SINGLE_LEG_DEFERRED_DETAIL = ("market closed — no order; the rule is judged again on "
                              "regular-hours marks at the next regular-hours tick")

# ── OPENING WINDOW (2026-11-05) ─────────────────────────────────────────────
# No option exit is sent before 09:45 ET, single legs and groups alike.
# research/options_bleed_2026-11-04.md: 53 lots closed in the session's first
# 5 minutes lost $5,083. 29 of them, whose underlying moved <= 3%, lost
# $1,378 where the price model says +$266, with a median fill of 1.41x fair
# value (the XLF put: $1.61 against ~$0.49 fair on an up day). Option quotes
# are widest in the opening rotation. A rule that fires in the window is
# reported as deferred and judged again on the first tick after 09:45.
OPENING_WINDOW_END_ET = (9, 45)
OPENING_WINDOW_DEFERRED_DETAIL = ("opening window — no option exit before 09:45 ET (the widest spreads "
                                  "of the day); the rule is judged again after 09:45")

# ── LEFTOVER LONG LEG (2026-11-05) ─────────────────────────────────────────
# A bought option whose structure is gone is closed. Live:
# GOOGL261127P00300000 was the long put wing of an earnings iron condor whose
# other three legs were closed on 10-29. It sat alone for a week with no
# rule that could exit it: $4.70 -> $22.49 -> sold at $10.65 by the 21-DTE
# rule on 11-05.
# Evidence (research review, 2026-11-05):
# - Long options carry a negative expected return, the variance risk premium
#   (Coval & Shumway 2001; Bakshi & Kapadia 2003; Carr & Wu 2009).
# - That return is worst for short-dated out-of-the-money contracts. A
#   leftover 30-day put wing 10% OTM at 24% IV decays about 7% of its value
#   per day.
# - No peer-reviewed test supports profit targets or trailing stops on option
#   PREMIUM. Closing leftover wings is the defensible rule.
# A leftover here is a LONG single leg that is the only option position on its
# underlying (so a spread's long leg is never closed while its short partner
# is still open: that would leave the short naked), whose recorded strategy is
# not a deliberate standalone long. A lone wing whose state still records its
# multi-leg strategy is kept out of the group code so this rule reaches it.
STANDALONE_LONG_STRATEGIES = ("buy_call", "buy_put", "convexity_hedge")


def _in_opening_window(now_et=None) -> bool:
    """True from 09:30 to 09:45 ET (call it only when the market is open).
    The time is the BROKER's: Alpaca /v2/clock's own timestamp, cached by
    _own_clock_is_open() and advanced by the seconds since that read. A missing
    or unreadable timestamp counts as outside the window: a close is never
    withheld by a broken clock."""
    try:
        if now_et is None:
            stamp = _CLOCK_CACHE.get("et")
            if stamp is None:
                return False
            now_et = stamp + timedelta(seconds=max(0.0, time.time() - float(_CLOCK_CACHE.get("et_at") or time.time())))
        if now_et.tzinfo is not None:
            from zoneinfo import ZoneInfo
            now_et = now_et.astimezone(ZoneInfo("America/New_York"))
        hm = (now_et.hour, now_et.minute)
    except Exception as e:
        logger.debug(f"opening-window clock read failed, treating as outside: {e}")
        return False
    return (9, 30) <= hm < OPENING_WINDOW_END_ET


def _parse_clock_timestamp(raw):
    """Alpaca's clock timestamp ('2026-11-05T09:31:02.123456789-05:00', ET
    with its offset) as an aware datetime in that offset. Fractions beyond
    microseconds are cut; anything unreadable is None."""
    if not isinstance(raw, str) or not raw:
        return None
    cleaned = re.sub(r"(\.\d{6})\d+", r"\1", raw.replace("Z", "+00:00"))
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _defer_single_leg_close(actions: list, ticker: str, occ_symbol: str, rule: str,
                            reason: str, dte: int) -> bool:
    """Outside regular hours: report the rule as a deferred CLOSE and return
    True; the caller `continue`s, leaving the leg untouched. In regular hours
    (or on a failed clock read): append nothing and return False, so the rule
    rolls / submits exactly as before."""
    if _market_is_open():
        if not _in_opening_window():
            return False
        actions.append({
            "action": "CLOSE",
            "ticker": ticker,
            "occ_symbol": occ_symbol,
            "reason": f"{reason} — deferred: opening window (before 09:45 ET)",
            "type": rule,
            "dte": dte,
            "order": {"status": "deferred", "method": "none", "detail": OPENING_WINDOW_DEFERRED_DETAIL},
        })
        return True
    actions.append({
        "action": "CLOSE",
        "ticker": ticker,
        "occ_symbol": occ_symbol,
        "reason": f"{reason} — deferred: market closed",
        "type": rule,
        "dte": dte,
        "order": {"status": "deferred", "method": "none", "detail": SINGLE_LEG_DEFERRED_DETAIL},
    })
    return True


# ── EARNINGS CLOSE (2026-11-05) ─────────────────────────────────────────────
# Human directive, 2026-11-05: "Existing short options into earnings: add a
# rule that closes any already-open sold option before the underlying's
# earnings (last session before the report, after 9:45 ET), not just one that
# blocks new ones. Backfill a check against current positions on deploy."
# The entry gates refuse NEW short premium into a report (KNOWN BROKEN #45:
# tiered_strategy's T1 guard, scanner Setup 3's 14-day blackout). Nothing
# closed a sold option already open. research/options_bleed_2026-11-04.md: 9
# structures held through a report lost -$4,751. Live 2026-11-04: an IREN put
# was sold the day before IREN's report (a calendar bug, since fixed).
#
# The rule: every open SOLD option whose underlying reports between today and
# the option's expiry is closed on the CLOSE SESSION, the last regular session
# before the report. A standalone short leg is bought back at the ask (like the
# loss-limit and assignment closes); a strategy group with a short leg is
# closed whole (_request_group_close).
# - Close session: the report date itself for an "amc" (after the close)
#   report; the previous trading day for "bmo", "dmh" or an unannounced hour
#   (market_calendar.py's weekends and holidays).
# - Judged BEFORE the 60-minute hold gate, like CRITICAL DTE: an option sold
#   30 minutes before the close session ends must not ride into the report.
# - The usual deferrals apply: nothing is sent while the market is closed or
#   before 09:45 ET.
# - Still held after its close session (a close that failed, a process that was
#   down): closed at the first eligible tick, up to and including the report
#   date.
# Exempt: a multi-leg group whose recorded setup is in system_config
# OPTIONS_EARNINGS_CLOSE_EXEMPT_SETUPS ("earnings_iv_crush", Setup 1) and that is
# defined-risk (every sold contract covered by a bought one of the same type on
# the same underlying: a condor or credit spread with its wings). It was opened
# FOR the report; closing it first would only pay the spread twice. The rule
# targets a sold option held into a report by accident. A lone short leg is
# never exempt.
# Report dates: Finnhub's per-symbol calendar (options_scanner.
# _fetch_symbol_earnings_reports), looked up once per underlying per ET day. A
# failed lookup closes nothing on a guess: one WARNING per underlying per ET
# day, retried every tick. Without FINNHUB_KEY the rule cannot run (logged).
# Backfill: the first pass after a process start (a deploy, a daemon restart)
# lists every sold option's next report and close session in one
# "earnings_backfill" action (an exempt group's legs as "exempt_setup"); a leg
# already due is closed in that same pass.
EARNINGS_BACKFILL_TYPE = "earnings_backfill"
EARNINGS_LOOKUP_FAILED_TYPE = "earnings_lookup_failed"
EARNINGS_HOUR_WORDS = {"bmo": "before the open", "amc": "after the close",
                       "dmh": "during market hours", "": "at an unannounced hour"}
_EARNINGS_REPORT_CACHE: dict = {}   # underlying -> (ET day, window end, [(date, hour), ...]); good answers only
_EARNINGS_FAILED: dict = {}         # underlying -> ET day its own lookup last failed (it is asked last)
_EARNINGS_WARNED: dict = {}         # underlying -> ET day of its last lookup-failure WARNING
_EARNINGS_RUN = {"backfill_done": False, "unconfigured_logged": False}


def _now_et() -> datetime:
    """New York wall-clock time."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def _expiry_date(parsed: dict):
    """The parsed OCC symbol's expiry as a date, None if it is not a date."""
    try:
        return datetime.strptime(parsed["expiry_date"], "%Y-%m-%d").date()
    except (KeyError, TypeError, ValueError):
        return None


def earnings_close_session(report_date, hour: str):
    """The last regular session before a report on `report_date` (a date): the
    report date itself for an "amc" report on a trading day; otherwise ("bmo",
    "dmh", unannounced, or a report dated on a closed day) the previous trading
    day. market_calendar.py lists 2026 holidays only until December's year-add."""
    from market_calendar import is_market_holiday
    if str(hour or "").strip().lower() == "amc" and not is_market_holiday(report_date):
        return report_date
    session = report_date - timedelta(days=1)
    while is_market_holiday(session):
        session -= timedelta(days=1)
    return session


def earnings_close_due(today, report_date, hour: str) -> bool:
    """True from the close session through the report date (all dates)."""
    return earnings_close_session(report_date, hour) <= today <= report_date


def _earnings_close_enabled() -> bool:
    try:
        from system_config import BASE_CONFIG
        return bool(BASE_CONFIG.get("OPTIONS_EARNINGS_CLOSE_ENABLED", True))
    except Exception as e:
        logger.warning(f"system_config unreadable, the earnings close stays on: {e}")
        return True


def _earnings_close_exempt(group: dict) -> bool:
    """True for a strategy group opened FOR its report: its recorded setup is in
    OPTIONS_EARNINGS_CLOSE_EXEMPT_SETUPS and every sold contract is covered by a
    bought contract of the same type (call/put) on the same underlying. A group
    with an uncovered sold leg (a lone short leg, or an adopted extra short) is
    never exempt."""
    setup = str(group.get("setup") or "")
    try:
        from system_config import BASE_CONFIG
        exempt_setups = tuple(BASE_CONFIG.get("OPTIONS_EARNINGS_CLOSE_EXEMPT_SETUPS", ()) or ())
    except Exception as e:
        logger.warning(f"system_config unreadable, no earnings-close exemption: {e}")
        return False
    if not setup or setup not in exempt_setups:
        return False
    sold, bought = {}, {}      # (underlying, call/put) -> contracts
    for pos in group.get("legs") or []:
        parsed = _parse_occ_symbol(pos.get("symbol", ""))
        if not parsed:
            return False
        book = sold if pos.get("side", "long") == "short" else bought
        kind = (parsed["ticker"], parsed["option_type"])
        book[kind] = book.get(kind, 0) + abs(int(float(pos.get("qty", 0) or 0)))
    return bool(sold) and all(bought.get(kind, 0) >= n for kind, n in sold.items())


def _finnhub_configured() -> bool:
    try:
        import options_scanner
        key = str(getattr(options_scanner, "FINNHUB_KEY", "") or "")
    except Exception as e:
        logger.warning(f"options_scanner unavailable to the earnings close: {type(e).__name__}: {e}")
        return False
    return bool(key) and key != "YOUR_FINNHUB_KEY_HERE"


def _earnings_reports(root: str, today, horizon, fetch: bool = True):
    """(reports, how) for `root` inside [today, horizon]. how is "cached" (this
    ET day's answer covers the window), "fetched" (one Finnhub request), or,
    with reports None, "failed" or "skipped" (fetch=False and nothing cached).
    Failures are never cached, so the next tick asks again."""
    hit = _EARNINGS_REPORT_CACHE.get(root)
    if hit and hit[0] == today.isoformat() and hit[1] >= horizon:
        return hit[2], "cached"
    if not fetch:
        return None, "skipped"
    try:
        import options_scanner
        reports, ok = options_scanner._fetch_symbol_earnings_reports(root, today, horizon)
    except Exception as e:
        logger.warning(f"earnings report lookup for {root} unavailable: {type(e).__name__}: {e}")
        return None, "failed"
    if not ok or reports is None:
        return None, "failed"
    _EARNINGS_REPORT_CACHE[root] = (today.isoformat(), horizon, list(reports))
    return list(reports), "fetched"


def _earnings_close_context(options_positions: list, actions: list, strategy_groups: dict = None) -> dict:
    """This pass's earnings-close inputs: {"active", "today", "by_root":
    {underlying: {"ok", "how", "report"}}, "exempt": {occ: setup}}, where
    `report` is the underlying's next (date, hour) inside the latest expiry
    among its sold options (an exempt group's legs left out), or None. Appends
    the lookup-failure WARNINGs (once per underlying per ET day) and, on the
    first active pass since process start, the "earnings_backfill" summary."""
    ctx = {"active": False, "today": None, "by_root": {}, "exempt": {}}
    if not _earnings_close_enabled():
        return ctx
    if not _finnhub_configured():
        if not _EARNINGS_RUN["unconfigured_logged"]:
            _EARNINGS_RUN["unconfigured_logged"] = True
            logger.warning("EARNINGS CLOSE inactive: FINNHUB_KEY is not set, so no report date can be looked up")
        return ctx
    today = _now_et().date()
    day = today.isoformat()
    ctx.update(active=True, today=today)
    for group in (strategy_groups or {}).values():
        if _earnings_close_exempt(group):
            ctx["exempt"].update({pos.get("symbol", ""): str(group.get("setup") or "") for pos in group["legs"]})
    shorts, exempt_shorts = [], []      # (occ symbol, underlying, expiry) of every sold option
    for p in options_positions:
        parsed = _parse_occ_symbol(p.get("symbol", ""))
        expiry = _expiry_date(parsed) if parsed else None
        if expiry is None or p.get("side", "long") != "short" or abs(int(float(p.get("qty", 0) or 0))) <= 0:
            continue
        leg = (p.get("symbol", ""), parsed["ticker"], expiry)
        (exempt_shorts if leg[0] in ctx["exempt"] else shorts).append(leg)
    horizon = {}
    for _occ, root, expiry in shorts:
        horizon[root] = max(expiry, horizon.get(root, expiry))
    for root in [r for r, v in _EARNINGS_REPORT_CACHE.items() if v[0] != day]:
        _EARNINGS_REPORT_CACHE.pop(root, None)
    for memo in (_EARNINGS_FAILED, _EARNINGS_WARNED):
        for root in [r for r, d in memo.items() if d != day]:
            memo.pop(root, None)
    # After one failed request the pass sends no more (a Finnhub outage costs
    # one timeout per tick, not one per underlying); cached answers still count.
    # An underlying whose own lookup failed today is asked last, so one bad
    # symbol cannot starve the others.
    fetch = True
    for root in sorted(horizon, key=lambda r: (_EARNINGS_FAILED.get(r) == day, r)):
        reports, how = _earnings_reports(root, today, horizon[root], fetch=fetch)
        if how == "failed":
            fetch = False
            _EARNINGS_FAILED[root] = day
        elif reports is not None:
            _EARNINGS_FAILED.pop(root, None)
        nxt = next((r for r in reports if today <= r[0] <= horizon[root]), None) if reports is not None else None
        ctx["by_root"][root] = {"ok": reports is not None, "how": how, "report": nxt}
    for root, info in sorted(ctx["by_root"].items()):
        if info["ok"] or _EARNINGS_WARNED.get(root) == day:
            continue
        _EARNINGS_WARNED[root] = day
        legs = sorted(occ for occ, r, _e in shorts if r == root)
        why = ("the Finnhub lookup failed" if info["how"] == "failed"
               else "not looked up this tick: another lookup failed first")
        actions.append({
            "action": "WARNING", "ticker": root, "type": EARNINGS_LOOKUP_FAILED_TYPE, "legs": legs,
            "reason": (f"EARNINGS CLOSE: {root}'s next report date is unknown ({why}) — {len(legs)} sold "
                       f"option(s) not closed on a guess ({', '.join(legs)}); retried every tick, warned once "
                       f"per ET day"),
        })
    if not _EARNINGS_RUN["backfill_done"]:
        _EARNINGS_RUN["backfill_done"] = True
        if shorts or exempt_shorts:
            actions.append(_earnings_backfill_action(ctx, shorts + exempt_shorts))
    return ctx


def _earnings_backfill_action(ctx: dict, shorts: list) -> dict:
    """The deploy-time check: every sold option's next report inside its own
    life and its close session, in one INFO action. An exempt group's legs are
    listed as "exempt_setup" (not looked up: the rule never closes them)."""
    rows, parts = [], []
    for occ, root, expiry in sorted(shorts):
        info = ctx["by_root"].get(root) or {}
        report = info.get("report")
        row = {"occ_symbol": occ, "underlying": root, "expiry": expiry.isoformat(),
               "report_date": None, "report_hour": None, "close_session": None}
        if occ in ctx["exempt"]:
            row.update(status="exempt_setup", setup=ctx["exempt"][occ])
            parts.append(f"{occ}: exempt (setup {ctx['exempt'][occ]}, a defined-risk structure opened for its report)")
        elif not info.get("ok"):
            row["status"] = "lookup_failed"
            parts.append(f"{occ}: report date unknown (lookup failed; not closed on a guess, retried every tick)")
        elif report is None or report[0] > expiry:
            row["status"] = "none_before_expiry"
            parts.append(f"{occ}: no {root} report before expiry {expiry.isoformat()}")
        else:
            rdate, hour = report
            session = earnings_close_session(rdate, hour)
            due = earnings_close_due(ctx["today"], rdate, hour)
            row.update(report_date=rdate.isoformat(), report_hour=hour, close_session=session.isoformat(),
                       status="due" if due else "scheduled")
            parts.append(f"{occ}: {root} reports {rdate.isoformat()} {EARNINGS_HOUR_WORDS.get(hour, EARNINGS_HOUR_WORDS[''])}"
                         f", close session {session.isoformat()}{' (due now)' if due else ''}")
        rows.append(row)
    return {"action": "INFO", "type": EARNINGS_BACKFILL_TYPE, "legs": rows,
            "reason": (f"EARNINGS BACKFILL (first pass since start): {len(rows)} sold option(s) checked — "
                       + "; ".join(parts))}


def _earnings_close_due_for(ctx: dict, root: str, expiry):
    """(report date, hour, close session) when a sold option on `root` expiring
    on `expiry` (a date) must be closed now, else None."""
    if not ctx or not ctx.get("active") or expiry is None:
        return None
    report = (ctx["by_root"].get(root) or {}).get("report")
    if not report or report[0] > expiry or not earnings_close_due(ctx["today"], report[0], report[1]):
        return None
    return report[0], report[1], earnings_close_session(report[0], report[1])


def _earnings_close_reason(ctx: dict, root: str, report_date, hour: str, session, what: str) -> str:
    head = f"EARNINGS CLOSE: {root} reports {report_date.isoformat()} {EARNINGS_HOUR_WORDS.get(hour, EARNINGS_HOUR_WORDS[''])}"
    if ctx["today"] > session:
        return (f"{head}; its last session before the report ({session.isoformat()}) has passed — closing {what} "
                f"at the first eligible tick")
    return (f"{head}; {session.isoformat()} is the last session before it — closing {what} rather than holding it "
            f"through the report")


def manage_options_positions(equity: float = 100000, allow_rolls: bool = True,
                             roll_skip_detail: str = ROLL_SKIPPED_HALTED_DETAIL) -> dict:
    """
    Professional options position management. Called every scan cycle.

    allow_rolls=False (exits-only mode, used while a halt blocks entries, and
    while entry_shadow shadows the options sleeve): every roll point closes
    the position instead of rolling it; roll_skip_detail names why.

    Regular hours only (REPAIR 2026-11-04): while the market is closed no
    single-leg close or roll is submitted. A rule that fires is reported as a
    CLOSE whose order is {"status": "deferred"} (_defer_single_leg_close) and
    is judged again on regular-hours marks. Strategy groups defer the same
    way (_request_group_close).

    Checks every open options position for:
      0. Earnings close (a sold option, on the last session before its
         underlying reports; see the EARNINGS CLOSE block above)
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

    # Skip positions managed by convexity overlay (protective QQQ puts)
    # These are managed by _run_convexity_overlay() in bot_engine.py
    convexity_symbols = set()
    try:
        meta_path = os.path.join(DATA_DIR, 'voltrade_options_state.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                _all_meta = json.load(f)
            for sym, meta in _all_meta.items():
                if meta.get('strategy') == 'convexity_hedge' or meta.get('managed_by') == 'convexity_overlay':
                    convexity_symbols.add(sym)
    except Exception:
        pass

    options_positions = [
        p for p in options_positions
        if p.get("symbol", "") not in convexity_symbols
    ]

    if not options_positions:
        # Clean stale state entries — but NEVER the convexity overlay's
        # markers (repair 2026-08-06, full-code-review finding, adversarially
        # verified): when the ONLY held options are the QQQ tail-hedge puts,
        # they are filtered out above and this branch used to wipe the WHOLE
        # state file — destroying the strategy/managed_by markers that
        # _run_convexity_overlay (bot_engine.py) and the filter above both
        # re-identify the hedge legs by. Orphaned hedge puts then get managed
        # as ordinary positions on the next cycle. Preserve every
        # convexity-managed entry; drop only the truly stale rest.
        preserved = {k: v for k, v in state.items() if k in convexity_symbols}
        if state != preserved:
            _save_options_state(preserved)
        return {"actions": [], "positions_checked": 0, "options_state": preserved}

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

    # LEFTOVER LONG LEG (2026-11-05): how many option legs are open per
    # underlying. A bought leg that is the only one left on its underlying is
    # a leftover (Check 1b in Phase 2), even when its state still records the
    # multi-leg strategy it came from: as a one-leg "group" no group rule can
    # exit it (a long wing has no net credit to take profit on).
    legs_per_underlying: dict = {}
    for _p in options_positions:
        _parsed = _parse_occ_symbol(_p.get("symbol", ""))
        if _parsed:
            legs_per_underlying[_parsed["ticker"]] = legs_per_underlying.get(_parsed["ticker"], 0) + 1

    for pos in options_positions:
        occ = pos.get("symbol", "")
        meta = options_meta.get(occ, {})
        strategy = meta.get("strategy", "")
        ticker = meta.get("ticker", "")
        _root = (_parse_occ_symbol(occ) or {}).get("ticker")
        lone_long = pos.get("side", "long") == "long" and legs_per_underlying.get(_root, 0) == 1

        # P0-7/M4 FIX: added bull_put_credit_spread and qqq_iron_condor so
        # they get group-level profit target / max-loss / coordinated exit
        # management. Previously their 4 (or 2) legs were managed as
        # standalone positions which can't coordinate exits.
        if strategy in MULTI_LEG_STRATEGIES and not lone_long:
            key = f"{ticker}_{strategy}"
            if key not in strategy_groups:
                strategy_groups[key] = {"legs": [], "strategy": strategy, "ticker": ticker, "setup": meta.get("setup", "")}
            strategy_groups[key]["legs"].append(pos)
        else:
            standalone_positions.append(pos)

    # ── ADOPTION (REPAIR 2026-11-02): a leg whose state carries no multi-leg
    # strategy but whose root + expiry match a group's legs is one of that
    # group's legs (the GOOGL condor's wings after their state was popped) —
    # it is managed WITH the group, never by the single-leg rules, which the
    # broker refuses for a spread wing anyway ("uncovered option contracts").
    if strategy_groups and standalone_positions:
        by_root_exp = {}
        for key, group in strategy_groups.items():
            for leg in group["legs"]:
                parsed_leg = _parse_occ_symbol(leg.get("symbol", ""))
                if parsed_leg:
                    by_root_exp.setdefault((parsed_leg["ticker"], parsed_leg["expiry_date"]), key)
        kept = []
        for pos in standalone_positions:
            parsed_pos = _parse_occ_symbol(pos.get("symbol", ""))
            key = by_root_exp.get((parsed_pos.get("ticker"), parsed_pos.get("expiry_date"))) if parsed_pos else None
            if key is None:
                kept.append(pos)
                continue
            group = strategy_groups[key]
            group["legs"].append(pos)
            occ = pos.get("symbol", "")
            st = state.get(occ) if isinstance(state.get(occ), dict) else {}
            st.update({"strategy": group["strategy"], "ticker": group["ticker"], "setup": st.get("setup") or group.get("setup", ""), "adopted_into_group": key})
            state[occ] = st
            actions.append({"action": "WARNING", "ticker": group["ticker"], "occ_symbol": occ, "type": "group_adopted",
                            "reason": f"{occ} carried no strategy but matches the {group['strategy']} legs on {group['ticker']} {parsed_pos['expiry_date']} — managed with the group"})
        standalone_positions = kept

    # ── EARNINGS CLOSE (2026-11-05): every sold option's next report, looked up
    # once per underlying per ET day; the first pass since process start also
    # reports the schedule (the deploy backfill). A bug here must never stop
    # the other exit rules: it is logged with its traceback and skipped.
    try:
        earnings = _earnings_close_context(options_positions, actions, strategy_groups)
    except Exception as e:
        logger.exception(f"EARNINGS CLOSE skipped this pass: {type(e).__name__}: {e}")
        earnings = None

    # ── Phase 1: Manage multi-leg strategy groups ─────────────────────
    for group_key, group in strategy_groups.items():
        group_actions = _manage_strategy_group(group, equity, state, earnings)
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
        # [RULE-REVIEW] 2026-10-27 (ORPHAN EXITS): an occ_symbol opened by the
        # Node tier dispatcher carries ONLY a pending_entry_feedback marker
        # (register_pending_entry_feedback) — initialize it exactly like an
        # unregistered position, keeping the marker so the entry resolves.
        _pending_marker = pos_state.get("pending_entry_feedback") if pos_state else None
        if not pos_state or "entry_price" not in pos_state:
            pos_state = {
                "entry_price": entry_price,
                "entry_delta": current_delta,
                "entry_date": time.strftime("%Y-%m-%d"),
                "entry_timestamp": datetime.now().isoformat(),  # v1.0.34 FIX: was missing — MIN_HOLD defaulted to 999min
                "initial_credit": entry_price if side == "short" else 0,
                "max_profit_target": entry_price * PROFIT_TARGET_PCT if side == "short" else 0,
                "highest_value": current_price,
                "strategy": "unknown",
                "side": side,
                "qty": qty,
            }
            if _pending_marker:
                pos_state["pending_entry_feedback"] = _pending_marker
        else:
            # Update high water mark
            pos_state["highest_value"] = max(
                pos_state.get("highest_value", 0),
                current_price
            )

        # Resolve any pending trade_feedback ENTRY left by
        # register_options_entry() (see that function's own comment) —
        # this occ_symbol just showed up in Alpaca's live `/v2/positions`,
        # which only lists REAL, confirmed-filled positions, so this is
        # the first point we can trust the fill actually happened. Uses
        # this loop's own `entry_price`/`qty` (Alpaca's real
        # `avg_entry_price`/filled qty, not the submission-time limit
        # price/requested qty the pending payload was built from).
        # Popped (not just read) so it fires exactly once per position.
        pending_fb = pos_state.pop("pending_entry_feedback", None)
        if pending_fb:
            try:
                from ml_model_v2 import track_fill
                now_iso = datetime.now().isoformat()
                track_fill({
                    **pending_fb,
                    "qty": qty,
                    "expected_price": entry_price,
                    "fill_price": entry_price,
                    "time_placed": now_iso,
                    "time_filled": now_iso,
                })
            except Exception as e:
                logger.warning(f"track_fill pending-entry record failed for {occ_symbol}: {e}")

        state[occ_symbol] = pos_state

        # ── Check 0: MINIMUM HOLD TIME ──────────────────────────────────
        _entry_ts = pos_state.get("entry_timestamp")
        _held_minutes = 999  # default: assume held long enough
        if _entry_ts:
            try:
                _entry_dt = datetime.fromisoformat(_entry_ts)
                _held_minutes = (datetime.now() - _entry_dt).total_seconds() / 60
            except Exception:
                pass

        # ── Check 1: CRITICAL DTE — Force close (ignores hold time) ─────
        if dte <= DTE_CRITICAL:
            if _defer_single_leg_close(actions, parsed["ticker"], occ_symbol, "dte_critical",
                                       f"CRITICAL DTE: {dte} days to expiry", dte):
                continue
            close_side = "sell" if side == "long" else "buy"
            # Use mid price for closing
            limit_px = snap.get("mid", current_price) if snap else current_price
            if limit_px <= 0:
                limit_px = current_price
            result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
            _record_options_exit_feedback(parsed["ticker"], side, qty, entry_price, current_price,
                                           unrealized_pnl, "dte_critical", pos_state.get("entry_timestamp"), order_result=result)
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

        # ── Check 1a: EARNINGS CLOSE (2026-11-05) — ignores hold time ───────
        # A sold option is bought back at the ask on the last session before
        # its underlying reports (the EARNINGS CLOSE block above). Judged before
        # the hold gate, like CRITICAL DTE: a leg sold late in that session
        # must not ride into the report.
        _earn = _earnings_close_due_for(earnings, parsed["ticker"], _expiry_date(parsed)) if side == "short" else None
        if _earn:
            _earn_reason = _earnings_close_reason(earnings, parsed["ticker"], *_earn, "the sold option")
            if _defer_single_leg_close(actions, parsed["ticker"], occ_symbol, "earnings_close", _earn_reason, dte):
                continue
            # Judged again every tick (before the hold gate): never a second
            # buy-to-close while one is working (it could open a long).
            if [o for o in _open_orders_for([occ_symbol]) if isinstance(o, dict) and str(o.get("side", "")).lower() == "buy"]:
                actions.append({"action": "WAIT", "ticker": parsed["ticker"], "occ_symbol": occ_symbol,
                                "reason": f"{_earn_reason} — a close order is already working at the broker",
                                "type": "earnings_close_working", "dte": dte})
                continue
            limit_px = snap.get("ask", 0) if snap else 0
            if not limit_px or limit_px <= 0:
                actions.append({"action": "HOLD", "ticker": parsed["ticker"], "occ_symbol": occ_symbol,
                                "reason": f"{_earn_reason} — no ask quoted, judged again next tick",
                                "type": "earnings_close_no_ask", "dte": dte})
                continue
            result = _submit_close_order(occ_symbol, qty, "buy", limit_px)
            _record_options_exit_feedback(parsed["ticker"], side, qty, entry_price, current_price,
                                           unrealized_pnl, "earnings_close", pos_state.get("entry_timestamp"), order_result=result)
            actions.append({
                "action": "CLOSE",
                "ticker": parsed["ticker"],
                "occ_symbol": occ_symbol,
                "reason": _earn_reason,
                "type": "earnings_close",
                "dte": dte,
                "pnl": round(unrealized_pnl, 2),
                "report_date": _earn[0].isoformat(),
                "report_hour": _earn[1],
                "close_session": _earn[2].isoformat(),
                "order": result,
            })
            # A rejected close keeps the leg's state (its strategy and entry
            # time); the rule simply fires again on the next tick.
            if result.get("status") == "submitted":
                state.pop(occ_symbol, None)
            continue

        # ── MINIMUM HOLD TIME GATE ──────────────────────────────────────
        # Skip all non-critical exit checks if position held < 1 hour
        if _held_minutes < MIN_HOLD_MINUTES:
            continue

        # ── Check 1b: LEFTOVER LONG LEG (2026-11-05) ─────────────────────
        # A bought option alone on its underlying, not a deliberate standalone
        # long: the structure it belonged to is gone, so close it (see
        # STANDALONE_LONG_STRATEGIES). Sold at the bid: a paper limit fills only
        # once marketable, and a leftover bleeds every day it is held.
        if (side == "long"
                and legs_per_underlying.get(parsed["ticker"], 0) == 1
                and pos_state.get("strategy") not in STANDALONE_LONG_STRATEGIES):
            _leftover_reason = (f"LEFTOVER LONG LEG: {occ_symbol} is the only option left on {parsed['ticker']} "
                              f"(recorded strategy {pos_state.get('strategy') or 'unknown'}) — its structure is gone")
            if _defer_single_leg_close(actions, parsed["ticker"], occ_symbol, "leftover_long_close", _leftover_reason, dte):
                continue
            limit_px = snap.get("bid", 0) if snap else 0
            if not limit_px or limit_px <= 0:
                actions.append({"action": "HOLD", "ticker": parsed["ticker"], "occ_symbol": occ_symbol,
                                "reason": f"{_leftover_reason} — no bid quoted, judged again next tick",
                                "type": "leftover_long_no_bid", "dte": dte})
                continue
            result = _submit_close_order(occ_symbol, qty, "sell", limit_px)
            # A wing still recording its multi-leg strategy is not attributed
            # (KNOWN BROKEN #12(c), the same as a group close); an unknown-
            # strategy leg is, like every other single-leg exit.
            if pos_state.get("strategy") not in MULTI_LEG_STRATEGIES:
                _record_options_exit_feedback(parsed["ticker"], side, qty, entry_price, current_price,
                                               unrealized_pnl, "leftover_long_close", pos_state.get("entry_timestamp"),
                                               order_result=result)
            actions.append({
                "action": "CLOSE",
                "ticker": parsed["ticker"],
                "occ_symbol": occ_symbol,
                "reason": _leftover_reason,
                "type": "leftover_long_close",
                "dte": dte,
                "pnl": round(unrealized_pnl, 2),
                "order": result,
            })
            state.pop(occ_symbol, None)
            continue

        # ── Check 2: Assignment Risk ─────────────────────────────────────
        if (side == "short" and
            abs(current_delta) > ASSIGNMENT_DELTA_THRESHOLD and
                dte <= 10):
            # Deep ITM sold option near expiry — high assignment risk
            if _defer_single_leg_close(actions, parsed["ticker"], occ_symbol, "assignment_close",
                                       f"ASSIGNMENT RISK: |delta|={abs(current_delta):.2f}, {dte} DTE", dte):
                continue
            # Try to roll first
            roll_result = _roll_or_skip(allow_rolls, occ_symbol, qty, "sell", parsed, equity, roll_skip_detail)
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
                    state[new_sym] = rolled_position_state(pos_state, roll_result.get("new_credit"))
                state.pop(occ_symbol, None)
                continue
            else:
                # Roll failed — force close
                close_side = "buy"  # Buy back the sold option
                limit_px = snap.get("ask", current_price) if snap else current_price
                result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                _record_options_exit_feedback(parsed["ticker"], side, qty, entry_price, current_price,
                                               unrealized_pnl, "assignment_close", pos_state.get("entry_timestamp"), order_result=result)
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
            if _defer_single_leg_close(actions, parsed["ticker"], occ_symbol,
                                       "dte_close" if side == "short" else "dte_close_bought",
                                       f"21 DTE EXIT: {dte} DTE", dte):
                continue
            if side == "short":
                # Sold option at 21 DTE — try to roll
                roll_result = _roll_or_skip(allow_rolls, occ_symbol, qty, "sell", parsed, equity, roll_skip_detail)
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
                        state[new_sym] = rolled_position_state(pos_state, roll_result.get("new_credit"))
                    state.pop(occ_symbol, None)
                    continue
                else:
                    # Roll failed — close
                    close_side = "buy"
                    limit_px = snap.get("ask", current_price) if snap else current_price
                    result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                    _record_options_exit_feedback(parsed["ticker"], side, qty, entry_price, current_price,
                                                   unrealized_pnl, "dte_close", pos_state.get("entry_timestamp"), order_result=result)
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
                _record_options_exit_feedback(parsed["ticker"], side, qty, entry_price, current_price,
                                               unrealized_pnl, "dte_close_bought", pos_state.get("entry_timestamp"), order_result=result)
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
                    if _defer_single_leg_close(actions, parsed["ticker"], occ_symbol, "profit_target",
                                               f"PROFIT TARGET: {profit_pct:.0%} of max profit reached", dte):
                        continue
                    # Hit 50% profit target — close and redeploy capital
                    close_side = "buy"
                    limit_px = snap.get("mid", current_price) if snap else current_price
                    if limit_px <= 0:
                        limit_px = current_price
                    result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                    _record_options_exit_feedback(parsed["ticker"], side, qty, entry_price, current_price,
                                                   unrealized_pnl, "profit_target", pos_state.get("entry_timestamp"), order_result=result)
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
                    if _defer_single_leg_close(actions, parsed["ticker"], occ_symbol, "loss_limit",
                                               (f"LOSS LIMIT: Cost to close ${current_price:.2f} is "
                                                f"{loss_ratio:.1f}x the credit received ${initial_credit:.2f}"), dte):
                        continue
                    close_side = "buy"
                    limit_px = snap.get("ask", current_price) if snap else current_price
                    result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                    _record_options_exit_feedback(parsed["ticker"], side, qty, entry_price, current_price,
                                                   unrealized_pnl, "loss_limit", pos_state.get("entry_timestamp"), order_result=result)
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
                    if _defer_single_leg_close(actions, parsed["ticker"], occ_symbol, "bought_loss_limit",
                                               (f"LOSS LIMIT: Bought option down {loss_pct:.0%} from entry "
                                                f"(${entry_price:.2f} → ${current_price:.2f})"), dte):
                        continue
                    close_side = "sell"
                    limit_px = snap.get("bid", current_price) if snap else current_price
                    if limit_px <= 0:
                        limit_px = current_price
                    result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
                    _record_options_exit_feedback(parsed["ticker"], side, qty, entry_price, current_price,
                                                   unrealized_pnl, "bought_loss_limit", pos_state.get("entry_timestamp"), order_result=result)
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
            if _defer_single_leg_close(actions, parsed["ticker"], occ_symbol, "gamma_risk",
                                       f"GAMMA RISK: gamma={current_gamma:.3f} exceeds {GAMMA_THRESHOLD} with {dte} DTE", dte):
                continue
            close_side = "sell" if side == "long" else "buy"
            limit_px = snap.get("mid", current_price) if snap else current_price
            if limit_px <= 0:
                limit_px = current_price
            result = _submit_close_order(occ_symbol, qty, close_side, limit_px)
            _record_options_exit_feedback(parsed["ticker"], side, qty, entry_price, current_price,
                                           unrealized_pnl, "gamma_risk", pos_state.get("entry_timestamp"), order_result=result)
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
                           strategy: str, delta: float = 0, qty: int = 1,
                           ticker: str = "", setup: str = "",
                           max_loss: float = 0):
    """
    Called when a new options position is opened. Records the entry state
    so the manager can track profit targets, delta drift, etc.

    v1.0.34: Added ticker and setup params for proper strategy grouping.
    Also extracts ticker from OCC symbol as fallback.
    Added max_loss param for iron condor 50%-of-max-loss early exit.
    """
    # Extract ticker from OCC symbol if not provided
    if not ticker:
        for i, ch in enumerate(occ_symbol):
            if ch.isdigit():
                ticker = occ_symbol[:i]
                break
    state = _load_options_state()
    state[occ_symbol] = {
        "entry_price": entry_price,
        "entry_delta": delta,
        "entry_date": time.strftime("%Y-%m-%d"),
        "entry_timestamp": datetime.now().isoformat(),
        "initial_credit": entry_price if side == "sell" else 0,
        "max_profit_target": entry_price * PROFIT_TARGET_PCT if side == "sell" else 0,
        "highest_value": entry_price,
        "strategy": strategy,
        "setup": setup,
        "ticker": ticker,
        "side": "short" if side == "sell" else "long",
        "qty": qty,
        "max_loss": max_loss,  # v1.0.34: for 50%-of-max-loss early exit
    }
    # KNOWN BROKEN #12(c) FURTHER REPAIR (found live 2026-09-08): this used
    # to call track_fill() unconditionally right here, at order SUBMISSION
    # time — but `register_options_entry` is called for every status in
    # `("submitted", "filled", "pending_new", "accepted")`, none of which
    # mean the order actually filled. Live `/api/diag/orders` this session
    # showed 42/48 (87.5%) of standalone CSP opening orders in a recent
    # window are `canceled` (day-limit orders that never reached their
    # price, the same fill-timing gap KNOWN BROKEN #30 already documented
    # for slot-counting) — so most of those track_fill calls wrote a
    # phantom trade_feedback ENTRY record (fill_price = the quoted limit,
    # never a real fill) for a position that never existed. Worse: several
    # tickers (e.g. HPE) show a canceled retry AFTER a real fill on the
    # same ticker — `ml_model_v2._find_entry_record` matches the MOST
    # RECENT unmatched record for a ticker, so a later phantom can get
    # matched to the real position's eventual exit instead of the real
    # entry, corrupting which record records the true fill data (qty,
    # timestamps) even though the exit's own pnl_pct comes from Alpaca's
    # real `unrealized_pl` and is unaffected. FIX: defer the track_fill
    # call — stash the payload as `pending_entry_feedback` and let
    # `manage_options_positions()` fire it the first time this occ_symbol
    # shows up as a REAL, confirmed-filled Alpaca position (mirroring the
    # bot.ts KNOWN BROKEN #35 "record on confirmed fill, not on submit"
    # precedent). A canceled order never appears in `/v2/positions`, so
    # its pending marker simply never fires — no phantom record, ever.
    if strategy not in MULTI_LEG_STRATEGIES:
        state[occ_symbol]["pending_entry_feedback"] = {
            "ticker": ticker,
            "side": side,
            "session": "regular",
            "entry_features": {},
            "code_version": _code_version(),
        }
    _save_options_state(state)


def register_pending_entry_feedback(occ_symbol: str, ticker: str, side: str) -> None:
    """[RULE-REVIEW] 2026-10-27 (ORPHAN EXITS, audit §7.2.4): stash the
    trade_feedback ENTRY marker for a single-leg option opened OUTSIDE
    register_options_entry() — the Node tier dispatcher's SELL_CSP / BUY_PUT
    branches call submit_options_order() directly and never registered, so
    manage_options_positions() had no pending entry to resolve and every
    later close of those positions (the bulk of the CSP sleeve) was written
    as an orphan_exit. Only the marker is written (merged, never replacing
    an existing state entry): the manager initializes the rest exactly as it
    already did for these positions, so exit thresholds are unchanged. The
    marker resolves on the first scan that sees the occ_symbol as a real,
    filled Alpaca position, at Alpaca's avg_entry_price/qty; a cancelled
    order never appears, so it never writes a phantom entry."""
    if not occ_symbol or not ticker:
        return
    state = _load_options_state()
    entry = state.get(occ_symbol) or {}
    if "pending_entry_feedback" in entry or "entry_price" in entry:
        return  # already registered by register_options_entry or a prior call
    entry["pending_entry_feedback"] = {
        "ticker": str(ticker).upper(),
        "side": side,
        "session": "regular",
        "entry_features": {},
        "code_version": _code_version(),
    }
    state[occ_symbol] = entry
    _save_options_state(state)


def merge_options_meta(occ_symbols, meta: dict) -> None:
    """
    Add display/grouping fields (strategy, setup, ticker, …) to the state of
    each key WITHOUT dropping what register_options_entry() already recorded.

    REPAIR 2026-10-20 (live: TSLA/META/GOOGL iron condors 2026-10-19/20,
    each market-closed 4-10 minutes after opening for -$66/-$70/-$84): the
    bot.ts scanner-direct path registered every leg here and then rewrote
    the same keys as a bare {strategy, setup, ticker} dict, deleting
    entry_timestamp (so _manage_strategy_group's MIN_HOLD_MINUTES gate saw
    no timestamp and let every exit rule run on the first cycle) and
    max_loss (so the iron-condor 50%-of-max-loss exit never armed). Every
    writer of display metadata goes through this merge instead.
    """
    keys = [k for k in (occ_symbols or []) if isinstance(k, str) and k]
    if not keys:
        return
    state = _load_options_state()
    for k in keys:
        prev = state.get(k)
        state[k] = {**(prev if isinstance(prev, dict) else {}), **meta}
    _save_options_state(state)
