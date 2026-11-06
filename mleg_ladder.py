"""
VolTradeAI — multi-leg option orders worked from the net mid (2026-11-06)
=========================================================================
Human refocus spec §5: "CBOE complex order book for multi-leg fills, work
limits near mid". Until now a strategy group was closed with a MARKET mleg
order (options_manager._close_strategy_mleg), the widest fill there is.

A multi-leg (mleg) order trades the combination at one net price. Its quote
is built from the legs' quotes (each leg on the side the ORDER takes):
  paying, a debit (the order buys the combination):
      natural = sum of the asks of the legs bought - sum of the bids of the legs sold
      far     = sum of the bids of the legs bought - sum of the asks of the legs sold
  receiving, a credit (the order sells it):
      natural = sum of the bids of the legs sold - sum of the asks of the legs bought
      far     = sum of the asks of the legs sold - sum of the bids of the legs bought
The side is the sign of the net mid. The price schedule is order_ladder's:
step 1 the net mid, the last step the natural (a marketable limit, never a
market order), evenly between, at valid increments, never backing off. The
caller (options_manager) steps it once per tick, no sooner than step_seconds,
re-reading the legs' quotes; once the natural has rested touch_seconds
unfilled the order is canceled and the caller falls back to today's order.

Alpaca's mleg limit_price is SIGNED: a debit is positive, a credit negative
(signed_limit). Evidence, live /api/diag/orders 2026-10-19..11-04: every
condor entry was sent with a POSITIVE limit (options_execution.
_submit_mleg_order sends abs(limit_price)) and filled at a NEGATIVE price (a
credit), three of them at a credit SMALLER than the limit (0.71 -> -0.70,
4.59 -> -4.55, 3.56 -> -3.53): a positive limit on a credit order is "pay up
to", which any credit satisfies. A credit close here is therefore sent with
a negative limit; if the broker refuses that, the caller sends today's order.

Paper vs live: Alpaca paper fills a limit only once it is marketable at the
NBBO (the natural), with no price improvement; the complex order book's
price improvement exists only in live routing. In paper this changes WHEN a
group close fills (at the natural, a few ticks after it starts, instead of a
market order at once) and tests the mechanics; no fill-improvement claim is
made from paper.

Pure: no broker calls (options_manager owns every POST, PATCH and DELETE).
"""
import logging
import math
import os

import order_ladder

logger = logging.getLogger("mleg_ladder")

ENV_VAR = "VOLTRADE_OPTIONS_MLEG_LADDER"   # on | off | auto; wins over system_config when set
DEFAULTS = {"mode": "on", "steps": 3, "step_seconds": 60.0, "touch_seconds": 120.0}
MAX_STEPS = 10
MAX_SECONDS = 3600.0


def settings(trading_base_url: str = "") -> dict:
    """system_config.BASE_CONFIG["OPTIONS_MLEG_LADDER"] (env VOLTRADE_OPTIONS_MLEG_LADDER
    wins for the mode; "auto" = on only against the live host). Unreadable: off."""
    try:
        from system_config import BASE_CONFIG
        raw = BASE_CONFIG.get("OPTIONS_MLEG_LADDER") or {}
    except Exception as e:
        logger.warning(f"mleg ladder config unreadable — ladder off: {e}")
        raw = {"mode": "off"}
    if not isinstance(raw, dict):
        raw = {"mode": "off"}
    mode = order_ladder.resolve_mode(raw.get("mode", "off"), os.environ.get(ENV_VAR), trading_base_url)
    return {
        "on": mode == "on",
        "steps": order_ladder._bounded(raw.get("steps", DEFAULTS["steps"]), DEFAULTS["steps"], 2, MAX_STEPS, int),
        "step_seconds": order_ladder._bounded(raw.get("step_seconds", DEFAULTS["step_seconds"]), DEFAULTS["step_seconds"],
                                              0.0, MAX_SECONDS, float),
        "touch_seconds": order_ladder._bounded(raw.get("touch_seconds", DEFAULTS["touch_seconds"]), DEFAULTS["touch_seconds"],
                                               0.0, MAX_SECONDS, float),
    }


def _px(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def combo_quote(legs: list):
    """legs: [{"side": "buy" | "sell", "bid", "ask", "ratio"}] — each leg on the
    side the order takes. Returns {"side", "bid", "ask", "mid"} for the
    combination in order_ladder's terms (a "buy" walks up from the mid to its
    ask = the natural debit; a "sell" walks down to its bid = the natural
    credit), all positive; None when a leg has no usable quote (ask > 0 and
    0 <= bid <= ask) or the combination's natural or far side is not > 0."""
    if not legs:
        return None
    nat_debit = far_debit = 0.0
    for leg in legs:
        bid, ask = _px(leg.get("bid")), _px(leg.get("ask"))
        ratio = int(leg.get("ratio") or 1)
        if bid is None or ask is None or ask <= 0 or bid < 0 or ask < bid or ratio < 1:
            return None
        if leg.get("side") == "buy":
            nat_debit += ask * ratio
            far_debit += bid * ratio
        elif leg.get("side") == "sell":
            nat_debit -= bid * ratio
            far_debit -= ask * ratio
        else:
            return None
    mid_debit = (nat_debit + far_debit) / 2.0
    if mid_debit > 0:
        q = {"side": "buy", "bid": far_debit, "ask": nat_debit}
    else:
        q = {"side": "sell", "bid": -nat_debit, "ask": -far_debit}
    if q["bid"] <= 0 or q["ask"] <= 0:
        return None
    q = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}
    q["mid"] = round((q["bid"] + q["ask"]) / 2.0, 4)
    return q


def signed_limit(side: str, price: float) -> float:
    """Alpaca's mleg sign: a debit (buy) positive, a credit (sell) negative."""
    return round(float(price), 2) if side == "buy" else -round(float(price), 2)


def plan(legs: list, steps: int) -> dict:
    """{"side", "bid", "ask", "prices": [mid .. natural]} for a combination worth
    working from the mid, else {"skip": why}."""
    q = combo_quote(legs)
    if q is None:
        return {"skip": "a leg has no usable two-sided quote, or the combination's natural or far side is not above zero"}
    prices = order_ladder.ladder_prices(q["side"], q["bid"], q["ask"], int(steps))
    natural, far = (q["ask"], q["bid"]) if q["side"] == "buy" else (q["bid"], q["ask"])
    if len(prices) < 2 or order_ladder._cents(prices[0]) in (order_ladder._cents(natural), order_ladder._cents(far)):
        return {"skip": f"no price strictly inside the combination's spread ({q['bid']:.2f} / {q['ask']:.2f})"}
    return {**q, "prices": prices}


def next_step(ladder: dict, legs: list, now: float, cfg: dict) -> dict:
    """What a working ladder does this tick: {"do": "wait"} | {"do": "step",
    "price", "step"} (re-price to the next step, from the legs' fresh quotes,
    never backing off) | {"do": "hold", "step"} (the next step's price equals
    the working one) | {"do": "cancel", "why"} (the natural has rested
    touch_seconds unfilled). Pure."""
    n, step = int(ladder.get("steps") or 0), int(ladder.get("step") or 0)
    since = now - float(ladder.get("step_at") or 0)
    if step >= n:
        if since >= float(cfg.get("touch_seconds", DEFAULTS["touch_seconds"])):
            return {"do": "cancel", "why": f"the natural ${float(ladder.get('price') or 0):.2f} rested "
                                           f"{int(since)}s unfilled"}
        return {"do": "wait"}
    if since < float(cfg.get("step_seconds", DEFAULTS["step_seconds"])):
        return {"do": "wait"}
    q = combo_quote(legs)
    if q is None or q["side"] != ladder.get("side"):
        return {"do": "wait"}   # no usable quote this tick (or it flipped sides): judged again next tick
    prices = order_ladder.ladder_prices(q["side"], q["bid"], q["ask"], n)
    if not prices:
        return {"do": "wait"}
    target = order_ladder._no_back_off(q["side"], prices[step], float(ladder.get("price") or 0))
    if order_ladder._cents(target) == order_ladder._cents(float(ladder.get("price") or 0)):
        return {"do": "hold", "step": step + 1, "price": target}
    return {"do": "step", "step": step + 1, "price": target, "bid": q["bid"], "ask": q["ask"]}
