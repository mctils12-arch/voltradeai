"""
VolTradeAI — mid-first exit ladder for single-leg option closes (2026-11-05)
===========================================================================
A single-leg option exit that would cross the spread (a sell at the bid, a buy
at the ask) is worked from the mid toward that touch instead of crossing at
once:

  step 1        the mid (rounded to a valid increment in our favour)
  steps 2..N-1  evenly spaced between the mid and the touch
  step N        the touch itself: a marketable limit, never a market order

Prices are re-read from the live quote at every step. Each step is at least as
aggressive as the one before it (a sell never re-prices up, a buy never down).
Steps advance at most once per options-manager tick and no sooner than
`step_seconds` after the previous step, so `step_seconds` is a minimum: at the
Tier 2 cadence (1-7 min) a 3-step ladder reaches the touch 2 ticks after it
starts. After the last step the order rests at that touch, like today's single
order (it is not chased).

Which closes are laddered: only one whose own price crosses the spread at the
live quote. A rule that prices at the mid (CRITICAL DTE, profit target, gamma)
already rests inside the spread and goes out exactly as before. Rolls and
multi-leg group closes never ladder (options_manager wraps them in paused()).

Paper vs live: Alpaca paper fills a limit only once it is marketable at the
NBBO, with no price improvement. In paper the ladder exercises the mechanics
(step timing, replace races, no duplicate order, one exit record per close)
and an exit normally fills at the touch after the last step, a few minutes
later than a single touch order. A fill before the last step is logged with
its step. No fill-improvement claim is made from paper.

State: one entry per contract being worked, in its own JSON file (the options
state entry is popped by the rule that submits, and the hold gate then blocks
the rule from firing again for an hour, so the ladder cannot rely on the rule
re-firing: options_manager calls pursue() once per tick, before the rules).
Every finished ladder appends one line to a JSONL log: outcome, the step it
filled at, prices and quotes per step.

This module holds no broker HTTP: options_manager injects every broker and
quote call through LadderIO, so the step logic is tested without a broker.
"""
import json
import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from urllib.parse import urlparse

logger = logging.getLogger("order_ladder")

ENV_VAR = "VOLTRADE_OPTIONS_EXIT_LADDER"   # on | off | auto; wins over system_config when set
LIVE_TRADING_HOST = "api.alpaca.markets"   # "auto" turns the ladder on only against this host
STATE_FILE = "voltrade_options_exit_ladder.json"
LOG_FILE = "voltrade_options_exit_ladder_log.jsonl"
DEFAULT_STEPS = 3
DEFAULT_STEP_SECONDS = 60.0
MAX_STEPS = 10
MAX_STEP_SECONDS = 3600.0
# A ladder older than this started in an earlier session: a regular session is
# 6.5 hours and every ladder order is a DAY order, so its order is gone. The
# entry is dropped without touching any order.
MAX_LADDER_AGE_SECONDS = 7 * 3600
# Option price increments (penny interval program): $0.01 below $3.00, $0.05
# at or above. Only intermediate steps are rounded; the last step is the quoted
# touch itself.
TICK_BREAK_CENTS = 300
TICK_BELOW_CENTS = 1
TICK_ABOVE_CENTS = 5

# Alpaca order statuses, as the ladder treats them.
FILLED = ("filled",)
ENDED = ("canceled", "expired", "rejected", "done_for_day", "not_found")   # not_found: the broker has no such order
REPLACEABLE = ("new",)      # Alpaca refuses to replace accepted / pending_* orders
PARTIAL = ("partially_filled",)


def wall_clock() -> float:
    return time.time()


# ── configuration ───────────────────────────────────────────────────────────

def is_live_trading_url(url) -> bool:
    """True only for Alpaca's live trading host. Paper, a mock, an empty or an
    unreadable URL is not live, so "auto" fails closed to today's single order."""
    if not isinstance(url, str) or not url:
        return False
    try:
        return (urlparse(url).hostname or "") == LIVE_TRADING_HOST
    except ValueError:
        return False


def resolve_mode(configured, env_value=None, trading_base_url="") -> str:
    """"on" or "off". The env value wins when it is set and non-empty; "auto"
    is on only against the live endpoint; anything unrecognised is off."""
    setting = str(configured or "").strip().lower()
    if env_value is not None and str(env_value).strip():
        setting = str(env_value).strip().lower()
    if setting == "on":
        return "on"
    if setting == "auto":
        return "on" if is_live_trading_url(trading_base_url) else "off"
    if setting != "off":
        logger.warning(f"exit ladder mode {setting!r} is not on/off/auto — treated as off")
    return "off"


def _bounded(value, default, lo, hi, cast):
    try:
        v = cast(value)
    except (TypeError, ValueError):
        return default
    if isinstance(v, float) and not math.isfinite(v):
        return default
    return min(max(v, lo), hi)


def settings(trading_base_url: str) -> dict:
    """The ladder's effective settings: system_config.BASE_CONFIG["OPTIONS_EXIT_LADDER"],
    with VOLTRADE_OPTIONS_EXIT_LADDER overriding its mode. An unreadable
    config is off."""
    try:
        from system_config import BASE_CONFIG
        raw = BASE_CONFIG.get("OPTIONS_EXIT_LADDER") or {}
    except Exception as e:
        logger.warning(f"exit ladder config unreadable — ladder off: {e}")
        raw = {"mode": "off"}
    if not isinstance(raw, dict):
        raw = {"mode": "off"}
    env_value = os.environ.get(ENV_VAR)
    mode = resolve_mode(raw.get("mode", "off"), env_value, trading_base_url)
    return {
        "on": mode == "on",
        "setting": str(env_value).strip().lower() if env_value and str(env_value).strip() else str(raw.get("mode", "off")),
        "steps": _bounded(raw.get("steps", DEFAULT_STEPS), DEFAULT_STEPS, 1, MAX_STEPS, int),
        "step_seconds": _bounded(raw.get("step_seconds", DEFAULT_STEP_SECONDS), DEFAULT_STEP_SECONDS,
                                 0.0, MAX_STEP_SECONDS, float),
    }


# ── pause: closes that must never ladder ────────────────────────────────────

_LOCAL = threading.local()


@contextmanager
def paused():
    """Closes submitted inside this block are never laddered (a roll's close
    and sell-to-open, a group's per-leg fallback, and the ladder's own orders)."""
    _LOCAL.depth = getattr(_LOCAL, "depth", 0) + 1
    try:
        yield
    finally:
        _LOCAL.depth -= 1


def is_paused() -> bool:
    return getattr(_LOCAL, "depth", 0) > 0


# ── the price schedule (pure) ───────────────────────────────────────────────

def _cents(price) -> int:
    return int(round(float(price) * 100))


def _usable_quote(bid, ask) -> bool:
    try:
        b, a = float(bid), float(ask)
    except (TypeError, ValueError):
        return False
    return math.isfinite(b) and math.isfinite(a) and b > 0 and a > 0 and a >= b


def _round_in_our_favour(raw_cents: float, side: str) -> int:
    """A sell rounds up, a buy rounds down, to the valid increment at that price."""
    tick = TICK_BELOW_CENTS if raw_cents < TICK_BREAK_CENTS else TICK_ABOVE_CENTS
    if side == "sell":
        return int(math.ceil(raw_cents / tick - 1e-9) * tick)
    return int(math.floor(raw_cents / tick + 1e-9) * tick)


def ladder_prices(side: str, bid, ask, steps: int) -> list:
    """The `steps` limit prices from the mid to the touch (a sell walks down to
    the bid, a buy up to the ask). Each is a valid increment, at least as
    aggressive as the one before, never past the touch or the far side; the
    last is the touch exactly. [] when the quote is not two-sided."""
    if side not in ("sell", "buy") or int(steps) < 1 or not _usable_quote(bid, ask):
        return []
    b, a = _cents(bid), _cents(ask)
    touch, far = (b, a) if side == "sell" else (a, b)
    mid = (a + b) / 2.0
    n = int(steps)
    out = []
    for k in range(n):
        if k == n - 1:
            c = touch
        else:
            c = _round_in_our_favour(mid + (touch - mid) * k / (n - 1), side)
            c = min(max(c, touch), far) if side == "sell" else max(min(c, touch), far)
        if out:
            c = min(c, out[-1]) if side == "sell" else max(c, out[-1])
        out.append(c)
    return [c / 100.0 for c in out]


def crosses_spread(side: str, limit_price, bid, ask) -> bool:
    """True when this limit is marketable at the quote: a sell at or under the
    bid, a buy at or over the ask (today's single order would cross)."""
    if not _usable_quote(bid, ask):
        return False
    p = _cents(limit_price)
    return p <= _cents(bid) if side == "sell" else p >= _cents(ask)


def plan(side: str, limit_price, bid, ask, steps: int) -> dict:
    """{"prices": [...], "bid", "ask"} for a close worth laddering, else {"skip": why}."""
    if int(steps) < 2:
        return {"skip": f"ladder configured with {steps} step(s)"}
    if not _usable_quote(bid, ask):
        return {"skip": "no two-sided quote"}
    if not crosses_spread(side, limit_price, bid, ask):
        return {"skip": f"the rule's ${float(limit_price):.2f} rests inside the spread (bid {float(bid):.2f} / ask {float(ask):.2f})"}
    prices = ladder_prices(side, bid, ask, steps)
    far = ask if side == "sell" else bid
    if not prices or _cents(prices[0]) in (_cents(prices[-1]), _cents(far)):
        return {"skip": f"no valid price strictly inside the spread (bid {float(bid):.2f} / ask {float(ask):.2f})"}
    return {"prices": prices, "bid": float(bid), "ask": float(ask)}


# ── broker / disk side, injected ────────────────────────────────────────────

class LadderIO:
    """What the step logic needs from the outside world. options_manager builds
    one per call; tests pass fakes.
      quote(occ) -> {"bid", "ask", ...} or {}
      post(occ, qty, side, price) -> _submit_close_order's result (never laddered)
      replace(order_id, price) -> (new order dict or None, detail)
      get_order(order_id) -> order dict, or None when the read failed
      cancel(order_id) -> bool
      market_ok() -> bool: regular hours and past the opening window"""

    def __init__(self, *, quote, post, replace, get_order, cancel, market_ok,
                 state_path, log_path, now=None):
        self.quote, self.post, self.replace = quote, post, replace
        self.get_order, self.cancel, self.market_ok = get_order, cancel, market_ok
        self.state_path, self.log_path = state_path, log_path
        self._now = now

    def now(self) -> float:
        return (self._now or wall_clock)()

    def load(self) -> dict:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path) as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as e:
            logger.warning(f"exit ladder state unreadable, treated as empty: {e}")
        return {}

    def save(self, state: dict) -> None:
        tmp = self.state_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, indent=1, default=str)
            os.replace(tmp, self.state_path)
        except OSError as e:
            logger.error(f"exit ladder state not saved: {e}")

    def record(self, rec: dict) -> None:
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError as e:
            logger.warning(f"exit ladder log line not written: {e}")


def _root(occ: str) -> str:
    """OCC root (the fixed-width date/type/strike suffix is 15 characters)."""
    return occ[:-15] if len(occ) > 15 else occ


def public_view(entry: dict) -> dict:
    return {"step": int(entry.get("step", 0) or 0), "of": int(entry.get("steps", 0) or 0),
            "price": entry.get("price"), "order_id": entry.get("order_id") or ""}


def _order_id(order) -> str:
    return str((order or {}).get("id") or "")


def _filled_qty(order) -> float:
    try:
        return float((order or {}).get("filled_qty") or 0)
    except (TypeError, ValueError):
        return 0.0


def _note_fill(entry: dict, order: dict, now: float) -> None:
    """Log the step a fill is first seen at (partial fills included). The
    broker's filled_qty is per order, so it is tracked per order id."""
    oid = _order_id(order) or entry.get("order_id") or ""
    seen = entry.setdefault("filled_by_order", {})
    prev, total = float(seen.get(oid, 0) or 0), _filled_qty(order)
    if total > prev:
        seen[oid] = total
        entry["filled_qty"] = sum(float(v or 0) for v in seen.values())
        entry.setdefault("fills", []).append({"step": entry.get("step"), "qty": total - prev, "at": now, "order_id": oid})


def _no_back_off(side: str, price: float, current: float) -> float:
    """A step never re-prices less aggressively than the order already working."""
    return min(price, current) if side == "sell" else max(price, current)


def _history(entry: dict, step: int, price: float, quote: dict, now: float, order_id: str, via: str) -> None:
    entry.setdefault("history", []).append({
        "step": step, "price": price, "bid": quote.get("bid"), "ask": quote.get("ask"),
        "at": now, "order_id": order_id, "via": via})


def _action(entry: dict, event: str, reason: str, status: str) -> dict:
    return {"action": "LADDER", "ticker": _root(entry["occ"]), "occ_symbol": entry["occ"],
            "type": f"exit_ladder_{event}", "reason": reason,
            "order": {"status": status, "order_id": entry.get("order_id") or "", "detail": reason,
                      "ladder": public_view(entry)}}


def _working_result(occ: str, entry: dict) -> dict:
    view = public_view(entry)
    if entry.get("pending"):
        detail = (f"exit ladder already working on {occ}: re-sending step {entry['pending'].get('step')}/{view['of']} "
                  f"next tick — no second order")
    else:
        detail = (f"exit ladder already working on {occ}: step {view['step']}/{view['of']} at "
                  f"${float(entry.get('price') or 0):.2f} — no second order")
    return {"status": "working", "order_id": view["order_id"], "detail": detail, "ladder": view}


# ── starting a ladder (called by options_manager._submit_close_order) ───────

def start(io: LadderIO, cfg: dict, occ: str, qty: int, side: str, limit_price: float) -> dict:
    """The ladder branch of a close. An active ladder on this contract answers
    "working" and sends nothing (a rule that fires again must not stack a
    second order, nor record a second exit). A close that does not cross the
    spread, or has no two-sided quote, goes out exactly as before. Otherwise
    step 1 goes out at the mid; if the broker does not accept it, the rule's
    own order goes out instead."""
    state = io.load()
    entry = state.get(occ)
    if isinstance(entry, dict) and io.now() - float(entry.get("started_at") or 0) <= MAX_LADDER_AGE_SECONDS:
        return _working_result(occ, entry)
    quote = io.quote(occ) or {}
    p = plan(side, limit_price, quote.get("bid"), quote.get("ask"), cfg.get("steps", DEFAULT_STEPS))
    if "skip" in p:
        logger.info(f"exit ladder not used for {occ} ({p['skip']}): one order at ${float(limit_price):.2f}, as before")
        return io.post(occ, qty, side, limit_price)
    prices = p["prices"]
    result = io.post(occ, qty, side, prices[0])
    if str(result.get("status", "")) != "submitted":
        logger.warning(f"exit ladder step 1 for {occ} at ${prices[0]:.2f} not accepted "
                       f"({str(result.get('detail', ''))[:120]}): sending the rule's ${float(limit_price):.2f} instead")
        return io.post(occ, qty, side, limit_price)
    order_id = str(result.get("order_id") or "")
    if not order_id:
        logger.warning(f"exit ladder step 1 for {occ} accepted without an order id: it cannot be stepped")
        return result
    now = io.now()
    entry = {"occ": occ, "side": side, "qty": int(qty), "steps": len(prices), "step": 1, "price": prices[0],
             "order_id": order_id, "first_order_id": order_id, "rule_price": float(limit_price),
             "started_at": now, "step_at": now, "pending": None, "filled_qty": 0.0}
    _history(entry, 1, prices[0], p, now, order_id, "post")
    state[occ] = entry
    io.save(state)
    touch = prices[-1]
    out = dict(result)
    out["detail"] = (f"{result.get('detail', '')} — exit ladder step 1/{len(prices)} (the mid); "
                     f"the touch ${touch:.2f} is step {len(prices)}")
    out["ladder"] = {**public_view(entry), "touch": touch, "plan": prices}
    logger.info(f"exit ladder started on {occ}: {side} {qty} at ${prices[0]:.2f}, plan {prices}")
    return out


# ── each tick (called by options_manager before the rules) ──────────────────

def pursue(io: LadderIO, cfg: dict, held: dict) -> list:
    """Advance or finish every working ladder; returns LADDER actions for the
    audit log. `held` maps each held option contract to its broker position.
    Runs whether or not the ladder is on (a switched-off ladder still finishes
    the orders it already placed); with nothing working it reads one file."""
    state = io.load()
    if not state:
        return []
    actions, changed = [], False
    gate = {}

    def market_ok() -> bool:
        if "ok" not in gate:
            gate["ok"] = bool(io.market_ok())
        return gate["ok"]

    for occ in list(state.keys()):
        entry = state[occ]
        if not isinstance(entry, dict) or entry.get("side") not in ("sell", "buy"):
            state.pop(occ, None)
            changed = True
            continue
        kind, action = _pursue_one(io, cfg, occ, entry, held.get(occ), market_ok)
        if action:
            actions.append(action)
        if kind == "drop":
            state.pop(occ, None)
            changed = True
        elif kind == "changed":
            changed = True
    if changed:
        io.save(state)
    return actions


def _pursue_one(io, cfg, occ, entry, pos, market_ok):
    now = io.now()
    if now - float(entry.get("started_at") or 0) > MAX_LADDER_AGE_SECONDS:
        return "drop", _finish(io, entry, "stale", None, now,
                               "started in an earlier session; its DAY order is gone (nothing touched)")
    oid = entry.get("order_id") or ""
    order = io.get_order(oid) if oid else None
    if order is None:
        return "keep", None                     # read failed: judged again next tick
    status = str(order.get("status") or "").lower()
    _note_fill(entry, order, now)
    if status in FILLED:
        return "drop", _finish(io, entry, "filled", order, now, "")
    if status == "replaced":
        successor = str(order.get("replaced_by") or "")
        if successor:
            entry["order_id"] = successor       # a replace whose response was lost
            return "changed", None
        return "drop", _finish(io, entry, "lost", order, now, "order replaced outside the ladder")
    if status in ENDED:
        if pos is None:
            return "drop", _finish_gone(io, entry, order, now)
        if entry.get("pending"):
            if not market_ok():
                return "keep", None
            return _resubmit(io, cfg, occ, entry, pos, now)
        return "drop", _finish(io, entry, status, order, now, "the order ended at the broker; the rule judges again")
    # the order is working (or in a transition the broker is still processing)
    if pos is None:
        io.cancel(oid)                          # a close left working without its position could open one
        return "drop", _finish(io, entry, "position_gone", order, now, "position no longer held; working order canceled")
    if entry.get("pending") or status not in REPLACEABLE + PARTIAL:
        return "keep", None                     # accepted / pending_* / our own cancel in flight
    n, step = int(entry["steps"]), int(entry["step"])
    if step >= n:
        return "keep", None                     # at the touch: it rests there, never a market order
    if now - float(entry.get("step_at") or 0) < float(cfg.get("step_seconds", DEFAULT_STEP_SECONDS)):
        return "keep", None
    if not market_ok():
        return "keep", None
    quote = io.quote(occ) or {}
    prices = ladder_prices(entry["side"], quote.get("bid"), quote.get("ask"), n)
    if not prices:
        return "keep", None                     # no two-sided quote this tick
    target = _no_back_off(entry["side"], prices[step], float(entry["price"]))
    if status in PARTIAL:
        # A partly filled order is never re-priced in place (the replacement's
        # quantity would be ambiguous): cancel it; next tick the position's
        # remaining quantity goes out at the next step.
        if io.cancel(oid):
            entry["pending"] = {"step": step + 1, "at": now, "why": "partial fill"}
            return "changed", _action(entry, "cancel_for_rest", f"exit ladder: partial fill at step {step}/{n}; "
                                      f"the rest is re-sent at step {step + 1} next tick", "cancel_requested")
        return "keep", None
    if _cents(target) == _cents(entry["price"]):
        entry["step"], entry["step_at"] = step + 1, now
        _history(entry, step + 1, target, quote, now, oid, "held")
        return "changed", _action(entry, "step", f"exit ladder step {step + 1}/{n}: limit stays "
                                  f"${target:.2f} (bid {quote.get('bid')} / ask {quote.get('ask')})", "working")
    new_order, detail = io.replace(oid, target)
    if new_order and _order_id(new_order):
        entry["order_id"] = _order_id(new_order)
        entry["step"], entry["price"], entry["step_at"] = step + 1, target, now
        _history(entry, step + 1, target, quote, now, entry["order_id"], "replace")
        return "changed", _action(entry, "step", f"exit ladder step {step + 1}/{n}: limit ${target:.2f} "
                                  f"(bid {quote.get('bid')} / ask {quote.get('ask')})", "replaced")
    # The replace was refused: find out why before doing anything else.
    again = io.get_order(oid)
    again_status = str((again or {}).get("status") or "").lower()
    if again_status in FILLED:
        _note_fill(entry, again, now)
        return "drop", _finish(io, entry, "filled", again, now, "")
    if again_status in REPLACEABLE + PARTIAL and io.cancel(oid):
        entry["pending"] = {"step": n, "at": now, "why": str(detail)[:160]}
        return "changed", _action(entry, "replace_refused", f"exit ladder: replace refused ({str(detail)[:120]}); "
                                  f"order canceled, the touch goes out next tick", "cancel_requested")
    return "keep", None                         # ended / in transition / unreadable: next tick reconciles


def _resubmit(io, cfg, occ, entry, pos, now):
    """After our own cancel landed: send the remaining position at the pending
    step's price (the touch after a refused replace). Never in the same tick as
    the cancel, so the old order cannot still fill alongside the new one."""
    try:
        qty = abs(int(float(pos.get("qty", 0) or 0)))
    except (TypeError, ValueError):
        qty = 0
    if qty <= 0:
        return "drop", _finish(io, entry, "position_gone", None, now, "nothing left to close")
    n = int(entry["steps"])
    target_step = min(max(int((entry.get("pending") or {}).get("step") or n), 1), n)
    quote = io.quote(occ) or {}
    prices = ladder_prices(entry["side"], quote.get("bid"), quote.get("ask"), n)
    if not prices:
        return "keep", None
    price = _no_back_off(entry["side"], prices[target_step - 1], float(entry["price"]))
    result = io.post(occ, qty, entry["side"], price)
    if str(result.get("status", "")) != "submitted" or not result.get("order_id"):
        return "drop", _finish(io, entry, "resubmit_rejected", None, now,
                               f"re-send refused ({str(result.get('detail', ''))[:120]}); the rule judges again")
    entry["order_id"] = str(result["order_id"])
    entry["step"], entry["price"], entry["step_at"], entry["pending"] = target_step, price, now, None
    _history(entry, target_step, price, quote, now, entry["order_id"], "resend")
    return "changed", _action(entry, "resent", f"exit ladder step {target_step}/{n}: re-sent {qty} at ${price:.2f}",
                              "submitted")


def _finish_gone(io, entry, order, now):
    """The position is gone and our current order ended unfilled. Usually a
    replace that lost the race: the previous step's order filled and its
    replacement was rejected. Check the earlier orders once each."""
    current = entry.get("order_id")
    seen = []
    for h in reversed(entry.get("history", [])):
        oid = h.get("order_id")
        if not oid or oid == current or oid in seen:
            continue
        seen.append(oid)
        earlier = io.get_order(oid)
        if earlier and _filled_qty(earlier) > 0:
            entry["step"], entry["price"], entry["order_id"] = h.get("step"), h.get("price"), oid
            _note_fill(entry, earlier, now)
            return _finish(io, entry, "filled", earlier, now, "an earlier step's order filled; its replacement was rejected")
    return _finish(io, entry, "position_gone", order, now, "position no longer held; the ladder's order did not fill")


def _finish(io, entry, outcome, order, now, note):
    """One log line and one LADDER action per finished ladder."""
    n = int(entry.get("steps", 0) or 0)
    step = int(entry.get("step", 0) or 0)
    fill_price = None
    try:
        fill_price = float((order or {}).get("filled_avg_price")) if (order or {}).get("filled_avg_price") else None
    except (TypeError, ValueError):
        fill_price = None
    seconds = max(0.0, now - float(entry.get("started_at") or now))
    rec = {"occ": entry.get("occ"), "side": entry.get("side"), "qty": entry.get("qty"), "steps": n,
           "outcome": outcome, "filled_step": step if outcome == "filled" else None,
           "fill_price": fill_price, "filled_qty": entry.get("filled_qty"), "fills": entry.get("fills", []),
           "rule_price": entry.get("rule_price"), "first_order_id": entry.get("first_order_id"),
           "last_order_id": entry.get("order_id"), "started_at": entry.get("started_at"), "finished_at": now,
           "seconds": round(seconds, 1), "history": entry.get("history", []), "note": note}
    io.record(rec)
    if outcome == "filled":
        where = "the touch" if step == n else ("the mid" if step == 1 else "between the mid and the touch")
        px = f" at ${fill_price:.2f}" if fill_price is not None else ""
        reason = f"exit ladder FILLED at step {step}/{n} ({where}){px}, {int(seconds // 60)}m {int(seconds % 60)}s after step 1"
    else:
        reason = f"exit ladder ended ({outcome}) at step {step}/{n}: {note}"
    logger.info(f"{entry.get('occ')}: {reason}")
    return _action(entry, "filled" if outcome == "filled" else "ended", reason, outcome)
