"""
VolTradeAI — Node's kill latch, read by Python before an OPENING order (2026-11-12)
================================================================================
KNOWN BROKEN #51 (research/open_questions.md): on 2026-11-05 a Tier-2 scan that
began at 17:28:11Z, in an unlatched minute, bought GLD 35 @ $376.60 at
17:33:15Z — 33 seconds after Node's -20% drawdown kill re-tripped. Node
re-checks its latch (state.killSwitch) at the top of each dispatch loop, but
the order paths that submit from PYTHON — the floor rebalance, the floor's
cash parking, the SVXY third leg, the options opens the dispatcher scripts
and evaluate_and_execute submit — never read it, so a scan that started
unlatched kept opening positions after the latch.

ONE reader of the file Node persists on every latch transition
(server/bot.ts KILL_SWITCH_PATH = /data/voltrade/voltrade_kill_switch.json,
fallback /tmp/voltrade_kill_switch.json; record {killSwitch: bool, reason,
savedAt, lastAutoResumeAt}, parsed exactly as server/killSwitchAutoResume.ts
parseKillSwitchRecord does). The policy is server/haltScope.ts's: a latch
blocks NEW ENTRIES only — protective exits, reduce-only floor sells and
cancels never ask this module.

  * check_opening_allowed(symbol, what) is called immediately before an
    opening order leaves Python (a buy, a sell-to-open, an option open). It
    returns while the latch is off and raises EntryBlocked while it is set.
  * EntryBlocked subclasses broker_clock.OrderBlocked, so every call site's
    existing "blocked -> nothing sent, reported as a failed order" handler
    covers it, and its message starts with broker_clock.ORDER_BLOCKED_PREFIX
    like the clock's.
  * No file, or no usable record, reads as NOT latched — exactly what Node
    itself boots with (loadKillSwitch returns false) — and the reason is
    logged. A corrupt file must not freeze the floor; Node owns the file.
  * The file is read on every call (a few reads per scan, a tiny JSON): no
    cache, so a latch set one second ago is seen by the next order.

Node's own order paths re-check state.killSwitch in memory before they POST;
the dispatcher scripts return BLOCKED_ORDER_STATUS (server/haltScope.ts
KILL_BLOCKED_ORDER_STATUS, same string) so bot.ts never falls back to a stock
buy for a blocked option open.
"""
import json
import logging
import os

import broker_clock
from storage_config import DATA_DIR

logger = logging.getLogger("kill_latch")

KILL_SWITCH_FILE = "voltrade_kill_switch.json"
FALLBACK_DIR = "/tmp"
ENTRY_BLOCKED_TAG = "ENTRY-BLOCKED-KILL"
# The order status a Python open returns to bot.ts when the latch blocked it.
# Pinned equal to server/haltScope.ts KILL_BLOCKED_ORDER_STATUS by test_kill_latch.py.
BLOCKED_ORDER_STATUS = "blocked_kill_latch"


class EntryBlocked(broker_clock.OrderBlocked):
    """Raised instead of sending an OPENING order while Node's kill latch is
    set. An OrderBlocked (a RuntimeError), so the call sites' existing
    handlers treat it as "not sent"; the message starts with
    broker_clock.ORDER_BLOCKED_PREFIX."""


def latch_paths() -> tuple:
    """The file Node writes, then its fallback — the same two paths as
    server/bot.ts (one path when DATA_DIR is /tmp, the local fallback)."""
    primary = os.path.join(DATA_DIR, KILL_SWITCH_FILE)
    fallback = os.path.join(FALLBACK_DIR, KILL_SWITCH_FILE)
    return (primary,) if primary == fallback else (primary, fallback)


def read_latch() -> dict:
    """{"latched": bool, "reason": str|None, "saved_at": str|None,
    "source": path|None, "error": str|None}. The first readable file with a
    boolean killSwitch wins, as in bot.ts loadKillSwitch; none -> not latched,
    with every failure named in `error` (and logged)."""
    errors = []
    for path in latch_paths():
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:
            errors.append(f"{path}: {type(e).__name__}: {str(e)[:120]}")
            continue
        if isinstance(raw, dict) and isinstance(raw.get("killSwitch"), bool):
            return {
                "latched": raw["killSwitch"],
                "reason": raw.get("reason") if isinstance(raw.get("reason"), str) else None,
                "saved_at": raw.get("savedAt") if isinstance(raw.get("savedAt"), str) else None,
                "source": path,
                "error": None,
            }
        errors.append(f"{path}: no boolean killSwitch field")
    error = "; ".join(errors) or None
    if error:
        logger.warning(f"[KILL-LATCH] unreadable latch file(s), treated as not latched like Node does: {error}")
    return {"latched": False, "reason": None, "saved_at": None, "source": None, "error": error}


def is_latched() -> bool:
    return bool(read_latch()["latched"])


def check_opening_allowed(symbol, what: str = "opening order") -> None:
    """Call immediately before an OPENING order leaves Python. Returns while
    Node's kill latch is off; raises EntryBlocked while it is set. Every block
    is logged (the human's 2026-11-05 rule for the floor: "log every action
    taken while tripped")."""
    st = read_latch()
    if not st["latched"]:
        return
    msg = (f"{broker_clock.ORDER_BLOCKED_PREFIX}: Node kill latch set "
           f"({st['reason'] or 'no reason recorded'}, since {st['saved_at'] or '?'}) — "
           f"{what} {symbol or '?'} not sent; new entries are blocked while the latch holds, "
           f"protective exits keep running")
    logger.warning(f"[{ENTRY_BLOCKED_TAG}] {msg}")
    raise EntryBlocked(msg)


def blocked_order(exc: Exception, **extra) -> dict:
    """The order-shaped dict a Python open returns to bot.ts when the latch
    blocked it: status BLOCKED_ORDER_STATUS (never "error", which bot.ts
    treats as "fall back to a stock buy")."""
    out = {"status": BLOCKED_ORDER_STATUS, "detail": str(exc)[:200], "reason": str(exc)[:200]}
    out.update(extra)
    return out
