#!/usr/bin/env python3
"""
position_mgmt.py — portfolio-level position management (refocus spec section 4,
human 2026-11-05: "cash reserve buffer that rises in risk-off regimes; capital
rotation ...; drawdown rules: portfolio-level max drawdown de-risking").

One source of truth for three rules that every satellite entry path reads.
Pure functions plus two small file touches (the floor's own plan file and the
equity-peak file, read; the decision log, appended). No orders are placed here.

1. CASH RESERVE — cash_reserve_pct() / satellite_headroom() / fit_entry()
   A cash buffer of CASH_RESERVE_PCT_<REGIME> x equity (system_config: BULL 5%,
   NEUTRAL 8%, CAUTION 15%, BEAR 25%, PANIC 35%). The room left for NEW
   satellite capital is

       equity x (1 - reserve%) - floor commitment - satellite commitments
              - working opening orders - caller-supplied pending dollars

   * floor commitment = max(the floor's market value, the floor's current
     target). The floor is its own sleeve (FLOOR_VM_CAPITAL_PCT of equity under
     "vol_managed_spy"): satellites may not spend cash the floor's own plan is
     about to deploy, and the reserve never asks the floor to sell. Cash the
     floor's volatility rule leaves idle DOES count toward the reserve.
   * satellite commitments = stock/ETF market value + long option premium +
     short option collateral (a cash-secured put is strike x 100 x qty; a
     vertical is its width; a call covered by shares is 0; an uncovered call is
     strike x 100 x qty). A short put counts at its collateral, not its premium:
     that is the cash it obliges the account to spend.
   * The regime engine's risk budget (regime_engine.risk_budget_for, when it
     applies) acts on the satellites' room above the floor only, as a reserve:
     reserve = max(label reserve, (1 - budget) x room above the floor) — max,
     never a sum or a product, and never a second cut on per-trade sizing.
   * The reserve only ever BLOCKS or TRIMS a new entry. It never sells:
     satellites give up room first and the floor is never force-sold. When the
     reserve is not met (the floor grew, or the regime turned risk-off) the
     shortfall is reported and new satellite entries stay blocked until
     satellite exits restore it.

2. DRAWDOWN RISK BUDGET — dd_risk_scalar() / combine_drawdown_scalars()
   Between DD_RISK_BUDGET_START_PCT (10%) and DD_RISK_BUDGET_END_PCT (18%, the
   Python entry halt) satellite sizing scales linearly from 1.0x to
   DD_RISK_BUDGET_MIN_SCALAR (0.5x). Where a sizer already carried a stepped
   drawdown ladder (position_sizing 5/10/15%, Tier 1 10/15%) the TIGHTER of the
   two applies, so no path is looser than before; DD_RISK_BUDGET_REPLACES_LADDERS
   = True makes the linear budget the only rule.

3. CAPITAL ROTATION — plan_rotations()
   Swap a held satellite for a candidate that could not get capital, when the
   held one has run its course (near or past its target, momentum fading) or is
   a loser, and the candidate's expected value beats it after trading costs and
   tax. Every planned and every rejected pair is logged with plain-English
   reasons; server/bot.ts runUpgradeCandidates executes a plan through the
   existing exit and entry calls.

Every decision lands in DATA_DIR/voltrade_position_mgmt.jsonl, one object per
line: {"ts", "kind": "cash_reserve"|"dd_budget"|"rotation", "action",
"ticker", "summary" (plain English), ...numbers}.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

try:
    from storage_config import DATA_DIR, POSITION_MGMT_LOG_PATH
except ImportError:  # older storage_config without the log path
    from storage_config import DATA_DIR
    POSITION_MGMT_LOG_PATH = os.path.join(DATA_DIR, "voltrade_position_mgmt.jsonl")

logger = logging.getLogger("voltrade.position_mgmt")

REGIMES = ("BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC")
DEFAULT_REGIME = "NEUTRAL"           # the label every other module falls back to
LOG_MAX_BYTES = 5_000_000            # rotated to .1 above this (one generation kept)
FLOOR_VM_STATE_FILE = "voltrade_floor_vm_state.json"   # = bot_engine.FLOOR_VM_STATE_FILE
FLOOR_PLAN_MAX_AGE_S = 4 * 86_400    # an older plan is ignored (covers a weekend + holiday)
EQUITY_PEAK_FILE = "voltrade_equity_peak.json"         # server/bot.ts saveEquityPeak
SCALE_IN_STATE_FILE = "voltrade_scale_in_state.json"   # server/scaleIn.ts SCALE_IN_STATE_FILE


def _cfg(cfg: Optional[dict] = None) -> dict:
    if cfg is not None:
        return cfg
    from system_config import BASE_CONFIG
    return BASE_CONFIG


def _f(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


# ═══════════════════════════════════════════════════════════════════════════
#  DECISION LOG
# ═══════════════════════════════════════════════════════════════════════════

_dedup_seen: Dict[str, float] = {}


def log_decision(kind: str, action: str, ticker: str, summary: str,
                 dedup_key: Optional[str] = None, dedup_seconds: float = 0.0,
                 path: Optional[str] = None, **fields) -> bool:
    """Append one decision row; True when written. dedup_key/dedup_seconds: the
    same key is written at most once per window in this process (a scan repeats
    every few minutes; the log keeps the first)."""
    now = time.time()
    if dedup_key and dedup_seconds > 0:
        last = _dedup_seen.get(dedup_key)
        if last is not None and now - last < dedup_seconds:
            return False
        _dedup_seen[dedup_key] = now
        if len(_dedup_seen) > 5000:
            for k in sorted(_dedup_seen, key=_dedup_seen.get)[:2500]:
                _dedup_seen.pop(k, None)
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "kind": kind, "action": action, "ticker": str(ticker or "").upper(),
           "summary": summary}
    row.update(fields)
    path = path or POSITION_MGMT_LOG_PATH
    try:
        if os.path.exists(path) and os.path.getsize(path) > LOG_MAX_BYTES:
            os.replace(path, path + ".1")
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return True
    except OSError as e:
        logger.warning(f"position_mgmt log not written ({kind}/{action} {ticker}): {e}")
        return False


def recent_decisions(kind: str, action: Optional[str] = None, since_seconds: float = 86_400,
                     path: Optional[str] = None, max_bytes: int = 200_000) -> List[dict]:
    """Rows of `kind` (and `action`) newer than since_seconds, from the log's tail."""
    path = path or POSITION_MGMT_LOG_PATH
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            tail = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    cutoff = time.time() - since_seconds
    out = []
    for line in tail:
        row = _json_row(line)
        if not row or row.get("kind") != kind or (action and row.get("action") != action):
            continue
        try:
            ts = datetime.fromisoformat(str(row.get("ts"))).timestamp()
        except ValueError:
            ts = 0.0
        if ts >= cutoff:
            out.append(row)
    return out


def _json_row(line: str) -> Optional[dict]:
    """One log line as a dict; None for the torn first line a tail-seek produces."""
    try:
        row = json.loads(line)
    except ValueError:
        return None
    return row if isinstance(row, dict) else None


# ═══════════════════════════════════════════════════════════════════════════
#  POSITION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def parse_occ(symbol: str) -> Optional[dict]:
    """End-anchored OCC parse (options_manager._parse_occ_symbol's rule: the
    15-char date/type/strike suffix is fixed width, the root is the rest)."""
    s = str(symbol or "").upper()
    if len(s) < 16:
        return None
    root, date, typ, strike = s[:-15], s[-15:-9], s[-9], s[-8:]
    if not root or not date.isdigit() or typ not in ("C", "P") or not strike.isdigit():
        return None
    return {"underlying": root, "expiry": date, "type": typ, "strike": int(strike) / 1000.0}


def is_option_position(p: dict) -> bool:
    return (p or {}).get("asset_class") in ("us_option", "option") or parse_occ((p or {}).get("symbol", "")) is not None


def floor_tickers(cfg: Optional[dict] = None) -> set:
    """Every ticker the floor sleeve holds or is still selling: under
    "vol_managed_spy" the SPY/SSO pair plus the legacy basket + GLD it sells;
    under "basket" the basket members, FLOOR_TICKER and the defensive ticker."""
    c = _cfg(cfg)
    out = {c.get("FLOOR_TICKER", "QQQ"), c.get("DEFENSIVE_FLOOR_TICKER", "GLD")}
    out |= {t for t in (c.get("FLOOR_BASKET") or {}) if t != "CASH"}
    if c.get("FLOOR_MODE", "basket") == "vol_managed_spy":
        out |= {c.get("FLOOR_VM_TICKER", "SPY"), c.get("FLOOR_VM_LEVERED_TICKER", "SSO")}
    return {str(t).upper() for t in out if t}


def option_commitments(positions: Iterable[dict]) -> Dict[str, float]:
    """Cash each option position obliges the account to spend, by symbol.

    Grouped by (underlying, expiry, type). Each short leg pairs with the nearest
    protective long leg (puts: a long strike below it; calls: a long strike above
    it): the paired quantity commits the width. An unpaired short put commits
    strike x 100; an unpaired short call commits 0 while 100 shares per contract
    of the underlying are held, else strike x 100. A long leg's unpaired part
    commits its premium (market value)."""
    positions = [p for p in positions if isinstance(p, dict)]
    shares: Dict[str, float] = {}
    for p in positions:
        if not is_option_position(p) and _f(p.get("qty")) > 0:
            sym = str(p.get("symbol", "")).upper()
            shares[sym] = shares.get(sym, 0.0) + _f(p.get("qty"))
    groups: Dict[tuple, List[dict]] = {}
    for p in positions:
        occ = parse_occ(p.get("symbol", ""))
        if occ is None:
            continue
        groups.setdefault((occ["underlying"], occ["expiry"], occ["type"]), []).append({
            "symbol": str(p.get("symbol", "")).upper(), "strike": occ["strike"],
            "qty": _f(p.get("qty")), "mv": abs(_f(p.get("market_value"))),
        })
    out: Dict[str, float] = {}
    covered_left = {u: n / 100.0 for u, n in shares.items()}
    for (underlying, _exp, typ), legs in sorted(groups.items()):
        longs = [dict(leg, free=leg["qty"]) for leg in legs if leg["qty"] > 0]
        shorts = sorted((leg for leg in legs if leg["qty"] < 0), key=lambda leg: leg["strike"],
                        reverse=(typ == "P"))
        for s in shorts:
            need = abs(s["qty"])
            commit = 0.0
            protect = sorted((lg for lg in longs if lg["free"] > 0 and
                              (lg["strike"] < s["strike"] if typ == "P" else lg["strike"] > s["strike"])),
                             key=lambda lg: abs(lg["strike"] - s["strike"]))
            for lg in protect:
                if need <= 0:
                    break
                m = min(need, lg["free"])
                commit += abs(s["strike"] - lg["strike"]) * 100.0 * m
                lg["free"] -= m
                need -= m
            if need > 0 and typ == "C" and covered_left.get(underlying, 0.0) > 0:
                m = min(need, covered_left[underlying])
                covered_left[underlying] -= m
                need -= m
            if need > 0:
                commit += s["strike"] * 100.0 * need
            out[s["symbol"]] = out.get(s["symbol"], 0.0) + commit
        for lg in longs:
            frac = lg["free"] / lg["qty"] if lg["qty"] > 0 else 0.0
            out[lg["symbol"]] = out.get(lg["symbol"], 0.0) + lg["mv"] * frac
    return out


def open_order_commitments(orders: Iterable[dict], positions: Iterable[dict],
                           skip: Iterable[str] = ()) -> float:
    """Dollars still working in OPENING orders: an equity buy (remaining qty x
    limit), a sell-to-open put (strike x 100 x remaining), a buy-to-open option
    (limit x 100 x remaining). Orders that close a held position count 0, and so
    do symbols in `skip` (the floor: its target already covers its orders)."""
    held = {str(p.get("symbol", "")).upper(): _f(p.get("qty")) for p in (positions or []) if isinstance(p, dict)}
    skip = {str(s).upper() for s in skip}
    total = 0.0
    for o in orders or []:
        if not isinstance(o, dict):
            continue
        sym = str(o.get("symbol", "")).upper()
        side = str(o.get("side", "")).lower()
        remaining = max(0.0, _f(o.get("qty")) - _f(o.get("filled_qty")))
        if remaining <= 0 or sym in skip:
            continue
        px = _f(o.get("limit_price")) or (_f(o.get("notional")) / remaining)
        occ = parse_occ(sym)
        have = held.get(sym, 0.0)
        if occ is None:
            if side == "buy" and have >= 0:
                total += remaining * px
        elif side == "sell" and have <= 0 and occ["type"] == "P":
            total += occ["strike"] * 100.0 * remaining
        elif side == "buy" and have >= 0:
            total += px * 100.0 * remaining
    return total


# ═══════════════════════════════════════════════════════════════════════════
#  1. CASH RESERVE
# ═══════════════════════════════════════════════════════════════════════════

def normalize_regime(regime) -> Tuple[str, bool]:
    """(label, known). Unknown labels fall back to NEUTRAL, the system default."""
    r = str(regime or "").upper().strip()
    if r in REGIMES:
        return r, True
    return DEFAULT_REGIME, False


def cash_reserve_pct(regime, cfg: Optional[dict] = None) -> float:
    c = _cfg(cfg)
    if not c.get("CASH_RESERVE_ENABLED", True):
        return 0.0
    label, _known = normalize_regime(regime)
    return max(0.0, min(0.95, _f(c.get(f"CASH_RESERVE_PCT_{label}", 0.0))))


def floor_plan_target_ratio(path: Optional[str] = None, now: Optional[float] = None) -> Optional[float]:
    """The vol-managed floor's latest target as a fraction of equity
    ((target SPY $ + target SSO $) / the plan's equity), or None without a
    usable plan from the last FLOOR_PLAN_MAX_AGE_S. Read from the file
    bot_engine._manage_vol_managed_floor writes on every scan."""
    path = path or os.path.join(DATA_DIR, FLOOR_VM_STATE_FILE)
    try:
        with open(path) as f:
            st = json.load(f)
        age = (now or time.time()) - datetime.fromisoformat(str(st.get("at"))).timestamp()
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if not isinstance(st, dict) or not st.get("ok") or age > FLOOR_PLAN_MAX_AGE_S or age < -3600:
        return None
    eq = _f(st.get("equity"))
    tgt = _f(st.get("target_base_dollars")) + _f(st.get("target_levered_dollars"))
    if eq <= 0 or tgt < 0:
        return None
    return tgt / eq


def floor_target_dollars(equity: float, regime: str, cfg: Optional[dict] = None,
                         plan_ratio: Optional[float] = None) -> Tuple[float, str]:
    """(dollars, plain-English source). vol_managed_spy: the floor's latest plan
    scaled to today's equity, else its whole capital sleeve
    (FLOOR_VM_CAPITAL_PCT) while no plan exists. basket (the rollback): the
    regime's floor share (SPY_FLOOR_<R> x the invested basket weight +
    DEFENSIVE_FLOOR_<R>)."""
    c = _cfg(cfg)
    equity = max(0.0, _f(equity))
    if c.get("FLOOR_MODE", "basket") == "vol_managed_spy":
        cap = _f(c.get("FLOOR_VM_CAPITAL_PCT", 0.85))
        if plan_ratio is not None:
            return equity * max(0.0, min(plan_ratio, cap)), "the floor's own latest plan"
        return equity * cap, "the floor's whole capital sleeve (no recent plan on file)"
    label, _ = normalize_regime(regime)
    basket = c.get("FLOOR_BASKET") or {}
    invested = 1.0 - _f(basket.get("CASH", 0.0)) if c.get("FLOOR_BASKET_ENABLED", False) else 1.0
    spy = _f(c.get(f"SPY_FLOOR_{label}", 0.0)) * invested
    gld = _f(c.get(f"DEFENSIVE_FLOOR_{label}", 0.0)) if c.get("DEFENSIVE_FLOOR_ENABLED", False) else 0.0
    return equity * (spy + gld), f"the basket floor's {label} share"


def regime_budget() -> Optional[dict]:
    """The regime engine's risk budget when it applies now (regime_engine.
    risk_budget_for(): a view {"budget" 0..1, "band", "summary", ...}, or None
    while the engine / VOLTRADE_REGIME_BUDGET is off or its state is stale).
    Defensive: an engine without that function (its change not merged yet) or a
    failing call is no budget, i.e. 1.0 — the reserve then behaves as before."""
    try:
        import regime_engine as _re
    except ImportError:
        return None
    fn = getattr(_re, "risk_budget_for", None)
    if not callable(fn):
        return None
    try:
        rb = fn()
    except Exception as e:  # its contract is never-raises; logged, not trusted
        logger.warning(f"regime risk budget unavailable, cash reserve uses its label only: {e}")
        return None
    if not isinstance(rb, dict) or rb.get("budget") is None:
        return None
    return rb


def satellite_headroom(equity: float, positions: Iterable[dict], regime,
                       cfg: Optional[dict] = None, open_orders: Iterable[dict] = (),
                       pending_dollars: float = 0.0,
                       floor_plan_ratio: Optional[float] = None,
                       risk_budget: Optional[dict] = None) -> dict:
    """The cash-reserve snapshot every entry path applies. Never raises on bad
    rows. "enabled": False means callers do not gate at all (one-setting off).

    risk_budget (regime_budget()'s view, 2026-11-06 regime engine): applied to
    the satellites' total room above the floor only, as a reserve —
    reserve = max(the label's cash reserve, (1 - budget) x room above the
    floor). max, never a sum or a product, and never on the floor itself."""
    c = _cfg(cfg)
    equity = max(0.0, _f(equity))
    label, known = normalize_regime(regime)
    enabled = bool(c.get("CASH_RESERVE_ENABLED", True))
    pct = cash_reserve_pct(label, c)
    positions = [p for p in (positions or []) if isinstance(p, dict)]
    floors = floor_tickers(c)
    floor_mv = sum(abs(_f(p.get("market_value"))) for p in positions
                   if str(p.get("symbol", "")).upper() in floors and not is_option_position(p))
    tgt, tgt_source = floor_target_dollars(equity, label, c, floor_plan_ratio)
    floor_commit = max(floor_mv, tgt)
    opt = option_commitments(positions)
    sat_commit = 0.0
    for p in positions:
        sym = str(p.get("symbol", "")).upper()
        if is_option_position(p):
            sat_commit += opt.get(sym, 0.0)
        elif sym not in floors:
            sat_commit += abs(_f(p.get("market_value")))
    working = open_order_commitments(open_orders, positions, skip=floors)
    extra = max(0.0, _f(pending_dollars))
    label_reserve = equity * pct
    room_above_floor = max(0.0, equity - floor_commit)
    budget = None
    budget_reserve = None
    if enabled and isinstance(risk_budget, dict) and risk_budget.get("budget") is not None:
        budget = max(0.0, min(1.0, _f(risk_budget.get("budget"), 1.0)))
        budget_reserve = (1.0 - budget) * room_above_floor
    reserve = max(label_reserve, budget_reserve or 0.0)
    binding = "risk_budget" if budget_reserve is not None and budget_reserve > label_reserve else "label"
    if binding == "risk_budget":
        why = str(risk_budget.get("summary") or "").strip()
        reserve_reason = (f"the regime risk budget x{budget:.2f} lets satellites use only {budget * 100:.0f}% of the "
                          f"${room_above_floor:,.0f} room above the floor" + (f" ({why})" if why else ""))
    else:
        reserve_reason = f"the {label} market's cash reserve is {pct * 100:.0f}% of equity"
    cash_after = equity - floor_commit - sat_commit - working - extra
    headroom = max(0.0, cash_after - reserve)
    shortfall = max(0.0, reserve - cash_after)

    def share(v: float) -> float:
        return v / equity * 100.0 if equity > 0 else 0.0

    if not enabled:
        summary = "The cash reserve is switched off (CASH_RESERVE_ENABLED=False): entries are not gated by it."
    elif headroom > 0:
        summary = (f"{reserve_reason[0].upper()}{reserve_reason[1:]}, so ${reserve:,.0f} ({share(reserve):.1f}%) stays "
                   f"in cash. The floor holds or is about to hold ${floor_commit:,.0f} ({share(floor_commit):.0f}%), "
                   f"satellites use ${sat_commit:,.0f} ({share(sat_commit):.1f}%) and working orders "
                   f"${working + extra:,.0f}; ${headroom:,.0f} ({share(headroom):.1f}%) is left for new satellite positions.")
    else:
        summary = (f"{reserve_reason[0].upper()}{reserve_reason[1:]}, so ${reserve:,.0f} ({share(reserve):.1f}%) stays "
                   f"in cash, and the floor (${floor_commit:,.0f}), satellites (${sat_commit:,.0f}) and working orders "
                   f"(${working + extra:,.0f}) already use that room, so no new satellite positions until "
                   f"satellites exit (short by ${shortfall:,.0f}; the floor is never sold for the reserve).")
    if not known:
        summary += f" Regime '{regime}' was not recognised, so the {DEFAULT_REGIME} reserve was used."
    return {
        "enabled": enabled,
        "regime": label,
        "regime_known": known,
        "reserve_pct": round(pct, 4),
        "reserve_dollars": round(reserve, 2),
        "label_reserve_dollars": round(label_reserve, 2),
        "room_above_floor": round(room_above_floor, 2),
        "risk_budget": None if budget is None else round(budget, 4),
        "risk_budget_reserve_dollars": None if budget_reserve is None else round(budget_reserve, 2),
        "risk_budget_summary": (risk_budget or {}).get("summary") if budget is not None else None,
        "reserve_binding": binding,
        "reserve_reason": reserve_reason,
        "equity": round(equity, 2),
        "floor_market_value": round(floor_mv, 2),
        "floor_target_dollars": round(tgt, 2),
        "floor_target_source": tgt_source,
        "floor_committed": round(floor_commit, 2),
        "satellite_committed": round(sat_commit, 2),
        "pending_orders": round(working, 2),
        "pending_other": round(extra, 2),
        "cash_after_commitments": round(cash_after, 2),
        "headroom_dollars": round(headroom, 2),
        "headroom_pct": round(share(headroom), 3),
        "reserve_shortfall": round(shortfall, 2),
        "min_entry_pct": _f(c.get("CASH_RESERVE_MIN_ENTRY_PCT", 0.01)),
        "summary": summary,
    }


def fit_entry(position_value: float, price: float, headroom_dollars: float,
              equity: float, cfg: Optional[dict] = None) -> dict:
    """Fit one new entry into the reserve headroom:
    {"action": "allow"|"trim"|"block", "value", "shares", "reason"}. A trim
    keeps whole shares; a trim below CASH_RESERVE_MIN_ENTRY_PCT of equity (the
    sizer's own floor, position_sizing.ABSOLUTE_MIN_POSITION_PCT) blocks."""
    c = _cfg(cfg)
    position_value, price, headroom = _f(position_value), _f(price), _f(headroom_dollars)
    min_value = max(0.0, _f(equity)) * _f(c.get("CASH_RESERVE_MIN_ENTRY_PCT", 0.01))
    if position_value <= headroom + 0.005:
        return {"action": "allow", "value": round(position_value, 2),
                "shares": int(position_value // price) if price > 0 else None,
                "reason": f"fits inside the cash reserve (${headroom:,.0f} of room)"}
    if headroom <= 0 or price <= 0:
        return {"action": "block", "value": 0.0, "shares": 0,
                "reason": "the cash reserve leaves no room for a new satellite position"}
    shares = int(math.floor(headroom / price))
    value = shares * price
    if shares < 1 or value < min_value:
        return {"action": "block", "value": 0.0, "shares": 0,
                "reason": (f"only ${headroom:,.0f} of room is left above the cash reserve, below the "
                           f"${max(min_value, price):,.0f} smallest worthwhile position")}
    return {"action": "trim", "value": round(value, 2), "shares": shares,
            "reason": (f"cut from ${position_value:,.0f} to ${value:,.0f} so the cash reserve stays whole "
                       f"(${headroom:,.0f} of room)")}


def log_reserve_decision(decision: dict, ticker: str, wanted_value: float, snapshot: dict,
                         source: str, score=None) -> bool:
    """One log row for a trim or block (allows are not logged). Deduplicated per
    source/ticker/action for 6 hours (scans repeat every few minutes)."""
    action = (decision or {}).get("action")
    if action not in ("trim", "block"):
        return False
    why = str(snapshot.get("reserve_reason") or f"the {snapshot.get('regime')} market's cash reserve is "
              f"{_f(snapshot.get('reserve_pct')) * 100:.0f}% of equity")
    summary = f"{str(ticker).upper()}: {decision.get('reason')}, because {why}."
    return log_decision("cash_reserve", action, ticker, summary,
                        dedup_key=f"cr|{source}|{str(ticker).upper()}|{action}", dedup_seconds=6 * 3600,
                        source=source, wanted_value=round(_f(wanted_value), 2),
                        allowed_value=round(_f(decision.get("value")), 2),
                        headroom_dollars=snapshot.get("headroom_dollars"),
                        reserve_pct=snapshot.get("reserve_pct"), regime=snapshot.get("regime"),
                        reserve_binding=snapshot.get("reserve_binding"), risk_budget=snapshot.get("risk_budget"),
                        score=score)


def current_regime_label(macro: Optional[dict] = None) -> str:
    """The 5-level label scan_market's own gates use (the macro snapshot's
    "regime", itself system_config.get_market_regime; recomputed when absent);
    NEUTRAL, logged, when the snapshot cannot be read."""
    try:
        if macro is None:
            from macro_data import get_macro_snapshot
            macro = get_macro_snapshot() or {}
        label, known = normalize_regime(macro.get("regime"))
        if known:
            return label
        from system_config import get_market_regime
        return get_market_regime(
            _f(macro.get("vxx_ratio", 1.0), 1.0) or 1.0,
            _f(macro.get("spy_vs_ma50", 1.0), 1.0) or 1.0,
            spy_below_200_days=int(_f(macro.get("spy_below_200_days", 0))),
            spy_above_200d=bool(macro.get("spy_above_200d", True)),
        )
    except Exception as e:  # macro feed down: the system default label, logged
        logger.warning(f"cash reserve regime unavailable, using {DEFAULT_REGIME}: {e}")
        return DEFAULT_REGIME


def pending_scale_in_dollars(path: Optional[str] = None) -> float:
    """Dollars of scale-in tranches not yet bought (server/scaleIn.ts plans:
    status "active", tranche status "pending", qty x the first fill's price).
    The reserve holds their room so later entries cannot spend it. A missing or
    unreadable file is 0."""
    path = path or os.path.join(DATA_DIR, SCALE_IN_STATE_FILE)
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        return 0.0
    plans = state.get("plans") if isinstance(state, dict) else None
    total = 0.0
    for plan in (plans.values() if isinstance(plans, dict) else []):
        if not isinstance(plan, dict) or plan.get("status") != "active":
            continue
        for t in plan.get("tranches") or []:
            if isinstance(t, dict) and t.get("status") == "pending":
                total += max(0.0, _f(t.get("qty"))) * max(0.0, _f(plan.get("ref_price")))
    return total


def live_headroom(equity: float, positions: Iterable[dict], regime, open_orders: Iterable[dict] = (),
                  pending_dollars: Optional[float] = None, source: str = "", cfg: Optional[dict] = None) -> dict:
    """satellite_headroom with its live inputs read here — the floor's latest
    plan, the scale-in tranches still to buy (when pending_dollars is None) and
    the regime engine's risk budget — and, when the budget applies, one
    counterfactual row (regime_engine.log_counterfactual kind "sizing", key
    "satellite_reserve": the label-only reserve vs the budgeted one)."""
    if pending_dollars is None:
        pending_dollars = pending_scale_in_dollars()
    rb = regime_budget()
    snap = satellite_headroom(equity, positions, regime, cfg=cfg, open_orders=open_orders,
                              pending_dollars=pending_dollars, floor_plan_ratio=floor_plan_target_ratio(),
                              risk_budget=rb)
    if rb is not None and snap.get("enabled"):
        legacy_headroom = max(0.0, _f(snap.get("cash_after_commitments")) - _f(snap.get("label_reserve_dollars")))
        cut = _f(snap.get("headroom_dollars")) / legacy_headroom if legacy_headroom > 0 else 1.0
        try:
            import regime_engine as _re
            _re.log_counterfactual(
                "sizing", "satellite_reserve",
                {"reserve_dollars": snap.get("label_reserve_dollars"), "headroom_dollars": round(legacy_headroom, 2),
                 "multiplier": 1.0},
                {"reserve_dollars": snap.get("reserve_dollars"), "headroom_dollars": snap.get("headroom_dollars"),
                 "budget": snap.get("risk_budget"), "binding": snap.get("reserve_binding"), "applied": True,
                 "multiplier": round(cut, 4)},
                source or "position_mgmt.live_headroom", sleeve="satellite_reserve")
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"satellite_reserve counterfactual not logged: {e}")
    return snap


def snapshot_rpc(equity=None, positions=None, open_orders=None, regime=None,
                 pending_dollars=None) -> dict:
    """Daemon route "cash_reserve_snapshot": server/bot.ts calls it with fresh
    positions + open orders before each entry loop. Regime: the caller's label,
    else the scan's own (macro snapshot)."""
    if not regime:
        regime = current_regime_label()
    return live_headroom(_f(equity), positions or [], regime, open_orders=open_orders or [],
                         pending_dollars=None if pending_dollars is None else _f(pending_dollars),
                         source="server/bot.ts cash_reserve_snapshot")


# ═══════════════════════════════════════════════════════════════════════════
#  2. DRAWDOWN RISK BUDGET
# ═══════════════════════════════════════════════════════════════════════════

def current_drawdown_pct(equity: float, path: Optional[str] = None) -> Optional[float]:
    """Account drawdown from the peak Node records (server/bot.ts saveEquityPeak
    key "equityPeak"; legacy "peak_equity"), as a positive percent. None when no
    peak is on file."""
    path = path or os.path.join(DATA_DIR, EQUITY_PEAK_FILE)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    peak = _f(data.get("equityPeak", data.get("peak_equity")))
    equity = _f(equity)
    if peak <= 0 or equity <= 0:
        return None
    return max(0.0, (peak - equity) / peak * 100.0)


def dd_risk_scalar(dd_pct: Optional[float], cfg: Optional[dict] = None) -> float:
    """1.0 at or below START, linear down to MIN at END, MIN beyond END. A
    drawdown passed as a negative number is read as its size."""
    c = _cfg(cfg)
    if not c.get("DD_RISK_BUDGET_ENABLED", True) or dd_pct is None:
        return 1.0
    start = _f(c.get("DD_RISK_BUDGET_START_PCT", 10.0))
    end = _f(c.get("DD_RISK_BUDGET_END_PCT", 18.0))
    floor = max(0.0, min(1.0, _f(c.get("DD_RISK_BUDGET_MIN_SCALAR", 0.5))))
    dd = abs(_f(dd_pct))
    if dd <= start:
        return 1.0
    if dd >= end or end <= start:
        return floor
    return round(1.0 - (1.0 - floor) * (dd - start) / (end - start), 4)


def combine_drawdown_scalars(ladder: float, budget: float, cfg: Optional[dict] = None) -> Tuple[float, str]:
    """The one drawdown multiplier a sizer applies: the tighter of its older
    stepped ladder and the linear budget ("ladder" or "budget" names the one
    that bound), or the budget alone under DD_RISK_BUDGET_REPLACES_LADDERS."""
    c = _cfg(cfg)
    if c.get("DD_RISK_BUDGET_REPLACES_LADDERS", False):
        return budget, "budget"
    if ladder <= budget:
        return ladder, "ladder"
    return budget, "budget"


def log_dd_budget(ticker: str, source: str, dd_pct: Optional[float], scalar: float,
                  rule: str, ladder: Optional[float] = None) -> bool:
    """One row per source/ticker per day while drawdown cuts a satellite size."""
    if dd_pct is None or scalar >= 1.0:
        return False
    which = "the older drawdown ladder" if rule == "ladder" else "the linear drawdown budget"
    summary = (f"{str(ticker).upper()}: the account is {dd_pct:.1f}% below its peak, so this new satellite "
               f"position is sized x{scalar:.2f} ({which} is the tighter rule).")
    return log_decision("dd_budget", "scale", ticker, summary,
                        dedup_key=f"dd|{source}|{str(ticker).upper()}", dedup_seconds=24 * 3600,
                        source=source, dd_pct=round(dd_pct, 2), scalar=scalar, rule=rule, ladder=ladder)


# ═══════════════════════════════════════════════════════════════════════════
#  3. CAPITAL ROTATION
# ═══════════════════════════════════════════════════════════════════════════

TaxFn = Callable[[float, float], float]   # (gain $, holding days) -> tax $ (negative = saved)


def default_tax_fn(cfg: Optional[dict] = None) -> TaxFn:
    """Tax at the after-tax benchmark's rates (server/taxLedger.ts
    DEFAULT_TAX_RATES, mirrored as ROTATION_TAX_RATE_* in system_config). A
    loss saves tax at the same rate, assumed to offset other gains of its term."""
    c = _cfg(cfg)
    st = _f(c.get("ROTATION_TAX_RATE_SHORT_TERM", 0.24))
    lt = _f(c.get("ROTATION_TAX_RATE_LONG_TERM", 0.15))

    def tax(gain: float, days: float) -> float:
        return gain * (lt if days > 365 else st)
    return tax


def resolve_tax_fn(cfg: Optional[dict] = None) -> Tuple[TaxFn, str]:
    """(tax function, its name for the log). tax_filter.estimate_tax(gain,
    holding_days) when that module is present; otherwise default_tax_fn. The
    comparison takes any (gain, days) -> tax callable, so the plug is one name."""
    c = _cfg(cfg)
    # tax_filter.py is FIRST-party, built by a sibling change that may land
    # before or after this one: loaded by name so its absence is not read as an
    # undeclared third-party import (scripts/program_status.sh D6).
    try:
        tax_filter = importlib.import_module("tax_filter")
    except ImportError:
        tax_filter = None
    fn = getattr(tax_filter, "estimate_tax", None) if tax_filter is not None else None
    if callable(fn):
        return (lambda gain, days: _f(fn(gain, days))), "tax_filter.estimate_tax"
    st = _f(c.get("ROTATION_TAX_RATE_SHORT_TERM", 0.24)) * 100
    lt = _f(c.get("ROTATION_TAX_RATE_LONG_TERM", 0.15)) * 100
    return default_tax_fn(c), f"estimated at {st:.0f}% short-term / {lt:.0f}% long-term"


def bucket_expectancy_pct(bucket: str, hist_stats: dict) -> Tuple[float, dict]:
    """Realized average result per trade of a sizing bucket, in percent:
    win_rate x avg_win - (1 - win_rate) x avg_loss — position_sizing's own
    stats, the numbers its Kelly sizing already trusts."""
    by = (hist_stats or {}).get("by_strategy") or {}
    s = by.get(bucket) or (hist_stats or {}).get("overall") or {}
    wr, aw, al = _f(s.get("win_rate"), 0.5), _f(s.get("avg_win")), _f(s.get("avg_loss"))
    return wr * aw - (1.0 - wr) * al, {"bucket": bucket, "win_rate": wr, "avg_win": aw, "avg_loss": al,
                                       "trades": s.get("trades", s.get("total_trades"))}


def _days_held(row: dict, now: datetime) -> Optional[float]:
    raw = (row or {}).get("entry_date")
    if not raw:
        return None
    try:
        d = datetime.strptime(str(raw)[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return max(0.0, (now.replace(tzinfo=None) - d).total_seconds() / 86_400.0)


def classify_held(position: dict, stop_row: dict, now: datetime, cfg: Optional[dict] = None) -> dict:
    """{"state": "run_its_course"|"loser"|"working"|"ineligible", "reason", ...}.

    Target width = manage_positions' take-profit (max(4, min(3 x ATR%, 15))).
    run_its_course: P&L >= ROTATION_NEAR_TARGET_FRAC of the target (or peak
    >= 2R) AND >= ROTATION_FADE_ATR_MULT x ATR off its high. loser: below cost.
    Under ROTATION_MIN_HOLD_DAYS (the stop's own dormant window): ineligible."""
    c = _cfg(cfg)
    row = stop_row or {}
    cur = _f(position.get("current_price"))
    pnl = _f(position.get("unrealized_plpc")) * 100.0
    atr = _f(row.get("current_stop_pct"), 2.0) or 2.0
    tp = max(4.0, min(atr * 3.0, 15.0))
    hi = max(_f(row.get("highest_price"), cur), cur)
    pullback = (hi - cur) / hi * 100.0 if hi > 0 else 0.0
    risk = _f(row.get("initial_risk_pct"), atr * 2.0) or atr * 2.0
    peak_r = max(_f(row.get("highest_pnl")), pnl) / risk if risk > 0 else 0.0
    days = _days_held(row, now)
    out = {"pnl_pct": round(pnl, 2), "days_held": None if days is None else round(days, 1),
           "target_pct": round(tp, 2), "pullback_pct": round(pullback, 2), "atr_pct": round(atr, 2),
           "peak_r": round(peak_r, 2)}
    min_hold = _f(c.get("ROTATION_MIN_HOLD_DAYS", 4))
    if days is None:
        return {**out, "state": "ineligible",
                "reason": "its entry date is not on file, so its minimum hold cannot be checked"}
    if days < min_hold:
        return {**out, "state": "ineligible",
                "reason": f"it has been held {days:.1f} days, under the {min_hold:.0f}-day minimum hold"}
    near = pnl >= _f(c.get("ROTATION_NEAR_TARGET_FRAC", 0.8)) * tp or peak_r >= 2.0
    fading = pullback >= _f(c.get("ROTATION_FADE_ATR_MULT", 1.0)) * atr
    if near and fading:
        return {**out, "state": "run_its_course",
                "reason": (f"it is up {pnl:.1f}% against a {tp:.1f}% target (peak {peak_r:.1f}R) and has slipped "
                           f"{pullback:.1f}% off its high, so its momentum is fading")}
    if pnl < 0:
        return {**out, "state": "loser", "reason": f"it is down {abs(pnl):.1f}% after {days:.0f} days"}
    if near:
        return {**out, "state": "working",
                "reason": f"it is near its target and still within one ATR of its high ({pullback:.1f}% off)"}
    return {**out, "state": "working", "reason": f"it is up {pnl:.1f}%, below its target, momentum intact"}


def rotation_value(held_value: float, buy_value: float, embedded_gain: float, days_held: float,
                   ev_hold_pct: float, ev_cand_pct: float, cost_pct_per_side: float,
                   horizon_days: float, tax_fn: TaxFn) -> dict:
    """After-tax dollars at the horizon from HOLD vs ROTATE, both sold at the
    horizon (the symmetric framing of server/afterTaxBenchmark.ts).
      hold:   V(1+r_h) - tax(G + V r_h, d + H) - sell cost
      rotate: sell all V now (cost, tax(G, d) now), buy B = min(proceeds,
              buy_value) of the candidate (cost), keep the rest in cash;
              at H: B'(1+r_c) - tax(B' r_c, H) - sell cost + cash."""
    v = max(0.0, _f(held_value))
    c = max(0.0, _f(cost_pct_per_side)) / 100.0
    hold_gain = v * ev_hold_pct / 100.0
    w_hold = (v + hold_gain) * (1.0 - c) - tax_fn(embedded_gain + hold_gain, days_held + horizon_days)
    proceeds = v * (1.0 - c)
    tax_now = tax_fn(embedded_gain, days_held)
    spend = min(proceeds, max(0.0, _f(buy_value)))
    invested = spend * (1.0 - c)
    cand_gain = invested * ev_cand_pct / 100.0
    w_rot = (proceeds - spend) + (invested + cand_gain) * (1.0 - c) - tax_fn(cand_gain, horizon_days) - tax_now
    gain = w_rot - w_hold
    return {"w_hold": round(w_hold, 2), "w_rotate": round(w_rot, 2), "gain_dollars": round(gain, 2),
            "gain_pct": round(gain / v * 100.0, 3) if v > 0 else 0.0, "tax_now": round(tax_now, 2),
            "cost_dollars": round(v * c + spend * c, 2), "buy_dollars": round(spend, 2)}


def _held_bucket(symbol: str) -> str:
    """The sizing bucket a held ticker trades in (position_sizing._infer_strategy:
    a known leveraged ETF is "etf", anything else with no live scores is "stocks")."""
    try:
        from position_sizing import _infer_strategy
    except ImportError:
        return "stocks"
    return _infer_strategy({"ticker": symbol})


def plan_rotations(held_positions: Iterable[dict], stop_state: dict, candidates: Iterable[dict],
                   hist_stats: dict, cfg: Optional[dict] = None, tax_fn: Optional[TaxFn] = None,
                   tax_source: str = "", now: Optional[datetime] = None) -> dict:
    """Pair held satellites with candidates that could not get capital.

    held_positions: the account's Alpaca position rows (the floor and options
    are never rotated). stop_state: voltrade_stop_state.json. candidates:
    {"ticker", "score", "bucket", "wanted_value", "buy_ticker", "buy_price",
    "instrument"}. Returns {"rotations": [...], "rejected": [...], "considered",
    "tax_source", "enabled"}; logs every planned and rejected pair."""
    c = _cfg(cfg)
    now = now or datetime.now()
    out = {"rotations": [], "rejected": [], "considered": 0, "tax_source": tax_source,
           "enabled": bool(c.get("ROTATION_ENABLED", True))}
    cands = []
    for cand in candidates or []:
        if isinstance(cand, dict) and cand.get("ticker"):
            ev, stats = bucket_expectancy_pct(str(cand.get("bucket") or "stocks"), hist_stats)
            cands.append((cand, ev, stats))
    if not out["enabled"] or not cands:
        return out
    if tax_fn is None:
        tax_fn, tax_source = resolve_tax_fn(c)
        out["tax_source"] = tax_source
    positions = [p for p in (held_positions or []) if isinstance(p, dict)]
    floors = floor_tickers(c)
    short_calls = set()
    for p in positions:
        occ = parse_occ(p.get("symbol", ""))
        if occ and occ["type"] == "C" and _f(p.get("qty")) < 0:
            short_calls.add(occ["underlying"])
    horizon = _f(c.get("ROTATION_HORIZON_DAYS", 10)) or 10.0
    cost = _f(c.get("ROTATION_COST_PCT_PER_SIDE", 0.10))
    hurdle = _f(c.get("ROTATION_MIN_EDGE_PCT", 0.25))
    max_n = int(_f(c.get("ROTATION_MAX_PER_SCAN", 1)))

    def reject(held_t, cand_t, reason, **nums):
        out["rejected"].append({"sell": held_t, "buy": cand_t, "reason": reason, **nums})
        tail = f" instead of rotating into {cand_t}" if cand_t else ""
        log_decision("rotation", "rejected", held_t or cand_t or "", f"Kept {held_t}{tail}: {reason}.",
                     dedup_key=f"rot|{held_t}|{cand_t}|{reason[:48]}", dedup_seconds=6 * 3600,
                     sell=held_t, buy=cand_t, **nums)

    eligible = []
    for p in positions:
        sym = str(p.get("symbol", "")).upper()
        if sym in floors or is_option_position(p):
            continue  # the floor and options are never rotation sells (nothing to log)
        if str(p.get("side", "long")).lower() != "long" or _f(p.get("qty")) <= 0:
            reject(sym, None, "only long stock/ETF satellites are rotated")
            continue
        if sym in short_calls:
            reject(sym, None, "a covered call is written against it, and selling the shares would leave the call naked")
            continue
        info = classify_held(p, (stop_state or {}).get(sym) or {}, now, c)
        if info["state"] == "ineligible":
            reject(sym, None, info["reason"], pnl_pct=info.get("pnl_pct"), days_held=info.get("days_held"))
            continue
        if info["state"] == "working":
            continue  # a working position is not a rotation sell
        eligible.append((p, info))
    cands.sort(key=lambda x: (x[1], _f(x[0].get("score"))), reverse=True)
    eligible.sort(key=lambda x: x[1]["pnl_pct"])          # the weakest holding first
    used = set()
    for p, info in eligible:
        if len(out["rotations"]) >= max_n:
            break
        sym = str(p.get("symbol", "")).upper()
        bucket_ev, _held_stats = bucket_expectancy_pct(_held_bucket(sym), hist_stats)
        days = _f(info.get("days_held"))
        if info["state"] == "run_its_course":
            ev_hold = 0.0
            hold_basis = "its planned upside is spent, so no remaining edge is assumed"
        else:
            left = max(0.0, horizon - days)
            ev_hold = bucket_ev * left / horizon
            hold_basis = (f"its bucket's average trade ({bucket_ev:+.2f}%) scaled to the {left:.0f} of "
                          f"{horizon:.0f} planned days left")
        value = abs(_f(p.get("market_value")))
        basis = _f(p.get("cost_basis"), value - _f(p.get("unrealized_pl")))
        embedded = value - basis
        for cand, ev_c, c_stats in cands:
            ct = str(cand.get("ticker")).upper()
            if ct in used or ct == sym:
                continue
            out["considered"] += 1
            if ev_c <= 0:
                reject(sym, ct, (f"{ct}'s sizing bucket ({c_stats['bucket']}) has lost money on average "
                                 f"({ev_c:+.2f}% a trade)"), ev_candidate_pct=round(ev_c, 3))
                continue
            want = _f(cand.get("wanted_value"), value) or value
            val = rotation_value(value, want, embedded, days, ev_hold, ev_c, cost, horizon, tax_fn)
            nums = {"ev_hold_pct": round(ev_hold, 3), "ev_candidate_pct": round(ev_c, 3),
                    "gain_pct": val["gain_pct"], "gain_dollars": val["gain_dollars"],
                    "tax_now": val["tax_now"], "cost_dollars": val["cost_dollars"],
                    "hurdle_pct": hurdle, "held_state": info["state"], "tax_source": out["tax_source"]}
            if val["gain_pct"] < hurdle:
                reject(sym, ct, (f"rotating would change the after-tax result by {val['gain_pct']:+.2f}% after "
                                 f"costs and tax, under the {hurdle:.2f}% hurdle"), **nums)
                continue
            tax_word = "a tax bill" if val["tax_now"] > 0 else "a tax saving"
            summary = (f"Sell {sym} ({info['reason']}; {hold_basis}) and buy {ct} (its bucket averages "
                       f"{ev_c:+.2f}% a trade): {val['gain_pct']:+.2f}% better after ${val['cost_dollars']:,.0f} "
                       f"of trading costs and {tax_word} of ${abs(val['tax_now']):,.0f} now "
                       f"({out['tax_source']}). These are estimates, not measurements.")
            out["rotations"].append({
                "sell": {"ticker": sym, "qty": abs(_f(p.get("qty"))), "market_value": round(value, 2),
                         "pnl_pct": info["pnl_pct"], "state": info["state"]},
                "buy": {"ticker": str(cand.get("buy_ticker") or ct).upper(), "underlying": ct,
                        "price": _f(cand.get("buy_price") or cand.get("price")),
                        "score": cand.get("score"), "instrument": cand.get("instrument", "stock"),
                        "value": val["buy_dollars"]},
                "summary": summary, **nums,
            })
            log_decision("rotation", "planned", sym, summary, dedup_key=f"rotp|{sym}|{ct}",
                         dedup_seconds=3600, sell=sym, buy=ct, **nums)
            used.add(ct)
            break
    return out
