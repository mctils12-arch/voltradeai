#!/usr/bin/env python3
"""
floor_vol_managed.py — the volatility-managed SPY floor (2026-11-05). Pure
planning: no I/O, no network, no orders. bot_engine._manage_vol_managed_floor
does the I/O and the trades.

WHY: research/floor_study_2026-11-05.md. Over 1993-2026, after financing and
costs, this is the only floor design that passed the pre-registered ship rule
against buy-and-hold SPY:
- Sharpe 0.573 vs 0.513.
- Max drawdown -48.4% vs -55.2% (2008: -22.6% vs -36.8%).
- Log growth 10.41% vs 10.26%.
Leverage alone (2x above the 200-day average, or a constant 1.25x) added
return but no Sharpe.

THE RULE (the study's V1, parameters unchanged — system_config FLOOR_VM_*):
- sigma = annualised sample stdev of the last WINDOW daily simple returns of
  the base ticker (SPY), from COMPLETED sessions only. Today's partial bar is
  never used.
- Target SPY-equivalent exposure: E* = min(CAP, TARGET_VOL / sigma).
- The floor's capital is C = CAPITAL_PCT x equity.
- Up to 1x, C holds E* x C of SPY and the rest stays in cash (since
  2026-11-06 that idle cash sits in a T-bill fund, see cash_plan below).
- Above 1x, the extra exposure is held in the 2x ETF (SSO), not on margin:
  (2 - E*) x C in SPY plus (E* - 1) x C in SSO. That is still C of capital,
  exposure E* x C.
- Trade only when the current exposure is more than BAND away from E*, or
  when a legacy floor ticker is still held.
- (2026-11-06) The institutional flow model may scale E* DOWN by a multiplier
  in [1 - FLOWS_FLOOR_MAX_CUT, 1] when net equity selling is expected
  (flows_model.floor_multiplier; plan() clamps any input to [0, 1]). 1.0 is the
  study's rule unchanged; the rule's value is kept as rule_target_exposure.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

TRADING_DAYS = 252
LEVERAGE_OF_LEVERED = 2.0          # the levered ticker is a 2x daily fund


def realized_vol(closes: Sequence[float], window: int) -> Optional[float]:
    """Annualised sample stdev of the last `window` simple daily returns, the
    study's estimator (scripts/floor_study.py run_v1). None when the history is
    too short or a close is not a positive finite number."""
    if window < 2 or len(closes) < window + 1:
        return None
    tail = [float(c) for c in closes[-(window + 1):]]
    if any(not math.isfinite(c) or c <= 0 for c in tail):
        return None
    rets = [tail[i] / tail[i - 1] - 1.0 for i in range(1, len(tail))]
    mu = sum(rets) / window
    var = sum((r - mu) ** 2 for r in rets) / (window - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def target_exposure(sigma: Optional[float], target_vol: float, cap: float) -> Optional[float]:
    """E* = min(cap, target_vol / sigma); a zero-volatility history takes the cap."""
    if sigma is None:
        return None
    if sigma <= 0:
        return cap
    return min(cap, target_vol / sigma)


def holdings_for(exposure: float, capital: float) -> Dict[str, float]:
    """Dollar targets that give `exposure` x `capital` of base-ticker exposure
    using only `capital` dollars (no margin)."""
    e = max(0.0, exposure)
    if e <= 1.0:
        return {"base": e * capital, "levered": 0.0}
    e = min(e, LEVERAGE_OF_LEVERED)               # beyond 2x needs margin: never
    return {"base": (LEVERAGE_OF_LEVERED - e) * capital, "levered": (e - 1.0) * capital}


def current_exposure(base_value: float, levered_value: float, capital: float) -> float:
    if capital <= 0:
        return 0.0
    return (max(0.0, base_value) + LEVERAGE_OF_LEVERED * max(0.0, levered_value)) / capital


def bounded_multiplier(m) -> float:
    """An exposure multiplier the floor accepts: in [0, 1]. Above 1 (or not a
    number) is refused as 1.0, so no input can raise exposure or add leverage."""
    try:
        m = float(m)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(m) or m >= 1.0:
        return 1.0
    return max(0.0, m)


def plan(closes: Sequence[float], equity: float, base_value: float, levered_value: float,
         legacy_value: float, cfg: Dict, exposure_multiplier: float = 1.0) -> Dict:
    """Everything _manage_vol_managed_floor needs to decide, as plain numbers.

    `legacy_value`: market value of floor tickers this mode no longer owns (the
    old basket and GLD); any amount > 0 forces a rebalance so they are sold.
    `exposure_multiplier` (2026-11-06, the institutional flow model): scales the
    study's E* down, bounded to [0, 1] — it can only cut the target, never raise
    it above the rule, so it never adds leverage. 1.0 is the study's rule."""
    window = int(cfg.get("FLOOR_VM_WINDOW", 20))
    target_vol = float(cfg.get("FLOOR_VM_TARGET_VOL", 0.16))
    cap = float(cfg.get("FLOOR_VM_CAP", 1.5))
    band = float(cfg.get("FLOOR_VM_BAND", 0.10))
    capital = max(0.0, float(equity)) * float(cfg.get("FLOOR_VM_CAPITAL_PCT", 0.85))
    mult = bounded_multiplier(exposure_multiplier)
    sigma = realized_vol(closes, window)
    e_rule = target_exposure(sigma, target_vol, cap)
    e_now = current_exposure(base_value, levered_value, capital)
    if e_rule is None:
        return {"ok": False, "reason": f"need {window + 1} completed daily closes, have {len(closes)}",
                "sigma": None, "target_exposure": None, "rule_target_exposure": None,
                "exposure_multiplier": mult, "current_exposure": round(e_now, 4),
                "capital": round(capital, 2), "rebalance": False}
    e_star = e_rule * mult
    drift = abs(e_star - e_now)
    legacy = legacy_value > 0
    targets = holdings_for(e_star, capital)
    return {
        "ok": True,
        "sigma": round(sigma, 6),
        "rule_target_exposure": round(e_rule, 4),
        "exposure_multiplier": mult,
        "target_exposure": round(e_star, 4),
        "current_exposure": round(e_now, 4),
        "drift": round(drift, 4),
        "band": band,
        "capital": round(capital, 2),
        "target_base_dollars": round(targets["base"], 2),
        "target_levered_dollars": round(targets["levered"], 2),
        "legacy_to_sell": legacy,
        "rebalance": bool(drift > band or legacy),
    }


def park_sleeve_enabled(cfg: Dict, env: Optional[Dict] = None) -> bool:
    """Whether the T-bill fund also holds the satellites' UNUSED cash (True,
    system_config FLOOR_VM_CASH_PARK_SLEEVE) or only the floor's idle capital,
    as before 2026-11-19. Env VOLTRADE_PARK_SLEEVE=0|1 overrides without a
    deploy; any other value is ignored."""
    import os
    v = str((os.environ if env is None else env).get("VOLTRADE_PARK_SLEEVE") or "").strip().lower()
    if v in ("0", "false", "off", "no"):
        return False
    if v in ("1", "true", "on", "yes"):
        return True
    return bool(cfg.get("FLOOR_VM_CASH_PARK_SLEEVE", True))


def parking_inputs(equity: float, short_put_collateral: float, cfg: Dict, env: Optional[Dict] = None) -> Dict:
    """What the cash step may park and what must stay in cash.

    Before 2026-11-19 ("floor"): the fund held only the floor's idle capital
    (FLOOR_VM_CAPITAL_PCT x equity minus SPY/SSO) and account cash stayed >=
    the satellites' whole sleeve ((1 - FLOOR_VM_CAPITAL_PCT) x equity) + every
    short put's collateral. On 2026-11-19 that sleeve was $15,815 of plain
    cash, 18% of the account, earning nothing while the satellites held no
    position at all.

    Now ("sleeve"): the fund may hold everything that is not the core, a small
    cash buffer (FLOOR_VM_CASH_BUFFER_PCT x equity) or short-put collateral.
    The satellites' budget is unchanged: the cash reserve reads the fund as
    cash (position_mgmt, CASH_RESERVE_PARK_IS_CASH), the exposure caps leave
    it out (bot_engine.without_floor_cash), and the step sells the fund as
    soon as cash falls under buffer + collateral, so a trade that needs the
    money gets it by selling the park, not by keeping cash idle for it."""
    equity = max(0.0, float(equity or 0.0))
    collateral = max(0.0, float(short_put_collateral or 0.0))
    capital_pct = float(cfg.get("FLOOR_VM_CAPITAL_PCT", 0.85))
    capital = equity * capital_pct
    if not park_sleeve_enabled(cfg, env):
        return {"mode": "floor", "capital": capital, "idle_capital": capital,
                "reserve": (1.0 - capital_pct) * equity + collateral, "buffer": (1.0 - capital_pct) * equity}
    buffer = equity * max(0.0, min(1.0 - capital_pct, float(cfg.get("FLOOR_VM_CASH_BUFFER_PCT", 0.02))))
    # idle_capital leaves out the buffer only: put collateral is kept in cash
    # through `reserve` (cash_plan's room), so the plan's reason names it
    return {"mode": "sleeve", "capital": capital, "idle_capital": max(capital, equity - buffer),
            "reserve": buffer + collateral, "buffer": buffer}


def cash_plan(capital: float, core_value: float, held_value: float, cash: Optional[float],
              reserve: float, cfg: Dict, enabled: bool = True, buys_allowed: bool = True,
              idle_capital: Optional[float] = None) -> Dict:
    """Where the floor's idle capital sits (2026-11-06): in the T-bill fund
    FLOOR_VM_CASH_TICKER, not in cash. The study credited it with the T-bill
    rate; Alpaca paper credits no interest and no dividends, so the fund must
    return its yield in price (system_config FLOOR_VM_CASH_* has the numbers).

    - idle = capital - core_value: the floor's capital not in SPY/SSO, never
      below 0.
    - room = held_value + cash - reserve: the most the fund can hold while
      account cash stays >= reserve. With cash unknown, room = held_value
      (no buy on a guess).
    - target = min(idle, room), traded only when |target - held_value| >
      band = max(FLOOR_VM_CASH_MIN_TRADE_DOLLARS, FLOOR_VM_CASH_BAND x
      capital).
    - Sells any time, so cash comes first. Buys only when buys_allowed (the
      engine passes False while the core is trading).
    - enabled=False sells everything held."""
    held = max(0.0, float(held_value or 0.0))
    band = max(float(cfg.get("FLOOR_VM_CASH_MIN_TRADE_DOLLARS", 2000)),
               float(cfg.get("FLOOR_VM_CASH_BAND", 0.05)) * max(0.0, float(capital)))
    out = {"enabled": bool(enabled), "held": round(held, 2), "band": round(band, 2),
           "idle": None, "room": None, "target": None, "gap": None,
           "action": "hold", "dollars": 0.0, "sell_all": False, "reason": ""}
    if not enabled:
        if held > 0:
            out.update(action="sell", dollars=round(held, 2), sell_all=True, target=0.0, gap=round(-held, 2),
                       reason="cash parking switched off: the fund goes back to cash")
        else:
            out["reason"] = "cash parking switched off, nothing held"
        return out
    if capital <= 0:
        out["reason"] = "no floor capital (equity unreadable)"
        return out
    # idle_capital (2026-11-19, parking_inputs "sleeve"): what the fund may hold
    # at most, when that is more than the floor's own capital. The band stays on
    # the floor's capital either way.
    whole = float(capital) if idle_capital is None else max(float(capital), float(idle_capital))
    idle = max(0.0, whole - max(0.0, float(core_value)))
    room = held if cash is None else max(0.0, held + float(cash) - max(0.0, float(reserve)))
    target = min(idle, room)
    gap = target - held
    out.update(idle=round(idle, 2), room=round(room, 2), target=round(target, 2), gap=round(gap, 2))
    if gap < -band:
        why = ("account cash below its reserve" if room < idle else "the floor needs the capital back")
        out.update(action="sell", dollars=round(-gap, 2), reason=f"{why}: hold ${target:,.0f}, not ${held:,.0f}")
    elif gap > band and buys_allowed:
        out.update(action="buy", dollars=round(gap, 2),
                   reason=(f"${idle:,.0f} of the floor's ${capital:,.0f} is idle; park ${target:,.0f} (cash would earn 0)"
                           if whole <= float(capital) else
                           f"${idle:,.0f} outside the core, the cash buffer and put collateral is idle; "
                           f"park ${target:,.0f} (cash would earn 0)"))
    elif gap > band:
        out["reason"] = "buy waits: the core traded this pass or an order is working"
    elif cash is None and idle > held + band:
        out["reason"] = "account cash unreadable: no buy"
    else:
        out["reason"] = "within the band of its target"
    return out
