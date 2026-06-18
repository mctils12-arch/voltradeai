"""
AlphaDesk — pre-trade planner (position sizing + risk management).

Pure functions (no network). Turns an entry idea into a disciplined plan:
  - a volatility-based stop-loss suggestion (ATR-style, from realized vol),
  - position size from your account value and risk-per-trade,
  - price targets at R-multiples (1R/2R/3R) plus an optional manual target,
  - reward:risk and account-exposure checks.

This is the "before you buy" workflow: it PLANS the trade — stop, size, targets —
so you can place it yourself in your broker. It does not execute anything and is
not personalized investment advice; it's a risk-management calculator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, floor
from typing import List, Optional, Sequence

TRADING_DAYS = 252


@dataclass
class Target:
    r: float                # R-multiple (reward/risk)
    price: float
    gain_per_share: float
    total_gain: float       # over the planned share count
    gain_pct: float


@dataclass
class TradePlan:
    ticker: str
    direction: str          # "long" | "short"
    entry: float
    stop: float
    stop_pct: float         # signed, vs entry
    per_share_risk: float
    risk_amount: float      # dollars at risk if stopped out
    risk_pct: float
    shares: int
    position_value: float
    account_pct: float      # position value / account value
    reward_risk: Optional[float]   # at the primary (2R or manual) target
    targets: List[Target] = field(default_factory=list)
    daily_vol_pct: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def daily_vol_from_annual(hv_annual: Optional[float]) -> Optional[float]:
    """Annualized realized vol (e.g. 0.40) -> ~daily move (e.g. 0.025)."""
    if not hv_annual or hv_annual <= 0:
        return None
    return hv_annual / sqrt(TRADING_DAYS)


def suggest_stop(entry: float, daily_vol_pct: float, atr_mult: float = 2.0,
                 direction: str = "long") -> float:
    """Volatility-based stop: `atr_mult` daily moves away from entry."""
    dist = entry * daily_vol_pct * atr_mult
    return round(entry - dist if direction == "long" else entry + dist, 2)


def build_plan(ticker: str, entry: float, account_value: float, risk_pct: float, *,
               stop: Optional[float] = None, stop_atr_mult: float = 2.0,
               daily_vol_pct: Optional[float] = None, direction: str = "long",
               r_multiples: Sequence[float] = (1.0, 2.0, 3.0),
               target_price: Optional[float] = None) -> TradePlan:
    direction = "short" if str(direction).lower() == "short" else "long"
    notes: List[str] = []
    warnings: List[str] = []

    # 1) Stop — manual if given, else volatility-based, else a default %.
    if stop is None:
        if daily_vol_pct:
            stop = suggest_stop(entry, daily_vol_pct, stop_atr_mult, direction)
            notes.append(f"Stop suggested at {stop_atr_mult:.1f}x the ~{daily_vol_pct*100:.1f}% "
                         f"daily move (volatility-based). Adjust to your own plan.")
        else:
            stop = round(entry * (0.92 if direction == "long" else 1.08), 2)
            notes.append("No volatility data available — used a default 8% stop. Set your own.")

    per_share_risk = round(abs(entry - stop), 2)
    valid_stop = per_share_risk > 0 and (
        (direction == "long" and stop < entry) or (direction == "short" and stop > entry))
    if not valid_stop:
        warnings.append("Stop is on the wrong side of entry (or equal) — fix the stop to size the trade.")

    # 2) Position size from risk budget.
    risk_amount = round(account_value * risk_pct, 2)
    shares = int(floor(risk_amount / per_share_risk)) if per_share_risk > 0 else 0
    position_value = round(shares * entry, 2)
    account_pct = round(position_value / account_value, 4) if account_value > 0 else 0.0
    if account_value > 0 and position_value > account_value:
        warnings.append("Position exceeds account value — needs margin. Lower risk %, or use a wider stop.")
    stop_pct = round((stop - entry) / entry, 4) if entry else 0.0

    # 3) Targets at R-multiples (+ optional manual target).
    targets: List[Target] = []

    def make_target(r: float) -> Target:
        price = round(entry + (per_share_risk * r if direction == "long" else -per_share_risk * r), 2)
        gps = round(abs(price - entry), 2)
        return Target(r=round(r, 2), price=price, gain_per_share=gps,
                      total_gain=round(gps * shares, 2),
                      gain_pct=round((price - entry) / entry, 4) if entry else 0.0)

    for r in r_multiples:
        targets.append(make_target(r))

    reward_risk: Optional[float] = None
    if target_price is not None and per_share_risk > 0:
        r = round(abs(target_price - entry) / per_share_risk, 2)
        gps = round(abs(target_price - entry), 2)
        targets.append(Target(r=r, price=round(target_price, 2), gain_per_share=gps,
                              total_gain=round(gps * shares, 2),
                              gain_pct=round((target_price - entry) / entry, 4) if entry else 0.0))
        reward_risk = r
    else:
        # default "primary" target is the 2R level if present.
        twoR = next((t for t in targets if abs(t.r - 2.0) < 1e-9), None)
        reward_risk = twoR.r if twoR else (targets[0].r if targets else None)

    return TradePlan(
        ticker=ticker.upper(), direction=direction, entry=round(entry, 2),
        stop=round(stop, 2), stop_pct=stop_pct, per_share_risk=per_share_risk,
        risk_amount=risk_amount, risk_pct=round(risk_pct, 4), shares=shares,
        position_value=position_value, account_pct=account_pct, reward_risk=reward_risk,
        targets=targets, daily_vol_pct=round(daily_vol_pct, 4) if daily_vol_pct else None,
        notes=notes, warnings=warnings,
    )


def plan_to_dict(p: TradePlan) -> dict:
    return {
        "ticker": p.ticker, "direction": p.direction, "entry": p.entry,
        "stop": p.stop, "stop_pct": p.stop_pct, "per_share_risk": p.per_share_risk,
        "risk_amount": p.risk_amount, "risk_pct": p.risk_pct, "shares": p.shares,
        "position_value": p.position_value, "account_pct": p.account_pct,
        "reward_risk": p.reward_risk, "daily_vol_pct": p.daily_vol_pct,
        "targets": [{"r": t.r, "price": t.price, "gain_per_share": t.gain_per_share,
                     "total_gain": t.total_gain, "gain_pct": t.gain_pct} for t in p.targets],
        "notes": p.notes, "warnings": p.warnings,
    }
