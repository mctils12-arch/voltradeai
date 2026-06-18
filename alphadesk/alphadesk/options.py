"""
AlphaDesk — options analytics + defined-risk strategy expression.

Pure functions (no network): realized volatility from a close series, the
implied ~1-sigma expected move over a horizon, and `build_strategy`, which turns
the equity verdict + an options snapshot into a single defined-risk structure
(credit/debit spread or iron condor) with strikes anchored to the expected move.

Education only — this never sizes positions or places orders, and every
structure is risk-defined by construction.
"""
from __future__ import annotations

import math
from typing import List, Optional

from .models import OptionsSnapshot, OptionsStrategy

_TRADING_DAYS = 252


def realized_vol(closes: List[float], window: int = 30) -> Optional[float]:
    """Annualized realized volatility from daily closes (stdev of daily log
    returns over the last `window` days, scaled by sqrt(252)). None if there
    aren't enough bars."""
    if len(closes) < window + 1:
        return None
    rets = [math.log(closes[i] / closes[i - 1])
            for i in range(len(closes) - window, len(closes))
            if closes[i - 1] > 0 and closes[i] > 0]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return round(math.sqrt(var) * math.sqrt(_TRADING_DAYS), 4)


def expected_move(vol_annual: float, days: int = 30) -> float:
    """~1-sigma move over `days`, as a fraction of spot, from an annualized vol."""
    return round(vol_annual * math.sqrt(days / 365.0), 4)


def _direction(verdict: str) -> str:
    v = verdict.lower()
    if "buy" in v:
        return "bullish"
    if "sell" in v:
        return "bearish"
    return "neutral"


def build_strategy(verdict: str, price: Optional[float], snap: OptionsSnapshot) -> OptionsStrategy:
    """Map verdict + IV regime + expected move to one defined-risk structure.

    Premium-rich regimes (high IV rank) favor *selling* premium (credit
    spreads); cheap regimes favor *buying* it (debit spreads). Strikes are placed
    ~1 expected move out of the money. Neutral verdicts get an iron condor when
    premium is rich, otherwise a stand-aside.
    """
    direction = _direction(verdict)
    em = snap.expected_move_30d or 0.05
    ivr = snap.iv_rank if snap.iv_rank is not None else 50.0
    rich = ivr >= 50  # elevated implied vol → sell premium

    def strike(mult: float) -> str:
        if not price:
            return f"~{mult:+.0%} from spot"
        return f"~${price * (1 + mult * em):.0f}"

    if direction == "neutral":
        if rich:
            return OptionsStrategy(
                structure="Iron condor",
                direction="neutral",
                rationale=(f"No directional edge and IV rank is elevated ({ivr:.0f}); "
                           "sell a range-bound, defined-risk condor to harvest premium."),
                legs=[
                    f"Sell call {strike(+1)}, buy call {strike(+2)} (upper wing)",
                    f"Sell put {strike(-1)}, buy put {strike(-2)} (lower wing)",
                ],
                max_risk_note="Max loss = wing width − net credit. Profits if price stays within ~1 expected move.",
            )
        return OptionsStrategy(
            structure="No options trade",
            direction="neutral",
            rationale=(f"Directional edge is unclear and IV rank is low ({ivr:.0f}); neither "
                       "buying nor selling premium is attractive. Stand aside."),
            legs=[],
            max_risk_note="No position.",
        )

    bullish = direction == "bullish"
    if bullish and rich:
        structure, verb = "Bull put credit spread", "sell premium below spot"
        legs = [f"Sell put {strike(-1)}", f"Buy put {strike(-2)} (protection)"]
        risk = "Max loss = strike width − net credit; max gain = net credit if price holds above the short put."
    elif bullish and not rich:
        structure, verb = "Bull call debit spread", "buy cheap upside"
        legs = [f"Buy call {strike(+0.25)} (near spot)", f"Sell call {strike(+1.5)}"]
        risk = "Max loss = net debit paid; max gain = strike width − net debit."
    elif not bullish and rich:
        structure, verb = "Bear call credit spread", "sell premium above spot"
        legs = [f"Sell call {strike(+1)}", f"Buy call {strike(+2)} (protection)"]
        risk = "Max loss = strike width − net credit; max gain = net credit if price stays below the short call."
    else:
        structure, verb = "Bear put debit spread", "buy cheap downside"
        legs = [f"Buy put {strike(-0.25)} (near spot)", f"Sell put {strike(-1.5)}"]
        risk = "Max loss = net debit paid; max gain = strike width − net debit."

    return OptionsStrategy(
        structure=structure,
        direction=direction,
        rationale=(f"Verdict is {direction}; IV rank {ivr:.0f} ({'rich' if rich else 'cheap'}) → "
                   f"{verb}. Strikes anchored to a ~{em:.0%} expected move."),
        legs=legs,
        max_risk_note=risk + " Defined-risk; education only.",
    )
