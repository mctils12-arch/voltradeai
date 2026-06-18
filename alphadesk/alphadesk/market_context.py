"""
AlphaDesk — top-down market context from live data.

Derives the shared market backdrop (the "Market Context" pillar) from real price
series instead of sample values:

  - spx_above_200dma     : SPY's last close vs its 200-day simple moving average
  - breadth_pct_above_50dma : fraction of the 11 SPDR sector ETFs above their own
                              50-day SMA — a cheap, robust breadth proxy
  - vix                  : best-effort via Polygon's index snapshot
  - trend_label          : risk_on / neutral / risk_off, derived from the above

Yields and credit spreads have no keyless live source wired, so they fall back
to sample values (the engine already tolerates partial fills). The math here is
pure (operates on already-fetched close series) so it's unit-checkable offline;
only `LiveProvider.market_context` touches the network.
"""
from __future__ import annotations

from typing import List, Optional

# The 11 S&P sector SPDRs — their breadth tracks the broad market closely and
# costs only 11 daily-bar requests.
SECTOR_ETFS = ("XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC")


def sma(closes: List[float], window: int) -> Optional[float]:
    if len(closes) < window or window <= 0:
        return None
    return sum(closes[-window:]) / window


def above_sma(closes: List[float], window: int) -> Optional[bool]:
    """True if the latest close is above its `window`-day SMA; None if there
    aren't enough bars to judge."""
    avg = sma(closes, window)
    if avg is None or not closes:
        return None
    return closes[-1] > avg


def breadth_fraction(series: List[List[float]], window: int = 50) -> Optional[float]:
    """Fraction of the given close-series that are above their own `window`-SMA.
    Series with too few bars are skipped; returns None if none qualify."""
    judged = [above_sma(s, window) for s in series]
    judged = [b for b in judged if b is not None]
    if not judged:
        return None
    return sum(1 for b in judged if b) / len(judged)


def derive_trend(spx_above: Optional[bool],
                 breadth: Optional[float],
                 vix: Optional[float]) -> str:
    """Combine the available live signals into a risk_on / neutral / risk_off
    label. Each signal votes +1 (risk-on) / -1 (risk-off); the sign of the sum
    decides. Unknown signals simply don't vote."""
    score = 0
    if spx_above is not None:
        score += 1 if spx_above else -1
    if breadth is not None:
        score += 1 if breadth >= 0.6 else (-1 if breadth <= 0.4 else 0)
    if vix is not None:
        score += 1 if vix <= 16 else (-1 if vix >= 25 else 0)
    if score > 0:
        return "risk_on"
    if score < 0:
        return "risk_off"
    return "neutral"
