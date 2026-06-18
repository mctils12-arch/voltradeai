"""
AlphaDesk — tax-aware after-tax return mechanics.

This models the *mechanics* of US capital-gains taxation so the engine can
compare a short-term play against a long-term one on an after-tax basis. It
does NOT hardcode current statutory rates as fact (those change and vary by
filer); instead it takes a `TaxProfile` with documented defaults you should
confirm against current IRS / state guidance for your situation.

Mechanics modeled:
  - Holding <= 365 days  -> short-term gain, taxed at your ordinary marginal rate.
  - Holding  > 365 days  -> long-term gain, taxed at the preferential LTCG rate.
  - Net investment income tax (NIIT) surtax may apply on top of either.
  - A flat state rate is added (states generally don't distinguish ST vs LT).

Not tax advice. For education only; consult a tax professional.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaxProfile:
    # All values are DEFAULTS to verify, not a statement of current law.
    ordinary_marginal_rate: float = 0.32   # your federal marginal bracket
    ltcg_rate: float = 0.15                 # 0 / 15 / 20 federal LTCG tier
    niit_rate: float = 0.038               # net investment income tax surtax
    niit_applies: bool = True
    state_rate: float = 0.05               # flat state income/cap-gains rate

    def rate_for(self, horizon_days: int) -> float:
        base = self.ordinary_marginal_rate if horizon_days <= 365 else self.ltcg_rate
        surtax = self.niit_rate if self.niit_applies else 0.0
        return round(base + surtax + self.state_rate, 4)


def after_tax_return(pretax_return: float, horizon_days: int, profile: TaxProfile) -> tuple[float, float]:
    """
    Returns (effective_tax_rate, after_tax_return). Losses pass through
    untaxed (no benefit modeled here — keeps the comparison conservative).
    """
    rate = profile.rate_for(horizon_days)
    if pretax_return <= 0:
        return rate, round(pretax_return, 4)
    return rate, round(pretax_return * (1 - rate), 4)
