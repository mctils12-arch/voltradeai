"""
AlphaDesk — data models.

Plain dataclasses describing what a data provider returns and what the engine
produces. Kept free of any analysis logic so providers and the engine can
evolve independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# --------------------------------------------------------------------------- #
# Inputs (what a DataProvider supplies)
# --------------------------------------------------------------------------- #
@dataclass
class Quote:
    ticker: str
    price: float
    prev_close: Optional[float] = None
    day_volume: Optional[int] = None
    avg_volume_30d: Optional[int] = None
    market_cap: Optional[float] = None


@dataclass
class Fundamentals:
    """Trailing-twelve-month + balance-sheet snapshot."""
    ticker: str
    revenue_ttm: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None        # 0.18 = +18%
    gross_margin: Optional[float] = None              # 0.42 = 42%
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    fcf_ttm: Optional[float] = None
    fcf_margin: Optional[float] = None
    roe: Optional[float] = None
    roic: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    current_ratio: Optional[float] = None
    shares_diluted: Optional[float] = None
    eps_ttm: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    # Valuation inputs
    pe: Optional[float] = None
    forward_pe: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_sales: Optional[float] = None
    pe_5y_median: Optional[float] = None              # own history, for mean-reversion
    sector_pe_median: Optional[float] = None


@dataclass
class Ownership:
    """Supply/demand structure."""
    ticker: str
    float_shares: Optional[float] = None
    shares_outstanding: Optional[float] = None
    short_pct_float: Optional[float] = None           # 0.12 = 12%
    short_days_to_cover: Optional[float] = None
    insider_net_6m: Optional[float] = None            # net shares bought(+)/sold(-)
    institutional_pct: Optional[float] = None
    institutional_net_qoq: Optional[float] = None     # change in inst. holders QoQ
    borrow_fee_pct: Optional[float] = None            # cost-to-borrow, squeeze proxy


@dataclass
class Filing:
    ticker: str
    form: str                                         # "10-K", "10-Q", "8-K", "4"
    filed_at: str                                     # ISO date
    title: str = ""
    url: str = ""
    text: str = ""                                    # extracted body (may be large)


@dataclass
class MarketContext:
    """Top-down backdrop shared across tickers."""
    spx_above_200dma: Optional[bool] = None
    breadth_pct_above_50dma: Optional[float] = None   # 0.55 = 55% of stocks
    vix: Optional[float] = None
    yield_10y: Optional[float] = None
    yield_curve_2s10s: Optional[float] = None         # 10y - 2y, negative = inverted
    credit_spread_hy: Optional[float] = None          # high-yield OAS, bps
    trend_label: str = "neutral"                       # risk_on / neutral / risk_off


@dataclass
class OptionsSnapshot:
    """Options-market inputs for the options pillar."""
    ticker: str
    iv_30d: Optional[float] = None            # 30d implied vol, annualized (0.35 = 35%)
    hv_30d: Optional[float] = None            # 30d realized vol, annualized
    iv_rank: Optional[float] = None           # 0-100: where IV sits in its ~1y range
    put_call_skew: Optional[float] = None     # 25-delta put IV - call IV, vol points (+ = downside fear)
    expected_move_30d: Optional[float] = None  # ~1-sigma move over 30d, fractional


@dataclass
class PriceBar:
    date: str
    close: float


@dataclass
class NewsItem:
    """One recent news / press-release headline for a ticker."""
    headline: str
    summary: str = ""
    url: str = ""
    source: str = ""
    date: str = ""        # ISO date


@dataclass
class Catalyst:
    """A detected hard catalyst (currently: pending cash acquisition)."""
    kind: str                         # "pending_acquisition" | "none"
    headline: str = ""
    url: str = ""
    deal_price: Optional[float] = None
    cash: bool = True
    status: str = "none"              # announced / shareholder_approved / regulatory_pending / closed / none


@dataclass
class CatalystView:
    """Catalyst + the merger-arb framing the engine derives from it, plus the
    recent news the read is based on. The actionable 'special situation' overlay."""
    catalyst: Catalyst
    current_price: Optional[float] = None
    upside_pct: Optional[float] = None        # to deal price if it closes
    downside_pct: Optional[float] = None      # to estimated break price
    downside_price: Optional[float] = None
    implied_close_prob: Optional[float] = None  # market-implied, 0..1
    assessment: str = ""
    news: List[NewsItem] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Outputs (what the engine produces)
# --------------------------------------------------------------------------- #
@dataclass
class Factor:
    """One auditable contribution to a sub-score."""
    name: str
    score: float            # 0-100 on this factor
    weight: float           # relative weight within its pillar
    detail: str


@dataclass
class Pillar:
    """A scored dimension (fundamentals, valuation, supply/demand, ...)."""
    name: str
    score: Optional[float]  # 0-100, None if no data
    factors: List[Factor] = field(default_factory=list)
    note: str = ""


@dataclass
class FilingRead:
    """Output of the filing reasoner (LLM in prod, heuristic offline)."""
    summary: str
    sentiment: float                  # -1 .. +1
    red_flags: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)
    source: str = "heuristic"


@dataclass
class OptionsStrategy:
    """A defined-risk options expression of the equity view (education only)."""
    structure: str                    # e.g. "Bull put credit spread"
    direction: str                    # bullish / bearish / neutral
    rationale: str
    legs: List[str] = field(default_factory=list)
    max_risk_note: str = ""


@dataclass
class OptionsView:
    """The options pillar's full output: the market snapshot + the suggested
    defined-risk structure derived from the equity verdict."""
    snapshot: OptionsSnapshot
    strategy: OptionsStrategy


@dataclass
class HorizonView:
    """Pre- and post-tax expected outcome for one holding horizon."""
    label: str                        # "swing (<1y)" / "core (>1y)"
    horizon_days: int
    expected_return: float            # pre-tax, fractional (0.12 = +12%)
    tax_rate: float                   # effective rate applied to the gain
    after_tax_return: float
    note: str = ""


@dataclass
class ResearchReport:
    ticker: str
    as_of: str
    price: Optional[float]
    verdict: str                      # Strong Buy / Buy / Hold / Sell / Strong Sell
    conviction: float                 # 0-100
    composite_score: float            # 0-100
    pillars: List[Pillar]
    supply_demand_read: str
    filing_read: Optional[FilingRead]
    market_note: str
    horizons: List[HorizonView]
    best_horizon: str
    thesis: str
    risks: List[str]
    data_completeness: float          # 0-1
    provider: str
    options: Optional[OptionsView] = None
    catalyst: Optional[CatalystView] = None
    disclaimer: str = (
        "Research and education only. Not investment advice. Verdicts are model "
        "output, not a recommendation, and do not account for your full situation."
    )

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
