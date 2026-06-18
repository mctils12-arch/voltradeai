"""
AlphaDesk — analysis pillars.

Each function turns raw provider data into a 0-100 `Pillar` with auditable
`Factor`s. Scores are intentionally transparent (no hidden weights): you can
read why a name scored what it did. The ML hook lives in engine.py; this file
is the explainable, rules-based quant layer that also serves as the model's
feature set.
"""
from __future__ import annotations

from typing import List, Optional

from .models import (
    Fundamentals, Ownership, MarketContext, Filing, Quote,
    Factor, Pillar, FilingRead,
)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _band(value: Optional[float], lo: float, hi: float) -> Optional[float]:
    """Linear map value in [lo,hi] -> [0,100]; None passes through."""
    if value is None:
        return None
    if hi == lo:
        return 50.0
    return _clamp((value - lo) / (hi - lo) * 100.0)


def _avg(scores: List[Optional[float]]) -> Optional[float]:
    vals = [s for s in scores if s is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


# --------------------------------------------------------------------------- #
def score_fundamentals(f: Fundamentals) -> Pillar:
    factors: List[Factor] = []

    def add(name, raw, lo, hi, w, fmt="{:.1%}"):
        s = _band(raw, lo, hi)
        if s is not None:
            factors.append(Factor(name, round(s, 1), w,
                                   f"{fmt.format(raw)} (mapped {s:.0f}/100)"))

    add("Revenue growth", f.revenue_growth_yoy, -0.05, 0.30, 1.2)
    add("Gross margin", f.gross_margin, 0.20, 0.70, 0.8)
    add("Operating margin", f.operating_margin, 0.0, 0.35, 1.0)
    add("FCF margin", f.fcf_margin, 0.0, 0.30, 1.2)
    add("Return on invested capital", f.roic, 0.05, 0.30, 1.3)
    add("EPS growth", f.eps_growth_yoy, -0.10, 0.30, 1.0)
    # Balance sheet: lower leverage is better -> invert.
    if f.net_debt_to_ebitda is not None:
        s = _band(-f.net_debt_to_ebitda, -4.0, 1.0)
        factors.append(Factor("Leverage (net debt/EBITDA)", round(s, 1), 1.1,
                              f"{f.net_debt_to_ebitda:.1f}x (lower is better)"))

    score = _weighted(factors)
    note = ("Quality + growth profile." if score and score >= 60
            else "Watch profitability/leverage." if score else "Insufficient data.")
    return Pillar("Fundamentals", score, factors, note)


def score_valuation(f: Fundamentals) -> Pillar:
    """Cheaper vs own history and sector -> higher score (room to re-rate)."""
    factors: List[Factor] = []
    if f.pe and f.pe_5y_median:
        rel = f.pe / f.pe_5y_median
        s = _band(-rel, -1.6, -0.6)  # rel<1 (below own median) scores high
        factors.append(Factor("P/E vs 5y median", round(s, 1), 1.3,
                              f"{f.pe:.0f} vs {f.pe_5y_median:.0f} ({rel:.2f}x)"))
    if f.pe and f.sector_pe_median:
        rel = f.pe / f.sector_pe_median
        s = _band(-rel, -1.6, -0.6)
        factors.append(Factor("P/E vs sector", round(s, 1), 1.0,
                              f"{f.pe:.0f} vs sector {f.sector_pe_median:.0f}"))
    if f.ev_ebitda is not None:
        factors.append(Factor("EV/EBITDA", round(_band(-f.ev_ebitda, -25, -6), 1), 0.9,
                              f"{f.ev_ebitda:.1f}x (lower cheaper)"))
    if f.forward_pe and f.pe and f.forward_pe < f.pe:
        factors.append(Factor("Forward earnings expansion", 65.0, 0.6,
                              f"fwd P/E {f.forward_pe:.0f} < trailing {f.pe:.0f}"))
    score = _weighted(factors)
    note = ("Looks cheap vs its own history." if score and score >= 60
            else "Richly valued — needs growth to justify." if score else "No valuation data.")
    return Pillar("Valuation", score, factors, note)


def score_supply_demand(o: Ownership, q: Quote) -> Pillar:
    """
    Reads the share-supply structure: squeeze potential (short interest, days to
    cover, borrow fee), accumulation (insider + institutional flow), and demand
    (volume vs average).
    """
    factors: List[Factor] = []
    if o.short_pct_float is not None:
        # Elevated short interest = fuel; very high can also mean broken thesis.
        s = _band(o.short_pct_float, 0.02, 0.25)
        factors.append(Factor("Short interest (% float)", round(s, 1), 1.0,
                              f"{o.short_pct_float:.1%} of float short"))
    if o.short_days_to_cover is not None:
        factors.append(Factor("Days to cover", round(_band(o.short_days_to_cover, 0.5, 7), 1), 0.8,
                              f"{o.short_days_to_cover:.1f} days of volume to cover"))
    if o.borrow_fee_pct is not None:
        factors.append(Factor("Borrow fee", round(_band(o.borrow_fee_pct, 0.01, 0.30), 1), 0.7,
                              f"{o.borrow_fee_pct:.1%} cost to borrow (squeeze proxy)"))
    if o.insider_net_6m is not None:
        s = _band(o.insider_net_6m, -1e6, 2e6)
        factors.append(Factor("Insider net buying (6m)", round(s, 1), 1.2,
                              f"{o.insider_net_6m:,.0f} net shares"))
    if o.institutional_net_qoq is not None:
        s = _band(o.institutional_net_qoq, -0.05, 0.08)
        factors.append(Factor("Institutional flow (QoQ)", round(s, 1), 1.1,
                              f"{o.institutional_net_qoq:+.1%} change in holders"))
    if q.avg_volume_30d and q.day_volume:
        ratio = q.day_volume / q.avg_volume_30d
        factors.append(Factor("Volume vs 30d avg", round(_band(ratio, 0.5, 2.5), 1), 0.6,
                              f"{ratio:.2f}x average volume"))

    score = _weighted(factors)
    return Pillar("Supply / Demand", score, factors, _squeeze_note(o))


def _squeeze_note(o: Ownership) -> str:
    if o.short_pct_float and o.short_pct_float > 0.20 and (o.short_days_to_cover or 0) > 4:
        return ("High short interest + slow cover + borrow cost = squeeze fuel if a "
                "catalyst hits. Asymmetric, but elevated shorts can also signal a "
                "broken story — pair with the fundamental read.")
    if o.insider_net_6m and o.insider_net_6m > 5e5:
        return "Insiders are net buyers — the people with the most information are accumulating."
    if o.insider_net_6m and o.insider_net_6m < -5e5:
        return "Insiders are net sellers over the last 6 months — a yellow flag for demand."
    return "Balanced supply/demand; no standout squeeze or accumulation signal."


def score_market_context(m: MarketContext) -> Pillar:
    factors: List[Factor] = []
    if m.spx_above_200dma is not None:
        factors.append(Factor("Index trend", 70.0 if m.spx_above_200dma else 30.0, 1.0,
                              "S&P above 200dma" if m.spx_above_200dma else "S&P below 200dma"))
    if m.breadth_pct_above_50dma is not None:
        factors.append(Factor("Breadth", round(_band(m.breadth_pct_above_50dma, 0.3, 0.7), 1), 1.0,
                              f"{m.breadth_pct_above_50dma:.0%} of stocks above 50dma"))
    if m.vix is not None:
        factors.append(Factor("Volatility (VIX)", round(_band(-m.vix, -30, -12), 1), 0.8,
                              f"VIX {m.vix:.0f} (lower calmer)"))
    if m.credit_spread_hy is not None:
        factors.append(Factor("Credit spreads", round(_band(-m.credit_spread_hy, -600, -300), 1), 0.9,
                              f"HY OAS {m.credit_spread_hy:.0f}bps (tighter = risk-on)"))
    score = _weighted(factors)
    return Pillar("Market Context", score, factors,
                  f"Backdrop is {m.trend_label.replace('_', '-')}.")


# --------------------------------------------------------------------------- #
class FilingReasoner:
    """
    Reads recent filings 'like a human'. In production this delegates to an LLM
    (Claude) for genuine reading of MD&A, risk factors, and 8-K events. Offline
    it uses a transparent keyword heuristic so the engine still runs.
    """

    RED = ["going concern", "investigation", "restatement", "subpoena", "default",
           "impairment", "decline", "weakness", "litigation", "dilution"]
    GREEN = ["raised guidance", "record", "buyback", "expanded", "strong demand",
             "beat", "accelerat", "margin expansion", "new contract"]

    def __init__(self, llm=None):
        self.llm = llm  # object with .read(filings) -> FilingRead, optional

    def read(self, filings: List[Filing]) -> FilingRead:
        if self.llm is not None:
            try:
                return self.llm.read(filings)
            except Exception:
                pass  # fall through to heuristic
        blob = " ".join(f.text.lower() for f in filings)
        reds = sorted({w for w in self.RED if w in blob})
        greens = sorted({w for w in self.GREEN if w in blob})
        sentiment = _clamp(50 + 8 * len(greens) - 10 * len(reds), 0, 100) / 50 - 1
        summary = (f"Heuristic scan of {len(filings)} filings: "
                   f"{len(greens)} positive vs {len(reds)} cautionary signals. "
                   "Wire an LLM reader for true narrative analysis.")
        return FilingRead(
            summary=summary, sentiment=round(sentiment, 2),
            red_flags=[r.title() for r in reds],
            catalysts=[g.title() for g in greens],
            source="heuristic",
        )


# --------------------------------------------------------------------------- #
def _weighted(factors: List[Factor]) -> Optional[float]:
    if not factors:
        return None
    tw = sum(f.weight for f in factors)
    if tw == 0:
        return None
    return round(sum(f.score * f.weight for f in factors) / tw, 1)
