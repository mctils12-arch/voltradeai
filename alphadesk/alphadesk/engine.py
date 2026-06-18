"""
AlphaDesk — the engine.

Pulls everything from a DataProvider, scores each pillar, reads filings, blends
into a composite + verdict, then runs the tax-aware horizon comparison to find
where the after-tax alpha actually is. The composite weighting is explicit and
overridable; an optional ML `model` can replace or augment the rules score
without changing anything else.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol

from . import analysis, options as options_mod, catalyst as catalyst_mod
from .models import (
    ResearchReport, Pillar, HorizonView, FilingRead, OptionsView,
)
from .providers import DataProvider, SampleProvider
from .tax import TaxProfile, after_tax_return


# Default pillar weights for the composite. Sum need not be 1; normalized.
# Options is a deliberately minor input — it shades the verdict via the implied-
# risk environment but doesn't drive it; the actionable options call is the
# defined-risk strategy attached to the report, not this score.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "Fundamentals": 0.30,
    "Valuation": 0.22,
    "Supply / Demand": 0.18,
    "Market Context": 0.12,
    "Filings": 0.18,
    "Options": 0.06,
}

VERDICT_BANDS = [
    (78, "Strong Buy"),
    (62, "Buy"),
    (45, "Hold"),
    (30, "Sell"),
    (0, "Strong Sell"),
]


class ModelScorer(Protocol):
    """Optional ML hook: maps pillar scores + features to a 0-100 score."""
    def score(self, features: Dict[str, float]) -> float: ...


def _verdict(score: float) -> str:
    for thresh, label in VERDICT_BANDS:
        if score >= thresh:
            return label
    return "Strong Sell"


def _filing_pillar_score(read: Optional[FilingRead]) -> Optional[float]:
    if read is None:
        return None
    # sentiment in [-1,1] -> [0,100]
    return round((read.sentiment + 1) / 2 * 100, 1)


def analyze(
    ticker: str,
    provider: Optional[DataProvider] = None,
    tax: Optional[TaxProfile] = None,
    weights: Optional[Dict[str, float]] = None,
    model: Optional[ModelScorer] = None,
    filing_llm=None,
) -> ResearchReport:
    provider = provider or SampleProvider()
    tax = tax or TaxProfile()
    weights = weights or DEFAULT_WEIGHTS

    q = provider.quote(ticker)
    f = provider.fundamentals(ticker)
    o = provider.ownership(ticker)
    m = provider.market_context()
    filings = provider.filings(ticker, limit=5)
    opt_snap = provider.options(ticker)
    news = provider.news(ticker)

    pillars: List[Pillar] = [
        analysis.score_fundamentals(f),
        analysis.score_valuation(f),
        analysis.score_supply_demand(o, q),
        analysis.score_market_context(m),
        analysis.score_options(opt_snap),
    ]
    filing_read = analysis.FilingReasoner(llm=filing_llm).read(filings)
    pillars.append(Pillar("Filings", _filing_pillar_score(filing_read),
                          [], filing_read.summary))

    # Composite (explainable, weighted average of available pillars).
    num = den = 0.0
    feature_vec: Dict[str, float] = {}
    for p in pillars:
        if p.score is None:
            continue
        w = weights.get(p.name, 0.1)
        num += p.score * w
        den += w
        feature_vec[p.name] = p.score
    rules_score = round(num / den, 1) if den else 50.0

    # Optional ML overlay: average the model and the rules score so the model
    # informs but never silently dominates an unexplainable verdict.
    if model is not None:
        try:
            ml = float(model.score(feature_vec))
            composite = round(0.5 * rules_score + 0.5 * _clamp01_to_100(ml), 1)
        except Exception:
            composite = rules_score
    else:
        composite = rules_score

    completeness = round(sum(1 for p in pillars if p.score is not None) / len(pillars), 2)
    # Conviction scales with data completeness and how far from neutral we are.
    conviction = round(min(100, abs(composite - 50) * 1.6 * (0.5 + 0.5 * completeness)), 1)

    horizons = _horizons(composite, pillars, tax)
    best = max(horizons, key=lambda h: h.after_tax_return)

    thesis, risks = _narrative(ticker, q.price, composite, pillars, filing_read, m)

    verdict = _verdict(composite)
    options_view = OptionsView(
        snapshot=opt_snap,
        strategy=options_mod.build_strategy(verdict, q.price, opt_snap),
    )

    # Catalyst overlay: a detected pending acquisition is a special situation that
    # supersedes the fundamentals read — lead the thesis with it so the verdict
    # pill never tells a misleading story on its own. Detection scans both recent
    # news and the (now real) SEC filing text — a merger 6-K reliably states the
    # deal terms even when the news window has moved on — while only the news is
    # shown to the user.
    from .models import NewsItem as _NewsItem
    _filing_sources = [
        _NewsItem(headline=f.title or f.form, summary=(f.text or "")[:6000],
                  url=f.url, source="SEC EDGAR", date=f.filed_at)
        for f in filings
    ]
    catalyst_view = catalyst_mod.build_view(
        catalyst_mod.detect_deal(news + _filing_sources), q.price, news)
    if catalyst_view.catalyst.kind == "pending_acquisition" and catalyst_view.assessment:
        thesis = catalyst_view.assessment + " (Fundamentals view below.) " + thesis
        risks.insert(0, "Outcome hinges on deal completion (regulatory/closing risk), "
                        "not fundamentals — size for the break scenario.")

    return ResearchReport(
        ticker=ticker.upper(),
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        price=q.price,
        verdict=verdict,
        conviction=conviction,
        composite_score=composite,
        pillars=pillars,
        supply_demand_read=next(p.note for p in pillars if p.name == "Supply / Demand"),
        filing_read=filing_read,
        market_note=next(p.note for p in pillars if p.name == "Market Context"),
        horizons=horizons,
        best_horizon=best.label,
        thesis=thesis,
        risks=risks,
        data_completeness=completeness,
        provider=provider.name,
        options=options_view,
        catalyst=catalyst_view,
    )


def _clamp01_to_100(x: float) -> float:
    return max(0.0, min(100.0, x if x > 1 else x * 100))


def _horizons(composite: float, pillars: List[Pillar], tax: TaxProfile) -> List[HorizonView]:
    """
    Translate the composite into a rough expected pre-tax return for two
    horizons, then tax each appropriately. The mapping is deliberately simple
    and transparent: edge above/below neutral (50) drives expected return,
    scaled differently for a swing vs a core hold.
    """
    edge = (composite - 50) / 50.0   # -1 .. +1
    val = next((p.score for p in pillars if p.name == "Valuation"), 50) or 50
    sd = next((p.score for p in pillars if p.name == "Supply / Demand"), 50) or 50

    # Swing leans on supply/demand + momentum; core leans on value + quality.
    swing_ret = round(edge * 0.18 + (sd - 50) / 50 * 0.06, 4)        # ~ +/-24%
    core_ret = round(edge * 0.40 + (val - 50) / 50 * 0.10, 4)        # ~ +/-50% over the hold

    out = []
    for label, days, ret, note in [
        ("swing (<1y)", 120, swing_ret, "taxed as ordinary income (short-term)"),
        ("core (>1y)", 540, core_ret, "qualifies for long-term cap-gains rate"),
    ]:
        rate, at = after_tax_return(ret, days, tax)
        out.append(HorizonView(
            label=label, horizon_days=days, expected_return=ret,
            tax_rate=rate, after_tax_return=at, note=note,
        ))
    return out


def _narrative(ticker, price, composite, pillars, fr: FilingRead, m):
    pscore = {p.name: p.score for p in pillars}
    bull = pscore.get("Fundamentals", 50) or 50
    val = pscore.get("Valuation", 50) or 50
    parts = []
    if bull >= 60:
        parts.append("quality fundamentals")
    if val >= 60:
        parts.append("an undemanding valuation")
    if (pscore.get("Supply / Demand") or 0) >= 60:
        parts.append("supportive supply/demand")
    if fr and fr.sentiment > 0.2:
        parts.append("constructive filing tone")
    driver = ", ".join(parts) if parts else "a mixed signal set"
    thesis = (f"{ticker.upper()} scores {composite:.0f}/100 overall, driven by {driver}. "
              f"The market backdrop is {m.trend_label.replace('_', '-')}.")

    risks: List[str] = list(fr.red_flags) if fr else []
    if val < 40:
        risks.append("Valuation is stretched — multiple compression risk if growth slows.")
    if (pscore.get("Fundamentals") or 100) < 40:
        risks.append("Weak fundamentals — profitability/leverage need watching.")
    if m.trend_label == "risk_off":
        risks.append("Risk-off market backdrop is a headwind regardless of the name.")
    if not risks:
        risks.append("No standout red flags, but model verdicts are not certainty.")
    return thesis, risks
