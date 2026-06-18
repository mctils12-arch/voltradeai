"""AlphaDesk — human-readable report formatting."""
from __future__ import annotations

from .models import ResearchReport


def render_text(r: ResearchReport) -> str:
    L = []
    L.append("=" * 64)
    L.append(f"  {r.ticker}   {r.verdict.upper()}   ({r.composite_score:.0f}/100, "
             f"conviction {r.conviction:.0f})")
    L.append(f"  price ${r.price:,.2f}   as of {r.as_of}   provider: {r.provider}")
    L.append("=" * 64)
    L.append("")
    if r.catalyst and r.catalyst.catalyst.kind == "pending_acquisition":
        c = r.catalyst
        L.append("!! SPECIAL SITUATION — PENDING ACQUISITION")
        L.append("  " + c.assessment)
        if c.implied_close_prob is not None:
            L.append(f"  deal ${c.catalyst.deal_price:.2f}  upside {c.upside_pct:+.0%}  "
                     f"downside {c.downside_pct:+.0%}  market-implied close odds ~{c.implied_close_prob:.0%}")
        if c.catalyst.url:
            L.append(f"  source: {c.catalyst.url}")
        L.append("")
    L.append("THESIS")
    L.append("  " + r.thesis)
    if r.catalyst and r.catalyst.news:
        L.append("")
        L.append("RECENT NEWS")
        for n in r.catalyst.news[:5]:
            d = f"[{n.date}] " if n.date else ""
            L.append(f"  - {d}{n.headline}")
    L.append("")
    L.append("PILLARS")
    for p in r.pillars:
        s = f"{p.score:.0f}/100" if p.score is not None else "  n/a"
        L.append(f"  {p.name:<18} {s:>8}   {p.note}")
        for fct in p.factors:
            L.append(f"       - {fct.name:<28} {fct.score:>5.0f}   {fct.detail}")
    L.append("")
    L.append("SUPPLY / DEMAND")
    L.append("  " + r.supply_demand_read)
    if r.filing_read:
        L.append("")
        L.append(f"FILINGS  ({r.filing_read.source})")
        L.append("  " + r.filing_read.summary)
        if r.filing_read.catalysts:
            L.append("  catalysts: " + ", ".join(r.filing_read.catalysts))
        if r.filing_read.red_flags:
            L.append("  red flags: " + ", ".join(r.filing_read.red_flags))
    if r.options:
        snap, strat = r.options.snapshot, r.options.strategy
        L.append("")
        L.append("OPTIONS")
        bits = []
        if snap.iv_rank is not None:
            bits.append(f"IV rank {snap.iv_rank:.0f}")
        if snap.expected_move_30d is not None:
            bits.append(f"~{snap.expected_move_30d:.0%} 30d expected move")
        if snap.hv_30d is not None:
            bits.append(f"HV {snap.hv_30d:.0%}")
        if bits:
            L.append("  " + "   ".join(bits))
        L.append(f"  {strat.structure} ({strat.direction})")
        L.append("  " + strat.rationale)
        for leg in strat.legs:
            L.append(f"       - {leg}")
        if strat.max_risk_note:
            L.append("  " + strat.max_risk_note)
    L.append("")
    L.append("HORIZON / AFTER-TAX ALPHA")
    for h in r.horizons:
        star = "  <-- best after-tax" if h.label == r.best_horizon else ""
        L.append(f"  {h.label:<12} pre-tax {h.expected_return:+.1%}  "
                 f"tax {h.tax_rate:.0%}  ->  after-tax {h.after_tax_return:+.1%}{star}")
        L.append(f"               ({h.note})")
    L.append("")
    L.append("RISKS")
    for risk in r.risks:
        L.append(f"  - {risk}")
    L.append("")
    L.append(f"data completeness: {r.data_completeness:.0%}")
    L.append(r.disclaimer)
    return "\n".join(L)
