"""
AlphaDesk — catalyst detection + merger-arbitrage framing.

Pure functions (no network). `detect_deal` scans recent news for a pending cash
acquisition and extracts the deal price + status; `build_view` turns that into
the merger-arb framing the engine surfaces: spread to the deal, an estimated
downside on a break, the market-implied close probability, and a plain-language
"should I buy" assessment.

This is what lets the engine see special situations (e.g. a $35 cash buyout)
that fundamentals alone miss. Education only — it states the odds the *market*
implies and the payoff scenarios; it does not predict the outcome.
"""
from __future__ import annotations

import re
from typing import List, Optional

from .models import NewsItem, Catalyst, CatalystView

# Phrases that signal the *subject company* is in play as a target (an arb
# floor / special situation) — definitive deals, tender offers, and earlier-stage
# takeover interest/bids alike. Combined with an extractable per-share price, this
# keeps false positives low while catching real-world headline phrasing.
_ACQUIRED_TERMS = (
    "to be acquired", "be acquired by", "acquired by", "agrees to be acquired",
    "agreed to acquire", "to acquire all", "acquisition of", "acquisition interest",
    "acquisition talks", "acquisition offer", "merger agreement", "definitive agreement",
    "cash merger", "all-cash", "per share in cash", "buyout", "takeover", "take private",
    "take-private", "going private", "go private", "tender offer", "rival bid",
    "rival offer", "rival $", "bid for", "bid of", "bid",
)

# Talk-y language that means an early/uncertain stage rather than a signed deal.
_TALKS_TERMS = ("talks", "interest", "rumor", "rumour", "exploring", "considering",
                "approached", "weighing", "in play", "rival bid", "rival offer", "rival $")

# Default assumed drop from the current price if the deal breaks, used to back
# out a market-implied close probability. Deliberately conservative and stated
# in the output so the user can judge it.
_DEFAULT_BREAK_DROP = 0.30

# Per-share deal prices in real-world phrasing: "$35.00 per share", "$35 a share",
# "$35/share", "$37.50 bid", "$35 offer", "bid of $37.50", "per share ... $35.00".
# A negative lookahead after the number rejects deal *sizes* like "$4.5B" /
# "$4.2 billion" / "$500M" so we never read total consideration as a share price.
_NUM = r"(\d{1,4}(?:\.\d{1,2})?)(?![\d.])(?![bBmMkK])(?!\s*(?:billion|million|bn|mn))"
_PRICE_PATTERNS = (
    re.compile(r"\$\s?" + _NUM + r"\s*(?:per|/|a)\s*share", re.I),
    re.compile(r"\$\s?" + _NUM + r"\s*(?:bid|offer|cash)\b", re.I),
    re.compile(r"(?:bid|offer|acquire[a-z]*|deal|takeover)\s+(?:of|at|for)?\s*\$\s?" + _NUM, re.I),
    re.compile(r"per\s*share[^$]{0,30}\$\s?" + _NUM, re.I),
)


def _extract_price(text: str) -> Optional[float]:
    for pat in _PRICE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _status(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("completed", "completion of the", "consummat", "deal closed", "transaction closed")):
        return "closed"
    if any(k in t for k in ("shareholders approve", "stockholders approve", "approve the", "approved the",
                            "voted in favor", "back the takeover", "backs", "approve $", "shareholder approval")):
        return "shareholder_approved"
    if any(k in t for k in ("regulatory", "antitrust", "awaiting approval", "pending approval",
                            "closing conditions", "regulatory approval", "subject to approval")):
        return "regulatory_pending"
    if any(k in t for k in ("definitive agreement", "merger agreement", "to be acquired",
                            "agreed to acquire", "tender offer")):
        return "announced"
    if any(k in t for k in _TALKS_TERMS):
        return "in_talks"
    return "announced"


def detect_deal(news: List[NewsItem]) -> Catalyst:
    """Scan recent news for a pending cash acquisition with a per-share price.
    Requires both an acquisition phrase and an extractable price to flag, which
    keeps false positives low. Returns a `none` catalyst otherwise."""
    best: Optional[Catalyst] = None
    blob_status = []
    for n in news:
        text = f"{n.headline} {n.summary}"
        low = text.lower()
        if not any(term in low for term in _ACQUIRED_TERMS):
            continue
        price = _extract_price(text)
        blob_status.append(_status(text))
        if price is None:
            continue
        cand = Catalyst(
            kind="pending_acquisition",
            headline=n.headline,
            url=n.url,
            deal_price=price,
            cash=("cash" in low or "all-cash" in low),
            status=_status(text),
        )
        # Prefer the highest stated per-share price (the headline deal number).
        if best is None or (cand.deal_price or 0) > (best.deal_price or 0):
            best = cand
    if best is None:
        return Catalyst(kind="none")
    # Upgrade status if any related headline shows a later stage.
    order = {"in_talks": 0, "announced": 1, "regulatory_pending": 2,
             "shareholder_approved": 3, "closed": 4}
    if blob_status:
        best.status = max(blob_status + [best.status], key=lambda s: order.get(s, 0))
    return best


def implied_probability(current: float, deal: float, downside: float) -> Optional[float]:
    """Market-implied probability the deal closes, from current = p*deal +
    (1-p)*downside. None if the inputs are degenerate; clamped to [0,1]."""
    if deal is None or current is None or downside is None or deal <= downside:
        return None
    p = (current - downside) / (deal - downside)
    return round(max(0.0, min(1.0, p)), 3)


def build_view(catalyst: Catalyst, current_price: Optional[float],
               news: Optional[List[NewsItem]] = None,
               break_drop: float = _DEFAULT_BREAK_DROP) -> CatalystView:
    """Turn a detected deal + current price into the merger-arb overlay."""
    news = news or []
    if catalyst.kind != "pending_acquisition" or not catalyst.deal_price or not current_price:
        return CatalystView(catalyst=catalyst, current_price=current_price, news=news[:5])

    deal = catalyst.deal_price
    downside = round(current_price * (1 - break_drop), 2)
    upside = round(deal / current_price - 1, 4)
    down = round(downside / current_price - 1, 4)
    p = implied_probability(current_price, deal, downside)

    status_note = {
        "closed": "The deal has reportedly closed — limited remaining upside.",
        "shareholder_approved": "Shareholders have approved; regulatory/closing approval is the main remaining risk.",
        "regulatory_pending": "Awaiting regulatory/closing approval — the key remaining risk.",
        "announced": "A definitive agreement is reported — deal-completion risk applies.",
        "in_talks": "Early stage — takeover interest/bid reported but not a signed deal, so treat the price as indicative and the odds as lower.",
        "none": "",
    }.get(catalyst.status, "")

    cash = "cash " if catalyst.cash else ""
    parts = [
        f"Special situation: pending {cash}acquisition at ${deal:.2f}/share "
        f"({upside:+.0%} vs current ${current_price:.2f}; ~{down:+.0%} to an estimated "
        f"${downside:.2f} break price)."
    ]
    if status_note:
        parts.append(status_note)
    if p is not None:
        parts.append(
            f"At today's price the market implies ~{p:.0%} odds the deal closes — "
            f"a buy if you believe approval is more likely than that, a pass if less. "
            f"This merger-arb setup, not the fundamentals, should drive the decision."
        )
    return CatalystView(
        catalyst=catalyst,
        current_price=current_price,
        upside_pct=upside,
        downside_pct=down,
        downside_price=downside,
        implied_close_prob=p,
        assessment=" ".join(parts),
        news=news[:5],
    )
