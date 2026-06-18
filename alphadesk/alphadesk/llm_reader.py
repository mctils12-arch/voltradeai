"""
AlphaDesk — LLM filing reader (Claude).

A drop-in replacement for the heuristic keyword scan in `FilingReasoner`: give
it the real SEC filing text that `edgar.py` now fetches and Claude reads it the
way an analyst would — producing a plain-language summary, an overall sentiment,
concrete red flags, and concrete catalysts.

Wiring: `ClaudeFilingReader` exposes `.read(filings) -> FilingRead`, the exact
interface `FilingReasoner(llm=...)` expects. `make_filing_reader()` returns one
when an Anthropic API key is configured and the SDK is importable, else `None`
(so the engine silently keeps using the heuristic). `FilingReasoner` already
wraps the call in try/except, so any runtime error also degrades gracefully.

Uses the official Anthropic Python SDK with structured outputs so the response
is always a valid object matching `FilingRead`. Research/education only — the
prompt instructs the model to ground every statement in the filing text and not
to give investment advice.
"""
from __future__ import annotations

import json
from typing import List, Optional

from .models import Filing, FilingRead

# Per-filing excerpt cap fed to the model. edgar.py already caps each document at
# ~60KB; trim further here so a 5-filing prompt stays a few tens of thousands of
# tokens — enough signal for the read, bounded latency inside the web request.
_PER_FILING_CHARS = 18_000

_SYSTEM = (
    "You are an equity research analyst reading excerpts from a company's recent "
    "SEC filings (10-K, 10-Q, 8-K). Read them the way a careful analyst would and "
    "return a structured assessment. Ground every statement in the provided text — "
    "do not speculate or use outside knowledge, and if the excerpts are thin, say so "
    "and keep sentiment near zero. This is research/education only, not investment "
    "advice.\n\n"
    "Fields:\n"
    "- summary: 2-4 sentences on what the filings say about the business right now "
    "(performance, guidance, notable events).\n"
    "- sentiment: a number from -1.0 (clearly negative) to +1.0 (clearly positive) "
    "reflecting the net tone of the filings for an investor.\n"
    "- red_flags: concrete cautionary items actually stated in the text (e.g. "
    "'going concern language', 'SEC investigation disclosed', 'guidance cut'). Empty "
    "if none.\n"
    "- catalysts: concrete positive developments actually stated (e.g. 'raised FY "
    "guidance', 'new $10B buyback', 'record quarterly revenue'). Empty if none."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {"type": "number"},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "catalysts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "sentiment", "red_flags", "catalysts"],
    "additionalProperties": False,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class ClaudeFilingReader:
    """Reads filings with Claude. `.read(filings)` returns a `FilingRead`."""

    def __init__(self, model: str = "claude-opus-4-8", client=None, api_key: Optional[str] = None):
        self.model = model
        if client is not None:
            self.client = client
        else:
            import anthropic  # lazy: core stays dependency-free
            self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def _prompt(self, filings: List[Filing]) -> str:
        parts = []
        for f in filings:
            body = (f.text or "").strip()
            if not body:
                continue
            parts.append(
                f"--- {f.form} filed {f.filed_at} ({f.title}) ---\n{body[:_PER_FILING_CHARS]}"
            )
        if not parts:
            # Metadata-only (no document text extracted) — let the model say so.
            listing = "\n".join(f"- {f.form} filed {f.filed_at}: {f.title}" for f in filings)
            return f"No filing body text was available. Recent filings on record:\n{listing}"
        return "\n\n".join(parts)

    def read(self, filings: List[Filing]) -> FilingRead:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _SCHEMA}},
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Read these filing excerpts and assess them:\n\n{self._prompt(filings)}",
            }],
        )
        # output_config.format guarantees a text block of valid JSON; skip any
        # thinking blocks that precede it.
        text = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(text)
        return FilingRead(
            summary=str(data.get("summary", "")).strip(),
            sentiment=round(_clamp(float(data.get("sentiment", 0.0)), -1.0, 1.0), 2),
            red_flags=[str(x) for x in (data.get("red_flags") or [])],
            catalysts=[str(x) for x in (data.get("catalysts") or [])],
            source=f"llm:{self.model}",
        )


def make_filing_reader(keys=None) -> Optional[ClaudeFilingReader]:
    """Return a ClaudeFilingReader when an Anthropic key + SDK are available,
    else None (engine falls back to the heuristic). Never raises."""
    try:
        from .config import load_keys
        k = keys or load_keys()
        if not k.have_anthropic:
            return None
        return ClaudeFilingReader(model=k.anthropic_model, api_key=k.anthropic_key)
    except Exception:
        return None
