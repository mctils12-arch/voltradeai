"""Lazy Prices PR 2c: year-over-year text similarity of archived 10-K/10-Q sections.

Pure functions over the sections JSON that sec_filings_text writes. No prices,
no trading imports, no network (SPINOUT-READY). Output rows are RAW MATERIAL for
ladder gate 2 (SIGNAL) — no predictive claim is made here.

Lookahead: a row is knowable only at the CURRENT filing's accepted_at (carried
through unchanged); the prior filing is by construction older.
"""
import math
import re
from collections import Counter
from typing import Optional

import sec_filings_text as sft

MIN_TOKENS = 50          # below this a cosine/Jaccard is noise, not a reading
_TOKEN_RE = re.compile(r"[a-z]{2,}")   # words only: numbers change every year and are not "language"
EXCLUDE_FLAGS = ("no_material_changes", "not_applicable", "incorporated_by_reference")


def tokenize(text: str) -> list:
    return _TOKEN_RE.findall((text or "").lower())


def cosine_tf(a: list, b: list) -> Optional[float]:
    """Cosine of raw term-frequency vectors (Cohen-Malloy-Nguyen 'Cosine')."""
    if not a or not b:
        return None
    ca, cb = Counter(a), Counter(b)
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else None


def jaccard_set(a: list, b: list) -> Optional[float]:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return None
    return len(sa & sb) / len(sa | sb)


def exclusion_reason(sec: Optional[dict]) -> Optional[str]:
    """Why a section cannot be scored (None = usable). Boilerplate flags are
    excluded rather than scored: 'no material changes' text is similar to
    itself by construction and says nothing about the business."""
    if not sec or not sec.get("found"):
        return "not_found"
    if sec.get("short"):
        return "short"
    for f in EXCLUDE_FLAGS:
        if (sec.get("flags") or {}).get(f):
            return f
    return None


def section_similarity(cur: Optional[dict], prior: Optional[dict]) -> dict:
    reason = exclusion_reason(cur) or exclusion_reason(prior)
    out = {"excluded": reason, "cosine": None, "jaccard": None, "tokens_cur": 0, "tokens_prior": 0}
    if reason:
        return out
    tc, tp = tokenize(cur["text"]), tokenize(prior["text"])
    out["tokens_cur"], out["tokens_prior"] = len(tc), len(tp)
    if min(len(tc), len(tp)) < MIN_TOKENS:
        out["excluded"] = "too_few_tokens"
        return out
    out["cosine"], out["jaccard"] = cosine_tf(tc, tp), jaccard_set(tc, tp)
    return out


def pair_similarity(cur_doc: dict, prior_doc: dict) -> dict:
    """One row per (current filing, prior-year filing): per-section scores plus
    the identifiers a downstream join needs."""
    secs = {}
    for key in (cur_doc.get("sections") or {}):
        secs[key] = section_similarity(cur_doc["sections"].get(key),
                                       (prior_doc.get("sections") or {}).get(key))
    return {"cik": cur_doc.get("cik"), "accession": cur_doc.get("accession"),
            "prior_accession": prior_doc.get("accession"), "form": cur_doc.get("form"),
            "period": cur_doc.get("period"), "accepted_at": cur_doc.get("accepted_at"),
            "accepted_at_source": cur_doc.get("accepted_at_source"), "sections": secs}


def build_rows(filings: list, load_sections) -> list:
    """filings: list_filings() dicts for ONE or many companies; load_sections(
    cik, accession) -> sections doc or None (archive read injected for tests).
    Unpaired or unreadable filings are skipped, never guessed."""
    pairs = sft.pair_prior_filings(filings)
    by_acc = {f["accession"]: f for f in filings}
    rows = []
    for acc, p in pairs.items():
        if not p["prior"]:
            continue
        cur = load_sections(by_acc[acc]["cik"], acc)
        prior = load_sections(by_acc[p["prior"]]["cik"], p["prior"])
        if cur and prior:
            row = pair_similarity(cur, prior)
            row["gap_days"] = p["gap_days"]
            rows.append(row)
    return rows
