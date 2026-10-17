"""Lazy Prices PR 2c: year-over-year text similarity of archived 10-K/10-Q sections.

Pure functions over the sections JSON that sec_filings_text writes. No prices,
no trading imports, no network (SPINOUT-READY). Output rows are RAW MATERIAL for
ladder gate 2 (SIGNAL) — no predictive claim is made here.

Lookahead: a row is knowable only at the CURRENT filing's accepted_at (carried
through unchanged); the prior filing is by construction older.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
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


def cosine_tfidf(a: list, b: list, idf: dict) -> Optional[float]:
    """Cosine of tf*idf vectors (PR 2e). idf comes from a corpus of the SAME
    section key, so words every filing uses ('the', 'company', 'risk') get low
    weight and the cosine is dominated by distinctive language. Unknown words
    get the maximum idf (rare by definition)."""
    if not a or not b or not idf:
        return None
    default = max(idf.values())
    ca, cb = Counter(a), Counter(b)
    w = lambda c: {k: v * idf.get(k, default) for k, v in c.items()}
    wa, wb = w(ca), w(cb)
    dot = sum(v * wb.get(k, 0.0) for k, v in wa.items())
    na = math.sqrt(sum(v * v for v in wa.values()))
    nb = math.sqrt(sum(v * v for v in wb.values()))
    return dot / (na * nb) if na and nb else None


def build_idf(token_lists) -> dict:
    """Smoothed idf = ln((N+1)/(df+1)) + 1 over an iterable of token lists."""
    df, n = Counter(), 0
    for toks in token_lists:
        n += 1
        df.update(set(toks))
    return {k: math.log((n + 1) / (v + 1)) + 1.0 for k, v in df.items()}


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


def section_similarity(cur: Optional[dict], prior: Optional[dict], idf: Optional[dict] = None) -> dict:
    reason = exclusion_reason(cur) or exclusion_reason(prior)
    out = {"excluded": reason, "cosine": None, "jaccard": None, "cosine_tfidf": None, "tokens_cur": 0, "tokens_prior": 0}
    if reason:
        return out
    tc, tp = tokenize(cur["text"]), tokenize(prior["text"])
    out["tokens_cur"], out["tokens_prior"] = len(tc), len(tp)
    if min(len(tc), len(tp)) < MIN_TOKENS:
        out["excluded"] = "too_few_tokens"
        return out
    out["cosine"], out["jaccard"] = cosine_tf(tc, tp), jaccard_set(tc, tp)
    if idf:
        out["cosine_tfidf"] = cosine_tfidf(tc, tp, idf)
    return out


def pair_similarity(cur_doc: dict, prior_doc: dict, idf_by_section: Optional[dict] = None) -> dict:
    """One row per (current filing, prior-year filing): per-section scores plus
    the identifiers a downstream join needs."""
    secs = {}
    for key in (cur_doc.get("sections") or {}):
        secs[key] = section_similarity(cur_doc["sections"].get(key),
                                       (prior_doc.get("sections") or {}).get(key),
                                       (idf_by_section or {}).get(key))
    return {"cik": cur_doc.get("cik"), "accession": cur_doc.get("accession"),
            "prior_accession": prior_doc.get("accession"), "form": cur_doc.get("form"),
            "period": cur_doc.get("period"), "accepted_at": cur_doc.get("accepted_at"),
            "accepted_at_source": cur_doc.get("accepted_at_source"), "sections": secs}


def build_rows(filings: list, load_sections, idf_by_section: Optional[dict] = None) -> list:
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
            row = pair_similarity(cur, prior, idf_by_section)
            row["gap_days"] = p["gap_days"]
            rows.append(row)
    return rows


# ── PR 2d: archive walk + distribution report (still no prices, no signal claim)

def archive_filings(archive: str) -> list:
    """Rebuild list_filings()-shaped dicts from the archived sections files, so
    pairing needs no network. Unreadable files are skipped, never guessed."""
    root = os.path.join(sft.stream_root(archive), "sections")
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*.json.gz"))):
        try:
            d = json.loads(sft.read_gz(path))
        except (OSError, ValueError, EOFError) as e:
            print(f"[sec_filings_similarity] skipping unreadable sections file {path}: {e}", file=sys.stderr)
            continue
        out.append({"cik": d["cik"], "accession": d["accession"], "form": d["form"],
                    "reportDate": d.get("period"), "filingDate": d.get("filing_date"),
                    "isAmendment": bool(d.get("is_amendment"))})
    return out


def _loader(archive: str):
    def load(cik, acc):
        try:
            return json.loads(sft.read_gz(sft.sections_path(archive, cik, acc)))
        except (OSError, ValueError, EOFError):
            return None
    return load


def idf_from_archive(archive: str, filings: Optional[list] = None) -> dict:
    """Per-section-key idf over every scoreable archived section. DESCRIPTIVE
    USE ONLY: the corpus includes filings after any given row's accepted_at, a
    mild lookahead on word rarity. Gate 2 must rebuild this point-in-time."""
    load = _loader(archive)
    toks = {}
    for f in (filings if filings is not None else archive_filings(archive)):
        d = load(f["cik"], f["accession"])
        for key, sec in ((d or {}).get("sections") or {}).items():
            if not exclusion_reason(sec):
                toks.setdefault(key, []).append(tokenize(sec["text"]))
    return {k: build_idf(v) for k, v in toks.items()}


def rows_from_archive(archive: str, with_idf: bool = True) -> list:
    filings = archive_filings(archive)
    idf = idf_from_archive(archive, filings) if with_idf else None
    return build_rows(filings, _loader(archive), idf)


# ── PR 2f: point-in-time idf (gate 2 must not see words' future rarity)

class PitIdf:
    """Live smoothed-idf view over the docs added so far for ONE section key.
    Same formula as build_idf (ln((N+1)/(df+1))+1) and same unknown-word
    default (max idf = the rarest seen word), but updated incrementally so a
    row can be scored against exactly the corpus knowable at its accepted_at."""

    def __init__(self):
        self.n, self.df, self._n_df1 = 0, Counter(), 0

    def add(self, token_set) -> None:
        self.n += 1
        for t in token_set:
            c = self.df[t]
            self.df[t] = c + 1
            if c == 0:
                self._n_df1 += 1
            elif c == 1:
                self._n_df1 -= 1

    def get(self, tok, default=None):
        v = self.df.get(tok)
        return default if v is None else math.log((self.n + 1) / (v + 1)) + 1.0

    def values(self):
        if self._n_df1 > 0:      # fast path: rarest seen word has df == 1
            return [math.log((self.n + 1) / 2) + 1.0]
        return [self.get(t) for t in self.df]

    def __bool__(self):
        return self.n > 0


def _knowable_at(doc: dict) -> str:
    """accepted_at if known, else the filing date padded to END of that day —
    a date-only filing is conservatively treated as knowable late, never early."""
    ts = doc.get("accepted_at")
    if ts:
        return ts
    fd = doc.get("filing_date")
    return f"{fd}T23:59:59Z" if fd else ""


# PR 2g: below this many prior filings in a section's corpus the idf is noise
# (a handful of docs make every word look rare). UNTUNED theory default, set
# before looking at any return; the library default stays 0 (off) so PR 2f
# behaviour is unchanged, the CLI --pit run applies this floor.
MIN_PIT_CORPUS = 30


def pit_rows_from_archive(archive: str, filings: Optional[list] = None, min_corpus: int = 0) -> list:
    """Like rows_from_archive, but each row's tfidf cosine uses an idf built
    ONLY from sections of filings knowable strictly before that row's
    accepted_at (the row's own filing and same-instant filings excluded).
    Rows with an empty corpus get cosine_tfidf=None (nothing to weight by); with
    min_corpus > 0 so do sections whose corpus holds fewer than min_corpus docs."""
    filings = filings if filings is not None else archive_filings(archive)
    load = _loader(archive)
    docs = {}
    for f in filings:
        d = load(f["cik"], f["accession"])
        if d:
            docs[f["accession"]] = d
    # corpus events: (knowable-at, per-section token sets), oldest first
    events = []
    for d in docs.values():
        ts = _knowable_at(d)
        if not ts:
            continue    # undated filing can never be proven knowable; never used as corpus
        sets = {k: set(tokenize(sec["text"])) for k, sec in (d.get("sections") or {}).items()
                if not exclusion_reason(sec)}
        events.append((ts, sets))
    events.sort(key=lambda e: e[0])
    pairs = sft.pair_prior_filings(filings)
    by_acc = {f["accession"]: f for f in filings}
    todo = [(acc, p) for acc, p in pairs.items()
            if p["prior"] and acc in docs and p["prior"] in docs and _knowable_at(docs[acc])]
    todo.sort(key=lambda t: _knowable_at(docs[t[0]]))
    idf, i, rows = {}, 0, []
    for acc, p in todo:
        cutoff = _knowable_at(docs[acc])
        while i < len(events) and events[i][0] < cutoff:
            for key, toks in events[i][1].items():
                idf.setdefault(key, PitIdf()).add(toks)
            i += 1
        usable = {k: v for k, v in idf.items() if v.n >= min_corpus} if min_corpus > 0 else idf
        row = pair_similarity(docs[acc], docs[p["prior"]], usable)
        row["gap_days"] = p["gap_days"]
        row["idf_pit"] = True
        row["idf_min_corpus"] = min_corpus
        row["idf_n"] = {k: v.n for k, v in idf.items()}
        rows.append(row)
    return rows


def _q(vals: list, q: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def distribution_report(rows: list) -> str:
    """Per (form, section): scored n, exclusion counts, cosine/Jaccard quantiles
    and the share of scored pairs below 0.9. Descriptive only."""
    cells = {}
    for r in rows:
        for key, s in r["sections"].items():
            c = cells.setdefault((r["form"], key), {"cos": [], "jac": [], "tf": [], "ex": Counter()})
            if s["excluded"]:
                c["ex"][s["excluded"]] += 1
            else:
                c["cos"].append(s["cosine"])
                c["jac"].append(s["jaccard"])
                if s.get("cosine_tfidf") is not None:
                    c["tf"].append(s["cosine_tfidf"])
    lines = [f"pairs: {len(rows)}",
             "form      section  n_scored  cos p05/p25/p50/p75/p95        <0.9   jac p05/p50/p95      tfidf p05/p50/p95    excluded"]
    for (form, key), c in sorted(cells.items()):
        n = len(c["cos"])
        f = lambda v: "  -  " if v is None else f"{v:.3f}"
        cq = "/".join(f(_q(c["cos"], x)) for x in (.05, .25, .5, .75, .95))
        jq = "/".join(f(_q(c["jac"], x)) for x in (.05, .5, .95))
        tq = "/".join(f(_q(c["tf"], x)) for x in (.05, .5, .95))
        lt = f"{sum(v < 0.9 for v in c['cos']) / n:5.1%}" if n else "  -  "
        lines.append(f"{form:<9} {key:<8} {n:>8}  {cq:<30} {lt:>6}   {jq:<20} {tq:<20} {dict(c['ex'])}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("distribution", help="similarity distribution over an archive")
    d.add_argument("--archive", default=None)
    d.add_argument("--pit", action="store_true", help="point-in-time idf (gate-2 safe) instead of whole-archive idf")
    d.add_argument("--min-corpus", type=int, default=MIN_PIT_CORPUS,
                   help="with --pit: null the tfidf score where the section corpus is smaller than this")
    d.add_argument("--rows-json", default=None, help="also write the scored rows here")
    args = ap.parse_args(argv)
    archive = args.archive or sft.archive_base_dir()
    rows = pit_rows_from_archive(archive, min_corpus=args.min_corpus) if args.pit else rows_from_archive(archive)
    print(distribution_report(rows))
    if args.rows_json:
        with open(args.rows_json, "w") as f:
            json.dump(rows, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
