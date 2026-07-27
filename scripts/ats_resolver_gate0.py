#!/usr/bin/env python3
"""
ats_resolver_gate0.py — GATE 0 (feasibility) probe for the job-postings /
ATS-panel data root named in research/open_questions.md's NEW DATA ROOTS
item 2 ("Job postings via ATS public JSON — hiring velocity / role mix").

CONTEXT (EDGE DOCTRINE #1, BUILD-FIRST RULE): Greenhouse/Lever/Ashby/
SmartRecruiters all expose free, keyless, public job-board JSON for
companies that use them — no ToS-prohibited scraping, no paid API. The
open question the item's own ladder flags before anything else is built:
"Russell-2000 ATS coverage is UNMEASURED — gate 0 exists to kill that
unknown." This script IS gate 0. Per the gtrends_probe.py precedent: this
is deliberately a PROBE, not a stream build — NO archiver, NO manifest,
NO server route, NO daemon wiring exists or will exist unless the
measured coverage clears the item's own stated bar (~10%).

METHOD: sample a small/micro-cap NASDAQ Capital Market panel (same live
symbol-directory + filter as illiquid_universe_probe.py, so the panel is
reproducible and not cherry-picked), derive board-token candidates from
each company's OWN listed security name (no guessing from training-data
memory of "which company uses which ATS" — that would be exactly the
kind of unverified assumption CLAUDE.md's READ BEFORE WRITE / no-
fabrication discipline forbids), and probe all four public ATS APIs for
each candidate slug. A schema-level "hit" requires a 200 response AND a
body that parses into the platform's real job-list schema — a
404/never-registered slug is the overwhelmingly common outcome and is
not itself informative; what's informative is the AGGREGATE hit rate
across the panel. **A schema-level hit is NOT an identity-confirmed
match** — see HONEST LIMITATION #5, the single most important finding
of this session's run.

HONEST LIMITATIONS (state before anyone reuses this result):
  1. NAME-DERIVED SLUG GUESSING UNDER-COUNTS real coverage: many real
     ATS boards use a company's internal short name or a historical
     rebrand, not a mechanical transform of the SEC-listed security
     name (e.g. a board token chosen at initial ATS signup years before
     a later name change). This script cannot find those — it measures
     "coverage discoverable by name-transform guessing," a LOWER BOUND
     on true coverage, not true coverage itself.
  2. SmartRecruiters' public postings endpoint is keyed by an internal
     numeric/company identifier for most accounts, not a guessable
     name — near-zero hits here are expected and are NOT informative
     about whether a company uses SmartRecruiters at all. WORSE,
     VERIFIED LIVE THIS SESSION: unlike the other three platforms
     (which correctly 404 an unknown slug), SmartRecruiters 200s
     `{"content": []}` for ANY identifier, real or fabricated — a
     schema match alone is not a valid hit signal here, so
     classify_response() additionally requires non-empty `content`,
     which under-counts real companies with zero currently-open
     postings. The first version of this script did not apply that
     filter and reported a false ~90%+ "coverage" purely from this
     endpoint's non-404 behavior on garbage slugs — caught by manually
     verifying one candidate's raw response before trusting the
     aggregate, not by the schema-shape check alone.
  3. SINGLE SAMPLE DRAW, no re-draw, no significance test — same
     caveat illiquid_universe_probe.py already states for its own panel;
     this is a feasibility gate, not a validated statistic.
  4. Panel is NASDAQ Capital Market (smallest listing tier) — a proxy
     for "small cap," not literally the Russell 2000 the item's own
     text names (no free, keyless Russell 2000 constituent list was
     available this session); a future session could substitute a real
     index constituent list without changing anything else here.
  5. STRUCTURAL FALSE-POSITIVE MODE, FOUND AND CONFIRMED THIS SESSION
     (not theoretical): a first-word or full-name slug guess that
     happens to be a common English word collides with an UNRELATED
     company that also chose that word as its brand — and Greenhouse/
     Ashby/Lever are disproportionately used by VC-backed private
     startups that gravitate toward exactly these short, generic,
     one-word names ("Forward", "Eagle", "Universe", "Pioneer" all
     turned out to be real, unrelated startups' boards, not the small-
     cap tickers being probed). Manually inspected via each hit's own
     content this session (Greenhouse's job-level `company_name` field;
     Ashby's job description "About {X}" text): every one of this run's
     4 raw schema-level hits (FWDI->'forward', NUCL->'eagle',
     PPSI->'pioneer', UPC->'universe') was a FALSE MATCH on manual
     review, none the ticker's actual company. A same-first-word overlap
     check does NOT reliably fix this (the colliding company's real name
     literally IS "Forward" — token-overlap against the ticker's own
     name would still pass), so no automated identity filter was
     shipped; a `resolved: true` result from this script means schema-
     match only and MUST be manually content-verified before being
     trusted as a real coverage data point — see the run's own final
     numbers in research/experiments.md, where all 4 were reclassified
     false after this check.

Usage: python3 scripts/ats_resolver_gate0.py [--panel-size 25] [--sleep 0.35]
Prints a JSON coverage report to stdout; per-platform/per-ticker detail
to stderr. The session records the verdict in research/open_questions.md
(item 2) and research/experiments.md, per the gtrends_probe precedent —
no separate artifact file.
"""
import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request

NASDAQ_SYMDIR_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_EXCLUDE_NAME_SUBSTRINGS = ("Warrant", "Right", "Unit", "Acquisition", "Preferred",
                            "Depositary", "Rights", "Units")

_CORP_SUFFIX_WORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "ltd", "llc", "lp", "plc", "nv", "sa", "ag", "the", "holdings", "holding",
    "group", "common", "stock", "class", "a", "b", "c", "ordinary", "shares",
}

ATS_ENDPOINTS = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
    "lever": "https://api.lever.co/v0/postings/{slug}?mode=json",
    "ashby": "https://api.ashbyhq.com/posting-api/job-board/{slug}",
    "smartrecruiters": "https://api.smartrecruiters.com/v1/companies/{slug}/postings",
}

_UA = "Mozilla/5.0 (compatible; voltradeai-research-probe/1.0)"


def fetch_capital_market_panel(panel_size: int = 25):
    """Live NASDAQ Capital Market small-cap panel, same filter as
    illiquid_universe_probe.py's fetch_nasdaq_capital_market_candidates
    (Market Category S, non-ETF, non-test, common-stock-shaped names),
    keeping the Security Name (needed here for slug derivation, which
    the existing probe discards). Returns [(symbol, security_name), ...],
    systematically strided to panel_size (not cherry-picked)."""
    req = urllib.request.Request(NASDAQ_SYMDIR_URL, headers={"User-Agent": _UA})
    raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw), delimiter="|")
    clean = []
    for row in reader:
        if row.get("Market Category") != "S":
            continue
        if row.get("ETF") != "N" or row.get("Test Issue") != "N":
            continue
        name = row.get("Security Name", "")
        sym = row.get("Symbol", "")
        if any(b in name for b in _EXCLUDE_NAME_SUBSTRINGS):
            continue
        if "." in sym or len(sym) > 5:
            continue
        clean.append((sym, name))
    if not clean or panel_size <= 0:
        return []
    stride = max(1, len(clean) // panel_size)
    return clean[::stride][:panel_size]


def normalize_company_name(security_name: str) -> str:
    """Strip the SEC listing's share-class/security-type suffix and
    common corporate-entity words, leaving a bare brand-ish name. Pure
    string function — no network, unit-testable."""
    name = security_name.split(" - ")[0]
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[.,]", "", name)
    words = [w.lower() for w in name.split()]
    while words and words[-1] in _CORP_SUFFIX_WORDS:
        words.pop()
    return " ".join(words).strip()


def derive_slug_candidates(security_name: str, ticker: str) -> list:
    """Candidate ATS board-token guesses for one company. Pure function,
    no network — see HONEST LIMITATIONS #1 in the module docstring for
    why this necessarily undercounts true coverage.

    Prioritizes the full no-space name and the first-word-only guess
    over a hyphenated variant: a manual sanity check this session
    (Palantir Technologies Inc -> real Lever slug "palantir") showed the
    hyphenated encoding ("palantir-technologies") is highly correlated
    with the no-space one (if a board doesn't exist at the full no-space
    name it's very unlikely to exist at the same name hyphenated) and
    was crowding out first_word, which is often the ACTUAL slug for a
    single-recognizable-brand-word multi-word legal name — a materially
    more useful guess than a second encoding of the same full name.
    """
    base = normalize_company_name(security_name)
    company_candidates = []
    if base:
        no_space = re.sub(r"[^a-z0-9]", "", base)
        if no_space:
            company_candidates.append(no_space)
        first_word = re.sub(r"[^a-z0-9]", "", base.split()[0]) if base.split() else ""
        if first_word and first_word not in company_candidates:
            company_candidates.append(first_word)
    # cap company-derived variants at 2 so the ticker itself always keeps
    # a guaranteed slot below, rather than being crowded out by a 3-word name
    candidates = company_candidates[:2]
    tick = ticker.lower()
    if tick not in candidates:
        candidates.append(tick)
    return candidates[:3]


def classify_response(platform: str, status: int, body_text: str) -> dict:
    """Pure function: HTTP status + raw body -> hit/miss + job count.
    A 'hit' requires the body to parse into the platform's real
    job-list schema, not just a 200 — some ATS deployments 200 a
    generic error page instead of 404ing an unknown slug."""
    if status != 200:
        return {"hit": False, "job_count": None}
    try:
        data = json.loads(body_text)
    except ValueError:
        return {"hit": False, "job_count": None}
    if platform in ("greenhouse", "ashby"):
        jobs = data.get("jobs") if isinstance(data, dict) else None
        if not isinstance(jobs, list):
            return {"hit": False, "job_count": None}
        return {"hit": True, "job_count": len(jobs)}
    if platform == "lever":
        if not isinstance(data, list):
            return {"hit": False, "job_count": None}
        return {"hit": True, "job_count": len(data)}
    if platform == "smartrecruiters":
        # SmartRecruiters' public endpoint 200s with {"content": []} for
        # ANY identifier, real or fabricated (verified live this session:
        # https://api.smartrecruiters.com/v1/companies/<garbage>/postings
        # -> 200, empty content) — unlike the other three platforms, which
        # correctly 404 an unknown slug. A schema match alone is therefore
        # not a valid hit signal here; requiring non-empty content is the
        # only honest signal available, at the cost of under-counting real
        # companies that currently have zero open postings (see the
        # module's HONEST LIMITATIONS).
        content = data.get("content") if isinstance(data, dict) else None
        if not isinstance(content, list) or len(content) == 0:
            return {"hit": False, "job_count": None}
        return {"hit": True, "job_count": len(content)}
    return {"hit": False, "job_count": None}


def _fetch(url: str, timeout: float = 8.0):
    """Returns (status_code, body_text). Network errors and non-2xx/404
    HTTPErrors both resolve to (status, body) rather than raising —
    a probe script's job is to observe outcomes, not to treat a 404 as
    exceptional."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return None, str(e)[:200]


def probe_ticker(ticker: str, security_name: str, sleep_s: float = 0.35) -> dict:
    """Probes every (slug candidate x platform) combination for one
    ticker, stopping early the moment a schema-level hit is found
    (politeness + runtime). NOTE: `resolved: True` means a candidate
    slug matched the platform's job-list SCHEMA, not that the board has
    been confirmed to belong to this ticker's actual company — see
    HONEST LIMITATION #5 in the module docstring; manual content review
    is required before trusting any individual hit."""
    slugs = derive_slug_candidates(security_name, ticker)
    attempts = []
    for slug in slugs:
        for platform, url_tmpl in ATS_ENDPOINTS.items():
            url = url_tmpl.format(slug=slug)
            status, body = _fetch(url)
            result = classify_response(platform, status, body) if status is not None else {"hit": False, "job_count": None}
            attempts.append({"slug": slug, "platform": platform, "status": status, **result})
            time.sleep(sleep_s)
            if result["hit"]:
                return {
                    "ticker": ticker, "security_name": security_name,
                    "resolved": True, "platform": platform, "slug": slug,
                    "job_count": result["job_count"], "n_attempts": len(attempts),
                }
    return {
        "ticker": ticker, "security_name": security_name,
        "resolved": False, "platform": None, "slug": None,
        "job_count": None, "n_attempts": len(attempts),
    }


def summarize(rows: list) -> dict:
    """Pure aggregation over already-probed rows — no network, unit
    testable (mirrors illiquid_universe_probe.py's summarize())."""
    n = len(rows)
    hits = [r for r in rows if r.get("resolved")]
    by_platform = {}
    for r in hits:
        by_platform[r["platform"]] = by_platform.get(r["platform"], 0) + 1
    return {
        "n_tickers": n,
        "n_resolved": len(hits),
        "coverage_pct": round(100.0 * len(hits) / n, 1) if n else 0.0,
        "hits_by_platform": by_platform,
        "resolved_tickers": [r["ticker"] for r in hits],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-size", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()

    panel = fetch_capital_market_panel(args.panel_size)
    print(f"panel: {len(panel)} tickers", file=sys.stderr)

    rows = []
    for ticker, name in panel:
        row = probe_ticker(ticker, name, sleep_s=args.sleep)
        rows.append(row)
        tag = f"HIT:{row['platform']}/{row['slug']}" if row["resolved"] else "miss"
        print(f"  {ticker:6s} ({name[:40]:40s}) -> {tag}", file=sys.stderr)

    report = summarize(rows)
    report["panel"] = rows
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
