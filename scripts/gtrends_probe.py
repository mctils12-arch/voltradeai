#!/usr/bin/env python3
"""
gtrends_probe.py — Google Trends GATE-1 STABILITY PROBE (stream #8 of the
DATA STREAM EXPANSION order — deliberately a PROBE, not a stream build).

Context (research 2026-07-05): pytrends upstream was archived read-only
2025-04-17 (abandoned); Google's official Trends API is invite-only alpha.
The filed ladder says gate 1 (series stability across re-pulls) may kill
this stream — this script IS that gate-1 test. NO archiver, NO manifest,
NO daemon route exists or will exist unless this passes (no dead code for
a stream likely dead at gate 1).

METHOD: pull a small-cap consumer BRAND panel (brand search terms, not
tickers — people search "Sweetgreen", not "SG") N times with spaced
intervals; per term, compute Pearson r between consecutive pulls of the
identical query. PASS CRITERION (stated before the run, Reasoning
Standard #10): median cross-pull r > 0.95 AND >= 80% of panel terms
successfully pulled in every round (429s/blocks count as failures).

Usage: python3 scripts/gtrends_probe.py [--pulls 3] [--gap-min 10]
Writes JSON results to stdout; the session appends the verdict to
research/experiments.md (layer-of-death log if failed).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

# Small-cap / search-visible consumer brands (EDGE doctrine: fish where
# whales can't; brand demand is search-visible).
PANEL = [
    ("BARK", "barkbox"), ("WRBY", "warby parker"), ("FIGS", "figs scrubs"),
    ("OLPX", "olaplex"), ("CAVA", "cava restaurant"), ("DNUT", "krispy kreme"),
    ("GRPN", "groupon"), ("REAL", "the realreal"), ("LOVE", "lovesac"),
    ("PRPL", "purple mattress"), ("YETI", "yeti cooler"), ("CROX", "crocs"),
    ("SFIX", "stitch fix"), ("WW", "weightwatchers"), ("CHWY", "chewy"),
    ("ELF", "elf cosmetics"), ("SG", "sweetgreen"), ("PTON", "peloton"),
    ("BYND", "beyond meat"), ("WEN", "wendys"),
]


def pull_panel(timeframe: str = "today 3-m", geo: str = "US"):
    """One full panel pull. Returns {term: [values...]} plus error list."""
    from pytrends.request import TrendReq

    series, errors = {}, []
    # NO retries kwarg: pytrends (abandoned upstream 2025-04) constructs
    # urllib3 Retry with the v1-only `method_whitelist` argument when
    # retries>0 — TypeError before any request is made. Discovered in the
    # first probe run 2026-07-05 (all 60 pulls died on it; a retries-free
    # control pull succeeded). Politeness spacing handles transient 429s.
    py = TrendReq(hl="en-US", tz=0, timeout=(10, 30))
    # pytrends compares up to 5 terms per request; pull one-at-a-time so each
    # series is independently normalized (comparability across pulls is the
    # point — a shared-normalization batch would confound the stability test).
    for _tkr, term in PANEL:
        try:
            py.build_payload([term], timeframe=timeframe, geo=geo)
            df = py.interest_over_time()
            if df is None or df.empty or term not in df.columns:
                errors.append({"term": term, "err": "empty"})
                continue
            series[term] = [int(v) for v in df[term].tolist()]
        except Exception as e:  # noqa: BLE001 — 429s and blocks ARE the data here
            errors.append({"term": term, "err": str(e)[:120]})
        time.sleep(3)  # politeness spacing within a pull
    return series, errors


def pearson(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return None
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    if va == 0 or vb == 0:
        return 1.0 if va == vb else 0.0
    return cov / (va * vb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulls", type=int, default=3)
    ap.add_argument("--gap-min", type=float, default=10)
    args = ap.parse_args()

    rounds = []
    for i in range(args.pulls):
        t0 = time.time()
        series, errors = pull_panel()
        rounds.append({"series": series, "errors": errors, "ok": len(series)})
        print(f"round {i + 1}/{args.pulls}: {len(series)}/{len(PANEL)} terms ok, "
              f"{len(errors)} errors, {time.time() - t0:.0f}s", file=sys.stderr)
        if i < args.pulls - 1:
            time.sleep(args.gap_min * 60)

    # stability: per-term Pearson r between consecutive rounds
    rs = []
    per_term = {}
    for a, b in zip(rounds, rounds[1:]):
        for term in a["series"]:
            if term in b["series"]:
                r = pearson(a["series"][term], b["series"][term])
                if r is not None:
                    rs.append(r)
                    per_term.setdefault(term, []).append(round(r, 4))
    rs.sort()
    median_r = rs[len(rs) // 2] if rs else None
    ok_rates = [r["ok"] / len(PANEL) for r in rounds]
    verdict = (median_r is not None and median_r > 0.95 and min(ok_rates) >= 0.8)

    print(json.dumps({
        "at": datetime.now(timezone.utc).isoformat(),
        "pulls": args.pulls,
        "gap_min": args.gap_min,
        "panel_size": len(PANEL),
        "ok_per_round": [r["ok"] for r in rounds],
        "errors_per_round": [len(r["errors"]) for r in rounds],
        "sample_errors": (rounds[0]["errors"] + rounds[-1]["errors"])[:6],
        "median_cross_pull_r": median_r,
        "min_r": rs[0] if rs else None,
        "per_term_r": per_term,
        "PASS_CRITERION": "median r > 0.95 AND >= 80% terms ok every round",
        "gate1_verdict": "PASS" if verdict else "FAIL",
    }, indent=1))


if __name__ == "__main__":
    main()
