#!/usr/bin/env python3
"""gem_methane_gate2c.py — GEM METHANE-PLUME × EXTRACTION-REGISTRY
PROXIMITY hypothesis, GATE-2(c) (SIGNAL layer of the ROOT VALIDATION
LADDER). Filed against research/open_questions.md's entry for this
hypothesis; direct continuation of gate-2(a) (server/gemMethaneProximity.ts,
2026-07-19) and gate-2(b) (computeAssetPlumeStats, 2026-07-20, v1.0.421).

QUESTION (verbatim from gate-2(b)'s own filed NEXT STEP): "do assets with
ANY nearby plume underperform peers with none, before conditioning on
rate/magnitude at all?" — a same-universe BASE RATE test (REASONING
STANDARD #3), tested here for the FIRST time against real data.

PRIOR — STATED BEFORE ANY RETURN WAS COMPUTED (REASONING STANDARD #10):
expect INSUFFICIENT SAMPLE at every horizon. Two compounding reasons: (a)
the operator/owner/parent -> public-ticker resolution below is
deliberately precision-first (exact-normalized-name or exact-GEM-Entity-ID
match only, ownership-chain walk for indirect parents, ambiguous
collisions dropped) — most GEM extraction/mine operators are private,
foreign-unlisted, or state-owned, so most of the ~1,000 matched plume
detections are expected to resolve to NO ticker at all, not a false one;
(b) methane-plume detection to public disclosure has an unmeasured
publication lag (see LOOKAHEAD CAVEAT below), diluting whatever signal
exists. If a usable N does emerge, direction is a coin flip a priori:
detected leaks could mean operational/regulatory risk (negative alpha)
OR could be immaterial to a large diversified operator's stock (no
alpha) — no directional prior is stated beyond "small if real."

LOOKAHEAD CAVEAT (REASONING STANDARD #7, stated honestly, not fixed here):
`observedAt` is the satellite's OWN detection timestamp (GEM's
"Observation Date"), not a recorded public-disclosure date — no such field
exists in this archive. GEM's underlying providers (Carbon Mapper /
GHGSat-class) typically publish detections with a lag of weeks to months;
this script's anchor-day convention (first trading day >= observedAt is
the "reaction" day, the prior close is the lookahead-free reference)
therefore UNDERSTATES true information latency — if anything this design
is favorable to finding a signal that a real trader could not have
captured this early. Any PASS verdict here is a preliminary data point
demanding further scrutiny of the true disclosure lag before ever being
treated as tradable, per the ladder's own gate-3/4 requirements.

TICKER RESOLUTION METHOD (precision-first, mirrors server/usaSpending.ts's
`normalizeCompanyName`/`buildTickerMap` collision-drop convention — same
codebase precedent, reused here rather than inventing new suffix logic):
  - COAL MINES: datacore/gem/coal_mine_tracker.json.gz carries GEM's own
    "Owners GEM Entity ID" / "Parent GEM Entity ID" columns — an ID join,
    not name-matching. Multi-owner rows (37 of 5,382, split ownership) are
    skipped entirely (no forced arbitrary pick).
  - OIL/GAS EXTRACTION: datacore/gem/oil_gas_extraction.json.gz's fields
    sheet carries ONLY free-text Operator/Owner(s)/Parent(s) — resolved by
    exact match against ownership.json.gz's entities' normalized Full
    Name/Name/Abbreviation (ambiguous normalized-name collisions across
    different Entity IDs are dropped, never guessed; Abbreviation added
    2026-07-23 gate-2(c) widening — GEM populates it on 1,833/26,250
    entities, e.g. short-form operator names that never match Full Name/
    Name verbatim).
  - BOTH: once an Entity ID is in hand (direct or name-matched), if that
    entity itself has no US SEC CIK, its own "Gem parents IDs" field
    (GEM's already-flattened immediate-to-ultimate ownership chain) is
    walked in order for the first ancestor that DOES carry a CIK — the
    NEAREST public ancestor, not forced to the ultimate parent if a closer
    one is already public. CIK -> ticker uses SEC's company_tickers.json
    (same source + suffix-preference de-dup as server/sec8kEarnings.ts's
    `getCikTickerMap`, ported here since this is a standalone Python
    script with no Node runtime dependency).

VERDICT RULE — PRE-STATED, PER HORIZON (never adjusted after seeing
results): MIN_SAMPLE = 20 distinct (ticker, detection-date) events with a
computable forward return at that horizon (smaller than
earnings_language_gate2.py's 30 — this universe of public energy/mining
operators is expected to be far smaller than the whole 8-K filer
population). N < MIN_SAMPLE => "INSUFFICIENT SAMPLE" (means still
reported as a preliminary data point, never a PASS/FAIL claim). N >=
MIN_SAMPLE: PASS requires the detection-group's mean alpha vs SPY to be
BELOW the same ticker-set's random-baseline-date mean alpha vs SPY by at
least PASS_SPREAD, i.e. detections underperform the base rate; else FAIL.
"alpha" = ticker's lookahead-free forward return minus SPY's forward
return over the identical trading-day window (REASONING STANDARD #3 base
rate). The random baseline is drawn from the SAME tickers (same universe,
same holding period) at dates outside a +/-max(HORIZONS) trading-day
exclusion window around every real detection for that ticker — this is
the base rate the question itself asks for ("peers with none" operationalized
as "this same instrument, on a random day with no detection nearby",
avoiding the much harder and less justifiable problem of picking a
cross-sectional "peer" universe of similar-but-undetected operators).
Baseline draws use a fixed seed (random.Random(1337)) for reproducibility
— never Python's unseeded global random state.

Usage:
  python3 scripts/gem_methane_gate2c.py
      [--api-url URL]          # default: production /api/data/methane-plumes
      [--gem-dir DIR]          # default: datacore/gem
      [--baseline-multiplier N]  # baseline draws per ticker = N * detections (default 5)
      [--out FILE]             # optional: also write full JSON result here
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from backtest_v2 import fetch_bars  # noqa: E402  (Alpaca-first/Yahoo-fallback, same data path as live)

MIN_SAMPLE = 20
PASS_SPREAD = 0.005  # 50 bps
HORIZONS = (5, 20, 60)
BASELINE_SEED = 1337
SEC_UA = {"User-Agent": "voltradeai-research/1.0 (research@voltradeai.com)"}
ID_RE = re.compile(r"E\d{6,}")

# ── Name normalization — ported verbatim (same tokens/logic) from
# server/usaSpending.ts's normalizeCompanyName/buildTickerMap, the
# established codebase precedent for name->ticker collision-drop matching
# (wishlist.md's "assignee->ticker mapping reuses the name-matcher
# pattern" note points at this same precedent). ─────────────────────────
SUFFIX_TOKENS = {
    "THE", "INC", "INCORPORATED", "CORP", "CORPORATION", "LLC", "L.L.C", "CO",
    "COMPANY", "LTD", "LIMITED", "LP", "L.P", "PLC", "HOLDINGS", "HOLDING",
    "GROUP", "INTERNATIONAL", "INTL",
}


def normalize_company_name(s: str | None) -> str:
    if not s:
        return ""
    tokens = [t for t in re.sub(r"[^A-Za-z0-9 ]+", " ", s.upper()).split() if t not in SUFFIX_TOKENS]
    out = " ".join(tokens).strip()
    if len(tokens) < 2 and len(out) < 5:
        return ""
    return out


# ── GEM data loading ─────────────────────────────────────────────────────
def load_gz(fp: str) -> dict:
    with gzip.open(fp, "rt") as f:
        return json.load(f)


def parse_ids(s: str | None) -> list[str]:
    return ID_RE.findall(s or "")


def build_ownership_index(gem_dir: str) -> dict:
    """Returns {entities_by_id, cik_by_entity, name_to_entity_id (collision-
    dropped)} from datacore/gem/ownership.json.gz."""
    d = load_gz(os.path.join(gem_dir, "ownership.json.gz"))
    entities = d["entities"]
    entities_by_id: dict[str, dict] = {}
    cik_by_entity: dict[str, str] = {}
    name_to_id: dict[str, str] = {}
    collided: set[str] = set()
    for e in entities:
        eid = e.get("Entity ID")
        if not eid:
            continue
        entities_by_id[eid] = e
        cik_raw = e.get("US SEC Central Index Key")
        if cik_raw and cik_raw != "not found":
            # ownership.json.gz zero-pads CIKs ("0001609253") and a few rows
            # carry a literal "CIK" prefix ("CIK0001422848") — SEC's own
            # company_tickers.json cik_str is a bare int (1609253) — same
            # mismatch class server/entityGraph.ts's ensureNode() already
            # guards with String(Number(cik)); strip non-digits then
            # normalize so every downstream lookup uses the SAME un-padded
            # key SEC's map does. A row with no digits at all is an honest
            # gap (skip), never coerced to 0.
            digits = re.sub(r"\D", "", cik_raw)
            if digits:
                cik_by_entity[eid] = str(int(digits))
        for field in ("Full Name", "Name", "Abbreviation"):
            key = normalize_company_name(e.get(field))
            if not key:
                continue
            if key in name_to_id and name_to_id[key] != eid:
                collided.add(key)
                continue
            name_to_id[key] = eid
    for k in collided:
        name_to_id.pop(k, None)
    return {"entities_by_id": entities_by_id, "cik_by_entity": cik_by_entity, "name_to_id": name_to_id}


def resolve_cik_from_entity_id(entity_id: str, idx: dict) -> str | None:
    """Direct CIK if the entity itself has one; else walks its own
    already-flattened 'Gem parents IDs' chain (immediate -> ultimate) for
    the FIRST ancestor with a CIK. Returns None (honest gap) if neither the
    entity nor any ancestor in the chain resolves."""
    if entity_id in idx["cik_by_entity"]:
        return idx["cik_by_entity"][entity_id]
    entity = idx["entities_by_id"].get(entity_id)
    if not entity:
        return None
    for pid in parse_ids(entity.get("Gem parents IDs")):
        if pid in idx["cik_by_entity"]:
            return idx["cik_by_entity"][pid]
    return None


def build_cik_ticker_map() -> dict[str, str]:
    """Ports server/sec8kEarnings.ts's getCikTickerMap: SEC lists multiple
    rows per CIK for multi-class issuers (e.g. common + warrants) — keep
    the non-suffixed (primary-class) ticker, never let whichever row
    appears last silently win."""
    req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=SEC_UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = json.loads(r.read())
    m: dict[str, str] = {}
    for row in raw.values():
        cik, ticker = row.get("cik_str"), row.get("ticker")
        if cik is None or not ticker:
            continue
        cik = str(cik)
        existing = m.get(cik)
        if existing is not None and "-" not in existing:
            continue
        if existing is not None and "-" in existing and "-" not in ticker:
            m[cik] = ticker
            continue
        if existing is None:
            m[cik] = ticker
    return m


def load_coal_entity_ids(gem_dir: str) -> dict[str, tuple[str | None, str | None]]:
    """GEM Mine ID -> (owner_entity_id, parent_entity_id), single-owner rows
    only (37 of 5,382 multi-owner split rows are skipped — no forced pick)."""
    d = load_gz(os.path.join(gem_dir, "coal_mine_tracker.json.gz"))
    out: dict[str, tuple[str | None, str | None]] = {}
    for r in d.get("non_closed", []):
        mine_id = r.get("GEM Mine ID")
        if not mine_id:
            continue
        owner_ids = parse_ids(r.get("Owners GEM Entity ID"))
        parent_ids = parse_ids(r.get("Parent GEM Entity ID"))
        owner_id = owner_ids[0] if len(owner_ids) == 1 else None
        parent_id = parent_ids[0] if len(parent_ids) == 1 else None
        out[mine_id] = (owner_id, parent_id)
    return out


def load_plumes(api_url: str) -> list[dict]:
    req = urllib.request.Request(api_url, headers={"User-Agent": "voltradeai-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read())
    return payload.get("plumes", [])


# ── Ticker resolution per plume ──────────────────────────────────────────
def resolve_ticker_for_plume(
    plume: dict, coal_ids: dict, ownership_idx: dict, cik_ticker: dict,
) -> tuple[str | None, str]:
    """Returns (ticker, method). method in {"coal_owner_id", "coal_parent_id",
    "name_operator", "name_owner", "name_parent", "none"}."""
    asset = plume.get("nearestAsset")
    if not asset or plume.get("ambiguousMatch"):
        return None, "none"
    if asset.get("kind") == "coal_mine":
        owner_id, parent_id = coal_ids.get(asset["id"], (None, None))
        for eid, method in ((owner_id, "coal_owner_id"), (parent_id, "coal_parent_id")):
            if not eid:
                continue
            cik = resolve_cik_from_entity_id(eid, ownership_idx)
            if cik and cik_ticker.get(cik):
                return cik_ticker[cik], method
        return None, "none"
    # oil_gas_extraction: name-matched
    for field, method in (("operator", "name_operator"), ("owner", "name_owner"), ("parent", "name_parent")):
        name = asset.get(field)
        key = normalize_company_name(name)
        if not key:
            continue
        eid = ownership_idx["name_to_id"].get(key)
        if not eid:
            continue
        cik = resolve_cik_from_entity_id(eid, ownership_idx)
        if cik and cik_ticker.get(cik):
            return cik_ticker[cik], method
    return None, "none"


# ── Price / return computation (same lookahead-safe convention as
# earnings_language_gate2.py's reaction_indices/forward_returns) ────────
def trading_day_indices(dates: list[str], event_date: str) -> tuple[int, int] | None:
    anchor_idx = next((i for i, d in enumerate(dates) if d >= event_date), None)
    if anchor_idx is None:
        return None
    pre_idx = anchor_idx - 1
    if pre_idx < 0:
        return None
    return pre_idx, anchor_idx


def forward_alpha(
    dates: list[str], closes: list[float], spy_map: dict[str, float],
    pre_idx: int, anchor_idx: int, horizons: tuple[int, ...],
) -> dict[int, float]:
    out = {}
    ref = closes[pre_idx]
    pre_date = dates[pre_idx]
    if not ref or pre_date not in spy_map:
        return out
    for h in horizons:
        idx = anchor_idx + h - 1
        if idx >= len(dates):
            continue
        as_of = dates[idx]
        if as_of not in spy_map:
            continue
        ret = closes[idx] / ref - 1
        spy_ret = spy_map[as_of] / spy_map[pre_date] - 1
        out[h] = ret - spy_ret
    return out


def compute(plumes: list[dict], coal_ids: dict, ownership_idx: dict, cik_ticker: dict,
            baseline_multiplier: int) -> dict:
    events_by_ticker: dict[str, set[str]] = {}
    method_counts: dict[str, int] = {}
    for p in plumes:
        ticker, method = resolve_ticker_for_plume(p, coal_ids, ownership_idx, cik_ticker)
        method_counts[method] = method_counts.get(method, 0) + 1
        if not ticker or not p.get("observedAt"):
            continue
        date = p["observedAt"][:10]
        events_by_ticker.setdefault(ticker, set()).add(date)

    unique_tickers = sorted(events_by_ticker)
    max_h = max(HORIZONS)

    spy_bars = fetch_bars("SPY", 2600)
    spy_map = dict(zip(spy_bars["date"], spy_bars["close"]))

    per_horizon_detect: dict[int, list[float]] = {h: [] for h in HORIZONS}
    per_horizon_baseline: dict[int, list[float]] = {h: [] for h in HORIZONS}
    fetch_errors = []
    ticker_detail = []

    for ticker in unique_tickers:
        try:
            bars = fetch_bars(ticker, 2600)
        except Exception as e:  # noqa: BLE001 — one bad ticker never kills the run
            fetch_errors.append({"ticker": ticker, "err": str(e)[:150]})
            continue
        dates, closes = bars.get("date") or [], bars.get("close") or []
        if not dates:
            fetch_errors.append({"ticker": ticker, "err": "no bars returned"})
            continue

        detect_dates = sorted(events_by_ticker[ticker])
        detect_anchor_idxs = []
        ticker_alphas: dict[int, list[float]] = {h: [] for h in HORIZONS}
        for d in detect_dates:
            ri = trading_day_indices(dates, d)
            if ri is None:
                continue
            pre_idx, anchor_idx = ri
            detect_anchor_idxs.append(anchor_idx)
            fwd = forward_alpha(dates, closes, spy_map, pre_idx, anchor_idx, HORIZONS)
            for h, a in fwd.items():
                per_horizon_detect[h].append(a)
                ticker_alphas[h].append(a)

        # Baseline: random anchor indices for this SAME ticker, excluded
        # within +/-max_h trading days of any real detection anchor, deep
        # enough into the series that a pre_idx and every horizon exist.
        excluded = set()
        for ai in detect_anchor_idxs:
            excluded.update(range(max(0, ai - max_h), min(len(dates), ai + max_h + 1)))
        candidates = [i for i in range(1, len(dates) - max_h) if i not in excluded]
        # crc32, not the builtin hash(): Python randomizes str hash() per
        # process (PYTHONHASHSEED) by default, which would have silently
        # broken the reproducibility this script's own header promises —
        # confirmed live (two runs of hash("CRC") in this same sandbox
        # returned different values) before shipping this fix.
        rng = random.Random(BASELINE_SEED + zlib.crc32(ticker.encode()) % 100_000)
        draw_n = min(len(candidates), max(1, baseline_multiplier) * max(1, len(detect_anchor_idxs)))
        baseline_idxs = rng.sample(candidates, draw_n) if candidates else []
        for anchor_idx in baseline_idxs:
            fwd = forward_alpha(dates, closes, spy_map, anchor_idx - 1, anchor_idx, HORIZONS)
            for h, a in fwd.items():
                per_horizon_baseline[h].append(a)

        ticker_detail.append({
            "ticker": ticker, "detection_dates": detect_dates, "detection_count": len(detect_dates),
            "mean_alpha_by_horizon": {
                str(h): (round(sum(vals) / len(vals), 5) if vals else None) for h, vals in ticker_alphas.items()
            },
            "computable_events_by_horizon": {str(h): len(vals) for h, vals in ticker_alphas.items()},
        })

    # OUTLIER-ROBUSTNESS DIAGNOSTIC (mirrors earnings_language_gate2.py's
    # own precedent: a small cross-sectional N is easily dominated by one
    # name's idiosyncratic price path, not a real cross-operator effect —
    # NOT part of the pre-stated PASS/FAIL rule, reported as a cross-check
    # that must accompany any PASS/FAIL claim, never silently omitted).
    concentration = {}
    for h in HORIZONS:
        contributions = [
            (t["ticker"], t["mean_alpha_by_horizon"][str(h)] * t["computable_events_by_horizon"][str(h)])
            for t in ticker_detail if t["mean_alpha_by_horizon"][str(h)] is not None
        ]
        total_abs = sum(abs(c) for _, c in contributions)
        if not contributions or total_abs == 0:
            concentration[str(h)] = None
            continue
        top_ticker, top_contribution = max(contributions, key=lambda tc: abs(tc[1]))
        concentration[str(h)] = {
            "dominant_ticker": top_ticker,
            "dominant_share_of_total_abs_alpha": round(abs(top_contribution) / total_abs, 3),
        }

    horizon_results = {}
    for h in HORIZONS:
        detect, baseline = per_horizon_detect[h], per_horizon_baseline[h]
        n = len(detect)
        entry = {
            "n_detections": n, "n_baseline": len(baseline),
            "detect_mean_alpha": round(sum(detect) / n, 5) if n else None,
            "baseline_mean_alpha": round(sum(baseline) / len(baseline), 5) if baseline else None,
        }
        if n < MIN_SAMPLE:
            entry["verdict"] = "INSUFFICIENT SAMPLE"
            entry["note"] = f"n={n} < MIN_SAMPLE={MIN_SAMPLE}; means logged as a preliminary data point only, no PASS/FAIL claim"
        else:
            spread = (entry["detect_mean_alpha"] - entry["baseline_mean_alpha"]) if baseline else None
            entry["spread_vs_baseline"] = round(spread, 5) if spread is not None else None
            entry["verdict"] = "PASS" if (spread is not None and spread <= -PASS_SPREAD) else "FAIL"
            conc = concentration.get(str(h))
            entry["concentration"] = conc
            if conc and conc["dominant_share_of_total_abs_alpha"] >= 0.5:
                entry["note"] = (
                    f"NOT CROSS-SECTIONALLY ROBUST — {conc['dominant_ticker']} alone accounts for "
                    f"{conc['dominant_share_of_total_abs_alpha']:.0%} of total |alpha| at this horizon; "
                    "this verdict is dominated by one name's idiosyncratic price path, not a "
                    "repeatable cross-operator effect (REASONING STANDARD #4/#8)"
                )
        horizon_results[str(h)] = entry

    return {
        "prior": "INSUFFICIENT SAMPLE expected at every horizon (stated in this script's own "
                 "header before any return was computed) — most GEM operators are private/foreign/"
                 "state-owned and will not resolve to a ticker; no directional prior beyond 'small if real'",
        "lookahead_caveat": "observedAt is GEM's satellite detection date, not a recorded public-"
                             "disclosure date (none exists in this archive) — true information "
                             "latency is unmeasured and likely understated by this script's anchor-day "
                             "convention; any PASS here is preliminary, not tradable",
        "resolution_counts": method_counts,
        "unique_tickers_resolved": len(unique_tickers),
        "tickers": ticker_detail,
        "ticker_fetch_errors": fetch_errors,
        "verdict_rule": f"n>={MIN_SAMPLE} required for PASS/FAIL; PASS requires detect_mean_alpha "
                         f"<= baseline_mean_alpha - {PASS_SPREAD}; else FAIL; n<{MIN_SAMPLE} => "
                         "INSUFFICIENT SAMPLE regardless of the numbers",
        "horizons": horizon_results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="https://voltradeai-production.up.railway.app/api/data/methane-plumes")
    ap.add_argument("--gem-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datacore", "gem"))
    ap.add_argument("--baseline-multiplier", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    plumes = load_plumes(args.api_url)
    coal_ids = load_coal_entity_ids(args.gem_dir)
    ownership_idx = build_ownership_index(args.gem_dir)
    cik_ticker = build_cik_ticker_map()

    result = compute(plumes, coal_ids, ownership_idx, cik_ticker, args.baseline_multiplier)
    result["source"] = {"plumes_api": args.api_url, "gem_dir": args.gem_dir, "plumes_loaded": len(plumes)}

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)

    print(json.dumps(result, indent=1))
    any_insufficient = any(h["verdict"] == "INSUFFICIENT SAMPLE" for h in result["horizons"].values())
    any_fail = any(h["verdict"] == "FAIL" for h in result["horizons"].values())
    return 2 if any_insufficient and not any_fail else (3 if any_fail else 0)


if __name__ == "__main__":
    sys.exit(main())
