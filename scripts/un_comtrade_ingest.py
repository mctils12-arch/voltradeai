#!/usr/bin/env python3
"""
un_comtrade_ingest.py — UN Comtrade bilateral goods-trade archive
(EDGE DOCTRINE axis (a), CLAUDE.md-named example; DATA CENSUS #10, Section
1: "UN Comtrade preview — keyless SDMX-JSON, probed 200 ... too lagged for
direct alpha, structural-thesis input only." Was the sole `candidate_unbuilt`
entry left in scripts/data_stream_registry_check.py's CANDIDATES table —
every other axis-(a) census item is already built or correctly declined
(re-confirmed dozens of times across sessions per that script's own
docstring); this closes the last open one rather than re-deriving that
same "is there anything left to build" question by hand again.

WHY (hypothesis, gate-locked — this session builds GATE 1 DATA only, no
trading logic touched): USA's month-over-month bilateral trade balance
with its largest partners is a slow, structural regime feature — tariff
escalation/de-escalation, reshoring/friend-shoring shifts (Mexico/Vietnam
share of imports rising as China's falls) show up here months before they
show up in any single company's earnings call. EDGE DOCTRINE #2 cuts
AGAINST this root (CBOT-grain-style: professionally covered, not a small/
illiquid corner) — the census's own prior was "too lagged for direct
alpha, structural-thesis input only," which is exactly the frame this
build keeps: RAW archive, no GATE 2 (SIGNAL) attempt this session.

SOURCE (probed live 2026-09-07): comtradeapi.un.org/public/v1/preview/C/M/HS,
keyless, monthly frequency. CONFIRMED CONSTRAINT (live 400 response, not
guessed): "Maximum number of periods for preview is 1" — one period per
request. Comma-joined partnerCode/flowCode DO combine into a single
request (verified live: 6 partners x 2 flows = 12 records in one call),
which is how this script keeps its request budget low: one HTTP call per
calendar month covering every partner and both flow directions at once.

REPORTER: USA (842) only. PARTNERS: the 6 chosen ALL have an independent,
keyless, official U.S. Census Bureau reference series on FRED (IMPCH/
IMPMX/IMPCA/IMPJP/IMPGE/IMPKR) for scripts/un_comtrade_gate1.py to
reconcile against — partners without an equally rigorous independent
check (India, Vietnam, etc.) are deliberately left out of v1 rather than
archived unvalidated; a future session can add them with their own gate-1
partner. cmdCode=TOTAL (the UN-computed aggregate across all HS lines,
`isAggregate: true` — every probed response carries that flag; it is NOT
evidence of estimation, per the API's own "TOTAL is never itself a
reported line" convention) for v1 — HS-6 commodity-level detail is a
documented future follow-up (research/open_questions.md), not attempted
here (one logical change per PR; would also multiply the request count).

STORAGE: datacore/un_comtrade/bilateral_trade.json — APPEND-ONLY accumulator
(unlike JODI's whole-file-rebuild: Comtrade never revises a closed month
in practice the way JODI's own monthly re-release does, and re-fetching
already-archived months would waste request budget for no benefit). A
period already present for a given partner|flow series is never re-fetched
or overwritten — same "captured once, never re-attempted" discipline as
server/portDwellCapture.ts's captureIfDue().

Run (session-side; keyless, no Railway wiring — same "seeded pattern" as
scripts/jodi_oil.py/scripts/gem_ingest.py, per their own manifests'
`written_by` convention):
  python3 scripts/un_comtrade_ingest.py [--backfill-months N] [--out PATH]
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone

BASE_URL = "https://comtradeapi.un.org/public/v1/preview/C/M/HS"
REPORTER_CODE = 842
REPORTER_NAME = "USA"
# partnerCode -> (name, FRED series id used by scripts/un_comtrade_gate1.py)
PARTNERS = {
    156: ("China", "IMPCH"),
    484: ("Mexico", "IMPMX"),
    124: ("Canada", "IMPCA"),
    392: ("Japan", "IMPJP"),
    276: ("Germany", "IMPGE"),
    410: ("South Korea", "IMPKR"),
}
FLOWS = ("M", "X")  # M = imports (US from partner), X = exports (US to partner)
CMD_CODE = "TOTAL"
DEFAULT_BACKFILL_MONTHS = 24
OUT = os.path.join(os.path.dirname(__file__), "..", "datacore", "un_comtrade", "bilateral_trade.json")
ATTRIBUTION = "UN Comtrade Database, https://comtradeapi.un.org"
LICENSE = "free with citation; bulk redistribution of the raw database needs UN Comtrade permission (informational/derived-signal use here, not raw resale)"
REQUEST_DELAY_S = 1.0
MAX_RETRIES = 3


def month_range_desc(end_period: str, n_months: int) -> list:
    """Returns the last n_months YYYYMM strings up to and including
    end_period, newest first. Pure, no I/O — end_period is a caller-passed
    'now' string so tests never depend on the real wall clock."""
    y, m = int(end_period[:4]), int(end_period[4:6])
    out = []
    for _ in range(n_months):
        out.append(f"{y:04d}{m:02d}")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return out


def parse_response(body: dict, period: str) -> list:
    """Raw Comtrade JSON -> list of {partner, flow, cif, fob}. Raises on a
    populated `error` field (a real API error, e.g. the 1-period-per-request
    cap) rather than silently returning nothing — an error response must
    never be mistaken for 'no data yet'. A response with count==0 and no
    error (period genuinely not published yet) returns []."""
    if body.get("error"):
        raise RuntimeError(f"UN Comtrade API error for period {period}: {body['error']}")
    out = []
    for row in body.get("data", []):
        if row.get("period") != period:
            continue  # defensive: never trust a row for a period we didn't ask for
        partner = row.get("partnerCode")
        flow = row.get("flowCode")
        if partner not in PARTNERS or flow not in FLOWS:
            continue
        out.append({
            "partner": partner,
            "flow": flow,
            "cif": row.get("cifvalue"),
            "fob": row.get("fobvalue"),
        })
    return out


def merge_into_series(series: dict, period: str, records: list) -> int:
    """Mutates `series` ({"<partner>|<flow>": {"points": [[period, cif, fob], ...]}})
    in place, adding one point per record UNLESS that partner|flow series
    already has this exact period archived (append-only, never re-fetch/
    overwrite a captured month — see module docstring). Returns the count
    of NEW points actually added, so callers can tell a no-op run from a
    real one without re-reading the whole structure."""
    added = 0
    for rec in records:
        key = f"{rec['partner']}|{rec['flow']}"
        s = series.setdefault(key, {"points": []})
        if any(p[0] == period for p in s["points"]):
            continue
        s["points"].append([period, rec["cif"], rec["fob"]])
        added += 1
    for s in series.values():
        s["points"].sort(key=lambda p: p[0])
    return added


def build_artifact(series: dict, now_iso: str) -> dict:
    for key, s in series.items():
        pts = s["points"]
        s["n"] = len(pts)
        s["first"] = pts[0][0] if pts else None
        s["last"] = pts[-1][0] if pts else None
    latest = max((s["last"] for s in series.values() if s["last"]), default=None)
    return {
        "source": BASE_URL,
        "attribution": ATTRIBUTION,
        "license": LICENSE,
        "built_at": now_iso,
        "reporter": {"code": REPORTER_CODE, "name": REPORTER_NAME},
        "partners": {str(code): name for code, (name, _fred) in PARTNERS.items()},
        "flows": {"M": "imports (US from partner)", "X": "exports (US to partner)"},
        "cmd_code": CMD_CODE,
        "value_fields": {
            "cif": "CIF import value, USD (cost+insurance+freight; null for export rows)",
            "fob": "FOB value, USD (free-on-board; populated for both import and export rows)",
        },
        "latest_period": latest,
        "series_count": len(series),
        "series": series,
    }


def _fetch_period(period: str, timeout: int = 30) -> dict:
    partner_codes = ",".join(str(c) for c in PARTNERS)
    flow_codes = ",".join(FLOWS)
    url = (
        f"{BASE_URL}?reporterCode={REPORTER_CODE}&partnerCode={partner_codes}"
        f"&period={period}&cmdCode={CMD_CODE}&flowCode={flow_codes}"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "voltradeai-datacore/1.0"})
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
    raise RuntimeError(f"period {period}: failed after {MAX_RETRIES} attempts: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill-months", type=int, default=DEFAULT_BACKFILL_MONTHS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        series = existing.get("series", {})
        print(f"loaded existing archive: {existing.get('series_count', 0)} series, latest {existing.get('latest_period')}")
    else:
        series = {}
        print("no existing archive — starting fresh")

    now = datetime.now(timezone.utc)
    current_period = f"{now.year:04d}{now.month:02d}"
    candidates = month_range_desc(current_period, args.backfill_months)
    # skip periods every series already has fully covered (all 6 partners x
    # 2 flows = 12 records) — cheap, avoids re-spending request budget on
    # an already-complete month even across repeated runs.
    already_full = {
        p for p in candidates
        if sum(1 for s in series.values() if any(pt[0] == p for pt in s["points"])) >= len(PARTNERS) * len(FLOWS)
    }
    to_fetch = [p for p in candidates if p not in already_full]
    print(f"fetching {len(to_fetch)}/{len(candidates)} periods (skipping {len(already_full)} already-complete)")

    total_added = 0
    fetched_ok = 0
    empty_periods = []
    for i, period in enumerate(to_fetch):
        try:
            body = _fetch_period(period)
            records = parse_response(body, period)
        except RuntimeError as e:
            print(f"  {period}: SKIP ({e})")
            continue
        if not records:
            empty_periods.append(period)
            print(f"  {period}: not yet published (0 records)")
        else:
            added = merge_into_series(series, period, records)
            total_added += added
            fetched_ok += 1
            print(f"  {period}: {len(records)} records, {added} new points")
        if i < len(to_fetch) - 1:
            time.sleep(REQUEST_DELAY_S)

    if total_added == 0 and not series:
        print("zero series and zero new points — refusing to write an empty archive")
        return 1

    artifact = build_artifact(series, now.isoformat())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=1, sort_keys=True)
        f.write("\n")
    print(
        f"WROTE {args.out}: {artifact['series_count']} series, latest {artifact['latest_period']}, "
        f"{total_added} new points this run ({fetched_ok} periods fetched, {len(empty_periods)} not yet published)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
