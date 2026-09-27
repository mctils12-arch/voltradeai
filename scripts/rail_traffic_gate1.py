#!/usr/bin/env python3
"""
rail_traffic_gate1.py — ROOT VALIDATION LADDER GATE 1 (DATA) for the
rail_ep724_carload_traffic root: reconciles datacore/rail/ep724_carloads.json
(7 Class I railroads' own STB EP724-reported weekly carloads, summed
system-wide) against TWO INDEPENDENT FRED/BTS-AAR national series — same
"external truth source, stable-ratio" discipline as
scripts/un_comtrade_gate1.py (CIF-vs-customs) and jodi_eia_reconcile.py.

WHY TWO SERIES, NOT ONE: our archive's "Weekly Carloads By 22 Commodity
Categories" measure includes the two intermodal categories (Containers,
Trailers) mixed in with the other 20 traditional-carload commodities.
FRED/BTS publish two SEPARATE national aggregates that explicitly EXCLUDE
each other — confirmed live this session from each series' own FRED page:
RAILFRTCARLOADS ("Carloads, Not Seasonally Adjusted" — intermodal
excluded, BTS/AAR's own page states carloads and intermodals "are tracked
independently") and RAILFRTINTERMODAL ("Containers and Trailers" only).
Summing our archive's 22 categories together and comparing the result
against either FRED series alone would silently compare two different
universes, understating one FRED series and overstating the mismatch
against the other. This script therefore splits our own sum the same way
FRED does — intermodal (Containers + Trailers) vs everything else — before
comparing either half to its matching FRED series.

PRE-REGISTERED PRIOR (stated before any pass/fail bound was chosen,
REASONING STANDARD #10): EP724 reports each Class I railroad's OWN
on-line carload activity. A shipment interchanged between two Class I
railroads (originated on one line, line-hauled onward by a second to
reach its destination) is plausibly reported by BOTH railroads' own
EP724 filings, while FRED/BTS-AAR's national total is understood to
count originations once. So our 7-railroad sum should run MODERATELY
ABOVE the FRED total, not below it — a ratio < 1.0 would mean our own
reporting subset misses volume FRED counts, which would be the real red
flag — and the excess should be a STABLE fraction over time (a real
structural interchange rate), not a noisy one (which would indicate a
data-integrity problem: a mis-summed commodity, a misparsed railroad
column, a unit inconsistency). Bar: mean ratio in [1.0, 2.5] (a generous
upper bound — the true interchange rate is not known precisely a priori,
only that it should be well short of a 3x+ multiple) with a
coefficient of variation (stdev/mean — scale-free, unlike
un_comtrade_gate1.py's raw stdev band, chosen because this ratio's
magnitude here is much larger than that script's near-1.0 CIF/customs
offset) under 0.20.

MONTHLY AGGREGATION CAVEAT (stated honestly, same spirit as the DTCC/
port-dwell unit-mismatch notes elsewhere in this repo): our archive is
WEEKLY ("week ending" columns); FRED's two series are MONTHLY, built —
per FRED's own stated methodology on each series' page, confirmed live
this session — "by dividing the weekly sum by 7 ... and then summing for
the number of days in the month." This script approximates that with a
coarser method: each of our weeks is assigned to the calendar month of
its week-ending date and summed directly (no day-weighting). That
approximation is a real, acknowledged source of some of the observed
ratio noise, not just the interchange effect — stated up front, not
discovered after the fact. Only calendar months with at least
MIN_WEEKS_PER_MONTH archived weeks are compared, so a partial first/last
month can't understate one side of the ratio.

Usage:
  python3 scripts/rail_traffic_gate1.py [--archive PATH] [--json]
Exit code: 0 = both series pass, 1 = at least one fails.
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(REPO_ROOT, "datacore", "rail", "ep724_carloads.json")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

CARLOAD_MEASURE = "Weekly Carloads By 22 Commodity Categories"
INTERMODAL_VARIABLES = {"Containers", "Trailers"}

MIN_WEEKS_PER_MONTH = 4
MEAN_RATIO_MIN = 1.0
MEAN_RATIO_MAX = 2.5
STABILITY_CV_MAX = 0.20
FRED_FETCH_RETRIES = 3
INTER_SERIES_DELAY_S = 10

# label -> FRED (not-seasonally-adjusted) series id
SERIES_UNDER_TEST = {
    "non_intermodal": "RAILFRTCARLOADS",
    "intermodal": "RAILFRTINTERMODAL",
}


def fetch_fred_series(series_id: str) -> dict:
    """FRED fredgraph.csv -> {"YYYY-MM": value}. Shells out to `curl
    --http1.1` rather than urllib.request/requests — un_comtrade_gate1.py
    already found (and documented) intermittent HTTP/2 stalls against
    fred.stlouisfed.org through this sandbox's egress proxy; --http1.1
    was that fix and is reused verbatim here rather than re-discovering
    the same failure mode.

    NO CUSTOM USER-AGENT HEADER — live-verified this session, deterministic
    across repeated alternating trials (3/3 each way): a `-H "User-Agent:
    voltradeai-datacore/1.0"` header (the exact string un_comtrade_gate1.py
    sends) makes this specific host/proxy combination hang to a full
    timeout with 0 bytes received (curl exit 28, http_code 000) EVERY
    time; the identical request with curl's own default User-Agent
    returns 200 EVERY time. Root cause not identified (Akamai/edge WAF
    behavior on this proxy is opaque from here) — the fix that matches
    the evidence is simply not sending the header, not a longer timeout
    or more retries (retrying a deterministic failure would just burn
    the retry budget for nothing)."""
    url = FRED_URL.format(series=series_id)
    last_err = None
    for attempt in range(1, FRED_FETCH_RETRIES + 1):
        try:
            proc = subprocess.run(
                ["curl", "-fsS", "--http1.1", "--max-time", "15", url],
                capture_output=True, text=True, timeout=20,
            )
            if proc.returncode == 0:
                return parse_fred_csv(proc.stdout)
            last_err = RuntimeError(f"curl exit {proc.returncode}: {proc.stderr.strip()}")
        except subprocess.TimeoutExpired as e:
            last_err = e
        if attempt < FRED_FETCH_RETRIES:
            time.sleep(3 * attempt)
    raise RuntimeError(f"FRED fetch for {series_id} failed after {FRED_FETCH_RETRIES} attempts: {last_err}")


def parse_fred_csv(text: str) -> dict:
    """Pure parser (no network) — 'observation_date,<SERIES>\\nYYYY-MM-DD,value\\n...'.
    Both RAILFRTCARLOADS/RAILFRTINTERMODAL report raw carload/unit COUNTS
    already (six-to-seven-digit monthly figures, live-verified against
    this archive's own weekly per-railroad totals this session) — unlike
    un_comtrade_gate1.py's dollar-value series, no *1_000_000 conversion
    applies here. '.' (FRED's own missing-value marker) is skipped, never
    coerced to 0."""
    out = {}
    r = csv.reader(io.StringIO(text))
    header = next(r, None)
    if not header or header[0] != "observation_date":
        raise ValueError(f"unexpected FRED CSV header: {header!r}")
    for row in r:
        if len(row) < 2:
            continue
        d, val = row[0], row[1]
        if val in (".", ""):
            continue
        out[d[:7]] = float(val)  # YYYY-MM-DD -> YYYY-MM
    return out


def monthly_totals(doc: dict, want_intermodal: bool) -> dict:
    """Sums the CARLOAD_MEASURE series across ALL railroads into
    {"YYYY-MM": total}, split by whether each commodity VARIABLE is one
    of the two intermodal categories. A null value for a given
    railroad/commodity/week is skipped, never zero-filled — matching
    stb_rail.py's own parse convention (a railroad reporting '*'/blank
    for a week is silent, not a real zero). Only months with
    >= MIN_WEEKS_PER_MONTH archived weeks are returned."""
    weeks = doc["weeks"]
    month_week_count = defaultdict(int)
    for wk in weeks:
        month_week_count[wk[:7]] += 1
    totals = defaultdict(float)
    for key, vals in doc["series"].items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _railroad, measure, variable = parts
        if measure != CARLOAD_MEASURE:
            continue
        is_inter = variable in INTERMODAL_VARIABLES
        if is_inter != want_intermodal:
            continue
        for wk, v in zip(weeks, vals):
            if v is None:
                continue
            totals[wk[:7]] += v
    full_months = {m for m, c in month_week_count.items() if c >= MIN_WEEKS_PER_MONTH}
    return {m: v for m, v in totals.items() if m in full_months}


def compute_ratios(ours: dict, fred: dict) -> list:
    """Returns [(month, ratio), ...] for every month present in BOTH
    sources with a non-zero FRED value — months only one side has
    (our archive ahead of FRED's own publication lag, or vice versa) are
    silently excluded, not treated as a mismatch."""
    out = []
    for month in sorted(set(ours) & set(fred)):
        if fred[month] == 0:
            continue
        out.append((month, ours[month] / fred[month]))
    return out


def evaluate(ratios: list) -> dict:
    if not ratios:
        return {"n": 0, "mean": None, "cv": None, "pass": False, "reason": "no overlapping months"}
    vals = [r for _, r in ratios]
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    stdev = variance ** 0.5
    cv = stdev / mean if mean else float("inf")
    mean_ok = MEAN_RATIO_MIN <= mean <= MEAN_RATIO_MAX
    stable_ok = cv < STABILITY_CV_MAX
    passed = mean_ok and stable_ok
    reason = "OK" if passed else (
        f"mean {mean:.4f} outside [{MEAN_RATIO_MIN},{MEAN_RATIO_MAX}]" if not mean_ok
        else f"cv {cv:.4f} >= {STABILITY_CV_MAX} (unstable)"
    )
    return {"n": n, "mean": mean, "cv": cv, "pass": passed, "reason": reason}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with open(args.archive) as f:
        archive = json.load(f)

    results = {}
    all_pass = True
    for i, (label, fred_id) in enumerate(SERIES_UNDER_TEST.items()):
        if i > 0:
            time.sleep(INTER_SERIES_DELAY_S)
        print(f"  fetching {label} ({fred_id}) ...", file=sys.stderr, flush=True)
        ours = monthly_totals(archive, want_intermodal=(label == "intermodal"))
        try:
            fred = fetch_fred_series(fred_id)
        except Exception as e:  # noqa: BLE001 - a live external fetch, surface any failure per-series
            results[label] = {"n": 0, "pass": False, "reason": f"FRED fetch failed: {e}"}
            all_pass = False
            continue
        ratios = compute_ratios(ours, fred)
        ev = evaluate(ratios)
        ev["fred_series"] = fred_id
        results[label] = ev
        all_pass = all_pass and ev["pass"]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("GATE 1 (DATA) — STB EP724 archived carloads (7 Class I railroads, system sum) vs FRED/BTS-AAR national totals")
        print(f"pre-registered bar: mean ratio in [{MEAN_RATIO_MIN},{MEAN_RATIO_MAX}], cv < {STABILITY_CV_MAX}\n")
        for label, ev in results.items():
            status = "PASS" if ev["pass"] else "FAIL"
            if ev.get("mean") is not None:
                print(f"  [{status}] {label:15s} n={ev['n']:3d} mean_ratio={ev['mean']:.4f} cv={ev['cv']:.4f}  ({ev['reason']})")
            else:
                print(f"  [{status}] {label:15s} {ev['reason']}")
        print(f"\nVERDICT: {'GATE 1 PASS (both series)' if all_pass else 'GATE 1 FAIL (see above)'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
