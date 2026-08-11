#!/usr/bin/env python3
"""cdc_cancer_rates.py — NCI State Cancer Profiles county-level cancer
incidence + mortality rates. Location Context Engine hazard layer #5
(research/location_context_engine.md) — the last hazard layer on that
file's list, logged as still-queued across roughly six prior session
entries (2026-07-13 through 2026-07-25) before this build.

Source: statecancerprofiles.cancer.gov (National Cancer Institute, public
domain), the same underlying SEER+NPCR surveillance data CDC's own U.S.
Cancer Statistics (USCS) product publishes, exposed here via the site's
own CSV export (output=1 query param) — no key, no HTML scraping, a
documented and stable parameter contract (live-probed this session).
County of residence, All Cancer Sites combined (all stages), all
races/sexes/ages, most recent published 5-year window (currently
2018-2022 age-adjusted rates per 100,000 US-2000-standard population).

GATE 1 (ROOT VALIDATION LADDER — "the reading is verified against an
external truth source before anything downstream"): each pull's national
aggregate row (FIPS 00000) is checked against CDC_NATIONAL_REFERENCE,
CDC's own separately-published USCS national headline figures. This is
the SAME underlying federal surveillance system reporting on itself
through two different publication channels, not an independent second
source — but it is a real publish-vs-publish consistency check that
catches a wrong query parameter, a truncated pull, or an export-format
regression on NCI's site before any of it reaches the app. A gate-1
failure aborts the build (old artifact stays in place) rather than
shipping unverified data.

ECOLOGICAL FALLACY GUARD (non-negotiable, per location_context_engine.md
DATA QUALITY GATE + honesty rails): this is a COUNTY-LEVEL AGGREGATE
statistic describing the county's population, not any specific address
or resident. Every consumer of this artifact must display it at the
county polygon, never implied down to a point. `meta.caveat` below states
this; nothing downstream may strip it.

SUPPRESSION: NCI suppresses rate/trend cells for any area-sex-race
category with fewer than 16 reported records ("*" in the CSV) to protect
confidentiality and estimate stability. Suppressed cells become `null`
with an explicit `*_suppressed: true` flag — never a false zero.

Usage:
  python3 scripts/cdc_cancer_rates.py [--out datacore/cdc_cancer/county_rates.json]
  python3 scripts/cdc_cancer_rates.py --incidence-csv f1.csv --mortality-csv f2.csv --out ...
      (replay already-downloaded exports — used by the test suite and for
      offline reruns; same parse/validate/gate-1 path either way)
"""
import argparse
import csv
import json
import re
import sys
import urllib.request

INCIDENCE_URL = (
    "https://statecancerprofiles.cancer.gov/incidencerates/index.php?"
    "stateFIPS=00&areatype=county&cancer=001&race=00&sex=0&age=001&"
    "stage=999&year=0&type=incd&sortVariableName=rate&sortOrder=default&output=1"
)
MORTALITY_URL = (
    "https://statecancerprofiles.cancer.gov/deathrates/index.php?"
    "stateFIPS=00&areatype=county&cancer=001&race=00&sex=0&age=001&"
    "year=0&type=death&sortVariableName=rate&sortOrder=default&output=1"
)

# CDC's own published USCS national headline figures (all cancer sites
# combined) — the GATE 1 cross-check reference, recorded with the date
# this session looked them up. Only move these numbers when a future
# session re-confirms CDC's own headline page has genuinely advanced to a
# newer data year — never to make a failing gate pass.
CDC_NATIONAL_REFERENCE = {
    "incidence_rate": 446.9,
    "mortality_rate": 146.0,
    "as_of": "CDC USCS national headline figures, confirmed 2026-08-11",
    "tolerance_pct": 10.0,
}

NATIONAL_FIPS = "00000"
ATTRIBUTION = (
    "National Cancer Institute, State Cancer Profiles (statecancerprofiles.cancer.gov) — "
    "SEER+NPCR county of residence data, public domain"
)
CAVEAT = (
    "COUNTY-LEVEL AGGREGATE STATISTIC ONLY. This rate describes the whole county's reported "
    "population over the stated 5-year window — it is NOT a claim about any specific address, "
    "resident, or property, and must never be displayed or interpreted below the county level "
    "(ecological fallacy). Age-adjusted to the 2000 US standard population. All Cancer Sites "
    "combined, all races/sexes/ages, all stages. Suppressed cells (fewer than 16 reported "
    "records) are null, not zero."
)


def _num(s):
    s = (s or "").strip()
    if s in ("", "*", "~", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_csv(text):
    """NCI export text -> list of raw CSV data rows (each a list of cell
    strings). Pure — no I/O. The header row is located by content, not a
    fixed line number, since the title block above it is not guaranteed
    to stay the same length across NCI site revisions."""
    lines = text.splitlines()
    hdr_idx = next((i for i, l in enumerate(lines) if l.startswith("County,FIPS")), None)
    if hdr_idx is None:
        raise ValueError("County,FIPS header row not found — NCI export format may have changed")
    data_lines = []
    for l in lines[hdr_idx + 1:]:
        if not l.strip() or l[0] != '"':
            break
        data_lines.append(l)
    rows = [row for row in csv.reader(data_lines) if len(row) >= 10]
    return rows


def parse_name(raw):
    """'Union County, Florida(2)' -> ('Union County', 'Florida'). Footnote
    markers like '(2)'/'(7)' are NCI source-citation superscripts, not
    part of the name. The national row ('US (SEER+NPCR)(1)' / 'United
    States') has no state part."""
    cleaned = re.sub(r"\(\d+\)\s*$", "", raw).strip()
    if "," in cleaned:
        county, state = cleaned.rsplit(",", 1)
        return county.strip(), state.strip()
    return cleaned, None


def row_to_record(row, kind):
    """One incidence-or-mortality CSV row -> a normalized dict. `kind` is
    'incidence' or 'mortality' — the mortality export has one extra
    column ('Met Healthy People Objective') before the rate, shifting
    every subsequent index by one."""
    name = row[0].strip()
    fips = row[1].strip()
    if kind == "incidence":
        rate, ci_lo, ci_hi = _num(row[3]), _num(row[4]), _num(row[5])
        avg_count_raw, trend = row[9].strip(), row[10].strip()
    else:
        rate, ci_lo, ci_hi = _num(row[4]), _num(row[5]), _num(row[6])
        avg_count_raw, trend = row[10].strip(), row[11].strip()
    avg_count = _num(avg_count_raw)
    return {
        "fips": fips,
        "name": name,
        "rate": rate,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "avg_annual_count": avg_count,
        "avg_annual_count_note": None if avg_count is not None else (avg_count_raw or None),
        "trend": trend if trend not in ("*", "") else None,
        "suppressed": rate is None,
    }


def gate1_check(national_inc, national_mort, ref=CDC_NATIONAL_REFERENCE):
    """Publish-vs-publish cross-check: the pulled national row vs CDC's
    own separately-published headline figures. Pure function of the two
    national records — testable without network access."""
    def pct_diff(a, b):
        if a is None or b in (None, 0):
            return None
        return abs(a - b) / b * 100.0

    inc_rate = national_inc["rate"] if national_inc else None
    mort_rate = national_mort["rate"] if national_mort else None
    inc_diff = pct_diff(inc_rate, ref["incidence_rate"])
    mort_diff = pct_diff(mort_rate, ref["mortality_rate"])
    passed = (
        inc_diff is not None and inc_diff <= ref["tolerance_pct"]
        and mort_diff is not None and mort_diff <= ref["tolerance_pct"]
    )
    return {
        "pulled_national_incidence_rate": inc_rate,
        "pulled_national_mortality_rate": mort_rate,
        "reference": ref,
        "incidence_pct_diff": inc_diff,
        "mortality_pct_diff": mort_diff,
        "passed": passed,
    }


def validate_fips(fips):
    return bool(re.fullmatch(r"\d{5}", fips or ""))


def build_artifact(incidence_rows, mortality_rows, built_at):
    """Rows from both CSVs -> the full artifact dict (county records +
    meta, including the GATE 1 result). Pure — no I/O, fully testable."""
    inc_by_fips = {r["fips"]: r for r in (row_to_record(row, "incidence") for row in incidence_rows)}
    mort_by_fips = {r["fips"]: r for r in (row_to_record(row, "mortality") for row in mortality_rows)}

    national_inc = inc_by_fips.pop(NATIONAL_FIPS, None)
    national_mort = mort_by_fips.pop(NATIONAL_FIPS, None)
    gate1 = gate1_check(national_inc, national_mort)

    counties = []
    quarantined = []
    for fips in sorted(set(inc_by_fips) | set(mort_by_fips)):
        inc = inc_by_fips.get(fips)
        mort = mort_by_fips.get(fips)
        name_source = inc or mort
        if not validate_fips(fips):
            quarantined.append({"fips": fips, "issue": "bad_fips_format"})
            continue
        # Sanity ceiling only (never a plausibility filter — small counties
        # legitimately post high-variance rates well above the national
        # figure; this only catches an impossible/corrupt value).
        bad_rate = any(
            r is not None and r["rate"] is not None and not (0 <= r["rate"] <= 5000)
            for r in (inc, mort) if r
        )
        if bad_rate:
            quarantined.append({"fips": fips, "issue": "rate_out_of_bounds"})
            continue
        county, state = parse_name(name_source["name"])
        counties.append({
            "fips": fips,
            "county": county,
            "state": state,
            "incidence_rate": inc["rate"] if inc else None,
            "incidence_ci": [inc["ci_lo"], inc["ci_hi"]] if inc and inc["rate"] is not None else None,
            "incidence_avg_annual_count": inc["avg_annual_count"] if inc else None,
            "incidence_trend": inc["trend"] if inc else None,
            "incidence_suppressed": inc["suppressed"] if inc else True,
            "mortality_rate": mort["rate"] if mort else None,
            "mortality_ci": [mort["ci_lo"], mort["ci_hi"]] if mort and mort["rate"] is not None else None,
            "mortality_avg_annual_count": mort["avg_annual_count"] if mort else None,
            "mortality_trend": mort["trend"] if mort else None,
            "mortality_suppressed": mort["suppressed"] if mort else True,
        })

    return {
        "source": ATTRIBUTION,
        "attribution": ATTRIBUTION,
        "predictive": False,
        "caveat": CAVEAT,
        "built_at": built_at,
        "county_count": len(counties),
        "quarantined_count": len(quarantined),
        "quarantined": quarantined,
        "gate1": gate1,
        "counties": counties,
    }


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datacore/cdc_cancer/county_rates.json")
    ap.add_argument("--incidence-csv", help="replay a saved incidence export instead of fetching live")
    ap.add_argument("--mortality-csv", help="replay a saved mortality export instead of fetching live")
    ap.add_argument("--built-at", help="override built_at (ISO timestamp) — used by tests for determinism")
    args = ap.parse_args()

    inc_text = open(args.incidence_csv, encoding="utf-8", errors="replace").read() if args.incidence_csv else fetch(INCIDENCE_URL)
    mort_text = open(args.mortality_csv, encoding="utf-8", errors="replace").read() if args.mortality_csv else fetch(MORTALITY_URL)

    incidence_rows = parse_csv(inc_text)
    mortality_rows = parse_csv(mort_text)
    print(f"[cdc_cancer_rates] parsed {len(incidence_rows)} incidence rows, {len(mortality_rows)} mortality rows", file=sys.stderr)

    built_at = args.built_at or __import__("datetime").datetime.utcnow().isoformat() + "Z"
    artifact = build_artifact(incidence_rows, mortality_rows, built_at)

    g1 = artifact["gate1"]
    print(f"[cdc_cancer_rates] GATE 1: national incidence {g1['pulled_national_incidence_rate']} "
          f"(ref {g1['reference']['incidence_rate']}, diff {g1['incidence_pct_diff']:.2f}% ) / "
          f"national mortality {g1['pulled_national_mortality_rate']} "
          f"(ref {g1['reference']['mortality_rate']}, diff {g1['mortality_pct_diff']:.2f}% ) "
          f"-> {'PASS' if g1['passed'] else 'FAIL'}", file=sys.stderr)
    print(f"[cdc_cancer_rates] {artifact['county_count']} counties, {artifact['quarantined_count']} quarantined", file=sys.stderr)

    if not g1["passed"]:
        print("[cdc_cancer_rates] GATE 1 FAILED — aborting build, NOT writing output "
              "(existing artifact, if any, is left in place)", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=1)
    print(f"[cdc_cancer_rates] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
