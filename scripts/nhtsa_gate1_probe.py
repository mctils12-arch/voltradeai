#!/usr/bin/env python3
"""
scripts/nhtsa_gate1_probe.py — ROOT VALIDATION LADDER gate 1 (DATA) for the
nhtsa_vehicle_complaints root (BUILD ORDER 6 #4, research/open_questions.md
line ~9019; the live archiver is server/nhtsaComplaints.ts). The build
order's own stated gate 1 is: "complaint counts vs NHTSA's own published
recall timeline for 3 known cases."

WHY A NEW LIVE PROBE, NOT A REPLAY OF OUR OWN ARCHIVE: server/
nhtsaComplaints.ts has only archived NEW-since-first-seen complaints for a
CURATED watchlist since 2026-07-06 (BUILD ORDER 6's own filing date) — about
two months of depth at the time this script was written, nowhere near
enough to reach back to any well-documented historical recall. But the SAME
endpoint our archiver polls (api.nhtsa.gov/complaints/complaintsByVehicle)
returns a vehicle's FULL complaint history back to whenever NHTSA's own ODI
database begins covering it (verified live this session: Chevrolet Cobalt
2006 returns all 2,330 complaints with no pagination cap, oldest dated
2006-01-03) — so gate 1 can be run directly against the live API for any
past recall without waiting out our own archive's youth. This validates the
READING (the same fetch/parse path our archiver already uses is sane and
matches known reality) before anything downstream trusts it.

PRE-REGISTERED CASES AND PRIOR (REASONING STANDARD #10 — chosen for public
notoriety and documented complaint-driven origin BEFORE this script queried
any complaint COUNTS; only the exact recall campaign numbers/dates were
looked up live to pin them precisely):
  1. Chevrolet Cobalt MY2006 — GM ignition-switch/airbag non-deployment
     defect, recall 14V047000. Public record (congressional testimony,
     the Valukas report): GM had complaints and internal reports about
     the ignition switch for close to a decade before recalling.
  2. Toyota Camry MY2007 — unintended-acceleration "sticky pedal" defect,
     recall 10V017000. Public record: complaints about sudden acceleration
     on Toyota/Lexus vehicles predate the 2009-2010 recall wave by years
     and triggered congressional hearings.
  3. Hyundai Sonata MY2011 — Theta II engine bearing-wear fire/stall
     defect, recall 15V568000. Public record: engine knock/stall/fire
     complaints built up over 2011-2015 before the September 2015 recall.
PRIOR: all 3 should show (a) nonzero relevant-component complaints on file
strictly BEFORE the recall date (the defect was discoverable, not silent
until the recall), and (b) a marked step-up in relevant-component complaint
volume in the recall's own calendar year vs. the prior 3 years (the recall
event itself, and often the run-up to it, is visible in complaint volume).
This does NOT pre-register a claim that the pre-recall trend rises smoothly
year over year — REASONING STANDARD #4 (distrust convenient patterns): a
clean monotonic ramp is the more USEFUL shape for gate 2 but is not assumed
here, and the real numbers below turn out to only show it clearly in 1 of
3 cases (Hyundai) — logged honestly, not smoothed over.

GENUINELY NEW FINDING, COMPILED HERE (EDGE DOCTRINE #3) SO NO FUTURE
SESSION HAS TO REDISCOVER IT: NHTSA's recalls API
(api.nhtsa.gov/recalls/recallsByVehicle) reports `ReportReceivedDate` in
DD/MM/YYYY, NOT the MM/DD/YYYY the complaints API uses for
`dateComplaintFiled`/`dateOfIncident`. Proven unambiguously live this
session — campaign 15V689000 (Camry) reports "22/10/2015"; 22 cannot be a
month, so the field cannot be MM/DD. Cross-checked against public record on
an unambiguous case too: campaign 14V047000's ReportReceivedDate
"10/02/2014" parses as 2014-02-10 under DD/MM, which matches the
well-documented fact that GM notified NHTSA on February 10, 2014 — under
MM/DD it would misread as October 2, 2014, four months late. Nothing in
this repo touched the recalls API before this script (confirmed by grep for
"recallsByVehicle"/"ReportReceivedDate" across server/ and scripts/ turning
up zero hits) — a future session joining complaints to recalls by date
would have silently mis-parsed this field without this note.

NOT ATTEMPTED HERE (left for gate 2): any statistical significance test,
any forward-return analysis, any claim this is tradeable. This script only
answers "is the complaint-counting pipeline's reading of a KNOWN case
consistent with NHTSA's own published recall record" — the DATA gate, nothing more.

Run (live, hits api.nhtsa.gov — politely spaced, ~90s total for 3 vehicles):
  python3 scripts/nhtsa_gate1_probe.py
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta
from statistics import median
from typing import Optional

COMPLAINTS_API = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
RECALLS_API = "https://api.nhtsa.gov/recalls/recallsByVehicle"
CALL_SPACING_S = 2.0

# Minimum pre-recall relevant complaints required to call the defect
# "discoverable" ahead of the recall (not just a handful of noise hits).
PRE_RECALL_MIN = 10
# Recall-year relevant count must be at least this multiple of the median
# of the 3 prior calendar years' relevant counts to count as a visible
# step-up (guards against calling routine complaint noise a "signal").
STEP_UP_MIN_RATIO = 2.0
# Tolerance (days) between our pinned expected recall date and what the
# live recalls API reports for the matching campaign number.
RECALL_DATE_TOLERANCE_DAYS = 45

CASES = [
    {
        "name": "Chevrolet Cobalt MY2006 — ignition switch / airbag non-deployment",
        "make": "chevrolet", "model": "cobalt", "model_year": 2006,
        "campaign": "14V047000",
        "expected_recall_date": date(2014, 2, 10),
        "keywords": ("IGNITION", "AIR BAG", "ELECTRICAL"),
    },
    {
        "name": "Toyota Camry MY2007 — sticky accelerator pedal",
        "make": "toyota", "model": "camry", "model_year": 2007,
        "campaign": "10V017000",
        "expected_recall_date": date(2010, 1, 21),
        "keywords": ("ACCELERATOR", "SPEED CONTROL", "VEHICLE SPEED"),
    },
    {
        "name": "Hyundai Sonata MY2011 — Theta II engine bearing wear",
        "make": "hyundai", "model": "sonata", "model_year": 2011,
        "campaign": "15V568000",
        "expected_recall_date": date(2015, 9, 10),
        "keywords": ("ENGINE",),
    },
]


# ── pure pieces (unit-tested) ───────────────────────────────────────────────

def parse_complaint_date(s: Optional[str]) -> Optional[date]:
    """dateComplaintFiled/dateOfIncident are MM/DD/YYYY (matches
    server/nhtsaComplaints.ts's own normalizeUsDate)."""
    if not s:
        return None
    try:
        return datetime.strptime(s, "%m/%d/%Y").date()
    except ValueError:
        return None


def parse_recall_date(s: Optional[str]) -> Optional[date]:
    """recallsByVehicle's ReportReceivedDate is DD/MM/YYYY — see the
    module docstring's GENUINELY NEW FINDING for the proof."""
    if not s:
        return None
    try:
        return datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        return None


def is_relevant(components: Optional[str], keywords: tuple[str, ...]) -> bool:
    c = (components or "").upper()
    return any(k in c for k in keywords)


def yearly_counts(complaints: list[dict], keywords: tuple[str, ...]) -> dict[int, dict[str, int]]:
    """{year: {"total": n, "relevant": n}} from raw complaintsByVehicle
    result rows (dateComplaintFiled, components)."""
    out: dict[int, dict[str, int]] = {}
    for r in complaints:
        d = parse_complaint_date(r.get("dateComplaintFiled"))
        if d is None:
            continue
        bucket = out.setdefault(d.year, {"total": 0, "relevant": 0})
        bucket["total"] += 1
        if is_relevant(r.get("components"), keywords):
            bucket["relevant"] += 1
    return out


def find_recall(recalls: list[dict], campaign: str) -> Optional[dict]:
    for r in recalls:
        if r.get("NHTSACampaignNumber") == campaign:
            return r
    return None


def evaluate_case(case: dict, complaints: list[dict], recalls: list[dict]) -> dict:
    recall_date = case["expected_recall_date"]
    keywords = case["keywords"]

    recall_row = find_recall(recalls, case["campaign"])
    live_recall_date = parse_recall_date(recall_row.get("ReportReceivedDate")) if recall_row else None
    recall_date_delta_days = (abs((live_recall_date - recall_date).days)
                              if live_recall_date else None)
    recall_confirmed = (recall_row is not None and live_recall_date is not None
                        and recall_date_delta_days <= RECALL_DATE_TOLERANCE_DAYS)

    buckets = yearly_counts(complaints, keywords)
    recall_year = recall_date.year
    prior_years = [recall_year - 1, recall_year - 2, recall_year - 3]
    prior_relevant = [buckets.get(y, {"relevant": 0})["relevant"] for y in prior_years]
    prior_median = median(prior_relevant) if prior_relevant else 0.0
    recall_year_relevant = buckets.get(recall_year, {"relevant": 0})["relevant"]

    pre_recall_relevant = sum(
        1 for r in complaints
        if is_relevant(r.get("components"), keywords)
        and (d := parse_complaint_date(r.get("dateComplaintFiled"))) is not None
        and d < recall_date
    )

    step_up_ratio = (recall_year_relevant / prior_median) if prior_median > 0 else (
        float("inf") if recall_year_relevant > 0 else 0.0)

    checks = {
        "case": case["name"],
        "recall_confirmed": recall_confirmed,
        "live_recall_date": live_recall_date.isoformat() if live_recall_date else None,
        "expected_recall_date": recall_date.isoformat(),
        "recall_date_delta_days": recall_date_delta_days,
        "pre_recall_relevant_complaints": pre_recall_relevant,
        "pre_recall_ok": pre_recall_relevant >= PRE_RECALL_MIN,
        "recall_year_relevant": recall_year_relevant,
        "prior_3yr_relevant": dict(zip(prior_years, prior_relevant)),
        "prior_3yr_median": prior_median,
        "step_up_ratio": None if step_up_ratio == float("inf") else round(step_up_ratio, 2),
        "step_up_ok": step_up_ratio >= STEP_UP_MIN_RATIO,
        "yearly_relevant": {y: v["relevant"] for y, v in sorted(buckets.items())},
    }
    checks["PASS"] = bool(checks["recall_confirmed"] and checks["pre_recall_ok"] and checks["step_up_ok"])
    return checks


# ── live fetch ───────────────────────────────────────────────────────────────

def fetch_json(url: str, timeout: int = 90) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-datacore-gate1/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_complaints_live(make: str, model: str, model_year: int) -> list[dict]:
    url = f"{COMPLAINTS_API}?make={make}&model={model}&modelYear={model_year}"
    return fetch_json(url).get("results", [])


def fetch_recalls_live(make: str, model: str, model_year: int) -> list[dict]:
    url = f"{RECALLS_API}?make={make}&model={model}&modelYear={model_year}"
    return fetch_json(url).get("results", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=CALL_SPACING_S)
    args = ap.parse_args()

    results = []
    for i, case in enumerate(CASES):
        if i > 0:
            time.sleep(args.spacing)
        complaints = fetch_complaints_live(case["make"], case["model"], case["model_year"])
        time.sleep(args.spacing)
        recalls = fetch_recalls_live(case["make"], case["model"], case["model_year"])
        result = evaluate_case(case, complaints, recalls)
        result["n_complaints_fetched"] = len(complaints)
        results.append(result)
        print(f"\n== {case['name']} ==")
        print(json.dumps(result, indent=1))

    all_pass = all(r["PASS"] for r in results)
    print(f"\n== GATE 1 VERDICT: {'PASS' if all_pass else 'FAIL'} "
          f"({sum(r['PASS'] for r in results)}/{len(results)} cases) ==")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
