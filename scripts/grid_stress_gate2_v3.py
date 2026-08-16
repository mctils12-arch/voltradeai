#!/usr/bin/env python3
"""grid_stress_gate2_v3.py — GRID VISION stress index v3, GATE-2 (SIGNAL)
computation against REAL ERCOT ground truth (not a demand-derived proxy).

Design LOCKED, quoted verbatim from research/grid_vision_products.md's
"A1 gate-2 v3 design" (filed 2026-07-07, BEFORE any computation) and
research/grid_vision_events_ercot.md's "Gate-2 v3 scoring sets derived
from this list" (filed the same day, also before computation). This is
VARIANT 3 of the stress-index family — v1 and v2 both VOIDED on their
own outcome-variable spot-validation rules (datacore/gridvision/
gate2_result.json: v1 detected 0/10 spot events, v2 detected <5/10);
every multiple-hypothesis discount from those attempts compounds here
(REASONING STANDARD #4).

INDEX (v3), quoted verbatim — growth-aware, ZERO fitted parameters:
  1. DETRENDED DEMAND EXTREMITY: per-BA daily peak demand divided by
     that BA's trailing-365-day mean demand; percentile of this ratio
     within same-calendar-month across all years.
  2. FORECAST EXTREMITY: the day-ahead FORECAST (type DF) for day D+1,
     detrended by the same trailing-365d realized mean, same-month
     percentile.
  3. WEATHER EXTREMITY: CPC TX degree-day same-month percentile
     (unchanged from v0/v1/v2; already growth-free — weather doesn't
     trend at 5%/yr over this window).
  Composite = EQUAL WEIGHTS (1/3 each), pre-committed. No training
  split for weights — there is nothing to fit.
  Timing: index at day D uses ONLY data available by day-D morning —
  component 1 and 3 from day D-1 (complete by D morning); component 2
  from the day-ahead forecast FOR day D+1 (published during D, i.e.
  also available by D's close, mirroring the v1/v2 script's existing
  forecast-availability convention — not a new precision claim).
  CAVEAT stated before running, not after: same-calendar-month
  percentile normalization for components 1-3 pools ALL years
  (2019-2025) as instructed by the filed design ("across all years").
  This is descriptive/backtest normalization, not a live-only-history
  computation — a real-time implementation would need a trailing-only
  percentile lookback. Immaterial to today's verdict either way: v3
  remains, at best, a descriptive dashboard surface per Amendment 5c
  and the design's own FAIL/PASS clause (a PASS still carries the
  "live forward confirmation required" discount) — this is not a
  live-trading computation.

OUTCOME (v3) — the real ERCOT event list, not a demand-derived proxy.
Ground truth: research/grid_vision_events_ercot.md, compiled by a
research subagent from ERCOT/FERC/NERC/EIA primary sources, reviewed
and filed 2026-07-07. TIER-E = EEA-1/2/3 days. TIER-C = voluntary
conservation days.

PRE-STATED SCORING SETS — transcribed VERBATIM from
grid_vision_events_ercot.md's own "Gate-2 v3 scoring sets derived from
this list" section (that document did the derivation; this script only
consumes it, to avoid any transcription drift introduced by re-deriving
it here):
  TIER-E, out-of-sample (2023-2025): {2023-09-06} — ONE EEA-2 day.
    "with a single out-of-sample TIER-E event, the v3 recall floor
    (>=half of TIER-E days) degenerates to hit-or-miss on ONE day."
  TIER-C, out-of-sample (2023-2025): 13 days — "Jun 20 + Aug
    17/20/24/25/26/27/29/30 + Sep 6/7 2023, Jan 15/16 2024" (Sep 6
    appears in both tiers; it is the EEA-2 day that also carried a
    same-day conservation appeal per the source table).
  Design-contaminated TIER-E (2019-2022, REPORT ONLY, no evidential
  weight — this window shaped the v1/v2 diagnoses that produced v3):
  2019-08-13, 2019-08-15, 2021-02-15..2021-02-19.

PRE-STATED PASS CRITERIA — quoted verbatim from grid_vision_products.md:
  "PASS requires ALL of: (a) lift >= 2.0x over the seasonal base
  (raised from 1.5x — the variant-3 discount is paid in the bar, not
  in prose); (b) >= half of TIER-E days detected (recall floor — lift
  alone can be carried by TIER-C appeals); (c) no-single-summer-carry:
  the lift must hold with each validation year removed, per the clause
  that correctly killed v1 and v2 twice."
  Operationalized here: (a)/(c) lift is computed on the 13-day TIER-C
  set per grid_vision_events_ercot.md's own instruction ("leans on
  TIER-C days ... for the lift statistic"); (b) is checked against the
  single 2023-09-06 TIER-E day, miss = automatic FAIL (design's own
  words); (c) recomputes the TIER-C lift with each of {2023,2024,2025}
  excluded in turn and requires it hold >=1.5x wherever the reduced
  set still contains at least one event (a year contributing zero
  events to begin with, per grid_vision_events_ercot.md's TOTALS
  2025=0, cannot "carry" the lift and is reported, not scored, when
  removed).

PRIOR, stated before this run (REASONING STANDARD #10): weak-to-
moderate at best, matching A1's own stated prior ("edge is JOIN
specificity, not the event itself"). The out-of-sample event set is
small (13 TIER-C + 1 TIER-E day) and badly skewed across years — 2023
carries 11 of 13 TIER-C days, 2024 carries 2, 2025 carries ZERO event
days of either tier (grid_vision_events_ercot.md TOTALS). Expect the
no-single-summer-carry clause to be the binding constraint and this
computation to VOID or FAIL from small-N fragility, largely independent
of the point-estimate lift. Stated here before running, not after.

STOPPING RULE: this is the ONE run of v3, per its own pre-registration
("computation deferred to a later session... single run per the
one-shot rule"). No further outcome variant follows this one in the
same session. If v3 also fails, grid_vision_products.md's v3 FAIL
clause applies: the stress-index predictive line CLOSES as a research
line absent new data (LMP archive or per-line ratings), not re-varied
a fourth time on the same ingredients.

Usage:
  python3 scripts/grid_stress_gate2_v3.py <dir-with-EIA930_BALANCE_csvs> \
      [--cpc datacore/cpc/degree_days.json] \
      [--out datacore/gridvision/gate2_result_v3.json]
"""
import argparse
import json
import os
import sys
from collections import defaultdict, deque
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_stress_gate2 import load_erco_hours, load_tx_dd, pct_rank, month_of, year_of  # noqa: E402

VALID_YEARS = (2023, 2024, 2025)
TIER_E_VALID = {"2023-09-06"}
TIER_C_VALID = {
    "2023-06-20", "2023-08-17", "2023-08-20", "2023-08-24", "2023-08-25",
    "2023-08-26", "2023-08-27", "2023-08-29", "2023-08-30", "2023-09-06",
    "2023-09-07", "2024-01-15", "2024-01-16",
}
assert len(TIER_E_VALID) == 1 and len(TIER_C_VALID) == 13, (
    "transcription of grid_vision_events_ercot.md's scoring sets drifted "
    "from the source document's own stated counts")

TIER_E_CONTAMINATED = {
    "2019-08-13", "2019-08-15",
    "2021-02-15", "2021-02-16", "2021-02-17", "2021-02-18", "2021-02-19",
}

TRAILING_DAYS = 365
LIFT_BAR = 2.0
STABILITY_LIFT_BAR = 1.5


def trailing_mean_series(daily_peak):
    """iso date -> trailing-365-day (inclusive) mean of daily_peak ending
    at that date. None until TRAILING_DAYS present days have accumulated."""
    out = {}
    window = deque()
    total = 0.0
    for iso in sorted(daily_peak):
        d = date.fromisoformat(iso)
        window.append((d, daily_peak[iso]))
        total += daily_peak[iso]
        while window and (d - window[0][0]).days >= TRAILING_DAYS:
            _, v = window.popleft()
            total -= v
        out[iso] = (total / len(window)) if len(window) >= TRAILING_DAYS else None
    return out


def same_month_pool(values_by_iso):
    """month(1-12) -> sorted list of values, pooled across ALL years
    present in values_by_iso (the filed design's own instruction:
    "same-calendar-month ... across all years")."""
    by_month = defaultdict(list)
    for iso, v in values_by_iso.items():
        if v is not None:
            by_month[month_of(iso)].append(v)
    for m in by_month:
        by_month[m].sort()
    return by_month


def shift(iso, days):
    return (date.fromisoformat(iso) + timedelta(days=days)).isoformat()


def compute(dir930, cpc_path):
    hours, fallbacks = load_erco_hours(dir930)
    dd = load_tx_dd(cpc_path)

    daily_peak, daily_forecast_mean = {}, {}
    for iso, rows in hours.items():
        dvs = [dv for dv, _fc in rows if dv is not None]
        fcs = [fc for _dv, fc in rows if fc is not None]
        if dvs:
            daily_peak[iso] = max(dvs)
        if fcs:
            daily_forecast_mean[iso] = sum(fcs) / len(fcs)

    trailing_mean = trailing_mean_series(daily_peak)

    ratio_a, ratio_b = {}, {}
    for iso, tm in trailing_mean.items():
        if tm is None or tm <= 0:
            continue
        if iso in daily_peak:
            ratio_a[iso] = daily_peak[iso] / tm
        if iso in daily_forecast_mean:
            ratio_b[iso] = daily_forecast_mean[iso] / tm

    pool_a = same_month_pool(ratio_a)
    pool_b = same_month_pool(ratio_b)
    pool_c = same_month_pool(dd)

    def pctA(iso):
        return pct_rank(pool_a.get(month_of(iso), []), ratio_a[iso]) if iso in ratio_a else None

    def pctB(iso):
        return pct_rank(pool_b.get(month_of(iso), []), ratio_b[iso]) if iso in ratio_b else None

    def pctC(iso):
        return pct_rank(pool_c.get(month_of(iso), []), dd[iso]) if iso in dd else None

    def index_for(d_iso):
        """index(D), using D-1 (components 1,3) and D+1 (component 2)."""
        prev, nxt = shift(d_iso, -1), shift(d_iso, 1)
        a = pctA(prev)
        b = pctB(nxt)
        c = pctC(prev)
        if None in (a, b, c):
            return None
        return (a + b + c) / 3.0

    all_days = sorted(daily_peak)
    if not all_days:
        raise SystemExit("no ERCO days loaded — check the input directory")
    lo, hi = date.fromisoformat(all_days[0]), date.fromisoformat(all_days[-1])

    index_by_day = {}
    d = lo
    while d <= hi:
        iso = d.isoformat()
        v = index_for(iso)
        if v is not None:
            index_by_day[iso] = v
        d += timedelta(days=1)

    # ── per (year, month) top-decile threshold, validation window only ──
    by_ym = defaultdict(list)
    for iso, v in index_by_day.items():
        y = year_of(iso)
        if y in VALID_YEARS:
            by_ym[(y, month_of(iso))].append((iso, v))
    thresh = {}
    for ym, pairs in by_ym.items():
        vals = sorted(v for _i, v in pairs)
        thresh[ym] = vals[int(0.9 * len(vals))] if vals else None

    flagged = {}  # iso(D) -> bool
    for ym, pairs in by_ym.items():
        th = thresh[ym]
        for iso, v in pairs:
            flagged[iso] = th is not None and v >= th

    def month_base_rate(event_set, years):
        """month -> fraction of days D (year in `years`) whose D+1 is
        in event_set, i.e. the same-calendar-month random-day control."""
        hit, tot = defaultdict(int), defaultdict(int)
        for iso in index_by_day:
            if year_of(iso) not in years:
                continue
            m = month_of(iso)
            tot[m] += 1
            hit[m] += 1 if shift(iso, 1) in event_set else 0
        return {m: hit[m] / tot[m] for m in tot if tot[m]}

    def lift_and_hits(event_set, years):
        base = month_base_rate(event_set, years)
        flagged_days = [iso for iso in flagged
                        if year_of(iso) in years and flagged[iso]]
        if not flagged_days:
            return None, 0, None
        hit_flags = [shift(iso, 1) in event_set for iso in flagged_days]
        hit_rate = sum(hit_flags) / len(flagged_days)
        exp = sum(base.get(month_of(iso), 0.0) for iso in flagged_days) / len(flagged_days)
        lift = (hit_rate / exp) if exp > 0 else None
        return lift, len(flagged_days), hit_rate

    overall_lift, overall_n, overall_hit = lift_and_hits(TIER_C_VALID, VALID_YEARS)

    # recall: was the single TIER-E day detected? (D = E-1 flagged?)
    tier_e_detected = [e for e in TIER_E_VALID if flagged.get(shift(e, -1), False)]
    recall = len(tier_e_detected) / len(TIER_E_VALID)

    # no-single-summer-carry: leave-one-year-out
    leave_one_out = {}
    for y in VALID_YEARS:
        remaining_years = tuple(yy for yy in VALID_YEARS if yy != y)
        remaining_events = {e for e in TIER_C_VALID if year_of(e) in remaining_years}
        removed_events = {e for e in TIER_C_VALID if year_of(e) == y}
        if not removed_events:
            leave_one_out[str(y)] = {
                "note": f"{y} contributed 0 TIER-C events — removing it cannot "
                        "change the event count; reported, not scored",
                "lift": overall_lift, "flagged": overall_n,
            }
            continue
        lift, n, hit = lift_and_hits(remaining_events, VALID_YEARS)
        leave_one_out[str(y)] = {
            "removed_event_days": len(removed_events),
            "remaining_event_days": len(remaining_events),
            "lift": round(lift, 3) if lift else None,
            "flagged": n,
            "hit_rate": round(hit, 3) if hit is not None else None,
            "holds": lift is not None and lift >= STABILITY_LIFT_BAR,
        }

    stability_scoreable = [v for v in leave_one_out.values() if "holds" in v]
    stability_ok = bool(stability_scoreable) and all(v["holds"] for v in stability_scoreable)

    pass_a = overall_lift is not None and overall_lift >= LIFT_BAR
    pass_b = recall >= 0.5
    pass_c = stability_ok
    verdict = "PASS" if (pass_a and pass_b and pass_c) else "FAIL"
    if not pass_b:
        verdict = "FAIL (recall floor missed — the sole out-of-sample TIER-E day was not flagged; automatic FAIL per the filed design)"

    # design-contaminated 2019-2022 TIER-E recall, report only
    contaminated_detected = [e for e in TIER_E_CONTAMINATED
                              if flagged.get(shift(e, -1), False)]

    return {
        "criteria": "quoted verbatim in scripts/grid_stress_gate2_v3.py's "
                    "header from research/grid_vision_products.md's v3 "
                    "design and grid_vision_events_ercot.md's scoring sets, "
                    "both filed 2026-07-07 before this computation ran",
        "multiple_hypothesis_note": "third outcome variant in the stress-index "
                                     "family (v1, v2 both VOID — see "
                                     "gate2_result.json); the discount from "
                                     "both prior attempts compounds onto any "
                                     "v3 PASS, which additionally requires "
                                     "live forward confirmation before trust",
        "data": {
            "erco_days": len(daily_peak),
            "adjusted_demand_fallback_hours": fallbacks,
            "index_computable_days": len(index_by_day),
            "trailing_window_days": TRAILING_DAYS,
        },
        "tier_e_out_of_sample": {
            "event_days": sorted(TIER_E_VALID),
            "detected": sorted(tier_e_detected),
            "recall": round(recall, 3),
            "recall_floor": 0.5,
            "passed_recall_floor": pass_b,
        },
        "tier_c_out_of_sample_lift": {
            "event_days": sorted(TIER_C_VALID),
            "overall_lift": round(overall_lift, 3) if overall_lift else None,
            "flagged_days": overall_n,
            "hit_rate": round(overall_hit, 3) if overall_hit is not None else None,
            "lift_bar": LIFT_BAR,
            "passed_lift_bar": pass_a,
        },
        "no_single_summer_carry": {
            "per_year_leave_one_out": leave_one_out,
            "stability_lift_bar": STABILITY_LIFT_BAR,
            "scoreable_years": len(stability_scoreable),
            "holds": stability_ok,
        },
        "design_contaminated_2019_2022_report_only": {
            "tier_e_event_days": sorted(TIER_E_CONTAMINATED),
            "detected": sorted(contaminated_detected),
            "note": "report-only per the filed design; this window shaped "
                    "the v1/v2 diagnoses that produced v3 and carries no "
                    "evidential weight toward the verdict",
        },
        "verdict": verdict,
        "prior_stated_before_run": "weak-to-moderate; expected the "
            "no-single-summer-carry clause to be the binding constraint "
            "given 2025 contributed zero TIER-C/TIER-E event days and 2023 "
            "carries 11 of 13 TIER-C days (grid_vision_events_ercot.md "
            "TOTALS) — recorded in the script header before this run",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir930")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--cpc", default=os.path.join(root, "datacore", "cpc", "degree_days.json"))
    ap.add_argument("--out", default=os.path.join(root, "datacore", "gridvision", "gate2_result_v3.json"))
    args = ap.parse_args()
    res = compute(args.dir930, args.cpc)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["verdict"] == "PASS" else 3)
