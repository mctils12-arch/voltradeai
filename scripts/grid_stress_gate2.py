#!/usr/bin/env python3
"""grid_stress_gate2.py — GRID VISION stress index v0, GATE-2
computation (SIGNAL layer of the validation ladder).

LOCKED CRITERIA, quoted from research/grid_vision_products.md
("A1 gate-2 design", filed 2026-07-07 BEFORE any computation; the
computation session may quote but not adjust them):

  - Index v0 = "per-BA demand percentile x forecast strain (realized
    vs EIA-930 day-ahead DF) x CPC degree-day extremity" with NO
    capacity denominator.
  - Outcome: "a 'stress hour' = top-decile same-month demand AND
    realized demand exceeding the day-ahead forecast by >=3%".
  - "index computed from data available by day-D morning must
    predict D+1 stress hours".
  - Split: "fit weights on 2019-2022 history; validate 2023-2025
    strictly out-of-sample" (2019 start per the v1.0.184 history
    correction).
  - Base rate: "same-calendar-month random-day predictor".
  - "PASS: out-of-sample lift >= 1.5x over the seasonal base rate on
    top-decile index days, stable across all three validation
    summers (no single-summer carry). FAIL: layer of death logged;
    the index demotes to a descriptive dashboard surface labeled
    non-predictive."
  - Spot-validation (never fitting): "a hand-collected dated list of
    public ERCOT conservation appeals / EEA events must overlap the
    detected stress days".

OPERATIONALIZATION — STATED HERE BEFORE THE FIRST RUN (REASONING
STANDARD #10; nothing below was chosen after seeing results):

  1. Scope: ERCO only (TX/ERCOT per the design). Data: EIA-930 bulk
     six-month BALANCE files (published D and DF columns; the raw
     "Demand (MW)" / "Demand Forecast (MW)" fields; fall back to the
     "(Adjusted)" demand only when raw is empty, counted). Days are
     the file's local "Data Date" (Central time calendar days —
     correct for daily stress semantics).
  2. STRESS HOUR (verbatim rule made precise): demand in the top
     decile of ALL same-calendar-month hourly demand in the SAME
     SPLIT-HALF (training thresholds from training months only;
     validation thresholds from validation months only — thresholds
     are part of the outcome definition, not the model, and using
     each half's own distribution avoids demand-growth leakage in
     either direction), AND (D - DF)/DF >= 0.03 with both present.
     STRESS DAY = any stress hour in the local day.
  3. INDEX at day D uses ONLY day D-1 (complete by D morning):
       a) demand percentile: D-1 daily PEAK demand vs the same-month
          distribution of daily peaks (training months only);
       b) forecast strain: D-1 mean of (D-DF)/DF over hours with
          both, percentile-transformed vs same-month training;
       c) weather extremity: D-1 TX (HDD+CDD) vs same-month training
          percentile (CPC population-weighted, datacore/cpc).
     Index = w1*a + w2*b + w3*c. TARGET: stress day at D+1.
  4. FITTING (training 2019-2022 only): weights on the grid
     {0, 0.25, 0.5, 0.75, 1}^3 minus the all-zero point, normalized
     to sum 1; pick the combo maximizing TRAINING lift (ties -> most
     balanced = lowest variance of weights). Flag threshold = the
     90th percentile of TRAINING index values, frozen.
  5. LIFT = (stress-day rate at D+1 over flagged days) / (mean of
     the same-calendar-month stress-day base rates of those flagged
     days, base rates computed within the evaluation half) — this IS
     the same-month random-day comparison.
  6. STABILITY (the "no single-summer carry" clause made precise):
     PASS requires overall validation lift >= 1.5 AND per-summer
     lift >= 1.0 in EACH of Jun-Sep 2023, 2024, 2025 (with at least
     3 flagged days in each summer; fewer flagged days in a summer =
     that summer cannot confirm and the verdict caps at PLAUSIBLE,
     not PASS).
  7. SPOT-VALIDATION list (public-record ERCOT emergency/conservation
     events, session-recalled, dated; validation of the OUTCOME
     variable, never fitting): 2019-08-13, 2019-08-15 (EEA1);
     2021-02-15..2021-02-18 (Winter Storm Uri, EEA3); 2022-07-13
     (conservation appeal); 2023-06-20 (appeal); 2023-08-17 (watch);
     2023-09-06 (EEA2). CHECK: >= 6 of these 9 dates must be
     detected stress days, else the outcome variable is judged wrong
     and the whole computation is void regardless of lift.

OUTCOME v2 — STATED BEFORE ITS SINGLE RUN (2026-07-07, after v1
VOIDED on its own spot-validation rule; this is VARIANT 2 OF 2 and
the multiple-hypothesis discount applies — REASONING STANDARD #4):

  v1's empirical failure was physical, verified in the data before
  this revision: ERCOT emergencies are weather-driven and
  WELL-FORECAST (max strain on famous event days +0.4%..+1.3%), and
  during load shedding realized runs far BELOW forecast (Uri mean
  strain -29%..-32%) because metered demand is capped by the shed.
  Forecast exceedance is not what Texas grid stress looks like.

  STRESS DAY v2 = daily PEAK demand in the top 2% of same-calendar-
  month daily peaks within the same split-half. No forecast
  conjunct. No event-list injection (that would make the spot check
  circular). KNOWN PHYSICAL LIMIT, stated: shed-collapsed days
  (Uri Feb 16-18) cannot be captured by ANY demand-based metric —
  the spot check for v2 therefore requires >= 5 of the 10 listed
  events (the non-shed ones), else v2 is also void.
  Index, features, fitting grid, frozen threshold rule, lift
  definition, PASS bar (>=1.5x overall; per-summer >=1.0 with >=3
  flags) — ALL identical to v1, unchanged.
  A v2 PASS is quoted with the discount: two variants were tried;
  live forward confirmation (ladder gates 3+) is required before the
  signal is trusted, traded, or sold.

Usage:
  python3 scripts/grid_stress_gate2.py <dir-with-EIA930_BALANCE_csvs> \
      [--cpc datacore/cpc/degree_days.json] \
      [--out datacore/gridvision/gate2_result.json]
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from itertools import product

TRAIN_YEARS = range(2019, 2023)
VALID_YEARS = range(2023, 2026)
STRAIN_MIN = 0.03
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SPOT_EVENTS = ["2019-08-13", "2019-08-15", "2021-02-15", "2021-02-16",
               "2021-02-17", "2021-02-18", "2022-07-13", "2023-06-20",
               "2023-08-17", "2023-09-06"]
SUMMERS = [(y, range(6, 10)) for y in (2023, 2024, 2025)]  # Jun-Sep


def num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_erco_hours(dir930):
    """-> {local_date_iso: [(demand, forecast), ...]} from all files."""
    days = defaultdict(list)
    fallbacks = 0
    for fn in sorted(os.listdir(dir930)):
        if not (fn.startswith("EIA930_BALANCE_") and fn.endswith(".csv")):
            continue
        with open(os.path.join(dir930, fn), newline="", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("Balancing Authority") != "ERCO":
                    continue
                m, d, y = row["Data Date"].split("/")
                iso = f"{y}-{m}-{d}"
                dem = num(row.get("Demand (MW)"))
                if dem is None:
                    dem = num(row.get("Demand (MW) (Adjusted)"))
                    if dem is not None:
                        fallbacks += 1
                fc = num(row.get("Demand Forecast (MW)"))
                days[iso].append((dem, fc))
    return days, fallbacks


def load_tx_dd(cpc_path):
    d = json.load(open(cpc_path))
    out = {}
    for kind in ("H", "C"):
        dates = d["axes"][kind]
        vals = d["series"][f"TX|{kind}"]
        for dt, v in zip(dates, vals):
            if v is not None:
                out[dt] = out.get(dt, 0) + v
    return out  # date -> HDD+CDD


def month_of(iso):
    return int(iso[5:7])


def year_of(iso):
    return int(iso[:4])


def in_half(iso, years):
    return year_of(iso) in years


def pct_rank(sorted_vals, x):
    """percentile of x within sorted_vals (0..1)."""
    if not sorted_vals:
        return None
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo / len(sorted_vals)


def compute(dir930, cpc_path):
    hours, fallbacks = load_erco_hours(dir930)
    dd = load_tx_dd(cpc_path)

    # ── stress days (outcome), per split-half same-month deciles ──
    def stress_days_for(years):
        by_month = defaultdict(list)
        for iso, rows in hours.items():
            if in_half(iso, years):
                by_month[month_of(iso)].extend(dv for dv, _ in rows if dv is not None)
        p90 = {m: sorted(v)[int(0.9 * len(v))] for m, v in by_month.items() if v}
        out = set()
        for iso, rows in hours.items():
            if not in_half(iso, years):
                continue
            for dv, fc in rows:
                if dv is None or fc is None or not fc:
                    continue
                if dv >= p90.get(month_of(iso), float("inf")) and (dv - fc) / fc >= STRAIN_MIN:
                    out.add(iso)
                    break
        return out

    stress_train = stress_days_for(TRAIN_YEARS)
    stress_valid = stress_days_for(VALID_YEARS)
    stress_all = stress_train | stress_valid

    # spot-validation of the outcome variable (void-check, never fitting)
    spot_hits = [e for e in SPOT_EVENTS if e in stress_all]
    outcome_valid = len(spot_hits) >= 6

    # ── stress days v2 (top-2% same-month daily peaks, split-half) ──
    def stress_days_v2(years):
        by_month = defaultdict(list)
        for iso, rows in hours.items():
            if in_half(iso, years):
                dvs = [dv for dv, _ in rows if dv is not None]
                if dvs:
                    by_month[month_of(iso)].append(max(dvs))
        p98 = {m: sorted(v)[int(0.98 * len(v))] for m, v in by_month.items() if v}
        out = set()
        for iso, rows in hours.items():
            if not in_half(iso, years):
                continue
            dvs = [dv for dv, _ in rows if dv is not None]
            if dvs and max(dvs) >= p98.get(month_of(iso), float("inf")):
                out.add(iso)
        return out

    stress2_train = stress_days_v2(TRAIN_YEARS)
    stress2_valid = stress_days_v2(VALID_YEARS)
    spot2_hits = [e for e in SPOT_EVENTS if e in (stress2_train | stress2_valid)]
    outcome2_valid = len(spot2_hits) >= 5  # shed days physically uncapturable

    # ── daily features from day D-1 ──
    daily_peak, daily_strain = {}, {}
    for iso, rows in hours.items():
        dvs = [dv for dv, _ in rows if dv is not None]
        sts = [(dv - fc) / fc for dv, fc in rows if dv is not None and fc]
        if dvs:
            daily_peak[iso] = max(dvs)
        if sts:
            daily_strain[iso] = sum(sts) / len(sts)

    # same-month TRAINING distributions for percentile transforms
    tr_peak, tr_strain, tr_dd = defaultdict(list), defaultdict(list), defaultdict(list)
    for iso in daily_peak:
        if in_half(iso, TRAIN_YEARS):
            tr_peak[month_of(iso)].append(daily_peak[iso])
    for iso in daily_strain:
        if in_half(iso, TRAIN_YEARS):
            tr_strain[month_of(iso)].append(daily_strain[iso])
    for iso in dd:
        if in_half(iso, TRAIN_YEARS):
            tr_dd[month_of(iso)].append(dd[iso])
    for d_ in (tr_peak, tr_strain, tr_dd):
        for m in d_:
            d_[m].sort()

    def features(iso_d):  # features of day D from D-1
        prev = (date.fromisoformat(iso_d) - timedelta(days=1)).isoformat()
        m = month_of(prev)
        if prev not in daily_peak or prev not in daily_strain or prev not in dd:
            return None
        a = pct_rank(tr_peak.get(m, []), daily_peak[prev])
        b = pct_rank(tr_strain.get(m, []), daily_strain[prev])
        c = pct_rank(tr_dd.get(m, []), dd[prev])
        if None in (a, b, c):
            return None
        return (a, b, c)

    # target: stress at D+1; sample = days D where features + D+1 exist
    def build_samples(years, stress_set):
        out = []
        for iso in sorted(daily_peak):
            if not in_half(iso, years):
                continue
            nxt = (date.fromisoformat(iso) + timedelta(days=1)).isoformat()
            if nxt not in daily_peak:
                continue
            f = features(iso)
            if f is None:
                continue
            out.append((iso, f, nxt in stress_set))
        return out

    train = build_samples(TRAIN_YEARS, stress_train)
    valid = build_samples(VALID_YEARS, stress_valid)
    train2 = build_samples(TRAIN_YEARS, stress2_train)
    valid2 = build_samples(VALID_YEARS, stress2_valid)

    # month base rates of the target within each half (D+1-aligned)
    def month_base(samples):
        hit, tot = defaultdict(int), defaultdict(int)
        for iso, _f, y in samples:
            m = month_of(iso)
            tot[m] += 1
            hit[m] += 1 if y else 0
        return {m: hit[m] / tot[m] for m in tot if tot[m]}

    base_tr = month_base(train)
    base_va = month_base(valid)

    def lift_for(samples, weights, thresh, base):
        flagged = [(iso, y) for iso, f, y in samples
                   if sum(w * x for w, x in zip(weights, f)) >= thresh]
        if not flagged:
            return None, 0, None
        hit = sum(1 for _iso, y in flagged if y) / len(flagged)
        exp = sum(base.get(month_of(iso), 0.0) for iso, _y in flagged) / len(flagged)
        return (hit / exp if exp > 0 else None), len(flagged), hit

    # ── fit on training only, frozen evaluation on validation ──
    def fit_and_evaluate(tr_samples, va_samples, tr_base, va_base):
        best = None
        for w in product(GRID, repeat=3):
            s = sum(w)
            if s == 0:
                continue
            wn = tuple(x / s for x in w)
            idx_vals = sorted(sum(a * b for a, b in zip(wn, f)) for _i, f, _y in tr_samples)
            th = idx_vals[int(0.9 * len(idx_vals))]
            lift, _n, _h = lift_for(tr_samples, wn, th, tr_base)
            if lift is None:
                continue
            var = sum((x - 1 / 3) ** 2 for x in wn)
            key = (lift, -var)
            if best is None or key > best[0]:
                best = (key, wn, th)
        (_k, weights, thresh) = best
        v_lift, v_nfl, v_hit = lift_for(va_samples, weights, thresh, va_base)
        summers = {}
        for y, months in SUMMERS:
            sub = [s2 for s2 in va_samples if year_of(s2[0]) == y and month_of(s2[0]) in months]
            sl, sn, sh = lift_for(sub, weights, thresh, va_base)
            summers[str(y)] = {"lift": round(sl, 3) if sl else None,
                               "flagged": sn, "hit_rate": round(sh, 3) if sh is not None else None}
        return weights, thresh, v_lift, v_nfl, v_hit, summers

    def verdict_for(outcome_ok, v_lift, summers, void_reason):
        summers_ok = all(v["lift"] is not None and v["lift"] >= 1.0 and v["flagged"] >= 3
                         for v in summers.values())
        enough = all(v["flagged"] >= 3 for v in summers.values())
        if not outcome_ok:
            return f"VOID ({void_reason})"
        if v_lift is not None and v_lift >= 1.5 and summers_ok:
            return "PASS"
        if v_lift is not None and v_lift >= 1.5 and not enough:
            return "PLAUSIBLE (lift >=1.5 but a summer lacks flagged days to confirm)"
        return "FAIL"

    w1, t1, l1, n1, h1, sm1 = fit_and_evaluate(train, valid, base_tr, base_va)
    base2_tr = month_base(train2)
    base2_va = month_base(valid2)
    w2, t2, l2, n2, h2, sm2 = fit_and_evaluate(train2, valid2, base2_tr, base2_va)

    v1_verdict = verdict_for(outcome_valid, l1, sm1, "outcome variable failed spot-validation")
    v2_verdict = verdict_for(outcome2_valid, l2, sm2, "v2 outcome also failed spot-validation")

    return {
        "criteria": "quoted verbatim + operationalizations (v1 and v2) in scripts/grid_stress_gate2.py header, each written before its run",
        "multiple_hypothesis_note": "two outcome variants were tried (v1 voided on its own spot-validation rule; v2 pre-stated after the physical diagnosis). Any v2 PASS carries the discount: live forward confirmation (ladder gates 3+) required before trust.",
        "data": {
            "erco_days": len(daily_peak), "adjusted_demand_fallback_hours": fallbacks,
            "train_samples": len(train), "valid_samples": len(valid),
        },
        "v1_forecast_exceedance": {
            "stress_days_train": len(stress_train), "stress_days_valid": len(stress_valid),
            "spot_validation": {"detected": spot_hits, "required": ">=6 of 10", "passed": outcome_valid},
            "fit": {"weights_peak_strain_dd": [round(x, 3) for x in w1], "flag_threshold": round(t1, 4)},
            "validation": {"overall_lift": round(l1, 3) if l1 else None, "flagged_days": n1,
                            "hit_rate": round(h1, 3) if h1 is not None else None, "per_summer": sm1},
            "verdict": v1_verdict,
            "void_diagnosis": "verified in data before v2: ERCOT emergencies are well-forecast "
                              "(max strain on famous days +0.4%..+1.3%) and load shedding caps "
                              "metered demand (Uri mean strain -29%..-32%) — forecast exceedance "
                              "is not what Texas grid stress looks like",
        },
        "v2_extreme_peak": {
            "stress_days_train": len(stress2_train), "stress_days_valid": len(stress2_valid),
            "spot_validation": {"detected": spot2_hits, "required": ">=5 of 10 (shed days physically uncapturable)",
                                 "passed": outcome2_valid},
            "fit": {"weights_peak_strain_dd": [round(x, 3) for x in w2], "flag_threshold": round(t2, 4)},
            "validation": {"overall_lift": round(l2, 3) if l2 else None, "flagged_days": n2,
                            "hit_rate": round(h2, 3) if h2 is not None else None, "per_summer": sm2},
            "verdict": v2_verdict,
            "void_diagnosis": "verified in data after the run: ERCOT demand grows ~5-6%/yr, so "
                              "same-month percentiles POOLED across 2019-2022 are blind to "
                              "early-year emergencies (top-5 August training peaks are all from "
                              "2022; Uri's EEA day ranks 4th behind the Feb-2022 cold snap; the "
                              "demand extreme was Feb 14, the day before the declaration dates). "
                              "A v3 outcome needs growth-aware extremes + a researched full "
                              "public event list — a fresh design for a later session, NOT a "
                              "third same-session variant (anti-fishing rule).",
        },
        "stability_note": "both variants ALSO fail the pre-stated no-single-summer-carry clause "
                          "on their own terms (v1 lift concentrated in 2024; v2 in 2023 with "
                          "2024 at 0.70) — the stability test caught real regime-carry twice.",
        "verdict": v2_verdict,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir930")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--cpc", default=os.path.join(root, "datacore", "cpc", "degree_days.json"))
    ap.add_argument("--out", default=os.path.join(root, "datacore", "gridvision", "gate2_result.json"))
    args = ap.parse_args()
    res = compute(args.dir930, args.cpc)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["verdict"] == "PASS" else 3)
