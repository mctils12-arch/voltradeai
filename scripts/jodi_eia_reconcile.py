#!/usr/bin/env python3
"""
jodi_eia_reconcile.py — GATE 1 (DATA) workup for the JODI oil-stocks root
(datacore/signal_ladder.json id jodi_oil_stocks), closing the "definitional
gap" left open by the 2026-07-07 first look (research/experiments.md
2026-07-07 [PIPELINE] entry, ~line 23254).

WHY THIS SCRIPT EXISTS (EDGE DOCTRINE #3 — compile knowledge into code):
the first look compared JODI's US|CRUDEOIL series against EIA's commercial
+ SPR total for a single month (2026-04) and found a ~19% non-match,
filed as "a definitional gap ... reconciling JODI's stock definition
against EIA monthlies is the gate-1 workup before any signal use." That
workup had never been run since. This script is the reusable, re-runnable
comparator — every future re-verification (new JODI/EIA vintages land
monthly) is one `python3 scripts/jodi_eia_reconcile.py`, not a repeated
one-off analysis.

METHOD: both datacore/eia/weekly_series.json (crude_stocks_us = EIA
"Ending Stocks excluding SPR", spr_crude_stocks = EIA SPR stocks; both
Thousand Barrels, weekly, git-committed static artifact) and
datacore/jodi/primary_stocks.json (US|CRUDEOIL, US|TOTCRUDE = CRUDEOIL +
OTHERCRUDE; KBBL, monthly, git-committed static artifact) are already
verified-source archives already in this repo — no network call, this
is a pure reconciliation of two things we already trust individually.
EIA weekly points are averaged into calendar months to match JODI's
monthly cadence. Two JODI comparators are tested against EIA's
commercial+SPR total: CRUDEOIL alone (the first look's choice) and
TOTCRUDE = CRUDEOIL + OTHERCRUDE.

PRIOR (stated before running, Reasoning Standard #10): CRUDEOIL alone
undershoots the EIA total (JODI's "closing stock" concept is broader,
IEA/JODI methodology also counts other-crude/condensate blending EIA's
"crude oil" total excludes) — TOTCRUDE is expected to reconcile much
more closely. Full result recorded in research/experiments.md (this
session's [PIPELINE] entry) and datacore/signal_ladder.json.

VERDICT RULE (stated before running): TOTCRUDE reconciles as a GATE 1
DATA pass if (a) the average level gap is a single-digit percent of the
EIA total over the full overlapping history, AND (b) month-over-month
CHANGES correlate at r >= 0.5 (a real, usable co-movement, not proof of
identical accounting) — with the residual gap and any drift reported
honestly regardless of the verdict, since JODI's own value proposition
(non-OECD stock visibility) can NEVER be checked this way (no US-style
comparator exists for Saudi Arabia/UAE/India) — this reconciliation only
establishes "does JODI measure a real, EIA-consistent quantity for a
country we CAN check," not "is JODI accurate for the countries the
hypothesis is actually about."
"""
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(__file__)
EIA_PATH = os.path.join(HERE, "..", "datacore", "eia", "weekly_series.json")
JODI_PATH = os.path.join(HERE, "..", "datacore", "jodi", "primary_stocks.json")

LEVEL_GAP_PASS_PCT = 10.0   # |mean diff| as % of EIA total, full-history
MOM_CORR_PASS = 0.5


def monthly_avg(points):
    """[[date, value, ...], ...] -> {YYYY-MM: mean(value)}. Handles both
    weekly (EIA) and monthly (JODI) point series identically."""
    by_month = defaultdict(list)
    for p in points:
        by_month[p[0][:7]].append(p[1])
    return {m: sum(v) / len(v) for m, v in by_month.items()}


def _corr(a, b):
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / n
    sda = (sum((x - ma) ** 2 for x in a) / n) ** 0.5
    sdb = (sum((x - mb) ** 2 for x in b) / n) ** 0.5
    if sda == 0 or sdb == 0:
        return None
    return cov / (sda * sdb)


def mom_delta_correlation(eia_series, jodi_series):
    """Correlation of month-over-month CHANGES (not levels) — the honest
    test for a usable co-movement signal when levels carry a definitional
    offset."""
    ed = [eia_series[i] - eia_series[i - 1] for i in range(1, len(eia_series))]
    jd = [jodi_series[i] - jodi_series[i - 1] for i in range(1, len(jodi_series))]
    return _corr(ed, jd)


def reconcile(eia_series: dict, jodi_series: dict, jodi_product: str) -> dict:
    """eia_series: {'crude_stocks_us': {...points...}, 'spr_crude_stocks': {...}}
    jodi_series: {'US|CRUDEOIL': {...}, 'US|TOTCRUDE': {...}}
    Returns a dict with the full reconciliation for one JODI product key."""
    crude_m = monthly_avg(eia_series["crude_stocks_us"]["points"])
    spr_m = monthly_avg(eia_series["spr_crude_stocks"]["points"])
    jodi_m = {p[0]: p[1] for p in jodi_series[jodi_product]["points"]}

    months = sorted(set(crude_m) & set(spr_m) & set(jodi_m))
    if len(months) < 2:
        return {"product": jodi_product, "n_months": len(months), "insufficient": True}

    eia_tot = [crude_m[m] + spr_m[m] for m in months]
    jodi_vals = [jodi_m[m] for m in months]
    diffs = [jodi_vals[i] - eia_tot[i] for i in range(len(months))]
    diff_pct = [d / e * 100 for d, e in zip(diffs, eia_tot)]

    n_recent = min(24, len(months))
    recent_diffs = diffs[-n_recent:]
    recent_pct = diff_pct[-n_recent:]

    return {
        "product": jodi_product,
        "n_months": len(months),
        "first_month": months[0],
        "last_month": months[-1],
        "diff_mean": statistics.mean(diffs),
        "diff_stdev": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
        "diff_pct_of_eia_mean": statistics.mean(diff_pct),
        "recent_n": n_recent,
        "recent_diff_mean": statistics.mean(recent_diffs),
        "recent_diff_pct_of_eia_mean": statistics.mean(recent_pct),
        "recent_diff_stdev": statistics.stdev(recent_diffs) if len(recent_diffs) > 1 else 0.0,
        "mom_corr_full": mom_delta_correlation(eia_tot, jodi_vals),
        "mom_corr_recent": mom_delta_correlation(eia_tot[-n_recent:], jodi_vals[-n_recent:]),
    }


def verdict(result: dict) -> str:
    if result.get("insufficient"):
        return "INSUFFICIENT_SAMPLE"
    gap_ok = abs(result["diff_pct_of_eia_mean"]) <= LEVEL_GAP_PASS_PCT
    corr = result["mom_corr_full"]
    corr_ok = corr is not None and corr >= MOM_CORR_PASS
    return "GATE1_PASS" if (gap_ok and corr_ok) else "GATE1_FAIL"


def main():
    with open(EIA_PATH) as f:
        eia = json.load(f)["series"]
    with open(JODI_PATH) as f:
        jodi = json.load(f)["series"]

    print("JODI vs EIA reconciliation — US crude closing stocks (Thousand Barrels)\n")
    results = {}
    for product in ("US|CRUDEOIL", "US|TOTCRUDE"):
        r = reconcile(eia, jodi, product)
        results[product] = r
        if r.get("insufficient"):
            print(f"{product}: INSUFFICIENT SAMPLE ({r['n_months']} months)")
            continue
        v = verdict(r)
        print(f"=== {product} vs EIA (commercial ex-SPR + SPR) === verdict: {v}")
        print(f"  {r['n_months']} months, {r['first_month']}..{r['last_month']}")
        print(f"  full-history: mean diff = {r['diff_mean']:+.0f} kbbl "
              f"({r['diff_pct_of_eia_mean']:+.1f}% of EIA total), "
              f"stdev {r['diff_stdev']:.0f}")
        print(f"  recent {r['recent_n']}mo: mean diff = {r['recent_diff_mean']:+.0f} kbbl "
              f"({r['recent_diff_pct_of_eia_mean']:+.1f}%), stdev {r['recent_diff_stdev']:.0f}")
        print(f"  MoM delta correlation: full={r['mom_corr_full']:.3f}  "
              f"recent={r['mom_corr_recent']:.3f}")
        print()

    print("NOTE: this reconciliation only covers the US (the one country with an\n"
          "independent EIA comparator). JODI's actual hypothesis value is non-OECD\n"
          "coverage (Saudi Arabia/UAE/India) where NO comparator exists — a US pass\n"
          "here validates that JODI measures a real, EIA-consistent quantity, not\n"
          "that non-OECD JODI figures are trustworthy to the same degree.")

    out_path = os.path.join(HERE, "..", "jodi_eia_reconcile_results.json")
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nFull results written to {out_path} (session-local, not committed).")


if __name__ == "__main__":
    main()
