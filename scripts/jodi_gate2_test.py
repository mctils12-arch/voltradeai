#!/usr/bin/env python3
"""
jodi_gate2_test.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for the JODI
non-OECD closing-stock root (datacore/jodi/primary_stocks.json, gate 1
DATA passed 2026-08-06 — see research/experiments.md and
datacore/signal_ladder.json's jodi_oil_stocks entry, which names this
exact test as its queued NEXT STEP: "non-OECD stock BUILDS (not levels)
vs forward Brent/USO returns, per the root's original hypothesis; use
US|TOTCRUDE (not CRUDEOIL) as the product key everywhere downstream").

PRE-REGISTERED HYPOTHESIS (stated before running, REASONING STANDARD #10):
an aggregate MONTH-OVER-MONTH BUILD in non-OECD crude-exporter closing
stocks (TOTCRUDE) predicts a NEGATIVE forward return in crude oil ETFs
(BNO/Brent proxy, USO/WTI proxy) over the following 20/60 trading days —
more supply sitting in tanks is bearish; a DRAW predicts a positive
forward return. Direction chosen from the plain supply/demand mechanism
(inventory accumulation signals oversupply), not fit to the data.

PRIOR (stated before running): LOW-MODERATE (~25-30%) confidence this
clears a strict bar. Second-order reasoning (REASONING STANDARD #5) cuts
against it: OPEC+ producers' output/stock trends are already tracked in
near-real-time by professional oil desks via tanker-tracking, satellite
imagery, and OPEC's own monthly reports — JODI's ~2-month-lagged official
release is, if anything, STALE by the time it publishes, the opposite of
an information edge. This is tested anyway because gate 1 just validated
the underlying data reconciles to a trusted external source (EIA) for the
one country checkable that way, and this is the root's own pre-filed next
step — but the honest prior is that this is more likely a confirmation of
EDGE DOCTRINE #1 (compiled, free processing of public data as an asset in
itself) than a live trading edge, and a null result here would not be a
surprise.

UNIVERSE SELECTION (data-quality-first pass, run and logged before any
price-fetch or return calculation — REASONING STANDARD #7, survivorship/
data-quality must be checked before results, not after): surveyed
TOTCRUDE coverage for all 96 non-OECD JODI REF_AREAs in the archive.
Exactly 5 have COMPLETE (zero missing/zero-value points) TOTCRUDE history
spanning the full archive window 2009-01..2026-04 (208/208 months):
Saudi Arabia (SA), Nigeria (NG), Algeria (DZ), Brunei (BN), and Taiwan
(TW). The root's originally-named motivating examples FAIL this bar
outright and are EXCLUDED, stated honestly rather than forced to fit:
UAE (AE) stopped reporting TOTCRUDE after 2018-12 (and was only 36/120
nonzero even before that); India (IN) reports TOTCRUDE with many literal
zero-value months (not credible as a real closing-stock level for a
country of India's refining capacity — treated as an unreliable series,
not silently used). Taiwan (TW) has complete data but is a refiner/
importer, not a net crude exporter — excluded to keep the universe
consistent with the pre-registered supply-side mechanism (producer
withholding/releasing barrels into the market), not muddied with a
demand-side buffer-stock country. UNIVERSE = SA, NG, DZ, BN — chosen
entirely by this data-quality/producer criterion, before any result was
looked at.

METHODOLOGY: per-country month-over-month level delta, z-scored against
its own TRAILING 36-month window (no lookahead — mirrors the COT-index
trailing-window convention already used in cot_gate2_test.py /
cftc_tff_gate2_test.py; the current period is never included in its own
normalization window). Composite = mean of the available country
z-scores for months where at least 3 of the 4 countries have a valid
z-score. Bucket: BUILD (composite z > 0) / DRAW (composite z <= 0) — a
plain sign split, no threshold tuned post hoc. Publish lag: JODI's own
documented ~2-month release lag (jodi_oil.py's docstring) — the entry
anchor is the first trading bar strictly after (period end-of-month + 60
calendar days), reusing find_entry_index and the Newey-West HAC test
byte-for-byte from cftc_tff_gate2_test.py (EDGE DOCTRINE #3: reuse the
already-built, already-tested machinery rather than re-deriving it). The
HAC lag is re-tuned to monthly-observation spacing — round(horizon / 21)
trading days per month, rather than that script's weekly
round(horizon / 5) — since JODI observations are monthly, not weekly.

LEFT-CENSORING GUARD (new — not needed in the weekly CFTC scripts, whose
CFTC history postdates every symbol's ETF listing): BNO's price history
only starts 2010-06-02, well after the JODI archive's 2009-01 start. Any
period whose publish_date falls before the asset's own first available
price bar is dropped entirely rather than silently entered at the
asset's IPO bar — the latter would fake an early-history entry against a
price 1+ year stale relative to the actual signal date.

HORIZONS: 20/60 trading days, matching every other gate-2 script in this
repo. ASSETS: BNO (Brent proxy), USO (WTI proxy). Comparisons counted for
the Bonferroni bar: 2 assets x 2 horizons = 4 (the BUILD-vs-complement
and DRAW-vs-complement tests are mirror images of each other under a
2-bucket split — only one direction per (asset, horizon) is a genuinely
independent comparison, so only "build" is tested). Bonferroni bar =
0.05 / 4 = 0.0125.

Pure statistical measurement only — SIGNAL gate, no trading involved.
Does not import or touch bot_engine.py / deep_score / system_config.py.

Usage: python3 jodi_gate2_test.py [--out jodi_gate2_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from cftc_tff_gate2_test import _newey_west_diff_test, find_entry_index  # noqa: E402

JODI_PATH = os.path.join(os.path.dirname(__file__), "..", "datacore", "jodi", "primary_stocks.json")
PRODUCT = "TOTCRUDE"
COUNTRIES = ("SA", "NG", "DZ", "BN")  # see docstring UNIVERSE SELECTION
ASSETS = ("BNO", "USO")
HORIZONS = (20, 60)  # trading days
PUBLISH_LAG_DAYS = 60  # ~2-month JODI release lag, jodi_oil.py's own docstring
ZSCORE_WINDOW = 36  # trailing months
MIN_COUNTRIES_FOR_COMPOSITE = 3


def period_end_date(period: str) -> str:
    """'2026-04' -> '2026-04-30' (calendar month-end, stdlib only)."""
    y, m = (int(p) for p in period.split("-"))
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    last_day = datetime(ny, nm, 1) - timedelta(days=1)
    return last_day.strftime("%Y-%m-%d")


def load_country_series(country: str, jodi_path: str = JODI_PATH) -> dict:
    """period -> level (KBBL) for country|TOTCRUDE, ascending."""
    with open(jodi_path) as f:
        archive = json.load(f)
    key = f"{country}|{PRODUCT}"
    points = archive["series"][key]["points"]
    return {p[0]: p[1] for p in sorted(points, key=lambda p: p[0])}


def build_deltas(level_by_period: dict) -> dict:
    """period -> MoM level delta (the first period has no prior, skipped)."""
    periods = sorted(level_by_period)
    out = {}
    for i in range(1, len(periods)):
        out[periods[i]] = level_by_period[periods[i]] - level_by_period[periods[i - 1]]
    return out


def zscore_trailing(delta_by_period: dict, window: int = ZSCORE_WINDOW) -> dict:
    """period -> z-score of that period's delta against the TRAILING
    `window` prior deltas (current period excluded from its own
    normalization window — no lookahead). Empty until `window` prior
    points exist; None (not zero) when the trailing window is degenerate
    (zero variance)."""
    periods = sorted(delta_by_period)
    out = {}
    for i, p in enumerate(periods):
        if i < window:
            continue
        prior = [delta_by_period[periods[j]] for j in range(i - window, i)]
        mean = sum(prior) / len(prior)
        var = sum((v - mean) ** 2 for v in prior) / len(prior)
        std = var ** 0.5
        out[p] = (delta_by_period[p] - mean) / std if std > 0 else None
    return out


def build_composite(countries=COUNTRIES, jodi_path: str = JODI_PATH,
                     window: int = ZSCORE_WINDOW,
                     min_countries: int = MIN_COUNTRIES_FOR_COMPOSITE) -> dict:
    """period -> {"z": composite z-score, "n_countries": int}."""
    per_country_z = {
        c: zscore_trailing(build_deltas(load_country_series(c, jodi_path)), window)
        for c in countries
    }
    all_periods = sorted(set().union(*(set(z) for z in per_country_z.values())) if per_country_z else set())
    composite = {}
    for p in all_periods:
        vals = [per_country_z[c][p] for c in countries if per_country_z[c].get(p) is not None]
        if len(vals) >= min_countries:
            composite[p] = {"z": sum(vals) / len(vals), "n_countries": len(vals)}
    return composite


def compute_forward_returns(composite: dict, bars: dict,
                             publish_lag_days: int = PUBLISH_LAG_DAYS,
                             horizons=HORIZONS) -> list:
    """Pure function: for each composite month, find its no-lookahead entry
    anchor in `bars` and compute forward N-day returns. Periods whose
    publish date precedes the asset's own first bar are dropped entirely
    (LEFT-CENSORING GUARD, see module docstring) — never entered at the
    IPO bar. Periods too close to the end of `bars` for a given horizon
    are dropped for that horizon only (right-censoring honesty, never
    zero-filled), matching cftc_tff_gate2_test.py's convention."""
    bar_dates = bars["date"]
    bar_closes = bars["close"]
    if not bar_dates:
        return []
    first_bar_date = bar_dates[0]
    out = []
    for period in sorted(composite):
        publish_date = (datetime.strptime(period_end_date(period), "%Y-%m-%d")
                         + timedelta(days=publish_lag_days)).strftime("%Y-%m-%d")
        if publish_date < first_bar_date:
            continue
        entry_idx = find_entry_index(bar_dates, publish_date)
        z = composite[period]["z"]
        row = {
            "period": period,
            "composite_z": round(z, 4),
            "bucket": "build" if z > 0 else "draw",
            "entry_date": bar_dates[entry_idx] if entry_idx is not None else None,
            "forward_returns": {},
        }
        if entry_idx is not None:
            entry_price = bar_closes[entry_idx]
            for h in horizons:
                exit_idx = entry_idx + h
                if exit_idx < len(bar_closes) and entry_price:
                    row["forward_returns"][h] = bar_closes[exit_idx] / entry_price - 1
        out.append(row)
    return out


def summarize(rows: list, horizons=HORIZONS) -> dict:
    summary = {}
    for h in horizons:
        vals = {"build": [], "draw": []}
        for r in rows:
            fr = r["forward_returns"].get(h)
            if fr is None:
                continue
            vals[r["bucket"]].append(fr)
        summary[str(h)] = {
            b: {"n": len(v), "mean_pct": round(sum(v) / len(v) * 100, 3) if v else None}
            for b, v in vals.items()
        }
    return summary


def hac_significance(rows: list, horizons=HORIZONS) -> dict:
    """Per-horizon Newey-West test of 'build' vs the complement ('draw'),
    lag re-tuned to monthly-observation spacing (see module docstring)."""
    out = {}
    for h in horizons:
        lag = max(1, round(h / 21))
        out[str(h)] = _newey_west_diff_test(rows, h, "build", lag=lag)
    return out


def run(asset: str, fetch_bars_fn, composite: dict) -> dict:
    days_needed = 6500  # covers the full 2009-present JODI span
    bars = fetch_bars_fn(asset, days_needed)
    if not bars or not bars.get("date"):
        return {"asset": asset, "status": "no_price_data"}
    rows = compute_forward_returns(composite, bars)
    if not rows:
        return {"asset": asset, "status": "no_usable_rows"}
    return {
        "asset": asset,
        "status": "ok",
        "n_periods_in_range": len(rows),
        "n_periods_with_entry": sum(1 for r in rows if r["entry_date"] is not None),
        "summary": summarize(rows),
        "significance": hac_significance(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="jodi_gate2_results.json")
    args = ap.parse_args()

    from backtest_v2 import fetch_bars  # local import: keeps this script's
    # unit tests free of backtest_v2's network/cache side effects

    composite = build_composite()
    print(f"composite built: {len(composite)} usable months "
          f"({min(composite) if composite else 'n/a'}..{max(composite) if composite else 'n/a'})")

    results = {"composite_months": len(composite), "universe": list(COUNTRIES), "assets": {}}
    for asset in ASSETS:
        print(f"--- {asset} ---")
        try:
            results["assets"][asset] = run(asset, fetch_bars, composite)
        except Exception as e:  # one asset's failure must not abort the other
            results["assets"][asset] = {"asset": asset, "status": "error", "reason": str(e)[:200]}
        print(json.dumps(results["assets"][asset], indent=2, default=str))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
