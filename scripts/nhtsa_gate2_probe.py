#!/usr/bin/env python3
"""
scripts/nhtsa_gate2_probe.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for the
nhtsa_vehicle_complaints root, the exact NEXT item the 2026-09-06 gate-1
session queued: "gate 2 (velocity anomalies vs forward returns) is the
ladder's own next real step, and ... must NOT reuse full-calendar-year
complaint buckets as the velocity measure — a future session should design a
trailing-window rate computed strictly from complaints dated before any
public recall-adjacent event, or the 'signal' will just be recall publicity
restated" (research/experiments.md, this UTC day's first session).

WHY THIS DESIGN SATISFIES THAT CAVEAT: gate 1's yearly-bucket design was
reflexive because a full calendar-year bucket mixes complaints filed BEFORE
and AFTER the recall's own public announcement in the same bin, so a
"recall-year spike" partly just measures media/litigation attention landing
complaints, not prediction. This script instead computes a per-CALENDAR-DAY
complaint-count z-score against a strictly-PRIOR trailing window (identical
causal contract to scripts/wikiattention_gate2.py's zscore_series — day i's
own count and every future day are excluded from day i's own baseline), then
measures the STOCK RETURN strictly AFTER the spike is detected. Nothing about
a future recall date is used anywhere in this script; the test only asks
"does an unusual jump in raw complaint volume, using only data available up
to that jump, precede unusual stock returns" — a genuinely prospective
question, unlike gate 1's retrospective known-case check.

FURTHER DEPARTURE FROM GATE 1, DELIBERATE: gate 1 filtered complaints by a
component KEYWORD list specific to each known defect (only knowable in
hindsight). Gate 2 uses ALL complaints, no keyword filter — a real trading
signal cannot know in advance which component will turn out to matter, so
testing the keyword-filtered version would silently smuggle hindsight back
in. This makes gate 2 a strictly harder, more honest test than gate 1's.

WATCHLIST: reuses the existing curated ticker-mapped fleet from
datacore/nhtsa_vehicles.json (server/nhtsaComplaints.ts's own live archiver
watchlist) rather than inventing a new one — the fleet BUILD ORDER 6 already
curated and ticker-mapped for exactly this purpose. Complaint rows for every
make/model/year entry that shares a ticker are pooled into that ticker's own
single daily count series (a company's total NHTSA complaint footprint
across its watched models, not any one model in isolation).

GROUPS (EDGE DOCTRINE #2 — fish where whales can't): PRIMARY = the less
analyst-covered names in the watchlist (RIVN, LCID — newer/smaller EV
entrants; STLA, HYMTF, NSANY, VWAGY — foreign OTC ADRs with materially
thinner US sell-side coverage than a domestic large-cap). SECONDARY
(informational-only comparison, same convention as wikiattention_gate2's
mega-cap row) = the large, heavily-covered US-listed automakers (TSLA, F,
GM, TM, HMC) where any real complaint-velocity edge this small is far more
likely to already be arbitraged away.

TWO REAL DATA-QUALITY FINDINGS FROM THIS SESSION'S OWN LIVE PROBING, COMPILED
HERE SO A FUTURE SESSION DOES NOT RE-DISCOVER THEM (EDGE DOCTRINE #3):
  1. api.nhtsa.gov/complaints/complaintsByVehicle returns HTTP 400 (not 200)
     for a query that legitimately matches ZERO complaints, even though the
     JSON body is well-formed ({"count":0,"results":[]}). Verified live:
     tesla/"model y"/2024 -> HTTP 400 with that exact body. A naive
     `if not response.ok: raise` would misclassify a true empty result as a
     fetch failure; this script's fetch_json() parses the error body as JSON
     on a 4xx before giving up, so a genuine zero-count result is preserved
     as [] rather than silently dropped as an error.
  2. HYMTF (Hyundai's US OTC ADR) has no usable daily bar history via
     backtest_v2.fetch_bars() in this session's sandbox (Alpaca miss, Yahoo
     404) — degrades honestly (ticker excluded, logged), not a hard failure
     of the whole run, matching this repo's own per-ticker degradation
     convention (e.g. wikiattention_gate2_newsfree.py's per-page fetch
     failures).
  3. Complaint density for CURRENT model-year vehicles (2024/2025, this
     watchlist's own curation convention — "complaint velocity on CURRENT
     product is the signal") is far lower than the well-established historic
     recall cases gate 1 tested: totals across this session's live pull
     ranged 10 (Lucid Air 2024) to 719 (GM's three watched models combined)
     over roughly 2.5 years, vs. gate 1's Cobalt case alone (2,330 complaints
     over 8 years). This directly motivates using a SHORT trailing window
     (30 calendar days, not wikiattention's 90) — a longer window would
     rarely accumulate enough non-zero days to produce a meaningful z-score
     at this volume, and low-volume tickers (RIVN, LCID) are expected to
     contribute few or no qualifying spike days at all, which is the correct,
     honest behavior (zscore_series's own std==0 guard, reused unmodified
     from wikiattention_gate2.py), not a bug to work around.

PRE-REGISTRATION (REASONING STANDARD #10, written before any live complaint
count or forward return was computed):
HYPOTHESIS: a complaint-count spike (z>=2.0 vs the trailing 30-calendar-day
mean/std, using only complaints filed strictly before the spike day) is
followed by NEGATIVE abnormal stock returns over the following 5/10/20
trading days, on the theory that a rising defect-complaint footprint raises
the forward probability of a recall, investigation, or negative coverage
event that the market has not yet priced (EDGE DOCTRINE #1/#2: this is a
free, public, unglamorous dataset that requires real curation labor to turn
into a per-ticker daily series, exactly the kind of processing edge this
system is built to find).
PRIOR (informal, ~20%, stated before running): (1) REASONING STANDARD #5,
second-order — NHTSA's ODI complaint database is fully public and has
existed for decades; if a simple complaint-count-velocity signal this
obvious were robust and liquid enough to trade, it plausibly would already
be arbitraged in the LARGE, heavily-covered names (this is exactly why the
MEGA group is only an informational comparison, never the group gate 2 can
pass on); the PRIMARY group's foreign-OTC/smaller-cap composition is the
one place EDGE DOCTRINE #2 gives a structural reason coverage could be
thin. (2) COSTS ASIDE (REASONING STANDARD #6 — costs are explicitly gate 3's
job, not gate 2's) the low complaint volumes found during this session's own
probing (see finding #3 above) mean per-ticker power is weak; the pooled
test is the only one with any chance of detecting a real effect, and even
that pool is small (11 watchlist tickers, 5 of which after Bonferroni-
relevant filtering land in each group). (3) REASONING STANDARD #4 — this is
ONE pre-registered design, not a family of variants; no keyword filter, no
crash/fire-only subset, no alternate window tried before or after seeing a
result.
VERDICT RULE (stated before running): GATE 2 passes only if the PRIMARY
group's pooled forward-return effect is BOTH (a) significant at the
Bonferroni bar for this 2-group x 3-horizon family (alpha/6 ~= 0.00833) at
ANY tested horizon, AND (b) in the hypothesized NEGATIVE direction. A
significant effect in the wrong direction, a significant MEGA-only effect
with PRIMARY null, or nothing surviving the bar at all, are all NOT a pass —
mirrors this repo's established discipline (e.g. wikiattention_gate2's own
insistence that only the small/mid PRIMARY group's own result can promote
the root, the mega row is informational-only).

NOT ATTEMPTED HERE (left for gate 3, if gate 2 passes): any entry/exit rule,
any cost/slippage deduction, any ablation, any claim this is tradeable.

Run (live, hits api.nhtsa.gov + backtest_v2's Alpaca/Yahoo price fetch,
politely spaced, ~30-60s for 20 watchlist rows across 11 tickers):
  python3 scripts/nhtsa_gate2_probe.py
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))

_G1_SPEC = importlib.util.spec_from_file_location("nhtsa_gate1_probe", os.path.join(_HERE, "nhtsa_gate1_probe.py"))
g1 = importlib.util.module_from_spec(_G1_SPEC)
_G1_SPEC.loader.exec_module(g1)

_WAG2_SPEC = importlib.util.spec_from_file_location("wikiattention_gate2", os.path.join(_HERE, "wikiattention_gate2.py"))
wag2 = importlib.util.module_from_spec(_WAG2_SPEC)
_WAG2_SPEC.loader.exec_module(wag2)

COMPLAINTS_API = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
WATCHLIST_PATH = os.path.join(os.path.dirname(_HERE), "datacore", "nhtsa_vehicles.json")

DEFAULT_TRAILING_WINDOW = 30   # calendar days (see finding #3 in the module docstring)
DEFAULT_Z_THRESHOLD = 2.0      # matches wikiattention_gate2's own bar
DEFAULT_HORIZONS = (5, 10, 20)  # trading days — a slower fundamental signal than attention/volume

# Curated 2026-09-06 per EDGE DOCTRINE #2 (see module docstring GROUPS section).
PRIMARY_GROUP_TICKERS = ("RIVN", "LCID", "STLA", "HYMTF", "NSANY", "VWAGY")
SECONDARY_GROUP_TICKERS = ("TSLA", "F", "GM", "TM", "HMC")


# ── Pure functions (unit-tested, no network) ────────────────────────────────

def load_watchlist(path: str = WATCHLIST_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)["vehicles"]


def group_by_ticker(vehicles: Sequence[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for v in vehicles:
        out.setdefault(v["ticker"], []).append(v)
    return out


def aggregate_daily_counts(complaint_dates: Sequence[date]) -> dict[date, int]:
    """{calendar_date: count} from a flat list of complaint-filed dates
    (already parsed; garbage/unparseable dates must be filtered by the
    caller before this is called — this function trusts its input)."""
    out: dict[date, int] = {}
    for d in complaint_dates:
        out[d] = out.get(d, 0) + 1
    return out


def build_calendar_daily_series(counts: dict[date, int], start: date, end: date) -> tuple[list[str], list[int]]:
    """Zero-filled contiguous calendar-day series over [start, end] inclusive.
    Zero is a REAL value here (a day with no complaints filed), unlike
    wikiattention's pageview series where a missing day means 'no data' —
    NHTSA's complaint feed has no such gaps, so zero-filling is honest, not a
    fabrication (contrast with align_views_to_trading_days's None-for-gaps
    convention, which this function deliberately does NOT reuse)."""
    dates: list[str] = []
    values: list[int] = []
    d = start
    while d <= end:
        dates.append(d.isoformat())
        values.append(counts.get(d, 0))
        d += timedelta(days=1)
    return dates, values


def map_calendar_to_trading_idx(cal_date_iso: str, trading_dates: Sequence[str]) -> Optional[int]:
    """Index of the first trading day on or after cal_date_iso — the
    earliest point a complaint-velocity spike detected on cal_date_iso could
    actually be acted on. None if the spike is at/after the tail of the
    trading calendar (no valid entry day)."""
    i = bisect.bisect_left(trading_dates, cal_date_iso)
    return i if i < len(trading_dates) else None


def forward_return(closes: Sequence[float], idx: int, horizon: int) -> Optional[float]:
    """close[idx+horizon] / close[idx] - 1 — buy at the close of the entry
    day, sell at the close `horizon` trading days later (the same buy/sell
    convention scripts/wikiattention_gate3.py's own docstring states for its
    entry/exit rule; gate 2 borrows the metric, not the trading claim — no
    cost is deducted here, that is explicitly gate 3's job)."""
    if idx < 0 or idx + horizon >= len(closes) or closes[idx] == 0:
        return None
    return closes[idx + horizon] / closes[idx] - 1


def evaluate_ticker(cal_dates: Sequence[str], counts: Sequence[int],
                     trading_dates: Sequence[str], closes: Sequence[float],
                     threshold: float = DEFAULT_Z_THRESHOLD,
                     window: int = DEFAULT_TRAILING_WINDOW,
                     horizons: Sequence[int] = DEFAULT_HORIZONS) -> dict:
    """One ticker's full pipeline: calendar-day complaint counts -> z-score
    spikes -> mapped trading-day entries -> forward returns vs baseline.
    Reuses wikiattention_gate2's zscore_series/spike_day_indices/
    welch_vs_baseline unmodified (EDGE DOCTRINE #3) — both are generic pure
    functions over a numeric sequence with no wiki-specific assumption."""
    z = wag2.zscore_series(list(counts), window)
    spike_cal_idxs = wag2.spike_day_indices(z, threshold)
    spike_trading_idxs: set[int] = set()
    for i in spike_cal_idxs:
        ti = map_calendar_to_trading_idx(cal_dates[i], trading_dates)
        if ti is not None:
            spike_trading_idxs.add(ti)
    all_trading_idxs = range(len(trading_dates))
    out = {
        "n_calendar_days": len(cal_dates),
        "n_spike_calendar_days": len(spike_cal_idxs),
        "n_spike_trading_days_mapped": len(spike_trading_idxs),
        "horizons": {},
    }
    for h in horizons:
        sample, baseline = [], []
        for i in all_trading_idxs:
            r = forward_return(closes, i, h)
            if r is None:
                continue
            (sample if i in spike_trading_idxs else baseline).append(r)
        out["horizons"][h] = {
            "forward_return": wag2.welch_vs_baseline(sample, baseline),
            "_raw": {"sample": sample, "baseline": baseline},
        }
    return out


def pool_forward_return(per_ticker: dict, tickers: Sequence[str], horizons: Sequence[int]) -> dict:
    """Pools _raw spike/baseline forward-return samples across `tickers`
    and reruns welch_vs_baseline on the pooled sample per horizon — the
    single pre-registered composite test per group (REASONING STANDARD #4),
    same pooling-treats-ticker-days-as-independent caveat wikiattention_
    gate2.pool_group already documents for this repo's other cross-ticker
    gate-2 scripts."""
    out = {}
    for h in horizons:
        s, b = [], []
        for t in tickers:
            row = per_ticker.get(t, {}).get("horizons", {}).get(h)
            if not row or "_raw" not in row:
                continue
            s.extend(row["_raw"]["sample"])
            b.extend(row["_raw"]["baseline"])
        out[h] = {
            "forward_return": wag2.welch_vs_baseline(s, b),
            "n_tickers_pooled": sum(1 for t in tickers if t in per_ticker and "horizons" in per_ticker[t]),
        }
    return out


# ── Network ──────────────────────────────────────────────────────────────────

def fetch_json(url: str, timeout: int = 60, retries: int = 3) -> dict:
    """NHTSA's complaintsByVehicle returns HTTP 400 (not 200) for a query
    that legitimately matches zero complaints, with a well-formed JSON body
    ({"count":0,"results":[]}) — see module docstring finding #1. This
    parses the error body as JSON before giving up, so a genuine empty
    result is preserved rather than treated as a fetch failure."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-datacore-gate2/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read().decode("utf-8"))
            except Exception:
                last_err = e
        except Exception as e:
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def fetch_complaints_live(make: str, model: str, model_year: int) -> list[dict]:
    url = (f"{COMPLAINTS_API}?make={urllib.parse.quote(make)}"
           f"&model={urllib.parse.quote(model)}&modelYear={model_year}")
    return fetch_json(url).get("results", []) or []


def fetch_ticker_complaint_dates(vehicles: Sequence[dict], spacing_s: float) -> list[date]:
    """One ticker's complaint dates, pooled across every curated
    make/model/year row that shares this ticker (BUILD ORDER 6's own
    ticker-mapping convention — a company's total watched-fleet footprint,
    not any single model in isolation)."""
    out: list[date] = []
    for i, v in enumerate(vehicles):
        if i > 0:
            time.sleep(spacing_s)
        try:
            rows = fetch_complaints_live(v["make"], v["model"], v["modelYear"])
        except Exception as e:
            print(f"  [warn] {v['ticker']} {v['make']}/{v['model']}/{v['modelYear']}: {e}", file=sys.stderr)
            continue
        for r in rows:
            d = g1.parse_complaint_date(r.get("dateComplaintFiled"))
            if d is not None:
                out.append(d)
    return out


def run_gate2(tickers: Optional[Sequence[str]] = None, days: int = 900,
              threshold: float = DEFAULT_Z_THRESHOLD, window: int = DEFAULT_TRAILING_WINDOW,
              horizons: Sequence[int] = DEFAULT_HORIZONS, spacing_s: float = 0.8) -> dict:
    sys.path.insert(0, os.path.dirname(_HERE))
    import backtest_v2 as bt

    by_ticker = group_by_ticker(load_watchlist())
    target_tickers = list(tickers) if tickers else sorted(by_ticker.keys())

    per_ticker = {}
    for t in target_tickers:
        vehicles = by_ticker.get(t, [])
        if not vehicles:
            per_ticker[t] = {"error": "ticker not in watchlist"}
            continue
        try:
            bars = bt.fetch_bars(t, days)
            if not bars or not bars.get("date"):
                raise RuntimeError("empty bars")
        except Exception as e:
            per_ticker[t] = {"error": f"price fetch failed: {e}"}
            continue
        trading_dates = bars["date"]
        complaint_dates = fetch_ticker_complaint_dates(vehicles, spacing_s)
        start = datetime.strptime(trading_dates[0], "%Y-%m-%d").date()
        end = datetime.strptime(trading_dates[-1], "%Y-%m-%d").date()
        counts = aggregate_daily_counts([d for d in complaint_dates if start <= d <= end])
        cal_dates, cal_series = build_calendar_daily_series(counts, start, end)
        row = evaluate_ticker(cal_dates, cal_series, trading_dates, bars["close"], threshold, window, horizons)
        row["n_complaints_total_in_window"] = sum(cal_series)
        row["n_complaints_fetched_all_time"] = len(complaint_dates)
        per_ticker[t] = row

    pooled = {
        "primary_edge_group": pool_forward_return(per_ticker, PRIMARY_GROUP_TICKERS, horizons),
        "secondary_mega_comparison": pool_forward_return(per_ticker, SECONDARY_GROUP_TICKERS, horizons),
    }

    per_ticker_summary = {}
    for t, row in per_ticker.items():
        if "horizons" not in row:
            per_ticker_summary[t] = row
            continue
        trimmed = dict(row)
        trimmed["horizons"] = {h: {k: v for k, v in hrow.items() if k != "_raw"} for h, hrow in row["horizons"].items()}
        per_ticker_summary[t] = trimmed

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "days": days, "threshold": threshold, "trailing_window_calendar_days": window,
        "horizons": list(horizons),
        "primary_group": list(PRIMARY_GROUP_TICKERS), "secondary_group": list(SECONDARY_GROUP_TICKERS),
        "pooled": pooled,
        "per_ticker": per_ticker_summary,
    }


ALPHA = 0.05
BONFERRONI_N = 2 * len(DEFAULT_HORIZONS)  # 2 groups x 3 horizons


def gate2_verdict(result: dict) -> dict:
    bar = ALPHA / BONFERRONI_N
    primary = result["pooled"]["primary_edge_group"]
    hits = []
    for h in result["horizons"]:
        fr = primary.get(h, {}).get("forward_return")
        if fr and fr["p_value"] < bar and fr["mean_diff"] < 0:
            hits.append({"horizon": h, **fr})
    return {"bonferroni_bar": round(bar, 5), "passing_horizons": hits, "PASS": len(hits) > 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default=None, help="comma-separated; default = full watchlist")
    ap.add_argument("--days", type=int, default=900)
    ap.add_argument("--threshold", type=float, default=DEFAULT_Z_THRESHOLD)
    ap.add_argument("--window", type=int, default=DEFAULT_TRAILING_WINDOW)
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument("--spacing", type=float, default=0.8)
    args = ap.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None
    horizons = [int(h) for h in args.horizons.split(",")]
    result = run_gate2(tickers, args.days, args.threshold, args.window, horizons, args.spacing)
    verdict = gate2_verdict(result)
    result["verdict"] = verdict
    print(json.dumps(result, indent=2, default=str))
    print(f"\n== GATE 2 VERDICT: {'PASS' if verdict['PASS'] else 'NOT PASSED'} "
          f"(bar={verdict['bonferroni_bar']}, {len(verdict['passing_horizons'])} horizon(s) hit) ==")
    return 0 if verdict["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
