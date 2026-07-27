#!/usr/bin/env python3
"""
scripts/usaspending_gate2.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for
the USAspending contract-award archive (server/usaSpending.ts). Gate 1
(recipient->ticker mapping precision) PASSED 2026-07-24 — see
research/open_questions.md item 4 under "USAspending contracts". Gate 2's
concrete blocker (no way to read the historical archive from outside the
Railway container) was closed 2026-07-26 by the generic `/api/diag/archive`
probe (server/datacoreArchive.ts's `readArchiveDay`) — this script is the
first real consumer of that probe.

QUESTION (PRIOR restated, written before running — REASONING STANDARD #10):
among SMALL-CAP recipients of a civilian-agency federal contract award, does
a LARGE award/market-cap ratio predict better forward 5-20 trading-day
returns than a small award/market-cap ratio, on that same ticker's own
unconditional baseline? PRIOR: positive, because a $1-25M award is immaterial
information to a $50B company but material to a $200M one, and large funds
are capacity-constrained out of thin small-cap names (EDGE DOCTRINE #2 —
"fish where whales can't") — the mechanism is nobody-can-size-in, not
nobody-noticed (REASONING STANDARD #5). Expect ~zero separation once mcap
is large enough that the award is immaterial. Kill if the small-cap,
high-ratio bucket shows no separation from its own ticker-baseline.

WHY CIVILIAN-ONLY (not a new finding — restates the archive's own documented
constraint): `datacore/manifests/usaspending.json`'s `confidence_model`
established (2026-07-05) that DoD/USACE awards are only PUBLISHED roughly
90 days after `ad` (action_date) — `rt` (the day we fetched it) is the only
honest "knowable on day X" event date, and even `rt` is contaminated for
DoD because the government itself sat on the disclosure that long. This
script filters `ag == "Department of Defense"` out entirely rather than
cohorting it, since the archive is currently too young (started 2026-07-05)
to have a DoD sub-sample worth a separate bucket.

WHY market-cap is an APPROXIMATION, stated honestly (not silently): no
market-cap data source with a working API key is available in this session
(Polygon/Finnhub keys absent; yfinance's `.info`/`.fast_info` fail in this
sandbox's network path — verified live, curl_cffi TLS reset through the
proxy). Market cap here = (SEC EDGAR's most recently REPORTED
dei:EntityCommonStockSharesOutstanding or us-gaap:CommonStockSharesOutstanding,
keyless, data.sec.gov) x (the no-lookahead entry price, same bar used for
the return calculation). Shares outstanding is not point-in-time to the
award date — it is whatever SEC's most recent filing reports as of when
this script runs, which changes far more slowly than price, so this is a
reasonable approximation for an exploratory SIGNAL-gate screen, not
something to trade on directly. A ticker with no resolvable CIK or no
outstanding-shares tag (e.g. a non-reporting/dormant entity — see gate 1's
CAVEAT 2 on SPCX) is silently excluded from the market-cap-dependent
comparison, which is the correct behavior (no fabricated mcap), not a bug.

SMALL-CAP UNIVERSE: mcap < SMALL_CAP_MAX_MCAP ($2B — a standard small-cap
ceiling; stricter than this session's exact wording would need a documented
follow-up re-run at a tighter micro-cap cutoff, filed as NEXT below).

BUCKETING: within the small-cap, mcap-resolved population, the median
award/mcap ratio for THIS RUN splits high vs. low — a data-adaptive median
split (not a fixed absolute threshold guessed in advance), documented as a
specific test rather than a search over thresholds (REASONING STANDARD #4).

NO LOOKAHEAD (REASONING STANDARD #7): entry = first trading day strictly
after `rt` (the as-seen date), matching the archive's own confidence_model
guidance ("'knowable on day X' = filter rt <= X, never ad <= X").

Reuses find_entry_index / compute_forward_returns / compute_ticker_baseline
unchanged from form4_gate2_test.py (COMPILE KNOWLEDGE INTO CODE — these
three are already generic over any event list keyed by "filing_date" plus a
bars dict; no USAspending-specific logic lives in them). Bucketing and
significance testing are written fresh here because form4_gate2_test.py's
summarize()/significance() hardcode the "cluster"/"single" bucket names,
which would be a misleading label for a ratio-median split.

Usage: python3 scripts/usaspending_gate2.py [--out usaspending_gate2_results.json]
Requires env DIAG_TOKEN (matches every other /api/diag/* consumer in this
repo) and reads the archive from VOLTRADE_BASE_URL (default
https://voltradeai.com).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta

from scipy.stats import ttest_ind

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from form4_gate2_test import find_entry_index  # noqa: E402  (repo-root module,
# see sys.path.insert above; find_entry_index takes no dependency on
# form4_gate2_test's own HORIZONS global so it is safe to reuse as-is —
# compute_forward_returns/compute_ticker_baseline are NOT reused: both read
# form4_gate2_test's module-level HORIZONS=(20,60) at call time rather than
# taking it as a parameter, so importing them here would silently compute
# the wrong horizons (20/60 instead of this screen's 5/20) — rewritten
# locally below, parametrized on THIS module's HORIZONS explicitly.)

HORIZONS = (5, 20)  # trading days, per the ladder's own gate-2 definition
SMALL_CAP_MAX_MCAP = 2_000_000_000.0  # $2B — see module docstring
SEC_USER_AGENT = "VolTradeAI research research@voltradeai.com"
DEFAULT_BASE_URL = "https://voltradeai.com"
CIVILIAN_EXCLUDE_AGENCY = "Department of Defense"
ALLOWED_MATCH_METHODS = ("name", "cache")  # excludes "parent" — see gate-1's
# CAVEAT 1 (BALL CORPORATION / BAE Systems parent-match residual risk),
# not re-investigated here; a plausibility check on mm=="parent" is its own
# future task, not silently trusted in this run.


def _http_get_json(url: str, headers: dict, timeout: int = 30, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:  # transient network / rate limit
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}: {last_err}")


def fetch_archive_rows(start_day: str, end_day: str, base_url: str, token: str) -> tuple[list[dict], dict]:
    """Pulls every day's usaspending archive rows in [start_day, end_day]
    (inclusive) via the generic /api/diag/archive probe. Returns (rows,
    day_status) — day_status records per-day row counts and any truncation
    so a truncated/missing day is visible in the result, never silent."""
    rows: list[dict] = []
    day_status: dict[str, dict] = {}
    d = datetime.strptime(start_day, "%Y-%m-%d").date()
    end = datetime.strptime(end_day, "%Y-%m-%d").date()
    while d <= end:
        day_str = d.isoformat()
        url = f"{base_url}/api/diag/archive?stream=usaspending&day={day_str}&limit=5000&token={token}"
        try:
            resp = _http_get_json(url, headers={})
        except Exception as e:
            day_status[day_str] = {"status": "error", "reason": str(e)[:200]}
            d += timedelta(days=1)
            continue
        day_rows = resp.get("rows") or []
        rows.extend(day_rows)
        day_status[day_str] = {"status": "ok", "count": len(day_rows), "truncated": bool(resp.get("truncated"))}
        d += timedelta(days=1)
    return rows, day_status


def build_events(rows: list[dict]) -> dict[str, list[dict]]:
    """Filters to civilian-agency, ticker-matched (name/cache method only)
    rows, dedupes by (aid, mod, amt) keeping the EARLIEST `rt` we observed
    it at (honest "as-seen" event date — a later re-poll of the same
    unmodified action is not a new event), and aggregates same-ticker
    same-day awards into one event (multiple small contracts to the same
    company on the same day are one economic event for return-attribution
    purposes, same collapsing logic as form4_gate2_test.py's same-day
    multi-owner filings)."""
    seen: set[tuple] = set()  # (aid, mod, amt) already recorded — first sighting wins
    by_ticker_day: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.get("ag") == CIVILIAN_EXCLUDE_AGENCY:
            continue
        tkr = r.get("tkr")
        if not tkr or r.get("mm") not in ALLOWED_MATCH_METHODS:
            continue
        amt = r.get("amt")
        rt = r.get("rt")
        if amt is None or not rt:
            continue
        key = (r.get("aid"), r.get("mod"), amt)
        if key in seen:
            continue  # a later re-poll of the same unmodified action, not a new event
        seen.add(key)
        by_ticker_day.setdefault(tkr, {})
        by_ticker_day[tkr][rt] = by_ticker_day[tkr].get(rt, 0.0) + float(amt)

    out: dict[str, list[dict]] = {}
    for tkr, by_day in by_ticker_day.items():
        events = [{"filing_date": d, "amt": amt} for d, amt in sorted(by_day.items())]
        out[tkr] = events
    return out


def compute_forward_returns(events: list[dict], bars: dict) -> list[dict]:
    """Same logic as form4_gate2_test.py's function of the same name, NOT
    imported from there (see the sys.path.insert comment above) because
    that version closes over form4_gate2_test's own module-level
    HORIZONS=(20,60) rather than taking horizons as a parameter — this copy
    uses THIS module's HORIZONS=(5,20). Pure function: for each event,
    finds its no-lookahead entry and computes forward N-day returns.
    Events too close to the end of `bars` for a horizon are dropped for
    that horizon (right-censoring honesty, never zero-filled)."""
    bar_dates = bars["date"]
    bar_closes = bars["close"]
    out = []
    for ev in events:
        entry_idx = find_entry_index(bar_dates, ev["filing_date"])
        row = dict(ev)
        row["entry_date"] = bar_dates[entry_idx] if entry_idx is not None else None
        row["entry_idx"] = entry_idx
        row["forward_returns"] = {}
        if entry_idx is not None:
            entry_price = bar_closes[entry_idx]
            for h in HORIZONS:
                exit_idx = entry_idx + h
                if exit_idx < len(bar_closes) and entry_price:
                    row["forward_returns"][h] = bar_closes[exit_idx] / entry_price - 1
        out.append(row)
    return out


def compute_ticker_baseline(bars: dict, event_entry_indices: list[int]) -> dict[int, list[float]]:
    """Same logic as form4_gate2_test.py's function of the same name (see
    compute_forward_returns's docstring for why it is copied, not
    imported). Per-horizon forward returns from EVERY trading day in
    `bars` as an entry point, EXCLUDING any entry within one horizon's
    reach of one of this ticker's own signal events (contaminated-baseline
    guard) — this is the ticker's own unconditional baseline, not a
    market-wide average."""
    bar_closes = bars["close"]
    n = len(bar_closes)
    out: dict[int, list[float]] = {h: [] for h in HORIZONS}
    for h in HORIZONS:
        excluded = set()
        for e in event_entry_indices:
            for i in range(max(0, e - h), min(n, e + h + 1)):
                excluded.add(i)
        for i in range(n - h):
            if i in excluded or not bar_closes[i]:
                continue
            out[h].append(bar_closes[i + h] / bar_closes[i] - 1)
    return out


def ticker_to_cik_map() -> dict[str, str]:
    """SEC's own public ticker->CIK reference file (same file
    server/usaSpending.ts's buildTickerMap already trusts for the reverse
    name->ticker direction — reused here for a different lookup, not
    duplicated matching logic)."""
    raw = _http_get_json(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": SEC_USER_AGENT},
    )
    out: dict[str, str] = {}
    for row in raw.values():
        t = row.get("ticker")
        cik = row.get("cik_str")
        if t and cik:
            out[t.upper()] = str(cik).zfill(10)
    return out


_SHARES_CACHE: dict[str, float | None] = {}


def shares_outstanding(cik: str) -> float | None:
    """Most recently REPORTED shares outstanding for `cik` (10-digit,
    zero-padded), trying the dei tag first (broadest coverage) then the
    us-gaap tag. Returns None (never fabricated) if neither tag exists —
    this is the correct, honest outcome for a non-reporting/dormant entity,
    not an error to retry."""
    if cik in _SHARES_CACHE:
        return _SHARES_CACHE[cik]
    result = None
    for tag_ns, tag in (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")):
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{tag_ns}/{tag}.json"
        try:
            raw = _http_get_json(url, headers={"User-Agent": SEC_USER_AGENT}, retries=1)
        except Exception:
            continue
        units = raw.get("units", {}).get("shares", [])
        if units:
            latest = max(units, key=lambda u: u.get("end", ""))
            val = latest.get("val")
            if val:
                result = float(val)
                break
    _SHARES_CACHE[cik] = result
    return result


def summarize_by_bucket(rows: list[dict], baseline: dict[int, list[float]]) -> dict:
    summary = {}
    for h in HORIZONS:
        vals = {
            "baseline": baseline.get(h, []),
            "high_ratio": [r["forward_returns"][h] for r in rows if r["high_ratio"] and h in r["forward_returns"]],
            "low_ratio": [r["forward_returns"][h] for r in rows if not r["high_ratio"] and h in r["forward_returns"]],
        }
        summary[str(h)] = {
            bucket: {"n": len(v), "mean_pct": round(sum(v) / len(v) * 100, 3) if v else None}
            for bucket, v in vals.items()
        }
    return summary


def significance_by_bucket(rows: list[dict], baseline: dict[int, list[float]]) -> dict:
    """Welch two-sample t-test, bucket vs. the pooled ticker-baseline —
    same design as form4_gate2_test.py's significance(), parametrized over
    the "high_ratio" boolean instead of "cluster". Returns None (never
    fabricated) when either sample has fewer than 5 observations."""
    out: dict[str, dict] = {}
    for h in HORIZONS:
        base = baseline.get(h, [])
        out[str(h)] = {}
        for bucket in ("high_ratio", "low_ratio"):
            sample = [r["forward_returns"][h] for r in rows if (r["high_ratio"] == (bucket == "high_ratio")) and h in r["forward_returns"]]
            if len(sample) < 5 or len(base) < 5:
                out[str(h)][bucket] = None
                continue
            t_stat, p_value = ttest_ind(sample, base, equal_var=False)
            out[str(h)][bucket] = {
                "n": len(sample),
                "n_baseline": len(base),
                "mean_diff_pct": round((sum(sample) / len(sample) - sum(base) / len(base)) * 100, 3),
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(p_value), 4),
            }
    return out


def run(start_day: str, end_day: str, base_url: str, token: str, fetch_bars_fn) -> dict:
    rows, day_status = fetch_archive_rows(start_day, end_day, base_url, token)
    events_by_ticker = build_events(rows)
    total_civ_tkr_events = sum(len(v) for v in events_by_ticker.values())
    if not events_by_ticker:
        return {"status": "no_events", "day_status": day_status}

    cik_map = ticker_to_cik_map()

    all_rows: list[dict] = []
    baseline_pooled: dict[int, list[float]] = {h: [] for h in HORIZONS}
    per_ticker_status: dict[str, dict] = {}
    no_mcap = 0
    large_cap_excluded = 0

    for ticker, events in events_by_ticker.items():
        earliest = min(e["filing_date"] for e in events)
        days_needed = (datetime.utcnow() - datetime.strptime(earliest, "%Y-%m-%d")).days + max(HORIZONS) + 30
        try:
            bars = fetch_bars_fn(ticker, days_needed)
        except Exception as e:
            per_ticker_status[ticker] = {"status": "error", "reason": str(e)[:200]}
            continue
        if not bars or not bars.get("date"):
            per_ticker_status[ticker] = {"status": "no_price_data"}
            continue

        computed = compute_forward_returns(events, bars)
        cik = cik_map.get(ticker)
        so = shares_outstanding(cik) if cik else None

        kept_rows = []
        for r in computed:
            if r["entry_idx"] is None:
                continue
            entry_price = bars["close"][r["entry_idx"]]
            if not so or not entry_price:
                no_mcap += 1
                continue
            mcap = so * entry_price
            if mcap >= SMALL_CAP_MAX_MCAP:
                large_cap_excluded += 1
                continue
            r["mcap_approx"] = round(mcap, 0)
            r["ratio"] = r["amt"] / mcap
            kept_rows.append(r)

        if not kept_rows:
            per_ticker_status[ticker] = {"status": "no_small_cap_events", "events": len(events)}
            continue

        entry_indices = [r["entry_idx"] for r in kept_rows]
        ticker_baseline = compute_ticker_baseline(bars, entry_indices)
        for h in HORIZONS:
            baseline_pooled[h].extend(ticker_baseline[h])
        all_rows.extend(kept_rows)
        per_ticker_status[ticker] = {"status": "ok", "events": len(events), "small_cap_events": len(kept_rows)}

    if not all_rows:
        return {
            "status": "no_small_cap_events_with_mcap",
            "day_status": day_status,
            "total_civ_tkr_events": total_civ_tkr_events,
            "tickers_in_archive": len(events_by_ticker),
            "no_mcap": no_mcap,
            "large_cap_excluded": large_cap_excluded,
        }

    ratios = sorted(r["ratio"] for r in all_rows)
    median_ratio = ratios[len(ratios) // 2]
    for r in all_rows:
        r["high_ratio"] = r["ratio"] >= median_ratio

    return {
        "status": "ok",
        "day_status": day_status,
        "total_civ_tkr_events": total_civ_tkr_events,
        "tickers_in_archive": len(events_by_ticker),
        "no_mcap": no_mcap,
        "large_cap_excluded": large_cap_excluded,
        "small_cap_events_tested": len(all_rows),
        "median_ratio": median_ratio,
        "summary": summarize_by_bucket(all_rows, baseline_pooled),
        "significance": significance_by_bucket(all_rows, baseline_pooled),
        "per_ticker_status": per_ticker_status,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-day", default="2026-07-05", help="archive start (datacore/manifests/usaspending.json 'started')")
    ap.add_argument("--end-day", default=None, help="default: yesterday (today's archive day is likely partial)")
    ap.add_argument("--base-url", default=os.environ.get("VOLTRADE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--out", default="usaspending_gate2_results.json")
    args = ap.parse_args()

    token = os.environ.get("DIAG_TOKEN")
    if not token:
        raise SystemExit("DIAG_TOKEN not set in environment")

    end_day = args.end_day or (date.today() - timedelta(days=1)).isoformat()

    from backtest_v2 import fetch_bars  # local import: keeps this script's
    # unit tests free of backtest_v2's network/cache side effects

    result = run(args.start_day, end_day, args.base_url, token, fetch_bars)
    print(json.dumps(result, indent=2, default=str))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
