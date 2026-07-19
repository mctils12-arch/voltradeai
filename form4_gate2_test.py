#!/usr/bin/env python3
"""
form4_gate2_test.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for the SEC
bulk-historical Form 4 insider-transaction archive (sec_form4_bulk.py, gate 1
DATA passed 2026-07-19, see research/open_questions.md's "Insider Form 4
clustering as a signal").

Question (PRIOR restated from open_questions.md, written before this run):
do clusters of open-market insider BUYS (transaction code P) at a given
issuer, within a short window, predict forward N/20/60-day excess return
over a same-universe random-entry baseline? Expect a small positive edge
concentrated in officer/director buys on small/mid caps (EDGE DOCTRINE #2 —
capacity-constrained corners), close to zero on names already heavily
covered. Kill if officer/director buys (clustered OR single) show no
separation from the ticker-matched baseline.

REASONING STANDARD #3 (demand base rates): the baseline is NOT a market-wide
average — for each ticker actually tested, it is that SAME ticker's own
unconditional forward-return distribution sampled at every other available
entry day, EXCLUDING any day within one horizon of one of that ticker's own
signal events (a contaminated-baseline guard: without this, the event's own
price impact would leak into its "neutral" comparison group). This isolates
"does an insider buy predict a better-than-usual day for THIS stock", not
"do small caps go up more than the market".

REASONING STANDARD #4 (discount for multiple comparisons): 2 buckets
(cluster/single) x 2 horizons = 4 comparisons per run; a screen, not a
conclusion.

Independence note (differs from cot_gate2_test.py's HAC/Newey-West
treatment): COT's autocorrelation problem comes from ONE symbol sampled
weekly with a 12-week-wide horizon — consecutive observations overlap ~11/12.
Here, events are spread across many DIFFERENT tickers (cross-sectional), so
neighboring observations in time are usually unrelated companies — a much
weaker dependence structure. A plain Welch two-sample t-test (bucket vs. its
ticker-matched baseline) is used instead of a HAC correction; residual
dependence only arises when the SAME ticker has 2+ events within one
horizon's reach of each other, which the baseline-contamination guard above
already partially addresses on the baseline side. Not zero risk — stated
honestly, not swept under a fancier-looking test that assumes a different
problem than the one this data actually has.

No lookahead (REASONING STANDARD #7): entry is the first trading day
STRICTLY AFTER the Form 4 FILING_DATE (the day the trade became public via
EDGAR), never the transaction date itself (insiders have up to 2 business
days to file — the market cannot react before disclosure).

SAMPLE CAP, stated honestly (not silent): fetching daily price bars for
every distinct ticker in the archive does not scale to a single interactive
session (a single closed quarter alone touches ~800-900 distinct officer/
director P-buy tickers). This run caps at MAX_TICKERS tickers, ALL
clustered-event tickers first (materially fewer, and the primary
hypothesis), then single-event tickers filling any remaining budget sorted
by total dollar value descending (economically largest, most legible buys
first) for reproducibility. Every run logs exactly how many tickers were
in-archive vs. actually fetched. The archive itself is NOT capped — the
sec_form4_bulk.py pipeline keeps backfilling every quarter via the bot's
Tier 3 hourly call; a future session can re-run this screen against a wider
fetched sample as that accumulates, exactly like the COT gate-2 precedent
re-running against fresh weeks.

Usage: python3 form4_gate2_test.py [--quarters 2025q4,2026q1] [--max-tickers 150] [--out form4_gate2_results.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta

from scipy.stats import ttest_ind

from sec_form4_bulk import load_all_records

HORIZONS = (20, 60)  # trading days
CLUSTER_WINDOW_DAYS = 5  # calendar days — "short window" per the stated prior
DEFAULT_MAX_TICKERS = 150


def find_entry_index(bar_dates: list[str], filing_date: str) -> int | None:
    """First bar strictly after `filing_date` (no lookahead)."""
    for i, d in enumerate(bar_dates):
        if d > filing_date:
            return i
    return None


def build_events(records: list[dict], trans_code: str = "P") -> dict[str, list[dict]]:
    """Group officer/director records of `trans_code` by ticker into
    chronological per-ticker event lists. One event per (ticker, filing_date)
    — same-day filings from multiple owners collapse into one event with a
    deduplicated owner set (Form 4's 2-business-day filing window means
    same-day multi-owner filings are already effectively simultaneous
    disclosure). Flags `cluster=True` when >=2 DISTINCT owners bought this
    ticker within CLUSTER_WINDOW_DAYS ending at this event's own date (a
    causal, backward-looking window — never uses a later event to flag an
    earlier one)."""
    by_ticker: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        if r.get("trans_code") != trans_code:
            continue
        ticker = r["ticker"]
        d = r["filing_date"]
        ev = by_ticker[ticker].setdefault(d, {
            "ticker": ticker, "filing_date": d, "owners": set(), "dollar_value": 0.0,
        })
        ev["owners"].add(r["owner_cik"])
        ev["dollar_value"] += r["dollar_value"]

    out: dict[str, list[dict]] = {}
    for ticker, by_date in by_ticker.items():
        dates_sorted = sorted(by_date.keys())
        events = []
        for d in dates_sorted:
            window_start = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=CLUSTER_WINDOW_DAYS)).strftime("%Y-%m-%d")
            owners_in_window: set[str] = set()
            for d2 in dates_sorted:
                if window_start <= d2 <= d:
                    owners_in_window |= by_date[d2]["owners"]
                if d2 > d:
                    break
            ev = by_date[d]
            events.append({
                "ticker": ticker,
                "filing_date": d,
                "dollar_value": round(ev["dollar_value"], 2),
                "n_owners_same_day": len(ev["owners"]),
                "cluster": len(owners_in_window) >= 2,
            })
        out[ticker] = events
    return out


def select_tickers(events_by_ticker: dict[str, list[dict]], max_tickers: int) -> list[str]:
    """Balanced ~50/50 split between clustered-event tickers and
    single-only tickers, each ranked by total dollar value descending,
    with unused share reallocated to the other bucket. Deterministic —
    same archive, same selection, every run.

    FIXED 2026-07-19 (found live this session): an earlier "cluster first,
    fill remainder with single" priority order silently starved the
    "single" bucket to ZERO whenever cluster-eligible tickers alone
    exceeded max_tickers (confirmed live: 1180 cluster tickers > a 500
    cap) — every event in the reported "single" bucket was then actually a
    non-cluster event belonging to a ticker that ALSO had a cluster
    elsewhere, not a clean isolated-buyer population, silently biasing
    that bucket's composition. A straight priority order looks correct in
    a small sample (where neither bucket saturates the cap) and only
    breaks once the archive is big enough to expose it — exactly the kind
    of latent selection bug this repo's REASONING STANDARD #4 exists to
    catch before trusting a result."""
    cluster_all = sorted(
        (t for t, evs in events_by_ticker.items() if any(e["cluster"] for e in evs)),
        key=lambda t: -sum(e["dollar_value"] for e in events_by_ticker[t] if e["cluster"]),
    )
    single_all = sorted(
        (t for t in events_by_ticker if t not in set(cluster_all)),
        key=lambda t: -sum(e["dollar_value"] for e in events_by_ticker[t]),
    )
    half = (max_tickers + 1) // 2  # ceiling — the primary hypothesis (cluster) gets the odd slot
    n_cluster = min(len(cluster_all), half)
    n_single = min(len(single_all), max_tickers - n_cluster)
    n_cluster = min(len(cluster_all), max_tickers - n_single)
    return cluster_all[:n_cluster] + single_all[:n_single]


def compute_forward_returns(events: list[dict], bars: dict) -> list[dict]:
    """Pure function: for each event, find its no-lookahead entry and
    compute forward N-day returns. Events too close to the end of `bars`
    for a horizon are dropped for that horizon (right-censoring honesty,
    never zero-filled)."""
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
    """Per-horizon forward returns from EVERY trading day in `bars` as an
    entry point, EXCLUDING any entry within one horizon's reach of one of
    this ticker's own signal events (contaminated-baseline guard — see
    module docstring). This is the ticker's own unconditional baseline, not
    a market-wide average."""
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


def summarize(rows: list[dict], baseline: dict[int, list[float]]) -> dict:
    summary = {}
    for h in HORIZONS:
        vals = {
            "baseline": baseline.get(h, []),
            "cluster": [r["forward_returns"][h] for r in rows if r["cluster"] and h in r["forward_returns"]],
            "single": [r["forward_returns"][h] for r in rows if not r["cluster"] and h in r["forward_returns"]],
        }
        summary[str(h)] = {
            bucket: {"n": len(v), "mean_pct": round(sum(v) / len(v) * 100, 3) if v else None}
            for bucket, v in vals.items()
        }
    return summary


def significance(rows: list[dict], baseline: dict[int, list[float]]) -> dict:
    """Welch two-sample t-test, bucket vs. that ticker-set's own baseline
    (see module docstring for why this replaces cot_gate2_test.py's HAC
    approach here). Returns None (never fabricated) when either sample is
    too thin to test."""
    out: dict[str, dict] = {}
    for h in HORIZONS:
        base = baseline.get(h, [])
        out[str(h)] = {}
        for bucket in ("cluster", "single"):
            sample = [r["forward_returns"][h] for r in rows if (r["cluster"] == (bucket == "cluster")) and h in r["forward_returns"]]
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


def run(quarters: list[str] | None, max_tickers: int, fetch_bars_fn, trans_code: str = "P") -> dict:
    records = load_all_records(quarters)
    events_by_ticker = build_events(records, trans_code=trans_code)
    if not events_by_ticker:
        return {"status": "no_events", "trans_code": trans_code}

    tickers_in_archive = len(events_by_ticker)
    selected = select_tickers(events_by_ticker, max_tickers)

    all_rows: list[dict] = []
    baseline_pooled: dict[int, list[float]] = {h: [] for h in HORIZONS}
    per_ticker_status = {}
    for ticker in selected:
        events = events_by_ticker[ticker]
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

        rows = compute_forward_returns(events, bars)
        entry_indices = [r["entry_idx"] for r in rows if r["entry_idx"] is not None]
        ticker_baseline = compute_ticker_baseline(bars, entry_indices)
        for h in HORIZONS:
            baseline_pooled[h].extend(ticker_baseline[h])
        all_rows.extend(rows)
        per_ticker_status[ticker] = {"status": "ok", "events": len(events)}

    return {
        "status": "ok",
        "trans_code": trans_code,
        "tickers_in_archive": tickers_in_archive,
        "tickers_selected": len(selected),
        "tickers_fetched_ok": sum(1 for v in per_ticker_status.values() if v["status"] == "ok"),
        "total_events": len(all_rows),
        "cluster_events": sum(1 for r in all_rows if r["cluster"]),
        "summary": summarize(all_rows, baseline_pooled),
        "significance": significance(all_rows, baseline_pooled),
        "per_ticker_status": per_ticker_status,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarters", default=None, help="comma-separated quarters, e.g. 2025q4,2026q1 (default: all archived)")
    ap.add_argument("--max-tickers", type=int, default=DEFAULT_MAX_TICKERS)
    ap.add_argument("--trans-code", default="P", choices=["P", "S"])
    ap.add_argument("--out", default="form4_gate2_results.json")
    args = ap.parse_args()

    from backtest_v2 import fetch_bars  # local import: keeps this script's
    # unit tests free of backtest_v2's network/cache side effects

    quarters = args.quarters.split(",") if args.quarters else None
    result = run(quarters, args.max_tickers, fetch_bars, trans_code=args.trans_code)
    print(json.dumps(result, indent=2, default=str))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
