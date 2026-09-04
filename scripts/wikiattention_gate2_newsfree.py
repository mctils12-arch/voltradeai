#!/usr/bin/env python3
"""
scripts/wikiattention_gate2_newsfree.py — the "attention without same-day
8-K" control for the wikimedia_pageviews_attention root, queued as NEXT by
the 2026-09-04 wikiattention_gate2.py session (research/open_questions.md).

THAT SESSION'S FINDING: a pageview attention spike is followed by
significantly elevated forward volume (small/mid-cap group: +42.6/25.5/18.1
pp over baseline at 1/3/5d, p<0.0001, survives a Bonferroni bar across 10
comparisons). It withheld promotion to gate2_pass because the result is
UNCONTROLLED for the confound the root's own header names as the interesting
case ("attention without news"): a same-day earnings release, 8-K, or other
corporate news event could drive BOTH the pageview spike and the volume
spike independently, producing this identical signature with pageviews
leading nothing at all.

THIS SCRIPT is the control that session named concretely buildable but did
not attempt (REASONING STANDARD #4 — one diagnostic design per pass):
re-run the exact same pooled test, but restrict the "spike" sample to spike
days with NO 8-K of any kind filed on the spike day itself or the trading
day before it (the ANY-8-K bar named in that session's NEXT note, broader
than Item-2.02-only — a company-news event that isn't an earnings release
can still be the confound). The non-spike BASELINE is unchanged from the
original script (every day that isn't a z>=threshold spike, news or not) —
only the spike/sample side is filtered, so this stays a direct extension of
the original design, not a redefinition of it.

VERDICT RULE (stated before computing anything, REASONING STANDARD #10):
if the small/mid-cap pooled volume_ratio effect survives on the news-free
spike subset at the Bonferroni bar this root's own gate-2 session already
established (alpha/10 = 0.005 across the same 10-cell family), that is the
clean gate-2 pass the prior session could not claim — promote
wikimedia_pageviews_attention to gate2_pass. If it collapses (loses
significance, or the point estimate falls to a small fraction of the
unfiltered effect), the earlier result was substantially news-coincidence
and the root should be marked KILLED at gate 2 for this exact spec, not
left ambiguous. PRIOR (informal, not a formal probability — the confound
mechanism is plausible but so is a genuine independent lead-lag effect):
no strong prior either way: recorded honestly as "informative but genuinely
uncertain" rather than back-filled after seeing the result below.

DATA SOURCES (three independent fetches, same "degrade honestly, don't
skip everything" discipline as wikiattention_gate2.py):
  - Pageviews: wikimedia.org, as in wikiattention_gate2.py.
  - Price/volume: backtest_v2.fetch_bars(), as in wikiattention_gate2.py.
  - 8-K filing dates + item codes: data.sec.gov/submissions/CIK##########
    .json ("filings.recent.form"/"filingDate"/"items" arrays) — the SAME
    ground truth wikiattention_gate1.ts hand-recorded from for 11 tickers;
    this script fetches it live for the full 23-ticker seed set instead.
    Ticker -> CIK resolved from sec.gov/files/company_tickers.json (the
    same map sec8kEarnings.ts's getCikTickerMap uses in the other
    direction, CIK -> ticker). Verified live this session that AAPL's
    "recent" filings window alone reaches back to 2015-07-22 — comfortably
    past this script's ~2-year lookback, so the "files" pagination array
    for older history is checked but expected unused for every seed ticker
    (recorded as an honesty flag per ticker if a filer's own "recent"
    window turns out shallower than the lookback needs).

COMPILE-ONCE (EDGE DOCTRINE #3): reuses wikiattention_gate2.py's pure
functions (zscore_series, spike_day_indices, daily_returns,
forward_volume_ratio, forward_realized_vol, welch_vs_baseline, pool_group,
align_views_to_trading_days, fetch_wiki_daily_views, _wiki_articles,
MEGA_CAP_TICKERS) by import rather than re-deriving them.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location("wikiattention_gate2", os.path.join(_HERE, "wikiattention_gate2.py"))
wag2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wag2)

SEC_UA = {"User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)"}
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ADDITIONAL_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"


# ── Pure functions (unit-tested, no network) ────────────────────────────────

def parse_company_tickers(raw: dict) -> dict:
    """SEC company_tickers.json's dict-of-rows shape -> {ticker: cik10str}.
    Each row's ticker is the map key SEC itself uses; a ticker collision
    would mean SEC listed the same symbol twice, which does not happen in
    practice — last-row-wins is an acceptable, honestly-unguarded default
    here (unlike sec8kEarnings.ts's CIK->ticker direction, which DOES need
    the suffixed-class guard, because there multiple tickers legitimately
    share one CIK; here each ticker still maps to exactly one CIK)."""
    out = {}
    for row in (raw or {}).values():
        t = row.get("ticker")
        cik = row.get("cik_str")
        if not t or cik is None:
            continue
        out[t] = str(cik).zfill(10)
    return out


def parse_submissions_8k_dates(filings_block: dict) -> set:
    """{'form': [...], 'filingDate': [...], ...} (either filings.recent, or
    one of filings.files' additional-page JSONs, both share this flat
    array shape) -> the set of filingDate strings where form == '8-K' or
    '8-K/A' (amendments still represent a same-day-news event). Ignores
    every other form type, and ignores item codes entirely (the ANY-8-K bar
    this script's own docstring states, broader than Item-2.02-only)."""
    forms = filings_block.get("form", []) or []
    dates = filings_block.get("filingDate", []) or []
    return {d for f, d in zip(forms, dates) if f in ("8-K", "8-K/A") and d}


def is_newsfree_spike_idx(i: int, dates: Sequence[str], filing_dates: set) -> bool:
    """A spike at trading-day index i is 'news-free' iff no 8-K was filed
    on that trading day itself, nor on the immediately preceding trading
    day (the after-hours-filing lag wikiattention_gate1.ts's own docstring
    already reasons about for the same event class)."""
    if dates[i] in filing_dates:
        return False
    if i > 0 and dates[i - 1] in filing_dates:
        return False
    return True


def evaluate_ticker_newsfree(dates: Sequence[str], views: Sequence[Optional[float]],
                              closes: Sequence[float], volumes: Sequence[float],
                              filing_dates: set,
                              threshold: float = wag2.DEFAULT_Z_THRESHOLD,
                              window: int = wag2.DEFAULT_TRAILING_WINDOW,
                              horizons: Sequence[int] = wag2.DEFAULT_HORIZONS) -> dict:
    """Same pipeline as wikiattention_gate2.evaluate_ticker, except the
    SAMPLE side is restricted to news-free spike days. The BASELINE side is
    unchanged: every day that is not a z>=threshold spike (news-day or not)
    stays in the baseline, exactly as in the original script — only the
    spike/sample definition is narrowed, so this is a direct extension of
    the existing design, not a different comparison."""
    n = len(dates)
    z = wag2.zscore_series(views, window)
    all_spikes = set(wag2.spike_day_indices(z, threshold))
    newsfree_spikes = {i for i in all_spikes if is_newsfree_spike_idx(i, dates, filing_dates)}
    rets = wag2.daily_returns(closes)
    out = {
        "n_days": n,
        "n_spike_days_total": len(all_spikes),
        "n_spike_days_newsfree": len(newsfree_spikes),
        "n_spike_days_excluded_for_news": len(all_spikes) - len(newsfree_spikes),
        "horizons": {},
    }
    for h in horizons:
        vol_spike, vol_base, rv_spike, rv_base = [], [], [], []
        for i in range(n):
            vr = wag2.forward_volume_ratio(volumes, i, h)
            rv = wag2.forward_realized_vol(rets, i, h)
            in_baseline = i not in all_spikes  # unchanged: excludes EVERY spike day, news-free or not
            in_sample = i in newsfree_spikes
            if vr is not None:
                if in_sample:
                    vol_spike.append(vr)
                elif in_baseline:
                    vol_base.append(vr)
            if rv is not None:
                if in_sample:
                    rv_spike.append(rv)
                elif in_baseline:
                    rv_base.append(rv)
        out["horizons"][h] = {
            "volume_ratio": wag2.welch_vs_baseline(vol_spike, vol_base),
            "realized_vol": wag2.welch_vs_baseline(rv_spike, rv_base),
            "_raw": {"vol_spike": vol_spike, "vol_base": vol_base, "rv_spike": rv_spike, "rv_base": rv_base},
        }
    return out


# ── Network ──────────────────────────────────────────────────────────────────

def _http_get_json(url: str, timeout: int = 20, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=SEC_UA)
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def fetch_cik_map() -> dict:
    return parse_company_tickers(_http_get_json(COMPANY_TICKERS_URL))


def fetch_8k_dates_for_cik(cik10: str, cutoff_iso: str, max_additional_pages: int = 3) -> tuple:
    """Returns (dates_set, fully_covered: bool). fully_covered is True iff
    the earliest filingDate seen is <= cutoff_iso (i.e. coverage reaches
    back far enough for this script's lookback window) — checked honestly
    rather than assumed, since 'filings.recent' is a fixed-size window
    (~1000 filings) that could in principle not reach back far enough for
    a filer with unusually high overall filing volume."""
    data = _http_get_json(SUBMISSIONS_URL.format(cik10=cik10))
    filings = data.get("filings", {})
    recent = filings.get("recent", {})
    dates = parse_submissions_8k_dates(recent)
    all_dates_seen = list(recent.get("filingDate", []) or [])
    pages = filings.get("files", []) or []
    for page in pages[:max_additional_pages]:
        name = page.get("name")
        if not name:
            continue
        try:
            page_data = _http_get_json(ADDITIONAL_SUBMISSIONS_URL.format(name=name))
        except Exception:
            continue  # one unreachable archive page degrades coverage, doesn't kill the ticker
        dates |= parse_submissions_8k_dates(page_data)
        all_dates_seen += list(page_data.get("filingDate", []) or [])
        earliest = min(all_dates_seen) if all_dates_seen else None
        if earliest and earliest <= cutoff_iso:
            break
    earliest = min(all_dates_seen) if all_dates_seen else None
    fully_covered = bool(earliest and earliest <= cutoff_iso)
    return dates, fully_covered


def run_gate2_newsfree(tickers: Sequence[str], days: int = wag2.DEFAULT_MIN_TRADING_DAYS + 250,
                        threshold: float = wag2.DEFAULT_Z_THRESHOLD,
                        window: int = wag2.DEFAULT_TRAILING_WINDOW,
                        horizons: Sequence[int] = wag2.DEFAULT_HORIZONS,
                        wiki_spacing_s: float = 0.6, sec_spacing_s: float = 0.3) -> dict:
    articles = wag2._wiki_articles()
    end = datetime.utcnow() - timedelta(days=1)
    start = end - timedelta(days=days)
    end_s, start_s = end.strftime("%Y%m%d"), start.strftime("%Y%m%d")
    cutoff_iso = start.strftime("%Y-%m-%d")

    sys.path.insert(0, os.path.dirname(_HERE))
    import backtest_v2 as bt

    cik_map = fetch_cik_map()

    per_ticker = {}
    coverage_flags = {}
    for t in tickers:
        article = articles.get(t)
        cik10 = cik_map.get(t)
        if not article or not cik10:
            per_ticker[t] = {"error": f"missing {'article' if not article else 'CIK'} for ticker"}
            continue
        try:
            views_by_date = wag2.fetch_wiki_daily_views(article, start_s, end_s)
        except Exception as e:
            per_ticker[t] = {"error": f"wiki fetch failed: {e}"}
            continue
        finally:
            time.sleep(wiki_spacing_s)
        try:
            filing_dates, fully_covered = fetch_8k_dates_for_cik(cik10, cutoff_iso)
            coverage_flags[t] = fully_covered
        except Exception as e:
            per_ticker[t] = {"error": f"EDGAR submissions fetch failed: {e}"}
            continue
        finally:
            time.sleep(sec_spacing_s)
        try:
            bars = bt.fetch_bars(t, days)
            if not bars or not bars.get("date"):
                raise RuntimeError("empty bars")
        except Exception as e:
            per_ticker[t] = {"error": f"price fetch failed: {e}"}
            continue
        trading_dates = bars["date"]
        views = wag2.align_views_to_trading_days(views_by_date, trading_dates)
        row = evaluate_ticker_newsfree(trading_dates, views, bars["close"], bars["volume"],
                                        filing_dates, threshold, window, horizons)
        row["cap_tier"] = "mega" if t in wag2.MEGA_CAP_TICKERS else "small_mid"
        row["edgar_coverage_full"] = fully_covered
        row["n_8k_dates_in_window"] = len({d for d in filing_dates if start_s <= d.replace("-", "") <= end_s})
        per_ticker[t] = row

    small_mid = [t for t in tickers if t not in wag2.MEGA_CAP_TICKERS]
    mega = [t for t in tickers if t in wag2.MEGA_CAP_TICKERS]
    pooled = {
        "small_mid_cap": wag2.pool_group(per_ticker, small_mid, horizons),
        "mega_cap_comparison": wag2.pool_group(per_ticker, mega, horizons),
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
        "window_days": days,
        "threshold": threshold,
        "trailing_window": window,
        "horizons": list(horizons),
        "tickers_requested": list(tickers),
        "edgar_coverage_incomplete_tickers": [t for t, ok in coverage_flags.items() if not ok],
        "pooled": pooled,
        "per_ticker": per_ticker_summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default=None, help="comma-separated; default = full seed set")
    ap.add_argument("--days", type=int, default=wag2.DEFAULT_MIN_TRADING_DAYS + 250)
    ap.add_argument("--threshold", type=float, default=wag2.DEFAULT_Z_THRESHOLD)
    ap.add_argument("--window", type=int, default=wag2.DEFAULT_TRAILING_WINDOW)
    ap.add_argument("--horizons", default=",".join(str(h) for h in wag2.DEFAULT_HORIZONS))
    args = ap.parse_args()

    tickers = args.tickers.split(",") if args.tickers else list(wag2._wiki_articles().keys())
    horizons = [int(h) for h in args.horizons.split(",")]
    result = run_gate2_newsfree(tickers, args.days, args.threshold, args.window, horizons)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
