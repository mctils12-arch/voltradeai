#!/usr/bin/env python3
"""
scripts/wikiattention_gate2.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for
the wikimedia_pageviews_attention root (research/open_questions.md, "3.
WIKIMEDIA PAGEVIEWS ATTENTION PROXY"). Gate 1 (DATA) passed 2026-08-18
(scripts/wikiattention_gate1.ts): 11/11 hand-checked tickers show pageviews
peaking around a known earnings date. Gate 2 (predictive power) has been
named "untouched" in every session since — this is that script, built per
EDGE DOCTRINE #3 (compile knowledge into code) so a future session with
working price-data access can run it immediately instead of re-deriving the
design.

PRE-REGISTERED HYPOTHESIS (REASONING STANDARD #10, written before any
statistic below was computed): a pageview ATTENTION SPIKE on a company's
Wikipedia article (daily views z-scored against its own trailing 90-day
history, no lookahead — only days strictly before the spike day feed the
baseline) is followed, over the next 1-5 trading days, by ELEVATED trading
volume and realized volatility relative to that same ticker's own non-spike
days. Per the module's own header ("most interesting on smaller names
without same-day news") and the GATE 1 write-up's own directional finding
("modest for large/mega-caps ... strong for the smaller/more retail-driven
names"), the SMALL/MID-CAP subset of the seed universe is the primary test;
the four unambiguously mega-cap seed names (NVDA/AAPL/TSLA/AMD) are a
secondary crowdedness-check comparison group, expected to show a WEAKER
effect (already-priced attention, per REASONING STANDARD #5 — second-order:
mega-cap attention is arbitraged by every options-flow and social-sentiment
desk already; a small/illiquid name's retail-attention spike is the
structurally under-arbitraged case EDGE DOCTRINE #2 names). PRIOR: P(gate 2
pass on the small/mid-cap subset) ~30%, stated in the root's own
open_questions.md entry at gate-1 time, carried forward unchanged here (not
re-estimated after seeing any gate-2 number).

CAP-TIER SPLIT IS COARSE, NOT REAL MARKET-CAP DATA: this script does not
fetch shares-outstanding or historical market cap for any ticker (no clean
free source wired into this repo, and precise 2026 caps are not something
this session can respectively verify against live sourcing without becoming
its own project). NVDA/AAPL/TSLA/AMD are treated as the mega-cap comparison
group purely because they are uncontroversially mega-cap across any
plausible date in this program's timeframe; every other seed ticker is
treated as the primary small/mid-cap test group. This mirrors the informal
split the GATE 1 write-up already used, made explicit and code-pinned here
rather than re-eyeballed each time. A future session with a real market-cap
feed wired in (none exists in datacore/ today) could sharpen this.

NO-LOOKAHEAD: z-scores at day i use only views[i-window:i] (strictly prior
days); the forward window for a spike at day i starts at trading day i+1.
ATTENTION-WITHOUT-NEWS SUBSET NOT BUILT HERE: the original hypothesis names
"attention without a same-day 8-K" as the more interesting case. Excluded
from this pass (REASONING STANDARD #4 — one diagnostic design per session,
not chased into a second variant) since it needs a per-ticker 8-K date feed
this script does not fetch; scripts/wikiattention_gate1.ts already proved
that data reachable via data.sec.gov/submissions, so this is a concretely
buildable follow-up, not a vague one — named in the NEXT note, not attempted.

DATA SOURCES:
  - Pageviews: wikimedia.org/api/rest_v1/metrics/pageviews/per-article,
    fetched DIRECTLY against Wikimedia's own history (stable back to 2015),
    NOT via this repo's own ~2-month rolling archive (server/wikiAttention.ts
    only retains RAW_RETENTION_DAYS) — this is what makes a real trailing-90d
    baseline and a multi-year test window possible at all without waiting
    out this repo's own archive depth.
  - Price/volume: backtest_v2.fetch_bars() (Alpaca-first, Yahoo fallback,
    disk-cached) — the same plumbing every other gate-2 script in this repo
    reuses (EDGE DOCTRINE #3). THIS HALF IS FREQUENTLY BLOCKED in sandboxes
    without ALPACA_KEY/ALPACA_SECRET when the Yahoo fallback's egress is
    also rate-limited or proxy-reset (documented repeatedly in
    research/open_questions.md/experiments.md across many sessions,
    including the immediately-preceding 2026-09-04 hazard-rate-probe
    session). This script fetches pageviews and price/volume as two
    INDEPENDENT steps so a session that only has the former can still
    verify the DATA half live rather than skip everything.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Optional, Sequence

WIKI_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
MEGA_CAP_TICKERS = ("NVDA", "AAPL", "TSLA", "AMD")
DEFAULT_TRAILING_WINDOW = 90
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_HORIZONS = (1, 3, 5)
DEFAULT_VOLUME_BASELINE_WINDOW = 20
DEFAULT_MIN_TRADING_DAYS = 500  # ~2 trading years


def _wiki_articles() -> dict:
    """datacore/wiki_articles.json's ARTICLES map, loaded by path (this
    script lives outside server/, mirrors _load_csd_module's by-path
    pattern rather than importing the TS module)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "datacore", "wiki_articles.json")) as f:
        return json.load(f)["articles"]


# ── Pure math (unit-tested, no network) ─────────────────────────────────────

def zscore_series(views: Sequence[Optional[float]], window: int = DEFAULT_TRAILING_WINDOW) -> list:
    """z-score of views[i] against the MEAN/STD of views[i-window:i]
    (strictly prior days — day i's own value never enters its own
    baseline, and no future day ever enters any baseline). None wherever
    the trailing window has fewer than `window` non-null observations, or
    the trailing std is 0 (a flat run makes any z-score fabricated)."""
    out: list = []
    for i in range(len(views)):
        lo = i - window
        if lo < 0:
            out.append(None)
            continue
        trailing = [v for v in views[lo:i] if v is not None]
        if len(trailing) < window or views[i] is None:
            out.append(None)
            continue
        mean = sum(trailing) / len(trailing)
        var = sum((v - mean) ** 2 for v in trailing) / len(trailing)
        std = math.sqrt(var)
        out.append(None if std == 0 else (views[i] - mean) / std)
    return out


def spike_day_indices(zscores: Sequence[Optional[float]], threshold: float = DEFAULT_Z_THRESHOLD) -> list:
    return [i for i, z in enumerate(zscores) if z is not None and z >= threshold]


def daily_returns(closes: Sequence[float]) -> list:
    """Simple pct-change; index 0 has no prior day, returns None there."""
    out: list = [None]
    for i in range(1, len(closes)):
        out.append(None if closes[i - 1] == 0 else (closes[i] / closes[i - 1] - 1))
    return out


def trailing_avg_volume(volumes: Sequence[float], idx: int, window: int = DEFAULT_VOLUME_BASELINE_WINDOW) -> Optional[float]:
    lo = idx - window
    if lo < 0:
        return None
    vals = volumes[lo:idx]
    return sum(vals) / len(vals) if vals else None


def forward_volume_ratio(volumes: Sequence[float], idx: int, horizon: int,
                          baseline_window: int = DEFAULT_VOLUME_BASELINE_WINDOW) -> Optional[float]:
    """mean(volume[idx+1 : idx+1+horizon]) / trailing `baseline_window`-day
    avg volume ending at idx (idx itself excluded from both, so a spike
    day's own volume never leaks into its own baseline)."""
    fwd = volumes[idx + 1: idx + 1 + horizon]
    if len(fwd) < horizon:
        return None
    base = trailing_avg_volume(volumes, idx, baseline_window)
    if not base:
        return None
    return (sum(fwd) / len(fwd)) / base


def forward_realized_vol(returns: Sequence[Optional[float]], idx: int, horizon: int) -> Optional[float]:
    """stdev of daily returns over (idx, idx+horizon] — the forward window
    strictly after the spike day, never including it."""
    fwd = [r for r in returns[idx + 1: idx + 1 + horizon] if r is not None]
    if len(fwd) < horizon:
        return None
    mean = sum(fwd) / len(fwd)
    var = sum((r - mean) ** 2 for r in fwd) / len(fwd)
    return math.sqrt(var)


def align_views_to_trading_days(views_by_date: dict, trading_dates: Sequence[str]) -> list:
    """Maps a {date: views} dict onto a trading-day calendar. A trading day
    with no archived pageview (Wikimedia gap, or a date outside the fetched
    window) is honestly None, never zero-filled (matches wikiAttention.ts's
    own "absence is data, never faked" convention)."""
    return [views_by_date.get(d) for d in trading_dates]


def welch_vs_baseline(sample: Sequence[float], baseline: Sequence[float]) -> Optional[dict]:
    """Welch two-sample t-test, spike-day forward metric vs this ticker's
    (or this group's) own non-spike-day forward metric — the random-entry,
    same-universe, same-horizon comparison REASONING STANDARD #3 demands,
    not an external/absolute bar. None (never fabricated) below n=5/side,
    same floor midas_gate2.py's welch_vs_baseline uses."""
    if len(sample) < 5 or len(baseline) < 5:
        return None
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(sample, baseline, equal_var=False)
    return {
        "n": len(sample),
        "n_baseline": len(baseline),
        "mean": round(sum(sample) / len(sample), 4),
        "baseline_mean": round(sum(baseline) / len(baseline), 4),
        "mean_diff": round(sum(sample) / len(sample) - sum(baseline) / len(baseline), 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_value), 4),
    }


def evaluate_ticker(dates: Sequence[str], views: Sequence[Optional[float]],
                     closes: Sequence[float], volumes: Sequence[float],
                     threshold: float = DEFAULT_Z_THRESHOLD,
                     window: int = DEFAULT_TRAILING_WINDOW,
                     horizons: Sequence[int] = DEFAULT_HORIZONS) -> dict:
    """One ticker's full pipeline: dates/views/closes/volumes must already
    be aligned 1:1 by trading day (caller's job — see run_gate2). Returns
    per-horizon spike-day vs non-spike-day forward-volume-ratio and
    forward-realized-vol samples, ready for welch_vs_baseline."""
    n = len(dates)
    z = zscore_series(views, window)
    spikes = set(spike_day_indices(z, threshold))
    rets = daily_returns(closes)
    out = {"n_days": n, "n_spike_days": len(spikes), "horizons": {}}
    for h in horizons:
        vol_spike, vol_base, rv_spike, rv_base = [], [], [], []
        for i in range(n):
            vr = forward_volume_ratio(volumes, i, h)
            rv = forward_realized_vol(rets, i, h)
            if vr is not None:
                (vol_spike if i in spikes else vol_base).append(vr)
            if rv is not None:
                (rv_spike if i in spikes else rv_base).append(rv)
        out["horizons"][h] = {
            "volume_ratio": welch_vs_baseline(vol_spike, vol_base),
            "realized_vol": welch_vs_baseline(rv_spike, rv_base),
            # raw samples, kept so run_gate2 can POOL across tickers within a
            # cap-tier group rather than only report each ticker's own
            # (usually underpowered, n_spike_days ~ 8-37) individual test.
            "_raw": {"vol_spike": vol_spike, "vol_base": vol_base,
                     "rv_spike": rv_spike, "rv_base": rv_base},
        }
    return out


def pool_group(per_ticker: dict, tickers: Sequence[str], horizons: Sequence[int]) -> dict:
    """Pools _raw spike/baseline samples across `tickers` (a cap-tier
    group) and reruns welch_vs_baseline on the POOLED sample per horizon —
    the single pre-registered composite test REASONING STANDARD #4 asks
    for, instead of eyeballing N separate per-ticker p-values. Caveat this
    carries honestly rather than hides: pooling treats each ticker-day as
    an independent draw, which is not strictly true (correlated market-wide
    days), the same simplification this repo's other cross-ticker gate-2
    scripts (e.g. usaspending_gate2.py's cross-sectional pooling) already
    make and document — not a new, unexamined assumption."""
    out = {}
    for h in horizons:
        vs, vb, rs, rb = [], [], [], []
        for t in tickers:
            row = per_ticker.get(t, {}).get("horizons", {}).get(h)
            if not row or "_raw" not in row:
                continue
            raw = row["_raw"]
            vs.extend(raw["vol_spike"]); vb.extend(raw["vol_base"])
            rs.extend(raw["rv_spike"]); rb.extend(raw["rv_base"])
        out[h] = {
            "volume_ratio": welch_vs_baseline(vs, vb),
            "realized_vol": welch_vs_baseline(rs, rb),
            "n_tickers_pooled": sum(1 for t in tickers if t in per_ticker and "horizons" in per_ticker[t]),
        }
    return out


# ── Network (skipped entirely by unit tests) ────────────────────────────────

def _http_get(url: str, timeout: int = 20, retries: int = 3) -> bytes:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-wikiattention-gate2/1.0 (+https://voltradeai.com)"})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def parse_wiki_response(raw: dict) -> dict:
    """{date: views} from the documented items[] shape (mirrors
    wikiAttention.ts's parsePageviews — same malformed-item-dropped
    discipline). Pure, unit-testable without network."""
    out = {}
    for it in (raw or {}).get("items", []) or []:
        ts = str(it.get("timestamp", ""))
        if len(ts) == 10 and isinstance(it.get("views"), int):
            out[f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"] = it["views"]
    return out


def fetch_wiki_daily_views(article: str, start: str, end: str) -> dict:
    """{date: views} for one article, [start, end] inclusive, YYYYMMDD."""
    url = f"{WIKI_API}/{urllib.parse.quote(article, safe='')}/daily/{start}00/{end}00"
    return parse_wiki_response(json.loads(_http_get(url)))


import urllib.parse  # noqa: E402 (used by fetch_wiki_daily_views above)


def run_gate2(tickers: Sequence[str], days: int = 730,
              threshold: float = DEFAULT_Z_THRESHOLD,
              window: int = DEFAULT_TRAILING_WINDOW,
              horizons: Sequence[int] = DEFAULT_HORIZONS,
              wiki_spacing_s: float = 0.6) -> dict:
    """Orchestrates the full gate-2 run against REAL data: Wikimedia
    pageviews (independent host, usually reachable) and backtest_v2's
    fetch_bars (Alpaca-first/Yahoo-fallback, frequently blocked in a
    sandbox without ALPACA_KEY/ALPACA_SECRET or open egress to Yahoo).
    The two fetches are deliberately independent so a run with only the
    former still reports something honest instead of failing outright."""
    articles = _wiki_articles()
    end = datetime.utcnow() - timedelta(days=1)
    start = end - timedelta(days=days)
    end_s, start_s = end.strftime("%Y%m%d"), start.strftime("%Y%m%d")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import backtest_v2 as bt

    per_ticker = {}
    price_blocked = 0
    for t in tickers:
        article = articles.get(t)
        if not article:
            per_ticker[t] = {"error": "no seed article for ticker"}
            continue
        try:
            views_by_date = fetch_wiki_daily_views(article, start_s, end_s)
        except Exception as e:
            per_ticker[t] = {"error": f"wiki fetch failed: {e}"}
            continue
        finally:
            # observed live this session: an un-spaced 23-article burst hits
            # Wikimedia's 429 on the tail end (wikiAttention.ts's own
            # REQUEST_SPACING_MS=600ms note already documents this same
            # rate limit for the poller; mirrored here for the same reason)
            time.sleep(wiki_spacing_s)
        try:
            bars = bt.fetch_bars(t, days)
            if not bars or not bars.get("date"):
                raise RuntimeError("empty bars")
        except Exception as e:
            price_blocked += 1
            per_ticker[t] = {
                "error": f"price fetch failed: {e}",
                "n_wiki_days_fetched": len(views_by_date),
            }
            continue
        trading_dates = bars["date"]
        views = align_views_to_trading_days(views_by_date, trading_dates)
        per_ticker[t] = evaluate_ticker(trading_dates, views, bars["close"], bars["volume"],
                                        threshold, window, horizons)
        per_ticker[t]["cap_tier"] = "mega" if t in MEGA_CAP_TICKERS else "small_mid"

    small_mid = [t for t in tickers if t not in MEGA_CAP_TICKERS]
    mega = [t for t in tickers if t in MEGA_CAP_TICKERS]
    pooled = {
        "small_mid_cap": pool_group(per_ticker, small_mid, horizons),
        "mega_cap_comparison": pool_group(per_ticker, mega, horizons),
    }

    # drop the raw per-day sample arrays from the printed per-ticker view
    # (already folded into `pooled` above) -- keeps the report readable.
    per_ticker_summary = {}
    for t, row in per_ticker.items():
        if "horizons" not in row:
            per_ticker_summary[t] = row
            continue
        trimmed = dict(row)
        trimmed["horizons"] = {
            h: {k: v for k, v in hrow.items() if k != "_raw"} for h, hrow in row["horizons"].items()
        }
        per_ticker_summary[t] = trimmed

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window_days": days,
        "threshold": threshold,
        "trailing_window": window,
        "horizons": list(horizons),
        "tickers_requested": list(tickers),
        "tickers_price_blocked": price_blocked,
        "pooled": pooled,
        "per_ticker": per_ticker_summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default=None, help="comma-separated; default = full seed set")
    ap.add_argument("--days", type=int, default=DEFAULT_MIN_TRADING_DAYS + 250)
    ap.add_argument("--threshold", type=float, default=DEFAULT_Z_THRESHOLD)
    ap.add_argument("--window", type=int, default=DEFAULT_TRAILING_WINDOW)
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    args = ap.parse_args()

    tickers = args.tickers.split(",") if args.tickers else list(_wiki_articles().keys())
    horizons = [int(h) for h in args.horizons.split(",")]
    result = run_gate2(tickers, args.days, args.threshold, args.window, horizons)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
