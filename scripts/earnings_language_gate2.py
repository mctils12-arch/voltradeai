#!/usr/bin/env python3
"""earnings_language_gate2.py — SEC 8-K earnings-language stream, GATE 2
(SIGNAL layer of the ROOT VALIDATION LADDER).

Context: research/open_questions.md NEW DATA ROOTS #1. Gate 1 (DATA)
shipped 2026-07-04 (server/sec8kEarnings.ts) — the archive has been
recording Item 2.02 8-K exhibit text since then. This script is gate 2:
"does guidance/results language predict forward returns better than a
size-matched random-entry base rate" (open_questions.md's own ladder
text), tested against real archived filings for the first time.

PRIOR — STATED BEFORE ANY RETURN WAS COMPUTED (REASONING STANDARD #10):
expect a positive but small/noisy association between filing tone and
forward return over 1-3 trading days (post-earnings-announcement-drift
literature), because (a) the archive is currently only ~8 days deep, so
N will be far below what a confident verdict needs, and (b) this pass
tests TONE LEVEL ONLY, not the QoQ LANGUAGE-DELTA feature the original
hypothesis leads with (Lazy-Prices-style guidance changes) — delta needs
at least one PRIOR quarter's filing per company, which does not exist
yet (Q2 2026 filings just started arriving; Q3 doesn't land until
~Oct-Nov 2026). Expected outcome: INSUFFICIENT SAMPLE is the most likely
honest verdict at every horizon; correlation direction/magnitude logged
as a preliminary data point, not a trading claim.

VERDICT RULE — PRE-STATED, PER HORIZON (never adjusted after seeing
results):
  - MIN_SAMPLE = 30 filings with a computable forward return at that
    horizon. N < MIN_SAMPLE => "INSUFFICIENT SAMPLE" (correlation still
    reported, never used to claim PASS/FAIL).
  - N >= MIN_SAMPLE: split into tone > median vs tone <= median; PASS
    requires (top_mean_alpha - bottom_mean_alpha) >= PASS_SPREAD_BPS
    AND same sign as the Pearson r(tone, alpha); else FAIL.
  - "alpha" = ticker's lookahead-free forward return minus SPY's forward
    return over the identical trading-day window (REASONING STANDARD #3
    — demand a base rate, not a raw number).
  - Regime split: NOT attempted this pass — insufficient N even before
    slicing by regime. Re-run instruction below states the trigger.

LOOKAHEAD GUARD (REASONING STANDARD #7): the reference ("pre-news")
price is always the close STRICTLY BEFORE the filing could have moved
the stock — the close on the filing's own calendar day only if the
acceptance timestamp is after 4:00pm ET (i.e. that day's session could
not have reacted yet); otherwise (pre-market, intraday, or unknown
timestamp) the prior trading day's close, treating the filing day itself
as the first reaction day. This never uses a close that already
reflects the filing.

RE-RUN TRIGGER: re-run this script (unchanged criteria) once the
archive holds >=90 days of history (enough filings for N>=30 at the
5-day horizon) OR once a second quarter of filings exists per company
(enables the QoQ language-delta feature the original hypothesis
actually leads with — that is a DIFFERENT script, not this one, since
it needs a paired-filing join this pass cannot do).

Usage:
  python3 scripts/earnings_language_gate2.py
      [--archive-dir DIR]      # local <base>/datacore_archive/earnings8k
      [--api-url URL]          # fallback: live /api/data/earnings-language/history
      [--input-json FILE]      # offline fixture (same shape as the API response)
      [--days N]               # archive lookback window (default 30)
      [--out FILE]             # optional: also write full JSON result here
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import urllib.request
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from backtest_v2 import fetch_bars  # noqa: E402  (Alpaca-first/Yahoo-fallback, same data path as live)

MIN_SAMPLE = 30
PASS_SPREAD = 0.005  # 50 bps
HORIZONS = (1, 3, 5)

# ── Compact finance-tone lexicon, inspired by the Loughran-McDonald
# Positive/Negative categories used in the accounting-and-finance NLP
# literature — a session-curated ~80-term subset per category, NOT the
# full published LM Master Dictionary (that requires a licensed download
# this sandbox cannot fetch). Labeled honestly: this is a compact
# approximation good enough for a first-pass tone score, not a
# reproduction of the academic instrument. ─────────────────────────────
NEGATIVE_WORDS = frozenset("""
abandon abandoned adverse adversely allegations alleging bankrupt
bankruptcy breach breached closure closures complaint complaints
contraction crisis decline declined declines declining default
defaults deficiencies deficiency deficit delinquency delinquent
deteriorate deteriorated deterioration difficult difficulties
disappointing discontinue discontinued disruption disruptions
downgrade downgraded downsizing downturn impairment impairments
inadequate ineffective infringement insolvency insolvent
investigation layoff layoffs liquidation litigation loss losses
misconduct misstatement negative noncompliance penalties penalty
recession restated restatement restructuring setback setbacks
shortfall shortfalls shrinkage slowdown suspend suspended suspension
termination terminated unfavorable unfavorably volatile volatility
weak weakened weakness weaknesses writedown writeoff write-down
write-off worsened worsening underperform underperformed impaired
""".split())

POSITIVE_WORDS = frozenset("""
achieve achieved achievement advantage advantages benefit benefits
benefited boost boosted deliver delivered exceed exceeded exceeding
excellent expand expanded expansion favorable favorably gain gained
gains grow growth grew improve improved improvement improvements
increase increased increasing innovation innovative leadership leading
milestone momentum opportunities opportunity outperform outperformed
positive profitability profitable record records recovery resilient
robust strength strengthen strengthened strong stronger strongest
success successful successfully surpass surpassed upgrade upgraded
accelerate accelerated outstanding solid rebound rebounded
""".split())

TOKEN_RE = re.compile(r"[a-zA-Z]+(?:-[a-zA-Z]+)?")


def tone_score(text: str) -> dict:
    tokens = [t.lower() for t in TOKEN_RE.findall(text or "")]
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    n = len(tokens)
    return {
        "pos_count": pos, "neg_count": neg, "word_count": n,
        "tone_per_1000w": round(1000 * (pos - neg) / n, 3) if n else None,
    }


# ── Loading filings ──────────────────────────────────────────────────
def default_archive_dir() -> str:
    data_dir = os.environ.get("DATA_DIR") or ("/data/voltrade" if os.path.isdir("/data") else "/tmp")
    return os.path.join(data_dir, "datacore_archive", "earnings8k")


def load_from_archive_dir(dir_path: str, days: int) -> list[dict]:
    if not os.path.isdir(dir_path):
        return []
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    out, seen = [], set()
    for d in range(days):
        day = (now - timedelta(days=d)).strftime("%Y-%m-%d")
        for suffix, opener in ((".jsonl", open), (".jsonl.gz", gzip.open)):
            fp = os.path.join(dir_path, f"{day}{suffix}")
            if not os.path.exists(fp):
                continue
            mode = "rt"
            with opener(fp, mode) as f:  # type: ignore[arg-type]
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    acc = rec.get("accession")
                    if acc and acc not in seen:
                        seen.add(acc)
                        out.append(rec)
    return out


def load_from_url(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read())
    return payload.get("filings", [])


def load_filings(args) -> tuple[list[dict], str]:
    if args.input_json:
        with open(args.input_json) as f:
            payload = json.load(f)
        return payload.get("filings", payload if isinstance(payload, list) else []), f"input-json:{args.input_json}"
    archive_dir = args.archive_dir or default_archive_dir()
    local = load_from_archive_dir(archive_dir, args.days)
    if local:
        return local, f"archive-dir:{archive_dir}"
    return load_from_url(args.api_url), f"api-url:{args.api_url}"


# ── Price / return computation ───────────────────────────────────────
def et_hour_from_iso(iso: str | None) -> int | None:
    if not iso or "T" not in iso:
        return None
    try:
        return int(iso.split("T")[1][:2])
    except (ValueError, IndexError):
        return None


def reaction_indices(dates: list[str], filing_date: str, accept_iso: str | None) -> tuple[int, int] | None:
    """Returns (pre_news_idx, anchor_idx) into `dates` (ascending), or None
    if there isn't enough bar history around the filing. `anchor_idx` is
    the first trading day whose CLOSE could reflect the filing; the day
    before it (`pre_news_idx`) is the lookahead-free reference price."""
    filing_idx = next((i for i, d in enumerate(dates) if d >= filing_date), None)
    if filing_idx is None:
        return None
    same_day = dates[filing_idx] == filing_date
    hour = et_hour_from_iso(accept_iso)
    if same_day and hour is not None and hour >= 16:
        anchor_idx, pre_idx = filing_idx + 1, filing_idx
    else:
        anchor_idx, pre_idx = filing_idx, filing_idx - 1
    if pre_idx < 0 or anchor_idx >= len(dates):
        return None
    return pre_idx, anchor_idx


def forward_returns(dates: list[str], closes: list[float], pre_idx: int, anchor_idx: int,
                     horizons: tuple[int, ...]) -> dict[int, tuple[str, float]]:
    """{horizon: (as_of_date, return)} for each horizon whose bar exists."""
    out = {}
    ref = closes[pre_idx]
    if not ref:
        return out
    for h in horizons:
        idx = anchor_idx + h - 1
        if idx < len(dates):
            out[h] = (dates[idx], closes[idx] / ref - 1)
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


def compute(filings: list[dict]) -> dict:
    skipped_no_ticker = sum(1 for f in filings if not f.get("ticker"))
    scored = []
    for f in filings:
        if not f.get("ticker") or not f.get("text"):
            continue
        ts = tone_score(f["text"])
        if ts["tone_per_1000w"] is None:
            continue
        scored.append({**f, "tone": ts})

    spy_bars = fetch_bars("SPY", 90)
    spy_map = dict(zip(spy_bars["date"], spy_bars["close"]))

    ticker_bars: dict[str, dict] = {}
    fetch_errors = []
    per_horizon_samples: dict[int, list[dict]] = {h: [] for h in HORIZONS}

    for rec in scored:
        ticker = rec["ticker"]
        filing_date = (rec.get("filedAt") or (rec.get("acceptanceDatetime") or "")[:10])
        if not filing_date:
            continue
        if ticker not in ticker_bars:
            try:
                ticker_bars[ticker] = fetch_bars(ticker, 90)
            except Exception as e:  # noqa: BLE001 — one bad ticker never kills the run
                ticker_bars[ticker] = None
                fetch_errors.append({"ticker": ticker, "err": str(e)[:150]})
        bars = ticker_bars[ticker]
        if not bars or not bars.get("date"):
            continue
        ri = reaction_indices(bars["date"], filing_date, rec.get("acceptanceDatetime"))
        if ri is None:
            continue
        pre_idx, anchor_idx = ri
        fwd = forward_returns(bars["date"], bars["close"], pre_idx, anchor_idx, HORIZONS)
        for h, (as_of, ret) in fwd.items():
            pre_date = bars["date"][pre_idx]
            if pre_date not in spy_map or as_of not in spy_map:
                continue
            spy_ret = spy_map[as_of] / spy_map[pre_date] - 1
            per_horizon_samples[h].append({
                "ticker": ticker, "accession": rec.get("accession"), "filedAt": filing_date,
                "tone_per_1000w": rec["tone"]["tone_per_1000w"], "raw_return": round(ret, 5),
                "spy_return": round(spy_ret, 5), "alpha": round(ret - spy_ret, 5),
            })

    horizon_results = {}
    for h, samples in per_horizon_samples.items():
        n = len(samples)
        r = pearson([s["tone_per_1000w"] for s in samples], [s["alpha"] for s in samples]) if n >= 3 else None
        entry = {"n": n, "pearson_r_tone_vs_alpha": round(r, 4) if r is not None else None}
        if n < MIN_SAMPLE:
            entry["verdict"] = "INSUFFICIENT SAMPLE"
            entry["note"] = f"n={n} < MIN_SAMPLE={MIN_SAMPLE}; correlation logged as a preliminary data point only, no PASS/FAIL claim"
        else:
            tones = sorted(s["tone_per_1000w"] for s in samples)
            m = median(tones)
            top = [s["alpha"] for s in samples if s["tone_per_1000w"] > m]
            bot = [s["alpha"] for s in samples if s["tone_per_1000w"] <= m]
            top_mean = sum(top) / len(top) if top else None
            bot_mean = sum(bot) / len(bot) if bot else None
            spread = (top_mean - bot_mean) if (top_mean is not None and bot_mean is not None) else None
            # OUTLIER-ROBUSTNESS DIAGNOSTIC (not part of the pre-stated PASS/
            # FAIL rule — added as a cross-check after the h=1 first run showed
            # a mean dominated by single-name idiosyncratic events, e.g. a
            # -52% print on a trust's final-distribution/wind-down 8-K; the
            # mechanical verdict below is unchanged from the pre-stated rule,
            # but a PASS whose sign flips under the median is flagged, not
            # trusted, per REASONING STANDARD #8/#4).
            top_med = median(top) if top else None
            bot_med = median(bot) if bot else None
            median_spread = (top_med - bot_med) if (top_med is not None and bot_med is not None) else None
            top_extremes = sorted(top, key=abs, reverse=True)[:3]
            bot_extremes = sorted(bot, key=abs, reverse=True)[:3]
            entry.update({
                "median_tone": round(m, 3), "top_group_n": len(top), "bottom_group_n": len(bot),
                "top_mean_alpha": round(top_mean, 5) if top_mean is not None else None,
                "bottom_mean_alpha": round(bot_mean, 5) if bot_mean is not None else None,
                "spread": round(spread, 5) if spread is not None else None,
                "top_median_alpha": round(top_med, 5) if top_med is not None else None,
                "bottom_median_alpha": round(bot_med, 5) if bot_med is not None else None,
                "median_spread": round(median_spread, 5) if median_spread is not None else None,
                "largest_magnitude_alphas_top_group": [round(x, 4) for x in top_extremes],
                "largest_magnitude_alphas_bottom_group": [round(x, 4) for x in bot_extremes],
            })
            same_sign = (spread is not None and r is not None and (spread > 0) == (r > 0))
            entry["verdict"] = ("PASS" if (spread is not None and spread >= PASS_SPREAD and same_sign) else "FAIL")
            median_agrees = (median_spread is not None and spread is not None and (median_spread > 0) == (spread > 0))
            entry["outlier_robust"] = bool(median_agrees)
            if entry["verdict"] == "PASS" and not median_agrees:
                entry["verdict"] = "PASS (NOT OUTLIER-ROBUST — mean and median disagree in sign; do not trust)"
        horizon_results[str(h)] = entry

    return {
        "prior": "positive but small/noisy tone-vs-forward-alpha association expected; "
                 "INSUFFICIENT SAMPLE the most likely honest verdict given ~8 days of archive "
                 "(stated in this script's own header before any return was computed)",
        "qoq_language_delta": "NOT TESTED this pass — needs >=1 prior-quarter filing per company; "
                               "archive is too young (gate 1 shipped 2026-07-04). Re-probe once Q3 "
                               "2026 filings begin arriving (~Oct-Nov 2026).",
        "data": {
            "filings_loaded": len(filings), "skipped_no_ticker": skipped_no_ticker,
            "scored_with_tone": len(scored), "unique_tickers": len(ticker_bars),
            "ticker_fetch_errors": fetch_errors,
        },
        "lexicon_size": {"positive": len(POSITIVE_WORDS), "negative": len(NEGATIVE_WORDS)},
        "verdict_rule": f"n>={MIN_SAMPLE} required for PASS/FAIL; PASS needs top-vs-bottom-tone-half "
                         f"alpha spread >= {PASS_SPREAD} same-signed as Pearson r, else FAIL; n<{MIN_SAMPLE} "
                         "=> INSUFFICIENT SAMPLE regardless of the numbers",
        "horizons": horizon_results,
        "samples": per_horizon_samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=None)
    ap.add_argument("--api-url", default="https://voltradeai-production.up.railway.app/api/data/earnings-language/history?days=30")
    ap.add_argument("--input-json", default=None)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    filings, source = load_filings(args)
    result = compute(filings)
    result["source"] = source

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)

    printable = {k: v for k, v in result.items() if k != "samples"}
    print(json.dumps(printable, indent=1))
    any_insufficient = any(h["verdict"] == "INSUFFICIENT SAMPLE" for h in result["horizons"].values())
    any_fail = any(h["verdict"] == "FAIL" for h in result["horizons"].values())
    return 2 if any_insufficient and not any_fail else (3 if any_fail else 0)


if __name__ == "__main__":
    sys.exit(main())
