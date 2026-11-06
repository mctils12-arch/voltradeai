#!/usr/bin/env python3
"""
stream_signal_gate2.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for the Tier-0
stream flags. No trading involved.

The flags come from server/streamSignals.ts and are logged since 2026-11-06
to DATA_DIR/voltrade_stream_signals.jsonl. A flag is a regular-session 1-min
bar with volume >= 2.5x the day's earlier bars, a move >= 0.8% and 5-bar
momentum >= 0.3%, on one of 43 mega caps.

PRE-REGISTERED 2026-11-06, before any outcome was seen:

- Question: after a flag, does the price CONTINUE in the bar's direction, or
  REVERSE?
- Prior: reversal. Llorente, Michaely, Saar & Wang (RFS 2002): "Stocks of
  larger firms, or stocks with smaller bid-ask spreads, show almost no
  continuation and mostly return reversal following high-volume days."
- Outcomes are measured from the flag's own price p (the bar's close), with
  daily closes:
    r0 = close(flag day) / p - 1
    r1 = close(next session) / p - 1
    r5 = close(fifth session after) / p - 1
  Each is signed by the direction s (+1 bullish, -1 bearish).
- Base rate (REASONING STANDARD #3): mu_h is the same ticker's mean
  h-session close-to-close return over the bars fetched (mu_0 = 0).
  excess_h = s * (r_h - mu_h).
- Independence: only a ticker's first flag of each ET day is tested; later
  flags overlap its outcome window. They are counted, not tested.
- Test: t = mean(excess_h) / (sd / sqrt(n)). Three horizons, so the
  Bonferroni bar is |t| > 2.39.
- Verdict, from h = 1 only, with n >= 200 tested flags:
    REVERSAL      if t < -2.39
    CONTINUATION  if t > +2.39
    NO_EDGE       otherwise
    INSUFFICIENT_N below 200 (the numbers so far are still printed).
- r0 and r5, and the splits (direction, first hour vs later), are
  descriptive only, never the verdict.
- Sampling caveat: bot.ts records no flag while entries are halted (the
  drawdown halt or the kill switch), so stressed sessions are
  under-sampled.

Usage: python3 scripts/stream_signal_gate2.py [--log PATH] [--days 400] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional

MIN_N = 200
T_CRIT = 2.39            # two-sided 0.05 / 3 horizons
HORIZONS = (0, 1, 5)


def load_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_bad_line": True})
    return rows


def ticker_drift(closes: List[float], h: int) -> float:
    """Mean h-session close-to-close return over the series (0 for h = 0)."""
    if h <= 0 or len(closes) <= h:
        return 0.0
    rets = [closes[j + h] / closes[j] - 1.0 for j in range(len(closes) - h) if closes[j] > 0]
    return sum(rets) / len(rets) if rets else 0.0


def _stats(xs: List[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean_pct": None, "t": None}
    mean = sum(xs) / n
    if n < 2:
        return {"n": n, "mean_pct": round(100 * mean, 4), "t": None}
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else None
    return {"n": n, "mean_pct": round(100 * mean, 4), "t": None if t is None else round(t, 3)}


def score(rows: Iterable[dict], bars_by_ticker: Dict[str, dict]) -> dict:
    """The pre-registered test on logged flags and daily bars
    ({ticker: {"date": [...], "close": [...]}})."""
    tested, skipped = [], Counter()
    seen_day = set()
    for r in rows:
        if r.get("_bad_line"):
            skipped["unparseable line"] += 1
            continue
        t, day, p = str(r.get("ticker") or ""), str(r.get("et_day") or ""), r.get("price")
        s = {"bullish": 1, "bearish": -1}.get(str(r.get("direction") or ""))
        if not t or not day or s is None or not isinstance(p, (int, float)) or p <= 0:
            skipped["incomplete row"] += 1
            continue
        if (t, day) in seen_day:
            skipped["later flag, same ticker and day (overlapping window)"] += 1
            continue
        seen_day.add((t, day))
        bars = bars_by_ticker.get(t) or {}
        dates, closes = list(bars.get("date") or []), [float(c) for c in (bars.get("close") or [])]
        if day not in dates:
            skipped["no daily bar for the flag day"] += 1
            continue
        i = dates.index(day)
        if i + max(HORIZONS) >= len(closes):
            skipped["outcome window not complete yet"] += 1
            continue
        rec = {"ticker": t, "day": day, "s": s, "first_hour": int(r.get("et_minute") or 0) < 10 * 60 + 30}
        for h in HORIZONS:
            raw = closes[i + h] / float(p) - 1.0
            rec[f"r{h}"] = raw
            rec[f"x{h}"] = s * (raw - ticker_drift(closes, h))
        tested.append(rec)

    by_h = {f"h{h}": _stats([x[f"x{h}"] for x in tested]) for h in HORIZONS}
    h1 = by_h["h1"]
    if h1["n"] < MIN_N:
        verdict = "INSUFFICIENT_N"
    elif h1["t"] is not None and h1["t"] < -T_CRIT:
        verdict = "REVERSAL"
    elif h1["t"] is not None and h1["t"] > T_CRIT:
        verdict = "CONTINUATION"
    else:
        verdict = "NO_EDGE"

    def split(pred) -> dict:
        return {f"h{h}": _stats([x[f"x{h}"] for x in tested if pred(x)]) for h in HORIZONS}

    return {
        "verdict": verdict,
        "rule": f"h=1 excess, |t| > {T_CRIT} (Bonferroni over {len(HORIZONS)} horizons), n >= {MIN_N}; prior REVERSAL",
        "n_tested": len(tested),
        "skipped": dict(skipped),
        "excess_by_horizon": by_h,
        "descriptive": {
            "bullish": split(lambda x: x["s"] > 0),
            "bearish": split(lambda x: x["s"] < 0),
            "first_hour": split(lambda x: x["first_hour"]),
            "after_first_hour": split(lambda x: not x["first_hour"]),
        },
    }


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_log() -> str:
    try:
        from storage_config import DATA_DIR
    except ImportError:
        DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"
    return os.path.join(DATA_DIR, "voltrade_stream_signals.jsonl")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--log", default=None)
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    rows = load_rows(a.log or _default_log())
    from backtest_v2 import fetch_bars        # Alpaca first, Yahoo fallback, cached a day
    bars = {}
    for t in sorted({str(r.get("ticker")) for r in rows if r.get("ticker")}):
        try:
            bars[t] = fetch_bars(t, a.days)
        except Exception as e:                # one ticker's failed fetch is reported, the rest still scored
            print(f"bars for {t} unavailable: {e}", file=sys.stderr)
    out = score(rows, bars)
    out["log_rows"] = len(rows)
    text = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
