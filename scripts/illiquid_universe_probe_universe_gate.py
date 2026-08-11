#!/usr/bin/env python3
"""
illiquid_universe_probe_universe_gate.py — LADDER PATH step 5
("LOGIC-gate ablation against the live bot's actual deep_score/
tier1_csp_core candidate path, not the ETF-rotation-style backtest engine")
for the open_questions.md entry filed 2026-07-24 ("Does mean_reversion have
a real, exploitable edge specifically in illiquid small-caps?").

Steps 1-4 all ran the pinned ILLIQUID/MODERATE/LIQUID ticker lists straight
through `backtest_v2.run_backtest(ticker, "mean_reversion", ...)` — calling
the strategy backtest directly, per-ticker. That bypasses a step every real
candidate must clear BEFORE deep_score() or mean_reversion.score() ever run
on it: `bot_engine.py`'s `scan_market()` quick-score pass hard-filters the
scanned universe on `c < MIN_PRICE or v < MIN_VOLUME` (bot_engine.py:2746,
using the live daily snapshot's close/volume) — a candidate that fails this
gate is invisible to the live bot regardless of what any strategy would have
scored it. Steps 1-4 never checked whether the pinned candidate lists could
pass this gate; this script asks that question directly, using the SAME
constants scan_market() itself uses (imported live from bot_engine, not
copied) so this check cannot silently drift from the live filter.

NO STRATEGY/THRESHOLD/CONFIG CHANGE SHIPS FROM THIS SCRIPT — pure read-only
diagnostic over live market data.

METHOD: for each pinned ticker (imported from illiquid_universe_probe.py,
not re-typed, so this can never silently diverge from the exact candidates
steps 1-4 evaluated), fetch recent daily bars via `backtest_v2.fetch_bars`
(Alpaca-first/Yahoo-fallback — same data source class scan_market's own
Alpaca snapshot approximates) and check the latest close against MIN_PRICE
and the trailing-20-trading-day average volume against MIN_VOLUME. A
20-day average is used instead of a single day's volume (scan_market
checks a single day) because it is a fairer, less noise-prone read of
"would this name typically clear the gate" — using a single low-volume day
would only make the illiquid group look WORSE, so this is the more
generous of the two readings, not a stacked-deck comparison.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot_engine import MIN_PRICE, MIN_VOLUME  # noqa: E402 — live constants, not copies
import backtest_v2  # noqa: E402
import illiquid_universe_probe as orig  # noqa: E402 — reuse pinned ticker lists verbatim

ILLIQUID = orig.ILLIQUID
MODERATE = orig.MODERATE
LIQUID = orig.LIQUID

AVG_VOLUME_WINDOW_DAYS = 20
FETCH_DAYS = 45  # a few extra calendar days of buffer over the 20 trading days needed


def passes_universe_gate(last_close, avg_volume, min_price=MIN_PRICE, min_volume=MIN_VOLUME):
    """Mirrors bot_engine.scan_market()'s `c < MIN_PRICE or v < _min_vol` skip
    condition exactly (inverted to a pass/fail boolean)."""
    if last_close is None or avg_volume is None:
        return False
    return last_close >= min_price and avg_volume >= min_volume


def check_ticker(ticker, fetch_fn=None, days=FETCH_DAYS, window=AVG_VOLUME_WINDOW_DAYS):
    """Fetch bars for `ticker` and evaluate the universe gate. `fetch_fn`
    defaults to backtest_v2.fetch_bars (injectable for offline tests)."""
    fetch_fn = fetch_fn or backtest_v2.fetch_bars
    bars = fetch_fn(ticker, days)
    closes = (bars or {}).get("close") or []
    volumes = (bars or {}).get("volume") or []
    if not closes or not volumes:
        return {"ticker": ticker, "last_close": None, "avg_volume": None,
                "passes_price": False, "passes_volume": False, "passes_gate": False,
                "error": "no data"}
    last_close = closes[-1]
    avg_volume = statistics.mean(volumes[-window:])
    passes_price = last_close >= MIN_PRICE
    passes_volume = avg_volume >= MIN_VOLUME
    return {
        "ticker": ticker,
        "last_close": last_close,
        "avg_volume": avg_volume,
        "passes_price": passes_price,
        "passes_volume": passes_volume,
        "passes_gate": passes_price and passes_volume,
        "error": None,
    }


def check_group(tickers, fetch_fn=None):
    rows = [check_ticker(t, fetch_fn=fetch_fn) for t in tickers]
    n = len(rows)
    n_pass = sum(1 for r in rows if r["passes_gate"])
    n_price_only = sum(1 for r in rows if r["passes_price"] and not r["passes_volume"])
    n_volume_only = sum(1 for r in rows if r["passes_volume"] and not r["passes_price"])
    return {
        "rows": rows,
        "n": n,
        "n_pass": n_pass,
        "pass_rate_pct": round(100.0 * n_pass / n, 1) if n else None,
        "n_price_only": n_price_only,
        "n_volume_only": n_volume_only,
    }


def main():
    print(f"Live universe gate: MIN_PRICE=${MIN_PRICE} MIN_VOLUME={MIN_VOLUME:,} "
          f"(imported live from bot_engine.py, not copied)\n")
    for name, tickers in (("ILLIQUID", ILLIQUID), ("MODERATE", MODERATE), ("LIQUID", LIQUID)):
        result = check_group(tickers)
        print(f"{name} (n={result['n']}):")
        for r in result["rows"]:
            if r["error"]:
                print(f"  {r['ticker']:6s} ERROR: {r['error']}")
                continue
            print(f"  {r['ticker']:6s} close={r['last_close']:>8.2f} "
                  f"avg_vol20d={r['avg_volume']:>12,.0f} "
                  f"price_ok={r['passes_price']!s:5s} vol_ok={r['passes_volume']!s:5s} "
                  f"GATE={'PASS' if r['passes_gate'] else 'FAIL'}")
        print(f"  -> {result['n_pass']}/{result['n']} pass the live universe gate "
              f"({result['pass_rate_pct']}%)\n")


if __name__ == "__main__":
    main()
