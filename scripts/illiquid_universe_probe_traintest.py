#!/usr/bin/env python3
"""
illiquid_universe_probe_traintest.py — LADDER PATH step 2 ("TRAIN/TEST
SPLIT") for the open_questions.md entry filed 2026-07-24 ("Does
mean_reversion have a real, exploitable edge specifically in illiquid
small-caps, or is the 2026-07-24 probe result a single-sample
artifact?"). Step 1 (INDEPENDENT RE-DRAW,
`scripts/illiquid_universe_probe_redraw.py`) already closed: an
independent, non-overlapping illiquid sample reproduced the same
direction (mean_reversion doing relatively better in illiquid than
moderate/liquid; momentum doing relatively worse) almost exactly in
magnitude. Step 2 asks a DIFFERENT question: does the SAME pinned
sample's pattern hold across TIME, or is the pooled 4-year read
dominated by one sub-period/regime?

PRIOR, stated 2026-07-28 BEFORE running (Reasoning Standard #10, also
recorded verbatim in the corresponding open_questions.md UPDATE): if
the illiquid mean_reversion edge is a real structural effect (thin
order books mechanically exhausting oversold pressure without
sustained institutional selling flow — see the open_questions.md
entry's "WHY THIS MIGHT BE REAL" paragraph, Reasoning Standard #5),
then splitting the SAME 4-year window into an early half and a late
half should show the SAME qualitative pattern (mean_reversion doing
relatively better in illiquid than moderate/liquid) in BOTH halves —
a pattern that only shows up in one half and reverses in the other
would suggest the pooled 4y result is dominated by a specific
sub-period (e.g. one regime), not a persistent structural effect.

COUNTER-PRIOR / CAVEAT stated up front (Reasoning Standard #4): the
pooled illiquid mean_reversion sample already has a small trade count
(147 trades over 4y across 10 tickers per the 2026-07-24 finding) —
splitting the window in roughly half necessarily roughly halves that
per-half, and the study is 10/7/7-name groups to begin with. Each half
has enough width to be evaluable, but this is emphatically NOT a
large-sample regime; a weak or reversed half-pattern is only weak
evidence against the structural story, not proof against it, and a
confirmed half-pattern is likewise not proof FOR it — it only clears
this one gate. Step 3 (bootstrap CI / significance test) is the actual
significance check and is explicitly OUT OF SCOPE for this script/PR
(CLAUDE.md PROMOTION RULES #5 — one logical change per PR).

METHOD: fits nothing new — the strategy logic (`momentum.py`,
`mean_reversion.py`) is completely unchanged from the pooled-window
version. This script only changes what slice of history each ticker's
own bars feed into `backtest_v2.run_backtest()`:
  - EARLY half: bars[0:mid] where mid = len(bars)//2. This exactly
    matches how the original pooled run already starts from index 0
    with the same natural warmup-through-zero-score behavior
    (`score_candidate` requires i>=252 for momentum, i>=20 for
    mean_reversion before it returns a nonzero score) — no new
    lookahead is introduced by this half.
  - LATE half: bars[max(0, mid-warmup_days):] — this PREPENDS up to
    `warmup_days` (default 252, matching momentum's own warmup) days
    of history immediately BEFORE the nominal midpoint, so
    `score_candidate` has a real (nonzero) signal from near the start
    of the late window too, instead of silently under-trading its
    first ~252 days the way a naive bars[mid:] slice would. Every
    prepended day is strictly BEFORE the late window's nominal start
    date — this is standard walk-forward slicing (warm up a model on
    history that already happened, then evaluate on what comes after)
    and introduces ZERO lookahead (Reasoning Standard #7): flagging
    this explicitly because a careless reviewer could mistake "the
    late window's data starts earlier than its midpoint" for peeking
    forward, when it is exactly the opposite — every extra bar in the
    late half is in the late half's own past, not its future.

SPY/VXX regime lookup (`simulate()` in backtest_v2.py) is by DATE
STRING against `spy["date"]`, not by array index, so SPY/VXX do NOT
need to be sliced to match a ticker's sliced bars — this script fetches
SPY and VXX ONCE each (full 4y window) and passes the SAME full-window
dict into both the early and late `run_backtest()` calls for every
ticker. Only each ticker's OWN OHLCV arrays are split.

SCOPE DECISION (deliberate — CLAUDE.md READ BEFORE WRITE / no silent
scope creep): this script runs the split ONLY against the ORIGINAL
2026-07-24 pinned ILLIQUID/MODERATE/LIQUID lists
(`illiquid_universe_probe.ILLIQUID/MODERATE/LIQUID`), not against
`illiquid_universe_probe_redraw.py`'s sample. The redraw script draws
live from NASDAQ's daily-changing symbol directory (non-deterministic
across days/runs) — pooling it into this split would conflate two
different underlying draws' regime exposure into one train/test
question, muddying exactly the "does the SAME sample's pattern hold
across time" question step 2 is asking. A future session COULD re-run
this identical split methodology against the redraw's own sample
independently (see open_questions.md NEXT) — that is a distinct,
not-yet-built follow-up, not something this script attempts.

Run with: python3 scripts/illiquid_universe_probe_traintest.py
(network calls: 1 Yahoo/Alpaca fetch per ticker (21 total, ILLIQUID
10 + MODERATE 7 + LIQUID 7) + 1 SPY fetch + 1 VXX fetch, shared across
both halves and all tickers — no API key required, same as the two
prior probe scripts. Takes a few minutes: 21 tickers x 2 halves x 2
strategies = 84 backtest runs.)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import illiquid_universe_probe as orig  # noqa: E402

YEARS = orig.YEARS  # 4
WARMUP_DAYS = 252  # matches momentum.py's own 252-day lookback (score_candidate)
_BAR_FIELDS = ("date", "open", "high", "low", "close", "volume")
MIN_BARS_FOR_SPLIT = 4  # below this, "early"/"late" stop meaning anything


def split_bars(bars: dict, warmup_days: int = WARMUP_DAYS) -> tuple[dict, dict]:
    """Splits one ticker's full OHLCV bars dict into (early, late) halves.

    early = bars[0:mid] — identical in shape to how the pooled 4y run
    already starts from index 0 (same natural warmup-through-zero-score
    behavior; no new lookahead).

    late = bars[max(0, mid-warmup_days):] — PREPENDS up to warmup_days
    trading days immediately before the midpoint so score_candidate has
    a real signal (not silently zero) from near the start of the late
    window too. Every prepended day is strictly BEFORE the late
    window's nominal start — standard walk-forward warmup, zero
    lookahead (Reasoning Standard #7): this is not the late half
    peeking into extra future data, it is the late half getting to see
    a little more of ITS OWN past before being scored, exactly like the
    early half already implicitly does by starting from day 0.

    Raises ValueError on bars too short to split meaningfully, rather
    than silently returning a misleading near-empty half.
    """
    missing = [k for k in _BAR_FIELDS if k not in bars]
    if missing:
        raise ValueError(f"split_bars: bars dict missing required field(s): {missing}")
    lengths = {k: len(bars[k]) for k in _BAR_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"split_bars: bar fields have mismatched lengths: {lengths}")
    n = lengths["date"]
    if n < MIN_BARS_FOR_SPLIT:
        raise ValueError(
            f"split_bars: need at least {MIN_BARS_FOR_SPLIT} bars to split "
            f"meaningfully, got {n}")

    mid = n // 2
    late_start = max(0, mid - warmup_days)

    early = {k: bars[k][:mid] for k in _BAR_FIELDS}
    late = {k: bars[k][late_start:] for k in _BAR_FIELDS}
    return early, late


def run_split_group(tickers, years=YEARS, warmup_days=WARMUP_DAYS):
    """Per ticker: fetch full bars ONCE, split into early/late, backtest
    both halves for momentum + mean_reversion via backtest_v2.run_backtest
    (bars/spy/vxx all injected -> hermetic, no extra network calls beyond
    the one fetch per ticker plus the shared SPY/VXX fetches). Returns
    (early_rows, late_rows) — same per-ticker row shape as
    illiquid_universe_probe.run_group, so illiquid_universe_probe.summarize
    can be reused unchanged on each half."""
    from backtest_v2 import run_backtest, fetch_bars

    days = int(years * 365) + 400  # same formula as illiquid_universe_probe.run_group
    spy_bars = fetch_bars("SPY", days)
    try:
        vxx_bars = fetch_bars("VXX", days)
    except Exception:
        vxx_bars = None  # degrade exactly like run_backtest's own live-fetch path

    early_rows, late_rows = [], []
    for t in tickers:
        try:
            bars = fetch_bars(t, days)
            early_bars, late_bars = split_bars(bars, warmup_days=warmup_days)

            early_resp = run_backtest(t, "all", years / 2, bars=early_bars,
                                       spy=spy_bars, vxx=vxx_bars)
            late_resp = run_backtest(t, "all", years / 2, bars=late_bars,
                                      spy=spy_bars, vxx=vxx_bars)
        except Exception as e:
            err_row = {"ticker": t, "error": str(e)[:200]}
            early_rows.append(err_row)
            late_rows.append(dict(err_row))
            continue

        for rows, resp, half_bars in ((early_rows, early_resp, early_bars),
                                       (late_rows, late_resp, late_bars)):
            row = {"ticker": t,
                   "buy_and_hold_pct": orig.buy_and_hold_pct(half_bars),
                   "n_bars": len(half_bars["close"])}
            for r in resp["results"]:
                if r["strategy"] in ("momentum", "mean_reversion"):
                    row[r["strategy"]] = {
                        "sharpe": r["sharpe"],
                        "total_return_pct": r["total_return_pct"],
                        "max_drawdown_pct": r["max_drawdown_pct"],
                        "n_trades": r["n_trades"],
                        "win_rate_pct": r["win_rate_pct"],
                    }
            rows.append(row)

    return early_rows, late_rows


if __name__ == "__main__":
    results = {}
    for name, tickers in [("illiquid", orig.ILLIQUID), ("moderate", orig.MODERATE),
                           ("liquid", orig.LIQUID)]:
        print(f"=== {name} (n={len(tickers)}) ===", file=sys.stderr)
        early_rows, late_rows = run_split_group(tickers)
        early_summary = orig.summarize(early_rows)
        late_summary = orig.summarize(late_rows)
        print(f"--- {name} EARLY ---", file=sys.stderr)
        print(json.dumps(early_summary, indent=2), file=sys.stderr)
        print(f"--- {name} LATE ---", file=sys.stderr)
        print(json.dumps(late_summary, indent=2), file=sys.stderr)
        results[name] = {"early": early_summary, "late": late_summary}

    print(json.dumps(results, indent=2))
