#!/usr/bin/env python3
"""
scripts/reconstruct_position_pnl.py — independent daily P&L reconstruction
from published market closes, for auditing the account's own reported
`equity`/`last_equity` fields against reality.

WHY THIS EXISTS: KNOWN BROKEN #42 (research/open_questions.md) — Alpaca's
`last_equity` for 2026-09-09 implied a -$12,059.74 (-11.7%) same-day loss
with ZERO orders filled that day (confirmed via /api/diag/orders), which
fired the -20% portfolio drawdown kill switch and halted production
trading. Three prior sessions inferred this was likely a data anomaly from
circumstantial evidence (flat live unrealized P&L, the account round-
tripping most of the move within 24-36h with no trades) but could not
independently confirm it without Alpaca dashboard access this sandbox
lacks. This script closes that gap: given a date and the held position
quantities, it computes what the SAME positions' combined close-to-close
market value change actually was, using `backtest_v2.fetch_bars` — the
identical Alpaca-first/Yahoo-fallback data path the backtest engine uses,
entirely independent of the account's own equity bookkeeping. A real loss
and the account's reported loss should roughly agree; a data anomaly will
show reconstructed P&L far smaller than the reported figure.

SCOPE: equities and ETFs only. Options legs (OCC-format symbols, e.g.
"BAC261016P00057500") are detected and excluded — no free historical
options-quote source exists (research/wishlist.md). This is an honest gap,
not a rounding error: state it, don't paper over it. In the incident this
script was built for, both option legs were single-contract positions with
double-digit-dollar cost bases, so excluding them cannot hide a five-figure
discrepancy.

USAGE:
    python3 scripts/reconstruct_position_pnl.py --date 2026-09-09 \\
        --positions '{"QQQ": 51, "KWEB": 257, "SMH": 20, "VXUS": 133, "FCEL": 70}'

    python3 scripts/reconstruct_position_pnl.py --date 2026-09-09 \\
        --positions-file /path/to/positions.json

`positions.json` may be the raw `/api/diag/positions-detail` response (this
script pulls `symbol`/`qty` off each row and silently skips non-equity
asset classes) or a plain {symbol: qty} map.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import backtest_v2  # noqa: E402  (repo-root module, see sys.path.insert above)

# OCC option symbol: root (1-6 letters) + YYMMDD + C/P + 8-digit strike.
_OCC_OPTION_RE = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")


def is_option_symbol(symbol: str) -> bool:
    return bool(_OCC_OPTION_RE.match(symbol))


def load_positions(positions_arg: str | None, positions_file: str | None) -> dict:
    """Accepts a plain {symbol: qty} map, or a /api/diag/positions-detail-shaped
    payload ({"positions": [{"symbol": ..., "qty": ...}, ...]}). Returns a
    plain {symbol: float_qty} map, options included (caller filters)."""
    if positions_file:
        with open(positions_file) as f:
            raw = json.load(f)
    else:
        raw = json.loads(positions_arg)
    if isinstance(raw, dict) and "positions" in raw:
        return {p["symbol"]: float(p["qty"]) for p in raw["positions"]}
    return {sym: float(qty) for sym, qty in raw.items()}


def reconstruct(date: str, positions: dict, days_lookback: int = 15) -> dict:
    """For each equity symbol in `positions`, finds `date`'s close and the
    prior trading day's close in the SAME data path the backtest engine
    uses (Alpaca-first, Yahoo fallback), and sums qty * (close - prev_close)
    across the book. No lookahead concern here (this is a retrospective
    audit, not a trading decision) but the date lookup is exact-match only
    — a `date` that isn't a trading day close (e.g. weekend/holiday) simply
    won't be found and that symbol is reported excluded, rather than
    silently substituting a nearby day."""
    legs = []
    excluded_options = []
    excluded_no_data = []
    total = 0.0

    for symbol, qty in positions.items():
        if is_option_symbol(symbol):
            excluded_options.append(symbol)
            continue
        try:
            bars = backtest_v2.fetch_bars(symbol, days_lookback, use_cache=True)
        except Exception as e:
            excluded_no_data.append({"symbol": symbol, "reason": str(e)})
            continue
        if not bars or not bars.get("date") or date not in bars["date"]:
            excluded_no_data.append({"symbol": symbol, "reason": "date not in fetched bars"})
            continue
        i = bars["date"].index(date)
        if i == 0:
            excluded_no_data.append({"symbol": symbol, "reason": "no prior-day bar in lookback window"})
            continue
        close_today = bars["close"][i]
        close_prev = bars["close"][i - 1]
        delta = close_today - close_prev
        contrib = qty * delta
        total += contrib
        legs.append({
            "symbol": symbol, "qty": qty,
            "prev_close": close_prev, "close": close_today,
            "delta": round(delta, 4), "contribution": round(contrib, 2),
        })

    return {
        "date": date,
        "reconstructed_pnl": round(total, 2),
        "legs": legs,
        "excluded_options": excluded_options,
        "excluded_no_data": excluded_no_data,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD trading day to reconstruct")
    ap.add_argument("--positions", help="JSON {symbol: qty} map or positions-detail payload")
    ap.add_argument("--positions-file", help="path to a JSON file with the same shape as --positions")
    ap.add_argument("--reported-pnl", type=float, default=None,
                     help="the account's own reported same-day pnl, for a side-by-side comparison line")
    args = ap.parse_args()
    if not args.positions and not args.positions_file:
        ap.error("one of --positions or --positions-file is required")

    positions = load_positions(args.positions, args.positions_file)
    result = reconstruct(args.date, positions)
    if args.reported_pnl is not None:
        result["reported_pnl"] = args.reported_pnl
        result["gap"] = round(result["reconstructed_pnl"] - args.reported_pnl, 2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
