#!/usr/bin/env python3
"""
backtest.py — CLI entrypoint for the backtest engine (backtest_v2.py).

Contract (server/bot.ts — POST /api/bot/backtest and the overnight scheduler):
    python3 backtest.py <ticker> <strategy> <years>
  - JSON-only on stdout (bot.ts does JSON.parse(stdout.trim()))
  - exit 0 always, even on failure (bot.ts treats nonzero/parse-fail as 500)
  - `results` is an array; the scheduler reads per-item
    strategy / sharpe / totalReturn / maxDrawdown

strategy: momentum | mean_reversion | squeeze | all   (default all)
years: 0.5 .. 10                                       (default 3)

The engine simulates the bot's equity strategy logic with live-identical
regime gating; options/squeeze legs are not simulated (no historical data —
see research/wishlist.md). Details in backtest_v2.py.
"""
import json
import sys


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    strategy = sys.argv[2] if len(sys.argv) > 2 else "all"
    years_raw = sys.argv[3] if len(sys.argv) > 3 else "3"

    try:
        years = float(years_raw)
    except ValueError:
        years = 3.0

    try:
        from backtest_v2 import run_backtest
        response = run_backtest(ticker, strategy, years)
        # Keep the endpoint payload bounded: bot.ts returns this verbatim to
        # the client. Cap per-strategy trade logs at the most recent 200.
        at = response.get("all_trades", {})
        for k in at:
            if len(at[k]) > 200:
                at[k] = at[k][-200:]
    except Exception as e:
        response = {
            "ok": False,
            "status": "error",
            "message": f"backtest failed: {type(e).__name__}: {e}",
            "request": {"ticker": ticker, "strategy": strategy, "years": years},
            "results": [],  # bot.ts iterates `results` — empty is safe
        }

    # JSON object only on stdout — nothing else may be printed
    sys.stdout.write(json.dumps(response))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
