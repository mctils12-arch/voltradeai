#!/usr/bin/env python3
"""
backtest.py — minimal stub for /api/bot/backtest endpoints.

ALPHA AUDIT 2026-05-03: server/bot.ts has two endpoints that spawn
this script via subprocess and JSON-parse stdout:
  - /api/bot/backtest        (line ~2188)
  - earlier btPath usage     (line ~3724)

The full backtest engine that originally produced backtest_10yr_results.json
is not in the repo (likely lived in the workbench). Without this stub, both
endpoints crash with "ENOENT: no such file" and return HTTP 500.

This stub returns a structured response so callers see a clean "not yet
ported" message instead of a 500. Replace with a real implementation when
the workbench backtest engine is ported to production source.

Invocation (from bot.ts):
    python3 backtest.py <ticker> <strategy> <years>

Output: JSON-only on stdout. Exit 0 always (so bot.ts JSON.parse succeeds).
"""
import json
import os
import sys


def main():
    ticker  = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    strategy = sys.argv[2] if len(sys.argv) > 2 else "all"
    years   = sys.argv[3] if len(sys.argv) > 3 else "3"

    # If the existing static backtest results file is available and the
    # caller is asking for a generic summary, serve a slice from it. This
    # at least gives the UI something to display rather than nothing.
    static_path = os.path.join(os.path.dirname(__file__),
                                "backtest_10yr_results.json")
    static_summary = None
    if os.path.exists(static_path):
        try:
            with open(static_path) as f:
                bt = json.load(f)
            results = bt.get("results", {})
            static_summary = {
                "period_start":   bt.get("period_start"),
                "period_end":     bt.get("period_end"),
                "spy_cagr":       bt.get("spy_cagr"),
                "configs": [
                    {
                        "name":   k,
                        "cagr":   v.get("cagr"),
                        "max_dd": v.get("max_dd"),
                        "sortino": v.get("sortino"),
                        "n_trades": len(bt.get("all_trades", {}).get(k, [])),
                    }
                    for k, v in results.items()
                ] if isinstance(results, dict) else [],
            }
        except Exception:
            static_summary = None

    response = {
        "ok": True,
        "status": "stub",
        "message": (
            "Live backtest engine not yet ported from workbench. "
            "Returning static 10-year results from "
            "backtest_10yr_results.json. Request the static file "
            "directly for trade-level data."
        ),
        "request": {"ticker": ticker, "strategy": strategy, "years": years},
        "results": [],  # bot.ts looks for `results` array — empty is safe
        "static_summary": static_summary,
    }
    # bot.ts does JSON.parse(stdout.trim()) — output a single JSON object only
    sys.stdout.write(json.dumps(response))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
