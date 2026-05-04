#!/usr/bin/env python3
"""
Seed trade_feedback.json from the 10-year backtest.

ALPHA AUDIT 2026-05-03 — pre-loads the live trading-history file with
the SUCCESSFUL subset of backtest trades so the Kelly gate has real
data to size from on day one, instead of falling back to hardcoded
defaults that may be wrong or out of date.

Subset selection (per per-instrument segmentation analysis):
  ✓ CSPs (sell_csp)        — 1,226 trades, 71.6% WR, +EV proven
  ✓ Leveraged ETFs (AAPU,
    AMU, etc.)             —   879 trades, 54.7% WR, +EV proven
  ✓ Stocks held ≥ 4 days   —   296 trades, 55.7% WR, +EV after the
                                short-hold fix in instrument_selector
                                + bot_engine
  ✗ Stocks held 1-3 days   — DROPPED (4.2% WR, -293% drag — the
                                short-hold fix prevents them in live)
  ✗ bull_spread            — DROPPED (already disabled in
                                options_execution.should_use_options)
  ✗ buy_call               — DROPPED (already disabled)

The strategy field on each seed record is set to the bucket name that
_infer_strategy() in position_sizing.py would assign — so
_get_historical_stats() finds them under the right key.

USAGE:
    python3 seed_feedback_from_backtest.py [--dry-run] [--force]

By default this is IDEMPOTENT — it skips records that already exist
in trade_feedback.json (matched by ticker+date_entry+date_exit). Use
--force to wipe and re-seed from scratch.
"""
import argparse
import json
import os
import sys

# Use the same path as production
try:
    from storage_config import DATA_DIR, TRADE_FEEDBACK_PATH
except ImportError:
    DATA_DIR = "/tmp"
    TRADE_FEEDBACK_PATH = os.path.join(DATA_DIR, "voltrade_trade_feedback.json")

BACKTEST_PATH = os.path.join(os.path.dirname(__file__), "backtest_10yr_results.json")

# Bucket mapping to match _infer_strategy() return values
def _bucket_for(t: dict) -> str | None:
    """Map a backtest trade to the strategy bucket _infer_strategy uses."""
    inst = (t.get("instrument") or "").lower()
    strat = (t.get("strategy") or "").lower()
    holding = int(t.get("holding_days", 0) or 0)

    # Drop disabled strategies
    if strat in ("bull_spread", "buy_call", "bear_put_spread"):
        return None

    # CSPs and other premium-selling options
    if strat in ("sell_csp", "covered_call", "iron_condor",
                 "bull_put_credit_spread", "qqq_iron_condor",
                 "short_straddle", "calendar_spread", "diagonal_spread"):
        return "csp_options"

    # Leveraged ETFs (the alpha generator)
    if inst == "etf":
        return "etf"

    # Stocks: only keep 4+ day holds (short-hold disaster excluded)
    if inst == "stock":
        if holding < 4:
            return None
        return "stocks"

    # Anything else: drop
    return None


def _record_id(t: dict) -> str:
    """Stable id for idempotency."""
    return f"{t.get('ticker')}|{t.get('date_entry')}|{t.get('date_exit')}|{t.get('strategy')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be written, but don't touch the file.")
    ap.add_argument("--force", action="store_true",
                    help="Wipe existing trade_feedback.json and write fresh.")
    args = ap.parse_args()

    if not os.path.exists(BACKTEST_PATH):
        print(f"ERROR: backtest file not found: {BACKTEST_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(BACKTEST_PATH) as f:
        bt = json.load(f)

    # Use OLD config — has the broadest mix (etf + stock + sell_csp).
    # CSP-only config has only sell_csp trades, so it doesn't have ETF
    # or stock buckets. OLD covers all four.
    raw_trades = bt["all_trades"]["OLD"]
    print(f"Loaded {len(raw_trades)} trades from backtest OLD config.")

    # Build seed records
    seeds = []
    drop_reasons = {"strategy_disabled": 0, "short_hold_stock": 0, "other": 0}

    for t in raw_trades:
        bucket = _bucket_for(t)
        if bucket is None:
            inst = (t.get("instrument") or "").lower()
            strat = (t.get("strategy") or "").lower()
            holding = int(t.get("holding_days", 0) or 0)
            if strat in ("bull_spread", "buy_call", "bear_put_spread"):
                drop_reasons["strategy_disabled"] += 1
            elif inst == "stock" and holding < 4:
                drop_reasons["short_hold_stock"] += 1
            else:
                drop_reasons["other"] += 1
            continue

        # Format as a feedback record. We preserve enough fields that the
        # ML training pipeline can use these too (entry_price, exit_price,
        # exit_reason, score, vrp, vxx_ratio).
        pnl_pct = float(t.get("net_pct", 0) or 0)
        seed = {
            "ticker":         t["ticker"],
            "side":           "sell" if "sell" in (t.get("strategy") or "").lower()
                              else "buy",
            "fill_price":     float(t.get("entry_price", 0) or 0),
            "exit_price":     float(t.get("exit_price", 0) or 0),
            "qty":            1,  # unit qty — only pnl_pct matters for stats
            "score":          float(t.get("score", 50) or 50),
            "vrp":            float(t.get("vrp", 0) or 0),
            "vxx_ratio":      float(t.get("vxx_ratio", 1.0) or 1.0),
            "strategy":       bucket,                # ← bucket name for Kelly lookup
            "raw_strategy":   t.get("strategy"),     # original label for traceability
            "instrument":     t.get("instrument"),
            "outcome":        "win" if pnl_pct > 0 else ("flat" if pnl_pct == 0 else "loss"),
            "pnl_pct":        round(pnl_pct, 3),
            "holding_days":   int(t.get("holding_days", 0) or 0),
            "exit_reason":    t.get("exit_reason"),
            "time_placed":    t.get("date_entry"),
            "time_filled":    t.get("date_entry"),
            "exit_time":      t.get("date_exit"),
            "session":        "regular",
            # Mark as backtest-seeded so the ML training pipeline can
            # weight/segregate them if desired
            "_seed":          True,
            "_seed_source":   "backtest_10yr_results.json",
        }
        seeds.append(seed)

    print(f"\nSeed candidates by bucket:")
    by_bucket = {}
    for s in seeds:
        by_bucket.setdefault(s["strategy"], []).append(s)
    for k in sorted(by_bucket):
        grp = by_bucket[k]
        wins = [r for r in grp if r["pnl_pct"] > 0]
        wr = len(wins) / len(grp) * 100 if grp else 0
        avg = sum(r["pnl_pct"] for r in grp) / len(grp) if grp else 0
        print(f"  {k:<14} n={len(grp):<5} WR={wr:5.1f}% avg={avg:+.3f}%")

    print(f"\nDropped (not seeded):")
    for k, v in drop_reasons.items():
        print(f"  {k:<25} {v}")

    # Load existing feedback (idempotency)
    existing = []
    if os.path.exists(TRADE_FEEDBACK_PATH) and not args.force:
        try:
            with open(TRADE_FEEDBACK_PATH) as f:
                existing = json.load(f)
            print(f"\nExisting trade_feedback.json: {len(existing)} records")
        except Exception as e:
            print(f"\nWarning: could not parse existing feedback file: {e}")
            existing = []

    if args.force and os.path.exists(TRADE_FEEDBACK_PATH):
        print(f"\n--force given: existing {TRADE_FEEDBACK_PATH} will be replaced.")

    # Idempotency: skip seeds that already exist by id
    existing_ids = {
        f"{r.get('ticker')}|{r.get('time_placed')}|{r.get('exit_time')}|{r.get('raw_strategy', r.get('strategy'))}"
        for r in existing if r.get("_seed")
    }
    new_seeds = [s for s in seeds if _record_id(
        {"ticker": s["ticker"], "date_entry": s["time_placed"],
         "date_exit": s["exit_time"], "strategy": s["raw_strategy"]}
    ) not in existing_ids]
    print(f"\nNew records to add: {len(new_seeds)}")

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return

    if args.force:
        out = new_seeds  # full reseed
    else:
        # Preserve any non-seed (live) records and any seed records already there
        out = existing + new_seeds

    os.makedirs(os.path.dirname(TRADE_FEEDBACK_PATH) or ".", exist_ok=True)

    # Atomic write
    tmp = TRADE_FEEDBACK_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=0)  # indent=0 keeps file scannable but compact
    os.replace(tmp, TRADE_FEEDBACK_PATH)
    print(f"\nWrote {len(out)} records to {TRADE_FEEDBACK_PATH}")


if __name__ == "__main__":
    main()
