#!/usr/bin/env python3
"""
One-time cleanup script for corrupted trade feedback records.
Run on Railway: python3 cleanup_feedback.py

Removes records where:
  - ticker is empty/missing
  - OR (pnl_pct is None/0 AND code_version is missing AND outcome is None)

Preserves legitimate records (those with real tickers and valid data).
"""
import json
import os
import sys

try:
    from storage_config import TRADE_FEEDBACK_PATH
except ImportError:
    TRADE_FEEDBACK_PATH = "/tmp/voltrade_trade_feedback.json"


def is_garbage(record: dict) -> bool:
    """Return True if the record should be removed."""
    ticker = str(record.get("ticker", "")).strip()
    if not ticker:
        return True

    pnl = record.get("pnl_pct")
    has_version = bool(record.get("code_version"))
    outcome = record.get("outcome")

    # Remove records with no useful trade data and no version tag
    if (pnl is None or pnl == 0) and not has_version and outcome is None:
        return True

    return False


def main():
    path = TRADE_FEEDBACK_PATH
    if not os.path.exists(path):
        print(f"No feedback file found at {path}")
        sys.exit(0)

    with open(path) as f:
        records = json.load(f)

    total = len(records)
    print(f"Loaded {total} records from {path}")

    kept = []
    removed_empty_ticker = 0
    removed_no_data = 0

    for r in records:
        ticker = str(r.get("ticker", "")).strip()
        if not ticker:
            removed_empty_ticker += 1
            continue

        pnl = r.get("pnl_pct")
        has_version = bool(r.get("code_version"))
        outcome = r.get("outcome")

        if (pnl is None or pnl == 0) and not has_version and outcome is None:
            removed_no_data += 1
            continue

        kept.append(r)

    removed_total = total - len(kept)

    print(f"\n--- Cleanup Summary ---")
    print(f"Total records:         {total}")
    print(f"Removed (empty ticker): {removed_empty_ticker}")
    print(f"Removed (no data):      {removed_no_data}")
    print(f"Total removed:          {removed_total}")
    print(f"Records kept:           {len(kept)}")

    if removed_total > 0:
        # Write cleaned data back
        with open(path, "w") as f:
            json.dump(kept, f, indent=2)
        print(f"\nCleaned file written to {path}")
    else:
        print("\nNo records to remove — file unchanged.")


if __name__ == "__main__":
    main()
