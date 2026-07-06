"""
Boot-time trade_feedback cleanup — extracted from the inline python snippet
in server/bot.ts so the filter is unit-testable ([REPAIR 2026-07-06], R12-D4).

WHY THIS EXISTS (and what the old inline filter got wrong):
  old filter: keep if code_version-string >= '1.0.33' and ticker non-empty.
  1. SEED WIPE-OUT: backtest seed records (_seed: True) carry no
     code_version -> '' fails the compare -> every deploy deleted all
     Kelly-prior seeds; the daemon's autoseed only refills under 100
     records, and 500 April fossils held the count above that. Seeds are
     now kept unconditionally.
  2. LEXICOGRAPHIC VERSION BUG: '1.0.153' < '1.0.34' as STRINGS. Any
     record ever stamped with a real 3-digit patch version would be
     silently wiped at the next boot. Versions are now compared as
     numeric tuples.
  3. FOSSIL RETENTION: the 2026-04-16..20 records written by the
     pre-Bug-25b trackClosedTrades variant (code_version 1.0.33,
     pnl_pct 0, outcome missing, entry_features null) are exactly the
     broken-code artifacts this cleanup exists to remove, yet 1.0.33
     passed the old floor. The floor is now (1,0,34) — the version whose
     writers are trusted (track_fill entry records + post-Bug-25b
     feedback records both stamp 1.0.34).

MEASUREMENT-INTEGRITY NOTE: raising the floor removes 500 pnl-0
outcome-less records that (a) blocked reseeding, (b) were matchable by
_find_entry_record on ticker collision (mislabel risk for R12-D2's exit
recording). It cannot make live performance LOOK better: those records
were already excluded from every performance metric (pnl_pct==0 with
outcome None is filtered by check_model_health and the dashboard).
"""

# Records written by code at or above this version are trusted.
MIN_TRUSTED_VERSION = (1, 0, 34)


def parse_version(v) -> tuple:
    """'1.0.153' -> (1, 0, 153); non-string/unparseable -> (0,) (fails floor).

    Strings only: code_version is always written as a string; a numeric or
    None value is data corruption and must not accidentally pass the floor
    (e.g. 42.5 -> (42, 5) would outrank every real version)."""
    if not isinstance(v, str):
        return (0,)
    try:
        parts = tuple(int(x) for x in v.strip().split("."))
        return parts if parts else (0,)
    except ValueError:
        return (0,)


def keep_record(rec) -> bool:
    if not isinstance(rec, dict):
        return False
    if rec.get("_seed"):
        # Backtest seeds are the Kelly-gate priors — boot must never
        # destroy them (R12-D4; they carry no code_version by design).
        return True
    if not str(rec.get("ticker", "")).strip():
        return False
    return parse_version(rec.get("code_version")) >= MIN_TRUSTED_VERSION


def clean_feedback(records: list) -> tuple:
    """Returns (valid_records, removed_count)."""
    if not isinstance(records, list):
        return [], 0
    valid = [r for r in records if keep_record(r)]
    return valid, len(records) - len(valid)
