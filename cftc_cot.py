"""
CFTC Commitments of Traders (COT) pipeline for VolTradeAI.
Free, no API key: CFTC's public Socrata "Legacy Futures Only" dataset
(publicreporting.cftc.gov, resource 6dca-aqww), published weekly (Fridays,
positions as of the prior Tuesday).

EDGE DOCTRINE #1 (BUILD DATA, DON'T BUY IT) standing example. This module
is DATA-LADDER GATE 1 ONLY: it fetches, validates, and archives commercial
(hedger) vs. non-commercial (speculator) futures positioning. The derived
COT Index below is a standard descriptive transform, NOT a validated
trading signal — gate 2 (does it predict forward returns better than a
random-entry base rate?) is open, see research/open_questions.md. Nothing
here is wired into deep_score/tier decisions yet.

Caveat for future gate-2 work: the "Legacy" report's commercial/
non-commercial split is the traditional hedger/speculator classification;
CFTC's newer "Disaggregated" and "TFF" (Traders in Financial Futures)
reports use finer categories (producer/merchant, swap dealer, money
manager, etc.) that may separate signal better for financial contracts
(equity index, treasuries) — worth an A/B before believing legacy-report
results on those two.
"""
import json
import os
import time

import requests

from storage_config import COT_ARCHIVE_PATH, COT_CHECKPOINT_PATH

USER_AGENT = "VolTradeAI research@voltradeai.com"
SOCRATA_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# Contract code chosen by cross-checking exact market_and_exchange_names +
# cftc_contract_market_code against the live dataset (2026-07-04) so we
# pull the single actively-reporting contract, not a same-named legacy/
# mini variant. Natural gas deliberately omitted: CFTC's classic NYMEX
# "NATURAL GAS" contract (023651) stopped reporting in 2022 and the
# successor code is ambiguous from the API alone — revisit rather than
# guess.
SYMBOLS = {
    "GLD":   {"code": "088691", "name": "GOLD - COMMODITY EXCHANGE INC."},
    "SLV":   {"code": "084691", "name": "SILVER - COMMODITY EXCHANGE INC."},
    "USO":   {"code": "067651", "name": "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE"},
    "CORN":  {"code": "002602", "name": "CORN - CHICAGO BOARD OF TRADE"},
    "TLT":   {"code": "020601", "name": "UST BOND - CHICAGO BOARD OF TRADE"},
    "SPY":   {"code": "13874A", "name": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"},
    "QQQ":   {"code": "209742", "name": "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE"},
}

# How much weekly history to hold per symbol for the COT-index lookback
# (standard industry convention uses up to a 3-year/156-week window).
LOOKBACK_WEEKS = 156

# Only hit the network once per this many seconds even if called more
# often (Tier 3 runs hourly; COT only updates weekly) — makes repeated
# invocation effectively free.
MIN_CHECK_INTERVAL_SEC = 20 * 3600

# Accounting identities hold exactly in CFTC's own data; allow a small
# tolerance for the rare report revision rounding artifact.
_TOLERANCE = 5


def _to_int(row, key):
    try:
        return int(float(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def validate_record(row):
    """
    GATE 1 (DATA): verify a fetched record against CFTC's own accounting
    identities before trusting it — catches field-mapping bugs and upstream
    data corruption deterministically, per-record, with no external ground
    truth needed beyond the report's internal arithmetic.

      noncomm_long + noncomm_spread + comm_long  == tot_rept_positions_long_all
      noncomm_short + noncomm_spread + comm_short == tot_rept_positions_short
      tot_rept_positions_long_all + nonrept_long  == open_interest_all
      tot_rept_positions_short + nonrept_short    == open_interest_all

    Returns (is_valid, reason).
    """
    noncomm_long = _to_int(row, "noncomm_positions_long_all")
    noncomm_short = _to_int(row, "noncomm_positions_short_all")
    noncomm_spread = _to_int(row, "noncomm_postions_spread_all")  # CFTC's own typo
    comm_long = _to_int(row, "comm_positions_long_all")
    comm_short = _to_int(row, "comm_positions_short_all")
    tot_long = _to_int(row, "tot_rept_positions_long_all")
    tot_short = _to_int(row, "tot_rept_positions_short")
    nonrept_long = _to_int(row, "nonrept_positions_long_all")
    nonrept_short = _to_int(row, "nonrept_positions_short_all")
    open_interest = _to_int(row, "open_interest_all")

    if open_interest <= 0:
        return False, "open_interest_all is zero/missing"

    checks = [
        (abs((noncomm_long + noncomm_spread + comm_long) - tot_long), "long-side reported total"),
        (abs((noncomm_short + noncomm_spread + comm_short) - tot_short), "short-side reported total"),
        (abs((tot_long + nonrept_long) - open_interest), "long-side open interest"),
        (abs((tot_short + nonrept_short) - open_interest), "short-side open interest"),
    ]
    for delta, label in checks:
        if delta > _TOLERANCE:
            return False, f"{label} mismatch by {delta}"
    return True, "ok"


def _derive_fields(row):
    noncomm_long = _to_int(row, "noncomm_positions_long_all")
    noncomm_short = _to_int(row, "noncomm_positions_short_all")
    comm_long = _to_int(row, "comm_positions_long_all")
    comm_short = _to_int(row, "comm_positions_short_all")
    open_interest = _to_int(row, "open_interest_all")

    net_noncomm = noncomm_long - noncomm_short
    net_comm = comm_long - comm_short
    return {
        "report_date": row.get("report_date_as_yyyy_mm_dd", "")[:10],
        "open_interest": open_interest,
        "noncomm_long": noncomm_long,
        "noncomm_short": noncomm_short,
        "comm_long": comm_long,
        "comm_short": comm_short,
        "net_noncomm": net_noncomm,
        "net_comm": net_comm,
        "net_noncomm_pct_oi": round(net_noncomm / open_interest * 100, 2) if open_interest else 0.0,
        "net_comm_pct_oi": round(net_comm / open_interest * 100, 2) if open_interest else 0.0,
    }


def fetch_symbol_history(contract_code, limit=LOOKBACK_WEEKS):
    """Fetch up to `limit` most recent weekly reports for one CFTC contract
    code. Returns validated, derived records sorted oldest -> newest.
    Raises requests.RequestException on network failure (caller decides
    whether that's fatal for the whole update)."""
    params = {
        "$where": f"cftc_contract_market_code='{contract_code}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": limit,
    }
    resp = requests.get(
        SOCRATA_URL, params=params,
        headers={"User-Agent": USER_AGENT}, timeout=20,
    )
    resp.raise_for_status()
    rows = resp.json()

    out = []
    rejected = 0
    for row in rows:
        ok, reason = validate_record(row)
        if not ok:
            rejected += 1
            continue
        out.append(_derive_fields(row))
    out.sort(key=lambda r: r["report_date"])
    return out, rejected


def _cot_index(values, lookback):
    """Standard Larry-Williams-style COT index: where the current reading
    sits (0-100) between the min/max of the trailing `lookback` window.
    Descriptive only — see module docstring re: gate 2."""
    if not values:
        return None
    window = values[-lookback:]
    lo, hi = min(window), max(window)
    if hi == lo:
        return 50.0
    return round((values[-1] - lo) / (hi - lo) * 100, 1)


def attach_cot_index(records, lookback=LOOKBACK_WEEKS):
    noncomm_series, comm_series = [], []
    for rec in records:
        noncomm_series.append(rec["net_noncomm"])
        comm_series.append(rec["net_comm"])
        rec["cot_index_noncomm"] = _cot_index(noncomm_series, lookback)
        rec["cot_index_comm"] = _cot_index(comm_series, lookback)
    return records


def load_archive():
    try:
        with open(COT_ARCHIVE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_archive(archive):
    tmp_path = COT_ARCHIVE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(archive, f)
    os.replace(tmp_path, COT_ARCHIVE_PATH)


def _load_last_check():
    try:
        with open(COT_CHECKPOINT_PATH) as f:
            return json.load(f).get("last_check_ts", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def _save_last_check(ts):
    with open(COT_CHECKPOINT_PATH, "w") as f:
        json.dump({"last_check_ts": ts}, f)


def run_daily_update(force=False):
    """
    Entry point wired into the bot's existing hourly Tier 3 maintenance
    call (server/bot.ts tier3Strategic). Actually hits the network at
    most once per MIN_CHECK_INTERVAL_SEC regardless of call frequency —
    safe (and nearly free) to invoke every hour.

    For each tracked symbol: backfills up to LOOKBACK_WEEKS of validated
    history on first run, or appends any new weekly report thereafter.
    Recomputes the COT index over the full archived series each time.
    """
    now = time.time()
    last_check = _load_last_check()
    if not force and (now - last_check) < MIN_CHECK_INTERVAL_SEC:
        return {"status": "skipped", "reason": "checked recently", "age_hours": round((now - last_check) / 3600, 1)}

    archive = load_archive()
    results = {}
    any_success = False
    for symbol, meta in SYMBOLS.items():
        try:
            existing = archive.get(symbol, [])
            latest_known_date = existing[-1]["report_date"] if existing else None
            fetch_limit = LOOKBACK_WEEKS if not existing else 12  # small trailing buffer once backfilled
            fresh, rejected = fetch_symbol_history(meta["code"], limit=fetch_limit)

            merged = {r["report_date"]: r for r in existing}
            for r in fresh:
                merged[r["report_date"]] = r
            combined = [merged[d] for d in sorted(merged.keys())][-LOOKBACK_WEEKS:]
            combined = attach_cot_index(combined)

            archive[symbol] = combined
            any_success = True
            results[symbol] = {
                "status": "ok",
                "weeks_held": len(combined),
                "latest_report_date": combined[-1]["report_date"] if combined else None,
                "new_since_last_check": combined[-1]["report_date"] != latest_known_date if combined else False,
                "rejected_records": rejected,
            }
        except requests.RequestException as e:
            results[symbol] = {"status": "error", "reason": str(e)[:200]}
        except Exception as e:
            results[symbol] = {"status": "error", "reason": f"unexpected: {e}"[:200]}

    if any_success:
        save_archive(archive)
    _save_last_check(now)
    return {"status": "done", "symbols": results}


def get_cot_snapshot(ticker):
    """Internal-API getter for future gate-2 consumers (unused by trading
    code today — see module docstring). Returns the latest archived
    record for `ticker`, or None if not tracked / not yet fetched."""
    archive = load_archive()
    records = archive.get(ticker.upper())
    if not records:
        return None
    return records[-1]


if __name__ == "__main__":
    print(json.dumps(run_daily_update(force=True), indent=2))
