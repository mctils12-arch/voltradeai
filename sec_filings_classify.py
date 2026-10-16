"""Lazy Prices PR 2b — classify EDGAR report filers as operating vs non-operating.

WHY: PR 2a measured that 72.8% of 2010-Q1 10-K/10-Q filers are absent from
today's company_tickers.json, but the index also holds filers that never had
an equity ticker (ABS/auto-lease trusts, commodity pools, blank-check shells).
Those are not survivorship loss, so the clean number needs a per-CIK class.
SOURCE: page 1 of data.sec.gov/submissions/CIK##########.json (sic,
entityType, tickers, exchanges, formerNames). One request per CIK; the
profile is cached (it is a CURRENT snapshot, stamped with fetch date).

HONEST LIMIT: submissions carries only today's SIC/tickers. A delisted company
keeps its last SIC and has an empty ticker list; a ticker was never recorded
point-in-time. This module classifies COMPANY KIND, nothing about prices.
Stdlib + sec_filings_text only; no trading imports (SPINOUT-READY).
CLI:  python3 sec_filings_classify.py sample --universe u.json --quarters 2010-Q1,2018-Q1 --n 300
"""
import argparse
import json
import os
import random
import sys
from datetime import date
from typing import Optional

import sec_filings_text as sft

# SIC codes whose registrants are vehicles, not operating businesses.
# 6189 asset-backed securities, 6770 blank check, 6221 commodity contracts
# brokers/pools, 6726 unit investment trusts/closed-end/other investment
# offices, 6722 management investment offices (open-end funds).
NON_OPERATING_SIC = frozenset({"6189", "6770", "6221", "6726", "6722"})
KIND_OPERATING, KIND_VEHICLE, KIND_UNKNOWN = "operating", "vehicle", "unknown"


def profile_path(archive: str, cik: str) -> str:
    return os.path.join(sft.stream_root(archive), "profiles", f"{sft.cik10(cik)}.json.gz")


def slim_profile(doc: dict, fetched: str) -> dict:
    return {"cik": sft.cik10(doc.get("cik", "")), "name": doc.get("name"),
            "entityType": doc.get("entityType") or None, "sic": (doc.get("sic") or "").strip() or None,
            "sicDescription": doc.get("sicDescription") or None,
            "tickers": list(doc.get("tickers") or []), "exchanges": list(doc.get("exchanges") or []),
            "formerNames": [fn.get("name") for fn in (doc.get("formerNames") or []) if fn.get("name")],
            "fetched": fetched}


def classify(profile: dict) -> str:
    """vehicle: known non-operating SIC. operating: has a real SIC (and not a
    vehicle). unknown: no SIC at all (never trusted as operating)."""
    sic = profile.get("sic")
    if not sic or sic in ("0", "0000"):
        return KIND_UNKNOWN
    return KIND_VEHICLE if sic in NON_OPERATING_SIC else KIND_OPERATING


def get_profile(cik: str, client: sft.EdgarClient, archive: str, today: Optional[date] = None) -> dict:
    path = profile_path(archive, cik)
    if os.path.exists(path):
        return json.loads(sft.read_gz(path))
    doc = client.get_json(sft.SUBMISSIONS_URL.format(cik10=sft.cik10(cik)))
    prof = slim_profile(doc, (today or date.today()).isoformat())
    prof["cik"] = sft.cik10(cik)
    sft.write_once_gz(path, json.dumps(prof).encode())
    return prof


def quarter_ciks(universe_quarter_rows: list) -> set:
    return {r["cik"] for r in universe_quarter_rows}


def survivorship_by_kind(ciks: list, profiles: dict, current_ciks: set) -> dict:
    """Among a quarter's filers: share absent from today's ticker file, overall
    and among operating companies only (the clean survivorship figure)."""
    out = {"n": len(ciks)}
    for kind in (KIND_OPERATING, KIND_VEHICLE, KIND_UNKNOWN):
        sel = [c for c in ciks if classify(profiles[c]) == kind]
        gone = [c for c in sel if c not in current_ciks]
        out[kind] = {"n": len(sel), "not_current": len(gone),
                     "not_current_share": round(len(gone) / len(sel), 4) if sel else None}
    return out


def main(argv=None) -> int:
    from sec_filings_universe import fetch_quarter, current_filer_ciks
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--quarters", default="2010-Q1,2014-Q1,2018-Q1,2022-Q1,2025-Q1")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=20261016)
    s.add_argument("--archive", default=None)
    s.add_argument("--rate", type=float, default=sft.DEFAULT_RATE_PER_SEC)
    a = ap.parse_args(argv)
    sft.process_limiter().set_rate(a.rate)
    archive = a.archive or sft.archive_base_dir()
    client = sft.EdgarClient()
    cur = current_filer_ciks(client)
    res = {}
    for qs in a.quarters.split(","):
        y, q = qs.split("-Q")
        rows, _ = fetch_quarter(int(y), int(q), client, archive)
        ciks = sorted({r["cik"] for r in rows})
        pick = sorted(random.Random(f"{a.seed}-{qs}").sample(ciks, min(a.n, len(ciks))))
        profs = {c: get_profile(c, client, archive) for c in pick}
        res[qs] = survivorship_by_kind(pick, profs, cur)
        res[qs]["population"] = len(ciks)
        print(qs, json.dumps(res[qs]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
