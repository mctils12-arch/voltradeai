#!/usr/bin/env python3
"""
un_comtrade_gate1.py — ROOT VALIDATION LADDER GATE 1 (DATA) for the
un_comtrade_bilateral_trade root: reconciles the archived import (M-flow)
CIF values in datacore/un_comtrade/bilateral_trade.json against an
INDEPENDENT official U.S. government source, FRED's Census-Bureau-sourced
customs-basis import series (keyless CSV, fred.stlouisfed.org/graph/
fredgraph.csv?id=<SERIES>) — same "external truth source" discipline as
JODI-vs-EIA (jodi_eia_reconcile.py) and DTCC-vs-checksum-standards.

PRE-REGISTERED PRIOR (stated before this script was run for real numbers,
REASONING STANDARD #10): UN Comtrade's cifvalue is a CIF (cost+insurance+
freight) import valuation; FRED/Census import series are on a customs
(≈FOB-equivalent) basis. Textbook trade-statistics literature puts the
CIF/FOB markup for U.S. imports at roughly 3-11% depending on partner/
mode mix. So the PASS bar is deliberately NOT "near-zero difference" (that
would be the wrong bar and a false failure) — it is:
  (a) every partner's CIF/customs ratio has a MEAN in [1.00, 1.20] (CIF
      can only be >= the customs-basis value: insurance+freight only add
      cost, so a ratio < 1.00 would itself be a red flag); and
  (b) every partner's ratio has a STANDARD DEVIATION across months <
      STABILITY_BAND — a real accounting-convention offset should be
      stable over time; a data-integrity problem (wrong country matched,
      unit error, stale row) would show up as an unstable/noisy ratio.
(a) checks accuracy, (b) checks the offset is a real structural constant
and not coincidental/noisy — the same "stable ratio = real signal, noisy
ratio = artifact" reasoning this repo's own port-dwell/DTCC gate-1 work
already uses elsewhere.

Usage:
  python3 scripts/un_comtrade_gate1.py [--archive PATH] [--json]
Exit code: 0 = every partner passes both bars, 1 = at least one fails.
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from un_comtrade_ingest import PARTNERS  # noqa: E402  (reuse the single source of truth)

ARCHIVE = os.path.join(os.path.dirname(__file__), "..", "datacore", "un_comtrade", "bilateral_trade.json")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
MEAN_RATIO_MIN = 1.00
MEAN_RATIO_MAX = 1.20
STABILITY_BAND = 0.05  # stdev of the monthly ratio, in ratio units (0.05 = 5 points)
FRED_FETCH_RETRIES = 3
INTER_PARTNER_DELAY_S = 20


def fetch_fred_series(series_id: str) -> dict:
    """FRED fredgraph.csv -> {"YYYY-MM": value_in_dollars}. FRED reports in
    millions of dollars; converted here so callers compare like-for-like
    against Comtrade's raw-dollar cifvalue.

    Shells out to `curl --http1.1` rather than using urllib.request/
    requests: both live-verified this session (repeatedly, alternating
    between hanging and working on IDENTICAL requests moments apart) to
    intermittently stall to a read timeout against this specific host
    through this sandbox's egress proxy (comtradeapi.un.org, fetched by
    un_comtrade_ingest.py's urllib.request, was reliable throughout — this
    is host-specific, not a general proxy failure). Root cause isolated by
    curl's own error text: plain `curl` (HTTP/2, curl's default over TLS)
    intermittently failed with "HTTP/2 stream 1 was not closed cleanly:
    INTERNAL_ERROR" against this proxy+host combination; `curl --http1.1`
    against the identical URL was consistently faster and more reliable
    across many direct trials this session (typically ~0.2-0.4s), though
    NOT perfectly deterministic either (one bare `curl --http1.1` call
    still hit a 30s stall) — this looks like a genuinely intermittent
    proxy/host issue, not a fully fixable client-side bug, so this function
    retries with backoff rather than assuming --http1.1 alone guarantees
    success. curl is already a hard runtime dependency of this repo's own
    tooling (jodi_oil.py, gem_ingest.py) and this sandbox's own environment
    docs reach for it as the reliability fallback — no new dependency
    added."""
    url = FRED_URL.format(series=series_id)
    last_err = None
    for attempt in range(1, FRED_FETCH_RETRIES + 1):
        try:
            proc = subprocess.run(
                ["curl", "-fsS", "--http1.1", "--max-time", "15", "-H", "User-Agent: voltradeai-datacore/1.0", url],
                capture_output=True, text=True, timeout=20,
            )
            if proc.returncode == 0:
                return parse_fred_csv(proc.stdout)
            last_err = RuntimeError(f"curl exit {proc.returncode}: {proc.stderr.strip()}")
        except subprocess.TimeoutExpired as e:
            last_err = e
        if attempt < FRED_FETCH_RETRIES:
            time.sleep(3 * attempt)
    raise RuntimeError(f"FRED fetch for {series_id} failed after {FRED_FETCH_RETRIES} attempts: {last_err}")


def parse_fred_csv(text: str) -> dict:
    """Pure parser (no network) — 'observation_date,<SERIES>\\nYYYY-MM-DD,value\\n...'
    Millions-of-dollars values converted to raw dollars. '.' (FRED's own
    missing-value marker) is skipped, never coerced to 0."""
    out = {}
    r = csv.reader(io.StringIO(text))
    header = next(r, None)
    if not header or header[0] != "observation_date":
        raise ValueError(f"unexpected FRED CSV header: {header!r}")
    for row in r:
        if len(row) < 2:
            continue
        d, val = row[0], row[1]
        if val in (".", ""):
            continue
        period = d[:4] + d[5:7]  # YYYY-MM-DD -> YYYYMM, matches Comtrade's period key
        out[period] = float(val) * 1_000_000
    return out


def compute_ratios(comtrade_points: list, fred_series: dict) -> list:
    """comtrade_points: [[period, cif, fob], ...] for one partner's M-flow
    series. Returns [(period, ratio), ...] for every period present in
    BOTH sources with a non-null cifvalue — periods only one side has
    (Comtrade ahead of FRED's own publication lag, or vice versa) are
    silently excluded, not treated as a mismatch."""
    out = []
    for period, cif, _fob in comtrade_points:
        if cif is None or period not in fred_series or fred_series[period] == 0:
            continue
        out.append((period, cif / fred_series[period]))
    return out


def evaluate_partner(ratios: list) -> dict:
    if not ratios:
        return {"n": 0, "mean": None, "stdev": None, "pass": False, "reason": "no overlapping periods"}
    vals = [r for _, r in ratios]
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    stdev = variance ** 0.5
    mean_ok = MEAN_RATIO_MIN <= mean <= MEAN_RATIO_MAX
    stable_ok = stdev < STABILITY_BAND
    passed = mean_ok and stable_ok
    reason = "OK" if passed else (
        f"mean {mean:.4f} outside [{MEAN_RATIO_MIN},{MEAN_RATIO_MAX}]" if not mean_ok
        else f"stdev {stdev:.4f} >= {STABILITY_BAND} (unstable)"
    )
    return {"n": n, "mean": mean, "stdev": stdev, "pass": passed, "reason": reason}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with open(args.archive) as f:
        archive = json.load(f)

    results = {}
    all_pass = True
    for i, (code, (name, fred_id)) in enumerate(PARTNERS.items()):
        if i > 0:
            # Live-observed this session: isolated one-off requests to
            # fred.stlouisfed.org succeeded reliably, while this loop's own
            # tight back-to-back requests (even with per-request retry)
            # failed uniformly across 3 separate full runs — a TLS
            # handshake completes but the server then never responds
            # (curl exit 28, 0 bytes, full 15s timeout on every retry).
            # That signature (handshake OK, response silently withheld)
            # is consistent with a request-rate-based block on the origin
            # or an intermediary, not a general connectivity failure —
            # spacing requests out is the fix that matches the evidence,
            # not a longer per-request timeout.
            time.sleep(INTER_PARTNER_DELAY_S)
        print(f"  fetching {name} ({fred_id}) ...", file=sys.stderr, flush=True)
        series = archive["series"].get(f"{code}|M")
        if not series:
            results[name] = {"n": 0, "pass": False, "reason": "no M-flow series in archive"}
            all_pass = False
            continue
        try:
            fred = fetch_fred_series(fred_id)
        except Exception as e:  # noqa: BLE001 - a live external fetch, surface any failure per-partner
            results[name] = {"n": 0, "pass": False, "reason": f"FRED fetch failed: {e}"}
            all_pass = False
            continue
        ratios = compute_ratios(series["points"], fred)
        ev = evaluate_partner(ratios)
        ev["fred_series"] = fred_id
        results[name] = ev
        all_pass = all_pass and ev["pass"]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"GATE 1 (DATA) — UN Comtrade CIF import value vs FRED/Census customs-basis import value")
        print(f"pre-registered bar: mean ratio in [{MEAN_RATIO_MIN},{MEAN_RATIO_MAX}], stdev < {STABILITY_BAND}\n")
        for name, ev in results.items():
            status = "PASS" if ev["pass"] else "FAIL"
            if ev.get("mean") is not None:
                print(f"  [{status}] {name:15s} n={ev['n']:2d} mean_ratio={ev['mean']:.4f} stdev={ev['stdev']:.4f}  ({ev['reason']})")
            else:
                print(f"  [{status}] {name:15s} {ev['reason']}")
        print(f"\nVERDICT: {'GATE 1 PASS (all partners)' if all_pass else 'GATE 1 FAIL (see above)'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
