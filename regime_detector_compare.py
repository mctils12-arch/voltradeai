#!/usr/bin/env python3
"""
regime_detector_compare.py — answers a standing OPEN RESEARCH QUESTION
(research/open_questions.md, "Which regime detector (markov_regime vs.
VXX-ratio heuristics) actually predicts forward volatility better? They
currently coexist."). Pure [RESEARCH] — no trading path touched, no
parameter changed. This is the ruler being pointed at itself, not tuned:
it measures which of the two EXISTING regime signals already computed in
production correlates more with subsequent realized volatility.

WHY THIS QUESTION, WHY THIS METHOD (REASONING STANDARD #10 — prior stated
before running): `system_config.get_market_regime()` blends a Markov-chain
state (markov_regime.py, derived purely from the sign/magnitude of recent
SPY daily returns — a DIRECTION classifier) with a VXX-ratio + SPY-vs-50MA
heuristic (a volatility-level + trend classifier) via hardcoded absolute
thresholds calibrated for VXX. VIX itself IS a market-implied forward-vol
estimate, so a VIX-ratio signal correlating with subsequent REALIZED vol is
close to definitional; the Markov chain never sees volatility, only return
direction, so PRIOR: VIX-ratio should show materially higher correlation
with forward realized vol than the Markov chain's bear-probability. If the
Markov signal matches or beats it, that is the surprising finding worth
promoting.

HONEST DATA SUBSTITUTION (stated up front, not buried): live SPY/VXX price
history is not reachable from this sandbox (yfinance 429s, Stooq requires a
JS challenge, Alpha Vantage needs a paid key). FRED's public fredgraph.csv
(no key) serves SP500 (S&P 500 index close, 2015-present) and VIXCLS (CBOE
VIX close, 1990-present) instead:
  - SP500 vs SPY: near-identical daily % moves (SPY tracks the index minus
    a ~9bp expense ratio drag with no meaningful tracking error at daily
    granularity) — a safe substitute for return/regime computation.
  - VIXCLS vs VXX: NOT a safe stand-in for the exact thresholds baked into
    system_config.py (VXX is a short-term-futures ETN that decays via
    contango and swings in a different range than spot VIX) — so this
    script does NOT reproduce get_market_regime()'s absolute thresholds.
    It instead treats "VIX / its own 30-day average" as a CONCEPT-LEVEL
    stand-in for "the VXX-ratio heuristic" and compares via rank
    correlation (threshold-free), which is honest about what is and is not
    being tested. Reproducing the exact live thresholds would need real
    VXX history — filed as a follow-up in open_questions.md if this script
    is ever revisited with real VXX data.

The Markov side, by contrast, is NOT reimplemented — this script imports
the actual `markov_regime.MarkovRegime` class and its production
TRANSITION_1/TRANSITION_2 tables, so whatever "Markov" means here is
exactly what bot_engine.py calls live, not a re-derivation that could
silently drift from production behavior.

Usage:
    # 1) fetch the two source CSVs with plain curl (reliable in this sandbox
    #    — see fetch_fred_csv()'s docstring for why this isn't done in-process):
    curl -sS "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500" -o sp500.csv
    curl -sS "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS" -o vixcls.csv
    # 2) run the comparison against the fetched files:
    python3 regime_detector_compare.py --sp500-csv sp500.csv --vix-csv vixcls.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markov_regime import MarkovRegime  # noqa: E402  (real production class, not reimplemented)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
MARKOV_LOOKBACK = 25  # days of trailing returns fed to MarkovRegime per call (>=20 lets _adapt_threshold use its full window)


def parse_fred_csv_text(text: str) -> dict:
    """Pure parser: FRED fredgraph.csv text -> {date_str: float}, skipping
    '.' (FRED's missing-value marker) rows. No network, no side effects —
    the part of the fetch pipeline that is actually unit-tested."""
    out: dict[str, float] = {}
    lines = text.strip().splitlines()
    for line in lines[1:]:  # skip header row ("observation_date,<SERIES_ID>")
        parts = line.split(",")
        if len(parts) != 2:
            continue
        date_s, val_s = parts
        if val_s == "." or val_s == "":
            continue
        try:
            out[date_s] = float(val_s)
        except ValueError:
            continue
    return out


def load_fred_csv_file(path: str) -> dict:
    """Read a FRED fredgraph.csv file already fetched to disk."""
    with open(path) as f:
        return parse_fred_csv_text(f.read())


def fetch_fred_series(series_id: str, timeout: int = 20) -> dict:
    """Fetch a FRED series live as {date_str: float}. No key required —
    fredgraph.csv is FRED's public CSV export endpoint.

    NOTE (found this session): plain `curl <url>` run directly in this dev
    sandbox's shell fetched this exact endpoint reliably every single time
    (dozens of manual attempts). But BOTH `requests.get()` called from
    inside a Python function AND a `subprocess.run(["curl", ...])` call
    inside that same function reproducibly hung/timed out or hit HTTP/2
    stream errors — identical URL/args that succeed as a one-off in the
    same interpreter. Root cause not isolated (looks like proxy behavior
    specific to this sandbox, possibly burst/retry related) and not worth
    more session time chasing. PRACTICAL FIX: this function is a thin,
    best-effort convenience wrapper (single attempt, no retry loop that
    could trip whatever the proxy doesn't like) — the actual research run
    in this session used pre-fetched files via load_fred_csv_file() /
    --sp500-csv / --vix-csv instead, which is also the reproducible path
    for anyone re-running this later. Never used in the trading path."""
    import subprocess
    url = FRED_CSV_URL.format(series_id=series_id)
    proc = subprocess.run(
        ["curl", "-sS", "--http1.1", "--max-time", str(timeout), "-A", "voltradeai-research/1.0", url],
        capture_output=True, text=True, timeout=timeout + 5,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"fetch_fred_series({series_id}): curl exit={proc.returncode} stderr={proc.stderr[:300]}")
    return parse_fred_csv_text(proc.stdout)


def align_series(spy: dict, vix: dict) -> tuple[list[str], list[float], list[float]]:
    """Inner-join two {date: value} dicts on shared dates, sorted ascending."""
    common = sorted(set(spy) & set(vix))
    dates = common
    spy_vals = [spy[d] for d in common]
    vix_vals = [vix[d] for d in common]
    return dates, spy_vals, vix_vals


def compute_features(dates: list[str], spy: list[float], vix: list[float]) -> list[dict]:
    """Pure function: given aligned daily closes, compute per-day features.
    Returns a list of dicts, one per day that has enough trailing/forward
    history to compute every feature (early/late days are dropped, stated
    via the returned list's length vs len(dates))."""
    n = len(dates)
    spy_arr = np.array(spy, dtype=float)
    vix_arr = np.array(vix, dtype=float)

    # daily pct returns (index i = return from day i-1 to day i)
    spy_ret = np.full(n, np.nan)
    spy_ret[1:] = (spy_arr[1:] / spy_arr[:-1] - 1.0) * 100.0

    rows = []
    mr = MarkovRegime()
    for i in range(n):
        if i < max(50, MARKOV_LOOKBACK) or i >= n - 20:
            continue  # need 50d trailing for spy_vs_ma50, MARKOV_LOOKBACK for markov, 20d forward for fwd_vol_20

        # vxx_ratio-style feature computed on VIX: VIX / 30-day trailing avg VIX
        vix_avg30 = float(np.mean(vix_arr[i - 29:i + 1]))
        vix_ratio = float(vix_arr[i]) / vix_avg30 if vix_avg30 else np.nan

        # spy_vs_ma50: identical formula to system_config's live input
        spy_ma50 = float(np.mean(spy_arr[i - 49:i + 1]))
        spy_vs_ma50 = float(spy_arr[i]) / spy_ma50 if spy_ma50 else np.nan

        # Markov: feed the real production class the trailing daily returns
        trailing_rets = [r for r in spy_ret[i - MARKOV_LOOKBACK + 1:i + 1] if not np.isnan(r)]
        state, probs, signal = mr.get_current_state(trailing_rets)
        p_bear = float(probs[0])

        # forward realized vol: stdev of daily returns over the NEXT 5 / 20
        # trading days (i+1..i+5 / i+1..i+20) — strictly forward, no lookahead
        fwd5 = spy_ret[i + 1:i + 6]
        fwd20 = spy_ret[i + 1:i + 21]
        fwd_vol_5 = float(np.std(fwd5, ddof=1)) if np.all(~np.isnan(fwd5)) and len(fwd5) == 5 else None
        fwd_vol_20 = float(np.std(fwd20, ddof=1)) if np.all(~np.isnan(fwd20)) and len(fwd20) == 20 else None

        rows.append({
            "date": dates[i],
            "vix_ratio": vix_ratio,
            "spy_vs_ma50": spy_vs_ma50,
            "markov_p_bear": p_bear,
            "markov_signal": signal,
            "fwd_vol_5": fwd_vol_5,
            "fwd_vol_20": fwd_vol_20,
        })
    return rows


def _clean_pairs(xs: list, ys: list) -> tuple[list[float], list[float]]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and not (isinstance(x, float) and np.isnan(x))]
    if not pairs:
        return [], []
    xa, ya = zip(*pairs)
    return list(xa), list(ya)


def compare_detectors(rows: list[dict]) -> dict:
    """Pure function: Spearman rank correlation of each candidate signal
    against forward realized vol at two horizons. Rank correlation is used
    deliberately (not Pearson) because it is threshold- and scale-free —
    the honest way to compare a VIX-ratio proxy against production-VXX
    thresholds we cannot reproduce here (see module docstring)."""
    out: dict = {"n": len(rows)}
    signals = {
        "vix_ratio": [r["vix_ratio"] for r in rows],
        "spy_vs_ma50": [r["spy_vs_ma50"] for r in rows],
        "markov_p_bear": [r["markov_p_bear"] for r in rows],
    }
    horizons = {
        "fwd_vol_5": [r["fwd_vol_5"] for r in rows],
        "fwd_vol_20": [r["fwd_vol_20"] for r in rows],
    }
    for sig_name, sig_vals in signals.items():
        out[sig_name] = {}
        for hz_name, hz_vals in horizons.items():
            xa, ya = _clean_pairs(sig_vals, hz_vals)
            if len(xa) < 10:
                out[sig_name][hz_name] = {"n": len(xa), "rho": None, "p": None}
                continue
            rho, p = spearmanr(xa, ya)
            out[sig_name][hz_name] = {"n": len(xa), "rho": round(float(rho), 4), "p": round(float(p), 6)}

    # base rate: unconditional forward vol mean/std (REASONING STANDARD #3)
    for hz_name, hz_vals in horizons.items():
        clean = [v for v in hz_vals if v is not None]
        out[f"{hz_name}_base_mean"] = round(float(np.mean(clean)), 4) if clean else None
        out[f"{hz_name}_base_std"] = round(float(np.std(clean)), 4) if clean else None
    return out


def run(sp500_csv: Optional[str] = None, vix_csv: Optional[str] = None,
        offline_json: Optional[str] = None) -> tuple[dict, list[dict]]:
    if offline_json:
        with open(offline_json) as f:
            cached = json.load(f)
        dates, spy, vix = cached["dates"], cached["spy"], cached["vix"]
    elif sp500_csv and vix_csv:
        spy_d = load_fred_csv_file(sp500_csv)
        vix_d = load_fred_csv_file(vix_csv)
        dates, spy, vix = align_series(spy_d, vix_d)
    else:
        spy_d = fetch_fred_series("SP500")
        vix_d = fetch_fred_series("VIXCLS")
        dates, spy, vix = align_series(spy_d, vix_d)
    rows = compute_features(dates, spy, vix)
    result = compare_detectors(rows)
    result["date_range"] = [rows[0]["date"], rows[-1]["date"]] if rows else None
    result["fetched_at"] = datetime.utcnow().isoformat() + "Z"
    return result, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp500-csv", help="path to a pre-fetched fredgraph.csv?id=SP500 file (see module docstring)")
    ap.add_argument("--vix-csv", help="path to a pre-fetched fredgraph.csv?id=VIXCLS file")
    ap.add_argument("--offline", help="path to a cached {dates,spy,vix} JSON instead of CSVs or a live fetch")
    ap.add_argument("--save-raw", help="optional path to dump the aligned {dates,spy,vix} JSON (for --offline reruns) — requires --sp500-csv/--vix-csv or a live fetch, not --offline")
    ap.add_argument("--save-result", help="optional path to write the comparison result JSON")
    args = ap.parse_args()

    if args.save_raw and not args.offline:
        spy_d = load_fred_csv_file(args.sp500_csv) if args.sp500_csv else fetch_fred_series("SP500")
        vix_d = load_fred_csv_file(args.vix_csv) if args.vix_csv else fetch_fred_series("VIXCLS")
        dates_r, spy_r, vix_r = align_series(spy_d, vix_d)
        with open(args.save_raw, "w") as f:
            json.dump({"dates": dates_r, "spy": spy_r, "vix": vix_r}, f)
        print(f"Saved raw aligned series to {args.save_raw}")

    result, rows = run(sp500_csv=args.sp500_csv, vix_csv=args.vix_csv, offline_json=args.offline)
    print(json.dumps(result, indent=2))
    if args.save_result:
        with open(args.save_result, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {args.save_result}")
