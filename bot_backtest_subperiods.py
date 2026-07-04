#!/usr/bin/env python3
"""
bot_backtest_subperiods.py — out-of-sample / regime-split judgment of the
"Dual-momentum SPY/QQQ" candidate logged in research/open_questions.md.

Why this exists: bot_backtest.py's 2016-2026 pooled result (16.31% CAGR /
0.90 Sharpe / -28.6% maxDD vs SPY 14.13% / 0.83 / -33.7%) is IN-SAMPLE and
tech-led-decade-biased (REASONING STANDARD #2/#4). The open question's
stated PRIOR (recorded before this script ran): "edge shrinks but survives
~+1% CAGR over SPY ex-2020-21; kill if negative in >=2 sub-periods." This
script tests that prior by splitting the same window into calendar
sub-periods and comparing dual-momentum vs SPY buy&hold in each.

2020-2021 (COVID crash + recovery) is reported but EXCLUDED from the kill
count — the prior explicitly frames the hypothesis "ex-2020-21" because
that period's V-shaped recovery is a known outlier confound, not a
representative regime.

    python3 bot_backtest_subperiods.py
"""
from __future__ import annotations

import sys
from datetime import datetime

from bot_backtest import UNIVERSE, BENCH, fetch, backtest, metrics

# Calendar sub-periods spanning the same window as bot_backtest.py's pooled
# run. 2020-2021 isolated on purpose (see module docstring).
SUBPERIODS = [
    ("2016-2019 (pre-COVID bull)",     "2016-01-01", "2019-12-31"),
    ("2020-2021 (COVID crash+recovery, EXCLUDED from kill count)", "2020-01-01", "2021-12-31"),
    ("2022-2023 (rate-hike bear+recovery)", "2022-01-01", "2023-12-31"),
    ("2024-2026 (recent)",             "2024-01-01", "2026-04-07"),
]
KILL_EXCLUDED_LABEL_PREFIX = "2020-2021"


def split_dates(dates: list[str], start: str, end: str) -> list[str]:
    """Inclusive slice of a sorted YYYY-MM-DD date list to [start, end]."""
    return [d for d in dates if start <= d <= end]


def slice_data(data: dict, dates: list[str]) -> dict:
    """Restrict each ticker's date->price map to the given dates only."""
    return {t: {d: px[d] for d in dates if d in px} for t, px in data.items()}


def run_subperiod(data: dict, dates: list[str], start: str, end: str) -> dict:
    sub_dates = split_dates(dates, start, end)
    if len(sub_dates) < 260:  # need >=1yr of daily bars for the 252d momentum lookback
        return {}
    sub_data = slice_data(data, sub_dates)

    spy_eq = [sub_data[BENCH][d] for d in sub_dates if sub_data[BENCH].get(d)]
    spy_dates = [d for d in sub_dates if sub_data[BENCH].get(d)]
    bench = metrics(spy_eq, spy_dates)

    dual = backtest(sub_data, sub_dates, top_n=1, min_score=0, regime=False,
                     universe=["SPY", "QQQ"])
    dual_regime = backtest(sub_data, sub_dates, top_n=1, min_score=0, regime=True,
                            universe=["SPY", "QQQ"])
    return {"bench": bench, "dual": dual, "dual_regime": dual_regime}


def judge(results: dict) -> dict:
    """Apply the pre-committed kill rule: kill if dual-momentum alpha vs SPY
    is negative in >=2 of the non-2020-21 sub-periods."""
    negative_count = 0
    counted = 0
    detail = []
    for label, r in results.items():
        if not r or not r.get("dual") or not r.get("bench"):
            continue
        alpha = r["dual"]["cagr_pct"] - r["bench"]["cagr_pct"]
        excluded = label.startswith(KILL_EXCLUDED_LABEL_PREFIX)
        if not excluded:
            counted += 1
            if alpha < 0:
                negative_count += 1
        detail.append({"label": label, "alpha_cagr_pct": round(alpha, 2), "excluded_from_kill_count": excluded})
    verdict = "KILL" if negative_count >= 2 else "SURVIVES"
    return {"verdict": verdict, "negative_subperiods": negative_count,
            "counted_subperiods": counted, "detail": detail}


def main():
    print(f"Fetching {len(UNIVERSE)} tickers …", file=sys.stderr)
    data = {}
    for t in ("SPY", "QQQ"):
        data[t] = dict(fetch(t))
    dates = sorted(data[BENCH].keys())

    results = {}
    for label, start, end in SUBPERIODS:
        results[label] = run_subperiod(data, dates, start, end)

    hdr = f"{'Sub-period':<58}{'SPY CAGR%':>10}{'Dual CAGR%':>11}{'Dual Sharpe':>12}{'Dual MaxDD%':>12}{'Alpha':>8}"
    print(f"\nOut-of-sample judgment: Dual-momentum SPY/QQQ (no regime filter)\n")
    print(hdr)
    print("-" * len(hdr))
    for label, r in results.items():
        if not r:
            print(f"{label:<58}{'insufficient data':>10}")
            continue
        b, d = r["bench"], r["dual"]
        alpha = d["cagr_pct"] - b["cagr_pct"]
        print(f"{label:<58}{b['cagr_pct']:>10}{d['cagr_pct']:>11}{d.get('sharpe',0):>12}"
              f"{d.get('max_drawdown_pct',0):>12}{alpha:>+8.2f}")

    v = judge(results)
    print(f"\nVERDICT: {v['verdict']} "
          f"({v['negative_subperiods']}/{v['counted_subperiods']} non-2020-21 sub-periods negative; "
          f"kill threshold = 2)")
    for d in v["detail"]:
        flag = " [excluded from kill count]" if d["excluded_from_kill_count"] else ""
        print(f"  {d['label']}: alpha {d['alpha_cagr_pct']:+.2f}pp{flag}")


if __name__ == "__main__":
    main()
