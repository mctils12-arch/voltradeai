#!/usr/bin/env python3
"""scripts/ml_v3_eval.py — session CLI: train + honestly evaluate the v3
CANDIDATE model (ml_model_v3.py) and print the report research/ml_v3_eval.md
is written from.

Production does NOT run this file (the Docker image ships only root *.py):
the bot runs ml_model_v3.run_job() nightly (server/bot.ts
checkMlV3CandidateJob) and the token-gated /api/diag/ml_v3 probe reads the
result. This CLI is the same pipeline, unbudgeted, for a session sandbox.

Data: --bars FILE (JSON {symbol: [ {t,o,h,l,c,v}, ... ]}), else Alpaca when
ALPACA_KEY/ALPACA_SECRET are set, else Yahoo daily bars (backtest_v2's
split/dividend-adjusted fetcher). Today's in-progress session is stripped.

  python3 scripts/ml_v3_eval.py --bars bars.json --out report.json [--save]
  python3 scripts/ml_v3_eval.py --v2-compare ...   # also legacy-vs-honest CV
                                                   # on the v2 dataset (slow)
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ml_cv  # noqa: E402
import ml_features_v3 as f3  # noqa: E402
import ml_model_v3 as m3  # noqa: E402


def load_bars(path, days):
    if path:
        with open(path) as f:
            return json.load(f), f"file:{os.path.basename(path)}"
    syms = list(f3.REF_SYMBOLS) + m3.UNIVERSE
    if os.environ.get("ALPACA_KEY") and os.environ.get("ALPACA_SECRET"):
        from datetime import datetime, timedelta, timezone
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        out = {}
        for i in range(0, len(syms), 10):
            out.update(m3._alpaca_daily_bars(syms[i:i + 10], start))
        return out, "alpaca"
    from backtest_v2 import _yahoo_bars
    out = {}
    for s in syms:
        b = _yahoo_bars(s, days)
        out[s] = [{"t": d + "T04:00:00Z", "o": o, "h": h, "l": l, "c": c, "v": v}
                  for d, o, h, l, c, v in zip(b["date"], b["open"], b["high"], b["low"],
                                              b["close"], b["volume"])]
        time.sleep(0.3)
    return out, "yahoo"


def v2_compare(bars):
    import ml_model_v2 as v2
    X, y, _, meta = v2._build_training_data(bars, return_meta=True)
    n_t = len([k for k in bars if k not in ("SPY", "QQQ", "IWM", "VXX", "GLD", "TLT", "HYG")])
    leg = ml_cv.legacy_cv(X, y, n_t)
    hon = ml_cv.honest_cv(X, y, meta["dates"], label_end_dates=meta["label_end_dates"],
                          embargo_dates=5, feature_names=list(v2.FEATURE_COLS))
    return {"rows": int(len(y)), "legacy": ml_cv.strip_arrays(leg), "honest": ml_cv.strip_arrays(hon)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars")
    ap.add_argument("--days", type=int, default=1400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", action="store_true", help="write the candidate artifact to DATA_DIR/ml_v3")
    ap.add_argument("--v2-compare", action="store_true")
    a = ap.parse_args()
    bars, source = load_bars(a.bars, a.days)
    today = m3._today_et()
    bars = {k: f3.strip_incomplete(v, today) for k, v in bars.items()}
    t0 = time.time()
    res = m3.run_pipeline(bars, save=a.save)
    rep = res["report"]
    rep["data_source"] = source
    rep["feature_importance_gain"] = res["bundle"]["feature_importance_gain"]
    rep["final_fit"] = res["bundle"]["fit"]
    rep["seconds"] = round(time.time() - t0, 1)
    if a.v2_compare:
        rep["v2_cv_compare"] = v2_compare(bars)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    p = rep["pooled"]
    print(json.dumps({"rows": rep["rows"], "auc": p["auc"], "log_loss": p["log_loss"],
                      "base_rate_log_loss": p["base_rate_log_loss"], "brier": p["brier"],
                      "auc_ci": rep["auc_ci95_date_block"],
                      "top_minus_bottom": rep["quintiles_net_return"]["top_minus_bottom"],
                      "seconds": rep["seconds"]}, default=str))


if __name__ == "__main__":
    main()
