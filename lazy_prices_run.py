"""Lazy Prices gate-2 production runner — ONE command for the weekly job
(server/lazyPrices.ts). Chains the three steps the 2026-11-10 adapter PR left
manual, then prints ONE JSON summary line for the scheduler:

  1. ensure_pairs_cache   all-form Form 3/4/5 issuer pairs (container tmp, refetchable)
  2. lazy_prices_backfill.load_rows  the point-in-time-idf similarity rows the
     nightly backfill accumulates on the volume (DATA_DIR/lazy_prices/backfill/
     rows/*.jsonl.gz) — SAME schema pit_rows_from_archive produced; since
     2026-11-10 the archive-walk path is gone from this runner because the
     filings archive never existed on the production volume and cannot fit it
     (see lazy_prices_backfill.py's docstring)
  3. lazy_prices_panel.run_gate2  the pre-registered harness, constants untouched

Research plumbing only: nothing here is read by a trade path and nothing is
surfaced on /data. latest.json carries DERIVED STATISTICS ONLY — no ticker, no
price (Alpaca data licence) and no CIK. Status "skipped" (never an error) when a
prerequisite is absent: no backfill rows yet, or no complete price-panel build.
A repeat run on the same archive re-evaluates the SAME pre-registered spec, so it
adds no variant to the ledger (variants stays 1). Last good latest.json is only
replaced by a successful run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time

OUT_SUBDIR = "lazy_prices"


def _write_atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, path)


def summarize(res: dict, build_id, code_version: str, extra: dict) -> dict:
    """Headline-only payload: verdict, counts, per-block spreads. No identifiers."""
    s = res["summary"]
    blocks = {k: {kk: v.get(kk) for kk in ("cohorts", "mean_gross", "mean_net", "t_nw_net", "q_excess_mean")}
              for k, v in s.items() if isinstance(v, dict)}
    return {
        "schema": "lazy_prices_gate2/1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code_version": code_version,
        "panel": {"build_id": build_id},
        "verdict": res["verdict"], "variants": res["variants"],
        "n_obs": res["n_obs"], "drops": res["drops"], "delisted_exits": res["delisted_exits"],
        "summary": blocks,
        "join": res.get("join"),
        "panel_stats": res.get("panel"),
        "caveat": res.get("caveat"),
        **extra,
    }


def run(out_dir: str, pairs_dir: str, backfill_dir: str, panel_root, code_version: str, max_rss_mb: float) -> dict:
    t0 = time.time()
    import price_panel as pp
    import lazy_prices_backfill as lpb
    cur = pp.current_build(panel_root)
    if not cur:
        return {"status": "skipped", "reason": "no complete price-panel build"}
    rows = lpb.load_rows(backfill_dir)
    if not rows:
        return {"status": "skipped", "reason": "no backfill rows yet (lazy_prices_backfill.py has not scored a pair on this volume)"}
    import sec_issuer_pairs as sip
    import sec_cik_symbol_map as cmap
    import lazy_prices_panel as lpp
    _ok, failed = sip.ensure_pairs_cache(pairs_dir)
    cm = cmap.build_map(sip.rows_from_pairs_cache(pairs_dir))
    res = lpp.run_gate2(rows, cm, cur["dir"], variants=1, max_rss_mb=max_rss_mb)
    bstat = lpb.rows_status(backfill_dir)
    payload = summarize(res, cur.get("build_id"), code_version,
                        {"pairs_quarters_failed": len(failed), "rows_scored": len(rows),
                         "backfill": {"cursor_day": bstat.get("cursor_day"), "cursor_closed": bstat.get("cursor_closed"),
                                      "first_month": bstat.get("first_month"), "last_month": bstat.get("last_month"),
                                      "counts": bstat.get("counts")},
                         "elapsed_s": round(time.time() - t0, 1)})
    _write_atomic(os.path.join(out_dir, "latest.json"), payload)
    _write_atomic(os.path.join(out_dir, "last_run.json"),
                  {"status": "ok", "finished_at": payload["generated_at"], "verdict": res["verdict"]["result"]})
    return {"status": "ok", "verdict": res["verdict"]["result"], "build_id": cur.get("build_id"),
            "n_obs": res["n_obs"], "elapsed_s": payload["elapsed_s"],
            "peak_rss_mb": (res.get("panel") or {}).get("peak_rss_mb")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--max-rss-mb", type=float, default=1500.0)
    ap.add_argument("--code-version", default="unknown")
    ap.add_argument("--pairs-dir", default=os.path.join(tempfile.gettempdir(), "voltrade_lazy_prices_pairs"))
    ap.add_argument("--backfill-dir", default=None, help="default DATA_DIR/lazy_prices/backfill")
    ap.add_argument("--panel-root", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    if a.nice:
        try:
            os.nice(a.nice)
        except OSError as e:
            print(f"[lazy_prices] os.nice failed: {e}", file=sys.stderr)
    import lazy_prices_backfill as lpb
    data_dir = os.environ.get("DATA_DIR") or ("/data/voltrade" if os.path.exists("/data") else "/tmp")
    out_dir = a.out_dir or os.path.join(data_dir, OUT_SUBDIR)
    try:
        out = run(out_dir, a.pairs_dir, a.backfill_dir or lpb.default_root(), a.panel_root, a.code_version, a.max_rss_mb)
    except Exception as e:  # noqa: BLE001 — one JSON line for the scheduler; last good latest.json untouched
        out = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        try:
            _write_atomic(os.path.join(out_dir, "last_run.json"), {"status": "error", "reason": out["reason"],
                          "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        except OSError as e2:
            print(f"[lazy_prices] could not write last_run.json: {e2}", file=sys.stderr)
    print(json.dumps(out))
    return 0 if out["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
