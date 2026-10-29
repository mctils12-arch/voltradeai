"""
brain_evidence.py — the evidence behind the owner's Brain page (2026-10-29,
human: "how would I know what is being used at the moment and what
combinations could have produced better results ... the short term is one
thing but the long term is different").

READ-ONLY. No order, no write, no trading input. One call returns:

  * config        — the BASE_CONFIG switches the page derives live/shadow/off
                    from (server/brain.ts applies the env-var-wins rule of
                    regime_engine.flag_raw / entry_shadow.flag_raw on top).
  * buckets       — SHORT-TERM outcomes per shadow-log decision bucket
                    (taken, shadow_mode_entry per sleeve, rejected_score,
                    rejected_regime_*, would_reject_regime_*, ...): one
                    first-seen record per (ticker, ET entry day)
                    (shadow_portfolio._first_seen_per_ticker_day, the
                    shadow_bands_dedup rule), per outcome ("xa" exit-aligned
                    = the live exit ladder; "+5d"/"+20d" = the old +2%/-4%
                    path ruler): n, mean return, hit rate (return > 0),
                    day-clustered bootstrap 95% CI of the mean, and the same
                    rows' SPY return over the SAME window (SPY close on the
                    entry day -> SPY close on the outcome's exit day) with the
                    mean excess and its CI. Long-stock returns of the scored
                    underlying — never option P&L.
  * source_validation — source_validation.py's own report (per-source IC,
                    Benjamini-Hochberg q, verdicts; forward selection with its
                    Bonferroni-per-step discount; ablation), trimmed to
                    aggregates. Nothing is re-derived here.
  * ml_v3         — ml_model_v3.status() (the candidate's honest OOS eval on
                    the volume) and the IC of the logged v3 probability on the
                    shadow population (same ic_stats as source_validation).
  * forecast      — forecast.report()'s accuracy summary + latest holdout.
  * walkforward   — the LONG-TERM test of the stock score's price inputs:
                    walkforward.py's last good result (DATA_DIR/walkforward/
                    latest.json, research/walkforward.md), trimmed to its
                    provenance, the per-input Brain long-term items and the
                    not-measurable reasons. Read from the file, never recomputed.

Every block degrades to {"error": ...} on its own; the page shows the gap.
Registered in voltrade_daemon.py as "brain_evidence"; bot.ts calls it through
pythonCall (daemon first, python3 -c fallback) and caches it for 10 minutes.
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np

BUCKET_MIN_N = 30          # FINDINGS step 4: a guard's read needs n >= 30
BUCKET_MIN_DAYS = 5        # == source_validation.MIN_DAYS (clustered bootstrap)
BUCKET_OUTCOMES = ("xa", "+5d", "+20d")
SPY_FETCH_BUDGET_S = 20.0
_SPY_CACHE: Dict[str, object] = {"at": 0.0, "closes": None, "key": None}
_SPY_TTL_S = 6 * 3600.0

# BASE_CONFIG keys the page reads (server/brain.ts BRAIN_CONFIG_MIRROR is the
# Node fallback; test_brain_evidence.py pins the two together).
CONFIG_KEYS = (
    "SHADOW_ENTRY_SLEEVES", "REGIME_ENGINE_ENABLED", "REGIME_SELECTION_DEFAULT",
    "REGIME_OPTIONS_DEFAULT", "REGIME_SIZING_DEFAULT", "REGIME_LABELS_DEFAULT",
    "FLOOR_BASKET_ENABLED", "DEFENSIVE_FLOOR_ENABLED", "LEG3_VRP_PCT",
    "DRAWDOWN_HALT_ENABLED", "POSITION_HARD_STOP_ENABLED",
)


def config_snapshot() -> dict:
    from system_config import BASE_CONFIG
    out = {k: BASE_CONFIG.get(k) for k in CONFIG_KEYS}
    cvx = BASE_CONFIG.get("CONVEXITY_OVERLAY") or {}
    out["CONVEXITY_OVERLAY_enabled"] = bool(cvx.get("enabled")) if isinstance(cvx, dict) else None
    try:
        from tiered_strategy import TIERS_ENABLED
        out["TIERS_ENABLED"] = {str(k): bool(v) for k, v in TIERS_ENABLED.items()}
    except Exception as e:  # tiered_strategy pulls heavy deps; the page shows the gap
        out["TIERS_ENABLED"] = None
        out["TIERS_ENABLED_error"] = str(e)[:120]
    return out


# ── short-term: decision buckets ─────────────────────────────────────────────

def bucket_key(rec: dict) -> str:
    d = str(rec.get("decision") or "unknown")
    if d in ("shadow_mode_entry", "shadow_mode_repeat"):
        se = rec.get("shadow_entry") if isinstance(rec.get("shadow_entry"), dict) else {}
        return f"{d}:{se.get('sleeve') or 'unknown'}"
    return d


def _outcome(rec: dict, outcome: str) -> Optional[dict]:
    from shadow_portfolio import XA_KEY
    o = rec.get(XA_KEY) if outcome == "xa" else (rec.get("outcomes") or {}).get(outcome)
    if not isinstance(o, dict) or o.get("label") not in (0, 1) or o.get("return_pct") is None:
        return None
    return o


def spy_window_return(closes: Dict[str, float], entry_day: str, exit_day: Optional[str]) -> Optional[float]:
    """SPY close on the entry day (the last close on/before it) -> SPY close on
    the exit day (on/before), in percent. None when either end is missing or
    the window is empty."""
    if not closes or not entry_day or not exit_day:
        return None
    days = sorted(closes)

    def on_or_before(d: str) -> Optional[str]:
        best = None
        for x in days:
            if x <= d:
                best = x
            else:
                break
        return best

    a, b = on_or_before(entry_day[:10]), on_or_before(exit_day[:10])
    if not a or not b or b <= a:
        return None
    return (closes[b] / closes[a] - 1.0) * 100.0


def mean_stats(ys: Sequence[float], days: Sequence[str], seed: int = 20261029) -> dict:
    """n, mean, hit rate and a day-clustered bootstrap 95% CI of the mean
    (source_validation.DayBootstrap: names entered the same day share one
    market move)."""
    from source_validation import DayBootstrap
    y = np.asarray(ys, dtype=float)
    n, n_days = len(y), len(set(days))
    out: dict = {"n": n, "n_days": n_days}
    if n == 0:
        return out
    out["mean_pct"] = round(float(y.mean()), 3)
    out["hit_rate"] = round(float((y > 0).mean()), 4)
    if n >= BUCKET_MIN_N and n_days >= BUCKET_MIN_DAYS:
        W = DayBootstrap(days, seed=seed).weights(days)
        s = W.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = (W @ y) / s
        reps = reps[np.isfinite(reps)]
        if len(reps) >= 50:
            lo, hi = np.percentile(reps, [2.5, 97.5])
            out["ci95"] = [round(float(lo), 3), round(float(hi), 3)]
    out["thin"] = n < BUCKET_MIN_N
    return out


def bucket_stats(records: Sequence[dict], spy_closes: Optional[Dict[str, float]]) -> dict:
    """{bucket: {outcome: stats}} over first-seen (ticker, ET day) records."""
    from shadow_portfolio import _first_seen_per_ticker_day, _xa_entry_date_et
    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        if isinstance(r, dict):
            by_bucket[bucket_key(r)].append(r)
    out: Dict[str, dict] = {}
    for b, recs in sorted(by_bucket.items()):
        cell: Dict[str, dict] = {"records": len(recs)}
        for oc in BUCKET_OUTCOMES:
            dd = _first_seen_per_ticker_day(recs, lambda r, oc=oc: _outcome(r, oc) is not None)
            ys, ds, ex, exd = [], [], [], []
            for r in dd:
                day = _xa_entry_date_et(r.get("timestamp", ""))
                o = _outcome(r, oc)
                if not day or o is None:
                    continue
                y = float(o["return_pct"])
                ys.append(y)
                ds.append(day)
                spy = spy_window_return(spy_closes or {}, day, o.get("exit_date"))
                if spy is not None:
                    ex.append(y - spy)
                    exd.append(day)
            st = mean_stats(ys, ds)
            if ds:
                st["first_day"], st["last_day"] = min(ds), max(ds)
            if spy_closes:
                xs = mean_stats(ex, exd, seed=20261030)
                st["vs_spy"] = {"n": xs.get("n", 0), "mean_excess_pct": xs.get("mean_pct"),
                                "ci95": xs.get("ci95"), "beat_rate": xs.get("hit_rate")}
            cell[oc] = st
        out[b] = cell
    return out


def _record_days(records: Sequence[dict]) -> List[str]:
    from shadow_portfolio import _xa_entry_date_et
    return sorted({d for d in (_xa_entry_date_et(r.get("timestamp", "")) for r in records if isinstance(r, dict)) if d})


def spy_closes_for(records: Sequence[dict], deadline_s: float = SPY_FETCH_BUDGET_S) -> dict:
    """SPY daily closes covering the log (one paginated Alpaca call through
    the shared bucket; cached 6 h in-process). {"closes": {...}} or {"error"}."""
    days = _record_days(records)
    if not days:
        return {"error": "no dated records"}
    start = (datetime.strptime(days[0], "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"{start}|{end}"
    now = time.time()
    if _SPY_CACHE["closes"] and _SPY_CACHE["key"] == key and now - float(_SPY_CACHE["at"]) < _SPY_TTL_S:
        return {"closes": _SPY_CACHE["closes"], "cached": True}
    from shadow_portfolio import _fetch_bars_paginated
    unfetched: set = set()
    bars = _fetch_bars_paginated(["SPY"], start, end, deadline=now + deadline_s, unfetched=unfetched)
    if "SPY" in unfetched:
        return {"error": "SPY bars fetch did not complete (Alpaca unreachable or out of time)"}
    closes = {str(b.get("t", ""))[:10]: float(b["c"]) for b in bars.get("SPY", []) if b.get("c")}
    if not closes:
        return {"error": "no SPY bars returned"}
    _SPY_CACHE.update({"at": now, "closes": closes, "key": key})
    return {"closes": closes, "cached": False}


# ── source validation (reused, trimmed) ──────────────────────────────────────

_SRC_KEYS = ("n", "n_days", "ic", "ci95", "se", "p", "q_bh", "verdict", "verdict_reason",
             "group", "kind", "market_wide", "constant")


def _trim_selection(sel: dict) -> dict:
    keep = ("status", "reason", "n_rows", "candidates", "split", "steps", "selected", "weights",
            "holdout_live_score", "holdout_selected", "holdout_vs_live_score")
    out = {k: sel.get(k) for k in keep if k in sel}
    for k in ("holdout_live_score", "holdout_selected"):
        if isinstance(out.get(k), dict):
            out[k] = {kk: out[k].get(kk) for kk in ("n", "n_days", "ic", "ci95") if kk in out[k]}
    out["variants_tried"] = sum(int(s.get("tried") or 0) for s in (sel.get("steps") or []))
    return out


def trim_source_validation(rep: dict) -> dict:
    srcs = {nm: {k: r.get(k) for k in _SRC_KEYS if k in r} for nm, r in (rep.get("sources") or {}).items()}
    ref = {nm: {k: r.get(k) for k in ("n", "n_days", "ic", "ci95", "p") if k in r}
           for nm, r in (rep.get("reference") or {}).items()}
    pop = (rep.get("population") or {}).get(rep.get("method", {}).get("primary_outcome", "+5d")) or {}
    sel = rep.get("selection") or {}
    abl = rep.get("ablation") or {}
    variants = {}
    for nm, v in (abl.get("variants") or {}).items():
        q = (v.get("quintiles") or {})
        variants[nm] = {k: v.get(k) for k in ("n", "n_days", "ic", "ci95", "status", "removed",
                                              "ic_change_vs_baseline", "ic_change_se") if k in v}
        if "Q5_minus_Q1" in q:
            variants[nm]["q5_minus_q1_pct"] = q["Q5_minus_Q1"]
    return {
        "method": rep.get("method"),
        "population": {k: pop.get(k) for k in ("n_unique", "n_days", "first_day", "last_day", "mean_outcome_pct")},
        "summary": rep.get("summary"),
        "sources": srcs,
        "reference": ref,
        "selection": {
            "method": sel.get("method"),
            "features": _trim_selection(sel.get("features") or {}),
            "breakdown": _trim_selection(sel.get("breakdown") or {}),
        },
        "ablation": {**{k: abl.get(k) for k in ("status", "reason", "n_rows_with_components",
                                                 "dropped_reconstruction_mismatch", "baseline_inverted",
                                                 "un_inverts", "improvement_significant", "removed_groups")
                        if k in abl}, "variants": variants},
    }


def v3_shadow_ic(records: Sequence[dict]) -> dict:
    """IC of the logged v3 candidate probability vs the primary +5d outcome on
    the source_validation population (reference row: outside the BH family)."""
    import source_validation as sv
    rows = sv.build_population(records, sv.PRIMARY_OUTCOME)
    boot = sv.DayBootstrap([r["day"] for r in rows])
    x, y, d, _ = sv._extract(rows, sv._ml_getter("v3_win_probability"))
    st = sv.ic_stats(x, y, d, boot)
    st.pop("se", None)
    if "p" in st:
        st["p"] = round(st["p"], 5)
    if d:
        st["first_day"], st["last_day"] = min(d), max(d)
    return st


def _forecast_summary() -> dict:
    import forecast
    rep = forecast.report()
    latest = rep.get("latest") or {}
    val = ((latest.get("model") or {}).get("validation") or {})
    acc = rep.get("accuracy") or {}
    return {
        "generated_at": rep.get("generated_at"), "model_version": rep.get("model_version"),
        "accuracy": {k: acc.get(k) for k in ("n", "thin", "hits", "hit_rate", "brier", "brier_skill_vs_coinflip", "up_rate")},
        "weeks": len(rep.get("weeks") or []),
        "latest_made_at": latest.get("made_at"),
        "holdout": {k: val.get(k) for k in ("split_date", "n_train", "n_test", "brier", "brier_climatology",
                                            "brier_skill", "hit_rate", "test_base_rate")} if val else None,
        "fit_window": [(latest.get("model") or {}).get("first_date"), (latest.get("model") or {}).get("last_date")] if latest else None,
    }


def _ml_v3_summary() -> dict:
    """ml_model_v3.status() trimmed to the pooled OOS headline (no per-quintile dump)."""
    import ml_model_v3
    st = ml_model_v3.status()
    pooled = st.get("pooled") if isinstance(st.get("pooled"), dict) else {}
    q = st.get("quintiles_net_return") if isinstance(st.get("quintiles_net_return"), dict) else {}
    return {
        "candidate_exists": st.get("candidate_exists"), "promoted": st.get("promoted"), "role": st.get("role"),
        "eval_age_days": st.get("eval_age_days"), "trained_at": st.get("trained_at"), "rows": st.get("rows"),
        "date_range": st.get("date_range"), "auc_ci95_date_block": st.get("auc_ci95_date_block"),
        "pooled": {k: pooled.get(k) for k in ("n", "auc", "log_loss", "base_rate_log_loss", "brier", "base_rate_brier", "base_rate")},
        "top_minus_bottom_pct": q.get("top_minus_bottom"),
        "base_rate_mean_net_return_pct": st.get("base_rate_mean_net_return_pct"),
        "variants_tried": st.get("variants_tried"),
    }


WALKFORWARD_SCHEMA = "walkforward/v1"


def _walkforward_summary(path: Optional[str] = None) -> dict:
    """walkforward.py's last good result, trimmed for the page (no ticker or
    price is in that file: derived statistics only)."""
    import json
    import os
    if path is None:
        import storage_config
        path = os.path.join(storage_config.WALKFORWARD_DIR, "latest.json")
    if not os.path.exists(path):
        return {"error": "no walk-forward result yet (walkforward.py has not completed a run on a complete price-panel build)"}
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if not isinstance(d, dict) or d.get("schema") != WALKFORWARD_SCHEMA:
        return {"error": f"unexpected walk-forward schema {d.get('schema') if isinstance(d, dict) else None!r}"}
    u = d.get("universe") or {}
    w = d.get("window") or {}
    inputs = d.get("inputs") or {}
    return {
        "schema": d["schema"], "generated_at": d.get("generated_at"), "as_of": d.get("as_of"), "code_version": d.get("code_version"),
        "panel": {"build_id": (d.get("panel") or {}).get("build_id")},
        "window": {k: w.get(k) for k in ("first_rebalance", "last_rebalance", "n_rebalances_scored", "primary_horizon_days")},
        "universe": {k: u.get(k) for k in ("n_symbols", "delisted_symbol_share", "rows")},
        "items": d.get("brain_long_term") or {},
        "verdicts": {k: (v or {}).get("verdict") for k, v in inputs.items()},
        "not_measurable": d.get("not_measurable") or {},
    }


def _guard(fn, *a, **k) -> dict:
    try:
        out = fn(*a, **k)
        return out if isinstance(out, dict) else {"value": out}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def brain_evidence(fetch_spy: bool = True) -> dict:
    """Production entry point (daemon method "brain_evidence"). See module docstring."""
    t0 = time.time()
    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "module": "brain_evidence"}
    out["config"] = _guard(config_snapshot)
    try:
        from shadow_portfolio import _load_shadow_log
        records = _load_shadow_log()
    except Exception as e:
        records = []
        out["shadow_log_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    days = _record_days(records)
    out["shadow_log"] = {"records": len(records), "first_day": days[0] if days else None,
                         "last_day": days[-1] if days else None}
    spy = _guard(spy_closes_for, records) if (fetch_spy and records) else {"error": "not fetched"}
    closes = spy.get("closes") if isinstance(spy.get("closes"), dict) else None
    out["spy"] = {"ok": bool(closes), "error": spy.get("error"), "cached": spy.get("cached"),
                  "first_day": min(closes) if closes else None, "last_day": max(closes) if closes else None,
                  "method": "SPY close on the entry ET day (last close on/before) -> SPY close on the outcome's exit day"}
    out["buckets"] = _guard(bucket_stats, records, closes) if records else {}
    out["source_validation"] = _guard(lambda: trim_source_validation(_full_sv(records))) if records else {"error": "shadow log empty"}
    out["ml_v3"] = _guard(_ml_v3_summary)
    out["ml_v3_shadow_ic"] = _guard(v3_shadow_ic, records) if records else {"error": "shadow log empty"}
    out["forecast"] = _guard(_forecast_summary)
    out["walkforward"] = _guard(_walkforward_summary)
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def _full_sv(records: Sequence[dict]) -> dict:
    import source_validation as sv
    rep = sv.validate_sources(records)
    rep.update(sv.selection_and_ablation(records, rep))
    return rep


if __name__ == "__main__":  # python3 brain_evidence.py
    import json
    print(json.dumps(brain_evidence(), default=str))
