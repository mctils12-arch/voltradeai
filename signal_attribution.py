"""
signal_attribution.py — [RULE-REVIEW] measurement, 2026-11-06 (human refocus
spec section 7: "feature attribution: which signals add return and which are
noise — cut what does not work").

ONE table, one row per SIGNAL the stock score uses (deep_score's sub-scores
and point adjustments, the universe pre-rank, the ML blend, the composites),
joining every honest test that exists for it. Nothing here is re-estimated:
each number is read from the harness that produced it.

  long_term  walkforward.py's last good result (DATA_DIR/walkforward/latest.json):
             2017+ weekly point-in-time walk-forward over the 2016+ price panel
             (delisted names included, survivorship-REDUCED), every input rebuilt
             exactly as live computes it; per-date rank IC averaged with a
             Newey-West t, Bonferroni over every input x horizon, quintile means
             and the Q5-Q1 spread net of costs, the top-10 vs RANDOM ENTRY, by
             regime and by year. Price-derivable inputs only.
  forward    source_validation.validate_sources on the production shadow log:
             one first-seen record per (ticker, ET day), deep-scored candidates,
             Spearman IC with a day-clustered bootstrap CI, Benjamini-Hochberg
             across the family. Every input, but only since it was logged
             (score components since 2026-10-26) and only in the regimes the
             sample visited. Outcomes: the old ruler's +5d path return (primary),
             +10d / +20d / exit-aligned (descriptive).
  model      ml_model_v3.status() (honest purged walk-forward OOS), the logged
             v2/v3 probabilities against forward outcomes, the v2 production
             fit's training-set DUPLICATION (shadow rows per unique ticker-day,
             measured from the same log), and the B3 serve_fix shadow comparison.

There is no +1d label in the shadow log; the walk-forward reports +1d once its
horizons include 1 (descriptive; the verdict horizon is 5 days, the live time
stop's horizon). Neither harness has deciles: the walk-forward reports
quintiles, the forward log terciles (its n is too small for finer buckets).

VERDICTS (pre-registered in this file before any production read of the table;
changing one after seeing results is fishing — bump VERDICT_VERSION and say
why). A signal's long-term verdict is the walk-forward's own (Bonferroni on the
5-day IC); its forward verdict is source_validation's (BH at FDR 0.10):
  anti_predictive  long-term "negative" OR forward FAIL_INVERTED
  predictive       long-term "positive" OR forward PASS
  conflicting      both of the above at once (one source each way)
  dead             forward CONSTANT (the input never moved on live candidates)
  noise            long-term "null" (>= 52 weekly dates) OR forward FAIL
  untested         neither harness has a verdict (INSUFFICIENT / not measurable)
The ACTION column reads BASE_CONFIG["SCORE_COMPOSITE_V2"] (when present) so the
table never claims a cut that is not live; a measured anti-predictive / noise
signal that is still weighted is flagged `review`.

Aggregate-only (no ticker, price or per-record timestamp): served read-only by
the diag probe "signals" and the owner route GET /api/bot/signals. Pure except
get_signal_attribution(), the production loader.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VERDICT_VERSION = "signal_attribution_v1 (2026-11-06)"
VERDICTS = ("anti_predictive", "predictive", "conflicting", "dead", "noise", "untested")
WALKFORWARD_SCHEMA = "walkforward/v1"
PRIMARY_HORIZON = "5"          # walkforward.PRIMARY_HORIZON (trading days)
MIN_DATES_FOR_VERDICT = 52     # walkforward.MIN_DATES_FOR_VERDICT
FAMILY_ALPHA = 0.05            # walkforward.ALPHA (Bonferroni)

# Score-breakdown keys deep_score writes when the composite v2 shadow is on
# (bot_engine.deep_score, shadow_portfolio.SCORE_BREAKDOWN_EXTRA_KEYS). Read
# here so the comparison is pre-registered before the strategy change ships.
COMPOSITE_LEGACY_KEY = "composite_legacy"
COMPOSITE_V2_KEY = "composite_v2"

# Every signal of the stock score, in the order the table shows them.
#   fw   source_validation source name on the forward population (None = not logged)
#   lt   walkforward input id (None = not reconstructable point-in-time; see
#        walkforward.NOT_MEASURABLE for the reason under the same key `lt_nm`)
#   role plain-English place in deep_score
SIGNAL_REGISTRY: Tuple[dict, ...] = (
    {"id": "quick_score", "label": "Universe pre-rank (quick score)", "group": "selection",
     "role": "ranks every scanned name by the size of today's move and volume; only the top ~5 are deep-scored",
     "fw": None, "lt": "quick_score"},
    {"id": "momentum", "label": "Momentum (12-1 month)", "group": "technical",
     "role": "25% of the base score", "fw": "adj.momentum", "lt": "momentum"},
    {"id": "mean_reversion", "label": "Mean reversion (RSI, 5-day move)", "group": "technical",
     "role": "20% of the base score", "fw": "adj.mean_reversion", "lt": "mean_reversion"},
    {"id": "vrp", "label": "Volatility risk premium", "group": "options",
     "role": "25% of the base score", "fw": "adj.vrp", "lt": None, "lt_nm": "vrp"},
    {"id": "squeeze", "label": "Short squeeze", "group": "sentiment",
     "role": "15% of the base score", "fw": "adj.squeeze", "lt": None, "lt_nm": "squeeze"},
    {"id": "volume", "label": "Volume surge (5-day vs 90-day)", "group": "technical",
     "role": "15% of the base score", "fw": "adj.volume", "lt": "volume"},
    {"id": "vol_premium", "label": "Volatility premium bonus (EWMA vs 20-day)", "group": "technical",
     "role": "adds 0, 5 or 8 points", "fw": "adj.vol_premium", "lt": "vol_premium"},
    {"id": "adx", "label": "Trend strength (ADX)", "group": "technical",
     "role": "adds -4 to +6 points", "fw": "adj.adx", "lt": "adx"},
    {"id": "obv", "label": "On-balance volume", "group": "technical",
     "role": "adds -5 or +5 points", "fw": "adj.obv", "lt": "obv"},
    {"id": "rel_strength", "label": "Relative strength vs SPY", "group": "technical",
     "role": "adds 6 points when the 19-day return beats SPY by more than 5 points", "fw": "adj.rel_strength",
     "lt": "rel_strength"},
    {"id": "macro", "label": "Macro (VIX regime, sector move, risk on/off)", "group": "macro",
     "role": "adds about -18 to +16 points", "fw": "adj.macro", "lt": None, "lt_nm": "macro_adj"},
    {"id": "news", "label": "News sentiment", "group": "news",
     "role": "adds -12 to +10 points", "fw": "adj.news", "lt": None, "lt_nm": "news"},
    {"id": "intel", "label": "Intel (news events, insiders, earnings pattern, analyst consensus)",
     "group": "news_insider_earnings", "role": "adds -15 to +15 points (more with the parts below)",
     "fw": "adj.intel", "lt": None, "lt_nm": "intel"},
    {"id": "insider_mspr", "label": "Finnhub insider sentiment (MSPR)", "group": "insider",
     "role": "adds -5 or +5 points inside intel", "fw": "adj.insider_mspr", "lt": None, "lt_nm": "insider_mspr"},
    {"id": "alt", "label": "Alternative data (Wikipedia, short volume, FTDs, GDELT, FRED)", "group": "alt_data",
     "role": "adds -20 to +20 points", "fw": "adj.alt", "lt": None, "lt_nm": "alt"},
    {"id": "social", "label": "Social and search buzz", "group": "social",
     "role": "adds -15 to +15 points", "fw": "adj.social", "lt": None, "lt_nm": "social"},
    {"id": "fundamental", "label": "yfinance fundamentals (ratings, targets, P/E, EV/EBITDA)",
     "group": "fundamentals", "role": "adds about -18 to +16 points", "fw": "adj.fundamental", "lt": None,
     "lt_nm": "fundamentals"},
    {"id": "recommendation", "label": "analyze.py recommendation boost", "group": "analyze",
     "role": "adds 6 or 8 points", "fw": "adj.recommendation", "lt": None,
     "lt_reason": "analyze.py's recommendation has no point-in-time history"},
    {"id": "contrarian", "label": "Contrarian sentiment flags", "group": "sentiment",
     "role": "adds 4, 8 or 10 points", "fw": "adj.contrarian", "lt": None, "lt_nm": "contrarian"},
    {"id": "earnings_beats", "label": "Earnings beats, last 3 quarters", "group": "earnings",
     "role": "not used by the score and not logged yet", "fw": None, "lt": None,
     "lt_reason": "Finnhub's last-4-quarters endpoint carries no report dates, so no point-in-time history exists"},
)
# Composite rows (reference: outside both families).
COMPOSITE_ROWS: Tuple[dict, ...] = (
    {"id": "composite_legacy", "label": "Composite score (legacy)", "lt": "price_composite"},
    {"id": "composite_v2", "label": "Composite score v2", "lt": "price_composite_v2"},
)
NOT_LOGGED_REASON = {
    "earnings_beats": "Forward: deep_score does not fetch it, so nothing is logged on the shadow record yet.",
}


# ── small helpers ────────────────────────────────────────────────────────────

def _num(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _r(v, nd: int = 4) -> Optional[float]:
    x = _num(v)
    return round(x, nd) if x is not None else None


def _signed(v, nd: int = 3) -> str:
    x = _num(v)
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}".replace("-", "−")


def _pts(v, nd: int = 2) -> str:
    x = _num(v)
    if x is None:
        return "n/a"
    return f"{abs(x):.{nd}f}"


def bonferroni_z(m_tests: int, alpha: float = FAMILY_ALPHA) -> float:
    """|t| a single test needs to survive Bonferroni over m two-sided tests."""
    m = max(1, int(m_tests or 1))
    return NormalDist().inv_cdf(1 - alpha / (2 * m))


# ── long-term block (walkforward latest.json) ────────────────────────────────

def long_term_block(wf: Optional[dict], key: Optional[str], nm_key: Optional[str] = None,
                    custom_reason: Optional[str] = None) -> dict:
    """The walk-forward's numbers for one input, or why there are none."""
    if not key:
        nm = ((wf or {}).get("not_measurable") or {}).get(nm_key or "") if nm_key else None
        return {"measurable": False,
                "reason": nm or custom_reason or "No point-in-time history exists for this input, so it cannot be tested long-term."}
    if not isinstance(wf, dict) or wf.get("schema") != WALKFORWARD_SCHEMA:
        return {"measurable": True, "verdict": "insufficient",
                "reason": "The walk-forward has not produced a result that includes this input yet."}
    inp = (wf.get("inputs") or {}).get(key)
    if not isinstance(inp, dict):
        return {"measurable": True, "verdict": "insufficient",
                "reason": f"The last walk-forward run ({wf.get('generated_at', '?')[:10]}) predates this input; "
                          "it is measured from the next weekly run."}
    hz = inp.get("horizons") or {}
    out: dict = {"measurable": True, "verdict": inp.get("verdict") or "insufficient",
                 "verdict_note": inp.get("verdict_note")}
    for h in ("1", "5", "20"):
        b = hz.get(h)
        if not isinstance(b, dict):
            continue
        ic = b.get("ic") or {}
        q = b.get("quintiles") or {}
        top = b.get("top_n") or {}
        out[f"h{h}"] = {
            "ic": _r(ic.get("mean")), "t": _r(ic.get("t_nw"), 3), "p_bonferroni": _r(ic.get("p_bonferroni"), 6),
            "n_dates": ic.get("n_dates"), "hit_rate": _r(ic.get("hit_rate"), 3),
            "quintile_mean_net_pct": [_r(x, 3) for x in (q.get("mean_net_pct") or [])][:5],
            "q5_minus_q1_net_pct": _r(q.get("spread_net_long_short_pct"), 3), "t_q5_minus_q1": _r(q.get("t_nw_long_short"), 3),
            "top10_vs_random_pct": _r(top.get("excess_vs_random_pct"), 3), "t_top10_vs_random": _r(top.get("t_nw_vs_random"), 3),
            "top10_vs_spy_pct": _r(top.get("excess_vs_spy_pct"), 3),
        }
    b5 = hz.get(PRIMARY_HORIZON) if isinstance(hz.get(PRIMARY_HORIZON), dict) else {}
    regimes = {}
    for reg, blk in sorted((b5.get("by_regime") or {}).items()):
        if isinstance(blk, dict):
            regimes[str(reg)] = {"ic": _r(blk.get("mean_ic")), "t": _r(blk.get("t_nw"), 3), "n_dates": blk.get("n_dates"),
                                 "top10_vs_random_pct": _r(blk.get("top_n_excess_vs_random_pct"), 3)}
    out["by_regime_5d"] = regimes
    years = [(_num((blk or {}).get("mean_ic"))) for blk in (b5.get("by_year") or {}).values()]
    years = [y for y in years if y is not None]
    out["years"] = {"n": len(years), "positive": sum(1 for y in years if y > 0), "negative": sum(1 for y in years if y < 0)}
    le = inp.get("live_exit") or {}
    if isinstance(le, dict) and le.get("n_picks"):
        out["live_exit_vs_random_pct"] = _r(le.get("excess_vs_random_pct"), 3)
        out["t_live_exit_vs_random"] = _r(le.get("t_nw_vs_random"), 3)
    return out


# ── forward block (source_validation on the shadow log) ─────────────────────

def forward_block(sources: Optional[dict], name: Optional[str]) -> dict:
    if not name:
        return {"logged": False, "verdict": None}
    r = (sources or {}).get(name)
    if not isinstance(r, dict):
        return {"logged": True, "verdict": "INSUFFICIENT", "n": 0}
    ic, se = _num(r.get("ic")), _num(r.get("se"))
    out = {"logged": True, "verdict": r.get("verdict") or "INSUFFICIENT", "n": r.get("n", 0), "n_days": r.get("n_days", 0),
           "ic": _r(ic), "t": _r(ic / se, 3) if ic is not None and se else None, "ci95": r.get("ci95"),
           "q_bh": _r(r.get("q_bh")), "distinct_values": r.get("distinct_values")}
    terc = r.get("terciles") or {}
    out["top_minus_bottom_tercile_pct"] = _r(terc.get("spread"), 3)
    sb = r.get("sign_buckets") or {}
    if sb:
        out["positive_minus_negative_pct"] = _r(sb.get("spread"), 3)
    regs = {}
    for reg, blk in sorted((r.get("by_regime") or {}).items()):
        if isinstance(blk, dict):
            regs[str(reg)] = {k: blk.get(k) for k in ("n", "ic", "ci95") if k in blk}
    out["by_regime"] = regs
    sec = {}
    for o, blk in (r.get("secondary") or {}).items():
        if isinstance(blk, dict):
            sec[str(o)] = {k: blk.get(k) for k in ("n", "ic", "ci95") if k in blk}
    out["secondary"] = sec
    return out


# ── the verdict (pre-registered; see module docstring) ──────────────────────

def combine_verdict(lt_verdict: Optional[str], fw_verdict: Optional[str]) -> str:
    pos = lt_verdict == "positive" or fw_verdict == "PASS"
    neg = lt_verdict == "negative" or fw_verdict == "FAIL_INVERTED"
    if pos and neg:
        return "conflicting"
    if neg:
        return "anti_predictive"
    if pos:
        return "predictive"
    if fw_verdict == "CONSTANT":
        return "dead"
    if lt_verdict == "null" or fw_verdict == "FAIL":
        return "noise"
    return "untested"


def evidence_of(lt: dict, fw: dict) -> str:
    has_lt = lt.get("verdict") in ("positive", "negative", "null")
    has_fw = fw.get("verdict") in ("PASS", "FAIL_INVERTED", "FAIL", "CONSTANT")
    return "both" if has_lt and has_fw else "long_term" if has_lt else "forward" if has_fw else "none"


# ── what the served score does with it ──────────────────────────────────────

def composite_config(cfg=None) -> Optional[dict]:
    """system_config.BASE_CONFIG["SCORE_COMPOSITE_V2"] when it exists (the
    strategy change that consumes this table); None before it ships. A dict
    passed in is used as is; False means "no composite config" (tests)."""
    if cfg is False:
        return None
    if cfg is not None:
        return cfg if isinstance(cfg, dict) else None
    try:
        from system_config import BASE_CONFIG
    except ImportError:  # the table still renders, every action reads "kept"
        return None
    c = BASE_CONFIG.get("SCORE_COMPOSITE_V2")
    return c if isinstance(c, dict) else None


def action_for(sig_id: str, cfg: Optional[dict]) -> Tuple[str, str]:
    """(action, plain text). action: kept | cut | not_used."""
    if sig_id == "earnings_beats":
        return "not_used", "Not used by the score."
    on = bool(cfg and cfg.get("enabled"))
    cut = set(cfg.get("neutralize") or ()) if on else set()
    if sig_id in cut:
        since = (cfg or {}).get("since", "")
        return "cut", (f"Cut: held at its neutral value in the served score (composite v2{', since ' + since if since else ''}); "
                       "still computed and logged so the cut can be judged.")
    if sig_id == "ml_v2" and on and cfg.get("ml_neutral"):
        return "cut", "Cut: the ML input is held at its neutral 50 in the served score (composite v2); its probability is still logged."
    if sig_id == "quick_score":
        return "kept", "Kept: it still decides which names reach the deep score."
    return "kept", "Kept: still weighted in the served score."


# ── plain-English reason ─────────────────────────────────────────────────────

def _lt_sentence(lt: dict, wf_window: dict, z_bonf: float) -> str:
    if not lt.get("measurable"):
        why = str(lt.get("reason") or "no point-in-time history").rstrip(". ")
        return f"Long-term: not testable — {why[:1].lower() + why[1:]}."
    if lt.get("verdict") == "insufficient" or "h5" not in lt:
        return f"Long-term: {lt.get('reason') or 'no walk-forward result yet.'}"
    h5, h20 = lt.get("h5") or {}, lt.get("h20") or {}
    span = f"{(wf_window.get('first_rebalance') or '?')[:4]}–{(wf_window.get('last_rebalance') or '?')[:4]}"
    n_dates = h5.get("n_dates")
    v = lt.get("verdict")
    if v == "negative":
        head = "a higher value went with a LOWER 5-day return"
    elif v == "positive":
        head = "a higher value went with a HIGHER 5-day return"
    else:
        head = "no reliable link to the 5-day return"
    sig = "survives" if v in ("negative", "positive") else "does not survive"
    s = (f"Long-term ({span} point-in-time walk-forward, {n_dates} weekly dates): {head} "
         f"(rank IC {_signed(h5.get('ic'))}, t {_signed(h5.get('t'), 2)}; {sig} Bonferroni).")
    if _num(h20.get("ic")) is not None:
        s += f" 20-day IC {_signed(h20.get('ic'))}."
    h1 = lt.get("h1") or {}
    if _num(h1.get("ic")) is not None:
        s += f" 1-day IC {_signed(h1.get('ic'))} (descriptive)."
    yrs = lt.get("years") or {}
    if yrs.get("n"):
        s += f" Same sign as the full period in {yrs.get('negative') if (_num(h5.get('ic')) or 0) < 0 else yrs.get('positive')} of {yrs['n']} years."
    tr, tt = _num(h5.get("top10_vs_random_pct")), _num(h5.get("t_top10_vs_random"))
    if tr is not None:
        word = "trailed" if tr < 0 else "beat"
        strong = tt is not None and abs(tt) >= z_bonf
        s += (f" Its top-10 picks {word} random entry by {_pts(tr)} points per 5 days"
              f" (t {_signed(tt, 2)}{', beyond the Bonferroni bar' if strong else ''}).")
    return s


def _fw_sentence(fw: dict, pop: dict) -> str:
    if not fw.get("logged"):
        return "Forward: not logged on the shadow record."
    v, n = fw.get("verdict"), fw.get("n") or 0
    if v == "INSUFFICIENT":
        last = pop.get("last_day")
        return (f"Forward (shadow log): {n} labeled first-seen records, fewer than the 100 a verdict needs"
                f"{' (labels so far run to ' + last + ')' if last else ''}.")
    if v == "CONSTANT":
        return f"Forward (shadow log): the value never changed across {n} live candidates — the input is not reaching the score."
    word = {"PASS": "predictive", "FAIL_INVERTED": "anti-predictive", "FAIL": "not significant"}.get(v, v)
    return (f"Forward (shadow log, {n} first-seen candidates over {fw.get('n_days')} days): "
            f"+5-day rank IC {_signed(fw.get('ic'))} (t {_signed(fw.get('t'), 2)}, BH q {fw.get('q_bh')}) — {word}.")


def reason_text(verdict: str, lt: dict, fw: dict, wf_window: dict, pop: dict, z_bonf: float,
                not_logged: Optional[str] = None) -> str:
    head = {"anti_predictive": "Anti-predictive.", "predictive": "Predictive.", "conflicting": "Conflicting evidence.",
            "dead": "Dead input.", "noise": "Noise.", "untested": "Untested."}[verdict]
    parts = [head, _lt_sentence(lt, wf_window, z_bonf)]
    parts.append(not_logged if not_logged and not fw.get("logged") else _fw_sentence(fw, pop))
    return " ".join(p for p in parts if p)


# ── model evidence ───────────────────────────────────────────────────────────

def _entry_day(rec: dict) -> Optional[str]:
    """ET entry day (None for an unparseable timestamp — shadow_portfolio's
    helper logs and returns None itself)."""
    from shadow_portfolio import _xa_entry_date_et
    return _xa_entry_date_et(rec.get("timestamp", ""))


def training_duplication(records: Sequence[dict], horizon_key: str = "+10d") -> dict:
    """What ml_model_v2's production fit actually learns from: load_shadow_data
    takes every labeled shadow record with features — one per SCAN, not one per
    candidate — and fast_mode keeps the last 10,000 stacked rows, which are the
    shadow rows once 10,000 are labeled. duplication = rows / unique
    (ticker, ET day). Mirrors load_shadow_data's filter (label 0/1, features)."""
    rows, keys = 0, set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        o = (rec.get("outcomes") or {}).get(horizon_key) if isinstance(rec.get("outcomes"), dict) else None
        if not isinstance(o, dict) or o.get("label") not in (0, 1) or not rec.get("features"):
            continue
        rows += 1
        day = _entry_day(rec)
        keys.add((str(rec.get("ticker", "")).upper(), day))
    uniq = len(keys)
    return {"label": horizon_key, "rows": rows, "unique_ticker_days": uniq,
            "duplication": round(rows / uniq, 2) if uniq else None, "fast_mode_window": 10_000,
            "note": ("ml_model_v2._train_model_impl stacks generic, feedback and shadow rows in that order and fast_mode "
                     "keeps the LAST 10,000: when shadow rows alone reach 10,000 the fit sees only them, each unique "
                     "candidate-day repeated `duplication` times — row-index CV folds then hold copies of the same "
                     "candidate-day on both sides, so the reported accuracy is not out-of-sample.")}


def _pop_rows(records: Sequence[dict]) -> List[dict]:
    import source_validation as sv
    return sv.build_population(records, sv.PRIMARY_OUTCOME)


def paired_compare(rows: List[dict], get_a, get_b, label_a: str, label_b: str, seed_offset: int = 7) -> dict:
    """IC of two logged scores on the SAME labeled rows, and the paired
    day-clustered difference (b - a)."""
    import numpy as np
    import source_validation as sv
    keep = [(get_a(r["rec"]), get_b(r["rec"]), r["y"], r["day"]) for r in rows]
    keep = [k for k in keep if k[0] is not None and k[1] is not None]
    out: dict = {"n": len(keep), "a": label_a, "b": label_b}
    if len(keep) < sv.MIN_N:
        out["status"] = "INSUFFICIENT"
        out["reason"] = f"{len(keep)} labeled first-seen records carry both, fewer than {sv.MIN_N}"
        return out
    a = np.array([k[0] for k in keep], float)
    b = np.array([k[1] for k in keep], float)
    y = np.array([k[2] for k in keep], float)
    days = [k[3] for k in keep]
    boot = sv.DayBootstrap(days, seed=sv.SEED + seed_offset)
    st_a, st_b = sv.ic_stats(a, y, days, boot), sv.ic_stats(b, y, days, boot)
    ic_b, ic_a, se = sv._paired_gain(b, a, y, days, boot)
    out.update({"status": "OK", "n_days": len(set(days)), "ic_a": st_a.get("ic"), "ci95_a": st_a.get("ci95"),
                "ic_b": st_b.get("ic"), "ci95_b": st_b.get("ci95"), "ic_b_minus_a": round(ic_b - ic_a, 4),
                "se_paired": round(se, 4) if se else None,
                "b_better": bool(se and (ic_b - ic_a) / se > 1.96), "b_worse": bool(se and (ic_b - ic_a) / se < -1.96)})
    return out


def serve_fix_summary(records: Sequence[dict], pop_rows: Optional[List[dict]] = None) -> dict:
    """The B3 shadow (serve_features.py): how far the corrected feature vector
    moves the v2 probability, and — once labeled — whether it ranks outcomes
    at least as well as the served vector (paired, first-seen records)."""
    n, modes, deltas, changed = 0, Counter(), [], Counter()
    for rec in records:
        sf = rec.get("serve_fix") if isinstance(rec, dict) else None
        if not isinstance(sf, dict):
            continue
        n += 1
        modes[str(sf.get("mode"))] += 1
        a, b = _num(sf.get("ml_prob_current")), _num(sf.get("ml_prob_fixed"))
        if a is not None and b is not None:
            deltas.append(abs(b - a))
        for k in (sf.get("features_fixed_diff") or {}):
            changed[str(k)] += 1
    out: dict = {"records": n, "modes": dict(modes)}
    if deltas:
        ds = sorted(deltas)
        out["prob_delta"] = {"n": len(ds), "mean": round(sum(ds) / len(ds), 4), "p50": round(ds[len(ds) // 2], 4),
                             "p90": round(ds[min(len(ds) - 1, int(0.9 * len(ds)))], 4),
                             "share_over_0_01": round(sum(1 for d in ds if d > 0.01) / len(ds), 3)}
    out["features_changed"] = dict(changed.most_common(10))

    def ga(rec):
        sf = rec.get("serve_fix") if isinstance(rec.get("serve_fix"), dict) else {}
        return _num(sf.get("ml_prob_current"))

    def gb(rec):
        sf = rec.get("serve_fix") if isinstance(rec.get("serve_fix"), dict) else {}
        return _num(sf.get("ml_prob_fixed"))

    if pop_rows is None:
        pop_rows = _pop_rows(records)
    out["oos"] = paired_compare(pop_rows, ga, gb, "ml_prob_current (served features)", "ml_prob_fixed (training formulas)")
    return out


def composite_comparison(pop_rows: List[dict]) -> dict:
    """Legacy vs v2 composite on the same labeled first-seen candidates."""
    def g(key):
        def f(rec):
            bd = rec.get("score_breakdown")
            return _num(bd.get(key)) if isinstance(bd, dict) else None
        return f
    return paired_compare(pop_rows, g(COMPOSITE_LEGACY_KEY), g(COMPOSITE_V2_KEY),
                          "composite_legacy", "composite_v2", seed_offset=11)


def blend_weights() -> dict:
    """The v2 ML blend weights deep_score reads (diagnostics.run_diagnostics
    writes them from the BOT's recent win rate)."""
    try:
        from storage_config import BLEND_TRACKER_PATH
        if os.path.exists(BLEND_TRACKER_PATH):
            with open(BLEND_TRACKER_PATH, encoding="utf-8") as fh:
                b = json.load(fh)
            return {"rule_weight": b.get("rule_weight"), "ml_weight": b.get("ml_weight"), "updated": b.get("updated"),
                    "source": "blend tracker (diagnostics.run_diagnostics)"}
        return {"rule_weight": 0.6, "ml_weight": 0.4, "source": "default (no blend tracker file)"}
    except Exception as e:  # report, never raise: the table must still render
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


# ── the table ────────────────────────────────────────────────────────────────

def build_attribution(records: Sequence[dict], walkforward: Optional[dict] = None, validation: Optional[dict] = None,
                      ml_v3: Optional[dict] = None, v3_forward: Optional[dict] = None, blend: Optional[dict] = None,
                      cfg=None, now: Optional[datetime] = None) -> dict:
    """The whole payload from already-loaded inputs (pure; see module docstring)."""
    import source_validation as sv
    now = now or datetime.now(timezone.utc)
    if validation is None:
        validation = sv.validate_sources(records)
    sources = validation.get("sources") or {}
    pop = (validation.get("population") or {}).get(sv.PRIMARY_OUTCOME) or {}
    wf = walkforward if isinstance(walkforward, dict) and walkforward.get("schema") == WALKFORWARD_SCHEMA else None
    wf_window = (wf or {}).get("window") or {}
    mt = (wf or {}).get("multiple_testing") or {}
    z_bonf = bonferroni_z(int(mt.get("ic_tests") or 18))
    cfg = composite_config(cfg)
    prim_rows = _pop_rows(records)   # first-seen +5d population, built once

    rows = []
    for s in SIGNAL_REGISTRY:
        lt = long_term_block(wf, s.get("lt"), s.get("lt_nm"), s.get("lt_reason"))
        fw = forward_block(sources, s.get("fw"))
        verdict = combine_verdict(lt.get("verdict"), fw.get("verdict"))
        action, action_text = action_for(s["id"], cfg)
        row = {"id": s["id"], "label": s["label"], "group": s["group"], "role": s["role"],
               "verdict": verdict, "evidence": evidence_of(lt, fw), "action": action, "action_text": action_text,
               "review": bool(verdict in ("anti_predictive", "noise", "dead") and action == "kept"),
               "reason": reason_text(verdict, lt, fw, wf_window, pop, z_bonf, NOT_LOGGED_REASON.get(s["id"])),
               "long_term": lt, "forward": fw}
        rows.append(row)

    # ML v2 (the live blend): forward IC of its logged probability + its fit's duplication.
    ref = validation.get("reference") or {}
    ml_fw = ref.get("ml_win_probability") or {}
    ml_fw_v = None
    if isinstance(ml_fw, dict) and ml_fw.get("n", 0) >= sv.MIN_N and ml_fw.get("ci95"):
        lo, hi = ml_fw["ci95"]
        ml_fw_v = "PASS" if lo > 0 else "FAIL_INVERTED" if hi < 0 else "FAIL"
    dup = training_duplication(records)
    ml_action, ml_action_text = action_for("ml_v2", cfg)
    v3 = ml_v3 if isinstance(ml_v3, dict) else {}
    ci3 = v3.get("auc_ci95_date_block")
    v3_lt = None
    if isinstance(ci3, (list, tuple)) and len(ci3) == 2 and _num(ci3[0]) is not None:
        v3_lt = "positive" if ci3[0] > 0.5 else "negative" if ci3[1] < 0.5 else "null"
    ml_verdict = combine_verdict(v3_lt, ml_fw_v) if (v3_lt or ml_fw_v) else "untested"
    ml_reason = ("Noise. " if ml_verdict == "noise" else f"{ml_verdict.replace('_', '-').capitalize()}. ")
    pooled = v3.get("pooled") or {}
    if v3_lt:
        ml_reason += (f"Honest out-of-sample test of the v3 model (purged walk-forward by date, {v3.get('rows', '?')} rows): "
                      f"AUC {_r(pooled.get('auc'), 3)} (95% CI {_r(ci3[0], 3)}–{_r(ci3[1], 3)}; 0.5 = no skill), "
                      f"log-loss {_r(pooled.get('log_loss'), 4)} vs {_r(pooled.get('base_rate_log_loss'), 4)} for always "
                      "predicting the base rate. ")
    ml_reason += ("The live v2 model has no honest out-of-sample number: its production fit learns from "
                  f"{dup['rows']} shadow rows that are {dup['unique_ticker_days']} unique candidate-days"
                  f"{' repeated ' + str(dup['duplication']) + ' times each' if dup.get('duplication') else ''}. ")
    ml_reason += (f"Forward: its logged probability has a +5-day IC of {_signed(ml_fw.get('ic'))} on {ml_fw.get('n', 0)} "
                  "first-seen records" + (" (too few for a verdict)." if ml_fw_v is None else "."))
    rows.append({"id": "ml_v2", "label": "ML blend (LightGBM v2)", "group": "model",
                 "role": f"blended at {(blend or {}).get('ml_weight', '?')} weight when its score is outside 42-58",
                 "verdict": ml_verdict, "evidence": "both" if (v3_lt and ml_fw_v) else "long_term" if v3_lt else "forward" if ml_fw_v else "none",
                 "action": ml_action, "action_text": ml_action_text,
                 "review": bool(ml_verdict in ("anti_predictive", "noise", "dead") and ml_action == "kept"),
                 "reason": ml_reason.strip(),
                 "long_term": {"measurable": bool(v3_lt), "source": "ml_model_v3 honest OOS (proxy: same feature families)",
                               "auc": _r(pooled.get("auc"), 4), "auc_ci95": ci3, "verdict": v3_lt or "insufficient"},
                 "forward": {"logged": True, "verdict": ml_fw_v or "INSUFFICIENT", "n": ml_fw.get("n", 0),
                             "ic": ml_fw.get("ic"), "ci95": ml_fw.get("ci95")}})

    # composites (reference rows: the comparison the strategy change is judged by)
    comp = composite_comparison(prim_rows)
    composites = []
    for c in COMPOSITE_ROWS:
        lt = long_term_block(wf, c["lt"])
        composites.append({"id": c["id"], "label": c["label"], "long_term": lt})
    served = "composite_v2" if (cfg and cfg.get("enabled")) else "composite_legacy"

    counts = Counter(r["verdict"] for r in rows)
    named = {v: [r["id"] for r in rows if r["verdict"] == v] for v in VERDICTS}
    qs = next((r for r in rows if r["id"] == "quick_score"), None)
    headline = (f"{len(rows)} signals: {counts.get('predictive', 0)} predictive, {counts.get('anti_predictive', 0)} "
                f"anti-predictive, {counts.get('noise', 0)} noise, {counts.get('dead', 0)} dead, "
                f"{counts.get('untested', 0)} untested, {counts.get('conflicting', 0)} conflicting.")
    if not counts.get("predictive"):
        headline += " No signal in the stock score has shown predictive power yet."
    if qs and (qs["long_term"].get("h5") or {}).get("top10_vs_random_pct") is not None:
        h5 = qs["long_term"]["h5"]
        headline += (f" The universe pre-rank's top-10 picks {'trailed' if h5['top10_vs_random_pct'] < 0 else 'beat'} random "
                     f"entry by {_pts(h5['top10_vs_random_pct'])} points per 5 days over the walk-forward.")
    tests_lt = int(mt.get("ic_tests") or 0)
    tests_fw = len([nm for nm, r in sources.items() if isinstance(r, dict) and "q_bh" in r])
    return {
        "verdict_version": VERDICT_VERSION,
        "generated_at": now.isoformat(),
        "headline": headline,
        "counts": dict(counts),
        "by_verdict": named,
        "served_composite": served,
        "composite_config": ({k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in cfg.items()} if cfg else None),
        "signals": rows,
        "composites": composites,
        "composite_comparison": comp,
        "ml": {
            "v2_live": {"blend": blend or {}, "training_set": dup, "forward": {k: ml_fw.get(k) for k in ("n", "n_days", "ic", "ci95", "p") if k in ml_fw},
                        "honest_cv_note": ("2026-10-28 (v1.0.1215): the v2 architecture re-tested with folds keyed on the "
                                           "decision DATE (embargo + purge, early stopping inside the training window) "
                                           "scored AUC 0.515 vs 0.658 under the live row-index CV on the same 90,210 rows "
                                           "— no out-of-time skill.")},
            "v3_candidate": {**{k: v3.get(k) for k in ("candidate_exists", "promoted", "trained_at", "rows", "date_range", "auc_ci95_date_block", "variants_tried") if k in v3},
                             "pooled": {k: pooled.get(k) for k in ("n", "auc", "log_loss", "base_rate_log_loss") if k in pooled},
                             "forward": v3_forward or {}},
            "serve_fix": serve_fix_summary(records, prim_rows),
            "features": {k: list(v)[:40] for k, v in (validation.get("summary") or {}).items()},
        },
        "multiple_testing": {
            "long_term": {"family": "walkforward inputs x horizons, Bonferroni", "tests": tests_lt,
                          "bonferroni_z": round(z_bonf, 3), "alpha": FAMILY_ALPHA},
            "forward": {"family": "source_validation sources (score components + model features), Benjamini-Hochberg",
                        "tests": tests_fw, "fdr": sv.FDR_Q},
            "note": ("A verdict needs the harness's own correction: Bonferroni over every walk-forward input x horizon, "
                     "BH at FDR 0.10 over every forward source. Top-10 vs random entry is flagged only beyond the same "
                     "Bonferroni bar. Composites and the ML probability are reference rows outside both families."),
        },
        "sources": {
            "long_term": ({"generated_at": wf.get("generated_at"), "as_of": wf.get("as_of"), "code_version": wf.get("code_version"),
                           "window": {k: wf_window.get(k) for k in ("first_rebalance", "last_rebalance", "n_rebalances_scored", "horizons_days")},
                           "names_per_date": ((wf.get("universe") or {}).get("names_per_date") or {}).get("mean"),
                           "delisted_symbol_share": (wf.get("universe") or {}).get("delisted_symbol_share"),
                           "baselines": wf.get("baselines")}
                          if wf else {"error": "no walk-forward result (walkforward.py has not completed a run)"}),
            "forward": {"population": pop, "method": (validation.get("method") or {}).get("prereg_version"),
                        "outcome": "old ruler +5d path return (+2% target / -4% stop / timeout close), long-stock",
                        "label_lag": "a +5d label needs 13 calendar days (shadow_portfolio HORIZON_CALENDAR_BUFFER)"},
        },
        "not_measured": {
            "+1d": ("The shadow log has no +1d label; the walk-forward reports +1d (descriptive) from its first run "
                    "that includes horizon 1."),
            "deciles": "Neither harness reports deciles: quintiles long-term, terciles forward (n too small).",
        },
    }


def _read_walkforward(path: Optional[str] = None) -> Optional[dict]:
    if path is None:
        import storage_config
        path = os.path.join(storage_config.WALKFORWARD_DIR, "latest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return d if isinstance(d, dict) else None


def get_signal_attribution() -> dict:
    """Production entry point (daemon "signal_attribution", diag probe
    "signals", GET /api/bot/signals): the volume's shadow log, the walk-forward
    result and the v3 status, read-only. One load of the log."""
    import source_validation as sv
    from shadow_portfolio import _load_shadow_log
    errors = {}
    records = _load_shadow_log()
    try:
        wf = _read_walkforward()
    except Exception as e:  # unreadable result file: the forward half still renders
        wf, errors["walkforward"] = None, f"{type(e).__name__}: {str(e)[:150]}"
    try:
        from brain_evidence import _ml_v3_summary, v3_shadow_ic
        v3, v3_fw = _ml_v3_summary(), v3_shadow_ic(records)
    except Exception as e:  # v3 files absent or unreadable: reported, not fatal
        v3, v3_fw, errors["ml_v3"] = None, None, f"{type(e).__name__}: {str(e)[:150]}"
    validation = sv.validate_sources(records)
    out = build_attribution(records, walkforward=wf, validation=validation, ml_v3=v3, v3_forward=v3_fw,
                            blend=blend_weights())
    out["shadow_log"] = {"records": len(records)}
    if errors:
        out["errors"] = errors
    return out


if __name__ == "__main__":  # python3 signal_attribution.py
    print(json.dumps(get_signal_attribution(), default=str))
