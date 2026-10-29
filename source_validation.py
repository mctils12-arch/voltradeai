"""
source_validation.py — [RESEARCH] 2026-10-27, Phase 5 "the brain", step 1.

Question: does each data source the scorer consumes predict forward returns
ON ITS OWN? deep_score adds news, social, insider/intel, alt-data, macro,
fundamentals and options-derived terms to every candidate, and the shadow log
says the result ranks BACKWARDS (2026-10-27 deduped population, 331
ticker-days: score quintile Q1 -1.07% ... Q5 -1.74% at +5d). Before any
ensemble is believed, every input is tested alone, against outcomes, with the
multiple-testing correction its number of tests demands.

Population: the shadow log (shadow_portfolio, voltrade_shadow_candidates.json
on the production volume), deep-scored records only (non-empty `features` —
the tier-1 masterkill CSP rows carry score 0 and no features), deduplicated
to ONE first-seen record per (ticker, ET entry day)
(shadow_portfolio._first_seen_per_ticker_day, the same rule as the
shadow_bands_dedup probe), separately per outcome.

Outcomes: "+5d"/"+10d"/"+20d" = the old ruler's path-dependent return_pct
(+2% target / -4% stop / timeout close — a capped path return, NOT a raw
close-to-close return); "xa" = the exit-aligned ruler (live exit ladder).
All are LONG-stock outcomes.

PRE-REGISTERED (written before any production run; changing one after
seeing results is the fishing CLAUDE.md forbids — bump PREREG_VERSION and
say why):
  * PRIMARY outcome "+5d" (the horizon of the observed inversion); the other
    three are descriptive only and never enter a verdict.
  * IC = Spearman rank correlation, source value vs outcome, pooled over the
    deduplicated population. 95% CI and standard error from a DAY-CLUSTERED
    bootstrap (resample entry days with replacement — names scored the same
    day share one market move, so rows are not independent); p from the
    bootstrap SE (two-sided normal).
  * Family = every SOURCE below (score components + model features) on the
    primary outcome; Benjamini-Hochberg at FDR_Q = 0.10 across the family.
    The composite scores (final score, rules-only, ML) are REFERENCE rows,
    outside the family.
  * Verdicts need n >= MIN_N unique ticker-days (else INSUFFICIENT) and a
    non-constant source (else CONSTANT — a dead input).
    - adjustment sources (intended direction +: a higher adjustment should
      mean a better outcome): PASS iff IC >= IC_MIN, BH q <= FDR_Q, CI > 0
      and top-minus-bottom spread > 0; FAIL_INVERTED iff IC <= -IC_MIN,
      q <= FDR_Q and CI < 0 (it predicts, backwards — the + it adds hurts);
      FAIL otherwise.
    - feature sources (no intended sign; the model may use either):
      PASS iff |IC| >= IC_MIN, q <= FDR_Q, CI excludes 0 and the spread
      has the IC's sign; FAIL otherwise.
  * By regime (5-level classify_regime_5level from the record's vxx_ratio
    and features.spy_vs_ma50 — records carry no 200-day inputs): IC and CI
    per regime with n >= MIN_N_REGIME, descriptive only.

Market-wide features (vxx_ratio, spy_vs_ma50, ...) take the same value for
every name on a day: their pooled IC is a TIMING signal, not a cross-
sectional one (flagged `market_wide`).

Aggregate-only output (no ticker, price or per-record timestamp) — served
read-only by the diag probe "source_validation"; scripts/source_validation.py
is the local CLI over a copied log. scripts/ is NOT in the production image
(Dockerfile copies *.py only), so the logic lives here at the repo root.
Pure: no network, no writes.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# v2 (same day, before any production run): + the four deep_score terms the
# breakdown did not persist (logged from 2026-10-27), + selection/ablation.
PREREG_VERSION = "source_validation_v2 (2026-10-27)"
PRIMARY_OUTCOME = "+5d"
SECONDARY_OUTCOMES = ("+10d", "+20d", "xa")
MIN_N = 100            # unique ticker-days for a verdict
MIN_N_REGIME = 30      # per-regime descriptive cell
MIN_DAYS = 5           # distinct entry days needed for a clustered bootstrap
MIN_BUCKET = 20        # rows per tercile / sign bucket for a spread
FDR_Q = 0.10           # Benjamini-Hochberg false-discovery rate
IC_MIN = 0.05          # economic floor on |IC|
N_BOOT = 500
SEED = 20261027

# deep_score's persisted sub-scores / adjustments (shadow_portfolio.
# SCORE_BREAKDOWN_KEYS; on records since 2026-10-26, v1.0.1161).
# (key, group). Strategy sub-scores are 0-100 and enter with a weight; the
# rest are additive points.
ADJUSTMENT_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("momentum", "technical"),
    ("mean_reversion", "technical"),
    ("vrp", "options_derived"),
    ("squeeze", "short_interest_sentiment"),
    ("volume", "technical"),
    ("vol_premium", "options_derived"),
    ("macro", "macro"),
    ("news", "news"),
    ("adx", "technical"),
    ("obv", "technical"),
    ("intel", "news_insider_earnings"),
    ("alt", "alt_data"),
    ("social", "social"),
    ("fundamental", "fundamentals"),
    # shadow_portfolio.SCORE_BREAKDOWN_EXTRA_KEYS (logged since 2026-10-27;
    # INSUFFICIENT until labeled). insider_mspr is also inside `intel`.
    ("recommendation", "analyze_recommendation"),
    ("contrarian", "sentiment"),
    ("rel_strength", "technical"),
    ("insider_mspr", "insider"),
)

# The ML feature vector persisted on every shadow record
# (ml_model_v2.FEATURE_COLS — pinned by test_source_validation.py).
FEATURE_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("momentum_1m", "technical"), ("momentum_3m", "technical"), ("rsi_14", "technical"),
    ("price_vs_52w_high", "technical"), ("volume_ratio", "technical"), ("ewma_vol", "technical"),
    ("atr_pct", "technical"), ("vrp", "options_derived"), ("iv_rank_proxy", "options_derived"),
    ("vwap_position", "technical"), ("adx", "technical"), ("range_pct", "technical"),
    ("vxx_ratio", "macro"), ("spy_vs_ma50", "macro"), ("markov_state", "macro"),
    ("regime_score", "macro"), ("sector_momentum", "technical"), ("cross_sec_rank", "technical"),
    ("earnings_surprise", "news_insider_earnings"), ("put_call_proxy", "options_derived"),
    ("vol_of_vol", "macro"), ("frac_diff_price", "technical"), ("idiosyncratic_ret", "technical"),
    ("change_pct_today", "technical"), ("above_ma10", "technical"), ("trend_strength", "technical"),
    ("intel_score", "news_insider_earnings"), ("news_sentiment", "news"),
    ("insider_signal", "insider"), ("iv_rank_stock", "options_derived"),
    ("days_to_earnings", "news_insider_earnings"), ("credit_spread", "macro"),
    ("market_breadth", "macro"), ("put_call_ratio", "macro"),
)
MARKET_WIDE_FEATURES = frozenset({
    "vxx_ratio", "spy_vs_ma50", "markov_state", "regime_score", "vol_of_vol",
    "credit_spread", "market_breadth", "put_call_ratio",
})


def _num(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _adj_getter(key: str) -> Callable[[dict], Optional[float]]:
    def g(rec: dict) -> Optional[float]:
        bd = rec.get("score_breakdown")
        return _num(bd.get(key)) if isinstance(bd, dict) else None
    return g


def _feat_getter(key: str) -> Callable[[dict], Optional[float]]:
    def g(rec: dict) -> Optional[float]:
        f = rec.get("features")
        return _num(f.get(key)) if isinstance(f, dict) and key in f else None
    return g


def _ml_getter(key: str) -> Callable[[dict], Optional[float]]:
    def g(rec: dict) -> Optional[float]:
        ml = rec.get("ml")
        return _num(ml.get(key)) if isinstance(ml, dict) else None
    return g


# Composite scores: reported for reference, outside the BH family.
REFERENCE_SOURCES: Tuple[Tuple[str, Callable[[dict], Optional[float]]], ...] = (
    ("score", lambda r: _num(r.get("score"))),
    ("rules_only_score", _ml_getter("rules_only_score")),
    ("ml_only_score", _ml_getter("ml_only_score")),
    ("ml_win_probability", _ml_getter("win_probability")),
)


def source_registry() -> List[dict]:
    """The pre-registered test family, in a fixed order."""
    reg = []
    for key, group in ADJUSTMENT_SOURCES:
        reg.append({"name": f"adj.{key}", "group": group, "kind": "adjustment",
                    "market_wide": False, "get": _adj_getter(key)})
    for key, group in FEATURE_SOURCES:
        reg.append({"name": f"feat.{key}", "group": group, "kind": "feature",
                    "market_wide": key in MARKET_WIDE_FEATURES, "get": _feat_getter(key)})
    return reg


# ── outcomes and population ──────────────────────────────────────────────────

def outcome_value(rec: dict, outcome: str) -> Optional[float]:
    if outcome == "xa":
        from shadow_portfolio import XA_KEY
        o = rec.get(XA_KEY)
    else:
        outs = rec.get("outcomes")
        o = outs.get(outcome) if isinstance(outs, dict) else None
    if not isinstance(o, dict) or o.get("label") not in (0, 1):
        return None
    return _num(o.get("return_pct"))


def _entry_day(rec: dict) -> Optional[str]:
    from shadow_portfolio import _xa_entry_date_et
    return _xa_entry_date_et(rec.get("timestamp", ""))


def _regime_of(rec: dict) -> str:
    from regime_util import classify_regime_5level
    feats = rec.get("features") if isinstance(rec.get("features"), dict) else {}
    vxx = _num(rec.get("vxx_ratio")) or 1.0
    spy = _num(feats.get("spy_vs_ma50")) or 1.0
    return classify_regime_5level(vxx, spy)


def is_deep_scored(rec: dict) -> bool:
    f = rec.get("features")
    return isinstance(f, dict) and len(f) > 0


def build_population(records: Sequence[dict], outcome: str) -> List[dict]:
    """Deduped first-seen (ticker, ET day) deep-scored records labeled by
    `outcome`, oldest first, as rows {rec, y, day, regime}."""
    from shadow_portfolio import _first_seen_per_ticker_day
    deep = [r for r in records if isinstance(r, dict) and is_deep_scored(r)]
    dd = _first_seen_per_ticker_day(deep, lambda r: outcome_value(r, outcome) is not None)
    rows = []
    for r in dd:
        day = _entry_day(r)
        if not day:
            continue
        rows.append({"rec": r, "y": outcome_value(r, outcome), "day": day,
                     "regime": _regime_of(r), "ts": str(r.get("timestamp", ""))})
    rows.sort(key=lambda x: x["ts"])
    return rows


# ── statistics ──────────────────────────────────────────────────────────────

def _rankdata(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(a, method="average")


class DayBootstrap:
    """Day-clustered bootstrap weights for one population: draw n_days days
    with replacement N_BOOT times; a row's weight in a replicate is the count
    of its entry day. Shared by every source tested on that population so
    replicates are comparable (and paired for differences)."""

    def __init__(self, days: Sequence[str], n_boot: int = N_BOOT, seed: int = SEED):
        uniq = sorted(set(days))
        self.day_index = {d: i for i, d in enumerate(uniq)}
        self.n_days = len(uniq)
        rng = np.random.default_rng(seed)
        if self.n_days:
            self.counts = rng.multinomial(self.n_days, np.full(self.n_days, 1.0 / self.n_days),
                                          size=n_boot).astype(float)
        else:
            self.counts = np.zeros((n_boot, 0))

    def weights(self, days: Sequence[str]) -> np.ndarray:
        idx = np.fromiter((self.day_index[d] for d in days), dtype=int, count=len(days))
        return self.counts[:, idx]


def _weighted_corr_rows(W: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = W @ a / s
        mb = W @ b / s
        cov = W @ (a * b) / s - ma * mb
        va = W @ (a * a) / s - ma * ma
        vb = W @ (b * b) / s - mb * mb
        r = cov / np.sqrt(va * vb)
    r[~np.isfinite(r)] = np.nan
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def ic_stats(x: np.ndarray, y: np.ndarray, days: Sequence[str], boot: DayBootstrap) -> dict:
    """Spearman IC with a day-clustered bootstrap SE / 95% CI / p. The
    replicate IC is the weighted Pearson of the full-sample ranks (the
    standard fast approximation to re-ranking each replicate)."""
    n = len(x)
    n_days = len(set(days))
    out = {"n": n, "n_days": n_days}
    if n < 3:
        return out
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        out["constant"] = True
        return out
    ic = float(np.corrcoef(rx, ry)[0, 1])
    out["ic"] = round(ic, 4)
    if n_days < MIN_DAYS:
        return out
    reps = _weighted_corr_rows(boot.weights(days), rx, ry)
    reps = reps[np.isfinite(reps)]
    if len(reps) < 50:
        return out
    se = float(np.std(reps, ddof=1))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    out["se"] = round(se, 4)
    out["ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    out["p"] = float(math.erfc(abs(ic) / se / math.sqrt(2))) if se > 0 else (0.0 if ic else 1.0)
    return out


def tercile_spread(x: np.ndarray, y: np.ndarray) -> dict:
    """Mean outcome of the top vs bottom third of the source's ranks (ties
    share a rank, so a tied block lands in one tercile)."""
    n = len(x)
    if n < 3:
        return {"spread": None}
    f = (_rankdata(x) - 0.5) / n
    bot, top = y[f < 1 / 3], y[f >= 2 / 3]
    res = {"n_bottom": int(len(bot)), "n_top": int(len(top)),
           "mean_bottom": round(float(bot.mean()), 3) if len(bot) else None,
           "mean_top": round(float(top.mean()), 3) if len(top) else None,
           "spread": None}
    if len(bot) >= MIN_BUCKET and len(top) >= MIN_BUCKET:
        res["spread"] = round(float(top.mean() - bot.mean()), 3)
    return res


def sign_buckets(x: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for name, m in (("pos", x > 0), ("zero", x == 0), ("neg", x < 0)):
        k = int(m.sum())
        out[name] = {"n": k, "mean": round(float(y[m].mean()), 3) if k else None}
    out["spread"] = (round(out["pos"]["mean"] - out["neg"]["mean"], 3)
                     if out["pos"]["n"] >= MIN_BUCKET and out["neg"]["n"] >= MIN_BUCKET else None)
    return out


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = min(1.0, prev)
    return q


def _extract(rows: List[dict], getter) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    xs, ys, ds, rg = [], [], [], []
    for row in rows:
        v = getter(row["rec"])
        if v is None:
            continue
        xs.append(v)
        ys.append(row["y"])
        ds.append(row["day"])
        rg.append(row["regime"])
    return np.asarray(xs, float), np.asarray(ys, float), ds, rg


def _test_source(rows, boot, getter, kind: str) -> dict:
    x, y, days, regimes = _extract(rows, getter)
    res = ic_stats(x, y, days, boot)
    if len(x):
        mode_share = Counter(x.tolist()).most_common(1)[0][1] / len(x)
        res["distinct_values"] = int(len(set(x.tolist())))
        res["mode_share"] = round(mode_share, 3)
    res["terciles"] = tercile_spread(x, y)
    if kind == "adjustment":
        res["sign_buckets"] = sign_buckets(x, y)
    by_reg = {}
    reg_arr = np.asarray(regimes)
    for reg in sorted(set(regimes)):
        m = reg_arr == reg
        if int(m.sum()) < MIN_N_REGIME:
            by_reg[reg] = {"n": int(m.sum())}
            continue
        by_reg[reg] = ic_stats(x[m], y[m], [d for d, k in zip(days, m) if k], boot)
        by_reg[reg].pop("p", None)
    res["by_regime"] = by_reg
    return res


def _spread_of(res: dict, kind: str) -> Optional[float]:
    s = (res.get("terciles") or {}).get("spread")
    if s is None and kind == "adjustment":
        s = (res.get("sign_buckets") or {}).get("spread")
    return s


def verdict(res: dict, kind: str, q: Optional[float]) -> Tuple[str, str]:
    n = res.get("n", 0)
    if res.get("constant") or res.get("distinct_values", 2) < 2:
        return "CONSTANT", "source never varies on this population (dead input)"
    if n < MIN_N:
        return "INSUFFICIENT", f"n={n} < MIN_N={MIN_N} unique ticker-days"
    ic, ci = res.get("ic"), res.get("ci95")
    if ic is None or ci is None or q is None:
        return "INSUFFICIENT", f"no clustered CI (n_days={res.get('n_days')} < {MIN_DAYS} or degenerate)"
    sig = q <= FDR_Q
    spread = _spread_of(res, kind)
    if kind == "adjustment":
        if ic >= IC_MIN and sig and ci[0] > 0 and spread is not None and spread > 0:
            return "PASS", "positive, BH-significant, CI > 0, top beats bottom"
        if ic <= -IC_MIN and sig and ci[1] < 0:
            return "FAIL_INVERTED", "BH-significant NEGATIVE IC: the + this adjustment adds predicts WORSE outcomes"
        return "FAIL", "no BH-significant positive IC above the floor"
    if abs(ic) >= IC_MIN and sig and (ci[0] > 0 or ci[1] < 0) and spread is not None \
            and (spread > 0) == (ic > 0):
        return "PASS", f"BH-significant {'positive' if ic > 0 else 'negative'} IC, spread agrees"
    return "FAIL", "no BH-significant IC above the floor"


def _population_summary(rows: List[dict]) -> dict:
    days = sorted({r["day"] for r in rows})
    return {"n_unique": len(rows), "n_days": len(days),
            "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
            "regimes": dict(Counter(r["regime"] for r in rows)),
            "with_score_breakdown": sum(1 for r in rows if isinstance(r["rec"].get("score_breakdown"), dict)),
            "mean_outcome_pct": round(float(np.mean([r["y"] for r in rows])), 3) if rows else None}


def validate_sources(records: Sequence[dict]) -> dict:
    """Every source alone vs forward outcomes (see module docstring)."""
    reg = source_registry()
    pops = {o: build_population(records, o) for o in (PRIMARY_OUTCOME,) + SECONDARY_OUTCOMES}
    boots = {o: DayBootstrap([r["day"] for r in rows]) for o, rows in pops.items()}

    prim_rows, prim_boot = pops[PRIMARY_OUTCOME], boots[PRIMARY_OUTCOME]
    results: Dict[str, dict] = {}
    for s in reg:
        res = _test_source(prim_rows, prim_boot, s["get"], s["kind"])
        res.update({"group": s["group"], "kind": s["kind"]})
        if s["market_wide"]:
            res["market_wide"] = True
        sec = {}
        for o in SECONDARY_OUTCOMES:
            x, y, d, _ = _extract(pops[o], s["get"])
            st = ic_stats(x, y, d, boots[o])
            sec[o] = {k: st[k] for k in ("n", "ic", "ci95") if k in st}
        res["secondary"] = sec
        results[s["name"]] = res

    # BH across the family: every source with a computable p.
    tested = [nm for nm, r in results.items() if "p" in r and r.get("n", 0) >= MIN_N]
    qs = benjamini_hochberg([results[nm]["p"] for nm in tested])
    qmap = dict(zip(tested, qs))
    summary: Dict[str, List[str]] = {"PASS": [], "FAIL_INVERTED": [], "FAIL": [],
                                     "INSUFFICIENT": [], "CONSTANT": []}
    for nm, r in results.items():
        q = qmap.get(nm)
        if q is not None:
            r["q_bh"] = round(q, 4)
        if "p" in r:
            r["p"] = round(r["p"], 5)
        v, why = verdict(r, r["kind"], q)
        r["verdict"], r["verdict_reason"] = v, why
        summary[v].append(nm)

    reference = {}
    for nm, getter in REFERENCE_SOURCES:
        res = _test_source(prim_rows, prim_boot, getter, "feature")
        if "p" in res:
            res["p"] = round(res["p"], 5)
        reference[nm] = res

    return {
        "method": {
            "prereg_version": PREREG_VERSION,
            "primary_outcome": PRIMARY_OUTCOME,
            "secondary_outcomes": list(SECONDARY_OUTCOMES),
            "ic": "Spearman, pooled over the deduped first-seen (ticker, ET day) deep-scored population",
            "ci": f"day-clustered bootstrap, {N_BOOT} replicates, seed {SEED}",
            "multiple_testing": f"Benjamini-Hochberg across {len(tested)} sources at FDR {FDR_Q}",
            "thresholds": {"MIN_N": MIN_N, "MIN_N_REGIME": MIN_N_REGIME, "MIN_DAYS": MIN_DAYS,
                           "MIN_BUCKET": MIN_BUCKET, "IC_MIN": IC_MIN, "FDR_Q": FDR_Q},
            "outcome_note": "+Nd = old ruler path return (+2% target / -4% stop / timeout close); "
                            "xa = exit-aligned ruler; LONG-stock outcomes",
            "regime_note": "classify_regime_5level(record vxx_ratio, features.spy_vs_ma50); no 200d inputs on records",
        },
        "population": {o: _population_summary(rows) for o, rows in pops.items()},
        "summary": summary,
        "sources": results,
        "reference": reference,
    }


# ══════════════════════════════════════════════════════════════════════════
# 2026-10-27 [RESEARCH] step 2: forward selection + ablation (same harness)
# ══════════════════════════════════════════════════════════════════════════
#
# FORWARD SELECTION (pre-registered): rows of the primary population sorted
# by time and split by ENTRY DAY into FIT (first 50% of days), SELECT (next
# 25%) and HOLDOUT (last 25%) — no day straddles two blocks. A model is a
# ridge regression (RIDGE_LAMBDA) on FIT-standardized source values. Greedy:
# at each step every remaining candidate is added in turn, fit on FIT and
# scored by Spearman IC on SELECT; the best one is kept only if its IC gain
# is >= MIN_GAIN and its day-clustered paired-bootstrap p-value, BONFERRONI-
# multiplied by the number of candidates tried at that step (the winner of
# m tries is the max of m), is <= SELECT_ALPHA. The selected set is refit on
# FIT+SELECT and scored ONCE on HOLDOUT, beside the live score on the same
# rows. Two arms: "features" (every non-constant model feature, all rows)
# and "breakdown" (adjustments + features, rows that carry score_breakdown).
#
# ABLATION: recompute deep_score with groups of score adjustments removed,
# from the logged components: rules = clamp(rules_raw - removed) where
# rules_raw is the logged pre-clamp total (score_breakdown.rules_raw, logged
# since 2026-10-27) or else the ML trace's rules_only_score (exact unless
# the rules score was clamped at 0/100 — counted as `approx_clamped`); then
# the logged ML blend (rule_weight/ml_weight, 50 when gated) is re-applied.
# Rows whose reconstructed baseline differs from the logged score by more
# than RECON_TOL are dropped and counted. Reported per variant: IC with CI,
# the paired IC change vs baseline, score-quintile mean outcomes, and
# whether removing the adjustments un-inverts the score-vs-outcome relation.

SPLIT_FRACTIONS = (0.50, 0.25, 0.25)
RIDGE_LAMBDA = 1.0
MIN_GAIN = 0.02
SELECT_ALPHA = 0.05
MAX_SELECT_STEPS = 8
MIN_SELECT_ROWS = 300
MIN_SPLIT_ROWS = 50
RECON_TOL = 0.6
UNVALIDATED_ADJUSTMENTS = ("intel", "alt", "social", "news", "fundamental")


def _day_split(rows: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    days = sorted({r["day"] for r in rows})
    n = len(days)
    a = int(round(n * SPLIT_FRACTIONS[0]))
    b = int(round(n * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1])))
    fit_d, sel_d = set(days[:a]), set(days[a:b])
    fit = [r for r in rows if r["day"] in fit_d]
    sel = [r for r in rows if r["day"] in sel_d]
    hold = [r for r in rows if r["day"] not in fit_d and r["day"] not in sel_d]
    return fit, sel, hold


def _matrix(rows: List[dict], getters: List[Callable]) -> np.ndarray:
    return np.array([[g(r["rec"]) for g in getters] for r in rows], dtype=float)


def _fit_ridge(X: np.ndarray, y: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.clip((X - mu) / sd, -5, 5)
    yc = y - y.mean()
    k = Z.shape[1]
    beta = np.linalg.solve(Z.T @ Z + RIDGE_LAMBDA * np.eye(k), Z.T @ yc)
    return mu, sd, beta


def _predict(model, X: np.ndarray) -> np.ndarray:
    mu, sd, beta = model
    return np.clip((X - mu) / sd, -5, 5) @ beta


def _paired_gain(pred_new: np.ndarray, pred_old: Optional[np.ndarray], y: np.ndarray,
                 days: List[str], boot: DayBootstrap) -> Tuple[float, float, Optional[float]]:
    """IC(pred_new) - IC(pred_old) on these rows, with its day-clustered
    bootstrap SE (same replicates for both, so the difference is paired)."""
    ry = _rankdata(y)
    W = boot.weights(days)
    rn = _rankdata(pred_new)
    ic_new = float(np.corrcoef(rn, ry)[0, 1]) if np.std(rn) > 0 else 0.0
    reps = _weighted_corr_rows(W, rn, ry)
    ic_old = 0.0
    if pred_old is not None:
        ro = _rankdata(pred_old)
        if np.std(ro) > 0:
            ic_old = float(np.corrcoef(ro, ry)[0, 1])
            reps = reps - _weighted_corr_rows(W, ro, ry)
    reps = reps[np.isfinite(reps)]
    se = float(np.std(reps, ddof=1)) if len(reps) > 50 else None
    return ic_new, ic_old, se


def forward_selection(rows: List[dict], candidates: List[dict]) -> dict:
    """See the block comment above. `candidates`: registry entries."""
    out = {"n_rows": len(rows), "candidates": len(candidates)}
    if len(rows) < MIN_SELECT_ROWS or not candidates:
        out["status"] = "INSUFFICIENT"
        out["reason"] = f"{len(rows)} rows < MIN_SELECT_ROWS={MIN_SELECT_ROWS}" if candidates else "no candidates"
        return out
    fit, sel, hold = _day_split(rows)
    out["split"] = {nm: {"n": len(p), "days": len({r['day'] for r in p}),
                         "first_day": min((r["day"] for r in p), default=None),
                         "last_day": max((r["day"] for r in p), default=None)}
                    for nm, p in (("fit", fit), ("select", sel), ("holdout", hold))}
    if min(len(fit), len(sel), len(hold)) < MIN_SPLIT_ROWS or \
            min(out["split"][k]["days"] for k in out["split"]) < MIN_DAYS:
        out["status"] = "INSUFFICIENT"
        out["reason"] = f"a split has < {MIN_SPLIT_ROWS} rows or < {MIN_DAYS} days"
        return out

    getters = {c["name"]: c["get"] for c in candidates}
    y_fit = np.array([r["y"] for r in fit])
    y_sel = np.array([r["y"] for r in sel])
    sel_days = [r["day"] for r in sel]
    sel_boot = DayBootstrap(sel_days, seed=SEED + 1)
    X_fit = {nm: _matrix(fit, [g])[:, 0] for nm, g in getters.items()}
    X_sel = {nm: _matrix(sel, [g])[:, 0] for nm, g in getters.items()}

    chosen: List[str] = []
    pred_old: Optional[np.ndarray] = None
    steps = []
    for _step in range(MAX_SELECT_STEPS):
        remaining = [nm for nm in getters if nm not in chosen]
        if not remaining:
            break
        best = None
        for nm in remaining:
            cols = chosen + [nm]
            model = _fit_ridge(np.column_stack([X_fit[c] for c in cols]), y_fit)
            pred = _predict(model, np.column_stack([X_sel[c] for c in cols]))
            r = _rankdata(pred)
            ic = float(np.corrcoef(r, _rankdata(y_sel))[0, 1]) if np.std(r) > 0 else 0.0
            if best is None or ic > best[1]:
                best = (nm, ic, pred)
        nm, _ic, pred = best
        ic_new, ic_old, se = _paired_gain(pred, pred_old, y_sel, sel_days, sel_boot)
        gain = ic_new - ic_old
        p_one = 0.5 * math.erfc(gain / se / math.sqrt(2)) if se else 1.0
        p_adj = min(1.0, p_one * len(remaining))
        accepted = gain >= MIN_GAIN and p_adj <= SELECT_ALPHA
        steps.append({"candidate": nm, "select_ic": round(ic_new, 4), "gain": round(gain, 4),
                      "se": round(se, 4) if se else None, "p_bonferroni": round(p_adj, 5),
                      "tried": len(remaining), "accepted": accepted})
        if not accepted:
            break
        chosen.append(nm)
        pred_old = pred
    out["steps"] = steps
    out["selected"] = chosen

    # One look at HOLDOUT.
    trainval = fit + sel
    y_hold = np.array([r["y"] for r in hold])
    hold_days = [r["day"] for r in hold]
    hold_boot = DayBootstrap(hold_days, seed=SEED + 2)
    live = np.array([_num(r["rec"].get("score")) or 0.0 for r in hold])
    out["holdout_live_score"] = ic_stats(live, y_hold, hold_days, hold_boot)
    if not chosen:
        out["status"] = "NOTHING_SELECTED"
        out["reason"] = "no source added significant out-of-sample IC on the SELECT block"
        return out
    model = _fit_ridge(_matrix(trainval, [getters[c] for c in chosen]),
                       np.array([r["y"] for r in trainval]))
    pred = _predict(model, _matrix(hold, [getters[c] for c in chosen]))
    out["holdout_selected"] = ic_stats(pred, y_hold, hold_days, hold_boot)
    ic_n, ic_l, se = _paired_gain(pred, live, y_hold, hold_days, hold_boot)
    out["holdout_vs_live_score"] = {"ic_gain": round(ic_n - ic_l, 4), "se": round(se, 4) if se else None}
    out["weights"] = {c: round(float(b), 4) for c, b in zip(chosen, model[2])}
    out["status"] = "SELECTED"
    return out


def _clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


def ablated_score(rec: dict, remove: Sequence[str]) -> Tuple[Optional[float], bool]:
    """deep_score's final score with `remove` adjustments taken out, from the
    logged components. Returns (score, approx) — approx=True when the pre-
    clamp rules total was not logged and the rules score sat at a clamp
    bound. None when the record lacks what the reconstruction needs (a
    removed component that was never logged also counts as missing)."""
    bd = rec.get("score_breakdown")
    if not isinstance(bd, dict):
        return None, False
    removed = 0.0
    for k in remove:
        v = _num(bd.get(k))
        if v is None:
            return None, False
        removed += v
    ml = rec.get("ml") if isinstance(rec.get("ml"), dict) else {}
    blended = ml.get("blended") is True
    raw = _num(bd.get("rules_raw"))
    approx = False
    if raw is None:
        raw = _num(ml.get("rules_only_score")) if blended else _num(rec.get("score"))
        if raw is None:
            return None, False
        approx = raw <= 0.0 or raw >= 100.0
    rules = _clamp(raw - removed)
    if not blended:
        return _clamp(rules), approx
    rw, mw = _num(ml.get("rule_weight")), _num(ml.get("ml_weight"))
    mls = 50.0 if ml.get("gated") else _num(ml.get("ml_only_score"))
    if rw is None or mw is None or mls is None:
        return None, False
    return _clamp(rules * rw + mls * mw), approx


def ablation_variants(extra_logged: bool) -> Dict[str, Tuple[str, ...]]:
    v: Dict[str, Tuple[str, ...]] = {"baseline": (), "drop_unvalidated": UNVALIDATED_ADJUSTMENTS}
    for k in UNVALIDATED_ADJUSTMENTS:
        v[f"drop_{k}"] = (k,)
    if extra_logged:
        v["drop_unvalidated_and_boosts"] = UNVALIDATED_ADJUSTMENTS + ("recommendation", "contrarian", "rel_strength")
    return v


def _quintiles(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 5:
        return {}
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    q = np.searchsorted(edges, x, side="right")
    out = {}
    for i in range(5):
        m = q == i
        out[f"Q{i + 1}"] = {"n": int(m.sum()), "mean_pct": round(float(y[m].mean()), 3) if m.any() else None}
    if out["Q1"]["mean_pct"] is not None and out["Q5"]["mean_pct"] is not None:
        out["Q5_minus_Q1"] = round(out["Q5"]["mean_pct"] - out["Q1"]["mean_pct"], 3)
    return out


def ablation_report(rows: List[dict]) -> dict:
    """Score-vs-outcome under each ablation variant, on rows that carry the
    components (see the block comment above)."""
    usable, dropped_recon, n_approx = [], 0, 0
    for r in rows:
        base, approx = ablated_score(r["rec"], ())
        logged = _num(r["rec"].get("score"))
        if base is None or logged is None:
            continue
        if abs(base - logged) > RECON_TOL:
            dropped_recon += 1
            continue
        usable.append(r)
        n_approx += 1 if approx else 0
    extra_logged = any(isinstance(r["rec"].get("score_breakdown"), dict) and
                       _num(r["rec"]["score_breakdown"].get("recommendation")) is not None for r in usable)
    out = {"n_rows_with_components": len(usable), "dropped_reconstruction_mismatch": dropped_recon,
           "approx_clamped": n_approx, "removed_groups": list(UNVALIDATED_ADJUSTMENTS)}
    if len(usable) < MIN_N:
        out["status"] = "INSUFFICIENT"
        out["reason"] = (f"{len(usable)} deduped rows with logged components < MIN_N={MIN_N} "
                         "(score_breakdown is logged since 2026-10-26; +5d labels lag ~1 week)")
        return out
    days = [r["day"] for r in usable]
    boot = DayBootstrap(days, seed=SEED + 3)
    y = np.array([r["y"] for r in usable])
    variants = {}
    base_scores = np.array([ablated_score(r["rec"], ())[0] for r in usable])
    for name, remove in ablation_variants(extra_logged).items():
        sc = [ablated_score(r["rec"], remove)[0] for r in usable]
        keep = np.array([s is not None for s in sc])
        if int(keep.sum()) < MIN_N:
            variants[name] = {"n": int(keep.sum()), "status": "INSUFFICIENT"}
            continue
        x = np.array([s for s in sc if s is not None])
        yk, dk = y[keep], [d for d, k in zip(days, keep) if k]
        st = ic_stats(x, yk, dk, boot)
        st["quintiles"] = _quintiles(x, yk)
        if name != "baseline":
            ic_n, ic_b, se = _paired_gain(x, base_scores[keep], yk, dk, boot)
            st["ic_change_vs_baseline"] = round(ic_n - ic_b, 4)
            st["ic_change_se"] = round(se, 4) if se else None
        st["removed"] = list(remove)
        variants[name] = st
    out["variants"] = variants
    b = variants.get("baseline", {})
    d = variants.get("drop_unvalidated", {})
    if "ic" in b and "ic" in d:
        out["baseline_inverted"] = b["ic"] < 0
        out["un_inverts"] = bool(b["ic"] < 0 <= d["ic"])
        ch, se = d.get("ic_change_vs_baseline"), d.get("ic_change_se")
        out["improvement_significant"] = bool(ch is not None and se and ch / se > 1.96)
    out["status"] = "OK"
    return out


def selection_and_ablation(records: Sequence[dict], validation: Optional[dict] = None) -> dict:
    rows = build_population(records, PRIMARY_OUTCOME)
    verdicts = (validation or {}).get("sources", {})
    reg = [s for s in source_registry() if verdicts.get(s["name"], {}).get("verdict") != "CONSTANT"]
    feats = [s for s in reg if s["kind"] == "feature"]

    def complete(rws, cands):
        return [r for r in rws if all(c["get"](r["rec"]) is not None for c in cands)]

    feat_rows = complete(rows, feats)
    bd_rows = [r for r in rows if isinstance(r["rec"].get("score_breakdown"), dict)]
    # adjustments enter the breakdown arm only where (almost) every breakdown
    # row carries them — the 2026-10-27 extras are absent on older rows.
    bd_cands = [s for s in reg if s["kind"] == "feature" or (
        bd_rows and sum(1 for r in bd_rows if s["get"](r["rec"]) is not None) >= 0.95 * len(bd_rows))]
    bd_rows = complete(bd_rows, bd_cands)
    return {
        "selection": {
            "method": {"split": "by entry day: fit/select/holdout = " + "/".join(str(f) for f in SPLIT_FRACTIONS),
                       "model": f"ridge (lambda {RIDGE_LAMBDA}) on fit-standardized values",
                       "accept": f"gain >= {MIN_GAIN} and Bonferroni(paired day-bootstrap p) <= {SELECT_ALPHA}",
                       "max_steps": MAX_SELECT_STEPS},
            "features": forward_selection(feat_rows, feats),
            "breakdown": forward_selection(bd_rows, bd_cands),
        },
        "ablation": ablation_report(rows),
    }


def get_source_validation_report() -> dict:
    """Production entry point (diag probe source_validation): the shadow log
    on the volume, read-only. Per-source validation + forward selection +
    ablation, one load of the log."""
    from shadow_portfolio import _load_shadow_log
    records = _load_shadow_log()
    rep = validate_sources(records)
    rep.update(selection_and_ablation(records, rep))
    return rep
