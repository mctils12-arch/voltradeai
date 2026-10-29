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

PREREG_VERSION = "source_validation_v1 (2026-10-27)"
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


def get_source_validation_report() -> dict:
    """Production entry point (diag probe source_validation): the shadow log
    on the volume, read-only."""
    from shadow_portfolio import _load_shadow_log
    return validate_sources(_load_shadow_log())
