#!/usr/bin/env python3
"""earnings_language_qoq_gate2.py — SEC 8-K earnings-language stream, GATE 2
(SIGNAL layer of the ROOT VALIDATION LADDER), PAIRED-FILING test: the QoQ
LANGUAGE-DELTA feature the original hypothesis leads with (research/
open_questions.md NEW DATA ROOTS #1: "Lazy-Prices-style QoQ changes").

WHY THIS SCRIPT EXISTS (compiled 2026-11-11, before the registered date):
- scripts/earnings_language_gate2.py tests tone LEVEL only; its header and
  the 2026-10-02 run both deferred the delta feature to "a DIFFERENT script
  (paired-filing join)" once a second quarterly filing per company exists.
  datacore/signal_ladder.json registers that trigger as not_before
  2026-11-15.
- The public /api/data/earnings-language/history route caps at the newest
  500 filings; on 2026-11-11 those 500 cover only 11-05..11-10 (Q3 season
  runs ~80-190 filings/day), so the test CANNOT be fed from the API any
  more — it has to run where the archive lives (the Railway volume,
  <DATA_DIR>/datacore_archive/earnings8k/YYYY-MM-DD.jsonl[.gz], continuous
  since 2026-07-04; the diag `archive` probe confirmed 192 rows on 07-30,
  171 on 08-04, 68 on 10-20, 134 on 11-09). server/earningsQoq.ts runs this
  file weekly (Sunday 18:00-19:00 ET) at zero token cost; the first run lands
  2026-11-15, the registered date. EDGE DOCTRINE #3: the join + test are
  code, not a session's reasoning.

PRIOR — STATED BEFORE ANY RETURN WAS COMPUTED (REASONING STANDARD #10):
- Pairing coverage: ~60-75% of Oct-Nov Item 2.02 filings find a same-CIK
  prior filing 50-130 days earlier (the Q2 season Jul 15-Aug 15 is in the
  archive; off-cycle reporters pair across Sep->Dec).
- F1 dtone (tone change per 1000 words, current minus prior): a small
  positive association with +1/+3/+5d alpha vs SPY, r ~ +0.03..0.06 — WEAKER
  than the level signal (press releases are templated quarter to quarter,
  so most deltas are lexical noise). A mechanical PASS is plausible at
  N ~ 400-800 because the level run's +2% spreads came from a thick-tailed
  alpha; the month-replication robustness gate is where it most likely dies
  again (~65% FRAGILE).
- F2 similarity (cosine similarity of term-frequency vectors, current vs
  prior; Lazy Prices direction: LESS change -> HIGHER forward return, so the
  feature is oriented positive): expected weakest, r ~ +0.02; ~75% FAIL or
  FRAGILE. 8-K press releases are not 10-Ks — the Lazy Prices mechanism
  (quiet edits to risk language) has far less surface here.
- Honest overall expectation: INSUFFICIENT at the first look is unlikely
  (N will clear 30 easily); the likely verdict is FAIL or PASS-but-FRAGILE,
  i.e. the root stays gate2_pending or is killed at layer 2.

VERDICT RULE — PRE-STATED, PER FEATURE PER HORIZON (never adjusted after
seeing results; identical to the level script's rule plus its robustness
gate, both reused by import so the bar cannot drift):
  - MIN_SAMPLE = 30 pairs with a computable forward return at that horizon.
    N < MIN_SAMPLE => INSUFFICIENT (correlation still reported, never used
    to claim PASS/FAIL).
  - N >= MIN_SAMPLE: split pairs into feature > median vs <= median; PASS
    requires (top_mean_alpha - bottom_mean_alpha) >= PASS_SPREAD (50 bps)
    AND the same sign as Pearson r(feature, alpha); else FAIL. A PASS whose
    mean and median spreads disagree in sign is flagged NOT OUTLIER-ROBUST.
  - ROBUSTNESS (earnings_language_gate2_robustness.horizon_report, unchanged):
    REPLICATES only if the pooled one-sided permutation p < 0.05 AND every
    calendar month with n >= 30 has a same-signed Spearman >= 0.05; else
    FRAGILE.
  - ROOT VERDICT (what would move signal_ladder.json): the PRIMARY feature
    F1 dtone must be PASS, outlier-robust AND REPLICATES at >= 2 of the 3
    horizons. F2 is SECONDARY: reported, never sufficient on its own, and it
    counts on the ledger — variants = 2, so any observed edge is read with a
    two-test discount (REASONING STANDARD #4).
  - "alpha" = the ticker's lookahead-free forward return minus SPY's over
    the identical trading-day window (base rate: REASONING STANDARD #3). A
    size-matched random-entry base rate is a declared LIMITATION, not done.
  - FIRST SUFFICIENT RUN IS THE REGISTERED VERDICT. The job re-runs weekly on
    a growing archive; latest.json carries `first_sufficient_run` forward
    unchanged from the first run whose root verdict was not INSUFFICIENT.
    Later runs are informational replication, never a second chance.

PAIRING RULE (fixed before the first run): per CIK, filings with a ticker and
>= MIN_WORDS words of exhibit text, sorted by filing date. Greedy ANCHORS: the
first filing is an anchor; each later filing is an anchor only if it is
>= MIN_GAP_DAYS (50) after the previous anchor, otherwise it is a FOLLOW-ON
(same-quarter re-release, conference deck, amendment) and dropped from both
sides. A PAIR is two consecutive anchors <= MAX_GAP_DAYS (130) apart; wider
gaps are dropped and counted (a missed quarter is not "the prior quarter").
The current filing's ticker is the one that filing carried; CIK is the join
key so a ticker change never breaks the pair (counted). Truncated exhibit
texts (the archive caps text length) are used as-is — both sides share the
cap — and counted.

LOOKAHEAD GUARD (REASONING STANDARD #7): reused verbatim from the level
script (reaction_indices): the reference price is the close STRICTLY BEFORE
the filing could have moved the stock — the filing day's close only if the
acceptance timestamp is after 16:00 ET, otherwise the prior trading day's
close with the filing day as the first reaction day.

OUTPUT (DATA_DIR/earnings_qoq, derived statistics only — no ticker, CIK,
company or price leaves this directory, Alpaca data licence):
  latest.json          aggregates per feature per horizon + pairing counts +
                       root verdict + first_sufficient_run
  samples_latest.json  anonymised rows {filedAt, feature values, alpha_h}
                       for later diagnostics (no identifiers)
  last_run.json        status of the last attempt (error never replaces
                       the last good latest.json)
stdout: ONE JSON summary line for the scheduler (status ok|skipped|error).

Usage:
  python3 scripts/earnings_language_qoq_gate2.py --run [--nice 19]
      [--max-rss-mb 900] [--code-version V]
      [--archive-dir DIR]   # default <DATA_DIR>/datacore_archive/earnings8k
      [--out-dir DIR]       # default <DATA_DIR>/earnings_qoq
      [--input-json FILE]   # offline fixture: {"filings": [...]} (API shape)
      [--bars-json FILE]    # offline fixture: {SYMBOL: {"date": [...], "close": [...]}}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from statistics import median

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import earnings_language_gate2 as lvl  # noqa: E402  (tone_score, reaction_indices, forward_returns, pearson, loaders)
import earnings_language_gate2_robustness as rob  # noqa: E402  (horizon_report — the unchanged robustness gate)

SCHEMA = "earnings_qoq_gate2/1"
OUT_SUBDIR = "earnings_qoq"
MIN_SAMPLE = lvl.MIN_SAMPLE
PASS_SPREAD = lvl.PASS_SPREAD
HORIZONS = lvl.HORIZONS
MIN_GAP_DAYS = 50
MAX_GAP_DAYS = 130
MIN_WORDS = 200
FEATURES = ("dtone", "similarity")  # F1 primary first; F2 secondary
PRIMARY = "dtone"
VARIANTS = len(FEATURES)
PROMOTION_MIN_HORIZONS = 2
ARCHIVE_LOOKBACK_DAYS = 400  # read the whole archive (gate 1 shipped 2026-07-04)
MIN_BARS_DAYS = 90

VERDICT_RULE = (
    f"per feature per horizon: n>={MIN_SAMPLE} else INSUFFICIENT; PASS needs top-vs-bottom-half alpha "
    f"spread >= {PASS_SPREAD} same-signed as Pearson r, else FAIL; outlier_robust = mean and median spreads agree "
    f"in sign; robustness (perm p<0.05 and every month n>={rob.MIN_MONTH_N} Spearman>={rob.MIN_MONTH_RHO}) => "
    f"REPLICATES else FRAGILE. ROOT PASS = primary '{PRIMARY}' PASS + outlier_robust + REPLICATES at "
    f">={PROMOTION_MIN_HORIZONS} of {len(HORIZONS)} horizons; variants={VARIANTS}; first sufficient run is the "
    "registered verdict."
)


# ── Text features ─────────────────────────────────────────────────────
def tokens(text: str) -> list[str]:
    return [t.lower() for t in lvl.TOKEN_RE.findall(text or "")]


def cosine_similarity(a: list[str], b: list[str]) -> float | None:
    """Cosine similarity of raw term-frequency vectors (Lazy Prices' simplest
    measure). None when either side is empty."""
    if not a or not b:
        return None
    ca, cb = Counter(a), Counter(b)
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def pair_features(cur_text: str, prior_text: str) -> dict | None:
    tc, tp = lvl.tone_score(cur_text), lvl.tone_score(prior_text)
    if tc["tone_per_1000w"] is None or tp["tone_per_1000w"] is None:
        return None
    sim = cosine_similarity(tokens(cur_text), tokens(prior_text))
    if sim is None:
        return None
    return {
        "dtone": round(tc["tone_per_1000w"] - tp["tone_per_1000w"], 4),
        "similarity": round(sim, 5),
        "cur_words": tc["word_count"], "prior_words": tp["word_count"],
    }


# ── Pairing ───────────────────────────────────────────────────────────
def filing_date(rec: dict) -> str:
    return (rec.get("filedAt") or (rec.get("acceptanceDatetime") or "")[:10] or "")[:10]


def _days_between(a: str, b: str) -> int:
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def build_pairs(filings: list[dict]) -> tuple[list[dict], dict]:
    """Greedy same-CIK anchors >= MIN_GAP_DAYS apart; consecutive anchors
    <= MAX_GAP_DAYS apart form a pair. Every drop is counted."""
    counts = {
        "filings_in": len(filings), "no_cik": 0, "no_ticker": 0, "no_date": 0, "short_text": 0,
        "usable": 0, "ciks": 0, "anchors": 0, "follow_on_dropped": 0, "unpaired_first": 0,
        "gap_too_long": 0, "pairs": 0, "ticker_changed": 0, "truncated_either": 0, "feature_none": 0,
    }
    by_cik: dict[str, list[dict]] = defaultdict(list)
    seen_acc: set[str] = set()
    for f in filings:
        acc = f.get("accession")
        if acc and acc in seen_acc:
            continue
        if acc:
            seen_acc.add(acc)
        cik = str(f.get("cik") or "").strip().lstrip("0")
        if not cik:
            counts["no_cik"] += 1
            continue
        if not f.get("ticker"):
            counts["no_ticker"] += 1
            continue
        if not filing_date(f):
            counts["no_date"] += 1
            continue
        if len(tokens(f.get("text") or "")) < MIN_WORDS:
            counts["short_text"] += 1
            continue
        counts["usable"] += 1
        by_cik[cik].append(f)
    counts["ciks"] = len(by_cik)
    pairs: list[dict] = []
    for cik, recs in by_cik.items():
        recs.sort(key=lambda r: (filing_date(r), r.get("acceptanceDatetime") or ""))
        anchors: list[dict] = []
        for r in recs:
            if anchors and _days_between(filing_date(anchors[-1]), filing_date(r)) < MIN_GAP_DAYS:
                counts["follow_on_dropped"] += 1
                continue
            anchors.append(r)
        counts["anchors"] += len(anchors)
        counts["unpaired_first"] += 1  # the first anchor has no prior by construction
        for prev, cur in zip(anchors, anchors[1:]):
            gap = _days_between(filing_date(prev), filing_date(cur))
            if gap > MAX_GAP_DAYS:
                counts["gap_too_long"] += 1
                continue
            feats = pair_features(cur.get("text") or "", prev.get("text") or "")
            if feats is None:
                counts["feature_none"] += 1
                continue
            if cur.get("ticker") != prev.get("ticker"):
                counts["ticker_changed"] += 1
            if cur.get("truncated") or prev.get("truncated"):
                counts["truncated_either"] += 1
            pairs.append({
                "cik": cik, "ticker": cur["ticker"], "accession": cur.get("accession"),
                "filedAt": filing_date(cur), "acceptanceDatetime": cur.get("acceptanceDatetime"),
                "prior_filedAt": filing_date(prev), "gap_days": gap, **feats,
            })
    counts["pairs"] = len(pairs)
    return pairs, counts


# ── Verdicts ──────────────────────────────────────────────────────────
def mechanical_verdict(samples: list[dict]) -> dict:
    """The level script's pre-stated rule, lifted into a function. `samples`
    rows carry `tone_per_1000w` (the FEATURE value — the key name is kept so
    rob.horizon_report can read the same rows unchanged) and `alpha`."""
    n = len(samples)
    xs = [s["tone_per_1000w"] for s in samples]
    ys = [s["alpha"] for s in samples]
    r = lvl.pearson(xs, ys) if n >= 3 else None
    entry: dict = {"n": n, "pearson_r": round(r, 4) if r is not None else None}
    if n < MIN_SAMPLE:
        entry["verdict"] = "INSUFFICIENT"
        entry["note"] = f"n={n} < MIN_SAMPLE={MIN_SAMPLE}; no PASS/FAIL claim"
        return entry
    m = median(xs)
    top = [s["alpha"] for s in samples if s["tone_per_1000w"] > m]
    bot = [s["alpha"] for s in samples if s["tone_per_1000w"] <= m]
    if not top or not bot:
        entry["verdict"] = "FAIL"
        entry["note"] = "feature has no spread around its median"
        return entry
    top_mean, bot_mean = sum(top) / len(top), sum(bot) / len(bot)
    spread = top_mean - bot_mean
    median_spread = median(top) - median(bot)
    same_sign = r is not None and (spread > 0) == (r > 0)
    median_agrees = (median_spread > 0) == (spread > 0)
    entry.update({
        "median_feature": round(m, 4), "top_group_n": len(top), "bottom_group_n": len(bot),
        "top_mean_alpha": round(top_mean, 5), "bottom_mean_alpha": round(bot_mean, 5),
        "spread": round(spread, 5), "median_spread": round(median_spread, 5),
        "outlier_robust": bool(median_agrees),
        "verdict": "PASS" if (spread >= PASS_SPREAD and same_sign) else "FAIL",
    })
    return entry


def root_verdict(features: dict) -> dict:
    """PASS only on the primary feature; INSUFFICIENT when no horizon of the
    primary reaches MIN_SAMPLE; FAIL otherwise."""
    prim = features.get(PRIMARY, {})
    horizons_passing = [
        h for h, e in prim.items()
        if e.get("verdict") == "PASS" and e.get("outlier_robust") and (e.get("robustness") or {}).get("verdict") == "REPLICATES"
    ]
    sufficient = [h for h, e in prim.items() if e.get("verdict") != "INSUFFICIENT"]
    if not sufficient:
        result = "INSUFFICIENT"
    elif len(horizons_passing) >= PROMOTION_MIN_HORIZONS:
        result = "PASS"
    else:
        result = "FAIL"
    return {
        "result": result, "primary": PRIMARY, "horizons_passing": horizons_passing,
        "horizons_sufficient": sufficient, "rule": VERDICT_RULE,
    }


# ── Returns ───────────────────────────────────────────────────────────
def default_bars_fn():
    """Alpaca-first/Yahoo-fallback bars via backtest_v2.fetch_bars (the same
    data path the level script and live trading use). The disk cache lives in
    the repo dir, never the volume; if it is unwritable we fetch uncached."""
    cache_ok = True
    try:
        os.makedirs(lvl.fetch_bars.__globals__["CACHE_DIR"], exist_ok=True)
        cache_ok = os.access(lvl.fetch_bars.__globals__["CACHE_DIR"], os.W_OK)
    except Exception:  # noqa: BLE001
        cache_ok = False

    def bars(symbol: str, days: int) -> dict:
        return lvl.fetch_bars(symbol, days, use_cache=cache_ok)
    return bars


def compute(filings: list[dict], bars_fn, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    pairs, counts = build_pairs(filings)
    if not pairs:
        return {"pairing": counts, "features": {}, "root": root_verdict({}), "samples": {}, "fetch_errors": [],
                "unique_tickers": 0}
    earliest = min(p["filedAt"] for p in pairs)
    days = max(MIN_BARS_DAYS, _days_between(earliest, today.isoformat()) + 10)
    spy = bars_fn("SPY", days)
    spy_map = dict(zip(spy.get("date", []), spy.get("close", [])))
    ticker_bars: dict[str, dict | None] = {}
    fetch_errors: list[dict] = []
    per_h: dict[int, list[dict]] = {h: [] for h in HORIZONS}
    for p in pairs:
        t = p["ticker"]
        if t not in ticker_bars:
            try:
                ticker_bars[t] = bars_fn(t, days)
            except Exception as e:  # noqa: BLE001 — one bad ticker never kills the run
                ticker_bars[t] = None
                fetch_errors.append({"err": str(e)[:120]})
        b = ticker_bars[t]
        if not b or not b.get("date"):
            continue
        ri = lvl.reaction_indices(b["date"], p["filedAt"], p.get("acceptanceDatetime"))
        if ri is None:
            continue
        pre_idx, anchor_idx = ri
        fwd = lvl.forward_returns(b["date"], b["close"], pre_idx, anchor_idx, HORIZONS)
        pre_date = b["date"][pre_idx]
        for h, (as_of, ret) in fwd.items():
            if pre_date not in spy_map or as_of not in spy_map or not spy_map[pre_date]:
                continue
            alpha = ret - (spy_map[as_of] / spy_map[pre_date] - 1)
            per_h[h].append({"filedAt": p["filedAt"], "dtone": p["dtone"], "similarity": p["similarity"],
                             "gap_days": p["gap_days"], "alpha": round(alpha, 5)})
    features: dict = {}
    for feat in FEATURES:
        features[feat] = {}
        for h in HORIZONS:
            rows = [{"filedAt": s["filedAt"], "tone_per_1000w": s[feat], "alpha": s["alpha"]} for s in per_h[h]]
            entry = mechanical_verdict(rows)
            if entry["verdict"] != "INSUFFICIENT":
                entry["robustness"] = rob.horizon_report(rows)
            features[feat][str(h)] = entry
    return {
        "pairing": counts, "features": features, "root": root_verdict(features),
        "samples": {str(h): rows for h, rows in per_h.items()},
        "fetch_errors": fetch_errors[:20], "fetch_error_count": len(fetch_errors),
        "unique_tickers": len(ticker_bars),
    }


# ── Runner ────────────────────────────────────────────────────────────
def _write_atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, path)


def _read_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            v = json.load(f)
        return v if isinstance(v, dict) else None
    except (OSError, ValueError):
        return None


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_filings(archive_dir: str | None, input_json: str | None) -> tuple[list[dict], str, dict]:
    if input_json:
        with open(input_json) as f:
            payload = json.load(f)
        rows = payload.get("filings", payload if isinstance(payload, list) else [])
        return rows, f"input-json:{input_json}", {"first_day": None, "last_day": None}
    d = archive_dir or lvl.default_archive_dir()
    rows = lvl.load_from_archive_dir(d, ARCHIVE_LOOKBACK_DAYS)
    days = sorted({filing_date(r) for r in rows if filing_date(r)})
    return rows, f"archive-dir:{d}", {"first_day": days[0] if days else None, "last_day": days[-1] if days else None}


def summarize(res: dict, source: str, archive_span: dict, code_version: str, prev_latest: dict | None,
              elapsed_s: float, now_iso: str) -> dict:
    first = (prev_latest or {}).get("first_sufficient_run")
    if not first and res["root"]["result"] != "INSUFFICIENT":
        first = {"generated_at": now_iso, "root_verdict": res["root"]["result"], "pairs": res["pairing"]["pairs"],
                 "code_version": code_version, "archive_last_day": archive_span.get("last_day")}
    return {
        "schema": SCHEMA, "generated_at": now_iso, "code_version": code_version,
        "source": source.split(":")[0], "archive": archive_span,
        "pairing": res["pairing"], "unique_tickers": res["unique_tickers"],
        "fetch_error_count": res.get("fetch_error_count", 0),
        "features": res["features"], "root_verdict": res["root"], "variants": VARIANTS,
        "first_sufficient_run": first,
        "verdict_rule": VERDICT_RULE,
        "caveat": ("tone LEVEL lexicon reused for the delta; alpha vs SPY (not size-matched); 8-K press releases are "
                   "not 10-Ks; truncated exhibit texts used as-is on both sides; weekly re-runs are informational — "
                   "first_sufficient_run is the registered verdict"),
        "elapsed_s": round(elapsed_s, 1), "peak_rss_mb": round(_rss_mb(), 1),
    }


def run(out_dir: str, archive_dir: str | None, input_json: str | None, bars_fn, code_version: str,
        max_rss_mb: float, today: dt.date | None = None) -> dict:
    t0 = time.time()
    filings, source, span = load_filings(archive_dir, input_json)
    if not filings:
        return {"status": "skipped", "reason": f"no earnings8k filings found ({source})"}
    if _rss_mb() > max_rss_mb:
        raise MemoryError(f"peak RSS {_rss_mb():.0f} MB > budget {max_rss_mb:.0f} MB after loading {len(filings)} filings")
    res = compute(filings, bars_fn, today)
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    payload = summarize(res, source, span, code_version, _read_json(os.path.join(out_dir, "latest.json")),
                        time.time() - t0, now_iso)
    _write_atomic(os.path.join(out_dir, "latest.json"), payload)
    _write_atomic(os.path.join(out_dir, "samples_latest.json"),
                  {"schema": SCHEMA, "generated_at": now_iso, "samples": res["samples"]})
    _write_atomic(os.path.join(out_dir, "last_run.json"),
                  {"status": "ok", "finished_at": now_iso, "verdict": res["root"]["result"]})
    n_obs = {h: e["n"] for h, e in res["features"].get(PRIMARY, {}).items()}
    return {"status": "ok", "verdict": res["root"]["result"], "pairs": res["pairing"]["pairs"],
            "n_obs": n_obs, "elapsed_s": payload["elapsed_s"], "peak_rss_mb": payload["peak_rss_mb"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--max-rss-mb", type=float, default=900.0)
    ap.add_argument("--code-version", default="unknown")
    ap.add_argument("--archive-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--input-json", default=None)
    ap.add_argument("--bars-json", default=None)
    a = ap.parse_args(argv)
    if a.nice:
        try:
            os.nice(a.nice)
        except OSError as e:
            print(f"[earnings_qoq] os.nice failed: {e}", file=sys.stderr)
    data_dir = os.environ.get("DATA_DIR") or ("/data/voltrade" if os.path.exists("/data") else "/tmp")
    out_dir = a.out_dir or os.path.join(data_dir, OUT_SUBDIR)
    if a.bars_json:
        with open(a.bars_json) as f:
            fixture = json.load(f)

        def bars_fn(symbol: str, days: int) -> dict:  # noqa: ARG001
            return fixture.get(symbol) or {"date": [], "close": []}
    else:
        bars_fn = default_bars_fn()
    try:
        out = run(out_dir, a.archive_dir, a.input_json, bars_fn, a.code_version, a.max_rss_mb)
    except Exception as e:  # noqa: BLE001 — one JSON line for the scheduler; last good latest.json untouched
        out = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        try:
            _write_atomic(os.path.join(out_dir, "last_run.json"),
                          {"status": "error", "reason": out["reason"],
                           "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        except OSError as e2:
            print(f"[earnings_qoq] could not write last_run.json: {e2}", file=sys.stderr)
    print(json.dumps(out))
    return 0 if out["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
