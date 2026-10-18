"""Lazy Prices PR 3: ladder gate 2 (SIGNAL) harness — quintile sort of text-change
scores against forward returns, evaluated out-of-sample.

Pure functions, stdlib only, no network, no trading imports (SPINOUT-READY).
Inputs are (a) the scored rows sec_filings_similarity writes (`--rows-json`) and
(b) a price panel supplied by the caller — this module never fetches prices and
never decides which vendor supplies them. Nothing here is a signal claim: with
no real panel it only proves, on synthetic data, that the MACHINERY would find a
planted effect, would not find a null one, and cannot see the future.

PRE-REGISTERED before any return was computed (the constants below are the
registration; do not tune them on outcomes — a different choice is a NEW
variant and must raise `variants`):
  * primary test  : 10-K, Item 1A, cosine_tfidf (PR 2f point-in-time idf,
                    PR 2g floor), quintiles formed WITHIN each calendar-month
                    cohort of entry dates so market-wide drift cancels.
  * direction     : least-changed (highest similarity, Q5) minus most-changed
                    (Q1) is POSITIVE (Cohen-Malloy-Nguyen).
  * entry         : the first session OPEN at/after the filing became public —
                    same-day open only if accepted before 09:00 ET, else the
                    next trading day's open. Never the filing-day close.
  * horizon       : 63 trading days, entry open to the close of the 63rd
                    session. Delisted-before-exit => last available close,
                    counted and reported (a delisting loss is not hidden).
  * costs         : 10 bps round trip charged on each leg (spread net = gross
                    minus 2 x 10 bps).
  * out of sample : cohorts from 2015-01 onward; the in-sample period is shown
                    as context and never decides the verdict.
  * significance  : Newey-West (lag 2, overlapping 63d windows on monthly
                    cohorts), t >= the Bonferroni-style critical value for the
                    number of variants tried, over >= 24 OOS cohorts.
"""
import argparse
import bisect
import json
import math
import sys
from collections import Counter
from datetime import datetime, time, timezone
from statistics import NormalDist
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - zoneinfo ships with 3.9+
    _ET = None

PRIMARY = {"form": "10-K", "section": "1A", "metric": "cosine_tfidf"}
N_QUANTILES = 5
HORIZON_DAYS = 63
SAME_DAY_CUTOFF_ET = time(9, 0)   # accepted before this => the same day's open is tradable
COST_BPS_PER_LEG = 10.0
OOS_START = "2015-01"
MIN_COHORT = 25                   # names per monthly cohort for a quintile sort to mean anything
NW_LAG = 2
MIN_TEST_COHORTS = 12             # below this no t-stat is reported at all
MIN_OOS_COHORTS = 24              # below this the verdict is INSUFFICIENT, not FAIL
ALPHA = 0.05


class SimplePanel:
    """Price panel: `days` = sorted ISO trading dates (the calendar of record);
    `px[cik][day] = (open, close)`; `eligible(cik, day)` is the liquidity floor
    ($1M ADV in production) — default permissive, injected by the caller."""

    def __init__(self, days: list, px: dict, eligible=None):
        self.days = sorted(days)
        self.px = px
        self._eligible = eligible

    def open(self, cik, day) -> Optional[float]:
        v = (self.px.get(cik) or {}).get(day)
        return v[0] if v and v[0] and v[0] > 0 else None

    def close(self, cik, day) -> Optional[float]:
        v = (self.px.get(cik) or {}).get(day)
        return v[1] if v and v[1] and v[1] > 0 else None

    def last_close_on_or_before(self, cik, day) -> Optional[tuple]:
        """(day, close) of the latest priced session <= day, or None."""
        s = self.px.get(cik) or {}
        best = None
        for d in s:
            if d <= day and (best is None or d > best) and self.close(cik, d) is not None:
                best = d
        return (best, self.close(cik, best)) if best else None

    def eligible(self, cik, day) -> bool:
        return True if self._eligible is None else bool(self._eligible(cik, day))


def load_panel(path: str) -> SimplePanel:
    """panel.json: {"days": [...], "px": {cik: {day: [open, close]}}}."""
    with open(path) as f:
        d = json.load(f)
    return SimplePanel(d["days"], {c: {k: tuple(v) for k, v in s.items()} for c, s in d["px"].items()})


# ── lookahead-critical: when can a filing first be traded? ───────────────────

def entry_index(days: list, accepted_at: Optional[str]) -> Optional[int]:
    """Index into `days` of the first tradable OPEN for a filing accepted at the
    given UTC ISO timestamp, or None if undated / past the panel. The signal
    cannot be known before `accepted_at`; entering at any earlier price is
    lookahead by construction."""
    if not accepted_at:
        return None
    try:
        ts = datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    et = ts.astimezone(_ET) if _ET else ts
    d = et.date().isoformat()
    i = bisect.bisect_left(days, d)           # first session on/after the ET date
    on_day = i < len(days) and days[i] == d
    if on_day and et.time() < SAME_DAY_CUTOFF_ET:
        idx = i
    else:
        idx = bisect.bisect_right(days, d)    # first session strictly after the ET date
    return idx if idx < len(days) else None


def forward_return(panel, cik, idx: int, horizon: int = HORIZON_DAYS):
    """(return, tag) from days[idx] open to the close of days[idx+horizon-1].
    tag: 'ok' | 'delisted_exit'. (None, reason) when it cannot be measured."""
    days = panel.days
    end = idx + horizon - 1
    if end >= len(days):
        return None, "no_future"
    entry_day = days[idx]
    if not panel.eligible(cik, entry_day):
        return None, "ineligible"
    o = panel.open(cik, entry_day)
    if o is None:
        return None, "no_entry_price"
    c = panel.close(cik, days[end])
    tag = "ok"
    if c is None:
        last = panel.last_close_on_or_before(cik, days[end])
        if last is None or last[0] <= entry_day:
            return None, "no_exit_price"
        c, tag = last[1], "delisted_exit"
    return c / o - 1.0, tag


# ── row selection + cohort formation ─────────────────────────────────────────

def signal_value(row: dict, spec: dict = PRIMARY) -> Optional[float]:
    if row.get("form") != spec["form"]:
        return None
    s = (row.get("sections") or {}).get(spec["section"])
    if not s or s.get("excluded"):
        return None
    v = s.get(spec["metric"])
    return v if isinstance(v, (int, float)) else None


def build_observations(rows: list, panel, spec: dict = PRIMARY, horizon: int = HORIZON_DAYS) -> tuple:
    """-> (observations, drop_counts). One observation per usable row:
    {cik, accession, signal, entry_day, cohort 'YYYY-MM', ret, tag}."""
    obs, drops = [], Counter()
    for r in rows:
        sig = signal_value(r, spec)
        if sig is None:
            drops["no_signal"] += 1
            continue
        idx = entry_index(panel.days, r.get("accepted_at"))
        if idx is None:
            drops["no_entry_date"] += 1
            continue
        ret, tag = forward_return(panel, r["cik"], idx, horizon)
        if ret is None:
            drops[tag] += 1
            continue
        if r.get("accepted_at_source") == "submissions_json_unverified":
            drops["note_unverified_timestamp"] += 1   # kept (can only delay), but visible
        day = panel.days[idx]
        obs.append({"cik": r["cik"], "accession": r.get("accession"), "signal": sig,
                    "entry_day": day, "cohort": day[:7], "ret": ret, "tag": tag})
    return obs, dict(drops)


def quantile_sort(cohort_obs: list, n_q: int = N_QUANTILES) -> list:
    """Q index 0 (lowest similarity = most changed) .. n_q-1, deterministic
    tie-break by cik so reruns are bit-identical."""
    ordered = sorted(cohort_obs, key=lambda o: (o["signal"], str(o["cik"])))
    n = len(ordered)
    buckets = [[] for _ in range(n_q)]
    for rank, o in enumerate(ordered):
        buckets[min(n_q - 1, rank * n_q // n)].append(o)
    return buckets


def _mean(v: list) -> float:
    return sum(v) / len(v)


def cohort_results(obs: list, min_cohort: int = MIN_COHORT, cost_bps: float = COST_BPS_PER_LEG) -> list:
    """Per monthly cohort: quintile mean returns, excess over the cohort's
    equal-weight universe mean (the random-entry base rate), and the long-short
    spread (top minus bottom quintile) gross and net of costs."""
    by_cohort = {}
    for o in obs:
        by_cohort.setdefault(o["cohort"], []).append(o)
    out = []
    for cohort in sorted(by_cohort):
        co = by_cohort[cohort]
        if len(co) < min_cohort:
            continue
        b = quantile_sort(co)
        qm = [_mean([o["ret"] for o in q]) for q in b]
        base = _mean([o["ret"] for o in co])
        gross = qm[-1] - qm[0]
        out.append({"cohort": cohort, "n": len(co), "q_mean": qm, "base": base,
                    "q_excess": [m - base for m in qm],
                    "spread_gross": gross, "spread_net": gross - 2 * cost_bps / 1e4})
    return out


# ── statistics ───────────────────────────────────────────────────────────────

def nw_tstat(x: list, lag: int = NW_LAG) -> Optional[float]:
    """Newey-West (Bartlett) t-stat of the mean. lag=0 is the plain t with
    population variance. None for < 2 points or zero variance."""
    n = len(x)
    if n < 2:
        return None
    m = _mean(x)
    d = [v - m for v in x]
    var = sum(v * v for v in d) / n
    for L in range(1, min(lag, n - 1) + 1):
        g = sum(d[i] * d[i - L] for i in range(L, n)) / n
        var += 2 * (1 - L / (lag + 1)) * g
    if var <= 0:
        return None
    return m / math.sqrt(var / n)


def critical_t(variants: int = 1, alpha: float = ALPHA) -> float:
    """Two-sided normal critical value with alpha split across the variants
    tried (CLAUDE.md reasoning standard 4: discount for how many things we tried)."""
    return NormalDist().inv_cdf(1 - alpha / (2 * max(1, variants)))


def summarize(cohorts: list, oos_start: str = OOS_START) -> dict:
    def block(cs):
        net = [c["spread_net"] for c in cs]
        return {"cohorts": len(cs),
                "mean_gross": _mean([c["spread_gross"] for c in cs]) if cs else None,
                "mean_net": _mean(net) if net else None,
                "t_nw_net": nw_tstat(net) if len(cs) >= MIN_TEST_COHORTS else None,
                "q_excess_mean": [_mean([c["q_excess"][i] for c in cs]) for i in range(N_QUANTILES)] if cs else None}
    ins = [c for c in cohorts if c["cohort"] < oos_start]
    oos = [c for c in cohorts if c["cohort"] >= oos_start]
    return {"in_sample": block(ins), "out_of_sample": block(oos)}


def verdict(summary: dict, variants: int = 1) -> dict:
    """The pre-registered decision, from the OUT-OF-SAMPLE block only."""
    o = summary["out_of_sample"]
    crit = critical_t(variants)
    if o["cohorts"] < MIN_OOS_COHORTS or o["t_nw_net"] is None:
        return {"result": "INSUFFICIENT", "critical_t": crit,
                "why": f"{o['cohorts']} OOS cohorts < {MIN_OOS_COHORTS} or no t-stat"}
    ok = o["mean_net"] > 0 and o["t_nw_net"] >= crit
    return {"result": "PASS" if ok else "FAIL", "critical_t": crit,
            "why": f"OOS net spread {o['mean_net']:+.4%}, NW t {o['t_nw_net']:.2f} vs critical {crit:.2f}"}


def run(rows: list, panel, variants: int = 1, spec: dict = PRIMARY, horizon: int = HORIZON_DAYS) -> dict:
    obs, drops = build_observations(rows, panel, spec, horizon)
    cohorts = cohort_results(obs)
    summ = summarize(cohorts)
    return {"spec": spec, "variants": variants, "n_obs": len(obs), "drops": drops,
            "delisted_exits": sum(o["tag"] == "delisted_exit" for o in obs),
            "cohorts": cohorts, "summary": summ, "verdict": verdict(summ, variants)}


def report(res: dict) -> str:
    f = lambda v: "   -   " if v is None else f"{v:+.3%}"
    L = [f"gate 2 — {res['spec']}  variants_tried={res['variants']}",
         f"observations {res['n_obs']}  delisted_exits {res['delisted_exits']}  drops {res['drops']}"]
    for name in ("in_sample", "out_of_sample"):
        b = res["summary"][name]
        t = "  -  " if b["t_nw_net"] is None else f"{b['t_nw_net']:.2f}"
        ex = "-" if not b["q_excess_mean"] else " ".join(f(v) for v in b["q_excess_mean"])
        L.append(f"{name:<14} cohorts {b['cohorts']:>3}  spread gross {f(b['mean_gross'])} net {f(b['mean_net'])}  NW t {t}  Q1..Q5 excess vs base: {ex}")
    v = res["verdict"]
    L.append(f"VERDICT (OOS only): {v['result']} — {v['why']}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", required=True, help="rows JSON from `sec_filings_similarity distribution --pit --rows-json`")
    ap.add_argument("--panel", required=True, help="price panel JSON (see load_panel)")
    ap.add_argument("--variants", type=int, default=1, help="number of metric/section/horizon variants tried so far")
    args = ap.parse_args(argv)
    with open(args.rows) as f:
        rows = json.load(f)
    print(report(run(rows, load_panel(args.panel), variants=args.variants)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
