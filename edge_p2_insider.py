#!/usr/bin/env python3
"""
edge_p2_insider.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for pre-registration
P2 (research/edge_preregistration.md): OPPORTUNISTIC insider purchases
(Cohen-Malloy-Pomorski 2012 classification) over the 2016+ point-in-time,
delisted-inclusive price panel (price_panel.py). No trading is involved; nothing
here is read by deep_score or any trade path.

Run weekly by server/edgeP2.ts (Sunday window, own long-job lane, derived
statistics only) — zero tokens. Output: DATA_DIR/edge_p2/latest.json (+
label_spotcheck.json, the gate-1 25-insider label sample). The verdict logic below
is the pre-registration's kill rules, fixed BEFORE any outcome was seen.

DECLARED DEVIATIONS from the pre-registration (each makes this a NEW VARIANT that
must be named in the result; none was chosen after seeing an outcome):
  D1 window   The panel starts 2016-01-04 and the momentum match needs 252 prior
              bars, so the OOS window is 2017-02 -> latest (~9 yrs), not 2012-06 ->.
              Every month is post-publication (CMP 2012). Fewer months => larger
              MDE; the MDE rule (INCONCLUSIVE, not kill, when MDE > 35 bp) guards it.
  D2 size     Size decile = trailing 60-bar mean dollar volume (point-in-time),
              not XBRL shares x price (no PIT shares source). Liquidity-size proxy.
  D3 cohorts  "Calendar-time monthly portfolio" is implemented as entry-month
              cohorts of 21-trading-day event returns (equal-weight, one event per
              name per month). Same 21-day hold, slightly overlapping months.
  D4 skipped  Value-weighted secondary skipped (no PIT market cap).
Survivorship rule (pre-reg): a survivor-only run may KILL, never PASS. The panel is
survivorship-REDUCED (not free; pre-2016 delistings invisible), so a pass is only
ever PASS_PROVISIONAL.

Usage: python3 edge_p2_insider.py --run [--panel-root DIR] [--out-dir DIR]
           [--form4-dir DIR] [--max-rss-mb N] [--nice N] [--code-version V]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import price_panel as pp

SCHEMA = "edge_p2/v1"
HOLDS = (21, 63)                 # primary 21 trading days; 63 secondary (pre-reg)
MIN_VALUE = 10_000.0             # pre-reg: price x shares >= $10,000
MIN_PRICE = 2.0
MIN_ADV = 250_000.0              # pre-reg: 20-day ADV >= $250k
MOM_BARS = 252
SIZE_BARS = 60
MIN_BARS = MOM_BARS + 2
HISTORY_FROM = "2014q1"          # 3 calendar years before the first 2017 event
WINDOW_START_MONTH = "2017-02"   # D1
T_THRESHOLD = 2.8                # 9 variants, two-sided Bonferroni (pre-reg)
MDE_BAR_BP = 35.0                # 0.42 x 82 bp (pre-reg)
MIN_NAMES_PER_MONTH = 3
CELL_MIN = 5
NW_LAG = 3
DEFAULT_MAX_RSS_MB = 1500.0


class MemoryBudgetExceeded(RuntimeError):
    pass


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _check_rss(limit: float, where: str) -> None:
    if _rss_mb() > limit:
        raise MemoryBudgetExceeded(f"{_rss_mb():.0f} MB > {limit:.0f} MB at {where}")


# ── Form 4 history → CMP classification ─────────────────────────────────────
# A cache row is [ticker, issuer_cik, owner_cik, code, filing_date, trans_date, value].
def _cache_path(form4_dir: str, quarter: str) -> str:
    return os.path.join(form4_dir, f"{quarter}.jsonl.gz")


def quarters_between(first: str, last: str) -> List[str]:
    y, q = int(first[:4]), int(first[-1])
    ly, lq = int(last[:4]), int(last[-1])
    out = []
    while (y, q) <= (ly, lq):
        out.append(f"{y}q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def ensure_form4_cache(form4_dir: str, first: str = HISTORY_FROM, sleep_s: float = 0.25) -> Tuple[List[str], List[str]]:
    """Download (once, resumable) each closed quarter's officer/director P/S rows in
    compact form. Returns (cached quarters, quarters that failed). Network trouble
    on one quarter never aborts the rest."""
    import sec_form4_bulk as sb
    os.makedirs(form4_dir, exist_ok=True)
    available = sb.list_available_quarters()
    wanted = [q for q in sorted(available) if q >= first]
    failed: List[str] = []
    for q in wanted:
        path = _cache_path(form4_dir, q)
        if os.path.exists(path):
            continue
        try:
            recs = sb.fetch_quarter(available[q])
        except Exception as e:   # noqa: BLE001 — logged by the caller via the failed list
            print(f"[edge_p2] {q} fetch failed: {e}", file=sys.stderr)
            failed.append(q)
            continue
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps([r["ticker"], r["issuer_cik"], r["owner_cik"], r["trans_code"],
                                     r["filing_date"], r["trans_date"], r["dollar_value"]], separators=(",", ":")) + "\n")
        os.replace(tmp, path)
        time.sleep(sleep_s)
    cached = [q for q in wanted if os.path.exists(_cache_path(form4_dir, q))]
    return cached, failed


def iter_cached_rows(form4_dir: str, quarters: Iterable[str]) -> Iterable[list]:
    for q in quarters:
        p = _cache_path(form4_dir, q)
        if not os.path.exists(p):
            continue
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def _ym(date_s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not date_s or len(date_s) < 7:
        return None
    try:
        return int(date_s[:4]), int(date_s[5:7])
    except ValueError:
        return None


def classify(history_months: Iterable[Tuple[int, int]], year: int) -> Optional[str]:
    """CMP: classifiable iff the insider traded in EACH of years y-3..y-1; ROUTINE
    iff the same calendar month appears in all three; else OPPORTUNISTIC; None if
    unclassifiable (excluded, never pooled). `history_months` = (year, month) of
    every open-market P/S trade by this (owner, issuer)."""
    by_year: Dict[int, set] = {year - 3: set(), year - 2: set(), year - 1: set()}
    for y, m in history_months:
        if y in by_year:
            by_year[y].add(m)
    if any(not s for s in by_year.values()):
        return None
    common = by_year[year - 3] & by_year[year - 2] & by_year[year - 1]
    return "routine" if common else "opportunistic"


def build_events(rows: Iterable[list]) -> Tuple[List[dict], dict]:
    """Classify every qualifying code-P purchase. One pass collects each
    (owner, issuer)'s P/S months, a second assigns classes (a trade's class never
    sees its own year or later — `classify` only reads y-3..y-1).
    Event = (issuer, filing_date): opportunistic if any qualifying owner is, else
    routine if any is, else unclassified."""
    months: Dict[Tuple[str, str], set] = defaultdict(set)
    buys: Dict[Tuple[str, str, str, str], float] = defaultdict(float)   # (ticker,issuer,owner,filing)->value
    for tkr, issuer, owner, code, filed, traded, value in rows:
        ym = _ym(traded) or _ym(filed)
        if ym is None or not owner or not issuer:
            continue
        months[(owner, issuer)].add(ym)
        if code == "P" and filed:
            buys[(tkr, issuer, owner, filed)] += float(value)
    counts = {"buys_total": len(buys), "below_min_value": 0, "unclassifiable": 0, "routine": 0, "opportunistic": 0}
    by_ev: Dict[Tuple[str, str, str], set] = defaultdict(set)
    for (tkr, issuer, owner, filed), value in buys.items():
        if value < MIN_VALUE:
            counts["below_min_value"] += 1
            continue
        cls = classify(months[(owner, issuer)], int(filed[:4]))
        if cls is None:
            counts["unclassifiable"] += 1
            continue
        counts[cls] += 1
        by_ev[(tkr, issuer, filed)].add(cls)
    events = [{"ticker": t, "issuer": i, "filing_date": f,
               "cls": "opportunistic" if "opportunistic" in c else "routine"}
              for (t, i, f), c in sorted(by_ev.items(), key=lambda kv: (kv[0][2], kv[0][0]))]
    return events, counts


def label_spotcheck(form4_dir: str, quarters: Sequence[str], n: int = 25, seed: int = 20261031) -> List[dict]:
    """Gate-1 clause: a reproducible sample of n classified purchases with the
    accession-free fields a reader needs to check the label against the raw filing
    (SEC public data — not subject to the Alpaca licence)."""
    rows = list(iter_cached_rows(form4_dir, quarters))
    months: Dict[Tuple[str, str], set] = defaultdict(set)
    for tkr, issuer, owner, code, filed, traded, value in rows:
        ym = _ym(traded) or _ym(filed)
        if ym:
            months[(owner, issuer)].add(ym)
    cand = [r for r in rows if r[3] == "P" and float(r[6]) >= MIN_VALUE and r[4] and r[4][:4] >= "2017"]
    rng = np.random.default_rng(seed)
    out: List[dict] = []
    for i in rng.permutation(len(cand)):
        tkr, issuer, owner, _c, filed, traded, value = cand[int(i)]
        cls = classify(months[(owner, issuer)], int(filed[:4]))
        if cls is None:
            continue
        y = int(filed[:4])
        hist = sorted(m for m in months[(owner, issuer)] if y - 3 <= m[0] <= y - 1)
        out.append({"ticker": tkr, "issuer_cik": issuer, "owner_cik": owner, "filing_date": filed, "value_usd": round(float(value)),
                    "class": cls, "prior_3y_trade_months": [f"{a}-{b:02d}" for a, b in hist]})
        if len(out) >= n:
            break
    return out


# ── panel pass ──────────────────────────────────────────────────────────────
def next_day_after(cal: np.ndarray, day: int) -> Optional[int]:
    """First calendar day strictly after `day` (no same-day entry — pre-reg)."""
    i = int(np.searchsorted(cal, day, side="right"))
    return int(cal[i]) if i < len(cal) else None


def spy_calendar(build_dir: str) -> Optional[np.ndarray]:
    for s in pp.iter_series(build_dir):
        if s.symbol == "SPY" and len(s.days) > 1000:
            return s.days.copy()
    return None


def fwd_returns(s: "pp.Series", ie: np.ndarray, hold: int, as_of_day: int) -> np.ndarray:
    """Open-to-open return over `hold` bars from entry index ie. A series that
    ENDS before the exit (delisting/merger) exits at its last close (counted by
    the caller); one that is merely too young (still trading) is NaN = censored."""
    n = len(s.days)
    ex = ie + hold
    ended = bool(s.last_day < as_of_day - 7)
    px = np.where(ex < n, s.o[np.minimum(ex, n - 1)], s.c[n - 1])
    out = px / s.o[ie] - 1.0
    if not ended:
        out[ex >= n] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def collect(build_dir: str, need_days: np.ndarray, events_by_ticker: Dict[str, List[Tuple[int, int, str]]],
            as_of: str, max_rss_mb: float) -> Tuple[dict, List[dict], dict]:
    """One pass over every series. For each needed entry day D at which a series is
    eligible on its PRE-entry bar, record (size, momentum, fwd21, fwd63, survivor).
    `events_by_ticker[symbol]` = [(day D, event index, class)] -> the same features
    for each event. Returns (universe arrays by column, event rows, stats)."""
    as_of_day = pp.day_num(as_of)
    cols: Dict[str, List[np.ndarray]] = {k: [] for k in ("k", "dv", "mom", "f21", "f63", "surv")}
    ev_rows: List[dict] = []
    st = {"series": 0, "series_used": 0, "universe_rows": 0, "ended_early_exits": 0}
    for s in pp.iter_series(build_dir):
        st["series"] += 1
        if st["series"] % 2000 == 0:
            _check_rss(max_rss_mb, f"series {st['series']}")
        n = len(s.days)
        if n < MIN_BARS:
            continue
        pos = np.searchsorted(s.days, need_days)
        ok = pos < n
        ok[ok] = s.days[pos[ok]] == need_days[ok]
        k = np.nonzero(ok)[0]
        ie = pos[k]
        ie_ok = ie >= MOM_BARS + 1
        k, ie = k[ie_ok], ie[ie_ok]
        if len(k) == 0:
            continue
        elig = pp.eligible_mask(s, MIN_PRICE, MIN_ADV)
        keep = elig[ie - 1] & (s.o[ie] > 0) & np.isfinite(s.o[ie])
        k, ie = k[keep], ie[keep]
        if len(k) == 0:
            continue
        dv = pp.trailing_mean(s.c * s.v.astype(np.float64), SIZE_BARS)[ie - 1]
        mom = s.c[ie - 1] / s.c[ie - 1 - MOM_BARS] - 1.0
        good = np.isfinite(mom) & np.isfinite(dv) & (dv > 0)
        k, ie, dv, mom = k[good], ie[good], dv[good], mom[good]
        if len(k) == 0:
            continue
        f21 = fwd_returns(s, ie, HOLDS[0], as_of_day)
        f63 = fwd_returns(s, ie, HOLDS[1], as_of_day)
        st["ended_early_exits"] += int(np.sum((ie + HOLDS[0] >= n) & (s.last_day < as_of_day - 7)))
        surv = np.full(len(k), not s.delisted)
        cols["k"].append(k.astype(np.int32)); cols["dv"].append(dv.astype(np.float32))
        cols["mom"].append(mom.astype(np.float32)); cols["f21"].append(f21.astype(np.float32))
        cols["f63"].append(f63.astype(np.float32)); cols["surv"].append(surv)
        st["series_used"] += 1
        for (day, ei, cls) in events_by_ticker.get(s.symbol, ()):
            j = np.nonzero(need_days[k] == day)[0]
            if len(j):
                j = int(j[0])
                ev_rows.append({"ei": ei, "cls": cls, "k": int(k[j]), "dv": float(dv[j]), "mom": float(mom[j]),
                                "f21": float(f21[j]), "f63": float(f63[j]), "surv": bool(surv[j])})
    uni = {c: (np.concatenate(v) if v else np.array([])) for c, v in cols.items()}
    st["universe_rows"] = int(len(uni["k"]))
    return uni, ev_rows, st


# ── matched benchmark ───────────────────────────────────────────────────────
class DayGrid:
    """One entry day's universe cross-section: size decile (by dollar volume) x
    momentum tercile WITHIN decile; per-cell mean forward returns. Cells under
    CELL_MIN fall back to the decile, then the whole day."""

    def __init__(self, dv: np.ndarray, mom: np.ndarray, fwd: Dict[int, np.ndarray]):
        self.dec_edges = np.quantile(dv, np.linspace(0.1, 0.9, 9)) if len(dv) >= 10 else np.array([])
        dec = np.searchsorted(self.dec_edges, dv) if len(self.dec_edges) else np.zeros(len(dv), dtype=int)
        self.ter_edges: List[np.ndarray] = []
        self.cell: Dict[int, np.ndarray] = {h: np.full((10, 3), np.nan) for h in fwd}
        self.cell_n = np.zeros((10, 3), dtype=int)
        self.dec_mean: Dict[int, np.ndarray] = {h: np.full(10, np.nan) for h in fwd}
        self.day_mean = {h: float(np.nanmean(f)) if np.isfinite(f).any() else float("nan") for h, f in fwd.items()}
        for d in range(10):
            m = dec == d
            if m.sum() == 0:
                self.ter_edges.append(np.array([]))
                continue
            edges = np.quantile(mom[m], [1 / 3, 2 / 3]) if m.sum() >= 3 else np.array([])
            self.ter_edges.append(edges)
            ter = np.searchsorted(edges, mom[m]) if len(edges) else np.zeros(int(m.sum()), dtype=int)
            for h, f in fwd.items():
                fm = f[m]
                self.dec_mean[h][d] = float(np.nanmean(fm)) if np.isfinite(fm).any() else float("nan")
                for t in range(3):
                    sel = ter == t
                    v = fm[sel][np.isfinite(fm[sel])]
                    if len(v) >= CELL_MIN:
                        self.cell[h][d, t] = float(v.mean())
                    if h == next(iter(fwd)):
                        self.cell_n[d, t] = int(sel.sum())

    def expected(self, dv: float, mom: float, h: int) -> float:
        d = int(np.searchsorted(self.dec_edges, dv)) if len(self.dec_edges) else 0
        e = self.ter_edges[d]
        t = int(np.searchsorted(e, mom)) if len(e) else 0
        v = self.cell[h][d, t]
        if np.isfinite(v):
            return float(v)
        v = self.dec_mean[h][d]
        return float(v) if np.isfinite(v) else self.day_mean[h]


def abnormal_returns(uni: dict, ev_rows: List[dict], survivor_only: bool) -> List[dict]:
    """Per event: fwd - matched-cell mean (the base rate) and fwd - day mean (raw)."""
    if len(uni["k"]) == 0:
        return []
    mask = uni["surv"] if survivor_only else np.ones(len(uni["k"]), dtype=bool)
    k, dv, mom = uni["k"][mask], uni["dv"][mask], uni["mom"][mask]
    f = {21: uni["f21"][mask], 63: uni["f63"][mask]}
    order = np.argsort(k, kind="stable")
    k, dv, mom = k[order], dv[order], mom[order]
    f = {h: v[order] for h, v in f.items()}
    uniq, start = np.unique(k, return_index=True)
    bounds = list(start) + [len(k)]
    grids = {int(d): (bounds[i], bounds[i + 1]) for i, d in enumerate(uniq)}
    cache: Dict[int, DayGrid] = {}
    out = []
    for e in ev_rows:
        if survivor_only and not e["surv"]:
            continue
        rng = grids.get(e["k"])
        if rng is None or rng[1] - rng[0] < 50:
            continue
        g = cache.get(e["k"])
        if g is None:
            a, b = rng
            g = cache[e["k"]] = DayGrid(dv[a:b], mom[a:b], {h: v[a:b] for h, v in f.items()})
        row = dict(e)
        for h, key in ((21, "f21"), (63, "f63")):
            row[f"abn{h}"] = e[key] - g.expected(e["dv"], e["mom"], h) if np.isfinite(e[key]) else float("nan")
            row[f"raw{h}"] = e[key] - g.day_mean[h] if np.isfinite(e[key]) else float("nan")
        out.append(row)
    return out


# ── statistics ──────────────────────────────────────────────────────────────
def newey_west_t(x: Sequence[float], lag: int = NW_LAG) -> Tuple[float, float]:
    """(mean, t) of a series with a Bartlett-weighted HAC variance of the mean."""
    a = np.asarray(x, dtype=np.float64)
    n = len(a)
    if n < 3:
        return float(a.mean()) if n else float("nan"), float("nan")
    mu = a.mean()
    e = a - mu
    lrv = float(e @ e) / n
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1.0)
        lrv += 2.0 * w * float(e[L:] @ e[:-L]) / n
    if lrv <= 0:
        return float(mu), float("nan")
    return float(mu), float(mu / math.sqrt(lrv / n))


def monthly_series(rows: List[dict], day_to_month: Dict[int, str], cls: str, key: str, tickers: Dict[int, str]) -> Dict[str, Tuple[float, int]]:
    """Per entry month: equal-weight mean of `key` over one event per name (the
    first), months under MIN_NAMES_PER_MONTH dropped by the caller."""
    seen: set = set()
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["k"], r["ei"])):
        if r["cls"] != cls or not np.isfinite(r[key]):
            continue
        m = day_to_month[r["k"]]
        if m < WINDOW_START_MONTH:
            continue
        nm = (m, tickers[r["ei"]])
        if nm in seen:
            continue
        seen.add(nm)
        acc[m].append(r[key])
    return {m: (float(np.mean(v)), len(v)) for m, v in acc.items()}


def summarize_series(by_month: Dict[str, Tuple[float, int]]) -> dict:
    kept = {m: v for m, v in by_month.items() if v[1] >= MIN_NAMES_PER_MONTH}
    months = sorted(kept)
    x = [kept[m][0] for m in months]
    if len(x) < 6:
        return {"months": len(x), "months_dropped_thin": len(by_month) - len(kept), "mean_bp": None, "t_nw3": None,
                "sd_bp": None, "mde_bp": None, "mean_names_per_month": None}
    mu, t = newey_west_t(x)
    sd = float(np.std(x, ddof=1))
    mde = T_THRESHOLD * sd / math.sqrt(len(x))
    return {"months": len(x), "months_dropped_thin": len(by_month) - len(kept), "mean_bp": round(mu * 1e4, 1),
            "t_nw3": None if not math.isfinite(t) else round(t, 2), "sd_bp": round(sd * 1e4, 1), "mde_bp": round(mde * 1e4, 1),
            "mean_names_per_month": round(float(np.mean([kept[m][1] for m in months])), 1),
            "first_month": months[0], "last_month": months[-1]}


def verdict(pit_opp: dict, pit_routine: dict, surv_opp: dict) -> dict:
    """The pre-registration's kill rules, in order (a)(c)(b); PASS only if t >=
    2.8 on the PIT panel AND the survivor run is positive AND routine < opp."""
    def has(d: dict) -> bool:
        return d.get("mean_bp") is not None
    if not has(pit_opp):
        return {"verdict": "INCONCLUSIVE", "why": "too few usable months/events on the PIT panel"}
    if has(surv_opp) and surv_opp["mean_bp"] <= 0:
        return {"verdict": "KILLED", "rule": "a", "why": f"survivor-only opportunistic mean {surv_opp['mean_bp']} bp/month <= 0 (a survivor panel flatters purchases)"}
    if has(pit_routine) and pit_routine["mean_bp"] >= pit_opp["mean_bp"]:
        return {"verdict": "KILLED", "rule": "c", "why": f"routine ({pit_routine['mean_bp']} bp) >= opportunistic ({pit_opp['mean_bp']} bp): the classification is not doing the work"}
    t = pit_opp.get("t_nw3")
    if t is not None and t >= T_THRESHOLD and pit_opp["mean_bp"] > 0:
        return {"verdict": "PASS_PROVISIONAL", "why": f"PIT t {t} >= {T_THRESHOLD}; survivor run and routine falsification consistent. Panel is survivorship-REDUCED, 2017+ only — gate 3 may proceed, ladder gate 2 is not final until the shadow book reads."}
    if t is not None and pit_opp["mde_bp"] is not None and pit_opp["mde_bp"] <= MDE_BAR_BP:
        return {"verdict": "KILLED", "rule": "b", "why": f"PIT t {t} < {T_THRESHOLD} with MDE {pit_opp['mde_bp']} bp <= {MDE_BAR_BP} bp"}
    return {"verdict": "INCONCLUSIVE", "why": f"PIT t {t}, MDE {pit_opp.get('mde_bp')} bp > {MDE_BAR_BP} bp bar: underpowered, not a kill"}


# ── run ─────────────────────────────────────────────────────────────────────
def _write_atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)


def run(panel_root: Optional[str], out_dir: str, form4_dir: str, max_rss_mb: float, code_version: Optional[str]) -> dict:
    t0 = time.time()
    build = pp.current_build(panel_root)     # complete builds only (current.json is the pointer)
    if not build or not build["manifest"].get("as_of"):
        return {"status": "skipped", "reason": "no complete price-panel build"}
    build_dir = build["dir"]
    as_of = str(build["manifest"]["as_of"])
    cached, failed = ensure_form4_cache(form4_dir)
    need_hist = quarters_between(HISTORY_FROM, f"{as_of[:4]}q{(int(as_of[5:7]) - 1) // 3 + 1}")
    missing = [q for q in need_hist[:-2] if q not in cached]    # the last 2 quarters may not be published yet
    if missing:
        return {"status": "insufficient", "reason": f"Form 4 history incomplete, missing {missing[:6]} (failed fetches: {failed[:6]})"}
    events, ev_counts = build_events(iter_cached_rows(form4_dir, cached))
    events = [e for e in events if e["filing_date"] >= "2016-12-01" and e["filing_date"] <= as_of]
    cal = spy_calendar(build_dir)
    if cal is None:
        return {"status": "insufficient", "reason": "SPY series not found in the panel (no trading calendar)"}
    ebt: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    need = set()
    tickers: Dict[int, str] = {}
    for i, e in enumerate(events):
        d = next_day_after(cal, pp.day_num(e["filing_date"]))
        if d is None:
            continue
        need.add(d)
        tickers[i] = e["ticker"]
        ebt[e["ticker"]].append((d, i, e["cls"]))
    need_days = np.array(sorted(need), dtype=np.int64)
    if len(need_days) == 0:
        return {"status": "insufficient", "reason": "no events with an entry day"}
    uni, ev_rows, st = collect(build_dir, need_days, ebt, as_of, max_rss_mb)
    day_to_month = {i: pp.day_str(int(d))[:7] for i, d in enumerate(need_days)}
    out_runs: Dict[str, dict] = {}
    for label, surv_only in (("pit", False), ("survivor_only", True)):
        rows = abnormal_returns(uni, ev_rows, surv_only)
        res = {}
        for cls in ("opportunistic", "routine"):
            for h in HOLDS:
                res[f"{cls}_{h}d_matched"] = summarize_series(monthly_series(rows, day_to_month, cls, f"abn{h}", tickers))
            res[f"{cls}_21d_raw_unmatched"] = summarize_series(monthly_series(rows, day_to_month, cls, "raw21", tickers))
        res["events_matched"] = len(rows)
        out_runs[label] = res
    v = verdict(out_runs["pit"]["opportunistic_21d_matched"], out_runs["pit"]["routine_21d_matched"],
                out_runs["survivor_only"]["opportunistic_21d_matched"])
    spot = label_spotcheck(form4_dir, cached)
    _write_atomic(os.path.join(out_dir, "label_spotcheck.json"), {"schema": SCHEMA, "n": len(spot), "rows": spot})
    payload = {
        "schema": SCHEMA, "status": "ok", "generated_at": datetime.now(timezone.utc).isoformat(), "code_version": code_version,
        "as_of": as_of, "panel": {"build_id": build["build_id"], "survivorship": "REDUCED (2016+; pre-2016 delistings invisible)"},
        "preregistration": "research/edge_preregistration.md#P2",
        "window": {"entry_months_from": WINDOW_START_MONTH, "hold_days": list(HOLDS), "primary": 21},
        "deviations": ["D1 window 2017-02 onward (panel starts 2016)", "D2 size = trailing 60-bar dollar volume", "D3 entry-month cohorts", "D4 value-weighted skipped"],
        "threshold": {"t": T_THRESHOLD, "mde_bar_bp": MDE_BAR_BP, "variants": 9},
        "events": {**ev_counts, "classified_in_window": len(events), "with_entry_day": len(tickers), "matched_rows": len(ev_rows)},
        "universe": {"entry_days": int(len(need_days)), "rows": st["universe_rows"], "series": st["series"], "series_used": st["series_used"],
                     "ended_early_exits": st["ended_early_exits"]},
        "form4": {"quarters_cached": len(cached), "failed": failed},
        "results": out_runs, **v,
        "elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": round(_rss_mb(), 1),
    }
    _write_atomic(os.path.join(out_dir, "latest.json"), payload)
    _write_atomic(os.path.join(out_dir, "last_run.json"), {"status": "ok", "finished_at": payload["generated_at"], "verdict": v["verdict"]})
    return {"status": "ok", "verdict": v["verdict"], "build_id": build["build_id"], "events": len(ev_rows),
            "elapsed_s": payload["elapsed_s"], "peak_rss_mb": payload["peak_rss_mb"]}


def main(argv: Optional[List[str]] = None) -> int:
    import storage_config
    ap = argparse.ArgumentParser(description="Gate-2 test of opportunistic insider purchases (pre-registration P2).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--panel-root", default=None)
    ap.add_argument("--out-dir", default=os.path.join(storage_config.DATA_DIR, "edge_p2"))
    ap.add_argument("--form4-dir", default=os.path.join(storage_config.DATA_DIR, "edge_p2", "form4_cache"))
    ap.add_argument("--max-rss-mb", type=float, default=DEFAULT_MAX_RSS_MB)
    ap.add_argument("--code-version", default=None)
    ap.add_argument("--nice", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.run:
        ap.print_help()
        return 2
    if a.nice > 0:
        try:
            os.nice(a.nice)
        except OSError as e:
            print(f"[edge_p2] could not lower priority: {e}", file=sys.stderr)
    try:
        res = run(a.panel_root, a.out_dir, a.form4_dir, a.max_rss_mb, a.code_version)
    except MemoryBudgetExceeded as e:
        res = {"status": "error", "reason": f"memory budget: {e}"}
    except Exception as e:   # noqa: BLE001 — one JSON line for the scheduler, last good latest.json untouched
        res = {"status": "error", "reason": f"{type(e).__name__}: {e}"[:300]}
    print(json.dumps(res, separators=(",", ":")))
    return 0 if res.get("status") in ("ok", "skipped", "insufficient") else 1


if __name__ == "__main__":
    sys.exit(main())
