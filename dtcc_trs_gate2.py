#!/usr/bin/env python3
"""dtcc_trs_gate2.py: the registered gate-2 (SIGNAL) test of
`dtcc_sbsdr_equity_swaps`, run on the server beside the price panel
(open_questions.md (as2) step 2; written 2026-11-27).

THE REGISTRATION is research/open_questions.md, 2026-11-25, "DTCC equity
swaps: GATE 2 PRE-REGISTERED", with its two addenda of 2026-11-26 (item 3
made exact; the join rule (au2)). This file adds nothing to it. Where the
registration left a reading open, the reading taken is listed under READINGS
below; every one was fixed here before any return was computed, by a session
that cannot read the price panel at all (it exists only on the server).

WHAT IT READS
  <dtcc root>/panel            the event panel and event list (dtcc_trs_panel.py)
  <dtcc root>/figi             what each CUSIP is (cusip_figi.py)
  <dtcc root>/ftd              the dated CUSIP-to-symbol record (cusip_ftd.py)
  the price panel              price_panel.py, the last complete build
No network. Nothing in the trading bot reads its output.

THE TEST, in the registration's own numbering
  1 events       the event list as the nightly job wrote it; the run refuses
                 to start unless that job's check says the registered list
                 was reproduced.
  2 comparison   for an event day, every other underlier with a panel row
                 that day that is not inside its own event's 20-day window,
                 after the same filter.
  3 filter       security type common stock, ADR or REIT; at least 126 bars
                 before the event day; raw close at least 2 USD on it; median
                 dollar volume of the 20 bars before it at least 500,000 and
                 under 25,000,000 USD.
  4 entry        the close of the event's known day (the trading day after
                 the event day); horizon 20 trading days; a series that ends
                 before the horizon exits at its last close, counted.
  5 prices       the price panel, by symbol AND date: the one series that
                 has a bar on the event day.
  6 cost         1.0% round trip from each event, none from the comparison.
  7 statistic    mean over events of (event return net of cost minus the
                 equal-weight mean return of that day's comparison set),
                 standard error clustered by entry day.
  8 verdict      on the first run with at least 300 tested events: PASS if
                 the statistic is positive in each half (split at the median
                 entry day) and the pooled t is at least 3.0; KILLED if the
                 pooled statistic is at or below 0 or t is under 1.0;
                 otherwise INCONCLUSIVE, with one more look after 126 more
                 trading days of vendor files, then killed. Under 300:
                 INSUFFICIENT, and no statistic is computed or written.
  9 beside it    never the verdict: the 5-day horizon; the mean absolute
                 excess return; events with a capped row; events at or above
                 25,000,000 USD dollar volume; a split by the bot's regime
                 label at entry.

THE JOIN (au2, written 2026-11-26 before any count): for a CUSIP on a day,
(a) the SEC pair known on the day; else (b) today's US ticker from OpenFIGI
when the name has a US listing today; else (c) an SEC pair first printed
within 45 days after the day, marked; else no symbol, counted. If (c)
carries more than 5% of tested events the statistic is also computed without
them. If the share of events with no symbol differs by more than 3 points
between the first and the last half-year, the verdict carries that.

ONE LOOK. A run computes a statistic only when a look is due: no look has
been recorded, or the one recorded read INCONCLUSIVE and the vendor files
have advanced 126 trading days since. Every such run is written to
registered.json and is never overwritten. With a verdict of record and no
look due the run stops before it opens a price file.

ARMING, a guard on the one look (fixed here, before any run). The runner was
written by a session that could not run it on real prices, so a join fault
would spend the look on a wrong number. Before any return is summarised the
run counts, with no return involved: (i) of tested-window events that have a
symbol, the share that find exactly one price series with a bar on the event
day; (ii) the median number of comparison names per entry day. If (i) is
under ARM_MIN_PRICED_SHARE or (ii) under ARM_MIN_COMPARISON_MEDIAN the run
ends "not_armed": it writes the counts, no statistic, and no look is taken.

READINGS taken where the registration is silent (each blind to returns):
  R1 "any swap row that day" uses the event list's own day: a vendor file of
     a weekend or holiday is counted on the next trading day, as item 1
     says of events.
  R2 "bars before the day" and "the 20 trading days before the day" are the
     series' own bars strictly before the event day; the raw close is the
     event day's. A name with no bar ON the event day has no price series
     for that day and is dropped, counted.
  R3 dollar volume is adjusted close times served volume, price_panel.py's
     convention (raw dollar volume divided by any later dividend factor: a
     hair low for dividend payers).
  R4 an event enters at the close of its known day and needs a bar on it;
     the exit is the last bar on or before the 20th trading day after it.
  R5 a day's comparison set needs at least one name; an event on a day with
     none is not tested, counted.
  R6 the "large names" line of item 9 compares events at or above the upper
     dollar-volume line with comparison names at or above it.
  R7 the regime split reads the bot's own dated history file
     (storage_config.REGIME_HISTORY_PATH), which begins 2026-11; an entry day
     with no row within 5 days is "no label recorded".
  R8 a symbol is matched to a price series with punctuation removed
     (BRK.B and BRK/B are one name). Two series with a bar on the same day
     under one such symbol: ambiguous, dropped, counted.

OUTPUT: <DATA_DIR>/dtcc_trs_gate2/{latest.json, registered.json,
last_run.json}. Derived statistics and counts only: no CUSIP, ticker, price
or per-event row (the price panel is Alpaca data, internal research only).
The last stdout line is one JSON summary for server/dtccTrsGate2.ts.

Usage: python3 dtcc_trs_gate2.py --run [--nice N] [--max-rss-mb N]
           [--code-version V] [--dtcc-root DIR] [--panel-root DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import array
import bisect
import datetime as dt
import json
import math
import os
import resource
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import cusip_figi
import cusip_ftd
import dtcc_trs_panel as panel
import price_panel as pp

SCHEMA = "dtcc_trs_gate2/1"
REGISTRATION = "research/open_questions.md 2026-11-25 'DTCC equity swaps: GATE 2 PRE-REGISTERED' + 2026-11-26 addenda"
OUT_SUBDIR = "dtcc_trs_gate2"
DTCC_SUBDIR = "dtcc_trs_panel"

# Registered constants (items 3, 4, 6, 8). Changing one after a look voids the test.
HORIZON = 20
HORIZON_SECONDARY = 5
COST = 0.01
MIN_BARS_BEFORE = 126
MIN_RAW_CLOSE = 2.0
DV_WINDOW = 20
DV_MIN = 500_000.0
DV_MAX = 25_000_000.0
MIN_TESTED = 300
T_PASS = 3.0
T_KILL = 1.0
SECOND_LOOK_TRADING_DAYS = 126
MAX_LOOKS = 2
# (au2) and (at2), 2026-11-26.
JOIN_AFTER_SHARE_LIMIT = 0.05
NO_JOIN_SPREAD_POINTS = 3.0
# ARMING (module header). Not part of the statistic.
ARM_MIN_PRICED_SHARE = 0.80
ARM_MIN_COMPARISON_MEDIAN = 30
REGIME_LABEL_MAX_AGE_DAYS = 5
DEFAULT_MAX_RSS_MB = 900.0

# Why a (CUSIP, day) row is or is not usable, in the filter's own order.
NO_SYMBOL, NO_SERIES, AMBIGUOUS, SHORT_HISTORY, LOW_PRICE, NO_VOLUME, TOO_THIN, NO_ENTRY_BAR, BAD_PRICE, SMALL, LARGE = range(11)
CODE_NAMES = ("no symbol", "no price series with a bar on the day", "two series under one symbol",
              "under 126 bars before the day", "raw close under 2 USD", "dollar volume unreadable",
              "median dollar volume under 500,000 USD", "no bar on the entry day", "entry or exit price unusable",
              "passes, under 25,000,000 USD", "passes, at or above 25,000,000 USD")


class MemoryBudgetExceeded(RuntimeError):
    pass


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _check_rss(limit: float, where: str) -> None:
    if _rss_mb() > limit:
        raise MemoryBudgetExceeded(f"{_rss_mb():.0f} MB > {limit:.0f} MB at {where}")


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        print(f"[dtcc_trs_gate2] unreadable {path}: {e}", file=sys.stderr)
        return None


def _write_atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, separators=(",", ":"))
    os.replace(tmp, path)


# ── the one-look rule ───────────────────────────────────────────────────────
def trading_days_between(cal: Sequence[str], a: str, b: str) -> int:
    """Trading days after `a` up to and including `b` (0 when b <= a)."""
    return max(0, bisect.bisect_right(cal, b) - bisect.bisect_right(cal, a))


def look_due(registered: Optional[dict], last_testable: Optional[str], cal: Sequence[str]) -> Tuple[bool, int, str]:
    """(due, look number, why). Item 8: one look; a second only after an
    INCONCLUSIVE first and 126 more trading days of vendor files."""
    looks = list((registered or {}).get("looks") or [])
    if not looks:
        return True, 1, "no look recorded"
    if (registered or {}).get("final"):
        return False, len(looks), f"verdict of record: {registered['final']} (look {len(looks)} of {looks[-1].get('at')})"
    if len(looks) >= MAX_LOOKS:
        return False, len(looks), "both looks are spent"
    first = looks[0]
    since = trading_days_between(cal, str(first.get("events_last_testable") or ""), str(last_testable or ""))
    if since < SECOND_LOOK_TRADING_DAYS:
        return False, 1, (f"look 1 read {first.get('verdict')}; the second look is due after {SECOND_LOOK_TRADING_DAYS} "
                          f"more trading days of vendor files ({since} so far)")
    return True, 2, f"look 1 read {first.get('verdict')}; {since} trading days of vendor files since"


def final_after(look_no: int, verdict_word: str) -> Optional[str]:
    """The verdict of record once this look is written, or None while a
    second look is still owed."""
    if verdict_word in ("PASS", "KILLED"):
        return verdict_word
    if look_no >= MAX_LOOKS:
        return "KILLED"            # "one more look at the same bar ..., then killed"
    return None


def ladder_word(final: Optional[str]) -> Optional[str]:
    """PASS / FAIL as scripts/ladder_verdict_check.py reads it."""
    return {"PASS": "PASS", "KILLED": "FAIL"}.get(final or "")


# ── statistics ──────────────────────────────────────────────────────────────
def clustered_mean(x: np.ndarray, groups: np.ndarray) -> dict:
    """Mean of x with a standard error clustered by `groups` (the usual
    cluster-robust variance of a mean, with the G/(G-1) correction)."""
    x = np.asarray(x, dtype=np.float64)
    n = int(len(x))
    if n == 0:
        return {"n": 0, "clusters": 0, "mean_pct": None, "se_pct": None, "t": None}
    mu = float(x.mean())
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    g = int(inv.max()) + 1
    out = {"n": n, "clusters": g, "mean_pct": round(mu * 100.0, 4), "se_pct": None, "t": None}
    if g < 2:
        return out
    s = np.bincount(inv, weights=x - mu, minlength=g)
    var = (g / (g - 1.0)) * float(s @ s) / (n * n)
    if var <= 0:
        return out
    se = math.sqrt(var)
    out["se_pct"] = round(se * 100.0, 4)
    out["t"] = round(mu / se, 3)
    return out


def halves(x: np.ndarray, entry_day: np.ndarray) -> dict:
    """The two halves of item 8: events split at the median entry day."""
    if len(x) == 0:
        return {"median_entry_day": None, "first": None, "second": None}
    med = float(np.median(entry_day))
    a, b = x[entry_day <= med], x[entry_day > med]
    f = lambda v: {"n": int(len(v)), "mean_pct": round(float(v.mean()) * 100.0, 4) if len(v) else None}  # noqa: E731
    return {"median_entry_day": pp.day_str(int(med)), "first": f(a), "second": f(b)}


def verdict(pooled: dict, split: dict) -> dict:
    """Item 8, on a pooled clustered mean and its two halves."""
    n, mu, t = pooled.get("n") or 0, pooled.get("mean_pct"), pooled.get("t")
    if n < MIN_TESTED:
        return {"verdict": "INSUFFICIENT", "why": f"{n} tested events, under {MIN_TESTED}"}
    if mu is None or t is None:
        return {"verdict": "INSUFFICIENT", "why": "no standard error (one entry day, or no variance)"}
    if mu <= 0:
        return {"verdict": "KILLED", "why": f"pooled statistic {mu}% is at or below 0"}
    if t < T_KILL:
        return {"verdict": "KILLED", "why": f"pooled t {t} is under {T_KILL}"}
    h1, h2 = (split.get("first") or {}).get("mean_pct"), (split.get("second") or {}).get("mean_pct")
    if t >= T_PASS and h1 is not None and h2 is not None and h1 > 0 and h2 > 0:
        return {"verdict": "PASS", "why": f"pooled t {t} >= {T_PASS} and both halves positive ({h1}%, {h2}%)"}
    return {"verdict": "INCONCLUSIVE", "why": f"pooled statistic {mu}%, t {t}; halves {h1}% and {h2}%"}


def quartiles(v: Iterable[float]) -> Optional[dict]:
    a = np.asarray(list(v), dtype=np.float64)
    if len(a) == 0:
        return None
    q = np.quantile(a, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"min": float(q[0]), "q1": float(q[1]), "median": float(q[2]), "q3": float(q[3]), "max": float(q[4])}


# ── the join (au2) ──────────────────────────────────────────────────────────
def join_symbol(spans: Sequence[cusip_ftd.Span], entry: Optional[dict], day_iso: str) -> Tuple[Optional[str], str]:
    """(symbol, carrier): "a" the SEC pair known on the day, "b" today's US
    ticker, "c" an SEC pair first printed after the day, "none"."""
    j = cusip_ftd.symbol_on(spans, day_iso) if spans else None
    if j is not None and j["known_on_the_day"] and j["symbol"]:
        return str(j["symbol"]), "a"
    if entry and entry.get("found") and entry.get("us") and entry.get("ticker"):
        return str(entry["ticker"]), "b"
    if j is not None and j["symbol"]:
        return str(j["symbol"]), "c"
    return None, "none"


def one_symbol(spans: Sequence[cusip_ftd.Span], entry: Optional[dict]) -> Optional[str]:
    """The symbol every branch of join_symbol would give on any day, when
    that is certain without looking at the day: the name has a US listing
    today and the SEC never printed it under any other symbol. Used only to
    spare the comparison rows a date search; None means "ask join_symbol"."""
    if not (entry and entry.get("found") and entry.get("us") and entry.get("ticker")):
        return None
    t = cusip_ftd.norm_symbol(entry["ticker"])
    if not t or any(cusip_ftd.norm_symbol(sp[2]) != t for sp in spans):
        return None
    return str(entry["ticker"])


# ── inputs ──────────────────────────────────────────────────────────────────
def calendar(first: dt.date, last: dt.date) -> List[str]:
    return [d.isoformat() for d in panel.trading_days(first, last, panel.nyse_closures())]


def fold_index(cal: Sequence[str], day_iso: str) -> Optional[int]:
    """The trading day a vendor day's rows count on (R1): the first trading
    day on or after it."""
    i = bisect.bisect_left(cal, day_iso)
    return i if i < len(cal) else None


def kept_types(cache: Dict[str, dict]) -> Dict[str, str]:
    out = {}
    for c, e in cache.items():
        cls, _ = cusip_figi.tradable_class(e)
        if cls:
            out[c] = cls
    return out


def read_presence(panel_dir: str, cal: Sequence[str], kept: Dict[str, str], cusip_id: Dict[str, int],
                  days_wanted: set) -> Tuple[np.ndarray, np.ndarray, set]:
    """One pass over the panel: (day index, cusip id) of every kept-type row
    on a wanted day, each pair once, and the pairs that carried a capped
    new-trade notional."""
    di, ci = array.array("i"), array.array("i")
    capped = set()
    fold_cache: Dict[str, Optional[int]] = {}
    for r in panel.read_panel(panel_dir):
        c = str(r["cusip"])
        if c not in kept:
            continue
        d = str(r["date"])
        i = fold_cache.get(d, -2)
        if i == -2:
            i = fold_cache[d] = fold_index(cal, d)
        if i is None or i not in days_wanted:
            continue
        k = cusip_id.setdefault(c, len(cusip_id))
        di.append(i)
        ci.append(k)
        if int(r["capped_n"]) > 0:
            capped.add((i, k))
    if not len(di):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), capped
    pairs = np.unique(np.stack([np.frombuffer(di, dtype=np.int32), np.frombuffer(ci, dtype=np.int32)], axis=1).astype(np.int64), axis=0)
    return pairs[:, 0], pairs[:, 1], capped


def read_regime_labels(path: Optional[str]) -> List[Tuple[str, str]]:
    """[(as_of, label)] in day order from the bot's regime history (R7). A
    missing file is "no label recorded" for every day; a line that is not
    JSON is counted on stderr."""
    if not path or not os.path.exists(path):
        return []
    out: Dict[str, str] = {}
    bad = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    bad += 1
                    row = None
                if isinstance(row, dict) and row.get("as_of") and row.get("composite_regime"):
                    out[str(row["as_of"])[:10]] = str(row["composite_regime"])
    except OSError as e:
        print(f"[dtcc_trs_gate2] regime history unreadable, the regime split reads 'no label recorded': {e}", file=sys.stderr)
        return []
    if bad:
        print(f"[dtcc_trs_gate2] {bad} unreadable line(s) in {path} skipped", file=sys.stderr)
    return sorted(out.items())


def regime_on(labels: Sequence[Tuple[str, str]], day_iso: str) -> Optional[str]:
    i = bisect.bisect_right(labels, (day_iso, "￿")) - 1
    if i < 0:
        return None
    as_of, label = labels[i]
    if (dt.date.fromisoformat(day_iso) - dt.date.fromisoformat(as_of)).days > REGIME_LABEL_MAX_AGE_DAYS:
        return None
    return label


# ── the price pass ──────────────────────────────────────────────────────────
class Rows:
    """Every (CUSIP, event day) the test prices, events and comparison names
    alike, as parallel arrays."""

    def __init__(self, n: int):
        self.n = n
        self.hits = np.zeros(n, dtype=np.int16)
        self.bars_before = np.zeros(n, dtype=np.int32)
        self.raw_close = np.full(n, np.nan)
        self.dv_median = np.full(n, np.nan)
        self.entry_ok = np.zeros(n, dtype=bool)
        self.ret = {HORIZON: np.full(n, np.nan), HORIZON_SECONDARY: np.full(n, np.nan)}
        self.ended = np.zeros(n, dtype=bool)


def price_series(s: "pp.Series", ids: np.ndarray, t_day: np.ndarray, known_day: np.ndarray,
                 exit_day: Dict[int, np.ndarray], rows: Rows) -> None:
    """Fill `rows` for the rows `ids` from one series, where the series has
    a bar ON the event day. Nothing after the exit day is read."""
    n = len(s.days)
    pos = np.searchsorted(s.days, t_day)
    ok = pos < n
    ok[ok] = s.days[pos[ok]] == t_day[ok]
    if not ok.any():
        return
    ids, pos, known_day = ids[ok], pos[ok], known_day[ok]
    rows.hits[ids] += 1
    first = rows.hits[ids] == 1          # a second series under the same symbol never overwrites
    if not first.any():
        return
    ids, pos, known_day = ids[first], pos[first], known_day[first]
    sel = np.nonzero(ok)[0][first]
    rows.bars_before[ids] = pos
    rows.raw_close[ids] = pp.pit_price(s)[pos]
    dv = s.c * s.v.astype(np.float64)
    has = pos >= DV_WINDOW
    if has.any() and n >= DV_WINDOW:
        med = np.median(np.lib.stride_tricks.sliding_window_view(dv, DV_WINDOW), axis=1)
        rows.dv_median[ids[has]] = med[pos[has] - DV_WINDOW]     # the 20 bars strictly before the event day (R2)
    ie = np.searchsorted(s.days, known_day)
    e_ok = ie < n
    e_ok[e_ok] = s.days[ie[e_ok]] == known_day[e_ok]
    e_ok &= np.isfinite(s.c[np.minimum(ie, n - 1)]) & (s.c[np.minimum(ie, n - 1)] > 0)
    rows.entry_ok[ids] = e_ok
    if not e_ok.any():
        return
    for h, ex in exit_day.items():
        x = ex[sel][e_ok]
        ix = np.searchsorted(s.days, x, side="right") - 1       # last bar on or before the exit day (R4)
        r = s.c[ix] / s.c[ie[e_ok]] - 1.0
        r[~np.isfinite(r)] = np.nan
        rows.ret[h][ids[e_ok]] = r
        if h == HORIZON:
            rows.ended[ids[e_ok]] = s.days[-1] < x


def classify(rows: Rows, has_symbol: np.ndarray) -> np.ndarray:
    """One code per row, the first filter line it fails (CODE_NAMES)."""
    code = np.full(rows.n, SMALL, dtype=np.int8)
    code[rows.dv_median >= DV_MAX] = LARGE
    # later assignments win, so the lines are applied last-first
    code[~np.isfinite(rows.ret[HORIZON])] = BAD_PRICE
    code[~rows.entry_ok] = NO_ENTRY_BAR
    code[rows.dv_median < DV_MIN] = TOO_THIN
    code[~np.isfinite(rows.dv_median)] = NO_VOLUME
    code[~(rows.raw_close >= MIN_RAW_CLOSE)] = LOW_PRICE
    code[rows.bars_before < MIN_BARS_BEFORE] = SHORT_HISTORY
    code[rows.hits > 1] = AMBIGUOUS
    code[rows.hits == 0] = NO_SERIES
    code[~has_symbol] = NO_SYMBOL
    return code


def comparison_means(day: np.ndarray, r: np.ndarray, n_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per day index: (equal-weight mean return, names) over the rows given."""
    fin = np.isfinite(r)
    cnt = np.bincount(day[fin], minlength=n_days).astype(np.float64)
    tot = np.bincount(day[fin], weights=r[fin], minlength=n_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, tot / cnt, np.nan)
    return mean, cnt


# ── run ─────────────────────────────────────────────────────────────────────
def run(dtcc_root: str, panel_root: Optional[str], out_dir: str, max_rss_mb: float = DEFAULT_MAX_RSS_MB,
        code_version: Optional[str] = None, regime_history: Optional[str] = None) -> dict:
    t0 = time.time()
    panel_dir = os.path.join(dtcc_root, "panel")
    check = _read_json(os.path.join(panel_dir, "events_check.json"))
    if not check or not os.path.exists(os.path.join(panel_dir, panel.EVENTS_FILE)):
        return {"status": "insufficient", "reason": "the nightly job has not written an event list yet"}
    if (check.get("registered") or {}).get("match") is not True:
        return {"status": "insufficient", "reason": "the nightly job's check says the registered event list was NOT reproduced; the test does not start on another list"}
    build = pp.current_build(panel_root)
    if not build or not build["manifest"].get("as_of"):
        return {"status": "skipped", "reason": "no complete price-panel build"}
    as_of = str(build["manifest"]["as_of"])[:10]

    doc = panel.read_events(panel_dir)
    events = sorted(doc["events"], key=lambda e: (str(e["date"]), str(e["cusip"])))
    if not events:
        return {"status": "insufficient", "reason": "the event list is empty"}
    last_testable = str(check.get("last_testable") or doc["summary"].get("last_testable") or events[-1]["date"])
    cal = calendar(dt.date.fromisoformat(str(events[0]["date"])) - dt.timedelta(days=10),
                   max(dt.date.fromisoformat(as_of), dt.date.fromisoformat(last_testable)) + dt.timedelta(days=60))
    registered = _read_json(os.path.join(out_dir, "registered.json"))
    due, look_no, why = look_due(registered, last_testable, cal)
    if not due:
        return {"status": "skipped", "reason": why, "final": (registered or {}).get("final")}

    cache = cusip_figi.read_cache(os.path.join(dtcc_root, "figi", cusip_figi.CACHE_FILE))
    if not cache:
        return {"status": "insufficient", "reason": "no security-type record yet (cusip_figi)"}
    kept = kept_types(cache)
    cal_idx = {d: i for i, d in enumerate(cal)}
    cal_day = np.array([pp.day_num(d) for d in cal], dtype=np.int64)
    as_of_day = pp.day_num(as_of)
    n_cal = len(cal)

    # Events: type first, then whether the 20-day window is inside the price panel.
    other_types: Dict[str, int] = {}
    own_events: Dict[str, List[int]] = {}
    ev_day: List[int] = []
    ev_cusip: List[str] = []
    in_window_all = not_complete = 0
    for e in events:
        c, i = str(e["cusip"]), cal_idx.get(str(e["date"]))
        if i is None:
            raise ValueError(f"event day {e['date']} is not a trading day of the calendar")
        own_events.setdefault(c, []).append(i)
        x = i + 1 + HORIZON
        complete = x < n_cal and int(cal_day[x]) <= as_of_day
        in_window_all += complete
        if c not in kept:
            _, bucket = cusip_figi.tradable_class(cache.get(c))
            other_types[bucket] = other_types.get(bucket, 0) + 1
            continue
        if not complete:
            not_complete += 1
            continue
        ev_day.append(i)
        ev_cusip.append(c)
    counts: Dict[str, object] = {
        "events_in_list": len(events), "events_of_other_types": dict(sorted(other_types.items(), key=lambda kv: (-kv[1], kv[0]))),
        "events_kept_type": len(ev_day) + not_complete, "events_window_after_price_panel": not_complete,
        "events_in_window_all_types": int(in_window_all), "events_in_window": len(ev_day),
    }
    if not ev_day:
        return _finish_no_stat(out_dir, "insufficient", "no kept-type event has its 20-day window inside the price panel",
                               counts, build, as_of, code_version, t0)

    # Comparison rows: kept-type panel rows on those days, outside their own event's window.
    cusip_id: Dict[str, int] = {}
    for c in ev_cusip:
        cusip_id.setdefault(c, len(cusip_id))
    days_wanted = set(ev_day)
    p_day, p_cus, capped = read_presence(panel_dir, cal, kept, cusip_id, days_wanted)
    _check_rss(max_rss_mb, "panel pass")
    id_cusip = [""] * len(cusip_id)
    for c, k in cusip_id.items():
        id_cusip[k] = c
    own = {cusip_id[c]: sorted(v) for c, v in own_events.items() if c in cusip_id}
    keep = np.ones(len(p_day), dtype=bool)
    after_own_event = 0            # rows on a later day of the window (the event day's own row is the event)
    for j in range(len(p_day)):
        ev = own.get(int(p_cus[j]))
        if ev:
            k = bisect.bisect_right(ev, int(p_day[j])) - 1
            if k >= 0 and int(p_day[j]) - ev[k] <= panel.EVENT_COOLDOWN:
                keep[j] = False
                after_own_event += int(p_day[j]) != ev[k]
    c_day, c_cus = p_day[keep], p_cus[keep]
    n_ev, n_cmp = len(ev_day), len(c_day)
    row_day = np.concatenate([np.array(ev_day, dtype=np.int64), c_day])
    row_cus = np.concatenate([np.array([cusip_id[c] for c in ev_cusip], dtype=np.int64), c_cus])
    is_event = np.zeros(n_ev + n_cmp, dtype=bool)
    is_event[:n_ev] = True
    counts["comparison_rows"] = int(n_cmp)
    counts["comparison_rows_dropped_inside_own_event_window"] = int(after_own_event)

    # The join (au2). Events one by one; comparison rows through the certain-symbol shortcut.
    spans = cusip_ftd.read_spans(os.path.join(dtcc_root, "ftd"), set(id_cusip))
    certain = {k: one_symbol(spans.get(c, ()), cache.get(c)) for k, c in enumerate(id_cusip)}
    need: Dict[str, List[int]] = {}
    has_symbol = np.zeros(n_ev + n_cmp, dtype=bool)
    carrier = [""] * n_ev
    carriers_cmp = {"certain": 0, "a": 0, "b": 0, "c": 0, "none": 0}
    for r in range(n_ev + n_cmp):
        k = int(row_cus[r])
        c = id_cusip[k]
        if r >= n_ev and certain[k] is not None:
            sym, how = certain[k], "certain"
        else:
            sym, how = join_symbol(spans.get(c, ()), cache.get(c), cal[int(row_day[r])])
        if r < n_ev:
            carrier[r] = how
        else:
            carriers_cmp[how] += 1
        key = cusip_ftd.norm_symbol(sym)
        if key:
            has_symbol[r] = True
            need.setdefault(key, []).append(r)
    _check_rss(max_rss_mb, "join")
    half = [cusip_ftd.half_year(cal[i]) for i in ev_day]
    by_half: Dict[str, Dict[str, int]] = {}
    for r in range(n_ev):
        h = by_half.setdefault(half[r], {"events": 0, "a": 0, "b": 0, "c": 0, "none": 0})
        h["events"] += 1
        h[carrier[r]] += 1
    for h in by_half.values():
        h["none_pct"] = round(100.0 * h["none"] / h["events"], 2)  # type: ignore[assignment]
    hk = sorted(by_half)
    spread = round(float(by_half[hk[0]]["none_pct"]) - float(by_half[hk[-1]]["none_pct"]), 2) if len(hk) >= 2 else None
    join_report = {"rule": "(au2) a: SEC pair known on the day; b: today's US ticker; c: SEC pair first printed within 45 days after; none: dropped",
                   "events_by_half_year": {k: by_half[k] for k in hk},
                   "no_symbol_first_minus_last_points": spread,
                   "no_symbol_spread_over_limit": bool(spread is not None and abs(spread) > NO_JOIN_SPREAD_POINTS),
                   "comparison_rows": carriers_cmp}

    # The price pass: every series whose symbol is wanted.
    rows = Rows(n_ev + n_cmp)
    t_day = cal_day[row_day]
    known_day = cal_day[row_day + 1]
    exit_day = {HORIZON: cal_day[row_day + 1 + HORIZON], HORIZON_SECONDARY: cal_day[row_day + 1 + HORIZON_SECONDARY]}
    need_np = {k: np.array(v, dtype=np.int64) for k, v in need.items()}
    n_series = n_used = 0
    for s in pp.iter_series(build["dir"]):
        n_series += 1
        if n_series % 2000 == 0:
            _check_rss(max_rss_mb, f"series {n_series}")
        ids = need_np.get(cusip_ftd.norm_symbol(s.symbol))
        if ids is None or len(s.days) == 0:
            continue
        n_used += 1
        price_series(s, ids, t_day[ids], known_day[ids], {h: v[ids] for h, v in exit_day.items()}, rows)
    code = classify(rows, has_symbol)

    ev = np.arange(n_ev)
    funnel = {CODE_NAMES[k]: int((code[ev] == k).sum()) for k in range(len(CODE_NAMES))}
    funnel_half: Dict[str, Dict[str, int]] = {}
    for r in range(n_ev):
        d = funnel_half.setdefault(half[r], {})
        d[CODE_NAMES[int(code[r])]] = d.get(CODE_NAMES[int(code[r])], 0) + 1
    cmp_rows = np.arange(n_ev, n_ev + n_cmp)
    funnel_cmp = {CODE_NAMES[k]: int((code[cmp_rows] == k).sum()) for k in range(len(CODE_NAMES))}
    passed = int(funnel[CODE_NAMES[SMALL]])
    counts["events_passing_filter"] = passed
    # item 3: "If under 15% of events pass, that share is reported first as the finding."
    counts["events_passing_share_of_all_in_window_pct"] = round(100.0 * passed / max(1, in_window_all), 2)
    counts["events_passing_share_of_kept_in_window_pct"] = round(100.0 * passed / n_ev, 2)

    # ARMING: two counts, no return involved.
    with_symbol = int(has_symbol[:n_ev].sum())
    priced = int(((code[:n_ev] != NO_SYMBOL) & (code[:n_ev] != NO_SERIES) & (code[:n_ev] != AMBIGUOUS)).sum())
    priced_share = priced / with_symbol if with_symbol else 0.0
    small_cmp = cmp_rows[code[cmp_rows] == SMALL]
    entry_days = np.unique(row_day[ev[code[ev] == SMALL]])
    per_day = np.bincount(row_day[small_cmp], minlength=n_cal)[entry_days] if len(entry_days) else np.zeros(0)
    cmp_median = float(np.median(per_day)) if len(per_day) else 0.0
    arming = {"events_with_symbol": with_symbol, "events_with_one_price_series": priced,
              "priced_share": round(priced_share, 4), "min_priced_share": ARM_MIN_PRICED_SHARE,
              "comparison_names_per_entry_day": quartiles(per_day), "min_comparison_median": ARM_MIN_COMPARISON_MEDIAN,
              "armed": bool(priced_share >= ARM_MIN_PRICED_SHARE and cmp_median >= ARM_MIN_COMPARISON_MEDIAN)}
    base = {
        "counts": counts, "join": join_report, "funnel_events": funnel, "funnel_events_by_half_year": {k: funnel_half[k] for k in sorted(funnel_half)},
        "funnel_comparison_rows": funnel_cmp, "arming": arming,
        "price_panel": {"build_id": build["build_id"], "as_of": as_of, "series": n_series, "series_read": n_used,
                        "corrupt_lines_skipped": len(pp.corrupt_lines()), "survivorship": "REDUCED (2016+; OTC excluded)"},
        "events_first_testable": check.get("first_testable"), "events_last_testable": last_testable,
        "entry_days": {"first": cal[int(entry_days[0]) + 1] if len(entry_days) else None,
                       "last": cal[int(entry_days[-1]) + 1] if len(entry_days) else None, "n": int(len(entry_days))},
    }
    if not arming["armed"]:
        return _finish_no_stat(out_dir, "not_armed",
                               f"priced share {arming['priced_share']} (needs {ARM_MIN_PRICED_SHARE}), median comparison names {cmp_median:.0f} "
                               f"(needs {ARM_MIN_COMPARISON_MEDIAN}): no statistic computed, no look taken",
                               base, build, as_of, code_version, t0)

    # Item 7. The comparison mean of each day, then each event's excess.
    def test(bucket: int, h: int, cost: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cb = cmp_rows[code[cmp_rows] == bucket]
        cm, cn = comparison_means(row_day[cb], rows.ret[h][cb], n_cal)
        e = ev[code[ev] == bucket]
        e = e[np.isfinite(rows.ret[h][e]) & (cn[row_day[e]] > 0)]
        return e, rows.ret[h][e] - cost - cm[row_day[e]], cm

    e20, x20, cm20 = test(SMALL, HORIZON, COST)
    tested = int(len(e20))
    base["counts"]["events_tested"] = tested
    base["counts"]["events_on_a_day_with_no_comparison_name"] = int(passed - tested)
    if tested < MIN_TESTED:
        return _finish_no_stat(out_dir, "insufficient", f"{tested} tested events, under {MIN_TESTED}: INSUFFICIENT, no statistic computed, no look taken",
                               base, build, as_of, code_version, t0, verdict_word="INSUFFICIENT")

    entry = known_day[e20]
    pooled = clustered_mean(x20, entry)
    split = halves(x20, entry)
    v = verdict(pooled, split)
    if v["verdict"] == "INSUFFICIENT":
        return _finish_no_stat(out_dir, "insufficient", v["why"] + ": no statistic written, no look taken",
                               base, build, as_of, code_version, t0, verdict_word="INSUFFICIENT")

    def block(e: np.ndarray, x: np.ndarray) -> dict:
        b = clustered_mean(x, known_day[e])
        b["halves"] = halves(x, known_day[e])
        return b

    beside: Dict[str, object] = {}
    _, g20, _ = test(SMALL, HORIZON, 0.0)
    beside["horizon_20d_before_cost"] = clustered_mean(g20, entry)
    e5, x5, _ = test(SMALL, HORIZON_SECONDARY, COST)
    beside["horizon_5d_net_of_cost"] = block(e5, x5)
    months = np.array([int(cal[int(i) + 1][:4]) * 12 + int(cal[int(i) + 1][5:7]) for i in row_day[e20]])
    beside["pooled_clustered_by_entry_month"] = clustered_mean(x20, months)
    small_c = cmp_rows[code[cmp_rows] == SMALL]
    small_c = small_c[np.isfinite(rows.ret[HORIZON][small_c])]
    abs_c_mean, _ = comparison_means(row_day[small_c], np.abs(rows.ret[HORIZON][small_c] - cm20[row_day[small_c]]), n_cal)
    abs_e = np.abs(rows.ret[HORIZON][e20] - cm20[row_day[e20]])
    ok_abs = np.isfinite(abs_c_mean[row_day[e20]])
    beside["absolute_excess_20d"] = {
        "events_mean_pct": round(float(abs_e.mean()) * 100.0, 4),
        "comparison_mean_pct": round(float(np.nanmean(abs_c_mean[row_day[e20]])) * 100.0, 4),
        "events_minus_comparison": clustered_mean((abs_e - abs_c_mean[row_day[e20]])[ok_abs], entry[ok_abs]),
    }
    is_capped = np.array([(int(row_day[r]), int(row_cus[r])) in capped for r in e20], dtype=bool)
    beside["events_with_a_capped_row"] = clustered_mean(x20[is_capped], entry[is_capped])
    eL, xL, _ = test(LARGE, HORIZON, COST)
    beside["events_at_or_above_25m_dollar_volume"] = block(eL, xL)
    labels = read_regime_labels(regime_history)
    reg = np.array([regime_on(labels, cal[int(i) + 1]) or "no label recorded" for i in row_day[e20]])
    beside["by_regime_at_entry"] = {str(k): clustered_mean(x20[reg == k], entry[reg == k]) for k in sorted(set(reg.tolist()))}
    beside["series_ended_before_the_horizon"] = int(rows.ended[e20].sum())
    carr = np.array([carrier[int(r)] for r in e20])
    c_share = float((carr == "c").mean())
    beside["tested_by_join_carrier"] = {k: int((carr == k).sum()) for k in ("a", "b", "c")}
    without_c = None
    if c_share > JOIN_AFTER_SHARE_LIMIT:
        m = carr != "c"
        without_c = {"pooled": clustered_mean(x20[m], entry[m]), "halves": halves(x20[m], entry[m])}
        without_c.update(verdict(without_c["pooled"], without_c["halves"]))
    beside["without_joins_printed_after_the_event"] = without_c

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    look = {"look": look_no, "at": now, "code_version": code_version, "price_panel_build": build["build_id"], "as_of": as_of,
            "events_last_testable": last_testable, "tested": tested, "pooled": pooled, "halves": split,
            "verdict": v["verdict"], "why": v["why"],
            "carries": {"no_symbol_spread_over_limit": join_report["no_symbol_spread_over_limit"],
                        "no_symbol_first_minus_last_points": spread,
                        "joins_printed_after_share_pct": round(100.0 * c_share, 2),
                        "verdict_without_joins_printed_after": (without_c or {}).get("verdict")}}
    looks = list((registered or {}).get("looks") or []) + [look]
    final = final_after(look_no, v["verdict"])
    reg_doc = {"schema": SCHEMA, "registration": REGISTRATION, "looks": looks, "final": final, "ladder_verdict": ladder_word(final)}
    _write_atomic(os.path.join(out_dir, "registered.json"), reg_doc)
    payload = {"schema": SCHEMA, "status": "ok", "generated_at": now, "code_version": code_version, "registration": REGISTRATION,
               "look": look_no, "verdict": v["verdict"], "why": v["why"], "final": final,
               "statistic": {"what": "mean over events of (20-day return net of 1.0% cost minus the equal-weight mean 20-day return of that day's comparison set), SE clustered by entry day",
                             "pooled": pooled, "halves": split, "thresholds": {"t_pass": T_PASS, "t_kill": T_KILL, "min_tested": MIN_TESTED}},
               "beside_never_the_verdict": beside, **base,
               "elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": round(_rss_mb(), 1)}
    _write_atomic(os.path.join(out_dir, "latest.json"), payload)
    summary = {"status": "ok", "look": look_no, "verdict": v["verdict"], "final": final, "tested": tested,
               "mean_pct": pooled["mean_pct"], "t": pooled["t"], "build_id": build["build_id"],
               "elapsed_s": payload["elapsed_s"], "peak_rss_mb": payload["peak_rss_mb"]}
    _write_atomic(os.path.join(out_dir, "last_run.json"), {**summary, "finished_at": now})
    return summary


def _finish_no_stat(out_dir: str, status: str, reason: str, body: dict, build: dict, as_of: str,
                    code_version: Optional[str], t0: float, verdict_word: Optional[str] = None) -> dict:
    """A run that ends before any statistic: the counts are the finding.
    registered.json is not touched, so no look is spent."""
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    payload = {"schema": SCHEMA, "status": status, "reason": reason, "generated_at": now, "code_version": code_version,
               "registration": REGISTRATION, "look": None, "verdict": verdict_word, "statistic": None,
               **(body if "counts" in body else {"counts": body}),
               "elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": round(_rss_mb(), 1)}
    payload.setdefault("price_panel", {"build_id": build["build_id"], "as_of": as_of})
    _write_atomic(os.path.join(out_dir, "latest.json"), payload)
    summary = {"status": status, "reason": reason, "build_id": build["build_id"],
               "tested": (payload["counts"] or {}).get("events_tested") if isinstance(payload.get("counts"), dict) else None,
               "elapsed_s": payload["elapsed_s"], "peak_rss_mb": payload["peak_rss_mb"]}
    _write_atomic(os.path.join(out_dir, "last_run.json"), {**summary, "finished_at": now})
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    import storage_config
    ap = argparse.ArgumentParser(description="The registered gate-2 test of the DTCC equity-swap root.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dtcc-root", default=os.path.join(storage_config.DATA_DIR, DTCC_SUBDIR))
    ap.add_argument("--panel-root", default=None)
    ap.add_argument("--out-dir", default=os.path.join(storage_config.DATA_DIR, OUT_SUBDIR))
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
            print(f"[dtcc_trs_gate2] could not lower priority: {e}", file=sys.stderr)
    try:
        res = run(a.dtcc_root, a.panel_root, a.out_dir, a.max_rss_mb, a.code_version,
                  regime_history=getattr(storage_config, "REGIME_HISTORY_PATH", None))
    except MemoryBudgetExceeded as e:
        res = {"status": "error", "reason": f"memory budget: {e}"}
    except Exception as e:   # noqa: BLE001 — one JSON line for the scheduler; registered.json is written only by a finished look
        res = {"status": "error", "reason": f"{type(e).__name__}: {e}"[:300]}
    print(json.dumps(res, separators=(",", ":")))
    return 0 if res.get("status") in ("ok", "skipped", "insufficient", "not_armed") else 1


if __name__ == "__main__":
    sys.exit(main())
