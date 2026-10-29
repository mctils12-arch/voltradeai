#!/usr/bin/env python3
"""price_panel.py — streaming Python reader for the 2016+ daily price panel.

[RULE-REVIEW] measurement-side code (CLAUDE.md MEASUREMENT INTEGRITY): this is
the reader every long-term (walk-forward) test stands on, so it decodes the
panel EXACTLY as the writer encodes it and adds nothing the writer did not
record. It never fetches, never writes the panel, and has no trading imports.

WRITER: server/pricePanel.ts (nightly job, human order 2026-10-28), format in
research/price_panel_coverage.md:

  <PRICE_PANEL_DIR>/current.json          {build_id, completed_at} of the last COMPLETE build
  <PRICE_PANEL_DIR>/<build_id>/manifest.json
  <PRICE_PANEL_DIR>/<build_id>/bars/bNNNNN.jsonl.gz
      one JSON object per (symbol, mapping mode, date segment):
      s, mode ("default"|"raw"), seg, seg_start, seg_end, ids[], st, ev, flags[],
      d0, n, k, dd[], c[], o[], h[], l[], v[], f[]

DECODING mirrors pricePanel.ts decodeSegment line for line (and is pinned to
it by server/pricePanelPyParity.test.ts, which encodes with the TS encoder and
compares this module's output number for number):
  day_i   = d0 + cumsum(dd)            (calendar days)
  cint_i  = cumsum(c)                  (integer, exact)
  close_i = cint_i / 10^k ; open/high/low_i = (cint_i + o/h/l_i) / 10^k
  v_i     = shares, rounded
  rc_i    = close_i * ratio, ratio = the last f entry [index, ratio] with index <= i
            (no applicable entry -> raw close unknown = NaN here, null in TS)
Prices are adjustment=all (splits + dividends, back-adjusted with today's
knowledge): use them for RETURNS; use the raw close for point-in-time PRICE
FILTERS (a $5 floor on a back-adjusted price would be lookahead).

SERIES POLICY (series(), documented because it changes what a backtest sees):
  * A symbol that carries "raw" (asof=-, unmapped) lines uses ONLY those and
    drops its default lines: the writer adds raw lines exactly when the default
    mapping is suspect (ticker reuse, or an active name the default mapping tied
    to an older company — the S -> Sprint case). research/price_panel_coverage.md
    caveat 4: "Within a reuse-suspect symbol, prefer the raw segments and the
    event date."
  * A raw segment that spans its symbol's corporate-action date is split there
    (the old company ends on the event date, the new one starts after it), so a
    return is never computed across two companies that share a ticker.
  * Default segments are taken as written (the writer already clipped delisted
    series at their dated event and flagged the rest inactive_no_event_date).

POINT-IN-TIME: a series is in the universe on date D only if it has a bar ON D
(alive_on / eligible_mask) — membership never consults a later bar, a segment
end or a status the writer learned afterwards. Filters use trailing windows
ending at D: min price on the RAW close, min trailing dollar volume.

MEMORY: iter_lines streams one shard and one line at a time (gzip + json per
line); series() holds only one shard's lines (one writer batch, <= 50 symbols)
at a time. Arrays are numpy float64 / int64.

CLI:
  python3 price_panel.py --summary [--root DIR]   universe + coverage stats (JSON)
  python3 price_panel.py --dump DIR               every line decoded, one JSON per line
                                                  (the TS parity test reads this)
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Iterator, List, Optional

import numpy as np

BUILD_ID_RE = re.compile(r"^b\d{8}$")
SHARD_RE = re.compile(r"^b\d{5}\.jsonl\.gz$")
_EPOCH_ORD = date(1970, 1, 1).toordinal()


def default_root() -> str:
    """storage_config.PRICE_PANEL_DIR (DATA_DIR/price_panel), the writer's root."""
    try:
        import storage_config
        return storage_config.PRICE_PANEL_DIR
    except ImportError as e:  # storage_config is in-repo; say so loudly if it is not importable
        print(f"[price_panel] storage_config unavailable ({e}); falling back to /tmp/price_panel", file=sys.stderr)
        return os.path.join("/tmp", "price_panel")


def day_num(d: str) -> int:
    """Days since 1970-01-01 (pricePanel.ts dayNum)."""
    return date.fromisoformat(d).toordinal() - _EPOCH_ORD


def day_str(n: int) -> str:
    """pricePanel.ts dayStr."""
    return date.fromordinal(int(n) + _EPOCH_ORD).isoformat()


# ── build discovery ─────────────────────────────────────────────────────────
def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            v = json.load(fh)
        return v if isinstance(v, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        print(f"[price_panel] unreadable {path}: {e}", file=sys.stderr)
        return None


def current_build(root: Optional[str] = None) -> Optional[dict]:
    """The last COMPLETE build ({build_id, completed_at, dir, manifest}) or None.
    Only current.json is trusted: a build still in progress (building.json) is
    never read, so a reader never sees a half-written panel."""
    root = root or default_root()
    cur = _read_json(os.path.join(root, "current.json"))
    if not cur or not BUILD_ID_RE.match(str(cur.get("build_id", ""))):
        return None
    d = os.path.join(root, cur["build_id"])
    if not os.path.isdir(os.path.join(d, "bars")):
        return None
    return {"build_id": cur["build_id"], "completed_at": cur.get("completed_at"), "dir": d,
            "manifest": _read_json(os.path.join(d, "manifest.json")) or {}}


def shard_paths(build_dir: str) -> List[str]:
    bars = os.path.join(build_dir, "bars")
    try:
        names = sorted(n for n in os.listdir(bars) if SHARD_RE.match(n))
    except FileNotFoundError:
        return []
    return [os.path.join(bars, n) for n in names]


def iter_shard(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def iter_lines(build_dir: str) -> Iterator[dict]:
    """Every panel line, shard by shard, one line in memory at a time."""
    for p in shard_paths(build_dir):
        yield from iter_shard(p)


# ── decoding ────────────────────────────────────────────────────────────────
def decode_rows(line: dict) -> List[dict]:
    """Scalar mirror of pricePanel.ts decodeSegment (the reference the numpy
    decoder is tested against): [{d, o, h, l, c, v, rc}], rc None when unknown."""
    m = 10 ** int(line["k"])
    rows = []
    day, c, fi, ratio = day_num(line["d0"]), 0, 0, None
    f = line.get("f") or []
    for i in range(int(line["n"])):
        day += line["dd"][i]
        c += line["c"][i]
        while fi < len(f) and f[fi][0] <= i:
            ratio = f[fi][1]
            fi += 1
        cc = c / m
        rows.append({"d": day_str(day), "o": (c + line["o"][i]) / m, "h": (c + line["h"][i]) / m,
                     "l": (c + line["l"][i]) / m, "c": cc, "v": line["v"][i],
                     "rc": None if ratio is None else cc * ratio})
    return rows


def decode_arrays(line: dict) -> dict:
    """Vectorized decodeSegment. Integer cumulative sums are exact (int64), and
    each price is one IEEE division / multiplication — the same operations, so
    the same doubles, as the TS decoder."""
    n = int(line["n"])
    m = float(10 ** int(line["k"]))
    days = day_num(line["d0"]) + np.cumsum(np.asarray(line["dd"], dtype=np.int64))
    cint = np.cumsum(np.asarray(line["c"], dtype=np.int64))
    close = cint / m
    out = {
        "days": days,
        "o": (cint + np.asarray(line["o"], dtype=np.int64)) / m,
        "h": (cint + np.asarray(line["h"], dtype=np.int64)) / m,
        "l": (cint + np.asarray(line["l"], dtype=np.int64)) / m,
        "c": close,
        "v": np.asarray(line["v"], dtype=np.int64),
    }
    ratio = np.full(n, np.nan)
    f = line.get("f") or []
    for j, (idx, r) in enumerate(f):
        end = f[j + 1][0] if j + 1 < len(f) else n
        ratio[max(0, int(idx)):max(0, int(end))] = float(r)
    out["rc"] = close * ratio
    if not (len(days) == len(close) == n):
        raise ValueError(f"line {line.get('s')}/{line.get('mode')}/{line.get('seg')}: n={n} but {len(close)} rows decoded")
    return out


# ── series ──────────────────────────────────────────────────────────────────
@dataclass
class Series:
    """One price series: a (symbol, mapping mode, date segment[, event part])."""
    symbol: str
    mode: str
    seg: int
    part: int                     # 0, or 1 for the post-event half of a split raw segment
    asset_ids: List[str]
    status: str                   # writer's universe status: active | inactive | corp_action
    event: Optional[dict]         # {kind, date} or None
    flags: List[str]
    days: np.ndarray              # int64 days since epoch
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray                 # adjusted close (adjustment=all)
    v: np.ndarray                 # shares (as served with adjustment=all)
    raw_close: np.ndarray         # NaN where unknown
    adjusted: bool = True
    event_ends_series: bool = False   # this series is the part that ends AT its event
    _dates: Optional[List[str]] = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return f"{self.symbol}|{self.mode}|{self.seg}|{self.part}"

    @property
    def asset_id(self) -> Optional[str]:
        return self.asset_ids[0] if self.asset_ids else None

    @property
    def dates(self) -> List[str]:
        if self._dates is None:
            self._dates = [day_str(d) for d in self.days]
        return self._dates

    @property
    def first_day(self) -> int:
        return int(self.days[0])

    @property
    def last_day(self) -> int:
        return int(self.days[-1])

    @property
    def delisted(self) -> bool:
        """The writer marked the symbol not-active, or this series is the part
        that ends at a dated corporate action (merger, rename, worthless)."""
        return self.status != "active" or self.event_ends_series or "clipped_after_event" in self.flags

    def ends_before(self, as_of: str, grace_days: int = 7) -> bool:
        """The series stops trading > grace_days before the panel's as_of (an
        outcome-side fact — never used for universe membership)."""
        return self.last_day < day_num(as_of) - grace_days

    def index_on(self, day: int) -> int:
        """Bar index ON `day`, or -1."""
        i = int(np.searchsorted(self.days, day))
        return i if i < len(self.days) and int(self.days[i]) == day else -1

    def index_at_or_before(self, day: int) -> int:
        """Last bar index with date <= day, or -1."""
        return int(np.searchsorted(self.days, day, side="right")) - 1


def _series_from(line: dict, arr: dict, lo: int, hi: int, part: int, event_ends: bool) -> Series:
    return Series(
        symbol=str(line["s"]), mode=str(line["mode"]), seg=int(line["seg"]), part=part,
        asset_ids=[str(x) for x in (line.get("ids") or [])], status=str(line.get("st") or ""),
        event=line.get("ev") if isinstance(line.get("ev"), dict) else None,
        flags=[str(x) for x in (line.get("flags") or [])],
        days=arr["days"][lo:hi], o=arr["o"][lo:hi], h=arr["h"][lo:hi], l=arr["l"][lo:hi],
        c=arr["c"][lo:hi], v=arr["v"][lo:hi], raw_close=arr["rc"][lo:hi], event_ends_series=event_ends,
    )


EVENT_END_TOLERANCE_DAYS = 10   # research/price_panel_coverage.md: probes "reach their event (within 10 days)"


def _ends_at_event(days: np.ndarray, ev_day: int) -> bool:
    """The series' last bar is on, or at most EVENT_END_TOLERANCE_DAYS before,
    the corporate-action date — i.e. this series is the one the event ends."""
    return len(days) > 0 and 0 <= ev_day - int(days[-1]) <= EVENT_END_TOLERANCE_DAYS


def series_from_lines(lines: List[dict], prefer_raw: bool = True, split_at_event: bool = True) -> List[Series]:
    """Apply the SERIES POLICY (module docstring) to the lines of one shard."""
    by_symbol: Dict[str, List[dict]] = {}
    for ln in lines:
        by_symbol.setdefault(str(ln["s"]), []).append(ln)
    out: List[Series] = []
    for sym in sorted(by_symbol):
        group = by_symbol[sym]
        if prefer_raw and any(ln.get("mode") == "raw" for ln in group):
            group = [ln for ln in group if ln.get("mode") == "raw"]
        for ln in sorted(group, key=lambda x: (str(x.get("mode")), int(x.get("seg", 0)))):
            if int(ln.get("n", 0)) <= 0:
                continue
            arr = decode_arrays(ln)
            n = len(arr["c"])
            ev = ln.get("ev") if isinstance(ln.get("ev"), dict) else None
            ev_day = day_num(ev["date"]) if ev and re.match(r"^\d{4}-\d{2}-\d{2}$", str(ev.get("date", ""))) else None
            if split_at_event and ln.get("mode") == "raw" and ev_day is not None:
                cut = int(np.searchsorted(arr["days"], ev_day, side="right"))  # bars <= event day stay with the old company
                if 0 < cut < n:
                    out.append(_series_from(ln, arr, 0, cut, 0, _ends_at_event(arr["days"][:cut], ev_day)))
                    out.append(_series_from(ln, arr, cut, n, 1, False))
                    continue
            out.append(_series_from(ln, arr, 0, n, 0, ev_day is not None and _ends_at_event(arr["days"], ev_day)))
    return out


def iter_series(build_dir: str, prefer_raw: bool = True, split_at_event: bool = True) -> Iterator[Series]:
    """Every series, shard by shard (the writer puts a symbol's default AND raw
    lines in the same batch shard, so per-shard grouping is complete)."""
    for p in shard_paths(build_dir):
        yield from series_from_lines(list(iter_shard(p)), prefer_raw=prefer_raw, split_at_event=split_at_event)


# ── point-in-time universe ──────────────────────────────────────────────────
def trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of x over the last `window` bars INCLUDING bar i (fewer at the
    start). Never reaches past bar i."""
    x = np.asarray(x, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - window)
    return (cs[idx] - cs[lo]) / (idx - lo)


def pit_price(s: Series) -> np.ndarray:
    """Point-in-time price for filters: the raw close; the adjusted close only
    where the writer recorded no raw close (counted by the caller)."""
    return np.where(np.isnan(s.raw_close), s.c, s.raw_close)


def eligible_mask(s: Series, min_price: float, min_dollar_volume: float, adv_window: int = 20) -> np.ndarray:
    """Per bar: the series has a bar that day (by construction), raw close >=
    min_price, and trailing adv_window-bar mean of close x volume >=
    min_dollar_volume. Dollar volume uses adjusted close x served volume, which
    equals raw dollar volume divided by the later dividend factor (splits
    cancel) — slightly UNDERSTATED for dividend payers, so the filter is a hair
    stricter than intended, never looser."""
    dv = trailing_mean(s.c * s.v.astype(np.float64), adv_window)
    return (pit_price(s) >= min_price) & (dv >= min_dollar_volume)


def alive_on(s: Series, day: int) -> bool:
    """In the universe on `day` = has a bar on `day`. Nothing after `day` is consulted."""
    return s.index_on(day) >= 0


def universe_on(series: List[Series], d: str, min_price: float = 0.0, min_dollar_volume: float = 0.0,
                adv_window: int = 20) -> List[str]:
    """Keys of the series eligible on date d (small-panel convenience; the
    walk-forward harness uses eligible_mask per series instead)."""
    day = day_num(d)
    out = []
    for s in series:
        i = s.index_on(day)
        if i >= 0 and bool(eligible_mask(s, min_price, min_dollar_volume, adv_window)[i]):
            out.append(s.key)
    return out


# ── stats / CLI ─────────────────────────────────────────────────────────────
def summarize(build: dict) -> dict:
    n_series = n_delisted = n_bars = n_no_raw = 0
    symbols, delisted_symbols = set(), set()
    first = last = None
    for s in iter_series(build["dir"]):
        n_series += 1
        n_bars += len(s.c)
        n_no_raw += int(np.isnan(s.raw_close).sum())
        symbols.add(s.symbol)
        if s.delisted:
            n_delisted += 1
            delisted_symbols.add(s.symbol)
        first = s.first_day if first is None else min(first, s.first_day)
        last = s.last_day if last is None else max(last, s.last_day)
    return {
        "build_id": build["build_id"], "as_of": build["manifest"].get("as_of"),
        "series": n_series, "symbols": len(symbols), "bars": n_bars,
        "delisted_series": n_delisted, "delisted_symbols": len(delisted_symbols),
        "delisted_symbol_share": round(len(delisted_symbols) / len(symbols), 4) if symbols else None,
        "bars_without_raw_close": n_no_raw,
        "first_bar": day_str(first) if first is not None else None,
        "last_bar": day_str(last) if last is not None else None,
    }


def _json_num(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def dump(build_dir: str, out=sys.stdout) -> int:
    """Every line decoded by decode_arrays, one JSON object per line, in file
    order: {s, mode, seg, rows: [{d, o, h, l, c, v, rc}]}."""
    n = 0
    for ln in iter_lines(build_dir):
        a = decode_arrays(ln)
        rows = [{"d": day_str(a["days"][i]), "o": float(a["o"][i]), "h": float(a["h"][i]), "l": float(a["l"][i]),
                 "c": float(a["c"][i]), "v": int(a["v"][i]), "rc": _json_num(float(a["rc"][i]))}
                for i in range(len(a["c"]))]
        out.write(json.dumps({"s": ln["s"], "mode": ln["mode"], "seg": ln["seg"], "rows": rows}) + "\n")
        n += 1
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="panel root (default storage_config.PRICE_PANEL_DIR)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--dump", metavar="ROOT_OR_BUILD_DIR")
    a = ap.parse_args(argv)
    if a.dump:
        b = current_build(a.dump)
        dump(b["dir"] if b else a.dump)
        return 0
    b = current_build(a.root)
    if not b:
        print(json.dumps({"status": "no_panel", "root": a.root or default_root()}))
        return 0
    print(json.dumps(summarize(b)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
