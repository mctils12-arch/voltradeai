"""Lazy Prices PR 3c: the panel adapter between sec_filings_gate2 (keyed by an
opaque id) and price_panel.py (keyed by ticker series, 2016+, delisted-inclusive).

Nothing here is a signal claim and nothing touches a trade path. It only does
the two jobs the harness deliberately leaves to the caller:

  1. JOIN   prepare_rows(): rewrite each scored 10-K row's `cik` to the TICKER
            that CIK carried ON THE FILING DATE (sec_cik_symbol_map.resolve:
            dated, delisting-inclusive, ambiguous -> dropped, never guessed).
            Every drop is COUNTED by reason; none is silent.
  2. PRICES PanelAdapter / build_panel(): stream price_panel.iter_series ONCE,
            keep only the symbols some row resolved to (memory-bounded; numpy
            arrays, RSS-guarded), and expose the SimplePanel duck interface
            (days, open, close, last_close_on_or_before, eligible).

LOOKAHEAD RULES (pinned by test_lazy_prices_panel.py):
  * eligibility for an entry on day D uses the bar BEFORE D (trailing $ADV and
    raw-close price floor evaluated on D-1) — never D's own bar.
  * the calendar of record is the SPY series; a symbol with no bar on a day has
    no price that day (no forward-filling an entry).
  * a symbol served by several price series (ticker reuse / segments) is merged
    by date, first series wins on a duplicate day; the count of such symbols is
    reported so a reuse-suspect result can be discounted. Merged series carry
    one residual risk: `last_close_on_or_before` could step across two
    companies that reused a ticker (price_panel already splits raw segments at
    the corporate-action date, which is the main guard).

ELIGIBILITY (fixed here BEFORE any return is computed; changing it is a new
variant): raw close >= $2 AND trailing 20-bar mean dollar volume >= $1M on the
pre-entry bar. $1M is the registered ADV floor in sec_filings_gate2; the $2
price floor is borrowed from Edge P2 and added so sub-$2 prints cannot carry
the sort.

BIAS DIRECTION: the join keeps only names that ever filed a Form 3/4/5 with a
ticker (insider-visible), so the tested universe tilts to names with insider
activity; the unmapped ~30% is excluded, not zero-filled. Read any result as
'insider-visible names only'.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
from collections import Counter
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

import price_panel as pp
import sec_cik_symbol_map as cmap
import sec_filings_gate2 as g2

MIN_PRICE = 2.0
MIN_ADV = 1_000_000.0
ADV_BARS = 20
DEFAULT_MAX_RSS_MB = 1500.0


class MemoryBudgetExceeded(RuntimeError):
    pass


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # KB on Linux


def prepare_rows(rows: Iterable[dict], cik_map: dict, slack_days: int = cmap.SLACK_DAYS) -> Tuple[list, dict]:
    """-> (rows with `cik` replaced by the dated ticker, counts). The original
    CIK is kept as `cik_orig`. Rows with no usable date or no resolution are
    dropped and counted: undated | none | ambiguous."""
    out, st = [], Counter()
    for r in rows:
        day = (r.get("accepted_at") or "")[:10]
        if not day:
            st["undated"] += 1
            continue
        tkr, status = cmap.resolve(cik_map, r.get("cik"), day, slack_days)
        st[status] += 1
        if tkr is None:
            continue
        nr = dict(r)
        nr["cik_orig"] = r.get("cik")
        nr["cik"] = tkr
        out.append(nr)
    st["rows_out"] = len(out)
    return out, dict(st)


class PanelAdapter:
    """SimplePanel duck type over per-symbol numpy arrays."""

    def __init__(self, days: list):
        self.days = list(days)
        self._s: Dict[str, tuple] = {}     # sym -> (day_num[], open[], close[], eligible_pre[])
        self.multi_series_symbols = 0

    def add(self, sym: str, day_num: np.ndarray, o: np.ndarray, c: np.ndarray, elig: np.ndarray) -> None:
        self._s[sym] = (day_num, o, c, elig)

    def symbols(self) -> set:
        return set(self._s)

    def _idx(self, sym: str, iso: str) -> Optional[int]:
        t = self._s.get(sym)
        if t is None:
            return None
        d = pp.day_num(iso)
        i = int(np.searchsorted(t[0], d))
        return i if i < len(t[0]) and int(t[0][i]) == d else None

    def open(self, sym, iso) -> Optional[float]:
        i = self._idx(sym, iso)
        if i is None:
            return None
        v = float(self._s[sym][1][i])
        return v if np.isfinite(v) and v > 0 else None

    def close(self, sym, iso) -> Optional[float]:
        i = self._idx(sym, iso)
        if i is None:
            return None
        v = float(self._s[sym][2][i])
        return v if np.isfinite(v) and v > 0 else None

    def last_close_on_or_before(self, sym, iso) -> Optional[tuple]:
        t = self._s.get(sym)
        if t is None:
            return None
        i = int(np.searchsorted(t[0], pp.day_num(iso), side="right")) - 1
        while i >= 0:
            v = float(t[2][i])
            if np.isfinite(v) and v > 0:
                return pp.day_str(int(t[0][i])), v
            i -= 1
        return None

    def eligible(self, sym, iso) -> bool:
        i = self._idx(sym, iso)
        return bool(i is not None and self._s[sym][3][i])


def _merge(series: list) -> tuple:
    d = np.concatenate([s.days for s in series])
    o = np.concatenate([s.o for s in series])
    c = np.concatenate([s.c for s in series])
    el = np.concatenate([_pre_entry_eligible(s) for s in series])
    order = np.argsort(d, kind="stable")
    d, o, c, el = d[order], o[order], c[order], el[order]
    keep = np.concatenate([[True], d[1:] != d[:-1]])        # first series wins on a duplicate day
    return d[keep], o[keep], c[keep], el[keep]


def _pre_entry_eligible(s: "pp.Series") -> np.ndarray:
    """eligible_pre[i] = the filters hold on bar i-1 (the last bar known before
    the entry open on bar i). Bar 0 has no prior bar -> False."""
    e = pp.eligible_mask(s, MIN_PRICE, MIN_ADV, ADV_BARS)
    out = np.zeros(len(e), dtype=bool)
    out[1:] = e[:-1]
    return out


def build_panel(build_dir: str, symbols: set, max_rss_mb: float = DEFAULT_MAX_RSS_MB) -> Tuple[PanelAdapter, dict]:
    """One streaming pass; keeps only `symbols` (+ SPY for the calendar)."""
    keep: Dict[str, list] = {}
    cal: Optional[np.ndarray] = None
    scanned = 0
    for s in pp.iter_series(build_dir):
        scanned += 1
        if scanned % 2000 == 0 and _rss_mb() > max_rss_mb:
            raise MemoryBudgetExceeded(f"RSS {_rss_mb():.0f} MB > {max_rss_mb:.0f} MB at series {scanned}")
        if s.symbol == "SPY" and len(s.days) > 1000 and (cal is None or len(s.days) > len(cal)):
            cal = s.days.copy()
        if s.symbol in symbols and len(s.days) > 0:
            keep.setdefault(s.symbol, []).append(s)
    if cal is None:
        raise RuntimeError("SPY calendar series not found in the panel build")
    pa = PanelAdapter([pp.day_str(int(x)) for x in cal])
    for sym, ss in keep.items():
        pa.add(sym, *_merge(ss))
        if len(ss) > 1:
            pa.multi_series_symbols += 1
    stats = {"series_scanned": scanned, "symbols_requested": len(symbols), "symbols_in_panel": len(keep),
             "multi_series_symbols": pa.multi_series_symbols, "peak_rss_mb": round(_rss_mb(), 1)}
    return pa, stats


def run_gate2(rows: list, cik_map: dict, build_dir: str, variants: int = 1,
              max_rss_mb: float = DEFAULT_MAX_RSS_MB) -> dict:
    prepared, join = prepare_rows(rows, cik_map)
    panel, pstats = build_panel(build_dir, {r["cik"] for r in prepared}, max_rss_mb)
    in_panel = [r for r in prepared if r["cik"] in panel.symbols()]
    join["resolved_not_in_panel"] = len(prepared) - len(in_panel)
    res = g2.run(in_panel, panel, variants=variants)
    res["join"] = join
    res["panel"] = pstats
    res["caveat"] = "insider-visible names only (CIK->ticker via Form 3/4/5); 2016+ panel; provisional until 12-month live shadow"
    return res


def main(argv=None) -> int:
    import sec_issuer_pairs as sip
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", required=True, help="rows JSON from `sec_filings_similarity distribution --pit --rows-json`")
    ap.add_argument("--pairs-dir", required=True, help="sec_issuer_pairs cache dir")
    ap.add_argument("--panel-root", default=None, help="price panel root (default: price_panel.default_root())")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--max-rss-mb", type=float, default=DEFAULT_MAX_RSS_MB)
    ap.add_argument("--out", default=None, help="write the full result JSON here")
    a = ap.parse_args(argv)
    with open(a.rows) as f:
        rows = json.load(f)
    cm = cmap.build_map(sip.rows_from_pairs_cache(a.pairs_dir))
    cur = pp.current_build(a.panel_root)
    if not cur:
        print("no complete price-panel build under the panel root", file=sys.stderr)
        return 2
    res = run_gate2(rows, cm, cur["dir"], a.variants, a.max_rss_mb)
    print(json.dumps({"join": res["join"], "panel": res["panel"]}, indent=1))
    print(g2.report(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
