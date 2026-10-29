"""walkforward_synth.py — synthetic price panels for testing walkforward.py.

TEST SUPPORT ONLY (the sandbox has no panel and no Alpaca keys; the real run
happens in production on server/pricePanel.ts's build). encode_segment mirrors
pricePanel.ts encodeSegment (Math.round = floor(x + 0.5); raw/adj factor runs
at 10 significant digits), so price_panel.py reads these panels through the
same decoder it reads production panels with; the decoder itself is pinned to
the TS encoder by server/pricePanelPyParity.test.ts.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
from datetime import date, timedelta
from typing import Dict, List, Optional

_EPOCH_ORD = date(1970, 1, 1).toordinal()


def _day(d: str) -> int:
    return date.fromisoformat(d).toordinal() - _EPOCH_ORD


def _jsround(x: float) -> int:
    return int(math.floor(x + 0.5))


def scale_for(rows: List[dict]) -> int:
    mn = min((p for r in rows for p in (r["o"], r["h"], r["l"], r["c"]) if p > 0), default=None)
    if mn is None:
        return 4
    return min(10, max(4, math.ceil(4 - math.log10(mn))))


def encode_segment(rows: List[dict]) -> dict:
    k = scale_for(rows)
    m = 10 ** k
    out = {"d0": rows[0]["d"], "n": len(rows), "k": k, "dd": [], "c": [], "o": [], "h": [], "l": [], "v": [], "f": []}
    prev_day, prev_c, last_ratio = _day(rows[0]["d"]), 0, None
    for i, r in enumerate(rows):
        day, c = _day(r["d"]), _jsround(r["c"] * m)
        out["dd"].append(day - prev_day)
        prev_day = day
        out["c"].append(c - prev_c)
        prev_c = c
        out["o"].append(_jsround(r["o"] * m) - c)
        out["h"].append(_jsround(r["h"] * m) - c)
        out["l"].append(_jsround(r["l"] * m) - c)
        out["v"].append(_jsround(r["v"]))
        if r.get("rc") is not None and r["c"] > 0:
            ratio = float(f"{r['rc'] / r['c']:.10g}")
            if last_ratio is None or abs(ratio / last_ratio - 1) > 1e-6:
                out["f"].append([i, ratio])
                last_ratio = ratio
    return out


def bdays(start: str, end: str, holidays=()) -> List[str]:
    d, e, out = date.fromisoformat(start), date.fromisoformat(end), []
    hol = set(holidays)
    while d <= e:
        if d.weekday() < 5 and d.isoformat() not in hol:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def path_rows(days: List[str], p0: float, drift: float, vol: float, v0: float, rng: random.Random,
              raw_ratio: float = 1.0) -> List[dict]:
    rows, p = [], p0
    for d in days:
        p *= math.exp(drift + rng.gauss(0, vol))
        o = p * math.exp(rng.gauss(0, vol / 3))
        c = p
        h = max(o, c) * (1 + abs(rng.gauss(0, vol / 2)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, vol / 2)))
        rows.append({"d": d, "o": round(o, 4), "h": round(h, 4), "l": round(lo, 4), "c": round(c, 4),
                     "v": int(v0 * math.exp(rng.gauss(0, 0.3))), "rc": round(c * raw_ratio, 6)})
    return rows


def line(symbol: str, rows: List[dict], status: str = "active", event: Optional[dict] = None, mode: str = "default",
         seg: int = 0, flags: Optional[List[str]] = None) -> dict:
    return {"s": symbol, "mode": mode, "seg": seg, "seg_start": rows[0]["d"], "seg_end": rows[-1]["d"], "ids": [f"id-{symbol}"],
            "st": status, "ev": event, "flags": flags or [], **encode_segment(rows)}


def write_panel(root: str, lines: List[dict], as_of: str, build_id: str = "b20261029", per_shard: int = 25) -> str:
    bars = os.path.join(root, build_id, "bars")
    os.makedirs(bars, exist_ok=True)
    by_sym: Dict[str, List[dict]] = {}
    for ln in lines:
        by_sym.setdefault(ln["s"], []).append(ln)
    syms = sorted(by_sym)
    for b in range(0, len(syms), per_shard):
        with gzip.open(os.path.join(bars, f"b{b // per_shard:05d}.jsonl.gz"), "wt", encoding="utf-8") as fh:
            for sym in syms[b:b + per_shard]:
                for ln in by_sym[sym]:
                    fh.write(json.dumps(ln, separators=(",", ":")) + "\n")
    with open(os.path.join(root, build_id, "manifest.json"), "w") as fh:
        json.dump({"version": 1, "build_id": build_id, "status": "complete", "as_of": as_of}, fh)
    with open(os.path.join(root, "current.json"), "w") as fh:
        json.dump({"build_id": build_id, "completed_at": f"{as_of}T23:00:00Z"}, fh)
    return root


def synthetic_panel(root: str, n_names: int = 60, start: str = "2016-01-04", end: str = "2018-12-31",
                    signal: float = 0.0, seed: int = 7, delist: int = 4, ipo: int = 3) -> dict:
    """SPY + VXX + n_names stocks. `signal` > 0 gives each stock a persistent
    daily drift in [-signal, +signal] (momentum continues); 0 = pure noise.
    The first `delist` names stop mid-sample (half at a dated merger, half
    undated/inactive); the next `ipo` names list late."""
    rng = random.Random(seed)
    days = bdays(start, end)
    lines = [line("SPY", path_rows(days, 200.0, 0.0003, 0.01, 8e7, rng)),
             line("VXX", path_rows(days, 30.0, -0.001, 0.04, 3e7, rng))]
    meta = {"delisted": [], "ipo": [], "drift": {}}
    for i in range(n_names):
        sym = f"S{i:03d}"
        drift = rng.uniform(-signal, signal) if signal else 0.0
        meta["drift"][sym] = drift
        dd = days
        status, ev, flags = "active", None, []
        if i < delist:
            cut = len(days) // 2 + i * 20
            dd = days[:cut]
            status = "inactive"
            if i % 2 == 0:
                ev = {"kind": "merger", "date": dd[-1]}
            else:
                flags = ["inactive_no_event_date"]
            meta["delisted"].append(sym)
        elif i < delist + ipo:
            dd = days[300 + 10 * i:]
            meta["ipo"].append(sym)
        rows = path_rows(dd, rng.uniform(15, 120), drift, 0.02, rng.uniform(1.5e6, 9e6), rng, raw_ratio=1.0 + 0.02 * (i % 3))
        lines.append(line(sym, rows, status=status, event=ev, flags=flags))
    write_panel(root, lines, as_of=end)
    meta["days"] = days
    return meta
