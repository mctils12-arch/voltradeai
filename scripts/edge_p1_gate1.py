#!/usr/bin/env python3
"""scripts/edge_p1_gate1.py — Edge P1 ladder GATE 1 (DATA): is our own
open-interest GEX consistent with SqueezeMetrics' published GEX?

Spec is FIXED by research/edge_preregistration.md § P1 (kill criteria, ladder
path): "Kill at gate 1 instead (data dies, not the idea) if our own
BS-recomputed GEX from Alpaca OI disagrees in sign with SqueezeMetrics on
> 30% of the overlapping days once >= 40 such days exist."

  own_gex(day) = sum over contracts of  sign * gamma * OI * 100 * spot^2 * 0.01
                 sign = +1 call, -1 put (SqueezeMetrics' dealer convention:
                 dealers long calls / short puts). Contracts: SPY chain rows
                 (optionchains/<day>.jsonl[.gz]) joined by OCC symbol to the OI
                 sidecar (optionchains/oi/<day>.jsonl[.gz]). gamma = the
                 archive's Alpaca gamma; if absent, Black-Scholes gamma from
                 the archived iv (r = q = 0). Rows missing gamma/iv, OI, or
                 spot are DROPPED and counted, never zero-filled.
  verdict      : PENDING (< 40 overlapping days) | PASS | KILL (> 30% sign
                 disagreement). Rank correlation (levels and day-over-day
                 changes) is REPORTED ONLY — the pre-registration gives it no
                 threshold, and none is invented here.
  caveat       : SqueezeMetrics GEX is positive on the great majority of days
                 so sign agreement alone is a weak test; the report states the
                 share of positive Squeeze days so a pass is read honestly.
                 Own GEX is SPY-only vs Squeeze's SPX — a proxy, not a twin.

Reads the Railway volume archive (default $DATA_DIR/datacore_archive, as
server/datacoreArchive.ts archiveBaseDir()). Nothing is written. Internal
research only; nothing here is served or traded.
"""
import argparse
import gzip
import json
import math
import os
import re
import sys

MIN_DAYS = 40            # pre-registered
MAX_DISAGREE = 0.30      # pre-registered
UNDERLYING = "SPY"
_OCC = re.compile(r"^([A-Z.]{1,6})(\d{6})([CP])(\d{8})$")


def archive_base():
    d = os.environ.get("DATA_DIR") or ("/data/voltrade" if os.path.exists("/data") else "/tmp")
    return os.path.join(d, "datacore_archive")


def parse_occ(sym):
    """-> (underlying, 'YYMMDD', 'C'|'P', strike) or None."""
    m = _OCC.match(sym or "")
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), int(m.group(4)) / 1000.0


def _read_jsonl(path):
    for p in (path, path + ".gz"):
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt", encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
            return


def bs_gamma(spot, strike, t_years, iv):
    """Black-Scholes gamma, r = q = 0. None when inputs can't support it."""
    if not (spot and strike and t_years and iv) or min(spot, strike, t_years, iv) <= 0:
        return None
    sd = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * t_years) / sd
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (spot * sd)


def _years_to_expiry(yymmdd, day):
    from datetime import date
    e = date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:]))
    d = date.fromisoformat(day)
    return max((e - d).days, 0) / 365.0


def own_gex(chain_rows, oi_rows, day, underlying=UNDERLYING):
    """-> (gex or None, stats). Pure: takes already-read rows."""
    oi = {}
    for r in oi_rows:
        if r.get("u") == underlying and r.get("oi") is not None:
            oi[r.get("s")] = float(r["oi"])
    total, used = 0.0, 0
    drop = {"no_oi": 0, "no_gamma": 0, "no_spot": 0, "bad_symbol": 0}
    for r in chain_rows:
        if r.get("u") != underlying:
            continue
        occ = parse_occ(r.get("s"))
        if not occ:
            drop["bad_symbol"] += 1
            continue
        n = oi.get(r["s"])
        if n is None:
            drop["no_oi"] += 1
            continue
        spot = r.get("spot")
        if not spot or spot <= 0:
            drop["no_spot"] += 1
            continue
        g = r.get("gamma")
        if g is None:
            g = bs_gamma(spot, occ[3], _years_to_expiry(occ[1], day), r.get("iv"))
        if g is None:
            drop["no_gamma"] += 1
            continue
        total += (1 if occ[2] == "C" else -1) * g * n * 100.0 * spot * spot * 0.01
        used += 1
    return (total if used else None), {"used": used, **drop}


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return out


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def compare(own, squeeze):
    """own, squeeze: {day: gex}. Verdict per the pre-registration."""
    days = sorted(d for d in own if d in squeeze and own[d] is not None)
    n = len(days)
    o = [own[d] for d in days]
    s = [squeeze[d] for d in days]
    disagree = sum(1 for x, y in zip(o, s) if (x > 0) != (y > 0) or x == 0 or y == 0)
    share = disagree / n if n else None
    if n < MIN_DAYS:
        verdict = "PENDING"
    else:
        verdict = "KILL" if share > MAX_DISAGREE else "PASS"
    d_o = [b - a for a, b in zip(o, o[1:])]
    d_s = [b - a for a, b in zip(s, s[1:])]
    return {
        "verdict": verdict, "overlap_days": n, "min_days": MIN_DAYS,
        "sign_disagree_days": disagree, "sign_disagree_share": share,
        "max_disagree_share": MAX_DISAGREE,
        "squeeze_positive_share": (sum(1 for y in s if y > 0) / n) if n else None,
        "rank_corr_levels": spearman(o, s), "rank_corr_changes": spearman(d_o, d_s),
        "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
    }


def load(base):
    ch = os.path.join(base, "optionchains")
    days = sorted({m.group(1) for f in (os.listdir(ch) if os.path.isdir(ch) else [])
                   for m in [re.match(r"^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$", f)] if m})
    sq = {}
    sdir = os.path.join(base, "squeeze_gex")
    for f in (sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []):
        if re.match(r"^\d{4}\.jsonl(\.gz)?$", f):
            for r in _read_jsonl(os.path.join(sdir, re.sub(r"\.gz$", "", f))):
                sq[r["date"]] = float(r["gex"])
    own, stats = {}, {}
    for d in days:
        oi_path = os.path.join(ch, "oi", d + ".jsonl")
        if not (os.path.exists(oi_path) or os.path.exists(oi_path + ".gz")):
            continue
        own[d], stats[d] = own_gex(_read_jsonl(os.path.join(ch, d + ".jsonl")),
                                   _read_jsonl(oi_path), d)
    return own, sq, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive-dir", default=archive_base())
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    own, sq, stats = load(a.archive_dir)
    rep = compare(own, sq)
    rep["days_with_own_gex"] = sum(1 for v in own.values() if v is not None)
    rep["days_with_oi_sidecar"] = len(own)
    print(json.dumps(rep, indent=2) if a.json else "\n".join(f"{k}: {v}" for k, v in rep.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
