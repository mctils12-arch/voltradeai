#!/usr/bin/env python3
"""
tankfill_gate1.py — tank-fill v2 PR-4: the ROOT VALIDATION LADDER gate-1
attempt. Matches readings_v2 scene composites to EIA weekly Cushing crude
stocks and scores the PRE-STATED criteria (workup, open_questions.md;
prior logged in experiments.md v1.0.109 BEFORE this ran):

  PASS = n_matched >= 20 scene-weeks spanning >= 1 EIA inventory reversal
         AND delta Pearson r >= +0.3
         AND delta-sign hit rate >= 65%
  (levels r >= +0.5 is SUPPORTING evidence only — deltas are binding.)

Matching rules, stated BEFORE seeing results (Reasoning Standard #10):
  - scene composite = D^2-weighted mean fill across ALL measured tanks
  - scene matched to the EIA week (Friday stock date) within +/-3 days,
    same as the v1 comparator; one scene per EIA week (lowest cloud wins)
  - DELTAS pair successive matched weeks with gap <= 14 days (2 EIA
    prints); the EIA delta is the stock change over the same gap. Longer
    gaps are dropped from the delta test, never bridged.
  - regime split reported (Reasoning Standard #2): winter/shoulder
    (sun_elev < 50 deg, reach >= ~1 px) vs summer (subpixel) — the workup
    predicts the signal, if any, lives in the first group.

Run (session-side; deps numpy + xlrd like the v1 script):
  python3 scripts/tankfill_gate1.py
"""
import json
import math
import os
import tempfile
import urllib.request
from datetime import date

BASE = os.path.join(os.path.dirname(__file__), "..")
READINGS = os.path.join(BASE, "datacore", "sentinel2", "readings_v2.jsonl")
EIA_XLS = "https://www.eia.gov/dnav/pet/hist_xls/W_EPC0_SAX_YCUOK_MBBLw.xls"
MATCH_DAYS = 3
MAX_DELTA_GAP_DAYS = 14
WINTER_ELEV_MAX = 50.0

N_MIN = 20
DELTA_R_MIN = 0.3
SIGN_HIT_MIN = 0.65
LEVELS_R_SUPPORT = 0.5


# ── pure pieces (unit-tested) ───────────────────────────────────────────────

def scene_composite(reading):
    """D^2-weighted mean fill across all measured tanks, or None."""
    tanks = reading.get("tanks") or []
    if not tanks:
        return None
    wsum = sum(t["d_m"] ** 2 for t in tanks)
    return sum(t["fill"] * t["d_m"] ** 2 for t in tanks) / wsum


def match_weeks(readings, eia, match_days=MATCH_DAYS):
    """One row per EIA week: (eia_date, kbbl, scene_date, composite,
    sun_elev, cloud). Scene within +/-match_days of the Friday; when
    several scenes match one week, lowest cloud wins."""
    rows = {}
    eia_dates = sorted(eia)
    for r in readings:
        comp = scene_composite(r)
        if comp is None:
            continue
        d = date.fromisoformat(r["date"])
        best = None
        for k in eia_dates:
            gap = abs((date.fromisoformat(k) - d).days)
            if gap <= match_days and (best is None or gap < best[1]):
                best = (k, gap)
        if not best:
            continue
        k = best[0]
        cloud = r.get("cloud_pct")
        prev = rows.get(k)
        if prev is None or (cloud is not None and prev["cloud"] is not None and cloud < prev["cloud"]):
            rows[k] = {"eia_date": k, "kbbl": eia[k], "scene_date": r["date"],
                       "composite": comp, "sun_elev": r.get("sun_elev"), "cloud": cloud}
    return [rows[k] for k in sorted(rows)]


def paired_deltas(matched, max_gap_days=MAX_DELTA_GAP_DAYS):
    """(d_composite, d_kbbl) for successive matched weeks with gap <= cap."""
    out = []
    for a, b in zip(matched, matched[1:]):
        gap = (date.fromisoformat(b["eia_date"]) - date.fromisoformat(a["eia_date"])).days
        if gap <= max_gap_days:
            out.append((b["composite"] - a["composite"], b["kbbl"] - a["kbbl"]))
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sxy / (sx * sy)


def sign_hit_rate(deltas):
    """Share of week-pairs where composite delta and EIA delta agree in
    sign; zero deltas on either side are excluded (no free credit)."""
    scored = [(dc, dk) for dc, dk in deltas if dc != 0 and dk != 0]
    if not scored:
        return None, 0
    hits = sum(1 for dc, dk in scored if (dc > 0) == (dk > 0))
    return hits / len(scored), len(scored)


def has_reversal(matched):
    signs = []
    for a, b in zip(matched, matched[1:]):
        d = b["kbbl"] - a["kbbl"]
        if d != 0:
            signs.append(d > 0)
    return any(x != signs[0] for x in signs) if signs else False


def verdict(matched):
    deltas = paired_deltas(matched)
    dr = pearson([a for a, _ in deltas], [b for _, b in deltas])
    hit, n_scored = sign_hit_rate(deltas)
    lr = pearson([m["composite"] for m in matched], [m["kbbl"] for m in matched])
    checks = {
        "n_matched": len(matched),
        "n_matched_ok": len(matched) >= N_MIN,
        "reversal_ok": has_reversal(matched),
        "n_delta_pairs": len(deltas),
        "delta_r": None if dr is None else round(dr, 3),
        "delta_r_ok": dr is not None and dr >= DELTA_R_MIN,
        "sign_hit": None if hit is None else round(hit, 3),
        "sign_hit_n": n_scored,
        "sign_hit_ok": hit is not None and hit >= SIGN_HIT_MIN,
        "levels_r_supporting": None if lr is None else round(lr, 3),
    }
    checks["PASS"] = bool(checks["n_matched_ok"] and checks["reversal_ok"]
                          and checks["delta_r_ok"] and checks["sign_hit_ok"])
    return checks


# ── data loading ────────────────────────────────────────────────────────────

def load_readings(path=READINGS):
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def eia_weekly():
    import xlrd
    with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as tf:
        with urllib.request.urlopen(EIA_XLS, timeout=60) as r:
            tf.write(r.read())
        p = tf.name
    wb = xlrd.open_workbook(p)
    sh = wb.sheet_by_index(1)
    out = {}
    for i in range(sh.nrows):
        try:
            d = xlrd.xldate_as_datetime(sh.cell_value(i, 0), wb.datemode).date()
            out[d.isoformat()] = float(sh.cell_value(i, 1))
        except Exception:
            continue
    os.unlink(p)
    return out


def main():
    readings = load_readings()
    # sun angles ride on the chip index; merge cloud/sun onto readings by scene
    idx = {}
    with open(os.path.join(BASE, "datacore", "sentinel2", "chips_index.jsonl")) as fh:
        for line in fh:
            try:
                r = json.loads(line)
                idx[r["scene"]] = r
            except Exception:
                pass
    for r in readings:
        meta = idx.get(r.get("scene"), {})
        r.setdefault("cloud_pct", meta.get("cloud_pct"))

    eia = eia_weekly()
    matched = match_weeks(readings, eia)
    print(f"{len(readings)} readings, {len(eia)} EIA weeks -> {len(matched)} matched scene-weeks")
    for m in matched:
        print(f"  EIA {m['eia_date']} {int(m['kbbl'])} kbbl | scene {m['scene_date']} "
              f"composite {m['composite']:.4f} sun_elev {m['sun_elev']}")

    print("\n== ALL matched weeks ==")
    print(json.dumps(verdict(matched), indent=1))

    winter = [m for m in matched if (m["sun_elev"] or 90) < WINTER_ELEV_MAX]
    print(f"\n== winter/shoulder only (sun_elev < {WINTER_ELEV_MAX}, n={len(winter)}) ==")
    print(json.dumps(verdict(winter), indent=1))


if __name__ == "__main__":
    main()
