"""edge_p2_insider.py — gate-2 test of opportunistic insider purchases (P2).

Pins: the CMP classification (3-year calendar rule, unclassifiable excluded, never
reads its own year), event construction (value floor, same-day owners collapse),
no same-day entry, Newey-West, the matched-cell benchmark, the pre-registered
verdict rules, and the end-to-end run on a synthetic panel (a planted effect is
found, pure noise is not PASSED, routine inserts do not pass as opportunistic).
"""
import gzip
import json
import math
import os
import random

import numpy as np
import pytest

import edge_p2_insider as e2
import price_panel as pp
import walkforward_synth as ws


# ── classification ──────────────────────────────────────────────────────────
def test_classify_rules():
    routine = [(2014, 3), (2015, 3), (2016, 3), (2016, 9)]
    assert e2.classify(routine, 2017) == "routine"
    opp = [(2014, 3), (2015, 5), (2016, 8)]
    assert e2.classify(opp, 2017) == "opportunistic"
    assert e2.classify([(2015, 3), (2016, 3)], 2017) is None          # missing y-3 => unclassifiable
    assert e2.classify([], 2017) is None


def test_classify_never_reads_its_own_year_or_later():
    base = [(2014, 4), (2015, 7), (2016, 10)]
    assert e2.classify(base, 2017) == "opportunistic"
    # trades in 2017+ (even matching months) must not turn it routine
    assert e2.classify(base + [(2017, 4), (2018, 4)], 2017) == "opportunistic"


def _row(tkr, issuer, owner, code, filed, traded, value):
    return [tkr, issuer, owner, code, filed, traded, value]


def test_build_events_floor_collapse_and_exclusions():
    hist = [_row("AAA", "1", "o1", "S", f"{y}-{m:02d}-10", f"{y}-{m:02d}-08", 5000.0) for y, m in ((2014, 2), (2015, 5), (2016, 8))]
    rows = hist + [
        _row("AAA", "1", "o1", "P", "2017-06-12", "2017-06-09", 6000.0),      # same filing: two P rows sum to 11k
        _row("AAA", "1", "o1", "P", "2017-06-12", "2017-06-09", 5000.0),
        _row("AAA", "1", "o2", "P", "2017-06-12", "2017-06-09", 50000.0),     # o2 has no history: unclassifiable
        _row("BBB", "2", "o3", "P", "2017-06-12", "2017-06-09", 9000.0),      # below floor
    ]
    events, c = e2.build_events(rows)
    assert [(e["ticker"], e["filing_date"], e["cls"]) for e in events] == [("AAA", "2017-06-12", "opportunistic")]
    assert c["below_min_value"] == 1 and c["unclassifiable"] == 1 and c["opportunistic"] == 1


def test_entry_is_strictly_after_filing_day():
    cal = np.array([100, 101, 102, 105], dtype=np.int64)
    assert e2.next_day_after(cal, 101) == 102
    assert e2.next_day_after(cal, 103) == 105
    assert e2.next_day_after(cal, 105) is None


# ── statistics ──────────────────────────────────────────────────────────────
def test_newey_west_iid_matches_plain_t_and_penalizes_autocorrelation():
    rng = np.random.default_rng(1)
    x = rng.normal(0.01, 0.05, 400)
    mu, t = e2.newey_west_t(x, lag=3)
    plain = x.mean() / (x.std(ddof=0) / math.sqrt(len(x)))
    assert abs(t - plain) < 0.35 * abs(plain)
    ar = np.zeros(400)
    for i in range(1, 400):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 0.05)
    ar += 0.005
    _, t_ar = e2.newey_west_t(ar, lag=3)
    assert abs(t_ar) < abs(ar.mean() / (ar.std() / math.sqrt(400)))


def test_daygrid_expected_is_the_cell_mean_with_fallback():
    rng = np.random.default_rng(3)
    n = 3000
    dv = rng.lognormal(15, 1.5, n)
    mom = rng.normal(0.1, 0.4, n)
    f = np.where(mom > np.quantile(mom, 0.9), 0.05, 0.0) + rng.normal(0, 0.001, n)
    g = e2.DayGrid(dv, mom, {21: f})
    hi_mom = float(np.quantile(mom, 0.97))
    mid_dv = float(np.median(dv))
    got = g.expected(mid_dv, hi_mom, 21)
    # recompute the same cell by hand: decile of mid_dv, then top momentum tercile within that decile
    dec = np.searchsorted(g.dec_edges, dv)
    d = int(np.searchsorted(g.dec_edges, mid_dv))
    m = dec == d
    t_edges = np.quantile(mom[m], [1 / 3, 2 / 3])
    cell = m & (np.searchsorted(t_edges, mom) == 2)
    assert abs(got - float(f[cell].mean())) < 1e-9
    assert got > g.day_mean[21] + 0.005                       # a high-momentum peer group, not the day mean
    # a thin cell falls back to the decile mean rather than NaN
    tiny = e2.DayGrid(dv[:40], mom[:40], {21: f[:40]})
    assert math.isfinite(tiny.expected(float(np.median(dv[:40])), 0.0, 21))


def test_verdict_rules_in_preregistered_order():
    good = {"mean_bp": 60.0, "t_nw3": 3.4, "mde_bp": 30.0}
    assert e2.verdict(good, {"mean_bp": 5.0}, {"mean_bp": 40.0})["verdict"] == "PASS_PROVISIONAL"
    assert e2.verdict(good, {"mean_bp": 5.0}, {"mean_bp": -3.0})["rule"] == "a"       # survivor panel only flatters
    assert e2.verdict(good, {"mean_bp": 70.0}, {"mean_bp": 40.0})["rule"] == "c"      # routine as strong
    weak_powered = {"mean_bp": 10.0, "t_nw3": 0.6, "mde_bp": 25.0}
    assert e2.verdict(weak_powered, {"mean_bp": 1.0}, {"mean_bp": 5.0})["rule"] == "b"
    weak_underpowered = {"mean_bp": 10.0, "t_nw3": 0.6, "mde_bp": 80.0}
    assert e2.verdict(weak_underpowered, {"mean_bp": 1.0}, {"mean_bp": 5.0})["verdict"] == "INCONCLUSIVE"
    assert e2.verdict({"mean_bp": None}, {}, {})["verdict"] == "INCONCLUSIVE"


# ── end to end on a synthetic panel ─────────────────────────────────────────
def _make_world(tmp_path, effect, seed=11, n_names=140):
    """Panel 2016-2019 + a Form 4 cache. Each month 8 opportunistic and 8 routine
    insiders buy random names; `effect` = fractional jump ramped over the 21 days
    after entry for OPPORTUNISTIC buys only."""
    rng = random.Random(seed)
    days = ws.bdays("2016-01-04", "2019-12-31")
    base = {}
    lines = [ws.line("SPY", ws.path_rows(days, 200.0, 0.0003, 0.01, 8e7, rng))]
    for i in range(n_names):
        base[f"S{i:03d}"] = ws.path_rows(days, rng.uniform(15, 120), 0.0, 0.02, rng.uniform(2e6, 9e6), rng)
    idx = {d: k for k, d in enumerate(days)}
    insiders = []                      # (symbol, owner, cls, filing_date)
    syms = sorted(base)
    months = [f"{y}-{m:02d}" for y in (2017, 2018, 2019) for m in range(1, 13) if f"{y}-{m:02d}" >= "2017-03"]
    for mo in months:
        pool = [d for d in days if d.startswith(mo)]
        for cls in ("opportunistic", "routine"):
            for _ in range(8):
                sym = rng.choice(syms)
                filed = pool[rng.randrange(3, len(pool) - 3)]
                insiders.append((sym, f"own{len(insiders)}", cls, filed))
    for sym, owner, cls, filed in insiders:
        if cls != "opportunistic" or not effect:
            continue
        i0 = idx[filed] + 1
        rows = base[sym]
        for j in range(i0, len(rows)):
            f = 1.0 + effect * min(1.0, (j - i0) / 21.0)
            for k in ("o", "h", "l", "c", "rc"):
                rows[j][k] = round(rows[j][k] * f, 4)
    for sym in syms:
        lines.append(ws.line(sym, base[sym]))
    root = str(tmp_path / "panel")
    ws.write_panel(root, lines, as_of="2019-12-31", build_id="b20191231")
    f4 = tmp_path / "f4"
    f4.mkdir()
    rowsq = {}
    def add(q, r):
        rowsq.setdefault(q, []).append(r)
    def qof(d):
        return f"{d[:4]}q{(int(d[5:7]) - 1) // 3 + 1}"
    for sym, owner, cls, filed in insiders:
        y = int(filed[:4]); mo = int(filed[5:7])
        issuer = "cik" + sym
        # three prior years of history: same month (routine) or three different months (opportunistic)
        for k, yy in enumerate((y - 3, y - 2, y - 1)):
            m = mo if cls == "routine" else ((mo + 3 * (k + 1) - 1) % 12) + 1
            d = f"{yy}-{m:02d}-10"
            add(qof(d), _row(sym, issuer, owner, "S", d, d, 4000.0))
        add(qof(filed), _row(sym, issuer, owner, "P", filed, filed, 25000.0))
    for q in e2.quarters_between("2014q1", "2019q4"):
        with gzip.open(f4 / f"{q}.jsonl.gz", "wt") as fh:
            for r in rowsq.get(q, []):
                fh.write(json.dumps(r) + "\n")
    return root, str(f4)


def _run(monkeypatch, tmp_path, root, f4):
    monkeypatch.setattr(e2, "ensure_form4_cache", lambda d, first=e2.HISTORY_FROM, sleep_s=0: (e2.quarters_between("2014q1", "2019q4"), []))
    out = tmp_path / "out"
    res = e2.run(root, str(out), f4, 1500.0, "test")
    return res, json.loads((out / "latest.json").read_text())


def test_planted_opportunistic_effect_is_found_and_routine_is_not(monkeypatch, tmp_path):
    root, f4 = _make_world(tmp_path, effect=0.06)
    res, doc = _run(monkeypatch, tmp_path, root, f4)
    assert res["status"] == "ok"
    pit = doc["results"]["pit"]
    assert pit["opportunistic_21d_matched"]["mean_bp"] > 250          # ~+6% planted, cohort-averaged
    assert pit["opportunistic_21d_matched"]["t_nw3"] > e2.T_THRESHOLD
    assert abs(pit["routine_21d_matched"]["mean_bp"]) < 150            # no effect planted on routine
    assert doc["verdict"] == "PASS_PROVISIONAL"
    # output contract: derived statistics only — no ticker/price/owner leaves latest.json
    blob = json.dumps(doc)
    assert "S000" not in blob and "own" not in blob and "cikS" not in blob
    spot = json.loads((tmp_path / "out" / "label_spotcheck.json").read_text())
    assert spot["n"] == 25 and {r["class"] for r in spot["rows"]} <= {"routine", "opportunistic"}


def test_pure_noise_is_never_passed(monkeypatch, tmp_path):
    root, f4 = _make_world(tmp_path, effect=0.0, seed=5)
    res, doc = _run(monkeypatch, tmp_path, root, f4)
    assert res["status"] == "ok"
    assert doc["verdict"] != "PASS_PROVISIONAL"
    t = doc["results"]["pit"]["opportunistic_21d_matched"]["t_nw3"]
    assert t is None or abs(t) < e2.T_THRESHOLD


def test_run_degrades_cleanly(monkeypatch, tmp_path):
    empty = str(tmp_path / "nopanel")
    os.makedirs(empty)
    assert e2.run(empty, str(tmp_path / "o"), str(tmp_path / "f"), 1500.0, "t")["status"] == "skipped"
    root, f4 = _make_world(tmp_path, effect=0.0, seed=2, n_names=30)
    monkeypatch.setattr(e2, "ensure_form4_cache", lambda d, first=e2.HISTORY_FROM, sleep_s=0: (["2019q1"], ["2014q1"]))
    r = e2.run(root, str(tmp_path / "o2"), f4, 1500.0, "t")
    assert r["status"] == "insufficient" and "history incomplete" in r["reason"]
    assert not os.path.exists(tmp_path / "o2" / "latest.json")        # last good result is never overwritten by a bad run


def test_memory_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(e2, "_rss_mb", lambda: 5000.0)
    with pytest.raises(e2.MemoryBudgetExceeded):
        e2._check_rss(1500.0, "x")
