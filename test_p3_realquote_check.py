import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import p3_realquote_check as m


def opt(sym, exp, cp, k, iv, delta=None):
    return {"option": f"{sym}{exp}{cp}{k:08d}", "iv": iv, "delta": delta}


def test_vs_matches_chainvs_definition():
    day = "2026-11-02"  # 14..60 DTE => expiries 2026-11-16 .. 2027-01-01
    o = []
    for k in (100000, 105000, 110000):
        o.append(opt("X", "261218", "C", k, 0.50, 0.5))
        o.append(opt("X", "261218", "P", k, 0.40))
    # excluded: too near (6 DTE), call delta out of band, zero iv put
    o.append(opt("X", "261108", "C", 100000, 0.9, 0.5)); o.append(opt("X", "261108", "P", 100000, 0.1))
    o.append(opt("X", "261218", "C", 120000, 0.9, 0.95)); o.append(opt("X", "261218", "P", 120000, 0.1))
    o.append(opt("X", "261218", "C", 125000, 0.9, 0.5)); o.append(opt("X", "261218", "P", 125000, 0, None))
    v, n = m.vs_from_options(o, day)
    assert n == 3 and abs(v - 0.10) < 1e-9


def test_vs_needs_min_pairs():
    o = [opt("X", "261218", "C", 100000, 0.5, 0.5), opt("X", "261218", "P", 100000, 0.4)]
    assert m.vs_from_options(o, "2026-11-02") == (None, 1)


def test_spearman_and_ties():
    assert abs(m.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    assert abs(m.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    assert m.ranks([5, 5, 1]) == [2.5, 2.5, 1.0]


def test_et_day_uses_eastern_date_not_utc():
    assert m.et_day("2026-11-03 00:10:31") == "2026-11-02"
    assert m.et_day("2026-11-03 06:00:00") == "2026-11-03"
