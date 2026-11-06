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


def _res(day, ours, theirs):
    return {"day": day, "spearman": None,
            "pairs": [["U%d" % i, o, t, 10, 10] for i, (o, t) in enumerate(zip(ours, theirs))]}


def test_pool_applies_preregistered_kill_rule():
    xs = list(range(40))
    good = m.pool_results([_res("d1", xs[:20], xs[:20]), _res("d2", xs[20:], xs[20:])])
    assert good["n"] == 40 and good["pooled_spearman"] == 1.0
    assert good["verdict"] == "PASSES_GATE_1_CLAUSE" and good["datasetKill"] is False
    bad = m.pool_results([_res("d1", xs, xs[::-1])])
    assert bad["verdict"] == "KILLED_AT_GATE_1" and bad["datasetKill"] is True
    assert m.pool_results([_res("d1", xs[:5], xs[:5])])["verdict"] == "insufficient_n"


def test_pool_matches_recorded_first_day():
    import json
    d = json.load(open(os.path.join(os.path.dirname(__file__), "research", "p3_realquote_gate1_2026-11-02.json")))
    r = m.pool_results([d])
    assert abs(r["pooled_spearman"] - d["spearman"]) < 1e-3 and r["datasetKill"] is True


def test_too_early_guard_blocks_pre_close_runs():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    assert m.too_early("2026-11-03", datetime(2026, 11, 3, 13, 10, tzinfo=et))
    assert m.too_early("2026-11-03", datetime(2026, 11, 3, 16, 29, tzinfo=et))
    assert not m.too_early("2026-11-03", datetime(2026, 11, 3, 16, 30, tzinfo=et))
    assert not m.too_early("2026-11-03", datetime(2026, 11, 4, 8, 0, tzinfo=et))


def test_fetch_chain_vs_sends_token_and_rejects_empty_day():
    import io, json, pytest
    seen = {}

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def opener(req, timeout=0):
        seen["tok"] = req.get_header("X-diag-token"); seen["url"] = req.full_url
        return R(json.dumps({"rows": [{"day": "2026-11-03", "u": "A"}]}).encode())
    body = m.fetch_chain_vs("2026-11-03", "tkn", opener)
    assert seen["tok"] == "tkn" and seen["url"].endswith("chain_vs?day=2026-11-03") and body["rows"]
    with pytest.raises(SystemExit):
        m.fetch_chain_vs("2026-11-04", "tkn", opener)
    with pytest.raises(SystemExit):
        m.fetch_chain_vs("2026-11-03", "", opener)
