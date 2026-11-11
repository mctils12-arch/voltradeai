# Tests for scripts/edge_p1_gate1.py (Edge P1 ladder gate 1). Synthetic only.
# Load-bearing: dropped rows are counted never zero-filled; the verdict is
# PENDING under 40 overlapping days; KILL only above 30% sign disagreement.
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import edge_p1_gate1 as g1  # noqa: E402


def _row(sym, gamma=0.01, spot=500.0, iv=0.2, u="SPY"):
    return {"s": sym, "u": u, "gamma": gamma, "spot": spot, "iv": iv}


C = "SPY261120C00500000"
P = "SPY261120P00500000"


def test_parse_occ():
    assert g1.parse_occ(C) == ("SPY", "261120", "C", 500.0)
    assert g1.parse_occ("garbage") is None


def test_own_gex_calls_positive_puts_negative():
    chain = [_row(C), _row(P)]
    oi = [{"s": C, "u": "SPY", "oi": 100}, {"s": P, "u": "SPY", "oi": 300}]
    gex, st = g1.own_gex(chain, oi, "2026-10-29")
    per = 0.01 * 100 * 100 * 500.0 ** 2 * 0.01
    assert abs(gex - (per - 3 * per)) < 1e-6 and gex < 0 and st["used"] == 2


def test_missing_inputs_are_dropped_and_counted_not_zero_filled():
    chain = [_row(C), _row(P), _row("SPY261120C00510000"), _row("SPY261120C00520000", spot=None),
             _row("SPY261120C00530000", gamma=None, iv=None), _row("junk")]
    oi = [{"s": C, "u": "SPY", "oi": 10}, {"s": P, "u": "SPY", "oi": None},
          {"s": "SPY261120C00520000", "u": "SPY", "oi": 5}, {"s": "SPY261120C00530000", "u": "SPY", "oi": 5}]
    gex, st = g1.own_gex(chain, oi, "2026-10-29")
    assert st["used"] == 1 and st["no_oi"] == 2 and st["no_spot"] == 1 and st["no_gamma"] == 1 and st["bad_symbol"] == 1
    assert gex > 0
    assert g1.own_gex([_row(C)], [], "2026-10-29")[0] is None


def test_bs_gamma_fallback_matches_known_value():
    # ATM, T=1y, iv=20%, r=q=0: gamma = phi(0.1)/(S*0.2)
    got = g1.bs_gamma(100.0, 100.0, 1.0, 0.2)
    import math
    assert abs(got - math.exp(-0.005) / math.sqrt(2 * math.pi) / 20.0) < 1e-12
    assert g1.bs_gamma(100.0, 100.0, 0.0, 0.2) is None


def _series(n, agree):
    days = [f"2026-11-{i + 1:02d}" if i < 28 else f"2026-12-{i - 27:02d}" for i in range(n)]
    sq = {d: 1.0 + i for i, d in enumerate(days)}
    own = {d: (1.0 + i if i < agree else -1.0 - i) for i, d in enumerate(days)}
    return own, sq


def test_verdict_pending_below_min_days():
    own, sq = _series(39, 39)
    assert g1.compare(own, sq)["verdict"] == "PENDING"


def test_verdict_pass_and_kill_boundary():
    own, sq = _series(50, 50)
    assert g1.compare(own, sq)["verdict"] == "PASS"
    own, sq = _series(50, 35)          # exactly 15/50 = 30% disagree -> not > 30%
    assert g1.compare(own, sq)["verdict"] == "PASS"
    own, sq = _series(50, 34)          # 16/50 = 32% -> KILL
    r = g1.compare(own, sq)
    assert r["verdict"] == "KILL" and r["sign_disagree_days"] == 16


def test_none_own_days_excluded_and_spearman_perfect():
    own = {"a": 1.0, "b": None, "c": 3.0, "d": 4.0}
    sq = {"a": 10.0, "b": 5.0, "c": 30.0, "d": 40.0}
    r = g1.compare(own, sq)
    assert r["overlap_days"] == 3 and abs(r["rank_corr_levels"] - 1.0) < 1e-12


def test_load_end_to_end_with_gz_and_missing_oi_day(tmp_path):
    # 2026-11-06: SqueezeMetrics' side is the live file's own history (no squeeze_gex archive any more)
    ch = tmp_path / "optionchains"
    (ch / "oi").mkdir(parents=True)
    for d in ("2026-10-29", "2026-10-30"):
        (ch / f"{d}.jsonl").write_text(json.dumps(_row(C)) + "\n")
    with gzip.open(ch / "oi" / "2026-10-29.jsonl.gz", "wt") as f:
        f.write(json.dumps({"s": C, "u": "SPY", "oi": 10}) + "\n")
    squeeze = g1.parse_dix_gex("date,price,dix,gex\n2026-10-29,7750,0.4,2000000000\n")
    own, sq, st = g1.load(str(tmp_path), squeeze)
    assert list(own) == ["2026-10-29"] and own["2026-10-29"] > 0 and sq == {"2026-10-29": 2e9}
    assert g1.compare(own, sq)["overlap_days"] == 1


def test_parse_dix_gex_follows_the_servers_rules():
    text = "\n".join([
        "date,price,dix,gex",
        "2026-10-27,7722.6,0.4549768975156808,5515099082.650005",
        "2026-10-28,7743.98,0.46105040243405043,5605140552.230006",
        "garbage,1,2,3",                      # not a date
        "2026-10-29,7750,0.4,notanumber",     # not a number
        "2026-10-30,7750,inf,1",              # not finite
        "2026-10-31,1,0.4",                   # too few columns
        "2026-10-28,7743.98,0.46,-1.5",       # a repeated date keeps the last row
    ]) + "\n"
    g1.MALFORMED_DIX_ROWS[0] = 0
    assert g1.parse_dix_gex(text) == {"2026-10-27": 5515099082.650005, "2026-10-28": -1.5}
    assert g1.MALFORMED_DIX_ROWS[0] == 4, "every skipped row is counted (and reported by main)"
    assert g1.parse_dix_gex("nope\n2026-10-27,1,2,3\n") == {}
    assert g1.parse_dix_gex("") == {}


def test_main_reads_the_live_file_in_memory_and_writes_nothing(tmp_path, monkeypatch, capsys):
    ch = tmp_path / "archive" / "optionchains"
    (ch / "oi").mkdir(parents=True)
    (ch / "2026-10-29.jsonl").write_text(json.dumps(_row(C)) + "\n")
    (ch / "oi" / "2026-10-29.jsonl").write_text(json.dumps({"s": C, "u": "SPY", "oi": 10}) + "\n")
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"date,price,dix,gex\n2026-10-29,7750,0.4,2000000000\n"

    def fake_urlopen(req, timeout):
        seen["url"], seen["agent"], seen["timeout"] = req.full_url, req.get_header("User-agent"), timeout
        return _Resp()

    monkeypatch.setattr(g1.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(g1, "MALFORMED_DIX_ROWS", [0])
    monkeypatch.chdir(tmp_path)
    assert g1.main(["--archive-dir", str(tmp_path / "archive"), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert seen["url"] == g1.DIX_URL == "https://squeezemetrics.com/monitor/static/DIX.csv"
    assert seen["agent"].startswith("voltradeai-datacore/1.0") and seen["timeout"] == 30
    assert rep["overlap_days"] == 1 and rep["squeeze_rows"] == 1 and rep["squeeze_last_day"] == "2026-10-29"
    assert rep["squeeze_source"] == g1.DIX_URL and rep["malformed_dix_rows"] == 0
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before, "nothing written: no copy of their file anywhere"
    # --dix PATH: a local copy, no fetch
    local = tmp_path / "DIX.csv"
    local.write_text("date,price,dix,gex\n2026-10-29,7750,0.4,-3\n")
    monkeypatch.setattr(g1.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    assert g1.main(["--archive-dir", str(tmp_path / "archive"), "--dix", str(local), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["squeeze_source"] == str(local) and rep["sign_disagree_days"] == 1
