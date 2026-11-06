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
    ch = tmp_path / "optionchains"
    (ch / "oi").mkdir(parents=True)
    (tmp_path / "squeeze_gex").mkdir()
    for d in ("2026-10-29", "2026-10-30"):
        (ch / f"{d}.jsonl").write_text(json.dumps(_row(C)) + "\n")
    (ch / "oi" / "2026-10-29.jsonl").write_text(json.dumps({"s": C, "u": "SPY", "oi": 10}) + "\n")
    with gzip.open(tmp_path / "squeeze_gex" / "2026.jsonl.gz", "wt") as f:
        f.write(json.dumps({"date": "2026-10-29", "gex": 2e9}) + "\n")
    own, sq, st = g1.load(str(tmp_path))
    assert list(own) == ["2026-10-29"] and own["2026-10-29"] > 0 and sq == {"2026-10-29": 2e9}
    assert g1.compare(own, sq)["overlap_days"] == 1
