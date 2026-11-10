"""lazy_prices_run: skip paths never error and never write latest.json; the
payload carries no identifiers."""
import json
import os

import lazy_prices_run as lr


def test_skips_without_panel(tmp_path):
    out = lr.run(str(tmp_path / "o"), str(tmp_path / "p"), str(tmp_path / "a"), str(tmp_path / "nopanel"), "1", 500)
    assert out == {"status": "skipped", "reason": "no complete price-panel build"}
    assert not os.path.exists(tmp_path / "o" / "latest.json")


def test_skips_without_archive(tmp_path, monkeypatch):
    import price_panel as pp
    monkeypatch.setattr(pp, "current_build", lambda root=None: {"build_id": "b1", "dir": str(tmp_path)})
    out = lr.run(str(tmp_path / "o"), str(tmp_path / "p"), str(tmp_path / "noarchive"), None, "1", 500)
    assert out["status"] == "skipped" and "archive" in out["reason"]


def test_summarize_has_no_identifiers():
    res = {"summary": {"in_sample": {"cohorts": 3, "mean_net": 0.001, "t_nw_net": 1.0, "q_excess_mean": [0.0]},
                       "out_of_sample": {"cohorts": 2, "mean_net": None, "t_nw_net": None, "q_excess_mean": None}},
           "verdict": {"result": "INSUFFICIENT", "why": "x"}, "variants": 1, "n_obs": 10, "drops": {"a": 1},
           "delisted_exits": 0, "join": {"dropped": 2}, "panel": {"symbols_in_panel": 5}, "caveat": "c"}
    p = lr.summarize(res, "b1", "1.0.0", {"rows_scored": 4})
    s = json.dumps(p)
    assert p["verdict"]["result"] == "INSUFFICIENT" and p["panel"] == {"build_id": "b1"}
    assert "cik" not in s.lower() and "ticker" not in s.lower()


def test_main_error_is_one_json_line_exit1(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(lr, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    rc = lr.main(["--run", "--out-dir", str(tmp_path), "--pairs-dir", str(tmp_path / "p")])
    assert rc == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == "error"
    assert json.load(open(tmp_path / "last_run.json"))["status"] == "error"
