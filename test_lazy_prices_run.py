"""lazy_prices_run: skip paths never error and never write latest.json; the
payload carries no identifiers."""
import json
import os

import lazy_prices_run as lr


def test_skips_without_panel(tmp_path):
    out = lr.run(str(tmp_path / "o"), str(tmp_path / "p"), str(tmp_path / "a"), str(tmp_path / "nopanel"), "1", 500)
    assert out == {"status": "skipped", "reason": "no complete price-panel build"}
    assert not os.path.exists(tmp_path / "o" / "latest.json")


def test_skips_without_backfill_rows(tmp_path, monkeypatch):
    import price_panel as pp
    monkeypatch.setattr(pp, "current_build", lambda root=None: {"build_id": "b1", "dir": str(tmp_path)})
    out = lr.run(str(tmp_path / "o"), str(tmp_path / "p"), str(tmp_path / "nobackfill"), None, "1", 500)
    assert out["status"] == "skipped" and "backfill rows" in out["reason"]
    assert not os.path.exists(tmp_path / "o" / "latest.json")


def test_reads_backfill_rows_and_runs_gate2(tmp_path, monkeypatch):
    """The runner's input is the backfill's month files, not an archive walk."""
    import gzip
    import price_panel as pp
    import lazy_prices_panel as lpp
    import sec_issuer_pairs as sip
    monkeypatch.setattr(pp, "current_build", lambda root=None: {"build_id": "b1", "dir": str(tmp_path)})
    monkeypatch.setattr(sip, "ensure_pairs_cache", lambda d: ([], []))
    monkeypatch.setattr(sip, "rows_from_pairs_cache", lambda d: iter([("AAA", "1", "2017-01-05")]))
    seen = {}

    def fake_gate2(rows, cm, build_dir, variants=1, max_rss_mb=0):
        seen["rows"] = rows
        return {"summary": {}, "verdict": {"result": "INSUFFICIENT", "why": "x"}, "variants": 1, "n_obs": 0,
                "drops": {}, "delisted_exits": 0, "join": {}, "panel": {}, "caveat": "c"}
    monkeypatch.setattr(lpp, "run_gate2", fake_gate2)
    bdir = tmp_path / "backfill"
    (bdir / "rows").mkdir(parents=True)
    row = {"cik": "0000000001", "accession": "a-1", "form": "10-K", "accepted_at": "2017-02-01T12:00:00Z",
           "filing_date": "2017-02-01", "sections": {"1A": {"cosine_tfidf": 0.9}}}
    with gzip.open(bdir / "rows" / "2017-02.jsonl.gz", "wt") as f:
        f.write(json.dumps(row) + "\n")
    _whole_plan_scored(bdir)
    out = lr.run(str(tmp_path / "o"), str(tmp_path / "p"), str(bdir), None, "1", 500)
    assert out["status"] == "ok" and seen["rows"] == [row]
    latest = json.load(open(tmp_path / "o" / "latest.json"))
    assert latest["rows_scored"] == 1 and latest["backfill"]["first_month"] == "2017-02"
    assert latest["backfill"]["coverage"]["complete"] is True and latest["backfill"]["coverage"]["pending"] == 0
    assert "0000000001" not in json.dumps(latest)       # derived statistics only, no identifiers


def _write_plan(bdir, filings, complete=True, since="2015-01-01"):
    import lazy_prices_backfill as lpb
    lpb._write_jsonl_gz(str(bdir / "plan.jsonl.gz"), [{"cik": "0000000001", "accession": a, "form": "10-K", "filingDate": d}
                                                     for a, d in filings])
    lpb._write_json(str(bdir / "plan_meta.json"), {"since": since, "complete": complete, "filings": len(filings),
                                                   "completed_at": "2017-03-01T00:00:00+00:00" if complete else None})


def _whole_plan_scored(bdir):
    """A finished listing of one filing and a checkpoint whose cursor closed its day."""
    import lazy_prices_backfill as lpb
    _write_plan(bdir, [("a-1", "2017-02-01")])
    lpb._write_json(str(bdir / "checkpoint.json"), {"schema": "lazy_prices_backfill/1", "since": "2015-01-01",
                                                    "cursor_day": "2017-02-01", "closed": True})


def _rows_and_gate2_spy(tmp_path, monkeypatch):
    import gzip
    import price_panel as pp
    import lazy_prices_panel as lpp
    monkeypatch.setattr(pp, "current_build", lambda root=None: {"build_id": "b1", "dir": str(tmp_path)})
    called = []
    monkeypatch.setattr(lpp, "run_gate2", lambda *a, **k: called.append(1))
    bdir = tmp_path / "backfill"
    (bdir / "rows").mkdir(parents=True)
    with gzip.open(bdir / "rows" / "2017-02.jsonl.gz", "wt") as f:
        f.write(json.dumps({"accession": "a-1", "filing_date": "2017-02-01"}) + "\n")
    return bdir, called


def test_no_verdict_while_the_backfill_cursor_is_short_of_the_plan(tmp_path, monkeypatch):
    """REGRESSION (production 2026-11-15, the first Sunday with rows): 51,016 rows
    through cursor 2024-10-24 with 10,054 plan filings still to score. The runner
    would have stamped PASS/FAIL on a sample missing the last two years, and read
    it again a week later on the full one. It must skip, say why, and never call
    the harness or write latest.json."""
    bdir, called = _rows_and_gate2_spy(tmp_path, monkeypatch)
    import lazy_prices_backfill as lpb
    _write_plan(bdir, [("a-1", "2017-02-01"), ("a-2", "2024-10-24"), ("a-3", "2025-03-01"), ("a-4", "2026-02-27")])
    lpb._write_json(str(bdir / "checkpoint.json"), {"schema": "lazy_prices_backfill/1", "since": "2015-01-01",
                                                    "cursor_day": "2024-10-24", "closed": True})
    out = lr.run(str(tmp_path / "o"), str(tmp_path / "p"), str(bdir), None, "1", 500)
    assert out["status"] == "skipped" and out["reason"].startswith("backfill incomplete: 2 of 4 plan filings not scored yet")
    assert "2025-03-01" in out["reason"] and "2026-02-27" in out["reason"]
    assert out["rows_so_far"] == 1 and out["coverage"]["pending"] == 2
    assert called == [] and not os.path.exists(tmp_path / "o" / "latest.json")
    assert not os.path.exists(tmp_path / "o" / "last_run.json")     # a skip is not a run of record
    # the same volume once the cursor has closed the plan's last day: the read happens
    lpb._write_json(str(bdir / "checkpoint.json"), {"schema": "lazy_prices_backfill/1", "since": "2015-01-01",
                                                    "cursor_day": "2026-02-27", "closed": True})
    assert lpb.coverage(str(bdir))["complete"] is True


def test_no_verdict_without_a_finished_plan_listing(tmp_path, monkeypatch):
    """Rows but no plan (or a re-listing in progress): completeness is unknown, so it is not assumed."""
    bdir, called = _rows_and_gate2_spy(tmp_path, monkeypatch)
    out = lr.run(str(tmp_path / "o"), str(tmp_path / "p"), str(bdir), None, "1", 500)
    assert out["status"] == "skipped" and "plan listing is not complete" in out["reason"]
    _write_plan(bdir, [("a-1", "2017-02-01")], complete=False)
    out = lr.run(str(tmp_path / "o"), str(tmp_path / "p"), str(bdir), None, "1", 500)
    assert out["status"] == "skipped" and "plan listing is not complete" in out["reason"]
    assert called == [] and not os.path.exists(tmp_path / "o" / "latest.json")


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
