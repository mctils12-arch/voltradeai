"""REPAIR 2026-11-18: the nightly Lazy Prices backfill ended one second into
the night of 2026-11-17 with "HTTPError: 429 Client Error: Too Many Requests
for url: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
(heartbeat lazy_prices_backfill_nightly, 03:13:36Z, two minutes after a deploy).

The CIK list is an input of the PLAN only. run_once fetched it on every slice
(a fresh container re-downloads ~50 quarter files for it) and let a failure of
that fetch raise, against its own docstring ("never raises for a prerequisite
problem"). The slice is once per night per process, so the night was lost.
"""

import gzip
import json
import os

import pytest

import lazy_prices_backfill as bf
import sec_issuer_pairs as sip
from test_lazy_prices_backfill import _fetch_from, _universe

SINCE = "2015-01-01"


class Http429(Exception):
    pass


def _boom(_pairs_dir):
    raise Http429("429 Client Error: Too Many Requests for url: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets")


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "free_bytes", lambda p: None)
    docs, plan = _universe(n_ciks=2, years=(2015, 2016))
    rows_by_cik = {}
    for r in plan:
        rows_by_cik.setdefault(str(int(r["cik"])), []).append(dict(r))
    monkeypatch.setattr(bf, "make_fetcher", lambda client, tmp: _fetch_from(docs))
    root = str(tmp_path / "root")
    listed = []

    def lf(c, s):
        listed.append(c)
        return rows_by_cik.get(c, [])

    def run(**kw):
        return bf.run_once(root, str(tmp_path / "pairs"), str(tmp_path / "t"), SINCE, 60, 500, "1",
                           client=object(), list_filings=lf, **kw)
    return root, run, listed


def _age_plan(root, days):
    path = bf._plan_paths(root)[1]
    meta = json.load(open(path))
    import datetime as dt
    meta["completed_at"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    json.dump(meta, open(path, "w"))


def test_a_fresh_plan_never_asks_for_the_cik_list(tmp_path, monkeypatch):
    root, run, listed = _setup(tmp_path, monkeypatch)
    assert run(ciks=["1", "2"])["status"] == "ok"
    monkeypatch.setattr(bf, "_ciks_from_pairs", _boom)           # would raise if it were called
    out = run()
    assert out["status"] == "ok" and out["phase"] == "score" and out["plan_built"] is False
    assert out["cik_source"] is None


def test_the_night_of_11_17_a_stale_plan_and_a_429_relists_from_the_plan_on_file(tmp_path, monkeypatch):
    root, run, listed = _setup(tmp_path, monkeypatch)
    assert run(ciks=["1", "2"])["status"] == "ok"
    _age_plan(root, bf.PLAN_STALE_DAYS + 1)
    del listed[:]
    monkeypatch.setattr(bf, "_ciks_from_pairs", _boom)
    out = run()                                                   # before the repair: Http429 raised out of run_once
    assert out["status"] == "ok" and out["plan_built"] is True
    assert sorted(listed) == ["1", "2"]                           # the plan was relisted from its own CIKs
    assert out["cik_source"].startswith("the plan of ") and "429" in out["cik_source"]


def test_a_working_fetch_is_saved_and_the_saved_list_is_the_first_fallback(tmp_path, monkeypatch):
    root, run, listed = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(bf, "_ciks_from_pairs", lambda d: (["1", "2"], 0))
    out = run()
    assert out["status"] == "ok" and out["cik_source"] == "pairs cache"
    saved = json.load(open(os.path.join(root, bf.CIK_LIST_FILE)))
    assert saved["ciks"] == ["1", "2"] and saved["saved_at"]
    # next week: the fetch fails; the plan on file is mid-rebuild (not complete), so only the saved list can serve
    meta_path = bf._plan_paths(root)[1]
    meta = json.load(open(meta_path)); meta["complete"] = False; meta["done_ciks"] = ["1"]; json.dump(meta, open(meta_path, "w"))
    monkeypatch.setattr(bf, "_ciks_from_pairs", _boom)
    ciks, source, problem = bf.resolve_ciks(root, "unused", json.load(open(meta_path)))
    assert ciks == ["1", "2"] and source.startswith("saved list of ") and "429" in problem
    out = run()
    assert out["status"] == "ok" and out["cik_source"].startswith("saved list of ")


def test_an_empty_map_also_falls_back_and_with_nothing_to_fall_back_on_the_slice_is_skipped_with_the_reason(tmp_path, monkeypatch):
    root, run, listed = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(bf, "_ciks_from_pairs", _boom)
    out = run()                                                   # no plan, no saved list
    assert out["status"] == "skipped"
    assert out["reason"].startswith("no CIK list for the plan: Http429: 429 Client Error")
    monkeypatch.setattr(bf, "_ciks_from_pairs", lambda d: ([], 3))
    out = run()
    assert out["status"] == "skipped" and "CIK map empty (3 pairs quarters failed)" in out["reason"]


def test_main_reports_a_skip_not_an_error_when_the_sec_page_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bf, "free_bytes", lambda p: None)
    monkeypatch.setattr(bf, "_ciks_from_pairs", _boom)
    rc = bf.main(["--run", "--root", str(tmp_path / "root"), "--pairs-dir", str(tmp_path / "pairs"),
                  "--tmp-archive", str(tmp_path / "t"), "--budget-seconds", "5"])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and line["status"] == "skipped" and "429" in line["reason"]


def test_pairs_cache_survives_a_failed_quarter_listing(tmp_path, monkeypatch):
    import sec_form4_bulk as sb
    d = str(tmp_path / "pairs")
    sip.write_pairs(d, "2024q1", [("AAPL", "320193", "2024-02-01")])
    sip.write_pairs(d, "2013q4", [("OLD", "1", "2013-11-01")])     # before `first`: never returned

    def refuse():
        raise Http429("429 Client Error: Too Many Requests")
    monkeypatch.setattr(sb, "list_available_quarters", refuse)
    cached, failed = sip.ensure_pairs_cache(d)                    # before the repair: raised
    assert cached == ["2024q1"]
    assert len(failed) == 1 and failed[0].startswith("listing: Http429: 429")
    assert list(sip.rows_from_pairs_cache(d))                     # what is cached is still readable
    # an empty or missing cache dir is an empty answer, not an exception
    assert sip.ensure_pairs_cache(str(tmp_path / "nope")) == ([], failed)
