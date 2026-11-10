"""Offline tests for lazy_prices_backfill.py: the streaming point-in-time scorer
must reproduce pit_rows_from_archive exactly, resume mid-day without changing a
single row, never let a filing see its own day, and skip (counted) anything
that arrives behind a closed day. No network: fetch and listing are injected."""
import gzip
import json
import math
import os

import pytest

import lazy_prices_backfill as bf
import sec_filings_similarity as sim
import sec_filings_text as sft

WORDS = ("revenue customers competition regulation supply inflation litigation cyber "
         "currency tariff interest liquidity covenant pension warranty recall").split()


@pytest.fixture(autouse=True)
def _no_rss_guard(monkeypatch):
    """peak_rss_mb reads the PROCESS's ru_maxrss. In production that is the job's
    own fresh Python; under pytest it is the whole suite's high-water mark, which
    passes 500 MB late in a full run and would trip the guard here for reasons
    that have nothing to do with the code under test."""
    monkeypatch.setattr(bf, "peak_rss_mb", lambda: 0.0)


def test_rss_guard_stops_cleanly_with_a_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "peak_rss_mb", lambda: 2000.0)
    docs, plan = _universe(n_ciks=1, years=(2015, 2016))
    out = bf.Backfill(str(tmp_path / "root"), str(tmp_path / "t"), _fetch_from(docs), since="2015-01-01", max_rss_mb=1500).run(plan)
    assert out["processed"] == 0 and out["stop_reason"].startswith("rss 2000 MB > 1500 MB")


def _sec(text):
    return {"found": True, "short": False, "text": text,
            "flags": {"no_material_changes": False, "not_applicable": False, "incorporated_by_reference": False}}


def _doc(cik, acc, fd, period, text, accepted=None):
    return {"cik": sft.cik10(cik), "accession": acc, "form": "10-K", "is_amendment": False, "filing_date": fd,
            "accepted_at": accepted or f"{fd}T21:00:00Z", "period": period,
            "sections": {"risk_factors": _sec(text), "mdna": _sec(text[::-1] + " " + text)}}


def _text(seed):
    ws = list(WORDS)
    ws[seed % len(ws)] = f"word{seed}"
    ws[(seed * 7) % len(ws)] = f"term{seed % 5}"
    return " ".join(ws * 6)


def _universe(n_ciks=4, years=(2014, 2015, 2016, 2017), same_day=False):
    """One 10-K per CIK per year. Distinct filing days unless same_day."""
    docs, plan = {}, []
    for ci in range(1, n_ciks + 1):
        for yi, y in enumerate(years):
            day = f"{y}-03-{(1 if same_day else ci):02d}"
            acc = f"{ci:010d}-{y % 100:02d}-000001"
            d = _doc(str(ci), acc, day, f"{y - 1}-12-31", _text(ci * 10 + yi))
            docs[acc] = d
            plan.append({"cik": sft.cik10(ci), "accession": acc, "form": "10-K", "filingDate": day,
                         "reportDate": f"{y - 1}-12-31", "acceptanceDateTime": None, "primaryDocument": "x.htm", "isAmendment": False})
    plan.sort(key=lambda r: (r["filingDate"], r["accession"]))
    return docs, plan


def _archive_of(docs, arch):
    for d in docs.values():
        p = sft.sections_path(arch, d["cik"], d["accession"])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        sft.write_once_gz(p, json.dumps(d).encode())


def _fetch_from(docs, calls=None):
    def fetch(filing):
        if calls is not None:
            calls.append(filing["accession"])
        return docs[filing["accession"]]
    return fetch


def _key(r):
    return r["accession"]


def test_pit_idf_state_roundtrip_and_add_delta_equals_sequential():
    seq = sim.PitIdf()
    docs = [set(sim.tokenize(_text(i))) for i in range(6)]
    for d in docs:
        seq.add(d)
    batch = sim.PitIdf()
    batch.add_delta({t: 1 for t in docs[0]}, 1)
    delta = {}
    for d in docs[1:]:
        for t in d:
            delta[t] = delta.get(t, 0) + 1
    batch.add_delta(delta, 5)
    assert batch.n == seq.n and batch.df == seq.df and batch._n_df1 == seq._n_df1
    back = sim.PitIdf.from_state(json.loads(json.dumps(batch.state())))
    assert back.n == seq.n and back.df == seq.df and back._n_df1 == seq._n_df1
    assert math.isclose(max(back.values()), max(seq.values()))


def test_streaming_rows_equal_pit_rows_from_archive_on_distinct_days(tmp_path):
    docs, plan = _universe()
    arch = str(tmp_path / "arch")
    _archive_of(docs, arch)
    ref = sorted(sim.pit_rows_from_archive(arch, min_corpus=0), key=_key)
    b = bf.Backfill(str(tmp_path / "root"), str(tmp_path / "tmp"), _fetch_from(docs), since="2014-01-01", min_corpus=0)
    out = b.run(plan)
    assert out["stop_reason"] == "plan exhausted" and out["counts"]["rows"] == len(ref) == 12
    got = sorted(bf.load_rows(str(tmp_path / "root")), key=_key)
    assert [r["accession"] for r in got] == [r["accession"] for r in ref]
    for g, r in zip(got, ref):
        assert g["prior_accession"] == r["prior_accession"] and g["idf_n"] == r["idf_n"]
        for k in ("risk_factors", "mdna"):
            a, e = g["sections"][k], r["sections"][k]
            assert a["excluded"] == e["excluded"] and math.isclose(a["cosine"], e["cosine"])
            if e["cosine_tfidf"] is None:
                assert a["cosine_tfidf"] is None
            else:
                assert math.isclose(a["cosine_tfidf"], e["cosine_tfidf"], rel_tol=1e-12)
    # 2014 filings have no prior (unpaired, corpus only); the first ROW is 2015 and sees exactly the four 2014 docs
    first = got[0]
    assert first["filing_date"] == "2015-03-01" and first["idf_n"]["risk_factors"] == 4
    assert first["sections"]["risk_factors"]["cosine_tfidf"] is not None and out["counts"]["unpaired"] == 4


def test_same_day_filings_never_see_each_other(tmp_path):
    docs, plan = _universe(n_ciks=3, same_day=True)
    b = bf.Backfill(str(tmp_path / "root"), str(tmp_path / "tmp"), _fetch_from(docs), since="2014-01-01", min_corpus=0)
    b.run(plan)
    rows = bf.load_rows(str(tmp_path / "root"))
    by_day = {}
    for r in rows:
        by_day.setdefault(r["filing_date"], set()).add(r["idf_n"]["risk_factors"])
    # every row of a day sees exactly the docs of the EARLIER days: 3 per earlier year, none of its own day
    assert by_day == {"2015-03-01": {3}, "2016-03-01": {6}, "2017-03-01": {9}}


def test_resume_mid_day_reproduces_the_uninterrupted_run_exactly(tmp_path):
    docs, plan = _universe(n_ciks=5, same_day=True)
    full_root = str(tmp_path / "full")
    bf.Backfill(full_root, str(tmp_path / "t1"), _fetch_from(docs), since="2014-01-01", min_corpus=0, chunk=2).run(plan)
    full = sorted(bf.load_rows(full_root), key=_key)
    root = str(tmp_path / "part")
    # three interrupted runs: 3 filings, then 4, then the rest (chunk 2 => checkpoints land mid-day)
    kills = [3, 4, None]
    outs = []
    for k in kills:
        b = bf.Backfill(root, str(tmp_path / "t2"), _fetch_from(docs), since="2014-01-01", min_corpus=0, chunk=2)
        outs.append(b.run(plan, max_filings=k))
    assert outs[0]["stop_reason"] == "max_filings" and outs[0]["cursor_closed"] is False
    assert outs[-1]["stop_reason"] == "plan exhausted"
    part = sorted(bf.load_rows(root), key=_key)
    assert json.dumps(part, sort_keys=True) == json.dumps(full, sort_keys=True)
    st_full, st_part = bf.load_pit_state(full_root), bf.load_pit_state(root)
    assert {k: v.state() for k, v in st_full.items()} == {k: v.state() for k, v in st_part.items()}
    # every filing processed exactly once across the three runs
    assert sum(o["processed"] for o in outs) == len(plan) == 20


def test_late_arrival_behind_a_closed_day_is_skipped_and_counted(tmp_path):
    docs, plan = _universe(n_ciks=2)
    root = str(tmp_path / "root")
    bf.Backfill(root, str(tmp_path / "t"), _fetch_from(docs), since="2014-01-01", min_corpus=0).run(plan)
    n_rows = len(bf.load_rows(root))
    # a third CIK appears in a refreshed plan with historical filings behind the cursor
    more_docs, more_plan = _universe(n_ciks=3)
    docs.update(more_docs)
    calls = []
    out = bf.Backfill(root, str(tmp_path / "t"), _fetch_from(docs, calls), since="2014-01-01", min_corpus=0).run(more_plan)
    # CIK 3 files on the 3rd of March: 2014-2016 land behind the closed cursor (2017-03-02) and are never scored
    # or added to the corpus; 2017-03-03 is ahead of it and IS scored, against its (unscored) 2016 prior
    assert out["processed"] == 1 and out["counts"]["behind_cursor_skipped"] == 3
    assert sorted(calls) == sorted([f"{3:010d}-17-000001", f"{3:010d}-16-000001"])
    rows = bf.load_rows(root)
    assert len(rows) == n_rows + 1
    new = [r for r in rows if r["accession"] == f"{3:010d}-17-000001"][0]
    assert new["idf_n"]["risk_factors"] == 8        # the 8 filings of CIKs 1-2 that were processed, none of CIK 3's


def test_open_tail_day_waits_for_a_newer_plan_then_closes(tmp_path):
    docs, plan = _universe(n_ciks=2, years=(2016, 2017))
    root = str(tmp_path / "root")
    # the plan was listed on 2017-03-02: day 2017-03-01/02 may not close yet (close_before = listing - 2 days)
    b = bf.Backfill(root, str(tmp_path / "t"), _fetch_from(docs), since="2016-01-01", min_corpus=0, close_before="2017-02-28")
    out = b.run(plan)
    assert out["stop_reason"] == "waiting for a plan listed after 2017-03-01 before closing it"
    assert out["cursor_day"] == "2017-03-01" and out["cursor_closed"] is False
    assert out["processed"] == 3                    # 2016 x2 + 2017-03-01; 2017-03-02 not started
    # a newer plan brings a late filing for the open day: it is scored, not skipped
    late = _doc("9", "0000000009-17-000001", "2017-03-01", "2016-12-31", _text(99))
    late_prior = _doc("9", "0000000009-16-000001", "2016-03-09", "2015-12-31", _text(98))
    docs.update({late["accession"]: late, late_prior["accession"]: late_prior})
    plan2 = plan + [{"cik": sft.cik10(9), "accession": a, "form": "10-K", "filingDate": d, "reportDate": p,
                     "acceptanceDateTime": None, "primaryDocument": "x.htm", "isAmendment": False}
                    for a, d, p in ((late["accession"], "2017-03-01", "2016-12-31"), (late_prior["accession"], "2016-03-09", "2015-12-31"))]
    plan2.sort(key=lambda r: (r["filingDate"], r["accession"]))
    out2 = bf.Backfill(root, str(tmp_path / "t"), _fetch_from(docs), since="2016-01-01", min_corpus=0, close_before="2017-03-10").run(plan2)
    assert out2["stop_reason"] == "plan exhausted" and out2["cursor_closed"] is True
    rows = {r["accession"]: r for r in bf.load_rows(root)}
    assert late["accession"] in rows
    # the 2016-03-09 prior arrived behind the closed 2016 days: counted, never scored or added to the corpus
    assert out2["counts"]["behind_cursor_skipped"] == 1
    assert rows[late["accession"]]["idf_n"]["risk_factors"] == 2       # the two 2016 filings that WERE processed


def test_failures_are_recorded_once_and_never_retried(tmp_path):
    docs, plan = _universe(n_ciks=2, years=(2015, 2016))
    bad = plan[1]["accession"]

    def fetch(filing):
        if filing["accession"] == bad:
            raise sft.EdgarHTTPError("u", 404, "gone")
        return docs[filing["accession"]]

    root = str(tmp_path / "root")
    out = bf.Backfill(root, str(tmp_path / "t"), fetch, since="2015-01-01", min_corpus=0).run(plan)
    assert out["counts"]["failed"] == 2        # the filing itself, and the 2016 filing whose prior it is
    ck = bf.load_checkpoint(root, "2015-01-01")
    assert bad in ck["failed"] and "404" in ck["failed"][bad]
    calls = []
    out2 = bf.Backfill(root, str(tmp_path / "t"), _fetch_from(docs, calls), since="2015-01-01", min_corpus=0).run(plan)
    assert out2["processed"] == 0 and calls == []


def test_budget_stop_checkpoints_cleanly(tmp_path):
    docs, plan = _universe(n_ciks=3)
    b = bf.Backfill(str(tmp_path / "root"), str(tmp_path / "t"), _fetch_from(docs), since="2014-01-01", min_corpus=0)
    out = b.run(plan, deadline=0.0)          # already expired
    assert out["processed"] == 0 and out["stop_reason"] == "budget"
    assert os.path.exists(tmp_path / "root" / "checkpoint.json")


def test_build_plan_resumes_drops_amendments_and_lists_priors_before_since(tmp_path):
    root = str(tmp_path / "root")
    listed = []

    def lf(cik, since):
        listed.append(cik)
        if cik == "7":
            raise RuntimeError("edgar down")
        return [{"cik": cik, "accession": f"{int(cik):010d}-15-1", "form": "10-K", "filingDate": "2015-03-01", "reportDate": "2014-12-31"},
                {"cik": cik, "accession": f"{int(cik):010d}-15-2", "form": "10-K/A", "filingDate": "2015-04-01", "isAmendment": True},
                {"cik": cik, "accession": f"{int(cik):010d}-14-1", "form": "10-K", "filingDate": "2014-03-01", "reportDate": "2013-12-31"}]

    meta = bf.build_plan(root, ["1", "2", "7", "3"], "2015-01-01", lf, deadline=0.0)
    assert meta["complete"] is False and listed == []                   # nothing listed: deadline already passed
    meta = bf.build_plan(root, ["1", "2", "7", "3"], "2015-01-01", lf, checkpoint_every=1)
    assert meta["complete"] is True and meta["list_errors"] == 1 and sorted(meta["done_ciks"]) == ["1", "2", "3", "7"]
    plan = bf.load_plan(root)
    assert len(plan) == 6 and all(not r["isAmendment"] for r in plan)
    assert plan[0]["filingDate"] == "2014-03-01"                        # prior-only rows kept for pairing
    assert meta["list_since"] == "2013-11-17"                           # since - (365 + 45) days
    # a complete plan is not re-listed by a plain call; reset=True starts over
    listed.clear()
    bf.build_plan(root, ["1", "2", "7", "3"], "2015-01-01", lf)
    assert listed == ["1", "2", "3", "7"]                               # complete => rebuilt from scratch (new filings join the tail)
    assert bf.plan_is_stale({"complete": True, "completed_at": "2026-01-01T00:00:00+00:00"}) is True
    assert bf.plan_is_stale({"complete": True, "completed_at": bf._now_iso()}) is False


def test_run_once_skips_on_volume_floor_and_reports_plan_phase(tmp_path, monkeypatch):
    root = str(tmp_path / "root")
    monkeypatch.setattr(bf, "free_bytes", lambda p: 10)
    out = bf.run_once(root, str(tmp_path / "pairs"), str(tmp_path / "t"), "2015-01-01", 60, 500, "1", ciks=["1"], client=object(),
                      list_filings=lambda c, s: [])
    assert out["status"] == "skipped" and "free" in out["reason"]
    monkeypatch.setattr(bf, "free_bytes", lambda p: None)
    docs, plan = _universe(n_ciks=2, years=(2015, 2016))
    rows_by_cik = {}
    for r in plan:
        rows_by_cik.setdefault(str(int(r["cik"])), []).append(dict(r))
    monkeypatch.setattr(bf, "make_fetcher", lambda client, tmp: _fetch_from(docs))
    out = bf.run_once(root, str(tmp_path / "pairs"), str(tmp_path / "t"), "2015-01-01", 60, 500, "1", ciks=["1", "2"], client=object(),
                      list_filings=lambda c, s: rows_by_cik[c])
    assert out["status"] == "ok" and out["phase"] == "score" and out["plan_built"] is True and out["counts"]["rows"] == 2
    st = bf.rows_status(root)
    assert st["months"] == 1 and st["plan"]["complete"] is True and st["status"]["phase"] == "score"
    assert "cik" not in json.dumps({k: v for k, v in st.items() if k != "status"}).lower().replace("ciks", "")


def test_prune_sections_removes_only_old_priors(tmp_path):
    tmp_arch = str(tmp_path / "t")
    for acc, fd in (("A-1", "2015-03-01"), ("A-2", "2016-03-01"), ("A-3", "2017-03-01")):
        p = sft.sections_path(tmp_arch, "1", acc)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with gzip.open(p, "wb") as f:
            f.write(b"{}")
    dates = {"A-1": "2015-03-01", "A-2": "2016-03-01", "A-3": "2017-03-01"}
    removed = bf.prune_sections(tmp_arch, dates, "2017-04-01")
    assert removed == 1 and not os.path.exists(sft.sections_path(tmp_arch, "1", "A-1"))
    assert os.path.exists(sft.sections_path(tmp_arch, "1", "A-2"))       # 396 days behind the cursor: inside the 440-day keep window
    assert bf.prune_sections(tmp_arch, dates, None) == 0                 # no cursor yet: nothing pruned
