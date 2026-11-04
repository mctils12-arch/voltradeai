"""Offline tests for sec_cik_symbol_map.py (Lazy Prices PR 3b): the join must be
dated, delisting-inclusive, conservative on ambiguity, and never guess."""
import gzip
import json

import sec_cik_symbol_map as m


def _map():
    return m.build_map([
        ("OLDCO", "0000001234", "2016-02-01"), ("OLDCO", "1234", "2018-05-01"),   # delisted name, zero-padded cik
        ("NEWCO", "1234", "2021-03-01"),                                          # same CIK, renamed later
        ("GOOG", "999", "2019-01-10"), ("GOOGL", "999", "2019-01-12"),            # share classes overlap
        ("", "5", "2019-01-01"), ("X", "5", ""), (None, "5", "2019-01-01"),       # junk is skipped
    ])


def test_build_normalizes_cik_and_spans():
    mp = _map()
    assert mp["1234"]["OLDCO"] == ["2016-02-01", "2018-05-01", 2]
    assert "5" not in mp


def test_resolve_exact_padded_none_and_rename():
    mp = _map()
    assert m.resolve(mp, "1234", "2017-01-01") == ("OLDCO", "exact")
    assert m.resolve(mp, "0001234", "2018-09-01") == ("OLDCO", "padded")   # inside slack after last filing
    assert m.resolve(mp, "1234", "2021-01-01") == ("NEWCO", "padded")      # NEWCO span +/- slack, OLDCO's slack ended
    assert m.resolve(mp, "1234", "2030-01-01") == (None, "none")           # far outside: never extrapolated
    assert m.resolve(mp, "777", "2018-01-01") == (None, "none")


def test_overlap_is_ambiguous_not_picked():
    mp = _map()
    assert m.resolve(mp, "999", "2019-01-11") == (None, "ambiguous")


def test_coverage_counts_every_row_and_panel_membership():
    mp = _map()
    rows = [{"cik": "1234", "accepted_at": "2017-03-01T12:00:00Z"},   # OLDCO exact, in panel
            {"cik": "1234", "accepted_at": "2021-03-05T12:00:00Z"},   # NEWCO, NOT in panel
            {"cik": "999", "accepted_at": "2019-01-11T00:00:00Z"},    # ambiguous
            {"cik": "42", "accepted_at": "2019-01-11T00:00:00Z"},     # unmapped
            {"cik": "42"}]                                            # undated
    rep = m.coverage(rows, mp, {"OLDCO"})
    assert rep["n_rows"] == 5 and rep["joined"] == 1
    assert rep["status"]["exact"] == 1 and rep["status"]["padded"] == 1 and rep["status"]["ambiguous"] == 1
    assert rep["status"]["none"] == 1 and rep["status"]["undated"] == 1
    assert rep["status"]["resolved_not_in_panel"] == 1
    assert rep["by_year"]["2017"]["frac"] == 1.0


def test_future_filings_cannot_change_a_resolved_identity_inside_a_seen_span():
    mp = m.build_map([("A", "7", "2016-01-01"), ("A", "7", "2018-01-01")])
    before = m.resolve(mp, "7", "2017-01-01")
    mp2 = m.build_map([("A", "7", "2016-01-01"), ("A", "7", "2018-01-01"), ("B", "7", "2024-01-01")])
    assert m.resolve(mp2, "7", "2017-01-01") == before == ("A", "exact")


def test_cache_reader_roundtrip(tmp_path):
    p = tmp_path / "2016q1.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(json.dumps(["ABC", "55", "9", "P", "2016-02-02", "2016-02-01", 1000]) + "\n")
        fh.write("not json\n")
    skipped = m.Counter()
    rows = list(m.rows_from_form4_cache(str(tmp_path), skipped))
    assert rows == [("ABC", "55", "2016-02-02")]
    assert skipped["bad_line"] == 1
    assert m.resolve(m.build_map(rows), "55", "2016-03-01")[0] == "ABC"


def test_coverage_accepts_form_idx_filingDate_key():
    mp = m.build_map([("A", "7", "2016-01-01")])
    rep = m.coverage([{"cik": "7", "filingDate": "2016-06-01"}], mp)
    assert rep["status"] == {"exact": 1} or rep["status"].get("padded") == 1
    assert rep["joined"] == 1
