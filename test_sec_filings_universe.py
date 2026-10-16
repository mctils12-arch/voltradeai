"""Offline tests for sec_filings_universe.py (Lazy Prices PR 2a)."""
import gzip
from datetime import date

import pytest

import sec_filings_text as sft
import sec_filings_universe as u

# Real layout: header text is 5 columns LEFT of the data columns (verified 1999-2026).
HDR = ("Form Type   Company Name                                                  CIK         Date Filed  File Name\n"
       + "-" * 120 + "\n")


def line(form, name, cik, d, fname):
    return f"{form:<17}{name:<62}{cik:<12}{d:<12}{fname}\n"


IDX = HDR + "".join([
    line("10-K", "ACME, INC. /DE/", "34782", "2012-03-23", "edgar/data/34782/0001104659-12-010878.txt"),
    line("10-Q", "Zeta Corp  (formerly Beta)", "1420525", "2012-02-01", "edgar/data/1420525/0001420525-12-000021.txt"),
    line("10-K/A", "ACME, INC. /DE/", "34782", "2012-04-01", "edgar/data/34782/0001104659-12-020000.txt"),
    line("8-K", "ACME, INC. /DE/", "34782", "2012-03-01", "edgar/data/34782/0001104659-12-000001.txt"),
    line("10-K405", "OLD CO", "99", "2012-03-02", "edgar/data/99/0000000099-12-000002.txt"),
    line("10-Q", "BAD ROW", "notacik", "2012-03-02", "edgar/data/1/0000000001-12-000002.txt"),
    line("10-Q", "BAD DATE", "5", "2012-3-2", "edgar/data/5/0000000005-12-000002.txt"),
])


def test_parse_keeps_report_forms_and_fixed_width_names():
    rows = u.parse_form_idx(IDX)
    assert [r["accession"] for r in rows] == ["0001104659-12-010878", "0001420525-12-000021", "0000000099-12-000002"]
    assert rows[0]["name"] == "ACME, INC. /DE/" and rows[0]["cik"] == "0000034782"
    assert rows[1]["name"] == "Zeta Corp  (formerly Beta)"
    assert rows[2]["form"] == "10-K405"


def test_amendments_excluded_by_default_included_on_request():
    assert not any(r["form"].endswith("/A") for r in u.parse_form_idx(IDX))
    assert any(r["form"] == "10-K/A" for r in u.parse_form_idx(IDX, include_amendments=True))


def test_real_sample_rows_parse():
    real = HDR + ("10-K             1ST CONSTITUTION BANCORP                                      1141807     2012-03-23  edgar/data/1141807/0001214659-12-001265.txt         \n"
                  "10-Q             Name With 123 Digits Inc 7                                    77          2012-03-24  edgar/data/77/0000000077-12-000009.txt\n")
    rows = u.parse_form_idx(real)
    assert [(r["cik"], r["name"]) for r in rows] == [("0001141807", "1ST CONSTITUTION BANCORP"), ("0000000077", "Name With 123 Digits Inc 7")]


def test_missing_header_raises():
    with pytest.raises(ValueError):
        u.parse_form_idx("garbage")


def test_quarter_completeness():
    assert not u.quarter_is_complete(2026, 4, today=date(2026, 10, 16))
    assert not u.quarter_is_complete(2026, 3, today=date(2026, 10, 3))   # inside slack week
    assert u.quarter_is_complete(2026, 3, today=date(2026, 10, 8))
    assert u.quarter_is_complete(2025, 4, today=date(2026, 1, 8))
    assert not u.quarter_is_complete(2025, 4, today=date(2026, 1, 7))


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, url):
        self.calls.append(url)
        return IDX.encode("latin-1")


def test_fetch_caches_complete_quarter_write_once_and_not_the_open_one(tmp_path):
    c = FakeClient()
    arch = str(tmp_path)
    rows, cached = u.fetch_quarter(2012, 1, c, arch, today=date(2026, 10, 16))
    assert len(rows) == 3 and not cached
    rows2, cached2 = u.fetch_quarter(2012, 1, c, arch, today=date(2026, 10, 16))
    assert cached2 and rows2 == rows and len(c.calls) == 1
    _, cached3 = u.fetch_quarter(2026, 4, c, arch, today=date(2026, 10, 16))
    _, cached4 = u.fetch_quarter(2026, 4, c, arch, today=date(2026, 10, 16))
    assert not cached3 and not cached4  # open quarter refetched, never cached
    with gzip.open(u.idx_cache_path(arch, 2012, 1)) as f:
        assert f.read().startswith(b"Form Type")


def test_summarize_survivorship_share():
    rows = u.parse_form_idx(IDX)
    s = u.summarize(rows, current_ciks={"0000034782"})
    assert s["ciks"] == 3 and s["annual"] == 2 and s["quarterly"] == 1
    assert s["not_current_ciks"] == 2 and s["not_current_share"] == pytest.approx(2 / 3, abs=1e-3)
    assert "not_current_ciks" not in u.summarize(rows)
