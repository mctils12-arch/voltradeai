"""scripts/edgar13f_index_check.py — the 13F archive measured against EDGAR's
daily form index (ladder gate 1). Index lines copied from the real
form.20261116.idx; the day rows from the 2026-11-19 reading."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "edgar13f_index_check", os.path.join(os.path.dirname(__file__), "scripts", "edgar13f_index_check.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

LINES = "\n".join([
    "Form Type   Company Name                                                  CIK      Date Filed  File Name",
    "13F-HR           &PARTNERS                                                     107136      20261116    edgar/data/107136/0001214659-26-014210.txt                                                ",
    "13F-HR           1015 Capital Partners, LLC                                    2133991     20261116    edgar/data/2133991/0001376474-26-000734.txt                                                ",
    "13F-HR           1015 Capital Partners, LLC                                    2133991     20261116    edgar/data/2133991/0001376474-26-000734.txt                                                ",
    "13F-HR/A         ADAR1 Capital Management, LLC                                 1940272     20261116    edgar/data/1940272/0001940272-26-000034.txt                                                ",
    "13F-NT           140 Summer Partners Master Fund LP                            1844084     20261116    edgar/data/1844084/0000919574-26-007847.txt                                                ",
    "4                ADVANCED MICRO DEVICES INC                                    2488        20261116    edgar/data/2488/0000002488-26-000193.txt                                ",
])


def test_index_keeps_exact_13f_hr_types_once_per_accession():
    idx = m.parse_form_index(LINES)
    assert list(idx) == ["0001214659-26-014210", "0001376474-26-000734", "0001940272-26-000034"]
    assert idx["0001940272-26-000034"] == {"form": "13F-HR/A", "cik": "1940272", "filed": "2026-11-16"}


def test_index_url_is_edgars_quarter_path():
    assert m.daily_form_index_url("2026-11-16") == "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR4/form.20261116.idx"
    assert m.daily_form_index_url("2026-07-06").endswith("/2026/QTR3/form.20260706.idx")


def test_compare_day_counts_held_by_form_and_a_day_with_no_index_is_not_zero_filings():
    idx = m.parse_form_index(LINES)
    row = m.compare_day("2026-11-16", idx, {"0001214659-26-014210", "0009999999-26-000001"})
    assert (row["index"], row["held"], row["missing"], row["share"]) == (3, 1, 2, 0.3333)
    assert row["by_form"] == {"13F-HR": {"index": 2, "held": 1}, "13F-HR/A": {"index": 1, "held": 0}}
    closed = m.compare_day("2026-11-11", None, set())
    assert closed["index"] is None and closed["share"] is None


def test_summary_counts_complete_days_and_held_rows_outside_the_window():
    idx = m.parse_form_index(LINES)
    held = {"0001214659-26-014210", "0009999999-26-000001"}
    rows = [m.compare_day("2026-11-16", idx, held), m.compare_day("2026-11-14", None, held),
            m.compare_day("2026-11-17", {"0009999999-26-000001": {"form": "13F-HR", "cik": "1", "filed": "2026-11-17"}}, held)]
    total = m.summarize(rows, held, set(idx) | {"0009999999-26-000001"})
    assert total == {"index_days": 2, "index": 4, "held": 2, "missing": 2, "share": 0.5,
                     "complete_days": 1, "held_not_in_window_index": 0}


def test_paging_follows_the_routes_query_string_and_stops_when_not_truncated():
    route = "https://x/api/data/filings13f/history"
    assert m.next_page_url(route, {"truncated": True, "next": "?from=2026-11-02&to=2026-11-16"}) == \
        route + "?from=2026-11-02&to=2026-11-16&limit=500"
    assert m.next_page_url(route, {"truncated": False, "next": None}) is None


def test_a_weekday_with_no_index_is_never_printed_as_a_weekend():
    assert m.no_index_note("2026-11-14") == "no index (weekend)"
    assert "closed that day, or refused" in m.no_index_note("2026-11-11")
