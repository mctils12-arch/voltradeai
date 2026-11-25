"""
test_dtcc_trs_panel.py: pure-function battery for dtcc_trs_panel.py
(the event panel the registered DTCC equity-swap gate-2 test reads).
No network: build_day's fetch is exercised only by running the script. The
zip and compaction paths run on files written here.
"""
import datetime as dt
import gzip
import importlib.util
import json
import os
import zipfile

import pytest

_spec = importlib.util.spec_from_file_location(
    "dtcc_trs_panel", os.path.join(os.path.dirname(__file__), "dtcc_trs_panel.py"))
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)

HEADER = ["Dissemination Identifier", "Action type", "Event type", "Execution Timestamp",
          "Notional amount-Leg 1", "Notional currency-Leg 1",
          "Underlier ID-Leg 1", "Underlier ID source-Leg 1"]
DAY = dt.date(2025, 6, 2)


def row(act="NEWT", evt="TRAD", exe="2025-06-02T14:00:00Z", amt="1,000", cur="USD",
        uid="US0378331005", src="ISIN", did="1"):
    return [did, act, evt, exe, amt, cur, uid, src]


def test_us_cusip_takes_the_isin_middle_and_refuses_baskets_and_foreign():
    assert panel.us_cusip("US0378331005", "ISIN") == "037833100"
    assert panel.us_cusip("037833100", "CUSIP") == "037833100"
    assert panel.us_cusip("CNE000001R84", "ISIN") is None
    assert panel.us_cusip("US0378331005;US5949181045", "ISIN") is None
    assert panel.us_cusip("US0378331005", "ISIN;ISIN") is None
    assert panel.us_cusip("US03783310", "ISIN") is None      # wrong length
    assert panel.us_cusip("0378331", "CUSIP") is None
    assert panel.us_cusip("US0378331005", "RIC") is None


def test_parse_notional_reads_commas_caps_and_refuses_what_is_not_a_number():
    assert panel.parse_notional("2,000") == (2000, False)
    assert panel.parse_notional("250,000,000+") == (250000000, True)
    assert panel.parse_notional("1500.6") == (1501, False)
    assert panel.parse_notional("") == (None, False)
    assert panel.parse_notional("N/A") == (None, False)
    assert panel.parse_notional("-5") == (None, False)
    assert panel.parse_notional("nan") == (None, False)


def test_aggregate_counts_fresh_new_trades_and_sums_usd_only():
    rows = [
        row(amt="1,000"),
        row(amt="250,000,000+"),                               # capped: floor added, counted
        row(exe="2025-06-01T23:59:00Z", amt="500"),           # the day before is still fresh
        row(cur="EUR", amt="9,000"),                           # counted, not summed
        row(exe="2025-05-20T10:00:00Z", amt="7,000,000"),     # late report: not news
        row(act="MODI"), row(act="MODI"), row(act="TERM", evt="ETRM"),
        row(act="CORR", evt=""), row(act="NEWT", evt="CORP", amt="77"),
    ]
    out, stats = panel.aggregate(HEADER, rows, DAY)
    assert list(out) == ["037833100"]
    assert out["037833100"] == {"newt_n": 4, "newt_usd": 250001500, "capped_n": 1, "fx_n": 1,
                                "late_n": 1, "modi_n": 2, "term_n": 1, "other_n": 2}
    assert stats["rows"] == 10 and stats["us_rows"] == 10
    assert stats["malformed"] == 0 and stats["baskets"] == 0


def test_aggregate_drops_and_counts_baskets_foreign_and_malformed_rows():
    rows = [
        row(uid="US0378331005;US5949181045"),
        row(src="ISIN;ISIN"),
        row(uid="CNE000001R84"),
        row()[:-1],                                            # short row
        row(uid="594918104", src="CUSIP", amt="40"),
    ]
    out, stats = panel.aggregate(HEADER, rows, DAY)
    assert set(out) == {"594918104"}
    assert out["594918104"]["newt_usd"] == 40
    assert stats == {"rows": 5, "malformed": 1, "baskets": 2, "us_rows": 1,
                     "newt_notional_unreadable": 0, "newt_notional_over_cap": 0, "columns": 8}


def test_aggregate_counts_a_fresh_usd_trade_with_no_readable_notional():
    out, stats = panel.aggregate(HEADER, [row(amt=""), row(amt="12")], DAY)
    assert out["037833100"]["newt_n"] == 2          # the trade happened
    assert out["037833100"]["newt_usd"] == 12       # nothing invented for the blank
    assert stats["newt_notional_unreadable"] == 1


def test_an_unmarked_notional_above_the_vendor_cap_is_a_placeholder_and_adds_nothing():
    rows = [row(amt="100,000,000,000,000,000,000"),      # the placeholder seen on 62 of 731 days
            row(amt="250,000,001"),                       # above the cap, not marked: not a figure
            row(amt="250,000,000"),                       # at the cap: a figure
            row(amt="250,000,000+")]                      # marked capped: its floor
    out, stats = panel.aggregate(HEADER, rows, DAY)
    assert out["037833100"]["newt_n"] == 4
    assert out["037833100"]["newt_usd"] == 500_000_000
    assert out["037833100"]["capped_n"] == 1
    assert stats["newt_notional_over_cap"] == 2 and stats["newt_notional_unreadable"] == 0
    assert panel.NOTIONAL_CAP_USD == 250_000_000


def test_aggregate_is_indifferent_to_column_order_and_fails_on_a_missing_column():
    order = [7, 6, 5, 4, 3, 2, 1, 0]
    h = [HEADER[i] for i in order]
    r = [row(amt="300")[i] for i in order]
    out, _ = panel.aggregate(h, [r], DAY)
    assert out["037833100"]["newt_usd"] == 300
    with pytest.raises(ValueError, match="schema drift: missing columns: Event type"):
        panel.aggregate([c for c in HEADER if c != "Event type"], [], DAY)


def test_read_zip_rows_reads_a_single_member_and_refuses_two(tmp_path):
    z = tmp_path / "a.zip"
    body = ",".join(HEADER) + "\n" + '9,NEWT,TRAD,2025-06-02T01:00:00Z,"1,250",USD,US0378331005,ISIN\n'
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        f.writestr("SEC_CUMULATIVE_EQUITIES_2025_06_02.csv", body)
    header, rows = panel.read_zip_rows(str(z))
    out, _ = panel.aggregate(header, rows, DAY)
    assert out == {"037833100": {"newt_n": 1, "newt_usd": 1250, "capped_n": 0, "fx_n": 0,
                                 "late_n": 0, "modi_n": 0, "term_n": 0, "other_n": 0}}
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("a.csv", body)
        f.writestr("b.csv", body)
    with pytest.raises(ValueError, match="expected one CSV member, found 2"):
        panel.read_zip_rows(str(z))


def _seed_day(work, day, per_cusip, **stats):
    days = os.path.join(work, "days")
    os.makedirs(days, exist_ok=True)
    panel.write_day(os.path.join(days, day + ".csv"), dt.date.fromisoformat(day), per_cusip)
    with open(os.path.join(days, day + ".json"), "w") as f:
        json.dump({"date": day, "status": "ok", "underliers": len(per_cusip), **stats}, f)


def test_compact_writes_quarter_files_a_manifest_and_reads_back(tmp_path):
    work, out = str(tmp_path / "w"), str(tmp_path / "o")
    c = dict.fromkeys(panel.PANEL_FIELDS[2:], 0)
    _seed_day(work, "2025-03-31", {"037833100": {**c, "newt_n": 2, "newt_usd": 900}}, rows=5, us_rows=2)
    _seed_day(work, "2025-04-01", {"594918104": {**c, "modi_n": 3},
                                   "037833100": {**c, "newt_n": 1, "newt_usd": 10}}, rows=9, us_rows=4)
    with open(os.path.join(work, "days", "2025-04-02.json"), "w") as f:
        json.dump({"date": "2025-04-02", "status": "unavailable", "http": 403}, f)
    m = panel.compact(work, out)
    assert m["days_ok"] == 2 and m["rows"] == 3
    assert m["first_day"] == "2025-03-31" and m["last_day"] == "2025-04-01"
    assert m["days_unavailable"] == ["2025-04-02"]       # a refused day is named, never an empty day
    assert m["files"] == {"panel_2025Q1.csv.gz": {"days": 1, "rows": 1},
                          "panel_2025Q2.csv.gz": {"days": 1, "rows": 2}}
    assert m["days"]["2025-04-01"] == {"rows": 9, "us_rows": 4, "underliers": 2}
    got = list(panel.read_panel(out))
    assert [(r["date"], r["cusip"], r["newt_n"], r["newt_usd"], r["modi_n"]) for r in got] == [
        ("2025-03-31", "037833100", 2, 900, 0),
        ("2025-04-01", "037833100", 1, 10, 0),
        ("2025-04-01", "594918104", 0, 0, 3),
    ]
    with gzip.open(os.path.join(out, "panel_2025Q2.csv.gz"), "rt") as f:
        assert f.readline().strip() == ",".join(panel.PANEL_FIELDS)


def test_compact_is_byte_stable_for_the_same_days(tmp_path):
    work = str(tmp_path / "w")
    c = dict.fromkeys(panel.PANEL_FIELDS[2:], 0)
    _seed_day(work, "2025-06-02", {"037833100": {**c, "newt_n": 1, "newt_usd": 5}}, rows=1, us_rows=1)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    panel.compact(work, a)
    panel.compact(work, b)
    for name in ("panel_2025Q2.csv.gz", "manifest.json"):
        with open(os.path.join(a, name), "rb") as fa, open(os.path.join(b, name), "rb") as fb:
            assert fa.read() == fb.read()


def test_build_day_returns_the_recorded_line_without_fetching(tmp_path):
    work = str(tmp_path)
    os.makedirs(os.path.join(work, "days"))
    line = {"date": "2025-06-02", "status": "ok", "underliers": 7}
    with open(os.path.join(work, "days", "2025-06-02.json"), "w") as f:
        json.dump(line, f)
    assert panel.build_day("2025-06-02", work) == line     # resumable: no second download


def test_dates_quarters_and_urls():
    assert panel.daterange(dt.date(2024, 12, 30), dt.date(2025, 1, 1)) == [
        "2024-12-30", "2024-12-31", "2025-01-01"]
    assert [panel.quarter_of(d) for d in ("2025-01-01", "2025-03-31", "2025-04-01", "2025-12-31")] == [
        "2025Q1", "2025Q1", "2025Q2", "2025Q4"]
    assert panel.vendor_url(dt.date(2024, 11, 24)).endswith("/sec/eod/SEC_CUMULATIVE_EQUITIES_2024_11_24.zip")


# ── the registered event list ────────────────────────────────────────────────

def _weekdays(start, n):
    out, d = [], dt.date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _prow(day, cusip, usd, n=1):
    return {"date": day.isoformat(), "cusip": cusip, "newt_usd": usd, "newt_n": n}


def _all_days(tdays):
    return [(tdays[0] + dt.timedelta(days=i)).isoformat() for i in range((tdays[-1] - tdays[0]).days + 1)]


def test_event_definition_constants_are_the_registered_ones():
    assert (panel.EVENT_LOOKBACK, panel.EVENT_MIN_COVERED, panel.EVENT_MAX_RANK,
            panel.EVENT_FLOOR_USD, panel.EVENT_COOLDOWN) == (60, 55, 3, 1_000_000, 20)


def test_an_event_is_a_value_fewer_than_three_prior_days_reached_and_is_known_the_next_day():
    t = _weekdays("2025-01-06", 70)
    rows = [_prow(d, "AAAAAAAA1", 2_000_000) for d in t[:60]]      # a steady name
    rows += [_prow(t[60], "AAAAAAAA1", 2_000_000)]                  # 60 prior days as large: not an event
    rows += [_prow(t[61], "AAAAAAAA1", 9_000_000, n=4)]             # none as large: an event
    events, summary = panel.detect_events(rows, _all_days(t), frozenset())
    assert [(e["cusip"], e["date"], e["known"], e["newt_usd"], e["newt_n"]) for e in events] == [
        ("AAAAAAAA1", t[61].isoformat(), t[62].isoformat(), 9_000_000, 4)]
    assert events[0]["prior_max_usd"] == 2_000_000 and events[0]["prior_days_with_new"] == 60
    assert summary["testable_days"] == 10 and summary["first_testable"] == t[60].isoformat()


def test_two_prior_days_as_large_still_an_event_three_is_not():
    t = _weekdays("2025-01-06", 62)
    two = [_prow(t[5], "B", 5_000_000), _prow(t[9], "B", 5_000_000), _prow(t[60], "B", 5_000_000)]
    three = two + [_prow(t[20], "C", 1)] + [_prow(d, "C", 5_000_000) for d in (t[1], t[2], t[3], t[61])]
    ev, _ = panel.detect_events(three, _all_days(t), frozenset())
    assert [(e["cusip"], e["date"]) for e in ev] == [("B", t[60].isoformat())]


def test_no_event_under_the_floor_and_none_before_sixty_days_of_history():
    t = _weekdays("2025-01-06", 62)
    rows = [_prow(t[30], "D", 50_000_000),      # day 30: no 60-day lookback yet
            _prow(t[61], "E", 999_999)]         # a 60-day high, under $1,000,000
    ev, _ = panel.detect_events(rows, _all_days(t), frozenset())
    assert ev == []


def test_cooldown_makes_one_burst_one_event():
    t = _weekdays("2025-01-06", 95)
    rows = [_prow(t[60], "F", 3_000_000), _prow(t[61], "F", 8_000_000),     # inside the cooldown
            _prow(t[80], "F", 20_000_000),                                    # day 80 = 60 + 20: still inside
            _prow(t[81], "F", 30_000_000)]                                    # first day after it
    ev, _ = panel.detect_events(rows, _all_days(t), frozenset())
    assert [e["date"] for e in ev] == [t[60].isoformat(), t[81].isoformat()]


def test_weekend_and_holiday_rows_count_on_the_next_trading_day():
    t = _weekdays("2025-01-06", 62)
    holiday = t[61]
    closed = frozenset([holiday])
    sat = t[59] + dt.timedelta(days=1)
    assert sat.weekday() == 5
    rows = [_prow(sat, "G", 4_000_000), _prow(t[60], "G", 1_000_000),       # Saturday + Monday = 5,000,000
            _prow(holiday, "H", 7_000_000)]                                  # no trading day after it in range
    days = _all_days(t)
    ev, summary = panel.detect_events(rows, days, closed)
    assert [(e["cusip"], e["date"], e["newt_usd"], e["known"]) for e in ev] == [("G", t[60].isoformat(), 5_000_000, None)]
    assert summary["trading_days"] == 61


def test_a_day_whose_file_was_not_read_is_not_tested_and_is_not_a_zero():
    t = _weekdays("2025-01-06", 70)
    days = [d for d in _all_days(t) if d not in {x.isoformat() for x in t[2:8]}]   # six files missing
    rows = [_prow(t[62], "I", 6_000_000), _prow(t[68], "I", 60_000_000)]
    ev, summary = panel.detect_events(rows, days, frozenset())
    # t[62]'s lookback holds 54 read days (under 55): untested. t[68]'s holds all 60.
    assert [e["date"] for e in ev] == [t[68].isoformat()]
    assert summary["covered"] == 64
    miss = [d for d in _all_days(t) if d != t[68].isoformat()]
    assert panel.detect_events(rows[1:], miss, frozenset())[0] == []    # the day itself unread: no event


def test_closures_cover_the_panel_years_and_events_file_is_written(tmp_path):
    closed = panel.nyse_closures()
    for d in ("2024-11-28", "2025-01-09", "2025-12-25", "2026-11-26"):
        assert dt.date.fromisoformat(d) in closed
    assert dt.date(2025, 11, 28) not in closed          # a half day is a trading day
    t = panel.trading_days(dt.date(2026, 2, 2), dt.date(2026, 6, 30), closed)[:62]
    assert dt.date(2026, 2, 16) not in t and dt.date(2026, 4, 3) not in t
    work, out = str(tmp_path / "w"), str(tmp_path / "o")
    c = dict.fromkeys(panel.PANEL_FIELDS[2:], 0)
    for d in t:
        per = {"037833100": {**c, "newt_n": 1, "newt_usd": 100}}
        if d == t[61]:
            per = {"037833100": {**c, "newt_n": 3, "newt_usd": 2_500_000}}
        _seed_day(work, d.isoformat(), per, rows=1, us_rows=1)
    panel.compact(work, out)
    doc = panel.write_events(out)
    assert panel.read_events(out) == doc
    with open(os.path.join(out, "events.json.gz"), "rb") as f:
        first = f.read()
    panel.write_events(out)
    with open(os.path.join(out, "events.json.gz"), "rb") as f:
        assert f.read() == first                        # same panel, same bytes
    assert doc["definition"] == {"lookback": 60, "min_covered": 55, "max_rank": 3,
                                 "floor_usd": 1_000_000, "cooldown": 20}
    assert [(e["cusip"], e["date"], e["newt_usd"]) for e in doc["events"]] == [
        ("037833100", t[61].isoformat(), 2_500_000)]
