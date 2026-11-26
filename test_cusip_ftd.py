"""cusip_ftd.py: the dated CUSIP-to-ticker record read from the SEC's
fails-to-deliver files, and the gate-1 count of how much of an event list
it covers (open_questions.md (at2), 2026-11-26). No network: every request
goes through a stand-in."""
import datetime as dt
import gzip
import io
import os
import re
import zipfile

import pytest

import cusip_ftd as ftd

REPO = os.path.dirname(os.path.abspath(__file__))
D = dt.date


def ftd_text(rows, count=None, qty=None, crlf=False):
    """A file in the SEC's shape: rows are (yyyymmdd, cusip, symbol, qty)."""
    lines = [ftd.HEADER] + [f"{d}|{c}|{s}|{q}|SOME COMPANY INC            |12.34" for d, c, s, q in rows]
    lines.append(f"Trailer record count {len(rows) if count is None else count}")
    lines.append(f"Trailer total quantity of shares {sum(r[3] for r in rows) if qty is None else qty}")
    return ("\r\n" if crlf else "\n").join(lines) + "\n"


def zipped(text, name="cnsfails202501a.txt", extra=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, text)
        if extra:
            z.writestr(extra, "x")
    return buf.getvalue()


ROWS = [("20250102", "037833100", "AAPL", 100), ("20250103", "037833100", "AAPL", 50),
        ("20250110", "037833100", "AAPL", 7), ("20250106", "G0750C108", "AXTA", 9),
        ("20250107", "03662Q105", "ANSS", 1), ("20250108", "03662Q105", "", 4),
        ("20250108", "BADCUSIP", "ZZZ", 4)]


def test_a_half_month_is_the_1st_to_the_14th_or_the_15th_to_the_end():
    assert ftd.period_of(D(2025, 1, 14)) == "202501a" and ftd.period_of(D(2025, 1, 15)) == "202501b"
    assert ftd.period_bounds("202502b") == (D(2025, 2, 15), D(2025, 2, 28))
    assert ftd.period_bounds("202412b") == (D(2024, 12, 15), D(2024, 12, 31))
    assert ftd.period_bounds("202501a") == (D(2025, 1, 1), D(2025, 1, 14))
    with pytest.raises(ValueError):
        ftd.period_bounds("2025-01")
    assert ftd.periods_between(D(2024, 12, 20), D(2025, 1, 15)) == ["202412b", "202501a", "202501b"]
    assert ftd.periods_between(D(2025, 3, 3), D(2025, 3, 3)) == ["202503a"]
    assert ftd.periods_between(D(2025, 3, 3), D(2025, 3, 2)) == []


def test_the_wanted_periods_are_a_45_day_window_around_the_events_and_never_past_today():
    w = ftd.wanted_periods(D(2025, 2, 25), D(2026, 11, 25), D(2026, 11, 26))
    assert w[0] == "202501a" and w[-1] == "202611b" and len(w) == 46 and w == sorted(w)
    # the newest events' forward window is not asked for before it exists
    assert ftd.wanted_periods(D(2025, 2, 25), D(2025, 3, 1), D(2026, 11, 26))[-1] == "202504b"


def test_the_file_is_parsed_only_when_its_header_and_both_trailers_hold():
    rows = ftd.parse(ftd_text(ROWS))
    assert len(rows) == 7 and rows[0] == ("2025-01-02", "037833100", "AAPL")
    assert ftd.parse(ftd_text(ROWS, crlf=True)) == rows
    with pytest.raises(ValueError, match="trailer count"):
        ftd.parse(ftd_text(ROWS, count=8))                      # a cut file
    with pytest.raises(ValueError, match="trailer quantity"):
        ftd.parse(ftd_text(ROWS, qty=1))
    with pytest.raises(ValueError, match="unexpected header"):
        ftd.parse("<html>Request Rate Threshold Exceeded</html>")
    with pytest.raises(ValueError, match="unexpected header"):
        ftd.parse("")
    # no trailer at all is a cut file too
    with pytest.raises(ValueError, match="trailer count None"):
        ftd.parse(ftd.HEADER + "\n20250102|037833100|AAPL|100|APPLE|1.00\n")


def test_the_zip_must_hold_exactly_one_member():
    assert ftd.unzip_single_text(zipped("abc")) == "abc"
    with pytest.raises(ValueError, match="2 members"):
        ftd.unzip_single_text(zipped("abc", extra="other.txt"))
    with pytest.raises(ValueError, match="not a zip"):
        ftd.unzip_single_text(b"<html>403</html>")


def test_pairs_keep_the_first_day_the_last_day_and_the_day_count_and_nothing_else(tmp_path):
    prs = ftd.pairs(ftd.parse(ftd_text(ROWS)))
    # a row with no symbol, and a CUSIP field that is not nine characters, are not pairs
    assert prs == {("037833100", "AAPL"): ["2025-01-02", "2025-01-10", 3],
                   ("G0750C108", "AXTA"): ["2025-01-06", "2025-01-06", 1],
                   ("03662Q105", "ANSS"): ["2025-01-07", "2025-01-07", 1]}
    root = str(tmp_path / "ftd")
    ftd.write_period(root, "202501a", prs)
    with gzip.open(ftd.period_path(root, "202501a"), "rt") as f:
        body = f.read()
    assert body == "03662Q105,ANSS,2025-01-07,2025-01-07,1\n037833100,AAPL,2025-01-02,2025-01-10,3\nG0750C108,AXTA,2025-01-06,2025-01-06,1\n"
    assert "12.34" not in body and "COMPANY" not in body          # no price, no description
    assert ftd.held_periods(root) == ["202501a"]
    # the same pairs give the same bytes (mtime 0)
    first = open(ftd.period_path(root, "202501a"), "rb").read()
    ftd.write_period(root, "202501a", prs)
    assert open(ftd.period_path(root, "202501a"), "rb").read() == first
    spans = ftd.read_spans(root, {"037833100", "000000000"})
    assert spans == {"037833100": [("2025-01-02", "2025-01-10", "AAPL", 3)]}


def test_the_three_known_addresses_are_tried_in_order_and_a_403_stops():
    seen = []

    def get_second(url):
        seen.append(url)
        return (200, b"zip") if "/other/" in url and "_0" not in url else (404, b"")
    assert ftd.fetch_period("202605a", get_second, sleep=lambda s: None) == ("ok", b"zip", 2)
    assert seen[0].endswith("/files/data/fails-deliver-data/cnsfails202605a.zip")
    assert ftd.fetch_period("202605a", lambda u: (404, b""), sleep=lambda s: None) == ("unavailable", None, 3)
    st, why, n = ftd.fetch_period("202605a", lambda u: (403, b""), sleep=lambda s: None)
    assert (st, n) == ("error", 1) and "403" in why
    st, why, n = ftd.fetch_period("202605a", lambda u: (0, b"timed out"), sleep=lambda s: None)
    assert (st, n) == ("error", 3) and "timed out" in why
    # the declared client is the one every SEC reader in the repository sends
    with open(os.path.join(REPO, "sec_filings_text.py")) as f:
        assert re.search(r'^SEC_UA = "(.+)"$', f.read(), re.M).group(1) == ftd.SEC_UA
    with open(os.path.join(REPO, "server", "secFtd.ts")) as f:
        ts = f.read()
    assert ftd.SEC_UA in ts and ftd.HEADER in ts
    for v in ftd.URL_VARIANTS:                                    # the same three addresses as the server's poller
        assert v.split("cnsfails")[0] in ts


def vendor(files, calls=None):
    """files: {period: bytes | status int}. Only the first address answers."""
    def get(url):
        if calls is not None:
            calls.append(url)
        m = re.search(r"cnsfails(\d{6}[ab])(_0)?\.zip$", url)
        v = files.get(m.group(1), 404)
        if "/other/" in url:
            return 404, b""
        return (v, b"") if isinstance(v, int) else (200, v)
    return get


def test_build_reads_the_periods_not_held_oldest_first_and_resumes(tmp_path):
    root = str(tmp_path / "ftd")
    files = {p: zipped(ftd_text(ROWS)) for p in ("202501a", "202501b", "202502a")}
    wanted = ["202501a", "202501b", "202502a"]
    calls, t = [], {"now": 0.0}

    def clock():
        t["now"] += 100.0
        return t["now"]
    out = ftd.build(root, wanted, D(2026, 11, 26), budget_s=250.0, get=vendor(files, calls), clock=clock, sleep=lambda s: None)
    # the budget is read between periods: two fit
    assert out["read"] == 2 and out["held"] == 2 and out["owed"] == 1 and out["stopped"] == "time budget" and out["status"] == "ok"
    assert [re.search(r"cnsfails(\w+)\.zip", u).group(1) for u in calls] == ["202501a", "202501b"]
    calls.clear()
    out = ftd.build(root, wanted, D(2026, 11, 26), get=vendor(files, calls), sleep=lambda s: None)
    assert out["read"] == 1 and out["owed"] == 0 and out["requests"] == 1 and out["stopped"] is None
    assert (out["first_period"], out["last_period"]) == ("202501a", "202502a")
    calls.clear()
    out = ftd.build(root, wanted, D(2026, 11, 26), get=vendor(files, calls), sleep=lambda s: None)
    assert calls == [] and out["read"] == 0 and out["status"] == "ok"   # nothing held is asked again


def test_a_new_period_the_sec_has_not_published_is_not_owed_and_an_old_missing_one_is(tmp_path):
    root = str(tmp_path / "ftd")
    files = {"202610b": zipped(ftd_text(ROWS))}
    out = ftd.build(root, ["202605a", "202610b", "202611a", "202611b"], D(2026, 11, 26), get=vendor(files), sleep=lambda s: None)
    assert out["read"] == 1 and out["not_published"] == 2 and out["unavailable"] == 1
    assert out["oldest_unavailable"] == "202605a" and out["owed"] == 1 and out["status"] == "ok"
    # the missing old period was not written down as missing: the next run asks again
    assert ftd.held_periods(root) == ["202610b"]


def test_a_refusal_stops_the_run_and_a_run_that_read_nothing_reads_error(tmp_path):
    root = str(tmp_path / "ftd")
    calls = []
    out = ftd.build(root, ["202501a", "202501b"], D(2026, 11, 26), get=vendor({"202501a": 403, "202501b": 403}, calls), sleep=lambda s: None)
    assert out["status"] == "error" and out["stopped"] == "refused by the SEC" and len(calls) == 1
    assert "202501a: HTTP 403" in out["last_error"]
    # a file of the wrong shape is refused whole and nothing is written for it
    cut = zipped(ftd_text(ROWS, count=99))
    out = ftd.build(root, ["202501a", "202501b"], D(2026, 11, 26), get=vendor({"202501a": cut, "202501b": zipped(ftd_text(ROWS))}), sleep=lambda s: None)
    assert out["read"] == 1 and out["errors"] == 1 and out["status"] == "ok" and ftd.held_periods(root) == ["202501b"]
    assert "trailer count 99" in out["last_error"]
    # three unreadable periods in a row stop the run
    bad = {p: 500 for p in ("202503a", "202503b", "202504a", "202504b")}
    calls.clear()
    out = ftd.build(str(tmp_path / "x"), sorted(bad), D(2026, 11, 26), get=vendor(bad, calls), sleep=lambda s: None)
    assert out["status"] == "error" and out["stopped"] == "3 periods in a row could not be read" and out["errors"] == 3


def test_a_pair_known_on_the_day_is_preferred_to_one_printed_afterwards():
    spans = [("2025-01-02", "2025-01-10", "OLD", 3), ("2025-03-03", "2025-03-12", "NEW", 5)]
    # inside a span
    assert ftd.symbol_on(spans, "2025-01-05") == {"symbol": "OLD", "gap_days": 0, "known_on_the_day": True,
                                                  "first_day": "2025-01-02", "last_day": "2025-01-10"}
    # 30 days after the old print and 22 before the new one: the nearer is ahead, the known one is used
    j = ftd.symbol_on(spans, "2025-02-09")
    assert (j["symbol"], j["gap_days"], j["known_on_the_day"]) == ("OLD", 30, True)
    # past the window behind, a print ahead inside it is used and marked as not known on the day
    j = ftd.symbol_on(spans, "2025-02-26")
    assert (j["symbol"], j["gap_days"], j["known_on_the_day"]) == ("NEW", 5, False)
    assert ftd.symbol_on(spans, "2025-01-01")["known_on_the_day"] is False
    # 46 days from anything: no pair
    assert ftd.symbol_on([spans[0]], "2025-02-25") is None
    assert ftd.symbol_on([spans[0]], "2025-02-24")["gap_days"] == 45
    assert ftd.symbol_on([], "2025-02-24") is None
    # of two prints equally near behind, the later one
    two = [("2025-01-02", "2025-01-10", "A", 1), ("2025-01-05", "2025-01-10", "B", 1)]
    assert ftd.symbol_on(two, "2025-01-20")["symbol"] == "B"


def test_the_groups_are_the_three_kept_types_and_the_share_like_unknowns():
    common = {"found": True, "us": True, "market_sector": "Equity", "security_type": "Common Stock", "ticker": "AAPL"}
    assert ftd.group_of("037833100", common) == "kept_us"
    assert ftd.group_of("037833100", dict(common, us=False)) == "kept_no_us"
    assert ftd.group_of("037833100", dict(common, security_type="ETP")) is None
    assert ftd.group_of("037833100", {"found": False}) == "unresolved_share_like"
    assert ftd.group_of("03783AB12", {"found": False}) is None        # a letter in the issue number: a note
    assert ftd.group_of("SPXSRT5E0", {"found": False}) is None
    assert ftd.group_of("037833100", None) is None                    # not asked yet: not counted as anything
    assert ftd.half_year("2025-06-30") == "2025H1" and ftd.half_year("2025-07-01") == "2025H2"
    assert ftd.norm_symbol("BRK.B") == ftd.norm_symbol("brk/b") == "BRKB" and ftd.norm_symbol(None) == ""


def test_coverage_counts_events_and_underliers_by_group_names_no_security_and_reads_no_price():
    eq = {"found": True, "us": True, "market_sector": "Equity", "security_type": "Common Stock"}
    cache = {
        "111111101": dict(eq, ticker="AAA"),                      # joined, same symbol
        "222222202": dict(eq, ticker="NEWB"),                     # joined, renamed since
        "333333303": dict(eq, us=False, ticker="FOREIGN"),        # bought since: joined by the dated record only
        "444444404": {"found": False},                            # unknown, share-like, no fail on file
        "555555505": dict(eq, security_type="ETP", ticker="SPY"),  # dropped by the filter: not counted
        "666666606": dict(eq, ticker="BRK/B"),
    }
    spans = {
        "111111101": [("2025-03-03", "2025-03-10", "AAA", 4)],
        "222222202": [("2025-03-03", "2025-03-10", "OLDB", 4)],
        "333333303": [("2025-04-01", "2025-04-05", "TGT", 2)],     # only after its first event
        "666666606": [("2026-07-01", "2026-07-05", "BRK.B", 2)],
    }
    events = [
        {"date": "2025-03-12", "cusip": "111111101"}, {"date": "2025-09-01", "cusip": "111111101"},   # second: no pair in 45 days
        {"date": "2025-03-12", "cusip": "222222202"},
        {"date": "2025-03-12", "cusip": "333333303"}, {"date": "2025-04-10", "cusip": "333333303"},
        {"date": "2025-03-12", "cusip": "444444404"},
        {"date": "2025-03-12", "cusip": "555555505"},
        {"date": "2026-07-10", "cusip": "666666606"},
    ]
    out = ftd.coverage(events, cache, spans)
    g = out["groups"]
    assert g["kept_us"] == {"underliers": 3, "underliers_joined": 3, "underliers_joined_back": 3,
                            "events": 4, "events_joined": 3, "events_joined_back": 3}
    # the first event is joined only by a print 20 days later; the second by one known on the day
    assert g["kept_no_us"] == {"underliers": 1, "underliers_joined": 1, "underliers_joined_back": 0,
                               "events": 2, "events_joined": 2, "events_joined_back": 1}
    assert g["unresolved_share_like"] == {"underliers": 1, "underliers_joined": 0, "underliers_joined_back": 0,
                                          "events": 1, "events_joined": 0, "events_joined_back": 0}
    assert out["agreement_kept_us"] == {"events": 3, "same": 2, "different": 1, "underliers": 3, "underliers_same": 2}
    assert out["by_half_year"] == {"2025H1": {"events": 5, "no_join": 1, "no_join_pct": 20.0},
                                   "2025H2": {"events": 1, "no_join": 1, "no_join_pct": 100.0},
                                   "2026H2": {"events": 1, "no_join": 0, "no_join_pct": 0.0}}
    assert out["no_join_first_minus_last_points"] == 20.0 and out["events_of_other_types"] == 1 and out["window_days"] == 45
    import json
    text = json.dumps(out)
    assert not re.search(r'"[0-9A-Z]{8}[0-9]"', text) and "AAA" not in text and "TGT" not in text
    # with no events there is nothing to divide
    assert ftd.coverage([], cache, spans)["no_join_first_minus_last_points"] is None


def test_the_module_reads_no_price_source_and_keeps_no_price_column():
    with open(os.path.join(REPO, "cusip_ftd.py")) as f:
        code = f.read().split('"""', 2)[2]
    for word in ("price_panel", "alpaca", "yfinance"):
        assert word not in code.lower(), word
    parse_src = code.split("def parse")[1].split("def pairs")[0]
    assert "p[5]" not in parse_src and "p[4]" not in parse_src     # the PRICE and DESCRIPTION columns are never read
    assert all(len(r) == 3 for r in ftd.parse(ftd_text(ROWS)))
