"""
test_dtcc_trs_panel_job.py: the server's builder of the DTCC equity-swap
panel (dtcc_trs_panel_job.py). No network: the vendor read is a stand-in
that writes the same day file and stat line `dtcc_trs_panel.build_day` does.
"""
import csv
import datetime as dt
import gzip
import json
import os

import dtcc_trs_panel as panel
import dtcc_trs_panel_job as job

REPO = os.path.dirname(os.path.abspath(__file__))
TODAY = dt.date(2026, 11, 26)
COUNTERS = panel.PANEL_FIELDS[2:]


def fake_vendor(refuse=(), fail=(), calls=None, usd=None):
    """A build_day stand-in. `refuse`: days answered 403; `fail`: days that
    raise inside the reader (no stat file, as the real one leaves none)."""
    def build(day_iso, work, tmp_dir=None):
        if calls is not None:
            calls.append((day_iso, tmp_dir))
        days = os.path.join(work, "days")
        if day_iso in fail:
            return {"date": day_iso, "status": "error", "error": "URLError: timed out"}
        if day_iso in refuse:
            line = {"date": day_iso, "status": "unavailable", "http": 403}
        else:
            c = dict.fromkeys(COUNTERS, 0)
            c.update(newt_n=1, newt_usd=(usd or {}).get(day_iso, 1000))
            panel.write_day(os.path.join(days, day_iso + ".csv"), dt.date.fromisoformat(day_iso), {"037833100": c})
            line = {"date": day_iso, "status": "ok", "underliers": 1, "rows": 1, "us_rows": 1}
        with open(os.path.join(days, day_iso + ".json"), "w") as f:
            json.dump(line, f)
        return line
    return build


def run(tmp_path, **kw):
    kw.setdefault("today", TODAY)
    kw.setdefault("tmp_dir", str(tmp_path / "tmp"))
    kw.setdefault("build", fake_vendor())
    kw.setdefault("max_days", 10_000)
    kw.setdefault("ftd_fetch", lambda root, budget_s: None)      # no SEC request from a test
    return job.run(str(tmp_path / "root"), **kw)


def test_the_registered_digest_is_the_committed_event_list():
    with gzip.open(os.path.join(REPO, "datacore", "dtcc_trs_panel", "events.json.gz"), "rt") as f:
        doc = json.load(f)
    reg = job.REGISTERED_EVENTS
    assert reg["through"] == doc["panel"]["last_day"] == "2026-11-24"
    assert reg["events"] == len(doc["events"]) == 27869
    got = job.digest_by_month(doc["events"], reg["through"])
    assert {m: list(v) for m, v in got.items()} == reg["by_month"]
    check = job.compare_with_registered(doc["events"])
    assert check == {"through": "2026-11-24", "registered": 27869, "here": 27869,
                     "match": True, "months_differing": []}


def test_a_changed_missing_or_extra_event_names_its_month_and_later_days_are_not_compared():
    with gzip.open(os.path.join(REPO, "datacore", "dtcc_trs_panel", "events.json.gz"), "rt") as f:
        events = json.load(f)["events"]
    later = {"cusip": "037833100", "date": "2026-11-25", "newt_usd": 5_000_000}
    assert job.compare_with_registered(events + [later])["match"] is True     # after the registered last day
    dropped = job.compare_with_registered(events[1:])
    assert dropped["match"] is False and dropped["here"] == 27868
    assert dropped["months_differing"] == [{"month": "2025-02", "here": 454, "registered": 455}]
    changed = [dict(e) for e in events]
    changed[-1]["newt_usd"] += 1                                             # same count, another value
    got = job.compare_with_registered(changed)
    assert got["match"] is False and got["here"] == 27869
    assert got["months_differing"] == [{"month": "2026-11", "here": 1206, "registered": 1206}]
    extra = job.compare_with_registered(events + [{"cusip": "037833100", "date": "2025-01-31", "newt_usd": 1}])
    assert extra["months_differing"] == [{"month": "2025-01", "here": 1, "registered": 0}]


def test_wanted_days_run_from_the_floor_or_the_readable_edge_to_yesterday():
    w = job.wanted_days(TODAY)
    assert w[0] == "2024-11-24" and w[-1] == "2026-11-25" and len(w) == 732
    later = job.wanted_days(dt.date(2027, 3, 1))
    assert later[0] == (dt.date(2027, 3, 1) - dt.timedelta(days=job.READABLE_DAYS)).isoformat() == "2025-02-27"
    assert later[-1] == "2027-02-28"
    assert job.wanted_days(dt.date(2024, 11, 24)) == []


def test_days_are_read_oldest_first_the_zip_stays_off_the_volume_and_day_files_are_gzipped(tmp_path):
    calls = []
    out = run(tmp_path, max_days=3, build=fake_vendor(calls=calls))
    assert [c[0] for c in calls] == ["2024-11-24", "2024-11-25", "2024-11-26"]
    assert all(c[1] == str(tmp_path / "tmp") for c in calls)
    assert out["read"] == 3 and out["stopped"] == "day limit" and out["status"] == "ok"
    assert out["wanted"] == 732 and out["pending"] == 729 and out["backlog"] == 725
    days = sorted(os.listdir(tmp_path / "root" / "days"))
    assert days == ["2024-11-24.csv.gz", "2024-11-24.json", "2024-11-25.csv.gz", "2024-11-25.json",
                    "2024-11-26.csv.gz", "2024-11-26.json"]
    assert out["events"] is None                       # nothing is compacted while days are owed
    assert not os.path.exists(tmp_path / "root" / "panel")
    with open(tmp_path / "root" / "last_run.json") as f:
        assert json.load(f)["read"] == 3


def test_a_second_run_resumes_where_the_first_stopped(tmp_path):
    run(tmp_path, max_days=2)
    calls = []
    out = run(tmp_path, max_days=2, build=fake_vendor(calls=calls))
    assert [c[0] for c in calls] == ["2024-11-26", "2024-11-27"]
    assert out["days_ok"] == 4 and out["first_day"] == "2024-11-24" and out["last_day"] == "2024-11-27"


def test_the_time_budget_stops_the_run_between_days(tmp_path):
    ticks = iter(range(0, 10_000, 100))
    out = run(tmp_path, budget_s=250, clock=lambda: next(ticks))
    assert out["stopped"] == "time budget" and 1 <= out["read"] <= 3


def test_a_refused_old_day_is_recorded_and_a_refused_new_day_is_asked_again(tmp_path):
    calls = []
    out = run(tmp_path, build=fake_vendor(refuse={"2024-11-24", "2026-11-25"}, calls=calls))
    assert out["read"] == 730 and out["unavailable_new"] == 1 and out["not_published"] == 1
    assert out["days_unavailable"] == 1 and out["oldest_unavailable"] == "2024-11-24"
    assert out["pending"] == 1 and out["backlog"] == 0 and out["status"] == "ok"
    assert not os.path.exists(tmp_path / "root" / "days" / "2026-11-25.json")
    calls.clear()
    again = run(tmp_path, build=fake_vendor(calls=calls))
    assert [c[0] for c in calls] == ["2026-11-25"]      # the cold-storage day is never asked twice
    assert again["read"] == 1 and again["pending"] == 0


def test_three_unreadable_days_in_a_row_stop_the_run_and_it_reads_error(tmp_path):
    calls = []
    fail = {"2024-11-25", "2024-11-26", "2024-11-27"}
    out = run(tmp_path, build=fake_vendor(fail=fail, calls=calls))
    assert [c[0] for c in calls] == ["2024-11-24", "2024-11-25", "2024-11-26", "2024-11-27"]
    assert out["status"] == "error" and out["errors"] == 3 and out["read"] == 1
    assert out["stopped"] == "3 days in a row could not be read"
    assert out["last_error"].startswith("2024-11-27: URLError")
    assert out["backlog"] > 0 and out["events"] is None


def test_when_every_day_is_held_the_panel_and_the_event_list_are_written_and_compared(tmp_path):
    # one underlier, 1,000 USD a day, 5,000,000 on a Tuesday in 2026: one event
    out = run(tmp_path, build=fake_vendor(usd={"2026-06-09": 5_000_000}))
    assert out["read"] == 732 and out["pending"] == 0 and out["stopped"] is None
    ev = out["events"]
    assert ev["events"] == 1 and ev["first_testable"] == "2025-02-25"
    assert ev["registered"]["match"] is False and ev["registered"]["here"] == 1
    assert ev["registered"]["registered"] == 27869
    # the unequal list is reported; the run itself is not an error
    assert out["status"] == "ok"
    root = tmp_path / "root"
    doc = panel.read_events(str(root / "panel"))
    assert [(e["cusip"], e["date"], e["known"]) for e in doc["events"]] == [("037833100", "2026-06-09", "2026-06-10")]
    with open(root / "panel" / "manifest.json") as f:
        assert json.load(f)["days_ok"] == 732
    # a run with nothing new keeps the list and reports the kept check
    calls = []
    again = run(tmp_path, build=fake_vendor(calls=calls))
    assert calls == [] and again["read"] == 0 and again["events"] == ev


def test_gzipped_day_files_compact_to_the_same_bytes_as_plain_ones(tmp_path):
    plain, zipped = str(tmp_path / "plain"), str(tmp_path / "zipped")
    for work in (plain, zipped):
        os.makedirs(os.path.join(work, "days"))
        for day in ("2025-06-02", "2025-06-03"):
            fake_vendor(usd={"2025-06-03": 7})(day, work)
    for day in ("2025-06-02", "2025-06-03"):
        job.gzip_day_file(os.path.join(zipped, "days"), day)
        job.gzip_day_file(os.path.join(zipped, "days"), day)        # a second call is harmless
    assert sorted(n for n in os.listdir(os.path.join(zipped, "days")) if "csv" in n) == [
        "2025-06-02.csv.gz", "2025-06-03.csv.gz"]
    panel.compact(plain, str(tmp_path / "a"))
    panel.compact(zipped, str(tmp_path / "b"))
    for name in ("panel_2025Q2.csv.gz", "manifest.json"):
        with open(tmp_path / "a" / name, "rb") as fa, open(tmp_path / "b" / name, "rb") as fb:
            assert fa.read() == fb.read()
    with gzip.open(tmp_path / "b" / "panel_2025Q2.csv.gz", "rt", newline="") as f:
        assert [r["newt_usd"] for r in csv.DictReader(f)] == ["1000", "7"]


def test_a_zip_left_by_a_killed_run_is_removed_from_the_temporary_directory(tmp_path):
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    (tmp / "tmpabc.zip").write_bytes(b"x" * 10)
    run(tmp_path, max_days=1)
    assert os.listdir(tmp) == []


def test_main_prints_one_json_line_and_an_exception_reads_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(job, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("volume gone")))
    assert job.main(["--run", "--out-root", str(tmp_path)]) == 1
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line == {"status": "error", "reason": "OSError: volume gone"}


def test_build_day_holds_the_zip_where_it_is_told(tmp_path, monkeypatch):
    seen = {}

    def no_network(req, timeout=0):
        raise panel.urllib.error.HTTPError(req.full_url, 403, "Forbidden", None, None)
    real = panel.tempfile.mkstemp

    def spy(suffix="", dir=None):
        seen["dir"] = dir
        return real(suffix=suffix, dir=dir)
    monkeypatch.setattr(panel.urllib.request, "urlopen", no_network)
    monkeypatch.setattr(panel.tempfile, "mkstemp", spy)
    work, tmp = tmp_path / "w", tmp_path / "t"
    (work / "days").mkdir(parents=True)
    tmp.mkdir()
    line = panel.build_day("2024-11-20", str(work), str(tmp))
    assert line["status"] == "unavailable" and seen["dir"] == str(tmp)
    assert os.listdir(tmp) == []                        # the zip is removed whatever the answer


# --- the security-type step (open_questions.md (as2) step 1, 2026-11-26) ---

def _whole_panel(tmp_path, cusips_by_day):
    """A two-year panel with every day held; `cusips_by_day` adds rows."""
    def build(day_iso, work, tmp_dir=None):
        per = {"037833100": dict(dict.fromkeys(COUNTERS, 0), newt_n=1,
                                 newt_usd=5_000_000 if day_iso == "2026-06-09" else 1000)}
        for c in cusips_by_day.get(day_iso, cusips_by_day.get("*", ())):
            per[c] = dict(dict.fromkeys(COUNTERS, 0), modi_n=1)
        days = os.path.join(work, "days")
        panel.write_day(os.path.join(days, day_iso + ".csv"), dt.date.fromisoformat(day_iso), per)
        line = {"date": day_iso, "status": "ok", "underliers": len(per), "rows": len(per), "us_rows": len(per)}
        with open(os.path.join(days, day_iso + ".json"), "w") as f:
            json.dump(line, f)
        return line
    return run(tmp_path, build=build)


def test_the_cusips_to_ask_are_the_event_underliers_first_then_the_rest_by_panel_days(tmp_path):
    _whole_panel(tmp_path, {"*": ("78462F103",), "2026-06-01": ("78462F103", "000000AA1"),
                            "2026-06-02": ("78462F103", "000000AA1", "000000BB2")})
    root = str(tmp_path / "root")
    doc = job.figi_wanted(root)
    # the one event underlier, then 732 days, 2 days, 1 day
    assert doc["event_cusips"] == 1
    assert doc["cusips"] == ["037833100", "78462F103", "000000AA1", "000000BB2"]
    # kept: a second call does not read the panel again
    real = panel.read_panel
    try:
        panel.read_panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("panel read twice"))
        assert job.figi_wanted(root) == doc
    finally:
        panel.read_panel = real


def test_the_type_step_gets_what_the_day_reads_left_and_only_once_an_event_list_exists(tmp_path):
    seen = []

    def figi(root, budget_s):
        seen.append(round(budget_s))
        return {"status": "ok", "wanted": 2, "held": 2, "owed": 0, "asked": 2}
    # days still owed: no event list, so nothing is asked
    out = run(tmp_path, max_days=3, total_s=840.0, figi=figi)
    assert seen == [] and out["figi"] is None and out["backlog"] > 0
    # total_s 0 (the default) is the step switched off
    out = run(tmp_path, figi=figi)
    assert seen == [] and out["figi"] is None and out["events"]["events"] == 0
    # whole panel: the step runs with the run's remaining seconds
    t = {"now": 1000.0}

    def clock():
        t["now"] += 50.0
        return t["now"]
    out = run(tmp_path, total_s=840.0, figi=figi, clock=clock)
    assert len(seen) == 1 and 0 < seen[0] < 840 and out["figi"]["held"] == 2 and out["status"] == "ok"
    with open(tmp_path / "root" / "last_run.json") as f:
        assert json.load(f)["figi"]["asked"] == 2
    # too little time left: it waits for the next run
    out = run(tmp_path, total_s=60.0, figi=figi, clock=clock)
    assert len(seen) == 1 and out["figi"] is None


def test_a_type_step_that_got_no_answer_makes_the_run_read_error_and_keeps_the_panel(tmp_path):
    _whole_panel(tmp_path, {})
    root = str(tmp_path / "root")

    def refused(cusips, cache_path, budget_s, api_key=None):
        return {"status": "error", "wanted": len(cusips), "held": 0, "owed": len(cusips), "asked": 0,
                "last_error": "HTTP 403: forbidden"}
    step = job.figi_step(root, 100.0, resolve=refused)
    assert step["status"] == "error" and step["event_cusips"] == 1 and step["event_cusips_held"] == 0
    out = run(tmp_path, total_s=840.0, figi=lambda r, b: step)
    assert out["status"] == "error" and out["events"]["events"] == 1 and out["days_ok"] == 732
    # an exception inside the step is a reason on the row, never a lost run
    boom = job.figi_step(root, 100.0, resolve=lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert boom == {"status": "error", "last_error": "OSError: disk full"}


def test_the_type_step_counts_classes_of_what_it_holds(tmp_path):
    import cusip_figi
    _whole_panel(tmp_path, {"*": ("78462F103",)})
    root = str(tmp_path / "root")

    def resolve(cusips, cache_path, budget_s, api_key=None):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(json.dumps({"cusip": "037833100", "found": True, "us": True, "market_sector": "Equity",
                                "security_type": "Common Stock"}) + "\n")
            f.write(json.dumps({"cusip": "78462F103", "found": True, "us": True, "market_sector": "Equity",
                                "security_type": "ETP"}) + "\n")
        return {"status": "ok", "wanted": 2, "held": 2, "owed": 0, "asked": 2}
    step = job.figi_step(root, 100.0, resolve=resolve)
    assert step["classes_event_cusips"] == {"common": 1} and step["classes_all"] == {"ETP": 1, "common": 1}
    assert step["event_cusips"] == 1 and step["event_cusips_held"] == 1
    assert os.path.basename(os.path.dirname(os.path.join(root, job.FIGI_SUBDIR, cusip_figi.CACHE_FILE))) == "figi"


# --- the dated ticker record (open_questions.md (at2), 2026-11-26) ---

def test_the_sec_files_are_read_before_the_type_step_within_their_own_cap_and_only_once_an_event_list_exists(tmp_path):
    order = []

    def fetch(root, budget_s):
        order.append(("ftd", round(budget_s)))
        return {"status": "ok", "wanted": 46, "held": 44, "owed": 2, "read": 44}

    def figi(root, budget_s):
        order.append(("figi", round(budget_s)))
        return {"status": "ok", "wanted": 2, "held": 2, "owed": 0, "asked": 0}
    cov = lambda root: {"groups": {}, "window_days": 45}  # noqa: E731
    # days still owed: no event list, so no SEC file is asked for
    out = run(tmp_path, max_days=3, total_s=840.0, figi=figi, ftd_fetch=fetch, ftd_coverage=cov)
    assert order == [] and out["ftd"] is None
    # total_s 0 (the default) is both steps switched off
    out = run(tmp_path, figi=figi, ftd_fetch=fetch, ftd_coverage=cov)
    assert order == [] and out["ftd"] is None
    t = {"now": 1000.0}

    def clock():
        t["now"] += 50.0
        return t["now"]
    out = run(tmp_path, total_s=840.0, figi=figi, ftd_fetch=fetch, ftd_coverage=cov, clock=clock)
    assert [o[0] for o in order] == ["ftd", "figi"]
    assert order[0][1] == job.FTD_BUDGET_S and 0 < order[1][1] < 840        # capped; the type step has the rest
    assert out["ftd"]["held"] == 44 and out["ftd"]["coverage"] == {"groups": {}, "window_days": 45} and out["status"] == "ok"
    with open(tmp_path / "root" / "last_run.json") as f:
        assert json.load(f)["ftd"]["owed"] == 2
    # little time left: the SEC step takes what is left less the type step's minimum, or waits
    order.clear()
    out = run(tmp_path, total_s=200.0, figi=figi, ftd_fetch=fetch, ftd_coverage=cov, clock=clock)
    assert order[0][0] == "ftd" and order[0][1] < job.FTD_BUDGET_S
    order.clear()
    out = run(tmp_path, total_s=80.0, figi=figi, ftd_fetch=fetch, ftd_coverage=cov, clock=clock)
    assert order == [] and out["ftd"] is None
    assert job.FTD_BUDGET_S + job.FIGI_MIN_BUDGET_S < 840 - 600                 # both fit after a full day-read budget


def test_a_refused_sec_step_makes_the_run_read_error_keeps_the_panel_and_counts_nothing(tmp_path):
    _whole_panel(tmp_path, {})
    refused = lambda root, b: {"status": "error", "wanted": 46, "held": 0, "owed": 46, "read": 0,  # noqa: E731
                               "last_error": "202501a: HTTP 403: the SEC refused the client or the rate"}
    called = []
    out = run(tmp_path, total_s=840.0, figi=lambda r, b: {"status": "ok"}, ftd_fetch=refused,
              ftd_coverage=lambda r: called.append(r) or {})
    assert out["status"] == "error" and out["events"]["events"] == 1 and out["days_ok"] == 732
    assert called == [] and "coverage" not in out["ftd"]           # nothing held: nothing to count
    # an exception inside either step is a reason on the row, never a lost run
    root = str(tmp_path / "root")
    boom = job.ftd_fetch_step(root, 100.0, build=lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert boom == {"status": "error", "last_error": "OSError: disk full"}
    os.makedirs(os.path.join(root, job.FTD_SUBDIR), exist_ok=True)
    with open(os.path.join(root, job.FTD_SUBDIR, "202606a.csv.gz"), "wb") as f:
        f.write(b"not gzip")
    assert job.ftd_coverage_step(root)["groups"]["kept_us"]["events"] == 0      # an unreadable file is left out, not fatal


def test_the_real_steps_ask_for_the_events_windows_and_count_what_the_volume_holds(tmp_path):
    import cusip_figi
    import cusip_ftd
    _whole_panel(tmp_path, {})                    # one event: 037833100 on 2026-06-09
    root = str(tmp_path / "root")
    seen = {}

    def build(ftd_root, wanted, today, budget_s):
        seen.update(root=ftd_root, wanted=list(wanted), budget=budget_s)
        cusip_ftd.write_period(ftd_root, "202606a", {("037833100", "AAPL"): ["2026-06-01", "2026-06-12", 9]})
        return {"status": "ok", "wanted": len(wanted), "held": 1, "owed": len(wanted) - 1, "read": 1}
    step = job.ftd_fetch_step(root, 123.0, today=TODAY, build=build)
    assert seen["root"] == os.path.join(root, "ftd") and seen["budget"] == 123.0
    assert seen["wanted"] == cusip_ftd.wanted_periods(dt.date(2026, 6, 9), dt.date(2026, 6, 9), TODAY)
    assert seen["wanted"][0] == "202604b" and seen["wanted"][-1] == "202607b" and step["read"] == 1
    assert 0 < step["bytes"] < 200                 # the record's size on the volume is reported
    os.makedirs(os.path.join(root, job.FIGI_SUBDIR), exist_ok=True)
    with open(os.path.join(root, job.FIGI_SUBDIR, cusip_figi.CACHE_FILE), "w") as f:
        f.write(json.dumps({"cusip": "037833100", "found": True, "us": True, "market_sector": "Equity",
                            "security_type": "Common Stock", "ticker": "AAPL"}) + "\n")
    cov = job.ftd_coverage_step(root)
    assert cov["groups"]["kept_us"] == {"underliers": 1, "underliers_joined": 1, "underliers_joined_back": 1,
                                        "events": 1, "events_joined": 1, "events_joined_back": 1}
    assert cov["agreement_kept_us"]["same"] == 1 and "037833100" not in json.dumps(cov)
    # the whole run, with the real coverage step, writes it into last_run.json
    out = run(tmp_path, total_s=840.0, figi=lambda r, b: {"status": "ok"},
              ftd_fetch=lambda r, b: job.ftd_fetch_step(r, b, today=TODAY, build=build))
    assert out["ftd"]["coverage"]["groups"]["kept_us"]["events_joined"] == 1 and out["status"] == "ok"
