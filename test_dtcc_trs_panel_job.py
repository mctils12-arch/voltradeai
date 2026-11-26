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
