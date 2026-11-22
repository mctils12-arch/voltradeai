"""filings13f_clusters.py: the 13F new-position cluster detector.

Rows are in the archive's own shape (server/edgar13f.ts Filing13F; the first
holding below is a row production served on 2026-11-22).
"""
import gzip
import json
import os
import subprocess
import sys

import filings13f_clusters as fc

REPO = os.path.dirname(os.path.abspath(__file__))
Q2, Q3 = "2026-06-30", "2026-09-30"


def holding(cusip, issuer="THIN CO INC", value=1000, **kw):
    h = {"issuer": issuer, "titleOfClass": "COM", "cusip": cusip, "value": value, "shares": 10,
         "sharesType": "SH", "putCall": None, "discretion": "SOLE"}
    h.update(kw)
    return h


def filing(mgr, period, filed, holdings, acc=None, **kw):
    f = {"managerCik": f"{mgr:010d}", "managerName": f"Manager {mgr}", "periodOfReport": period,
         "submissionType": "13F-HR", "entryTotal": len(holdings or []), "valueTotal": 1,
         "otherManagersCount": 0, "holdings": holdings, "holdingsOmitted": holdings is None,
         "accession": acc or f"{mgr:010d}-26-{period[5:7]}{filed[8:]}", "filedAt": filed, "cik": f"{mgr:010d}"}
    f.update(kw)
    return f


CME = holding("12572Q105", "CME Group Inc", 7859400)  # as served by production
THIN = "88888A101"
OTHER = holding("99999B102", "SOMETHING ELSE CORP")


def pairs_of(rows):
    counts = fc.new_counts()
    return fc.detect(*fc.load_holdings(iter(rows), counts)), counts


def base_and_entries(n_new, prior=1, filed_days=("2026-11-10", "2026-11-12", "2026-11-13", "2026-11-16")):
    rows = []
    for m in range(1, n_new + 1):  # managers who did not hold THIN in Q2 and do in Q3
        rows.append(filing(m, Q2, "2026-08-10", [CME]))
        rows.append(filing(m, Q3, filed_days[m - 1], [CME, holding(THIN, value=100 * m)]))
    for m in range(100, 100 + prior):  # managers who held it before and still do
        rows.append(filing(m, Q2, "2026-08-11", [holding(THIN)]))
        rows.append(filing(m, Q3, "2026-11-09", [holding(THIN)]))
    return rows


def test_prev_quarter_end():
    assert fc.prev_quarter_end("2026-09-30") == "2026-06-30"
    assert fc.prev_quarter_end("2026-03-31") == "2025-12-31"
    assert fc.prev_quarter_end("2026-12-31") == "2026-09-30"


def test_three_new_entrants_make_a_cluster_dated_by_the_third_filing():
    pairs, _ = pairs_of(base_and_entries(3, prior=2))
    assert len(pairs) == 1
    p = pairs[0]
    assert (p["period"], p["base_period"]) == (Q3, Q2)
    assert p["managers"] == {"base": 5, "period": 5, "both": 5, "period_without_base": 0,
                             "base_filed_late_ignored": 0}
    assert p["clusters"] == 1 and p["clusters_by_class"] == {"thin": 1, "no_prior": 0, "broad": 0}
    e = p["events"][0]
    assert e["cusip"] == THIN and e["new_entrants"] == 3 and e["prior_holders"] == 2 and e["class"] == "thin"
    # the event is public the day the THIRD entrant filed, not the first
    assert e["event_date"] == "2026-11-13"
    assert e["first_entry_filed"] == "2026-11-10" and e["last_entry_filed"] == "2026-11-13"
    assert e["new_value_usd"] == 100 + 200 + 300


def test_two_new_entrants_are_not_a_cluster_and_are_a_thin_near_miss():
    pairs, _ = pairs_of(base_and_entries(2))
    p = pairs[0]
    assert p["clusters"] == 0 and p["securities_with_a_new_entry"] == 1
    assert p["thin_near_misses"] == 1
    # dated by its LAST new entry: nothing later is known about it
    assert p["controls"] == [{"cusip": THIN, "issuer": "THIN CO INC", "date": "2026-11-12",
                              "new_entrants": 2, "prior_holders": 1}]


def test_near_misses_are_thin_only_and_never_also_events():
    assert pairs_of(base_and_entries(2, prior=0))[0][0]["controls"] == []   # no prior holder
    assert pairs_of(base_and_entries(2, prior=11))[0][0]["controls"] == []  # broadly held
    p = pairs_of(base_and_entries(3))[0][0]
    assert p["clusters"] == 1 and p["controls"] == []


def test_a_fourth_entrant_does_not_move_the_event_date():
    e = pairs_of(base_and_entries(4))[0][0]["events"][0]
    assert e["new_entrants"] == 4 and e["event_date"] == "2026-11-13" and e["last_entry_filed"] == "2026-11-16"


def test_a_manager_without_a_base_filing_is_never_a_new_entrant():
    rows = base_and_entries(2)
    rows.append(filing(50, Q3, "2026-11-12", [holding(THIN)]))  # first filing we hold from this manager
    p = pairs_of(rows)[0][0]
    assert p["clusters"] == 0
    assert p["managers"]["period_without_base"] == 1


def test_a_summary_only_base_is_no_base():
    rows = base_and_entries(2)
    rows.append(filing(50, Q2, "2026-08-12", None))  # more than 250 rows: table not archived
    rows.append(filing(50, Q3, "2026-11-12", [holding(THIN)]))
    pairs, counts = pairs_of(rows)
    assert pairs[0]["clusters"] == 0 and counts["summary_only"] == 1


def test_classes_no_prior_thin_broad():
    assert fc.classify(0) == "no_prior"
    assert fc.classify(1) == "thin" and fc.classify(fc.THIN_PRIOR_MAX) == "thin"
    assert fc.classify(fc.THIN_PRIOR_MAX + 1) == "broad"
    assert pairs_of(base_and_entries(3, prior=0))[0][0]["clusters_by_class"]["no_prior"] == 1
    assert pairs_of(base_and_entries(3, prior=11))[0][0]["clusters_by_class"]["broad"] == 1


def test_a_cusip_change_is_not_a_purchase():
    rows = []
    for m in (1, 2, 3):
        # same issuer number, new issue number (a new share class or reissue)
        rows.append(filing(m, Q2, "2026-08-10", [holding("88888A200")]))
        rows.append(filing(m, Q3, "2026-11-12", [holding(THIN)]))
    for m in (4, 5, 6):
        # a wholly new CUSIP, same company name up to punctuation and suffix
        rows.append(filing(m, Q2, "2026-08-10", [holding("77777C100", "Renamed Holdings, Inc.")]))
        rows.append(filing(m, Q3, "2026-11-12", [holding("66666D100", "RENAMED HLDGS INC")]))
    assert pairs_of(rows)[0][0]["clusters"] == 0


def test_options_and_principal_amounts_are_not_positions():
    rows = []
    for m in (1, 2, 3):
        rows.append(filing(m, Q2, "2026-08-10", [CME]))
        rows.append(filing(m, Q3, "2026-11-12", [CME, holding(THIN, putCall="Call"),
                                                    holding("55555E109", sharesType="PRN"), holding("BAD")]))
    pairs, counts = pairs_of(rows)
    assert pairs[0]["clusters"] == 0 and counts["rows_not_positions"] == 9


def test_a_base_filed_after_the_quarter_ended_is_not_a_base():
    rows = base_and_entries(2)
    rows.append(filing(3, Q2, "2026-10-05", [CME]))  # Q2 filed in October: after Q3 ended
    rows.append(filing(3, Q3, "2026-11-12", [CME, holding(THIN)]))
    p = pairs_of(rows)[0][0]
    assert p["clusters"] == 0 and p["managers"]["base_filed_late_ignored"] == 1


def test_an_amendment_adds_to_the_period_and_dates_its_own_rows():
    rows = base_and_entries(2)
    rows.append(filing(3, Q2, "2026-08-10", [CME]))
    rows.append(filing(3, Q3, "2026-11-12", [CME]))
    rows.append(filing(3, Q3, "2026-11-19", [holding(THIN)], acc="amend-1", submissionType="13F-HR/A"))
    e = pairs_of(rows)[0][0]["events"][0]
    assert e["new_entrants"] == 3 and e["event_date"] == "2026-11-19"


def test_an_amended_base_counts_as_held_before():
    rows = base_and_entries(2)
    rows.append(filing(3, Q2, "2026-08-10", [CME]))
    rows.append(filing(3, Q2, "2026-08-20", [holding(THIN)], acc="amend-2", submissionType="13F-HR/A"))
    rows.append(filing(3, Q3, "2026-11-12", [CME, holding(THIN)]))
    assert pairs_of(rows)[0][0]["clusters"] == 0


def test_duplicate_accessions_and_bad_rows_are_counted_not_used():
    rows = base_and_entries(3)
    rows.append(dict(rows[1]))                                    # the same filing archived twice
    rows.append(filing(9, "2026-08-15", "2026-09-01", [CME]))      # not a quarter end
    rows.append(filing(9, Q3, "2026-09-15", [CME], acc="early"))   # filed before the period ended
    rows.append({"accession": "x", "holdings": [CME]})             # no manager, no period
    pairs, counts = pairs_of(rows)
    assert pairs[0]["events"][0]["new_entrants"] == 3
    assert (counts["duplicate_accessions"], counts["not_quarter_end"], counts["filed_before_period_end"],
            counts["no_manager_or_period"]) == (1, 1, 1, 1)


def test_one_security_on_several_rows_of_a_filing_is_one_position_with_the_summed_value():
    rows = base_and_entries(2)
    rows.append(filing(3, Q2, "2026-08-10", [CME]))
    rows.append(filing(3, Q3, "2026-11-12", [holding(THIN, value=5), holding(THIN, value=7)]))
    e = pairs_of(rows)[0][0]["events"][0]
    assert e["new_entrants"] == 3 and e["new_value_usd"] == 100 + 200 + 12


def test_exits_and_holders_without_a_base_are_recorded():
    rows = base_and_entries(3, prior=1)
    rows.append(filing(200, Q2, "2026-08-10", [holding(THIN)]))
    rows.append(filing(200, Q3, "2026-11-12", [CME]))            # held it, holds it no longer
    rows.append(filing(201, Q3, "2026-11-12", [holding(THIN)]))  # no base
    e = pairs_of(rows)[0][0]["events"][0]
    assert e["exits"] == 1 and e["holders_without_base"] == 1 and e["prior_holders"] == 2


def _write_archive(tmp_path, rows, days=("2026-08-11", "2026-11-13")):
    d = tmp_path / "datacore_archive" / "filings13f"
    d.mkdir(parents=True)
    half = len(rows) // 2
    with gzip.open(d / f"{days[0]}.jsonl.gz", "wt") as f:
        f.write("\n".join(json.dumps(r) for r in rows[:half]) + "\n{not json\n")
    (d / f"{days[1]}.jsonl").write_text("\n".join(json.dumps(r) for r in rows[half:]) + "\n")
    (d / "voltrade_13f_day_ledger.json").write_text("{}")  # not a day file
    return str(d)


def test_run_writes_the_files_and_first_seen_is_append_only(tmp_path):
    d = _write_archive(tmp_path, base_and_entries(3))
    out = str(tmp_path / "out")
    import datetime as dt
    today = dt.date(2026, 11, 22)
    r = fc.run(out, d, "1.2.3", 700, today=today)
    assert r["status"] == "ok" and r["thin_clusters"] == 1 and r["first_seen_appended"] == 1
    latest = json.loads(open(os.path.join(out, "latest.json")).read())
    assert latest["schema"] == fc.SCHEMA and latest["code_version"] == "1.2.3"
    assert latest["archive"]["day_files"] == 2 and latest["archive"]["malformed_lines"] == 1
    assert latest["archive"]["first_day"] == "2026-08-11" and latest["archive"]["last_day"] == "2026-11-13"
    assert latest["periods_full_filings"] == {Q2: 4, Q3: 4}
    assert latest["newest_pair_top_thin"][0]["cusip"] == THIN
    assert "events" not in latest["pairs"][0]
    assert latest["definitions"]["min_new_entrants"] == 3 and latest["definitions"]["thin_prior_max"] == 10
    events = json.loads(open(os.path.join(out, f"events_{Q3}.json")).read())
    assert len(events["events"]) == 1 and "controls" not in events
    controls = json.loads(open(os.path.join(out, f"controls_{Q3}.json")).read())
    assert controls["period"] == Q3 and controls["controls"] == []
    assert "controls" not in latest["pairs"][0] and latest["pairs"][0]["thin_near_misses"] == 0
    first = open(os.path.join(out, "first_seen.jsonl")).read().strip().split("\n")
    assert len(first) == 1 and json.loads(first[0])["archive_last_day"] == "2026-11-13"
    # a second run over the same archive detects the same event and appends nothing
    r2 = fc.run(out, d, "1.2.4", 700, today=today)
    assert r2["first_seen_appended"] == 0
    again = open(os.path.join(out, "first_seen.jsonl")).read().strip().split("\n")
    assert again == first, "first_seen.jsonl keeps the day an event was first detected"
    assert json.loads(open(os.path.join(out, "last_run.json")).read())["status"] == "ok"
    # a line cut short by a restart is skipped and said, and the event is appended again
    with open(os.path.join(out, "first_seen.jsonl"), "w") as f:
        f.write(first[0][:40] + "\n")
    assert fc.run(out, d, "1.2.5", 700, today=today)["first_seen_appended"] == 1


def test_old_day_files_are_not_read_and_a_pair_without_its_whole_base_season_is_left_alone(tmp_path):
    import datetime as dt
    d = _write_archive(tmp_path, base_and_entries(3))  # day files 2026-08-11 and 2026-11-13
    out = str(tmp_path / "out")
    assert fc.run(out, d, "1", 700, today=dt.date(2026, 11, 22))["thin_clusters"] == 1
    before = open(os.path.join(out, f"events_{Q3}.json")).read()
    # 250 days after 2026-06-30 is 2027-03-07: from then on the Q2 season is no longer whole in the window
    assert fc.window_start(dt.date(2027, 3, 7)) == Q2
    still = fc.run(out, d, "2", 700, today=dt.date(2027, 3, 7))
    assert still["pairs"] == 1
    later = fc.run(out, d, "3", 700, today=dt.date(2027, 5, 1))
    assert later["status"] == "ok" and later["pairs"] == 0 and later["first_seen_appended"] == 0
    latest = json.loads(open(os.path.join(out, "latest.json")).read())
    assert latest["archive"]["day_files_before_window"] == 1 and latest["window"]["lookback_days"] == 250
    frozen = json.loads(open(os.path.join(out, f"events_{Q3}.json")).read())
    assert frozen["code_version"] == "2", "the pair's file stays as last written inside the window"
    assert frozen["events"] == json.loads(before)["events"]


def test_no_price_and_no_trading_import():
    src = open(os.path.join(REPO, "filings13f_clusters.py"), encoding="utf-8").read()
    for banned in ("backtest_v2", "bot_engine", "alpaca", "fetch_bars", "system_config", "urllib", "requests"):
        assert f"import {banned}" not in src and f"from {banned}" not in src, banned


def test_cli_empty_volume_is_skipped_and_recorded(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["DATA_DIR"] = str(tmp_path)
    r = subprocess.run([sys.executable, os.path.join(REPO, "filings13f_clusters.py"), "--run", "--nice", "19"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    summary = json.loads(r.stdout.strip().split("\n")[-1])
    assert summary["status"] == "skipped" and "filings13f" in summary["reason"]
    last = json.loads((tmp_path / "filings13f_clusters" / "last_run.json").read_text())
    assert last["status"] == "skipped" and last["finished_at"]


def test_cli_over_budget_is_an_error_line_not_a_crash(tmp_path):
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc).date()  # the CLI reads the real clock: day files inside its window
    d = _write_archive(tmp_path, base_and_entries(3),
                       days=((now - dt.timedelta(days=1)).isoformat(), now.isoformat()))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["DATA_DIR"] = str(tmp_path)
    r = subprocess.run([sys.executable, os.path.join(REPO, "filings13f_clusters.py"), "--run",
                        "--archive-dir", d, "--max-rss-mb", "1"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    summary = json.loads(r.stdout.strip().split("\n")[-1])
    assert summary["status"] == "error" and "MemoryError" in summary["reason"]
    assert json.loads((tmp_path / "filings13f_clusters" / "last_run.json").read_text())["status"] == "error"
