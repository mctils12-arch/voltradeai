"""
2026-11-24, KNOWN BROKEN #60 recurrence (research/open_questions.md (al3)).

Every scan overwrote the one timing file, so a scan that ran past the daemon's
300 s left no timings once the next scan began, and the length of the steps
after the options stage was never on record. scan_timing_log keeps one line
per scan by UTC day.

Pins:
  - a finished scan's line: its total, where the options stage ended, what
    followed it, the deadline's counts, whether the CSP scores were recomputed;
  - a scan that never finished is kept as unfinished with the last phase it
    reached (the row a timeout used to erase);
  - a file with no scan in it writes nothing; a broken file raises nothing;
  - lines append by the scan's own UTC day; old day files are pruned;
  - bot_engine keeps the previous scan BEFORE it clears the timing file.
"""
import inspect
import json
import os

import scan_timing_log as stl

START = 1795543786.0  # 2026-11-24T18:09:46Z


def _completed(total=262.9, cold=False):
    return {
        "scan_started": START,
        "status": "completed",
        "total_sec": total,
        "phases": [
            {"phase": "universe_load", "duration_sec": 0.13, "total_sec": 0.13, "rss_mb": 581},
            {"phase": "deep_score", "duration_sec": 41.3, "total_sec": 48.2, "rss_mb": 590},
            {"phase": "options_stage_deadline", "deadline_sec": 240.0, "deadline_given": True,
             "candidates": 412, "not_reached": 37, "first_not_reached": ["AAA", "BBB"], "stage_secs": 158.7},
            {"phase": "step6_trade_loop_and_options", "duration_sec": 192.4, "total_sec": 240.9, "rss_mb": 600},
            {"phase": "step10_spy_floor", "duration_sec": 17.0, "total_sec": 257.9, "rss_mb": 600},
            {"phase": "tier_engine_start", "duration_sec": 0.0, "total_sec": 257.9, "rss_mb": 600},
            {"phase": "tier_engine_tier1", "duration_sec": 3.1, "total_sec": 261.0, "rss_mb": 600},
            {"phase": "tier_engine_breakdown", "tier_timings": {
                "tier1_sec": 3.1, "tier2_sec": 0.2, "total_sec": 4.4, "t1_earnings_coverage": {"x": 1},
                "csp_layer2_prefetch": {"cache_hit": not cold, "elapsed_sec": 33.4 if cold else 0.0,
                                        "completed": 150 if cold else 0, "total": 150 if cold else 0}}},
            {"phase": "scan_complete", "duration_sec": 1.9, "total_sec": total, "rss_mb": 600},
        ],
    }


def test_a_finished_scan_becomes_one_line_with_what_followed_the_options_stage():
    row = stl.summarize(_completed(total=262.9, cold=True))
    assert row["status"] == "completed" and row["total_sec"] == 262.9
    assert row["day"] == "2026-11-24"
    assert row["options_stage_end_sec"] == 240.9
    assert row["tier_engine_start_sec"] == 257.9
    assert row["after_options_sec"] == 22.0          # 262.9 - 240.9
    assert row["options"] == {"deadline_sec": 240.0, "candidates": 412, "not_reached": 37, "loop_sec": 158.7}
    assert row["options_outside_loop_sec"] == 33.7   # 192.4 - 158.7: the part of the step no deadline covers
    assert row["layer2"] == {"cache_hit": False, "elapsed_sec": 33.4, "completed": 150, "total": 150}
    assert row["tier"] == {"tier1_sec": 3.1, "tier2_sec": 0.2, "total_sec": 4.4}   # seconds only, no nested records
    assert row["phase_sec"]["step6_trade_loop_and_options"] == 192.4
    assert "options_stage_deadline" not in row["phase_sec"]                        # it has no duration
    assert "first_not_reached" not in json.dumps(row)                              # counts, not names
    assert "rss_mb" not in json.dumps(row)


def test_a_scan_that_never_finished_is_kept_with_the_phase_it_reached():
    t = _completed()
    t["status"] = "in_progress"
    del t["total_sec"]
    t["phases"] = t["phases"][:6]                    # stopped after tier_engine_start, as the 300 s timeouts did
    row = stl.summarize(t)
    assert row["status"] == "unfinished" and row["total_sec"] is None
    assert row["last_phase"] == "tier_engine_start" and row["reached_sec"] == 257.9
    assert row["after_options_sec"] == 17.0          # to the last phase reached, not to an end it never had
    assert row["layer2"] == {} and row["tier"] == {}


def test_a_scan_that_stopped_before_the_options_stage_has_no_after_options_figure():
    t = _completed()
    t["status"] = "in_progress"
    t["phases"] = t["phases"][:2]
    row = stl.summarize(t)
    assert row["options_stage_end_sec"] is None and row["after_options_sec"] is None
    assert row["options_outside_loop_sec"] is None
    assert row["last_phase"] == "deep_score"


def test_a_file_with_no_scan_in_it_is_not_a_row():
    assert stl.summarize({"scan_started": START, "phases": [], "status": "in_progress"}) is None
    assert stl.summarize({"phases": [{"phase": "x", "duration_sec": 1, "total_sec": 1}]}) is None
    assert stl.summarize({"scan_started": START, "phases": [{"phase": "options_stage_deadline"}]}) is None
    assert stl.summarize([]) is None


def test_roll_previous_appends_by_the_scans_own_day_and_survives_a_broken_file(tmp_path, capsys):
    timing = tmp_path / "voltrade_scan_timings.json"
    assert stl.roll_previous(str(timing), str(tmp_path)) is None           # no file yet: nothing written
    assert not (tmp_path / stl.LOG_DIR_NAME).exists()

    timing.write_text(json.dumps(_completed(total=250.0)))
    assert stl.roll_previous(str(timing), str(tmp_path))["total_sec"] == 250.0
    timing.write_text(json.dumps(_completed(total=301.2)))
    stl.roll_previous(str(timing), str(tmp_path))
    day = tmp_path / stl.LOG_DIR_NAME / "2026-11-24.jsonl"
    lines = [json.loads(x) for x in day.read_text().splitlines()]
    assert [x["total_sec"] for x in lines] == [250.0, 301.2]

    timing.write_text('{"scan_started": 17955')                            # cut mid-write
    assert stl.roll_previous(str(timing), str(tmp_path)) is None
    assert "previous scan not recorded" in capsys.readouterr().err         # said, not swallowed
    assert len(day.read_text().splitlines()) == 2


def test_a_new_day_prunes_day_files_older_than_the_keep(tmp_path):
    log_dir = tmp_path / stl.LOG_DIR_NAME
    log_dir.mkdir()
    (log_dir / "2026-10-20.jsonl").write_text("{}\n")                      # 35 days before
    (log_dir / "2026-10-26.jsonl").write_text("{}\n")                      # 29 days before
    (log_dir / "notes.txt").write_text("kept: not a day file")
    timing = tmp_path / "t.json"
    timing.write_text(json.dumps(_completed()))
    stl.roll_previous(str(timing), str(tmp_path), keep_days=30)
    assert sorted(os.listdir(log_dir)) == ["2026-10-26.jsonl", "2026-11-24.jsonl", "notes.txt"]


def test_bot_engine_keeps_the_previous_scan_before_it_clears_the_timing_file():
    import bot_engine
    src = inspect.getsource(bot_engine._scan_market_inner)
    keep = src.index("roll_previous(_TIMING_PATH, _TIMING_DIR)")
    clear = src.index('"phases": [],')
    assert keep < clear, "the previous scan's timings must be read before the file is overwritten"
