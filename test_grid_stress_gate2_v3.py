"""test_grid_stress_gate2_v3.py — gate-2 v3 computation battery
(scripts/grid_stress_gate2_v3.py). Pure-function unit checks plus
committed-artifact coherence: the verdict must be recomputable from the
recorded counts so the conclusion cannot drift from the data (same
precedent as test_grid_stress_gate2.py for v1/v2)."""
import importlib.util
import json
import os
from datetime import date

_spec = importlib.util.spec_from_file_location(
    "grid_stress_gate2_v3",
    os.path.join(os.path.dirname(__file__), "scripts", "grid_stress_gate2_v3.py"))
g3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g3)


def test_scoring_sets_match_source_document_counts():
    # transcribed from research/grid_vision_events_ercot.md's own
    # "Gate-2 v3 scoring sets" section — this test pins the transcription,
    # not the source document, so a future drift in either is visible.
    assert g3.TIER_E_VALID == {"2023-09-06"}
    assert len(g3.TIER_C_VALID) == 13
    assert "2023-09-06" in g3.TIER_C_VALID, "the EEA-2 day also carried a same-day conservation appeal"
    assert g3.TIER_E_CONTAMINATED == {
        "2019-08-13", "2019-08-15",
        "2021-02-15", "2021-02-16", "2021-02-17", "2021-02-18", "2021-02-19",
    }


def test_trailing_mean_series_requires_full_window():
    daily = {(date(2023, 1, 1) - date(2023, 1, 1)).days and "" or f"2023-01-{d:02d}": 100.0
             for d in range(1, 32)}
    # simpler direct construction: 400 consecutive days of constant value
    from datetime import timedelta
    start = date(2023, 1, 1)
    daily = {(start + timedelta(days=i)).isoformat(): 100.0 for i in range(400)}
    tm = g3.trailing_mean_series(daily)
    early = (start + timedelta(days=10)).isoformat()
    late = (start + timedelta(days=399)).isoformat()
    assert tm[early] is None, "fewer than 365 days of history must not produce a trailing mean"
    assert tm[late] == 100.0


def test_trailing_mean_series_detects_a_recent_spike():
    from datetime import timedelta
    start = date(2023, 1, 1)
    daily = {(start + timedelta(days=i)).isoformat(): 100.0 for i in range(400)}
    daily[(start + timedelta(days=399)).isoformat()] = 200.0
    tm = g3.trailing_mean_series(daily)
    late = (start + timedelta(days=399)).isoformat()
    assert 100.0 < tm[late] < 100.3, "one spike day in a 365-day window should barely move the mean"


def test_shift_is_calendar_correct():
    assert g3.shift("2023-02-28", 1) == "2023-03-01"
    assert g3.shift("2023-03-01", -1) == "2023-02-28"


def test_same_month_pool_ignores_none_and_pools_years():
    pool = g3.same_month_pool({"2019-06-01": 1.0, "2023-06-15": 3.0, "2023-06-16": None, "2023-07-01": 9.0})
    assert pool[6] == [1.0, 3.0]
    assert pool[7] == [9.0]


def test_committed_v3_artifact_coherent():
    path = os.path.join(os.path.dirname(__file__), "datacore", "gridvision", "gate2_result_v3.json")
    a = json.load(open(path))
    te, tc, stab = a["tier_e_out_of_sample"], a["tier_c_out_of_sample_lift"], a["no_single_summer_carry"]

    # recall floor: verdict's PASS/FAIL on tier-E must match the recorded detection
    recall_ok = len(te["detected"]) / len(te["event_days"]) >= te["recall_floor"]
    assert te["passed_recall_floor"] == recall_ok

    # lift bar
    assert tc["passed_lift_bar"] == (tc["overall_lift"] is not None and tc["overall_lift"] >= tc["lift_bar"])

    # stability: recomputable from the per-year leave-one-out records
    scoreable = [v for v in stab["per_year_leave_one_out"].values() if "holds" in v]
    assert stab["scoreable_years"] == len(scoreable)
    assert stab["holds"] == (bool(scoreable) and all(v["holds"] for v in scoreable))

    # overall verdict is exactly the conjunction of the three pre-stated criteria
    all_three = te["passed_recall_floor"] and tc["passed_lift_bar"] and stab["holds"]
    assert (a["verdict"] == "PASS") == all_three
    if not te["passed_recall_floor"]:
        assert a["verdict"].startswith("FAIL (recall floor missed")

    # honesty scaffolding must be present, not just the numbers
    assert "multiple_hypothesis_note" in a
    assert "prior_stated_before_run" in a
    assert a["design_contaminated_2019_2022_report_only"]["note"]
