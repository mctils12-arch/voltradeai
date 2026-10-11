"""test_sentinel2_refresh.py — pure pieces of scripts/sentinel2_refresh.py."""
import importlib.util
import json
import os
from datetime import date

_spec = importlib.util.spec_from_file_location(
    "sentinel2_refresh", os.path.join(os.path.dirname(__file__), "scripts", "sentinel2_refresh.py"))
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)


def test_since_for_subtracts_overlap_and_falls_back():
    assert sr.since_for(date(2026, 9, 22)) == "2026-09-17"
    assert sr.since_for(None) == "2026-06-20"


def test_newest_date_reads_max_and_tolerates_junk(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"date": "2026-09-01"}) + "\nnot json\n" + json.dumps({"date": "2026-10-03"}) + "\n")
    assert sr.newest_date(str(p)) == date(2026, 10, 3)
    assert sr.newest_date(str(p), oldest=True) == date(2026, 9, 1)
    assert sr.newest_date(str(tmp_path / "missing.jsonl")) is None


def test_plan_order_rehydrate_before_s1_rebuild_and_all_scripts_exist():
    plan = sr.build_plan("py", "a", "b", "c")
    labels = [l for l, _ in plan]
    assert labels.index("v3 chips (rehydrate)") < labels.index("v3 estimator")
    assert "--redownload-missing" in dict(plan)["v3 chips (rehydrate)"]
    for _, argv in plan:
        assert os.path.exists(argv[1]), argv[1]


def test_pick_python_first_capable_else_none():
    assert sr.pick_python(("a", "b"), ok=lambda c: c == "b") == "b"
    assert sr.pick_python(("a",), ok=lambda c: False) is None
