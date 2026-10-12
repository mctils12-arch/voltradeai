"""test_snapshot_refresh.py — scripts/snapshot_refresh.py + the json_object freshness extractor."""
import datetime
import importlib.util
import json
import os


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sr = _load("snapshot_refresh", "scripts/snapshot_refresh.py")
rsc = _load("research_state_check", "scripts/research_state_check.py")


def test_plan_covers_four_scripts_that_exist_and_only_filters():
    plan = sr.build_plan("py")
    assert [k for k, _ in plan] == ["eia", "cpc", "rail", "jodi"]
    for _, argv in plan:
        assert os.path.exists(argv[1]), argv[1]
    assert [k for k, _ in sr.build_plan("py", ["jodi"])] == ["jodi"]


def test_pick_python_first_capable_else_none():
    assert sr.pick_python(("a", "b"), ok=lambda c: c == "b") == "b"
    assert sr.pick_python(("a",), ok=lambda c: False) is None


def test_json_object_extractor(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"built": "2026-10-12T02:38:44+00:00"}))
    assert rsc._max_json_object_date(str(p), "built") == "2026-10-12"
    assert rsc._max_json_object_date(str(p), "nope") is None
    assert rsc._max_json_object_date(str(tmp_path / "missing.json"), "built") is None
    p.write_text("[1]")
    assert rsc._max_json_object_date(str(p), "built") is None


def test_bundled_entries_resolve_and_honor_per_entry_warn_days():
    got = rsc.gather_archive_freshness(os.path.dirname(__file__))
    for e in rsc.ARCHIVE_MANIFEST:
        if e["name"].startswith("bundled_"):
            assert got[e["name"]], e["name"]
            assert e["format"] == "json_object" and e["warn_days"] > 0
    today = datetime.date(2030, 1, 1)
    # jodi: 40d old is inside its 45d monthly trigger; eia at 40d is past 21d.
    fresh = {"bundled_jodi_stocks": "2029-11-22", "bundled_eia_weekly": "2029-11-22"}
    by = {c["label"]: c["severity"] for c in rsc.run_all_checks([], [], [], today, archive_freshness=fresh)}
    assert by["archive_freshness:bundled_jodi_stocks"] == rsc.OK
    assert by["archive_freshness:bundled_eia_weekly"] == rsc.WARN
