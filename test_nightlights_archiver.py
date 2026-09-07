"""
test_nightlights_archiver.py — pure-function battery for
scripts/nightlights_archiver.py. No network (capture_date takes an
injectable fetch_fn, same convention as test_nasa_gibs_nightlights_gate1.py's
evaluate_date coverage); a separate coherence check reads the real committed
archive once it exists, same "test_committed_artifact_is_coherent"
convention as test_un_comtrade_ingest.py/test_jodi_oil.py.
"""
import importlib.util
import io
import json
import os

import numpy as np
import pytest
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "nightlights_archiver", os.path.join(os.path.dirname(__file__), "scripts", "nightlights_archiver.py"))
archiver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archiver)


def _solid_png(gray_value: int) -> bytes:
    img = Image.new("RGB", (4, 4), (gray_value, gray_value, gray_value))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_fetch(bright_gray: int, dark_gray: int):
    """Returns a fetch_fn(layer, date, y, x) that reads bright/dark purely
    from which coordinate table the (y, x) pair belongs to — independent of
    gate1's real tile coordinates ever changing."""
    bright_coords = set(archiver.gate1.BRIGHT_LOCATIONS.values())

    def _fetch(layer, date, y, x):
        return _solid_png(bright_gray if (y, x) in bright_coords else dark_gray)
    return _fetch


def test_candidate_dates_walks_backward_from_as_of():
    import datetime
    as_of = datetime.datetime(2026, 9, 7, tzinfo=datetime.timezone.utc)
    assert archiver.candidate_dates(as_of, 3) == ["2026-09-06", "2026-09-05", "2026-09-04"]


def test_capture_date_computes_ratio_and_quality_pass():
    capture = archiver.capture_date("2026-09-01", fetch_fn=_fake_fetch(bright_gray=200, dark_gray=50))
    assert capture["date"] == "2026-09-01"
    assert len(capture["bright"]) == len(archiver.gate1.BRIGHT_LOCATIONS)
    assert len(capture["dark"]) == len(archiver.gate1.DARK_LOCATIONS)
    assert capture["ratio"] == pytest.approx(4.0)
    assert capture["quality_pass"] is True


def test_capture_date_quality_fails_below_gate1_bar():
    capture = archiver.capture_date("2026-09-01", fetch_fn=_fake_fetch(bright_gray=60, dark_gray=50))
    assert capture["ratio"] < archiver.gate1.BRIGHT_DARK_RATIO_MIN
    assert capture["quality_pass"] is False


def test_already_captured_requires_every_location():
    series = {name: {"points": [["2026-09-01", 1.0, 1.0, True]]} for name in archiver.gate1.BRIGHT_LOCATIONS}
    # dark locations still missing this date
    assert archiver.already_captured(series, "2026-09-01") is False
    for name in archiver.gate1.DARK_LOCATIONS:
        series[name] = {"points": [["2026-09-01", 1.0, 1.0, True]]}
    assert archiver.already_captured(series, "2026-09-01") is True


def test_merge_capture_adds_one_point_per_location():
    series = {}
    capture = archiver.capture_date("2026-09-01", fetch_fn=_fake_fetch(bright_gray=200, dark_gray=50))
    added = archiver.merge_capture(series, capture)
    n_locations = len(archiver.gate1.BRIGHT_LOCATIONS) + len(archiver.gate1.DARK_LOCATIONS)
    assert added == n_locations
    assert series["vegas"]["kind"] == "bright"
    assert series["ocean_pacific"]["kind"] == "dark"
    assert series["vegas"]["points"] == [["2026-09-01", 200.0, 4.0, True]]


def test_merge_capture_never_overwrites_an_already_archived_date():
    series = {"vegas": {"kind": "bright", "points": [["2026-09-01", 999.0, 999.0, True]]}}
    capture = {
        "date": "2026-09-01",
        "bright": {"vegas": 1.0, "tokyo": 1.0, "london": 1.0},
        "dark": {"ocean_pacific": 1.0, "ocean_atlantic": 1.0, "ocean_indian": 1.0, "ocean_south_pacific": 1.0},
        "ratio": 1.0, "quality_pass": False,
    }
    added = archiver.merge_capture(series, capture)
    assert series["vegas"]["points"] == [["2026-09-01", 999.0, 999.0, True]]
    assert added == len(capture["bright"]) + len(capture["dark"]) - 1


def test_build_artifact_reports_latest_date_and_quality_pass_dates():
    series = {}
    archiver.merge_capture(series, archiver.capture_date("2026-09-01", fetch_fn=_fake_fetch(200, 50)))
    archiver.merge_capture(series, archiver.capture_date("2026-09-02", fetch_fn=_fake_fetch(60, 50)))
    art = archiver.build_artifact(series, "2026-09-07T00:00:00+00:00")
    assert art["latest_date"] == "2026-09-02"
    assert art["quality_pass_dates"] == ["2026-09-01"]
    assert art["layer"] == archiver.gate1.CANDIDATE_LAYER
    assert art["series_count"] == len(series)
    for s in series.values():
        assert s["n"] == len(s["points"]) == 2
        assert s["first"] == "2026-09-01" and s["last"] == "2026-09-02"


def test_build_artifact_empty_series_latest_date_is_none():
    art = archiver.build_artifact({}, "t")
    assert art["latest_date"] is None
    assert art["quality_pass_dates"] == []
    assert art["series_count"] == 0


def test_committed_archive_is_coherent():
    """The repo artifact, once it exists, satisfies its own invariants —
    same discipline as test_un_comtrade_ingest.py's committed-archive check.
    Skips cleanly if no archive has been captured yet in this checkout."""
    fp = os.path.join(os.path.dirname(__file__), "datacore", "nightlights_metro_brightness.json")
    if not os.path.exists(fp):
        pytest.skip("no committed nightlights archive yet")
    art = json.load(open(fp))
    all_locations = set(archiver.gate1.BRIGHT_LOCATIONS) | set(archiver.gate1.DARK_LOCATIONS)
    assert set(art["series"]) == all_locations
    assert art["series_count"] == len(all_locations)
    for name, s in art["series"].items():
        expected_kind = "bright" if name in archiver.gate1.BRIGHT_LOCATIONS else "dark"
        assert s["kind"] == expected_kind
        assert s["n"] == len(s["points"]) > 0
        dates = [p[0] for p in s["points"]]
        assert dates == sorted(dates), f"{name} points not sorted"
        assert len(dates) == len(set(dates)), f"{name} has a duplicate date"
        assert s["first"] == dates[0] and s["last"] == dates[-1]
