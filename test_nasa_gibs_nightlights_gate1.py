"""
test_nasa_gibs_nightlights_gate1.py — pure-function battery for
scripts/nasa_gibs_nightlights_gate1.py (tile_xy/tile_url/mean_brightness/
evaluate_date/evaluate_layer). No network — fetch_tile (the one networked
function) is exercised live only by running the script directly, same
convention as test_un_comtrade_gate1.py leaving fetch_fred_series untested
in pytest. evaluate_date/evaluate_layer take an injectable fetch_fn so
their orchestration logic (ratio math, pass bar, error handling) is
covered without hitting the network.
"""
import importlib.util
import io
import os

import numpy as np
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "nasa_gibs_nightlights_gate1",
    os.path.join(os.path.dirname(__file__), "scripts", "nasa_gibs_nightlights_gate1.py"))
gate1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate1)


def _solid_png(gray_value: int) -> bytes:
    """A tiny synthetic solid-color PNG at the given gray level (0-255)."""
    img = Image.new("RGB", (4, 4), (gray_value, gray_value, gray_value))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_tile_xy_matches_known_las_vegas_tile_at_z6():
    # cross-checked against BRIGHT_LOCATIONS["vegas"] = (25, 11), derived
    # with this exact formula when the location table was built
    x, y = gate1.tile_xy(36.1699, -115.1398, 6)
    assert (x, y) == (11, 25)


def test_tile_xy_matches_known_ocean_tile_at_z6():
    x, y = gate1.tile_xy(0.0, -150.0, 6)
    assert (x, y) == (5, 32)


def test_tile_url_builds_expected_wmts_rest_path():
    url = gate1.tile_url("SOME_LAYER", "2026-01-15", 25, 11)
    assert url == (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "SOME_LAYER/default/2026-01-15/GoogleMapsCompatible_Level8/6/25/11.png"
    )


def test_mean_brightness_reads_solid_color_png_exactly():
    assert gate1.mean_brightness(_solid_png(0)) == 0.0
    assert gate1.mean_brightness(_solid_png(255)) == 255.0
    assert gate1.mean_brightness(_solid_png(100)) == 100.0


def test_evaluate_date_computes_bright_dark_ratio_and_passes_above_bar():
    # bright locations all render 200, dark locations all render 50 ->
    # ratio 4.0, well above BRIGHT_DARK_RATIO_MIN (2.0)
    def fake_fetch(layer, date, y, x):
        is_bright = (y, x) in gate1.BRIGHT_LOCATIONS.values()
        return _solid_png(200 if is_bright else 50)

    r = gate1.evaluate_date("ANY_LAYER", "2026-01-01", fetch_fn=fake_fetch)
    assert r["bright_mean"] == 200.0
    assert r["dark_mean"] == 50.0
    assert r["ratio"] == 4.0
    assert r["pass"] is True


def test_evaluate_date_fails_when_dark_locations_are_not_meaningfully_darker():
    # the SHIPPED_LAYER's real 2026-09-01 finding, reproduced synthetically:
    # bright and dark render nearly the same -> ratio near 1.0, below the bar
    def fake_fetch(layer, date, y, x):
        is_bright = (y, x) in gate1.BRIGHT_LOCATIONS.values()
        return _solid_png(91 if is_bright else 73)

    r = gate1.evaluate_date("ANY_LAYER", "2026-01-01", fetch_fn=fake_fetch)
    assert round(r["ratio"], 2) == 1.25
    assert r["pass"] is False


def test_evaluate_layer_passes_only_when_every_sampled_date_passes():
    def fetch_pass(layer, date, y, x):
        return _solid_png(200 if (y, x) in gate1.BRIGHT_LOCATIONS.values() else 50)

    def fetch_mixed(layer, date, y, x):
        # first date passes, second date fails the ratio bar
        level = 200 if (y, x) in gate1.BRIGHT_LOCATIONS.values() else (50 if date == "2026-01-01" else 190)
        return _solid_png(level)

    always_pass = gate1.evaluate_layer("ANY_LAYER", ["2026-01-01", "2026-01-02"], fetch_fn=fetch_pass)
    assert always_pass["pass"] is True
    assert len(always_pass["results"]) == 2
    assert always_pass["errors"] == []

    one_fails = gate1.evaluate_layer("ANY_LAYER", ["2026-01-01", "2026-01-02"], fetch_fn=fetch_mixed)
    assert one_fails["pass"] is False
    assert len(one_fails["results"]) == 2


def test_evaluate_layer_records_fetch_errors_without_dropping_them_silently():
    def flaky_fetch(layer, date, y, x):
        if date == "2026-02-01":
            raise RuntimeError("simulated 404")
        return _solid_png(200 if (y, x) in gate1.BRIGHT_LOCATIONS.values() else 50)

    v = gate1.evaluate_layer("ANY_LAYER", ["2026-01-01", "2026-02-01"], fetch_fn=flaky_fetch)
    assert len(v["results"]) == 1
    assert len(v["errors"]) == 1
    assert v["errors"][0]["date"] == "2026-02-01"
    assert v["pass"] is True  # the one date that succeeded passed the bar


def test_evaluate_layer_fails_when_every_date_errors():
    def always_fails(layer, date, y, x):
        raise RuntimeError("simulated outage")

    v = gate1.evaluate_layer("ANY_LAYER", ["2026-01-01"], fetch_fn=always_fails)
    assert v["results"] == []
    assert len(v["errors"]) == 1
    assert v["pass"] is False


def test_bright_and_dark_location_tile_coords_are_disjoint():
    bright_coords = set(gate1.BRIGHT_LOCATIONS.values())
    dark_coords = set(gate1.DARK_LOCATIONS.values())
    assert bright_coords.isdisjoint(dark_coords)


def test_shipped_and_candidate_layers_are_different_gibs_products():
    assert gate1.SHIPPED_LAYER != gate1.CANDIDATE_LAYER
    assert gate1.SHIPPED_LAYER == "VIIRS_SNPP_DayNightBand_At_Sensor_Radiance"
    assert gate1.CANDIDATE_LAYER == "VIIRS_SNPP_GapFilled_BRDF_Corrected_DayNightBand_Radiance"
