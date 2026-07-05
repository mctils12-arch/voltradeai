"""
test_tankfill_estimator.py — tank-fill v2 PR-3 battery. Fully synthetic —
rasters are built in-test with KNOWN crescent geometry, so the estimator is
judged against ground truth it has never seen. No network, no chip files.
"""
import importlib.util
import math
import os

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "tankfill_estimator", os.path.join(os.path.dirname(__file__), "scripts", "tankfill_estimator.py"))
est = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(est)

BBOX = [-96.770, 35.922, -96.712, 35.960]
GROUND_DN = 800
ROOF_DN = 3000


SS = 4  # supersampling factor — models the sensor integrating radiance per cell


def synth_chip(w=200, h=160, tanks=(), sun_azim=135.0, shadow_k=est.SHADOW_K):
    """5-band chip: dark ground, bright roof disks, and for each tank a
    painted rim crescent of KNOWN reach s_px on the up-sun side (interior
    points left uncovered when the disk is shifted down-sun by s). Rendered
    at SSx resolution and box-averaged down: a hard-threshold raster cannot
    represent sub-pixel crescents, but the real sensor integrates radiance
    over each 10 m cell — supersampling reproduces that, and the estimator's
    linear-darkening inversion depends on it."""
    hi = np.full((h * SS, w * SS), float(GROUND_DN))
    yy, xx = (np.mgrid[0:h * SS, 0:w * SS] + 0.5) / SS
    az = math.radians(sun_azim)
    ux, uy = math.sin(az), -math.cos(az)  # image-coords unit vector toward sun
    for (cx, cy, r_px, s_px) in tanks:
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 < r_px ** 2
        hi[disk] = ROOF_DN
        if s_px > 0:
            c2x, c2y = cx - s_px * ux, cy - s_px * uy
            crescent = disk & (((xx - c2x) ** 2 + (yy - c2y) ** 2) >= r_px ** 2)
            hi[crescent] = ROOF_DN * shadow_k
    lo = hi.reshape(h, SS, w, SS).mean(axis=(1, 3))
    chip = np.empty((h, w, 5), dtype=np.uint16)
    for b in ("B02", "B03", "B04", "B08"):
        chip[..., est.BAND_IDX[b]] = np.round(lo).astype(np.uint16)
    chip[..., est.BAND_IDX["SCL"]] = 4  # vegetation = valid
    return chip


def px_to_geo(x, y, w=200, h=160):
    lon = BBOX[0] + x / w * (BBOX[2] - BBOX[0])
    lat = BBOX[3] - y / h * (BBOX[3] - BBOX[1])
    return lon, lat


def meta_for(scene="SYNTH_1", date="2026-01-15", sun_elev=30.0, sun_azim=135.0):
    return {"scene": scene, "date": date, "bbox": BBOX,
            "sun_elev": sun_elev, "sun_azim": sun_azim}


def tank_row(x, y, r_px, site="cushing_hub", height=14.6):
    lon, lat = px_to_geo(x, y)
    return {"id": f"t_{x}_{y}", "lon": lon, "lat": lat,
            "diameter_m": r_px * 2 * est.RES_M, "site": site,
            "height_m": height, "height_src": "assumed_api650_48ft"}


def test_geo_to_px_roundtrip():
    x, y = est.geo_to_px(*px_to_geo(120.0, 40.0), BBOX, 200, 160)
    assert abs(x - 120.0) < 1e-9 and abs(y - 40.0) < 1e-9
    # corners
    assert est.geo_to_px(BBOX[0], BBOX[3], BBOX, 200, 160) == (0.0, 0.0)


def test_lens_inversion_forward_roundtrip():
    """Invert the exact forward model the inversion assumes — must round-trip."""
    r_px, elev = 8.0, 30.0
    for depth_true in (2.0, 6.0, 12.0):
        s_px = depth_true / math.tan(math.radians(elev)) / est.RES_M
        f = 4.0 * s_px / (math.pi * r_px)
        ratio = 1.0 - f * (1.0 - est.SHADOW_K)
        assert est.lens_darkening_to_depth(ratio, r_px, elev) == pytest.approx(depth_true, rel=1e-9)
    # up-sun brighter than down-sun can never go negative
    assert est.lens_darkening_to_depth(1.05, r_px, elev) == 0.0


def test_half_weights_geometry():
    (y0, y1, x0, x1), w_up, w_dn = est.half_weights((160, 200), 100.0, 80.0, 10.0, sun_azim_deg=90.0)  # sun due east
    total = w_up + w_dn
    assert total.max() <= 1.0 + 1e-9
    # weights integrate to the disk area
    assert float(total.sum()) == pytest.approx(math.pi * 100.0, rel=0.02)
    # halves are exactly symmetric in mass
    assert float(w_up.sum()) == pytest.approx(float(w_dn.sum()), rel=1e-6)
    # up-sun mass must sit east of center (larger x)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx_up = float((w_up * (xx + 0.5)).sum() / w_up.sum())
    cx_dn = float((w_dn * (xx + 0.5)).sum() / w_dn.sum())
    assert cx_up > 100.0 > cx_dn


def test_estimator_recovers_known_depths_and_ordering():
    """End-to-end on synthetic rasters: two big tanks with different painted
    crescent depths — recovered depths must order correctly and land within
    the small-s approximation's honesty band."""
    elev = 30.0
    t = math.tan(math.radians(elev))
    d_shallow, d_deep = 3.0, 9.0
    s1, s2 = d_shallow / t / est.RES_M, d_deep / t / est.RES_M
    chip = synth_chip(tanks=[(60, 60, 8.0, s1), (140, 90, 8.0, s2)])
    tanks = [tank_row(60, 60, 8.0), tank_row(140, 90, 8.0)]
    reading = est.process_scene(chip, meta_for(sun_elev=elev), tanks)
    assert reading["n_measured"] == 2
    by_id = {r["id"]: r for r in reading["tanks"]}
    dep1 = by_id["t_60_60"]["depth_m"]
    dep2 = by_id["t_140_90"]["depth_m"]
    assert dep2 > dep1, "deeper crescent must read deeper"
    # rasterized crescent + small-s lens approx: allow 40% band, exact center
    assert dep1 == pytest.approx(d_shallow, rel=0.4)
    assert dep2 == pytest.approx(d_deep, rel=0.4)
    assert by_id["t_140_90"]["fill"] < by_id["t_60_60"]["fill"] < 1.0
    assert reading["tanks"][0]["q"] == "ok"  # elev 30 on 14.6m: reach 2.5px


def test_full_tank_reads_full():
    chip = synth_chip(tanks=[(100, 80, 8.0, 0.0)])
    reading = est.process_scene(chip, meta_for(), [tank_row(100, 80, 8.0)])
    assert reading["n_measured"] == 1
    assert reading["tanks"][0]["fill"] == pytest.approx(1.0, abs=0.02)


def test_registration_offset_detected_and_corrected():
    tanks_geom = [(60, 60, 8.0, 2.0), (140, 90, 8.0, 1.0)]
    ref = synth_chip(tanks=tanks_geom)
    shifted = np.roll(np.roll(ref, 2, axis=1), 1, axis=0)  # content moves +2x, +1y
    rows = [tank_row(60, 60, 8.0), tank_row(140, 90, 8.0)]
    reading = est.process_scene(shifted, meta_for(scene="SYNTH_SHIFT"), rows,
                                ref_band=ref[..., est.BAND_IDX["B04"]])
    assert (reading["registration"]["dx"], reading["registration"]["dy"]) == (2, 1)
    assert reading["registration"]["r"] > 0.95
    baseline = est.process_scene(ref, meta_for(), rows)
    got = {r["id"]: r["fill"] for r in reading["tanks"]}
    want = {r["id"]: r["fill"] for r in baseline["tanks"]}
    for k in want:
        assert got[k] == pytest.approx(want[k], abs=0.05), "registration must restore the measurement"


def test_scl_masking_skips_cloudy_tank():
    chip = synth_chip(tanks=[(60, 60, 8.0, 1.0), (140, 90, 8.0, 1.0)])
    chip[50:71, 50:71, est.BAND_IDX["SCL"]] = 9  # high-prob cloud over tank 1
    reading = est.process_scene(chip, meta_for(), [tank_row(60, 60, 8.0), tank_row(140, 90, 8.0)])
    assert reading["n_measured"] == 1
    assert reading["skipped_tanks"]["cloud"] == 1
    assert reading["tanks"][0]["id"] == "t_140_90"


def test_mostly_cloudy_scene_skipped_whole():
    chip = synth_chip(tanks=[(60, 60, 8.0, 1.0)])
    chip[:, :, est.BAND_IDX["SCL"]][:100, :] = 9  # ~62% masked
    reading = est.process_scene(chip, meta_for(), [tank_row(60, 60, 8.0)])
    assert reading["skipped"] == "cloud"
    assert "tanks" not in reading


def test_missing_sun_angles_skipped_never_guessed():
    chip = synth_chip(tanks=[(60, 60, 8.0, 1.0)])
    meta = meta_for()
    meta["sun_elev"] = None
    assert est.process_scene(chip, meta, [tank_row(60, 60, 8.0)])["skipped"] == "no_sun_angles"


def test_site_aggregate_is_d2_weighted():
    chip = synth_chip(tanks=[(60, 60, 10.0, 3.0), (140, 90, 5.0, 0.0)])
    rows = [tank_row(60, 60, 10.0), tank_row(140, 90, 5.0)]
    reading = est.process_scene(chip, meta_for(), rows)
    assert reading["n_measured"] == 2
    manual = (sum(r["fill"] * r["d_m"] ** 2 for r in reading["tanks"])
              / sum(r["d_m"] ** 2 for r in reading["tanks"]))
    assert reading["sites"][0]["fill_d2w"] == pytest.approx(manual, abs=1e-4)
    assert reading["sites"][0]["n_tanks"] == 2


def test_summer_subpixel_flagged():
    """elev 75 deg on a 14.6 m shell: full-depth reach 0.39 px < 0.5 — the
    v1-poisoning artifact must surface as a per-tank quality flag."""
    chip = synth_chip(tanks=[(100, 80, 8.0, 0.2)])
    reading = est.process_scene(chip, meta_for(sun_elev=75.0), [tank_row(100, 80, 8.0)])
    assert reading["n_measured"] == 1
    assert reading["tanks"][0]["q"] == "subpixel"


def test_load_registry_measurable_only():
    tanks = est.load_registry()
    assert len(tanks) == 234
    assert all(t["diameter_m"] >= 40 for t in tanks)
    assert all(t["height_src"] == "assumed_api650_48ft" for t in tanks)
